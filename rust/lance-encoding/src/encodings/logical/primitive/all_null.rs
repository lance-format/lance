// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;
use std::vec;

use arrow_buffer::ScalarBuffer;
use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use lance_core::Result;
use lance_core::cache::{Context, DeepSizeOf};
use lance_core::error::Error;

use crate::EncodingsIo;
use crate::buffer::LanceBuffer;
use crate::compression::{BlockDecompressor, DecompressionStrategy, create_rle_decompressor};
use crate::data::{AllNullDataBlock, DataBlock};
use crate::decoder::{DecodePageTask, DecodedPage, StructuralPageDecoder};
use crate::encodings::physical::rle::{RleDecompressor, RleRuns};
use crate::format::pb21::CompressiveEncoding;
use crate::format::pb21::compressive_encoding::Compression;
use crate::repdef::{DefinitionInterpretation, RepDefUnraveler, max_visible_level};

use super::{
    CachedPageData, NoCachedPageData, PageInitialization, PageInitializationBuffers, PageLoadTask,
    StructuralPageScheduler,
};

/// How a complex-all-null page's rep/def level buffer is compressed on disk.
/// Captured at scheduler construction so `initialize` can keep RLE levels in run
/// form instead of expanding them.
#[derive(Debug, Clone)]
pub enum LevelCodec {
    /// Raw little-endian u16 levels (no block compression).
    Uncompressed,
    /// RLE-compressed levels; the validated physical runs select their cached representation.
    Rle(Arc<RleDecompressor>),
    /// Any other block compression; decoded eagerly into [`LazyLevels::Dense`]
    /// (these encodings don't expand, so laziness buys nothing).
    Block(Arc<dyn BlockDecompressor>),
}

impl LevelCodec {
    pub(super) fn try_new(
        encoding: Option<&CompressiveEncoding>,
        decompression_strategy: &dyn DecompressionStrategy,
    ) -> Result<Self> {
        match encoding {
            None => Ok(Self::Uncompressed),
            Some(encoding) => match encoding.compression.as_ref() {
                Some(Compression::Rle(rle)) => Ok(Self::Rle(Arc::new(create_rle_decompressor(
                    rle,
                    decompression_strategy,
                )?))),
                _ => Ok(Self::Block(Arc::from(
                    decompression_strategy.create_block_decompressor(encoding)?,
                ))),
            },
        }
    }
}

#[derive(Debug)]
enum RunEnds {
    U16(Box<[u16]>),
    U32(Box<[u32]>),
    U64(Box<[u64]>),
}

impl RunEnds {
    fn width_for(num_values: usize) -> usize {
        if u16::try_from(num_values).is_ok() {
            std::mem::size_of::<u16>()
        } else if u32::try_from(num_values).is_ok() {
            std::mem::size_of::<u32>()
        } else {
            std::mem::size_of::<u64>()
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::U16(ends) => ends.len(),
            Self::U32(ends) => ends.len(),
            Self::U64(ends) => ends.len(),
        }
    }

    fn get(&self, run: usize) -> usize {
        match self {
            Self::U16(ends) => ends[run] as usize,
            Self::U32(ends) => ends[run] as usize,
            Self::U64(ends) => ends[run] as usize,
        }
    }

    fn partition_point(&self, logical_index: usize) -> usize {
        match self {
            Self::U16(ends) => ends.partition_point(|&end| end as usize <= logical_index),
            Self::U32(ends) => ends.partition_point(|&end| end as usize <= logical_index),
            Self::U64(ends) => ends.partition_point(|&end| end as usize <= logical_index),
        }
    }

    fn deep_size(&self) -> usize {
        match self {
            Self::U16(ends) => std::mem::size_of_val(ends.as_ref()),
            Self::U32(ends) => std::mem::size_of_val(ends.as_ref()),
            Self::U64(ends) => std::mem::size_of_val(ends.as_ref()),
        }
    }
}

enum RunEndsBuilder {
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl RunEndsBuilder {
    fn with_capacity(num_values: usize, capacity: usize) -> Self {
        if u16::try_from(num_values).is_ok() {
            Self::U16(Vec::with_capacity(capacity))
        } else if u32::try_from(num_values).is_ok() {
            Self::U32(Vec::with_capacity(capacity))
        } else {
            Self::U64(Vec::with_capacity(capacity))
        }
    }

    fn push(&mut self, end: usize) -> Result<()> {
        match self {
            Self::U16(ends) => ends.push(
                u16::try_from(end)
                    .map_err(|_| Error::internal(format!("Run end {end} does not fit in u16")))?,
            ),
            Self::U32(ends) => ends.push(
                u32::try_from(end)
                    .map_err(|_| Error::internal(format!("Run end {end} does not fit in u32")))?,
            ),
            Self::U64(ends) => ends.push(end as u64),
        }
        Ok(())
    }

    fn set_last(&mut self, end: usize) -> Result<()> {
        match self {
            Self::U16(ends) => {
                let last = ends.last_mut().ok_or_else(|| {
                    Error::internal("Cannot extend an empty coalesced run buffer")
                })?;
                *last = u16::try_from(end)
                    .map_err(|_| Error::internal(format!("Run end {end} does not fit in u16")))?;
            }
            Self::U32(ends) => {
                let last = ends.last_mut().ok_or_else(|| {
                    Error::internal("Cannot extend an empty coalesced run buffer")
                })?;
                *last = u32::try_from(end)
                    .map_err(|_| Error::internal(format!("Run end {end} does not fit in u32")))?;
            }
            Self::U64(ends) => {
                let last = ends.last_mut().ok_or_else(|| {
                    Error::internal("Cannot extend an empty coalesced run buffer")
                })?;
                *last = end as u64;
            }
        }
        Ok(())
    }

    fn finish(self) -> RunEnds {
        match self {
            Self::U16(ends) => RunEnds::U16(ends.into_boxed_slice()),
            Self::U32(ends) => RunEnds::U32(ends.into_boxed_slice()),
            Self::U64(ends) => RunEnds::U64(ends.into_boxed_slice()),
        }
    }
}

#[derive(Debug)]
enum RunStorage {
    Physical(RleRuns),
    Coalesced { values: Box<[u16]>, ends: RunEnds },
}

impl RunStorage {
    fn len(&self) -> usize {
        match self {
            Self::Physical(runs) => runs.num_values(),
            Self::Coalesced { ends, .. } => ends.get(ends.len() - 1),
        }
    }

    fn num_runs(&self) -> usize {
        match self {
            Self::Physical(runs) => runs.num_runs(),
            Self::Coalesced { values, .. } => values.len(),
        }
    }

    fn value(&self, run: usize) -> u16 {
        match self {
            Self::Physical(runs) => runs.value(run),
            Self::Coalesced { values, .. } => values[run],
        }
    }

    fn first_value_above(&self, max: u16) -> Option<(usize, u16)> {
        (0..self.num_runs()).find_map(|run| {
            let value = self.value(run);
            (value > max).then_some((run, value))
        })
    }

    fn seek(&self, position: &mut RunPosition, logical_index: usize) {
        if logical_index >= self.len() {
            *position = RunPosition {
                run: self.num_runs(),
                start: self.len(),
                end: self.len(),
            };
            return;
        }

        match self {
            Self::Physical(runs) => {
                if position.run >= runs.num_runs()
                    || position.end == 0
                    || logical_index < position.start
                {
                    *position = RunPosition {
                        run: 0,
                        start: 0,
                        end: runs.length(0),
                    };
                }
                while position.end <= logical_index {
                    self.advance(position);
                }
            }
            Self::Coalesced { ends, .. } => {
                if logical_index < position.start || logical_index >= position.end {
                    let run = ends.partition_point(logical_index);
                    *position = RunPosition {
                        run,
                        start: if run == 0 { 0 } else { ends.get(run - 1) },
                        end: ends.get(run),
                    };
                }
            }
        }
    }

    fn advance(&self, position: &mut RunPosition) {
        let next_run = position.run + 1;
        if next_run >= self.num_runs() {
            *position = RunPosition {
                run: self.num_runs(),
                start: self.len(),
                end: self.len(),
            };
            return;
        }

        let start = position.end;
        position.run = next_run;
        position.start = start;
        position.end = match self {
            Self::Physical(runs) => start + runs.length(next_run),
            Self::Coalesced { ends, .. } => ends.get(next_run),
        };
    }

    fn deep_size(&self) -> usize {
        match self {
            Self::Physical(runs) => runs.deep_size(),
            Self::Coalesced { values, ends } => {
                std::mem::size_of_val(values.as_ref()) + ends.deep_size()
            }
        }
    }
}

/// Rep/def levels for a complex-all-null page.
///
/// RLE pages retain the smallest of their validated physical runs, coalesced
/// runs, and dense values. The decoder materializes only the per-drain slices
/// it touches.
#[derive(Debug, Clone)]
enum LazyLevels {
    Dense(ScalarBuffer<u16>),
    Runs(Arc<RunStorage>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LevelPlan {
    Physical,
    Coalesced,
    Dense,
}

#[derive(Debug, Default, Clone, Copy)]
struct RunPosition {
    run: usize,
    start: usize,
    end: usize,
}

/// Monotonic forward cursor into a [`LazyLevels`] sequence.
///
/// Drains seek to strictly increasing rows, so each [`LazyLevels::seek_row_start`]
/// resumes from the last position instead of rescanning — every run is visited at
/// most once per page while locating and counting level ranges.
#[derive(Debug, Default, Clone, Copy)]
struct LevelCursor {
    /// Logical level index where the current row begins.
    level: usize,
    /// Row index at `level` (the number of `max_rep` occurrences before it).
    row: u64,
    /// Run containing `level`. Unused for [`LazyLevels::Dense`].
    run: RunPosition,
}

impl LazyLevels {
    fn from_rle_runs(runs: RleRuns) -> Result<Self> {
        let plan = Self::select_plan(&runs);
        match plan {
            LevelPlan::Physical => Ok(Self::Runs(Arc::new(RunStorage::Physical(
                runs.into_owned(),
            )))),
            LevelPlan::Coalesced => Self::build_coalesced(runs),
            LevelPlan::Dense => Self::build_dense(runs),
        }
    }

    /// Minimize retained payload bytes first, then expected traversal work.
    /// If both are equal, keep the physical runs and avoid another allocation.
    fn select_plan(runs: &RleRuns) -> LevelPlan {
        if runs.num_values() == 0 {
            return LevelPlan::Dense;
        }

        let run_storage_size = std::mem::size_of::<RunStorage>() as u128;
        let physical_size = run_storage_size + runs.owned_size() as u128;
        let coalesced_size = run_storage_size
            + (runs.coalesced_runs() as u128)
                * (std::mem::size_of::<u16>() + RunEnds::width_for(runs.num_values())) as u128;
        let dense_size = (runs.num_values() as u128) * std::mem::size_of::<u16>() as u128;
        [
            (physical_size, runs.num_runs(), 0usize, LevelPlan::Physical),
            (
                coalesced_size,
                runs.coalesced_runs(),
                1usize,
                LevelPlan::Coalesced,
            ),
            (dense_size, runs.num_values(), 2usize, LevelPlan::Dense),
        ]
        .into_iter()
        .min_by_key(|(size, traversal, priority, _)| (*size, *traversal, *priority))
        .map(|(_, _, _, plan)| plan)
        .unwrap_or(LevelPlan::Dense)
    }

    fn build_coalesced(runs: RleRuns) -> Result<Self> {
        let mut values = Vec::with_capacity(runs.coalesced_runs());
        let mut ends = RunEndsBuilder::with_capacity(runs.num_values(), runs.coalesced_runs());
        let mut logical_end = 0usize;
        for (value, length) in runs.iter() {
            logical_end = logical_end
                .checked_add(length)
                .ok_or_else(|| Error::internal("Validated RLE run length sum overflowed usize"))?;
            if values.last().copied() == Some(value) {
                ends.set_last(logical_end)?;
            } else {
                values.push(value);
                ends.push(logical_end)?;
            }
        }
        Ok(Self::Runs(Arc::new(RunStorage::Coalesced {
            values: values.into_boxed_slice(),
            ends: ends.finish(),
        })))
    }

    fn build_dense(runs: RleRuns) -> Result<Self> {
        let mut values = Vec::new();
        values.try_reserve_exact(runs.num_values()).map_err(|_| {
            Error::internal(format!(
                "Cannot allocate {} dense repetition/definition levels",
                runs.num_values()
            ))
        })?;
        for (value, length) in runs.iter() {
            values.resize(values.len() + length, value);
        }
        Ok(Self::Dense(ScalarBuffer::from(values)))
    }

    fn len(&self) -> usize {
        match self {
            Self::Dense(buf) => buf.len(),
            Self::Runs(runs) => runs.len(),
        }
    }

    fn validate_max(&self, level_type: &str, max: u16) -> Result<()> {
        let invalid = match self {
            Self::Dense(levels) => levels
                .iter()
                .enumerate()
                .find_map(|(index, &value)| (value > max).then_some(("index", index, value))),
            Self::Runs(runs) => runs
                .first_value_above(max)
                .map(|(run, value)| ("run", run, value)),
        };
        if let Some((position_type, position, value)) = invalid {
            return Err(Error::invalid_input_source(
                format!(
                    "Invalid {level_type} level {value} at {position_type} {position}: maximum is {max}"
                )
                .into(),
            ));
        }
        Ok(())
    }

    /// Advance `cursor` to the start of row `target_row`, returning that row's
    /// starting level index.
    ///
    /// Rows begin at `max_rep` positions, so this finds the `target_row`-th one.
    /// `target_row` must be `>= cursor.row`: the cursor only moves forward, which
    /// is what keeps a full page decode O(runs) rather than O(rows).
    fn seek_row_start(
        &self,
        cursor: &mut LevelCursor,
        target_row: u64,
        max_rep: u16,
    ) -> Result<usize> {
        let mut need = target_row.checked_sub(cursor.row).ok_or_else(|| {
            Error::internal(format!(
                "Complex all-null row ranges are not sorted: target row {target_row} follows {}",
                cursor.row
            ))
        })?;
        if need == 0 {
            return Ok(cursor.level);
        }
        match self {
            Self::Dense(buf) => {
                let mut level = cursor.level;
                while need > 0 {
                    if level >= buf.len() {
                        return Err(Error::internal(
                            "Invalid complex all-null layout: repetition buffer too short",
                        ));
                    }
                    if buf[level] != max_rep {
                        return Err(Error::internal(
                            "Invalid complex all-null layout: row did not start at max repetition level",
                        ));
                    }
                    level += 1;
                    while level < buf.len() && buf[level] != max_rep {
                        level += 1;
                    }
                    need -= 1;
                }
                cursor.level = level;
                cursor.row = target_row;
                Ok(level)
            }
            Self::Runs(runs) => {
                let mut level = cursor.level;
                let mut run = cursor.run;
                runs.seek(&mut run, level);
                while need > 0 {
                    if run.run >= runs.num_runs() {
                        return Err(Error::internal(
                            "Invalid complex all-null layout: repetition buffer too short",
                        ));
                    }
                    if runs.value(run.run) != max_rep {
                        return Err(Error::internal(
                            "Invalid complex all-null layout: row did not start at max repetition level",
                        ));
                    }
                    let avail = (run.end - level) as u64;
                    if need < avail {
                        // Target lands inside this max-rep run.
                        level += need as usize;
                        need = 0;
                    } else {
                        // Consume every row start in this run, then skip the
                        // trailing non-max-rep runs to reach the next row start.
                        need -= avail;
                        runs.advance(&mut run);
                        while run.run < runs.num_runs() && runs.value(run.run) != max_rep {
                            runs.advance(&mut run);
                        }
                        level = if run.run < runs.num_runs() {
                            run.start
                        } else {
                            self.len()
                        };
                    }
                }
                cursor.level = level;
                cursor.row = target_row;
                cursor.run = run;
                Ok(level)
            }
        }
    }

    /// Count of levels in `range` that are `<= max`, resuming from `*run_cursor`
    /// and leaving it on the last run that overlaps `range`.
    ///
    /// Successive calls must pass ascending, non-overlapping ranges (`range.start
    /// >=` the previous `range.end`) so runs are swept at most once per page.
    fn count_le_cursor(
        &self,
        run_cursor: &mut RunPosition,
        range: Range<usize>,
        max: u16,
    ) -> (u64, RunPosition) {
        if range.is_empty() {
            return (0, *run_cursor);
        }
        match self {
            Self::Dense(buf) => (
                buf[range].iter().filter(|&&d| d <= max).count() as u64,
                RunPosition::default(),
            ),
            Self::Runs(runs) => {
                // Advance to the first run overlapping the range.
                runs.seek(run_cursor, range.start);
                let start = *run_cursor;
                let mut count = 0u64;
                let mut current = *run_cursor;
                while current.run < runs.num_runs() && current.start < range.end {
                    if runs.value(current.run) <= max {
                        let lo = current.start.max(range.start);
                        let hi = current.end.min(range.end);
                        count += (hi - lo) as u64;
                    }
                    if current.end >= range.end {
                        break;
                    }
                    runs.advance(&mut current);
                }
                // Resume the next (ascending) range from the last overlapping run;
                // `current` remains valid because `range` is non-empty.
                *run_cursor = current;
                (count, start)
            }
        }
    }

    fn extend_into(&self, range: Range<usize>, run: RunPosition, out: &mut Vec<u16>) {
        if range.is_empty() {
            return;
        }
        match self {
            Self::Dense(buf) => out.extend_from_slice(&buf[range]),
            Self::Runs(runs) => {
                let mut current = run;
                runs.seek(&mut current, range.start);
                while current.run < runs.num_runs() && current.start < range.end {
                    let lo = current.start.max(range.start);
                    let hi = current.end.min(range.end);
                    if hi > lo {
                        out.resize(out.len() + (hi - lo), runs.value(current.run));
                    }
                    runs.advance(&mut current);
                }
            }
        }
    }

    #[cfg(test)]
    fn deep_size(&self) -> usize {
        self.deep_size_of_children(&mut Context::new())
    }
}

impl DeepSizeOf for LazyLevels {
    fn deep_size_of_children(&self, ctx: &mut Context) -> usize {
        match self {
            Self::Dense(buf) => buf.deep_size_of_children(ctx),
            Self::Runs(runs) => {
                let pointer = Arc::as_ptr(runs) as *const () as usize;
                if ctx.mark_seen(pointer) {
                    std::mem::size_of_val(runs.as_ref()) + runs.deep_size()
                } else {
                    0
                }
            }
        }
    }
}

fn validate_complex_all_null_levels(
    rep: &Option<LazyLevels>,
    def: &Option<LazyLevels>,
    max_rep: u16,
    max_def: u16,
) -> Result<()> {
    if let Some(rep) = rep {
        rep.validate_max("repetition", max_rep)?;
    }
    if let Some(def) = def {
        def.validate_max("definition", max_def)?;
    }
    if let (Some(rep), Some(def)) = (rep, def)
        && rep.len() != def.len()
    {
        return Err(Error::invalid_input_source(
            format!(
                "Mismatched complex all-null level counts: repetition has {}, definition has {}",
                rep.len(),
                def.len()
            )
            .into(),
        ));
    }
    Ok(())
}

fn expected_level_bytes(num_values: u64, level_type: &str) -> Result<usize> {
    usize::try_from(num_values)
        .ok()
        .and_then(|num_values| num_values.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| {
            Error::invalid_input_source(
                format!("{level_type} level count {num_values} does not fit in memory").into(),
            )
        })
}

fn dense_levels_from_block(
    decompressed: DataBlock,
    num_values: u64,
    level_type: &str,
) -> Result<LazyLevels> {
    let DataBlock::FixedWidth(block) = decompressed else {
        return Err(Error::invalid_input_source(
            format!("Expected fixed-width data block for {level_type} levels").into(),
        ));
    };
    if block.num_values != num_values {
        return Err(Error::invalid_input_source(
            format!(
                "Unexpected {level_type} level count after decompression: expected {num_values}, got {}",
                block.num_values
            )
            .into(),
        ));
    }
    if block.bits_per_value != 16 {
        return Err(Error::invalid_input_source(
            format!(
                "Unexpected {level_type} level bit width after decompression: expected 16, got {}",
                block.bits_per_value
            )
            .into(),
        ));
    }
    let expected_bytes = expected_level_bytes(num_values, level_type)?;
    if block.data.len() != expected_bytes {
        return Err(Error::invalid_input_source(
            format!(
                "Unexpected decompressed {level_type} level size: expected {expected_bytes} bytes for {num_values} values, got {}",
                block.data.len()
            )
            .into(),
        ));
    }
    Ok(LazyLevels::Dense(block.data.borrow_to_typed_slice::<u16>()))
}

#[derive(Debug)]
struct CachedComplexAllNullState {
    rep: Option<LazyLevels>,
    def: Option<LazyLevels>,
}

impl DeepSizeOf for CachedComplexAllNullState {
    fn deep_size_of_children(&self, ctx: &mut Context) -> usize {
        self.rep.deep_size_of_children(ctx) + self.def.deep_size_of_children(ctx)
    }
}

impl CachedPageData for CachedComplexAllNullState {
    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }
}

/// A scheduler for all-null data that has repetition and definition levels
///
/// We still need to do some I/O in this case because we need to figure out what kind of null we
/// are dealing with (null list, null struct, what level null struct, etc.)
///
/// TODO: Right now we just load the entire rep/def at initialization time and cache it.  This is a touch
/// RAM aggressive and maybe we want something more lazy in the future.  On the other hand, it's simple
/// and fast so...maybe not :)
#[derive(Debug)]
pub struct ComplexAllNullScheduler {
    // Set from protobuf
    buffer_offsets_and_sizes: Arc<[(u64, u64)]>,
    def_meaning: Arc<[DefinitionInterpretation]>,
    repdef: Option<Arc<CachedComplexAllNullState>>,
    max_rep: u16,
    max_def: u16,
    max_visible_level: u16,
    rep_codec: LevelCodec,
    def_codec: LevelCodec,
    num_rep_values: u64,
    num_def_values: u64,
}

impl ComplexAllNullScheduler {
    pub(crate) fn new(
        buffer_offsets_and_sizes: Arc<[(u64, u64)]>,
        def_meaning: Arc<[DefinitionInterpretation]>,
        rep_codec: LevelCodec,
        def_codec: LevelCodec,
        num_rep_values: u64,
        num_def_values: u64,
    ) -> Self {
        let max_rep = def_meaning.iter().filter(|l| l.is_list()).count() as u16;
        let max_def = def_meaning
            .iter()
            .map(|meaning| meaning.num_def_levels())
            .sum::<u16>();
        let max_visible_level = max_visible_level(&def_meaning);
        Self {
            buffer_offsets_and_sizes,
            def_meaning,
            repdef: None,
            max_rep,
            max_def,
            max_visible_level,
            rep_codec,
            def_codec,
            num_rep_values,
            num_def_values,
        }
    }
}

impl StructuralPageScheduler for ComplexAllNullScheduler {
    fn init_layout(&self) -> Result<PageInitialization> {
        // Fully load the rep & def buffers, as needed
        let (rep_pos, rep_size) = self.buffer_offsets_and_sizes[0];
        let (def_pos, def_size) = self.buffer_offsets_and_sizes[1];

        Ok(PageInitialization::new([
            (rep_size > 0).then_some(rep_pos..rep_pos + rep_size),
            (def_size > 0).then_some(def_pos..def_pos + def_size),
        ]))
    }

    fn init_from_buffers<'a>(
        &'a mut self,
        buffers: PageInitializationBuffers,
        _io: &Arc<dyn EncodingsIo>,
    ) -> BoxFuture<'a, Result<Arc<dyn CachedPageData>>> {
        let rep_codec = self.rep_codec.clone();
        let def_codec = self.def_codec.clone();
        let num_rep_values = self.num_rep_values;
        let num_def_values = self.num_def_values;
        let max_rep = self.max_rep;
        let max_def = self.max_def;

        async move {
            let [rep_bytes, def_bytes] = buffers.try_into_array("complex all-null")?;

            // RLE levels select the smallest validated cache representation;
            // everything else expands eagerly to `LazyLevels::Dense`.
            let build_levels = |compressed_bytes: Bytes,
                                codec: &LevelCodec,
                                num_values: u64,
                                level_type: &str|
             -> Result<LazyLevels> {
                match codec {
                    LevelCodec::Uncompressed => {
                        if num_values == 0 {
                            if !compressed_bytes
                                .len()
                                .is_multiple_of(std::mem::size_of::<u16>())
                            {
                                return Err(Error::invalid_input_source(
                                    format!(
                                        "Unexpected uncompressed {level_type} level size: {} bytes is not divisible by {}",
                                        compressed_bytes.len(),
                                        std::mem::size_of::<u16>()
                                    )
                                    .into(),
                                ));
                            }
                        } else {
                            let expected_bytes = expected_level_bytes(num_values, level_type)?;
                            if compressed_bytes.len() != expected_bytes {
                                return Err(Error::invalid_input_source(
                                    format!(
                                        "Unexpected uncompressed {level_type} level size: expected {expected_bytes} bytes for {num_values} values, got {}",
                                        compressed_bytes.len()
                                    )
                                    .into(),
                                ));
                            }
                        }
                        let buffer = LanceBuffer::from_bytes(compressed_bytes, 2);
                        Ok(LazyLevels::Dense(buffer.borrow_to_typed_slice::<u16>()))
                    }
                    LevelCodec::Rle(decompressor) => {
                        let frame = LanceBuffer::from_bytes(compressed_bytes, 1);
                        let runs = decompressor.decode_u16_runs(frame, num_values)?;
                        LazyLevels::from_rle_runs(runs)
                    }
                    LevelCodec::Block(decompressor) => {
                        let frame = LanceBuffer::from_bytes(compressed_bytes, 1);
                        let decompressed = decompressor.decompress(Some(frame), num_values)?;
                        dense_levels_from_block(decompressed, num_values, level_type)
                    }
                }
            };

            let rep = if let Some(rep) = rep_bytes {
                Some(build_levels(rep, &rep_codec, num_rep_values, "repetition")?)
            } else {
                None
            };

            let def = if let Some(def) = def_bytes {
                Some(build_levels(def, &def_codec, num_def_values, "definition")?)
            } else {
                None
            };

            validate_complex_all_null_levels(&rep, &def, max_rep, max_def)?;
            let repdef = Arc::new(CachedComplexAllNullState { rep, def });

            Ok(repdef as Arc<dyn CachedPageData>)
        }
        .boxed()
    }

    fn try_load(&mut self, data: &Arc<dyn CachedPageData>) -> Result<()> {
        self.repdef = Some(data.clone().as_arc_any().downcast().map_err(|_| {
            Error::invalid_input_source(
                "Cached complex all-null page data has an unexpected type".into(),
            )
        })?);
        Ok(())
    }

    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        _io: &Arc<dyn EncodingsIo>,
    ) -> Result<Vec<PageLoadTask>> {
        let ranges = VecDeque::from_iter(ranges.iter().cloned());
        let num_rows = ranges.iter().map(|r| r.end - r.start).sum::<u64>();
        let decoder = Box::new(ComplexAllNullPageDecoder {
            ranges,
            rep: self.repdef.as_ref().unwrap().rep.clone(),
            def: self.repdef.as_ref().unwrap().def.clone(),
            num_rows,
            def_meaning: self.def_meaning.clone(),
            max_rep: self.max_rep,
            max_visible_level: self.max_visible_level,
            rep_cursor: LevelCursor::default(),
            def_run_cursor: RunPosition::default(),
        }) as Box<dyn StructuralPageDecoder>;
        let page_load_task = PageLoadTask {
            decoder_fut: std::future::ready(Ok(decoder)).boxed(),
            num_rows,
        };
        Ok(vec![page_load_task])
    }
}

#[derive(Debug)]
pub struct ComplexAllNullPageDecoder {
    ranges: VecDeque<Range<u64>>,
    rep: Option<LazyLevels>,
    def: Option<LazyLevels>,
    num_rows: u64,
    def_meaning: Arc<[DefinitionInterpretation]>,
    max_rep: u16,
    max_visible_level: u16,
    /// Monotonic cursor into `rep` tracking the current row's level start.
    rep_cursor: LevelCursor,
    /// Monotonic run cursor into `def` for `count_le_cursor`.
    def_run_cursor: RunPosition,
}

impl ComplexAllNullPageDecoder {
    fn drain_ranges(&mut self, num_rows: u64) -> Vec<Range<u64>> {
        let mut rows_desired = num_rows;
        let mut ranges = Vec::with_capacity(self.ranges.len());
        while rows_desired > 0 {
            let front = self.ranges.front_mut().unwrap();
            let avail = front.end - front.start;
            if avail > rows_desired {
                ranges.push(front.start..front.start + rows_desired);
                front.start += rows_desired;
                rows_desired = 0;
            } else {
                ranges.push(self.ranges.pop_front().unwrap());
                rows_desired -= avail;
            }
        }
        ranges
    }

    /// Level index at which row `target_row` starts, advancing the monotonic
    /// repetition cursor. Callers must request non-decreasing `target_row`.
    fn seek_row_start(&mut self, target_row: u64) -> Result<usize> {
        match &self.rep {
            Some(rep) => rep.seek_row_start(&mut self.rep_cursor, target_row, self.max_rep),
            None => {
                // Without repetition every level is its own row.
                self.rep_cursor.row = target_row;
                self.rep_cursor.level = target_row as usize;
                Ok(target_row as usize)
            }
        }
    }

    /// Number of visible items in the level range `levels` (definition levels
    /// `<= max_visible_level`), advancing the monotonic definition cursor.
    fn count_visible(&mut self, levels: Range<usize>) -> Result<(u64, RunPosition)> {
        match &self.def {
            Some(def) => {
                if levels.end > def.len() {
                    return Err(Error::internal(
                        "Invalid complex all-null layout: definition buffer too short",
                    ));
                }
                Ok(def.count_le_cursor(&mut self.def_run_cursor, levels, self.max_visible_level))
            }
            None => Ok(((levels.end - levels.start) as u64, RunPosition::default())),
        }
    }
}

impl StructuralPageDecoder for ComplexAllNullPageDecoder {
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        let drained_ranges = self.drain_ranges(num_rows);
        let mut level_slices: Vec<LevelSlice> = Vec::with_capacity(drained_ranges.len());
        let mut visible_items_total = 0;

        // Each row range is one contiguous level slice `[start_row_level,
        // end_row_level)`, so we seek both boundaries and count its visibility at
        // once rather than per row. The cursors only move forward, so locating and
        // counting all requested ranges visits each intervening run at most once.
        for range in drained_ranges {
            let level_start = self.seek_row_start(range.start)?;
            let rep_run = self.rep_cursor.run;
            let level_end = self.seek_row_start(range.end)?;
            let (visible_items, def_run) = self.count_visible(level_start..level_end)?;
            visible_items_total += visible_items;
            if let Some(last) = level_slices.last_mut()
                && last.range.end == level_start
            {
                last.range.end = level_end;
            } else {
                level_slices.push(LevelSlice {
                    range: level_start..level_end,
                    rep_run,
                    def_run,
                });
            }
        }

        Ok(Box::new(DecodeComplexAllNullTask {
            level_slices,
            visible_items_total,
            rep: self.rep.clone(),
            def: self.def.clone(),
            def_meaning: self.def_meaning.clone(),
            max_visible_level: self.max_visible_level,
        }))
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

/// We use `level_slices` to slice into `rep` and `def` and create rep/def buffers
/// for the null data.
#[derive(Debug, Clone)]
struct LevelSlice {
    range: Range<usize>,
    rep_run: RunPosition,
    def_run: RunPosition,
}

#[derive(Clone, Copy)]
enum LevelKind {
    Repetition,
    Definition,
}

impl LevelSlice {
    fn run(&self, kind: LevelKind) -> RunPosition {
        match kind {
            LevelKind::Repetition => self.rep_run,
            LevelKind::Definition => self.def_run,
        }
    }
}

#[derive(Debug)]
pub struct DecodeComplexAllNullTask {
    level_slices: Vec<LevelSlice>,
    visible_items_total: u64,
    rep: Option<LazyLevels>,
    def: Option<LazyLevels>,
    def_meaning: Arc<[DefinitionInterpretation]>,
    max_visible_level: u16,
}

impl DecodeComplexAllNullTask {
    fn decode_level(&self, levels: &Option<LazyLevels>, kind: LevelKind) -> Option<Vec<u16>> {
        levels.as_ref().map(|levels| {
            let num_levels = self
                .level_slices
                .iter()
                .map(|slice| slice.range.end - slice.range.start)
                .sum();
            let mut referenced_levels = Vec::with_capacity(num_levels);
            for slice in &self.level_slices {
                levels.extend_into(slice.range.clone(), slice.run(kind), &mut referenced_levels);
            }
            referenced_levels
        })
    }
}

impl DecodePageTask for DecodeComplexAllNullTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        let rep = self.decode_level(&self.rep, LevelKind::Repetition);
        let def = self.decode_level(&self.def, LevelKind::Definition);

        // If there are definition levels there may be empty / null lists which are not visible
        // in the items array.  We need to account for that here to figure out how many values
        // should be in the items array.
        let num_values = if let Some(def) = &def {
            def.iter().filter(|&d| *d <= self.max_visible_level).count() as u64
        } else {
            self.visible_items_total
        };

        let data = DataBlock::AllNull(AllNullDataBlock { num_values });
        let unraveler = RepDefUnraveler::new(rep, def, self.def_meaning, num_values);
        Ok(DecodedPage {
            data,
            repdef: unraveler,
        })
    }
}

/// A scheduler for simple all-null data
///
/// "simple" all-null data is data that is all null and only has a single level of definition and
/// no repetition.  We don't need to read any data at all in this case.
#[derive(Debug, Default)]
pub struct SimpleAllNullScheduler {}

impl StructuralPageScheduler for SimpleAllNullScheduler {
    fn needs_initialization(&self) -> bool {
        false
    }

    fn init_layout(&self) -> Result<PageInitialization> {
        Ok(PageInitialization::new([]))
    }

    fn init_from_buffers<'a>(
        &'a mut self,
        _buffers: PageInitializationBuffers,
        _io: &Arc<dyn EncodingsIo>,
    ) -> BoxFuture<'a, Result<Arc<dyn CachedPageData>>> {
        std::future::ready(Ok(Arc::new(NoCachedPageData) as Arc<dyn CachedPageData>)).boxed()
    }

    fn try_load(&mut self, _cache: &Arc<dyn CachedPageData>) -> Result<()> {
        Ok(())
    }

    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        _io: &Arc<dyn EncodingsIo>,
    ) -> Result<Vec<PageLoadTask>> {
        let num_rows = ranges.iter().map(|r| r.end - r.start).sum::<u64>();
        let decoder =
            Box::new(SimpleAllNullPageDecoder { num_rows }) as Box<dyn StructuralPageDecoder>;
        let page_load_task = PageLoadTask {
            decoder_fut: std::future::ready(Ok(decoder)).boxed(),
            num_rows,
        };
        Ok(vec![page_load_task])
    }
}

/// A page decode task for all-null data without any
/// repetition and only a single level of definition
#[derive(Debug)]
struct SimpleAllNullDecodePageTask {
    num_values: u64,
}
impl DecodePageTask for SimpleAllNullDecodePageTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        let unraveler = RepDefUnraveler::new(
            None,
            Some(vec![1; self.num_values as usize]),
            Arc::new([DefinitionInterpretation::NullableItem]),
            self.num_values,
        );
        Ok(DecodedPage {
            data: DataBlock::AllNull(AllNullDataBlock {
                num_values: self.num_values,
            }),
            repdef: unraveler,
        })
    }
}

#[derive(Debug)]
pub struct SimpleAllNullPageDecoder {
    num_rows: u64,
}

impl StructuralPageDecoder for SimpleAllNullPageDecoder {
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        Ok(Box::new(SimpleAllNullDecodePageTask {
            num_values: num_rows,
        }))
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_buffer::ScalarBuffer;

    use crate::buffer::LanceBuffer;
    use crate::compression::DefaultDecompressionStrategy;
    use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
    use crate::encodings::logical::primitive::tests::encoded_u16_frame;
    use crate::encodings::physical::rle::{RleDecompressor, RleRuns, RunLengthWidth};
    use crate::format::pb21::compressive_encoding::Compression;
    use crate::format::{ProtobufUtils21, pb21};
    use crate::testing::{TestCases, check_round_trip_encoding_of_data};

    use super::{
        LazyLevels, LevelCodec, LevelCursor, LevelPlan, RunEndsBuilder, RunPosition, RunStorage,
        dense_levels_from_block, validate_complex_all_null_levels,
    };

    fn encoded_u16_runs(levels: &[u16], run_length_width: RunLengthWidth) -> RleRuns {
        let frame = encoded_u16_frame(levels, run_length_width);
        RleDecompressor::with_run_length_width(16, run_length_width)
            .decode_u16_runs(frame, levels.len() as u64)
            .unwrap()
    }

    fn physical_levels(levels: &[u16]) -> LazyLevels {
        LazyLevels::Runs(Arc::new(RunStorage::Physical(
            encoded_u16_runs(levels, RunLengthWidth::U8).into_owned(),
        )))
    }

    fn coalesced_levels(levels: &[u16]) -> LazyLevels {
        let mut values = Vec::new();
        let mut ends = RunEndsBuilder::with_capacity(levels.len(), levels.len());
        for (index, &value) in levels.iter().enumerate() {
            if values.last() == Some(&value) {
                ends.set_last(index + 1).unwrap();
            } else {
                values.push(value);
                ends.push(index + 1).unwrap();
            }
        }
        LazyLevels::Runs(Arc::new(RunStorage::Coalesced {
            values: values.into_boxed_slice(),
            ends: ends.finish(),
        }))
    }

    #[test]
    fn lazy_levels_runs_match_dense() {
        // Runs: 3x2, 1x1, 3x3, 0x2  =>  [3,3,1,3,3,3,0,0]
        let expanded: Vec<u16> = vec![3, 3, 1, 3, 3, 3, 0, 0];
        let coalesced = coalesced_levels(&expanded);
        let physical = physical_levels(&expanded);
        let dense = LazyLevels::Dense(ScalarBuffer::<u16>::from(expanded.clone()));
        let n = expanded.len();

        assert_eq!(coalesced.len(), n);
        assert_eq!(physical.len(), n);
        assert_eq!(dense.len(), n);

        // Rows begin at each `max_rep` (3) position; row `num_rows` maps to `len`.
        let max_rep = 3u16;
        let row_starts: Vec<usize> = (0..n).filter(|&i| expanded[i] == max_rep).collect();
        for target in 0..=row_starts.len() as u64 {
            let want = row_starts.get(target as usize).copied().unwrap_or(n);
            for runs in [&coalesced, &physical] {
                let mut cursor = LevelCursor::default();
                assert_eq!(
                    runs.seek_row_start(&mut cursor, target, max_rep).unwrap(),
                    want,
                    "seek_row_start({target})"
                );
            }
            let mut c_dense = LevelCursor::default();
            assert_eq!(
                dense.seek_row_start(&mut c_dense, target, max_rep).unwrap(),
                want
            );
        }

        // `count_le_cursor` (fresh cursor per range) and `extend_into` agree with
        // the dense reference on every sub-range.
        for start in 0..=n {
            for end in start..=n {
                for max in [0u16, 1, 2, 3] {
                    let want = expanded[start..end].iter().filter(|&&d| d <= max).count() as u64;
                    for runs in [&coalesced, &physical] {
                        let mut cursor = RunPosition::default();
                        assert_eq!(
                            runs.count_le_cursor(&mut cursor, start..end, max).0,
                            want,
                            "count_le_cursor({start}..{end}, {max})"
                        );
                    }
                    let mut d_cur = RunPosition::default();
                    assert_eq!(dense.count_le_cursor(&mut d_cur, start..end, max).0, want);
                }
                for runs in [&coalesced, &physical] {
                    let mut got = Vec::new();
                    runs.extend_into(start..end, RunPosition::default(), &mut got);
                    assert_eq!(
                        got,
                        expanded[start..end].to_vec(),
                        "extend_into({start}..{end})"
                    );
                }
                let mut got_dense = Vec::new();
                dense.extend_into(start..end, RunPosition::default(), &mut got_dense);
                assert_eq!(got_dense, expanded[start..end].to_vec());
            }
        }
    }

    #[test]
    fn physical_run_hints_support_deferred_materialization() {
        let expanded: Vec<u16> = vec![3, 3, 1, 1, 2, 2, 0, 0];
        let physical = physical_levels(&expanded);
        let LazyLevels::Runs(runs) = &physical else {
            panic!("expected physical runs");
        };
        let mut first_hint = RunPosition::default();
        runs.seek(&mut first_hint, 2);
        let mut second_hint = RunPosition::default();
        runs.seek(&mut second_hint, 6);

        let mut second = Vec::new();
        physical.extend_into(6..8, second_hint, &mut second);
        let mut first = Vec::new();
        physical.extend_into(2..4, first_hint, &mut first);
        assert_eq!(second, expanded[6..8]);
        assert_eq!(first, expanded[2..4]);
    }

    /// Fuzz parity for the run-oriented complex-all-null drain: the cursor walk
    /// over `LazyLevels` must yield the exact level slices and visible
    /// count that a brute-force reference over the fully expanded levels does, for
    /// dense, physical-run, and coalesced-run forms and arbitrarily shaped range requests.
    mod complex_all_null_drain_parity {
        use std::ops::Range;

        use arrow_buffer::ScalarBuffer;
        use proptest::prelude::*;

        use super::super::{LazyLevels, LevelCursor, RunPosition};
        use super::{coalesced_levels, physical_levels};
        use crate::Result;

        #[derive(Debug, Clone)]
        struct DrainInput {
            max_rep: u16,
            max_visible: u16,
            rep: Option<Vec<u16>>,
            def: Option<Vec<u16>>,
            ranges: Vec<Range<u64>>,
        }

        fn dense_levels(levels: &[u16]) -> LazyLevels {
            LazyLevels::Dense(ScalarBuffer::from(levels.to_vec()))
        }

        fn rle_levels(levels: &[u16]) -> LazyLevels {
            coalesced_levels(levels)
        }

        fn seek(
            rep: Option<&LazyLevels>,
            cursor: &mut LevelCursor,
            row: u64,
            max_rep: u16,
        ) -> Result<usize> {
            match rep {
                Some(rep) => rep.seek_row_start(cursor, row, max_rep),
                None => {
                    cursor.row = row;
                    cursor.level = row as usize;
                    Ok(row as usize)
                }
            }
        }

        /// Mirror of `ComplexAllNullPageDecoder::drain`, driving the real
        /// `seek_row_start` / `count_le_cursor` with monotonic cursors.
        fn simulate_drain(
            rep: Option<&LazyLevels>,
            def: Option<&LazyLevels>,
            max_rep: u16,
            max_visible: u16,
            ranges: &[Range<u64>],
        ) -> Result<(Vec<Range<usize>>, u64)> {
            let mut rep_cursor = LevelCursor::default();
            let mut def_run_cursor = RunPosition::default();
            let mut slices: Vec<Range<usize>> = Vec::new();
            let mut visible = 0u64;
            for range in ranges {
                let level_start = seek(rep, &mut rep_cursor, range.start, max_rep)?;
                let level_end = seek(rep, &mut rep_cursor, range.end, max_rep)?;
                visible += match def {
                    Some(def) => {
                        def.count_le_cursor(
                            &mut def_run_cursor,
                            level_start..level_end,
                            max_visible,
                        )
                        .0
                    }
                    None => (level_end - level_start) as u64,
                };
                match slices.last_mut() {
                    Some(last) if last.end == level_start => last.end = level_end,
                    _ => slices.push(level_start..level_end),
                }
            }
            Ok((slices, visible))
        }

        /// Independent brute-force reference over fully expanded levels.
        fn reference_drain(
            rep: Option<&[u16]>,
            def: Option<&[u16]>,
            max_rep: u16,
            max_visible: u16,
            ranges: &[Range<u64>],
        ) -> (Vec<Range<usize>>, u64) {
            let total_levels = rep
                .map(|r| r.len())
                .or_else(|| def.map(|d| d.len()))
                .unwrap_or(0);
            // Level index where each row starts (or `total_levels` for the end row).
            let row_starts: Vec<usize> = match rep {
                Some(rep) => (0..rep.len()).filter(|&i| rep[i] == max_rep).collect(),
                None => (0..total_levels).collect(),
            };
            let level_of_row = |row: u64| {
                row_starts
                    .get(row as usize)
                    .copied()
                    .unwrap_or(total_levels)
            };

            let mut slices: Vec<Range<usize>> = Vec::new();
            let mut visible = 0u64;
            for range in ranges {
                let ls = level_of_row(range.start);
                let le = level_of_row(range.end);
                visible += match def {
                    Some(def) => def[ls..le].iter().filter(|&&d| d <= max_visible).count() as u64,
                    None => (le - ls) as u64,
                };
                match slices.last_mut() {
                    Some(last) if last.end == ls => last.end = le,
                    _ => slices.push(ls..le),
                }
            }
            (slices, visible)
        }

        fn ranges_strategy(num_rows: u64) -> BoxedStrategy<Vec<Range<u64>>> {
            if num_rows == 0 {
                return Just(Vec::new()).boxed();
            }
            // (gap, len) pairs; a zero gap yields ranges adjacent in row space,
            // which exercises the level-slice coalescing path.
            proptest::collection::vec((0u64..=3, 1u64..=4), 0..=8)
                .prop_map(move |pairs| {
                    let mut ranges = Vec::new();
                    let mut pos = 0u64;
                    for (gap, len) in pairs {
                        pos = pos.saturating_add(gap);
                        if pos >= num_rows {
                            break;
                        }
                        let end = (pos + len).min(num_rows);
                        ranges.push(pos..end);
                        pos = end;
                    }
                    ranges
                })
                .boxed()
        }

        fn drain_input() -> impl Strategy<Value = DrainInput> {
            (
                1u16..=3,
                0u16..=3,
                any::<bool>(),
                any::<bool>(),
                1usize..=48,
            )
                .prop_flat_map(|(max_rep, max_visible, has_rep, has_def, len)| {
                    // Complex-all-null always has definition levels when there is
                    // no repetition, so force `def` present in that case.
                    let has_def = has_def || !has_rep;
                    let rep = if has_rep {
                        proptest::collection::vec(0u16..=max_rep, len)
                            .prop_map(move |mut v| {
                                // Row 0 must start at a max-rep boundary.
                                v[0] = max_rep;
                                Some(v)
                            })
                            .boxed()
                    } else {
                        Just(None).boxed()
                    };
                    let def = if has_def {
                        proptest::collection::vec(0u16..=(max_visible + 2), len)
                            .prop_map(Some)
                            .boxed()
                    } else {
                        Just(None).boxed()
                    };
                    (Just(max_rep), Just(max_visible), rep, def)
                })
                .prop_flat_map(|(max_rep, max_visible, rep, def)| {
                    let num_rows = match &rep {
                        Some(rep) => rep.iter().filter(|&&v| v == max_rep).count() as u64,
                        None => def.as_ref().map(|d| d.len() as u64).unwrap_or(0),
                    };
                    ranges_strategy(num_rows).prop_map(move |ranges| DrainInput {
                        max_rep,
                        max_visible,
                        rep: rep.clone(),
                        def: def.clone(),
                        ranges,
                    })
                })
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(512))]

            #[test]
            fn drain_matches_reference(input in drain_input()) {
                let DrainInput { max_rep, max_visible, rep, def, ranges } = input;

                let reference =
                    reference_drain(rep.as_deref(), def.as_deref(), max_rep, max_visible, &ranges);

                let rep_dense = rep.as_deref().map(dense_levels);
                let def_dense = def.as_deref().map(dense_levels);
                let got_dense =
                    simulate_drain(rep_dense.as_ref(), def_dense.as_ref(), max_rep, max_visible, &ranges)
                        .unwrap();
                prop_assert_eq!(&got_dense, &reference, "dense form diverged from reference");

                let rep_rle = rep.as_deref().map(rle_levels);
                let def_rle = def.as_deref().map(rle_levels);
                let got_rle =
                    simulate_drain(rep_rle.as_ref(), def_rle.as_ref(), max_rep, max_visible, &ranges)
                        .unwrap();
                prop_assert_eq!(&got_rle, &reference, "rle form diverged from reference");

                let rep_physical = rep.as_deref().map(physical_levels);
                let def_physical = def.as_deref().map(physical_levels);
                let got_physical =
                    simulate_drain(rep_physical.as_ref(), def_physical.as_ref(), max_rep, max_visible, &ranges)
                        .unwrap();
                prop_assert_eq!(&got_physical, &reference, "physical form diverged from reference");
            }
        }
    }

    #[test]
    fn lazy_levels_runs_are_compact() {
        let single_run = |n: usize| {
            let mut ends = RunEndsBuilder::with_capacity(n, 1);
            ends.push(n).unwrap();
            LazyLevels::Runs(Arc::new(RunStorage::Coalesced {
                values: vec![1u16].into_boxed_slice(),
                ends: ends.finish(),
            }))
        };
        // Run-form footprint is independent of the logical length within an end width...
        assert_eq!(single_run(100).deep_size(), single_run(10_000).deep_size());
        assert!(single_run(10_000_000).deep_size() < 100);
        assert_eq!(single_run(10_000_000).len(), 10_000_000);
        // ...while Dense pays 2 bytes per value.
        assert_eq!(
            LazyLevels::Dense(ScalarBuffer::<u16>::from(vec![1u16; 1000])).deep_size(),
            2000
        );
    }

    #[test]
    fn lazy_levels_selects_smallest_representation() {
        let runs = encoded_u16_runs(&[7u16; 10], RunLengthWidth::U8);
        assert_eq!(LazyLevels::select_plan(&runs), LevelPlan::Dense);

        let equal_size: Vec<u16> = std::iter::repeat_n(0, 256)
            .chain(std::iter::repeat_n(1, 100))
            .chain(std::iter::repeat_n(2, 100))
            .collect();
        let runs = encoded_u16_runs(&equal_size, RunLengthWidth::U8);
        assert_eq!(LazyLevels::select_plan(&runs), LevelPlan::Coalesced);

        let moderate_runs: Vec<u16> = (0..250)
            .flat_map(|run| std::iter::repeat_n((run % 2) as u16, 4))
            .collect();
        let runs = encoded_u16_runs(&moderate_runs, RunLengthWidth::U8);
        assert_eq!(LazyLevels::select_plan(&runs), LevelPlan::Physical);

        let split_constant = vec![7u16; 5000];
        let runs = encoded_u16_runs(&split_constant, RunLengthWidth::U8);
        assert_eq!(LazyLevels::select_plan(&runs), LevelPlan::Coalesced);

        let high_density: Vec<u16> = (0..70_000).map(|index| (index % 2) as u16).collect();
        let runs = encoded_u16_runs(&high_density, RunLengthWidth::U32);
        assert_eq!(LazyLevels::select_plan(&runs), LevelPlan::Dense);
    }

    #[test]
    fn physical_runs_detach_from_large_encoded_frame() {
        let levels: Vec<u16> = (0..250)
            .flat_map(|run| std::iter::repeat_n((run % 2) as u16, 4))
            .collect();
        let frame = encoded_u16_frame(&levels, RunLengthWidth::U8);
        let frame_offset = 4096;
        let mut allocation = vec![0; frame_offset + frame.len() + 1_000_000];
        allocation[frame_offset..frame_offset + frame.len()].copy_from_slice(frame.as_ref());
        let frame = LanceBuffer::from(allocation).slice_with_length(frame_offset, frame.len());
        let runs = RleDecompressor::with_run_length_width(16, RunLengthWidth::U8)
            .decode_u16_runs(frame, levels.len() as u64)
            .unwrap();

        assert_eq!(LazyLevels::select_plan(&runs), LevelPlan::Physical);
        let cached = LazyLevels::from_rle_runs(runs).unwrap();
        assert!(
            matches!(cached, LazyLevels::Runs(ref runs) if matches!(runs.as_ref(), RunStorage::Physical(_)))
        );
        assert_eq!(cached.len(), levels.len());
        assert!(cached.deep_size() < 4096);
    }

    #[test]
    fn complex_all_null_levels_reject_invalid_values_and_lengths() {
        let invalid_levels = vec![0u16, 3];
        for levels in [
            LazyLevels::Dense(ScalarBuffer::from(invalid_levels.clone())),
            physical_levels(&invalid_levels),
            coalesced_levels(&invalid_levels),
        ] {
            let error = validate_complex_all_null_levels(&None, &Some(levels), 0, 2).unwrap_err();
            assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
            assert!(error.to_string().contains("Invalid definition level 3"));
        }

        let rep = Some(LazyLevels::Dense(ScalarBuffer::from(vec![0u16; 2])));
        let def = Some(LazyLevels::Dense(ScalarBuffer::from(vec![0u16])));
        let error = validate_complex_all_null_levels(&rep, &def, 0, 0).unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("repetition has 2, definition has 1")
        );
    }

    #[test]
    fn block_levels_reject_malformed_payload_size() {
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(vec![0]),
            bits_per_value: 16,
            num_values: 1,
            block_info: BlockInfo::new(),
        });
        let error = dense_levels_from_block(block, 1, "definition").unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("expected 2 bytes for 1 values, got 1")
        );
    }

    #[test]
    fn complex_all_null_level_codec_validates_rle_metadata() {
        let encoding = pb21::CompressiveEncoding {
            compression: Some(Compression::Rle(Box::new(pb21::Rle {
                values: None,
                run_lengths: Some(Box::new(ProtobufUtils21::flat(8, None))),
            }))),
        };

        let error = LevelCodec::try_new(Some(&encoding), &DefaultDecompressionStrategy::default())
            .unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("RLE compression missing values encoding")
        );
    }

    // https://github.com/lance-format/lance/issues/6681
    #[tokio::test]
    async fn test_sparse_boolean_list_roundtrip() {
        use arrow_array::builder::{BooleanBuilder, ListBuilder};

        let mut list_builder = ListBuilder::new(BooleanBuilder::new());
        for i in 0..1000i32 {
            if i % 64 == 0 {
                // Alternate true/false so the array is not constant (constant path avoids the bug).
                list_builder.values().append_value(i % 128 == 0);
                list_builder.append(true);
            } else {
                list_builder.append(false);
            }
        }
        let list_array = Arc::new(list_builder.finish());

        let test_cases = TestCases::default().with_structural_encodings();
        check_round_trip_encoding_of_data(vec![list_array], &test_cases, HashMap::new()).await;
    }
}
