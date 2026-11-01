// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared Zone Training Utilities
//!
//! This module provides common infrastructure for building zone-based scalar indexes.
//! It handles chunking data streams into fixed-size zones while respecting fragment
//! boundaries and computing zone bounds that remain valid after row deletions.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, UInt64Array};
use datafusion_execution::SendableRecordBatchStream;
use futures::{TryStreamExt, stream};
use lance_core::error::Error;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::{ROW_ADDR, Result};
use lance_select::RowAddrTreeMap;

/// Minimum number of scanned rows to dispatch to the CPU pool at once.
///
/// This is a floor, not a target: zones are never divided, so a single zone
/// larger than this is dispatched as-is. Cheap zones are grouped to amortize
/// dispatch overhead.
const MIN_ROWS_PER_SPLIT: usize = 8192;

//
// Example: Suppose we have two fragments, each with 4 rows.
// Fragment 0: start = 0, length = 4  // covers rows 0, 1, 2, 3 in fragment 0
// The row addresses for fragment 0 are: 0, 1, 2, 3
// Fragment 1: start = 0, length = 4  // covers rows 0, 1, 2, 3 in fragment 1
// The row addresses for fragment 1 are: (1<<32), (1<<32)+1, (1<<32)+2, (1<<32)+3
//
// Deletion is 0 index based. We delete the 0th and 1st row in fragment 0,
// and the 1st and 2nd row in fragment 1,
// Fragment 0: start = 2, length = 2 // covers rows 2, 3 in fragment 0
// The row addresses for fragment 0 are: 2, 3
// Fragment 1: start = 0, length = 4  // covers rows 0, 3 in fragment 1
// The row addresses for fragment 1 are: (1<<32), (1<<32)+3
/// Zone bound within a fragment
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneBound {
    pub fragment_id: u64,
    // start is start row of the zone in the fragment, also known
    // as the local offset. To get the actual first row address,
    // use `(fragment_id << 32) | start`.
    pub start: u64,
    // length is the span of row offsets between the first and last row in the zone,
    // calculated as (last_row_offset - first_row_offset + 1). It is not the count
    // of physical rows, since deletions may create gaps within the span.
    pub length: usize,
}

/// Index-specific logic used while building zones.
pub trait ZoneProcessor {
    type ZoneStatistics;

    /// Process a slice of values that belongs to the current zone.
    fn process_chunk(&mut self, values: &ArrayRef) -> Result<()>;

    /// Emit statistics when the zone is full or the fragment changes.
    fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics>;

    /// Reset state so the processor can handle the next zone.
    fn reset(&mut self) -> Result<()>;
}

/// Trainer that handles chunking, fragment boundaries, and zone flushing.
#[derive(Debug)]
pub struct ZoneTrainer<P> {
    processor: P,
    zone_capacity: u64,
    /// Whether to record the address of every null row. Off for indexes that
    /// keep only per-zone counts, so the scan does not build a bitmap it will
    /// discard.
    track_null_rows: bool,
    /// Physical-offset alignment: when set, zone `k` of a fragment is exactly
    /// the offsets `[k * capacity, (k + 1) * capacity)`, and the map gives the
    /// fragment's physical row count so the last zone's length is known even
    /// when its trailing rows are deleted or outside the scanned selection.
    aligned: Option<HashMap<u64, u64>>,
}

impl<P> ZoneTrainer<P>
where
    P: ZoneProcessor,
{
    /// Create a new trainer that buffers at most `zone_capacity` rows per zone.
    pub fn new(processor: P, zone_capacity: u64) -> Result<Self> {
        validate_zone_capacity(zone_capacity)?;
        Ok(Self {
            processor,
            zone_capacity,
            track_null_rows: true,
            aligned: None,
        })
    }

    /// Cut zones at fixed physical offsets instead of counting rows:
    /// zone `k` of a fragment spans `[k * capacity, min((k + 1) * capacity, P))`
    /// where `P` is that fragment's entry in `physical_rows`. A fragment
    /// missing from the map falls back to the span up to its last scanned
    /// row for its final zone. Rows of a zone with no scanned rows produce
    /// no zone. Fragments must still arrive grouped and in offset order.
    pub fn aligned_to(mut self, physical_rows: HashMap<u64, u64>) -> Self {
        self.aligned = Some(physical_rows);
        self
    }

    /// Enable or disable per-row null address tracking. When disabled the
    /// bitmap returned by [`Self::train`] is empty.
    pub fn with_null_tracking(mut self, track: bool) -> Self {
        self.track_null_rows = track;
        self
    }

    /// Consume the `_rowaddr`-annotated stream, split it into zones, and let the
    /// processor compute zone statistics.
    ///
    /// The caller must provide record batches where the first column is the
    /// value array that the zone processor understands, and the schema includes
    /// the `_rowaddr` column with physical row addresses. Future zone-based
    /// indexes should maintain this ordering or extend the trainer to accept an
    /// explicit column index.
    pub async fn train(
        mut self,
        stream: SendableRecordBatchStream,
    ) -> Result<(Vec<P::ZoneStatistics>, RowAddrTreeMap)> {
        let zone_size = usize::try_from(self.zone_capacity).map_err(|_| {
            Error::invalid_input("zone capacity does not fit into usize on this platform")
        })?;
        let mut assembler =
            ZoneSplitAssembler::new(stream, zone_size, self.zone_capacity, self.aligned);
        let mut zones = Vec::new();
        let mut null_rows = RowAddrTreeMap::new();

        self.processor.reset()?;
        while let Some(zone) = assembler.next_zone().await? {
            zones.push(process_zone(
                &mut self.processor,
                zone,
                self.track_null_rows,
                &mut null_rows,
            )?);
            self.processor.reset()?;
        }

        Ok((zones, null_rows))
    }
}

/// Parallel trainer that creates independent processors from a factory.
///
/// Use this entry when zone statistics can be computed independently. The
/// factory and statistics cross the CPU-pool boundary; the processor itself is
/// constructed and consumed on the worker thread.
pub struct ParallelZoneTrainer<P> {
    processor_factory: Arc<dyn Fn() -> Result<P> + Send + Sync>,
    zone_capacity: u64,
    track_null_rows: bool,
    aligned: Option<HashMap<u64, u64>>,
}

impl<P> fmt::Debug for ParallelZoneTrainer<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParallelZoneTrainer")
            .field("zone_capacity", &self.zone_capacity)
            .field("track_null_rows", &self.track_null_rows)
            .field("aligned", &self.aligned)
            .finish_non_exhaustive()
    }
}

impl<P> ParallelZoneTrainer<P>
where
    P: ZoneProcessor + 'static,
    P::ZoneStatistics: Send + 'static,
{
    /// Create a parallel trainer from a pure-CPU processor factory.
    ///
    /// The factory is invoked once per zone. It must not perform I/O or wait.
    pub fn new(
        processor_factory: impl Fn() -> Result<P> + Send + Sync + 'static,
        zone_capacity: u64,
    ) -> Result<Self> {
        validate_zone_capacity(zone_capacity)?;
        processor_factory()?;
        Ok(Self {
            processor_factory: Arc::new(processor_factory),
            zone_capacity,
            track_null_rows: true,
            aligned: None,
        })
    }

    /// Cut zones at fixed physical offsets instead of counting rows.
    ///
    /// See [`ZoneTrainer::aligned_to`] for the alignment contract.
    pub fn aligned_to(mut self, physical_rows: HashMap<u64, u64>) -> Self {
        self.aligned = Some(physical_rows);
        self
    }

    /// Enable or disable per-row null address tracking.
    pub fn with_null_tracking(mut self, track: bool) -> Self {
        self.track_null_rows = track;
        self
    }

    /// Consume the stream and compute zone statistics on the CPU pool.
    pub async fn train(
        self,
        stream: SendableRecordBatchStream,
    ) -> Result<(Vec<P::ZoneStatistics>, RowAddrTreeMap)> {
        let zone_size = usize::try_from(self.zone_capacity).map_err(|_| {
            Error::invalid_input("zone capacity does not fit into usize on this platform")
        })?;
        let splits = stream::try_unfold(
            ZoneSplitAssembler::new(stream, zone_size, self.zone_capacity, self.aligned),
            |mut assembler| async move {
                Ok(assembler
                    .next_split()
                    .await?
                    .map(|split| (split, assembler)))
            },
        );

        let processor_factory = self.processor_factory;
        let track_null_rows = self.track_null_rows;
        let split_tasks = splits.map_ok(move |split| {
            let processor_factory = processor_factory.clone();
            spawn_cpu(move || {
                process_zone_split(processor_factory.as_ref(), split, track_null_rows)
            })
        });
        let results = split_tasks.try_buffered(get_num_compute_intensive_cpus());
        futures::pin_mut!(results);

        let mut zones = Vec::new();
        let mut null_rows = RowAddrTreeMap::new();
        while let Some((mut split_zones, split_null_rows)) = results.try_next().await? {
            zones.append(&mut split_zones);
            null_rows |= &split_null_rows;
        }
        Ok((zones, null_rows))
    }
}

type ZoneSplit = Vec<Zone>;

#[derive(Debug)]
struct Zone {
    batches: Vec<RecordBatch>,
    bound: ZoneBound,
    scanned_rows: usize,
}

/// Serial accumulator shared by the sequential and parallel trainers.
///
/// This is the only implementation of fragment/aligned boundary handling and
/// zone-bound computation.
struct ZoneSplitAssembler {
    stream: SendableRecordBatchStream,
    zone_size: usize,
    capacity: u64,
    aligned: Option<HashMap<u64, u64>>,
    input_batch: Option<RecordBatch>,
    input_offset: usize,
    current_batches: Vec<RecordBatch>,
    current_rows: usize,
    current_fragment: Option<u64>,
    current_zone_index: Option<u64>,
    current_start_offset: Option<u64>,
    current_end_offset: Option<u64>,
    current_split: ZoneSplit,
    current_split_rows: usize,
}

impl ZoneSplitAssembler {
    fn new(
        stream: SendableRecordBatchStream,
        zone_size: usize,
        capacity: u64,
        aligned: Option<HashMap<u64, u64>>,
    ) -> Self {
        Self {
            stream,
            zone_size,
            capacity,
            aligned,
            input_batch: None,
            input_offset: 0,
            current_batches: Vec::new(),
            current_rows: 0,
            current_fragment: None,
            current_zone_index: None,
            current_start_offset: None,
            current_end_offset: None,
            current_split: Vec::new(),
            current_split_rows: 0,
        }
    }

    async fn next_zone(&mut self) -> Result<Option<Zone>> {
        loop {
            if self.input_batch.is_none() {
                let Some(batch) = self.stream.try_next().await? else {
                    if self.current_batches.is_empty() {
                        return Ok(None);
                    }
                    return self.take_zone().map(Some);
                };
                if batch.num_rows() == 0 {
                    continue;
                }
                row_addr_column(&batch)?;
                self.input_batch = Some(batch);
                self.input_offset = 0;
            }

            let input_rows = self
                .input_batch
                .as_ref()
                .map(RecordBatch::num_rows)
                .ok_or_else(|| Error::internal("zone split assembler lost its input batch"))?;
            if self.input_offset == input_rows {
                self.input_batch = None;
                self.input_offset = 0;
                continue;
            }

            let has_alignment = self.aligned.is_some();
            let capacity = self.capacity;
            let batch = self
                .input_batch
                .as_ref()
                .ok_or_else(|| Error::internal("zone split assembler lost its input batch"))?;
            let row_addrs = row_addr_column(batch)?.values();
            let row_addr = row_addrs[self.input_offset];
            let fragment_id = row_addr >> 32;
            let zone_index = RowAddress::new_from_u64(row_addr).row_offset() as u64 / capacity;

            let at_boundary = self.current_rows > 0
                && (self.current_fragment != Some(fragment_id)
                    || (has_alignment && self.current_zone_index != Some(zone_index)));
            if at_boundary {
                return self.take_zone().map(Some);
            }

            self.current_fragment = Some(fragment_id);
            self.current_zone_index = Some(zone_index);

            let remaining_rows = batch.num_rows() - self.input_offset;
            let room = self
                .zone_size
                .checked_sub(self.current_rows)
                .filter(|room| *room > 0)
                .ok_or_else(|| Error::internal("zone split assembler has a full current zone"))?;
            let limit = self.input_offset + remaining_rows.min(room);
            let run_len = row_addrs[self.input_offset..limit]
                .iter()
                .take_while(|addr| {
                    (addr >> 32) == fragment_id
                        && (!has_alignment
                            || RowAddress::new_from_u64(**addr).row_offset() as u64 / capacity
                                == zone_index)
                })
                .count();
            let end = self.input_offset + run_len;
            let first_offset =
                RowAddress::new_from_u64(row_addrs[self.input_offset]).row_offset() as u64;
            let last_offset = RowAddress::new_from_u64(row_addrs[end - 1]).row_offset() as u64;
            self.current_batches
                .push(batch.slice(self.input_offset, run_len));
            self.current_rows += run_len;
            self.input_offset = end;
            if self.current_start_offset.is_none() {
                self.current_start_offset = Some(first_offset);
            }
            self.current_end_offset = Some(last_offset);

            if self.current_rows == self.zone_size {
                return self.take_zone().map(Some);
            }
        }
    }

    async fn next_split(&mut self) -> Result<Option<ZoneSplit>> {
        while self.current_split_rows < MIN_ROWS_PER_SPLIT {
            let Some(zone) = self.next_zone().await? else {
                break;
            };
            self.current_split_rows = self
                .current_split_rows
                .checked_add(zone.scanned_rows)
                .ok_or_else(|| {
                    Error::invalid_input("zone split row count exceeds usize on this platform")
                })?;
            self.current_split.push(zone);
        }

        if self.current_split.is_empty() {
            Ok(None)
        } else {
            self.current_split_rows = 0;
            Ok(Some(std::mem::take(&mut self.current_split)))
        }
    }

    fn take_zone(&mut self) -> Result<Zone> {
        let fragment_id = self
            .current_fragment
            .ok_or_else(|| Error::internal("zone split assembler lost its fragment"))?;
        let start = self.current_start_offset.ok_or_else(|| {
            Error::internal("zone split assembler lost its first observed offset")
        })?;
        let inferred_end = self
            .current_end_offset
            .ok_or_else(|| Error::internal("zone split assembler lost its last observed offset"))?;
        if inferred_end < start {
            return Err(Error::invalid_input("zone row offsets are out of order"));
        }

        let bound = match &self.aligned {
            Some(physical_rows) => {
                let zone_index = self.current_zone_index.ok_or_else(|| {
                    Error::internal("zone split assembler lost its aligned zone index")
                })?;
                let zone_start = zone_index.checked_mul(self.capacity).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "fragment {fragment_id}: zone {zone_index} of {} rows overflows",
                        self.capacity
                    ))
                })?;
                let end = match physical_rows.get(&fragment_id) {
                    Some(rows) => {
                        // The supplied row count is authoritative for the
                        // zone's length, so every observed offset must lie
                        // inside it (and inside this zone); otherwise the
                        // zone would summarize rows it does not cover.
                        if start < zone_start || inferred_end >= *rows {
                            return Err(Error::invalid_input(format!(
                                "fragment {fragment_id}: zone {zone_index} (rows {zone_start}..) \
                                 saw offsets {start}..={inferred_end}, outside its physical row \
                                 count {rows}"
                            )));
                        }
                        zone_start
                            .checked_add(self.capacity)
                            .map_or(*rows, |zone_end| zone_end.min(*rows))
                    }
                    None => inferred_end + 1,
                };
                let length = end
                    .checked_sub(zone_start)
                    .filter(|length| *length > 0)
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "fragment {fragment_id}: zone {zone_index} ends at {end} before its \
                             start {zone_start}"
                        ))
                    })?;
                ZoneBound {
                    fragment_id,
                    start: zone_start,
                    length: length as usize,
                }
            }
            None => ZoneBound {
                fragment_id,
                start,
                length: (inferred_end - start + 1) as usize,
            },
        };

        let scanned_rows = self.current_rows;
        self.current_rows = 0;
        self.current_fragment = None;
        self.current_zone_index = None;
        self.current_start_offset = None;
        self.current_end_offset = None;
        Ok(Zone {
            batches: std::mem::take(&mut self.current_batches),
            bound,
            scanned_rows,
        })
    }
}

fn validate_zone_capacity(zone_capacity: u64) -> Result<()> {
    if zone_capacity == 0 {
        Err(Error::invalid_input(
            "zone capacity must be greater than zero",
        ))
    } else {
        Ok(())
    }
}

fn row_addr_column(batch: &RecordBatch) -> Result<&UInt64Array> {
    batch
        .column_by_name(ROW_ADDR)
        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "zone training batches must include a UInt64 `{ROW_ADDR}` column"
            ))
        })
}

fn process_zone<P: ZoneProcessor>(
    processor: &mut P,
    zone: Zone,
    track_null_rows: bool,
    null_rows: &mut RowAddrTreeMap,
) -> Result<P::ZoneStatistics> {
    for batch in &zone.batches {
        let values = batch.column(0);
        processor.process_chunk(values)?;

        if track_null_rows
            && values.null_count() > 0
            && let Some(nulls) = values.nulls()
        {
            let row_addrs = row_addr_column(batch)?;
            null_rows.extend(
                nulls
                    .iter()
                    .enumerate()
                    .filter(|(_, is_valid)| !is_valid)
                    .map(|(i, _)| row_addrs.value(i)),
            );
        }
    }
    processor.finish_zone(zone.bound)
}

fn process_zone_split<P: ZoneProcessor>(
    processor_factory: &dyn Fn() -> Result<P>,
    split: ZoneSplit,
    track_null_rows: bool,
) -> Result<(Vec<P::ZoneStatistics>, RowAddrTreeMap)> {
    let mut statistics = Vec::with_capacity(split.len());
    let mut null_rows = RowAddrTreeMap::new();
    for zone in split {
        let mut processor = processor_factory()?;
        statistics.push(process_zone(
            &mut processor,
            zone,
            track_null_rows,
            &mut null_rows,
        )?);
    }
    Ok((statistics, null_rows))
}

/// Shared search helper that loops over zones, records metrics, and
/// collects row address ranges for matching zones. The result is always
/// returned as `SearchResult::AtMost` because zone-level pruning can only
/// guarantee a superset of the true matches.
pub fn search_zones<T, F>(
    zones: &[T],
    metrics: &dyn crate::metrics::MetricsCollector,
    mut zone_matches: F,
) -> Result<crate::scalar::SearchResult>
where
    T: AsRef<ZoneBound>,
    F: FnMut(&T) -> Result<bool>,
{
    metrics.record_comparisons(zones.len());
    let mut row_addr_tree_map = RowAddrTreeMap::new();

    // For each zone, check if it might contain the queried value
    for zone in zones {
        if zone_matches(zone)? {
            let bound = zone.as_ref();
            // Calculate the range of row addresses for this zone
            let zone_start_addr = (bound.fragment_id << 32) + bound.start;
            let zone_end_addr = zone_start_addr + bound.length as u64;

            // Add all row addresses in this zone to the result
            row_addr_tree_map.insert_range(zone_start_addr..zone_end_addr);
        }
    }

    Ok(crate::scalar::SearchResult::at_most(row_addr_tree_map))
}

/// Helper that retrains zones from `stream` and appends them to the existing
/// statistics. Returns the combined zone list and the null-row bitmap for the
/// new data only — callers are responsible for merging with any existing bitmap.
pub async fn rebuild_zones<P>(
    existing: &[P::ZoneStatistics],
    trainer: ZoneTrainer<P>,
    stream: SendableRecordBatchStream,
) -> Result<(Vec<P::ZoneStatistics>, RowAddrTreeMap)>
where
    P: ZoneProcessor,
    P::ZoneStatistics: Clone,
{
    let mut combined = existing.to_vec();
    let (mut new_zones, null_rows) = trainer.train(stream).await?;
    combined.append(&mut new_zones);
    Ok((combined, null_rows))
}

/// Parallel counterpart to [`rebuild_zones`].
pub async fn rebuild_zones_parallel<P>(
    existing: &[P::ZoneStatistics],
    trainer: ParallelZoneTrainer<P>,
    stream: SendableRecordBatchStream,
) -> Result<(Vec<P::ZoneStatistics>, RowAddrTreeMap)>
where
    P: ZoneProcessor + 'static,
    P::ZoneStatistics: Clone + Send + 'static,
{
    let mut combined = existing.to_vec();
    let (mut new_zones, null_rows) = trainer.train(stream).await?;
    combined.append(&mut new_zones);
    Ok((combined, null_rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{metrics::LocalMetricsCollector, scalar::SearchResult};
    use arrow_array::{ArrayRef, Int32Array, RecordBatch, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_core::ROW_ADDR;
    use std::sync::Arc;

    #[derive(Debug, Clone, PartialEq)]
    struct MockStats {
        sum: i32,
        bound: ZoneBound,
    }

    #[derive(Debug)]
    struct MockProcessor {
        current_sum: i32,
    }

    impl MockProcessor {
        fn new() -> Self {
            Self { current_sum: 0 }
        }
    }

    impl ZoneProcessor for MockProcessor {
        type ZoneStatistics = MockStats;

        fn process_chunk(&mut self, values: &ArrayRef) -> Result<()> {
            let arr = values.as_any().downcast_ref::<Int32Array>().unwrap();
            self.current_sum += arr.iter().map(|v| v.unwrap_or(0)).sum::<i32>();
            Ok(())
        }

        fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics> {
            Ok(MockStats {
                sum: self.current_sum,
                bound,
            })
        }

        fn reset(&mut self) -> Result<()> {
            self.current_sum = 0;
            Ok(())
        }
    }

    fn batch(values: Vec<i32>, fragments: Vec<u64>, offsets: Vec<u64>) -> RecordBatch {
        let val_array = Arc::new(Int32Array::from(values));
        let row_addrs: Vec<u64> = fragments
            .into_iter()
            .zip(offsets)
            .map(|(frag, off)| (frag << 32) | off)
            .collect();
        let addr_array = Arc::new(UInt64Array::from(row_addrs));
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        RecordBatch::try_new(schema, vec![val_array, addr_array]).unwrap()
    }

    fn nullable_batch(values: Vec<Option<i32>>, row_addrs: Vec<u64>) -> RecordBatch {
        let val_array = Arc::new(Int32Array::from(values));
        let addr_array = Arc::new(UInt64Array::from(row_addrs));
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, true),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        RecordBatch::try_new(schema, vec![val_array, addr_array]).unwrap()
    }

    fn stream_from_batches(batches: Vec<RecordBatch>) -> SendableRecordBatchStream {
        let schema = batches[0].schema();
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(batches.into_iter().map(Ok)),
        ))
    }

    fn reference_zones(rows: &[(u64, Option<i32>)], capacity: usize) -> (Vec<MockStats>, Vec<u64>) {
        fn flush(current: &mut Vec<(u64, Option<i32>)>, zones: &mut Vec<MockStats>) {
            if current.is_empty() {
                return;
            }
            let (first, last) = (current[0].0, current[current.len() - 1].0);
            let (start, end) = (first & 0xFFFF_FFFF, last & 0xFFFF_FFFF);
            zones.push(MockStats {
                sum: current.iter().map(|(_, value)| value.unwrap_or(0)).sum(),
                bound: ZoneBound {
                    fragment_id: first >> 32,
                    start,
                    length: (end - start + 1) as usize,
                },
            });
            current.clear();
        }

        let mut zones = Vec::new();
        let mut nulls = Vec::new();
        let mut current = Vec::new();
        for &(addr, value) in rows {
            if !current.is_empty() && current[current.len() - 1].0 >> 32 != addr >> 32 {
                flush(&mut current, &mut zones);
            }
            if value.is_none() {
                nulls.push(addr);
            }
            current.push((addr, value));
            if current.len() == capacity {
                flush(&mut current, &mut zones);
            }
        }
        flush(&mut current, &mut zones);
        (zones, nulls)
    }

    #[tokio::test]
    async fn splits_single_fragment() {
        // Single fragment with 10 rows, zone capacity = 4.
        // Expect three zones with lengths [4, 4, 2].
        let values = vec![1; 10];
        let offsets: Vec<u64> = (0..10).collect();
        let batch = batch(values, vec![0; 10], offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let processor = MockProcessor::new();
        let trainer = ZoneTrainer::new(processor, 4).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Three zones: offsets [0..=3], [4..=7], [8..=9]
        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 4);
        assert_eq!(stats[1].bound.start, 4);
        assert_eq!(stats[1].bound.length, 4);
        assert_eq!(stats[2].bound.start, 8);
        assert_eq!(stats[2].bound.length, 2); // Last zone has only 2 rows
        assert_eq!(
            stats.iter().map(|s| s.sum).collect::<Vec<_>>(),
            vec![4, 4, 2]
        );
    }

    #[tokio::test]
    async fn flushes_on_fragment_boundary() {
        // Two fragments back to back, capacity is large enough that only fragment
        // boundaries cause zone flushes. Expect two zones (one per fragment).
        let values = vec![1, 1, 1, 2, 2, 2];
        let fragments = vec![0, 0, 0, 1, 1, 1];
        let offsets = vec![0, 1, 2, 0, 1, 2];
        let batch = batch(values, fragments, offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let processor = MockProcessor::new();
        let trainer = ZoneTrainer::new(processor, 10).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Two zones, one per fragment (capacity=10 is large enough)
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].bound.fragment_id, 0);
        assert_eq!(stats[0].bound.length, 3); // Fragment 0: offsets 0,1,2 → length = 2-0+1 = 3
        assert_eq!(stats[1].bound.fragment_id, 1);
        assert_eq!(stats[1].bound.length, 3); // Fragment 1: offsets 0,1,2 → length = 2-0+1 = 3
    }

    #[tokio::test]
    async fn errors_on_out_of_order_offsets() {
        // Offsets go backwards (5 -> 3). Trainer should treat this as invalid input
        // rather than silently emitting a zero-length zone.
        let values = vec![1, 2, 3];
        let fragments = vec![0, 0, 0];
        let offsets = vec![5, 3, 4];
        let batch = batch(values, fragments, offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let processor = MockProcessor::new();
        let trainer = ZoneTrainer::new(processor, 10).unwrap();
        let err = trainer.train(stream).await.unwrap_err();
        assert!(
            format!("{}", err).contains("zone row offsets are out of order"),
            "unexpected error: {err:?}"
        );
    }

    /// Aligned zones take their length from the supplied physical row count,
    /// so every observed offset must lie inside it; a count below an observed
    /// offset would hide a row the zone already summarized.
    #[tokio::test]
    async fn aligned_zone_rejects_offsets_outside_physical_rows() {
        async fn train(offsets: Vec<u64>, physical_rows: u64) -> Result<Vec<MockStats>> {
            let batch = batch(vec![1; offsets.len()], vec![0; offsets.len()], offsets);
            let stream = Box::pin(RecordBatchStreamAdapter::new(
                batch.schema(),
                stream::once(async { Ok(batch) }),
            ));
            let trainer = ZoneTrainer::new(MockProcessor::new(), 10)
                .unwrap()
                .aligned_to(HashMap::from([(0, physical_rows)]));
            trainer.train(stream).await.map(|(zones, _)| zones)
        }

        // The last zone ends at the physical row count even when its
        // trailing rows are deleted.
        let zones = train(vec![0, 1, 2, 3], 8).await.unwrap();
        assert_eq!(zones.len(), 1);
        assert_eq!(zones[0].bound.start, 0);
        assert_eq!(zones[0].bound.length, 8);

        for (offsets, physical_rows) in [
            // The count lies inside the zone but below an observed offset.
            (vec![0, 1, 2, 7], 5),
            // The count equals the zone's start.
            (vec![10, 11, 12], 10),
            // The count lies below the zone's start.
            (vec![10, 11, 12], 5),
        ] {
            let message = train(offsets.clone(), physical_rows)
                .await
                .unwrap_err()
                .to_string();
            assert!(
                message.contains("fragment 0")
                    && message.contains(&format!("physical row count {physical_rows}"))
                    && message.contains(&format!("..={}", offsets.last().unwrap())),
                "{offsets:?} in {physical_rows} rows: {message}"
            );
        }
    }

    #[tokio::test]
    async fn handles_empty_batches() {
        // Empty batches in the stream should be properly skipped without affecting zones.
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));

        let empty_batch = RecordBatch::new_empty(schema.clone());
        let valid_batch = batch(vec![1, 2, 3], vec![0, 0, 0], vec![0, 1, 2]);

        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![
                Ok(empty_batch.clone()),
                Ok(valid_batch),
                Ok(empty_batch),
            ]),
        ));

        let processor = MockProcessor::new();
        let trainer = ZoneTrainer::new(processor, 10).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // One zone containing the 3 valid rows (empty batches skipped)
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].sum, 6);
        assert_eq!(stats[0].bound.fragment_id, 0);
        assert_eq!(stats[0].bound.length, 3);
    }

    #[tokio::test]
    async fn handles_zone_capacity_one() {
        // Each row becomes its own zone when capacity is 1.
        let values = vec![10, 20, 30];
        let offsets = vec![0, 1, 2];
        let batch = batch(values.clone(), vec![0, 0, 0], offsets.clone());
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let processor = MockProcessor::new();
        let trainer = ZoneTrainer::new(processor, 1).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Three zones, one per row (capacity=1)
        assert_eq!(stats.len(), 3);
        for (i, stat) in stats.iter().enumerate() {
            assert_eq!(stat.bound.fragment_id, 0);
            assert_eq!(stat.bound.start, offsets[i]);
            assert_eq!(stat.bound.length, 1); // Each zone contains exactly one row
            assert_eq!(stat.sum, values[i]);
        }
    }

    #[tokio::test]
    async fn handles_large_capacity() {
        // When capacity >> data size, all data fits in one zone.
        let values = vec![1; 100];
        let offsets: Vec<u64> = (0..100).collect();
        let batch = batch(values, vec![0; 100], offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let processor = MockProcessor::new();
        let trainer = ZoneTrainer::new(processor, 10000).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // One zone containing all 100 rows (capacity is large enough)
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].sum, 100);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 100);
    }

    #[tokio::test]
    async fn rejects_zero_capacity() {
        let processor = MockProcessor::new();
        let result = ZoneTrainer::new(processor, 0);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("zone capacity must be greater than zero")
        );
    }

    #[tokio::test]
    async fn sequential_trainer_accepts_borrowed_non_send_processors() {
        #[derive(Debug)]
        struct BorrowedStats {
            sum: i32,
            _not_send: std::rc::Rc<()>,
        }

        #[derive(Debug)]
        struct BorrowedProcessor<'a> {
            sum: i32,
            marker: &'a std::cell::Cell<()>,
        }

        impl ZoneProcessor for BorrowedProcessor<'_> {
            type ZoneStatistics = BorrowedStats;

            fn process_chunk(&mut self, values: &ArrayRef) -> Result<()> {
                let values = values.as_any().downcast_ref::<Int32Array>().unwrap();
                self.sum += values.iter().map(|value| value.unwrap_or(0)).sum::<i32>();
                self.marker.set(());
                Ok(())
            }

            fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics> {
                Ok(BorrowedStats {
                    sum: std::mem::take(&mut self.sum),
                    _not_send: std::rc::Rc::new(()),
                })
            }

            fn reset(&mut self) -> Result<()> {
                self.sum = 0;
                Ok(())
            }
        }

        let marker = std::cell::Cell::new(());
        let trainer = ZoneTrainer::new(
            BorrowedProcessor {
                sum: 0,
                marker: &marker,
            },
            2,
        )
        .unwrap();
        assert!(format!("{trainer:?}").contains("ZoneTrainer"));

        let input = batch(vec![1, 2, 3], vec![0, 0, 0], vec![0, 1, 2]);
        let (stats, _) = trainer
            .train(stream_from_batches(vec![input]))
            .await
            .unwrap();
        assert_eq!(
            stats.iter().map(|stats| stats.sum).collect::<Vec<_>>(),
            vec![3, 3]
        );
    }

    #[tokio::test]
    async fn handles_multiple_batches_same_fragment() {
        // Multiple batches from the same fragment should be properly accumulated into zones.
        let b1 = batch(vec![1, 1], vec![0, 0], vec![0, 1]);
        let b2 = batch(vec![1, 1], vec![0, 0], vec![2, 3]);
        let b3 = batch(vec![1, 1], vec![0, 0], vec![4, 5]);

        let stream = Box::pin(RecordBatchStreamAdapter::new(
            b1.schema(),
            stream::iter(vec![Ok(b1), Ok(b2), Ok(b3)]),
        ));

        let processor = MockProcessor::new();
        let trainer = ZoneTrainer::new(processor, 4).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Two zones: first 4 rows, then remaining 2 rows
        assert_eq!(stats.len(), 2);
        // First zone: offsets [0..=3]
        assert_eq!(stats[0].bound.fragment_id, 0);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 4);
        assert_eq!(stats[0].sum, 4);
        // Second zone: offsets [4..=5]
        assert_eq!(stats[1].bound.fragment_id, 0);
        assert_eq!(stats[1].bound.start, 4);
        assert_eq!(stats[1].bound.length, 2);
        assert_eq!(stats[1].sum, 2);
    }

    #[tokio::test]
    async fn handles_multi_batch_with_fragment_change() {
        // Complex scenario: multiple batches with fragment changes mid-batch.
        // This tests that zones flush correctly at fragment boundaries.
        let b1 = batch(vec![1, 1], vec![0, 0], vec![0, 1]);
        // b2 has fragment change: starts with frag 0, switches to frag 1
        let b2 = batch(vec![1, 1, 2, 2], vec![0, 0, 1, 1], vec![2, 3, 0, 1]);

        let stream = Box::pin(RecordBatchStreamAdapter::new(
            b1.schema(),
            stream::iter(vec![Ok(b1), Ok(b2)]),
        ));

        let processor = MockProcessor::new();
        let trainer = ZoneTrainer::new(processor, 3).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Three zones: frag 0 full zone, frag 0 partial (flushed at boundary), frag 1
        assert_eq!(stats.len(), 3);

        // Zone 0: Fragment 0, offsets [0..=2] (fills capacity)
        assert_eq!(stats[0].bound.fragment_id, 0);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 3);
        assert_eq!(stats[0].sum, 3);

        // Zone 1: Fragment 0, offset 3 (partial, flushed at fragment boundary)
        assert_eq!(stats[1].bound.fragment_id, 0);
        assert_eq!(stats[1].bound.start, 3);
        assert_eq!(stats[1].bound.length, 1);
        assert_eq!(stats[1].sum, 1);

        // Zone 2: Fragment 1, offsets [0..=1]
        assert_eq!(stats[2].bound.fragment_id, 1);
        assert_eq!(stats[2].bound.start, 0);
        assert_eq!(stats[2].bound.length, 2);
        assert_eq!(stats[2].sum, 4);
    }

    #[tokio::test]
    async fn handles_non_contiguous_offsets_after_deletion() {
        // CRITICAL: Test deletion scenario with non-contiguous row offsets.
        // This is the main reason for tracking first/last offsets.
        // Simulate a zone where rows 2, 3, 4, 6 have been deleted.
        let values = vec![1, 1, 1, 1, 1, 1]; // 6 actual rows
        let fragments = vec![0, 0, 0, 0, 0, 0];
        let offsets = vec![0, 1, 5, 7, 8, 9]; // Non-contiguous!

        let batch = batch(values, fragments, offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let processor = MockProcessor::new();
        let trainer = ZoneTrainer::new(processor, 4).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Should create 2 zones (capacity=4):
        // Zone 0: rows at offsets [0, 1, 5, 7] (4 rows)
        // Zone 1: rows at offsets [8, 9] (2 rows)
        assert_eq!(stats.len(), 2);

        // First zone: 4 rows, but offset span is [0..=7] so length=8 (due to gaps)
        assert_eq!(stats[0].sum, 4);
        assert_eq!(stats[0].bound.fragment_id, 0);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 8); // Address span: 7 - 0 + 1

        // Second zone: 2 rows, offset span is [8..=9] so length=2
        assert_eq!(stats[1].sum, 2);
        assert_eq!(stats[1].bound.fragment_id, 0);
        assert_eq!(stats[1].bound.start, 8);
        assert_eq!(stats[1].bound.length, 2); // Address span: 9 - 8 + 1
    }

    #[tokio::test]
    async fn handles_deletion_with_large_gaps() {
        // Extreme deletion scenario: very large gaps between consecutive rows.
        let values = vec![1, 1, 1];
        let fragments = vec![0, 0, 0];
        let offsets = vec![0, 100, 200]; // Huge gaps!

        let batch = batch(values, fragments, offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let processor = MockProcessor::new();
        let trainer = ZoneTrainer::new(processor, 10).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // One zone with 3 rows, but offset span [0..=200] so length=201 due to large gaps
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].sum, 3);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 201); // Span: 200 - 0 + 1
    }

    #[tokio::test]
    async fn handles_non_contiguous_fragment_ids() {
        // CRITICAL: Test fragment IDs that are not consecutive (e.g., after fragment deletion).
        // Original code assumed fragment_id + 1, which would fail here.
        // Fragment IDs: 0, 5, 10 (non-consecutive!)
        let values = vec![1, 1, 2, 2, 3, 3];
        let fragments = vec![0, 0, 5, 5, 10, 10]; // Gaps in fragment IDs
        let offsets = vec![0, 1, 0, 1, 0, 1];

        let batch = batch(values, fragments, offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let processor = MockProcessor::new();
        let trainer = ZoneTrainer::new(processor, 10).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Should create 3 zones (one per fragment)
        assert_eq!(stats.len(), 3);

        // Fragment 0
        assert_eq!(stats[0].bound.fragment_id, 0);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 2);
        assert_eq!(stats[0].sum, 2);

        // Fragment 5 (not 1!)
        assert_eq!(stats[1].bound.fragment_id, 5);
        assert_eq!(stats[1].bound.start, 0);
        assert_eq!(stats[1].bound.length, 2);
        assert_eq!(stats[1].sum, 4);

        // Fragment 10 (not 2!)
        assert_eq!(stats[2].bound.fragment_id, 10);
        assert_eq!(stats[2].bound.start, 0);
        assert_eq!(stats[2].bound.length, 2);
        assert_eq!(stats[2].sum, 6);
    }

    #[test]
    fn search_zones_collects_row_ranges() {
        // Ensure the shared helper converts matching zones into the correct row-id
        // ranges (fragment upper bits + local offsets) while skipping non-matching
        // zones. This protects the helper if we modify how RowAddrTreeMap ranges are
        // inserted in the future.
        #[derive(Debug)]
        struct DummyZone {
            bound: ZoneBound,
            matches: bool,
        }

        impl AsRef<ZoneBound> for DummyZone {
            fn as_ref(&self) -> &ZoneBound {
                &self.bound
            }
        }

        let zones = vec![
            DummyZone {
                bound: ZoneBound {
                    fragment_id: 0,
                    start: 0,
                    length: 2,
                },
                matches: true,
            },
            DummyZone {
                bound: ZoneBound {
                    fragment_id: 1,
                    start: 5,
                    length: 3,
                },
                matches: false,
            },
            DummyZone {
                bound: ZoneBound {
                    fragment_id: 2,
                    start: 10,
                    length: 1,
                },
                matches: true,
            },
        ];

        let metrics = LocalMetricsCollector::default();
        let result = search_zones(&zones, &metrics, |zone| Ok(zone.matches)).unwrap();
        let SearchResult::AtMost(map) = result else {
            panic!("search_zones should return AtMost for dummy zones");
        };

        // Fragment 0, offsets 0 and 1
        assert!(map.selected(0));
        assert!(map.selected(1));
        // Fragment 1 should be skipped entirely
        assert!(!map.selected((1_u64 << 32) + 5));
        assert!(!map.selected((1_u64 << 32) + 7));
        // Fragment 2 includes only the single offset 10
        assert!(map.selected((2_u64 << 32) + 10));
        assert!(!map.selected((2_u64 << 32) + 11));
    }

    #[test]
    fn search_zones_returns_empty_when_no_match() {
        #[derive(Debug)]
        struct DummyZone {
            bound: ZoneBound,
            matches: bool,
        }

        impl AsRef<ZoneBound> for DummyZone {
            fn as_ref(&self) -> &ZoneBound {
                &self.bound
            }
        }

        // Both zones are marked as non-matching. The helper should return an empty map.
        let zones = vec![
            DummyZone {
                bound: ZoneBound {
                    fragment_id: 0,
                    start: 0,
                    length: 4,
                },
                matches: false,
            },
            DummyZone {
                bound: ZoneBound {
                    fragment_id: 1,
                    start: 10,
                    length: 2,
                },
                matches: false,
            },
        ];

        let metrics = LocalMetricsCollector::default();
        let result = search_zones(&zones, &metrics, |zone| Ok(zone.matches)).unwrap();
        let SearchResult::AtMost(map) = result else {
            panic!("expected AtMost result");
        };
        // No zones should be inserted when every predicate evaluates to false
        assert!(map.is_empty());
    }

    #[tokio::test]
    async fn rebuild_zones_appends_new_stats() {
        let existing = vec![MockStats {
            sum: 50,
            bound: ZoneBound {
                fragment_id: 0,
                start: 0,
                length: 2,
            },
        }];

        let batch = batch(vec![3, 4], vec![1, 1], vec![0, 1]);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let trainer = ZoneTrainer::new(MockProcessor::new(), 2).unwrap();
        let (rebuilt, _) = rebuild_zones(&existing, trainer, stream).await.unwrap();
        // Existing zone should remain unchanged and new stats appended afterwards
        assert_eq!(rebuilt.len(), 2);
        assert_eq!(rebuilt[0].sum, 50);
        assert_eq!(rebuilt[1].sum, 7);
        assert_eq!(rebuilt[1].bound.fragment_id, 1);
        assert_eq!(rebuilt[1].bound.start, 0);
        assert_eq!(rebuilt[1].bound.length, 2);
    }

    #[tokio::test]
    async fn rebuild_zones_handles_multi_fragment_stream() {
        let existing = vec![MockStats {
            sum: 10,
            bound: ZoneBound {
                fragment_id: 0,
                start: 0,
                length: 1,
            },
        }];

        // Construct a stream with two fragments. Trainer should emit two zones that
        // get appended after the existing entries.
        let batch = batch(vec![5, 5, 6, 6], vec![1, 1, 2, 2], vec![0, 1, 0, 1]);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let trainer = ZoneTrainer::new(MockProcessor::new(), 2).unwrap();
        let (rebuilt, _) = rebuild_zones(&existing, trainer, stream).await.unwrap();
        // Existing zone plus two new fragments should yield three total zones
        assert_eq!(rebuilt.len(), 3);
        assert_eq!(rebuilt[0].bound.fragment_id, 0);
        assert_eq!(rebuilt[1].bound.fragment_id, 1);
        assert_eq!(rebuilt[2].bound.fragment_id, 2);
        assert_eq!(rebuilt[1].sum, 10);
        assert_eq!(rebuilt[2].sum, 12);
    }

    #[tokio::test]
    async fn parallel_assembler_batches_tiny_zones_incrementally() {
        let num_rows = MIN_ROWS_PER_SPLIT * 2 + 17;
        let input = batch(
            vec![1; num_rows],
            vec![0; num_rows],
            (0..num_rows as u64).collect(),
        );
        let mut assembler = ZoneSplitAssembler::new(stream_from_batches(vec![input]), 1, 1, None);

        let first_split = assembler.next_split().await.unwrap().unwrap();
        assert_eq!(first_split.len(), MIN_ROWS_PER_SPLIT);
        for zone in &first_split {
            assert_eq!(zone.scanned_rows, 1);
        }
        assert_eq!(assembler.input_offset, MIN_ROWS_PER_SPLIT);

        let second_split = assembler.next_split().await.unwrap().unwrap();
        assert_eq!(second_split.len(), MIN_ROWS_PER_SPLIT);

        let trailing_split = assembler.next_split().await.unwrap().unwrap();
        assert_eq!(trailing_split.len(), 17);
        assert!(assembler.next_split().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn parallel_trainer_preserves_order_across_splits() {
        let num_rows = MIN_ROWS_PER_SPLIT * 2 + 17;
        let values: Vec<i32> = (0..num_rows).map(|value| value as i32).collect();
        let input = batch(
            values.clone(),
            vec![0; num_rows],
            (0..num_rows as u64).collect(),
        );
        let trainer = ParallelZoneTrainer::new(|| Ok(MockProcessor::new()), 1).unwrap();
        let (stats, _) = trainer
            .train(stream_from_batches(vec![input]))
            .await
            .unwrap();

        assert_eq!(
            stats.iter().map(|stats| stats.sum).collect::<Vec<_>>(),
            values
        );
    }

    #[tokio::test]
    async fn parallel_trainer_matches_reference_on_random_streams() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        for trial in 0..4u64 {
            let mut rng = SmallRng::seed_from_u64(trial);
            let mut rows: Vec<(u64, Option<i32>)> = Vec::new();
            for fragment_id in [0u64, 3, 4, 11, 100] {
                let num_rows = rng.random_range(0..60);
                let mut offset = rng.random_range(0..4);
                for _ in 0..num_rows {
                    let value = (!rng.random_bool(0.25)).then(|| rng.random_range(-100..100));
                    rows.push(((fragment_id << 32) | offset, value));
                    offset += rng.random_range(1..5);
                }
            }

            let batch_size = [1, 3, 7, 100][(trial % 4) as usize];
            let batches: Vec<RecordBatch> = rows
                .chunks(batch_size)
                .map(|chunk| {
                    nullable_batch(
                        chunk.iter().map(|(_, value)| *value).collect(),
                        chunk.iter().map(|(addr, _)| *addr).collect(),
                    )
                })
                .collect();
            if batches.is_empty() {
                continue;
            }

            for capacity in [1u64, 3, 8, 8192] {
                let (expected_zones, expected_nulls) = reference_zones(&rows, capacity as usize);
                let trainer =
                    ParallelZoneTrainer::new(|| Ok(MockProcessor::new()), capacity).unwrap();
                let (zones, null_rows) = trainer
                    .train(stream_from_batches(batches.clone()))
                    .await
                    .unwrap();
                assert_eq!(zones, expected_zones, "trial={trial} capacity={capacity}");
                assert_eq!(
                    null_rows,
                    RowAddrTreeMap::from_iter(expected_nulls),
                    "trial={trial} capacity={capacity}"
                );
            }
        }
    }

    #[tokio::test]
    async fn parallel_trainer_honors_alignment_and_null_tracking() {
        let input = batch(vec![1; 6], vec![0; 6], vec![0, 1, 2, 3, 8, 9]);
        let trainer = ParallelZoneTrainer::new(|| Ok(MockProcessor::new()), 4)
            .unwrap()
            .aligned_to(HashMap::from([(0, 10)]));
        let (zones, _) = trainer
            .train(stream_from_batches(vec![input]))
            .await
            .unwrap();
        assert_eq!(zones.len(), 2);
        assert_eq!(
            zones[0].bound,
            ZoneBound {
                fragment_id: 0,
                start: 0,
                length: 4,
            }
        );
        assert_eq!(
            zones[1].bound,
            ZoneBound {
                fragment_id: 0,
                start: 8,
                length: 2,
            }
        );

        let input = nullable_batch(vec![Some(1), None, Some(3), None], vec![7, 9, 11, 15]);
        let trainer = ParallelZoneTrainer::new(|| Ok(MockProcessor::new()), 4).unwrap();
        let (_, null_rows) = trainer
            .train(stream_from_batches(vec![input.clone()]))
            .await
            .unwrap();
        assert_eq!(null_rows, RowAddrTreeMap::from_iter([9, 15]));

        let trainer = ParallelZoneTrainer::new(|| Ok(MockProcessor::new()), 4)
            .unwrap()
            .with_null_tracking(false);
        let (_, null_rows) = trainer
            .train(stream_from_batches(vec![input]))
            .await
            .unwrap();
        assert_eq!(null_rows, RowAddrTreeMap::new());
    }

    #[tokio::test]
    async fn parallel_rebuild_zones_appends_new_stats() {
        let existing = vec![MockStats {
            sum: 50,
            bound: ZoneBound {
                fragment_id: 0,
                start: 0,
                length: 2,
            },
        }];
        let input = batch(vec![3, 4], vec![1, 1], vec![0, 1]);
        let trainer = ParallelZoneTrainer::new(|| Ok(MockProcessor::new()), 2).unwrap();
        let (rebuilt, _) =
            rebuild_zones_parallel(&existing, trainer, stream_from_batches(vec![input]))
                .await
                .unwrap();
        assert_eq!(
            rebuilt.iter().map(|stats| stats.sum).collect::<Vec<_>>(),
            vec![50, 7]
        );
    }
}
