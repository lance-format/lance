// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read Constant scalar and selected levels without an uncharged page cache.

use std::{future::Future, ops::Range, sync::Arc};

use arrow_array::{ArrayRef, make_array};
use arrow_buffer::{Buffer, ScalarBuffer};
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;
use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::{DecodeConstantTask, admitted};
use crate::{
    decoder::PageInfo,
    encodings::logical::primitive::fullzip_decode::array_admitted::Shape,
    format::{EncodingUtils, metadata::ConstantLayout},
    repdef::{DefinitionInterpretation, RepDefUnraveler, max_visible_level},
};

fn corrupt(message: &str) -> Error {
    Error::corrupt_file_named("Constant", message)
}

#[derive(Debug)]
pub(in super::super) struct ConstantArrayScheduler {
    scalar: Range<u64>,
    rep: Option<Range<u64>>,
    def: Option<Range<u64>>,
    meanings: Arc<[DefinitionInterpretation]>,
    rows: u64,
}

impl ConstantArrayScheduler {
    pub(in super::super) fn try_new(page: &PageInfo, layout: &ConstantLayout) -> Result<Self> {
        // Byte scalars have two Arrow buffers and are never inline. All-null and
        // fixed-width constant layouts need their own explicitly admitted path.
        if layout.inline_value.is_some() || !matches!(page.buffer_offsets_and_sizes.len(), 1 | 3) {
            return Err(Error::not_supported(
                "admitted Constant pages require a stored byte scalar",
            ));
        }
        let ranges = page
            .buffer_offsets_and_sizes
            .iter()
            .map(|(start, len)| {
                start
                    .checked_add(*len)
                    .map(|end| *start..end)
                    .ok_or_else(|| corrupt("buffer range overflows"))
            })
            .collect::<Result<Vec<_>>>()?;
        let meanings: Arc<[_]> = layout
            .layers
            .iter()
            .map(|layer| EncodingUtils::try_repdef_layer_to_def_interp(*layer))
            .collect::<Result<Vec<_>>>()?
            .into();
        RepDefUnraveler::validate_dense_levels(None, None, &meanings)?;
        let rep = ranges.get(1).filter(|range| !range.is_empty()).cloned();
        let def = ranges.get(2).filter(|range| !range.is_empty()).cloned();
        if rep.is_none() && meanings.iter().any(|layer| layer.is_list()) {
            return Err(corrupt("list page has no repetition levels"));
        }
        if rep
            .iter()
            .chain(def.iter())
            .any(|range| (range.end - range.start) % 2 != 0)
        {
            return Err(corrupt("level buffer has an odd byte length"));
        }
        let levels = rep
            .as_ref()
            .map_or(page.num_rows, |range| (range.end - range.start) / 2);
        if def
            .as_ref()
            .is_some_and(|range| (range.end - range.start) / 2 != levels)
        {
            return Err(corrupt(
                "definition level count differs from repetition or row count",
            ));
        }
        Ok(Self {
            scalar: ranges[0].clone(),
            rep,
            def,
            meanings,
            rows: page.num_rows,
        })
    }

    pub(in super::super) fn validate_array_type(&self, data_type: &DataType) -> Result<()> {
        Shape::from_layers(data_type, &self.meanings, None).map(|_| ())
    }

    // Constant pages lack a repetition index. Locate just this interval with a
    // reusable <=4KiB charged window. Never retain an entire page's levels or a
    // row-sized index, even when a late one-row selection crosses many windows.
    async fn level_range<F, Fut>(
        &self,
        rows: &Range<u64>,
        work: &mut Reservation,
        read: &mut F,
    ) -> Result<Option<Range<u64>>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        let Some(rep) = &self.rep else {
            return Ok(Some(rows.clone()));
        };
        let capacity = work
            .len()
            .min(4096)
            .min(usize::try_from(rep.end - rep.start).unwrap_or(usize::MAX))
            / 2
            * 2;
        if capacity == 0 {
            return Ok(None);
        }
        let Some(mut scratch) = work.try_allocate(capacity)? else {
            return Ok(None);
        };
        let max_rep = self.meanings.iter().filter(|layer| layer.is_list()).count() as u16;
        let mut row = 0_u64;
        let mut selected_start = None;
        let mut position = rep.start;
        while position < rep.end {
            let len = (rep.end - position).min(capacity as u64) as usize;
            if scratch.as_ref().len() != len {
                // A shorter final window needs no old contents. Release first;
                // resize would require unnecessary old/new overlap headroom.
                scratch.recycle(work)?;
                let Some(replacement) = work.try_allocate(len)? else {
                    return Ok(None);
                };
                scratch = replacement;
            }
            scratch = read(position..position + len as u64, scratch).await?;
            if scratch.as_ref().len() != len {
                return Err(corrupt("short repetition read"));
            }
            for (offset, bytes) in scratch.as_ref().as_chunks::<2>().0.iter().enumerate() {
                let level = u16::from_le_bytes(*bytes);
                let index = (position - rep.start) / 2 + offset as u64;
                if level > max_rep || (index == 0 && level != max_rep) {
                    return Err(corrupt("invalid repetition row boundary"));
                }
                if level == max_rep {
                    if row == rows.start {
                        selected_start = Some(index);
                    }
                    if row == rows.end {
                        if rows.end == self.rows {
                            return Err(corrupt("more repetition rows than page metadata"));
                        }
                        scratch.recycle(work)?;
                        return Ok(Some(
                            selected_start.ok_or_else(|| corrupt("missing selected row start"))?
                                ..index,
                        ));
                    }
                    row += 1;
                }
            }
            position += len as u64;
        }
        if row != self.rows || rows.end != row {
            return Err(corrupt("repetition row count differs from page metadata"));
        }
        scratch.recycle(work)?;
        Ok(Some(
            selected_start.ok_or_else(|| corrupt("missing selected row start"))?
                ..(rep.end - rep.start) / 2,
        ))
    }

    pub(in super::super) async fn prepare<F, Fut>(
        &self,
        rows: Range<u64>,
        data_type: DataType,
        work: &mut Reservation,
        mut read: F,
    ) -> Result<Option<ConstantArrayInput>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        self.validate_array_type(&data_type)?;
        if rows.start >= rows.end || rows.end > self.rows {
            return Err(Error::invalid_input(
                "Constant array range must be nonempty and within the page",
            ));
        }
        let Some(levels) = self.level_range(&rows, work, &mut read).await? else {
            return Ok(None);
        };
        let ranges = [
            Some(self.scalar.clone()),
            self.rep
                .as_ref()
                .map(|range| range.start + levels.start * 2..range.start + levels.end * 2),
            self.def
                .as_ref()
                .map(|range| range.start + levels.start * 2..range.start + levels.end * 2),
        ];
        let required = ranges.iter().flatten().try_fold(0_usize, |sum, range| {
            usize::try_from(range.end - range.start)
                .ok()
                .and_then(|bytes| sum.checked_add(bytes))
        });
        if required.is_none_or(|required| required > work.len()) {
            return Ok(None);
        }
        let mut inputs = [None, None, None];
        for (slot, range) in inputs.iter_mut().zip(ranges) {
            if let Some(range) = range {
                let len = (range.end - range.start) as usize;
                let Some(buffer) = work.try_allocate(len)? else {
                    return Ok(None);
                };
                let buffer = read(range, buffer).await?;
                if buffer.as_ref().len() != len {
                    return Err(corrupt("short scalar or level read"));
                }
                *slot = Some(buffer);
            }
        }
        let [scalar, rep, def] = inputs;
        Ok(Some(ConstantArrayInput {
            scalar: scalar.ok_or_else(|| corrupt("missing scalar input"))?,
            rep,
            def,
            meanings: self.meanings.clone(),
            data_type,
            count: rows.end - rows.start,
            levels: usize::try_from(levels.end - levels.start)
                .map_err(|_| corrupt("level range exceeds address space"))?,
        }))
    }
}

#[derive(Debug)]
pub(in super::super) struct ConstantArrayInput {
    scalar: ReservedBuffer,
    rep: Option<ReservedBuffer>,
    def: Option<ReservedBuffer>,
    meanings: Arc<[DefinitionInterpretation]>,
    data_type: DataType,
    count: u64,
    levels: usize,
}

// Parse the existing scalar framing without the ordinary decoder's to_vec copies.
// Only offsets need an aligned copy (the framing can misalign i64); the values
// slice retains the full physical serialized input owner until expansion ends.
fn scalar_array(
    input: ReservedBuffer,
    data_type: &DataType,
    work: &mut Reservation,
) -> Result<Option<ArrayRef>> {
    let bytes = input.as_ref();
    let header = bytes
        .get(..12)
        .ok_or_else(|| corrupt("truncated scalar header"))?;
    let words: Vec<_> = header
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word) as usize)
        .collect();
    let width = match data_type {
        DataType::Binary | DataType::Utf8 => 4,
        DataType::LargeBinary | DataType::LargeUtf8 => 8,
        _ => {
            return Err(Error::not_supported(
                "Constant scalar must be a byte/string value",
            ));
        }
    };
    if words[0] != 2
        || words[1] != width * 2
        || words[2].checked_add(12 + words[1]) != Some(bytes.len())
    {
        return Err(corrupt("invalid byte scalar buffer framing"));
    }
    let Some(mut offsets) = work.try_allocate(words[1])? else {
        return Ok(None);
    };
    offsets.as_mut().copy_from_slice(&bytes[12..12 + words[1]]);
    let start = 12 + words[1];
    let values = Buffer::from(input.freeze().slice(start..));
    let array = ArrayDataBuilder::new(data_type.clone())
        .len(1)
        .null_count(0)
        .buffers(vec![Buffer::from(offsets.freeze()), values])
        .build()?;
    Ok(Some(make_array(array)))
}

impl ConstantArrayInput {
    pub(in super::super) fn decode_in(self, work: &mut Reservation) -> Result<Option<ArrayRef>> {
        let shape = Shape::from_layers(&self.data_type, &self.meanings, None)?;
        let Some(scalar) = scalar_array(self.scalar, shape.leaf, work)? else {
            return Ok(None);
        };
        let as_levels =
            |buffer: ReservedBuffer| ScalarBuffer::<u16>::from(Buffer::from(buffer.freeze()));
        let task = DecodeConstantTask {
            scalar,
            rep: self.rep.map(as_levels),
            def: self.def.map(as_levels),
            level_slices: vec![0..self.levels],
            visible_items_total: self.levels as u64,
            max_visible_def: max_visible_level(&self.meanings),
            def_meaning: self.meanings,
        };
        let page = admitted::decode(&task, work)?;
        drop(task);
        match page {
            Some(page) => shape.finish(page, self.count, work),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests;
