// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! No-values List pages still distinguish empty from null parent slots.

use std::ops::Range;

use arrow_array::{ArrayRef, make_array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use lance_core::{Error, Result, utils::allocation::Reservation};

use crate::{
    data::{AllNullDataBlock, DataBlock},
    decoder::{PageEncoding, PageInfo},
    encodings::physical::sequence::{SequenceDecoder, SequenceMetadata},
    format::{metadata::CompressiveEncoding, pb23},
};

fn corrupt(message: &str) -> Error {
    Error::corrupt_file_named("admitted empty List", message)
}

fn sequence(encoding: Option<&CompressiveEncoding>, count: u64) -> Result<SequenceMetadata> {
    let encoding = encoding.ok_or_else(|| corrupt("missing unsigned descriptor"))?;
    let sequence = SequenceDecoder::try_new(encoding, 64, count)?;
    sequence
        .metadata()
        .filter(|_| !sequence.requires_payload())
        .ok_or_else(|| {
            Error::not_supported("admitted empty List requires metadata-only positions/counts")
        })
}

#[derive(Clone, Copy, Debug)]
struct Validity {
    null_positions: bool,
    start: u64,
    step: u64,
    count: u64,
}

impl Validity {
    fn parse(validity: Option<&pb23::SparseValiditySet>, domain: u64) -> Result<Self> {
        let validity = validity.ok_or_else(|| corrupt("missing validity"))?;
        let null_positions = match validity.meaning() {
            pb23::sparse_validity_set::Meaning::SparseValidityNullPositions => true,
            pb23::sparse_validity_set::Meaning::SparseValidityValidPositions => false,
            _ => return Err(corrupt("invalid validity meaning")),
        };
        let count = validity.num_positions;
        if count > domain {
            return Err(corrupt("validity count exceeds slot domain"));
        }
        let (start, step) = match sequence(validity.positions.as_ref(), count)? {
            SequenceMetadata::Empty => (0, 1),
            SequenceMetadata::Constant(value) if count == 1 => (value, 1),
            SequenceMetadata::Range { start, step } => (start, step),
            _ => return Err(corrupt("validity positions must strictly increase")),
        };
        // SequenceDecoder checks range overflow and a positive step.
        if count > 0 && start + (count - 1) * step >= domain {
            return Err(corrupt("validity position exceeds slot domain"));
        }
        Ok(Self {
            null_positions,
            start,
            step,
            count,
        })
    }

    fn valid(&self, row: u64) -> bool {
        let selected = row >= self.start
            && (row - self.start).is_multiple_of(self.step)
            && (row - self.start) / self.step < self.count;
        selected != self.null_positions
    }

    fn null_count(&self, rows: &Range<u64>) -> u64 {
        let lower = rows
            .start
            .saturating_sub(self.start)
            .div_ceil(self.step)
            .min(self.count);
        let upper = rows
            .end
            .saturating_sub(self.start)
            .div_ceil(self.step)
            .min(self.count);
        let selected = upper - lower;
        if self.null_positions {
            selected
        } else {
            rows.end - rows.start - selected
        }
    }
}

#[derive(Debug)]
pub(super) struct EmptyListScheduler {
    rows: u64,
    depth: usize,
    validity: Validity,
}

impl EmptyListScheduler {
    pub(super) fn try_from_page(page: &PageInfo) -> Result<Option<Self>> {
        let PageEncoding::Planar(layout) = &page.encoding else {
            return Ok(None);
        };
        if !matches!(
            layout.values.as_ref().and_then(|v| v.codec.as_ref()),
            Some(pb23::value_plane::Codec::None(_))
        ) {
            return Ok(None);
        }
        let Some(structure) = layout.sparse() else {
            return Ok(None);
        };
        let Some(pb23::sparse_structural_layer::Layer::List(first)) =
            structure.layers.first().and_then(|l| l.layer.as_ref())
        else {
            return Ok(None);
        };
        if first.num_child_slots != 0 {
            return Ok(None);
        }
        let validity = Validity::parse(first.validity.as_ref(), page.num_rows)?;
        let mut depth = 0;
        let mut leaf = false;
        for layer in &structure.layers {
            let domain = if depth == 0 { page.num_rows } else { 0 };
            match &layer.layer {
                Some(pb23::sparse_structural_layer::Layer::List(list)) if !leaf => {
                    if list.num_child_slots != 0 || list.num_non_empty_positions != 0 {
                        return Err(corrupt("zero-child page has nonempty descendant domain"));
                    }
                    sequence(list.non_empty_positions.as_ref(), 0)?;
                    sequence(list.counts.as_ref(), 0)?;
                    Validity::parse(list.validity.as_ref(), domain)?;
                    depth += 1;
                }
                Some(pb23::sparse_structural_layer::Layer::Validity(v)) if !leaf => {
                    Validity::parse(Some(v), 0)?;
                    leaf = true;
                }
                _ => {
                    return Err(Error::not_supported(
                        "admitted empty List requires a List chain and one byte leaf",
                    ));
                }
            }
        }
        if !leaf {
            return Err(corrupt("missing leaf validity"));
        }
        if !page.buffer_offsets_and_sizes.is_empty() {
            return Err(corrupt("metadata-only List has unexpected buffers"));
        }
        Ok(Some(Self {
            rows: page.num_rows,
            depth,
            validity,
        }))
    }

    pub(super) fn validate_array_type(&self, data_type: &DataType) -> Result<()> {
        let mut current = data_type;
        for _ in 0..self.depth {
            let (DataType::List(field) | DataType::LargeList(field)) = current else {
                return Err(Error::not_supported(
                    "empty List Arrow depth differs from stored structure",
                ));
            };
            current = field.data_type();
        }
        if !matches!(
            current,
            DataType::Binary | DataType::Utf8 | DataType::LargeBinary | DataType::LargeUtf8
        ) {
            return Err(Error::not_supported(
                "admitted empty List requires a byte/string leaf",
            ));
        }
        Ok(())
    }

    pub(super) fn prepare(&self, rows: Range<u64>, data_type: DataType) -> Result<EmptyListInput> {
        self.validate_array_type(&data_type)?;
        if rows.start >= rows.end || rows.end > self.rows {
            return Err(Error::invalid_input(
                "empty List selection is outside the page or empty",
            ));
        }
        Ok(EmptyListInput {
            rows,
            data_type,
            validity: self.validity,
        })
    }
}

// A zero-length byte/List tree owns one zero offset at every level and no bitmap.
fn empty_bytes(data_type: &DataType) -> Result<usize> {
    match data_type {
        DataType::List(field) => empty_bytes(field.data_type())?.checked_add(4),
        DataType::LargeList(field) => empty_bytes(field.data_type())?.checked_add(8),
        DataType::Binary | DataType::Utf8 => Some(4),
        DataType::LargeBinary | DataType::LargeUtf8 => Some(8),
        _ => {
            return Err(Error::not_supported(
                "admitted empty List requires a byte/string leaf",
            ));
        }
    }
    .ok_or_else(|| Error::invalid_input("empty List offsets size overflows"))
}

#[derive(Debug)]
pub(super) struct EmptyListInput {
    rows: Range<u64>,
    data_type: DataType,
    validity: Validity,
}

impl EmptyListInput {
    pub(super) fn decode_in(self, workspace: &mut Reservation) -> Result<Option<ArrayRef>> {
        let len = usize::try_from(self.rows.end - self.rows.start)
            .map_err(|_| Error::invalid_input("empty List length exceeds usize"))?;
        let (width, field) = match &self.data_type {
            DataType::List(field) => (4, field),
            DataType::LargeList(field) => (8, field),
            _ => return Err(Error::invalid_input("expected admitted List shape")),
        };
        let offsets = len
            .checked_add(1)
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| Error::invalid_input("empty List offsets size overflows"))?;
        let bitmap = if self.validity.null_count(&self.rows) == 0 {
            0
        } else {
            len.div_ceil(8)
        };
        let own = offsets
            .checked_add(bitmap)
            .ok_or_else(|| Error::invalid_input("empty List size overflows"))?;
        let total = own
            .checked_add(empty_bytes(field.data_type())?)
            .ok_or_else(|| Error::invalid_input("empty List tree size overflows"))?;
        if total > workspace.len() {
            return Ok(None);
        }
        let Some(mut buffer) = workspace.try_allocate(own)? else {
            return Ok(None);
        };
        buffer.as_mut().fill(0);
        if bitmap > 0 {
            for (index, row) in self.rows.clone().enumerate() {
                if self.validity.valid(row) {
                    buffer.as_mut()[offsets + index / 8] |= 1 << (index % 8);
                }
            }
        }
        let buffer = Buffer::from(buffer.freeze());
        let nulls = (bitmap > 0).then(|| {
            NullBuffer::new(BooleanBuffer::new(
                buffer.slice_with_length(offsets, bitmap),
                0,
                len,
            ))
        });
        let child = DataBlock::AllNull(AllNullDataBlock { num_values: 0 })
            .try_to_arrow_in(field.data_type(), workspace)?
            .ok_or_else(|| Error::internal("empty List preflight underestimated child offsets"))?;
        let array = ArrayData::builder(self.data_type)
            .len(len)
            .add_buffer(buffer.slice_with_length(0, offsets))
            .add_child_data(child)
            .nulls(nulls)
            .build()?;
        Ok(Some(make_array(array)))
    }
}

#[cfg(test)]
mod tests;
