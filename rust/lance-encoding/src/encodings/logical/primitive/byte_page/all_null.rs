// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Materialize only flat null pages whose complete validity is proven by metadata.

use std::ops::Range;

use arrow_array::{ArrayRef, make_array};
use arrow_schema::DataType;
use lance_core::{Error, Result, utils::allocation::Reservation};

use crate::{
    data::{AllNullDataBlock, DataBlock},
    decoder::{PageEncoding, PageInfo},
    encodings::physical::sequence::{SequenceDecoder, SequenceMetadata},
    format::{metadata::page_layout::Layout, pb23},
};

#[derive(Debug)]
pub(super) struct AllNullScheduler {
    rows: u64,
}

fn corrupt(message: &str) -> Error {
    Error::corrupt_file_named("admitted all-null", message)
}

impl AllNullScheduler {
    pub(super) fn try_from_page(page: &PageInfo) -> Result<Option<Self>> {
        match &page.encoding {
            PageEncoding::Structural(layout) => {
                let Some(Layout::ConstantLayout(layout)) = &layout.layout else {
                    return Ok(None);
                };
                if layout.inline_value.is_some() || !page.buffer_offsets_and_sizes.is_empty() {
                    return Ok(None);
                }
                if layout.layers.as_slice() != [pb23::RepDefLayer::RepdefNullableItem as i32] {
                    // With nested layers, no values does not imply null parent rows.
                    return Ok(None);
                }
                if layout.rep_compression.is_some()
                    || layout.def_compression.is_some()
                    || layout.num_rep_values != 0
                    || layout.num_def_values != 0
                {
                    return Err(corrupt(
                        "flat null page unexpectedly describes level streams",
                    ));
                }
            }
            PageEncoding::Planar(layout) => {
                if !matches!(
                    layout
                        .values
                        .as_ref()
                        .and_then(|values| values.codec.as_ref()),
                    Some(pb23::value_plane::Codec::None(_))
                ) {
                    return Ok(None);
                }
                let Some(structure) = layout.sparse() else {
                    // The dense planar contract also permits a single nullable
                    // item layer with no values. Reuse its validated lowering.
                    return Self::try_from_page(&PageInfo {
                        encoding: PageEncoding::Structural(layout.dense_execution_layout()?),
                        num_rows: page.num_rows,
                        priority: page.priority,
                        buffer_offsets_and_sizes: page.buffer_offsets_and_sizes.clone(),
                    });
                };
                let [layer] = structure.layers.as_slice() else {
                    return Ok(None);
                };
                let Some(pb23::sparse_structural_layer::Layer::Validity(validity)) = &layer.layer
                else {
                    return Ok(None);
                };
                let encoding = validity
                    .positions
                    .as_ref()
                    .ok_or_else(|| corrupt("missing validity positions descriptor"))?;
                // Validate the unsigned codec grammar even for an empty position set.
                let sequence = SequenceDecoder::try_new(encoding, 64, validity.num_positions)?;
                let is_all_null = match validity.meaning() {
                    pb23::sparse_validity_set::Meaning::SparseValidityValidPositions => {
                        validity.num_positions == 0
                            && sequence.metadata() == Some(SequenceMetadata::Empty)
                    }
                    pb23::sparse_validity_set::Meaning::SparseValidityNullPositions => {
                        if validity.num_positions == page.num_rows && sequence.requires_payload() {
                            return Err(Error::not_supported(
                                "admitted all-null validity requires metadata-only positions",
                            ));
                        }
                        validity.num_positions == page.num_rows
                            && matches!(
                                (page.num_rows, sequence.metadata()),
                                (0, Some(SequenceMetadata::Empty))
                                    | (1, Some(SequenceMetadata::Constant(0)))
                                    | (2.., Some(SequenceMetadata::Range { start: 0, step: 1 }))
                            )
                    }
                    _ => return Err(corrupt("invalid validity meaning")),
                };
                if !is_all_null || sequence.requires_payload() {
                    return Err(corrupt("no-values flat page does not prove all rows null"));
                }
                if !page.buffer_offsets_and_sizes.is_empty() {
                    return Err(corrupt("metadata-only null page has unexpected buffers"));
                }
            }
            _ => return Ok(None),
        }
        Ok(Some(Self {
            rows: page.num_rows,
        }))
    }

    pub(super) fn validate_array_type(&self, data_type: &DataType) -> Result<()> {
        if matches!(
            data_type,
            DataType::Binary | DataType::LargeBinary | DataType::Utf8 | DataType::LargeUtf8
        ) {
            Ok(())
        } else {
            Err(Error::not_supported(
                "admitted metadata-only null page requires a flat byte/string type",
            ))
        }
    }

    pub(super) fn prepare(&self, rows: Range<u64>, data_type: DataType) -> Result<AllNullInput> {
        self.validate_array_type(&data_type)?;
        if rows.start >= rows.end || rows.end > self.rows {
            return Err(Error::invalid_input(format!(
                "all-null range {rows:?} is outside {} rows or empty",
                self.rows
            )));
        }
        Ok(AllNullInput {
            rows: rows.end - rows.start,
            data_type,
        })
    }
}

#[derive(Debug)]
pub(super) struct AllNullInput {
    rows: u64,
    data_type: DataType,
}

impl AllNullInput {
    pub(super) fn decode_in(self, workspace: &mut Reservation) -> Result<Option<ArrayRef>> {
        // No fabricated O(rows) definition stream. Offsets, validity and Arrow
        // validation scratch all use the existing whole-array admission plan.
        DataBlock::AllNull(AllNullDataBlock {
            num_values: self.rows,
        })
        .try_to_arrow_in(&self.data_type, workspace)
        .map(|array| array.map(make_array))
    }
}

#[cfg(test)]
mod tests;
