// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Selected sparse List structure with null byte leaves, under one admission.

use std::{future::Future, ops::Range, sync::Arc};

use arrow_array::{ArrayRef, make_array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::{
    SparsePositionDecoder, SparsePositionPlan, SparseValidityMeaning, SparseValiditySet,
    position_decoder, validity_set_decoder,
};
use crate::{
    data::{AllNullDataBlock, DataBlock},
    decoder::{PageEncoding, PageInfo},
    format::pb23,
};

fn corrupt(message: &str) -> Error {
    Error::corrupt_file_named("admitted sparse List", message)
}

fn ready_positions(decoder: SparsePositionDecoder) -> Result<SparsePositionPlan> {
    match decoder {
        SparsePositionDecoder::Ready(positions) => Ok(positions),
        SparsePositionDecoder::Encoded { .. } => Err(Error::not_supported(
            "admitted sparse List requires metadata-only positions",
        )),
    }
}

fn validity(
    descriptor: Option<&pb23::SparseValiditySet>,
    domain: u64,
) -> Result<SparseValiditySet> {
    let descriptor = descriptor.ok_or_else(|| corrupt("missing validity"))?;
    let (decoder, _) = validity_set_decoder(descriptor, domain, "admitted List validity")?;
    Ok(SparseValiditySet {
        meaning: decoder.meaning,
        positions: ready_positions(decoder.positions)?,
    })
}

fn null_count(validity: &SparseValiditySet, rows: &Range<u64>) -> u64 {
    let selected = validity.positions.rank(rows.end) - validity.positions.rank(rows.start);
    match validity.meaning {
        SparseValidityMeaning::NullPositions => selected,
        SparseValidityMeaning::ValidPositions => rows.end - rows.start - selected,
    }
}

mod counts;
use counts::Counts;
mod positions;
use positions::Positions;

#[derive(Clone, Debug)]
struct ListMetadata {
    non_empty: Positions,
    counts: Counts,
    child_slots: u64,
    validity: SparseValiditySet,
}

impl ListMetadata {
    fn child_before(&self, row: u64) -> Result<u64> {
        self.counts.prefix(self.non_empty.rank(row)?)
    }
}

#[derive(Debug)]
pub(in crate::encodings::logical::primitive) struct NullListScheduler {
    rows: u64,
    lists: Arc<[ListMetadata]>,
}

impl NullListScheduler {
    pub(in crate::encodings::logical::primitive) fn try_from_page(
        page: &PageInfo,
    ) -> Result<Option<Self>> {
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
        if !matches!(
            structure.layers.first().and_then(|l| l.layer.as_ref()),
            Some(pb23::sparse_structural_layer::Layer::List(_))
        ) {
            return Ok(None);
        }
        let mut domain = page.num_rows;
        let mut buffers = page.buffer_offsets_and_sizes.iter();
        let mut lists = Vec::with_capacity(structure.layers.len());
        let mut has_leaf = false;
        for layer in &structure.layers {
            match &layer.layer {
                Some(pb23::sparse_structural_layer::Layer::List(list)) if !has_leaf => {
                    let positions = list
                        .non_empty_positions
                        .as_ref()
                        .ok_or_else(|| corrupt("missing nonempty positions descriptor"))?;
                    let non_empty = Positions::parse(
                        positions,
                        list.num_non_empty_positions,
                        domain,
                        &mut buffers,
                    )?;
                    let counts = Counts::parse(
                        list.counts.as_ref(),
                        list.num_non_empty_positions,
                        &mut buffers,
                    )?;
                    if counts
                        .metadata_total()?
                        .is_some_and(|total| total != list.num_child_slots)
                    {
                        return Err(corrupt("List count sum differs from child domain"));
                    }
                    let validity = validity(list.validity.as_ref(), domain)?;
                    // Reuse the sparse reader's checked set containment/intersection,
                    // including its arithmetic-set path, without materializing rows.
                    if let Positions::Metadata(plan) = &non_empty
                        && !validity.contains_only_valid_positions(plan, domain)?
                    {
                        return Err(corrupt("null List parent contains child slots"));
                    }
                    lists.push(ListMetadata {
                        non_empty,
                        counts,
                        child_slots: list.num_child_slots,
                        validity,
                    });
                    domain = list.num_child_slots;
                }
                Some(pb23::sparse_structural_layer::Layer::Validity(leaf)) if !has_leaf => {
                    if null_count(&validity(Some(leaf), domain)?, &(0..domain)) != domain {
                        return Err(corrupt(
                            "no-values List leaf does not prove all children null",
                        ));
                    }
                    has_leaf = true;
                }
                _ => {
                    return Err(Error::not_supported(
                        "admitted sparse List requires a List chain and one byte leaf",
                    ));
                }
            }
        }
        if !has_leaf {
            return Err(corrupt("missing leaf validity"));
        }
        if buffers.next().is_some() {
            return Err(corrupt("sparse List has unexpected buffers"));
        }
        Ok(Some(Self {
            rows: page.num_rows,
            lists: lists.into(),
        }))
    }

    pub(in crate::encodings::logical::primitive) fn validate_array_type(
        &self,
        data_type: &DataType,
    ) -> Result<()> {
        let mut current = data_type;
        for _ in self.lists.iter() {
            let (DataType::List(field) | DataType::LargeList(field)) = current else {
                return Err(Error::not_supported(
                    "sparse List Arrow depth differs from stored structure",
                ));
            };
            current = field.data_type();
        }
        if !matches!(
            current,
            DataType::Binary | DataType::Utf8 | DataType::LargeBinary | DataType::LargeUtf8
        ) {
            return Err(Error::not_supported(
                "admitted sparse List requires a byte/string leaf",
            ));
        }
        Ok(())
    }

    pub(in crate::encodings::logical::primitive) async fn prepare<F, Fut>(
        &self,
        rows: Range<u64>,
        data_type: DataType,
        work: &mut Reservation,
        mut read: F,
    ) -> Result<Option<NullListInput>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        self.validate_array_type(&data_type)?;
        if rows.start >= rows.end || rows.end > self.rows {
            return Err(Error::invalid_input(
                "sparse List selection is outside the page or empty",
            ));
        }
        let mut selected = rows.clone();
        let mut lists = Vec::with_capacity(self.lists.len());
        for layer in self.lists.iter() {
            let Some(non_empty) = layer
                .non_empty
                .prepare(selected.clone(), &layer.validity, work, &mut read)
                .await?
            else {
                return Ok(None);
            };
            let indices = non_empty.rank(selected.start)?..non_empty.rank(selected.end)?;
            let Some(counts) = layer
                .counts
                .prepare(indices, layer.child_slots, work, &mut read)
                .await?
            else {
                return Ok(None);
            };
            let layer = ListMetadata {
                non_empty,
                counts,
                ..layer.clone()
            };
            selected = layer.child_before(selected.start)?..layer.child_before(selected.end)?;
            lists.push(layer);
        }
        Ok(Some(NullListInput {
            rows,
            data_type,
            lists: lists.into(),
        }))
    }
}

struct LayerOutput<'a> {
    metadata: &'a ListMetadata,
    data_type: &'a DataType,
    rows: Range<u64>,
    child_start: u64,
    len: usize,
    width: usize,
    offsets: usize,
    bitmap: usize,
}

fn offset_bytes(rows: u64, width: usize) -> Result<usize> {
    usize::try_from(rows)
        .ok()
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_mul(width))
        .ok_or_else(|| Error::invalid_input("List offsets size exceeds addressable allocation"))
}

#[derive(Debug)]
pub(in crate::encodings::logical::primitive) struct NullListInput {
    rows: Range<u64>,
    data_type: DataType,
    lists: Arc<[ListMetadata]>,
}

impl NullListInput {
    pub(in crate::encodings::logical::primitive) fn decode_in(
        self,
        workspace: &mut Reservation,
    ) -> Result<Option<ArrayRef>> {
        let mut rows = self.rows;
        let mut current_type = &self.data_type;
        let mut plan = Vec::with_capacity(self.lists.len());
        let mut total = 0_usize;
        for metadata in self.lists.iter() {
            let (width, field, max_offset) = match current_type {
                DataType::List(field) => (4, field, i32::MAX as u64),
                DataType::LargeList(field) => (8, field, i64::MAX as u64),
                _ => return Err(Error::invalid_input("expected admitted List shape")),
            };
            let children = metadata.child_before(rows.start)?..metadata.child_before(rows.end)?;
            if children.end - children.start > max_offset {
                return Err(Error::invalid_input(
                    "selected List children exceed Arrow offset width",
                ));
            }
            let count = rows.end - rows.start;
            let offsets = offset_bytes(count, width)?;
            let len = usize::try_from(count)
                .map_err(|_| Error::invalid_input("List length exceeds usize"))?;
            let bitmap = if null_count(&metadata.validity, &rows) == 0 {
                0
            } else {
                len.div_ceil(8)
            };
            total = total
                .checked_add(offsets)
                .and_then(|n| n.checked_add(bitmap))
                .ok_or_else(|| Error::invalid_input("List output size overflows"))?;
            plan.push(LayerOutput {
                metadata,
                data_type: current_type,
                rows,
                child_start: children.start,
                len,
                width,
                offsets,
                bitmap,
            });
            current_type = field.data_type();
            rows = children;
        }
        let leaf_rows = rows.end - rows.start;
        let leaf_width = if matches!(current_type, DataType::Binary | DataType::Utf8) {
            4
        } else {
            8
        };
        let leaf_bitmap = usize::try_from(leaf_rows.div_ceil(8))
            .map_err(|_| Error::invalid_input("List leaf bitmap exceeds usize"))?;
        total = total
            .checked_add(offset_bytes(leaf_rows, leaf_width)?)
            .and_then(|n| n.checked_add(leaf_bitmap))
            .ok_or_else(|| Error::invalid_input("List output size overflows"))?;
        if total > workspace.len() {
            return Ok(None);
        }
        let mut child = DataBlock::AllNull(AllNullDataBlock {
            num_values: leaf_rows,
        })
        .try_to_arrow_in(current_type, workspace)?
        .ok_or_else(|| Error::internal("List preflight underestimated null leaf"))?;
        for layer in plan.into_iter().rev() {
            let mut allocation = workspace
                .try_allocate(layer.offsets + layer.bitmap)?
                .ok_or_else(|| Error::internal("List preflight underestimated parent"))?;
            allocation.as_mut().fill(0);
            for (index, row) in (layer.rows.start..=layer.rows.end).enumerate() {
                let offset = layer.metadata.child_before(row)? - layer.child_start;
                let target =
                    &mut allocation.as_mut()[index * layer.width..(index + 1) * layer.width];
                // The complete selected child domain was checked against this width.
                if layer.width == 4 {
                    target.copy_from_slice(&(offset as i32).to_ne_bytes());
                } else {
                    target.copy_from_slice(&(offset as i64).to_ne_bytes());
                }
            }
            if layer.bitmap > 0 {
                let null_positions = matches!(
                    layer.metadata.validity.meaning,
                    SparseValidityMeaning::NullPositions
                );
                for (index, row) in layer.rows.clone().enumerate() {
                    if layer.metadata.validity.positions.contains(row) != null_positions {
                        allocation.as_mut()[layer.offsets + index / 8] |= 1 << (index % 8);
                    }
                }
            }
            let buffer = Buffer::from(allocation.freeze());
            let nulls = (layer.bitmap > 0).then(|| {
                NullBuffer::new(BooleanBuffer::new(
                    buffer.slice_with_length(layer.offsets, layer.bitmap),
                    0,
                    layer.len,
                ))
            });
            child = ArrayData::builder(layer.data_type.clone())
                .len(layer.len)
                .add_buffer(buffer.slice_with_length(0, layer.offsets))
                .nulls(nulls)
                .add_child_data(child)
                .build()?;
        }
        Ok(Some(make_array(child)))
    }
}

#[cfg(test)]
mod tests;
