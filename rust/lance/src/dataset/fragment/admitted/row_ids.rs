// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Sparse generated identifiers keep their output credit beside payload inputs.

use arrow_array::{ArrayRef, UInt32Array, UInt64Array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use lance_core::{
    Error, ROW_ADDR, ROW_ID, Result,
    utils::{address::RowAddress, allocation::Reservation, deletion::DeletionVector},
};
use lance_table::{format::RowIdMeta, rowids::RowIdSequence};
use std::sync::Arc;

use super::FragmentReader;

pub(super) struct PreparedRowIds {
    fragment_id: u32,
    id_column: Option<usize>,
    address_column: Option<usize>,
    sequence: Option<Arc<RowIdSequence>>,
    deleted: Option<Arc<DeletionVector>>,
    credit: Reservation,
}

impl PreparedRowIds {
    pub(super) async fn try_prepare(
        reader: &FragmentReader,
        indices: &UInt32Array,
        workspace: &mut Reservation,
    ) -> Result<Option<Self>> {
        let id_column = reader
            .with_row_id
            .then(|| reader.output_schema.index_of(ROW_ID))
            .transpose()?;
        let address_column = reader
            .with_row_addr
            .then(|| reader.output_schema.index_of(ROW_ADDR))
            .transpose()?;
        let has_ids = id_column.is_some() || address_column.is_some();
        let fragment_id = if has_ids {
            u32::try_from(reader.fragment_id)
                .map_err(|_| Error::invalid_input("fragment ID exceeds the row-address domain"))?
        } else {
            0
        };
        let stable = id_column.is_some() && reader.fragment.dataset.manifest.uses_stable_row_ids();
        if stable
            && matches!(
                reader.fragment.metadata.row_id_meta,
                Some(RowIdMeta::Column)
            )
        {
            // Loading a spilled sequence currently materializes its entire data
            // column. Do not disguise that ordinary read as shared metadata.
            return Err(Error::not_supported(
                "admitted fragment row IDs do not yet read spilled row-ID columns",
            ));
        }
        let count = indices.len();
        let deleted = reader
            .deletion_vec
            .as_ref()
            .filter(|deleted| {
                reader.make_deletions_null
                    && indices.values().iter().any(|row| deleted.contains(*row))
            })
            .cloned();
        let buffers = usize::from(has_ids) + usize::from(stable && address_column.is_some());
        let bytes = count
            .checked_mul(size_of::<u64>())
            .and_then(|bytes| bytes.checked_mul(buffers))
            .and_then(|bytes| {
                bytes.checked_add(if deleted.is_some() {
                    count.div_ceil(8)
                } else {
                    0
                })
            })
            .ok_or_else(|| {
                Error::invalid_input("generated fragment identifiers exceed address space")
            })?;
        let Some(credit) = workspace.try_split(bytes) else {
            return Ok(None);
        };
        // Lazy snapshot loading stays outside the CPU closure. Only inline
        // sequence metadata is allowed; output is never collected into a Vec.
        let sequence = reader.row_id_sequence().await?;
        if let Some(sequence) = &sequence
            && sequence.len() != reader.num_physical_rows as u64
        {
            return Err(Error::invalid_input(format!(
                "row ID sequence length {} differs from fragment physical length {}",
                sequence.len(),
                reader.num_physical_rows
            )));
        }
        Ok(Some(Self {
            fragment_id,
            id_column,
            address_column,
            sequence,
            deleted,
            credit,
        }))
    }

    pub(super) fn mark_columns(&self, covered: &mut [bool]) {
        for column in [self.id_column, self.address_column].into_iter().flatten() {
            covered[column] = true;
        }
    }

    pub(super) fn fill_columns(
        mut self,
        indices: &UInt32Array,
        columns: &mut [Option<ArrayRef>],
    ) -> Result<()> {
        let nulls = self.deletion_nulls(indices)?;
        let needs_addresses =
            self.address_column.is_some() || (self.id_column.is_some() && self.sequence.is_none());
        let addresses = if needs_addresses {
            Some(row_array(
                &mut self.credit,
                indices.len(),
                indices
                    .values()
                    .iter()
                    .map(|&offset| u64::from(RowAddress::new_from_parts(self.fragment_id, offset))),
                nulls.clone(),
            )?)
        } else {
            None
        };
        if let Some(column) = self.id_column {
            columns[column] = Some(if let Some(sequence) = &self.sequence {
                // The original selection was validated before deletion pruning;
                // select's sorted-input requirement and bounds are established.
                // Cardinality is also checked below, never silently truncated.
                row_array(
                    &mut self.credit,
                    indices.len(),
                    sequence.select(indices.values().iter().map(|&offset| offset as usize)),
                    nulls,
                )?
            } else {
                addresses
                    .clone()
                    .ok_or_else(|| Error::internal("missing generated row addresses"))?
            });
        }
        if let Some(column) = self.address_column {
            columns[column] = addresses;
        }
        Ok(())
    }

    fn deletion_nulls(&mut self, indices: &UInt32Array) -> Result<Option<NullBuffer>> {
        let Some(deleted) = &self.deleted else {
            return Ok(None);
        };
        let mut mask = self
            .credit
            .try_allocate(indices.len().div_ceil(8))?
            .ok_or_else(|| Error::internal("missing pre-admitted deletion marker credit"))?;
        mask.as_mut().fill(0xff);
        for (index, row) in indices.values().iter().enumerate() {
            if deleted.contains(*row) {
                mask.as_mut()[index / 8] &= !(1 << (index % 8));
            }
        }
        // Both identifier columns share this exact owner. No Arrow builder or
        // predicate array introduces another bitmap while payload is live.
        Ok(Some(NullBuffer::new(BooleanBuffer::new(
            Buffer::from(mask.freeze()),
            0,
            indices.len(),
        ))))
    }
}

fn row_array(
    credit: &mut Reservation,
    count: usize,
    mut values: impl Iterator<Item = u64>,
    nulls: Option<NullBuffer>,
) -> Result<ArrayRef> {
    // count * 8 was checked and reserved before any dependent payload read.
    let mut buffer = credit
        .try_allocate(count * size_of::<u64>())?
        .ok_or_else(|| Error::internal("missing pre-admitted fragment row ID credit"))?;
    for slot in buffer.as_mut().as_chunks_mut::<{ size_of::<u64>() }>().0 {
        let value = values.next().ok_or_else(|| {
            Error::invalid_input("row ID sequence ended before physical selection")
        })?;
        slot.copy_from_slice(&value.to_ne_bytes());
    }
    if values.next().is_some() {
        return Err(Error::invalid_input(
            "row ID sequence exceeded physical selection",
        ));
    }
    Ok(Arc::new(UInt64Array::new(
        ScalarBuffer::new(Buffer::from(buffer.freeze()), 0, count),
        nulls,
    )))
}
