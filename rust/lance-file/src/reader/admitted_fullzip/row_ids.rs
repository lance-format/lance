// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Generated row columns share the batch admission and retain their own charges.

use std::{fmt::Debug, ops::Range, sync::Arc};

use arrow_array::{ArrayRef, UInt64Array};
use arrow_buffer::{Buffer, ScalarBuffer};
use arrow_schema::Schema;
use lance_core::{
    Error, ROW_ADDR_FIELD, ROW_ID_FIELD, Result,
    utils::{address::RowAddress, allocation::Reservation},
};

use super::FullZipScan;

/// Immutable mapping from physical file-row positions to stable row IDs.
///
/// Implement this in the layer that owns row metadata. The file reader does not
/// depend on table serialization or row ID encodings. Values are streamed into
/// admitted output; implementations must use bounded iterator state over their
/// shared metadata, without collecting an output array or blocking for I/O/credit.
/// The mapping and physical length must remain unchanged for the source lifetime.
///
/// ```
/// use std::sync::Arc;
/// use lance_core::Result;
/// use lance_file::reader::{FullZipScan, PhysicalRowIdSource};
/// fn attach(scan: FullZipScan, source: Arc<dyn PhysicalRowIdSource>) -> Result<FullZipScan> {
///     scan.with_row_id(7, Some(source))
/// }
/// ```
pub trait PhysicalRowIdSource: Debug + Send + Sync {
    /// Physical rows covered, including deleted positions.
    fn num_rows(&self) -> u64;

    /// Return exactly one ID per physical position, in the requested order.
    /// Invalid or reversed bounds must return an error, never silently clip.
    /// The file reader also checks iterator cardinality before delivering a batch.
    fn select_range(&self, rows: Range<usize>) -> Result<Box<dyn Iterator<Item = u64> + '_>>;
}

#[derive(Debug, Clone)]
pub(super) struct RowIdColumns {
    fragment_id: u32,
    sequence: Option<Arc<dyn PhysicalRowIdSource>>,
    has_row_id: bool,
    has_row_address: bool,
}

pub(super) struct PreparedRowIds {
    columns: RowIdColumns,
    rows: Range<usize>,
    credit: Reservation,
}

impl FullZipScan {
    /// Append `_rowid` using a physical-offset-to-stable-ID snapshot, or row
    /// addresses when `sequence` is `None` (the non-stable row ID convention).
    ///
    /// Configure after merging physical files and before any read attempt or
    /// logical range selection. The sequence must cover every physical file row,
    /// including deleted rows. The caller establishes that it describes these
    /// files; equal lengths alone do not establish identity. Snapshot metadata is
    /// shared and outside the payload budget. Generated values are admitted with
    /// each batch, and Arrow aliases retain that charge until their final drop.
    ///
    /// This does not implement make-deletions-null, version columns, overlays or
    /// ordinary fragment/scanner adoption. Rows excluded before decoding have no
    /// output ID. Forward live-range selections still use physical offsets here.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use lance_core::Result;
    /// use lance_file::reader::{FullZipScan, PhysicalRowIdSource};
    /// fn attach(scan: FullZipScan, ids: Arc<dyn PhysicalRowIdSource>) -> Result<FullZipScan> {
    ///     scan.with_row_id(7, Some(ids))?.with_row_address(7)
    /// }
    /// ```
    pub fn with_row_id(
        mut self,
        fragment_id: u32,
        sequence: Option<Arc<dyn PhysicalRowIdSource>>,
    ) -> Result<Self> {
        self.validate_row_columns(fragment_id)?;
        if self.row_ids.as_ref().is_some_and(|ids| ids.has_row_id) {
            return Err(Error::invalid_input("FullZip row ID is already configured"));
        }
        if let Some(sequence) = &sequence
            && sequence.num_rows() != self.read_len
        {
            return Err(Error::invalid_input(format!(
                "FullZip row ID sequence length {} differs from physical length {}",
                sequence.num_rows(),
                self.read_len
            )));
        }
        let ids = self
            .row_ids
            .get_or_insert_with(|| RowIdColumns::new(fragment_id));
        ids.has_row_id = true;
        ids.sequence = sequence;
        self.update_row_column_schema()?;
        Ok(self)
    }

    /// Append `_rowaddr` from this fragment ID and each physical file-row offset.
    ///
    /// The same configuration and lifetime rules as [`Self::with_row_id`] apply.
    /// When both columns use addresses, they share one allocation and charge.
    /// Fields are appended in `_rowid`, `_rowaddr` order regardless of builder order.
    ///
    /// ```
    /// use lance_core::Result;
    /// use lance_file::reader::FullZipScan;
    /// fn attach(scan: FullZipScan) -> Result<FullZipScan> {
    ///     scan.with_row_address(7)
    /// }
    /// ```
    pub fn with_row_address(mut self, fragment_id: u32) -> Result<Self> {
        self.validate_row_columns(fragment_id)?;
        let ids = self
            .row_ids
            .get_or_insert_with(|| RowIdColumns::new(fragment_id));
        if ids.has_row_address {
            return Err(Error::invalid_input(
                "FullZip row address is already configured",
            ));
        }
        ids.has_row_address = true;
        self.update_row_column_schema()?;
        Ok(self)
    }

    fn validate_row_columns(&self, fragment_id: u32) -> Result<()> {
        if self.has_started {
            return Err(Error::invalid_input(
                "FullZip row columns must be configured before the first read or logical selection",
            ));
        }
        if fragment_id == RowAddress::TOMBSTONE_FRAG
            || self.read_len > RowAddress::FRAGMENT_SIZE
            || usize::try_from(self.read_len).is_err()
        {
            return Err(Error::invalid_input(
                "FullZip row columns require a valid fragment ID and addressable physical offsets",
            ));
        }
        if self
            .row_ids
            .as_ref()
            .is_some_and(|ids| ids.fragment_id != fragment_id)
        {
            return Err(Error::invalid_input(
                "FullZip row columns require the same fragment ID",
            ));
        }
        Ok(())
    }

    fn update_row_column_schema(&mut self) -> Result<()> {
        let mut fields = self.schema.fields()[..self.columns.len()].to_vec();
        if let Some(ids) = &self.row_ids {
            for (included, field) in [
                (ids.has_row_id, &*ROW_ID_FIELD),
                (ids.has_row_address, &*ROW_ADDR_FIELD),
            ] {
                if included {
                    if fields
                        .iter()
                        .any(|existing| existing.name() == field.name())
                    {
                        return Err(Error::invalid_input(format!(
                            "FullZip generated field conflicts with projected field {}",
                            field.name()
                        )));
                    }
                    fields.push(Arc::new(field.clone()));
                }
            }
        }
        self.schema = Arc::new(Schema::new_with_metadata(
            fields,
            self.schema.metadata().clone(),
        ));
        Ok(())
    }
}

impl RowIdColumns {
    fn new(fragment_id: u32) -> Self {
        Self {
            fragment_id,
            sequence: None,
            has_row_id: false,
            has_row_address: false,
        }
    }

    pub(super) fn prepare(
        &self,
        rows: Range<u64>,
        workspace: &mut Reservation,
    ) -> Result<Option<PreparedRowIds>> {
        // Configuration validated all offsets against both u32 and usize domains.
        let rows = rows.start as usize..rows.end as usize;
        let buffers = 1 + usize::from(self.has_row_address && self.sequence.is_some());
        let bytes = rows
            .len()
            .checked_mul(8)
            .and_then(|bytes| bytes.checked_mul(buffers))
            .ok_or_else(|| Error::invalid_input("FullZip row column allocation overflows"))?;
        Ok(workspace.try_split(bytes).map(|credit| PreparedRowIds {
            columns: self.clone(),
            rows,
            credit,
        }))
    }
}

impl PreparedRowIds {
    pub(super) fn append_to(mut self, arrays: &mut Vec<ArrayRef>) -> Result<()> {
        let address_needed = self.columns.has_row_address
            || (self.columns.has_row_id && self.columns.sequence.is_none());
        let addresses = if address_needed {
            Some(row_array(
                &mut self.credit,
                self.rows.len(),
                self.rows.clone().map(|offset| {
                    u64::from(RowAddress::new_from_parts(
                        self.columns.fragment_id,
                        offset as u32,
                    ))
                }),
            )?)
        } else {
            None
        };
        if self.columns.has_row_id {
            let array = if let Some(sequence) = &self.columns.sequence {
                row_array(
                    &mut self.credit,
                    self.rows.len(),
                    sequence.select_range(self.rows.clone())?,
                )?
            } else {
                addresses
                    .clone()
                    .ok_or_else(|| Error::internal("missing admitted row addresses"))?
            };
            arrays.push(array);
        }
        if self.columns.has_row_address {
            arrays
                .push(addresses.ok_or_else(|| Error::internal("missing admitted row addresses"))?);
        }
        Ok(())
    }
}

fn row_array(
    credit: &mut Reservation,
    count: usize,
    mut values: impl Iterator<Item = u64>,
) -> Result<ArrayRef> {
    let mut buffer = credit
        .try_allocate(count * 8)?
        .ok_or_else(|| Error::internal("missing pre-admitted row column credit"))?;
    for slot in buffer.as_mut().chunks_exact_mut(8) {
        let value = values.next().ok_or_else(|| {
            Error::invalid_input("row ID sequence ended before physical selection")
        })?;
        slot.copy_from_slice(&value.to_ne_bytes());
    }
    if values.next().is_some() {
        return Err(Error::invalid_input(
            "row ID source exceeded physical selection",
        ));
    }
    let values = ScalarBuffer::new(Buffer::from(buffer.freeze()), 0, count);
    Ok(Arc::new(UInt64Array::new(values, None)))
}
