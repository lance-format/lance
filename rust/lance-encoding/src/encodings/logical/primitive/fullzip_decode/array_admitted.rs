// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keep structural reconstruction inside the read/decode admission envelope.

use std::{future::Future, ops::Range, sync::Arc};

use arrow_array::{ArrayRef, LargeListArray, ListArray, make_array};
use arrow_schema::{DataType, FieldRef};
use lance_core::{
    Error, Result,
    utils::allocation::{AllocationBudget, Reservation, ReservedBuffer},
};

use super::FullZipScheduler;
use crate::{decoder::DecodedPage, repdef::CompositeRepDefUnraveler};

/// A complete Arrow array for an exact, nonempty page-local top-level row prefix.
pub struct FullZipArrayPrefix {
    /// Advance the caller's page cursor only to this end, never the requested end.
    pub rows: Range<u64>,
    /// All owned value, offset and validity payloads keep their shared-budget charge.
    pub array: ArrayRef,
}

enum ListLayer<'a> {
    Small(&'a FieldRef),
    Large(&'a FieldRef),
}

struct Shape<'a> {
    leaf: &'a DataType,
    // Outer to inner, rebuilt in reverse after leaf validity is applied.
    lists: Vec<ListLayer<'a>>,
}

impl<'a> Shape<'a> {
    fn new(scheduler: &FullZipScheduler, data_type: &'a DataType) -> Result<Self> {
        let mut leaf = data_type;
        let mut lists = Vec::new();
        loop {
            match leaf {
                DataType::List(field) => {
                    lists.push(ListLayer::Small(field));
                    leaf = field.data_type();
                }
                DataType::LargeList(field) => {
                    lists.push(ListLayer::Large(field));
                    leaf = field.data_type();
                }
                _ => break,
            }
        }
        let offset_bits = match leaf {
            DataType::Binary | DataType::Utf8 => 32,
            DataType::LargeBinary | DataType::LargeUtf8 => 64,
            _ => {
                return Err(Error::not_supported(
                    "admitted FullZip arrays require binary/string values with optional List/LargeList nesting",
                ));
            }
        };
        let layers = &scheduler.details.def_meaning;
        if offset_bits != scheduler.bits_per_offset
            || layers.len() != lists.len() + 1
            || layers.first().is_none_or(|layer| layer.is_list())
            || layers.iter().skip(1).any(|layer| !layer.is_list())
        {
            return Err(Error::invalid_input(
                "FullZip Arrow type does not match page offset width or structural layers",
            ));
        }
        Ok(Self { leaf, lists })
    }

    fn materialize(
        &self,
        page: DecodedPage,
        workspace: &mut Reservation,
    ) -> Result<Option<ArrayRef>> {
        let Some(leaf) = page.data.try_to_arrow_in(self.leaf, workspace)? else {
            return Ok(None);
        };
        let mut levels = CompositeRepDefUnraveler::new(vec![page.repdef]);
        let Some(validity) = levels.try_unravel_validity_in(leaf.len(), workspace)? else {
            return Ok(None);
        };
        // Only supported binary/string leaves reach this path. Their pinned Arrow
        // validation and these List constructors borrow buffers without payload
        // scratch. FSL/struct reconstruction has different overlap requirements.
        let mut array = make_array(leaf.into_builder().nulls(validity).build()?);
        for layer in self.lists.iter().rev() {
            array = match layer {
                ListLayer::Small(field) => {
                    let Some((offsets, validity)) =
                        levels.try_unravel_offsets_in::<i32>(workspace)?
                    else {
                        return Ok(None);
                    };
                    Arc::new(ListArray::try_new(
                        (*field).clone(),
                        offsets,
                        array,
                        validity,
                    )?)
                }
                ListLayer::Large(field) => {
                    let Some((offsets, validity)) =
                        levels.try_unravel_offsets_in::<i64>(workspace)?
                    else {
                        return Ok(None);
                    };
                    Arc::new(LargeListArray::try_new(
                        (*field).clone(),
                        offsets,
                        array,
                        validity,
                    )?)
                }
            };
        }
        Ok(Some(array))
    }
}

impl FullZipScheduler {
    /// Validate the Arrow shape and structural unzip metadata without I/O.
    /// This does not validate payload framing or data-dependent codec support.
    /// File-level cursors use this before returning any batch from a selection.
    ///
    /// ```
    /// use arrow_schema::DataType;
    /// use lance_core::Result;
    /// use lance_encoding::encodings::logical::primitive::FullZipScheduler;
    /// fn validate(decoder: &FullZipScheduler, ty: &DataType) -> Result<()> {
    ///     decoder.validate_array_type(ty)
    /// }
    /// ```
    pub fn validate_array_type(&self, data_type: &DataType) -> Result<()> {
        Shape::new(self, data_type)?;
        self.plan_unzip_range(&[], 0, 0..0)?;
        Ok(())
    }

    /// Read a fitting prefix and finish its Arrow reconstruction in one admission.
    ///
    /// Supports Binary/Utf8/LargeBinary/LargeUtf8 leaves nested in List/LargeList.
    /// The supplied type must describe the whole raw leaf path, not a projection
    /// or a scalarized list. Unsupported/mismatched shapes fail before I/O. Reader
    /// projections, filters, struct/multi-column alignment and CPU dispatch are
    /// not implemented by this low-level API; ordinary scans are unchanged.
    ///
    /// Input, unzip, codec, rep/def, typed values, validity and list offsets share
    /// the same `workspace_bytes` reservation. Reconstruction refusal drops all
    /// intermediate owners before retrying a smaller prefix. It never reacquires
    /// credit while holding intermediate data. `None` is refusal, not EOF; keep the
    /// cursor and drain/replan. Errors are propagated without prefix retry.
    /// Metadata, transport/cache and allocator overhead remain outside accounting.
    /// Repeated reads and inline CPU have not been performance qualified.
    ///
    /// ```
    /// use std::{ops::Range, sync::Arc};
    /// use arrow_schema::DataType;
    /// use lance_core::{Result, utils::allocation::AllocationBudget};
    /// use lance_encoding::{EncodingsIo,
    ///     encodings::logical::primitive::{FullZipArrayPrefix, FullZipScheduler}};
    /// async fn next(scheduler: &FullZipScheduler, io: &dyn EncodingsIo,
    ///     rows: Range<u64>, ty: &DataType, budget: &Arc<AllocationBudget>,
    ///     priority: u64) -> Result<Option<FullZipArrayPrefix>> {
    ///     scheduler.try_read_array_prefix_in(rows, ty, budget, 65536,
    ///         |range, output| io.submit_single_into(range, priority, output)).await
    /// }
    /// ```
    pub async fn try_read_array_prefix_in<F, Fut>(
        &self,
        rows: Range<u64>,
        data_type: &DataType,
        budget: &Arc<AllocationBudget>,
        workspace_bytes: usize,
        read: F,
    ) -> Result<Option<FullZipArrayPrefix>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        let shape = Shape::new(self, data_type)?;
        let Some((rows, array)) = self
            .try_read_prefix_in(rows, budget, workspace_bytes, read, |page, workspace| {
                shape.materialize(page, workspace)
            })
            .await?
        else {
            return Ok(None);
        };
        if array.len() as u64 != rows.end - rows.start {
            return Err(Error::corrupt_file_named(
                "FullZip",
                "reconstructed Arrow row count differs from selected page rows",
            ));
        }
        Ok(Some(FullZipArrayPrefix { rows, array }))
    }
}
