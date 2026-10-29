// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::Field;
use lance_arrow::FieldExt;
use lance_core::datatypes::{BlobKind, BlobV2Layout, Field as LanceField};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};

use super::locate::{BlobV2DescriptorColumns, BlobV2ReadContext};
use crate::Dataset;
use crate::blob::reference::{BlobRefBatch, BlobSourceContext, NestedBlobRefBatch};

mod nested;

/// Capture portable references from a Blob V2 descriptor scan.
///
/// Create one session per source snapshot and field. It caches fragment, object
/// store, and registered-base resolution across batches. Capture neither reads
/// payloads nor looks up descriptors again. External whole-object lengths remain
/// unknown until a reader resolves them; capture performs no object HEAD requests.
///
/// The caller must retain the source snapshot through all deferred reads and
/// retries. This session does not acquire a cleanup protection or retention lease.
///
/// ```
/// # use std::sync::Arc;
/// # use arrow_array::StructArray;
/// # use arrow_schema::Field;
/// # use lance::{Dataset, Result};
/// # use lance::blob::reference::{BlobRefCapture, BlobRefBatch, BlobSourceContext};
/// # use lance_core::utils::address::RowAddress;
/// # async fn example(dataset: Arc<Dataset>, field_id: i32, field: &Field,
/// #     addresses: &[RowAddress], descriptors: &StructArray) -> Result<BlobRefBatch> {
/// let source = BlobSourceContext::from_dataset(
///     &dataset, "catalog/table-incarnation", "query/storage-binding",
/// ).await?;
/// let mut capture = BlobRefCapture::try_new(dataset, source.clone(), field_id)?;
/// // Preserve this source context alongside batches from the matching scan.
/// capture.capture(&source, field, addresses, descriptors).await
/// # }
/// ```
pub struct BlobRefCapture {
    dataset: Arc<Dataset>,
    context: BlobSourceContext,
    field_id: i32,
    resolver: BlobV2ReadContext,
    field_path: Vec<i32>,
}

impl BlobRefCapture {
    /// Bind capture to the exact snapshot and Blob V2 leaf that produced the scan.
    ///
    /// `field_id` is the Lance field ID from `dataset.schema()`, not its positional
    /// column index. The source context must identify this exact loaded manifest.
    pub fn try_new(
        dataset: Arc<Dataset>,
        context: BlobSourceContext,
        field_id: i32,
    ) -> Result<Self> {
        context.validate_dataset(&dataset)?;
        let field_path = find_field_path(&dataset.schema().fields, field_id).ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob capture field_id {field_id} is not a field in this snapshot",
            ))
        })?;
        let field = dataset.schema().field_by_id(field_id).ok_or_else(|| {
            Error::internal(format!("Blob capture field_id {field_id} disappeared"))
        })?;
        if !field.is_blob_v2() {
            return Err(Error::invalid_input(format!(
                "Blob capture field '{}' (field_id {field_id}) is not Blob V2",
                field.name,
            )));
        }
        let resolver = BlobV2ReadContext::new(&dataset, field_id as u32);
        Ok(Self {
            dataset,
            context,
            field_id,
            resolver,
            field_path,
        })
    }

    /// Capture one reference per descriptor, preserving validity, order, and duplicates.
    ///
    /// `scan_source` must be the context recorded for the scan that produced these
    /// descriptors, and must match this session. Untagged descriptor arrays cannot
    /// prove their own provenance; callers must preserve that association through
    /// engine operators. `field` must be the original scan field, before aliasing.
    ///
    /// `row_addresses` contains physical `_rowaddr` values from the same scan, not
    /// logical `_rowid` values. It must have one entry per descriptor, including nulls.
    /// Nested leaves require [`Self::capture_nested`] to preserve element identity.
    pub async fn capture(
        &mut self,
        scan_source: &BlobSourceContext,
        field: &Field,
        row_addresses: &[RowAddress],
        descriptors: &StructArray,
    ) -> Result<BlobRefBatch> {
        if self.field_path.len() != 1 {
            return Err(Error::invalid_input(
                "Use capture_nested for a Blob V2 leaf below a container",
            ));
        }
        self.capture_descriptors(scan_source, field, row_addresses, descriptors)
            .await
    }

    /// Capture a selected Blob V2 leaf below structs and List/LargeList containers.
    /// Pass the original top-level scan field/array and one physical address per
    /// root row. The result maps each element to a flat reference without reading
    /// siblings or losing parent nulls, list ordinals, or empty lists. Other nested
    /// layouts are rejected. Field IDs avoid ambiguity in quoted field names.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_array::ArrayRef;
    /// # use arrow_schema::Field;
    /// # use lance::{Dataset, Result};
    /// # use lance::blob::reference::{BlobRefCapture, BlobSourceContext};
    /// # use lance_core::utils::address::RowAddress;
    /// # async fn example(dataset: Arc<Dataset>, source: BlobSourceContext,
    /// #     leaf_id: i32, field: &Field, addresses: &[RowAddress], array: &ArrayRef) -> Result<()> {
    /// let mut capture = BlobRefCapture::try_new(dataset, source.clone(), leaf_id)?;
    /// let nested = capture.capture_nested(&source, field, addresses, array).await?;
    /// assert_eq!(nested.indices().num_rows(), addresses.len());
    /// // Carry nested.references() and nested.indices() through the engine's Arrow transport.
    /// # Ok(())
    /// # }
    /// ```
    pub async fn capture_nested(
        &mut self,
        scan_source: &BlobSourceContext,
        field: &Field,
        row_addresses: &[RowAddress],
        descriptors: &ArrayRef,
    ) -> Result<NestedBlobRefBatch> {
        let prepared = nested::prepare(
            &self.dataset,
            &self.field_path,
            field,
            row_addresses,
            descriptors,
        )?;
        let references = self
            .capture_descriptors(
                scan_source,
                &prepared.field,
                &prepared.addresses,
                &prepared.descriptors,
            )
            .await?;
        NestedBlobRefBatch::try_new(references, prepared.indices)
    }

    async fn capture_descriptors(
        &mut self,
        scan_source: &BlobSourceContext,
        field: &Field,
        row_addresses: &[RowAddress],
        descriptors: &StructArray,
    ) -> Result<BlobRefBatch> {
        if scan_source.source_id() != self.context.source_id() {
            return Err(Error::invalid_input(
                "Blob capture scan source does not match the session snapshot or storage binding",
            ));
        }
        let expected = self
            .dataset
            .schema()
            .field_by_id(self.field_id)
            .ok_or_else(|| {
                Error::internal(format!(
                    "Blob capture field_id {} disappeared",
                    self.field_id
                ))
            })?;
        if field.name() != &expected.name
            || !field.is_blob()
            || BlobV2Layout::classify(descriptors.fields()) != Some(BlobV2Layout::Descriptor)
            || field.data_type() != descriptors.data_type()
        {
            return Err(Error::invalid_input(format!(
                "Blob capture expected descriptor field '{}' (field_id {}), got {field:?} with array type {}",
                expected.name,
                self.field_id,
                descriptors.data_type(),
            )));
        }
        if descriptors.len() != row_addresses.len() {
            return Err(Error::invalid_input(format!(
                "Blob capture descriptor count {} does not match row address count {} for field_id {}",
                descriptors.len(),
                row_addresses.len(),
                self.field_id,
            )));
        }
        let columns = BlobV2DescriptorColumns::new(descriptors);
        // Validate the entire batch before resolving stores or producing references.
        for (row, address) in row_addresses.iter().enumerate() {
            if columns.is_null_blob(row) {
                continue;
            }
            let fragment = self.dataset.get_fragment(address.fragment_id() as usize).ok_or_else(|| {
                Error::invalid_input(format!("Blob capture row {row} references missing fragment {} at address {address}", address.fragment_id()))
            })?;
            let physical_rows = fragment.metadata().physical_rows.ok_or_else(|| {
                Error::invalid_input(format!(
                    "Blob capture fragment {} has no physical row count",
                    fragment.id()
                ))
            })?;
            if address.row_offset() as usize >= physical_rows
                || fragment.data_file_for_field(self.field_id as u32).is_none()
            {
                return Err(Error::invalid_input(format!(
                    "Blob capture row {row} has invalid address {address} for field_id {} (fragment physical_rows {physical_rows})",
                    self.field_id,
                )));
            }
            if columns.positions.is_null(row)
                || columns.sizes.is_null(row)
                || columns.blob_ids.is_null(row)
                || columns.blob_uris.is_null(row)
            {
                return Err(Error::invalid_input(format!(
                    "Blob capture non-null descriptor row {row} contains null children"
                )));
            }
            let kind = BlobKind::try_from(columns.kinds.value(row)).map_err(|error| {
                Error::invalid_input(format!(
                    "Blob capture field_id {} descriptor row {row}: {error}",
                    self.field_id
                ))
            })?;
            if kind == BlobKind::Dedicated && columns.positions.value(row) != 0 {
                return Err(Error::invalid_input(format!(
                    "Blob capture Dedicated descriptor row {row} has nonzero position {}",
                    columns.positions.value(row)
                )));
            }
            columns
                .positions
                .value(row)
                .checked_add(columns.sizes.value(row))
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Blob capture descriptor row {row} position {} + size {} overflows u64",
                        columns.positions.value(row),
                        columns.sizes.value(row)
                    ))
                })?;
        }
        let mut references = Vec::with_capacity(descriptors.len());
        for (row, address) in row_addresses.iter().enumerate() {
            let reference = self
                .resolver
                .capture_reference(&columns, row, u64::from(*address), &self.context)
                .await?;
            references.push(reference);
        }
        BlobRefBatch::try_new(vec![self.context.clone()], references)
    }
}

fn find_field_path(fields: &[LanceField], field_id: i32) -> Option<Vec<i32>> {
    for field in fields {
        if field.id == field_id {
            return Some(vec![field.id]);
        }
        if !field.is_blob_v2()
            && let Some(mut path) = find_field_path(&field.children, field_id)
        {
            path.insert(0, field.id);
            return Some(path);
        }
    }
    None
}
