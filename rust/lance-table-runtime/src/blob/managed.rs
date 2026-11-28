// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};

use arrow_array::{Array, ArrayRef, StructArray, cast::AsArray, types::UInt64Type};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField};
use futures::TryStreamExt;
use lance_core::{
    Error, ROW_ADDR, Result,
    datatypes::{BlobV2Layout, Field as LanceField},
};
use lance_io::stream::SendableRecordBatchStream;

use super::locate::{BlobLocator, StoredBlob};
use super::{BlobDescriptor, BlobDescriptorArrayBuilder};
use crate::Table;

#[doc(hidden)]
pub fn field_contains_blob(field: &LanceField) -> bool {
    field.is_blob_v2() || field.children.iter().any(field_contains_blob)
}

/// The objects referenced by Blob descriptors, collected one row at a time.
struct ManagedReferenceScan<'a> {
    table: &'a Table,
    include_registered_sidecars: bool,
    locators: HashMap<u32, BlobLocator<'a>>,
    references: HashSet<(Option<u32>, String)>,
}

fn visit_managed_row(
    field: &LanceField,
    array: &dyn Array,
    row: usize,
    row_addr: u64,
    scan: &mut ManagedReferenceScan<'_>,
) -> Result<()> {
    if array.is_null(row) {
        return Ok(());
    }
    if field.is_blob_v2() {
        let values = array.as_struct_opt().ok_or_else(|| {
            Error::internal("Expected Blob descriptor array during reference scan")
        })?;
        let field_id = field.id as u32;
        let table = scan.table;
        let locator = scan
            .locators
            .entry(field_id)
            .or_insert_with(|| BlobLocator::new(table, field_id));
        match StoredBlob::parse(values, row)? {
            Some(StoredBlob::Managed { base_id, uri, .. }) => {
                let base_id = locator.managed_base(base_id, row_addr)?;
                scan.references.insert((base_id, uri.to_string()));
            }
            Some(StoredBlob::Packed { blob_id, .. } | StoredBlob::Dedicated { blob_id, .. })
                if scan.include_registered_sidecars =>
            {
                if let (Some(base_id), uri) = locator.sidecar(row_addr, blob_id)? {
                    scan.references.insert((Some(base_id), uri.to_string()));
                }
            }
            _ => {}
        }
        return Ok(());
    }
    match array.data_type() {
        ArrowDataType::Struct(_) => {
            for child in field
                .children
                .iter()
                .filter(|child| field_contains_blob(child))
            {
                let values = array
                    .as_struct()
                    .column_by_name(&child.name)
                    .ok_or_else(|| Error::internal("Missing Blob child during reference scan"))?;
                visit_managed_row(child, values.as_ref(), row, row_addr, scan)?;
            }
        }
        ArrowDataType::List(_) => {
            let values = array.as_list::<i32>();
            let child = field
                .children
                .first()
                .ok_or_else(|| Error::internal("Missing Blob list item field"))?;
            for index in values.value_offsets()[row]..values.value_offsets()[row + 1] {
                visit_managed_row(
                    child,
                    values.values().as_ref(),
                    index as usize,
                    row_addr,
                    scan,
                )?;
            }
        }
        ArrowDataType::LargeList(_) => {
            let values = array.as_list::<i64>();
            let child = field
                .children
                .first()
                .ok_or_else(|| Error::internal("Missing Blob list item field"))?;
            for index in values.value_offsets()[row]..values.value_offsets()[row + 1] {
                visit_managed_row(
                    child,
                    values.values().as_ref(),
                    index as usize,
                    row_addr,
                    scan,
                )?;
            }
        }
        _ => {
            return Err(Error::not_supported(format!(
                "Cannot scan Managed references in {}",
                array.data_type()
            )));
        }
    }
    Ok(())
}

/// Discover referenced objects from stored descriptors without reading payloads.
#[doc(hidden)]
pub async fn managed_references(
    table: &Table,
    mut stream: SendableRecordBatchStream,
    include_registered_sidecars: bool,
) -> Result<HashSet<(Option<u32>, String)>> {
    let fields = table
        .schema()
        .fields
        .iter()
        .filter(|field| field_contains_blob(field))
        .collect::<Vec<_>>();
    if fields.is_empty() {
        return Ok(HashSet::new());
    }
    let mut reference_scan = ManagedReferenceScan {
        table,
        include_registered_sidecars,
        locators: HashMap::new(),
        references: HashSet::new(),
    };
    while let Some(batch) = stream.try_next().await? {
        let row_addrs = batch
            .column_by_name(ROW_ADDR)
            .ok_or_else(|| Error::internal("Missing row addresses during reference scan"))?
            .as_primitive::<UInt64Type>();
        for field in &fields {
            let values = batch
                .column_by_name(&field.name)
                .ok_or_else(|| Error::internal("Missing Blob column during reference scan"))?;
            for row in 0..batch.num_rows() {
                visit_managed_row(
                    field,
                    values.as_ref(),
                    row,
                    row_addrs.value(row),
                    &mut reference_scan,
                )?;
            }
        }
    }
    Ok(reference_scan.references)
}

/// Preserve Managed references and adopt Packed/Dedicated objects in place.
/// Only Inline payloads are copied. Source bases are made explicit when needed
/// to preserve their identity at the destination.
/// Convert scanned descriptors into prepared write values without reading
/// out-of-line payloads. `row_addrs` must identify each descriptor's source
/// row in `dataset`; `field_id` identifies its blob leaf. Inline values are
/// read from their source data file. The caller must retain the source snapshot
/// until the new references have been committed.
///
/// `field` supplies the destination name, nullability and metadata. The returned
/// array has the prepared Blob v2 layout and can be interleaved with values from
/// [`BlobDescriptorArrayBuilder`] before writing.
///
/// `target_base_id` is the destination data file's base in the same dataset:
/// `None` for primary storage, or `Some(id)` for a registered base. Write the
/// returned array only to that base, using the same base registry. Reusing
/// primary-storage objects in a registered base requires a registered base
/// whose URI matches the dataset URI; otherwise this function returns an error.
///
/// ```
/// # use std::sync::Arc;
/// # use arrow_array::StructArray;
/// # use arrow_schema::Field;
/// # use lance_table_runtime::{Table, blob::preserve_managed_descriptors};
/// # async fn preserve(dataset: Arc<Table>, values: &StructArray, rows: &[u64], field: &Field) -> lance_core::Result<()> {
/// let prepared = preserve_managed_descriptors(
///     &dataset, 0, values, rows, field, None,
/// ).await?;
/// # let _ = prepared;
/// # Ok(()) }
/// ```
pub async fn preserve_managed_descriptors(
    dataset: &Table,
    field_id: u32,
    values: &StructArray,
    row_addrs: &[u64],
    field: &ArrowField,
    target_base_id: Option<u32>,
) -> Result<ArrayRef> {
    if values.len() != row_addrs.len() {
        return Err(Error::invalid_input(format!(
            "Blob descriptor count {} does not match row address count {}",
            values.len(),
            row_addrs.len()
        )));
    }
    if BlobV2Layout::classify(values.fields()) != Some(BlobV2Layout::Descriptor) {
        return Err(Error::invalid_input(format!(
            "Expected Blob v2 descriptors, got {:?}",
            values.data_type()
        )));
    }
    let source_field = i32::try_from(field_id)
        .ok()
        .and_then(|id| dataset.schema().field_by_id(id));
    if !source_field.is_some_and(|field| field.is_blob_v2()) {
        return Err(Error::invalid_input(format!(
            "Source field {field_id} is not a Blob v2 field"
        )));
    }
    dataset.view().blob_base_path(target_base_id)?;
    let primary_base_id = dataset
        .manifest
        .base_paths
        .values()
        .find(|base| base.path == dataset.uri())
        .map(|base| base.id);
    let preserve_base = |source_base_id: Option<u32>| -> Result<Option<u32>> {
        match (source_base_id, target_base_id) {
            (None, Some(target)) => primary_base_id.map(Some).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Cannot preserve primary-storage blob references in target base {target}: \
                     dataset URI '{}' has no registered base",
                    dataset.uri()
                ))
            }),
            _ => Ok(source_base_id),
        }
    };
    let mut builder = BlobDescriptorArrayBuilder::new_with_metadata(
        field.name(),
        field.is_nullable(),
        field.metadata().clone(),
    );
    let mut locator = BlobLocator::new(dataset, field_id);
    for (row, row_addr) in row_addrs.iter().copied().enumerate() {
        let Some(blob) = StoredBlob::parse(values, row)? else {
            builder.push_null()?;
            continue;
        };
        match blob {
            StoredBlob::Managed {
                base_id,
                uri,
                position,
                size,
            } => {
                // Reject descriptors whose base cannot be resolved before
                // making their base explicit.
                locator.locate(&blob, row_addr).await?;
                builder.push(BlobDescriptor::Managed {
                    base_id: preserve_base(locator.managed_base(base_id, row_addr)?)?,
                    uri: uri.to_string(),
                    offset: position,
                    size,
                })?;
            }
            StoredBlob::Packed {
                blob_id,
                position,
                size,
            } => {
                let (base_id, uri) = locator.sidecar(row_addr, blob_id)?;
                builder.push(BlobDescriptor::Managed {
                    base_id: preserve_base(base_id)?,
                    uri: uri.to_string(),
                    offset: position,
                    size,
                })?;
            }
            StoredBlob::Dedicated { blob_id, size } => {
                let (base_id, uri) = locator.sidecar(row_addr, blob_id)?;
                builder.push(BlobDescriptor::Managed {
                    base_id: preserve_base(base_id)?,
                    uri: uri.to_string(),
                    offset: 0,
                    size,
                })?;
            }
            StoredBlob::External {
                base_id,
                uri,
                position,
                size,
            } => builder.push(BlobDescriptor::External {
                base_id,
                uri: uri.to_string(),
                offset: position,
                size,
            })?,
            StoredBlob::Empty | StoredBlob::Inline { .. } => {
                let file = locator.locate(&blob, row_addr).await?;
                builder.push_inline(file.read().await?)?;
            }
        }
    }
    Ok(builder.finish()?.into_parts().1)
}
