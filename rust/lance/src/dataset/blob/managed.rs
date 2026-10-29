// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{HashMap, HashSet},
    ops::Range,
    sync::Arc,
};

use arrow::array::AsArray;
use arrow::datatypes::UInt64Type;
use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField};
use futures::TryStreamExt;
use object_store::path::Path;

use super::locate::{
    BlobV2DescriptorColumns, BlobV2ReadContext, data_file_key_from_path,
    join_base_and_relative_path, resolve_managed_base_id,
};
use crate::blob::{BlobDescriptor, BlobDescriptorArrayBuilder};
use crate::dataset::Dataset;
use lance_core::datatypes::{BlobKind, BlobV2Layout, Field as LanceField};
use lance_core::utils::blob::blob_path;
use lance_core::{Error, ROW_ADDR, Result, utils::address::RowAddress};

pub(in crate::dataset) fn field_contains_blob(field: &LanceField) -> bool {
    field.is_blob_v2() || field.children.iter().any(field_contains_blob)
}

fn visit_managed_row(
    field: &LanceField,
    array: &dyn Array,
    row: usize,
    sidecar_source: Option<(&Dataset, u64)>,
    visit: &mut impl FnMut(u32, Option<u32>, &str, Range<u64>) -> Result<()>,
) -> Result<()> {
    if array.is_null(row) {
        return Ok(());
    }
    if field.is_blob_v2() {
        let values = array.as_struct_opt().ok_or_else(|| {
            Error::internal("Expected Blob descriptor array during reference scan")
        })?;
        let columns = BlobV2DescriptorColumns::new(values);
        if !columns.is_null_blob(row)
            && matches!(
                BlobKind::try_from(columns.kinds.value(row))?,
                BlobKind::Managed | BlobKind::ManagedWithBase
            )
        {
            let uri = columns.blob_uris.value(row);
            lance_core::utils::blob::validate_managed_reference(
                uri,
                columns.positions.value(row),
                columns.sizes.value(row),
            )?;
            visit(
                field.id as u32,
                columns.managed_base_id(row),
                uri,
                columns.positions.value(row)
                    ..columns.positions.value(row) + columns.sizes.value(row),
            )?;
        } else if !columns.is_null_blob(row)
            && matches!(
                BlobKind::try_from(columns.kinds.value(row))?,
                BlobKind::Packed | BlobKind::Dedicated
            )
            && let Some((dataset, row_addr)) = sidecar_source
        {
            let fragment = dataset
                .get_fragment(RowAddress::from(row_addr).fragment_id() as usize)
                .ok_or_else(|| Error::internal("Blob sidecar fragment is missing"))?;
            let file = fragment
                .data_file_for_field(field.id as u32)
                .ok_or_else(|| Error::internal("Blob sidecar data file is missing"))?;
            let Some(base_id) = file.base_id else {
                return Ok(());
            };
            let root = dataset.blob_base_path(Some(base_id))?;
            let path = blob_path(
                &dataset.data_file_dir(file)?,
                data_file_key_from_path(&file.path),
                columns.blob_ids.value(row),
            );
            let uri = path
                .prefix_match(&root)
                .ok_or_else(|| {
                    Error::internal(format!("Blob source {path} is outside base {root}"))
                })?
                .map(|part| part.as_ref().to_string())
                .collect::<Vec<_>>()
                .join("/");
            let start = if columns.kinds.value(row) == BlobKind::Dedicated as u8 {
                0
            } else {
                columns.positions.value(row)
            };
            let size = columns.sizes.value(row);
            let end = start.checked_add(size).ok_or_else(|| {
                Error::corrupt_file_named(&file.path, format!(
                    "Blob sidecar range overflows u64 for field '{}': position={start}, size={size}",
                    field.name,
                ))
            })?;
            visit(field.id as u32, Some(base_id), &uri, start..end)?;
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
                visit_managed_row(child, values.as_ref(), row, sidecar_source, visit)?;
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
                    sidecar_source,
                    visit,
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
                    sidecar_source,
                    visit,
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
pub(super) async fn managed_references(
    dataset: &Dataset,
    mut scan: crate::dataset::scanner::Scanner,
    include_registered_sidecars: bool,
) -> Result<HashSet<(Option<u32>, String)>> {
    let fields = dataset
        .schema()
        .fields
        .iter()
        .filter(|field| field_contains_blob(field))
        .collect::<Vec<_>>();
    if fields.is_empty() {
        return Ok(HashSet::new());
    }
    let names = fields
        .iter()
        .map(|field| field.name.as_str())
        .collect::<Vec<_>>();
    scan.project(&names)?;
    scan.with_row_address();
    let mut stream = scan.try_into_stream().await?;
    let mut references = HashSet::new();
    while let Some(batch) = stream.try_next().await? {
        for field in &fields {
            let values = batch
                .column_by_name(&field.name)
                .ok_or_else(|| Error::internal("Missing Blob column during reference scan"))?;
            for row in 0..batch.num_rows() {
                let row_addr = batch[ROW_ADDR].as_primitive::<UInt64Type>().value(row);
                visit_managed_row(
                    field,
                    values.as_ref(),
                    row,
                    include_registered_sidecars.then_some((dataset, row_addr)),
                    &mut |field_id, id, uri, _range| {
                        let id = resolve_managed_base_id(dataset, field_id, row_addr, id)?;
                        references.insert((id, uri.to_string()));
                        Ok(())
                    },
                )?;
            }
        }
    }
    Ok(references)
}

/// Resolve only objects within the cleanup owner's deletion jurisdiction.
pub(in crate::dataset) async fn managed_paths(
    dataset: &Dataset,
    owner: &Dataset,
    scan: crate::dataset::scanner::Scanner,
) -> Result<HashSet<Path>> {
    let references = managed_references(dataset, scan, false).await?;
    let mut paths = HashSet::new();
    let mut bases = HashMap::new();
    for (id, uri) in references {
        if let std::collections::hash_map::Entry::Vacant(entry) = bases.entry(id) {
            let root = dataset.blob_base_path(id)?;
            let store = dataset.object_store(id).await?;
            entry.insert((store.store_prefix == owner.object_store.store_prefix, root));
        }
        let (same_store, root) = bases
            .get(&id)
            .ok_or_else(|| Error::internal("Missing resolved Managed base"))?;
        if *same_store {
            let path = join_base_and_relative_path(root, &uri)?;
            if path.prefix_match(&owner.base).is_some() {
                paths.insert(path);
            }
        }
    }
    Ok(paths)
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
/// # use lance::Dataset;
/// # async fn preserve(dataset: Arc<Dataset>, values: &StructArray, rows: &[u64], field: &Field) -> lance::Result<()> {
/// let prepared = lance::dataset::preserve_managed_descriptors(
///     &dataset, 0, values, rows, field, None,
/// ).await?;
/// # let _ = prepared;
/// # Ok(()) }
/// ```
pub async fn preserve_managed_descriptors(
    dataset: &Arc<Dataset>,
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
    dataset.blob_base_path(target_base_id)?;
    let primary_base_id = dataset
        .manifest
        .base_paths
        .values()
        .find(|base| base.path == dataset.uri)
        .map(|base| base.id);
    let preserve_base = |source_base_id: Option<u32>| -> Result<Option<u32>> {
        match (source_base_id, target_base_id) {
            (None, Some(target)) => primary_base_id.map(Some).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Cannot preserve primary-storage blob references in target base {target}: \
                     dataset URI '{}' has no registered base",
                    dataset.uri
                ))
            }),
            _ => Ok(source_base_id),
        }
    };
    let columns = BlobV2DescriptorColumns::new(values);
    let mut builder = BlobDescriptorArrayBuilder::new_with_metadata(
        field.name(),
        field.is_nullable(),
        field.metadata().clone(),
    );
    let mut context = BlobV2ReadContext::new(dataset, field_id);
    for (row, row_addr) in row_addrs.iter().enumerate() {
        if columns.is_null_blob(row) {
            builder.push_null()?;
            continue;
        }
        let kind = BlobKind::try_from(columns.kinds.value(row))?;
        match kind {
            BlobKind::Managed | BlobKind::ManagedWithBase => {
                context.resolve_managed(&columns, row, *row_addr).await?;
                builder.push(BlobDescriptor::Managed {
                    base_id: preserve_base(resolve_managed_base_id(
                        dataset,
                        field_id,
                        *row_addr,
                        columns.managed_base_id(row),
                    )?)?,
                    uri: columns.blob_uris.value(row).to_string(),
                    offset: columns.positions.value(row),
                    size: columns.sizes.value(row),
                })?;
            }
            BlobKind::Packed | BlobKind::Dedicated => {
                let location = context.blob_read_location(*row_addr).await?;
                let root = dataset.blob_base_path(location.base_id)?;
                let path = blob_path(
                    &location.data_file_dir,
                    &location.data_file_key,
                    columns.blob_ids.value(row),
                );
                let uri = path
                    .prefix_match(&root)
                    .ok_or_else(|| {
                        Error::internal(format!("Blob source {path} is outside base {root}"))
                    })?
                    .map(|part| part.as_ref().to_string())
                    .collect::<Vec<_>>()
                    .join("/");
                builder.push(BlobDescriptor::Managed {
                    base_id: preserve_base(location.base_id)?,
                    uri,
                    offset: if kind == BlobKind::Dedicated {
                        0
                    } else {
                        columns.positions.value(row)
                    },
                    size: columns.sizes.value(row),
                })?;
            }
            BlobKind::External => builder.push(BlobDescriptor::External {
                base_id: columns.blob_ids.value(row),
                uri: columns.blob_uris.value(row).to_string(),
                offset: columns.positions.value(row),
                size: columns.sizes.value(row),
            })?,
            BlobKind::Inline => {
                let file = context
                    .collect_file(&columns, row, *row_addr)
                    .await?
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "Inline blob descriptor row {row} unexpectedly resolved to null"
                        ))
                    })?;
                builder.push_inline(file.read().await?)?;
            }
        }
    }
    Ok(builder.finish()?.into_parts().1)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dataset::blob::file::{BlobFile, BlobSource};
    use crate::dataset::blob::test_util::{managed_blob_schema, recording_range_store};
    use crate::{
        Dataset,
        blob::{BlobArrayBuilder, blob_field},
        dataset::WriteParams,
    };
    use arrow_array::RecordBatch;
    use arrow_array::RecordBatchIterator;
    use arrow_schema::Schema;
    use bytes::Bytes;
    use lance_arrow::{
        ARROW_EXT_NAME_KEY, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
        BLOB_INLINE_SIZE_THRESHOLD_META_KEY, BLOB_V2_EXT_NAME,
    };
    use lance_core::datatypes::BlobKind;
    use lance_core::{Error, utils::tempfile::TempStrDir};
    use lance_file::version::LanceFileVersion;
    use lance_table::feature_flags::FLAG_MANAGED_BLOBS;
    use lance_table::format::BasePath;
    use object_store::path::Path;
    use rstest::rstest;
    use std::collections::HashMap;
    use std::sync::{Arc, atomic::Ordering};
    use uuid::Uuid;

    #[rstest]
    #[case::inline(6, BlobKind::Inline)]
    #[case::packed(16, BlobKind::Packed)]
    #[case::dedicated(32, BlobKind::Dedicated)]
    #[tokio::test]
    async fn managed_write_read(
        #[case] size: usize,
        #[case] kind: BlobKind,
        #[values(false, true)] enabled: bool,
        #[values(false, true)] maximum_base_id: bool,
        #[values(false, true)] primary_write: bool,
        #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let payload = vec![42u8; size];
        let mut blobs = BlobArrayBuilder::new(3);
        blobs.push_bytes(&payload).unwrap();
        blobs.push_bytes(b"").unwrap();
        blobs.push_null().unwrap();
        let field = blob_field("blob", true).with_metadata(HashMap::from([
            (ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string()),
            (
                BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
                "8".to_string(),
            ),
            (
                BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
                "24".to_string(),
            ),
        ]));
        let schema = Arc::new(if enabled {
            managed_blob_schema(vec![field])
        } else {
            Schema::new(vec![field])
        });
        let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema),
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(version),
                    initial_bases: maximum_base_id.then(|| {
                        vec![BasePath::new(
                            u32::MAX,
                            test_dir.to_string(),
                            Some("payload".to_string()),
                            true,
                        )]
                    }),
                    target_bases: (maximum_base_id && !primary_write).then_some(vec![u32::MAX]),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        let flag = FLAG_MANAGED_BLOBS;
        let expected_flags = if enabled { flag } else { 0 };
        assert_eq!(dataset.manifest.reader_feature_flags & flag, expected_flags);
        assert_eq!(dataset.manifest.writer_feature_flags & flag, expected_flags);
        for (_, uri) in managed_references(&dataset, dataset.scan(), false)
            .await
            .unwrap()
        {
            assert!(uri.starts_with("_blobs/"), "{uri}");
            assert_eq!(uri.split('/').count(), 2);
            Uuid::parse_str(uri.trim_start_matches("_blobs/").trim_end_matches(".blob")).unwrap();
        }
        let id = (maximum_base_id && !primary_write).then_some(u32::MAX);
        assert_eq!(
            dataset.manifest.base_paths.len(),
            usize::from(maximum_base_id)
        );
        for fragment in dataset.get_fragments() {
            assert_eq!(fragment.metadata.files[0].base_id, id);
        }
        let values = dataset
            .take_blobs_by_indices(&[0, 1, 2], "blob")
            .await
            .unwrap();
        let kind = if enabled && kind != BlobKind::Inline {
            BlobKind::Managed
        } else {
            kind
        };
        assert_eq!(values[0].as_ref().unwrap().kind(), kind);
        assert_eq!(
            values[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            payload
        );
        assert!(values[1].as_ref().unwrap().read().await.unwrap().is_empty());
        assert!(values[2].is_none());
        if kind == BlobKind::Managed
            && let Some(id) = id
        {
            let mut missing_base = dataset.as_ref().clone();
            let manifest = Arc::make_mut(&mut missing_base.manifest);
            // The data file lives at the dataset root even when the writer
            // selected an explicit base. Isolate the descriptor's base lookup.
            for fragment in Arc::make_mut(&mut manifest.fragments) {
                for file in fragment.referenced_lance_files_mut() {
                    file.base_id = None;
                }
            }
            manifest.base_paths.clear();
            let error = Arc::new(missing_base)
                .take_blobs_by_indices(&[0], "blob")
                .await
                .err()
                .unwrap();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(
                error.to_string().contains(&format!("unknown base_id {id}")),
                "{error}"
            );
        }
    }

    #[tokio::test]
    async fn managed_empty_range_does_not_discover_object_length() {
        let (store, recording) = recording_range_store(Bytes::from_static(b"not empty"));
        let source = Arc::new(BlobSource::new(store, Path::from("data/a/payload.blob")));
        let blob = BlobFile::with_source(source, 4, 0, BlobKind::Managed, None);
        assert!(blob.read().await.unwrap().is_empty());
        assert!(blob.read_range(0..0).await.unwrap().is_empty());
        assert_eq!(recording.head_requests.load(Ordering::Relaxed), 0);
        assert!(recording.requested_ranges().is_empty());
    }
}
