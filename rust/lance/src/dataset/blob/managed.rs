// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow::array::AsArray;
use arrow::datatypes::UInt64Type;
use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField};
use futures::TryStreamExt;
use object_store::path::Path;

use super::locate::{BlobLocator, StoredBlob, join_base_and_relative_path};
use crate::blob::{BlobDescriptor, BlobDescriptorArrayBuilder};
use crate::dataset::Dataset;
use lance_core::datatypes::Field as LanceField;
use lance_core::{Error, ROW_ADDR, Result};

pub(in crate::dataset) fn field_contains_blob(field: &LanceField) -> bool {
    field.is_blob_v2() || field.children.iter().any(field_contains_blob)
}

/// The objects referenced by Blob descriptors, collected one row at a time.
struct ManagedReferenceScan<'a> {
    dataset: &'a Dataset,
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
        let dataset = scan.dataset;
        let locator = scan
            .locators
            .entry(field_id)
            .or_insert_with(|| BlobLocator::new(dataset, field_id));
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
    let mut reference_scan = ManagedReferenceScan {
        dataset,
        include_registered_sidecars,
        locators: HashMap::new(),
        references: HashSet::new(),
    };
    while let Some(batch) = stream.try_next().await? {
        let row_addrs = batch[ROW_ADDR].as_primitive::<UInt64Type>();
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
/// Only Inline payloads are copied. Implicit source bases become explicit when
/// a shallow clone rewrites descriptors into its own table base.
pub(in crate::dataset) async fn preserve_managed_descriptors(
    dataset: &Arc<Dataset>,
    field_id: u32,
    values: &StructArray,
    row_addrs: &[u64],
    field: &ArrowField,
) -> Result<ArrayRef> {
    if values.len() != row_addrs.len() {
        return Err(Error::internal(
            "Blob descriptors and row addresses have different lengths",
        ));
    }
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
                    base_id: locator.managed_base(base_id, row_addr)?,
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
                    base_id,
                    uri: uri.to_string(),
                    offset: position,
                    size,
                })?;
            }
            StoredBlob::Dedicated { blob_id, size } => {
                let (base_id, uri) = locator.sidecar(row_addr, blob_id)?;
                builder.push(BlobDescriptor::Managed {
                    base_id,
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
