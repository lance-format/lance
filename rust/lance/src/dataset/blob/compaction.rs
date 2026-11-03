// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Copy ordinary sidecars without materializing their payloads. IDs are unique
//! within a rewrite task; output files copy only the objects their rows use.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use arrow_array::{Array, ArrayRef, StructArray, cast::AsArray};
use arrow_schema::Field;
use futures::TryStreamExt;
use lance_arrow::{list::ListArrayExt, r#struct::StructArrayExt};
use lance_core::datatypes::{BlobKind, Field as LanceField};
use lance_core::utils::blob::blob_path;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_table::format::Fragment;
use object_store::path::Path;

use super::locate::{
    BlobV2DescriptorColumns, BlobV2ReadContext, data_file_key_from_path, resolve_managed_base_id,
};
use crate::Dataset;
use crate::blob::{BlobDescriptor, BlobDescriptorArrayBuilder, BlobRange};
use crate::dataset::fragment::FileFragment;

#[derive(Clone)]
struct Source {
    store: Arc<ObjectStore>,
    path: Path,
}

#[derive(Default)]
struct Sources {
    ids: HashMap<(Option<u32>, Path), u32>,
    objects: HashMap<u32, Source>,
}

#[derive(Default)]
pub(in crate::dataset) struct SidecarCopy(Mutex<Sources>);

impl SidecarCopy {
    fn register(&self, base: Option<u32>, source: Source) -> Result<u32> {
        let mut sources = self
            .0
            .lock()
            .map_err(|_| Error::internal("Poisoned sidecar copy map"))?;
        let key = (base, source.path.clone());
        if let Some(id) = sources.ids.get(&key) {
            return Ok(*id);
        }
        let id = sources
            .ids
            .len()
            .checked_add(1)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| Error::invalid_input("Too many sidecar objects in compaction task"))?;
        sources.ids.insert(key, id);
        sources.objects.insert(id, source);
        Ok(id)
    }

    pub(super) async fn prepare(
        &self,
        dataset: &Arc<Dataset>,
        field_id: u32,
        values: &StructArray,
        rows: &[u64],
        field: &Field,
    ) -> Result<ArrayRef> {
        if values.len() != rows.len() {
            return Err(Error::internal(
                "Blob descriptor and source row counts differ",
            ));
        }
        let columns = BlobV2DescriptorColumns::new(values);
        let mut context = BlobV2ReadContext::new(dataset, field_id);
        let mut builder = BlobDescriptorArrayBuilder::new_with_metadata(
            field.name(),
            field.is_nullable(),
            field.metadata().clone(),
        );
        for (row, row_addr) in rows.iter().enumerate() {
            if columns.is_null_blob(row) {
                builder.push_null()?;
                continue;
            }
            match BlobKind::try_from(columns.kinds.value(row))? {
                kind @ (BlobKind::Packed | BlobKind::Dedicated) => {
                    let location = context.blob_read_location(*row_addr).await?;
                    let path = blob_path(
                        &location.data_file_dir,
                        &location.data_file_key,
                        columns.blob_ids.value(row),
                    );
                    let id = self.register(
                        location.base_id,
                        Source {
                            store: location.object_store,
                            path,
                        },
                    )?;
                    if kind == BlobKind::Packed {
                        builder.push_packed(
                            id,
                            BlobRange {
                                offset: columns.positions.value(row),
                                size: columns.sizes.value(row),
                            },
                        )?;
                    } else {
                        builder.push_dedicated(id, columns.sizes.value(row))?;
                    }
                }
                BlobKind::Inline => {
                    let file = context.collect_inline(&columns, row, *row_addr).await?;
                    builder.push_inline(file.read().await?)?;
                }
                BlobKind::External => builder.push(BlobDescriptor::External {
                    base_id: columns.blob_ids.value(row),
                    uri: columns.blob_uris.value(row).to_string(),
                    offset: columns.positions.value(row),
                    size: columns.sizes.value(row),
                })?,
                BlobKind::Managed | BlobKind::ManagedWithBase => {
                    // A clone can retain Managed descriptors after the schema's
                    // write opt-in is removed. Copy those ranges into ordinary
                    // packed objects rather than changing the write capability.
                    let file = context.collect_managed(&columns, row, *row_addr).await?;
                    let base = resolve_managed_base_id(
                        dataset,
                        field_id,
                        *row_addr,
                        columns.managed_base_id(row),
                    )?;
                    let id = self.register(
                        base,
                        Source {
                            store: file.source.object_store.clone(),
                            path: file.source.path.clone(),
                        },
                    )?;
                    builder.push_packed(
                        id,
                        BlobRange {
                            offset: file.position,
                            size: file.size,
                        },
                    )?;
                }
            }
        }
        Ok(builder.finish()?.into_parts().1)
    }

    pub(in crate::dataset) async fn copy_to_fragments(
        &self,
        dataset: &Dataset,
        fragments: &[Fragment],
    ) -> Result<()> {
        let sources = self
            .0
            .lock()
            .map_err(|_| Error::internal("Poisoned sidecar copy map"))?
            .objects
            .clone();
        if sources.is_empty() {
            return Ok(());
        }
        let fields = dataset
            .schema()
            .fields
            .iter()
            .filter(|f| super::field_contains_blob(f))
            .collect::<Vec<_>>();
        let names = fields.iter().map(|f| f.name.as_str()).collect::<Vec<_>>();
        let dataset = Arc::new(dataset.clone());
        for fragment in fragments {
            let file_fragment = FileFragment::new(dataset.clone(), fragment.clone());
            let mut scan = file_fragment.scan();
            scan.project(&names)?;
            let mut stream = scan.try_into_stream().await?;
            let mut copied = HashSet::new();
            while let Some(batch) = stream.try_next().await? {
                for field in &fields {
                    let array = batch
                        .column_by_name(&field.name)
                        .ok_or_else(|| Error::internal("Missing compacted Blob column"))?;
                    let mut ids = HashSet::new();
                    collect_ids(field, array.as_ref(), &mut ids)?;
                    for (field_id, id) in ids {
                        let file = file_fragment
                            .data_file_for_field(field_id)
                            .ok_or_else(|| Error::internal("Missing compacted Blob data file"))?;
                        if !copied.insert((file.path.clone(), id)) {
                            continue;
                        }
                        let source = sources.get(&id).ok_or_else(|| {
                            Error::internal(format!("Missing source for compacted sidecar {id}"))
                        })?;
                        let destination = blob_path(
                            &dataset.data_file_dir(file)?,
                            data_file_key_from_path(&file.path),
                            id,
                        );
                        let store = dataset.object_store(file.base_id).await?;
                        if Arc::ptr_eq(&source.store, &store) {
                            store.copy(&source.path, &destination).await?;
                        } else {
                            source
                                .store
                                .copy_bulk(&source.path, store.as_ref(), &destination)
                                .await?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn collect_ids(field: &LanceField, array: &dyn Array, ids: &mut HashSet<(u32, u32)>) -> Result<()> {
    if field.is_blob_v2() {
        let values = array.as_struct();
        let columns = BlobV2DescriptorColumns::new(values);
        for row in 0..values.len() {
            if !columns.is_null_blob(row)
                && matches!(
                    BlobKind::try_from(columns.kinds.value(row))?,
                    BlobKind::Packed | BlobKind::Dedicated
                )
            {
                ids.insert((field.id as u32, columns.blob_ids.value(row)));
            }
        }
    } else {
        match array.data_type() {
            arrow_schema::DataType::Struct(_) => {
                let values = array.as_struct().normalize_slicing()?.pushdown_nulls()?;
                for child in &field.children {
                    if super::field_contains_blob(child) {
                        let values = values
                            .column_by_name(&child.name)
                            .ok_or_else(|| Error::internal("Missing Blob child"))?;
                        collect_ids(child, values.as_ref(), ids)?;
                    }
                }
            }
            arrow_schema::DataType::List(_) => collect_ids(
                &field.children[0],
                array
                    .as_list::<i32>()
                    .filter_garbage_nulls()
                    .trimmed_values()
                    .as_ref(),
                ids,
            )?,
            arrow_schema::DataType::LargeList(_) => collect_ids(
                &field.children[0],
                array
                    .as_list::<i64>()
                    .filter_garbage_nulls()
                    .trimmed_values()
                    .as_ref(),
                ids,
            )?,
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BlobArrayBuilder;
    use crate::dataset::blob::{BlobV2BatchRewritePlan, prepared_blob_ids};
    use crate::dataset::{
        WriteParams,
        optimize::{CompactionOptions, compact_files},
    };
    use arrow_array::{RecordBatch, RecordBatchIterator};
    use arrow_schema::Schema;
    use lance_arrow::{
        BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, BLOB_INLINE_SIZE_THRESHOLD_META_KEY,
    };
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;

    async fn fixture() -> (TempStrDir, Dataset) {
        let dir = TempStrDir::default();
        let field = crate::blob_field("blob", true).with_metadata(HashMap::from([
            ("ARROW:extension:name".into(), "lance.blob.v2".into()),
            (BLOB_INLINE_SIZE_THRESHOLD_META_KEY.into(), "8".into()),
            (BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.into(), "24".into()),
        ]));
        let schema = Arc::new(Schema::new(vec![field]));
        let mut blobs = BlobArrayBuilder::new(8);
        for size in [16, 16, 32, 4, 16, 16, 32, 4] {
            blobs.push_bytes(vec![size as u8; size]).unwrap();
        }
        let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &dir,
            Some(WriteParams {
                max_rows_per_file: 2,
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        (dir, dataset)
    }

    #[tokio::test]
    async fn sidecar_compaction_copies_objects_across_batches_and_rollover() {
        let (_dir, mut dataset) = fixture().await;
        let old = Arc::new(dataset.clone());
        let before = old
            .take_blobs_by_indices(&(0..8).collect::<Vec<_>>(), "blob")
            .await
            .unwrap();
        let options = CompactionOptions {
            target_rows_per_fragment: 3,
            batch_size: Some(1),
            ..Default::default()
        };
        // Force one oversized task so its writer must roll over while the
        // same task-wide source-ID map remains in use.
        let task = crate::dataset::optimize::CompactionTask {
            task: crate::dataset::optimize::TaskData {
                fragments: dataset.manifest.fragments.as_ref().clone(),
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            },
            read_version: dataset.manifest.version,
            options: options.clone(),
        };
        let result = task.execute(&dataset).await.unwrap();
        crate::dataset::optimize::commit_compaction(
            &mut dataset,
            vec![result],
            Arc::new(crate::dataset::index::DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();
        assert_eq!(dataset.manifest.fragments.len(), 2);
        assert!(!dataset.manifest.has_managed_blobs());
        let new = Arc::new(dataset.clone());
        let after = new
            .take_blobs_by_indices(&(0..8).collect::<Vec<_>>(), "blob")
            .await
            .unwrap();
        for (before, after) in before.iter().zip(&after) {
            let before = before.as_ref().unwrap();
            let after = after.as_ref().unwrap();
            assert_eq!(before.kind(), after.kind());
            assert_eq!(before.read().await.unwrap(), after.read().await.unwrap());
        }
        crate::dataset::cleanup::cleanup_old_versions(
            &dataset,
            crate::dataset::cleanup::CleanupPolicy {
                before_timestamp: Some(chrono::Utc::now() + chrono::TimeDelta::seconds(1)),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let reopened = Arc::new(Dataset::open(dataset.uri()).await.unwrap());
        let blobs = reopened
            .take_blobs_by_indices(&(0..8).collect::<Vec<_>>(), "blob")
            .await
            .unwrap();
        for (blob, size) in blobs.iter().zip([16, 16, 32, 4, 16, 16, 32, 4]) {
            assert_eq!(
                blob.as_ref().unwrap().read().await.unwrap().as_ref(),
                vec![size as u8; size]
            );
        }
        // Each output owns exactly its packed and dedicated copy, despite
        // multiple batches referencing the same packed object.

        for fragment in dataset.get_fragments() {
            let objects = dataset
                .object_store
                .read_dir_all(
                    &dataset
                        .data_file_dir(&fragment.metadata().files[0])
                        .unwrap()
                        .join(data_file_key_from_path(&fragment.metadata().files[0].path)),
                    None,
                )
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            assert_eq!(objects.len(), 2);
        }
    }

    #[tokio::test]
    async fn sidecar_compaction_copy_failure_does_not_publish_fragments() {
        let (_dir, mut dataset) = fixture().await;
        let version = dataset.manifest.version;
        let before = dataset
            .object_store
            .read_dir_all(&dataset.base, None)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let sidecar = before
            .iter()
            .find(|o| o.location.extension() == Some("blob"))
            .unwrap();
        dataset
            .object_store
            .delete(&sidecar.location)
            .await
            .unwrap();
        let error = compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(".blob"));
        assert!(matches!(error, Error::NotFound { .. } | Error::IO { .. }));
        assert_eq!(dataset.manifest.version, version);
        let after = dataset
            .object_store
            .read_dir_all(&dataset.base, None)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let data_paths = |objects: &[object_store::ObjectMeta]| {
            objects
                .iter()
                .filter(|o| o.location.extension() == Some("lance"))
                .map(|o| o.location.clone())
                .collect::<HashSet<_>>()
        };
        assert_eq!(data_paths(&before), data_paths(&after));
    }

    #[tokio::test]
    async fn sidecar_compaction_prepares_descriptors_without_payload_reads() {
        let (_dir, dataset) = fixture().await;
        let mut scan = dataset.scan();
        scan.with_row_address();
        let batch = scan.try_into_batch().await.unwrap();
        let objects = dataset
            .object_store
            .read_dir_all(&dataset.base, None)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        for object in objects
            .iter()
            .filter(|o| o.location.extension() == Some("blob"))
        {
            dataset.object_store.delete(&object.location).await.unwrap();
        }
        let plan =
            BlobV2BatchRewritePlan::try_new(dataset.schema(), batch.schema().as_ref(), false, true)
                .unwrap()
                .with_sidecar_copy(Arc::new(SidecarCopy::default()));
        let prepared = plan
            .transform_batch(&Arc::new(dataset), batch)
            .await
            .unwrap();
        // Missing payload objects cannot affect descriptor preparation. The
        // previous logical rewrite attempted to read them into LargeBinary.
        assert_eq!(prepared.num_rows(), 8);
        assert_eq!(prepared_blob_ids(&prepared).unwrap().len(), 4);
    }
}
