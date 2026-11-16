// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Resolve ordinary sidecars without materializing their payloads. Task-local
//! IDs let destination writers stream live ranges and copy dedicated objects.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::Field;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;

use super::locate::{BlobLocator, StoredBlob};
use crate::Dataset;
use crate::blob::{BlobDescriptor, BlobDescriptorArrayBuilder, BlobRange};

#[derive(Clone)]
pub(super) struct Source {
    pub(super) store: Arc<ObjectStore>,
    pub(super) path: Path,
}

#[derive(Default)]
struct Sources {
    ids: HashMap<(Option<u32>, Path), u32>,
    objects: HashMap<u32, Source>,
}

#[derive(Default)]
pub(in crate::dataset) struct SidecarRewrite(Mutex<Sources>);

impl SidecarRewrite {
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
        let mut locator = BlobLocator::new(dataset, field_id);
        let mut builder = BlobDescriptorArrayBuilder::new_with_metadata(
            field.name(),
            field.is_nullable(),
            field.metadata().clone(),
        );
        for (row, row_addr) in rows.iter().copied().enumerate() {
            let Some(blob) = StoredBlob::parse(values, row)? else {
                builder.push_null()?;
                continue;
            };
            match blob {
                StoredBlob::Packed { blob_id, .. } | StoredBlob::Dedicated { blob_id, .. } => {
                    let base_id = locator.sidecar(row_addr, blob_id)?.0;
                    let file = locator.locate(&blob, row_addr).await?;
                    let id = self.register(
                        base_id,
                        Source {
                            store: file.source.object_store.clone(),
                            path: file.source.path.clone(),
                        },
                    )?;
                    if matches!(blob, StoredBlob::Dedicated { .. }) {
                        builder.push_dedicated(id, file.size)?;
                    } else {
                        builder.push_packed(
                            id,
                            BlobRange {
                                offset: file.position,
                                size: file.size,
                            },
                        )?;
                    }
                }
                StoredBlob::Empty | StoredBlob::Inline { .. } => {
                    let file = locator.locate(&blob, row_addr).await?;
                    builder.push_inline(file.read().await?)?;
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
                StoredBlob::Managed { base_id, .. } => {
                    // A clone can retain Managed descriptors after the schema's
                    // write opt-in is removed. Stream those ranges into ordinary
                    // packs rather than changing the write capability.
                    let base = locator.managed_base(base_id, row_addr)?;
                    let file = locator.locate(&blob, row_addr).await?;
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

    pub(super) fn source(&self, id: u32) -> Result<Source> {
        self.0
            .lock()
            .map_err(|_| Error::internal("Poisoned sidecar copy map"))?
            .objects
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::internal(format!("Missing source for compacted sidecar {id}")))
    }
}

#[cfg(test)]
mod tests {
    use super::super::locate::data_file_key_from_path;
    use super::*;
    use crate::BlobArrayBuilder;
    use crate::dataset::blob::{BlobV2BatchRewritePlan, prepared_blob_ids};
    use crate::dataset::{
        WriteParams,
        optimize::{CompactionOptions, compact_files},
    };
    use arrow_array::{RecordBatch, RecordBatchIterator};
    use arrow_schema::Schema;
    use futures::TryStreamExt;
    use lance_arrow::{
        BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, BLOB_INLINE_SIZE_THRESHOLD_META_KEY,
    };
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;
    use std::collections::HashSet;

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

    #[rstest::rstest]
    #[tokio::test]
    async fn sidecar_compaction_reclaims_deleted_packed_bytes(
        #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
        #[values(1, 128)] batch_size: usize,
        #[values(1, 4)] live_rows: u64,
    ) {
        let dir = TempStrDir::default();
        let field = crate::blob_field("blob", true).with_metadata(HashMap::from([
            ("ARROW:extension:name".into(), "lance.blob.v2".into()),
            (BLOB_INLINE_SIZE_THRESHOLD_META_KEY.into(), "8".into()),
            (BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.into(), "24".into()),
        ]));
        let schema = Arc::new(Schema::new(vec![field]));
        let mut blobs = BlobArrayBuilder::new(1000);
        for row in 0..1000 {
            blobs.push_bytes(vec![row as u8; 16]).unwrap();
        }
        let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &dir,
            Some(WriteParams {
                data_storage_version: Some(version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        // Retain a range in the middle of the source pack: compaction must
        // update its offset as well as dropping the unreferenced payloads.
        dataset
            .delete(&format!("_rowid < 500 OR _rowid >= {}", 500 + live_rows))
            .await
            .unwrap();
        let options = CompactionOptions {
            target_rows_per_fragment: 2,
            batch_size: Some(batch_size),
            ..Default::default()
        };
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
        assert_eq!(dataset.count_rows(None).await.unwrap(), live_rows as usize);
        assert_eq!(
            dataset.manifest.fragments.len(),
            if live_rows == 1 { 1 } else { 2 }
        );
        crate::dataset::cleanup::cleanup_old_versions(
            &dataset,
            crate::dataset::cleanup::CleanupPolicy {
                before_timestamp: Some(chrono::Utc::now() + chrono::TimeDelta::seconds(1)),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        for fragment in dataset.get_fragments() {
            let file = &fragment.metadata().files[0];
            let sidecar_dir = dataset
                .data_file_dir(file)
                .unwrap()
                .join(data_file_key_from_path(&file.path));
            let retained = dataset
                .object_store
                .read_dir_all(&sidecar_dir, None)
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            let retained_blob_bytes: u64 = retained
                .iter()
                .filter(|o| o.location.extension() == Some("blob"))
                .map(|o| o.size)
                .sum();
            assert_eq!(
                retained_blob_bytes,
                fragment.metadata().physical_rows.unwrap() as u64 * 16,
                "a replacement pack must contain only its own live values"
            );
        }
        let reopened = Arc::new(Dataset::open(dataset.uri()).await.unwrap());
        let values = reopened
            .take_blobs_by_indices(&(0..live_rows).collect::<Vec<_>>(), "blob")
            .await
            .unwrap();
        for (row, blob) in values.iter().enumerate() {
            assert_eq!(
                blob.as_ref().unwrap().read().await.unwrap().as_ref(),
                &[(500 + row) as u8; 16]
            );
        }
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
                .with_sidecar_rewrite(Arc::new(SidecarRewrite::default()));
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
