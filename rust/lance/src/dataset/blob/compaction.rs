// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(test)]
mod tests {
    use super::super::locate::data_file_key_from_path;
    use crate::BlobArrayBuilder;
    use crate::Dataset;
    use crate::dataset::blob::{BlobV2BatchRewritePlan, SidecarRewrite, prepared_blob_ids};
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
    use lance_core::{Error, utils::tempfile::TempStrDir};
    use lance_file::version::LanceFileVersion;
    use std::{
        collections::{HashMap, HashSet},
        sync::Arc,
    };

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
            task: crate::dataset::optimize::TaskData::rewrite_fragments(
                dataset.manifest.fragments.as_ref().clone(),
            ),
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
                .table_state
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
            .table_state
            .object_store
            .read_dir_all(&dataset.table_state.base, None)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let sidecar = before
            .iter()
            .find(|o| o.location.extension() == Some("blob"))
            .unwrap();
        dataset
            .table_state
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
            .table_state
            .object_store
            .read_dir_all(&dataset.table_state.base, None)
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
            task: crate::dataset::optimize::TaskData::rewrite_fragments(
                dataset.manifest.fragments.as_ref().clone(),
            ),
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
                .table_state
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
            .table_state
            .object_store
            .read_dir_all(&dataset.table_state.base, None)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        for object in objects
            .iter()
            .filter(|o| o.location.extension() == Some("blob"))
        {
            dataset
                .table_state
                .object_store
                .delete(&object.location)
                .await
                .unwrap();
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
