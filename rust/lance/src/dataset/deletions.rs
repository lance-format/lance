// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Deletion vectors stored in the manifest instead of in `_deletions/` files.

use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_table::format::{DeletionFileType, Fragment};
use lance_table::transaction::{Operation, Transaction};
use roaring::RoaringBitmap;

use super::write::CommitBuilder;
use crate::Dataset;
use crate::io::deletion::read_dataset_deletion_file;

/// Table config key: the largest bitmap encoding, in bytes, of a deletion
/// vector that writers keep in the manifest rather than in a file under
/// `_deletions/`. Unset or `0` always writes a file.
///
/// A reader pays one request per deletion file it needs, so a table whose
/// fragments each lost a few rows pays one request per fragment on its first
/// take; inline vectors come with the manifest instead.
pub const INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY: &str = "lance.deletions.inline_max_bytes";

impl Dataset {
    /// The inline budget from [`INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY`], `0` when unset.
    pub fn inline_deletions_max_bytes(&self) -> Result<usize> {
        let Some(value) = self
            .manifest
            .config
            .get(INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY)
        else {
            return Ok(0);
        };
        value.trim().parse().map_err(|err| {
            Error::invalid_input(format!(
                "table config {INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY} must be a number of bytes, got `{value}`: {err}"
            ))
        })
    }

    /// Moves every deletion vector that fits the inline budget from its file
    /// into the manifest, in one commit, and returns how many fragments changed.
    ///
    /// Only the deletion file of a fragment changes; its data files, row ids
    /// and the indices that cover it stay as they are. The old deletion files
    /// become unreferenced and are removed by a later cleanup. Requires the
    /// budget to be set in the table config, see
    /// [`INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY`].
    ///
    /// Snapshots that still contain an inline vector carry
    /// `FLAG_INLINE_DELETIONS`. The move is one way: setting the budget back to
    /// `0` only stops new inline writes and does not externalize vectors that
    /// are already inline. Upgrade every reader of the table before calling this.
    pub async fn inline_deletions(&mut self) -> Result<usize> {
        let max_bytes = self.inline_deletions_max_bytes()?;
        if max_bytes == 0 {
            return Err(Error::invalid_input(format!(
                "set table config {INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY} to the inline budget in bytes first"
            )));
        }
        let candidates = self
            .manifest
            .fragments
            .iter()
            .filter(|fragment| {
                fragment
                    .deletion_file
                    .as_ref()
                    .is_some_and(|file| !file.is_inline())
            })
            .cloned()
            .collect::<Vec<_>>();
        let dataset: &Self = self;
        let updated_fragments =
            futures::stream::iter(candidates)
                .map(|fragment| async move {
                    dataset.inline_fragment_deletions(fragment, max_bytes).await
                })
                .buffer_unordered(self.object_store.io_parallelism())
                .try_filter_map(|fragment| async move { Ok(fragment) })
                .try_collect::<Vec<_>>()
                .await?;
        let changed = updated_fragments.len();
        if changed == 0 {
            return Ok(0);
        }
        let operation = Operation::Delete {
            updated_fragments,
            deleted_fragment_ids: vec![],
            predicate: "inline deletions".to_string(),
        };
        let transaction = Transaction::new(self.manifest.version, operation, None);
        *self = CommitBuilder::new(Arc::new(self.clone()))
            .execute(transaction)
            .await?;
        Ok(changed)
    }

    /// `fragment` with its deletion vector moved inline when the bitmap
    /// encoding fits `max_bytes`, else `None`.
    async fn inline_fragment_deletions(
        &self,
        mut fragment: Fragment,
        max_bytes: usize,
    ) -> Result<Option<Fragment>> {
        let Some(deletion_file) = fragment.deletion_file.as_mut() else {
            return Ok(None);
        };
        let deletion_vector = read_dataset_deletion_file(self, fragment.id, deletion_file).await?;
        let bitmap = RoaringBitmap::from(deletion_vector.as_ref());
        let encoded_size = bitmap.serialized_size();
        if encoded_size > max_bytes {
            return Ok(None);
        }
        let mut encoded = Vec::with_capacity(encoded_size);
        bitmap.serialize_into(&mut encoded)?;
        deletion_file.file_type = DeletionFileType::Bitmap;
        deletion_file.num_deleted_rows = Some(bitmap.len() as usize);
        deletion_file.base_id = None;
        deletion_file.inline = Some(encoded.into());
        Ok(Some(fragment))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, cast::AsArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;
    use lance_table::feature_flags::FLAG_INLINE_DELETIONS;

    use crate::dataset::WriteParams;
    use crate::index::DatasetIndexExt;

    async fn dataset_with_deletions(
        inline_max_bytes: Option<&str>,
    ) -> (tempfile::TempDir, Dataset) {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..40))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        let params = WriteParams {
            max_rows_per_file: 10,
            ..Default::default()
        };
        // On disk, so the dataset can be reopened with empty caches.
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap().to_string();
        let mut dataset = Dataset::write(reader, &uri, Some(params)).await.unwrap();
        if let Some(max_bytes) = inline_max_bytes {
            dataset
                .update_config([(
                    INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY.to_string(),
                    max_bytes.to_string(),
                )])
                .await
                .unwrap();
        }
        dataset.delete("i in (3, 17, 18, 39)").await.unwrap();
        (dir, dataset)
    }

    fn deletion_file_requests(dataset: &Dataset) -> usize {
        dataset
            .object_store
            .io_stats_incremental()
            .requests
            .iter()
            .filter(|request| request.path.as_ref().contains("_deletions"))
            .count()
    }

    async fn live_values(dataset: &Dataset) -> Vec<i32> {
        dataset
            .scan()
            .try_into_batch()
            .await
            .unwrap()
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .values()
            .to_vec()
    }

    /// With a budget in the table config a delete keeps its vectors in the
    /// manifest: the flag is set, nothing lands in `_deletions/`, and reads
    /// need no deletion-file request.
    #[tokio::test]
    async fn test_delete_writes_inline_within_budget() {
        let (_dir, dataset) = dataset_with_deletions(Some("4096")).await;
        assert!(
            dataset
                .manifest
                .fragments
                .iter()
                .filter_map(|fragment| fragment.deletion_file.as_ref())
                .all(|file| file.is_inline())
        );
        assert_ne!(
            dataset.manifest.reader_feature_flags & FLAG_INLINE_DELETIONS,
            0
        );
        assert_ne!(
            dataset.manifest.writer_feature_flags & FLAG_INLINE_DELETIONS,
            0
        );
        assert!(
            dataset
                .object_store
                .read_dir(dataset.base.clone().join("_deletions"))
                .await
                .unwrap_or_default()
                .is_empty()
        );

        let reopened = Dataset::open(&dataset.uri).await.unwrap();
        reopened.object_store.io_stats_incremental();
        let values = live_values(&reopened).await;
        assert_eq!(values.len(), 36);
        assert!(!values.iter().any(|v| [3, 17, 18, 39].contains(v)));
        assert_eq!(deletion_file_requests(&reopened), 0);
    }

    /// Without a budget, or with one too small, the file is written as before.
    #[tokio::test]
    async fn test_delete_writes_file_without_budget() {
        for budget in [None, Some("1")] {
            let (_dir, dataset) = dataset_with_deletions(budget).await;
            assert!(
                dataset
                    .manifest
                    .fragments
                    .iter()
                    .filter_map(|fragment| fragment.deletion_file.as_ref())
                    .all(|file| !file.is_inline())
            );
            assert_eq!(
                dataset.manifest.reader_feature_flags & FLAG_INLINE_DELETIONS,
                0
            );
        }
    }

    #[tokio::test]
    async fn test_invalid_budget_is_rejected() {
        let (_dir, mut dataset) = dataset_with_deletions(None).await;
        dataset
            .update_config([(
                INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY.to_string(),
                "4KiB".to_string(),
            )])
            .await
            .unwrap();
        let error = dataset.delete("i = 0").await.unwrap_err().to_string();
        assert!(error.contains("4KiB"), "{error}");
    }

    /// Existing deletion files move into the manifest in one commit that
    /// changes nothing else, so the same rows stay deleted and reads stop
    /// requesting deletion files.
    #[tokio::test]
    async fn test_inline_deletions_moves_existing_files() {
        let (_dir, mut dataset) = dataset_with_deletions(None).await;
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        assert!(
            dataset.inline_deletions().await.is_err(),
            "the budget must be set first"
        );
        dataset
            .update_config([(
                INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY.to_string(),
                "4096".to_string(),
            )])
            .await
            .unwrap();
        let version_before = dataset.manifest.version;
        let fragments_before = dataset.manifest.fragments.clone();

        assert_eq!(dataset.inline_deletions().await.unwrap(), 3);
        assert_eq!(dataset.manifest.version, version_before + 1);
        assert_eq!(
            dataset.inline_deletions().await.unwrap(),
            0,
            "nothing left to move"
        );
        assert_eq!(dataset.manifest.version, version_before + 1);

        for (before, after) in fragments_before
            .iter()
            .zip(dataset.manifest.fragments.iter())
        {
            assert_eq!(before.id, after.id);
            assert_eq!(before.files, after.files);
            assert_eq!(
                before.deletion_file.is_some(),
                after
                    .deletion_file
                    .as_ref()
                    .is_some_and(|file| file.is_inline())
            );
        }
        assert_ne!(
            dataset.manifest.reader_feature_flags & FLAG_INLINE_DELETIONS,
            0
        );
        assert_eq!(
            dataset.load_indices().await.unwrap().len(),
            1,
            "only deletion files change, so the index stays"
        );

        let reopened = Dataset::open(&dataset.uri).await.unwrap();
        reopened.object_store.io_stats_incremental();
        let values = live_values(&reopened).await;
        assert_eq!(values.len(), 36);
        assert!(!values.iter().any(|v| [3, 17, 18, 39].contains(v)));
        assert_eq!(deletion_file_requests(&reopened), 0);
    }
}
