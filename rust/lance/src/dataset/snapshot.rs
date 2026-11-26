// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Capture, validate, and reopen lightweight references to table snapshots.

use lance_core::Result;
use lance_table_runtime::Table;

pub use lance_table::snapshot::{SnapshotRef, SnapshotVerification};

use crate::Dataset;

impl Dataset {
    fn snapshot_table(&self) -> Table {
        Table::from_parts_with_snapshot_digest(
            self.table_state.clone(),
            self.manifest.clone(),
            self.refs.clone(),
            self.snapshot_digest.clone(),
        )
    }

    /// Capture an execution reference using commit identity or a conditional-read
    /// ETag, falling back to a full-manifest digest if neither is available.
    /// The caller supplies and validates the table incarnation and authority.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let snapshot = dataset.snapshot_ref("catalog/table-incarnation").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn snapshot_ref(&self, table_identity: impl Into<String>) -> Result<SnapshotRef> {
        self.snapshot_table().snapshot_ref(table_identity).await
    }

    /// Capture a reference with explicit manifest verification requirements.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// use lance::dataset::snapshot::SnapshotVerification;
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let snapshot = dataset.snapshot_ref_with_verification(
    ///     "catalog/table-incarnation", SnapshotVerification::FullManifestDigest,
    /// ).await?;
    /// assert_eq!(snapshot.manifest_digest().len(), 32);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn snapshot_ref_with_verification(
        &self,
        table_identity: impl Into<String>,
        verification: SnapshotVerification,
    ) -> Result<SnapshotRef> {
        self.snapshot_table()
            .snapshot_ref_with_verification(table_identity, verification)
            .await
    }

    /// Validate the selected version, path, and verification identity.
    /// Full-digest references require a previously computed digest on this handle.
    /// Table incarnation and storage authority remain the trusted resolver's responsibility.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let snapshot = dataset.snapshot_ref("catalog/table-incarnation").await?;
    /// dataset.validate_snapshot(&snapshot)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate_snapshot(&self, snapshot: &SnapshotRef) -> Result<()> {
        self.snapshot_table().validate_snapshot(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{DatasetBuilder, WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use arrow_array::{
        ArrayRef, RecordBatch, RecordBatchIterator, builder::StringDictionaryBuilder, record_batch,
        types::Int8Type,
    };
    use byteorder::{ByteOrder, LittleEndian};
    use lance_core::Error;
    use lance_index::{IndexType, scalar::ScalarIndexParams};
    use lance_table::format::pb;
    use object_store::ObjectStoreExt;
    use prost::Message;
    use rstest::rstest;
    use std::sync::Arc;

    #[cfg_attr(coverage, coverage(off))]
    async fn write_dataset(uri: &str, max_rows_per_file: usize) -> Dataset {
        let batch = record_batch!(("id", Int32, [1, 2, 3, 4, 5, 6])).unwrap();
        Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            uri,
            Some(WriteParams {
                max_rows_per_file,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    #[cfg_attr(coverage, coverage(off))]
    async fn write_indexed_dataset(
        uri: &str,
        max_rows_per_file: usize,
        name: &str,
        is_outside_tail: bool,
    ) -> Dataset {
        let mut dataset = write_dataset(uri, max_rows_per_file).await;
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some(name.to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        if is_outside_tail {
            let padding = "x".repeat(2 * dataset.table_state.object_store.block_size());
            dataset
                .update_schema_metadata([("padding", padding.as_str())])
                .await
                .unwrap();
        }
        dataset
    }

    #[rstest]
    #[tokio::test]
    async fn lightweight_snapshot_preserves_tail_only_manifest_reads(
        #[values(
            SnapshotVerification::Identity,
            SnapshotVerification::FullManifestDigest
        )]
        verification: SnapshotVerification,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().join("table");
        let written =
            write_indexed_dataset(uri.to_str().unwrap(), 2, &"x".repeat(1 << 20), false).await;
        let raw = written
            .table_state
            .object_store
            .inner
            .get(&written.table_state.manifest_location.path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let opened = DatasetBuilder::from_uri(uri.to_str().unwrap())
            .with_block_size(4096)
            .with_metadata_cache_size_bytes(0)
            .load()
            .await
            .unwrap();
        let before = opened.table_state.object_store.io_stats_snapshot();
        let snapshot = opened
            .snapshot_ref_with_verification("catalog/table", verification)
            .await
            .unwrap();
        let captured = opened.table_state.object_store.io_stats_snapshot();
        match verification {
            SnapshotVerification::Identity => {
                assert!(snapshot.manifest_digest().is_empty());
                assert!(snapshot.manifest_etag().is_some());
                assert_eq!(captured.read_bytes, before.read_bytes);
            }
            SnapshotVerification::FullManifestDigest => {
                assert_eq!(snapshot.manifest_digest(), blake3::hash(&raw).as_bytes());
                assert_eq!(captured.read_bytes - before.read_bytes, raw.len() as u64);
            }
        }
        let worker = DatasetBuilder::from_snapshot(snapshot.clone())
            .with_block_size(4096)
            .with_metadata_cache_size_bytes(0)
            .load()
            .await
            .unwrap();
        worker.validate_snapshot(&snapshot).unwrap();
        let worker_bytes = worker
            .table_state
            .object_store
            .io_stats_snapshot()
            .read_bytes;
        match verification {
            SnapshotVerification::Identity => assert_eq!(worker_bytes, 4096),
            SnapshotVerification::FullManifestDigest => assert!(worker_bytes >= raw.len() as u64),
        }
    }

    #[rstest]
    #[tokio::test]
    async fn ordinary_open_defers_leading_manifest_io(
        #[values(0, 1 << 20)] metadata_cache_size_bytes: usize,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().join("table");
        let written =
            write_indexed_dataset(uri.to_str().unwrap(), 2, &"x".repeat(1 << 20), false).await;
        let path = written.table_state.manifest_location.path.clone();
        let raw = written
            .table_state
            .object_store
            .inner
            .get(&path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let opened = DatasetBuilder::from_uri(uri.to_str().unwrap())
            .with_block_size(4096)
            .with_metadata_cache_size_bytes(metadata_cache_size_bytes)
            .load()
            .await
            .unwrap();
        let before = opened.table_state.object_store.io_stats_snapshot();
        assert!(raw.len() > 1 << 20);
        assert!(opened.snapshot_digest.get().is_none());
        assert_eq!(before.read_bytes, 4096);
        assert_eq!(before.read_iops, 1);
        let snapshot = opened
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        let captured = opened.table_state.object_store.io_stats_snapshot();
        assert_eq!(captured.read_bytes - before.read_bytes, raw.len() as u64);
        assert_eq!(snapshot.manifest_digest(), blake3::hash(&raw).as_bytes());
        assert_eq!(
            opened
                .clone()
                .snapshot_ref_with_verification(
                    "catalog/table",
                    SnapshotVerification::FullManifestDigest
                )
                .await
                .unwrap(),
            snapshot
        );
        assert_eq!(
            opened
                .table_state
                .object_store
                .io_stats_snapshot()
                .requests
                .len(),
            captured.requests.len()
        );

        let worker = Arc::new(
            DatasetBuilder::from_uri(uri.to_str().unwrap())
                .with_block_size(4096)
                .with_metadata_cache_size_bytes(metadata_cache_size_bytes)
                .load()
                .await
                .unwrap(),
        );
        assert!(worker.snapshot_digest.get().is_none());
        let before = worker.table_state.object_store.io_stats_snapshot();
        let captured = worker
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        assert_eq!(captured, snapshot);
        worker.validate_snapshot(&snapshot).unwrap();
        let verified = worker.table_state.object_store.io_stats_snapshot();
        assert_eq!(verified.read_iops - before.read_iops, 1);
        worker
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        assert_eq!(
            worker
                .table_state
                .object_store
                .io_stats_snapshot()
                .read_iops,
            verified.read_iops
        );
    }

    #[tokio::test]
    async fn deferred_capture_rejects_replaced_manifest_generation() {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().join("table");
        let written =
            write_indexed_dataset(uri.to_str().unwrap(), 2, &"x".repeat(1 << 20), false).await;
        let opened = Dataset::open(uri.to_str().unwrap()).await.unwrap();
        assert!(opened.snapshot_digest.get().is_none());
        let path = &written.table_state.manifest_location.path;
        let mut bytes = written
            .table_state
            .object_store
            .inner
            .get(path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap()
            .to_vec();
        bytes[0] ^= 1;
        written
            .table_state
            .object_store
            .inner
            .put(path, bytes.into())
            .await
            .unwrap();
        let error = opened
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("no longer matches the loaded snapshot")
        );
        assert!(opened.snapshot_digest.get().is_none());
    }

    #[rstest]
    #[case::index_metadata(false)]
    #[case::inline_transaction(true)]
    #[tokio::test]
    async fn changed_outer_section_rejects_old_snapshot(
        #[case] is_transaction: bool,
        #[values(false, true)] is_outside_tail: bool,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().join("table");
        let dataset =
            write_indexed_dataset(uri.to_str().unwrap(), 2, "old_idx", is_outside_tail).await;
        let snapshot = dataset
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        let path = &dataset.table_state.manifest_location.path;
        let original = dataset
            .table_state
            .object_store
            .inner
            .get(path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(
            snapshot.manifest_digest(),
            blake3::hash(&original).as_bytes()
        );
        let manifest_offset = LittleEndian::read_u64(&original[original.len() - 16..]) as usize;
        let mut changed = original.to_vec();
        let section_offset = if is_transaction {
            dataset.manifest.transaction_section.unwrap()
        } else {
            dataset.manifest.index_section.unwrap()
        };
        assert_eq!(
            original.len() - section_offset > dataset.table_state.object_store.block_size(),
            is_outside_tail
        );
        let length = LittleEndian::read_u32(&original[section_offset..]) as usize;
        let section_range = section_offset + 4..section_offset + 4 + length;
        let replacement = if is_transaction {
            let mut transaction =
                pb::Transaction::decode(&original[section_range.clone()]).unwrap();
            transaction.uuid = uuid::Uuid::new_v4().to_string();
            transaction.encode_to_vec()
        } else {
            let mut section = pb::IndexSection::decode(&original[section_range.clone()]).unwrap();
            section.indices[0].name = "new_idx".into();
            section.encode_to_vec()
        };
        assert_eq!(replacement.len(), section_range.len());
        changed[section_range].copy_from_slice(&replacement);
        assert_eq!(&changed[manifest_offset..], &original[manifest_offset..]);
        dataset
            .table_state
            .object_store
            .inner
            .put(path, changed.into())
            .await
            .unwrap();

        let reopened = Dataset::open(uri.to_str().unwrap()).await.unwrap();
        if !is_transaction {
            assert_eq!(reopened.load_indices().await.unwrap()[0].name, "new_idx");
        }
        assert_ne!(
            reopened
                .snapshot_ref_with_verification(
                    "catalog/table",
                    SnapshotVerification::FullManifestDigest
                )
                .await
                .unwrap()
                .manifest_digest(),
            snapshot.manifest_digest()
        );
        let error = DatasetBuilder::from_snapshot(snapshot)
            .with_session(dataset.session())
            .load()
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("does not match dataset snapshot"),
            "{error}"
        );
    }

    #[rstest]
    #[case::main(None)]
    #[case::nested_branch(Some("experiment/nested"))]
    #[tokio::test]
    async fn transport_reopens_exact_snapshot(
        #[case] branch: Option<&str>,
        #[values(0, 1 << 20)] metadata_cache_size_bytes: usize,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().join("table");
        let mut main = write_dataset(uri.to_str().unwrap(), 2).await;
        let dataset = if let Some(branch) = branch {
            main.create_branch(branch, main.manifest.version, None)
                .await
                .unwrap()
        } else {
            main
        };
        assert_eq!(dataset.get_fragments().len(), 3);
        dataset
            .tags()
            .create("selected", dataset.manifest.version)
            .await
            .unwrap();
        let dataset = dataset.checkout_version("selected").await.unwrap();
        let snapshot = dataset
            .snapshot_ref_with_verification(
                "catalog/incarnation",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        assert_eq!(snapshot.branch(), branch);
        assert!(snapshot.encode().len() < 1024);
        let snapshot = SnapshotRef::decode(&snapshot.encode()).unwrap();
        let batch = record_batch!(("id", Int32, [7])).unwrap();
        let mut latest = dataset.clone();
        latest
            .append(
                RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        assert!(latest.version().version > snapshot.version());
        assert!(latest.snapshot_digest.get().is_none());
        latest
            .tags()
            .update("selected", latest.manifest.version)
            .await
            .unwrap();
        let worker = DatasetBuilder::from_snapshot(snapshot.clone())
            .with_metadata_cache_size_bytes(metadata_cache_size_bytes)
            .load()
            .await
            .unwrap();
        assert_eq!(worker.version().version, dataset.version().version);
        assert_eq!(worker.count_rows(None).await.unwrap(), 6);
        assert_eq!(
            worker
                .snapshot_ref_with_verification(
                    "catalog/incarnation",
                    SnapshotVerification::FullManifestDigest
                )
                .await
                .unwrap(),
            snapshot
        );
        let ordinary = DatasetBuilder::from_uri(snapshot.table_uri())
            .with_branch(
                snapshot.branch().unwrap_or("main"),
                Some(snapshot.version()),
            )
            .with_metadata_cache_size_bytes(metadata_cache_size_bytes)
            .load()
            .await
            .unwrap();
        ordinary.validate_snapshot(&snapshot).unwrap();
        let before_capture = ordinary
            .table_state
            .object_store
            .io_stats_snapshot()
            .requests
            .len();
        assert_eq!(
            ordinary
                .clone()
                .snapshot_ref_with_verification(
                    "catalog/incarnation",
                    SnapshotVerification::FullManifestDigest
                )
                .await
                .unwrap(),
            snapshot
        );
        assert_eq!(
            ordinary
                .table_state
                .object_store
                .io_stats_snapshot()
                .requests
                .len(),
            before_capture
        );
        let latest_snapshot = latest
            .snapshot_ref_with_verification(
                "catalog/incarnation",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        assert_ne!(
            latest_snapshot.manifest_digest(),
            snapshot.manifest_digest()
        );
        let mut updated = ordinary.clone();
        updated.checkout_latest().await.unwrap();
        updated.validate_snapshot(&latest_snapshot).unwrap();
        dataset.validate_snapshot(&snapshot).unwrap();
        let error = latest.validate_snapshot(&snapshot).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("does not match dataset snapshot"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn transport_size_is_independent_of_fragment_count() {
        let directory = tempfile::tempdir().unwrap();
        let one = write_dataset(directory.path().join("one").to_str().unwrap(), 6).await;
        let six = write_dataset(directory.path().join("six").to_str().unwrap(), 1).await;
        assert_eq!(one.get_fragments().len(), 1);
        assert_eq!(six.get_fragments().len(), 6);
        let one = one
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        let six = six
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        assert_eq!(one.encode().len(), six.encode().len());
        assert_ne!(one.manifest_digest(), six.manifest_digest());
    }

    #[rstest]
    #[tokio::test]
    async fn stale_manifest_size_does_not_change_digest(#[values(-1, 1)] size_delta: i64) {
        let directory = tempfile::tempdir().unwrap();
        let dataset = write_dataset(directory.path().join("table").to_str().unwrap(), 2).await;
        let snapshot = dataset
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        let mut location = dataset.table_state.manifest_location.clone();
        location.size = Some(
            dataset
                .table_state
                .object_store
                .size(&location.path)
                .await
                .unwrap()
                .checked_add_signed(size_delta)
                .unwrap(),
        );
        let (manifest, digest) = Table::load_manifest_with_digest(
            &dataset.table_state.object_store,
            &location,
            dataset.uri(),
            &dataset.session.table_session,
        )
        .await
        .unwrap();
        assert_eq!(manifest, *dataset.manifest);
        assert_eq!(digest.as_slice(), snapshot.manifest_digest());
    }

    #[tokio::test]
    async fn dictionary_snapshot_capture_matches_reopened_file() {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().join("table");
        let mut dictionary = StringDictionaryBuilder::<Int8Type>::new();
        for value in ["one", "two", "one", "two", "one", "two"] {
            dictionary.append(value).unwrap();
        }
        let batch =
            RecordBatch::try_from_iter([("category", Arc::new(dictionary.finish()) as ArrayRef)])
                .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            uri.to_str().unwrap(),
            Some(WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let snapshot = dataset
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        let reopened = DatasetBuilder::from_snapshot(snapshot.clone())
            .load()
            .await
            .unwrap();
        assert_eq!(reopened.get_fragments().len(), 3);
        assert_eq!(reopened.count_rows(None).await.unwrap(), 6);
        assert_eq!(
            reopened
                .snapshot_ref_with_verification(
                    "catalog/table",
                    SnapshotVerification::FullManifestDigest
                )
                .await
                .unwrap(),
            snapshot
        );
    }

    #[tokio::test]
    async fn missing_snapshot_never_falls_back_to_latest() {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().join("table");
        let dataset = write_dataset(uri.to_str().unwrap(), 2).await;
        let snapshot = dataset
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        let batch = record_batch!(("id", Int32, [7])).unwrap();
        let mut latest = dataset.clone();
        latest
            .append(
                RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                None,
            )
            .await
            .unwrap();
        dataset
            .table_state
            .object_store
            .delete(&dataset.table_state.manifest_location.path)
            .await
            .unwrap();
        // Capturing an already observed identity does not reread its manifest or
        // promise availability; reopening must still observe the missing version.
        assert_eq!(
            dataset
                .clone()
                .snapshot_ref_with_verification(
                    "catalog/table",
                    SnapshotVerification::FullManifestDigest
                )
                .await
                .unwrap(),
            snapshot
        );
        let error = DatasetBuilder::from_snapshot(snapshot)
            .with_session(latest.session())
            .load()
            .await
            .unwrap_err();
        assert!(
            matches!(
                error,
                Error::DatasetNotFound { .. }
                    | Error::VersionNotFound { .. }
                    | Error::NotFound { .. }
            ),
            "{error}"
        );
        assert!(
            error.to_string().contains("manifest") || error.to_string().contains("version"),
            "{error}"
        );
    }

    #[rstest]
    #[case::fresh_session(false)]
    #[case::cached_session(true)]
    #[tokio::test]
    async fn recreated_table_rejects_old_snapshot(#[case] has_cached_session: bool) {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().join("table");
        let dataset = write_dataset(uri.to_str().unwrap(), 2).await;
        let snapshot = dataset
            .snapshot_ref_with_verification(
                "catalog/old-incarnation",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        // Populate the ordinary read cache too, before recreating the same URI/version.
        let cached = DatasetBuilder::from_uri(uri.to_str().unwrap())
            .with_session(dataset.session())
            .load()
            .await
            .unwrap();
        std::fs::remove_dir_all(&uri).unwrap();
        let recreated = write_dataset(uri.to_str().unwrap(), 3).await;
        assert_eq!(dataset.version().version, recreated.version().version);
        let builder = DatasetBuilder::from_snapshot(snapshot);
        let builder = if has_cached_session {
            builder.with_session(cached.session())
        } else {
            builder
        };
        let error = builder.load().await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("does not match dataset snapshot"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn missing_generation_token_cannot_reuse_an_old_digest() {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().join("table");
        let mut dataset = write_indexed_dataset(uri.to_str().unwrap(), 2, "old_idx", true).await;
        let mut location = dataset.table_state.manifest_location.clone();
        location.e_tag = None;
        location.size = Some(
            dataset
                .table_state
                .object_store
                .size(&location.path)
                .await
                .unwrap(),
        );
        let (old_manifest, old_digest) = Table::get_manifest_with_digest(
            &dataset.table_state.object_store,
            &location,
            dataset.uri(),
            &dataset.session.table_session,
        )
        .await
        .unwrap();
        let index_offset = old_manifest.index_section.unwrap();
        assert!(
            location.size.unwrap() as usize - index_offset
                > dataset.table_state.object_store.block_size()
        );
        dataset.manifest = old_manifest.clone();
        dataset.table_state.manifest_location = location.clone();
        dataset.snapshot_digest = old_digest.clone();
        let old_snapshot = dataset
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        assert_eq!(dataset.load_indices().await.unwrap()[0].name, "old_idx");
        std::fs::remove_dir_all(&uri).unwrap();
        let recreated = write_indexed_dataset(uri.to_str().unwrap(), 3, "new_idx", true).await;
        location.size = Some(
            recreated
                .table_state
                .object_store
                .size(&location.path)
                .await
                .unwrap(),
        );
        let (new_manifest, new_digest) = Table::get_manifest_with_digest(
            &recreated.table_state.object_store,
            &location,
            dataset.uri(),
            &dataset.session.table_session,
        )
        .await
        .unwrap();
        assert!(!Arc::ptr_eq(&old_manifest, &new_manifest));
        assert!(new_digest.get().is_none());
        assert_eq!(new_manifest.fragments.len(), 2);
        dataset.manifest = new_manifest;
        dataset.table_state.manifest_location = location;
        dataset.snapshot_digest = new_digest;
        assert_ne!(
            dataset
                .snapshot_ref_with_verification(
                    "catalog/table",
                    SnapshotVerification::FullManifestDigest
                )
                .await
                .unwrap()
                .manifest_digest(),
            old_snapshot.manifest_digest()
        );
        assert_eq!(dataset.load_indices().await.unwrap()[0].name, "new_idx");
    }

    #[rstest]
    #[case::store("store")]
    #[case::same_prefix_different_store("store_handle")]
    #[case::manifest_path("path")]
    #[case::commit_identity("identity")]
    #[tokio::test]
    async fn digest_cache_is_scoped_to_the_resolved_manifest(#[case] changed: &str) {
        let directory = tempfile::tempdir().unwrap();
        let first = write_dataset(directory.path().join("one").to_str().unwrap(), 2).await;
        let second = write_dataset(directory.path().join("two").to_str().unwrap(), 3).await;
        let first_bytes = first
            .table_state
            .object_store
            .inner
            .get(&first.table_state.manifest_location.path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let second_bytes = second
            .table_state
            .object_store
            .inner
            .get(&second.table_state.manifest_location.path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let mut store = lance_io::object_store::ObjectStore::memory();
        store.store_prefix = "endpoint-a".into();
        let mut location = first.table_state.manifest_location.clone();
        location.path = object_store::path::Path::from("table/_versions/1.manifest");
        location.size = Some(first_bytes.len() as u64);
        location.identity = Some("record-a".into());
        location.e_tag = store
            .inner
            .put(&location.path, first_bytes.into())
            .await
            .unwrap()
            .e_tag;
        let first_store = store.clone();
        let mut cached_location = location.clone();
        let (first_manifest, first_digest) = Table::get_manifest_with_digest(
            &store,
            &location,
            "memory://table",
            &first.session.table_session,
        )
        .await
        .unwrap();
        match changed {
            "store" => {
                store = lance_io::object_store::ObjectStore::memory();
                store.store_prefix = "endpoint-b".into();
            }
            "store_handle" => {
                store = lance_io::object_store::ObjectStore::memory();
                store.store_prefix = "endpoint-a".into();
            }
            "path" => location.path = object_store::path::Path::from("staging/1.manifest"),
            "identity" => location.identity = Some("record-b".into()),
            _ => unreachable!(),
        }
        location.size = Some(second_bytes.len() as u64);
        location.e_tag = store
            .inner
            .put(&location.path, second_bytes.into())
            .await
            .unwrap()
            .e_tag;
        // Isolate the provenance dimension even if this backend assigns B a
        // different ETag. The cached allocation still belongs to A's location/store.
        cached_location.e_tag = location.e_tag.clone();
        let cache = first
            .session
            .table_session
            .metadata_cache()
            .for_dataset("memory://table");
        lance_table_runtime::caches::cache_manifest(
            &cache,
            &first_store,
            &cached_location,
            first_manifest.clone(),
            first_digest.clone(),
        )
        .await;
        let (second_manifest, second_digest) = Table::get_manifest_with_digest(
            &store,
            &location,
            "memory://table",
            &first.session.table_session,
        )
        .await
        .unwrap();
        assert!(!Arc::ptr_eq(&first_manifest, &second_manifest));
        assert_eq!(first_manifest.fragments.len(), 3);
        assert_eq!(second_manifest.fragments.len(), 2);
        assert_ne!(first_digest.get().unwrap(), second_digest.get().unwrap());
    }

    #[rstest]
    #[case::digest("digest")]
    #[case::manifest_path("path")]
    #[case::commit_identity("identity")]
    #[tokio::test]
    async fn altered_identity_is_rejected(#[case] field: &str) {
        let directory = tempfile::tempdir().unwrap();
        let dataset = write_dataset(directory.path().join("table").to_str().unwrap(), 2).await;
        let snapshot = dataset
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        let mut proto =
            lance_table::snapshot::pb::SnapshotRef::decode(snapshot.encode().as_slice()).unwrap();
        match field {
            "digest" => proto.manifest_digest[0] ^= 1,
            "path" => proto.manifest_path.push_str(".different"),
            "identity" => proto.manifest_identity = Some("different-record".into()),
            _ => unreachable!(),
        }
        let altered = SnapshotRef::decode(&proto.encode_to_vec()).unwrap();
        let error = DatasetBuilder::from_snapshot(altered)
            .load()
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("does not match dataset snapshot"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn serialized_manifest_reads_remain_independent_of_snapshot_capture() {
        let directory = tempfile::tempdir().unwrap();
        let dataset = write_dataset(directory.path().join("table").to_str().unwrap(), 2).await;
        let serialized = pb::Manifest::from(dataset.manifest()).encode_to_vec();
        dataset
            .table_state
            .object_store
            .delete(&dataset.table_state.manifest_location.path)
            .await
            .unwrap();
        let restored = DatasetBuilder::from_uri(dataset.uri())
            .with_serialized_manifest(&serialized)
            .unwrap()
            .load()
            .await
            .unwrap();
        assert_eq!(restored.count_rows(None).await.unwrap(), 6);
        assert!(restored.snapshot_digest.get().is_none());
        let error = restored
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::DatasetNotFound { .. }), "{error}");
        assert!(error.to_string().contains("manifest"), "{error}");
    }

    #[rstest]
    #[tokio::test]
    async fn serialized_manifest_cannot_capture_another_generation(
        #[values(false, true)] is_replaced: bool,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let dataset = write_dataset(directory.path().join("table").to_str().unwrap(), 2).await;
        let serialized = pb::Manifest::from(dataset.manifest()).encode_to_vec();
        if is_replaced {
            let mut replacement = dataset.manifest().clone();
            replacement
                .schema
                .metadata
                .insert("generation".into(), "B".into());
            lance_table::io::commit::write_manifest_file_to_path(
                &dataset.table_state.object_store,
                &mut replacement,
                None,
                &dataset.table_state.manifest_location.path,
                None,
            )
            .await
            .unwrap();
        }
        let restored = DatasetBuilder::from_uri(dataset.uri())
            .with_serialized_manifest(&serialized)
            .unwrap()
            .load()
            .await
            .unwrap();
        let captured = restored.snapshot_ref("catalog/table").await;
        if is_replaced {
            let error = captured.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(
                error
                    .to_string()
                    .contains("no longer matches the loaded snapshot")
            );
        } else {
            let captured = captured.unwrap();
            assert_eq!(captured.manifest_digest().len(), 32);
            let worker = DatasetBuilder::from_snapshot(captured)
                .load()
                .await
                .unwrap();
            assert_eq!(worker.count_rows(None).await.unwrap(), 6);
        }
    }

    #[rstest]
    #[case::index(false)]
    #[case::transaction(true)]
    #[tokio::test]
    async fn lightweight_snapshot_guards_deferred_manifest_sections(
        #[case] is_transaction: bool,
        #[values(false, true)] has_stale_size: bool,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let dataset = write_indexed_dataset(
            directory.path().join("table").to_str().unwrap(),
            2,
            "old_idx",
            true,
        )
        .await;
        let source = dataset.snapshot_ref("catalog/table").await.unwrap();
        assert!(source.manifest_digest().is_empty());
        let mut worker = DatasetBuilder::from_snapshot(source)
            .with_block_size(4096)
            .load()
            .await
            .unwrap();
        let location = &dataset.table_state.manifest_location;
        let original = dataset
            .table_state
            .object_store
            .inner
            .get(&location.path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let offset = if is_transaction {
            dataset.manifest.transaction_section.unwrap()
        } else {
            dataset.manifest.index_section.unwrap()
        };
        assert!(original.len() - offset > 4096);
        let length = LittleEndian::read_u32(&original[offset..]) as usize;
        let range = offset + 4..offset + 4 + length;
        let new_uuid = uuid::Uuid::new_v4().to_string();
        let replacement = if is_transaction {
            let mut tx = pb::Transaction::decode(&original[range.clone()]).unwrap();
            tx.uuid = new_uuid.clone();
            tx.encode_to_vec()
        } else {
            let mut section = pb::IndexSection::decode(&original[range.clone()]).unwrap();
            section.indices[0].name = "new_idx".into();
            section.encode_to_vec()
        };
        assert_eq!(replacement.len(), range.len());
        let mut changed = original.to_vec();
        changed[range].copy_from_slice(&replacement);
        dataset
            .table_state
            .object_store
            .inner
            .put(&location.path, changed.into())
            .await
            .unwrap();

        // Populate this session with B first. A must not reuse B's cached metadata.
        let current = DatasetBuilder::from_uri(dataset.uri())
            .with_session(worker.session())
            .load()
            .await
            .unwrap();
        if is_transaction {
            assert_eq!(
                current
                    .read_transaction()
                    .await
                    .unwrap()
                    .unwrap()
                    .uuid
                    .to_string(),
                new_uuid
            );
        } else {
            assert_eq!(current.load_indices().await.unwrap()[0].name, "new_idx");
        }
        if has_stale_size {
            worker.table_state.manifest_location.size = Some(1);
        }
        for _ in 0..2 {
            let error = if is_transaction {
                worker.read_transaction().await.unwrap_err()
            } else {
                worker.load_indices().await.unwrap_err()
            };
            assert!(matches!(error, Error::IO { .. }), "{error}");
            assert!(
                error
                    .to_string()
                    .to_ascii_lowercase()
                    .contains("precondition"),
                "{error}"
            );
        }
    }

    #[rstest]
    #[case::version("version")]
    #[case::branch("branch")]
    #[case::tag("tag")]
    #[case::manifest("manifest")]
    #[tokio::test]
    async fn conflicting_selectors_are_rejected(#[case] selector: &str) {
        let directory = tempfile::tempdir().unwrap();
        let dataset = write_dataset(directory.path().join("table").to_str().unwrap(), 2).await;
        let snapshot = dataset
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        let builder = DatasetBuilder::from_snapshot(snapshot);
        let builder = match selector {
            "version" => builder.with_version(1),
            "branch" => builder.with_branch("other", Some(1)),
            "tag" => builder.with_tag("tag"),
            "manifest" => builder
                .with_serialized_manifest(&pb::Manifest::from(dataset.manifest()).encode_to_vec())
                .unwrap(),
            _ => unreachable!(),
        };
        let error = builder.load().await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("cannot be combined"), "{error}");
    }
}
