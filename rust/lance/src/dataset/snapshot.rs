// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Capture, validate, and reopen lightweight references to table snapshots.

use std::sync::Arc;

use lance_core::{Error, Result};
use lance_table::format::Manifest;

pub use lance_table::snapshot::SnapshotRef;

use crate::Dataset;
use crate::session::caches::{ManifestDigestKey, cache_manifest};

impl Dataset {
    /// Capture a compact, portable reference to this exact snapshot.
    ///
    /// `table_identity` identifies the table incarnation and catalog/storage
    /// authority. It must distinguish tables recreated at the same URI. Workers
    /// supply their own authenticated storage settings when reopening the reference.
    ///
    /// Manifest digests are computed once per loaded snapshot from original bytes.
    /// A dataset returned by a write may read its manifest on the first capture;
    /// subsequent captures share the digest. No data payloads are read. The
    /// reference does not acquire retention or guarantee continued file availability.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// use lance::dataset::{builder::DatasetBuilder, snapshot::SnapshotRef};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let snapshot = dataset.snapshot_ref("catalog/table-incarnation").await?;
    /// let transported = SnapshotRef::decode(&snapshot.encode())?;
    /// let worker = DatasetBuilder::from_snapshot(transported).load().await?;
    /// assert_eq!(dataset.version().version, worker.version().version);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn snapshot_ref(&self, table_identity: impl Into<String>) -> Result<SnapshotRef> {
        let table_identity = table_identity.into();
        if table_identity.trim().is_empty() {
            return Err(Error::invalid_input(
                "Snapshot table_identity cannot be empty",
            ));
        }
        if self.manifest.version != self.manifest_location.version {
            return Err(Error::invalid_input(format!(
                "Loaded manifest version {} does not match resolved version {}",
                self.manifest.version, self.manifest_location.version,
            )));
        }
        let digest = self.ensure_snapshot_digest().await?;
        SnapshotRef::try_new(
            table_identity,
            &self.refs.root()?.uri,
            self.manifest.branch.clone(),
            &self.manifest_location,
            digest,
        )
    }

    /// Check a reference against this dataset's exact manifest and commit identity.
    ///
    /// Datasets opened with [`super::builder::DatasetBuilder::from_snapshot`] have a verified
    /// digest. For handles returned by writes, served from writer metadata caches,
    /// or loaded from serialized manifests, first call [`Self::snapshot_ref`]. Table incarnation authority and credentials must be
    /// verified by the caller's catalog/storage resolver.
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
        let mismatch = || {
            Error::invalid_input(format!(
                "Snapshot {:?}/{} at {} does not match dataset snapshot {:?}/{} at {} (manifest {}, expected {})",
                snapshot.branch(),
                snapshot.version(),
                snapshot.table_uri(),
                self.manifest.branch,
                self.manifest.version,
                self.uri(),
                self.manifest_location.path,
                snapshot.manifest_path(),
            ))
        };
        if snapshot.validate_table_uri(&self.refs.root()?.uri).is_err()
            || self.manifest.branch.as_deref() != snapshot.branch()
            || self.manifest.version != snapshot.version()
            || self.manifest_location.version != snapshot.version()
            || self.manifest_location.path.as_ref() != snapshot.manifest_path()
            || self.manifest_location.identity.as_deref() != snapshot.manifest_identity()
        {
            return Err(mismatch());
        }
        let digest = self.snapshot_digest.get().ok_or_else(|| {
            Error::invalid_input("Dataset snapshot digest is unavailable; call snapshot_ref or reopen with DatasetBuilder::from_snapshot before binding a transported reference")
        })?;
        if digest.as_slice() != snapshot.manifest_digest() {
            return Err(mismatch());
        }

        Ok(())
    }

    pub(crate) async fn ensure_snapshot_digest(&self) -> Result<[u8; 32]> {
        self.snapshot_digest
            .get_or_try_init(|| async {
                let key = ManifestDigestKey {
                    version: self.manifest.version,
                    store_identity: &self.object_store.store_prefix,
                    manifest_path: &self.manifest_location.path,
                    manifest_identity: self.manifest_location.identity.as_deref(),
                    e_tag: self.manifest_location.e_tag.as_deref(),
                };
                if let Some(cached) = self.metadata_cache.get_with_key(&key).await
                    && cached.manifest.ptr_eq(&Arc::downgrade(&self.manifest))
                    && cached
                        .store
                        .ptr_eq(&Arc::downgrade(&self.object_store.inner))
                    && let Some(digest) = cached.digest.get()
                {
                    return Ok(*digest);
                }
                let (observed, digest) = Self::load_manifest_with_digest(
                    &self.object_store,
                    &self.manifest_location,
                    self.uri(),
                    &self.session,
                )
                .await?;
                if !matches_loaded_manifest(observed, &self.manifest) {
                    return Err(Error::invalid_input(format!(
                        "Manifest {} no longer matches the loaded snapshot at version {}",
                        self.manifest_location.path, self.manifest.version,
                    )));
                }
                cache_manifest(
                    &self.metadata_cache,
                    &self.object_store,
                    &self.manifest_location,
                    self.manifest.clone(),
                    self.snapshot_digest.clone(),
                )
                .await;
                Ok(digest)
            })
            .await
            .copied()
    }
}

fn matches_loaded_manifest(mut observed: Manifest, loaded: &Manifest) -> bool {
    // Writes and reads can hydrate different dictionary values. Compare the
    // persisted schema metadata; the digest covers any persisted dictionary bytes.
    let mut expected = loaded.clone();
    for manifest in [&mut observed, &mut expected] {
        let mut fields = manifest.schema.fields.iter_mut().collect::<Vec<_>>();
        while let Some(field) = fields.pop() {
            if let Some(dictionary) = &mut field.dictionary {
                dictionary.values = None;
            }
            fields.extend(&mut field.children);
        }
    }
    observed == expected
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
    use lance_index::{IndexType, scalar::ScalarIndexParams};
    use lance_table::format::pb;
    use object_store::ObjectStoreExt;
    use prost::Message;
    use rstest::rstest;

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
            let padding = "x".repeat(2 * dataset.object_store.block_size());
            dataset
                .update_schema_metadata([("padding", padding.as_str())])
                .await
                .unwrap();
        }
        dataset
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
        let snapshot = dataset.snapshot_ref("catalog/table").await.unwrap();
        let path = &dataset.manifest_location.path;
        let original = dataset
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
            original.len() - section_offset > dataset.object_store.block_size(),
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
                .snapshot_ref("catalog/table")
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
    async fn transport_reopens_exact_snapshot(#[case] branch: Option<&str>) {
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
        let snapshot = dataset.snapshot_ref("catalog/incarnation").await.unwrap();
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
        latest
            .tags()
            .update("selected", latest.manifest.version)
            .await
            .unwrap();
        let worker = DatasetBuilder::from_snapshot(snapshot.clone())
            .load()
            .await
            .unwrap();
        assert_eq!(worker.version().version, dataset.version().version);
        assert_eq!(worker.count_rows(None).await.unwrap(), 6);
        assert_eq!(
            worker.snapshot_ref("catalog/incarnation").await.unwrap(),
            snapshot
        );
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
        let one = one.snapshot_ref("catalog/table").await.unwrap();
        let six = six.snapshot_ref("catalog/table").await.unwrap();
        assert_eq!(one.encode().len(), six.encode().len());
        assert_ne!(one.manifest_digest(), six.manifest_digest());
    }

    #[rstest]
    #[tokio::test]
    async fn stale_manifest_size_does_not_change_digest(#[values(-1, 1)] size_delta: i64) {
        let directory = tempfile::tempdir().unwrap();
        let dataset = write_dataset(directory.path().join("table").to_str().unwrap(), 2).await;
        let snapshot = dataset.snapshot_ref("catalog/table").await.unwrap();
        let mut location = dataset.manifest_location.clone();
        location.size = Some(
            dataset
                .object_store
                .size(&location.path)
                .await
                .unwrap()
                .checked_add_signed(size_delta)
                .unwrap(),
        );
        let (manifest, digest) = Dataset::load_manifest_with_digest(
            &dataset.object_store,
            &location,
            dataset.uri(),
            &dataset.session,
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
        let snapshot = dataset.snapshot_ref("catalog/table").await.unwrap();
        let reopened = DatasetBuilder::from_snapshot(snapshot.clone())
            .load()
            .await
            .unwrap();
        assert_eq!(reopened.get_fragments().len(), 3);
        assert_eq!(reopened.count_rows(None).await.unwrap(), 6);
        assert_eq!(
            reopened.snapshot_ref("catalog/table").await.unwrap(),
            snapshot
        );
    }

    #[tokio::test]
    async fn missing_snapshot_never_falls_back_to_latest() {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().join("table");
        let dataset = write_dataset(uri.to_str().unwrap(), 2).await;
        let snapshot = dataset.snapshot_ref("catalog/table").await.unwrap();
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
            .object_store
            .delete(&dataset.manifest_location.path)
            .await
            .unwrap();
        // Capturing an already observed identity does not reread its manifest or
        // promise availability; reopening must still observe the missing version.
        assert_eq!(
            dataset.clone().snapshot_ref("catalog/table").await.unwrap(),
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
            .snapshot_ref("catalog/old-incarnation")
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
        let mut location = dataset.manifest_location.clone();
        location.e_tag = None;
        location.size = Some(dataset.object_store.size(&location.path).await.unwrap());
        let (old_manifest, old_digest) = Dataset::get_manifest(
            &dataset.object_store,
            &location,
            dataset.uri(),
            &dataset.session,
        )
        .await
        .unwrap();
        let index_offset = old_manifest.index_section.unwrap();
        assert!(location.size.unwrap() as usize - index_offset > dataset.object_store.block_size());
        dataset.manifest = old_manifest.clone();
        dataset.manifest_location = location.clone();
        dataset.snapshot_digest = old_digest.clone();
        assert_eq!(dataset.load_indices().await.unwrap()[0].name, "old_idx");
        std::fs::remove_dir_all(&uri).unwrap();
        let recreated = write_indexed_dataset(uri.to_str().unwrap(), 3, "new_idx", true).await;
        location.size = Some(recreated.object_store.size(&location.path).await.unwrap());
        let (new_manifest, new_digest) = Dataset::get_manifest(
            &recreated.object_store,
            &location,
            dataset.uri(),
            &dataset.session,
        )
        .await
        .unwrap();
        assert!(!Arc::ptr_eq(&old_manifest, &new_manifest));
        assert_ne!(old_digest.get().unwrap(), new_digest.get().unwrap());
        assert_eq!(new_manifest.fragments.len(), 2);
        dataset.manifest = new_manifest;
        dataset.manifest_location = location;
        dataset.snapshot_digest = new_digest;
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
            .object_store
            .inner
            .get(&first.manifest_location.path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let second_bytes = second
            .object_store
            .inner
            .get(&second.manifest_location.path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let mut store = lance_io::object_store::ObjectStore::memory();
        store.store_prefix = "endpoint-a".into();
        let mut location = first.manifest_location.clone();
        location.path = object_store::path::Path::from("table/_versions/1.manifest");
        location.size = Some(first_bytes.len() as u64);
        location.e_tag = Some("opaque-generation".into());
        location.identity = Some("record-a".into());
        store
            .inner
            .put(&location.path, first_bytes.into())
            .await
            .unwrap();
        let (first_manifest, first_digest) =
            Dataset::get_manifest(&store, &location, "memory://table", &first.session)
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
        store
            .inner
            .put(&location.path, second_bytes.into())
            .await
            .unwrap();
        let (second_manifest, second_digest) =
            Dataset::get_manifest(&store, &location, "memory://table", &first.session)
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
        let snapshot = dataset.snapshot_ref("catalog/table").await.unwrap();
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
            .object_store
            .delete(&dataset.manifest_location.path)
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
        let error = restored.snapshot_ref("catalog/table").await.unwrap_err();
        assert!(matches!(error, Error::DatasetNotFound { .. }), "{error}");
        assert!(error.to_string().contains("manifest"), "{error}");
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
        let snapshot = dataset.snapshot_ref("catalog/table").await.unwrap();
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
