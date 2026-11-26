// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Capture and validate execution references to immutable committed snapshots.

use std::sync::Arc;

use lance_core::{Error, Result};
use lance_table::format::Manifest;

pub use lance_table::snapshot::{SnapshotRef, SnapshotVerification};

use crate::Table;
use crate::caches::{ManifestDigestKey, cache_manifest};

impl Table {
    /// Capture a compact, portable reference to this exact snapshot.
    ///
    /// `table_identity` identifies the table incarnation and catalog/storage
    /// authority. It must distinguish tables recreated at the same URI. Workers
    /// supply their own authenticated storage settings when reopening the reference.
    ///
    /// Uses an authoritative commit identity or an ETag without reading the file.
    /// If neither is available, capture reads and hashes the complete original
    /// manifest once; clones share that digest. No data payloads are read. The
    /// reference does not acquire retention or guarantee continued file availability.
    /// Serialized manifests and arbitrary manifest views require full-file validation:
    /// a freshly resolved token alone does not establish their original generation.
    ///
    /// ```
    /// # use lance_core::Result;
    /// use lance_table_runtime::{Table, TableBuilder, TableSession};
    /// use lance_table::snapshot::SnapshotRef;
    /// # async fn example(dataset: &Table) -> Result<()> {
    /// let snapshot = dataset.snapshot_ref("catalog/table-incarnation").await?;
    /// let transported = SnapshotRef::decode(&snapshot.encode())?;
    /// let worker = TableBuilder::from_uri(transported.table_uri())
    ///     .load_snapshot_with_session(&transported, std::sync::Arc::new(TableSession::default())).await?;
    /// assert_eq!(dataset.version_id(), worker.version_id());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn snapshot_ref(&self, table_identity: impl Into<String>) -> Result<SnapshotRef> {
        self.snapshot_ref_with_verification(table_identity, SnapshotVerification::Identity)
            .await
    }

    /// Capture with explicit verification: lightweight immutable identity or
    /// complete original-file BLAKE3. Identity mode falls back to BLAKE3 when
    /// the handle has no verified record identity or ETag for its manifest.
    ///
    /// ```
    /// # use lance_core::Result;
    /// # use lance_table_runtime::Table;
    /// use lance_table::snapshot::SnapshotVerification;
    /// # async fn example(table: &Table) -> Result<()> {
    /// let reference = table.snapshot_ref_with_verification(
    ///     "catalog/table-incarnation", SnapshotVerification::FullManifestDigest,
    /// ).await?;
    /// assert_eq!(reference.manifest_digest().len(), 32);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn snapshot_ref_with_verification(
        &self,
        table_identity: impl Into<String>,
        verification: SnapshotVerification,
    ) -> Result<SnapshotRef> {
        let table_identity = table_identity.into();
        if table_identity.trim().is_empty() {
            return Err(Error::invalid_input(
                "Snapshot table_identity cannot be empty",
            ));
        }
        self.validate_manifest_provenance()?;
        if self.manifest.version != self.state.manifest_location.version {
            return Err(Error::invalid_input(format!(
                "Loaded manifest version {} does not match resolved version {}",
                self.manifest.version, self.state.manifest_location.version,
            )));
        }
        if verification == SnapshotVerification::Identity
            && self.state.is_manifest_generation_verified
            && (self.state.manifest_location.identity.is_some()
                || self.state.manifest_location.e_tag.is_some())
        {
            let mut location = self.state.manifest_location.clone();
            if let Some(token) = &location.e_tag {
                location.e_tag = Some(
                    crate::manifest::conditional_etag(&self.state.object_store, token).into_owned(),
                );
            }
            return SnapshotRef::try_new_with_identity(
                table_identity,
                &self.refs.root()?.uri,
                self.manifest.branch.clone(),
                &location,
            );
        }
        let digest = self.ensure_snapshot_digest().await?;
        SnapshotRef::try_new(
            table_identity,
            &self.refs.root()?.uri,
            self.manifest.branch.clone(),
            &self.state.manifest_location,
            digest,
        )
    }

    /// Check a reference against this dataset's exact manifest and commit identity.
    ///
    /// Reopening with [`crate::TableBuilder::load_snapshot_with_session`] validates
    /// the mechanism carried by the reference. Full-digest references require an
    /// initialized digest on other handles; use [`Self::snapshot_ref_with_verification`]
    /// with [`SnapshotVerification::FullManifestDigest`] first. The caller's
    /// trusted resolver verifies table incarnation, authority, and credentials.
    ///
    /// ```
    /// # use lance_core::Result;
    /// use lance_table_runtime::{Table, TableBuilder, TableSession};
    /// # async fn example(dataset: &Table) -> Result<()> {
    /// let snapshot = dataset.snapshot_ref("catalog/table-incarnation").await?;
    /// dataset.validate_snapshot(&snapshot)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate_snapshot(&self, snapshot: &SnapshotRef) -> Result<()> {
        self.validate_manifest_provenance()?;
        let mismatch = || {
            Error::invalid_input(format!(
                "Snapshot {:?}/{} at {} does not match dataset snapshot {:?}/{} at {} (manifest {}, expected {})",
                snapshot.branch(),
                snapshot.version(),
                snapshot.table_uri(),
                self.manifest.branch,
                self.manifest.version,
                self.uri(),
                self.state.manifest_location.path,
                snapshot.manifest_path(),
            ))
        };
        if snapshot.validate_table_uri(&self.refs.root()?.uri).is_err()
            || self.manifest.branch.as_deref() != snapshot.branch()
            || self.manifest.version != snapshot.version()
            || self.state.manifest_location.version != snapshot.version()
            || self.state.manifest_location.path.as_ref() != snapshot.manifest_path()
            || self.state.manifest_location.identity.as_deref() != snapshot.manifest_identity()
        {
            return Err(mismatch());
        }
        if let Some(expected) = snapshot.manifest_etag()
            && !self
                .state
                .manifest_location
                .e_tag
                .as_deref()
                .is_some_and(|observed| {
                    crate::manifest::conditional_etag(&self.state.object_store, observed)
                        == crate::manifest::conditional_etag(&self.state.object_store, expected)
                })
        {
            return Err(mismatch());
        }
        if snapshot.manifest_digest().is_empty() {
            if !self.state.is_manifest_generation_verified {
                return Err(Error::invalid_input(
                    "Loaded manifest generation has not been verified; reopen the committed snapshot before validating a lightweight reference",
                ));
            }
            return Ok(());
        }
        let digest = self.snapshot_digest.get().ok_or_else(|| {
            Error::invalid_input("Dataset snapshot digest is unavailable; capture with SnapshotVerification::FullManifestDigest or reopen with TableBuilder::load_snapshot_with_session before binding a transported reference")
        })?;
        if digest.as_slice() != snapshot.manifest_digest() {
            return Err(mismatch());
        }

        Ok(())
    }

    /// Compute and share the BLAKE3 digest of the complete original manifest file.
    /// The resolved generation is checked before associating bytes with this handle.
    ///
    /// ```
    /// # use lance_core::Result;
    /// # use lance_table_runtime::Table;
    /// # async fn example(table: &Table) -> Result<()> {
    /// let digest = table.ensure_snapshot_digest().await?;
    /// assert_eq!(digest.len(), 32);
    /// # Ok(())
    /// # }
    /// ```
    #[doc(hidden)]
    pub async fn ensure_snapshot_digest(&self) -> Result<[u8; 32]> {
        self.validate_manifest_provenance()?;
        self.snapshot_digest
            .get_or_try_init(|| async {
                let key = ManifestDigestKey {
                    version: self.manifest.version,
                    store_identity: &self.state.object_store.store_prefix,
                    manifest_path: &self.state.manifest_location.path,
                    manifest_identity: self.state.manifest_location.identity.as_deref(),
                    e_tag: self.state.manifest_location.e_tag.as_deref(),
                };
                if let Some(cached) = self.state.metadata_cache.get_with_key(&key).await
                    && cached.manifest.ptr_eq(&Arc::downgrade(&self.manifest))
                    && cached
                        .store
                        .ptr_eq(&Arc::downgrade(&self.state.object_store.inner))
                    && let Some(digest) = cached.digest.get()
                {
                    return Ok(*digest);
                }
                let (observed, digest) = Self::load_manifest_with_digest(
                    &self.state.object_store,
                    &self.state.manifest_location,
                    self.uri(),
                    &self.state.session,
                )
                .await?;
                if !matches_loaded_manifest(observed, &self.manifest) {
                    return Err(Error::invalid_input(format!(
                        "Manifest {} no longer matches the loaded snapshot at version {}",
                        self.state.manifest_location.path, self.manifest.version,
                    )));
                }
                cache_manifest(
                    &self.state.metadata_cache,
                    &self.state.object_store,
                    &self.state.manifest_location,
                    self.manifest.clone(),
                    self.snapshot_digest.clone(),
                )
                .await;
                Ok(digest)
            })
            .await
            .copied()
    }

    fn validate_manifest_provenance(&self) -> Result<()> {
        if !self.state.owns_manifest_provenance(&self.manifest) {
            return Err(Error::invalid_input(format!(
                "Snapshot verification evidence does not belong to the loaded manifest at {} version {}; reopen the committed snapshot",
                self.state.manifest_location.path, self.manifest.version,
            )));
        }
        Ok(())
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
    use crate::{CommitBuilder, InsertBuilder, WriteParams};
    use arrow_array::record_batch;
    use lance_table::transaction::{Operation, Transaction};
    use object_store::ObjectStoreExt;
    use prost::Message;
    use rstest::rstest;

    async fn local_multi_fragment_table() -> (tempfile::TempDir, Table) {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().join("table");
        let batch = record_batch!(("id", Int32, [1, 2, 3, 4, 5, 6])).unwrap();
        let table = InsertBuilder::new(uri.to_str().unwrap())
            .with_params(&WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            })
            .execute(vec![batch])
            .await
            .unwrap();
        (directory, table)
    }

    #[rstest]
    #[case::commit_identity(true, true)]
    #[case::etag(false, true)]
    #[case::digest_fallback(false, false)]
    #[tokio::test]
    async fn identity_capture_selects_available_validation(
        #[case] has_commit_identity: bool,
        #[case] has_etag: bool,
    ) {
        let mut table = crate::test_util::test_commit_table(6, 3).await;
        table.state.manifest_location.identity = has_commit_identity.then(|| "record".into());
        if !has_etag {
            table.state.manifest_location.e_tag = None;
        }
        table.state.record_manifest_provenance(&table.manifest);
        let snapshot = table.snapshot_ref("catalog/table").await.unwrap();
        assert_eq!(snapshot.manifest_identity().is_some(), has_commit_identity);
        assert_eq!(
            snapshot.manifest_etag().is_some(),
            !has_commit_identity && has_etag
        );
        assert_eq!(
            snapshot.manifest_digest().len(),
            if has_commit_identity || has_etag {
                0
            } else {
                32
            }
        );
        table.validate_snapshot(&snapshot).unwrap();
    }

    #[tokio::test]
    async fn runtime_commit_replaces_initialized_digest_and_preserves_old_clone() {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().join("table");
        let uri = uri.to_str().unwrap();
        let batch = record_batch!(("id", Int32, [1, 2, 3, 4, 5, 6])).unwrap();
        InsertBuilder::new(uri)
            .with_params(&WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            })
            .execute(vec![batch])
            .await
            .unwrap();
        let opened = Table::open(uri).await.unwrap();
        assert_eq!(opened.manifest.fragments.len(), 3);
        let before = opened
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        let old = opened.clone();
        assert!(opened.snapshot_digest.get().is_some());
        let transaction = Transaction::new(
            opened.version_id(),
            Operation::Append {
                fragments: vec![opened.manifest.fragments[0].clone()],
            },
            None,
        );
        let committed = CommitBuilder::new(opened)
            .execute(transaction)
            .await
            .unwrap();
        assert!(committed.snapshot_digest.get().is_none());
        let after = committed
            .snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap();
        let raw = committed
            .state
            .object_store
            .inner
            .get(&committed.state.manifest_location.path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(after.manifest_digest(), blake3::hash(&raw).as_bytes());
        assert_ne!(after.manifest_digest(), before.manifest_digest());
        assert_eq!(
            old.snapshot_ref_with_verification(
                "catalog/table",
                SnapshotVerification::FullManifestDigest,
            )
            .await
            .unwrap(),
            before
        );
    }

    #[rstest]
    #[case::serialized_manifest(true)]
    #[case::manifest_view(false)]
    #[tokio::test]
    async fn refresh_reloads_unverified_manifest(#[case] is_serialized: bool) {
        let (_directory, source) = local_multi_fragment_table().await;
        let mut replacement = source.manifest.as_ref().clone();
        Arc::make_mut(&mut replacement.fragments).truncate(1);
        let mut producer = if is_serialized {
            let serialized =
                lance_table::format::pb::Manifest::from(source.manifest.as_ref()).encode_to_vec();
            lance_table::io::commit::write_manifest_file_to_path(
                &source.state.object_store,
                &mut replacement,
                None,
                &source.state.manifest_location.path,
                None,
            )
            .await
            .unwrap();
            crate::TableBuilder::from_uri(source.uri())
                .with_session(source.session())
                .with_serialized_manifest(&serialized)
                .unwrap()
                .load_with_session(source.session())
                .await
                .unwrap()
        } else {
            source.view().with_manifest_view(Arc::new(replacement))
        };
        let error = producer.snapshot_ref("catalog/table").await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("loaded"), "{error}");
        producer.checkout_latest().await.unwrap();
        let snapshot = producer.snapshot_ref("catalog/table").await.unwrap();
        let worker = crate::TableBuilder::from_uri(source.uri())
            .with_session(source.session())
            .load_snapshot_with_session(&snapshot, source.session())
            .await
            .unwrap();
        assert_eq!(
            producer.count_rows().await.unwrap(),
            if is_serialized { 2 } else { 6 }
        );
        assert_eq!(
            producer.count_rows().await.unwrap(),
            worker.count_rows().await.unwrap()
        );
    }

    #[tokio::test]
    async fn lightweight_validation_rejects_unverified_serialized_manifest() {
        let (_directory, source) = local_multi_fragment_table().await;
        let serialized =
            lance_table::format::pb::Manifest::from(source.manifest.as_ref()).encode_to_vec();
        let mut replacement = source.manifest.as_ref().clone();
        Arc::make_mut(&mut replacement.fragments).truncate(1);
        lance_table::io::commit::write_manifest_file_to_path(
            &source.state.object_store,
            &mut replacement,
            None,
            &source.state.manifest_location.path,
            None,
        )
        .await
        .unwrap();
        let producer = crate::TableBuilder::from_uri(source.uri())
            .with_session(source.session())
            .with_serialized_manifest(&serialized)
            .unwrap()
            .load_with_session(source.session())
            .await
            .unwrap();
        let current = crate::TableBuilder::from_uri(source.uri())
            .with_session(source.session())
            .load_with_session(source.session())
            .await
            .unwrap();
        let snapshot = current.snapshot_ref("catalog/table").await.unwrap();
        assert!(snapshot.manifest_digest().is_empty());
        assert_eq!(producer.count_rows().await.unwrap(), 6);
        assert_eq!(current.count_rows().await.unwrap(), 2);
        let error = producer.validate_snapshot(&snapshot).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("generation has not been verified"),
            "{error}"
        );
    }

    #[rstest]
    #[case::identity(SnapshotVerification::Identity, false)]
    #[case::cold_digest(SnapshotVerification::FullManifestDigest, false)]
    #[case::warm_digest(SnapshotVerification::FullManifestDigest, true)]
    #[tokio::test]
    async fn mutated_manifest_cannot_reuse_verification(
        #[case] verification: SnapshotVerification,
        #[case] has_digest: bool,
    ) {
        let mut producer = crate::test_util::test_commit_table(6, 3).await;
        let snapshot = producer
            .snapshot_ref_with_verification(
                "catalog/table",
                if has_digest {
                    SnapshotVerification::FullManifestDigest
                } else {
                    SnapshotVerification::Identity
                },
            )
            .await
            .unwrap();
        assert_eq!(producer.snapshot_digest.get().is_some(), has_digest);
        Arc::make_mut(&mut Arc::make_mut(&mut producer.manifest).fragments).truncate(1);
        assert_eq!(producer.count_rows().await.unwrap(), 2);
        let error = producer
            .snapshot_ref_with_verification("catalog/table", verification)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("does not belong to the loaded manifest"),
            "{error}"
        );
        let error = producer.ensure_snapshot_digest().await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("does not belong to the loaded manifest"),
            "{error}"
        );
        let error = producer.validate_snapshot(&snapshot).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("does not belong to the loaded manifest"),
            "{error}"
        );
    }
}
