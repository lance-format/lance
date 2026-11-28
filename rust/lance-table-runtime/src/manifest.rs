// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use byteorder::{ByteOrder, LittleEndian};
use lance_core::{Error, Result, box_error};
use lance_io::object_store::ObjectStore;
use lance_io::utils::{read_last_block, read_message, read_metadata_offset, read_struct};
use lance_table::feature_flags::{CoveringContract, ensure_can_read_manifest};
use lance_table::format::{
    IndexMetadata, MAGIC, Manifest, pb, populate_manifest_schema_dictionaries,
    storage_contract::check_manifest_storage_version,
};
use lance_table::io::{commit::ManifestLocation, manifest::read_manifest_with_size};
use lance_table::transaction::Transaction;
use prost::Message;

use crate::caches::{IndexMetadataKey, ManifestKey, TransactionKey};
use crate::{Table, TableSession, TableState, TableView, VersionTransaction, refs};

impl Table {
    /// Read and validate a manifest, prefetching sections available in its tail block.
    #[doc(hidden)]
    pub async fn load_manifest(
        object_store: &ObjectStore,
        manifest_location: &ManifestLocation,
        uri: &str,
        session: &TableSession,
    ) -> Result<Manifest> {
        let object_reader = if let Some(size) = manifest_location.size {
            object_store
                .open_with_size(&manifest_location.path, size as usize)
                .await
        } else {
            object_store.open(&manifest_location.path).await
        };
        let object_reader = object_reader.map_err(|e| match &e {
            Error::NotFound { uri, .. } => Error::dataset_not_found(uri.clone(), box_error(e)),
            _ => e,
        })?;

        let manifest_size = object_reader.size().await.map_err(|err| match err {
            object_store::Error::NotFound { path, source } => {
                Error::dataset_not_found(path, source)
            }
            _ => Error::io(format!(
                "failed to read manifest {}: {}",
                manifest_location.path, err
            )),
        })?;
        if manifest_size < 16 {
            if manifest_location.size.is_some() {
                let manifest_location = ManifestLocation {
                    size: None,
                    ..manifest_location.clone()
                };
                return Box::pin(Self::load_manifest(
                    object_store,
                    &manifest_location,
                    uri,
                    session,
                ))
                .await;
            }
            return Err(Error::corrupt_file(
                manifest_location.path.clone(),
                format!("Invalid format: manifest is only {manifest_size} bytes (minimum 16)"),
            ));
        }

        let last_block =
            read_last_block(object_reader.as_ref())
                .await
                .map_err(|err| match err {
                    object_store::Error::NotFound { path, source } => {
                        Error::dataset_not_found(path, source)
                    }
                    _ => Error::io(format!(
                        "failed to read manifest {}: {}",
                        manifest_location.path, err
                    )),
                })?;

        // A stale cached size yields a bogus footer offset. Detect it (the block
        // lacks the trailing magic) and retry with the true size, like
        // read_manifest. At the true size, the file itself is damaged.
        if !last_block.ends_with(MAGIC) {
            if manifest_location.size.is_some() {
                let manifest_location = ManifestLocation {
                    size: None,
                    ..manifest_location.clone()
                };
                return Box::pin(Self::load_manifest(
                    object_store,
                    &manifest_location,
                    uri,
                    session,
                ))
                .await;
            }
            return Err(Error::corrupt_file(
                manifest_location.path.clone(),
                format!(
                    "Invalid format: magic number does not match, manifest is {manifest_size} bytes"
                ),
            ));
        }

        let offset = read_metadata_offset(&last_block)?;

        // If manifest is in the last block, we can decode directly from memory.
        let manifest_size = object_reader.size().await?;
        let mut manifest = if manifest_size - offset <= last_block.len() {
            let manifest_len = manifest_size - offset;
            let offset_in_block = last_block.len() - manifest_len;
            let message_len =
                LittleEndian::read_u32(&last_block[offset_in_block..offset_in_block + 4]) as usize;
            let message_data = &last_block[offset_in_block + 4..offset_in_block + 4 + message_len];
            Manifest::try_from(lance_table::format::pb::Manifest::decode(message_data)?)
        } else {
            let mut manifest: Manifest = read_struct(object_reader.as_ref(), offset).await?;
            manifest.detach_sparse_inline_row_ids(manifest_size - offset);
            Ok(manifest)
        }?;

        // A zeroed length prefix decodes to an empty message without an error, which would
        // open as an empty dataset. Versions start at 1.
        if manifest.version == 0 {
            return Err(Error::corrupt_file(
                manifest_location.path.clone(),
                format!(
                    "Invalid format: manifest at offset {offset} of {manifest_size} bytes has version 0"
                ),
            ));
        }

        ensure_can_read_manifest(&manifest)?;

        check_manifest_storage_version(&mut manifest)?;

        // If indices were also in the last block, we can take the opportunity to
        // decode them now and cache them.
        if let Some(index_offset) = manifest.index_section
            && manifest_size - index_offset <= last_block.len()
        {
            let offset_in_block = last_block.len() - (manifest_size - index_offset);
            let message_len =
                LittleEndian::read_u32(&last_block[offset_in_block..offset_in_block + 4]) as usize;
            let message_data = &last_block[offset_in_block + 4..offset_in_block + 4 + message_len];
            let section = lance_table::format::pb::IndexSection::decode(message_data)?;
            // Cached unfiltered: this is the same cache the commit path reads
            // from, and an index this build cannot decode still has to survive
            // into the next manifest. Version filtering happens on the way out,
            // when a consumer chooses indices for a query.
            let contract = CoveringContract::of(&manifest);
            let indices: Vec<IndexMetadata> = section
                .indices
                .into_iter()
                .map(|proto| IndexMetadata::from_manifest_proto(proto, contract))
                .collect::<Result<Vec<_>>>()?;
            let metadata_key = IndexMetadataKey {
                version: manifest_location.version,
                store_identity: &object_store.store_prefix,
                e_tag: manifest_location.e_tag.as_deref(),
            };
            session
                .cache_index_metadata(uri, &metadata_key, &manifest, indices)
                .await;
        }

        // If transaction is also in the last block, we can take the opportunity to
        // decode them now and cache them.
        if let Some(transaction_offset) = manifest.transaction_section
            && manifest_size - transaction_offset <= last_block.len()
        {
            let offset_in_block = last_block.len() - (manifest_size - transaction_offset);
            let message_len =
                LittleEndian::read_u32(&last_block[offset_in_block..offset_in_block + 4]) as usize;
            let message_data = &last_block[offset_in_block + 4..offset_in_block + 4 + message_len];
            if let Some(transaction) =
                decode_inline_transaction(message_data, manifest_location.version)
            {
                let metadata_cache = session.metadata_cache().for_dataset(uri);
                let metadata_key = TransactionKey {
                    version: manifest_location.version,
                };
                metadata_cache
                    .insert_with_key(&metadata_key, Arc::new(transaction))
                    .await;
            }
        }

        populate_manifest_schema_dictionaries(&mut manifest, object_reader.as_ref()).await?;

        Ok(manifest)
    }

    /// Fetch the manifest for `manifest_location` from the session metadata
    /// cache, loading and caching it on a miss.
    #[doc(hidden)]
    pub async fn get_manifest(
        object_store: &ObjectStore,
        manifest_location: &ManifestLocation,
        uri: &str,
        session: &TableSession,
    ) -> Result<Arc<Manifest>> {
        if session.require_manifest_etag && manifest_location.e_tag.is_none() {
            // A live branch can reuse a deleted version number. Keep the manifest
            // and its inline transaction and index metadata out of the shared caches.
            let isolated = session.with_cache_identity(None);
            return Self::load_manifest(object_store, manifest_location, uri, &isolated)
                .await
                .map(Arc::new);
        }
        if manifest_location.size.is_none() {
            return Ok(Arc::new(
                Self::load_manifest(object_store, manifest_location, uri, session).await?,
            ));
        }
        let metadata_cache = session.metadata_cache().for_dataset(uri);
        let manifest_key = ManifestKey {
            version: manifest_location.version,
            e_tag: manifest_location.e_tag.as_deref(),
        };
        if let Some(cached) = metadata_cache.get_with_key(&manifest_key).await {
            ensure_can_read_manifest(&cached)?;
            return Ok(cached);
        }
        let loaded =
            Arc::new(Self::load_manifest(object_store, manifest_location, uri, session).await?);
        metadata_cache
            .insert_with_key(&manifest_key, loaded.clone())
            .await;
        Ok(loaded)
    }
}

impl TableView<'_> {
    /// Read the transaction file for this version of the dataset.
    ///
    /// If there was no transaction file written for this version of the dataset
    /// then this will return None.
    pub async fn read_transaction(&self) -> Result<Option<Transaction>> {
        let transaction_key = TransactionKey {
            version: self.manifest.version,
        };
        if let Some(transaction) = self
            .state
            .metadata_cache
            .get_with_key(&transaction_key)
            .await
        {
            return Ok(Some((*transaction).clone()));
        }

        let transaction = self
            .state
            .read_transaction_from_storage(self.manifest, &self.state.manifest_location)
            .await?;

        if let Some(tx) = transaction.as_ref() {
            self.state
                .metadata_cache
                .insert_with_key(&transaction_key, Arc::new(tx.clone()))
                .await;
        }
        Ok(transaction)
    }

    /// Read the transaction (if any) and commit timestamp of a version of the
    /// dataset. `version` is a version number on this dataset's current branch.
    ///
    /// When `version` is the version this handle is checked out at, the
    /// in-memory manifest is used and the transaction is read directly from
    /// storage, so no manifest download is needed. Otherwise the version's
    /// manifest is read transiently: no historical `Table` is constructed,
    /// no `IndexSection` is decoded, and no session cache is written, so
    /// scanning many historical versions does not fill the shared caches. A
    /// manifest that the session cache already holds (for example after
    /// `checkout_version`) is reused instead of being downloaded again.
    ///
    /// Neither path consults or populates the session's transaction cache
    /// (unlike [`Self::read_transaction`]): that cache is keyed only by
    /// version number, which is not safe to reuse across a dataset dropped
    /// and recreated at the same URI within the same session (see #8916).
    ///
    /// Returns an error if the version does not exist (for example, if it has
    /// been cleaned up).
    ///
    pub async fn read_version_transaction(&self, version: u64) -> Result<VersionTransaction> {
        if version == self.manifest.version {
            let transaction = self
                .state
                .read_transaction_from_storage(self.manifest, &self.state.manifest_location)
                .await?;
            return Ok(VersionTransaction {
                version,
                timestamp: self.manifest.timestamp(),
                transaction,
            });
        }

        // Resolve against this dataset's current branch.
        let mut manifest_location = self
            .state
            .commit_handler
            .resolve_version_location(&self.state.base, version, &self.state.object_store.inner)
            .await?;

        // Mirror `get_manifest`: only a location with a known size (and hence
        // a trustworthy e_tag) identifies a cache entry precisely enough to
        // reuse without re-reading the manifest body.
        let cached_manifest = if manifest_location.size.is_some() {
            let manifest_key = ManifestKey {
                version: manifest_location.version,
                e_tag: manifest_location.e_tag.as_deref(),
            };
            self.state.metadata_cache.get_with_key(&manifest_key).await
        } else {
            None
        };

        let manifest = match cached_manifest {
            Some(manifest) => manifest,
            None => {
                // Keep the DatasetNotFound variant callers expect for a missing
                // version.
                let (manifest, file_size) = read_manifest_with_size(
                    &self.state.object_store,
                    &manifest_location.path,
                    manifest_location.size,
                )
                .await
                .map_err(|e| match &e {
                    Error::NotFound { uri, .. } => {
                        Error::dataset_not_found(uri.clone(), box_error(e))
                    }
                    _ => e,
                })?;
                // Reuse the size from the footer read when opening the inline transaction.
                manifest_location.size = Some(file_size);
                Arc::new(manifest)
            }
        };

        // The resolved manifest must belong to this dataset's branch. A
        // mismatch means the commit handler resolved against a different chain
        // (for example an external manifest store that ignores
        // branch-qualified paths); error loudly rather than hand back another
        // branch's transaction.
        if manifest.branch != self.manifest.branch {
            return Err(Error::internal(format!(
                "reading version {} on branch '{}' resolved a manifest belonging to branch '{}'",
                version,
                refs::normalize_branch(self.manifest.branch.as_deref()),
                refs::normalize_branch(manifest.branch.as_deref()),
            )));
        }

        let transaction = self
            .state
            .read_transaction_from_storage(&manifest, &manifest_location)
            .await?;

        Ok(VersionTransaction {
            version: manifest.version,
            timestamp: manifest.timestamp(),
            transaction,
        })
    }

    /// Read the transaction file for this version of the dataset.
    ///
    /// If there was no transaction file written for this version of the dataset
    /// then this will return None.
    ///
    /// Does not populate the session caches; see
    /// [`Self::read_version_transaction`].
    ///
    pub async fn read_transaction_by_version(&self, version: u64) -> Result<Option<Transaction>> {
        Ok(self.read_version_transaction(version).await?.transaction)
    }

    pub async fn get_transactions(
        &self,
        recent_transactions: usize,
    ) -> Result<Vec<Option<Transaction>>> {
        let mut transactions = vec![];
        let mut table = self.to_table();

        loop {
            let transaction = table.read_transaction().await?;
            transactions.push(transaction);

            if transactions.len() >= recent_transactions {
                break;
            } else {
                match table.checkout_version(table.version_id() - 1).await {
                    Ok(checked_out) => table = checked_out,
                    Err(Error::DatasetNotFound { .. }) => break,
                    Err(err) => return Err(err),
                }
            }
        }

        Ok(transactions)
    }
}

impl TableState {
    /// Read the transaction recorded by `manifest` directly from storage,
    /// without consulting or populating any session cache.
    #[doc(hidden)]
    pub async fn read_transaction_from_storage(
        &self,
        manifest: &Manifest,
        manifest_location: &ManifestLocation,
    ) -> Result<Option<Transaction>> {
        // Prefer inline transaction from manifest when available
        if let Some(pos) = manifest.transaction_section {
            let reader = match manifest_location.size {
                Some(size) => {
                    self.object_store
                        .open_with_size(&manifest_location.path, size as usize)
                        .await?
                }
                None => self.object_store.open(&manifest_location.path).await?,
            };

            // A concurrent overwrite can leave the listed size too small; retry
            // once with the true size.
            let tx: pb::Transaction = match read_message(reader.as_ref(), pos).await {
                Err(e)
                    if manifest_location.size.is_some()
                        && e.to_string().contains("file size is too small") =>
                {
                    let reader = self.object_store.open(&manifest_location.path).await?;
                    read_message(reader.as_ref(), pos).await?
                }
                other => other?,
            };
            Transaction::try_from(tx).map(Some)
        } else if let Some(path) = &manifest.transaction_file {
            // Fallback: read external transaction file if present
            let path = self.transactions_dir().join(path.as_str());
            let data = self.object_store.read_object(&path).await?.bytes().await?;
            let transaction = lance_table::format::pb::Transaction::decode(data)?;
            Transaction::try_from(transaction).map(Some)
        } else {
            Ok(None)
        }
    }
}

/// Decode an inline transaction section for opportunistic caching.
///
/// Returns `None` instead of failing when the transaction cannot be decoded:
/// the section may have been written by a newer version of Lance with an
/// operation type this version does not know, and that must not prevent
/// opening the dataset. Paths that need the transaction contents surface the
/// error at their call sites instead.
fn decode_inline_transaction(message_data: &[u8], version: u64) -> Option<Transaction> {
    match lance_table::format::pb::Transaction::decode(message_data)
        .map_err(Error::from)
        .and_then(Transaction::try_from)
    {
        Ok(transaction) => Some(transaction),
        Err(err) => {
            log::warn!(
                "Failed to decode the inline transaction of version {}; \
                 it may have been written by a newer version of Lance: {}",
                version,
                err
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_table::transaction::Operation;

    #[test]
    fn test_decode_inline_transaction_tolerates_unknown_operations() {
        // A transaction written by a newer version of Lance may carry an operation
        // this version cannot decode; prost surfaces it as a missing oneof. It
        // decodes as `Operation::Unknown` rather than failing.
        let unknown_operation = pb::Transaction {
            read_version: 1,
            uuid: "test".to_string(),
            ..Default::default()
        };
        let decoded = decode_inline_transaction(&unknown_operation.encode_to_vec(), 42).unwrap();
        assert!(matches!(decoded.operation, Operation::Unknown { .. }));

        // Corrupt bytes are likewise tolerated.
        assert!(decode_inline_transaction(&[0xff, 0xff, 0xff], 42).is_none());

        // A decodable transaction is returned.
        let known = pb::Transaction::try_from(&Transaction::new(
            1,
            Operation::Append { fragments: vec![] },
            None,
        ))
        .unwrap();
        let decoded = decode_inline_transaction(&known.encode_to_vec(), 42).unwrap();
        assert!(matches!(decoded.operation, Operation::Append { .. }));
    }
}
