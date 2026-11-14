// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_table::format::{IndexMetadata, Manifest};
use lance_table::system_index::is_system_index;
use lance_table::transaction::Operation;
use object_store::path::Path;
use roaring::RoaringBitmap;

use crate::Table;

use super::MAX_INLINE_TRANSACTION_BYTES;

/// User index inspection required by some table commits.
///
/// The runtime owns the table protocol and carries index metadata even when a
/// reader for that index is unavailable. An index implementation can supply
/// this hook to validate, recover, or classify the indices it understands.
#[async_trait]
pub trait CommitIndexHook: Send + Sync {
    /// The predicate for indices this implementation can use, including when
    /// deciding which segments still need fragment-reuse history.
    fn usability_predicate(&self) -> fn(&IndexMetadata) -> bool;

    /// Whether an entry may need detail inference. The runtime uses this
    /// predicate to avoid cloning a complete listing and to retry failed
    /// inference.
    fn needs_legacy_details(&self, _table: &Table, index: &IndexMetadata) -> bool {
        index
            .index_details
            .as_ref()
            .is_none_or(|details| details.value.is_empty())
    }

    /// Share successful inference between hooks with the same implementation
    /// and configuration. The namespace must identify both. `None` disables
    /// this cache.
    #[doc(hidden)]
    fn legacy_details_cache_namespace(&self) -> Option<&'static str> {
        None
    }

    /// Fill in legacy details when they can be inferred from the index files.
    /// Leave indices whose details are unknown or unreadable alone.
    async fn infer_legacy_details(&self, table: &Table, indices: &mut [IndexMetadata]);

    /// Recover fragment coverage from a user index's files.
    async fn recover_coverage(&self, table: &Table, index: &IndexMetadata)
    -> Result<RoaringBitmap>;

    /// Check the segments created by the operation against the candidate
    /// manifest, resolving newly added fields and bases through that manifest.
    async fn validate_new_segments(
        &self,
        table: &Table,
        operation: &Operation,
        candidate: &Manifest,
        indices: &[IndexMetadata],
    ) -> Result<()>;

    /// Whether the index files can be read through a batch row-address mapping.
    /// Only `Ok(Some(false))` proves coverage must be withdrawn. `None` and
    /// read errors leave the metadata available for a later attempt.
    async fn supports_batch_remapping(
        &self,
        table: &Table,
        index: &IndexMetadata,
    ) -> Result<Option<bool>>;

    /// Verify that the user indices can coexist with fragment reuse on a table
    /// with stable row IDs.
    fn validate_stable_row_id_indices(
        &self,
        _manifest: &Manifest,
        indices: &[IndexMetadata],
    ) -> Result<()> {
        if indices.iter().any(|index| !is_system_index(index)) {
            return Err(Error::not_supported(
                "Cannot verify user indices on a stable-row-id table with fragment reuse",
            ));
        }
        Ok(())
    }

    /// Recompute which user index segments can be removed after a prune rebases.
    /// This requires the index implementation's verified query coverage.
    async fn superseded_segments(&self, _table: &Table) -> Result<Vec<IndexMetadata>> {
        Err(Error::not_supported(
            "Cannot rederive superseded index segments without verified index coverage",
        ))
    }
}

/// Host actions that surround manifest publication.
#[doc(hidden)]
#[async_trait]
pub trait CommitLifecycleHook: Send + Sync {
    /// Copy managed blob payloads and rewrite their descriptors before
    /// publishing a deep clone.
    async fn copy_managed_blobs(
        &self,
        source: Table,
        destination_store: &ObjectStore,
        destination_base: &Path,
        destination_uri: &str,
        manifest: &mut Manifest,
    ) -> Result<()>;

    /// Called on a confirmed mainline commit, with the snapshot immediately
    /// before publication. Best-effort failures are handled by the host.
    async fn after_commit(&self, table: &Table, manifest: &Manifest);

    /// Supply the clock used to timestamp a commit attempt.
    fn timestamp_nanos(&self) -> u128 {
        now_nanos()
    }

    /// Maximum serialized size of a transaction inlined into the manifest.
    fn inline_transaction_limit(&self) -> usize {
        MAX_INLINE_TRANSACTION_BYTES
    }
}

/// Hooks bound to a single commit.
#[doc(hidden)]
#[derive(Clone, Copy, Default)]
pub struct CommitHooks<'a> {
    pub index: Option<&'a dyn CommitIndexHook>,
    pub lifecycle: Option<&'a dyn CommitLifecycleHook>,
}

impl CommitHooks<'_> {
    pub(crate) fn timestamp_nanos(&self) -> u128 {
        self.lifecycle
            .map_or_else(now_nanos, |hook| hook.timestamp_nanos())
    }

    pub(crate) fn inline_transaction_limit(&self) -> usize {
        self.lifecycle.map_or(MAX_INLINE_TRANSACTION_BYTES, |hook| {
            hook.inline_transaction_limit()
        })
    }
}

fn now_nanos() -> u128 {
    std::time::UNIX_EPOCH
        .elapsed()
        .map_or(0, |elapsed| elapsed.as_nanos())
}
