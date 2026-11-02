// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reclaim session-cache entries whose dataset versions cleanup removes.

use super::{CleanupInspection, CleanupTask};
use crate::session::caches::{ManifestKey, RowAddrMaskKey, RowIdIndexKey, TransactionKey};
use crate::session::index_caches::IndexMetadataKey;

/// A manifest that has aged out and is queued for deletion.
#[derive(Clone, Debug)]
pub(super) struct ExpiredManifest {
    pub(super) version: u64,
    /// Size from the manifest listing. `None` means it must be fetched before it can
    /// be counted in `crate::dataset::cleanup::RemovalStats::bytes_removed`.
    pub(super) size_bytes: Option<u64>,
    /// E-tag from the manifest listing, used to rebuild this version's exact
    /// session-cache keys.
    pub(super) e_tag: Option<String>,
}

/// Snapshot cache identities before cleanup consumes the inspection result.
pub(super) fn stale_versions(inspection: &CleanupInspection) -> Vec<(u64, Option<String>)> {
    inspection
        .old_manifests
        .values()
        .map(|expired| (expired.version, expired.e_tag.clone()))
        .collect()
}

impl<'a> CleanupTask<'a> {
    /// Evict known session-cache entries for versions selected by cleanup.
    ///
    /// Cache keys are opaque one-way digests, so entries cannot be found by
    /// prefix; instead each stale version's exact
    /// [`CacheKey`](lance_core::cache::CacheKey)s are rebuilt the same way the
    /// write and read paths build them and invalidated one by one. This covers
    /// the key families whose identity can be rebuilt from the version and
    /// manifest-listing e-tag: [`ManifestKey`],
    /// [`TransactionKey`], whole-dataset [`RowAddrMaskKey`], both
    /// [`RowIdIndexKey`] variants, and [`IndexMetadataKey`].
    ///
    /// Other version-scoped entries remain subject to capacity-based eviction,
    /// including manifest and index-metadata entries written without an e-tag
    /// by Rename commit handlers, derived index listings, segment ownership
    /// masks, row address masks restricted to a query-specific fragment subset,
    /// and per-fragment deletion-vector and row-id-sequence entries.
    pub(super) async fn invalidate_stale_version_caches(
        &self,
        stale_versions: &[(u64, Option<String>)],
    ) {
        // Explain runs must stay side-effect free. Per-object deletion is
        // best-effort; evicting an entry whose object remains only causes the
        // next access to refill it.
        if !self.action.deletes_files() || stale_versions.is_empty() {
            return;
        }

        let store_identity = self.dataset.object_store.store_prefix.clone();
        let index_cache = self
            .dataset
            .session
            .index_cache
            .for_dataset(&self.dataset.uri);
        for (version, e_tag) in stale_versions {
            let e_tag = e_tag.as_deref();
            self.dataset
                .metadata_cache
                .invalidate_key(&ManifestKey {
                    version: *version,
                    e_tag,
                })
                .await;
            self.dataset
                .metadata_cache
                .invalidate_key(&TransactionKey { version: *version })
                .await;
            self.dataset
                .metadata_cache
                .invalidate_key(&RowAddrMaskKey {
                    version: *version,
                    e_tag,
                    restrict_hash: None,
                })
                .await;
            // The cache keeps one row id index entry per `with_deletions`
            // variant, and cleanup cannot know which variants this session
            // populated, so both are invalidated.
            self.dataset
                .metadata_cache
                .invalidate_key(&RowIdIndexKey {
                    version: *version,
                    with_deletions: true,
                    e_tag,
                })
                .await;
            self.dataset
                .metadata_cache
                .invalidate_key(&RowIdIndexKey {
                    version: *version,
                    with_deletions: false,
                    e_tag,
                })
                .await;
            index_cache
                .invalidate_key(&IndexMetadataKey {
                    version: *version,
                    store_identity: &store_identity,
                    e_tag,
                })
                .await;
        }
    }
}
