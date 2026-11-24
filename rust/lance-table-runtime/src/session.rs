// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_core::cache::{CacheMetricsKind, LanceCache, QuickCacheBackend, QuickCacheShardPolicy};
use lance_io::object_store::ObjectStoreRegistry;
use lance_table::format::{IndexMetadata, Manifest};

use crate::caches::GlobalMetadataCache;

/// Access policy for external blob descriptors that do not name a registered base.
///
/// This policy also prevents dataset writers from accepting unregistered URI
/// references and gives controlled services an error without local opt-in advice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UnregisteredExternalBlobPolicy {
    /// Resolve absolute URIs using this session's providers and credentials.
    #[default]
    Allow,
    /// Reject `base_id == 0` before parsing its URI or accessing storage.
    /// Registered bases remain available under their configured credentials.
    Deny,
}

/// Shared caches and object store providers for table metadata operations.
///
/// Pass the same session to multiple [builders](crate::TableBuilder) to reuse
/// metadata and store providers across opened snapshots.
#[derive(Clone)]
pub struct TableSession {
    metadata_cache: GlobalMetadataCache,
    index_cache: LanceCache,
    store_registry: Arc<ObjectStoreRegistry>,
    unregistered_external_blob_policy: UnregisteredExternalBlobPolicy,
    index_metadata_observer: Option<fn(&Manifest, &[IndexMetadata])>,
}

impl TableSession {
    /// Use existing cache handles and object store providers.
    ///
    /// Both caches are scoped by table URI when a table opens. Embedders can
    /// supply the cache used by their index readers so it shares entries and
    /// accounting with manifest prefetch.
    pub fn with_caches(
        index_cache: LanceCache,
        metadata_cache: LanceCache,
        store_registry: Arc<ObjectStoreRegistry>,
    ) -> Self {
        Self {
            metadata_cache: GlobalMetadataCache(metadata_cache),
            index_cache,
            store_registry,
            unregistered_external_blob_policy: UnregisteredExternalBlobPolicy::Allow,
            index_metadata_observer: None,
        }
    }

    /// Scope both caches to a snapshot identity. An unknown identity disables
    /// caching for this handle, so discovery cannot evict reusable entries.
    ///
    /// ```
    /// use lance_table_runtime::session::TableSession;
    /// let session = TableSession::default();
    /// let branch = session.with_cache_identity(Some("generation-1"));
    /// assert!(std::sync::Arc::ptr_eq(&session.store_registry(), &branch.store_registry()));
    /// ```
    pub fn with_cache_identity(&self, identity: Option<&str>) -> Self {
        let scope = |cache: &LanceCache, role| match identity {
            Some(key) => cache.with_key_prefix(key),
            None => LanceCache::with_backend_and_metrics_kind(
                Arc::new(QuickCacheBackend::with_capacity(0)),
                role,
            ),
        };
        Self {
            metadata_cache: GlobalMetadataCache(scope(
                self.metadata_cache.cache(),
                CacheMetricsKind::Metadata,
            )),
            index_cache: scope(&self.index_cache, CacheMetricsKind::Index),
            ..self.clone()
        }
    }

    /// Receive the manifest and its raw index metadata when reading the index
    /// section. The runtime treats each index's implementation details as opaque.
    #[doc(hidden)]
    pub fn with_index_metadata_observer(
        mut self,
        observer: fn(&Manifest, &[IndexMetadata]),
    ) -> Self {
        self.index_metadata_observer = Some(observer);
        self
    }

    pub(crate) async fn cache_index_metadata(
        &self,
        uri: &str,
        key: &crate::caches::IndexMetadataKey<'_>,
        manifest: &Manifest,
        indices: Vec<IndexMetadata>,
    ) {
        self.observe_index_metadata(manifest, &indices);
        self.index_cache
            .with_key_prefix(uri)
            .insert_with_key(key, Arc::new(indices))
            .await;
    }

    pub(crate) fn observe_index_metadata(&self, manifest: &Manifest, indices: &[IndexMetadata]) {
        if let Some(observer) = self.index_metadata_observer {
            observer(manifest, indices);
        }
    }

    /// The shared cache for table and file metadata.
    pub fn metadata_cache(&self) -> &GlobalMetadataCache {
        &self.metadata_cache
    }

    /// The cache for opaque index metadata and row-id lookups.
    pub fn index_cache(&self) -> &LanceCache {
        &self.index_cache
    }

    /// The registry used to resolve primary and additional object stores.
    pub fn store_registry(&self) -> Arc<ObjectStoreRegistry> {
        self.store_registry.clone()
    }

    /// Set the policy for external blob descriptors without a registered base.
    pub fn with_unregistered_external_blob_policy(
        mut self,
        policy: UnregisteredExternalBlobPolicy,
    ) -> Self {
        self.unregistered_external_blob_policy = policy;
        self
    }

    /// The policy for external blob descriptors without a registered base.
    pub fn unregistered_external_blob_policy(&self) -> UnregisteredExternalBlobPolicy {
        self.unregistered_external_blob_policy
    }
}

impl Default for TableSession {
    fn default() -> Self {
        Self::with_caches(
            LanceCache::with_backend_and_metrics_kind(
                Arc::new(QuickCacheBackend::with_shard_policy(
                    DEFAULT_INDEX_METADATA_CACHE_SIZE,
                    QuickCacheShardPolicy::Single,
                )),
                CacheMetricsKind::Index,
            ),
            LanceCache::with_backend_and_metrics_kind(
                Arc::new(QuickCacheBackend::with_shard_policy(
                    DEFAULT_METADATA_CACHE_SIZE,
                    QuickCacheShardPolicy::Recommended,
                )),
                CacheMetricsKind::Metadata,
            ),
            Arc::default(),
        )
    }
}

/// Default budget for manifest, transaction, and file metadata caching.
pub const DEFAULT_METADATA_CACHE_SIZE: usize = 1024 * 1024 * 1024;
const DEFAULT_INDEX_METADATA_CACHE_SIZE: usize = 64 * 1024 * 1024;
