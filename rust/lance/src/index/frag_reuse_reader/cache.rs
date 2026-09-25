// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Immutable ledgers and store-bound mappings shared across dataset snapshots.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use lance_core::Result;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, WeakLanceCache};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::fragment_reuse::{MappingReader, OrderedCompactionMapping};
use lance_index::frag_reuse::stable_partition::{MAPPING_FILE, StablePartitionMapping};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_table::format::IndexMetadata;
use lance_table::system_index::frag_reuse::ledger::{FragReuseLedger, Mapping, Transition};
use uuid::Uuid;

use crate::Dataset;

struct LedgerKey {
    uuid: Uuid,
    version: i32,
    store: String,
    directory: String,
}

impl CacheKey for LedgerKey {
    type ValueType = FragReuseLedger;
    fn key(&self) -> Cow<'_, str> {
        format!(
            "{}:{}:{}:{}",
            self.uuid, self.version, self.store, self.directory
        )
        .into()
    }
    fn type_name() -> &'static str {
        "FragmentReuseLedger"
    }
    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fri-ledger", 1)
    }
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.uuid.as_bytes());
        builder.write_i32(self.version);
        builder.write_str(&self.store);
        builder.write_str(&self.directory);
    }
}

pub(super) async fn open_ledger(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> Result<Arc<FragReuseLedger>> {
    let store = dataset.object_store_for_index(index).await?;
    let key = LedgerKey {
        uuid: index.uuid,
        version: index.index_version,
        store: store.store_prefix.clone(),
        directory: dataset.indice_files_dir(index)?.to_string(),
    };
    dataset
        .index_cache
        .get_or_insert_with_key(key, || super::load_ledger(dataset, index))
        .await
}

#[derive(Clone)]
struct MappingKey {
    fingerprint: [u8; 32],
    directory: String,
    // In-memory only. The cached reader retains this ObjectStore, preventing
    // pointer reuse while the entry exists. Store rotation gets a fresh reader.
    binding: u64,
}

impl CacheKey for MappingKey {
    type ValueType = CachedMapping;
    fn key(&self) -> Cow<'_, str> {
        format!(
            "{:x?}:{}:{}",
            self.fingerprint, self.binding, self.directory
        )
        .into()
    }
    fn type_name() -> &'static str {
        "FragmentReuseMappingReader"
    }
    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fri-mapping-reader", 1)
    }
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(&self.fingerprint);
        builder.write_u64(self.binding);
        builder.write_str(&self.directory);
    }
}

pub(super) struct CachedMapping {
    reader: Arc<dyn MappingReader>,
    cache: Option<(WeakLanceCache, MappingKey)>,
    grows_on_open: bool,
    accounted_bytes: AtomicUsize,
}

impl DeepSizeOf for CachedMapping {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.reader.deep_size_of_children(context)
            + self
                .cache
                .as_ref()
                .map_or(0, |(_, key)| key.directory.capacity())
    }
}

impl CachedMapping {
    pub(super) async fn remap_row_ids(
        self: &Arc<Self>,
        row_ids: &[u64],
    ) -> Result<Vec<Option<u64>>> {
        let result = self.reader.remap_row_ids(row_ids).await;
        // TODO: Evaluate cache size accuracy versus latency impact.
        if self.grows_on_open {
            // Counts are loaded lazily. Refresh the cache's weight even when a
            // subsequent label read fails; invalid input alone does not grow it.
            let bytes = self.reader.deep_size_of();
            if bytes != self.accounted_bytes.swap(bytes, Ordering::Relaxed)
                && let Some((cache, key)) = &self.cache
            {
                cache.insert_with_key(key, self.clone()).await;
            }
        }
        result
    }

    #[cfg(test)]
    pub(super) fn uncached(reader: Arc<dyn MappingReader>) -> Arc<Self> {
        Arc::new(Self {
            reader,
            cache: None,
            grows_on_open: false,
            accounted_bytes: AtomicUsize::new(0),
        })
    }
}

pub(super) async fn open_mapping(
    dataset: &Dataset,
    transition: &Transition,
) -> Result<Arc<CachedMapping>> {
    let (key, store) = match transition.mapping() {
        Mapping::OrderedCompaction(_) => (
            MappingKey {
                fingerprint: *transition.fingerprint(),
                directory: String::new(),
                binding: 0,
            },
            None,
        ),
        Mapping::StablePartition(reference) => {
            let base = match reference.base_id {
                None => dataset.base.clone(),
                Some(id) => dataset
                    .manifest
                    .base_paths
                    .get(&id)
                    .ok_or_else(|| {
                        super::corrupt(format!(
                            "mapping {} references missing base {id}",
                            reference.map_id
                        ))
                    })?
                    .extract_path(dataset.session.store_registry())?,
            };
            let directory = base.join("_fri").join(reference.map_id.as_str());
            let store = dataset.object_store(reference.base_id).await?;
            (
                MappingKey {
                    fingerprint: *transition.fingerprint(),
                    directory: directory.to_string(),
                    binding: Arc::as_ptr(&store) as usize as u64,
                },
                Some((store, directory)),
            )
        }
    };
    let cache = WeakLanceCache::from(&dataset.index_cache);
    dataset
        .index_cache
        .get_or_insert_with_key(key.clone(), || async move {
            let reader: Arc<dyn MappingReader> = match transition.mapping() {
                Mapping::OrderedCompaction(remap) => {
                    Arc::new(OrderedCompactionMapping::new(remap.clone()))
                }
                Mapping::StablePartition(reference) => {
                    let (store, directory) = store.ok_or_else(|| {
                        lance_core::Error::internal("missing stable-partition store binding")
                    })?;
                    let metadata_cache = dataset.metadata_cache.file_metadata_cache(&directory);
                    let store = LanceIndexStore::new(store, directory, Arc::new(metadata_cache))
                        .with_file_sizes(HashMap::from([(
                            MAPPING_FILE.to_string(),
                            reference.map_size_bytes,
                        )]));
                    Arc::new(StablePartitionMapping::try_new(
                        Arc::new(store),
                        transition.sources().to_vec(),
                        transition.destinations().to_vec(),
                    )?)
                }
            };
            let accounted_bytes = AtomicUsize::new(reader.deep_size_of());
            Ok(CachedMapping {
                reader,
                cache: Some((cache, key)),
                grows_on_open: matches!(transition.mapping(), Mapping::StablePartition(_)),
                accounted_bytes,
            })
        })
        .await
}
