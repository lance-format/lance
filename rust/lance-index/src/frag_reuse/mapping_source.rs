// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The [`MappingSource`] that reads stable-partition row maps from object
//! storage.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::UInt16Array;
use async_trait::async_trait;
use lance_core::Result;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache, WeakLanceCache};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::stable_partition::CountsMatrix;
use lance_io::object_store::ObjectStore;
use lance_table::system_index::frag_reuse::mapping::{MappingRef, MappingSource};
use object_store::path::Path;

use super::row_map::{ROW_MAP_CACHE_CHUNK_BYTES, RowMapBlockCache, RowMapReader};
use super::stable_partition::MAPPING_FILE;
use crate::scalar::IndexStore;
use crate::scalar::lance_format::LanceIndexStore;

/// Resolves a row map's base (`None` for the dataset root) to the object
/// store holding it and the base's root path.
pub type BaseResolver = dyn Fn(Option<u32>) -> Result<(Arc<ObjectStore>, Path)> + Send + Sync;

/// Reads row maps under `<base root>/_fri/<map id>/` through the object
/// stores `stores` resolves.
///
/// Opened readers (their counts matrices) and decoded label chunks are kept
/// in `cache`, keyed by the row map's path: row maps are immutable, so an
/// entry stays valid across dataset versions. The cache is held weakly
/// because remappers bound to this source live inside cached indices; a
/// strong reference from there would keep the cache alive through itself.
pub struct RowMapMappingSource {
    stores: Arc<BaseResolver>,
    cache: WeakLanceCache,
}

impl std::fmt::Debug for RowMapMappingSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowMapMappingSource")
            .finish_non_exhaustive()
    }
}

impl RowMapMappingSource {
    pub fn new(stores: Arc<BaseResolver>, cache: WeakLanceCache) -> Self {
        Self { stores, cache }
    }

    async fn open(&self, mapping: &MappingRef) -> Result<Arc<OpenedRowMap>> {
        let (store, root) = (self.stores)(mapping.base_id())?;
        let directory = root.clone().join("_fri").join(mapping.map_id());
        let key = OpenedRowMapKey {
            store: store.store_prefix.clone(),
            path: directory.clone().join(MAPPING_FILE).to_string(),
            binding: Arc::as_ptr(&store) as usize as u64,
        };
        let row_map_id = *blake3::hash(format!("{}:{}", key.store, key.path).as_bytes()).as_bytes();
        // A cache known to be smaller than one chunk can never retain a
        // chunk: every read would load a whole chunk only to discard it, so
        // labels are read directly instead.
        let block_cache = match self.cache.capacity_bytes() {
            Some(capacity) if capacity < ROW_MAP_CACHE_CHUNK_BYTES => None,
            _ => Some(RowMapBlockCache::new(self.cache.clone(), row_map_id)),
        };
        let size_bytes = mapping.size_bytes();
        self.cache
            .get_or_insert_with_key(key, || async move {
                // The opened reader is cached itself; its file metadata needs
                // no second cache.
                let index_store =
                    LanceIndexStore::new(store, directory, Arc::new(LanceCache::no_cache()))
                        .with_file_sizes(HashMap::from([(MAPPING_FILE.to_string(), size_bytes)]));
                let reader = RowMapReader::open_with_cache(
                    index_store.open_index_file(MAPPING_FILE).await?,
                    block_cache,
                )
                .await?;
                Ok(OpenedRowMap(reader))
            })
            .await
    }
}

#[async_trait]
impl MappingSource for RowMapMappingSource {
    async fn counts(&self, mapping: &MappingRef) -> Result<Arc<CountsMatrix>> {
        Ok(self.open(mapping).await?.0.shared_counts())
    }

    async fn read(&self, mapping: &MappingRef, rows: Range<u64>) -> Result<UInt16Array> {
        self.open(mapping).await?.0.labels(rows).await
    }
}

/// An opened row map: its counts matrix, and the reader its labels are read
/// through.
struct OpenedRowMap(RowMapReader);

impl DeepSizeOf for OpenedRowMap {
    // The file's metadata is not cached alongside; labels are cached as
    // chunks of their own.
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.counts().deep_size_of_children(context)
    }
}

struct OpenedRowMapKey {
    store: String,
    path: String,
    /// The object store the reader was opened with, by address: the cached
    /// reader keeps the store alive, so the address is not reused while the
    /// entry exists, and a rotated store (new credentials) opens afresh.
    binding: u64,
}

impl CacheKey for OpenedRowMapKey {
    type ValueType = OpenedRowMap;

    fn key(&self) -> Cow<'_, str> {
        format!("{}:{}:{}", self.binding, self.store, self.path).into()
    }

    fn type_name() -> &'static str {
        "OpenedRowMap"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fri-row-map-reader", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.binding);
        builder.write_str(&self.store);
        builder.write_str(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frag_reuse::row_map::{RowMapWriter, SourceRows};
    use lance_table::system_index::frag_reuse::mapping::MappingSource;

    /// Write a row map of `labels` (one per live source row, all in one
    /// source fragment of `rows` rows with `deleted` rows) in blocks of
    /// `block_rows`, under `root/_fri/map`.
    async fn write_row_map(
        store: &Arc<ObjectStore>,
        root: &Path,
        labels: &[u16],
        deleted: &[u32],
        block_rows: u32,
    ) -> u64 {
        let directory = root.clone().join("_fri").join("map");
        let index_store =
            LanceIndexStore::new(store.clone(), directory, Arc::new(LanceCache::no_cache()));
        let writer = index_store
            .new_index_file(MAPPING_FILE, RowMapWriter::schema())
            .await
            .unwrap();
        let mut row_map = RowMapWriter::try_new_with_block_rows(
            writer,
            vec![SourceRows {
                physical_rows: (labels.len() + deleted.len()) as u64,
                deleted: Some(deleted.iter().copied().collect()),
            }],
            labels.iter().max().map_or(1, |&max| u32::from(max) + 1),
            block_rows,
        )
        .unwrap();
        row_map.append_labels(labels).await.unwrap();
        let (file, _) = row_map.finish().await.unwrap();
        file.size_bytes
    }

    #[tokio::test]
    async fn reads_counts_and_labels_of_a_row_map_through_its_base() {
        let (store, root) = ObjectStore::from_uri("memory://").await.unwrap();
        // Row 2 of 6 was deleted; blocks hold 2 rows.
        let size = write_row_map(&store, &root, &[0, 1, 1, 0, 1], &[2], 2).await;

        let resolved_root = root.clone();
        let cache = LanceCache::with_capacity(64 * 1024 * 1024);
        let source = RowMapMappingSource::new(
            Arc::new(move |base| {
                assert_eq!(base, Some(4));
                Ok((store.clone(), resolved_root.clone()))
            }),
            WeakLanceCache::from(&cache),
        );
        let mapping = MappingRef::new("map", Some(4), size);
        let counts = source.counts(&mapping).await.unwrap();
        assert_eq!(counts.total_rows(), 6);
        assert_eq!(counts.num_blocks(), 3);
        let labels = source.read(&mapping, 2..6).await.unwrap();
        assert_eq!(
            labels,
            UInt16Array::from(vec![None, Some(1), Some(0), Some(1)])
        );
        let error = source.read(&mapping, 4..7).await.unwrap_err();
        assert!(
            matches!(error, lance_core::Error::InvalidInput { .. }),
            "{error}"
        );
    }
}
