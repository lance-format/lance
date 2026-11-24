// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stable-partition mapping for callers that supply an index store.

use std::sync::Arc;

use async_trait::async_trait;
use lance_core::Result;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::fragment_reuse::MappingReader;
use lance_table::format::pb::fragment_reuse_index_details::FragmentDigest;
use lance_table::system_index::frag_reuse::stable_partition;

use super::row_map::{RowMapBlockCache, RowMapReader};
use crate::scalar::IndexStore;

pub use stable_partition::MAPPING_FILE;

/// Index-store adapter for the table-format stable-partition mapping reader.
pub struct StablePartitionMapping(stable_partition::StablePartitionMapping);

impl StablePartitionMapping {
    /// Bind an index store to source scan order and destination label order.
    /// Cached label chunks are used when `block_cache` is present.
    pub fn try_new_with_cache(
        store: Arc<dyn IndexStore>,
        sources: Vec<FragmentDigest>,
        destinations: Vec<FragmentDigest>,
        block_cache: Option<RowMapBlockCache>,
    ) -> Result<Self> {
        stable_partition::StablePartitionMapping::try_new(sources, destinations, move || {
            let store = store.clone();
            let block_cache = block_cache.clone();
            async move {
                let file = store.open_index_file(MAPPING_FILE).await?;
                Ok(RowMapReader::open_with_cache(file, block_cache).await?.0)
            }
        })
        .map(Self)
    }

    /// Bind an index store and read label ranges directly on demand.
    pub fn try_new(
        store: Arc<dyn IndexStore>,
        sources: Vec<FragmentDigest>,
        destinations: Vec<FragmentDigest>,
    ) -> Result<Self> {
        Self::try_new_with_cache(store, sources, destinations, None)
    }
}

impl std::fmt::Debug for StablePartitionMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl DeepSizeOf for StablePartitionMapping {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}

#[async_trait]
impl MappingReader for StablePartitionMapping {
    async fn remap_row_id(&self, row_id: u64) -> Result<Option<u64>> {
        self.0.remap_row_id(row_id).await
    }

    async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
        self.0.remap_row_ids(row_ids).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frag_reuse::row_map::{RowMapWriter, SourceRows};
    use crate::scalar::lance_format::LanceIndexStore;
    use lance_core::Error;
    use lance_core::cache::LanceCache;
    use lance_core::utils::address::RowAddress;
    use lance_io::object_store::ObjectStore;

    #[tokio::test]
    async fn mapping_single_and_batch_translation_are_lazy() {
        let (object_store, path) = ObjectStore::from_uri("memory://").await.unwrap();
        let store = Arc::new(LanceIndexStore::new(
            object_store,
            path,
            Arc::new(LanceCache::with_capacity(1024 * 1024)),
        ));
        let source = FragmentDigest {
            id: 1,
            physical_rows: 5,
            num_deleted_rows: 1,
        };
        let destinations = vec![
            FragmentDigest {
                id: 2,
                physical_rows: 2,
                num_deleted_rows: 0,
            },
            FragmentDigest {
                id: 3,
                physical_rows: 2,
                num_deleted_rows: 0,
            },
        ];
        let mapping =
            StablePartitionMapping::try_new(store.clone(), vec![source], destinations).unwrap();
        assert!(mapping.remap_row_ids(&[]).await.unwrap().is_empty());
        // Empty input must not open the file.
        let writer = store
            .new_index_file(MAPPING_FILE, RowMapWriter::schema())
            .await
            .unwrap();
        let mut writer = RowMapWriter::try_new_with_block_rows(
            writer,
            vec![SourceRows {
                physical_rows: 5,
                deleted: Some([1].into_iter().collect()),
            }],
            2,
            2,
        )
        .unwrap();
        writer.append_labels(&[1, 0, 1, 0]).await.unwrap();
        writer.finish().await.unwrap();
        let addr = |fragment, offset| u64::from(RowAddress::new_from_parts(fragment, offset));
        assert_eq!(
            mapping
                .remap_row_ids(&[addr(1, 4), addr(1, 1), addr(1, 0), addr(1, 0), addr(1, 2)])
                .await
                .unwrap(),
            vec![
                Some(addr(2, 1)),
                None,
                Some(addr(3, 0)),
                Some(addr(3, 0)),
                Some(addr(2, 0))
            ]
        );
        assert_eq!(
            mapping.remap_row_id(addr(1, 4)).await.unwrap(),
            Some(addr(2, 1))
        );
        assert_eq!(mapping.remap_row_id(addr(1, 1)).await.unwrap(), None);
        for (row_id, message) in [
            (addr(9, 0), "outside stable-partition sources"),
            (addr(1, 5), "exceeds source length"),
        ] {
            for error in [
                mapping.remap_row_id(row_id).await.unwrap_err(),
                mapping.remap_row_ids(&[row_id]).await.unwrap_err(),
            ] {
                assert!(matches!(error, Error::InvalidInput { .. }));
                assert!(error.to_string().contains(message));
            }
        }
    }
}
