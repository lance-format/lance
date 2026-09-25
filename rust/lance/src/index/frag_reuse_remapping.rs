// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bridge common FRI address translation to scalar and vector index loading.
use super::frag_reuse_reader::FragmentReuseIndex;
use crate::Dataset;
use async_trait::async_trait;
use lance_core::Result;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::utils::address::RowAddress;
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use std::sync::Arc;
#[derive(Clone)]
struct VectorFormatKey(uuid::Uuid);

impl CacheKey for VectorFormatKey {
    type ValueType = bool;
    fn key(&self) -> std::borrow::Cow<'_, str> {
        self.0.to_string().into()
    }
    fn type_name() -> &'static str {
        "VectorBatchRemapping"
    }
    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.vector-batch-remapping", 1)
    }
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.0.as_bytes());
    }
}

#[derive(Clone)]
struct InvertedFormatKey(uuid::Uuid);

impl CacheKey for InvertedFormatKey {
    type ValueType = bool;
    fn key(&self) -> std::borrow::Cow<'_, str> {
        self.0.to_string().into()
    }
    fn type_name() -> &'static str {
        "InvertedBatchRemapping"
    }
    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.inverted-batch-remapping", 1)
    }
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.0.as_bytes());
    }
}

/// Whether a full-text segment can be loaded under a translating remapper.
///
/// The plugin supports batch remapping, but a legacy-layout index (no
/// `metadata.lance`; the loader's own format probe) stores per-document
/// lengths aligned with the row ids it was built with and cannot be
/// translated. Such a segment is excluded from coverage so the planner scans
/// its fragments; an untouched (identity) legacy segment still loads.
pub async fn inverted_supports_batch_remapping(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> Result<bool> {
    let supported = dataset
        .index_cache
        .get_or_insert_with_key(InvertedFormatKey(index.uuid), || async {
            use crate::dataset::index::LanceIndexStoreExt;
            let store =
                lance_index::scalar::lance_format::LanceIndexStore::from_dataset_for_existing(
                    dataset, index,
                )
                .await?;
            Ok(lance_index::scalar::IndexStore::open_index_file(
                &store,
                lance_index::scalar::inverted::METADATA_FILE,
            )
            .await
            .is_ok())
        })
        .await?;
    Ok(*supported)
}

/// Legacy vector file readers keep their existing V1 behavior and scan fallback.
pub async fn vector_supports_batch_remapping(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> Result<bool> {
    let supported = dataset
        .index_cache
        .get_or_insert_with_key(VectorFormatKey(index.uuid), || async {
            let path = dataset
                .indice_files_dir(index)?
                .join(index.uuid.to_string())
                .join(super::INDEX_FILE_NAME);
            let store = dataset.object_store_for_index(index).await?;
            let reader = super::vector::open_index_file(
                store.as_ref(),
                &path,
                super::INDEX_FILE_NAME,
                &index.file_size_map(),
            )
            .await?;
            let tail = lance_io::utils::read_last_block(reader.as_ref()).await?;
            Ok(matches!(
                lance_io::utils::read_version(&tail)?,
                (0, 3) | (2, _)
            ))
        })
        .await?;
    Ok(*supported)
}

/// Applies the shared FRI history and limits translated rows to this segment's coverage.
pub(super) struct QueryRowIdRemapper {
    mapping: Arc<FragmentReuseIndex>,
    coverage: RoaringBitmap,
    excluded_fragments: RoaringBitmap,
}

impl std::fmt::Debug for QueryRowIdRemapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryRowIdRemapper")
            .field("coverage", &self.coverage)
            .field("excluded_fragments", &self.excluded_fragments)
            .finish_non_exhaustive()
    }
}

impl QueryRowIdRemapper {
    pub(crate) fn new(
        mapping: Arc<FragmentReuseIndex>,
        coverage: RoaringBitmap,
        excluded_fragments: RoaringBitmap,
    ) -> Self {
        Self {
            mapping,
            coverage,
            excluded_fragments,
        }
    }
}

#[async_trait]
impl lance_index::scalar::BatchRowIdRemapper for QueryRowIdRemapper {
    async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
        lance_index::scalar::check_batch_remapping_entry()?;
        Ok(self
            .mapping
            .remap_row_ids_excluding(row_ids, &self.excluded_fragments)
            .await?
            .into_iter()
            .map(|address| {
                address.filter(|row_id| {
                    self.coverage
                        .contains(RowAddress::from(*row_id).fragment_id())
                })
            })
            .collect())
    }
}
