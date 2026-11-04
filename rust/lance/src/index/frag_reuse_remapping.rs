// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bridge common FRI address translation to scalar and vector index loading.
use super::frag_reuse_reader::FragmentReuseIndex;
use crate::Dataset;
use async_trait::async_trait;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, WeakLanceCache};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_index::INDEX_FILE_NAME;
use lance_index::frag_reuse::mapping_source::RowMapMappingSource;
use lance_table::format::IndexMetadata;
use lance_table::system_index::frag_reuse::history::FragReuseHistory;
use lance_table::system_index::frag_reuse::ledger::Mapping;
use lance_table::system_index::frag_reuse::mapping::MappingSource;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;

/// Built-in scalar index kinds that apply fragment reuse translation through
/// the deprecated loaders rather than a
/// [`RemappingIndexLoader`](lance_index::scalar::registry::RemappingIndexLoader).
/// A kind leaves this list when it implements one.
const DEPRECATED_LOADER_BUILTINS: &[&str] = &["FmIndexDetails", "MinHashLshIndexDetails"];

/// Whether this build translates the row addresses of scalar segments with
/// `details`: `None` when no plugin reads them.
///
/// A plugin translates when it implements a remapping loader. Built-in
/// kinds not yet on one keep their existing support: every one under a
/// version 0 history (`is_tagged` false), the ones with a batch loader under
/// a tagged history. A plugin outside the build that implements only the
/// deprecated loaders does not translate, under either format.
pub(super) fn scalar_plugin_translates(
    details: &prost_types::Any,
    is_tagged: bool,
) -> Option<bool> {
    let plugin = super::scalar::SCALAR_INDEX_PLUGIN_REGISTRY
        .get_plugin_by_details(details)
        .ok()?;
    if plugin.remapping_loader().is_some() {
        return Some(true);
    }
    let message = details
        .type_url
        .rsplit(['/', '.'])
        .next()
        .unwrap_or_default();
    let is_deprecated_builtin = DEPRECATED_LOADER_BUILTINS
        .iter()
        .any(|builtin| builtin.eq_ignore_ascii_case(message));
    Some(is_deprecated_builtin && (!is_tagged || plugin.supports_batch_row_id_remapping()))
}

/// The source of the row maps `history`'s transitions reference, reading
/// through `dataset`'s base stores and caching in its index cache.
///
/// Every base a row map lives under is resolved now, so the source holds the
/// stores but not the dataset: remappers bound to it live in cached indices.
pub(super) async fn frag_reuse_mapping_source(
    dataset: &Dataset,
    history: &FragReuseHistory,
) -> Result<Arc<dyn MappingSource>> {
    let mut stores = HashMap::new();
    for transition in history.ledger().transitions() {
        let Mapping::StablePartition(reference) = transition.mapping() else {
            continue;
        };
        if stores.contains_key(&reference.base_id) {
            continue;
        }
        let root = match reference.base_id {
            None => dataset.base.clone(),
            Some(id) => dataset
                .manifest
                .base_paths
                .get(&id)
                .ok_or_else(|| {
                    Error::corrupt_file_named(
                        "FRI query",
                        format!("mapping {} references missing base {id}", reference.map_id),
                    )
                })?
                .extract_path(dataset.session.store_registry())?,
        };
        let store = dataset.object_store(reference.base_id).await?;
        stores.insert(reference.base_id, (store, root));
    }
    Ok(Arc::new(
        RowMapMappingSource::new(
            Arc::new(move |base| {
                stores.get(&base).cloned().ok_or_else(|| {
                    Error::internal(format!(
                        "row map base {base:?} is not referenced by the fragment reuse history"
                    ))
                })
            }),
            WeakLanceCache::from(&dataset.index_cache),
        )
        .with_metadata_cache(dataset.metadata_cache.0.clone()),
    ))
}
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
                .join(INDEX_FILE_NAME);
            let store = dataset.object_store_for_index(index).await?;
            let reader = super::vector::open_index_file(
                store.as_ref(),
                &path,
                INDEX_FILE_NAME,
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
