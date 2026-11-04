// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bridge common FRI address translation to scalar and vector index loading.
use crate::Dataset;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, WeakLanceCache};
use lance_core::{Error, Result};
use lance_index::INDEX_FILE_NAME;
use lance_index::frag_reuse::mapping_source::RowMapMappingSource;
use lance_index::pb::JsonIndexDetails;
use lance_table::format::IndexMetadata;
use lance_table::system_index::frag_reuse::history::FragReuseHistory;
use lance_table::system_index::frag_reuse::ledger::Mapping;
use lance_table::system_index::frag_reuse::mapping::MappingSource;
use prost::Name;
use std::collections::HashMap;
use std::sync::Arc;

/// Whether this build translates the row addresses of scalar segments with
/// `details`: `None` when no plugin reads them.
///
/// A plugin translates when it implements a remapping loader. One that
/// implements only the deprecated loaders does not, under any history
/// format. A JSON index answers through its target index, so it translates
/// only when its target's plugin does.
pub(super) fn scalar_plugin_translates(details: &prost_types::Any) -> Option<bool> {
    let plugin = super::scalar::SCALAR_INDEX_PLUGIN_REGISTRY
        .get_plugin_by_details(details)
        .ok()?;
    if plugin.remapping_loader().is_none() {
        return Some(false);
    }
    if details.type_url.ends_with(&JsonIndexDetails::full_name()) {
        let json = details.to_msg::<JsonIndexDetails>().ok()?;
        return scalar_plugin_translates(json.target_details.as_ref()?);
    }
    Some(true)
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
