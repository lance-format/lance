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
/// A JSON index answers through its target index, so it translates only
/// when its target's plugin does.
pub(super) fn scalar_plugin_translates(details: &prost_types::Any) -> Option<bool> {
    translates_through_targets(details, &|details| {
        super::scalar::SCALAR_INDEX_PLUGIN_REGISTRY
            .get_plugin_by_details(details)
            .ok()
            .map(|plugin| plugin.translates_row_addrs())
    })
}

/// [`scalar_plugin_translates`], with `plugin_translates` answering for the
/// plugin of one index's details.
fn translates_through_targets(
    details: &prost_types::Any,
    plugin_translates: &dyn Fn(&prost_types::Any) -> Option<bool>,
) -> Option<bool> {
    if !plugin_translates(details)? {
        return Some(false);
    }
    if details.type_url.ends_with(&JsonIndexDetails::full_name()) {
        let json = details.to_msg::<JsonIndexDetails>().ok()?;
        return translates_through_targets(json.target_details.as_ref()?, plugin_translates);
    }
    Some(true)
}

/// The source of the row maps `history`'s transitions reference, reading
/// through `dataset`'s base stores and caching in its index cache.
///
/// Every base a row map lives under is resolved now, so the source holds the
/// stores but not the dataset: remappers bound to it live in cached indices.
pub async fn frag_reuse_mapping_source(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn details(type_url: &str) -> prost_types::Any {
        prost_types::Any {
            type_url: type_url.to_string(),
            value: Vec::new(),
        }
    }

    fn json_over(target: Option<prost_types::Any>) -> prost_types::Any {
        prost_types::Any::from_msg(&JsonIndexDetails {
            path: "x".to_string(),
            target_details: target,
            ..Default::default()
        })
        .unwrap()
    }

    /// A plain index translates when its plugin does; a JSON index answers
    /// through its target, so it translates only when the target's plugin
    /// does, however deeply nested. Details no plugin reads, and a JSON index
    /// without a target, are not known to translate either way.
    #[test]
    fn json_indices_translate_through_their_target() {
        let plugin_translates = |details: &prost_types::Any| match details.type_url.as_str() {
            "translating" => Some(true),
            "stale" => Some(false),
            url if url.ends_with(&JsonIndexDetails::full_name()) => Some(true),
            _ => None,
        };
        let translates =
            |details: &prost_types::Any| translates_through_targets(details, &plugin_translates);
        assert_eq!(translates(&details("translating")), Some(true));
        assert_eq!(translates(&details("stale")), Some(false));
        assert_eq!(translates(&details("unknown")), None);
        assert_eq!(
            translates(&json_over(Some(details("translating")))),
            Some(true)
        );
        assert_eq!(translates(&json_over(Some(details("stale")))), Some(false));
        assert_eq!(
            translates(&json_over(Some(json_over(Some(details("stale")))))),
            Some(false)
        );
        assert_eq!(translates(&json_over(Some(details("unknown")))), None);
        assert_eq!(translates(&json_over(None)), None);
    }
}
