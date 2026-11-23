// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read and stage index segments whose files keep the fragment ids from training.
//!
//! [`IndexMetadata::local_fragment_aliases`] maps each stored fragment id to
//! the committed id. The translation runs once, before fragment-reuse
//! translation. A row address keeps its row offset. An id missing from the map
//! keeps its stored value.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{RecordBatch, UInt64Array};
use async_trait::async_trait;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_index::scalar::{
    BatchRowIdRemapper, BuiltinIndexType, RowAddrTranslator, RowAddrTranslatorRef, RowIdRemapper,
    ScalarIndexParams,
};
use lance_index::{IndexParams, IndexType};
use lance_select::{RowAddrSelection, RowAddrTreeMap};
use lance_table::format::{IndexMetadata, Manifest};
use roaring::RoaringTreemap;

use super::scalar::SCALAR_INDEX_PLUGIN_REGISTRY;
use crate::Dataset;

/// Translates stored fragment ids, then applies `then` when a fragment-reuse
/// remapper is also in effect.
#[derive(Debug)]
pub struct LocalFragmentRemapper {
    aliases: BTreeMap<u32, u32>,
    then: Option<Arc<dyn RowIdRemapper>>,
}

impl LocalFragmentRemapper {
    /// `then` when the segment stores committed ids. Otherwise this remapper
    /// in front of `then`.
    pub(crate) fn compose(
        index: &IndexMetadata,
        then: Option<Arc<dyn RowIdRemapper>>,
    ) -> Option<Arc<dyn RowIdRemapper>> {
        if index.local_fragment_aliases.is_empty() {
            return then;
        }
        Some(Arc::new(Self {
            aliases: index.local_fragment_aliases.clone(),
            then,
        }))
    }

    fn translate(&self, address: u64) -> u64 {
        translate_address(&self.aliases, address)
    }
}

fn translate_address(aliases: &BTreeMap<u32, u32>, address: u64) -> u64 {
    let address = RowAddress::from(address);
    match aliases.get(&address.fragment_id()) {
        Some(committed) => u64::from(RowAddress::new_from_parts(*committed, address.row_offset())),
        None => u64::from(address),
    }
}

impl RowIdRemapper for LocalFragmentRemapper {
    fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        let translated = self.translate(row_id);
        match &self.then {
            Some(then) => then.remap_row_id(translated),
            None => Some(translated),
        }
    }

    fn remap_row_addrs_tree_map(&self, row_addrs: &RowAddrTreeMap) -> RowAddrTreeMap {
        let mut translated = RowAddrTreeMap::new();
        for (fragment, selection) in row_addrs.iter() {
            let fragment = self.aliases.get(fragment).copied().unwrap_or(*fragment);
            let mut selected = RowAddrTreeMap::new();
            match selection {
                RowAddrSelection::Full => selected.insert_fragment(fragment),
                RowAddrSelection::Partial(rows) => selected.insert_bitmap(fragment, rows.clone()),
            }
            translated |= selected;
        }
        match &self.then {
            Some(then) => then.remap_row_addrs_tree_map(&translated),
            None => translated,
        }
    }

    fn remap_row_ids_roaring_tree_map(&self, row_ids: &RoaringTreemap) -> RoaringTreemap {
        RoaringTreemap::from_iter(row_ids.iter().filter_map(|id| self.remap_row_id(id)))
    }

    fn remap_row_ids_record_batch(
        &self,
        batch: RecordBatch,
        row_id_idx: usize,
    ) -> Result<RecordBatch> {
        let addresses = batch
            .column(row_id_idx)
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "column {row_id_idx} of an index batch holds row addresses and must be UInt64, not {}",
                    batch.column(row_id_idx).data_type()
                ))
            })?;
        let translated = UInt64Array::from_unary(addresses, |address| self.translate(address));
        let mut columns = batch.columns().to_vec();
        columns[row_id_idx] = Arc::new(translated);
        let batch = RecordBatch::try_new(batch.schema(), columns)?;
        match &self.then {
            Some(then) => then.remap_row_ids_record_batch(batch, row_id_idx),
            None => Ok(batch),
        }
    }
}

/// Cache namespace for pages decoded through the segment's map.
///
/// The same index files can be read with a different map after a commit retry.
/// `None` when the segment stores committed ids.
pub fn alias_cache_namespace(index: &IndexMetadata) -> Option<String> {
    if index.local_fragment_aliases.is_empty() {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"local-fragment-aliases/1");
    for (stored, committed) in &index.local_fragment_aliases {
        hasher.update(&stored.to_le_bytes());
        hasher.update(&committed.to_le_bytes());
    }
    Some(format!("local-frag/{}", hasher.finalize().to_hex()))
}

/// Compose `index.local_fragment_aliases` ahead of `then` when reading an
/// index segment through a tagged (asynchronous batch) fragment-reuse remapper.
pub fn compose_batch_remapper(
    index: &IndexMetadata,
    then: Arc<dyn BatchRowIdRemapper>,
) -> Arc<dyn BatchRowIdRemapper> {
    if index.local_fragment_aliases.is_empty() {
        return then;
    }
    Arc::new(LocalFragmentBatchRemapper {
        aliases: index.local_fragment_aliases.clone(),
        fragment_rows: HashMap::new(),
        then: RowAddrTranslator::batch(then),
    })
}

/// Compose `index.local_fragment_aliases` ahead of `translator` when rewriting
/// an index segment's files so stored provisional ids resolve to committed ids
/// before `translator` maps committed ids to destination ids.
pub fn prepend_fragment_aliases<'a>(
    dataset: &Dataset,
    index: &IndexMetadata,
    translator: &'a RowAddrTranslator,
) -> Cow<'a, RowAddrTranslator> {
    if index.local_fragment_aliases.is_empty() {
        return Cow::Borrowed(translator);
    }
    let fragment_rows = dataset
        .manifest
        .fragments
        .iter()
        .filter_map(|fragment| {
            let id = u32::try_from(fragment.id).ok()?;
            let rows = u64::try_from(fragment.physical_rows?).ok()?;
            Some((id, rows))
        })
        .collect();
    Cow::Owned(RowAddrTranslator::batch(Arc::new(
        LocalFragmentBatchRemapper {
            aliases: index.local_fragment_aliases.clone(),
            fragment_rows,
            then: translator.clone(),
        },
    )))
}

#[derive(Debug)]
struct LocalFragmentBatchRemapper {
    aliases: BTreeMap<u32, u32>,
    fragment_rows: HashMap<u32, u64>,
    then: RowAddrTranslator,
}

#[async_trait]
impl BatchRowIdRemapper for LocalFragmentBatchRemapper {
    async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
        let committed: Vec<u64> = row_ids
            .iter()
            .map(|&address| translate_address(&self.aliases, address))
            .collect();
        self.then.remap_row_addrs(&committed).await
    }

    fn fragment_physical_rows(&self, fragment: u32) -> Option<u64> {
        let committed = self.aliases.get(&fragment).copied().unwrap_or(fragment);
        self.fragment_rows
            .get(&committed)
            .copied()
            .or_else(|| match self.then.as_ref() {
                RowAddrTranslatorRef::Batch(batch) => batch.fragment_physical_rows(committed),
                _ => None,
            })
    }

    fn materialization_budget_bytes(&self) -> u64 {
        match self.then.as_ref() {
            RowAddrTranslatorRef::Batch(batch) => batch.materialization_budget_bytes(),
            _ => lance_index::scalar::DEFAULT_MATERIALIZATION_BUDGET_BYTES,
        }
    }
}

/// Reject an index before training if its row addresses cannot be rebased
/// with `local_fragment_aliases`.
pub fn ensure_index_can_rebase_local_fragments(
    index_type: IndexType,
    params: &dyn IndexParams,
) -> Result<()> {
    match index_type {
        IndexType::FragmentReuse => Err(Error::invalid_input(
            "Fragment reuse index can only be created through compaction",
        )),
        IndexType::MemWal => Err(Error::not_supported(
            "Staged create_index cannot build the MemWAL system index",
        )),
        IndexType::Vector
        | IndexType::IvfFlat
        | IndexType::IvfSq
        | IndexType::IvfPq
        | IndexType::IvfHnswSq
        | IndexType::IvfHnswPq
        | IndexType::IvfHnswFlat
        | IndexType::IvfRq
        | IndexType::IvfHnswRq => Err(Error::not_supported(
            "Staged create_index cannot build a vector index over uncommitted fragments because vector indices store logical row ids that a commit retry can reassign",
        )),
        IndexType::Inverted => Err(Error::not_supported(
            "Staged create_index cannot build an inverted index over uncommitted fragments because inverted indices store logical row ids that a commit retry can reassign",
        )),
        IndexType::Scalar => {
            let scalar_params = params
                .as_any()
                .downcast_ref::<ScalarIndexParams>()
                .ok_or_else(|| {
                    Error::invalid_input("Scalar index type must take a ScalarIndexParams")
                })?;
            ensure_scalar_plugin_can_rebase_local_fragments(&scalar_params.index_type)
        }
        IndexType::BTree
        | IndexType::Bitmap
        | IndexType::LabelList
        | IndexType::NGram
        | IndexType::ZoneMap
        | IndexType::BloomFilter
        | IndexType::RTree
        | IndexType::Fm
        | IndexType::MinHashLsh => {
            let builtin = BuiltinIndexType::try_from(index_type)?;
            ensure_scalar_plugin_can_rebase_local_fragments(builtin.as_str())
        }
    }
}

fn ensure_scalar_plugin_can_rebase_local_fragments(plugin_name: &str) -> Result<()> {
    if plugin_name.eq_ignore_ascii_case("inverted") || plugin_name.eq_ignore_ascii_case("fts") {
        return Err(Error::not_supported(
            "Staged create_index cannot build an inverted index over uncommitted fragments because inverted indices store logical row ids that a commit retry can reassign",
        ));
    }
    let plugin = SCALAR_INDEX_PLUGIN_REGISTRY.get_plugin_by_name(plugin_name)?;
    if !plugin.supports_batch_row_id_remapping() {
        return Err(Error::not_supported(format!(
            "Staged create_index over uncommitted fragments requires local_fragment_aliases support, which {} indices do not implement yet",
            plugin.name()
        )));
    }
    Ok(())
}

/// Check that a trained segment can be rebased with `local_fragment_aliases`.
pub fn ensure_segment_can_rebase_local_fragments(
    base_manifest: &Manifest,
    segment: &IndexMetadata,
) -> Result<()> {
    let details = segment.index_details.as_ref().ok_or_else(|| {
        Error::not_supported(format!(
            "Staged index {} has no index details",
            segment.name
        ))
    })?;
    let plugin = SCALAR_INDEX_PLUGIN_REGISTRY.get_plugin_by_details(details.as_ref())?;
    if !plugin.supports_batch_row_id_remapping() {
        return Err(Error::not_supported(format!(
            "Staged index {} uses {}, which does not implement local_fragment_aliases support yet",
            segment.name,
            plugin.name()
        )));
    }
    if base_manifest.uses_stable_row_ids() && !segment.results_are_row_addrs() {
        return Err(Error::not_supported(format!(
            "Staged index {} stores logical row ids, and a retry can reassign those",
            segment.name
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
