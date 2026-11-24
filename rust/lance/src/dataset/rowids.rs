// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub(crate) mod cache;
pub(crate) mod group;
pub(crate) mod resolve;
mod spill;
#[cfg(test)]
mod validate;

use super::Dataset;
use crate::io::deletion::read_dataset_deletion_file;
use crate::session::index_caches::RowIdIndexKey;
use crate::{Error, Result};
use futures::{Stream, StreamExt, TryFutureExt, TryStreamExt};
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::address::RowAddress;
use lance_select::{
    NullableRowAddrMask, NullableRowAddrSet, RowAddrMask, RowAddrSelection, RowAddrTreeMap,
};
use lance_table::{
    format::{Fragment, RowDatasetVersionSequence, RowIdMeta},
    rowids::{
        RowIdIndex, RowIdSequence, read_row_ids, segment::U64Segment, version::SpilledRowLineage,
    },
};
use std::borrow::Cow;
use std::sync::Arc;

pub use lance_table_runtime::rowids::RowVersionKind;
pub(crate) use spill::place_carried_row_lineage;
pub use spill::{
    DEFAULT_INLINE_ROW_LINEAGE_MAX_BYTES, INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY,
    PlacedRowLineage, RowLineage, SPILL_ROW_LINEAGE_CONFIG_KEY, inline_row_lineage_max_bytes,
    place_row_lineage,
};
pub(crate) use spill::{RowLineagePlan, RowLineageSpill, plan_row_lineage_spill};

/// Load a row id sequence from the given dataset and fragment.
pub async fn load_row_id_sequence(
    dataset: &Dataset,
    fragment: &Fragment,
) -> Result<Arc<RowIdSequence>> {
    lance_table_runtime::rowids::load_row_id_sequence(&dataset.table_view().to_table(), fragment)
        .await
}

/// Decode the row id sequence of `fragment`, bypassing every cache.
async fn read_row_id_sequence(dataset: &Dataset, fragment: &Fragment) -> Result<RowIdSequence> {
    match &fragment.row_id_meta {
        None => Err(Error::internal("Missing row id meta")),
        Some(RowIdMeta::Inline(data)) => read_row_ids(data),
        Some(RowIdMeta::Column) => read_spilled_row_ids(dataset, fragment).await,
    }
}

/// Load one of the fragment's per-row version sequences, wherever it is stored.
///
/// Requires stable row ids. When version metadata is absent, readers use version 1.
pub async fn load_row_version_sequence(
    dataset: &Dataset,
    fragment: &Fragment,
    kind: RowVersionKind,
) -> Result<Option<Arc<RowDatasetVersionSequence>>> {
    lance_table_runtime::rowids::load_row_version_sequence(
        &dataset.table_view().to_table(),
        fragment,
        kind,
    )
    .await
}

/// Read spilled row lineage before building a manifest that needs it.
pub async fn load_spilled_row_lineage<'a>(
    dataset: &Dataset,
    fragments: impl IntoIterator<Item = &'a Fragment>,
) -> Result<Arc<SpilledRowLineage>> {
    lance_table_runtime::rowids::load_spilled_row_lineage(
        &dataset.table_view().to_table(),
        fragments,
    )
    .await
}

/// Read the fragment's row ID sequence from its carrier data file.
pub async fn read_spilled_row_ids(dataset: &Dataset, fragment: &Fragment) -> Result<RowIdSequence> {
    lance_table_runtime::rowids::read_spilled_row_ids(&dataset.table_view().to_table(), fragment)
        .await
}

/// Read the version sequence stored under the given reserved field id.
pub async fn read_spilled_versions(
    dataset: &Dataset,
    fragment: &Fragment,
    field_id: i32,
) -> Result<RowDatasetVersionSequence> {
    lance_table_runtime::rowids::read_spilled_versions(
        &dataset.table_view().to_table(),
        fragment,
        field_id,
    )
    .await
}

pub(super) async fn validate_stable_row_ids(dataset: &Dataset) -> Result<()> {
    lance_table_runtime::rowids::validate_stable_row_ids(&dataset.table_view().to_table()).await
}

/// Load row id sequences from the given dataset and fragments.
///
/// Returned as a stream of (fragment_id, sequence) pairs. These are not
/// guaranteed to be in the same order as the input fragments.
pub fn load_row_id_sequences<'a>(
    dataset: &'a Dataset,
    fragments: &'a [Fragment],
) -> impl Stream<Item = Result<(u32, Arc<RowIdSequence>)>> + 'a {
    let table = Arc::new(dataset.table_view().to_table());
    futures::stream::iter(fragments)
        .map(move |fragment| {
            let table = table.clone();
            async move {
                lance_table_runtime::rowids::load_row_id_sequence(&table, fragment)
                    .map_ok(move |seq| (fragment.id as u32, seq))
                    .await
            }
        })
        .buffer_unordered(dataset.table_state.object_store.io_parallelism())
}

pub async fn get_row_id_index(dataset: &Dataset) -> Result<Option<Arc<RowIdIndex>>> {
    get_row_id_index_with(dataset, true).await
}

pub(crate) use resolve::resolve_row_ids;

/// The row id index, built with or without the deletion vectors.
///
/// Without them the index maps a deleted row id to its address instead of
/// `None`, but reads no deletion file: on a table with many fragments that is
/// one request per fragment saved for a caller that already knows its ids are
/// live.
pub(crate) async fn get_row_id_index_with(
    dataset: &Dataset,
    with_deletions: bool,
) -> Result<Option<Arc<RowIdIndex>>> {
    lance_table_runtime::rowids::get_row_id_index_with_cache(
        &dataset.table_view().to_table(),
        with_deletions,
        &dataset.index_cache,
    )
    .await
}

/// A dense fragment maps an ID to a physical offset by subtraction. Cache only
/// its endpoints and manifest slot. Prefix maxima let a point lookup skip
/// disjoint ranges and still find old and new copies after an update.
#[derive(DeepSizeOf)]
struct InlineRowIdRange {
    start: u64,
    end: u64,
    prefix_end: u64,
    slot: usize,
}

struct InlineRowIdRangesKey<'a> {
    version: u64,
    e_tag: &'a str,
}

impl CacheKey for InlineRowIdRangesKey<'_> {
    type ValueType = Option<Vec<InlineRowIdRange>>;

    fn key(&self) -> Cow<'_, str> {
        format!("inline_row_id_ranges/{}/{}", self.version, self.e_tag).into()
    }

    fn type_name() -> &'static str {
        "InlineRowIdRanges"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.inline-row-id-ranges", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.version);
        builder.write_str(self.e_tag);
    }
}

fn inline_row_id_ranges(dataset: &Dataset) -> Result<Option<Vec<InlineRowIdRange>>> {
    let mut ranges = Vec::new();
    for (slot, fragment) in dataset.iter_fragments().enumerate() {
        let Some(RowIdMeta::Inline(data)) = &fragment.row_id_meta else {
            return Ok(None);
        };
        // A protobuf sequence with one Range (two u64 varints) fits in 32
        // bytes. Avoid decoding an arbitrarily large inline array to reject it.
        if data.len() > 32 {
            return Ok(None);
        }
        let sequence = read_row_ids(data)?;
        let [U64Segment::Range(range)] = sequence.segments() else {
            return Ok(None);
        };
        // Leave legacy or unrepresentable physical addresses to the existing index.
        if u32::try_from(fragment.id).is_err()
            || range.end.saturating_sub(range.start) > u64::from(u32::MAX) + 1
            || slot > 0 && dataset.manifest.fragments[slot - 1].id >= fragment.id
        {
            return Ok(None);
        }
        if range.is_empty() {
            continue;
        }
        ranges.push(InlineRowIdRange {
            start: range.start,
            end: range.end,
            prefix_end: 0,
            slot,
        });
    }
    ranges.sort_unstable_by_key(|range| range.start);
    let mut prefix_end = 0;
    for range in &mut ranges {
        prefix_end = prefix_end.max(range.end);
        range.prefix_end = prefix_end;
    }
    Ok(Some(ranges))
}

/// Resolve one stable ID using only a dense inline snapshot and its candidate
/// deletion files. The outer None means the snapshot needs the general index;
/// the inner None means the ID is missing or deleted.
async fn lookup_inline_row_id(
    dataset: &Dataset,
    row_id: u64,
    with_deletions: bool,
) -> Result<Option<Option<RowAddress>>> {
    let ranges = if let Some(e_tag) = dataset.table_state.manifest_location.e_tag.as_deref() {
        let key = InlineRowIdRangesKey {
            version: dataset.manifest.version,
            e_tag,
        };
        dataset
            .index_cache
            .get_or_insert_with_key(key, || async { inline_row_id_ranges(dataset) })
            .await?
    } else {
        // As with RowIdIndex, an unversioned cache identity cannot be shared
        // across a drop and recreate at the same URI.
        Arc::new(inline_row_id_ranges(dataset)?)
    };
    let Some(ranges) = ranges.as_ref() else {
        return Ok(None);
    };
    let lower = ranges.partition_point(|range| range.prefix_end <= row_id);
    let upper = ranges.partition_point(|range| range.start <= row_id);
    let mut lookups = Vec::new();
    for idx in lower..upper {
        let range = &ranges[idx];
        if range.end <= row_id {
            continue;
        }
        let fragment = &dataset.manifest.fragments[range.slot];
        let address = RowAddress::new_from_parts(fragment.id as u32, (row_id - range.start) as u32);
        // The general ignore-deletions index also applies tombstones to a
        // fragment whose span overlaps another, even for its unique IDs.
        let overlaps = idx > 0 && ranges[idx - 1].prefix_end > range.start
            || ranges
                .get(idx + 1)
                .is_some_and(|next| next.start < range.end);
        lookups.push(async move {
            if (with_deletions || overlaps)
                && let Some(file) = &fragment.deletion_file
                && read_dataset_deletion_file(dataset, fragment.id, file)
                    .await?
                    .contains(address.row_offset())
            {
                return Ok(None);
            }
            Ok::<_, Error>(Some(address))
        });
    }
    let mut live = None;
    let mut candidates = futures::stream::iter(lookups)
        .buffer_unordered(dataset.table_state.object_store.io_parallelism());
    while let Some(candidate) = candidates.try_next().await? {
        if let Some(address) = candidate
            && live.replace(address).is_some()
        {
            return Err(Error::internal(format!(
                "row id index corrupt: stable row id {row_id} is live in multiple fragments"
            )));
        }
    }
    Ok(Some(live))
}

pub(super) async fn resolve_take_row_ids(
    dataset: &Dataset,
    row_ids: &[u64],
    with_deletions: bool,
) -> Result<Option<Vec<Option<RowAddress>>>> {
    if !dataset.manifest.uses_stable_row_ids() {
        return Ok(None);
    }
    if let [id] = row_ids {
        if let Some(e_tag) = dataset.table_state.manifest_location.e_tag.as_deref()
            && let Some(index) = dataset
                .index_cache
                .get_with_key(&RowIdIndexKey {
                    version: dataset.manifest.version,
                    with_deletions,
                    e_tag: Some(e_tag),
                })
                .await
        {
            return Ok(Some(vec![index.get(*id)?]));
        }
        if let Some(address) = lookup_inline_row_id(dataset, *id, with_deletions).await? {
            return Ok(Some(vec![address]));
        }
    }
    resolve_row_ids(dataset, row_ids, with_deletions).await
}

/// The [`RowIdIndex`] a row-id -> row-address translation needs, or `None` when
/// the dataset has no stable row ids and an id already *is* an address, so there
/// is nothing to translate.
///
/// [`get_row_id_index`] returns `None` for exactly that one case, so a stable-row-id
/// dataset without an index is a broken invariant rather than a case for the caller
/// to guess at — guessing is how a block-list mask ends up inverted into "allow
/// nothing". Every caller that brings a row-id-domain value into the address domain
/// goes through here.
pub(crate) async fn row_id_index_for_translation(
    dataset: &Dataset,
) -> Result<Option<Arc<RowIdIndex>>> {
    if !dataset.manifest.uses_stable_row_ids() {
        return Ok(None);
    }
    get_row_id_index(dataset).await?.map(Some).ok_or_else(|| {
        Error::internal("dataset uses stable row ids but has no row id index to translate with")
    })
}

/// Map a set of physical row addresses to their stable row ids
///
/// A fragment's [`RowIdSequence`] holds one id per physical row in offset order;
/// deletions are tracked by the deletion vector and do not compact the sequence,
/// so a physical offset indexes the sequence directly (see [`RowIdIndex`], which
/// maps ids to `start_address + position` and filters deletions separately).
/// Addresses that no longer have a live counterpart are dropped, which is correct:
/// those rows are not part of the answer. That covers both a deleted row and a
/// whole fragment that a maintenance operation replaced — an address-domain index
/// is not remapped by compaction or `update` (neither the zone map nor the bloom
/// filter index supports remap), so its results outlive the fragments they address.
/// The replacement fragments are absent from the index's fragment bitmap, so the
/// scanner routes them to a full scan and no match is lost.
pub(crate) async fn translate_addr_treemap_to_row_ids(
    dataset: &Dataset,
    addrs: &RowAddrTreeMap,
) -> Result<RowAddrTreeMap> {
    let mut row_ids = RowAddrTreeMap::new();
    for (fragment_id, selection) in addrs.iter() {
        let Some(file_fragment) = dataset.get_fragment(*fragment_id as usize) else {
            continue;
        };
        let sequence = load_row_id_sequence(dataset, file_fragment.metadata()).await?;

        match selection {
            RowAddrSelection::Full => {
                // The whole fragment is selected: every live row's id qualifies.
                row_ids |= RowAddrTreeMap::from(sequence.as_ref());
            }
            RowAddrSelection::Partial(offsets) => {
                let deletion_vector = file_fragment.get_deletion_vector().await?;
                for physical_offset in offsets.iter() {
                    // A deletion does not compact the row id sequence; the deleted row keeps
                    // its slot (the deletion vector tracks it separately). So a physical offset
                    // is a direct index into the sequence, regardless of any deletions below it.
                    // A stale offset that points at a deleted row has no live counterpart and is
                    // not part of any answer, so it contributes no id to the block/take set.
                    let deleted = deletion_vector
                        .as_ref()
                        .is_some_and(|dv| dv.contains(physical_offset));
                    if deleted {
                        continue;
                    }
                    // A selected offset with no sequence entry points past the fragment's rows.
                    // Silently dropping it would let a stale index result escape masking, so
                    // treat the mismatch as corruption.
                    let id = sequence.get(physical_offset as usize).ok_or_else(|| {
                        Error::internal(format!(
                            "fragment_id={fragment_id} row-id sequence has no entry at \
                             physical_offset={physical_offset} (sequence len={})",
                            sequence.len()
                        ))
                    })?;
                    row_ids.insert(id);
                }
            }
        }
    }
    Ok(row_ids)
}

/// Translate stable row ids into their current physical row addresses, via
/// the dataset's [`RowIdIndex`] (an id -> address lookup, batched and
/// amortized O(log n) — cheap, unlike the reverse direction above, which
/// needs a per-fragment [`RowIdSequence`] scan). This is the opposite
/// direction from [`translate_addr_treemap_to_row_ids`], used to bring a
/// row-id-domain scalar-index result into row-address domain (see
/// `ScalarIndexLoader::row_id_result_to_row_addrs`).
///
/// An id with no live address (deleted, or superseded by compaction) is
/// silently dropped, mirroring the "no live counterpart -> not part of the
/// answer" contract of [`translate_addr_treemap_to_row_ids`]. `ids` must not
/// contain a `RowAddrSelection::Full` marker — that is an address-domain-only
/// shortcut ("this whole physical fragment matches") that no row-id-domain
/// index ever produces, since a row-id-domain index has no notion of
/// physical fragment layout to begin with.
pub(crate) fn translate_row_id_treemap_to_row_addrs(
    row_id_index: &RowIdIndex,
    ids: &RowAddrTreeMap,
) -> Result<RowAddrTreeMap> {
    let Some(id_iter) = ids.row_addrs() else {
        return Err(Error::internal(
            "a row-id-domain mask used a 'whole fragment' marker; no row-id-domain index \
             produces one",
        ));
    };
    let ids: Vec<u64> = id_iter.map(u64::from).collect();
    let addrs = row_id_index.get_many(&ids)?;
    Ok(RowAddrTreeMap::from_iter(
        addrs.into_iter().flatten().map(u64::from),
    ))
}

fn translate_row_id_set_to_row_addrs(
    row_id_index: &RowIdIndex,
    set: NullableRowAddrSet,
) -> Result<NullableRowAddrSet> {
    let (selected, nulls) = set.into_parts();
    Ok(NullableRowAddrSet::new(
        translate_row_id_treemap_to_row_addrs(row_id_index, &selected)?,
        translate_row_id_treemap_to_row_addrs(row_id_index, &nulls)?,
    ))
}

/// Translate a row-id-domain [`NullableRowAddrMask`] into the row-address
/// domain. Unlike the address -> row-id direction, both `AllowList` and
/// `BlockList` are supported here: either way, only the ids actually listed
/// in the mask need a lookup, never the full id universe (see
/// [`translate_row_id_treemap_to_row_addrs`]).
pub(crate) fn translate_row_id_mask_to_row_addrs(
    row_id_index: &RowIdIndex,
    mask: NullableRowAddrMask,
) -> Result<NullableRowAddrMask> {
    match mask {
        NullableRowAddrMask::AllowList(set) => Ok(NullableRowAddrMask::AllowList(
            translate_row_id_set_to_row_addrs(row_id_index, set)?,
        )),
        NullableRowAddrMask::BlockList(set) => Ok(NullableRowAddrMask::BlockList(
            translate_row_id_set_to_row_addrs(row_id_index, set)?,
        )),
    }
}

/// Plain-[`RowAddrMask`] (post-`drop_nulls`) sibling of
/// [`translate_row_id_mask_to_row_addrs`], for finalizing an
/// [`lance_select::IndexExprResult`] to row-address domain.
pub(crate) fn translate_row_id_plain_mask_to_row_addrs(
    row_id_index: &RowIdIndex,
    mask: RowAddrMask,
) -> Result<RowAddrMask> {
    match mask {
        RowAddrMask::AllowList(set) => Ok(RowAddrMask::AllowList(
            translate_row_id_treemap_to_row_addrs(row_id_index, &set)?,
        )),
        RowAddrMask::BlockList(set) => Ok(RowAddrMask::BlockList(
            translate_row_id_treemap_to_row_addrs(row_id_index, &set)?,
        )),
    }
}

/// Resolve row addresses to row ids, positionally: `addr = (fragment << 32) | offset`
/// looks up `offset` in the fragment's [`RowIdSequence`]. The inverse companion of
/// [`get_row_id_index`].
///
/// Returns one entry per input address, in order. Addresses that no longer
/// resolve — a missing fragment, an out-of-range offset, or a `None` input —
/// yield `None`. On datasets without stable row ids, addresses are the row
/// ids, so the input is returned unchanged.
pub async fn row_addrs_to_row_ids(
    dataset: &Dataset,
    addrs: impl IntoIterator<Item = Option<u64>>,
) -> Result<Vec<Option<u64>>> {
    let addrs: Vec<Option<u64>> = addrs.into_iter().collect();
    if !dataset.manifest.uses_stable_row_ids() {
        return Ok(addrs);
    }

    let mut positions_by_fragment: std::collections::HashMap<u32, Vec<usize>> =
        std::collections::HashMap::new();
    for (position, addr) in addrs.iter().enumerate() {
        if let Some(addr) = addr {
            positions_by_fragment
                .entry(RowAddress::from(*addr).fragment_id())
                .or_default()
                .push(position);
        }
    }

    let mut ids: Vec<Option<u64>> = vec![None; addrs.len()];
    for (fragment_id, positions) in positions_by_fragment {
        let Some(fragment) = dataset.get_fragment(fragment_id as usize) else {
            continue;
        };
        let sequence = load_row_id_sequence(dataset, fragment.metadata()).await?;
        for position in positions {
            let offset = RowAddress::from(addrs[position].expect("grouped from Some")).row_offset();
            ids[position] = sequence.get(offset as usize);
        }
    }
    Ok(ids)
}

/// Resolve row ids to their current row addresses, in input order.
///
/// A row address is `(fragment_id << 32) | offset`. With stable row ids, ids
/// that are unknown or deleted yield `None`. Without stable row ids, row ids
/// are row addresses, so each id is returned unchanged.
///
/// With stable row ids, this may read row-id metadata and deletion vectors from
/// the dataset and returns an error if either read fails.
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::dataset::rowids::row_ids_to_row_addrs;
/// # async fn test(dataset: &Dataset) -> Result<()> {
/// let addrs = row_ids_to_row_addrs(dataset, &[3, 21]).await?;
/// let fragment_ids: Vec<u64> = addrs.into_iter().flatten().map(|addr| addr >> 32).collect();
/// # Ok(())
/// # }
/// ```
pub async fn row_ids_to_row_addrs(dataset: &Dataset, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
    if row_ids.is_empty() {
        return Ok(Vec::new());
    }
    let Some(index) = get_row_id_index(dataset).await? else {
        return Ok(row_ids.iter().copied().map(Some).collect());
    };
    Ok(index
        .get_many(row_ids)?
        .into_iter()
        .map(|addr| addr.map(u64::from))
        .collect())
}

/// Builds a synchronous lookup, for the rows of `fragment_ids`, that a caller
/// can use one address at a time (e.g. inside a scoring loop) instead of as a
/// batch. This function itself is still async: it awaits loading every
/// fragment's row id sequence and deletion vector up front, so the returned
/// closure never has to.
///
/// The returned lookup yields `None` for an address in a fragment outside
/// `fragment_ids` or no longer in the dataset, past the end of its fragment,
/// or at a deleted physical slot. Requires stable row ids.
pub(crate) async fn live_row_addr_to_row_id_lookup(
    dataset: &Dataset,
    fragment_ids: impl IntoIterator<Item = u32>,
) -> Result<impl Fn(u64) -> Option<u64> + Send + Sync + 'static> {
    // Loaded concurrently (bounded by io_parallelism): on a dataset with many
    // fragments, loading each one's row id sequence and deletion vector in
    // turn serializes IO that has no reason to wait on the last one.
    let to_load: Vec<_> = fragment_ids
        .into_iter()
        .filter_map(|fragment_id| {
            dataset
                .get_fragment(fragment_id as usize)
                .map(|fragment| (fragment_id, fragment))
        })
        .collect();
    let loaded: Vec<_> = futures::stream::iter(to_load)
        .map(|(fragment_id, fragment)| async move {
            let sequence = load_row_id_sequence(dataset, fragment.metadata()).await?;
            let deletion_vector = fragment.get_deletion_vector().await?;
            Result::Ok((fragment_id, sequence, deletion_vector))
        })
        .buffer_unordered(dataset.table_state.object_store.io_parallelism())
        .try_collect()
        .await?;
    let fragments: std::collections::HashMap<_, _> = loaded
        .into_iter()
        .map(|(fragment_id, sequence, deletion_vector)| (fragment_id, (sequence, deletion_vector)))
        .collect();
    Ok(move |row_addr: u64| {
        let row_addr = RowAddress::from(row_addr);
        let (sequence, deletion_vector) = fragments.get(&row_addr.fragment_id())?;
        let offset = row_addr.row_offset();
        if deletion_vector
            .as_ref()
            .is_some_and(|deletions| deletions.contains(offset))
        {
            return None;
        }
        sequence.get(offset as usize)
    })
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    use crate::dataset::{
        ProjectionRequest, ReadParams, UpdateBuilder, WriteMode, WriteParams,
        builder::DatasetBuilder,
    };

    use super::*;

    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::index::DatasetIndexExt;
    use crate::index::visibility::IndexVisibility;
    use crate::session::Session;
    use crate::utils::test::{DatagenExt, FailingProxyStore, FragmentCount, FragmentRowCount};
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, Int32Type, UInt64Type};
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, UInt64Array};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::Future;
    use lance_core::datatypes::Schema;
    use lance_core::{ROW_ADDR, ROW_ID, utils::address::RowAddress};
    use lance_datagen::Dimension;
    use lance_index::{IndexType, scalar::ScalarIndexParams};
    use lance_io::object_store::ObjectStoreParams;
    use std::collections::HashMap;
    use std::collections::HashSet;

    fn sequence_batch(values: Range<i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from_iter_values(values))]).unwrap()
    }

    #[tokio::test]
    async fn test_empty_dataset_rowids() {
        let schema = sequence_batch(0..0).schema();
        let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        assert!(dataset.manifest.uses_stable_row_ids());

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).unwrap().is_none());

        assert_eq!(dataset.manifest().next_row_id, 0);
    }

    #[tokio::test]
    async fn test_must_set_on_creation() {
        let tmp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let tmp_path = &tmp_dir;

        let batch = sequence_batch(0..10);
        let reader =
            RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: false,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, tmp_path, Some(write_params))
            .await
            .unwrap();
        assert!(!dataset.manifest().uses_stable_row_ids());

        // Trying to append without stable row ids should pass (a warning is emitted) but should not
        // affect the stable_row_ids setting.
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            mode: WriteMode::Append,
            ..Default::default()
        };
        let reader =
            RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), batch.schema());
        let dataset = Dataset::write(reader, tmp_path, Some(write_params))
            .await
            .unwrap();
        assert!(!dataset.manifest().uses_stable_row_ids());
    }

    #[tokio::test]
    async fn test_new_row_ids() {
        let num_rows = 25u64;
        let batch = sequence_batch(0..num_rows as i32);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 10,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();

        let found_addresses = (0..num_rows)
            .map(|i| index.get(i).unwrap().unwrap())
            .collect::<Vec<_>>();
        let expected_addresses = (0..num_rows)
            .map(|i| {
                let fragment_id = i / 10;
                RowAddress::new_from_parts(fragment_id as u32, (i % 10) as u32)
            })
            .collect::<Vec<_>>();
        assert_eq!(found_addresses, expected_addresses);

        assert_eq!(dataset.manifest().next_row_id, num_rows);
    }

    #[tokio::test]
    async fn test_row_id_index_owns_the_sequences_it_caches() {
        let batch = sequence_batch(0..30);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 10,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();
        let session = dataset.session();

        // Building the index adds one cache entry: the index, which holds its
        // sequences itself rather than reading them through their own entries.
        let metadata_before = session.metadata_cache_stats().await;
        let index_before = session.index_cache_stats().await;
        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert_eq!(
            session.index_cache_stats().await.num_entries,
            index_before.num_entries + 1
        );
        assert!(
            session.index_cache_stats().await.size_bytes - index_before.size_bytes
                >= lance_core::deepsize::DeepSizeOf::deep_size_of(index.as_ref())
        );
        let metadata_after = session.metadata_cache_stats().await;
        assert_eq!(metadata_after.num_entries, metadata_before.num_entries);
        assert_eq!(metadata_after.size_bytes, metadata_before.size_bytes);
        let again = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(Arc::ptr_eq(&index, &again));

        // A sequence lookup keeps its own content-keyed entry.
        let fragment = &dataset.manifest.fragments[1];
        let sequence = load_row_id_sequence(&dataset, fragment).await.unwrap();
        assert_eq!(
            sequence.iter().collect::<Vec<_>>(),
            (10..20).collect::<Vec<_>>()
        );
        assert_eq!(
            session.metadata_cache_stats().await.num_entries,
            metadata_before.num_entries + 1
        );
    }

    #[rstest::rstest]
    #[case::metadata_disabled(4096, 0, true)]
    #[case::index_disabled(0, 1 << 20, false)]
    #[case::index_too_small(1, 1 << 20, false)]
    #[tokio::test]
    async fn test_row_id_index_cache_budget(
        #[case] index_capacity: usize,
        #[case] metadata_capacity: usize,
        #[case] is_cached: bool,
    ) {
        let session = Arc::new(Session::new(
            index_capacity,
            metadata_capacity,
            Default::default(),
        ));
        let batch = sequence_batch(0..30);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 10,
                session: Some(session.clone()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let metadata_before = session.metadata_cache_stats().await;
        let entries_before = session.index_cache_stats().await.num_entries;
        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert_eq!(
            index.get(21).unwrap(),
            Some(RowAddress::new_from_parts(2, 1))
        );
        let weak = Arc::downgrade(&index);
        drop(index);
        assert_eq!(weak.upgrade().is_some(), is_cached);
        assert_eq!(
            session.index_cache_stats().await.num_entries,
            entries_before + usize::from(is_cached)
        );
        assert_eq!(
            session.metadata_cache_stats().await.size_bytes,
            metadata_before.size_bytes
        );

        let reloaded = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert_eq!(
            reloaded.get(21).unwrap(),
            Some(RowAddress::new_from_parts(2, 1))
        );
        assert!(reloaded.get(30).unwrap().is_none());
    }

    #[tokio::test]
    async fn test_row_id_index_eviction_and_reload() {
        let session = Arc::new(Session::new(4096, 4096, Default::default()));
        let batch = sequence_batch(0..30);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 10,
                session: Some(session.clone()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let metadata_before = session.metadata_cache_stats().await;
        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        let weak = Arc::downgrade(&index);
        drop(index);
        assert!(weak.upgrade().is_some());

        // QuickCache resists scans of small cold entries. Compete with other
        // loaded indices of the same size, without touching this one's recency.
        let table = dataset.table_view().to_table();
        let no_cache = lance_core::cache::LanceCache::no_cache();
        for version in 100..132 {
            let competing_index =
                lance_table_runtime::rowids::get_row_id_index_with_cache(&table, true, &no_cache)
                    .await
                    .unwrap()
                    .unwrap();
            let key = RowIdIndexKey {
                version,
                with_deletions: true,
                e_tag: Some("pressure"),
            };
            dataset
                .index_cache
                .insert_with_key(&key, competing_index)
                .await;
            assert!(dataset.index_cache.get_with_key(&key).await.is_some());
        }
        assert!(weak.upgrade().is_none());
        let reloaded = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert_eq!(
            reloaded.get(21).unwrap(),
            Some(RowAddress::new_from_parts(2, 1))
        );
        assert!(reloaded.get(30).unwrap().is_none());
        assert!(session.index_cache_stats().await.size_bytes <= 4096);
        assert_eq!(
            session.metadata_cache_stats().await.size_bytes,
            metadata_before.size_bytes
        );
    }

    #[tokio::test]
    async fn test_row_id_index_is_not_shared_without_a_generation_token() {
        let batch = sequence_batch(0..10);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();
        // Without an e-tag the version-only key could alias a dataset recreated
        // at the same URI, so the index must be built privately, never cached.
        dataset.table_state.manifest_location.e_tag = None;
        let session = dataset.session();
        let entries_before = session.index_cache_stats().await.num_entries;
        let first = get_row_id_index(&dataset).await.unwrap().unwrap();
        let second = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(
            session.index_cache_stats().await.num_entries,
            entries_before
        );
        assert_eq!(
            second.get(9).unwrap(),
            Some(RowAddress::new_from_parts(0, 9))
        );
    }

    #[tokio::test]
    async fn test_visibility_retries_spilled_row_id_read() {
        let batch = sequence_batch(0..8);
        let directory = lance_core::utils::tempfile::TempStrDir::default();
        let test_uri = directory.as_str();
        let path_prefix = if test_uri.starts_with('/') { "" } else { "/" };
        let uri = format!("file-object-store://{path_prefix}{test_uri}");
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            &uri,
            Some(WriteParams {
                max_rows_per_file: 4,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .update_config([
                ("lance.row_lineage.spill", "true"),
                ("lance.row_lineage.inline_max_bytes", "0"),
            ])
            .await
            .unwrap();
        let dataset = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id >= 4")
            .unwrap()
            .set("id", "id + 100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;
        let mut dataset = dataset.as_ref().clone();
        let fragment_id = dataset.fragments().last().unwrap().id as u32;
        assert!(matches!(
            dataset.fragments().last().unwrap().row_id_meta,
            Some(RowIdMeta::Column)
        ));
        dataset.delete("id = 104 OR id = 106").await.unwrap();
        drop(dataset);
        let failing = Arc::new(FailingProxyStore::new());
        let dataset = DatasetBuilder::from_uri(&uri)
            .with_read_params(ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(failing.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();
        let store = dataset.object_store(None).await.unwrap();
        let visibility = IndexVisibility::from_physical_rows(
            Arc::new(dataset),
            HashMap::from([(fragment_id, [1, 2, 3].into_iter().collect())]),
        );
        failing.fail_when("get_opts", "data/", "injected row-id sequence failure");
        let error = visibility.result_rows().await.unwrap_err();
        assert!(matches!(error, Error::IO { .. }));
        assert!(
            error
                .to_string()
                .contains("injected row-id sequence failure")
        );
        failing.clear_fail_when("get_opts", "data/");
        store.io_stats_incremental();
        let rows = visibility.result_rows().await.unwrap();
        assert_eq!(rows, &RowAddrTreeMap::from_iter([5, 7]));
        assert!(store.io_stats_incremental().read_iops > 0);
        visibility.block_mask().await.unwrap();
        assert!(std::ptr::eq(rows, visibility.result_rows().await.unwrap()));
        assert_eq!(store.io_stats_incremental().read_iops, 0);
    }

    #[tokio::test]
    async fn test_row_id_index_propagates_deletion_vector_read_error() {
        let batch = sequence_batch(0..10);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let test_uri = temp_dir.as_str();
        let path_prefix = if test_uri.starts_with('/') { "" } else { "/" };
        let routed_uri = format!("file-object-store://{path_prefix}{test_uri}");
        let mut dataset = Dataset::write(
            reader,
            &routed_uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.delete("id = 3").await.unwrap();
        drop(dataset);

        let failing_store = Arc::new(FailingProxyStore::new());
        failing_store.fail_when(
            "get_opts",
            "_deletions",
            "injected deletion-vector read failure",
        );
        let dataset = DatasetBuilder::from_uri(&routed_uri)
            .with_read_params(ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(failing_store.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();

        let error = get_row_id_index(&dataset).await.unwrap_err();
        assert!(matches!(&error, Error::IO { .. }));
        assert!(
            error
                .to_string()
                .contains("injected deletion-vector read failure")
        );

        let visibility = IndexVisibility::from_physical_rows(
            Arc::new(dataset.clone()),
            HashMap::from([(0, [4].into_iter().collect())]),
        );
        let error = visibility.block_mask().await.unwrap_err();
        assert!(matches!(&error, Error::IO { .. }));
        assert!(
            error
                .to_string()
                .contains("injected deletion-vector read failure")
        );

        failing_store.clear_fail_when("get_opts", "_deletions");
        assert_eq!(
            visibility.result_rows().await.unwrap(),
            &RowAddrTreeMap::from_iter([4])
        );
        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(2).unwrap().is_some());
        assert!(index.get(3).unwrap().is_none());
    }

    #[tokio::test]
    async fn test_row_addrs_to_row_ids() {
        let num_rows = 25u64;
        let batch = sequence_batch(0..num_rows as i32);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 10,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        // Sequential assignment: row n lives at (n / 10, n % 10) with id n
        let addr = |frag: u64, offset: u64| Some((frag << 32) | offset);
        let addrs = vec![
            addr(2, 1),   // id 21
            addr(0, 3),   // id 3
            None,         // null stays null
            addr(0, 3),   // duplicates allowed
            addr(9, 0),   // missing fragment
            addr(1, 100), // out-of-range offset
        ];
        let ids = row_addrs_to_row_ids(&dataset, addrs).await.unwrap();
        assert_eq!(ids, vec![Some(21), Some(3), None, Some(3), None, None]);

        // Without stable row ids, addresses are the ids: input passes through
        let batch = sequence_batch(0..num_rows as i32);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let plain = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let addrs = vec![addr(1, 2), None];
        let ids = row_addrs_to_row_ids(&plain, addrs.clone()).await.unwrap();
        assert_eq!(ids, addrs);
    }

    #[tokio::test]
    async fn test_row_ids_to_row_addrs() {
        let batch = sequence_batch(0..25);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 10,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();
        // Rewriting row 3 gives it a new address in a new fragment but keeps its id.
        UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id = 3")
            .unwrap()
            .set("id", "100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset.checkout_latest().await.unwrap();
        dataset.delete("id = 21").await.unwrap();

        let addr = |frag: u64, offset: u64| Some((frag << 32) | offset);
        let addrs = row_ids_to_row_addrs(&dataset, &[12, 3, 21, 999, 12])
            .await
            .unwrap();
        assert_eq!(addrs, vec![addr(1, 2), addr(3, 0), None, None, addr(1, 2)]);
        assert!(
            row_ids_to_row_addrs(&dataset, &[])
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            row_addrs_to_row_ids(&dataset, addrs).await.unwrap(),
            vec![Some(12), Some(3), None, None, Some(12)]
        );

        // Without stable row ids, ids are the addresses: input passes through
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let plain = Dataset::write(reader, "memory://", None).await.unwrap();
        let ids = [(1u64 << 32) | 2, 999];
        assert_eq!(
            row_ids_to_row_addrs(&plain, &ids).await.unwrap(),
            vec![Some(ids[0]), Some(ids[1])]
        );
    }

    #[tokio::test]
    async fn test_row_ids_overwrite() {
        // Validate we don't re-use after overwriting
        let num_rows = 10u64;
        let batch = sequence_batch(0..num_rows as i32);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let tmp_path = &temp_dir;
        let dataset = Dataset::write(reader, tmp_path, Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.manifest().next_row_id, num_rows);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, tmp_path, Some(write_params))
            .await
            .unwrap();

        // Overwriting should NOT reset the row id counter.
        assert_eq!(dataset.manifest().next_row_id, 2 * num_rows);
        // Nor the fragment id counter: ids are a high water mark, so the
        // overwritten fragment cannot alias the one it replaced.
        assert_eq!(dataset.manifest.fragments[0].id, 1);

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).unwrap().is_none());
        assert!(index.get(num_rows).unwrap().is_some());
    }

    /// Fragment ids are a high water mark within one dataset, but a dataset
    /// dropped and recreated at the same URI restarts them at 0 while sharing the
    /// cache namespace of its predecessor (see #7645). The two generations must
    /// still be told apart.
    #[tokio::test]
    async fn test_row_ids_recreate_at_same_uri() {
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let tmp_path = &temp_dir;
        // Shared so the cache stays warm across the drop, as it would for a host
        // that keeps one session open for the lifetime of the process.
        let session = Arc::new(Session::default());
        let write = |rows: Range<i32>| {
            let batch = sequence_batch(rows);
            let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
            let params = WriteParams {
                enable_stable_row_ids: true,
                session: Some(session.clone()),
                ..Default::default()
            };
            async move {
                Dataset::write(reader, tmp_path, Some(params))
                    .await
                    .unwrap()
            }
        };

        let dataset = write(0..100).await;
        let sequence = load_row_id_sequence(&dataset, &dataset.manifest.fragments[0])
            .await
            .unwrap();
        assert_eq!(sequence.len(), 100);
        // Keep this generation's index warm to check that a recreated dataset
        // at the same URI and version cannot reuse it.
        get_row_id_index(&dataset).await.unwrap().unwrap();

        // Reloading the unchanged fragment must still hit: keying on contents has
        // to leave the sequence cacheable, not just make it distinguishable.
        let hits_before = session.metadata_cache_stats().await.hits;
        load_row_id_sequence(&dataset, &dataset.manifest.fragments[0])
            .await
            .unwrap();
        assert_eq!(session.metadata_cache_stats().await.hits, hits_before + 1);

        drop(dataset);
        std::fs::remove_dir_all(tmp_path.as_str()).unwrap();

        // Shorter than the dataset it replaces, so a stale hit is observable: an
        // equal-length sequence would be byte-identical and harmless.
        let dataset = write(0..60).await;
        assert_eq!(dataset.manifest.fragments[0].id, 0);

        let sequence = load_row_id_sequence(&dataset, &dataset.manifest.fragments[0])
            .await
            .unwrap();
        assert_eq!(
            sequence.iter().collect::<Vec<_>>(),
            (0..60).collect::<Vec<_>>()
        );

        // The same goes for the index: row id 60 exists only in the dropped
        // generation, whose index is still in the session cache.
        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(60).unwrap().is_none());
        assert_eq!(
            index.get(59).unwrap(),
            Some(RowAddress::new_from_parts(0, 59))
        );
    }

    #[tokio::test]
    async fn test_compaction_after_recreate() {
        // Compaction rechunks the row id sequences of the fragments it merges, so
        // a sequence cached for a dropped dataset does not just misreport ids, it
        // writes ids the fragments do not hold back into the manifest.
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let tmp_path = &temp_dir;
        let session = Arc::new(Session::default());
        let params = WriteParams {
            enable_stable_row_ids: true,
            session: Some(session.clone()),
            ..Default::default()
        };

        let write = |rows: Range<i32>, mode: WriteMode| {
            let batch = sequence_batch(rows);
            let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
            let params = WriteParams {
                mode,
                ..params.clone()
            };
            async move {
                Dataset::write(reader, tmp_path, Some(params))
                    .await
                    .unwrap()
            }
        };

        let dataset = write(0..100, WriteMode::Create).await;
        load_row_id_sequence(&dataset, &dataset.manifest.fragments[0])
            .await
            .unwrap();

        drop(dataset);
        std::fs::remove_dir_all(tmp_path.as_str()).unwrap();

        // The recreated dataset holds 60 rows in fragment 0, not the 100 cached
        // above, and a second fragment so compaction has something to merge.
        write(0..60, WriteMode::Create).await;
        let mut dataset = write(0..100, WriteMode::Append).await;

        compact(&mut dataset, 1024).await;

        assert_eq!(dataset.count_rows(None).await.unwrap(), 160);
        let mut scan = dataset.scan();
        scan.with_row_id();
        let batch = scan.try_into_batch().await.unwrap();
        let row_ids = batch[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        // A stale 100-long sequence would rechunk 0..100 onto the 60 rows of
        // fragment 0 and shift everything after it.
        assert_eq!(row_ids, (0..160).collect::<HashSet<_>>());
    }

    #[tokio::test]
    async fn test_row_ids_append() {
        // Validate we handle row ids well when appending concurrently.
        fn write_batch(uri: &str, start: i32) -> impl Future<Output = Result<()>> + '_ {
            let batch = sequence_batch(start..(start + 10));
            let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
            let write_params = WriteParams {
                enable_stable_row_ids: true,
                mode: WriteMode::Append,
                ..Default::default()
            };
            async move {
                let _ = Dataset::write(reader, uri, Some(write_params)).await?;
                Ok(())
            }
        }

        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let tmp_path = &temp_dir;
        let mut start = 0;
        // Just do one first to create the dataset.
        write_batch(tmp_path, start).await.unwrap();
        start += 10;
        // Now do the rest concurrently.
        let futures = (0..5)
            .map(|offset| write_batch(tmp_path, start + offset * 10))
            .collect::<Vec<_>>();
        futures::future::try_join_all(futures).await.unwrap();

        let dataset = DatasetBuilder::from_uri(tmp_path).load().await.unwrap();

        assert_eq!(dataset.manifest().next_row_id, 60);

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).unwrap().is_some());
        assert!(index.get(60).unwrap().is_none());
    }

    #[tokio::test]
    async fn test_scan_row_ids() {
        // Write dataset with multiple files -> _rowid != _rowaddr
        // Scan with and without each.;
        let batch = sequence_batch(0..6);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 2,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);

        for with_row_id in [true, false] {
            for with_row_address in &[true, false] {
                for projection in &[vec![], vec!["id"]] {
                    if !with_row_id && !with_row_address && projection.is_empty() {
                        continue;
                    }

                    let mut scan = dataset.scan();
                    if with_row_id {
                        scan.with_row_id();
                    }
                    if *with_row_address {
                        scan.with_row_address();
                    }
                    let scan = scan.project(projection).unwrap();
                    let result = scan.try_into_batch().await.unwrap();

                    if with_row_id {
                        let row_ids = result[ROW_ID]
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .unwrap();
                        let expected = vec![0, 1, 2, 3, 4, 5].into();
                        assert_eq!(row_ids, &expected);
                    }

                    if *with_row_address {
                        let row_addrs = result[ROW_ADDR]
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .unwrap();
                        let expected =
                            vec![0, 1, 1 << 32, (1 << 32) + 1, 2 << 32, (2 << 32) + 1].into();
                        assert_eq!(row_addrs, &expected);
                    }

                    if !projection.is_empty() {
                        let ids = result["id"].as_any().downcast_ref::<Int32Array>().unwrap();
                        let expected = vec![0, 1, 2, 3, 4, 5].into();
                        assert_eq!(ids, &expected);
                    }
                }
            }
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn test_delete_with_row_ids(#[values(true, false)] with_scalar_index: bool) {
        let batch = sequence_batch(0..6);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 2,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);

        if with_scalar_index {
            dataset
                .create_index(
                    &["id"],
                    IndexType::Scalar,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
        }

        dataset.delete("id = 3 or id = 4").await.unwrap();

        let mut scan = dataset.scan();
        scan.with_row_id().with_row_address();
        let result = scan.try_into_batch().await.unwrap();

        let row_ids = result[ROW_ID]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let expected = vec![0, 1, 2, 5].into();
        assert_eq!(row_ids, &expected);

        let row_addrs = result[ROW_ADDR]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let expected = vec![0, 1, 1 << 32, (2 << 32) + 1].into();
        assert_eq!(row_addrs, &expected);
    }

    #[tokio::test]
    async fn test_row_ids_update() {
        // With stable row ids an update remaps the row's id onto the rewritten
        // row's address; it does not mint a new id.
        let num_rows = 5u64;
        let batch = sequence_batch(0..num_rows as i32);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.manifest().next_row_id, num_rows);

        let update_result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 3")
            .unwrap()
            .set("id", "100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let dataset = update_result.new_dataset;
        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).unwrap().is_some());
        // the updated row ids mapping to new address
        assert_eq!(
            index.get(3).unwrap(),
            Some(RowAddress::new_from_parts(1, 0))
        );
        // there is no new row id
        assert_eq!(index.get(5).unwrap(), None);
    }

    /// 100 sequential rows across 4 fragments with every third row deleted.
    async fn deleted_thirds_dataset(enable_stable_row_ids: bool) -> Dataset {
        let batch = sequence_batch(0..100);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids,
            max_rows_per_file: 25,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();
        dataset.delete("id % 3 = 0").await.unwrap();
        dataset
    }

    #[rstest::rstest]
    #[case::interleaved((0..100).collect(), (0..100).filter(|id| id % 3 != 0).collect())]
    #[case::all_deleted(vec![0, 3, 9], vec![])]
    #[case::all_live(vec![1, 2, 50, 98], vec![1, 2, 50, 98])]
    #[tokio::test]
    async fn test_filter_deleted_ids_with_stable_row_ids(
        #[case] ids: Vec<u64>,
        #[case] expected: Vec<u64>,
    ) {
        // Regression test for https://github.com/lance-format/lance/issues/7701:
        // filter_deleted_ids must return exactly the live ids, in order.
        // Sequential inserts, so stable row id == id column value.
        let dataset = deleted_thirds_dataset(true).await;
        assert_eq!(dataset.filter_deleted_ids(&ids).await.unwrap(), expected);
    }

    #[tokio::test]
    async fn test_filter_deleted_ids_without_stable_row_ids() {
        // Non-stable: ids are row addresses; only deleted rows drop out.
        let dataset = deleted_thirds_dataset(false).await;

        // Row addresses: fragment (id / 25) << 32 | offset (id % 25).
        let addr = |id: u64| (id / 25) << 32 | (id % 25);
        let ids = (0..100).map(addr).collect::<Vec<u64>>();
        let expected = (0..100)
            .filter(|id| id % 3 != 0)
            .map(addr)
            .collect::<Vec<u64>>();
        assert_eq!(dataset.filter_deleted_ids(&ids).await.unwrap(), expected);
    }

    fn build_rowid_to_i_map(row_ids: &UInt64Array, i_array: &Int32Array) -> HashMap<u64, i32> {
        row_ids
            .values()
            .iter()
            .zip(i_array.values().iter())
            .map(|(&row_id, &i)| (row_id, i))
            .collect()
    }

    async fn scan_rowid_map(dataset: &Dataset) -> HashMap<u64, i32> {
        let mut scan = dataset.scan();
        scan.with_row_id();
        scan.scan_in_order(true);
        let result = scan.try_into_batch().await.unwrap();
        let i = result["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let row_ids = result[ROW_ID]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        build_rowid_to_i_map(row_ids, i)
    }

    pub(super) async fn compact(dataset: &mut Dataset, target_rows: usize) {
        let options = CompactionOptions {
            target_rows_per_fragment: target_rows,
            ..Default::default()
        };
        let _ = compact_files(dataset, options, None).await.unwrap();
    }

    pub(super) async fn delete(dataset: &mut Dataset, expr: &str) {
        dataset.delete(expr).await.unwrap();
    }

    #[tokio::test]
    async fn test_range_segments_are_opt_in() {
        use lance_table::format::pb;
        use lance_table::transaction::RANGE_SEGMENTS_CONFIG_KEY;
        use prost::Message;

        /// Wire-level segment kinds of every fragment, and whether the decoded
        /// form folded any of them into a compact `Ranges` segment.
        fn segment_shapes(dataset: &Dataset) -> (Vec<&'static str>, bool, usize) {
            let mut kinds = Vec::new();
            let mut folded = false;
            let mut bytes = 0;
            for fragment in dataset.manifest.fragments.iter() {
                let Some(RowIdMeta::Inline(data)) = &fragment.row_id_meta else {
                    panic!("expected inline row ids, got {:?}", fragment.row_id_meta);
                };
                bytes += data.len();
                let wire = pb::RowIdSequence::decode(&data[..]).unwrap();
                kinds.extend(wire.segments.iter().map(|segment| match segment.segment {
                    Some(pb::u64_segment::Segment::Range(_)) => "range",
                    Some(pb::u64_segment::Segment::RangeWithBitmap(_)) => "bitmap",
                    _ => "other",
                }));
                folded |= read_row_ids(&data[..])
                    .unwrap()
                    .segments()
                    .iter()
                    .any(|segment| matches!(segment, U64Segment::Ranges { .. }));
            }
            (kinds, folded, bytes)
        }

        // Deleting a contiguous block inside every fragment leaves each row id
        // sequence as a range with one long run of holes, which `delete`
        // encodes as a bitmap segment; compaction carries those segments over.
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset_with_params(
                FragmentCount::from(4),
                FragmentRowCount::from(500),
                Some(WriteParams {
                    max_rows_per_file: 500,
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        delete(&mut dataset, "i % 500 >= 100 and i % 500 < 400").await;
        compact(&mut dataset, 5000).await;
        let map_before = scan_rowid_map(&dataset).await;
        assert_eq!(map_before.len(), 800);

        // Without the opt-in the bitmaps stay.
        let (kinds, folded, bytes_before) = segment_shapes(&dataset);
        assert!(kinds.contains(&"bitmap") && !folded, "{kinds:?}");

        // Opting in re-encodes every eligible fragment in the same commit: on
        // the wire only plain `Range` segments, folded back when decoded.
        dataset
            .update_config([(RANGE_SEGMENTS_CONFIG_KEY, "true")])
            .await
            .unwrap();
        let (kinds, folded, bytes_after) = segment_shapes(&dataset);
        assert!(kinds.iter().all(|kind| *kind == "range"), "{kinds:?}");
        assert!(
            bytes_after < bytes_before,
            "{bytes_after} >= {bytes_before}"
        );
        assert!(
            folded,
            "fragments with 300 contiguous holes each should fold into Ranges"
        );
        assert_eq!(
            dataset.manifest().reader_feature_flags,
            lance_table::feature_flags::FLAG_STABLE_ROW_IDS,
            "the wire format did not change, so no new reader flag"
        );

        // Reads see the same row ids; a take by row id resolves through the
        // compact segments.
        let map_after = scan_rowid_map(&dataset).await;
        assert_eq!(map_before, map_after);
        let mut sample: Vec<u64> = map_before.keys().copied().collect();
        sample.sort_unstable();
        let sample: Vec<u64> = sample.into_iter().step_by(97).collect();
        let taken = dataset
            .take_rows(
                &sample,
                ProjectionRequest::from_columns(["i"], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(taken.num_rows(), sample.len());
        let index = get_row_id_index(&Arc::new(dataset.clone()))
            .await
            .unwrap()
            .unwrap();
        for row_id in &sample {
            assert!(index.get(*row_id).unwrap().is_some(), "row id {row_id}");
        }

        // Later commits keep re-encoding only what changed, and the manifest
        // re-read from storage folds the same way.
        delete(&mut dataset, "i = 10").await;
        let reopened = dataset
            .checkout_version(dataset.manifest().version)
            .await
            .unwrap();
        let (kinds, folded, _) = segment_shapes(&reopened);
        assert!(
            kinds.iter().all(|kind| *kind == "range") && folded,
            "{kinds:?}"
        );
        assert_eq!(scan_rowid_map(&reopened).await.len(), 799);
    }

    #[tokio::test]
    async fn test_stable_row_id_after_multiple_deletion_and_compaction() {
        async fn delete(dataset: &mut Dataset, expr: &str) {
            dataset.delete(expr).await.unwrap();
        }

        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col(
                "category",
                lance_datagen::array::cycle::<Int32Type>(vec![1, 2, 3]),
            )
            .into_ram_dataset_with_params(
                FragmentCount::from(6),
                FragmentRowCount::from(10),
                Some(WriteParams {
                    max_rows_per_file: 10,
                    enable_stable_row_ids: true,
                    enable_v2_manifest_paths: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        // first delete and compact
        delete(&mut dataset, "i = 2 or i = 3 or i = 5").await;
        let map_before = scan_rowid_map(&dataset).await;
        compact(&mut dataset, 20).await;
        let map_after = scan_rowid_map(&dataset).await;

        // verify row id
        assert_eq!(
            map_before.keys().collect::<HashSet<_>>(),
            map_after.keys().collect::<HashSet<_>>()
        );
        for row_id in map_before.keys() {
            assert_eq!(map_before[row_id], map_after[row_id]);
        }

        // second delete
        delete(&mut dataset, "i = 9").await;
        let mut scan = dataset.scan();
        let result = scan
            .filter("i >= 0")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids = result["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let id_set = ids.values().iter().cloned().collect::<HashSet<_>>();
        let expected: Vec<i32> = (0..60)
            .filter(|&i| i != 2 && i != 3 && i != 5 && i != 9)
            .collect();
        assert_eq!(id_set, expected.iter().cloned().collect::<HashSet<_>>());

        // get the row_id where i == 15
        let mut scan = dataset.scan();
        scan.with_row_id();
        scan.scan_in_order(true);
        let result = scan
            .filter("i == 15")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let row_id_vec = result[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();

        // third delete and compact
        delete(&mut dataset, "i = 15 or i = 25").await;
        let map_before = scan_rowid_map(&dataset).await;
        compact(&mut dataset, 30).await;
        let map_after = scan_rowid_map(&dataset).await;

        assert_eq!(
            map_before.keys().collect::<HashSet<_>>(),
            map_after.keys().collect::<HashSet<_>>()
        );
        for row_id in map_before.keys() {
            assert_eq!(map_before[row_id], map_after[row_id]);
        }

        // verify the rowid represent i == 15 has been deleted
        let result = dataset
            .take_rows(&row_id_vec, Schema::try_from(dataset.schema()).unwrap())
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[tokio::test]
    async fn test_stable_row_id_after_deletion_update_and_compaction() {
        // gen dataset
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "i",
                lance_datagen::array::step::<arrow_array::types::Int32Type>(),
            )
            .col(
                "category",
                lance_datagen::array::cycle::<Int32Type>(vec![1, 2, 3]),
            )
            .into_ram_dataset_with_params(
                FragmentCount::from(6),
                FragmentRowCount::from(10),
                Some(WriteParams {
                    max_rows_per_file: 10,
                    enable_stable_row_ids: true,
                    enable_v2_manifest_paths: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        // delete some rows
        delete(&mut dataset, "i = 2 or i = 3 or i = 5").await;
        let map_before = scan_rowid_map(&dataset).await;

        // update some rows
        let updated_dataset = UpdateBuilder::new(Arc::new(dataset))
            .update_where("i >= 15")
            .unwrap()
            .set("category", "999")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;

        // compact the dataset
        let mut dataset = Arc::try_unwrap(updated_dataset).expect("no other Arc references");
        compact(&mut dataset, 20).await;
        let map_after = scan_rowid_map(&dataset).await;

        // verify row id
        assert_eq!(
            map_before.keys().collect::<HashSet<_>>(),
            map_after.keys().collect::<HashSet<_>>()
        );
        for row_id in map_before.keys() {
            assert_eq!(map_before[row_id], map_after[row_id]);
        }

        // verify category filed
        let mut scan = dataset.scan();
        scan.with_row_id();
        scan.scan_in_order(true);
        let result = scan.try_into_batch().await.unwrap();
        let i = result["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let category = result["category"]
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for idx in 0..i.len() {
            if i.value(idx) >= 15 {
                assert_eq!(category.value(idx), 999);
            }
        }
    }
}
