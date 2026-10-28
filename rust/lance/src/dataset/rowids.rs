// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod spill;
mod validate;

use super::Dataset;
use crate::io::deletion::read_dataset_deletion_file;
use crate::session::caches::{RowIdIndexKey, RowIdSequenceKey, RowVersionSequenceKey};
use crate::{Error, Result};
use arrow_array::UInt64Array;
use futures::{Stream, StreamExt, TryFutureExt, TryStreamExt};
use lance_core::utils::{address::RowAddress, deletion::DeletionVector};
use lance_select::{RowAddrSelection, RowAddrTreeMap};
use lance_table::{
    format::{
        Fragment, ROW_CREATED_AT_VERSION_FIELD_ID, ROW_ID_FIELD_ID,
        ROW_LAST_UPDATED_AT_VERSION_FIELD_ID, RowDatasetVersionMeta, RowDatasetVersionSequence,
        RowIdMeta,
    },
    rowids::{
        FragmentRowIdIndex, RowIdIndex, RowIdSequence, read_row_ids,
        version::{LoadedRowLineage, SpilledRowLineage},
    },
};
use std::sync::Arc;

pub(crate) use spill::place_carried_row_lineage;
pub use spill::{
    DEFAULT_INLINE_ROW_LINEAGE_MAX_BYTES, INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY,
    PlacedRowLineage, RowLineage, SPILL_ROW_LINEAGE_CONFIG_KEY, inline_row_lineage_max_bytes,
    place_row_lineage, read_spilled_row_ids, read_spilled_versions,
};
pub(crate) use spill::{RowLineagePlan, RowLineageSpill, plan_row_lineage_spill};
pub(super) use validate::validate_stable_row_ids;

/// Load a row id sequence from the given dataset and fragment.
pub async fn load_row_id_sequence(
    dataset: &Dataset,
    fragment: &Fragment,
) -> Result<Arc<RowIdSequence>> {
    let Some(row_id_meta) = &fragment.row_id_meta else {
        return Err(Error::internal("Missing row id meta"));
    };
    let key = RowIdSequenceKey {
        fragment_id: fragment.id,
        row_id_meta,
        lineage_file: fragment.row_lineage_file(ROW_ID_FIELD_ID)?,
    };
    dataset
        .metadata_cache
        .get_or_insert_with_key(key, || read_row_id_sequence(dataset, fragment))
        .await
}

/// Decode the row id sequence of `fragment`, bypassing every cache.
async fn read_row_id_sequence(dataset: &Dataset, fragment: &Fragment) -> Result<RowIdSequence> {
    match &fragment.row_id_meta {
        None => Err(Error::internal("Missing row id meta")),
        Some(RowIdMeta::Inline(data)) => read_row_ids(data),
        Some(RowIdMeta::Column) => spill::read_spilled_row_ids(dataset, fragment).await,
    }
}

/// Which of a fragment's two per-row version sequences is meant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowVersionKind {
    /// The dataset version each row first appeared at.
    CreatedAt,
    /// The dataset version each row was last written at.
    LastUpdatedAt,
}

impl RowVersionKind {
    fn meta(self, fragment: &Fragment) -> Option<&RowDatasetVersionMeta> {
        match self {
            Self::CreatedAt => fragment.created_at_version_meta.as_ref(),
            Self::LastUpdatedAt => fragment.last_updated_at_version_meta.as_ref(),
        }
    }

    /// The reserved field id of the hidden column a spilled sequence lives in.
    pub fn field_id(self) -> i32 {
        match self {
            Self::CreatedAt => ROW_CREATED_AT_VERSION_FIELD_ID,
            Self::LastUpdatedAt => ROW_LAST_UPDATED_AT_VERSION_FIELD_ID,
        }
    }
}

/// Load one of `fragment`'s per-row version sequences, wherever it is stored.
///
/// `None` when the fragment carries no such metadata, which readers treat as
/// every row being at version 1. A sequence spilled to a data file is cached
/// per fragment and file; an inline one is decoded from the manifest bytes.
pub async fn load_row_version_sequence(
    dataset: &Dataset,
    fragment: &Fragment,
    kind: RowVersionKind,
) -> Result<Option<Arc<RowDatasetVersionSequence>>> {
    let Some(meta) = kind.meta(fragment) else {
        return Ok(None);
    };
    match meta {
        RowDatasetVersionMeta::Column => {
            let data_file = fragment.row_lineage_file(kind.field_id())?.ok_or_else(|| {
                Error::corrupt_file(
                    dataset.base.clone(),
                    format!(
                        "fragment {} marks its {kind:?} versions as spilled but none of its \
                         data files carries field {}",
                        fragment.id,
                        kind.field_id()
                    ),
                )
            })?;
            let key = RowVersionSequenceKey {
                fragment_id: fragment.id,
                field_id: kind.field_id(),
                data_file,
            };
            dataset
                .metadata_cache
                .get_or_insert_with_key(key, || {
                    spill::read_spilled_versions(dataset, fragment, kind.field_id())
                })
                .await
                .map(Some)
        }
        RowDatasetVersionMeta::Inline(_) => meta
            .load_sequence()
            .map(|sequence| Some(Arc::new(sequence))),
    }
}

/// Read ahead every lineage sequence of `fragments` that lives outside the
/// manifest, for a commit that will need to consult them.
///
/// Building a manifest is synchronous and cannot read a data file, so the
/// commit path calls this first, over the whole manifest, and hands the result
/// over in `ManifestBuildConfig::spilled_row_lineage`. Only spilled sequences
/// are loaded; when none is, this returns an empty map without IO.
pub async fn load_spilled_row_lineage<'a>(
    dataset: &Dataset,
    fragments: impl IntoIterator<Item = &'a Fragment>,
) -> Result<Arc<SpilledRowLineage>> {
    // A `for` loop rather than `map`: a closure returning a future that borrows
    // its argument trips the higher-ranked lifetime check on the outer future.
    let mut loads = Vec::new();
    for fragment in fragments {
        if fragment.has_spilled_row_lineage() {
            loads.push(load_fragment_spilled_lineage(dataset, fragment));
        }
    }
    let loaded: SpilledRowLineage = futures::stream::iter(loads)
        .buffer_unordered(dataset.object_store.io_parallelism())
        .try_collect()
        .await?;
    Ok(Arc::new(loaded))
}

/// The spilled sequences of one fragment, for [`load_spilled_row_lineage`].
async fn load_fragment_spilled_lineage(
    dataset: &Dataset,
    fragment: &Fragment,
) -> Result<(u64, LoadedRowLineage)> {
    let row_ids = match &fragment.row_id_meta {
        Some(RowIdMeta::Column) => Some(load_row_id_sequence(dataset, fragment).await?),
        _ => None,
    };
    let mut versions = [None, None];
    for (slot, kind) in versions
        .iter_mut()
        .zip([RowVersionKind::CreatedAt, RowVersionKind::LastUpdatedAt])
    {
        if let Some(RowDatasetVersionMeta::Column) = kind.meta(fragment) {
            *slot = load_row_version_sequence(dataset, fragment, kind).await?;
        }
    }
    let [created_at, last_updated_at] = versions;
    Ok((
        fragment.id,
        LoadedRowLineage {
            row_ids,
            created_at,
            last_updated_at,
        },
    ))
}

/// Load row id sequences from the given dataset and fragments.
///
/// Returned as a vector of (fragment_id, sequence) pairs. These are not
/// guaranteed to be in the same order as the input fragments.
pub fn load_row_id_sequences<'a>(
    dataset: &'a Dataset,
    fragments: &'a [Fragment],
) -> impl Stream<Item = Result<(u32, Arc<RowIdSequence>)>> + 'a {
    futures::stream::iter(fragments)
        .map(|fragment| {
            load_row_id_sequence(dataset, fragment).map_ok(move |seq| (fragment.id as u32, seq))
        })
        .buffer_unordered(dataset.object_store.io_parallelism())
}

/// Build the legacy in-memory reverse index explicitly.
///
/// This API remains available for callers that need the complete mapping.
/// Normal live-row reads and writes use [`row_ids_to_row_addrs`] instead, which
/// resolves batches through an ordinary BTree without building this map. Explicit
/// takes that ignore deletions retain sequence-based resolution for stored rows
/// that index maintenance may already have removed from the BTree.
pub async fn get_row_id_index(dataset: &Dataset) -> Result<Option<Arc<RowIdIndex>>> {
    get_row_id_index_with(dataset, true).await
}

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
    if !dataset.manifest.uses_stable_row_ids() {
        return Ok(None);
    }
    // The cache is shared by every dataset opened at this URI, and one dropped
    // and recreated there restarts at version 1. Without a token for this
    // manifest generation a cached index could be the old generation's, so
    // build a private one instead of sharing.
    let Some(e_tag) = dataset.manifest_location.e_tag.as_deref() else {
        return Ok(Some(Arc::new(
            load_row_id_index(dataset, with_deletions).await?,
        )));
    };
    let key = RowIdIndexKey {
        version: dataset.manifest.version,
        with_deletions,
        e_tag: Some(e_tag),
    };
    let index = dataset
        .metadata_cache
        .get_or_insert_with_key(key, || load_row_id_index(dataset, with_deletions))
        .await?;
    Ok(Some(index))
}

/// Resolve row IDs to live addresses, preserving order, duplicates, and misses.
///
/// Stable IDs use compact inline ranges directly, an ordinary BTree on `_rowid`
/// for other covered fragments, and a column scan for the remainder. Null,
/// missing, and deleted stable IDs
/// yield `None`. Without stable IDs, each non-null ID is already an address
/// and is returned unchanged; the caller applies physical deletion filtering.
///
/// ```
/// # use arrow_array::UInt64Array;
/// # use lance::{Dataset, Result, dataset::rowids::row_ids_to_row_addrs};
/// # async fn example(dataset: &Dataset) -> Result<()> {
/// let ids = UInt64Array::from(vec![Some(7), None, Some(7)]);
/// let addresses = row_ids_to_row_addrs(dataset, &ids).await?;
/// assert_eq!(addresses[0], addresses[2]);
/// assert_eq!(addresses[1], None);
/// # Ok(())
/// # }
/// ```
pub async fn row_ids_to_row_addrs(
    dataset: &Dataset,
    row_ids: &UInt64Array,
) -> Result<Vec<Option<RowAddress>>> {
    if !dataset.manifest.uses_stable_row_ids() {
        return Ok(row_ids.iter().map(|id| id.map(RowAddress::from)).collect());
    }
    Box::pin(dataset.lookup_row_addresses(lance_core::ROW_ID, row_ids)).await
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

/// Resolve row addresses to row ids, positionally: `addr = (fragment << 32) | offset`
/// looks up `offset` in the fragment's [`RowIdSequence`]. The inverse companion of
/// [`row_ids_to_row_addrs`].
///
/// Returns one entry per input address, in order. Addresses that no longer
/// resolve — a missing fragment, an out-of-range offset, or a `None` input —
/// yield `None`. On datasets without stable row ids, addresses are the row
/// ids, so the input is returned unchanged.
pub async fn row_addrs_to_row_ids(
    dataset: &Dataset,
    addrs: impl IntoIterator<Item = Option<u64>>,
) -> Result<Vec<Option<u64>>> {
    row_addrs_to_row_ids_impl(dataset, addrs, DeletedRowBehavior::Include).await
}

/// Resolve physical row addresses to stable row ids only for live physical slots.
///
/// When stable row ids are enabled, missing fragments, out-of-range offsets,
/// deleted physical slots, and `None` inputs yield `None`. This prevents a
/// deleted slot's stable id from selecting a replacement row written by an
/// update. Without stable row ids, the input is returned unchanged.
pub(crate) async fn live_row_addrs_to_row_ids(
    dataset: &Dataset,
    addrs: impl IntoIterator<Item = Option<u64>>,
) -> Result<Vec<Option<u64>>> {
    row_addrs_to_row_ids_impl(dataset, addrs, DeletedRowBehavior::Exclude).await
}

#[derive(Clone, Copy)]
enum DeletedRowBehavior {
    Include,
    Exclude,
}

async fn row_addrs_to_row_ids_impl(
    dataset: &Dataset,
    addrs: impl IntoIterator<Item = Option<u64>>,
    deleted_row_behavior: DeletedRowBehavior,
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
        let deletion_vector = match deleted_row_behavior {
            DeletedRowBehavior::Include => None,
            DeletedRowBehavior::Exclude => fragment.get_deletion_vector().await?,
        };
        for position in positions {
            let offset = RowAddress::from(addrs[position].expect("grouped from Some")).row_offset();
            if deletion_vector
                .as_ref()
                .is_some_and(|deletions| deletions.contains(offset))
            {
                continue;
            }
            ids[position] = sequence.get(offset as usize);
        }
    }
    Ok(ids)
}

/// Build the index from freshly decoded sequences. The index then owns their
/// memory alone, so its cache charge is exact and it stays cached whenever it
/// fits; reading them through the sequence cache instead would charge each
/// sequence twice and leave a large table's index too heavy to keep. A scan
/// that needs a sequence still caches its own copy under `RowIdSequenceKey`,
/// keyed by the fragment's content; the index is keyed by manifest generation
/// and cannot stand in for it.
async fn load_row_id_index(dataset: &Dataset, with_deletions: bool) -> Result<RowIdIndex> {
    load_row_id_index_for_fragments(dataset, &dataset.manifest.fragments, with_deletions).await
}

/// Build a row id index covering only `fragments`, not the whole dataset.
///
/// A caller that already knows the row ids it will look up were all read from
/// a known, small set of fragments -- compaction rewriting a handful of
/// fragments out of a much larger table, for instance -- gets no benefit from
/// [`get_row_id_index`]'s whole-dataset index and pays for reading every other
/// fragment's row id sequence and deletion vector for nothing. This builds an
/// index scoped to just those fragments instead.
pub(crate) async fn load_row_id_index_for_fragments(
    dataset: &Dataset,
    fragments: &[Fragment],
    with_deletions: bool,
) -> Result<RowIdIndex> {
    // A `for` loop rather than `map`: a closure returning a future that borrows
    // its argument trips the higher-ranked lifetime check on the outer future.
    let mut loads = Vec::with_capacity(fragments.len());
    for fragment in fragments.iter() {
        loads.push(read_row_id_sequence(dataset, fragment).map_ok(Arc::new));
    }
    let sequences: Vec<Arc<RowIdSequence>> = futures::stream::iter(loads)
        .buffered(dataset.object_store.io_parallelism())
        .try_collect()
        .await?;

    // Without deletions the index must still tell the copies of a row apart:
    // an update keeps the old copy under a deletion vector and writes the id
    // again elsewhere. Only fragments whose id spans overlap can hold such
    // copies, so only they need their deletion vector.
    let needs_deletions = if with_deletions {
        vec![true; fragments.len()]
    } else {
        overlapping_id_spans(&sequences)
    };

    let mut loads = Vec::with_capacity(fragments.len());
    for ((fragment, row_id_sequence), needs_deletions) in
        fragments.iter().zip(sequences).zip(needs_deletions)
    {
        loads.push(async move {
            let deletion_vector = match &fragment.deletion_file {
                Some(deletion_file) if needs_deletions => {
                    read_dataset_deletion_file(dataset, fragment.id, deletion_file).await?
                }
                _ => Arc::new(DeletionVector::default()),
            };
            Ok::<_, Error>(FragmentRowIdIndex {
                fragment_id: fragment.id as u32,
                row_id_sequence,
                deletion_vector,
            })
        });
    }
    let fragment_indices: Vec<FragmentRowIdIndex> = futures::stream::iter(loads)
        .buffer_unordered(dataset.object_store.io_parallelism())
        .try_collect()
        .await?;
    RowIdIndex::new(&fragment_indices)
}

/// For each sequence, whether its span of ids (lowest to highest) intersects
/// another sequence's span. An empty sequence overlaps nothing.
fn overlapping_id_spans(sequences: &[Arc<RowIdSequence>]) -> Vec<bool> {
    let mut spans: Vec<(u64, u64, usize)> = sequences
        .iter()
        .enumerate()
        .filter_map(|(index, sequence)| {
            let span = sequence.row_id_range()?;
            Some((*span.start(), *span.end(), index))
        })
        .collect();
    spans.sort_unstable();
    let mut overlapping = vec![false; sequences.len()];
    let mut reach: Option<(u64, usize)> = None;
    for &(start, end, index) in &spans {
        if let Some((reach_end, reach_index)) = reach
            && start <= reach_end
        {
            overlapping[index] = true;
            overlapping[reach_index] = true;
        }
        if reach.is_none_or(|(reach_end, _)| end > reach_end) {
            reach = Some((end, index));
        }
    }
    overlapping
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    /// Containment, a chain of pairwise overlaps, a disjoint span and an empty
    /// sequence; the sweep must mark exactly the spans that touch another.
    #[test]
    fn test_overlapping_id_spans() {
        let seq = |ids: Vec<u64>| Arc::new(RowIdSequence::try_from_iter(ids).unwrap());
        let contained = vec![
            seq((0..=100).collect()),
            seq(vec![1, 2]),
            seq(vec![50, 60]),
            seq(vec![200, 201]),
            seq(vec![]),
        ];
        assert_eq!(
            overlapping_id_spans(&contained),
            vec![true, true, true, false, false]
        );
        // The last span overlaps only the middle one, not the one that set the
        // reach first.
        let chain = vec![seq(vec![0, 10]), seq(vec![20, 30]), seq(vec![5, 25])];
        assert_eq!(overlapping_id_spans(&chain), vec![true, true, true]);
        let disjoint = vec![seq(vec![0, 10]), seq(vec![11, 20]), seq(vec![21])];
        assert_eq!(overlapping_id_spans(&disjoint), vec![false, false, false]);
    }

    use crate::dataset::{
        ProjectionRequest, ReadParams, UpdateBuilder, WriteMode, WriteParams,
        builder::DatasetBuilder,
    };

    use super::*;

    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::index::optimize::{commit_optimize_indices, plan_optimize_indices};
    use crate::index::{DatasetIndexExt, DatasetIndexInternalExt, scalar::open_scalar_index};
    use crate::session::Session;
    use crate::utils::test::{
        DatagenExt, FailingProxyStore, FragmentCount, FragmentRowCount, copy_test_data_to_tmp,
    };
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, Int32Type, UInt32Type, UInt64Type};
    use arrow_array::{
        Array, ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, UInt32Array, UInt64Array,
        record_batch,
    };
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::Future;
    use lance_core::datatypes::Schema;
    use lance_core::{ROW_ADDR, ROW_ID, utils::address::RowAddress};
    use lance_datagen::{ArrayGeneratorExt, Dimension};
    use lance_index::{
        IndexType, metrics::NoOpMetricsCollector, optimize::OptimizeOptions,
        scalar::ScalarIndexParams,
    };
    use lance_io::object_store::ObjectStoreParams;
    use lance_table::rowids::segment::U64Segment;
    use roaring::RoaringBitmap;
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

    async fn row_id_lookup_dataset() -> Dataset {
        lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset_with_params(
                FragmentCount::from(3),
                FragmentRowCount::from(4),
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: 4,
                    ..Default::default()
                }),
            )
            .await
            .unwrap()
    }

    async fn assert_row_id_lookup_matches_scan(dataset: &Dataset) {
        let rows = dataset
            .scan()
            .with_row_id()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let ids = rows[ROW_ID].as_primitive::<UInt64Type>();
        let addresses = rows[ROW_ADDR].as_primitive::<UInt64Type>();
        let expected: HashMap<_, _> = ids
            .values()
            .iter()
            .copied()
            .zip(addresses.values().iter().copied().map(RowAddress::from))
            .collect();
        let mut keys = (0..dataset.manifest.next_row_id + 1)
            .rev()
            .map(Some)
            .collect::<Vec<_>>();
        keys.extend([Some(0), None, Some(7)]);
        let actual = row_ids_to_row_addrs(dataset, &UInt64Array::from(keys.clone()))
            .await
            .unwrap();
        let expected = keys
            .iter()
            .map(|key| key.and_then(|key| expected.get(&key).copied()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    async fn assert_row_id_btree_entries_match_scan(dataset: &Dataset) {
        let keys = UInt64Array::from_iter_values(0..dataset.manifest.next_row_id);
        let mut actual = Vec::new();
        for segment in dataset.load_indices_by_name("row_ids").await.unwrap() {
            let index = open_scalar_index(dataset, ROW_ID, &segment, &NoOpMetricsCollector)
                .await
                .unwrap();
            let mut matches = index
                .lookup(Arc::new(keys.clone()), &NoOpMetricsCollector)
                .unwrap();
            while let Some(batch) = matches.try_next().await.unwrap() {
                actual.extend(
                    batch["key_index"]
                        .as_primitive::<UInt32Type>()
                        .values()
                        .iter()
                        .map(|position| u64::from(*position))
                        .zip(
                            batch[ROW_ADDR]
                                .as_primitive::<UInt64Type>()
                                .values()
                                .iter()
                                .copied(),
                        ),
                );
            }
        }
        let scan = dataset
            .scan()
            .with_row_id()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let mut expected = scan[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .zip(
                scan[ROW_ADDR]
                    .as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied(),
            )
            .collect::<Vec<_>>();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(
            actual, expected,
            "index files must contain exactly the live addresses"
        );
    }

    #[rstest::rstest]
    #[case::unindexed(false)]
    #[case::indexed(true)]
    #[tokio::test]
    async fn test_row_id_btree_lookup_contract(#[case] indexed: bool) {
        let mut dataset = row_id_lookup_dataset().await;
        if indexed {
            dataset
                .create_index(
                    &[ROW_ID],
                    IndexType::BTree,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
            let indices = dataset.load_indices().await.unwrap();
            assert_eq!(indices[0].fields, vec![ROW_ID_FIELD_ID]);
            assert!(indices[0].results_are_row_addrs());
            assert_eq!(
                dataset.describe_indices(None).await.unwrap()[0].index_type(),
                "BTree"
            );
            dataset.index_statistics(&indices[0].name).await.unwrap();
            dataset.scalar_index_info().await.unwrap();
        }
        assert!(dataset.schema().field(ROW_ID).is_none());
        let keys = UInt64Array::from(vec![Some(9), Some(1), None, Some(9), Some(20)]);
        let failing = Arc::new(FailingProxyStore::new());
        failing.fail_when("get_opts", "", "compact ranges must not perform IO");
        let metadata_only = dataset.with_object_store_wrappers([
            failing as Arc<dyn lance_io::object_store::WrappingObjectStore>
        ]);
        assert_eq!(
            row_ids_to_row_addrs(&metadata_only, &keys).await.unwrap(),
            vec![
                Some(RowAddress::new_from_parts(2, 1)),
                Some(RowAddress::new_from_parts(0, 1)),
                None,
                Some(RowAddress::new_from_parts(2, 1)),
                None
            ],
        );
        assert_eq!(
            row_ids_to_row_addrs(&dataset, &keys).await.unwrap(),
            vec![
                Some(RowAddress::new_from_parts(2, 1)),
                Some(RowAddress::new_from_parts(0, 1)),
                None,
                Some(RowAddress::new_from_parts(2, 1)),
                None,
            ]
        );
        assert!(
            row_ids_to_row_addrs(&dataset, &UInt64Array::from(Vec::<u64>::new()))
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            row_ids_to_row_addrs(&dataset, &UInt64Array::from(vec![None, None]))
                .await
                .unwrap(),
            vec![None, None]
        );
        let taken = dataset
            .take_rows(&[9, 1, 9], dataset.schema().clone())
            .await
            .unwrap();
        assert_eq!(taken["i"].as_primitive::<Int32Type>().values(), &[9, 1, 9]);

        let before_delete = dataset.clone();
        dataset.delete("i = 1").await.unwrap();
        assert_row_id_lookup_matches_scan(&dataset).await;
        assert_eq!(
            row_ids_to_row_addrs(&before_delete, &keys).await.unwrap()[1],
            Some(RowAddress::new_from_parts(0, 1))
        );
        assert_eq!(
            row_ids_to_row_addrs(&dataset, &keys).await.unwrap()[1],
            None
        );
        assert_eq!(
            dataset.load_indices().await.unwrap().len(),
            usize::from(indexed)
        );
        assert_eq!(
            dataset.filter_deleted_ids(&[9, 1, 9, 20]).await.unwrap(),
            vec![9, 9]
        );
        // An index with reserved field ID -3 must not break unrelated filters.
        assert_eq!(
            dataset
                .scan()
                .filter("i >= 8")
                .unwrap()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows(),
            4
        );
    }

    #[tokio::test]
    async fn test_inline_range_lookup_offsets_and_snapshot_changes() {
        let mut dataset = row_id_lookup_dataset().await;
        let keys = UInt64Array::from(vec![Some(0), Some(100), Some(101), Some(2), None]);
        let original = row_ids_to_row_addrs(&dataset, &keys).await.unwrap();
        let old_snapshot = dataset.clone();
        let mut sequence = RowIdSequence::from(100..102);
        sequence.extend(RowIdSequence::from(0..2));
        let manifest = Arc::make_mut(&mut dataset.manifest);
        Arc::make_mut(&mut manifest.fragments)[0].row_id_meta = Some(RowIdMeta::Inline(
            lance_table::rowids::write_row_ids(&sequence).into(),
        ));
        assert_eq!(
            row_ids_to_row_addrs(&dataset, &keys).await.unwrap(),
            vec![
                Some(RowAddress::new_from_parts(0, 2)),
                Some(RowAddress::new_from_parts(0, 0)),
                Some(RowAddress::new_from_parts(0, 1)),
                None,
                None
            ],
        );
        assert_eq!(
            row_ids_to_row_addrs(&old_snapshot, &keys).await.unwrap(),
            original
        );

        let manifest = Arc::make_mut(&mut dataset.manifest);
        Arc::make_mut(&mut manifest.fragments)[1].row_id_meta = Some(RowIdMeta::Inline(
            lance_table::rowids::write_row_ids(&RowIdSequence::from(0..4)).into(),
        ));
        let error = row_ids_to_row_addrs(&dataset, &keys).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("not unique for key at position 0")
        );
        assert_eq!(
            row_ids_to_row_addrs(&dataset, &UInt64Array::from(vec![9]))
                .await
                .unwrap(),
            vec![Some(RowAddress::new_from_parts(2, 1))]
        );
    }

    #[tokio::test]
    async fn test_row_id_btree_partial_coverage_and_optimize() {
        let mut dataset = row_id_lookup_dataset().await;
        let params = ScalarIndexParams::default();
        dataset
            .create_index_builder(&[ROW_ID], IndexType::BTree, &params)
            .fragments(vec![0, 1])
            .name("row_ids".into())
            .await
            .unwrap();
        assert_row_id_lookup_matches_scan(&dataset).await;
        let appended = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step_custom::<Int32Type>(12, 1))
            .into_reader_rows(4.into(), 1.into());
        dataset.append(appended, None).await.unwrap();
        assert_row_id_lookup_matches_scan(&dataset).await;
        dataset.optimize_indices(&Default::default()).await.unwrap();
        assert_row_id_lookup_matches_scan(&dataset).await;
        let indices = dataset.load_indices().await.unwrap();
        let covered: roaring::RoaringBitmap = indices
            .iter()
            .flat_map(|index| {
                index
                    .fragment_bitmap
                    .iter()
                    .flat_map(|bitmap| bitmap.iter())
            })
            .collect();
        assert_eq!(&covered, dataset.fragment_bitmap().as_ref());
    }

    #[tokio::test]
    async fn test_row_id_btree_optimize_prunes_deleted_addresses() {
        let mut dataset = row_id_lookup_dataset().await;
        dataset
            .create_index(
                &[ROW_ID],
                IndexType::BTree,
                Some("row_ids".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let before_update = dataset.clone();
        let update = UpdateBuilder::new(Arc::new(dataset))
            .update_where("i = 5")
            .unwrap()
            .set("i", "100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = Arc::try_unwrap(update.new_dataset).unwrap();
        dataset.delete("i = 1").await.unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let segments = dataset.load_indices_by_name("row_ids").await.unwrap();
        assert_eq!(segments.len(), 2);
        dataset
            .optimize_indices(&OptimizeOptions::merge(segments.len()))
            .await
            .unwrap();

        let segments = dataset.load_indices_by_name("row_ids").await.unwrap();
        assert_eq!(segments.len(), 1);
        assert_row_id_btree_entries_match_scan(&dataset).await;
        assert_row_id_lookup_matches_scan(&dataset).await;
        assert_row_id_lookup_matches_scan(&before_update).await;
    }

    #[tokio::test]
    async fn test_row_id_btree_distributed_build_and_optimize() {
        let mut dataset = row_id_lookup_dataset().await;
        let source = dataset.clone();
        let params = ScalarIndexParams::default();
        let mut segments = Vec::new();
        for fragment_id in 0..3 {
            let mut worker = source.clone();
            let segment = worker
                .create_index_builder(&[ROW_ID], IndexType::BTree, &params)
                .name("row_ids".into())
                .fragments(vec![fragment_id])
                .execute_uncommitted()
                .await
                .unwrap();
            assert_eq!(segment.fields, vec![ROW_ID_FIELD_ID]);
            assert!(segment.results_are_row_addrs());
            segments.push(segment);
        }
        assert!(source.load_indices().await.unwrap().is_empty());

        // A write after the workers' snapshot must remain outside their coverage.
        dataset
            .append(
                lance_datagen::gen_batch()
                    .col("i", lance_datagen::array::step_custom::<Int32Type>(12, 1))
                    .into_reader_rows(4.into(), 1.into()),
                None,
            )
            .await
            .unwrap();
        dataset
            .commit_existing_index_segments("row_ids", ROW_ID, segments)
            .await
            .unwrap();
        let segments = dataset.load_indices_by_name("row_ids").await.unwrap();
        assert_eq!(segments.len(), 3);
        let coverage = segments
            .iter()
            .flat_map(|segment| segment.fragment_bitmap.as_ref().unwrap().iter())
            .collect::<RoaringBitmap>();
        assert_eq!(coverage, RoaringBitmap::from_iter([0, 1, 2]));
        assert_row_id_lookup_matches_scan(&dataset).await;

        let options = OptimizeOptions::merge(3)
            .max_rows_per_task(Some(4))
            .max_rows_per_segment(Some(8));
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert!(!plan.tasks.is_empty());
        let mut built = Vec::new();
        for task in &plan.tasks {
            built.extend(task.execute(&dataset).await.unwrap());
        }
        let mut merged = Vec::new();
        for merge in &plan.merges {
            merged.extend(merge.execute(&dataset, &built).await.unwrap());
        }
        commit_optimize_indices(&mut dataset, merged, &options)
            .await
            .unwrap();
        assert_row_id_btree_entries_match_scan(&dataset).await;
        assert_row_id_lookup_matches_scan(&dataset).await;
        assert_row_id_lookup_matches_scan(&source).await;
    }

    #[tokio::test]
    async fn test_row_id_btree_on_released_table() {
        let directory = copy_test_data_to_tmp("v13.0.0/stable_row_id_btree.lance").unwrap();
        let mut dataset = Dataset::open(&directory.path_str()).await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 10);
        let historical = dataset.clone();
        let original_schema = dataset.schema().clone();
        let legacy = dataset
            .load_indices_by_name("legacy_key_idx")
            .await
            .unwrap();
        assert_eq!(legacy.len(), 1);
        assert!(!legacy[0].results_are_row_addrs());
        let legacy_uuid = legacy[0].uuid;
        assert_row_id_lookup_matches_scan(&dataset).await;

        dataset
            .create_index(
                &[ROW_ID],
                IndexType::BTree,
                Some("row_ids".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let dataset = Dataset::open(&directory.path_str()).await.unwrap();
        assert_eq!(dataset.schema(), &original_schema);
        assert!(dataset.schema().field(ROW_ID).is_none());
        let retained = dataset
            .load_indices_by_name("legacy_key_idx")
            .await
            .unwrap();
        assert_eq!(retained[0].uuid, legacy_uuid);
        assert!(!retained[0].results_are_row_addrs());
        let indexed = dataset
            .scan()
            .filter("key >= 4")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(indexed.num_rows(), 7);
        assert_row_id_btree_entries_match_scan(&dataset).await;
        assert_row_id_lookup_matches_scan(&dataset).await;
        assert_row_id_lookup_matches_scan(&historical).await;
    }

    #[rstest::rstest]
    #[case::inline(false, false)]
    #[case::spilled(true, false)]
    #[case::inline_remapped(false, true)]
    #[case::spilled_remapped(true, true)]
    #[tokio::test]
    async fn test_row_id_btree_update_and_compact(#[case] spilled: bool, #[case] defer: bool) {
        let mut dataset = row_id_lookup_dataset().await;
        if spilled {
            dataset
                .update_config([
                    (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                    (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
                ])
                .await
                .unwrap();
        }
        dataset
            .create_index(
                &[ROW_ID],
                IndexType::BTree,
                Some("row_ids".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let update = UpdateBuilder::new(Arc::new(dataset))
            .update_where("i = 5")
            .unwrap()
            .set("i", "100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = Arc::try_unwrap(update.new_dataset).unwrap();
        assert_row_id_lookup_matches_scan(&dataset).await;
        dataset.delete("i = 1").await.unwrap();
        let before_compaction = dataset.clone();
        assert_row_id_lookup_matches_scan(&before_compaction).await;
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 20,
                defer_index_remap: defer,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        if spilled {
            assert!(
                dataset
                    .fragments()
                    .iter()
                    .all(|fragment| fragment.row_id_meta == Some(RowIdMeta::Column))
            );
        }
        assert_row_id_lookup_matches_scan(&dataset).await;
        assert_row_id_lookup_matches_scan(&before_compaction).await;
        let indices = dataset.load_indices().await.unwrap();
        let row_ids = indices
            .iter()
            .find(|index| index.name == "row_ids")
            .unwrap();
        let coverage =
            row_ids.fragment_bitmap.as_ref().unwrap() & dataset.fragment_bitmap().as_ref();
        assert!(
            !coverage.is_empty(),
            "eager remapping and recorded address translation preserve compacted index coverage"
        );
        dataset.optimize_indices(&Default::default()).await.unwrap();
        assert_row_id_lookup_matches_scan(&dataset).await;
        let coverage: roaring::RoaringBitmap = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .filter(|index| index.name == "row_ids")
            .flat_map(|index| index.fragment_bitmap.as_ref().unwrap().iter())
            .collect();
        assert_eq!(
            &(coverage & dataset.fragment_bitmap().as_ref()),
            dataset.fragment_bitmap().as_ref()
        );
        let taken = dataset
            .take_rows(&[5, 9, 5], dataset.schema().clone())
            .await
            .unwrap();
        assert_eq!(
            taken["i"].as_primitive::<Int32Type>().values(),
            &[100, 9, 100]
        );
    }

    #[rstest::rstest]
    #[case::signed("i")]
    #[case::unsigned("u")]
    #[tokio::test]
    async fn test_row_id_btree_unique_key_lookup(
        #[case] column: &str,
        #[values(1, 8193)] repeats: usize,
    ) {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "i",
                lance_datagen::array::cycle::<Int32Type>(vec![7, 8, 7, 9])
                    .with_nulls(&[false, true, false, false]),
            )
            .col(
                "u",
                lance_datagen::array::cycle::<UInt64Type>(vec![7, 8, 7, 9])
                    .with_nulls(&[false, true, false, false]),
            )
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(2))
            .await
            .unwrap();
        let keys: ArrayRef = match column {
            "i" => Arc::new(Int32Array::from(vec![
                Some(7),
                None,
                Some(9),
                Some(7),
                Some(99),
            ])),
            "u" => Arc::new(UInt64Array::from(vec![
                Some(7),
                None,
                Some(9),
                Some(7),
                Some(99),
            ])),
            _ => unreachable!(),
        };
        let null_value: ArrayRef = match column {
            "i" => Arc::new(Int32Array::from(vec![8])),
            "u" => Arc::new(UInt64Array::from(vec![8])),
            _ => unreachable!(),
        };
        let key_positions =
            UInt32Array::from_iter_values((0..repeats).flat_map(|_| 0..keys.len() as u32));
        let keys = arrow_select::take::take(&keys, &key_positions, None).unwrap();
        for indexed_fragments in 0..=2 {
            if indexed_fragments == 1 {
                let params = ScalarIndexParams::default();
                let fragment_id = dataset.fragments()[0].id as u32;
                dataset
                    .create_index_builder(&[column], IndexType::BTree, &params)
                    .fragments(vec![fragment_id])
                    .await
                    .unwrap();
            } else if indexed_fragments == 2 {
                dataset.optimize_indices(&Default::default()).await.unwrap();
            }
            assert_eq!(
                dataset
                    .lookup_row_addresses(column, null_value.as_ref())
                    .await
                    .unwrap(),
                vec![None],
                "a null value must not match its underlying buffer value"
            );
            let error = dataset
                .lookup_row_addresses(column, keys.as_ref())
                .await
                .unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(
                error
                    .to_string()
                    .contains("not unique for key at position 0")
            );
        }
        dataset.delete("_rowaddr = 0").await.unwrap();
        let resolved = dataset
            .lookup_row_addresses(column, keys.as_ref())
            .await
            .unwrap();
        assert_eq!(
            resolved,
            [
                Some(RowAddress::new_from_parts(1, 0)),
                None,
                Some(RowAddress::new_from_parts(1, 1)),
                Some(RowAddress::new_from_parts(1, 0)),
                None,
            ]
            .repeat(repeats)
        );
        let error = dataset
            .lookup_row_addresses("i", &UInt64Array::from(vec![7]))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("expected Int32"));
    }

    #[derive(Clone, Copy)]
    enum U64ScanLayout {
        Sorted,
        Descending,
        InteriorInversion,
        Nullable,
        AllNull,
    }

    #[rstest::rstest]
    #[case::sorted(U64ScanLayout::Sorted)]
    #[case::descending(U64ScanLayout::Descending)]
    #[case::interior_inversion(U64ScanLayout::InteriorInversion)]
    #[case::nullable(U64ScanLayout::Nullable)]
    #[case::all_null(U64ScanLayout::AllNull)]
    #[tokio::test]
    async fn test_uncovered_u64_lookup_matches_scan(
        #[case] layout: U64ScanLayout,
        #[values(false, true)] has_partial_index: bool,
    ) {
        let mut values = (0..255)
            .map(Some)
            .chain([Some(u64::MAX)])
            .collect::<Vec<_>>();
        match layout {
            U64ScanLayout::Sorted => {}
            U64ScanLayout::Descending => values.reverse(),
            U64ScanLayout::InteriorInversion => {
                // Both fragment endpoints remain ordered despite these inversions.
                values.swap(32, 33);
                values.swap(160, 161);
            }
            U64ScanLayout::Nullable => values[65] = None,
            U64ScanLayout::AllNull => values.fill(None),
        }
        let batch = record_batch!(("u", UInt64, values)).unwrap();
        let reader = RecordBatchIterator::new(
            vec![Ok(batch.slice(0, 128)), Ok(batch.slice(128, 128))],
            batch.schema(),
        );
        let mut dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 128,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments().len(), 2);
        if has_partial_index {
            let fragment_id = dataset.fragments()[0].id as u32;
            dataset
                .create_index_builder(&["u"], IndexType::BTree, &ScalarIndexParams::default())
                .fragments(vec![fragment_id])
                .await
                .unwrap();
        }
        let keys = UInt64Array::from(vec![
            Some(9999),
            Some(65),
            None,
            Some(u64::MAX),
            Some(0),
            Some(65),
            Some(128),
            Some(255),
            Some(511),
            Some(9999),
        ])
        .slice(1, 8);
        let single_key = UInt64Array::from(vec![Some(65), None, Some(65)]);
        let dense_keys = UInt64Array::from(
            (0..256)
                .map(Some)
                .chain([None, Some(u64::MAX), Some(65)])
                .collect::<Vec<_>>(),
        );
        for is_after_delete in [false, true] {
            if is_after_delete {
                dataset.delete("u = 65 OR u = 128").await.unwrap();
            }
            let scan = dataset
                .scan()
                .with_row_address()
                .try_into_batch()
                .await
                .unwrap();
            let expected = scan["u"]
                .as_primitive::<UInt64Type>()
                .iter()
                .zip(scan[ROW_ADDR].as_primitive::<UInt64Type>().values())
                .filter_map(|(value, &address)| {
                    value.map(|value| (value, RowAddress::from(address)))
                })
                .collect::<HashMap<_, _>>();
            for keys in [&keys, &single_key, &dense_keys] {
                let actual = dataset.lookup_row_addresses("u", keys).await.unwrap();
                assert_eq!(
                    actual,
                    keys.iter()
                        .map(|key| key.and_then(|key| expected.get(&key).copied()))
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn test_uncovered_sorted_u64_lookup_rejects_duplicates(
        #[values(false, true)] has_partial_index: bool,
    ) {
        let mut values = (0..256).collect::<Vec<u64>>();
        values[136] = 135;
        let batch = record_batch!(("u", UInt64, values)).unwrap();
        let reader = RecordBatchIterator::new(
            vec![Ok(batch.slice(0, 128)), Ok(batch.slice(128, 128))],
            batch.schema(),
        );
        let mut dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 128,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments().len(), 2);
        if has_partial_index {
            let fragment_id = dataset.fragments()[0].id as u32;
            dataset
                .create_index_builder(&["u"], IndexType::BTree, &ScalarIndexParams::default())
                .fragments(vec![fragment_id])
                .await
                .unwrap();
        }
        let keys = UInt64Array::from(vec![Some(135), None, Some(135), Some(511), Some(0)]);
        let error = dataset.lookup_row_addresses("u", &keys).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("not unique for key at position 0")
        );

        let fragment = dataset.fragments()[1].id as u32;
        let removed = RowAddress::new_from_parts(fragment, 7);
        dataset
            .delete(&format!("{ROW_ADDR} = {}", u64::from(removed)))
            .await
            .unwrap();
        assert_eq!(
            dataset.lookup_row_addresses("u", &keys).await.unwrap(),
            vec![
                Some(RowAddress::new_from_parts(fragment, 8)),
                None,
                Some(RowAddress::new_from_parts(fragment, 8)),
                None,
                Some(RowAddress::new_from_parts(
                    dataset.fragments()[0].id as u32,
                    0
                )),
            ]
        );
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum LookupMaintenance {
        Build,
        Eager,
        Defer,
        Optimize,
    }

    #[rstest::rstest]
    #[case::built_after_compaction(LookupMaintenance::Build)]
    #[case::remapped_eagerly(LookupMaintenance::Eager)]
    #[case::remapped_by_fri(LookupMaintenance::Defer)]
    #[case::optimized_after_compaction(LookupMaintenance::Optimize)]
    #[tokio::test]
    async fn test_row_id_btree_covered_lookup_does_not_read_lineage(
        #[case] maintenance: LookupMaintenance,
    ) {
        let directory = lance_core::utils::tempfile::TempStrDir::default();
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_dataset_with_params(
                &directory,
                FragmentCount::from(3),
                FragmentRowCount::from(4),
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: 4,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        dataset
            .update_config([
                (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
            ])
            .await
            .unwrap();
        if maintenance != LookupMaintenance::Build {
            dataset
                .create_index(
                    &[ROW_ID],
                    IndexType::BTree,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
        }
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 20,
                defer_index_remap: maintenance == LookupMaintenance::Defer,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments()[0].row_id_meta, Some(RowIdMeta::Column));
        if maintenance == LookupMaintenance::Build {
            dataset
                .create_index(
                    &[ROW_ID],
                    IndexType::BTree,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
        } else if maintenance == LookupMaintenance::Optimize {
            dataset.optimize_indices(&Default::default()).await.unwrap();
        }
        dataset
            .update_config([(SPILL_ROW_LINEAGE_CONFIG_KEY, "false")])
            .await
            .unwrap();
        dataset
            .append(
                lance_datagen::gen_batch()
                    .col("i", lance_datagen::array::step_custom::<Int32Type>(12, 1))
                    .into_reader_rows(4.into(), 1.into()),
                None,
            )
            .await
            .unwrap();
        let tail_fragment = dataset.fragments().last().unwrap().id as u32;
        drop(dataset);

        let failing = Arc::new(FailingProxyStore::new());
        failing.fail_when(
            "get_opts",
            "data/",
            "lookup must not read the row-ID column",
        );
        let dataset = DatasetBuilder::from_uri(&directory)
            .with_read_params(ReadParams {
                index_cache_size_bytes: 0,
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(failing),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();
        let keys = UInt64Array::from(vec![Some(9), Some(1), None, Some(9), Some(99), Some(14)]);
        assert_eq!(
            row_ids_to_row_addrs(&dataset, &keys).await.unwrap(),
            vec![
                Some(RowAddress::new_from_parts(3, 9)),
                Some(RowAddress::new_from_parts(3, 1)),
                None,
                Some(RowAddress::new_from_parts(3, 9)),
                None,
                Some(RowAddress::new_from_parts(tail_fragment, 2)),
            ]
        );
        let rebound = Arc::new(FailingProxyStore::new());
        rebound.fail_when("get_opts", "_indices/", "lookup must use the rebound store");
        let rebound = dataset.with_object_store_wrappers([
            rebound as Arc<dyn lance_io::object_store::WrappingObjectStore>
        ]);
        let error = row_ids_to_row_addrs(&rebound, &keys).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("lookup must use the rebound store")
        );
        let error = get_row_id_index(&dataset).await.unwrap_err();
        assert!(matches!(error, Error::IO { .. }));
        assert!(
            error
                .to_string()
                .contains("lookup must not read the row-ID column")
        );
    }

    #[tokio::test]
    async fn test_row_id_btree_shuffled_training() {
        let dataset = row_id_lookup_dataset().await;
        let updated = UpdateBuilder::new(Arc::new(dataset))
            .update_where("i % 2 = 0")
            .unwrap()
            .set("i", "i + 100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let mut dataset = Arc::try_unwrap(updated.new_dataset).unwrap();
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 20,
                defer_index_remap: false,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let sequence = load_row_id_sequence(&dataset, &dataset.fragments()[0])
            .await
            .unwrap();
        assert!(!sequence.iter().is_sorted());
        dataset
            .create_index(
                &[ROW_ID],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        assert_row_id_lookup_matches_scan(&dataset).await;
    }

    #[tokio::test]
    async fn test_empty_dataset_rowids() {
        let schema = sequence_batch(0..0).schema();
        let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        assert!(dataset.manifest.uses_stable_row_ids());

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).unwrap().is_none());

        assert_eq!(dataset.manifest().next_row_id, 0);
        dataset
            .create_index(
                &[ROW_ID],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        assert_eq!(
            row_ids_to_row_addrs(&dataset, &UInt64Array::from(vec![Some(0), None]))
                .await
                .unwrap(),
            vec![None, None]
        );
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
        let entries_before = session.metadata_cache_stats().await.num_entries;
        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert_eq!(
            session.metadata_cache_stats().await.num_entries,
            entries_before + 1
        );
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
            entries_before + 2
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
        dataset.manifest_location.e_tag = None;
        let session = dataset.session();
        let entries_before = session.metadata_cache_stats().await.num_entries;
        let first = get_row_id_index(&dataset).await.unwrap().unwrap();
        let second = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(
            session.metadata_cache_stats().await.num_entries,
            entries_before
        );
        assert_eq!(
            second.get(9).unwrap(),
            Some(RowAddress::new_from_parts(0, 9))
        );
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

        failing_store.clear_fail_when("get_opts", "_deletions");
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
        // Leave this generation's index in the session cache: `RowIdIndexKey`
        // is scoped by version alone, so the recreated dataset below reaches
        // the same key, and no sequence load may be answered from it.
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
        // Updated fragments get fresh row ids.
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
        fn segment_shapes(dataset: &Dataset) -> (Vec<&'static str>, bool) {
            let mut kinds = Vec::new();
            let mut folded = false;
            for fragment in dataset.manifest.fragments.iter() {
                let Some(RowIdMeta::Inline(data)) = &fragment.row_id_meta else {
                    panic!("expected inline row ids, got {:?}", fragment.row_id_meta);
                };
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
            (kinds, folded)
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
        let (kinds, folded) = segment_shapes(&dataset);
        assert!(kinds.contains(&"bitmap") && !folded, "{kinds:?}");

        // Opting in re-encodes every eligible fragment in the same commit: on
        // the wire only plain `Range` segments, folded back when decoded.
        dataset
            .update_config([(RANGE_SEGMENTS_CONFIG_KEY, "true")])
            .await
            .unwrap();
        let (kinds, folded) = segment_shapes(&dataset);
        assert!(kinds.iter().all(|kind| *kind == "range"), "{kinds:?}");
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
        let (kinds, folded) = segment_shapes(&reopened);
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
