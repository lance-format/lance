// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared source-merged row-ID lookup for high-overlap fragment sets.
//!
//! When more than [`GROUP_OVERLAP_THRESHOLD`] non-range fragments overlap in
//! row-ID space, per-fragment warm probing becomes O(F) per request. Building a
//! single deletion-free [`MergedSourceGroup`] over those immutable fragment
//! sources restores O(1) warm resolution while sharing the group across
//! snapshots (deletes and incremental appends only filter deletions or probe
//! the small uncovered delta). Cache capacity and live-density gates preserve
//! pruned per-fragment lookups under tight memory pressure.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use futures::{StreamExt, TryStreamExt};
use lance_core::cache::{CacheKey, CacheNamespace, InternalCacheKey, KeyBuilder};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::DeletionVector;
use lance_core::utils::tokio::spawn_cpu;
use lance_table::format::{Fragment, RowIdMeta};
use lance_table::rowids::{RowIdSequence, read_row_ids, segment::U64Segment};
use rayon::prelude::*;

use crate::session::index_caches::{
    MergedRowIdGroupKey, RowIdFragmentSpanKey, SnapshotRowIdRouteKey,
};
use crate::{Dataset, Error, Result};

use super::cache::{deletion_identity, load_deletions, sequence_identity};
use super::read_row_id_sequence;
use super::resolve::{FragmentSpan, resolve_fragments};

/// Overlap depth above which a shared source-merged lookup beats probing
/// individual fragments (matches `lance_table::rowids::index::MAX_PROBE_DEPTH`).
const GROUP_OVERLAP_THRESHOLD: usize = 64;
/// Number of evenly spaced candidate fragments sampled to confirm high overlap
/// before allocating a merged group.
const SAMPLE_STRIDE_COUNT: usize = 64;
/// Minimum overlap depth within the stride sample to trigger group merging.
const SAMPLE_OVERLAP_TRIGGER: usize = 8;
/// Encoded value in [`GroupLookup::Dense`] indicating an unpopulated row ID.
const EMPTY_SLOT: u64 = 0;
/// Encoded value in [`GroupLookup::Dense`] indicating multiple candidate
/// physical addresses stored in `overflow`.
const MULTI_SLOT: u64 = u64::MAX;

/// Immutable, deletion-free lookup shared across snapshots that contain the
/// same underlying fragment sources.
#[derive(Debug)]
pub struct MergedSourceGroup {
    /// Sorted unique immutable source identities covered by this group.
    pub(super) sources: Vec<InternalCacheKey>,
    lookup: GroupLookup,
}

#[derive(Debug)]
enum GroupLookup {
    /// Direct `addrs[row_id]` table when row IDs are dense (`next_row_id <= 2 * rows`).
    /// Each slot stores `0` for absent, `u64::MAX` for overflow/multi-candidate,
    /// or `u64::from(RowAddress) + 1` for a single candidate.
    Dense {
        addrs: Vec<AtomicU64>,
        overflow: HashMap<u64, Vec<RowAddress>>,
    },
    /// Sorted `(ids, addrs)` table when overlapping row IDs have wide gaps.
    Sparse {
        ids: Vec<u64>,
        addrs: Vec<u64>,
        overflow: HashMap<u64, Vec<RowAddress>>,
    },
}

fn overflow_deep_size(overflow: &HashMap<u64, Vec<RowAddress>>) -> usize {
    overflow.capacity() * std::mem::size_of::<(u64, Vec<RowAddress>)>()
        + overflow
            .values()
            .map(|addrs| addrs.capacity() * std::mem::size_of::<RowAddress>())
            .sum::<usize>()
}

impl DeepSizeOf for MergedSourceGroup {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        let sources_bytes = self.sources.capacity() * std::mem::size_of::<InternalCacheKey>();
        let lookup_bytes = match &self.lookup {
            GroupLookup::Dense { addrs, overflow } => {
                addrs.capacity() * std::mem::size_of::<AtomicU64>() + overflow_deep_size(overflow)
            }
            GroupLookup::Sparse {
                ids,
                addrs,
                overflow,
            } => {
                ids.capacity() * std::mem::size_of::<u64>()
                    + addrs.capacity() * std::mem::size_of::<u64>()
                    + overflow_deep_size(overflow)
            }
        };
        sources_bytes + lookup_bytes
    }
}

/// Lightweight per-snapshot routing table cached under `(version, e_tag)`.
///
/// Holds only a [`Weak`] reference to the heavy [`MergedSourceGroup`] so cache
/// eviction is never bypassed.
#[derive(Debug)]
pub struct SnapshotRowIdRoute {
    group_key: InternalCacheKey,
    #[allow(dead_code)]
    group: Weak<MergedSourceGroup>,
    /// Covered fragments that have a deletion file in this snapshot, keyed by
    /// `fragment.id as u32` -> `(manifest_fragment_index, deletion_identity)`.
    group_deletions: HashMap<u32, (usize, InternalCacheKey)>,
    /// Indices into `dataset.manifest.fragments` not covered by `group`.
    uncovered_indices: Vec<usize>,
}

impl DeepSizeOf for SnapshotRowIdRoute {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.group_deletions.capacity() * std::mem::size_of::<(u32, (usize, InternalCacheKey))>()
            + self.uncovered_indices.capacity() * std::mem::size_of::<usize>()
    }
}

/// Attempt to resolve `row_ids` via a cached or newly built [`MergedSourceGroup`].
///
/// Returns `Ok(Some(addresses))` when a group handled the request, or `Ok(None)`
/// when this snapshot should use the per-fragment cache path.
pub(super) async fn try_resolve_with_group(
    dataset: &Dataset,
    row_ids: &[u64],
) -> Result<Option<Vec<Option<RowAddress>>>> {
    try_resolve_with_threshold(
        dataset,
        row_ids,
        GROUP_OVERLAP_THRESHOLD,
        SAMPLE_OVERLAP_TRIGGER,
    )
    .await
}

async fn try_resolve_with_threshold(
    dataset: &Dataset,
    row_ids: &[u64],
    min_candidates: usize,
    sample_overlap_trigger: usize,
) -> Result<Option<Vec<Option<RowAddress>>>> {
    let Some(e_tag) = dataset.table_state.manifest_location.e_tag.as_deref() else {
        return Ok(None);
    };
    let route_key = SnapshotRowIdRouteKey {
        version: dataset.manifest.version,
        e_tag,
    };
    if let Some(route) = dataset.index_cache.get_with_key(&route_key).await
        && let Some(group) = dataset
            .index_cache
            .get_with_key(&MergedRowIdGroupKey {
                group: route.group_key,
            })
            .await
    {
        return resolve_with_route(dataset, &route, &group, row_ids)
            .await
            .map(Some);
    }

    let fragments = dataset.manifest.fragments.as_ref();
    if fragments.len() <= min_candidates {
        return Ok(None);
    }

    let mut source_keys = Vec::with_capacity(fragments.len());
    for fragment in fragments.iter() {
        source_keys.push(sequence_identity(dataset, fragment)?);
    }

    // Step 1: Reuse an already-resident group whose sources are a subset of this
    // snapshot (e.g. after deletions or small incremental appends).
    if let Some((group_key, group)) =
        find_reusable_group(dataset, &source_keys, min_candidates).await?
    {
        let route = Arc::new(build_snapshot_route(
            dataset,
            &source_keys,
            group_key,
            &group,
        )?);
        dataset
            .index_cache
            .insert_with_key(&route_key, route.clone())
            .await;
        return resolve_with_route(dataset, &route, &group, row_ids)
            .await
            .map(Some);
    }

    // Step 2: Check whether this snapshot qualifies for building a new shared group.
    let Some((group_key, group)) = maybe_build_group(
        dataset,
        &source_keys,
        min_candidates,
        sample_overlap_trigger,
    )
    .await?
    else {
        return Ok(None);
    };
    let route = Arc::new(build_snapshot_route(
        dataset,
        &source_keys,
        group_key,
        &group,
    )?);
    dataset
        .index_cache
        .insert_with_key(&route_key, route.clone())
        .await;
    resolve_with_route(dataset, &route, &group, row_ids)
        .await
        .map(Some)
}

async fn find_reusable_group(
    dataset: &Dataset,
    source_keys: &[InternalCacheKey],
    min_candidates: usize,
) -> Result<Option<(InternalCacheKey, Arc<MergedSourceGroup>)>> {
    let live = dataset
        .session()
        .row_id_fragment_registry
        .live_groups(&dataset.table_state.uri)?;
    if live.is_empty() {
        return Ok(None);
    }
    let snapshot_set: HashSet<InternalCacheKey> = source_keys.iter().copied().collect();
    let mut best: Option<(InternalCacheKey, Arc<MergedSourceGroup>)> = None;
    for (group_key, candidate) in live {
        if candidate.sources.len() <= min_candidates || candidate.sources.len() > snapshot_set.len()
        {
            continue;
        }
        if best
            .as_ref()
            .is_some_and(|(_, current)| current.sources.len() >= candidate.sources.len())
        {
            continue;
        }
        if !candidate
            .sources
            .iter()
            .all(|source| snapshot_set.contains(source))
        {
            continue;
        }
        // Only reuse groups still resident in index_cache so evicted allocations
        // are never kept alive by the registry.
        if let Some(resident) = dataset
            .index_cache
            .get_with_key(&MergedRowIdGroupKey { group: group_key })
            .await
        {
            best = Some((group_key, resident));
        }
    }
    Ok(best)
}

fn build_snapshot_route(
    dataset: &Dataset,
    source_keys: &[InternalCacheKey],
    group_key: InternalCacheKey,
    group: &Arc<MergedSourceGroup>,
) -> Result<SnapshotRowIdRoute> {
    let covered: HashSet<InternalCacheKey> = group.sources.iter().copied().collect();
    let mut group_deletions = HashMap::new();
    let mut uncovered_indices = Vec::new();
    for (idx, (fragment, source)) in dataset
        .manifest
        .fragments
        .iter()
        .zip(source_keys)
        .enumerate()
    {
        if covered.contains(source) {
            if let Some(del_key) = deletion_identity(dataset, fragment, true)? {
                let frag_id = u32::try_from(fragment.id).map_err(|_| {
                    Error::invalid_input(format!("fragment id {} exceeds u32", fragment.id))
                })?;
                group_deletions.insert(frag_id, (idx, del_key));
            }
        } else {
            uncovered_indices.push(idx);
        }
    }
    Ok(SnapshotRowIdRoute {
        group_key,
        group: Arc::downgrade(group),
        group_deletions,
        uncovered_indices,
    })
}

async fn maybe_build_group(
    dataset: &Dataset,
    source_keys: &[InternalCacheKey],
    min_candidates: usize,
    sample_overlap_trigger: usize,
) -> Result<Option<(InternalCacheKey, Arc<MergedSourceGroup>)>> {
    let Some(cache_capacity) = dataset.index_cache.capacity_bytes() else {
        return Ok(None);
    };
    if cache_capacity == 0 {
        return Ok(None);
    }
    let max_group_bytes = cache_capacity.saturating_mul(3) / 4;
    if max_group_bytes == 0 {
        return Ok(None);
    }

    let fragments = dataset.manifest.fragments.as_ref();
    let mut candidates: Vec<(usize, InternalCacheKey)> = Vec::new();
    let mut total_physical_rows: u64 = 0;
    let mut total_deleted_rows: u64 = 0;
    for (idx, (fragment, &source)) in fragments.iter().zip(source_keys).enumerate() {
        let Some(rows) = fragment.physical_rows else {
            continue;
        };
        if rows == 0 {
            continue;
        }
        // Compact Range-only inline fragments are already O(1) and tiny; never
        // fold them into an 8-bytes/row merged table.
        if let Some(RowIdMeta::Inline(bytes)) = &fragment.row_id_meta
            && let Ok(seq) = read_row_ids(bytes)
            && seq
                .segments()
                .iter()
                .all(|seg| matches!(seg, U64Segment::Range(_)))
        {
            continue;
        }
        candidates.push((idx, source));
        total_physical_rows = total_physical_rows.saturating_add(rows as u64);
        if let Some(del) = &fragment.deletion_file
            && let Some(deleted) = del.num_deleted_rows
        {
            total_deleted_rows = total_deleted_rows.saturating_add(deleted as u64);
        }
    }
    if candidates.len() <= min_candidates || total_physical_rows == 0 {
        return Ok(None);
    }

    // Preserve pruned per-fragment lookup for cold heavily-deleted snapshots.
    let total_live_rows = total_physical_rows.saturating_sub(total_deleted_rows);
    if total_live_rows.saturating_mul(2) < total_physical_rows {
        return Ok(None);
    }

    let next_row_id = dataset.manifest.next_row_id;
    let dense_len = usize::try_from(next_row_id).ok().filter(|&len| {
        len > 0
            && (len as u64) <= total_physical_rows.saturating_mul(2)
            && len.saturating_mul(std::mem::size_of::<AtomicU64>()) <= max_group_bytes
    });
    let sparse_fits = usize::try_from(total_physical_rows)
        .ok()
        .is_some_and(|rows| rows.saturating_mul(16) <= max_group_bytes);
    if dense_len.is_none() && !sparse_fits {
        return Ok(None);
    }

    // Sample evenly spaced candidate fragments to verify high span overlap
    // before allocating the merged table. Keep loaded sequences to avoid double reads.
    let sample_count = SAMPLE_STRIDE_COUNT.min(candidates.len());
    let mut sample_indices = Vec::with_capacity(sample_count);
    for s in 0..sample_count {
        let pos = s * candidates.len() / sample_count;
        sample_indices.push(candidates[pos]);
    }
    let loaded_sample: Vec<(usize, InternalCacheKey, Arc<RowIdSequence>)> =
        futures::stream::iter(sample_indices)
            .map(|(frag_idx, source)| async move {
                let fragment = &dataset.manifest.fragments[frag_idx];
                let seq = Arc::new(read_row_id_sequence(dataset, fragment).await?);
                let span = seq.row_id_range();
                dataset
                    .table_state
                    .metadata_cache
                    .insert_with_key(
                        &RowIdFragmentSpanKey { source },
                        Arc::new(FragmentSpan(span)),
                    )
                    .await;
                Ok::<_, Error>((frag_idx, source, seq))
            })
            .buffer_unordered(dataset.table_state.object_store.io_parallelism())
            .try_collect()
            .await?;

    let sample_spans: Vec<(u64, u64)> = loaded_sample
        .iter()
        .filter_map(|(_, _, seq)| seq.row_id_range().map(|r| (*r.start(), *r.end())))
        .collect();
    if max_span_overlap(&sample_spans) < sample_overlap_trigger {
        return Ok(None);
    }

    let mut sorted_sources: Vec<InternalCacheKey> =
        candidates.iter().map(|&(_, source)| source).collect();
    sorted_sources.sort_unstable_by_key(|k| *k.as_bytes());
    sorted_sources.dedup();
    if sorted_sources.len() != candidates.len() {
        // Identical source repeated in manifest: let per-fragment path validate.
        return Ok(None);
    }
    let group_key = group_identity(&sorted_sources);
    let build_lock = dataset
        .session()
        .row_id_fragment_registry
        .for_group(&dataset.table_state.uri, group_key)?;
    let mut slot = build_lock.lock().await;
    if let Some(resident) = dataset
        .index_cache
        .get_with_key(&MergedRowIdGroupKey { group: group_key })
        .await
    {
        *slot = Some(Arc::downgrade(&resident));
        return Ok(Some((group_key, resident)));
    }

    let preloaded: HashMap<usize, Arc<RowIdSequence>> = loaded_sample
        .into_iter()
        .map(|(idx, _, seq)| (idx, seq))
        .collect();

    let lookup = if let Some(len) = dense_len {
        build_dense_lookup(dataset, &candidates, preloaded, len).await?
    } else {
        build_sparse_lookup(
            dataset,
            &candidates,
            preloaded,
            total_physical_rows as usize,
        )
        .await?
    };

    let group = Arc::new(MergedSourceGroup {
        sources: sorted_sources,
        lookup,
    });
    dataset
        .index_cache
        .insert_with_key(&MergedRowIdGroupKey { group: group_key }, group.clone())
        .await;
    // Only register if the cache actually retained the entry.
    if dataset
        .index_cache
        .get_with_key(&MergedRowIdGroupKey { group: group_key })
        .await
        .is_some()
    {
        dataset.session().row_id_fragment_registry.register_group(
            &dataset.table_state.uri,
            group_key,
            &group,
        )?;
        *slot = Some(Arc::downgrade(&group));
    }
    Ok(Some((group_key, group)))
}

async fn build_dense_lookup(
    dataset: &Dataset,
    candidates: &[(usize, InternalCacheKey)],
    mut preloaded: HashMap<usize, Arc<RowIdSequence>>,
    len: usize,
) -> Result<GroupLookup> {
    let addrs = spawn_cpu(move || {
        let mut v = Vec::with_capacity(len);
        v.resize_with(len, || AtomicU64::new(EMPTY_SLOT));
        Ok::<_, Error>(Arc::new(v))
    })
    .await?;
    let overflow: Arc<Mutex<HashMap<u64, Vec<RowAddress>>>> = Arc::new(Mutex::new(HashMap::new()));

    let mut initial = Vec::with_capacity(preloaded.len());
    let mut remaining = Vec::with_capacity(candidates.len().saturating_sub(preloaded.len()));
    for &(frag_idx, source) in candidates {
        if let Some(seq) = preloaded.remove(&frag_idx) {
            let frag_id = u32::try_from(dataset.manifest.fragments[frag_idx].id).map_err(|_| {
                Error::invalid_input(format!(
                    "fragment id {} exceeds u32",
                    dataset.manifest.fragments[frag_idx].id
                ))
            })?;
            initial.push((frag_id, seq));
        } else {
            remaining.push((frag_idx, source));
        }
    }

    if !initial.is_empty() {
        let addrs_ref = addrs.clone();
        let overflow_ref = overflow.clone();
        spawn_cpu(move || {
            initial.into_par_iter().try_for_each(|(frag_id, seq)| {
                scatter_sequence(frag_id, &seq, &addrs_ref, &overflow_ref)
            })
        })
        .await?;
    }

    let io_par = dataset.table_state.object_store.io_parallelism();
    let batch_size = io_par.clamp(1, 32);
    let mut stream = futures::stream::iter(remaining)
        .map(|(frag_idx, source)| async move {
            let fragment = &dataset.manifest.fragments[frag_idx];
            let frag_id = u32::try_from(fragment.id).map_err(|_| {
                Error::invalid_input(format!("fragment id {} exceeds u32", fragment.id))
            })?;
            let seq = read_row_id_sequence(dataset, fragment).await?;
            let span = seq.row_id_range();
            dataset
                .table_state
                .metadata_cache
                .insert_with_key(
                    &RowIdFragmentSpanKey { source },
                    Arc::new(FragmentSpan(span)),
                )
                .await;
            Ok::<_, Error>((frag_id, seq))
        })
        .buffer_unordered(io_par);

    let mut batch = Vec::with_capacity(batch_size);
    while let Some(item) = stream.try_next().await? {
        batch.push(item);
        if batch.len() >= batch_size {
            let chunk = std::mem::replace(&mut batch, Vec::with_capacity(batch_size));
            let addrs_ref = addrs.clone();
            let overflow_ref = overflow.clone();
            spawn_cpu(move || {
                chunk.into_par_iter().try_for_each(|(frag_id, seq)| {
                    scatter_sequence(frag_id, &seq, &addrs_ref, &overflow_ref)
                })
            })
            .await?;
        }
    }
    if !batch.is_empty() {
        let addrs_ref = addrs.clone();
        let overflow_ref = overflow.clone();
        spawn_cpu(move || {
            batch.into_par_iter().try_for_each(|(frag_id, seq)| {
                scatter_sequence(frag_id, &seq, &addrs_ref, &overflow_ref)
            })
        })
        .await?;
    }

    let addrs = Arc::try_unwrap(addrs)
        .map_err(|_| Error::internal("dense group addresses still referenced after build"))?;
    let overflow = Arc::try_unwrap(overflow)
        .map_err(|_| Error::internal("dense group overflow still referenced after build"))?
        .into_inner()
        .map_err(|error| Error::internal(format!("dense group overflow poisoned: {error}")))?;
    Ok(GroupLookup::Dense { addrs, overflow })
}

fn scatter_sequence(
    frag_id: u32,
    seq: &RowIdSequence,
    addrs: &[AtomicU64],
    overflow: &Mutex<HashMap<u64, Vec<RowAddress>>>,
) -> Result<()> {
    for (offset, id) in seq.iter().enumerate() {
        let offset = u32::try_from(offset).map_err(|_| {
            Error::invalid_input(format!(
                "fragment {frag_id} row offset {offset} exceeds u32 for row ID {id}"
            ))
        })?;
        let addr = RowAddress::new_from_parts(frag_id, offset);
        insert_dense_slot(addrs, overflow, id, addr)?;
    }
    Ok(())
}

fn insert_dense_slot(
    addrs: &[AtomicU64],
    overflow: &Mutex<HashMap<u64, Vec<RowAddress>>>,
    id: u64,
    addr: RowAddress,
) -> Result<()> {
    let raw = u64::from(addr);
    let idx = usize::try_from(id).ok().filter(|&i| i < addrs.len());
    if let Some(idx) = idx
        && raw < MULTI_SLOT - 1
    {
        let encoded = raw + 1;
        if addrs[idx]
            .compare_exchange(EMPTY_SLOT, encoded, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return Ok(());
        }
    }
    let mut map = overflow
        .lock()
        .map_err(|error| Error::internal(format!("dense group overflow poisoned: {error}")))?;
    let entry = map.entry(id).or_default();
    if let Some(idx) = idx {
        let previous = addrs[idx].swap(MULTI_SLOT, Ordering::AcqRel);
        if previous != EMPTY_SLOT && previous != MULTI_SLOT {
            let prev_addr = RowAddress::from(previous - 1);
            if prev_addr.fragment_id() == addr.fragment_id() {
                return Err(Error::internal(format!(
                    "row id index corrupt: stable row id {id} has multiple physical positions"
                )));
            }
            entry.push(prev_addr);
        }
    }
    if entry
        .iter()
        .any(|existing| existing.fragment_id() == addr.fragment_id())
    {
        return Err(Error::internal(format!(
            "row id index corrupt: stable row id {id} has multiple physical positions"
        )));
    }
    entry.push(addr);
    Ok(())
}

async fn build_sparse_lookup(
    dataset: &Dataset,
    candidates: &[(usize, InternalCacheKey)],
    mut preloaded: HashMap<usize, Arc<RowIdSequence>>,
    expected_rows: usize,
) -> Result<GroupLookup> {
    let mut pairs: Vec<(u64, u64)> = Vec::with_capacity(expected_rows);
    let mut remaining = Vec::with_capacity(candidates.len().saturating_sub(preloaded.len()));
    for &(frag_idx, source) in candidates {
        if let Some(seq) = preloaded.remove(&frag_idx) {
            let frag_id = u32::try_from(dataset.manifest.fragments[frag_idx].id).map_err(|_| {
                Error::invalid_input(format!(
                    "fragment id {} exceeds u32",
                    dataset.manifest.fragments[frag_idx].id
                ))
            })?;
            append_pairs(frag_id, &seq, &mut pairs)?;
        } else {
            remaining.push((frag_idx, source));
        }
    }

    let io_par = dataset.table_state.object_store.io_parallelism();
    let mut stream = futures::stream::iter(remaining)
        .map(|(frag_idx, source)| async move {
            let fragment = &dataset.manifest.fragments[frag_idx];
            let frag_id = u32::try_from(fragment.id).map_err(|_| {
                Error::invalid_input(format!("fragment id {} exceeds u32", fragment.id))
            })?;
            let seq = read_row_id_sequence(dataset, fragment).await?;
            let span = seq.row_id_range();
            dataset
                .table_state
                .metadata_cache
                .insert_with_key(
                    &RowIdFragmentSpanKey { source },
                    Arc::new(FragmentSpan(span)),
                )
                .await;
            Ok::<_, Error>((frag_id, seq))
        })
        .buffer_unordered(io_par);
    while let Some((frag_id, seq)) = stream.try_next().await? {
        append_pairs(frag_id, &seq, &mut pairs)?;
    }

    spawn_cpu(move || {
        pairs.par_sort_unstable_by_key(|&(id, _)| id);
        let mut ids = Vec::with_capacity(pairs.len());
        let mut addrs = Vec::with_capacity(pairs.len());
        let mut overflow: HashMap<u64, Vec<RowAddress>> = HashMap::new();
        let mut i = 0;
        while i < pairs.len() {
            let id = pairs[i].0;
            let mut j = i + 1;
            while j < pairs.len() && pairs[j].0 == id {
                j += 1;
            }
            if j - i == 1 {
                ids.push(id);
                addrs.push(pairs[i].1);
            } else {
                let mut seen_frags = HashSet::with_capacity(j - i);
                let mut group_addrs = Vec::with_capacity(j - i);
                for &(_, raw) in &pairs[i..j] {
                    let addr = RowAddress::from(raw);
                    if !seen_frags.insert(addr.fragment_id()) {
                        return Err(Error::internal(format!(
                            "row id index corrupt: stable row id {id} has multiple physical positions"
                        )));
                    }
                    group_addrs.push(addr);
                }
                ids.push(id);
                addrs.push(MULTI_SLOT);
                overflow.insert(id, group_addrs);
            }
            i = j;
        }
        ids.shrink_to_fit();
        addrs.shrink_to_fit();
        Ok(GroupLookup::Sparse {
            ids,
            addrs,
            overflow,
        })
    })
    .await
}

fn append_pairs(frag_id: u32, seq: &RowIdSequence, out: &mut Vec<(u64, u64)>) -> Result<()> {
    for (offset, id) in seq.iter().enumerate() {
        let offset = u32::try_from(offset).map_err(|_| {
            Error::invalid_input(format!(
                "fragment {frag_id} row offset {offset} exceeds u32 for row ID {id}"
            ))
        })?;
        out.push((id, u64::from(RowAddress::new_from_parts(frag_id, offset))));
    }
    Ok(())
}

async fn resolve_with_route(
    dataset: &Dataset,
    route: &SnapshotRowIdRoute,
    group: &MergedSourceGroup,
    row_ids: &[u64],
) -> Result<Vec<Option<RowAddress>>> {
    let next_row_id = dataset.manifest.next_row_id;

    // Fast path: every fragment is covered by the group and none have deletions.
    // Avoids sorting `row_ids` or allocating deletion maps.
    if route.group_deletions.is_empty() && route.uncovered_indices.is_empty() {
        let mut result = vec![None; row_ids.len()];
        match &group.lookup {
            GroupLookup::Dense { addrs, overflow } => {
                for (slot, &id) in result.iter_mut().zip(row_ids) {
                    if id >= next_row_id {
                        continue;
                    }
                    let Some(stored) = usize::try_from(id).ok().and_then(|i| addrs.get(i)) else {
                        if let Some(cands) = overflow.get(&id) {
                            *slot = single_live_no_deletions(id, cands)?;
                        }
                        continue;
                    };
                    let val = stored.load(Ordering::Relaxed);
                    if val == EMPTY_SLOT {
                        continue;
                    }
                    if val != MULTI_SLOT {
                        *slot = Some(RowAddress::from(val - 1));
                    } else if let Some(cands) = overflow.get(&id) {
                        *slot = single_live_no_deletions(id, cands)?;
                    }
                }
            }
            GroupLookup::Sparse {
                ids,
                addrs,
                overflow,
            } => {
                for (slot, &id) in result.iter_mut().zip(row_ids) {
                    if id >= next_row_id {
                        continue;
                    }
                    if let Ok(pos) = ids.binary_search(&id) {
                        let raw = addrs[pos];
                        if raw != MULTI_SLOT {
                            *slot = Some(RowAddress::from(raw));
                        } else if let Some(cands) = overflow.get(&id) {
                            *slot = single_live_no_deletions(id, cands)?;
                        }
                    }
                }
            }
        }
        tracing::debug!(
            event = "row_id_fragment_resolution",
            total_fragments = dataset.manifest.fragments.len(),
            grouped_fragments = group.sources.len(),
            cached_fragments = 0usize,
            loaded_fragments = 0usize,
            prepared_fragments = 0usize,
            excluded_fragments = 0usize,
            preparation_parallelism = 1usize,
            requested_ids = row_ids.len(),
            unique_requested_ids = row_ids.len(),
            fragment_batch_calls = 1usize,
            "resolved row ids with shared source group"
        );
        return Ok(result);
    }

    // General path: group hits + snapshot deletion filtering + uncovered delta fragments.
    let mut sorted: Vec<_> = row_ids
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, id)| *id < next_row_id)
        .map(|(position, id)| (id, position))
        .collect();
    sorted.sort_unstable_by_key(|&(id, _)| id);
    let mut unique_ids: Vec<u64> = sorted.iter().map(|&(id, _)| id).collect();
    unique_ids.dedup();

    enum Candidate<'a> {
        None,
        Single(RowAddress),
        Multi(&'a [RowAddress]),
    }

    let mut candidates: Vec<Candidate<'_>> = Vec::with_capacity(unique_ids.len());
    let mut needed_del_frags: HashSet<u32> = HashSet::new();
    let track_del = |frag_id: u32, needed: &mut HashSet<u32>| {
        if route.group_deletions.contains_key(&frag_id) {
            needed.insert(frag_id);
        }
    };

    match &group.lookup {
        GroupLookup::Dense { addrs, overflow } => {
            for &id in &unique_ids {
                let val = usize::try_from(id)
                    .ok()
                    .and_then(|i| addrs.get(i))
                    .map_or(MULTI_SLOT, |a| a.load(Ordering::Relaxed));
                if val == EMPTY_SLOT {
                    candidates.push(Candidate::None);
                } else if val != MULTI_SLOT {
                    let addr = RowAddress::from(val - 1);
                    track_del(addr.fragment_id(), &mut needed_del_frags);
                    candidates.push(Candidate::Single(addr));
                } else if let Some(cands) = overflow.get(&id) {
                    for addr in cands {
                        track_del(addr.fragment_id(), &mut needed_del_frags);
                    }
                    candidates.push(Candidate::Multi(cands.as_slice()));
                } else {
                    candidates.push(Candidate::None);
                }
            }
        }
        GroupLookup::Sparse {
            ids,
            addrs,
            overflow,
        } => {
            for &id in &unique_ids {
                if let Ok(pos) = ids.binary_search(&id) {
                    let raw = addrs[pos];
                    if raw != MULTI_SLOT {
                        let addr = RowAddress::from(raw);
                        track_del(addr.fragment_id(), &mut needed_del_frags);
                        candidates.push(Candidate::Single(addr));
                    } else if let Some(cands) = overflow.get(&id) {
                        for addr in cands {
                            track_del(addr.fragment_id(), &mut needed_del_frags);
                        }
                        candidates.push(Candidate::Multi(cands.as_slice()));
                    } else {
                        candidates.push(Candidate::None);
                    }
                } else {
                    candidates.push(Candidate::None);
                }
            }
        }
    }

    // Load deletion vectors only for hit fragments that actually have deletions.
    let loaded_deletions: HashMap<u32, Arc<DeletionVector>> = if needed_del_frags.is_empty() {
        HashMap::new()
    } else {
        let fragments = dataset.manifest.fragments.as_ref();
        futures::stream::iter(needed_del_frags)
            .map(|frag_id| {
                let (frag_idx, del_key) = route.group_deletions[&frag_id];
                async move {
                    let dv = load_deletions(dataset, &fragments[frag_idx], del_key).await?;
                    Ok::<_, Error>((frag_id, dv))
                }
            })
            .buffer_unordered(dataset.table_state.object_store.io_parallelism())
            .try_collect()
            .await?
    };

    let is_visible = |addr: RowAddress| -> bool {
        loaded_deletions
            .get(&addr.fragment_id())
            .is_none_or(|dv| !dv.contains(addr.row_offset()))
    };

    let mut found = vec![None; unique_ids.len()];
    for ((slot, &id), cand) in found.iter_mut().zip(&unique_ids).zip(candidates) {
        match cand {
            Candidate::None => {}
            Candidate::Single(addr) => {
                if is_visible(addr) {
                    *slot = Some(addr);
                }
            }
            Candidate::Multi(cands) => {
                let mut live: Option<RowAddress> = None;
                for &addr in cands {
                    if is_visible(addr) && live.replace(addr).is_some() {
                        return Err(Error::internal(format!(
                            "row id index corrupt: stable row id {id} is live in multiple fragments"
                        )));
                    }
                }
                *slot = live;
            }
        }
    }

    let (cached_fragments, loaded_fragments, prepared_fragments, excluded_fragments, parallelism) =
        if route.uncovered_indices.is_empty() {
            (0, 0, 0, 0, 1)
        } else {
            let uncovered: Vec<&Fragment> = route
                .uncovered_indices
                .iter()
                .map(|&idx| &dataset.manifest.fragments[idx])
                .collect();
            resolve_fragments(dataset, &uncovered, &unique_ids, &mut found).await?
        };

    tracing::debug!(
        event = "row_id_fragment_resolution",
        total_fragments = dataset.manifest.fragments.len(),
        grouped_fragments = group.sources.len(),
        cached_fragments,
        loaded_fragments,
        prepared_fragments,
        excluded_fragments,
        preparation_parallelism = parallelism,
        requested_ids = row_ids.len(),
        unique_requested_ids = unique_ids.len(),
        fragment_batch_calls = 1 + cached_fragments + prepared_fragments,
        "resolved row ids with shared source group"
    );

    let mut result = vec![None; row_ids.len()];
    let mut unique_position = 0;
    for (id, original_position) in sorted {
        if unique_ids[unique_position] != id {
            unique_position += 1;
        }
        result[original_position] = found[unique_position];
    }
    Ok(result)
}

fn single_live_no_deletions(id: u64, cands: &[RowAddress]) -> Result<Option<RowAddress>> {
    if cands.len() > 1 {
        return Err(Error::internal(format!(
            "row id index corrupt: stable row id {id} is live in multiple fragments"
        )));
    }
    Ok(cands.first().copied())
}

fn max_span_overlap(spans: &[(u64, u64)]) -> usize {
    if spans.is_empty() {
        return 0;
    }
    let mut starts: Vec<u64> = spans.iter().map(|&(s, _)| s).collect();
    let mut ends: Vec<u64> = spans.iter().map(|&(_, e)| e).collect();
    starts.sort_unstable();
    ends.sort_unstable();
    let mut closed = 0;
    let mut depth = 0;
    for (opened, &start) in starts.iter().enumerate() {
        while closed < ends.len() && ends[closed] < start {
            closed += 1;
        }
        depth = depth.max(opened + 1 - closed);
    }
    depth
}

fn group_identity(sorted_sources: &[InternalCacheKey]) -> InternalCacheKey {
    let mut builder = KeyBuilder::new(
        CacheNamespace::root(),
        MergedRowIdGroupKey::type_name(),
        MergedRowIdGroupKey::schema(),
    );
    builder.write_u64(sorted_sources.len() as u64);
    for source in sorted_sources {
        builder.write_fixed_bytes(source.as_bytes());
    }
    builder.finish()
}

#[cfg(test)]
mod tests {
    use arrow_array::types::UInt64Type;
    use arrow_array::{RecordBatchIterator, UInt64Array, cast::AsArray};
    use arrow_schema::{DataType, Field, Schema};
    use lance_table::rowids::version::RowDatasetVersionSequence;
    use lance_table::transaction::{Operation, Transaction};

    use super::*;
    use crate::dataset::rowids::{
        INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, RowLineage, SPILL_ROW_LINEAGE_CONFIG_KEY,
        get_row_id_index, place_row_lineage, resolve_row_ids,
    };
    use crate::dataset::write::CommitBuilder;
    use crate::dataset::{UpdateBuilder, WriteParams};
    use crate::session::Session;

    async fn overlapping_spilled_dataset(
        num_frags: usize,
        rows_per_frag: usize,
        overlapping: bool,
    ) -> Dataset {
        let total = num_frags * rows_per_frag;
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::UInt64, false),
            Field::new("val", DataType::UInt64, false),
        ]));
        let id_at = move |f: usize, off: usize| -> u64 {
            if overlapping {
                // Bijective scatter across all fragments (stride coprime to power-of-2 total)
                let global = f * rows_per_frag + off;
                ((global * 37 + 13) % total) as u64
            } else {
                (f * rows_per_frag + off) as u64
            }
        };
        let batch_schema = schema.clone();
        let batches = (0..num_frags).map(move |f| {
            let ids: Vec<u64> = (0..rows_per_frag).map(|o| id_at(f, o)).collect();
            let vals: Vec<u64> = ids.iter().map(|id| id * 3 + 1).collect();
            arrow_array::RecordBatch::try_new(
                batch_schema.clone(),
                vec![
                    Arc::new(UInt64Array::from(ids)),
                    Arc::new(UInt64Array::from(vals)),
                ],
            )
        });
        let uri = format!("memory://group_test_{}_{}", num_frags, overlapping);
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(batches, schema),
            &uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: rows_per_frag,
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
        let mut fragments = dataset.manifest().fragments.as_ref().clone();
        for (f, frag) in fragments.iter_mut().enumerate() {
            let lineage = RowLineage {
                row_ids: RowIdSequence::try_from_iter((0..rows_per_frag).map(|o| id_at(f, o)))
                    .unwrap(),
                created_at: RowDatasetVersionSequence::from_uniform_row_count(
                    rows_per_frag as u64,
                    1,
                ),
                last_updated_at: RowDatasetVersionSequence::from_uniform_row_count(
                    rows_per_frag as u64,
                    1,
                ),
            };
            place_row_lineage(&dataset, &lineage)
                .await
                .unwrap()
                .apply(frag);
        }
        let tx = Transaction::new_from_version(
            dataset.version().version,
            Operation::Merge {
                fragments,
                schema: dataset.schema().clone(),
                preserves_nullability: true,
            },
        );
        CommitBuilder::new(Arc::new(dataset))
            .execute(tx)
            .await
            .unwrap()
    }

    fn with_fresh_session(mut ds: Dataset, index_cap: usize) -> Dataset {
        let session = Arc::new(Session::new(
            index_cap,
            16 * 1024 * 1024,
            ds.session().store_registry(),
        ));
        ds.index_cache = Arc::new(session.index_cache.for_dataset(&ds.table_state.uri));
        ds.table_state.metadata_cache =
            Arc::new(session.metadata_cache().for_dataset(&ds.table_state.uri));
        ds.table_state.set_session(session.table_session.clone());
        ds.session = session;
        ds
    }

    #[tokio::test]
    async fn dense_group_reuses_across_updates_deletes_and_timetravel() {
        // 80 fragments > GROUP_OVERLAP_THRESHOLD (64), 32 rows each -> 2560 rows.
        let ds0 = with_fresh_session(overlapping_spilled_dataset(80, 32, true).await, 32 << 20);
        let oracle0 = get_row_id_index(&ds0).await.unwrap().unwrap();
        let query: Vec<u64> = vec![0, 5, 5, 42, 1024, 2559, 999_999];

        // Cold build + warm route hit on ds0
        let first = resolve_row_ids(&ds0, &query, true).await.unwrap().unwrap();
        let warm = resolve_row_ids(&ds0, &query, true).await.unwrap().unwrap();
        let expected0 = oracle0.get_many(&query).unwrap();
        assert_eq!(first, expected0);
        assert_eq!(warm, expected0);

        // Confirm a MergedSourceGroup is registered and resident
        let groups_v0 = ds0
            .session()
            .row_id_fragment_registry
            .live_groups(&ds0.table_state.uri)
            .unwrap();
        assert_eq!(groups_v0.len(), 1);
        let group_ptr = Arc::as_ptr(&groups_v0[0].1);

        // Update row 42 and delete row 5 on a new snapshot sharing the same session
        let updated = UpdateBuilder::new(Arc::new(ds0.clone()))
            .update_where("key = 42")
            .unwrap()
            .set("val", "99999")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;
        let mut ds1 = updated.as_ref().clone();
        ds1.delete("key = 5").await.unwrap();
        let oracle1 = get_row_id_index(&ds1).await.unwrap().unwrap();

        let res1 = resolve_row_ids(&ds1, &query, true).await.unwrap().unwrap();
        let res1_warm = resolve_row_ids(&ds1, &query, true).await.unwrap().unwrap();
        let expected1 = oracle1.get_many(&query).unwrap();
        assert_eq!(res1, expected1);
        assert_eq!(res1_warm, expected1);
        assert_eq!(res1[1], None); // deleted row 5
        assert_eq!(res1[2], None); // duplicate deleted row 5
        assert_ne!(res1[3], expected0[3]); // updated row 42 moved to new fragment

        // Still only ONE MergedSourceGroup shared across ds0 and ds1!
        let groups_v1 = ds1
            .session()
            .row_id_fragment_registry
            .live_groups(&ds1.table_state.uri)
            .unwrap();
        assert_eq!(groups_v1.len(), 1);
        assert_eq!(Arc::as_ptr(&groups_v1[0].1), group_ptr);

        // Time-travel back to ds0 still returns original addresses for rows 5 & 42
        let back0 = resolve_row_ids(&ds0, &query, true).await.unwrap().unwrap();
        assert_eq!(back0, expected0);
    }

    #[tokio::test]
    async fn disjoint_spilled_fragments_do_not_build_merged_group() {
        let ds = with_fresh_session(overlapping_spilled_dataset(80, 32, false).await, 32 << 20);
        let oracle = get_row_id_index(&ds).await.unwrap().unwrap();
        let query = vec![0, 500, 2559, 9999];
        let actual = resolve_row_ids(&ds, &query, true).await.unwrap().unwrap();
        assert_eq!(actual, oracle.get_many(&query).unwrap());
        let groups = ds
            .session()
            .row_id_fragment_registry
            .live_groups(&ds.table_state.uri)
            .unwrap();
        assert!(
            groups.is_empty(),
            "disjoint fragments should stay per-fragment"
        );
    }

    #[tokio::test]
    async fn small_cache_or_heavy_deletions_skips_merged_group() {
        let mut ds = overlapping_spilled_dataset(80, 32, true).await;
        ds.delete("key % 10 < 8").await.unwrap(); // 80% deleted -> <50% live
        let ds = with_fresh_session(ds, 32 << 20);
        let oracle = get_row_id_index(&ds).await.unwrap().unwrap();
        let query = vec![8, 9, 18, 19, 0, 1];
        let actual = resolve_row_ids(&ds, &query, true).await.unwrap().unwrap();
        assert_eq!(actual, oracle.get_many(&query).unwrap());
        let groups = ds
            .session()
            .row_id_fragment_registry
            .live_groups(&ds.table_state.uri)
            .unwrap();
        assert!(
            groups.is_empty(),
            "heavily deleted cold snapshot should use pruned per-fragment path"
        );
    }

    #[tokio::test]
    async fn eviction_does_not_retain_merged_group_via_route() {
        let ds = with_fresh_session(overlapping_spilled_dataset(80, 32, true).await, 32 << 20);
        let _ = resolve_row_ids(&ds, &[0, 42], true).await.unwrap().unwrap();
        let groups = ds
            .session()
            .row_id_fragment_registry
            .live_groups(&ds.table_state.uri)
            .unwrap();
        assert_eq!(groups.len(), 1);
        let weak = Arc::downgrade(&groups[0].1);
        drop(groups);

        ds.index_cache.clear().await;
        assert_eq!(
            weak.strong_count(),
            0,
            "clearing index_cache must release MergedSourceGroup"
        );
        // Subsequent query transparently rebuilds and returns exact results.
        let batch = ds
            .take_rows(
                &[0, 42],
                crate::dataset::ProjectionRequest::from(ds.schema().clone()),
            )
            .await
            .unwrap();
        let keys = batch["key"].as_primitive::<UInt64Type>();
        assert_eq!(keys.values().as_ref(), &[0, 42]);
    }
}
