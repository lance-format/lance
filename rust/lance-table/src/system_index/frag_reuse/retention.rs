// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Which records of a fragment reuse history some index still needs.
//!
//! Computed from the history's digests, the stored index metadata and the
//! live fragment set alone, with no payload IO, so both the maintenance trim
//! and an action-based commit's `TrimFragReuse` reach the same verdict.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use lance_core::{Error, Result};
use log::warn;
use prost::Message;
use roaring::RoaringBitmap;

use super::FragReuseVersion;
use crate::format::IndexMetadata;
use crate::format::pb::fragment_reuse_index_details as pb_fri;
use crate::system_index::is_system_index;

/// Every fragment the reuse chain touches (old + new) across all versions.
///
/// Prefer [`reuse_version_reaches`], [`reuse_version_reach`], or
/// [`ReuseVersionReachIndex`] when checking [`is_index_remap_caught_up`], so an
/// index rebuilt after an earlier version does not falsely pin later disjoint
/// versions.
pub fn reuse_chain_frag_bitmap(versions: &[FragReuseVersion]) -> RoaringBitmap {
    let mut bitmap = RoaringBitmap::new();
    for version in versions {
        bitmap.extend(version.old_frag_ids().iter().map(|&id| id as u32));
        bitmap.extend(version.new_frag_ids().iter().map(|&id| id as u32));
    }
    bitmap
}

/// Pre-indexed view of a fragment reuse history for on-demand reach queries.
///
/// Building the index sorts the versions stably by `dataset_version` and
/// indexes consumer rewrite groups by their `old_frags` without cloning
/// version payloads. Each [`Self::reach`] call traverses only groups reachable
/// from that version's `new_frags`, avoiding quadratic scans across unrelated
/// versions.
///
/// # Example
///
/// ```
/// use lance_table::system_index::frag_reuse::{
///     FragDigest, FragReuseGroup, FragReuseVersion,
///     retention::ReuseVersionReachIndex,
/// };
/// use roaring::RoaringBitmap;
///
/// # fn digest(id: u64) -> FragDigest {
/// #     FragDigest { id, physical_rows: 10, num_deleted_rows: 0 }
/// # }
/// # fn version(dataset_version: u64, old: &[u64], new: &[u64]) -> FragReuseVersion {
/// #     FragReuseVersion {
/// #         dataset_version,
/// #         groups: vec![FragReuseGroup {
/// #             changed_row_addrs: Vec::new(),
/// #             old_frags: old.iter().copied().map(digest).collect(),
/// #             new_frags: new.iter().copied().map(digest).collect(),
/// #         }],
/// #     }
/// # }
/// let versions = [version(10, &[1, 2], &[3]), version(20, &[3], &[4])];
/// let index = ReuseVersionReachIndex::new(versions.iter());
/// assert_eq!(index.reach(0), RoaringBitmap::from_iter([1, 2, 3, 4]));
/// assert_eq!(index.reach(1), RoaringBitmap::from_iter([3, 4]));
/// ```
pub struct ReuseVersionReachIndex<'a> {
    versions: Vec<&'a FragReuseVersion>,
    /// Input indices sorted stably by `dataset_version`.
    order: Vec<usize>,
    /// `rank_of[input_idx]` is the position of `input_idx` in `order`.
    rank_of: Vec<usize>,
    /// For each `old_frag.id`, the `(rank, group_idx)` pairs of groups that
    /// consume it, sorted by `(rank, group_idx)`.
    consumers_by_old_frag: HashMap<u64, Vec<(usize, usize)>>,
}

impl<'a> ReuseVersionReachIndex<'a> {
    /// Build a reach index borrowing `versions`.
    pub fn new(versions: impl IntoIterator<Item = &'a FragReuseVersion>) -> Self {
        let versions: Vec<&'a FragReuseVersion> = versions.into_iter().collect();
        let mut order: Vec<usize> = (0..versions.len()).collect();
        order.sort_by_key(|&i| versions[i].dataset_version);

        let mut rank_of = vec![0; versions.len()];
        let mut consumers_by_old_frag: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
        for (rank, &input_idx) in order.iter().enumerate() {
            rank_of[input_idx] = rank;
            for (group_idx, group) in versions[input_idx].groups.iter().enumerate() {
                for old_frag in &group.old_frags {
                    consumers_by_old_frag
                        .entry(old_frag.id)
                        .or_default()
                        .push((rank, group_idx));
                }
            }
        }

        Self {
            versions,
            order,
            rank_of,
            consumers_by_old_frag,
        }
    }

    /// Fragments reachable from `versions[input_idx]`: its `old_frags` plus the
    /// forward closure of its `new_frags` through later versions in
    /// `dataset_version` order (preserving input order on ties).
    ///
    /// # Panics
    ///
    /// Panics if `input_idx >= versions.len()`.
    pub fn reach(&self, input_idx: usize) -> RoaringBitmap {
        self.reach_with_work(input_idx).0
    }

    /// Per-version reach bitmaps for all indexed versions, aligned with input
    /// positions.
    ///
    /// Computes group closures in a single reverse-topological pass so every
    /// rewrite group is visited once across the entire history, avoiding
    /// quadratic re-traversal on long chains.
    ///
    /// # Example
    ///
    /// ```
    /// use lance_table::system_index::frag_reuse::{
    ///     FragDigest, FragReuseGroup, FragReuseVersion,
    ///     retention::ReuseVersionReachIndex,
    /// };
    /// use roaring::RoaringBitmap;
    ///
    /// # fn digest(id: u64) -> FragDigest {
    /// #     FragDigest { id, physical_rows: 10, num_deleted_rows: 0 }
    /// # }
    /// # fn version(dataset_version: u64, old: &[u64], new: &[u64]) -> FragReuseVersion {
    /// #     FragReuseVersion {
    /// #         dataset_version,
    /// #         groups: vec![FragReuseGroup {
    /// #             changed_row_addrs: Vec::new(),
    /// #             old_frags: old.iter().copied().map(digest).collect(),
    /// #             new_frags: new.iter().copied().map(digest).collect(),
    /// #         }],
    /// #     }
    /// # }
    /// let versions = [version(10, &[1], &[2]), version(20, &[2], &[3])];
    /// let index = ReuseVersionReachIndex::new(versions.iter());
    /// assert_eq!(
    ///     index.all_reaches(),
    ///     vec![
    ///         RoaringBitmap::from_iter([1, 2, 3]),
    ///         RoaringBitmap::from_iter([2, 3]),
    ///     ],
    /// );
    /// ```
    pub fn all_reaches(&self) -> Vec<RoaringBitmap> {
        self.all_reaches_with_work().0
    }

    /// Whether each indexed version's reach intersects `bitmap`, aligned with
    /// input positions, computed via reverse-topological boolean propagation
    /// without materializing per-version reach bitmaps.
    ///
    /// # Example
    ///
    /// ```
    /// use lance_table::system_index::frag_reuse::{
    ///     FragDigest, FragReuseGroup, FragReuseVersion,
    ///     retention::ReuseVersionReachIndex,
    /// };
    /// use roaring::RoaringBitmap;
    ///
    /// # fn digest(id: u64) -> FragDigest {
    /// #     FragDigest { id, physical_rows: 10, num_deleted_rows: 0 }
    /// # }
    /// # fn version(dataset_version: u64, old: &[u64], new: &[u64]) -> FragReuseVersion {
    /// #     FragReuseVersion {
    /// #         dataset_version,
    /// #         groups: vec![FragReuseGroup {
    /// #             changed_row_addrs: Vec::new(),
    /// #             old_frags: old.iter().copied().map(digest).collect(),
    /// #             new_frags: new.iter().copied().map(digest).collect(),
    /// #         }],
    /// #     }
    /// # }
    /// let versions = [
    ///     version(10, &[1], &[2]),
    ///     version(20, &[2], &[3]),
    ///     version(30, &[8], &[9]),
    /// ];
    /// let index = ReuseVersionReachIndex::new(versions.iter());
    /// assert_eq!(
    ///     index.versions_overlapping_bitmap(&RoaringBitmap::from_iter([3])),
    ///     vec![true, true, false],
    /// );
    /// ```
    pub fn versions_overlapping_bitmap(&self, bitmap: &RoaringBitmap) -> Vec<bool> {
        self.versions_overlapping_bitmap_with_work(bitmap).0
    }

    fn versions_overlapping_bitmap_with_work(
        &self,
        bitmap: &RoaringBitmap,
    ) -> (Vec<bool>, usize, usize) {
        if bitmap.is_empty() {
            return (vec![false; self.versions.len()], 0, 0);
        }

        let mut group_hits: Vec<Vec<bool>> = self
            .order
            .iter()
            .map(|&input_idx| vec![false; self.versions[input_idx].groups.len()])
            .collect();
        let mut overlaps = vec![false; self.versions.len()];
        let mut visited_groups = 0usize;
        let mut downstream_edges = 0usize;

        for rank in (0..self.order.len()).rev() {
            let input_idx = self.order[rank];
            let version = self.versions[input_idx];
            for (group_idx, group) in version.groups.iter().enumerate().rev() {
                visited_groups += 1;
                let mut hit = false;
                for new_frag in &group.new_frags {
                    if bitmap_contains(bitmap, new_frag.id) {
                        hit = true;
                        break;
                    }
                    if let Some(consumers) = self.consumers_by_old_frag.get(&new_frag.id) {
                        let pos = consumers.partition_point(|&entry| entry <= (rank, group_idx));
                        for &(consumer_rank, consumer_group_idx) in &consumers[pos..] {
                            downstream_edges += 1;
                            if group_hits[consumer_rank][consumer_group_idx] {
                                hit = true;
                                break;
                            }
                        }
                        if hit {
                            break;
                        }
                    }
                }
                group_hits[rank][group_idx] = hit;
            }

            let mut version_hit = false;
            'version_check: for group in &version.groups {
                for old_frag in &group.old_frags {
                    if bitmap_contains(bitmap, old_frag.id) {
                        version_hit = true;
                        break 'version_check;
                    }
                }
                for new_frag in &group.new_frags {
                    if bitmap_contains(bitmap, new_frag.id) {
                        version_hit = true;
                        break 'version_check;
                    }
                    if let Some(consumers) = self.consumers_by_old_frag.get(&new_frag.id) {
                        let pos = consumers.partition_point(|&(r, _)| r <= rank);
                        for &(consumer_rank, consumer_group_idx) in &consumers[pos..] {
                            if group_hits[consumer_rank][consumer_group_idx] {
                                version_hit = true;
                                break 'version_check;
                            }
                        }
                    }
                }
            }
            overlaps[input_idx] = version_hit;
        }

        (overlaps, visited_groups, downstream_edges)
    }

    fn all_reaches_with_work(&self) -> (Vec<RoaringBitmap>, usize, usize) {
        let mut group_tail_reaches: Vec<Vec<RoaringBitmap>> = self
            .order
            .iter()
            .map(|&input_idx| vec![RoaringBitmap::new(); self.versions[input_idx].groups.len()])
            .collect();
        let mut reaches = vec![RoaringBitmap::new(); self.versions.len()];
        let mut visited_groups = 0usize;
        let mut downstream_edges = 0usize;

        for rank in (0..self.order.len()).rev() {
            let input_idx = self.order[rank];
            let version = self.versions[input_idx];
            for (group_idx, group) in version.groups.iter().enumerate().rev() {
                visited_groups += 1;
                let mut tail = RoaringBitmap::new();
                for new_frag in &group.new_frags {
                    if let Ok(id) = u32::try_from(new_frag.id) {
                        tail.insert(id);
                    }
                    if let Some(consumers) = self.consumers_by_old_frag.get(&new_frag.id) {
                        let pos = consumers.partition_point(|&entry| entry <= (rank, group_idx));
                        for &(consumer_rank, consumer_group_idx) in &consumers[pos..] {
                            downstream_edges += 1;
                            tail |= &group_tail_reaches[consumer_rank][consumer_group_idx];
                        }
                    }
                }
                group_tail_reaches[rank][group_idx] = tail;
            }

            let mut reach = RoaringBitmap::new();
            for group in &version.groups {
                for old_frag in &group.old_frags {
                    if let Ok(id) = u32::try_from(old_frag.id) {
                        reach.insert(id);
                    }
                }
                for new_frag in &group.new_frags {
                    if let Ok(id) = u32::try_from(new_frag.id) {
                        reach.insert(id);
                    }
                    if let Some(consumers) = self.consumers_by_old_frag.get(&new_frag.id) {
                        let pos = consumers.partition_point(|&(r, _)| r <= rank);
                        for &(consumer_rank, consumer_group_idx) in &consumers[pos..] {
                            reach |= &group_tail_reaches[consumer_rank][consumer_group_idx];
                        }
                    }
                }
            }
            reaches[input_idx] = reach;
        }

        (reaches, visited_groups, downstream_edges)
    }

    fn reach_with_work(&self, input_idx: usize) -> (RoaringBitmap, usize) {
        let version = self.versions[input_idx];
        let start_rank = self.rank_of[input_idx];

        let mut reach = RoaringBitmap::new();
        let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();

        for group in &version.groups {
            for old_frag in &group.old_frags {
                if let Ok(id) = u32::try_from(old_frag.id) {
                    reach.insert(id);
                }
            }
            for new_frag in &group.new_frags {
                if let Ok(id) = u32::try_from(new_frag.id) {
                    reach.insert(id);
                }
                if let Some(consumers) = self.consumers_by_old_frag.get(&new_frag.id) {
                    let pos = consumers.partition_point(|&(r, _)| r <= start_rank);
                    for &target in &consumers[pos..] {
                        heap.push(Reverse(target));
                    }
                }
            }
        }

        if heap.is_empty() {
            return (reach, 0);
        }

        let mut visited: HashSet<(usize, usize)> = HashSet::new();
        let mut visited_groups = 0usize;
        while let Some(Reverse((rank, group_idx))) = heap.pop() {
            if !visited.insert((rank, group_idx)) {
                continue;
            }
            visited_groups += 1;
            let group = &self.versions[self.order[rank]].groups[group_idx];
            for new_frag in &group.new_frags {
                if let Ok(id) = u32::try_from(new_frag.id) {
                    reach.insert(id);
                }
                if let Some(consumers) = self.consumers_by_old_frag.get(&new_frag.id) {
                    let pos = consumers.partition_point(|&entry| entry <= (rank, group_idx));
                    for &target in &consumers[pos..] {
                        if !visited.contains(&target) {
                            heap.push(Reverse(target));
                        }
                    }
                }
            }
        }

        (reach, visited_groups)
    }
}

/// Fragments reachable from a single reuse version: `old_frags(version)` plus
/// the forward closure of `new_frags(version)` across `later_versions` walked
/// in `dataset_version` order.
///
/// Records in `later_versions` with `dataset_version < version.dataset_version`
/// are ignored. When records share `version.dataset_version`, callers must pass
/// only those that occurred after `version` (ties are walked in their input
/// order after `version`).
///
/// For each later version, any rewrite group whose `old_frags` intersect the
/// current reachable set adds its `new_frags`. Because fragment IDs are never
/// reused and deferred-remap commits only advance a covering index's
/// `fragment_bitmap` along rewrite chains, an index disjoint from this reach
/// holds no row address `version` remaps.
///
/// # Example
///
/// ```
/// use lance_table::system_index::frag_reuse::{
///     FragDigest, FragReuseGroup, FragReuseVersion,
///     retention::reuse_version_reach,
/// };
/// use roaring::RoaringBitmap;
///
/// # fn digest(id: u64) -> FragDigest {
/// #     FragDigest { id, physical_rows: 10, num_deleted_rows: 0 }
/// # }
/// # fn version(dataset_version: u64, old: &[u64], new: &[u64]) -> FragReuseVersion {
/// #     FragReuseVersion {
/// #         dataset_version,
/// #         groups: vec![FragReuseGroup {
/// #             changed_row_addrs: Vec::new(),
/// #             old_frags: old.iter().copied().map(digest).collect(),
/// #             new_frags: new.iter().copied().map(digest).collect(),
/// #         }],
/// #     }
/// # }
/// let v1 = version(10, &[1, 2], &[3]);
/// let v2 = version(20, &[3], &[4]);
/// let disjoint = version(25, &[8, 9], &[10]);
/// assert_eq!(
///     reuse_version_reach(&v1, &[disjoint, v2]),
///     RoaringBitmap::from_iter([1, 2, 3, 4]),
/// );
/// ```
pub fn reuse_version_reach(
    version: &FragReuseVersion,
    later_versions: &[FragReuseVersion],
) -> RoaringBitmap {
    let index = ReuseVersionReachIndex::new(
        std::iter::once(version).chain(
            later_versions
                .iter()
                .filter(|v| v.dataset_version >= version.dataset_version),
        ),
    );
    index.reach(0)
}

/// Per-version reach bitmaps for `versions`, aligned with input positions
/// (`result[i]` is the reach of `versions[i]`).
///
/// Versions are walked in `dataset_version` order (preserving input order on
/// ties) so each version's closure follows only later versions in that order.
///
/// # Example
///
/// ```
/// use lance_table::system_index::frag_reuse::{
///     FragDigest, FragReuseGroup, FragReuseVersion,
///     retention::reuse_version_reaches,
/// };
/// use roaring::RoaringBitmap;
///
/// # fn digest(id: u64) -> FragDigest {
/// #     FragDigest { id, physical_rows: 10, num_deleted_rows: 0 }
/// # }
/// # fn version(dataset_version: u64, old: &[u64], new: &[u64]) -> FragReuseVersion {
/// #     FragReuseVersion {
/// #         dataset_version,
/// #         groups: vec![FragReuseGroup {
/// #             changed_row_addrs: Vec::new(),
/// #             old_frags: old.iter().copied().map(digest).collect(),
/// #             new_frags: new.iter().copied().map(digest).collect(),
/// #         }],
/// #     }
/// # }
/// let v1 = version(10, &[1, 2], &[3]);
/// let v2 = version(20, &[8, 9], &[10]);
/// let reaches = reuse_version_reaches(&[v1, v2]);
/// assert_eq!(reaches[0], RoaringBitmap::from_iter([1, 2, 3]));
/// assert_eq!(reaches[1], RoaringBitmap::from_iter([8, 9, 10]));
/// ```
pub fn reuse_version_reaches(versions: &[FragReuseVersion]) -> Vec<RoaringBitmap> {
    ReuseVersionReachIndex::new(versions.iter()).all_reaches()
}

/// Which versions of a non-stable-row-id v0 fragment reuse history are still
/// required by `indices`, aligned with input positions.
///
/// Seeds are computed per version via [`is_index_remap_caught_up`] against each
/// version's own reach. If an index's coverage splits a rewrite group
/// ([`Error::InvalidInput`]), that version is not seeded on its own (matching
/// legacy v0 seed handling and [`compute_tagged_retention`]). Final retention
/// is the forward closure of all seeds along producer-to-consumer edges
/// (`new_frags` -> `old_frags`) in `dataset_version` order at whole-version
/// granularity: an index committed from a pre-compaction snapshot can still
/// hold raw source fragment IDs whose row addresses must translate through
/// every downstream hop to reach live fragments (so a retained predecessor
/// still retains its downstream hops even if another index straddled a later
/// group), while unrelated later versions remain eligible for cleanup.
///
/// # Example
///
/// ```
/// use lance_table::format::IndexMetadata;
/// use lance_table::system_index::frag_reuse::{
///     FragDigest, FragReuseGroup, FragReuseVersion,
///     retention::compute_v0_retention,
/// };
/// use roaring::RoaringBitmap;
///
/// # fn digest(id: u64) -> FragDigest {
/// #     FragDigest { id, physical_rows: 10, num_deleted_rows: 0 }
/// # }
/// # fn version(dataset_version: u64, old: &[u64], new: &[u64]) -> FragReuseVersion {
/// #     FragReuseVersion {
/// #         dataset_version,
/// #         groups: vec![FragReuseGroup {
/// #             changed_row_addrs: Vec::new(),
/// #             old_frags: old.iter().copied().map(digest).collect(),
/// #             new_frags: new.iter().copied().map(digest).collect(),
/// #         }],
/// #     }
/// # }
/// # fn index_meta(name: &str, dataset_version: u64, frags: &[u32]) -> IndexMetadata {
/// #     IndexMetadata {
/// #         uuid: uuid::Uuid::new_v4(),
/// #         name: name.to_string(),
/// #         fields: vec![0],
/// #         covering_fields: vec![],
/// #         dataset_version,
/// #         fragment_bitmap: Some(RoaringBitmap::from_iter(frags.iter().copied())),
/// #         index_details: None,
/// #         index_version: 0,
/// #         created_at: None,
/// #         base_id: None,
/// #         files: None,
/// #         clustering_id: None,
/// #         metadata: Default::default(),
/// #     }
/// # }
/// let v1 = version(10, &[1], &[2]);
/// let v2 = version(20, &[2], &[3]);
/// let v3 = version(30, &[8], &[9]);
/// // Stale index still holding raw fragment [1] needs both v1 and v2, not disjoint v3.
/// let stale = index_meta("stale", 5, &[1]);
/// assert_eq!(compute_v0_retention(&[v1, v2, v3], &[stale]), vec![true, true, false]);
/// ```
pub fn compute_v0_retention(versions: &[FragReuseVersion], indices: &[IndexMetadata]) -> Vec<bool> {
    if versions.is_empty() {
        return Vec::new();
    }

    let reach_index = ReuseVersionReachIndex::new(versions.iter());
    let mut saw_invalid = vec![false; versions.len()];
    let mut retained = vec![false; versions.len()];

    // Evaluate each index via boolean reach propagation without materializing
    // per-version suffix bitmaps across long chains.
    for index in indices {
        if is_system_index(index) {
            continue;
        }
        let overlaps = index
            .fragment_bitmap
            .as_ref()
            .map(|bitmap| reach_index.versions_overlapping_bitmap(bitmap));
        for (idx, version) in versions.iter().enumerate() {
            if saw_invalid[idx] {
                continue;
            }
            if overlaps.as_ref().is_some_and(|hits| !hits[idx]) {
                continue;
            }
            match index_postdates_rewrite(version, index) {
                Ok(true) => {}
                Ok(false) => retained[idx] = true,
                Err(Error::InvalidInput { .. }) => {
                    saw_invalid[idx] = true;
                    retained[idx] = false;
                }
                Err(_) => unreachable!("index_postdates_rewrite only returns InvalidInput"),
            }
        }
    }

    let mut queue: VecDeque<usize> = retained
        .iter()
        .enumerate()
        .filter_map(|(idx, &keep)| keep.then_some(idx))
        .collect();
    while let Some(idx) = queue.pop_front() {
        let prod_rank = reach_index.rank_of[idx];
        for group in &versions[idx].groups {
            for new_frag in &group.new_frags {
                let Some(consumers) = reach_index.consumers_by_old_frag.get(&new_frag.id) else {
                    continue;
                };
                let start = consumers.partition_point(|&(r, _)| r <= prod_rank);
                for &(consumer_rank, _) in &consumers[start..] {
                    let consumer_idx = reach_index.order[consumer_rank];
                    if !retained[consumer_idx] {
                        retained[consumer_idx] = true;
                        queue.push_back(consumer_idx);
                    }
                }
            }
        }
    }

    retained
}

pub fn is_index_remap_caught_up(
    frag_reuse_version: &FragReuseVersion,
    index_meta: &IndexMetadata,
    version_reach: &RoaringBitmap,
) -> Result<bool> {
    if is_system_index(index_meta) {
        return Ok(true);
    }

    // Disjoint coverage => caught up regardless of dataset_version, bypassing the
    // stale-version gate below. `version_reach` includes this version's NEW
    // fragments and their forward closure through later rewrites deliberately:
    // a deferred-remap commit advances a covering index's bitmap onto new
    // fragments before its data is remapped, so an old-frag-only check would
    // clear a still-stale index and trim a version it needs.
    if let Some(index_frag_bitmap) = &index_meta.fragment_bitmap
        && index_frag_bitmap.is_disjoint(version_reach)
    {
        return Ok(true);
    }

    index_postdates_rewrite(frag_reuse_version, index_meta)
}

/// Built after the rewrite and covering none of its sources. No disjoint
/// shortcut: coverage can be withdrawn while the index data keeps the rows.
pub fn index_postdates_rewrite(
    frag_reuse_version: &FragReuseVersion,
    index_meta: &IndexMetadata,
) -> Result<bool> {
    // Inclusive: an index built at dataset_version predates the rewrite.
    if index_meta.dataset_version <= frag_reuse_version.dataset_version {
        return Ok(false);
    }

    match &index_meta.fragment_bitmap {
        Some(index_frag_bitmap) => {
            for group in frag_reuse_version.groups.iter() {
                let mut old_frag_in_index = 0;
                for old_frag in group.old_frags.iter() {
                    if bitmap_contains(index_frag_bitmap, old_frag.id) {
                        old_frag_in_index += 1;
                    }
                }

                if old_frag_in_index > 0 {
                    if old_frag_in_index != group.old_frags.len() {
                        // This should never happen because we always commit a full rewrite group
                        // and we always reindex either the entire group or nothing.
                        // We use invalid input to be consistent with
                        // dataset::transaction::recalculate_fragment_bitmap
                        return Err(Error::invalid_input(format!(
                            "The compaction plan included a rewrite group that was a split of indexed and non-indexed data: {:?}",
                            group.old_frags
                        )));
                    }
                    return Ok(false);
                }
            }
            Ok(true)
        }
        None => {
            warn!(
                "Index {} ({}) missing fragment bitmap, cannot determine if it is caught up with the fragment reuse version, consider retraining the index",
                index_meta.name, index_meta.uuid
            );
            Ok(true)
        }
    }
}

/// One splice unit of the entry's content: a legacy version (field 1) or a
/// transition (field 2), kept with its exact wire form so retained records
/// are carried forward verbatim.
pub enum TrimRecord {
    LegacyVersion(FragReuseVersion),
    Transition(pb_fri::Transition),
}

pub struct TrimElement {
    pub raw: Vec<u8>,
    pub record: TrimRecord,
}

pub fn split_trim_elements(content: &[u8]) -> Result<Vec<TrimElement>> {
    use bytes::Buf;
    use prost::encoding::{WireType, decode_key, decode_varint};

    let corrupt = |message: String| Error::corrupt_file_named("FRI details", message);
    let total = content.len();
    let mut buf = bytes::Bytes::copy_from_slice(content);
    let mut elements = Vec::new();
    while buf.has_remaining() {
        let start = total - buf.remaining();
        let (tag, wire_type) = decode_key(&mut buf).map_err(|e| corrupt(e.to_string()))?;
        // Fields 1 and 2 are the only known records and both are messages. An
        // unknown record may participate in lineage in ways this client cannot
        // reason about, so refuse to trim rather than silently dropping or
        // blindly retaining it.
        if wire_type != WireType::LengthDelimited || !matches!(tag, 1 | 2) {
            return Err(Error::not_supported(format!(
                "the tagged FRI history carries an unknown record (field {tag}); \
                 upgrade to a newer version of Lance before trimming"
            )));
        }
        let length = decode_varint(&mut buf).map_err(|e| corrupt(e.to_string()))?;
        if length > buf.remaining() as u64 {
            return Err(corrupt(format!(
                "field {tag} length {length} exceeds remaining {} bytes",
                buf.remaining()
            )));
        }
        let payload = buf.split_to(length as usize);
        let end = total - buf.remaining();
        let record = match tag {
            1 => TrimRecord::LegacyVersion(
                pb_fri::Version::decode(payload)
                    .map_err(|e| corrupt(e.to_string()))?
                    .try_into()?,
            ),
            _ => TrimRecord::Transition(
                pb_fri::Transition::decode(payload).map_err(|e| corrupt(e.to_string()))?,
            ),
        };
        elements.push(TrimElement {
            raw: content[start..end].to_vec(),
            record,
        });
    }
    Ok(elements)
}

/// One record's lineage edge as the trim walks it: a transition, or one group
/// of a lifted legacy version.
struct Node {
    element: usize,
    sources: Vec<u64>,
    destinations: Vec<u64>,
}

/// One logical index's coverage as the trim sees it: provenance is the union
/// of every stored segment bitmap (any segment, usable by this build or not,
/// pins the records its addresses need), while direct coverage counts only
/// segments this build can actually serve (`index_is_usable`) -- an unusable
/// segment answers no query, so its bitmap must not mask anyone's need for a
/// translation.
#[derive(Default)]
struct IndexGroupCoverage {
    provenance: RoaringBitmap,
    direct: RoaringBitmap,
}

/// Bitmaps hold u32 ids, so an id outside that range is in none of them.
fn bitmap_contains(bitmap: &RoaringBitmap, fragment: u64) -> bool {
    u32::try_from(fragment).is_ok_and(|id| bitmap.contains(id))
}

/// Every record naming each fragment as a destination. A valid history has
/// one producer per fragment, but only tagged transitions are checked for a
/// duplicate producer (`order_lineage`); lifted legacy groups are not, so
/// every producer is kept and walked.
fn producers_of(nodes: &[Node]) -> HashMap<u64, Vec<usize>> {
    let mut producer_of: HashMap<u64, Vec<usize>> = HashMap::new();
    for (node_index, node) in nodes.iter().enumerate() {
        for destination in &node.destinations {
            producer_of
                .entry(*destination)
                .or_default()
                .push(node_index);
        }
    }
    producer_of
}

/// The producers the walk follows out of `fragment`: none once it is directly
/// covered.
fn upstream_producers<'a>(
    fragment: u64,
    direct: &RoaringBitmap,
    producer_of: &'a HashMap<u64, Vec<usize>>,
) -> &'a [usize] {
    if bitmap_contains(direct, fragment) {
        return &[];
    }
    producer_of
        .get(&fragment)
        .map(Vec::as_slice)
        .unwrap_or_default()
}

/// One node's verdict in the walk of one logical index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    InProgress,
    Pinned,
    NotPinned,
}

struct IndexWalk {
    /// `None` for a node the walk did not reach.
    verdicts: Vec<Option<Verdict>>,
    /// Node resolutions performed, so tests can check the memoisation.
    #[cfg(test)]
    resolutions: usize,
}

/// Which records lie on a translation path one logical index still needs,
/// and whether that index pins them.
///
/// The walk starts at every live fragment that some record produces and the
/// index does not cover directly, visits its producers, and from each visited
/// record continues to the producers of its sources outside direct coverage;
/// it never crosses a directly covered fragment. A visited record is pinned
/// when one of its sources is in the index's provenance, or a source outside
/// direct coverage has a pinned producer. Every such producer is visited even
/// once the record is known pinned, since upstream records need their own
/// verdict.
///
/// This mirrors the reader's `FragmentReuseIndex::segment_plans` (per index,
/// direct coverage first, otherwise back through producers) with two
/// differences: every stored bitmap pins, while only usable segments stop the
/// walk.
///
/// Post-order with an explicit stack so long chains cannot overflow, and
/// memoised so each node is resolved at most once per index. A node reached
/// again while in progress (a cycle, which a valid history cannot hold) counts
/// as pinned, so every node on the cycle resolves pinned and the trim retains.
fn walk_index(
    nodes: &[Node],
    producer_of: &HashMap<u64, Vec<usize>>,
    coverage: &IndexGroupCoverage,
    live_fragments: &RoaringBitmap,
) -> IndexWalk {
    struct Frame {
        node: usize,
        next_source: usize,
        next_producer: usize,
        pinned: bool,
    }
    let open = |node: usize| Frame {
        node,
        next_source: 0,
        next_producer: 0,
        pinned: nodes[node]
            .sources
            .iter()
            .any(|&source| bitmap_contains(&coverage.provenance, source)),
    };

    let mut verdicts: Vec<Option<Verdict>> = vec![None; nodes.len()];
    #[cfg(test)]
    let mut resolutions = 0usize;
    let mut stack: Vec<Frame> = Vec::new();
    for (&fragment, producers) in producer_of {
        if bitmap_contains(&coverage.direct, fragment) || !bitmap_contains(live_fragments, fragment)
        {
            continue;
        }
        for &root in producers {
            if verdicts[root].is_some() {
                continue;
            }
            verdicts[root] = Some(Verdict::InProgress);
            stack.push(open(root));
            while let Some(frame) = stack.last_mut() {
                let sources = &nodes[frame.node].sources;
                let mut descend = None;
                while let Some(&source) = sources.get(frame.next_source) {
                    let producers = upstream_producers(source, &coverage.direct, producer_of);
                    let Some(&producer) = producers.get(frame.next_producer) else {
                        frame.next_source += 1;
                        frame.next_producer = 0;
                        continue;
                    };
                    frame.next_producer += 1;
                    match verdicts[producer] {
                        None => {
                            descend = Some(producer);
                            break;
                        }
                        Some(Verdict::Pinned | Verdict::InProgress) => frame.pinned = true,
                        Some(Verdict::NotPinned) => {}
                    }
                }
                if let Some(child) = descend {
                    verdicts[child] = Some(Verdict::InProgress);
                    stack.push(open(child));
                    continue;
                }
                let (node, pinned) = (frame.node, frame.pinned);
                stack.pop();
                verdicts[node] = Some(if pinned {
                    Verdict::Pinned
                } else {
                    Verdict::NotPinned
                });
                #[cfg(test)]
                {
                    resolutions += 1;
                }
                if pinned && let Some(parent) = stack.last_mut() {
                    parent.pinned = true;
                }
            }
        }
    }
    IndexWalk {
        verdicts,
        #[cfg(test)]
        resolutions,
    }
}

/// Which elements of a tagged history are still needed, from the entry's
/// digests, the stored index metadata and the live fragment set alone (zero
/// payload IO). `live_fragments` must be the fragments of the same manifest
/// `indices` was read from.
///
/// Seeds. Legacy versions keep the exact v0 caught-up predicate. A tagged
/// transition is a seed when, for some logical index (its stored segments
/// grouped by name, system indices exempt), the producer walk of
/// `walk_index` visits it and finds it pinned: every stored segment bitmap
/// of the index pins (a segment without a bitmap imposes nothing, mirroring
/// the v0 leniency), and only the bitmaps of usable segments stop the walk.
/// Only an index's own direct coverage stops its walk: the reader gives direct
/// coverage precedence per logical index, so another index's segments never
/// stand in for its translation. Lifted legacy groups pass pins through the
/// walk but their verdict does not decide a legacy version.
///
/// A transition none of whose destinations leads to a live fragment outside
/// direct coverage is released: a retired destination that no record
/// consumes had its rows deleted, since the trim does not run on a history
/// with unsupported transitions, and the forward closure never releases a
/// consumer before the record producing its sources. A fan-in whose sources
/// are only partly in provenance stays retained, since a later segment over
/// the other sources may complete it.
///
/// Final retention is the forward closure of all seeds: a retained record's
/// translation path walks through every downstream consumer of its
/// destinations, so those records survive too, and legacy versions retain at
/// whole-version granularity (their groups stand or fall together, as in
/// v0). A record can therefore be retained although no walk visited or pinned
/// it. Each index's walk resolves every node at most once.
pub fn compute_tagged_retention(
    elements: &[TrimElement],
    indices: &[IndexMetadata],
    live_fragments: &RoaringBitmap,
    is_usable: impl Fn(&IndexMetadata) -> bool,
) -> Vec<bool> {
    // The v0 predicate's disjointness rule is scoped to each version's reach
    // within the legacy chain: an index touching only tagged-transition
    // fragments or disjoint legacy versions holds nothing this version remaps.
    let legacy_entries: Vec<(usize, &FragReuseVersion)> = elements
        .iter()
        .enumerate()
        .filter_map(|(position, element)| match &element.record {
            TrimRecord::LegacyVersion(version) => Some((position, version)),
            TrimRecord::Transition(_) => None,
        })
        .collect();
    let legacy_reach_index =
        ReuseVersionReachIndex::new(legacy_entries.iter().map(|&(_, version)| version));

    // Stored segments grouped by logical index name. Every segment's bitmap
    // counts as provenance (pinning); only usable segments' bitmaps count as
    // direct coverage (masking).
    let mut index_groups: HashMap<&str, IndexGroupCoverage> = HashMap::new();
    for index in indices.iter() {
        if is_system_index(index) {
            continue;
        }
        let group = index_groups.entry(index.name.as_str()).or_default();
        match &index.fragment_bitmap {
            Some(bitmap) => {
                group.provenance |= bitmap;
                if is_usable(index) {
                    group.direct |= bitmap;
                }
            }
            None => warn!(
                "Index {} ({}) missing fragment bitmap, it cannot pin tagged fragment reuse records, consider retraining the index",
                index.name, index.uuid
            ),
        }
    }

    // Lineage edges, shared by the producer walk and the forward closure.
    let mut nodes = Vec::new();
    for (position, element) in elements.iter().enumerate() {
        match &element.record {
            TrimRecord::LegacyVersion(version) => {
                for group in version.groups.iter() {
                    nodes.push(Node {
                        element: position,
                        sources: group.old_frags.iter().map(|f| f.id).collect(),
                        destinations: group.new_frags.iter().map(|f| f.id).collect(),
                    });
                }
            }
            TrimRecord::Transition(transition) => nodes.push(Node {
                element: position,
                sources: transition.sources.iter().map(|d| d.id).collect(),
                destinations: transition.destinations.iter().map(|d| d.id).collect(),
            }),
        }
    }
    // Relies on each fragment being consumed by at most one record; a repeated
    // source would be last-wins. The ledger rejects duplicate consumers among
    // tagged transitions (`order_lineage`); lifted legacy groups are not
    // checked there.
    let mut consumer_of: HashMap<u64, usize> = HashMap::new();
    let mut element_nodes: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node_index, node) in nodes.iter().enumerate() {
        for source in &node.sources {
            consumer_of.insert(*source, node_index);
        }
        element_nodes
            .entry(node.element)
            .or_default()
            .push(node_index);
    }

    let producer_of = producers_of(&nodes);

    let mut retained = vec![false; elements.len()];
    if !legacy_entries.is_empty() {
        let mut saw_invalid = vec![false; legacy_entries.len()];
        for index in indices {
            if is_system_index(index) {
                continue;
            }
            let overlaps = index
                .fragment_bitmap
                .as_ref()
                .map(|bitmap| legacy_reach_index.versions_overlapping_bitmap(bitmap));
            for (legacy_idx, &(position, version)) in legacy_entries.iter().enumerate() {
                if saw_invalid[legacy_idx] {
                    continue;
                }
                if overlaps.as_ref().is_some_and(|hits| !hits[legacy_idx]) {
                    continue;
                }
                match index_postdates_rewrite(version, index) {
                    Ok(true) => {}
                    Ok(false) => retained[position] = true,
                    Err(Error::InvalidInput { .. }) => {
                        saw_invalid[legacy_idx] = true;
                        retained[position] = false;
                    }
                    Err(_) => unreachable!("index_postdates_rewrite only returns InvalidInput"),
                }
            }
        }
    }
    for coverage in index_groups.values() {
        let walk = walk_index(&nodes, &producer_of, coverage, live_fragments);
        for (node, verdict) in nodes.iter().zip(walk.verdicts) {
            if verdict == Some(Verdict::Pinned)
                && matches!(elements[node.element].record, TrimRecord::Transition(_))
            {
                retained[node.element] = true;
            }
        }
    }

    // Transitive chain retention (forward closure over producer->consumer
    // edges).
    let mut node_retained = vec![false; nodes.len()];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (node_index, node) in nodes.iter().enumerate() {
        if retained[node.element] {
            node_retained[node_index] = true;
            queue.push_back(node_index);
        }
    }
    while let Some(node_index) = queue.pop_front() {
        for destination in nodes[node_index].destinations.clone() {
            let Some(&consumer) = consumer_of.get(&destination) else {
                continue;
            };
            if node_retained[consumer] {
                continue;
            }
            node_retained[consumer] = true;
            queue.push_back(consumer);
            let element = nodes[consumer].element;
            if !retained[element] {
                retained[element] = true;
                for &sibling in element_nodes[&element].iter() {
                    if !node_retained[sibling] {
                        node_retained[sibling] = true;
                        queue.push_back(sibling);
                    }
                }
            }
        }
    }

    retained
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(
        sources: impl IntoIterator<Item = u64>,
        destinations: impl IntoIterator<Item = u64>,
    ) -> Node {
        Node {
            element: 0,
            sources: sources.into_iter().collect(),
            destinations: destinations.into_iter().collect(),
        }
    }

    fn pinning(fragments: &[u32]) -> IndexGroupCoverage {
        IndexGroupCoverage {
            provenance: fragments.iter().copied().collect(),
            direct: RoaringBitmap::new(),
        }
    }

    /// Every destination of a wide transition is a root, yet the transition
    /// is resolved once.
    #[test]
    fn wide_transition_is_resolved_once() {
        const SOURCES: u64 = 4_096;
        const DESTINATIONS: u64 = 2_048;
        let nodes = vec![node(1..=SOURCES, SOURCES + 1..=SOURCES + DESTINATIONS)];
        let live: RoaringBitmap = (SOURCES as u32 + 1..=(SOURCES + DESTINATIONS) as u32).collect();
        let walk = walk_index(&nodes, &producers_of(&nodes), &pinning(&[1]), &live);
        assert_eq!(walk.resolutions, 1);
        assert_eq!(walk.verdicts, vec![Some(Verdict::Pinned)]);
    }

    /// Many roots behind one shared upstream chain: each reachable node is
    /// resolved once, not once per root.
    #[test]
    fn shared_upstream_chain_is_resolved_once_per_node() {
        const HOPS: u64 = 1_000;
        const ROOTS: u64 = 500;
        // Chain `k -> k + 1` for k in 1..=HOPS, a split of HOPS + 1 into
        // ROOTS fragments, and one hop from each of them onto a live root.
        let mut nodes: Vec<Node> = (1..=HOPS).map(|hop| node([hop], [hop + 1])).collect();
        let split_base = HOPS + 2;
        nodes.push(node([HOPS + 1], split_base..split_base + ROOTS));
        let root_base = split_base + ROOTS;
        nodes.extend((0..ROOTS).map(|i| node([split_base + i], [root_base + i])));
        let live: RoaringBitmap = (root_base as u32..(root_base + ROOTS) as u32).collect();

        let walk = walk_index(&nodes, &producers_of(&nodes), &pinning(&[1]), &live);
        assert_eq!(walk.resolutions, nodes.len());
        assert!(
            walk.verdicts
                .iter()
                .all(|verdict| *verdict == Some(Verdict::Pinned))
        );
    }

    /// A cycle (which a valid history cannot hold) resolves every node on it
    /// pinned, even without any provenance.
    #[test]
    fn cycle_resolves_pinned() {
        let nodes = vec![node([2], [1]), node([1], [2]), node([1], [3])];
        let live: RoaringBitmap = [3u32].into_iter().collect();
        let walk = walk_index(&nodes, &producers_of(&nodes), &pinning(&[]), &live);
        assert_eq!(walk.resolutions, 3);
        assert!(
            walk.verdicts
                .iter()
                .all(|verdict| *verdict == Some(Verdict::Pinned))
        );
    }

    fn frag_digest(id: u64) -> super::super::FragDigest {
        super::super::FragDigest {
            id,
            physical_rows: 100,
            num_deleted_rows: 0,
        }
    }

    fn reuse_group(old: &[u64], new: &[u64]) -> super::super::FragReuseGroup {
        super::super::FragReuseGroup {
            changed_row_addrs: Vec::new(),
            old_frags: old.iter().copied().map(frag_digest).collect(),
            new_frags: new.iter().copied().map(frag_digest).collect(),
        }
    }

    fn reuse_version(dataset_version: u64, old: &[u64], new: &[u64]) -> FragReuseVersion {
        FragReuseVersion {
            dataset_version,
            groups: vec![reuse_group(old, new)],
        }
    }

    fn index_meta(name: &str, dataset_version: u64, frags: &[u32]) -> IndexMetadata {
        IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            name: name.to_string(),
            fields: vec![0],
            covering_fields: vec![],
            dataset_version,
            fragment_bitmap: Some(frags.iter().copied().collect()),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        }
    }

    #[test]
    fn test_caught_up_is_judged_per_version() {
        let v1 = reuse_version(10, &[4, 5], &[6]);
        let v2 = reuse_version(20, &[8, 9], &[10]);
        let reaches = reuse_version_reaches(&[v1.clone(), v2.clone()]);

        assert_eq!(reaches[0], RoaringBitmap::from_iter([4, 5, 6]));
        assert_eq!(reaches[1], RoaringBitmap::from_iter([8, 9, 10]));

        // Rebuilt after v1 over v1's output [6]: caught up with v1 AND v2
        // despite dataset_version 15 <= 20, because v2's reach is disjoint.
        let idx_a = index_meta("a", 15, &[6]);
        assert!(is_index_remap_caught_up(&v1, &idx_a, &reaches[0]).unwrap());
        assert!(is_index_remap_caught_up(&v2, &idx_a, &reaches[1]).unwrap());

        // Stale index still behind v1 does not pin unrelated v2.
        let idx_f = index_meta("f", 5, &[6]);
        assert!(!is_index_remap_caught_up(&v1, &idx_f, &reaches[0]).unwrap());
        assert!(is_index_remap_caught_up(&v2, &idx_f, &reaches[1]).unwrap());

        // Index covering v2's output pins v2, not v1.
        let idx_c = index_meta("c", 15, &[10]);
        assert!(is_index_remap_caught_up(&v1, &idx_c, &reaches[0]).unwrap());
        assert!(!is_index_remap_caught_up(&v2, &idx_c, &reaches[1]).unwrap());
    }

    #[test]
    fn test_reuse_version_reaches_follows_group_closure_and_order() {
        let v1 = reuse_version(10, &[1, 2], &[3]);
        // Two groups in v2: only [3, 4] -> [5] connects to v1's output [3].
        let v2 = FragReuseVersion {
            dataset_version: 20,
            groups: vec![reuse_group(&[3, 4], &[5]), reuse_group(&[10, 11], &[12])],
        };
        let v3 = reuse_version(30, &[5], &[6]);

        // Unsorted input order must still follow dataset_version order and
        // return reaches aligned with input indices.
        let reaches = reuse_version_reaches(&[v3.clone(), v1.clone(), v2.clone()]);
        assert_eq!(reaches[0], RoaringBitmap::from_iter([5, 6]));
        assert_eq!(reaches[1], RoaringBitmap::from_iter([1, 2, 3, 5, 6]));
        assert_eq!(
            reaches[2],
            RoaringBitmap::from_iter([3, 4, 5, 6, 10, 11, 12])
        );

        // Single-version helper matches even when passed unsorted later versions.
        assert_eq!(reuse_version_reach(&v1, &[v3, v2.clone()]), reaches[1]);

        // Stale index advanced along v1 -> v2 -> v3 onto [6] is still behind v1.
        let stale = index_meta("stale", 5, &[6]);
        assert!(!is_index_remap_caught_up(&v1, &stale, &reaches[1]).unwrap());

        // Stale index on unrelated group's output [12] is disjoint from v1's reach.
        let unrelated = index_meta("unrelated", 5, &[12]);
        assert!(is_index_remap_caught_up(&v1, &unrelated, &reaches[1]).unwrap());
        assert!(!is_index_remap_caught_up(&v2, &unrelated, &reaches[2]).unwrap());
    }

    #[test]
    fn test_tagged_retention_legacy_branch_uses_per_version_reach() {
        let v1 = reuse_version(10, &[4, 5], &[6]);
        let v2 = reuse_version(20, &[8, 9], &[10]);
        let elements = vec![
            TrimElement {
                raw: Vec::new(),
                record: TrimRecord::LegacyVersion(v1),
            },
            TrimElement {
                raw: Vec::new(),
                record: TrimRecord::LegacyVersion(v2),
            },
        ];
        // `f` still pins v1; `a` was rebuilt after v1 over [6]. Neither touches v2.
        let indices = vec![index_meta("a", 15, &[6]), index_meta("f", 5, &[6])];
        let live: RoaringBitmap = [6, 10].into_iter().collect();

        let retained = compute_tagged_retention(&elements, &indices, &live, |_| true);
        assert_eq!(retained, vec![true, false]);
    }

    #[rstest::rstest]
    #[case::same_stamp_preserves_input_order(
        vec![reuse_version(10, &[1], &[2]), reuse_version(10, &[2], &[3])],
        vec![vec![1, 2, 3], vec![2, 3]],
    )]
    #[case::same_stamp_does_not_flow_backwards(
        vec![reuse_version(10, &[2], &[3]), reuse_version(10, &[1], &[2])],
        vec![vec![2, 3], vec![1, 2]],
    )]
    #[case::whole_fragment_removal_does_not_propagate(
        vec![reuse_version(10, &[7], &[]), reuse_version(20, &[7], &[8])],
        vec![vec![7], vec![7, 8]],
    )]
    #[case::u32_max_and_overflow_id_do_not_alias(
        vec![
            reuse_version(10, &[u64::from(u32::MAX)], &[u64::from(u32::MAX) + 1]),
            reuse_version(20, &[0], &[99]),
        ],
        vec![vec![u32::MAX], vec![0, 99]],
    )]
    fn test_reuse_version_reaches_boundaries(
        #[case] versions: Vec<FragReuseVersion>,
        #[case] expected: Vec<Vec<u32>>,
    ) {
        let reaches = reuse_version_reaches(&versions);
        let expected: Vec<RoaringBitmap> = expected
            .into_iter()
            .map(|ids| ids.into_iter().collect())
            .collect();
        assert_eq!(reaches, expected);
    }

    #[test]
    fn test_compute_v0_retention_preserves_downstream_hops_and_drops_disjoint() {
        let v1 = reuse_version(10, &[1, 2], &[3]);
        let v2 = reuse_version(20, &[3], &[4]);
        let v3 = reuse_version(30, &[8, 9], &[10]);

        // Index committed from a pre-v1 snapshot still holding raw {1, 2}:
        // v1 is seeded directly, v2 is retained by forward closure, disjoint v3 is dropped.
        let stale_raw = index_meta("stale_raw", 5, &[1, 2]);
        assert_eq!(
            compute_v0_retention(&[v1.clone(), v2.clone(), v3.clone()], &[stale_raw]),
            vec![true, true, false]
        );

        // Index rebuilt after v1 over [3] needs v2 (stamp 20 >= 15) but neither v1 nor disjoint v3.
        let rebuilt_after_v1 = index_meta("rebuilt", 15, &[3]);
        assert_eq!(
            compute_v0_retention(&[v1.clone(), v2.clone(), v3.clone()], &[rebuilt_after_v1]),
            vec![false, true, false]
        );

        // Unsorted input order still propagates forward in dataset_version order and
        // returns retention aligned with input positions.
        let stale_raw_unsorted = index_meta("stale_raw", 5, &[1, 2]);
        assert_eq!(
            compute_v0_retention(&[v3, v2, v1], &[stale_raw_unsorted]),
            vec![false, true, true]
        );
    }

    #[test]
    fn test_reach_index_avoids_quadratic_work_on_disjoint_and_chained_histories() {
        // 2,000 disjoint single-group versions: no reach query should visit any other group.
        const DISJOINT: usize = 2_000;
        let disjoint: Vec<FragReuseVersion> = (0..DISJOINT)
            .map(|i| {
                let base = (i as u64) * 2;
                reuse_version(i as u64 + 1, &[base], &[base + 1])
            })
            .collect();
        let disjoint_index = ReuseVersionReachIndex::new(disjoint.iter());
        let mut total_visited = 0usize;
        for i in 0..DISJOINT {
            let (reach, visited) = disjoint_index.reach_with_work(i);
            total_visited += visited;
            let base = (i as u32) * 2;
            assert_eq!(reach, RoaringBitmap::from_iter([base, base + 1]));
        }
        assert_eq!(total_visited, 0);

        // A 500-hop linear chain alongside 1,000 unrelated disjoint versions:
        // the chain head visits only its 499 downstream chain groups, and a
        // full-history `all_reaches` pass visits every group exactly once
        // across all 1,500 versions (linear rather than quadratic on the chain).
        const HOPS: usize = 500;
        let mut mixed: Vec<FragReuseVersion> = (0..HOPS)
            .map(|i| reuse_version(i as u64 + 1, &[i as u64], &[i as u64 + 1]))
            .collect();
        for i in 0..1_000usize {
            let base = 100_000 + (i as u64) * 2;
            mixed.push(reuse_version((HOPS + i + 1) as u64, &[base], &[base + 1]));
        }
        let mixed_index = ReuseVersionReachIndex::new(mixed.iter());
        let (head_reach, head_visited) = mixed_index.reach_with_work(0);
        assert_eq!(head_visited, HOPS - 1);
        assert_eq!(head_reach, RoaringBitmap::from_iter(0..=(HOPS as u32)));

        let (all_reaches, all_visited, all_edges) = mixed_index.all_reaches_with_work();
        assert_eq!(all_visited, mixed.len());
        assert_eq!(all_edges, HOPS - 1);
        assert_eq!(all_reaches[0], RoaringBitmap::from_iter(0..=(HOPS as u32)));
        assert_eq!(
            all_reaches[HOPS - 1],
            RoaringBitmap::from_iter([(HOPS - 1) as u32, HOPS as u32])
        );
        assert_eq!(
            all_reaches[HOPS],
            RoaringBitmap::from_iter([100_000, 100_001])
        );

        // Multi-container chain (every hop in its own Roaring container):
        // boolean reach propagation evaluates every version in O(V + E)
        // without materializing per-version suffix bitmaps.
        const MULTI_HOPS: usize = 500;
        let multi_chain: Vec<FragReuseVersion> = (0..MULTI_HOPS)
            .map(|i| {
                let src = ((i as u64) + 1) * 100_000;
                let dst = ((i as u64) + 2) * 100_000;
                reuse_version(i as u64 + 1, &[src], &[dst])
            })
            .collect();
        let multi_index = ReuseVersionReachIndex::new(multi_chain.iter());
        let leaf_bitmap = RoaringBitmap::from_iter([((MULTI_HOPS as u32) + 1) * 100_000]);
        let (overlaps, prop_groups, prop_edges) =
            multi_index.versions_overlapping_bitmap_with_work(&leaf_bitmap);
        assert_eq!(prop_groups, MULTI_HOPS);
        assert_eq!(prop_edges, MULTI_HOPS - 1);
        assert!(overlaps.iter().all(|&hit| hit));
        let stale_leaf = index_meta("stale_leaf", 0, &[((MULTI_HOPS as u32) + 1) * 100_000]);
        assert_eq!(
            compute_v0_retention(&multi_chain, &[stale_leaf]),
            vec![true; MULTI_HOPS]
        );
    }

    #[test]
    fn test_overflow_old_frag_id_does_not_alias_in_postdate_check() {
        // Output fragment 0 prevents the disjoint shortcut from firing for an
        // index covering {0}; the out-of-range old_frag `u32::MAX + 1` must
        // not truncate to 0 and trigger a spurious partial-group InvalidInput.
        let version = reuse_version(10, &[u64::from(u32::MAX) + 1, 7], &[0]);
        let reach = reuse_version_reach(&version, &[]);
        let post_index = index_meta("post", 11, &[0]);

        assert!(is_index_remap_caught_up(&version, &post_index, &reach).unwrap());
        assert_eq!(compute_v0_retention(&[version], &[post_index]), vec![false]);
    }

    #[test]
    fn test_compute_v0_retention_straddled_downstream_stays_when_predecessor_pinned() {
        let v1 = reuse_version(10, &[1], &[2]);
        let v2 = reuse_version(20, &[2, 3], &[4]);
        // Post-v2 index straddles v2's old_frags [2, 3] (covers only [2]),
        // causing `is_index_remap_caught_up` on v2 to return `InvalidInput`.
        let straddling = index_meta("straddling", 25, &[2]);

        // Without a pinned predecessor, v2 is not seeded on its own.
        assert_eq!(
            compute_v0_retention(&[v1.clone(), v2.clone()], std::slice::from_ref(&straddling)),
            vec![false, false]
        );

        // When a stale index pins v1 on raw [1], forward closure retains v2 so
        // v1's mapping chain is not truncated at retired fragment [2].
        let stale_raw = index_meta("stale_raw", 5, &[1]);
        assert_eq!(
            compute_v0_retention(&[v1, v2], &[straddling, stale_raw]),
            vec![true, true]
        );
    }
}
