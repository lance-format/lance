// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compact row-address remapping for compaction.
//!
//! Compaction rewrites rows into new fragments, so indices that store physical
//! row addresses need an old-address to new-address mapping without building an
//! O(total rows) `HashMap<u64, Option<u64>>`.
//!
//! Layout:
//!
//! * Old rows: `old_fragment_id -> (old_offsets, old_rows_before)`
//!     * `old_offsets`: rewritten old row offsets in this old fragment.
//!     * `old_rows_before`: rewritten row count before this old fragment.
//! * New rows: ordered new-fragment ranges
//!   `(fragment_id, new_rows_before, physical_rows)`
//!     * `new_rows_before`: rewritten row count before this new fragment.
//!
//! Lookup:
//!
//! * An address whose fragment was not rewritten returns `None`.
//! * For an address whose fragment was rewritten:
//!     * Read `(old_offsets, old_rows_before)` from the old-row layout.
//!     * If `offset` is outside the old fragment's physical row range, return
//!       `None`; the direct-map representation would not contain that address.
//!     * If a valid `offset` is not in `old_offsets`, return `Some(None)`
//!       because the row was deleted.
//!     * Otherwise, `old_offsets.rank(offset) - 1` is this row's 0-based
//!       position among rewritten old rows in this old fragment. Add
//!       `old_rows_before` to get `k`, the row's 0-based position among all
//!       rewritten old rows.
//!     * In the new-row layout, find the range
//!       `(fragment_id, new_rows_before, physical_rows)` where
//!       `new_rows_before <= k < new_rows_before + physical_rows`.
//!     * The new address is `(fragment_id, k - new_rows_before)`.
//!
//! Ordering:
//!
//! Compact remap does not store each old-to-new row mapping. It computes `k`
//! from the old-row layout, then maps it to the k-th row written to the new
//! fragments. This requires the reader-to-writer pipeline to preserve row order.
//!
//! * `old_frag_ids` must match the order old fragments are read. Within each
//!   old fragment, rewritten rows are interpreted by ascending old row offset.
//! * `new_frags` must match the order new rows are written.
//! * Current compaction satisfies this because it scans selected fragments in
//!   order and writes the resulting stream without reordering rows.

use crate::deepsize::{Context, DeepSizeOf};
use crate::utils::address::RowAddress;
use crate::{Error, Result};
use roaring::{RoaringBitmap, RoaringTreemap};
use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::Arc;

/// A queryable row-address remapping with the exact semantics of
/// `HashMap<u64, Option<u64>>::get(&addr).copied()`:
///
/// * `None` — the address is not affected by this remap (keep it unchanged)
/// * `Some(None)` — the row was deleted
/// * `Some(Some(addr))` — the row moved to `addr`
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowAddrRemap {
    /// Compact, `O(#fragments)` remap built from per-group rewritten-row
    /// bitmaps and new-fragment layouts.
    Compact(CompactRowAddrRemap),
    /// Full materialized old-to-new address map. Uses `O(#rows)` memory.
    Direct(HashMap<u64, Option<u64>>),
}

impl RowAddrRemap {
    pub fn compact(groups: impl IntoIterator<Item = GroupInput>) -> Result<Self> {
        Ok(Self::Compact(CompactRowAddrRemap::new(groups)?))
    }

    /// Build a compact remap with physical row counts for exact validation of
    /// addresses loaded from persisted fragment layouts.
    #[doc(hidden)]
    pub fn compact_with_layout(
        groups: impl IntoIterator<Item = GroupInputWithLayout>,
    ) -> Result<Self> {
        Ok(Self::Compact(CompactRowAddrRemap::new_with_layout(groups)?))
    }

    /// Build a remap from a fully materialized old-to-new address map.
    pub fn direct(map: HashMap<u64, Option<u64>>) -> Self {
        Self::Direct(map)
    }

    /// Build a compact view that deletes addresses in `removed_fragments`
    /// before lazily applying `mapping` to every other address.
    #[doc(hidden)]
    pub fn with_removed_fragments(mapping: Arc<Self>, removed_fragments: RoaringBitmap) -> Self {
        Self::Compact(CompactRowAddrRemap::with_removed_fragments(
            mapping,
            removed_fragments,
        ))
    }

    /// Build an ordered remap chain, flattening nested chains and omitting
    /// empty remaps.
    pub fn chained(remaps: impl IntoIterator<Item = Self>) -> Self {
        let mut remaps = remaps
            .into_iter()
            .filter(|remap| !remap.is_empty())
            .collect::<Vec<_>>();
        match remaps.len() {
            0 => Self::empty(),
            1 => remaps.pop().unwrap(),
            _ => Self::Compact(CompactRowAddrRemap::chained(remaps)),
        }
    }

    /// An empty remap that leaves every address unchanged.
    pub fn empty() -> Self {
        Self::Direct(HashMap::new())
    }

    /// Look up `addr`. See [`RowAddrRemap`] for the tri-state return semantics.
    #[inline]
    pub fn get(&self, addr: u64) -> Option<Option<u64>> {
        match self {
            Self::Compact(c) => c.get(addr),
            Self::Direct(m) => m.get(&addr).copied(),
        }
    }

    /// Apply this remap to a batch in place.
    ///
    /// A `None` input remains deleted. An address missing from a remap remains
    /// unchanged. Chained remaps are applied version-by-version so this path is
    /// suitable for bulk index and transaction remapping without materializing
    /// a composed per-row map.
    pub fn remap_in_place(&self, row_addrs: &mut [Option<u64>]) {
        match self {
            Self::Compact(compact) => compact.remap_in_place(row_addrs),
            Self::Direct(_) => {
                for row_addr in row_addrs {
                    if let Some(addr) = *row_addr
                        && let Some(mapped) = self.get(addr)
                    {
                        *row_addr = mapped;
                    }
                }
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Compact(c) => c.is_empty(),
            Self::Direct(m) => m.is_empty(),
        }
    }

    pub fn affected_fragments(&self) -> RoaringBitmap {
        match self {
            Self::Compact(c) => c.affected_fragments(),
            Self::Direct(m) => RoaringBitmap::from_iter(m.keys().map(|addr| (addr >> 32) as u32)),
        }
    }

    pub fn fully_deleted_fragments(&self) -> Option<RoaringBitmap> {
        match self {
            Self::Compact(c) => c.fully_deleted_fragments(),
            Self::Direct(m) => {
                if m.values().all(|v| v.is_none()) {
                    Some(RoaringBitmap::from_iter(
                        m.keys().map(|addr| (addr >> 32) as u32),
                    ))
                } else {
                    None
                }
            }
        }
    }
}

impl DeepSizeOf for RowAddrRemap {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            Self::Compact(compact) => compact.deep_size_of_children(context),
            Self::Direct(map) => map.deep_size_of_children(context),
        }
    }
}

/// Input describing one rewrite group: the old row addresses that were
/// rewritten plus the fragment layout before/after the rewrite.
pub struct GroupInput {
    /// Old row addresses that were read and re-written into the new fragments.
    pub rewritten_old_row_addrs: RoaringTreemap,
    /// Old fragment ids covered by this group.
    pub old_frag_ids: Vec<u32>,
    /// New fragments produced by this group, as `(fragment_id, physical_rows)`,
    pub new_frags: Vec<(u32, u32)>,
}

/// Internal compact-remap input that includes old-fragment physical row counts.
#[doc(hidden)]
pub struct GroupInputWithLayout {
    pub rewritten_old_row_addrs: RoaringTreemap,
    pub old_frags: Vec<(u32, u32)>,
    pub new_frags: Vec<(u32, u32)>,
}

/// Keep Roaring only when its serialized representation is substantially
/// smaller than either rank-friendly representation. This preserves compact
/// run containers while avoiding Roaring's linear word scan for dense rank.
/// Binary-copy compaction creates these runs with `RoaringTreemap::insert_range`,
/// and serialization preserves them without an explicit `optimize()` call.
const ROARING_SIZE_ADVANTAGE_FOR_RANK: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
enum RankedOffsets {
    /// Retained for highly compressible run layouts.
    Roaring(RoaringBitmap),
    /// Sorted rewritten offsets. Binary search returns membership and rank in
    /// one operation.
    Sparse(Vec<u32>),
    /// Dense bits with the number of rewritten rows before every word.
    Dense(DenseRankedOffsets),
}

impl RankedOffsets {
    fn try_new(offsets: RoaringBitmap, physical_rows: Option<u32>) -> Result<Self> {
        let universe_rows = physical_rows.map(u64::from).unwrap_or_else(|| {
            offsets
                .max()
                .map(|offset| u64::from(offset) + 1)
                .unwrap_or(0)
        });
        let word_count = usize::try_from(universe_rows.div_ceil(64)).map_err(|_| {
            Error::invalid_input(format!(
                "fragment row range {universe_rows} is too large for compact rank lookup"
            ))
        })?;
        let sparse_bytes = usize::try_from(offsets.len())
            .ok()
            .and_then(|len| len.checked_mul(size_of::<u32>()))
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "rewritten row count {} is too large for sparse rank lookup",
                    offsets.len()
                ))
            })?;
        let dense_bytes = word_count
            .checked_mul(size_of::<u64>() + size_of::<u32>())
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "fragment row range {universe_rows} is too large for dense rank lookup"
                ))
            })?;
        let rank_friendly_bytes = sparse_bytes.min(dense_bytes);
        if offsets
            .serialized_size()
            .checked_mul(ROARING_SIZE_ADVANTAGE_FOR_RANK)
            .is_some_and(|roaring_bytes| roaring_bytes < rank_friendly_bytes)
        {
            return Ok(Self::Roaring(offsets));
        }
        if sparse_bytes <= dense_bytes {
            return Ok(Self::Sparse(offsets.into_iter().collect()));
        }
        Ok(Self::Dense(DenseRankedOffsets::try_new(
            offsets, word_count,
        )?))
    }

    /// Return the zero-based rank when `offset` was rewritten.
    #[inline]
    fn rank_if_present(&self, offset: u32) -> Option<u64> {
        match self {
            Self::Roaring(offsets) => offsets.contains(offset).then(|| offsets.rank(offset) - 1),
            Self::Sparse(offsets) => offsets.binary_search(&offset).ok().map(|rank| rank as u64),
            Self::Dense(offsets) => offsets.rank_if_present(offset),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Roaring(offsets) => offsets.is_empty(),
            Self::Sparse(offsets) => offsets.is_empty(),
            Self::Dense(offsets) => offsets.words.is_empty(),
        }
    }
}

impl DeepSizeOf for RankedOffsets {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            // Roaring does not expose its allocation capacity. Its serialized
            // size is a stable proxy for the retained containers.
            Self::Roaring(offsets) => offsets.serialized_size(),
            Self::Sparse(offsets) => offsets.deep_size_of_children(context),
            Self::Dense(offsets) => offsets.deep_size_of_children(context),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DenseRankedOffsets {
    words: Vec<u64>,
    rank_before_word: Vec<u32>,
}

impl DenseRankedOffsets {
    fn try_new(offsets: RoaringBitmap, word_count: usize) -> Result<Self> {
        let mut words = vec![0u64; word_count];
        for offset in offsets {
            let word_idx = (offset / 64) as usize;
            let Some(word) = words.get_mut(word_idx) else {
                return Err(Error::invalid_input(format!(
                    "rewritten row offset {offset} is outside dense rank word_count={word_count}"
                )));
            };
            *word |= 1u64 << (offset % 64);
        }

        let mut rank_before_word = Vec::with_capacity(word_count);
        let mut rewritten_rows_before = 0u64;
        for word in &words {
            rank_before_word.push(u32::try_from(rewritten_rows_before).map_err(|_| {
                Error::invalid_input(format!(
                    "rewritten row count {rewritten_rows_before} exceeds the row-address offset range"
                ))
            })?);
            rewritten_rows_before += u64::from(word.count_ones());
        }
        Ok(Self {
            words,
            rank_before_word,
        })
    }

    #[inline]
    fn rank_if_present(&self, offset: u32) -> Option<u64> {
        let word_idx = (offset / 64) as usize;
        let word = *self.words.get(word_idx)?;
        let bit = 1u64 << (offset % 64);
        if word & bit == 0 {
            return None;
        }
        Some(
            u64::from(self.rank_before_word[word_idx]) + u64::from((word & (bit - 1)).count_ones()),
        )
    }
}

impl DeepSizeOf for DenseRankedOffsets {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.words.deep_size_of_children(context)
            + self.rank_before_word.deep_size_of_children(context)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OldFragmentRemap {
    group_idx: usize,
    rewritten_offsets: RankedOffsets,
    rewritten_rows_before: u64,
    physical_rows: Option<u32>,
}

impl DeepSizeOf for OldFragmentRemap {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.rewritten_offsets.deep_size_of_children(context)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GroupRemap {
    /// New fragment ranges as `(fragment_id, rewritten_rows_before, physical_rows)`,
    /// used to map a rewritten row's group-local index to its new address via binary search.
    new_frag_row_ranges: Vec<(u32, u64, u32)>,
}

impl GroupRemap {
    fn new(input: GroupInput, group_idx: usize) -> Result<(Self, Vec<(u32, OldFragmentRemap)>)> {
        Self::new_with_old_frags(
            input.rewritten_old_row_addrs,
            input.old_frag_ids.into_iter().map(|id| (id, None)),
            input.new_frags,
            group_idx,
        )
    }

    fn new_with_layout(
        input: GroupInputWithLayout,
        group_idx: usize,
    ) -> Result<(Self, Vec<(u32, OldFragmentRemap)>)> {
        Self::new_with_old_frags(
            input.rewritten_old_row_addrs,
            input
                .old_frags
                .into_iter()
                .map(|(id, rows)| (id, Some(rows))),
            input.new_frags,
            group_idx,
        )
    }

    fn new_with_old_frags(
        rewritten_old_row_addrs: RoaringTreemap,
        old_frags: impl IntoIterator<Item = (u32, Option<u32>)>,
        new_frags: Vec<(u32, u32)>,
        group_idx: usize,
    ) -> Result<(Self, Vec<(u32, OldFragmentRemap)>)> {
        // `compute_new_addr` maps a rewritten row's group-local index by
        // accumulating `physical_rows` in the caller-provided write order.
        let mut new_frag_row_ranges = Vec::with_capacity(new_frags.len());
        let mut rewritten_rows_before = 0u64;
        for (frag_id, physical_rows) in new_frags {
            if physical_rows == 0 {
                continue;
            }
            new_frag_row_ranges.push((frag_id, rewritten_rows_before, physical_rows));
            rewritten_rows_before += physical_rows as u64;
        }
        let total_new_rows = rewritten_rows_before;

        let mut per_frag: HashMap<u32, RoaringBitmap> = rewritten_old_row_addrs
            .bitmaps()
            .map(|(frag_id, bitmap)| (frag_id, bitmap.clone()))
            .collect();
        let old_frags = old_frags.into_iter().collect::<Vec<_>>();
        let mut frags = Vec::with_capacity(old_frags.len());
        let mut seen_frag_ids = HashSet::with_capacity(old_frags.len());
        let mut rewritten_rows_before = 0u64;
        for &(frag_id, physical_rows) in &old_frags {
            if !seen_frag_ids.insert(frag_id) {
                return Err(Error::invalid_input(format!(
                    "rewrite group {group_idx} contains old fragment {frag_id} more than once"
                )));
            }
            let bitmap = per_frag.remove(&frag_id).unwrap_or_default();
            if let Some(physical_rows) = physical_rows
                && bitmap.max().is_some_and(|offset| offset >= physical_rows)
            {
                return Err(Error::invalid_input(format!(
                    "rewrite group {group_idx} contains a row offset outside old fragment {frag_id} with physical_rows={physical_rows}"
                )));
            }
            let num_rewritten_rows = bitmap.len();
            let rewritten_offsets = RankedOffsets::try_new(bitmap, physical_rows)?;
            frags.push((
                frag_id,
                OldFragmentRemap {
                    group_idx,
                    rewritten_offsets,
                    rewritten_rows_before,
                    physical_rows,
                },
            ));
            rewritten_rows_before += num_rewritten_rows;
        }
        // Rewritten old row addresses must reference only listed old fragments.
        if !per_frag.is_empty() {
            return Err(Error::invalid_input(format!(
                "compaction rewrite group {group_idx} references rewritten old row addresses from fragments {:?} not in its old fragments {:?}",
                per_frag.keys().collect::<Vec<_>>(),
                old_frags,
            )));
        }

        // Rewritten old rows are mapped positionally onto the new rows, so the
        // two counts must match exactly
        let total_rewritten_old_rows = rewritten_old_row_addrs.len();
        if total_new_rows != total_rewritten_old_rows {
            return Err(Error::invalid_input(format!(
                "compaction rewrite group {group_idx} rewrote {total_rewritten_old_rows} old rows from fragments {:?} but the new fragments hold {total_new_rows} rows",
                old_frags,
            )));
        }

        Ok((
            Self {
                new_frag_row_ranges,
            },
            frags,
        ))
    }

    fn compute_new_addr(&self, rewritten_row_index: u64) -> u64 {
        let idx =
            match self
                .new_frag_row_ranges
                .binary_search_by(|(_, rewritten_rows_before, _)| {
                    rewritten_rows_before.cmp(&rewritten_row_index)
                }) {
                Ok(i) => i,
                Err(i) => i - 1,
            };
        let (frag_id, rewritten_rows_before, _rows) = self.new_frag_row_ranges[idx];
        let offset = (rewritten_row_index - rewritten_rows_before) as u32;
        u64::from(RowAddress::new_from_parts(frag_id, offset))
    }
}

impl DeepSizeOf for GroupRemap {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.new_frag_row_ranges.deep_size_of_children(context)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompactRemapStep {
    groups: Vec<GroupRemap>,
    /// Old fragment id -> its bitmap/rank layout and rewrite group. Size is
    /// O(#fragments), not rows.
    frags: HashMap<u32, OldFragmentRemap>,
}

impl CompactRemapStep {
    fn new(groups: impl IntoIterator<Item = GroupInput>) -> Result<Self> {
        let mut frags = HashMap::new();
        let mut group_remaps = Vec::new();
        for input in groups {
            let gi = group_remaps.len();
            let (group_remap, group_frags) = GroupRemap::new(input, gi)?;
            for (frag_id, frag) in group_frags {
                if frags.insert(frag_id, frag).is_some() {
                    return Err(Error::invalid_input(format!(
                        "old fragment {frag_id} appears in more than one rewrite group, including group {gi}"
                    )));
                }
            }
            group_remaps.push(group_remap);
        }
        Ok(Self {
            groups: group_remaps,
            frags,
        })
    }

    fn new_with_layout(groups: impl IntoIterator<Item = GroupInputWithLayout>) -> Result<Self> {
        let mut frags = HashMap::new();
        let mut group_remaps = Vec::new();
        for input in groups {
            let gi = group_remaps.len();
            let (group_remap, group_frags) = GroupRemap::new_with_layout(input, gi)?;
            for (frag_id, frag) in group_frags {
                if frags.insert(frag_id, frag).is_some() {
                    return Err(Error::invalid_input(format!(
                        "old fragment {frag_id} appears in more than one rewrite group, including group {gi}"
                    )));
                }
            }
            group_remaps.push(group_remap);
        }
        Ok(Self {
            groups: group_remaps,
            frags,
        })
    }

    #[inline]
    pub fn get(&self, addr: u64) -> Option<Option<u64>> {
        let frag = (addr >> 32) as u32;
        // Not in any rewrite group -> unaffected by this remap.
        let old_frag = self.frags.get(&frag)?;
        let offset = addr as u32;
        if old_frag
            .physical_rows
            .is_some_and(|physical_rows| offset >= physical_rows)
        {
            return None;
        }
        let Some(rewritten_rank) = old_frag.rewritten_offsets.rank_if_present(offset) else {
            return Some(None);
        };
        let rewritten_row_index = old_frag.rewritten_rows_before + rewritten_rank;
        Some(Some(
            self.groups[old_frag.group_idx].compute_new_addr(rewritten_row_index),
        ))
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    fn fully_deleted_fragments(&self) -> Option<RoaringBitmap> {
        // A group with any rewritten row moved at least one row.
        if self
            .frags
            .values()
            .any(|frag| !frag.rewritten_offsets.is_empty())
        {
            return None;
        }
        Some(RoaringBitmap::from_iter(self.frags.keys().copied()))
    }

    fn affected_fragments(&self) -> RoaringBitmap {
        RoaringBitmap::from_iter(self.frags.keys().copied())
    }
}

impl DeepSizeOf for CompactRemapStep {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.groups.deep_size_of_children(context) + self.frags.deep_size_of_children(context)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RemapStep {
    Compact(CompactRemapStep),
    Direct(HashMap<u64, Option<u64>>),
    RemoveFragments(RoaringBitmap),
    Shared(Arc<RowAddrRemap>),
}

impl RemapStep {
    fn get(&self, addr: u64) -> Option<Option<u64>> {
        match self {
            Self::Compact(compact) => compact.get(addr),
            Self::Direct(direct) => direct.get(&addr).copied(),
            Self::RemoveFragments(fragments) => fragments
                .contains(RowAddress::from(addr).fragment_id())
                .then_some(None),
            Self::Shared(mapping) => mapping.get(addr),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Compact(compact) => compact.is_empty(),
            Self::Direct(direct) => direct.is_empty(),
            Self::RemoveFragments(fragments) => fragments.is_empty(),
            Self::Shared(mapping) => mapping.is_empty(),
        }
    }

    fn affected_fragments(&self) -> RoaringBitmap {
        match self {
            Self::Compact(compact) => compact.affected_fragments(),
            Self::Direct(direct) => {
                RoaringBitmap::from_iter(direct.keys().map(|addr| (addr >> 32) as u32))
            }
            Self::RemoveFragments(fragments) => fragments.clone(),
            Self::Shared(mapping) => mapping.affected_fragments(),
        }
    }

    fn fully_deleted_fragments(&self) -> Option<RoaringBitmap> {
        match self {
            Self::Compact(compact) => compact.fully_deleted_fragments(),
            Self::Direct(direct) if direct.values().all(Option::is_none) => Some(
                RoaringBitmap::from_iter(direct.keys().map(|addr| (addr >> 32) as u32)),
            ),
            Self::Direct(_) => None,
            Self::RemoveFragments(fragments) => Some(fragments.clone()),
            Self::Shared(mapping) => mapping.fully_deleted_fragments(),
        }
    }
}

impl DeepSizeOf for RemapStep {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            Self::Compact(compact) => compact.deep_size_of_children(context),
            Self::Direct(direct) => direct.deep_size_of_children(context),
            // Roaring does not expose its allocation capacity. Its serialized
            // size is a stable proxy for the retained containers.
            Self::RemoveFragments(fragments) => fragments.serialized_size(),
            Self::Shared(mapping) => mapping.deep_size_of_children(context),
        }
    }
}

/// Compact remap backed by per-group rewritten row bitmaps + new-fragment layouts.
///
/// Multiple remaps are retained as ordered private steps so a version chain
/// does not require another public [`RowAddrRemap`] variant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactRowAddrRemap {
    steps: Vec<RemapStep>,
}

impl CompactRowAddrRemap {
    fn new(groups: impl IntoIterator<Item = GroupInput>) -> Result<Self> {
        Ok(Self {
            steps: vec![RemapStep::Compact(CompactRemapStep::new(groups)?)],
        })
    }

    fn new_with_layout(groups: impl IntoIterator<Item = GroupInputWithLayout>) -> Result<Self> {
        Ok(Self {
            steps: vec![RemapStep::Compact(CompactRemapStep::new_with_layout(
                groups,
            )?)],
        })
    }

    fn with_removed_fragments(
        mapping: Arc<RowAddrRemap>,
        removed_fragments: RoaringBitmap,
    ) -> Self {
        Self {
            steps: vec![
                RemapStep::RemoveFragments(removed_fragments),
                RemapStep::Shared(mapping),
            ],
        }
    }

    fn chained(remaps: Vec<RowAddrRemap>) -> Self {
        let mut steps = Vec::with_capacity(remaps.len());
        for remap in remaps {
            match remap {
                RowAddrRemap::Compact(compact) => steps.extend(compact.steps),
                RowAddrRemap::Direct(direct) => steps.push(RemapStep::Direct(direct)),
            }
        }
        Self { steps }
    }

    #[inline]
    pub fn get(&self, addr: u64) -> Option<Option<u64>> {
        let mut current = addr;
        let mut was_affected = false;
        for step in &self.steps {
            match step.get(current) {
                None => {}
                Some(None) => return Some(None),
                Some(Some(mapped)) => {
                    current = mapped;
                    was_affected = true;
                }
            }
        }
        was_affected.then_some(Some(current))
    }

    fn remap_in_place(&self, row_addrs: &mut [Option<u64>]) {
        for step in &self.steps {
            for row_addr in row_addrs.iter_mut() {
                if let Some(addr) = *row_addr
                    && let Some(mapped) = step.get(addr)
                {
                    *row_addr = mapped;
                }
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.steps.iter().all(RemapStep::is_empty)
    }

    fn affected_fragments(&self) -> RoaringBitmap {
        self.steps
            .iter()
            .fold(RoaringBitmap::new(), |mut affected, step| {
                affected |= step.affected_fragments();
                affected
            })
    }

    fn fully_deleted_fragments(&self) -> Option<RoaringBitmap> {
        self.steps
            .iter()
            .try_fold(RoaringBitmap::new(), |mut deleted, step| {
                deleted |= step.fully_deleted_fragments()?;
                Some(deleted)
            })
    }
}

impl DeepSizeOf for CompactRowAddrRemap {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.steps.deep_size_of_children(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(frag: u32, offset: u32) -> u64 {
        u64::from(RowAddress::new_from_parts(frag, offset))
    }

    #[derive(Clone, Copy)]
    enum ExpectedRankedOffsets {
        Sparse,
        Dense,
        Roaring,
    }

    fn assert_layout_matches_legacy(
        frag_id: u32,
        physical_rows: u32,
        rewritten_old_row_addrs: RoaringTreemap,
        new_frags: Vec<(u32, u32)>,
        expected_representation: ExpectedRankedOffsets,
    ) {
        let rewritten_addrs = rewritten_old_row_addrs.iter().collect::<Vec<_>>();
        let new_addrs = new_frags
            .iter()
            .flat_map(|(new_frag_id, rows)| (0..*rows).map(|offset| addr(*new_frag_id, offset)))
            .collect::<Vec<_>>();
        assert_eq!(rewritten_addrs.len(), new_addrs.len());
        let expected_moved = rewritten_addrs
            .iter()
            .copied()
            .zip(new_addrs)
            .collect::<HashMap<_, _>>();

        let remap = RowAddrRemap::compact_with_layout([GroupInputWithLayout {
            rewritten_old_row_addrs,
            old_frags: vec![(frag_id, physical_rows)],
            new_frags,
        }])
        .unwrap();

        let RowAddrRemap::Compact(compact) = &remap else {
            panic!("compact_with_layout must produce a compact remap");
        };
        let RemapStep::Compact(step) = &compact.steps[0] else {
            panic!("compact_with_layout must produce a compact step");
        };
        let offsets = &step.frags[&frag_id].rewritten_offsets;
        assert!(match expected_representation {
            ExpectedRankedOffsets::Sparse => matches!(offsets, RankedOffsets::Sparse(_)),
            ExpectedRankedOffsets::Dense => matches!(offsets, RankedOffsets::Dense(_)),
            ExpectedRankedOffsets::Roaring => matches!(offsets, RankedOffsets::Roaring(_)),
        });

        for offset in 0..physical_rows {
            let old_addr = addr(frag_id, offset);
            assert_eq!(
                remap.get(old_addr),
                Some(expected_moved.get(&old_addr).copied()),
                "mismatch at ({frag_id}, {offset})"
            );
        }
        assert_eq!(remap.get(addr(frag_id, physical_rows)), None);
        assert_eq!(remap.get(addr(frag_id + 1, 0)), None);
    }

    #[test]
    fn test_sparse_ranked_offsets() {
        let offsets = RankedOffsets::try_new(
            RoaringBitmap::from_iter([1u32, 63, 511, 9_999]),
            Some(10_000),
        )
        .unwrap();
        assert!(matches!(offsets, RankedOffsets::Sparse(_)));
        assert_eq!(offsets.rank_if_present(0), None);
        assert_eq!(offsets.rank_if_present(1), Some(0));
        assert_eq!(offsets.rank_if_present(63), Some(1));
        assert_eq!(offsets.rank_if_present(511), Some(2));
        assert_eq!(offsets.rank_if_present(9_999), Some(3));
    }

    #[test]
    fn test_dense_ranked_offsets_across_words() {
        let rewritten = (0..1_024u32)
            .filter(|offset| offset % 10 != 0)
            .collect::<RoaringBitmap>();
        let offsets = RankedOffsets::try_new(rewritten.clone(), Some(1_024)).unwrap();
        assert!(matches!(offsets, RankedOffsets::Dense(_)));

        let mut expected_rank = 0u64;
        for offset in 0..1_024 {
            if rewritten.contains(offset) {
                assert_eq!(offsets.rank_if_present(offset), Some(expected_rank));
                expected_rank += 1;
            } else {
                assert_eq!(offsets.rank_if_present(offset), None);
            }
        }
        assert_eq!(expected_rank, rewritten.len());
    }

    #[test]
    fn test_run_compressed_ranked_offsets() {
        let mut rewritten = RoaringBitmap::new();
        rewritten.insert_range(100..9_900);
        let offsets = RankedOffsets::try_new(rewritten, Some(10_000)).unwrap();
        assert!(matches!(offsets, RankedOffsets::Roaring(_)));
        assert_eq!(offsets.rank_if_present(99), None);
        assert_eq!(offsets.rank_if_present(100), Some(0));
        assert_eq!(offsets.rank_if_present(9_899), Some(9_799));
        assert_eq!(offsets.rank_if_present(9_900), None);
    }

    #[test]
    fn test_compact_with_layout_matches_legacy_across_rank_representations() {
        assert_layout_matches_legacy(
            1,
            10_000,
            RoaringTreemap::from_iter(
                [1u32, 63, 511, 9_999]
                    .into_iter()
                    .map(|offset| addr(1, offset)),
            ),
            vec![(10, 2), (11, 2)],
            ExpectedRankedOffsets::Sparse,
        );

        let dense = (0..1_024u32)
            .filter(|offset| offset % 10 != 0)
            .map(|offset| addr(2, offset))
            .collect::<RoaringTreemap>();
        let dense_rows = u32::try_from(dense.len()).unwrap();
        assert_layout_matches_legacy(
            2,
            1_024,
            dense,
            vec![(20, 400), (21, dense_rows - 400)],
            ExpectedRankedOffsets::Dense,
        );

        // Binary-copy compaction captures complete fragment ranges with
        // `RoaringTreemap::insert_range`, then persists that bitmap. The
        // serialized round trip retains run containers without `optimize()`.
        let mut captured = RoaringTreemap::new();
        captured.insert_range(addr(3, 100)..addr(3, 9_900));
        let mut serialized = Vec::with_capacity(captured.serialized_size());
        captured.serialize_into(&mut serialized).unwrap();
        let persisted = RoaringTreemap::deserialize_from(std::io::Cursor::new(serialized)).unwrap();
        assert_layout_matches_legacy(
            3,
            10_000,
            persisted,
            vec![(31, 5_000), (30, 4_800)],
            ExpectedRankedOffsets::Roaring,
        );
    }

    #[test]
    fn test_compact_lookup() {
        // Group A: out-of-order old frags [4, 3], split new frags (11 empty),
        // some deletions. frag 4 (5 rows) keeps 0,2,4; frag 3 keeps 0,1, so the
        // rewritten rows (4,0)(4,2)(4,4)(3,0)(3,1) go to new frags 10(2), 12(3).
        // Group B is a fully-deleted fragment.
        let group_a = GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::from_iter([
                addr(4, 0),
                addr(4, 2),
                addr(4, 4),
                addr(3, 0),
                addr(3, 1),
            ]),
            old_frag_ids: vec![4, 3],
            new_frags: vec![(10, 2), (11, 0), (12, 3)],
        };
        let group_b = GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::new(),
            old_frag_ids: vec![7],
            new_frags: vec![],
        };
        let remap = RowAddrRemap::compact([group_a, group_b]).unwrap();

        // Moves, in rewrite order; frag 4 comes first despite the larger id.
        assert_eq!(remap.get(addr(4, 0)), Some(Some(addr(10, 0))));
        assert_eq!(remap.get(addr(4, 2)), Some(Some(addr(10, 1))));
        // Rank 2 skips the zero-row new fragment 11 and lands in fragment 12.
        assert_eq!(remap.get(addr(4, 4)), Some(Some(addr(12, 0))));
        assert_eq!(remap.get(addr(3, 0)), Some(Some(addr(12, 1))));
        assert_eq!(remap.get(addr(3, 1)), Some(Some(addr(12, 2))));
        // Deleted offsets inside a rewritten fragment.
        assert_eq!(remap.get(addr(4, 1)), Some(None));
        assert_eq!(remap.get(addr(4, 3)), Some(None));
        // Covered but fully-deleted fragment -> Some(None), not None.
        assert_eq!(remap.get(addr(7, 0)), Some(None));
        // Fragment in no group -> unaffected.
        assert_eq!(remap.get(addr(9, 0)), None);
        assert_eq!(remap.get(addr(4, 5)), Some(None));
        assert!(!remap.is_empty());
    }

    #[test]
    fn test_fragment_sets() {
        // Each deferred version deletes a different covered fragment. The
        // chain must retain the flat direct map's union semantics.
        let first_dead = RowAddrRemap::compact([GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::new(),
            old_frag_ids: vec![3],
            new_frags: vec![],
        }])
        .unwrap();
        let second_dead = RowAddrRemap::compact([GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::new(),
            old_frag_ids: vec![7],
            new_frags: vec![],
        }])
        .unwrap();
        let dead = RowAddrRemap::chained([first_dead.clone(), second_dead]);
        assert_eq!(
            dead.fully_deleted_fragments(),
            Some(RoaringBitmap::from_iter([3u32, 7u32]))
        );
        assert_eq!(
            dead.affected_fragments(),
            RoaringBitmap::from_iter([3u32, 7u32])
        );

        // At least one rewritten row -> not fully deleted, but both covered
        // fragments (including the fully-deleted frag 1) are still affected.
        let alive = RowAddrRemap::compact([GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::from_iter([addr(0, 0)]),
            old_frag_ids: vec![0, 1],
            new_frags: vec![(10, 1)],
        }])
        .unwrap();
        assert!(alive.fully_deleted_fragments().is_none());
        assert_eq!(
            alive.affected_fragments(),
            RoaringBitmap::from_iter([0u32, 1u32])
        );
        assert!(
            RowAddrRemap::chained([first_dead, alive])
                .fully_deleted_fragments()
                .is_none()
        );
    }

    #[test]
    fn test_compact_rejects_rewritten_addrs_outside_old_frags() {
        // Rewritten addresses reference frag 5, not in old_frags. The count
        // still matches (2 == 2), so only the per-fragment split catches it.
        let input = GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::from_iter([addr(0, 0), addr(5, 0)]),
            old_frag_ids: vec![0],
            new_frags: vec![(10, 2)],
        };
        assert!(RowAddrRemap::compact([input]).is_err());
    }

    #[test]
    fn test_compact_preserves_explicit_fragment_order() {
        let remap = RowAddrRemap::compact([GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::from_iter([addr(0, 0), addr(0, 1)]),
            old_frag_ids: vec![0],
            new_frags: vec![(12, 1), (11, 1)],
        }])
        .unwrap();
        assert_eq!(remap.get(addr(0, 0)), Some(Some(addr(12, 0))));
        assert_eq!(remap.get(addr(0, 1)), Some(Some(addr(11, 0))));
    }

    #[test]
    fn test_with_removed_fragments_keeps_mapping_lazy() {
        let mapping = Arc::new(
            RowAddrRemap::compact([GroupInput {
                rewritten_old_row_addrs: RoaringTreemap::from_iter([addr(1, 0), addr(1, 1)]),
                old_frag_ids: vec![1],
                new_frags: vec![(4, 2)],
            }])
            .unwrap(),
        );
        let filtered =
            RowAddrRemap::with_removed_fragments(mapping, RoaringBitmap::from_iter([2u32]));

        assert!(matches!(filtered, RowAddrRemap::Compact(_)));
        assert_eq!(filtered.get(addr(1, 0)), Some(Some(addr(4, 0))));
        assert_eq!(filtered.get(addr(2, 0)), Some(None));
        assert_eq!(filtered.get(addr(3, 0)), None);
        assert_eq!(
            filtered.affected_fragments(),
            RoaringBitmap::from_iter([1u32, 2u32])
        );

        let mut batch = vec![Some(addr(1, 1)), Some(addr(2, 0)), Some(addr(3, 0))];
        filtered.remap_in_place(&mut batch);
        assert_eq!(batch, vec![Some(addr(4, 1)), None, Some(addr(3, 0))]);
    }

    #[test]
    fn test_direct_and_empty() {
        // Direct covers arbitrary maps the compact form can't express.
        let mut map = HashMap::new();
        map.insert(addr(2, 0), Some(addr(9, 9)));
        map.insert(addr(5, 1), None);
        let remap = RowAddrRemap::direct(map);
        assert_eq!(remap.get(addr(2, 0)), Some(Some(addr(9, 9))));
        assert_eq!(remap.get(addr(5, 1)), Some(None));
        assert_eq!(remap.get(addr(2, 1)), None);
        // affected_fragments over an explicit map: the fragment of every key.
        assert_eq!(
            remap.affected_fragments(),
            RoaringBitmap::from_iter([2u32, 5u32])
        );

        let empty = RowAddrRemap::empty();
        assert!(empty.is_empty());
        assert_eq!(empty.get(addr(0, 0)), None);
    }

    #[test]
    fn test_chained_lookup_and_batch() {
        let first = RowAddrRemap::compact([GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::from_iter([addr(0, 0), addr(0, 2)]),
            old_frag_ids: vec![0],
            new_frags: vec![(10, 2)],
        }])
        .unwrap();
        let second = RowAddrRemap::compact([GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::from_iter([addr(10, 1)]),
            old_frag_ids: vec![10],
            new_frags: vec![(20, 1)],
        }])
        .unwrap();
        let chain = RowAddrRemap::chained([first, second]);

        assert_eq!(chain.get(addr(0, 0)), Some(None));
        assert_eq!(chain.get(addr(0, 1)), Some(None));
        assert_eq!(chain.get(addr(0, 2)), Some(Some(addr(20, 0))));
        assert_eq!(chain.get(addr(1, 0)), None);

        let mut batch = vec![
            Some(addr(0, 0)),
            Some(addr(0, 1)),
            Some(addr(0, 2)),
            Some(addr(1, 0)),
            None,
        ];
        chain.remap_in_place(&mut batch);
        assert_eq!(
            batch,
            vec![None, None, Some(addr(20, 0)), Some(addr(1, 0)), None]
        );
    }
}
