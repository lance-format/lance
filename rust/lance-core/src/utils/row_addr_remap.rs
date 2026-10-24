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
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::size_of;

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

    /// Returns fragments this remap can prove have no surviving physical rows.
    ///
    /// A non-empty direct map cannot provide this proof because it does not
    /// record the physical row count of each fragment.
    pub fn fully_deleted_fragments(&self) -> Option<RoaringBitmap> {
        match self {
            Self::Compact(c) => c.fully_deleted_fragments(),
            Self::Direct(m) if m.is_empty() => Some(RoaringBitmap::new()),
            Self::Direct(_) => None,
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

        let mut per_frag: IntMap<u32, RoaringBitmap> = rewritten_old_row_addrs
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

/// Hasher for this module's integer-keyed maps.
///
/// `remap_row_id` probes `CompactRemapStep::frags` once per remap step per row
/// address, so at hundreds of millions of rows the default SipHash is a large
/// share of consolidation CPU. The keys are internal fragment ids, never
/// attacker-supplied, so the hash-flooding resistance it buys is worth nothing here.
///
/// Only single-integer keys are hashed well: `write_u32`/`write_u64`/`write_usize`
/// replace the state rather than mixing into it, so a composite key would collide on
/// its last field alone. Correct either way — `Eq` still decides — but keep these maps
/// keyed by one integer.
#[derive(Clone, Copy)]
struct IntHasher(u64);

/// FNV-1a offset basis, so the byte fallback does not absorb leading zero bytes.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

impl Default for IntHasher {
    #[inline]
    fn default() -> Self {
        Self(FNV_OFFSET_BASIS)
    }
}

impl Hasher for IntHasher {
    /// Folds the high half down: multiplication only carries upward, so without this
    /// the low bits — which hashbrown uses to pick the bucket — would ignore the high
    /// bits of the key, and fragment ids strided by a power of two would all land in
    /// one bucket.
    #[inline]
    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 32)
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Fallback for key types this module does not use.
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.0 = u64::from(value).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = value.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
}

/// `HashMap` over single-integer keys, hashed with [`IntHasher`].
type IntMap<K, V> = HashMap<K, V, BuildHasherDefault<IntHasher>>;

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompactRemapStep {
    groups: Vec<GroupRemap>,
    /// Old fragment id -> its bitmap/rank layout and rewrite group. Size is
    /// O(#fragments), not rows.
    frags: IntMap<u32, OldFragmentRemap>,
}

impl CompactRemapStep {
    fn new(groups: impl IntoIterator<Item = GroupInput>) -> Result<Self> {
        let mut frags = IntMap::default();
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
        let mut frags = IntMap::default();
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
        self.get_in(old_frag, addr)
    }

    // Always inlined, so `get` compiles as it did before the split.
    #[inline(always)]
    fn get_in(&self, old_frag: &OldFragmentRemap, addr: u64) -> Option<Option<u64>> {
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
}

impl RemapStep {
    fn get(&self, addr: u64) -> Option<Option<u64>> {
        match self {
            Self::Compact(compact) => compact.get(addr),
            Self::Direct(direct) => direct.get(&addr).copied(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Compact(compact) => compact.is_empty(),
            Self::Direct(direct) => direct.is_empty(),
        }
    }

    fn affected_fragments(&self) -> RoaringBitmap {
        match self {
            Self::Compact(compact) => compact.affected_fragments(),
            Self::Direct(direct) => {
                RoaringBitmap::from_iter(direct.keys().map(|addr| (addr >> 32) as u32))
            }
        }
    }

    fn fully_deleted_fragments(&self) -> Option<RoaringBitmap> {
        match self {
            Self::Compact(compact) => compact.fully_deleted_fragments(),
            Self::Direct(direct) if direct.is_empty() => Some(RoaringBitmap::new()),
            Self::Direct(_) => None,
        }
    }
}

impl DeepSizeOf for RemapStep {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            Self::Compact(compact) => compact.deep_size_of_children(context),
            Self::Direct(direct) => direct.deep_size_of_children(context),
        }
    }
}

/// Chains with at least this many steps also index their steps by fragment.
const MIN_STEPS_TO_INDEX: usize = 32;

/// An indexed lookup walks step by step when the next step naming the row's
/// fragment is at most this many steps ahead: a short walk costs less than
/// asking the index.
const WALK_WITHIN_STEPS: usize = 3;

/// A walk goes back to the index after this many steps in a row do not name the
/// row's fragment.
const WALK_UNTIL_IDLE_STEPS: usize = 8;

/// Rows a batch remap over an indexed chain tracks at a time.
const REMAP_BATCH_ROWS: usize = 1 << 16;

/// For each fragment id, the steps that name it, in step order.
///
/// A step answers only for addresses in the fragments it names, so asking just
/// these steps gives the same answer as asking every step in turn.
#[derive(Clone)]
struct FragmentSteps(IntMap<u32, Vec<u32>>);

impl FragmentSteps {
    fn new(steps: &[RemapStep]) -> Self {
        let mut by_fragment = IntMap::<u32, Vec<u32>>::default();
        // Most fragments are named by one step.
        let mut add = |frag_id: u32, step: u32| {
            by_fragment
                .entry(frag_id)
                .or_insert_with(|| Vec::with_capacity(1))
                .push(step);
        };
        for (step, remap) in (0u32..).zip(steps) {
            match remap {
                RemapStep::Compact(compact) => {
                    for &frag_id in compact.frags.keys() {
                        add(frag_id, step);
                    }
                }
                RemapStep::Direct(direct) => {
                    let frag_ids = direct
                        .keys()
                        .map(|addr| (addr >> 32) as u32)
                        .collect::<HashSet<_>>();
                    for frag_id in frag_ids {
                        add(frag_id, step);
                    }
                }
            }
        }
        Self(by_fragment)
    }
}

impl std::fmt::Debug for FragmentSteps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FragmentSteps({} fragments)", self.0.len())
    }
}

impl DeepSizeOf for FragmentSteps {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}

/// Compact remap backed by per-group rewritten row bitmaps + new-fragment layouts.
///
/// Multiple remaps are retained as ordered private steps so a version chain
/// does not require another public [`RowAddrRemap`] variant.
#[derive(Clone, Debug)]
pub struct CompactRowAddrRemap {
    steps: Vec<RemapStep>,
    /// `None` keeps short chains on the step-by-step path.
    fragment_steps: Option<FragmentSteps>,
}

impl PartialEq for CompactRowAddrRemap {
    fn eq(&self, other: &Self) -> bool {
        self.steps == other.steps
    }
}

impl Eq for CompactRowAddrRemap {}

impl CompactRowAddrRemap {
    fn new(groups: impl IntoIterator<Item = GroupInput>) -> Result<Self> {
        Ok(Self::from_steps(vec![RemapStep::Compact(
            CompactRemapStep::new(groups)?,
        )]))
    }

    fn new_with_layout(groups: impl IntoIterator<Item = GroupInputWithLayout>) -> Result<Self> {
        Ok(Self::from_steps(vec![RemapStep::Compact(
            CompactRemapStep::new_with_layout(groups)?,
        )]))
    }

    fn chained(remaps: Vec<RowAddrRemap>) -> Self {
        let mut steps = Vec::with_capacity(remaps.len());
        for remap in remaps {
            match remap {
                RowAddrRemap::Compact(compact) => steps.extend(compact.steps),
                RowAddrRemap::Direct(direct) => steps.push(RemapStep::Direct(direct)),
            }
        }
        Self::from_steps(steps)
    }

    fn from_steps(steps: Vec<RemapStep>) -> Self {
        let fragment_steps = (steps.len() >= MIN_STEPS_TO_INDEX
            && u32::try_from(steps.len()).is_ok())
        .then(|| FragmentSteps::new(&steps));
        Self {
            steps,
            fragment_steps,
        }
    }

    #[inline]
    pub fn get(&self, addr: u64) -> Option<Option<u64>> {
        match &self.fragment_steps {
            Some(fragment_steps) => self.get_by_fragment(fragment_steps, addr),
            None => self.get_by_step(addr),
        }
    }

    #[inline]
    fn get_by_step(&self, addr: u64) -> Option<Option<u64>> {
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

    /// [`Self::get_by_step`], skipping the steps that do not name the fragment
    /// the row is in.
    // Out of line, so `get` stays small where it is inlined.
    #[inline(never)]
    fn get_by_fragment(&self, fragment_steps: &FragmentSteps, addr: u64) -> Option<Option<u64>> {
        let mut current = addr;
        let mut next_step = 0;
        let mut was_affected = false;
        while let Some(steps) = fragment_steps.0.get(&((current >> 32) as u32)) {
            let first = steps.partition_point(|&step| (step as usize) < next_step);
            let Some(&step) = steps.get(first) else {
                break;
            };
            let step = step as usize;
            if step - next_step <= WALK_WITHIN_STEPS {
                let mut idle = 0;
                for remap in &self.steps[next_step..] {
                    next_step += 1;
                    // `None` if the step does not name the row's fragment.
                    let named = match remap {
                        RemapStep::Compact(compact) => compact
                            .frags
                            .get(&((current >> 32) as u32))
                            .map(|old_frag| compact.get_in(old_frag, current)),
                        RemapStep::Direct(direct) => {
                            direct.get(&current).map(|&answer| Some(answer))
                        }
                    };
                    match named {
                        None => {
                            idle += 1;
                            if idle == WALK_UNTIL_IDLE_STEPS {
                                break;
                            }
                        }
                        Some(answer) => {
                            idle = 0;
                            match answer {
                                None => {}
                                Some(None) => return Some(None),
                                Some(Some(mapped)) => {
                                    current = mapped;
                                    was_affected = true;
                                }
                            }
                        }
                    }
                }
                continue;
            }
            next_step = step + 1;
            match self.steps[step].get(current) {
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
        if self.fragment_steps.is_some() {
            self.remap_in_place_by_fragment(row_addrs);
            return;
        }
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

    /// The step-major walk, with each step visiting only the rows in the
    /// fragments it names. Rows go [`REMAP_BATCH_ROWS`] at a time, and no list
    /// keeps more than about four times the rows it holds, so the scratch memory
    /// is bounded by the rows of a pass, whatever the number of steps.
    #[inline(never)]
    fn remap_in_place_by_fragment(&self, row_addrs: &mut [Option<u64>]) {
        let mut rows_by_fragment = IntMap::<u32, Vec<u32>>::default();
        let mut moved = Vec::new();
        for batch in row_addrs.chunks_mut(REMAP_BATCH_ROWS) {
            add_rows(&mut rows_by_fragment, batch, 0..batch.len() as u32);
            for step in &self.steps {
                match step {
                    RemapStep::Compact(compact)
                        if compact.frags.len() <= rows_by_fragment.len() =>
                    {
                        for (&frag_id, old_frag) in &compact.frags {
                            if let Entry::Occupied(mut rows) = rows_by_fragment.entry(frag_id) {
                                apply_step(rows.get_mut(), batch, &mut moved, |addr| {
                                    compact.get_in(old_frag, addr)
                                });
                                if rows.get().is_empty() {
                                    rows.remove();
                                }
                            }
                        }
                    }
                    RemapStep::Compact(compact) => rows_by_fragment.retain(|frag_id, rows| {
                        if let Some(old_frag) = compact.frags.get(frag_id) {
                            apply_step(rows, batch, &mut moved, |addr| {
                                compact.get_in(old_frag, addr)
                            });
                        }
                        !rows.is_empty()
                    }),
                    RemapStep::Direct(direct) => rows_by_fragment.retain(|_, rows| {
                        apply_step(rows, batch, &mut moved, |addr| direct.get(&addr).copied());
                        !rows.is_empty()
                    }),
                }
                // Rows join their new fragments after the step, so no step applies twice.
                add_rows(&mut rows_by_fragment, batch, moved.drain(..));
            }
            rows_by_fragment.clear();
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

/// Hash once per run of rows in the same fragment.
fn add_rows(
    rows_by_fragment: &mut IntMap<u32, Vec<u32>>,
    row_addrs: &[Option<u64>],
    positions: impl Iterator<Item = u32>,
) {
    let mut rows = positions
        .filter_map(|position| Some((position, (row_addrs[position as usize]? >> 32) as u32)))
        .peekable();
    while let Some((position, frag_id)) = rows.next() {
        let list = rows_by_fragment.entry(frag_id).or_default();
        list.push(position);
        while let Some((position, _)) = rows.next_if(|&(_, next)| next == frag_id) {
            list.push(position);
        }
    }
}

fn apply_step(
    positions: &mut Vec<u32>,
    row_addrs: &mut [Option<u64>],
    moved: &mut Vec<u32>,
    get: impl Fn(u64) -> Option<Option<u64>>,
) {
    positions.retain(|&position| {
        let row_addr = &mut row_addrs[position as usize];
        let Some(addr) = *row_addr else {
            return false;
        };
        match get(addr) {
            None => true,
            Some(mapped) => {
                *row_addr = mapped;
                if mapped.is_some() {
                    moved.push(position);
                }
                false
            }
        }
    });
    // Release excess capacity so mostly emptied lists do not accumulate per step.
    if positions.len() < positions.capacity() / 4 {
        shrink(positions);
    }
}

/// Out of line: inlined, it slowed the batch loops.
#[cold]
#[inline(never)]
fn shrink(positions: &mut Vec<u32>) {
    // Copy instead of shrink_to_fit: in-place shrinking caused glibc RSS growth
    // by leaving block remainders unusable for later lists.
    *positions = positions.to_vec();
}

impl DeepSizeOf for CompactRowAddrRemap {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.steps.deep_size_of_children(context)
            + self.fragment_steps.deep_size_of_children(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::ProptestConfig;
    use proptest::test_runner::TestCaseResult;
    use proptest::{prop_assert, prop_assert_eq};
    use std::ops::Range;

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
    fn test_direct_partial_map_is_not_fully_deleted() {
        let remap = RowAddrRemap::direct(HashMap::from([(addr(5, 1), None)]));

        assert_eq!(remap.get(addr(5, 0)), None);
        assert_eq!(remap.fully_deleted_fragments(), None);
    }

    #[test]
    fn test_chained_partial_direct_maps_are_not_fully_deleted() {
        let remap = RowAddrRemap::chained([
            RowAddrRemap::direct(HashMap::from([(addr(5, 1), None)])),
            RowAddrRemap::direct(HashMap::from([(addr(6, 1), None)])),
        ]);

        assert!(matches!(remap, RowAddrRemap::Compact(_)));
        assert_eq!(remap.get(addr(5, 0)), None);
        assert_eq!(remap.get(addr(6, 0)), None);
        assert_eq!(remap.fully_deleted_fragments(), None);
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

    fn int_hash_of<K: std::hash::Hash>(key: &K) -> u64 {
        use std::hash::BuildHasher;
        BuildHasherDefault::<IntHasher>::default().hash_one(key)
    }

    #[test]
    fn test_int_hasher_is_stateless_and_spreads_low_bits() {
        // A `BuildHasherDefault` carries no seed, so a key must hash the same at insert,
        // at lookup and after a resize; if it did not, a key could become unreachable.
        for value in [0u32, 1, 7, 12345, 1 << 31, u32::MAX] {
            assert_eq!(int_hash_of(&value), int_hash_of(&value));
        }
        // `get` borrows the key, so the borrowed form has to agree with the owned one.
        let key = 424242u32;
        assert_eq!(int_hash_of(&key), int_hash_of(&&key));

        // Multiplication only carries upward, so without the fold in `finish` the low
        // bits would ignore the high bits of the key and every fragment id below would
        // land in the same bucket.
        let buckets = |ids: &[u32]| -> usize {
            let table_mask = 2048 - 1;
            ids.iter()
                .map(|id| int_hash_of(id) & table_mask)
                .collect::<std::collections::HashSet<_>>()
                .len()
        };
        let strided: Vec<u32> = (0..1024).map(|i| i << 20).collect();
        let dense: Vec<u32> = (0..1024).collect();
        assert!(
            buckets(&strided) > 512,
            "power-of-two-strided fragment ids clustered into {} of 2048 buckets",
            buckets(&strided)
        );
        assert!(buckets(&dense) > 512, "dense fragment ids clustered");
    }

    #[test]
    fn test_int_hasher_composite_keys_collide_on_last_field() {
        // `write_u32`/`write_u64` replace the state instead of mixing into it, so a
        // composite key keeps only its last integer field. Pinned because it is the
        // documented reason these maps are single-integer-keyed: the map still answers
        // correctly, it just degenerates to one bucket.
        assert_eq!(int_hash_of(&(1u32, 5u32)), int_hash_of(&(999u32, 5u32)));
        assert_eq!(int_hash_of(&("abc", 4u32)), int_hash_of(&("zzzzzz", 4u32)));

        let mut map: IntMap<(u32, u32), u32> = IntMap::default();
        for a in 0..48u32 {
            for b in 0..48u32 {
                map.insert((a, b), a * 1000 + b);
            }
        }
        assert_eq!(map.len(), 48 * 48);
        for a in 0..48u32 {
            for b in 0..48u32 {
                assert_eq!(map.get(&(a, b)), Some(&(a * 1000 + b)));
            }
        }
        assert_eq!(map.get(&(48, 0)), None);
    }

    #[test]
    fn test_outputs_do_not_depend_on_fragment_insertion_order() {
        // The maps are hashed with a fixed seed, so their iteration order is stable but
        // different from the default hasher's. Nothing observable may depend on it.
        let remap_over = |old_frag_ids: Vec<u32>| {
            let rewritten = RoaringTreemap::from_iter(old_frag_ids.iter().map(|f| addr(*f, 0)));
            let num_rows = old_frag_ids.len() as u32;
            RowAddrRemap::compact([GroupInput {
                rewritten_old_row_addrs: rewritten,
                old_frag_ids,
                new_frags: vec![(1000, num_rows)],
            }])
            .unwrap()
        };
        let forward = remap_over(vec![3, 1, 4, 1 << 20, 2 << 20, 9, 700_000]);
        let reverse = remap_over(vec![700_000, 9, 2 << 20, 1 << 20, 4, 1, 3]);

        // `old_frag_ids` order is load-bearing for the row-to-row mapping, so only the
        // set-shaped outputs are comparable across the two.
        assert_eq!(
            forward.affected_fragments(),
            reverse.affected_fragments(),
            "affected fragments must not depend on insertion order"
        );
        assert_eq!(
            forward.fully_deleted_fragments(),
            reverse.fully_deleted_fragments()
        );
    }

    fn rewrite(old: &[(u32, u32)], deleted: &[u64], new_id: u32) -> RowAddrRemap {
        let rewritten = old
            .iter()
            .flat_map(|&(frag, rows)| (0..rows).map(move |offset| addr(frag, offset)))
            .filter(|old_addr| !deleted.contains(old_addr))
            .collect::<RoaringTreemap>();
        let rows = rewritten.len() as u32;
        RowAddrRemap::compact_with_layout([GroupInputWithLayout {
            rewritten_old_row_addrs: rewritten,
            old_frags: old.to_vec(),
            new_frags: if rows == 0 {
                vec![]
            } else {
                vec![(new_id, rows)]
            },
        }])
        .unwrap()
    }

    fn compact_chain(remap: &RowAddrRemap) -> &CompactRowAddrRemap {
        let RowAddrRemap::Compact(chain) = remap else {
            panic!("expected a chain");
        };
        chain
    }

    #[test]
    fn test_indexed_chain_matches_step_by_step() {
        let steps = vec![
            // Fragment 5 has 4 rows here and 8 rows when named again below, so
            // (5, 6) is out of range for this step and moves at the later one.
            rewrite(&[(5, 4)], &[], 20),
            rewrite(&[(30, 3)], &[], 30),
            rewrite(&[(30, 3)], &[], 31),
            rewrite(&[(40, 4)], &[addr(40, 2)], 41),
            rewrite(&[(41, 3)], &[addr(41, 0)], 42),
            rewrite(&[(50, 2), (55, 1)], &[], 51),
            rewrite(&[(51, 3)], &[], 52),
            rewrite(&[(52, 3)], &[addr(52, 2)], 53),
            RowAddrRemap::direct(HashMap::from([
                (addr(60, 0), Some(addr(61, 5))),
                (addr(60, 1), None),
            ])),
            rewrite(&[(5, 8)], &[addr(5, 7)], 21),
            rewrite(&[(61, 6)], &[addr(61, 0)], 62),
            rewrite(&[(53, 2)], &[], 54),
            // Steps writing into one of their own sources: rows from the other
            // source land inside its old range, and the step must not move them
            // again.
            rewrite(&[(71, 2), (70, 2)], &[], 70),
            rewrite(&[(81, 2), (80, 2)], &[], 80),
        ];
        let expected = [
            (addr(5, 2), Some(Some(addr(20, 2)))),
            (addr(5, 6), Some(Some(addr(21, 6)))),
            (addr(5, 7), Some(None)),
            (addr(5, 8), None),
            (addr(30, 1), Some(Some(addr(31, 1)))),
            (addr(40, 0), Some(None)),
            (addr(40, 1), Some(Some(addr(42, 0)))),
            (addr(40, 2), Some(None)),
            (addr(50, 1), Some(Some(addr(54, 1)))),
            (addr(55, 0), Some(None)),
            (addr(51, 2), Some(None)),
            (addr(60, 0), Some(Some(addr(62, 4)))),
            (addr(60, 1), Some(None)),
            (addr(60, 2), None),
            (addr(70, 0), Some(Some(addr(70, 2)))),
            (addr(71, 1), Some(Some(addr(70, 1)))),
            (addr(80, 1), Some(Some(addr(80, 3)))),
            (addr(81, 0), Some(Some(addr(80, 0)))),
            (addr(999, 0), None),
        ];

        // Unrelated steps before and after, to cross the threshold.
        let pad = |k: u32| rewrite(&[(1_000 + k, 1)], &[], 2_000 + k);
        let half = MIN_STEPS_TO_INDEX as u32 / 2;
        let padded = (0..half)
            .map(pad)
            .chain(steps.iter().cloned())
            .chain((half..2 * half).map(pad))
            .collect::<Vec<_>>();

        let short = RowAddrRemap::chained(steps);
        let long = RowAddrRemap::chained(padded);
        assert!(compact_chain(&short).fragment_steps.is_none());
        assert!(compact_chain(&long).fragment_steps.is_some());
        for chain in [&short, &long] {
            for &(old_addr, answer) in &expected {
                assert_eq!(chain.get(old_addr), answer, "address {old_addr:#x}");
            }
            let mut batch = expected
                .iter()
                .map(|&(old_addr, _)| Some(old_addr))
                .chain([None])
                .collect::<Vec<_>>();
            chain.remap_in_place(&mut batch);
            let remapped = expected
                .iter()
                .map(|&(old_addr, answer)| answer.unwrap_or(Some(old_addr)))
                .chain([None])
                .collect::<Vec<_>>();
            assert_eq!(batch, remapped);
        }
    }

    #[test]
    fn test_batch_remap_spanning_several_passes_matches_step_by_step() {
        let mut rows = 4u32;
        let steps = (0..MIN_STEPS_TO_INDEX as u32 + 8)
            .map(|k| {
                let deleted = if k % 10 == 9 {
                    vec![addr(k, 0)]
                } else {
                    vec![]
                };
                let step = rewrite(&[(k, rows)], &deleted, k + 1);
                rows -= deleted.len() as u32;
                step
            })
            .collect::<Vec<_>>();
        let chain = RowAddrRemap::chained(steps.clone());
        assert!(compact_chain(&chain).fragment_steps.is_some());

        let mut batch = (0..2 * REMAP_BATCH_ROWS + 5)
            .map(|i| (i % 1_000 != 999).then(|| addr((i % 7) as u32 * 5, (i % 5) as u32)))
            .collect::<Vec<_>>();
        let step_by_step = |old_addr: u64| {
            steps
                .iter()
                .try_fold(old_addr, |current, step| match step.get(current) {
                    None => Some(current),
                    Some(mapped) => mapped,
                })
        };
        let answers = batch
            .iter()
            .flatten()
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|old_addr| (old_addr, step_by_step(old_addr)))
            .collect::<HashMap<_, _>>();
        for (&old_addr, &answer) in &answers {
            assert_eq!(
                chain.get(old_addr).unwrap_or(Some(old_addr)),
                answer,
                "address {old_addr:#x}"
            );
        }
        let expected = batch
            .iter()
            .map(|row_addr| row_addr.and_then(|old_addr| answers[&old_addr]))
            .collect::<Vec<_>>();
        chain.remap_in_place(&mut batch);
        assert_eq!(batch, expected);
    }

    #[test]
    fn test_fragment_index_size_is_its_map_and_lists() {
        let RowAddrRemap::Compact(indexed) =
            RowAddrRemap::chained((0..MIN_STEPS_TO_INDEX as u32 + 4).map(|k| {
                rewrite(
                    &[(2 * k, 64), (2 * k + 1, 64)],
                    &[addr(2 * k, 3)],
                    1_000 + k,
                )
            }))
        else {
            panic!("expected a chain");
        };
        let index = indexed.fragment_steps.as_ref().unwrap();
        let index_bytes = index.0.capacity() * (size_of::<u32>() + size_of::<Vec<u32>>() + 1)
            + index
                .0
                .values()
                .map(|steps| steps.capacity() * size_of::<u32>())
                .sum::<usize>();
        let indexed_bytes = indexed.deep_size_of();
        let unindexed = CompactRowAddrRemap {
            steps: indexed.steps.clone(),
            fragment_steps: None,
        };
        assert_eq!(indexed, unindexed);

        // The same steps, not clones, whose vectors would have other capacities.
        let unindexed = CompactRowAddrRemap {
            steps: indexed.steps,
            fragment_steps: None,
        };
        assert_eq!(indexed_bytes, unindexed.deep_size_of() + index_bytes);
    }

    /// splitmix64, so a failing history can be rebuilt from its seed.
    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) % n
        }

        fn chance(&mut self, percent: u64) -> bool {
            self.below(100) < percent
        }
    }

    type GeneratedStep = (RowAddrRemap, HashMap<u64, Option<u64>>);

    /// Histories include deletions, multiple rewrite groups, reused fragment ids,
    /// and direct steps. Materialized maps provide an independent oracle.
    fn random_history(seed: u64, num_steps: usize) -> (Vec<GeneratedStep>, Vec<(u32, u32)>) {
        let mut rng = Rng(seed);
        let mut live = (0..6u32)
            .map(|id| {
                let rows = 1 + rng.below(10) as u32;
                (id, rows, vec![true; rows as usize])
            })
            .collect::<Vec<_>>();
        let mut seen = live
            .iter()
            .map(|(id, rows, _)| (*id, *rows))
            .collect::<Vec<_>>();
        let mut retired = Vec::new();
        let mut next_id = 100u32;
        let mut steps = Vec::new();
        for _ in 0..num_steps {
            for (_, _, alive) in live.iter_mut() {
                for row in alive.iter_mut() {
                    *row &= !rng.chance(10);
                }
            }
            let mut groups = Vec::new();
            let mut map = HashMap::new();
            let mut created = Vec::new();
            let mut written = Vec::new();
            for _ in 0..1 + rng.below(2) {
                if live.is_empty() {
                    break;
                }
                let sources = (0..1 + rng.below(3).min(live.len() as u64 - 1))
                    .map(|_| live.swap_remove(rng.below(live.len() as u64) as usize))
                    .collect::<Vec<_>>();
                let read_order = sources
                    .iter()
                    .flat_map(|(id, rows, alive)| {
                        (0..*rows)
                            .filter(move |offset| alive[*offset as usize])
                            .map(move |offset| addr(*id, offset))
                    })
                    .collect::<Vec<_>>();
                let mut new_frags: Vec<(u32, u32)> = Vec::new();
                let mut remaining = read_order.len() as u32;
                while remaining > 0 {
                    let rows = if remaining > 1 && rng.chance(30) {
                        1 + rng.below(u64::from(remaining - 1)) as u32
                    } else {
                        remaining
                    };
                    let own = sources
                        .iter()
                        .map(|(id, _, _)| *id)
                        .filter(|id| !created.contains(id))
                        .collect::<Vec<_>>();
                    let old = retired
                        .iter()
                        .copied()
                        .filter(|id| !created.contains(id))
                        .collect::<Vec<_>>();
                    let id = match rng.below(100) {
                        0..20 if !own.is_empty() => own[rng.below(own.len() as u64) as usize],
                        20..35 if !old.is_empty() => old[rng.below(old.len() as u64) as usize],
                        _ => {
                            next_id += 1;
                            next_id
                        }
                    };
                    created.push(id);
                    new_frags.push((id, rows));
                    remaining -= rows;
                }
                let new_addrs = new_frags
                    .iter()
                    .flat_map(|&(id, rows)| (0..rows).map(move |offset| addr(id, offset)));
                for (old_addr, new_addr) in read_order.iter().zip(new_addrs) {
                    map.insert(*old_addr, Some(new_addr));
                }
                for (id, rows, _) in &sources {
                    for offset in 0..*rows {
                        map.entry(addr(*id, offset)).or_insert(None);
                    }
                }
                groups.push(GroupInputWithLayout {
                    rewritten_old_row_addrs: read_order.iter().copied().collect(),
                    old_frags: sources.iter().map(|(id, rows, _)| (*id, *rows)).collect(),
                    new_frags: new_frags.clone(),
                });
                retired.extend(sources.iter().map(|(id, _, _)| *id));
                written.extend(new_frags);
            }
            retired.retain(|id| !created.contains(id));
            for (id, rows) in written {
                live.push((id, rows, vec![true; rows as usize]));
                seen.push((id, rows));
            }
            let remap = if rng.chance(10) {
                RowAddrRemap::direct(map.clone())
            } else {
                RowAddrRemap::compact_with_layout(groups).unwrap()
            };
            steps.push((remap, map));
        }
        (steps, seen)
    }

    // History lengths split three ways, so each property test stays short; together
    // they cover chains on both sides of the indexing threshold.
    const SHORT_CHAINS: Range<usize> = 0..MIN_STEPS_TO_INDEX;
    const INDEXED_CHAINS: Range<usize> = MIN_STEPS_TO_INDEX..MIN_STEPS_TO_INDEX + 20;
    const LONG_INDEXED_CHAINS: Range<usize> = MIN_STEPS_TO_INDEX + 20..2 * MIN_STEPS_TO_INDEX + 8;

    /// The share of the default case budget for histories of `num_steps` steps,
    /// in proportion to the range's width. `PROPTEST_CASES`, when set, applies to
    /// each test in full.
    fn cases_for(num_steps: Range<usize>) -> ProptestConfig {
        let cases = ProptestConfig::default().cases as usize * num_steps.len();
        ProptestConfig::with_cases(cases.div_ceil(LONG_INDEXED_CHAINS.end) as u32)
    }

    fn check_chain_matches_materialized_steps(
        seed: u64,
        num_steps: usize,
        splits: Vec<usize>,
    ) -> TestCaseResult {
        let (steps, seen) = random_history(seed, num_steps);
        let reference = |old_addr: u64| {
            let mut current = old_addr;
            let mut was_affected = false;
            for (_, map) in &steps {
                match map.get(&current) {
                    None => {}
                    Some(None) => return Some(None),
                    Some(Some(mapped)) => {
                        current = *mapped;
                        was_affected = true;
                    }
                }
            }
            was_affected.then_some(Some(current))
        };

        let flat = RowAddrRemap::chained(steps.iter().map(|(remap, _)| remap.clone()));
        let mut bounds = splits
            .into_iter()
            .filter(|&at| at < steps.len())
            .collect::<Vec<_>>();
        bounds.extend([0, steps.len()]);
        bounds.sort_unstable();
        let nested = RowAddrRemap::chained(bounds.windows(2).map(|piece| {
            RowAddrRemap::chained(
                steps[piece[0]..piece[1]]
                    .iter()
                    .map(|(remap, _)| remap.clone()),
            )
        }));
        if let RowAddrRemap::Compact(chain) = &flat {
            prop_assert_eq!(
                chain.fragment_steps.is_some(),
                chain.steps.len() >= MIN_STEPS_TO_INDEX
            );
        }

        let mut queries = seen
            .iter()
            .flat_map(|&(id, rows)| (0..rows + 2).map(move |offset| addr(id, offset)))
            .collect::<Vec<_>>();
        queries.extend([addr(99, 0), addr(0xFFFF_FFF0, 3)]);
        let expected = queries.iter().map(|&a| reference(a)).collect::<Vec<_>>();
        for remap in [&flat, &nested] {
            for (&old_addr, answer) in queries.iter().zip(&expected) {
                prop_assert_eq!(
                    remap.get(old_addr),
                    *answer,
                    "address {:#x}, seed {}",
                    old_addr,
                    seed
                );
            }
            for reversed in [false, true] {
                let mut batch = Vec::new();
                let mut remapped = Vec::new();
                for (i, (&old_addr, answer)) in queries.iter().zip(&expected).enumerate() {
                    if i % 7 == 0 {
                        batch.push(None);
                        remapped.push(None);
                    }
                    batch.push(Some(old_addr));
                    remapped.push(answer.unwrap_or(Some(old_addr)));
                }
                if reversed {
                    batch.reverse();
                    remapped.reverse();
                }
                remap.remap_in_place(&mut batch);
                prop_assert_eq!(&batch, &remapped, "seed {}", seed);
            }
        }
        Ok(())
    }

    proptest::proptest! {
        #![proptest_config(cases_for(SHORT_CHAINS))]
        #[test]
        fn test_short_chain_matches_materialized_steps(
            seed in proptest::num::u64::ANY,
            num_steps in SHORT_CHAINS,
            splits in proptest::collection::vec(0..LONG_INDEXED_CHAINS.end, 0..3),
        ) {
            check_chain_matches_materialized_steps(seed, num_steps, splits)?;
        }
    }

    proptest::proptest! {
        #![proptest_config(cases_for(INDEXED_CHAINS))]
        #[test]
        fn test_indexed_chain_matches_materialized_steps(
            seed in proptest::num::u64::ANY,
            num_steps in INDEXED_CHAINS,
            splits in proptest::collection::vec(0..LONG_INDEXED_CHAINS.end, 0..3),
        ) {
            check_chain_matches_materialized_steps(seed, num_steps, splits)?;
        }
    }

    proptest::proptest! {
        #![proptest_config(cases_for(LONG_INDEXED_CHAINS))]
        #[test]
        fn test_long_indexed_chain_matches_materialized_steps(
            seed in proptest::num::u64::ANY,
            num_steps in LONG_INDEXED_CHAINS,
            splits in proptest::collection::vec(0..LONG_INDEXED_CHAINS.end, 0..3),
        ) {
            check_chain_matches_materialized_steps(seed, num_steps, splits)?;
        }
    }

    proptest::proptest! {
        /// The compact remap must answer exactly like a materialized old-to-new map. A
        /// wrong answer here silently points an index at the wrong physical row, so this
        /// compares the two forms address by address over randomized rewrites.
        #[test]
        fn test_compact_matches_direct_over_random_rewrites(
            // Fragment id shapes: dense, strided by a power of two (the integer hasher's
            // worst case), and sparse.
            frag_seeds in proptest::collection::vec((0..3usize, 0..64u32), 1..7),
            rows_per_frag in 1..24u32,
            keep in proptest::collection::vec(proptest::bool::weighted(0.75), 6 * 24),
            gaps in proptest::collection::vec(1..4u32, 1..5),
        ) {
            let mut old_frag_ids: Vec<u32> = Vec::new();
            for (shape, seed) in &frag_seeds {
                let frag_id = match shape {
                    0 => *seed,
                    1 => (*seed + 1) << 20,
                    _ => seed.wrapping_mul(7919),
                };
                if !old_frag_ids.contains(&frag_id) {
                    old_frag_ids.push(frag_id);
                }
            }

            // Rewritten rows are read fragment by fragment in `old_frag_ids` order, and
            // by ascending offset within each fragment.
            let mut rewritten = RoaringTreemap::new();
            let mut read_order: Vec<u64> = Vec::new();
            let mut covered: Vec<u64> = Vec::new();
            for (frag_index, frag_id) in old_frag_ids.iter().enumerate() {
                for offset in 0..rows_per_frag {
                    covered.push(addr(*frag_id, offset));
                    if keep[frag_index * rows_per_frag as usize + offset as usize] {
                        rewritten.insert(addr(*frag_id, offset));
                        read_order.push(addr(*frag_id, offset));
                    }
                }
            }
            let total_rewritten = read_order.len() as u32;
            proptest::prop_assume!(total_rewritten > 0);

            // Spread the rewritten rows over ascending new fragment ids, every fragment
            // non-empty so the row counts still add up.
            let parts = gaps.len().min(total_rewritten as usize);
            let base = total_rewritten / parts as u32;
            let remainder = total_rewritten % parts as u32;
            let mut new_frags: Vec<(u32, u32)> = Vec::with_capacity(parts);
            let mut new_frag_id = 10_000u32;
            for (part, gap) in gaps.iter().take(parts).enumerate() {
                let rows = base + u32::from((part as u32) < remainder);
                new_frags.push((new_frag_id, rows));
                new_frag_id += gap;
            }

            // The same mapping, materialized one address at a time.
            let mut direct: HashMap<u64, Option<u64>> = HashMap::new();
            let mut read_order_iter = read_order.iter();
            for (new_frag_id, rows) in &new_frags {
                for offset in 0..*rows {
                    let old = read_order_iter.next().unwrap();
                    direct.insert(*old, Some(addr(*new_frag_id, offset)));
                }
            }
            prop_assert!(read_order_iter.next().is_none());
            for old in &covered {
                direct.entry(*old).or_insert(None);
            }

            let compact = RowAddrRemap::compact([GroupInput {
                rewritten_old_row_addrs: rewritten,
                old_frag_ids: old_frag_ids.clone(),
                new_frags,
            }])
            .unwrap();
            let direct = RowAddrRemap::direct(direct);

            for old in &covered {
                prop_assert_eq!(
                    compact.get(*old),
                    direct.get(*old),
                    "address {:#x} in fragments {:?}",
                    old,
                    old_frag_ids
                );
            }
            // Fragments no group covers are left alone.
            for frag_id in [0xFFFF_FFF0u32, 0x7EEE_EEEE] {
                prop_assert_eq!(compact.get(addr(frag_id, 0)), None);
            }
            prop_assert_eq!(compact.affected_fragments(), direct.affected_fragments());
        }
    }
}
