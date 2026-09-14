// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::RangeInclusive;
use std::sync::Arc;

use super::bitmap::{Bitmap, count_ones};
use super::{RowIdSequence, U64Segment};
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result};
use rangemap::RangeInclusiveMap;

/// Fragments one lookup may have to probe before the merged map is worth its
/// build, whatever that build costs. A compacted table interleaves its
/// fragments, and one measured at 46.
const MAX_PROBE_DEPTH: u64 = 64;

/// Row ids the merged build may read before probing is worth its per-lookup
/// cost instead.
const MERGE_ROWS_BUDGET: u64 = 1 << 20;

/// Bitmap bytes summarized by one entry of a bitmap rank directory. A lookup
/// counts at most this many bytes; the directory costs one `u32` per block,
/// 1/64 of the bitmap.
const RANK_BLOCK_BYTES: usize = 256;

/// An index of row ids
///
/// This index is used to map row ids to their corresponding addresses. These
/// addresses correspond to physical positions in the dataset. See [RowAddress].
///
/// This structure only contains rows that physically exist. However, it may
/// map to addresses that have been tombstoned. A separate tombstone index is
/// used to track tombstoned rows.
// (Implementation)
// Two representations answer the same lookups, chosen once by `new`. The merged
// map keys disjoint ranges of row ids to a pair of segments, the row ids and
// the addresses, and reads every row id to build. A probe instead reads each
// segment's bounds and asks the covering segment for the position of the id;
// `new` takes that when few fragments cover one id and the merged build would
// read a lot of them.
#[derive(Debug)]
pub struct RowIdIndex {
    /// Fragments that hold at least one row id, sorted by their lowest row id.
    fragments: Vec<FragmentEntry>,
    /// Max-`end` heap over `fragments`: `end_tree[1]` is the root and leaf `i`
    /// sits at `end_tree[len() / 2 + i]`.
    end_tree: Vec<u64>,
    merged: Option<MergedIndex>,
}

type MergedIndex = RangeInclusiveMap<u64, (U64Segment, U64Segment)>;

pub struct FragmentRowIdIndex {
    pub fragment_id: u32,
    pub row_id_sequence: Arc<RowIdSequence>,
    pub deletion_vector: Arc<DeletionVector>,
}

impl RowIdIndex {
    /// Create a new index from a list of fragment ids and their corresponding row id sequences.
    pub fn new(fragment_indices: &[FragmentRowIdIndex]) -> Result<Self> {
        let mut fragments: Vec<FragmentEntry> = fragment_indices
            .iter()
            .filter_map(FragmentEntry::new)
            .collect();
        fragments.sort_unstable_by_key(|entry| entry.start);

        let mut index = Self {
            end_tree: build_end_tree(&fragments),
            fragments,
            merged: None,
        };
        if !probing_beats_merging(&index.fragments) {
            index.merged = Some(index.build_merged()?);
        }
        Ok(index)
    }

    fn build_merged(&self) -> Result<MergedIndex> {
        let sources: Vec<FragmentRowIdIndex> = self
            .fragments
            .iter()
            .map(|entry| FragmentRowIdIndex {
                fragment_id: entry.fragment_id,
                row_id_sequence: entry.sequence.clone(),
                deletion_vector: entry.deletion_vector.clone(),
            })
            .collect();
        let chunks = sources
            .iter()
            .flat_map(decompose_sequence)
            .collect::<Vec<_>>();

        let mut final_chunks = Vec::new();
        for processed_chunk in prep_index_chunks(chunks) {
            match processed_chunk {
                RawIndexChunk::NonOverlapping(chunk) => {
                    final_chunks.push(chunk);
                }
                RawIndexChunk::Overlapping(_range, overlapping_chunks) => {
                    // Intersecting row-id ranges don't imply intersecting id sets;
                    // sparse ids and deletion holes leave the union short of the span.
                    // The real invariant (no id in two fragments) is checked in the merge.
                    let merged_chunk = merge_overlapping_chunks(overlapping_chunks)?;
                    final_chunks.push(merged_chunk);
                }
            }
        }

        Ok(RangeInclusiveMap::from_iter(final_chunks))
    }

    /// Get the address for a given row id.
    ///
    /// Will return None if the row id does not exist in the index.
    ///
    /// # Errors
    ///
    /// Returns an error if the row id is live in more than one fragment,
    /// which means the stable row ids are corrupt.
    pub fn get(&self, row_id: u64) -> Result<Option<RowAddress>> {
        if let Some(merged) = &self.merged {
            return Ok(merged_get(merged, row_id));
        }
        self.probe(row_id)
    }

    /// Get addresses for many row ids in one pass over the index.
    ///
    /// Returns one entry per input id, in input order (`None` for missing).
    /// Sorts a working copy of the input internally so the chunk iterator
    /// is advanced at most once per chunk, amortizing the per-id tree walk
    /// from O(N · log F) to O(F + N).
    ///
    /// # Errors
    ///
    /// Returns an error if any requested row id is live in more than one
    /// fragment, which means the stable row ids are corrupt.
    pub fn get_many(&self, row_ids: &[u64]) -> Result<Vec<Option<RowAddress>>> {
        let n = row_ids.len();
        let mut out = vec![None; n];
        if n == 0 {
            return Ok(out);
        }

        let mut sorted: Vec<(u64, usize)> = row_ids.iter().copied().zip(0..n).collect();
        sorted.sort_unstable_by_key(|&(id, _)| id);

        let Some(merged) = &self.merged else {
            // Sorted ids keep one fragment and its segments warm across the run.
            for (id, orig_idx) in sorted {
                out[orig_idx] = self.probe(id)?;
            }
            return Ok(out);
        };

        let mut chunks = merged.iter().peekable();
        for (id, orig_idx) in sorted {
            // Advance past chunks that end before this id.
            while let Some((range, _)) = chunks.peek() {
                if *range.end() < id {
                    chunks.next();
                } else {
                    break;
                }
            }
            let Some((range, (row_id_seg, addr_seg))) = chunks.peek() else {
                break;
            };
            if id < *range.start() {
                continue; // falls in a gap between chunks
            }
            if let Some(pos) = row_id_seg.position(id)
                && let Some(addr) = addr_seg.get(pos)
            {
                out[orig_idx] = Some(RowAddress::from(addr));
            }
        }
        Ok(out)
    }

    /// Address of `row_id`, from the fragment that holds it live. Descends the
    /// max-`end` tree, so a fragment out of reach of the id costs nothing.
    ///
    /// Visits every candidate rather than stopping at the first hit, and
    /// errors when a second fragment holds the id live.
    fn probe(&self, row_id: u64) -> Result<Option<RowAddress>> {
        let fragments = self.fragments.len();
        if fragments == 0 {
            return Ok(None);
        }
        // Only a fragment that starts at or below the id can hold it.
        let upper = self
            .fragments
            .partition_point(|entry| entry.start <= row_id);
        if upper == 0 {
            return Ok(None);
        }
        let mut found: Option<RowAddress> = None;
        // Depth-first, left to right, without a stack: `node` covers `width`
        // slots from `lo`. A pruned subtree or a visited leaf moves on to the
        // next subtree: up while `node` is a right child, then over to the
        // sibling.
        let (mut node, mut lo, mut width) = (1usize, 0usize, self.end_tree.len() / 2);
        loop {
            if lo >= upper {
                // Every slot from here on starts above the id.
                break;
            }
            if self.end_tree[node] >= row_id {
                if width > 1 {
                    node *= 2;
                    width /= 2;
                    continue;
                }
                if lo < fragments
                    && let Some(candidate) = self.fragments[lo].resolve(row_id)
                {
                    if found.is_some() {
                        return Err(Error::internal(format!(
                            "row id index corrupt: stable row id {row_id} is \
                             live in multiple fragments",
                        )));
                    }
                    found = Some(candidate);
                }
            }
            while node & 1 == 1 {
                if node == 1 {
                    return Ok(found);
                }
                node /= 2;
                lo -= width;
                width *= 2;
            }
            node += 1;
            lo += width;
        }
        Ok(found)
    }
}

fn merged_get(merged: &MergedIndex, row_id: u64) -> Option<RowAddress> {
    let (row_id_segment, address_segment) = merged.get(&row_id)?;
    let pos = row_id_segment.position(row_id)?;
    let address = address_segment.get(pos)?;
    Some(RowAddress::from(address))
}

/// One segment of a sequence, and the offset its first row sits at.
#[derive(Debug)]
struct SegmentEntry {
    seq_idx: usize,
    range: RangeInclusive<u64>,
    start_offset: u32,
    lookup: SegmentLookup,
}

/// How a [`SegmentEntry`] finds the position of a row id.
#[derive(Debug)]
enum SegmentLookup {
    /// The encoding searches itself in logarithmic or constant time.
    Native,
    /// Row id to position for an unsorted [`U64Segment::Array`], whose own
    /// `position` scans.
    Positions(Vec<(u64, u32)>),
    /// Set bits before each [`RANK_BLOCK_BYTES`] block of a
    /// [`U64Segment::RangeWithBitmap`] bitmap, whose own `position` counts
    /// the whole prefix.
    BitmapRank(Vec<u32>),
}

impl SegmentEntry {
    /// Position of `row_id` in this segment, or `None` if it holds no such id.
    fn position(&self, sequence: &RowIdSequence, row_id: u64) -> Option<usize> {
        let segment = &sequence.0[self.seq_idx];
        match &self.lookup {
            SegmentLookup::Native => segment.position(row_id),
            SegmentLookup::Positions(positions) => positions
                .binary_search_by_key(&row_id, |(id, _)| *id)
                .ok()
                .map(|found| positions[found].1 as usize),
            SegmentLookup::BitmapRank(rank) => {
                let U64Segment::RangeWithBitmap { range, bitmap } = segment else {
                    return None;
                };
                if !range.contains(&row_id) {
                    return None;
                }
                let offset = (row_id - range.start) as usize;
                if !bitmap.get(offset) {
                    return None;
                }
                let block_start = offset / (RANK_BLOCK_BYTES * 8) * (RANK_BLOCK_BYTES * 8);
                let ones_before = rank[offset / (RANK_BLOCK_BYTES * 8)] as usize
                    + bitmap.slice(block_start, offset - block_start).count_ones();
                Some(ones_before)
            }
        }
    }
}

/// The lookup structure for `segment`, and the ids it holds. A bitmap's rank
/// directory counts its ones on the way, so the segment is read once.
fn build_lookup(segment: &U64Segment) -> (SegmentLookup, usize) {
    match segment {
        U64Segment::Array(_) => {
            let mut positions: Vec<(u64, u32)> = segment
                .iter()
                .enumerate()
                .map(|(position, row_id)| (row_id, position as u32))
                .collect();
            positions.sort_unstable();
            // The first position of a repeated id wins, as `position` does.
            positions.dedup_by_key(|(row_id, _)| *row_id);
            (SegmentLookup::Positions(positions), segment.len())
        }
        U64Segment::RangeWithBitmap { bitmap, .. } => {
            let (rank, ones) = build_bitmap_rank(bitmap);
            (SegmentLookup::BitmapRank(rank), ones)
        }
        _ => (SegmentLookup::Native, segment.len()),
    }
}

/// Set bits before each [`RANK_BLOCK_BYTES`] block of `bitmap`, and in total.
fn build_bitmap_rank(bitmap: &Bitmap) -> (Vec<u32>, usize) {
    let mut rank = Vec::with_capacity(bitmap.data.len().div_ceil(RANK_BLOCK_BYTES));
    let mut ones = 0usize;
    for block in bitmap.data.chunks(RANK_BLOCK_BYTES) {
        rank.push(ones as u32);
        ones += count_ones(block);
    }
    (rank, ones)
}

#[derive(Debug)]
struct FragmentEntry {
    fragment_id: u32,
    sequence: Arc<RowIdSequence>,
    deletion_vector: Arc<DeletionVector>,
    segments: Vec<SegmentEntry>,
    start: u64,
    end: u64,
    /// Row ids the merged build reads one by one.
    merge_rows: u64,
}

impl FragmentEntry {
    fn new(source: &FragmentRowIdIndex) -> Option<Self> {
        let mut segments: Vec<SegmentEntry> = Vec::new();
        let mut start_offset: u32 = 0;
        let mut merge_rows: u64 = 0;
        let deleted = !source.deletion_vector.is_empty();
        for (seq_idx, segment) in source.row_id_sequence.0.iter().enumerate() {
            let (lookup, len) = build_lookup(segment);
            // A `Range` without deletions decomposes in constant time.
            if deleted || !matches!(segment, U64Segment::Range(_)) {
                merge_rows += len as u64;
            }
            // `range()` reports the span of a holed encoding, so ask `len` which
            // ids the segment actually holds before trusting those bounds.
            if len > 0
                && let Some(range) = segment.range()
            {
                segments.push(SegmentEntry {
                    seq_idx,
                    range,
                    start_offset,
                    lookup,
                });
            }
            start_offset += len as u32;
        }
        let start = segments.iter().map(|entry| *entry.range.start()).min()?;
        let end = segments.iter().map(|entry| *entry.range.end()).max()?;
        Some(Self {
            fragment_id: source.fragment_id,
            sequence: source.row_id_sequence.clone(),
            deletion_vector: source.deletion_vector.clone(),
            segments,
            start,
            end,
            merge_rows,
        })
    }

    /// Address of `row_id` here, or `None` when the fragment lacks it or holds
    /// it deleted.
    fn resolve(&self, row_id: u64) -> Option<RowAddress> {
        for entry in &self.segments {
            if !entry.range.contains(&row_id) {
                continue;
            }
            let Some(position) = entry.position(&self.sequence, row_id) else {
                continue;
            };
            let row_offset = entry.start_offset + position as u32;
            if self.deletion_vector.contains(row_offset) {
                continue;
            }
            return Some(RowAddress::new_from_parts(self.fragment_id, row_offset));
        }
        None
    }
}

/// Whether to answer lookups by probing the fragments rather than by merging
/// every row id.
///
/// Probing costs the fragments that cover one id, per lookup; merging costs the
/// row ids it reads, once. So probe only when both stay on the right side of
/// [`MAX_PROBE_DEPTH`] and [`MERGE_ROWS_BUDGET`].
fn probing_beats_merging(fragments: &[FragmentEntry]) -> bool {
    let merge_rows: u64 = fragments.iter().map(|entry| entry.merge_rows).sum();
    merge_rows > MERGE_ROWS_BUDGET && max_overlap_depth(fragments) <= MAX_PROBE_DEPTH
}

/// Most fragments that cover any one row id.
fn max_overlap_depth(fragments: &[FragmentEntry]) -> u64 {
    let mut ends: Vec<u64> = fragments.iter().map(|entry| entry.end).collect();
    ends.sort_unstable();
    let mut closed = 0;
    let mut depth: u64 = 0;
    for (opened, entry) in fragments.iter().enumerate() {
        while closed < ends.len() && ends[closed] < entry.start {
            closed += 1;
        }
        depth = depth.max((opened + 1 - closed) as u64);
    }
    depth
}

/// Implicit max-`end` heap over `fragments`, padded to a power of two. Padding
/// leaves hold 0, which prunes for every id above 0 and is filtered by slot.
fn build_end_tree(fragments: &[FragmentEntry]) -> Vec<u64> {
    if fragments.is_empty() {
        return Vec::new();
    }
    let leaves = fragments.len().next_power_of_two();
    let mut tree = vec![0_u64; 2 * leaves];
    for (slot, entry) in fragments.iter().enumerate() {
        tree[leaves + slot] = entry.end;
    }
    for node in (1..leaves).rev() {
        tree[node] = tree[2 * node].max(tree[2 * node + 1]);
    }
    tree
}

impl DeepSizeOf for RowIdIndex {
    /// Charges the sequences and deletion vectors the `Arc`s keep alive, which
    /// a sequence cached under its own key is charged for as well.
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        let fragment_bytes: usize = self
            .fragments
            .iter()
            .map(|entry| {
                entry.sequence.deep_size_of_children(context)
                    + entry.deletion_vector.deep_size_of_children(context)
                    + entry.segments.capacity() * std::mem::size_of::<SegmentEntry>()
                    + entry
                        .segments
                        .iter()
                        .map(|segment| match &segment.lookup {
                            SegmentLookup::Native => 0,
                            SegmentLookup::Positions(positions) => {
                                positions.capacity() * std::mem::size_of::<(u64, u32)>()
                            }
                            SegmentLookup::BitmapRank(rank) => {
                                rank.capacity() * std::mem::size_of::<u32>()
                            }
                        })
                        .sum::<usize>()
            })
            .sum();
        let merged_bytes: usize = self
            .merged
            .as_ref()
            .map(|merged| {
                merged
                    .iter()
                    .map(|(_, (row_id_segment, address_segment))| {
                        (2 * std::mem::size_of::<u64>())
                            + std::mem::size_of::<(U64Segment, U64Segment)>()
                            + row_id_segment.deep_size_of_children(context)
                            + address_segment.deep_size_of_children(context)
                    })
                    .sum()
            })
            .unwrap_or(0);
        fragment_bytes
            + merged_bytes
            + self.fragments.capacity() * std::mem::size_of::<FragmentEntry>()
            + self.end_tree.capacity() * std::mem::size_of::<u64>()
    }
}

fn decompose_sequence(
    frag_index: &FragmentRowIdIndex,
) -> Vec<(RangeInclusive<u64>, (U64Segment, U64Segment))> {
    let mut start_address: u64 = RowAddress::first_row(frag_index.fragment_id).into();
    let mut current_offset = 0u32;
    let no_deletions = frag_index.deletion_vector.is_empty();

    frag_index
        .row_id_sequence
        .0
        .iter()
        .filter_map(|segment| {
            let segment_len = segment.len();

            let result = if no_deletions {
                decompose_segment_no_deletions(segment, start_address)
            } else {
                decompose_segment_with_deletions(
                    segment,
                    start_address,
                    current_offset,
                    &frag_index.deletion_vector,
                )
            };

            current_offset += segment_len as u32;
            start_address += segment_len as u64;

            result
        })
        .collect()
}

/// Build an IndexChunk from a list of (row_id, address) pairs.
fn build_chunk_from_pairs(mut pairs: Vec<(u64, u64)>) -> Option<IndexChunk> {
    if pairs.is_empty() {
        return None;
    }
    // Sorted, so the row id segment encodes as one a lookup can search rather
    // than an `Array` it has to scan. The address segment follows the pairing.
    pairs.sort_unstable_by_key(|(row_id, _)| *row_id);
    let (row_ids, addresses): (Vec<u64>, Vec<u64>) = pairs.into_iter().unzip();
    let row_id_segment = U64Segment::from_iter(row_ids);
    let address_segment = U64Segment::from_iter(addresses);
    let coverage = row_id_segment.range()?;
    Some((coverage, (row_id_segment, address_segment)))
}

/// Fast path: no deletions. O(1) for Range segments.
fn decompose_segment_no_deletions(segment: &U64Segment, start_address: u64) -> Option<IndexChunk> {
    match segment {
        U64Segment::Range(range) if !range.is_empty() => {
            let len = range.end - range.start;
            let row_id_segment = U64Segment::Range(range.clone());
            let address_segment = U64Segment::Range(start_address..start_address + len);
            let coverage = range.start..=range.end - 1;
            Some((coverage, (row_id_segment, address_segment)))
        }
        _ if segment.is_empty() => None,
        _ => {
            // Non-Range segments: must iterate to build address mapping.
            let pairs: Vec<(u64, u64)> = segment
                .iter()
                .enumerate()
                .map(|(i, row_id)| (row_id, start_address + i as u64))
                .collect();
            build_chunk_from_pairs(pairs)
        }
    }
}

/// Slow path: has deletions, must check each row.
fn decompose_segment_with_deletions(
    segment: &U64Segment,
    start_address: u64,
    current_offset: u32,
    deletion_vector: &DeletionVector,
) -> Option<IndexChunk> {
    let pairs: Vec<(u64, u64)> = segment
        .iter()
        .enumerate()
        .filter_map(|(i, row_id)| {
            let row_offset = current_offset + i as u32;
            if !deletion_vector.contains(row_offset) {
                Some((row_id, start_address + i as u64))
            } else {
                None
            }
        })
        .collect();
    build_chunk_from_pairs(pairs)
}

type IndexChunk = (RangeInclusive<u64>, (U64Segment, U64Segment));

#[derive(Debug)]
enum RawIndexChunk {
    NonOverlapping(IndexChunk),
    Overlapping(RangeInclusive<u64>, Vec<IndexChunk>),
}

impl RawIndexChunk {
    fn range_end(&self) -> u64 {
        match self {
            Self::NonOverlapping((range, _)) => *range.end(),
            Self::Overlapping(range, _) => *range.end(),
        }
    }
}

/// Given a vector of index chunks, sort them and return an iterator of index chunks.
///
/// The iterator will yield chunks that are non-overlapping or a set of chunks
/// that are overlapping.
fn prep_index_chunks(mut chunks: Vec<IndexChunk>) -> impl Iterator<Item = RawIndexChunk> {
    chunks.sort_by_key(|(range, _)| u64::MAX - *range.start());

    let mut output = Vec::new();

    // Start assuming non-overlapping in first chunk.
    if let Some(first_chunk) = chunks.pop() {
        output.push(RawIndexChunk::NonOverlapping(first_chunk));
    } else {
        // Early return for empty.
        return output.into_iter();
    }

    let mut current_range = 0..=0;
    let mut current_overlap = Vec::new();
    while let Some(chunk) = chunks.pop() {
        debug_assert_eq!(
            current_overlap
                .iter()
                .map(|(range, _): &IndexChunk| *range.start())
                .min()
                .unwrap_or_default(),
            *current_range.start(),
        );
        debug_assert_eq!(
            current_overlap
                .iter()
                .map(|(range, _): &IndexChunk| *range.end())
                .max()
                .unwrap_or_default(),
            *current_range.end(),
        );

        if current_overlap.is_empty() {
            // We haven't found overlap yet.
            let last_chunk_end = output.last().unwrap().range_end();
            if *chunk.0.start() <= last_chunk_end {
                // We have found overlap.
                match output.pop().unwrap() {
                    RawIndexChunk::NonOverlapping(chunk) => {
                        current_overlap.push(chunk);
                    }
                    _ => unreachable!(),
                }
                current_overlap.push(chunk);

                let range_start = *current_overlap.first().unwrap().0.start();
                let range_end = *current_overlap
                    .last()
                    .unwrap()
                    .0
                    .end()
                    .max(current_overlap.first().unwrap().0.end());
                current_range = range_start..=range_end;
            } else {
                // We are still in non-overlapping space.
                output.push(RawIndexChunk::NonOverlapping(chunk));
            }
        } else {
            // We are making an overlap chunk
            if chunk.0.start() <= current_range.end() {
                // We are still in overlap.
                let range_end = *chunk.0.end().max(current_range.end());
                current_range = *current_range.start()..=range_end;

                current_overlap.push(chunk);
            } else {
                // We have exited overlap.
                output.push(RawIndexChunk::Overlapping(
                    std::mem::replace(&mut current_range, 0..=0),
                    std::mem::take(&mut current_overlap),
                ));
                output.push(RawIndexChunk::NonOverlapping(chunk));
            }
        }
    }
    debug_assert_eq!(
        current_overlap
            .iter()
            .map(|(range, _): &IndexChunk| *range.start())
            .min()
            .unwrap_or_default(),
        *current_range.start(),
    );
    debug_assert_eq!(
        current_overlap
            .iter()
            .map(|(range, _): &IndexChunk| *range.end())
            .max()
            .unwrap_or_default(),
        *current_range.end(),
    );

    if !current_overlap.is_empty() {
        output.push(RawIndexChunk::Overlapping(
            current_range.clone(),
            current_overlap,
        ));
    }

    output.into_iter()
}

fn merge_overlapping_chunks(overlapping_chunks: Vec<IndexChunk>) -> Result<IndexChunk> {
    let total_capacity = overlapping_chunks
        .iter()
        .map(|(_, (row_ids, _))| row_ids.len())
        .sum();
    let mut values = Vec::with_capacity(total_capacity);
    for (_, (row_ids, row_addrs)) in overlapping_chunks.iter() {
        values.extend(row_ids.iter().zip(row_addrs.iter()));
    }
    values.sort_by_key(|(row_id, _)| *row_id);
    // A duplicate row id here means two fragments claim the same live id: a
    // corrupt index, not a resolvable sparse-coverage case.
    if let Some(w) = values.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(Error::internal(format!(
            "row id index corrupt: stable row id {} is live in multiple fragments",
            w[0].0
        )));
    }
    let row_id_segment = U64Segment::from_iter(values.iter().map(|(row_id, _)| *row_id));
    let address_segment = U64Segment::from_iter(values.iter().map(|(_, row_addr)| *row_addr));

    let range = row_id_segment.range().unwrap();

    Ok((range, (row_id_segment, address_segment)))
}

#[cfg(test)]
impl RowIdIndex {
    /// Index that answers from the probe path, whatever the gate decided.
    fn probing(fragment_indices: &[FragmentRowIdIndex]) -> Result<Self> {
        let mut fragments: Vec<FragmentEntry> = fragment_indices
            .iter()
            .filter_map(FragmentEntry::new)
            .collect();
        fragments.sort_unstable_by_key(|entry| entry.start);
        Ok(Self {
            end_tree: build_end_tree(&fragments),
            fragments,
            merged: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{
        prelude::{Just, Strategy, any},
        prop_assert, prop_assert_eq,
    };

    /// Sequence of `len` even row ids, held as a sorted array.
    fn sparse_sequence(len: u64) -> RowIdSequence {
        RowIdSequence(vec![U64Segment::SortedArray(
            (0..len).map(|value| value * 2).collect::<Vec<u64>>().into(),
        )])
    }

    fn fragment(fragment_id: u32, sequence: RowIdSequence) -> FragmentRowIdIndex {
        FragmentRowIdIndex {
            fragment_id,
            row_id_sequence: Arc::new(sequence),
            deletion_vector: Arc::new(DeletionVector::default()),
        }
    }

    #[test]
    fn test_new_builds_the_merged_map_unless_probing_wins() {
        // Ranges decompose in constant time, and a small sequence is cheap to
        // read whatever its encoding.
        let ranges = fragment(1, RowIdSequence(vec![U64Segment::Range(0..1_000_000)]));
        assert!(RowIdIndex::new(&[ranges]).unwrap().merged.is_some());
        let small = fragment(1, sparse_sequence(16));
        assert!(RowIdIndex::new(&[small]).unwrap().merged.is_some());

        // Past the row budget, with one fragment covering any id.
        let wide = fragment(1, sparse_sequence(MERGE_ROWS_BUDGET + 1));
        let index = RowIdIndex::new(&[wide]).unwrap();
        assert!(index.merged.is_none());
        assert_eq!(
            index.get(6).unwrap(),
            Some(RowAddress::new_from_parts(1, 3))
        );
    }

    #[test]
    fn test_deep_overlap_merges_however_many_rows_it_reads() {
        // Just past the row budget in total, interleaved so every fragment
        // covers every id: the depth alone forces the merged build.
        let fragments = MAX_PROBE_DEPTH + 1;
        let rows_per_fragment = MERGE_ROWS_BUDGET / fragments + 1;
        let deep: Vec<FragmentRowIdIndex> = (0..fragments as u32)
            .map(|id| {
                let ids: Vec<u64> = (0..rows_per_fragment)
                    .map(|value| value * fragments + id as u64)
                    .collect();
                fragment(id, RowIdSequence(vec![U64Segment::SortedArray(ids.into())]))
            })
            .collect();

        assert!(RowIdIndex::new(&deep).unwrap().merged.is_some());
    }

    #[test]
    fn test_probe_resolves_a_row_id_the_merged_map_rejects() {
        let sources = [
            fragment(1, RowIdSequence::from(&[0, 2][..])),
            fragment(2, RowIdSequence::from(&[1, 2][..])),
        ];
        assert!(RowIdIndex::new(&sources).is_err());

        let index = RowIdIndex::probing(&sources[..1]).unwrap();
        assert_eq!(
            index.get(2).unwrap(),
            Some(RowAddress::new_from_parts(1, 1))
        );
    }

    #[test]
    fn test_probe_errors_when_two_fragments_hold_an_id_live() {
        let sources = [
            fragment(1, RowIdSequence::from(&[0, 2][..])),
            fragment(2, RowIdSequence::from(&[1, 2][..])),
        ];
        let index = RowIdIndex::probing(&sources).unwrap();
        assert_eq!(
            index.get(0).unwrap(),
            Some(RowAddress::new_from_parts(1, 0))
        );

        let error = index.get(2).unwrap_err();
        assert!(matches!(&error, Error::Internal { .. }));
        assert!(
            error
                .to_string()
                .contains("stable row id 2 is live in multiple fragments")
        );

        let error = index.get_many(&[0, 2]).unwrap_err();
        assert!(matches!(&error, Error::Internal { .. }));
    }

    #[test]
    fn test_probe_ranks_a_bitmap_wider_than_one_block() {
        // Three rank blocks and a partial fourth, with every third slot a hole,
        // so each lookup combines a directory entry with a partial popcount.
        let span = (RANK_BLOCK_BYTES * 8 * 3 + 17) as u64;
        let present: Vec<bool> = (0..span).map(|slot| slot % 3 != 1).collect();
        let segment = U64Segment::RangeWithBitmap {
            range: 1000..1000 + span,
            bitmap: present.as_slice().into(),
        };
        let index = RowIdIndex::probing(&[fragment(7, RowIdSequence(vec![segment]))]).unwrap();
        let mut position = 0;
        for slot in 0..span {
            let found = index.get(1000 + slot).unwrap();
            if slot % 3 == 1 {
                assert_eq!(found, None);
            } else {
                assert_eq!(found, Some(RowAddress::new_from_parts(7, position)));
                position += 1;
            }
        }
        assert_eq!(index.get(999).unwrap(), None);
        assert_eq!(index.get(1000 + span).unwrap(), None);
    }

    #[test]
    fn test_probe_finds_every_position_of_an_unsorted_array() {
        let row_ids: Vec<u64> = (0..2048).map(|value| (value * 7919) % 2048).collect();
        let index = RowIdIndex::probing(&[fragment(
            3,
            RowIdSequence(vec![U64Segment::Array(row_ids.clone().into())]),
        )])
        .unwrap();
        for (offset, row_id) in row_ids.iter().enumerate() {
            assert_eq!(
                index.get(*row_id).unwrap(),
                Some(RowAddress::new_from_parts(3, offset as u32))
            );
        }
        assert!(index.merged.is_none());
    }

    #[test]
    fn test_new_index() {
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 10,
                row_id_sequence: Arc::new(RowIdSequence(vec![
                    U64Segment::Range(0..10),
                    U64Segment::RangeWithHoles {
                        range: 10..17,
                        holes: vec![12, 15].into(),
                    },
                    U64Segment::SortedArray(vec![20, 25, 30].into()),
                ])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 20,
                row_id_sequence: Arc::new(RowIdSequence(vec![
                    U64Segment::RangeWithBitmap {
                        range: 17..20,
                        bitmap: [true, false, true].as_slice().into(),
                    },
                    U64Segment::Array(vec![40, 50, 60].into()),
                ])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        // Check various queries.
        assert_eq!(
            index.get(0).unwrap(),
            Some(RowAddress::new_from_parts(10, 0))
        );
        assert_eq!(index.get(15).unwrap(), None);
        assert_eq!(
            index.get(16).unwrap(),
            Some(RowAddress::new_from_parts(10, 14))
        );
        assert_eq!(
            index.get(17).unwrap(),
            Some(RowAddress::new_from_parts(20, 0))
        );
        assert_eq!(
            index.get(25).unwrap(),
            Some(RowAddress::new_from_parts(10, 16))
        );
        assert_eq!(
            index.get(40).unwrap(),
            Some(RowAddress::new_from_parts(20, 2))
        );
        assert_eq!(
            index.get(60).unwrap(),
            Some(RowAddress::new_from_parts(20, 4))
        );
        assert_eq!(index.get(61).unwrap(), None);
    }

    #[test]
    fn test_new_index_overlap() {
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 23,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::SortedArray(
                    vec![3, 6, 9].into(),
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 42,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::SortedArray(
                    vec![2, 5, 8].into(),
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 10,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::SortedArray(
                    vec![1, 4, 7].into(),
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        // Check various queries.
        assert_eq!(
            index.get(1).unwrap(),
            Some(RowAddress::new_from_parts(10, 0))
        );
        assert_eq!(
            index.get(2).unwrap(),
            Some(RowAddress::new_from_parts(42, 0))
        );
        assert_eq!(
            index.get(3).unwrap(),
            Some(RowAddress::new_from_parts(23, 0))
        );
        assert_eq!(
            index.get(4).unwrap(),
            Some(RowAddress::new_from_parts(10, 1))
        );
        assert_eq!(
            index.get(5).unwrap(),
            Some(RowAddress::new_from_parts(42, 1))
        );
        assert_eq!(
            index.get(6).unwrap(),
            Some(RowAddress::new_from_parts(23, 1))
        );
        assert_eq!(
            index.get(7).unwrap(),
            Some(RowAddress::new_from_parts(10, 2))
        );
        assert_eq!(
            index.get(8).unwrap(),
            Some(RowAddress::new_from_parts(42, 2))
        );
        assert_eq!(
            index.get(9).unwrap(),
            Some(RowAddress::new_from_parts(23, 2))
        );
    }

    #[test]
    fn test_new_index_unsorted_row_ids() {
        // Test case with unsorted row ids within fragments
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 10,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Array(
                    vec![9, 3, 6].into(), // Unsorted array
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 20,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Array(
                    vec![8, 2, 5].into(), // Unsorted array
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 30,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Array(
                    vec![7, 1, 4].into(), // Unsorted array
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        // Check that all row ids can be found regardless of their order in the segments
        assert_eq!(
            index.get(1).unwrap(),
            Some(RowAddress::new_from_parts(30, 1))
        );
        assert_eq!(
            index.get(2).unwrap(),
            Some(RowAddress::new_from_parts(20, 1))
        );
        assert_eq!(
            index.get(3).unwrap(),
            Some(RowAddress::new_from_parts(10, 1))
        );
        assert_eq!(
            index.get(4).unwrap(),
            Some(RowAddress::new_from_parts(30, 2))
        );
        assert_eq!(
            index.get(5).unwrap(),
            Some(RowAddress::new_from_parts(20, 2))
        );
        assert_eq!(
            index.get(6).unwrap(),
            Some(RowAddress::new_from_parts(10, 2))
        );
        assert_eq!(
            index.get(7).unwrap(),
            Some(RowAddress::new_from_parts(30, 0))
        );
        assert_eq!(
            index.get(8).unwrap(),
            Some(RowAddress::new_from_parts(20, 0))
        );
        assert_eq!(
            index.get(9).unwrap(),
            Some(RowAddress::new_from_parts(10, 0))
        );

        // Check that non-existent row ids return None
        assert_eq!(index.get(0).unwrap(), None);
        assert_eq!(index.get(10).unwrap(), None);
    }

    #[test]
    fn test_new_index_partial_overlap() {
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 0,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::RangeWithHoles {
                    range: 0..100,
                    holes: vec![50].into(),
                }])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 1,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(50..51)])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        // Check various queries.
        assert_eq!(
            index.get(0).unwrap(),
            Some(RowAddress::new_from_parts(0, 0))
        );
        assert_eq!(
            index.get(49).unwrap(),
            Some(RowAddress::new_from_parts(0, 49))
        );
        assert_eq!(
            index.get(50).unwrap(),
            Some(RowAddress::new_from_parts(1, 0))
        );
        assert_eq!(
            index.get(51).unwrap(),
            Some(RowAddress::new_from_parts(0, 50))
        );
        assert_eq!(
            index.get(99).unwrap(),
            Some(RowAddress::new_from_parts(0, 98))
        );
    }

    #[test]
    fn test_overlapping_chunks_sparse_with_deletions() {
        // Interleaved (overlapping) id ranges plus a deletion that leaves a hole,
        // so the union doesn't tile the span. Every live id must still resolve.
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 10,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::SortedArray(
                    vec![1, 3, 5, 7, 9].into(),
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 20,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::SortedArray(
                    vec![0, 2, 4, 6, 8].into(),
                )])),
                // Delete offset 2 (id 4) -> a hole in the span.
                deletion_vector: Arc::new(DeletionVector::from_iter(vec![2])),
            },
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        assert_eq!(
            index.get(0).unwrap(),
            Some(RowAddress::new_from_parts(20, 0))
        );
        assert_eq!(
            index.get(1).unwrap(),
            Some(RowAddress::new_from_parts(10, 0))
        );
        assert_eq!(
            index.get(2).unwrap(),
            Some(RowAddress::new_from_parts(20, 1))
        );
        assert_eq!(
            index.get(3).unwrap(),
            Some(RowAddress::new_from_parts(10, 1))
        );
        assert_eq!(index.get(4).unwrap(), None);
        // Surviving ids keep their original offsets (the hole is not compacted).
        assert_eq!(
            index.get(6).unwrap(),
            Some(RowAddress::new_from_parts(20, 3))
        );
        assert_eq!(
            index.get(8).unwrap(),
            Some(RowAddress::new_from_parts(20, 4))
        );
        assert_eq!(
            index.get(9).unwrap(),
            Some(RowAddress::new_from_parts(10, 4))
        );
    }

    #[test]
    fn test_index_with_deletion_vector() {
        let deletion_vector = DeletionVector::from_iter(vec![2, 3]);

        let fragment_indices = vec![FragmentRowIdIndex {
            fragment_id: 10,
            row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(0..6)])),
            deletion_vector: Arc::new(deletion_vector),
        }];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        assert_eq!(
            index.get(0).unwrap(),
            Some(RowAddress::new_from_parts(10, 0))
        );
        assert_eq!(
            index.get(1).unwrap(),
            Some(RowAddress::new_from_parts(10, 1))
        );
        assert_eq!(
            index.get(4).unwrap(),
            Some(RowAddress::new_from_parts(10, 4))
        );
        assert_eq!(
            index.get(5).unwrap(),
            Some(RowAddress::new_from_parts(10, 5))
        );

        assert_eq!(index.get(2).unwrap(), None);
        assert_eq!(index.get(3).unwrap(), None);
    }

    #[test]
    fn test_empty_fragment_sequences() {
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 10,
                row_id_sequence: Arc::new(RowIdSequence(vec![])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 20,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(5..8)])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        assert_eq!(
            index.get(5).unwrap(),
            Some(RowAddress::new_from_parts(20, 0))
        );
        assert_eq!(
            index.get(7).unwrap(),
            Some(RowAddress::new_from_parts(20, 2))
        );
        assert_eq!(index.get(4).unwrap(), None);
    }

    #[test]
    fn test_completely_empty_index() {
        let fragment_indices = vec![];
        let index = RowIdIndex::new(&fragment_indices).unwrap();

        assert_eq!(index.get(0).unwrap(), None);
        assert_eq!(index.get(100).unwrap(), None);
    }

    #[test]
    fn test_non_overlapping_ranges() {
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 10,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(0..5)])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 20,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(5..10)])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 30,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(10..15)])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        assert_eq!(
            index.get(0).unwrap(),
            Some(RowAddress::new_from_parts(10, 0))
        );
        assert_eq!(
            index.get(4).unwrap(),
            Some(RowAddress::new_from_parts(10, 4))
        );
        assert_eq!(
            index.get(5).unwrap(),
            Some(RowAddress::new_from_parts(20, 0))
        );
        assert_eq!(
            index.get(9).unwrap(),
            Some(RowAddress::new_from_parts(20, 4))
        );
        assert_eq!(
            index.get(10).unwrap(),
            Some(RowAddress::new_from_parts(30, 0))
        );
        assert_eq!(
            index.get(14).unwrap(),
            Some(RowAddress::new_from_parts(30, 4))
        );
    }

    fn arbitrary_row_ids(
        num_fragments_range: std::ops::Range<usize>,
        frag_size_range: std::ops::Range<usize>,
    ) -> impl Strategy<Value = Vec<(u32, Arc<RowIdSequence>)>> {
        let fragment_sizes = proptest::collection::vec(frag_size_range, num_fragments_range);
        fragment_sizes.prop_flat_map(|fragment_sizes| {
            let num_rows = fragment_sizes.iter().sum::<usize>() as u64;
            let row_ids = 0..num_rows;
            let row_ids = row_ids.collect::<Vec<_>>();
            let row_ids_shuffled = proptest::strategy::Just(row_ids).prop_shuffle();
            row_ids_shuffled.prop_map(move |row_ids| {
                let mut sequences = Vec::with_capacity(fragment_sizes.len());
                let mut i = 0;
                for size in &fragment_sizes {
                    let end = i + size;
                    let sequence =
                        RowIdSequence(vec![U64Segment::from_slice(row_ids[i..end].into())]);
                    sequences.push((i as u32, Arc::new(sequence)));
                    i = end;
                }
                sequences
            })
        })
    }

    fn arbitrary_row_ids_with_deletions(
        num_fragments_range: std::ops::Range<usize>,
        frag_size_range: std::ops::Range<usize>,
    ) -> impl Strategy<Value = Vec<(u32, Arc<RowIdSequence>, Arc<DeletionVector>)>> {
        arbitrary_row_ids(num_fragments_range, frag_size_range)
            .prop_flat_map(|row_ids| {
                let num_rows = row_ids
                    .iter()
                    .map(|(_, sequence)| sequence.len() as usize)
                    .sum::<usize>();
                (
                    Just(row_ids),
                    proptest::collection::vec(any::<bool>(), num_rows),
                )
            })
            .prop_map(|(row_ids, deleted_rows)| {
                let mut deleted_rows = deleted_rows.into_iter();
                row_ids
                    .into_iter()
                    .map(|(fragment_id, sequence)| {
                        let mut deletion_bitmap = roaring::RoaringBitmap::new();
                        for offset in 0..sequence.len() as u32 {
                            if deleted_rows.next().unwrap() {
                                deletion_bitmap.insert(offset);
                            }
                        }
                        (
                            fragment_id,
                            sequence,
                            Arc::new(DeletionVector::Bitmap(deletion_bitmap)),
                        )
                    })
                    .collect()
            })
    }

    #[test]
    fn test_large_range_segments_no_deletions() {
        // Simulates a real-world scenario: many fragments with large Range segments
        // and no deletions. Before optimization, this would iterate over all rows
        // (O(total_rows)). After optimization, it's O(num_fragments).
        let rows_per_fragment = 250_000u64;
        let num_fragments = 100u32;
        let mut offset = 0u64;

        let fragment_indices: Vec<FragmentRowIdIndex> = (0..num_fragments)
            .map(|frag_id| {
                let start = offset;
                offset += rows_per_fragment;
                FragmentRowIdIndex {
                    fragment_id: frag_id,
                    row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(
                        start..start + rows_per_fragment,
                    )])),
                    deletion_vector: Arc::new(DeletionVector::default()),
                }
            })
            .collect();

        let start = std::time::Instant::now();
        let index = RowIdIndex::new(&fragment_indices).unwrap();
        let elapsed = start.elapsed();

        // Verify correctness at boundaries
        assert_eq!(
            index.get(0).unwrap(),
            Some(RowAddress::new_from_parts(0, 0))
        );
        assert_eq!(
            index.get(rows_per_fragment - 1).unwrap(),
            Some(RowAddress::new_from_parts(0, rows_per_fragment as u32 - 1))
        );
        assert_eq!(
            index.get(rows_per_fragment).unwrap(),
            Some(RowAddress::new_from_parts(1, 0))
        );
        let last_row = num_fragments as u64 * rows_per_fragment - 1;
        assert_eq!(
            index.get(last_row).unwrap(),
            Some(RowAddress::new_from_parts(
                num_fragments - 1,
                rows_per_fragment as u32 - 1
            ))
        );
        assert_eq!(index.get(last_row + 1).unwrap(), None);

        // With the optimization, building an index for 25M rows across 100 fragments
        // should complete in well under 1 second (typically < 1ms).
        assert!(
            elapsed.as_secs() < 1,
            "Index build took {:?} for {} fragments x {} rows = {} total rows. \
             This suggests the O(rows) -> O(fragments) optimization is not working.",
            elapsed,
            num_fragments,
            rows_per_fragment,
            num_fragments as u64 * rows_per_fragment,
        );
    }

    #[test]
    fn test_large_range_segments_with_deletions() {
        let rows_per_fragment = 1_000u64;
        let num_fragments = 10u32;
        let mut offset = 0u64;

        let fragment_indices: Vec<FragmentRowIdIndex> = (0..num_fragments)
            .map(|frag_id| {
                let start = offset;
                offset += rows_per_fragment;

                // Delete every 3rd row (offsets 0, 3, 6, ...) within each fragment.
                let mut deleted = roaring::RoaringBitmap::new();
                for i in (0..rows_per_fragment as u32).step_by(3) {
                    deleted.insert(i);
                }

                FragmentRowIdIndex {
                    fragment_id: frag_id,
                    row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(
                        start..start + rows_per_fragment,
                    )])),
                    deletion_vector: Arc::new(DeletionVector::Bitmap(deleted)),
                }
            })
            .collect();

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        // Deleted rows (offset 0, 3, 6, ...) should not be found.
        // Row ID 0 has offset 0 in fragment 0 -> deleted.
        assert_eq!(index.get(0).unwrap(), None);
        // Row ID 3 has offset 3 in fragment 0 -> deleted.
        assert_eq!(index.get(3).unwrap(), None);

        // Non-deleted rows should resolve correctly.
        // Row ID 1 has offset 1 in fragment 0 -> address (frag=0, row=1).
        assert_eq!(
            index.get(1).unwrap(),
            Some(RowAddress::new_from_parts(0, 1))
        );
        // Row ID 2 has offset 2 in fragment 0 -> address (frag=0, row=2).
        assert_eq!(
            index.get(2).unwrap(),
            Some(RowAddress::new_from_parts(0, 2))
        );
        // Row ID 4 has offset 4 in fragment 0 -> address (frag=0, row=4).
        assert_eq!(
            index.get(4).unwrap(),
            Some(RowAddress::new_from_parts(0, 4))
        );

        // Check second fragment: row IDs start at 1000.
        // Row ID 1000 has offset 0 in fragment 1 -> deleted.
        assert_eq!(index.get(rows_per_fragment).unwrap(), None);
        // Row ID 1001 has offset 1 in fragment 1 -> address (frag=1, row=1).
        assert_eq!(
            index.get(rows_per_fragment + 1).unwrap(),
            Some(RowAddress::new_from_parts(1, 1))
        );

        // Last fragment, last non-deleted row.
        // Row ID 9999 has offset 999 in fragment 9 -> 999 % 3 == 0 -> deleted.
        let last_row = num_fragments as u64 * rows_per_fragment - 1;
        assert_eq!(index.get(last_row).unwrap(), None);
        // Row ID 9998 has offset 998 -> 998 % 3 == 2 -> not deleted.
        assert_eq!(
            index.get(last_row - 1).unwrap(),
            Some(RowAddress::new_from_parts(num_fragments - 1, 998))
        );

        // Out of range.
        assert_eq!(index.get(last_row + 1).unwrap(), None);
    }

    proptest::proptest! {
        #[test]
        fn test_new_index_robustness(
            row_ids in arbitrary_row_ids_with_deletions(0..5, 0..32)
        ) {
            let fragment_indices: Vec<FragmentRowIdIndex> = row_ids
                .iter()
                .map(|(frag_id, sequence, deletion_vector)| FragmentRowIdIndex {
                    fragment_id: *frag_id,
                    row_id_sequence: sequence.clone(),
                    deletion_vector: deletion_vector.clone(),
                })
                .collect();

            let merged = RowIdIndex::new(&fragment_indices).unwrap();
            let probing = RowIdIndex::probing(&fragment_indices).unwrap();
            for index in [&merged, &probing] {
                for (frag_id, sequence, deletion_vector) in row_ids.iter() {
                    for (local_offset, row_id) in sequence.iter().enumerate() {
                        let expected = if deletion_vector.contains(local_offset as u32) {
                            None
                        } else {
                            Some(RowAddress::new_from_parts(*frag_id, local_offset as u32))
                        };
                        prop_assert_eq!(
                            index.get(row_id).unwrap(),
                            expected,
                            "Row id {} in sequence {:?} not found in index {:?}",
                            row_id,
                            sequence,
                            index
                        );
                    }
                }
            }
        }

        #[test]
        fn test_new_index_moved_row_id(
            row_id in any::<u64>(),
            source_fragment in 0u32..1024,
            fragment_delta in 1u32..1024,
        ) {
            let target_fragment = source_fragment + fragment_delta;
            let fragment_indices = [
                FragmentRowIdIndex {
                    fragment_id: source_fragment,
                    row_id_sequence: Arc::new(RowIdSequence::from(&[row_id][..])),
                    deletion_vector: Arc::new(DeletionVector::Bitmap(
                        roaring::RoaringBitmap::from_iter([0]),
                    )),
                },
                FragmentRowIdIndex {
                    fragment_id: target_fragment,
                    row_id_sequence: Arc::new(RowIdSequence::from(&[row_id][..])),
                    deletion_vector: Arc::new(DeletionVector::default()),
                },
            ];

            let index = RowIdIndex::new(&fragment_indices).unwrap();
            prop_assert_eq!(
                index.get(row_id).unwrap(),
                Some(RowAddress::new_from_parts(target_fragment, 0))
            );
        }

        #[test]
        fn test_new_index_rejects_duplicate_live_row_id(
            row_id in any::<u64>(),
            first_fragment in 0u32..1024,
            fragment_delta in 1u32..1024,
        ) {
            let second_fragment = first_fragment + fragment_delta;
            let fragment_indices = [
                FragmentRowIdIndex {
                    fragment_id: first_fragment,
                    row_id_sequence: Arc::new(RowIdSequence::from(&[row_id][..])),
                    deletion_vector: Arc::new(DeletionVector::default()),
                },
                FragmentRowIdIndex {
                    fragment_id: second_fragment,
                    row_id_sequence: Arc::new(RowIdSequence::from(&[row_id][..])),
                    deletion_vector: Arc::new(DeletionVector::default()),
                },
            ];

            let error = RowIdIndex::new(&fragment_indices).unwrap_err();
            let is_internal = matches!(&error, Error::Internal { .. });
            let expected_message =
                format!("stable row id {row_id} is live in multiple fragments");
            let error_message = error.to_string();
            prop_assert!(is_internal);
            prop_assert!(error_message.contains(&expected_message));
        }
    }
}
