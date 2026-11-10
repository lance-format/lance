// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::ops::RangeInclusive;
use std::sync::{Arc, Weak};

use super::bitmap::{Bitmap, count_ones};
use super::encoded_array::EncodedU64Array;
use super::{RowIdSequence, U64Segment};
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result};
use rangemap::RangeInclusiveMap;
use rayon::prelude::*;
use roaring::RoaringBitmap;

mod batch;
mod combine;
mod positions;
mod sparse;

use positions::Positions;

/// Fragments one lookup may have to probe before the merged map is worth its
/// build, whatever that build costs. A compacted table interleaves its
/// fragments, and one measured at 46.
const MAX_PROBE_DEPTH: u64 = 64;

/// Row ids the merged build may read before probing is worth its per-lookup
/// cost instead.
const MERGE_ROWS_BUDGET: u64 = 1 << 20;
/// Consecutive ids of one segment at least this far apart get separate probe
/// entries, so a segment does not cover the gap between the pieces it holds.
const PROBE_GAP: u64 = 1 << 16;

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
// `new` takes that when few segments cover one id and the merged build would
// read a lot of them.
#[derive(Debug)]
pub struct RowIdIndex {
    /// Fragments that hold at least one row id.
    fragments: Vec<Arc<FragmentEntry>>,
    /// Every piece of every segment of `fragments`, sorted by its lowest row
    /// id (see [`probe_spans`]).
    probes: Vec<ProbeEntry>,
    /// Max-`end` heap over `probes`: `end_tree[1]` is the root and leaf `i`
    /// sits at `end_tree[len() / 2 + i]`.
    end_tree: Vec<u64>,
    merged: Option<MergedIndex>,
    sources: Vec<FragmentIdentity>,
}

/// One piece of one segment of one fragment, as the probe tree sees it. The
/// tree is over pieces rather than fragments: a compacted fragment's rows come
/// from far-apart places, so its span covers most of the table while each
/// piece covers only the ids it holds. A segment that holds several pieces
/// has one entry per piece, split at gaps of [`PROBE_GAP`].
#[derive(Debug, Clone, Copy)]
struct ProbeEntry {
    /// Index into `RowIdIndex::fragments`.
    fragment: u32,
    /// Index into that fragment's `segments`.
    segment: u32,
    /// Lowest and highest id of the piece, inclusive.
    start: u64,
    end: u64,
}

type MergedIndex = RangeInclusiveMap<u64, Arc<MergedChunk>>;

#[derive(Debug, DeepSizeOf)]
struct MergedChunk {
    segments: (U64Segment, U64Segment),
    /// Chunks built together share one source group. An entire disjoint append
    /// can be removed without rebuilding older groups or keeping old indices.
    sources: Arc<Vec<FragmentIdentity>>,
}

impl PartialEq for MergedChunk {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.sources, &other.sources) && self.segments == other.segments
    }
}

impl Eq for MergedChunk {}

pub struct FragmentRowIdIndex {
    pub fragment_id: u32,
    pub row_id_sequence: Arc<RowIdSequence>,
    pub deletion_vector: Arc<DeletionVector>,
}

#[derive(Debug, Clone)]
struct FragmentIdentity {
    fragment_id: u32,
    sequence: Weak<RowIdSequence>,
    deletion_vector: Weak<DeletionVector>,
}

type FragmentKey = (u32, *const RowIdSequence, *const DeletionVector);

impl FragmentIdentity {
    fn new(source: &FragmentRowIdIndex) -> Self {
        Self {
            fragment_id: source.fragment_id,
            sequence: Arc::downgrade(&source.row_id_sequence),
            deletion_vector: Arc::downgrade(&source.deletion_vector),
        }
    }

    fn key(&self) -> FragmentKey {
        // Weak references keep allocation identities from being recycled while
        // this index can still use them to recognize a source.
        (
            self.fragment_id,
            self.sequence.as_ptr(),
            self.deletion_vector.as_ptr(),
        )
    }
}

impl DeepSizeOf for FragmentIdentity {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        // A dead Weak still retains space for T, but none of T's former heap
        // children. Live allocations belong to their strong owners. As with
        // DeepSizeOf for Arc, allocator/control-block overhead is excluded.
        fn expired_size<T>(weak: &Weak<T>, context: &mut lance_core::deepsize::Context) -> usize {
            if weak.strong_count() == 0 && context.mark_seen(weak.as_ptr() as usize) {
                std::mem::size_of::<T>()
            } else {
                0
            }
        }
        expired_size(&self.sequence, context) + expired_size(&self.deletion_vector, context)
    }
}

impl RowIdIndex {
    /// Create a new index from fragment ids and their row id sequences.
    pub fn new(fragment_indices: &[FragmentRowIdIndex]) -> Result<Self> {
        Self::new_reusing(fragment_indices, None)
    }

    /// Build a snapshot's index, reusing immutable lookup data from `previous`.
    ///
    /// Reuse requires the same fragment id and the same sequence and deletion
    /// vector allocations. The caller must establish content identity before
    /// sharing those allocations. The previous index itself is never retained.
    /// Unchanged merged maps also share their chunks across a disjoint append
    /// of undeleted range segments, including when revisiting the old snapshot.
    /// Partially changed source groups rebuild through the normal cost gate.
    ///
    /// ```
    /// # use lance_table::rowids::{RowIdIndex, FragmentRowIdIndex};
    /// # fn example(fragments: &[FragmentRowIdIndex]) -> lance_core::Result<()> {
    /// let first = RowIdIndex::new(fragments)?;
    /// let next = RowIdIndex::new_reusing(fragments, Some(&first))?;
    /// assert_eq!(first.get(0)?, next.get(0)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_reusing(
        fragment_indices: &[FragmentRowIdIndex],
        previous: Option<&Self>,
    ) -> Result<Self> {
        let sources: Vec<_> = fragment_indices.iter().map(FragmentIdentity::new).collect();
        if let Some(previous) = previous
            && let Some(merged) = previous.reuse_merged(fragment_indices, &sources)?
        {
            return Ok(Self {
                fragments: Vec::new(),
                probes: Vec::new(),
                end_tree: Vec::new(),
                merged: Some(merged),
                sources,
            });
        }
        let prepared: HashMap<_, _> = previous
            .into_iter()
            .flat_map(|previous| &previous.fragments)
            .map(|entry| (entry.identity.key(), entry))
            .collect();
        let fragments: Vec<Arc<FragmentEntry>> = fragment_indices
            .iter()
            .zip(&sources)
            .filter_map(|(source, identity)| {
                if let Some(reused) = prepared.get(&identity.key()) {
                    return Some(Arc::clone(reused));
                }
                FragmentEntry::new(source).map(Arc::new)
            })
            .collect();
        let probes = probe_entries(&fragments);
        if !probing_beats_merging(&fragments, &probes) {
            return Ok(Self {
                fragments: Vec::new(),
                probes: Vec::new(),
                end_tree: Vec::new(),
                merged: Some(Self::build_merged(fragment_indices)?),
                sources,
            });
        }
        Ok(Self::build_probe(fragments, probes, sources))
    }

    /// Reuse prepared fragments after the caller has verified unchanged content.
    ///
    /// Each retained id refers to a fragment in this index. The caller must
    /// verify that its row IDs, deletions and physical addresses are unchanged;
    /// matching a fragment id alone is insufficient. Original row-ID sequences
    /// need not remain alive. Only `new_fragments` require preparation.
    ///
    /// Returns `None` when a probe-to-merged transition or a partially changed
    /// merged source group requires original sequences. The caller can then
    /// load those sequences and use [`Self::new`]. No previous index is retained.
    /// Duplicate or missing retained ids, and duplicate ids across retained and
    /// new fragments, return an invalid-input error.
    ///
    /// ```
    /// # use lance_table::rowids::{RowIdIndex, FragmentRowIdIndex};
    /// # fn example(first: &RowIdIndex, appended: &[FragmentRowIdIndex])
    /// # -> lance_core::Result<()> {
    /// // The caller verified that fragment 7 has unchanged content.
    /// if let Some(next) = first.try_reuse_fragments(&[7], appended)? {
    ///     let addresses = next.get_many(&[10, 20])?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_reuse_fragments(
        &self,
        retained_fragment_ids: &[u32],
        new_fragments: &[FragmentRowIdIndex],
    ) -> Result<Option<Self>> {
        let previous: HashMap<_, _> = self
            .sources
            .iter()
            .map(|source| (source.fragment_id, source))
            .collect();
        // An id cannot select one of several sources with the same address base.
        if previous.len() != self.sources.len() {
            return Ok(None);
        }
        let mut seen = RoaringBitmap::new();
        let mut sources = Vec::with_capacity(retained_fragment_ids.len() + new_fragments.len());
        for &id in retained_fragment_ids {
            if !seen.insert(id) {
                return Err(Error::invalid_input(format!(
                    "retained_fragment_ids contains duplicate fragment_id {id}"
                )));
            }
            let source = previous.get(&id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "retained fragment_id {id} is missing from the previous row id index"
                ))
            })?;
            sources.push((*source).clone());
        }
        for source in new_fragments {
            if !seen.insert(source.fragment_id) {
                return Err(Error::invalid_input(format!(
                    "new_fragments contains duplicate fragment_id {}",
                    source.fragment_id
                )));
            }
            sources.push(FragmentIdentity::new(source));
        }
        if retained_fragment_ids.is_empty() {
            return Self::new(new_fragments).map(Some);
        }
        if self.merged.is_some() {
            return Ok(self
                .reuse_merged(new_fragments, &sources)?
                .map(|merged| Self {
                    fragments: Vec::new(),
                    probes: Vec::new(),
                    end_tree: Vec::new(),
                    merged: Some(merged),
                    sources,
                }));
        }
        let prepared: HashMap<_, _> = self
            .fragments
            .iter()
            .map(|entry| (entry.fragment_id, entry))
            .collect();
        let fragments: Vec<_> = retained_fragment_ids
            .iter()
            // Empty source fragments have no prepared entry.
            .filter_map(|id| prepared.get(id).map(|entry| Arc::clone(entry)))
            .chain(
                new_fragments
                    .iter()
                    .filter_map(FragmentEntry::new)
                    .map(Arc::new),
            )
            .collect();
        let probes = probe_entries(&fragments);
        if !probing_beats_merging(&fragments, &probes) {
            return Ok(None);
        }
        Ok(Some(Self::build_probe(fragments, probes, sources)))
    }

    /// Combine independently prepared indices without rebuilding their lookups.
    ///
    /// The caller must verify that each input's row IDs, deletions and physical
    /// addresses belong to the desired snapshot. Fragment IDs must be disjoint
    /// across all input sources, including empty fragments; duplicates return an
    /// invalid-input error. Original source sequences need not remain alive.
    ///
    /// Shares prepared probe allocations directly and rebuilds only routing
    /// metadata. All-merged inputs share disjoint chunks and merge overlapping
    /// sorted chunks with bounded scratch space, without original sequences or
    /// input-index parents. Returns `None` for mixed representations or when
    /// combined probes require a merged representation; the caller can load
    /// original sequences and use [`Self::new`]. Empty input produces an empty
    /// index. Duplicate live IDs error during merged composition or probe lookup.
    ///
    /// ```
    /// # use lance_table::rowids::RowIdIndex;
    /// # fn example(prepared: &[&RowIdIndex]) -> lance_core::Result<()> {
    /// if let Some(combined) = RowIdIndex::try_combine(prepared)? {
    ///     let addresses = combined.get_many(&[10, 20, 10])?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_combine(indices: &[&Self]) -> Result<Option<Self>> {
        if indices.is_empty() {
            return Self::new(&[]).map(Some);
        }
        let mut seen = RoaringBitmap::new();
        let mut sources = Vec::with_capacity(indices.iter().map(|index| index.sources.len()).sum());
        for index in indices {
            for source in &index.sources {
                if !seen.insert(source.fragment_id) {
                    return Err(Error::invalid_input(format!(
                        "indices contains duplicate fragment_id {}",
                        source.fragment_id
                    )));
                }
                sources.push(source.clone());
            }
        }
        if indices.iter().any(|index| index.merged.is_some()) {
            if indices.iter().all(|index| index.merged.is_some()) {
                return combine::merged(indices, sources).map(Some);
            }
            return Ok(None);
        }
        let fragments: Vec<_> = indices
            .iter()
            .flat_map(|index| index.fragments.iter().cloned())
            .collect();
        let probes = probe_entries(&fragments);
        if !probing_beats_merging(&fragments, &probes) {
            return Ok(None);
        }
        Ok(Some(Self::build_probe(fragments, probes, sources)))
    }

    fn build_probe(
        mut fragments: Vec<Arc<FragmentEntry>>,
        probes: Vec<ProbeEntry>,
        sources: Vec<FragmentIdentity>,
    ) -> Self {
        // Shared entries are already prepared. Native encodings need no work;
        // avoid even starting the CPU pool for a directory-only version change.
        let lookups: Vec<_> = fragments
            .iter_mut()
            .filter_map(Arc::get_mut)
            .filter(|entry| {
                entry.segments.iter().any(|segment| {
                    matches!(
                        entry.sequence.0[segment.seq_idx],
                        U64Segment::Array(_) | U64Segment::RangeWithBitmap { .. }
                    )
                })
            })
            .collect();
        if lookups.len() > 1 {
            lookups
                .into_par_iter()
                .for_each(FragmentEntry::build_lookups);
        } else {
            for entry in lookups {
                entry.build_lookups();
            }
        }
        Self {
            end_tree: build_end_tree(&probes),
            probes,
            fragments,
            merged: None,
            sources,
        }
    }

    /// Keep complete unchanged source groups and drop groups absent in this
    /// snapshot. Partial changes and intersecting additions need normal validation.
    fn reuse_merged(
        &self,
        sources: &[FragmentRowIdIndex],
        identities: &[FragmentIdentity],
    ) -> Result<Option<MergedIndex>> {
        let Some(previous) = &self.merged else {
            return Ok(None);
        };
        let old_keys: HashSet<_> = self.sources.iter().map(FragmentIdentity::key).collect();
        let new_keys: HashSet<_> = identities.iter().map(FragmentIdentity::key).collect();
        // Repeating a source must still go through normal duplicate validation.
        if old_keys.len() != self.sources.len() || new_keys.len() != identities.len() {
            return Ok(None);
        }
        let mut groups = HashMap::new();
        let mut retained_keys = HashSet::new();
        for (_, chunk) in previous.iter() {
            let group = Arc::as_ptr(&chunk.sources);
            if groups.contains_key(&group) {
                continue;
            }
            let present = chunk
                .sources
                .iter()
                .filter(|source| new_keys.contains(&source.key()))
                .count();
            if present != 0 && present != chunk.sources.len() {
                return Ok(None);
            }
            let keep = present == chunk.sources.len();
            if keep {
                retained_keys.extend(chunk.sources.iter().map(FragmentIdentity::key));
            }
            groups.insert(group, keep);
        }
        let mut merged: MergedIndex = previous
            .iter()
            .filter(|(_, chunk)| groups[&Arc::as_ptr(&chunk.sources)])
            .map(|(range, chunk)| (range.clone(), chunk.clone()))
            .collect();
        let mut additions = Vec::new();
        for source in sources {
            if retained_keys.contains(&FragmentIdentity::new(source).key()) {
                continue;
            }
            // Preserve the normal cost gate for additions that enumerate IDs:
            // a previously cheap merged map can become cheaper to probe.
            if !source.deletion_vector.is_empty()
                || source
                    .row_id_sequence
                    .0
                    .iter()
                    .any(|segment| !matches!(segment, U64Segment::Range(_)))
            {
                return Ok(None);
            }
            if source
                .row_id_sequence
                .row_id_range()
                .is_some_and(|range| merged.overlapping(range).next().is_some())
            {
                return Ok(None);
            }
            additions.push(FragmentRowIdIndex {
                fragment_id: source.fragment_id,
                row_id_sequence: source.row_id_sequence.clone(),
                deletion_vector: source.deletion_vector.clone(),
            });
        }
        for (range, chunk) in Self::build_merged(&additions)? {
            merged.insert(range, chunk);
        }
        Ok(Some(merged))
    }

    fn build_merged(sources: &[FragmentRowIdIndex]) -> Result<MergedIndex> {
        let identities = Arc::new(
            sources
                .iter()
                .map(FragmentIdentity::new)
                .collect::<Vec<_>>(),
        );
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

        Ok(RangeInclusiveMap::from_iter(final_chunks.into_iter().map(
            |(range, segments)| {
                (
                    range,
                    Arc::new(MergedChunk {
                        segments,
                        sources: identities.clone(),
                    }),
                )
            },
        )))
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

    /// Get addresses for many row ids, preserving input order and duplicates.
    ///
    /// Returns one entry per input id, in input order (`None` for missing).
    /// Sorts a working copy of the input internally. The merged representation
    /// advances its chunk iterator at most once per chunk. The probe representation
    /// prunes the interval tree for the batch and interleaves independent searches
    /// within unsorted segments; small batches retain individual tree lookups.
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
            if n < batch::LANES {
                for (id, orig_idx) in sorted {
                    out[orig_idx] = self.probe(id)?;
                }
            } else {
                self.probe_many(&sorted, &mut out)?;
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
            let Some((range, segments)) = chunks.peek() else {
                break;
            };
            let (row_id_seg, addr_seg) = &segments.segments;
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

    /// Address of `row_id`, from the segment that holds it live. Descends the
    /// max-`end` tree, so a segment out of reach of the id costs nothing.
    ///
    /// Visits every candidate rather than stopping at the first hit, and
    /// errors when a second fragment holds the id live.
    fn probe(&self, row_id: u64) -> Result<Option<RowAddress>> {
        let probes = self.probes.len();
        if probes == 0 {
            return Ok(None);
        }
        // Only a segment that starts at or below the id can hold it.
        let upper = self.probes.partition_point(|probe| probe.start <= row_id);
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
                if let Some(probe) = self.probes.get(lo)
                    && let Some(candidate) = self.fragments[probe.fragment as usize]
                        .resolve_segment(probe.segment as usize, row_id)
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
    let (row_id_segment, address_segment) = &merged.get(&row_id)?.segments;
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
    /// Cached once so sharing a sorted segment never rescans all its IDs.
    probe_spans: Vec<(u64, u64)>,
}

/// How a [`SegmentEntry`] finds the position of a row id.
#[derive(Debug)]
enum SegmentLookup {
    /// The encoding searches itself in logarithmic or constant time.
    Native,
    /// Row id to position for an unsorted [`U64Segment::Array`], whose own
    /// `position` scans.
    Positions(Positions),
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
            SegmentLookup::Positions(positions) => {
                positions.get(row_id).map(|offset| offset as usize)
            }
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
        U64Segment::Array(values) => (
            SegmentLookup::Positions(Positions::new(values)),
            values.len(),
        ),
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
    identity: FragmentIdentity,
    sequence: Arc<RowIdSequence>,
    deletion_vector: Arc<DeletionVector>,
    segments: Vec<SegmentEntry>,
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
            let len = segment.len();
            let lookup = SegmentLookup::Native;
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
                    probe_spans: probe_spans(segment, &range),
                    range,
                    start_offset,
                    lookup,
                });
            }
            start_offset += len as u32;
        }
        if segments.is_empty() {
            return None;
        }
        Some(Self {
            fragment_id: source.fragment_id,
            identity: FragmentIdentity::new(source),
            sequence: source.row_id_sequence.clone(),
            deletion_vector: source.deletion_vector.clone(),
            segments,
            merge_rows,
        })
    }

    /// Build only the selected probe representation. Array lookup owns the IDs
    /// it needs, so replace those source segments instead of retaining both copies.
    fn build_lookups(&mut self) {
        for entry in &mut self.segments {
            entry.lookup = build_lookup(&self.sequence.0[entry.seq_idx]).0;
        }
        if self
            .sequence
            .0
            .iter()
            .any(|segment| matches!(segment, U64Segment::Array(_)))
        {
            self.sequence = Arc::new(RowIdSequence(
                self.sequence
                    .0
                    .iter()
                    .map(|segment| match segment {
                        U64Segment::Array(_) => U64Segment::Range(0..0),
                        _ => segment.clone(),
                    })
                    .collect(),
            ));
        }
    }

    /// Address of `row_id` in segment `segment` of this fragment, or `None`
    /// when the segment lacks it or the fragment holds it deleted.
    fn resolve_segment(&self, segment: usize, row_id: u64) -> Option<RowAddress> {
        let entry = &self.segments[segment];
        if !entry.range.contains(&row_id) {
            return None;
        }
        let position = entry.position(&self.sequence, row_id)?;
        let row_offset = entry.start_offset + position as u32;
        if self.deletion_vector.contains(row_offset) {
            return None;
        }
        Some(RowAddress::new_from_parts(self.fragment_id, row_offset))
    }
}

/// Every piece of every segment of `fragments`, sorted by its lowest row id.
fn probe_entries(fragments: &[Arc<FragmentEntry>]) -> Vec<ProbeEntry> {
    let mut probes: Vec<ProbeEntry> = fragments
        .iter()
        .enumerate()
        .flat_map(|(fragment, entry)| {
            entry
                .segments
                .iter()
                .enumerate()
                .flat_map(move |(segment, seg)| {
                    seg.probe_spans.iter().map(move |&(start, end)| ProbeEntry {
                        fragment: fragment as u32,
                        segment: segment as u32,
                        start,
                        end,
                    })
                })
        })
        .collect();
    probes.sort_unstable_by_key(|probe| probe.start);
    probes
}

/// The inclusive spans of ids `segment` holds, split where consecutive ids are
/// more than [`PROBE_GAP`] apart.
///
/// Compaction leaves a sorted array holding the ids of far-apart source
/// fragments, and decoding folds consecutive `Range` segments into one
/// `Ranges` across any gap below 4 billion ids. One entry per such segment
/// would cover every piece that lives in its gaps and push the overlap depth
/// past [`MAX_PROBE_DEPTH`]; one entry per piece keeps the depth at what the
/// pieces themselves overlap. Costs the runs of a `Ranges` or the length of a
/// sorted array; the other kinds are one span by construction.
fn probe_spans(segment: &U64Segment, range: &RangeInclusive<u64>) -> Vec<(u64, u64)> {
    let mut spans: Vec<(u64, u64)> = Vec::new();
    let mut push = |start: u64, end: u64| match spans.last_mut() {
        Some((_, last_end)) if start - *last_end <= PROBE_GAP => *last_end = end,
        _ => spans.push((start, end)),
    };
    match segment {
        U64Segment::Ranges { range, runs } => {
            for run in runs.present_ranges() {
                push(
                    range.start + run.start as u64,
                    range.start + run.end as u64 - 1,
                );
            }
        }
        // Offsets fit u16, so no gap inside the array can reach PROBE_GAP.
        U64Segment::SortedArray(EncodedU64Array::U32 { base, offsets }) => {
            sorted_gap_spans(*base, offsets, &mut spans)
        }
        U64Segment::SortedArray(EncodedU64Array::U64(values)) => {
            sorted_gap_spans(0, values, &mut spans)
        }
        _ => return vec![(*range.start(), *range.end())],
    }
    spans
}

/// Spans of a sorted packed array split at gaps above [`PROBE_GAP`], read
/// straight from the slice. Gaps that wide are rare (a compacted fragment
/// holds a handful of pieces), so each block of values is first checked for
/// one with a branch-free reduction that vectorizes, and only a block that
/// has one is walked value by value.
fn sorted_gap_spans<T: Copy + Into<u64>>(base: u64, offsets: &[T], spans: &mut Vec<(u64, u64)>) {
    const BLOCK: usize = 256;
    let Some(&first) = offsets.first() else {
        return;
    };
    let mut start = base + first.into();
    let mut at = 0;
    while at + 1 < offsets.len() {
        let end = (at + BLOCK).min(offsets.len() - 1);
        let (block, next) = (&offsets[at..end], &offsets[at + 1..=end]);
        let has_gap = block.iter().zip(next).fold(false, |found, (&a, &b)| {
            found | (b.into() - a.into() > PROBE_GAP)
        });
        if has_gap {
            for (&a, &b) in block.iter().zip(next) {
                if b.into() - a.into() > PROBE_GAP {
                    spans.push((start, base + a.into()));
                    start = base + b.into();
                }
            }
        }
        at = end;
    }
    spans.push((start, base + offsets[offsets.len() - 1].into()));
}

/// Whether to answer lookups by probing the segments rather than by merging
/// every row id.
///
/// Probing costs the segments that cover one id, per lookup; merging costs the
/// row ids it reads, once. So probe only when both stay on the right side of
/// [`MAX_PROBE_DEPTH`] and [`MERGE_ROWS_BUDGET`].
fn probing_beats_merging(fragments: &[Arc<FragmentEntry>], probes: &[ProbeEntry]) -> bool {
    let merge_rows: u64 = fragments.iter().map(|entry| entry.merge_rows).sum();
    merge_rows > MERGE_ROWS_BUDGET && max_overlap_depth(probes) <= MAX_PROBE_DEPTH
}

/// Most segments that cover any one row id. `probes` is sorted by `start`.
fn max_overlap_depth(probes: &[ProbeEntry]) -> u64 {
    let mut ends: Vec<u64> = probes.iter().map(|probe| probe.end).collect();
    ends.sort_unstable();
    let mut closed = 0;
    let mut depth: u64 = 0;
    for (opened, probe) in probes.iter().enumerate() {
        while closed < ends.len() && ends[closed] < probe.start {
            closed += 1;
        }
        depth = depth.max((opened + 1 - closed) as u64);
    }
    depth
}

/// Implicit max-`end` heap over `probes`, padded to a power of two. Padding
/// leaves hold 0, which prunes for every id above 0 and is filtered by slot.
fn build_end_tree(probes: &[ProbeEntry]) -> Vec<u64> {
    if probes.is_empty() {
        return Vec::new();
    }
    let leaves = probes.len().next_power_of_two();
    let mut tree = vec![0_u64; 2 * leaves];
    for (slot, probe) in probes.iter().enumerate() {
        tree[leaves + slot] = probe.end;
    }
    for node in (1..leaves).rev() {
        tree[node] = tree[2 * node].max(tree[2 * node + 1]);
    }
    tree
}

impl DeepSizeOf for FragmentEntry {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.sequence.deep_size_of_children(context)
            + self.deletion_vector.deep_size_of_children(context)
            + self.identity.deep_size_of_children(context)
            + self.segments.capacity() * std::mem::size_of::<SegmentEntry>()
            + self
                .segments
                .iter()
                .map(|segment| {
                    segment.probe_spans.capacity() * std::mem::size_of::<(u64, u64)>()
                        + match &segment.lookup {
                            SegmentLookup::Native => 0,
                            SegmentLookup::Positions(positions) => {
                                positions.deep_size_of_children(context)
                            }
                            SegmentLookup::BitmapRank(rank) => {
                                rank.capacity() * std::mem::size_of::<u32>()
                            }
                        }
                })
                .sum::<usize>()
    }
}

impl DeepSizeOf for RowIdIndex {
    /// Count shared immutable allocations once within the caller's context.
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        let merged_bytes: usize = self
            .merged
            .as_ref()
            .map(|merged| {
                merged
                    .iter()
                    .map(|(_, chunk)| {
                        2 * std::mem::size_of::<u64>()
                            + std::mem::size_of::<Arc<MergedChunk>>()
                            + chunk.deep_size_of_children(context)
                    })
                    .sum()
            })
            .unwrap_or(0);
        self.fragments.deep_size_of_children(context)
            + merged_bytes
            + self.probes.capacity() * std::mem::size_of::<ProbeEntry>()
            + self.end_tree.capacity() * std::mem::size_of::<u64>()
            + self.sources.deep_size_of_children(context)
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
        U64Segment::Ranges { .. } if !segment.is_empty() => {
            // Without deletions, a live ID's position is also its physical
            // offset within this segment.
            let row_id_segment = segment.clone();
            let address_segment =
                U64Segment::Range(start_address..start_address + row_id_segment.len() as u64);
            let coverage = row_id_segment.range()?;
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
    if values.len() as u64 > MERGE_ROWS_BUDGET {
        values.par_sort_unstable_by_key(|(row_id, _)| *row_id);
    } else {
        values.sort_unstable_by_key(|(row_id, _)| *row_id);
    }
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
        let mut fragments: Vec<Arc<FragmentEntry>> = fragment_indices
            .iter()
            .filter_map(FragmentEntry::new)
            .map(Arc::new)
            .collect();
        let probes = probe_entries(&fragments);
        for fragment in &mut fragments {
            Arc::get_mut(fragment).unwrap().build_lookups();
        }
        Ok(Self {
            end_tree: build_end_tree(&probes),
            probes,
            fragments,
            merged: None,
            sources: fragment_indices.iter().map(FragmentIdentity::new).collect(),
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
    fn source_identity_counts_only_expired_allocation_storage() {
        let source = fragment(3, sparse_sequence(1024));
        let identity = FragmentIdentity::new(&source);
        let mut context = lance_core::deepsize::Context::default();
        assert_eq!(identity.deep_size_of_children(&mut context), 0);
        let source_size = source.row_id_sequence.deep_size_of_children(&mut context);
        assert!(source_size > 1024);
        assert_eq!(identity.deep_size_of_children(&mut context), 0);
        drop(source);

        let mut context = lance_core::deepsize::Context::default();
        assert_eq!(
            identity.deep_size_of_children(&mut context),
            std::mem::size_of::<RowIdSequence>() + std::mem::size_of::<DeletionVector>()
        );
        assert_eq!(identity.clone().deep_size_of_children(&mut context), 0);
    }

    #[test]
    fn reuse_merged_append_and_changed_deletions() {
        let mut sources = vec![
            fragment(
                3,
                RowIdSequence(vec![U64Segment::Array(vec![4, 0, 2].into())]),
            ),
            fragment(
                8,
                RowIdSequence(vec![U64Segment::Array(vec![5, 1, 3].into())]),
            ),
        ];
        let first = Arc::new(RowIdIndex::new(&sources).unwrap());
        let originals: Vec<_> = sources
            .iter()
            .map(|source| Arc::downgrade(&source.row_id_sequence))
            .collect();
        let old = Arc::downgrade(&first);
        sources.push(fragment(9, RowIdSequence(vec![U64Segment::Range(6..8)])));
        let appended = RowIdIndex::new_reusing(&sources, Some(&first)).unwrap();
        assert!(Arc::ptr_eq(
            first.merged.as_ref().unwrap().get(&0).unwrap(),
            appended.merged.as_ref().unwrap().get(&0).unwrap(),
        ));
        let queries = [5, 0, 6, 2, 7, 5, 8];
        let expected = [(8, 0), (3, 1), (9, 0), (3, 2), (9, 1), (8, 0)]
            .into_iter()
            .map(|(f, o)| Some(RowAddress::new_from_parts(f, o)))
            .chain([None])
            .collect::<Vec<_>>();
        assert_eq!(appended.get_many(&queries).unwrap(), expected);
        assert_eq!(first.get(6).unwrap(), None);
        let revisited = RowIdIndex::new_reusing(&sources[..2], Some(&appended)).unwrap();
        assert!(Arc::ptr_eq(
            first.merged.as_ref().unwrap().get(&0).unwrap(),
            revisited.merged.as_ref().unwrap().get(&0).unwrap(),
        ));
        assert_eq!(revisited.get(6).unwrap(), None);
        assert_eq!(revisited.get(5).unwrap(), expected[0]);
        drop(first);
        assert!(old.upgrade().is_none());
        assert_eq!(appended.get_many(&queries).unwrap(), expected);

        sources[0].deletion_vector = Arc::new(DeletionVector::Bitmap([1].into_iter().collect()));
        let deleted = RowIdIndex::new_reusing(&sources, Some(&appended)).unwrap();
        assert_eq!(deleted.get(0).unwrap(), None);
        assert_eq!(
            appended.get(0).unwrap(),
            Some(RowAddress::new_from_parts(3, 1))
        );
        sources.push(fragment(12, RowIdSequence(vec![U64Segment::Range(7..9)])));
        let error = RowIdIndex::new_reusing(&sources, Some(&deleted)).unwrap_err();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(error.to_string().contains("stable row id 7"));

        drop(sources);
        assert!(originals.iter().all(|source| source.upgrade().is_none()));
        let revisited = appended.try_reuse_fragments(&[8, 3], &[]).unwrap().unwrap();
        assert!(Arc::ptr_eq(
            appended.merged.as_ref().unwrap().get(&0).unwrap(),
            revisited.merged.as_ref().unwrap().get(&0).unwrap(),
        ));
        assert_eq!(revisited.get(6).unwrap(), None);
        assert_eq!(revisited.get(5).unwrap(), expected[0]);
        assert!(appended.try_reuse_fragments(&[3], &[]).unwrap().is_none());
        let added = fragment(10, RowIdSequence(vec![U64Segment::Range(10..12)]));
        let next = revisited
            .try_reuse_fragments(&[3, 8], &[added])
            .unwrap()
            .unwrap();
        assert_eq!(
            next.get(10).unwrap(),
            Some(RowAddress::new_from_parts(10, 0))
        );
        assert_eq!(next.get(5).unwrap(), expected[0]);
        assert_eq!(revisited.get(10).unwrap(), None);
        for ids in [&[3, 3][..], &[99][..]] {
            let error = revisited.try_reuse_fragments(ids, &[]).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("fragment_id"));
        }
        let duplicate = fragment(3, RowIdSequence(vec![U64Segment::Range(20..21)]));
        let error = revisited
            .try_reuse_fragments(&[3, 8], &[duplicate])
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("duplicate fragment_id 3"));
    }

    #[test]
    fn dense_probe_drops_source_and_keeps_physical_offsets() {
        let ids: Vec<_> = (0..1024).rev().map(|i| i * 10).collect();
        let source = fragment(
            7,
            RowIdSequence(vec![U64Segment::Array(ids.clone().into())]),
        );
        let sequence = Arc::downgrade(&source.row_id_sequence);
        let index = RowIdIndex::probing(&[source]).unwrap();
        assert!(sequence.upgrade().is_none());
        assert!(index.deep_size_of() < 8 * ids.len());
        let queries: Vec<_> = (0..10250).rev().collect();
        let expected: Vec<_> = queries
            .iter()
            .map(|id| {
                ids.iter()
                    .position(|candidate| candidate == id)
                    .map(|offset| RowAddress::new_from_parts(7, offset as u32))
            })
            .collect();
        assert_eq!(index.get_many(&queries).unwrap(), expected);
    }

    #[rstest::rstest]
    #[case::sorted(true)]
    #[case::unsorted(false)]
    fn reuse_probe_preserves_positions_and_snapshot_deletions(#[case] sorted: bool) {
        let len = MERGE_ROWS_BUDGET / 2 + 1;
        let ids = |parity| {
            let mut ids: Vec<_> = (0..len)
                .map(|offset| 2 * offset + parity + u64::from(offset >= len / 2) * PROBE_GAP * 2)
                .collect();
            if !sorted {
                ids.reverse();
            }
            ids
        };
        let rows = [ids(0), ids(1)];
        let mut sources: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(i, ids)| {
                let array = ids.clone().into();
                let segment = if sorted {
                    U64Segment::SortedArray(array)
                } else {
                    U64Segment::Array(array)
                };
                fragment(i as u32 + 7, RowIdSequence(vec![segment]))
            })
            .collect();
        let first = Arc::new(RowIdIndex::new(&sources).unwrap());
        assert!(first.merged.is_none());
        let old = Arc::downgrade(&first);
        sources.reverse();
        sources.push(fragment(
            9,
            RowIdSequence(vec![U64Segment::Range(9_000_000..9_000_002)]),
        ));
        let appended = RowIdIndex::new_reusing(&sources, Some(&first)).unwrap();
        assert!(Arc::ptr_eq(&first.fragments[0], &appended.fragments[1]));
        assert!(Arc::ptr_eq(&first.fragments[1], &appended.fragments[0]));
        let queries = [
            rows[0][0],
            rows[1][2],
            rows[0][len as usize - 1],
            9_000_001,
            u64::MAX,
            rows[0][0],
        ];
        let expected = vec![
            Some(RowAddress::new_from_parts(7, 0)),
            Some(RowAddress::new_from_parts(8, 2)),
            Some(RowAddress::new_from_parts(7, len as u32 - 1)),
            Some(RowAddress::new_from_parts(9, 1)),
            None,
            Some(RowAddress::new_from_parts(7, 0)),
        ];
        assert_eq!(appended.get_many(&queries).unwrap(), expected);
        drop(first);
        assert!(old.upgrade().is_none());

        sources[1].deletion_vector = Arc::new(DeletionVector::Bitmap([0].into_iter().collect()));
        let deleted = RowIdIndex::new_reusing(&sources, Some(&appended)).unwrap();
        assert!(Arc::ptr_eq(&appended.fragments[0], &deleted.fragments[0]));
        assert!(!Arc::ptr_eq(&appended.fragments[1], &deleted.fragments[1]));
        assert_eq!(deleted.get(queries[0]).unwrap(), None);
        assert_eq!(appended.get_many(&queries).unwrap(), expected);
        assert_eq!(deleted.get(queries[2]).unwrap(), expected[2]);

        let originals: Vec<_> = sources
            .iter()
            .map(|source| Arc::downgrade(&source.row_id_sequence))
            .collect();
        drop(sources);
        if !sorted {
            assert!(
                originals[..2]
                    .iter()
                    .all(|source| source.upgrade().is_none())
            );
        }
        let deleted = Arc::new(deleted);
        let previous = Arc::downgrade(&deleted);
        let added = fragment(
            10,
            RowIdSequence(vec![U64Segment::Range(9_100_000..9_100_002)]),
        );
        let next = deleted
            .try_reuse_fragments(&[9, 7, 8], &[added])
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&deleted.fragments[1], &next.fragments[1]));
        let mut context = lance_core::deepsize::Context::default();
        let old_charge = deleted.deep_size_of_children(&mut context);
        let new_charge = next.deep_size_of_children(&mut context);
        assert!(old_charge > 1_000_000);
        assert!(
            new_charge < 4096,
            "unchanged lookup allocations must be shared"
        );
        assert!(deleted.try_reuse_fragments(&[7], &[]).unwrap().is_none());
        drop(deleted);
        assert!(previous.upgrade().is_none());
        assert_eq!(next.get(queries[0]).unwrap(), None);
        assert_eq!(next.get(queries[2]).unwrap(), expected[2]);
        assert_eq!(
            next.get(9_100_001).unwrap(),
            Some(RowAddress::new_from_parts(10, 1))
        );
        assert_eq!(appended.get(9_100_001).unwrap(), None);
        let collision = fragment(
            11,
            RowIdSequence(vec![U64Segment::Range(queries[2]..queries[2] + 1)]),
        );
        let invalid = next
            .try_reuse_fragments(&[7, 8, 9, 10], &[collision])
            .unwrap()
            .unwrap();
        let error = invalid.get(queries[2]).unwrap_err();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(error.to_string().contains("live in multiple fragments"));
    }

    #[test]
    fn combine_empty_and_duplicate_fragment_sources() {
        let empty = RowIdIndex::try_combine(&[]).unwrap().unwrap();
        assert_eq!(empty.get(0).unwrap(), None);
        assert_eq!(empty.get_many(&[0, u64::MAX]).unwrap(), vec![None; 2]);

        // Empty fragments still own their physical address namespace. Validate
        // all sources even when their merged representation cannot be combined.
        let sources = [
            fragment(7, RowIdSequence::default()),
            fragment(7, RowIdSequence::default()),
        ];
        let first = RowIdIndex::new(&sources[..1]).unwrap();
        let second = RowIdIndex::new(&sources[1..]).unwrap();
        let repeated = RowIdIndex::new(&sources).unwrap();
        for indices in [vec![&first, &first], vec![&first, &second], vec![&repeated]] {
            let error = RowIdIndex::try_combine(&indices).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("duplicate fragment_id 7"));
        }
    }

    #[rstest::rstest]
    #[case::sorted(true)]
    #[case::unsorted(false)]
    fn combine_independent_probes_preserves_lookups_and_shares_memory(#[case] sorted: bool) {
        let len = MERGE_ROWS_BUDGET + 1;
        let rows: Vec<Vec<_>> = (0..2)
            .map(|parity| {
                let mut ids: Vec<_> = (0..len).map(|offset| offset * 4 + parity).collect();
                if !sorted {
                    ids.reverse();
                }
                ids
            })
            .collect();
        let sources: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(i, ids)| {
                let array = ids.clone().into();
                let segment = if sorted {
                    U64Segment::SortedArray(array)
                } else {
                    U64Segment::Array(array)
                };
                let mut source = fragment(i as u32 + 7, RowIdSequence(vec![segment]));
                source.deletion_vector =
                    Arc::new(DeletionVector::Bitmap([1].into_iter().collect()));
                source
            })
            .collect();
        let originals: Vec<_> = sources
            .iter()
            .map(|source| Arc::downgrade(&source.row_id_sequence))
            .collect();
        let indices: Vec<_> = sources
            .iter()
            .map(|source| Arc::new(RowIdIndex::new(std::slice::from_ref(source)).unwrap()))
            .collect();
        assert!(indices.iter().all(|index| index.merged.is_none()));
        drop(sources);
        if !sorted {
            assert!(originals.iter().all(|source| source.upgrade().is_none()));
        }
        // Reverse the inputs to exercise remapping of the routing directory.
        let combined = RowIdIndex::try_combine(&[&indices[1], &indices[0]])
            .unwrap()
            .unwrap();
        for (i, index) in indices.iter().enumerate() {
            assert!(Arc::ptr_eq(&index.fragments[0], &combined.fragments[1 - i]));
        }
        let mut queries = vec![2, u64::MAX];
        let mut expected = vec![None, None];
        for (i, ids) in rows.iter().enumerate() {
            for offset in [0, 1, 17, len as usize - 1, 0] {
                queries.push(ids[offset]);
                expected.push(
                    (offset != 1).then(|| RowAddress::new_from_parts(i as u32 + 7, offset as u32)),
                );
            }
        }
        // Both the scalar path and the interleaved batch path retain input
        // order, repeated requests, deletion holes and in-range misses.
        for (&id, &address) in queries.iter().zip(&expected) {
            assert_eq!(combined.get(id).unwrap(), address);
        }
        queries = queries.repeat(4);
        expected = expected.repeat(4);
        assert_eq!(combined.get_many(&queries).unwrap(), expected);
        let mut context = lance_core::deepsize::Context::default();
        let units_charge: usize = indices
            .iter()
            .map(|index| index.deep_size_of_children(&mut context))
            .sum();
        assert!(units_charge > 1_000_000);
        assert!(combined.deep_size_of_children(&mut context) < 4096);
        assert!(combined.deep_size_of() > 1_000_000);
        let parents: Vec<_> = indices.iter().map(Arc::downgrade).collect();
        drop(indices);
        assert!(parents.iter().all(|index| index.upgrade().is_none()));
        assert_eq!(combined.get_many(&queries).unwrap(), expected);
    }

    #[test]
    fn combine_disjoint_merged_chunks_shares_allocations_and_reuses_groups() {
        let first =
            RowIdIndex::new(&[fragment(7, RowIdSequence::try_from_iter([20, 10]).unwrap())])
                .unwrap();
        let second = RowIdIndex::new(&[fragment(
            9,
            RowIdSequence::try_from_iter([200, 100]).unwrap(),
        )])
        .unwrap();
        let combined = RowIdIndex::try_combine(&[&second, &first])
            .unwrap()
            .unwrap();
        for (id, input) in [(10, &first), (100, &second)] {
            assert!(Arc::ptr_eq(
                combined.merged.as_ref().unwrap().get(&id).unwrap(),
                input.merged.as_ref().unwrap().get(&id).unwrap()
            ));
        }
        let retained = combined.try_reuse_fragments(&[7], &[]).unwrap().unwrap();
        assert_eq!(
            retained.get_many(&[10, 20, 100]).unwrap(),
            vec![
                Some(RowAddress::new_from_parts(7, 1)),
                Some(RowAddress::new_from_parts(7, 0)),
                None
            ]
        );
    }

    #[rstest::rstest]
    #[case::sorted(false)]
    #[case::shuffled(true)]
    fn combine_merged_preserves_physical_oracle_and_releases_parents(#[case] shuffled: bool) {
        const ROWS: u64 = 31;
        let sources: Vec<_> = (0..3)
            .map(|number| {
                let ids = (0..ROWS)
                    .map(|offset| {
                        let logical = if shuffled { offset * 7 % ROWS } else { offset };
                        logical * 4 + number
                    })
                    .collect::<Vec<_>>();
                let mut source = fragment(
                    number as u32 * 13 + 7,
                    RowIdSequence::try_from_iter(ids).unwrap(),
                );
                source.deletion_vector =
                    Arc::new(DeletionVector::Bitmap([5, 17].into_iter().collect()));
                source
            })
            .collect();
        let originals: Vec<_> = sources
            .iter()
            .map(|source| Arc::downgrade(&source.row_id_sequence))
            .collect();
        let indices: Vec<_> = sources
            .iter()
            .map(|source| Arc::new(RowIdIndex::new(std::slice::from_ref(source)).unwrap()))
            .collect();
        assert!(indices.iter().all(|index| index.merged.is_some()));
        drop(sources);
        assert!(originals.iter().all(|source| source.upgrade().is_none()));
        let combined = RowIdIndex::try_combine(&[&indices[2], &indices[0], &indices[1]])
            .unwrap()
            .unwrap();
        let parents: Vec<_> = indices.iter().map(Arc::downgrade).collect();
        drop(indices);
        assert!(parents.iter().all(|parent| parent.upgrade().is_none()));
        let mut ids: Vec<_> = (0..=ROWS * 4).rev().collect();
        ids.extend([0, 0, u64::MAX]);
        let expected: Vec<_> = ids
            .iter()
            .map(|&id| {
                if id >= ROWS * 4 || id % 4 == 3 {
                    return None;
                }
                // 9 is the inverse of 7 modulo31; this oracle uses only the
                // fixture's arithmetic,not an index or another lookup path.
                let offset = if shuffled { id / 4 * 9 % ROWS } else { id / 4 };
                (![5, 17].contains(&offset))
                    .then(|| RowAddress::new_from_parts((id % 4) as u32 * 13 + 7, offset as u32))
            })
            .collect();
        assert_eq!(combined.get_many(&ids).unwrap(), expected);
        for (&id, &address) in ids.iter().zip(&expected) {
            assert_eq!(combined.get(id).unwrap(), address);
        }
        let appended = combined
            .try_reuse_fragments(
                &[7, 20, 33],
                &[fragment(99, RowIdSequence::from(1000..1007))],
            )
            .unwrap()
            .unwrap();
        assert_eq!(appended.get_many(&ids).unwrap(), expected);
        assert_eq!(
            appended.get_many(&[1000, 1006, 1007]).unwrap(),
            vec![
                Some(RowAddress::new_from_parts(99, 0)),
                Some(RowAddress::new_from_parts(99, 6)),
                None
            ]
        );
    }

    #[test]
    fn combine_merged_rejects_live_duplicates_without_original_sequences() {
        let first =
            RowIdIndex::new(&[fragment(1, RowIdSequence::try_from_iter([9, 3]).unwrap())]).unwrap();
        let second =
            RowIdIndex::new(&[fragment(2, RowIdSequence::try_from_iter([7, 3]).unwrap())]).unwrap();
        let error = RowIdIndex::try_combine(&[&first, &second]).unwrap_err();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(
            error
                .to_string()
                .contains("stable row id 3 is live in multiple fragments")
        );
    }

    #[test]
    fn combine_preserves_overlap_gate_and_live_duplicate_errors() {
        // A deletion makes each large range costly to merge, but preparing its
        // probe remains constant-time. No large test array is needed here.
        let indices: Vec<_> = (0..=MAX_PROBE_DEPTH)
            .map(|id| {
                let mut source = fragment(
                    id as u32,
                    RowIdSequence(vec![U64Segment::Range(0..MERGE_ROWS_BUDGET + 1)]),
                );
                source.deletion_vector = Arc::new(DeletionVector::Bitmap(
                    [u32::from(id != 0)].into_iter().collect(),
                ));
                RowIdIndex::new(&[source]).unwrap()
            })
            .collect();
        let references: Vec<_> = indices.iter().collect();
        let pair = RowIdIndex::try_combine(&references[..2]).unwrap().unwrap();
        assert_eq!(pair.get(0).unwrap(), Some(RowAddress::new_from_parts(1, 0)));
        assert_eq!(pair.get(1).unwrap(), Some(RowAddress::new_from_parts(0, 1)));
        let error = pair.get(2).unwrap_err();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(error.to_string().contains("live in multiple fragments"));
        let error = pair.get_many(&[2; 32]).unwrap_err();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(error.to_string().contains("live in multiple fragments"));
        assert!(
            RowIdIndex::try_combine(&references[..MAX_PROBE_DEPTH as usize])
                .unwrap()
                .is_some()
        );
        assert!(RowIdIndex::try_combine(&references).unwrap().is_none());

        let small = RowIdIndex::new(&[fragment(99, sparse_sequence(3))]).unwrap();
        assert!(small.merged.is_some());
        let copied = RowIdIndex::try_combine(&[&small]).unwrap().unwrap();
        assert_eq!(
            copied.get_many(&[0, 2, 4, 6]).unwrap(),
            small.get_many(&[0, 2, 4, 6]).unwrap()
        );
        assert!(
            RowIdIndex::try_combine(&[&indices[0], &small])
                .unwrap()
                .is_none()
        );
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
        let mut sources = vec![fragment(
            8,
            RowIdSequence(vec![U64Segment::Range(10_000_000..10_000_010)]),
        )];
        let previous = RowIdIndex::new(&sources).unwrap();
        let empty = RowIdIndex::new(&[]).unwrap();
        sources.push(wide);
        // A cheap old merged map must not force a large disjoint addition into
        // the merged path when constructing the complete input would probe.
        for previous in [None, Some(&empty), Some(&previous)] {
            let index = RowIdIndex::new_reusing(&sources, previous).unwrap();
            assert!(index.merged.is_none());
            assert_eq!(
                index.get(6).unwrap(),
                Some(RowAddress::new_from_parts(1, 3))
            );
        }
    }

    #[test]
    fn test_merged_index_keeps_ranges_segments_compact() {
        let row_id_ranges = [100..103, 108..110, 115..119];
        let sequence = RowIdSequence(vec![
            U64Segment::Range(50..53),
            U64Segment::from_sorted_ranges(&row_id_ranges).unwrap(),
        ]);
        let source = fragment(7, sequence);
        let index = RowIdIndex::new(std::slice::from_ref(&source)).unwrap();

        let merged = index.merged.as_ref().expect("small input should be merged");
        assert!(matches!(
            merged.get(&100).unwrap().segments.0,
            U64Segment::Ranges { .. }
        ));

        for (row_id, row_offset) in [
            (50, 0),
            (51, 1),
            (52, 2),
            (100, 3),
            (101, 4),
            (102, 5),
            (108, 6),
            (109, 7),
            (115, 8),
            (116, 9),
            (117, 10),
            (118, 11),
        ] {
            assert_eq!(
                index.get(row_id).unwrap(),
                Some(RowAddress::new_from_parts(7, row_offset)),
                "row id {row_id}"
            );
        }
        for missing in [49, 53, 99, 103, 107, 110, 114, 119, 120] {
            assert_eq!(index.get(missing).unwrap(), None, "row id {missing}");
        }
        assert_eq!(
            index.get_many(&[108, 53, 100, 108]).unwrap(),
            vec![
                Some(RowAddress::new_from_parts(7, 6)),
                None,
                Some(RowAddress::new_from_parts(7, 3)),
                Some(RowAddress::new_from_parts(7, 6)),
            ]
        );
    }

    #[test]
    fn test_overlapping_ranges_chunks_keep_row_addresses() {
        let left_ranges = [100..103, 108..110, 115..119];
        let right_ranges = [103..108, 110..115];
        let sources = [
            fragment(
                7,
                RowIdSequence(vec![U64Segment::from_sorted_ranges(&left_ranges).unwrap()]),
            ),
            fragment(
                8,
                RowIdSequence(vec![U64Segment::from_sorted_ranges(&right_ranges).unwrap()]),
            ),
        ];
        let index = RowIdIndex::new(&sources).unwrap();

        for row_id in 100..119 {
            let (fragment_id, row_offset) = if (100..103).contains(&row_id)
                || (108..110).contains(&row_id)
                || (115..119).contains(&row_id)
            {
                let row_offset = match row_id {
                    100..=102 => row_id - 100,
                    108..=109 => row_id - 105,
                    115..=118 => row_id - 110,
                    _ => unreachable!(),
                };
                (7, row_offset)
            } else {
                let row_offset = match row_id {
                    103..=107 => row_id - 103,
                    110..=114 => row_id - 105,
                    _ => unreachable!(),
                };
                (8, row_offset)
            };
            assert_eq!(
                index.get(row_id).unwrap(),
                Some(RowAddress::new_from_parts(fragment_id, row_offset as u32)),
                "row id {row_id}"
            );
        }
    }

    /// How one fragment of [`interleaved_table`] holds its two far-apart pieces.
    #[derive(Clone, Copy)]
    enum PieceLayout {
        /// One hole array per piece.
        TwoSegments,
        /// One sorted array holding both, as compaction writes it; the far
        /// piece past 2^32 packs it as u64.
        OneSortedArray,
        /// The same, with the far piece below 2^32 so it packs as u32.
        OneSortedArrayU32,
        /// One `Ranges` folded across the gap, as decoding `Range` runs does.
        OneRanges,
    }

    /// `MAX_PROBE_DEPTH + 1` fragments of two far-apart pieces of `piece` ids,
    /// each piece missing its second id, so every fragment span covers every
    /// other while no two pieces overlap. Returns the fragments, `piece` and
    /// `far`, the offset of the high pieces.
    fn interleaved_table(layout: PieceLayout) -> (Vec<FragmentRowIdIndex>, u64, u64) {
        let fragments = MAX_PROBE_DEPTH + 1;
        let piece = MERGE_ROWS_BUDGET / fragments + 1;
        // `Ranges` offsets are u32, so that layout keeps the gap below 4 billion.
        let far = match layout {
            PieceLayout::OneRanges | PieceLayout::OneSortedArrayU32 => 1u64 << 30,
            _ => 1u64 << 40,
        };
        let table = (0..fragments)
            .map(|i| {
                let low = i * piece..(i + 1) * piece;
                let high = far + low.start..far + low.end;
                let segments = match layout {
                    PieceLayout::TwoSegments => [&low, &high]
                        .map(|range| U64Segment::RangeWithHoles {
                            range: range.clone(),
                            holes: vec![range.start + 1].into(),
                        })
                        .to_vec(),
                    PieceLayout::OneSortedArray | PieceLayout::OneSortedArrayU32 => {
                        vec![U64Segment::SortedArray(
                            low.clone()
                                .chain(high.clone())
                                .filter(|id| *id != low.start + 1 && *id != high.start + 1)
                                .collect::<Vec<_>>()
                                .into(),
                        )]
                    }
                    PieceLayout::OneRanges => vec![
                        U64Segment::from_sorted_ranges(&[
                            low.start..low.start + 1,
                            low.start + 2..low.end,
                            high.start..high.start + 1,
                            high.start + 2..high.end,
                        ])
                        .unwrap(),
                    ],
                };
                fragment(i as u32, RowIdSequence(segments))
            })
            .collect();
        (table, piece, far)
    }

    #[rstest::rstest]
    #[case::two_segments(PieceLayout::TwoSegments)]
    #[case::one_sorted_array(PieceLayout::OneSortedArray)]
    #[case::one_sorted_array_u32(PieceLayout::OneSortedArrayU32)]
    #[case::one_ranges(PieceLayout::OneRanges)]
    fn test_probes_pieces_not_fragment_spans(#[case] layout: PieceLayout) {
        let (deep, piece, far) = interleaved_table(layout);
        let index = RowIdIndex::new(&deep).unwrap();
        assert!(
            index.merged.is_none(),
            "pieces do not overlap, so probing wins"
        );

        // Offsets skip the hole: id `start + k` sits at offset `k - 1`, and the
        // high piece starts after the low piece's `piece - 1` live rows.
        let addr = |fragment: u64, offset: u64| {
            Some(RowAddress::new_from_parts(fragment as u32, offset as u32))
        };
        assert_eq!(index.get(0).unwrap(), addr(0, 0));
        assert_eq!(index.get(3 * piece + 1).unwrap(), None, "the hole");
        assert_eq!(
            index.get(far + 3 * piece + 10).unwrap(),
            addr(3, piece - 1 + 9)
        );
        assert_eq!(
            index
                .get_many(&[far + 7 * piece + 2, 5 * piece + 1, 5 * piece + 2])
                .unwrap(),
            vec![addr(7, piece - 1 + 1), None, addr(5, 1)]
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

    #[test]
    fn test_index_resolves_range_segments() {
        let live: Vec<u64> = (1000..3000u64)
            .filter(|v| !(1200..1900).contains(v))
            .collect();
        let runs_sequence = RowIdSequence::from(live.as_slice()).reencode();
        assert!(matches!(
            runs_sequence.segments(),
            [crate::rowids::segment::U64Segment::Ranges { .. }]
        ));
        let index = RowIdIndex::new(&[fragment(7, runs_sequence)]).unwrap();
        for (position, &row_id) in live.iter().enumerate() {
            assert_eq!(
                index.get(row_id).unwrap(),
                Some(RowAddress::new_from_parts(7, position as u32)),
                "row id {row_id}"
            );
        }
        assert_eq!(index.get(1500).unwrap(), None);
        let addr = |position: u32| Some(RowAddress::new_from_parts(7, position));
        assert_eq!(
            index.get_many(&[2999, 1000, 1199, 1900, 1500]).unwrap(),
            vec![
                addr(live.len() as u32 - 1),
                addr(0),
                addr(199),
                addr(200),
                None
            ]
        );
    }
}
