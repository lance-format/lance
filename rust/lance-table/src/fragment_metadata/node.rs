// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Routing and encoded-byte policies for fragment metadata nodes.
//!
//! Node policies use encoded bytes because fragments and actions vary in size.
//! Leaf encoding and object I/O live in `store`; recursive mutation lives in `tree`.
//!
//! An interior is a stored `FragmentTreeNode`. An internal node is the root or
//! an interior. A mutation is a `FragmentTreeMutation`, which wraps one action.

use std::sync::{Arc, OnceLock};

use prost::Message;
use roaring::RoaringTreemap;

use super::content::{Totals, TotalsDelta, action_key, fragment_row_counts, sum_aggregate_values};
use super::store::CheckedInterior;
use crate::format::Fragment;
use crate::format::pb;
use lance_core::{Error, Result};

/// Default interior and leaf split limits. These are independent writer policies.
const DEFAULT_MAX_NODE_BYTES: u64 = 1024 * 1024;
const DEFAULT_MAX_LEAF_BYTES: u64 = 1024 * 1024;
const DEFAULT_MAX_CHILDREN_PER_NODE: u32 = 16;

/// Encoded-byte policy for immutable nodes and their buffers.
#[derive(Debug, Clone)]
pub struct FragmentTreeConfig {
    /// Encoded routing and buffer bytes at which an internal node must flush or
    /// split. A node the writer cannot divide stays above this until a flush
    /// drains its buffer.
    pub max_node_bytes: u64,
    /// Encoded bytes above which a Lance leaf is split. Only a leaf holding one
    /// fragment stays above this, and `hard_capacity_bytes` still bounds it.
    pub max_leaf_bytes: u64,
    /// Pending bytes at which a flush drains children whose batches amortize a
    /// rewrite. Smaller batches stay buffered until `max_node_bytes` forces them.
    pub flush_threshold_bytes: u64,
    /// Complete object limit, including root envelopes and singleton leaves.
    pub hard_capacity_bytes: u64,
    /// Maximum children before splitting. Bounding fanout preserves useful
    /// flush batches as the tree grows.
    pub(crate) max_children_per_node: u32,
}

impl Default for FragmentTreeConfig {
    fn default() -> Self {
        Self {
            max_node_bytes: DEFAULT_MAX_NODE_BYTES,
            max_leaf_bytes: DEFAULT_MAX_LEAF_BYTES,
            flush_threshold_bytes: 256 * 1024,
            hard_capacity_bytes: 64 * 1024 * 1024,
            max_children_per_node: DEFAULT_MAX_CHILDREN_PER_NODE,
        }
    }
}

impl FragmentTreeConfig {
    /// Reject budgets the tree cannot honor. Every byte budget must be nonzero
    /// and at most `hard_capacity_bytes`. `max_node_bytes` and `max_leaf_bytes`
    /// must stay at or below `u64::MAX / 3`, so the sizes derived from them
    /// cannot overflow. A node must allow at least two children.
    /// [`FragmentTree::build`](crate::fragment_metadata::FragmentTree::build),
    /// [`FragmentTree::open`](crate::fragment_metadata::FragmentTree::open) and
    /// [`FragmentTree::into_published`](crate::fragment_metadata::FragmentTree::into_published)
    /// all call this.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::FragmentTreeConfig;
    /// let mut config = FragmentTreeConfig::default();
    /// config.validate().unwrap();
    /// config.max_leaf_bytes = config.hard_capacity_bytes + 1;
    /// assert!(config.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<()> {
        if self.max_node_bytes == 0
            || self.max_leaf_bytes == 0
            || self.flush_threshold_bytes == 0
            || self.hard_capacity_bytes == 0
            || self.max_leaf_bytes > self.hard_capacity_bytes
            || self.max_node_bytes > self.hard_capacity_bytes
            || self.flush_threshold_bytes > self.hard_capacity_bytes
            // The coalesce ceilings and initial packing multiply these two by 3.
            || self.max_node_bytes > u64::MAX / 3
            || self.max_leaf_bytes > u64::MAX / 3
            || self.max_children_per_node < 2
        {
            return Err(Error::invalid_input(format!(
                "fragment metadata tree config is invalid: byte limits must be positive and \
                 at most hard_capacity_bytes, max_node_bytes and max_leaf_bytes must be at most \
                 u64::MAX / 3, and max_children_per_node must be at least 2, got \
                 max_node_bytes={}, max_leaf_bytes={}, flush_threshold_bytes={}, \
                 hard_capacity_bytes={}, max_children_per_node={}",
                self.max_node_bytes,
                self.max_leaf_bytes,
                self.flush_threshold_bytes,
                self.hard_capacity_bytes,
                self.max_children_per_node
            )));
        }
        Ok(())
    }

    pub(crate) fn split_piece_bytes(&self) -> u64 {
        self.max_node_bytes / 2
    }
    pub(crate) fn merge_floor(&self) -> u64 {
        self.max_node_bytes / 4
    }
    pub(crate) fn coalesce_ceiling(&self) -> u64 {
        self.max_node_bytes * 3 / 5
    }
    pub(crate) fn leaf_merge_floor(&self) -> u64 {
        self.max_leaf_bytes / 4
    }
    pub(crate) fn leaf_coalesce_ceiling(&self) -> u64 {
        self.max_leaf_bytes * 3 / 5
    }
}

/// Test-only constructors. Production writers set the public budget fields.
#[cfg(test)]
impl FragmentTreeConfig {
    pub(crate) fn new(max_node_bytes: u64, max_children_per_node: u32) -> Self {
        Self {
            max_node_bytes,
            max_leaf_bytes: max_node_bytes,
            flush_threshold_bytes: max_node_bytes,
            max_children_per_node,
            ..Self::default()
        }
    }

    pub(crate) fn with_max_leaf_bytes(mut self, bytes: u64) -> Self {
        self.max_leaf_bytes = bytes;
        self
    }

    pub(crate) fn with_flush_threshold_bytes(mut self, bytes: u64) -> Self {
        self.flush_threshold_bytes = bytes;
        self
    }

    pub(crate) fn with_hard_capacity_bytes(mut self, bytes: u64) -> Self {
        self.hard_capacity_bytes = bytes;
        self
    }
}

/// Sorted child pivots and buffered mutations. Replay orders mutations
/// by action sequence within each fragment.
#[derive(Debug, Clone, Default)]
pub(super) struct InteriorNode {
    pub(super) children: Vec<pb::FragmentTreeChild>,
    pub(super) buffer: Vec<pb::FragmentTreeMutation>,
}

/// Derived offsets into an immutable buffer. The wire representation stays a
/// plain action sequence; a sparse lookup need not examine unrelated actions.
#[derive(Debug, Clone)]
pub(super) struct BufferIndex {
    offsets: Vec<(u64, usize)>,
    sequences: OnceLock<DecodedSequences>,
}

impl BufferIndex {
    pub(super) fn new(buffer: &[pb::FragmentTreeMutation]) -> Self {
        let mut offsets: Vec<_> = buffer
            .iter()
            .enumerate()
            .map(|(offset, mutation)| (action_key(mutation), offset))
            .collect();
        offsets.sort_unstable();
        Self {
            offsets,
            sequences: OnceLock::new(),
        }
    }

    /// An empty per-operation set that shares this root buffer's sequences.
    pub(super) fn sequences(&self, root_buffer: &[pb::FragmentTreeMutation]) -> DecodedSequences {
        self.sequences
            .get_or_init(|| DecodedSequences::new(root_buffer))
            .clone()
    }

    pub(super) fn for_fragment(&self, id: u64) -> impl Iterator<Item = usize> + '_ {
        let start = self.offsets.partition_point(|(key, _)| *key < id);
        self.offsets[start..]
            .iter()
            .take_while(move |(key, _)| *key == id)
            .map(|(_, offset)| *offset)
    }
}

/// The action sequences one operation has decoded. Opening a tree checks the
/// root buffer, and decoding a node checks its own buffer, but replay sees
/// only the actions routed to one leaf. Every other buffer the operation
/// decodes is admitted here, so a sequence two of them share is rejected.
#[derive(Debug, Clone, Default)]
pub(super) struct DecodedSequences {
    /// Shared by every operation on one root buffer, so a point read copies
    /// nothing up front.
    root: Arc<RoaringTreemap>,
    below_root: RoaringTreemap,
}

impl DecodedSequences {
    pub(super) fn new(root_buffer: &[pb::FragmentTreeMutation]) -> Self {
        Self {
            root: Arc::new(
                root_buffer
                    .iter()
                    .map(|tagged| tagged.action_sequence)
                    .collect(),
            ),
            below_root: RoaringTreemap::new(),
        }
    }

    /// Admit the buffer of the interior at `path`.
    pub(super) fn admit(&mut self, path: &str, buffer: &[pb::FragmentTreeMutation]) -> Result<()> {
        for tagged in buffer {
            if self.root.contains(tagged.action_sequence)
                || !self.below_root.insert(tagged.action_sequence)
            {
                return Err(super::validation::corrupt(format!(
                    "Interior {path} buffers action sequence {}, which another buffer read by this operation also holds",
                    tagged.action_sequence
                )));
            }
        }
        Ok(())
    }

    /// Admit a checked interior, whose decode proved that its own buffer
    /// repeats nothing. A collision takes [`Self::admit`], which names the
    /// repeated sequence that comes first in the buffer.
    pub(super) fn admit_interior(&mut self, path: &str, interior: &CheckedInterior) -> Result<()> {
        let sequences = &interior.sequences;
        if self.root.is_disjoint(sequences) && self.below_root.is_disjoint(sequences) {
            self.below_root |= sequences;
            return Ok(());
        }
        self.admit(path, &interior.node.buffer)
    }
}

/// Fragment protobuf bytes, used to bound candidate encoding work and report amplification.
pub(super) fn fragment_logical_bytes(fragment: &Fragment) -> u64 {
    pb::DataFragment::from(fragment).encoded_len() as u64
}

/// Fragment protobuf bytes before Lance encoding; this is not the leaf capacity metric.
pub(super) fn leaf_logical_bytes(fragments: &[Fragment]) -> u64 {
    fragments.iter().map(fragment_logical_bytes).sum()
}

fn varint_bytes(mut value: u64) -> u64 {
    let mut bytes = 1;
    while value >= 0x80 {
        value >>= 7;
        bytes += 1;
    }
    bytes
}

fn repeated_message_bytes(message: &impl Message) -> u64 {
    let payload_bytes = message.encoded_len() as u64;
    1 + varint_bytes(payload_bytes) + payload_bytes
}

/// Logical byte size of an internal node = its exact encoded protobuf size.
pub(super) fn internal_logical_bytes(
    children: &[pb::FragmentTreeChild],
    buffer: &[pb::FragmentTreeMutation],
) -> u64 {
    children.iter().map(repeated_message_bytes).sum::<u64>()
        + buffer.iter().map(repeated_message_bytes).sum::<u64>()
}

/// Whether an internal node violates its encoded-byte or fanout limit. This is
/// the structural bound. A node over it must drain or split.
pub(super) fn internal_overflows(
    children: &[pb::FragmentTreeChild],
    buffer: &[pb::FragmentTreeMutation],
    config: &FragmentTreeConfig,
) -> bool {
    overflows_with(children, internal_logical_bytes(&[], buffer), config)
}

/// [`internal_overflows`] for a buffer whose encoded bytes are already known.
pub(super) fn overflows_with(
    children: &[pb::FragmentTreeChild],
    buffer_bytes: u64,
    config: &FragmentTreeConfig,
) -> bool {
    children.len() as u32 > config.max_children_per_node
        || internal_logical_bytes(children, &[]) + buffer_bytes >= config.max_node_bytes
}

/// Whether pending bytes reach the flush threshold. This is the soft bound.
/// The writer drains children whose batches amortize a rewrite and keeps the
/// rest buffered until structural pressure forces progress.
pub(super) fn buffer_pressured(
    buffer: &[pb::FragmentTreeMutation],
    config: &FragmentTreeConfig,
) -> bool {
    internal_logical_bytes(&[], buffer) >= config.flush_threshold_bytes
}

/// Index of the child owning `key`: the rightmost child whose `min_key` is at
/// most `key`, and the first child for keys below every fence.
///
/// Routing invariant: sibling `min_key` values are the only fences. The last
/// child's exclusive end is inherited from its parent. At the root it is
/// `next_fragment_id`, which is never above the format's `2^32` end.
/// Keys below the first fence belong to the first child, so an emptied
/// node must be dropped rather than kept with `min_key = 0`.
pub(super) fn child_index_for(children: &[pb::FragmentTreeChild], key: u64) -> usize {
    match children.binary_search_by(|c| c.min_key.cmp(&key)) {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) => i - 1,
    }
}

/// Pin the first child's fence to the node's lower bound. A rebuilt child list
/// starts at its first stored key, but the range it owns starts where the
/// parent says it does. That is 0 at the root and the parent's entry below it.
pub(super) fn fence(children: &mut [pb::FragmentTreeChild], lower_bound: u64) {
    if let Some(first) = children.first_mut() {
        debug_assert!(lower_bound <= first.min_key);
        first.min_key = lower_bound;
    }
}

/// The format's exclusive end for a root's last child, and the highest
/// `next_fragment_id`. Tree traversals end the root at `next_fragment_id`.
pub(super) const ROOT_EXCLUSIVE_END: u64 = 1u64 << 32;

/// Exclusive end of `children[index]` inside a parent whose last child ends
/// at `parent_end`.
pub(super) fn exclusive_end(
    children: &[pb::FragmentTreeChild],
    index: usize,
    parent_end: u64,
) -> u64 {
    children
        .get(index + 1)
        .map(|child| child.min_key)
        .unwrap_or(parent_end)
}

/// Pair each child with the exclusive end of its range inside a parent whose
/// last child ends at `parent_end`.
pub(super) fn with_exclusive_ends(
    children: Vec<pb::FragmentTreeChild>,
    parent_end: u64,
) -> impl DoubleEndedIterator<Item = (pb::FragmentTreeChild, u64)> + ExactSizeIterator {
    let ends: Vec<u64> = (0..children.len())
        .map(|index| exclusive_end(&children, index, parent_end))
        .collect();
    children.into_iter().zip(ends)
}

/// Build a child reference for a leaf that has just been written.
pub(super) fn leaf_ref(
    path: String,
    fragments: &[Fragment],
    object_size: u64,
    materialized_through_action_sequence: u64,
) -> Result<pb::FragmentTreeChild> {
    let (total_rows, visible_rows) = fragment_row_counts(fragments)?;
    Ok(pb::FragmentTreeChild {
        path,
        base_id: None,
        min_key: fragments.first().map(|f| f.id).unwrap_or(0),
        num_keys: fragments.len() as u64,
        height: 0,
        num_children: 0,
        total_rows,
        object_size,
        materialized_through_action_sequence,
        visible_rows,
    })
}

/// Build a child reference for an internal node that has just been written.
pub(super) fn internal_ref(
    path: String,
    children: &[pb::FragmentTreeChild],
    buffer: &[pb::FragmentTreeMutation],
    object_size: u64,
) -> Result<pb::FragmentTreeChild> {
    let totals = Totals {
        fragments: sum_aggregate_values(children.iter().map(|child| child.num_keys), "Fragments")?,
        rows: sum_aggregate_values(
            children.iter().map(|child| child.total_rows),
            "physical rows",
        )?,
        visible_rows: sum_aggregate_values(
            children.iter().map(|child| child.visible_rows),
            "visible rows",
        )?,
    }
    .apply(TotalsDelta::of(buffer)?)?;
    let height = children
        .iter()
        .map(|c| c.height)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| super::validation::corrupt("Node height exceeds u32"))?;
    Ok(pb::FragmentTreeChild {
        path,
        base_id: None,
        min_key: children.first().map(|c| c.min_key).unwrap_or(0),
        num_keys: totals.fragments,
        height,
        num_children: children.len() as u32,
        total_rows: totals.rows,
        object_size,
        materialized_through_action_sequence: 0,
        visible_rows: totals.visible_rows,
    })
}

/// Is this child underflowing (a merge candidate)? Leaves underflow by bytes.
/// Interiors must be sparse by both direct-child count and their exact
/// encoded size so a hot buffer is never merged as "small."
pub(super) fn is_underflow(child: &pb::FragmentTreeChild, config: &FragmentTreeConfig) -> bool {
    if child.height == 0 {
        child.object_size <= config.leaf_merge_floor()
    } else {
        child.num_children < (config.max_children_per_node / 4).max(1)
            && child.object_size <= config.merge_floor()
    }
}

/// One interior that itself has one child. The format forbids it and no
/// sibling remains to repair it.
pub(super) fn is_single_child_chain(children: &[pb::FragmentTreeChild]) -> bool {
    matches!(children, [only] if only.height > 0 && only.num_children == 1)
}

/// Parents of leaves can combine all their history. They have no deeper buffers,
/// and ancestors drain whole buckets into them. Above interiors, older history
/// may interleave with deeper buffers, so the sequence boundary still applies.
pub(super) fn combine_from(children: &[pb::FragmentTreeChild], contiguous_from: u64) -> u64 {
    if children.iter().all(|child| child.height == 0) {
        0
    } else {
        contiguous_from
    }
}

/// Partition a buffer into one bucket per child by the routing fences. The
/// returned vec has `children.len()` buckets; the buffer is consumed. Callers
/// must not pass an empty `children` slice.
pub(super) fn partition_buffer_by_child(
    children: &[pb::FragmentTreeChild],
    buffer: Vec<pb::FragmentTreeMutation>,
) -> Vec<Vec<pb::FragmentTreeMutation>> {
    let mut buckets: Vec<Vec<pb::FragmentTreeMutation>> = vec![Vec::new(); children.len()];
    for tagged in buffer {
        let idx = child_index_for(children, action_key(&tagged));
        buckets[idx].push(tagged);
    }
    buckets
}

/// Reject buckets that routing could only have produced from a corrupt node.
/// A target outside the child's inherited range, or at or below its leaf
/// watermark, is already applied or was never owned by this child.
pub(super) fn validate_routed(
    children: &[pb::FragmentTreeChild],
    buckets: &[Vec<pb::FragmentTreeMutation>],
    parent_end: u64,
) -> Result<()> {
    for (index, (child, bucket)) in children.iter().zip(buckets).enumerate() {
        let exclusive_end = exclusive_end(children, index, parent_end);
        for tagged in bucket {
            let target = action_key(tagged);
            if target < child.min_key || target >= exclusive_end {
                return Err(super::validation::corrupt(format!(
                    "Mutation action_sequence={} targets fragment {target} outside [{}, {exclusive_end})",
                    tagged.action_sequence, child.min_key
                )));
            }
            if child.height == 0
                && tagged.action_sequence <= child.materialized_through_action_sequence
            {
                return Err(super::validation::corrupt(format!(
                    "Mutation action_sequence={} for fragment {target} is at or below the watermark {} of leaf {}",
                    tagged.action_sequence, child.materialized_through_action_sequence, child.path
                )));
            }
        }
    }
    Ok(())
}

/// Children and buffer of one piece of a split internal node.
type SplitPiece = (Vec<pb::FragmentTreeChild>, Vec<pb::FragmentTreeMutation>);

/// Split an internal node's children and buffer into contiguous pieces targeting
/// `piece_bytes` and at most `max_children_per_node` children each. The buffer
/// follows its child by key range. One child and its mutations may exceed the
/// byte target alone, and folding a lone tail may reach twice the target.
///
/// A piece may hold one child. A child whose mutations fill the target sits
/// alone at any position, and a lone tail stays when neither repair fits.
/// Readers reject such an interior, so the caller must repair or refuse it
/// before publishing.
pub(super) fn split_internal(
    children: Vec<pb::FragmentTreeChild>,
    buffer: Vec<pb::FragmentTreeMutation>,
    piece_bytes: u64,
    max_children_per_node: u32,
) -> Vec<SplitPiece> {
    if children.is_empty() {
        return vec![(children, buffer)];
    }
    let action_buckets = partition_buffer_by_child(&children, buffer);
    let max_children_per_node = max_children_per_node as usize;
    let mut pieces = Vec::new();
    let mut piece_children = Vec::new();
    let mut piece_buffer = Vec::new();
    let mut encoded_bytes = 0u64;

    for (child, actions) in children.into_iter().zip(action_buckets) {
        let unit_bytes = repeated_message_bytes(&child)
            + actions.iter().map(repeated_message_bytes).sum::<u64>();
        let exceeds_piece = !piece_children.is_empty()
            && (piece_children.len() >= max_children_per_node
                || encoded_bytes + unit_bytes > piece_bytes);
        if exceeds_piece {
            pieces.push((
                std::mem::take(&mut piece_children),
                std::mem::take(&mut piece_buffer),
            ));
            encoded_bytes = 0;
        }
        encoded_bytes += unit_bytes;
        piece_children.push(child);
        piece_buffer.extend(actions);
    }
    pieces.push((piece_children, piece_buffer));
    if let [.., left, tail] = pieces.as_mut_slice()
        && tail.0.len() == 1
    {
        borrow_for_lone_tail(left, tail, piece_bytes);
    }
    fold_lone_tail(&mut pieces, piece_bytes, max_children_per_node);
    pieces
}

/// Greedy byte packing can leave one child at the right edge. Borrow the
/// preceding range when both resulting nodes still fit the target. Its
/// mutations must move with the routing fence, including inserts in gaps.
fn borrow_for_lone_tail(left: &mut SplitPiece, tail: &mut SplitPiece, piece_bytes: u64) {
    let (left_children, left_buffer) = left;
    let (right_children, right_buffer) = tail;
    if left_children.len() > 2 {
        let borrowed = &left_children[left_children.len() - 1];
        let extra = repeated_message_bytes(borrowed)
            + left_buffer
                .iter()
                .filter(|mutation| action_key(mutation) >= borrowed.min_key)
                .map(repeated_message_bytes)
                .sum::<u64>();
        if internal_logical_bytes(right_children, right_buffer) + extra <= piece_bytes {
            let fence = borrowed.min_key;
            let (kept, moved): (Vec<_>, Vec<_>) = std::mem::take(left_buffer)
                .into_iter()
                .partition(|mutation| action_key(mutation) < fence);
            *left_buffer = kept;
            right_buffer.extend(moved);
            right_children.insert(0, left_children.remove(left_children.len() - 1));
        }
    }
}

/// Fold a lone child into its neighbor when the merged node fits within two
/// pieces and the fanout limit.
fn fold_lone_tail(pieces: &mut Vec<SplitPiece>, piece_bytes: u64, max_children_per_node: usize) {
    if pieces.len() > 1 {
        let last = pieces.len() - 1;
        let (left_children, left_buffer) = &pieces[last - 1];
        let (right_children, right_buffer) = &pieces[last];
        if right_children.len() == 1
            && left_children.len() < max_children_per_node
            && internal_logical_bytes(left_children, left_buffer)
                + internal_logical_bytes(right_children, right_buffer)
                < 2 * piece_bytes
        {
            let (children, buffer) = pieces.pop().expect("pieces has more than one");
            let (left_children, left_buffer) = pieces.last_mut().expect("pieces is not empty");
            left_children.extend(children);
            left_buffer.extend(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment_metadata::action;
    use crate::fragment_metadata::support::make_backfill_data_file;
    use rstest::rstest;

    fn tagged(action: pb::FragmentAction) -> pb::FragmentTreeMutation {
        pb::FragmentTreeMutation {
            action_sequence: 1,
            action: Some(action),
            fragment_count_delta: 0,
            total_rows_delta: 0,
            visible_rows_delta: 0,
        }
    }

    fn child_ref(index: u64, object_size: u64, height: u32) -> pb::FragmentTreeChild {
        pb::FragmentTreeChild {
            path: format!("_bt/node/{index}.node"),
            base_id: None,
            min_key: index * 10,
            num_keys: 10,
            height,
            num_children: u32::from(height > 0),
            total_rows: 10,
            object_size,
            materialized_through_action_sequence: 0,
            visible_rows: 10,
        }
    }

    #[rstest]
    #[case::zero_leaf_limit(FragmentTreeConfig {
        max_leaf_bytes: 0,
        ..FragmentTreeConfig::default()
    })]
    #[case::leaf_limit_over_hard_capacity(FragmentTreeConfig {
        max_leaf_bytes: FragmentTreeConfig::default().hard_capacity_bytes + 1,
        ..FragmentTreeConfig::default()
    })]
    // An ordinary hard capacity would reject this first and hide the bound.
    #[case::node_limit_over_overflow_bound(FragmentTreeConfig {
        max_node_bytes: u64::MAX / 3 + 1,
        hard_capacity_bytes: u64::MAX,
        ..FragmentTreeConfig::default()
    })]
    fn rejects_byte_limits_outside_the_writer_bounds(#[case] config: FragmentTreeConfig) {
        let error = config.validate().unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }

    #[test]
    fn internal_logical_bytes_matches_protobuf_encoding() {
        let children = vec![child_ref(0, 1_000, 0), child_ref(1, 2_000, 0)];
        let buffer = vec![
            tagged(action::remove_fragment(3)),
            tagged(action::add_data_file(7, &make_backfill_data_file(7, 0))),
        ];
        let encoded = pb::FragmentTreeNode {
            children: children.clone(),
            buffer: buffer.clone(),
        }
        .encoded_len() as u64;

        assert_eq!(internal_logical_bytes(&children, &buffer), encoded);
    }

    #[test]
    fn split_internal_uses_parent_encoding_not_child_payload_sizes() {
        let children = (0..8)
            .map(|index| child_ref(index, 1024 * 1024, 0))
            .collect();
        let pieces = split_internal(children, Vec::new(), 1024, 4);

        assert_eq!(pieces.len(), 2);
        assert!(pieces.iter().all(|(children, _)| children.len() == 4));
        assert!(
            pieces
                .iter()
                .all(|(children, buffer)| internal_logical_bytes(children, buffer) <= 1024)
        );
    }

    #[test]
    fn split_internal_repairs_singleton_tail_with_its_gap_mutations() {
        let children = (0..7).map(|index| child_ref(index, 1, 0)).collect();
        let buffer = vec![tagged(action::remove_fragment(55))];
        let pieces = split_internal(children, buffer, 1024, 3);
        assert_eq!(
            pieces
                .iter()
                .map(|(children, _)| children.len())
                .collect::<Vec<_>>(),
            vec![3, 2, 2]
        );
        assert!(pieces[1].1.is_empty());
        assert_eq!(action_key(&pieces[2].1[0]), 55);
        assert_eq!(pieces[2].0[0].min_key, 50);
        assert!(
            pieces
                .iter()
                .all(|(children, buffer)| internal_logical_bytes(children, buffer) <= 1024)
        );
    }

    #[test]
    fn split_internal_leaves_a_hot_child_alone_in_its_piece() {
        let children = (0..3).map(|index| child_ref(index, 1, 0)).collect();
        let buffer = (0..64)
            .map(|_| tagged(action::remove_fragment(15)))
            .collect();
        let pieces = split_internal(children, buffer, 128, 4);
        assert_eq!(
            pieces
                .iter()
                .map(|(children, _)| children.len())
                .collect::<Vec<_>>(),
            vec![1, 1, 1]
        );
        assert_eq!(pieces[1].1.len(), 64);
    }

    #[test]
    fn hot_interior_is_not_an_underflow_merge_candidate() {
        let config = FragmentTreeConfig::new(1024, 16);
        let mut child = child_ref(0, 512, 1);
        child.num_children = 1;
        assert!(!is_underflow(&child, &config));

        child.object_size = 128;
        assert!(is_underflow(&child, &config));
    }

    #[test]
    fn internal_overflow_checks_encoded_bytes_and_fanout() {
        let config = FragmentTreeConfig::new(256, 4);
        let four_children: Vec<_> = (0..4).map(|index| child_ref(index, 1_000_000, 0)).collect();
        assert!(!internal_overflows(&four_children, &[], &config));

        let five_children: Vec<_> = (0..5).map(|index| child_ref(index, 1, 0)).collect();
        assert!(internal_overflows(&five_children, &[], &config));

        let hot_buffer: Vec<_> = (0..20)
            .map(|fragment_id| tagged(action::remove_fragment(fragment_id)))
            .collect();
        assert!(internal_overflows(&four_children, &hot_buffer, &config));
    }

    #[test]
    fn buffer_pressure_is_not_structural_overflow() {
        let config = FragmentTreeConfig::new(4096, 4).with_flush_threshold_bytes(64);
        let children: Vec<_> = (0..4).map(|index| child_ref(index, 1024, 0)).collect();
        let pending: Vec<_> = (0..20)
            .map(|fragment_id| tagged(action::remove_fragment(fragment_id)))
            .collect();
        assert!(buffer_pressured(&pending, &config));
        assert!(!internal_overflows(&children, &pending, &config));
    }
}
