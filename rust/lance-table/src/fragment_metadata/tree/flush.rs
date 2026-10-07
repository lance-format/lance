// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bε drain policy: which children a pressured buffer drains, and how many at once.
//!
//! Qualified for root-to-leaf scatter. An interior child's drain recurses, so
//! its object size understates the work below it.

use crate::format::pb::{self, fragment_action::Action};
use crate::fragment_metadata::node::{self, FragmentTreeConfig};

// Drain at most eight children and 32 MiB of encoded child-object bytes
// per batch. Decoded fragments and replacement buffers can use more memory.
// The same count bounds a store's leaf PUTs in flight, so concurrent drains
// share one window of encoded leaves.
pub(in crate::fragment_metadata) const MAX_CONCURRENT_LEAF_DRAINS: usize = 8;
const MAX_BATCH_OBJECT_BYTES: u64 = 32 * 1024 * 1024;

/// Soft-pressure target. Once pending edits reach this fraction of the
/// child object's encoded size, further batching has diminishing value against
/// the read and write of that object.
const REWRITE_AMORTIZATION_DIVISOR: u64 = 16;

/// Pending bytes a child must hold before draining it under semantic pressure.
///
/// The fair share of the node's remaining routing room guarantees progress by
/// pigeonhole no later than structural pressure. The object fraction keeps a
/// narrow directory from waiting on a share larger than a rewrite deserves.
pub(super) fn amortization_gate(fair_share: u64, child: &pb::FragmentTreeChild) -> u64 {
    fair_share
        .min(child.object_size / REWRITE_AMORTIZATION_DIVISOR)
        .max(1)
}

/// A node's pending actions partitioned by child, with each bucket's encoded
/// bytes. A drain consumes its bucket and splices new children in its place,
/// which leaves every other bucket and its size unchanged, so one partition
/// serves every flush pass.
pub(super) struct Buckets {
    actions: Vec<Vec<pb::FragmentTreeMutation>>,
    bytes: Vec<u64>,
}

impl From<Vec<Vec<pb::FragmentTreeMutation>>> for Buckets {
    fn from(actions: Vec<Vec<pb::FragmentTreeMutation>>) -> Self {
        let bytes = actions
            .iter()
            .map(|bucket| node::internal_logical_bytes(&[], bucket))
            .collect();
        Self { actions, bytes }
    }
}

impl Buckets {
    pub(super) fn actions(&self) -> &[Vec<pb::FragmentTreeMutation>] {
        &self.actions
    }

    pub(super) fn total_bytes(&self) -> u64 {
        self.bytes.iter().sum()
    }

    pub(super) fn pending(&self) -> usize {
        self.actions.iter().map(Vec::len).sum()
    }

    pub(super) fn take(&mut self, idx: usize) -> Vec<pb::FragmentTreeMutation> {
        self.bytes[idx] = 0;
        std::mem::take(&mut self.actions[idx])
    }

    /// Replace a drained child's empty bucket with one per new child.
    pub(super) fn replace(&mut self, idx: usize, children: usize) {
        self.actions.splice(
            idx..idx + 1,
            std::iter::repeat_with(Vec::new).take(children),
        );
        self.bytes
            .splice(idx..idx + 1, std::iter::repeat_n(0, children));
    }

    pub(super) fn into_buffer(self) -> Vec<pb::FragmentTreeMutation> {
        self.actions.into_iter().flatten().collect()
    }
}

/// Whether a buffered action removes a fragment the subtree below still
/// holds. A squashed append-then-remove keeps a removal that retires nothing.
pub(super) fn retires_fragment(mutation: &pb::FragmentTreeMutation) -> bool {
    mutation.fragment_count_delta < 0
        && matches!(
            mutation
                .action
                .as_ref()
                .and_then(|action| action.action.as_ref()),
            Some(Action::RemoveFragment(_))
        )
}

/// Whether a buffered action overwrites a record the subtree below holds.
/// After squashing, a whole-record upsert that does not change the fragment
/// count replaces a stored record.
pub(super) fn replaces_fragment(mutation: &pb::FragmentTreeMutation) -> bool {
    mutation.fragment_count_delta == 0
        && matches!(
            mutation
                .action
                .as_ref()
                .and_then(|action| action.action.as_ref()),
            Some(Action::UpsertFragment(_))
        )
}

/// Fragments of `child` that draining `actions` removes.
pub(super) fn retired_keys(actions: &[pb::FragmentTreeMutation]) -> u64 {
    actions
        .iter()
        .filter(|action| retires_fragment(action))
        .count() as u64
}

/// Estimated child bytes reclaimed by the removed fragments.
fn reclaimed_bytes(child: &pb::FragmentTreeChild, actions: &[pb::FragmentTreeMutation]) -> u64 {
    child.object_size * retired_keys(actions).min(child.num_keys) / child.num_keys.max(1)
}

pub(super) fn fair_share(children: &[pb::FragmentTreeChild], config: &FragmentTreeConfig) -> u64 {
    let routing = node::internal_logical_bytes(children, &[]);
    config.max_node_bytes.saturating_sub(routing) / children.len().max(1) as u64
}

/// Whether the buffer holding a batch is over a byte budget.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Pressure {
    /// Over the semantic or structural budget: drain batches that amortize
    /// their rewrite, counting both pending and reclaimed bytes.
    Pressured,
    /// Within budget: drain only batches whose removals retire at least the
    /// amortization fraction of their child, so shrinking tables contract.
    Relaxed,
}

/// Children worth draining now, fullest first, bounded for one batch.
///
/// A byte overflow always leaves one bucket at or above the fair share by
/// pigeonhole, so structural pressure needs no separate path here. When the
/// result is empty only routing or fanout overflows, which the caller splits.
pub(super) fn select_children(
    children: &[pb::FragmentTreeChild],
    buckets: &Buckets,
    config: &FragmentTreeConfig,
    io_parallelism: usize,
    pressure: Pressure,
) -> Vec<usize> {
    let fair_share = fair_share(children, config);
    let mut ranked: Vec<_> = buckets
        .actions
        .iter()
        .zip(&buckets.bytes)
        .enumerate()
        .filter(|(_, (actions, _))| !actions.is_empty())
        .map(|(idx, (actions, bytes))| {
            let reclaimed = reclaimed_bytes(&children[idx], actions);
            let credit = match pressure {
                Pressure::Pressured => bytes + reclaimed,
                Pressure::Relaxed => reclaimed,
            };
            (credit, idx)
        })
        .collect();
    ranked.sort_unstable_by(|a, b| b.cmp(a));
    let worthwhile = ranked
        .iter()
        .filter(|(credit, idx)| {
            let gate = match pressure {
                Pressure::Pressured => amortization_gate(fair_share, &children[*idx]),
                Pressure::Relaxed => {
                    (children[*idx].object_size / REWRITE_AMORTIZATION_DIVISOR).max(1)
                }
            };
            *credit >= gate
        })
        .map(|(_, idx)| *idx);
    let limit = io_parallelism.clamp(1, MAX_CONCURRENT_LEAF_DRAINS);
    let mut selected = Vec::with_capacity(limit);
    let mut batch_bytes = 0;
    for idx in worthwhile {
        if !selected.is_empty()
            && (children[idx].height > 0
                || selected.len() >= limit
                || batch_bytes + children[idx].object_size > MAX_BATCH_OBJECT_BYTES)
        {
            break;
        }
        selected.push(idx);
        batch_bytes += children[idx].object_size;
        // Interior drains recurse, so running them together would multiply the bound.
        if children[idx].height > 0 {
            break;
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment_metadata::action;
    use rstest::rstest;

    fn children(count: usize, height: u32, object_size: u64) -> Vec<pb::FragmentTreeChild> {
        vec![
            pb::FragmentTreeChild {
                height,
                object_size,
                ..Default::default()
            };
            count
        ]
    }

    fn bucket(id: u64, actions: usize) -> Vec<pb::FragmentTreeMutation> {
        vec![
            pb::FragmentTreeMutation {
                action: Some(action::remove_fragment(id)),
                fragment_count_delta: -1,
                ..Default::default()
            };
            actions
        ]
    }

    #[rstest]
    #[case::below_the_gate_waits(1, vec![])]
    #[case::worthwhile_batches_drain_together(64, vec![9, 8, 7, 6, 5, 4, 3, 2])]
    #[test]
    fn selection_respects_the_gate_and_concurrency(
        #[case] actions_per_child: usize,
        #[case] expected: Vec<usize>,
    ) {
        let children = children(10, 0, 1024 * 1024);
        let config = FragmentTreeConfig::default();
        let buckets: Vec<_> = (0..10)
            .map(|id| bucket(id, actions_per_child * (id as usize + 1) * 200))
            .collect();
        assert_eq!(
            select_children(&children, &buckets.into(), &config, 8, Pressure::Pressured),
            expected
        );
    }

    #[test]
    fn fair_share_binds_a_wide_directory() {
        let children = children(256, 0, 1024 * 1024);
        let config = FragmentTreeConfig::default();
        let share = fair_share(&children, &config);
        assert!(share < 1024 * 1024 / REWRITE_AMORTIZATION_DIVISOR);
        assert_eq!(amortization_gate(share, &children[0]), share);
        let narrow = children[..2].to_vec();
        let share = fair_share(&narrow, &config);
        assert_eq!(
            amortization_gate(share, &narrow[0]),
            1024 * 1024 / REWRITE_AMORTIZATION_DIVISOR
        );
    }

    #[test]
    fn interior_children_drain_alone_and_leaves_respect_the_store_limit() {
        let config = FragmentTreeConfig::default();
        let mut mixed = children(4, 0, 4096);
        mixed[3].height = 1;
        let buckets: Buckets = (0..4).map(|id| bucket(id, 4000)).collect::<Vec<_>>().into();
        assert_eq!(
            select_children(&mixed, &buckets, &config, 8, Pressure::Pressured),
            vec![3]
        );
        let leaves = children(4, 0, 4096);
        assert_eq!(
            select_children(&leaves, &buckets, &config, 2, Pressure::Pressured),
            vec![3, 2]
        );
        assert!(
            select_children(
                &leaves,
                &vec![vec![]; 4].into(),
                &config,
                8,
                Pressure::Pressured
            )
            .is_empty()
        );
    }

    fn leaves_with_keys(count: usize, num_keys: u64) -> Vec<pb::FragmentTreeChild> {
        let mut leaves = children(count, 0, 1024 * 1024);
        for leaf in &mut leaves {
            leaf.num_keys = num_keys;
        }
        leaves
    }

    #[rstest]
    #[case::emptied_leaf_drains(2048, vec![0])]
    #[case::one_sixteenth_drains(128, vec![0])]
    #[case::sparse_removals_wait(127, vec![])]
    #[test]
    fn relaxed_selection_drains_removals_that_shrink_a_leaf(
        #[case] removed: usize,
        #[case] expected: Vec<usize>,
    ) {
        let leaves = leaves_with_keys(2, 2048);
        let buckets = vec![bucket(0, removed), vec![]];
        let config = FragmentTreeConfig::default();
        assert_eq!(
            select_children(&leaves, &buckets.into(), &config, 8, Pressure::Relaxed),
            expected
        );
    }

    #[test]
    fn relaxed_selection_ignores_removals_squashed_with_their_append() {
        let leaves = leaves_with_keys(1, 2048);
        let mut squashed = bucket(0, 2048);
        for removal in &mut squashed {
            removal.fragment_count_delta = 0;
        }
        let config = FragmentTreeConfig::default();
        assert!(
            select_children(
                &leaves,
                &vec![squashed].into(),
                &config,
                8,
                Pressure::Relaxed
            )
            .is_empty()
        );
    }

    #[test]
    fn relaxed_selection_ignores_edits_that_remove_nothing() {
        let leaves = leaves_with_keys(1, 2048);
        let upserts = vec![
            pb::FragmentTreeMutation {
                action: Some(action::upsert_fragment(&crate::format::Fragment::new(7))),
                ..Default::default()
            };
            4096
        ];
        let config = FragmentTreeConfig::default();
        assert!(
            select_children(
                &leaves,
                &vec![upserts].into(),
                &config,
                8,
                Pressure::Relaxed
            )
            .is_empty()
        );
    }

    #[test]
    fn reclaimed_bytes_lift_removals_over_the_pressured_gate() {
        // Removing every key of a leaf buffers far fewer bytes than the
        // leaf's amortization share, so only the reclaim credit drains it.
        let leaves = leaves_with_keys(1, 2048);
        let emptied = vec![bucket(0, 2048)];
        let config = FragmentTreeConfig::default();
        assert!(
            node::internal_logical_bytes(&[], &emptied[0])
                < amortization_gate(fair_share(&leaves, &config), &leaves[0])
        );
        assert_eq!(
            select_children(&leaves, &emptied.into(), &config, 8, Pressure::Pressured),
            vec![0]
        );
    }

    #[test]
    fn selection_bounds_encoded_bytes_in_flight() {
        let oversized = children(3, 0, MAX_BATCH_OBJECT_BYTES / 2 + 1);
        let config = FragmentTreeConfig::default().with_hard_capacity_bytes(u64::MAX / 4);
        let buckets: Vec<_> = (0..3).map(|id| bucket(id, 200_000)).collect();
        assert_eq!(
            select_children(&oversized, &buckets.into(), &config, 8, Pressure::Pressured),
            vec![2]
        );
    }
}
