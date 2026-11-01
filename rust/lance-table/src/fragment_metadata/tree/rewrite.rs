// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One commit's copy-on-write rewrite. Drains buffers, splits and coalesces
//! nodes, shrinks the root, and validates sequences across the nodes it decodes.

use std::sync::Mutex;

use super::*;
use crate::fragment_metadata::node::{DecodedSequences, InteriorNode};
use crate::fragment_metadata::store::{NodeLocation, Written};

pub(super) struct RewriteNodes {
    /// The commit's frontier. It bounds the interiors this rewrite writes,
    /// while the store bounds every other interior by the root's frontier.
    next_action_sequence: u64,
    state: Mutex<State>,
}

struct State {
    sequences: DecodedSequences,
    /// Interiors of the source tree whose buffers this rewrite admitted.
    admitted: BTreeSet<NodeLocation>,
    /// Interiors this rewrite wrote. Their actions came from the root or an
    /// admitted interior, so a later drain, merge, or root shrink must not
    /// admit them again.
    written: BTreeSet<NodeLocation>,
}

impl RewriteNodes {
    pub(super) fn new(root_buffer: &[pb::FragmentTreeMutation], next_action_sequence: u64) -> Self {
        Self {
            next_action_sequence,
            state: Mutex::new(State {
                sequences: DecodedSequences::new(root_buffer),
                admitted: BTreeSet::new(),
                written: BTreeSet::new(),
            }),
        }
    }

    fn state(&self) -> Result<std::sync::MutexGuard<'_, State>> {
        self.state.lock().map_err(|_| {
            Error::internal("fragment metadata rewrite sequence validation mutex poisoned")
        })
    }

    pub(super) async fn read_interior(
        &self,
        store: &NodeStore,
        child: &pb::FragmentTreeChild,
        end: u64,
    ) -> Result<InteriorNode> {
        let location = store.child_location(child)?;
        if self.state()?.written.contains(&location) {
            let mut own = store.clone();
            own.root_frontier = self.next_action_sequence;
            return own.read_interior_in_range(child, end).await;
        }
        // Range validation still runs on every read, including a repeated
        // source reached through a different parent after a structural repair.
        let node = store.read_interior_in_range(child, end).await?;
        let mut state = self.state()?;
        if state.admitted.insert(location) {
            state.sequences.admit(&child.path, &node.buffer)?;
        }
        Ok(node)
    }

    pub(super) async fn write_interior(
        &self,
        store: &NodeStore,
        children: Vec<pb::FragmentTreeChild>,
        buffer: Vec<pb::FragmentTreeMutation>,
    ) -> Result<Written> {
        let written = store.write_interior(children, buffer).await?;
        let location = store.child_location(&written.child_ref)?;
        self.state()?.written.insert(location);
        Ok(written)
    }
}

impl FragmentTree {
    /// Flush the root buffer as far as it goes, then split, coalesce, and
    /// shrink the root. Every touched node is copy-on-write, so nothing here
    /// is visible until the root is published. A collapse leaves the root
    /// routing unusable, so the caller must restore it.
    pub(super) async fn rewrite_tree(&mut self, mode: CommitMode) -> Result<Rewrite<WriteAcc>> {
        let mut acc = WriteAcc::default();
        let rewrite = RewriteNodes::new(&self.buffer, self.next_action_sequence);

        let children = std::mem::take(&mut self.children);
        let buffer = std::mem::take(&mut self.buffer);
        let (children, buffer, work) = if mode == CommitMode::Bulk {
            let (children, work) = bulk::materialize(
                &self.store,
                &self.config,
                children,
                buffer,
                self.next_action_sequence.saturating_sub(1),
                self.next_fragment_id,
            )
            .await?;
            (children, Vec::new(), work)
        } else {
            match self
                .flush_internal(children, buffer, 0, 0..self.next_fragment_id, &rewrite)
                .await?
            {
                Rewrite::Done(flushed) => flushed,
                Rewrite::Collapsed { abandoned_bytes } => {
                    return Ok(Rewrite::Collapsed { abandoned_bytes });
                }
            }
        };
        acc.add(work);
        self.children = children;
        self.buffer = buffer;
        node::fence(&mut self.children, 0);

        if self.children.is_empty()
            && (node::internal_overflows(&self.children, &self.buffer, &self.config)
                || node::buffer_pressured(&self.buffer, &self.config))
        {
            self.seed_first_leaves(&mut acc).await?;
        }

        // Repair children that drains left underfull or single-child before
        // routing is split, as ingest does below the root, so no split piece
        // persists a single-child interior.
        if mode == CommitMode::Incremental {
            self.repair_root_children(&rewrite, &mut acc).await?;
        }

        let (children, buffer, routing) = Self::lift_root(
            &self.store,
            &self.config,
            std::mem::take(&mut self.children),
            std::mem::take(&mut self.buffer),
            Some(&rewrite),
        )
        .await?;
        self.children = children;
        self.buffer = buffer;
        acc.add(routing);

        if mode == CommitMode::Incremental {
            self.repair_root_children(&rewrite, &mut acc).await?;
        }

        self.maybe_shrink_root(&rewrite).await?;
        // The root keeps a lone interior only when adopting its level would
        // overflow. Readers reject that interior if it has one child, and no
        // sibling is left to pair it with, so the commit rebuilds instead.
        if node::is_single_child_chain(&self.children) {
            return Ok(Rewrite::Collapsed {
                abandoned_bytes: acc.io_bytes,
            });
        }
        node::fence(&mut self.children, 0);
        Ok(Rewrite::Done(acc))
    }

    /// A childless root, which is an empty table grown by appends, cannot flush
    /// anywhere. Materialize its buffer into the first leaves instead of
    /// splitting into a childless interior.
    async fn seed_first_leaves(&mut self, acc: &mut WriteAcc) -> Result<()> {
        let mut fragments = BTreeMap::new();
        let actions = std::mem::take(&mut self.buffer);
        self.store.apply_verified(&mut fragments, actions)?;
        let fragments: Vec<Fragment> = fragments.into_values().collect();
        let watermark = self.next_action_sequence.saturating_sub(1);
        for written in self
            .store
            .write_leaves(&fragments, None, watermark, &self.config)
            .await?
        {
            acc.io_bytes += written.io_bytes;
            self.children.push(written.child_ref);
        }
        // lift_root then stores these leaves inside interior nodes, and a
        // reader checks each node's first stored fence against the entry
        // that points at it. The first leaf must already own the root's
        // range, not start at the lowest live id.
        node::fence(&mut self.children, 0);
        Ok(())
    }

    async fn repair_root_children(
        &mut self,
        rewrite: &RewriteNodes,
        acc: &mut WriteAcc,
    ) -> Result<()> {
        let children = std::mem::take(&mut self.children);
        let (children, work) = self
            .merge_small_children(children, self.next_fragment_id, rewrite)
            .await?;
        acc.add(work);
        self.children = children;
        Ok(())
    }

    /// Build as many routing levels as needed for the root to fit. A build
    /// has no source nodes to track. A rewrite retains its sequence admission.
    pub(super) async fn lift_root(
        store: &NodeStore,
        config: &FragmentTreeConfig,
        mut children: Vec<pb::FragmentTreeChild>,
        mut buffer: Vec<pb::FragmentTreeMutation>,
        rewrite: Option<&RewriteNodes>,
    ) -> Result<FlushResult> {
        let mut acc = WriteAcc::default();
        while node::internal_overflows(&children, &buffer, config) {
            let previous_width = children.len();
            let pieces = node::split_internal(
                children,
                buffer,
                config.split_piece_bytes(),
                config.max_children_per_node,
            );
            if pieces.len() >= previous_width
                || pieces.iter().any(|(children, _)| children.len() < 2)
            {
                return Err(Error::invalid_input(format!(
                    "node budget {} cannot group two child references into fewer root nodes \
                     (children={previous_width}); raise max_node_bytes",
                    config.max_node_bytes
                )));
            }
            children = Vec::with_capacity(pieces.len());
            buffer = Vec::new();
            for (group, actions) in pieces {
                let written = match rewrite {
                    Some(rewrite) => rewrite.write_interior(store, group, actions).await?,
                    None => store.write_interior(group, actions).await?,
                };
                acc.io_bytes += written.io_bytes;
                children.push(written.child_ref);
            }
            acc.splits += 1;
        }
        Ok((children, buffer, acc))
    }

    /// Flush an internal node's buffer to its children while it is pressured.
    /// Only children whose pending batch amortizes a rewrite drain. Under
    /// buffer pressure the rest keep buffering. Under structural pressure
    /// the caller splits whatever routing still overflows. `depth` is this
    /// node's level below the root, 0 for the root itself, and `range` the
    /// key range its parent assigned. Returns the children, split if needed,
    /// and the residual buffer.
    fn flush_internal<'a>(
        &'a self,
        mut children: Vec<pb::FragmentTreeChild>,
        mut buffer: Vec<pb::FragmentTreeMutation>,
        depth: u32,
        range: Range<u64>,
        rewrite: &'a RewriteNodes,
    ) -> BoxFuture<'a, Result<Rewrite<FlushResult>>> {
        Box::pin(async move {
            let mut acc = WriteAcc::default();
            // Partitioned on the first pressured pass, then carried.
            let mut routed: Option<flush::Buckets> = None;
            loop {
                // A childless node has nowhere to flush; the caller turns the
                // buffer into leaves instead.
                if children.is_empty() {
                    break;
                }
                let buffer_bytes = match &routed {
                    Some(buckets) => buckets.total_bytes(),
                    None => node::internal_logical_bytes(&[], &buffer),
                };
                let structural = node::overflows_with(&children, buffer_bytes, &self.config);
                let retires = || match &routed {
                    Some(buckets) => buckets
                        .actions()
                        .iter()
                        .flatten()
                        .any(content::retires_fragment),
                    None => buffer.iter().any(content::retires_fragment),
                };
                let pressure = if structural || buffer_bytes >= self.config.flush_threshold_bytes {
                    flush::Pressure::Pressured
                } else if retires() {
                    flush::Pressure::Relaxed
                } else {
                    break;
                };
                let mut buckets = match routed.take() {
                    Some(buckets) => buckets,
                    None => {
                        let buckets =
                            node::partition_buffer_by_child(&children, std::mem::take(&mut buffer));
                        node::validate_routed(&children, &buckets, range.end)?;
                        flush::Buckets::from(buckets)
                    }
                };
                let pending = buckets.pending();
                let indices = flush::select_children(
                    &children,
                    &buckets,
                    &self.config,
                    self.store.object_store.io_parallelism(),
                    pressure,
                );
                if indices.is_empty() {
                    // Under buffer pressure no batch pays for its object yet,
                    // so keep buffering. Under structural pressure a byte
                    // overflow always leaves a bucket at or above the fair
                    // share, so only routing or fanout remains and the caller
                    // splits it.
                    debug_assert!(
                        !structural
                            || pending == 0
                            || children.len() as u32 > self.config.max_children_per_node
                    );
                    routed = Some(buckets);
                    break;
                }
                match self
                    .drain_children(
                        indices,
                        &mut children,
                        &mut buckets,
                        depth,
                        range.end,
                        rewrite,
                    )
                    .await?
                {
                    Rewrite::Done(drained) => acc.add(drained),
                    Rewrite::Collapsed { abandoned_bytes } => {
                        return Ok(Rewrite::Collapsed {
                            abandoned_bytes: acc.io_bytes + abandoned_bytes,
                        });
                    }
                }
                routed = Some(buckets);
            }
            if let Some(buckets) = routed {
                buffer = buckets.into_buffer();
            }
            self.fence_left_edge(&mut children, range.start, range.end, &mut acc, rewrite)
                .await?;
            Ok(Rewrite::Done((children, buffer, acc)))
        })
    }

    /// Drain the buckets of `indices` into their children at once, then replace
    /// each child with the references that now represent it. `end` is the
    /// exclusive end of the node holding `children`.
    async fn drain_children(
        &self,
        indices: Vec<usize>,
        children: &mut Vec<pb::FragmentTreeChild>,
        buckets: &mut flush::Buckets,
        depth: u32,
        end: u64,
        rewrite: &RewriteNodes,
    ) -> Result<Rewrite<WriteAcc>> {
        let concurrency = indices.len();
        let drains: Vec<_> = indices
            .into_iter()
            .map(|idx| {
                (
                    idx,
                    children[idx].clone(),
                    buckets.take(idx),
                    node::exclusive_end(children, idx, end),
                )
            })
            .collect();
        let mut results: Vec<_> = futures::stream::iter(drains)
            .map(|(idx, child, actions, end)| async move {
                self.ingest(child, actions, depth, end, rewrite)
                    .await
                    .map(|result| (idx, result))
            })
            .buffered(concurrency)
            .try_collect()
            .await?;
        let mut acc = WriteAcc::default();
        // Apply from the right so splits and removals keep earlier indices valid.
        results.sort_unstable_by_key(|(idx, _)| std::cmp::Reverse(*idx));
        let mut collapsed = false;
        for (idx, drained) in results {
            match drained {
                Rewrite::Done((new_refs, work)) => {
                    acc.add(work);
                    acc.flushes += 1;
                    acc.max_flush_depth = acc.max_flush_depth.max(depth);
                    buckets.replace(idx, new_refs.len());
                    children.splice(idx..idx + 1, new_refs);
                }
                Rewrite::Collapsed { abandoned_bytes } => {
                    acc.io_bytes += abandoned_bytes;
                    collapsed = true;
                }
            }
        }
        if collapsed {
            return Ok(Rewrite::Collapsed {
                abandoned_bytes: acc.io_bytes,
            });
        }
        Ok(Rewrite::Done(acc))
    }

    /// Fence `children` at `lower_bound`. A drain that empties leading
    /// children lowers the first survivor's fence. A leaf may start below its
    /// first key, but an interior node stores the fence on its own first
    /// child, so the left edge is rewritten down to the leaf.
    fn fence_left_edge<'a>(
        &'a self,
        children: &'a mut [pb::FragmentTreeChild],
        lower_bound: u64,
        end: u64,
        acc: &'a mut WriteAcc,
        rewrite: &'a RewriteNodes,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let Some(first) = children.first() else {
                return Ok(());
            };
            if first.height == 0 || first.min_key == lower_bound {
                node::fence(children, lower_bound);
                return Ok(());
            }
            let child_end = node::exclusive_end(children, 0, end);
            let InteriorNode {
                children: mut grandchildren,
                buffer,
            } = rewrite.read_interior(&self.store, first, child_end).await?;
            self.fence_left_edge(&mut grandchildren, lower_bound, child_end, acc, rewrite)
                .await?;
            let written = rewrite
                .write_interior(&self.store, grandchildren, buffer)
                .await?;
            acc.io_bytes += written.io_bytes;
            children[0] = written.child_ref;
            Ok(())
        })
    }

    /// Push `incoming` mutations into the subtree rooted at `child`, `depth`
    /// levels below the root and owning keys below `end`. A leaf applies them.
    /// An interior buffers them, recurses and splits on overflow. Returns the
    /// child references that now represent the subtree.
    fn ingest<'a>(
        &'a self,
        child: pb::FragmentTreeChild,
        incoming: Vec<pb::FragmentTreeMutation>,
        depth: u32,
        end: u64,
        rewrite: &'a RewriteNodes,
    ) -> BoxFuture<'a, Result<Rewrite<IngestResult>>> {
        Box::pin(async move {
            if child.height == 0 {
                self.ingest_leaf(&child, incoming, end)
                    .await
                    .map(Rewrite::Done)
            } else {
                self.ingest_interior(&child, incoming, depth, end, rewrite)
                    .await
            }
        })
    }

    async fn ingest_leaf(
        &self,
        child: &pb::FragmentTreeChild,
        incoming: Vec<pb::FragmentTreeMutation>,
        end: u64,
    ) -> Result<IngestResult> {
        let mut acc = WriteAcc::default();
        acc.materialized += incoming.len() as u64;
        let new_frags = content::replay_leaf(&self.store, child, end, incoming).await?;

        // A fully-emptied leaf is dropped from its parent. Keeping it would
        // create a phantom child with min_key=0 that corrupts the
        // sorted-by-min_key invariant `child_index_for` relies on.
        if new_frags.is_empty() {
            return Ok((vec![], acc));
        }
        // The flush that reaches a leaf drained every buffer on its
        // path for this range first, so the leaf now holds the
        // result of every applicable action up to the action_sequence ceiling.
        let watermark = self.next_action_sequence.saturating_sub(1);
        let written = self
            .store
            .write_leaves(&new_frags, None, watermark, &self.config)
            .await?;
        acc.splits += written.len().saturating_sub(1) as u64;
        let mut refs: Vec<_> = written
            .into_iter()
            .map(|leaf| {
                acc.io_bytes += leaf.io_bytes;
                leaf.child_ref
            })
            .collect();
        node::fence(&mut refs, child.min_key);
        Ok((refs, acc))
    }

    async fn ingest_interior(
        &self,
        child: &pb::FragmentTreeChild,
        incoming: Vec<pb::FragmentTreeMutation>,
        depth: u32,
        end: u64,
        rewrite: &RewriteNodes,
    ) -> Result<Rewrite<IngestResult>> {
        let mut acc = WriteAcc::default();
        let InteriorNode {
            children,
            mut buffer,
        } = rewrite.read_interior(&self.store, child, end).await?;
        buffer.extend(incoming);
        buffer =
            content::squash_buffer(buffer, node::combine_from(&children, self.contiguous_from));
        // This node is one level deeper than the parent that flushed to it.
        let (children, buffer, work) = match self
            .flush_internal(children, buffer, depth + 1, child.min_key..end, rewrite)
            .await?
        {
            Rewrite::Done(flushed) => flushed,
            Rewrite::Collapsed { abandoned_bytes } => {
                return Ok(Rewrite::Collapsed { abandoned_bytes });
            }
        };
        acc.add(work);
        let (children, work) = self.merge_small_children(children, end, rewrite).await?;
        acc.add(work);
        if node::is_single_child_chain(&children) {
            return Ok(Rewrite::Collapsed {
                abandoned_bytes: acc.io_bytes,
            });
        }

        // An interior whose children all vanished is dropped too, for the
        // same phantom min_key=0 hazard as an empty leaf.
        if children.is_empty() {
            return Ok(Rewrite::Done((vec![], acc)));
        }
        if node::internal_overflows(&children, &buffer, &self.config) {
            let mut refs = Vec::new();
            // A one-child piece is never published. The caller's
            // merge_small_children pairs it with a sibling, or the
            // commit rebuilds when no sibling is left.
            for (piece_children, piece_buffer) in node::split_internal(
                children,
                buffer,
                self.config.split_piece_bytes(),
                self.config.max_children_per_node,
            ) {
                let written = rewrite
                    .write_interior(&self.store, piece_children, piece_buffer)
                    .await?;
                acc.io_bytes += written.io_bytes;
                refs.push(written.child_ref);
            }
            acc.splits += 1;
            node::fence(&mut refs, child.min_key);
            Ok(Rewrite::Done((refs, acc)))
        } else {
            let written = rewrite
                .write_interior(&self.store, children, buffer)
                .await?;
            acc.io_bytes += written.io_bytes;
            let mut refs = vec![written.child_ref];
            node::fence(&mut refs, child.min_key);
            Ok(Rewrite::Done((refs, acc)))
        }
    }

    /// Coalesce underfull runs of children, then rebalance any single-child
    /// interior left over with a neighbor under the hard node budget.
    async fn merge_small_children(
        &self,
        children: Vec<pb::FragmentTreeChild>,
        end: u64,
        rewrite: &RewriteNodes,
    ) -> Result<(Vec<pb::FragmentTreeChild>, WriteAcc)> {
        let (children, mut acc) = self.coalesce_underfull_runs(children, end, rewrite).await?;
        let (children, repaired) = self.repair_singletons(children, end, rewrite).await?;
        acc.add(repaired);
        Ok((children, acc))
    }

    /// Coalesce runs of adjacent children when one underflows by
    /// `node::is_underflow`. A leaf run stays within `leaf_coalesce_ceiling`. An
    /// interior run stays within `coalesce_ceiling` and `max_children_per_node`.
    /// Merged leaves join their fragments. Merged interiors join their children
    /// and buffers.
    async fn coalesce_underfull_runs(
        &self,
        children: Vec<pb::FragmentTreeChild>,
        end: u64,
        rewrite: &RewriteNodes,
    ) -> Result<(Vec<pb::FragmentTreeChild>, WriteAcc)> {
        let mut acc = WriteAcc::default();
        let mut out: Vec<pb::FragmentTreeChild> = Vec::with_capacity(children.len());
        let mut i = 0;
        while i < children.len() {
            if !node::is_underflow(&children[i], &self.config) {
                out.push(children[i].clone());
                i += 1;
                continue;
            }
            // Grow a coalesce group with adjacent siblings, bounded by node kind.
            let is_leaf = children[i].height == 0;
            let mut group = vec![children[i].clone()];
            let mut bytes = children[i].object_size;
            let mut fan = children[i].num_children;
            let mut j = i + 1;
            while j < children.len() {
                let c = &children[j];
                // A sibling's reference is checked only when the sibling is
                // read, so a misstated size or fanout must not wrap the sums.
                let (Some(next_bytes), Some(next_fan)) = (
                    bytes.checked_add(c.object_size),
                    fan.checked_add(c.num_children),
                ) else {
                    break;
                };
                let fits = if is_leaf {
                    next_bytes <= self.config.leaf_coalesce_ceiling()
                } else {
                    next_fan <= self.config.max_children_per_node
                        && next_bytes <= self.config.coalesce_ceiling()
                };
                if !fits {
                    break;
                }
                bytes = next_bytes;
                fan = next_fan;
                group.push(c.clone());
                j += 1;
            }
            if group.len() == 1 {
                out.extend(group);
            } else {
                let lower_bound = group[0].min_key;
                let (mut merged, work) = self
                    .coalesce(
                        group,
                        children.get(j).map_or(end, |child| child.min_key),
                        rewrite,
                    )
                    .await?;
                node::fence(&mut merged, lower_bound);
                acc.add(work);
                acc.merges += 1;
                out.extend(merged);
            }
            i = j;
        }
        Ok((out, acc))
    }

    /// A singleton at either edge needs its neighbor even when the normal
    /// soft coalesce ceiling would leave it alone. Rebalance the combined
    /// range under the hard node budget, keeping mutations with their fences.
    async fn repair_singletons(
        &self,
        children: Vec<pb::FragmentTreeChild>,
        end: u64,
        rewrite: &RewriteNodes,
    ) -> Result<(Vec<pb::FragmentTreeChild>, WriteAcc)> {
        let mut acc = WriteAcc::default();
        let mut repaired: Vec<pb::FragmentTreeChild> = Vec::with_capacity(children.len());
        let mut pending: VecDeque<_> = children.into();
        while let Some(child) = pending.pop_front() {
            if child.height == 0 || child.num_children > 1 {
                repaired.push(child);
                continue;
            }
            let pair = if let Some(right) = pending.pop_front() {
                vec![child, right]
            } else if let Some(left) = repaired.pop() {
                vec![left, child]
            } else {
                // Its parent must provide a sibling or shrink this root.
                repaired.push(child);
                break;
            };
            let lower_bound = pair[0].min_key;
            let pair_end = pending.front().map_or(end, |child| child.min_key);
            let mut children = Vec::new();
            let mut buffer = Vec::new();
            for (sibling, sibling_end) in node::with_exclusive_ends(pair, pair_end) {
                let node = rewrite
                    .read_interior(&self.store, &sibling, sibling_end)
                    .await?;
                children.extend(node.children);
                buffer.extend(node.buffer);
            }
            // The repair must not flush. The caller still holds this range's
            // residual actions in its own buffer, so any leaf rewritten here
            // at the frontier watermark would claim changes it never applied.
            // Merge and, if needed, split; the parent's next pressured flush
            // drains the range with every buffered action on the path.
            let (mut children, coalesced) =
                Box::pin(self.merge_small_children(children, pair_end, rewrite)).await?;
            acc.add(coalesced);
            node::fence(&mut children, lower_bound);
            if children.is_empty() && buffer.is_empty() {
                continue;
            }
            if node::internal_overflows(&children, &buffer, &self.config) {
                let pieces = node::split_internal(
                    children.clone(),
                    buffer.clone(),
                    self.config.max_node_bytes - 1,
                    self.config.max_children_per_node,
                );
                if pieces.iter().all(|(piece, _)| piece.len() >= 2) {
                    for (children, buffer) in pieces {
                        let written = rewrite
                            .write_interior(&self.store, children, buffer)
                            .await?;
                        acc.io_bytes += written.io_bytes;
                        repaired.push(written.child_ref);
                    }
                    continue;
                }
                // Two singletons whose buffers overflow together stay one
                // oversized node until the parent drains it under pressure.
            }
            let written = rewrite
                .write_interior(&self.store, children, buffer)
                .await?;
            acc.io_bytes += written.io_bytes;
            acc.merges += 1;
            // A pair that coalesced to one child is a singleton again; while
            // siblings remain it pairs once more, shrinking the list each time.
            if written.child_ref.num_children == 1 && !(pending.is_empty() && repaired.is_empty()) {
                pending.push_front(written.child_ref);
            } else {
                repaired.push(written.child_ref);
            }
        }
        Ok((repaired, acc))
    }

    /// Combine an adjacent group of same-height children into one node.
    async fn coalesce(
        &self,
        group: Vec<pb::FragmentTreeChild>,
        end: u64,
        rewrite: &RewriteNodes,
    ) -> Result<(Vec<pb::FragmentTreeChild>, WriteAcc)> {
        let mut acc = WriteAcc::default();
        if group[0].height == 0 {
            let mut fragments: Vec<Fragment> = Vec::new();
            for (index, child) in group.iter().enumerate() {
                let child_end = node::exclusive_end(&group, index, end);
                fragments.extend(
                    self.store
                        .read_leaf_shared(child, child_end)
                        .await?
                        .into_vec(),
                );
            }
            fragments.sort_by_key(|f| f.id);
            let watermark = group
                .iter()
                .map(|child| child.materialized_through_action_sequence)
                .min()
                .unwrap_or(0);
            let written = self
                .store
                .write_leaves(&fragments, None, watermark, &self.config)
                .await?;
            let refs = written
                .into_iter()
                .map(|leaf| {
                    acc.io_bytes += leaf.io_bytes;
                    leaf.child_ref
                })
                .collect();
            Ok((refs, acc))
        } else {
            let mut children: Vec<pb::FragmentTreeChild> = Vec::new();
            let mut buffer: Vec<pb::FragmentTreeMutation> = Vec::new();
            for (index, child) in group.iter().enumerate() {
                let child_end = node::exclusive_end(&group, index, end);
                let node = rewrite.read_interior(&self.store, child, child_end).await?;
                children.extend(node.children);
                buffer.extend(node.buffer);
            }
            // Joining parents exposes siblings that previously belonged to
            // separate ranges; repair their occupancy before persisting them.
            let (children, repaired) =
                Box::pin(self.merge_small_children(children, end, rewrite)).await?;
            acc.add(repaired);
            let written = rewrite
                .write_interior(&self.store, children, buffer)
                .await?;
            acc.io_bytes += written.io_bytes;
            Ok((vec![written.child_ref], acc))
        }
    }

    /// Remove a routing level when its contents fit in the root. Multiple
    /// children use the coalesce ceiling so growth and shrinkage have hysteresis.
    pub(super) async fn maybe_shrink_root(&mut self, rewrite: &RewriteNodes) -> Result<()> {
        while !self.children.is_empty() && self.children[0].height > 0 {
            if self.children.len() > 1 {
                // References carry each child's validated fanout, so a joined
                // level that would overflow it is ruled out without a read.
                let fanout: u64 = self
                    .children
                    .iter()
                    .map(|child| u64::from(child.num_children))
                    .sum();
                if fanout > u64::from(self.config.max_children_per_node) {
                    break;
                }
                // Interior byte_size is the exact encoded children-plus-buffer
                // body. Rule out a collapse without fetching every child.
                let bytes = self.children.iter().try_fold(
                    node::internal_logical_bytes(&[], &self.buffer),
                    |bytes, child| bytes.checked_add(child.object_size),
                );
                if bytes.is_none_or(|bytes| bytes > self.config.coalesce_ceiling()) {
                    break;
                }
            }
            let mut children = Vec::new();
            let mut buffer = self.buffer.clone();
            for (index, child) in self.children.iter().enumerate() {
                let end = node::exclusive_end(&self.children, index, self.next_fragment_id);
                let node = rewrite.read_interior(&self.store, child, end).await?;
                children.extend(node.children);
                buffer.extend(node.buffer);
            }
            if node::internal_overflows(&children, &buffer, &self.config) {
                break;
            }
            self.children = children;
            self.buffer = buffer;
            self.buffer_index.take();
            node::fence(&mut self.children, 0);
        }
        Ok(())
    }
}
