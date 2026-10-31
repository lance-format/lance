// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Immutable routing with buffered fragment changes.
//!
//! Pressure drains the children whose pending bytes amortize a rewrite.
//! Splits and coalesces transfer messages with their ranges. The Version
//! Manifest publishes the prepared root after every dependency has been written.

use std::borrow::Cow;
#[cfg(any(test, feature = "test-util"))]
use std::collections::HashSet;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::num::NonZeroUsize;
use std::ops::Range;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};

use futures::future::{BoxFuture, RemoteHandle};
use futures::stream::{BoxStream, FuturesOrdered};
use futures::{FutureExt, Stream, StreamExt, TryStreamExt};

use crate::format::Fragment;
use crate::format::pb;
use crate::fragment_metadata::TreeContext;
use crate::fragment_metadata::commit::{self, TouchedFragments, ValidatedCommit};
use crate::fragment_metadata::content::{self, apply_aggregate_delta, sum_aggregate_deltas};
use crate::fragment_metadata::node::{self, FragmentTreeConfig, InternalNode};
use crate::fragment_metadata::store::{
    LeafLocation, NodeStore, PendingLeafRead, SingleFlightLeafCache, StoredFragments,
};
use lance_core::{Error, Result};
use lance_io::object_store::{ObjectStore, WrappingObjectStore};
use object_store::path::Path;
use prost::Message;
use roaring::RoaringBitmap;

mod audit;
mod bulk;
mod flush;
mod publication;
mod rewrite;
mod summary;
mod transaction;
pub(super) use flush::MAX_CONCURRENT_LEAF_DRAINS;
pub use publication::SnapshotPolicy;
pub use summary::LeafSummaries;
pub use transaction::{CommitMode, InventoryRetention, TreeCommit};

/// Accumulated write work over one commit (for stats / benchmark accounting).
#[derive(Debug, Default, Clone, Copy)]
struct WriteAcc {
    io_bytes: u64,
    flushes: u64,
    splits: u64,
    merges: u64,
    /// Actions applied at leaves this commit.
    materialized: u64,
    /// Actions removed by same-key squashing this commit.
    squashed: u64,
    /// Deepest tree level at which a flush occurred this commit (0 = root only).
    /// Values above 0 mean an interior ε-buffer drained.
    max_flush_depth: u32,
    /// Subtrees that shrank to a chain of single-child interiors, which the
    /// format forbids and no sibling can repair. The commit rebuilds in bulk.
    collapses: u64,
}

impl WriteAcc {
    fn add(&mut self, o: Self) {
        self.io_bytes += o.io_bytes;
        self.flushes += o.flushes;
        self.splits += o.splits;
        self.merges += o.merges;
        self.materialized += o.materialized;
        self.squashed += o.squashed;
        self.max_flush_depth = self.max_flush_depth.max(o.max_flush_depth);
        self.collapses += o.collapses;
    }
}

/// Result of flushing an internal node: (possibly split) children, residual
/// buffer, and accumulated write work.
type FlushResult = (
    Vec<pb::FragmentTreeChild>,
    Vec<pb::FragmentTreeMutation>,
    WriteAcc,
);
/// Result of ingesting into a subtree: the child ref(s) that now represent it
/// (>1 if it split), and accumulated write work.
type IngestResult = (Vec<pb::FragmentTreeChild>, WriteAcc);
type BufferedChild = (pb::FragmentTreeChild, Vec<pb::FragmentTreeMutation>, u64);

/// A leaf full materialization has walked to and not yet decoded.
struct StartedLeaf {
    child: pb::FragmentTreeChild,
    end: u64,
    actions: Vec<pb::FragmentTreeMutation>,
    read: PendingLeafRead,
    /// Its object GET, running on its own task. Dropping it cancels the GET.
    fetch: Option<RemoteHandle<()>>,
}

impl StartedLeaf {
    async fn start(store: &NodeStore, (child, actions, end): BufferedChild) -> Result<Self> {
        let read = store.start_leaf_read(&child).await?;
        let fetch = read.object_to_fetch().map(|object| {
            let (fetch, fetched) = object.clone().fetch().remote_handle();
            tokio::spawn(fetch);
            fetched
        });
        Ok(Self {
            child,
            end,
            actions,
            read,
            fetch,
        })
    }
}

/// Bytes/structure written while bootstrapping.
#[derive(Debug, Default, Clone, Copy)]
pub struct BootstrapStats {
    pub io_write_bytes: u64,
    pub num_leaves: u64,
    /// Root-to-leaf edge count; see [`FragmentTree::height`].
    pub height: u32,
}

/// Result of one commit.
#[derive(Debug, Default, Clone, Copy)]
pub struct CommitStats {
    /// Compacted root, internal-node, and leaf bytes written by this commit.
    pub tree_write_bytes: u64,
    /// Inline or external roots prepared by this commit.
    pub checkpoints: u64,
    pub flushes: u64,
    pub splits: u64,
    pub merges: u64,
    /// Root-to-leaf edge count; see [`FragmentTree::height`].
    pub height: u32,
    /// Deepest level flushed this commit (0 = root buffer only; ≥1 = cascaded
    /// into internal nodes — the deep-flush regime).
    pub max_flush_depth: u32,
    /// Actions this commit staged into the root buffer.
    pub messages_in: u64,
    /// Actions that reached a leaf this commit (any commit's actions).
    pub messages_materialized: u64,
    /// Actions removed by same-key squashing this commit.
    pub messages_squashed: u64,
    /// Root buffer occupancy after the commit.
    pub root_buffer_len: u64,
}

/// Tree size and occupancy statistics.
#[cfg(any(test, feature = "test-util"))]
#[derive(Debug, Default, Clone)]
pub struct ShapeReport {
    /// Root-to-leaf edge count; see [`FragmentTree::height`].
    pub height: u32,
    pub root_bytes: u64,
    pub root_buffer_len: u64,
    pub root_buffer_bytes: u64,
    pub root_fanout: u32,
    /// Encoded leaf object sizes from child references.
    pub leaf_bytes: Vec<u64>,
    /// Actual encoded Lance bytes used for the leaf-size policy.
    pub leaf_object_bytes: Vec<u64>,
    pub leaf_keys: Vec<u64>,
    pub node_bytes: Vec<u64>,
    pub node_fanouts: Vec<u32>,
    pub node_buffer_lens: Vec<u64>,
    pub node_buffer_bytes: Vec<u64>,
}

/// State of an in-order fragment walk: subtrees still to visit, each with
/// the buffered actions routed to it and the exclusive end of its range,
/// fragments ready to yield, and the sequences of every buffer it decoded.
#[derive(Default)]
struct FragmentWalk {
    lower_bound: u64,
    stack: Vec<(pb::FragmentTreeChild, Vec<pb::FragmentTreeMutation>, u64)>,
    ready: VecDeque<Fragment>,
    pending_error: Option<Error>,
    sequences: node::DecodedSequences,
}

impl FragmentWalk {
    /// Queue the children that may hold keys at or above the walk's lower
    /// bound, in a parent whose range ends at `parent_end`.
    fn push_children(
        &mut self,
        children: Vec<pb::FragmentTreeChild>,
        buckets: Vec<Vec<pb::FragmentTreeMutation>>,
        parent_end: u64,
    ) {
        let first = node::child_index_for(&children, self.lower_bound);
        for ((child, end), actions) in node::with_exclusive_ends(children, parent_end)
            .zip(buckets)
            .skip(first)
            .rev()
        {
            self.stack.push((child, actions, end));
        }
    }
}

/// Bounds the node reads a set read keeps in flight and the window a ramped
/// scan grows to, and with them the decoded leaves either holds at once.
const MAX_READS_IN_FLIGHT: usize = 8;

/// How many leaf reads a fragment scan keeps in flight.
#[derive(Debug, Clone, Copy)]
struct LeafReadahead {
    depth: usize,
    max: usize,
}

impl LeafReadahead {
    fn fixed(depth: NonZeroUsize) -> Self {
        Self {
            depth: depth.get(),
            max: depth.get(),
        }
    }

    fn ramp(max: usize) -> Self {
        Self { depth: 1, max }
    }

    fn widen(&mut self) {
        self.depth = self.depth.saturating_mul(2).min(self.max);
    }
}

/// A spawned leaf read. Dropping it aborts the task, so a stream dropped
/// early neither decodes nor admits leaves nobody will consume. A GET the
/// task already handed to the scan scheduler still completes.
struct SpawnedRead<T>(tokio::task::JoinHandle<Result<T>>);

impl<T> Future for SpawnedRead<T> {
    type Output = Result<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.0).poll(cx).map(|joined| match joined {
            Ok(fragments) => fragments,
            Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
            Err(error) => Err(Error::internal(format!(
                "fragment tree leaf read task did not complete: {error}"
            ))),
        })
    }
}

impl<T> Drop for SpawnedRead<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

enum Routing {
    Walking,
    /// Held back until every leaf admitted before it has been yielded, so
    /// errors surface in walk order.
    Failed(Error),
    Done,
}

/// Leaf reads in walk order behind a bounded window. Routing, including
/// interior reads, runs on the polling task. Leaves are admitted only when
/// the consumer polls, so the window never reads ahead of a leaf the
/// consumer is still draining.
struct LeafWindow {
    store: NodeStore,
    leaves: BoxStream<'static, Result<BufferedChild>>,
    routing: Routing,
    reads: FuturesOrdered<SpawnedRead<Vec<Fragment>>>,
    readahead: LeafReadahead,
}

impl LeafWindow {
    fn spawn_read(&self, (child, actions, end): BufferedChild) -> SpawnedRead<Vec<Fragment>> {
        let store = self.store.clone();
        // Decoding is CPU-bound. A task per leaf lets the leaves in flight
        // decode in parallel instead of on the task polling this stream.
        SpawnedRead(tokio::spawn(async move {
            let fragments = store.read_leaf_in_range(&child, end).await?;
            store.replay(fragments, actions)
        }))
    }
}

impl Stream for LeafWindow {
    type Item = Result<Vec<Fragment>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        while matches!(this.routing, Routing::Walking) && this.reads.len() < this.readahead.depth {
            match this.leaves.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(leaf))) => {
                    let read = this.spawn_read(leaf);
                    this.reads.push_back(read);
                }
                Poll::Ready(Some(Err(error))) => this.routing = Routing::Failed(error),
                Poll::Ready(None) => this.routing = Routing::Done,
                Poll::Pending => break,
            }
        }
        match this.reads.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(fragments))) => {
                this.readahead.widen();
                Poll::Ready(Some(Ok(fragments)))
            }
            Poll::Ready(Some(Err(error))) => {
                this.routing = Routing::Done;
                this.reads = FuturesOrdered::new();
                Poll::Ready(Some(Err(error)))
            }
            // The admission loop polled the walk, which will wake this task.
            Poll::Ready(None) if matches!(this.routing, Routing::Walking) => Poll::Pending,
            Poll::Ready(None) => match std::mem::replace(&mut this.routing, Routing::Done) {
                Routing::Failed(error) => Poll::Ready(Some(Err(error))),
                Routing::Walking | Routing::Done => Poll::Ready(None),
            },
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Where the table's `offset`-th visible row lives.
#[derive(Debug, Clone, PartialEq)]
pub enum OffsetResolution {
    /// The fragment holding the row and the visible-row offset within it.
    Found(Box<Fragment>, u64),
    /// The offset is at or past the table's visible row count.
    BeyondEnd,
}

/// One leaf's records, in id order, as this tree's version sees them.
#[derive(Debug, PartialEq)]
pub enum LeafFragments {
    /// No buffered action routes to the leaf, so the records are exactly the
    /// checked object at `location`. Their ids and file paths are the same in
    /// every version naming it. Inherited base ids follow the naming reference,
    /// as [`LeafLocation`] explains.
    Stored {
        location: LeafLocation,
        fragments: StoredFragments,
    },
    /// Records with this version's buffered actions applied, which no stored
    /// object holds. A root without children yields its whole table this way.
    Replayed(Vec<Fragment>),
}

/// One leaf of a [`FragmentTree::leaf_stream`] walk. `stored_at` is the
/// location of a leaf no buffered action routes to.
async fn read_leaf_fragments(
    store: NodeStore,
    leaves: SingleFlightLeafCache,
    (child, actions, end): BufferedChild,
    stored_at: Option<LeafLocation>,
) -> Result<LeafFragments> {
    let read = store.read_leaf_single_flight(&leaves, &child, end).await?;
    if let Some(location) = stored_at {
        return Ok(LeafFragments::Stored {
            location,
            fragments: StoredFragments(read),
        });
    }
    Ok(LeafFragments::Replayed(
        store.replay(read.into_vec(), actions)?,
    ))
}

/// An immutable fragment metadata snapshot. Holds the root, buffered changes
/// and aggregates. Interior and leaf nodes are read on demand. Begin an
/// isolated [`TreeCommit`] to prepare changes without modifying this snapshot.
#[derive(Clone)]
pub struct FragmentTree {
    store: NodeStore,
    config: FragmentTreeConfig,
    version: u64,
    children: Vec<pb::FragmentTreeChild>,
    buffer: Vec<pb::FragmentTreeMutation>,
    buffer_index: OnceLock<node::BufferIndex>,
    next_action_sequence: u64,
    /// The opened root's next sequence. The suffix and this writer's mutations
    /// start here. Whole-bucket drains keep their per-fragment history contiguous
    /// within each buffer. Older history may interleave across buffers.
    contiguous_from: u64,
    total_fragments: u64,
    total_rows: u64,
    /// The next fragment id an append allocates; see `next_fragment_id`.
    /// Reads end the root's last child here, because no stored record or
    /// buffered action may name an id that is not allocated yet.
    next_fragment_id: u64,
    visible_rows: u64,
    /// Exact descriptor this snapshot opened or prepared.
    descriptor: Arc<pb::FragmentTree>,
    /// Complete records owned by this exact snapshot, when explicitly retained.
    resident: Option<Arc<Vec<Fragment>>>,
}

impl FragmentTree {
    /// Bind an immutable snapshot to another handle for the same object namespace.
    pub fn with_object_store(mut self, store: Arc<ObjectStore>) -> Self {
        self.store.rebind(store, &[]);
        self
    }

    /// Bind the already-wrapped primary store and apply the same wrappers to
    /// inherited stores. Aliased bases share one wrapped store. Checked caches
    /// remain shared because these wrappers preserve the object namespace.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance_io::object_store::{ObjectStore, WrappingObjectStore};
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # fn wrapped(tree: FragmentTree, primary: Arc<ObjectStore>,
    /// #            wrappers: &[Arc<dyn WrappingObjectStore>]) -> FragmentTree {
    /// tree.with_object_store_wrappers(primary, wrappers)
    /// # }
    /// ```
    pub fn with_object_store_wrappers(
        mut self,
        primary: Arc<ObjectStore>,
        wrappers: &[Arc<dyn WrappingObjectStore>],
    ) -> Self {
        self.store.rebind(primary, wrappers);
        self
    }

    /// Borrow the complete inventory, if this snapshot explicitly retained it.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # fn example(tree: &FragmentTree) {
    /// if let Some(fragments) = tree.resident_fragments() {
    ///     assert_eq!(fragments.len() as u64, tree.count_fragments());
    /// }
    /// # }
    /// ```
    pub fn resident_fragments(&self) -> Option<&Arc<Vec<Fragment>>> {
        self.resident.as_ref()
    }

    /// The descriptor a Version Manifest publishes for this snapshot.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # fn example(tree: &FragmentTree) {
    /// assert!(tree.descriptor().root.is_some());
    /// # }
    /// ```
    pub fn descriptor(&self) -> &Arc<pb::FragmentTree> {
        &self.descriptor
    }

    /// Retain this snapshot's complete inventory for eager APIs. Ordinary walks
    /// and materialization do not change residency. The load keeps up to the
    /// object store's I/O parallelism of leaf reads in flight.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # async fn example(tree: FragmentTree) -> lance_core::Result<()> {
    /// let eager = tree.hydrate().await?;
    /// assert!(eager.resident_fragments().is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn hydrate(mut self) -> Result<Self> {
        if self.resident.is_none() {
            // The store clamps its I/O parallelism to at least one, so the
            // fallback only satisfies the type.
            let readahead = NonZeroUsize::new(self.store.object_store.io_parallelism())
                .unwrap_or(NonZeroUsize::MIN);
            let fragments = Arc::new(self.clone())
                .scan_with_readahead(0, readahead)
                .try_collect()
                .await?;
            self.resident = Some(Arc::new(fragments));
        }
        Ok(self)
    }

    /// Bind a successfully published snapshot to its manifest version and storage.
    /// Call only after the Version Manifest has confirmed this exact descriptor.
    /// Writer budgets come from that manifest, including any config update.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::{FragmentTree, FragmentTreeConfig, TreeContext};
    /// # fn confirmed(tree: FragmentTree, version: u64, config: FragmentTreeConfig,
    /// #              context: TreeContext) -> lance_core::Result<FragmentTree> {
    /// tree.into_published(version, config, context)
    /// # }
    /// ```
    pub fn into_published(
        mut self,
        version: u64,
        config: FragmentTreeConfig,
        context: TreeContext,
    ) -> Result<Self> {
        config.validate()?;
        self.version = version;
        self.config = config;
        self.store = context.into_store(&self.config, self.store.next_action_sequence);
        Ok(self)
    }

    /// Raise the next fragment id an append allocates to `next_fragment_id`.
    /// A rebuild of an existing inventory keeps every id its manifest already
    /// allocated, since index coverage may still name a removed fragment. A
    /// lower value changes nothing.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # fn example(rebuilt: FragmentTree) {
    /// let rebuilt = rebuilt.with_allocation_frontier(10);
    /// assert!(rebuilt.next_fragment_id() >= 10);
    /// # }
    /// ```
    pub fn with_allocation_frontier(mut self, next_fragment_id: u64) -> Self {
        self.next_fragment_id = self.next_fragment_id.max(next_fragment_id);
        self
    }

    async fn build(
        store: NodeStore,
        config: FragmentTreeConfig,
        fragments: &[Fragment],
        version: u64,
        policy: SnapshotPolicy,
    ) -> Result<(Self, BootstrapStats)> {
        config.validate()?;
        let mut io = 0u64;
        let mut layer: Vec<pb::FragmentTreeChild> = Vec::new();
        let (next_fragment_id, total_rows, visible_rows) =
            commit::validate_bootstrap(fragments, store.base())?;
        let num_fragments = fragments.len() as u64;
        for w in store.write_initial_leaves(fragments, &config).await? {
            io += w.io_bytes;
            layer.push(w.child_ref);
        }
        let num_leaves = layer.len() as u64;
        node::fence(&mut layer, 0);

        let (layer, _, routing) = Self::lift_root(&store, &config, layer, Vec::new(), None).await?;
        io += routing.io_bytes;
        let height = layer.iter().map(|c| c.height).max().unwrap_or(0) + 1;
        let root = pb::FragmentTreeRoot {
            children: layer.clone(),
            buffer: Vec::new(),
            next_action_sequence: 1,
        };
        let (descriptor, root_bytes) =
            publication::checkpoint(&store, &config, root, policy).await?;

        let tree = Self {
            store,
            config,
            version,
            children: layer,
            buffer: Vec::new(),
            buffer_index: OnceLock::new(),
            next_action_sequence: 1,
            contiguous_from: 1,
            total_fragments: num_fragments,
            total_rows,

            next_fragment_id,
            visible_rows,
            descriptor: Arc::new(descriptor),
            resident: None,
        };
        Ok((
            tree,
            BootstrapStats {
                io_write_bytes: io + root_bytes,
                num_leaves,
                height,
            },
        ))
    }

    /// Root-to-leaf edge count: one for a root over leaves, two with an
    /// intervening routing level. An empty tree also reports one.
    pub fn height(&self) -> u32 {
        self.children.iter().map(|c| c.height).max().unwrap_or(0) + 1
    }

    /// Authoritative manifest version this snapshot was opened from. Preparing
    /// changes does not assign a publication version. The manifest owns that.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Number of logical fragments, including buffered adds and removes.
    ///
    /// This reads root metadata only and performs no object-store IO.
    pub fn count_fragments(&self) -> u64 {
        self.total_fragments
    }

    /// Sum of physical rows across logical fragments.
    ///
    /// Admitted fragments have known counts. This reads root metadata only
    /// and performs no object-store IO.
    pub fn count_rows(&self) -> u64 {
        self.total_rows
    }

    /// Visible rows (physical minus deleted) across the table. Exact by the
    /// writer invariant on known counts. Root metadata only, no IO.
    pub fn count_visible_rows(&self) -> u64 {
        self.visible_rows
    }

    /// The next fragment id an append may allocate: zero on an empty table,
    /// otherwise one above the largest id ever assigned, buffered adds
    /// included. This is Lance's manifest allocation rule.
    pub fn next_fragment_id(&self) -> u64 {
        self.next_fragment_id
    }

    /// Number of actions currently buffered directly in the in-memory root.
    #[cfg(any(test, feature = "test-util"))]
    pub fn root_buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether `bytes` more of buffered actions keep the root under its
    /// semantic budget, so a commit of that size can stay buffered instead
    /// of draining in the same commit.
    fn can_buffer(&self, bytes: u64) -> bool {
        node::buffer_bytes(&self.buffer) + bytes < self.config.semantic_buffer_bytes
    }

    /// Number of direct child references currently held by the in-memory root.
    #[cfg(any(test, feature = "test-util"))]
    pub fn root_child_count(&self) -> usize {
        self.children.len()
    }

    /// Encoded leaf sizes when the root points directly to leaves. Returns an
    /// empty list when there is an intervening routing level.
    #[cfg(any(test, feature = "test-util"))]
    pub fn leaf_object_sizes(&self) -> Vec<u64> {
        self.children
            .iter()
            .filter(|child| child.height == 0)
            .map(|child| child.object_size)
            .collect()
    }

    fn buffer_index(&self) -> &node::BufferIndex {
        self.buffer_index
            .get_or_init(|| node::BufferIndex::new(&self.buffer))
    }

    /// Resolve one fragment by loading only the root-to-leaf path: at most
    /// `height` reads on a compacted root.
    ///
    /// Routing follows sibling lower-bound fences, and buffered actions from
    /// every node on the path are applied with the
    /// same action_sequence ordering as [`Self::materialize`].
    pub async fn resolve_fragment(&self, frag_id: u64) -> Result<Option<Fragment>> {
        if let Some(fragments) = &self.resident {
            return Ok(fragments
                .binary_search_by_key(&frag_id, |fragment| fragment.id)
                .ok()
                .map(|index| fragments[index].clone()));
        }
        let index = self.buffer_index();
        let mut sequences = None;
        let mut actions: Vec<pb::FragmentTreeMutation> = index
            .for_fragment(frag_id)
            .map(|offset| self.buffer[offset].clone())
            .collect();
        let mut children = Cow::Borrowed(self.children.as_slice());
        let mut fragment = None;
        let mut end = self.next_fragment_id;

        while !children.is_empty() {
            let index = node::child_index_for(&children, frag_id);
            end = node::exclusive_end(&children, index, end);
            let child = children[index].clone();
            if child.height == 0 {
                node::validate_routed(std::slice::from_ref(&child), &[actions.clone()], end)?;
                fragment = self.store.read_leaf_fragment(&child, frag_id, end).await?;
                break;
            }

            let internal = self.store.read_internal_shared(&child, end).await?;
            // The root is already validated; cross-buffer checks begin with the
            // first nonempty interior buffer.
            if !internal.node.buffer.is_empty() {
                sequences
                    .get_or_insert_with(|| self.buffer_index().sequences(&self.buffer))
                    .admit(&child.path, &internal.node.buffer)?;
            }
            actions.extend(
                internal
                    .node
                    .buffer
                    .iter()
                    .filter(|tagged| content::action_key(tagged) == frag_id)
                    .cloned(),
            );
            children = Cow::Owned(internal.node.children.clone());
        }

        let mut fragments = BTreeMap::new();
        if let Some(fragment) = fragment {
            fragments.insert(fragment.id, fragment);
        }
        self.store.apply_verified(&mut fragments, actions)?;
        Ok(fragments.remove(&frag_id))
    }

    /// Resolve the selected fragment ids in ascending order, omitting ids
    /// the table does not hold. Only the subtrees whose routing range
    /// intersects the set are read, each at most once. Routing prunes by
    /// fences, since buffered inserts may extend a leaf's stored key range.
    /// Up to eight reads of sibling interiors, and then of covering leaves,
    /// run at once.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # use roaring::RoaringBitmap;
    /// # async fn example(tree: &FragmentTree) -> lance_core::Result<()> {
    /// let ids = RoaringBitmap::from_iter([2, 7, 90]);
    /// for fragment in tree.resolve_fragments(&ids).await? {
    ///     assert!(ids.contains(fragment.id as u32));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn resolve_fragments(&self, fragment_ids: &RoaringBitmap) -> Result<Vec<Fragment>> {
        if let Some(fragments) = &self.resident {
            return Ok(fragment_ids
                .iter()
                .filter_map(|id| {
                    fragments
                        .binary_search_by_key(&u64::from(id), |fragment| fragment.id)
                        .ok()
                        .map(|index| fragments[index].clone())
                })
                .collect());
        }
        let actions: Vec<_> = self
            .buffer
            .iter()
            .filter(|tagged| bitmap_contains(fragment_ids, content::action_key(tagged)))
            .cloned()
            .collect();
        if self.children.is_empty() {
            let mut fragments = BTreeMap::new();
            self.store.apply_verified(&mut fragments, actions)?;
            return Ok(fragments.into_values().collect());
        }
        let mut leaves = Vec::new();
        self.collect_covering_leaves(
            self.children.clone(),
            actions,
            0..self.next_fragment_id,
            fragment_ids,
            &mut self.buffer_index().sequences(&self.buffer),
            &mut leaves,
        )
        .await?;
        let store = self.store.clone();
        // Covering leaves arrive in key order over disjoint routing ranges, and
        // `buffered` keeps that order, so concatenating per-leaf results that
        // are each ascending yields ascending distinct IDs without a global map.
        let fragments: Vec<Fragment> = futures::stream::iter(leaves)
            .map(move |(child, actions, end)| {
                let store = store.clone();
                async move {
                    let selected = store
                        .read_leaf_shared(&child, end)
                        .await?
                        .select(|fragment| bitmap_contains(fragment_ids, fragment.id));
                    store.replay(selected, actions)
                }
            })
            .buffered(MAX_READS_IN_FLIGHT)
            .try_concat()
            .await?;
        debug_assert!(
            fragments.windows(2).all(|pair| pair[0].id < pair[1].id),
            "resolved fragment IDs must strictly increase across covering leaves"
        );
        Ok(fragments)
    }

    /// Resolve which fragment holds the table's `offset`-th visible row and
    /// the offset within that fragment, by descending subtree visible-row
    /// totals: O(height) reads, never a stream from the first fragment.
    ///
    /// Totals are exact by the writer invariant on known counts. Buffered
    /// actions above a child adjust its total and are applied when its leaf
    /// is read, so the descent and a full stream agree. A resident snapshot
    /// descends the tree too, because its inventory holds no subtree totals.
    pub async fn fragment_at_row_offset(&self, offset: u64) -> Result<OffsetResolution> {
        if offset >= self.visible_rows {
            return Ok(OffsetResolution::BeyondEnd);
        }
        let mut remaining = offset;
        let mut children = self.children.clone();
        let mut actions: Vec<pb::FragmentTreeMutation> = self.buffer.clone();
        let mut end = self.next_fragment_id;
        let mut sequences = self.buffer_index().sequences(&self.buffer);
        loop {
            if children.is_empty() {
                // The whole table lives in buffered actions.
                let mut fragments = BTreeMap::new();
                self.store.apply_verified(&mut fragments, actions)?;
                return Ok(walk_fragments(fragments.into_values(), remaining));
            }
            let buckets = node::partition_buffer_by_child(&children, actions);
            node::validate_routed(&children, &buckets, end)?;
            let mut chosen = None;
            for (index, (child, bucket)) in children.iter().zip(buckets).enumerate() {
                let delta = sum_aggregate_deltas(
                    bucket.iter().map(|tagged| tagged.visible_rows_delta),
                    "visible_rows_delta",
                )?;
                let adjusted = apply_aggregate_delta(child.visible_rows, delta, "visible_rows")?;
                if remaining < adjusted {
                    chosen = Some((
                        child.clone(),
                        bucket,
                        node::exclusive_end(&children, index, end),
                    ));
                    break;
                }
                remaining -= adjusted;
            }
            let Some((child, bucket, child_end)) = chosen else {
                // Totals said the offset is inside, but the walk fell off:
                // an accounting bug, not a caller error.
                return Err(Error::internal(format!(
                    "row-offset descent exhausted children with {remaining} rows remaining"
                )));
            };
            if child.height == 0 {
                let stored = self.store.read_leaf_in_range(&child, child_end).await?;
                let fragments = self.store.replay(stored, bucket)?;
                return Ok(walk_fragments(fragments, remaining));
            }
            let internal = self.store.read_internal_in_range(&child, child_end).await?;
            sequences.admit(&child.path, &internal.buffer)?;
            actions = bucket;
            actions.extend(internal.buffer);
            children = internal.children;
            end = child_end;
        }
    }

    /// Collect all leaf fragments and replay buffered actions in action
    /// sequence number order. Up to the store's I/O parallelism of leaf GETs
    /// run ahead on their own tasks, while this task decodes and checks every
    /// leaf one at a time in key order, so decode costs what a serial read
    /// costs. Replay stays one pass over every decoded action, so a sequence
    /// repeated anywhere in the tree is rejected. The first failure in key
    /// order is the one returned, and dropping the call cancels its GETs.
    pub async fn materialize(&self) -> Result<Vec<Fragment>> {
        if let Some(fragments) = &self.resident {
            return Ok(fragments.as_ref().clone());
        }
        let (mut fragments, actions) = self.stored_and_routed().await?;
        self.store.apply_verified(&mut fragments, actions)?;
        Ok(fragments.into_values().collect())
    }

    /// Every leaf's stored records and every buffered action, not yet
    /// replayed.
    async fn stored_and_routed(
        &self,
    ) -> Result<(BTreeMap<u64, Fragment>, Vec<pb::FragmentTreeMutation>)> {
        let mut fragments: BTreeMap<u64, Fragment> = BTreeMap::new();
        if self.children.is_empty() {
            // Only a root keeps its whole table in buffered actions.
            return Ok((fragments, self.buffer.clone()));
        }
        // Beyond the decoded table, memory in flight is at most `width`
        // started leaves, each holding its fetched object or a cached leaf.
        let width = self.store.object_store.io_parallelism();
        let mut started: VecDeque<Result<StartedLeaf>> = VecDeque::with_capacity(width);
        let mut walk = Some(self.start_walk_from(0));
        let mut actions = Vec::new();
        loop {
            while started.len() < width
                && let Some(active) = walk.as_mut()
            {
                let next = match self.walk_next_leaf(active).await {
                    Ok(Some(leaf)) => StartedLeaf::start(&self.store, leaf).await,
                    Ok(None) => {
                        walk = None;
                        break;
                    }
                    Err(error) => Err(error),
                };
                // Only leaves before the first failure in key order are needed.
                if next.is_err() {
                    walk = None;
                }
                started.push_back(next);
            }
            let Some(leaf) = started.pop_front() else {
                break;
            };
            let leaf = leaf?;
            if let Some(fetch) = leaf.fetch {
                fetch.await;
            }
            let leaf_fragments = self
                .store
                .finish_leaf_read_in_range(&leaf.child, leaf.end, leaf.read)
                .await?;
            fragments.extend(
                leaf_fragments
                    .into_vec()
                    .into_iter()
                    .map(|fragment| (fragment.id, fragment)),
            );
            actions.extend(leaf.actions);
        }
        Ok((fragments, actions))
    }

    /// Stream this snapshot's fragments in ascending id order, leaf by leaf,
    /// starting at `lower_bound` inclusively. Holes and removed ids are
    /// omitted, and routing skips the subtrees that hold only smaller ids.
    /// This is the source a lazy scanner consumes.
    ///
    /// The scan starts with one leaf read in flight. Each time the consumer
    /// takes a leaf, the number of leaf reads allowed in flight doubles, up to
    /// eight. A consumer that stops inside the first leaf reads only that leaf,
    /// and a long scan overlaps its leaf reads. A resident snapshot answers
    /// from its retained inventory without storage IO, as point, set and
    /// complete reads do.
    ///
    /// The stream owns the snapshot, so it can outlive the caller, and stays
    /// pinned to the snapshot's version. Dropping it aborts the leaf decodes
    /// still in flight.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # use futures::TryStreamExt;
    /// # async fn example(tree: Arc<FragmentTree>) -> lance_core::Result<()> {
    /// let mut stream = tree.scan(100);
    /// while let Some(fragment) = stream.try_next().await? {
    ///     assert!(fragment.id >= 100);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn scan(self: Arc<Self>, lower_bound: u64) -> BoxStream<'static, Result<Fragment>> {
        self.scan_from(lower_bound, LeafReadahead::ramp(MAX_READS_IN_FLIGHT))
    }

    /// Stream as [`Self::scan`] does, with at most `readahead` leaf reads in
    /// flight from the first leaf on and no doubling. A caller that polls many
    /// scans at once can bound each to one read.
    ///
    /// ```no_run
    /// # use std::num::NonZeroUsize;
    /// # use std::sync::Arc;
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # use futures::TryStreamExt;
    /// # async fn example(tree: Arc<FragmentTree>) -> lance_core::Result<()> {
    /// let fragments: Vec<_> = tree
    ///     .clone()
    ///     .scan_with_readahead(0, NonZeroUsize::MIN)
    ///     .try_collect()
    ///     .await?;
    /// assert_eq!(fragments.len() as u64, tree.count_fragments());
    /// # Ok(())
    /// # }
    /// ```
    pub fn scan_with_readahead(
        self: Arc<Self>,
        lower_bound: u64,
        readahead: NonZeroUsize,
    ) -> BoxStream<'static, Result<Fragment>> {
        self.scan_from(lower_bound, LeafReadahead::fixed(readahead))
    }

    fn scan_from(
        self: Arc<Self>,
        lower_bound: u64,
        readahead: LeafReadahead,
    ) -> BoxStream<'static, Result<Fragment>> {
        if let Some(fragments) = self.resident.clone() {
            let first = fragments.partition_point(|fragment| fragment.id < lower_bound);
            return futures::stream::iter(first..fragments.len())
                .map(move |index| Ok(fragments[index].clone()))
                .boxed();
        }
        let walk = self.start_walk_from(lower_bound);
        if self.children.is_empty() {
            // A root without children holds its whole table in its buffer,
            // which starting the walk has already replayed.
            return match walk.pending_error {
                Some(error) => futures::stream::once(std::future::ready(Err(error))).boxed(),
                None => futures::stream::iter(
                    walk.ready
                        .into_iter()
                        .filter(move |fragment| fragment.id >= lower_bound)
                        .map(Ok),
                )
                .boxed(),
            };
        }
        let store = self.store.clone();
        let leaves = futures::stream::try_unfold((self, walk), |(tree, mut walk)| async move {
            let leaf = tree.walk_next_leaf(&mut walk).await?;
            Ok(leaf.map(|leaf| (leaf, (tree, walk))))
        })
        .boxed();
        LeafWindow {
            store,
            leaves,
            routing: Routing::Walking,
            reads: FuturesOrdered::new(),
            readahead,
        }
        // Routing skips subtrees below the bound, so only the first leaf can
        // hold smaller keys.
        .map_ok(move |fragments| {
            futures::stream::iter(
                fragments
                    .into_iter()
                    .filter(move |fragment| fragment.id >= lower_bound)
                    .map(Ok),
            )
        })
        .try_flatten()
        .boxed()
    }

    /// The walk of [`Self::scan`], one leaf at a time, so a caller that visits
    /// many versions can tell a leaf it already saw by its location. A leaf
    /// no buffered action routes to is read only when `read_stored` accepts
    /// its location, so a leaf the caller already holds costs no GET. A leaf
    /// with routed actions is always read and replayed. Leaves are read
    /// through `leaves`, so concurrent walks of versions that share a leaf
    /// read it once. Interiors are read through this tree's configured leaf
    /// cache, see [`TreeContext::with_leaf_cache`]. A resident snapshot reads
    /// its leaves too, because a leaf's identity needs the stored leaf.
    ///
    /// Each leaf is read in a spawned task, which dropping the stream aborts,
    /// and the next leaf is routed only once the consumer takes this one.
    ///
    /// ```no_run
    /// # use std::collections::BTreeSet;
    /// # use std::sync::{Arc, Mutex};
    /// # use lance_core::cache::LanceCache;
    /// # use lance_table::fragment_metadata::{FragmentTree, LeafFragments};
    /// # use lance_table::fragment_metadata::SingleFlightLeafCache;
    /// # use futures::TryStreamExt;
    /// # async fn example(versions: Vec<Arc<FragmentTree>>) -> lance_core::Result<()> {
    /// let leaves = SingleFlightLeafCache::new(LanceCache::with_capacity(64 * 1024 * 1024));
    /// let seen = Mutex::new(BTreeSet::new());
    /// for tree in versions {
    ///     let mut stream =
    ///         tree.leaf_stream(&leaves, |location| !seen.lock().unwrap().contains(location));
    ///     while let Some(leaf) = stream.try_next().await? {
    ///         if let LeafFragments::Stored { location, .. } = leaf {
    ///             seen.lock().unwrap().insert(location);
    ///         }
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn leaf_stream<'a>(
        self: Arc<Self>,
        leaves: &SingleFlightLeafCache,
        read_stored: impl Fn(&LeafLocation) -> bool + Send + 'a,
    ) -> BoxStream<'a, Result<LeafFragments>> {
        let mut walk = self.start_walk_from(0);
        if self.children.is_empty() {
            let leaf = match walk.pending_error.take() {
                Some(error) => Err(error),
                None => Ok(LeafFragments::Replayed(walk.ready.into())),
            };
            return futures::stream::once(std::future::ready(leaf)).boxed();
        }
        let leaves = leaves.clone();
        futures::stream::try_unfold(
            (self, walk, read_stored),
            move |(tree, mut walk, read_stored)| {
                let leaves = leaves.clone();
                async move {
                    loop {
                        let Some((child, actions, end)) = tree.walk_next_leaf(&mut walk).await?
                        else {
                            return Ok(None);
                        };
                        let stored_at = if actions.is_empty() {
                            let location = LeafLocation(tree.store.child_location(&child)?);
                            if !read_stored(&location) {
                                continue;
                            }
                            Some(location)
                        } else {
                            None
                        };
                        // Decoding is CPU-bound, and a caller such as cleanup
                        // polls many walks from one task, so each leaf decodes
                        // on its own.
                        let read = SpawnedRead(tokio::spawn(read_leaf_fragments(
                            tree.store.clone(),
                            leaves,
                            (child, actions, end),
                            stored_at,
                        )));
                        return Ok(Some((read.await?, (tree, walk, read_stored))));
                    }
                }
            },
        )
        .boxed()
    }

    /// Fragment ids that still have a buffered action somewhere in the tree,
    /// root included. Empty means every committed action has reached a leaf.
    #[cfg(any(test, feature = "test-util"))]
    pub async fn buffered_action_keys(&self) -> Result<HashSet<u64>> {
        let mut keys: HashSet<u64> = self.buffer.iter().map(content::action_key).collect();
        let mut pending = self.children.clone();
        while let Some(child) = pending.pop() {
            if child.height == 0 {
                continue;
            }
            let internal = self.store.read_internal(&child).await?;
            keys.extend(internal.buffer.iter().map(content::action_key));
            pending.extend(internal.children);
        }
        Ok(keys)
    }

    /// Read node occupancy and encoded sizes. Leaf sizes come from child references.
    #[cfg(any(test, feature = "test-util"))]
    pub async fn shape_report(&self) -> Result<ShapeReport> {
        let root = self.compacted_root();
        let mut report = ShapeReport {
            height: self.height(),
            root_bytes: root.encoded_len() as u64,
            root_buffer_len: self.buffer.len() as u64,
            root_buffer_bytes: node::buffer_bytes(&self.buffer),
            root_fanout: self.children.len() as u32,
            ..Default::default()
        };
        let mut stack: Vec<pb::FragmentTreeChild> = self.children.clone();
        while let Some(child) = stack.pop() {
            if child.height == 0 {
                report.leaf_bytes.push(child.object_size);
                report.leaf_object_bytes.push(child.object_size);
                report.leaf_keys.push(child.num_keys);
                continue;
            }
            let node = self.store.read_internal(&child).await?;
            report.node_bytes.push(child.object_size);
            report.node_fanouts.push(node.children.len() as u32);
            report.node_buffer_lens.push(node.buffer.len() as u64);
            report
                .node_buffer_bytes
                .push(node::buffer_bytes(&node.buffer));
            stack.extend(node.children);
        }
        Ok(report)
    }

    /// Advance aggregates and the ID allocator, and tag each action
    /// with its sequence number and deltas. The caller buffers the actions.
    fn stage_commit(
        &mut self,
        commit: ValidatedCommit,
        aggregate_deltas: Vec<commit::ActionDeltas>,
    ) -> Result<Vec<pb::FragmentTreeMutation>> {
        let actions = &commit.fragment_actions;
        let action_count = u64::try_from(actions.len()).map_err(|_| {
            Error::invalid_input(format!(
                "fragment metadata tree action count does not fit u64: {}",
                actions.len()
            ))
        })?;
        let next_action_sequence = self.next_action_sequence.checked_add(action_count).ok_or_else(|| {
            Error::invalid_input(format!(
                "fragment metadata tree action_sequence overflow: next_action_sequence={}, action_count={action_count}",
                self.next_action_sequence
            ))
        })?;
        let fragment_count_delta = sum_aggregate_deltas(
            aggregate_deltas.iter().map(|deltas| deltas.fragment_count),
            "fragment_count_delta",
        )?;
        let total_rows_delta = sum_aggregate_deltas(
            aggregate_deltas.iter().map(|deltas| deltas.physical_rows),
            "total_rows_delta",
        )?;
        let visible_rows_delta = sum_aggregate_deltas(
            aggregate_deltas.iter().map(|deltas| deltas.visible_rows),
            "visible_rows_delta",
        )?;

        let total_fragments = apply_aggregate_delta(
            self.total_fragments,
            fragment_count_delta,
            "total_fragments",
        )?;
        let total_rows = apply_aggregate_delta(self.total_rows, total_rows_delta, "total_rows")?;
        let visible_rows =
            apply_aggregate_delta(self.visible_rows, visible_rows_delta, "visible_rows")?;

        let next_fragment_id = commit.allocation_frontier(self.next_fragment_id)?;
        let first_action_sequence = self.next_action_sequence;
        let tagged = commit
            .fragment_actions
            .into_iter()
            .zip(aggregate_deltas)
            .enumerate()
            .map(|(offset, (action, deltas))| pb::FragmentTreeMutation {
                action_sequence: first_action_sequence + offset as u64,
                action: Some(action),
                fragment_count_delta: deltas.fragment_count,
                total_rows_delta: deltas.physical_rows,
                visible_rows_delta: deltas.visible_rows,
            })
            .collect();
        self.next_action_sequence = next_action_sequence;
        self.total_fragments = total_fragments;
        self.total_rows = total_rows;
        self.visible_rows = visible_rows;
        self.next_fragment_id = next_fragment_id;
        Ok(tagged)
    }

    fn compacted_root(&self) -> pb::FragmentTreeRoot {
        pb::FragmentTreeRoot {
            children: self.children.clone(),
            buffer: self.buffer.clone(),
            next_action_sequence: self.next_action_sequence,
        }
    }

    /// Flush the root buffer as far as it goes, then split, coalesce, and
    /// shrink the root. Every touched node is copy-on-write, so nothing here
    /// is visible until the root is published.
    async fn rewrite_tree(&mut self, mode: CommitMode) -> Result<WriteAcc> {
        let mut acc = WriteAcc::default();
        let rewrite = rewrite::RewriteNodes::new(&self.buffer, self.next_action_sequence);

        let children = std::mem::take(&mut self.children);
        let buffer = std::mem::take(&mut self.buffer);
        let (children, buffer, a) = if mode == CommitMode::Bulk {
            let (children, work) = bulk::materialize(
                &self.store,
                &self.config,
                children,
                buffer,
                self.next_action_sequence.saturating_sub(1),
            )
            .await?;
            (children, Vec::new(), work)
        } else {
            self.flush_internal(children, buffer, 0, 0..node::ROOT_EXCLUSIVE_END, &rewrite)
                .await?
        };
        acc.add(a);
        if acc.collapses > 0 {
            return Ok(acc);
        }
        self.children = children;
        self.buffer = buffer;
        node::fence(&mut self.children, 0);

        // A childless root (an empty table grown by appends) cannot flush
        // anywhere: materialize its buffer into the first leaves instead of
        // splitting into a childless internal node.
        if self.children.is_empty()
            && (node::internal_overflows(&self.children, &self.buffer, &self.config)
                || node::buffer_pressured(&self.buffer, &self.config))
        {
            let mut fragments = BTreeMap::new();
            let actions = std::mem::take(&mut self.buffer);
            self.store.apply_verified(&mut fragments, actions)?;
            let fragments: Vec<Fragment> = fragments.into_values().collect();
            let watermark = self.next_action_sequence.saturating_sub(1);
            for w in self
                .store
                .write_leaves(&fragments, None, watermark, &self.config)
                .await?
            {
                acc.io_bytes += w.io_bytes;
                self.children.push(w.child_ref);
            }
            // The split below stores these leaves inside interior nodes, and a
            // reader checks each node's first stored fence against the entry
            // that points at it. The first leaf must already own the root's
            // range, not start at the lowest live id.
            node::fence(&mut self.children, 0);
        }

        // Repair children that drains left underfull or single-child before
        // routing is split, as ingest does below the root, so no split piece
        // persists a single-child interior.
        if mode == CommitMode::Incremental {
            let children = std::mem::take(&mut self.children);
            let (children, a) = self
                .merge_small_children(children, node::ROOT_EXCLUSIVE_END, &rewrite)
                .await?;
            acc.add(a);
            self.children = children;
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
            let children = std::mem::take(&mut self.children);
            let (children, a) = self
                .merge_small_children(children, node::ROOT_EXCLUSIVE_END, &rewrite)
                .await?;
            acc.add(a);
            self.children = children;
        }

        self.maybe_shrink_root(&rewrite).await?;
        // The root keeps a lone interior only when adopting its level would
        // overflow. Readers reject that interior if it has one child, and no
        // sibling is left to pair it with, so the commit rebuilds instead.
        if let [only] = self.children.as_slice()
            && only.height > 0
            && only.num_children == 1
        {
            acc.collapses += 1;
            return Ok(acc);
        }
        node::fence(&mut self.children, 0);
        Ok(acc)
    }

    /// Build as many routing levels as needed for the root to fit. Bootstrap
    /// has no source nodes to track. A rewrite retains its sequence admission.
    async fn lift_root(
        store: &NodeStore,
        config: &FragmentTreeConfig,
        mut children: Vec<pb::FragmentTreeChild>,
        mut buffer: Vec<pb::FragmentTreeMutation>,
        rewrite: Option<&rewrite::RewriteNodes>,
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
                     (children={previous_width}); widen the directory budget",
                    config.max_node_bytes
                )));
            }
            children = Vec::with_capacity(pieces.len());
            buffer = Vec::new();
            for (group, actions) in pieces {
                let written = match rewrite {
                    Some(rewrite) => rewrite.write_internal(store, group, actions).await?,
                    None => store.write_internal(group, actions).await?,
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
    /// semantic pressure the rest keep buffering. Under structural pressure
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
        rewrite: &'a rewrite::RewriteNodes,
    ) -> BoxFuture<'a, Result<FlushResult>> {
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
                let pressure = if structural || buffer_bytes >= self.config.semantic_buffer_bytes {
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
                    // Under semantic pressure no batch pays for its object yet,
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
                let concurrency = indices.len();
                let drains: Vec<_> = indices
                    .into_iter()
                    .map(|idx| {
                        (
                            idx,
                            children[idx].clone(),
                            buckets.take(idx),
                            node::exclusive_end(&children, idx, range.end),
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
                if results.iter().any(|(_, (_, a))| a.collapses > 0) {
                    for (_, (_, work)) in results {
                        acc.io_bytes += work.io_bytes;
                    }
                    acc.collapses += 1;
                    return Ok((children, buckets.into_buffer(), acc));
                }
                // Apply from the right so splits and removals keep earlier indices valid.
                results.sort_unstable_by_key(|(idx, _)| std::cmp::Reverse(*idx));
                for (idx, (new_refs, a)) in results {
                    acc.add(a);
                    acc.flushes += 1;
                    acc.max_flush_depth = acc.max_flush_depth.max(depth);
                    buckets.replace(idx, new_refs.len());
                    children.splice(idx..idx + 1, new_refs);
                }
                routed = Some(buckets);
            }
            if let Some(buckets) = routed {
                buffer = buckets.into_buffer();
            }
            self.fence_left_edge(&mut children, range.start, range.end, &mut acc, rewrite)
                .await?;
            Ok((children, buffer, acc))
        })
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
        rewrite: &'a rewrite::RewriteNodes,
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
            let InternalNode {
                children: mut grandchildren,
                buffer,
            } = rewrite.read_internal(&self.store, first, child_end).await?;
            self.fence_left_edge(&mut grandchildren, lower_bound, child_end, acc, rewrite)
                .await?;
            let written = rewrite
                .write_internal(&self.store, grandchildren, buffer)
                .await?;
            acc.io_bytes += written.io_bytes;
            children[0] = written.child_ref;
            Ok(())
        })
    }

    /// Push `incoming` messages into the subtree rooted at `child` (at `depth`
    /// below the root, owning keys below `end`); apply at a leaf,
    /// recurse+buffer at an internal node; split on overflow. Returns the
    /// child ref(s) that now represent the subtree.
    fn ingest<'a>(
        &'a self,
        child: pb::FragmentTreeChild,
        incoming: Vec<pb::FragmentTreeMutation>,
        depth: u32,
        end: u64,
        rewrite: &'a rewrite::RewriteNodes,
    ) -> BoxFuture<'a, Result<IngestResult>> {
        Box::pin(async move {
            let mut acc = WriteAcc::default();
            if child.height == 0 {
                acc.materialized += incoming.len() as u64;
                let new_frags = content::replay_leaf(&self.store, &child, end, incoming).await?;

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
                    .map(|w| {
                        acc.io_bytes += w.io_bytes;
                        w.child_ref
                    })
                    .collect();
                node::fence(&mut refs, child.min_key);
                Ok((refs, acc))
            } else {
                let InternalNode {
                    children,
                    mut buffer,
                } = rewrite.read_internal(&self.store, &child, end).await?;
                buffer.extend(incoming);
                let before = buffer.len();
                buffer = content::squash_buffer(
                    buffer,
                    node::combine_from(&children, self.contiguous_from),
                );
                acc.squashed += (before - buffer.len()) as u64;
                // This node is one level deeper than the parent that flushed to it.
                let (children, buffer, a) = self
                    .flush_internal(children, buffer, depth + 1, child.min_key..end, rewrite)
                    .await?;
                acc.add(a);
                if acc.collapses > 0 {
                    return Ok((vec![child], acc));
                }
                let (children, a) = self.merge_small_children(children, end, rewrite).await?;
                acc.add(a);
                if let [only] = children.as_slice()
                    && only.height > 0
                    && only.num_children == 1
                {
                    acc.collapses += 1;
                    return Ok((vec![child], acc));
                }

                // An internal node whose children all vanished is dropped too (same
                // phantom-min_key=0 hazard as an empty leaf).
                if children.is_empty() {
                    return Ok((vec![], acc));
                }
                if node::internal_overflows(&children, &buffer, &self.config) {
                    let mut refs = Vec::new();
                    for (ch, buf) in node::split_internal(
                        children,
                        buffer,
                        self.config.split_piece_bytes(),
                        self.config.max_children_per_node,
                    ) {
                        let w = rewrite.write_internal(&self.store, ch, buf).await?;
                        acc.io_bytes += w.io_bytes;
                        refs.push(w.child_ref);
                    }
                    acc.splits += 1;
                    node::fence(&mut refs, child.min_key);
                    Ok((refs, acc))
                } else {
                    let w = rewrite
                        .write_internal(&self.store, children, buffer)
                        .await?;
                    acc.io_bytes += w.io_bytes;
                    let mut refs = vec![w.child_ref];
                    node::fence(&mut refs, child.min_key);
                    Ok((refs, acc))
                }
            }
        })
    }

    /// Coalesce runs of adjacent children when one underflows (leaf ≤ 0.25 B;
    /// internal < max_children_per_node/4 children), bounded so the merged node stays valid
    /// (leaves ≤ 0.6 B; internal ≤ max_children_per_node children). Reads/writes the merged
    /// node(s). Leaves concat fragments; internal nodes concat children + buffers.
    async fn merge_small_children(
        &self,
        children: Vec<pb::FragmentTreeChild>,
        end: u64,
        rewrite: &rewrite::RewriteNodes,
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
                let (mut merged, a) = self
                    .coalesce(
                        group,
                        children.get(j).map_or(end, |child| child.min_key),
                        rewrite,
                    )
                    .await?;
                node::fence(&mut merged, lower_bound);
                acc.add(a);
                acc.merges += 1;
                out.extend(merged);
            }
            i = j;
        }
        // A singleton at either edge needs its neighbor even when the normal
        // soft coalesce ceiling would leave it alone. Rebalance the combined
        // range under the hard node budget, keeping messages with their fences.
        let mut repaired: Vec<pb::FragmentTreeChild> = Vec::with_capacity(out.len());
        let mut pending: VecDeque<_> = out.into();
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
                    .read_internal(&self.store, &sibling, sibling_end)
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
                            .write_internal(&self.store, children, buffer)
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
                .write_internal(&self.store, children, buffer)
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
        rewrite: &rewrite::RewriteNodes,
    ) -> Result<(Vec<pb::FragmentTreeChild>, WriteAcc)> {
        let mut acc = WriteAcc::default();
        if group[0].height == 0 {
            let mut fragments: Vec<Fragment> = Vec::new();
            for (index, child) in group.iter().enumerate() {
                let child_end = node::exclusive_end(&group, index, end);
                fragments.extend(self.store.read_leaf_in_range(child, child_end).await?);
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
                .map(|w| {
                    acc.io_bytes += w.io_bytes;
                    w.child_ref
                })
                .collect();
            Ok((refs, acc))
        } else {
            let mut children: Vec<pb::FragmentTreeChild> = Vec::new();
            let mut buffer: Vec<pb::FragmentTreeMutation> = Vec::new();
            for (index, child) in group.iter().enumerate() {
                let child_end = node::exclusive_end(&group, index, end);
                let node = rewrite.read_internal(&self.store, child, child_end).await?;
                children.extend(node.children);
                buffer.extend(node.buffer);
            }
            // Joining parents exposes siblings that previously belonged to
            // separate ranges; repair their occupancy before persisting them.
            let (children, repaired) =
                Box::pin(self.merge_small_children(children, end, rewrite)).await?;
            acc.add(repaired);
            let w = rewrite
                .write_internal(&self.store, children, buffer)
                .await?;
            acc.io_bytes += w.io_bytes;
            Ok((vec![w.child_ref], acc))
        }
    }

    /// Remove a routing level when its contents fit in the root. Multiple
    /// children use the coalesce ceiling so growth and shrinkage have hysteresis.
    async fn maybe_shrink_root(&mut self, rewrite: &rewrite::RewriteNodes) -> Result<()> {
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
                let end = node::exclusive_end(&self.children, index, node::ROOT_EXCLUSIVE_END);
                let node = rewrite.read_internal(&self.store, child, end).await?;
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

    fn start_walk_from(&self, lower_bound: u64) -> FragmentWalk {
        let mut walk = FragmentWalk {
            lower_bound,
            ..Default::default()
        };
        if self.children.is_empty() {
            let mut fragments = BTreeMap::new();
            match self
                .store
                .apply_verified(&mut fragments, self.buffer.clone())
            {
                Ok(()) => walk.ready.extend(fragments.into_values()),
                Err(error) => walk.pending_error = Some(error),
            }
        } else {
            walk.sequences = self.buffer_index().sequences(&self.buffer);
            let action_buckets =
                node::partition_buffer_by_child(&self.children, self.buffer.clone());
            if let Err(error) =
                node::validate_routed(&self.children, &action_buckets, self.next_fragment_id)
            {
                walk.pending_error = Some(error);
                return walk;
            }
            walk.push_children(self.children.clone(), action_buckets, self.next_fragment_id);
        }
        walk
    }

    /// Traverse routing once, yielding leaves and their inherited actions in
    /// order. Leaf I/O can then be prefetched without buffering the full table.
    async fn walk_next_leaf(&self, walk: &mut FragmentWalk) -> Result<Option<BufferedChild>> {
        // Every walk reader steps through here, so a routing error found when
        // the walk started must surface here.
        if let Some(error) = walk.pending_error.take() {
            return Err(error);
        }
        while let Some((child, mut actions, end)) = walk.stack.pop() {
            if child.height == 0 {
                return Ok(Some((child, actions, end)));
            }
            let internal = self.store.read_internal_shared(&child, end).await?;
            walk.sequences.admit(&child.path, &internal.node.buffer)?;
            actions.extend(internal.node.buffer.iter().cloned());
            if internal.node.children.is_empty() {
                return Err(super::validation::corrupt(format!(
                    "Interior {} has no children",
                    child.path
                )));
            }
            let children = internal.node.children.clone();
            let action_buckets = node::partition_buffer_by_child(&children, actions);
            node::validate_routed(&children, &action_buckets, end)?;
            walk.push_children(children, action_buckets, end);
        }
        Ok(None)
    }

    /// Collect the leaves whose range intersects `fragment_ids`, each with
    /// the buffered actions for those ids routed to it. `routed` holds the
    /// actions for this node's `range`. Up to [`MAX_READS_IN_FLIGHT`] covering
    /// sibling interiors are read at once.
    fn collect_covering_leaves<'a>(
        &'a self,
        children: Vec<pb::FragmentTreeChild>,
        routed: Vec<pb::FragmentTreeMutation>,
        range: Range<u64>,
        fragment_ids: &'a RoaringBitmap,
        sequences: &'a mut node::DecodedSequences,
        leaves: &'a mut Vec<BufferedChild>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let buckets = node::partition_buffer_by_child(&children, routed);
            node::validate_routed(&children, &buckets, range.end)?;
            let covering: Vec<_> = node::with_exclusive_ends(children, range.end)
                .zip(buckets)
                .enumerate()
                .filter_map(|(index, ((child, end), actions))| {
                    let start = if index == 0 {
                        range.start
                    } else {
                        child.min_key
                    };
                    bitmap_intersects_range(fragment_ids, start, Some(end)).then_some((
                        child,
                        actions,
                        start..end,
                    ))
                })
                .collect();
            // Siblings depend only on this node, which is already verified, so
            // their reads overlap. Each result waits for its turn in the walk,
            // so the error returned is the first one a serial walk would meet.
            // The reads are built before the first await so no borrowing
            // closure is held across it, which the boxed future's `Send`
            // bound cannot prove for every lifetime.
            let reads: Vec<_> = covering
                .iter()
                .map(|(child, _, child_range)| async move {
                    if child.height == 0 {
                        None
                    } else {
                        Some(
                            self.store
                                .read_internal_shared(child, child_range.end)
                                .await,
                        )
                    }
                })
                .collect();
            let interiors: Vec<_> = futures::stream::iter(reads)
                .buffered(MAX_READS_IN_FLIGHT)
                .collect()
                .await;
            for ((child, mut actions, child_range), interior) in covering.into_iter().zip(interiors)
            {
                let Some(interior) = interior else {
                    leaves.push((child, actions, child_range.end));
                    continue;
                };
                let internal = interior?;
                sequences.admit(&child.path, &internal.node.buffer)?;
                actions.extend(
                    internal
                        .node
                        .buffer
                        .iter()
                        .filter(|tagged| bitmap_contains(fragment_ids, content::action_key(tagged)))
                        .cloned(),
                );
                self.collect_covering_leaves(
                    internal.node.children.clone(),
                    actions,
                    child_range,
                    fragment_ids,
                    sequences,
                    leaves,
                )
                .await?;
            }
            Ok(())
        })
    }
}

fn walk_fragments(
    fragments: impl IntoIterator<Item = Fragment>,
    mut remaining: u64,
) -> OffsetResolution {
    for fragment in fragments {
        // Decoded records always carry known counts.
        let visible = fragment.num_rows().unwrap_or(0) as u64;
        if remaining < visible {
            return OffsetResolution::Found(Box::new(fragment), remaining);
        }
        remaining -= visible;
    }
    OffsetResolution::BeyondEnd
}

fn bitmap_contains(fragment_ids: &RoaringBitmap, fragment_id: u64) -> bool {
    u32::try_from(fragment_id)
        .map(|fragment_id| fragment_ids.contains(fragment_id))
        .unwrap_or(false)
}

fn bitmap_intersects_range(
    fragment_ids: &RoaringBitmap,
    lower_bound: u64,
    upper_bound: Option<u64>,
) -> bool {
    if upper_bound.is_some_and(|upper_bound| upper_bound <= lower_bound) {
        return false;
    }
    let Ok(lower_bound) = u32::try_from(lower_bound) else {
        return false;
    };
    let upper_bound = upper_bound
        .and_then(|upper_bound| u32::try_from(upper_bound).ok())
        .map(|upper_bound| upper_bound.saturating_sub(1))
        .unwrap_or(u32::MAX);
    lower_bound <= upper_bound && fragment_ids.range_cardinality(lower_bound..=upper_bound) > 0
}

#[cfg(test)]
mod tests;
