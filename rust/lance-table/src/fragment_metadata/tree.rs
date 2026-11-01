// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Immutable routing with buffered fragment changes.
//!
//! Pressure drains the children whose pending bytes amortize a rewrite.
//! Splits and coalesces carry buffered mutations with their ranges. The
//! Version Manifest publishes the prepared root after every dependency has
//! been written.

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

use crate::format::pb;
use crate::format::{Fragment, Manifest};
use crate::fragment_metadata::TreeContext;
use crate::fragment_metadata::commit::{self, FragmentChanges, TouchedFragments};
use crate::fragment_metadata::content::{
    self, Totals, TotalsDelta, apply_aggregate_delta, sum_aggregate_deltas,
};
use crate::fragment_metadata::node::{self, FragmentTreeConfig};
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
pub use publication::PublicationPolicy;
pub use summary::LeafSummaries;
pub use transaction::{CommitMode, InventoryRetention, TreeCommit};

/// Accumulated write work over one commit.
#[derive(Debug, Default, Clone, Copy)]
struct WriteAcc {
    io_bytes: u64,
    flushes: u64,
    splits: u64,
    merges: u64,
    /// Mutations applied at leaves this commit.
    materialized: u64,
    /// Deepest tree level at which a flush occurred this commit (0 = root only).
    /// Values above 0 mean an interior buffer drained.
    max_flush_depth: u32,
}

impl WriteAcc {
    fn add(&mut self, o: Self) {
        self.io_bytes += o.io_bytes;
        self.flushes += o.flushes;
        self.splits += o.splits;
        self.merges += o.merges;
        self.materialized += o.materialized;
        self.max_flush_depth = self.max_flush_depth.max(o.max_flush_depth);
    }
}

/// How a rewrite of a subtree or of the root ended.
#[derive(Debug)]
enum Rewrite<T> {
    Done(T),
    /// A subtree shrank to a chain of single-child interiors, which the format
    /// forbids and no sibling can repair. Nothing this rewrite wrote is
    /// referenced, and the commit rebuilds in bulk.
    Collapsed {
        abandoned_bytes: u64,
    },
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

/// A child with the buffered mutations routed to it.
struct RoutedChild {
    child: pb::FragmentTreeChild,
    actions: Vec<pb::FragmentTreeMutation>,
    /// Exclusive end of the child's key range.
    end: u64,
}

/// A leaf a full read has walked to and not yet decoded.
struct StartedLeaf {
    child: pb::FragmentTreeChild,
    end: u64,
    actions: Vec<pb::FragmentTreeMutation>,
    read: PendingLeafRead,
    /// Its object GET, running on its own task. Dropping it cancels the GET.
    fetch: Option<RemoteHandle<()>>,
}

impl StartedLeaf {
    async fn start(
        store: &NodeStore,
        RoutedChild {
            child,
            actions,
            end,
        }: RoutedChild,
    ) -> Result<Self> {
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
    stack: Vec<RoutedChild>,
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
            self.stack.push(RoutedChild {
                child,
                actions,
                end,
            });
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
    leaves: BoxStream<'static, Result<RoutedChild>>,
    routing: Routing,
    reads: FuturesOrdered<SpawnedRead<Vec<Fragment>>>,
    readahead: LeafReadahead,
}

impl LeafWindow {
    fn spawn_read(
        &self,
        RoutedChild {
            child,
            actions,
            end,
        }: RoutedChild,
    ) -> SpawnedRead<Vec<Fragment>> {
        let store = self.store.clone();
        // Decoding is CPU-bound. A task per leaf lets the leaves in flight
        // decode in parallel instead of on the task polling this stream.
        SpawnedRead(tokio::spawn(async move {
            let fragments = store.read_leaf_shared(&child, end).await?.into_vec();
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
    /// The fragment holding the row. `visible_offset` counts the fragment's
    /// visible rows, so it equals a physical row offset only in a fragment
    /// without deletions.
    Found {
        fragment: Box<Fragment>,
        visible_offset: u64,
    },
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
    RoutedChild {
        child,
        actions,
        end,
    }: RoutedChild,
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
    totals: Totals,
    /// The next fragment id an append allocates; see `next_fragment_id`.
    /// Reads and rewrites end the root's last child here, because no stored
    /// record or buffered action may name an id that is not allocated yet.
    next_fragment_id: u64,
    /// Exact descriptor this snapshot opened or prepared.
    descriptor: Arc<pb::FragmentTree>,
    /// Complete records owned by this exact snapshot, when explicitly retained.
    inventory: Option<Arc<Vec<Fragment>>>,
}

impl FragmentTree {
    /// Bind an immutable snapshot to another handle for the same object namespace.
    pub fn with_object_store(mut self, store: Arc<ObjectStore>) -> Self {
        self.store.replace_store(store);
        self
    }

    /// Bind the already-wrapped primary store and apply the same wrappers to
    /// foreign stores. Aliased bases share one wrapped store. Checked caches
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
        self.store.wrap_stores(primary, wrappers);
        self
    }

    /// Borrow the complete inventory, if this snapshot explicitly retained it.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # fn example(tree: &FragmentTree) {
    /// if let Some(fragments) = tree.retained_inventory() {
    ///     assert_eq!(fragments.len() as u64, tree.count_fragments());
    /// }
    /// # }
    /// ```
    pub fn retained_inventory(&self) -> Option<&Arc<Vec<Fragment>>> {
        self.inventory.as_ref()
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

    /// Read and keep this snapshot's complete inventory for callers that need
    /// the whole list in memory. Other reads do not retain one. The load keeps
    /// up to the object store's I/O parallelism of leaf reads in flight.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # async fn example(tree: FragmentTree) -> lance_core::Result<()> {
    /// let retained = tree.retain_inventory().await?;
    /// assert!(retained.retained_inventory().is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn retain_inventory(mut self) -> Result<Self> {
        if self.inventory.is_none() {
            // The store clamps its I/O parallelism to at least one, so the
            // fallback only satisfies the type.
            let readahead = NonZeroUsize::new(self.store.object_store.io_parallelism())
                .unwrap_or(NonZeroUsize::MIN);
            let mut fragments: Vec<Fragment> = Arc::new(self.clone())
                .scan_with_readahead(0, readahead)
                .try_collect()
                .await?;
            // A cached leaf holds the lists of the store that decoded it.
            self.store.share_data_file_lists(&mut fragments);
            self.inventory = Some(Arc::new(fragments));
        }
        Ok(self)
    }

    /// Move this snapshot onto `store`. A commit interns only the records it
    /// changes, so a retained inventory keeps the interner that holds its lists.
    fn rebind_store(&mut self, mut store: NodeStore) {
        if self.inventory.is_some() {
            store.share_interner(&self.store);
        }
        self.store = store;
    }

    /// Bind a successfully published snapshot to its manifest version and storage.
    /// Call only after the Version Manifest has confirmed this exact descriptor.
    /// Writer budgets come from that manifest, including any config update.
    /// A manifest with another descriptor or `next_fragment_id` is refused.
    /// The descriptor does not carry that id, and a stale one would let the
    /// next append reuse a removed fragment id.
    ///
    /// ```
    /// # use lance_table::format::Manifest;
    /// # use lance_table::fragment_metadata::{FragmentTree, FragmentTreeConfig, TreeContext};
    /// # fn confirmed(tree: FragmentTree, manifest: &Manifest, config: FragmentTreeConfig,
    /// #              context: TreeContext) -> lance_core::Result<FragmentTree> {
    /// tree.into_published(manifest, config, context)
    /// # }
    /// ```
    pub fn into_published(
        mut self,
        manifest: &Manifest,
        config: FragmentTreeConfig,
        context: TreeContext,
    ) -> Result<Self> {
        config.validate()?;
        if manifest.fragment_tree.as_ref() != Some(&self.descriptor) {
            return Err(Error::invalid_input(format!(
                "Manifest version {} publishes a different fragment tree than this snapshot",
                manifest.version
            )));
        }
        let next_fragment_id = manifest.next_fragment_id();
        if next_fragment_id != self.next_fragment_id {
            return Err(Error::invalid_input(format!(
                "Manifest version {} allocates fragment ids from {next_fragment_id}, but this \
                 snapshot allocates from {}",
                manifest.version, self.next_fragment_id
            )));
        }
        self.version = manifest.version;
        self.config = config;
        self.rebind_store(context.into_store(&self.config, self.store.root_frontier));
        Ok(self)
    }

    /// Raise the next fragment id an append allocates to `next_fragment_id`.
    /// A rebuild of an existing inventory keeps every id its manifest already
    /// allocated, since index coverage may still name a removed fragment. A
    /// lower value changes nothing. A value above 2^32 is rejected, because
    /// fragment ids are u32.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # fn example(rebuilt: FragmentTree) -> lance_core::Result<()> {
    /// let rebuilt = rebuilt.raise_next_fragment_id(10)?;
    /// assert!(rebuilt.next_fragment_id() >= 10);
    /// # Ok(())
    /// # }
    /// ```
    pub fn raise_next_fragment_id(mut self, next_fragment_id: u64) -> Result<Self> {
        if next_fragment_id > node::ROOT_EXCLUSIVE_END {
            return Err(Error::invalid_input(format!(
                "next_fragment_id {next_fragment_id} must be at most 2^32"
            )));
        }
        self.next_fragment_id = self.next_fragment_id.max(next_fragment_id);
        Ok(self)
    }

    async fn build_initial_tree(
        store: NodeStore,
        config: FragmentTreeConfig,
        fragments: &[Fragment],
        version: u64,
        policy: PublicationPolicy,
    ) -> Result<Self> {
        config.validate()?;
        let mut layer: Vec<pb::FragmentTreeChild> = Vec::new();
        let next_fragment_id = commit::validate_initial_fragments(fragments, store.base())?;
        // Fragment::num_rows subtracts deleted rows unchecked, so this runs
        // after the checks above and before any leaf is written. As in a
        // commit, counts that do not fit are the caller's.
        let (rows, visible_rows) = content::fragment_row_counts(fragments)
            .map_err(|error| content::DeltaCheck::Skip.failure(error))?;
        for w in store.write_initial_leaves(fragments, &config).await? {
            layer.push(w.child_ref);
        }
        node::fence(&mut layer, 0);

        let (layer, _, _) = Self::lift_root(&store, &config, layer, Vec::new(), None).await?;
        let root = pb::FragmentTreeRoot {
            children: layer.clone(),
            buffer: Vec::new(),
            next_action_sequence: 1,
        };
        let (descriptor, _) = publication::checkpoint(&store, &config, root, policy).await?;

        Ok(Self {
            store,
            config,
            version,
            children: layer,
            buffer: Vec::new(),
            buffer_index: OnceLock::new(),
            next_action_sequence: 1,
            contiguous_from: 1,
            totals: Totals {
                fragments: fragments.len() as u64,
                rows,
                visible_rows,
            },
            next_fragment_id,
            descriptor: Arc::new(descriptor),
            inventory: None,
        })
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
        self.totals.fragments
    }

    /// Sum of physical rows across logical fragments.
    ///
    /// Admitted fragments have known counts. This reads root metadata only
    /// and performs no object-store IO.
    pub fn count_physical_rows(&self) -> u64 {
        self.totals.rows
    }

    /// Visible rows (physical minus deleted) across the table. Exact by the
    /// writer invariant on known counts. Root metadata only, no IO.
    pub fn count_visible_rows(&self) -> u64 {
        self.totals.visible_rows
    }

    /// The next fragment id an append may allocate: zero on an empty table,
    /// otherwise one above the largest id ever assigned, buffered adds
    /// included. A published tree matches [`Manifest::next_fragment_id`].
    pub fn next_fragment_id(&self) -> u64 {
        self.next_fragment_id
    }

    /// Number of actions currently buffered directly in the in-memory root.
    #[cfg(any(test, feature = "test-util"))]
    pub fn root_buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether `bytes` more of buffered actions keep the root under its
    /// flush threshold, so a commit of that size can stay buffered instead
    /// of draining in the same commit.
    fn can_buffer(&self, bytes: u64) -> bool {
        node::internal_logical_bytes(&[], &self.buffer)
            .checked_add(bytes)
            .is_some_and(|total| total < self.config.flush_threshold_bytes)
    }

    /// Number of direct child references currently held by the in-memory root.
    #[cfg(test)]
    fn root_child_count(&self) -> usize {
        self.children.len()
    }

    /// Encoded leaf sizes when the root points directly to leaves. Returns an
    /// empty list when there is an intervening routing level.
    #[cfg(test)]
    fn leaf_object_sizes(&self) -> Vec<u64> {
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
    /// same action_sequence ordering as [`Self::resolve_all_fragments`].
    pub async fn resolve_fragment(&self, frag_id: u64) -> Result<Option<Fragment>> {
        if let Some(fragments) = &self.inventory {
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

            let interior = self.store.read_interior_shared(&child, end).await?;
            // The root is already validated; cross-buffer checks begin with the
            // first nonempty interior buffer.
            if !interior.node.buffer.is_empty() {
                sequences
                    .get_or_insert_with(|| self.buffer_index().sequences(&self.buffer))
                    .admit_interior(&child.path, &interior)?;
            }
            actions.extend(
                interior
                    .node
                    .buffer
                    .iter()
                    .filter(|tagged| content::action_key(tagged) == frag_id)
                    .cloned(),
            );
            children = Cow::Owned(interior.node.children.clone());
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
        if let Some(fragments) = &self.inventory {
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
            .map(move |leaf| {
                let store = store.clone();
                async move {
                    let selected = store
                        .read_leaf_shared(&leaf.child, leaf.end)
                        .await?
                        .select(fragment_ids);
                    store.replay(selected, leaf.actions)
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
    /// is read, so the descent and a full stream agree. A snapshot with a
    /// retained inventory descends the tree too, because the inventory holds
    /// no subtree totals.
    pub async fn fragment_at_visible_row_offset(&self, offset: u64) -> Result<OffsetResolution> {
        if offset >= self.totals.visible_rows {
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
                return locate_row_offset(fragments.into_values(), remaining);
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
                let stored = self
                    .store
                    .read_leaf_shared(&child, child_end)
                    .await?
                    .into_vec();
                let fragments = self.store.replay(stored, bucket)?;
                return locate_row_offset(fragments, remaining);
            }
            let interior = self.store.read_interior_in_range(&child, child_end).await?;
            sequences.admit(&child.path, &interior.buffer)?;
            actions = bucket;
            actions.extend(interior.buffer);
            children = interior.children;
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
    pub async fn resolve_all_fragments(&self) -> Result<Vec<Fragment>> {
        if let Some(fragments) = &self.inventory {
            return Ok(fragments.as_ref().clone());
        }
        let (mut fragments, actions) = self.stored_and_routed().await?;
        self.store.apply_verified(&mut fragments, actions)?;
        Ok(fragments.into_values().collect())
    }

    /// Load every stored leaf fragment and collect every buffered action
    /// without replaying them yet.
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
    /// and a long scan overlaps its leaf reads. A snapshot with a retained
    /// inventory answers from it without storage IO, as point, set and
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
        if let Some(fragments) = self.inventory.clone() {
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
    /// read it once. Interiors are read through this tree's cache, see
    /// [`TreeContext::with_cache`]. A snapshot with a retained inventory reads
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
                        let Some(leaf) = tree.walk_next_leaf(&mut walk).await? else {
                            return Ok(None);
                        };
                        let stored_at = if leaf.actions.is_empty() {
                            let location = LeafLocation(tree.store.child_location(&leaf.child)?);
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
                            leaf,
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
            let interior = self
                .store
                .read_interior_in_range(&child, node::ROOT_EXCLUSIVE_END)
                .await?;
            keys.extend(interior.buffer.iter().map(content::action_key));
            pending.extend(interior.children);
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
            root_buffer_bytes: node::internal_logical_bytes(&[], &self.buffer),
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
            let node = self
                .store
                .read_interior_in_range(&child, node::ROOT_EXCLUSIVE_END)
                .await?;
            report.node_bytes.push(child.object_size);
            report.node_fanouts.push(node.children.len() as u32);
            report.node_buffer_lens.push(node.buffer.len() as u64);
            report
                .node_buffer_bytes
                .push(node::internal_logical_bytes(&[], &node.buffer));
            stack.extend(node.children);
        }
        Ok(report)
    }

    /// Advance aggregates and the ID allocator, and tag each action
    /// with its sequence number and deltas. The caller buffers the actions.
    fn stage_commit(
        &mut self,
        changes: FragmentChanges,
        aggregate_deltas: Vec<commit::ActionDeltas>,
    ) -> Result<Vec<pb::FragmentTreeMutation>> {
        let actions = &changes.fragment_actions;
        let action_count = actions.len() as u64;
        let next_action_sequence = self.next_action_sequence.checked_add(action_count).ok_or_else(|| {
            Error::invalid_input(format!(
                "fragment metadata tree action_sequence overflow: next_action_sequence={}, action_count={action_count}",
                self.next_action_sequence
            ))
        })?;
        let next_fragment_id = changes.checked_next_fragment_id(self.next_fragment_id)?;
        let first_action_sequence = self.next_action_sequence;
        let tagged: Vec<_> = changes
            .fragment_actions
            .into_iter()
            .zip(aggregate_deltas)
            .enumerate()
            .map(|(offset, (action, deltas))| pb::FragmentTreeMutation {
                action_sequence: first_action_sequence + offset as u64,
                action: Some(action),
                fragment_count_delta: deltas.fragment_count,
                total_rows_delta: deltas.total_rows,
                visible_rows_delta: deltas.visible_rows,
            })
            .collect();
        // Sums of a commit's own deltas are the caller's to keep in range.
        // Applying them is not, because a resolved record supplied each
        // decrease, so an underflow means the stored totals disagree.
        let delta =
            TotalsDelta::of(&tagged).map_err(|error| content::DeltaCheck::Skip.failure(error))?;
        self.totals = self.totals.apply(delta)?;
        self.next_action_sequence = next_action_sequence;
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
    async fn walk_next_leaf(&self, walk: &mut FragmentWalk) -> Result<Option<RoutedChild>> {
        // Every walk reader steps through here, so a routing error found when
        // the walk started must surface here.
        if let Some(error) = walk.pending_error.take() {
            return Err(error);
        }
        while let Some(routed) = walk.stack.pop() {
            if routed.child.height == 0 {
                return Ok(Some(routed));
            }
            let RoutedChild {
                child,
                mut actions,
                end,
            } = routed;
            let interior = self.store.read_interior_shared(&child, end).await?;
            walk.sequences.admit_interior(&child.path, &interior)?;
            actions.extend(interior.node.buffer.iter().cloned());
            if interior.node.children.is_empty() {
                return Err(super::validation::corrupt(format!(
                    "Interior {} has no children",
                    child.path
                )));
            }
            let children = interior.node.children.clone();
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
        leaves: &'a mut Vec<RoutedChild>,
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
                                .read_interior_shared(child, child_range.end)
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
                    leaves.push(RoutedChild {
                        child,
                        actions,
                        end: child_range.end,
                    });
                    continue;
                };
                let interior = interior?;
                sequences.admit_interior(&child.path, &interior)?;
                actions.extend(
                    interior
                        .node
                        .buffer
                        .iter()
                        .filter(|tagged| bitmap_contains(fragment_ids, content::action_key(tagged)))
                        .cloned(),
                );
                self.collect_covering_leaves(
                    interior.node.children.clone(),
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

fn locate_row_offset(
    fragments: impl IntoIterator<Item = Fragment>,
    mut remaining: u64,
) -> Result<OffsetResolution> {
    for fragment in fragments {
        // Decoded records always carry known counts.
        let visible = fragment.num_rows().unwrap_or(0) as u64;
        if remaining < visible {
            return Ok(OffsetResolution::Found {
                fragment: Box::new(fragment),
                visible_offset: remaining,
            });
        }
        remaining -= visible;
    }
    // Totals said the offset is inside, as in the descent that called this.
    Err(Error::internal(format!(
        "row-offset search exhausted fragments with {remaining} rows remaining"
    )))
}

pub(super) fn bitmap_contains(fragment_ids: &RoaringBitmap, fragment_id: u64) -> bool {
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
