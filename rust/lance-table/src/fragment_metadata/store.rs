// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Immutable protobuf routing nodes and columnar Lance leaves.
//!
//! Leaves hold one row per fragment with ordered files and overlays.
//! Object paths are relative to the dataset root; every rewrite gets a new UUID.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex};

use futures::{FutureExt, TryStreamExt, future::BoxFuture};
use prost::Message;

use crate::format::pb;
use crate::format::pb::fragment_action::Action;
use crate::format::{DataFileFieldInterner, Fragment};
use crate::fragment_metadata::content;
use crate::fragment_metadata::node::{self, InteriorNode};
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_io::ReadBatchParams;
use lance_io::object_reader::SmallReader;
use lance_io::object_store::{ObjectStore, WrappingObjectStore};
use lance_io::scheduler::ScanScheduler;
use lance_io::traits::Reader;
use object_store::path::Path;
use object_store::{GetOptions, ObjectStore as OSObjectStore, PutOptions, PutPayload};
use roaring::{RoaringBitmap, RoaringTreemap};
use tokio::sync::Semaphore;
use uuid::Uuid;
mod single_flight;
mod upload;
mod validation_reads;

pub use single_flight::SingleFlightLeafCache;
use upload::leaf_put_window;
pub(super) use upload::{LeafUploads, Uploaded};

/// Directory under a dataset base that holds every tree leaf, node and root.
pub const TREE_DIR: &str = "_bt";

// Decode each leaf in batches of READ_BATCH_ROWS, with up to
// READ_BATCH_READAHEAD decoder batches in flight for that leaf. A scan may
// keep several leaves in flight separately.
const READ_BATCH_ROWS: u32 = 16 * 1024;
const READ_BATCH_READAHEAD: u32 = 16;

/// A written node: its parent child reference plus the actual bytes written to
/// storage (for write-amplification accounting).
#[derive(Debug)]
pub(super) struct Written {
    pub(super) child_ref: pb::FragmentTreeChild,
    pub(super) io_bytes: u64,
}

/// Reads and writes fragment metadata tree node files against an object store.
#[derive(Clone)]
pub(super) struct NodeStore {
    pub(super) object_store: Arc<ObjectStore>,
    base: Path,
    scheduler: Arc<ScanScheduler>,
    /// Checked leaves, checked interiors and leaf sightings, shared by every
    /// tree that holds the same cache. It never changes what a read returns.
    cache: Option<LanceCache>,
    validation_reads: Option<Arc<validation_reads::ValidationReads>>,
    /// Shared by clones, so concurrent drains that each write leaves still
    /// hold at most this many encoded leaves in PUTs at once.
    leaf_puts: Arc<Semaphore>,
    /// The frontier of the root above every interior this store reads. It
    /// bounds their buffered sequences and leaf watermarks.
    pub(super) root_frontier: u64,
    pub(super) hard_capacity_bytes: u64,
    foreign_bases: HashMap<u32, ForeignBase>,
    /// Each bound in-memory store's id mapped to the store it wraps, which
    /// names its namespace. Holding the Arc pins that address, so no other
    /// store can take the id while this map names it.
    memory_store_owners: HashMap<usize, Arc<dyn OSObjectStore>>,
    interner: Arc<Mutex<DataFileFieldInterner>>,
    /// Records passed to `encode_leaf`, so tests can bound encode work.
    #[cfg(test)]
    encoded_records: Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Clone)]
struct ForeignBase {
    store: Arc<ObjectStore>,
    base: Path,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct NodeLocation {
    store_prefix: String,
    memory_store: Option<usize>,
    path: Path,
}

/// Where a leaf object resolves. Leaf objects are never rewritten in place,
/// so one location always holds the same stored records, and a foreign leaf
/// never shares a location with a local leaf of the same relative path. The
/// ids and file paths read from a location are the same in every version,
/// but an inherited base id is stamped from the child reference that names
/// the leaf, so references through aliased bases see different ones.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LeafLocation(pub(super) NodeLocation);

/// A leaf's location and the parent reference its contents were checked
/// against. Any change to the reference, including its inherited base, forms
/// a different key, so the leaf is checked again.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LeafKey {
    location: NodeLocation,
    reference: Vec<u8>,
}

impl LeafKey {
    fn new(store: &NodeStore, child: &pb::FragmentTreeChild) -> Result<Self> {
        Ok(Self {
            location: store.child_location(child)?,
            reference: child.encode_to_vec(),
        })
    }
}

impl CacheKey for LeafKey {
    type ValueType = CheckedLeaf;

    // A label only. write_key carries the identity, including the store and reference.
    fn key(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.location.path.as_ref())
    }

    fn type_name() -> &'static str {
        "FragmentTreeLeaf"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.fragment_metadata.leaf", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(&self.location.store_prefix);
        match self.location.memory_store {
            Some(store) => {
                builder.write_some();
                builder.write_u64(store as u64);
            }
            None => builder.write_none(),
        }
        builder.write_str(self.location.path.as_ref());
        builder.write_bytes(&self.reference);
    }
}

/// Identifies an interior as [`LeafKey`] identifies a leaf.
struct InteriorKey(LeafKey);

impl CacheKey for InteriorKey {
    type ValueType = CheckedInterior;

    fn key(&self) -> Cow<'_, str> {
        self.0.key()
    }

    fn type_name() -> &'static str {
        "FragmentTreeInterior"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.fragment_metadata.interior", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        self.0.write_key(builder);
    }
}

/// An interior after every check against its parent reference.
pub(super) struct CheckedInterior {
    pub(super) node: InteriorNode,
    /// The newest action sequence the node or its leaf watermarks name. Trees
    /// share cached interiors, so each reader checks this against its own frontier.
    newest_action_sequence: u64,
    /// The cache key omits where the inherited range ends, so each reference
    /// checks these bounds against its own end.
    buffered_targets: Option<RangeInclusive<u64>>,
    /// The action sequences of `node.buffer`. Buffers are not in sequence
    /// order, so admitting them one at a time would keep shifting a sorted
    /// array. A hit merges this set instead.
    pub(super) sequences: RoaringTreemap,
    // Sized once, since sizing walks every buffered mutation.
    heap_bytes: usize,
}

impl CheckedInterior {
    fn new(mut node: InteriorNode) -> Self {
        // Decoded `bytes` fields are slices of the fetched object, and a slice
        // keeps the whole object allocated. Copy them so the cache retains only
        // the bytes it weighs.
        for mutation in &mut node.buffer {
            let Some(action) = mutation
                .action
                .as_mut()
                .and_then(|action| action.action.as_mut())
            else {
                continue;
            };
            let deletion_file = match action {
                Action::UpsertFragment(fragment) => {
                    if let Some(pb::data_fragment::RowIdSequence::InlineRowIds(ids)) =
                        &mut fragment.row_id_sequence
                    {
                        *ids = bytes::Bytes::copy_from_slice(ids);
                    }
                    fragment.deletion_file.as_mut()
                }
                Action::AddDeletionFile(add) => add.deletion_file.as_mut(),
                _ => None,
            };
            if let Some(inline) = deletion_file.and_then(|file| file.inline.as_mut()) {
                *inline = bytes::Bytes::copy_from_slice(inline);
            }
        }
        let newest_action_sequence = node
            .buffer
            .iter()
            .map(|mutation| mutation.action_sequence)
            .chain(
                node.children
                    .iter()
                    .map(|child| child.materialized_through_action_sequence),
            )
            .max()
            .unwrap_or(0);
        let buffered_targets = buffered_targets(&node.buffer);
        let sequences: RoaringTreemap = node
            .buffer
            .iter()
            .map(|mutation| mutation.action_sequence)
            .collect();
        let mut context = Context::new();
        let heap_bytes = node.children.deep_size_of_children(&mut context)
            + node.buffer.deep_size_of_children(&mut context)
            + sequences.serialized_size();
        Self {
            node,
            newest_action_sequence,
            buffered_targets,
            sequences,
            heap_bytes,
        }
    }
}

/// The lowest through the highest fragment a buffer targets.
fn buffered_targets(buffer: &[pb::FragmentTreeMutation]) -> Option<RangeInclusive<u64>> {
    buffer
        .iter()
        .map(content::action_key)
        .fold(None, |bounds, target| {
            Some(match bounds {
                None => target..=target,
                Some(bounds) => (*bounds.start()).min(target)..=(*bounds.end()).max(target),
            })
        })
}

impl DeepSizeOf for CheckedInterior {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.heap_bytes
    }
}

/// A successful whole read. The next whole decode admits the leaf.
struct Sighting<'a>(&'a LeafKey);

impl CacheKey for Sighting<'_> {
    type ValueType = Sighted;

    fn key(&self) -> Cow<'_, str> {
        self.0.key()
    }

    fn type_name() -> &'static str {
        "FragmentTreeLeafSighting"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.fragment_metadata.leaf-sighting", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        self.0.write_key(builder);
    }
}

/// Weighed by its key alone.
struct Sighted;

impl DeepSizeOf for Sighted {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

/// A leaf's fragments after every check against its parent reference.
pub(super) struct CheckedLeaf {
    fragments: Vec<Fragment>,
    // Sized once, since sizing walks every file of every fragment.
    heap_bytes: usize,
}

impl CheckedLeaf {
    fn new(fragments: Vec<Fragment>) -> Self {
        let heap_bytes = fragments.deep_size_of_children(&mut Context::new());
        Self {
            fragments,
            heap_bytes,
        }
    }
}

impl DeepSizeOf for CheckedLeaf {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.heap_bytes
    }
}

/// Separate in-memory stores can have identical URL prefixes, so a location
/// names one by its address. The address names a store only while it lives,
/// so a location must not outlive its memory store.
fn memory_store_id(inner: &Arc<dyn OSObjectStore>) -> usize {
    Arc::as_ptr(inner) as *const () as usize
}

/// A checked leaf, shared with the cache when the read found or admitted
/// it there, so a caller copies only the records it keeps.
pub(super) enum LeafRead {
    Cached(Arc<CheckedLeaf>),
    Owned(Vec<Fragment>),
}

impl LeafRead {
    fn fragments(&self) -> &[Fragment] {
        match self {
            Self::Cached(leaf) => &leaf.fragments,
            Self::Owned(fragments) => fragments,
        }
    }

    pub(super) fn select(self, ids: &RoaringBitmap) -> Vec<Fragment> {
        match self {
            Self::Cached(leaf) => {
                let (Some(first), Some(last)) = (leaf.fragments.first(), leaf.fragments.last())
                else {
                    return Vec::new();
                };
                // Leaf decode rejects an id above u32::MAX and keeps ids ascending.
                ids.range(first.id as u32..=last.id as u32)
                    .filter_map(|id| {
                        leaf.fragments
                            .binary_search_by_key(&u64::from(id), |fragment| fragment.id)
                            .ok()
                    })
                    .map(|row| leaf.fragments[row].clone())
                    .collect()
            }
            Self::Owned(fragments) => fragments
                .into_iter()
                .filter(|fragment| super::tree::bitmap_contains(ids, fragment.id))
                .collect(),
        }
    }

    pub(super) fn into_vec(self) -> Vec<Fragment> {
        match self {
            Self::Cached(leaf) => leaf.fragments.clone(),
            Self::Owned(fragments) => fragments,
        }
    }
}

/// A stored leaf's checked records, shared with the cache when the read
/// found or admitted the leaf there, so a walk over many versions copies none.
pub struct StoredFragments(pub(super) LeafRead);

impl std::ops::Deref for StoredFragments {
    type Target = [Fragment];

    fn deref(&self) -> &[Fragment] {
        self.0.fragments()
    }
}

impl std::fmt::Debug for StoredFragments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl PartialEq for StoredFragments {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

/// A whole leaf read whose source is decided and whose decode is still to
/// come. Splitting a read here lets a walk fetch several leaf objects ahead
/// of the task that decodes them one at a time.
pub(super) struct PendingLeafRead(LeafSource);

impl PendingLeafRead {
    /// The object this read has yet to GET, or None when the leaf is served
    /// from the session cache or from bulk validation scratch.
    pub(super) fn object_to_fetch(&self) -> Option<&LeafObject> {
        match &self.0 {
            LeafSource::Cached(_) => None,
            LeafSource::Retained { cell, object, .. } => (!cell.initialized()).then_some(object),
            LeafSource::Missed { object, .. } | LeafSource::Uncached(object) => Some(object),
        }
    }
}

enum LeafSource {
    Cached(Arc<CheckedLeaf>),
    /// Loaded once per bulk commit, then read back from scratch.
    Retained {
        reads: Arc<validation_reads::ValidationReads>,
        cell: Arc<tokio::sync::OnceCell<std::path::PathBuf>>,
        object: LeafObject,
    },
    /// Admitted to `leaves` on its second decode.
    Missed {
        leaves: LanceCache,
        key: LeafKey,
        object: LeafObject,
    },
    Uncached(LeafObject),
}

/// A leaf's stored object, requested at most once. Every clone shares the
/// one response, so a GET issued ahead of the decode serves it.
#[derive(Clone)]
pub(super) struct LeafObject(SmallReader);

impl LeafObject {
    /// Receive the whole object. The decode reports a failed GET, so the
    /// error surfaces where a serial read would surface it.
    pub(super) async fn fetch(self) {
        let _ = self.0.get_all().await;
    }
}

impl NodeStore {
    pub(super) fn new(
        object_store: Arc<ObjectStore>,
        base: Path,
        scheduler: Arc<ScanScheduler>,
    ) -> Self {
        Self {
            leaf_puts: leaf_put_window(&object_store),
            object_store,
            base,
            scheduler,
            cache: None,
            validation_reads: None,
            root_frontier: u64::MAX,
            hard_capacity_bytes: u64::MAX,
            foreign_bases: HashMap::new(),
            memory_store_owners: HashMap::new(),
            interner: Arc::new(Mutex::new(DataFileFieldInterner::default())),
            #[cfg(test)]
            encoded_records: Arc::default(),
        }
    }

    #[cfg(test)]
    pub(super) fn context(&self) -> super::TreeContext {
        let bases = self
            .foreign_bases
            .iter()
            .map(|(id, owner)| (*id, (owner.store.clone(), owner.base.clone())))
            .collect();
        let context = super::TreeContext::new(
            self.object_store.clone(),
            self.base.clone(),
            self.scheduler.clone(),
        )
        .with_foreign_bases(bases);
        match &self.cache {
            Some(cache) => context.with_cache(cache.clone()),
            None => context,
        }
    }

    pub(super) fn set_foreign_bases(
        &mut self,
        foreign_bases: HashMap<u32, (Arc<ObjectStore>, Path)>,
    ) {
        self.foreign_bases = foreign_bases
            .into_iter()
            .map(|(id, (store, base))| (id, ForeignBase { store, base }))
            .collect();
    }

    pub(super) fn set_cache(&mut self, cache: LanceCache) {
        self.cache = Some(cache);
    }

    /// Bind another handle as the primary store. Foreign stores keep their
    /// handles. An in-memory primary keeps its location only when the new
    /// handle shares its `inner`.
    pub(super) fn replace_store(&mut self, object_store: Arc<ObjectStore>) {
        let unchanged: Vec<_> = std::iter::once(&object_store)
            .chain(self.foreign_bases.values().map(|foreign| &foreign.store))
            .map(|store| (store.clone(), store.clone()))
            .collect();
        self.memory_store_owners = self.rebound_memory_owners(&unchanged);
        self.bind_primary(object_store);
    }

    /// Wrap every store. `primary` must be `self.object_store` with `wrappers`
    /// already applied, and the same wrappers are applied to foreign stores.
    /// Aliased stores share one wrapped store. Wrappers preserve the object
    /// namespace, so every location stays the same.
    pub(super) fn wrap_stores(
        &mut self,
        primary: Arc<ObjectStore>,
        wrappers: &[Arc<dyn WrappingObjectStore>],
    ) {
        let mut rebound_stores = Vec::with_capacity(self.foreign_bases.len() + 1);
        rebound_stores.push((self.object_store.clone(), primary.clone()));
        for foreign in self.foreign_bases.values_mut() {
            let wrapped = match rebound_stores
                .iter()
                .find(|(original, _)| Arc::ptr_eq(original, &foreign.store))
            {
                Some((_, wrapped)) => wrapped.clone(),
                None => {
                    let mut wrapped = foreign.store.as_ref().clone();
                    for wrapper in wrappers {
                        wrapped.apply_wrapper(wrapper.as_ref());
                    }
                    Arc::new(wrapped)
                }
            };
            rebound_stores.push((foreign.store.clone(), wrapped.clone()));
            foreign.store = wrapped;
        }
        self.memory_store_owners = self.rebound_memory_owners(&rebound_stores);
        self.bind_primary(primary);
    }

    /// Owners after rebinding each `(original, rebound)` pair. A rebound store
    /// inherits the owner of its original.
    fn rebound_memory_owners(
        &self,
        rebound_stores: &[(Arc<ObjectStore>, Arc<ObjectStore>)],
    ) -> HashMap<usize, Arc<dyn OSObjectStore>> {
        rebound_stores
            .iter()
            .filter(|(original, _)| original.scheme() == "memory")
            .map(|(original, rebound)| {
                let owner = self
                    .memory_store_owners
                    .get(&memory_store_id(&original.inner))
                    .unwrap_or(&original.inner)
                    .clone();
                (memory_store_id(&rebound.inner), owner)
            })
            .collect()
    }

    fn bind_primary(&mut self, object_store: Arc<ObjectStore>) {
        self.scheduler = ScanScheduler::new(
            object_store.clone(),
            lance_io::scheduler::SchedulerConfig::max_bandwidth(&object_store),
        );
        self.leaf_puts = leaf_put_window(&object_store);
        self.object_store = object_store;
        self.validation_reads = None;
    }

    /// Start retaining whole leaf reads, keeping any leaves this commit already
    /// retained. Leaf objects are immutable and keyed by their full parent
    /// reference, so every retained read stays exact until the commit clears it.
    pub(super) fn retain_validation_reads(&mut self) -> Result<()> {
        if self.validation_reads.is_none() {
            self.validation_reads = Some(Arc::new(validation_reads::ValidationReads::new()?));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn has_validation_reads(&self) -> bool {
        self.validation_reads.is_some()
    }

    pub(super) fn clear_validation_reads(&mut self) {
        self.validation_reads = None;
    }

    /// A copy that reads every object from storage and bounds sequences by
    /// `root_frontier`. It shares no checked leaves or interiors and
    /// retains no validation reads. Wrappers and foreign bases stay, so IO
    /// trackers see its reads.
    pub(super) fn uncached(&self, root_frontier: u64) -> Self {
        Self {
            cache: None,
            validation_reads: None,
            root_frontier,
            ..self.clone()
        }
    }

    /// Read a columnar leaf into fragment records.
    #[cfg(test)]
    pub(super) async fn read_leaf(&self, child: &pb::FragmentTreeChild) -> Result<Vec<Fragment>> {
        let read = self.start_leaf_read(child).await?;
        Ok(self.finish_leaf_read(child, read).await?.into_vec())
    }

    /// A leaf checked against its parent reference and against the end of its
    /// routing range. The range check runs on cache hits too, since the cache
    /// key does not include the end a traversal routes the leaf under.
    pub(super) async fn read_leaf_shared(
        &self,
        child: &pb::FragmentTreeChild,
        end: u64,
    ) -> Result<LeafRead> {
        let read = self.start_leaf_read(child).await?;
        self.finish_leaf_read_in_range(child, end, read).await
    }

    /// Decide where a whole read of `child` is served from, before any GET.
    pub(super) async fn start_leaf_read(
        &self,
        child: &pb::FragmentTreeChild,
    ) -> Result<PendingLeafRead> {
        let source = match (&self.validation_reads, &self.cache) {
            (Some(reads), _) => LeafSource::Retained {
                cell: reads.cell(self, child).await?,
                reads: reads.clone(),
                object: self.leaf_object(child)?,
            },
            (None, Some(leaves)) => {
                let key = LeafKey::new(self, child)?;
                match leaves.get_with_key(&key).await {
                    Some(leaf) => LeafSource::Cached(leaf),
                    None => LeafSource::Missed {
                        leaves: leaves.clone(),
                        key,
                        object: self.leaf_object(child)?,
                    },
                }
            }
            (None, None) => LeafSource::Uncached(self.leaf_object(child)?),
        };
        Ok(PendingLeafRead(source))
    }

    /// Check and decode a leaf read begun by [`Self::start_leaf_read`], on
    /// the calling task, reusing whatever its object fetch already received.
    // Boxed to keep decode state out of every caller's future, which the
    // workspace denies through clippy::large_futures.
    fn finish_leaf_read<'a>(
        &'a self,
        child: &'a pb::FragmentTreeChild,
        read: PendingLeafRead,
    ) -> BoxFuture<'a, Result<LeafRead>> {
        async move {
            match read.0 {
                LeafSource::Cached(leaf) => Ok(LeafRead::Cached(leaf)),
                LeafSource::Retained {
                    reads,
                    cell,
                    object,
                } => reads
                    .read(&cell, Box::pin(self.decode_leaf(child, object)))
                    .await
                    .map(LeafRead::Owned),
                LeafSource::Missed {
                    leaves,
                    key,
                    object,
                } => {
                    let fragments = self.decode_leaf(child, object).await?;
                    let sighting = Sighting(&key);
                    if leaves.get_with_key(&sighting).await.is_none() {
                        leaves.insert_with_key(&sighting, Arc::new(Sighted)).await;
                        return Ok(LeafRead::Owned(fragments));
                    }
                    let leaf = Arc::new(CheckedLeaf::new(fragments));
                    leaves.insert_with_key(&key, leaf.clone()).await;
                    Ok(LeafRead::Cached(leaf))
                }
                LeafSource::Uncached(object) => {
                    self.decode_leaf(child, object).await.map(LeafRead::Owned)
                }
            }
        }
        .boxed()
    }

    pub(super) async fn finish_leaf_read_in_range(
        &self,
        child: &pb::FragmentTreeChild,
        end: u64,
        read: PendingLeafRead,
    ) -> Result<LeafRead> {
        let fragments = self.finish_leaf_read(child, read).await?;
        super::validation::leaf_range(
            child,
            fragments.fragments().last().map(|fragment| fragment.id),
            end,
        )?;
        Ok(fragments)
    }

    /// Read an interior for a traversal that does not modify it, sharing one
    /// checked copy through the cache when one is set.
    pub(super) async fn read_interior_shared(
        &self,
        child: &pb::FragmentTreeChild,
        end: u64,
    ) -> Result<Arc<CheckedInterior>> {
        let Some(cache) = &self.cache else {
            let interior = CheckedInterior::new(self.read_interior(child).await?);
            super::validation::interior_range(
                child,
                &interior.node.children,
                interior.buffered_targets.as_ref(),
                end,
            )?;
            return Ok(Arc::new(interior));
        };
        // Coalescing misses through `get_or_insert_with_key` would replace a
        // failed load's error with `Error::Cloned`. A duplicate decode under a
        // concurrent miss yields the same value.
        let key = InteriorKey(LeafKey::new(self, child)?);
        let interior = match cache.get_with_key(&key).await {
            Some(interior) => interior,
            None => {
                let interior = Arc::new(CheckedInterior::new(self.read_interior(child).await?));
                cache.insert_with_key(&key, interior.clone()).await;
                interior
            }
        };
        super::validation::interior_range(
            child,
            &interior.node.children,
            interior.buffered_targets.as_ref(),
            end,
        )?;
        if interior.newest_action_sequence >= self.root_frontier {
            return Err(super::validation::corrupt(format!(
                "Interior {} names action sequence {}, at or above the frontier {}",
                child.path, interior.newest_action_sequence, self.root_frontier
            )));
        }
        Ok(interior)
    }

    /// Write an interior, its children and buffer, as a protobuf object.
    pub(super) async fn write_interior(
        &self,
        children: Vec<pb::FragmentTreeChild>,
        buffer: Vec<pb::FragmentTreeMutation>,
    ) -> Result<Written> {
        let path = self.node_path();
        let node = pb::FragmentTreeNode { children, buffer };
        let io_bytes = node.encoded_len() as u64;
        if io_bytes > self.hard_capacity_bytes {
            return Err(Error::invalid_input(format!(
                "Interior requires {io_bytes} bytes, exceeding hard_capacity_bytes={}",
                self.hard_capacity_bytes
            )));
        }
        let bytes = node.encode_to_vec();
        self.object_store
            .inner
            .put_opts(
                &self.resolve_path(path.as_ref())?,
                PutPayload::from(bytes),
                PutOptions::default(),
            )
            .await?;
        Ok(Written {
            child_ref: node::internal_ref(
                path.to_string(),
                &node.children,
                &node.buffer,
                io_bytes,
            )?,
            io_bytes,
        })
    }

    /// Read an interior.
    async fn read_interior(&self, child: &pb::FragmentTreeChild) -> Result<InteriorNode> {
        let (store, path) = self.resolve_child(child)?;
        let bytes = store
            .inner
            .get_opts(&path, GetOptions::default())
            .await?
            .bytes()
            .await?;
        let object_size = bytes.len() as u64;
        if object_size != child.object_size {
            return Err(Error::corrupt_file(
                path,
                format!(
                    "Interior {} has {object_size} bytes but its parent declares {}",
                    child.path, child.object_size,
                ),
            ));
        }
        let mut node = pb::FragmentTreeNode::decode(bytes)
            .map_err(|error| Error::corrupt_file(path, error.to_string()))?;
        super::validation::children(&node.children, self.root_frontier, child.min_key)
            .map_err(|error| self.name_child(child, error))?;
        super::validation::buffer(&node.buffer, 1, self.root_frontier)
            .map_err(|error| self.name_child(child, error))?;
        let mut summary = node::internal_ref(
            child.path.clone(),
            &node.children,
            &node.buffer,
            object_size,
        )
        .map_err(|error| self.name_child(child, error))?;
        // The base comes from the parent. The object derives every other field.
        summary.base_id = child.base_id;
        if summary != *child {
            return Err(super::validation::corrupt(format!(
                "Interior {} contents differ from its parent reference",
                child.path
            )));
        }
        if let Some(base_id) = child.base_id {
            inherit_refs(&mut node.children, &mut node.buffer, base_id)?;
        }
        Ok(InteriorNode {
            children: node.children,
            buffer: node.buffer,
        })
    }

    pub(super) async fn read_interior_in_range(
        &self,
        child: &pb::FragmentTreeChild,
        end: u64,
    ) -> Result<InteriorNode> {
        let node = self.read_interior(child).await?;
        super::validation::interior_range(
            child,
            &node.children,
            buffered_targets(&node.buffer).as_ref(),
            end,
        )?;
        Ok(node)
    }

    /// Write an immutable root base without publishing a dataset version.
    /// The caller must publish its reference through the Version Manifest.
    pub(super) async fn write_root_base(
        &self,
        root: &pb::FragmentTreeRoot,
    ) -> Result<(Vec<u8>, u64)> {
        let uuid = Uuid::new_v4();
        let path = root_path(uuid.as_bytes())?;
        let bytes = root.encode_to_vec();
        let size = bytes.len() as u64;
        self.object_store
            .put(&self.resolve_path(path.as_ref())?, &bytes)
            .await?;
        Ok((uuid.as_bytes().to_vec(), size))
    }

    /// Read a base named directly by a Version Manifest, without version lookup.
    pub(super) async fn read_root_base(&self, uuid: &[u8]) -> Result<pb::FragmentTreeRoot> {
        let path = self.resolve_path(&root_path(uuid)?)?;
        let bytes = get_whole(&self.object_store, &path).await?;
        pb::FragmentTreeRoot::decode(bytes.as_ref())
            .map_err(|error| Error::corrupt_file(path, error.to_string()))
    }

    /// A leaf's stored records in id order with the actions routed to it applied.
    pub(super) fn replay(
        &self,
        stored: Vec<Fragment>,
        actions: Vec<pb::FragmentTreeMutation>,
    ) -> Result<Vec<Fragment>> {
        if actions.is_empty() {
            return Ok(stored);
        }
        let mut fragments: BTreeMap<u64, Fragment> = stored
            .into_iter()
            .map(|fragment| (fragment.id, fragment))
            .collect();
        self.apply_verified(&mut fragments, actions)?;
        Ok(fragments.into_values().collect())
    }

    pub(super) fn apply_verified(
        &self,
        fragments: &mut BTreeMap<u64, Fragment>,
        actions: Vec<pb::FragmentTreeMutation>,
    ) -> Result<()> {
        if actions.is_empty() {
            return Ok(());
        }
        let new_file_keys: BTreeSet<u64> = actions
            .iter()
            .filter(|tagged| adds_data_files(tagged))
            .map(content::action_key)
            .collect();
        crate::fragment_metadata::content::apply_verified(fragments, actions)?;
        if new_file_keys.is_empty() {
            return Ok(());
        }
        let mut intern = self.lock_interner();
        for id in new_file_keys {
            if let Some(fragment) = fragments.get_mut(&id) {
                intern_data_file_lists(&mut intern, fragment);
            }
        }
        Ok(())
    }

    pub(super) fn share_interner(&mut self, source: &Self) {
        self.interner = source.interner.clone();
    }

    pub(super) fn share_data_file_lists<'a>(
        &self,
        fragments: impl IntoIterator<Item = &'a mut Fragment>,
    ) {
        let mut intern = self.lock_interner();
        for fragment in fragments {
            intern_data_file_lists(&mut intern, fragment);
        }
    }

    /// One fragment of a leaf. Every check [`Self::read_leaf_shared`] makes against
    /// the parent reference still holds, but an uncached point read decodes and
    /// validates only the returned record in full. Point reads use cached leaves
    /// without admitting them; corruption confined to other records surfaces
    /// when the leaf is read whole.
    pub(super) async fn read_leaf_fragment(
        &self,
        child: &pb::FragmentTreeChild,
        fragment_id: u64,
        end: u64,
    ) -> Result<Option<Fragment>> {
        if self.validation_reads.is_some() {
            // A bulk commit retains whole leaves for its materializer.
            return Ok(self
                .read_leaf_shared(child, end)
                .await?
                .into_vec()
                .into_iter()
                .find(|fragment| fragment.id == fragment_id));
        }
        if let Some(leaves) = &self.cache
            && let Some(leaf) = leaves.get_with_key(&LeafKey::new(self, child)?).await
        {
            super::validation::leaf_range(
                child,
                leaf.fragments.last().map(|fragment| fragment.id),
                end,
            )?;
            let row = leaf
                .fragments
                .binary_search_by_key(&fragment_id, |fragment| fragment.id);
            return Ok(row.ok().map(|row| leaf.fragments[row].clone()));
        }
        self.read_leaf_record(child, fragment_id, end).await
    }

    /// One fragment of a leaf read from storage, with its headers checked
    /// against the parent reference and only that record decoded in full.
    async fn read_leaf_record(
        &self,
        child: &pb::FragmentTreeChild,
        fragment_id: u64,
        end: u64,
    ) -> Result<Option<Fragment>> {
        let reader = self.open_leaf(child, self.leaf_object(child)?).await?;
        let Ok(row) = self
            .headers_of(child, &reader, end, |header| header.id)
            .await?
            .binary_search(&fragment_id)
        else {
            return Ok(None);
        };
        let batch = reader
            .read_stream(
                ReadBatchParams::Range(row..row + 1),
                1,
                1,
                FilterExpression::no_filter(),
            )
            .await?
            .try_next()
            .await?;
        let mut fragments = match &batch {
            Some(batch) => super::leaf::decode(batch, || self.lock_interner())
                .map_err(|error| self.name_child(child, error))?,
            None => Vec::new(),
        };
        let mut fragment = match (fragments.pop(), fragments.is_empty()) {
            (Some(fragment), true) if fragment.id == fragment_id => fragment,
            _ => {
                return Err(super::validation::corrupt(format!(
                    "Leaf {} row {row} does not decode to fragment {fragment_id}",
                    child.path
                )));
            }
        };
        if let Some(base_id) = child.base_id {
            stamp_unset_base_id(&mut fragment, base_id);
        }
        Ok(Some(fragment))
    }

    /// A leaf's records without their files and overlays, checked against the
    /// parent reference as a whole read is. They carry every count a buffered
    /// action changes.
    pub(super) async fn read_leaf_headers(
        &self,
        child: &pb::FragmentTreeChild,
        end: u64,
    ) -> Result<Vec<Fragment>> {
        if self.validation_reads.is_some() {
            // A bulk commit retains whole leaves for its materializer.
            return Ok(self.read_leaf_shared(child, end).await?.into_vec());
        }
        let reader = self.open_leaf(child, self.leaf_object(child)?).await?;
        self.headers_of(child, &reader, end, |header| header).await
    }

    /// Read the leaf headers, validate them against the child reference, and
    /// project each record with `keep`.
    async fn headers_of<T>(
        &self,
        child: &pb::FragmentTreeChild,
        reader: &FileReader,
        end: u64,
        keep: impl Fn(Fragment) -> T,
    ) -> Result<Vec<T>> {
        if reader.num_rows() != child.num_keys {
            return Err(super::validation::corrupt(format!(
                "Leaf {} has {} rows but its parent declares num_keys={}",
                child.path,
                reader.num_rows(),
                child.num_keys
            )));
        }
        let projection = lance_file::versions::reader_projection_from_column_names(
            reader.version(),
            reader.schema(),
            &["id", "fragment_meta"],
        )?;
        let mut headers = super::leaf::LeafHeaders::with_capacity(
            child.num_keys.min(READ_BATCH_ROWS as u64) as usize,
        );
        let mut stream = reader
            .read_stream_projected(
                ReadBatchParams::RangeFull,
                READ_BATCH_ROWS,
                READ_BATCH_READAHEAD,
                projection,
                FilterExpression::no_filter(),
            )
            .await?;
        while let Some(batch) = stream.try_next().await? {
            headers
                .extend(&batch, &keep)
                .map_err(|error| self.name_child(child, error))?;
        }
        let first = headers.first_id;
        if headers.kept.len() as u64 != child.num_keys
            || first.is_some_and(|first| child.min_key > first)
            || headers.total_rows != child.total_rows
            || headers.visible_rows != child.visible_rows
        {
            return Err(super::validation::corrupt(format!(
                "Leaf {} contents differ from its parent reference: keys={}, first={first:?}, \
                 total_rows={}, visible_rows={}; the parent declares num_keys={}, min_key={}, \
                 total_rows={}, visible_rows={}",
                child.path,
                headers.kept.len(),
                headers.total_rows,
                headers.visible_rows,
                child.num_keys,
                child.min_key,
                child.total_rows,
                child.visible_rows
            )));
        }
        super::validation::leaf_range(child, headers.last_id, end)?;
        Ok(headers.kept)
    }

    /// The stored object of `child`, not yet requested.
    fn leaf_object(&self, child: &pb::FragmentTreeChild) -> Result<LeafObject> {
        let (store, path) = self.resolve_child(child)?;
        let object_size = usize::try_from(child.object_size).map_err(|_| {
            super::validation::corrupt(format!(
                "leaf object_size does not fit usize: path={}, object_size={}",
                child.path, child.object_size
            ))
        })?;
        Ok(LeafObject(SmallReader::new(
            store.inner.clone(),
            path,
            store.download_retry_count(),
            object_size,
        )))
    }

    /// Open a leaf after checking its stored size against the parent reference.
    async fn open_leaf(
        &self,
        child: &pb::FragmentTreeChild,
        object: LeafObject,
    ) -> Result<FileReader> {
        let object_reader = Arc::new(object.0);
        let file_scheduler = self.scheduler.open_reader(object_reader.clone());
        let opened = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &Arc::new(LanceCache::no_cache()),
            FileReaderOptions::default(),
        )
        .await;

        // Opening fetched the complete SmallReader object through the
        // scheduler, so its size is read from those bytes, not a HEAD or a
        // second GET. A shorter object fails the footer read as an IO error,
        // so the size is checked before that error is returned.
        if let Ok(bytes) = object_reader.get_all().await
            && bytes.len() as u64 != child.object_size
        {
            return Err(super::validation::corrupt(format!(
                "Leaf {} has {} bytes but its parent declares {}",
                child.path,
                bytes.len(),
                child.object_size,
            )));
        }
        opened
    }

    /// Read a leaf from storage and check it against its parent reference,
    /// bypassing every cache.
    #[cfg(test)]
    pub(super) async fn read_leaf_uncached(
        &self,
        child: &pb::FragmentTreeChild,
    ) -> Result<Vec<Fragment>> {
        self.decode_leaf(child, self.leaf_object(child)?).await
    }

    /// Decode and check a leaf from its object, fetching it unless an
    /// earlier request already has.
    async fn decode_leaf(
        &self,
        child: &pb::FragmentTreeChild,
        object: LeafObject,
    ) -> Result<Vec<Fragment>> {
        let reader = self.open_leaf(child, object).await?;

        // A reference is untrusted until the leaf contents have been checked.
        let mut fragments: Vec<Fragment> =
            Vec::with_capacity(child.num_keys.min(READ_BATCH_ROWS as u64) as usize);
        let mut stream = reader
            .read_stream(
                ReadBatchParams::RangeFull,
                READ_BATCH_ROWS,
                READ_BATCH_READAHEAD,
                FilterExpression::no_filter(),
            )
            .await?;
        while let Some(batch) = stream.try_next().await? {
            let decoded = super::leaf::decode(&batch, || self.lock_interner())
                .map_err(|error| self.name_child(child, error))?;
            for fragment in decoded {
                if fragments
                    .last()
                    .is_some_and(|previous| previous.id >= fragment.id)
                {
                    return Err(super::validation::corrupt(format!(
                        "Leaf {} fragment IDs are out of order across batches",
                        child.path
                    )));
                }
                fragments.push(fragment);
            }
        }
        let summary = node::leaf_ref(
            child.path.clone(),
            &fragments,
            child.object_size,
            child.materialized_through_action_sequence,
        )?;
        // The fence comes from the parent and may sit below the first stored key.
        if child.min_key > summary.min_key {
            return Err(super::validation::corrupt(format!(
                "Leaf {} fence {} is above its first fragment {}",
                child.path, child.min_key, summary.min_key
            )));
        }
        if summary.num_keys != child.num_keys
            || summary.total_rows != child.total_rows
            || summary.visible_rows != child.visible_rows
        {
            return Err(super::validation::corrupt(format!(
                "Leaf {} contents differ from its parent reference",
                child.path
            )));
        }
        if let Some(base_id) = child.base_id {
            for fragment in &mut fragments {
                stamp_unset_base_id(fragment, base_id);
            }
        }
        Ok(fragments)
    }

    fn leaf_path(&self) -> Path {
        Path::from(format!("{TREE_DIR}/leaf/{}.lance", Uuid::new_v4()))
    }

    fn node_path(&self) -> Path {
        Path::from(format!("{TREE_DIR}/node/{}.node", Uuid::new_v4()))
    }

    fn resolve_path(&self, path: &str) -> Result<Path> {
        self.resolve_under(&self.base, path)
    }

    fn resolve_child(&self, child: &pb::FragmentTreeChild) -> Result<(Arc<ObjectStore>, Path)> {
        let (store, base) = self.child_dataset(child)?;
        Ok((store, self.resolve_under(&base, &child.path)?))
    }

    /// Batch decoders and list validators hold no object path, so the reader
    /// that fetched `child` names it.
    fn name_child(&self, child: &pb::FragmentTreeChild, mut error: Error) -> Error {
        if let Error::CorruptFile { path, .. } = &mut error
            && let Ok((_, object)) = self.resolve_child(child)
        {
            *path = object;
        }
        error
    }

    pub(super) fn child_location(&self, child: &pb::FragmentTreeChild) -> Result<NodeLocation> {
        let (store, base) = self.child_dataset(child)?;
        self.location_under(&store, &base, &child.path)
    }

    /// Where `path` resolves when it belongs to the dataset at `base` on
    /// `store`, the identity every ownership decision compares against.
    pub(super) fn location_under(
        &self,
        store: &Arc<ObjectStore>,
        base: &Path,
        path: &str,
    ) -> Result<NodeLocation> {
        Ok(NodeLocation {
            store_prefix: store.store_prefix.clone(),
            memory_store: (store.scheme() == "memory").then(|| {
                let id = memory_store_id(&store.inner);
                self.memory_store_owners
                    .get(&id)
                    .map_or(id, memory_store_id)
            }),
            path: self.resolve_under(base, path)?,
        })
    }

    pub(super) fn base(&self) -> &Path {
        &self.base
    }

    fn resolve_under(&self, base: &Path, path: &str) -> Result<Path> {
        let relative = Path::parse(path).map_err(|error| {
            super::validation::corrupt(format!("Invalid metadata path {path:?}: {error}"))
        })?;
        if !path.starts_with(&format!("{TREE_DIR}/")) || path != relative.as_ref() {
            return Err(super::validation::corrupt(format!(
                "Metadata path {path:?} must be relative to {TREE_DIR}/"
            )));
        }
        let mut resolved = base.clone();
        resolved.extend(&relative);
        Ok(resolved)
    }

    fn child_dataset(&self, child: &pb::FragmentTreeChild) -> Result<(Arc<ObjectStore>, Path)> {
        match child.base_id {
            Some(id) => {
                let foreign = self.foreign_bases.get(&id).ok_or_else(|| {
                    super::validation::corrupt(format!(
                        "Tree child {} names unknown base_id {id}",
                        child.path
                    ))
                })?;
                Ok((foreign.store.clone(), foreign.base.clone()))
            }
            None => Ok((self.object_store.clone(), self.base.clone())),
        }
    }

    fn lock_interner(&self) -> std::sync::MutexGuard<'_, DataFileFieldInterner> {
        self.interner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

// Use the object-store trait so wrappers observe metadata reads on local storage too.
async fn get_whole(object_store: &ObjectStore, path: &Path) -> Result<bytes::Bytes> {
    let result = object_store
        .inner
        .get_opts(path, object_store::GetOptions::default())
        .await?;
    Ok(result.bytes().await?)
}

/// Only UpsertFragment and AddDataFile can introduce field-id or column-index
/// lists that need interning. Leaf decode already interns every file it reads.
fn adds_data_files(tagged: &pb::FragmentTreeMutation) -> bool {
    matches!(
        tagged
            .action
            .as_ref()
            .and_then(|action| action.action.as_ref()),
        Some(Action::UpsertFragment(_) | Action::AddDataFile(_))
    )
}

fn intern_data_file_lists(intern: &mut DataFileFieldInterner, fragment: &mut Fragment) {
    for file in fragment.referenced_lance_files_mut() {
        file.fields = intern.intern_field_ids(file.fields.as_ref());
        file.column_indices = intern.intern_column_indices(file.column_indices.as_ref());
    }
}

/// Resolve the Version Manifest's root UUID within its dataset.
pub fn root_path(bytes: &[u8]) -> Result<String> {
    let uuid = Uuid::from_slice(bytes).map_err(|_| {
        super::validation::corrupt(format!("root_uuid has {} bytes; expected 16", bytes.len()))
    })?;
    Ok(format!("{TREE_DIR}/root/{uuid}.root"))
}

pub(super) fn stamp_unset_base_id(fragment: &mut Fragment, base_id: u32) {
    for file in fragment.referenced_lance_files_mut() {
        if file.base_id.is_none() {
            file.base_id = Some(base_id);
        }
    }
    if let Some(deletion) = &mut fragment.deletion_file
        && deletion.base_id.is_none()
    {
        deletion.base_id = Some(base_id);
    }
}

/// Stamp `base_id` on every child and file reference that has none, row
/// lineage column carriers included, so a node read or copied out of its
/// owning dataset still resolves there.
pub(super) fn inherit_refs(
    children: &mut [pb::FragmentTreeChild],
    buffer: &mut [pb::FragmentTreeMutation],
    base_id: u32,
) -> Result<()> {
    for child in children {
        child.base_id.get_or_insert(base_id);
    }
    for tagged in buffer {
        let Some(action) = tagged
            .action
            .as_mut()
            .and_then(|action| action.action.as_mut())
        else {
            continue;
        };
        match action {
            Action::UpsertFragment(encoded) => {
                let mut fragment = Fragment::try_from(encoded.clone())?;
                stamp_unset_base_id(&mut fragment, base_id);
                *encoded = pb::DataFragment::from(&fragment);
            }
            Action::AddDataFile(value) => {
                if let Some(file) = value.file.as_mut()
                    && file.base_id.is_none()
                {
                    file.base_id = Some(base_id);
                }
            }
            Action::AddDeletionFile(value) => {
                if let Some(file) = value.deletion_file.as_mut()
                    && file.base_id.is_none()
                {
                    file.base_id = Some(base_id);
                }
            }
            Action::ReplaceDataFile(value) if value.base_id.is_none() => {
                value.base_id = Some(base_id);
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
