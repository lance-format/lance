// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Immutable protobuf routing nodes and columnar Lance leaves.
//!
//! Leaves hold one row per fragment with ordered files and overlays.
//! Object paths are relative to the dataset root; every rewrite gets a new UUID.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::{Range, RangeInclusive};
use std::sync::{Arc, Mutex};

use futures::{FutureExt, TryStreamExt, future::BoxFuture};
use prost::Message;

use crate::format::pb;
use crate::format::pb::fragment_action::Action;
use crate::format::{DataFileFieldInterner, Fragment};
use crate::fragment_metadata::node::{self, InternalNode};
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache};
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::version::ConcreteFileVersion;
use lance_file::writer::FileWriterOptions;
use lance_io::ReadBatchParams;
use lance_io::object_reader::SmallReader;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::ScanScheduler;
use lance_io::traits::Reader;
use object_store::path::Path;
use object_store::{GetOptions, ObjectStore as OSObjectStore, PutOptions, PutPayload};
use tokio::sync::Semaphore;
use tokio::task::{JoinError, JoinSet};
use uuid::Uuid;
mod validation_reads;

// Decode each leaf in batches of READ_BATCH_ROWS, with up to
// READ_BATCH_READAHEAD decoder batches in flight for that leaf. A scan may
// keep several leaves in flight separately.
const READ_BATCH_ROWS: u32 = 16 * 1024;
const READ_BATCH_READAHEAD: u32 = 16;

/// A written node: its parent child reference plus the actual bytes written to
/// storage (for write-amplification accounting).
#[derive(Debug)]
pub struct Written {
    pub child_ref: pb::FragmentTreeChild,
    pub io_bytes: u64,
}

struct EncodedLeaf {
    range: Range<usize>,
    bytes: bytes::Bytes,
}

/// Leaf references in key order, each backed by a durable object.
pub(super) struct Uploaded {
    pub(super) children: Vec<pb::FragmentTreeChild>,
    pub(super) io_bytes: u64,
}

impl Uploaded {
    /// Valid only when no child was reused, so each object size is the bytes
    /// this upload wrote.
    fn into_written(self) -> Vec<Written> {
        self.children
            .into_iter()
            .map(|child_ref| Written {
                io_bytes: child_ref.object_size,
                child_ref,
            })
            .collect()
    }
}

/// Leaf PUTs that run while the caller encodes the next leaf. A reference
/// depends only on the fragments and the encoded size, so it is built before
/// its PUT, but it leaves only through [`Self::finish`], once every PUT is
/// durable, so no parent can name a missing object. Dropping the queue aborts
/// PUTs still running. Any that landed are unreachable garbage.
pub(super) struct LeafUploads<'a> {
    store: &'a NodeStore,
    puts: JoinSet<Result<()>>,
    children: Vec<pb::FragmentTreeChild>,
    io_bytes: u64,
}

impl<'a> LeafUploads<'a> {
    pub(super) fn new(store: &'a NodeStore) -> Self {
        Self {
            store,
            puts: JoinSet::new(),
            children: Vec::new(),
            io_bytes: 0,
        }
    }

    /// Start the PUT of one encoded leaf once the store's window has room.
    pub(super) async fn upload(
        &mut self,
        fragments: &[Fragment],
        bytes: bytes::Bytes,
        materialized_through_action_sequence: u64,
    ) -> Result<()> {
        let permit = self
            .store
            .leaf_puts
            .clone()
            .acquire_owned()
            .await
            .map_err(|error| {
                Error::internal(format!("fragment tree leaf PUT window closed: {error}"))
            })?;
        // Stop producing at the first failed PUT instead of at finish.
        while let Some(put) = self.puts.try_join_next() {
            put_outcome(put)?;
        }
        let path = self.store.leaf_path();
        let size = bytes.len() as u64;
        let location = self.store.resolve_path(path.as_ref())?;
        self.children.push(node::leaf_ref(
            path.to_string(),
            fragments,
            size,
            materialized_through_action_sequence,
        )?);
        self.io_bytes += size;
        let inner = self.store.object_store.inner.clone();
        // A spawned task keeps the PUT moving while the caller encodes the
        // next leaf, which an unpolled future would not.
        self.puts.spawn(async move {
            let _permit = permit;
            inner
                .put_opts(&location, bytes.into(), PutOptions::default())
                .await?;
            Ok(())
        });
        Ok(())
    }

    /// Encode `fragments` as one or more leaves, split only when actual Lance
    /// bytes exceed the target, and upload each. Returns the leaf count.
    pub(super) async fn upload_leaves(
        &mut self,
        fragments: &[Fragment],
        bytes: Option<bytes::Bytes>,
        watermark: u64,
        config: &node::FragmentTreeConfig,
    ) -> Result<usize> {
        let store = self.store;
        let encoded = store.encoded_leaves(fragments, 0..fragments.len(), bytes, config);
        futures::pin_mut!(encoded);
        let mut leaves = 0;
        while let Some(leaf) = encoded.try_next().await? {
            self.upload(&fragments[leaf.range], leaf.bytes, watermark)
                .await?;
            leaves += 1;
        }
        Ok(leaves)
    }

    /// Keep an existing leaf in its key position.
    pub(super) fn reuse(&mut self, child: pb::FragmentTreeChild) {
        self.children.push(child);
    }

    pub(super) async fn finish(mut self) -> Result<Uploaded> {
        while let Some(put) = self.puts.join_next().await {
            put_outcome(put)?;
        }
        Ok(Uploaded {
            children: self.children,
            io_bytes: self.io_bytes,
        })
    }
}

fn put_outcome(put: std::result::Result<Result<()>, JoinError>) -> Result<()> {
    match put {
        Ok(outcome) => outcome,
        Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
        Err(error) => Err(Error::internal(format!(
            "fragment tree leaf PUT task did not complete: {error}"
        ))),
    }
}

/// Reads and writes fragment metadata tree node files against an object store.
#[derive(Clone)]
pub struct NodeStore {
    pub(super) object_store: Arc<ObjectStore>,
    base: Path,
    scheduler: Arc<ScanScheduler>,
    caches: ReadCaches,
    validation_reads: Option<Arc<validation_reads::ValidationReads>>,
    /// Shared by clones, so concurrent drains that each write leaves still
    /// hold at most this many encoded leaves in PUTs at once.
    leaf_puts: Arc<Semaphore>,
    pub(super) next_action_sequence: u64,
    pub(super) hard_capacity_bytes: u64,
    foreign_bases: HashMap<u32, ForeignBase>,
    interner: Arc<Mutex<DataFileFieldInterner>>,
    /// Records passed to `encode_leaf`, so tests can bound encode work.
    #[cfg(test)]
    encoded_records: Arc<std::sync::atomic::AtomicUsize>,
}

/// Caches reads go through. Neither changes what a read returns.
#[derive(Clone)]
struct ReadCaches {
    /// File metadata of opened leaves.
    files: Arc<LanceCache>,
    /// Checked leaves and interiors, shared by every tree that holds the same cache.
    leaves: Option<LanceCache>,
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

/// A leaf's location and the parent reference its contents were checked
/// against. Any change to the reference, including its inherited base, forms
/// a different key, so the leaf is checked again.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
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
    pub(super) node: InternalNode,
    /// The newest action sequence the node or its leaf watermarks name. Trees
    /// share cached interiors, so each reader checks this against its own frontier.
    newest_action_sequence: u64,
    /// The cache key omits where the inherited range ends, so each reference
    /// checks these bounds against its own end.
    buffered_targets: Option<RangeInclusive<u64>>,
    // Sized once, since sizing walks every buffered mutation.
    heap_bytes: usize,
}

impl CheckedInterior {
    fn new(mut node: InternalNode) -> Self {
        // Decoded `bytes` fields are slices of the fetched object, and a slice
        // keeps the whole object allocated. Copy them so the cache retains only
        // the bytes it weighs.
        for mutation in &mut node.buffer {
            if let Some(pb::FragmentAction {
                action: Some(Action::UpsertFragment(fragment)),
            }) = &mut mutation.action
                && let Some(pb::data_fragment::RowIdSequence::InlineRowIds(ids)) =
                    &mut fragment.row_id_sequence
            {
                *ids = bytes::Bytes::copy_from_slice(ids);
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
        let mut context = Context::new();
        let heap_bytes = node.children.deep_size_of_children(&mut context)
            + node.buffer.deep_size_of_children(&mut context);
        Self {
            node,
            newest_action_sequence,
            buffered_targets,
            heap_bytes,
        }
    }
}

/// The lowest through the highest fragment a buffer targets.
fn buffered_targets(buffer: &[pb::FragmentTreeMutation]) -> Option<RangeInclusive<u64>> {
    buffer
        .iter()
        .map(node::action_key)
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

fn leaf_put_window(object_store: &ObjectStore) -> Arc<Semaphore> {
    Arc::new(Semaphore::new(
        object_store
            .io_parallelism()
            .clamp(1, super::tree::MAX_CONCURRENT_LEAF_DRAINS),
    ))
}

/// A checked leaf, shared with the leaf cache when the read found or admitted
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

    pub(super) fn select(self, keep: impl Fn(&Fragment) -> bool) -> Vec<Fragment> {
        match self {
            Self::Cached(leaf) => leaf
                .fragments
                .iter()
                .filter(|fragment| keep(fragment))
                .cloned()
                .collect(),
            Self::Owned(fragments) => fragments
                .into_iter()
                .filter(|fragment| keep(fragment))
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
    pub fn new(
        object_store: Arc<ObjectStore>,
        base: Path,
        scheduler: Arc<ScanScheduler>,
        cache: Arc<LanceCache>,
    ) -> Self {
        Self {
            leaf_puts: leaf_put_window(&object_store),
            object_store,
            base,
            scheduler,
            caches: ReadCaches {
                files: cache,
                leaves: None,
            },
            validation_reads: None,
            next_action_sequence: u64::MAX,
            hard_capacity_bytes: u64::MAX,
            foreign_bases: HashMap::new(),
            interner: Arc::new(Mutex::new(DataFileFieldInterner::default())),
            #[cfg(test)]
            encoded_records: Arc::default(),
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

    pub(super) fn set_leaf_cache(&mut self, leaves: LanceCache) {
        self.caches.leaves = Some(leaves);
    }

    pub(super) fn rebind(&mut self, object_store: Arc<ObjectStore>) {
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

    pub(super) fn clear_validation_reads(&mut self) {
        self.validation_reads = None;
    }

    /// Write sorted fragments as a columnar Lance leaf, one row per fragment.
    pub async fn write_leaf(
        &self,
        fragments: &[Fragment],
        materialized_through_action_sequence: u64,
    ) -> Result<Written> {
        let bytes = self.encode_leaf(fragments).await?;
        let mut uploads = LeafUploads::new(self);
        uploads
            .upload(fragments, bytes, materialized_through_action_sequence)
            .await?;
        Ok(uploads
            .finish()
            .await?
            .into_written()
            .pop()
            .expect("one upload yields one leaf"))
    }

    /// Encode before publishing, splitting only when actual Lance bytes exceed
    /// the target. A single fragment may exceed the target, but never the hard
    /// object limit. Rejected encodings create no remote intermediate objects.
    pub(super) async fn write_leaves(
        &self,
        fragments: &[Fragment],
        bytes: Option<bytes::Bytes>,
        watermark: u64,
        config: &node::FragmentTreeConfig,
    ) -> Result<Vec<Written>> {
        let mut uploads = LeafUploads::new(self);
        uploads
            .upload_leaves(fragments, bytes, watermark, config)
            .await?;
        Ok(uploads.finish().await?.into_written())
    }

    /// Pack adjacent bootstrap batches when their combined encoding fits. Keep
    /// at most three targets of logical metadata in a candidate, except for an
    /// indivisible fragment. Normal mutation splits retain their own headroom.
    pub(super) async fn write_initial_leaves(
        &self,
        fragments: &[Fragment],
        config: &node::FragmentTreeConfig,
    ) -> Result<Vec<Written>> {
        let mut pending: Option<EncodedLeaf> = None;
        let mut uploads = LeafUploads::new(self);
        let mut start = 0;
        while start < fragments.len() {
            let mut end = start;
            let mut logical_bytes = 0;
            while end < fragments.len() && logical_bytes < config.max_leaf_bytes {
                logical_bytes += node::fragment_logical_bytes(&fragments[end]);
                end += 1;
            }
            let encoded = self.encoded_leaves(fragments, start..end, None, config);
            futures::pin_mut!(encoded);
            while let Some(leaf) = encoded.try_next().await? {
                if let Some(previous) = pending.take() {
                    let encoded_bytes = previous.bytes.len().checked_add(leaf.bytes.len());
                    let range = previous.range.start..leaf.range.end;
                    if encoded_bytes.is_some_and(|bytes| bytes as u64 <= config.max_leaf_bytes)
                        && node::leaf_logical_bytes(&fragments[range.clone()])
                            <= config.max_leaf_bytes * 3
                    {
                        let bytes = self.encode_leaf(&fragments[range.clone()]).await?;
                        // Compression is not additive. The estimate only chooses
                        // a candidate; its actual encoding decides admission.
                        if bytes.len() as u64 <= config.max_leaf_bytes {
                            pending = Some(EncodedLeaf { range, bytes });
                            continue;
                        }
                    }
                    uploads
                        .upload(&fragments[previous.range], previous.bytes, 0)
                        .await?;
                }
                pending = Some(leaf);
            }
            start = end;
        }
        if let Some(leaf) = pending {
            uploads
                .upload(&fragments[leaf.range], leaf.bytes, 0)
                .await?;
        }
        Ok(uploads.finish().await?.into_written())
    }

    /// Read a columnar leaf into fragment records.
    pub fn read_leaf<'a>(
        &'a self,
        child: &'a pb::FragmentTreeChild,
    ) -> BoxFuture<'a, Result<Vec<Fragment>>> {
        async move {
            let read = self.start_leaf_read(child).await?;
            Ok(self.finish_leaf_read(child, read).await?.into_vec())
        }
        .boxed()
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

    pub(super) async fn read_leaf_in_range(
        &self,
        child: &pb::FragmentTreeChild,
        end: u64,
    ) -> Result<Vec<Fragment>> {
        Ok(self.read_leaf_shared(child, end).await?.into_vec())
    }

    /// Decide where a whole read of `child` is served from, before any GET.
    pub(super) async fn start_leaf_read(
        &self,
        child: &pb::FragmentTreeChild,
    ) -> Result<PendingLeafRead> {
        let source = match (&self.validation_reads, &self.caches.leaves) {
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
                    .read(&cell, self.decode_leaf(child, object))
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
    /// checked copy through the leaf cache when one is set.
    pub(super) async fn read_internal_shared(
        &self,
        child: &pb::FragmentTreeChild,
        end: u64,
    ) -> Result<Arc<CheckedInterior>> {
        let Some(cache) = &self.caches.leaves else {
            let interior = CheckedInterior::new(self.read_internal(child).await?);
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
                let interior = Arc::new(CheckedInterior::new(self.read_internal(child).await?));
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
        if interior.newest_action_sequence >= self.next_action_sequence {
            return Err(super::validation::corrupt(format!(
                "Interior {} names action sequence {}, at or above the frontier {}",
                child.path, interior.newest_action_sequence, self.next_action_sequence
            )));
        }
        Ok(interior)
    }

    /// Write an internal node (children + buffer) as a protobuf object.
    pub async fn write_internal(
        &self,
        children: Vec<pb::FragmentTreeChild>,
        buffer: Vec<pb::FragmentTreeMutation>,
    ) -> Result<Written> {
        let path = self.node_path();
        let node = pb::FragmentTreeNode { children, buffer };
        let io_bytes = node.encoded_len() as u64;
        if io_bytes > self.hard_capacity_bytes {
            return Err(Error::invalid_input(format!(
                "Internal node requires {io_bytes} bytes, exceeding hard_capacity_bytes={}",
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

    /// Read an internal node.
    pub async fn read_internal(&self, child: &pb::FragmentTreeChild) -> Result<InternalNode> {
        let (store, path) = self.resolve_child(child)?;
        let bytes = store
            .inner
            .get_opts(&path, GetOptions::default())
            .await?
            .bytes()
            .await?;
        let object_size = bytes.len() as u64;
        if object_size != child.object_size {
            return Err(super::validation::corrupt(format!(
                "Internal node {} has {object_size} bytes but its parent declares {}",
                child.path, child.object_size,
            )));
        }
        let mut node = pb::FragmentTreeNode::decode(bytes)?;
        super::validation::children(&node.children, self.next_action_sequence, child.min_key)?;
        super::validation::buffer(&node.buffer, 1, self.next_action_sequence)?;
        let summary = node::internal_ref(
            child.path.clone(),
            &node.children,
            &node.buffer,
            object_size,
        )?;
        if !derived_child_fields_match(&summary, child) {
            return Err(super::validation::corrupt(format!(
                "Internal node {} contents differ from its parent reference",
                child.path
            )));
        }
        // Inherited nodes belong to another dataset. Unset child and file
        // refs, including hidden row lineage columns, take that dataset's base_id.
        if let Some(base_id) = child.base_id {
            for descendant in &mut node.children {
                if descendant.base_id.is_none() {
                    descendant.base_id = Some(base_id);
                }
            }
            preserve_inherited_refs(&mut node.buffer, base_id)?;
        }
        Ok(InternalNode {
            children: node.children,
            buffer: node.buffer,
        })
    }

    pub(super) async fn read_internal_in_range(
        &self,
        child: &pb::FragmentTreeChild,
        end: u64,
    ) -> Result<InternalNode> {
        let node = self.read_internal(child).await?;
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
    pub async fn write_root_base(&self, root: &pb::FragmentTreeRoot) -> Result<(Vec<u8>, u64)> {
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
    pub async fn read_root_base(&self, uuid: &[u8]) -> Result<pb::FragmentTreeRoot> {
        let path = root_path(uuid)?;
        let bytes = get_whole(&self.object_store, &self.resolve_path(&path)?).await?;
        Ok(pb::FragmentTreeRoot::decode(bytes.as_ref())?)
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
            .map(node::action_key)
            .collect();
        node::apply_verified(fragments, actions)?;
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

    pub(super) fn share_data_file_lists<'a>(
        &self,
        fragments: impl IntoIterator<Item = &'a mut Fragment>,
    ) {
        let mut intern = self.lock_interner();
        for fragment in fragments {
            intern_data_file_lists(&mut intern, fragment);
        }
    }

    pub(super) async fn encode_leaf(&self, fragments: &[Fragment]) -> Result<bytes::Bytes> {
        #[cfg(test)]
        self.encoded_records
            .fetch_add(fragments.len(), std::sync::atomic::Ordering::Relaxed);
        let batch = super::leaf::encode(fragments)?;
        let arrow_schema = batch.schema();

        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref())?;
        let memory = ObjectStore::memory();
        let path = Path::from("encoded-leaf");
        let writer = memory.create(&path).await?;
        let mut file_writer = lance_file::versions::create_writer(
            ConcreteFileVersion::V2_1,
            writer,
            lance_schema,
            FileWriterOptions::default(),
        )?;
        file_writer.write_batch(&batch).await?;
        file_writer.finish().await?;
        get_whole(&memory, &path).await
    }

    fn encoded_leaves<'a>(
        &'a self,
        fragments: &'a [Fragment],
        range: Range<usize>,
        bytes: Option<bytes::Bytes>,
        config: &'a node::FragmentTreeConfig,
    ) -> impl futures::Stream<Item = Result<EncodedLeaf>> + 'a {
        futures::stream::try_unfold(vec![(range, bytes)], move |mut pending| async move {
            while let Some((range, bytes)) = pending.pop() {
                if range.is_empty() {
                    continue;
                }
                let piece = &fragments[range.clone()];
                let bytes = match bytes {
                    Some(bytes) => bytes,
                    None => self.encode_leaf(piece).await?,
                };
                if bytes.len() as u64 > config.max_leaf_bytes && piece.len() > 1 {
                    let pieces =
                        split_ranges(fragments, range, bytes.len() as u64, config.max_leaf_bytes);
                    pending.extend(pieces.into_iter().rev().map(|piece| (piece, None)));
                    continue;
                }
                if bytes.len() as u64 > config.hard_capacity_bytes {
                    return Err(Error::invalid_input(format!(
                        "fragment {} requires an encoded leaf of {} bytes, exceeding hard_capacity_bytes={}",
                        piece[0].id,
                        bytes.len(),
                        config.hard_capacity_bytes
                    )));
                }
                return Ok(Some((EncodedLeaf { range, bytes }, pending)));
            }
            Ok(None)
        })
    }

    /// One fragment of a leaf. Every check [`Self::read_leaf`] makes against
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
                .read_leaf_in_range(child, end)
                .await?
                .into_iter()
                .find(|fragment| fragment.id == fragment_id));
        }
        if let Some(leaves) = &self.caches.leaves
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
            Some(batch) => super::leaf::decode(batch, || self.lock_interner())?,
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
            return self.read_leaf_in_range(child, end).await;
        }
        let reader = self.open_leaf(child, self.leaf_object(child)?).await?;
        self.headers_of(child, &reader, end, |header| header).await
    }

    /// What `keep` takes from each header-only record of an open leaf, after
    /// the records are checked against every parent field the leaf determines.
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
            headers.extend(&batch, &keep)?;
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
            Error::invalid_input(format!(
                "leaf object_size does not fit usize: path={}, object_size={}",
                child.path, child.object_size
            ))
        })?;
        Ok(LeafObject(SmallReader::new(
            store.inner.clone(),
            path,
            3,
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
        // Relative paths can name different files in cloned dataset bases.
        let cache = Arc::new(
            self.caches
                .files
                .with_key_prefix(&format!("{:?}", self.child_location(child)?)),
        );
        let reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &cache,
            FileReaderOptions::default(),
        )
        .await?;

        // Opening the footer has already fetched the complete SmallReader
        // object through the scheduler. Inspect those cached bytes, not a HEAD
        // or a second GET, before trusting the reference's size.
        let actual_size = object_reader.get_all().await?.len() as u64;
        if actual_size != child.object_size {
            return Err(super::validation::corrupt(format!(
                "Leaf {} has {actual_size} bytes but its parent declares {}",
                child.path, child.object_size,
            )));
        }
        Ok(reader)
    }

    /// Read a leaf from storage and check it against its parent reference,
    /// bypassing every cache.
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
            let decoded = super::leaf::decode(&batch, || self.lock_interner())?;
            for fragment in decoded {
                if fragments
                    .last()
                    .is_some_and(|previous| previous.id >= fragment.id)
                {
                    return Err(super::validation::corrupt(
                        "leaf fragment IDs are out of order across batches",
                    ));
                }
                fragments.push(fragment);
            }
        }
        let mut summary = node::leaf_ref(
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
        summary.min_key = child.min_key;
        summary.base_id = child.base_id;
        if summary.num_keys != child.num_keys
            || summary.total_rows != child.total_rows
            || summary.visible_rows != child.visible_rows
            || summary.height != 0
            || summary.num_children != 0
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
        Path::from("_bt/leaf").join(format!("{}.lance", Uuid::new_v4()))
    }

    fn node_path(&self) -> Path {
        Path::from("_bt/node").join(format!("{}.node", Uuid::new_v4()))
    }

    fn resolve_path(&self, path: &str) -> Result<Path> {
        self.resolve_under(&self.base, path)
    }

    fn resolve_child(&self, child: &pb::FragmentTreeChild) -> Result<(Arc<ObjectStore>, Path)> {
        let (store, base) = self.child_dataset(child)?;
        Ok((store, self.resolve_under(&base, &child.path)?))
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
            // Separate in-memory stores can have identical URL prefixes.
            memory_store: (store.scheme() == "memory")
                .then(|| Arc::as_ptr(&store.inner) as *const () as usize),
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
        if !path.starts_with("_bt/") || path != relative.as_ref() {
            return Err(super::validation::corrupt(format!(
                "Metadata path {path:?} must be relative to _bt/"
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

/// Leaf decode interns every file it reads, and the other variants only drop
/// files or edit a file's path, size, base or deletion file. So only these two
/// variants can leave a fragment holding field lists the interner has not seen.
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

/// Cut a range whose encoding exceeds the leaf target into ordered, contiguous,
/// non-empty pieces in one step, so no intermediate half is encoded only to be
/// rejected. A piece that still encodes over the target is cut again.
fn split_ranges(
    fragments: &[Fragment],
    range: Range<usize>,
    encoded_bytes: u64,
    max_leaf_bytes: u64,
) -> Vec<Range<usize>> {
    // A power of two reproduces the leaf count that halving reaches on uniform
    // records. With exactly ceil(encoded / target) pieces each would sit within
    // a fraction of a percent of the target, and FSST sampling would push about
    // half of them over it into another cut.
    let piece_count = encoded_bytes
        .div_ceil(max_leaf_bytes)
        .next_power_of_two()
        .min(range.len() as u64);
    // Encoded size follows the data file count, so equal file weight keeps
    // pieces near equal bytes when a few fragments hold most of the files.
    // Weights only place cuts. Each piece's actual encoding decides admission.
    let weight = |fragment: &Fragment| 1 + fragment.files.len() as u64;
    let total_weight: u64 = fragments[range.clone()].iter().map(weight).sum();
    // A first piece of at least one fragment guarantees two pieces even when
    // the leading fragment outweighs every other cut target.
    let first_weight = weight(&fragments[range.start]);
    let mut pieces = Vec::new();
    let mut start = range.start;
    let mut end = range.start;
    let mut weight_before_end = 0;
    for piece in 1..=piece_count {
        let cut_weight = (total_weight * piece / piece_count).max(first_weight);
        while end < range.end && weight_before_end + weight(&fragments[end]) <= cut_weight {
            weight_before_end += weight(&fragments[end]);
            end += 1;
        }
        // A fragment heavier than one piece's share would leave an empty piece.
        // Dropping it keeps that fragment with its neighbors instead of
        // publishing one-fragment leaves beside it.
        if end > start {
            pieces.push(start..end);
            start = end;
        }
    }
    pieces
}

/// Resolve the Version Manifest's root UUID within its dataset.
pub fn root_path(bytes: &[u8]) -> Result<String> {
    let uuid = Uuid::from_slice(bytes).map_err(|_| {
        super::validation::corrupt(format!("root_uuid has {} bytes; expected 16", bytes.len()))
    })?;
    Ok(format!("_bt/root/{uuid}.root"))
}

fn stamp_unset_base_id(fragment: &mut Fragment, base_id: u32) {
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

/// Stamp unset file `base_id`s, including row lineage column carriers.
pub fn preserve_inherited_refs(
    mutations: &mut [pb::FragmentTreeMutation],
    base_id: u32,
) -> Result<()> {
    for tagged in mutations {
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

fn derived_child_fields_match(
    summary: &pb::FragmentTreeChild,
    child: &pb::FragmentTreeChild,
) -> bool {
    summary.num_keys == child.num_keys
        && summary.total_rows == child.total_rows
        && summary.visible_rows == child.visible_rows
        && summary.height == child.height
        && summary.num_children == child.num_children
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::format::{
        DataFile, DeletionFile, DeletionFileType, ROW_CREATED_AT_VERSION_FIELD_ID, ROW_ID_FIELD_ID,
        ROW_LAST_UPDATED_AT_VERSION_FIELD_ID, RowDatasetVersionMeta, RowIdMeta,
    };
    use crate::fragment_metadata::action;
    use crate::fragment_metadata::support::{
        data_file_path, make_backfill_data_file, make_fragment, make_fragment_with_files,
        make_replacement_data_file,
    };
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::scheduler::SchedulerConfig;
    use object_store::ObjectStoreExt;
    use rstest::rstest;

    fn column_lineage_fragment(id: u64) -> Fragment {
        let mut fragment = make_fragment(id);
        fragment.files.push(DataFile::new(
            format!("lineage-{id}.lance"),
            vec![
                ROW_ID_FIELD_ID,
                ROW_CREATED_AT_VERSION_FIELD_ID,
                ROW_LAST_UPDATED_AT_VERSION_FIELD_ID,
            ],
            vec![0, 1, 2],
            ConcreteFileVersion::V2_2,
            None,
            None,
        ));
        fragment.row_id_meta = Some(RowIdMeta::Column);
        fragment.created_at_version_meta = Some(RowDatasetVersionMeta::Column);
        fragment.last_updated_at_version_meta = Some(RowDatasetVersionMeta::Column);
        fragment
    }

    #[rstest]
    #[case::historical_numbers(0, 3)]
    #[case::unknown_numbers(17, 42)]
    #[tokio::test]
    async fn leaf_preserves_data_file_version_numbers(#[case] major: u32, #[case] minor: u32) {
        let base = TempObjDir::default();
        let store = test_store(base.clone());
        let mut fragment = make_fragment(0);
        fragment.files[0].file_major_version = major;
        fragment.files[0].file_minor_version = minor;
        let written = store.write_leaf(&[fragment.clone()], 0).await.unwrap();
        assert_eq!(
            store.read_leaf(&written.child_ref).await.unwrap(),
            vec![fragment]
        );
    }

    #[tokio::test]
    async fn internal_read_rejects_an_understated_object_size() {
        let base = TempObjDir::default();
        let store = test_store(base.clone());
        let children = (0..2)
            .map(|id| {
                node::leaf_ref(format!("_bt/leaf/{id}.lance"), &[make_fragment(id)], 1, 0).unwrap()
            })
            .collect();
        let mut written = store.write_internal(children, Vec::new()).await.unwrap();
        written.child_ref.object_size = 1;
        let error = store.read_internal(&written.child_ref).await.unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains("parent declares"), "{error}");
    }

    #[rstest]
    #[case::at_limit(false)]
    #[case::over_limit(true)]
    #[tokio::test]
    async fn internal_write_respects_hard_capacity(#[case] over_limit: bool) {
        let base = TempObjDir::default();
        let mut store = test_store(base.clone());
        let children: Vec<_> = (0..2)
            .map(|id| {
                node::leaf_ref(format!("_bt/leaf/{id}.lance"), &[make_fragment(id)], 1, 0).unwrap()
            })
            .collect();
        let size = pb::FragmentTreeNode {
            children: children.clone(),
            buffer: Vec::new(),
        }
        .encoded_len() as u64;
        store.hard_capacity_bytes = size - u64::from(over_limit);
        let result = store.write_internal(children, Vec::new()).await;
        if over_limit {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(error.to_string().contains("hard_capacity_bytes"), "{error}");
            assert!(
                store
                    .object_store
                    .inner
                    .list(Some(&base))
                    .try_next()
                    .await
                    .unwrap()
                    .is_none()
            );
        } else {
            assert_eq!(result.unwrap().child_ref.object_size, size);
        }
    }

    #[rstest]
    #[case::reader_cache(false)]
    #[case::validation_scratch(true)]
    #[tokio::test]
    async fn leaf_caches_preserve_resolved_owner(#[case] retain: bool) {
        let cache = Arc::new(LanceCache::with_capacity(1024 * 1024));
        let mut bases = HashMap::new();
        let mut children = Vec::new();
        for id in 0..2 {
            let object_store = Arc::new(ObjectStore::memory());
            let source = NodeStore::new(
                object_store.clone(),
                Path::default(),
                ScanScheduler::new(object_store.clone(), SchedulerConfig::default_for_testing()),
                cache.clone(),
            );
            let mut child = source
                .write_leaf(&[make_fragment_with_files(id, id as u32 + 1)], 0)
                .await
                .unwrap()
                .child_ref;
            let bytes = object_store
                .inner
                .get(&Path::from(child.path.clone()))
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            child.path = "_bt/leaf/shared-name.lance".into();
            object_store
                .put(&Path::from(child.path.clone()), &bytes)
                .await
                .unwrap();
            child.base_id = Some(id as u32);
            bases.insert(id as u32, (object_store, Path::default()));
            children.push(child);
        }
        // Aliases resolve to the same bytes but stamp different inherited base IDs.
        bases.insert(2, bases[&0].clone());
        let mut alias = children[0].clone();
        alias.base_id = Some(2);
        children.push(alias);
        let object_store = Arc::new(ObjectStore::memory());
        let mut dest = NodeStore::new(
            object_store.clone(),
            Path::default(),
            ScanScheduler::new(object_store, SchedulerConfig::default_for_testing()),
            cache,
        );
        dest.set_foreign_bases(bases);
        dest.set_leaf_cache(LanceCache::with_capacity(1024 * 1024));
        if retain {
            dest.retain_validation_reads().unwrap();
        }
        // First sighting, admission and a cache hit select the stamped record.
        for _ in 0..3 {
            for (child, id) in children.iter().zip([0, 1, 0]) {
                let mut expected = make_fragment_with_files(id, id as u32 + 1);
                for file in &mut expected.files {
                    file.base_id = child.base_id;
                }
                let selected = dest
                    .read_leaf_shared(child, node::ROOT_EXCLUSIVE_END)
                    .await
                    .unwrap()
                    .select(|fragment| fragment.id == id);
                assert_eq!(selected, vec![expected.clone()]);
                assert_eq!(dest.read_leaf(child).await.unwrap(), vec![expected]);
            }
        }
    }

    #[tokio::test]
    async fn root_read_preserves_unknown_protobuf_fields_without_writer_limits() {
        let base = TempObjDir::default();
        let store = test_store(base.clone());
        let mut bytes = pb::FragmentTreeRoot::default().encode_to_vec();
        bytes.extend_from_slice(&[0xa2, 0x06, 0x80, 0x01]);
        bytes.extend_from_slice(&[0; 128]);
        let uuid = Uuid::new_v4();
        let path = root_path(uuid.as_bytes()).unwrap();
        store
            .object_store
            .put(&store.resolve_path(&path).unwrap(), &bytes)
            .await
            .unwrap();
        store.read_root_base(uuid.as_bytes()).await.unwrap();
    }

    #[tokio::test]
    async fn leaf_read_rejects_an_understated_object_size() {
        let base = TempObjDir::default();
        let store = test_store(base.clone());
        let written = store.write_leaf(&[make_fragment(0)], 0).await.unwrap();
        let path = store.resolve_path(&written.child_ref.path).unwrap();
        let mut bytes = get_whole(&store.object_store, &path)
            .await
            .unwrap()
            .to_vec();
        bytes.push(0);
        store.object_store.put(&path, &bytes).await.unwrap();
        let error = store.read_leaf(&written.child_ref).await.unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains("parent declares"), "{error}");
    }

    fn test_store(base: Path) -> NodeStore {
        let object_store = Arc::new(ObjectStore::local());
        let scheduler =
            ScanScheduler::new(object_store.clone(), SchedulerConfig::default_for_testing());
        NodeStore::new(
            object_store,
            base,
            scheduler,
            Arc::new(LanceCache::with_capacity(64 * 1024 * 1024)),
        )
    }

    #[rstest]
    #[case::encode_on_write(false)]
    #[case::reuse_encoding(true)]
    #[tokio::test]
    async fn encoded_leaf_target_and_oversized_singleton_limit(#[case] has_encoding: bool) {
        let base = TempObjDir::default();
        let store = test_store(base.clone());
        let config = node::FragmentTreeConfig::new(4096, 8)
            .with_max_leaf_bytes(8192)
            .with_hard_capacity_bytes(4 * 1024 * 1024);
        let fragments: Vec<_> = (0..256).map(make_fragment).collect();
        let encoded = if has_encoding {
            Some(store.encode_leaf(&fragments).await.unwrap())
        } else {
            None
        };
        let written = store
            .write_leaves(&fragments, encoded, 17, &config)
            .await
            .unwrap();
        assert!(written.len() > 1);
        let mut read = Vec::new();
        for leaf in &written {
            assert!(leaf.io_bytes <= config.max_leaf_bytes || leaf.child_ref.num_keys == 1);
            assert_eq!(leaf.child_ref.object_size, leaf.io_bytes);
            assert_eq!(leaf.child_ref.materialized_through_action_sequence, 17);
            read.extend(store.read_leaf(&leaf.child_ref).await.unwrap());
        }
        assert_eq!(read, fragments);

        let wide = make_fragment_with_files(257, 4096);
        let encoded = if has_encoding {
            Some(
                store
                    .encode_leaf(std::slice::from_ref(&wide))
                    .await
                    .unwrap(),
            )
        } else {
            None
        };
        let written = store
            .write_leaves(std::slice::from_ref(&wide), encoded, 18, &config)
            .await
            .unwrap();
        assert_eq!(written.len(), 1);
        assert!(written[0].io_bytes > config.max_leaf_bytes);
        assert_eq!(
            store.read_leaf(&written[0].child_ref).await.unwrap(),
            vec![wide.clone()]
        );
        let before = store
            .object_store
            .read_dir_all(&store.base.clone().join("_bt").join("leaf"), None)
            .map_ok(|meta| meta.location)
            .try_collect::<std::collections::BTreeSet<_>>()
            .await
            .unwrap();
        let restricted = config.with_hard_capacity_bytes(16 * 1024);
        let encoded = if has_encoding {
            Some(
                store
                    .encode_leaf(std::slice::from_ref(&wide))
                    .await
                    .unwrap(),
            )
        } else {
            None
        };
        let error = store
            .write_leaves(&[wide], encoded, 19, &restricted)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("fragment 257"));
        assert!(error.to_string().contains("hard_capacity_bytes=16384"));
        assert_eq!(
            store
                .object_store
                .read_dir_all(&store.base.clone().join("_bt").join("leaf"), None)
                .map_ok(|meta| meta.location)
                .try_collect::<std::collections::BTreeSet<_>>()
                .await
                .unwrap(),
            before
        );
    }

    /// `count` fragments with 16 files, where `wide` adjacent fragments in the
    /// middle carry 512 files.
    fn fragments_with_wide_cluster(count: u64, wide: u64) -> Vec<Fragment> {
        let wide_ids = (count - wide) / 2..(count + wide) / 2;
        (0..count)
            .map(|id| make_fragment_with_files(id, if wide_ids.contains(&id) { 512 } else { 16 }))
            .collect()
    }

    fn uniform_fragments(count: u64) -> Vec<Fragment> {
        (0..count).map(make_fragment).collect()
    }

    fn assert_ordered_cover(pieces: &[Range<usize>], range: &Range<usize>) {
        assert_eq!(pieces.first().map(|piece| piece.start), Some(range.start));
        assert_eq!(pieces.last().map(|piece| piece.end), Some(range.end));
        for piece in pieces {
            assert!(!piece.is_empty(), "{pieces:?}");
        }
        for pair in pieces.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "{pieces:?}");
        }
    }

    const TARGET: u64 = 1 << 20;

    #[rstest]
    #[case::uniform_4096(uniform_fragments(4096), 0..4096, 12 * TARGET, 16)]
    #[case::uniform_1003(uniform_fragments(1100), 7..1010, 3 * TARGET + 1, 4)]
    #[case::clustered_wide(fragments_with_wide_cluster(4096, 41), 0..4096, 12 * TARGET, 16)]
    #[case::pieces_exceed_records(uniform_fragments(3), 0..3, 100 * TARGET, 3)]
    #[case::dominant_first_fragment(
        std::iter::once(make_fragment_with_files(0, 4096))
            .chain((1..5).map(make_fragment))
            .collect(),
        0..5,
        3 * TARGET,
        2
    )]
    fn split_ranges_cover_the_range_in_order(
        #[case] fragments: Vec<Fragment>,
        #[case] range: Range<usize>,
        #[case] encoded_bytes: u64,
        #[case] expected_pieces: usize,
    ) {
        let pieces = split_ranges(&fragments, range.clone(), encoded_bytes, TARGET);
        assert_ordered_cover(&pieces, &range);
        assert_eq!(pieces.len(), expected_pieces, "{pieces:?}");
    }

    /// The pieces repeated halving at `floor(len / 2)` reaches after `depth` levels.
    fn bisect(range: Range<usize>, depth: u32) -> Vec<Range<usize>> {
        if depth == 0 {
            return vec![range];
        }
        let middle = range.start + range.len() / 2;
        let mut pieces = bisect(range.start..middle, depth - 1);
        pieces.extend(bisect(middle..range.end, depth - 1));
        pieces
    }

    #[rstest]
    #[case::pow2_4096_into_16(4096, 0..4096, 12 * TARGET, 4)]
    #[case::odd_1003_into_4(1100, 7..1010, 3 * TARGET, 2)]
    fn split_ranges_match_bisection_on_uniform_records(
        #[case] count: u64,
        #[case] range: Range<usize>,
        #[case] encoded_bytes: u64,
        #[case] depth: u32,
    ) {
        let fragments = uniform_fragments(count);
        assert_eq!(
            split_ranges(&fragments, range.clone(), encoded_bytes, TARGET),
            bisect(range, depth)
        );
    }

    #[rstest]
    #[case::uniform(fragments_with_wide_cluster(4096, 0))]
    #[case::clustered_wide(fragments_with_wide_cluster(4096, 41))]
    #[tokio::test]
    async fn oversized_range_encodes_each_record_at_most_twice_plus_resplits(
        #[case] fragments: Vec<Fragment>,
    ) {
        let base = TempObjDir::default();
        let store = test_store(base.clone());
        // More than eight targets, so halving would encode every record at
        // least five times, the probe and four levels. Sixteen-file records
        // keep the per-leaf file overhead small next to a target, as in real leaves.
        let encoded_bytes = store.encode_leaf(&fragments).await.unwrap().len() as u64;
        let config = node::FragmentTreeConfig::new(4096, 8)
            .with_max_leaf_bytes(encoded_bytes / 12)
            .with_hard_capacity_bytes(64 * 1024 * 1024);
        store
            .encoded_records
            .store(0, std::sync::atomic::Ordering::Relaxed);
        let written = store
            .write_leaves(&fragments, None, 0, &config)
            .await
            .unwrap();
        let encoded_records = store
            .encoded_records
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            encoded_records * 2 <= fragments.len() * 5,
            "{encoded_records} record encodes for {} fragments in {} leaves",
            fragments.len(),
            written.len()
        );
        let mut read = Vec::new();
        for leaf in &written {
            assert!(leaf.io_bytes <= config.max_leaf_bytes || leaf.child_ref.num_keys == 1);
            read.extend(store.read_leaf(&leaf.child_ref).await.unwrap());
        }
        assert_eq!(read, fragments);
    }

    #[tokio::test]
    async fn skewed_range_leaves_stay_above_quarter_target() {
        let base = TempObjDir::default();
        let store = test_store(base.clone());
        let fragments = fragments_with_wide_cluster(4096, 41);
        let config = node::FragmentTreeConfig::new(4096, 8)
            .with_max_leaf_bytes(256 * 1024)
            .with_hard_capacity_bytes(64 * 1024 * 1024);
        let written = store
            .write_leaves(&fragments, None, 0, &config)
            .await
            .unwrap();
        let sizes: Vec<_> = written.iter().map(|leaf| leaf.io_bytes).collect();
        assert!(written.len() > 8, "{sizes:?}");
        for leaf in &written {
            let oversized_singleton =
                leaf.child_ref.num_keys == 1 && leaf.io_bytes > config.max_leaf_bytes;
            assert!(
                oversized_singleton || leaf.io_bytes >= config.max_leaf_bytes / 4,
                "{sizes:?}"
            );
        }
    }

    #[tokio::test]
    async fn leaf_round_trips_fragment_metadata_and_empty_fragment() {
        let mut fragment = make_fragment(7);
        fragment.deletion_file = Some(DeletionFile {
            read_version: 3,
            id: 11,
            file_type: DeletionFileType::Bitmap,
            num_deleted_rows: Some(1),
            base_id: Some(2),
        });
        fragment
            .overlays
            .push(crate::format::overlay::DataOverlayFile {
                data_file: fragment.files[0].clone(),
                coverage: crate::format::overlay::OverlayCoverage::dense(
                    roaring::RoaringBitmap::from_iter([0]),
                ),
                committed_version: 3,
            });
        fragment.row_id_meta = Some(RowIdMeta::Inline(vec![1, 2, 3, 4].into()));
        fragment.created_at_version_meta =
            Some(RowDatasetVersionMeta::Inline(Arc::from([5, 6, 7])));
        fragment.last_updated_at_version_meta =
            Some(RowDatasetVersionMeta::Inline(Arc::from([8, 9, 10])));

        let mut empty_fragment = Fragment::new(8);
        empty_fragment.physical_rows = Some(12);
        empty_fragment.row_id_meta = Some(RowIdMeta::Inline(vec![12, 13].into()));

        let expected = vec![fragment, empty_fragment, column_lineage_fragment(9)];
        let tempdir = TempObjDir::default();
        let store = test_store(tempdir.clone().join("fragment_metadata"));
        let written = store.write_leaf(&expected, 0).await.unwrap();
        let actual = store.read_leaf(&written.child_ref).await.unwrap();

        assert_eq!(actual, expected);
    }
    #[tokio::test]
    async fn validation_spill_preserves_known_zero_counts_and_relative_paths() {
        let base = TempObjDir::default();
        let mut store = test_store(base.clone());
        let mut fragment = make_fragment(0);
        fragment.physical_rows = Some(0);
        fragment.deletion_file = Some(DeletionFile {
            read_version: 1,
            id: 1,
            file_type: DeletionFileType::Bitmap,
            num_deleted_rows: Some(0),
            base_id: None,
        });
        let written = store
            .write_leaf(std::slice::from_ref(&fragment), 0)
            .await
            .unwrap();
        assert!(written.child_ref.path.starts_with("_bt/leaf/"));
        assert!(!written.child_ref.path.contains("%2F"));
        let path = store.resolve_path(&written.child_ref.path).unwrap();
        assert!(store.object_store.inner.head(&path).await.is_ok());
        store.retain_validation_reads().unwrap();
        // First read populates scratch; the second decodes the scratch record.
        for _ in 0..2 {
            assert_eq!(
                store.read_leaf(&written.child_ref).await.unwrap(),
                vec![fragment.clone()]
            );
        }
        let mut corrupt = written.child_ref.clone();
        corrupt.num_keys += 1;
        let error = store.read_leaf(&corrupt).await.unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains("parent reference"), "{error}");
        for path in [
            "../_bt/leaf/escape",
            "/_bt/leaf/absolute",
            "data/not-metadata",
        ] {
            assert!(store.resolve_path(path).is_err(), "{path}");
        }
    }

    /// A leaf decoded twice is kept, and then answers only for the parent
    /// reference it was checked against, whole or by record. Reads that
    /// succeed after the object is gone prove they came from the cache.
    #[tokio::test]
    async fn leaf_cache_serves_only_the_reference_it_checked() {
        let base = TempObjDir::default();
        let mut store = test_store(base.clone());
        store.set_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024));
        let fragments: Vec<_> = (0..4).map(make_fragment).collect();
        let once = store.write_leaf(&fragments, 0).await.unwrap();
        let written = store.write_leaf(&fragments, 0).await.unwrap();
        assert_eq!(store.read_leaf(&once.child_ref).await.unwrap(), fragments);
        for _ in 0..2 {
            let read = store.read_leaf(&written.child_ref).await.unwrap();
            assert_eq!(read, fragments);
        }

        let once_path = store.resolve_path(&once.child_ref.path).unwrap();
        store.object_store.inner.delete(&once_path).await.unwrap();
        assert!(store.read_leaf(&once.child_ref).await.is_err());

        let path = store.resolve_path(&written.child_ref.path).unwrap();
        store.object_store.inner.delete(&path).await.unwrap();
        assert_eq!(
            store.read_leaf(&written.child_ref).await.unwrap(),
            fragments
        );
        let selected = store
            .read_leaf_shared(&written.child_ref, node::ROOT_EXCLUSIVE_END)
            .await
            .unwrap()
            .select(|fragment| [1, 3, 9].contains(&fragment.id));
        assert_eq!(selected, vec![fragments[1].clone(), fragments[3].clone()]);
        let point = store
            .read_leaf_fragment(&written.child_ref, 2, node::ROOT_EXCLUSIVE_END)
            .await;
        assert_eq!(point.unwrap().as_ref(), Some(&fragments[2]));
        let missing = store
            .read_leaf_fragment(&written.child_ref, 9, node::ROOT_EXCLUSIVE_END)
            .await;
        assert!(missing.unwrap().is_none());

        let mut changed = written.child_ref.clone();
        changed.visible_rows += 1;
        assert!(store.read_leaf(&changed).await.is_err());
        assert!(
            store
                .read_leaf_shared(&changed, node::ROOT_EXCLUSIVE_END)
                .await
                .is_err()
        );
        assert!(
            store
                .read_leaf_fragment(&changed, 2, node::ROOT_EXCLUSIVE_END)
                .await
                .is_err()
        );
        assert!(store.read_leaf_uncached(&written.child_ref).await.is_err());
    }

    #[derive(Clone, Copy)]
    enum Touch {
        Point,
        Whole,
    }

    async fn touch(
        store: &NodeStore,
        child: &pb::FragmentTreeChild,
        fragments: &[Fragment],
        how: Touch,
    ) {
        match how {
            Touch::Point => {
                let point = store
                    .read_leaf_fragment(child, fragments[1].id, node::ROOT_EXCLUSIVE_END)
                    .await
                    .unwrap();
                assert_eq!(point.as_ref(), Some(&fragments[1]));
            }
            Touch::Whole => assert_eq!(store.read_leaf(child).await.unwrap(), fragments),
        }
    }

    /// Points neither advance whole-read admission nor prevent its cache hits.
    /// Reads after deleting the object prove whole reads admitted the leaf.
    #[rstest]
    #[case::point_then_point(Touch::Point, Touch::Point, 2)]
    #[case::whole_then_point(Touch::Whole, Touch::Point, 1)]
    #[case::point_then_whole(Touch::Point, Touch::Whole, 1)]
    #[tokio::test]
    async fn point_reads_preserve_whole_read_admission(
        #[case] first: Touch,
        #[case] second: Touch,
        #[case] additional_whole_reads: usize,
    ) {
        let base = TempObjDir::default();
        let mut store = test_store(base.clone());
        store.set_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024));
        let fragments: Vec<_> = (0..4).map(make_fragment).collect();
        let written = store.write_leaf(&fragments, 0).await.unwrap();
        for how in [first, second] {
            touch(&store, &written.child_ref, &fragments, how).await;
        }
        let key = LeafKey::new(&store, &written.child_ref).unwrap();
        let leaves = store.caches.leaves.as_ref().unwrap();
        assert!(leaves.get_with_key(&key).await.is_none());
        for read in 0..additional_whole_reads {
            touch(&store, &written.child_ref, &fragments, Touch::Whole).await;
            assert_eq!(
                leaves.get_with_key(&key).await.is_some(),
                read + 1 == additional_whole_reads
            );
        }

        let path = store.resolve_path(&written.child_ref.path).unwrap();
        store.object_store.inner.delete(&path).await.unwrap();
        let point = store
            .read_leaf_fragment(&written.child_ref, 3, node::ROOT_EXCLUSIVE_END)
            .await;
        assert_eq!(point.unwrap().as_ref(), Some(&fragments[3]));
        let missing = store
            .read_leaf_fragment(&written.child_ref, 9, node::ROOT_EXCLUSIVE_END)
            .await;
        assert!(missing.unwrap().is_none());
        assert_eq!(
            store.read_leaf(&written.child_ref).await.unwrap(),
            fragments
        );

        let mut changed = written.child_ref.clone();
        changed.visible_rows += 1;
        assert!(
            store
                .read_leaf_fragment(&changed, 3, node::ROOT_EXCLUSIVE_END)
                .await
                .is_err()
        );
    }

    /// Repeated point reads still need the object until whole reads admit it.
    #[tokio::test]
    async fn point_reads_do_not_admit_leaves() {
        let base = TempObjDir::default();
        let mut store = test_store(base.clone());
        store.set_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024));
        let fragments: Vec<_> = (0..4).map(make_fragment).collect();
        let written = store.write_leaf(&fragments, 0).await.unwrap();
        for _ in 0..3 {
            touch(&store, &written.child_ref, &fragments, Touch::Point).await;
        }

        let path = store.resolve_path(&written.child_ref.path).unwrap();
        store.object_store.inner.delete(&path).await.unwrap();
        assert!(
            store
                .read_leaf_fragment(&written.child_ref, 2, node::ROOT_EXCLUSIVE_END)
                .await
                .is_err()
        );
    }

    /// Header checks pass a leaf whose other records carry files or overlays
    /// that fail full validation. Selective reads validate the requested record;
    /// every whole read rejects the corruption without admitting the leaf.
    #[rstest]
    #[case::file_mapping(|fragment: &mut Fragment| fragment.files[0].fields = Arc::from([0, 1, 2]))]
    #[case::overlay_mapping(|fragment: &mut Fragment| {
        let mut data_file = fragment.files[0].clone();
        data_file.fields = Arc::from([0, 1, 2]);
        fragment.overlays.push(crate::format::overlay::DataOverlayFile {
            data_file,
            coverage: crate::format::overlay::OverlayCoverage::dense(
                roaring::RoaringBitmap::from_iter([0]),
            ),
            committed_version: 1,
        });
    })]
    #[tokio::test]
    async fn point_reads_leave_unrequested_corruption_for_whole_reads(
        #[case] corrupt: fn(&mut Fragment),
    ) {
        let base = TempObjDir::default();
        let mut store = test_store(base.clone());
        store.set_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024));
        let mut fragments: Vec<_> = (0..4).map(make_fragment).collect();
        corrupt(&mut fragments[2]);
        let written = store.write_leaf(&fragments, 0).await.unwrap();
        let whole = store
            .read_leaf_uncached(&written.child_ref)
            .await
            .unwrap_err();
        assert!(matches!(whole, Error::CorruptFile { .. }), "{whole}");

        for _ in 0..3 {
            let point = store
                .read_leaf_fragment(&written.child_ref, 0, node::ROOT_EXCLUSIVE_END)
                .await
                .unwrap();
            assert_eq!(point.as_ref(), Some(&fragments[0]));
            let point = store
                .read_leaf_fragment(&written.child_ref, 2, node::ROOT_EXCLUSIVE_END)
                .await
                .unwrap_err();
            assert!(matches!(point, Error::CorruptFile { .. }), "{point}");
            let whole = store
                .read_leaf_in_range(&written.child_ref, node::ROOT_EXCLUSIVE_END)
                .await
                .unwrap_err();
            assert!(matches!(whole, Error::CorruptFile { .. }), "{whole}");
        }
        let key = LeafKey::new(&store, &written.child_ref).unwrap();
        let leaves = store.caches.leaves.as_ref().unwrap();
        assert!(leaves.get_with_key(&key).await.is_none());
    }

    /// A set read decodes every record of an uncached leaf, so a malformed
    /// record it did not request fails the sighting and the admitting read,
    /// and the leaf is never admitted.
    #[tokio::test]
    async fn set_reads_validate_unrequested_records_on_every_uncached_read() {
        let base = TempObjDir::default();
        let mut store = test_store(base.clone());
        store.set_leaf_cache(LanceCache::with_capacity(1024 * 1024));
        let mut fragments: Vec<_> = (0..2).map(make_fragment).collect();
        fragments[1].files[0] = DataFile::new(
            "invalid.lance",
            vec![0],
            vec![],
            ConcreteFileVersion::V2_1,
            None,
            None,
        );
        let written = store.write_leaf(&fragments, 0).await.unwrap();
        for _ in 0..3 {
            let error = store
                .read_leaf_shared(&written.child_ref, node::ROOT_EXCLUSIVE_END)
                .await
                .err()
                .expect("a malformed unrequested record must fail the read");
            assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
            assert!(
                error.to_string().contains("fewer column_indices"),
                "{error}"
            );
        }
    }

    #[tokio::test]
    async fn retained_validation_reads_bypass_the_leaf_cache_for_shared_reads() {
        let base = TempObjDir::default();
        let mut store = test_store(base.clone());
        store.set_leaf_cache(LanceCache::with_capacity(1024 * 1024));
        let fragments: Vec<_> = (0..2).map(make_fragment).collect();
        let written = store.write_leaf(&fragments, 0).await.unwrap();
        for _ in 0..2 {
            assert_eq!(
                store.read_leaf(&written.child_ref).await.unwrap(),
                fragments
            );
        }
        store.retain_validation_reads().unwrap();
        let path = store.resolve_path(&written.child_ref.path).unwrap();
        store.object_store.inner.delete(&path).await.unwrap();
        let error = store
            .read_leaf_shared(&written.child_ref, node::ROOT_EXCLUSIVE_END)
            .await
            .err()
            .expect("retained reads must fetch the deleted leaf");
        assert!(error.is_not_found(), "{error}");
        // Local-store errors use platform path separators.
        assert!(
            error
                .to_string()
                .replace('\\', "/")
                .contains(&written.child_ref.path),
            "{error}"
        );
    }

    /// The admitting read and later hits hand out the cached leaf itself, so
    /// admission copies no record.
    #[tokio::test]
    async fn admission_shares_the_cached_leaf_with_its_reader() {
        let base = TempObjDir::default();
        let mut store = test_store(base.clone());
        store.set_leaf_cache(LanceCache::with_capacity(1024 * 1024));
        let fragments: Vec<_> = (0..4).map(make_fragment).collect();
        let written = store.write_leaf(&fragments, 0).await.unwrap();
        let child = &written.child_ref;
        let first = store
            .read_leaf_shared(child, node::ROOT_EXCLUSIVE_END)
            .await
            .unwrap();
        assert!(matches!(&first, LeafRead::Owned(read) if *read == fragments));
        let key = LeafKey::new(&store, child).unwrap();
        let leaves = store.caches.leaves.as_ref().unwrap();
        assert!(leaves.get_with_key(&key).await.is_none());
        for _ in 0..2 {
            let LeafRead::Cached(read) = store
                .read_leaf_shared(child, node::ROOT_EXCLUSIVE_END)
                .await
                .unwrap()
            else {
                panic!("the second and later reads must share the cached leaf");
            };
            let cached = leaves.get_with_key(&key).await.unwrap();
            assert!(Arc::ptr_eq(&read, &cached));
            assert_eq!(read.fragments, fragments);
        }
    }

    /// A point read decodes one record but checks every parent field a whole
    /// read checks, so a tampered reference fails either way.
    #[rstest]
    #[case::extra_key(|child: &mut pb::FragmentTreeChild| child.num_keys += 1)]
    #[case::extra_row(|child: &mut pb::FragmentTreeChild| child.total_rows += 1)]
    #[case::extra_visible_row(|child: &mut pb::FragmentTreeChild| child.visible_rows += 1)]
    #[case::fence_above_first_key(|child: &mut pb::FragmentTreeChild| child.min_key = 1)]
    #[tokio::test]
    async fn point_reads_reject_references_whole_reads_reject(
        #[case] tamper: fn(&mut pb::FragmentTreeChild),
    ) {
        let base = TempObjDir::default();
        let store = test_store(base.clone());
        let mut fragments: Vec<_> = (0..4).map(make_fragment).collect();
        fragments[2].deletion_file = Some(DeletionFile {
            read_version: 1,
            id: 1,
            file_type: DeletionFileType::Bitmap,
            num_deleted_rows: Some(0),
            base_id: None,
        });
        let written = store.write_leaf(&fragments, 0).await.unwrap();
        for fragment in &fragments {
            let point = store
                .read_leaf_fragment(&written.child_ref, fragment.id, node::ROOT_EXCLUSIVE_END)
                .await
                .unwrap();
            assert_eq!(point.as_ref(), Some(fragment));
        }
        let missing = store
            .read_leaf_fragment(&written.child_ref, 4, node::ROOT_EXCLUSIVE_END)
            .await;
        assert!(missing.unwrap().is_none());

        let mut corrupt = written.child_ref.clone();
        tamper(&mut corrupt);
        let whole = store.read_leaf(&corrupt).await.unwrap_err();
        let point = store
            .read_leaf_fragment(&corrupt, 3, node::ROOT_EXCLUSIVE_END)
            .await
            .unwrap_err();
        assert!(matches!(whole, Error::CorruptFile { .. }), "{whole}");
        assert!(matches!(point, Error::CorruptFile { .. }), "{point}");
    }

    #[tokio::test]
    async fn foreign_interior_buffer_inherits_dataset_on_file_refs() {
        let base = TempObjDir::default();
        let mut store = test_store(base.clone());
        let children = (0..2)
            .map(|id| {
                node::leaf_ref(format!("_bt/leaf/{id}.lance"), &[make_fragment(id)], 1, 0).unwrap()
            })
            .collect();
        let file = make_backfill_data_file(0, 0);
        let mutation = pb::FragmentTreeMutation {
            action_sequence: 1,
            action: Some(action::add_data_file(0, &file)),
            fragment_count_delta: 0,
            total_rows_delta: 0,
            visible_rows_delta: 0,
        };
        let mut written = store
            .write_internal(children, vec![mutation])
            .await
            .unwrap();
        let object_store = store.object_store.clone();
        let dataset_base = store.base.clone();
        store.set_foreign_bases(HashMap::from([(7, (object_store, dataset_base))]));
        written.child_ref.base_id = Some(7);
        let node = store.read_internal(&written.child_ref).await.unwrap();
        let Some(pb::fragment_action::Action::AddDataFile(add)) = node.buffer[0]
            .action
            .as_ref()
            .and_then(|a| a.action.as_ref())
        else {
            panic!("expected AddDataFile");
        };
        assert_eq!(add.file.as_ref().and_then(|file| file.base_id), Some(7));
    }

    #[tokio::test]
    async fn foreign_interior_buffer_preserves_column_lineage_ownership() {
        let source_dir = TempObjDir::default();
        let source = test_store(source_dir.clone());
        let fragment = column_lineage_fragment(2);
        let mutation = pb::FragmentTreeMutation {
            action_sequence: 1,
            action: Some(action::upsert_fragment(&fragment)),
            fragment_count_delta: 1,
            total_rows_delta: fragment.physical_rows.unwrap_or(0) as i64,
            visible_rows_delta: fragment.num_rows().unwrap_or(0) as i64,
        };
        let children = (0..2)
            .map(|id| {
                node::leaf_ref(format!("_bt/leaf/{id}.lance"), &[make_fragment(id)], 1, 0).unwrap()
            })
            .collect();
        let mut written = source
            .write_internal(children, vec![mutation])
            .await
            .unwrap();
        let dest_dir = TempObjDir::default();
        let mut dest = test_store(dest_dir.clone());
        dest.set_foreign_bases(HashMap::from([(
            7,
            (source.object_store.clone(), source.base.clone()),
        )]));
        written.child_ref.base_id = Some(7);
        let node = dest.read_internal(&written.child_ref).await.unwrap();
        let Some(pb::fragment_action::Action::UpsertFragment(encoded)) = node.buffer[0]
            .action
            .as_ref()
            .and_then(|a| a.action.as_ref())
        else {
            panic!("expected UpsertFragment");
        };
        let read = Fragment::try_from(encoded.clone()).unwrap();
        let mut expected = fragment;
        for file in &mut expected.files {
            file.base_id = Some(7);
        }
        assert_eq!(read, expected);
    }

    #[tokio::test]
    async fn leaf_reads_share_field_list_allocations_across_leaves() {
        let base = TempObjDir::default();
        let store = test_store(base.clone());
        let first = store.write_leaf(&[make_fragment(0)], 0).await.unwrap();
        let second = store.write_leaf(&[make_fragment(1)], 0).await.unwrap();
        let a = store.read_leaf(&first.child_ref).await.unwrap();
        let b = store.read_leaf(&second.child_ref).await.unwrap();
        assert!(std::sync::Arc::ptr_eq(
            &a[0].files[0].fields,
            &b[0].files[0].fields
        ));
        assert!(std::sync::Arc::ptr_eq(
            &a[0].files[0].column_indices,
            &b[0].files[0].column_indices
        ));
    }

    fn deletion_file(num_deleted_rows: usize) -> DeletionFile {
        DeletionFile {
            read_version: 1,
            id: num_deleted_rows as u64,
            file_type: DeletionFileType::Array,
            num_deleted_rows: Some(num_deleted_rows),
            base_id: None,
        }
    }

    /// Replays target fragment 1 in the first leaf. Every fragment of the
    /// second leaf holds one more backfill file than the first leaf's, so any
    /// file a replay leaves or adds on fragment 1 has a decoded twin in fragment
    /// 5 with equal field lists.
    async fn decoded_leaf_and_twin(store: &NodeStore) -> (BTreeMap<u64, Fragment>, Fragment) {
        let first: Vec<Fragment> = (0..4)
            .map(|id| {
                let mut fragment = make_fragment_with_files(id, 3);
                fragment.physical_rows = Some(10);
                fragment.deletion_file = Some(deletion_file(2));
                fragment
            })
            .collect();
        let second: Vec<Fragment> = (4..8).map(|id| make_fragment_with_files(id, 4)).collect();
        let first = store.write_leaf(&first, 0).await.unwrap();
        let second = store.write_leaf(&second, 0).await.unwrap();
        let leaf = store
            .read_leaf(&first.child_ref)
            .await
            .unwrap()
            .into_iter()
            .map(|fragment| (fragment.id, fragment))
            .collect();
        let twin = store
            .read_leaf(&second.child_ref)
            .await
            .unwrap()
            .swap_remove(1);
        assert_eq!(twin.id, 5);
        (leaf, twin)
    }

    fn assert_shares_field_lists(fragment: &Fragment, twin: &Fragment) {
        for file in &fragment.files {
            let decoded = twin
                .files
                .iter()
                .find(|decoded| decoded.fields == file.fields)
                .unwrap_or_else(|| panic!("fragment {} has no twin for {}", twin.id, file.path));
            assert!(
                Arc::ptr_eq(&file.fields, &decoded.fields),
                "fields of {} are a private copy",
                file.path
            );
            assert!(
                Arc::ptr_eq(&file.column_indices, &decoded.column_indices),
                "column_indices of {} are a private copy",
                file.path
            );
        }
    }

    fn replay_of_fragment_one(
        action: pb::FragmentAction,
        visible_rows_delta: i64,
    ) -> pb::FragmentTreeMutation {
        pb::FragmentTreeMutation {
            action_sequence: 1,
            action: Some(action),
            fragment_count_delta: 0,
            total_rows_delta: 0,
            visible_rows_delta,
        }
    }

    #[rstest]
    #[case::add_deletion_file(action::add_deletion_file(1, &deletion_file(3)), -1)]
    #[case::clear_deletion_file(action::clear_deletion_file(1), 2)]
    #[case::remove_data_file(action::remove_data_file(1, data_file_path(1, 1)), 0)]
    #[case::replace_data_file(
        action::replace_data_file(1, &data_file_path(1, 1), &make_replacement_data_file(1, 0)),
        0
    )]
    #[tokio::test]
    async fn replay_keeps_leaf_field_lists_for_actions_without_data_files(
        #[case] action: pb::FragmentAction,
        #[case] visible_rows_delta: i64,
    ) {
        let base = TempObjDir::default();
        let store = test_store(base.clone());
        let (mut leaf, twin) = decoded_leaf_and_twin(&store).await;
        let actions = vec![replay_of_fragment_one(action, visible_rows_delta)];
        let mut expected = leaf.clone();
        node::apply_verified(&mut expected, actions.clone()).unwrap();

        store.apply_verified(&mut leaf, actions).unwrap();

        assert_eq!(leaf, expected);
        assert_shares_field_lists(&leaf[&1], &twin);
    }

    #[rstest]
    #[case::upsert_fragment(action::upsert_fragment(&{
        let mut fragment = make_fragment_with_files(1, 4);
        fragment.physical_rows = Some(10);
        fragment.deletion_file = Some(deletion_file(2));
        fragment
    }))]
    #[case::add_data_file(action::add_data_file(1, &make_backfill_data_file(1, 2)))]
    #[tokio::test]
    async fn replay_shares_field_lists_of_new_data_files(#[case] action: pb::FragmentAction) {
        let base = TempObjDir::default();
        let store = test_store(base.clone());
        let (mut leaf, twin) = decoded_leaf_and_twin(&store).await;
        let actions = vec![replay_of_fragment_one(action, 0)];
        let mut expected = leaf.clone();
        node::apply_verified(&mut expected, actions.clone()).unwrap();

        store.apply_verified(&mut leaf, actions).unwrap();

        assert_eq!(leaf, expected);
        assert_eq!(leaf[&1].files.len(), 4);
        assert_shares_field_lists(&leaf[&1], &twin);
    }

    #[tokio::test]
    async fn foreign_leaf_preserves_column_lineage_ownership() {
        let source_dir = TempObjDir::default();
        let source = test_store(source_dir.clone());
        let first = column_lineage_fragment(0);
        let mut second = column_lineage_fragment(1);
        // A chained clone already names the original carrier's owner.
        second.files[1].base_id = Some(9);
        let fragments = vec![first, second];
        let written = source.write_leaf(&fragments, 0).await.unwrap();

        let dest_dir = TempObjDir::default();
        let mut dest = test_store(dest_dir.clone());
        dest.set_foreign_bases(HashMap::from([(
            4,
            (source.object_store.clone(), source.base.clone()),
        )]));
        dest.set_leaf_cache(LanceCache::with_capacity(1024 * 1024));
        let mut child = written.child_ref.clone();
        child.base_id = Some(4);
        let mut expected = fragments;
        for fragment in &mut expected {
            for file in &mut fragment.files {
                if file.base_id.is_none() {
                    file.base_id = Some(4);
                }
            }
        }
        // First sighting, admission and a cache hit keep the stamped owners.
        for _ in 0..3 {
            let selected = dest
                .read_leaf_shared(&child, node::ROOT_EXCLUSIVE_END)
                .await
                .unwrap()
                .select(|fragment| [0, 1, 9].contains(&fragment.id));
            assert_eq!(selected, expected);
        }
        let read = dest.read_leaf(&child).await.unwrap();
        assert_eq!(read, expected);
        // Point reads must apply the same inherited ownership as whole leaves.
        for fragment in expected {
            assert_eq!(
                dest.read_leaf_fragment(&child, fragment.id, node::ROOT_EXCLUSIVE_END)
                    .await
                    .unwrap(),
                Some(fragment)
            );
        }
    }
    #[rstest]
    #[case(0)]
    #[case(15)]
    #[case(17)]
    fn rejects_invalid_root_uuid_length(#[case] length: usize) {
        let result = root_path(&vec![0; length]);
        assert!(matches!(result, Err(Error::CorruptFile { .. })));
    }

    #[test]
    fn root_uuid_uses_display_order() {
        let uuid = uuid::Uuid::parse_str("00112233-4455-6677-8899-aabbccddeeff").unwrap();
        assert_eq!(
            root_path(uuid.as_bytes()).unwrap(),
            "_bt/root/00112233-4455-6677-8899-aabbccddeeff.root"
        );
    }
}
