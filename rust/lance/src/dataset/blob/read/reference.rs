// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{HashMap, HashSet, VecDeque},
    pin::Pin,
    sync::{Arc, Mutex, atomic::Ordering},
    task::{Context, Poll},
};

use bytes::Bytes;
use futures::{Stream, StreamExt, TryStreamExt, stream};
use lance_core::datatypes::BlobKind;
use lance_core::{Error, Result};
use lance_io::object_store::{ObjectStore, uri_to_url};
use lance_io::traits::Reader;
use object_store::path::Path;
use roaring::RoaringTreemap;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use super::plan::{BlobReadBatch, plan_blob_read_batches_with_limits};
use super::{BlobEntry, BlobReadRange, ReadBlobsExecution, execute_blob_read_plan};
use crate::Dataset;
use crate::blob::reference::{BlobRef, BlobRefMetrics, BlobSourceContext, ReaderMetrics, pb};
use crate::dataset::blob::file::{BlobFile, BlobSource};
use crate::dataset::blob::materialize::{
    BlobMaterializationContext, BlobMaterializationReservation,
};

/// One transported source context paired with an authenticated worker snapshot.
///
/// The worker must resolve the context's `access_context_key` to the correct
/// storage settings, credentials, registry, and per-base routing, then open the
/// exact branch/version. Construction checks the loaded manifest; it never opens
/// the latest snapshot or infers authentication from the locator URI.
#[derive(Clone)]
pub struct BlobRefBinding {
    context: BlobSourceContext,
    dataset: Arc<Dataset>,
}

impl BlobRefBinding {
    /// Bind a source to its independently authenticated, exact worker snapshot.
    pub fn try_new(context: BlobSourceContext, dataset: Arc<Dataset>) -> Result<Self> {
        context.validate_dataset(&dataset)?;
        Ok(Self { context, dataset })
    }

    /// The source context whose fingerprint this binding accepts.
    pub fn context(&self) -> &BlobSourceContext {
        &self.context
    }

    fn object_path(&self, object: &pb::ObjectLocator) -> Result<Path> {
        let path = ObjectStore::extract_path_from_uri(
            self.dataset.session.store_registry(),
            &object.object_uri,
        )?;
        let root = match object.base_id {
            Some(id) => Some(
                self.dataset
                    .manifest()
                    .base_paths
                    .get(&id)
                    .ok_or_else(|| {
                        Error::invalid_input(format!("Blob reference uses unknown base_id {id}"))
                    })?
                    .path
                    .as_str(),
            ),
            None if object.origin == pb::object_locator::Origin::Lance as i32 => {
                Some(self.context.table_uri())
            }
            None => None,
        };
        if let Some(root) = root {
            let root_url = uri_to_url(root)?;
            let object_url = uri_to_url(&object.object_uri)?;
            let authority = |url: &url::Url| {
                (
                    url.scheme().to_owned(),
                    url.username().to_owned(),
                    url.host_str().map(str::to_owned),
                    url.port(),
                    url.query().map(str::to_owned),
                )
            };
            let root_path =
                ObjectStore::extract_path_from_uri(self.dataset.session.store_registry(), root)?;
            if authority(&root_url) != authority(&object_url)
                || path.prefix_match(&root_path).is_none()
            {
                return Err(Error::invalid_input(format!(
                    "Blob object {} is outside its bound storage root {root}",
                    object.object_uri,
                )));
            }
        }
        Ok(path)
    }
}

/// Bytes required from one logical blob value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobRefDemand {
    /// Read the entire logical blob, resolving an unknown external length once.
    Whole,
    /// Read a blob-local range. An empty range at the end is valid.
    Range(BlobReadRange),
}

/// An engine-indexed demand for a nullable portable reference.
#[derive(Debug, Clone)]
pub struct BlobRefRequest {
    /// Engine-supplied correlation index, unique within one read operation.
    pub request_index: u64,
    /// `None` represents SQL null and does not require a source binding.
    pub reference: Option<BlobRef>,
    /// Full-value or blob-local range demand.
    pub demand: BlobRefDemand,
}

impl BlobRefRequest {
    /// Create a demand while preserving the engine's correlation index.
    pub fn new(request_index: u64, reference: Option<BlobRef>, demand: BlobRefDemand) -> Self {
        Self {
            request_index,
            reference,
            demand,
        }
    }
}

/// One completed reference demand. Completion order is unspecified.
///
/// The bytes retain their shared materialization reservation, including through
/// `Bytes::clone` and slicing. Consume/drop results to admit later batches. An
/// adapter copying into one output per request can retain these owners alongside
/// the output to keep its logical bytes charged. Additional copies require
/// separate reservations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadBlobRef {
    /// The correlation index supplied in [`BlobRefRequest`].
    pub request_index: u64,
    /// SQL null is `None`; a valid empty result is `Some(Bytes::new())`.
    pub data: Option<Bytes>,
}

/// A cancellable stream of indexed results.
///
/// Cancellation or a read error terminates the operation with an error: pending
/// requests are not successful nulls. Previously yielded buffers remain valid and
/// keep their reservations until released. Dropping the stream releases its
/// pending outputs, admissions, and readers; backend I/O already submitted to a
/// scheduler may finish before its cancelled response is discarded.
pub struct BlobRefReadStream {
    inner: futures::stream::BoxStream<'static, Result<ReadBlobRef>>,
    cancellation: CancellationToken,
}

impl BlobRefReadStream {
    /// Cancel this operation, including a wait for metadata, I/O, or budget space.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// Obtain a token for cancelling the operation from another task.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }
}

impl Stream for BlobRefReadStream {
    type Item = Result<ReadBlobRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// Reusable reader for transported references, with shared scheduling and budgets.
///
/// Bindings use the same slot order as the reference context table. Each locator
/// is checked against that slot's fingerprint before metadata or payload I/O.
/// Metadata caches grow with the distinct objects touched during this session;
/// payload memory is controlled by the shared [`BlobMaterializationContext`].
/// The application must retain source snapshots throughout reads and retries.
///
/// ```
/// # use std::sync::Arc;
/// # use futures::TryStreamExt;
/// # use lance::{Dataset, Result};
/// use lance::blob::reference::{BlobRef, BlobRefBinding, BlobRefDemand,
///     BlobRefReader, BlobRefRequest, BlobSourceContext, BlobMaterializationContext};
/// # async fn example(dataset: Arc<Dataset>, source: BlobSourceContext,
/// #     reference: Option<BlobRef>) -> Result<()> {
/// let binding = BlobRefBinding::try_new(source, dataset)?;
/// let budget = BlobMaterializationContext::try_new(1 << 20, 8 << 20)?;
/// let reader = BlobRefReader::try_new(vec![binding], budget)?;
/// let mut stream = reader.read([BlobRefRequest::new(42, reference, BlobRefDemand::Whole)])?;
/// while let Some(result) = stream.try_next().await? {
///     assert_eq!(result.request_index, 42);
///     // Consume result.data here; dropping it releases budget capacity.
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct BlobRefReader {
    inner: Arc<ReaderInner>,
}

struct ReaderInner {
    bindings: Vec<BlobRefBinding>,
    context: Arc<BlobMaterializationContext>,
    io_limit: u64,
    materialization_limit: u64,
    parallelism: usize,
    objects: Mutex<HashMap<ObjectKey, Arc<OnceCell<CachedObject>>>>,
    metrics: ReaderMetrics,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ObjectKey {
    slot: usize,
    base_id: Option<u32>,
    origin: i32,
    uri: String,
}

struct CachedObject {
    source: Arc<BlobSource>,
    size: OnceCell<u64>,
}

struct BoundLocator {
    slot: usize,
    object: pb::ObjectLocator,
    path: Path,
}

impl BoundLocator {
    fn key(&self) -> ObjectKey {
        ObjectKey {
            slot: self.slot,
            base_id: self.object.base_id,
            origin: self.object.origin,
            uri: self.object.object_uri.clone(),
        }
    }
}

struct PreparedRequest {
    request_index: u64,
    locator: Option<BoundLocator>,
    demand: BlobRefDemand,
}

impl BlobRefReader {
    /// Snapshot cumulative session I/O/demand counters and shared reservation peaks.
    /// See [`BlobRefMetrics`] for counter scope and cancellation semantics.
    pub fn metrics(&self) -> Result<BlobRefMetrics> {
        let mut metrics = self.inner.metrics.snapshot();
        metrics.reserved_bytes = self.inner.context.reserved_bytes()?;
        metrics.io_reserved_bytes = self.inner.context.io_reserved_bytes()?;
        (metrics.peak_reserved_bytes, metrics.peak_io_reserved_bytes) =
            self.inner.context.reservation_peaks()?;
        Ok(metrics)
    }

    /// Create a reader sharing the supplied finite budgets and schedulers.
    /// Bindings must be unique and follow the transported context table's order.
    pub fn try_new(
        bindings: Vec<BlobRefBinding>,
        context: Arc<BlobMaterializationContext>,
    ) -> Result<Self> {
        let io_limit = context.io_budget.as_ref().map(|budget| budget.limit);
        let materialization_limit = context.budget.as_ref().map(|budget| budget.limit);
        let (Some(io_limit), Some(materialization_limit)) = (io_limit, materialization_limit)
        else {
            return Err(Error::invalid_input(
                "Blob reference readers require finite budgets; use BlobMaterializationContext::try_new",
            ));
        };
        if io_limit == 0 || materialization_limit == 0 {
            return Err(Error::invalid_input(
                "Blob reference reader budgets must be nonzero",
            ));
        }
        let mut sources = HashSet::new();
        for binding in &bindings {
            if !sources.insert(*binding.context.source_id()) {
                return Err(Error::invalid_input(
                    "Duplicate blob reference source bindings",
                ));
            }
        }
        let parallelism = bindings
            .iter()
            .map(|binding| binding.dataset.object_store.io_parallelism())
            .max()
            .unwrap_or(1)
            .max(1);
        Ok(Self {
            inner: Arc::new(ReaderInner {
                bindings,
                context,
                io_limit,
                materialization_limit,
                parallelism,
                objects: Mutex::new(HashMap::new()),
                metrics: ReaderMetrics::default(),
            }),
        })
    }

    /// Validate demands and return a lazy, cancellable stream.
    ///
    /// Known lengths and arithmetic are checked before any I/O. Unknown external
    /// lengths use one cached metadata lookup per object. Planning retains
    /// `O(request_count)` metadata. Physical batches charge their coalesced ranges
    /// plus logical result lengths, counting duplicates separately. One oversized
    /// demand runs alone. Polling the next batch can wait for earlier results to
    /// be dropped; collecting the whole stream while retaining all results can
    /// therefore block when the shared budget fills.
    pub fn read(
        &self,
        requests: impl IntoIterator<Item = BlobRefRequest>,
    ) -> Result<BlobRefReadStream> {
        let mut indices = RoaringTreemap::new();
        let mut prepared = Vec::new();
        for request in requests {
            if !indices.insert(request.request_index) {
                return Err(Error::invalid_input(format!(
                    "Duplicate blob request_index {}",
                    request.request_index
                )));
            }
            if let BlobRefDemand::Range(range) = request.demand {
                range.offset.checked_add(range.length).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Blob reference request {} range offset {} + length {} overflows u64",
                        request.request_index, range.offset, range.length
                    ))
                })?;
            }
            let locator = if let Some(reference) = request.reference {
                let slot = reference.source_context() as usize;
                let binding = self.inner.bindings.get(slot).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Blob reference request {} uses missing source_context slot {slot}",
                        request.request_index
                    ))
                })?;
                let validate = || -> Result<BoundLocator> {
                    let object = reference.object_locator_in_context(&binding.context)?;
                    if let BlobRefDemand::Range(range) = request.demand {
                        reference.validate_range(range.offset, range.length)?;
                    }
                    let path = binding.object_path(&object)?;
                    Ok(BoundLocator { slot, object, path })
                };
                Some(validate().map_err(|error| {
                    Error::invalid_input(format!(
                        "Blob reference request {} source_context {slot} ({}): {error}",
                        request.request_index,
                        binding.context.table_identity(),
                    ))
                })?)
            } else {
                None
            };
            prepared.push(PreparedRequest {
                request_index: request.request_index,
                locator,
                demand: request.demand,
            });
        }
        let outstanding = (0..prepared.len() as u64).collect();
        let state = ReadState {
            reader: self.clone(),
            requests: prepared,
            outstanding,
            batches: None,
            ready: VecDeque::new(),
        };
        let cancellation = CancellationToken::new();
        let token = cancellation.clone();
        let inner = stream::try_unfold(state, move |state| {
            let token = token.clone();
            let label = format!("{} ({} requests incomplete)", state.pending_label(), state.outstanding.len());
            async move {
                if state.outstanding.is_empty() {
                    return state.next().await;
                }
                tokio::select! {
                    biased;
                    _ = token.cancelled() => Err(Error::io(format!("{label}: blob reference read cancelled"))),
                    result = state.next() => result,
                }
            }
        }).fuse().boxed();
        Ok(BlobRefReadStream {
            inner,
            cancellation,
        })
    }

    async fn object(&self, locator: &BoundLocator) -> Result<Arc<OnceCell<CachedObject>>> {
        let cell = self
            .inner
            .objects
            .lock()
            .map_err(|_| Error::internal("Blob reference object cache lock was poisoned"))?
            .entry(locator.key())
            .or_default()
            .clone();
        cell.get_or_try_init(|| async {
            let binding = &self.inner.bindings[locator.slot];
            let store = if locator.object.base_id.is_some()
                || locator.object.origin == pb::object_locator::Origin::Lance as i32
            {
                binding.dataset.object_store(locator.object.base_id).await?
            } else {
                let params = binding
                    .dataset
                    .store_params
                    .as_deref()
                    .cloned()
                    .unwrap_or_default();
                ObjectStore::from_uri_and_params(
                    binding.dataset.session.store_registry(),
                    &locator.object.object_uri,
                    &params,
                )
                .await?
                .0
            };
            Ok::<_, Error>(CachedObject {
                source: Arc::new(BlobSource::new(store, locator.path.clone())),
                size: OnceCell::new(),
            })
        })
        .await?;
        Ok(cell)
    }
}

struct ReadState {
    reader: BlobRefReader,
    requests: Vec<PreparedRequest>,
    outstanding: RoaringTreemap,
    batches: Option<VecDeque<BlobReadBatch>>,
    ready: VecDeque<(usize, Option<Bytes>)>,
}

impl ReadState {
    fn label(&self, index: usize) -> String {
        let request = &self.requests[index];
        match &request.locator {
            Some(locator) => format!(
                "Blob reference request {} source_context {} ({})",
                request.request_index,
                locator.slot,
                self.reader.inner.bindings[locator.slot]
                    .context
                    .table_identity()
            ),
            None => format!(
                "Blob reference request {} (SQL null)",
                request.request_index
            ),
        }
    }

    fn pending_label(&self) -> String {
        self.outstanding
            .min()
            .map(|index| self.label(index as usize))
            .unwrap_or_else(|| "Blob reference read".to_string())
    }

    async fn initialize(&mut self) -> Result<()> {
        let mut pending = Vec::with_capacity(self.requests.len());
        let mut objects = HashMap::new();
        for (index, request) in self.requests.iter().enumerate() {
            let Some(locator) = &request.locator else {
                self.ready.push_back((index, None));
                continue;
            };
            let empty = match request.demand {
                BlobRefDemand::Whole => locator.object.length == Some(0),
                BlobRefDemand::Range(range) => range.length == 0 && locator.object.length.is_some(),
            };
            if empty {
                self.ready.push_back((index, Some(Bytes::new())));
                continue;
            }
            pending.push((index, locator));
            // Prefer an unknown-length demand so the single lookup for this
            // object also initializes its cached size, regardless of input order.
            objects
                .entry(locator.key())
                .and_modify(|object_index| {
                    if locator.object.length.is_none() {
                        *object_index = index;
                    }
                })
                .or_insert(index);
        }
        // Deduplicate before limiting concurrency: repeated demands must not fill
        // the window with waiters for the same OnceCell and serialize other objects.
        let state = &*self;
        let objects =
            stream::iter(objects)
                .map(|(key, index)| async move {
                    let locator = state.requests[index].locator.as_ref().ok_or_else(|| {
                        Error::internal("Metadata lookup requires a blob locator")
                    })?;
                    let cell =
                        state.reader.object(locator).await.map_err(|error| {
                            Error::io(format!("{}: {error}", state.label(index)))
                        })?;
                    if locator.object.length.is_none() {
                        let cached = cell.get().ok_or_else(|| {
                            Error::internal("Resolved blob reference object was not cached")
                        })?;
                        cached
                            .size
                            .get_or_try_init(|| async {
                                state
                                    .reader
                                    .inner
                                    .metrics
                                    .metadata_requests
                                    .fetch_add(1, Ordering::Relaxed);
                                cached.source.object_store.size(&cached.source.path).await
                            })
                            .await
                            .map_err(|error| {
                                Error::io(format!(
                                    "{}: metadata lookup failed: {error}",
                                    state.label(index)
                                ))
                            })?;
                    }
                    Ok::<_, Error>((key, cell))
                })
                .buffer_unordered(self.reader.inner.parallelism)
                .try_collect::<HashMap<_, _>>()
                .await?;
        let mut entries = Vec::with_capacity(pending.len());
        let mut logical_bytes = 0_u64;
        let mut duplicates = 0_u64;
        let mut demands = HashSet::new();
        for (index, locator) in pending {
            let request = &self.requests[index];
            let cell = objects
                .get(&locator.key())
                .ok_or_else(|| Error::internal("Blob reference object was not resolved"))?;
            let cached = cell
                .get()
                .ok_or_else(|| Error::internal("Resolved blob reference object was not cached"))?;
            let size = match locator.object.length {
                Some(size) => size,
                None => {
                    let object_size = *cached.size.get().ok_or_else(|| {
                        Error::internal("Blob reference object size was not resolved")
                    })?;
                    object_size
                        .checked_sub(locator.object.offset)
                        .ok_or_else(|| {
                            Error::invalid_input(format!(
                                "{}: object offset {} exceeds object length {object_size}",
                                self.label(index),
                                locator.object.offset,
                            ))
                        })?
                }
            };
            if let BlobRefDemand::Range(range) = request.demand {
                let end = range.offset.checked_add(range.length).ok_or_else(|| {
                    Error::invalid_input(format!("{}: range overflows u64", self.label(index)))
                })?;
                if end > size {
                    return Err(Error::invalid_input(format!(
                        "{}: blob-local range {}..{end} exceeds length {size}",
                        self.label(index),
                        range.offset
                    )));
                }
            }
            let (offset, length) = match request.demand {
                BlobRefDemand::Whole => (locator.object.offset, size),
                BlobRefDemand::Range(range) => (locator.object.offset + range.offset, range.length),
            };
            logical_bytes = logical_bytes.checked_add(length).ok_or_else(|| {
                Error::invalid_input("Blob reference logical demand bytes overflow u64")
            })?;
            if length > 0 && !demands.insert((locator.key(), offset, length)) {
                duplicates += 1;
            }
            let kind = if locator.object.origin == pb::object_locator::Origin::Lance as i32 {
                BlobKind::Managed
            } else {
                BlobKind::External
            };
            entries.push(BlobEntry {
                selection_index: index,
                row_address: 0,
                file: BlobFile::with_source(
                    cached.source.clone(),
                    locator.object.offset,
                    size,
                    kind,
                    None,
                ),
                requested_range: match request.demand {
                    BlobRefDemand::Whole => None,
                    BlobRefDemand::Range(range) => Some(range),
                },
            });
        }
        self.batches = Some(
            plan_blob_read_batches_with_limits(
                entries,
                self.reader.inner.io_limit,
                Some(self.reader.inner.materialization_limit),
            )
            .map_err(|error| {
                Error::invalid_input(format!(
                    "{}: planning failed: {error}",
                    self.pending_label()
                ))
            })?
            .into(),
        );
        self.reader
            .inner
            .metrics
            .requested_logical_bytes
            .fetch_add(logical_bytes, Ordering::Relaxed);
        self.reader
            .inner
            .metrics
            .duplicate_demands
            .fetch_add(duplicates, Ordering::Relaxed);
        Ok(())
    }

    async fn next(mut self) -> Result<Option<(ReadBlobRef, Self)>> {
        if self.batches.is_none() {
            self.initialize().await?;
        }
        loop {
            if let Some((index, data)) = self.ready.pop_front() {
                if !self.outstanding.remove(index as u64) {
                    return Err(Error::internal(format!(
                        "{}: duplicate result",
                        self.label(index)
                    )));
                }
                self.reader.inner.metrics.returned_logical_bytes.fetch_add(
                    data.as_ref().map_or(0, |data| data.len() as u64),
                    Ordering::Relaxed,
                );
                return Ok(Some((
                    ReadBlobRef {
                        request_index: self.requests[index].request_index,
                        data,
                    },
                    self,
                )));
            }
            let batch = self.batches.as_mut().and_then(VecDeque::pop_front);
            let Some(batch) = batch else {
                return if self.outstanding.is_empty() {
                    Ok(None)
                } else {
                    Err(Error::internal(format!(
                        "{}: stream ended with {} incomplete requests",
                        self.pending_label(),
                        self.outstanding.len()
                    )))
                };
            };
            let context = &self.reader.inner.context;
            let reservation = context
                .admission()
                .reserve(batch.materialized_bytes)
                .await
                .map(Arc::new);
            let io_budget = context
                .io_budget
                .as_ref()
                .ok_or_else(|| Error::internal("Blob reference reader lost its I/O budget"))?;
            let io_reservation = io_budget
                .admission()
                .reserve(batch.physical_bytes)
                .await
                .map(Arc::new);
            let execution = Arc::new(ReadBlobsExecution {
                io_buffer_size_bytes: context.execution.io_buffer_size_bytes,
                schedulers: context.execution.schedulers.clone(),
                reservation: Some(Arc::new(ReadReservation {
                    _materialization: reservation.clone(),
                    _io: io_reservation,
                })),
                io_stats: Some(self.reader.inner.metrics.payload.clone()),
            });
            self.reader
                .inner
                .metrics
                .coalesced_bytes
                .fetch_add(batch.physical_bytes, Ordering::Relaxed);
            let incomplete = self.outstanding.len();
            let blobs = stream::iter(batch.plans.into_iter().map(|plan| {
                let label = plan
                    .reads
                    .first()
                    .map(|read| self.label(read.selection_index));
                let execution = execution.clone();
                async move {
                    let label = label.ok_or_else(|| {
                        Error::internal("Blob reference planner produced an empty source plan")
                    })?;
                    execute_blob_read_plan(plan, execution)
                        .await
                        .map_err(|error| {
                            Error::io(format!("{label}: payload read failed ({incomplete} requests incomplete): {error}"))
                        })
                }
            }))
            .buffer_unordered(self.reader.inner.parallelism)
            .try_collect::<Vec<_>>()
            .await?;
            drop(execution);
            for blob in blobs.into_iter().flatten() {
                let data = Bytes::from_owner(ReservedBytes {
                    data: blob.data,
                    _reservation: reservation.clone(),
                });
                self.ready.push_back((blob.selection_index, Some(data)));
            }
        }
    }
}

struct ReservedBytes {
    data: Bytes,
    _reservation: Option<Arc<BlobMaterializationReservation>>,
}

impl AsRef<[u8]> for ReservedBytes {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

/// Scheduled reader futures retain both reservations even if the consumer drops
/// its stream while backend I/O is still in flight.
#[derive(Debug)]
pub(super) struct ReadReservation {
    _materialization: Option<Arc<BlobMaterializationReservation>>,
    _io: Option<Arc<BlobMaterializationReservation>>,
}

#[derive(Debug)]
pub(super) struct ReservedReader {
    pub(super) reader: Arc<dyn Reader>,
    pub(super) reservation: Arc<ReadReservation>,
}

impl lance_core::deepsize::DeepSizeOf for ReservedReader {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.reader.deep_size_of_children(context)
    }
}

impl Reader for ReservedReader {
    fn path(&self) -> &Path {
        self.reader.path()
    }
    fn block_size(&self) -> usize {
        self.reader.block_size()
    }
    fn io_parallelism(&self) -> usize {
        self.reader.io_parallelism()
    }
    fn size(&self) -> futures::future::BoxFuture<'_, object_store::Result<usize>> {
        self.reader.size()
    }

    fn get_range(
        &self,
        range: std::ops::Range<usize>,
    ) -> futures::future::BoxFuture<'static, object_store::Result<Bytes>> {
        let future = self.reader.get_range(range);
        let reservation = self.reservation.clone();
        Box::pin(async move {
            let result = future.await;
            drop(reservation);
            result
        })
    }

    fn get_all(&self) -> futures::future::BoxFuture<'_, object_store::Result<Bytes>> {
        self.reader.get_all()
    }
}
