// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Scheduling of multi-partition IVF searches.
//!
//! [`IVFIndex`](super::IVFIndex) is compiled once per sub-index × quantizer
//! pair, but the work of a multi-partition search that does not touch vectors
//! (prepare windows, chunking, channels, CPU-pool dispatch, early stop) is the
//! same for all of them. It is compiled once here, against
//! [`PartitionSearchSource`]. The per-partition scoring behind
//! [`PreparedPartition`] and [`BatchPartition`] stays monomorphized; the
//! boundary costs one virtual call per probed partition, not per vector.

use std::any::Any;
use std::collections::{BinaryHeap, HashMap};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{ArrayRef, Float32Array, RecordBatch, UInt32Array};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::future::BoxFuture;
use futures::{Stream, StreamExt, stream};
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::{Error, Result};
use lance_index::metrics::{IndexTimer, IndexTiming, MetricsCollector};
use lance_index::vector::graph::OrderedNode;
use lance_index::vector::storage::{QueryScratch, QueryScratchPool, RabitRawQueryContext};
use lance_index::vector::{PartitionSearchControl, Query, VECTOR_RESULT_SCHEMA};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::rabit_scan::{self, RabitIndex, RabitScan, RabitScanContext};
use super::{
    GLOBAL_TOPK_CHUNK_BYTES, GLOBAL_TOPK_CHUNK_MAX_PARTITIONS, GLOBAL_TOPK_INLINE_HEAP_LEN,
    STREAMING_SEARCH_BATCH_SIZE, global_heap_to_batch,
};
use crate::index::PreFilter;

/// A partition loaded and filtered for one query, ready to score on the CPU
/// pool. It pins the partition's storage until it is scored or dropped.
pub(super) trait PreparedPartition: Send {
    /// Bytes of partition storage this value keeps alive.
    fn size_bytes(&self) -> usize;

    /// The partition's top `k * refine_factor` rows for the query.
    fn search(
        self: Box<Self>,
        metrics: &dyn MetricsCollector,
        scratch: &mut QueryScratch,
    ) -> Result<RecordBatch>;

    /// Push the partition's candidates into a heap shared across partitions.
    fn accumulate_topk(
        self: Box<Self>,
        heap: &mut BinaryHeap<OrderedNode<u64>>,
        scratch: &mut QueryScratch,
        metrics: &dyn MetricsCollector,
    ) -> Result<()>;
}

/// A partition loaded once for a batch of queries; each probing query scores
/// it into its own heap.
pub(super) trait BatchPartition: Send + Sync {
    fn accumulate_topk(
        &self,
        query: &Arc<Query>,
        dist_q_c: f32,
        raw_query_context: Option<&Arc<RabitRawQueryContext>>,
        heap: &mut BinaryHeap<OrderedNode<u64>>,
        scratch: &mut QueryScratch,
        metrics: &dyn MetricsCollector,
    ) -> Result<()>;
}

/// What the search schedulers need from an IVF index.
pub(super) trait PartitionSearchSource: Send + Sync + 'static {
    /// Load and filter one partition for `query`. With `wait_for_prefilter`
    /// the prefilter wait is overlapped with the partition load; without it
    /// the caller must have waited already.
    #[allow(clippy::too_many_arguments)]
    fn prepare_scored_partition(
        self: Arc<Self>,
        partition_id: usize,
        query: Arc<Query>,
        dist_q_c: f32,
        pre_filter: Arc<dyn PreFilter>,
        metrics: Arc<dyn MetricsCollector>,
        raw_query_context: Option<Arc<RabitRawQueryContext>>,
        wait_for_prefilter: bool,
    ) -> BoxFuture<'static, Result<Box<dyn PreparedPartition>>>;

    /// Load and filter one partition for a batch of queries. The prefilter
    /// must be ready.
    fn load_batch_partition(
        self: Arc<Self>,
        partition_id: usize,
        pre_filter: Arc<dyn PreFilter>,
        metrics: Arc<dyn MetricsCollector>,
    ) -> BoxFuture<'static, Result<Box<dyn BatchPartition>>>;

    fn raw_query_context(&self, query: &ArrayRef) -> Result<Option<Arc<RabitRawQueryContext>>>;

    fn scratch_pool(&self) -> Arc<QueryScratchPool>;
}

/// The native RaBitQ scan an IVF_RQ index (flat sub-index) may delegate to.
pub(super) struct NativeRabitScan {
    pub(super) index: Arc<dyn Any + Send + Sync>,
    pub(super) strategy: Arc<dyn RabitScan>,
}

/// The arguments of [`VectorIndex::search_partitions`](crate::index::vector::VectorIndex::search_partitions).
pub(super) struct PartitionSearchRequest {
    pub(super) query: Query,
    pub(super) partitions: Arc<UInt32Array>,
    pub(super) q_c_dists: Arc<Float32Array>,
    pub(super) start_idx: usize,
    pub(super) end_idx: usize,
    pub(super) pre_filter: Arc<dyn PreFilter>,
    pub(super) control: Option<Arc<dyn PartitionSearchControl>>,
    pub(super) metrics: Arc<dyn MetricsCollector>,
}

/// Pull prepared partitions off `prepared` until their pinned storage reaches
/// `chunk_bytes` or the chunk holds [`GLOBAL_TOPK_CHUNK_MAX_PARTITIONS`],
/// always taking at least one. `None` once the stream is exhausted; a failed
/// prepare ends the chunk (and the search) immediately.
async fn next_scoring_chunk<St>(
    prepared: &mut St,
    chunk_bytes: usize,
) -> Option<Result<Vec<Box<dyn PreparedPartition>>>>
where
    St: Stream<Item = Result<Box<dyn PreparedPartition>>> + Unpin,
{
    let mut chunk = Vec::new();
    let mut bytes = 0;
    while bytes < chunk_bytes && chunk.len() < GLOBAL_TOPK_CHUNK_MAX_PARTITIONS {
        match prepared.next().await {
            Some(Ok(prepared)) => {
                bytes += prepared.size_bytes();
                chunk.push(prepared);
            }
            Some(Err(err)) => return Some(Err(err)),
            None => break,
        }
    }
    if chunk.is_empty() {
        None
    } else {
        Some(Ok(chunk))
    }
}

pub(super) async fn search_partitions(
    source: Arc<dyn PartitionSearchSource>,
    supports_global_topk_heap: bool,
    native_rabit_scan: Option<NativeRabitScan>,
    request: PartitionSearchRequest,
) -> Result<SendableRecordBatchStream> {
    let PartitionSearchRequest {
        query,
        partitions,
        q_c_dists,
        start_idx,
        end_idx,
        pre_filter,
        control,
        metrics,
    } = request;
    if partitions.len() != q_c_dists.len() {
        return Err(Error::invalid_input(format!(
            "partition count {} does not match centroid distance count {}",
            partitions.len(),
            q_c_dists.len()
        )));
    }
    if start_idx > end_idx || end_idx > partitions.len() {
        return Err(Error::invalid_input(format!(
            "invalid partition search range [{start_idx}, {end_idx}) for {} partitions",
            partitions.len()
        )));
    }

    let prepare_parallelism = get_num_compute_intensive_cpus().max(1);
    let raw_query_context = source.raw_query_context(&query.key)?;

    if supports_global_topk_heap
        && let Some(NativeRabitScan { index, strategy }) = native_rabit_scan
        && let Ok(index) = index.downcast::<RabitIndex>()
    {
        let context = Arc::new(RabitScanContext::new(
            index,
            raw_query_context.clone(),
            query.clone(),
            partitions.clone(),
            q_c_dists.clone(),
            start_idx..end_idx,
            pre_filter.clone(),
            metrics.clone(),
        ));
        if let Some(control) = &control {
            if control.should_stop() {
                return Ok(Box::pin(RecordBatchStreamAdapter::new(
                    VECTOR_RESULT_SCHEMA.clone(),
                    stream::empty(),
                )));
            }
            let factor = query.refine_factor.unwrap_or(1);
            if factor == 0 {
                return Err(Error::invalid_input(
                    "refine_factor must be positive, got 0",
                ));
            }
            let candidate_limit = query.k.checked_mul(factor as usize).ok_or_else(|| {
                Error::invalid_input(format!(
                    "RaBitQ candidate count overflow: k={}, refine_factor={factor}",
                    query.k
                ))
            })?;
            if let Some(batches) = strategy.search_partition_stream(context)? {
                return rabit_scan::controlled_partition_stream(
                    batches,
                    end_idx - start_idx,
                    candidate_limit,
                    control.clone(),
                );
            }
        } else if let Some(batch) = strategy.search(context).await? {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                VECTOR_RESULT_SCHEMA.clone(),
                stream::once(async move { Ok(batch) }),
            )));
        }
    }
    if control.is_none() && supports_global_topk_heap {
        let heap_capacity = query.k * query.refine_factor.unwrap_or(1) as usize;
        {
            let _wait_timer = IndexTimer::new(metrics.as_ref(), IndexTiming::PrefilterWait);
            pre_filter.wait_for_ready().await?;
        }
        let prepare_source = source.clone();
        let prepare_metrics = metrics.clone();
        let prepare_raw_query_context = raw_query_context.clone();
        // Stream prepared partitions through scoring in chunks rather than
        // collecting all of them first. A prepared partition pins its whole
        // quantized storage, so collecting `nprobes` of them before scoring
        // makes peak memory scale with `nprobes` (a 4096-probe query over a
        // large RQ index pinned hundreds of GiB per index segment). Chunking
        // bounds resident partitions to the prepare window plus two chunks:
        // one being assembled by `next_scoring_chunk` while another is scored.
        // `buffered` preserves the probe order, so the heap accumulates
        // partitions in the same order as before (which decides which of
        // several rows tied at the k-th distance the capped heap keeps).
        let shared_query = Arc::new(query.clone());
        let mut prepared = stream::iter(start_idx..end_idx)
            .map(move |idx| {
                prepare_source.clone().prepare_scored_partition(
                    partitions.value(idx) as usize,
                    shared_query.clone(),
                    q_c_dists.value(idx),
                    pre_filter.clone(),
                    prepare_metrics.clone(),
                    prepare_raw_query_context.clone(),
                    false,
                )
            })
            .buffered(prepare_parallelism)
            .fuse();
        let chunk_bytes = *GLOBAL_TOPK_CHUNK_BYTES;

        let mut heap = BinaryHeap::with_capacity(heap_capacity);
        // Score each chunk on the CPU pool while the next chunk prepares (the
        // same overlap `search_partitions_batch` uses): `spawn_cpu` starts the
        // scoring immediately and the async task, never a CPU-pool thread, does
        // the waiting (#7642). The heap is threaded through each dispatch so
        // scoring stays sequential, and a scored chunk is dropped before the
        // next one is scored.
        let mut pending = next_scoring_chunk(&mut prepared, chunk_bytes).await;
        while let Some(chunk) = pending {
            let chunk = chunk?;
            let search_metrics = metrics.clone();
            let scratch_pool = source.scratch_pool();
            let queued = Instant::now();
            let score = spawn_cpu(move || -> Result<BinaryHeap<OrderedNode<u64>>> {
                search_metrics.record_timing(IndexTiming::CpuQueueWait, queued.elapsed());
                scratch_pool.with_scratch(|scratch| -> Result<()> {
                    for prepared in chunk {
                        prepared.accumulate_topk(&mut heap, scratch, search_metrics.as_ref())?;
                    }
                    Ok(())
                })?;
                Ok(heap)
            });
            let (scored, next) =
                futures::join!(score, next_scoring_chunk(&mut prepared, chunk_bytes));
            heap = scored?;
            pending = next;
        }

        // Turning the heap into the result batch is a sort of `k * refine_factor`
        // entries. Below a few thousand that is well under the ~100µs where a
        // `spawn_cpu` dispatch pays for itself, so do it inline and keep the
        // common small-k query at one dispatch (as before the chunked scoring).
        let batch = if heap.len() <= GLOBAL_TOPK_INLINE_HEAP_LEN {
            global_heap_to_batch(heap, metrics.as_ref())?
        } else {
            let result_metrics = metrics.clone();
            let queued = Instant::now();
            spawn_cpu(move || {
                result_metrics.record_timing(IndexTiming::CpuQueueWait, queued.elapsed());
                global_heap_to_batch(heap, result_metrics.as_ref())
            })
            .await?
        };

        return Ok(Box::pin(RecordBatchStreamAdapter::new(
            VECTOR_RESULT_SCHEMA.clone(),
            stream::once(async move { Ok(batch) }),
        )));
    }

    // The prepared channel holds a full search batch so that partitions prepared
    // while the previous batch is being searched are ready for the next greedy
    // drain, instead of serializing producer and consumer through a single slot.
    let (prepared_tx, mut prepared_rx) =
        mpsc::channel::<Result<Box<dyn PreparedPartition>>>(*STREAMING_SEARCH_BATCH_SIZE);
    let (batch_tx, batch_rx) = mpsc::channel::<DataFusionResult<RecordBatch>>(1);

    let prepare_source = source.clone();
    let prepare_metrics = metrics.clone();
    let prepare_raw_query_context = raw_query_context.clone();
    lance_core::utils::tokio::spawn(async move {
        let shared_query = Arc::new(query.clone());
        let prepare_stream = stream::iter(start_idx..end_idx)
            .map(move |idx| {
                prepare_source.clone().prepare_scored_partition(
                    partitions.value(idx) as usize,
                    shared_query.clone(),
                    q_c_dists.value(idx),
                    pre_filter.clone(),
                    prepare_metrics.clone(),
                    prepare_raw_query_context.clone(),
                    true,
                )
            })
            .buffered(prepare_parallelism);

        futures::pin_mut!(prepare_stream);
        while let Some(prepared) = prepare_stream.next().await {
            let has_error = prepared.is_err();
            if prepared_tx.send(prepared).await.is_err() || has_error {
                break;
            }
        }
    });

    let search_metrics = metrics.clone();
    let search_control = control.clone();
    let scratch_pool = source.scratch_pool();
    // Search prepared partitions in batches. Each batch is searched in a single
    // `spawn_cpu` dispatch (amortizing the per-dispatch overhead the single-worker
    // design in #6475 avoided), but the channel `recv`/`send` stay in async code so
    // no CPU-pool thread ever parks on a channel — parking one can deadlock the pool
    // on small hosts (#7642). `should_stop` is checked per partition, so early-stop
    // granularity is unchanged.
    //
    // Batches are formed greedily: wait for one prepared partition, then drain
    // whatever else is already prepared, up to the batch size. Waiting for a full
    // batch instead would delay the first search (and the early-stop feedback it
    // produces) behind up to a whole batch of prepare I/O, which is significant
    // when prepare parallelism is low.
    lance_core::utils::tokio::spawn(async move {
        loop {
            // Stop pulling as soon as the search is done — or the receiver of our
            // results is gone — so the producer stops preparing partitions we
            // would never search.
            if search_control
                .as_ref()
                .is_some_and(|control| control.should_stop())
                || batch_tx.is_closed()
            {
                return;
            }

            let mut prepared_batch = Vec::with_capacity(*STREAMING_SEARCH_BATCH_SIZE);
            let mut prepare_error = None;
            let mut producer_done = false;
            match prepared_rx.recv().await {
                Some(Ok(prepared)) => prepared_batch.push(prepared),
                Some(Err(err)) => prepare_error = Some(DataFusionError::from(err)),
                None => producer_done = true,
            }
            while prepare_error.is_none()
                && !producer_done
                && prepared_batch.len() < *STREAMING_SEARCH_BATCH_SIZE
            {
                match prepared_rx.try_recv() {
                    Ok(Ok(prepared)) => prepared_batch.push(prepared),
                    Ok(Err(err)) => {
                        prepare_error = Some(DataFusionError::from(err));
                    }
                    // Nothing else is prepared yet; search what we have rather
                    // than waiting for more.
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        producer_done = true;
                    }
                }
            }

            if !prepared_batch.is_empty() {
                let scratch_pool = scratch_pool.clone();
                let search_metrics = search_metrics.clone();
                let search_control = search_control.clone();
                // `is_closed` is synchronously callable, so a sender clone lets the
                // CPU loop notice a dropped receiver between partitions instead of
                // searching out the whole batch for a cancelled query. (A `select!`
                // on `closed()` would not help here: `spawn_cpu` closures are not
                // cancellable, so abandoning the await leaves the work running.)
                let cancel_probe = batch_tx.clone();
                let queued = Instant::now();
                let search_output = spawn_cpu(move || {
                    search_metrics.record_timing(IndexTiming::CpuQueueWait, queued.elapsed());
                    let mut outputs: Vec<DataFusionResult<RecordBatch>> =
                        Vec::with_capacity(prepared_batch.len());
                    // `stopped` means the whole search should end (an error, an
                    // early-stop signal, or cancellation), not just this batch.
                    let mut stopped = false;
                    scratch_pool.with_scratch(|scratch| {
                        for prepared in prepared_batch {
                            if search_control
                                .as_ref()
                                .is_some_and(|control| control.should_stop())
                                || cancel_probe.is_closed()
                            {
                                stopped = true;
                                break;
                            }
                            match prepared
                                .search(search_metrics.as_ref(), scratch)
                                .map_err(DataFusionError::from)
                            {
                                Ok(batch) => {
                                    if let Some(control) = search_control.as_ref() {
                                        control.record_batch(&batch);
                                    }
                                    outputs.push(Ok(batch));
                                }
                                Err(err) => {
                                    outputs.push(Err(err));
                                    stopped = true;
                                    break;
                                }
                            }
                        }
                    });
                    Ok::<_, DataFusionError>((outputs, stopped))
                })
                .await;

                let (outputs, stopped) = match search_output {
                    Ok(output) => output,
                    // Defensive: the closure always returns Ok (search errors are
                    // captured per partition in `outputs`), so this arm should be
                    // unreachable. Forward and stop rather than drop silently.
                    Err(err) => {
                        let _ = batch_tx.send(Err(err)).await;
                        return;
                    }
                };
                for output in outputs {
                    if batch_tx.send(output).await.is_err() {
                        return;
                    }
                }
                if stopped {
                    return;
                }
            }

            if let Some(err) = prepare_error {
                let _ = batch_tx.send(Err(err)).await;
                return;
            }
            if producer_done {
                return;
            }
        }
    });

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        VECTOR_RESULT_SCHEMA.clone(),
        ReceiverStream::new(batch_rx),
    )))
}

/// One top-k heap per query of a batch.
type QueryHeaps = Vec<BinaryHeap<OrderedNode<u64>>>;

/// Fewest (query, partition) probes a batch scoring task is given when a loaded
/// chunk is split across CPU tasks. A probe costs from about a microsecond (a
/// small flat partition) to far more (a high-dimensional PQ distance table), so
/// this keeps the cheapest tasks above the cost of dispatching them.
const MIN_PROBES_PER_SCORING_TASK: usize = 8;

/// Split `0..probes_per_query.len()` into at most `max_tasks` contiguous query
/// ranges with roughly equal numbers of probes, giving each range at least
/// [`MIN_PROBES_PER_SCORING_TASK`] probes where the chunk has enough. Ranges
/// without probes are omitted.
pub(super) fn balanced_query_ranges(
    probes_per_query: &[usize],
    max_tasks: usize,
) -> Vec<Range<usize>> {
    let total: usize = probes_per_query.iter().sum();
    let task_count = (total / MIN_PROBES_PER_SCORING_TASK).clamp(1, max_tasks.max(1));
    let target = total.div_ceil(task_count);
    let mut ranges = Vec::with_capacity(task_count);
    let mut start = 0;
    let mut probes = 0;
    for (query_index, &count) in probes_per_query.iter().enumerate() {
        probes += count;
        if probes >= target && ranges.len() + 1 < task_count {
            ranges.push(start..query_index + 1);
            start = query_index + 1;
            probes = 0;
        }
    }
    if probes > 0 {
        ranges.push(start..probes_per_query.len());
    }
    ranges
}

pub(super) async fn search_partitions_batch(
    source: Arc<dyn PartitionSearchSource>,
    query: Query,
    partitions_per_query: Vec<Arc<UInt32Array>>,
    q_c_dists_per_query: Vec<Arc<Float32Array>>,
    pre_filter: Arc<dyn PreFilter>,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<Vec<RecordBatch>> {
    let query_count = partitions_per_query.len();
    if q_c_dists_per_query.len() != query_count {
        return Err(Error::invalid_input(format!(
            "batch partition search: {query_count} query partition lists but {} distance lists",
            q_c_dists_per_query.len()
        )));
    }
    if query_count == 0 {
        return Ok(Vec::new());
    }
    if !query.key.len().is_multiple_of(query_count) {
        return Err(Error::invalid_input(format!(
            "batch partition search: query key length {} is not divisible by query count {query_count}",
            query.key.len()
        )));
    }
    let dim = query.key.len() / query_count;

    // Per-query immutable search state: the query vector slice and the
    // optional Rabit raw-query context both depend only on the query vector,
    // so compute them once up front rather than per probed partition.
    let mut base_queries = Vec::with_capacity(query_count);
    let mut raw_query_contexts = Vec::with_capacity(query_count);
    for query_index in 0..query_count {
        if partitions_per_query[query_index].len() != q_c_dists_per_query[query_index].len() {
            return Err(Error::invalid_input(format!(
                "batch partition search: query {query_index} has {} partitions but {} distances",
                partitions_per_query[query_index].len(),
                q_c_dists_per_query[query_index].len()
            )));
        }
        let mut single_query = query.clone();
        single_query.key = query.key.slice(query_index * dim, dim);
        raw_query_contexts.push(source.raw_query_context(&single_query.key)?);
        base_queries.push(single_query);
    }
    // Shared across every chunk's scoring dispatch below, so wrap once in an
    // `Arc` instead of cloning the whole `Vec` per chunk.
    let base_queries = base_queries.into_iter().map(Arc::new).collect::<Vec<_>>();
    let base_queries = Arc::new(base_queries);
    let raw_query_contexts = Arc::new(raw_query_contexts);

    // Invert the per-query partition lists so each distinct partition is
    // loaded once and scored against every query that probes it.
    let mut assignments: HashMap<u32, Vec<(usize, f32)>> = HashMap::new();
    for (query_index, (parts, dists)) in partitions_per_query
        .iter()
        .zip(q_c_dists_per_query.iter())
        .enumerate()
    {
        for (part_id, dist_q_c) in parts.values().iter().zip(dists.values().iter()) {
            assignments
                .entry(*part_id)
                .or_default()
                .push((query_index, *dist_q_c));
        }
    }

    {
        let _wait_timer = IndexTimer::new(metrics.as_ref(), IndexTiming::PrefilterWait);
        pre_filter.wait_for_ready().await?;
    }

    // Score partitions in a deterministic order. `assignments` is a HashMap,
    // so its iteration order (and hence the order partitions accumulate into
    // each per-query heap) is otherwise arbitrary. When several rows tie at
    // the k-th distance, which one the capped heap keeps depends on insertion
    // order, so a stable partition order is what makes the selected top-k
    // deterministic across runs. (Any of the tied rows is an equally valid
    // k-th neighbor, so this does not affect recall.)
    let mut assignment_list: Vec<(u32, Vec<(usize, f32)>)> = assignments.into_iter().collect();
    assignment_list.sort_by_key(|(part_id, _)| *part_id);

    // Load each distinct partition's storage exactly once (the shared I/O
    // that batch search exists to save), but *stream* the loaded partitions
    // through scoring in chunks rather than materializing them all. A wide
    // batch probes up to `min(query_count * nprobes, num_partitions)` distinct
    // partitions, so collecting every loaded partition before scoring would
    // make peak memory scale with the batch width — up to the whole index.
    // Streaming bounds resident partition storage to the load window plus one
    // chunk. `buffered` preserves the sorted load order above, so scoring order
    // (and thus the k-th-distance tie-break) stays deterministic.
    let load_parallelism = get_num_compute_intensive_cpus().max(1);
    let load_source = source.clone();
    let load_metrics = metrics.clone();
    let mut loaded_chunks = stream::iter(assignment_list)
        .map(move |(part_id, probing_queries)| {
            let loaded = load_source.clone().load_batch_partition(
                part_id as usize,
                pre_filter.clone(),
                load_metrics.clone(),
            );
            async move { Result::Ok((loaded.await?, probing_queries)) }
        })
        .buffered(load_parallelism)
        .chunks(*STREAMING_SEARCH_BATCH_SIZE);

    let heap_capacity = query.k * query.refine_factor.unwrap_or(1) as usize;
    let mut heaps: Vec<BinaryHeap<OrderedNode<u64>>> = (0..query_count)
        .map(|_| BinaryHeap::with_capacity(heap_capacity))
        .collect();

    // Score each chunk on the CPU pool while the next chunk loads. `spawn_cpu`
    // dispatches the scoring immediately and only touches CPU-bound state, so
    // `join!`-ing it with the next `loaded_chunks` pull keeps partition I/O in
    // flight during scoring: the async task, never a CPU-pool thread, does the
    // waiting (#7642), and the load stream is not paused (the pairing `spawn_cpu`'s
    // docs recommend with `buffered`). Scoring stays sequential across chunks —
    // each mutates the same per-query heaps — so a step costs about
    // max(load, score) rather than their sum, and a scored chunk's storage is
    // dropped before the next is scored, keeping peak memory bounded.
    let max_scoring_tasks = get_num_compute_intensive_cpus().max(1);
    let mut probes_per_query = vec![0; query_count];
    let mut pending = loaded_chunks.next().await;
    while let Some(chunk) = pending {
        let chunk = Arc::new(chunk.into_iter().collect::<Result<Vec<_>>>()?);
        // Score disjoint query ranges on separate CPU tasks. Each task owns
        // its queries' heaps, and every query still visits the chunk's
        // partitions in sorted order, so heap insertion order (and thus the
        // tie-break above) matches scoring the chunk on a single task.
        probes_per_query.fill(0);
        for (_, probing_queries) in chunk.iter() {
            for (query_index, _) in probing_queries {
                probes_per_query[*query_index] += 1;
            }
        }
        let ranges = balanced_query_ranges(&probes_per_query, max_scoring_tasks);
        let mut scoring = Vec::with_capacity(ranges.len());
        for range in ranges {
            let range_heaps: Vec<_> = heaps[range.clone()]
                .iter_mut()
                .map(std::mem::take)
                .collect();
            let chunk = chunk.clone();
            let base_queries = base_queries.clone();
            let raw_query_contexts = raw_query_contexts.clone();
            let scratch_pool = source.scratch_pool();
            let search_metrics = metrics.clone();
            let queued = Instant::now();
            scoring.push(spawn_cpu(move || -> Result<(Range<usize>, QueryHeaps)> {
                search_metrics.record_timing(IndexTiming::CpuQueueWait, queued.elapsed());
                let mut range_heaps = range_heaps;
                scratch_pool.with_scratch(|scratch| -> Result<()> {
                    for (partition, probing_queries) in chunk.iter() {
                        // Each partition's probing queries are in ascending order.
                        let start = probing_queries.partition_point(|(q, _)| *q < range.start);
                        let end = probing_queries.partition_point(|(q, _)| *q < range.end);
                        for (query_index, dist_q_c) in &probing_queries[start..end] {
                            partition.accumulate_topk(
                                &base_queries[*query_index],
                                *dist_q_c,
                                raw_query_contexts[*query_index].as_ref(),
                                &mut range_heaps[*query_index - range.start],
                                scratch,
                                search_metrics.as_ref(),
                            )?;
                        }
                    }
                    Ok(())
                })?;
                Ok((range, range_heaps))
            }));
        }
        // Load the next chunk while this one is scored on the CPU pool.
        let (scored, next) =
            futures::join!(futures::future::try_join_all(scoring), loaded_chunks.next());
        for (range, range_heaps) in scored? {
            for (heap, scored_heap) in heaps[range].iter_mut().zip(range_heaps) {
                *heap = scored_heap;
            }
        }
        pending = next;
    }

    heaps
        .into_iter()
        .map(|heap| global_heap_to_batch(heap, metrics.as_ref()))
        .collect::<Result<Vec<_>>>()
}
