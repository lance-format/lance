// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use datafusion::common::runtime::SpawnedTask;
use futures::{FutureExt, StreamExt, TryStreamExt, stream::BoxStream};
use lance_core::{Error, Result};
use lance_file::reader::ReadEstimate;
use tracing::Instrument;

use super::super::read_budget::{ChunkAdmission, ChunkSize, ReadChunk, ScanBudget};
use super::{
    BlobMaterializationContext, Dataset, FilteredReadGlobalMetrics, FilteredReadStream,
    MaterializedReadBatchesFut, OpenedFragmentRead, ScopedFragmentRead,
};

pub(super) type FragmentChunk = (Arc<OpenedFragmentRead>, ReadChunk);
pub(super) type FragmentChunkStream = BoxStream<'static, Result<FragmentChunk>>;

/// The chunks of a budgeted read that its partitions take from.
pub(super) struct SharedChunks {
    pub(super) chunks: Arc<ChunkQueue>,
    pub(super) scan_budget: ScanBudget,
    /// Chunks each partition reads ahead: the scan's read-ahead split among
    /// its partitions.
    pub(super) readahead: usize,
}

/// The chunks of a budgeted read, shared by its partitions through a bounded
/// channel, so a partition whose consumer stops polling holds no lock the
/// others need.
pub(super) struct ChunkQueue {
    receiver: async_channel::Receiver<Result<FragmentChunk>>,
    /// The chunks and their sender, until the first partition starts the
    /// producer task; the read is planned outside a runtime.
    unstarted: std::sync::Mutex<
        Option<(
            FragmentChunkStream,
            async_channel::Sender<Result<FragmentChunk>>,
        )>,
    >,
    producer: std::sync::OnceLock<SpawnedTask<()>>,
}

impl ChunkQueue {
    pub(super) fn new(chunks: FragmentChunkStream, capacity: usize) -> Arc<Self> {
        let (sender, receiver) = async_channel::bounded(capacity.max(1));
        Arc::new(Self {
            receiver,
            unstarted: std::sync::Mutex::new(Some((chunks, sender))),
            producer: Default::default(),
        })
    }

    /// The next chunk, or `None` once the read has none left. Shaped for
    /// [`futures::stream::unfold`].
    async fn pop(self: Arc<Self>) -> Option<(Result<FragmentChunk>, Arc<Self>)> {
        if let Some((mut chunks, sender)) = self.unstarted.lock().unwrap().take() {
            let _ = self.producer.set(SpawnedTask::spawn(async move {
                while let Some(chunk) = chunks.next().await {
                    if sender.send(chunk).await.is_err() {
                        return;
                    }
                }
            }));
        }
        let chunk = self.receiver.recv().await.ok()?;
        Some((chunk, self))
    }
}

impl FilteredReadStream {
    /// Opens the fragments and splits each one's ranges into chunks of about
    /// `chunk_size` of memory.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn budgeted_chunk_stream(
        dataset: Arc<Dataset>,
        scoped_fragments: Vec<ScopedFragmentRead>,
        global_metrics: Arc<FilteredReadGlobalMetrics>,
        fragment_soft_limit: Option<u64>,
        materialization_context: Arc<BlobMaterializationContext>,
        materialize_blob_v2_binary: bool,
        readahead: usize,
        chunk_size: ChunkSize,
    ) -> FragmentChunkStream {
        futures::stream::iter(scoped_fragments)
            .map(move |scoped_fragment| {
                let chunk_size = chunk_size.clone();
                let open = Self::open_fragment(
                    dataset.clone(),
                    scoped_fragment,
                    global_metrics.clone(),
                    fragment_soft_limit,
                    materialization_context.clone(),
                    materialize_blob_v2_binary,
                );
                SpawnedTask::spawn(
                    async move {
                        let fragment = open.await?;
                        let chunks = fragment.plan_chunks(&chunk_size).await?;
                        Ok::<_, Error>(futures::stream::iter(
                            chunks
                                .into_iter()
                                .map(move |chunk| Ok((fragment.clone(), chunk))),
                        ))
                    }
                    .in_current_span(),
                )
                .map(|result| result.map_err(Error::from)?)
            })
            .buffered(readahead)
            .try_flatten()
            .boxed()
    }

    /// Reads a chunk once the budget admits it, in its own task so chunks are
    /// set up in parallel.
    fn read_budgeted_chunk(
        admission: ChunkAdmission,
        chunk: Result<FragmentChunk>,
    ) -> impl Future<Output = Result<BoxStream<'static, Result<MaterializedReadBatchesFut>>>> {
        let read = async move {
            let (fragment, chunk) = chunk?;
            if fragment.reached_soft_limit() {
                return Ok::<_, Error>(futures::stream::empty().boxed());
            }
            let chunk_budget = admission.reserve(chunk.rows(), chunk.estimate).await?;
            fragment.read_ranges(chunk.ranges, Some(chunk_budget)).await
        };
        SpawnedTask::spawn(read.in_current_span()).map(|result| result.map_err(Error::from)?)
    }

    /// Reads the chunks with up to `readahead` of them admitted ahead, taking
    /// admissions in the order the scan consumes the chunks.
    pub(super) fn budgeted_task_stream(
        chunks: FragmentChunkStream,
        readahead: usize,
        scan_budget: ScanBudget,
    ) -> BoxStream<'static, Result<MaterializedReadBatchesFut>> {
        chunks
            .map(move |chunk| Self::read_budgeted_chunk(scan_budget.admission(), chunk))
            .buffered(readahead)
            .try_flatten()
            .boxed()
    }

    /// One partition's tasks from the shared chunks, admitted with its own
    /// reader since partitions are consumed independently.
    pub(super) fn partition_task_stream(
        shared: &SharedChunks,
    ) -> BoxStream<'static, Result<MaterializedReadBatchesFut>> {
        let scan_budget = shared.scan_budget.sibling();
        futures::stream::unfold(shared.chunks.clone(), ChunkQueue::pop)
            .map(move |chunk| Self::read_budgeted_chunk(scan_budget.admission(), chunk))
            .buffered(shared.readahead.max(1))
            .try_flatten()
            .boxed()
    }

    /// Reads the fragments under one reservation for the whole read, taken
    /// before any of their pages are requested.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn budgeted_whole_read_stream(
        dataset: Arc<Dataset>,
        scoped_fragments: Vec<ScopedFragmentRead>,
        global_metrics: Arc<FilteredReadGlobalMetrics>,
        fragment_soft_limit: Option<u64>,
        materialization_context: Arc<BlobMaterializationContext>,
        materialize_blob_v2_binary: bool,
        readahead: usize,
        scan_budget: ScanBudget,
    ) -> BoxStream<'static, Result<MaterializedReadBatchesFut>> {
        let admission = scan_budget.admission();
        futures::stream::once(async move {
            let fragments: Vec<Arc<OpenedFragmentRead>> = futures::stream::iter(scoped_fragments)
                .map(|scoped_fragment| {
                    Self::open_fragment(
                        dataset.clone(),
                        scoped_fragment,
                        global_metrics.clone(),
                        fragment_soft_limit,
                        materialization_context.clone(),
                        materialize_blob_v2_binary,
                    )
                })
                .buffered(readahead)
                .try_collect()
                .await?;
            let (mut rows, mut estimate) = (0, ReadEstimate::default());
            for fragment in &fragments {
                rows += fragment.ranges.iter().map(|r| r.end - r.start).sum::<u64>();
                estimate = estimate
                    + fragment
                        .reader
                        .estimate_read(fragment.ranges.clone().into())
                        .await?;
            }
            let chunk_budget = admission.reserve(rows, estimate).await?;
            let reads = futures::future::try_join_all(fragments.iter().map(|fragment| {
                fragment.read_ranges(fragment.ranges.clone(), Some(chunk_budget.clone()))
            }))
            .await?;
            Ok::<_, Error>(futures::stream::iter(reads).flatten())
        })
        .try_flatten()
        .boxed()
    }
}
