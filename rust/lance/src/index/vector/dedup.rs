// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Streaming embedding duplicate pairs over existing index representations.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::{Array, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use futures::{StreamExt, TryStreamExt, stream};
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::{Error, Result};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::prefilter::PreFilter;
use lance_index::vector::{
    VectorIndex,
    pairwise::{PAIRWISE_MEMORY_LIMIT, PairwisePartition, PairwiseVectorBatch},
};
use lance_select::RowAddrMask;
use uuid::Uuid;

use crate::Dataset;
use crate::index::{
    DatasetIndexExt, DatasetIndexInternalExt, prefilter::DatasetPreFilter,
    segment_has_vector_details,
};

/// Scoring and output remain bounded even when the encoded partition spills.
/// A multiple of 32 also matches RQ's packed sign-code group size.
const MAX_VECTOR_BATCH_SIZE: usize = 8192;

// Small SIMD batches do not justify a CPU-pool round trip. Larger distances
// run on the CPU pool; inline batches yield cooperatively in the scan loop.
const MIN_OFFLOAD_BYTES: usize = 256 * 1024;

/// Per-invocation code staging and ordered scoring concurrency.
///
/// The staging budget is not a process-wide memory limit. It excludes quantizer
/// models, row masks, spill metadata and in-flight scoring buffers.
/// With spilled codes, each in-flight job may retain an anchor batch, a
/// candidate batch and at most one output batch, each bounded to 8,192 rows (smaller batches for wide codes).
///
/// ```
/// use lance::index::vector::dedup::DuplicatePairsOptions;
/// let options = DuplicatePairsOptions::default()
///     .with_memory_limit(128 * 1024 * 1024)
///     .with_max_concurrency(4);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DuplicatePairsOptions {
    memory_limit: usize,
    max_concurrency: usize,
}

impl Default for DuplicatePairsOptions {
    fn default() -> Self {
        Self {
            memory_limit: PAIRWISE_MEMORY_LIMIT,
            max_concurrency: get_num_compute_intensive_cpus().min(8),
        }
    }
}

impl DuplicatePairsOptions {
    /// Set the compact code staging budget in bytes (default 256 MiB).
    /// Partitions estimated to exceed it spill; zero forces spill. Source reads,
    /// quantizer preparation and scoring retain additional bounded batches.
    pub fn with_memory_limit(mut self, memory_limit: usize) -> Self {
        self.memory_limit = memory_limit;
        self
    }

    /// Set the maximum number of in-flight scoring jobs. Must be positive.
    /// The default is the CPU pool size capped at eight. Output remains ordered.
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }
}

fn pair_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("row_id_a", DataType::UInt64, false),
        Field::new("row_id_b", DataType::UInt64, false),
        Field::new("distance", DataType::Float32, false),
    ]))
}

/// Enumerate threshold-matching pairs independently in every index partition.
///
/// `column` must have exactly one logical, current-format vector index covering
/// all current fragments. Deleted rows are excluded at the dataset snapshot.
/// Quantized distances use native code-to-code batch kernels, not exact
/// source-vector distances. Cross-partition and cross-segment
/// pairs are not evaluated. No top-k limit is applied.
///
/// Pairs follow stable index traversal (`i < j`), not numeric row-ID order.
/// All output for one `row_id_a` is contiguous, even across output batches.
/// Dropping the stream cancels further reads; only bounded in-flight CPU work
/// can finish. See [`find_duplicate_pairs_in_partition`] for distributed use.
/// Each partition's compact codes are prepared once; larger partitions use
/// session spill storage, reclaimed when advancing partitions or dropping the stream.
///
/// ```
/// # use std::sync::Arc;
/// # use lance::{Dataset, Result};
/// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
/// let pairs = lance::index::vector::dedup::find_duplicate_pairs(
///     dataset, "embedding", 0.05,
/// ).await?;
/// drop(pairs);
/// # Ok(()) }
/// ```
pub async fn find_duplicate_pairs(
    dataset: Arc<Dataset>,
    column: &str,
    distance_threshold: f32,
) -> Result<SendableRecordBatchStream> {
    find_duplicate_pairs_with_options(
        dataset,
        column,
        distance_threshold,
        DuplicatePairsOptions::default(),
    )
    .await
}

/// Enumerate pairs with explicit resource settings; semantics are identical to
/// [`find_duplicate_pairs`]. See [`DuplicatePairsOptions`] for memory accounting.
///
/// ```
/// # use std::sync::Arc;
/// # use lance::{Dataset, Result};
/// # use lance::index::vector::dedup::{find_duplicate_pairs_with_options, DuplicatePairsOptions};
/// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
/// let pairs = find_duplicate_pairs_with_options(
///     dataset, "embedding", 0.05, DuplicatePairsOptions::default().with_max_concurrency(4),
/// ).await?;
/// # Ok(()) }
/// ```
pub async fn find_duplicate_pairs_with_options(
    dataset: Arc<Dataset>,
    column: &str,
    distance_threshold: f32,
    options: DuplicatePairsOptions,
) -> Result<SendableRecordBatchStream> {
    plan(dataset, column, None, distance_threshold, options).await
}

/// Enumerate pairs only within the specified physical segment and partition.
///
/// The segment UUID must belong to the index of `column`; `partition_id` is
/// local to that segment. This uses the same scorer and ordering as
/// [`find_duplicate_pairs`], and does not require other fragments to be indexed.
/// The caller must pin the same dataset version on every worker.
///
/// ```
/// # use std::sync::Arc;
/// # use lance::{Dataset, Result};
/// # async fn example(dataset: Arc<Dataset>, segment_id: uuid::Uuid) -> Result<()> {
/// let pairs = lance::index::vector::dedup::find_duplicate_pairs_in_partition(
///     dataset, "embedding", segment_id, 0, 0.05,
/// ).await?;
/// drop(pairs);
/// # Ok(()) }
/// ```
pub async fn find_duplicate_pairs_in_partition(
    dataset: Arc<Dataset>,
    column: &str,
    segment_id: Uuid,
    partition_id: usize,
    distance_threshold: f32,
) -> Result<SendableRecordBatchStream> {
    find_duplicate_pairs_in_partition_with_options(
        dataset,
        column,
        segment_id,
        partition_id,
        distance_threshold,
        DuplicatePairsOptions::default(),
    )
    .await
}

/// Run one segment/partition with explicit code staging and concurrency.
/// Uses the snapshot, distance and ordering contract of
/// [`find_duplicate_pairs_in_partition`].
///
/// ```
/// # use std::sync::Arc;
/// # use lance::{Dataset, Result};
/// # use lance::index::vector::dedup::{find_duplicate_pairs_in_partition_with_options, DuplicatePairsOptions};
/// # async fn example(dataset: Arc<Dataset>, segment: uuid::Uuid) -> Result<()> {
/// let pairs = find_duplicate_pairs_in_partition_with_options(
///     dataset, "embedding", segment, 0, 0.05,
///     DuplicatePairsOptions::default().with_memory_limit(0),
/// ).await?;
/// # Ok(()) }
/// ```
pub async fn find_duplicate_pairs_in_partition_with_options(
    dataset: Arc<Dataset>,
    column: &str,
    segment_id: Uuid,
    partition_id: usize,
    distance_threshold: f32,
    options: DuplicatePairsOptions,
) -> Result<SendableRecordBatchStream> {
    plan(
        dataset,
        column,
        Some((segment_id, partition_id)),
        distance_threshold,
        options,
    )
    .await
}

async fn plan(
    dataset: Arc<Dataset>,
    column: &str,
    selection: Option<(Uuid, usize)>,
    threshold: f32,
    options: DuplicatePairsOptions,
) -> Result<SendableRecordBatchStream> {
    if options.max_concurrency == 0 {
        return Err(Error::invalid_input(
            "max_concurrency must be positive, got 0",
        ));
    }
    if !threshold.is_finite() {
        return Err(Error::invalid_input(format!(
            "distance_threshold must be finite, got {threshold}"
        )));
    }
    let field = dataset.schema().field_id(column)?;
    let indices = dataset.load_indices().await?;
    let names = indices
        .iter()
        .filter(|m| m.keyed_fields() == [field] && segment_has_vector_details(m))
        .map(|m| m.name.as_str())
        .collect::<BTreeSet<_>>();
    if names.len() != 1 {
        return Err(Error::invalid_input(format!(
            "column '{column}' requires exactly one vector index, found {}",
            names.len()
        )));
    }
    let name = names
        .into_iter()
        .next()
        .ok_or_else(|| Error::internal("missing resolved vector index"))?;
    if selection.is_none() && !dataset.unindexed_fragments(name).await?.is_empty() {
        return Err(Error::invalid_input(format!(
            "vector index on column '{column}' does not cover all fragments; optimize the index first"
        )));
    }
    let segments = indices
        .iter()
        .filter(|meta| meta.name == name && meta.keyed_fields() == [field])
        .collect::<Vec<_>>();
    if let Some((id, _)) = selection
        && !segments.iter().any(|m| m.uuid == id)
    {
        return Err(Error::invalid_input(format!(
            "segment_id={id} is not an active segment of column '{column}'"
        )));
    }
    let mut partitions = Vec::new();
    for meta in segments
        .into_iter()
        .filter(|meta| selection.is_none_or(|(id, _)| id == meta.uuid))
    {
        let index = dataset
            .open_vector_index(column, &meta.uuid, &NoOpMetricsCollector)
            .await?;
        if !index.supports_pairwise_vectors() {
            return Err(Error::not_supported(format!(
                "segment {} requires a current-format vector index; rebuild the index",
                meta.uuid
            )));
        }
        for fragment in dataset.fragments().iter() {
            if meta
                .fragment_bitmap
                .as_ref()
                .is_none_or(|ids| ids.contains(fragment.id as u32))
                && fragment.overlays.iter().any(|overlay| {
                    overlay.committed_version > meta.dataset_version
                        && overlay.data_file.fields.contains(&field)
                })
            {
                return Err(Error::invalid_input(format!(
                    "column '{column}' has stale index values in segment {} after an overlay update; rebuild the index",
                    meta.uuid
                )));
            }
        }
        let mask = if dataset.manifest().uses_stable_row_ids() {
            // Compaction can materialize deletions without rewriting a stable
            // row-ID index. Its old IDs then survive in storage even though no
            // fragment has a deletion file: require current row-ID membership.
            DatasetPreFilter::do_create_deletion_mask_row_id(
                dataset.clone(),
                meta.fragment_bitmap.clone(),
            )
            .await?
        } else {
            let filter = DatasetPreFilter::new(dataset.clone(), std::slice::from_ref(meta), None);
            filter.wait_for_ready().await?;
            filter.mask()
        };
        let count = index.ivf_model().num_partitions();
        if let Some((_, p)) = selection {
            if p >= count {
                return Err(Error::invalid_input(format!(
                    "partition_id={p} out of range 0..{count} for segment {}",
                    meta.uuid
                )));
            }
            partitions.push(Partition { index, id: p, mask });
        } else {
            partitions.extend((0..count).map(|id| Partition {
                index: index.clone(),
                id,
                mask: mask.clone(),
            }));
        }
    }
    let session = dataset.session();
    let stream = stream::iter(partitions)
        .then(move |partition| partition_stream(partition, session.clone(), threshold, options))
        .map_err(datafusion::error::DataFusionError::from)
        .try_flatten();
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        pair_schema(),
        stream,
    )))
}

struct Partition {
    index: Arc<dyn VectorIndex>,
    id: usize,
    mask: Arc<RowAddrMask>,
}

async fn partition_stream(
    partition: Partition,
    session: Arc<crate::session::Session>,
    threshold: f32,
    options: DuplicatePairsOptions,
) -> Result<SendableRecordBatchStream> {
    let count = partition.index.partition_size(partition.id);
    let schema = pair_schema();
    if count == 0 {
        return Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::empty(),
        )));
    }
    let prepared = partition
        .index
        .prepare_pairwise_partition(
            partition.id,
            MAX_VECTOR_BATCH_SIZE,
            options.memory_limit,
            session.spill_store(),
        )
        .await?;
    let batch_size = prepared.vector_batch_size();
    let prepared = Arc::new(prepared);
    let state = PairWork {
        prepared,
        count,
        batch_size,
        filter: partition.mask,
        threshold,
        schema: schema.clone(),
        anchor: None,
        anchor_start: 0,
        anchor_row: 0,
        candidate_start: 0,
    };
    let jobs = stream::try_unfold(state, |mut state| async move {
        state
            .next_work()
            .await
            .map(|work| work.map(|work| (work, state)))
    });
    // Bounded ordered completion preserves contiguous anchors without collecting
    // an entire anchor's matches or an entire block-pair distance matrix.
    let batches = jobs
        .map_ok(|work| work.score())
        .try_buffered(options.max_concurrency)
        .try_filter(|batch| std::future::ready(batch.num_rows() != 0))
        .map_err(datafusion::error::DataFusionError::from);
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, batches)))
}

struct PairWork {
    prepared: Arc<PairwisePartition>,
    count: usize,
    batch_size: usize,
    filter: Arc<RowAddrMask>,
    threshold: f32,
    schema: SchemaRef,
    anchor: Option<PairwiseVectorBatch>,
    anchor_start: usize,
    anchor_row: usize,
    candidate_start: usize,
}

impl PairWork {
    async fn next_work(&mut self) -> Result<Option<ScoreWork>> {
        loop {
            tokio::task::consume_budget().await;
            if self.anchor_start >= self.count {
                return Ok(None);
            }
            if self.anchor.is_none() {
                self.anchor = Some(
                    self.prepared
                        .read_vectors(self.anchor_start / self.batch_size)
                        .await?,
                );
                self.anchor_row = 0;
                self.candidate_start = self.anchor_start;
            }
            let anchor = self
                .anchor
                .as_ref()
                .ok_or_else(|| Error::internal("missing anchor batch"))?;
            if self.anchor_row == anchor.row_ids.len() {
                self.anchor_start += anchor.row_ids.len();
                self.anchor = None;
                continue;
            }
            let a = anchor.row_ids.value(self.anchor_row);
            if !anchor.row_ids.is_valid(self.anchor_row)
                || !self.filter.selected(a)
                || self.candidate_start >= self.count
            {
                self.anchor_row += 1;
                self.candidate_start = self.anchor_start;
                continue;
            }
            let start = self.candidate_start;
            let end = start.saturating_add(self.batch_size).min(self.count);
            let query = anchor.clone();
            let first = (self.anchor_start + self.anchor_row + 1)
                .saturating_sub(start)
                .min(end - start);
            self.candidate_start = end;
            if first == end - start {
                continue;
            }
            return Ok(Some(ScoreWork {
                prepared: self.prepared.clone(),
                a,
                query,
                query_row: self.anchor_row,
                first,
                candidate_batch: start / self.batch_size,
                // Reuse the anchor's codes for same-batch comparisons.
                candidate: (start == self.anchor_start).then(|| anchor.clone()),
                filter: self.filter.clone(),
                threshold: self.threshold,
                schema: self.schema.clone(),
            }));
        }
    }
}

struct ScoreWork {
    prepared: Arc<PairwisePartition>,
    a: u64,
    query: PairwiseVectorBatch,
    query_row: usize,
    first: usize,
    candidate_batch: usize,
    candidate: Option<PairwiseVectorBatch>,
    filter: Arc<RowAddrMask>,
    threshold: f32,
    schema: SchemaRef,
}

impl ScoreWork {
    async fn score(self) -> Result<RecordBatch> {
        let candidates = match self.candidate {
            Some(candidate) => candidate,
            None => self.prepared.read_vectors(self.candidate_batch).await?,
        };
        let Self {
            a,
            query,
            query_row,
            prepared,
            first,
            filter,
            threshold,
            schema,
            ..
        } = self;
        let work_bytes = candidates.codes.get_array_memory_size();
        let score = move || -> Result<RecordBatch> {
            // Score only the upper triangle, including within one batch.
            let candidates = PairwiseVectorBatch {
                row_ids: candidates
                    .row_ids
                    .slice(first, candidates.row_ids.len() - first),
                codes: candidates
                    .codes
                    .slice(first, candidates.row_ids.len() - first),
            };
            let distances = prepared.distance_batch(&query, query_row, &candidates)?;
            let mut b = Vec::new();
            let mut d = Vec::new();
            for (i, distance) in distances.into_iter().enumerate() {
                let id = candidates.row_ids.value(i);
                if candidates.row_ids.is_valid(i)
                    && filter.selected(id)
                    && id != a
                    && distance.is_finite()
                    && distance <= threshold
                {
                    b.push(id);
                    d.push(distance);
                }
            }
            Ok(RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt64Array::from(vec![a; b.len()])),
                    Arc::new(UInt64Array::from(b)),
                    Arc::new(Float32Array::from(d)),
                ],
            )?)
        };
        if work_bytes < MIN_OFFLOAD_BYTES {
            score()
        } else {
            spawn_cpu(score).await
        }
    }
}
