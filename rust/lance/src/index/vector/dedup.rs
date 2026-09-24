// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Streaming embedding duplicate pairs over existing index representations.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use futures::{StreamExt, TryStreamExt, stream};
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::{Error, Result};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::prefilter::PreFilter;
use lance_index::vector::{
    VectorIndex,
    pairwise::{DecodedPairwisePartition, PAIRWISE_MEMORY_LIMIT, PairwiseVectorBatch},
};
use lance_linalg::distance::DistanceType;
use lance_select::RowAddrMask;
use uuid::Uuid;

use crate::Dataset;
use crate::index::{
    DatasetIndexExt, DatasetIndexInternalExt, prefilter::DatasetPreFilter,
    segment_has_vector_details,
};

/// Scoring and output remain bounded even when the decoded partition spills.
/// A multiple of 32 also matches RQ's packed sign-code group size.
const VECTOR_BATCH_SIZE: usize = 1024;

// Small SIMD batches do not justify a CPU-pool round trip. Larger distances
// run on the CPU pool; inline batches yield cooperatively in the scan loop.
const MIN_OFFLOAD_COORDINATES: usize = 256 * 1024;

/// Per-invocation decoded-vector caching and ordered scoring concurrency.
///
/// The cache budget is not a process-wide memory limit. It excludes encoded
/// staging, models, row masks, spill metadata and in-flight scoring buffers.
/// With spilled vectors, each in-flight job may retain an anchor batch, a
/// candidate batch and at most one output batch, each bounded to 1,024 rows.
///
/// ```
/// use lance::index::vector::dedup::DuplicatePairsOptions;
/// let options = DuplicatePairsOptions::default()
///     .with_decoded_cache_size(128 * 1024 * 1024)
///     .with_max_concurrency(4);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DuplicatePairsOptions {
    decoded_cache_size: usize,
    max_concurrency: usize,
}

impl Default for DuplicatePairsOptions {
    fn default() -> Self {
        Self {
            decoded_cache_size: 256 * 1024 * 1024,
            max_concurrency: get_num_compute_intensive_cpus().min(8),
        }
    }
}

impl DuplicatePairsOptions {
    /// Set the decoded cache budget in bytes (default 256 MiB). Zero forces
    /// decoded spill. Reconstruction retains at most one extra in-flight batch.
    pub fn with_decoded_cache_size(mut self, decoded_cache_size: usize) -> Self {
        self.decoded_cache_size = decoded_cache_size;
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
/// Quantized distances are symmetric distances between reconstructed index
/// vectors, not exact source-vector distances. Cross-partition and cross-segment
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

/// Run one segment/partition with explicit decoded caching and concurrency.
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
///     DuplicatePairsOptions::default().with_decoded_cache_size(0),
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
    let metric = partition.index.metric_type();
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
            VECTOR_BATCH_SIZE,
            PAIRWISE_MEMORY_LIMIT,
            session.spill_store(),
        )
        .await?;
    let prepared = Arc::new(
        prepared
            .materialize(options.decoded_cache_size, session.spill_store())
            .await?,
    );
    let state = PairWork {
        prepared,
        count,
        metric,
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
    prepared: Arc<DecodedPairwisePartition>,
    count: usize,
    metric: DistanceType,
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
                        .read_vectors(self.anchor_start / VECTOR_BATCH_SIZE)
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
            let end = start.saturating_add(VECTOR_BATCH_SIZE).min(self.count);
            let query = anchor.vectors.value(self.anchor_row);
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
                first,
                candidate_batch: start / VECTOR_BATCH_SIZE,
                // Reuse the anchor's decoded view for same-batch comparisons.
                candidate: (start == self.anchor_start).then(|| anchor.clone()),
                metric: self.metric,
                filter: self.filter.clone(),
                threshold: self.threshold,
                schema: self.schema.clone(),
            }));
        }
    }
}

struct ScoreWork {
    prepared: Arc<DecodedPairwisePartition>,
    a: u64,
    query: ArrayRef,
    first: usize,
    candidate_batch: usize,
    candidate: Option<PairwiseVectorBatch>,
    metric: DistanceType,
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
            first,
            metric,
            filter,
            threshold,
            schema,
            ..
        } = self;
        let coordinates = (candidates.row_ids.len() - first).saturating_mul(query.len());
        let score = move || -> Result<RecordBatch> {
            // Score only the upper triangle, including within one batch.
            let vectors = candidates
                .vectors
                .slice(first, candidates.row_ids.len() - first);
            let distances = metric.arrow_batch_func()(query.as_ref(), &vectors)?;
            let mut b = Vec::new();
            let mut d = Vec::new();
            for i in first..candidates.row_ids.len() {
                let id = candidates.row_ids.value(i);
                let distance = distances.value(i - first);
                if candidates.row_ids.is_valid(i)
                    && filter.selected(id)
                    && id != a
                    && distances.is_valid(i - first)
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
        if coordinates < MIN_OFFLOAD_COORDINATES {
            score()
        } else {
            spawn_cpu(score).await
        }
    }
}
