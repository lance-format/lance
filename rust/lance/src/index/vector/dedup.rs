// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Streaming embedding duplicate pairs over existing index representations.

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch, UInt64Array};
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
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uuid::Uuid;

use crate::Dataset;
use crate::index::{
    DatasetIndexExt, DatasetIndexInternalExt, prefilter::DatasetPreFilter,
    segment_has_vector_details,
};

/// Scoring and output remain bounded even when the staged partition spills.
/// A multiple of 32 also matches RQ's packed sign-code group size.
const MAX_VECTOR_BATCH_SIZE: usize = 8192;

/// Anchor rows per scoring job. A job scores these rows against one candidate
/// batch (at most 32 x 8,192 pairs, a few milliseconds for typical codes), so
/// jobs are uniform enough for ordered completion and dense output per job
/// stays near 5 MiB, while each candidate row loaded into cache is reused by
/// 32 anchors.
const ANCHOR_BLOCK_ROWS: usize = 32;

/// Jobs touching less staged data than this (anchor rows x candidate rows x
/// staged row bytes) run inline; a CPU-pool round trip would cost more.
const MIN_OFFLOAD_WORK_BYTES: usize = 1024 * 1024;

/// Lower bound on decoded spilled batches kept alive at once: the current
/// anchor plus the candidate batches of two consecutive tiles, so loading the
/// next tile never waits on the previous one when tiles have many jobs.
const MIN_LIVE_SPILLED_BATCHES: usize = 3;

/// Per-invocation code staging and ordered scoring concurrency.
///
/// The staging budget is not a process-wide memory limit. It excludes quantizer
/// models, row masks, spill metadata and in-flight scoring buffers. Each
/// in-flight job shares its anchor and candidate batches (at most 8,192 rows
/// and about 16 MiB of staged codes) with the other jobs of its tile, and
/// produces at most one output batch of `32 x vector batch size` pairs. A
/// spilled partition's batches are read back as decoded copies; those alive at
/// once hold at most `memory_limit` bytes of staged codes (or three batches, if
/// larger), even when sparse selection leaves only one job per tile.
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
            max_concurrency: get_num_compute_intensive_cpus(),
        }
    }
}

impl DuplicatePairsOptions {
    /// Set the staged code budget in bytes (default 256 MiB).
    /// Partitions estimated to exceed it spill; zero forces spill. Source reads,
    /// quantizer preparation and scoring retain additional bounded batches.
    pub fn with_memory_limit(mut self, memory_limit: usize) -> Self {
        self.memory_limit = memory_limit;
        self
    }

    /// Set the maximum number of in-flight scoring jobs. Must be positive.
    /// The default is the CPU pool size. Output order does not depend on it.
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
/// Cross-partition and cross-segment pairs are not evaluated. No top-k limit
/// is applied. A pair qualifies when its finite distance is `<= threshold`.
///
/// Distances are computed natively between the index representations `x̂`
/// (the vectors the codes reconstruct, exact vectors for IVF_FLAT) without
/// reconstructing them, using the metric's definition:
///
/// | metric  | distance                                          |
/// |---------|---------------------------------------------------|
/// | l2      | squared L2 `‖x̂a − x̂b‖²`                           |
/// | cosine  | `1 − cos(x̂a, x̂b)` in `[0, 2]`; 0 for identical codes |
/// | dot     | `1 − x̂a·x̂b`                                       |
/// | hamming | number of differing bits                          |
///
/// Cosine renormalizes quantized representations, as exact or refined search
/// does; unrefined ANN search over quantized cosine indices reports roughly
/// twice this value. Distances are bit-identical across concurrency, memory
/// limits and the scoped [`find_duplicate_pairs_in_partition`] API.
///
/// Output order is deterministic: segments, partitions ascending, then tiles
/// of staged vector batches `(I, J >= I)`, row-major within a tile (anchor row,
/// then candidate row). Each pair appears once with `row_id_a` at the lower
/// storage position; `row_id_a` values are not contiguous across tiles.
/// Dropping the stream cancels further reads; only bounded in-flight CPU work
/// can finish. Each partition's codes are staged once; larger partitions use
/// session spill storage, reclaimed when advancing partitions or dropping the
/// stream, and are read O(B²) times for B vector batches.
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
/// local to that segment. This uses the same scorer, distances and ordering as
/// [`find_duplicate_pairs`]. It does not require other fragments to be indexed,
/// so rows of unindexed fragments are not paired.
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
    // In-memory reads share the staged buffers; spilled reads decode copies.
    let live_batches = prepared.is_spilled().then(|| {
        let batch_bytes = prepared
            .vector_batch_size()
            .saturating_mul(prepared.row_bytes())
            .max(1);
        Arc::new(Semaphore::new(
            (options.memory_limit / batch_bytes).max(MIN_LIVE_SPILLED_BATCHES),
        ))
    });
    let state = TileJobs {
        num_batches: prepared.num_batches(),
        live_batches,
        prepared: Arc::new(prepared),
        filter: partition.mask,
        threshold,
        schema: schema.clone(),
        anchor_batch: 0,
        candidate_batch: 0,
        anchor: None,
        candidates: None,
        next_row: 0,
    };
    let jobs = stream::try_unfold(state, |mut state| async move {
        state
            .next_job()
            .await
            .map(|job| job.map(|job| (job, state)))
    });
    // Ordered completion keeps the tile order independent of concurrency
    // without collecting a whole tile's matches.
    let batches = jobs
        .map_ok(|job| job.score())
        .try_buffered(options.max_concurrency)
        .try_filter(|batch| std::future::ready(batch.num_rows() != 0))
        .map_err(datafusion::error::DataFusionError::from);
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, batches)))
}

/// Produces scoring jobs lazily in tile order: anchor batch `I`, candidate
/// batch `J >= I`, then blocks of anchor rows. Each batch is read once per
/// tile side (O(B²) reads for B batches) and shared by that tile's jobs.
struct TileJobs {
    prepared: Arc<PairwisePartition>,
    num_batches: usize,
    /// Bounds decoded spilled batches held by the generator and in-flight jobs.
    /// The generator holds at most one permit (its anchor) while waiting, and
    /// jobs release theirs on completion, so the wait always makes progress.
    live_batches: Option<Arc<Semaphore>>,
    filter: Arc<RowAddrMask>,
    threshold: f32,
    schema: SchemaRef,
    anchor_batch: usize,
    candidate_batch: usize,
    anchor: Option<Arc<LoadedBatch>>,
    candidates: Option<Arc<LoadedBatch>>,
    next_row: usize,
}

/// A loaded batch with its row selection, holding a live-batch permit while
/// any job still references it.
struct LoadedBatch {
    batch: PairwiseVectorBatch,
    _permit: Option<OwnedSemaphorePermit>,
}

fn has_selected(selected: &[bool]) -> bool {
    selected.iter().any(|&selected| selected)
}

impl TileJobs {
    /// Read a batch and resolve row selection once for every pair it joins.
    async fn load(&self, batch_id: usize) -> Result<Arc<LoadedBatch>> {
        let permit = match &self.live_batches {
            Some(live) => Some(
                live.clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| Error::internal("pairwise live batch semaphore closed"))?,
            ),
            None => None,
        };
        let mut batch = self.prepared.read_vectors(batch_id).await?;
        batch.filter_rows(|id| self.filter.selected(id));
        Ok(Arc::new(LoadedBatch {
            batch,
            _permit: permit,
        }))
    }

    async fn next_job(&mut self) -> Result<Option<ScoreJob>> {
        loop {
            tokio::task::consume_budget().await;
            if self.anchor_batch >= self.num_batches {
                return Ok(None);
            }
            let anchor = match &self.anchor {
                Some(anchor) => anchor.clone(),
                None => {
                    let anchor = self.load(self.anchor_batch).await?;
                    if !has_selected(anchor.batch.selected()) {
                        self.anchor_batch += 1;
                        continue;
                    }
                    self.anchor = Some(anchor.clone());
                    self.candidate_batch = self.anchor_batch;
                    self.next_row = 0;
                    anchor
                }
            };
            let is_diagonal = self.candidate_batch == self.anchor_batch;
            let candidates = match &self.candidates {
                Some(candidates) => candidates.clone(),
                None => {
                    let candidates = if is_diagonal {
                        anchor.clone()
                    } else {
                        self.load(self.candidate_batch).await?
                    };
                    self.candidates = Some(candidates.clone());
                    self.next_row = 0;
                    candidates
                }
            };
            // The last row of a diagonal tile has no later candidate.
            let anchor_end = anchor.batch.num_rows() - usize::from(is_diagonal);
            if self.next_row >= anchor_end || !has_selected(candidates.batch.selected()) {
                self.candidates = None;
                self.candidate_batch += 1;
                if self.candidate_batch >= self.num_batches {
                    self.anchor = None;
                    self.anchor_batch += 1;
                }
                continue;
            }
            let rows = self.next_row..(self.next_row + ANCHOR_BLOCK_ROWS).min(anchor_end);
            self.next_row = rows.end;
            if !has_selected(&anchor.batch.selected()[rows.clone()]) {
                continue;
            }
            return Ok(Some(ScoreJob {
                prepared: self.prepared.clone(),
                anchor,
                rows,
                candidates,
                threshold: self.threshold,
                schema: self.schema.clone(),
            }));
        }
    }
}

struct ScoreJob {
    prepared: Arc<PairwisePartition>,
    anchor: Arc<LoadedBatch>,
    rows: Range<usize>,
    candidates: Arc<LoadedBatch>,
    threshold: f32,
    schema: SchemaRef,
}

impl ScoreJob {
    async fn score(self) -> Result<RecordBatch> {
        let work_bytes = self
            .rows
            .len()
            .saturating_mul(self.candidates.batch.num_rows())
            .saturating_mul(self.prepared.row_bytes());
        let score = move || -> Result<RecordBatch> {
            let hits = self.prepared.score_block(
                &self.anchor.batch,
                self.rows,
                &self.candidates.batch,
                self.threshold,
            )?;
            Ok(RecordBatch::try_new(
                self.schema,
                vec![
                    Arc::new(UInt64Array::from(hits.row_id_a)),
                    Arc::new(UInt64Array::from(hits.row_id_b)),
                    Arc::new(Float32Array::from(hits.distances)),
                ],
            )?)
        };
        if work_bytes < MIN_OFFLOAD_WORK_BYTES {
            score()
        } else {
            spawn_cpu(score).await
        }
    }
}
