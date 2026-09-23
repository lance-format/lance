// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Streaming embedding duplicate pairs over existing index representations.

use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;

use arrow_array::{Array, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use futures::stream;
use lance_core::utils::tokio::spawn_cpu;
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

/// At most two decoded input batches and one output batch are retained. A
/// multiple of 32 also matches RQ's packed sign-code group size.
/// Compact index codes are separately staged in bounded memory or session spill.
const VECTOR_BATCH_SIZE: usize = 1024;

// Small SIMD batches do not justify a CPU-pool round trip. Larger distances
// run on the CPU pool; inline batches yield cooperatively in the scan loop.
const MIN_OFFLOAD_COORDINATES: usize = 256 * 1024;

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
    plan(dataset, column, None, distance_threshold).await
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
    plan(
        dataset,
        column,
        Some((segment_id, partition_id)),
        distance_threshold,
    )
    .await
}

async fn plan(
    dataset: Arc<Dataset>,
    column: &str,
    selection: Option<(Uuid, usize)>,
    threshold: f32,
) -> Result<SendableRecordBatchStream> {
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
    let mut partitions = VecDeque::new();
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
            partitions.push_back(Partition { index, id: p, mask });
        } else {
            partitions.extend((0..count).map(|id| Partition {
                index: index.clone(),
                id,
                mask: mask.clone(),
            }));
        }
    }
    let state = PairStream {
        partitions,
        session: dataset.session(),
        prepared: None,
        threshold,
        anchor: None,
        candidate: None,
        anchor_start: 0,
        anchor_row: 0,
        candidate_start: 0,
    };
    let stream = stream::try_unfold(state, |mut state| async move {
        state
            .next_batch()
            .await
            .map(|batch| batch.map(|batch| (batch, state)))
    });
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

struct PairStream {
    partitions: VecDeque<Partition>,
    session: Arc<crate::session::Session>,
    prepared: Option<PairwisePartition>,
    threshold: f32,
    anchor: Option<PairwiseVectorBatch>,
    candidate: Option<(usize, PairwiseVectorBatch)>,
    anchor_start: usize,
    anchor_row: usize,
    candidate_start: usize,
}

impl PairStream {
    async fn next_batch(&mut self) -> datafusion::error::Result<Option<RecordBatch>> {
        loop {
            tokio::task::consume_budget().await;
            let Some(Partition {
                index,
                id: partition,
                mask: filter,
            }) = self.partitions.front()
            else {
                return Ok(None);
            };
            let count = index.partition_size(*partition);
            if self.anchor_start >= count {
                self.partitions.pop_front();
                self.prepared = None;
                self.anchor = None;
                self.candidate = None;
                self.anchor_start = 0;
                self.anchor_row = 0;
                continue;
            }
            if self.prepared.is_none() {
                self.prepared = Some(
                    index
                        .prepare_pairwise_partition(
                            *partition,
                            VECTOR_BATCH_SIZE,
                            PAIRWISE_MEMORY_LIMIT,
                            self.session.spill_store(),
                        )
                        .await?,
                );
            }
            let prepared = self
                .prepared
                .as_ref()
                .ok_or_else(|| Error::internal("missing prepared pairwise partition"))?;
            if self.anchor.is_none() {
                self.anchor = Some(
                    prepared
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
                || !filter.selected(a)
                || self.candidate_start >= count
            {
                self.anchor_row += 1;
                self.candidate_start = self.anchor_start;
                continue;
            }
            let start = self.candidate_start;
            let end = (start + VECTOR_BATCH_SIZE).min(count);
            let candidates = if start == self.anchor_start {
                anchor.clone()
            } else if let Some((_, batch)) = self.candidate.as_ref().filter(|(id, _)| *id == start)
            {
                batch.clone()
            } else {
                self.candidate = None;
                let batch = prepared.read_vectors(start / VECTOR_BATCH_SIZE).await?;
                self.candidate = Some((start, batch.clone()));
                batch
            };
            let query = anchor.vectors.value(self.anchor_row);
            let first = (self.anchor_start + self.anchor_row + 1)
                .saturating_sub(start)
                .min(candidates.row_ids.len());
            let metric = index.metric_type();
            let filter = filter.clone();
            let threshold = self.threshold;
            self.candidate_start = end;
            if first == candidates.row_ids.len() {
                continue;
            }
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
                    pair_schema(),
                    vec![
                        Arc::new(UInt64Array::from(vec![a; b.len()])),
                        Arc::new(UInt64Array::from(b)),
                        Arc::new(Float32Array::from(d)),
                    ],
                )?)
            };
            let batch = if coordinates < MIN_OFFLOAD_COORDINATES {
                score()?
            } else {
                spawn_cpu(score).await?
            };
            if batch.num_rows() > 0 {
                return Ok(Some(batch));
            }
        }
    }
}
