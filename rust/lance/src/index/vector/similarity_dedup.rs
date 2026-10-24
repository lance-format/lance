// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Directed IVF radius discovery, followed by exact source-vector verification.
//! Callers canonicalize and deduplicate edges after combining all source scopes.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, RecordBatch, UInt64Array,
};
use arrow_schema::DataType;
use datafusion::physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use futures::{StreamExt, TryStreamExt, stream};
use lance_arrow::RecordBatchExt;
use lance_core::{Error, ROW_ID, Result};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::prefilter::PreFilter;
use lance_index::vector::{ApproxMode, Query, VectorIndex};
use lance_linalg::{distance::DistanceType, kernels::normalize_arrow};
use uuid::Uuid;

use crate::index::{
    DatasetIndexExt, DatasetIndexInternalExt, prefilter::DatasetPreFilter,
    segment_has_vector_details,
};
use crate::{Dataset, dataset::ProjectionRequest};

const SOURCE_BATCH_ROWS: usize = 128;
const VERIFY_BATCH_ROWS: usize = 1024;
const DEFAULT_AUTO_MAX_PROBES: usize = 2;

/// Probe budgets apply independently to each target index segment. Automatic
/// probing tests centroid boundaries among the nearest two centroids by default.
/// `max_nprobes` changes that budget; fixed `nprobes` uses its own count. Both
/// restricted probes and quantization can omit threshold matches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SimilaritySearchOptions {
    pub nprobes: Option<usize>,
    pub max_nprobes: Option<usize>,
}

impl SimilaritySearchOptions {
    pub fn validate(&self) -> Result<()> {
        if self.nprobes == Some(0) || self.max_nprobes == Some(0) {
            return Err(Error::invalid_input("probe counts must be positive"));
        }
        if self
            .nprobes
            .zip(self.max_nprobes)
            .is_some_and(|(n, max)| n > max)
        {
            return Err(Error::invalid_input("nprobes exceeds max_nprobes"));
        }
        Ok(())
    }
}

struct Target {
    index: Arc<dyn VectorIndex>,
    filter: Arc<dyn PreFilter>,
}

/// Query every live row in one source IVF scope against every target segment.
/// Results have the native duplicate-pair schema and canonical row IDs, but may
/// repeat when both directions discover an edge. There is no top-k truncation.
/// Distances are verified on original vectors using the index metric. Discovery
/// remains approximate for quantized indexes and restricted probe budgets.
pub async fn find_similar_pairs_in_partition(
    dataset: Arc<Dataset>,
    column: &str,
    segment_id: Uuid,
    partition_id: usize,
    threshold: f32,
    options: SimilaritySearchOptions,
) -> Result<SendableRecordBatchStream> {
    options.validate()?;
    if !threshold.is_finite() {
        return Err(Error::invalid_input("distance_threshold must be finite"));
    }
    let field = dataset.schema().field_id(column)?;
    let metadata = dataset.load_indices().await?;
    let names = metadata
        .iter()
        .filter(|m| m.keyed_fields() == [field] && segment_has_vector_details(m))
        .map(|m| m.name.as_str())
        .collect::<BTreeSet<_>>();
    if names.len() != 1 {
        return Err(Error::invalid_input(
            "similarity_search requires exactly one vector index",
        ));
    }
    let name = names
        .into_iter()
        .next()
        .ok_or_else(|| Error::internal("missing vector index"))?;
    if !dataset.unindexed_fragments(name).await?.is_empty() {
        return Err(Error::invalid_input(
            "vector index does not cover all fragments; optimize the index first",
        ));
    }
    let segments = metadata
        .iter()
        .filter(|m| m.name == name && m.keyed_fields() == [field])
        .collect::<Vec<_>>();
    if !segments.iter().any(|m| m.uuid == segment_id) {
        return Err(Error::invalid_input(
            "source segment is not active in the pinned snapshot",
        ));
    }
    let mut targets = Vec::new();
    let mut source = None;
    let mut source_mask = None;
    for meta in segments {
        let index = dataset
            .open_vector_index(column, &meta.uuid, &NoOpMetricsCollector)
            .await?;
        if !index.supports_pairwise_vectors() {
            return Err(Error::not_supported(
                "similarity_search requires a current-format vector index; rebuild the index",
            ));
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
                return Err(Error::invalid_input(
                    "vector index has stale values after an overlay update; rebuild the index",
                ));
            }
        }
        if threshold < 0.0 && index.metric_type() != DistanceType::Dot {
            return Err(Error::invalid_input(
                "negative radius requires dot distance",
            ));
        }
        let filter = Arc::new(DatasetPreFilter::new(
            dataset.clone(),
            std::slice::from_ref(meta),
            None,
        ));
        filter.wait_for_ready().await?;
        let mask = if dataset.manifest().uses_stable_row_ids() {
            DatasetPreFilter::do_create_deletion_mask_row_id(
                dataset.clone(),
                meta.fragment_bitmap.clone(),
            )
            .await?
        } else {
            filter.mask()
        };
        if meta.uuid == segment_id {
            source = Some(index.clone());
            source_mask = Some(mask.clone());
        }
        targets.push(Target {
            index,
            filter: Arc::new(MaskFilter(mask)),
        });
    }
    let source = source.ok_or_else(|| Error::internal("missing source index"))?;
    if partition_id >= source.total_partitions() {
        return Err(Error::invalid_input("source partition out of range"));
    }
    let source_mask = source_mask.ok_or_else(|| Error::internal("missing source row mask"))?;
    let source = source
        .partition_reader(partition_id, false, &NoOpMetricsCollector)
        .await?;
    let targets = Arc::new(targets);
    let column = column.to_owned();
    let schema = super::dedup::pair_schema();
    let batches = source
        .map_ok(|batch| {
            stream::iter(
                (0..batch.num_rows())
                    .step_by(SOURCE_BATCH_ROWS)
                    .map(move |offset| {
                        Ok::<_, datafusion::error::DataFusionError>(
                            batch.slice(offset, SOURCE_BATCH_ROWS.min(batch.num_rows() - offset)),
                        )
                    }),
            )
        })
        .try_flatten();
    let queries = batches
        .then(move |batch| {
            let dataset = dataset.clone();
            let column = column.clone();
            let mask = source_mask.clone();
            let targets = targets.clone();
            async move {
                let batch = batch?;
                let ids = batch
                    .column_by_name(ROW_ID)
                    .ok_or_else(|| Error::internal("missing source row IDs"))?
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::internal("invalid source row IDs"))?;
                let ids = ids
                    .values()
                    .iter()
                    .copied()
                    .filter(|id| mask.selected(*id))
                    .collect::<Vec<_>>();
                let projection =
                    ProjectionRequest::from_columns([column.as_str()], dataset.schema());
                let values = dataset.take_rows(&ids, projection).await?;
                if values.num_rows() != ids.len() {
                    return Err(Error::internal("source row missing from pinned snapshot"));
                }
                let vectors = values
                    .column_by_qualified_name(&column)
                    .ok_or_else(|| {
                        Error::invalid_input("vector column missing from projected source")
                    })?
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| Error::invalid_input("expected fixed-size vector column"))?;
                let queries = ids
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| vectors.is_valid(*i))
                    .map(|(i, id)| (id, vectors.value(i)))
                    .collect::<Vec<_>>();
                Ok::<_, Error>(
                    stream::iter(queries)
                        .then(move |(id, key)| {
                            query_stream(
                                dataset.clone(),
                                column.clone(),
                                targets.clone(),
                                id,
                                key,
                                threshold,
                                options,
                            )
                        })
                        .map_err(datafusion::error::DataFusionError::from)
                        .try_flatten(),
                )
            }
        })
        .map_err(datafusion::error::DataFusionError::from)
        .try_flatten();
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, queries)))
}

struct MaskFilter(Arc<lance_select::RowAddrMask>);
#[async_trait::async_trait]
impl PreFilter for MaskFilter {
    async fn wait_for_ready(&self) -> Result<()> {
        Ok(())
    }
    fn is_empty(&self) -> bool {
        false
    }
    fn mask(&self) -> Arc<lance_select::RowAddrMask> {
        self.0.clone()
    }
    fn filter_row_ids<'a>(&self, rows: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        rows.enumerate()
            .filter(|(_, id)| self.0.selected(**id))
            .map(|(i, _)| i as u64)
            .collect()
    }
}

fn make_query(column: String, key: ArrayRef) -> Query {
    Query {
        column,
        key,
        k: 1,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: 1,
        maximum_nprobes: None,
        search_effort: 0.5,
        ef: None,
        refine_factor: None,
        metric_type: None,
        use_index: true,
        query_parallelism: 1,
        dist_q_c: 0.0,
        approx_mode: ApproxMode::Accurate,
    }
}

fn probes(
    index: &dyn VectorIndex,
    query: &Query,
    threshold: f32,
    options: SimilaritySearchOptions,
) -> Result<Vec<(usize, f32)>> {
    let mut query = query.clone();
    query.maximum_nprobes = options
        .nprobes
        .or(options.max_nprobes)
        .or(Some(DEFAULT_AUTO_MAX_PROBES))
        .map(|n| n.min(index.total_partitions()));
    let (ids, distances) = index.find_partitions(&query)?;
    if options.nprobes.is_some() {
        return Ok(ids
            .values()
            .iter()
            .zip(distances.values())
            .map(|(&p, &d)| (p as usize, d))
            .collect());
    }
    let mut selected = Vec::new();
    let Some(&nearest) = ids.values().first() else {
        return Ok(selected);
    };
    let metric = index.metric_type();
    let first = index.ivf_model().centroid(nearest as usize);
    for (&id, &distance) in ids.values().iter().zip(distances.values()) {
        let include = if id == nearest {
            true
        } else if metric == DistanceType::Hamming {
            distance - distances.value(0) <= 2.0 * threshold
        } else if matches!(metric, DistanceType::L2 | DistanceType::Cosine)
            && query.key.len() == index.ivf_model().dimension()
        {
            let a = cast(
                first
                    .as_ref()
                    .ok_or_else(|| Error::internal("missing centroid"))?
                    .as_ref(),
                &DataType::Float64,
            )?;
            let b = cast(
                &index
                    .ivf_model()
                    .centroid(id as usize)
                    .ok_or_else(|| Error::internal("missing centroid"))?,
                &DataType::Float64,
            )?;
            let a = a.as_any().downcast_ref::<Float64Array>().unwrap();
            let b = b.as_any().downcast_ref::<Float64Array>().unwrap();
            let separation = a
                .values()
                .iter()
                .zip(b.values())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            let radius = if metric == DistanceType::Cosine {
                (2.0 * threshold as f64).sqrt()
            } else {
                (threshold as f64).sqrt()
            };
            separation == 0.0
                || (distance as f64 - distances.value(0) as f64) <= 2.0 * separation * radius
        } else {
            true
        };
        if include {
            selected.push((id as usize, distance));
        }
    }
    Ok(selected)
}

async fn query_stream(
    dataset: Arc<Dataset>,
    column: String,
    targets: Arc<Vec<Target>>,
    id: u64,
    raw_key: ArrayRef,
    threshold: f32,
    options: SimilaritySearchOptions,
) -> Result<SendableRecordBatchStream> {
    let schema = super::dedup::pair_schema();
    let output = stream::iter(0..targets.len())
        .then(move |target_id| {
            let dataset = dataset.clone();
            let column = column.clone();
            let targets = targets.clone();
            let raw_key = raw_key.clone();
            async move {
                let target = &targets[target_id];
                let metric = target.index.metric_type();
                let key = if metric == DistanceType::Cosine {
                    normalize_arrow(raw_key.as_ref())?.0
                } else {
                    raw_key.clone()
                };
                let query = make_query(column.clone(), key);
                let selected = probes(target.index.as_ref(), &query, threshold, options)?;
                Ok::<_, Error>(
                    stream::iter(selected)
                        .then(move |(partition, dist_q_c)| {
                            let dataset = dataset.clone();
                            let column = column.clone();
                            let targets = targets.clone();
                            let raw_key = raw_key.clone();
                            let mut query = query.clone();
                            query.dist_q_c = dist_q_c;
                            async move {
                                let target = &targets[target_id];
                                // Quantized cosine search uses squared distance on normalized inputs.
                                let radius = if metric == DistanceType::Cosine {
                                    2.0 * threshold
                                } else {
                                    threshold
                                };
                                let candidates = target
                                    .index
                                    .range_search_in_partition(
                                        partition,
                                        &query,
                                        radius,
                                        target.filter.clone(),
                                        &NoOpMetricsCollector,
                                    )
                                    .await?;
                                let batches = (0..candidates.num_rows())
                                    .step_by(VERIFY_BATCH_ROWS)
                                    .map(move |offset| {
                                        candidates.slice(
                                            offset,
                                            VERIFY_BATCH_ROWS.min(candidates.num_rows() - offset),
                                        )
                                    })
                                    .collect::<Vec<_>>();
                                Ok::<_, Error>(
                                    stream::iter(batches)
                                        .then(move |batch| {
                                            verify(
                                                dataset.clone(),
                                                column.clone(),
                                                id,
                                                raw_key.clone(),
                                                metric,
                                                threshold,
                                                batch,
                                            )
                                        })
                                        .map_err(datafusion::error::DataFusionError::from),
                                )
                            }
                        })
                        .map_err(datafusion::error::DataFusionError::from)
                        .try_flatten(),
                )
            }
        })
        .map_err(datafusion::error::DataFusionError::from)
        .try_flatten();
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
}

async fn verify(
    dataset: Arc<Dataset>,
    column: String,
    id: u64,
    key: ArrayRef,
    metric: DistanceType,
    threshold: f32,
    candidates: RecordBatch,
) -> Result<RecordBatch> {
    let ids = candidates
        .column(1)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| Error::internal("invalid candidate IDs"))?;
    let ids = ids
        .values()
        .iter()
        .copied()
        .filter(|other| *other != id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let projection = ProjectionRequest::from_columns([column.as_str()], dataset.schema());
    let values = dataset.take_rows(&ids, projection).await?;
    if values.num_rows() != ids.len() {
        return Err(Error::internal("candidate missing from pinned snapshot"));
    }
    let vectors = values
        .column_by_qualified_name(&column)
        .ok_or_else(|| Error::invalid_input("vector column missing from projected candidates"))?
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| Error::internal("invalid candidate vectors"))?;
    let distances = (metric.arrow_batch_func())(key.as_ref(), vectors)?;
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut d = Vec::new();
    for (i, other) in ids.into_iter().enumerate() {
        let distance = distances.value(i);
        if distances.is_valid(i) && distance.is_finite() && distance <= threshold {
            a.push(id.min(other));
            b.push(id.max(other));
            d.push(distance);
        }
    }
    Ok(RecordBatch::try_new(
        super::dedup::pair_schema(),
        vec![
            Arc::new(UInt64Array::from(a)),
            Arc::new(UInt64Array::from(b)),
            Arc::new(Float32Array::from(d)),
        ],
    )?)
}
