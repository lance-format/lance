// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Execution node for MinHash LSH similarity search.
//!
//! Opens every segment of the MinHash index on the queried column, runs the
//! query text against each segment under the shared prefilter, and merges the
//! per-segment top-k hits by Jaccard distance. Distances do not depend on
//! corpus statistics, so the merge across segments is exact.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{ArrayRef, Float32Array, RecordBatch, UInt64Array};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::{Distribution, EquivalenceProperties, Partitioning};
use futures::stream::FuturesUnordered;
use futures::{StreamExt, TryStreamExt, future::try_join_all, stream};
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::utils::tracing::StreamTracingExt;
use lance_core::{Error, ROW_ID, Result};
use lance_index::prefilter::PreFilter;
use lance_index::scalar::ScalarIndex;
use lance_index::scalar::minhash_lsh::{
    MinHashHit, MinHashLshIndex, MinHashLshIndexParams, MinHashQuery, QuerySignature,
    SignatureGenerator, SignatureValue, TopHits, estimate_jaccard, text_values,
};
use lance_select::RowAddrMask;
use lance_table::format::IndexMetadata;
use tracing::instrument;

use super::PreFilterSource;
use super::knn::KNN_INDEX_SCHEMA;
use super::utils::{IndexMetrics, PreFilterMasks, build_prefilter};
use crate::Dataset;
use crate::index::{DatasetIndexInternalExt, validate_segment_params_compatible};

async fn open_minhash_segment(
    dataset: &Dataset,
    column: &str,
    segment: &IndexMetadata,
    metrics: &IndexMetrics,
) -> Result<Arc<dyn ScalarIndex>> {
    let index = dataset
        .open_scalar_index(column, &segment.uuid, metrics)
        .await?;
    if index.as_any().downcast_ref::<MinHashLshIndex>().is_none() {
        return Err(Error::invalid_input(format!(
            "Index segment {} on column {column} is a {} index, not a MinHash LSH index",
            segment.uuid,
            index.index_type()
        )));
    }
    Ok(index)
}

fn minhash_segment(index: &dyn ScalarIndex) -> DataFusionResult<&MinHashLshIndex> {
    index
        .as_any()
        .downcast_ref::<MinHashLshIndex>()
        .ok_or_else(|| {
            DataFusionError::Execution("opened index is not a MinHash LSH index".to_string())
        })
}

/// Searches every segment of a MinHash LSH index and emits the `limit` rows
/// with the smallest Jaccard distance as `_distance` and `_rowid`, ordered by
/// ascending distance.
#[derive(Debug)]
pub struct MinHashSearchExec {
    dataset: Arc<Dataset>,
    query: MinHashQuery,
    limit: usize,
    segments: Arc<[IndexMetadata]>,
    prefilter_source: PreFilterSource,
    /// Rows whose indexed value was replaced by a data overlay after the
    /// index was built; they are excluded from the results.
    overlay_block: Option<RowAddrMask>,
    external_mask: Option<Arc<RowAddrMask>>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl MinHashSearchExec {
    pub fn new(
        dataset: Arc<Dataset>,
        query: MinHashQuery,
        limit: usize,
        segments: Vec<IndexMetadata>,
        prefilter_source: PreFilterSource,
        overlay_block: Option<RowAddrMask>,
        external_mask: Option<Arc<RowAddrMask>>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(KNN_INDEX_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            dataset,
            query,
            limit,
            segments: Arc::from(segments),
            prefilter_source,
            overlay_block,
            external_mask,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for MinHashSearchExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "MinHashSearch: column={}, limit={}, segments={}",
                    self.query.column,
                    self.limit,
                    self.segments.len()
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "MinHashSearch\ncolumn={}\nlimit={}",
                    self.query.column, self.limit
                )
            }
        }
    }
}

impl ExecutionPlan for MinHashSearchExec {
    fn name(&self) -> &str {
        "MinHashSearchExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.prefilter_source.execution_plan().into_iter().collect()
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let prefilter_source = match children.len() {
            0 => {
                if !matches!(self.prefilter_source, PreFilterSource::None) {
                    return Err(DataFusionError::Internal(
                        "MinHashSearchExec with a prefilter requires one child".to_string(),
                    ));
                }
                PreFilterSource::None
            }
            1 => {
                let child = children.pop().ok_or_else(|| {
                    DataFusionError::Internal("child vanished after length check".to_string())
                })?;
                self.prefilter_source.with_execution_plan(child)?
            }
            other => {
                return Err(DataFusionError::Internal(format!(
                    "MinHashSearchExec accepts at most one child, got {other}"
                )));
            }
        };
        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            query: self.query.clone(),
            limit: self.limit,
            segments: self.segments.clone(),
            prefilter_source,
            overlay_block: self.overlay_block.clone(),
            external_mask: self.external_mask.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(name = "minhash_search_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let dataset = self.dataset.clone();
        let query = self.query.clone();
        let limit = self.limit;
        let segments = self.segments.clone();
        let prefilter_source = self.prefilter_source.clone();
        let overlay_block = self.overlay_block.clone();
        let external_mask = self.external_mask.clone();
        let index_metrics = Arc::new(IndexMetrics::new(&self.metrics, partition));
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let stream = stream::once(async move {
            let _timer = baseline_metrics.elapsed_compute().timer();
            let empty = || hits_batch(&[]);
            // Signatures are only comparable across segments built with the
            // same parameters; guard against segments that bypassed the
            // commit-time validation (for example written by external tools).
            validate_segment_params_compatible(&[], &segments)?;
            let indices = try_join_all(segments.iter().map(|segment| {
                open_minhash_segment(&dataset, &query.column, segment, &index_metrics)
            }))
            .await?;
            let Some(first_index) = indices.first() else {
                return empty();
            };
            // One signature serves every segment; the query text can be a whole document.
            let Some(query_signature) =
                minhash_segment(first_index.as_ref())?.query_signature(&query.text)
            else {
                return empty();
            };
            let query_signature = Arc::new(query_signature);
            let pre_filter = build_prefilter(
                context,
                partition,
                &prefilter_source,
                dataset,
                &segments,
                PreFilterMasks {
                    overlay_block,
                    external_mask,
                },
            )?;
            pre_filter.wait_for_ready().await?;
            let mask = pre_filter.mask();

            // Built with a loop rather than `map` so the futures own their
            // `Arc<dyn ScalarIndex>` without a higher-ranked closure lifetime.
            let mut searches = Vec::with_capacity(indices.len());
            for index in &indices {
                let index: Arc<dyn ScalarIndex> = index.clone();
                let mask = mask.clone();
                let query_signature = query_signature.clone();
                let metrics = index_metrics.clone();
                searches.push(async move {
                    minhash_segment(index.as_ref())?
                        .search_signature(&query_signature, limit, &mask, metrics.as_ref())
                        .await
                        .map_err(DataFusionError::from)
                });
            }
            let mut merged = TopHits::new(limit);
            let mut per_segment = stream::iter(searches)
                .buffer_unordered(get_num_compute_intensive_cpus().min(indices.len().max(1)));
            while let Some(hits) = per_segment.try_next().await? {
                for hit in hits {
                    merged.push(hit);
                }
            }
            let hits = merged.into_sorted();
            index_metrics.flush_io();
            baseline_metrics.record_output(hits.len());
            hits_batch(&hits)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            KNN_INDEX_SCHEMA.clone(),
            stream.stream_in_current_span().boxed(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

fn hits_batch(hits: &[MinHashHit]) -> DataFusionResult<RecordBatch> {
    let distances = Float32Array::from_iter_values(hits.iter().map(|hit| hit.distance));
    let row_ids = UInt64Array::from_iter_values(hits.iter().map(|hit| hit.row_id));
    Ok(RecordBatch::try_new(
        KNN_INDEX_SCHEMA.clone(),
        vec![
            Arc::new(distances) as ArrayRef,
            Arc::new(row_ids) as ArrayRef,
        ],
    )?)
}

/// Score one input batch: rows that share at least one band with the query
/// (the index's own candidate rule) keep their Jaccard distance; at most
/// `limit` hits survive so the caller merges only tiny per-batch results.
fn score_flat_batch(
    mut generator: SignatureGenerator,
    query: &QuerySignature,
    batch: &RecordBatch,
    column: &str,
    limit: usize,
) -> Result<Vec<MinHashHit>> {
    let row_ids = batch
        .column_by_name(ROW_ID)
        .and_then(|column| column.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| {
            Error::internal(format!(
                "flat MinHash input batch has no UInt64 column {ROW_ID}"
            ))
        })?;
    let values = batch.column_by_name(column).ok_or_else(|| {
        Error::internal(format!("flat MinHash input batch has no column {column}"))
    })?;
    let mut hits = TopHits::new(limit);
    let mut signature = vec![SignatureValue::MAX; generator.num_hashes()];
    let mut band_keys = Vec::with_capacity(generator.num_bands());
    for (row, text) in text_values(values.as_ref())?.enumerate() {
        if let Some(text) = text
            && generator.signature(text, &mut signature)
            && query.shares_band(&generator, &signature, &mut band_keys)
        {
            hits.push(MinHashHit {
                row_id: row_ids.value(row),
                distance: 1.0 - estimate_jaccard(query.signature(), &signature),
            });
        }
    }
    Ok(hits.into_sorted())
}

/// Rows per signing task on the flat path: scan batches are split this fine so
/// a scan of a few large batches still spreads over every CPU.
const FLAT_SIGN_CHUNK_ROWS: usize = 1024;

/// Scores rows the index does not cover (unindexed fragments and rows whose
/// value was replaced by a data overlay) by computing their signatures on the
/// fly, emitting the `limit` closest rows as `_distance` and `_rowid`.
///
/// The input scans only those rows, projected to the text column and
/// `_rowid`; each batch is signed on the CPU pool with a per-batch top-k so
/// memory stays bounded by the batches in flight.
#[derive(Debug)]
pub struct FlatMinHashExec {
    input: Arc<dyn ExecutionPlan>,
    query: MinHashQuery,
    params: MinHashLshIndexParams,
    limit: usize,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl FlatMinHashExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        query: MinHashQuery,
        params: MinHashLshIndexParams,
        limit: usize,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(KNN_INDEX_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            input,
            query,
            params,
            limit,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for FlatMinHashExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "MinHashFlatSearch: column={}, limit={}",
                    self.query.column, self.limit
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "MinHashFlatSearch\ncolumn={}\nlimit={}",
                    self.query.column, self.limit
                )
            }
        }
    }
}

impl ExecutionPlan for FlatMinHashExec {
    fn name(&self) -> &str {
        "FlatMinHashExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // `execute` reads a single input partition; without this the input
        // could be split across partitions of which only one is consumed.
        vec![Distribution::SinglePartition]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "FlatMinHashExec requires one child, got {}",
                children.len()
            )));
        }
        let input = children.pop().ok_or_else(|| {
            DataFusionError::Internal("child vanished after length check".to_string())
        })?;
        Ok(Arc::new(Self {
            input,
            query: self.query.clone(),
            params: self.params.clone(),
            limit: self.limit,
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(name = "flat_minhash_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let mut input = self.input.execute(partition, context)?;
        let query = self.query.clone();
        let params = self.params.clone();
        let limit = self.limit;
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let stream = stream::once(async move {
            let mut generator = SignatureGenerator::try_new(&params)?;
            let Some(query_signature) = QuerySignature::compute(&mut generator, &query.text) else {
                return hits_batch(&[]);
            };
            let query_signature = Arc::new(query_signature);
            let column = Arc::new(query.column);
            let mut hits = TopHits::new(limit);
            let mut in_flight = FuturesUnordered::new();
            let concurrency = get_num_compute_intensive_cpus().max(1);
            while let Some(batch) = input.try_next().await? {
                for offset in (0..batch.num_rows()).step_by(FLAT_SIGN_CHUNK_ROWS) {
                    let chunk =
                        batch.slice(offset, FLAT_SIGN_CHUNK_ROWS.min(batch.num_rows() - offset));
                    let generator = generator.clone();
                    let query_signature = query_signature.clone();
                    let column = column.clone();
                    in_flight.push(spawn_cpu(move || {
                        score_flat_batch(generator, &query_signature, &chunk, &column, limit)
                    }));
                    if in_flight.len() >= concurrency
                        && let Some(batch_hits) = in_flight.next().await
                    {
                        hits.extend(batch_hits?);
                    }
                }
            }
            while let Some(batch_hits) = in_flight.next().await {
                hits.extend(batch_hits?);
            }
            let hits = hits.into_sorted();
            baseline_metrics.record_output(hits.len());
            hits_batch(&hits)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            KNN_INDEX_SCHEMA.clone(),
            stream.stream_in_current_span().boxed(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, Int32Type, UInt64Type};
    use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use lance_core::{Error, ROW_ID};
    use lance_index::IndexType;
    use lance_index::scalar::inverted::query::MatchQuery;
    use lance_index::scalar::minhash_lsh::MinHashQuery;
    use lance_index::scalar::{BuiltinIndexType, FullTextSearchQuery, ScalarIndexParams};
    use lance_select::{RowAddrMask, RowAddrTreeMap};

    use crate::dataset::scanner::Scanner;
    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::{Dataset, Result};

    const BASE: &str = "the quick brown fox jumps over the lazy dog and runs away very fast";
    const NEAR: &str = "the quick brown fox jumps over the lazy dog and runs away very quickly";

    fn texts() -> Vec<&'static str> {
        vec![
            BASE,
            NEAR,
            "completely unrelated sentence about columnar storage in lance files",
            BASE,
            "another unrelated row that talks about vector indices and recall",
            "yet another unrelated row discussing full text search tokenizers",
            "a row about fragments compaction and stable row identifiers",
            "a row about the manifest and transaction commit protocol",
            "some words about python bindings and pyarrow tables",
            "some words about java bindings and arrow record batches",
            "rows describing bloom filters and zone maps for pruning",
            "rows describing btree pages and bitmap postings for equality",
        ]
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
        ]))
    }

    fn batch(first_id: i32, texts: &[&str]) -> RecordBatch {
        let ids = Int32Array::from_iter_values(first_id..first_id + texts.len() as i32);
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(ids) as ArrayRef,
                Arc::new(StringArray::from(texts.to_vec())) as ArrayRef,
            ],
        )
        .unwrap()
    }

    /// Twelve rows split over three fragments with a MinHash index on `text`.
    async fn indexed_dataset() -> Dataset {
        let reader = RecordBatchIterator::new(vec![Ok(batch(0, &texts()))], schema());
        let mut dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments().len(), 3);
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::MinHashLsh)
            .with_params(&serde_json::json!({"num_hashes": 64, "num_bands": 16}));
        dataset
            .create_index(&["text"], IndexType::MinHashLsh, None, &params, true)
            .await
            .unwrap();
        dataset
    }

    /// Append `texts` as one new, unindexed fragment.
    async fn append(dataset: Dataset, first_id: i32, texts: &[&str]) -> Dataset {
        let reader = RecordBatchIterator::new(vec![Ok(batch(first_id, texts))], schema());
        Dataset::write(
            reader,
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    fn scan(dataset: &Dataset, text: &str, limit: i64) -> Scanner {
        let mut scan = dataset.scan();
        scan.minhash_search(MinHashQuery::new(text, "text"))
            .unwrap()
            .limit(Some(limit), None)
            .unwrap()
            .project(&["id"])
            .unwrap();
        scan
    }

    /// `(id, _distance)` of a MinHash scan configured by `configure`.
    async fn hits(
        dataset: &Dataset,
        text: &str,
        limit: i64,
        configure: impl FnOnce(&mut Scanner),
    ) -> Result<Vec<(i32, f32)>> {
        let mut scan = scan(dataset, text, limit);
        configure(&mut scan);
        let batch = scan.try_into_batch().await?;
        let ids = batch["id"].as_primitive::<Int32Type>();
        let distances = batch["_distance"].as_primitive::<Float32Type>();
        Ok(ids
            .values()
            .iter()
            .copied()
            .zip(distances.values().iter().copied())
            .collect())
    }

    async fn ids(
        dataset: &Dataset,
        text: &str,
        limit: i64,
        configure: impl FnOnce(&mut Scanner),
    ) -> Vec<i32> {
        hits(dataset, text, limit, configure)
            .await
            .unwrap()
            .into_iter()
            .map(|hit| hit.0)
            .collect()
    }

    async fn plan(dataset: &Dataset, configure: impl FnOnce(&mut Scanner)) -> String {
        let mut scan = scan(dataset, BASE, 4);
        configure(&mut scan);
        scan.explain_plan(false).await.unwrap()
    }

    #[tokio::test]
    async fn test_minhash_search_end_to_end() {
        let dataset = indexed_dataset().await;
        let explained = plan(&dataset, |_| {}).await;
        assert!(
            explained.contains("MinHashSearch: column=text, limit=4"),
            "{explained}"
        );

        let found = hits(&dataset, BASE, 3, |_| {}).await.unwrap();
        assert_eq!(found[0], (0, 0.0));
        assert_eq!(found[1], (3, 0.0));
        assert_eq!(found[2].0, 1);
        assert!(found[2].1 > 0.0 && found[2].1 < 0.5, "{found:?}");
        // A query with no similar rows returns an empty result, not an error
        assert!(
            hits(&dataset, "nothing shares any shingle here", 3, |_| {})
                .await
                .unwrap()
                .is_empty()
        );
        // Offset rows are skipped after ranking
        let offset = ids(&dataset, BASE, 2, |scan| {
            scan.limit(Some(2), Some(1)).unwrap();
        })
        .await;
        assert_eq!(offset, vec![3, 1]);
    }

    #[tokio::test]
    async fn test_minhash_search_applies_filters_and_deletes() {
        let mut dataset = indexed_dataset().await;
        let prefiltered = ids(&dataset, BASE, 2, |scan| {
            scan.filter("id >= 1").unwrap().prefilter(true);
        })
        .await;
        assert_eq!(prefiltered, vec![3, 1]);
        // Postfiltering ranks first and filters the ranked rows afterwards
        let postfiltered = ids(&dataset, BASE, 2, |scan| {
            scan.filter("id >= 1").unwrap().prefilter(false);
        })
        .await;
        assert_eq!(postfiltered, vec![3]);

        dataset.delete("id = 3").await.unwrap();
        assert_eq!(ids(&dataset, BASE, 2, |_| {}).await, vec![0, 1]);
        // Deleting every row of a fragment drops it from the manifest while
        // the index still lists it; the prefilter masks those rows too.
        let dropped = texts()[6];
        assert_eq!(ids(&dataset, dropped, 2, |_| {}).await, vec![6]);
        dataset.delete("id >= 4 AND id <= 7").await.unwrap();
        assert_eq!(dataset.fragments().len(), 2);
        assert!(ids(&dataset, dropped, 2, |_| {}).await.is_empty());
        assert_eq!(ids(&dataset, BASE, 3, |_| {}).await, vec![0, 1]);
    }

    #[tokio::test]
    async fn test_minhash_search_covers_unindexed_fragments() {
        let dataset = append(indexed_dataset().await, 12, &[BASE, "unrelated append"]).await;
        assert_eq!(dataset.fragments().len(), 4);

        // Appended rows are scored on the fly and merged with the index hits
        let explained = plan(&dataset, |_| {}).await;
        assert!(
            explained.contains("MinHashSearch: column=text")
                && explained.contains("MinHashFlatSearch: column=text, limit=4")
                && explained.contains("SortExec"),
            "{explained}"
        );
        let found = hits(&dataset, BASE, 4, |_| {}).await.unwrap();
        assert_eq!(
            found.iter().map(|hit| hit.0).collect::<Vec<_>>(),
            vec![0, 3, 12, 1],
            "{found:?}"
        );
        assert_eq!(found[2].1, 0.0);
        // Prefilters reach the flat path too
        let filtered = ids(&dataset, BASE, 4, |scan| {
            scan.filter("id >= 12").unwrap().prefilter(true);
        })
        .await;
        assert_eq!(filtered, vec![12]);

        // Only the unindexed fragment: the plan has no index branch
        let appended = dataset.fragments()[3].clone();
        let explained = plan(&dataset, |scan| {
            scan.with_fragments(vec![appended.clone()]);
        })
        .await;
        assert!(!explained.contains("MinHashSearch: column"), "{explained}");
        assert!(explained.contains("MinHashFlatSearch"), "{explained}");
        let flat_only = ids(&dataset, BASE, 4, |scan| {
            scan.with_fragments(vec![appended]);
        })
        .await;
        assert_eq!(flat_only, vec![12]);

        // fast_search searches the index only
        let explained = plan(&dataset, |scan| {
            scan.fast_search();
        })
        .await;
        assert!(!explained.contains("MinHashFlatSearch"), "{explained}");
        let fast = ids(&dataset, BASE, 4, |scan| {
            scan.fast_search();
        })
        .await;
        assert_eq!(fast, vec![0, 3, 1]);
    }

    #[tokio::test]
    async fn test_minhash_search_applies_external_mask_before_flat_top_k() {
        // Two exact copies of BASE land in an unindexed fragment; with limit 1
        // the flat branch keeps only the first, so a mask allowing only the
        // second must be applied before that top-k, not after it.
        let dataset = append(indexed_dataset().await, 12, &[BASE, BASE]).await;
        let batch = dataset
            .scan()
            .filter("id = 13")
            .unwrap()
            .with_row_id()
            .project(&["id"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let allowed = batch[ROW_ID].as_primitive::<UInt64Type>().value(0);
        let found = hits(&dataset, BASE, 1, |scan| {
            scan.with_row_addr_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([
                allowed,
            ])));
        })
        .await
        .unwrap();
        assert_eq!(found, vec![(13, 0.0)]);
    }

    #[tokio::test]
    async fn test_minhash_search_rejects_invalid_scans() {
        let dataset = indexed_dataset().await;

        let mut scan = dataset.scan();
        scan.minhash_search(MinHashQuery::new(BASE, "text"))
            .unwrap();
        let err = scan.try_into_batch().await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(err.to_string().contains("requires a limit"), "{err}");

        let mut scan = dataset.scan();
        scan.minhash_search(MinHashQuery::new(BASE, "id"))
            .unwrap()
            .limit(Some(3), None)
            .unwrap();
        let err = scan.try_into_batch().await.unwrap_err();
        assert!(
            err.to_string()
                .contains("No MinHash LSH index found for column id"),
            "{err}"
        );

        let mut scan = dataset.scan();
        assert!(matches!(
            scan.minhash_search(MinHashQuery::new(BASE, "")),
            Err(Error::InvalidInput { .. })
        ));

        let err = hits(&dataset, BASE, 3, |scan| {
            scan.full_text_search(FullTextSearchQuery::new_query(
                MatchQuery::new("nothing".to_owned())
                    .with_column(Some("text".to_owned()))
                    .into(),
            ))
            .unwrap();
        })
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("Cannot combine a MinHash search"),
            "{err}"
        );

        let err = hits(&dataset, BASE, 3, |scan| {
            scan.with_row_id().include_deleted_rows();
        })
        .await
        .unwrap_err();
        assert!(err.to_string().contains("deleted rows"), "{err}");
    }
}
