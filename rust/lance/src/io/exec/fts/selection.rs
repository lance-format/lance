// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exact FTS segment selection and execution metrics.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, Gauge, Time,
};
use lance_datafusion::utils::{
    ExecutionPlanMetricsSetExt, INDEX_SEGMENTS_SKIPPED_METRIC, PARTITIONS_SEARCHED_METRIC,
};
use lance_index::metrics::MetricsCollector;
use lance_index::scalar::inverted::DocumentGranularity;
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use crate::Dataset;
use crate::index::scalar::inverted::{load_segment_details, load_segments};
use crate::io::exec::utils::IndexMetrics;

/// Time spent resolving an exact ordered UUID selection to committed FTS segments.
pub const FTS_SEGMENT_BIND_DURATION_METRIC: &str = "fts_segment_bind_duration";

#[derive(Debug, Clone)]
pub(super) enum FtsSegmentSelection {
    AllCommitted,
    ExactResolved(Arc<[IndexMetadata]>),
    ExactUuids(Arc<[Uuid]>),
}

impl FtsSegmentSelection {
    pub(super) fn exact_uuids(mut uuids: Vec<Uuid>) -> Self {
        let mut seen = HashSet::with_capacity(uuids.len());
        uuids.retain(|uuid| seen.insert(*uuid));
        Self::ExactUuids(Arc::from(uuids))
    }

    pub(super) fn preset_segments(&self) -> Option<&[IndexMetadata]> {
        match self {
            Self::ExactResolved(segments) => Some(segments),
            Self::AllCommitted | Self::ExactUuids(_) => None,
        }
    }

    pub(super) fn searches_all_committed(&self) -> bool {
        matches!(self, Self::AllCommitted)
    }

    pub(super) async fn covers_all_committed(
        &self,
        dataset: &Dataset,
        column: &str,
        document_granularity: DocumentGranularity,
        resolved: &[IndexMetadata],
    ) -> DataFusionResult<bool> {
        if self.searches_all_committed() {
            return Ok(true);
        }
        let Some(committed) = load_segments(dataset, column, document_granularity).await? else {
            return Ok(false);
        };
        let selected = resolved
            .iter()
            .map(|segment| segment.uuid)
            .collect::<HashSet<_>>();
        let committed = committed
            .iter()
            .map(|segment| segment.uuid)
            .collect::<HashSet<_>>();
        Ok(selected == committed)
    }

    pub(super) fn explicit_segment_uuids(&self) -> Option<Vec<Uuid>> {
        match self {
            Self::AllCommitted => None,
            Self::ExactResolved(segments) => {
                Some(segments.iter().map(|segment| segment.uuid).collect())
            }
            Self::ExactUuids(uuids) => Some(uuids.to_vec()),
        }
    }

    pub(super) async fn resolve(
        &self,
        dataset: &Dataset,
        column: &str,
        document_granularity: DocumentGranularity,
        segment_bind_duration: &Time,
    ) -> DataFusionResult<Arc<[IndexMetadata]>> {
        let segments = match self {
            Self::AllCommitted => load_segments(dataset, column, document_granularity)
                .await?
                .map(Arc::from)
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "No Inverted index found for column {}",
                        column,
                    ))
                }),
            Self::ExactResolved(segments) => Ok(segments.clone()),
            Self::ExactUuids(uuids) => {
                let _timer = segment_bind_duration.timer();
                let dataset_version = dataset.version_id();
                if uuids.is_empty() {
                    return Err(DataFusionError::Execution(format!(
                        "Exact FTS segment selection for column {} at dataset version {} \
                         requires at least one segment UUID",
                        column, dataset_version
                    )));
                }

                let committed_segments = load_segments(dataset, column, document_granularity)
                    .await?
                    .ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "Cannot resolve exact FTS segment selection for column {} at dataset \
                             version {}: no Inverted index found",
                            column, dataset_version
                        ))
                    })?;
                let mut segments_by_uuid = HashMap::with_capacity(committed_segments.len());
                for segment in committed_segments {
                    let uuid = segment.uuid;
                    if segments_by_uuid.insert(uuid, segment).is_some() {
                        return Err(DataFusionError::Execution(format!(
                            "FTS metadata for column {} at dataset version {} contains duplicate \
                             segment UUID {}",
                            column, dataset_version, uuid
                        )));
                    }
                }

                let mut resolved = Vec::with_capacity(uuids.len());
                for uuid in uuids.iter() {
                    let segment = segments_by_uuid.get(uuid).ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "Requested FTS segment UUID {} for column {} is not committed in \
                             dataset version {}",
                            uuid, column, dataset_version
                        ))
                    })?;
                    resolved.push(segment.clone());
                }
                Ok(Arc::from(resolved))
            }
        }?;
        let details = load_segment_details(dataset, column, &segments).await?;
        let indexed_granularity = DocumentGranularity::try_from(details.document_granularity)?;
        if indexed_granularity != document_granularity {
            return Err(DataFusionError::Execution(format!(
                "FTS segments selected for column {column} use {indexed_granularity:?} document \
                 granularity, but the query was resolved as {document_granularity:?}"
            )));
        }
        Ok(segments)
    }
}

pub struct FtsIndexMetrics {
    pub(super) index_metrics: IndexMetrics,
    partitions_searched: Count,
    segments_skipped: Count,
    /// Wall time (ms) of the exec-local `build_global_bm25_scorer`
    /// fallback; zero when a preset base scorer was injected.
    scorer_build_ms: Gauge,
    pub(super) segment_bind_duration: Time,
    pub(super) baseline_metrics: BaselineMetrics,
}

impl FtsIndexMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            index_metrics: IndexMetrics::new(metrics, partition),
            partitions_searched: metrics.new_count(PARTITIONS_SEARCHED_METRIC, partition),
            segments_skipped: metrics.new_count(INDEX_SEGMENTS_SKIPPED_METRIC, partition),
            scorer_build_ms: metrics.new_gauge("scorer_build_ms", partition),
            segment_bind_duration: metrics.new_time(FTS_SEGMENT_BIND_DURATION_METRIC, partition),
            baseline_metrics: BaselineMetrics::new(metrics, partition),
        }
    }

    pub fn record_parts_searched(&self, num_parts: usize) {
        self.partitions_searched.add(num_parts);
    }

    pub fn record_segments_skipped(&self, num_segments: usize) {
        self.segments_skipped.add(num_segments);
    }

    pub fn record_scorer_build(&self, elapsed: std::time::Duration) {
        self.scorer_build_ms.set(elapsed.as_millis() as usize);
    }
}

impl MetricsCollector for FtsIndexMetrics {
    fn record_parts_loaded(&self, num_parts: usize) {
        self.index_metrics.record_parts_loaded(num_parts);
    }

    fn record_index_loads(&self, num_indexes: usize) {
        self.index_metrics.record_index_loads(num_indexes);
    }

    fn record_comparisons(&self, num_comparisons: usize) {
        self.index_metrics.record_comparisons(num_comparisons);
    }

    fn record_index_cache_hits(&self, num_hits: usize) {
        self.index_metrics.record_index_cache_hits(num_hits);
    }

    fn record_index_cache_misses(&self, num_misses: usize) {
        self.index_metrics.record_index_cache_misses(num_misses);
    }
}
