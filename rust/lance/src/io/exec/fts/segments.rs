// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! FTS segment access, coverage filtering, and posting search helpers.
//!
//! Segment algorithms live in `lance-index`; this module only adapts dataset
//! metadata and prefilters into calls on opened inverted indexes.

use std::sync::Arc;

use futures::TryStreamExt;
use futures::future::try_join_all;
use futures::stream::{self, StreamExt};
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{Error, Result};
use lance_index::metrics::MetricsCollector;
use lance_index::prefilter::PreFilter;
use lance_index::scalar::inverted::builder::ScoredDoc;
use lance_index::scalar::inverted::query::{FtsSearchParams, Operator, Tokens};
use lance_index::scalar::inverted::{CombinedFieldColumn, InvertedIndex, MemBM25Scorer};
use lance_table::format::IndexMetadata;

use super::matches::PreparedMatch;
use super::selection::FtsIndexMetrics;
use crate::Dataset;
use crate::index::loading::DatasetIndexInternalExt;
use crate::index::prefilter::prefilter_may_match_segment;
use crate::index::segment_coverage::{SegmentCoverage, SegmentRowDomain};

/// Open one FTS segment as an [`InvertedIndex`].
///
/// With `coverage` the segment may only report live rows currently stored in the fragments
/// it covers (see [`open_fts_segments`]).
pub(super) async fn open_fts_segment(
    dataset: &Dataset,
    column: &str,
    segment: &IndexMetadata,
    coverage: Option<(&SegmentCoverage, usize)>,
    metrics: &dyn MetricsCollector,
) -> Result<Arc<InvertedIndex>> {
    let open = async {
        let index = dataset
            .open_scalar_index(column, &segment.uuid, metrics)
            .await?;
        index
            .as_any()
            .downcast_ref::<InvertedIndex>()
            .cloned()
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Index for column {} and segment {} is not an inverted index",
                    column, segment.uuid
                ))
            })
    };
    let allowed_rows = async {
        match coverage {
            Some((coverage, index)) => Ok(Some(coverage.live_rows(index).await?)),
            None => Ok(None),
        }
    };
    let (inverted, allowed_rows) = futures::try_join!(open, allowed_rows)?;
    Ok(Arc::new(match allowed_rows {
        Some(allowed_rows) => inverted.with_allowed_rows(allowed_rows),
        None => inverted,
    }))
}

/// Open all committed FTS segments for a column.
///
/// Exact multi-segment BM25 still needs every segment's local corpus statistics, so the
/// current correctness-first path opens each committed segment before scoring.
///
/// Under stable row ids a rewritten row keeps its row id while it moves to a fragment
/// another segment may cover, so the old segment still lists it under its old tokens. The
/// shared prefilter only knows the union of all segments, so each segment is restricted to
/// the live rows currently stored in the fragments it covers.
pub(super) async fn open_fts_segments(
    dataset: &Dataset,
    column: &str,
    segments: &[IndexMetadata],
    metrics: &dyn MetricsCollector,
) -> Result<Vec<Arc<InvertedIndex>>> {
    let segment_refs = segments.iter().collect::<Vec<_>>();
    let coverage =
        SegmentCoverage::resolve(dataset, &segment_refs, SegmentRowDomain::RowId).await?;
    try_join_all(segments.iter().enumerate().map(|(index, segment)| {
        let coverage = coverage.as_ref().map(|coverage| (coverage, index));
        open_fts_segment(dataset, column, segment, coverage, metrics)
    }))
    .await
}

/// Open all committed FTS segments without resolving live-row coverage.
///
/// Global statistics describe the committed corpus rather than the rows currently stored
/// in each segment's fragments, so resolving coverage would add deletion and row-id I/O
/// without changing the result.
pub async fn open_fts_segments_without_coverage(
    dataset: &Dataset,
    column: &str,
    segments: &[IndexMetadata],
    metrics: &dyn MetricsCollector,
) -> Result<Vec<Arc<InvertedIndex>>> {
    try_join_all(
        segments
            .iter()
            .map(|segment| open_fts_segment(dataset, column, segment, None, metrics)),
    )
    .await
}

/// The segments whose candidate search the prefilter cannot rule out, in
/// `indices` order; `segments` is their metadata in the same order and the
/// prefilter must be ready.
///
/// Corpus statistics and vocabulary expansion were done over every segment
/// before this point and are not affected: only the candidate search of a
/// segment the complete prefilter provably excludes is skipped. Under stable
/// row ids a segment restricted by the opener decides through its allowed
/// rows; otherwise its resolved query coverage (not its stored provenance)
/// decides by fragment key. When that coverage cannot be resolved, or the
/// inputs do not line up, every segment is kept.
pub(super) async fn segments_the_prefilter_may_match(
    dataset: &Dataset,
    indices: &[Arc<InvertedIndex>],
    segments: &[IndexMetadata],
    pre_filter: &dyn PreFilter,
    address_domain: bool,
    metrics: &FtsIndexMetrics,
) -> Vec<Arc<InvertedIndex>> {
    if indices.len() != segments.len() {
        return indices.to_vec();
    }
    let segment_refs: Vec<&IndexMetadata> = segments.iter().collect();
    let coverages = if address_domain {
        match crate::index::frag_reuse::resolve_query_segment_coverages(dataset, &segment_refs)
            .await
        {
            Ok(coverages) => Some(coverages),
            Err(err) => {
                log::debug!("keeping every FTS segment: coverage unresolved ({err})");
                None
            }
        }
    } else {
        None
    };
    let mask = pre_filter.mask();
    let retained: Vec<Arc<InvertedIndex>> = indices
        .iter()
        .enumerate()
        .filter(|(position, index)| {
            prefilter_may_match_segment(
                &mask,
                index.allowed_rows().map(Arc::as_ref),
                coverages.as_ref().map(|coverages| &coverages[*position]),
                address_domain,
            )
        })
        .map(|(_, index)| index.clone())
        .collect();
    metrics.record_segments_skipped(indices.len() - retained.len());
    retained
}

/// [`segments_the_prefilter_may_match`] per target column of a combined
/// fields scan, whose `segments` list every column's segments in column
/// order. A column may end up with no segment to search.
pub(super) async fn combined_columns_the_prefilter_may_match(
    dataset: &Dataset,
    columns: Vec<CombinedFieldColumn>,
    segments: &[IndexMetadata],
    pre_filter: &dyn PreFilter,
    address_domain: bool,
    metrics: &FtsIndexMetrics,
) -> Vec<CombinedFieldColumn> {
    let total: usize = columns.iter().map(|column| column.indices.len()).sum();
    if total != segments.len() {
        return columns;
    }
    let mut offset = 0;
    let mut retained = Vec::with_capacity(columns.len());
    for column in columns {
        let count = column.indices.len();
        let indices = segments_the_prefilter_may_match(
            dataset,
            &column.indices,
            &segments[offset..offset + count],
            pre_filter,
            address_domain,
            metrics,
        )
        .await;
        offset += count;
        retained.push(CombinedFieldColumn { indices, ..column });
    }
    retained
}

pub(super) async fn search_prepared_segments(
    indices: &[Arc<InvertedIndex>],
    prepared: Arc<PreparedMatch>,
    pre_filter: Arc<dyn PreFilter>,
    metrics: Arc<FtsIndexMetrics>,
    initial_score_floor: Option<f32>,
) -> Result<Vec<ScoredDoc>> {
    let limit = prepared.params.limit.unwrap_or(usize::MAX);
    let mut candidates = std::collections::BinaryHeap::new();
    let searches = indices
        .iter()
        .map(|index| {
            let index = Arc::clone(index);
            let prepared = prepared.clone();
            let pre_filter = pre_filter.clone();
            let metrics = metrics.clone();
            async move {
                if let Some(initial_score_floor) = initial_score_floor {
                    index
                        .bm25_search_prepared_documents_with_score_floor(
                            prepared.query.clone(),
                            prepared.params.clone(),
                            prepared.operator,
                            pre_filter,
                            metrics,
                            initial_score_floor,
                        )
                        .await
                } else {
                    index
                        .bm25_search_prepared_documents(
                            prepared.query.clone(),
                            prepared.params.clone(),
                            prepared.operator,
                            pre_filter,
                            metrics,
                        )
                        .await
                }
            }
        })
        .collect::<Vec<_>>();
    let searches = stream::iter(searches).buffer_unordered(get_num_compute_intensive_cpus());
    let mut searches = searches;

    while let Some(documents) = searches.try_next().await? {
        for document in documents {
            if candidates.len() < limit {
                candidates.push(std::cmp::Reverse(document));
            } else if candidates.peek().unwrap().0.score < document.score {
                candidates.pop();
                candidates.push(std::cmp::Reverse(document));
            }
        }
    }

    Ok(candidates
        .into_sorted_vec()
        .into_iter()
        .map(|std::cmp::Reverse(document)| document)
        .collect())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn search_segments(
    indices: &[Arc<InvertedIndex>],
    tokens: Arc<Tokens>,
    params: Arc<FtsSearchParams>,
    operator: Operator,
    pre_filter: Arc<dyn PreFilter>,
    metrics: Arc<FtsIndexMetrics>,
    base_scorer: Arc<MemBM25Scorer>,
    initial_score_floor: Option<f32>,
) -> Result<Vec<ScoredDoc>> {
    let limit = params.limit.unwrap_or(usize::MAX);
    let mut candidates = std::collections::BinaryHeap::new();
    let searches = indices
        .iter()
        .map(|index| {
            let index = Arc::clone(index);
            let tokens = tokens.clone();
            let params = params.clone();
            let pre_filter = pre_filter.clone();
            let metrics = metrics.clone();
            let base_scorer = base_scorer.clone();
            async move {
                if let Some(initial_score_floor) = initial_score_floor {
                    index
                        .bm25_search_documents_with_score_floor(
                            tokens,
                            params,
                            operator,
                            pre_filter,
                            metrics,
                            Some(base_scorer.as_ref()),
                            initial_score_floor,
                        )
                        .await
                } else {
                    index
                        .bm25_search_documents(
                            tokens,
                            params,
                            operator,
                            pre_filter,
                            metrics,
                            Some(base_scorer.as_ref()),
                        )
                        .await
                }
            }
        })
        .collect::<Vec<_>>();
    let searches = stream::iter(searches).buffer_unordered(get_num_compute_intensive_cpus());
    let mut searches = searches;

    while let Some(documents) = searches.try_next().await? {
        for document in documents {
            if candidates.len() < limit {
                candidates.push(std::cmp::Reverse(document));
            } else if candidates.peek().unwrap().0.score < document.score {
                candidates.pop();
                candidates.push(std::cmp::Reverse(document));
            }
        }
    }

    Ok(candidates
        .into_sorted_vec()
        .into_iter()
        .map(|std::cmp::Reverse(document)| document)
        .collect())
}
