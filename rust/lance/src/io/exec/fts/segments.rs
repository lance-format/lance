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

/// Require that every FTS segment this query reads stores physical row
/// addresses, which the whole FTS execution path assumes.
///
/// Every node between the postings and the take above the plan carries the
/// identifier the postings store, under the name `_rowaddr`
/// ([`FTS_SCHEMA`](lance_index::scalar::inverted::FTS_SCHEMA)): nothing
/// translates, the plan's `(score DESC, identity ASC)` sort tiebreaks on the
/// same identifier top-k selection ranks by, and `TakeExec` reads it by
/// address. That holds for a dataset without stable row ids too, where a row
/// id *is* a row address: a segment there reports
/// `results_are_row_addrs() == false` only because `build_scalar_index`
/// relabels a fresh build down to a pre-[`INVERTED_ROW_ADDR_DOMAIN_VERSION`]
/// version so an older reader still understands it, not because its stored
/// values differ in kind. Mixed labels are therefore both expected and
/// harmless there, and are not checked.
///
/// With stable row ids the two domains genuinely differ, and a segment
/// persisted before [`INVERTED_ROW_ADDR_DOMAIN_VERSION`] really does store row
/// ids. There is no sound way to run it through an address-domain path --
/// translating its hits to addresses after selection would rank ties on row
/// ids while the sort above tiebreaks on addresses, which is the mismatch this
/// path exists to avoid -- so it is refused and the index must be rebuilt. The
/// state is transient: it only arises between turning stable row ids on and
/// the index's next rebuild or merge, which `optimize_indices`'
/// `legacy_domain_mismatch` already forces.
///
/// The same check catches segments that disagree with each other. A caller
/// passes every segment the query will read, which for a multi-column query
/// (`combined_fields`, a cross-column compound) spans *several named indexes*:
/// their postings are merged on the stored identifier alone, so one column's
/// row id `2` and another's row address `1 << 32` for the same row would be
/// scored as two unrelated documents. Within a single named index that mix is
/// already refused at write time -- `optimize_indices`' `legacy_domain_mismatch`
/// rebuilds rather than merging across domains, and `manifest_build` rejects a
/// commit that would publish one -- but nothing couples two columns' indexes,
/// and nothing could: rebuilding one column's index is a legitimate commit on
/// its own, and only a query spanning both makes the pair wrong.
///
/// [`INVERTED_ROW_ADDR_DOMAIN_VERSION`]: lance_index::scalar::inverted::INVERTED_ROW_ADDR_DOMAIN_VERSION
pub(super) fn require_fts_address_domain(
    dataset: &Dataset,
    segments: &[IndexMetadata],
) -> Result<()> {
    if !dataset.manifest.uses_stable_row_ids() {
        return Ok(());
    }
    reject_row_id_domain_segments(segments)
}

/// The stable-row-id half of [`require_fts_address_domain`], split out so it
/// can be checked without a dataset.
fn reject_row_id_domain_segments(segments: &[IndexMetadata]) -> Result<()> {
    let Some(legacy) = segments
        .iter()
        .find(|segment| !segment.results_are_row_addrs())
    else {
        return Ok(());
    };
    Err(Error::invalid_input(format!(
        "FTS index segment '{}' ({}, index_version {}) stores row ids, which predates \
         INVERTED_ROW_ADDR_DOMAIN_VERSION, and cannot answer a query on a dataset with stable \
         row IDs: full-text search ranks and pages on the identifier its postings store, and a \
         row id is not the row address the rest of the scan reads by. Rebuild the index with \
         create_index(..., replace: true).",
        legacy.name, legacy.uuid, legacy.index_version,
    )))
}

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
            Some((coverage, index)) => Ok(Some(coverage.allowed_row_mask(index).await?)),
            None => Ok(None),
        }
    };
    let (inverted, allowed_rows) = futures::try_join!(open, allowed_rows)?;
    let result = Arc::new(match allowed_rows {
        Some(allowed_rows) => inverted.with_allowed_rows(allowed_rows),
        None => inverted,
    });
    Ok(result)
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
    // Opening is the chokepoint every indexed FTS path funnels through, so
    // this is where a segment the address-domain execution path cannot answer
    // from fails loudly rather than ranking one numbering system against
    // another.
    require_fts_address_domain(dataset, segments)?;
    // Without stable row ids a row id *is* a row address, so a segment
    // relabelled down for older readers still covers by address; with them the
    // check above has already rejected anything else (see `SegmentRowDomain`).
    let row_domain = match segments.first() {
        Some(first) if first.results_are_row_addrs() => SegmentRowDomain::RowAddress,
        _ => SegmentRowDomain::RowId,
    };
    let coverage = SegmentCoverage::resolve(dataset, &segment_refs, row_domain).await?;
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
/// segment the complete prefilter provably excludes is skipped. A segment the
/// opener restricted decides through its allowed rows; otherwise its resolved
/// query coverage (not its stored provenance) decides by fragment key. When
/// that coverage cannot be resolved, or the inputs do not line up, every
/// segment is kept.
pub(super) async fn segments_the_prefilter_may_match(
    dataset: &Dataset,
    indices: &[Arc<InvertedIndex>],
    segments: &[IndexMetadata],
    pre_filter: &dyn PreFilter,
    metrics: &FtsIndexMetrics,
) -> Vec<Arc<InvertedIndex>> {
    if indices.len() != segments.len() {
        return indices.to_vec();
    }
    let segment_refs: Vec<&IndexMetadata> = segments.iter().collect();
    let coverages =
        match crate::index::frag_reuse::resolve_query_segment_coverages(dataset, &segment_refs)
            .await
        {
            Ok(coverages) => Some(coverages),
            Err(err) => {
                log::debug!("keeping every FTS segment: coverage unresolved ({err})");
                None
            }
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
                // Every segment a query reads is address domain
                // (`require_fts_address_domain`), so the mask it is checked
                // against is read in that domain too.
                /*address_domain=*/
                true,
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
            metrics,
        )
        .await;
        offset += count;
        retained.push(CombinedFieldColumn { indices, ..column });
    }
    retained
}

/// A [`ScoredDoc`] under the order every FTS node ranks and emits on: higher
/// score first, then the lower stored identifier (a row address), then the
/// lower element coordinate. Greater compares better.
///
/// `ScoredDoc`'s own `Ord` compares the score alone, so a tie's order falls to
/// heap and arrival order -- and segments are searched concurrently, so arrival
/// order is not even stable run to run. That matters twice over: the plan above
/// sorts on `(score DESC, _rowaddr ASC)`, and a bounded search's result has to
/// be a *prefix* of that order or `limit`/`offset` paging through a tie skips
/// and repeats rows. Keying the merge on the order directly costs a comparison
/// and gives both, with no tie group held anywhere.
#[derive(Debug, PartialEq, Eq)]
struct RankedDoc(ScoredDoc);

impl Ord for RankedDoc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .score
            .cmp(&other.0.score)
            .then_with(|| other.0.row_id.cmp(&self.0.row_id))
            .then_with(|| other.0.doc_index.cmp(&self.0.doc_index))
    }
}

impl PartialOrd for RankedDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Merge one segment's hits into a bounded best-`limit` heap keyed on
/// [`RankedDoc`]'s order.
fn merge_ranked_documents(
    candidates: &mut std::collections::BinaryHeap<std::cmp::Reverse<RankedDoc>>,
    limit: usize,
    documents: Vec<ScoredDoc>,
) {
    for document in documents {
        let document = RankedDoc(document);
        if candidates.len() < limit {
            candidates.push(std::cmp::Reverse(document));
        } else if candidates.peek().is_some_and(|worst| worst.0 < document) {
            candidates.pop();
            candidates.push(std::cmp::Reverse(document));
        }
    }
}

/// The merged heap in emit order: best first.
fn ranked_documents_in_order(
    candidates: std::collections::BinaryHeap<std::cmp::Reverse<RankedDoc>>,
) -> Vec<ScoredDoc> {
    candidates
        .into_sorted_vec()
        .into_iter()
        .map(|std::cmp::Reverse(RankedDoc(document))| document)
        .collect()
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
        merge_ranked_documents(&mut candidates, limit, documents);
    }

    Ok(ranked_documents_in_order(candidates))
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
        merge_ranked_documents(&mut candidates, limit, documents);
    }

    Ok(ranked_documents_in_order(candidates))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_index::scalar::inverted::INVERTED_ROW_ADDR_DOMAIN_VERSION;

    fn segment(name: &str, index_version: i32) -> IndexMetadata {
        IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            fields: vec![0],
            covering_fields: Vec::new(),
            name: name.to_string(),
            dataset_version: 1,
            fragment_bitmap: None,
            // `results_are_row_addrs` reads the details' type alongside the
            // version, so the stub has to carry an FTS identity.
            index_details: Some(Arc::new(prost_types::Any {
                type_url: "type.googleapis.com/lance.table.InvertedIndexDetails".to_string(),
                value: Vec::new(),
            })),
            index_version,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        }
    }

    const ADDR_VERSION: i32 = INVERTED_ROW_ADDR_DOMAIN_VERSION as i32;
    const LEGACY_VERSION: i32 = ADDR_VERSION - 1;

    #[test]
    fn address_domain_segment_sets_are_accepted() {
        reject_row_id_domain_segments(&[]).unwrap();
        reject_row_id_domain_segments(&[
            segment("title_idx", ADDR_VERSION),
            segment("title_idx", ADDR_VERSION),
        ])
        .unwrap();
    }

    /// A segment persisted before the row-address domain stores row ids, which
    /// the address-domain execution path cannot rank or page on. Under stable
    /// row ids that is refused rather than silently answered from the wrong
    /// numbering, and the error has to name the segment so the caller knows
    /// which index to rebuild.
    #[rstest::rstest]
    #[case::only_segment(vec![("title_idx", LEGACY_VERSION)])]
    #[case::alongside_an_address_segment(vec![
        ("title_idx", ADDR_VERSION),
        ("title_idx", LEGACY_VERSION),
    ])]
    // Order must not decide the outcome.
    #[case::legacy_first(vec![
        ("title_idx", LEGACY_VERSION),
        ("body_idx", ADDR_VERSION),
    ])]
    // A multi-column query pairs two named indexes that nothing checks
    // together at write time, so only the read side can catch this one.
    #[case::across_named_indexes(vec![
        ("title_idx", ADDR_VERSION),
        ("body_idx", LEGACY_VERSION),
    ])]
    fn a_row_id_domain_segment_is_rejected(#[case] versions: Vec<(&str, i32)>) {
        let segments = versions
            .iter()
            .map(|(name, version)| segment(name, *version))
            .collect::<Vec<_>>();
        let legacy = versions
            .iter()
            .find(|(_, version)| *version == LEGACY_VERSION)
            .expect("case must contain a legacy segment")
            .0;
        let error = reject_row_id_domain_segments(&segments)
            .expect_err("a row-id-domain segment should be rejected under stable row ids");
        let message = error.to_string();
        assert!(
            message.contains("stores row ids"),
            "unexpected error: {message}"
        );
        assert!(
            message.contains(legacy),
            "error should name the segment to rebuild: {message}"
        );
    }
}
