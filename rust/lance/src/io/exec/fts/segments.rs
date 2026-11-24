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
use crate::dataset::rowids::live_row_addr_to_row_id_lookup;
use crate::index::loading::DatasetIndexInternalExt;
use crate::index::prefilter::prefilter_may_match_segment;
use crate::index::segment_coverage::{SegmentCoverage, SegmentRowDomain};

/// The translation from a stored physical row address to its row id.
pub(super) type RowAddrToRowId = dyn Fn(u64) -> Option<u64> + Send + Sync;

/// Whether `segments`' own stored identifiers are physical row addresses, for
/// code that must pick between an address-domain and a row-id-domain
/// representation of the same restriction (a `PreFilter` mask, a prefilter's
/// per-segment coverage check -- see `fts_row_addr_to_row_id`'s doc for why
/// FTS cannot just translate its result afterward the way `ScalarIndexExec`
/// does).
///
/// True whenever the two domains coincide regardless of what a segment's own
/// `results_are_row_addrs()` reports: without stable row ids a row id and a
/// row address are the same value, and `build_scalar_index` relabels a fresh
/// build's segment to a pre-`INVERTED_ROW_ADDR_DOMAIN_VERSION` version in
/// that case specifically so an old reader still understands it -- not
/// because its stored values differ in kind. Mixed labels are therefore both
/// expected and harmless on such a dataset, and are not checked.
///
/// With stable row ids the two domains genuinely differ, and every segment
/// reaching one query has to agree on which one it stores. A caller passes
/// every segment the query will read, which for a multi-column query
/// (`combined_fields`, a cross-column compound) spans *several named
/// indexes*: their postings are merged on the stored identifier alone, so
/// one column's row id `2` and another's row address `1 << 32` for the same
/// row would be scored as two unrelated documents. Within a single named
/// index that mix is already refused at write time -- `optimize_indices`'
/// `legacy_domain_mismatch` rebuilds rather than merging across domains, and
/// `manifest_build` rejects a commit that would publish one -- but nothing
/// couples two columns' indexes, and nothing could: rebuilding one column's
/// index is a legitimate commit on its own, and only a query spanning both
/// makes the pair wrong. Disagreement is therefore rejected here, at the one
/// point that sees every index a query reads.
pub(super) fn fts_segments_are_address_domain(
    dataset: &Dataset,
    segments: &[IndexMetadata],
) -> Result<bool> {
    if !dataset.manifest.uses_stable_row_ids() {
        return Ok(true);
    }
    shared_segment_domain(segments)
}

/// The one domain every segment in `segments` agrees on, or an error naming
/// the two that disagree. `true` for an empty set, which has nothing to
/// contradict.
///
/// Within one named index a mixed set is already refused when it is committed
/// (`manifest_build`'s per-name check), so reaching this with one is a sign of
/// a manifest written by something other than that path. Across named indexes
/// -- the columns of one multi-column query -- nothing refuses it at write
/// time and nothing could: rebuilding one column's index is a legitimate
/// commit on its own, and only a query spanning both makes the pair wrong.
fn shared_segment_domain(segments: &[IndexMetadata]) -> Result<bool> {
    let Some(first) = segments.first() else {
        return Ok(true);
    };
    let first_is_row_addrs = first.results_are_row_addrs();
    if let Some(other) = segments
        .iter()
        .find(|segment| segment.results_are_row_addrs() != first_is_row_addrs)
    {
        let describe = |segment: &IndexMetadata| {
            format!(
                "'{}' ({}, index_version {}) stores row {}",
                segment.name,
                segment.uuid,
                segment.index_version,
                if segment.results_are_row_addrs() {
                    "addresses"
                } else {
                    "ids"
                },
            )
        };
        return Err(Error::invalid_input(format!(
            "FTS index segments disagree on the identity domain they store: segment {} \
             while segment {}. A query merges their postings on that identifier alone and \
             cannot combine the two domains. Rebuild the older index with \
             create_index(..., replace: true).",
            describe(first),
            describe(other),
        )));
    }
    Ok(first_is_row_addrs)
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
    // A segment persisted before `INVERTED_ROW_ADDR_DOMAIN_VERSION` stores row
    // ids, which can remain live in a sibling segment after a stable-row-id
    // row moves; a segment built since stores row addresses, which do not
    // (see `SegmentRowDomain`). Opening is the chokepoint every indexed FTS
    // path funnels through, so this is where a committed index whose segments
    // disagree fails loudly rather than answering from two numbering systems
    // at once.
    fts_segments_are_address_domain(dataset, segments)?;
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

/// Build the row-address-to-row-id translation this query's final output
/// needs, for a query whose segments resolved to `address_domain` via
/// [`fts_segments_are_address_domain`]. Taking the resolved domain rather
/// than re-deriving one from a representative index keeps the translation of
/// the results and the domain the postings and masks were read in from
/// drifting apart -- on a multi-column query they are not the same index.
///
/// `None` (a no-op) when the two coincide: no stable row ids, or a
/// row-id-domain segment. The latter is a transient state right after a
/// dataset turns stable row ids on, before its existing segments are next
/// rebuilt or merged -- that does not happen eagerly, so a query can still
/// land here with a row-id-domain segment under stable row ids, and the
/// segment's own values are already row ids needing no translation.
///
/// Unlike MinHash's per-segment lookup (`live_row_addr_to_row_id_lookup`
/// scoped to one segment's own fragments), FTS merges per-segment top-k hits
/// into one global ranking before this can run, so it must already cover
/// every fragment any segment of the index could answer from; this uses the
/// dataset's whole current fragment set rather than tracking each query's
/// narrower per-segment union.
pub(super) async fn fts_row_addr_to_row_id(
    dataset: &Dataset,
    address_domain: bool,
) -> Result<Option<Arc<RowAddrToRowId>>> {
    if !dataset.manifest.uses_stable_row_ids() || !address_domain {
        return Ok(None);
    }
    let fragment_ids = dataset
        .manifest
        .fragments
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect::<Vec<_>>();
    let lookup = live_row_addr_to_row_id_lookup(dataset, fragment_ids).await?;
    Ok(Some(Arc::new(lookup)))
}

/// Extra candidates to retain past a query's own `limit` so that a tie
/// straddling the top-k cutoff survives long enough to be re-ranked on row id.
///
/// Mirrors `WAND_TIE_COMPLETION_BUDGET`: it bounds how large a tie group can
/// be and still be resolved exactly, rather than promising to resolve one of
/// any size.
pub(super) const FTS_TIE_RERANK_BUDGET: usize = 128;

/// The limit a search should actually run with, given the one the query asked
/// for. Widened only when results will be translated afterward, so a query
/// that needs no translation pays nothing.
pub(super) fn widen_limit_for_tie_rerank(limit: Option<usize>, translating: bool) -> Option<usize> {
    match limit {
        Some(limit) if translating => Some(limit.saturating_add(FTS_TIE_RERANK_BUDGET)),
        other => other,
    }
}

/// Whether a bounded search may have cut a tie group that straddles the top-k
/// cutoff, so its result cannot be re-ranked into the right order.
///
/// True only when the search actually filled the widened cap *and* the score
/// at the requested cutoff still equals the last score retained: then rows
/// beyond the cap may tie the cutoff too, and which of them belong in the
/// top-k cannot be decided from what came back. The caller completes the tie
/// by re-running the search exactly, the same escape hatch the WAND tie
/// certificate takes when its own completion budget is exhausted --
/// [`FTS_TIE_RERANK_BUDGET`] alone silently truncates a larger tie, and equal
/// scores across many short documents are an ordinary FTS case, not a corner
/// one.
pub(super) fn cutoff_tie_unresolved(
    scores: &[f32],
    requested_limit: Option<usize>,
    searched_limit: Option<usize>,
) -> bool {
    let (Some(requested), Some(searched)) = (requested_limit, searched_limit) else {
        return false;
    };
    if requested == 0 || scores.len() < searched || scores.len() <= requested {
        // Nothing was cut: the search ran out of matches before its cap, or
        // everything it found survives the requested limit anyway.
        return false;
    }
    scores
        .get(requested - 1)
        .zip(scores.last())
        .is_some_and(|(cutoff, last)| cutoff == last)
}

/// Translate a final `(row_ids, scores)` pair out of the index's stored domain
/// and put it back into the order a caller sees, dropping unresolvable rows
/// and keeping the two vectors aligned.
///
/// Selection ranks by `(score DESC, stored identifier ASC)` and cannot do
/// otherwise: `combined_maxscore` discovers candidates in ascending stored-id
/// order (`.min()` over the cursor heads), and its pruning is only sound
/// because a candidate that ties the threshold arrives after -- and so loses
/// to -- every incumbent. Once that stored identifier is a row address, the
/// order selection imposes among tied scores is physical, while the plan above
/// sorts by `(score DESC, _rowid ASC)`. Paging through a tie would then skip
/// and repeat rows, because each page's selection keeps the lowest *address*
/// while the sort reorders by *row id*.
///
/// Re-ranking the survivors here settles it: with [`FTS_TIE_RERANK_BUDGET`]
/// extra candidates retained (see [`widen_limit_for_tie_rerank`]), a tie group
/// at the cutoff is held whole, translated, and ordered by row id before
/// `limit` is applied -- so the rows that survive are the ones the plan's own
/// ordering would have picked. Translating earlier, inside selection, is not
/// an option: it would key the heap on row ids while candidates still arrive
/// in address order, breaking the invariant the pruning rests on.
pub(super) fn finalize_row_ids_and_scores(
    row_ids: Vec<u64>,
    scores: Vec<f32>,
    row_addr_to_row_id: Option<&RowAddrToRowId>,
    limit: Option<usize>,
) -> (Vec<u64>, Vec<f32>) {
    if row_addr_to_row_id.is_none() {
        // Stored identifiers are already row ids, so selection ordered by the
        // same key the plan sorts on and the limit was never widened.
        return (row_ids, scores);
    }
    let (row_ids, scores) = translate_row_ids_and_scores(row_ids, scores, row_addr_to_row_id);
    let mut ranked = row_ids.into_iter().zip(scores).collect::<Vec<_>>();
    ranked.sort_by(|(left_row, left_score), (right_row, right_score)| {
        right_score
            .partial_cmp(left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left_row.cmp(right_row))
    });
    if let Some(limit) = limit {
        ranked.truncate(limit);
    }
    ranked.into_iter().unzip()
}

/// Translate and drop unresolvable rows from a final `(row_ids, scores)`
/// result pair, keeping the two aligned. A `None` translator is a no-op.
pub(super) fn translate_row_ids_and_scores(
    row_ids: Vec<u64>,
    scores: Vec<f32>,
    row_addr_to_row_id: Option<&RowAddrToRowId>,
) -> (Vec<u64>, Vec<f32>) {
    let Some(translate) = row_addr_to_row_id else {
        return (row_ids, scores);
    };
    row_ids
        .into_iter()
        .zip(scores)
        .filter_map(|(row_id, score)| translate(row_id).map(|row_id| (row_id, score)))
        .unzip()
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
    fn empty_and_uniform_segment_sets_resolve_to_one_domain() {
        assert!(shared_segment_domain(&[]).unwrap());
        assert!(
            shared_segment_domain(&[
                segment("title_idx", ADDR_VERSION),
                segment("title_idx", ADDR_VERSION),
            ])
            .unwrap()
        );
        assert!(
            !shared_segment_domain(&[
                segment("title_idx", LEGACY_VERSION),
                segment("title_idx", LEGACY_VERSION),
            ])
            .unwrap()
        );
    }

    /// A manifest written outside `manifest_build`'s commit-time check can
    /// still pair the two domains, and a multi-column query pairs two named
    /// indexes that nothing checks together at all. Either way the read side
    /// has to refuse rather than let one segment's domain speak for the rest.
    #[test]
    fn disagreeing_segments_are_rejected_and_both_are_named() {
        for (left, right) in [("title_idx", "body_idx"), ("title_idx", "title_idx")] {
            let error = shared_segment_domain(&[
                segment(left, ADDR_VERSION),
                segment(right, LEGACY_VERSION),
            ])
            .expect_err("disagreeing segments should be rejected");
            let message = error.to_string();
            assert!(
                message.contains("disagree on the identity domain"),
                "unexpected error: {message}"
            );
            assert!(
                message.contains("stores row addresses") && message.contains("stores row ids"),
                "error should say what each segment stores: {message}"
            );
            assert!(
                message.contains(left) && message.contains(right),
                "error should name both segments: {message}"
            );
        }
    }

    /// Order must not decide the outcome: the disagreement is symmetric.
    #[test]
    fn a_legacy_segment_first_is_rejected_too() {
        assert!(
            shared_segment_domain(&[
                segment("title_idx", LEGACY_VERSION),
                segment("body_idx", ADDR_VERSION),
            ])
            .is_err()
        );
    }
}
