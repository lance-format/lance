// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Corpus statistics for `combined_fields`: the flat-scan contribution
//! ([`FlatFieldStats`]) and the cross-column BM25F scorer build.

use std::collections::HashMap;
use std::sync::Arc;

use lance_core::{Error, Result};
use lance_select::RowAddrMask;

use super::super::query::Tokens;
use super::super::scorer::CombinedFieldsBM25Scorer;
use super::{CombinedFieldColumn, unique_terms};
use crate::metrics::MetricsCollector;

/// Per-column corpus contributions of the rows a `combined_fields` flat scan
/// covers, i.e. rows in fragments that at least one target column's index is
/// missing.
///
/// Folded into each column's own totals ahead of the BM25F cross-column blend so
/// that `docCount'`, `docFreq'`, and `avgdl'` describe the whole scanned corpus
/// rather than only its indexed part. Folding after the blend would be wrong:
/// `docCount'`/`docFreq'` are a `max_f`, and `max_f(a_f) + max_f(b_f)` is not
/// `max_f(a_f + b_f)`.
///
/// Every vector is indexed by the position of the column in the
/// [`CombinedFieldColumn`] slice, and `doc_freqs[column]` is indexed by the
/// deduplicated query term position (the order
/// [`combined_fields_search`](super::combined_fields_search) uses).
#[derive(Debug, Default, Clone)]
pub struct FlatFieldStats {
    /// Per column: flat docs with at least one token in that column.
    pub doc_counts: Vec<usize>,
    /// Per column: total tokens over all flat docs.
    pub total_tokens: Vec<u64>,
    /// `[column][term]`: flat docs containing that term in that column.
    pub doc_freqs: Vec<Vec<usize>>,
}

impl FlatFieldStats {
    /// An empty accumulator sized for `num_columns` columns and `num_terms`
    /// deduplicated query terms.
    pub(in crate::scalar::inverted) fn zeros(num_columns: usize, num_terms: usize) -> Self {
        Self {
            doc_counts: vec![0; num_columns],
            total_tokens: vec![0; num_columns],
            doc_freqs: vec![vec![0; num_terms]; num_columns],
        }
    }

    /// Fold one flat row's per-column contribution in. The row's per-column
    /// breakdown is then dropped: these fixed-size aggregates are all the scorer
    /// needs from it.
    ///
    /// `doc_lengths` is `dl_f` per column and `term_counts` is `[column][term]`,
    /// both for a single row. A row absent from a field (`dl_f == 0`) counts
    /// toward neither `docCount_f` nor `docFreq_f`, matching the indexed side
    /// where it owns no document in that field's index.
    ///
    /// `stats_masks[column]` selects the rows that column's index does not
    /// already account for. The flat scan has to read every target column for
    /// every scanned row, or `tf'`/`dl'` would be partial, yet a scanned fragment
    /// may still be indexed for a subset of those columns. Folding such a row
    /// into that column's statistics would double count it against the same
    /// column's index statistics, inflating `docCount_f`, `sumTotalTermFreq_f`,
    /// and `docFreq_f`.
    ///
    /// A mask may still admit an overlay-stale row on purpose, so that the current
    /// value reaches `docFreq_f` at the price of that bounded inflation; the caller
    /// owns that trade.
    pub(in crate::scalar::inverted) fn fold_row(
        &mut self,
        row_id: u64,
        stats_masks: &[Arc<RowAddrMask>],
        doc_lengths: &[u64],
        term_counts: &[u64],
    ) {
        let num_terms = self.doc_freqs.first().map_or(0, |freqs| freqs.len());
        for (column, doc_count) in self.doc_counts.iter_mut().enumerate() {
            if doc_lengths[column] == 0 || !stats_masks[column].selected(row_id) {
                continue;
            }
            *doc_count += 1;
            self.total_tokens[column] += doc_lengths[column];
            let base = column * num_terms;
            for (term, doc_freq) in self.doc_freqs[column].iter_mut().enumerate() {
                if term_counts[base + term] > 0 {
                    *doc_freq += 1;
                }
            }
        }
    }

    /// Add another accumulator's totals. Every field is an integer count, so
    /// merging per-batch folds is exact and order-independent, letting the fold run
    /// inside the per-batch tokenization task rather than over one materialized
    /// corpus.
    pub(in crate::scalar::inverted) fn merge(&mut self, other: Self) {
        for (total, part) in self.doc_counts.iter_mut().zip(other.doc_counts) {
            *total += part;
        }
        for (total, part) in self.total_tokens.iter_mut().zip(other.total_tokens) {
            *total += part;
        }
        for (total, part) in self.doc_freqs.iter_mut().zip(other.doc_freqs) {
            for (total, part) in total.iter_mut().zip(part) {
                *total += part;
            }
        }
    }
}

/// Which statistics [`build_combined_bm25_scorer`] folds into the BM25F blend.
#[derive(Debug, Clone, Copy)]
pub enum CombinedCorpusStats<'a> {
    /// Index statistics alone: every target column's index covers every scanned
    /// fragment, so no row was flat-scanned.
    IndexOnly,
    /// Index statistics plus a flat scan's contribution for the rows no index
    /// covers. The masks [`FlatFieldStats::fold_row`] applied decide which rows those
    /// are, and so whether a row lands on one side or, deliberately, on both.
    Blended(&'a FlatFieldStats),
    /// A flat scan's statistics alone, covering the whole target corpus.
    ///
    /// For a column with overlay-stale index entries: the index still counts the
    /// pre-overlay document and the flat scan cannot subtract it, so neither
    /// folding the row in nor leaving it out describes current data. Measuring the
    /// whole corpus from the scanned values sidesteps that.
    FlatOnly(&'a FlatFieldStats),
}

impl<'a> CombinedCorpusStats<'a> {
    /// The variant for a flat scan's statistics, by whether the scan read every
    /// target fragment or only the rows no index covers.
    pub fn for_flat_scan(flat: &'a FlatFieldStats, covers_whole_corpus: bool) -> Self {
        if covers_whole_corpus {
            Self::FlatOnly(flat)
        } else {
            Self::Blended(flat)
        }
    }

    /// The flat contribution, for the arity checks the caller cannot make.
    fn flat(&self) -> Option<&'a FlatFieldStats> {
        match self {
            Self::IndexOnly => None,
            Self::Blended(flat) | Self::FlatOnly(flat) => Some(flat),
        }
    }

    fn folds_index_stats(&self) -> bool {
        !matches!(self, Self::FlatOnly(_))
    }
}

/// Fold the target columns' segment statistics into a single [`CombinedFieldsBM25Scorer`].
///
/// Generalizes [`build_global_bm25_scorer`](super::super::build_global_bm25_scorer)
/// (which folds segments of one column) to fold across columns too, with
/// per-column weights and the BM25F blend:
/// `docCount'`/`docFreq'` take the max across columns while `sumTotalTermFreq'`
/// is the weighted sum. The query terms are deduplicated the same way the scan
/// deduplicates them, so the resulting per-term `docFreq'` covers exactly the
/// terms [`combined_fields_search`](super::combined_fields_search) scores.
///
/// `corpus` selects which statistics the blend is built from; see
/// [`CombinedCorpusStats`].
///
/// `metrics`, when provided, receives the per-token posting-metadata cache
/// lookups this fold triggers, exactly as
/// [`build_global_bm25_scorer`](super::super::build_global_bm25_scorer)
/// reports them on the single-column path. Without it a cold cross-field query
/// undercounts `index_cache_misses` by one lookup per (term, partition, column).
/// [`CombinedCorpusStats::FlatOnly`] reads none, so it reports none.
pub async fn build_combined_bm25_scorer(
    columns: &[CombinedFieldColumn],
    tokens: &Tokens,
    corpus: CombinedCorpusStats<'_>,
    metrics: Option<&dyn MetricsCollector>,
) -> Result<CombinedFieldsBM25Scorer> {
    let terms = unique_terms(tokens);
    if let Some(flat) = corpus.flat() {
        if flat.doc_counts.len() != columns.len()
            || flat.total_tokens.len() != columns.len()
            || flat.doc_freqs.len() != columns.len()
        {
            return Err(Error::invalid_input(format!(
                "combined_fields flat statistics cover {} / {} / {} columns but the query has {}",
                flat.doc_counts.len(),
                flat.total_tokens.len(),
                flat.doc_freqs.len(),
                columns.len()
            )));
        }
        if let Some(bad) = flat.doc_freqs.iter().find(|df| df.len() != terms.len()) {
            return Err(Error::invalid_input(format!(
                "combined_fields flat document frequencies cover {} terms but the query has {}",
                bad.len(),
                terms.len()
            )));
        }
    }
    let mut doc_count = 0usize;
    let mut sum_total_term_freq = 0f64;
    let mut doc_freq: HashMap<String, usize> = terms.iter().map(|t| (t.clone(), 0)).collect();

    for (column_slot, column) in columns.iter().enumerate() {
        let mut column_num_docs = 0usize;
        let mut column_total_tokens = 0u64;
        let mut column_doc_freq = vec![0usize; terms.len()];
        if corpus.folds_index_stats() {
            for index in &column.indices {
                // Row granularity, not document granularity: `combined_fields_search`
                // blends every posting a row owns into one `tf'` and every document
                // length it owns into one `dl'`, and released V1/V2 indexes may hold
                // one document per list element. See
                // [`InvertedIndex::bm25_row_stats_for_terms`].
                let (total_tokens, num_docs, token_docs) =
                    index.bm25_row_stats_for_terms(&terms, metrics).await?;
                column_total_tokens += total_tokens;
                column_num_docs += num_docs;
                for (slot, df) in token_docs.into_iter().enumerate() {
                    column_doc_freq[slot] += df;
                }
            }
        }
        if let Some(flat) = corpus.flat() {
            column_num_docs += flat.doc_counts[column_slot];
            column_total_tokens += flat.total_tokens[column_slot];
            for (slot, df) in flat.doc_freqs[column_slot].iter().enumerate() {
                column_doc_freq[slot] += df;
            }
        }

        doc_count = doc_count.max(column_num_docs);
        sum_total_term_freq += column.weight as f64 * column_total_tokens as f64;
        for (term, df) in terms.iter().zip(column_doc_freq) {
            let entry = doc_freq
                .get_mut(term)
                .expect("doc_freq initialized for every term");
            *entry = (*entry).max(df);
        }
    }

    let avg_doc_length = if doc_count > 0 {
        (sum_total_term_freq / doc_count as f64) as f32
    } else {
        0.0
    };
    Ok(CombinedFieldsBM25Scorer::new(
        doc_count,
        avg_doc_length,
        doc_freq,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::super::index::InvertedListFormatVersion;
    use super::super::super::tokenizer::InvertedIndexParams;
    use super::super::super::tokenizer::document_tokenizer::DocType;
    use super::super::flat::flat_combined_fields_search_stream;
    use super::super::testing::{
        ElementRows, combined_columns, element_document_index, flat_input, flat_scores,
    };
    use super::*;
    use crate::metrics::LocalMetricsCollector;
    use crate::scalar::inverted::query::Operator;
    use lance_select::RowAddrTreeMap;
    use rstest::rstest;

    /// The statistics fold runs per batch and the partial results are merged, so
    /// it must not matter how the scan is chunked.
    #[test]
    fn test_flat_field_stats_fold_is_split_invariant() {
        // Two columns, two terms. Row 7 is empty in column 1 and row 9 is masked
        // out of column 1, so neither contributes there.
        let rows: [(u64, [u64; 2], [u64; 4]); 3] = [
            (7, [4, 0], [1, 0, 0, 0]),
            (8, [3, 5], [1, 1, 2, 0]),
            (9, [2, 6], [0, 1, 1, 1]),
        ];
        let masks = vec![
            Arc::new(RowAddrMask::all_rows()),
            Arc::new(RowAddrMask::all_rows().also_block(RowAddrTreeMap::from_iter([9u64]))),
        ];
        let fold = |rows: &[(u64, [u64; 2], [u64; 4])]| {
            let mut stats = FlatFieldStats::zeros(2, 2);
            for (row_id, lengths, counts) in rows {
                stats.fold_row(*row_id, &masks, lengths, counts);
            }
            stats
        };

        let whole = fold(&rows);
        assert_eq!(whole.doc_counts, vec![3, 1]);
        assert_eq!(whole.total_tokens, vec![9, 5]);
        // Column 0: "cat" in rows 7 and 8, "dog" in rows 8 and 9.
        // Column 1: only row 8 counts, with "cat" present and "dog" absent.
        assert_eq!(whole.doc_freqs, vec![vec![2, 2], vec![1, 0]]);

        for split in 0..=rows.len() {
            let mut merged = fold(&rows[..split]);
            merged.merge(fold(&rows[split..]));
            assert_eq!(merged.doc_counts, whole.doc_counts, "split at {split}");
            assert_eq!(merged.total_tokens, whole.total_tokens, "split at {split}");
            assert_eq!(merged.doc_freqs, whole.doc_freqs, "split at {split}");
        }
    }

    /// Both scorer builds must report their index cache lookups to the query's
    /// collector, the way the single-column `build_global_bm25_scorer` does; see
    /// [`build_combined_bm25_scorer`] for what goes wrong otherwise.
    ///
    /// The two format versions cover both statistics arms:
    /// `bm25_row_stats_for_terms` delegates to the document-granularity path on
    /// V3, and takes `InvertedPartition::row_stats_for_terms` on V1/V2, where this
    /// element-per-document fixture makes it read the posting lists themselves.
    #[rstest]
    #[case::modern(InvertedListFormatVersion::V3)]
    #[case::legacy_elements(InvertedListFormatVersion::V2)]
    #[tokio::test]
    async fn test_combined_scorer_build_reports_index_cache_lookups(
        #[case] format_version: InvertedListFormatVersion,
    ) {
        let vocab = ["alpha", "beta"];
        // Row 0 owns two elements, so the legacy fixture is element-per-document
        // and its row-granularity statistics take the deduplicating branch.
        let rows: ElementRows = vec![vec![vec!["alpha"], vec!["beta"]], vec![vec!["alpha"]]];
        let (first, _first_dir) = element_document_index(format_version, &vocab, &rows).await;
        let (second, _second_dir) = element_document_index(format_version, &vocab, &rows).await;
        let columns = combined_columns(vec![first, second]);
        let tokens = Tokens::new(vec!["alpha".to_owned(), "beta".to_owned()], DocType::Text);
        // One lookup per (term, column, partition), over 2 terms and the single
        // partition each fixture index holds.
        let expected_lookups = 2 * columns.len();

        let indexed = LocalMetricsCollector::default();
        build_combined_bm25_scorer(
            &columns,
            &tokens,
            CombinedCorpusStats::IndexOnly,
            Some(&indexed),
        )
        .await
        .unwrap();
        assert_eq!(
            indexed.index_cache_hits() + indexed.index_cache_misses(),
            expected_lookups,
            "indexed scorer build must report every cache lookup it makes",
        );

        // The flat sibling folds the same per-column index statistics in, so its
        // build reports the same lookups. An empty input isolates them: the
        // scanned rows contribute no index reads at all.
        let flat = LocalMetricsCollector::default();
        let stream = flat_combined_fields_search_stream(
            flat_input(&[], &[Vec::new(), Vec::new()], 1),
            &columns,
            vec![1, 2],
            &[
                Arc::new(RowAddrMask::all_rows()),
                Arc::new(RowAddrMask::all_rows()),
            ],
            /*emit_mask=*/ None,
            /*flat_covers_whole_corpus=*/ false,
            &tokens,
            InvertedIndexParams::default().build().unwrap(),
            Operator::Or,
            None,
            16,
            None,
            Some(&flat),
        )
        .await
        .unwrap()
        .0;
        assert!(flat_scores(stream).await.1.is_empty());
        assert_eq!(
            flat.index_cache_hits() + flat.index_cache_misses(),
            expected_lookups,
            "flat scorer build must report every cache lookup it makes",
        );
    }

    /// A legacy index whose documents already map one-to-one onto rows (a plain
    /// `Utf8` column, or a list column with a single element per row) must be
    /// completely unaffected: the row-granularity statistics equal the
    /// document-granularity ones, so no score can move.
    #[rstest]
    #[case::v1(InvertedListFormatVersion::V1)]
    #[case::v2(InvertedListFormatVersion::V2)]
    #[case::v3(InvertedListFormatVersion::V3)]
    #[tokio::test]
    async fn test_row_stats_match_document_stats_without_list_multiplicity(
        #[case] format_version: InvertedListFormatVersion,
    ) {
        let vocab = ["alpha", "beta"];
        // Row 0: a multi-token `Utf8` document. Row 1: a single-element list.
        // Row 2: an empty list, so the row owns no document at all.
        let rows: ElementRows = vec![
            vec![vec!["alpha", "beta", "alpha"]],
            vec![vec!["beta"]],
            vec![Vec::new()],
            vec![vec!["alpha"]],
        ];
        let (index, _dir) = element_document_index(format_version, &vocab, &rows).await;

        let docs = index.partitions[0].docs.address_keyed().await.unwrap();
        assert_eq!(docs.len(), 3, "the empty list must not become a document");
        assert_eq!(docs.num_distinct_rows(), docs.len());
        let terms = ["alpha".to_owned(), "beta".to_owned(), "absent".to_owned()];
        let documents = index.bm25_stats_for_terms(&terms, None).await.unwrap();
        assert_eq!(documents, (5, 3, vec![2, 2, 0]));
        assert_eq!(
            index.bm25_row_stats_for_terms(&terms, None).await.unwrap(),
            documents
        );
    }

    /// Rows a legacy list index never indexed (a null or empty list, a list of
    /// empty strings) own no document, so they must not count toward
    /// `docCount_f` or `docFreq_f`. This is the same rule the indexed and flat
    /// sides already follow (`AddressKeyedDocuments::doc_length_at` reports 0 for them and
    /// `FlatFieldStats::fold_row` skips a column with `dl_f == 0`), applied to
    /// distinct-row counting.
    #[rstest]
    #[case::v1(InvertedListFormatVersion::V1)]
    #[case::v2(InvertedListFormatVersion::V2)]
    #[tokio::test]
    async fn test_row_stats_skip_rows_a_legacy_list_index_never_indexed(
        #[case] format_version: InvertedListFormatVersion,
    ) {
        let vocab = ["alpha", "beta"];
        let rows: ElementRows = vec![
            // Several elements, one of them empty.
            vec![vec!["alpha"], Vec::new(), vec!["alpha"], vec!["beta"]],
            // A null or empty list.
            Vec::new(),
            vec![vec!["beta"]],
            // A list of nothing but empty strings.
            vec![Vec::new(), Vec::new()],
        ];
        let (index, _dir) = element_document_index(format_version, &vocab, &rows).await;

        let docs = index.partitions[0].docs.address_keyed().await.unwrap();
        assert_eq!(docs.len(), 4, "only rows 0 and 2 own documents");
        assert_eq!(docs.num_distinct_rows(), 2);
        for unindexed in [1u64, 3] {
            assert_eq!(docs.doc_length_at(unindexed), 0);
        }

        let terms = ["alpha".to_owned(), "beta".to_owned()];
        assert_eq!(
            index.bm25_stats_for_terms(&terms, None).await.unwrap(),
            (4, 4, vec![2, 2]),
        );
        assert_eq!(
            index.bm25_row_stats_for_terms(&terms, None).await.unwrap(),
            (4, 2, vec![1, 2]),
        );
    }
}
