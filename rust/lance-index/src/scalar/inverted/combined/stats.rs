// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Corpus statistics for `combined_fields`: the cross-column BM25F scorer
//! build.

use std::collections::HashMap;

use lance_core::Result;

use super::super::query::Tokens;
use super::super::scorer::CombinedFieldsBM25Scorer;
use super::{CombinedFieldColumn, unique_terms};
use crate::metrics::MetricsCollector;

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
/// `metrics`, when provided, receives the per-token posting-metadata cache
/// lookups this fold triggers, exactly as
/// [`build_global_bm25_scorer`](super::super::build_global_bm25_scorer)
/// reports them on the single-column path. Without it a cold cross-field query
/// undercounts `index_cache_misses` by one lookup per (term, partition, column).
pub async fn build_combined_bm25_scorer(
    columns: &[CombinedFieldColumn],
    tokens: &Tokens,
    metrics: Option<&dyn MetricsCollector>,
) -> Result<CombinedFieldsBM25Scorer> {
    let terms = unique_terms(tokens);
    let mut doc_count = 0usize;
    let mut sum_total_term_freq = 0f64;
    let mut doc_freq: HashMap<String, usize> = terms.iter().map(|t| (t.clone(), 0)).collect();

    for column in columns {
        let mut column_num_docs = 0usize;
        let mut column_total_tokens = 0u64;
        let mut column_doc_freq = vec![0usize; terms.len()];
        {
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
    use super::super::super::tokenizer::document_tokenizer::DocType;
    use super::super::testing::{ElementRows, combined_columns, element_document_index};
    use super::*;
    use crate::metrics::LocalMetricsCollector;
    use rstest::rstest;

    /// The scorer build must report its index cache lookups to the query's
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
        build_combined_bm25_scorer(&columns, &tokens, Some(&indexed))
            .await
            .unwrap();
        assert_eq!(
            indexed.index_cache_hits() + indexed.index_cache_misses(),
            expected_lookups,
            "the scorer build must report every cache lookup it makes",
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
