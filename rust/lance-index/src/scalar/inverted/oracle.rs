// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Brute-force BM25F reference for the `combined_fields` tests and bench.
//!
//! Re-derives every statistic from the raw text on each call, so it shares no code with
//! the scan it checks. `cfg(test)` here and the `test-oracle` feature downstream keep it
//! out of normal builds.

use std::collections::HashSet;

const K1: f32 = 1.2;
const B: f32 = 0.75;

/// Splits on whitespace only, while the `simple` tokenizer splits on every
/// non-alphanumeric character. Keep test corpora alphanumeric or scores diverge.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| word.to_lowercase())
        .collect()
}

/// Exact BM25F: `docFreq'`/`docCount'` are the max across fields, `tf'`/`dl'`/
/// `sumTotalTermFreq'` weighted sums, and the `(k1 + 1)` numerator is Lance's. `None`
/// for a document that does not match. `columns` pairs each field's weight with its
/// text, one entry per document.
///
/// A `""` value stands for anything that tokenizes to nothing (NULL, empty string,
/// empty list): absent from that column's statistics, so it must not raise
/// `docCount_f`/`docFreq_f`. List columns are their elements joined by a space.
#[cfg_attr(coverage, coverage(off))]
pub fn brute_force_bm25f(
    columns: &[(f32, Vec<&str>)],
    query: &str,
    require_all_terms: bool,
) -> Vec<Option<f32>> {
    let mut seen = HashSet::new();
    let terms: Vec<String> = tokenize(query)
        .into_iter()
        .filter(|term| seen.insert(term.clone()))
        .collect();
    let num_docs = columns[0].1.len();
    let tokenized: Vec<Vec<Vec<String>>> = columns
        .iter()
        .map(|(_, texts)| texts.iter().map(|text| tokenize(text)).collect())
        .collect();

    // Zero-token documents are absent from a column's `DocSet`, and `flat_field_stats`
    // skips a column whose `dl_f == 0`, so they cannot raise `docCount_f` either.
    let doc_count = tokenized
        .iter()
        .map(|column| column.iter().filter(|doc| !doc.is_empty()).count())
        .max()
        .unwrap_or(0);
    let mut sum_total_term_freq = 0f64;
    let mut doc_freq = vec![0usize; terms.len()];
    for (column_index, (weight, _)) in columns.iter().enumerate() {
        let total_tokens: usize = tokenized[column_index].iter().map(|doc| doc.len()).sum();
        sum_total_term_freq += *weight as f64 * total_tokens as f64;
        for (term_index, term) in terms.iter().enumerate() {
            let df = tokenized[column_index]
                .iter()
                .filter(|doc| doc.contains(term))
                .count();
            doc_freq[term_index] = doc_freq[term_index].max(df);
        }
    }
    let avgdl = if doc_count == 0 {
        0.0
    } else {
        (sum_total_term_freq / doc_count as f64) as f32
    };
    let idf: Vec<f32> = doc_freq
        .iter()
        .map(|&df| {
            if df == 0 {
                0.0
            } else {
                ((doc_count as f32 - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln()
            }
        })
        .collect();

    (0..num_docs)
        .map(|doc| {
            let mut tf = vec![0f32; terms.len()];
            let mut dl = 0f32;
            for (column_index, (weight, _)) in columns.iter().enumerate() {
                dl += weight * tokenized[column_index][doc].len() as f32;
                for (term_index, term) in terms.iter().enumerate() {
                    let count = tokenized[column_index][doc]
                        .iter()
                        .filter(|token| *token == term)
                        .count();
                    tf[term_index] += weight * count as f32;
                }
            }
            let matched = if require_all_terms {
                tf.iter().all(|&freq| freq > 0.0)
            } else {
                tf.iter().any(|&freq| freq > 0.0)
            };
            if !matched {
                return None;
            }
            let mut score = 0.0;
            for term_index in 0..terms.len() {
                if tf[term_index] <= 0.0 {
                    continue;
                }
                let doc_norm = K1 * (1.0 - B + B * dl / avgdl);
                score +=
                    idf[term_index] * (K1 + 1.0) * tf[term_index] / (tf[term_index] + doc_norm);
            }
            Some(score)
        })
        .collect()
}

/// The ids [`brute_force_bm25f`] matched.
#[cfg_attr(coverage, coverage(off))]
pub fn brute_force_ids(scores: &[Option<f32>]) -> HashSet<i32> {
    (0..scores.len())
        .filter(|&doc| scores[doc].is_some())
        .map(|doc| doc as i32)
        .collect()
}

/// The `k` highest-scoring ids, descending, id as tiebreak.
#[cfg_attr(coverage, coverage(off))]
pub fn brute_force_top_k(scores: &[Option<f32>], k: usize) -> Vec<i32> {
    let mut ranked: Vec<(i32, f32)> = scores
        .iter()
        .enumerate()
        .filter_map(|(doc, score)| score.map(|score| (doc as i32, score)))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    ranked.into_iter().take(k).map(|(doc, _)| doc).collect()
}
