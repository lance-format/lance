// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The indexed `combined_fields` entry point: load every term's postings across
//! the target columns, merge them into the shared row-id space, and score every
//! candidate.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::sync::Arc;

use lance_core::Result;
use lance_core::utils::tokio::spawn_cpu;

use super::super::builder::ScoredDoc;
use super::super::documents::AddressKeyedDocuments;
use super::super::query::{FtsSearchParams, Operator, Tokens};
use super::super::scorer::CombinedFieldsBM25Scorer;
use super::cursor::{CombinedTermPostings, LoadedSource, build_term_postings};
use super::{CombinedFieldColumn, unique_terms};
use crate::metrics::MetricsCollector;
use crate::prefilter::PreFilter;

/// Exact cross-field BM25F search over the target columns.
///
/// Loads each query term's postings across every column and partition, then runs
/// the term-at-a-time `combined_maxscore`, which prunes candidate scoring without
/// changing the top-k it returns.
///
/// `operator` applies across the virtual field: `And` keeps only docs where
/// every query term appears in at least one column; `Or` keeps docs matching
/// any term. Per-column `boost` is folded into `tf'`.
pub async fn combined_fields_search(
    columns: &[CombinedFieldColumn],
    tokens: &Tokens,
    params: &FtsSearchParams,
    operator: Operator,
    scorer: &CombinedFieldsBM25Scorer,
    prefilter: Arc<dyn PreFilter>,
    metrics: &dyn MetricsCollector,
) -> Result<(Vec<u64>, Vec<f32>)> {
    let terms = unique_terms(tokens);
    let limit = params.limit.unwrap_or(usize::MAX);
    if terms.is_empty() || limit == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mask = prefilter.mask();
    let require_all_terms = operator == Operator::And;

    // Load every term's postings across all columns, in the canonical
    // column → index → partition order. The length sources are collected in that
    // same order so `dl'` sums in the exact scan's order too (float addition is
    // order-sensitive; matching the order keeps every score bit-identical).
    let mut loaded: Vec<Vec<LoadedSource>> = (0..terms.len()).map(|_| Vec::new()).collect();
    let mut length_sources: Vec<(f32, AddressKeyedDocuments)> = Vec::new();
    for column in columns {
        let weight = column.weight;
        for index in &column.indices {
            for partition in &index.partitions {
                let docs = partition.docs.address_keyed().await?;
                let is_legacy = partition.is_legacy();
                for (term_index, term) in terms.iter().enumerate() {
                    let Some(token_id) = partition.tokens.get(term) else {
                        continue;
                    };
                    let posting = partition
                        .inverted_list
                        .posting_list(token_id, false, metrics)
                        .await?;
                    loaded[term_index].push(LoadedSource {
                        weight,
                        docs: docs.clone(),
                        is_legacy,
                        posting,
                    });
                }
                length_sources.push((weight, docs));
            }
        }
    }
    // Everything past the loads is uninterruptible CPU work: building and sorting
    // a per-term `HashMap`, then the whole scoring loop with no await. Offload it
    // so a large query cannot hold a DataFusion
    // worker past a stream drop or task cancellation, matching how the
    // single-column `InvertedIndex::bm25_search` dispatches its per-partition
    // scoring and how `flat_combined_fields_search_stream` dispatches its own
    // scoring loop. The `'static` closure clones the borrowed `scorer` (a handful
    // of per-term statistics) and moves everything else in.
    let scorer = Arc::new(scorer.clone());
    let top = spawn_cpu(move || {
        let dl_prime = |row_id: u64| -> f32 {
            length_sources
                .iter()
                .map(|(weight, docs)| weight * docs.doc_length_at(row_id) as f32)
                .sum()
        };
        let terms: Vec<CombinedTermPostings> = terms
            .iter()
            .zip(loaded)
            .map(|(term, sources)| build_term_postings(term, sources, &mask, scorer.as_ref()))
            .collect();

        // Score every candidate: the union of the terms' postings for `Or`, and the
        // same union filtered to the documents holding every term for `And`. A row
        // absent from a term contributes no `tf'`, so it is skipped rather than
        // scored as zero.
        //
        // Candidates are visited in ascending row-id order, which makes the top-k
        // deterministic: `ScoredDoc` breaks a score tie on the row id, so equal
        // scores come back in a stable order run to run.
        let candidates: BTreeSet<u64> = terms
            .iter()
            .flat_map(|term| term.postings.iter().map(|(row_id, _)| *row_id))
            .collect();
        let mut top: BinaryHeap<Reverse<ScoredDoc>> = BinaryHeap::new();
        for row_id in candidates {
            let dl = dl_prime(row_id);
            let mut score = 0.0f32;
            let mut missing_term = false;
            for term in &terms {
                let tf = term.tf_prime(row_id);
                if tf <= 0.0 {
                    missing_term = true;
                    continue;
                }
                score += term.idf * scorer.doc_weight(tf, dl);
            }
            if require_all_terms && missing_term {
                continue;
            }
            top.push(Reverse(ScoredDoc::new(row_id, score)));
            if top.len() > limit {
                top.pop();
            }
        }
        Result::Ok(top)
    })
    .await?;

    Ok(top
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse(doc)| (doc.row_id, doc.score.0))
        .unzip())
}
