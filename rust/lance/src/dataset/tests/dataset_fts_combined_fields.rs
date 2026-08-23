// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::vec;

use super::dataset_index::nested_fts_batch;

use crate::Dataset;

use crate::dataset::write::{WriteMode, WriteParams};
use crate::index::DatasetIndexExt;
use crate::utils::test::copy_test_data_to_tmp;
use arrow::array::{AsArray, GenericListBuilder, GenericStringBuilder};
use arrow::datatypes::UInt64Type;
use arrow_array::RecordBatch;
use arrow_array::record_batch;
use arrow_array::{Array, GenericStringArray, ListArray, StructArray};
use arrow_array::{
    ArrayRef, Int32Array, RecordBatchIterator, StringArray,
    types::{Float32Type, Int32Type, Int64Type},
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema};
use lance_core::utils::tempfile::TempStrDir;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::{
    Language,
    query::{MatchQuery, Operator},
    tokenizer::InvertedIndexParams,
};
use lance_index::{IndexType, scalar::ScalarIndexParams};

use lance_index::scalar::inverted::builder::BLOCK_SIZE;
use lance_index::scalar::inverted::oracle::{brute_force_bm25f, brute_force_ids};
use lance_index::scalar::inverted::query::{CombinedFieldsQuery, FtsQuery, MultiMatchQuery};
use rstest::rstest;

/// Whitespace-tokenizing index params, so `brute_force_bm25f` can mirror them.
fn combined_fields_test_params() -> InvertedIndexParams {
    InvertedIndexParams::new("simple".to_string(), Language::English)
        .lower_case(true)
        .stem(false)
        .remove_stop_words(false)
        .ascii_folding(false)
        .max_token_length(None)
}

/// The standard `id`/`title`/`body` batch these tests score over.
fn combined_fields_batch(ids: Vec<i32>, titles: Vec<&str>, bodies: Vec<&str>) -> RecordBatch {
    record_batch!(
        ("id", Int32, ids),
        ("title", Utf8, titles),
        ("body", Utf8, bodies)
    )
    .unwrap()
}

/// Write one batch to `uri`, creating the dataset or appending to it.
///
/// `write_params` of `None` takes the write defaults: one fragment, row addresses
/// rather than stable row ids. Pass a [`WriteParams`] to pick a fragment layout
/// (`max_rows_per_file`), append instead of create (`mode`), or switch row-id
/// scheme (`enable_stable_row_ids`).
async fn write_fts_dataset(
    uri: &str,
    batch: RecordBatch,
    write_params: Option<WriteParams>,
) -> Dataset {
    let schema = batch.schema();
    Dataset::write(
        RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema),
        uri,
        write_params,
    )
    .await
    .unwrap()
}

/// Build one inverted index per column in `columns`, at the dataset's current
/// version. Call sites list the columns explicitly because which ones carry an
/// index, and at which version, is what the coverage-skew tests vary.
async fn create_inverted_indices(
    dataset: &mut Dataset,
    columns: &[&str],
    params: &InvertedIndexParams,
) {
    for &column in columns {
        dataset
            .create_index(&[column], IndexType::Inverted, None, params, true)
            .await
            .unwrap();
    }
}

fn combined_query(terms: &str, operator: Operator) -> FtsQuery {
    combined_query_with_boosts(terms, operator, None)
}

/// `boosts` of `None` leaves every weight at the default `1.0`. Explicit weights
/// exercise the `w_f` factors in `tf'`/`dl'`, which are invisible at 1.0: an
/// implementation that dropped `weight` still scores correctly with unit weights.
fn combined_query_with_boosts(
    terms: &str,
    operator: Operator,
    boosts: Option<Vec<f32>>,
) -> FtsQuery {
    combined_query_over(&["title", "body"], terms, operator, boosts)
}

/// Like [`combined_query_with_boosts`] but over an explicit column list, for the
/// datasets that are not the two-column `title`/`body` shape.
fn combined_query_over(
    columns: &[&str],
    terms: &str,
    operator: Operator,
    boosts: Option<Vec<f32>>,
) -> FtsQuery {
    let query = CombinedFieldsQuery::try_new(
        terms.to_string(),
        columns.iter().map(|c| c.to_string()).collect(),
    )
    .unwrap();
    let query = match boosts {
        Some(boosts) => query.try_with_boosts(boosts).unwrap(),
        None => query,
    };
    FtsQuery::CombinedFields(query.with_operator(operator))
}

/// Run an unlimited full-text query and return the matched `id`s in result order.
async fn fts_result_ids(dataset: &Dataset, query: FtsQuery) -> Vec<i32> {
    fts_result_id_scores(dataset, query, None)
        .await
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

/// Run a full-text query and return `(id, score)` pairs in result order.
///
/// `limit` is the query's top-k: `Some(k)` arms the pruning path, `None` takes the
/// exact scan over every match.
async fn fts_result_id_scores(
    dataset: &Dataset,
    query: FtsQuery,
    limit: Option<i64>,
) -> Vec<(i32, f32)> {
    let batch = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(query).limit(limit))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let ids = batch["id"].as_primitive::<Int32Type>();
    let scores = batch["_score"].as_primitive::<Float32Type>();
    (0..batch.num_rows())
        .map(|i| (ids.value(i), scores.value(i)))
        .collect()
}

/// The scan's ids as a set. A set compare on its own would hide a duplicate
/// emission, so the count is pinned here.
fn unique_ids(ids: Vec<i32>) -> HashSet<i32> {
    let set: HashSet<i32> = ids.iter().copied().collect();
    assert_eq!(set.len(), ids.len(), "duplicate row ids emitted: {ids:?}");
    set
}

/// Assert a scan's `(id, score)` results are exactly the documents
/// [`brute_force_bm25f`] matched, with scores equal to within 1e-3.
///
/// `context` names the invariant the call site is pinning down and is appended to
/// every failure message, so a failure says which one broke.
fn assert_matches_brute_force(actual: &[(i32, f32)], expected: &[Option<f32>], context: &str) {
    let expected_ids = brute_force_ids(expected);
    let actual_ids: HashSet<i32> = actual.iter().map(|(id, _)| *id).collect();
    // A set compare would hide a duplicate emission, so pin the row count too.
    assert_eq!(
        actual.len(),
        expected_ids.len(),
        "duplicate row emitted ({context}): {actual:?}"
    );
    assert_eq!(actual_ids, expected_ids, "matched ids differ ({context})");
    for (id, score) in actual {
        let want = expected[*id as usize].expect("scan returned an unmatched doc");
        assert!(
            (score - want).abs() < 1e-3,
            "score mismatch for id {id} ({context}): scan={score}, brute force={want}"
        );
    }
}

/// Assert `actual` is a valid exact top-`k` of `expected`: the right hit count, the exact
/// ranking's scores, each returned doc carrying its own score, no duplicate or
/// below-cutoff id. Identity is membership rather than an exact id set because ties at
/// the k-th score make the winning set ambiguous.
fn assert_topk_matches_brute_force(
    actual: &[(i32, f32)],
    expected: &[Option<f32>],
    k: usize,
    context: &str,
) {
    let mut expected_ranked: Vec<f32> = expected.iter().filter_map(|score| *score).collect();
    expected_ranked.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    // Not an early return: an all-empty comparison satisfies everything below.
    assert!(
        !expected_ranked.is_empty(),
        "the oracle matched nothing, so there is nothing to compare ({context})"
    );

    assert_eq!(
        actual.len(),
        expected_ranked.len().min(k),
        "hit count mismatch for k={k} ({context})"
    );

    let mut actual_scores: Vec<f32> = actual.iter().map(|(_, score)| *score).collect();
    actual_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    for (got, want) in actual_scores.iter().zip(&expected_ranked) {
        assert!(
            (got - want).abs() < 1e-3,
            "top-{k} score mismatch ({context}): pruned={actual_scores:?} exact={expected_ranked:?}"
        );
    }

    // Per id: sorted scores alone accept ids and scores paired up wrongly.
    for (id, score) in actual {
        let want = expected[*id as usize].expect("scan returned an unmatched doc");
        assert!(
            (score - want).abs() < 1e-3,
            "score mismatch for id {id} at k={k} ({context}): scan={score}, brute force={want}"
        );
    }

    let returned_ids = unique_ids(actual.iter().map(|(id, _)| *id).collect());
    let cutoff = expected_ranked[actual.len() - 1];
    for id in &returned_ids {
        let want = expected[*id as usize].expect("scan returned an unmatched doc");
        assert!(
            want >= cutoff - 1e-3,
            "id {id} scores {want}, below the top-{k} cutoff {cutoff} ({context})"
        );
    }
}

#[tokio::test]
async fn test_fts_combined_fields_cross_field_and() {
    // combined_fields (BM25F) treats the target columns as one virtual field, so an
    // AND query matches when each term appears in at least one field. best_fields
    // (MultiMatch) evaluates AND per field, so it only matches a single field that
    // contains every term.
    let params = InvertedIndexParams::default();
    // row 0: the two terms are split across the fields;
    // row 1: both terms live in a single field;
    // row 2: only one of the two terms appears anywhere.
    let batch = combined_fields_batch(
        vec![0, 1, 2],
        vec!["john", "john smith", "john"],
        vec!["smith", "foo", "alice"],
    );
    let test_uri = TempStrDir::default();
    // Spread the rows across fragments so the merged scan exercises multi-fragment
    // row-id resolution.
    let mut dataset = write_fts_dataset(
        &test_uri,
        batch,
        Some(WriteParams {
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;

    let columns = vec!["title".to_string(), "body".to_string()];
    let combined = |op| combined_query("john smith", op);
    let multi = |op| {
        FtsQuery::MultiMatch(
            MultiMatchQuery::try_new("john smith".to_string(), columns.clone())
                .unwrap()
                .with_operator(op),
        )
    };

    // AND over the virtual field: row 0 (john|title + smith|body) and row 1
    // (both terms in title) match; row 2 (no "smith" anywhere) does not.
    assert_eq!(
        unique_ids(fts_result_ids(&dataset, combined(Operator::And)).await),
        HashSet::from([0, 1])
    );
    // best_fields AND matches only row 1, where a single field holds both terms.
    assert_eq!(
        unique_ids(fts_result_ids(&dataset, multi(Operator::And)).await),
        HashSet::from([1])
    );

    // OR matches any doc containing either term: every row has "john".
    assert_eq!(
        unique_ids(fts_result_ids(&dataset, combined(Operator::Or)).await),
        HashSet::from([0, 1, 2])
    );
    assert_eq!(
        unique_ids(fts_result_ids(&dataset, multi(Operator::Or)).await),
        HashSet::from([0, 1, 2])
    );
}

#[tokio::test]
async fn test_fts_combined_fields_boost_ranking() {
    // Per-column boosts move a document up the ranking in the BM25F direction:
    // boosting the field a term lives in counts that term more. Stemming and
    // stop words are disabled so the filler tokens contribute to document length
    // ("other" is an English stop word).
    let params = InvertedIndexParams::new("simple".to_string(), Language::English)
        .stem(false)
        .remove_stop_words(false);
    // id 0 has the term only in `title`; id 1 has it only in the shorter `body`.
    let batch = combined_fields_batch(
        vec![0, 1],
        vec!["lance", "other"],
        vec!["other other other", "lance"],
    );
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, batch, None).await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;

    let combined =
        |boosts: Vec<f32>| combined_query_with_boosts("lance", Operator::Or, Some(boosts));
    // Compare by score rather than result row-order (FTS batch order is not a
    // guaranteed ranking; only the scores are).
    let scores = |ids: Vec<(i32, f32)>| ids.into_iter().collect::<HashMap<i32, f32>>();

    // Equal weights: the shorter document (id 1, term in the 1-token body) wins.
    let equal = scores(fts_result_id_scores(&dataset, combined(vec![1.0, 1.0]), None).await);
    assert!(
        equal[&1] > equal[&0],
        "equal weights: {equal:?} should rank id 1 above id 0"
    );
    // Boosting `title` 3x lifts id 0 (term in title) above id 1.
    let boosted = scores(fts_result_id_scores(&dataset, combined(vec![3.0, 1.0]), None).await);
    assert!(
        boosted[&0] > boosted[&1],
        "title-boosted: {boosted:?} should rank id 0 above id 1"
    );
}

#[tokio::test]
async fn test_fts_combined_fields_nulls() {
    // NULL edge cases: a doc null in one field is still scored via the other; a
    // doc null in every field never matches; an entirely-null column is a no-op.
    let params = InvertedIndexParams::default();
    let batch = record_batch!(
        ("id", Int32, [0, 1, 2, 3]),
        ("title", Utf8, [Some("lance"), None, None, Some("lance")]),
        ("body", Utf8, [None, Some("lance"), None, Some("lance")]),
        ("empty", Utf8, [None::<&str>, None, None, None])
    )
    .unwrap();
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, batch, None).await;
    create_inverted_indices(&mut dataset, &["title", "body", "empty"], &params).await;

    // Null in one field (0, 1) is scored via the other; null in both (2) never
    // matches; present in both (3) matches once.
    let query = combined_query("lance", Operator::Or);
    assert_eq!(
        unique_ids(fts_result_ids(&dataset, query).await),
        HashSet::from([0, 1, 3])
    );

    // An entirely-null column contributes nothing (docCount' uses the max), so
    // combining it with `title` matches exactly the `title` hits.
    let with_empty = combined_query_over(&["title", "empty"], "lance", Operator::Or, None);
    assert_eq!(
        unique_ids(fts_result_ids(&dataset, with_empty).await),
        HashSet::from([0, 3])
    );
}

#[tokio::test]
async fn test_fts_combined_fields_matches_brute_force_bm25f() {
    // Validate the combined-fields scan against an independent brute-force BM25F
    // reference (the primary correctness oracle). Stemming and stop words are
    // disabled so the reference can tokenize by whitespace.
    let params = combined_fields_test_params();

    let titles = vec!["aa bb", "aa", "cc dd", "aa aa bb", "cc"];
    let bodies = vec!["cc", "bb cc dd", "aa", "dd", "aa bb"];
    let batch = combined_fields_batch(
        (0..titles.len() as i32).collect(),
        titles.clone(),
        bodies.clone(),
    );
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, batch, None).await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;

    let weights = [2.0f32, 1.0f32];
    let query = combined_query_with_boosts("aa bb", Operator::Or, Some(weights.to_vec()));

    let expected = brute_force_bm25f(
        &[(weights[0], titles.clone()), (weights[1], bodies.clone())],
        "aa bb",
        false,
    );
    let actual = fts_result_id_scores(&dataset, query, None).await;
    assert_matches_brute_force(&actual, &expected, "the fully-indexed two-column scan");
}

/// `combined_fields` against a real released-format index read from disk rather
/// than a synthetic fixture, so the row-granularity statistics path
/// ([`lance_index::scalar::inverted::index::InvertedIndex::bm25_row_stats_for_terms`])
/// is exercised on actual V1 and V2 files.
///
/// The checked-in fixtures index a plain `Utf8` column, so every row owns exactly
/// one document and they cannot reproduce the list-element multiplicity the
/// row-granularity path exists for (that lives in `combined/search.rs`'s unit tests).
/// What they do pin is that a released-format index scores against the
/// brute-force BM25F reference.
#[rstest]
#[case::v1("v3.0.1/fts_v1", 1)]
#[case::v2("v4.0.1/fts_v2", 2)]
#[tokio::test]
async fn test_fts_combined_fields_on_released_format_fixture(
    #[case] fixture_path: &str,
    #[case] expected_version: i32,
) {
    let test_dir = copy_test_data_to_tmp(fixture_path).unwrap();
    let dataset = Dataset::open(&test_dir.path_str()).await.unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].index_version, expected_version);

    // `test_data/{v3.0.1,v4.0.1}/datagen.py` writes 300 rows of
    // "lance database compatibility shared" for id % 3 == 0 and
    // "database lance compatibility shared" otherwise.
    const NUM_ROWS: usize = 300;
    let texts: Vec<&str> = (0..NUM_ROWS)
        .map(|id| {
            if id % 3 == 0 {
                "lance database compatibility shared"
            } else {
                "database lance compatibility shared"
            }
        })
        .collect();
    let expected = brute_force_bm25f(&[(1.0, texts)], "lance compatibility", false);

    let query = combined_query_over(&["text"], "lance compatibility", Operator::Or, None);
    let batch = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(query))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int64Type>();
    let scores = batch
        .column_by_name("_score")
        .unwrap()
        .as_primitive::<Float32Type>();
    assert_eq!(batch.num_rows(), NUM_ROWS);
    for i in 0..batch.num_rows() {
        let id = ids.value(i) as usize;
        let want = expected[id].expect("every fixture row carries both query terms");
        assert!(
            (scores.value(i) - want).abs() < 1e-3,
            "score mismatch for id {id}: scan={}, brute force={want}",
            scores.value(i),
        );
    }
}

/// Drive the top-k path end-to-end through the scanner and confirm the pruned
/// top-k equals the exact brute-force BM25F top-k for OR and AND across every k.
///
/// Only a limited query arms MAXSCORE pruning; an unlimited one takes the exact
/// scan. The corpus is skewed, mixing a rare high-idf term ("zeta") with a common
/// low-idf term ("beta"), so MAXSCORE makes "beta" non-essential. The OR query
/// additionally carries an absent term ("missingterm", idf' == 0) to check the
/// clamped-ceiling handling.
///
/// `max_rows_per_file` picks the partition layout, so the multi-partition case also
/// exercises the cursors' cross-partition row-id merge.
#[rstest]
#[case::single_partition(None)]
#[case::multi_partition(Some(7))]
#[tokio::test]
async fn test_fts_combined_fields_topk_matches_brute_force(
    #[case] max_rows_per_file: Option<usize>,
) {
    let params = combined_fields_test_params();

    let n = 40usize;
    let titles: Vec<String> = (0..n)
        .map(|i| {
            if i % 9 == 0 {
                "zeta gamma".to_string()
            } else {
                "gamma".to_string()
            }
        })
        .collect();
    let bodies: Vec<String> = (0..n)
        .map(|i| {
            // "beta" fills every body with a growing count (common, low idf);
            // one body also carries the rare "zeta" so a doc can hold it in
            // either field.
            let beta = vec!["beta"; 1 + i % 3].join(" ");
            if i == 3 { format!("{beta} zeta") } else { beta }
        })
        .collect();

    let title_refs: Vec<&str> = titles.iter().map(|s| s.as_str()).collect();
    let body_refs: Vec<&str> = bodies.iter().map(|s| s.as_str()).collect();
    let batch = combined_fields_batch(
        (0..n as i32).collect(),
        title_refs.clone(),
        body_refs.clone(),
    );
    let test_uri = TempStrDir::default();
    let write_params = max_rows_per_file.map(|max_rows_per_file| WriteParams {
        max_rows_per_file,
        ..Default::default()
    });
    let mut dataset = write_fts_dataset(&test_uri, batch, write_params).await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;

    let weights = [2.0f32, 1.0f32];

    for operator in [Operator::Or, Operator::And] {
        let require_all = operator == Operator::And;
        // The absent term is only valid for OR (AND with an absent term matches
        // nothing); keep it out of the AND case.
        let query_str = if require_all {
            "zeta beta"
        } else {
            "zeta beta missingterm"
        };
        let expected = brute_force_bm25f(
            &[
                (weights[0], title_refs.clone()),
                (weights[1], body_refs.clone()),
            ],
            query_str,
            require_all,
        );
        for k in [1usize, 2, 3, 5, 10, 50] {
            let query = combined_query_with_boosts(query_str, operator, Some(weights.to_vec()));
            let actual = fts_result_id_scores(&dataset, query, Some(k as i64)).await;
            assert_topk_matches_brute_force(&actual, &expected, k, &format!("op={operator:?}"));
        }
    }
}

/// Top-k recall where a term's postings span several [`BLOCK_SIZE`] blocks, so the
/// cursors decode and merge across block boundaries. `beta` sits under `docCount'`
/// (at equality its ceiling is 0 and pruning is trivial) with fewer `zeta`
/// documents than `beta` has blocks.
///
/// `alpha` has no such margin and covers the same paths with pruning that cannot
/// fire. `max_rows_per_file` varies fragments, not partitions.
#[rstest]
#[case::single_fragment(None)]
#[case::multi_fragment(Some(BLOCK_SIZE))]
#[tokio::test]
async fn test_fts_combined_fields_topk_matches_brute_force_across_blocks(
    #[case] max_rows_per_file: Option<usize>,
) {
    const CORPUS_BLOCKS: usize = 16;
    let num_docs = CORPUS_BLOCKS * BLOCK_SIZE;
    let params = combined_fields_test_params();

    // Also derive the expected match counts, so corpus and expectations cannot drift.
    // Three `zeta` probes leave most of `beta`'s ten blocks untouched, and the first is
    // early so the threshold rises before much is decoded.
    let has_zeta = |doc: usize| [3, 703, 1403].contains(&doc);
    let has_beta = |doc: usize| doc % 5 >= 2;
    let has_alpha = |doc: usize| doc.is_multiple_of(7);
    // A subset, so its cursor merges two sources and no match count changes.
    let has_beta_in_title = |doc: usize| doc % 5 == 3;
    let count = |carries: &dyn Fn(usize) -> bool| (0..num_docs).filter(|&d| carries(d)).count();

    // `gamma` and `delta` are filler: no query uses them, they just vary `dl'`.
    let titles: Vec<String> = (0..num_docs)
        .map(|doc| {
            let mut title = if has_zeta(doc) {
                "zeta gamma".to_string()
            } else {
                "gamma".to_string()
            };
            if has_beta_in_title(doc) {
                title.push_str(" beta");
            }
            title
        })
        .collect();
    let bodies: Vec<String> = (0..num_docs)
        .map(|doc| {
            let mut body = if has_beta(doc) {
                vec!["beta"; 1 + doc % 3].join(" ")
            } else {
                "delta".to_string()
            };
            if has_alpha(doc) {
                body.push_str(" alpha");
            }
            body
        })
        .collect();

    // `beta` multi-block with a real ceiling, and rarer terms above it by ceiling.
    let beta_docs = count(&has_beta);
    let zeta_docs = count(&has_zeta);
    let alpha_docs = count(&has_alpha);
    assert!(
        beta_docs > 4 * BLOCK_SIZE && beta_docs < num_docs,
        "beta must span several blocks without covering the corpus, got {beta_docs} of {num_docs}"
    );
    assert!(
        zeta_docs < beta_docs / BLOCK_SIZE,
        "zeta ({zeta_docs} docs) must stay under beta's block count ({})",
        beta_docs / BLOCK_SIZE
    );
    assert!(
        2 * alpha_docs < beta_docs && 2 * zeta_docs < beta_docs,
        "the rare terms must stay well below beta ({beta_docs}): zeta={zeta_docs} alpha={alpha_docs}"
    );
    let beta_title_docs = count(&has_beta_in_title);
    assert!(
        beta_title_docs > 0 && count(&|doc| has_beta_in_title(doc) && !has_beta(doc)) == 0,
        "beta needs a second source in `title`, drawn from its own documents, got {beta_title_docs}"
    );

    let title_refs: Vec<&str> = titles.iter().map(|title| title.as_str()).collect();
    let body_refs: Vec<&str> = bodies.iter().map(|body| body.as_str()).collect();
    let batch = combined_fields_batch(
        (0..num_docs as i32).collect(),
        title_refs.clone(),
        body_refs.clone(),
    );
    let test_uri = TempStrDir::default();
    let write_params = max_rows_per_file.map(|max_rows_per_file| WriteParams {
        max_rows_per_file,
        ..Default::default()
    });
    let mut dataset = write_fts_dataset(&test_uri, batch, write_params).await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;

    let weights = vec![2.0f32, 1.0f32];

    for (terms, operator, want_matches) in [
        (
            "zeta beta",
            Operator::Or,
            count(&|doc| has_zeta(doc) || has_beta(doc)),
        ),
        (
            "alpha beta",
            Operator::Or,
            count(&|doc| has_alpha(doc) || has_beta(doc)),
        ),
        (
            "zeta beta",
            Operator::And,
            count(&|doc| has_zeta(doc) && has_beta(doc)),
        ),
    ] {
        let expected = brute_force_bm25f(
            &[
                (weights[0], title_refs.clone()),
                (weights[1], body_refs.clone()),
            ],
            terms,
            operator == Operator::And,
        );
        // The oracle has to agree with the predicates the corpus was built from.
        assert_eq!(
            brute_force_ids(&expected).len(),
            want_matches,
            "corpus no longer matches as intended for {terms:?} {operator:?}"
        );
        assert!(want_matches > 0, "no matches for {terms:?} {operator:?}");
        // Below, at, and above the match count, to cover the exhausted-cursor path.
        for k in [1usize, 10, 100, num_docs] {
            let query = combined_query_with_boosts(terms, operator, Some(weights.clone()));
            let actual = fts_result_id_scores(&dataset, query, Some(k as i64)).await;
            assert_topk_matches_brute_force(
                &actual,
                &expected,
                k,
                &format!("terms={terms:?} op={operator:?}"),
            );
        }
    }
}

#[tokio::test]
async fn test_fts_combined_fields_tokenizer_validation() {
    // combined_fields accepts columns that differ only in storage-only params
    // (e.g. with_position) but rejects columns configured with different
    // tokenizers.
    let with_pos = InvertedIndexParams::new("simple".to_string(), Language::English)
        .with_position(true)
        .stem(false)
        .remove_stop_words(false);
    let without_pos = InvertedIndexParams::new("simple".to_string(), Language::English)
        .with_position(false)
        .stem(false)
        .remove_stop_words(false);
    let whitespace = InvertedIndexParams::new("whitespace".to_string(), Language::English)
        .stem(false)
        .remove_stop_words(false);

    let batch = record_batch!(
        ("id", Int32, [0, 1]),
        ("title", Utf8, ["aa", "bb"]),
        ("body", Utf8, ["bb", "aa"]),
        ("alt", Utf8, ["aa", "aa"])
    )
    .unwrap();
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, batch, None).await;
    // One index per column, each with its own tokenizer configuration, which is
    // what the query-time validation reads back.
    create_inverted_indices(&mut dataset, &["title"], &with_pos).await;
    create_inverted_indices(&mut dataset, &["body"], &without_pos).await;
    create_inverted_indices(&mut dataset, &["alt"], &whitespace).await;

    // title (with positions) and body (without) tokenize identically: accepted.
    let accepted = combined_query("aa", Operator::Or);
    assert_eq!(
        unique_ids(fts_result_ids(&dataset, accepted).await),
        HashSet::from([0, 1])
    );

    // title (simple) and alt (whitespace) use different tokenizers: rejected.
    let rejected = combined_query_over(&["title", "alt"], "aa", Operator::Or, None);
    let result = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new_query(rejected))
        .unwrap()
        .try_into_batch()
        .await;
    let message = result
        .expect_err("expected a tokenizer-mismatch error")
        .to_string();
    assert!(
        message.contains("combined_fields") && message.contains("tokenizer"),
        "unexpected error: {message}"
    );
}

#[tokio::test]
async fn test_fts_combined_fields_concatenation_identity() {
    // With integer weights, BM25F over (title^w_t, body^w_b) is identical to plain
    // BM25 over a single column that concatenates title repeated w_t times with body
    // repeated w_b times, as long as every title token also appears in body so the
    // max-based docFreq' blend matches the concatenated union. This cross-checks the
    // combined scan against Lance's own single-field BM25, an independent code path,
    // catching spec-interpretation errors.
    let params = combined_fields_test_params();
    let titles = ["cat", "dog", "bird", "cat dog"];
    let bodies = ["cat dog", "dog bird cat", "bird cat", "cat dog bird"];
    let (w_title, w_body) = (2usize, 1usize);
    let concat: Vec<String> = titles
        .iter()
        .zip(&bodies)
        .map(|(title, body)| {
            let mut parts: Vec<&str> = Vec::with_capacity(w_title + w_body);
            for _ in 0..w_title {
                parts.push(title);
            }
            for _ in 0..w_body {
                parts.push(body);
            }
            parts.join(" ")
        })
        .collect();

    let concat_refs: Vec<&str> = concat.iter().map(|s| s.as_str()).collect();
    let batch = record_batch!(
        ("id", Int32, (0..titles.len() as i32).collect::<Vec<_>>()),
        ("title", Utf8, titles.to_vec()),
        ("body", Utf8, bodies.to_vec()),
        ("concat", Utf8, concat_refs)
    )
    .unwrap();
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, batch, None).await;
    create_inverted_indices(&mut dataset, &["title", "body", "concat"], &params).await;

    let combined = combined_query_with_boosts(
        "cat dog",
        Operator::Or,
        Some(vec![w_title as f32, w_body as f32]),
    );
    let plain = FtsQuery::Match(
        MatchQuery::new("cat dog".to_string()).with_column(Some("concat".to_string())),
    );

    let combined_scores: HashMap<i32, f32> = fts_result_id_scores(&dataset, combined, None)
        .await
        .into_iter()
        .collect();
    let plain_scores: HashMap<i32, f32> = fts_result_id_scores(&dataset, plain, None)
        .await
        .into_iter()
        .collect();

    assert_eq!(
        combined_scores.keys().copied().collect::<HashSet<_>>(),
        plain_scores.keys().copied().collect::<HashSet<_>>(),
    );
    for (id, combined_score) in &combined_scores {
        let plain_score = plain_scores[id];
        assert!(
            (combined_score - plain_score).abs() < 1e-3,
            "id {id}: combined={combined_score}, concatenated single-field={plain_score}"
        );
    }
}

#[rstest]
#[case::or(Operator::Or)]
#[case::and(Operator::And)]
#[tokio::test]
async fn test_fts_combined_fields_covers_unindexed_fragments(#[case] operator: Operator) {
    // A combined_fields query must not silently drop rows appended after the indexes
    // were built. The indexed scan cannot score them (their fragment is in no
    // index), so the planner has to union in a flat scan the same way a single
    // MatchQuery does.
    //
    // Every row here matches, so both children of the union emit and the exact
    // scores also pin down that they scored against a single shared corpus. Only the
    // flat side sees the appended rows, so an indexed child that folded its own
    // statistics would score row 0 on a 1-document corpus while rows 1 and 2 were
    // scored on a 3-document one.
    let params = combined_fields_test_params();
    let titles = vec!["alpha", "alpha alpha", "gamma"];
    let bodies = vec!["omega", "omega", "omega omega"];

    let first = combined_fields_batch(vec![0], vec![titles[0]], vec![bodies[0]]);
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, first, None).await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;

    // Two appended fragments, so the flat side has to cover more than one.
    for row in 1..3 {
        let batch = combined_fields_batch(vec![row as i32], vec![titles[row]], vec![bodies[row]]);
        dataset = write_fts_dataset(
            &test_uri,
            batch,
            Some(WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await;
    }

    // A single-column match already falls back to a flat scan; combined_fields
    // must reach the same rows.
    let match_ids = fts_result_ids(
        &dataset,
        FtsQuery::Match(
            MatchQuery::new("alpha".to_string()).with_column(Some("title".to_string())),
        ),
    )
    .await;
    // Score order, so compare as a set: both alpha-bearing rows must be reached.
    assert_eq!(
        match_ids.iter().copied().collect::<HashSet<_>>(),
        HashSet::from([0, 1])
    );

    let expected = brute_force_bm25f(
        &[(1.0, titles.clone()), (1.0, bodies.clone())],
        "alpha omega",
        operator == Operator::And,
    );
    let actual =
        fts_result_id_scores(&dataset, combined_query("alpha omega", operator), None).await;
    // Row 0 comes from the indexed child and rows 1 and 2 from the flat one, so
    // matching the reference on all three proves the two share a corpus: the flat
    // scan publishes the blend that folds in the appended rows, and the indexed scan
    // waits for it instead of using its own `docCount'`/`docFreq'`/`avgdl'`.
    assert_matches_brute_force(
        &actual,
        &expected,
        "rows in unindexed fragments must be reached exactly once through the index/flat union",
    );
}

/// Per-column index skew, over both row-id schemes.
///
/// The indexed scan is restricted to fragments every target column covers. That
/// restriction has to be expressed in the same id space the index stores, and the
/// inverted index trains with `_rowid` (`TrainingCriteria::with_row_id`), which is a
/// logical stable id when the dataset uses stable row ids and a row address
/// otherwise. A fragment-address block list therefore matches nothing under stable
/// row ids, letting the indexed scan emit the very rows the flat scan also emits,
/// the same duplicate-across-a-UNION failure `do_create_deletion_mask_row_id`
/// documents for issue #6877.
///
/// `Operator::And` is checked here too: it decides matching across the virtual field
/// as a whole, so a row whose terms are split between an indexed and a stale-indexed
/// column must still be admitted, and the flat side has to evaluate the operator over
/// the same blended `tf'` the indexed side does.
#[rstest]
#[case::or_row_addresses(false, Operator::Or)]
#[case::or_stable_row_ids(true, Operator::Or)]
#[case::and_row_addresses(false, Operator::And)]
#[case::and_stable_row_ids(true, Operator::And)]
#[tokio::test]
async fn test_fts_combined_fields_per_column_index_skew(
    #[case] stable_row_ids: bool,
    #[case] operator: Operator,
) {
    // Columns indexed at different dataset versions: `title` covers every
    // fragment, `body` only the first. A fragment indexed for some target columns
    // but not others cannot be scored by the indexed scan at all, `dl'` sums each
    // column's document length and a row missing from a column's DocSet
    // contributes 0, so those fragments must be routed to the flat scan whole,
    // not emitted with a partial `tf'`/`dl'`.
    let params = combined_fields_test_params();
    // Fragment 0 (rows 0-1) ends up indexed for both columns. Fragment 1 (rows
    // 2-3) is indexed for `title` only: row 2 carries the term in `title`, which a
    // partial score would rank wrongly, and row 3 carries it only in `body`, which
    // a partial score would miss entirely.
    let titles = vec!["aa bb", "cc", "aa aa", "dd"];
    let bodies = vec!["cc dd", "aa bb", "dd", "aa bb bb"];
    let write_params = |mode: WriteMode| WriteParams {
        mode,
        max_rows_per_file: 2,
        enable_stable_row_ids: stable_row_ids,
        ..Default::default()
    };

    let first = combined_fields_batch(
        vec![0, 1],
        vec![titles[0], titles[1]],
        vec![bodies[0], bodies[1]],
    );
    let test_uri = TempStrDir::default();
    let mut dataset =
        write_fts_dataset(&test_uri, first, Some(write_params(WriteMode::Create))).await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;

    let second = combined_fields_batch(
        vec![2, 3],
        vec![titles[2], titles[3]],
        vec![bodies[2], bodies[3]],
    );
    let mut dataset =
        write_fts_dataset(&test_uri, second, Some(write_params(WriteMode::Append))).await;
    // Re-index `title` only, so the two columns disagree on fragment coverage.
    create_inverted_indices(&mut dataset, &["title"], &params).await;
    assert_eq!(
        dataset.manifest.uses_stable_row_ids(),
        stable_row_ids,
        "row-id scheme precondition"
    );

    // Non-unit weights so `tf'`/`dl'` actually depend on `w_f` on the flat path.
    // Two terms so `And` differs from `Or`: row 2 carries only "aa", so it drops out
    // under `And` while every other matching row keeps both terms somewhere.
    let weights = [3.0f32, 1.0f32];
    let expected = brute_force_bm25f(
        &[(weights[0], titles.clone()), (weights[1], bodies.clone())],
        "aa bb",
        operator == Operator::And,
    );
    let expected_ids = brute_force_ids(&expected);

    let actual = fts_result_id_scores(
        &dataset,
        combined_query_with_boosts("aa bb", operator, Some(weights.to_vec())),
        None,
    )
    .await;
    let actual_ids: HashSet<i32> = actual.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        actual.len(),
        expected_ids.len(),
        "row emitted twice by both sides of the union: {actual:?}"
    );
    // Row 3 has "aa" only in `body`, whose index is missing fragment 1; an
    // index-only plan cannot see it at all.
    assert!(
        actual_ids.contains(&3),
        "row matching only through the column with stale index coverage was dropped: {actual_ids:?}"
    );
    assert_eq!(actual_ids, expected_ids);

    // Rows 2 and 3 are scored by the flat side, which folds `body`'s missing
    // fragment into the blended statistics, so it reaches the exact BM25F score.
    // A partial `dl'` (missing `body`) would inflate row 2's score instead.
    let scores: HashMap<i32, f32> = actual.into_iter().collect();
    for id in [2, 3] {
        match expected[id as usize] {
            Some(want) => {
                let got = scores[&id];
                assert!(
                    (got - want).abs() < 1e-3,
                    "id {id} scored with partial cross-field data: scan={got}, brute force={want}"
                );
            }
            None => assert!(
                !scores.contains_key(&id),
                "id {id} does not carry every term but {operator:?} returned it"
            ),
        }
    }
}

#[tokio::test]
async fn test_fts_combined_fields_partially_indexed_columns() {
    // Only `title` is indexed. Every fragment is then uncovered for `body`, so the
    // whole scan takes the flat-only planner branch, with no index child at all. That
    // branch is the only one that builds a `CombinedFieldColumn` with no segments,
    // and the only one where a column contributes zero index statistics and its
    // entire contribution comes from the flat scan.
    let params = combined_fields_test_params();
    let titles = vec!["aa bb", "cc", "aa aa", "dd"];
    let bodies = vec!["cc dd", "aa bb", "dd", "aa bb bb"];
    let batch = combined_fields_batch(vec![0, 1, 2, 3], titles.clone(), bodies.clone());
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(
        &test_uri,
        batch,
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await;
    create_inverted_indices(&mut dataset, &["title"], &params).await;

    let weights = [2.0f32, 1.0f32];
    let mut plan_scan = dataset.scan();
    plan_scan
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(combined_query_with_boosts(
            "aa",
            Operator::Or,
            Some(weights.to_vec()),
        )))
        .unwrap();
    let plan = plan_scan.explain_plan(true).await.unwrap();
    assert!(
        plan.contains("FlatCombinedFields"),
        "the flat-only branch must carry a flat child:\n{plan}"
    );
    assert!(
        !plan.contains("CombinedFieldsQuery"),
        "no fragment is covered for `body`, so nothing can be scored by an index \
         child:\n{plan}"
    );

    let expected = brute_force_bm25f(
        &[(weights[0], titles.clone()), (weights[1], bodies.clone())],
        "aa",
        false,
    );
    let actual = fts_result_id_scores(
        &dataset,
        combined_query_with_boosts("aa", Operator::Or, Some(weights.to_vec())),
        None,
    )
    .await;
    // The flat scan owns the entire corpus for `body` and reads `title` from its
    // index, so the blended statistics are exact.
    assert_matches_brute_force(
        &actual,
        &expected,
        "the flat-only plan must score against the blend of its own corpus and \
         `title`'s index",
    );

    // The flat exec emits in scan order and applies no limit, so this branch needs
    // its own score sort: without one the rows come back unordered and `limit`
    // truncates positionally instead of taking the top-k.
    let scores: Vec<f32> = actual.iter().map(|(_, score)| *score).collect();
    assert!(
        scores.windows(2).all(|w| w[0] >= w[1]),
        "flat-only plan is not score-ordered: {actual:?}"
    );

    let best = actual
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(id, _)| *id)
        .unwrap();
    let batch = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(combined_query_with_boosts(
            "aa",
            Operator::Or,
            Some(weights.to_vec()),
        )))
        .unwrap()
        .limit(Some(1), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let top = batch
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>()
        .values()
        .to_vec();
    assert_eq!(
        top,
        vec![best],
        "limit=1 on the flat-only plan did not return the top-scoring row"
    );
}

#[tokio::test]
async fn test_fts_combined_fields_tied_scores_are_deterministic() {
    // The indexed scan returns a deterministic top-k (`combined_maxscore` walks
    // ascending row ids and admits on strict `>`). Once a flat scan is unioned in,
    // ties would break by fragment arrival order unless the sort carries a
    // `row_id ASC` key, so `limit`/`offset` pagination could skip and repeat rows.
    let params = combined_fields_test_params();
    let rows = 24usize;
    let first = combined_fields_batch(vec![0, 1], vec!["aa", "aa"], vec!["aa", "aa"]);
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, first, None).await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;
    // Every row is identical, so the correct answer is a full tie decided by row
    // id. Spread the appended rows over several fragments so the flat scan reads
    // them concurrently and arrival order is genuinely nondeterministic.
    let second = combined_fields_batch(
        (2..rows as i32).collect(),
        vec!["aa"; rows - 2],
        vec!["aa"; rows - 2],
    );
    let dataset = write_fts_dataset(
        &test_uri,
        second,
        Some(WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 3,
            ..Default::default()
        }),
    )
    .await;

    // Rows were written in id order across fragments, so row-id order is id order.
    // With every score tied, `(score DESC, row_id ASC)` pins the top-5 to the five
    // lowest ids, which asserts the specification rather than observing that
    // repeated runs happen to agree.
    let expected_top: Vec<i32> = (0..5).collect();
    let mut seen = HashSet::new();
    for _ in 0..8 {
        let batch = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new_query(combined_query(
                "aa",
                Operator::Or,
            )))
            .unwrap()
            .limit(Some(5), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>()
            .values()
            .to_vec();
        assert_eq!(
            ids, expected_top,
            "tied scores must break by ascending row id"
        );
        seen.insert(ids);
    }
    assert_eq!(
        seen.len(),
        1,
        "top-k over tied scores is nondeterministic across runs: {seen:?}"
    );
}

#[tokio::test]
async fn test_fts_combined_fields_fast_search_without_full_coverage_is_empty() {
    // `fast_search` is index-only. When a target column has no index at all no
    // fragment is fully covered, so the answer is definitionally empty, and the
    // indexed exec must not be built, because it requires every target column to
    // have segments. Mirrors `plan_match_query`, which returns `EmptyExec` here.
    let params = combined_fields_test_params();
    let batch = combined_fields_batch(vec![0, 1], vec!["aa", "bb"], vec!["bb", "aa"]);
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, batch, None).await;
    create_inverted_indices(&mut dataset, &["title"], &params).await;

    let batch = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(combined_query(
            "aa",
            Operator::Or,
        )))
        .unwrap()
        .fast_search()
        .try_into_batch()
        .await
        .expect("fast_search without full coverage should be empty, not an error");
    assert_eq!(batch.num_rows(), 0);
}

#[tokio::test]
async fn test_fts_combined_fields_unindexed_row_wins_topk() {
    // The union applies the top-k limit after re-sorting both sides, so a row in an
    // unindexed fragment must be able to take the single top slot. A plan that pushed
    // the limit into the indexed side only would return the best row the index holds
    // instead.
    let params = combined_fields_test_params();
    let titles = ["filler", "filler", "aa aa aa"];
    let bodies = ["aa", "filler", "aa aa aa"];

    let first = combined_fields_batch(
        vec![0, 1],
        vec![titles[0], titles[1]],
        vec![bodies[0], bodies[1]],
    );
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, first, None).await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;

    // Row 2 is the strongest match for "aa" but lands in an unindexed fragment.
    let second = combined_fields_batch(vec![2], vec![titles[2]], vec![bodies[2]]);
    let dataset = write_fts_dataset(
        &test_uri,
        second,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await;

    let batch = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(combined_query(
            "aa",
            Operator::Or,
        )))
        .unwrap()
        .limit(Some(1), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>()
        .values()
        .to_vec();
    assert_eq!(ids, vec![2], "top-1 was taken from the indexed side only");
}

#[tokio::test]
async fn test_fts_combined_fields_fast_search_skips_uncovered_fragments() {
    // `fast_search` is index-only by contract, so appended rows stay invisible. It
    // must still exclude a fragment that not every target column indexes, otherwise
    // those rows come back with a partial `tf'`/`dl'`.
    let params = combined_fields_test_params();
    let first = combined_fields_batch(vec![0], vec!["aa"], vec!["aa"]);
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, first, None).await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;

    let second = combined_fields_batch(vec![1], vec!["aa"], vec!["aa"]);
    let mut dataset = write_fts_dataset(
        &test_uri,
        second,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await;
    // `title` now covers both fragments, `body` only the first.
    create_inverted_indices(&mut dataset, &["title"], &params).await;

    let batch = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(combined_query(
            "aa",
            Operator::Or,
        )))
        .unwrap()
        .fast_search()
        .try_into_batch()
        .await
        .unwrap();
    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>()
        .values()
        .to_vec();
    assert_eq!(
        ids,
        vec![0],
        "fast_search returned a row whose cross-field data is only partly indexed"
    );
}

#[tokio::test]
async fn test_fts_combined_fields_empty_fragment_list_is_empty() {
    // An explicitly empty fragment list selects no rows. The fully indexed plan
    // carries no fragment restriction, so without an explicit short circuit the
    // indexed scan answers from every fragment the index holds.
    let params = combined_fields_test_params();
    let batch = combined_fields_batch(vec![0, 1], vec!["aa", "bb"], vec!["bb", "aa"]);
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, batch, None).await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;

    let mut scan = dataset.scan();
    scan.project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(combined_query(
            "aa",
            Operator::Or,
        )))
        .unwrap()
        .with_fragments(vec![]);
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("EmptyExec"), "unexpected plan: {plan}");
    assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), 0);
}

#[tokio::test]
async fn test_fts_combined_fields_requires_an_index() {
    // BM25F reads its shared tokenizer configuration off an index, so a query
    // whose target columns are all unindexed is rejected rather than scored with a
    // default tokenizer that may not match how the data would be indexed.
    let batch = combined_fields_batch(vec![0, 1], vec!["aa", "bb"], vec!["bb", "aa"]);
    let test_uri = TempStrDir::default();
    let dataset = write_fts_dataset(&test_uri, batch, None).await;

    let result = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(combined_query(
            "aa",
            Operator::Or,
        )))
        .unwrap()
        .try_into_batch()
        .await;
    let message = result.expect_err("expected an error").to_string();
    assert!(
        message.contains("combined_fields") && message.contains("inverted index"),
        "unexpected error: {message}"
    );
}

/// Like [`combined_fields_batch`] but with optional text values, so a row can be
/// NULL in one target column or in every one of them.
fn nullable_combined_fields_batch(
    ids: Vec<i32>,
    titles: Vec<Option<&str>>,
    bodies: Vec<Option<&str>>,
) -> RecordBatch {
    record_batch!(
        ("id", Int32, ids),
        ("title", Utf8, titles),
        ("body", Utf8, bodies)
    )
    .unwrap()
}

#[tokio::test]
async fn test_fts_combined_fields_flat_nulls_and_empty_strings() {
    // NULL / empty-string handling on the flat path, which has its own counting
    // code: `count_column_into` leaves a row's slots at zero for a NULL,
    // `tokenize_and_count_multi` drops a row that is empty in all target columns,
    // and `flat_field_stats` skips a column whose `dl_f == 0` so the row raises
    // neither that column's `docCount_f` nor its `docFreq_f`. Unlike
    // `test_fts_combined_fields_nulls`, which covers the fully-indexed path, this
    // one also pins the exact scores, where a miscounted `docCount'` shows up.
    let params = combined_fields_test_params();
    // Rows 0-1 are indexed for both columns and carry no query term, so the indexed
    // child emits nothing and every returned row comes from the flat scan, whose
    // blended statistics must still fold in the indexed rows.
    let titles = vec![
        Some("zz zz"),
        Some("zz"),
        None,          // NULL in one column
        Some("aa aa"), // paired with an empty string in the other column
        None,          // NULL in every column: never scored
        Some(""),      // empty in every column: never scored
        Some("aa"),
        None, // NULL in one column, and no query term anywhere
    ];
    let bodies = vec![
        Some("zz"),
        Some("zz zz"),
        Some("aa"),
        Some(""),
        None,
        Some(""),
        Some("aa zz"),
        Some("zz"),
    ];

    let first =
        nullable_combined_fields_batch(vec![0, 1], titles[..2].to_vec(), bodies[..2].to_vec());
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, first, None).await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;

    // Spread the appended rows over several fragments, one of which holds only
    // rows that tokenize to nothing at all.
    let second = nullable_combined_fields_batch(
        (2..titles.len() as i32).collect(),
        titles[2..].to_vec(),
        bodies[2..].to_vec(),
    );
    let dataset = write_fts_dataset(
        &test_uri,
        second,
        Some(WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await;

    // A NULL and an empty string both tokenize to nothing, so the reference models
    // either as `""`.
    fn as_text<'a>(values: &[Option<&'a str>]) -> Vec<&'a str> {
        values.iter().map(|v| v.unwrap_or("")).collect()
    }
    let weights = [2.0f32, 1.0f32];
    let expected = brute_force_bm25f(
        &[
            (weights[0], as_text(&titles)),
            (weights[1], as_text(&bodies)),
        ],
        "aa",
        false,
    );
    assert_eq!(
        brute_force_ids(&expected),
        HashSet::from([2, 3, 6]),
        "reference corpus changed; the null rows must stay unmatched"
    );

    let actual = fts_result_id_scores(
        &dataset,
        combined_query_with_boosts("aa", Operator::Or, Some(weights.to_vec())),
        None,
    )
    .await;
    assert_matches_brute_force(
        &actual,
        &expected,
        "a row empty in every target column must not be scored, and docCount' must \
         count only rows with tokens in that column",
    );
}

/// An `id` + `tags` (a list of strings) + `body` batch, for the list-valued target
/// column cases. `ListOffset` picks `List` vs `LargeList` and `StringOffset` picks
/// `Utf8` vs `LargeUtf8`, both for the list elements and for `body`. A `None` entry
/// in `tags` is a NULL list, an empty `Vec` an empty one.
fn list_column_batch<
    ListOffset: arrow::array::OffsetSizeTrait,
    StringOffset: arrow::array::OffsetSizeTrait,
>(
    ids: &[i32],
    tags: &[Option<Vec<&str>>],
    bodies: &[&str],
) -> RecordBatch {
    let mut builder =
        GenericListBuilder::<ListOffset, _>::new(GenericStringBuilder::<StringOffset>::new());
    for row in tags {
        match row {
            Some(elements) => {
                for element in elements {
                    builder.values().append_value(element);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    let tags_col = Arc::new(builder.finish()) as ArrayRef;
    let bodies_col =
        Arc::new(GenericStringArray::<StringOffset>::from(bodies.to_vec())) as ArrayRef;
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("tags", tags_col.data_type().clone(), true),
            ArrowField::new("body", bodies_col.data_type().clone(), false),
        ])),
        vec![
            Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef,
            tags_col,
            bodies_col,
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn test_fts_combined_fields_flat_list_column() {
    assert_combined_fields_flat_list_column::<i32, i32>().await;
}

#[tokio::test]
async fn test_fts_combined_fields_flat_large_list_column() {
    assert_combined_fields_flat_list_column::<i64, i64>().await;
}

/// The flat scan must reduce a list column to a row's document text the same way
/// the index builder does, which is to join the elements with a space.
///
/// Under `simple` the two readings agree, because counting each element on its own
/// yields the same tokens as counting them joined; only a tokenizer that is
/// sensitive to the element boundary can tell them apart. `raw` emits one token
/// per document, so a 2-element list is `dl_f = 1` when joined and `dl_f = 2` when
/// counted per element.
///
/// Scoring the same rows once through each side pins the agreement down without a
/// hand-computed reference: with `tags` indexed the statistics come from the
/// index, and with only `body` indexed they come from the flat fold instead.
#[tokio::test]
async fn test_fts_combined_fields_flat_list_matches_index_under_raw_tokenizer() {
    let params = InvertedIndexParams::new("raw".to_string(), Language::English)
        .lower_case(false)
        .stem(false)
        .remove_stop_words(false);

    // Row 1 is the discriminator: joined, its tags are the single token
    // "alpha beta gamma", which does not match, but counted per element it holds
    // the token "alpha beta", which does. Rows 0 and 2 match either way and pin
    // down the scores.
    let tags: Vec<Option<Vec<&str>>> = vec![
        Some(vec!["alpha beta"]),
        Some(vec!["alpha beta", "gamma"]),
        Some(vec!["gamma", "delta"]),
    ];
    let bodies = vec!["alpha beta", "gamma", "alpha beta"];

    let build = |indexed_columns: &'static [&'static str]| {
        let params = params.clone();
        let tags = tags.clone();
        let bodies = bodies.clone();
        async move {
            let batch = list_column_batch::<i32, i32>(
                &(0..tags.len() as i32).collect::<Vec<_>>(),
                &tags,
                &bodies,
            );
            let test_uri = TempStrDir::default();
            let mut dataset = write_fts_dataset(&test_uri, batch, None).await;
            create_inverted_indices(&mut dataset, indexed_columns, &params).await;
            let scored = fts_result_id_scores(
                &dataset,
                combined_query_over(&["tags", "body"], "alpha beta", Operator::Or, None),
                None,
            )
            .await;
            // Keep the temporary directory alive until the scan has finished.
            drop(test_uri);
            scored
        }
    };

    // `tags` indexed: its statistics and per-row lengths come from the index.
    let indexed = build(&["tags", "body"]).await;
    // `tags` unindexed: every fragment is uncovered for it, so the flat scan reads
    // the raw list and folds its own statistics in.
    let flat = build(&["body"]).await;

    assert_eq!(
        indexed.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![0, 2],
        "row 1 only matches when its tag elements are counted separately"
    );
    assert_eq!(
        indexed.len(),
        flat.len(),
        "flat={flat:?} indexed={indexed:?}"
    );
    for ((indexed_id, indexed_score), (flat_id, flat_score)) in indexed.iter().zip(&flat) {
        assert_eq!(indexed_id, flat_id, "flat={flat:?} indexed={indexed:?}");
        assert!(
            (indexed_score - flat_score).abs() < 1e-5,
            "id {indexed_id}: indexed={indexed_score}, flat={flat_score}; the flat scan must \
             join a row's list elements the way the index builder does"
        );
    }
}

/// A list-valued target column on the flat path: the scan reduces the list to the
/// row's document text and counts it as one document, so every element lands in
/// the row's single `dl_f`/`tf_f`.
async fn assert_combined_fields_flat_list_column<
    ListOffset: arrow::array::OffsetSizeTrait,
    StringOffset: arrow::array::OffsetSizeTrait,
>() {
    let params = combined_fields_test_params();
    // `tags` is the list column. Row 2 spreads one term over two elements of
    // differing length and row 3 repeats the term in two elements, so a reader that
    // took only the first element, or treated each element as its own document,
    // lands on a different `dl_f`/`tf_f` and a different score.
    let tags: Vec<Option<Vec<&str>>> = vec![
        Some(vec!["zz"]),
        Some(vec!["zz", "zz"]),
        Some(vec!["aa", "bb cc"]),
        Some(vec!["aa", "aa"]),
        None,         // NULL list
        Some(vec![]), // empty list
        Some(vec!["zz"]),
    ];
    let bodies = vec!["zz", "zz zz", "zz", "zz zz", "aa", "aa aa", "zz"];

    // Rows 0-1 are indexed for both columns and match nothing, so every returned
    // row is scored by the flat scan.
    let first = list_column_batch::<ListOffset, StringOffset>(&[0, 1], &tags[..2], &bodies[..2]);
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, first, None).await;
    create_inverted_indices(&mut dataset, &["tags", "body"], &params).await;

    let second = list_column_batch::<ListOffset, StringOffset>(
        &(2..tags.len() as i32).collect::<Vec<_>>(),
        &tags[2..],
        &bodies[2..],
    );
    let dataset = write_fts_dataset(
        &test_uri,
        second,
        Some(WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await;

    // The index joins list elements with a space before tokenizing, so the
    // reference sees the same token counts either way.
    let flattened: Vec<String> = tags
        .iter()
        .map(|row| row.as_ref().map(|e| e.join(" ")).unwrap_or_default())
        .collect();
    let weights = [2.0f32, 1.0f32];
    let expected = brute_force_bm25f(
        &[
            (weights[0], flattened.iter().map(|s| s.as_str()).collect()),
            (weights[1], bodies.clone()),
        ],
        "aa",
        false,
    );
    assert_eq!(brute_force_ids(&expected), HashSet::from([2, 3, 4, 5]));

    let actual = fts_result_id_scores(
        &dataset,
        combined_query_over(
            &["tags", "body"],
            "aa",
            Operator::Or,
            Some(weights.to_vec()),
        ),
        None,
    )
    .await;
    assert_matches_brute_force(
        &actual,
        &expected,
        "a row's list elements must sum into one dl_f",
    );
}

/// A target column reached through a list (`List<Struct<Utf8>>`) is a shape a Lance
/// projection cannot address by its public path, so the flat side scans the list
/// root and walks down to the leaf itself.
#[tokio::test]
async fn test_fts_combined_fields_column_under_a_list() {
    let params = combined_fields_test_params();
    // Rows 0-1 are indexed and match "aa" nowhere, so every "aa" hit comes from
    // the flat side and its score is the exact blended BM25F. Row 2 spreads the
    // term over two struct elements and row 4 pairs an empty list with a matching
    // title, so a reader that stopped at the first element, treated each element
    // as its own document, or skipped empty lists lands on a different score.
    let docs: Vec<Vec<&str>> = vec![
        vec!["zz"],
        vec!["zz zz"],
        vec!["aa", "bb aa"],
        vec!["aa aa"],
        vec![],
    ];
    let titles = vec!["zz", "zz", "zz", "aa", "aa aa"];

    let nested_batch = |ids: &[i32], docs: &[Vec<&str>], titles: &[&str]| {
        let content_field = Arc::new(ArrowField::new("content", DataType::Utf8, true));
        let struct_fields = ArrowFields::from(vec![content_field]);
        let contents = docs.iter().flatten().copied().collect::<Vec<_>>();
        let struct_values = StructArray::new(
            struct_fields.clone(),
            vec![Arc::new(StringArray::from(contents)) as ArrayRef],
            None,
        );
        let item = Arc::new(ArrowField::new(
            "item",
            DataType::Struct(struct_fields),
            true,
        ));
        let mut offsets = Vec::with_capacity(docs.len() + 1);
        offsets.push(0i32);
        for row in docs {
            offsets.push(offsets.last().unwrap() + row.len() as i32);
        }
        let docs_col = Arc::new(ListArray::new(
            item,
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(struct_values),
            None,
        )) as ArrayRef;
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, false),
                ArrowField::new("docs", docs_col.data_type().clone(), true),
                ArrowField::new("title", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef,
                docs_col,
                Arc::new(StringArray::from(titles.to_vec())) as ArrayRef,
            ],
        )
        .unwrap()
    };

    let first = nested_batch(&[0, 1], &docs[..2], &titles[..2]);
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, first, None).await;
    create_inverted_indices(&mut dataset, &["docs.content", "title"], &params).await;

    let weights = [2.0f32, 1.0f32];
    // The index joins a row's list elements with a space before tokenizing, so the
    // reference counts the same tokens either way.
    let joined: Vec<String> = docs.iter().map(|row| row.join(" ")).collect();
    let assert_scored_like_reference = |dataset: Dataset, terms: &'static str, rows: usize| {
        let joined = joined.clone();
        let titles = titles.clone();
        async move {
            let expected = brute_force_bm25f(
                &[
                    (
                        weights[0],
                        joined[..rows].iter().map(|s| s.as_str()).collect(),
                    ),
                    (weights[1], titles[..rows].to_vec()),
                ],
                terms,
                false,
            );
            assert!(
                !brute_force_ids(&expected).is_empty(),
                "the reference must match something"
            );
            let actual = fts_result_id_scores(
                &dataset,
                combined_query_over(
                    &["docs.content", "title"],
                    terms,
                    Operator::Or,
                    Some(weights.to_vec()),
                ),
                None,
            )
            .await;
            assert_matches_brute_force(
                &actual,
                &expected,
                "a target column reached through a list must be walked down to its leaf",
            );
        }
    };

    // Fully indexed, so the indexed child owns every hit.
    assert_scored_like_reference(dataset.clone(), "zz", 2).await;

    // Appending fragments no index covers routes the new rows to the flat sibling,
    // which has to reach `docs.content` without projecting it.
    let second = nested_batch(
        &(2..docs.len() as i32).collect::<Vec<_>>(),
        &docs[2..],
        &titles[2..],
    );
    let dataset = write_fts_dataset(
        &test_uri,
        second,
        Some(WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await;

    assert_scored_like_reference(dataset, "aa", docs.len()).await;
}

/// A `title`/`body`/`tags` batch, for the three-column stride case.
fn three_column_batch(
    ids: Vec<i32>,
    titles: Vec<&str>,
    bodies: Vec<&str>,
    tags: Vec<&str>,
) -> RecordBatch {
    record_batch!(
        ("id", Int32, ids),
        ("title", Utf8, titles),
        ("body", Utf8, bodies),
        ("tags", Utf8, tags)
    )
    .unwrap()
}

/// Three columns and two terms, with distinct per-column weights.
///
/// Every per-row count is addressed as `(row * num_columns + column) * num_terms +
/// term`. With two columns and two terms that stride is symmetric enough for a
/// transposition to survive. The corpus places each term in a different column per
/// row, so swapping the column and term strides permutes `tf'` and changes the
/// scores.
///
/// Both cases share one corpus and one expected answer: the `flat_side` case only
/// moves where the rows are scored, which also asserts that the flat scan
/// reproduces the fully-indexed scores exactly.
#[rstest]
#[case::fully_indexed(false)]
#[case::flat_side(true)]
#[tokio::test]
async fn test_fts_combined_fields_three_columns_two_terms(#[case] append_unindexed: bool) {
    let params = combined_fields_test_params();
    // Rows 0-1 carry no query term, so they are never returned and the `flat_side`
    // case can be compared against the same exact scores as the indexed one.
    let titles = vec!["zz", "zz zz", "aa", "bb bb", "zz", "aa aa"];
    let bodies = vec!["zz zz", "zz", "bb", "zz", "aa bb", "zz zz"];
    let tags = vec!["zz", "zz zz", "zz", "aa", "bb", "aa bb"];
    let columns = ["title", "body", "tags"];
    let weights = [3.0f32, 2.0f32, 1.0f32];

    let test_uri = TempStrDir::default();
    let write_params = |mode: WriteMode| WriteParams {
        mode,
        max_rows_per_file: 2,
        ..Default::default()
    };
    let indexed_rows = if append_unindexed { 2 } else { titles.len() };
    let first = three_column_batch(
        (0..indexed_rows as i32).collect(),
        titles[..indexed_rows].to_vec(),
        bodies[..indexed_rows].to_vec(),
        tags[..indexed_rows].to_vec(),
    );
    let mut dataset =
        write_fts_dataset(&test_uri, first, Some(write_params(WriteMode::Create))).await;
    create_inverted_indices(&mut dataset, &columns, &params).await;

    let dataset = if append_unindexed {
        let second = three_column_batch(
            (indexed_rows as i32..titles.len() as i32).collect(),
            titles[indexed_rows..].to_vec(),
            bodies[indexed_rows..].to_vec(),
            tags[indexed_rows..].to_vec(),
        );
        write_fts_dataset(&test_uri, second, Some(write_params(WriteMode::Append))).await
    } else {
        dataset
    };

    for operator in [Operator::Or, Operator::And] {
        let expected = brute_force_bm25f(
            &[
                (weights[0], titles.clone()),
                (weights[1], bodies.clone()),
                (weights[2], tags.clone()),
            ],
            "aa bb",
            operator == Operator::And,
        );
        assert_eq!(
            brute_force_ids(&expected),
            HashSet::from([2, 3, 4, 5]),
            "reference corpus changed; rows 0-1 must stay unmatched"
        );

        let actual = fts_result_id_scores(
            &dataset,
            combined_query_over(&columns, "aa bb", operator, Some(weights.to_vec())),
            None,
        )
        .await;
        assert_matches_brute_force(
            &actual,
            &expected,
            &format!("op={operator:?}: the three-column, two-term stride must not transpose"),
        );
    }
}

/// A `combined_fields` query with a filter, over a mixed-coverage dataset.
///
/// `plan_flat_combined_fields_query` threads the filter plan into
/// `filtered_read(is_prefilter=true)`. When the filter needs a refine expression it
/// also extends the scan projection with that expression's columns and wraps the
/// read in a `LanceFilterExec`, which means the document columns no longer sit at
/// the positions the query lists them in, so `doc_col_indices` has to resolve them
/// by name.
///
/// Both cases filter to the same rows through different plan shapes: `id >= 4`
/// resolves entirely through the scalar index (no refine), while `category` has no
/// index so the whole predicate becomes a refine expression.
#[rstest]
#[case::scalar_index_prefilter("id >= 4", true)]
#[case::refine_expression("category = 'keep'", false)]
#[tokio::test]
async fn test_fts_combined_fields_with_filter(
    #[case] filter: &str,
    #[case] index_filter_column: bool,
) {
    let params = combined_fields_test_params();
    // Rows 0-1 are indexed for both text columns and match nothing, so every
    // returned row comes from the flat side and its score can be checked exactly.
    // Rows 2-3 match but are filtered out, so they contribute no corpus statistics
    // either: the flat scan never sees them.
    let titles = vec!["zz zz", "zz", "aa", "zz", "aa aa", "zz"];
    let bodies = vec!["zz", "zz zz", "zz", "aa aa", "zz", "aa"];
    let categories = ["keep", "keep", "drop", "drop", "keep", "keep"];
    // `category` is placed ahead of the document columns, so the refine expression
    // pulling it into the projection shifts their positions.
    let batch = |ids: Vec<i32>, rows: std::ops::Range<usize>| {
        record_batch!(
            ("id", Int32, ids),
            ("category", Utf8, categories[rows.clone()].to_vec()),
            ("title", Utf8, titles[rows.clone()].to_vec()),
            ("body", Utf8, bodies[rows].to_vec())
        )
        .unwrap()
    };

    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, batch(vec![0, 1], 0..2), None).await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;
    if index_filter_column {
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
    }

    let dataset = write_fts_dataset(
        &test_uri,
        batch((2..6).collect(), 2..6),
        Some(WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await;

    // The corpus the scan can see: every row of the indexed fragment (the index
    // statistics are not filtered) plus only the flat rows that survive the filter.
    let stats_rows = [0usize, 1, 4, 5];
    let pick = |values: &[&'static str]| {
        stats_rows
            .iter()
            .map(|&row| values[row])
            .collect::<Vec<_>>()
    };
    let weights = [2.0f32, 1.0f32];
    let expected = brute_force_bm25f(
        &[(weights[0], pick(&titles)), (weights[1], pick(&bodies))],
        "aa",
        false,
    );
    let expected_scores: HashMap<i32, f32> = stats_rows
        .iter()
        .enumerate()
        .filter_map(|(slot, &row)| expected[slot].map(|score| (row as i32, score)))
        .collect();
    assert_eq!(
        expected_scores.keys().copied().collect::<HashSet<_>>(),
        HashSet::from([4, 5]),
        "reference corpus changed; only the kept rows may match"
    );

    let mut scan = dataset.scan();
    scan.project(&["id"])
        .unwrap()
        .prefilter(true)
        .full_text_search(FullTextSearchQuery::new_query(combined_query_with_boosts(
            "aa",
            Operator::Or,
            Some(weights.to_vec()),
        )))
        .unwrap()
        .filter(filter)
        .unwrap();
    let plan = scan.explain_plan(true).await.unwrap();
    assert!(
        plan.contains("FlatCombinedFields"),
        "filtered plan lost its flat child:\n{plan}"
    );
    // Confirm the two cases really took the two different branches, rather than
    // both landing on the same one.
    if index_filter_column {
        assert!(
            plan.contains("ScalarIndexQuery") && plan.contains("refine_filter=--"),
            "expected the filter to resolve entirely through the scalar index:\n{plan}"
        );
        assert!(
            !plan.contains("category"),
            "no refine expression, so `category` must stay out of the projection:\n{plan}"
        );
    } else {
        assert!(
            plan.contains("refine_filter=category") && plan.contains("FilterExec"),
            "expected the filter to become a refine expression:\n{plan}"
        );
        assert!(
            plan.contains("projection=[category, title, body]"),
            "the refine expression's column must be added to the flat scan \
             projection:\n{plan}"
        );
    }

    let scan_batch = scan.try_into_batch().await.unwrap();
    let ids = scan_batch
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>();
    let scores = scan_batch
        .column_by_name("_score")
        .unwrap()
        .as_primitive::<Float32Type>();
    let actual: Vec<(i32, f32)> = (0..scan_batch.num_rows())
        .map(|i| (ids.value(i), scores.value(i)))
        .collect();

    assert_eq!(
        actual.len(),
        expected_scores.len(),
        "filter did not restrict the result set:\n{plan}\n{actual:?}"
    );
    for (id, score) in actual {
        let want = *expected_scores
            .get(&id)
            .unwrap_or_else(|| panic!("filtered-out row {id} was returned:\n{plan}"));
        assert!(
            (score - want).abs() < 1e-3,
            "id {id}: scan={score}, brute force={want}"
        );
    }
}

#[tokio::test]
async fn test_fts_combined_fields_deletions_then_optimize() {
    // Deleted rows must disappear from both sides of the union: the indexed child
    // sees them through the restricted deletion mask, and the flat child through
    // its filtered read. The surviving scores must then be unchanged by
    // `optimize_indices`, because folding a fragment's flat statistics in has to
    // produce exactly the totals that indexing it produces.
    let params = combined_fields_test_params();
    let titles = vec!["aa", "aa aa", "zz", "aa aa aa", "aa", "zz zz"];
    let bodies = vec!["zz", "zz", "aa aa", "zz", "aa aa", "aa"];

    let test_uri = TempStrDir::default();
    let first = combined_fields_batch(vec![0, 1, 2], titles[..3].to_vec(), bodies[..3].to_vec());
    let mut dataset = write_fts_dataset(&test_uri, first, None).await;
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;
    let second = combined_fields_batch(vec![3, 4, 5], titles[3..].to_vec(), bodies[3..].to_vec());
    let mut dataset = write_fts_dataset(
        &test_uri,
        second,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await;

    // One deletion in the indexed fragment, one in the appended (flat) fragment.
    dataset.delete("id in (1, 4)").await.unwrap();

    // Row 1 stays in the index's own statistics, because a delete does not rewrite
    // the `DocSet`, while row 4 never reaches the flat counter, so it drops out of
    // the corpus entirely.
    let stats_rows = [0usize, 1, 2, 3, 5];
    let pick = |values: &[&'static str]| {
        stats_rows
            .iter()
            .map(|&row| values[row])
            .collect::<Vec<_>>()
    };
    let weights = [2.0f32, 1.0f32];
    let expected = brute_force_bm25f(
        &[(weights[0], pick(&titles)), (weights[1], pick(&bodies))],
        "aa",
        false,
    );
    let expected_scores: HashMap<i32, f32> = stats_rows
        .iter()
        .enumerate()
        .filter_map(|(slot, &row)| expected[slot].map(|score| (row as i32, score)))
        .collect();

    let query = || combined_query_with_boosts("aa", Operator::Or, Some(weights.to_vec()));
    let live_ids = HashSet::from([0, 2, 3, 5]);
    let mut scan = dataset.scan();
    scan.project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(query()))
        .unwrap();
    let mixed_plan = scan.explain_plan(true).await.unwrap();
    assert!(
        mixed_plan.contains("FlatCombinedFields"),
        "the appended fragment is unindexed, so the plan must have a flat child:\n{mixed_plan}"
    );

    let mixed = fts_result_id_scores(&dataset, query(), None).await;
    assert_eq!(
        mixed.iter().map(|(id, _)| *id).collect::<HashSet<_>>(),
        live_ids,
        "deleted rows leaked into the mixed index/flat plan: {mixed:?}"
    );
    assert_eq!(mixed.len(), live_ids.len(), "duplicate row: {mixed:?}");
    // Rows 3 and 5 are the ones the flat child scores, so they see the blended
    // statistics in full. Rows 0 and 2 come from the indexed child, which still
    // scores against index-only statistics (a known gap documented at the union).
    let mixed_scores: HashMap<i32, f32> = mixed.into_iter().collect();
    for id in [3, 5] {
        let want = expected_scores[&id];
        let got = mixed_scores[&id];
        assert!(
            (got - want).abs() < 1e-3,
            "id {id} after delete: scan={got}, brute force={want}"
        );
    }

    // Indexing the appended fragment must leave those scores untouched, and the
    // plan must collapse back to a single indexed scan.
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let mut scan = dataset.scan();
    scan.project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(query()))
        .unwrap();
    let plan = scan.explain_plan(true).await.unwrap();
    assert!(
        plan.contains("CombinedFieldsQuery"),
        "expected an indexed combined-fields scan:\n{plan}"
    );
    assert!(
        !plan.contains("FlatCombinedFields"),
        "every fragment is indexed but the plan still has a flat child:\n{plan}"
    );

    let optimized = fts_result_id_scores(&dataset, query(), None).await;
    assert_eq!(
        optimized.iter().map(|(id, _)| *id).collect::<HashSet<_>>(),
        live_ids,
        "deleted rows came back after optimize: {optimized:?}"
    );
    // Now that every row is indexed there is a single corpus, so every score has to
    // match the reference, including the rows the mixed plan scored from the index
    // only.
    for (id, score) in optimized {
        let want = expected_scores[&id];
        assert!(
            (score - want).abs() < 1e-3,
            "id {id} after optimize: scan={score}, brute force={want}"
        );
    }
}

#[tokio::test]
async fn test_fts_combined_fields_nested_columns() {
    // Nested target columns: the flat plan projects the struct column from storage,
    // so it calls `ensure_column_alias` once per target column to expose `s.a` and
    // `s.b` as top-level names the exec can resolve. Both aliases survive only
    // because one projection is nested inside the other.
    let params = combined_fields_test_params();
    let a_values = [
        Some("zz zz"),
        Some("zz"),
        Some("aa"),
        Some("zz"),
        Some("aa aa"),
    ];
    let b_values = [
        Some("zz"),
        Some("zz zz"),
        Some("zz"),
        Some("aa aa"),
        Some("aa"),
    ];

    let first = nested_fts_batch(vec![0, 1], a_values[..2].to_vec(), b_values[..2].to_vec());
    let test_uri = TempStrDir::default();
    let mut dataset = write_fts_dataset(&test_uri, first, None).await;
    create_inverted_indices(&mut dataset, &["s.a", "s.b"], &params).await;
    // Rows 0-1 are indexed and match nothing, so the appended rows are the ones
    // returned and the flat path owns every score.
    let second = nested_fts_batch(
        vec![2, 3, 4],
        a_values[2..].to_vec(),
        b_values[2..].to_vec(),
    );
    let schema = second.schema();
    dataset
        .append(
            RecordBatchIterator::new(vec![second].into_iter().map(Ok), schema),
            None,
        )
        .await
        .unwrap();

    let weights = [2.0f32, 1.0f32];
    let expected = brute_force_bm25f(
        &[
            (
                weights[0],
                a_values.iter().map(|v| v.unwrap_or("")).collect(),
            ),
            (
                weights[1],
                b_values.iter().map(|v| v.unwrap_or("")).collect(),
            ),
        ],
        "aa",
        false,
    );
    let expected_ids: HashSet<u64> = (0..expected.len())
        .filter(|&i| expected[i].is_some())
        .map(|i| i as u64)
        .collect();
    assert_eq!(expected_ids, HashSet::from([2, 3, 4]));

    let mut scan = dataset.scan();
    scan.full_text_search(FullTextSearchQuery::new_query(combined_query_over(
        &["s.a", "s.b"],
        "aa",
        Operator::Or,
        Some(weights.to_vec()),
    )))
    .unwrap();
    let plan = scan.explain_plan(true).await.unwrap();
    assert!(
        plan.contains("FlatCombinedFields"),
        "the appended fragment is unindexed, so the nested columns must be aliased \
         for a flat child:\n{plan}"
    );

    let batch = scan.try_into_batch().await.unwrap();
    let ids = batch["id"].as_primitive::<UInt64Type>();
    let scores = batch["_score"].as_primitive::<Float32Type>();
    let actual: Vec<(u64, f32)> = (0..batch.num_rows())
        .map(|i| (ids.value(i), scores.value(i)))
        .collect();
    assert_eq!(
        actual.len(),
        expected_ids.len(),
        "duplicate row: {actual:?}"
    );
    assert_eq!(
        actual.iter().map(|(id, _)| *id).collect::<HashSet<_>>(),
        expected_ids
    );
    for (id, score) in actual {
        let want = expected[id as usize].expect("scan returned an unmatched doc");
        assert!(
            (score - want).abs() < 1e-3,
            "id {id}: scan={score}, brute force={want}"
        );
    }
}
