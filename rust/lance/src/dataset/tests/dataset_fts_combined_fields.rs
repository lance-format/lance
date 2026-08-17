// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::vec;

use crate::Dataset;

use crate::dataset::write::{WriteMode, WriteParams};
use crate::index::DatasetIndexExt;
use crate::utils::test::copy_test_data_to_tmp;
use arrow::array::AsArray;
use arrow_array::RecordBatch;
use arrow_array::record_batch;
use arrow_array::{
    RecordBatchIterator,
    types::{Float32Type, Int32Type, Int64Type},
};
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::{
    Language,
    query::{MatchQuery, Operator},
    tokenizer::InvertedIndexParams,
};

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
