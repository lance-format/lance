// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A fragment rewritten in place is withdrawn from segment A's stored bitmap
//! while A's file keeps the old entries; optimize indexes it again as segment
//! B. A compaction then joins it with a fragment A still owns, so both
//! segments translate into the same destination. Indexed reads must equal a
//! scan and each row must be answered by one segment; a later optimize merge
//! must not copy the withdrawn entries.
//!
//! Fixture steps are asserted on the raw manifest (`fixture:` messages); the
//! read checks are collected so one mismatch does not hide another.

use super::*;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Bound;

use arrow_array::types::Float32Type;
use arrow_array::{Int32Array, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::inverted::SCORE_COL;
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_table::system_index::frag_reuse::metadata::is_tagged;
use uuid::Uuid;

use crate::dataset::optimize::{
    CompactionPlan, CompactionPlanner, TaskData, compact_files_with_planner,
};
use crate::index::tests::support::raw_manifest_indices;

const BTREE: &str = "i_idx";
const TEXT: &str = "text_idx";

/// Fragments 0 and 1, two rows each.
const ROWS: [(i32, &str); 4] = [
    (0, "alpha graph"),
    (1, "alpha search"),
    (2, "graph"),
    (3, "search theory"),
];

/// Appended later and stably partitioned to tag the history; no index
/// covers them.
const APPENDED: [[(i32, &str); 2]; 2] = [
    [(4, "theory omega"), (5, "omega")],
    [(6, "graph omega"), (7, "search")],
];

/// New `i` and `text` of rows `w = 0` and `w = 1` (fragment 0).
const REWRITTEN_I: [i32; 2] = [100, 101];
const REWRITTEN_TEXT: [&str; 2] = ["beta graph", "gamma"];

const FTS_TERMS: [&str; 6] = ["graph", "search", "theory", "omega", "beta", "gamma"];
/// Only in the old text of fragment 0.
const OLD_ONLY_TERM: &str = "alpha";

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int32, true),
        Field::new("text", DataType::Utf8, true),
        Field::new("w", DataType::Int32, true),
    ]))
}

fn rows_batch(rows: &[(i32, &str)]) -> RecordBatch {
    RecordBatch::try_new(
        table_schema(),
        vec![
            Arc::new(Int32Array::from_iter_values(rows.iter().map(|(i, _)| *i))),
            Arc::new(StringArray::from_iter_values(rows.iter().map(|(_, t)| *t))),
            Arc::new(Int32Array::from_iter_values(rows.iter().map(|(i, _)| *i))),
        ],
    )
    .unwrap()
}

async fn append_rows(dataset: Dataset, rows: &[(i32, &str)]) -> Dataset {
    crate::dataset::InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![rows_batch(rows)])
        .await
        .unwrap()
}

fn text_params() -> InvertedIndexParams {
    InvertedIndexParams::default()
        .lower_case(true)
        .stem(false)
        .remove_stop_words(false)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    BTree,
    Fts,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::BTree => BTREE,
            Self::Fts => TEXT,
        }
    }

    async fn create(self, dataset: &mut Dataset) {
        let result = match self {
            Self::BTree => {
                dataset
                    .create_index(
                        &["i"],
                        IndexType::BTree,
                        Some(BTREE.into()),
                        &ScalarIndexParams::default(),
                        false,
                    )
                    .await
            }
            Self::Fts => {
                dataset
                    .create_index(
                        &["text"],
                        IndexType::Inverted,
                        Some(TEXT.into()),
                        &text_params(),
                        false,
                    )
                    .await
            }
        };
        result.expect("fixture: the index builds");
    }

    async fn rewrite(self, uri: &str, key: i32) {
        let index = key as usize;
        match self {
            Self::BTree => {
                rewrite_in_place(
                    uri,
                    "i",
                    key,
                    Arc::new(Int32Array::from(vec![REWRITTEN_I[index]])),
                )
                .await
            }
            Self::Fts => {
                rewrite_in_place(
                    uri,
                    "text",
                    key,
                    Arc::new(StringArray::from(vec![REWRITTEN_TEXT[index]])),
                )
                .await
            }
        }
    }
}

/// Deferred-remap compaction of each of `groups`, planned here because the
/// default planner never bins fragments of different segments together.
async fn compact_groups(dataset: &mut Dataset, groups: &[&[u32]]) {
    struct Groups(CompactionPlan);

    #[async_trait::async_trait]
    impl CompactionPlanner for Groups {
        async fn plan(&self, _dataset: &Dataset) -> crate::Result<CompactionPlan> {
            Ok(self.0.clone())
        }
    }

    let tasks = groups
        .iter()
        .map(|ids| {
            let fragments: Vec<Fragment> = dataset
                .manifest
                .fragments
                .iter()
                .filter(|fragment| ids.contains(&(fragment.id as u32)))
                .cloned()
                .collect();
            assert_eq!(
                fragments.len(),
                ids.len(),
                "fixture: every fragment of {ids:?} is live"
            );
            TaskData::rewrite_fragments(fragments)
        })
        .collect();
    let plan = CompactionPlan {
        tasks,
        read_version: dataset.manifest.version,
        options: CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        },
    };
    compact_files_with_planner(dataset, None, &Groups(plan))
        .await
        .expect("fixture: the planned compaction commits");
}

fn sorted_fragments(dataset: &Dataset) -> Vec<u32> {
    let mut ids = fragment_ids(dataset);
    ids.sort_unstable();
    ids
}

fn assert_live(dataset: &Dataset, expected: &[u32], step: &str) {
    assert_eq!(
        sorted_fragments(dataset),
        expected,
        "fixture: {step}: live fragments"
    );
}

fn only_new<T: Copy + Ord + std::fmt::Debug>(before: &[T], after: &[T], step: &str) -> T {
    let new: Vec<T> = after
        .iter()
        .copied()
        .filter(|item| !before.contains(item))
        .collect();
    assert_eq!(
        new.len(),
        1,
        "fixture: {step}: exactly one new item: {before:?} -> {after:?}"
    );
    new[0]
}

async fn raw_segments(dataset: &Dataset, name: &str) -> Vec<IndexMetadata> {
    raw_manifest_indices(dataset)
        .await
        .into_iter()
        .filter(|idx| idx.name == name)
        .collect()
}

/// Each stored segment of `name` with its RAW manifest fragment bitmap.
async fn raw_bitmaps(dataset: &Dataset, name: &str) -> BTreeMap<Uuid, Vec<u32>> {
    raw_segments(dataset, name)
        .await
        .into_iter()
        .map(|idx| {
            let bitmap = idx
                .fragment_bitmap
                .as_ref()
                .expect("fixture: every stored segment records a fragment bitmap")
                .iter()
                .collect();
            (idx.uuid, bitmap)
        })
        .collect()
}

async fn assert_stored(dataset: &Dataset, name: &str, expected: &[(Uuid, &[u32])], step: &str) {
    let expected: BTreeMap<Uuid, Vec<u32>> = expected
        .iter()
        .map(|(uuid, bitmap)| (*uuid, bitmap.to_vec()))
        .collect();
    assert_eq!(
        raw_bitmaps(dataset, name).await,
        expected,
        "fixture: {step}: stored bitmaps of {name}"
    );
}

async fn raw_segment(dataset: &Dataset, name: &str, uuid: Uuid) -> IndexMetadata {
    raw_segments(dataset, name)
        .await
        .into_iter()
        .find(|idx| idx.uuid == uuid)
        .unwrap_or_else(|| panic!("fixture: segment {uuid} of {name} is still committed"))
}

async fn raw_fri(dataset: &Dataset) -> Option<IndexMetadata> {
    raw_manifest_indices(dataset)
        .await
        .into_iter()
        .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
}

/// `(source ids, destination ids)` of every transition the history holds.
async fn ledger_transitions(dataset: &Dataset) -> Vec<(Vec<u64>, Vec<u64>)> {
    let Some(entry) = raw_fri(dataset).await else {
        return Vec::new();
    };
    crate::index::frag_reuse::decode_frag_reuse_ledger(dataset, &entry)
        .await
        .unwrap()
        .transitions()
        .iter()
        .map(|transition| {
            (
                transition.sources().iter().map(|d| d.id).collect(),
                transition.destinations().iter().map(|d| d.id).collect(),
            )
        })
        .collect()
}

fn short(uuid: &Uuid) -> String {
    uuid.to_string()[..8].to_string()
}

fn every_value() -> SargableQuery {
    SargableQuery::Range(
        Bound::Included(ScalarValue::Int32(Some(i32::MIN))),
        Bound::Included(ScalarValue::Int32(Some(i32::MAX))),
    )
}

fn fragments_of(addrs: &[u64]) -> Vec<u32> {
    let mut fragments: Vec<u32> = addrs
        .iter()
        .map(|addr| lance_core::utils::address::RowAddress::from(*addr).fragment_id())
        .collect();
    fragments.sort_unstable();
    fragments.dedup();
    fragments
}

struct Released {
    _dir: TempStrDir,
    dataset: Dataset,
}

/// Steps 1-5: A over {0, 1}; fragment 0 rewritten in place (A keeps {1});
/// optimize adds B over {0}; fragments 2 and 3 appended and stably
/// partitioned into {10, 11} (the history is tagged); {0, 1} compacted
/// together into one destination with deferred remap.
async fn released_state(kind: Kind) -> Released {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let name = kind.name();

    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(rows_batch(&ROWS))], table_schema()),
        uri,
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert!(
        !dataset.manifest.uses_stable_row_ids(),
        "fixture: the table is written without stable row ids"
    );
    assert_live(&dataset, &[0, 1], "step 1: two fragments");
    kind.create(&mut dataset).await;
    let built = raw_bitmaps(&dataset, name).await;
    assert_eq!(built.len(), 1, "fixture: A is the only segment: {built:?}");
    let a = *built.keys().next().unwrap();
    assert_stored(&dataset, name, &[(a, &[0, 1][..])], "step 1: A is built").await;
    let a_built = raw_segment(&dataset, name, a).await;

    for key in [0, 1] {
        kind.rewrite(uri, key).await;
    }
    let mut dataset = fresh_session(uri).await;
    assert_live(&dataset, &[0, 1], "step 2: the rewrite keeps fragment 0");
    assert_stored(
        &dataset,
        name,
        &[(a, &[1][..])],
        "step 2: the rewrite withdraws fragment 0 from A",
    )
    .await;
    let a_withdrawn = raw_segment(&dataset, name, a).await;
    assert_eq!(
        format!("{:?}", a_withdrawn.files),
        format!("{:?}", a_built.files),
        "fixture: step 2: A's files are unchanged"
    );
    assert_eq!(
        a_withdrawn.dataset_version, a_built.dataset_version,
        "fixture: step 2: A is not rebuilt"
    );
    if kind == Kind::BTree {
        for old in [0, 1] {
            assert_eq!(
                fragments_of(&raw_segment_row_addrs(&dataset, &a_withdrawn, old).await),
                vec![0],
                "fixture: step 2: A's file keeps the old value {old} at fragment 0"
            );
        }
    }

    let before: Vec<Uuid> = raw_bitmaps(&dataset, name).await.into_keys().collect();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .expect("fixture: optimize indexes the rewritten fragment");
    let dataset = fresh_session(uri).await;
    let after: Vec<Uuid> = raw_bitmaps(&dataset, name).await.into_keys().collect();
    let b = only_new(&before, &after, "step 3: optimize adds B");
    let stored: [(Uuid, &[u32]); 2] = [(a, &[1]), (b, &[0])];
    assert_stored(&dataset, name, &stored, "step 3: A {1}, B {0}").await;

    let dataset = append_rows(dataset, &APPENDED[0]).await;
    let mut dataset = append_rows(dataset, &APPENDED[1]).await;
    assert_live(&dataset, &[0, 1, 2, 3], "step 4: two fragments appended");
    reserve_fragments(&mut dataset, 20).await;
    commit_stable_partition(dataset, &[2, 3], 10).await;
    let mut dataset = fresh_session(uri).await;
    assert_live(&dataset, &[0, 1, 10, 11], "step 4: {2, 3} -> {10, 11}");
    assert!(
        raw_fri(&dataset).await.as_ref().is_some_and(is_tagged),
        "fixture: step 4: the history is tagged"
    );
    assert_stored(
        &dataset,
        name,
        &stored,
        "step 4: the partition leaves A and B",
    )
    .await;

    let live = sorted_fragments(&dataset);
    compact_groups(&mut dataset, &[&[0, 1]]).await;
    let dataset = fresh_session(uri).await;
    let destination = only_new(
        &live,
        &sorted_fragments(&dataset),
        "step 5: {0, 1} -> one destination",
    );
    assert_live(&dataset, &[10, 11, destination], "step 5");
    assert!(
        raw_fri(&dataset).await.as_ref().is_some_and(is_tagged),
        "fixture: step 5: the history is tagged"
    );
    assert!(
        ledger_transitions(&dataset)
            .await
            .contains(&(vec![0, 1], vec![destination as u64])),
        "fixture: step 5: the history retains {{0, 1}} -> {{{destination}}}: {:?}",
        ledger_transitions(&dataset).await
    );
    assert_stored(
        &dataset,
        name,
        &stored,
        "step 5: a deferred compaction leaves A and B committed as stored",
    )
    .await;
    Released { _dir: dir, dataset }
}

/// The row addresses each committed segment of [`BTREE`] answers through the
/// query path (fragment reuse translation and coverage applied), per segment.
pub(super) async fn segment_answers(
    dataset: &Dataset,
) -> Vec<(Uuid, std::result::Result<Vec<u64>, String>)> {
    let mut answers = Vec::new();
    for segment in raw_segments(dataset, BTREE).await {
        let opened = crate::index::scalar_logical::open_scalar_index_segments(
            dataset,
            "i",
            BTREE,
            None,
            Some(&HashSet::from([segment.uuid])),
            &NoOpMetricsCollector,
        )
        .await;
        let answer = match opened {
            Err(err) => Err(format!("open: {err}")),
            Ok(index) => match index.search(&every_value(), &NoOpMetricsCollector).await {
                Err(err) => Err(format!("search: {err}")),
                Ok(SearchResult::Exact(rows)) => Ok(rows
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(u64::from)
                    .collect()),
                Ok(_) => Err("not an exact result".to_string()),
            },
        };
        answers.push((segment.uuid, answer));
    }
    answers
}

/// Index reads of the old, new, untouched and unindexed values against the
/// scan, and the number of segments answering each live row.
async fn btree_mismatches(dataset: &Dataset, label: &str) -> Vec<String> {
    let mut mismatches = Vec::new();
    let mut check =
        |predicate: String, indexed: Vec<i32>, scanned: Vec<i32>, expected: Vec<i32>| {
            if indexed != scanned {
                mismatches.push(format!(
                    "{label}: {predicate}: index {indexed:?} != scan {scanned:?}"
                ));
            }
            if scanned != expected {
                mismatches.push(format!(
                    "{label}: fixture: {predicate}: scan {scanned:?} != expected {expected:?}"
                ));
            }
        };
    let cases = [0, 1]
        .into_iter()
        .map(|old| (old, vec![]))
        .chain(REWRITTEN_I.into_iter().map(|new| (new, vec![new])))
        .chain(
            ROWS[2..]
                .iter()
                .chain(APPENDED.iter().flatten())
                .map(|(i, _)| (*i, vec![*i])),
        );
    let mut plans = Vec::new();
    for (value, expected) in cases {
        let predicate = format!("i = {value}");
        let plan = dataset
            .scan()
            .filter(&predicate)
            .unwrap()
            .use_scalar_index(true)
            .explain_plan(false)
            .await
            .unwrap();
        if !plan.contains("ScalarIndexQuery") {
            plans.push(format!(
                "{label}: {predicate}: the plan does not use the index"
            ));
        }
        let indexed = values(dataset, Some(&predicate), true).await;
        let scanned = values(dataset, Some(&predicate), false).await;
        check(predicate, indexed, scanned, expected);
    }
    for predicate in ["i >= 0 AND i < 4", "i >= 100", "i < 1000"] {
        let indexed = values(dataset, Some(predicate), true).await;
        let scanned = values(dataset, Some(predicate), false).await;
        let expected = scanned.clone();
        check(predicate.to_string(), indexed, scanned, expected);
    }
    let (indexed, scanned) = (
        values(dataset, None, true).await,
        values(dataset, None, false).await,
    );
    let expected = scanned.clone();
    check("full scan".to_string(), indexed, scanned, expected);
    mismatches.extend(plans);

    let live: HashMap<u64, i32> = row_addrs_by_i(dataset)
        .await
        .into_iter()
        .map(|(value, addr)| (addr, value))
        .collect();
    let mut served: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    for (uuid, answer) in segment_answers(dataset).await {
        match answer {
            Err(err) => mismatches.push(format!(
                "{label}: segment {} has no answer: {err}",
                short(&uuid)
            )),
            Ok(addrs) => {
                for addr in addrs {
                    served.entry(addr).or_default().push(short(&uuid));
                }
            }
        }
    }
    let mut by_value: Vec<(i32, u64)> = live.iter().map(|(addr, value)| (*value, *addr)).collect();
    by_value.sort_unstable();
    for (value, addr) in by_value {
        let segments = served.get(&addr).cloned().unwrap_or_default();
        if segments.len() > 1 {
            mismatches.push(format!(
                "{label}: the row i = {value} is answered by {} segments {segments:?}",
                segments.len()
            ));
        }
    }
    for (addr, segments) in &served {
        if !live.contains_key(addr) {
            mismatches.push(format!(
                "{label}: segments {segments:?} answer row address {addr} (fragment {}), which is not live",
                lance_core::utils::address::RowAddress::from(*addr).fragment_id()
            ));
        }
    }
    mismatches
}

fn report(mismatches: &[String], label: &str) {
    assert!(
        mismatches.is_empty(),
        "{label}: {} mismatches:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// Reads right after the joined compaction.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn released_in_place_withdrawal_then_joined_compaction_serves_each_row_once() {
    let state = Box::pin(released_state(Kind::BTree)).await;
    let mismatches = btree_mismatches(&state.dataset, "after compaction").await;
    report(&mismatches, "after compaction");
}

/// Reads after the joined compaction and an optimize that merges A and B.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn released_in_place_withdrawal_then_joined_compaction_then_merge_serves_each_row_once() {
    let state = Box::pin(released_state(Kind::BTree)).await;
    let mut dataset = state.dataset;
    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .expect("fixture: optimize merges A and B");
    let dataset = fresh_session(dataset.uri()).await;
    let mismatches = btree_mismatches(&dataset, "after merge").await;
    report(&mismatches, "after merge");
}

/// The joined compaction, then the explicit segment merge of the committed A
/// and B, published in their place. Materialization is what the merged file
/// stores before publication; publication is what reads see after it.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn released_in_place_withdrawal_then_joined_compaction_then_explicit_merge_serves_each_row_once()
 {
    let state = Box::pin(released_state(Kind::BTree)).await;
    let mut dataset = state.dataset;
    let committed = raw_segments(&dataset, BTREE).await;
    assert_eq!(
        committed.len(),
        2,
        "fixture: A and B are committed: {committed:?}"
    );
    let merged = dataset
        .merge_existing_index_segments(committed)
        .await
        .expect("the explicit merge of the committed A and B succeeds");

    let destination = only_new(
        &[10, 11],
        &sorted_fragments(&dataset),
        "the compaction destination",
    );
    let mut current: Vec<(i32, u64)> = row_addrs_by_i(&dataset)
        .await
        .into_iter()
        .filter(|(_, addr)| {
            lance_core::utils::address::RowAddress::from(*addr).fragment_id() == destination
        })
        .collect();
    current.sort_unstable();
    assert_eq!(
        current.iter().map(|(value, _)| *value).collect::<Vec<_>>(),
        vec![2, 3, REWRITTEN_I[0], REWRITTEN_I[1]],
        "fixture: the destination holds the current rows of fragments 0 and 1"
    );
    let mut materialization = Vec::new();
    for old in [0, 1] {
        let stored = raw_segment_row_addrs(&dataset, &merged, old).await;
        if !stored.is_empty() {
            materialization.push(format!(
                "materialization: the merged file holds the withdrawn value {old} at {stored:?}"
            ));
        }
    }
    for (value, addr) in &current {
        let stored = raw_segment_row_addrs(&dataset, &merged, *value).await;
        if stored != vec![*addr] {
            materialization.push(format!(
                "materialization: the merged file stores i = {value} at {stored:?}, expected once at {addr}"
            ));
        }
    }

    dataset
        .commit_existing_index_segments(BTREE, "i", vec![merged.clone()])
        .await
        .expect("publication: the merged segment commits");
    let dataset = fresh_session(dataset.uri()).await;
    let published: Vec<Uuid> = raw_segments(&dataset, BTREE)
        .await
        .into_iter()
        .map(|segment| segment.uuid)
        .collect();
    let mut publication = Vec::new();
    if published != vec![merged.uuid] {
        publication.push(format!(
            "publication: the raw manifest holds {published:?} for {BTREE}, expected only the merged segment {}",
            merged.uuid
        ));
    }
    publication.extend(btree_mismatches(&dataset, "publication").await);
    assert!(
        materialization.is_empty() && publication.is_empty(),
        "materialization: {} mismatches:\n{}\npublication: {} mismatches:\n{}",
        materialization.len(),
        materialization.join("\n"),
        publication.len(),
        publication.join("\n")
    );
}

/// `(i, score)` of every hit of `terms` on `text`, sorted by `i`, and the plan.
async fn fts_hits(dataset: &Dataset, terms: &str) -> (Vec<(i32, f32)>, String) {
    let mut scan = dataset.scan();
    scan.full_text_search(
        FullTextSearchQuery::new(terms.to_string())
            .with_column("text".to_string())
            .unwrap()
            .limit(Some(100)),
    )
    .unwrap();
    scan.project(&["i"]).unwrap();
    let plan = scan.explain_plan(false).await.unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let ids = batch["i"].as_primitive::<Int32Type>();
    let scores = batch[SCORE_COL].as_primitive::<Float32Type>();
    let mut hits: Vec<(i32, f32)> = ids
        .values()
        .iter()
        .copied()
        .zip(scores.values().iter().copied())
        .collect();
    hits.sort_by_key(|(i, _)| *i);
    (hits, plan)
}

/// The sorted `i` of every live row whose text holds a word of `terms`.
pub(super) async fn scanned_matches(dataset: &Dataset, terms: &str) -> Vec<i32> {
    let words: Vec<&str> = terms.split_whitespace().collect();
    let batch = dataset
        .scan()
        .project(&["i", "text"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let ids = batch["i"].as_primitive::<Int32Type>();
    let texts = batch["text"].as_string::<i32>();
    let mut matches: Vec<i32> = ids
        .values()
        .iter()
        .zip(texts.iter())
        .filter(|(_, text)| {
            text.unwrap()
                .split_whitespace()
                .any(|word| words.contains(&word))
        })
        .map(|(i, _)| *i)
        .collect();
    matches.sort_unstable();
    matches
}

/// The joined compaction on an inverted index over `text`: every term through the index
/// equals the scan, no row twice, and the old-only term returns nothing.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn released_in_place_withdrawal_then_joined_compaction_serves_each_fts_row_once() {
    let state = Box::pin(released_state(Kind::Fts)).await;
    let dataset = &state.dataset;
    let mut mismatches = Vec::new();
    for query in FTS_TERMS.iter().chain([&OLD_ONLY_TERM]) {
        let (hits, plan) = fts_hits(dataset, query).await;
        let ids: Vec<i32> = hits.iter().map(|(i, _)| *i).collect();
        let scanned = scanned_matches(dataset, query).await;
        let mut distinct = ids.clone();
        distinct.dedup();
        if distinct != ids {
            mismatches.push(format!("{query}: a row is returned twice: {ids:?}"));
        }
        if !plan
            .lines()
            .any(|line| line.trim_start().starts_with("MatchQuery"))
        {
            mismatches.push(format!("{query}: the plan does not use the index"));
        }
        if ids != scanned {
            mismatches.push(format!(
                "{query}: index hits {ids:?} differ from the scan {scanned:?}"
            ));
        }
    }
    report(&mismatches, "FTS after compaction");
}
