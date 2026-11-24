// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Optimize merges of committed segments on a tagged history commit the live
//! coverage their file is written for. S {0} and T {1} both derive the
//! destinations {10, 11} of one transition: a merge that would take part of a
//! shared destination from a segment it leaves behind indexes only the
//! unindexed fragments instead, and a merge of every holder commits the
//! destinations themselves.
//!
//! Fixture steps are asserted on the raw manifest (`fixture:` messages); the
//! read checks are collected so one mismatch does not hide another.

use super::*;

use std::collections::{BTreeMap, HashMap};

use arrow_array::{ArrayRef, Int32Array, RecordBatchIterator, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use lance_core::utils::address::RowAddress;
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::scalar::InvertedIndexParams;
use lance_table::system_index::frag_reuse::metadata::is_tagged;
use uuid::Uuid;

use super::released_in_place::{scanned_matches, segment_answers};
use crate::dataset::overlay::overlaid_fragments;
use crate::index::tests::support::raw_manifest_indices;

const BTREE: &str = "i_idx";
const TEXT: &str = "text_idx";

/// Fragment 0, indexed by S.
const FIRST: [(i32, &str); 2] = [(0, "alpha graph"), (1, "beta")];
/// Fragment 1, indexed by T.
const SECOND: [(i32, &str); 2] = [(2, "gamma graph"), (3, "delta")];
/// Appended after the partition; no segment covers it.
const APPENDED: [(i32, &str); 2] = [(4, "omega graph"), (5, "alpha")];

const TERMS: [&str; 6] = ["alpha", "beta", "gamma", "delta", "omega", "graph"];

/// The destinations the stable partition of {0, 1} produces.
const DESTINATIONS: [u32; 2] = [10, 11];

/// The partition puts the even `i` of {0, 1} in 10, so its offset 1 holds
/// `i = 2`, "gamma graph": the row [`overlay_destination`] rewrites.
const OVERLAID_OFFSET: u32 = 1;
/// The overlaid row's `i`, obsolete then replacement (BTree).
const OVERLAID_I: [i32; 2] = [2, 102];
/// The overlaid row's text term, obsolete then replacement (FTS).
const OVERLAID_TERMS: [&str; 2] = ["gamma", "zeta"];
const REPLACEMENT_TEXT: &str = "zeta graph";
/// [`TERMS`] and the replacement term.
const OVERLAID_TERMS_CHECKED: [&str; 7] =
    ["alpha", "beta", "gamma", "delta", "omega", "graph", "zeta"];

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
                        &InvertedIndexParams::default()
                            .lower_case(true)
                            .stem(false)
                            .remove_stop_words(false),
                        false,
                    )
                    .await
            }
        };
        result.expect("fixture: the index builds");
    }
}

fn sorted_fragments(dataset: &Dataset) -> Vec<u32> {
    let mut ids = fragment_ids(dataset);
    ids.sort_unstable();
    ids
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

fn expected_bitmaps(expected: &[(Uuid, &[u32])]) -> BTreeMap<Uuid, Vec<u32>> {
    expected
        .iter()
        .map(|(uuid, bitmap)| (*uuid, bitmap.to_vec()))
        .collect()
}

async fn assert_stored(dataset: &Dataset, name: &str, expected: &[(Uuid, &[u32])], step: &str) {
    assert_eq!(
        raw_bitmaps(dataset, name).await,
        expected_bitmaps(expected),
        "fixture: {step}: stored bitmaps of {name}"
    );
}

/// The one segment of `name` the manifest lists now and did not list before.
async fn only_new_segment(dataset: &Dataset, name: &str, before: &[Uuid]) -> IndexMetadata {
    let new: Vec<IndexMetadata> = raw_segments(dataset, name)
        .await
        .into_iter()
        .filter(|idx| !before.contains(&idx.uuid))
        .collect();
    assert_eq!(new.len(), 1, "exactly one new segment of {name}: {new:?}");
    new.into_iter().next().unwrap()
}

/// Whether `now` is `before` as committed: same uuid, bitmap, files and stamp.
fn unchanged(before: &IndexMetadata, now: &IndexMetadata) -> bool {
    before.uuid == now.uuid
        && before.fragment_bitmap == now.fragment_bitmap
        && format!("{:?}", before.files) == format!("{:?}", now.files)
        && before.dataset_version == now.dataset_version
}

async fn assert_unchanged(dataset: &Dataset, name: &str, before: &IndexMetadata, label: &str) {
    let now = raw_segments(dataset, name)
        .await
        .into_iter()
        .find(|idx| idx.uuid == before.uuid)
        .unwrap_or_else(|| panic!("{label}: segment {} is still committed", before.uuid));
    assert!(
        unchanged(before, &now),
        "{label}: segment {} is untouched: {before:?} -> {now:?}",
        before.uuid
    );
}

async fn raw_fri(dataset: &Dataset) -> Option<IndexMetadata> {
    raw_manifest_indices(dataset)
        .await
        .into_iter()
        .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
}

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

/// The coverage the tagged reader derives for each segment of `name`.
async fn derived_bitmaps(dataset: &Dataset, name: &str) -> BTreeMap<Uuid, Vec<u32>> {
    dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .filter(|idx| idx.name == name)
        .map(|idx| {
            let bitmap = idx
                .fragment_bitmap
                .as_ref()
                .map(|bitmap| bitmap.iter().collect())
                .unwrap_or_default();
            (idx.uuid, bitmap)
        })
        .collect()
}

struct Joined {
    _dir: TempStrDir,
    uri: String,
    dataset: Dataset,
    s: IndexMetadata,
    t: IndexMetadata,
    /// The direct sibling over the destinations asked for, if any.
    d: Option<IndexMetadata>,
    /// The appended fragment no segment covers.
    appended: u32,
}

/// S over {0} (built), T over {1} (optimize append); then, when `tagged`,
/// {0, 1} stably partitioned into {10, 11} and, when `sibling` is not
/// empty, a BTree segment D built over those destinations alone; finally one
/// fragment appended.
async fn joined_state(kind: Kind, tagged: bool, sibling: &[u32]) -> Joined {
    assert!(
        tagged || sibling.is_empty(),
        "fixture: a sibling needs the partition"
    );
    assert!(
        kind == Kind::BTree || sibling.is_empty(),
        "fixture: the sibling is a BTree segment"
    );
    let dir = TempStrDir::default();
    let uri = dir.as_str().to_string();
    let name = kind.name();

    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(rows_batch(&FIRST))], table_schema()),
        &uri,
        None,
    )
    .await
    .unwrap();
    assert!(
        !dataset.manifest.uses_stable_row_ids(),
        "fixture: the table is written without stable row ids"
    );
    assert_eq!(sorted_fragments(&dataset), vec![0], "fixture: step 1");
    kind.create(&mut dataset).await;
    let built = raw_segments(&dataset, name).await;
    assert_eq!(built.len(), 1, "fixture: step 1: S is the only segment");
    let s = built[0].clone();
    assert_stored(&dataset, name, &[(s.uuid, &[0][..])], "step 1: S {0}").await;

    let mut dataset = append_rows(dataset, &SECOND).await;
    assert_eq!(sorted_fragments(&dataset), vec![0, 1], "fixture: step 2");
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .expect("fixture: optimize indexes fragment 1");
    let mut dataset = fresh_session(&uri).await;
    let t = only_new_segment(&dataset, name, &[s.uuid]).await;
    let pair: [(Uuid, &[u32]); 2] = [(s.uuid, &[0]), (t.uuid, &[1])];
    assert_stored(&dataset, name, &pair, "step 2: S {0}, T {1}").await;

    let mut d = None;
    if tagged {
        reserve_fragments(&mut dataset, 20).await;
        commit_stable_partition(dataset, &[0, 1], DESTINATIONS[0] as u64).await;
        dataset = fresh_session(&uri).await;
        assert_eq!(
            sorted_fragments(&dataset),
            DESTINATIONS.to_vec(),
            "fixture: step 3: {{0, 1}} -> {{10, 11}}"
        );
        assert!(
            raw_fri(&dataset).await.as_ref().is_some_and(is_tagged),
            "fixture: step 3: the history is tagged"
        );
        assert_eq!(
            ledger_transitions(&dataset).await,
            vec![(vec![0, 1], vec![10, 11])],
            "fixture: step 3: the history holds the joined transition"
        );
        assert_stored(&dataset, name, &pair, "step 3: the partition leaves S, T").await;

        if !sibling.is_empty() {
            let params = ScalarIndexParams::default();
            let mut direct =
                CreateIndexBuilder::new(&mut dataset, &["i"], IndexType::BTree, &params)
                    .name(format!("{BTREE}_sibling"))
                    .fragments(sibling.to_vec())
                    .execute_uncommitted()
                    .await
                    .expect("fixture: D builds over its destinations");
            direct.name = BTREE.into();
            dataset
                .apply_commit(
                    Transaction::new(
                        dataset.manifest.version,
                        Operation::CreateIndex {
                            new_indices: vec![direct.clone()],
                            removed_indices: vec![],
                        },
                        None,
                    ),
                    &Default::default(),
                    &Default::default(),
                )
                .await
                .expect("fixture: D commits");
            dataset = fresh_session(&uri).await;
            assert_stored(
                &dataset,
                name,
                &[(s.uuid, &[0]), (t.uuid, &[1]), (direct.uuid, sibling)],
                "step 4: D beside S, T",
            )
            .await;
            d = Some(
                raw_segments(&dataset, name)
                    .await
                    .into_iter()
                    .find(|idx| idx.uuid == direct.uuid)
                    .unwrap(),
            );
        }

        let mut expected = BTreeMap::new();
        let shared: Vec<u32> = DESTINATIONS
            .into_iter()
            .filter(|id| !sibling.contains(id))
            .collect();
        // A segment the reader derives nothing for is not listed.
        if !shared.is_empty() {
            expected.insert(s.uuid, shared.clone());
            expected.insert(t.uuid, shared);
        }
        if let Some(d) = &d {
            expected.insert(d.uuid, sibling.to_vec());
        }
        assert_eq!(
            derived_bitmaps(&dataset, name).await,
            expected,
            "fixture: S and T derive the destinations they share"
        );
    }

    let live = sorted_fragments(&dataset);
    let before: Vec<(Uuid, Vec<u32>)> = raw_bitmaps(&dataset, name).await.into_iter().collect();
    append_rows(dataset, &APPENDED).await;
    let dataset = fresh_session(&uri).await;
    let added: Vec<u32> = sorted_fragments(&dataset)
        .into_iter()
        .filter(|id| !live.contains(id))
        .collect();
    assert_eq!(added.len(), 1, "fixture: step 5: one fragment appended");
    assert_eq!(
        raw_bitmaps(&dataset, name)
            .await
            .into_iter()
            .collect::<Vec<_>>(),
        before,
        "fixture: step 5: the append leaves every segment"
    );
    let s = raw_segments(&dataset, name)
        .await
        .into_iter()
        .find(|idx| idx.uuid == s.uuid)
        .unwrap();
    let t = raw_segments(&dataset, name)
        .await
        .into_iter()
        .find(|idx| idx.uuid == t.uuid)
        .unwrap();
    Joined {
        _dir: dir,
        uri,
        dataset,
        s,
        t,
        d,
        appended: added[0],
    }
}

fn report(mismatches: &[String], label: &str) {
    assert!(
        mismatches.is_empty(),
        "{label}: {} mismatches:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// Every BTree point query of a live value, every range and the full scan
/// against the scan.
async fn btree_mismatches(dataset: &Dataset, label: &str) -> Vec<String> {
    let live: Vec<i32> = FIRST
        .iter()
        .chain(&SECOND)
        .chain(&APPENDED)
        .map(|(i, _)| *i)
        .collect();
    btree_mismatches_over(dataset, label, &live).await
}

/// [`btree_mismatches`] with the point queries over the given live values.
async fn btree_mismatches_over(dataset: &Dataset, label: &str, live: &[i32]) -> Vec<String> {
    let mut mismatches = Vec::new();
    for value in live {
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
            mismatches.push(format!(
                "{label}: {predicate}: the plan does not use the index"
            ));
        }
        let indexed = values(dataset, Some(&predicate), true).await;
        let scanned = values(dataset, Some(&predicate), false).await;
        if indexed != scanned {
            mismatches.push(format!(
                "{label}: {predicate}: index {indexed:?} != scan {scanned:?}"
            ));
        }
        if scanned != vec![*value] {
            mismatches.push(format!("{label}: fixture: {predicate}: scan {scanned:?}"));
        }
    }
    for predicate in ["i >= 0 AND i < 4", "i >= 2", "i < 1000"] {
        let indexed = values(dataset, Some(predicate), true).await;
        let scanned = values(dataset, Some(predicate), false).await;
        if indexed != scanned {
            mismatches.push(format!(
                "{label}: {predicate}: index {indexed:?} != scan {scanned:?}"
            ));
        }
    }
    let (indexed, scanned) = (
        values(dataset, None, true).await,
        values(dataset, None, false).await,
    );
    if indexed != scanned {
        mismatches.push(format!(
            "{label}: full scan: index {indexed:?} != scan {scanned:?}"
        ));
    }
    mismatches
}

/// Which segment answers each live row through the query path, against
/// `expected(value, live fragment)`; a row answered twice, by the wrong
/// segment, or an answered address that is not live is a mismatch.
async fn answer_mismatches(
    dataset: &Dataset,
    label: &str,
    expected: impl Fn(i32, u32) -> Uuid,
) -> Vec<String> {
    let mut mismatches = Vec::new();
    let live: HashMap<u64, i32> = row_addrs_by_i(dataset)
        .await
        .into_iter()
        .map(|(value, addr)| (addr, value))
        .collect();
    let mut served: BTreeMap<u64, Vec<Uuid>> = BTreeMap::new();
    for (uuid, answer) in segment_answers(dataset).await {
        match answer {
            Err(err) => mismatches.push(format!("{label}: segment {uuid} has no answer: {err}")),
            Ok(addrs) => {
                for addr in addrs {
                    served.entry(addr).or_default().push(uuid);
                }
            }
        }
    }
    let mut by_value: Vec<(i32, u64)> = live.iter().map(|(addr, value)| (*value, *addr)).collect();
    by_value.sort_unstable();
    for (value, addr) in by_value {
        let fragment = lance_core::utils::address::RowAddress::from(addr).fragment_id();
        let want = vec![expected(value, fragment)];
        let got = served.get(&addr).cloned().unwrap_or_default();
        if got != want {
            mismatches.push(format!(
                "{label}: the row i = {value} (fragment {fragment}) is answered by {got:?}, \
                 expected {want:?}"
            ));
        }
    }
    for (addr, segments) in &served {
        if !live.contains_key(addr) {
            mismatches.push(format!(
                "{label}: segments {segments:?} answer row address {addr}, which is not live"
            ));
        }
    }
    mismatches
}

/// Every term through the full-text index against the scan, no row twice.
async fn fts_mismatches(dataset: &Dataset, label: &str) -> Vec<String> {
    fts_mismatches_over(dataset, label, &TERMS).await
}

/// [`fts_mismatches`] over the given terms.
async fn fts_mismatches_over(dataset: &Dataset, label: &str, terms: &[&str]) -> Vec<String> {
    let mut mismatches = Vec::new();
    for &term in terms {
        let (ids, plan) = fts_ids(dataset, "text", term).await;
        let scanned = scanned_matches(dataset, term).await;
        let mut distinct = ids.clone();
        distinct.dedup();
        if distinct != ids {
            mismatches.push(format!("{label}: {term}: a row is returned twice: {ids:?}"));
        }
        if !plan
            .lines()
            .any(|line| line.trim_start().starts_with("MatchQuery"))
        {
            mismatches.push(format!("{label}: {term}: the plan does not use the index"));
        }
        if ids != scanned {
            mismatches.push(format!(
                "{label}: {term}: index hits {ids:?} differ from the scan {scanned:?}"
            ));
        }
    }
    mismatches
}

/// The read checks of `kind` against the scan; for a BTree also which segment
/// answers each live row, per `expected(value, live fragment)`.
async fn read_mismatches(
    kind: Kind,
    dataset: &Dataset,
    label: &str,
    expected: impl Fn(i32, u32) -> Uuid,
) -> Vec<String> {
    match kind {
        Kind::BTree => {
            let mut mismatches = btree_mismatches(dataset, label).await;
            mismatches.extend(answer_mismatches(dataset, label, expected).await);
            mismatches
        }
        Kind::Fts => fts_mismatches(dataset, label).await,
    }
}

/// [`read_mismatches`] on the table [`overlay_destination`] rewrote one row
/// of, plus the overlaid row's obsolete and replacement value through the
/// index; `expected` as for [`answer_mismatches`], or `None` while a fragment
/// is unindexed.
async fn overlaid_read_mismatches(
    kind: Kind,
    dataset: &Dataset,
    label: &str,
    expected: Option<&dyn Fn(i32, u32) -> Uuid>,
) -> Vec<String> {
    let mut mismatches = match kind {
        Kind::BTree => btree_mismatches_over(dataset, label, &overlaid_values(kind)).await,
        Kind::Fts => fts_mismatches_over(dataset, label, &OVERLAID_TERMS_CHECKED).await,
    };
    if let (Kind::BTree, Some(expected)) = (kind, expected) {
        mismatches.extend(answer_mismatches(dataset, label, expected).await);
    }
    mismatches.extend(overlaid_row_mismatches(kind, dataset, label).await);
    mismatches
}

/// The overlaid row's obsolete value (no hit) and replacement value (exactly
/// that row) through the index, against the scan.
async fn overlaid_row_mismatches(kind: Kind, dataset: &Dataset, label: &str) -> Vec<String> {
    let mut mismatches = Vec::new();
    match kind {
        Kind::BTree => {
            let [obsolete, replacement] = OVERLAID_I;
            for (value, want) in [(obsolete, vec![]), (replacement, vec![replacement])] {
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
                    mismatches.push(format!(
                        "{label}: overlaid {predicate}: the plan does not use the index"
                    ));
                }
                let indexed = values(dataset, Some(&predicate), true).await;
                let scanned = values(dataset, Some(&predicate), false).await;
                if indexed != want || scanned != want {
                    mismatches.push(format!(
                        "{label}: overlaid {predicate}: index {indexed:?}, scan {scanned:?}, \
                         expected {want:?}"
                    ));
                }
            }
        }
        Kind::Fts => {
            let [obsolete, replacement] = OVERLAID_TERMS;
            for (term, want) in [(obsolete, vec![]), (replacement, vec![OVERLAID_I[0]])] {
                let (ids, plan) = fts_ids(dataset, "text", term).await;
                let scanned = scanned_matches(dataset, term).await;
                if !plan
                    .lines()
                    .any(|line| line.trim_start().starts_with("MatchQuery"))
                {
                    mismatches.push(format!(
                        "{label}: overlaid {term}: the plan does not use the index"
                    ));
                }
                if ids != want || scanned != want {
                    mismatches.push(format!(
                        "{label}: overlaid {term}: index hits {ids:?}, scan {scanned:?}, \
                         expected {want:?}"
                    ));
                }
            }
        }
    }
    mismatches
}

/// Runs the fragment reuse trim and reopens the table in a fresh session.
async fn trim_and_reopen(dataset: Dataset, uri: &str) -> Dataset {
    let mut dataset = dataset;
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    fresh_session(uri).await
}

fn source_of(value: i32, s: Uuid, t: Uuid) -> Uuid {
    if FIRST.iter().any(|(i, _)| *i == value) {
        s
    } else {
        t
    }
}

/// The live `i` values once [`overlay_destination`] has rewritten its row.
fn overlaid_values(kind: Kind) -> Vec<i32> {
    FIRST
        .iter()
        .chain(&SECOND)
        .chain(&APPENDED)
        .map(|(i, _)| match kind {
            Kind::BTree if *i == OVERLAID_I[0] => OVERLAID_I[1],
            _ => *i,
        })
        .collect()
}

/// [`source_of`] for a row [`overlay_destination`] may have rewritten.
fn source_before_overlay(value: i32, s: Uuid, t: Uuid) -> Uuid {
    let value = if value == OVERLAID_I[1] {
        OVERLAID_I[0]
    } else {
        value
    };
    source_of(value, s, t)
}

/// Rewrites the indexed value of the row at [`OVERLAID_OFFSET`] of
/// destination 10 with a data overlay committed after S and T were built,
/// and returns the reopened table with the overlay's commit version. An
/// overlay commit on a tagged table is refused by the gate, so the staged
/// overlay is persisted as a manifest fixture, as the tagged overlay
/// masking tests do.
async fn overlay_destination(
    dataset: Dataset,
    uri: &str,
    kind: Kind,
    sources: [&IndexMetadata; 2],
) -> (Dataset, u64) {
    let destination = DESTINATIONS[0];
    let address = u64::from(RowAddress::new_from_parts(destination, OVERLAID_OFFSET));
    assert_eq!(
        row_addrs_by_i(&dataset).await.get(&OVERLAID_I[0]),
        Some(&address),
        "fixture: i = {} is offset {OVERLAID_OFFSET} of {destination}",
        OVERLAID_I[0]
    );
    assert_eq!(
        scanned_matches(&dataset, OVERLAID_TERMS[0]).await,
        vec![OVERLAID_I[0]],
        "fixture: only the overlaid row holds the obsolete term"
    );
    assert!(
        overlaid_fragments(&dataset.manifest.fragments).is_empty(),
        "fixture: no fragment carries an overlay yet"
    );

    let (column, data_type, values): (&str, DataType, ArrayRef) = match kind {
        Kind::BTree => (
            "i",
            DataType::Int32,
            Arc::new(Int32Array::from(vec![OVERLAID_I[1]])),
        ),
        Kind::Fts => (
            "text",
            DataType::Utf8,
            Arc::new(StringArray::from(vec![REPLACEMENT_TEXT])),
        ),
    };
    let field_id = dataset.schema().field(column).unwrap().id;
    let fragment = dataset.get_fragment(destination as usize).unwrap();
    let mut writer = fragment
        .write_overlay(&dataset.schema().project_by_ids(&[field_id], true))
        .await
        .unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new(lance_core::ROW_ADDR, DataType::UInt64, true),
            Field::new(column, data_type, true),
        ])),
        vec![Arc::new(UInt64Array::from(vec![address])), values],
    )
    .unwrap();
    writer.write_batch(&batch).await.unwrap();
    let mut group = writer
        .finish()
        .await
        .unwrap()
        .expect("fixture: the overlay stages a file");
    assert_eq!(
        group.fragment_id, destination as u64,
        "fixture: overlay target"
    );
    let version = dataset.manifest.version + 1;
    for overlay in &mut group.overlays {
        overlay.committed_version = version;
    }

    let fragments_before = dataset.manifest.fragments.as_ref().clone();
    let indices = raw_manifest_indices(&dataset).await;
    let mut dataset = dataset;
    let manifest = Arc::make_mut(&mut dataset.manifest);
    Arc::make_mut(&mut manifest.fragments)
        .iter_mut()
        .find(|f| f.id == destination as u64)
        .unwrap()
        .overlays = group.overlays;
    reader_tests::persist_fixture(&mut dataset, indices).await;
    let dataset = fresh_session(uri).await;

    assert_eq!(
        dataset.manifest.version, version,
        "fixture: the overlay is the latest version"
    );
    let overlaid = overlaid_fragments(&dataset.manifest.fragments);
    assert_eq!(
        overlaid.keys().copied().collect::<Vec<_>>(),
        vec![destination],
        "fixture: only destination {destination} carries an overlay"
    );
    assert_eq!(
        overlaid[&destination]
            .overlays
            .iter()
            .map(|overlay| overlay.committed_version)
            .collect::<Vec<_>>(),
        vec![version],
        "fixture: the overlay is committed at the new version"
    );
    for source in sources {
        assert!(
            source.dataset_version < version,
            "fixture: segment {} (stamp {}) predates the overlay (version {version})",
            source.uuid,
            source.dataset_version
        );
    }
    let mut fragments_now = dataset.manifest.fragments.as_ref().clone();
    for fragment in &mut fragments_now {
        fragment.overlays.clear();
    }
    assert_eq!(
        fragments_now, fragments_before,
        "fixture: the overlay rewrites no fragment: ids, files and rows unchanged"
    );
    assert_stored(
        &dataset,
        kind.name(),
        &[(sources[0].uuid, &[0]), (sources[1].uuid, &[1])],
        "the overlay leaves S, T",
    )
    .await;
    for source in sources {
        assert_unchanged(&dataset, kind.name(), source, "fixture: across the overlay").await;
    }
    assert_eq!(
        ledger_transitions(&dataset).await,
        vec![(vec![0, 1], vec![10, 11])],
        "fixture: the overlay leaves the history"
    );

    // The scan reads the overlaid value at the same address.
    let addrs = row_addrs_by_i(&dataset).await;
    let [obsolete, replacement] = OVERLAID_TERMS;
    let (live_i, gone_i, live_term, gone_term) = match kind {
        Kind::BTree => (OVERLAID_I[1], Some(OVERLAID_I[0]), obsolete, replacement),
        Kind::Fts => (OVERLAID_I[0], None, replacement, obsolete),
    };
    assert_eq!(
        addrs.get(&live_i),
        Some(&address),
        "fixture: the scan reads i = {live_i} at the overlaid address"
    );
    if let Some(gone_i) = gone_i {
        assert_eq!(
            addrs.get(&gone_i),
            None,
            "fixture: no row holds i = {gone_i}"
        );
    }
    assert_eq!(
        scanned_matches(&dataset, live_term).await,
        vec![live_i],
        "fixture: the scan of {live_term} after the overlay"
    );
    assert_eq!(
        scanned_matches(&dataset, gone_term).await,
        Vec::<i32>::new(),
        "fixture: the scan of {gone_term} after the overlay"
    );
    (dataset, version)
}

/// (T1) The default optimize selects only T, which shares both destinations
/// with S: the merge is declined, S and T stay as committed, and a new
/// segment indexes the appended fragment alone.
#[rstest::rstest]
#[case::btree(Kind::BTree)]
#[case::fts(Kind::Fts)]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn partial_merge_of_a_shared_destination_indexes_only_new_data(#[case] kind: Kind) {
    let state = Box::pin(joined_state(kind, true, &[])).await;
    let name = kind.name();
    let mut dataset = state.dataset;
    let read_version = dataset.manifest.version;
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let dataset = fresh_session(&state.uri).await;

    let n = only_new_segment(&dataset, name, &[state.s.uuid, state.t.uuid]).await;
    assert_eq!(
        raw_bitmaps(&dataset, name).await,
        expected_bitmaps(&[
            (state.s.uuid, &[0]),
            (state.t.uuid, &[1]),
            (n.uuid, &[state.appended]),
        ]),
        "S and T stay; the new segment covers the appended fragment"
    );
    assert_unchanged(&dataset, name, &state.s, "S").await;
    assert_unchanged(&dataset, name, &state.t, "T").await;

    let label = "after the declined merge";
    let mut mismatches = match kind {
        Kind::BTree => btree_mismatches(&dataset, label).await,
        Kind::Fts => fts_mismatches(&dataset, label).await,
    };
    if kind == Kind::BTree {
        let (s, t, appended) = (state.s.uuid, state.t.uuid, state.appended);
        mismatches.extend(
            answer_mismatches(&dataset, label, |value, fragment| {
                if fragment == appended {
                    n.uuid
                } else {
                    source_of(value, s, t)
                }
            })
            .await,
        );
    }
    report(&mismatches, label);

    // The delta reads no old segment, so it is stamped with the manifest the
    // optimize read.
    assert_eq!(
        n.dataset_version, read_version,
        "the new segment is stamped with the manifest the optimize read"
    );

    // S and T still name the transition's sources, so the trim keeps it and
    // every segment stays as committed.
    let dataset = trim_and_reopen(dataset, &state.uri).await;
    assert_eq!(
        ledger_transitions(&dataset).await,
        vec![(vec![0, 1], vec![10, 11])],
        "the trim keeps the transition S and T still need"
    );
    for (before, segment) in [(&state.s, "S"), (&state.t, "T"), (&n, "N")] {
        assert_unchanged(
            &dataset,
            name,
            before,
            &format!("{segment} across the trim"),
        )
        .await;
    }
    assert_eq!(
        raw_segments(&dataset, name).await.len(),
        3,
        "the trim leaves S, T and the new segment"
    );

    let label = "after the trim beside the declined merge";
    let (s, t, appended) = (state.s.uuid, state.t.uuid, state.appended);
    let mismatches = read_mismatches(kind, &dataset, label, |value, fragment| {
        if fragment == appended {
            n.uuid
        } else {
            source_of(value, s, t)
        }
    })
    .await;
    report(&mismatches, label);
}

/// (T2) A merge of both holders of the shared destinations commits exactly
/// the live fragments its file is written for: the destinations and the
/// appended fragment.
#[rstest::rstest]
#[case::btree(Kind::BTree)]
#[case::fts(Kind::Fts)]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn merge_of_every_holder_commits_the_live_coverage(#[case] kind: Kind) {
    let state = Box::pin(joined_state(kind, true, &[])).await;
    let name = kind.name();
    let mut dataset = state.dataset;
    let read_version = dataset.manifest.version;
    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    let dataset = fresh_session(&state.uri).await;

    let m = only_new_segment(&dataset, name, &[state.s.uuid, state.t.uuid]).await;
    assert_eq!(
        raw_bitmaps(&dataset, name).await,
        expected_bitmaps(&[(m.uuid, &[DESTINATIONS[0], DESTINATIONS[1], state.appended])]),
        "one merged segment over the live coverage of S, T and the appended fragment"
    );

    let label = "after the merge";
    let mut mismatches = match kind {
        Kind::BTree => btree_mismatches(&dataset, label).await,
        Kind::Fts => fts_mismatches(&dataset, label).await,
    };
    if kind == Kind::BTree {
        let claimed = [DESTINATIONS[0], DESTINATIONS[1], state.appended];
        for (value, addr) in row_addrs_by_i(&dataset).await {
            let stored = raw_segment_row_addrs(&dataset, &m, value).await;
            if stored != vec![addr] {
                mismatches.push(format!(
                    "{label}: M's file stores {stored:?} for i = {value}, the live address is {addr}"
                ));
            }
            for addr in stored {
                let fragment = lance_core::utils::address::RowAddress::from(addr).fragment_id();
                if !claimed.contains(&fragment) {
                    mismatches.push(format!(
                        "{label}: M's file addresses fragment {fragment}, outside its bitmap"
                    ));
                }
            }
        }
        mismatches.extend(answer_mismatches(&dataset, label, |_, _| m.uuid).await);
    }
    report(&mismatches, label);

    // What the merge stamps today: a remapping (BTree) merge the manifest it
    // read, an FTS merge the oldest selected source's stamp.
    let expected_stamp = match kind {
        Kind::BTree => read_version,
        Kind::Fts => state.s.dataset_version.min(state.t.dataset_version),
    };
    assert_eq!(
        m.dataset_version, expected_stamp,
        "M's stamp (read {read_version}, S {}, T {})",
        state.s.dataset_version, state.t.dataset_version
    );

    // No segment names the transition's sources any more, so the trim
    // releases it; M must keep answering, as committed, without it.
    let dataset = trim_and_reopen(dataset, &state.uri).await;
    assert!(
        ledger_transitions(&dataset).await.is_empty(),
        "the trim releases the transition every holder was merged out of: {:?}",
        raw_fri(&dataset).await
    );
    assert_unchanged(&dataset, name, &m, "M across the trim").await;
    assert_eq!(
        raw_segments(&dataset, name).await.len(),
        1,
        "M is still the only segment"
    );

    let label = "after the trim";
    let mismatches = read_mismatches(kind, &dataset, label, |_, _| m.uuid).await;
    report(&mismatches, label);
}

/// (T3) With D directly covering 10, S and T share only 11. Selection takes
/// the trailing segments, so `merge(2)` selects T and D, never S and T
/// without D: T's share of 11 would be taken from S, so the merge is
/// declined and only the appended fragment is indexed.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn merge_window_splitting_a_shared_destination_is_declined_beside_a_direct_sibling() {
    let state = Box::pin(joined_state(Kind::BTree, true, &[DESTINATIONS[0]])).await;
    let d = state.d.clone().unwrap();
    let mut dataset = state.dataset;
    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    let dataset = fresh_session(&state.uri).await;

    let n = only_new_segment(&dataset, BTREE, &[state.s.uuid, state.t.uuid, d.uuid]).await;
    assert_eq!(
        raw_bitmaps(&dataset, BTREE).await,
        expected_bitmaps(&[
            (state.s.uuid, &[0]),
            (state.t.uuid, &[1]),
            (d.uuid, &[DESTINATIONS[0]]),
            (n.uuid, &[state.appended]),
        ]),
        "S, T and D stay; the new segment covers the appended fragment"
    );
    for (before, label) in [(&state.s, "S"), (&state.t, "T"), (&d, "D")] {
        assert_unchanged(&dataset, BTREE, before, label).await;
    }

    let label = "after the declined merge beside D";
    let mut mismatches = btree_mismatches(&dataset, label).await;
    let (s, t, appended) = (state.s.uuid, state.t.uuid, state.appended);
    mismatches.extend(
        answer_mismatches(&dataset, label, |value, fragment| {
            if fragment == appended {
                n.uuid
            } else if fragment == DESTINATIONS[0] {
                d.uuid
            } else {
                source_of(value, s, t)
            }
        })
        .await,
    );
    report(&mismatches, label);
}

/// (T3') The default optimize selects D alone: no other segment derives 10,
/// so D merges with the appended fragment, and S and T keep sharing 11.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn merge_of_a_direct_sibling_leaves_the_shared_destination_alone() {
    let state = Box::pin(joined_state(Kind::BTree, true, &[DESTINATIONS[0]])).await;
    let d = state.d.clone().unwrap();
    let mut dataset = state.dataset;
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let dataset = fresh_session(&state.uri).await;

    let m = only_new_segment(&dataset, BTREE, &[state.s.uuid, state.t.uuid, d.uuid]).await;
    assert_eq!(
        raw_bitmaps(&dataset, BTREE).await,
        expected_bitmaps(&[
            (state.s.uuid, &[0]),
            (state.t.uuid, &[1]),
            (m.uuid, &[DESTINATIONS[0], state.appended]),
        ]),
        "D merges with the appended fragment; S and T stay"
    );
    assert_unchanged(&dataset, BTREE, &state.s, "S").await;
    assert_unchanged(&dataset, BTREE, &state.t, "T").await;

    let label = "after merging D";
    let mut mismatches = btree_mismatches(&dataset, label).await;
    let (s, t, appended) = (state.s.uuid, state.t.uuid, state.appended);
    mismatches.extend(
        answer_mismatches(&dataset, label, |value, fragment| {
            if fragment == appended || fragment == DESTINATIONS[0] {
                m.uuid
            } else {
                source_of(value, s, t)
            }
        })
        .await,
    );
    report(&mismatches, label);
}

/// (T4) Without a fragment reuse history the default optimize merges T with
/// the appended fragment, as before, and leaves S.
#[tokio::test]
async fn untagged_merge_commits_the_merged_fragments() {
    let state = Box::pin(joined_state(Kind::BTree, false, &[])).await;
    let mut dataset = state.dataset;
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let dataset = fresh_session(&state.uri).await;

    let m = only_new_segment(&dataset, BTREE, &[state.s.uuid, state.t.uuid]).await;
    assert_eq!(
        raw_bitmaps(&dataset, BTREE).await,
        expected_bitmaps(&[(state.s.uuid, &[0]), (m.uuid, &[1, state.appended])]),
        "T merges with the appended fragment; S stays"
    );
    assert_unchanged(&dataset, BTREE, &state.s, "S").await;

    let label = "after the untagged merge";
    let mut mismatches = btree_mismatches(&dataset, label).await;
    let s = state.s.uuid;
    mismatches.extend(
        answer_mismatches(
            &dataset,
            label,
            |_, fragment| {
                if fragment == 0 { s } else { m.uuid }
            },
        )
        .await,
    );
    report(&mismatches, label);
}

/// The one committed segment of [`BTREE`] after a merge that replaced every
/// other, checked against the live fragments and the scan.
async fn assert_sole_merged_segment(dataset: &Dataset, before: &[Uuid], label: &str) {
    let m = only_new_segment(dataset, BTREE, before).await;
    let live = sorted_fragments(dataset);
    assert_eq!(
        raw_bitmaps(dataset, BTREE).await,
        expected_bitmaps(&[(m.uuid, live.as_slice())]),
        "{label}: one merged segment over the live fragments; {before:?} are replaced"
    );
    let mut mismatches = btree_mismatches(dataset, label).await;
    mismatches.extend(answer_mismatches(dataset, label, |_, _| m.uuid).await);
    report(&mismatches, label);
}

/// (T5) With D directly covering 10 and 11, S and T derive nothing: a merge
/// window over all three reads D alone and replaces S and T.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn optimize_merges_away_a_superseded_segment() {
    let state = Box::pin(joined_state(Kind::BTree, true, &DESTINATIONS)).await;
    let d = state.d.clone().unwrap();
    let mut dataset = state.dataset;
    dataset
        .optimize_indices(&OptimizeOptions::merge(3))
        .await
        .unwrap();
    let dataset = fresh_session(&state.uri).await;
    assert_eq!(
        sorted_fragments(&dataset),
        vec![DESTINATIONS[0], DESTINATIONS[1], state.appended],
        "fixture: the merge leaves the live fragments"
    );
    assert_sole_merged_segment(
        &dataset,
        &[state.s.uuid, state.t.uuid, d.uuid],
        "after merging away the superseded segments",
    )
    .await;
}

async fn assert_overlay_kept(dataset: &Dataset, overlay_version: u64, label: &str) {
    let overlaid = overlaid_fragments(&dataset.manifest.fragments);
    let kept: Vec<(u32, Vec<u64>)> = overlaid
        .iter()
        .map(|(id, fragment)| {
            let versions = fragment
                .overlays
                .iter()
                .map(|overlay| overlay.committed_version)
                .collect();
            (*id, versions)
        })
        .collect();
    assert_eq!(
        kept,
        vec![(DESTINATIONS[0], vec![overlay_version])],
        "{label}: the overlay stays on destination {}",
        DESTINATIONS[0]
    );
}

/// (T6) An overlay committed on destination 10 after S and T were built
/// rewrites the indexed value of one row. A merge of every holder still
/// commits the live coverage, and the overlaid row answers only its
/// replacement value before the merge, after it and across the trim.
#[rstest::rstest]
#[case::btree(Kind::BTree)]
#[case::fts(Kind::Fts)]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn overlay_before_full_merge_is_honored(#[case] kind: Kind) {
    let state = Box::pin(joined_state(kind, true, &[])).await;
    let name = kind.name();
    let (mut dataset, overlay_version) = Box::pin(overlay_destination(
        state.dataset,
        &state.uri,
        kind,
        [&state.s, &state.t],
    ))
    .await;
    let label = "before the merge";
    report(
        &overlaid_read_mismatches(kind, &dataset, label, None).await,
        label,
    );

    let read_version = dataset.manifest.version;
    assert_eq!(
        read_version, overlay_version,
        "fixture: the optimize reads the overlay's version"
    );
    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    let dataset = fresh_session(&state.uri).await;
    assert_overlay_kept(&dataset, overlay_version, "fixture: after the merge").await;

    let m = only_new_segment(&dataset, name, &[state.s.uuid, state.t.uuid]).await;
    assert_eq!(
        raw_bitmaps(&dataset, name).await,
        expected_bitmaps(&[(m.uuid, &[DESTINATIONS[0], DESTINATIONS[1], state.appended])]),
        "one merged segment over the live coverage of S, T and the appended fragment"
    );

    let label = "after the merge";
    let mut mismatches = Vec::new();
    if kind == Kind::BTree {
        let claimed = [DESTINATIONS[0], DESTINATIONS[1], state.appended];
        for (value, addr) in row_addrs_by_i(&dataset).await {
            let stored = raw_segment_row_addrs(&dataset, &m, value).await;
            if stored != vec![addr] {
                mismatches.push(format!(
                    "{label}: M's file stores {stored:?} for i = {value}, the live address is {addr}"
                ));
            }
            for addr in stored {
                let fragment = RowAddress::from(addr).fragment_id();
                if !claimed.contains(&fragment) {
                    mismatches.push(format!(
                        "{label}: M's file addresses fragment {fragment}, outside its bitmap"
                    ));
                }
            }
        }
        let obsolete = raw_segment_row_addrs(&dataset, &m, OVERLAID_I[0]).await;
        if !obsolete.is_empty() {
            mismatches.push(format!(
                "{label}: M's file stores {obsolete:?} for the obsolete i = {}",
                OVERLAID_I[0]
            ));
        }
    }
    let m_uuid = m.uuid;
    let answered_by_m = |_: i32, _: u32| m_uuid;
    mismatches.extend(overlaid_read_mismatches(kind, &dataset, label, Some(&answered_by_m)).await);
    report(&mismatches, label);

    // A remapping (BTree) merge re-reads the overlaid row and is stamped with
    // the manifest it read; an FTS merge keeps the oldest selected source's
    // stamp, below the overlay, so readers keep replaying the overlaid row.
    let expected_stamp = match kind {
        Kind::BTree => read_version,
        Kind::Fts => state.s.dataset_version.min(state.t.dataset_version),
    };
    assert_eq!(
        m.dataset_version, expected_stamp,
        "M's stamp (read {read_version}, overlay {overlay_version}, S {}, T {})",
        state.s.dataset_version, state.t.dataset_version
    );
    if kind == Kind::Fts {
        assert!(
            m.dataset_version < overlay_version,
            "M's stamp {} stays below the overlay's version {overlay_version}",
            m.dataset_version
        );
    }

    let dataset = trim_and_reopen(dataset, &state.uri).await;
    assert!(
        ledger_transitions(&dataset).await.is_empty(),
        "the trim releases the transition every holder was merged out of: {:?}",
        raw_fri(&dataset).await
    );
    assert_unchanged(&dataset, name, &m, "M across the trim").await;
    assert_eq!(
        raw_segments(&dataset, name).await.len(),
        1,
        "M is still the only segment"
    );
    assert_overlay_kept(&dataset, overlay_version, "fixture: after the trim").await;

    let label = "after the trim";
    let mismatches = overlaid_read_mismatches(kind, &dataset, label, Some(&answered_by_m)).await;
    report(&mismatches, label);
}

/// (T7) The same overlay before the default optimize, which selects only T:
/// the merge is declined, S and T stay as committed below the overlay, a
/// delta indexes the appended fragment, and the overlaid row answers only
/// its replacement value before the optimize, after it and across the trim.
#[rstest::rstest]
#[case::btree(Kind::BTree)]
#[case::fts(Kind::Fts)]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn overlay_before_declined_merge_is_honored(#[case] kind: Kind) {
    let state = Box::pin(joined_state(kind, true, &[])).await;
    let name = kind.name();
    let (mut dataset, overlay_version) = Box::pin(overlay_destination(
        state.dataset,
        &state.uri,
        kind,
        [&state.s, &state.t],
    ))
    .await;
    let label = "before the optimize";
    report(
        &overlaid_read_mismatches(kind, &dataset, label, None).await,
        label,
    );

    let read_version = dataset.manifest.version;
    assert_eq!(
        read_version, overlay_version,
        "fixture: the optimize reads the overlay's version"
    );
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let dataset = fresh_session(&state.uri).await;
    assert_overlay_kept(&dataset, overlay_version, "fixture: after the optimize").await;

    let n = only_new_segment(&dataset, name, &[state.s.uuid, state.t.uuid]).await;
    assert_eq!(
        raw_bitmaps(&dataset, name).await,
        expected_bitmaps(&[
            (state.s.uuid, &[0]),
            (state.t.uuid, &[1]),
            (n.uuid, &[state.appended]),
        ]),
        "S and T stay; the new segment covers the appended fragment"
    );
    assert_unchanged(&dataset, name, &state.s, "S").await;
    assert_unchanged(&dataset, name, &state.t, "T").await;

    let (s, t, appended, n_uuid) = (state.s.uuid, state.t.uuid, state.appended, n.uuid);
    let answering = move |value: i32, fragment: u32| {
        if fragment == appended {
            n_uuid
        } else {
            source_before_overlay(value, s, t)
        }
    };
    let label = "after the declined merge";
    report(
        &overlaid_read_mismatches(kind, &dataset, label, Some(&answering)).await,
        label,
    );

    // The delta is stamped with the manifest the optimize read; S and T keep
    // their stamps below the overlay, so readers keep replaying the overlaid row.
    assert_eq!(
        n.dataset_version, read_version,
        "the new segment is stamped with the manifest the optimize read"
    );
    for source in [&state.s, &state.t] {
        assert!(
            source.dataset_version < overlay_version,
            "segment {} keeps its stamp {} below the overlay's version {overlay_version}",
            source.uuid,
            source.dataset_version
        );
    }

    let dataset = trim_and_reopen(dataset, &state.uri).await;
    assert_eq!(
        ledger_transitions(&dataset).await,
        vec![(vec![0, 1], vec![10, 11])],
        "the trim keeps the transition S and T still need"
    );
    for (before, segment) in [(&state.s, "S"), (&state.t, "T"), (&n, "N")] {
        assert_unchanged(
            &dataset,
            name,
            before,
            &format!("{segment} across the trim"),
        )
        .await;
    }
    assert_eq!(
        raw_segments(&dataset, name).await.len(),
        3,
        "the trim leaves S, T and the new segment"
    );
    assert_overlay_kept(&dataset, overlay_version, "fixture: after the trim").await;

    let label = "after the trim beside the declined merge";
    report(
        &overlaid_read_mismatches(kind, &dataset, label, Some(&answering)).await,
        label,
    );
}
