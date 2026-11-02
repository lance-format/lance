// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The fragment reuse contract, checked for every index type and history form:
//! whatever the history records, an indexed query returns exactly the rows a
//! query computed from the table's current contents returns.
//!
//! The oracle never consults an index or the history: it scans the table and
//! evaluates each query in Rust. Each step is checked through the handle that
//! ran it and through a cold handle with a fresh session, so a cache keyed
//! without the history's identity cannot hide behind a warm entry.
//!
//! Several steps leave an index's declared coverage narrower than the entries
//! its files physically hold (a whole-fragment delete, an in-place update, a
//! staged build committed after a concurrent delete). Readers must treat
//! coverage as an allow list rather than a description of the files.

use std::collections::BTreeSet;

use super::*;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::optimize::remapping::remap_column_index;
use crate::dataset::{MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched};
use crate::index::frag_reuse::frag_reuse_entry_appending;
use crate::index::frag_reuse_reader::tests::prepare_partition;
use crate::session::Session;
use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ListArray, StringArray};
use lance_index::IndexParams;
use lance_index::optimize::OptimizeOptions;

const ROWS_PER_FRAGMENT: i32 = 8;
const DIM: i32 = 4;
const IVF_PARTITIONS: usize = 2;
/// Added to `v` by an in-place update, so updated rows answer different
/// queries than they did when the index was built.
const UPDATED: i32 = 1000;

/// The record form a fragment rewrite takes.
#[derive(Debug, Clone, Copy)]
enum HistoryForm {
    /// A deferred compaction on an untagged table records a v0 reuse version.
    V0,
    /// Deferred compaction on a tagged table records ordered-compaction
    /// transitions. A stable partition committed before the index exists
    /// makes the table tagged.
    TaggedOrdered,
    /// Every rewrite is a stable partition.
    TaggedPartition,
}

#[derive(Debug, Clone, Copy)]
enum Query {
    Filter(&'static str, fn(&Row) -> bool),
    FullText(&'static str),
    /// `k` close to the live row count makes a candidate the prefilter
    /// should have excluded show up as a missing row.
    Nearest {
        target: f32,
        k: usize,
    },
}

#[derive(Debug, Clone, Copy)]
enum IndexKind {
    BTree,
    Bitmap,
    ZoneMap,
    BloomFilter,
    LabelList,
    NGram,
    Inverted,
    IvfFlat,
}

impl IndexKind {
    fn column(self) -> &'static str {
        match self {
            Self::BTree | Self::Bitmap | Self::ZoneMap | Self::BloomFilter => "v",
            Self::LabelList => "tags",
            Self::NGram | Self::Inverted => "text",
            Self::IvfFlat => "vec",
        }
    }

    fn queries(self) -> Vec<Query> {
        match self {
            Self::BTree | Self::Bitmap | Self::ZoneMap => vec![
                Query::Filter("v = 13", |row| row.v == 13),
                Query::Filter("v >= 6 AND v < 20", |row| (6..20).contains(&row.v)),
                Query::Filter("v >= 1000", |row| row.v >= UPDATED),
            ],
            Self::BloomFilter => vec![
                Query::Filter("v = 13", |row| row.v == 13),
                Query::Filter("v IN (3, 13, 1013)", |row| [3, 13, 1013].contains(&row.v)),
            ],
            Self::LabelList => vec![
                Query::Filter("array_has_any(tags, ['t1'])", |row| row.tag == "t1"),
                Query::Filter("array_has_any(tags, ['updated'])", |row| {
                    row.tag == "updated"
                }),
            ],
            Self::NGram => vec![
                Query::Filter("contains(text, 'w1')", |row| row.text.contains("w1")),
                Query::Filter("contains(text, 'updated')", |row| {
                    row.text.contains("updated")
                }),
            ],
            Self::Inverted => vec![Query::FullText("w1"), Query::FullText("updated")],
            Self::IvfFlat => vec![
                Query::Nearest { target: 10.3, k: 5 },
                Query::Nearest {
                    target: 10.3,
                    k: 30,
                },
            ],
        }
    }

    fn index_type(self) -> IndexType {
        match self {
            Self::BTree => IndexType::BTree,
            Self::Bitmap => IndexType::Bitmap,
            Self::ZoneMap => IndexType::ZoneMap,
            Self::BloomFilter => IndexType::BloomFilter,
            Self::LabelList => IndexType::LabelList,
            Self::NGram => IndexType::NGram,
            Self::Inverted => IndexType::Inverted,
            Self::IvfFlat => IndexType::Vector,
        }
    }

    fn params(self) -> Box<dyn IndexParams> {
        let builtin = match self {
            Self::Inverted => return Box::new(InvertedIndexParams::default()),
            Self::IvfFlat => {
                return Box::new(VectorIndexParams::ivf_flat(IVF_PARTITIONS, MetricType::L2));
            }
            Self::BTree => BuiltinIndexType::BTree,
            Self::Bitmap => BuiltinIndexType::Bitmap,
            Self::ZoneMap => BuiltinIndexType::ZoneMap,
            Self::BloomFilter => BuiltinIndexType::BloomFilter,
            Self::LabelList => BuiltinIndexType::LabelList,
            Self::NGram => BuiltinIndexType::NGram,
        };
        Box::new(ScalarIndexParams::for_builtin(builtin))
    }

    async fn create(self, dataset: &mut Dataset) {
        let params = self.params();
        dataset
            .create_index(
                &[self.column()],
                self.index_type(),
                Some("idx".to_string()),
                params.as_ref(),
                false,
            )
            .await
            .unwrap();
    }
}

/// The columns the oracle evaluates. `i` is unique and never updated, so it
/// identifies a row across every rewrite; the other columns are indexed and
/// change on an in-place update.
#[derive(Debug)]
struct Row {
    i: i32,
    v: i32,
    tag: String,
    text: String,
    vec: Vec<f32>,
}

/// Rows `ids` as written, or as an in-place update rewrites them.
fn rows(
    ids: impl IntoIterator<Item = i32>,
    is_updated: bool,
) -> impl arrow_array::RecordBatchReader + Send + 'static {
    let ids: Vec<i32> = ids.into_iter().collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int32, false),
        Field::new("pad", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
            false,
        ),
    ]));
    let mut tags = ListBuilder::new(StringBuilder::new());
    for i in &ids {
        if is_updated {
            tags.values().append_value("updated");
        } else {
            tags.values().append_value(format!("t{}", i % 4));
        }
        tags.append(true);
    }
    let tags: ListArray = tags.finish();
    // An updated vector moves next to the query target, so it changes which
    // rows are nearest.
    let vectors = Float32Array::from_iter_values(ids.iter().flat_map(|&i| {
        let x = if is_updated {
            10.3 + 0.01 * i as f32
        } else {
            i as f32
        };
        [x, 0.0, 0.0, 0.0].into_iter()
    }));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids.clone())),
            Arc::new(Int32Array::from(ids.clone())),
            Arc::new(Int32Array::from_iter_values(
                ids.iter()
                    .map(|&i| if is_updated { UPDATED + i } else { i }),
            )),
            Arc::new(StringArray::from_iter_values(ids.iter().map(|&i| {
                if is_updated {
                    format!("updated x{}", i % 3)
                } else {
                    format!("w{} x{}", i % 7, i % 3)
                }
            }))),
            Arc::new(tags),
            Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM).unwrap()),
        ],
    )
    .unwrap();
    RecordBatchIterator::new(vec![Ok(batch)], schema)
}

fn write_params() -> WriteParams {
    WriteParams {
        max_rows_per_file: ROWS_PER_FRAGMENT as usize,
        ..Default::default()
    }
}

async fn live_rows(dataset: &Dataset) -> Vec<Row> {
    let batch = dataset.scan().try_into_batch().await.unwrap();
    let ids = batch["i"].as_primitive::<Int32Type>();
    let values = batch["v"].as_primitive::<Int32Type>();
    let texts = batch["text"].as_string::<i32>();
    let tags = batch["tags"].as_list::<i32>();
    let vectors = batch["vec"].as_fixed_size_list();
    (0..batch.num_rows())
        .map(|row| Row {
            i: ids.value(row),
            v: values.value(row),
            tag: tags.value(row).as_string::<i32>().value(0).to_string(),
            text: texts.value(row).to_string(),
            vec: vectors
                .value(row)
                .as_primitive::<Float32Type>()
                .values()
                .to_vec(),
        })
        .collect()
}

fn expected(rows: &[Row], query: Query) -> BTreeSet<i32> {
    match query {
        Query::Filter(_, predicate) => rows
            .iter()
            .filter(|row| predicate(row))
            .map(|row| row.i)
            .collect(),
        Query::FullText(token) => rows
            .iter()
            .filter(|row| row.text.split_whitespace().any(|word| word == token))
            .map(|row| row.i)
            .collect(),
        Query::Nearest { target, k } => {
            let mut by_distance: Vec<(f32, i32)> = rows
                .iter()
                .map(|row| {
                    let distance = (row.vec[0] - target).powi(2)
                        + row.vec[1..].iter().map(|v| v * v).sum::<f32>();
                    (distance, row.i)
                })
                .collect();
            by_distance.sort_by(|a, b| a.0.total_cmp(&b.0));
            by_distance.into_iter().take(k).map(|(_, i)| i).collect()
        }
    }
}

async fn indexed(dataset: &Dataset, query: Query) -> Result<BTreeSet<i32>> {
    let mut scanner = dataset.scan();
    match query {
        Query::Filter(filter, _) => {
            scanner.filter(filter)?;
        }
        Query::FullText(token) => {
            scanner.full_text_search(FullTextSearchQuery::new(token.to_string()))?;
        }
        Query::Nearest { target, k } => {
            let mut vector = vec![0.0f32; DIM as usize];
            vector[0] = target;
            scanner
                .nearest("vec", &Float32Array::from(vector), k)?
                .minimum_nprobes(IVF_PARTITIONS);
        }
    }
    scanner.project(&["i"])?;
    let batch = scanner.try_into_batch().await?;
    let ids = batch["i"].as_primitive::<Int32Type>().values();
    let unique: BTreeSet<i32> = ids.iter().copied().collect();
    assert_eq!(unique.len(), ids.len(), "{query:?} returned a row twice");
    Ok(unique)
}

async fn assert_contract(dataset: &Dataset, kind: IndexKind, form: HistoryForm, step: &str) {
    let cold = DatasetBuilder::from_uri(dataset.uri())
        .with_session(Arc::new(Session::default()))
        .load()
        .await
        .unwrap();
    let rows = live_rows(dataset).await;
    for query in kind.queries() {
        let expected = expected(&rows, query);
        for (handle, which) in [(dataset, "warm"), (&cold, "cold")] {
            let context = format!(
                "{kind:?} under {form:?} after {step}: {query:?} through the {which} handle"
            );
            let actual = indexed(handle, query)
                .await
                .unwrap_or_else(|error| panic!("{context}: {error}"));
            assert_eq!(actual, expected, "{context}");
        }
    }
}

async fn reserve_fragments(dataset: &mut Dataset, num_fragments: u32) {
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::ReserveFragments { num_fragments },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
}

/// Partitions every live fragment into two destinations through the commit
/// path, recording a stable-partition transition.
async fn stable_partition(dataset: &mut Dataset) {
    reserve_fragments(dataset, 2).await;
    let dest_base_id = dataset.manifest.max_fragment_id().unwrap() - 1;
    let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
    let source_ids: Vec<u64> = old_fragments.iter().map(|fragment| fragment.id).collect();
    let (transition, new_fragments) = prepare_partition(dataset, &source_ids, dest_base_id).await;
    let read_version = dataset.manifest.version;
    let frag_reuse_index = Some(
        frag_reuse_entry_appending(dataset, vec![transition])
            .await
            .unwrap(),
    );
    *dataset = crate::dataset::write::CommitBuilder::new(Arc::new(dataset.clone()))
        .execute(Transaction::new(
            read_version,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments,
                    new_fragments,
                }],
                rewritten_indices: vec![],
                frag_reuse_index,
            },
            None,
        ))
        .await
        .unwrap();
}

async fn rewrite(dataset: &mut Dataset, form: HistoryForm) {
    match form {
        HistoryForm::V0 | HistoryForm::TaggedOrdered => {
            let options = CompactionOptions {
                defer_index_remap: true,
                ..Default::default()
            };
            compact_files(dataset, options, None).await.unwrap();
        }
        HistoryForm::TaggedPartition => stable_partition(dataset).await,
    }
    let entry = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .expect("a deferred rewrite of indexed fragments records its moves");
    let is_tagged = entry.index_version >= 1;
    assert_eq!(
        is_tagged,
        !matches!(form, HistoryForm::V0),
        "{form:?} recorded index_version {}",
        entry.index_version
    );
}

/// Deletes every row of the live fragment at `position`, whatever its layout
/// after earlier rewrites.
async fn delete_whole_fragment(dataset: &mut Dataset, position: usize) {
    let fragment = dataset.fragments()[position].clone();
    let mut scan = dataset.scan();
    scan.with_fragments(vec![fragment]).project(&["i"]).unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let ids: Vec<String> = batch["i"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .map(|i| i.to_string())
        .collect();
    dataset
        .delete(&format!("i IN ({})", ids.join(", ")))
        .await
        .unwrap();
}

/// Rewrites every indexed column of rows `ids` in place, keeping their
/// addresses.
async fn update_in_place(dataset: &mut Dataset, ids: &[i32]) {
    let mut merge =
        MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["i".into()]).unwrap();
    merge
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns);
    // Leaving `pad` out of the patch is what makes the update a column
    // rewrite that keeps row addresses.
    let patch = rows(ids.iter().copied(), true)
        .next()
        .unwrap()
        .unwrap()
        .project(&[0, 2, 3, 4, 5])
        .unwrap();
    let schema = patch.schema();
    let (updated, _) = merge
        .try_build()
        .unwrap()
        .execute_reader(RecordBatchIterator::new([Ok(patch)], schema))
        .await
        .unwrap();
    *dataset = updated.as_ref().clone();
}

/// Stages a merge of the index's segments, lets a concurrent writer delete a
/// whole fragment, append rows and index them, then commits the staged
/// segment on the retry: its files still hold the deleted fragment's entries.
async fn staged_merge_racing_a_delete(dataset: &mut Dataset, kind: IndexKind, next: i32) {
    let segments = dataset.load_indices_by_name("idx").await.unwrap();
    // Some index types refuse to merge segments over retired fragments or
    // under a tagged history; a refusal leaves the table as it was.
    let staged = match dataset.merge_existing_index_segments(segments).await {
        Ok(staged) => staged,
        Err(Error::NotSupported { .. } | Error::InvalidInput { .. }) => return,
        Err(error) => panic!("{kind:?}: staging a merge failed: {error}"),
    };

    let mut concurrent = dataset.clone();
    delete_whole_fragment(&mut concurrent, 0).await;
    concurrent
        .append(
            rows(next..next + ROWS_PER_FRAGMENT, false),
            Some(write_params()),
        )
        .await
        .unwrap();
    concurrent
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();

    let conflict = dataset
        .commit_existing_index_segments("idx", kind.column(), vec![staged.clone()])
        .await
        .unwrap_err();
    assert!(
        matches!(conflict, Error::RetryableCommitConflict { .. }),
        "{conflict}"
    );
    concurrent
        .commit_existing_index_segments("idx", kind.column(), vec![staged])
        .await
        .unwrap();
    *dataset = concurrent;
}

async fn run_contract(kind: IndexKind, form: HistoryForm) {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut next = 4 * ROWS_PER_FRAGMENT;
    let mut dataset = Dataset::write(rows(0..next, false), uri, Some(write_params()))
        .await
        .unwrap();
    if matches!(form, HistoryForm::TaggedOrdered) {
        stable_partition(&mut dataset).await;
    }
    kind.create(&mut dataset).await;
    assert_contract(&dataset, kind, form, "build").await;

    dataset.delete("i % 5 = 0").await.unwrap();
    assert_contract(&dataset, kind, form, "partial delete").await;

    delete_whole_fragment(&mut dataset, 0).await;
    assert_contract(&dataset, kind, form, "whole-fragment delete").await;

    staged_merge_racing_a_delete(&mut dataset, kind, next).await;
    next += ROWS_PER_FRAGMENT;
    assert_contract(&dataset, kind, form, "staged merge racing a delete").await;

    rewrite(&mut dataset, form).await;
    assert_contract(&dataset, kind, form, "first rewrite").await;

    dataset
        .append(
            rows(next..next + ROWS_PER_FRAGMENT, false),
            Some(write_params()),
        )
        .await
        .unwrap();
    dataset.delete("i % 7 = 3").await.unwrap();
    assert_contract(&dataset, kind, form, "append and delete over the rewrite").await;

    update_in_place(&mut dataset, &[11, 13, 17, 33]).await;
    assert_contract(&dataset, kind, form, "in-place update").await;

    rewrite(&mut dataset, form).await;
    assert_contract(&dataset, kind, form, "chained rewrite").await;

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    assert_contract(&dataset, kind, form, "optimize_indices").await;

    remap_column_index(&mut dataset, &[kind.column()], Some("idx".to_string()))
        .await
        .unwrap();
    assert_contract(&dataset, kind, form, "remap").await;

    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert_contract(&dataset, kind, form, "trim").await;
}

#[rstest]
#[tokio::test]
async fn test_indexed_queries_match_the_table_through_fragment_reuse(
    #[values(
        IndexKind::BTree,
        IndexKind::Bitmap,
        IndexKind::ZoneMap,
        IndexKind::BloomFilter,
        IndexKind::LabelList,
        IndexKind::NGram,
        IndexKind::IvfFlat
    )]
    kind: IndexKind,
    #[values(
        HistoryForm::V0,
        HistoryForm::TaggedOrdered,
        HistoryForm::TaggedPartition
    )]
    form: HistoryForm,
) {
    run_contract(kind, form).await;
}

#[tokio::test]
async fn test_full_text_search_matches_the_table_through_v0_fragment_reuse() {
    run_contract(IndexKind::Inverted, HistoryForm::V0).await;
}

#[rstest]
#[case::ordered(HistoryForm::TaggedOrdered)]
#[case::partition(HistoryForm::TaggedPartition)]
#[ignore = "https://github.com/lance-format/lance/issues/10623"]
#[tokio::test]
async fn test_full_text_search_matches_the_table_through_tagged_fragment_reuse(
    #[case] form: HistoryForm,
) {
    run_contract(IndexKind::Inverted, form).await;
}
