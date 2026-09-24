// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Staged (uncommitted) index segments merged and committed on a table that
//! carries a tagged fragment reuse history: the distributed build flow
//! (`execute_uncommitted` per fragment, `merge_existing_index_segments`,
//! `commit_existing_index_segments`).
//!
//! Staged segments are unknown to the snapshot plan, so the merge plans them
//! as one group with the reader's own algorithm. The merged segment claims the
//! live coverage the reader derives for the group (the destinations it covers
//! completely), never a destination the group cannot serve.

use super::*;
use crate::dataset::index::frag_reuse::cleanup_frag_reuse_index;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::write::CommitBuilder;
use crate::index::frag_reuse_reader::tests as reader_tests;
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
use lance_core::utils::tempfile::TempStrDir;
use lance_table::format::Fragment;
use lance_table::transaction::{
    FragReuseUpdate, FragmentReuseRewrite, Operation, RewriteGroup, Transaction,
};
use roaring::RoaringBitmap;

/// Two fragments of four rows: `i` 0..8, a constant `text` and a payload
/// `w` equal to `i` (a key no index covers, for in-place rewrites).
async fn disk_fixture(uri: &str) -> Dataset {
    lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col(
            "text",
            lance_datagen::array::fill_utf8("document".to_string()),
        )
        .col("w", lance_datagen::array::step::<Int32Type>())
        .into_dataset(uri, FragmentCount::from(2), FragmentRowCount::from(4))
        .await
        .unwrap()
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

/// Stable partition of `source_ids` into destinations numbered from
/// `dest_base_id`, committed through the real commit path.
async fn commit_stable_partition(
    dataset: Dataset,
    source_ids: &[u64],
    dest_base_id: u64,
) -> Dataset {
    let old_fragments: Vec<Fragment> = source_ids
        .iter()
        .map(|id| {
            dataset
                .fragments()
                .iter()
                .find(|f| f.id == *id)
                .unwrap()
                .clone()
        })
        .collect();
    let (transition, destinations) =
        reader_tests::prepare_partition(&dataset, source_ids, dest_base_id).await;
    let read_version = dataset.manifest.version;
    CommitBuilder::new(Arc::new(dataset))
        .execute(Transaction::new(
            read_version,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments,
                    new_fragments: destinations,
                }],
                rewritten_indices: vec![],
                frag_reuse: Some(FragReuseUpdate::AppendTransitions(
                    FragmentReuseRewrite::new(vec![transition]),
                )),
            },
            None,
        ))
        .await
        .unwrap()
}

async fn staged_segment(
    dataset: &mut Dataset,
    column: &str,
    index_type: IndexType,
    fragments: Vec<u32>,
) -> IndexMetadata {
    let params = ScalarIndexParams::for_builtin(index_type.try_into().unwrap());
    CreateIndexBuilder::new(dataset, &[column], index_type, &params)
        .name("staged".to_string())
        .fragments(fragments)
        .execute_uncommitted()
        .await
        .unwrap()
}

/// Sorted `i` values, optionally under a predicate, with or without the index.
async fn values(dataset: &Dataset, predicate: Option<&str>, use_index: bool) -> Vec<i32> {
    let mut scan = dataset.scan();
    if let Some(predicate) = predicate {
        scan.filter(predicate).unwrap();
    }
    scan.use_scalar_index(use_index);
    let batch = scan.try_into_batch().await.unwrap();
    let mut out: Vec<i32> = batch["i"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect();
    out.sort_unstable();
    out
}

/// Every indexed point query and the full scan equal their index-disabled
/// twins; `indexed` says whether the plan must use (or must not use) the index.
async fn assert_queries_match_scans(dataset: &Dataset, indexed: bool) {
    for value in 0..8 {
        let predicate = format!("i = {value}");
        let plan = dataset
            .scan()
            .filter(&predicate)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert_eq!(plan.contains("ScalarIndexQuery"), indexed, "{plan}");
        assert_eq!(
            values(dataset, Some(&predicate), true).await,
            values(dataset, Some(&predicate), false).await,
            "{predicate}"
        );
    }
    assert_eq!(
        values(dataset, None, true).await,
        values(dataset, None, false).await
    );
}

async fn stored_segment(dataset: &Dataset, name: &str) -> IndexMetadata {
    crate::index::load_all_indices(dataset)
        .await
        .unwrap()
        .iter()
        .find(|idx| idx.name == name)
        .unwrap()
        .clone()
}

async fn derived_coverage(dataset: &Dataset, name: &str) -> Option<RoaringBitmap> {
    dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .find(|idx| idx.name == name)
        .and_then(|idx| idx.fragment_bitmap.clone())
}

/// Both staged contributors are needed to cover a destination: the merged
/// segment claims both destinations, holds live addresses, and indexed
/// queries equal scans; it no longer translates.
#[rstest::rstest]
#[case::btree(IndexType::BTree)]
#[case::bitmap(IndexType::Bitmap)]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_segments_merge_after_stable_partition(#[case] index_type: IndexType) {
    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    let s1 = staged_segment(&mut dataset, "i", index_type, vec![0]).await;
    let s2 = staged_segment(&mut dataset, "i", index_type, vec![1]).await;
    reserve_fragments(&mut dataset, 20).await;
    let mut dataset = commit_stable_partition(dataset, &[0, 1], 10).await;

    let merged = dataset
        .merge_existing_index_segments(vec![s1, s2])
        .await
        .unwrap();
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([10u32, 11]),
        "the merged segment claims the destinations the group covers completely"
    );
    dataset
        .commit_existing_index_segments("staged", "i", vec![merged.clone()])
        .await
        .unwrap();

    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(stored_segment(&dataset, "staged").await.uuid, merged.uuid);
    assert_eq!(
        derived_coverage(&dataset, "staged").await,
        Some(RoaringBitmap::from_iter([10u32, 11])),
        "the committed segment serves both destinations directly"
    );
    assert_queries_match_scans(&dataset, true).await;
    assert_eq!(values(&dataset, Some("i = 3"), true).await, vec![3]);

    // The merged segment serves the destinations directly; whatever trim
    // decides about the transition, the index keeps serving.
    let mut dataset = dataset;
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert_queries_match_scans(&dataset, true).await;
}

/// One staged contributor covers only part of the partition: the merge
/// succeeds, the merged segment claims nothing the group cannot serve (its
/// derived coverage is empty), and the queries scan.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_segment_covering_part_of_a_partition_shrinks_coverage() {
    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    let s1 = staged_segment(&mut dataset, "i", IndexType::BTree, vec![0]).await;
    reserve_fragments(&mut dataset, 20).await;
    let mut dataset = commit_stable_partition(dataset, &[0, 1], 10).await;

    let merged = dataset
        .merge_existing_index_segments(vec![s1])
        .await
        .unwrap();
    assert!(
        merged.fragment_bitmap.as_ref().unwrap().is_empty(),
        "no destination is covered completely, so nothing is claimed"
    );
    dataset
        .commit_existing_index_segments("staged", "i", vec![merged])
        .await
        .unwrap();
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert!(
        derived_coverage(&dataset, "staged")
            .await
            .is_none_or(|coverage| coverage.is_empty()),
        "a destination the group only partly covers is not claimed"
    );
    assert_queries_match_scans(&dataset, false).await;
}

/// Staged segments over fragments no rewrite touched are identity segments:
/// they merge and commit exactly as on an untagged table.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_segments_untouched_by_the_rewrite_merge_as_identity() {
    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[0, 1], 10).await;
    let batch = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step_custom::<Int32Type>(8, 1))
        .col(
            "text",
            lance_datagen::array::fill_utf8("document".to_string()),
        )
        .into_batch_rows(lance_datagen::RowCount::from(8))
        .unwrap();
    let mut dataset = crate::dataset::InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: crate::dataset::WriteMode::Append,
            max_rows_per_file: 4,
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap();
    let appended: Vec<u32> = dataset
        .fragments()
        .iter()
        .map(|f| f.id as u32)
        .filter(|id| *id > 11)
        .collect();
    assert_eq!(appended.len(), 2, "{appended:?}");
    let s1 = staged_segment(&mut dataset, "i", IndexType::BTree, vec![appended[0]]).await;
    let s2 = staged_segment(&mut dataset, "i", IndexType::BTree, vec![appended[1]]).await;

    let merged = dataset
        .merge_existing_index_segments(vec![s1, s2])
        .await
        .unwrap();
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &appended.iter().copied().collect::<RoaringBitmap>()
    );
    dataset
        .commit_existing_index_segments("staged", "i", vec![merged])
        .await
        .unwrap();
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        derived_coverage(&dataset, "staged").await,
        Some(appended.iter().copied().collect::<RoaringBitmap>())
    );
    for value in [9, 12, 15] {
        let predicate = format!("i = {value}");
        let plan = dataset
            .scan()
            .filter(&predicate)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(plan.contains("ScalarIndexQuery"), "{plan}");
        assert_eq!(values(&dataset, Some(&predicate), true).await, vec![value]);
    }
    assert_eq!(
        values(&dataset, None, true).await,
        (0..16).collect::<Vec<_>>()
    );
}

/// The #9421 flow on a tagged table: staged segments built per fragment, a
/// deferred compaction that records an ordered-compaction transition on the
/// tagged history, then merge and commit. The merged segment claims the
/// compacted fragment, which both staged segments together cover.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_segments_merge_across_deferred_compaction_on_tagged_table() {
    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[0, 1], 10).await;
    let batch = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step_custom::<Int32Type>(8, 1))
        .col(
            "text",
            lance_datagen::array::fill_utf8("document".to_string()),
        )
        .into_batch_rows(lance_datagen::RowCount::from(4))
        .unwrap();
    let mut dataset = crate::dataset::InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: crate::dataset::WriteMode::Append,
            max_rows_per_file: 2,
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap();
    let appended: Vec<u32> = dataset
        .fragments()
        .iter()
        .map(|f| f.id as u32)
        .filter(|id| *id > 11)
        .collect();
    assert_eq!(appended.len(), 2, "{appended:?}");
    let s1 = staged_segment(&mut dataset, "i", IndexType::BTree, vec![appended[0]]).await;
    let s2 = staged_segment(&mut dataset, "i", IndexType::BTree, vec![appended[1]]).await;
    // A committed index over the appended fragments makes the compaction
    // record a transition for them (uncovered fragments would compact plainly).
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            Some("committed".to_string()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 4,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let compacted: Vec<u32> = dataset
        .fragments()
        .iter()
        .map(|f| f.id as u32)
        .filter(|id| !appended.contains(id) && *id > 11)
        .collect();
    assert_eq!(compacted.len(), 1, "{compacted:?}");

    let merged = dataset
        .merge_existing_index_segments(vec![s1, s2])
        .await
        .unwrap();
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &compacted.iter().copied().collect::<RoaringBitmap>(),
        "the merged segment claims the compacted destination"
    );
    dataset
        .commit_existing_index_segments("staged", "i", vec![merged])
        .await
        .unwrap();
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        derived_coverage(&dataset, "staged").await,
        Some(compacted.iter().copied().collect::<RoaringBitmap>())
    );
    for value in 8..12 {
        let predicate = format!("i = {value}");
        assert_eq!(
            values(&dataset, Some(&predicate), true).await,
            values(&dataset, Some(&predicate), false).await
        );
        assert_eq!(values(&dataset, Some(&predicate), true).await, vec![value]);
    }
}

/// The NGram merge reads spill files and translates only through the v0
/// remapper: a staged NGram segment that needs translation is refused by the
/// replay (the resolver's NGram rule) rather than merged with stale
/// addresses; the error says to rebuild.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_ngram_segments_needing_translation_are_refused() {
    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    let s1 = staged_segment(&mut dataset, "text", IndexType::NGram, vec![0]).await;
    let s2 = staged_segment(&mut dataset, "text", IndexType::NGram, vec![1]).await;
    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[0, 1], 10).await;
    let error = dataset
        .merge_existing_index_segments(vec![s1, s2])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("Rebuild"), "{error}");
}

/// A segment that is neither staged nor listed in the manifest is an error
/// for the coverage builder, never empty coverage; with its plan, the same
/// segment gets the coverage the plan derives.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn unlisted_segment_without_a_plan_is_an_error() {
    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    let s1 = staged_segment(&mut dataset, "i", IndexType::BTree, vec![0]).await;
    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[0, 1], 10).await;

    let error = crate::index::append::tagged_segment_coverage(&dataset, &[&s1], None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("not registered"), "{error}");

    let plans = crate::index::frag_reuse::plan_staged_segments(&dataset, std::slice::from_ref(&s1))
        .await
        .unwrap()
        .expect("a tagged table plans staged segments");
    let coverage = crate::index::append::tagged_segment_coverage(&dataset, &[&s1], Some(&plans))
        .await
        .unwrap()
        .expect("tagged coverage");
    assert!(
        coverage[&s1.uuid].is_empty(),
        "half a partition covers no destination completely: {:?}",
        coverage[&s1.uuid]
    );
}

/// Two fragments of four rows: `i` 0..8, a random 4-dim `vector` and a
/// payload `w` equal to `i`.
async fn vector_fixture(uri: &str) -> Dataset {
    lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col(
            "vector",
            lance_datagen::array::rand_vec::<arrow_array::types::Float32Type>(4.into()),
        )
        .col("w", lance_datagen::array::step::<Int32Type>())
        .into_dataset(uri, FragmentCount::from(2), FragmentRowCount::from(4))
        .await
        .unwrap()
}

/// The plan and the sorted `i` of the `k` rows nearest to `query`, with or
/// without the vector index.
async fn nearest(
    dataset: &Dataset,
    query: &arrow_array::PrimitiveArray<arrow_array::types::Float32Type>,
    k: usize,
    use_index: bool,
) -> (String, Vec<i32>) {
    let mut scan = dataset.scan();
    scan.nearest("vector", query, k).unwrap();
    scan.use_index(use_index);
    let plan = scan.explain_plan(false).await.unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let mut ids: Vec<i32> = batch["i"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect();
    ids.sort_unstable();
    (plan, ids)
}

/// Staged vector segments never open the dataset: the merge filters each by
/// its stored bitmap in the raw address domain and the merged segment keeps
/// the provenance union, so it claims no destination and translates at query
/// time exactly like a committed vector merge. With one IVF partition the
/// index answers every query the flat scan answers.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_vector_segments_merge_across_stable_partition_keep_provenance() {
    let dir = TempStrDir::default();
    let mut dataset = vector_fixture(dir.as_str()).await;
    // Shards merge only when they share centroids: one fixed centroid, as a
    // distributed build trains once and hands the centroids to every shard.
    let centroids = arrow_array::FixedSizeListArray::try_new_from_values(
        arrow_array::Float32Array::from(vec![0.0f32; 4]),
        4,
    )
    .unwrap();
    let params = crate::index::vector::VectorIndexParams::with_ivf_flat_params(
        lance_linalg::distance::DistanceType::L2,
        lance_index::vector::ivf::IvfBuildParams::try_with_centroids(1, Arc::new(centroids))
            .unwrap(),
    );
    let mut segments = Vec::new();
    for fragment in [0u32, 1] {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
                .name("staged".to_string())
                .fragments(vec![fragment])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    let original = dataset
        .scan()
        .filter("i = 6")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let query = original["vector"].as_fixed_size_list().value(0);
    let query = query
        .as_primitive::<arrow_array::types::Float32Type>()
        .clone();
    reserve_fragments(&mut dataset, 20).await;
    let mut dataset = commit_stable_partition(dataset, &[0, 1], 10).await;

    let merged = dataset
        .merge_existing_index_segments(segments)
        .await
        .unwrap();
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1]),
        "a vector merge keeps the provenance union"
    );
    dataset
        .commit_existing_index_segments("staged", "vector", vec![merged.clone()])
        .await
        .unwrap();

    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    let stored = stored_segment(&dataset, "staged").await;
    assert_eq!(stored.uuid, merged.uuid);
    assert_eq!(
        stored.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1]),
        "the commit keeps the retired provenance that the lineage records"
    );
    assert_eq!(
        derived_coverage(&dataset, "staged").await,
        Some(RoaringBitmap::from_iter([10u32, 11])),
        "the committed segment translates to both destinations"
    );
    let (_, truth) = nearest(&dataset, &query, 1, false).await;
    assert_eq!(truth, vec![6]);
    let (plan, found) = nearest(&dataset, &query, 1, true).await;
    assert!(plan.contains("ANN"), "{plan}");
    assert_eq!(found, truth);
    let (_, flat) = nearest(&dataset, &query, 8, false).await;
    let (_, indexed) = nearest(&dataset, &query, 8, true).await;
    assert_eq!(flat.len(), 8);
    assert_eq!(indexed, flat, "every row is reachable through the index");

    // Trim keeps the transition the merged segment's provenance still names.
    let mut dataset = dataset;
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert!(
        dataset
            .load_index_by_name(lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some()
    );
    let (plan, found) = nearest(&dataset, &query, 1, true).await;
    assert!(plan.contains("ANN"), "{plan}");
    assert_eq!(found, vec![6]);
}

/// Rewrite `column` of the row `w = key` in place with `value`: a
/// partial-schema merge insert keyed on `w`, which no index covers, in
/// `RewriteColumns` mode. The dataset is reopened afterwards.
async fn rewrite_in_place(uri: &str, column: &str, key: i32, value: arrow_array::ArrayRef) {
    use crate::dataset::{MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched};

    let dataset = Dataset::open(uri).await.unwrap();
    let schema = Arc::new(arrow_schema::Schema::from(
        &dataset.schema().project(&["w", column]).unwrap(),
    ));
    let source = arrow_array::RecordBatch::try_new(
        schema,
        vec![Arc::new(arrow_array::Int32Array::from(vec![key])), value],
    )
    .unwrap();
    MergeInsertBuilder::try_new(Arc::new(dataset), vec!["w".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .execute_batches(vec![source])
        .await
        .unwrap();
}

/// The sequence that forces the replay to run BEFORE the merge: segments
/// staged over F0 and F1, `i` of a row in F1 rewritten in place (admitted:
/// no committed index covers `i`), F1 partitioned into F10 and F11, the
/// segments merged, the result committed. The merge translates addresses
/// but never rereads values, so without the replay the merged segment
/// would claim the destinations with the pre-rewrite value inside.
/// Replayed first, the rewrite withdraws F1 from its segment; the merge has
/// nothing to claim for F10 and F11, keeps F0, and the new value comes from
/// the scan.
#[rstest::rstest]
#[case::btree(IndexType::BTree)]
#[case::bitmap(IndexType::Bitmap)]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_segments_rewritten_before_the_partition_are_validated_before_the_merge(
    #[case] index_type: IndexType,
) {
    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    let s1 = staged_segment(&mut dataset, "i", index_type, vec![0]).await;
    let s2 = staged_segment(&mut dataset, "i", index_type, vec![1]).await;
    // Row w = 6 lives in F1.
    rewrite_in_place(
        dir.as_str(),
        "i",
        6,
        Arc::new(arrow_array::Int32Array::from(vec![666])),
    )
    .await;
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let mut dataset = commit_stable_partition(dataset, &[1], 10).await;
    assert_eq!(
        dataset.fragments().iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![0, 10, 11]
    );

    let merged = dataset
        .merge_existing_index_segments(vec![s1, s2])
        .await
        .unwrap();
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32]),
        "the rewritten source was withdrawn before the merge; F0 stays"
    );
    dataset
        .commit_existing_index_segments("staged", "i", vec![merged])
        .await
        .unwrap();
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        derived_coverage(&dataset, "staged").await,
        Some(RoaringBitmap::from_iter([0u32]))
    );
    assert_eq!(values(&dataset, Some("i = 666"), true).await, vec![666]);
    assert_eq!(
        values(&dataset, Some("i = 6"), true).await,
        Vec::<i32>::new()
    );
    assert_eq!(values(&dataset, Some("i = 2"), true).await, vec![2]);
    assert_eq!(
        values(&dataset, None, true).await,
        values(&dataset, None, false).await
    );
}

/// The replay needs the whole history between the build and the snapshot.
/// The listing only knows the manifests that still exist: with the build
/// version kept by a tag and the versions after it cleaned up, the partition
/// and the in-place rewrite between them leave no trace, so the segments
/// cannot be validated and both the merge and the commit refuse them.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_segments_refuse_a_history_with_a_cleaned_up_version() {
    use crate::dataset::cleanup::{CleanupPolicyBuilder, cleanup_old_versions};

    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    let s1 = staged_segment(&mut dataset, "i", IndexType::BTree, vec![0]).await;
    let s2 = staged_segment(&mut dataset, "i", IndexType::BTree, vec![1]).await;
    let build_version = dataset.manifest.version;
    dataset.tags().create("build", build_version).await.unwrap();
    reserve_fragments(&mut dataset, 20).await;
    commit_stable_partition(dataset, &[1], 10).await;
    // Row w = 6 was in F1 and now lives in a partition destination.
    rewrite_in_place(
        dir.as_str(),
        "i",
        6,
        Arc::new(arrow_array::Int32Array::from(vec![666])),
    )
    .await;
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(dataset.manifest.version, build_version + 3);
    let policy = CleanupPolicyBuilder::default()
        .before_timestamp(chrono::Utc::now() + chrono::Duration::days(1))
        .error_if_tagged_old_versions(false)
        .build();
    cleanup_old_versions(&dataset, policy).await.unwrap();
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert!(dataset.checkout_version(build_version).await.is_ok());
    assert!(
        dataset.checkout_version(build_version + 1).await.is_err(),
        "the intermediate versions are gone"
    );

    let error = dataset
        .merge_existing_index_segments(vec![s1.clone(), s2.clone()])
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("cleaned up") && error.to_string().contains("Rebuild"),
        "{error}"
    );
    let error = dataset
        .commit_existing_index_segments("staged", "i", vec![s1, s2])
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("cleaned up") && error.to_string().contains("Rebuild"),
        "{error}"
    );
    assert!(
        crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .iter()
            .all(|idx| idx.name != "staged"),
        "nothing was committed"
    );
}

/// The same sequence for IVF_FLAT shards: the vector of a row in F1 is
/// rewritten in place, F1 is partitioned, the shards merged and committed.
/// The merged segment keeps F0 only and the nearest neighbour of the new
/// vector is found by the scan of the uncovered destinations.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_vector_segments_rewritten_before_the_partition_are_validated_before_the_merge() {
    let dir = TempStrDir::default();
    let mut dataset = vector_fixture(dir.as_str()).await;
    let centroids = arrow_array::FixedSizeListArray::try_new_from_values(
        arrow_array::Float32Array::from(vec![0.0f32; 4]),
        4,
    )
    .unwrap();
    let params = crate::index::vector::VectorIndexParams::with_ivf_flat_params(
        lance_linalg::distance::DistanceType::L2,
        lance_index::vector::ivf::IvfBuildParams::try_with_centroids(1, Arc::new(centroids))
            .unwrap(),
    );
    let mut segments = Vec::new();
    for fragment in [0u32, 1] {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
                .name("staged".to_string())
                .fragments(vec![fragment])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    // Row w = 6 (F1) gets a vector far from every other; the shard over F1
    // still holds its old vector.
    let far = arrow_array::FixedSizeListArray::from_iter_primitive::<
        arrow_array::types::Float32Type,
        _,
        _,
    >(vec![Some(vec![Some(1000.0f32); 4])], 4);
    rewrite_in_place(dir.as_str(), "vector", 6, Arc::new(far)).await;
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let mut dataset = commit_stable_partition(dataset, &[1], 10).await;

    let merged = dataset
        .merge_existing_index_segments(segments)
        .await
        .unwrap();
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32]),
        "the rewritten source was withdrawn before the merge; F0 stays"
    );
    dataset
        .commit_existing_index_segments("staged", "vector", vec![merged])
        .await
        .unwrap();
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        derived_coverage(&dataset, "staged").await,
        Some(RoaringBitmap::from_iter([0u32]))
    );
    let query = arrow_array::Float32Array::from(vec![1000.0f32; 4]);
    let (_, truth) = nearest(&dataset, &query, 1, false).await;
    assert_eq!(truth, vec![6]);
    let (_, found) = nearest(&dataset, &query, 1, true).await;
    assert_eq!(found, vec![6], "the rewritten row is found by the scan");
    let (_, flat) = nearest(&dataset, &query, 8, false).await;
    let (_, indexed) = nearest(&dataset, &query, 8, true).await;
    assert_eq!(indexed, flat);
}
