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
use lance_table::transaction::{Operation, RewriteGroup, Transaction};
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
    let frag_reuse_index = Some(
        crate::index::frag_reuse::frag_reuse_entry_appending(&dataset, vec![transition])
            .await
            .unwrap(),
    );
    CommitBuilder::new(Arc::new(dataset))
        .execute(Transaction::new(
            read_version,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments,
                    new_fragments: destinations,
                }],
                rewritten_indices: vec![],
                frag_reuse_index,
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

/// A committed vector index whose coverage an in-place rewrite of the
/// vector column withdrew entirely: queries scan, and `optimize_indices`
/// rebuilds the index from the live fragments (the withdrawn segment is
/// dormant, opened through the maintenance entry for its parameters only),
/// replacing it with a segment that claims the live fragments and answers
/// through the ANN path exactly like the flat scan.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn withdrawn_vector_index_is_rebuilt_by_optimize() {
    use lance_index::optimize::OptimizeOptions;

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
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vec_idx".into()),
            &params,
            true,
        )
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[0, 1], 10).await;
    let before = stored_segment(&dataset, "vec_idx").await;
    assert_eq!(
        before.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1])
    );

    // Row w = 6 gets a vector far from every other, rewritten in place: the
    // whole transition is withdrawn and the segment is left empty.
    let far = arrow_array::FixedSizeListArray::from_iter_primitive::<
        arrow_array::types::Float32Type,
        _,
        _,
    >(vec![Some(vec![Some(1000.0f32); 4])], 4);
    rewrite_in_place(dir.as_str(), "vector", 6, Arc::new(far)).await;
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    let withdrawn = stored_segment(&dataset, "vec_idx").await;
    assert_eq!(withdrawn.uuid, before.uuid);
    assert!(withdrawn.fragment_bitmap.as_ref().unwrap().is_empty());
    assert!(
        derived_coverage(&dataset, "vec_idx")
            .await
            .unwrap_or_default()
            .is_empty()
    );
    let query = arrow_array::Float32Array::from(vec![1000.0f32; 4]);
    let (plan, found) = nearest(&dataset, &query, 1, true).await;
    assert!(!plan.contains("ANN"), "scanned while withdrawn: {plan}");
    assert_eq!(found, vec![6]);

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    let rebuilt = stored_segment(&dataset, "vec_idx").await;
    assert_ne!(rebuilt.uuid, withdrawn.uuid);
    assert_eq!(
        rebuilt.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([10u32, 11]),
        "the new segment claims the live fragments"
    );
    assert!(
        crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .iter()
            .all(|idx| idx.uuid != withdrawn.uuid),
        "the withdrawn segment is replaced"
    );
    assert_eq!(
        derived_coverage(&dataset, "vec_idx").await,
        Some(RoaringBitmap::from_iter([10u32, 11]))
    );
    let (plan, found) = nearest(&dataset, &query, 1, true).await;
    assert!(plan.contains("ANN"), "{plan}");
    assert_eq!(found, vec![6]);
    let (_, flat) = nearest(&dataset, &query, 8, false).await;
    let (_, indexed) = nearest(&dataset, &query, 8, true).await;
    assert_eq!(indexed, flat);
}

/// Routine maintenance of the logical index the staged segments rebuild is
/// no conflict for the replay. A distributed rebuild of `staged` (committed
/// over F0 and F1) stages one segment per fragment; F1 is partitioned, a
/// fragment is appended and `optimize_indices` adds a same-name delta for
/// it in the window. The staged segments still merge and commit, replacing
/// the old segment while the delta is retained, and the whole index answers.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_segments_survive_a_same_name_optimize_in_the_window() {
    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
    use lance_index::optimize::OptimizeOptions;

    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    let params = ScalarIndexParams::for_builtin(IndexType::BTree.try_into().unwrap());
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            Some("staged".into()),
            &params,
            false,
        )
        .await
        .unwrap();
    let old = stored_segment(&dataset, "staged").await;
    let mut staged = Vec::new();
    for fragment in [0u32, 1] {
        staged.push(
            CreateIndexBuilder::new(&mut dataset, &["i"], IndexType::BTree, &params)
                .name("staged".to_string())
                .fragments(vec![fragment])
                .replace(true)
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[1], 10).await;
    let batch = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step_custom::<Int32Type>(8, 1))
        .col(
            "text",
            lance_datagen::array::fill_utf8("document".to_string()),
        )
        .col("w", lance_datagen::array::step_custom::<Int32Type>(8, 1))
        .into_batch_rows(lance_datagen::RowCount::from(4))
        .unwrap();
    let mut dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap();
    let appended = dataset
        .fragments()
        .iter()
        .map(|f| f.id as u32)
        .find(|id| *id > 11)
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    let delta = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .find(|idx| idx.name == "staged" && idx.uuid != old.uuid)
        .cloned()
        .expect("the same-name delta the window added");
    assert_eq!(
        delta.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([appended])
    );

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 10, 11]),
        "the staged segments translate across the partition despite the same-name delta"
    );
    dataset
        .commit_existing_index_segments("staged", "i", vec![merged])
        .await
        .unwrap();
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    let remaining: Vec<Uuid> = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .filter(|idx| idx.name == "staged")
        .map(|idx| idx.uuid)
        .collect();
    assert!(
        !remaining.contains(&old.uuid),
        "the old segment is replaced"
    );
    assert!(remaining.contains(&delta.uuid), "the delta is retained");
    let listed = dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .filter(|idx| idx.name == "staged")
        .filter_map(|idx| idx.fragment_bitmap.clone())
        .fold(RoaringBitmap::new(), |acc, bitmap| acc | bitmap);
    assert_eq!(listed, RoaringBitmap::from_iter([0u32, 10, 11, appended]));
    assert_queries_match_scans(&dataset, true).await;
}

/// A same-name CreateIndex in the window that changed what the index is (a
/// replacement of another index type) is a real conflict: the staged
/// segments are refused with the rebuild error.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_segments_refuse_an_incompatible_same_name_replacement() {
    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    let params = ScalarIndexParams::for_builtin(IndexType::BTree.try_into().unwrap());
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            Some("staged".into()),
            &params,
            false,
        )
        .await
        .unwrap();
    let mut staged = Vec::new();
    for fragment in [0u32, 1] {
        staged.push(
            CreateIndexBuilder::new(&mut dataset, &["i"], IndexType::BTree, &params)
                .name("staged".to_string())
                .fragments(vec![fragment])
                .replace(true)
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    reserve_fragments(&mut dataset, 20).await;
    let mut dataset = commit_stable_partition(dataset, &[1], 10).await;
    dataset
        .create_index(
            &["i"],
            IndexType::Bitmap,
            Some("staged".into()),
            &ScalarIndexParams::for_builtin(IndexType::Bitmap.try_into().unwrap()),
            true,
        )
        .await
        .unwrap();
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    let error = dataset
        .merge_existing_index_segments(staged)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("Rebuild"), "{error}");
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

// ---------------------------------------------------------------------------
// The maintenance chain after a v0 deferred-remap upgrade.
//
// Under the legacy v0 fragment reuse index a deferred compaction leaves the
// index FILE holding SOURCE row addresses while the listing swaps the
// segment's bitmap to the destination and the next commit persists that
// swap. The first stable partition lifts the v0 bytes verbatim into a tagged
// history, where the legacy version is an ordered-compaction transition. The
// tests below pin what `optimize_indices` (merge) and
// `cleanup_frag_reuse_index` (trim) do to such a segment afterwards.
// ---------------------------------------------------------------------------

/// Reopen `uri` in a fresh session: nothing cached from the writer survives.
async fn fresh_session(uri: &str) -> Dataset {
    crate::dataset::builder::DatasetBuilder::from_uri(uri)
        .with_session(Arc::new(crate::session::Session::default()))
        .load()
        .await
        .unwrap()
}

/// Append `rows` rows of the scalar fixture's three columns, `i` and `w`
/// counting up from `start`, as one new fragment.
async fn append_scalar_rows(dataset: Dataset, start: i32, rows: u64) -> Dataset {
    let batch = lance_datagen::gen_batch()
        .col(
            "i",
            lance_datagen::array::step_custom::<Int32Type>(start, 1),
        )
        .col(
            "text",
            lance_datagen::array::fill_utf8("document".to_string()),
        )
        .col(
            "w",
            lance_datagen::array::step_custom::<Int32Type>(start, 1),
        )
        .into_batch_rows(lance_datagen::RowCount::from(rows))
        .unwrap();
    crate::dataset::InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap()
}

/// Append `rows` rows of the vector fixture's three columns as one new
/// fragment.
async fn append_vector_rows(dataset: Dataset, start: i32, rows: u64) -> Dataset {
    // Every `gen_batch` starts from the same seed: reseed by `start` so no
    // appended vector duplicates a fixture vector (ties would make top-1
    // ambiguous).
    let batch = lance_datagen::gen_batch()
        .with_seed(lance_datagen::Seed(start as u64))
        .col(
            "i",
            lance_datagen::array::step_custom::<Int32Type>(start, 1),
        )
        .col(
            "vector",
            lance_datagen::array::rand_vec::<arrow_array::types::Float32Type>(4.into()),
        )
        .col(
            "w",
            lance_datagen::array::step_custom::<Int32Type>(start, 1),
        )
        .into_batch_rows(lance_datagen::RowCount::from(rows))
        .unwrap();
    crate::dataset::InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap()
}

fn fragment_ids(dataset: &Dataset) -> Vec<u32> {
    dataset.fragments().iter().map(|f| f.id as u32).collect()
}

/// The stored segments of the logical index `name`, in manifest order.
async fn stored_segments(dataset: &Dataset, name: &str) -> Vec<IndexMetadata> {
    crate::index::load_all_indices(dataset)
        .await
        .unwrap()
        .iter()
        .filter(|idx| idx.name == name)
        .cloned()
        .collect()
}

async fn fri_entry(dataset: &Dataset) -> Option<IndexMetadata> {
    crate::index::load_all_indices(dataset)
        .await
        .unwrap()
        .iter()
        .find(|idx| idx.name == lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME)
        .cloned()
}

/// The legacy (v0) compaction versions the entry still carries, whatever
/// its index version.
async fn legacy_versions(
    dataset: &Dataset,
) -> Vec<lance_table::format::pb::fragment_reuse_index_details::Version> {
    match fri_entry(dataset).await {
        None => Vec::new(),
        Some(entry) => {
            crate::index::frag_reuse::load_frag_reuse_records(dataset, &entry)
                .await
                .unwrap()
                .legacy_versions
        }
    }
}

/// The row address of every `i` in the live table.
async fn row_addrs_by_i(dataset: &Dataset) -> std::collections::HashMap<i32, u64> {
    let batch = dataset
        .scan()
        .project(&["i"])
        .unwrap()
        .with_row_address()
        .try_into_batch()
        .await
        .unwrap();
    let ids = batch["i"].as_primitive::<Int32Type>();
    let addrs = batch[lance_core::ROW_ADDR].as_primitive::<arrow_array::types::UInt64Type>();
    ids.values()
        .iter()
        .copied()
        .zip(addrs.values().iter().copied())
        .collect()
}

/// The row addresses a BTree segment's FILE stores for `i = value`: the
/// segment is loaded straight from its store, with no fragment reuse
/// translation, so this is what is on disk rather than what a query sees.
async fn raw_segment_row_addrs(dataset: &Dataset, segment: &IndexMetadata, value: i32) -> Vec<u64> {
    use crate::dataset::index::LanceIndexStoreExt;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::scalar::btree::BTreeIndexPlugin;
    use lance_index::scalar::lance_format::LanceIndexStore;
    use lance_index::scalar::registry::ScalarIndexPlugin;

    let store = LanceIndexStore::from_dataset_for_existing(dataset, segment)
        .await
        .unwrap();
    let index = BTreeIndexPlugin
        .load_index(
            Arc::new(store),
            &prost_types::Any::default(),
            None,
            &lance_core::cache::LanceCache::no_cache(),
        )
        .await
        .unwrap();
    let SearchResult::Exact(rows) = index
        .search(
            &SargableQuery::Equals(ScalarValue::Int32(Some(value))),
            &NoOpMetricsCollector,
        )
        .await
        .unwrap()
    else {
        panic!("expected an exact result");
    };
    let mut addrs: Vec<u64> = rows
        .true_rows()
        .row_addrs()
        .unwrap()
        .map(u64::from)
        .collect();
    addrs.sort_unstable();
    addrs
}

/// The v0 table: the two fixture fragments plus two 2-row appends (`i`
/// 100..104), `i_idx` over all four, then a real v0 deferred compaction of
/// the two small fragments into one destination. Stops there: the index is
/// NOT re-created, so its file still holds the source addresses.
///
/// Returns the table and the destination fragment id.
async fn v0_deferred_scalar_fixture(uri: &str) -> (Dataset, u32) {
    let dataset = disk_fixture(uri).await;
    let dataset = append_scalar_rows(dataset, 100, 2).await;
    let mut dataset = append_scalar_rows(dataset, 102, 2).await;
    assert_eq!(fragment_ids(&dataset), vec![0, 1, 2, 3]);
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let built = stored_segment(&dataset, "i_idx").await;
    assert_eq!(
        built.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1, 2, 3])
    );
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
    let destination = *fragment_ids(&dataset)
        .iter()
        .find(|id| **id > 3)
        .expect("the compaction produced a destination");
    assert_eq!(fragment_ids(&dataset), vec![0, 1, destination]);
    assert_v0_deferred_state(&dataset, "i_idx", &built, &[2, 3], destination).await;
    (dataset, destination)
}

/// The state a v0 deferred compaction of `sources` into `destination`
/// leaves: a v0 entry with one more legacy version, the segment listed over
/// the destination (bitmap swapped, file untouched) and still stamped with
/// its pre-compaction dataset version.
async fn assert_v0_deferred_state(
    dataset: &Dataset,
    name: &str,
    built: &IndexMetadata,
    sources: &[u32],
    destination: u32,
) {
    let entry = fri_entry(dataset).await.expect("a v0 entry");
    assert_eq!(entry.index_version, 0, "a v0 (legacy) entry");
    let versions = legacy_versions(dataset).await;
    let version = versions.last().expect("the legacy version");
    let old_ids: Vec<u64> = version
        .groups
        .iter()
        .flat_map(|g| g.old_fragments.iter().map(|d| d.id))
        .collect();
    let new_ids: Vec<u64> = version
        .groups
        .iter()
        .flat_map(|g| g.new_fragments.iter().map(|d| d.id))
        .collect();
    assert_eq!(
        old_ids,
        sources.iter().map(|id| *id as u64).collect::<Vec<_>>()
    );
    assert_eq!(new_ids, vec![destination as u64]);
    let segment = stored_segment(dataset, name).await;
    assert_eq!(segment.uuid, built.uuid, "the segment is not rewritten");
    let bitmap = segment.fragment_bitmap.as_ref().unwrap();
    assert!(
        bitmap.contains(destination),
        "the listed bitmap claims the destination: {bitmap:?}"
    );
    assert!(
        sources.iter().all(|id| !bitmap.contains(*id)),
        "the listed bitmap no longer names the sources: {bitmap:?}"
    );
    assert!(
        segment.dataset_version < version.dataset_version,
        "the segment ({}) predates the legacy version ({})",
        segment.dataset_version,
        version.dataset_version
    );
}

/// Upgrade the table to a tagged history with a stable partition of two
/// fresh fragments (`i` from `start`, two rows each) that no index covers:
/// the lift must not touch the index's lineage. Returns the table and the
/// two destination ids.
async fn upgrade_by_stable_partition(dataset: Dataset, start: i32) -> (Dataset, Vec<u32>) {
    let legacy_before = legacy_versions(&dataset).await;
    let dataset = append_scalar_rows(dataset, start, 2).await;
    let mut dataset = append_scalar_rows(dataset, start + 2, 2).await;
    upgrade_appended_pair(&mut dataset, legacy_before.len()).await
}

async fn upgrade_appended_pair(dataset: &mut Dataset, legacy_count: usize) -> (Dataset, Vec<u32>) {
    let ids = fragment_ids(dataset);
    let sources: Vec<u64> = ids[ids.len() - 2..].iter().map(|id| *id as u64).collect();
    let dest_base = *ids.iter().max().unwrap() as u64 + 1;
    reserve_fragments(dataset, 20).await;
    let dataset = commit_stable_partition(dataset.clone(), &sources, dest_base).await;
    let entry = fri_entry(&dataset).await.unwrap();
    assert_eq!(entry.index_version, 1, "the lift tags the entry");
    assert_eq!(
        legacy_versions(&dataset).await.len(),
        legacy_count,
        "the legacy versions are lifted verbatim"
    );
    let destinations = vec![dest_base as u32, dest_base as u32 + 1];
    assert!(
        destinations
            .iter()
            .all(|id| fragment_ids(&dataset).contains(id)),
        "{:?}",
        fragment_ids(&dataset)
    );
    (dataset, destinations)
}

/// Every indexed point query of the values `i` takes in the live table
/// equals the index-disabled scan, by identity and multiplicity, and the
/// full scan agrees with and without the index.
async fn assert_scalar_queries_match(dataset: &Dataset) {
    let live: Vec<i32> = values(dataset, None, false).await;
    assert_eq!(values(dataset, None, true).await, live);
    let mut distinct = live.clone();
    distinct.dedup();
    for value in distinct {
        let predicate = format!("i = {value}");
        let plan = dataset
            .scan()
            .filter(&predicate)
            .unwrap()
            .use_scalar_index(true)
            .explain_plan(false)
            .await
            .unwrap();
        assert!(plan.contains("ScalarIndexQuery"), "{predicate}: {plan}");
        assert_eq!(
            values(dataset, Some(&predicate), true).await,
            values(dataset, Some(&predicate), false).await,
            "{predicate}"
        );
    }
}

/// (T3) After the upgrade, `optimize_indices` merges the v0-deferred
/// segment with a newer one. The merge translates the addresses: the
/// merged FILE holds live addresses (the old file's source addresses are
/// gone) and indexed queries equal the scan after a fresh reopen. It keeps
/// the old stamp and provenance, though: the merged segment carries the
/// OLD segment's dataset version (the minimum of its sources) and the union
/// of the stored bitmaps, so the trim that follows KEEPS the legacy version
/// and rebuilds the entry bitmap to the legacy sources and destination. The
/// legacy version stays pinned until a remap advances the stamp; that is
/// the conservative side.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn v0_deferred_segment_merges_into_live_addresses_after_upgrade() {
    let dir = TempStrDir::default();
    let (dataset, destination) = v0_deferred_scalar_fixture(dir.as_str()).await;
    let old = stored_segment(&dataset, "i_idx").await;
    // The file still addresses the retired sources.
    let stale = raw_segment_row_addrs(&dataset, &old, 101).await;
    assert_eq!(stale.len(), 1);
    assert_eq!(RowAddress::from(stale[0]).fragment_id(), 2);

    let (dataset, sp_destinations) = upgrade_by_stable_partition(dataset, 300).await;
    // A newer segment over an appended fragment (and the partition's
    // destinations, which no segment covers either).
    let mut dataset = append_scalar_rows(dataset, 400, 4).await;
    let appended = *fragment_ids(&dataset).iter().max().unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    let segments = stored_segments(&dataset, "i_idx").await;
    assert_eq!(segments.len(), 2, "{segments:?}");
    assert_eq!(segments[0].uuid, old.uuid);
    let newer = segments[1].clone();
    assert_eq!(
        newer.fragment_bitmap.as_ref().unwrap(),
        &sp_destinations
            .iter()
            .copied()
            .chain([appended])
            .collect::<RoaringBitmap>()
    );

    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let segments = stored_segments(&dataset, "i_idx").await;
    assert_eq!(segments.len(), 1, "one merged segment: {segments:?}");
    let merged = segments[0].clone();
    assert_ne!(merged.uuid, old.uuid);
    assert_ne!(merged.uuid, newer.uuid);
    // The merged file holds live addresses: the compacted rows now point at
    // the destination, everything else at its (still live) fragment.
    let live = row_addrs_by_i(&dataset).await;
    for (value, addr) in &live {
        assert_eq!(
            raw_segment_row_addrs(&dataset, &merged, *value).await,
            vec![*addr],
            "i = {value}"
        );
    }
    for value in 100..104 {
        assert_eq!(RowAddress::from(live[&value]).fragment_id(), destination);
    }
    assert_eq!(
        merged.dataset_version, old.dataset_version,
        "the merge keeps the oldest source stamp (below the legacy version)"
    );
    assert!(merged.dataset_version < legacy_versions(&dataset).await[0].dataset_version);
    // Provenance union: the old segment's listed bitmap {0, 1, destination}
    // and the newer segment's fragments (the two partition destinations and
    // the appended fragment).
    assert_eq!(
        old.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1, destination])
    );
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &(old.fragment_bitmap.clone().unwrap() | newer.fragment_bitmap.clone().unwrap()),
        "the merge keeps the provenance union"
    );
    assert_scalar_queries_match(&dataset).await;

    // The merged segment is still stamped before the legacy version: the
    // trim keeps it and rebuilds the entry bitmap to the legacy sources and
    // destination.
    let mut dataset = dataset;
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert_eq!(
        legacy_versions(&dataset).await.len(),
        1,
        "the legacy version stays pinned until a remap advances the stamp"
    );
    assert_eq!(
        fri_entry(&dataset).await.unwrap().fragment_bitmap.unwrap(),
        RoaringBitmap::from_iter([2u32, 3, destination])
    );
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        stored_segments(&dataset, "i_idx").await[0].uuid,
        merged.uuid
    );
    assert_scalar_queries_match(&dataset).await;
}

/// (Multi-round) Two v0 deferred compactions before the upgrade, the second
/// consuming the first's destination together with the other indexed
/// fragments, then the same merge and trim checks: the merged file holds
/// live addresses two hops away from what the old file stored, while the
/// merge keeps the old stamp and provenance, so both legacy versions stay
/// pinned (the conservative side) until a remap advances the stamp.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn two_v0_deferred_rounds_merge_and_trim_after_upgrade() {
    let dir = TempStrDir::default();
    let (mut dataset, first_destination) = v0_deferred_scalar_fixture(dir.as_str()).await;
    let old = stored_segment(&dataset, "i_idx").await;
    // Round two: every remaining fragment (F0, F1 and the first destination)
    // is indexed, so the planner bins them together.
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 16,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let ids = fragment_ids(&dataset);
    assert_eq!(ids.len(), 1, "{ids:?}");
    let second_destination = ids[0];
    assert!(second_destination > first_destination);
    assert_v0_deferred_state(
        &dataset,
        "i_idx",
        &old,
        &[0, 1, first_destination],
        second_destination,
    )
    .await;
    assert_eq!(legacy_versions(&dataset).await.len(), 2);
    assert_eq!(
        stored_segment(&dataset, "i_idx")
            .await
            .fragment_bitmap
            .as_ref()
            .unwrap(),
        &RoaringBitmap::from_iter([second_destination])
    );
    // The file is untouched: still the original source addresses.
    let stale = raw_segment_row_addrs(&dataset, &old, 101).await;
    assert_eq!(RowAddress::from(stale[0]).fragment_id(), 2);
    let stale = raw_segment_row_addrs(&dataset, &old, 5).await;
    assert_eq!(RowAddress::from(stale[0]).fragment_id(), 1);

    let (dataset, _) = upgrade_by_stable_partition(dataset, 300).await;
    let mut dataset = append_scalar_rows(dataset, 400, 4).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    let segments = stored_segments(&dataset, "i_idx").await;
    assert_eq!(segments.len(), 2, "{segments:?}");
    let old_listed = segments[0].clone();
    assert_eq!(old_listed.uuid, old.uuid);
    assert_eq!(
        old_listed.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([second_destination])
    );
    let newer = segments[1].clone();
    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let segments = stored_segments(&dataset, "i_idx").await;
    assert_eq!(segments.len(), 1, "{segments:?}");
    let merged = segments[0].clone();
    let live = row_addrs_by_i(&dataset).await;
    for (value, addr) in &live {
        assert_eq!(
            raw_segment_row_addrs(&dataset, &merged, *value).await,
            vec![*addr],
            "i = {value}"
        );
    }
    for value in (0..8).chain(100..104) {
        assert_eq!(
            RowAddress::from(live[&value]).fragment_id(),
            second_destination,
            "i = {value} lives in the second destination"
        );
    }
    assert_eq!(
        merged.dataset_version, old.dataset_version,
        "the merge keeps the oldest source stamp"
    );
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &(old_listed.fragment_bitmap.clone().unwrap() | newer.fragment_bitmap.clone().unwrap()),
        "the merge keeps the provenance union"
    );
    assert_scalar_queries_match(&dataset).await;

    // Both legacy versions stay pinned; the entry bitmap is rebuilt to the
    // sources and destinations of both rounds.
    let mut dataset = dataset;
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert_eq!(legacy_versions(&dataset).await.len(), 2);
    assert_eq!(
        fri_entry(&dataset).await.unwrap().fragment_bitmap.unwrap(),
        RoaringBitmap::from_iter([0u32, 1, 2, 3, first_destination, second_destination])
    );
    let dataset = fresh_session(dir.as_str()).await;
    assert_scalar_queries_match(&dataset).await;
}

/// (Mixed catch-up) Next to the v0-deferred segment sits a segment built
/// after the upgrade over appended fragments. Maintenance that does not
/// select the caught-up segment (the trim, and a merge that selects only
/// the trailing segment with nothing new to add) leaves its uuid, bitmap
/// and dataset version alone; the legacy version stays pinned by the old
/// segment meanwhile. The full merge then covers both provenances and the
/// index answers from a fresh session.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn upgrade_merge_leaves_the_caught_up_segment_alone() {
    let dir = TempStrDir::default();
    let (dataset, destination) = v0_deferred_scalar_fixture(dir.as_str()).await;
    let old = stored_segment(&dataset, "i_idx").await;
    let (dataset, sp_destinations) = upgrade_by_stable_partition(dataset, 300).await;
    let mut dataset = append_scalar_rows(dataset, 400, 4).await;
    let appended = *fragment_ids(&dataset).iter().max().unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    let segments = stored_segments(&dataset, "i_idx").await;
    assert_eq!(segments.len(), 2, "{segments:?}");
    let caught_up = segments[1].clone();
    assert_ne!(caught_up.uuid, old.uuid);
    assert_eq!(
        caught_up.fragment_bitmap.as_ref().unwrap(),
        &sp_destinations
            .iter()
            .copied()
            .chain([appended])
            .collect::<RoaringBitmap>()
    );
    assert!(
        caught_up.dataset_version > legacy_versions(&dataset).await[0].dataset_version,
        "the new segment is caught up past the legacy version"
    );

    // The trim: the old segment still pins the legacy version; the caught-up
    // segment is untouched.
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert_eq!(legacy_versions(&dataset).await.len(), 1);
    let segments = stored_segments(&dataset, "i_idx").await;
    assert_eq!(segments.len(), 2, "{segments:?}");
    assert_eq!(segments[0].uuid, old.uuid);
    assert_eq!(segments[0].fragment_bitmap, old.fragment_bitmap);
    assert_eq!(segments[0].dataset_version, old.dataset_version);
    assert_eq!(segments[1].uuid, caught_up.uuid);
    assert_eq!(segments[1].fragment_bitmap, caught_up.fragment_bitmap);
    assert_eq!(segments[1].dataset_version, caught_up.dataset_version);

    // A merge selecting only the trailing segment with nothing unindexed
    // rewrites nothing.
    let version_before = dataset.manifest.version;
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(dataset.manifest.version, version_before);
    let segments = stored_segments(&dataset, "i_idx").await;
    assert_eq!(
        segments.iter().map(|s| s.uuid).collect::<Vec<_>>(),
        vec![old.uuid, caught_up.uuid]
    );

    // The full merge consumes both: the merged provenance covers the old
    // segment's live coverage and the caught-up segment's fragments.
    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let segments = stored_segments(&dataset, "i_idx").await;
    assert_eq!(segments.len(), 1, "{segments:?}");
    let merged = &segments[0];
    assert_ne!(merged.uuid, caught_up.uuid);
    let merged_bitmap = merged.fragment_bitmap.as_ref().unwrap();
    assert!(
        merged_bitmap.is_superset(caught_up.fragment_bitmap.as_ref().unwrap()),
        "{merged_bitmap:?}"
    );
    assert!(merged_bitmap.contains(destination), "{merged_bitmap:?}");
    assert!(
        merged_bitmap.contains(0) && merged_bitmap.contains(1),
        "{merged_bitmap:?}"
    );
    let live = row_addrs_by_i(&dataset).await;
    for (value, addr) in &live {
        assert_eq!(
            raw_segment_row_addrs(&dataset, merged, *value).await,
            vec![*addr],
            "i = {value}"
        );
    }
    assert_scalar_queries_match(&dataset).await;
}

/// (T4') The vector twin over IVF_FLAT: the vector merge copies addresses
/// as-is with `dataset_version = min(source versions)` and the union of
/// the stored bitmaps; the trim that follows must KEEP the legacy version
/// (the merged segment is still stale against it), and the top-1 neighbour
/// of a compacted row's own vector is that row after a fresh reopen.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn v0_deferred_vector_segment_merge_keeps_the_legacy_version_pinned() {
    let dir = TempStrDir::default();
    let dataset = vector_fixture(dir.as_str()).await;
    let dataset = append_vector_rows(dataset, 100, 2).await;
    let mut dataset = append_vector_rows(dataset, 102, 2).await;
    assert_eq!(fragment_ids(&dataset), vec![0, 1, 2, 3]);
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
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vec_idx".into()),
            &params,
            false,
        )
        .await
        .unwrap();
    let old = stored_segment(&dataset, "vec_idx").await;
    // The query: the vector of a row the compaction moves.
    let original = dataset
        .scan()
        .filter("i = 101")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let query = original["vector"].as_fixed_size_list().value(0);
    let query = query
        .as_primitive::<arrow_array::types::Float32Type>()
        .clone();

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
    let destination = *fragment_ids(&dataset).iter().max().unwrap();
    assert_eq!(fragment_ids(&dataset), vec![0, 1, destination]);
    assert_v0_deferred_state(&dataset, "vec_idx", &old, &[2, 3], destination).await;
    let (_, found) = nearest(&dataset, &query, 1, true).await;
    assert_eq!(found, vec![101], "the v0 window still answers");

    // Upgrade on fragments outside the index's lineage.
    let dataset = append_vector_rows(dataset, 300, 2).await;
    let mut dataset = append_vector_rows(dataset, 302, 2).await;
    let (dataset, sp_destinations) = upgrade_appended_pair(&mut dataset, 1).await;
    // A newer segment over an appended fragment and the partition's
    // destinations.
    let mut dataset = append_vector_rows(dataset, 400, 4).await;
    let appended = *fragment_ids(&dataset).iter().max().unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    let segments = stored_segments(&dataset, "vec_idx").await;
    assert_eq!(segments.len(), 2, "{segments:?}");
    assert_eq!(segments[0].uuid, old.uuid);
    let newer = segments[1].clone();
    assert_eq!(
        newer.fragment_bitmap.as_ref().unwrap(),
        &sp_destinations
            .iter()
            .copied()
            .chain([appended])
            .collect::<RoaringBitmap>()
    );
    let old_listed = segments[0].clone();

    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let segments = stored_segments(&dataset, "vec_idx").await;
    assert_eq!(segments.len(), 1, "{segments:?}");
    let merged = segments[0].clone();
    assert_ne!(merged.uuid, old.uuid);
    assert_eq!(
        merged.dataset_version,
        old.dataset_version.min(newer.dataset_version),
        "the vector merge keeps the oldest source version"
    );
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &(old_listed.fragment_bitmap.clone().unwrap() | newer.fragment_bitmap.clone().unwrap()),
        "the vector merge keeps the union of the stored bitmaps"
    );
    let (plan, found) = nearest(&dataset, &query, 1, true).await;
    assert!(plan.contains("ANN"), "{plan}");
    assert_eq!(found, vec![101]);

    // The merged segment is still stale against the legacy version: the
    // trim keeps it.
    let mut dataset = dataset;
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert_eq!(
        legacy_versions(&dataset).await.len(),
        1,
        "the legacy version stays pinned"
    );
    assert_eq!(
        stored_segments(&dataset, "vec_idx").await[0].uuid,
        merged.uuid
    );

    let dataset = fresh_session(dir.as_str()).await;
    let (plan, found) = nearest(&dataset, &query, 1, true).await;
    assert!(plan.contains("ANN"), "{plan}");
    assert_eq!(found, vec![101]);
    let total = dataset.count_rows(None).await.unwrap();
    let (_, flat) = nearest(&dataset, &query, total, false).await;
    let (_, indexed) = nearest(&dataset, &query, total, true).await;
    assert_eq!(flat.len(), total);
    assert_eq!(indexed, flat, "every row is reachable through the index");
}
