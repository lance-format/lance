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
use crate::dataset::optimize::tests::undeferrable_withdrawal::open_with_failing_reads;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::write::CommitBuilder;
use crate::index::frag_reuse_reader::tests as reader_tests;
use crate::index::loading::DatasetIndexInternalExt;
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
use lance_core::utils::tempfile::TempStrDir;
use lance_index::metrics::NoOpMetricsCollector;
use lance_table::format::Fragment;
use lance_table::transaction::{Operation, RewriteGroup, Transaction};
use prost::Message;
use roaring::RoaringBitmap;
use std::collections::HashSet;
use std::sync::atomic::Ordering;
use uuid::Uuid;
#[cfg(feature = "geo")]
use {
    crate::utils::test::geo,
    arrow_array::{Int32Array, RecordBatch, RecordBatchIterator},
    arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema},
};

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
        // Later segments join an index that already has committed segments.
        .replace(true)
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
    crate::index::loading::load_all_indices(dataset)
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

/// Bloom filter segments on a tagged history are not merged by their zones,
/// which that merge cannot translate: a bounded plan keeps the index in one
/// task, and optimize rebuilds it over the destinations.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn bloom_filter_segments_rebuild_on_a_tagged_table() {
    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    let s1 = staged_segment(&mut dataset, "i", IndexType::BloomFilter, vec![0]).await;
    let s2 = staged_segment(&mut dataset, "i", IndexType::BloomFilter, vec![1]).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![s1, s2])
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let mut dataset = commit_stable_partition(dataset, &[0, 1], 10).await;
    assert_eq!(fri_entry(&dataset).await.unwrap().index_version, 1);

    let bounded = OptimizeOptions::merge(2).max_rows_per_task(Some(4));
    let plan = crate::index::optimize::plan_optimize_indices(&dataset, &bounded)
        .await
        .unwrap();
    assert_eq!(plan.tasks.len(), 1);
    assert_eq!(plan.tasks[0].old_segments.len(), 2);

    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        dataset.load_indices_by_name("staged").await.unwrap().len(),
        1
    );
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

/// A staged inverted segment whose metadata is still only under `staging/`:
/// the fragment-masked distributed build writes partition metadata there and
/// the root `metadata.lance` is published by the finalize that the commit
/// runs. Reading that absence as the LEGACY layout makes the staged replay's
/// withdrawal selector treat a modern segment as untranslatable, so a
/// deferred compaction committed in the build's window withdraws exactly the
/// coverage the merge is about to claim.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_inverted_segments_merge_across_deferred_compaction_on_tagged_table() {
    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[0, 1], 10).await;
    // The appended rows the staged build indexes. Two hold the query term in
    // another case form or inflection, which only the index's own analyzer
    // (lowercasing and stemming) reaches; the bare tokenizer the flat path
    // falls back to matches neither.
    let texts = ["Running", "RUNS", "walked", "walking"];
    let batch = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step_custom::<Int32Type>(8, 1))
        .col("text", lance_datagen::array::cycle_utf8_literals(&texts))
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

    // The distributed build: one fragment-masked worker segment each, never
    // finalized, so neither has a root metadata file yet.
    let s1 = staged_inverted_segment(&mut dataset, "text", vec![appended[0]]).await;
    let s2 = staged_inverted_segment(&mut dataset, "text", vec![appended[1]]).await;
    for segment in [&s1, &s2] {
        assert!(
            !root_metadata_exists(&dataset, segment).await,
            "the staged build must not have published a root metadata file yet"
        );
    }

    // The row set the staged configuration's analyzer defines for this query.
    // Taken against the fallback tokenizer first: with no committed index on
    // the column the flat path answers with a STRICTLY SMALLER set, so a
    // withdrawn segment is detectable in the row set and not merely in the
    // coverage bitmap.
    let expected_ids = vec![8, 9];
    let (fallback_ids, fallback_plan) = fts_ids(&dataset, "text", ANALYZER_SENSITIVE_QUERY).await;
    assert!(
        fallback_plan
            .lines()
            .any(|line| line.trim_start().starts_with("FlatMatchQuery:")),
        "with no committed index on the column this baseline must come from \
         the flat (fallback tokenizer) path: {fallback_plan}"
    );
    assert_ne!(
        fallback_ids, expected_ids,
        "the fallback tokenizer must answer differently from the staged \
         analyzer, or the test cannot tell the two configurations apart: \
         {fallback_plan}"
    );

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
    let destination: RoaringBitmap = compacted.iter().copied().collect();

    let merged = dataset
        .merge_existing_index_segments(vec![s1, s2])
        .await
        .unwrap();
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &destination,
        "the staged replay must not withdraw a modern segment's coverage for \
         a root metadata file the build has not published yet"
    );
    dataset
        .commit_existing_index_segments("staged", "text", vec![merged])
        .await
        .unwrap();

    // Immediately after the replay committed, on the handle that performed
    // it: the row set must already be the staged configuration's, so the
    // answer never passes through the default tokenizer.
    let (replayed_ids, replayed_plan) = fts_ids(&dataset, "text", ANALYZER_SENSITIVE_QUERY).await;
    assert_eq!(
        replayed_ids, expected_ids,
        "the replayed segment must answer with its own tokenizer's row set: \
         {replayed_plan}"
    );

    // Finalization happened at commit time, so the committed segment now has
    // its root metadata and the reader derives the compacted coverage.
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert!(
        root_metadata_exists(&dataset, &stored_segment(&dataset, "staged").await).await,
        "the commit must have finalized the staged segment"
    );
    assert_eq!(
        derived_coverage(&dataset, "staged").await,
        Some(destination)
    );

    // The same row set the staged configuration defined, now answered through
    // the finalized index rather than by a stemming flat scan.
    let (final_ids, final_plan) = fts_ids(&dataset, "text", ANALYZER_SENSITIVE_QUERY).await;
    assert!(
        final_plan
            .lines()
            .any(|line| line.trim_start().starts_with("MatchQuery:")),
        "the finalized segment must serve full-text search from the index: \
         {final_plan}"
    );
    assert_eq!(
        final_ids, expected_ids,
        "the committed index must answer with the row set its own tokenizer \
         defines, not the default tokenizer's stemmed superset: {final_plan}"
    );
}

/// The staged build's inverted configuration. Its analyzer lowercases and
/// stems; the tokenizer the flat path falls back to when no segment is
/// available is a bare simple tokenizer that does neither. So a query in one
/// case form matches a differently-cased or inflected document only through
/// the index, which is what makes a withdrawn segment visible in the ROW SET
/// and not merely in the coverage bitmap.
fn analyzed_params() -> InvertedIndexParams {
    InvertedIndexParams::default()
        .lower_case(true)
        .stem(true)
        .remove_stop_words(false)
}

/// A lowercase query whose matches in the fixture are written in other case
/// forms, so only the index's own analyzer finds them.
const ANALYZER_SENSITIVE_QUERY: &str = "running";

/// A fragment-masked uncommitted inverted build, as a distributed worker
/// produces it.
async fn staged_inverted_segment(
    dataset: &mut Dataset,
    column: &str,
    fragments: Vec<u32>,
) -> IndexMetadata {
    let params = analyzed_params();
    CreateIndexBuilder::new(dataset, &[column], IndexType::Inverted, &params)
        .name("staged".to_string())
        .fragments(fragments)
        .execute_uncommitted()
        .await
        .unwrap()
}

/// The sorted `i` values full-text search returns for `query` on `column`,
/// with the plan, so a test can compare the ACTUAL matching row set rather
/// than a count.
async fn fts_ids(dataset: &Dataset, column: &str, query: &str) -> (Vec<i32>, String) {
    let mut scan = dataset.scan();
    scan.full_text_search(
        FullTextSearchQuery::new(query.to_string())
            .with_column(column.to_string())
            .unwrap(),
    )
    .unwrap();
    scan.project(&["i"]).unwrap();
    let plan = scan.explain_plan(false).await.unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let mut ids: Vec<i32> = batch["i"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect();
    ids.sort_unstable();
    (ids, plan)
}

/// Whether the segment has published the root metadata file that marks a
/// finalized inverted index.
async fn root_metadata_exists(dataset: &Dataset, segment: &IndexMetadata) -> bool {
    let path = dataset
        .indices_dir()
        .join(segment.uuid.to_string())
        .join(lance_index::scalar::inverted::METADATA_FILE);
    dataset.object_store.exists(&path).await.unwrap()
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
        crate::index::loading::load_all_indices(&dataset)
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
    let delta = crate::index::loading::load_all_indices(&dataset)
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
    let remaining: Vec<Uuid> = crate::index::loading::load_all_indices(&dataset)
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
        crate::index::loading::load_all_indices(&dataset)
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
    crate::index::loading::load_all_indices(dataset)
        .await
        .unwrap()
        .iter()
        .filter(|idx| idx.name == name)
        .cloned()
        .collect()
}

async fn fri_entry(dataset: &Dataset) -> Option<IndexMetadata> {
    crate::index::loading::load_all_indices(dataset)
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
            segment.index_version as u32,
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
/// gone) and indexed queries equal the scan after a fresh reopen. The
/// merged segment is stamped with the manifest it was built from and keeps
/// the union of the stored bitmaps, so the trim that follows releases the
/// legacy version: the merged stamp is past it and its retired sources are
/// in no segment's bitmap.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn v0_deferred_segment_merges_into_live_addresses_after_upgrade() {
    let dir = TempStrDir::default();
    let (dataset, destination) = Box::pin(v0_deferred_scalar_fixture(dir.as_str())).await;
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

    let built_from = dataset.manifest.version;
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
        merged.dataset_version, built_from,
        "the merge is stamped with the manifest it was built from"
    );
    assert!(merged.dataset_version > legacy_versions(&dataset).await[0].dataset_version);
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

    // The merged segment is past the legacy version and covers none of its
    // retired sources: the trim releases it, and with nothing left to
    // retain, drops the entry.
    let mut dataset = dataset;
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert!(
        legacy_versions(&dataset).await.is_empty(),
        "the legacy version is released once the merged stamp is past it"
    );
    assert!(fri_entry(&dataset).await.is_none());
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
/// live addresses two hops away from what the old file stored, the merge
/// is stamped with the manifest it was built from, and the trim releases
/// both legacy versions.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn two_v0_deferred_rounds_merge_and_trim_after_upgrade() {
    let dir = TempStrDir::default();
    let (mut dataset, first_destination) = Box::pin(v0_deferred_scalar_fixture(dir.as_str())).await;
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
    let built_from = dataset.manifest.version;
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
        merged.dataset_version, built_from,
        "the merge is stamped with the manifest it was built from"
    );
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &(old_listed.fragment_bitmap.clone().unwrap() | newer.fragment_bitmap.clone().unwrap()),
        "the merge keeps the provenance union"
    );
    assert_scalar_queries_match(&dataset).await;

    // The merged segment is past both legacy versions and covers none of
    // their retired sources: the trim releases both and drops the entry.
    let mut dataset = dataset;
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert!(legacy_versions(&dataset).await.is_empty());
    assert!(fri_entry(&dataset).await.is_none());
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
    let (dataset, destination) = Box::pin(v0_deferred_scalar_fixture(dir.as_str())).await;
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

/// A committed bloom filter segment over a source a stable partition retired,
/// which the reader translates, and over a live fragment rewritten in place:
/// the default optimize indexes that fragment into the segment again. The
/// rewritten fragment's old null rows go, and the retired source's zones and
/// null rows are translated rather than dropped.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn bloom_filter_optimize_drops_rewritten_null_rows_on_a_tagged_history() {
    let dir = TempStrDir::default();
    // Three fragments of four rows; `value` is null where `i % 4 == 1`.
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("i", arrow_schema::DataType::Int32, false),
        arrow_schema::Field::new("value", arrow_schema::DataType::Int32, true),
        arrow_schema::Field::new("w", arrow_schema::DataType::Int32, false),
    ]));
    let batch = arrow_array::RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(arrow_array::Int32Array::from_iter_values(0..12)),
            Arc::new(arrow_array::Int32Array::from_iter(
                (0..12).map(|i| (i % 4 != 1).then_some(i)),
            )),
            Arc::new(arrow_array::Int32Array::from_iter_values(0..12)),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        arrow_array::RecordBatchIterator::new(vec![Ok(batch)], schema),
        dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["value"],
            IndexType::BloomFilter,
            Some("value_idx".into()),
            &ScalarIndexParams::for_builtin(IndexType::BloomFilter.try_into().unwrap()),
            false,
        )
        .await
        .unwrap();

    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[0], 10).await;
    assert_eq!(fri_entry(&dataset).await.unwrap().index_version, 1);
    let mut fragments = fragment_ids(&dataset);
    fragments.sort_unstable();
    assert_eq!(fragments, vec![1, 2, 10, 11]);
    let stored = stored_segment(&dataset, "value_idx").await;
    assert_eq!(
        stored.fragment_bitmap.unwrap(),
        RoaringBitmap::from_iter([0u32, 1, 2])
    );

    // `w = 5` is the null of fragment 1.
    rewrite_in_place(
        dir.as_str(),
        "value",
        5,
        Arc::new(arrow_array::Int32Array::from(vec![500])),
    )
    .await;
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        derived_coverage(&dataset, "value_idx").await,
        Some(RoaringBitmap::from_iter([2u32, 10, 11]))
    );

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(stored_segments(&dataset, "value_idx").await.len(), 1);
    // Merged rather than rebuilt: the segment keeps the provenance it was
    // stored with, where a rebuild would claim the live fragments.
    assert_eq!(
        stored_segment(&dataset, "value_idx")
            .await
            .fragment_bitmap
            .unwrap(),
        RoaringBitmap::from_iter([0u32, 1, 2])
    );
    let plan = dataset
        .scan()
        .filter("value IS NULL")
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();
    assert!(plan.contains("@value_idx(BloomFilter)"), "{plan}");
    assert_eq!(
        values(&dataset, Some("value IS NULL"), false).await,
        vec![1, 9]
    );
    assert_eq!(
        values(&dataset, Some("value IS NULL"), true).await,
        vec![1, 9]
    );
    for value in [0, 2, 3, 4, 6, 7, 8, 10, 11, 500] {
        let predicate = format!("value = {value}");
        assert_eq!(
            values(&dataset, Some(&predicate), true).await,
            values(&dataset, Some(&predicate), false).await,
            "{predicate}"
        );
    }
}

/// RTree on a tagged history: segments covering a stable partition's sources
/// claim its destinations and hold every row of them.
#[cfg(feature = "geo")]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_rtree_segments_merge_after_stable_partition() {
    const ROWS_PER_FRAGMENT: i32 = 4;
    let dir = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("i", DataType::Int32, false),
        geo::line_string_type().to_field("geometry", true),
    ]));
    let batches = (0..2)
        .map(|fragment| {
            let first = fragment * ROWS_PER_FRAGMENT;
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(
                        first..first + ROWS_PER_FRAGMENT,
                    )),
                    geo::line_strings(first, ROWS_PER_FRAGMENT),
                ],
            )
        })
        .collect::<Vec<_>>();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(batches, schema),
        dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FRAGMENT as usize,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::RTree);
    let sources = fragment_ids(&dataset);
    let staged = geo::stage_rtree_segments(&mut dataset, &params, sources).await;
    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[0, 1], 10).await;

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([10u32, 11]),
        "the merged segment claims the destinations the group covers completely"
    );
    let mut dataset = dataset;
    dataset
        .commit_existing_index_segments(&merged.name, "geometry", vec![merged.clone()])
        .await
        .unwrap();
    let committed = stored_segment(&dataset, &merged.name).await;
    assert_eq!(
        derived_coverage(&dataset, &merged.name).await,
        Some(RoaringBitmap::from_iter([10u32, 11])),
        "the committed segment serves both destinations directly"
    );
    let index = crate::index::scalar::open_scalar_index(
        &dataset,
        "geometry",
        &committed,
        &NoOpMetricsCollector,
    )
    .await
    .unwrap();
    assert_eq!(
        index.statistics().unwrap()["num_items"].as_u64().unwrap(),
        (ROWS_PER_FRAGMENT * 2) as u64,
        "the merged index must hold every row of the fragments it claims"
    );
}

// ---------------------------------------------------------------------------
// Committing a segment next to historical segments: segments whose stored
// fragments were all retired by a rewrite but which still serve the rewrite's
// destinations through the tagged history.
// ---------------------------------------------------------------------------

/// Index `staged` with segment P over fragment 0 and Q over fragment 1, a
/// stable partition of {0, 1} into {10, 11} that P and Q serve together
/// through the history, and one unrelated live fragment appended after it.
/// Returns the dataset, the uuids of P and Q, and the appended fragment.
async fn historical_fixture(
    uri: &str,
    column: &str,
    index_type: IndexType,
) -> (Dataset, [Uuid; 2], u32) {
    let mut dataset = disk_fixture(uri).await;
    let p = staged_segment(&mut dataset, column, index_type, vec![0]).await;
    let q = staged_segment(&mut dataset, column, index_type, vec![1]).await;
    dataset
        .commit_existing_index_segments("staged", column, vec![p.clone(), q.clone()])
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[0, 1], 10).await;
    let dataset = append_scalar_rows(dataset, 8, 4).await;
    let mut fragments = fragment_ids(&dataset);
    fragments.sort_unstable();
    let appended = *fragments.last().unwrap();
    assert_eq!(fragments, vec![10, 11, appended]);
    (dataset, [p.uuid, q.uuid], appended)
}

async fn stored_uuids(dataset: &Dataset, name: &str) -> HashSet<Uuid> {
    stored_segments(dataset, name)
        .await
        .iter()
        .map(|segment| segment.uuid)
        .collect()
}

async fn unindexed_fragment_ids(dataset: &Dataset, name: &str) -> Vec<u32> {
    dataset
        .unindexed_fragments(name)
        .await
        .unwrap()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect()
}

async fn transition_count(dataset: &Dataset) -> usize {
    match fri_entry(dataset).await {
        None => 0,
        Some(entry) => crate::index::frag_reuse::decode_frag_reuse_ledger(dataset, &entry)
            .await
            .unwrap()
            .transitions()
            .len(),
    }
}

/// A segment over an unrelated fragment leaves the destinations to the
/// historical segments: P and Q are kept and still serve {10, 11}, so the
/// trim keeps the transition. Once a segment covers the destinations
/// directly, it replaces P and Q and the trim releases the transition.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn partial_commit_keeps_historical_segments_that_serve_destinations() {
    let dir = TempStrDir::default();
    let (mut dataset, [p, q], appended) =
        historical_fixture(dir.as_str(), "i", IndexType::BTree).await;
    let incoming = staged_segment(&mut dataset, "i", IndexType::BTree, vec![appended]).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![incoming.clone()])
        .await
        .unwrap();

    let mut dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([p, q, incoming.uuid]),
        "the historical segments still serve the destinations"
    );
    assert!(unindexed_fragment_ids(&dataset, "staged").await.is_empty());
    let derived = dataset.load_indices().await.unwrap();
    assert_eq!(
        derived
            .iter()
            .find(|idx| idx.uuid == incoming.uuid)
            .and_then(|idx| idx.fragment_bitmap.clone()),
        Some(RoaringBitmap::from_iter([appended]))
    );
    assert_queries_match_scans(&dataset, true).await;
    assert_eq!(values(&dataset, Some("i = 9"), true).await, vec![9]);

    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([p, q, incoming.uuid]),
        "the trim prunes no segment that still serves a destination"
    );
    assert_eq!(transition_count(&dataset).await, 1);

    let direct = staged_segment(&mut dataset, "i", IndexType::BTree, vec![10, 11]).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![direct.clone()])
        .await
        .unwrap();
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([incoming.uuid, direct.uuid])
    );
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert_eq!(transition_count(&dataset).await, 0);
    assert_queries_match_scans(&dataset, true).await;
}

/// A commit covering everything the historical segments serve replaces them,
/// whether it covers only the destinations or every live fragment.
#[rstest::rstest]
#[case::destinations(false)]
#[case::every_live_fragment(true)]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn commit_covering_the_destinations_replaces_historical_segments(
    #[case] covers_appended: bool,
) {
    let dir = TempStrDir::default();
    let (mut dataset, _, appended) = historical_fixture(dir.as_str(), "i", IndexType::BTree).await;
    let mut fragments = vec![10, 11];
    if covers_appended {
        fragments.push(appended);
    }
    let incoming = staged_segment(&mut dataset, "i", IndexType::BTree, fragments.clone()).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![incoming.clone()])
        .await
        .unwrap();

    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([incoming.uuid])
    );
    assert_eq!(
        derived_coverage(&dataset, "staged").await,
        Some(fragments.into_iter().collect::<RoaringBitmap>())
    );
    assert_queries_match_scans(&dataset, true).await;
}

/// A type without a batch remapper cannot translate through the history, so
/// the rewrite withdraws its coverage of the retired fragments: the
/// historical segments are left with empty stored bitmaps and serve nothing.
/// A partial commit removes them like any definition-only segment, and the
/// destinations stay unindexed.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn partial_commit_removes_historical_segments_whose_coverage_was_withdrawn() {
    let dir = TempStrDir::default();
    let (mut dataset, [p, q], appended) =
        historical_fixture(dir.as_str(), "text", IndexType::Fm).await;
    let historical: Vec<_> = stored_segments(&dataset, "staged")
        .await
        .into_iter()
        .filter(|segment| segment.uuid == p || segment.uuid == q)
        .collect();
    assert_eq!(historical.len(), 2);
    assert!(
        historical.iter().all(|segment| segment
            .fragment_bitmap
            .as_ref()
            .is_some_and(RoaringBitmap::is_empty)),
        "the rewrite withdraws coverage an FM segment cannot translate"
    );

    let incoming = staged_segment(&mut dataset, "text", IndexType::Fm, vec![appended]).await;
    dataset
        .commit_existing_index_segments("staged", "text", vec![incoming.clone()])
        .await
        .unwrap();

    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([incoming.uuid])
    );
    let mut unindexed = unindexed_fragment_ids(&dataset, "staged").await;
    unindexed.sort_unstable();
    assert_eq!(unindexed, vec![10, 11]);
}

/// A kept historical segment outlives the commit, so the incoming segments
/// must declare the same fields as it does.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn partial_commit_refuses_a_declaration_a_historical_segment_does_not_share() {
    let dir = TempStrDir::default();
    let (mut dataset, [p, q], appended) =
        historical_fixture(dir.as_str(), "i", IndexType::BTree).await;
    let w = dataset.schema().field_id("w").unwrap();
    let incoming = staged_segment(&mut dataset, "i", IndexType::BTree, vec![appended]).await;
    let incoming = IndexMetadata {
        covering_fields: vec![w],
        ..incoming
    };

    let error = dataset
        .commit_existing_index_segments("staged", "i", vec![incoming])
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("cannot mix declarations"),
        "unexpected error: {error}"
    );
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([p, q])
    );
}

/// Without a fragment reuse history a segment over deleted fragments covers
/// nothing, and a partial commit removes it.
#[tokio::test]
async fn partial_commit_removes_a_segment_over_deleted_fragments_without_history() {
    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    let old = staged_segment(&mut dataset, "i", IndexType::BTree, vec![0]).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![old])
        .await
        .unwrap();
    dataset.delete("i < 4").await.unwrap();
    let mut dataset = append_scalar_rows(dataset, 8, 4).await;
    assert!(!fragment_ids(&dataset).contains(&0));
    assert_eq!(fragment_ids(&dataset).len(), 2);

    let incoming = staged_segment(&mut dataset, "i", IndexType::BTree, vec![1]).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![incoming.clone()])
        .await
        .unwrap();
    assert!(fri_entry(&dataset).await.is_none());
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([incoming.uuid])
    );
}

/// The segments a commit removes are chosen at its read version; a stable
/// partition of the destinations commits before it. A commit that only adds
/// a segment rebases: the kept historical segments serve the new destinations
/// through both transitions. A commit whose removal was chosen from derived
/// coverage does not land on the newer version; within the same call the
/// choice is made afresh on it and the replacing segment carries its own
/// coverage across the new transition. Either way nothing the index covered
/// before the commit loses coverage.
#[rstest::rstest]
#[case::kept(false)]
#[case::replaced(true)]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn historical_segment_selection_survives_a_concurrent_stable_partition(
    #[case] covers_destinations: bool,
) {
    let dir = TempStrDir::default();
    let (dataset, [p, q], appended) = historical_fixture(dir.as_str(), "i", IndexType::BTree).await;
    let mut stale = dataset.clone();
    let fragments = if covers_destinations {
        vec![10, 11]
    } else {
        vec![appended]
    };
    let incoming = staged_segment(&mut stale, "i", IndexType::BTree, fragments).await;

    let mut latest = dataset;
    reserve_fragments(&mut latest, 20).await;
    let latest = commit_stable_partition(latest, &[10, 11], 30).await;
    let covered_before = derived_union(&latest, "staged").await;
    assert_eq!(covered_before, RoaringBitmap::from_iter([30u32, 31]));
    stale
        .commit_existing_index_segments("staged", "i", vec![incoming.clone()])
        .await
        .unwrap();
    assert_eq!(stale.manifest.version, latest.manifest.version + 1);

    let dataset = fresh_session(dir.as_str()).await;
    assert!(
        covered_before.is_subset(&derived_union(&dataset, "staged").await),
        "the commit must not lose coverage the index had on the version it landed on"
    );
    assert_each_row_served_once(&dataset, "staged").await;
    let mut fragments = fragment_ids(&dataset);
    fragments.sort_unstable();
    let mut expected_fragments = vec![appended, 30, 31];
    expected_fragments.sort_unstable();
    assert_eq!(fragments, expected_fragments);
    let (expected_segments, expected_unindexed) = if covers_destinations {
        (HashSet::from([incoming.uuid]), vec![appended])
    } else {
        (HashSet::from([p, q, incoming.uuid]), vec![])
    };
    assert_eq!(stored_uuids(&dataset, "staged").await, expected_segments);
    assert_eq!(
        unindexed_fragment_ids(&dataset, "staged").await,
        expected_unindexed
    );
    assert_queries_match_scans(&dataset, true).await;
}

// ---------------------------------------------------------------------------
// A partial commit beside segments that serve other fragments through the
// tagged history: segments are kept or removed whole, and a commit that would
// leave a covered live fragment without coverage is refused.
// ---------------------------------------------------------------------------

/// [`disk_fixture`] with four fragments: `i` 0..16.
async fn four_fragment_fixture(uri: &str) -> Dataset {
    lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col(
            "text",
            lance_datagen::array::fill_utf8("document".to_string()),
        )
        .col("w", lance_datagen::array::step::<Int32Type>())
        .into_dataset(uri, FragmentCount::from(4), FragmentRowCount::from(4))
        .await
        .unwrap()
}

/// A staged inverted segment that joins an index with committed segments.
async fn staged_fts_segment(
    dataset: &mut Dataset,
    column: &str,
    fragments: Vec<u32>,
) -> IndexMetadata {
    let params = analyzed_params();
    CreateIndexBuilder::new(dataset, &[column], IndexType::Inverted, &params)
        .name("staged".to_string())
        .fragments(fragments)
        .replace(true)
        .execute_uncommitted()
        .await
        .unwrap()
}

async fn staged_of_type(
    dataset: &mut Dataset,
    column: &str,
    index_type: IndexType,
    fragments: Vec<u32>,
) -> IndexMetadata {
    if index_type == IndexType::Inverted {
        staged_fts_segment(dataset, column, fragments).await
    } else {
        staged_segment(dataset, column, index_type, fragments).await
    }
}

/// Index `staged` with A over {0, 2} and B over {1, 3}, then stable-partition
/// {0, 1} into {10, 11}. Every destination holds rows of both sources, so A
/// and B serve 10 and 11 only together, while A alone serves 2 and B alone 3.
/// Returns the dataset and the uuids of A and B.
async fn mixed_destination_fixture(
    uri: &str,
    column: &str,
    index_type: IndexType,
) -> (Dataset, [Uuid; 2]) {
    let mut dataset = four_fragment_fixture(uri).await;
    let a = staged_of_type(&mut dataset, column, index_type, vec![0, 2]).await;
    let b = staged_of_type(&mut dataset, column, index_type, vec![1, 3]).await;
    dataset
        .commit_existing_index_segments("staged", column, vec![a.clone(), b.clone()])
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[0, 1], 10).await;
    let mut fragments = fragment_ids(&dataset);
    fragments.sort_unstable();
    assert_eq!(fragments, vec![2, 3, 10, 11]);
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([2u32, 3, 10, 11])
    );
    (dataset, [a.uuid, b.uuid])
}

/// The live fragments the reader derives coverage for, over every segment of
/// the logical index `name`.
async fn derived_union(dataset: &Dataset, name: &str) -> RoaringBitmap {
    dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .filter(|idx| idx.name == name)
        .filter_map(|idx| idx.fragment_bitmap.clone())
        .fold(RoaringBitmap::new(), |covered, bitmap| covered | bitmap)
}

/// The row addresses each BTree segment of `name` on `i` answers with on this
/// snapshot, through the reader's own translation. A query unions these sets,
/// so a row two segments both serve is invisible in query results; this is
/// what tells them apart.
async fn segment_contributions(dataset: &Dataset, name: &str) -> Vec<(Uuid, Vec<u64>)> {
    let segments: Vec<Uuid> = dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .filter(|idx| idx.name == name)
        .map(|idx| idx.uuid)
        .collect();
    let mut contributions = Vec::with_capacity(segments.len());
    for uuid in segments {
        let index = dataset
            .open_scalar_index("i", &uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let SearchResult::Exact(rows) = index
            .search(
                &SargableQuery::Range(
                    std::ops::Bound::Included(ScalarValue::Int32(Some(i32::MIN))),
                    std::ops::Bound::Included(ScalarValue::Int32(Some(i32::MAX))),
                ),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap()
        else {
            panic!("expected an exact result");
        };
        let addrs: Vec<u64> = rows
            .true_rows()
            .row_addrs()
            .unwrap()
            .map(u64::from)
            .collect();
        contributions.push((uuid, addrs));
    }
    contributions
}

/// Every live row of a covered fragment is answered by exactly one segment of
/// `name`, and no segment answers with a row that is not live.
async fn assert_each_row_served_once(dataset: &Dataset, name: &str) {
    let live: std::collections::HashMap<u64, i32> = row_addrs_by_i(dataset)
        .await
        .into_iter()
        .map(|(value, addr)| (addr, value))
        .collect();
    let mut served: std::collections::HashMap<u64, Vec<Uuid>> = Default::default();
    for (uuid, addrs) in segment_contributions(dataset, name).await {
        for addr in addrs {
            served.entry(addr).or_default().push(uuid);
        }
    }
    for (addr, segments) in &served {
        let Some(value) = live.get(addr) else {
            panic!("segments {segments:?} answer with row address {addr}, which is not live");
        };
        assert_eq!(
            segments.len(),
            1,
            "the row i = {value} is answered by segments {segments:?}"
        );
    }
    let covered = derived_union(dataset, name).await;
    for (addr, value) in &live {
        let fragment = lance_core::utils::address::RowAddress::from(*addr).fragment_id();
        if covered.contains(fragment) {
            assert!(
                served.contains_key(addr),
                "the row i = {value} of covered fragment {fragment} is answered by no segment"
            );
        }
    }
}

/// Every live `i` is found through the index, exactly once.
async fn assert_indexed_lookups(dataset: &Dataset) {
    for value in values(dataset, None, false).await {
        let predicate = format!("i = {value}");
        let plan = dataset
            .scan()
            .filter(&predicate)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(plan.contains("ScalarIndexQuery"), "{plan}");
        assert_eq!(
            values(dataset, Some(&predicate), true).await,
            vec![value],
            "{predicate}"
        );
    }
}

/// C over {2, 10} would remove A (its live fragment 2 is rebuilt) and with it
/// the source-0 rows 11 needs from A: refused, the manifest untouched, and
/// the index still covers 11.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn partial_commit_refuses_to_lose_coverage_another_fragment_depends_on() {
    let dir = TempStrDir::default();
    let (mut dataset, [a, b]) =
        mixed_destination_fixture(dir.as_str(), "i", IndexType::BTree).await;
    let version = dataset.manifest.version;
    let incoming = staged_segment(&mut dataset, "i", IndexType::BTree, vec![2, 10]).await;

    let error = dataset
        .commit_existing_index_segments("staged", "i", vec![incoming])
        .await
        .unwrap_err();
    assert!(
        matches!(error, crate::Error::InvalidInput { .. }),
        "unexpected error: {error}"
    );
    let message = error.to_string();
    assert!(
        message.contains(&format!(
            "segment {a} (live fragments the incoming segments overlap: [2])"
        )),
        "{message}"
    );
    assert!(
        message.contains("leave live fragments [11] without"),
        "{message}"
    );
    assert!(!message.contains(&b.to_string()), "{message}");

    assert_eq!(dataset.manifest.version, version);
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, version);
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([a, b])
    );
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([2u32, 3, 10, 11])
    );
    assert!(unindexed_fragment_ids(&dataset, "staged").await.is_empty());
    assert_each_row_served_once(&dataset, "staged").await;
    assert_indexed_lookups(&dataset).await;
}

/// C over {10} only overlaps neither A's nor B's live fragments: both stay
/// whole, and every row is answered by exactly one segment, on the session
/// that warmed the old snapshot, after a reopen, and after a later rewrite of
/// {2, 11} that no index remap follows (A's file still holds its rows of
/// fragment 2, which now translate into the rewrite's destinations).
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn partial_commit_beside_a_partly_live_segment_serves_each_row_once() {
    let dir = TempStrDir::default();
    let (dataset, [a, b]) = mixed_destination_fixture(dir.as_str(), "i", IndexType::BTree).await;
    drop(dataset);
    let mut dataset = fresh_session(dir.as_str()).await;
    assert_each_row_served_once(&dataset, "staged").await;
    assert_indexed_lookups(&dataset).await;

    let incoming = staged_segment(&mut dataset, "i", IndexType::BTree, vec![10]).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![incoming.clone()])
        .await
        .unwrap();
    let all = HashSet::from([a, b, incoming.uuid]);
    assert_eq!(stored_uuids(&dataset, "staged").await, all);
    assert_each_row_served_once(&dataset, "staged").await;
    assert_indexed_lookups(&dataset).await;

    let mut dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([2u32, 3, 10, 11])
    );
    assert_each_row_served_once(&dataset, "staged").await;
    assert_indexed_lookups(&dataset).await;

    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[2, 11], 30).await;
    let mut fragments = fragment_ids(&dataset);
    fragments.sort_unstable();
    assert_eq!(fragments, vec![3, 10, 30, 31]);
    assert_eq!(stored_uuids(&dataset, "staged").await, all);
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([3u32, 10, 30, 31])
    );
    assert_each_row_served_once(&dataset, "staged").await;
    assert_indexed_lookups(&dataset).await;

    let dataset = fresh_session(dir.as_str()).await;
    assert_each_row_served_once(&dataset, "staged").await;
    assert_indexed_lookups(&dataset).await;
}

/// C over {2, 10, 11} rebuilds A's live fragment and every destination A
/// serves: A is removed, B stays for 3, and nothing loses coverage.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn partial_commit_over_a_segment_and_its_destinations_replaces_it() {
    let dir = TempStrDir::default();
    let (mut dataset, [_, b]) =
        mixed_destination_fixture(dir.as_str(), "i", IndexType::BTree).await;
    let incoming = staged_segment(&mut dataset, "i", IndexType::BTree, vec![2, 10, 11]).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![incoming.clone()])
        .await
        .unwrap();
    assert_each_row_served_once(&dataset, "staged").await;

    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([b, incoming.uuid])
    );
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([2u32, 3, 10, 11])
    );
    assert!(unindexed_fragment_ids(&dataset, "staged").await.is_empty());
    assert_each_row_served_once(&dataset, "staged").await;
    assert_indexed_lookups(&dataset).await;
}

/// A staged segment built over source 0 before {0, 1} was partitioned shares
/// provenance with E over {0, 1}: keeping E whole would answer source 0's rows
/// twice, and removing it loses source 1's contribution to 10 and 11. Refused.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn retired_staged_provenance_cannot_replace_part_of_a_historical_segment() {
    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    let e = staged_segment(&mut dataset, "i", IndexType::BTree, vec![0, 1]).await;
    let retired = staged_segment(&mut dataset, "i", IndexType::BTree, vec![0]).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![e.clone()])
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let mut dataset = commit_stable_partition(dataset, &[0, 1], 10).await;
    let version = dataset.manifest.version;

    let error = dataset
        .commit_existing_index_segments("staged", "i", vec![retired])
        .await
        .unwrap_err();
    assert!(
        matches!(error, crate::Error::InvalidInput { .. }),
        "unexpected error: {error}"
    );
    let message = error.to_string();
    assert!(
        message.contains(&format!(
            "segment {} (live fragments the incoming segments overlap: [])",
            e.uuid
        )),
        "{message}"
    );
    assert!(
        message.contains("leave live fragments [10, 11] without"),
        "{message}"
    );

    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, version);
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([e.uuid])
    );
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([10u32, 11])
    );
    assert_queries_match_scans(&dataset, true).await;
}

/// P over 0 is committed alone and Q over 1 later, after {0, 1} was
/// partitioned: P derives nothing before the commit (10 and 11 lack source 1),
/// and P and Q serve both destinations together after it. P stays.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn staged_segment_completing_a_historical_segment_keeps_it() {
    let dir = TempStrDir::default();
    let mut dataset = disk_fixture(dir.as_str()).await;
    let p = staged_segment(&mut dataset, "i", IndexType::BTree, vec![0]).await;
    let q = staged_segment(&mut dataset, "i", IndexType::BTree, vec![1]).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![p.clone()])
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let mut dataset = commit_stable_partition(dataset, &[0, 1], 10).await;
    assert!(derived_union(&dataset, "staged").await.is_empty());

    dataset
        .commit_existing_index_segments("staged", "i", vec![q.clone()])
        .await
        .unwrap();

    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([p.uuid, q.uuid])
    );
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([10u32, 11])
    );
    assert!(unindexed_fragment_ids(&dataset, "staged").await.is_empty());
    assert_each_row_served_once(&dataset, "staged").await;
    assert_queries_match_scans(&dataset, true).await;
}

/// The relative path of a segment's inverted metadata file (its format
/// marker), as the index store reads it.
fn fts_marker_suffix(segment: Uuid) -> String {
    format!(
        "{}/{}",
        segment,
        lance_index::scalar::inverted::METADATA_FILE
    )
}

/// FTS segments P over 0 and Q over 1, a stable partition of {0, 1} into
/// {10, 11}, and a segment C over {10, 11} staged from that version. With
/// every format marker readable, C supersedes P and Q and the partial commit
/// removes them. When P's marker cannot be read in a fresh session, P's
/// derived coverage is unknown, which also leaves Q's absent coverage
/// unproven: both stay, and are still there once the store answers again.
#[rstest::rstest]
#[case::readable(false)]
#[case::unreadable(true)]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn an_unreadable_format_marker_keeps_historical_segments_on_a_partial_commit(
    #[case] fails: bool,
) {
    let dir = TempStrDir::default();
    let path = dir.as_str();
    let mut dataset = disk_fixture(path).await;
    let p = staged_fts_segment(&mut dataset, "text", vec![0]).await;
    let q = staged_fts_segment(&mut dataset, "text", vec![1]).await;
    dataset
        .commit_existing_index_segments("staged", "text", vec![p.clone(), q.clone()])
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[0, 1], 10).await;
    let mut dataset = append_scalar_rows(dataset, 8, 4).await;
    let appended = *fragment_ids(&dataset).iter().max().unwrap();
    let incoming = staged_fts_segment(&mut dataset, "text", vec![10, 11]).await;
    drop(dataset);

    let (mut dataset, failing) = open_with_failing_reads(path, fts_marker_suffix(p.uuid)).await;
    failing.store(fails, Ordering::SeqCst);
    dataset
        .commit_existing_index_segments("staged", "text", vec![incoming.clone()])
        .await
        .unwrap();
    failing.store(false, Ordering::SeqCst);

    let dataset = fresh_session(path).await;
    let expected = if fails {
        HashSet::from([p.uuid, q.uuid, incoming.uuid])
    } else {
        HashSet::from([incoming.uuid])
    };
    assert_eq!(stored_uuids(&dataset, "staged").await, expected);
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([10u32, 11])
    );
    assert_eq!(
        unindexed_fragment_ids(&dataset, "staged").await,
        vec![appended]
    );
}

/// FTS segments A over {0, 2} and B over {1, 3} as in
/// [`mixed_destination_fixture`]. C over {2, 10, 11} replaces A when every
/// marker is readable; with A's marker unreadable whether removing A loses
/// coverage cannot be established, so the commit fails with the read error
/// itself and the manifest is untouched. Once the store answers, the same
/// commit lands.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn an_unreadable_format_marker_refuses_an_unproven_replacement() {
    let dir = TempStrDir::default();
    let path = dir.as_str();
    let (mut dataset, [a, b]) = mixed_destination_fixture(path, "text", IndexType::Inverted).await;
    let version = dataset.manifest.version;
    let incoming = staged_fts_segment(&mut dataset, "text", vec![2, 10, 11]).await;
    drop(dataset);
    let segments_before = stored_bitmaps(&fresh_session(path).await, "staged").await;

    let (mut dataset, failing) = open_with_failing_reads(path, fts_marker_suffix(a)).await;
    failing.store(true, Ordering::SeqCst);
    let error = dataset
        .commit_existing_index_segments("staged", "text", vec![incoming.clone()])
        .await
        .unwrap_err();
    assert!(
        matches!(error, crate::Error::IO { .. }),
        "unexpected error: {error:?}"
    );
    assert!(
        format!("{error:?}").contains("injected transient read failure"),
        "{error:?}"
    );
    assert_eq!(dataset.manifest.version, version);
    let untouched = fresh_session(path).await;
    assert_eq!(untouched.manifest.version, version);
    assert_eq!(
        stored_uuids(&untouched, "staged").await,
        HashSet::from([a, b])
    );
    assert_eq!(stored_bitmaps(&untouched, "staged").await, segments_before);

    failing.store(false, Ordering::SeqCst);
    dataset
        .commit_existing_index_segments("staged", "text", vec![incoming.clone()])
        .await
        .unwrap();
    let dataset = fresh_session(path).await;
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([b, incoming.uuid])
    );
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([2u32, 3, 10, 11])
    );
}

/// C over {2, 10, 11} is staged at the fixture's version and replaces A
/// there, B staying for its live fragment 3. A stable partition of {3, 11}
/// commits first. The replacement was chosen from coverage derived at the
/// read version, so it does not land on the newer one; within the same call
/// the choice is made again on the latest version: A is still replaced, and
/// B, whose fragments are all retired by now, stays because together with C
/// it serves the new destinations. Nothing the index covered at the latest
/// version loses coverage.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn coverage_dependent_commit_is_decided_again_after_a_concurrent_stable_partition() {
    let dir = TempStrDir::default();
    let (dataset, [a, b]) = mixed_destination_fixture(dir.as_str(), "i", IndexType::BTree).await;
    let mut stale = dataset.clone();
    let incoming = staged_segment(&mut stale, "i", IndexType::BTree, vec![2, 10, 11]).await;
    assert!(requires_unchanged_snapshot(&stale, "i", vec![incoming.clone()]).await);

    let mut latest = dataset;
    reserve_fragments(&mut latest, 20).await;
    let latest = commit_stable_partition(latest, &[3, 11], 30).await;
    let covered_before = derived_union(&latest, "staged").await;
    assert_eq!(covered_before, RoaringBitmap::from_iter([2u32, 10, 30, 31]));

    stale
        .commit_existing_index_segments("staged", "i", vec![incoming.clone()])
        .await
        .unwrap();
    assert_eq!(stale.manifest.version, latest.manifest.version + 1);
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, latest.manifest.version + 1);
    let stored = stored_uuids(&dataset, "staged").await;
    assert!(!stored.contains(&a), "{a} is replaced");
    assert_eq!(stored, HashSet::from([b, incoming.uuid]));
    assert!(covered_before.is_subset(&derived_union(&dataset, "staged").await));
    assert!(unindexed_fragment_ids(&dataset, "staged").await.is_empty());
    assert_each_row_served_once(&dataset, "staged").await;
    assert_indexed_lookups(&dataset).await;
}

/// C over {2, 10, 11} is staged at the fixture's version, where it replaces
/// A. Before it commits, the index is rebuilt as D over every live fragment,
/// also replacing A. The ordinary conflict rules refuse C over the rebuild,
/// so the call returns a retryable conflict without deciding again: the
/// manifest stays at D's version and the caller's dataset is back at the
/// version it was called with. Repeated on the latest version, C would leave
/// D's fragment 3 without coverage, so the request is refused.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn coverage_dependent_commit_after_a_concurrent_rebuild_conflicts_then_is_refused() {
    let dir = TempStrDir::default();
    let (dataset, _) = mixed_destination_fixture(dir.as_str(), "i", IndexType::BTree).await;
    let mut stale = dataset.clone();
    let incoming = staged_segment(&mut stale, "i", IndexType::BTree, vec![2, 10, 11]).await;

    let mut latest = dataset;
    let rebuilt = staged_segment(&mut latest, "i", IndexType::BTree, vec![2, 3, 10, 11]).await;
    latest
        .commit_existing_index_segments("staged", "i", vec![rebuilt.clone()])
        .await
        .unwrap();
    let before = stored_bitmaps(&latest, "staged").await;
    let called_at = stale.manifest.version;
    assert!(called_at < latest.manifest.version);

    let error = stale
        .commit_existing_index_segments("staged", "i", vec![incoming.clone()])
        .await
        .unwrap_err();
    assert_eq!(stale.manifest.version, called_at);
    assert!(
        matches!(error, crate::Error::RetryableCommitConflict { .. }),
        "unexpected error: {error:?}"
    );
    let untouched = fresh_session(dir.as_str()).await;
    assert_eq!(untouched.manifest.version, latest.manifest.version);
    assert_eq!(stored_bitmaps(&untouched, "staged").await, before);

    stale.checkout_latest().await.unwrap();
    let error = stale
        .commit_existing_index_segments("staged", "i", vec![incoming])
        .await
        .unwrap_err();
    assert_eq!(stale.manifest.version, latest.manifest.version);
    assert!(
        matches!(error, crate::Error::InvalidInput { .. }),
        "unexpected error: {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains(&format!(
            "would orphan fragments [3] from existing segment {}",
            rebuilt.uuid
        )),
        "{message}"
    );
    let untouched = fresh_session(dir.as_str()).await;
    assert_eq!(untouched.manifest.version, latest.manifest.version);
    assert_eq!(stored_bitmaps(&untouched, "staged").await, before);
}

/// C over {2, 10, 11} is staged at the fixture's version, where it replaces
/// A. Before it commits, the index is dropped. The ordinary conflict rules
/// refuse C over the drop, so the call returns a retryable conflict: the
/// index stays dropped, no manifest is written for the request and the
/// caller's dataset is back at the version it was called with.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn coverage_dependent_commit_conflicts_with_a_concurrent_drop() {
    let dir = TempStrDir::default();
    let (dataset, _) = mixed_destination_fixture(dir.as_str(), "i", IndexType::BTree).await;
    let mut stale = dataset.clone();
    let incoming = staged_segment(&mut stale, "i", IndexType::BTree, vec![2, 10, 11]).await;

    let mut latest = dataset;
    latest.drop_index("staged").await.unwrap();
    let called_at = stale.manifest.version;
    assert!(called_at < latest.manifest.version);

    let error = stale
        .commit_existing_index_segments("staged", "i", vec![incoming])
        .await
        .unwrap_err();
    assert_eq!(stale.manifest.version, called_at);
    assert!(
        matches!(error, crate::Error::RetryableCommitConflict { .. }),
        "unexpected error: {error:?}"
    );
    let untouched = fresh_session(dir.as_str()).await;
    assert_eq!(untouched.manifest.version, latest.manifest.version);
    assert!(stored_uuids(&untouched, "staged").await.is_empty());
}

/// C over {2, 10, 11} is staged at the fixture's version, where it replaces
/// A. Before it commits, a different index is dropped. The ordinary conflict
/// rules accept C over that drop, so within the same call the choice is made
/// again on the latest version and C lands there.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn coverage_dependent_commit_is_decided_again_after_a_concurrent_drop_of_another_index() {
    let dir = TempStrDir::default();
    let (mut dataset, [a, b]) =
        mixed_destination_fixture(dir.as_str(), "i", IndexType::BTree).await;
    dataset
        .create_index(
            &["w"],
            IndexType::BTree,
            Some("other".into()),
            &ScalarIndexParams::for_builtin(IndexType::BTree.try_into().unwrap()),
            false,
        )
        .await
        .unwrap();
    let mut stale = dataset.clone();
    let incoming = staged_segment(&mut stale, "i", IndexType::BTree, vec![2, 10, 11]).await;

    let mut latest = dataset;
    latest.drop_index("other").await.unwrap();

    stale
        .commit_existing_index_segments("staged", "i", vec![incoming.clone()])
        .await
        .unwrap();
    assert_eq!(stale.manifest.version, latest.manifest.version + 1);
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, latest.manifest.version + 1);
    let stored = stored_uuids(&dataset, "staged").await;
    assert!(!stored.contains(&a), "{a} is replaced");
    assert_eq!(stored, HashSet::from([b, incoming.uuid]));
    assert!(stored_uuids(&dataset, "other").await.is_empty());
}

/// FTS segments A over {0, 2} and B over {1, 3}, and C over {0, 2} staged
/// before {0, 1} is partitioned into {10, 11}, then C committed with its
/// format marker unreadable. The read failure surfaces before the commit
/// decides which segments stay, so nothing is committed.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn an_unreadable_incoming_marker_commits_nothing() {
    let dir = TempStrDir::default();
    let path = dir.as_str();
    let mut dataset = four_fragment_fixture(path).await;
    let a = staged_fts_segment(&mut dataset, "text", vec![0, 2]).await;
    let b = staged_fts_segment(&mut dataset, "text", vec![1, 3]).await;
    let incoming = staged_fts_segment(&mut dataset, "text", vec![0, 2]).await;
    dataset
        .commit_existing_index_segments("staged", "text", vec![a.clone(), b.clone()])
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let dataset = commit_stable_partition(dataset, &[0, 1], 10).await;
    let version = dataset.manifest.version;
    drop(dataset);
    let segments_before = segment_bitmaps(&fresh_session(path).await, "staged").await;

    let (mut dataset, failing) =
        open_with_failing_reads(path, fts_marker_suffix(incoming.uuid)).await;
    failing.store(true, Ordering::SeqCst);
    let result = dataset
        .commit_existing_index_segments("staged", "text", vec![incoming.clone()])
        .await;
    failing.store(false, Ordering::SeqCst);
    let error = result.unwrap_err();
    assert!(
        matches!(error, crate::Error::IO { .. }),
        "unexpected error: {error:?}"
    );
    assert!(
        format!("{error:?}").contains("injected transient read failure"),
        "{error:?}"
    );
    assert_eq!(dataset.manifest.version, version);

    let dataset = fresh_session(path).await;
    assert_eq!(dataset.manifest.version, version);
    assert_eq!(segment_bitmaps(&dataset, "staged").await, segments_before);
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([a.uuid, b.uuid])
    );
}

/// Each segment of `name` with the fragment bitmap the reader loads for it.
async fn segment_bitmaps(
    dataset: &Dataset,
    name: &str,
) -> std::collections::BTreeMap<Uuid, Option<RoaringBitmap>> {
    dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .filter(|idx| idx.name == name)
        .map(|idx| (idx.uuid, idx.fragment_bitmap.clone()))
        .collect()
}

/// Each stored segment of `name` with the fragment bitmap the manifest
/// records for it, including segments the reader leaves out of the listing.
async fn stored_bitmaps(
    dataset: &Dataset,
    name: &str,
) -> std::collections::BTreeMap<Uuid, Option<RoaringBitmap>> {
    stored_segments(dataset, name)
        .await
        .into_iter()
        .map(|idx| (idx.uuid, idx.fragment_bitmap))
        .collect()
}

/// Segments P over 0 and Q over 1, then a history whose only transition (of
/// {0, 1} into {10, 11}) this build cannot interpret, and one appended
/// fragments a1 and a2. The reader lets no user segment take part in
/// coverage. A segment T over {a1, a2} commits beside P and Q, which stay with
/// their stored bitmaps. U over {a1, a2} would remove T, whose coverage this
/// build cannot establish under that history, and is refused with every
/// segment left as it was, although U stores all of T's fragments again.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn an_unsupported_history_keeps_segments_and_refuses_an_unproven_removal() {
    let dir = TempStrDir::default();
    let path = dir.as_str();
    let mut dataset = disk_fixture(path).await;
    let p = staged_segment(&mut dataset, "i", IndexType::BTree, vec![0]).await;
    let q = staged_segment(&mut dataset, "i", IndexType::BTree, vec![1]).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![p.clone(), q.clone()])
        .await
        .unwrap();
    let (mut transition, destinations) = reader_tests::prepare(&dataset).await;
    transition.mapping = None;
    let mut raw = transition.encode_to_vec();
    raw.extend(reader_tests::field(17, b"future mapping"));
    reader_tests::install(
        &mut dataset,
        reader_tests::field(2, &raw),
        destinations,
        false,
    )
    .await;
    let indices = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    reader_tests::persist_fixture(&mut dataset, indices).await;
    let dataset = append_scalar_rows(dataset, 8, 4).await;
    let a1 = *fragment_ids(&dataset).iter().max().unwrap();
    let mut dataset = append_scalar_rows(dataset, 12, 4).await;
    let a2 = *fragment_ids(&dataset).iter().max().unwrap();
    assert!(a2 > a1);
    let appended = RoaringBitmap::from_iter([a1, a2]);
    assert!(
        dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .all(|idx| idx.name != "staged")
    );
    let historical = stored_bitmaps(&dataset, "staged").await;
    assert_eq!(
        historical,
        std::collections::BTreeMap::from([
            (p.uuid, Some(RoaringBitmap::from_iter([0u32]))),
            (q.uuid, Some(RoaringBitmap::from_iter([1u32]))),
        ])
    );

    let t = staged_segment(&mut dataset, "i", IndexType::BTree, vec![a1, a2]).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![t.clone()])
        .await
        .unwrap();
    let mut expected = historical.clone();
    expected.insert(t.uuid, Some(appended.clone()));
    let reopened = fresh_session(path).await;
    assert_eq!(reopened.manifest.version, dataset.manifest.version);
    assert_eq!(stored_bitmaps(&reopened, "staged").await, expected);

    let version = dataset.manifest.version;
    let u = staged_segment(&mut dataset, "i", IndexType::BTree, vec![a1, a2]).await;
    let error = dataset
        .commit_existing_index_segments("staged", "i", vec![u])
        .await
        .unwrap_err();
    assert!(
        matches!(error, crate::Error::InvalidInput { .. }),
        "unexpected error: {error}"
    );
    let message = error.to_string();
    assert!(message.contains("cannot establish"), "{message}");
    assert!(
        message.contains("this build cannot translate it"),
        "{message}"
    );
    assert!(message.contains(&t.uuid.to_string()), "{message}");
    assert_eq!(dataset.manifest.version, version);
    let untouched = fresh_session(path).await;
    assert_eq!(untouched.manifest.version, version);
    assert_eq!(stored_bitmaps(&untouched, "staged").await, expected);
}

/// A over {0, 2} and B over {1, 3}, and C over {0} and D over {2} staged
/// before {0, 1} is partitioned into {10, 11}. Neither C nor D alone stores
/// every fragment A stores; together they do, and with B they serve 10 and
/// 11. One commit of C and D replaces A, and every row is answered once.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn incoming_segments_together_resupply_a_removed_segment() {
    let dir = TempStrDir::default();
    let mut dataset = four_fragment_fixture(dir.as_str()).await;
    let a = staged_segment(&mut dataset, "i", IndexType::BTree, vec![0, 2]).await;
    let b = staged_segment(&mut dataset, "i", IndexType::BTree, vec![1, 3]).await;
    let c = staged_segment(&mut dataset, "i", IndexType::BTree, vec![0]).await;
    let d = staged_segment(&mut dataset, "i", IndexType::BTree, vec![2]).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![a.clone(), b.clone()])
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let mut dataset = commit_stable_partition(dataset, &[0, 1], 10).await;

    dataset
        .commit_existing_index_segments("staged", "i", vec![c.clone(), d.clone()])
        .await
        .unwrap();
    assert_each_row_served_once(&dataset, "staged").await;

    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([b.uuid, c.uuid, d.uuid])
    );
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([2u32, 3, 10, 11])
    );
    assert!(unindexed_fragment_ids(&dataset, "staged").await.is_empty());
    assert_each_row_served_once(&dataset, "staged").await;
    assert_indexed_lookups(&dataset).await;
}

/// A over {0, 2} and B over {1, 3}, and C over {0} staged before {0, 1} is
/// partitioned into {10, 11}. C overlaps none of A's live fragments but
/// shares A's rewritten source 0, so keeping both would answer source 0's
/// rows twice. Refused with the shared source named, the manifest untouched.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn retired_staged_provenance_shared_with_a_live_segment_is_refused() {
    let dir = TempStrDir::default();
    let mut dataset = four_fragment_fixture(dir.as_str()).await;
    let a = staged_segment(&mut dataset, "i", IndexType::BTree, vec![0, 2]).await;
    let b = staged_segment(&mut dataset, "i", IndexType::BTree, vec![1, 3]).await;
    let incoming = staged_segment(&mut dataset, "i", IndexType::BTree, vec![0]).await;
    dataset
        .commit_existing_index_segments("staged", "i", vec![a.clone(), b.clone()])
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let mut dataset = commit_stable_partition(dataset, &[0, 1], 10).await;
    let version = dataset.manifest.version;

    let error = dataset
        .commit_existing_index_segments("staged", "i", vec![incoming])
        .await
        .unwrap_err();
    assert!(
        matches!(error, crate::Error::InvalidInput { .. }),
        "unexpected error: {error}"
    );
    let message = error.to_string();
    assert!(
        message.contains(&format!(
            "existing segment {} for 'staged' shares already-rewritten source fragments [0]",
            a.uuid
        )),
        "{message}"
    );
    assert!(message.contains("live fragments [2]"), "{message}");
    assert!(!message.contains("orphan"), "{message}");
    assert!(!message.contains(&b.uuid.to_string()), "{message}");

    assert_eq!(dataset.manifest.version, version);
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, version);
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([a.uuid, b.uuid])
    );
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([2u32, 3, 10, 11])
    );
    assert_each_row_served_once(&dataset, "staged").await;
}

/// Data fragments {0, 1} partitioned into {10, 11}.
async fn partitioned_fixture(uri: &str) -> Dataset {
    let mut dataset = disk_fixture(uri).await;
    reserve_fragments(&mut dataset, 20).await;
    commit_stable_partition(dataset, &[0, 1], 10).await
}

/// Data fragments {0, 1} partitioned into {10, 11}, then an FM segment S over
/// destination 10 alone built and committed at the current version. The
/// history records 10, so S may need translation, and the reader leaves it
/// out of coverage because FM has no batch remapper. Returns the dataset and
/// S.
async fn fm_segment_over_a_destination(uri: &str) -> (Dataset, IndexMetadata) {
    let mut dataset = partitioned_fixture(uri).await;
    let s = staged_segment(&mut dataset, "text", IndexType::Fm, vec![10]).await;
    dataset
        .commit_existing_index_segments("staged", "text", vec![s.clone()])
        .await
        .unwrap();
    (dataset, s)
}

async fn assert_replaced_by(dir: &str, version: u64, s: Uuid, incoming: &IndexMetadata) {
    let dataset = fresh_session(dir).await;
    assert!(dataset.manifest.version > version);
    let stored = stored_segments(&dataset, "staged").await;
    assert_eq!(
        stored
            .iter()
            .map(|segment| segment.uuid)
            .collect::<HashSet<_>>(),
        HashSet::from([incoming.uuid]),
        "{s} is replaced"
    );
    assert_eq!(stored[0].fragment_bitmap, incoming.fragment_bitmap);
    // The incoming segment stores destination 10, which the history records,
    // and FM cannot translate, so the reader lists no coverage for it.
    assert_eq!(derived_coverage(&dataset, "staged").await, None);
}

/// I over {10} replaces S over {10}. The reader leaves both out of coverage
/// for lacking a batch remapper; the replacement is accepted because S is in
/// that same class and I stores all of S's fragments again.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn a_segment_without_a_batch_remapper_can_still_be_replaced_over_the_same_fragments() {
    let dir = TempStrDir::default();
    let (mut dataset, s) = fm_segment_over_a_destination(dir.as_str()).await;
    let version = dataset.manifest.version;
    let incoming = staged_segment(&mut dataset, "text", IndexType::Fm, vec![10]).await;
    assert_eq!(
        incoming.fragment_bitmap,
        Some(RoaringBitmap::from_iter([10u32]))
    );

    dataset
        .commit_existing_index_segments("staged", "text", vec![incoming.clone()])
        .await
        .unwrap();
    assert_replaced_by(dir.as_str(), version, s.uuid, &incoming).await;
}

/// I over {10, appended} replaces S over {10} the same way.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn a_segment_without_a_batch_remapper_can_be_replaced_by_one_over_more_fragments() {
    let dir = TempStrDir::default();
    let (dataset, s) = fm_segment_over_a_destination(dir.as_str()).await;
    let mut dataset = append_scalar_rows(dataset, 8, 4).await;
    let appended = *fragment_ids(&dataset).iter().max().unwrap();
    assert!(appended > 11);
    let version = dataset.manifest.version;
    let incoming = staged_segment(&mut dataset, "text", IndexType::Fm, vec![10, appended]).await;
    assert_eq!(
        incoming.fragment_bitmap,
        Some(RoaringBitmap::from_iter([10u32, appended]))
    );

    dataset
        .commit_existing_index_segments("staged", "text", vec![incoming.clone()])
        .await
        .unwrap();
    assert_replaced_by(dir.as_str(), version, s.uuid, &incoming).await;
}

/// Appends one fragment of four rows and returns its id, which lies outside
/// the history.
async fn append_live_fragment(dataset: Dataset, start: i32) -> (Dataset, u32) {
    let dataset = append_scalar_rows(dataset, start, 4).await;
    let appended = *fragment_ids(&dataset).iter().max().unwrap();
    assert!(appended > 11);
    (dataset, appended)
}

/// Asserts a refused commit left the version and every stored bitmap as they
/// were.
async fn assert_refused_unchanged(
    error: crate::Error,
    dir: &str,
    dataset: &Dataset,
    version: u64,
    before: &std::collections::BTreeMap<Uuid, Option<RoaringBitmap>>,
    removed_member: Uuid,
) {
    assert!(
        matches!(error, crate::Error::InvalidInput { .. }),
        "unexpected error: {error}"
    );
    let message = error.to_string();
    assert!(message.contains("cannot establish"), "{message}");
    assert!(message.contains(&removed_member.to_string()), "{message}");
    assert_eq!(dataset.manifest.version, version);
    let untouched = fresh_session(dir).await;
    assert_eq!(untouched.manifest.version, version);
    assert_eq!(&stored_bitmaps(&untouched, "staged").await, before);
}

/// On {0, 1} partitioned into {10, 11} and an appended fragment 12, FM
/// segment R over {12} needs no translation, so it takes part in coverage and
/// directly covers 12. Incoming I over {12, 10} meets the history and FM has
/// no batch remapper, so the reader leaves I out. Replacing R by I would lose
/// the indexed coverage of 12, so the commit is refused and nothing changes.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn a_member_is_not_replaced_by_a_segment_without_a_batch_remapper() {
    let dir = TempStrDir::default();
    let dataset = partitioned_fixture(dir.as_str()).await;
    let (mut dataset, appended) = append_live_fragment(dataset, 8).await;
    let r = staged_segment(&mut dataset, "text", IndexType::Fm, vec![appended]).await;
    dataset
        .commit_existing_index_segments("staged", "text", vec![r.clone()])
        .await
        .unwrap();
    assert_eq!(
        derived_coverage(&dataset, "staged").await,
        Some(RoaringBitmap::from_iter([appended]))
    );
    let version = dataset.manifest.version;
    let before = stored_bitmaps(&dataset, "staged").await;
    assert_eq!(
        before,
        std::collections::BTreeMap::from([(r.uuid, Some(RoaringBitmap::from_iter([appended])))])
    );

    let incoming = staged_segment(&mut dataset, "text", IndexType::Fm, vec![appended, 10]).await;
    let error = dataset
        .commit_existing_index_segments("staged", "text", vec![incoming])
        .await
        .unwrap_err();
    assert_refused_unchanged(error, dir.as_str(), &dataset, version, &before, r.uuid).await;
}

/// FM segments S over {10} (left out: it meets the history and FM has no
/// batch remapper) and M over two appended fragments {a1, a2} (a member).
/// Returns the dataset, S, M and the appended ids.
async fn fm_member_beside_an_excluded_segment(
    uri: &str,
) -> (Dataset, IndexMetadata, IndexMetadata, [u32; 2]) {
    let (dataset, s) = fm_segment_over_a_destination(uri).await;
    let (dataset, a1) = append_live_fragment(dataset, 8).await;
    let (mut dataset, a2) = append_live_fragment(dataset, 12).await;
    let m = staged_segment(&mut dataset, "text", IndexType::Fm, vec![a1, a2]).await;
    dataset
        .commit_existing_index_segments("staged", "text", vec![m.clone()])
        .await
        .unwrap();
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([s.uuid, m.uuid])
    );
    assert_eq!(
        derived_coverage(&dataset, "staged").await,
        Some(RoaringBitmap::from_iter([a1, a2]))
    );
    (dataset, s, m, [a1, a2])
}

/// Incoming J over {a1, a2} (a member) supplies M again; incoming K over {10}
/// (left out for the same missing batch remapper) supplies S again. Both
/// removals are decided on their own and the commit lands.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn mixed_removals_land_when_each_is_supplied_again() {
    let dir = TempStrDir::default();
    let (mut dataset, s, m, [a1, a2]) = fm_member_beside_an_excluded_segment(dir.as_str()).await;
    let version = dataset.manifest.version;
    let j = staged_segment(&mut dataset, "text", IndexType::Fm, vec![a1, a2]).await;
    let k = staged_segment(&mut dataset, "text", IndexType::Fm, vec![10]).await;

    dataset
        .commit_existing_index_segments("staged", "text", vec![j.clone(), k.clone()])
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, version + 1);
    assert_eq!(
        stored_bitmaps(&dataset, "staged").await,
        std::collections::BTreeMap::from([
            (j.uuid, j.fragment_bitmap.clone()),
            (k.uuid, k.fragment_bitmap.clone()),
        ]),
        "{} and {} are replaced",
        s.uuid,
        m.uuid
    );
    assert_eq!(
        derived_coverage(&dataset, "staged").await,
        Some(RoaringBitmap::from_iter([a1, a2]))
    );
}

/// Incoming J over {a1} (a member) and K over {10, a2} (left out): S would be
/// supplied again by the exception, but M's a2 is stored again only by K,
/// which cannot stand in for a member. The commit is refused and nothing
/// changes.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn mixed_removals_are_refused_when_a_member_is_not_supplied_again() {
    let dir = TempStrDir::default();
    let (mut dataset, _s, m, [a1, a2]) = fm_member_beside_an_excluded_segment(dir.as_str()).await;
    let version = dataset.manifest.version;
    let before = stored_bitmaps(&dataset, "staged").await;
    let j = staged_segment(&mut dataset, "text", IndexType::Fm, vec![a1]).await;
    let k = staged_segment(&mut dataset, "text", IndexType::Fm, vec![10, a2]).await;

    let error = dataset
        .commit_existing_index_segments("staged", "text", vec![j, k])
        .await
        .unwrap_err();
    assert_refused_unchanged(error, dir.as_str(), &dataset, version, &before, m.uuid).await;
}

/// I over {10} replaces S over {10} only by the exception for a type without
/// a batch remapper, which rests on this snapshot's classification. An
/// append commits first, so the replacement does not land on the newer
/// version; within the same call it is decided again on the latest version
/// and lands there.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn a_replacement_by_the_exception_is_decided_again_after_a_concurrent_append() {
    let dir = TempStrDir::default();
    let (dataset, s) = fm_segment_over_a_destination(dir.as_str()).await;
    let mut stale = dataset.clone();
    let incoming = staged_segment(&mut stale, "text", IndexType::Fm, vec![10]).await;
    let latest = append_scalar_rows(dataset, 8, 4).await;

    stale
        .commit_existing_index_segments("staged", "text", vec![incoming.clone()])
        .await
        .unwrap();
    assert_eq!(stale.manifest.version, latest.manifest.version + 1);
    assert_replaced_by(dir.as_str(), latest.manifest.version, s.uuid, &incoming).await;
}

/// The same replacement through a store whose manifest write fails without
/// landing, or lands with an outcome that cannot be read back. Neither is
/// another writer's commit: the failure is returned with its source and the
/// decision is not made again.
#[rstest::rstest]
#[case::failed_write(crate::utils::test::AmbiguousFailure::FailOutright)]
#[case::unknown_outcome(crate::utils::test::AmbiguousFailure::LandAndErrorWithoutVerification)]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn a_failed_conditional_write_is_returned_not_decided_again(
    #[case] failure: crate::utils::test::AmbiguousFailure,
) {
    let dir = TempStrDir::default();
    let (dataset, s) = fm_segment_over_a_destination(dir.as_str()).await;
    let version = dataset.manifest.version;
    drop(dataset);
    let handler = Arc::new(crate::utils::test::AmbiguousCommitHandler::default());
    let mut dataset = crate::dataset::builder::DatasetBuilder::from_uri(dir.as_str())
        .with_commit_handler(handler.clone())
        .load()
        .await
        .unwrap();
    let incoming = staged_segment(&mut dataset, "text", IndexType::Fm, vec![10]).await;

    handler.fail_next(failure);
    let error = dataset
        .commit_existing_index_segments("staged", "text", vec![incoming.clone()])
        .await
        .unwrap_err();
    handler.fail_resolve.store(false, Ordering::SeqCst);
    assert_eq!(dataset.manifest.version, version);
    if failure == crate::utils::test::AmbiguousFailure::FailOutright {
        assert!(
            matches!(error, crate::Error::RetryableCommitConflict { .. }),
            "unexpected error: {error:?}"
        );
        assert!(
            format!("{error:?}").contains("simulated outright commit failure"),
            "{error:?}"
        );
        assert!(
            !error.to_string().contains("another commit landed first"),
            "{error}"
        );
        let untouched = fresh_session(dir.as_str()).await;
        assert_eq!(untouched.manifest.version, version);
        assert_eq!(
            stored_uuids(&untouched, "staged").await,
            HashSet::from([s.uuid])
        );

        dataset
            .commit_existing_index_segments("staged", "text", vec![incoming.clone()])
            .await
            .unwrap();
    } else {
        assert!(error.is_commit_status_unknown(), "{error:?}");
        assert!(
            format!("{error:?}").contains("simulated ambiguous commit failure"),
            "{error:?}"
        );
        // The write landed; nothing was committed after it.
        assert_eq!(
            fresh_session(dir.as_str()).await.manifest.version,
            version + 1
        );
    }
    assert_replaced_by(dir.as_str(), version, s.uuid, &incoming).await;
}

/// Appends fragments a1 and a2 to {0, 1} partitioned into {10, 11}, commits a
/// segment S of `index_type` on `column` over {a1, a2}, then deletes every row
/// of a2. S still stores a2, which is neither live nor named by the history.
/// Returns the dataset, S and a1.
async fn segment_over_an_outright_deleted_fragment(
    uri: &str,
    column: &str,
    index_type: IndexType,
) -> (Dataset, IndexMetadata, u32) {
    let dataset = partitioned_fixture(uri).await;
    let (dataset, a1) = append_live_fragment(dataset, 8).await;
    let (mut dataset, a2) = append_live_fragment(dataset, 12).await;
    let s = staged_segment(&mut dataset, column, index_type, vec![a1, a2]).await;
    dataset
        .commit_existing_index_segments("staged", column, vec![s.clone()])
        .await
        .unwrap();
    dataset.delete("i >= 12").await.unwrap();
    assert!(!fragment_ids(&dataset).contains(&a2));
    assert!(fragment_ids(&dataset).contains(&a1));
    assert_eq!(
        stored_bitmaps(&dataset, "staged").await,
        std::collections::BTreeMap::from([(s.uuid, Some(RoaringBitmap::from_iter([a1, a2])))])
    );
    (dataset, s, a1)
}

/// FM has no batch remapper. I over {a1} replaces S over {a1, a2} after a2
/// was deleted outright: a2 carries no coverage, so I supplies S again.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn a_rebuild_after_an_outright_delete_replaces_a_segment_without_a_batch_remapper() {
    let dir = TempStrDir::default();
    let (mut dataset, s, a1) =
        segment_over_an_outright_deleted_fragment(dir.as_str(), "text", IndexType::Fm).await;
    let version = dataset.manifest.version;
    let incoming = staged_segment(&mut dataset, "text", IndexType::Fm, vec![a1]).await;

    dataset
        .commit_existing_index_segments("staged", "text", vec![incoming.clone()])
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, version + 1);
    assert_eq!(
        stored_bitmaps(&dataset, "staged").await,
        std::collections::BTreeMap::from([(incoming.uuid, incoming.fragment_bitmap.clone())]),
        "{} is replaced",
        s.uuid
    );
    assert_eq!(
        derived_coverage(&dataset, "staged").await,
        Some(RoaringBitmap::from_iter([a1]))
    );
}

/// The same replacement for BTree rests on no derived coverage, so it rebases
/// past an append that commits between the build and the commit.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn a_rebuild_after_an_outright_delete_rebases_past_a_concurrent_append() {
    let dir = TempStrDir::default();
    let (dataset, s, a1) =
        segment_over_an_outright_deleted_fragment(dir.as_str(), "i", IndexType::BTree).await;
    let mut stale = dataset.clone();
    let incoming = staged_segment(&mut stale, "i", IndexType::BTree, vec![a1]).await;
    let latest = append_scalar_rows(dataset, 20, 4).await;

    stale
        .commit_existing_index_segments("staged", "i", vec![incoming.clone()])
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, latest.manifest.version + 1);
    assert_eq!(
        stored_bitmaps(&dataset, "staged").await,
        std::collections::BTreeMap::from([(incoming.uuid, incoming.fragment_bitmap.clone())]),
        "{} is replaced",
        s.uuid
    );
    assert_eq!(
        derived_coverage(&dataset, "staged").await,
        Some(RoaringBitmap::from_iter([a1]))
    );
}

/// On {0, 1} partitioned into {10, 11}, segments M over appended a1 and D over
/// appended a2 of `index_type` on `column`; then every row of a2 is deleted
/// and a3 appended. D stores only a2, which is neither live nor named by the
/// history. Returns the dataset, M, D, a1 and a3.
async fn segment_over_only_an_outright_deleted_fragment(
    uri: &str,
    column: &str,
    index_type: IndexType,
) -> (Dataset, IndexMetadata, IndexMetadata, u32, u32) {
    let dataset = partitioned_fixture(uri).await;
    let (dataset, a1) = append_live_fragment(dataset, 8).await;
    let (mut dataset, a2) = append_live_fragment(dataset, 12).await;
    let m = staged_segment(&mut dataset, column, index_type, vec![a1]).await;
    let d = staged_segment(&mut dataset, column, index_type, vec![a2]).await;
    dataset
        .commit_existing_index_segments("staged", column, vec![m.clone(), d.clone()])
        .await
        .unwrap();
    dataset.delete("i >= 12").await.unwrap();
    assert!(!fragment_ids(&dataset).contains(&a2));
    let (dataset, a3) = append_live_fragment(dataset, 16).await;
    assert_eq!(
        stored_bitmaps(&dataset, "staged").await,
        std::collections::BTreeMap::from([
            (m.uuid, Some(RoaringBitmap::from_iter([a1]))),
            (d.uuid, Some(RoaringBitmap::from_iter([a2]))),
        ])
    );
    (dataset, m, d, a1, a3)
}

/// Sorted live `i` values of [`segment_over_only_an_outright_deleted_fragment`]
/// plus `appended` rows from 20.
fn live_values_after_the_outright_delete(appended: i32) -> Vec<i32> {
    (0..12).chain(16..20 + appended).collect()
}

/// Whether committing `segments` as `staged` on `column` is decided to land
/// only on `dataset`'s snapshot.
async fn requires_unchanged_snapshot(
    dataset: &Dataset,
    column: &str,
    segments: Vec<IndexMetadata>,
) -> bool {
    let segments = segments
        .into_iter()
        .map(crate::index::api::IntoIndexSegment::into_index_segment)
        .collect::<crate::Result<Vec<_>>>()
        .unwrap();
    crate::index::lifecycle::decide_segment_commit(dataset, "staged", column, segments)
        .await
        .unwrap()
        .1
}

/// FM has no batch remapper. D serves nothing: its only stored fragment was
/// deleted outright. I over a3 removes it like a segment storing no fragment
/// and keeps M: the decision needs no coverage proof, so it does not require
/// an unchanged snapshot. The manifest cleanup is what this asserts; the
/// scans afterwards only check that no row was lost.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn a_partial_commit_removes_a_segment_over_only_an_outright_deleted_fragment() {
    let dir = TempStrDir::default();
    let (mut dataset, m, d, a1, a3) =
        segment_over_only_an_outright_deleted_fragment(dir.as_str(), "text", IndexType::Fm).await;
    let version = dataset.manifest.version;
    let incoming = staged_segment(&mut dataset, "text", IndexType::Fm, vec![a3]).await;
    assert!(!requires_unchanged_snapshot(&dataset, "text", vec![incoming.clone()]).await);

    dataset
        .commit_existing_index_segments("staged", "text", vec![incoming.clone()])
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, version + 1);
    assert_eq!(
        stored_bitmaps(&dataset, "staged").await,
        std::collections::BTreeMap::from([
            (m.uuid, m.fragment_bitmap.clone()),
            (incoming.uuid, incoming.fragment_bitmap.clone()),
        ]),
        "{} is removed",
        d.uuid
    );
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([a1, a3])
    );
    let mut unindexed = unindexed_fragment_ids(&dataset, "staged").await;
    unindexed.sort_unstable();
    assert_eq!(unindexed, vec![10, 11]);
    let expected = live_values_after_the_outright_delete(0);
    assert_eq!(values(&dataset, None, true).await, expected);
    assert_eq!(
        values(&dataset, Some("text = 'document'"), true).await,
        expected
    );
}

/// The same removal for BTree rests on no derived coverage either, so it does
/// not require an unchanged snapshot and rebases past an append that commits
/// between the build and the commit.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn a_removal_over_only_an_outright_deleted_fragment_rebases_past_a_concurrent_append() {
    let dir = TempStrDir::default();
    let (dataset, m, d, a1, a3) =
        segment_over_only_an_outright_deleted_fragment(dir.as_str(), "i", IndexType::BTree).await;
    let mut stale = dataset.clone();
    let incoming = staged_segment(&mut stale, "i", IndexType::BTree, vec![a3]).await;
    assert!(!requires_unchanged_snapshot(&stale, "i", vec![incoming.clone()]).await);
    let latest = append_scalar_rows(dataset, 20, 4).await;

    stale
        .commit_existing_index_segments("staged", "i", vec![incoming.clone()])
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, latest.manifest.version + 1);
    assert_eq!(
        stored_bitmaps(&dataset, "staged").await,
        std::collections::BTreeMap::from([
            (m.uuid, m.fragment_bitmap.clone()),
            (incoming.uuid, incoming.fragment_bitmap.clone()),
        ]),
        "{} is removed",
        d.uuid
    );
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([a1, a3])
    );
    assert_each_row_served_once(&dataset, "staged").await;
    let expected = live_values_after_the_outright_delete(4);
    assert_eq!(values(&dataset, None, true).await, expected);
    for value in expected {
        assert_eq!(
            values(&dataset, Some(&format!("i = {value}")), true).await,
            vec![value]
        );
    }
}

/// FTS segments A over {0, 2} and B over {1, 3} as in
/// [`mixed_destination_fixture`], and D over an appended fragment whose rows
/// are then all deleted: D serves nothing. C over {2, 10, 11} replaces A,
/// which rests on the coverage proof. D's unreadable format marker does not
/// leave that proof unproven: D takes no part in it, so the commit lands and
/// removes A and D.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn a_segment_serving_nothing_does_not_leave_a_replacement_unproven() {
    let dir = TempStrDir::default();
    let path = dir.as_str();
    let (dataset, [a, b]) = mixed_destination_fixture(path, "text", IndexType::Inverted).await;
    let (mut dataset, deleted) = append_live_fragment(dataset, 16).await;
    let d = staged_fts_segment(&mut dataset, "text", vec![deleted]).await;
    dataset
        .commit_existing_index_segments("staged", "text", vec![d.clone()])
        .await
        .unwrap();
    dataset.delete("i >= 16").await.unwrap();
    assert!(!fragment_ids(&dataset).contains(&deleted));
    let incoming = staged_fts_segment(&mut dataset, "text", vec![2, 10, 11]).await;
    let version = dataset.manifest.version;
    drop(dataset);

    let (mut dataset, failing) = open_with_failing_reads(path, fts_marker_suffix(d.uuid)).await;
    failing.store(true, Ordering::SeqCst);
    assert!(requires_unchanged_snapshot(&dataset, "text", vec![incoming.clone()]).await);
    dataset
        .commit_existing_index_segments("staged", "text", vec![incoming.clone()])
        .await
        .unwrap();
    failing.store(false, Ordering::SeqCst);

    let dataset = fresh_session(path).await;
    assert_eq!(dataset.manifest.version, version + 1);
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([b, incoming.uuid]),
        "{a} and {} are removed",
        d.uuid
    );
    assert_eq!(
        derived_union(&dataset, "staged").await,
        RoaringBitmap::from_iter([2u32, 3, 10, 11])
    );
}

/// Fragments {0, 1} of 64 rows each, then {0} partitioned into {10, 11}, and
/// a legacy IVF_PQ segment H committed over {1, 10, 11}. This build has no
/// batch remapper for H's file. A second partition retires every fragment H
/// covers, so the rewrite withdraws H's coverage of them in the same commit.
/// A vector segment over a fragment appended afterwards then commits beside
/// H: H stores nothing, so the commit removes it.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn a_disjoint_vector_commit_beside_a_legacy_segment_after_a_rewrite() {
    let dir = TempStrDir::default();
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col(
            "vector",
            lance_datagen::array::rand_vec::<arrow_array::types::Float32Type>(4.into()),
        )
        .col("w", lance_datagen::array::step::<Int32Type>())
        .into_dataset(
            dir.as_str(),
            FragmentCount::from(2),
            FragmentRowCount::from(64),
        )
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 40).await;
    let mut dataset = commit_stable_partition(dataset, &[0], 10).await;
    let mut legacy_params = crate::index::vector::VectorIndexParams::ivf_pq(
        1,
        4,
        2,
        lance_linalg::distance::DistanceType::L2,
        10,
    );
    legacy_params.version(crate::index::vector::IndexFileVersion::Legacy);
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("staged".to_string()),
            &legacy_params,
            true,
        )
        .await
        .unwrap();
    let h = stored_segment(&dataset, "staged").await;
    assert_eq!(
        h.fragment_bitmap,
        Some(RoaringBitmap::from_iter([1u32, 10, 11]))
    );

    let dataset = commit_stable_partition(dataset, &[1, 10, 11], 30).await;
    assert_eq!(
        stored_bitmaps(&dataset, "staged").await,
        std::collections::BTreeMap::from([(h.uuid, Some(RoaringBitmap::new()))])
    );
    let mut dataset = append_vector_rows(dataset, 128, 64).await;
    let appended = *fragment_ids(&dataset).iter().max().unwrap();
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
    let incoming = CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
        .name("staged".to_string())
        .fragments(vec![appended])
        .replace(true)
        .execute_uncommitted()
        .await
        .unwrap();

    dataset
        .commit_existing_index_segments("staged", "vector", vec![incoming.clone()])
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([incoming.uuid])
    );
}

/// A staged IVF_FLAT segment of `staged` over `fragments`, on one fixed
/// centroid so every such segment shares the query contract.
async fn staged_vector_segment(dataset: &mut Dataset, fragments: Vec<u32>) -> IndexMetadata {
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
    CreateIndexBuilder::new(dataset, &["vector"], IndexType::Vector, &params)
        .name("staged".to_string())
        .fragments(fragments)
        .replace(true)
        .execute_uncommitted()
        .await
        .unwrap()
}

/// A vector segment P over {0}, then {0} stable-partitioned. P stores only
/// the retired 0, so a partial commit of Q over {1} keeps it whenever its
/// coverage cannot be ruled out, and then has to check P against Q. With P's
/// index file unreadable that check cannot run: the commit fails with the
/// read failure itself and commits nothing. Once the store answers, the same
/// commit keeps P beside Q.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn a_partial_vector_commit_beside_an_unreadable_retained_segment_commits_nothing() {
    let dir = TempStrDir::default();
    let path = dir.as_str();
    let mut dataset = vector_fixture(path).await;
    let p = staged_vector_segment(&mut dataset, vec![0]).await;
    dataset
        .commit_existing_index_segments("staged", "vector", vec![p.clone()])
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let mut dataset = commit_stable_partition(dataset, &[0], 10).await;
    assert!(!fragment_ids(&dataset).contains(&0));
    let q = staged_vector_segment(&mut dataset, vec![1]).await;
    let version = dataset.manifest.version;
    drop(dataset);
    let before = stored_bitmaps(&fresh_session(path).await, "staged").await;

    let (mut dataset, failing) =
        open_with_failing_reads(path, format!("{}/{}", p.uuid, lance_index::INDEX_FILE_NAME)).await;
    failing.store(true, Ordering::SeqCst);
    let result = dataset
        .commit_existing_index_segments("staged", "vector", vec![q.clone()])
        .await;
    failing.store(false, Ordering::SeqCst);
    let error = result.unwrap_err();
    assert!(
        matches!(error, crate::Error::IO { .. }),
        "unexpected error: {error:?}"
    );
    assert!(
        format!("{error:?}").contains("injected transient read failure"),
        "{error:?}"
    );
    assert_eq!(dataset.manifest.version, version);
    let untouched = fresh_session(path).await;
    assert_eq!(untouched.manifest.version, version);
    assert_eq!(stored_bitmaps(&untouched, "staged").await, before);

    dataset
        .commit_existing_index_segments("staged", "vector", vec![q.clone()])
        .await
        .unwrap();
    let dataset = fresh_session(path).await;
    assert_eq!(dataset.manifest.version, version + 1);
    assert_eq!(
        stored_uuids(&dataset, "staged").await,
        HashSet::from([p.uuid, q.uuid])
    );
}
