// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index segments staged before a deferred compaction, merged and committed
//! after it: the merged coverage follows rows through the compaction only for
//! fragments whose indexed data has not changed since the segments were built.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::scalar::BuiltinIndexType;
use lance_index::scalar::ScalarIndexParams;
use lance_table::format::IndexMetadata;
use lance_table::format::overlay::OverlayCoverage;
use roaring::RoaringBitmap;
use rstest::rstest;

use super::dataset_overlay_index_masking::{
    commit_overlay, create_base_dataset_with, i32_array, ids_matching,
};
use crate::Dataset;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::transaction::Operation;
use crate::dataset::{
    MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched, WriteDestination,
    WriteParams,
};
use crate::index::DatasetIndexExt;
#[cfg(feature = "geo")]
use crate::utils::test::geo;

/// `age` staged one segment per fragment; a committed index on `id` gives the
/// compaction an index to defer.
async fn btree_staged() -> (Dataset, Vec<lance_table::format::IndexMetadata>) {
    let mut dataset = create_base_dataset_with(false).await;
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
    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect();
    let staged = crate::utils::test::stage_index_segments(
        &mut dataset,
        "age",
        IndexType::BTree,
        &ScalarIndexParams::default(),
        "age_staged",
        fragment_ids,
    )
    .await;
    (dataset, staged)
}

/// Compacts everything into one fragment with deferred remap; returns its id.
async fn btree_compact(dataset: &mut Dataset) -> u64 {
    compact_files(
        dataset,
        CompactionOptions {
            target_rows_per_fragment: 12,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    dataset.get_fragments()[0].id() as u64
}

/// [`btree_staged`], then [`btree_compact`].
async fn btree_staged_over_a_compaction() -> (Dataset, Vec<lance_table::format::IndexMetadata>, u64)
{
    let (mut dataset, staged) = btree_staged().await;
    let rewritten = btree_compact(&mut dataset).await;
    (dataset, staged, rewritten)
}

/// Compacts, then merges and commits the `age` segments.
async fn btree_merge_and_commit(
    mut dataset: Dataset,
    staged: Vec<lance_table::format::IndexMetadata>,
) -> Dataset {
    btree_compact(&mut dataset).await;
    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    dataset
        .commit_existing_index_segments("age_staged", "age", vec![merged])
        .await
        .unwrap();
    dataset
}

/// A restore removes an overlay the segments indexed: coverage is dropped.
#[tokio::test]
async fn test_btree_merge_drops_coverage_a_restore_took_an_overlay_from() {
    let mut dataset = create_base_dataset_with(false).await;
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
    let before_overlay = dataset.manifest.version;
    let fragment = dataset.get_fragments()[0].id() as u64;
    let mut dataset = commit_overlay(
        dataset,
        "age_overlay",
        fragment,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
        vec![i32_array([Some(999)])],
    )
    .await;
    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect();
    let staged = crate::utils::test::stage_index_segments(
        &mut dataset,
        "age",
        IndexType::BTree,
        &ScalarIndexParams::default(),
        "age_staged",
        fragment_ids,
    )
    .await;

    let mut dataset = dataset.checkout_version(before_overlay).await.unwrap();
    dataset.restore().await.unwrap();
    let dataset = btree_merge_and_commit(dataset, staged).await;

    assert_eq!(
        ids_matching(&dataset, "age = 0").await,
        vec![0],
        "the restored value is missing: the merged index still answers with the \
         overlay the restore took away"
    );
    assert!(ids_matching(&dataset, "age = 999").await.is_empty());
}

/// A restore revives a row the segments never indexed: coverage is dropped.
#[tokio::test]
async fn test_btree_merge_drops_coverage_a_restore_revived_a_row_in() {
    let mut dataset = create_base_dataset_with(false).await;
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
    let before_delete = dataset.manifest.version;
    dataset.delete("id = 3").await.unwrap();
    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect();
    let staged = crate::utils::test::stage_index_segments(
        &mut dataset,
        "age",
        IndexType::BTree,
        &ScalarIndexParams::default(),
        "age_staged",
        fragment_ids,
    )
    .await;

    let mut dataset = dataset.checkout_version(before_delete).await.unwrap();
    dataset.restore().await.unwrap();
    let dataset = btree_merge_and_commit(dataset, staged).await;

    assert_eq!(
        ids_matching(&dataset, "age = 30").await,
        vec![3],
        "the revived row is missing: the merged index claimed a fragment holding a \
         row it never indexed"
    );
}

/// Rewrites `column` of `fragment` into a new file.
async fn replace_column(
    dataset: Dataset,
    fragment: u64,
    column: &str,
    values: ArrayRef,
) -> Dataset {
    let schema = dataset.schema().project(&[column]).unwrap();
    let batch = RecordBatch::try_new(Arc::new(ArrowSchema::from(&schema)), vec![values]).unwrap();
    let replacement = dataset
        .get_fragment(fragment as usize)
        .unwrap()
        .write_columns(futures::stream::iter([Ok(batch)]), &schema)
        .await
        .unwrap();
    let read_version = dataset.version().version;
    Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::DataReplacement {
            replacements: vec![replacement],
            data_change: true,
        },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap()
}

/// Rows per fragment in the RTree tests.
#[cfg(feature = "geo")]
const RTREE_ROWS_PER_FRAGMENT: i32 = 10;

fn fragment_ids(dataset: &Dataset) -> Vec<u32> {
    dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect()
}

#[cfg(feature = "geo")]
async fn rtree_compact(dataset: &mut Dataset, fragments_per_group: i32) {
    compact_files(
        dataset,
        geo::deferred_compaction(RTREE_ROWS_PER_FRAGMENT, fragments_per_group),
        None,
    )
    .await
    .unwrap();
}

/// Overlays the geometry of one row of `fragment_id`.
#[cfg(feature = "geo")]
async fn rtree_overlay_geometry(dataset: Dataset, field_id: i32, fragment_id: u64) -> Dataset {
    commit_overlay(
        dataset,
        "geometry_overlay",
        fragment_id,
        &[field_id],
        OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
        vec![geo::line_strings(9_999, 1)],
    )
    .await
}

/// Asserts the merged index covers nothing, so its rows are scanned.
#[cfg(feature = "geo")]
async fn assert_merge_covers_nothing(
    dataset: &Dataset,
    staged: Vec<lance_table::format::IndexMetadata>,
    what: &str,
) {
    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    let coverage = merged
        .fragment_bitmap
        .as_ref()
        .expect("a merged segment records what it covers");
    assert!(
        coverage.is_empty(),
        "the merged index claimed a fragment {what}, so its superseded entries \
         would be served instead of rescanned"
    );
}

/// A dataset with a committed RTree index and RTree segments staged over every
/// fragment, plus the geometry field id.
#[cfg(feature = "geo")]
async fn rtree_staged(
    dir: &TempStrDir,
    fragments: i32,
) -> (Dataset, Vec<lance_table::format::IndexMetadata>, i32) {
    let (mut dataset, params) =
        geo::dataset_with_committed_rtree_index(dir.as_str(), RTREE_ROWS_PER_FRAGMENT, fragments)
            .await;
    let geometry = dataset.schema().field("geometry").unwrap().id;
    let sources = fragment_ids(&dataset);
    let staged = geo::stage_rtree_segments(&mut dataset, &params, sources).await;
    (dataset, staged, geometry)
}

/// When an overlay on the indexed column lands, relative to the compactions.
#[cfg(feature = "geo")]
#[derive(Clone, Copy, Debug)]
enum OverlayTiming {
    /// On the compaction's output.
    AfterCompaction,
    /// On a source; the compaction folds it into the output.
    BeforeCompaction,
    /// On a source, then two compactions; the fragment that folded it is gone.
    BeforeTwoCompactions,
    /// On the first compaction's output, folded in by the second.
    BetweenCompactions,
}

/// Segments built before an overlay hold stale values: wherever it lands, the
/// merge covers nothing.
#[cfg(feature = "geo")]
#[rstest]
#[case::after_compaction(OverlayTiming::AfterCompaction)]
#[case::before_compaction(OverlayTiming::BeforeCompaction)]
#[case::before_two_compactions(OverlayTiming::BeforeTwoCompactions)]
#[case::between_compactions(OverlayTiming::BetweenCompactions)]
#[tokio::test]
async fn test_rtree_merge_drops_coverage_an_overlay_reaches(#[case] timing: OverlayTiming) {
    let dir = TempStrDir::default();
    let fragments = match timing {
        OverlayTiming::AfterCompaction | OverlayTiming::BeforeCompaction => 3,
        OverlayTiming::BeforeTwoCompactions | OverlayTiming::BetweenCompactions => 4,
    };
    let (mut dataset, staged, geometry) = rtree_staged(&dir, fragments).await;
    let first = |dataset: &Dataset| fragment_ids(dataset)[0] as u64;

    match timing {
        OverlayTiming::AfterCompaction => {
            rtree_compact(&mut dataset, 3).await;
            let produced = first(&dataset);
            dataset = rtree_overlay_geometry(dataset, geometry, produced).await;
        }
        OverlayTiming::BeforeCompaction => {
            let source = first(&dataset);
            dataset = rtree_overlay_geometry(dataset, geometry, source).await;
            rtree_compact(&mut dataset, 3).await;
        }
        OverlayTiming::BeforeTwoCompactions => {
            let source = first(&dataset);
            dataset = rtree_overlay_geometry(dataset, geometry, source).await;
            rtree_compact(&mut dataset, 2).await;
            rtree_compact(&mut dataset, 4).await;
            assert_eq!(dataset.get_fragments().len(), 1);
        }
        OverlayTiming::BetweenCompactions => {
            rtree_compact(&mut dataset, 2).await;
            assert_eq!(dataset.get_fragments().len(), 2);
            let intermediate = first(&dataset);
            dataset = rtree_overlay_geometry(dataset, geometry, intermediate).await;
            rtree_compact(&mut dataset, 4).await;
            assert_eq!(dataset.get_fragments().len(), 1);
        }
    }

    assert_merge_covers_nothing(
        &dataset,
        staged,
        &format!("an overlay reached ({timing:?})"),
    )
    .await;
}

/// An overlay folded into one group does not cost coverage of another group.
#[cfg(feature = "geo")]
#[tokio::test]
async fn test_rtree_merge_keeps_coverage_when_a_materialized_overlay_is_on_another_group() {
    let dir = TempStrDir::default();
    let (mut dataset, params) =
        geo::dataset_with_committed_rtree_index(dir.as_str(), RTREE_ROWS_PER_FRAGMENT, 4).await;
    let geometry = dataset.schema().field("geometry").unwrap().id;
    // Only the pair the overlay leaves alone.
    let ours = fragment_ids(&dataset).split_off(2);
    let staged = geo::stage_rtree_segments(&mut dataset, &params, ours).await;

    let overlaid = fragment_ids(&dataset)[0] as u64;
    let mut dataset = rtree_overlay_geometry(dataset, geometry, overlaid).await;
    rtree_compact(&mut dataset, 2).await;
    assert_eq!(dataset.get_fragments().len(), 2);

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    let coverage = merged
        .fragment_bitmap
        .as_ref()
        .expect("a merged segment records what it covers");
    assert_eq!(
        coverage.len(),
        1,
        "the merged index gave up coverage over an overlay materialized into a \
         rewrite group its segments never covered, costing a flat scan for nothing"
    );
}

/// The same rule for BTree: an overlay folded in by compaction drops coverage.
#[tokio::test]
async fn test_btree_merge_drops_a_fragment_a_compaction_materialized_an_overlay_into() {
    let (dataset, staged) = btree_staged().await;
    let source = dataset.get_fragments()[0].id() as u64;
    let mut dataset = commit_overlay(
        dataset,
        "age_overlay",
        source,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
        vec![i32_array([Some(999)])],
    )
    .await;
    btree_compact(&mut dataset).await;
    assert!(
        dataset
            .get_fragments()
            .iter()
            .all(|fragment| fragment.metadata().overlays.is_empty()),
        "the compaction materializes the overlay into the fragment it writes"
    );

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    dataset
        .commit_existing_index_segments("age_staged", "age", vec![merged])
        .await
        .unwrap();
    assert_eq!(
        ids_matching(&dataset, "age = 999").await,
        vec![0],
        "the overlaid value is missing: the merged index claimed the fragment the \
         compaction materialized the overlay into"
    );
    assert!(
        ids_matching(&dataset, "age = 0").await.is_empty(),
        "the value the overlay replaced is still answered"
    );
}

/// A data replacement after the compaction drops coverage (RTree).
#[cfg(feature = "geo")]
#[tokio::test]
async fn test_rtree_merge_drops_a_fragment_replaced_since_the_compaction() {
    let dir = TempStrDir::default();
    let (mut dataset, params) =
        geo::dataset_with_committed_rtree_index(dir.as_str(), RTREE_ROWS_PER_FRAGMENT, 3).await;
    let sources = fragment_ids(&dataset);
    let staged = geo::stage_rtree_segments(&mut dataset, &params, sources).await;
    rtree_compact(&mut dataset, 3).await;
    let rewritten = fragment_ids(&dataset)[0];

    let dataset = replace_column(
        dataset,
        rewritten as u64,
        "geometry",
        geo::line_strings(9_999, RTREE_ROWS_PER_FRAGMENT * 3),
    )
    .await;

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    assert!(
        !merged.fragment_bitmap.as_ref().unwrap().contains(rewritten),
        "the merged index claimed a remapped fragment whose geometry file was replaced"
    );
}

/// A data replacement after the compaction drops coverage (BTree).
#[tokio::test]
async fn test_btree_merge_drops_a_fragment_replaced_since_the_compaction() {
    let (dataset, staged, rewritten) = btree_staged_over_a_compaction().await;
    let replaced = Arc::new(Int32Array::from_iter_values((0..12).map(|_| 999))) as ArrayRef;
    let mut dataset = replace_column(dataset, rewritten, "age", replaced).await;

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    dataset
        .commit_existing_index_segments("age_staged", "age", vec![merged])
        .await
        .unwrap();
    assert_eq!(
        ids_matching(&dataset, "age = 999").await.len(),
        12,
        "the replaced values are missing: the merged index claimed a fragment whose \
         indexed column was rewritten after the compaction produced it"
    );
}

/// An in-place update on the compaction's output drops coverage (BTree).
#[tokio::test]
async fn test_btree_merge_drops_a_fragment_updated_in_place_since_the_compaction() {
    use crate::dataset::{MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched};
    let (mut dataset, staged, rewritten) = btree_staged_over_a_compaction().await;
    // A column the patch leaves alone, so the update rewrites `age` in place.
    dataset
        .add_columns(
            crate::dataset::NewColumnTransform::SqlExpressions(vec![("spare".into(), "42".into())]),
            None,
            None,
        )
        .await
        .unwrap();
    let patch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, true),
            ArrowField::new("age", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Int32Array::from(vec![999])),
        ],
    )
    .unwrap();
    let schema = patch.schema();
    let mut merge = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".into()]).unwrap();
    merge
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns);
    let (dataset, _) = merge
        .try_build()
        .unwrap()
        .execute_reader(RecordBatchIterator::new([Ok(patch)], schema))
        .await
        .unwrap();
    let mut dataset = Arc::unwrap_or_clone(dataset);
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1, "the update must stay in place");
    assert_eq!(fragments[0].id() as u64, rewritten);

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    dataset
        .commit_existing_index_segments("age_staged", "age", vec![merged])
        .await
        .unwrap();
    assert_eq!(ids_matching(&dataset, "age = 999").await.len(), 1);
    assert_eq!(ids_matching(&dataset, "age = 0").await, Vec::<i32>::new());
    let covered = dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .filter(|index| index.name == "age_staged")
        .map(|index| index.fragment_bitmap.clone().unwrap())
        .fold(RoaringBitmap::new(), |covered, bitmap| covered | bitmap);
    assert!(!covered.contains(rewritten as u32));
}

/// An overlay on the compaction's output drops coverage (BTree).
#[tokio::test]
async fn test_btree_merge_drops_a_fragment_overlaid_since_the_compaction() {
    let (dataset, staged, rewritten) = btree_staged_over_a_compaction().await;

    let dataset = commit_overlay(
        dataset,
        "age_overlay",
        rewritten,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
        vec![i32_array([Some(999)])],
    )
    .await;

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    assert!(
        merged.fragment_bitmap.as_ref().unwrap().is_empty(),
        "the merged index claimed a fragment an overlay has changed since the \
         compaction produced it"
    );
}

/// A data replacement on a source before the compaction drops coverage.
#[cfg(feature = "geo")]
#[tokio::test]
async fn test_rtree_merge_drops_a_fragment_replaced_before_the_compaction() {
    let dir = TempStrDir::default();
    let (schema, batches) = geo::batches(RTREE_ROWS_PER_FRAGMENT, 3);
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(
        reader,
        dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: RTREE_ROWS_PER_FRAGMENT as usize,
            enable_stable_row_ids: false,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    // On `id`, so the compaction has an index to defer.
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("committed_id_idx".to_string()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::RTree);
    let source = fragment_ids(&dataset);
    let staged = geo::stage_rtree_segments(&mut dataset, &params, source.clone()).await;
    let replaced = source[0];

    let mut dataset = replace_column(
        dataset,
        replaced as u64,
        "geometry",
        geo::line_strings(9_999, RTREE_ROWS_PER_FRAGMENT),
    )
    .await;

    let before = dataset
        .merge_existing_index_segments(staged.clone())
        .await
        .unwrap();
    assert!(
        !before.fragment_bitmap.as_ref().unwrap().contains(replaced),
        "a merge before the compaction must already refuse the replaced source"
    );

    rtree_compact(&mut dataset, 3).await;
    let rewritten = fragment_ids(&dataset)[0];
    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    assert!(
        !merged.fragment_bitmap.as_ref().unwrap().contains(rewritten),
        "the merged index claimed a fragment the compaction wrote replacement \
         geometry into"
    );
}

/// If the manifest the compaction committed is cleaned up, coverage is dropped.
#[tokio::test]
async fn test_btree_merge_drops_a_fragment_whose_creation_manifest_was_cleaned_up() {
    let (dataset, staged, rewritten) = btree_staged_over_a_compaction().await;
    let compaction = dataset.manifest.version;
    let kept = dataset
        .merge_existing_index_segments(staged.clone())
        .await
        .unwrap();
    assert_eq!(
        kept.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([rewritten as u32]),
        "with its history intact the merge follows the rows through the compaction"
    );

    let compaction_manifest = dataset
        .checkout_version(compaction)
        .await
        .unwrap()
        .manifest_location
        .path
        .clone();
    dataset
        .object_store
        .delete(&compaction_manifest)
        .await
        .unwrap();
    let retained = dataset
        .versions()
        .await
        .unwrap()
        .iter()
        .map(|version| version.version)
        .collect::<Vec<_>>();
    assert!(!retained.contains(&compaction));
    assert!(
        (1..compaction).all(|version| retained.contains(&version)),
        "only the compaction's own manifest is gone"
    );

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    assert!(
        merged.fragment_bitmap.as_ref().unwrap().is_empty(),
        "the merged index claimed a fragment it can no longer prove anything about: \
         the manifest the compaction wrote has been cleaned up"
    );
}

/// Two groups compacted separately, the second overlaid first: the first
/// compaction does not vouch for the second's inputs, and the second's
/// staleness does not cost the first its coverage.
#[cfg(feature = "geo")]
#[tokio::test]
async fn test_rtree_merge_drops_a_group_overlaid_while_another_was_compacted() {
    let dir = TempStrDir::default();
    let (mut dataset, params) =
        geo::dataset_with_committed_rtree_index(dir.as_str(), RTREE_ROWS_PER_FRAGMENT, 4).await;
    let sources = fragment_ids(&dataset);
    let geometry = dataset.schema().field("geometry").unwrap().id;
    let staged = geo::stage_rtree_segments(&mut dataset, &params, sources.clone()).await;

    // Overlay the second group before either compaction.
    let held_back = [sources[2], sources[3]];
    let mut dataset = rtree_overlay_geometry(dataset, geometry, held_back[0] as u64).await;

    let pair = CompactionOptions {
        target_rows_per_fragment: (RTREE_ROWS_PER_FRAGMENT * 2) as usize,
        defer_index_remap: true,
        ..Default::default()
    };
    compact_files(
        &mut dataset,
        CompactionOptions {
            excluded_fragment_ids: held_back.to_vec(),
            ..pair.clone()
        },
        None,
    )
    .await
    .unwrap();
    let clean = fragment_ids(&dataset)
        .into_iter()
        .find(|id| !held_back.contains(id))
        .unwrap();
    compact_files(&mut dataset, pair, None).await.unwrap();

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    let coverage = merged
        .fragment_bitmap
        .as_ref()
        .expect("a merged segment records what it covers");
    assert_eq!(
        coverage.iter().collect::<Vec<_>>(),
        vec![clean],
        "only the clean group's output may be covered: the overlaid group changed, \
         and another group's compaction does not vouch for it"
    );
}

/// Version-history reads by a merge over `fragments` fragments compacted in pairs.
#[cfg(feature = "geo")]
async fn rtree_merge_history_reads(dir: &TempStrDir, fragments: i32) -> usize {
    let (mut dataset, params) =
        geo::dataset_with_committed_rtree_index(dir.as_str(), RTREE_ROWS_PER_FRAGMENT, fragments)
            .await;
    let sources = fragment_ids(&dataset);
    let staged = geo::stage_rtree_segments(&mut dataset, &params, sources).await;
    rtree_compact(&mut dataset, 2).await;
    let groups = fragments / 2;
    assert_eq!(dataset.get_fragments().len(), groups as usize);

    let _ = dataset.object_store.as_ref().io_stats_incremental();
    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap().len() as i32,
        groups,
        "every group is covered whole, so coverage follows the rows"
    );
    dataset
        .object_store
        .as_ref()
        .io_stats_incremental()
        .requests
        .iter()
        .filter(|request| request.path.to_string().contains("_versions"))
        .count()
}

/// History reads do not grow with the number of groups one compaction rewrites.
#[cfg(feature = "geo")]
#[tokio::test]
async fn test_rtree_merge_dates_every_group_of_a_compaction_together() {
    let two = TempStrDir::default();
    let four = TempStrDir::default();
    let two_groups = rtree_merge_history_reads(&two, 4).await;
    let four_groups = rtree_merge_history_reads(&four, 8).await;
    assert_eq!(
        four_groups, two_groups,
        "doubling the groups took the merge from {two_groups} reads of version \
         history to {four_groups}: it reads history per group rather than once per \
         compaction"
    );
}

/// Unrelated commits after a retired intermediate output do not cost coverage.
#[cfg(feature = "geo")]
#[tokio::test]
async fn test_rtree_merge_keeps_coverage_when_later_commits_follow_a_retired_output() {
    let dir = TempStrDir::default();
    let (mut dataset, params) =
        geo::dataset_with_committed_rtree_index(dir.as_str(), RTREE_ROWS_PER_FRAGMENT, 4).await;
    let sources = fragment_ids(&dataset);
    let staged = geo::stage_rtree_segments(&mut dataset, &params, sources).await;

    // The second compaction retires the first one's outputs.
    rtree_compact(&mut dataset, 2).await;
    rtree_compact(&mut dataset, 4).await;
    assert_eq!(dataset.get_fragments().len(), 1);

    // Commits that touch no fragment at all.
    for setting in 0..10 {
        dataset
            .update_config([("unrelated".to_string(), setting.to_string())])
            .await
            .unwrap();
    }

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap().len(),
        1,
        "coverage was given up over commits that changed no geometry"
    );
}

/// A group retired without a reuse mapping does not cost a mapped group its
/// coverage.
#[cfg(feature = "geo")]
#[tokio::test]
async fn test_rtree_merge_keeps_a_mapped_group_when_another_was_retired_unmapped() {
    let dir = TempStrDir::default();
    let (mut dataset, params) =
        geo::dataset_with_committed_rtree_index(dir.as_str(), RTREE_ROWS_PER_FRAGMENT, 4).await;
    let sources = fragment_ids(&dataset);
    let staged = geo::stage_rtree_segments(&mut dataset, &params, sources.clone()).await;

    let pair = |excluded: Vec<u32>, defer: bool| CompactionOptions {
        target_rows_per_fragment: (RTREE_ROWS_PER_FRAGMENT * 2) as usize,
        defer_index_remap: defer,
        excluded_fragment_ids: excluded,
        ..Default::default()
    };
    // One pair remapped inline, so it leaves no mapping.
    compact_files(&mut dataset, pair(sources[2..].to_vec(), false), None)
        .await
        .unwrap();
    // The other pair deferred.
    compact_files(&mut dataset, pair(sources[..2].to_vec(), true), None)
        .await
        .unwrap();

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap().len(),
        1,
        "the deferred pair's coverage was given up because an unrelated pair was \
         retired without a reuse mapping"
    );
}

const ROWS: i32 = 6;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Append,
    Delete,
    UpdateInPlace,
    Overlay,
    Compact,
    CompactFoldingOverlays,
    AddColumn,
    Config,
}

struct Workload {
    rng: Mutex<u64>,
    next_id: Mutex<i32>,
    values: Mutex<i32>,
}

impl Workload {
    fn next(&self, bound: u64) -> u64 {
        let mut rng = self.rng.lock().unwrap();
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        *rng % bound
    }

    /// `len` operations, with compactions only if `compact`.
    fn draw(&self, len: usize, compact: bool) -> Vec<Op> {
        use Op::*;
        let all = [
            Append,
            Delete,
            UpdateInPlace,
            Overlay,
            Compact,
            Compact,
            CompactFoldingOverlays,
            AddColumn,
            Config,
        ];
        let kinds = all
            .into_iter()
            .filter(|op| compact || !matches!(op, Compact | CompactFoldingOverlays))
            .collect::<Vec<_>>();
        (0..len)
            .map(|_| kinds[self.next(kinds.len() as u64) as usize])
            .collect()
    }

    fn new_value(&self) -> i32 {
        let mut values = self.values.lock().unwrap();
        *values += 1;
        1_000_000 + *values
    }

    async fn run(&self, mut dataset: Dataset, ops: &[Op]) -> Dataset {
        for op in ops {
            dataset.checkout_latest().await.unwrap();
            match op {
                Op::Append => {
                    let first = {
                        let mut next_id = self.next_id.lock().unwrap();
                        *next_id += ROWS;
                        *next_id - ROWS
                    };
                    let ids = (first..first + ROWS).collect::<Vec<_>>();
                    let schema = Arc::new(ArrowSchema::from(dataset.schema()));
                    let columns = schema
                        .fields()
                        .iter()
                        .map(|field| match field.name().as_str() {
                            "id" => Arc::new(Int32Array::from(ids.clone())) as ArrayRef,
                            "val" => {
                                Arc::new(Int32Array::from_iter_values(ids.iter().map(|id| id * 10)))
                            }
                            "spare" => Arc::new(Int32Array::from(vec![42; ROWS as usize])),
                            _ => arrow_array::new_null_array(field.data_type(), ROWS as usize),
                        })
                        .collect();
                    let batch = RecordBatch::try_new(schema, columns).unwrap();
                    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
                    dataset.append(reader, None).await.unwrap();
                }
                Op::Delete => {
                    let filter = format!("id % 11 = {}", self.next(11));
                    dataset.delete(&filter).await.unwrap();
                }
                Op::UpdateInPlace => {
                    let id = self.next(*self.next_id.lock().unwrap() as u64) as i32;
                    let patch = RecordBatch::try_new(
                        Arc::new(ArrowSchema::new(vec![
                            ArrowField::new("id", DataType::Int32, true),
                            ArrowField::new("val", DataType::Int32, true),
                        ])),
                        vec![
                            Arc::new(Int32Array::from(vec![id])),
                            Arc::new(Int32Array::from(vec![self.new_value()])),
                        ],
                    )
                    .unwrap();
                    let schema = patch.schema();
                    let mut merge =
                        MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["id".into()])
                            .unwrap();
                    merge
                        .when_matched(WhenMatched::UpdateAll)
                        .when_not_matched(WhenNotMatched::DoNothing)
                        .write_mode(MergeInsertWriteMode::RewriteColumns);
                    merge
                        .try_build()
                        .unwrap()
                        .execute_reader(RecordBatchIterator::new([Ok(patch)], schema))
                        .await
                        .unwrap();
                }
                Op::Overlay => {
                    let fragments = dataset.get_fragments();
                    let fragment = fragments[self.next(fragments.len() as u64) as usize].id();
                    let val = dataset.schema().field("val").unwrap().id;
                    let value = self.new_value();
                    dataset = commit_overlay(
                        dataset,
                        &format!("overlay_{value}"),
                        fragment as u64,
                        &[val],
                        OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
                        vec![Arc::new(Int32Array::from(vec![Some(value)]))],
                    )
                    .await;
                }
                Op::Compact | Op::CompactFoldingOverlays => {
                    let options = CompactionOptions {
                        target_rows_per_fragment: (ROWS * (2 + self.next(3) as i32)) as usize,
                        defer_index_remap: true,
                        max_overlays_per_fragment: (*op == Op::CompactFoldingOverlays).then_some(0),
                        ..Default::default()
                    };
                    compact_files(&mut dataset, options, None).await.unwrap();
                }
                Op::AddColumn => {
                    let name = format!("extra_{}", dataset.manifest.version);
                    dataset
                        .add_columns(
                            crate::dataset::NewColumnTransform::SqlExpressions(vec![(
                                name,
                                "CAST(NULL AS INT)".into(),
                            )]),
                            None,
                            None,
                        )
                        .await
                        .unwrap();
                }
                Op::Config => {
                    dataset.update_config([("touched", "true")]).await.unwrap();
                }
            }
        }
        dataset.checkout_latest().await.unwrap();
        dataset
    }
}

/// `fragments` fragments of `id`, `val = id * 10` and `spare`, on disk, with a
/// committed index on `id` so a deferred compaction records its groups.
async fn val_table(uri: &str, fragments: i32) -> Dataset {
    let rows = fragments * ROWS;
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, true),
            ArrowField::new("val", DataType::Int32, true),
            ArrowField::new("spare", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(0..rows)),
            Arc::new(Int32Array::from_iter_values((0..rows).map(|id| id * 10))),
            Arc::new(Int32Array::from(vec![42; rows as usize])),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    let params = WriteParams {
        max_rows_per_file: ROWS as usize,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, uri, Some(params)).await.unwrap();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    dataset
}

/// One uncommitted `val_idx` segment per fragment in `fragments`.
async fn stage_val(dataset: &Dataset, fragments: Vec<u32>) -> Vec<IndexMetadata> {
    crate::utils::test::stage_index_segments(
        &mut dataset.clone(),
        "val",
        IndexType::BTree,
        &ScalarIndexParams::default(),
        "val_idx",
        fragments,
    )
    .await
}

/// Checks that looking up each of `values`, and a few ranges, through
/// `val_idx` finds what a scan finds.
async fn assert_val_lookups_match_scan(
    dataset: &Dataset,
    values: impl IntoIterator<Item = i32>,
    trace: &str,
) {
    let mut scan = dataset.scan();
    scan.use_scalar_index(false).project(&["val"]).unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let mut scanned = HashMap::<i32, usize>::new();
    for val in batch["val"].as_primitive::<Int32Type>().values() {
        *scanned.entry(*val).or_default() += 1;
    }
    for value in values {
        let found = dataset
            .count_rows(Some(format!("val = {value}")))
            .await
            .unwrap();
        let want = scanned.get(&value).copied().unwrap_or(0);
        assert_eq!(found, want, "{trace}: val = {value}");
    }
    for bound in [0, 100, 500, 1_000, 1_000_000, 2_000_000] {
        let found = dataset
            .count_rows(Some(format!("val < {bound}")))
            .await
            .unwrap();
        let want = scanned
            .iter()
            .filter(|(value, _)| **value < bound)
            .map(|(_, count)| count)
            .sum::<usize>();
        assert_eq!(found, want, "{trace}: val < {bound}");
    }
}

async fn val_idx_coverage(dataset: &Dataset) -> RoaringBitmap {
    let index = dataset
        .load_index_by_name("val_idx")
        .await
        .unwrap()
        .unwrap();
    index.fragment_bitmap.unwrap()
}

/// A group compacted while an overlay lands on a fragment outside it: only the
/// overlaid fragment is left out of the merged coverage, and lookups stay
/// right even though the merged index is stamped current, past the overlay.
#[tokio::test]
async fn test_btree_merge_leaves_out_only_an_overlaid_fragment() {
    let dir = TempStrDir::default();
    let dataset = val_table(dir.as_str(), 3).await;
    let staged = stage_val(&dataset, vec![0, 1, 2]).await;
    let val = dataset.schema().field("val").unwrap().id;
    let mut dataset = commit_overlay(
        dataset,
        "overlay",
        2,
        &[val],
        OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
        vec![i32_array([Some(999)])],
    )
    .await;
    let options = CompactionOptions {
        target_rows_per_fragment: (2 * ROWS) as usize,
        defer_index_remap: true,
        excluded_fragment_ids: vec![2],
        ..Default::default()
    };
    compact_files(&mut dataset, options, None).await.unwrap();
    let compacted = fragment_ids(&dataset)
        .into_iter()
        .find(|id| *id != 2)
        .unwrap();

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    dataset
        .commit_existing_index_segments("val_idx", "val", vec![merged])
        .await
        .unwrap();

    let probes = (0..18).map(|id| id * 10).chain([999]);
    assert_val_lookups_match_scan(&dataset, probes, "").await;
    assert_eq!(
        val_idx_coverage(&dataset).await,
        RoaringBitmap::from_iter([compacted])
    );
}

/// Stages `val` segments in two batches at different versions while a seeded
/// workload runs, merges them, runs more of it before the commit, and checks
/// every lookup against a scan. Returns how many fragments the merged index
/// covers.
async fn merge_under_workload(seed: u64) -> u64 {
    let dir = TempStrDir::default();
    let rows = 12 * ROWS;
    let dataset = val_table(dir.as_str(), 12).await;
    let workload = Workload {
        rng: Mutex::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1),
        next_id: Mutex::new(rows),
        values: Mutex::new(0),
    };

    let mut staged = stage_val(&dataset, (0..6).collect()).await;
    // No compaction yet, so the second batch is staged at a later version.
    let before = workload.draw(3, false);
    let dataset = Box::pin(workload.run(dataset, &before)).await;
    let later = fragment_ids(&dataset)
        .into_iter()
        .filter(|id| (6..12).contains(id))
        .collect::<Vec<_>>();
    staged.extend(stage_val(&dataset, later).await);
    let between = workload.draw(3, true);
    let dataset = Box::pin(workload.run(dataset, &between)).await;
    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    let racing = workload.draw(workload.next(3) as usize, true);
    let mut dataset = Box::pin(workload.run(dataset, &racing)).await;
    dataset
        .commit_existing_index_segments("val_idx", "val", vec![merged])
        .await
        .unwrap();
    dataset.checkout_latest().await.unwrap();

    let trace = format!("seed {seed} {before:?} {between:?} {racing:?}");
    let coverage = val_idx_coverage(&dataset).await;
    let live = fragment_ids(&dataset)
        .into_iter()
        .collect::<RoaringBitmap>();
    assert!(
        coverage.is_subset(&live),
        "{trace}: {coverage:?} is not a subset of {live:?}"
    );
    let next_id = *workload.next_id.lock().unwrap();
    let values = *workload.values.lock().unwrap();
    let probes = (0..next_id)
        .map(|id| id * 10)
        .chain((1..=values).map(|n| 1_000_000 + n));
    assert_val_lookups_match_scan(&dataset, probes, &trace).await;
    coverage.len()
}

/// Segments staged in two batches, merged and committed while other writes,
/// overlays and deferred compactions run before, between and after: every
/// lookup through the merged index matches a scan, and fragments the changes
/// did not reach keep their coverage.
#[tokio::test]
async fn test_merged_index_matches_a_scan_under_a_workload() {
    let mut covered = 0;
    for seed in 0..24 {
        covered += Box::pin(merge_under_workload(seed)).await;
    }
    assert!(covered >= 16, "only {covered} fragments stayed covered");
}

/// The fragment that holds the row with `id`.
async fn fragment_of(dataset: &Dataset, id: i32) -> u32 {
    let mut scan = dataset.scan();
    scan.filter(&format!("id = {id}"))
        .unwrap()
        .with_row_address()
        .project(&["id"])
        .unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    (batch[lance_core::ROW_ADDR]
        .as_primitive::<arrow_array::types::UInt64Type>()
        .value(0)
        >> 32) as u32
}

async fn clean_up(dataset: &Dataset, versions: Vec<u64>) {
    let policy = crate::dataset::cleanup::CleanupPolicyBuilder::default()
        .versions(versions)
        .unwrap()
        .build();
    dataset.cleanup_with_policy(policy).await.unwrap();
}

/// Segments staged at two versions, one of which is cleaned up before the
/// merge: only the group staged at the lost version is left out, and the merge
/// and its commit go through.
#[tokio::test]
async fn test_btree_merge_leaves_out_only_a_group_whose_build_version_is_gone() {
    let dir = TempStrDir::default();
    let mut dataset = val_table(dir.as_str(), 4).await;
    let lost = dataset.manifest.version;
    let mut staged = stage_val(&dataset, vec![0, 1]).await;
    dataset.update_config([("next", "batch")]).await.unwrap();
    staged.extend(stage_val(&dataset, vec![2, 3]).await);
    let options = CompactionOptions {
        target_rows_per_fragment: (2 * ROWS) as usize,
        defer_index_remap: true,
        ..Default::default()
    };
    compact_files(&mut dataset, options, None).await.unwrap();
    clean_up(&dataset, vec![lost]).await;

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    dataset
        .commit_existing_index_segments("val_idx", "val", vec![merged])
        .await
        .unwrap();

    assert_val_lookups_match_scan(&dataset, (0..24).map(|id| id * 10), "").await;
    assert_eq!(
        val_idx_coverage(&dataset).await,
        RoaringBitmap::from_iter([fragment_of(&dataset, 12).await])
    );
}

/// Rewriting a column the index does not cover, even one stored in the same
/// file as the indexed column, keeps the coverage.
#[tokio::test]
async fn test_btree_merge_keeps_a_fragment_whose_other_column_was_rewritten() {
    let dir = TempStrDir::default();
    let mut dataset = val_table(dir.as_str(), 2).await;
    let staged = stage_val(&dataset, vec![0, 1]).await;
    let options = CompactionOptions {
        target_rows_per_fragment: (2 * ROWS) as usize,
        defer_index_remap: true,
        ..Default::default()
    };
    compact_files(&mut dataset, options, None).await.unwrap();
    let patch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, true),
            ArrowField::new("spare", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Int32Array::from(vec![7])),
        ],
    )
    .unwrap();
    let schema = patch.schema();
    let mut merge = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".into()]).unwrap();
    merge
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns);
    let (dataset, _) = merge
        .try_build()
        .unwrap()
        .execute_reader(RecordBatchIterator::new([Ok(patch)], schema))
        .await
        .unwrap();
    let mut dataset = Arc::unwrap_or_clone(dataset);

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    dataset
        .commit_existing_index_segments("val_idx", "val", vec![merged])
        .await
        .unwrap();

    assert_val_lookups_match_scan(&dataset, (0..12).map(|id| id * 10), "").await;
    assert_eq!(
        val_idx_coverage(&dataset).await,
        fragment_ids(&dataset)
            .into_iter()
            .collect::<RoaringBitmap>()
    );
}

/// An index on `s.x`, where `s` is a packed struct: rewriting `s` in place on
/// the compaction's output rewrites one physical column for the parent, and
/// that fragment is left out of the merged coverage.
#[tokio::test]
async fn test_btree_merge_leaves_out_a_rewritten_packed_parent() {
    use arrow_array::StructArray;
    use arrow_schema::Fields;
    use lance_encoding::constants::PACKED_STRUCT_META_KEY;

    let children = Fields::from(vec![ArrowField::new("x", DataType::Int32, false)]);
    let mut packed = ArrowField::new("s", DataType::Struct(children.clone()), false);
    packed.set_metadata([(PACKED_STRUCT_META_KEY.to_string(), "true".to_string())].into());
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("spare", DataType::Int32, false),
        packed.clone(),
    ]));
    let ids = Arc::new(Int32Array::from_iter_values(0..12)) as ArrayRef;
    let xs = Arc::new(StructArray::new(children.clone(), vec![ids.clone()], None)) as ArrayRef;
    let batch = RecordBatch::try_new(schema.clone(), vec![ids.clone(), ids, xs]).unwrap();
    let dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: ROWS as usize,
            data_storage_version: Some(lance_file::version::LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let staged = crate::utils::test::stage_index_segments(
        &mut dataset.clone(),
        "s.x",
        IndexType::BTree,
        &ScalarIndexParams::default(),
        "x_idx",
        vec![0, 1],
    )
    .await;
    let options = CompactionOptions {
        target_rows_per_fragment: (2 * ROWS) as usize,
        defer_index_remap: true,
        ..Default::default()
    };
    compact_files(&mut dataset, options, None).await.unwrap();
    let patch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            packed,
        ])),
        vec![
            Arc::new(Int32Array::from(vec![3])) as ArrayRef,
            Arc::new(StructArray::new(
                children,
                vec![Arc::new(Int32Array::from(vec![333])) as ArrayRef],
                None,
            )) as ArrayRef,
        ],
    )
    .unwrap();
    MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["id".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .execute_batches(vec![patch])
        .await
        .unwrap();
    dataset.checkout_latest().await.unwrap();

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    dataset
        .commit_existing_index_segments("x_idx", "s.x", vec![merged])
        .await
        .unwrap();

    for (filter, expected) in [("s.x = 333", 1), ("s.x = 3", 0)] {
        let mut scan = dataset.scan();
        scan.filter(filter).unwrap().use_scalar_index(false);
        assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), expected);
        assert_eq!(
            dataset.count_rows(Some(filter.into())).await.unwrap(),
            expected,
            "{filter} through the index"
        );
    }
}

/// RTree segments whose build version was cleaned up merge into an index that
/// covers nothing rather than failing the merge.
#[cfg(feature = "geo")]
#[tokio::test]
async fn test_rtree_merge_covers_nothing_once_its_build_version_is_gone() {
    let dir = TempStrDir::default();
    let (mut dataset, params) =
        geo::dataset_with_committed_rtree_index(dir.as_str(), RTREE_ROWS_PER_FRAGMENT, 2).await;
    let built = dataset.manifest.version;
    let sources = fragment_ids(&dataset);
    let staged = geo::stage_rtree_segments(&mut dataset, &params, sources).await;
    rtree_compact(&mut dataset, 2).await;
    clean_up(&dataset, vec![built]).await;

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    assert!(merged.fragment_bitmap.as_ref().unwrap().is_empty());
}
