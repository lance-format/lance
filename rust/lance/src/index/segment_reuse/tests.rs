// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::ROW_ID;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_io::utils::tracking_store::IoStats;
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::{
    CommittedSegmentSource, ReusedSegment, STALE_REUSE_SOURCE_ERROR,
    UNSUPPORTED_REUSE_SOURCE_ERROR, merge_committed_index_segments_with_budget,
};
use crate::dataset::WriteParams;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::cleanup::CleanupPolicyBuilder;
use crate::index::{CreateIndexBuilder, DatasetIndexExt};
use crate::{Dataset, Error};

const INDEX_NAME: &str = "x_zm";
const ROWS_PER_FRAGMENT: i32 = 1000;
const BASE_FRAGMENTS: i32 = 4;
const ROWS_PER_ZONE: u64 = 100;
const QUERY_FILTERS: [&str; 2] = ["x >= 4500", "x < 10"];

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "x",
        DataType::Int32,
        false,
    )]))
}

fn rows(start: i32, count: i32) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int32Array::from_iter_values(start..start + count))],
    )
    .unwrap()
}

fn zonemap_params() -> ScalarIndexParams {
    let params = lance_index::scalar::zonemap::ZoneMapIndexBuilderParams::new(ROWS_PER_ZONE);
    let params_json = serde_json::to_value(&params).unwrap();
    ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap).with_params(&params_json)
}

async fn write_base(uri: &str) -> Dataset {
    let batch = rows(0, BASE_FRAGMENTS * ROWS_PER_FRAGMENT);
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema());
    Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FRAGMENT as usize,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

async fn append_fragment(dataset: &mut Dataset) {
    let start = dataset.count_rows(None).await.unwrap() as i32;
    let reader = RecordBatchIterator::new(vec![Ok(rows(start, ROWS_PER_FRAGMENT))], schema());
    dataset.append(reader, None).await.unwrap();
}

/// Delete every manifest but the latest, then reopen in a fresh session so no
/// cache from earlier steps hides reads.
async fn cleanup_and_reopen(dataset: Dataset, uri: &str) -> Dataset {
    let policy = CleanupPolicyBuilder::default()
        .retain_n_versions(&dataset, 1)
        .await
        .unwrap()
        .build();
    dataset.cleanup(policy).execute().await.unwrap();
    let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
    let versions = dataset.versions().await.unwrap();
    assert_eq!(versions.len(), 1, "only the latest manifest remains");
    dataset
}

async fn build_uncommitted(dataset: &mut Dataset, fragments: Vec<u32>) -> IndexMetadata {
    let params = zonemap_params();
    CreateIndexBuilder::new(dataset, &["x"], IndexType::ZoneMap, &params)
        .name(INDEX_NAME.to_string())
        // A further segment of an existing index is built under its name.
        .replace(true)
        .fragments(fragments)
        .execute_uncommitted()
        .await
        .unwrap()
}

/// One committed segment per fragment group, each committed separately so
/// their build versions differ; then append fragment 4, clean up to the
/// latest manifest and reopen.
async fn segments_then_append(uri: &str, groups: &[Vec<u32>]) -> Dataset {
    let mut dataset = write_base(uri).await;
    for fragments in groups {
        let segment = build_uncommitted(&mut dataset, fragments.clone()).await;
        dataset
            .commit_existing_index_segments(INDEX_NAME, "x", vec![segment])
            .await
            .unwrap();
    }
    append_fragment(&mut dataset).await;
    cleanup_and_reopen(dataset, uri).await
}

/// Live segments of the index, ordered by their first covered fragment.
async fn segments(dataset: &Dataset) -> Vec<IndexMetadata> {
    let mut segments = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
    segments.sort_by_key(|segment| segment.fragment_bitmap.as_ref().and_then(|b| b.min()));
    segments
}

/// The manifest's index segments as (uuid, coverage), read from the latest
/// version at `uri`.
async fn manifest_segments(uri: &str) -> Vec<(Uuid, Vec<u32>)> {
    let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
    segments(&dataset)
        .await
        .iter()
        .map(|segment| {
            (
                segment.uuid,
                segment
                    .fragment_bitmap
                    .as_ref()
                    .map(|bitmap| bitmap.iter().collect())
                    .unwrap_or_default(),
            )
        })
        .collect()
}

fn source_of(dataset: &Dataset, segment: &IndexMetadata) -> CommittedSegmentSource {
    CommittedSegmentSource {
        uuid: segment.uuid,
        expected_fragments: segment
            .effective_fragment_bitmap(dataset.fragment_bitmap.as_ref())
            .unwrap(),
        expected_dataset_version: segment.dataset_version,
    }
}

fn bitmap(fragments: impl IntoIterator<Item = u32>) -> RoaringBitmap {
    RoaringBitmap::from_iter(fragments)
}

fn data_file_names(dataset: &Dataset, fragment_ids: impl Fn(u64) -> bool) -> HashSet<String> {
    dataset
        .manifest
        .fragments
        .iter()
        .filter(|fragment| fragment_ids(fragment.id))
        .flat_map(|fragment| fragment.files.iter().map(|file| file.path.clone()))
        .collect()
}

/// Data files of the base fragments and of every other fragment, captured
/// from the manifest the maintenance call starts from.
struct FileClasses {
    old_data: HashSet<String>,
    new_data: HashSet<String>,
}

impl FileClasses {
    fn capture(dataset: &Dataset) -> Self {
        let base = |id: u64| id < BASE_FRAGMENTS as u64;
        Self {
            old_data: data_file_names(dataset, base),
            new_data: data_file_names(dataset, |id| !base(id)),
        }
    }

    /// Paths of the read requests against data files of the given class.
    fn data_reads(&self, io: &IoStats, include_new: bool) -> Vec<String> {
        io.requests
            .iter()
            .filter(|request| request.method.starts_with("get"))
            .map(|request| request.path.to_string())
            .filter(|path| {
                if !path.contains("data/") || path.contains("_indices/") {
                    return false;
                }
                let name = path.rsplit('/').next().unwrap_or(path);
                self.old_data.iter().any(|file| file.ends_with(name))
                    || (include_new && self.new_data.iter().any(|file| file.ends_with(name)))
            })
            .collect()
    }
}

async fn matching_rows(dataset: &Dataset, filter: &str, use_index: bool) -> Vec<(u64, i32)> {
    let mut scan = dataset.scan();
    scan.project(&["x"]).unwrap();
    scan.with_row_id();
    scan.filter(filter).unwrap();
    scan.use_scalar_index(use_index);
    let batch = scan.try_into_batch().await.unwrap();
    let row_ids = batch
        .column_by_name(ROW_ID)
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let values = batch
        .column_by_name("x")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let mut rows = row_ids
        .values()
        .iter()
        .copied()
        .zip(values.values().iter().copied())
        .collect::<Vec<_>>();
    rows.sort_unstable();
    rows
}

/// Indexed answers equal a scan row by row, and the index is used.
async fn assert_answers_match_scan(dataset: &Dataset) {
    for filter in QUERY_FILTERS {
        let indexed = matching_rows(dataset, filter, true).await;
        let scanned = matching_rows(dataset, filter, false).await;
        assert!(!scanned.is_empty(), "filter {filter}");
        assert_eq!(indexed, scanned, "filter {filter}");

        let mut scan = dataset.scan();
        scan.filter(filter).unwrap();
        let plan = scan.explain_plan(false).await.unwrap();
        assert!(plan.contains("ScalarIndexQuery"), "filter {filter}: {plan}");
    }
}

fn assert_error_prefix(error: &Error, prefix: &str) {
    assert!(error.to_string().contains(prefix), "{error}");
}

#[tokio::test]
async fn test_reuse_committed_segments_with_fresh_segment() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = segments_then_append(uri, &[vec![0, 1], vec![2, 3]]).await;
    let committed = segments(&dataset).await;
    assert_eq!(committed.len(), 2);
    let sources = committed
        .iter()
        .map(|segment| source_of(&dataset, segment))
        .collect::<Vec<_>>();
    let fresh = build_uncommitted(&mut dataset, vec![4]).await;
    let fresh_read_version = fresh.dataset_version;

    let classes = FileClasses::capture(&dataset);
    dataset.object_store.io_stats_incremental();
    let merged = dataset
        .merge_committed_index_segments(&sources, vec![fresh])
        .await
        .unwrap();
    assert_eq!(merged.fragment_bitmap, Some(bitmap(0..5)));
    let merged_uuid = merged.uuid;
    dataset
        .commit_reused_index_segments(
            INDEX_NAME,
            "x",
            vec![ReusedSegment {
                segment: merged,
                sources,
                fresh_fragments: bitmap([4]),
                fresh_read_version: Some(fresh_read_version),
            }],
        )
        .await
        .unwrap();
    let io = dataset.object_store.io_stats_incremental();
    let old_reads = classes.data_reads(&io, false);
    assert!(
        old_reads.is_empty(),
        "old fragment data was read: {old_reads:?}"
    );

    assert_eq!(
        manifest_segments(uri).await,
        vec![(merged_uuid, (0..5).collect())]
    );
    let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
    assert_answers_match_scan(&dataset).await;
}

#[tokio::test]
async fn test_reuse_consolidates_committed_segments_without_data_reads() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = segments_then_append(uri, &[vec![0, 1], vec![2, 3]]).await;
    let sources = segments(&dataset)
        .await
        .iter()
        .map(|segment| source_of(&dataset, segment))
        .collect::<Vec<_>>();

    let classes = FileClasses::capture(&dataset);
    dataset.object_store.io_stats_incremental();
    let merged = dataset
        .merge_committed_index_segments(&sources, vec![])
        .await
        .unwrap();
    let merged_uuid = merged.uuid;
    dataset
        .commit_reused_index_segments(
            INDEX_NAME,
            "x",
            vec![ReusedSegment {
                segment: merged,
                sources,
                fresh_fragments: RoaringBitmap::new(),
                fresh_read_version: None,
            }],
        )
        .await
        .unwrap();
    let io = dataset.object_store.io_stats_incremental();
    let data_reads = classes.data_reads(&io, true);
    assert!(
        data_reads.is_empty(),
        "fragment data was read: {data_reads:?}"
    );

    assert_eq!(
        manifest_segments(uri).await,
        vec![(merged_uuid, (0..4).collect())]
    );
    let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
    assert!(
        dataset.fragment_bitmap.contains(4),
        "fragment 4 stays uncovered"
    );
    assert_answers_match_scan(&dataset).await;
}

#[tokio::test]
async fn test_reuse_consolidates_a_subset_of_segments() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = segments_then_append(uri, &[vec![0], vec![1], vec![2, 3]]).await;
    let committed = segments(&dataset).await;
    assert_eq!(committed.len(), 3);
    let kept = committed[2].uuid;
    let sources = committed[..2]
        .iter()
        .map(|segment| source_of(&dataset, segment))
        .collect::<Vec<_>>();

    let merged = dataset
        .merge_committed_index_segments(&sources, vec![])
        .await
        .unwrap();
    let merged_uuid = merged.uuid;
    dataset
        .commit_reused_index_segments(
            INDEX_NAME,
            "x",
            vec![ReusedSegment {
                segment: merged,
                sources,
                fresh_fragments: RoaringBitmap::new(),
                fresh_read_version: None,
            }],
        )
        .await
        .unwrap();

    assert_eq!(
        manifest_segments(uri).await,
        vec![(merged_uuid, vec![0, 1]), (kept, vec![2, 3])]
    );
}

#[tokio::test]
async fn test_reuse_output_keeps_oldest_source_version() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let dataset = segments_then_append(uri, &[vec![0, 1], vec![2, 3]]).await;
    let committed = segments(&dataset).await;
    let oldest = committed
        .iter()
        .map(|segment| segment.dataset_version)
        .min()
        .unwrap();
    let newest = committed
        .iter()
        .map(|segment| segment.dataset_version)
        .max()
        .unwrap();
    assert!(
        oldest < newest,
        "the sources were built at different versions"
    );
    let sources = committed
        .iter()
        .map(|segment| source_of(&dataset, segment))
        .collect::<Vec<_>>();

    let merged = dataset
        .merge_committed_index_segments(&sources, vec![])
        .await
        .unwrap();
    assert_eq!(merged.dataset_version, oldest);
}

/// Merges S1 {0, 1} and S2 {2, 3} into one output.
async fn merged_two_segments(dataset: &Dataset) -> ReusedSegment {
    let sources = segments(dataset)
        .await
        .iter()
        .map(|segment| source_of(dataset, segment))
        .collect::<Vec<_>>();
    let segment = dataset
        .merge_committed_index_segments(&sources, vec![])
        .await
        .unwrap();
    ReusedSegment {
        segment,
        sources,
        fresh_fragments: RoaringBitmap::new(),
        fresh_read_version: None,
    }
}

#[tokio::test]
async fn test_reuse_commit_refuses_output_missing_source_fragment() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = segments_then_append(uri, &[vec![0, 1], vec![2, 3]]).await;
    let mut output = merged_two_segments(&dataset).await;
    output.segment.fragment_bitmap = Some(bitmap([0, 1, 2]));
    let before = manifest_segments(uri).await;

    let error = dataset
        .commit_reused_index_segments(INDEX_NAME, "x", vec![output])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert_eq!(manifest_segments(uri).await, before);
}

#[tokio::test]
async fn test_reuse_commit_refuses_replaced_source() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = segments_then_append(uri, &[vec![0, 1], vec![2, 3]]).await;
    let output = merged_two_segments(&dataset).await;
    // Rebuild S1 over the same fragments under a new UUID.
    let rebuilt = build_uncommitted(&mut dataset, vec![0, 1]).await;
    dataset
        .commit_existing_index_segments(INDEX_NAME, "x", vec![rebuilt])
        .await
        .unwrap();
    let before = manifest_segments(uri).await;
    assert!(
        before
            .iter()
            .all(|(uuid, _)| *uuid != output.sources[0].uuid)
    );

    let error = dataset
        .commit_reused_index_segments(INDEX_NAME, "x", vec![output])
        .await
        .unwrap_err();
    assert_error_prefix(&error, STALE_REUSE_SOURCE_ERROR);
    assert_eq!(manifest_segments(uri).await, before);
}

#[tokio::test]
async fn test_reuse_commit_refuses_changed_source_coverage() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = segments_then_append(uri, &[vec![0, 1], vec![2, 3]]).await;
    let output = merged_two_segments(&dataset).await;
    // Deleting every row of fragment 1 removes it from S1's effective coverage.
    dataset.delete("x >= 1000 AND x < 2000").await.unwrap();
    assert!(!dataset.fragment_bitmap.contains(1));
    let before = manifest_segments(uri).await;

    let error = dataset
        .commit_reused_index_segments(INDEX_NAME, "x", vec![output])
        .await
        .unwrap_err();
    assert_error_prefix(&error, STALE_REUSE_SOURCE_ERROR);
    assert_eq!(manifest_segments(uri).await, before);
}

#[tokio::test]
async fn test_reuse_commit_refuses_undeclared_overlapping_segment() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = segments_then_append(uri, &[vec![0, 1], vec![2, 3]]).await;
    let sources = segments(&dataset)
        .await
        .iter()
        .map(|segment| source_of(&dataset, segment))
        .collect::<Vec<_>>();
    let fresh = build_uncommitted(&mut dataset, vec![4]).await;
    let fresh_read_version = fresh.dataset_version;
    let merged = dataset
        .merge_committed_index_segments(&sources, vec![fresh])
        .await
        .unwrap();
    // Another writer indexes fragment 4 in the meantime.
    let other = build_uncommitted(&mut dataset, vec![4]).await;
    dataset
        .commit_existing_index_segments(INDEX_NAME, "x", vec![other])
        .await
        .unwrap();
    let before = manifest_segments(uri).await;
    assert_eq!(before.len(), 3);

    let error = dataset
        .commit_reused_index_segments(
            INDEX_NAME,
            "x",
            vec![ReusedSegment {
                segment: merged,
                sources,
                fresh_fragments: bitmap([4]),
                fresh_read_version: Some(fresh_read_version),
            }],
        )
        .await
        .unwrap_err();
    assert_error_prefix(&error, STALE_REUSE_SOURCE_ERROR);
    assert_eq!(manifest_segments(uri).await, before);
}

#[tokio::test]
async fn test_reuse_commit_refuses_cleaned_fresh_read_version() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = segments_then_append(uri, &[vec![0, 1], vec![2, 3]]).await;
    let sources = segments(&dataset)
        .await
        .iter()
        .map(|segment| source_of(&dataset, segment))
        .collect::<Vec<_>>();
    let fresh = build_uncommitted(&mut dataset, vec![4]).await;
    let merged = dataset
        .merge_committed_index_segments(&sources, vec![fresh])
        .await
        .unwrap();
    // Version 1 was removed by the cleanup in the setup.
    let cleaned_version = 1;
    assert!(dataset.checkout_version(cleaned_version).await.is_err());
    let before = manifest_segments(uri).await;

    let error = dataset
        .commit_reused_index_segments(
            INDEX_NAME,
            "x",
            vec![ReusedSegment {
                segment: merged,
                sources,
                fresh_fragments: bitmap([4]),
                fresh_read_version: Some(cleaned_version),
            }],
        )
        .await
        .unwrap_err();
    assert_error_prefix(&error, STALE_REUSE_SOURCE_ERROR);
    assert_eq!(manifest_segments(uri).await, before);
    assert_eq!(
        before
            .iter()
            .map(|(_, fragments)| fragments.clone())
            .collect::<Vec<_>>(),
        vec![vec![0, 1], vec![2, 3]]
    );
}

#[tokio::test]
async fn test_reuse_merge_refuses_over_budget() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let dataset = segments_then_append(uri, &[vec![0, 1], vec![2, 3]]).await;
    let sources = segments(&dataset)
        .await
        .iter()
        .map(|segment| source_of(&dataset, segment))
        .collect::<Vec<_>>();
    let before = manifest_segments(uri).await;

    let error = merge_committed_index_segments_with_budget(&dataset, &sources, vec![], 1)
        .await
        .unwrap_err();
    assert_error_prefix(&error, UNSUPPORTED_REUSE_SOURCE_ERROR);
    assert_eq!(manifest_segments(uri).await, before);
}

/// The admission check compares against the metadata estimate itself: a
/// budget of exactly the estimate is admitted, one byte less is refused.
#[tokio::test]
async fn test_reuse_merge_admits_exactly_its_estimate() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let dataset = segments_then_append(uri, &[vec![0, 1], vec![2, 3]]).await;
    let committed = segments(&dataset).await;
    let sources = committed
        .iter()
        .map(|segment| source_of(&dataset, segment))
        .collect::<Vec<_>>();
    let estimate = super::estimate_merge_bytes(&dataset, &committed)
        .await
        .unwrap();
    assert!(estimate > 0);
    assert!(estimate <= super::REUSE_MERGE_BUDGET_BYTES);

    let error =
        merge_committed_index_segments_with_budget(&dataset, &sources, vec![], estimate - 1)
            .await
            .unwrap_err();
    assert_error_prefix(&error, UNSUPPORTED_REUSE_SOURCE_ERROR);
    let merged = merge_committed_index_segments_with_budget(&dataset, &sources, vec![], estimate)
        .await
        .unwrap();
    assert_eq!(merged.fragment_bitmap, Some(bitmap(0..4)));
}

/// A source whose values the file metadata cannot bound in memory (strings)
/// is refused visibly instead of estimated, and nothing is published.
#[tokio::test]
async fn test_reuse_merge_refuses_source_without_safe_estimate() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        DataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(arrow_array::StringArray::from_iter_values(
            (0..4000).map(|row| format!("value-{row:05}")),
        ))],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FRAGMENT as usize,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let params = zonemap_params();
    for fragments in [vec![0, 1], vec![2, 3]] {
        let segment = CreateIndexBuilder::new(&mut dataset, &["s"], IndexType::ZoneMap, &params)
            .name("s_zm".to_string())
            .replace(true)
            .fragments(fragments)
            .execute_uncommitted()
            .await
            .unwrap();
        dataset
            .commit_existing_index_segments("s_zm", "s", vec![segment])
            .await
            .unwrap();
    }
    let committed = dataset.load_indices_by_name("s_zm").await.unwrap();
    assert_eq!(committed.len(), 2);
    let sources = committed
        .iter()
        .map(|segment| source_of(&dataset, segment))
        .collect::<Vec<_>>();
    let error = dataset
        .merge_committed_index_segments(&sources, vec![])
        .await
        .unwrap_err();
    assert_error_prefix(&error, UNSUPPORTED_REUSE_SOURCE_ERROR);
    assert!(
        error.to_string().contains("no safe memory estimate"),
        "{error}"
    );
    assert_eq!(dataset.load_indices_by_name("s_zm").await.unwrap().len(), 2);
}

/// A compaction that rewrites a declared source's fragments after the merge
/// and before the commit makes the publication fail, whether the commit is
/// built on the manifest the merge saw (conflict) or on the latest one
/// (stale source). The reused output is never published.
#[rstest::rstest]
#[tokio::test]
async fn test_reuse_commit_fails_after_concurrent_compaction(
    #[values(false, true)] defer_index_remap: bool,
    #[values(false, true)] reload_before_commit: bool,
) {
    use crate::dataset::optimize::{CompactionOptions, compact_files};

    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let dataset = segments_then_append(uri, &[vec![0, 1], vec![2, 3]]).await;
    let output = merged_two_segments(&dataset).await;
    let merged_uuid = output.segment.uuid;
    let merged_at = dataset.manifest.version;

    let mut compactor = DatasetBuilder::from_uri(uri).load().await.unwrap();
    compact_files(
        &mut compactor,
        CompactionOptions {
            defer_index_remap,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert!(compactor.manifest.version > merged_at);
    assert!(
        compactor
            .manifest
            .fragments
            .iter()
            .all(|fragment| fragment.id >= BASE_FRAGMENTS as u64),
        "compaction must rewrite the source fragments: {:?}",
        compactor
            .manifest
            .fragments
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>()
    );
    let after_compaction = manifest_segments(uri).await;

    let mut committer = if reload_before_commit {
        DatasetBuilder::from_uri(uri).load().await.unwrap()
    } else {
        dataset
    };
    let error = committer
        .commit_reused_index_segments(INDEX_NAME, "x", vec![output])
        .await
        .expect_err("the reused output was published");
    if reload_before_commit {
        assert_error_prefix(&error, STALE_REUSE_SOURCE_ERROR);
    } else {
        assert!(
            matches!(error, Error::RetryableCommitConflict { .. }),
            "{error}"
        );
    }

    let published = manifest_segments(uri).await;
    assert!(published.iter().all(|(uuid, _)| *uuid != merged_uuid));
    assert_eq!(published, after_compaction);
    let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
    for filter in QUERY_FILTERS {
        assert_eq!(
            matching_rows(&dataset, filter, true).await,
            matching_rows(&dataset, filter, false).await,
            "filter {filter}"
        );
    }
}

/// A source declared with a different stamp than the manifest reports is
/// stale, and an output stamped newer than its oldest source is refused.
#[tokio::test]
async fn test_reuse_commit_checks_source_and_output_stamps() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = segments_then_append(uri, &[vec![0, 1], vec![2, 3]]).await;
    let before = manifest_segments(uri).await;

    let mut output = merged_two_segments(&dataset).await;
    output.sources[1].expected_dataset_version += 1;
    let error = dataset
        .commit_reused_index_segments(INDEX_NAME, "x", vec![output])
        .await
        .unwrap_err();
    assert_error_prefix(&error, STALE_REUSE_SOURCE_ERROR);
    assert_eq!(manifest_segments(uri).await, before);

    let mut output = merged_two_segments(&dataset).await;
    let oldest = output
        .sources
        .iter()
        .map(|source| source.expected_dataset_version)
        .min()
        .unwrap();
    assert_eq!(output.segment.dataset_version, oldest);
    output.segment.dataset_version = dataset.manifest.version;
    assert!(output.segment.dataset_version > oldest);
    let error = dataset
        .commit_reused_index_segments(INDEX_NAME, "x", vec![output])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert_eq!(manifest_segments(uri).await, before);
}

/// An unrelated append between the merge and the commit also fails the
/// commit on the old manifest (it is validated against that manifest only);
/// after a reload the same output is revalidated and published, and the
/// appended fragment stays uncovered.
#[tokio::test]
async fn test_reuse_commit_after_unrelated_append_needs_reload() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = segments_then_append(uri, &[vec![0, 1], vec![2, 3]]).await;
    let output = merged_two_segments(&dataset).await;
    let merged_uuid = output.segment.uuid;

    let mut writer = DatasetBuilder::from_uri(uri).load().await.unwrap();
    append_fragment(&mut writer).await;
    let before = manifest_segments(uri).await;

    let error = dataset
        .commit_reused_index_segments(INDEX_NAME, "x", vec![output.clone()])
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
    assert_eq!(manifest_segments(uri).await, before);

    let mut dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
    dataset
        .commit_reused_index_segments(INDEX_NAME, "x", vec![output])
        .await
        .unwrap();
    assert_eq!(
        manifest_segments(uri).await,
        vec![(merged_uuid, (0..4).collect())]
    );
    let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
    assert_answers_match_scan(&dataset).await;
}
