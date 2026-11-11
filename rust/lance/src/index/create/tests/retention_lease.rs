// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index builds under a retention lease: the managed single-shot build and
//! the distributed flow that publishes staged segments for a lease.

use std::time::Duration;

use mock_instant::thread_local::MockClock;

use super::*;
use crate::dataset::cleanup::CleanupPolicyBuilder;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::{LeaseOptions, RetentionLease};
use crate::utils::temporal::utc_now;

const START: Duration = Duration::from_secs(1_000_000);

async fn four_fragments(uri: &str) -> Dataset {
    MockClock::set_system_time(START);
    gen_batch()
        .col("value", lance_datagen::array::step::<Int32Type>())
        .into_dataset(uri, FragmentCount::from(4), FragmentRowCount::from(16))
        .await
        .unwrap()
}

async fn listed_leases(dataset: &Dataset) -> Vec<RetentionLease> {
    let mut latest = dataset.clone();
    latest.checkout_latest().await.unwrap();
    latest.retention_leases().to_vec()
}

fn btree_params() -> ScalarIndexParams {
    ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree)
}

/// Compact everything into one fragment and remove every version cleanup
/// may, as a concurrent maintenance job would while an index builds.
async fn compact_and_clean(dataset: &Dataset) {
    let mut maintained = dataset.clone();
    maintained.checkout_latest().await.unwrap();
    compact_files(&mut maintained, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert_eq!(maintained.get_fragments().len(), 1);
    MockClock::set_system_time(START + Duration::from_secs(60));
    let policy = CleanupPolicyBuilder::default()
        .before_timestamp(utc_now())
        .delete_unverified(true)
        .build();
    maintained.cleanup_with_policy(policy).await.unwrap();
}

async fn assert_point_query_finds_one_row(dataset: &Dataset, index_name: &str) {
    let logical = crate::index::scalar_logical::open_named_scalar_index(
        dataset,
        "value",
        index_name,
        &NoOpMetricsCollector,
    )
    .await
    .unwrap();
    let query = SargableQuery::Equals(ScalarValue::Int32(Some(33)));
    match logical.search(&query, &NoOpMetricsCollector).await.unwrap() {
        SearchResult::Exact(row_addrs) => {
            assert_eq!(row_addrs.true_rows().row_addrs().unwrap().count(), 1)
        }
        other => panic!("expected an exact result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_leased_build_publishes_on_the_leased_version_and_ends_the_lease() {
    let test_dir = TempStrDir::default();
    let mut dataset = four_fragments(test_dir.as_str()).await;
    let params = btree_params();

    let index = dataset
        .create_index_builder(&["value"], IndexType::BTree, &params)
        .name("value_idx".to_string())
        .retention_lease(LeaseOptions::default())
        .await
        .unwrap();

    // Version 2 is the lease acquisition; the index publishes at version 3.
    assert_eq!(index.dataset_version, 2);
    assert_eq!(dataset.version().version, 3);
    assert_eq!(listed_leases(&dataset).await, vec![]);
    assert_point_query_finds_one_row(&dataset, "value_idx").await;
}

#[tokio::test]
async fn test_failed_leased_build_releases_the_lease() {
    let test_dir = TempStrDir::default();
    let mut dataset = four_fragments(test_dir.as_str()).await;
    let params = btree_params();

    let result = dataset
        .create_index_builder(&["missing"], IndexType::BTree, &params)
        .retention_lease(LeaseOptions::default())
        .await;

    assert!(result.is_err());
    assert_eq!(listed_leases(&dataset).await, vec![]);
    // The caller's handle is left where it was.
    assert_eq!(dataset.version().version, 1);
}

#[tokio::test]
async fn test_uncommitted_build_refuses_a_retention_lease() {
    let test_dir = TempStrDir::default();
    let mut dataset = four_fragments(test_dir.as_str()).await;
    let params = btree_params();

    let error = CreateIndexBuilder::new(&mut dataset, &["value"], IndexType::BTree, &params)
        .retention_lease(LeaseOptions::default())
        .execute_uncommitted()
        .await
        .unwrap_err();

    assert!(error.to_string().contains("retention_lease"), "{error}");
    assert_eq!(listed_leases(&dataset).await, vec![]);
}

/// Stage one B-tree segment per fragment of `leased`, as distributed workers
/// would.
async fn stage_segments(leased: &mut Dataset) -> Vec<IndexMetadata> {
    let params = btree_params();
    let mut segments = Vec::new();
    for fragment in leased.get_fragments() {
        segments.push(
            CreateIndexBuilder::new(leased, &["value"], IndexType::BTree, &params)
                .name("value_idx".to_string())
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    segments
}

#[tokio::test]
async fn test_leased_segments_publish_after_compaction_and_cleanup() {
    let test_dir = TempStrDir::default();
    let dataset = four_fragments(test_dir.as_str()).await;
    let (lease, mut leased) = dataset
        .acquire_retention_lease(LeaseOptions::default())
        .await
        .unwrap();
    let segments = stage_segments(&mut leased).await;

    compact_and_clean(&dataset).await;
    // The coordinator may hold any version; publication reads the leased one.
    let mut coordinator = dataset.clone();
    coordinator.checkout_latest().await.unwrap();
    coordinator
        .commit_existing_index_segments_for_lease("value_idx", "value", segments, &lease)
        .await
        .unwrap();

    assert_eq!(listed_leases(&coordinator).await, vec![]);
    assert_eq!(coordinator.get_fragments().len(), 1);
    assert_point_query_finds_one_row(&coordinator, "value_idx").await;
}

#[tokio::test]
async fn test_leased_segments_with_missing_files_are_refused() {
    let test_dir = TempStrDir::default();
    let dataset = four_fragments(test_dir.as_str()).await;
    let (lease, mut leased) = dataset
        .acquire_retention_lease(LeaseOptions::default())
        .await
        .unwrap();
    let segments = stage_segments(&mut leased).await;
    let removed = leased.indices_dir().join(segments[0].uuid.to_string());
    leased.object_store.remove_dir_all(removed).await.unwrap();

    let mut coordinator = dataset.clone();
    let error = coordinator
        .commit_existing_index_segments_for_lease("value_idx", "value", segments, &lease)
        .await
        .unwrap_err();

    let message = error.to_string();
    assert!(message.contains("has no files"), "{message}");
    assert_eq!(listed_leases(&dataset).await, vec![lease]);
    // A failed commit leaves the caller's handle where it was.
    assert_eq!(coordinator.version().version, 1);
}
