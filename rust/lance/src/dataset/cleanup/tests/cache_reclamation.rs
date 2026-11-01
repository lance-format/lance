// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

/// Regression test for https://github.com/lance-format/lance/issues/1983.
///
/// Every commit inserts new version-scoped entries into the session's
/// metadata/index caches (manifests, transactions, row id indexes, and
/// index metadata are all keyed by version). Nothing evicted them for
/// versions cleanup had already made unreachable on disk, so a
/// long-running writer sharing one `Session` across many commits saw
/// these caches grow roughly linearly with commit count -- looking like a
/// memory leak even though every entry was legitimately referenced.
///
/// This test writes many versions under one shared `Session`, including a
/// real vector index build and a later `optimize_indices()` call so the
/// index cache (not just the metadata cache) accumulates version-keyed
/// entries the way production code populates them, runs cleanup to retain
/// only the latest version, and asserts that both caches shrink because
/// the stale versions' entries were invalidated.
#[tokio::test]
async fn cleanup_evicts_stale_version_caches() {
    use crate::session::Session;
    use lance_index::optimize::OptimizeOptions;

    let tmpdir = TempStrDir::default();
    let dataset_path = format!("{}/my_db", tmpdir.as_str());
    // Large capacity so nothing is evicted by the normal capacity-based
    // policy during this test -- any shrinkage we observe must come from
    // cleanup's explicit invalidation, not incidental LRU eviction.
    let session = Arc::new(Session::new(
        1 << 30,
        1 << 30,
        Arc::new(ObjectStoreRegistry::default()),
    ));

    // v1: create the dataset.
    Dataset::write(
        some_batch(),
        &dataset_path,
        Some(WriteParams {
            session: Some(session.clone()),
            mode: WriteMode::Create,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // v2: build a vector index -- a real, separate commit whose
    // transaction carries non-empty index metadata, inserting a genuine
    // `IndexMetadataKey` entry into the index cache.
    let mut dataset = DatasetBuilder::from_uri(&dataset_path)
        .with_session(session.clone())
        .load()
        .await
        .unwrap();
    let index_params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 5);
    dataset
        .create_index(
            &["indexable"],
            IndexType::Vector,
            Some("some_index".to_owned()),
            &index_params,
            false,
        )
        .await
        .unwrap();

    // v3-v5: appends. These don't touch the index, so they only add
    // metadata-cache entries (matching most of production traffic).
    const NUM_APPENDS_BEFORE_OPTIMIZE: usize = 3;
    for _ in 0..NUM_APPENDS_BEFORE_OPTIMIZE {
        Dataset::write(
            some_batch(),
            &dataset_path,
            Some(WriteParams {
                session: Some(session.clone()),
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    }

    // v6: fold the newly-appended, unindexed fragments into the index.
    // Another real commit with non-empty index metadata -- a second,
    // distinct `IndexMetadataKey` entry, at a version that will itself
    // become stale once more commits follow.
    let mut dataset = DatasetBuilder::from_uri(&dataset_path)
        .with_session(session.clone())
        .load()
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();

    // v7-v11: more appends, so v1-v6 are all old by the time cleanup runs.
    const NUM_APPENDS_AFTER_OPTIMIZE: usize = 5;
    for _ in 0..NUM_APPENDS_AFTER_OPTIMIZE {
        Dataset::write(
            some_batch(),
            &dataset_path,
            Some(WriteParams {
                session: Some(session.clone()),
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    }

    let metadata_before = session.metadata_cache_stats().await;
    let index_before = session.index_cache_stats().await;
    let expected_rows = {
        let dataset = DatasetBuilder::from_uri(&dataset_path)
            .with_session(session.clone())
            .load()
            .await
            .unwrap();
        dataset.count_rows(None).await.unwrap()
    };

    let dataset = DatasetBuilder::from_uri(&dataset_path)
        .with_session(session.clone())
        .load()
        .await
        .unwrap();
    cleanup_old_versions(
        &dataset,
        CleanupPolicy {
            before_version: Some(11),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let metadata_after = session.metadata_cache_stats().await;
    let index_after = session.index_cache_stats().await;
    assert_lt!(metadata_after.num_entries, metadata_before.num_entries);
    assert_lt!(index_after.num_entries, index_before.num_entries);

    // The surviving dataset must still be fully readable after the
    // invalidation: evicting stale versions must not touch live entries.
    let dataset = DatasetBuilder::from_uri(&dataset_path)
        .with_session(session.clone())
        .load()
        .await
        .unwrap();
    assert_eq!(dataset.version().version, 11);
    assert_eq!(dataset.count_rows(None).await.unwrap(), expected_rows);
}
