// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use futures::TryStreamExt;
use std::time::Duration;

#[rstest]
#[tokio::test]
async fn prefetched_batches_preserve_snapshot_overlay_deletions_and_markers(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
) {
    let mut dataset = write_payload_dataset(version, true, true).await;
    let field = dataset.schema().field("payload").unwrap().id;
    for fragment in 0..2 {
        dataset = commit_overlay(
            dataset,
            &format!("prefetch-{fragment}"),
            fragment,
            &[field],
            OverlayCoverage::dense(bitmap([1, 3, 4])),
            vec![payload("new", 3, true, 0)],
            version,
        )
        .await;
    }
    dataset
        .delete("untouched IN ('other0', 'other1', 'other6', 'other7')")
        .await
        .unwrap();
    Arc::make_mut(&mut Arc::make_mut(&mut dataset.manifest).fragments).reverse();
    let oracle = dataset.clone();
    let config = FragReadConfig::default()
        .with_row_id(true)
        .with_row_address(true)
        .with_scan_scheduler(ScanScheduler::new(
            dataset.object_store.clone(),
            SchedulerConfig {
                io_buffer_size_bytes: 1024,
                use_lite_scheduler: Some(lite),
            },
        ));
    let budget = AllocationBudget::new(65536);
    let scan = dataset
        .try_scan_admitted_fullzip(
            dataset.schema().clone(),
            config.clone(),
            2,
            32768,
            budget.clone(),
        )
        .unwrap();
    let mut stream = scan.into_prefetched_batches();
    Arc::make_mut(&mut dataset.manifest).fragments = Arc::new(Vec::new());
    for id in [1, 0] {
        let reader = oracle
            .get_fragment(id)
            .unwrap()
            .open(oracle.schema(), config.clone())
            .await
            .unwrap();
        for start in [0, 2, 4] {
            let expected = reader
                .take_as_batch(&[start, start + 1], None)
                .await
                .unwrap();
            let actual = tokio::time::timeout(Duration::from_secs(5), stream.try_next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(actual, expected);
            if start == 0 {
                assert_eq!(actual.num_rows(), 0);
            }
            assert!(budget.reserved() <= 65536);
        }
    }
    assert!(stream.try_next().await.unwrap().is_none());
    assert!(stream.try_next().await.unwrap().is_none());
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn one_queued_batch_overlaps_consumer_and_drop_preserves_last_alias() {
    let dataset = null_dataset(10).await;
    let budget = AllocationBudget::new(128);
    let make_scan = || {
        dataset
            .try_scan_admitted_fullzip(
                dataset.schema().project(&["late"]).unwrap(),
                Default::default(),
                2,
                64,
                budget.clone(),
            )
            .unwrap()
    };
    let Outcome::Batch { batch, .. } = make_scan().try_next().await.unwrap() else {
        panic!("reference prefix fits");
    };
    let batch_charge = budget.reserved();
    assert!(batch_charge > 1 && batch_charge * 2 < 128);
    drop(batch);
    let mut stream = make_scan().into_prefetched_batches();
    let first = stream.try_next().await.unwrap().unwrap();
    assert_eq!(first.num_rows(), 2);
    // The consumer still holds its output. A second complete batch can be
    // prepared, but the single reserved delivery slot prevents a third attempt.
    tokio::time::timeout(Duration::from_secs(5), async {
        while budget.reserved() != 2 * batch_charge {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert_eq!(budget.reserved(), 2 * batch_charge);
    let alias = first["late"].to_data().buffers()[0].slice_with_length(0, 1);
    drop(first);
    drop(stream);
    tokio::time::timeout(
        Duration::from_secs(5),
        budget.wait_for_capacity(128 - batch_charge),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(budget.reserved() > 1 && budget.reserved() <= batch_charge);
    assert_eq!(alias.as_slice(), &[0]);
    drop(alias);
    assert_eq!(budget.reserved(), 0);
    let mut resumed = make_scan().into_prefetched_batches();
    let mut count = 0;
    while let Some(batch) = resumed.try_next().await.unwrap() {
        count += batch.num_rows();
    }
    assert_eq!(count, 10);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::too_small(1, "minimal prefix")]
#[case::above_pool(129, "exceeds allocation budget")]
#[tokio::test]
async fn workspace_refusal_is_a_terminal_error(#[case] bytes: usize, #[case] message: &str) {
    let dataset = null_dataset(3).await;
    let budget = AllocationBudget::new(128);
    let mut stream = dataset
        .try_scan_admitted_fullzip(
            dataset.schema().project(&["late"]).unwrap(),
            Default::default(),
            2,
            bytes,
            budget.clone(),
        )
        .unwrap()
        .into_prefetched_batches();
    let error = tokio::time::timeout(Duration::from_secs(5), stream.try_next())
        .await
        .unwrap()
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message));
    assert!(stream.try_next().await.unwrap().is_none());
    tokio::time::timeout(Duration::from_secs(5), budget.wait_for_capacity(128))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn empty_snapshot_finishes_while_pool_is_occupied() {
    let dataset = null_dataset(0).await;
    let budget = AllocationBudget::new(128);
    let held = budget.try_reserve(128).unwrap();
    let mut stream = dataset
        .try_scan_admitted_fullzip(
            dataset.schema().project(&["late"]).unwrap(),
            Default::default(),
            2,
            64,
            budget.clone(),
        )
        .unwrap()
        .into_prefetched_batches();
    assert!(
        tokio::time::timeout(Duration::from_secs(5), stream.try_next())
            .await
            .unwrap()
            .unwrap()
            .is_none()
    );
    assert_eq!(budget.reserved(), 128);
    drop(held);
    assert_eq!(budget.reserved(), 0);
}
