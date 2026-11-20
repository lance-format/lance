// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::AdmittedDatasetScan;
use std::{task::Poll, time::Duration};

fn null_scan(
    dataset: &Dataset,
    budget: Arc<AllocationBudget>,
    bytes: usize,
) -> AdmittedDatasetScan {
    dataset
        .try_scan_admitted_fullzip(
            dataset.schema().project(&["late"]).unwrap(),
            Default::default(),
            2,
            bytes,
            budget,
        )
        .unwrap()
}

#[tokio::test]
async fn waiter_keeps_position_and_cancellation_does_not_consume_another_wake() {
    let dataset = null_dataset(3).await;
    let budget = AllocationBudget::new(64);
    let mut producer = null_scan(&dataset, budget.clone(), 64);
    let mut left = null_scan(&dataset, budget.clone(), 64);
    let mut cancelled = null_scan(&dataset, budget.clone(), 64);
    let Outcome::Batch { batch, .. } = producer.try_next().await.unwrap() else {
        panic!("first prefix fits");
    };
    let alias = batch["late"].to_data().buffers()[0].slice_with_length(0, 1);
    drop(batch);
    let owned = budget.reserved();
    assert!(owned > 1 && owned <= 64);
    // Open both readers before polling the capacity waits, without consuming rows.
    for scan in [&mut left, &mut cancelled] {
        assert!(matches!(
            scan.try_next().await.unwrap(),
            Outcome::NeedsCapacity
        ));
    }
    let left_position = left.position();
    let cancelled_position = cancelled.position();
    let mut waiting = Box::pin(left.next_when_available());
    let mut abandoned = Box::pin(cancelled.next_when_available());
    assert!(matches!(futures::poll!(waiting.as_mut()), Poll::Pending));
    assert!(matches!(futures::poll!(abandoned.as_mut()), Poll::Pending));
    assert_eq!(budget.reserved(), owned);
    drop(abandoned);
    assert_eq!(cancelled.position(), cancelled_position);
    assert_eq!(budget.reserved(), owned);
    drop(alias);
    let Outcome::Batch { rows, batch, .. } = tokio::time::timeout(Duration::from_secs(5), waiting)
        .await
        .unwrap()
        .unwrap()
    else {
        panic!("final alias release must wake the remaining waiter");
    };
    assert_eq!(rows, 0..2);
    assert_eq!(left_position.unwrap().row_offset, 0);
    assert_eq!(left.position().unwrap().row_offset, 2);
    assert_eq!(
        batch["late"].as_ref(),
        arrow_array::new_null_array(&DataType::LargeBinary, 2).as_ref()
    );
    drop(batch);
    let Outcome::Batch { rows, batch, .. } = cancelled.next_when_available().await.unwrap() else {
        panic!("cancelled waiter can resume the same original prefix");
    };
    assert_eq!(rows, 0..2);
    drop(batch);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::cannot_store_selection(1)]
#[case::cannot_decode_minimal_row(20)]
#[tokio::test]
async fn admitted_workspace_refusal_returns_without_waiting(#[case] bytes: usize) {
    let dataset = null_dataset(3).await;
    let budget = AllocationBudget::new(64);
    let mut scan = null_scan(&dataset, budget.clone(), bytes);
    let position = scan.position();
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(5), scan.next_when_available())
            .await
            .unwrap()
            .unwrap(),
        Outcome::NeedsCapacity
    ));
    assert_eq!(scan.position(), position);
    assert_eq!(budget.reserved(), 0);
    scan.set_workspace_bytes(64).unwrap();
    let Outcome::Batch { rows, batch, .. } = scan.next_when_available().await.unwrap() else {
        panic!("replanning preserves the original prefix");
    };
    assert_eq!(rows, 0..2);
    drop(batch);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn wait_rejects_impossible_admission_but_does_not_admit_eof() {
    let dataset = null_dataset(1).await;
    let budget = AllocationBudget::new(64);
    let mut scan = null_scan(&dataset, budget.clone(), 65);
    let position = scan.position();
    let error = scan.next_when_available().await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("exceeds allocation budget"));
    assert_eq!(scan.position(), position);
    assert_eq!(budget.reserved(), 0);
    scan.set_workspace_bytes(64).unwrap();
    let Outcome::Batch { batch, .. } = scan.next_when_available().await.unwrap() else {
        panic!("replanned row fits");
    };
    drop(batch);
    let held = budget.try_reserve(64).unwrap();
    assert!(matches!(
        scan.next_when_available().await.unwrap(),
        Outcome::Finished
    ));
    let empty = null_dataset(0).await;
    let mut empty_scan = null_scan(&empty, budget.clone(), 64);
    assert!(matches!(
        empty_scan.next_when_available().await.unwrap(),
        Outcome::Finished
    ));
    assert_eq!(budget.reserved(), 64);
    drop(held);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn concurrent_dataset_drains_share_one_workspace(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
) {
    let mut dataset = write_payload_dataset(version, true, true).await;
    let field = dataset.schema().field("payload").unwrap().id;
    for fragment in 0..2 {
        dataset = commit_overlay(
            dataset,
            &format!("wait-{fragment}"),
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
    let config = FragReadConfig::default().with_scan_scheduler(ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig {
            io_buffer_size_bytes: 1024,
            use_lite_scheduler: Some(lite),
        },
    ));
    let mut expected = Vec::new();
    for id in 0..2 {
        let reader = dataset
            .get_fragment(id)
            .unwrap()
            .open(dataset.schema(), config.clone())
            .await
            .unwrap();
        for start in [0, 2, 4] {
            expected.push((
                id as u64,
                start as u64..start as u64 + 2,
                reader
                    .take_as_batch(&[start, start + 1], None)
                    .await
                    .unwrap(),
            ));
        }
    }
    let budget = AllocationBudget::new(65536);
    let scans = (0..2)
        .map(|_| {
            dataset
                .try_scan_admitted_fullzip(
                    dataset.schema().clone(),
                    config.clone(),
                    2,
                    65536,
                    budget.clone(),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    let drain = |mut scan: AdmittedDatasetScan| {
        let expected = &expected;
        let budget = &budget;
        async move {
            for (id, range, oracle) in expected {
                let Outcome::Batch {
                    fragment_id,
                    rows,
                    batch,
                } = scan.next_when_available().await.unwrap()
                else {
                    panic!("admission contention must not escape as workspace refusal");
                };
                assert_eq!(fragment_id, *id);
                assert_eq!(&rows, range);
                assert_eq!(&batch, oracle);
                assert!(budget.reserved() <= 65536);
                // A slow consumer keeps output across a scheduling point. The
                // other cursor must wait in async code, not occupy the CPU worker.
                tokio::task::yield_now().await;
                drop(batch);
            }
            assert!(matches!(
                scan.next_when_available().await.unwrap(),
                Outcome::Finished
            ));
        }
    };
    tokio::time::timeout(
        Duration::from_secs(5),
        futures::future::join_all(scans.into_iter().map(drain)),
    )
    .await
    .expect("both finite scans must drain with one workspace and one CPU worker");
    assert_eq!(budget.reserved(), 0);
}
