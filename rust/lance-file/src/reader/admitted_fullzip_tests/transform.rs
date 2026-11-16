// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    cell::RefCell,
    sync::atomic::{AtomicUsize, Ordering},
};

use lance_core::{Result, utils::allocation::Reservation, utils::tokio::spawn_cpu};
use lance_encoding::data::try_interleave_nested_binary_in;

use super::*;
use crate::reader::{FullZipScan, FullZipScanOutcome};

const WORKSPACE: usize = 16384;

async fn merged_fixture(
    version: ConcreteFileVersion,
    lite: bool,
    depth: usize,
    budget: Arc<AllocationBudget>,
) -> (FsFixture, FsFixture, FullZipScan, ArrayRef) {
    let left = [array(0, depth), array(1, depth)];
    let right = [array(17, depth), array(18, depth)];
    let make_batches = |name: &str, arrays: &[ArrayRef]| {
        let schema = Arc::new(Schema::new(vec![
            Field::new(name, arrays[0].data_type().clone(), true).with_metadata(HashMap::from([(
                STRUCTURAL_ENCODING_META_KEY.to_string(),
                STRUCTURAL_ENCODING_FULLZIP.to_string(),
            )])),
        ]));
        arrays
            .iter()
            .map(|array| RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap())
            .collect()
    };
    let (left_fs, left_reader) =
        fixture_batches(version, lite, make_batches("left", &left), 1).await;
    let (right_fs, right_reader) =
        fixture_batches(version, lite, make_batches("right", &right), 1).await;
    let left = arrow_select::concat::concat(&[left[0].as_ref(), left[1].as_ref()]).unwrap();
    let right = arrow_select::concat::concat(&[right[0].as_ref(), right[1].as_ref()]).unwrap();
    let indices = (0..left.len())
        .map(|row| (row % 2, row))
        .collect::<Vec<_>>();
    let expected =
        arrow_select::interleave::interleave(&[left.as_ref(), right.as_ref()], &indices).unwrap();
    let left = left_reader
        .open_admitted_fullzip_scan(0..left_reader.num_rows(), budget.clone(), WORKSPACE, 3)
        .await
        .unwrap();
    let right = right_reader
        .open_admitted_fullzip_scan(0..right_reader.num_rows(), budget, WORKSPACE, 3)
        .await
        .unwrap();
    (left_fs, right_fs, left.try_merge(right).unwrap(), expected)
}

fn assemble(
    rows: Range<u64>,
    batch: RecordBatch,
    workspace: &mut Reservation,
) -> Result<Option<RecordBatch>> {
    // This small test routing vector is metadata, not part of the payload charge.
    let indices = rows
        .enumerate()
        .map(|(local, row)| ((row % 2) as usize, local))
        .collect::<Vec<_>>();
    let Some(output) = try_interleave_nested_binary_in(
        &[batch.column(0).as_ref(), batch.column(1).as_ref()],
        &indices,
        workspace,
    )?
    else {
        return Ok(None);
    };
    let schema = Arc::new(Schema::new(vec![Field::new(
        "frames",
        output.data_type().clone(),
        true,
    )]));
    Ok(Some(RecordBatch::try_new(schema, vec![output])?))
}

#[rstest]
#[tokio::test]
async fn admitted_transform_interleaves_real_inputs(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(1, 2)] depth: usize,
    #[values(false, true)] needs_scratch: bool,
) {
    let budget = AllocationBudget::new(WORKSPACE + 512);
    let peer = budget.try_reserve(512).unwrap();
    let (_left_fs, _right_fs, mut scan, expected) =
        merged_fixture(version, lite, depth, budget.clone()).await;
    let largest_attempt = Arc::new(AtomicUsize::new(0));
    let mut position = 0;
    while position < expected.len() {
        let observed = largest_attempt.clone();
        let pool = budget.clone();
        let outcome = scan
            .try_next_with_transform(move |rows, batch, workspace| {
                observed.fetch_max(batch.num_rows(), Ordering::SeqCst);
                // A downstream stage's scratch must overlap both decoded input
                // and assembly. Two rows cannot fit this finite workspace; one can.
                let scratch = if needs_scratch {
                    let Some(scratch) = workspace.try_allocate(batch.num_rows() * 8192)? else {
                        return Ok(None);
                    };
                    Some(scratch)
                } else {
                    None
                };
                let credit = workspace.len();
                let charged = pool.reserved();
                let output = assemble(rows, batch, workspace)?;
                assert!(
                    output.is_some(),
                    "selected output must fit after scratch admission"
                );
                assert!(
                    workspace.len() < credit,
                    "assembly must use the original workspace"
                );
                assert!(
                    pool.reserved() < charged,
                    "decoded input owners must be released"
                );
                assert!(pool.reserved() <= WORKSPACE + 512);
                drop(scratch);
                Ok(output)
            })
            .await
            .unwrap();
        let FullZipScanOutcome::Batch { rows, batch } = outcome else {
            panic!("a finite prefix must progress after its predecessor is drained")
        };
        assert_eq!(rows.start as usize, position);
        assert_eq!(rows.end - rows.start, batch.num_rows() as u64);
        if needs_scratch {
            assert_eq!(
                batch.num_rows(),
                1,
                "assembly refusal must replan the whole read"
            );
        }
        assert_eq!(
            batch.column(0).to_data(),
            expected.slice(position, batch.num_rows()).to_data()
        );
        position += batch.num_rows();
        assert_eq!(scan.position() as usize, position);
        let alias = batch.column(0).slice(0, 1);
        drop(batch);
        assert!(budget.reserved() > peer.len());
        if position < expected.len() {
            assert!(matches!(
                scan.try_next_with_transform(|_, _, _| panic!("no second admission fits"))
                    .await
                    .unwrap(),
                FullZipScanOutcome::NeedsCapacity
            ));
            assert_eq!(scan.position() as usize, position);
        }
        drop(alias);
        assert_eq!(budget.reserved(), peer.len());
    }
    assert!(largest_attempt.load(Ordering::SeqCst) > 1);
    assert!(matches!(
        scan.try_next().await.unwrap(),
        FullZipScanOutcome::Finished
    ));
    drop(peer);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::refuse("refuse")]
#[case::error("error")]
#[case::row_count("row_count")]
#[tokio::test]
async fn admitted_transform_failure_does_not_commit(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[case] outcome: &'static str,
) {
    let budget = AllocationBudget::new(WORKSPACE);
    let (_left_fs, _right_fs, mut scan, expected) =
        merged_fixture(version, false, 1, budget.clone()).await;
    let smallest = Arc::new(AtomicUsize::new(usize::MAX));
    let observed = smallest.clone();
    let result = scan
        .try_next_with_transform(move |rows, batch, workspace| {
            observed.fetch_min(batch.num_rows(), Ordering::SeqCst);
            let batch = assemble(rows, batch, workspace)?.unwrap();
            match outcome {
                "refuse" => Ok(None),
                "error" => Err(Error::invalid_input("assembly rejected fixture")),
                "row_count" => Ok(Some(batch.slice(0, batch.num_rows() - 1))),
                _ => unreachable!(),
            }
        })
        .await;
    if outcome == "refuse" {
        assert!(matches!(result.unwrap(), FullZipScanOutcome::NeedsCapacity));
        assert_eq!(smallest.load(Ordering::SeqCst), 1);
    } else {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(if outcome == "error" {
            "assembly rejected fixture"
        } else {
            "FullZip transform returned 2 rows, expected 3"
        }));
    }
    assert_eq!(scan.position(), 0);
    assert_eq!(
        budget.reserved(),
        0,
        "all attempted inputs and outputs must drop"
    );
    let FullZipScanOutcome::Batch { rows, batch } =
        scan.try_next_with_transform(assemble).await.unwrap()
    else {
        panic!("retry must deliver the original rows")
    };
    assert_eq!(rows, 0..3);
    assert_eq!(batch.column(0).to_data(), expected.slice(0, 3).to_data());
    drop(batch);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test(flavor = "current_thread")]
async fn admitted_transform_cancellation_retains_job_credit(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] is_completed: bool,
) {
    let budget = AllocationBudget::new(WORKSPACE);
    let (_left_fs, _right_fs, mut scan, _) =
        merged_fixture(version, false, 1, budget.clone()).await;
    let queued = RefCell::new(None);
    let entered = tokio::sync::Notify::new();
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let mut receiver = Some(receiver);
    let mut future = Box::pin(scan.try_next_with(|job| {
        *queued.borrow_mut() = Some(job);
        entered.notify_one();
        let receiver = receiver.take().expect("first attempt fits");
        async move { receiver.await.unwrap() }
    }));
    tokio::select! {
        _ = entered.notified() => (),
        result = &mut future => panic!("queued job completed: {result:?}"),
    }
    let job = queued.borrow_mut().take().unwrap();
    if is_completed {
        sender
            .send(spawn_cpu(move || job.decode_with(assemble)).await)
            .unwrap();
        assert!(
            budget.reserved() > 0,
            "undelivered assembly still owns its payload"
        );
        drop(future);
    } else {
        drop(future);
        assert_eq!(
            budget.reserved(),
            WORKSPACE,
            "queued CPU job still owns admission"
        );
        let result = spawn_cpu(move || job.decode_with(assemble)).await;
        assert!(result.as_ref().unwrap().is_some());
        drop(sender.send(result).unwrap_err());
    }
    assert_eq!(scan.position(), 0);
    assert_eq!(budget.reserved(), 0);
    let FullZipScanOutcome::Batch { rows, batch } = scan.try_next().await.unwrap() else {
        panic!("default decoding must still deliver the cancelled rows")
    };
    assert_eq!(rows, 0..3);
    assert_eq!(batch.column(0).to_data(), array(0, 1).slice(0, 3).to_data());
    assert_eq!(
        batch.column(1).to_data(),
        array(17, 1).slice(0, 3).to_data()
    );
    drop(batch);
    assert_eq!(budget.reserved(), 0);
}
