// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::{
    fragment::AdmittedFragmentScanOutcome as Outcome, schema_evolution::NewColumnTransform,
};
use std::{future::pending, task::Poll};

#[rstest]
#[tokio::test]
async fn cursor_composes_real_overlay_missing_and_deleted_rows(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
) {
    let mut dataset = write_payload_dataset(version, true, true).await;
    let field = dataset.schema().field("payload").unwrap().id;
    for fragment in 0..2 {
        dataset = commit_overlay(
            dataset,
            &format!("cursor-{fragment}"),
            fragment,
            &[field],
            OverlayCoverage::dense(bitmap([1, 3, 4])),
            vec![payload("new", 3, true, 0)],
            version,
        )
        .await;
    }
    dataset
        .add_columns(
            NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "late",
                DataType::LargeBinary,
                true,
            )]))),
            None,
            None,
        )
        .await
        .unwrap();
    dataset
        .delete("untouched IN ('other0', 'other1', 'other6', 'other7')")
        .await
        .unwrap();
    for fragment in 0..2 {
        for markers in [false, true] {
            let mut reader = open(&dataset, fragment, dataset.schema(), lite).await;
            reader.with_row_id();
            reader.with_row_address();
            if markers {
                reader.with_make_deletions_null();
            }
            let budget = AllocationBudget::new(2 * 65536);
            let mut left = reader
                .try_scan_admitted_fullzip(0..6, 2, 65536, budget.clone())
                .unwrap();
            let mut right = reader
                .try_scan_admitted_fullzip(0..6, 2, 65536, budget.clone())
                .unwrap();
            for start in [0, 2, 4] {
                let (a, b) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
                    tokio::join!(left.try_next(), right.try_next())
                })
                .await
                .expect("independent fragment cursors must progress with one CPU worker");
                let expected = reader
                    .take_as_batch(&[start, start + 1], None)
                    .await
                    .unwrap();
                for result in [a, b] {
                    let Outcome::Batch { rows, batch } = result.unwrap() else {
                        panic!("a complete physical prefix must be delivered");
                    };
                    assert_eq!(rows, u64::from(start)..u64::from(start + 2));
                    assert_eq!(batch, expected);
                    if start == 0 && !markers {
                        assert_eq!(batch.num_rows(), 0);
                    }
                    drop(batch);
                }
                assert_eq!(left.position(), u64::from(start + 2));
                assert_eq!(right.position(), left.position());
                assert_eq!(budget.reserved(), 0);
            }
            assert!(matches!(left.try_next().await.unwrap(), Outcome::Finished));
            assert!(matches!(right.try_next().await.unwrap(), Outcome::Finished));
            assert_eq!(budget.reserved(), 0);
        }
    }
}

async fn null_reader(rows: u32) -> FragmentReader {
    let batch = arrow_array::record_batch!(("id", UInt32, (0..rows).collect::<Vec<_>>())).unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: rows as usize,
            data_storage_version: Some(LanceFileVersion::V2_3),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .add_columns(
            NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "late",
                DataType::LargeBinary,
                true,
            )]))),
            None,
            None,
        )
        .await
        .unwrap();
    open(
        &dataset,
        0,
        &dataset.schema().project(&["late"]).unwrap(),
        false,
    )
    .await
}

#[rstest]
#[case::smaller(128)]
#[case::larger(512)]
#[tokio::test]
async fn cursor_drains_more_than_its_workspace_and_honors_alias_backpressure(#[case] total: u32) {
    let reader = null_reader(total).await;
    let budget = AllocationBudget::new(64);
    let mut scan = reader
        .try_scan_admitted_fullzip(0..u64::from(total), u32::MAX, 64, budget.clone())
        .unwrap();
    let mut delivered = 0;
    let mut cumulative_offsets = 0;
    let mut largest_batch = 0;
    loop {
        match scan.try_next().await.unwrap() {
            Outcome::Batch { rows, batch } => {
                assert_eq!(rows.start, delivered);
                assert_eq!(batch.num_rows() as u64, rows.end - rows.start);
                assert_eq!(batch["late"].null_count(), batch.num_rows());
                assert_eq!(
                    batch["late"].as_ref(),
                    arrow_array::new_null_array(&DataType::LargeBinary, batch.num_rows()).as_ref()
                );
                assert!(budget.reserved() <= 64);
                largest_batch = largest_batch.max(batch.num_rows());
                cumulative_offsets += (batch.num_rows() + 1) * 8;
                delivered = rows.end;
                let alias = batch["late"].to_data().buffers()[0].slice_with_length(0, 1);
                drop(batch);
                if delivered < u64::from(total) {
                    assert!(matches!(
                        scan.try_next().await.unwrap(),
                        Outcome::NeedsCapacity
                    ));
                    assert_eq!(scan.position(), delivered);
                }
                drop(alias);
                assert_eq!(budget.reserved(), 0);
            }
            Outcome::NeedsCapacity => panic!("a minimal null row fits the empty workspace"),
            Outcome::Finished => break,
        }
    }
    assert_eq!(delivered, u64::from(total));
    assert!(cumulative_offsets > 64 * 16);
    assert!(
        largest_batch > 1,
        "do not degrade to mandatory single-row output"
    );
    assert_eq!(scan.position(), u64::from(total));
    assert!(matches!(scan.try_next().await.unwrap(), Outcome::Finished));
}

#[tokio::test]
async fn cursor_refusal_and_cancel_do_not_consume_rows_or_hold_credit() {
    let reader = null_reader(6).await;
    let budget = AllocationBudget::new(64);
    let mut too_small = reader
        .try_scan_admitted_fullzip(1..4, 3, 20, budget.clone())
        .unwrap();
    assert!(matches!(
        too_small.try_next().await.unwrap(),
        Outcome::NeedsCapacity
    ));
    assert_eq!(too_small.position(), 1);
    assert_eq!(budget.reserved(), 0);
    let mut scan = reader
        .try_scan_admitted_fullzip(1..4, 3, 64, budget.clone())
        .unwrap();
    let held = budget.try_reserve(1).unwrap();
    assert!(matches!(
        scan.try_next().await.unwrap(),
        Outcome::NeedsCapacity
    ));
    assert_eq!(scan.position(), 1);
    drop(held);

    let mut cancelled = Box::pin(scan.try_next_with(|indices, workspace| async move {
        assert_eq!(indices.values().as_ref(), &[1, 2, 3]);
        assert_eq!(workspace.len(), 64 - 3 * size_of::<u32>());
        pending::<()>().await;
        drop((indices, workspace));
        Ok(None)
    }));
    assert!(matches!(futures::poll!(cancelled.as_mut()), Poll::Pending));
    assert_eq!(budget.reserved(), 64);
    drop(cancelled);
    assert_eq!(scan.position(), 1);
    assert_eq!(budget.reserved(), 0);

    let error = scan
        .try_next_with(|indices, workspace| async move {
            drop((indices, workspace));
            Err(Error::corrupt_file_named(
                "fixture",
                "injected input failure",
            ))
        })
        .await
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("injected input failure"));
    assert_eq!(scan.position(), 1);
    assert_eq!(budget.reserved(), 0);

    let Outcome::Batch { rows, batch } = scan.try_next().await.unwrap() else {
        panic!("retry must return the unconsumed range");
    };
    assert_eq!(rows, 1..4);
    assert_eq!(batch, reader.take_as_batch(&[1, 2, 3], None).await.unwrap());
    drop(batch);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn cursor_propagates_unsupported_input_without_skipping() {
    let dataset = create_base_dataset(LanceFileVersion::V2_2).await;
    let reader = open(&dataset, 0, dataset.schema(), false).await;
    let budget = AllocationBudget::new(8192);
    let mut scan = reader
        .try_scan_admitted_fullzip(1..4, 3, 8192, budget.clone())
        .unwrap();
    let error = scan.try_next().await.unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("binary/string leaves"));
    assert_eq!(scan.position(), 1);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn cursor_validates_ranges_and_preserves_final_u32_offset() {
    let mut reader = null_reader(1).await;
    let budget = AllocationBudget::new(64);
    for (start, end, max, bytes, text) in [
        (2, 1, 1, 64, "physical domain"),
        (0, 2, 1, 64, "physical domain"),
        (0, 1, 0, 64, "positive"),
        (0, 1, 1, 0, "positive"),
    ] {
        let Err(error) = reader.try_scan_admitted_fullzip(start..end, max, bytes, budget.clone())
        else {
            panic!("invalid range or workspace was accepted");
        };
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(text));
    }
    let mut empty = reader
        .try_scan_admitted_fullzip(1..1, 1, 64, budget.clone())
        .unwrap();
    let held = budget.try_reserve(64).unwrap();
    assert!(matches!(empty.try_next().await.unwrap(), Outcome::Finished));
    drop((empty, held));
    // Exercise the cursor's complete physical domain without creating a 2^32-row file.
    reader.num_physical_rows = (u64::from(u32::MAX) + 1) as usize;
    let mut final_row = reader
        .try_scan_admitted_fullzip(
            u64::from(u32::MAX)..u64::from(u32::MAX) + 1,
            1,
            64,
            budget.clone(),
        )
        .unwrap();
    let result = final_row
        .try_next_with(|indices, workspace| async move {
            assert_eq!(indices.values().as_ref(), &[u32::MAX]);
            drop((indices, workspace));
            Ok(Some(RecordBatch::new_empty(Arc::new(ArrowSchema::empty()))))
        })
        .await
        .unwrap();
    assert!(matches!(result, Outcome::Batch { .. }));
    assert_eq!(final_row.position(), u64::from(u32::MAX) + 1);
    assert!(matches!(
        final_row.try_next().await.unwrap(),
        Outcome::Finished
    ));
    assert_eq!(budget.reserved(), 0);
}
