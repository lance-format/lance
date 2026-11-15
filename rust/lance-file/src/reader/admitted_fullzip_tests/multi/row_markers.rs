// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{UInt64Array, cast::AsArray, types::UInt64Type};
use lance_core::{ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_ID};

use super::*;

#[rstest]
#[case::address(false, true, false)]
#[case::stable(true, false, true)]
#[case::address_id(true, false, false)]
#[case::both_stable(true, true, true)]
#[case::both_address(true, true, false)]
#[tokio::test]
async fn physical_deletion_markers_preserve_values_and_shared_bitmap(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values("set", "bitmap", "all", "empty")] kind: &str,
    #[case] has_id: bool,
    #[case] has_address: bool,
    #[case] is_stable: bool,
) {
    let input = batches();
    let expected = arrow_select::concat::concat_batches(&input[0].schema(), &input[..5]).unwrap();
    let (_left_fs, left) =
        fixture_batches(version, lite, vec![expected.project(&[0]).unwrap()], 1).await;
    // The second file has a different page boundary. A 13-row batch crosses
    // validity byte boundaries and leaves non-byte-aligned tails.
    let frames = expected.project(&[1]).unwrap();
    let (_right_fs, right) = fixture_batches(
        version,
        !lite,
        vec![frames.slice(0, 17), frames.slice(17, 18)],
        1,
    )
    .await;
    let offsets = match kind {
        "all" => (0..35).collect::<Vec<u32>>(),
        "empty" => Vec::new(),
        _ => vec![0, 1, 7, 8, 12, 16, 17, 24, 25, 31, 34],
    };
    let deleted = Arc::new(if kind == "set" {
        DeletionVector::Set(offsets.iter().copied().collect())
    } else {
        DeletionVector::Bitmap(offsets.iter().copied().collect())
    });
    let (ids, sequence) = row_ids::stable_ids(35);
    let stats = IoStats::new();
    let left = left.with_io_stats(stats.recorder());
    let right = right.with_io_stats(stats.recorder());
    let budget = AllocationBudget::new(16384);
    let a = left
        .open_admitted_fullzip_scan(1..27, budget.clone(), 16384, 13)
        .await
        .unwrap();
    let b = right
        .open_admitted_fullzip_scan(1..27, budget.clone(), 16384, 13)
        .await
        .unwrap();
    let mut scan = a.try_merge(b).unwrap();
    if has_address {
        scan = scan.with_row_address(23).unwrap();
    }
    if has_id {
        scan = scan.with_row_id(23, is_stable.then_some(sequence)).unwrap();
    }
    scan = scan
        .with_deleted_rows_as_nulls(deleted)
        .unwrap()
        .with_row_created_at_version(None)
        .unwrap();
    assert_eq!(stats.snapshot().iops, 0);
    for selection in [1..27, 29..35] {
        if scan.position() != selection.start {
            scan.advance_to_range(selection.clone()).unwrap();
        }
        let mut next = selection.start;
        loop {
            match scan.try_next().await.unwrap() {
                FullZipScanOutcome::Batch { rows, batch } => {
                    assert_eq!(rows.start, next);
                    assert!(rows.end > next && rows.end <= selection.end);
                    next = rows.end;
                    assert_batch(&batch.project(&[0, 1]).unwrap(), &rows, &expected);
                    let addresses = rows
                        .clone()
                        .map(|row| {
                            (!offsets.contains(&(row as u32)))
                                .then_some(23_u64 * (1_u64 << 32) + row)
                        })
                        .collect::<Vec<_>>();
                    if has_address {
                        assert_eq!(
                            batch[ROW_ADDR]
                                .as_primitive::<UInt64Type>()
                                .iter()
                                .collect::<Vec<_>>(),
                            addresses
                        );
                    }
                    if has_id {
                        let values = if is_stable {
                            rows.clone()
                                .map(|row| {
                                    (!offsets.contains(&(row as u32))).then_some(ids[row as usize])
                                })
                                .collect()
                        } else {
                            addresses
                        };
                        assert_eq!(
                            batch[ROW_ID]
                                .as_primitive::<UInt64Type>()
                                .iter()
                                .collect::<Vec<_>>(),
                            values
                        );
                    }
                    assert_eq!(
                        batch[ROW_CREATED_AT_VERSION].to_data(),
                        UInt64Array::from(vec![1; batch.num_rows()]).to_data()
                    );
                    let identifier = &batch[if has_id { ROW_ID } else { ROW_ADDR }];
                    let has_deleted = rows.clone().any(|row| offsets.contains(&(row as u32)));
                    assert_eq!(identifier.nulls().is_some(), has_deleted);
                    if has_id && has_address {
                        if has_deleted {
                            assert_eq!(
                                batch[ROW_ID].nulls().unwrap().buffer().as_ptr(),
                                batch[ROW_ADDR].nulls().unwrap().buffer().as_ptr()
                            );
                        }
                        if !is_stable {
                            assert!(Arc::ptr_eq(&batch[ROW_ID], &batch[ROW_ADDR]));
                        }
                    }
                    // Retain only one bit, independent of all value-buffer owners.
                    let bit = identifier.nulls().map(|nulls| nulls.slice(0, 1));
                    let expected_bit = !offsets.contains(&(rows.start as u32));
                    let count = batch.num_rows();
                    drop(batch);
                    assert_eq!(
                        budget.reserved(),
                        if has_deleted { count.div_ceil(8) } else { 0 }
                    );
                    if let Some(bit) = bit {
                        assert_eq!(bit.is_valid(0), expected_bit);
                        let before = stats.snapshot().iops;
                        if next < selection.end {
                            assert!(matches!(
                                scan.try_next().await.unwrap(),
                                FullZipScanOutcome::NeedsCapacity
                            ));
                            assert_eq!(scan.position(), next);
                        }
                        assert_eq!(stats.snapshot().iops, before);
                        drop(bit);
                        assert_eq!(budget.reserved(), 0);
                    }
                }
                FullZipScanOutcome::Finished => break,
                FullZipScanOutcome::NeedsCapacity => {
                    panic!("complete marker batch must fit after drain")
                }
            }
        }
        assert_eq!(next, selection.end);
    }
    assert!(
        stats.snapshot().iops > 0,
        "deleted payload must still be delivered"
    );
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn marker_bitmap_is_admitted_before_payload(#[values(false, true)] empty: bool) {
    let (_fs, reader) = fixture_batches(
        ConcreteFileVersion::V2_3,
        false,
        vec![batches().remove(0)],
        1,
    )
    .await;
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    // Eight bytes can hold one identifier, but cannot also hold its validity bit.
    let budget = AllocationBudget::new(8);
    let mut scan = reader
        .open_admitted_fullzip_scan(0..if empty { 0 } else { 7 }, budget.clone(), 8, 7)
        .await
        .unwrap()
        .with_row_address(7)
        .unwrap()
        .with_deleted_rows_as_nulls(Arc::new((0_u32..7).collect()))
        .unwrap();
    let result = scan.try_next().await.unwrap();
    if empty {
        assert!(matches!(result, FullZipScanOutcome::Finished));
    } else {
        assert!(matches!(result, FullZipScanOutcome::NeedsCapacity));
    }
    assert_eq!(scan.position(), 0);
    assert_eq!(stats.snapshot().iops, 0);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::missing("missing", "generated row ID or row address")]
#[case::invalid("invalid", "offsets below")]
#[case::started("started", "before the first read")]
#[case::logical("logical", "physical ranges")]
#[tokio::test]
async fn invalid_marker_configuration_preserves_no_io(#[case] kind: &str, #[case] message: &str) {
    let (_fs, reader) = fixture_batches(
        ConcreteFileVersion::V2_3,
        false,
        vec![batches().remove(0)],
        1,
    )
    .await;
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    let budget = AllocationBudget::new(4096);
    let held = budget.try_reserve(4096).unwrap();
    let scan = reader
        .open_admitted_fullzip_scan(0..0, budget.clone(), 4096, 7)
        .await
        .unwrap();
    let deleted = Arc::new([1_u32].into_iter().collect());
    let error = match kind {
        "missing" => scan.with_deleted_rows_as_nulls(deleted).unwrap_err(),
        "invalid" => scan
            .with_row_address(7)
            .unwrap()
            .with_deleted_rows_as_nulls(Arc::new([7_u32].into_iter().collect()))
            .unwrap_err(),
        "started" => {
            let mut scan = scan.with_row_address(7).unwrap();
            scan.advance_to_range(0..7).unwrap();
            assert!(matches!(
                scan.try_next().await.unwrap(),
                FullZipScanOutcome::NeedsCapacity
            ));
            scan.with_deleted_rows_as_nulls(deleted).unwrap_err()
        }
        "logical" => {
            let mut scan = scan
                .with_row_address(7)
                .unwrap()
                .with_deleted_rows_as_nulls(deleted)
                .unwrap();
            let error = scan.advance_to_live_range(0..2).unwrap_err();
            assert_eq!(scan.position(), 0);
            assert!(matches!(
                scan.try_next().await.unwrap(),
                FullZipScanOutcome::Finished
            ));
            error
        }
        _ => unreachable!(),
    };
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(stats.snapshot().iops, 0);
    assert_eq!(budget.reserved(), 4096);
    drop(held);
    assert_eq!(budget.reserved(), 0);
}
