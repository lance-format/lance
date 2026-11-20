// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[rstest]
#[tokio::test]
async fn empty_and_invalid_marker_mapping_reads_no_payload(
    #[values(false, true)] all_deleted: bool,
) {
    let (_fs, reader) = fixture_batches(
        ConcreteFileVersion::V2_3,
        false,
        vec![batches().remove(0)],
        1,
    )
    .await;
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    let deleted: DeletionVector = if all_deleted {
        (0_u32..7).collect()
    } else {
        [0_u32, 1, 3, 6].into_iter().collect()
    };
    let live_count = if all_deleted { 0 } else { 3 };
    let budget = AllocationBudget::new(4096);
    let held = budget.try_reserve(4096).unwrap();
    let mut scan = reader
        .open_admitted_fullzip_scan(0..0, budget.clone(), 4096, 7)
        .await
        .unwrap()
        .with_row_address(7)
        .unwrap()
        .with_deleted_rows_as_nulls(Arc::new(deleted))
        .unwrap();
    scan.advance_to_live_range(0..0).unwrap();
    let position = if all_deleted { 7 } else { 2 };
    assert_eq!(scan.position(), position);
    assert!(matches!(
        scan.try_next().await.unwrap(),
        FullZipScanOutcome::Finished
    ));
    for (start, end) in [
        (2, 1),
        (0, live_count + 1),
        (live_count + 1, live_count + 1),
    ] {
        assert!(matches!(
            scan.advance_to_live_range(start..end).unwrap_err(),
            Error::InvalidInput { .. }
        ));
        assert_eq!(scan.position(), position);
    }
    scan.advance_to_live_range(live_count..live_count).unwrap();
    assert_eq!(scan.position(), 7);
    assert!(matches!(
        scan.try_next().await.unwrap(),
        FullZipScanOutcome::Finished
    ));
    assert!(
        scan.with_deleted_rows(Arc::new(DeletionVector::NoDeletions))
            .unwrap_err()
            .to_string()
            .contains("before the first read")
    );
    assert_eq!(stats.snapshot().iops, 0);
    assert_eq!(budget.reserved(), 4096);
    drop(held);
    assert_eq!(budget.reserved(), 0);
}
