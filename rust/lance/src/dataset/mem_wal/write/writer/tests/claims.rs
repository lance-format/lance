// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Write-path WAL trigger claims.

use super::super::*;

#[test]
fn test_wal_size_trigger_coalesces_crossings() {
    let threshold = 64;
    let mut claimed = 0;

    assert!(!claim_wal_size_trigger(&mut claimed, 63, threshold));
    assert!(claim_wal_size_trigger(
        &mut claimed,
        3 * threshold + 1,
        threshold
    ));
    assert_eq!(claimed, 3);

    assert!(!claim_wal_size_trigger(
        &mut claimed,
        3 * threshold + 63,
        threshold
    ));
    assert!(claim_wal_size_trigger(
        &mut claimed,
        4 * threshold,
        threshold
    ));
    assert_eq!(claimed, 4);
    assert!(!claim_wal_size_trigger(&mut claimed, usize::MAX, 0));
}

#[test]
fn test_wal_time_trigger_preserves_interval_without_fixed_boundary() {
    let interval = Some(Duration::from_millis(100));
    let mut last = 0;

    assert!(!claim_wal_time_trigger(&mut last, 1_000, interval));
    assert_eq!(last, 1_000);
    assert!(!claim_wal_time_trigger(&mut last, 1_099, interval));
    assert!(claim_wal_time_trigger(&mut last, 1_100, interval));
    assert_eq!(last, 1_100);
    assert!(!claim_wal_time_trigger(&mut last, 1_199, interval));
    assert!(claim_wal_time_trigger(&mut last, 1_200, interval));
    assert!(!claim_wal_time_trigger(&mut last, 1_300, None));
    assert!(!claim_wal_time_trigger(
        &mut last,
        1_300,
        Some(Duration::ZERO)
    ));
}

#[test]
fn test_pending_wal_size_trigger_coalesces_and_resets_after_drain() {
    let threshold = 64;
    let mut claimed_bytes = 0;

    assert!(claim_pending_wal_size_trigger(
        &mut claimed_bytes,
        3 * threshold + 1,
        threshold
    ));
    assert_eq!(claimed_bytes, 3 * threshold);
    assert!(!claim_pending_wal_size_trigger(
        &mut claimed_bytes,
        3 * threshold + 63,
        threshold
    ));

    assert!(!claim_pending_wal_size_trigger(
        &mut claimed_bytes,
        threshold - 1,
        threshold
    ));
    assert_eq!(claimed_bytes, 0);
    assert!(claim_pending_wal_size_trigger(
        &mut claimed_bytes,
        threshold,
        threshold
    ));
    assert_eq!(claimed_bytes, threshold);
}
