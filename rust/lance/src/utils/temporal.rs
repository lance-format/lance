// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! These utilities make it possible to mock out the "current time" when running
//! unit tests.  Anywhere in production code where we need to get the current time
//! we should use the below methods and types instead of the builtin methods and types

use chrono::{DateTime, TimeZone, Utc};
use lance_core::{Error, Result};
#[cfg(test)]
use mock_instant::thread_local::{SystemTime as NativeSystemTime, UNIX_EPOCH};

#[cfg(not(test))]
use std::time::{SystemTime as NativeSystemTime, UNIX_EPOCH};

pub type SystemTime = NativeSystemTime;

/// Mirror function that mimics DateTime<Utc>::now() with the exception that it
/// uses the potentially mocked system time.
pub fn utc_now() -> DateTime<Utc> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before Unix epoch");
    let naive = DateTime::from_timestamp(now.as_secs() as i64, now.subsec_nanos())
        .expect("DateTime::from_timestamp")
        .naive_utc();
    Utc.from_utc_datetime(&naive)
}

pub fn timestamp_to_nanos(timestamp: Option<SystemTime>) -> u128 {
    let timestamp = timestamp.unwrap_or_else(SystemTime::now);
    timestamp
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

const MILLIS_TO_NANOS: u128 = 1_000_000;

/// Returns a timestamp in nanoseconds that is guaranteed to be strictly greater
/// than `prev_timestamp_nanos` (by at least 1 millisecond).
///
/// Formula: `max(current_time_nanos, prev_timestamp_nanos + 1ms)`
///
/// If `prev_timestamp_nanos` is `None` (first version), returns the current time.
pub fn monotonic_timestamp_nanos(
    timestamp: Option<SystemTime>,
    prev_timestamp_nanos: Option<u128>,
) -> Result<u128> {
    let current_nanos = timestamp_to_nanos(timestamp);
    match prev_timestamp_nanos {
        Some(prev) => {
            let minimum = prev.checked_add(MILLIS_TO_NANOS).ok_or_else(|| {
                Error::invalid_input(format!(
                    "previous manifest timestamp {prev} cannot be increased by {MILLIS_TO_NANOS} nanoseconds"
                ))
            })?;
            Ok(current_nanos.max(minimum))
        }
        None => Ok(current_nanos),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use mock_instant::thread_local::MockClock;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::first(100, None, 100_000_000_000)]
    #[case::clock_ahead(200, Some(100_000_000_000), 200_000_000_000)]
    #[case::clock_behind(100, Some(200_000_000_000), 200_001_000_000)]
    #[case::same_time(100, Some(100_000_000_000), 100_001_000_000)]
    fn test_monotonic_timestamp(
        #[case] now_seconds: u64,
        #[case] previous: Option<u128>,
        #[case] expected: u128,
    ) {
        MockClock::set_system_time(Duration::from_secs(now_seconds));
        assert_eq!(monotonic_timestamp_nanos(None, previous).unwrap(), expected);
    }

    #[test]
    fn test_monotonic_timestamp_overflow() {
        let error = monotonic_timestamp_nanos(None, Some(u128::MAX)).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(&u128::MAX.to_string()));
    }
}
