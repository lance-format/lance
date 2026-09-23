// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use std::cmp::Ordering;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

use jiff::SignedDuration;
use jiff::civil::date;
use jiff::fmt::strtime;
use jiff::tz::Offset;

const MICROS_PER_SEC: i64 = 1_000_000;
const MICROS_PER_MINUTE: i64 = 60 * MICROS_PER_SEC;
const MICROS_PER_HOUR: i64 = 60 * MICROS_PER_MINUTE;
const MONTHS_PER_YEAR: i32 = 12;

const TIMESTAMP_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.6f";
const TIMESTAMP_TIMEZONE_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.6f %z";

/// Represents extended JSON value types that are not supported in standard JSON.
///
/// Standard JSON only supports strings, numbers, booleans, null, arrays, and objects.
/// This enum provides additional data types commonly needed in database systems and
/// other applications that require more specialized data representations.
#[derive(Debug, Clone)]
pub enum ExtensionValue<'a> {
    /// Binary data (byte array), allowing efficient storage of binary content
    /// that would otherwise require base64 encoding in standard JSON
    Binary(&'a [u8]),
    /// Calendar date without time component (year, month, day)
    Date(Date),
    /// Timestamp with microsecond precision but without timezone information
    Timestamp(Timestamp),
    /// Timestamp with microsecond precision and timezone offset information
    TimestampTz(TimestampTz),
    /// Time interval representation for duration calculations
    Interval(Interval),
}

/// Represents a calendar date (year, month, day) without time component.
///
/// The value is stored as days since the Unix epoch (January 1, 1970).
/// This allows for efficient date arithmetic and comparison operations.
/// Standard JSON has no native date type and typically uses ISO 8601 strings.
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Date {
    /// Days since Unix epoch (January 1, 1970)
    /// Positive values represent dates after the epoch, negative values represent dates before
    pub value: i32,
}

/// Represents a timestamp (date and time) without timezone information.
///
/// The value is stored as microseconds since the Unix epoch (January 1, 1970 00:00:00 UTC).
/// This provides microsecond precision for timestamp operations.
/// Standard JSON has no native timestamp type and typically uses ISO 8601 strings.
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Timestamp {
    /// Microseconds since Unix epoch (January 1, 1970 00:00:00 UTC)
    pub value: i64,
}

/// Represents a timestamp with timezone information.
///
/// Combines a timestamp value with a timezone offset, allowing for
/// timezone-aware datetime operations. The timestamp is stored in UTC,
/// and the offset indicates the local timezone.
/// Standard JSON has no native timezone-aware timestamp type.
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct TimestampTz {
    /// Timezone offset in seconds from UTC
    pub offset: i32,
    /// Microseconds since Unix epoch (January 1, 1970 00:00:00 UTC)
    pub value: i64,
}

/// Represents a time interval or duration.
///
/// This structure can represent complex time intervals with separate
/// components for months, days, and microseconds, allowing for precise
/// duration calculations that account for calendar irregularities.
/// Standard JSON has no native interval/duration type.
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Interval {
    /// Number of months in the interval
    pub months: i32,
    /// Number of days in the interval
    pub days: i32,
    /// Number of microseconds in the interval
    pub micros: i64,
}

impl Display for Date {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let dur = SignedDuration::from_hours(self.value as i64 * 24);
        let date = date(1970, 1, 1).checked_add(dur).unwrap();
        write!(f, "{}", date)
    }
}

impl Display for Timestamp {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let micros = self.value;
        let (mut secs, mut nanos) = (micros / MICROS_PER_SEC, (micros % MICROS_PER_SEC) * 1_000);
        if nanos < 0 {
            secs -= 1;
            nanos += 1_000_000_000;
        }

        if secs > 253402207200 {
            secs = 253402207200;
            nanos = 0;
        } else if secs < -377705023201 {
            secs = -377705023201;
            nanos = 0;
        }
        let ts = jiff::Timestamp::new(secs, nanos as i32).unwrap();

        write!(f, "{}", strtime::format(TIMESTAMP_FORMAT, ts).unwrap())
    }
}

impl Display for TimestampTz {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let micros = self.value;
        let (mut secs, mut nanos) = (micros / MICROS_PER_SEC, (micros % MICROS_PER_SEC) * 1_000);
        if nanos < 0 {
            secs -= 1;
            nanos += 1_000_000_000;
        }

        if secs > 253402207200 {
            secs = 253402207200;
            nanos = 0;
        } else if secs < -377705023201 {
            secs = -377705023201;
            nanos = 0;
        }
        let ts = jiff::Timestamp::new(secs, nanos as i32).unwrap();
        let tz_offset = Offset::from_seconds(self.offset).expect("invalid timezone offset seconds");
        let tz = tz_offset.to_time_zone();
        let zoned = ts.to_zoned(tz);

        write!(
            f,
            "{}",
            strtime::format(TIMESTAMP_TIMEZONE_FORMAT, &zoned).unwrap()
        )
    }
}

impl Display for Interval {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let mut wrote_date_part = false;
        let years = self.months / MONTHS_PER_YEAR;
        let months = self.months % MONTHS_PER_YEAR;

        let mut write_component = |value: i32, singular: &str, plural: &str| -> std::fmt::Result {
            if value != 0 {
                if wrote_date_part {
                    write!(f, " ")?;
                }
                let abs_val = value.abs();
                let unit = if abs_val == 1 { singular } else { plural };
                if value < 0 {
                    write!(f, "-{} {}", abs_val, unit)?;
                } else {
                    write!(f, "{} {}", abs_val, unit)?;
                }
                wrote_date_part = true;
            }
            Ok(())
        };

        write_component(years, "year", "years")?;
        write_component(months, "month", "months")?;
        write_component(self.days, "day", "days")?;

        if self.micros != 0 {
            if wrote_date_part {
                write!(f, " ")?;
            }
            let mut micros = self.micros;
            if micros < 0 {
                write!(f, "-")?;
                micros = -micros;
            }
            let hour = micros / MICROS_PER_HOUR;
            micros -= hour * MICROS_PER_HOUR;
            let min = micros / MICROS_PER_MINUTE;
            micros -= min * MICROS_PER_MINUTE;
            let sec = micros / MICROS_PER_SEC;
            micros -= sec * MICROS_PER_SEC;

            if hour < 100 {
                write!(f, "{:02}:{:02}:{:02}", hour, min, sec)?;
            } else {
                write!(f, "{}:{:02}:{:02}", hour, min, sec)?;
            }
            if micros != 0 {
                write!(f, ".{:06}", micros)?;
            }
        } else if !wrote_date_part {
            write!(f, "00:00:00")?;
        }
        Ok(())
    }
}

impl Display for ExtensionValue<'_> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            ExtensionValue::Binary(v) => {
                for c in *v {
                    write!(f, "{c:02X}")?;
                }
                Ok(())
            }
            ExtensionValue::Date(v) => write!(f, "{}", v),
            ExtensionValue::Timestamp(v) => write!(f, "{}", v),
            ExtensionValue::TimestampTz(v) => write!(f, "{}", v),
            ExtensionValue::Interval(v) => write!(f, "{}", v),
        }
    }
}

impl Eq for ExtensionValue<'_> {}

impl PartialEq for ExtensionValue<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for ExtensionValue<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let self_level = match self {
            ExtensionValue::Binary(_) => 0,
            ExtensionValue::Date(_) => 1,
            ExtensionValue::Timestamp(_) => 2,
            ExtensionValue::TimestampTz(_) => 3,
            ExtensionValue::Interval(_) => 4,
        };
        let other_level = match other {
            ExtensionValue::Binary(_) => 0,
            ExtensionValue::Date(_) => 1,
            ExtensionValue::Timestamp(_) => 2,
            ExtensionValue::TimestampTz(_) => 3,
            ExtensionValue::Interval(_) => 4,
        };
        let res = self_level.cmp(&other_level);
        if matches!(res, Ordering::Greater | Ordering::Less) {
            return Some(res);
        }

        match (self, other) {
            (ExtensionValue::Binary(self_data), ExtensionValue::Binary(other_data)) => {
                Some(self_data.cmp(other_data))
            }
            (ExtensionValue::Date(self_data), ExtensionValue::Date(other_data)) => {
                Some(self_data.cmp(other_data))
            }
            (ExtensionValue::Timestamp(self_data), ExtensionValue::Timestamp(other_data)) => {
                Some(self_data.cmp(other_data))
            }
            (ExtensionValue::TimestampTz(self_data), ExtensionValue::TimestampTz(other_data)) => {
                Some(self_data.cmp(other_data))
            }
            (ExtensionValue::Interval(self_data), ExtensionValue::Interval(other_data)) => {
                Some(self_data.cmp(other_data))
            }
            (_, _) => None,
        }
    }
}
