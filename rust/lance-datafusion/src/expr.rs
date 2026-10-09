// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for working with datafusion expressions

use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::{ArrayRef, cast::AsArray};
use arrow_schema::{DataType, TimeUnit};
use datafusion_common::ScalarValue;
use half::f16;

const MS_PER_DAY: i64 = 86400000;

/// The exact tie between the largest finite `f16` and the value that would
/// follow it. A tie goes to the even mantissa, which there is the one that does
/// not exist, so this magnitude and anything above it rounds to an infinity.
const F16_OVERFLOW_THRESHOLD: f64 = 65520.0;

/// The finite `f16` nearest `value`, with a tie going to the even mantissa.
///
/// Neither of `half`'s conversions is correctly rounded, in two different ways:
/// the software path truncates the low 32 bits of the `f64` mantissa before it
/// rounds ([half-rs#151]), and the x86 hardware path rounds through `f32` first
/// ([half-rs#116]). Both land within one step of the right answer, so this starts
/// from the software path, which at least does not vary by target, and then looks
/// at the two neighbours. Widening `f16` to `f64` is exact, so comparing the three
/// distances in `f64` decides exactly.
///
/// The caller must have excluded the overflow range first. Truncation only ever
/// shrinks the magnitude, so below the threshold the starting point is finite.
///
/// [half-rs#151]: https://github.com/VoidStarKat/half-rs/issues/151
/// [half-rs#116]: https://github.com/VoidStarKat/half-rs/issues/116
fn nearest_finite_f16(value: f64) -> f16 {
    let start = f16::from_f64_const(value);
    debug_assert!(
        start.is_finite(),
        "caller must reject the overflow range before calling: {value}"
    );
    let mut best = start;
    let mut best_distance = (value - start.to_f64()).abs();
    let start_bits = start.to_bits();
    // Stepping the bit pattern by one walks to the adjacent magnitude on either
    // side of `start`, for negatives as well. Stepping off an end lands on an
    // infinity or a NaN, which is not a candidate.
    for candidate in [start_bits.wrapping_sub(1), start_bits.wrapping_add(1)].map(f16::from_bits) {
        if !candidate.is_finite() {
            continue;
        }
        let distance = (value - candidate.to_f64()).abs();
        let wins =
            distance < best_distance || (distance == best_distance && candidate.to_bits() & 1 == 0);
        if wins {
            best = candidate;
            best_distance = distance;
        }
    }
    best
}

/// Coerce a float to `f16`, rejecting a finite value that leaves the `f16` range
/// rather than saturating it to an infinity.
///
/// A value inside the range lands on the `f16` grid and is inexact in a way that
/// shows: past 2048 the grid is coarser than the integers, so `= 2049` matches
/// rows holding 2048. The `Float32` and `Float64` arms below are inexact too, on
/// a finer grid, and that is not what the rejection is for.
///
/// The rejection is for the literal changing kind. Saturated to an infinity,
/// `= 100000` matches rows that really hold infinity; collapsed to zero,
/// `= 1e-30` matches real zeros. The cost is the queries where saturating would
/// have been right: `< 100000` errors instead of returning every finite row.
/// This function cannot see the operator, so it cannot allow saturation only
/// where it is harmless, and an error the caller reports beats a wrong row set.
/// An infinite or NaN input converts faithfully and is kept.
///
/// `Float64` to `Float32` still saturates. Reaching that takes a literal above
/// 1e38, while `f16` overflows at 65520, which an ordinary literal passes.
fn coerce_to_f16(value: f64) -> Option<f16> {
    if value.is_nan() {
        return Some(f16::NAN);
    }
    if value.is_infinite() {
        return Some(if value.is_sign_positive() {
            f16::INFINITY
        } else {
            f16::NEG_INFINITY
        });
    }
    if value.abs() >= F16_OVERFLOW_THRESHOLD {
        return None;
    }
    let nearest = nearest_finite_f16(value);
    // `f16::ZERO == f16::NEG_ZERO`, and so does `-0.0 == 0.0`, so a signed zero
    // literal keeps its sign here and only a genuinely nonzero value is rejected.
    if nearest == f16::ZERO && value != 0.0 {
        return None;
    }
    Some(nearest)
}

// This is slightly tedious but when we convert expressions from SQL strings to logical
// datafusion expressions there is no type coercion that happens.  In other words "x = 7"
// will always yield "x = 7_u64" regardless of the type of the column "x".  As a result, we
// need to do that literal coercion ourselves.
pub fn safe_coerce_scalar(value: &ScalarValue, ty: &DataType) -> Option<ScalarValue> {
    // A dictionary target coerces the value to the dictionary's value type and
    // re-wraps it as a dictionary literal. Only an untyped `ScalarValue::Null`
    // keeps its untyped form, matching the behavior for all other targets; a
    // *typed* null (e.g. `Utf8(None)`) is coerced and wrapped like any other
    // value so it produces a `Dictionary(..)` literal that matches the column.
    if let DataType::Dictionary(key_type, value_type) = ty {
        if matches!(value, ScalarValue::Null) {
            return Some(value.clone());
        }
        let inner = safe_coerce_scalar(value, value_type)?;
        return Some(ScalarValue::Dictionary(key_type.clone(), Box::new(inner)));
    }
    match value {
        ScalarValue::Int8(val) => match ty {
            DataType::Int8 => Some(value.clone()),
            DataType::Int16 => val.map(|v| ScalarValue::Int16(Some(i16::from(v)))),
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(i32::from(v)))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            DataType::Float16 => {
                val.and_then(|v| coerce_to_f16(v as f64).map(|v| ScalarValue::Float16(Some(v))))
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::Int16(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => Some(value.clone()),
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(i32::from(v)))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            DataType::Float16 => {
                val.and_then(|v| coerce_to_f16(v as f64).map(|v| ScalarValue::Float16(Some(v))))
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::Int32(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => Some(value.clone()),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            // These conversions are inherently lossy as the full range of i32 cannot
            // be represented in f32.  However, there is no f32::TryFrom(i32) and its not
            // clear users would want that anyways
            DataType::Float16 => {
                val.and_then(|v| coerce_to_f16(v as f64).map(|v| ScalarValue::Float16(Some(v))))
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::Int64(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => {
                val.and_then(|v| i32::try_from(v).map(|v| ScalarValue::Int32(Some(v))).ok())
            }
            DataType::Int64 => Some(value.clone()),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            // See above warning about lossy float conversion
            DataType::Float16 => {
                val.and_then(|v| coerce_to_f16(v as f64).map(|v| ScalarValue::Float16(Some(v))))
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            DataType::Decimal128(_, _) | DataType::Decimal256(_, _) => value.cast_to(ty).ok(),
            DataType::Time32(TimeUnit::Second) => val.and_then(|v| {
                i32::try_from(v)
                    .ok()
                    .map(|v| ScalarValue::Time32Second(Some(v)))
            }),
            DataType::Time32(TimeUnit::Millisecond) => val.and_then(|v| {
                i32::try_from(v)
                    .ok()
                    .map(|v| ScalarValue::Time32Millisecond(Some(v)))
            }),
            _ => None,
        },
        ScalarValue::UInt8(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => val.map(|v| ScalarValue::Int16(Some(v.into()))),
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(v.into()))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(v.into()))),
            DataType::UInt8 => Some(value.clone()),
            DataType::UInt16 => val.map(|v| ScalarValue::UInt16(Some(u16::from(v)))),
            DataType::UInt32 => val.map(|v| ScalarValue::UInt32(Some(u32::from(v)))),
            DataType::UInt64 => val.map(|v| ScalarValue::UInt64(Some(u64::from(v)))),
            DataType::Float16 => {
                val.and_then(|v| coerce_to_f16(v as f64).map(|v| ScalarValue::Float16(Some(v))))
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::UInt16(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(v.into()))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(v.into()))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => Some(value.clone()),
            DataType::UInt32 => val.map(|v| ScalarValue::UInt32(Some(u32::from(v)))),
            DataType::UInt64 => val.map(|v| ScalarValue::UInt64(Some(u64::from(v)))),
            DataType::Float16 => {
                val.and_then(|v| coerce_to_f16(v as f64).map(|v| ScalarValue::Float16(Some(v))))
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::UInt32(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => {
                val.and_then(|v| i32::try_from(v).map(|v| ScalarValue::Int32(Some(v))).ok())
            }
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(v.into()))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => Some(value.clone()),
            DataType::UInt64 => val.map(|v| ScalarValue::UInt64(Some(u64::from(v)))),
            // See above warning about lossy float conversion
            DataType::Float16 => {
                val.and_then(|v| coerce_to_f16(v as f64).map(|v| ScalarValue::Float16(Some(v))))
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::UInt64(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => {
                val.and_then(|v| i32::try_from(v).map(|v| ScalarValue::Int32(Some(v))).ok())
            }
            DataType::Int64 => {
                val.and_then(|v| i64::try_from(v).map(|v| ScalarValue::Int64(Some(v))).ok())
            }
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => Some(value.clone()),
            // See above warning about lossy float conversion
            DataType::Float16 => {
                val.and_then(|v| coerce_to_f16(v as f64).map(|v| ScalarValue::Float16(Some(v))))
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::Float16(val) => match ty {
            DataType::Float16 => Some(value.clone()),
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v.to_f32()))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v.to_f64()))),
            _ => None,
        },
        ScalarValue::Float32(val) => match ty {
            DataType::Float16 => {
                val.and_then(|v| coerce_to_f16(f64::from(v)).map(|v| ScalarValue::Float16(Some(v))))
            }
            DataType::Float32 => Some(value.clone()),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::Float64(val) => match ty {
            DataType::Float16 => {
                val.and_then(|v| coerce_to_f16(v).map(|v| ScalarValue::Float16(Some(v))))
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Utf8(val) => match ty {
            DataType::Utf8 => Some(value.clone()),
            DataType::LargeUtf8 => Some(ScalarValue::LargeUtf8(val.clone())),
            DataType::Utf8View => Some(ScalarValue::Utf8View(val.clone())),
            _ => None,
        },
        ScalarValue::LargeUtf8(val) => match ty {
            DataType::Utf8 => Some(ScalarValue::Utf8(val.clone())),
            DataType::LargeUtf8 => Some(value.clone()),
            DataType::Utf8View => Some(ScalarValue::Utf8View(val.clone())),
            _ => None,
        },
        ScalarValue::Utf8View(val) => match ty {
            DataType::Utf8 => Some(ScalarValue::Utf8(val.clone())),
            DataType::LargeUtf8 => Some(ScalarValue::LargeUtf8(val.clone())),
            DataType::Utf8View => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Boolean(_) => match ty {
            DataType::Boolean => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Null => Some(value.clone()),
        ScalarValue::List(values) => {
            let values = values.clone() as ArrayRef;
            let new_values = cast(&values, ty).ok()?;
            match ty {
                DataType::List(_) => {
                    Some(ScalarValue::List(Arc::new(new_values.as_list().clone())))
                }
                DataType::LargeList(_) => Some(ScalarValue::LargeList(Arc::new(
                    new_values.as_list().clone(),
                ))),
                DataType::FixedSizeList(_, _) => Some(ScalarValue::FixedSizeList(Arc::new(
                    new_values.as_fixed_size_list().clone(),
                ))),
                _ => None,
            }
        }
        ScalarValue::TimestampSecond(seconds, _) => match ty {
            DataType::Timestamp(TimeUnit::Second, _) => Some(value.clone()),
            DataType::Timestamp(TimeUnit::Millisecond, tz) => seconds
                .and_then(|v| v.checked_mul(1000))
                .map(|val| ScalarValue::TimestampMillisecond(Some(val), tz.clone())),
            DataType::Timestamp(TimeUnit::Microsecond, tz) => seconds
                .and_then(|v| v.checked_mul(1000000))
                .map(|val| ScalarValue::TimestampMicrosecond(Some(val), tz.clone())),
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => seconds
                .and_then(|v| v.checked_mul(1000000000))
                .map(|val| ScalarValue::TimestampNanosecond(Some(val), tz.clone())),
            _ => None,
        },
        ScalarValue::TimestampMillisecond(millis, _) => match ty {
            DataType::Timestamp(TimeUnit::Second, tz) => {
                millis.map(|val| ScalarValue::TimestampSecond(Some(val / 1000), tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => Some(value.clone()),
            DataType::Timestamp(TimeUnit::Microsecond, tz) => millis
                .and_then(|v| v.checked_mul(1000))
                .map(|val| ScalarValue::TimestampMicrosecond(Some(val), tz.clone())),
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => millis
                .and_then(|v| v.checked_mul(1000000))
                .map(|val| ScalarValue::TimestampNanosecond(Some(val), tz.clone())),
            _ => None,
        },
        ScalarValue::TimestampMicrosecond(micros, _) => match ty {
            DataType::Timestamp(TimeUnit::Second, tz) => {
                micros.map(|val| ScalarValue::TimestampSecond(Some(val / 1000000), tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Millisecond, tz) => {
                micros.map(|val| ScalarValue::TimestampMillisecond(Some(val / 1000), tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => Some(value.clone()),
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => micros
                .and_then(|v| v.checked_mul(1000))
                .map(|val| ScalarValue::TimestampNanosecond(Some(val), tz.clone())),
            _ => None,
        },
        ScalarValue::TimestampNanosecond(nanos, _) => {
            match ty {
                DataType::Timestamp(TimeUnit::Second, tz) => nanos
                    .map(|val| ScalarValue::TimestampSecond(Some(val / 1000000000), tz.clone())),
                DataType::Timestamp(TimeUnit::Millisecond, tz) => nanos
                    .map(|val| ScalarValue::TimestampMillisecond(Some(val / 1000000), tz.clone())),
                DataType::Timestamp(TimeUnit::Microsecond, tz) => {
                    nanos.map(|val| ScalarValue::TimestampMicrosecond(Some(val / 1000), tz.clone()))
                }
                DataType::Timestamp(TimeUnit::Nanosecond, _) => Some(value.clone()),
                _ => None,
            }
        }
        ScalarValue::Date32(ticks) => match ty {
            DataType::Date32 => Some(value.clone()),
            DataType::Date64 => Some(ScalarValue::Date64(
                ticks.map(|v| i64::from(v) * MS_PER_DAY),
            )),
            _ => None,
        },
        ScalarValue::Date64(ticks) => match ty {
            DataType::Date32 => Some(ScalarValue::Date32(ticks.map(|v| (v / MS_PER_DAY) as i32))),
            DataType::Date64 => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Time32Second(seconds) => {
            match ty {
                DataType::Time32(TimeUnit::Second) => Some(value.clone()),
                DataType::Time32(TimeUnit::Millisecond) => {
                    seconds.map(|val| ScalarValue::Time32Millisecond(Some(val * 1000)))
                }
                DataType::Time64(TimeUnit::Microsecond) => seconds
                    .map(|val| ScalarValue::Time64Microsecond(Some(i64::from(val) * 1000000))),
                DataType::Time64(TimeUnit::Nanosecond) => seconds
                    .map(|val| ScalarValue::Time64Nanosecond(Some(i64::from(val) * 1000000000))),
                _ => None,
            }
        }
        ScalarValue::Time32Millisecond(millis) => match ty {
            DataType::Time32(TimeUnit::Second) => {
                millis.map(|val| ScalarValue::Time32Second(Some(val / 1000)))
            }
            DataType::Time32(TimeUnit::Millisecond) => Some(value.clone()),
            DataType::Time64(TimeUnit::Microsecond) => {
                millis.map(|val| ScalarValue::Time64Microsecond(Some(i64::from(val) * 1000)))
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                millis.map(|val| ScalarValue::Time64Nanosecond(Some(i64::from(val) * 1000000)))
            }
            _ => None,
        },
        ScalarValue::Time64Microsecond(micros) => match ty {
            DataType::Time32(TimeUnit::Second) => {
                micros.map(|val| ScalarValue::Time32Second(Some((val / 1000000) as i32)))
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                micros.map(|val| ScalarValue::Time32Millisecond(Some((val / 1000) as i32)))
            }
            DataType::Time64(TimeUnit::Microsecond) => Some(value.clone()),
            DataType::Time64(TimeUnit::Nanosecond) => {
                micros.map(|val| ScalarValue::Time64Nanosecond(Some(val * 1000)))
            }
            _ => None,
        },
        ScalarValue::Time64Nanosecond(nanos) => match ty {
            DataType::Time32(TimeUnit::Second) => {
                nanos.map(|val| ScalarValue::Time32Second(Some((val / 1000000000) as i32)))
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                nanos.map(|val| ScalarValue::Time32Millisecond(Some((val / 1000000) as i32)))
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                nanos.map(|val| ScalarValue::Time64Microsecond(Some(val / 1000)))
            }
            DataType::Time64(TimeUnit::Nanosecond) => Some(value.clone()),
            _ => None,
        },
        ScalarValue::LargeList(values) => {
            let values = values.clone() as ArrayRef;
            let new_values = cast(&values, ty).ok()?;
            match ty {
                DataType::List(_) => {
                    Some(ScalarValue::List(Arc::new(new_values.as_list().clone())))
                }
                DataType::LargeList(_) => Some(ScalarValue::LargeList(Arc::new(
                    new_values.as_list().clone(),
                ))),
                DataType::FixedSizeList(_, _) => Some(ScalarValue::FixedSizeList(Arc::new(
                    new_values.as_fixed_size_list().clone(),
                ))),
                _ => None,
            }
        }
        ScalarValue::FixedSizeList(values) => {
            let values = values.clone() as ArrayRef;
            let new_values = cast(&values, ty).ok()?;
            match ty {
                DataType::List(_) => {
                    Some(ScalarValue::List(Arc::new(new_values.as_list().clone())))
                }
                DataType::LargeList(_) => Some(ScalarValue::LargeList(Arc::new(
                    new_values.as_list().clone(),
                ))),
                DataType::FixedSizeList(_, _) => Some(ScalarValue::FixedSizeList(Arc::new(
                    new_values.as_fixed_size_list().clone(),
                ))),
                _ => None,
            }
        }
        ScalarValue::FixedSizeBinary(len, value) => match ty {
            DataType::FixedSizeBinary(len2) => {
                if len == len2 {
                    Some(ScalarValue::FixedSizeBinary(*len, value.clone()))
                } else {
                    None
                }
            }
            DataType::Binary => Some(ScalarValue::Binary(value.clone())),
            _ => None,
        },
        ScalarValue::Binary(value) => match ty {
            DataType::Binary => Some(ScalarValue::Binary(value.clone())),
            DataType::LargeBinary => Some(ScalarValue::LargeBinary(value.clone())),
            DataType::BinaryView => Some(ScalarValue::BinaryView(value.clone())),
            DataType::FixedSizeBinary(len) => {
                if let Some(value) = value {
                    if value.len() == *len as usize {
                        Some(ScalarValue::FixedSizeBinary(*len, Some(value.clone())))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        },
        ScalarValue::BinaryView(val) => match ty {
            DataType::Binary => Some(ScalarValue::Binary(val.clone())),
            DataType::LargeBinary => Some(ScalarValue::LargeBinary(val.clone())),
            DataType::BinaryView => Some(value.clone()),
            _ => None,
        },
        ScalarValue::LargeBinary(_) => match ty {
            DataType::LargeBinary => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Decimal128(_, _, _) => match ty {
            DataType::Decimal128(_, _) => value.cast_to(ty).ok(),
            _ => None,
        },
        ScalarValue::Decimal256(_, _, _) => match ty {
            DataType::Decimal256(_, _) => value.cast_to(ty).ok(),
            _ => None,
        },
        ScalarValue::DurationSecond(_)
        | ScalarValue::DurationMillisecond(_)
        | ScalarValue::DurationMicrosecond(_)
        | ScalarValue::DurationNanosecond(_) => match ty {
            DataType::Duration(_) => value.cast_to(ty).ok(),
            _ => None,
        },
        // A dictionary-encoded literal (e.g. produced by DataFusion's dictionary
        // cast in the scalar-index path) coerces by unwrapping its underlying value.
        ScalarValue::Dictionary(_, inner) => safe_coerce_scalar(inner, ty),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::i256;

    use super::*;

    #[test]
    fn test_temporal_coerce() {
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Int64(Some(5)),
                &DataType::Time32(TimeUnit::Second),
            ),
            Some(ScalarValue::Time32Second(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Int64(Some(5000)),
                &DataType::Time32(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::Time32Millisecond(Some(5000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Int64(Some(i64::MAX)),
                &DataType::Time32(TimeUnit::Second),
            ),
            None
        );

        // Conversion from timestamps in one resolution to timestamps in another resolution is allowed
        // s->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampSecond(Some(5), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // s->ms
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampSecond(Some(5), None),
                &DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            Some(ScalarValue::TimestampMillisecond(Some(5000), None))
        );
        // s->us
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampSecond(Some(5), None),
                &DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            Some(ScalarValue::TimestampMicrosecond(Some(5000000), None))
        );
        // s->ns
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampSecond(Some(5), None),
                &DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            Some(ScalarValue::TimestampNanosecond(Some(5000000000), None))
        );
        // ms->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMillisecond(Some(5000), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // ms->ms
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMillisecond(Some(5000), None),
                &DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            Some(ScalarValue::TimestampMillisecond(Some(5000), None))
        );
        // ms->us
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMillisecond(Some(5000), None),
                &DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            Some(ScalarValue::TimestampMicrosecond(Some(5000000), None))
        );
        // ms->ns
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMillisecond(Some(5000), None),
                &DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            Some(ScalarValue::TimestampNanosecond(Some(5000000000), None))
        );
        // us->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMicrosecond(Some(5000000), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // us->ms
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMicrosecond(Some(5000000), None),
                &DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            Some(ScalarValue::TimestampMillisecond(Some(5000), None))
        );
        // us->us
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMicrosecond(Some(5000000), None),
                &DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            Some(ScalarValue::TimestampMicrosecond(Some(5000000), None))
        );
        // us->ns
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMicrosecond(Some(5000000), None),
                &DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            Some(ScalarValue::TimestampNanosecond(Some(5000000000), None))
        );
        // ns->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5000000000), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // ns->ms
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5000000000), None),
                &DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            Some(ScalarValue::TimestampMillisecond(Some(5000), None))
        );
        // ns->us
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5000000000), None),
                &DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            Some(ScalarValue::TimestampMicrosecond(Some(5000000), None))
        );
        // ns->ns
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5000000000), None),
                &DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            Some(ScalarValue::TimestampNanosecond(Some(5000000000), None))
        );
        // Precision loss on coercion is allowed (truncation)
        // ns->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5987654321), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // Conversions from date-32 to date-64 is allowed
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Date32(Some(5)), &DataType::Date32,),
            Some(ScalarValue::Date32(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Date32(Some(5)), &DataType::Date64,),
            Some(ScalarValue::Date64(Some(5 * MS_PER_DAY)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Date64(Some(5 * MS_PER_DAY)),
                &DataType::Date32,
            ),
            Some(ScalarValue::Date32(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Date64(Some(5)), &DataType::Date64,),
            Some(ScalarValue::Date64(Some(5)))
        );
        // Time-32 to time-64 (and within time-32 and time-64) is allowed
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Second(Some(5)),
                &DataType::Time32(TimeUnit::Second),
            ),
            Some(ScalarValue::Time32Second(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Second(Some(5)),
                &DataType::Time32(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::Time32Millisecond(Some(5000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Second(Some(5)),
                &DataType::Time64(TimeUnit::Microsecond),
            ),
            Some(ScalarValue::Time64Microsecond(Some(5000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Second(Some(5)),
                &DataType::Time64(TimeUnit::Nanosecond),
            ),
            Some(ScalarValue::Time64Nanosecond(Some(5000000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Millisecond(Some(5000)),
                &DataType::Time32(TimeUnit::Second),
            ),
            Some(ScalarValue::Time32Second(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Millisecond(Some(5000)),
                &DataType::Time32(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::Time32Millisecond(Some(5000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Millisecond(Some(5000)),
                &DataType::Time64(TimeUnit::Microsecond),
            ),
            Some(ScalarValue::Time64Microsecond(Some(5000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Millisecond(Some(5000)),
                &DataType::Time64(TimeUnit::Nanosecond),
            ),
            Some(ScalarValue::Time64Nanosecond(Some(5000000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Microsecond(Some(5000000)),
                &DataType::Time32(TimeUnit::Second),
            ),
            Some(ScalarValue::Time32Second(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Microsecond(Some(5000000)),
                &DataType::Time32(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::Time32Millisecond(Some(5000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Microsecond(Some(5000000)),
                &DataType::Time64(TimeUnit::Microsecond),
            ),
            Some(ScalarValue::Time64Microsecond(Some(5000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Microsecond(Some(5000000)),
                &DataType::Time64(TimeUnit::Nanosecond),
            ),
            Some(ScalarValue::Time64Nanosecond(Some(5000000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Nanosecond(Some(5000000000)),
                &DataType::Time32(TimeUnit::Second),
            ),
            Some(ScalarValue::Time32Second(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Nanosecond(Some(5000000000)),
                &DataType::Time32(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::Time32Millisecond(Some(5000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Nanosecond(Some(5000000000)),
                &DataType::Time64(TimeUnit::Microsecond),
            ),
            Some(ScalarValue::Time64Microsecond(Some(5000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Nanosecond(Some(5000000000)),
                &DataType::Time64(TimeUnit::Nanosecond),
            ),
            Some(ScalarValue::Time64Nanosecond(Some(5000000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::DurationNanosecond(Some(2_000_000)),
                &DataType::Duration(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::DurationMillisecond(Some(2)))
        );
    }

    #[test]
    fn test_string_view_coerce() {
        // Utf8 <-> Utf8View
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Utf8(Some("hi".into())), &DataType::Utf8View),
            Some(ScalarValue::Utf8View(Some("hi".into())))
        );
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Utf8View(Some("hi".into())), &DataType::Utf8),
            Some(ScalarValue::Utf8(Some("hi".into())))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Utf8View(Some("hi".into())),
                &DataType::LargeUtf8
            ),
            Some(ScalarValue::LargeUtf8(Some("hi".into())))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::LargeUtf8(Some("hi".into())),
                &DataType::Utf8View
            ),
            Some(ScalarValue::Utf8View(Some("hi".into())))
        );
        // identity
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Utf8View(Some("hi".into())),
                &DataType::Utf8View
            ),
            Some(ScalarValue::Utf8View(Some("hi".into())))
        );
        // Binary <-> BinaryView
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Binary(Some(vec![1, 2, 3])),
                &DataType::BinaryView
            ),
            Some(ScalarValue::BinaryView(Some(vec![1, 2, 3])))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::BinaryView(Some(vec![1, 2, 3])),
                &DataType::Binary
            ),
            Some(ScalarValue::Binary(Some(vec![1, 2, 3])))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::BinaryView(Some(vec![1, 2, 3])),
                &DataType::BinaryView
            ),
            Some(ScalarValue::BinaryView(Some(vec![1, 2, 3])))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::LargeBinary(Some(vec![1, 2, 3])),
                &DataType::LargeBinary
            ),
            Some(ScalarValue::LargeBinary(Some(vec![1, 2, 3])))
        );
    }

    /// Every numeric literal type reaches `Float16`. SQL only produces `Int64`
    /// and `Float64`, but `safe_coerce_scalar` is public and the index layer
    /// feeds it whatever the planner already coerced, including `Float16`
    /// itself.
    #[rstest::rstest]
    #[case::int8(ScalarValue::Int8(Some(-2)))]
    #[case::int16(ScalarValue::Int16(Some(-2)))]
    #[case::int32(ScalarValue::Int32(Some(-2)))]
    #[case::int64(ScalarValue::Int64(Some(-2)))]
    #[case::float32(ScalarValue::Float32(Some(-2.0)))]
    #[case::float64(ScalarValue::Float64(Some(-2.0)))]
    #[case::float16(ScalarValue::Float16(Some(f16::from_f32(-2.0))))]
    fn numeric_literals_coerce_to_f16(#[case] value: ScalarValue) {
        assert_eq!(
            safe_coerce_scalar(&value, &DataType::Float16),
            Some(ScalarValue::Float16(Some(f16::from_f32(-2.0)))),
        );
    }

    #[rstest::rstest]
    #[case::uint8(ScalarValue::UInt8(Some(2)))]
    #[case::uint16(ScalarValue::UInt16(Some(2)))]
    #[case::uint32(ScalarValue::UInt32(Some(2)))]
    #[case::uint64(ScalarValue::UInt64(Some(2)))]
    fn unsigned_literals_coerce_to_f16(#[case] value: ScalarValue) {
        assert_eq!(
            safe_coerce_scalar(&value, &DataType::Float16),
            Some(ScalarValue::Float16(Some(f16::from_f32(2.0)))),
        );
    }

    /// A `Float16` literal also has to reach the wider float columns, so a
    /// predicate written against one column type still filters another.
    #[test]
    fn test_f16_literal_widens() {
        let half = ScalarValue::Float16(Some(f16::from_f32(0.5)));
        assert_eq!(
            safe_coerce_scalar(&half, &DataType::Float32),
            Some(ScalarValue::Float32(Some(0.5))),
        );
        assert_eq!(
            safe_coerce_scalar(&half, &DataType::Float64),
            Some(ScalarValue::Float64(Some(0.5))),
        );
        assert_eq!(safe_coerce_scalar(&half, &DataType::Int32), None);
    }

    /// Rounding inside the `f16` range is accepted; leaving the range is not.
    /// Expected values are spelled as bits so a change in how the input is
    /// rounded fails here rather than being recomputed by the same library call
    /// the code under test uses.
    #[rstest::rstest]
    // 0.1 has no exact binary form, so it lands on the nearest f16, 0x2E66.
    #[case::rounds(0.1, Some(0x2E66))]
    // Past 2048 the f16 grid is coarser than the integers: 2049 is the exact
    // midpoint of 2048 and 2050, and the tie goes to the even mantissa. This is
    // the case the doc comment cites for `= 2049` matching rows holding 2048.
    #[case::odd_integer_ties_down(2049.0, Some(0x6800))]
    // Above that midpoint 2050 is the nearer neighbour, and the coercion picks it.
    // `half`'s own conversion returns 2048 here, which is the defect
    // `nearest_finite_f16` exists to correct.
    #[case::just_above_a_tie(2049.001, Some(0x6801))]
    #[case::largest_finite(65504.0, Some(0x7BFF))]
    // Rounds down to the largest finite f16 rather than overflowing.
    #[case::just_under_overflow(65519.0, Some(0x7BFF))]
    #[case::smallest_subnormal(6e-8, Some(0x0001))]
    // 65520 is the first value that rounds to infinity, not 65504.
    #[case::overflow_threshold(65520.0, None)]
    #[case::overflow(70000.0, None)]
    #[case::negative_overflow(-70000.0, None)]
    #[case::f32_max(f32::MAX as f64, None)]
    // Underflows to zero rather than silently matching real zeros.
    #[case::underflow(1e-30, None)]
    #[case::negative_underflow(-1e-30, None)]
    fn test_f16_range_edges(#[case] input: f64, #[case] expected: Option<u16>) {
        let coerced = safe_coerce_scalar(&ScalarValue::Float64(Some(input)), &DataType::Float16);
        match expected {
            Some(bits) => assert_eq!(
                coerced,
                Some(ScalarValue::Float16(Some(f16::from_bits(bits)))),
            ),
            None => assert_eq!(coerced, None),
        }
    }

    /// A literal that is already infinite or NaN converts faithfully. Only a
    /// finite value that overflowed is rejected.
    #[test]
    fn test_f16_keeps_non_finite_literals() {
        for (input, expected) in [
            (f64::INFINITY, f16::INFINITY),
            (f64::NEG_INFINITY, f16::NEG_INFINITY),
        ] {
            assert_eq!(
                safe_coerce_scalar(&ScalarValue::Float64(Some(input)), &DataType::Float16),
                Some(ScalarValue::Float16(Some(expected))),
            );
        }
        let nan = safe_coerce_scalar(&ScalarValue::Float64(Some(f64::NAN)), &DataType::Float16);
        assert!(matches!(nan, Some(ScalarValue::Float16(Some(v))) if v.is_nan()));
    }

    /// Both zeros survive as themselves. Signed-zero comparison semantics are a
    /// separate problem (#5868); coercion must at least not erase the sign.
    #[test]
    fn test_f16_keeps_zero_sign() {
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Float64(Some(-0.0)), &DataType::Float16),
            Some(ScalarValue::Float16(Some(f16::NEG_ZERO))),
        );
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Float64(Some(0.0)), &DataType::Float16),
            Some(ScalarValue::Float16(Some(f16::ZERO))),
        );
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Int64(Some(0)), &DataType::Float16),
            Some(ScalarValue::Float16(Some(f16::ZERO))),
        );
    }

    /// Sweep every rounding decision the conversion can make instead of trusting
    /// the handful of points named above: for each adjacent pair of finite `f16`
    /// values, the exact midpoint and the two `f64` values either side of it. A
    /// misrounding anywhere in the range shows up here, which is how the
    /// `2049.001` case was found in the first place.
    ///
    /// The expectation is stated, not recomputed: below the midpoint the lower
    /// neighbour, above it the upper one, at it the even mantissa.
    #[test]
    fn test_f16_rounds_to_nearest_even_across_the_whole_range() {
        fn coerce(value: f64) -> Option<f16> {
            match safe_coerce_scalar(&ScalarValue::Float64(Some(value)), &DataType::Float16) {
                Some(ScalarValue::Float16(Some(v))) => Some(v),
                // Rejected, which the underflow side of the range expects.
                None => None,
                other => panic!("expected a Float16 literal for {value}, got {other:?}"),
            }
        }
        // A nonzero literal that lands on zero is rejected rather than coerced,
        // so the smallest pair expects `None` on its lower side.
        fn want(expected: f16, input: f64) -> Option<f16> {
            if expected == f16::ZERO && input != 0.0 {
                None
            } else {
                Some(expected)
            }
        }

        // 0x7BFF is the largest finite f16, so pairing each bit pattern with the
        // next covers every adjacent finite pair on the positive side. Negatives
        // are covered by the symmetry check below.
        for lower_bits in 0..0x7BFFu16 {
            let lower = f16::from_bits(lower_bits);
            let upper = f16::from_bits(lower_bits + 1);
            // Both operands are f16 widened to f64, so the average is exact.
            let midpoint = (lower.to_f64() + upper.to_f64()) / 2.0;
            let below = f64::from_bits(midpoint.to_bits() - 1);
            let above = f64::from_bits(midpoint.to_bits() + 1);

            assert_eq!(coerce(below), want(lower, below), "just below {midpoint}");
            assert_eq!(coerce(above), want(upper, above), "just above {midpoint}");
            let even = if lower.to_bits() & 1 == 0 {
                lower
            } else {
                upper
            };
            assert_eq!(coerce(midpoint), want(even, midpoint), "at {midpoint}");
        }
    }

    /// Sign is not part of the rounding decision, so negating the input negates
    /// the result. This is what lets the sweep above cover only positives.
    #[rstest::rstest]
    #[case(0.1)]
    #[case(2049.001)]
    #[case(65504.0)]
    #[case(6e-8)]
    #[case(70000.0)]
    #[case(1e-30)]
    fn test_f16_coercion_is_sign_symmetric(#[case] magnitude: f64) {
        let positive =
            safe_coerce_scalar(&ScalarValue::Float64(Some(magnitude)), &DataType::Float16);
        let negative =
            safe_coerce_scalar(&ScalarValue::Float64(Some(-magnitude)), &DataType::Float16);
        let flipped = match positive {
            Some(ScalarValue::Float16(Some(v))) => Some(ScalarValue::Float16(Some(-v))),
            other => other,
        };
        assert_eq!(negative, flipped);
    }

    #[test]
    fn test_decimal_coerce() {
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Decimal128(Some(2), 10, 0),
                &DataType::Decimal128(12, 2),
            ),
            Some(ScalarValue::Decimal128(Some(200), 12, 2))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Decimal256(Some(i256::from_i128(2)), 76, 0),
                &DataType::Decimal256(76, 2),
            ),
            Some(ScalarValue::Decimal256(Some(i256::from_i128(200)), 76, 2))
        );
    }

    #[test]
    fn test_dictionary_coerce() {
        let dict_ty = DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8));

        // A string literal coerces to a dictionary target by wrapping the
        // coerced value in a dictionary scalar.
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Utf8(Some("com".to_string())), &dict_ty),
            Some(ScalarValue::Dictionary(
                Box::new(DataType::Int16),
                Box::new(ScalarValue::Utf8(Some("com".to_string()))),
            ))
        );

        // The inner value is coerced through to the dictionary value type, so a
        // LargeUtf8 literal lands as a Utf8 value inside the dictionary.
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::LargeUtf8(Some("com".to_string())), &dict_ty),
            Some(ScalarValue::Dictionary(
                Box::new(DataType::Int16),
                Box::new(ScalarValue::Utf8(Some("com".to_string()))),
            ))
        );

        // A dictionary literal round-trips back to its value type.
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Dictionary(
                    Box::new(DataType::Int16),
                    Box::new(ScalarValue::Utf8(Some("com".to_string()))),
                ),
                &DataType::Utf8,
            ),
            Some(ScalarValue::Utf8(Some("com".to_string())))
        );

        // A dictionary literal coerces to a dictionary target, adopting the
        // target's key type.
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Dictionary(
                    Box::new(DataType::Int32),
                    Box::new(ScalarValue::Utf8(Some("com".to_string()))),
                ),
                &dict_ty,
            ),
            Some(ScalarValue::Dictionary(
                Box::new(DataType::Int16),
                Box::new(ScalarValue::Utf8(Some("com".to_string()))),
            ))
        );

        // An untyped null keeps its untyped form for a dictionary target, just
        // like for every other target type.
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Null, &dict_ty),
            Some(ScalarValue::Null)
        );

        // A *typed* null (e.g. an API-built `Utf8(None)` literal, or an IN value
        // already typed as Utf8) is still wrapped in the dictionary type so it
        // matches the dictionary column. Returning a bare `Utf8(None)` here would
        // leave `resolve_value` with a literal whose type does not line up with
        // the column, breaking planning/evaluation the same way non-null strings
        // used to break.
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Utf8(None), &dict_ty),
            Some(ScalarValue::Dictionary(
                Box::new(DataType::Int16),
                Box::new(ScalarValue::Utf8(None)),
            ))
        );

        // The inner null is coerced through to the dictionary value type as well,
        // so a LargeUtf8 typed null lands as a Utf8 null inside the dictionary.
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::LargeUtf8(None), &dict_ty),
            Some(ScalarValue::Dictionary(
                Box::new(DataType::Int16),
                Box::new(ScalarValue::Utf8(None)),
            ))
        );

        // A value that cannot be coerced to the dictionary value type fails.
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Utf8(Some("com".to_string())),
                &DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Int32)),
            ),
            None
        );
    }
}
