// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exact comparisons of timestamp columns with operands in a finer unit.
//!
//! Comparing a `timestamp(s)` column with a `timestamp(us)` operand, as in
//! `ts >= TIMESTAMP '2024-01-01 00:00:00.5'`, coerces the column to
//! microseconds, which is exact. DataFusion's simplifier then moves that cast
//! onto the operand and truncates the operand to the column's unit, so the
//! comparison became `ts >= TIMESTAMP '2024-01-01 00:00:00'` and matched a row
//! half a second before the bound. [`unwrap_timestamp_widening_casts`] runs
//! before the simplifier and compares such a column in its own unit instead,
//! rounding the operand down and adjusting the operator where it falls between
//! two of the column's values.

use arrow_schema::{DataType, TimeUnit};
use datafusion::common::DFSchema;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::{Between, BinaryExpr, ExprSchemable, Operator};
use datafusion::prelude::{Expr, lit};
use datafusion::scalar::ScalarValue;
use lance_core::Result;

use crate::numeric_coercion::{Comparison, comparison_expr, extract_literal};

/// The operators whose operand is truncated when DataFusion unwraps a cast on
/// the other side: the comparison operators of its `supports_propagation`.
const COMPARISONS: [Operator; 8] = [
    Operator::Eq,
    Operator::NotEq,
    Operator::Lt,
    Operator::LtEq,
    Operator::Gt,
    Operator::GtEq,
    Operator::IsDistinctFrom,
    Operator::IsNotDistinctFrom,
];

/// A timestamp expression, usually a column, read through a cast to a finer
/// unit.
struct WidenedTimestamp<'a> {
    /// The expression the cast wraps. DataFusion unwraps the cast whatever it
    /// wraps, so this is not only a column or struct field.
    inner: &'a Expr,
    /// The type of `inner`.
    inner_type: DataType,
    /// The type the column is cast to, which the operands have.
    widened_type: DataType,
    /// Units of the widened type per unit of the column, such as 1000 from
    /// seconds to milliseconds.
    factor: i64,
}

impl WidenedTimestamp<'_> {
    /// A value of the column's type, `units` in the column's unit.
    fn scalar(&self, units: Option<i64>) -> Option<ScalarValue> {
        let DataType::Timestamp(unit, tz) = &self.inner_type else {
            return None;
        };
        let tz = tz.clone();
        Some(match unit {
            TimeUnit::Second => ScalarValue::TimestampSecond(units, tz),
            TimeUnit::Millisecond => ScalarValue::TimestampMillisecond(units, tz),
            TimeUnit::Microsecond => ScalarValue::TimestampMicrosecond(units, tz),
            TimeUnit::Nanosecond => ScalarValue::TimestampNanosecond(units, tz),
        })
    }

    /// The column's value equal to `value`, an operand in the widened unit, if
    /// there is one.
    fn exact(&self, value: i64) -> Option<i64> {
        (value.rem_euclid(self.factor) == 0).then(|| value.div_euclid(self.factor))
    }

    /// The largest value of the column below `value`, an operand in the
    /// widened unit that is not one of the column's values.
    fn floor(&self, value: i64) -> Option<ScalarValue> {
        self.scalar(Some(value.div_euclid(self.factor)))
    }
}

fn units_per_second(unit: &TimeUnit) -> i64 {
    match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    }
}

/// Whether `tz` is a zone whose offset from UTC is always zero.
///
/// Arrow parses a zone as a fixed offset or, failing that, as an IANA name,
/// case-sensitively; these are its spellings of a zero offset and the IANA
/// names that alias UTC.
fn is_utc(tz: &str) -> bool {
    const UTC_SPELLINGS: [&str; 24] = [
        "+00:00",
        "-00:00",
        "+0000",
        "-0000",
        "+00",
        "-00",
        "UTC",
        "UCT",
        "GMT",
        "GMT0",
        "GMT+0",
        "GMT-0",
        "Zulu",
        "Universal",
        "Greenwich",
        "Etc/UTC",
        "Etc/UCT",
        "Etc/GMT",
        "Etc/GMT0",
        "Etc/GMT+0",
        "Etc/GMT-0",
        "Etc/Zulu",
        "Etc/Universal",
        "Etc/Greenwich",
    ];
    UTC_SPELLINGS.contains(&tz)
}

fn widened_timestamp<'a>(expr: &'a Expr, schema: &DFSchema) -> Option<WidenedTimestamp<'a>> {
    // DataFusion unwraps `TRY_CAST` the same way as `CAST`.
    let (inner, widened_type) = match expr {
        Expr::Cast(cast) => (cast.expr.as_ref(), cast.field.data_type()),
        Expr::TryCast(cast) => (cast.expr.as_ref(), cast.field.data_type()),
        _ => return None,
    };
    let inner_type = inner.get_type(schema).ok()?;
    let (
        DataType::Timestamp(column_unit, column_tz),
        DataType::Timestamp(widened_unit, widened_tz),
    ) = (&inner_type, widened_type)
    else {
        return None;
    };
    // A cast that gives a zoneless column a time zone also shifts its values by
    // that zone's offset, so it is more than a change of unit unless the offset
    // is always zero, as for the `+00:00` zone of a literal such as
    // `TIMESTAMP '2024-01-01T12:00:00.5Z'` or of `now()`. Other casts between
    // zones keep the values.
    if let (None, Some(tz)) = (column_tz, widened_tz)
        && !is_utc(tz)
    {
        return None;
    }
    let factor = units_per_second(widened_unit) / units_per_second(column_unit);
    (factor > 1).then(|| WidenedTimestamp {
        inner,
        inner_type: inner_type.clone(),
        widened_type: widened_type.clone(),
        factor,
    })
}

/// The value of an operand of the widened type, in its unit, once folded to a
/// literal.
///
/// The operand is folded first because an operand such as `now()`,
/// `arrow_cast(...)` or `to_timestamp_micros(...)` only becomes a literal in
/// `simplify`, which unwraps the cast in the same pass.
fn widened_value(
    operand: &Expr,
    widened_type: &DataType,
    simplify: &dyn Fn(Expr) -> DFResult<Expr>,
) -> DFResult<Option<Option<i64>>> {
    let folded;
    let operand = if matches!(operand, Expr::Literal(..)) {
        operand
    } else {
        folded = simplify(operand.clone())?;
        &folded
    };
    let Some(value) = extract_literal(operand) else {
        return Ok(None);
    };
    if value.data_type() != *widened_type {
        return Ok(None);
    }
    Ok(match value {
        ScalarValue::TimestampSecond(value, _)
        | ScalarValue::TimestampMillisecond(value, _)
        | ScalarValue::TimestampMicrosecond(value, _)
        | ScalarValue::TimestampNanosecond(value, _) => Some(value),
        _ => None,
    })
}

/// `column op value` on the column's own unit, where `op` is one of
/// [`COMPARISONS`] and `value` an operand in the widened unit.
fn unwrap_comparison(widened: &WidenedTimestamp, op: Operator, value: Option<i64>) -> Option<Expr> {
    use Operator::*;
    let units = match value {
        None => None,
        Some(value) => match widened.exact(value) {
            Some(units) => Some(units),
            // The operand lies strictly between two values of the column.
            None => {
                let comparison = match op {
                    // A NULL row is distinct from any value, which
                    // `Comparison::Constant` would answer as NULL.
                    IsDistinctFrom => return Some(lit(true)),
                    IsNotDistinctFrom => return Some(lit(false)),
                    Eq => Comparison::Constant(false),
                    NotEq => Comparison::Constant(true),
                    Lt | LtEq => Comparison::Literal(LtEq, widened.floor(value)?),
                    Gt | GtEq => Comparison::Literal(Gt, widened.floor(value)?),
                    // Callers only pass the operators in `COMPARISONS`.
                    _ => return None,
                };
                return comparison_expr(widened.inner, &widened.inner_type, comparison);
            }
        },
    };
    let comparison = Comparison::Literal(op, widened.scalar(units)?);
    comparison_expr(widened.inner, &widened.inner_type, comparison)
}

fn unwrap_node(
    expr: &Expr,
    simplify: &dyn Fn(Expr) -> DFResult<Expr>,
    schema: &DFSchema,
) -> DFResult<Option<Expr>> {
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            if !COMPARISONS.contains(op) {
                return Ok(None);
            }
            let (widened, op, operand) = match widened_timestamp(left, schema) {
                Some(widened) => (widened, *op, right),
                None => match (widened_timestamp(right, schema), op.swap()) {
                    (Some(widened), Some(op)) => (widened, op, left),
                    _ => return Ok(None),
                },
            };
            let Some(value) = widened_value(operand, &widened.widened_type, simplify)? else {
                return Ok(None);
            };
            Ok(unwrap_comparison(&widened, op, value))
        }
        Expr::Between(Between {
            expr,
            negated,
            low,
            high,
        }) => {
            let Some(widened) = widened_timestamp(expr, schema) else {
                return Ok(None);
            };
            let (Some(low), Some(high)) = (
                widened_value(low, &widened.widened_type, simplify)?,
                widened_value(high, &widened.widened_type, simplify)?,
            ) else {
                return Ok(None);
            };
            let between = unwrap_comparison(&widened, Operator::GtEq, low)
                .zip(unwrap_comparison(&widened, Operator::LtEq, high))
                .map(|(low, high)| low.and(high));
            Ok(between.map(|between| if *negated { !between } else { between }))
        }
        Expr::InList(InList {
            expr,
            list,
            negated,
        }) => {
            let Some(widened) = widened_timestamp(expr, schema) else {
                return Ok(None);
            };
            let mut values = Vec::with_capacity(list.len());
            for item in list {
                let Some(value) = widened_value(item, &widened.widened_type, simplify)? else {
                    return Ok(None);
                };
                let units = match value {
                    // A NULL item keeps the result NULL for rows that match no
                    // other item.
                    None => None,
                    Some(value) => match widened.exact(value) {
                        Some(units) => Some(units),
                        // An item between two of the column's values matches
                        // no row.
                        None => continue,
                    },
                };
                let Some(scalar) = widened.scalar(units) else {
                    return Ok(None);
                };
                values.push(lit(scalar));
            }
            let matches = if values.is_empty() {
                comparison_expr(
                    widened.inner,
                    &widened.inner_type,
                    Comparison::Constant(false),
                )
            } else {
                Some(widened.inner.clone().in_list(values, false))
            };
            Ok(matches.map(|matches| if *negated { !matches } else { matches }))
        }
        _ => Ok(None),
    }
}

/// Rewrites each comparison, `BETWEEN` and `IN` list of a timestamp column, or
/// other timestamp expression, cast to a finer unit against operands that fold
/// to literals of that unit into one on the uncast expression that gives the
/// same result for every row.
///
/// Runs after type coercion, which inserts the casts, and before
/// simplification, which would otherwise truncate the operands. `simplify`
/// folds the operands.
pub fn unwrap_timestamp_widening_casts(
    expr: Expr,
    simplify: &dyn Fn(Expr) -> DFResult<Expr>,
    schema: &DFSchema,
) -> Result<Expr> {
    Ok(expr
        .transform_up(|node| {
            let Some(mut rewritten) = unwrap_node(&node, simplify, schema)? else {
                return Ok(Transformed::no(node));
            };
            // A rewrite strips one cast. The expression it wrapped can be a
            // widening cast itself, such as `CAST(ts AS TIMESTAMP(3))` once
            // coercion widens it again to a microsecond operand, which
            // `simplify` would unwrap by truncation in turn.
            while let Some(again) = unwrap_node(&rewritten, simplify, schema)? {
                rewritten = again;
            }
            Ok(Transformed::yes(rewritten))
        })?
        .data)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, RecordBatch, StructArray, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray, cast::AsArray,
    };
    use arrow_buffer::IntervalDayTime;
    use arrow_schema::{Field, Fields, TimeUnit};
    use chrono::Utc;
    use datafusion::logical_expr::{Expr, Operator, binary_expr};
    use datafusion::prelude::{col, lit};
    use datafusion::scalar::ScalarValue;
    use rstest::rstest;

    use super::{COMPARISONS, is_utc};
    use crate::planner::Planner;

    const SECOND: &str = "1970-01-01 00:00:01";
    const BEFORE_EPOCH: &str = "1969-12-31 23:59:58";

    fn evaluate(planner: &Planner, batch: &RecordBatch, expr: Expr) -> Vec<Option<bool>> {
        let expr = planner.optimize_expr(expr).unwrap();
        let result = planner
            .create_physical_expr(&expr)
            .unwrap()
            .evaluate(batch)
            .unwrap()
            .into_array(batch.num_rows())
            .unwrap();
        result.as_boolean().iter().collect()
    }

    /// A `timestamp(s)` column holding -2..=2 seconds and a null.
    fn seconds_batch() -> RecordBatch {
        let ts =
            TimestampSecondArray::from(vec![Some(-2), Some(-1), Some(0), Some(1), Some(2), None]);
        RecordBatch::try_from_iter(vec![("ts", Arc::new(ts) as ArrayRef)]).unwrap()
    }

    /// Rows of [`seconds_batch`] that match, by their value in seconds.
    fn matching(seconds: &[i64]) -> Vec<Option<bool>> {
        let mut expected: Vec<_> = (-2..=2).map(|s| Some(seconds.contains(&s))).collect();
        expected.push(None);
        expected
    }

    // The literal is a half second (or a nanosecond) past a whole second, so
    // it can only be compared with the column exactly, not on a truncated value.
    #[rstest]
    #[case::eq(format!("ts = TIMESTAMP '{SECOND}.5'"), &[])]
    #[case::not_eq(format!("ts != TIMESTAMP '{SECOND}.5'"), &[-2, -1, 0, 1, 2])]
    #[case::lt(format!("ts < TIMESTAMP '{SECOND}.5'"), &[-2, -1, 0, 1])]
    #[case::lt_eq(format!("ts <= TIMESTAMP '{SECOND}.5'"), &[-2, -1, 0, 1])]
    #[case::gt(format!("ts > TIMESTAMP '{SECOND}.5'"), &[2])]
    #[case::gt_eq(format!("ts >= TIMESTAMP '{SECOND}.5'"), &[2])]
    #[case::reversed(format!("TIMESTAMP '{SECOND}.5' <= ts"), &[2])]
    #[case::before_epoch_gt(format!("ts > TIMESTAMP '{BEFORE_EPOCH}.5'"), &[-1, 0, 1, 2])]
    #[case::before_epoch_lt(format!("ts < TIMESTAMP '{BEFORE_EPOCH}.5'"), &[-2])]
    #[case::nanosecond(
        format!("ts >= CAST('{SECOND}.000000001' AS TIMESTAMP(9))"),
        &[2]
    )]
    #[case::exact(format!("ts >= TIMESTAMP '{SECOND}'"), &[1, 2])]
    #[case::exact_eq(format!("ts = TIMESTAMP '{SECOND}.000'"), &[1])]
    #[case::utc_designator(format!("ts >= TIMESTAMP '{SECOND}.5Z'"), &[2])]
    #[case::utc_designator_eq(format!("ts = TIMESTAMP '{SECOND}.5Z'"), &[])]
    #[case::in_list(
        format!("ts IN (TIMESTAMP '{SECOND}.5', TIMESTAMP '1970-01-01 00:00:02')"),
        &[2]
    )]
    #[case::not_in(format!("ts NOT IN (TIMESTAMP '{SECOND}.5')"), &[-2, -1, 0, 1, 2])]
    #[case::between(
        format!("ts BETWEEN TIMESTAMP '1970-01-01 00:00:00.5' AND TIMESTAMP '{SECOND}.5'"),
        &[1]
    )]
    #[case::not_between(
        format!("ts NOT BETWEEN TIMESTAMP '1970-01-01 00:00:00.5' AND TIMESTAMP '{SECOND}.5'"),
        &[-2, -1, 0, 2]
    )]
    // A cast written in the filter, which coercion widens again for the
    // microsecond operand: both casts would be unwrapped by truncation.
    #[case::explicit_cast(
        format!("CAST(ts AS TIMESTAMP(3)) >= TIMESTAMP '{SECOND}.0005'"),
        &[2]
    )]
    #[case::explicit_cast_exact_in_cast_unit(
        format!("CAST(ts AS TIMESTAMP(3)) >= TIMESTAMP '{SECOND}.5'"),
        &[2]
    )]
    #[case::explicit_cast_eq(
        format!("CAST(ts AS TIMESTAMP(3)) = TIMESTAMP '{SECOND}.5'"),
        &[]
    )]
    // Operands that only become literals once DataFusion folds them.
    #[case::arrow_cast(
        format!("ts >= arrow_cast('{SECOND}.5', 'Timestamp(Microsecond, None)')"),
        &[2]
    )]
    #[case::to_timestamp_micros(format!("ts >= to_timestamp_micros('{SECOND}.5')"), &[2])]
    #[case::to_timestamp_nanos(format!("ts < to_timestamp_nanos('{SECOND}.000000001')"), &[-2, -1, 0, 1])]
    #[case::to_timestamp_millis(format!("ts = to_timestamp_millis('{SECOND}.5')"), &[])]
    #[case::folded_in_list(
        format!("ts IN (to_timestamp_micros('{SECOND}.5'), to_timestamp_micros('1970-01-01 00:00:02'))"),
        &[2]
    )]
    #[case::folded_between(
        format!("ts BETWEEN to_timestamp_micros('1970-01-01 00:00:00.5') AND to_timestamp_micros('{SECOND}.5')"),
        &[1]
    )]
    fn sql_filters_compare_exactly(#[case] filter: String, #[case] expected: &[i64]) {
        let batch = seconds_batch();
        let planner = Planner::new(batch.schema());
        let expr = planner.parse_filter(&filter).unwrap();
        assert_eq!(
            evaluate(&planner, &batch, expr),
            matching(expected),
            "{filter}"
        );
    }

    /// `IS [NOT] DISTINCT FROM` is never NULL, so the NULL row has an answer.
    #[rstest]
    #[case::not_distinct(
        format!("ts IS NOT DISTINCT FROM TIMESTAMP '{SECOND}.5'"),
        [false, false, false, false, false, false]
    )]
    #[case::distinct(
        format!("ts IS DISTINCT FROM TIMESTAMP '{SECOND}.5'"),
        [true, true, true, true, true, true]
    )]
    #[case::not_distinct_exact(
        format!("ts IS NOT DISTINCT FROM TIMESTAMP '{SECOND}'"),
        [false, false, false, true, false, false]
    )]
    #[case::distinct_exact(
        format!("ts IS DISTINCT FROM TIMESTAMP '{SECOND}'"),
        [true, true, true, false, true, true]
    )]
    #[case::not_distinct_null(
        "ts IS NOT DISTINCT FROM CAST(NULL AS TIMESTAMP)".to_string(),
        [false, false, false, false, false, true]
    )]
    #[case::distinct_null(
        "ts IS DISTINCT FROM CAST(NULL AS TIMESTAMP)".to_string(),
        [true, true, true, true, true, false]
    )]
    fn distinct_from_compares_exactly(#[case] filter: String, #[case] expected: [bool; 6]) {
        let batch = seconds_batch();
        let planner = Planner::new(batch.schema());
        let expr = planner.parse_filter(&filter).unwrap();
        assert_eq!(
            evaluate(&planner, &batch, expr),
            expected.map(Some).to_vec(),
            "{filter}"
        );
    }

    /// `now()` is a zoned nanosecond timestamp that `simplify` folds, so it
    /// reaches a zoneless second column through both the cast and the fold.
    #[test]
    fn now_compares_exactly() {
        let planner = Planner::new(seconds_batch().schema());
        let before = Utc::now();
        let expr = planner
            .optimize_expr(planner.parse_filter("ts = now()").unwrap())
            .unwrap();
        let after = Utc::now();
        // `now()` lies between `before` and `after`, so it is known not to be a
        // whole second when both fall in the same second past its start, and
        // then no row of a second column can equal it.
        if before.timestamp() == after.timestamp() && before.timestamp_subsec_nanos() > 0 {
            let never = col("ts").lt(lit(ScalarValue::TimestampSecond(Some(i64::MIN), None)));
            assert_eq!(expr, never);
        }

        let expr = planner
            .optimize_expr(planner.parse_filter("ts >= now()").unwrap())
            .unwrap();
        let Expr::BinaryExpr(comparison) = &expr else {
            panic!("unexpected plan {expr}");
        };
        assert_eq!(*comparison.left, col("ts"), "{expr}");
        assert!(
            matches!(
                comparison.right.as_ref(),
                Expr::Literal(ScalarValue::TimestampSecond(Some(_), None), _)
            ),
            "{expr}"
        );
    }

    /// A zoned column compared with a zoneless literal.
    #[rstest]
    #[case::gt_eq(format!("ts >= TIMESTAMP '{SECOND}.5'"), [false, false, false, false, true])]
    #[case::lt(format!("ts < TIMESTAMP '{SECOND}.5'"), [true, true, true, true, false])]
    #[case::eq(format!("ts = TIMESTAMP '{SECOND}.5Z'"), [false; 5])]
    fn zoned_column_compares_exactly(#[case] filter: String, #[case] expected: [bool; 5]) {
        let ts = TimestampSecondArray::from(vec![-2, -1, 0, 1, 2]).with_timezone("UTC");
        let batch = RecordBatch::try_from_iter(vec![("ts", Arc::new(ts) as ArrayRef)]).unwrap();
        let planner = Planner::new(batch.schema());
        let expr = planner.parse_filter(&filter).unwrap();
        assert_eq!(
            evaluate(&planner, &batch, expr),
            expected.map(Some).to_vec(),
            "{filter}"
        );
    }

    #[rstest]
    #[case::offset("+00:00", true)]
    #[case::utc("UTC", true)]
    #[case::lowercase("utc", false)]
    #[case::designator("Z", false)]
    #[case::etc_gmt("Etc/GMT", true)]
    #[case::short_offset("+00", true)]
    #[case::offset_hours("+05:00", false)]
    #[case::named_zone("Europe/London", false)]
    #[case::gmt_offset("Etc/GMT+1", false)]
    fn recognizes_utc(#[case] tz: &str, #[case] expected: bool) {
        assert_eq!(is_utc(tz), expected, "{tz}");
    }

    #[rstest]
    #[case::null_eq("ts = CAST(NULL AS TIMESTAMP)".to_string())]
    #[case::null_in(format!("ts IN (TIMESTAMP '{SECOND}.5', CAST(NULL AS TIMESTAMP))"))]
    fn null_operands_stay_null(#[case] filter: String) {
        let batch = seconds_batch();
        let planner = Planner::new(batch.schema());
        let expr = planner.parse_filter(&filter).unwrap();
        assert_eq!(evaluate(&planner, &batch, expr), vec![None; 6], "{filter}");
    }

    fn nanos_per_unit(unit: TimeUnit) -> i64 {
        match unit {
            TimeUnit::Second => 1_000_000_000,
            TimeUnit::Millisecond => 1_000_000,
            TimeUnit::Microsecond => 1_000,
            TimeUnit::Nanosecond => 1,
        }
    }

    fn timestamp_array(unit: TimeUnit, values: Vec<Option<i64>>, tz: Option<&str>) -> ArrayRef {
        // Sets the zone without shifting the values, which a cast from a
        // zoneless array would do.
        let tz = tz.map(Arc::<str>::from);
        match unit {
            TimeUnit::Second => Arc::new(TimestampSecondArray::from(values).with_timezone_opt(tz)),
            TimeUnit::Millisecond => {
                Arc::new(TimestampMillisecondArray::from(values).with_timezone_opt(tz))
            }
            TimeUnit::Microsecond => {
                Arc::new(TimestampMicrosecondArray::from(values).with_timezone_opt(tz))
            }
            TimeUnit::Nanosecond => {
                Arc::new(TimestampNanosecondArray::from(values).with_timezone_opt(tz))
            }
        }
    }

    fn timestamp_literal(unit: TimeUnit, value: i64, tz: Option<&str>) -> ScalarValue {
        let tz = tz.map(Arc::from);
        match unit {
            TimeUnit::Second => ScalarValue::TimestampSecond(Some(value), tz),
            TimeUnit::Millisecond => ScalarValue::TimestampMillisecond(Some(value), tz),
            TimeUnit::Microsecond => ScalarValue::TimestampMicrosecond(Some(value), tz),
            TimeUnit::Nanosecond => ScalarValue::TimestampNanosecond(Some(value), tz),
        }
    }

    /// Filters built from a typed literal, as Substrait filters and the Rust
    /// API produce them, agree with comparing both sides in nanoseconds, for
    /// every column unit and every finer literal unit.
    #[rstest]
    #[case::s_ms(TimeUnit::Second, TimeUnit::Millisecond)]
    #[case::s_us(TimeUnit::Second, TimeUnit::Microsecond)]
    #[case::s_ns(TimeUnit::Second, TimeUnit::Nanosecond)]
    #[case::ms_us(TimeUnit::Millisecond, TimeUnit::Microsecond)]
    #[case::ms_ns(TimeUnit::Millisecond, TimeUnit::Nanosecond)]
    #[case::us_ns(TimeUnit::Microsecond, TimeUnit::Nanosecond)]
    fn typed_literals_compare_exactly(
        #[case] column_unit: TimeUnit,
        #[case] literal_unit: TimeUnit,
        #[values(None, Some("UTC"), Some("America/New_York"))] tz: Option<&str>,
    ) {
        let column_scale = nanos_per_unit(column_unit);
        let literal_scale = nanos_per_unit(literal_unit);
        let values: Vec<i64> = (-3..=3).collect();
        let mut column = values.iter().copied().map(Some).collect::<Vec<_>>();
        column.push(None);
        let batch =
            RecordBatch::try_from_iter(vec![("ts", timestamp_array(column_unit, column, tz))])
                .unwrap();
        let planner = Planner::new(batch.schema());

        let step = column_scale / literal_scale;
        // Whole column units, and values just past, halfway past, and just
        // before them, on both sides of the epoch.
        let literals = [
            -2 * step - 1,
            -step - step / 2,
            -1,
            0,
            1,
            step / 2,
            step,
            step + 1,
        ];
        for literal in literals {
            for op in COMPARISONS {
                let scalar = timestamp_literal(literal_unit, literal, tz);
                let literal_nanos = i128::from(literal) * i128::from(literal_scale);
                let mut expected = values
                    .iter()
                    .map(|value| {
                        let value_nanos = i128::from(*value) * i128::from(column_scale);
                        Some(match op {
                            Operator::Eq | Operator::IsNotDistinctFrom => {
                                value_nanos == literal_nanos
                            }
                            Operator::NotEq | Operator::IsDistinctFrom => {
                                value_nanos != literal_nanos
                            }
                            Operator::Lt => value_nanos < literal_nanos,
                            Operator::LtEq => value_nanos <= literal_nanos,
                            Operator::Gt => value_nanos > literal_nanos,
                            Operator::GtEq => value_nanos >= literal_nanos,
                            _ => unreachable!(),
                        })
                    })
                    .collect::<Vec<_>>();
                // A NULL row is NULL for the comparisons, distinct from a value
                // for the others.
                expected.push(match op {
                    Operator::IsDistinctFrom => Some(true),
                    Operator::IsNotDistinctFrom => Some(false),
                    _ => None,
                });

                let expr = binary_expr(col("ts"), op, lit(scalar.clone()));
                assert_eq!(
                    evaluate(&planner, &batch, expr),
                    expected,
                    "ts {op} {scalar:?} on {column_unit:?}"
                );
                let swapped = binary_expr(lit(scalar.clone()), op.swap().unwrap(), col("ts"));
                assert_eq!(
                    evaluate(&planner, &batch, swapped),
                    expected,
                    "{scalar:?} {} ts on {column_unit:?}",
                    op.swap().unwrap()
                );
            }
        }
    }

    /// DataFusion unwraps the cast on any expression, not only a column, so
    /// `ts + INTERVAL '1 second' >= <1.5 s>` would compare `ts + 1 s` with 1 s.
    #[test]
    fn expressions_compare_exactly() {
        let batch = seconds_batch();
        let planner = Planner::new(batch.schema());
        let one_second = lit(ScalarValue::IntervalDayTime(Some(IntervalDayTime::new(
            0, 1_000,
        ))));
        let bound = lit(ScalarValue::TimestampMicrosecond(Some(1_500_000), None));
        let expr = (col("ts") + one_second).gt_eq(bound);
        assert_eq!(evaluate(&planner, &batch, expr), matching(&[1, 2]));
    }

    #[test]
    fn struct_fields_compare_exactly() {
        let ts: ArrayRef = Arc::new(TimestampSecondArray::from(vec![1, 2]));
        let fields = Fields::from(vec![Field::new("ts", ts.data_type().clone(), true)]);
        let event = StructArray::new(fields, vec![ts], None);
        let batch =
            RecordBatch::try_from_iter(vec![("event", Arc::new(event) as ArrayRef)]).unwrap();
        let planner = Planner::new(batch.schema());
        let expr = planner
            .parse_filter(&format!("event.ts >= TIMESTAMP '{SECOND}.5'"))
            .unwrap();
        assert_eq!(
            evaluate(&planner, &batch, expr),
            vec![Some(false), Some(true)]
        );
    }

    /// The comparison is planned on the column itself, so a scalar index on it
    /// still applies, and the result is a fixed point of `optimize_expr`, which
    /// the scan path runs twice.
    #[rstest]
    #[case::gt_eq(
        format!("ts >= TIMESTAMP '{SECOND}.5'"),
        col("ts").gt(lit(ScalarValue::TimestampSecond(Some(1), None)))
    )]
    #[case::lt(
        format!("ts < TIMESTAMP '{BEFORE_EPOCH}.5'"),
        col("ts").lt_eq(lit(ScalarValue::TimestampSecond(Some(-2), None)))
    )]
    #[case::exact(
        format!("ts = TIMESTAMP '{SECOND}'"),
        col("ts").eq(lit(ScalarValue::TimestampSecond(Some(1), None)))
    )]
    #[case::explicit_cast(
        format!("CAST(ts AS TIMESTAMP(3)) >= TIMESTAMP '{SECOND}.5'"),
        col("ts").gt(lit(ScalarValue::TimestampSecond(Some(1), None)))
    )]
    fn plans_on_column(#[case] filter: String, #[case] expected: Expr) {
        let planner = Planner::new(seconds_batch().schema());
        let once = planner
            .optimize_expr(planner.parse_filter(&filter).unwrap())
            .unwrap();
        assert_eq!(once, expected, "{filter}");
        assert_eq!(planner.optimize_expr(once.clone()).unwrap(), once);
    }
}
