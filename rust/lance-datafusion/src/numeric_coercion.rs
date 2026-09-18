// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index-friendly numeric comparisons. Bare integer/float comparisons use exact
//! mathematical values, rather than rounding the integer to a float. Explicit
//! column casts retain their rounding semantics unless lossless over the domain.

use arrow_schema::DataType;
use datafusion::logical_expr::{Between, BinaryExpr, Operator};
use datafusion::prelude::{Expr, lit};
use datafusion::scalar::ScalarValue;
use lance_core::datatypes::Schema;

use crate::expr::safe_coerce_scalar;
use crate::logical_expr::resolve_column_type;

/// Inclusive lower and exclusive upper bounds, exactly representable as f64.
fn int_bounds(data_type: &DataType) -> Option<(f64, f64)> {
    use DataType::*;
    let (bits, signed) = match data_type {
        Int8 => (8, true),
        Int16 => (16, true),
        Int32 => (32, true),
        Int64 => (64, true),
        UInt8 => (8, false),
        UInt16 => (16, false),
        UInt32 => (32, false),
        UInt64 => (64, false),
        _ => return None,
    };
    let upper = 2_f64.powi(bits - i32::from(signed));
    Some((if signed { -upper } else { 0.0 }, upper))
}

fn extract_literal(expr: &Expr) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(value, _) => Some(value.clone()),
        Expr::Cast(cast) => extract_literal(&cast.expr)?
            .cast_to(cast.field.data_type())
            .ok(),
        Expr::TryCast(cast) => extract_literal(&cast.expr)?
            .cast_to(cast.field.data_type())
            .or_else(|_| ScalarValue::try_new_null(cast.field.data_type()))
            .ok(),
        Expr::Negative(inner) => extract_literal(inner)?.arithmetic_negate().ok(),
        _ => None,
    }
}

/// The column behind float-targeted casts that are lossless over its domain,
/// its type, and whether a cast was removed.
fn extract_column(expr: &Expr, schema: &Schema) -> Option<(Expr, DataType, bool)> {
    let (inner, target) = match expr {
        Expr::Cast(cast) => (&cast.expr, cast.field.data_type()),
        Expr::TryCast(cast) => (&cast.expr, cast.field.data_type()),
        _ => return Some((expr.clone(), resolve_column_type(expr, schema)?, false)),
    };
    let (column, source, _) = extract_column(inner, schema)?;
    // Check every step: a narrowing intermediate cast has already rounded.
    let intermediate = match inner.as_ref() {
        Expr::Cast(cast) => cast.field.data_type().clone(),
        Expr::TryCast(cast) => cast.field.data_type().clone(),
        _ => source.clone(),
    };
    let precision = match target {
        DataType::Float32 => 24,
        DataType::Float64 => 53,
        _ => return None,
    };
    let lossless = match intermediate {
        DataType::Float32 => true,
        DataType::Float64 => *target == DataType::Float64,
        _ => int_bounds(&intermediate).is_some_and(|(min, max)| {
            min >= -2_f64.powi(precision) && max <= 2_f64.powi(precision)
        }),
    };
    lossless.then_some((column, source, true))
}

fn is_bare_literal(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(..) => true,
        Expr::Negative(inner) => is_bare_literal(inner),
        _ => false,
    }
}

/// A comparison of a column against one operand, lowered to the column type.
enum Comparison {
    Literal(Operator, ScalarValue),
    /// Always true or false for non-null rows.
    Constant(bool),
}

struct Lowered {
    column: Expr,
    data_type: DataType,
    comparison: Comparison,
    /// The existing coercion would error or leave the column cast.
    is_required: bool,
}

impl Lowered {
    fn into_expr(self) -> Option<Expr> {
        let binary = |op, value| {
            Expr::BinaryExpr(BinaryExpr::new(
                Box::new(self.column.clone()),
                op,
                Box::new(lit(value)),
            ))
        };
        Some(match self.comparison {
            Comparison::Literal(op, value) => binary(op, value),
            // Domain bounds keep NULL rows NULL and stay indexable.
            Comparison::Constant(value) => {
                let minimum = ScalarValue::Float64(Some(int_bounds(&self.data_type)?.0))
                    .cast_to(&self.data_type)
                    .ok()?;
                binary(if value { Operator::GtEq } else { Operator::Lt }, minimum)
            }
        })
    }
}

fn lower(column: &Expr, op: Operator, operand: &Expr, schema: &Schema) -> Option<Lowered> {
    use Operator::*;
    let (column, data_type, is_cast) = extract_column(column, schema)?;
    let scalar = extract_literal(operand)?;
    let is_float = matches!(
        scalar,
        ScalarValue::Float16(_) | ScalarValue::Float32(_) | ScalarValue::Float64(_)
    );
    let is_float32 = data_type == DataType::Float32;
    let bounds = int_bounds(&data_type);
    if !is_float32 && bounds.is_none() {
        return None;
    }
    let is_required = is_cast
        || if is_float32 {
            !is_bare_literal(operand) && scalar.data_type() == DataType::Float64
        } else {
            is_float
        };
    let comparison = if scalar.is_null() {
        Comparison::Literal(op, ScalarValue::try_new_null(&data_type).ok()?)
    } else if (is_float32 && !is_cast && is_bare_literal(operand)) || (!is_float32 && !is_float) {
        // Operands the existing coercion already handles keep its behavior.
        Comparison::Literal(op, safe_coerce_scalar(&scalar, &data_type)?)
    } else {
        let ScalarValue::Float64(Some(value)) = scalar.cast_to(&DataType::Float64).ok()? else {
            return None;
        };
        let (min, max) = bounds.unwrap_or_default();
        if is_float32 {
            // Narrow only when the total-order value survives the round trip.
            let narrowed = value as f32;
            if f64::from(narrowed).total_cmp(&value).is_ne() {
                return None;
            }
            Comparison::Literal(op, ScalarValue::Float32(Some(narrowed)))
        } else if !value.is_finite() || value < min || value >= max {
            // Arrow orders negative NaNs below all numbers and positive NaNs
            // above them. Infinities and out-of-domain bounds follow suit.
            let is_above = if value.is_nan() {
                value.is_sign_positive()
            } else {
                value >= max
            };
            Comparison::Constant(match op {
                Eq => false,
                NotEq => true,
                Lt | LtEq => is_above,
                Gt | GtEq => !is_above,
                _ => return None,
            })
        } else if value.fract() != 0.0 && matches!(op, Eq | NotEq) {
            Comparison::Constant(op == NotEq)
        } else {
            let op = match op {
                Lt if value.fract() != 0.0 => LtEq,
                GtEq if value.fract() != 0.0 => Gt,
                op => op,
            };
            let floor = ScalarValue::Float64(Some(value.floor()));
            Comparison::Literal(op, floor.cast_to(&data_type).ok()?)
        }
    };
    Some(Lowered {
        column,
        data_type,
        comparison,
        is_required,
    })
}

/// Rewrites integer and `Float32` comparisons that the existing coercion
/// rejects or would evaluate on a cast column. Returns `None` when no operand
/// needs it, so other filters keep their current shape.
pub fn rewrite(expr: &Expr, schema: &Schema) -> Option<Expr> {
    let negate = |expr: Expr, negated: bool| if negated { !expr } else { expr };
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            if !matches!(
                op,
                Operator::Eq
                    | Operator::NotEq
                    | Operator::Lt
                    | Operator::LtEq
                    | Operator::Gt
                    | Operator::GtEq
            ) {
                return None;
            }
            let lowered = lower(left, *op, right, schema)
                .or_else(|| lower(right, op.swap()?, left, schema))?;
            lowered.is_required.then(|| lowered.into_expr())?
        }
        Expr::Between(Between {
            expr: column,
            low,
            high,
            negated,
        }) => {
            let low = lower(column, Operator::GtEq, low, schema)?;
            let high = lower(column, Operator::LtEq, high, schema)?;
            if !(low.is_required || high.is_required) {
                return None;
            }
            Some(negate(low.into_expr()?.and(high.into_expr()?), *negated))
        }
        Expr::InList(in_list) => {
            let lowered = in_list
                .list
                .iter()
                .map(|item| lower(&in_list.expr, Operator::Eq, item, schema))
                .collect::<Option<Vec<_>>>()?;
            let first = lowered.iter().find(|item| item.is_required)?;
            let (column, data_type) = (first.column.clone(), first.data_type.clone());
            // Values that can never match are dropped; NULLs stay so the
            // result remains NULL for unmatched rows.
            let values = lowered
                .into_iter()
                .filter_map(|item| match item.comparison {
                    Comparison::Literal(_, value) => Some(lit(value)),
                    Comparison::Constant(_) => None,
                })
                .collect::<Vec<_>>();
            let result = if values.is_empty() {
                Lowered {
                    column,
                    data_type,
                    comparison: Comparison::Constant(false),
                    is_required: true,
                }
                .into_expr()?
            } else {
                column.in_list(values, false)
            };
            Some(negate(result, in_list.negated))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::Planner;
    use arrow_array::{BooleanArray, cast::AsArray, record_batch};
    use datafusion::prelude::col;
    use rstest::rstest;

    #[rstest]
    #[case::fraction("i > 1.5", 2, Some(true))]
    #[case::negative_fraction("i <= -1.5", -2, Some(true))]
    #[case::fraction_eq("i = 1.5", 1, Some(false))]
    #[case::fraction_ne("i != 1.5", 1, Some(true))]
    #[case::reversed("1.5 < i", 2, Some(true))]
    #[case::negative_zero("CAST(i AS DOUBLE) <= -0.0", 0, Some(true))]
    #[case::exact_large("i > 9007199254740992.0", 9007199254740993, Some(true))]
    #[case::rounded_double(
        "CAST(i AS DOUBLE) > 9007199254740992.0",
        9007199254740993,
        Some(false)
    )]
    #[case::rounded_float("CAST(i AS FLOAT) = 16777216.0", 16777217, Some(true))]
    #[case::rounded_between(
        "CAST(i AS DOUBLE) BETWEEN 9007199254740992.0 AND 9007199254740992.0",
        9007199254740993,
        Some(true)
    )]
    #[case::rounded_in(
        "CAST(i AS DOUBLE) IN (9007199254740992.0)",
        9007199254740993,
        Some(true)
    )]
    #[case::try_cast(
        "TRY_CAST(i AS DOUBLE) = 9007199254740992.0",
        9007199254740993,
        Some(true)
    )]
    #[case::cast_chain("CAST(CAST(i AS FLOAT) AS DOUBLE) = 16777216.0", 16777217, Some(true))]
    #[case::null_between("i NOT BETWEEN CAST(NULL AS DOUBLE) AND -1e100", 0, Some(true))]
    #[case::empty_between("i BETWEEN 1.5 AND -1.5", 0, Some(false))]
    #[case::mixed_between("i BETWEEN 1 AND 2.5", 2, Some(true))]
    #[case::mixed_in("i IN (1.5, 2)", 2, Some(true))]
    #[case::never_in("i NOT IN (1.5)", 1, Some(true))]
    #[case::null_in("i IN (1.5, NULL)", 0, None)]
    #[case::null_not_in("i NOT IN (1.5, CAST(NULL AS DOUBLE))", 0, None)]
    #[case::upper("i < 9223372036854775808.0", i64::MAX, Some(true))]
    #[case::lower("i = -9223372036854775808.0", i64::MIN, Some(true))]
    #[case::positive_nan("i < CAST('NaN' AS DOUBLE)", 0, Some(true))]
    #[case::negative_nan("i > -CAST('NaN' AS DOUBLE)", 0, Some(true))]
    #[case::nan_eq("i = CAST('NaN' AS DOUBLE)", 0, Some(false))]
    #[case::infinity("i > CAST('-inf' AS DOUBLE)", 0, Some(true))]
    #[case::float_bare("f = 0.1", 0, Some(true))]
    #[case::float_widened("CAST(f AS DOUBLE) = 0.1", 0, Some(false))]
    #[case::float_typed("f = CAST(0.1 AS DOUBLE)", 0, Some(false))]
    #[case::float_mixed_between("f BETWEEN 0.0 AND CAST(0.5 AS DOUBLE)", 0, Some(true))]
    fn predicates(#[case] sql: &str, #[case] value: i64, #[case] expected: Option<bool>) {
        let batch = record_batch!(
            ("i", Int64, [Some(value), None]),
            ("f", Float32, [Some(0.1), None])
        )
        .unwrap();
        let planner = Planner::new(batch.schema());
        let expr = planner.optimize_expr(planner.parse_filter(sql).unwrap());
        let result = planner.create_physical_expr(&expr.unwrap()).unwrap();
        let result = result.evaluate(&batch).unwrap().into_array(2).unwrap();
        let expected = BooleanArray::from(vec![expected, None]);
        assert_eq!(result.as_boolean(), &expected, "{sql}");
    }

    #[rstest]
    #[case::half("f > CAST(0.5 AS DOUBLE)", col("f").gt(lit(0.5_f32)))]
    #[case::widened("CAST(f AS DOUBLE) > 0.5", col("f").gt(lit(0.5_f32)))]
    #[case::negative_infinity("f > CAST('-inf' AS DOUBLE)", col("f").gt(lit(f32::NEG_INFINITY)))]
    #[case::null("f > CAST(NULL AS DOUBLE)", col("f").gt(lit(ScalarValue::Float32(None))))]
    #[case::fraction("i > 1.5", col("i").gt(lit(1_i64)))]
    #[case::in_list("i IN (1.5, 2.0)", col("i").in_list(vec![lit(2_i64)], false))]
    #[case::never_in("i IN (1.5)", col("i").lt(lit(i64::MIN)))]
    fn plans_on_column(#[case] sql: &str, #[case] expected: Expr) {
        let batch = record_batch!(("i", Int64, [Some(1)]), ("f", Float32, [Some(0.5)])).unwrap();
        let planner = Planner::new(batch.schema());
        assert_eq!(planner.parse_filter(sql).unwrap(), expected, "{sql}");
    }
}
