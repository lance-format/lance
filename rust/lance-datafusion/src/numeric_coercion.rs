// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index-friendly integer/float comparisons. They use exact mathematical
//! values, rather than rounding the integer to a float.

use arrow_schema::DataType;
use datafusion::logical_expr::{Between, BinaryExpr, Operator};
use datafusion::prelude::{Expr, lit};
use datafusion::scalar::ScalarValue;
use lance_core::datatypes::Schema;

use crate::expr::safe_coerce_scalar;
use crate::logical_expr::resolve_column_type;

/// Inclusive lower and exclusive upper bounds, exactly representable as f64.
fn int_bounds(data_type: &DataType) -> Option<(f64, f64)> {
    if !data_type.is_integer() {
        return None;
    }
    let signed = data_type.is_signed_integer();
    let upper = 2_f64.powi(data_type.primitive_width()? as i32 * 8 - i32::from(signed));
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
        _ => None,
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
    /// The operand is a float, which the existing coercion rejects.
    is_float: bool,
}

fn comparison_expr(column: &Expr, data_type: &DataType, comparison: Comparison) -> Option<Expr> {
    let (op, value) = match comparison {
        Comparison::Literal(op, value) => (op, value),
        // Domain bounds keep NULL rows NULL and stay indexable.
        Comparison::Constant(value) => (
            if value { Operator::GtEq } else { Operator::Lt },
            ScalarValue::min(data_type)?,
        ),
    };
    Some(Expr::BinaryExpr(BinaryExpr::new(
        Box::new(column.clone()),
        op,
        Box::new(lit(value)),
    )))
}

fn lower(column: &Expr, op: Operator, operand: &Expr, schema: &Schema) -> Option<Lowered> {
    use Operator::*;
    let data_type = resolve_column_type(column, schema)?;
    let (min, max) = int_bounds(&data_type)?;
    let scalar = extract_literal(operand)?;
    let is_float = matches!(
        scalar,
        ScalarValue::Float16(_) | ScalarValue::Float32(_) | ScalarValue::Float64(_)
    );
    let comparison = if scalar.is_null() {
        Comparison::Literal(op, ScalarValue::try_new_null(&data_type).ok()?)
    } else if !is_float {
        // Operands the existing coercion already handles keep its behavior.
        Comparison::Literal(op, safe_coerce_scalar(&scalar, &data_type)?)
    } else {
        let ScalarValue::Float64(Some(value)) = scalar.cast_to(&DataType::Float64).ok()? else {
            return None;
        };
        if !value.is_finite() || value < min || value >= max {
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
        column: column.clone(),
        data_type,
        comparison,
        is_float,
    })
}

/// Rewrites integer column comparisons against float operands, which the
/// existing coercion rejects. Returns `None` when no operand is a float, so
/// other filters keep their current shape.
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
            if !lowered.is_float {
                return None;
            }
            comparison_expr(&lowered.column, &lowered.data_type, lowered.comparison)
        }
        Expr::Between(Between {
            expr: column,
            low,
            high,
            negated,
        }) => {
            let low = lower(column, Operator::GtEq, low, schema)?;
            let high = lower(column, Operator::LtEq, high, schema)?;
            if !(low.is_float || high.is_float) {
                return None;
            }
            let low = comparison_expr(&low.column, &low.data_type, low.comparison)?;
            let high = comparison_expr(&high.column, &high.data_type, high.comparison)?;
            Some(negate(low.and(high), *negated))
        }
        Expr::InList(in_list) => {
            let lowered = in_list
                .list
                .iter()
                .map(|item| lower(&in_list.expr, Operator::Eq, item, schema))
                .collect::<Option<Vec<_>>>()?;
            let first = lowered.iter().find(|item| item.is_float)?;
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
                comparison_expr(&column, &data_type, Comparison::Constant(false))?
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
    #[case::exact_large("i > 9007199254740992.0", 9007199254740993, Some(true))]
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
    #[case::nan_eq("i = CAST('NaN' AS DOUBLE)", 0, Some(false))]
    #[case::infinity("i > CAST('-inf' AS DOUBLE)", 0, Some(true))]
    fn predicates(#[case] sql: &str, #[case] value: i64, #[case] expected: Option<bool>) {
        let batch = record_batch!(("i", Int64, [Some(value), None])).unwrap();
        let planner = Planner::new(batch.schema());
        let expr = planner.optimize_expr(planner.parse_filter(sql).unwrap());
        let result = planner.create_physical_expr(&expr.unwrap()).unwrap();
        let result = result.evaluate(&batch).unwrap().into_array(2).unwrap();
        let expected = BooleanArray::from(vec![expected, None]);
        assert_eq!(result.as_boolean(), &expected, "{sql}");
    }

    #[rstest]
    #[case::fraction("i > 1.5", col("i").gt(lit(1_i64)))]
    #[case::in_list("i IN (1.5, 2.0)", col("i").in_list(vec![lit(2_i64)], false))]
    #[case::never_in("i IN (1.5)", col("i").lt(lit(i64::MIN)))]
    #[case::below_unsigned("u > -1.5", col("u").gt_eq(lit(0_u8)))]
    #[case::above_unsigned("u < 255.5", col("u").lt_eq(lit(255_u8)))]
    fn plans_on_column(#[case] sql: &str, #[case] expected: Expr) {
        let batch = record_batch!(("i", Int64, [Some(1)]), ("u", UInt8, [Some(1)])).unwrap();
        let planner = Planner::new(batch.schema());
        assert_eq!(planner.parse_filter(sql).unwrap(), expected, "{sql}");
    }
}
