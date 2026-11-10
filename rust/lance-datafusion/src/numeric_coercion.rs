// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index-friendly comparisons of integer and decimal columns with numeric
//! operands. They use exact mathematical values, rather than rounding the
//! operand or the column to a float.

use arrow_buffer::i256;
use arrow_schema::{DECIMAL128_MAX_PRECISION, DataType};
use datafusion::logical_expr::{Between, BinaryExpr, Operator};
use datafusion::prelude::{Expr, lit};
use datafusion::scalar::ScalarValue;
use lance_core::datatypes::Schema;

use crate::expr::safe_coerce_scalar;
use crate::logical_expr::{coerce_expr, resolve_column_type};

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
    /// The existing coercion rejects the operand or rounds it.
    needs_rewrite: bool,
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

fn is_decimal(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Decimal128(..) | DataType::Decimal256(..)
    )
}

fn lower(column: &Expr, op: Operator, operand: &Expr, schema: &Schema) -> Option<Lowered> {
    use Operator::*;
    let data_type = resolve_column_type(column, schema)?;
    if is_decimal(&data_type) {
        return lower_decimal(column, op, operand, data_type);
    }
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
            // Float filters treat every NaN as above finite values, regardless
            // of its sign bit. Infinities and out-of-domain bounds retain their
            // Arrow total-order position.
            out_of_range(op, value.is_nan() || value >= max)?
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
        needs_rewrite: is_float,
    })
}

/// The comparison with an operand above every value the column holds when
/// `is_above`, or below every value otherwise.
fn out_of_range(op: Operator, is_above: bool) -> Option<Comparison> {
    use Operator::*;
    Some(Comparison::Constant(match op {
        Eq => false,
        NotEq => true,
        Lt | LtEq => is_above,
        Gt | GtEq => !is_above,
        _ => return None,
    }))
}

/// An integer or decimal operand as an unscaled value and its scale, so the
/// operand is `value * 10^-scale`.
fn exact_value(scalar: &ScalarValue) -> Option<(i256, i8)> {
    match scalar {
        ScalarValue::Decimal128(Some(value), _, scale) => Some((i256::from_i128(*value), *scale)),
        ScalarValue::Decimal256(Some(value), _, scale) => Some((*value, *scale)),
        _ if scalar.data_type().is_integer() => {
            // Every 64-bit integer has at most 20 digits.
            let ScalarValue::Decimal128(Some(value), _, _) = scalar
                .cast_to(&DataType::Decimal128(DECIMAL128_MAX_PRECISION, 0))
                .ok()?
            else {
                return None;
            };
            Some((i256::from_i128(value), 0))
        }
        _ => None,
    }
}

/// `units` of a decimal column type's scale, as a value of that type.
fn decimal_scalar(units: i256, data_type: &DataType) -> Option<ScalarValue> {
    match data_type {
        DataType::Decimal128(precision, scale) => Some(ScalarValue::Decimal128(
            Some(units.to_i128()?),
            *precision,
            *scale,
        )),
        DataType::Decimal256(precision, scale) => {
            Some(ScalarValue::Decimal256(Some(units), *precision, *scale))
        }
        _ => None,
    }
}

/// Lowers a comparison of a decimal column with an integer or a decimal of any
/// precision and scale, such as the `0.055` in `price < 0.055` on a
/// `Decimal128(15, 2)` column. The existing coercion casts the operand to the
/// column type, which rounds `0.055` to `0.06` and rejects operands outside the
/// column's precision.
///
/// Operands outside the precision compare as constants: Arrow requires every
/// decimal value to fit its precision.
fn lower_decimal(
    column: &Expr,
    op: Operator,
    operand: &Expr,
    data_type: DataType,
) -> Option<Lowered> {
    use Operator::*;
    let (DataType::Decimal128(precision, scale) | DataType::Decimal256(precision, scale)) =
        data_type
    else {
        return None;
    };
    let scalar = extract_literal(operand)?;
    if scalar.is_null() {
        let null_type = scalar.data_type();
        return Some(Lowered {
            column: column.clone(),
            comparison: Comparison::Literal(op, ScalarValue::try_new_null(&data_type).ok()?),
            needs_rewrite: null_type != data_type
                && !null_type.is_integer()
                && !matches!(scalar, ScalarValue::Null),
            data_type,
        });
    }
    let (value, value_scale) = exact_value(&scalar)?;
    let ten = i256::from_i128(10);
    let max = ten
        .checked_pow(u32::from(precision))?
        .checked_sub(i256::ONE)?;
    // The operand in units of the column's scale, rounded down, and whether
    // rounding dropped any digits. `None` when it overflows every decimal.
    let shift = i32::from(scale) - i32::from(value_scale);
    let units = if value == i256::ZERO {
        Some((i256::ZERO, true))
    } else if shift >= 0 {
        ten.checked_pow(shift.unsigned_abs())
            .and_then(|factor| value.checked_mul(factor))
            .map(|units| (units, true))
    } else if let Some(divisor) = ten.checked_pow(shift.unsigned_abs()) {
        let quotient = value.checked_div(divisor)?;
        let remainder = value.checked_rem(divisor)?;
        let floor = if remainder.is_negative() {
            quotient.checked_sub(i256::ONE)?
        } else {
            quotient
        };
        Some((floor, remainder == i256::ZERO))
    } else {
        // A divisor past i256 exceeds every operand, which is then strictly
        // within one unit of zero.
        let floor = if value.is_negative() {
            i256::MINUS_ONE
        } else {
            i256::ZERO
        };
        Some((floor, false))
    };
    let comparison = match units {
        None => out_of_range(op, !value.is_negative())?,
        Some((floor, is_exact)) if floor > max || (floor == max && !is_exact) => {
            out_of_range(op, true)?
        }
        Some((floor, _)) if floor < max.wrapping_neg() => out_of_range(op, false)?,
        Some((floor, true)) => Comparison::Literal(op, decimal_scalar(floor, &data_type)?),
        // The operand lies strictly between `floor` and the next unit.
        Some((floor, false)) => match op {
            Eq => Comparison::Constant(false),
            NotEq => Comparison::Constant(true),
            Lt | LtEq => Comparison::Literal(LtEq, decimal_scalar(floor, &data_type)?),
            Gt | GtEq => Comparison::Literal(Gt, decimal_scalar(floor, &data_type)?),
            _ => return None,
        },
    };
    // A comparison the existing coercion turns into this exact literal keeps
    // its current shape.
    let needs_rewrite = match &comparison {
        Comparison::Literal(lowered_op, value) => {
            *lowered_op != op || safe_coerce_scalar(&scalar, &data_type).as_ref() != Some(value)
        }
        Comparison::Constant(_) => true,
    };
    Some(Lowered {
        column: column.clone(),
        data_type,
        comparison,
        needs_rewrite,
    })
}

/// Rewrites integer and decimal column comparisons against operands the
/// existing coercion rejects or rounds: floats for integer columns, and
/// integers and decimals whose exact value differs from what the coercion
/// produces for decimal columns. Returns `None` for any other filter, so it
/// keeps its current shape.
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
            if !lowered.needs_rewrite {
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
            let low_lowered = lower(column, Operator::GtEq, low, schema);
            let high_lowered = lower(column, Operator::LtEq, high, schema);
            if ![&low_lowered, &high_lowered]
                .iter()
                .any(|bound| bound.as_ref().is_some_and(|bound| bound.needs_rewrite))
            {
                return None;
            }
            let data_type = resolve_column_type(column, schema)?;
            // On a decimal column a bound that is not a literal is coerced the
            // way the existing coercion coerces it; leaving the rewrite there
            // would round the other bound. Other columns keep their behavior.
            let bound = |lowered: Option<Lowered>, op: Operator, operand: &Expr| match lowered {
                Some(lowered) => {
                    comparison_expr(&lowered.column, &lowered.data_type, lowered.comparison)
                }
                None if is_decimal(&data_type) => Some(Expr::BinaryExpr(BinaryExpr::new(
                    Box::new((**column).clone()),
                    op,
                    Box::new(coerce_expr(operand, &data_type).ok()?),
                ))),
                None => None,
            };
            let low = bound(low_lowered, Operator::GtEq, low)?;
            let high = bound(high_lowered, Operator::LtEq, high)?;
            Some(negate(low.and(high), *negated))
        }
        Expr::InList(in_list) => {
            let items = in_list
                .list
                .iter()
                .map(|item| (item, lower(&in_list.expr, Operator::Eq, item, schema)))
                .collect::<Vec<_>>();
            if !items
                .iter()
                .any(|(_, lowered)| lowered.as_ref().is_some_and(|item| item.needs_rewrite))
            {
                return None;
            }
            let column = in_list.expr.as_ref().clone();
            let data_type = resolve_column_type(&column, schema)?;
            // Values that can never match are dropped; NULLs stay so the
            // result remains NULL for unmatched rows. On a decimal column,
            // items that are not literals are left as the existing coercion
            // leaves them, which coerces them only for a top-level column.
            // Other columns keep their behavior.
            let mut values = Vec::with_capacity(items.len());
            for (item, lowered) in items {
                match lowered.map(|lowered| lowered.comparison) {
                    Some(Comparison::Literal(_, value)) => values.push(lit(value)),
                    Some(Comparison::Constant(_)) => {}
                    None if !is_decimal(&data_type) => return None,
                    None if matches!(column, Expr::Column(_)) => {
                        values.push(coerce_expr(item, &data_type).ok()?)
                    }
                    None => values.push(item.clone()),
                }
            }
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
    use arrow_array::{
        ArrayRef, BooleanArray, Decimal128Array, Decimal256Array, RecordBatch, cast::AsArray,
        record_batch,
    };
    use datafusion::prelude::col;
    use rstest::rstest;
    use std::sync::Arc;

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
    #[case::negative_nan_lt("i < CAST('-NaN' AS DOUBLE)", 0, Some(true))]
    #[case::negative_nan_lte("i <= CAST('-NaN' AS DOUBLE)", 0, Some(true))]
    #[case::negative_nan_gt("i > CAST('-NaN' AS DOUBLE)", 0, Some(false))]
    #[case::negative_nan_gte("i >= CAST('-NaN' AS DOUBLE)", 0, Some(false))]
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

    /// Mixing a float with a column operand on an integer column keeps the
    /// existing error; only decimal columns lower such lists.
    #[rstest]
    #[case::between("i BETWEEN 1.5 AND u")]
    #[case::in_list("i IN (1.5, u)")]
    fn integer_column_with_float_and_column_operands(#[case] sql: &str) {
        let batch = record_batch!(("i", Int64, [Some(1)]), ("u", UInt8, [Some(1)])).unwrap();
        let planner = Planner::new(batch.schema());
        let error = planner.parse_filter(sql).unwrap_err();
        assert!(
            matches!(error, lance_core::Error::InvalidInput { .. })
                && error.to_string().contains("could not convert"),
            "{sql}: {error}"
        );
    }

    /// Numbers no 256-bit decimal can hold exactly stay floats and keep the
    /// existing error, as the filter guide documents.
    #[rstest]
    #[case::too_many_fraction_digits("d > 1e-80")]
    #[case::exponent_past_scale("d < 1e200")]
    fn decimal_column_with_unrepresentable_numbers(#[case] sql: &str) {
        let planner = Planner::new(decimal_batch(&PRICE, 0).schema());
        let error = planner.parse_filter(sql).unwrap_err();
        assert!(
            matches!(error, lance_core::Error::InvalidInput { .. })
                && error.to_string().contains("could not convert"),
            "{sql}: {error}"
        );
    }

    /// `d` holds `units` of the type's scale and a NULL.
    fn decimal_batch(data_type: &DataType, units: i128) -> RecordBatch {
        let array: ArrayRef = match data_type {
            DataType::Decimal128(precision, scale) => Arc::new(
                Decimal128Array::from(vec![Some(units), None])
                    .with_precision_and_scale(*precision, *scale)
                    .unwrap(),
            ),
            DataType::Decimal256(precision, scale) => Arc::new(
                Decimal256Array::from(vec![Some(i256::from_i128(units)), None])
                    .with_precision_and_scale(*precision, *scale)
                    .unwrap(),
            ),
            _ => unreachable!(),
        };
        RecordBatch::try_from_iter([("d", array)]).unwrap()
    }

    const PRICE: DataType = DataType::Decimal128(15, 2);
    const FINE: DataType = DataType::Decimal128(38, 18);
    const WIDE: DataType = DataType::Decimal256(76, 20);
    const WHOLE: DataType = DataType::Decimal128(20, 0);

    #[rstest]
    #[case::eq(PRICE, "d = 0.05", 5, Some(true))]
    #[case::eq_more_digits(PRICE, "d = 0.050", 5, Some(true))]
    #[case::eq_between_units(PRICE, "d = 0.055", 5, Some(false))]
    #[case::ne_between_units(PRICE, "d != 0.055", 5, Some(true))]
    #[case::lt_between_units(PRICE, "d < 0.055", 5, Some(true))]
    #[case::lt_between_units_above(PRICE, "d < 0.055", 6, Some(false))]
    #[case::gt_between_units(PRICE, "d > 0.055", 6, Some(true))]
    #[case::gte_between_units(PRICE, "d >= 0.055", 5, Some(false))]
    #[case::reversed(PRICE, "0.055 > d", 5, Some(true))]
    #[case::negative(PRICE, "d < -0.005", -1, Some(true))]
    #[case::negative_zero(PRICE, "d < -0.005", 0, Some(false))]
    #[case::negative_gt(PRICE, "d > -0.005", 0, Some(true))]
    #[case::exponent(PRICE, "d = 5e-2", 5, Some(true))]
    #[case::exponent_whole(PRICE, "d = 1.5E1", 1500, Some(true))]
    #[case::leading_dot(PRICE, "d = .05", 5, Some(true))]
    #[case::q6_between(PRICE, "d BETWEEN 0.05 AND 0.07", 7, Some(true))]
    #[case::q6_between_below(PRICE, "d BETWEEN 0.05 AND 0.07", 4, Some(false))]
    #[case::between_between_units(PRICE, "d BETWEEN 0.051 AND 0.07", 5, Some(false))]
    #[case::not_between(PRICE, "d NOT BETWEEN 0.051 AND 0.07", 5, Some(true))]
    #[case::in_list(PRICE, "d IN (0.055, 0.05)", 5, Some(true))]
    #[case::never_in(PRICE, "d IN (0.055)", 5, Some(false))]
    #[case::not_in(PRICE, "d NOT IN (0.055)", 5, Some(true))]
    #[case::null_in(PRICE, "d IN (0.055, NULL)", 5, None)]
    #[case::integer(PRICE, "d < 24", 2399, Some(true))]
    #[case::max(PRICE, "d <= 9999999999999.99", 999_999_999_999_999, Some(true))]
    #[case::above_precision(PRICE, "d < 10000000000000.001", 999_999_999_999_999, Some(true))]
    #[case::above_precision_eq(PRICE, "d = 10000000000000.5", 0, Some(false))]
    #[case::below_precision(PRICE, "d > -10000000000000.5", -999_999_999_999_999, Some(true))]
    #[case::above_every_decimal(PRICE, "d < 1e100", 0, Some(true))]
    #[case::below_one_unit(PRICE, "d > 0.001", 0, Some(false))]
    #[case::below_one_unit_gt(PRICE, "d > 0.001", 1, Some(true))]
    #[case::nested(PRICE, "d = (0.05)", 5, Some(true))]
    #[case::cast(PRICE, "d < CAST('0.055' AS DECIMAL(5, 3))", 5, Some(true))]
    #[case::float_null(PRICE, "d = CAST(NULL AS DOUBLE)", 5, None)]
    #[case::fine_unit(
        FINE,
        "d = 1.000000000000000001",
        1_000_000_000_000_000_001,
        Some(true)
    )]
    #[case::fine_not_rounded(
        FINE,
        "d = 1.000000000000000001",
        1_000_000_000_000_000_000,
        Some(false)
    )]
    #[case::wide(WIDE, "d > 0.5", 50_000_000_000_000_000_001, Some(true))]
    #[case::wide_between_units(WIDE, "d < 0.000000000000000000005", 0, Some(true))]
    #[case::unsigned_integer(
        WHOLE,
        "d = 12345678901234567890",
        12_345_678_901_234_567_890,
        Some(true)
    )]
    #[case::negative_past_i64(WHOLE, "d > -9223372036854775809", -9_223_372_036_854_775_808, Some(true))]
    #[case::unsigned_integer_scaled(FINE, "d < 18446744073709551615", 1, Some(true))]
    fn decimal_predicates(
        #[case] data_type: DataType,
        #[case] sql: &str,
        #[case] units: i128,
        #[case] expected: Option<bool>,
    ) {
        let batch = decimal_batch(&data_type, units);
        let planner = Planner::new(batch.schema());
        let expr = planner.optimize_expr(planner.parse_filter(sql).unwrap());
        let result = planner.create_physical_expr(&expr.unwrap()).unwrap();
        let result = result.evaluate(&batch).unwrap().into_array(2).unwrap();
        let expected = BooleanArray::from(vec![expected, None]);
        assert_eq!(result.as_boolean(), &expected, "{sql}");
    }

    /// A `BETWEEN` bound or `IN` item that is not a literal keeps the exact
    /// lowering of the literal ones. `d` holds `units` hundredths and a NULL,
    /// `e` holds `other` hundredths.
    #[rstest]
    #[case::between_low("d BETWEEN 0.051 AND e", 5, 10, Some(false))]
    #[case::between_low_above("d BETWEEN 0.051 AND e", 6, 10, Some(true))]
    #[case::between_high("d BETWEEN e AND 0.055", 6, 0, Some(false))]
    #[case::not_between("d NOT BETWEEN 0.051 AND e", 5, 10, Some(true))]
    #[case::in_list("d IN (0.055, e)", 6, 1, Some(false))]
    #[case::in_list_column("d IN (0.055, e)", 1, 1, Some(true))]
    #[case::not_in("d NOT IN (0.055, e)", 6, 1, Some(true))]
    fn decimal_predicates_with_columns(
        #[case] sql: &str,
        #[case] units: i128,
        #[case] other: i128,
        #[case] expected: Option<bool>,
    ) {
        let column = |values: Vec<Option<i128>>| -> ArrayRef {
            Arc::new(
                Decimal128Array::from(values)
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
            )
        };
        let batch = RecordBatch::try_from_iter([
            ("d", column(vec![Some(units), None])),
            ("e", column(vec![Some(other), Some(other)])),
        ])
        .unwrap();
        let planner = Planner::new(batch.schema());
        let expr = planner.optimize_expr(planner.parse_filter(sql).unwrap());
        let result = planner.create_physical_expr(&expr.unwrap()).unwrap();
        let result = result.evaluate(&batch).unwrap().into_array(2).unwrap();
        let expected = BooleanArray::from(vec![expected, None]);
        assert_eq!(result.as_boolean(), &expected, "{sql}");
    }

    /// The same lowering on a decimal field of a struct, whose non-literal `IN`
    /// items are left as they are. `s.d` holds `units` hundredths and a NULL,
    /// `s.e` holds `other` hundredths.
    #[rstest]
    #[case::eq("s.d = 0.05", 5, 0, Some(true))]
    #[case::between_units("s.d < 0.055", 6, 0, Some(false))]
    #[case::in_list("s.d IN (0.055, s.e)", 6, 1, Some(false))]
    #[case::in_list_field("s.d IN (0.055, s.e)", 1, 1, Some(true))]
    #[case::in_list_literal("s.d IN (0.05, s.e)", 5, 1, Some(true))]
    fn nested_decimal_predicates(
        #[case] sql: &str,
        #[case] units: i128,
        #[case] other: i128,
        #[case] expected: Option<bool>,
    ) {
        let column = |values: Vec<Option<i128>>| -> ArrayRef {
            Arc::new(
                Decimal128Array::from(values)
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
            )
        };
        let field = |name: &str| Arc::new(arrow_schema::Field::new(name, PRICE, true));
        let struct_array = arrow_array::StructArray::from(vec![
            (field("d"), column(vec![Some(units), None])),
            (field("e"), column(vec![Some(other), Some(other)])),
        ]);
        let batch =
            RecordBatch::try_from_iter([("s", Arc::new(struct_array) as ArrayRef)]).unwrap();
        let planner = Planner::new(batch.schema());
        let expr = planner.optimize_expr(planner.parse_filter(sql).unwrap());
        let result = planner.create_physical_expr(&expr.unwrap()).unwrap();
        let result = result.evaluate(&batch).unwrap().into_array(2).unwrap();
        let expected = BooleanArray::from(vec![expected, None]);
        assert_eq!(result.as_boolean(), &expected, "{sql}");
    }

    /// Comparisons stay on the column with a literal of its type, so a scalar
    /// index on it can answer them.
    #[rstest]
    #[case::eq("d = 0.05", col("d").eq(lit(ScalarValue::Decimal128(Some(5), 15, 2))))]
    #[case::between_units("d < 0.055", col("d").lt_eq(lit(ScalarValue::Decimal128(Some(5), 15, 2))))]
    #[case::q6_between(
        "d BETWEEN 0.05 AND 0.07",
        col("d").between(
            lit(ScalarValue::Decimal128(Some(5), 15, 2)),
            lit(ScalarValue::Decimal128(Some(7), 15, 2))
        )
    )]
    #[case::between_bound_between_units(
        "d BETWEEN 0.051 AND 0.07",
        col("d")
            .gt(lit(ScalarValue::Decimal128(Some(5), 15, 2)))
            .and(col("d").lt_eq(lit(ScalarValue::Decimal128(Some(7), 15, 2))))
    )]
    #[case::in_list(
        "d IN (0.055, 0.05)",
        col("d").in_list(vec![lit(ScalarValue::Decimal128(Some(5), 15, 2))], false)
    )]
    #[case::never("d = 0.055", col("d").lt(lit(ScalarValue::Decimal128(Some(-999_999_999_999_999), 15, 2))))]
    #[case::cast(
        "d < CAST('0.055' AS DECIMAL(5, 3))",
        col("d").lt_eq(lit(ScalarValue::Decimal128(Some(5), 15, 2)))
    )]
    #[case::integer_above_precision(
        "d < 10000000000000",
        col("d").gt_eq(lit(ScalarValue::Decimal128(Some(-999_999_999_999_999), 15, 2)))
    )]
    fn plans_decimal_on_column(#[case] sql: &str, #[case] expected: Expr) {
        let planner = Planner::new(decimal_batch(&PRICE, 0).schema());
        assert_eq!(planner.parse_filter(sql).unwrap(), expected, "{sql}");
    }
}
