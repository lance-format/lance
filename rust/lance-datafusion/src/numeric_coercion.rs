// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index-friendly comparisons of integer and decimal columns with numeric
//! operands.
//!
//! [`rewrite`] compares an integer or decimal column with a numeric operand by
//! exact mathematical value, rather than rounding the operand or the column to
//! a float. [`unwrap_widening_casts`] handles a column that is explicitly cast
//! to a float first, and keeps the rounding of that cast.

use arrow_buffer::i256;
use arrow_schema::{DECIMAL128_MAX_PRECISION, DataType};
use datafusion::common::DFSchema;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::{Between, BinaryExpr, ExprSchemable, Operator};
use datafusion::prelude::{Expr, lit};
use datafusion::scalar::ScalarValue;
use lance_core::Result;
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

/// A column read through a cast that widens it to a float type.
struct WidenedColumn<'a> {
    column: &'a Expr,
    column_type: DataType,
    /// `Float32` or `Float64`; for a `Float32` column, always `Float64`.
    float_type: DataType,
}

fn is_field_reference(expr: &Expr) -> bool {
    match expr {
        Expr::Column(_) => true,
        Expr::ScalarFunction(udf) if udf.name() == "get_field" => {
            udf.args.first().is_some_and(is_field_reference)
        }
        _ => false,
    }
}

fn widened_column<'a>(expr: &'a Expr, schema: &DFSchema) -> Option<WidenedColumn<'a>> {
    // An integer or `Float32` never fails to convert to a float, so `TRY_CAST`
    // is the same function as `CAST` here.
    let (column, float_type) = match expr {
        Expr::Cast(cast) => (cast.expr.as_ref(), cast.field.data_type()),
        Expr::TryCast(cast) => (cast.expr.as_ref(), cast.field.data_type()),
        _ => return None,
    };
    if !is_field_reference(column) {
        return None;
    }
    let column_type = column.get_type(schema).ok()?;
    let is_widening = match (&column_type, float_type) {
        (column_type, DataType::Float32 | DataType::Float64) if column_type.is_integer() => true,
        (DataType::Float32, DataType::Float64) => true,
        _ => false,
    };
    is_widening.then(|| WidenedColumn {
        column,
        column_type,
        float_type: float_type.clone(),
    })
}

/// The literal operand as an `f64`, `Some(None)` for NULL, or `None` for
/// anything else. Only a literal of the cast's own type is taken, which is what
/// coercion leaves next to the cast.
fn widened_literal(expr: &Expr, float_type: &DataType) -> Option<Option<f64>> {
    let value = extract_literal(expr)?;
    if value.data_type() != *float_type {
        return None;
    }
    match value.cast_to(&DataType::Float64).ok()? {
        ScalarValue::Float64(value) => Some(value),
        _ => None,
    }
}

/// The integers `start..end` matching a comparison, or all others when
/// `negated`.
struct Interval {
    start: i128,
    end: i128,
    negated: bool,
}

/// The first integer in `start..end` for which `holds` is true, or `end`.
/// `holds` has to be false up to some point and true from there on.
fn partition_point(mut start: i128, mut end: i128, holds: impl Fn(i128) -> bool) -> i128 {
    while start < end {
        let mid = start + (end - start) / 2;
        if holds(mid) {
            end = mid;
        } else {
            start = mid + 1;
        }
    }
    start
}

/// The integers whose conversion to `float_type` compares with `value` like
/// `op`.
///
/// Rounding an integer to a float is monotonic, so the integers that compare
/// above (or not below) `value` are a suffix of the domain and each comparison
/// selects one interval. Comparisons against NaN are false, which places NaN
/// above every integer, as float filters do regardless of its sign bit.
fn integer_interval(
    op: Operator,
    value: f64,
    float_type: &DataType,
    (min, end): (i128, i128),
) -> Option<Interval> {
    let round = |integer: i128| match float_type {
        DataType::Float32 => f64::from(integer as f32),
        _ => integer as f64,
    };
    let first_not_below = || partition_point(min, end, |integer| round(integer) >= value);
    let first_above = || partition_point(min, end, |integer| round(integer) > value);
    let interval = |start, end| Interval {
        start,
        end,
        negated: false,
    };
    Some(match op {
        Operator::Gt => interval(first_above(), end),
        Operator::GtEq => interval(first_not_below(), end),
        Operator::Lt => interval(min, first_not_below()),
        Operator::LtEq => interval(min, first_above()),
        Operator::Eq | Operator::NotEq => Interval {
            start: first_not_below(),
            end: first_above(),
            negated: op == Operator::NotEq,
        },
        _ => return None,
    })
}

fn integer_domain(data_type: &DataType) -> Option<(i128, i128)> {
    let bits = data_type.primitive_width()? as u32 * 8;
    Some(if data_type.is_signed_integer() {
        (-(1_i128 << (bits - 1)), 1_i128 << (bits - 1))
    } else {
        (0, 1_i128 << bits)
    })
}

fn integer_literal(value: i128, data_type: &DataType) -> Option<Expr> {
    let value = match i64::try_from(value) {
        Ok(value) => ScalarValue::Int64(Some(value)),
        Err(_) => ScalarValue::UInt64(Some(u64::try_from(value).ok()?)),
    };
    Some(lit(value.cast_to(data_type).ok()?))
}

fn interval_expr(column: &Expr, data_type: &DataType, interval: Interval) -> Option<Expr> {
    let (min, domain_end) = integer_domain(data_type)?;
    let Interval {
        start,
        end,
        negated,
    } = interval;
    let compare = |op, value| -> Option<Expr> {
        Some(Expr::BinaryExpr(BinaryExpr::new(
            Box::new(column.clone()),
            op,
            Box::new(integer_literal(value, data_type)?),
        )))
    };
    let constant = |value| comparison_expr(column, data_type, Comparison::Constant(value));
    let matches = if start >= end {
        constant(false)?
    } else {
        match (start == min, end == domain_end) {
            (true, true) => constant(true)?,
            (true, false) => compare(Operator::Lt, end)?,
            (false, true) => compare(Operator::GtEq, start)?,
            (false, false) if end == start + 1 => compare(Operator::Eq, start)?,
            (false, false) => compare(Operator::GtEq, start)?.and(compare(Operator::Lt, end)?),
        }
    };
    Some(if negated { !matches } else { matches })
}

/// `value` as a `Float32` when that is exact, or else the largest `Float32`
/// below it.
fn float32_at_or_below(value: f64) -> (f32, bool) {
    let nearest = value as f32;
    match f64::from(nearest).partial_cmp(&value) {
        Some(std::cmp::Ordering::Equal) => (nearest, true),
        Some(std::cmp::Ordering::Less) => (nearest, false),
        _ => (nearest.next_down(), false),
    }
}

/// `column op value` for a column that the comparison reads widened to a
/// float, without the cast.
fn unwrap_comparison(widened: &WidenedColumn, op: Operator, value: Option<f64>) -> Option<Expr> {
    let WidenedColumn {
        column,
        column_type,
        float_type,
    } = widened;
    let Some(value) = value else {
        let null = ScalarValue::try_new_null(column_type).ok()?;
        return comparison_expr(column, column_type, Comparison::Literal(op, null));
    };
    if column_type.is_integer() {
        let interval = integer_interval(op, value, float_type, integer_domain(column_type)?)?;
        return interval_expr(column, column_type, interval);
    }
    // A float comparison against NaN also orders NaN payloads, which a Float32
    // bound would not keep.
    if value.is_nan() {
        return None;
    }
    let (below, is_exact) = float32_at_or_below(value);
    let comparison = if is_exact {
        Comparison::Literal(op, ScalarValue::Float32(Some(below)))
    } else {
        // Between `below` and the next Float32 there is no Float32, so `value`
        // can be replaced by `below` once `<` and `>=` include it.
        let below = ScalarValue::Float32(Some(below));
        match op {
            Operator::Gt | Operator::GtEq => Comparison::Literal(Operator::Gt, below),
            Operator::Lt | Operator::LtEq => Comparison::Literal(Operator::LtEq, below),
            Operator::Eq => Comparison::Constant(false),
            Operator::NotEq => Comparison::Constant(true),
            _ => return None,
        }
    };
    comparison_expr(column, column_type, comparison)
}

/// The `IN` list values of a widened column, or `None` when a value matches
/// more than one column value or cannot be unwrapped.
fn unwrap_in_list(widened: &WidenedColumn, values: &[Option<f64>]) -> Option<Vec<Expr>> {
    let mut list = Vec::with_capacity(values.len());
    for value in values {
        let Some(value) = *value else {
            list.push(lit(ScalarValue::try_new_null(&widened.column_type).ok()?));
            continue;
        };
        if widened.column_type.is_integer() {
            let domain = integer_domain(&widened.column_type)?;
            let interval = integer_interval(Operator::Eq, value, &widened.float_type, domain)?;
            match interval.end - interval.start {
                // No integer rounds to this value.
                ..=0 => {}
                1 => list.push(integer_literal(interval.start, &widened.column_type)?),
                _ => return None,
            }
        } else {
            if value.is_nan() {
                return None;
            }
            // An inexact value matches no Float32 and is dropped.
            if let (value, true) = float32_at_or_below(value) {
                list.push(lit(ScalarValue::Float32(Some(value))));
            }
        }
    }
    Some(list)
}

fn unwrap_node(expr: &Expr, schema: &DFSchema) -> Option<Expr> {
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
            let (widened, op, literal) = match widened_column(left, schema) {
                Some(widened) => (widened, *op, right),
                None => (widened_column(right, schema)?, op.swap()?, left),
            };
            let value = widened_literal(literal, &widened.float_type)?;
            unwrap_comparison(&widened, op, value)
        }
        Expr::Between(Between {
            expr,
            negated,
            low,
            high,
        }) => {
            let widened = widened_column(expr, schema)?;
            let low = widened_literal(low, &widened.float_type)?;
            let high = widened_literal(high, &widened.float_type)?;
            let between = unwrap_comparison(&widened, Operator::GtEq, low)?.and(unwrap_comparison(
                &widened,
                Operator::LtEq,
                high,
            )?);
            Some(if *negated { !between } else { between })
        }
        Expr::InList(InList {
            expr,
            list,
            negated,
        }) => {
            let widened = widened_column(expr, schema)?;
            let values = list
                .iter()
                .map(|item| widened_literal(item, &widened.float_type))
                .collect::<Option<Vec<_>>>()?;
            let Some(values) = unwrap_in_list(&widened, &values) else {
                // A value that several integers round to needs a range. The
                // simplifier turns a short list into this same `OR` chain anyway,
                // and leaving the list alone would only unwrap it there, on the
                // next pass.
                let matches = values
                    .into_iter()
                    .map(|value| unwrap_comparison(&widened, Operator::Eq, value))
                    .reduce(|left, right| Some(left?.or(right?)))??;
                return Some(if *negated { !matches } else { matches });
            };
            if values.is_empty() {
                return comparison_expr(
                    widened.column,
                    &widened.column_type,
                    Comparison::Constant(*negated),
                );
            }
            Some(widened.column.clone().in_list(values, *negated))
        }
        _ => None,
    }
}

/// Rewrites comparisons of a column widened to a float against a literal, such
/// as `CAST(i AS double) > 2.5` or `TRY_CAST(f32 AS double) = 0.1`, into
/// comparisons on the column itself, so that its scalar index applies.
///
/// The result keeps the semantics of the cast: an integer is compared after
/// rounding it to the float type, so `CAST(i AS double) = 9007199254740992.0`
/// also matches `i = 9007199254740993`. A `Float32` widens exactly, and a bound
/// it cannot represent is replaced by the nearest `Float32` below it.
///
/// Expects coerced input, in which a `Float32` column compared with a `Float64`
/// literal carries the cast, and has to run before the float comparison rewrite
/// turns these comparisons into ranges on the cast.
pub fn unwrap_widening_casts(expr: Expr, schema: &DFSchema) -> Result<Expr> {
    Ok(expr
        .transform_up(|node| {
            Ok(match unwrap_node(&node, schema) {
                Some(rewritten) => Transformed::yes(rewritten),
                None => Transformed::no(node),
            })
        })?
        .data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::Planner;
    use arrow_array::{
        ArrayRef, BooleanArray, Decimal128Array, Decimal256Array, Float32Array, Int64Array,
        RecordBatch, UInt64Array, cast::AsArray, record_batch,
    };
    use datafusion::prelude::col;
    use datafusion_common::tree_node::TreeNodeRecursion;
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

    const COMPARISONS: [&str; 6] = ["=", "!=", "<", "<=", ">", ">="];

    fn compare(x: f64, op: &str, value: f64) -> bool {
        match op {
            "=" => x == value,
            "!=" => x != value,
            "<" => x < value,
            "<=" => x <= value,
            ">" => x > value,
            _ => x >= value,
        }
    }

    fn float_sql(value: f64, sql_type: &str) -> String {
        let text = match value {
            value if value.is_nan() && value.is_sign_negative() => "'-NaN'".to_string(),
            value if value.is_nan() => "'NaN'".to_string(),
            f64::INFINITY => "'inf'".to_string(),
            f64::NEG_INFINITY => "'-inf'".to_string(),
            value => format!("{value:?}"),
        };
        format!("CAST({text} AS {sql_type})")
    }

    fn has_cast(expr: &Expr) -> bool {
        let mut found = false;
        expr.apply(|node| {
            found |= matches!(node, Expr::Cast(_) | Expr::TryCast(_));
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        found
    }

    fn evaluate(batch: &RecordBatch, sql: &str) -> (Expr, BooleanArray) {
        let planner = Planner::new(batch.schema());
        let expr = planner
            .optimize_expr(planner.parse_filter(sql).unwrap())
            .unwrap();
        // The scan path optimizes a filter twice.
        assert_eq!(planner.optimize_expr(expr.clone()).unwrap(), expr, "{sql}");
        let result = planner.create_physical_expr(&expr).unwrap();
        let result = result.evaluate(batch).unwrap();
        let result = result.into_array(batch.num_rows()).unwrap();
        (expr, result.as_boolean().clone())
    }

    fn integers(data_type: &DataType) -> ArrayRef {
        let two_53 = 1_i128 << 53;
        let values = [
            i128::from(i64::MIN),
            -two_53 - 1,
            -two_53,
            -1,
            0,
            1,
            2,
            3,
            (1 << 24) + 1,
            two_53 - 1,
            two_53,
            two_53 + 1,
            two_53 + 2,
            two_53 + 3,
            i128::from(i64::MAX),
            i128::from(u64::MAX),
        ];
        let array: ArrayRef = if data_type.is_signed_integer() {
            let values = values.iter().filter_map(|v| i64::try_from(*v).ok());
            Arc::new(Int64Array::from_iter(values.map(Some).chain([None])))
        } else {
            let values = values.iter().filter_map(|v| u64::try_from(*v).ok());
            Arc::new(UInt64Array::from_iter(values.map(Some).chain([None])))
        };
        arrow_cast::cast(&array, data_type).unwrap()
    }

    /// Every comparison of an integer cast to a float gives the rows that
    /// comparing the rounded value does, and plans on the column.
    #[rstest]
    fn widened_integers_compare_rounded(
        #[values(DataType::Int64, DataType::UInt64, DataType::Int32, DataType::Int8)]
        data_type: DataType,
        #[values("CAST(i AS double)", "TRY_CAST(i AS double)", "CAST(i AS float)")] operand: &str,
    ) {
        let column = integers(&data_type);
        let batch = RecordBatch::try_from_iter([("i", column.clone())]).unwrap();
        let is_float32 = operand.ends_with("float)");
        let column = arrow_cast::cast(
            &column,
            if is_float32 {
                &DataType::Float32
            } else {
                &DataType::Float64
            },
        )
        .unwrap();
        let column = arrow_cast::cast(&column, &DataType::Float64).unwrap();
        let rounded = column.as_primitive::<arrow_array::types::Float64Type>();
        let two_53 = 2_f64.powi(53);
        for value in [
            f64::NEG_INFINITY,
            -1e19,
            -9223372036854775808.0,
            -two_53,
            -129.0,
            -2.5,
            -0.0,
            0.0,
            0.5,
            1.0,
            2.5,
            16777218.0,
            two_53,
            two_53 + 2.0,
            9223372036854775808.0,
            18446744073709551616.0,
            1e300,
            f64::INFINITY,
            f64::NAN,
            -f64::NAN,
        ] {
            let value = if is_float32 {
                f64::from(value as f32)
            } else {
                value
            };
            let literal = float_sql(value, if is_float32 { "float" } else { "double" });
            for op in COMPARISONS {
                let sql = format!("{operand} {op} {literal}");
                let (expr, result) = evaluate(&batch, &sql);
                assert!(!has_cast(&expr), "{sql} kept a cast: {expr}");
                // Float filters place every NaN above every number.
                let holds = |x: f64| {
                    if value.is_nan() {
                        matches!(op, "!=" | "<" | "<=")
                    } else {
                        compare(x, op, value)
                    }
                };
                let expected = rounded
                    .iter()
                    .map(|x| x.map(holds))
                    .collect::<BooleanArray>();
                assert_eq!(result, expected, "{sql} on {data_type}");
            }
        }
    }

    /// Every comparison of a `Float32` widened to `Float64`, written as a cast
    /// on the column or as a typed literal, gives the rows that comparing the
    /// widened value does, and plans on the column.
    #[rstest]
    fn widened_float32_compares_exactly(
        #[values(
            "CAST(f AS double) {op} {literal}",
            "TRY_CAST(f AS double) {op} {literal}",
            "f {op} {literal}"
        )]
        template: &str,
    ) {
        let one_tenth = 0.1_f32;
        let values = [
            f32::NEG_INFINITY,
            -f32::MAX,
            -1.0,
            -f32::from_bits(1),
            -0.0,
            0.0,
            f32::from_bits(1),
            one_tenth,
            one_tenth.next_up(),
            0.5,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
            -f32::NAN,
        ];
        let column = Float32Array::from_iter(values.map(Some).into_iter().chain([None]));
        let batch =
            RecordBatch::try_from_iter([("f", Arc::new(column.clone()) as ArrayRef)]).unwrap();
        for value in [
            f64::NEG_INFINITY,
            -1e300,
            -1.0,
            -1e-50,
            -0.0,
            0.0,
            1e-50,
            0.1,
            f64::from(one_tenth),
            0.5,
            f64::from(f32::MAX),
            3.5e38,
            1e300,
            f64::INFINITY,
        ] {
            let literal = float_sql(value, "double");
            for op in COMPARISONS {
                let sql = template.replace("{op}", op).replace("{literal}", &literal);
                let (expr, result) = evaluate(&batch, &sql);
                assert!(!has_cast(&expr), "{sql} kept a cast: {expr}");
                // Float filters place every NaN above every number.
                let holds = |x: f32| {
                    if x.is_nan() {
                        matches!(op, "!=" | ">" | ">=")
                    } else {
                        compare(f64::from(x), op, value)
                    }
                };
                let expected = column
                    .iter()
                    .map(|x| x.map(holds))
                    .collect::<BooleanArray>();
                assert_eq!(result, expected, "{sql}");
            }
        }
    }

    #[rstest]
    #[case::between("CAST(i AS double) BETWEEN 1.5 AND 3.0", [false, false, true, true, false])]
    #[case::not_between("CAST(i AS double) NOT BETWEEN 1.5 AND 3.0", [true, true, false, false, true])]
    #[case::in_list("CAST(i AS double) IN (1.0, 2.5, 9007199254740994.0)", [false, true, false, false, true])]
    #[case::not_in_list("TRY_CAST(i AS double) NOT IN (1.0, 2.5)", [true, false, true, true, true])]
    #[case::never_in("CAST(i AS double) IN (2.5, CAST('NaN' AS double))", [false; 5])]
    // 2^53 is the rounding of two integers, which a list cannot name with one
    // value.
    #[case::rounded_in_list("CAST(i AS double) IN (9007199254740992.0)", [false; 5])]
    #[case::in_list_rounded("CAST(i AS double) IN (1.0, 9007199254740992.0)", [false, true, false, false, false])]
    #[case::not_in_list_rounded("CAST(i AS double) NOT IN (1.0, 9007199254740992.0)", [true, false, true, true, true])]
    #[case::reversed("2.5 < CAST(i AS double)", [false, false, false, true, true])]
    fn widened_integer_lists_and_ranges(#[case] sql: &str, #[case] expected: [bool; 5]) {
        let batch = record_batch!((
            "i",
            Int64,
            [
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(9007199254740994),
                None
            ]
        ))
        .unwrap();
        let (expr, result) = evaluate(&batch, sql);
        assert!(!has_cast(&expr), "{sql} kept a cast: {expr}");
        let expected = expected.map(Some).into_iter().chain([None]);
        let expected = expected.collect::<BooleanArray>();
        assert_eq!(result, expected, "{sql}");
    }

    #[rstest]
    #[case::fraction("CAST(i AS double) > 2.5", col("i").gt_eq(lit(3_i64)))]
    #[case::try_cast("TRY_CAST(i AS double) <= 2.5", col("i").lt(lit(3_i64)))]
    #[case::exact("CAST(i AS double) = 3.0", col("i").eq(lit(3_i64)))]
    #[case::not_equal("CAST(i AS double) != 3.0", col("i").not_eq(lit(3_i64)))]
    #[case::rounded(
        "CAST(i AS double) = 9007199254740992.0",
        col("i")
            .gt_eq(lit(9007199254740992_i64))
            .and(col("i").lt(lit(9007199254740994_i64)))
    )]
    #[case::never("CAST(u AS double) < -1.0", col("u").lt(lit(0_u8)))]
    #[case::always("CAST(u AS float) <= CAST(255.0 AS float)", col("u").gt_eq(lit(0_u8)))]
    fn widened_plans_on_column(#[case] sql: &str, #[case] expected: Expr) {
        let batch = record_batch!(("i", Int64, [Some(1)]), ("u", UInt8, [Some(1)])).unwrap();
        let planner = Planner::new(batch.schema());
        let expr = planner.optimize_expr(planner.parse_filter(sql).unwrap());
        assert_eq!(expr.unwrap(), expected, "{sql}");
    }
}
