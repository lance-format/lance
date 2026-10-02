// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Rewrites for signed-zero literals and NaN sign bits in comparisons.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{ArrayRef, ArrowNativeTypeOp, ArrowPrimitiveType, BooleanArray};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType;
use datafusion::error::Result as DFResult;
use datafusion::functions_nested::expr_fn::array_has_any;
use datafusion::logical_expr::expr::{Between, InList, ScalarFunction};
use datafusion::logical_expr::{
    BinaryExpr, ColumnarValue, ExprSchemable, Operator, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use datafusion::prelude::{Expr, lit};
use datafusion::scalar::ScalarValue::{self, Float16, Float32, Float64};
use datafusion_common::metadata::FieldMetadata;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::utils::take_function_args;
use datafusion_common::{DFSchema, exec_err};
use half::f16;
use lance_core::Result;

const COMPARE_FLOATS_NAME: &str = "_lance_compare_floats";
const RAW_FLOAT_LITERAL_MARKER: &str = "lance:raw-float-comparison";

/// The comparisons the rewrite acts on. [`CompareFloats`] names them in its
/// third argument by their `Display` form.
const FLOAT_COMPARISONS: [Operator; 8] = [
    Operator::Lt,
    Operator::LtEq,
    Operator::Gt,
    Operator::GtEq,
    Operator::Eq,
    Operator::NotEq,
    Operator::IsDistinctFrom,
    Operator::IsNotDistinctFrom,
];

/// `_lance_compare_floats(left, right, op)` compares two floats like `op`,
/// except that the sign of a zero or a NaN does not take part.
///
/// Comparisons against non-NaN literals are rewritten into indexable ranges
/// instead. This UDF is reserved for column-to-column, computed, and NaN-bound
/// comparisons, where no range can express the comparison. It clears the sign
/// bits in registers while comparing, so it allocates nothing but the result.
#[derive(Debug, Eq, PartialEq, Hash)]
struct CompareFloats {
    signature: Signature,
}

impl CompareFloats {
    fn new() -> Self {
        let exact = |t: DataType| TypeSignature::Exact(vec![t.clone(), t, DataType::Utf8]);
        Self {
            signature: Signature::one_of(
                vec![
                    exact(DataType::Float16),
                    exact(DataType::Float32),
                    exact(DataType::Float64),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for CompareFloats {
    fn name(&self) -> &str {
        COMPARE_FLOATS_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(&self, func_args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let [left, right, op_name] = take_function_args(self.name(), func_args.args)?;
        let op = match &op_name {
            ColumnarValue::Scalar(ScalarValue::Utf8(Some(name))) => FLOAT_COMPARISONS
                .into_iter()
                .find(|op| op.to_string() == *name),
            _ => None,
        };
        let Some(op) = op else {
            return exec_err!("{COMPARE_FLOATS_NAME} got an unsupported operator {op_name:?}");
        };
        // Only the right operand is ever a literal, a NaN bound, and it stays a
        // one-element array instead of being broadcast.
        let right = match right {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(value) => value.to_array()?,
        };
        let left = left.into_array(func_args.number_rows)?;
        let result = compare_float_arrays(&left, &right, op, func_args.number_rows)?;
        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

/// Compare `num_rows` rows, reading a one-element `right` as a scalar.
fn compare_float_arrays(
    left: &ArrayRef,
    right: &ArrayRef,
    op: Operator,
    num_rows: usize,
) -> DFResult<BooleanArray> {
    match left.data_type() {
        DataType::Float16 => Ok(compare_floats::<Float16Type>(left, right, op, num_rows)),
        DataType::Float32 => Ok(compare_floats::<Float32Type>(left, right, op, num_rows)),
        DataType::Float64 => Ok(compare_floats::<Float64Type>(left, right, op, num_rows)),
        data_type => {
            exec_err!("{COMPARE_FLOATS_NAME} expected floating-point operands, got {data_type}")
        }
    }
}

fn compare_floats<T: ArrowPrimitiveType>(
    left: &ArrayRef,
    right: &ArrayRef,
    op: Operator,
    num_rows: usize,
) -> BooleanArray {
    let canonical = clear_sign_of_zero_and_nan::<T::Native>;
    let (left_values, right_values) = (
        left.as_primitive::<T>().values(),
        right.as_primitive::<T>().values(),
    );
    // A scalar is canonicalized once, and each shape keeps its own loop so the
    // array-to-array one can vectorize.
    let right_is_scalar = right.len() != num_rows;
    let values = if right_is_scalar {
        let right = canonical(right_values[0]);
        collect_ordering(op, num_rows, |row| {
            canonical(left_values[row]).compare(right)
        })
    } else {
        collect_ordering(op, num_rows, |row| {
            canonical(left_values[row]).compare(canonical(right_values[row]))
        })
    };
    let left_nulls = left.nulls().cloned();
    let right_nulls = match right.nulls() {
        Some(nulls) if right_is_scalar => nulls.is_null(0).then(|| NullBuffer::new_null(num_rows)),
        nulls => nulls.cloned(),
    };
    if !matches!(op, Operator::IsDistinctFrom | Operator::IsNotDistinctFrom) {
        return BooleanArray::new(
            values,
            NullBuffer::union(left_nulls.as_ref(), right_nulls.as_ref()),
        );
    }
    // Two NULLs are not distinct, and a NULL is distinct from any value.
    let valid = |nulls: Option<NullBuffer>| {
        nulls.map_or_else(|| BooleanBuffer::new_set(num_rows), NullBuffer::into_inner)
    };
    let (left_valid, right_valid) = (valid(left_nulls), valid(right_nulls));
    let both_valid_values = &(&left_valid & &right_valid) & &values;
    BooleanArray::from(match op {
        Operator::IsNotDistinctFrom => &both_valid_values | &!&(&left_valid | &right_valid),
        _ => &both_valid_values | &(&left_valid ^ &right_valid),
    })
}

/// `value` with the sign bit cleared if it is a zero or a NaN.
///
/// A zero or NaN sorts below `+0.0` in total order exactly when its sign bit is
/// set, and negating it clears that bit while keeping a NaN's payload. Written
/// without short-circuiting so that the comparison loops stay branch-free.
#[inline(always)]
fn clear_sign_of_zero_and_nan<N: ArrowNativeTypeOp>(value: N) -> N {
    let zero = N::ZERO;
    let is_zero_or_nan = !((value < zero) | (value > zero));
    if is_zero_or_nan & value.is_lt(zero) {
        value.neg_wrapping()
    } else {
        value
    }
}

/// Evaluate `op` on the total-order `ordering` of each row's canonical operands.
fn collect_ordering(
    op: Operator,
    num_rows: usize,
    ordering: impl Fn(usize) -> Ordering,
) -> BooleanBuffer {
    match op {
        Operator::Lt => BooleanBuffer::collect_bool(num_rows, |row| ordering(row).is_lt()),
        Operator::LtEq => BooleanBuffer::collect_bool(num_rows, |row| ordering(row).is_le()),
        Operator::Gt => BooleanBuffer::collect_bool(num_rows, |row| ordering(row).is_gt()),
        Operator::GtEq => BooleanBuffer::collect_bool(num_rows, |row| ordering(row).is_ge()),
        Operator::Eq | Operator::IsNotDistinctFrom => {
            BooleanBuffer::collect_bool(num_rows, |row| ordering(row).is_eq())
        }
        _ => BooleanBuffer::collect_bool(num_rows, |row| ordering(row).is_ne()),
    }
}

/// Registered by the Substrait decoder, which resolves functions by name.
pub static COMPARE_FLOATS_UDF: LazyLock<Arc<ScalarUDF>> =
    LazyLock::new(|| Arc::new(ScalarUDF::new_from_impl(CompareFloats::new())));

fn compare_floats_expr(left: Expr, op: Operator, right: Expr) -> Expr {
    Expr::ScalarFunction(ScalarFunction::new_udf(
        COMPARE_FLOATS_UDF.clone(),
        vec![left, right, lit(op.to_string())],
    ))
}

/// Rewrite floating-point comparisons so sign-only encodings have value semantics.
///
/// Arrow sorts `-0.0` strictly below `+0.0` and compares the two encodings for
/// equality by bit pattern. It also sorts a NaN with its sign bit set below
/// negative infinity, while the same NaN without that bit sorts above positive
/// infinity. IEEE 754 and SQL do not make either value depend on its sign bit.
/// Literal comparisons have indexable equivalent total-order forms:
///
/// | written                    | evaluated                                      |
/// |----------------------------|------------------------------------------------|
/// | `x < 0`, `x >= 0`          | zero bound uses `-0.0`                          |
/// | `x <= 0`, `x > 0`          | zero bound uses `+0.0`                          |
/// | `x < c`, `x <= c`          | for non-NaN `c`, comparison plus `x >= -inf`    |
/// | `x > c`, `x >= c`          | for non-NaN `c`, comparison plus `x < -inf`     |
/// | `x = 0` / `x = NaN`        | `IN` both sign encodings                        |
/// | `x IN (0, NaN, ..)`        | missing sign encodings are added                |
/// | `array_has(xs, 0 / NaN)`   | `array_has_any` over both sign encodings        |
///
/// The extra NaN range is combined with `AND` for `<`/`<=` and `OR` for `>`/`>=`.
/// Equality names both encodings because scalar indices key on the bit pattern.
/// A comparison without a literal runs through an internal UDF that ignores the
/// sign of zeros and NaNs, preserving payload bits and evaluating each operand once.
///
/// Runs as the last step of [`crate::planner::Planner::optimize_expr`], after
/// coercion has given the literal the column's type and the simplifier has
/// expanded `BETWEEN` into two comparisons. Filters, computed output columns and
/// update expressions all compile through there, which is what keeps a filter and
/// a projected copy of the same predicate in agreement.
///
pub fn rewrite_float_comparisons(expr: Expr, schema: &DFSchema) -> Result<Expr> {
    Ok(expr
        .transform_up(|node| {
            Ok(match rewrite_node(&node, schema) {
                Some(rewritten) => Transformed::yes(rewritten),
                None => Transformed::no(node),
            })
        })?
        .data)
}

/// Fold each sign-sensitive comparison's own operands and rewrite it, bottom-up,
/// before anything above it has a chance to fold.
///
/// [`rewrite_float_comparisons`] alone cannot reach a comparison whose special
/// value does not exist yet. `ExprSimplifier::simplify` folds an operand and everything
/// above it in one pass, so `-1.0 * 0.0 < (1.0 - 1.0)` goes straight to a boolean
/// decided by Arrow's total order, and a wrapper like `IS TRUE` or a `CAST` around
/// it does the same to the comparison's own result.
///
/// Visiting bottom-up and folding only the operands of the node in hand is what
/// closes that: by the time any container is folded, every comparison inside it
/// already carries the corrected literal. This deliberately does not enumerate
/// which containers are allowed above a comparison. Enumerating them is what left
/// `IS TRUE`, `IS FALSE`, `= TRUE`, `CAST(.. AS BOOLEAN)` and `IN (TRUE)` exposed,
/// and any list would keep missing the next spelling.
pub fn normalize_float_comparisons(
    expr: Expr,
    simplify: &dyn Fn(Expr) -> DFResult<Expr>,
    schema: &DFSchema,
) -> Result<Expr> {
    // A literal is already folded, and an `IN` list can hold hundreds of them.
    // Handing each one to the simplifier anyway roughly doubled planning time on
    // large lists, for an operand that cannot change.
    let fold = |operand: Expr| -> DFResult<Expr> {
        if matches!(operand, Expr::Literal(..)) {
            return Ok(operand);
        }
        simplify(operand)
    };
    Ok(expr
        .transform_up(|node| {
            let folded = match node {
                Expr::BinaryExpr(BinaryExpr { left, op, right })
                    if FLOAT_COMPARISONS.contains(&op) =>
                {
                    Expr::BinaryExpr(BinaryExpr {
                        left: Box::new(fold(*left)?),
                        op,
                        right: Box::new(fold(*right)?),
                    })
                }
                Expr::Between(between) => Expr::Between(Between {
                    expr: Box::new(fold(*between.expr)?),
                    negated: between.negated,
                    low: Box::new(fold(*between.low)?),
                    high: Box::new(fold(*between.high)?),
                }),
                Expr::InList(in_list) => Expr::InList(InList {
                    expr: Box::new(fold(*in_list.expr)?),
                    list: in_list
                        .list
                        .into_iter()
                        .map(fold)
                        .collect::<DFResult<Vec<_>>>()?,
                    negated: in_list.negated,
                }),
                other => return Ok(Transformed::no(other)),
            };
            Ok(match rewrite_node(&folded, schema) {
                Some(rewritten) => Transformed::yes(rewritten),
                // The operands were still folded, so this is a change either way.
                None => Transformed::yes(folded),
            })
        })?
        .data)
}

/// The two sign encodings of zero or NaN, negative first.
///
/// NaN payload bits are preserved. Distinct payloads therefore remain distinct;
/// only the sign bit, which does not change the logical value, is ignored.
fn equivalent_encodings(value: &ScalarValue) -> Option<(ScalarValue, ScalarValue)> {
    // `copysign` only touches the sign bit, so a NaN keeps its payload.
    match value {
        Float16(Some(v)) if *v == f16::ZERO || v.is_nan() => Some((
            Float16(Some(v.copysign(f16::NEG_ONE))),
            Float16(Some(v.copysign(f16::ONE))),
        )),
        Float32(Some(v)) if *v == 0.0 || v.is_nan() => Some((
            Float32(Some(v.copysign(-1.0))),
            Float32(Some(v.copysign(1.0))),
        )),
        Float64(Some(v)) if *v == 0.0 || v.is_nan() => Some((
            Float64(Some(v.copysign(-1.0))),
            Float64(Some(v.copysign(1.0))),
        )),
        _ => None,
    }
}

fn is_nan(value: &ScalarValue) -> bool {
    match value {
        Float16(Some(v)) => v.is_nan(),
        Float32(Some(v)) => v.is_nan(),
        Float64(Some(v)) => v.is_nan(),
        _ => false,
    }
}

fn negative_infinity(value: &ScalarValue) -> Option<ScalarValue> {
    match value {
        Float16(Some(_)) => Some(Float16(Some(f16::NEG_INFINITY))),
        Float32(Some(_)) => Some(Float32(Some(f32::NEG_INFINITY))),
        Float64(Some(_)) => Some(Float64(Some(f64::NEG_INFINITY))),
        _ => None,
    }
}

/// Mark a literal that intentionally relies on Arrow's raw total order.
///
/// The range rewrite emits ordinary comparisons so scalar indices can consume
/// them. The marker keeps a later optimizer pass from recursively normalizing
/// those implementation-level comparisons while remaining invisible to the
/// physical literal value and index query.
fn raw_float_literal(value: ScalarValue, metadata: Option<&FieldMetadata>) -> Expr {
    let mut inner: BTreeMap<String, String> = metadata
        .map(|metadata| metadata.inner().clone())
        .unwrap_or_default();
    inner.insert(RAW_FLOAT_LITERAL_MARKER.to_string(), "1".to_string());
    Expr::Literal(value, Some(FieldMetadata::new(inner)))
}

fn is_raw_float_literal(metadata: Option<&FieldMetadata>) -> bool {
    metadata.is_some_and(|metadata| metadata.inner().contains_key(RAW_FLOAT_LITERAL_MARKER))
}

/// Collect the terms of an `AND`/`OR` chain, in order, ignoring nesting.
fn flatten_chain<'a>(expr: &'a Expr, op: Operator, terms: &mut Vec<&'a Expr>) {
    if let Expr::BinaryExpr(BinaryExpr {
        left,
        op: inner,
        right,
    }) = expr
        && *inner == op
    {
        flatten_chain(left, op, terms);
        flatten_chain(right, op, terms);
        return;
    }
    terms.push(expr);
}

/// True for the shape this rewrite emits for `=` and `!=`: a column tested
/// against both sign encodings of a floating point zero or NaN, negated or not.
/// Only these terms are deduplicated, so an expression the caller wrote twice is
/// left alone.
fn is_equivalent_pair_over_column(expr: &Expr) -> bool {
    let Expr::InList(InList { expr, list, .. }) = expr else {
        return false;
    };
    if !matches!(expr.as_ref(), Expr::Column(_)) {
        return false;
    }
    let [Expr::Literal(first, _), ..] = list.as_slice() else {
        return false;
    };
    equivalent_encodings(first)
        .is_some_and(|(negative, positive)| list_is_pair(list, &negative, &positive))
}

/// True when `list` is exactly two sign encodings, negative first.
fn list_is_pair(list: &[Expr], negative: &ScalarValue, positive: &ScalarValue) -> bool {
    let [Expr::Literal(first, _), Expr::Literal(second, _)] = list else {
        return false;
    };
    first == negative && second == positive
}

/// The encoding a zero bound needs to answer `op` correctly, or `None` when the
/// expression is not a floating-point literal that needs canonicalization.
fn rewrite_bound(bound: &Expr, op: Operator) -> Option<Expr> {
    let Expr::Literal(value, metadata) = bound else {
        return None;
    };
    let (negative, positive) = equivalent_encodings(value)?;
    let encoding = if is_nan(value) {
        positive
    } else {
        match op {
            Operator::GtEq => negative,
            Operator::LtEq => positive,
            _ => return None,
        }
    };
    Some(Expr::Literal(encoding, metadata.clone()))
}

fn comparison(left: Expr, op: Operator, right: Expr) -> Expr {
    Expr::BinaryExpr(BinaryExpr {
        left: Box::new(left),
        op,
        right: Box::new(right),
    })
}

fn paired_comparison(
    other: &Expr,
    op: Operator,
    negative: &ScalarValue,
    positive: &ScalarValue,
    metadata: Option<&FieldMetadata>,
) -> Option<Expr> {
    let covered = |negated| {
        Expr::InList(InList {
            expr: Box::new(other.clone()),
            list: vec![
                Expr::Literal(negative.clone(), metadata.cloned()),
                Expr::Literal(positive.clone(), metadata.cloned()),
            ],
            negated,
        })
    };
    match op {
        Operator::Eq => Some(covered(false)),
        Operator::NotEq => Some(covered(true)),
        // The list names `other` once and `IS [NOT] TRUE` keeps the null case
        // decided: `NULL IN (..)` is NULL, and `NULL IS TRUE` is false.
        Operator::IsNotDistinctFrom => Some(covered(false).is_true()),
        Operator::IsDistinctFrom => Some(covered(false).is_not_true()),
        _ => None,
    }
}

/// Rewrite `other op literal`, with `op` already mirrored when the literal was
/// written on the left.
///
/// An ordered comparison against a non-NaN bound becomes raw total-order ranges.
/// Positive NaNs already sort above the bound, and the extra `-inf` range moves
/// negative NaNs to that same side while leaving the primary comparison
/// available to the scalar-index planner.
fn rewrite_literal_comparison(
    other: &Expr,
    op: Operator,
    value: &ScalarValue,
    metadata: Option<&FieldMetadata>,
) -> Option<Expr> {
    if is_raw_float_literal(metadata) {
        return None;
    }
    let negative_infinity = negative_infinity(value)?;
    let encodings = equivalent_encodings(value);
    if let Some((negative, positive)) = &encodings
        && let Some(rewritten) = paired_comparison(other, op, negative, positive, metadata)
    {
        return Some(rewritten);
    }
    if !matches!(
        op,
        Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
    ) {
        return None;
    }
    let bound = match (encodings, op) {
        // A range cannot reorder NaN payloads, so compare without the sign instead.
        (Some((_, positive)), _) if is_nan(value) => {
            return Some(compare_floats_expr(
                other.clone(),
                op,
                Expr::Literal(positive, metadata.cloned()),
            ));
        }
        (Some((negative, _)), Operator::Lt | Operator::GtEq) => negative,
        (Some((_, positive)), _) => positive,
        (None, _) => value.clone(),
    };
    let primary = comparison(other.clone(), op, raw_float_literal(bound, metadata));
    let negative_infinity = raw_float_literal(negative_infinity, None);
    Some(match op {
        Operator::Lt | Operator::LtEq => {
            primary.and(comparison(other.clone(), Operator::GtEq, negative_infinity))
        }
        _ => primary.or(comparison(other.clone(), Operator::Lt, negative_infinity)),
    })
}

/// The operand of `operand >= -inf` as the range rewrite emits it for an upper
/// bound: true for every value except a negative NaN.
fn excluded_negative_nans(expr: &Expr) -> Option<&Expr> {
    let Expr::BinaryExpr(BinaryExpr {
        left,
        op: Operator::GtEq,
        right,
    }) = expr
    else {
        return None;
    };
    let Expr::Literal(value, metadata) = right.as_ref() else {
        return None;
    };
    (is_raw_float_literal(metadata.as_ref()) && negative_infinity(value).as_ref() == Some(value))
        .then_some(left.as_ref())
}

/// Split `operand > c OR operand < -inf`, as the range rewrite emits it for a
/// lower bound, into the bound itself and its operand.
fn lower_bound_with_negative_nans(expr: &Expr) -> Option<(&Expr, &Expr)> {
    let Expr::BinaryExpr(BinaryExpr {
        left: primary,
        op: Operator::Or,
        right: negative_nans,
    }) = expr
    else {
        return None;
    };
    let (
        Expr::BinaryExpr(BinaryExpr {
            left: operand,
            op: Operator::Gt | Operator::GtEq,
            right: bound,
        }),
        Expr::BinaryExpr(BinaryExpr {
            left: nan_operand,
            op: Operator::Lt,
            right: negative_infinity,
        }),
    ) = (primary.as_ref(), negative_nans.as_ref())
    else {
        return None;
    };
    let is_raw = |literal: &Expr| matches!(literal, Expr::Literal(_, metadata) if is_raw_float_literal(metadata.as_ref()));
    (operand == nan_operand && is_raw(bound) && is_raw(negative_infinity))
        .then_some((primary.as_ref(), operand.as_ref()))
}

/// Drop the negative-NaN ranges that cancel out when an `AND` chain bounds an
/// operand from both sides, and report whether any did.
///
/// A lower bound `o > c` or `o >= c` against a non-NaN `c` already excludes every
/// negative NaN, so next to `o >= -inf` from an upper bound its `OR o < -inf`
/// only re-adds rows that `o >= -inf` removes again, and `o >= -inf` itself is
/// implied. Dropping both keeps `o BETWEEN a AND b` a single index range; with
/// the `OR` in place the index could not intersect the two bounds and searched
/// every row above `a`.
fn drop_redundant_negative_nan_ranges(terms: &mut Vec<&Expr>) -> bool {
    let guarded: Vec<&Expr> = terms
        .iter()
        .filter_map(|term| excluded_negative_nans(term))
        .collect();
    if guarded.is_empty() {
        return false;
    }
    let bounded: Vec<&Expr> = terms
        .iter()
        .filter_map(|term| lower_bound_with_negative_nans(term))
        .map(|(_, operand)| operand)
        .filter(|operand| guarded.contains(operand))
        .collect();
    if bounded.is_empty() {
        return false;
    }
    // Drop the guards before unwrapping the lower bounds: `o >= -inf` unwraps to
    // the guard's own shape, and as the only lower bound it has to stay.
    terms.retain(|term| !excluded_negative_nans(term).is_some_and(|o| bounded.contains(&o)));
    for term in terms.iter_mut() {
        if let Some((primary, operand)) = lower_bound_with_negative_nans(term)
            && bounded.contains(&operand)
        {
            *term = primary;
        }
    }
    true
}

fn rewrite_node(expr: &Expr, schema: &DFSchema) -> Option<Expr> {
    match expr {
        // DataFusion's simplifier expands an `IN` list of three or fewer values
        // over a bare column back into an OR chain of equalities, so a second
        // `optimize_expr` splits this rewrite's own output and re-runs it on each
        // half. Both halves then produce the same list, and dropping the repeat is
        // what makes the rewrite survive that round trip. An `AND` chain also
        // drops the negative-NaN ranges that a two-sided bound makes redundant.
        Expr::BinaryExpr(BinaryExpr { op, .. }) if matches!(op, Operator::Or | Operator::And) => {
            let mut kept: Vec<&Expr> = Vec::new();
            flatten_chain(expr, *op, &mut kept);
            let mut deduped: Vec<&Expr> = Vec::with_capacity(kept.len());
            for term in kept.iter() {
                if is_equivalent_pair_over_column(term) && deduped.contains(term) {
                    continue;
                }
                deduped.push(term);
            }
            let is_bounded =
                *op == Operator::And && drop_redundant_negative_nan_ranges(&mut deduped);
            if deduped.len() == kept.len() && !is_bounded {
                return None;
            }
            deduped.into_iter().cloned().reduce(|left, right| match op {
                Operator::Or => left.or(right),
                _ => left.and(right),
            })
        }
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            // `resolve_expr` accepts the literal on either side, and the
            // operator mirrors when it sits on the left.
            match (left.as_ref(), right.as_ref()) {
                (_, Expr::Literal(value, metadata)) => {
                    rewrite_literal_comparison(left, *op, value, metadata.as_ref())
                }
                (Expr::Literal(value, metadata), _) => {
                    rewrite_literal_comparison(right, op.swap()?, value, metadata.as_ref())
                }
                _ if FLOAT_COMPARISONS.contains(op)
                    && matches!(
                        left.get_type(schema),
                        Ok(float @ (DataType::Float16 | DataType::Float32 | DataType::Float64))
                            if right.get_type(schema).ok().as_ref() == Some(&float)
                    ) =>
                {
                    Some(compare_floats_expr(
                        (**left).clone(),
                        *op,
                        (**right).clone(),
                    ))
                }
                _ => None,
            }
        }
        // `BETWEEN` normally reaches this rewrite already expanded into `>=` and
        // `<=` by the simplifier. It survives unexpanded when every operand is
        // constant, because then the simplifier expands and folds it in one pass
        // and the comparison is gone before the post-pass looks. The bounds take
        // the encodings their expanded operators would: `low` is a `>=` bound and
        // `high` is a `<=` bound.
        Expr::Between(between) => {
            let normalized_expr = match between.expr.as_ref() {
                Expr::Literal(value, metadata) if is_nan(value) => {
                    let (_, positive) = equivalent_encodings(value)?;
                    Some(Expr::Literal(positive, metadata.clone()))
                }
                _ => None,
            };
            let low = rewrite_bound(&between.low, Operator::GtEq);
            let high = rewrite_bound(&between.high, Operator::LtEq);
            if normalized_expr.is_none() && low.is_none() && high.is_none() {
                return None;
            }
            Some(Expr::Between(Between {
                expr: Box::new(normalized_expr.unwrap_or_else(|| (*between.expr).clone())),
                negated: between.negated,
                low: Box::new(low.unwrap_or_else(|| (*between.low).clone())),
                high: Box::new(high.unwrap_or_else(|| (*between.high).clone())),
            }))
        }
        Expr::InList(InList {
            expr,
            list,
            negated,
        }) => {
            // A zero or NaN literal on the probe side needs the same treatment.
            // The list elements are arbitrary expressions there, so expand into the
            // equality form the binary arm already covers. A literal probe that is
            // not sign-sensitive compares the same way against either encoding, so it
            // needs no widening either.
            if let Expr::Literal(value, metadata) = expr.as_ref() {
                let (negative, positive) = equivalent_encodings(value)?;
                // The expansion below puts a paired literal in front of exactly this
                // list, so stop rather than expanding that term again.
                if list_is_pair(list, &negative, &positive) {
                    return None;
                }
                let matches_any = list
                    .iter()
                    .map(|item| {
                        Expr::InList(InList {
                            expr: Box::new(item.clone()),
                            list: vec![
                                Expr::Literal(negative.clone(), metadata.clone()),
                                Expr::Literal(positive.clone(), metadata.clone()),
                            ],
                            negated: false,
                        })
                    })
                    .reduce(Expr::or)?;
                return Some(if *negated {
                    Expr::Not(Box::new(matches_any))
                } else {
                    matches_any
                });
            }
            Some(Expr::InList(InList {
                expr: expr.clone(),
                list: widen_equivalent_list(list)?,
                negated: *negated,
            }))
        }
        // One probe for both encodings, rather than two `array_has` calls joined
        // by `OR`, so a volatile haystack is still evaluated once.
        Expr::ScalarFunction(ScalarFunction { func, args }) if func.name() == "array_has" => {
            let [haystack, Expr::Literal(value, metadata)] = args.as_slice() else {
                return None;
            };
            let (negative, positive) = equivalent_encodings(value)?;
            let data_type = negative.data_type();
            let needles = ScalarValue::new_list(&[negative, positive], &data_type, true);
            Some(array_has_any(
                haystack.clone(),
                Expr::Literal(ScalarValue::List(needles), metadata.clone()),
            ))
        }
        _ => None,
    }
}

/// Add the missing sign encoding for each zero or NaN in an `IN` list.
///
/// Returns `None` when the list holds no sign-sensitive value, or already spells
/// out both encodings of each value it holds.
fn widen_equivalent_list(list: &[Expr]) -> Option<Vec<Expr>> {
    // Most lists hold neither value, so collect what is missing before copying
    // anything.
    let mut missing: Vec<Expr> = Vec::new();
    for item in list {
        let Expr::Literal(value, metadata) = item else {
            continue;
        };
        let Some((negative, positive)) = equivalent_encodings(value) else {
            continue;
        };
        let counterpart = if *value == negative {
            positive
        } else {
            negative
        };
        // `ScalarValue` compares floats by bit pattern, so this distinguishes
        // the two encodings rather than collapsing them.
        let is_counterpart =
            |other: &Expr| matches!(other, Expr::Literal(v, _) if *v == counterpart);
        if list.iter().any(is_counterpart) || missing.iter().any(is_counterpart) {
            continue;
        }
        missing.push(Expr::Literal(counterpart, metadata.clone()));
    }
    if missing.is_empty() {
        return None;
    }
    let mut widened = Vec::with_capacity(list.len() + missing.len());
    widened.extend(list.iter().cloned());
    widened.append(&mut missing);
    Some(widened)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Float64Array};
    use arrow_schema::{Field, Schema};
    use datafusion::functions_nested::expr_fn::array_has;
    use datafusion::prelude::{col, lit};
    use rstest::rstest;

    use super::*;

    fn rewrite(expr: Expr) -> Expr {
        let schema = DFSchema::try_from(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Float64, true),
            Field::new_list("l", Field::new_list_field(DataType::Float64, true), true),
        ]))
        .unwrap();
        rewrite_float_comparisons(expr, &schema).unwrap()
    }

    fn compare(left: Expr, op: Operator, right: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(left),
            op,
            right: Box::new(right),
        })
    }

    fn raw(value: ScalarValue) -> Expr {
        raw_float_literal(value, None)
    }

    fn expected_ordered(column: &str, op: Operator, bound: ScalarValue) -> Expr {
        let primary = compare(col(column), op, raw(bound.clone()));
        let guard = match op {
            Operator::Lt | Operator::LtEq => compare(
                col(column),
                Operator::GtEq,
                raw(negative_infinity(&bound).unwrap()),
            ),
            Operator::Gt | Operator::GtEq => compare(
                col(column),
                Operator::Lt,
                raw(negative_infinity(&bound).unwrap()),
            ),
            _ => unreachable!(),
        };
        match op {
            Operator::Lt | Operator::LtEq => primary.and(guard),
            _ => primary.or(guard),
        }
    }

    #[rstest]
    #[case::lt_from_positive(Operator::Lt, 0.0, -0.0)]
    #[case::lt_from_negative(Operator::Lt, -0.0, -0.0)]
    #[case::lt_eq_from_positive(Operator::LtEq, 0.0, 0.0)]
    #[case::lt_eq_from_negative(Operator::LtEq, -0.0, 0.0)]
    #[case::gt_from_positive(Operator::Gt, 0.0, 0.0)]
    #[case::gt_from_negative(Operator::Gt, -0.0, 0.0)]
    #[case::gt_eq_from_positive(Operator::GtEq, 0.0, -0.0)]
    #[case::gt_eq_from_negative(Operator::GtEq, -0.0, -0.0)]
    // A finite bound still gets the range that moves negative NaNs above it.
    #[case::gt_finite(Operator::Gt, 1.0, 1.0)]
    #[case::lt_eq_finite(Operator::LtEq, 1.0, 1.0)]
    fn range_comparison_uses_the_encoding_for_the_operator(
        #[case] op: Operator,
        #[case] written: f64,
        #[case] evaluated: f64,
    ) {
        assert_eq!(
            rewrite(compare(col("x"), op, lit(written))),
            expected_ordered("x", op, Float64(Some(evaluated)))
        );
    }

    #[test]
    fn computed_literal_comparison_uses_indexable_ranges() {
        let computed = col("x") * lit(2.0);
        let primary = compare(computed.clone(), Operator::Gt, raw(Float64(Some(1.0))));
        let negative_nan_range = compare(
            computed.clone(),
            Operator::Lt,
            raw(Float64(Some(f64::NEG_INFINITY))),
        );
        assert_eq!(
            rewrite(computed.gt(lit(1.0))),
            primary.or(negative_nan_range)
        );
    }

    #[rstest]
    #[case::zero(0.0, -0.0)]
    #[case::nan(
        f64::from_bits(0x7ff8_0000_0000_0042),
        f64::from_bits(0xfff8_0000_0000_0042)
    )]
    fn array_has_probes_both_sign_encodings_at_once(#[case] positive: f64, #[case] negative: f64) {
        let needles = ScalarValue::new_list(
            &[Float64(Some(negative)), Float64(Some(positive))],
            &DataType::Float64,
            true,
        );
        let expected = array_has_any(col("l"), lit(ScalarValue::List(needles)));
        assert_eq!(rewrite(array_has(col("l"), lit(positive))), expected);
        assert_eq!(rewrite(array_has(col("l"), lit(negative))), expected);
    }

    #[test]
    fn column_comparison_compares_each_operand_once() {
        let rewritten = rewrite(col("x").lt(col("y")));
        assert_eq!(
            rewritten,
            compare_floats_expr(col("x"), Operator::Lt, col("y"))
        );
        assert_eq!(rewrite(rewritten.clone()), rewritten);
    }

    /// Rows: both zero signs, both NaN signs, an ordinary pair, one NULL, two NULLs.
    #[rstest]
    #[case::float16(DataType::Float16)]
    #[case::float32(DataType::Float32)]
    #[case::float64(DataType::Float64)]
    fn compare_floats_ignores_the_sign_of_zero_and_nan(#[case] data_type: DataType) {
        let (t, f) = (Some(true), Some(false));
        let column = |values: Vec<Option<f64>>| {
            arrow::compute::cast(&Float64Array::from(values), &data_type).unwrap()
        };
        let nan = f64::NAN;
        let left = column(vec![
            Some(-0.0),
            Some(0.0),
            Some(-nan),
            Some(nan),
            Some(1.0),
            None,
            None,
        ]);
        let right = column(vec![
            Some(0.0),
            Some(-0.0),
            Some(nan),
            Some(-nan),
            Some(-1.0),
            Some(1.0),
            None,
        ]);
        let compare = |right: &ArrayRef, op| {
            compare_float_arrays(&left, right, op, left.len())
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        };
        assert_eq!(compare(&right, Operator::Eq), [t, t, t, t, f, None, None]);
        assert_eq!(
            compare(&right, Operator::NotEq),
            [f, f, f, f, t, None, None]
        );
        assert_eq!(compare(&right, Operator::Lt), [f, f, f, f, f, None, None]);
        assert_eq!(compare(&right, Operator::GtEq), [t, t, t, t, t, None, None]);
        assert_eq!(
            compare(&right, Operator::IsNotDistinctFrom),
            [t, t, t, t, f, f, t]
        );
        assert_eq!(
            compare(&right, Operator::IsDistinctFrom),
            [f, f, f, f, t, t, f]
        );
        // A one-element operand is a scalar applied to every row.
        let negative_nan = column(vec![Some(-nan)]);
        assert_eq!(
            compare(&negative_nan, Operator::Lt),
            [t, t, f, f, t, None, None]
        );
        let null = column(vec![None]);
        assert_eq!(
            compare(&null, Operator::IsNotDistinctFrom),
            [f, f, f, f, f, t, t]
        );
    }

    #[test]
    fn compare_floats_keeps_nan_payloads_apart() {
        let payload =
            |bits: u64| Arc::new(Float64Array::from(vec![f64::from_bits(bits)])) as ArrayRef;
        let positive = payload(0x7ff8_0000_0000_0042);
        let negative = payload(0xfff8_0000_0000_0042);
        let other = payload(0x7ff8_0000_0000_0043);
        let equal = |left, right| {
            compare_float_arrays(left, right, Operator::Eq, 1)
                .unwrap()
                .value(0)
        };
        assert!(equal(&negative, &positive));
        assert!(!equal(&negative, &other));
    }

    #[rstest]
    #[case::eq(Operator::Eq, false, 0.0, -0.0)]
    #[case::not_eq(Operator::NotEq, true, 0.0, -0.0)]
    // A NaN keeps its payload in both encodings.
    #[case::nan(
        Operator::Eq,
        false,
        f64::from_bits(0xfff8_0000_0000_0042),
        f64::from_bits(0xfff8_0000_0000_0042)
    )]
    fn equality_covers_both_encodings(
        #[case] op: Operator,
        #[case] negated: bool,
        #[case] written: f64,
        #[case] negative: f64,
    ) {
        assert_eq!(
            rewrite(compare(col("x"), op, lit(written))),
            Expr::InList(InList {
                expr: Box::new(col("x")),
                list: vec![lit(negative), lit(negative.abs())],
                negated,
            })
        );
    }

    #[test]
    fn a_literal_on_the_left_mirrors_the_operator() {
        // `0.0 > x` is `x < 0.0`, which evaluates against the negative encoding.
        assert_eq!(
            rewrite(compare(lit(0.0), Operator::Gt, col("x"))),
            expected_ordered("x", Operator::Lt, Float64(Some(-0.0)))
        );
    }

    #[rstest]
    #[case::float32(Float32(Some(-0.0)), Float32(Some(0.0)))]
    #[case::float16(Float16(Some(f16::NEG_ZERO)), Float16(Some(f16::ZERO)))]
    fn narrow_floats_are_rewritten_too(
        #[case] written: ScalarValue,
        #[case] evaluated: ScalarValue,
    ) {
        assert_eq!(
            rewrite(compare(
                col("x"),
                Operator::LtEq,
                Expr::Literal(written, None)
            )),
            expected_ordered("x", Operator::LtEq, evaluated)
        );
    }

    #[test]
    fn an_in_list_gains_the_missing_encoding() {
        assert_eq!(
            rewrite(Expr::InList(InList {
                expr: Box::new(col("x")),
                list: vec![lit(0.0), lit(5.0)],
                negated: true,
            })),
            Expr::InList(InList {
                expr: Box::new(col("x")),
                list: vec![lit(0.0), lit(5.0), lit(-0.0)],
                negated: true,
            })
        );
    }

    #[test]
    fn only_the_zero_comparison_in_a_conjunction_changes() {
        assert_eq!(
            rewrite(col("x").lt(lit(0.0)).and(col("y").eq(lit(1.0)))),
            expected_ordered("x", Operator::Lt, Float64(Some(-0.0))).and(col("y").eq(lit(1.0)))
        );
    }

    #[rstest]
    #[case::between(col("x").gt_eq(lit(1.0)), col("x").lt_eq(lit(5.0)), 1.0, 5.0)]
    #[case::zero_bounds(col("x").gt(lit(0.0)), col("x").lt(lit(0.0)), 0.0, -0.0)]
    #[case::upper_first(col("x").lt(lit(5.0)), col("x").gt(lit(1.0)), 5.0, 1.0)]
    // The lower bound unwraps to the same shape as the upper bound's guard, and
    // must survive dropping that guard.
    #[case::negative_infinity_lower_bound(col("x").gt_eq(lit(f64::NEG_INFINITY)), col("x").lt_eq(lit(1.0)), f64::NEG_INFINITY, 1.0)]
    fn two_sided_bound_keeps_one_range(
        #[case] first: Expr,
        #[case] second: Expr,
        #[case] first_bound: f64,
        #[case] second_bound: f64,
    ) {
        let raw_bound = |bound: &Expr, value: f64| {
            let Expr::BinaryExpr(BinaryExpr { left, op, .. }) = bound else {
                unreachable!()
            };
            compare(*left.clone(), *op, raw(Float64(Some(value))))
        };
        assert_eq!(
            rewrite(first.clone().and(col("y").eq(lit(1.0))).and(second.clone())),
            raw_bound(&first, first_bound)
                .and(col("y").eq(lit(1.0)))
                .and(raw_bound(&second, second_bound))
        );
    }

    /// Without both a lower bound and the upper bound's `>= -inf` range on the
    /// same operand in one `AND` chain, every negative-NaN range is still needed.
    #[rstest]
    #[case::different_operands(col("x").gt_eq(lit(1.0)), col("y").lt_eq(lit(5.0)), Operator::And)]
    #[case::two_lower_bounds(col("x").gt_eq(lit(1.0)), col("x").gt(lit(2.0)), Operator::And)]
    #[case::two_upper_bounds(col("x").lt_eq(lit(1.0)), col("x").lt(lit(2.0)), Operator::And)]
    #[case::disjunction(col("x").gt_eq(lit(1.0)), col("x").lt_eq(lit(5.0)), Operator::Or)]
    fn one_sided_bounds_keep_negative_nan_ranges(
        #[case] first: Expr,
        #[case] second: Expr,
        #[case] op: Operator,
    ) {
        assert_eq!(
            rewrite(compare(first.clone(), op, second.clone())),
            compare(rewrite(first), op, rewrite(second))
        );
    }

    #[rstest]
    #[case::integer_zero(col("x").eq(lit(0_i64)))]
    #[case::null_equality(compare(col("x"), Operator::Eq, Expr::Literal(Float64(None), None)))]
    #[case::null_range(compare(col("x"), Operator::Lt, Expr::Literal(Float64(None), None)))]
    #[case::both_encodings_listed(Expr::InList(InList {
        expr: Box::new(col("x")),
        list: vec![lit(-0.0), lit(0.0)],
        negated: false,
    }))]
    fn unrelated_comparisons_are_left_alone(#[case] expr: Expr) {
        assert_eq!(rewrite(expr.clone()), expr);
    }

    /// Distinctness has to stay decided for a null operand, and it has to name
    /// the operand once so a computed one is not evaluated twice.
    #[rstest]
    #[case::is_not_distinct_from(Operator::IsNotDistinctFrom)]
    #[case::is_distinct_from(Operator::IsDistinctFrom)]
    fn distinct_from_lowers_through_a_null_defaulted_list(#[case] op: Operator) {
        let covered = Expr::InList(InList {
            expr: Box::new(col("x")),
            list: vec![lit(-0.0), lit(0.0)],
            negated: false,
        });
        let expected = if op == Operator::IsDistinctFrom {
            covered.is_not_true()
        } else {
            covered.is_true()
        };
        assert_eq!(rewrite(compare(col("x"), op, lit(0.0))), expected);
    }

    /// The operand does not have to be a column. Bailing out on anything else
    /// used to leave `filter_expr` answering computed operands on Arrow's
    /// sign-sensitive order, which returns wrong rows.
    #[rstest]
    #[case::is_not_distinct_from(Operator::IsNotDistinctFrom)]
    #[case::is_distinct_from(Operator::IsDistinctFrom)]
    fn distinct_from_rewrites_a_computed_operand(#[case] op: Operator) {
        let computed = col("x") * lit(2.0);
        let covered = Expr::InList(InList {
            expr: Box::new(computed.clone()),
            list: vec![lit(-0.0), lit(0.0)],
            negated: false,
        });
        let expected = if op == Operator::IsDistinctFrom {
            covered.is_not_true()
        } else {
            covered.is_true()
        };
        assert_eq!(rewrite(compare(computed, op, lit(0.0))), expected);
    }

    /// Several paths optimize the same expression more than once, so every shape
    /// the rewrite emits has to be a fixed point.
    #[rstest]
    #[case::lt(col("x").lt(lit(0.0)))]
    #[case::gt_eq(col("x").gt_eq(lit(0.0)))]
    #[case::eq(col("x").eq(lit(0.0)))]
    #[case::not_eq(col("x").not_eq(lit(0.0)))]
    #[case::in_list(Expr::InList(InList {
        expr: Box::new(col("x")),
        list: vec![lit(0.0), lit(5.0)],
        negated: false,
    }))]
    #[case::zero_probe(Expr::InList(InList {
        expr: Box::new(lit(0.0)),
        list: vec![col("a"), col("b")],
        negated: false,
    }))]
    #[case::zero_probe_over_literals(Expr::InList(InList {
        expr: Box::new(lit(0.0)),
        list: vec![col("a"), lit(0.0)],
        negated: false,
    }))]
    #[case::is_not_distinct_from(compare(col("x"), Operator::IsNotDistinctFrom, lit(0.0)))]
    #[case::is_distinct_from(compare(col("x"), Operator::IsDistinctFrom, lit(0.0)))]
    fn rewriting_twice_changes_nothing(#[case] expr: Expr) {
        let once = rewrite(expr);
        assert_eq!(rewrite(once.clone()), once);
    }

    #[rstest]
    #[case::probe(false)]
    #[case::negated_probe(true)]
    fn a_zero_probe_expands_into_equalities(#[case] negated: bool) {
        let covers = |column| {
            Expr::InList(InList {
                expr: Box::new(col(column)),
                list: vec![lit(-0.0), lit(0.0)],
                negated: false,
            })
        };
        let matches_any = covers("a").or(covers("b"));
        assert_eq!(
            rewrite(Expr::InList(InList {
                expr: Box::new(lit(0.0)),
                list: vec![col("a"), col("b")],
                negated,
            })),
            if negated {
                Expr::Not(Box::new(matches_any))
            } else {
                matches_any
            }
        );
    }

    #[test]
    fn scalar_value_keeps_the_two_zero_encodings_apart() {
        // The `IN` list widening decides "already listed" with this comparison. A
        // DataFusion release that made the two encodings equal would silently stop
        // it.
        assert_ne!(Float64(Some(-0.0)), Float64(Some(0.0)));
        assert_ne!(Float32(Some(-0.0)), Float32(Some(0.0)));
        assert_ne!(Float16(Some(f16::NEG_ZERO)), Float16(Some(f16::ZERO)));
    }

    /// The scan path optimizes the same expression twice, and the simplifier
    /// expands a short `IN` list over a column back into an OR chain in between,
    /// so a fixed point of the rewrite alone would not be enough.
    #[rstest]
    #[case::eq("value = 0.0")]
    #[case::not_eq("value != 0.0")]
    #[case::in_list("value IN (0.0, 1.0)")]
    #[case::lt("value < 0.0")]
    #[case::gt_eq("value >= 0.0")]
    #[case::between("value BETWEEN -0.0 AND 0.0")]
    // The dedup that makes the first three cases hold keys on the probe being a
    // bare column, which is also what DataFusion requires before it shortens a
    // list. This case fails if a release ever relaxes that.
    #[case::non_column_probe("abs(value) = 0.0")]
    #[case::between("value BETWEEN 1.0 AND 5.0")]
    #[case::two_sided_zero_bound("value > 0.0 AND value < 1.0")]
    #[case::not_between("NOT (value >= -1.0 AND value <= 0.0)")]
    #[case::column_comparison("value < other")]
    #[case::array_has("array_has(values, 0.0)")]
    // `IS [NOT] DISTINCT FROM` is missing because `Planner::parse_filter` rejects
    // it as unsupported SQL; that arm is reachable only from a programmatically
    // built expression, and `rewriting_twice_changes_nothing` covers it there.
    fn optimizing_twice_changes_nothing(#[case] filter: &str) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, true),
            Field::new("other", DataType::Float64, true),
            Field::new_list(
                "values",
                Field::new_list_field(DataType::Float64, true),
                true,
            ),
        ]));
        let planner = crate::planner::Planner::new(schema);
        let once = planner
            .optimize_expr(planner.parse_filter(filter).unwrap())
            .unwrap();
        assert_eq!(planner.optimize_expr(once.clone()).unwrap(), once);
    }

    /// A comparison whose operands are all constant never reaches the rewrite if
    /// the rewrite only runs after `simplify`: the simplifier folds it to a bare
    /// boolean under Arrow's total order first, and there is nothing left to
    /// repair. These fold to the IEEE answer only because the rewrite also runs
    /// before `simplify`.
    #[rstest]
    #[case::lt("-1.0 * 0.0 < 0.0", false)]
    #[case::eq("(-1.0 * 0.0) = 0.0", true)]
    #[case::gt_eq("(-1.0 * 0.0) >= 0.0", true)]
    #[case::not_eq("(-1.0 * 0.0) != 0.0", false)]
    #[case::gt("(-1.0 * 0.0) > 0.0", false)]
    #[case::lt_eq("(-1.0 * 0.0) <= 0.0", true)]
    // The zero on the right is produced by folding rather than written, so these
    // reach the rewrite only because the operands are folded before the
    // comparison is.
    #[case::folded_rhs_lt("-1.0 * 0.0 < (1.0 - 1.0)", false)]
    #[case::folded_rhs_eq("(-1.0 * 0.0) = (1.0 - 1.0)", true)]
    #[case::folded_rhs_gt_eq("(-1.0 * 0.0) >= (1.0 - 1.0)", true)]
    #[case::folded_rhs_not_eq("(-1.0 * 0.0) != (1.0 - 1.0)", false)]
    #[case::both_sides_folded("(0.0 * -1.0) < (1.0 - 1.0)", false)]
    // `BETWEEN` and `IN` fold the same way, and a fully constant `BETWEEN` never
    // reaches the rewrite already expanded, which is why the rewrite has its own
    // arm for it.
    #[case::folded_between("(-1.0 * 0.0) BETWEEN (1.0 - 1.0) AND 1.0", true)]
    #[case::folded_in_list("(-1.0 * 0.0) IN ((1.0 - 1.0), 1.0)", true)]
    #[case::folded_not_in_list("(-1.0 * 0.0) NOT IN ((1.0 - 1.0), 1.0)", false)]
    // Nested under a connective, so the operand pass has to descend.
    #[case::under_or("(-1.0 * 0.0) < (1.0 - 1.0) OR 1.0 > 2.0", false)]
    #[case::under_not("NOT ((-1.0 * 0.0) < (1.0 - 1.0))", true)]
    // Wrapped in something that folds the comparison's own result. These are why
    // the operand folding walks every container instead of a list of allowed
    // parents: each of these is a different spelling of the same exposure.
    #[case::under_is_true("((-1.0 * 0.0) < (1.0 - 1.0)) IS TRUE", false)]
    #[case::under_is_false("((-1.0 * 0.0) < (1.0 - 1.0)) IS FALSE", true)]
    #[case::under_is_not_true("((-1.0 * 0.0) < (1.0 - 1.0)) IS NOT TRUE", true)]
    #[case::under_eq_true("((-1.0 * 0.0) < (1.0 - 1.0)) = TRUE", false)]
    #[case::under_cast("CAST(((-1.0 * 0.0) < (1.0 - 1.0)) AS BOOLEAN)", false)]
    #[case::under_in_true("((-1.0 * 0.0) < (1.0 - 1.0)) IN (TRUE)", false)]
    #[case::under_is_true_eq("((-1.0 * 0.0) = (1.0 - 1.0)) IS TRUE", true)]
    #[case::under_nested_wrappers("NOT (((-1.0 * 0.0) < (1.0 - 1.0)) IS TRUE)", true)]
    // A folded NaN with its sign bit set. The ordered comparison against a NaN
    // literal goes through the UDF, which the simplifier then folds as a scalar.
    #[case::negative_nan_lt_nan("-CAST('NaN' AS DOUBLE) < CAST('NaN' AS DOUBLE)", false)]
    #[case::negative_nan_gt_eq_nan("-CAST('NaN' AS DOUBLE) >= CAST('NaN' AS DOUBLE)", true)]
    #[case::negative_nan_gt_finite("-CAST('NaN' AS DOUBLE) > 1.0", true)]
    fn folded_constant_comparisons_use_ieee_semantics(
        #[case] filter: &str,
        #[case] expected: bool,
    ) {
        let schema =
            std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                "value",
                arrow_schema::DataType::Float64,
                true,
            )]));
        let planner = crate::planner::Planner::new(schema);
        let optimized = planner
            .optimize_expr(planner.parse_filter(filter).unwrap())
            .unwrap();
        assert_eq!(
            optimized,
            Expr::Literal(ScalarValue::Boolean(Some(expected)), None),
            "filter: {filter}"
        );
    }
}
