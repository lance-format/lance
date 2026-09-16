// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Rewrites for signed-zero literals and NaN sign bits in comparisons.

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock};

use arrow_array::ArrayRef;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_schema::DataType;
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::expr::{Between, InList, ScalarFunction};
use datafusion::logical_expr::{
    BinaryExpr, ColumnarValue, ExprSchemable, Operator, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, Volatility,
};
use datafusion::prelude::Expr;
use datafusion::scalar::ScalarValue::{self, Float16, Float32, Float64};
use datafusion_common::DFSchema;
use datafusion_common::metadata::FieldMetadata;
use datafusion_common::tree_node::{Transformed, TreeNode};
use half::f16;
use lance_core::Result;

const NORMALIZE_NAN_SIGN_NAME: &str = "_lance_normalize_nan_sign";
const RAW_FLOAT_LITERAL_MARKER: &str = "lance:raw-float-comparison";

/// Clears the sign bit only when a comparison operand is NaN.
///
/// Comparisons against non-NaN literals are rewritten into indexable ranges
/// instead. This UDF is reserved for column-to-column, computed, and NaN-bound
/// comparisons where ranges cannot normalize an operand without changing its
/// payload ordering.
#[derive(Debug, Eq, PartialEq, Hash)]
struct NormalizeNanSign {
    signature: Signature,
}

impl NormalizeNanSign {
    fn new() -> Self {
        Self {
            signature: Signature::uniform(
                1,
                vec![DataType::Float16, DataType::Float32, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for NormalizeNanSign {
    fn name(&self) -> &str {
        NORMALIZE_NAN_SIGN_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> DFResult<DataType> {
        match arg_types {
            [data_type @ (DataType::Float16 | DataType::Float32 | DataType::Float64)] => {
                Ok(data_type.clone())
            }
            _ => Err(datafusion::error::DataFusionError::Execution(format!(
                "{NORMALIZE_NAN_SIGN_NAME} expected one floating-point argument, got {arg_types:?}"
            ))),
        }
    }

    fn invoke_with_args(&self, func_args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let mut args = func_args.args.into_iter();
        let Some(value) = args.next() else {
            return Err(datafusion::error::DataFusionError::Execution(format!(
                "{NORMALIZE_NAN_SIGN_NAME} expected one argument, got none"
            )));
        };
        if args.next().is_some() {
            return Err(datafusion::error::DataFusionError::Execution(format!(
                "{NORMALIZE_NAN_SIGN_NAME} expected one argument, got more than one"
            )));
        }
        match value {
            ColumnarValue::Array(array) => normalize_nan_array(array).map(ColumnarValue::Array),
            ColumnarValue::Scalar(value) => normalize_nan_scalar(value).map(ColumnarValue::Scalar),
        }
    }
}

fn normalize_nan_array(array: ArrayRef) -> DFResult<ArrayRef> {
    match array.data_type() {
        DataType::Float16 => {
            let values = array.as_primitive::<Float16Type>();
            if !values
                .values()
                .iter()
                .any(|value| value.is_nan() && value.is_sign_negative())
            {
                return Ok(array);
            }
            Ok(Arc::new(values.unary::<_, Float16Type>(|value| {
                if value.is_nan() {
                    f16::from_bits(value.to_bits() & 0x7fff)
                } else {
                    value
                }
            })))
        }
        DataType::Float32 => {
            let values = array.as_primitive::<Float32Type>();
            if !values
                .values()
                .iter()
                .any(|value| value.is_nan() && value.is_sign_negative())
            {
                return Ok(array);
            }
            Ok(Arc::new(values.unary::<_, Float32Type>(|value| {
                if value.is_nan() {
                    f32::from_bits(value.to_bits() & 0x7fff_ffff)
                } else {
                    value
                }
            })))
        }
        DataType::Float64 => {
            let values = array.as_primitive::<Float64Type>();
            if !values
                .values()
                .iter()
                .any(|value| value.is_nan() && value.is_sign_negative())
            {
                return Ok(array);
            }
            Ok(Arc::new(values.unary::<_, Float64Type>(|value| {
                if value.is_nan() {
                    f64::from_bits(value.to_bits() & 0x7fff_ffff_ffff_ffff)
                } else {
                    value
                }
            })))
        }
        data_type => Err(datafusion::error::DataFusionError::Execution(format!(
            "{NORMALIZE_NAN_SIGN_NAME} expected a floating-point array, got {data_type}"
        ))),
    }
}

fn normalize_nan_scalar(value: ScalarValue) -> DFResult<ScalarValue> {
    match value {
        Float16(Some(value)) if value.is_nan() => {
            Ok(Float16(Some(f16::from_bits(value.to_bits() & 0x7fff))))
        }
        Float32(Some(value)) if value.is_nan() => {
            Ok(Float32(Some(f32::from_bits(value.to_bits() & 0x7fff_ffff))))
        }
        Float64(Some(value)) if value.is_nan() => Ok(Float64(Some(f64::from_bits(
            value.to_bits() & 0x7fff_ffff_ffff_ffff,
        )))),
        value @ (Float16(_) | Float32(_) | Float64(_)) => Ok(value),
        value => Err(datafusion::error::DataFusionError::Execution(format!(
            "{NORMALIZE_NAN_SIGN_NAME} expected a floating-point scalar, got {value:?}"
        ))),
    }
}

fn normalize_nan_expr(expr: &Expr) -> Expr {
    if matches!(expr, Expr::ScalarFunction(function) if function.name() == NORMALIZE_NAN_SIGN_NAME)
    {
        return expr.clone();
    }
    static UDF: LazyLock<Arc<ScalarUDF>> =
        LazyLock::new(|| Arc::new(ScalarUDF::new_from_impl(NormalizeNanSign::new())));
    Expr::ScalarFunction(ScalarFunction::new_udf(UDF.clone(), vec![expr.clone()]))
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
///
/// The extra NaN range is combined with `AND` for `<`/`<=` and `OR` for `>`/`>=`.
/// Equality names both encodings because scalar indices key on the bit pattern.
/// A comparison without a literal normalizes NaN operands through an internal
/// physical expression, preserving payload bits and evaluating each operand once.
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

/// Whether the rewrite acts on comparisons under `op`.
fn is_float_comparison(op: Operator) -> bool {
    matches!(
        op,
        Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
            | Operator::Eq
            | Operator::NotEq
            | Operator::IsDistinctFrom
            | Operator::IsNotDistinctFrom
    )
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
                Expr::BinaryExpr(BinaryExpr { left, op, right }) if is_float_comparison(op) => {
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
    match value {
        Float16(Some(v)) if *v == f16::ZERO => {
            Some((Float16(Some(f16::NEG_ZERO)), Float16(Some(f16::ZERO))))
        }
        Float16(Some(v)) if v.is_nan() => Some((
            Float16(Some(f16::from_bits(v.to_bits() | 0x8000))),
            Float16(Some(f16::from_bits(v.to_bits() & 0x7fff))),
        )),
        Float32(Some(v)) if *v == 0.0 => Some((Float32(Some(-0.0)), Float32(Some(0.0)))),
        Float32(Some(v)) if v.is_nan() => Some((
            Float32(Some(f32::from_bits(v.to_bits() | 0x8000_0000))),
            Float32(Some(f32::from_bits(v.to_bits() & 0x7fff_ffff))),
        )),
        Float64(Some(v)) if *v == 0.0 => Some((Float64(Some(-0.0)), Float64(Some(0.0)))),
        Float64(Some(v)) if v.is_nan() => Some((
            Float64(Some(f64::from_bits(v.to_bits() | 0x8000_0000_0000_0000))),
            Float64(Some(f64::from_bits(v.to_bits() & 0x7fff_ffff_ffff_ffff))),
        )),
        _ => None,
    }
}

fn is_nan(value: &ScalarValue) -> bool {
    matches!(value, Float16(Some(value)) if value.is_nan())
        || matches!(value, Float32(Some(value)) if value.is_nan())
        || matches!(value, Float64(Some(value)) if value.is_nan())
}

fn negative_infinity(value: &ScalarValue) -> Option<ScalarValue> {
    match value {
        Float16(Some(_)) => Some(Float16(Some(f16::NEG_INFINITY))),
        Float32(Some(_)) => Some(Float32(Some(f32::NEG_INFINITY))),
        Float64(Some(_)) => Some(Float64(Some(f64::NEG_INFINITY))),
        _ => None,
    }
}

fn is_float_expr(expr: &Expr, schema: &DFSchema) -> bool {
    matches!(
        expr.get_type(schema),
        Ok(DataType::Float16 | DataType::Float32 | DataType::Float64)
    )
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
    negative: ScalarValue,
    positive: ScalarValue,
    metadata: Option<&FieldMetadata>,
) -> Option<Expr> {
    match op {
        Operator::Eq | Operator::NotEq => Some(Expr::InList(InList {
            expr: Box::new(other.clone()),
            list: vec![
                Expr::Literal(negative, metadata.cloned()),
                Expr::Literal(positive, metadata.cloned()),
            ],
            negated: op == Operator::NotEq,
        })),
        Operator::IsNotDistinctFrom | Operator::IsDistinctFrom => {
            // The list names `other` once and `IS [NOT] TRUE` keeps the null case
            // decided: `NULL IN (..)` is NULL, and `NULL IS TRUE` is false.
            let covered = Expr::InList(InList {
                expr: Box::new(other.clone()),
                list: vec![
                    Expr::Literal(negative, metadata.cloned()),
                    Expr::Literal(positive, metadata.cloned()),
                ],
                negated: false,
            });
            Some(if op == Operator::IsDistinctFrom {
                covered.is_not_true()
            } else {
                covered.is_true()
            })
        }
        _ => None,
    }
}

fn rewrite_literal_comparison(
    other: &Expr,
    op: Operator,
    value: &ScalarValue,
    metadata: Option<&FieldMetadata>,
) -> Option<Expr> {
    if is_raw_float_literal(metadata) {
        return None;
    }

    if let Some((negative, positive)) = equivalent_encodings(value) {
        if let Some(rewritten) =
            paired_comparison(other, op, negative.clone(), positive.clone(), metadata)
        {
            return Some(rewritten);
        }
        if is_nan(value) {
            return matches!(
                op,
                Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
            )
            .then(|| {
                comparison(
                    normalize_nan_expr(other),
                    op,
                    Expr::Literal(positive, metadata.cloned()),
                )
            });
        }

        let bound = match op {
            Operator::Lt | Operator::GtEq => negative,
            Operator::LtEq | Operator::Gt => positive,
            _ => return None,
        };
        return Some(rewrite_ordered_comparison(other, op, bound, metadata));
    }

    if negative_infinity(value).is_some()
        && matches!(
            op,
            Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
        )
    {
        return Some(rewrite_ordered_comparison(
            other,
            op,
            value.clone(),
            metadata,
        ));
    }

    None
}

/// Express a comparison against a non-NaN bound using raw total-order ranges.
///
/// Positive NaNs already sort above the bound. The extra range moves negative
/// NaNs to that same side while leaving the primary comparison available to the
/// scalar-index planner.
fn rewrite_ordered_comparison(
    other: &Expr,
    op: Operator,
    bound: ScalarValue,
    metadata: Option<&FieldMetadata>,
) -> Expr {
    if !matches!(other, Expr::Column(_)) {
        return comparison(
            normalize_nan_expr(other),
            op,
            raw_float_literal(bound, metadata),
        );
    }

    let Some(negative_infinity) = negative_infinity(&bound) else {
        return comparison(other.clone(), op, raw_float_literal(bound, metadata));
    };
    let primary = comparison(other.clone(), op, raw_float_literal(bound, metadata));
    match op {
        Operator::Lt | Operator::LtEq => primary.and(comparison(
            other.clone(),
            Operator::GtEq,
            raw_float_literal(negative_infinity, None),
        )),
        Operator::Gt | Operator::GtEq => primary.or(comparison(
            other.clone(),
            Operator::Lt,
            raw_float_literal(negative_infinity, None),
        )),
        _ => primary,
    }
}

fn rewrite_node(expr: &Expr, schema: &DFSchema) -> Option<Expr> {
    match expr {
        // DataFusion's simplifier expands an `IN` list of three or fewer values
        // over a bare column back into an OR chain of equalities, so a second
        // `optimize_expr` splits this rewrite's own output and re-runs it on each
        // half. Both halves then produce the same list, and dropping the repeat is
        // what makes the rewrite survive that round trip.
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
            if deduped.len() == kept.len() {
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
            let rewritten = match (left.as_ref(), right.as_ref()) {
                (_, Expr::Literal(value, metadata)) => {
                    rewrite_literal_comparison(left, *op, value, metadata.as_ref())
                }
                (Expr::Literal(value, metadata), _) => {
                    rewrite_literal_comparison(right, op.swap()?, value, metadata.as_ref())
                }
                _ if is_float_comparison(*op)
                    && is_float_expr(left, schema)
                    && is_float_expr(right, schema) =>
                {
                    Some(comparison(
                        normalize_nan_expr(left),
                        *op,
                        normalize_nan_expr(right),
                    ))
                }
                _ => None,
            }?;
            (rewritten != *expr).then_some(rewritten)
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
    use datafusion::prelude::{col, lit};
    use rstest::rstest;

    use super::*;

    fn rewrite(expr: Expr) -> Expr {
        let schema = DFSchema::try_from(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Float64, true),
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
    fn finite_range_moves_both_nan_signs_above_the_bound() {
        assert_eq!(
            rewrite(col("x").gt(lit(1.0))),
            expected_ordered("x", Operator::Gt, Float64(Some(1.0)))
        );
        assert_eq!(
            rewrite(col("x").lt_eq(lit(1.0))),
            expected_ordered("x", Operator::LtEq, Float64(Some(1.0)))
        );
    }

    #[test]
    fn nan_equality_covers_both_sign_encodings() {
        let positive = f64::from_bits(0x7ff8_0000_0000_0042);
        let negative = f64::from_bits(0xfff8_0000_0000_0042);
        assert_eq!(
            rewrite(col("x").eq(lit(negative))),
            Expr::InList(InList {
                expr: Box::new(col("x")),
                list: vec![lit(negative), lit(positive)],
                negated: false,
            })
        );
    }

    #[test]
    fn column_comparison_normalizes_each_nan_operand_once() {
        let rewritten = rewrite(col("x").lt(col("y")));
        let expected = compare(
            normalize_nan_expr(&col("x")),
            Operator::Lt,
            normalize_nan_expr(&col("y")),
        );
        assert_eq!(rewritten, expected);
        assert_eq!(rewrite(rewritten.clone()), rewritten);
    }

    #[test]
    fn nan_array_normalization_preserves_payload_and_non_nan_values() {
        let negative = f64::from_bits(0xfff8_0000_0000_0042);
        let positive = f64::from_bits(0x7ff8_0000_0000_0042);
        let input = Arc::new(Float64Array::from(vec![
            Some(negative),
            Some(-1.0),
            Some(positive),
            None,
        ])) as ArrayRef;
        let normalized = normalize_nan_array(input).unwrap();
        let normalized = normalized.as_primitive::<Float64Type>();
        assert_eq!(normalized.value(0).to_bits(), positive.to_bits());
        assert_eq!(normalized.value(1).to_bits(), (-1.0_f64).to_bits());
        assert_eq!(normalized.value(2).to_bits(), positive.to_bits());
        assert!(normalized.is_null(3));
    }

    #[rstest]
    #[case::eq(Operator::Eq, false)]
    #[case::not_eq(Operator::NotEq, true)]
    fn equality_covers_both_encodings(#[case] op: Operator, #[case] negated: bool) {
        assert_eq!(
            rewrite(compare(col("x"), op, lit(0.0))),
            Expr::InList(InList {
                expr: Box::new(col("x")),
                list: vec![lit(-0.0), lit(0.0)],
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
    // `IS [NOT] DISTINCT FROM` is missing because `Planner::parse_filter` rejects
    // it as unsupported SQL; that arm is reachable only from a programmatically
    // built expression, and `rewriting_twice_changes_nothing` covers it there.
    fn optimizing_twice_changes_nothing(#[case] filter: &str) {
        let schema =
            std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                "value",
                arrow_schema::DataType::Float64,
                true,
            )]));
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
