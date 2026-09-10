// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Rewrites of comparisons against a floating point zero literal.

use datafusion::error::Result as DFResult;
use datafusion::logical_expr::{BinaryExpr, Operator, expr::Between, expr::InList};
use datafusion::prelude::Expr;
use datafusion::scalar::ScalarValue::{self, Float16, Float32, Float64};
use datafusion_common::tree_node::{Transformed, TreeNode};
use half::f16;
use lance_core::Result;

/// Rewrite every comparison against a floating point zero literal into the form
/// that Arrow's total-order kernels answer the way IEEE 754 and SQL define it.
///
/// Arrow sorts `-0.0` strictly below `+0.0` and compares the two encodings for
/// equality by bit pattern, while IEEE 754 and SQL treat them as one number.
/// Each comparison against a zero literal has an equivalent total-order form:
///
/// | written                      | evaluated                    |
/// |------------------------------|------------------------------|
/// | `x < 0`, `x >= 0`            | literal becomes `-0.0`       |
/// | `x <= 0`, `x > 0`            | literal becomes `+0.0`       |
/// | `x = 0`                      | `x IN (-0.0, 0.0)`           |
/// | `x != 0`                     | `x NOT IN (-0.0, 0.0)`       |
/// | `x IN (0, ..)`               | the missing encoding is added |
/// | `0 IN (a, b)`                | `a IN (-0.0, 0.0) OR b IN (-0.0, 0.0)` |
/// | `x IS NOT DISTINCT FROM 0`   | `x IS NOT NULL AND x IN (-0.0, 0.0)` |
/// | `x IS DISTINCT FROM 0`       | `x IS NULL OR x NOT IN (-0.0, 0.0)` |
///
/// Equality has to name both encodings because a scalar index keys on the bit
/// pattern: the btree and bitmap indices order candidates by `total_cmp`, and the
/// bloom filter hashes the value.
///
/// Runs as the last step of [`crate::planner::Planner::optimize_expr`], after
/// coercion has given the literal the column's type and the simplifier has
/// expanded `BETWEEN` into two comparisons. Filters, computed output columns and
/// update expressions all compile through there, which is what keeps a filter and
/// a projected copy of the same predicate in agreement.
///
/// NaN is out of scope. Arrow sorts it above every other value, so `x >= -0.0`
/// admits NaN where IEEE would not, and that holds for every comparison rather
/// than only the ones against zero.
pub fn rewrite_signed_zero_comparisons(expr: Expr) -> Result<Expr> {
    Ok(expr
        .transform_up(|node| {
            Ok(match rewrite_node(&node) {
                Some(rewritten) => Transformed::yes(rewritten),
                None => Transformed::no(node),
            })
        })?
        .data)
}

/// Whether the rewrite acts on comparisons under `op`.
fn is_zero_sensitive(op: Operator) -> bool {
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

/// Fold each zero-sensitive comparison's own operands and rewrite it, bottom-up,
/// before anything above it has a chance to fold.
///
/// [`rewrite_signed_zero_comparisons`] alone cannot reach a comparison whose zero
/// does not exist yet. `ExprSimplifier::simplify` folds an operand and everything
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
pub fn normalize_zero_comparisons(
    expr: Expr,
    simplify: &dyn Fn(Expr) -> DFResult<Expr>,
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
                Expr::BinaryExpr(BinaryExpr { left, op, right }) if is_zero_sensitive(op) => {
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
            Ok(match rewrite_node(&folded) {
                Some(rewritten) => Transformed::yes(rewritten),
                // The operands were still folded, so this is a change either way.
                None => Transformed::yes(folded),
            })
        })?
        .data)
}

/// Both encodings of a floating point zero, negative first.
///
/// Returns `None` for anything else, including NULL, NaN, and integer zero.
fn zero_encodings(value: &ScalarValue) -> Option<(ScalarValue, ScalarValue)> {
    match value {
        Float16(Some(v)) if *v == f16::ZERO => {
            Some((Float16(Some(f16::NEG_ZERO)), Float16(Some(f16::ZERO))))
        }
        Float32(Some(v)) if *v == 0.0 => Some((Float32(Some(-0.0)), Float32(Some(0.0)))),
        Float64(Some(v)) if *v == 0.0 => Some((Float64(Some(-0.0)), Float64(Some(0.0)))),
        _ => None,
    }
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
/// against both encodings of a floating point zero, negated or not. Only these
/// terms are deduplicated, so an expression the caller wrote twice is left alone.
fn is_zero_pair_over_column(expr: &Expr) -> bool {
    let Expr::InList(InList { expr, list, .. }) = expr else {
        return false;
    };
    if !matches!(expr.as_ref(), Expr::Column(_)) {
        return false;
    }
    let [Expr::Literal(first, _), ..] = list.as_slice() else {
        return false;
    };
    zero_encodings(first)
        .is_some_and(|(negative, positive)| list_is_pair(list, &negative, &positive))
}

/// True when `list` is exactly the two encodings of a zero, negative first.
fn list_is_pair(list: &[Expr], negative: &ScalarValue, positive: &ScalarValue) -> bool {
    let [Expr::Literal(first, _), Expr::Literal(second, _)] = list else {
        return false;
    };
    first == negative && second == positive
}

/// The rewritten expression, or `None` when `expr` is not a comparison against a
/// floating point zero.
/// The encoding a zero bound needs to answer `op` correctly, or `None` when the
/// expression is not a zero literal.
fn rewrite_bound(bound: &Expr, op: Operator) -> Option<Expr> {
    let Expr::Literal(value, metadata) = bound else {
        return None;
    };
    let (negative, positive) = zero_encodings(value)?;
    let encoding = match op {
        Operator::GtEq => negative,
        Operator::LtEq => positive,
        _ => return None,
    };
    Some(Expr::Literal(encoding, metadata.clone()))
}

fn rewrite_node(expr: &Expr) -> Option<Expr> {
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
                if is_zero_pair_over_column(term) && deduped.contains(term) {
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
            let (literal, other, op) = match (left.as_ref(), right.as_ref()) {
                (_, Expr::Literal(..)) => (right.as_ref(), left.as_ref(), *op),
                (Expr::Literal(..), _) => (left.as_ref(), right.as_ref(), op.swap()?),
                _ => return None,
            };
            let Expr::Literal(value, metadata) = literal else {
                return None;
            };
            let (negative, positive) = zero_encodings(value)?;
            let zero = match op {
                Operator::Lt | Operator::GtEq => negative,
                Operator::LtEq | Operator::Gt => positive,
                Operator::Eq | Operator::NotEq => {
                    return Some(Expr::InList(InList {
                        expr: Box::new(other.clone()),
                        list: vec![
                            Expr::Literal(negative, metadata.clone()),
                            Expr::Literal(positive, metadata.clone()),
                        ],
                        negated: op == Operator::NotEq,
                    }));
                }
                Operator::IsNotDistinctFrom | Operator::IsDistinctFrom => {
                    // Both encodings have to be listed, and the null case has to
                    // stay decided rather than becoming NULL, so this pairs the
                    // list with `IS [NOT] TRUE`. `NULL IN (..)` is NULL, and
                    // `NULL IS TRUE` is false, which is what distinctness means
                    // for a null against a non-null literal.
                    //
                    // The list carries `other` once. An earlier version guarded
                    // this arm to a bare column so it could name `other` twice as
                    // `IS NOT NULL AND IN (..)`, but bailing out left the
                    // `filter_expr` path answering computed operands on Arrow's
                    // sign-sensitive order, which is wrong rows rather than an
                    // unsupported spelling.
                    //
                    // Always the non-negated list, so a second pass sees the same
                    // complete pair it would leave alone anywhere else.
                    let covered = Expr::InList(InList {
                        expr: Box::new(other.clone()),
                        list: vec![
                            Expr::Literal(negative, metadata.clone()),
                            Expr::Literal(positive, metadata.clone()),
                        ],
                        negated: false,
                    });
                    return Some(if op == Operator::IsDistinctFrom {
                        covered.is_not_true()
                    } else {
                        covered.is_true()
                    });
                }
                _ => return None,
            };
            Some(Expr::BinaryExpr(BinaryExpr {
                left: Box::new(other.clone()),
                op,
                right: Box::new(Expr::Literal(zero, metadata.clone())),
            }))
        }
        // `BETWEEN` normally reaches this rewrite already expanded into `>=` and
        // `<=` by the simplifier. It survives unexpanded when every operand is
        // constant, because then the simplifier expands and folds it in one pass
        // and the comparison is gone before the post-pass looks. The bounds take
        // the encodings their expanded operators would: `low` is a `>=` bound and
        // `high` is a `<=` bound.
        Expr::Between(between) => {
            let low = rewrite_bound(&between.low, Operator::GtEq);
            let high = rewrite_bound(&between.high, Operator::LtEq);
            if low.is_none() && high.is_none() {
                return None;
            }
            Some(Expr::Between(Between {
                expr: between.expr.clone(),
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
            // A zero literal on the probe side needs the same treatment. The list
            // elements are arbitrary expressions there, so expand into the
            // equality form the binary arm already covers. A literal probe that is
            // not a zero compares the same way against either encoding, so it
            // needs no widening either.
            if let Expr::Literal(value, metadata) = expr.as_ref() {
                let (negative, positive) = zero_encodings(value)?;
                // The expansion below puts a zero literal in front of exactly this
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
                list: widen_zero_list(list)?,
                negated: *negated,
            }))
        }
        _ => None,
    }
}

/// Add the missing encoding next to every floating point zero in an `IN` list.
///
/// Returns `None` when the list holds no zero, or already spells out both
/// encodings of each zero it holds.
fn widen_zero_list(list: &[Expr]) -> Option<Vec<Expr>> {
    // Most lists hold no zero, so collect what is missing before copying anything.
    let mut missing: Vec<Expr> = Vec::new();
    for item in list {
        let Expr::Literal(value, metadata) = item else {
            continue;
        };
        let Some((negative, positive)) = zero_encodings(value) else {
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
    use datafusion::prelude::{col, lit};
    use rstest::rstest;

    use super::*;

    fn rewrite(expr: Expr) -> Expr {
        rewrite_signed_zero_comparisons(expr).unwrap()
    }

    fn compare(left: Expr, op: Operator, right: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(left),
            op,
            right: Box::new(right),
        })
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
            compare(col("x"), op, lit(evaluated))
        );
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
            compare(col("x"), Operator::Lt, lit(-0.0))
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
            compare(col("x"), Operator::LtEq, Expr::Literal(evaluated, None))
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
            col("x").lt(lit(-0.0)).and(col("y").eq(lit(1.0)))
        );
    }

    #[rstest]
    #[case::non_zero(col("x").lt(lit(1.0)))]
    #[case::integer_zero(col("x").eq(lit(0_i64)))]
    #[case::null(compare(col("x"), Operator::Eq, Expr::Literal(Float64(None), None)))]
    #[case::nan(col("x").lt(lit(f64::NAN)))]
    #[case::column_on_both_sides(col("x").lt(col("y")))]
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
