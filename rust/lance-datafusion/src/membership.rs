// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Expose separated membership predicates to DataFusion's set simplification rules.

use std::collections::{HashMap, HashSet};
use std::mem;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use datafusion::scalar::ScalarValue;

// Computed memberships are never merged. Bound their structural comparisons
// so generated filters cannot turn this safety check into quadratic work.
const MAX_COMPUTED_MEMBERSHIPS: usize = 32;
// Avoid repeatedly scanning large nested regions merely to prove a boundary safe.
const MAX_BOUNDARY_TERMS: usize = 32;

/// A borrowed membership, including equality and explicit NOT spellings.
struct Membership<'a> {
    expr: &'a Expr,
    values: &'a [Expr],
    negated: bool,
}

enum Replacement {
    Keep,
    Remove,
    Replace(Expr),
}

enum Term<'a> {
    Membership(Membership<'a>),
    /// Columns inside a nested AND/OR boundary that could collapse into a
    /// membership once constant or absorbed branches simplify away.
    Boundary(Vec<&'a Column>),
    /// Negating or unwrapping a compound predicate can expose memberships after this pass.
    Barrier,
    Other,
}

/// Group separated memberships within each AND/OR region before simplification.
///
/// Replacements are applied together: moving one column's predicates must not
/// expose an unsafe group on another column to DataFusion's set algebra.
pub fn simplify_membership_regions(
    mut expr: Expr,
    simplify: &impl Fn(Expr) -> Result<Expr>,
) -> Result<Expr> {
    rewrite(&mut expr, simplify)?;
    Ok(expr)
}

fn rewrite(expr: &mut Expr, simplify: &impl Fn(Expr) -> Result<Expr>) -> Result<()> {
    match expr {
        Expr::BinaryExpr(binary) if matches!(binary.op, Operator::And | Operator::Or) => {
            let operator = binary.op;
            // Visit only this region. Boundary expressions recurse independently,
            // including regions inside NOT; no ancestor rescans those subtrees.
            let mut pending = vec![&mut *expr];
            let mut terms = Vec::new();
            let mut shape = Vec::new();
            while let Some(term) = pending.pop() {
                if matches!(term, Expr::BinaryExpr(binary) if binary.op == operator) {
                    shape.push(true);
                    if let Expr::BinaryExpr(binary) = term {
                        pending.push(binary.right.as_mut());
                        pending.push(binary.left.as_mut());
                    }
                } else {
                    shape.push(false);
                    rewrite(term, simplify)?;
                    terms.push(term);
                }
            }
            let replacements = plan_replacements(&terms, operator, simplify)?;
            if replacements.is_empty() {
                return Ok(());
            }
            let mut edits: Vec<_> = (0..terms.len()).map(|_| Replacement::Keep).collect();
            for (indices, replacement) in replacements {
                let last = indices[indices.len() - 1];
                for index in indices {
                    edits[index] = Replacement::Remove;
                }
                edits[last] = Replacement::Replace(replacement);
            }
            rebuild_region(expr, operator, &shape, &mut edits)?;
        }
        Expr::BinaryExpr(binary) => {
            rewrite(&mut binary.left, simplify)?;
            rewrite(&mut binary.right, simplify)?;
        }
        Expr::InList(list) => {
            rewrite(&mut list.expr, simplify)?;
            for value in &mut list.list {
                rewrite(value, simplify)?;
            }
        }
        Expr::Column(_) | Expr::Literal(_, _) => {}
        _ => {
            let original = mem::replace(expr, Expr::Literal(ScalarValue::Null, None));
            *expr = original
                .map_children(|mut child| {
                    rewrite(&mut child, simplify)?;
                    Ok(Transformed::yes(child))
                })?
                .data;
        }
    }
    Ok(())
}

fn is_eligible_literal(value: &ScalarValue) -> bool {
    if value.is_null() {
        return false;
    }
    let data_type = match value {
        ScalarValue::Dictionary(_, inner) => {
            if inner.is_null() {
                return false;
            }
            inner.data_type()
        }
        _ => value.data_type(),
    };
    !data_type.is_floating() && !data_type.is_nested()
}

fn plan_replacements(
    terms: &[&mut Expr],
    operator: Operator,
    simplify: &impl Fn(Expr) -> Result<Expr>,
) -> Result<Vec<(Vec<usize>, Expr)>> {
    // Column lookup is constant-time. Keep only a bounded list of computed
    // expressions to detect unsafe repeats without hashing Exprs containing UDFs.
    let mut columns: HashMap<&Column, usize> = HashMap::new();
    let mut boundary_columns: HashSet<&Column> = HashSet::new();
    let mut groups: Vec<Vec<(usize, Membership<'_>)>> = Vec::new();
    let mut computed: Vec<(Membership<'_>, Vec<&Column>)> = Vec::new();
    for (index, term) in terms.iter().enumerate() {
        let membership = match classify_term(term)? {
            Term::Membership(membership) => membership,
            Term::Boundary(cols) => {
                // Reject boundaries whose columns overlap another membership or
                // boundary in this region (e.g. `(s NOT IN (...) OR FALSE)` or
                // `(s IN (...) AND TRUE)` collapsing beside an opposing `s` term).
                for col in cols {
                    if columns.contains_key(col)
                        || !boundary_columns.insert(col)
                        || computed.iter().any(|(_, c_cols)| c_cols.contains(&col))
                    {
                        return Ok(Vec::new());
                    }
                }
                continue;
            }
            Term::Barrier => return Ok(Vec::new()),
            Term::Other => continue,
        };
        let group_index = if let Expr::Column(column) = membership.expr {
            if boundary_columns.contains(column) {
                return Ok(Vec::new());
            }
            *columns.entry(column).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            })
        } else {
            if computed.len() == MAX_COMPUTED_MEMBERSHIPS
                || computed
                    .iter()
                    .any(|(prev, _)| prev.expr == membership.expr)
            {
                return Ok(Vec::new());
            }
            let mut cols = Vec::new();
            collect_columns(membership.expr, &mut cols)?;
            if cols.iter().any(|col| boundary_columns.contains(col)) {
                return Ok(Vec::new());
            }
            computed.push((membership, cols));
            continue;
        };
        groups[group_index].push((index, membership));
    }

    let mut replacements = Vec::with_capacity(groups.len());
    for group in groups {
        if group.len() < 2 {
            continue;
        }
        let has_positive = group.iter().any(|(_, membership)| !membership.negated);
        let has_negative = group.iter().any(|(_, membership)| membership.negated);
        if operator == Operator::Or && has_positive && has_negative {
            return Ok(Vec::new());
        }
        // Reject known-ineligible operands across the whole group before
        // cloning even its first list or folding non-literal operands.
        if group.iter().any(|(_, membership)| {
            membership.values.is_empty()
                || membership.values.iter().any(
                    |value| matches!(value, Expr::Literal(value, _) if !is_eligible_literal(value)),
                )
        }) {
            return Ok(Vec::new());
        }
        let mut indices = Vec::with_capacity(group.len());
        let mut level = Vec::with_capacity(group.len());
        for (index, membership) in group {
            // Clone operands only after a repeated column is found. Fold casts
            // on constants without expanding the membership predicate itself.
            let values = membership
                .values
                .iter()
                .map(|value| match value {
                    Expr::Literal(_, _) => Ok(value.clone()),
                    _ => simplify(value.clone()),
                })
                .collect::<Result<Vec<_>>>()?;
            // Float normalization may introduce NaN and signed-zero alternatives.
            // Do not reduce them using structural literal equality. NULL and nested
            // list elements also do not obey the ordinary set-algebra rules.
            if !values
                .iter()
                .all(|value| matches!(value, Expr::Literal(value, _) if is_eligible_literal(value)))
            {
                return Ok(Vec::new());
            }
            indices.push(index);
            level.push(Expr::InList(InList::new(
                Box::new(membership.expr.clone()),
                values,
                membership.negated,
            )));
        }
        // Keep the input to the recursive DataFusion simplifier logarithmic in
        // the number of predicates, even for a long original AND/OR chain.
        while level.len() > 1 {
            let mut parents = Vec::with_capacity(level.len().div_ceil(2));
            let mut children = level.into_iter();
            while let Some(left) = children.next() {
                parents.push(match children.next() {
                    Some(right) => {
                        Expr::BinaryExpr(BinaryExpr::new(Box::new(left), operator, Box::new(right)))
                    }
                    None => left,
                });
            }
            level = parents;
        }
        let simplified = simplify(level.remove(0))?;
        // A constant drops UNKNOWN. A CASE guard introduces residual filtering
        // and can disable an enclosing OR's index plan. Preserve the whole
        // region instead, including separators belonging to other columns.
        if matches!(simplified, Expr::Literal(ScalarValue::Boolean(_), _)) {
            return Ok(Vec::new());
        }
        replacements.push((indices, simplified));
    }

    if !replacements.is_empty() {
        for (i, (membership, cols)) in computed.iter().enumerate() {
            let overlaps = cols.iter().any(|col| {
                columns.contains_key(col)
                    || computed[..i].iter().any(|(_, prev)| prev.contains(col))
                    || computed[i + 1..].iter().any(|(_, next)| next.contains(col))
            });
            if overlaps {
                // Reject computed memberships whose LHS simplifies (e.g. `coalesce(s)`
                // or `i * 1`) into a predicate that can collide with a sibling term.
                let probe = simplify(Expr::InList(InList::new(
                    Box::new(membership.expr.clone()),
                    membership.values.to_vec(),
                    membership.negated,
                )))?;
                if !matches!(
                    classify_term(&probe)?,
                    Term::Membership(simplified) if simplified.expr == membership.expr
                ) {
                    return Ok(Vec::new());
                }
            }
        }
    }

    Ok(replacements)
}

fn has_column(expr: &Expr) -> Result<bool> {
    let mut found = false;
    expr.apply(|node| {
        if matches!(node, Expr::Column(_)) {
            found = true;
            Ok(TreeNodeRecursion::Stop)
        } else {
            Ok(TreeNodeRecursion::Continue)
        }
    })?;
    Ok(found)
}

fn collect_columns<'a>(expr: &'a Expr, columns: &mut Vec<&'a Column>) -> Result<()> {
    expr.apply(|node| {
        if let Expr::Column(column) = node
            && !columns.contains(&column)
        {
            columns.push(column);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(())
}

/// Look through scalar wrappers without evaluating them or cloning their data.
/// A boolean-valued operation or column cast below a wrapper may become a
/// region term later.
fn contains_predicate(expr: &Expr) -> Result<bool> {
    let mut found = false;
    expr.apply(|node| {
        if matches!(
            node,
            Expr::InList(_)
                | Expr::Between(_)
                | Expr::Like(_)
                | Expr::SimilarTo(_)
                | Expr::Not(_)
                | Expr::IsTrue(_)
                | Expr::IsFalse(_)
                | Expr::IsNotTrue(_)
                | Expr::IsNotFalse(_)
                | Expr::IsUnknown(_)
                | Expr::IsNotUnknown(_)
                | Expr::IsNull(_)
                | Expr::IsNotNull(_)
                | Expr::Case(_)
        ) || matches!(node, Expr::BinaryExpr(binary) if matches!(binary.op,
                Operator::Eq | Operator::NotEq | Operator::Lt | Operator::LtEq
                | Operator::Gt | Operator::GtEq | Operator::And | Operator::Or
                | Operator::IsDistinctFrom | Operator::IsNotDistinctFrom
                | Operator::RegexMatch | Operator::RegexIMatch
                | Operator::RegexNotMatch | Operator::RegexNotIMatch
                | Operator::LikeMatch | Operator::ILikeMatch
                | Operator::NotLikeMatch | Operator::NotILikeMatch))
            || matches!(node, Expr::Cast(_) | Expr::TryCast(_) if has_column(node)?)
        {
            found = true;
            Ok(TreeNodeRecursion::Stop)
        } else {
            Ok(TreeNodeRecursion::Continue)
        }
    })?;
    Ok(found)
}

fn is_safe_membership(membership: &Membership<'_>) -> Result<bool> {
    if matches!(membership.expr, Expr::Column(_)) {
        for value in membership.values {
            if !matches!(value, Expr::Literal(_, _) | Expr::Column(_))
                && (has_column(value)? || contains_predicate(value)?)
            {
                return Ok(false);
            }
        }
        Ok(true)
    } else {
        Ok(!contains_predicate(membership.expr)?
            && membership.values.iter().all(|value| {
                matches!(
                    value,
                    Expr::Literal(scalar, _) if !matches!(scalar, ScalarValue::Boolean(_))
                )
            }))
    }
}

fn classify_term(mut term: &Expr) -> Result<Term<'_>> {
    let mut negated = false;
    while let Expr::Not(inner) = term {
        negated = !negated;
        term = inner;
    }
    let classified = match term {
        Expr::InList(list) if matches!(list.expr.as_ref(), Expr::Literal(_, _)) => {
            if list
                .list
                .iter()
                .all(|value| matches!(value, Expr::Literal(_, _)))
            {
                Term::Other
            } else {
                return Ok(Term::Barrier);
            }
        }
        Expr::InList(list) => Term::Membership(Membership {
            expr: &list.expr,
            values: &list.list,
            negated: negated ^ list.negated,
        }),
        Expr::BinaryExpr(binary) if matches!(binary.op, Operator::Eq | Operator::NotEq) => {
            let (expr, value) = match (binary.left.as_ref(), binary.right.as_ref()) {
                (Expr::Column(_), Expr::Column(_)) | (Expr::Literal(_, _), Expr::Literal(_, _)) => {
                    return Ok(Term::Other);
                }
                (expr @ Expr::Column(_), value)
                | (value, expr @ Expr::Column(_))
                | (expr, value @ Expr::Literal(_, _))
                | (value @ Expr::Literal(_, _), expr) => (expr, value),
                _ => return Ok(Term::Barrier),
            };
            Term::Membership(Membership {
                expr,
                values: std::slice::from_ref(value),
                negated: negated ^ (binary.op == Operator::NotEq),
            })
        }
        Expr::BinaryExpr(binary)
            if matches!(
                binary.op,
                Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
            ) && matches!((binary.left.as_ref(), binary.right.as_ref()),
                    (Expr::Column(_), Expr::Literal(value, _))
                    | (Expr::Literal(value, _), Expr::Column(_))
                    if !value.is_null() && !matches!(value, ScalarValue::Boolean(_))) =>
        {
            // Negation only flips an ordinary ordering comparison's operator;
            // it cannot expose IN/Eq predicates or distribute a boolean region.
            Term::Other
        }
        Expr::BinaryExpr(binary) if matches!(binary.op, Operator::And | Operator::Or) => {
            if negated {
                return Ok(Term::Barrier);
            }
            // A boundary is safe only if every term is understood and its columns
            // do not overlap another membership or boundary in the enclosing region.
            let mut pending = vec![term];
            let mut visited = 0;
            let mut boundary_columns = Vec::new();
            while let Some(child) = pending.pop() {
                visited += 1;
                if visited > MAX_BOUNDARY_TERMS {
                    return Ok(Term::Barrier);
                }
                match child {
                    Expr::BinaryExpr(inner) if matches!(inner.op, Operator::And | Operator::Or) => {
                        pending.push(&inner.left);
                        pending.push(&inner.right);
                    }
                    child => match classify_term(child)? {
                        Term::Membership(membership)
                            if matches!(membership.expr, Expr::Column(_)) =>
                        {
                            collect_columns(child, &mut boundary_columns)?;
                        }
                        Term::Other => {
                            collect_columns(child, &mut boundary_columns)?;
                        }
                        Term::Membership(_) | Term::Boundary(_) | Term::Barrier => {
                            return Ok(Term::Barrier);
                        }
                    },
                }
            }
            return Ok(Term::Boundary(boundary_columns));
        }
        // Only explicit safe separators are allowed. Unknown predicates may
        // simplify into memberships, even if they contain no predicate today.
        Expr::Column(_) | Expr::Literal(_, _) | Expr::IsNull(_) | Expr::IsNotNull(_) => Term::Other,
        _ => Term::Barrier,
    };
    // Boolean wrappers may disappear in a later simplifier pass. Keep their
    // separators unless this is an independently validated AND/OR boundary.
    let is_safe = match &classified {
        Term::Membership(membership) => is_safe_membership(membership)?,
        Term::Boundary(_) | Term::Barrier => return Ok(classified),
        Term::Other => {
            let mut found = false;
            term.apply_children(|child| {
                found = contains_predicate(child)?;
                Ok(if found {
                    TreeNodeRecursion::Stop
                } else {
                    TreeNodeRecursion::Continue
                })
            })?;
            !found
        }
    };
    Ok(if is_safe { classified } else { Term::Barrier })
}

/// Prune replaced terms without changing the parenthesization of those kept.
fn rebuild_region(
    expr: &mut Expr,
    operator: Operator,
    shape: &[bool],
    edits: &mut [Replacement],
) -> Result<()> {
    let original = mem::replace(expr, Expr::Literal(ScalarValue::Null, None));
    let mut pending = vec![Some(original)];
    let mut rebuilt = Vec::new();
    let mut index = 0;
    let mut nodes = shape.iter();
    while let Some(item) = pending.pop() {
        match item {
            Some(term) => {
                // A rewritten boundary can now have this region's operator.
                // Follow the original shape so it still occupies one term.
                if nodes.next() == Some(&true) {
                    let Expr::BinaryExpr(binary) = term else {
                        return Err(datafusion::common::internal_datafusion_err!(
                            "Membership region lost a binary node"
                        ));
                    };
                    pending.push(None);
                    pending.push(Some(*binary.right));
                    pending.push(Some(*binary.left));
                } else {
                    rebuilt.push(match mem::replace(&mut edits[index], Replacement::Keep) {
                        Replacement::Keep => Some(term),
                        Replacement::Remove => None,
                        Replacement::Replace(expr) => Some(expr),
                    });
                    index += 1;
                }
            }
            None => {
                let right = rebuilt.pop().flatten();
                let left = rebuilt.pop().flatten();
                rebuilt.push(match (left, right) {
                    (Some(left), Some(right)) => Some(Expr::BinaryExpr(BinaryExpr::new(
                        Box::new(left),
                        operator,
                        Box::new(right),
                    ))),
                    (remaining, None) | (None, remaining) => remaining,
                });
            }
        }
    }
    *expr = rebuilt.pop().flatten().ok_or_else(|| {
        datafusion::common::internal_datafusion_err!(
            "Membership simplification removed every region term"
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::logical_expr::{col, lit};
    use std::cell::Cell;

    #[test]
    fn preserves_unmerged_regions_without_simplifier_calls() {
        let mut level: Vec<_> = (0..1024).map(|i| col(format!("c{i}")).eq(lit(i))).collect();
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| pair[0].clone().and(pair[1].clone()))
                .collect();
        }
        let original = level.pop().unwrap();
        let calls = Cell::new(0);
        let rewritten = simplify_membership_regions(original.clone(), &|expr| {
            calls.set(calls.get() + 1);
            Ok(expr)
        })
        .unwrap();
        assert_eq!(rewritten, original);
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn preserves_allocations_when_no_column_repeats() {
        let original = col("s")
            .in_list((0..100_000).map(lit).collect(), false)
            .and(col("i").lt(lit(1)));
        let Expr::BinaryExpr(binary) = &original else {
            unreachable!()
        };
        let left = binary.left.as_ref() as *const Expr;
        let Expr::InList(list) = binary.left.as_ref() else {
            unreachable!()
        };
        let values = list.list.as_ptr();
        let rewritten = simplify_membership_regions(original, &|_| {
            panic!("A single membership must not invoke the simplifier")
        })
        .unwrap();
        let Expr::BinaryExpr(binary) = rewritten else {
            unreachable!()
        };
        assert_eq!(binary.left.as_ref() as *const Expr, left);
        let Expr::InList(list) = binary.left.as_ref() else {
            unreachable!()
        };
        assert_eq!(list.list.as_ptr(), values);
        assert_eq!(list.list.len(), 100_000);
    }

    #[rstest::rstest]
    #[case::float(col("f"), vec![lit(1.0), lit(2.0)])]
    #[case::null_element(col("f"), vec![lit(ScalarValue::Int32(None))])]
    #[case::computed(col("f") + lit(1), vec![lit(1), lit(2)])]
    fn preserves_repeated_ineligible_memberships(#[case] expr: Expr, #[case] values: Vec<Expr>) {
        let original = expr
            .clone()
            .in_list(values.clone(), false)
            .and(col("s").eq(lit("a")))
            .and(expr.in_list(values, true))
            .and(col("s").eq(lit("a")));
        let rewritten = simplify_membership_regions(original.clone(), &|_| {
            panic!("Repeated ineligible memberships must preserve the region")
        })
        .unwrap();
        assert_eq!(rewritten, original);
    }

    #[rstest::rstest]
    #[case::float(lit(1.0))]
    #[case::null(lit(ScalarValue::Int32(None)))]
    fn rejects_known_ineligible_literals_before_folding(#[case] ineligible: Expr) {
        let original = col("f")
            .in_list(vec![col("other")], false)
            .and(col("f").in_list(vec![ineligible], false));
        let rewritten = simplify_membership_regions(original.clone(), &|_| {
            panic!("Known-ineligible literals must reject the group before operand folding")
        })
        .unwrap();
        assert_eq!(rewritten, original);
    }

    #[rstest::rstest]
    #[case::bounded(MAX_COMPUTED_MEMBERSHIPS, true)]
    #[case::over_limit(MAX_COMPUTED_MEMBERSHIPS + 1, false)]
    #[case::large(1024, false)]
    fn bounds_computed_membership_detection(#[case] count: usize, #[case] merges: bool) {
        let mut level: Vec<_> = (0..count)
            .map(|i| (col(format!("c{i}")) + lit(1)).in_list(vec![lit(2)], false))
            .collect();
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut terms = level.into_iter();
            while let Some(left) = terms.next() {
                next.push(match terms.next() {
                    Some(right) => left.and(right),
                    None => left,
                });
            }
            level = next;
        }
        let original = col("s")
            .in_list(vec![lit("a"), lit("b")], false)
            .and(level.pop().unwrap())
            .and(col("s").in_list(vec![lit("b")], false));
        let calls = Cell::new(0);
        let rewritten = simplify_membership_regions(original.clone(), &|_| {
            calls.set(calls.get() + 1);
            Ok(col("s").eq(lit("b")))
        })
        .unwrap();
        assert_eq!(calls.get(), usize::from(merges));
        if merges {
            assert_ne!(rewritten, original);
        } else {
            assert_eq!(rewritten, original);
        }
    }

    #[test]
    fn preserves_mixed_or_candidates() {
        let original = col("s")
            .eq(lit("a"))
            .or(col("i").eq(lit(1)))
            .or(col("s").not_eq(lit("b")))
            .or(col("i").eq(lit(1)));
        let rewritten = simplify_membership_regions(original.clone(), &|_| {
            panic!("Mixed-polarity OR must reject the region before simplifying a group")
        })
        .unwrap();
        assert_eq!(rewritten, original);
    }

    #[rstest::rstest]
    #[case::empty_intersection(Operator::And, false)]
    #[case::empty_negated_intersection(Operator::Or, true)]
    fn preserves_regions_when_simplification_is_constant(
        #[case] operator: Operator,
        #[case] negated: bool,
    ) {
        let first = col("s").in_list(vec![lit("a"), lit("b")], negated);
        let middle = col("i").eq(lit(1));
        let last = col("s").in_list(vec![lit("c"), lit("d")], negated);
        let original = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(first),
            operator,
            Box::new(Expr::BinaryExpr(BinaryExpr::new(
                Box::new(middle),
                operator,
                Box::new(last),
            ))),
        ));
        let calls = Cell::new(0);
        let rewritten = simplify_membership_regions(original.clone(), &|_| {
            calls.set(calls.get() + 1);
            Ok(lit(negated))
        })
        .unwrap();
        assert_eq!(rewritten, original);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn discards_staged_merges_when_another_group_needs_fallback() {
        let original = col("s")
            .eq(lit("a"))
            .and(col("t").eq(lit("a")))
            .and(col("s").not_eq(lit("a")))
            .and(col("t").eq(lit("a")));
        let calls = Cell::new(0);
        let rewritten = simplify_membership_regions(original.clone(), &|expr| {
            calls.set(calls.get() + 1);
            // Simulate a successful group followed by an unsafe constant.
            if calls.get() == 1 {
                Ok(expr)
            } else {
                Ok(lit(false))
            }
        })
        .unwrap();
        assert_eq!(calls.get(), 2);
        assert_eq!(rewritten, original);
    }

    #[rstest::rstest]
    #[case::column_comparison(col("t").eq(col("u")))]
    #[case::literal_comparison(col("i").eq(lit(1)))]
    #[case::disjoint_boundary(col("u").in_list(vec![lit("x"), lit("y")], true).or(lit(false)))]
    fn groups_separated_memberships_once(#[case] separator: Expr) {
        let original = col("s")
            .in_list(vec![lit("a"), lit("b")], false)
            .and(separator)
            .and(col("s").in_list(vec![lit("b"), lit("c")], false));
        let calls = Cell::new(0);
        // Even if simplification leaves the group unchanged, do not revisit it.
        let rewritten = simplify_membership_regions(original.clone(), &|expr| {
            calls.set(calls.get() + 1);
            Ok(expr)
        })
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_ne!(rewritten, original);
    }

    #[rstest::rstest]
    #[case::and_with_or_false(
        Operator::And,
        col("s").in_list(vec![lit("a"), lit("b"), lit("c"), lit("d")], false),
        col("s")
            .in_list(vec![lit("a"), lit("b"), lit("c"), lit("d")], true)
            .or(lit(false))
    )]
    #[case::or_with_and_true(
        Operator::Or,
        col("s").in_list(vec![lit("a"), lit("b")], true),
        col("s").in_list(vec![lit("c"), lit("d")], true).and(lit(true))
    )]
    #[case::two_boundaries_same_column(
        Operator::And,
        col("s")
            .in_list(vec![lit("a"), lit("b"), lit("c"), lit("d")], false)
            .or(lit(false)),
        col("s")
            .in_list(vec![lit("a"), lit("b"), lit("c"), lit("d")], true)
            .or(lit(false))
    )]
    fn preserves_overlapping_boundary_columns(
        #[case] operator: Operator,
        #[case] first: Expr,
        #[case] boundary: Expr,
    ) {
        let join = |left: Expr, right: Expr| {
            Expr::BinaryExpr(BinaryExpr::new(Box::new(left), operator, Box::new(right)))
        };
        let sep = col("t").eq(lit("a"));
        let original = join(join(join(first, sep.clone()), boundary), sep);
        let rewritten = simplify_membership_regions(original.clone(), &|_| {
            panic!("Overlapping boundary columns must reject the region before simplifying")
        })
        .unwrap();
        assert_eq!(rewritten, original);
    }
}
