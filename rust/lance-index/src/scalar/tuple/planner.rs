// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plan a conjunctive filter as a [`TupleQuery`] over ordered key columns.
//!
//! [`plan_tuple_predicate`] reads each conjunct on its own. It returns `None`
//! when no conjunct binds the first key, and the caller keeps every conjunct.
//! Otherwise it returns a [`TuplePlan`] that sorts each conjunct by these rules:
//!
//! - `key = v`, `key IN (...)`, `key IS NULL`, and an OR whose terms are all
//!   `key = v` or `key IN (...)` on one key fix a key to a set of values. A
//!   bare Boolean key means `key = true`, and `NOT key` means `key = false`,
//!   alone or as an OR term. `Planner::optimize_expr` writes a short
//!   `key IN (...)` list as such an OR, `key = true` as `key`, `key = false`
//!   as `NOT key`, and `x = 0.0` on a float key as `x IN (-0.0, 0.0)`. These
//!   conjuncts bind a leading prefix of keys: key 0, then key 1, and so on, up
//!   to the first key that no conjunct fixes or whose smallest set would grow
//!   the prefix past the seek budget. When several conjuncts fix one key, the
//!   one with the smallest set binds it. A set of more distinct values than
//!   the budget never binds, even when an earlier key's set is empty.
//! - `key < v`, `key <= v`, `key > v`, `key >= v`, `key BETWEEN v AND w`,
//!   `key IS NOT NULL`, and an AND or OR of these on one key select a union of
//!   ranges on that key. Conjuncts on the key after the prefix bind, and the
//!   query keeps the intersection of their unions, with overlapping and
//!   touching ranges merged. So `a < 1 OR a > 5` binds two ranges, and
//!   `(a < 1 OR a > 5) AND a < 9` binds `(a < 1 OR (a > 5 AND a < 9))`. A
//!   range OR of more terms than the seek budget never binds, even when they
//!   would merge into fewer ranges, and neither does a conjunct that would
//!   raise the query's seeks past the budget. `Planner::optimize_expr` writes
//!   `x >= 0.0` on a float key as `x >= -0.0 OR x < -inf`, which also selects
//!   the NaNs with the sign bit set, so it binds two ranges.
//! - Every other conjunct that reads only key columns joins the key residual,
//!   which the index evaluates on decoded keys. This includes any other OR or
//!   NOT, `!=`, `NOT IN`, expressions over keys, and conjuncts on keys after the
//!   range.
//! - A conjunct that reads any other column goes back to the caller untouched,
//!   in the row residual. A column qualified by a relation, such as `l.tenant`,
//!   counts as outside the key, because the keys carry no relation to match it.
//!
//! The seek budget bounds the seeks a plan costs an index, which runs one seek
//! per prefix tuple and range, or one per prefix tuple when no range binds. It
//! is [`DEFAULT_MAX_TUPLE_SEEKS`] unless the `LANCE_TUPLE_MAX_SEEKS`
//! environment variable sets another. With the default, `tenant IN (5,000
//! values) AND (ts < 1 OR ts > 5)` binds the prefix and leaves the OR, which
//! would cost 10,000 seeks, in the key residual.
//!
//! Literals, casts of literals, and `arrow_cast` of literals are coerced to the
//! key type when the coercion is lossless. In a returned plan, any other literal
//! leaves its conjunct in the key residual.
//!
//! NULL literals follow SQL, where a comparison with NULL is never TRUE.
//! `key = NULL`, alone or as an OR term, and the NULL items of `key IN (...)`
//! match no row, so the planner drops them from the key's set. In a filter,
//! `x OR NULL` selects the rows that `x` selects, so the planner drops a NULL
//! literal OR term too. `key = NULL` and `key IN (NULL)` leave an empty set,
//! which selects no rows. In a returned plan, a conjunct with a NULL range
//! bound stays in the key residual. Only `key IS NULL` matches NULL keys.
//!
//! `Planner::optimize_expr` is in `lance-datafusion`.

use std::collections::HashSet;
use std::ffi::OsStr;
use std::num::NonZeroUsize;
use std::ops::Bound;
use std::sync::LazyLock;

use arrow_schema::{DataType, FieldRef, Fields};
use datafusion_common::ScalarValue;
use datafusion_expr::expr::{Between, BinaryExpr, Cast, InList, ScalarFunction};
use datafusion_expr::utils::split_binary;
use datafusion_expr::{Expr, Operator};
use lance_datafusion_exec::expr::safe_coerce_scalar;

use super::{Cut, KeyRange, KeyValues, TupleQuery};

/// The default seek budget, the most seeks one planned [`TupleQuery`] may cost.
///
/// The prefix covers the cross product of its keys' sets, and an index runs one
/// seek per prefix tuple and range, or one per prefix tuple when the query has
/// no ranges. The [module documentation](self) lists the conjuncts the budget
/// keeps out of the query.
///
/// The `LANCE_TUPLE_MAX_SEEKS` environment variable, read once per process,
/// sets another budget. When it is set to anything but a positive integer that
/// fits in a `usize`, the planner logs a warning and keeps this default.
pub const DEFAULT_MAX_TUPLE_SEEKS: usize = 8192;

static MAX_TUPLE_SEEKS: LazyLock<usize> = LazyLock::new(|| {
    parse_max_tuple_seeks(std::env::var_os("LANCE_TUPLE_MAX_SEEKS").as_deref()).unwrap_or_else(
        |warning| {
            log::warn!("{warning}");
            DEFAULT_MAX_TUPLE_SEEKS
        },
    )
});

fn parse_max_tuple_seeks(value: Option<&OsStr>) -> Result<usize, String> {
    let Some(value) = value else {
        return Ok(DEFAULT_MAX_TUPLE_SEEKS);
    };
    value
        .to_str()
        .and_then(|value| value.parse::<NonZeroUsize>().ok())
        .map(NonZeroUsize::get)
        .ok_or_else(|| {
            format!(
                "Failed to parse LANCE_TUPLE_MAX_SEEKS {value:?} as a positive integer, using default {DEFAULT_MAX_TUPLE_SEEKS}"
            )
        })
}

/// A conjunctive filter, split by what a [`TupleQuery`] answers.
///
/// Each input conjunct is in exactly one of `consumed`, `key_residual`, and
/// `row_residual`, in input order.
#[derive(Clone, Debug, PartialEq)]
pub struct TuplePlan<'a> {
    /// The key tuples to search.
    pub query: TupleQuery,
    /// Conjuncts whose AND selects exactly the rows `query` selects.
    pub consumed: Vec<&'a Expr>,
    /// Conjuncts that read only key columns, constants included, but that
    /// `query` does not express.
    pub key_residual: Vec<&'a Expr>,
    /// Conjuncts that read a column outside the key, including any column
    /// qualified by a relation.
    pub row_residual: Vec<&'a Expr>,
}

enum Constraint {
    Values { key: usize, values: KeyValues },
    Ranges { key: usize, ranges: Vec<KeyRange> },
}

/// Plan `conjuncts`, the terms of an AND, as one query over `keys`, the index's
/// key columns in key order.
///
/// Pass conjuncts that went through `Planner::optimize_expr` in
/// `lance-datafusion`. [`create_filter_plan`] expects the same pass before it
/// plans per-column indices. That pass coerces literal types, so
/// `at >= '2024-01-01'` on a timestamp key arrives here with a timestamp
/// literal. This function never parses a string into a timestamp, so without
/// that pass the raw string leaves its conjunct unplanned.
///
/// Returns `None` when no conjunct binds the first key, and the caller keeps
/// every conjunct. When the smallest set on the first key holds more values
/// than the seek budget (see [`DEFAULT_MAX_TUPLE_SEEKS`]), no set binds it, so
/// the result is `None` unless a range binds that key.
///
/// [`create_filter_plan`]: crate::scalar::expression::create_filter_plan
///
/// ```
/// use arrow_schema::{DataType, Field, Fields};
/// use datafusion_expr::{col, lit};
/// use lance_index::scalar::tuple::planner::plan_tuple_predicate;
///
/// let keys = Fields::from(vec![
///     Field::new("tenant", DataType::Int32, true),
///     Field::new("ts", DataType::Int64, true),
/// ]);
/// let conjuncts = [
///     col("tenant").in_list(vec![lit(1), lit(2)], false),
///     col("ts").gt_eq(lit(10_i64)),
///     col("ts").lt(lit(20_i64)),
///     col("status").eq(lit("a")),
/// ];
/// let plan = plan_tuple_predicate(&conjuncts, &keys).expect("tenant binds the first key");
/// assert_eq!(
///     plan.query.to_string(),
///     "TupleQuery[tenant IN (1, 2), ts >= 10 AND ts < 20]"
/// );
/// assert_eq!(plan.row_residual, vec![&conjuncts[3]]);
/// ```
pub fn plan_tuple_predicate<'a>(conjuncts: &'a [Expr], keys: &Fields) -> Option<TuplePlan<'a>> {
    plan(conjuncts, keys, *MAX_TUPLE_SEEKS)
}

fn plan<'a>(conjuncts: &'a [Expr], keys: &Fields, max_seeks: usize) -> Option<TuplePlan<'a>> {
    let mut row_residual = Vec::new();
    let mut key_conjuncts = Vec::new();
    for conjunct in conjuncts {
        let reads_only_keys = conjunct
            .column_refs()
            .iter()
            .all(|column| column.relation.is_none() && keys.find(&column.name).is_some());
        if reads_only_keys {
            key_conjuncts.push((conjunct, constraint(conjunct, keys, max_seeks)));
        } else {
            row_residual.push(conjunct);
        }
    }
    let mut is_consumed = vec![false; key_conjuncts.len()];

    let mut prefix = Vec::new();
    let mut tuples = 1_usize;
    for index in 0..keys.len() {
        let smallest = key_conjuncts
            .iter()
            .enumerate()
            .filter_map(|(position, (_, constraint))| match constraint {
                Some(Constraint::Values { key, values }) if *key == index => {
                    Some((position, set_size(values)))
                }
                _ => None,
            })
            .min_by_key(|(_, size)| *size);
        let Some((position, size)) = smallest else {
            break;
        };
        let Some(product) = tuples
            .checked_mul(size)
            .filter(|product| *product <= max_seeks)
        else {
            break;
        };
        tuples = product;
        is_consumed[position] = true;
        let Some(Constraint::Values { values, .. }) = key_conjuncts[position].1.take() else {
            unreachable!("position {position} holds the smallest set on key {index}");
        };
        prefix.push(values);
    }

    let mut ranges: Option<Vec<KeyRange>> = None;
    for (position, (_, constraint)) in key_conjuncts.iter().enumerate() {
        let Some(Constraint::Ranges {
            key,
            ranges: conjunct_ranges,
        }) = constraint
        else {
            continue;
        };
        if *key != prefix.len() {
            continue;
        }
        let bound = match &ranges {
            None => Some(conjunct_ranges.clone()),
            Some(bound) => intersect(bound, conjunct_ranges),
        };
        let Some(bound) = bound.filter(|bound| {
            tuples
                .checked_mul(bound.len())
                .is_some_and(|seeks| seeks <= max_seeks)
        }) else {
            continue;
        };
        ranges = Some(bound);
        is_consumed[position] = true;
    }

    if prefix.is_empty() && ranges.is_none() {
        return None;
    }
    let mut consumed = Vec::new();
    let mut key_residual = Vec::new();
    for ((conjunct, _), is_consumed) in key_conjuncts.into_iter().zip(is_consumed) {
        if is_consumed {
            consumed.push(conjunct);
        } else {
            key_residual.push(conjunct);
        }
    }
    Some(TuplePlan {
        query: TupleQuery {
            keys: keys.clone(),
            prefix,
            ranges,
        },
        consumed,
        key_residual,
        row_residual,
    })
}

fn set_size(values: &KeyValues) -> usize {
    match values {
        KeyValues::Null => 1,
        KeyValues::In(values) => values.len(),
    }
}

fn constraint(conjunct: &Expr, keys: &Fields, max_seeks: usize) -> Option<Constraint> {
    let ranges = || {
        key_ranges(conjunct, keys, max_seeks)
            .map(|(key, ranges)| Constraint::Ranges { key, ranges })
    };
    match conjunct {
        Expr::IsNull(expr) => Some(Constraint::Values {
            key: key_column(expr, keys)?.0,
            values: KeyValues::Null,
        }),
        Expr::BinaryExpr(BinaryExpr {
            op: Operator::Eq, ..
        })
        | Expr::InList(_)
        | Expr::Column(_)
        | Expr::Not(_) => key_set(conjunct, keys, max_seeks),
        Expr::BinaryExpr(BinaryExpr {
            op: Operator::Or, ..
        }) => key_set(conjunct, keys, max_seeks).or_else(ranges),
        _ => ranges(),
    }
}

fn key_ranges(expr: &Expr, keys: &Fields, max_seeks: usize) -> Option<(usize, Vec<KeyRange>)> {
    let (key, range) = match expr {
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: op @ (Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq),
            right,
        }) => {
            let ((key, field), op, literal) = match key_column(left, keys) {
                Some(column) => (column, *op, right),
                None => (key_column(right, keys)?, op.swap()?, left),
            };
            let value = bound(literal, field.data_type())?;
            let range = match op {
                Operator::Lt => (Bound::Unbounded, Bound::Excluded(value)),
                Operator::LtEq => (Bound::Unbounded, Bound::Included(value)),
                Operator::Gt => (Bound::Excluded(value), Bound::Unbounded),
                Operator::GtEq => (Bound::Included(value), Bound::Unbounded),
                _ => return None,
            };
            (key, range)
        }
        Expr::Between(Between {
            expr,
            negated: false,
            low,
            high,
        }) => {
            let (key, field) = key_column(expr, keys)?;
            let low = bound(low, field.data_type())?;
            let high = bound(high, field.data_type())?;
            (key, (Bound::Included(low), Bound::Included(high)))
        }
        Expr::IsNotNull(expr) => (
            key_column(expr, keys)?.0,
            (Bound::Unbounded, Bound::Unbounded),
        ),
        Expr::BinaryExpr(BinaryExpr {
            op: op @ (Operator::And | Operator::Or),
            ..
        }) => {
            let is_or = *op == Operator::Or;
            let terms = split_binary(expr, *op);
            if is_or && terms.len() > max_seeks {
                return None;
            }
            let mut key = None;
            let mut merged = Vec::with_capacity(terms.len());
            for term in terms {
                let (term_key, term_ranges) = key_ranges(term, keys, max_seeks)?;
                if *key.get_or_insert(term_key) != term_key {
                    return None;
                }
                merged.push(term_ranges);
            }
            while merged.len() > 1 {
                let mut pairs = std::mem::take(&mut merged).into_iter();
                while let Some(a) = pairs.next() {
                    merged.push(match pairs.next() {
                        Some(b) if is_or => union(&a, &b)?,
                        Some(b) => intersect(&a, &b)?,
                        None => a,
                    });
                }
            }
            return Some((key?, merged.pop()?));
        }
        _ => return None,
    };
    Some((key, union_of(range)?))
}

fn union_of(range: KeyRange) -> Option<Vec<KeyRange>> {
    let is_empty = Cut::lower(&range.0)
        .partial_cmp(&Cut::upper(&range.1))?
        .is_ge();
    Some(if is_empty { Vec::new() } else { vec![range] })
}

fn intersect(a: &[KeyRange], b: &[KeyRange]) -> Option<Vec<KeyRange>> {
    let mut ranges = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let a_starts_last = Cut::lower(&a[i].0)
            .partial_cmp(&Cut::lower(&b[j].0))?
            .is_ge();
        let a_ends_first = Cut::upper(&a[i].1)
            .partial_cmp(&Cut::upper(&b[j].1))?
            .is_le();
        let lower = if a_starts_last { &a[i].0 } else { &b[j].0 };
        let upper = if a_ends_first { &a[i].1 } else { &b[j].1 };
        ranges.extend(union_of((lower.clone(), upper.clone()))?);
        if a_ends_first {
            i += 1;
        } else {
            j += 1;
        }
    }
    Some(ranges)
}

fn union(a: &[KeyRange], b: &[KeyRange]) -> Option<Vec<KeyRange>> {
    let mut ranges: Vec<KeyRange> = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let next = if j == b.len()
            || (i < a.len()
                && Cut::lower(&a[i].0)
                    .partial_cmp(&Cut::lower(&b[j].0))?
                    .is_le())
        {
            i += 1;
            &a[i - 1]
        } else {
            j += 1;
            &b[j - 1]
        };
        match ranges.last_mut() {
            Some(last)
                if Cut::lower(&next.0)
                    .partial_cmp(&Cut::upper(&last.1))?
                    .is_le() =>
            {
                if Cut::upper(&next.1)
                    .partial_cmp(&Cut::upper(&last.1))?
                    .is_gt()
                {
                    last.1 = next.1.clone();
                }
            }
            _ => ranges.push(next.clone()),
        }
    }
    Some(ranges)
}

fn key_column<'k>(expr: &Expr, keys: &'k Fields) -> Option<(usize, &'k FieldRef)> {
    match expr {
        Expr::Column(column) => keys.find(&column.name),
        _ => None,
    }
}

fn key_set(conjunct: &Expr, keys: &Fields, max_seeks: usize) -> Option<Constraint> {
    let boolean_key = |expr: &Expr| {
        key_column(expr, keys)
            .filter(|(_, field)| field.data_type() == &DataType::Boolean)
            .map(|(key, _)| key)
    };
    let mut key = None;
    let mut values = Vec::new();
    for term in split_binary(conjunct, Operator::Or) {
        let term_key = match term {
            Expr::Literal(value, _) if value.is_null() => continue,
            Expr::BinaryExpr(BinaryExpr {
                left,
                op: Operator::Eq,
                right,
            }) => {
                let (term_key, value) = match key_column(left, keys) {
                    Some((term_key, _)) => (term_key, right),
                    None => (key_column(right, keys)?.0, left),
                };
                values.push(literal(value)?);
                term_key
            }
            Expr::InList(InList {
                expr,
                list,
                negated: false,
            }) => {
                for item in list {
                    values.push(literal(item)?);
                }
                key_column(expr, keys)?.0
            }
            Expr::Column(_) => {
                values.push(ScalarValue::Boolean(Some(true)));
                boolean_key(term)?
            }
            Expr::Not(expr) => {
                values.push(ScalarValue::Boolean(Some(false)));
                boolean_key(expr)?
            }
            _ => return None,
        };
        if *key.get_or_insert(term_key) != term_key {
            return None;
        }
    }
    let key = key?;
    let values = key_values(values, keys[key].data_type(), max_seeks)?;
    Some(Constraint::Values { key, values })
}

fn key_values(
    items: impl IntoIterator<Item = ScalarValue>,
    data_type: &DataType,
    max_seeks: usize,
) -> Option<KeyValues> {
    let mut values = Vec::new();
    let mut seen = HashSet::new();
    for value in items {
        if value.is_null() {
            continue;
        }
        let value = coerce_lossless(value, data_type)?;
        if seen.insert(value.clone()) {
            if values.len() == max_seeks {
                return None;
            }
            values.push(value);
        }
    }
    Some(KeyValues::In(values))
}

fn bound(expr: &Expr, data_type: &DataType) -> Option<ScalarValue> {
    literal(expr)
        .filter(|value| !value.is_null())
        .and_then(|value| coerce_lossless(value, data_type))
}

fn literal(expr: &Expr) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(value, _) => Some(value.clone()),
        Expr::Cast(Cast { expr, field }) => match expr.as_ref() {
            Expr::Literal(value, _) => value.cast_to(field.data_type()).ok(),
            _ => None,
        },
        Expr::ScalarFunction(ScalarFunction { func, args }) if func.name() == "arrow_cast" => {
            match args.as_slice() {
                [Expr::Literal(value, _), Expr::Literal(data_type, _)] => {
                    let data_type = data_type.try_as_str()??.parse::<DataType>().ok()?;
                    value.cast_to(&data_type).ok()
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn coerce_lossless(value: ScalarValue, data_type: &DataType) -> Option<ScalarValue> {
    if value.data_type() == *data_type {
        return Some(value);
    }
    let coerced = safe_coerce_scalar(&value, data_type)?;
    let is_lossless =
        coerced.data_type() == *data_type && coerced.cast_to(&value.data_type()).ok()? == value;
    is_lossless.then_some(coerced)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, LazyLock};

    use arrow_array::cast::AsArray;
    use arrow_array::{
        BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
        TimestampMillisecondArray,
    };
    use arrow_schema::{Field, Schema, TimeUnit};
    use datafusion::optimizer::simplify_expressions::{ExprSimplifier, SimplifyContext};
    use datafusion::prelude::SessionContext;
    use datafusion_common::{Column, DFSchema};
    use datafusion_expr::utils::split_conjunction;
    use datafusion_expr::{col, lit};
    use itertools::{Itertools, iproduct};
    use lance_datafusion::planner::Planner;
    use rstest::rstest;

    use super::*;
    use crate::scalar::AnyQuery;

    const NEW_YEAR_2024_MS: i64 = 1_704_067_200_000;

    struct Fixture {
        keys: Fields,
        boolean_keys: Fields,
        schema: DFSchema,
        rows: RecordBatch,
        session: SessionContext,
    }

    static FIXTURE: LazyLock<Fixture> = LazyLock::new(|| {
        let schema = Arc::new(Schema::new(vec![
            Field::new("tenant", DataType::Int32, true),
            Field::new("ts", DataType::Int64, true),
            Field::new("at", DataType::Timestamp(TimeUnit::Millisecond, None), true),
            Field::new("status", DataType::Utf8, true),
            Field::new("flag", DataType::Boolean, true),
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float32, true),
        ]));
        let tenants = [None, Some(0), Some(1), Some(2), Some(3), Some(7)]
            .into_iter()
            .chain([63, 64, 8190, 8191, 8192].map(Some));
        let ts = [None, Some(0), Some(1), Some(5), Some(9), Some(10), Some(11)]
            .into_iter()
            .chain([19, 20, 21, 127, 128, 2730, 2731].map(Some));
        let at = [
            None,
            Some(NEW_YEAR_2024_MS),
            Some(NEW_YEAR_2024_MS + 1),
            Some(NEW_YEAR_2024_MS + 86_400_000),
        ];
        let status = [None, Some("a"), Some("b")];
        let flag = [None, Some(false), Some(true)];
        let x = [
            None,
            Some(-f64::NAN),
            Some(f64::NEG_INFINITY),
            Some(-1.5),
            Some(-0.0),
            Some(0.0),
            Some(1.0),
            Some(f64::INFINITY),
            Some(f64::NAN),
        ];
        let (tenants, ts, at, status, flag, x): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
            iproduct!(tenants, ts, at, status, flag, x).multiunzip();
        // Rust leaves the sign of a NaN cast to f32 unspecified.
        let y = x
            .iter()
            .map(|value| {
                value.map(|value| {
                    let sign = if value.is_sign_negative() { -1.0 } else { 1.0 };
                    (value as f32).copysign(sign)
                })
            })
            .collect::<Vec<_>>();
        let rows = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(tenants)),
                Arc::new(Int64Array::from(ts)),
                Arc::new(TimestampMillisecondArray::from(at)),
                Arc::new(StringArray::from(status)),
                Arc::new(BooleanArray::from(flag)),
                Arc::new(Float64Array::from(x)),
                Arc::new(Float32Array::from(y)),
            ],
        )
        .unwrap();
        Fixture {
            keys: schema.fields()[..3].into(),
            boolean_keys: vec![schema.field(4).clone(), schema.field(0).clone()].into(),
            schema: DFSchema::try_from(schema.as_ref().clone()).unwrap(),
            rows,
            session: SessionContext::new(),
        }
    });

    // Neither side gets another `Planner::optimize_expr` pass, whose signed-zero
    // rewrite would widen a planned `x = 0.0` to both zeros and hide a plan that
    // lost `-0.0`. DataFusion's simplifier has no such rewrite, and a raw
    // `arrow_cast` fails at evaluation until the simplifier turns it into a cast.
    fn selected_rows(predicate: Expr) -> Vec<bool> {
        let fixture = &*FIXTURE;
        let context = SimplifyContext::builder()
            .with_schema(Arc::new(fixture.schema.clone()))
            .build();
        let simplifier = ExprSimplifier::new(context);
        let predicate = simplifier
            .coerce(predicate, &fixture.schema)
            .and_then(|predicate| simplifier.simplify(predicate))
            .unwrap();
        let predicate = fixture
            .session
            .create_physical_expr(predicate, &fixture.schema)
            .unwrap();
        let selected = predicate
            .evaluate(&fixture.rows)
            .unwrap()
            .into_array(fixture.rows.num_rows())
            .unwrap();
        selected
            .as_boolean()
            .iter()
            .map(|is_selected| is_selected == Some(true))
            .collect()
    }

    struct Expected {
        query: String,
        key_residual: &'static [usize],
        row_residual: &'static [usize],
    }

    fn planned(
        query: impl Into<String>,
        key_residual: &'static [usize],
        row_residual: &'static [usize],
    ) -> Option<Expected> {
        Some(Expected {
            query: query.into(),
            key_residual,
            row_residual,
        })
    }

    fn check_plan(conjuncts: &[Expr], keys: &Fields, max_seeks: usize, expected: Option<Expected>) {
        let plan = plan(conjuncts, keys, max_seeks);
        let Some(expected) = expected else {
            assert_eq!(plan, None);
            return;
        };
        let plan = plan.expect("predicate should plan");

        let query = &plan.query;
        let revalidated = TupleQuery::try_new(
            query.keys().clone(),
            query.prefix().to_vec(),
            query.ranges().map(<[_]>::to_vec),
        )
        .unwrap();
        assert_eq!(&revalidated, query);

        let conjuncts_at =
            |positions: &[usize]| positions.iter().map(|&p| &conjuncts[p]).collect::<Vec<_>>();
        let consumed = (0..conjuncts.len())
            .filter(|p| !expected.key_residual.contains(p) && !expected.row_residual.contains(p))
            .collect::<Vec<_>>();
        assert_eq!(query.to_string(), expected.query);
        assert_eq!(plan.consumed, conjuncts_at(&consumed));
        assert_eq!(plan.key_residual, conjuncts_at(expected.key_residual));
        assert_eq!(plan.row_residual, conjuncts_at(expected.row_residual));

        let selected_by_conjunct = conjuncts
            .iter()
            .cloned()
            .map(selected_rows)
            .collect::<Vec<_>>();
        let and = |selected: Vec<bool>, conjunct: &Vec<bool>| {
            selected
                .iter()
                .zip(conjunct)
                .map(|(a, b)| *a && *b)
                .collect()
        };
        let planned = expected
            .key_residual
            .iter()
            .chain(expected.row_residual)
            .map(|&position| &selected_by_conjunct[position])
            .fold(selected_rows(query.to_expr(String::new())), and);
        let original = selected_by_conjunct
            .iter()
            .fold(vec![true; FIXTURE.rows.num_rows()], and);
        assert_eq!(planned, original);
    }

    fn in_list(key: &str, len: usize) -> String {
        format!("{key} IN ({})", (0..len).join(", "))
    }

    fn ranges(key: &str, len: usize, start: usize) -> String {
        let ranges = (0..len).map(|range| {
            let lower = start + 2 * range;
            format!("({key} > {lower} AND {key} < {})", lower + 2)
        });
        format!("({})", ranges.format(" OR "))
    }

    fn split_ranges(key: &str, len: usize) -> String {
        let terms = (0..len).map(|term| {
            let (a, b, c, d) = (4 * term, 4 * term + 1, 4 * term + 2, 4 * term + 3);
            format!("({key} BETWEEN {a} AND {d} AND ({key} < {b} OR {key} > {c}))")
        });
        format!("({})", terms.format(" OR "))
    }

    // A SQL filter reaches the planner after two optimizer passes, one when the
    // scanner parses it and one in `create_filter_plan`.
    fn conjuncts(predicate: &str, passes: usize) -> Vec<Expr> {
        let mut predicate = FIXTURE
            .session
            .state()
            .create_logical_expr(predicate, &FIXTURE.schema)
            .unwrap();
        for _ in 0..passes {
            predicate = Planner::new(FIXTURE.rows.schema())
                .optimize_expr(predicate)
                .unwrap();
        }
        split_conjunction(&predicate).into_iter().cloned().collect()
    }

    #[rstest]
    #[case::you_see(
        "tenant IN (1, 2) AND ts >= 10 AND ts < 20 AND status = 'a'",
        planned("TupleQuery[tenant IN (1, 2), ts >= 10 AND ts < 20]", &[], &[3])
    )]
    #[case::equality_on_one_key("tenant = 1", planned("TupleQuery[tenant = 1]", &[], &[]))]
    #[case::equality_on_two_keys(
        "tenant = 1 AND ts = 10",
        planned("TupleQuery[tenant = 1, ts = 10]", &[], &[])
    )]
    #[case::equality_on_three_keys(
        "tenant = 1 AND ts = 10 AND at = TIMESTAMP '2024-01-01 00:00:00'",
        planned("TupleQuery[tenant = 1, ts = 10, at = 1704067200000]", &[], &[])
    )]
    #[case::equality_out_of_key_order(
        "ts = 10 AND tenant = 1",
        planned("TupleQuery[tenant = 1, ts = 10]", &[], &[])
    )]
    #[case::in_on_two_keys(
        "tenant IN (1, 2) AND ts IN (10, 20)",
        planned("TupleQuery[tenant IN (1, 2), ts IN (10, 20)]", &[], &[])
    )]
    #[case::in_three_values(
        "tenant IN (1, 2, 3) AND ts = 10",
        planned("TupleQuery[tenant IN (1, 2, 3), ts = 10]", &[], &[])
    )]
    #[case::in_four_values(
        "tenant IN (1, 2, 3, 4)",
        planned("TupleQuery[tenant IN (1, 2, 3, 4)]", &[], &[])
    )]
    #[case::range_gt(
        "tenant = 1 AND ts > 10",
        planned("TupleQuery[tenant = 1, ts > 10]", &[], &[])
    )]
    #[case::range_gt_eq("tenant >= 2", planned("TupleQuery[tenant >= 2]", &[], &[]))]
    #[case::range_lt(
        "tenant = 1 AND ts < 10",
        planned("TupleQuery[tenant = 1, ts < 10]", &[], &[])
    )]
    #[case::range_lt_eq(
        "tenant = 1 AND ts <= 10",
        planned("TupleQuery[tenant = 1, ts <= 10]", &[], &[])
    )]
    #[case::range_on_first_key(
        "tenant > 1 AND tenant < 7",
        planned("TupleQuery[tenant > 1 AND tenant < 7]", &[], &[])
    )]
    #[case::empty_range_lower_above_upper(
        "tenant = 1 AND ts > 20 AND ts < 10",
        planned("TupleQuery[tenant = 1, FALSE]", &[], &[])
    )]
    #[case::empty_range_equal_bounds_both_excluded(
        "tenant = 1 AND ts > 5 AND ts < 5",
        planned("TupleQuery[tenant = 1, FALSE]", &[], &[])
    )]
    #[case::empty_range_equal_bounds_one_excluded(
        "tenant = 1 AND ts >= 5 AND ts < 5",
        planned("TupleQuery[tenant = 1, FALSE]", &[], &[])
    )]
    #[case::second_lower_bound(
        "tenant = 1 AND ts > 5 AND ts >= 10",
        planned("TupleQuery[tenant = 1, ts >= 10]", &[], &[])
    )]
    #[case::between_after_lower_bound(
        "tenant > 10 AND tenant BETWEEN 0 AND 100",
        planned("TupleQuery[tenant > 10 AND tenant <= 100]", &[], &[])
    )]
    #[case::or_of_ranges(
        "tenant < 1 OR tenant > 5",
        planned("TupleQuery[(tenant < 1 OR tenant > 5)]", &[], &[])
    )]
    #[case::or_of_ranges_after_equality(
        "tenant = 1 AND (ts < 1 OR ts > 5)",
        planned("TupleQuery[tenant = 1, (ts < 1 OR ts > 5)]", &[], &[])
    )]
    #[case::or_of_ranges_apart_at_excluded_value(
        "tenant = 1 AND (ts < 10 OR ts > 10)",
        planned("TupleQuery[tenant = 1, (ts < 10 OR ts > 10)]", &[], &[])
    )]
    #[case::or_of_overlapping_ranges(
        "tenant = 1 AND (ts < 10 OR ts BETWEEN 5 AND 20)",
        planned("TupleQuery[tenant = 1, ts <= 20]", &[], &[])
    )]
    #[case::or_of_ranges_touching_at_excluded_value(
        "tenant = 1 AND (ts < 10 OR ts >= 10)",
        planned("TupleQuery[tenant = 1, ts IS NOT NULL]", &[], &[])
    )]
    #[case::or_of_ranges_touching_at_included_value(
        "tenant = 1 AND (ts <= 10 OR ts > 10 AND ts < 20)",
        planned("TupleQuery[tenant = 1, ts < 20]", &[], &[])
    )]
    #[case::or_intersected_with_range(
        "tenant = 1 AND (ts < 1 OR ts > 5) AND ts < 20",
        planned("TupleQuery[tenant = 1, (ts < 1 OR (ts > 5 AND ts < 20))]", &[], &[])
    )]
    #[case::or_intersected_with_or(
        "tenant = 1 AND (ts < 5 OR ts > 10) AND (ts < 1 OR ts > 19)",
        planned("TupleQuery[tenant = 1, (ts < 1 OR ts > 19)]", &[], &[])
    )]
    #[case::or_intersected_to_empty(
        "tenant = 1 AND (ts < 1 OR ts > 20) AND ts BETWEEN 5 AND 10",
        planned("TupleQuery[tenant = 1, FALSE]", &[], &[])
    )]
    #[case::or_of_empty_ranges(
        "tenant = 1 AND (ts > 20 AND ts < 10 OR ts > 5 AND ts < 5)",
        planned("TupleQuery[tenant = 1, FALSE]", &[], &[])
    )]
    #[case::or_of_ranges_across_keys("tenant < 1 OR ts > 5", None)]
    #[case::or_of_range_and_is_null(
        "tenant = 1 AND (ts < 10 OR ts IS NULL)",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::or_of_range_and_expression(
        "tenant = 1 AND (ts < 10 OR ts % 7 = 0)",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::or_of_ranges_on_bound_prefix(
        "tenant = 1 AND (tenant < 0 OR tenant > 5) AND ts > 3",
        planned("TupleQuery[tenant = 1, ts > 3]", &[1], &[])
    )]
    #[case::is_null_in_prefix(
        "tenant IS NULL AND ts = 10",
        planned("TupleQuery[tenant IS NULL, ts = 10]", &[], &[])
    )]
    #[case::is_null_on_last_bound_key(
        "tenant = 1 AND ts IS NULL",
        planned("TupleQuery[tenant = 1, ts IS NULL]", &[], &[])
    )]
    #[case::is_not_null(
        "tenant = 1 AND ts IS NOT NULL",
        planned("TupleQuery[tenant = 1, ts IS NOT NULL]", &[], &[])
    )]
    #[case::is_not_null_with_bound(
        "tenant = 1 AND ts IS NOT NULL AND ts < 10",
        planned("TupleQuery[tenant = 1, ts < 10]", &[], &[])
    )]
    #[case::in_with_null("tenant IN (1, NULL)", planned("TupleQuery[tenant = 1]", &[], &[]))]
    #[case::duplicate_in_values(
        "tenant IN (2, 1, 2)",
        planned("TupleQuery[tenant IN (2, 1)]", &[], &[])
    )]
    #[case::smallest_set_binds(
        "tenant IN (1, 2, 3) AND tenant = 2",
        planned("TupleQuery[tenant = 2]", &[0], &[])
    )]
    #[case::contradiction(
        "tenant = 1 AND tenant = 2",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::contradiction_with_is_null(
        "tenant = 1 AND tenant IS NULL",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::expression_over_key(
        "tenant = 1 AND ts % 7 = 0",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::range_on_bound_key(
        "tenant = 1 AND tenant > 0",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::equality_after_range(
        "tenant > 1 AND ts = 10",
        planned("TupleQuery[tenant > 1]", &[1], &[])
    )]
    #[case::skipped_key(
        "tenant = 1 AND at = TIMESTAMP '2024-01-01 00:00:00'",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::or_on_bound_prefix(
        "tenant = 1 AND (ts = 10 OR ts = 20)",
        planned("TupleQuery[tenant = 1, ts IN (10, 20)]", &[], &[])
    )]
    #[case::or_mixed_with_range(
        "tenant = 1 AND (ts = 10 OR ts > 20)",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::or_mixed_with_is_null(
        "tenant = 1 AND (ts = 10 OR ts IS NULL)",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::not_on_bound_prefix(
        "tenant = 1 AND NOT ts = 10",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::non_key_column(
        "tenant = 1 AND status = 'a'",
        planned("TupleQuery[tenant = 1]", &[], &[1])
    )]
    #[case::non_key_column_inside_or(
        "tenant = 1 AND (ts = 10 OR status = 'a')",
        planned("TupleQuery[tenant = 1]", &[], &[1])
    )]
    #[case::timestamp_literal_coerced(
        "tenant = 1 AND ts = 10 AND at >= TIMESTAMP '2024-01-01 00:00:00'",
        planned("TupleQuery[tenant = 1, ts = 10, at >= 1704067200000]", &[], &[])
    )]
    #[case::timestamp_string_literal(
        "tenant = 1 AND ts = 10 AND at >= '2024-01-01'",
        planned("TupleQuery[tenant = 1, ts = 10, at >= 1704067200000]", &[], &[])
    )]
    #[case::literal_overflows_key("tenant = 3000000000", None)]
    #[case::float_literal_on_integer_key("tenant = 1.5", None)]
    #[case::or_across_keys("tenant = 1 OR ts = 2", None)]
    #[case::or_on_one_key(
        "tenant = 1 OR tenant = 2",
        planned("TupleQuery[tenant IN (1, 2)]", &[], &[])
    )]
    #[case::not_equality("NOT tenant = 1", None)]
    #[case::not_equal("tenant != 1", None)]
    #[case::not_in("tenant NOT IN (1, 2)", None)]
    #[case::not_between(
        "tenant NOT BETWEEN 1 AND 2",
        planned("TupleQuery[(tenant < 1 OR tenant > 2)]", &[], &[])
    )]
    #[case::equality_on_second_key_only("ts = 10", None)]
    #[case::equality_on_later_keys_only("ts = 10 AND at = TIMESTAMP '2024-01-01 00:00:00'", None)]
    #[case::arrow_cast_of_key("tenant = arrow_cast(ts, 'Int32')", None)]
    #[case::key_compared_to_key("tenant = ts", None)]
    #[case::non_key_only("status = 'a'", None)]
    fn plans_conjunctive_predicate(
        #[case] predicate: String,
        #[case] expected: Option<Expected>,
        #[values(1, 2)] passes: usize,
    ) {
        check_plan(
            &conjuncts(&predicate, passes),
            &FIXTURE.keys,
            DEFAULT_MAX_TUPLE_SEEKS,
            expected,
        );
    }

    #[rstest]
    #[case::tuples_below_budget(
        "ts",
        DEFAULT_MAX_TUPLE_SEEKS,
        in_list("tenant", 8191),
        planned(format!("TupleQuery[{}]", in_list("tenant", 8191)), &[], &[])
    )]
    #[case::tuples_at_budget(
        "ts",
        DEFAULT_MAX_TUPLE_SEEKS,
        format!("{} AND {}", in_list("tenant", 64), in_list("ts", 128)),
        planned(
            format!("TupleQuery[{}, {}]", in_list("tenant", 64), in_list("ts", 128)),
            &[],
            &[]
        )
    )]
    #[case::tuples_past_budget(
        "ts",
        DEFAULT_MAX_TUPLE_SEEKS,
        format!("{} AND {}", in_list("tenant", 3), in_list("ts", 2731)),
        planned(format!("TupleQuery[{}]", in_list("tenant", 3)), &[1], &[])
    )]
    #[case::values_past_budget_on_one_key(
        "ts",
        DEFAULT_MAX_TUPLE_SEEKS,
        in_list("tenant", 8193),
        None
    )]
    #[case::values_past_budget_beside_range(
        "ts",
        DEFAULT_MAX_TUPLE_SEEKS,
        format!("{} AND tenant > 5", in_list("tenant", 8193)),
        planned("TupleQuery[tenant > 5]", &[0], &[])
    )]
    #[case::tuples_times_float_ranges_at_budget(
        "x",
        DEFAULT_MAX_TUPLE_SEEKS,
        format!("{} AND (x >= -0.0 OR x < CAST('-inf' AS DOUBLE))", in_list("tenant", 4096)),
        planned(
            format!("TupleQuery[{}, (x < -inf OR x >= -0)]", in_list("tenant", 4096)),
            &[],
            &[]
        )
    )]
    #[case::tuples_times_float_ranges_past_budget(
        "x",
        DEFAULT_MAX_TUPLE_SEEKS,
        format!("{} AND (x >= -0.0 OR x < CAST('-inf' AS DOUBLE))", in_list("tenant", 4097)),
        planned(format!("TupleQuery[{}]", in_list("tenant", 4097)), &[1], &[])
    )]
    #[case::tuples_times_ranges_at_budget(
        "ts",
        4,
        "tenant IN (1, 2) AND (ts < 1 OR ts > 5)".to_string(),
        planned("TupleQuery[tenant IN (1, 2), (ts < 1 OR ts > 5)]", &[], &[])
    )]
    #[case::tuples_times_ranges_past_budget(
        "ts",
        3,
        "tenant IN (1, 2) AND (ts < 1 OR ts > 5)".to_string(),
        planned("TupleQuery[tenant IN (1, 2)]", &[1], &[])
    )]
    #[case::ranges_at_budget(
        "ts",
        64,
        ranges("tenant", 64, 0),
        planned(format!("TupleQuery[{}]", ranges("tenant", 64, 0)), &[], &[])
    )]
    #[case::ranges_from_fewer_terms_at_budget(
        "ts",
        66,
        split_ranges("tenant", 33),
        planned(
            format!(
                "TupleQuery[({})]",
                (0..33)
                    .map(|v| (4 * v, 4 * v + 1, 4 * v + 2, 4 * v + 3))
                    .map(|(a, b, c, d)| format!(
                        "(tenant >= {a} AND tenant < {b}) OR (tenant > {c} AND tenant <= {d})"
                    ))
                    .join(" OR ")
            ),
            &[],
            &[]
        )
    )]
    #[case::ranges_from_fewer_terms_past_budget("ts", 65, split_ranges("tenant", 33), None)]
    #[case::ranges_past_budget_after_intersection(
        "ts",
        64,
        format!("tenant = 1 AND {} AND {}", ranges("ts", 64, 0), ranges("ts", 64, 1)),
        planned(format!("TupleQuery[tenant = 1, {}]", ranges("ts", 64, 0)), &[2], &[])
    )]
    #[case::or_terms_at_budget_merged_into_one_range(
        "ts",
        64,
        format!("tenant = 1 AND ({})", (1..=64).map(|v| format!("ts < {v}")).join(" OR ")),
        planned("TupleQuery[tenant = 1, ts < 64]", &[], &[])
    )]
    #[case::or_terms_past_budget_merged_into_one_range(
        "ts",
        64,
        format!("tenant = 1 AND ({})", (1..=65).map(|v| format!("ts < {v}")).join(" OR ")),
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::budget_of_one_binds_one_range(
        "ts",
        1,
        "tenant = 1 AND ts > 5".to_string(),
        planned("TupleQuery[tenant = 1, ts > 5]", &[], &[])
    )]
    #[case::budget_of_one_leaves_two_ranges(
        "ts",
        1,
        "tenant = 1 AND (ts < 1 OR ts > 5)".to_string(),
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::budget_of_one_leaves_two_values("ts", 1, "tenant IN (1, 2)".to_string(), None)]
    fn plans_within_seek_budget(
        #[case] second_key: &str,
        #[case] max_seeks: usize,
        #[case] predicate: String,
        #[case] expected: Option<Expected>,
        #[values(1, 2)] passes: usize,
    ) {
        let schema = FIXTURE.rows.schema();
        let keys = Fields::from(vec![
            schema.field_with_name("tenant").unwrap().clone(),
            schema.field_with_name(second_key).unwrap().clone(),
        ]);
        check_plan(&conjuncts(&predicate, passes), &keys, max_seeks, expected);
    }

    #[rstest]
    #[case::equals_true(
        "flag = true AND tenant = 1",
        planned("TupleQuery[flag = true, tenant = 1]", &[], &[])
    )]
    #[case::equals_false(
        "flag = false AND tenant = 1",
        planned("TupleQuery[flag = false, tenant = 1]", &[], &[])
    )]
    #[case::or_of_both_values(
        "(flag = true OR flag = false) AND tenant = 1",
        planned("TupleQuery[flag IN (true, false), tenant = 1]", &[], &[])
    )]
    #[case::or_across_keys("flag OR tenant = 1", None)]
    fn plans_predicate_on_boolean_key(
        #[case] predicate: &str,
        #[case] expected: Option<Expected>,
        #[values(1, 2)] passes: usize,
    ) {
        check_plan(
            &conjuncts(predicate, passes),
            &FIXTURE.boolean_keys,
            DEFAULT_MAX_TUPLE_SEEKS,
            expected,
        );
    }

    #[rstest]
    #[case::equals_zero("x", "x = 0.0", planned("TupleQuery[x IN (-0, 0)]", &[], &[]))]
    #[case::equals_negative_zero("x", "x = -0.0", planned("TupleQuery[x IN (-0, 0)]", &[], &[]))]
    #[case::at_least_zero(
        "x",
        "x >= 0.0",
        planned("TupleQuery[(x < -inf OR x >= -0)]", &[], &[])
    )]
    #[case::below_zero(
        "x",
        "x < 0.0",
        planned("TupleQuery[x >= -inf AND x < -0]", &[], &[])
    )]
    #[case::above_negative_infinity(
        "x",
        "x > CAST('-inf' AS DOUBLE)",
        planned("TupleQuery[(x < -inf OR x > -inf)]", &[], &[])
    )]
    #[case::in_zero_and_one(
        "x",
        "x IN (0.0, 1.0)",
        planned("TupleQuery[x IN (-0, 0, 1)]", &[], &[])
    )]
    #[case::zero_before_tenant(
        "x",
        "x = 0.0 AND tenant = 1",
        planned("TupleQuery[x IN (-0, 0), tenant = 1]", &[], &[])
    )]
    #[case::at_least_zero_or_negative_nan(
        "x",
        "x >= -0.0 OR x < CAST('-inf' AS DOUBLE)",
        planned("TupleQuery[(x < -inf OR x >= -0)]", &[], &[])
    )]
    #[case::above_negative_infinity_or_negative_nan(
        "x",
        "x > CAST('-inf' AS DOUBLE) OR x < CAST('-inf' AS DOUBLE)",
        planned("TupleQuery[(x < -inf OR x > -inf)]", &[], &[])
    )]
    // DataFusion's SQL parser types a bare `0.0` as DOUBLE and casts `y` to match.
    #[case::float32_equals_zero(
        "y",
        "y = CAST(0.0 AS REAL)",
        planned("TupleQuery[y IN (-0, 0)]", &[], &[])
    )]
    #[case::float32_at_least_zero_or_negative_nan(
        "y",
        "y >= CAST(-0.0 AS REAL) OR y < CAST('-inf' AS REAL)",
        planned("TupleQuery[(y < -inf OR y >= -0)]", &[], &[])
    )]
    fn plans_predicate_on_float_key(
        #[case] float_key: &str,
        #[case] predicate: &str,
        #[case] expected: Option<Expected>,
        #[values(1, 2)] passes: usize,
    ) {
        let schema = FIXTURE.rows.schema();
        let keys = Fields::from(vec![
            schema.field_with_name(float_key).unwrap().clone(),
            schema.field_with_name("tenant").unwrap().clone(),
        ]);
        check_plan(
            &conjuncts(predicate, passes),
            &keys,
            DEFAULT_MAX_TUPLE_SEEKS,
            expected,
        );
    }

    #[rstest]
    #[case::literal_on_left("1 = tenant", planned("TupleQuery[tenant = 1]", &[], &[]))]
    #[case::arrow_cast_on_first_key(
        "tenant = arrow_cast(1, 'Int32')",
        planned("TupleQuery[tenant = 1]", &[], &[])
    )]
    #[case::range_literal_on_left(
        "tenant = 1 AND 10 < ts",
        planned("TupleQuery[tenant = 1, ts > 10]", &[], &[])
    )]
    #[case::range_between(
        "tenant = 1 AND ts BETWEEN 5 AND 10",
        planned("TupleQuery[tenant = 1, ts >= 5 AND ts <= 10]", &[], &[])
    )]
    #[case::between_after_upper_bound(
        "tenant = 1 AND ts < 20 AND ts BETWEEN 5 AND 10",
        planned("TupleQuery[tenant = 1, ts >= 5 AND ts <= 10]", &[], &[])
    )]
    #[case::or_of_between_and_range(
        "tenant = 1 AND (ts BETWEEN 1 AND 3 OR ts > 10)",
        planned("TupleQuery[tenant = 1, ((ts >= 1 AND ts <= 3) OR ts > 10)]", &[], &[])
    )]
    #[case::equals_null("tenant = NULL", planned("TupleQuery[FALSE]", &[], &[]))]
    #[case::null_range_bound(
        "tenant = 1 AND ts > NULL",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::constant_conjunct(
        "tenant = 1 AND FALSE",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::timestamp_literal_lossy(
        "tenant = 1 AND ts = 10 AND at = TIMESTAMP '2024-01-01 00:00:00.000000100'",
        planned("TupleQuery[tenant = 1, ts = 10]", &[2], &[])
    )]
    #[case::timestamp_string_literal(
        "tenant = 1 AND ts = 10 AND at >= '2024-01-01'",
        planned("TupleQuery[tenant = 1, ts = 10]", &[2], &[])
    )]
    #[case::arrow_cast_string_to_timestamp(
        "tenant = 1 AND ts = 10 AND at >= arrow_cast('2024-01-01T00:00:00', 'Timestamp(Millisecond, None)')",
        planned("TupleQuery[tenant = 1, ts = 10, at >= 1704067200000]", &[], &[])
    )]
    #[case::cast_key("CAST(tenant AS BIGINT) = 1", None)]
    #[case::in_only_null("tenant IN (NULL)", planned("TupleQuery[FALSE]", &[], &[]))]
    #[case::not_equality("NOT tenant = 1", None)]
    #[case::not_between("tenant NOT BETWEEN 1 AND 2", None)]
    #[case::or_with_null_literal(
        "tenant = 1 OR NULL",
        planned("TupleQuery[tenant = 1]", &[], &[])
    )]
    #[case::or_with_true("tenant = 1 OR TRUE", None)]
    #[case::or_with_lossy_value("tenant = 1 OR tenant = 1.5", None)]
    #[case::or_with_in_term(
        "tenant IN (1, 2, 3, 4) OR tenant = 5",
        planned("TupleQuery[tenant IN (1, 2, 3, 4, 5)]", &[], &[])
    )]
    #[case::or_with_not_in_term("tenant NOT IN (1, 2, 3, 4) OR tenant = 5", None)]
    #[case::values_at_budget_after_empty_set(
        format!("tenant IN (NULL) AND {}", in_list("ts", 8192)),
        planned(format!("TupleQuery[FALSE, {}]", in_list("ts", 8192)), &[], &[])
    )]
    #[case::values_past_budget_after_empty_set(
        format!("tenant IN (NULL) AND {}", in_list("ts", 8193)),
        planned("TupleQuery[FALSE]", &[1], &[])
    )]
    fn plans_unoptimized_predicate(#[case] predicate: String, #[case] expected: Option<Expected>) {
        check_plan(
            &conjuncts(&predicate, 0),
            &FIXTURE.keys,
            DEFAULT_MAX_TUPLE_SEEKS,
            expected,
        );
    }

    // Parsing and optimizing 100,000 literals exceeds the test time budget, so
    // these lists are built with each literal already of the key's type, as
    // `optimize_expr` would leave it.
    #[rstest]
    #[case::items_of_10_values(
        vec![col("tenant").in_list((0..100_000).map(|item: i32| lit(item % 10)).collect(), false)],
        planned(format!("TupleQuery[{}]", in_list("tenant", 10)), &[], &[])
    )]
    #[case::distinct_values(
        vec![
            col("tenant").eq(lit(1)),
            col("ts").in_list((0..100_000_i64).map(lit).collect(), false),
        ],
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::distinct_values_after_empty_set(
        vec![
            col("tenant").in_list(vec![lit(ScalarValue::Null)], false),
            col("ts").in_list((0..100_000_i64).map(lit).collect(), false),
        ],
        planned("TupleQuery[FALSE]", &[1], &[])
    )]
    fn plans_100000_item_in_list(#[case] conjuncts: Vec<Expr>, #[case] expected: Option<Expected>) {
        check_plan(&conjuncts, &FIXTURE.keys, DEFAULT_MAX_TUPLE_SEEKS, expected);
    }

    #[rstest]
    #[case::unset(None, DEFAULT_MAX_TUPLE_SEEKS)]
    #[case::one(Some("1"), 1)]
    #[case::valid(Some("100"), 100)]
    fn reads_max_tuple_seeks(#[case] value: Option<&str>, #[case] expected: usize) {
        assert_eq!(parse_max_tuple_seeks(value.map(OsStr::new)), Ok(expected));
    }

    #[rstest]
    #[case::zero("0")]
    #[case::negative("-5")]
    #[case::non_numeric("many")]
    #[case::empty("")]
    fn warns_on_max_tuple_seeks_that_is_not_positive(#[case] value: &str) {
        let warning = parse_max_tuple_seeks(Some(OsStr::new(value))).unwrap_err();
        for part in [
            "LANCE_TUPLE_MAX_SEEKS".to_string(),
            format!("{value:?}"),
            DEFAULT_MAX_TUPLE_SEEKS.to_string(),
        ] {
            assert!(warning.contains(&part), "{warning}");
        }
    }

    #[test]
    fn qualified_column_never_binds_a_key() {
        let other_tenant = || Expr::Column(Column::new(Some("l"), "tenant"));
        check_plan(
            &[other_tenant().eq(lit(1))],
            &FIXTURE.keys,
            DEFAULT_MAX_TUPLE_SEEKS,
            None,
        );

        let conjuncts = [col("tenant").eq(lit(1)), other_tenant().eq(lit(2))];
        let plan = plan_tuple_predicate(&conjuncts, &FIXTURE.keys).expect("tenant binds key 0");
        assert_eq!(plan.consumed, vec![&conjuncts[0]]);
        assert_eq!(plan.row_residual, vec![&conjuncts[1]]);
    }

    #[test]
    fn or_over_two_keys_of_one_type_never_binds() {
        let keys = Fields::from(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]);
        check_plan(
            &[col("a").lt(lit(1)).or(col("b").gt(lit(5)))],
            &keys,
            DEFAULT_MAX_TUPLE_SEEKS,
            None,
        );
    }

    #[test]
    fn literal_in_other_time_zone_stays_in_key_residual() {
        let new_year_utc =
            ScalarValue::TimestampMillisecond(Some(NEW_YEAR_2024_MS), Some("UTC".into()));
        let conjuncts = [
            col("tenant").eq(lit(1)),
            col("ts").eq(lit(10_i64)),
            col("at").eq(lit(new_year_utc)),
        ];
        check_plan(
            &conjuncts,
            &FIXTURE.keys,
            DEFAULT_MAX_TUPLE_SEEKS,
            planned("TupleQuery[tenant = 1, ts = 10]", &[2], &[]),
        );
    }
}
