// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Logical queries over an ordered list of key columns.
//!
//! A [`TupleQuery`] says which key tuples match, not how an index stores them.
//! A multi-column index lowers it to its own representation, such as encoded
//! bytes or typed comparisons. [`planner`] builds one from a conjunctive filter.
//!
//! ```
//! use std::ops::Bound;
//!
//! use arrow_schema::{DataType, Field, Fields};
//! use datafusion_common::ScalarValue;
//! use lance_index::scalar::tuple::{KeyValues, TupleQuery};
//!
//! # fn main() -> lance_core::Result<()> {
//! let keys = Fields::from(vec![
//!     Field::new("tenant", DataType::Int32, true),
//!     Field::new("ts", DataType::Int64, true),
//! ]);
//! let tenants = vec![ScalarValue::Int32(Some(1)), ScalarValue::Int32(Some(2))];
//! let ts = vec![
//!     (
//!         Bound::Included(ScalarValue::Int64(Some(10))),
//!         Bound::Excluded(ScalarValue::Int64(Some(20))),
//!     ),
//!     (Bound::Excluded(ScalarValue::Int64(Some(90))), Bound::Unbounded),
//! ];
//! let query = TupleQuery::try_new(keys, vec![KeyValues::In(tenants)], Some(ts))?;
//! assert_eq!(
//!     query.to_string(),
//!     "TupleQuery[tenant IN (1, 2), ((ts >= 10 AND ts < 20) OR ts > 90)]"
//! );
//! # Ok(())
//! # }
//! ```

use std::any::Any;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::ops::Bound;

use arrow_schema::{Field, Fields};
use datafusion_common::{Column, ScalarValue};
use datafusion_expr::{Expr, lit};
use lance_core::{Error, Result};

use super::AnyQuery;

pub mod planner;

/// The values one key of a [`TupleQuery`] prefix may take.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KeyValues {
    /// The key is NULL.
    Null,
    /// The key equals one of these distinct non-null values. An empty list
    /// matches no key.
    In(Vec<ScalarValue>),
}

/// Lower and upper bounds on one key. A NULL key is never within them.
///
/// Bounds compare in [`ScalarValue`]'s order, which orders floats by IEEE 754
/// total order, as [`f64::total_cmp`] and the B-tree index do. A NaN with its
/// sign bit set sorts below `-inf`, `-0.0` sorts below `0.0`, and a NaN without
/// its sign bit sorts above `inf`, so `x < -inf` selects only the negative NaNs.
pub type KeyRange = (Bound<ScalarValue>, Bound<ScalarValue>);

/// A search over the leading columns of an ordered key.
///
/// The query fixes keys `0..prefix.len()`, each to its [`KeyValues`], so it
/// covers the cross product of those sets. When `ranges` is set, the next key
/// is non-null and within one of the ranges, so a single unbounded range means
/// `IS NOT NULL` and an empty list matches no key tuple. Later keys are
/// unconstrained.
///
/// The ranges are in ascending order. In each, the lower bound sorts below the
/// upper bound, or both include the same value. Each range ends before the next
/// begins: its upper bound sorts below the next lower bound, or both exclude
/// the same value, as in `x < 5 OR x > 5`. Ranges that overlap or touch, such
/// as `x < 5` and `x >= 5`, are one range.
///
/// `Display` prints one SQL term per constrained key, with strings quoted as
/// `'it''s'` and binary values as `X'0AFF'`. A key whose set is empty prints as
/// `FALSE`, because no key tuple can match it, and so does an empty list of
/// ranges. Two or more ranges print as `(x < 1 OR (x > 5 AND x < 9))`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TupleQuery {
    keys: Fields,
    prefix: Vec<KeyValues>,
    ranges: Option<Vec<KeyRange>>,
}

impl TupleQuery {
    /// Build a query over `keys`, the index's key columns in key order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] when `prefix` and `ranges` constrain
    /// more keys than `keys` holds, when a value or bound is NULL or does not
    /// have its key's type, when a [`KeyValues::In`] repeats a value, or when a
    /// range is empty or the ranges are out of order, overlap, or touch, as
    /// [`TupleQuery`] describes.
    pub fn try_new(
        keys: Fields,
        prefix: Vec<KeyValues>,
        ranges: Option<Vec<KeyRange>>,
    ) -> Result<Self> {
        let constrained = prefix.len() + usize::from(ranges.is_some());
        if constrained > keys.len() {
            return Err(Error::invalid_input(format!(
                "TupleQuery constrains {constrained} keys ({} in the prefix, ranges: {}) but the index has {} keys",
                prefix.len(),
                ranges.is_some(),
                keys.len()
            )));
        }
        for (values, key) in prefix.iter().zip(keys.iter()) {
            if let KeyValues::In(values) = values {
                let mut seen = HashSet::with_capacity(values.len());
                for value in values {
                    check_value(value, key)?;
                    if !seen.insert(value) {
                        return Err(Error::invalid_input(format!(
                            "TupleQuery repeats value {value:?} for key '{}'",
                            key.name()
                        )));
                    }
                }
            }
        }
        if let Some(ranges) = &ranges {
            let key = &keys[prefix.len()];
            for (lower, upper) in ranges {
                for bound in [lower, upper] {
                    if let Bound::Included(value) | Bound::Excluded(value) = bound {
                        check_value(value, key)?;
                    }
                }
            }
            let cuts = ranges
                .iter()
                .flat_map(|(lower, upper)| [Cut::lower(lower), Cut::upper(upper)])
                .collect::<Vec<_>>();
            if cuts
                .windows(2)
                .any(|pair| pair[0].partial_cmp(&pair[1]) != Some(Ordering::Less))
            {
                return Err(Error::invalid_input(format!(
                    "TupleQuery ranges for key '{}' must be non-empty, ascending, and apart, but are {ranges:?}",
                    key.name()
                )));
            }
        }
        Ok(Self {
            keys,
            prefix,
            ranges,
        })
    }

    /// The index's key columns, in key order.
    pub fn keys(&self) -> &Fields {
        &self.keys
    }

    /// The values each leading key may take.
    pub fn prefix(&self) -> &[KeyValues] {
        &self.prefix
    }

    /// The ranges the key after the prefix lies in, if it is constrained.
    pub fn ranges(&self) -> Option<&[KeyRange]> {
        self.ranges.as_deref()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Cut<'a> {
    Start,
    JustBelow(&'a ScalarValue),
    JustAbove(&'a ScalarValue),
    End,
}

impl<'a> Cut<'a> {
    fn lower(bound: &'a Bound<ScalarValue>) -> Self {
        match bound {
            Bound::Included(value) => Self::JustBelow(value),
            Bound::Excluded(value) => Self::JustAbove(value),
            Bound::Unbounded => Self::Start,
        }
    }

    fn upper(bound: &'a Bound<ScalarValue>) -> Self {
        match bound {
            Bound::Included(value) => Self::JustAbove(value),
            Bound::Excluded(value) => Self::JustBelow(value),
            Bound::Unbounded => Self::End,
        }
    }
}

impl PartialOrd for Cut<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Start, Self::Start) | (Self::End, Self::End) => Some(Ordering::Equal),
            (Self::Start, _) | (_, Self::End) => Some(Ordering::Less),
            (_, Self::Start) | (Self::End, _) => Some(Ordering::Greater),
            (Self::JustBelow(a) | Self::JustAbove(a), Self::JustBelow(b) | Self::JustAbove(b)) => {
                let is_above = |cut: &Self| matches!(cut, Self::JustAbove(_));
                Some(a.partial_cmp(b)?.then(is_above(self).cmp(&is_above(other))))
            }
        }
    }
}

fn check_value(value: &ScalarValue, key: &Field) -> Result<()> {
    if value.is_null() || value.data_type() != *key.data_type() {
        return Err(Error::invalid_input(format!(
            "TupleQuery value {value:?} for key '{}' must be a non-null {}",
            key.name(),
            key.data_type()
        )));
    }
    Ok(())
}

impl fmt::Display for TupleQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms = Vec::with_capacity(self.prefix.len() + 1);
        for (values, key) in self.prefix.iter().zip(self.keys.iter()) {
            let name = key.name();
            terms.push(match values {
                KeyValues::Null => format!("{name} IS NULL"),
                KeyValues::In(values) if values.is_empty() => "FALSE".to_string(),
                KeyValues::In(values) if values.len() == 1 => {
                    format!("{name} = {}", sql_literal(&values[0]))
                }
                KeyValues::In(values) => {
                    let values = values.iter().map(sql_literal).collect::<Vec<_>>();
                    format!("{name} IN ({})", values.join(", "))
                }
            });
        }
        if let Some(ranges) = &self.ranges {
            let name = self.keys[self.prefix.len()].name();
            terms.push(match ranges.as_slice() {
                [] => "FALSE".to_string(),
                [range] => range_sql(name, range),
                ranges => {
                    let ranges = ranges
                        .iter()
                        .map(|range| match range {
                            (Bound::Unbounded, _) | (_, Bound::Unbounded) => range_sql(name, range),
                            _ => format!("({})", range_sql(name, range)),
                        })
                        .collect::<Vec<_>>();
                    format!("({})", ranges.join(" OR "))
                }
            });
        }
        write!(f, "TupleQuery[{}]", terms.join(", "))
    }
}

fn range_sql(name: &str, (lower, upper): &KeyRange) -> String {
    let lower = match lower {
        Bound::Included(value) => Some(format!("{name} >= {}", sql_literal(value))),
        Bound::Excluded(value) => Some(format!("{name} > {}", sql_literal(value))),
        Bound::Unbounded => None,
    };
    let upper = match upper {
        Bound::Included(value) => Some(format!("{name} <= {}", sql_literal(value))),
        Bound::Excluded(value) => Some(format!("{name} < {}", sql_literal(value))),
        Bound::Unbounded => None,
    };
    match (lower, upper) {
        (Some(lower), Some(upper)) => format!("{lower} AND {upper}"),
        (Some(bound), None) | (None, Some(bound)) => bound,
        (None, None) => format!("{name} IS NOT NULL"),
    }
}

fn sql_literal(value: &ScalarValue) -> String {
    match value {
        ScalarValue::Utf8(Some(text))
        | ScalarValue::LargeUtf8(Some(text))
        | ScalarValue::Utf8View(Some(text)) => format!("'{}'", text.replace('\'', "''")),
        ScalarValue::Binary(Some(bytes))
        | ScalarValue::LargeBinary(Some(bytes))
        | ScalarValue::BinaryView(Some(bytes))
        | ScalarValue::FixedSizeBinary(_, Some(bytes)) => {
            format!("X'{}'", hex::encode_upper(bytes))
        }
        ScalarValue::Dictionary(_, value) => sql_literal(value),
        _ => value.to_string(),
    }
}

impl AnyQuery for TupleQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, _col: &str) -> String {
        self.to_string()
    }

    fn to_expr(&self, _col: String) -> Expr {
        let column = |key: &Field| Expr::Column(Column::new_unqualified(key.name()));
        let mut terms = Vec::with_capacity(self.prefix.len() + 2);
        for (values, key) in self.prefix.iter().zip(self.keys.iter()) {
            terms.push(match values {
                KeyValues::Null => column(key).is_null(),
                KeyValues::In(values) if values.is_empty() => lit(false),
                KeyValues::In(values) if values.len() == 1 => {
                    column(key).eq(lit(values[0].clone()))
                }
                KeyValues::In(values) => {
                    column(key).in_list(values.iter().cloned().map(lit).collect(), false)
                }
            });
        }
        if let Some(ranges) = &self.ranges {
            let key = column(&self.keys[self.prefix.len()]);
            let range_terms = |(lower, upper): &KeyRange| {
                let mut terms = Vec::with_capacity(2);
                match lower {
                    Bound::Included(value) => terms.push(key.clone().gt_eq(lit(value.clone()))),
                    Bound::Excluded(value) => terms.push(key.clone().gt(lit(value.clone()))),
                    Bound::Unbounded => {}
                }
                match upper {
                    Bound::Included(value) => terms.push(key.clone().lt_eq(lit(value.clone()))),
                    Bound::Excluded(value) => terms.push(key.clone().lt(lit(value.clone()))),
                    Bound::Unbounded => {}
                }
                if matches!((lower, upper), (Bound::Unbounded, Bound::Unbounded)) {
                    terms.push(key.clone().is_not_null());
                }
                terms
            };
            match ranges.as_slice() {
                [range] => terms.extend(range_terms(range)),
                ranges => terms.push(
                    ranges
                        .iter()
                        .filter_map(|range| range_terms(range).into_iter().reduce(Expr::and))
                        .reduce(Expr::or)
                        .unwrap_or(lit(false)),
                ),
            }
        }
        terms.into_iter().reduce(Expr::and).unwrap_or(lit(true))
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use rstest::rstest;

    use super::*;
    use crate::scalar::SargableQuery;

    fn keys() -> Fields {
        Fields::from(vec![
            Field::new("tenant", DataType::Int32, true),
            Field::new("ts", DataType::Int64, true),
        ])
    }

    fn ts(value: i64) -> ScalarValue {
        ScalarValue::Int64(Some(value))
    }

    #[rstest]
    #[case::prefix_longer_than_keys(vec![KeyValues::Null; 3], None)]
    #[case::range_after_last_key(
        vec![KeyValues::Null; 2],
        Some(vec![(Bound::Unbounded, Bound::Unbounded)])
    )]
    #[case::value_of_other_type(vec![KeyValues::In(vec![ScalarValue::Int64(Some(1))])], None)]
    #[case::null_value(vec![KeyValues::In(vec![ScalarValue::Int32(None)])], None)]
    #[case::null_bound(
        vec![KeyValues::Null],
        Some(vec![(Bound::Included(ScalarValue::Int64(None)), Bound::Unbounded)])
    )]
    #[case::bound_of_other_type(
        vec![KeyValues::Null],
        Some(vec![(Bound::Unbounded, Bound::Excluded(ScalarValue::Int32(Some(1))))])
    )]
    #[case::empty_range_lower_above_upper(
        vec![KeyValues::Null],
        Some(vec![(Bound::Excluded(ts(20)), Bound::Excluded(ts(10)))])
    )]
    #[case::empty_range_equal_bounds_one_excluded(
        vec![KeyValues::Null],
        Some(vec![(Bound::Included(ts(5)), Bound::Excluded(ts(5)))])
    )]
    #[case::ranges_out_of_order(
        vec![KeyValues::Null],
        Some(vec![(Bound::Excluded(ts(9)), Bound::Unbounded), (Bound::Unbounded, Bound::Excluded(ts(1)))])
    )]
    #[case::overlapping_ranges(
        vec![KeyValues::Null],
        Some(vec![(Bound::Unbounded, Bound::Excluded(ts(5))), (Bound::Excluded(ts(3)), Bound::Unbounded)])
    )]
    #[case::touching_ranges(
        vec![KeyValues::Null],
        Some(vec![(Bound::Unbounded, Bound::Excluded(ts(5))), (Bound::Included(ts(5)), Bound::Unbounded)])
    )]
    #[case::ranges_sharing_an_included_value(
        vec![KeyValues::Null],
        Some(vec![(Bound::Unbounded, Bound::Included(ts(5))), (Bound::Excluded(ts(5)), Bound::Unbounded)])
    )]
    #[case::range_after_unbounded_range(
        vec![KeyValues::Null],
        Some(vec![(Bound::Unbounded, Bound::Unbounded), (Bound::Excluded(ts(5)), Bound::Unbounded)])
    )]
    fn try_new_rejects_invalid_query(
        #[case] prefix: Vec<KeyValues>,
        #[case] ranges: Option<Vec<KeyRange>>,
    ) {
        let error = TupleQuery::try_new(keys(), prefix, ranges).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }

    #[test]
    fn try_new_names_a_repeated_value_and_its_key() {
        let values = [1, 2, 1].map(|v| ScalarValue::Int32(Some(v))).to_vec();
        let error = TupleQuery::try_new(keys(), vec![KeyValues::In(values)], None).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        let message = error.to_string();
        assert!(
            message.contains("'tenant'") && message.contains("Int32(1)"),
            "{message}"
        );
    }

    #[rstest]
    #[case::string_with_comma(
        vec![KeyValues::In(vec![ScalarValue::from("x, y"), ScalarValue::from("z")])],
        None,
        "TupleQuery[tag IN ('x, y', 'z')]"
    )]
    #[case::string_with_quote(
        vec![KeyValues::In(vec![ScalarValue::from("it's")])],
        None,
        "TupleQuery[tag = 'it''s']"
    )]
    #[case::binary_bound(
        vec![KeyValues::Null],
        Some(vec![(Bound::Included(ScalarValue::Binary(Some(vec![0x0a, 0xff]))), Bound::Unbounded)]),
        "TupleQuery[tag IS NULL, blob >= X'0AFF']"
    )]
    #[case::union_of_ranges(
        vec![KeyValues::Null],
        Some(vec![
            (Bound::Unbounded, Bound::Excluded(ScalarValue::Binary(Some(vec![0x0a])))),
            (
                Bound::Excluded(ScalarValue::Binary(Some(vec![0x0a]))),
                Bound::Included(ScalarValue::Binary(Some(vec![0xff]))),
            ),
        ]),
        "TupleQuery[tag IS NULL, (blob < X'0A' OR (blob > X'0A' AND blob <= X'FF'))]"
    )]
    #[case::empty_ranges(vec![KeyValues::Null], Some(vec![]), "TupleQuery[tag IS NULL, FALSE]")]
    #[case::dictionary_string(
        vec![
            KeyValues::Null,
            KeyValues::Null,
            KeyValues::In(vec![ScalarValue::Dictionary(
                Box::new(DataType::Int32),
                Box::new(ScalarValue::from("a")),
            )]),
        ],
        None,
        "TupleQuery[tag IS NULL, blob IS NULL, label = 'a']"
    )]
    #[case::empty_set(vec![KeyValues::In(vec![])], None, "TupleQuery[FALSE]")]
    #[case::empty_set_after_value(
        vec![KeyValues::In(vec![ScalarValue::from("a")]), KeyValues::In(vec![])],
        None,
        "TupleQuery[tag = 'a', FALSE]"
    )]
    fn display_quotes_values_and_marks_empty_sets(
        #[case] prefix: Vec<KeyValues>,
        #[case] ranges: Option<Vec<KeyRange>>,
        #[case] expected: &str,
    ) {
        let label_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let keys = Fields::from(vec![
            Field::new("tag", DataType::Utf8, true),
            Field::new("blob", DataType::Binary, true),
            Field::new("label", label_type, true),
        ]);
        let query = TupleQuery::try_new(keys, prefix, ranges).unwrap();
        assert_eq!(query.to_string(), expected);
    }

    #[test]
    fn any_query_compares_and_formats_by_value() {
        let query = TupleQuery::try_new(
            keys(),
            vec![KeyValues::Null],
            Some(vec![(Bound::Excluded(ts(5)), Bound::Unbounded)]),
        )
        .unwrap();
        let unbounded = TupleQuery::try_new(
            keys(),
            vec![KeyValues::Null],
            Some(vec![(Bound::Unbounded, Bound::Unbounded)]),
        )
        .unwrap();
        assert!(query.dyn_eq(&query.clone()));
        assert!(!query.dyn_eq(&unbounded));
        assert!(!query.dyn_eq(&SargableQuery::IsNull()));
        assert_eq!(query.format("unused"), "TupleQuery[tenant IS NULL, ts > 5]");
        assert_eq!(
            unbounded.format("unused"),
            "TupleQuery[tenant IS NULL, ts IS NOT NULL]"
        );
    }
}
