// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{MERGE_SOURCE_SENTINEL, MergeInsertParams, WhenNotMatchedBySource};
use crate::{Result, dataset::WhenMatched};
use datafusion::common::{
    Column, TableReference,
    tree_node::{Transformed, TransformedResult, TreeNode},
};
use datafusion::logical_expr::ScalarUDF;
use datafusion::scalar::ScalarValue;
use datafusion_expr::{Case, Expr, col, expr::ScalarFunction};
use datafusion_functions::core::getfield::GetFieldFunc;

// Note: right now, this is a fixed enum. In the future, this will need to be
// dynamic to support multiple merge insert update clauses like:
// ```sql
// MERGE my_table USING input ON table.id = input.id
// WHEN MATCHED AND input.event = "new_date" THEN UPDATE SET my_table.date = input.date
// WHEN MATCHED AND input.event = "new_name" THEN UPDATE SET my_table.name = input.new_name
// ```
// At that point we will have a variable number of actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Action {
    Nothing = 0,
    /// Update all columns with source values
    UpdateAll = 1,
    Insert = 2,
    Delete = 3,
    /// Fail the operation if a match is found
    Fail = 4,
    /// The row matched a target row but the conditional `when_matched` clause
    /// did not apply to it.
    ///
    /// Nothing is written for the row, but it still claims its target row for
    /// source-deduplication accounting, so a second source row matching the
    /// same target is detected regardless of the condition.
    MatchedNoOp = 5,
}

impl TryFrom<u8> for Action {
    type Error = crate::Error;

    fn try_from(value: u8) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Nothing),
            1 => Ok(Self::UpdateAll),
            2 => Ok(Self::Insert),
            3 => Ok(Self::Delete),
            4 => Ok(Self::Fail),
            5 => Ok(Self::MatchedNoOp),
            _ => Err(crate::Error::invalid_input(format!(
                "Invalid action code: {}",
                value
            ))),
        }
    }
}

impl Action {
    fn as_literal_expr(&self) -> Expr {
        Expr::Literal(ScalarValue::UInt8(Some(*self as u8)), None)
    }
}

/// Parses a raw SQL condition string for a fast-path `WhenMatched` arm.
///
/// The condition is parsed with a relation-enabled planner so it can reference columns
/// qualified with `source.` and `target.`, matching the join output schema used by the
/// fast path CASE expression. `variant_name` is used to name the condition in the parse
/// error message (e.g. "UpdateIf" or "DeleteIf"), and the failing condition string is
/// included so it can be identified when several conditions are present.
pub(super) fn parse_when_matched_condition(
    condition_str: &str,
    schema: Option<&arrow_schema::Schema>,
    variant_name: &str,
) -> Result<Expr> {
    let Some(dataset_schema) = schema else {
        return Err(crate::Error::internal(format!(
            "Schema required for {} parsing",
            variant_name
        )));
    };
    let planner =
        lance_datafusion::planner::Planner::new(std::sync::Arc::new(dataset_schema.clone()))
            .with_enable_relations(true);
    planner.parse_filter(condition_str).map_err(|e| {
        crate::Error::invalid_input(format!(
            "Failed to parse {} condition '{}': {}",
            variant_name, condition_str, e
        ))
    })
}

/// The two relations a `when_matched` condition may reference.
const MATCHED_RELATIONS: [&str; 2] = ["source", "target"];

/// Rewrites combined-schema struct-field access into relation-qualified columns.
///
/// A pre-built `UpdateIfExpr`/`DeleteIfExpr` may be written either way (see
/// [`crate::dataset::WhenMatched::delete_if_expr`]). The standard plan evaluates the
/// condition against a join whose sides are aliased `source` and `target`, so
/// `get_field(col("source"), "value")` is normalized to `col("source.value")` here and
/// both forms plan identically. Nested access such as `col("source").field("a").field("b")`
/// keeps its outer accessor, which then reads a struct column on the qualified side.
pub(super) fn qualify_combined_schema_fields(expr: Expr) -> Result<Expr> {
    expr.transform(|expr| {
        let Expr::ScalarFunction(function) = &expr else {
            return Ok(Transformed::no(expr));
        };
        if function.func.name() != "get_field" || function.args.len() != 2 {
            return Ok(Transformed::no(expr));
        }
        let (Expr::Column(relation), Expr::Literal(ScalarValue::Utf8(Some(field)), _)) =
            (&function.args[0], &function.args[1])
        else {
            return Ok(Transformed::no(expr));
        };
        if relation.relation.is_some() || !MATCHED_RELATIONS.contains(&relation.name.as_str()) {
            return Ok(Transformed::no(expr));
        }
        let qualified = Column::new_unqualified(field.clone())
            .with_relation(TableReference::bare(relation.name.clone()));
        Ok(Transformed::yes(Expr::Column(qualified)))
    })
    .data()
    .map_err(crate::Error::from)
}

/// Rewrites relation-qualified columns into combined-schema struct-field access.
///
/// This is the inverse of [`qualify_combined_schema_fields`], applied where the indexed
/// (`Merger`) path compiles the condition against the combined schema, so a pre-built
/// expression written as `col("source.value")` plans there too.
pub(super) fn combined_schema_fields_from_qualified(expr: Expr) -> Result<Expr> {
    let get_field = ScalarUDF::from(GetFieldFunc::default());
    expr.transform(|expr| {
        let Expr::Column(column) = &expr else {
            return Ok(Transformed::no(expr));
        };
        let Some(relation) = column.relation.as_ref() else {
            return Ok(Transformed::no(expr));
        };
        if !MATCHED_RELATIONS.contains(&relation.table()) {
            return Ok(Transformed::no(expr));
        }
        let accessor = Expr::ScalarFunction(ScalarFunction::new_udf(
            std::sync::Arc::new(get_field.clone()),
            vec![
                col(relation.table()),
                Expr::Literal(ScalarValue::Utf8(Some(column.name.clone())), None),
            ],
        ));
        Ok(Transformed::yes(accessor))
    })
    .data()
    .map_err(crate::Error::from)
}

fn qualify_unqualified_columns(expr: Expr, relation: &'static str) -> Result<Expr> {
    expr.transform(|expr| {
        Ok(if let Expr::Column(column) = expr {
            if column.relation.is_none() {
                let qualified = Column::new_unqualified(column.name)
                    .with_relation(TableReference::bare(relation));
                Transformed::yes(Expr::Column(qualified))
            } else {
                Transformed::no(Expr::Column(column))
            }
        } else {
            Transformed::no(expr)
        })
    })
    .data()
    .map_err(crate::Error::from)
}

/// Transforms merge insert parameters into a logical expression. The output
/// is a single "action" column, that describes what to do with each row.
pub fn merge_insert_action(
    params: &MergeInsertParams,
    schema: Option<&arrow_schema::Schema>,
) -> Result<Expr> {
    // Use a sentinel column to detect whether the source side contributed a row to the
    // join output.  This is NULL-safe: the sentinel is `true` for every source row and
    // is NULL-filled by the outer join for target-only rows, regardless of whether any
    // ON column contains NULL.  Using ON key columns for this purpose is incorrect
    // because a key column that is legitimately NULL is indistinguishable from a NULL
    // introduced by the outer join on the target side.
    let source_has_row = col(format!("source.\"{}\"", MERGE_SOURCE_SENTINEL)).is_not_null();

    let target_has_row = col("target._rowaddr").is_not_null();
    let matched = source_has_row.clone().and(target_has_row.clone());

    let source_only = source_has_row.and(col("target._rowaddr").is_null());

    let target_only =
        target_has_row.and(col(format!("source.\"{}\"", MERGE_SOURCE_SENTINEL)).is_null());

    let mut cases = vec![];

    if params.insert_not_matched {
        cases.push((source_only, Action::Insert.as_literal_expr()));
    }

    match &params.when_matched {
        WhenMatched::UpdateAll => {
            cases.push((matched, Action::UpdateAll.as_literal_expr()));
        }
        WhenMatched::UpdateIf(condition_str) => {
            let condition = parse_when_matched_condition(condition_str, schema, "UpdateIf")?;
            cases.push((
                matched.clone().and(condition),
                Action::UpdateAll.as_literal_expr(),
            ));
            cases.push((matched, Action::MatchedNoOp.as_literal_expr()));
        }
        WhenMatched::UpdateIfExpr(condition) => {
            let condition = qualify_combined_schema_fields(condition.clone())?;
            cases.push((
                matched.clone().and(condition),
                Action::UpdateAll.as_literal_expr(),
            ));
            cases.push((matched, Action::MatchedNoOp.as_literal_expr()));
        }
        WhenMatched::DoNothing => {}
        WhenMatched::Fail => {
            cases.push((matched, Action::Fail.as_literal_expr()));
        }
        WhenMatched::Delete => {
            cases.push((matched, Action::Delete.as_literal_expr()));
        }
        WhenMatched::DeleteIf(condition_str) => {
            let condition = parse_when_matched_condition(condition_str, schema, "DeleteIf")?;
            cases.push((
                matched.clone().and(condition),
                Action::Delete.as_literal_expr(),
            ));
            cases.push((matched, Action::MatchedNoOp.as_literal_expr()));
        }
        WhenMatched::DeleteIfExpr(condition) => {
            let condition = qualify_combined_schema_fields(condition.clone())?;
            cases.push((
                matched.clone().and(condition),
                Action::Delete.as_literal_expr(),
            ));
            cases.push((matched, Action::MatchedNoOp.as_literal_expr()));
        }
    }

    match &params.delete_not_matched_by_source {
        WhenNotMatchedBySource::Delete => {
            cases.push((target_only, Action::Delete.as_literal_expr()));
        }
        WhenNotMatchedBySource::DeleteIf(condition) => {
            let target_condition = qualify_unqualified_columns(condition.clone(), "target")?;
            cases.push((
                target_only.and(target_condition),
                Action::Delete.as_literal_expr(),
            ));
        }
        WhenNotMatchedBySource::Keep => {}
    }

    Ok(Expr::Case(Case {
        expr: None,
        when_then_expr: cases
            .into_iter()
            .map(|(when, then)| (Box::new(when), Box::new(then)))
            .collect(),
        else_expr: Some(Box::new(Action::Nothing.as_literal_expr())),
    }))
}
