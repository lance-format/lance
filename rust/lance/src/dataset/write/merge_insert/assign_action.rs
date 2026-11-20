// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{MERGE_SOURCE_SENTINEL, MergeInsertParams, WhenNotMatchedBySource};
use crate::{Result, dataset::WhenMatched};
use datafusion::common::{
    Column, TableReference,
    tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion},
};
use datafusion::functions::core::expr_fn::get_field;
use datafusion::scalar::ScalarValue;
use datafusion_expr::{Case, Expr, col};
use lance_core::datatypes::Schema;
use std::sync::Arc;

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

/// The `UpdateIf` condition, parsed against the dataset schema with the
/// `source` / `target` qualifiers enabled.
pub fn parse_update_condition(
    params: &MergeInsertParams,
    schema: &arrow_schema::Schema,
) -> Result<Option<Expr>> {
    match &params.when_matched {
        WhenMatched::UpdateIf(condition_str) => {
            let planner = lance_datafusion::planner::Planner::new(Arc::new(schema.clone()))
                .with_enable_relations(true);
            let condition = planner.parse_filter(condition_str).map_err(|e| {
                crate::Error::invalid_input(format!("Failed to parse UpdateIf condition: {}", e))
            })?;
            Ok(Some(condition))
        }
        WhenMatched::UpdateIfExpr(condition) => Ok(Some(condition.clone())),
        _ => Ok(None),
    }
}

/// The `DeleteIf` condition with its unqualified columns bound to `target`.
pub fn parse_delete_condition(params: &MergeInsertParams) -> Result<Option<Expr>> {
    match &params.delete_not_matched_by_source {
        WhenNotMatchedBySource::DeleteIf(condition) => Ok(Some(qualify_unqualified_columns(
            condition.clone(),
            "target",
        )?)),
        _ => Ok(None),
    }
}

/// Whether a merge condition reads a target column. An unqualified name
/// counts: the join resolves it to the target when the source lacks it, and
/// rejects it as ambiguous when both carry it, as the take does.
pub fn condition_reads_target(condition: &Expr, dataset_schema: &Schema) -> bool {
    condition.column_refs().into_iter().any(|column| {
        column
            .relation
            .as_ref()
            .is_none_or(|relation| relation.table() == "target")
            && dataset_schema.field(&column.name).is_some()
    })
}

/// Rewrites the column references of a condition planned over a `source` and a
/// `target` struct column (the merger's form) to fields of those structs,
/// following the rule of [`condition_reads_target`]: an unqualified dataset
/// column is the target's when the source lacks it, and ambiguous when both
/// carry it. An unqualified name is matched to a dataset column as the
/// planner matches column names: exactly, or else ignoring ASCII case.
///
/// A relation names the struct only when its first part is `source` or
/// `target`, so `target.s.x` (relation `target.s`) reads `target.s.x` while
/// `foo.target.a` is left for planning to reject. `source` and `target`
/// themselves, and names that are not dataset columns, are left unchanged.
pub fn bind_struct_columns(
    condition: Expr,
    dataset_schema: &Schema,
    source: &arrow_schema::Schema,
) -> Result<Expr> {
    condition
        .transform(|expr| {
            let Expr::Column(column) = &expr else {
                return Ok(Transformed::no(expr));
            };
            let mut name = column.name.as_str();
            let path = match &column.relation {
                None if name == "source" || name == "target" => None,
                None => {
                    let fields = &dataset_schema.fields;
                    let Some(field) = fields
                        .iter()
                        .find(|f| f.name == name)
                        .or_else(|| fields.iter().find(|f| f.name.eq_ignore_ascii_case(name)))
                    else {
                        return Ok(Transformed::no(expr));
                    };
                    name = field.name.as_str();
                    if source.column_with_name(name).is_some() {
                        return Err(crate::Error::invalid_input(format!(
                            "Ambiguous reference to unqualified column '{name}' in the merge \
                             insert condition: both the source and the target have it. Write \
                             'source.{name}' or 'target.{name}' instead."
                        ))
                        .into());
                    }
                    Some(vec!["target"])
                }
                Some(TableReference::Bare { table }) => Some(vec![table.as_ref()]),
                Some(TableReference::Partial { schema, table }) => {
                    Some(vec![schema.as_ref(), table.as_ref()])
                }
                Some(TableReference::Full {
                    catalog,
                    schema,
                    table,
                }) => Some(vec![catalog.as_ref(), schema.as_ref(), table.as_ref()]),
            };
            let Some(path) = path.filter(|path| matches!(path[0], "source" | "target")) else {
                return Ok(Transformed::no(expr));
            };
            let root = Expr::Column(Column::new_unqualified(path[0]));
            let bound = path[1..]
                .iter()
                .copied()
                .chain([name])
                .fold(root, get_field);
            Ok(Transformed::yes(bound))
        })
        .data()
        .map_err(crate::Error::from)
}

/// The top-level `target` struct fields a condition bound by
/// [`bind_struct_columns`] reads, in order of first reference.
pub fn struct_target_columns(condition: &Expr) -> Result<Vec<String>> {
    let mut columns: Vec<String> = Vec::new();
    condition.apply(|expr| {
        if let Expr::ScalarFunction(function) = expr
            && function.func.name() == "get_field"
            && let [Expr::Column(root), Expr::Literal(name, _), ..] = function.args.as_slice()
            && root.relation.is_none()
            && root.name == "target"
            && let Some(Some(name)) = name.try_as_str()
            && !columns.iter().any(|column| column == name)
        {
            columns.push(name.to_string());
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(columns)
}

/// Transforms merge insert parameters into a logical expression. The output
/// is a single "action" column, that describes what to do with each row.
///
/// `update_condition` and `delete_condition` are the `UpdateIf` and `DeleteIf`
/// conditions to fold into the action. `None` for a configured condition
/// leaves every matched row an update, or every target-only row a delete,
/// for `MergeConditionExec` to gate after the join.
pub fn merge_insert_action(
    params: &MergeInsertParams,
    update_condition: Option<Expr>,
    delete_condition: Option<Expr>,
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
        WhenMatched::UpdateIf(_) | WhenMatched::UpdateIfExpr(_) => {
            let when = match update_condition {
                Some(condition) => matched.and(condition),
                None => matched,
            };
            cases.push((when, Action::UpdateAll.as_literal_expr()));
        }
        WhenMatched::DoNothing => {}
        WhenMatched::Fail => {
            cases.push((matched, Action::Fail.as_literal_expr()));
        }
        WhenMatched::Delete => {
            cases.push((matched, Action::Delete.as_literal_expr()));
        }
    }

    match &params.delete_not_matched_by_source {
        WhenNotMatchedBySource::Delete => {
            cases.push((target_only, Action::Delete.as_literal_expr()));
        }
        WhenNotMatchedBySource::DeleteIf(_) => {
            let when = match delete_condition {
                Some(condition) => target_only.and(condition),
                None => target_only,
            };
            cases.push((when, Action::Delete.as_literal_expr()));
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
