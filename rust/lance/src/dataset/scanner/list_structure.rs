// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Prove that a scan's consumers only observe list structure before omitting values.

use std::collections::{BTreeMap, HashMap, HashSet};

use arrow_schema::DataType;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::functions_nested::length::ArrayLength;
use datafusion::logical_expr::{Expr, Operator};
use datafusion::scalar::ScalarValue;
use datafusion_functions::core::getfield::GetFieldFunc;
use lance_core::datatypes::Field;
use lance_encoding::decoder::ListReadMode;

use super::Scanner;
use crate::Result;

fn literal_is(expr: &Expr, expected: i64) -> bool {
    matches!(expr, Expr::Literal(ScalarValue::Int64(Some(value)), _) if *value == expected)
        || matches!(expr, Expr::Literal(ScalarValue::UInt64(Some(value)), _) if i64::try_from(*value) == Ok(expected))
}

fn column_path(expr: &Expr) -> Option<Vec<String>> {
    match expr {
        Expr::Column(column) => Some(vec![column.name.clone()]),
        Expr::ScalarFunction(function)
            if function.args.len() == 2
                && function
                    .func
                    .inner()
                    .downcast_ref::<GetFieldFunc>()
                    .is_some() =>
        {
            let mut path = column_path(&function.args[0])?;
            let Expr::Literal(ScalarValue::Utf8(Some(name)), _) = &function.args[1] else {
                return None;
            };
            path.push(name.clone());
            Some(path)
        }
        _ => None,
    }
}

fn length_column(expr: &Expr) -> Option<Vec<String>> {
    let Expr::ScalarFunction(function) = expr else {
        return None;
    };
    // A user-defined function with the same name may inspect the payload.
    function.func.inner().downcast_ref::<ArrayLength>()?;
    if !(1..=2).contains(&function.args.len()) {
        return None;
    }
    column_path(&function.args[0])
}

#[derive(Default)]
struct Demands {
    structure: HashMap<Vec<String>, ListReadMode>,
    values: HashSet<Vec<String>>,
}

impl Demands {
    fn add(&mut self, column: Vec<String>, mode: ListReadMode) {
        self.structure
            .entry(column)
            .and_modify(|current| {
                if mode == ListReadMode::Length {
                    *current = mode;
                }
            })
            .or_insert(mode);
    }

    fn visit(&mut self, expr: &Expr) -> Result<()> {
        expr.apply(|expr| {
            if let Expr::BinaryExpr(binary) = expr
                && binary.op == Operator::Gt
                && literal_is(&binary.right, 0)
                && let Some(column) = length_column(&binary.left)
                && matches!(binary.left.as_ref(), Expr::ScalarFunction(function) if function.args.len() == 1 || literal_is(&function.args[1], 1))
            {
                self.add(column, ListReadMode::NonEmpty);
                return Ok(TreeNodeRecursion::Jump);
            }
            if let Some(column) = length_column(expr) {
                self.add(column, ListReadMode::Length);
                if let Expr::ScalarFunction(function) = expr
                    && let Some(dimension) = function.args.get(1)
                {
                    self.visit(dimension).map_err(|error| datafusion::error::DataFusionError::External(Box::new(error)))?;
                }
                return Ok(TreeNodeRecursion::Jump);
            }
            if let Expr::IsNull(inner) | Expr::IsNotNull(inner) = expr
                && let Some(column) = column_path(inner)
            {
                self.add(column, ListReadMode::NonEmpty);
                return Ok(TreeNodeRecursion::Jump);
            }
            if let Some(column) = column_path(expr) {
                self.values.insert(column);
                return Ok(TreeNodeRecursion::Jump);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(())
    }
}

impl Scanner {
    pub(super) fn list_structure_modes(&self) -> Result<BTreeMap<i32, ListReadMode>> {
        if self.nearest.is_some() || self.full_text_query.is_some() || self.minhash_query.is_some()
        {
            return Ok(BTreeMap::new());
        }
        let mut demands = Demands::default();
        for output in &self.projection_plan.requested_output_expr {
            demands.visit(&output.expr)?;
        }
        if let Some(filter) = self.get_expr_filter()? {
            demands.visit(&filter)?;
        }
        if let Some(aggregate) = &self.aggregate {
            for expr in aggregate.group_by.iter().chain(&aggregate.aggregates) {
                demands.visit(expr)?;
            }
        }
        if let Some(ordering) = &self.ordering {
            demands.values.extend(ordering.iter().map(|order| {
                lance_core::datatypes::parse_field_path(&order.column_name)
                    .unwrap_or_else(|_| vec![order.column_name.clone()])
            }));
        }
        fn collect(
            field: &Field,
            path: &mut Vec<String>,
            demands: &Demands,
            modes: &mut BTreeMap<i32, ListReadMode>,
        ) {
            path.push(field.name.clone());
            if !demands.values.iter().any(|value| path.starts_with(value)) && !field.is_blob() {
                if let Some(mode) = demands.structure.get(path)
                    && !demands.values.iter().any(|value| value.starts_with(path))
                    && matches!(
                        field.data_type(),
                        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
                    )
                {
                    modes.insert(field.id, *mode);
                }
                for child in &field.children {
                    collect(child, path, demands, modes);
                }
            }
            path.pop();
        }
        let mut modes = BTreeMap::new();
        for field in &self.dataset.schema().fields {
            collect(field, &mut Vec::new(), &demands, &mut modes);
        }
        Ok(modes)
    }
}
