// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The lambda form of [`any_element`](super):
//!
//! ```sql
//! array_any_match(detections, d -> d.label = 'car' AND d.confidence > 0.9)
//! ```
//!
//! is rewritten, while the filter is parsed, to
//!
//! ```sql
//! any_element(detections, 'label = ''car'' AND confidence > 0.9')
//! ```
//!
//! The lambda never reaches DataFusion's lambda planning, Substrait or
//! execution: everything after parsing sees `any_element`. Its body is printed
//! back as written, with `d.field` and `d['field']` as `field`, so Lance's
//! literal handling applies inside it.
//!
//! Only this shape is accepted. The body may refer to the fields of the lambda
//! parameter and nothing else, and any other lambda is an error.

use std::collections::VecDeque;
use std::ops::ControlFlow;

use arrow_schema::DataType;
use datafusion::sql::sqlparser::ast::{
    AccessExpr, BinaryOperator, Expr as SQLExpr, FunctionArg, FunctionArgExpr,
    FunctionArgumentList, FunctionArguments, Ident, LambdaFunction, ObjectName, ObjectNamePart,
    OneOrManyWithParens, Subscript, Value, Visit, VisitMut, Visitor, VisitorMut, visit_expressions,
    visit_expressions_mut,
};
use lance_core::{Error, Result};

use super::{ANY_ELEMENT, element_schema};

/// Higher-order functions whose lambda form is rewritten to `any_element`.
const LAMBDA_FUNCTIONS: [&str; 2] = ["array_any_match", "any_match"];

const SUPPORTED_FORM: &str =
    "array_any_match(<list<struct> column>, x -> <predicate on the fields of x>)";

/// Rewrite every `array_any_match(list, x -> predicate)` in a parsed filter to
/// `any_element(list, 'predicate')`, and reject any other lambda.
///
/// Before visiting, each chain of one logical operator (`AND` or `OR`) is
/// regrouped into a balanced tree. The visitors below recurse over the AST,
/// and `a AND b AND ...` parses as a chain as deep as its operand count, so
/// balancing bounds that recursion. Only one operator is regrouped at a time,
/// which leaves the filter's meaning unchanged.
///
/// `type_of` returns the data type of an argument, to check that it is a
/// `list<struct>`.
pub fn rewrite_element_lambdas(
    filter: &mut SQLExpr,
    type_of: impl Fn(&SQLExpr) -> Result<DataType>,
) -> Result<()> {
    // The visitors below recurse over the AST, and a filter such as
    // `a AND b AND ...` parses as a chain as deep as its number of operands.
    // Balance those chains first; `Planner::binary_expr` rebuilds the same
    // chains balanced after parsing, so the result is unchanged.
    let _ = VisitMut::visit(filter, &mut BalanceLogicalBinaryExprs);
    if let ControlFlow::Break(err) = VisitMut::visit(filter, &mut LambdaRewriter { type_of }) {
        return Err(err);
    }
    let mut unsupported = None;
    let _ = visit_expressions(filter, |expr| match expr {
        SQLExpr::Lambda(lambda) => {
            unsupported = Some(lambda.to_string());
            ControlFlow::Break(())
        }
        _ => ControlFlow::Continue(()),
    });
    match unsupported {
        Some(lambda) => Err(Error::invalid_input(format!(
            "the lambda `{lambda}` is not supported: Lance filters only accept lambdas as {SUPPORTED_FORM}"
        ))),
        None => Ok(()),
    }
}

struct BalanceLogicalBinaryExprs;

impl VisitorMut for BalanceLogicalBinaryExprs {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<()> {
        balance_logical_binary_expr(expr);
        ControlFlow::Continue(())
    }
}

/// Rebuild `expr`, when it is a chain of `AND` or `OR`, as a balanced tree.
///
/// Only one associative operator is regrouped: `(a AND b) AND c` becomes
/// `a AND (b AND c)`, while `AND` and `OR` are never mixed.
fn balance_logical_binary_expr(expr: &mut SQLExpr) {
    let SQLExpr::BinaryOp { op, .. } = expr else {
        return;
    };
    if !matches!(op, BinaryOperator::And | BinaryOperator::Or) {
        return;
    }

    let placeholder = SQLExpr::value(Value::Null);
    let (left, op, right) = match std::mem::replace(expr, placeholder) {
        SQLExpr::BinaryOp {
            left,
            op: op @ (BinaryOperator::And | BinaryOperator::Or),
            right,
        } => (left, op, right),
        other => {
            *expr = other;
            return;
        }
    };

    let mut stack = vec![*right, *left];
    let mut leaves = Vec::new();
    while let Some(expr) = stack.pop() {
        match expr {
            SQLExpr::BinaryOp {
                left,
                op: child_op,
                right,
            } if child_op == op => {
                stack.push(*right);
                stack.push(*left);
            }
            leaf => leaves.push(leaf),
        }
    }

    let mut exprs = VecDeque::from(leaves);
    while exprs.len() > 1 {
        let mut next = VecDeque::with_capacity(exprs.len().div_ceil(2));
        while let Some(left) = exprs.pop_front() {
            if let Some(right) = exprs.pop_front() {
                next.push_back(SQLExpr::BinaryOp {
                    left: Box::new(left),
                    op: op.clone(),
                    right: Box::new(right),
                });
            } else {
                next.push_back(left);
            }
        }
        exprs = next;
    }

    debug_assert_eq!(exprs.len(), 1, "balancing keeps one expression");
    if let Some(balanced) = exprs.pop_front() {
        *expr = balanced;
    }
}

struct LambdaRewriter<F> {
    type_of: F,
}

impl<F: Fn(&SQLExpr) -> Result<DataType>> VisitorMut for LambdaRewriter<F> {
    type Break = Error;

    // Pre-order: a call is rewritten before the visitor reaches its lambda, so
    // only lambdas this rewrite does not handle are left for the final check.
    fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<Error> {
        match self.rewrite(expr) {
            Ok(Some(rewritten)) => {
                *expr = rewritten;
                ControlFlow::Continue(())
            }
            Ok(None) => ControlFlow::Continue(()),
            Err(err) => ControlFlow::Break(err),
        }
    }
}

impl<F: Fn(&SQLExpr) -> Result<DataType>> LambdaRewriter<F> {
    /// The `any_element` call for `expr`, when it is a supported lambda call.
    fn rewrite(&self, expr: &SQLExpr) -> Result<Option<SQLExpr>> {
        let SQLExpr::Function(function) = expr else {
            return Ok(None);
        };
        let [ObjectNamePart::Identifier(name)] = function.name.0.as_slice() else {
            return Ok(None);
        };
        let name = name.value.to_ascii_lowercase();
        if !LAMBDA_FUNCTIONS.contains(&name.as_str()) {
            return Ok(None);
        }
        let FunctionArguments::List(arguments) = &function.args else {
            return Ok(None);
        };
        let [
            FunctionArg::Unnamed(FunctionArgExpr::Expr(list)),
            FunctionArg::Unnamed(FunctionArgExpr::Expr(SQLExpr::Lambda(lambda))),
        ] = arguments.args.as_slice()
        else {
            return Ok(None);
        };
        let param = lambda_parameter(&name, lambda)?;
        let list_type = (self.type_of)(list)?;
        if element_schema(&list_type).is_err() {
            return Err(Error::invalid_input(format!(
                "{name}: `{list}` is {list_type}, but Lance filters only accept {SUPPORTED_FORM}"
            )));
        }
        let predicate = element_predicate(&name, param, &lambda.body)?;
        let mut call = function.clone();
        call.name = ObjectName::from(vec![Ident::new(ANY_ELEMENT)]);
        call.args = FunctionArguments::List(FunctionArgumentList {
            duplicate_treatment: None,
            args: vec![
                FunctionArg::Unnamed(FunctionArgExpr::Expr(list.clone())),
                FunctionArg::Unnamed(FunctionArgExpr::Expr(SQLExpr::value(
                    Value::SingleQuotedString(predicate),
                ))),
            ],
            clauses: vec![],
        });
        Ok(Some(SQLExpr::Function(call)))
    }
}

fn lambda_parameter<'a>(function: &str, lambda: &'a LambdaFunction) -> Result<&'a Ident> {
    match &lambda.params {
        OneOrManyWithParens::One(param) => Ok(&param.name),
        OneOrManyWithParens::Many(params) if params.len() == 1 => Ok(&params[0].name),
        OneOrManyWithParens::Many(params) => Err(Error::invalid_input(format!(
            "{function}: the lambda `{lambda}` has {} parameters, but it takes one, the element",
            params.len()
        ))),
    }
}

/// The `any_element` predicate for `param -> body`: the body with
/// `param.field` and `param['field']` written as `field`.
fn element_predicate(function: &str, param: &Ident, body: &SQLExpr) -> Result<String> {
    let mut checker = BodyChecker {
        function,
        param,
        is_param_root_next: false,
    };
    if let ControlFlow::Break(err) = Visit::visit(body, &mut checker) {
        return Err(err);
    }
    let mut body = body.clone();
    // Post-order: the root of `param.a[1].b` is rewritten before the access
    // chain around it.
    let _ = visit_expressions_mut(&mut body, |expr| {
        if let Some(field) = without_param(param, expr) {
            *expr = field;
        }
        ControlFlow::<()>::Continue(())
    });
    Ok(body.to_string())
}

/// `expr` with the lambda parameter removed, when it is `param.field...` or
/// `param['field']...`.
fn without_param(param: &Ident, expr: &SQLExpr) -> Option<SQLExpr> {
    match expr {
        SQLExpr::CompoundIdentifier(path) if starts_with_param(param, path) => {
            Some(match &path[1..] {
                [field] => SQLExpr::Identifier(field.clone()),
                fields => SQLExpr::CompoundIdentifier(fields.to_vec()),
            })
        }
        SQLExpr::CompoundFieldAccess { root, access_chain } if is_param(param, root) => {
            let (first, rest) = access_chain.split_first()?;
            let field = SQLExpr::Identifier(accessed_field(first)?);
            Some(if rest.is_empty() {
                field
            } else {
                SQLExpr::CompoundFieldAccess {
                    root: Box::new(field),
                    access_chain: rest.to_vec(),
                }
            })
        }
        _ => None,
    }
}

fn is_param(param: &Ident, expr: &SQLExpr) -> bool {
    matches!(expr, SQLExpr::Identifier(name) if name.value == param.value)
}

/// Whether `path` is `param.field...`.
fn starts_with_param(param: &Ident, path: &[Ident]) -> bool {
    path.len() > 1 && path[0].value == param.value
}

/// The field that `.field` or `['field']` names.
fn accessed_field(access: &AccessExpr) -> Option<Ident> {
    match access {
        AccessExpr::Dot(SQLExpr::Identifier(field)) => Some(field.clone()),
        AccessExpr::Subscript(Subscript::Index {
            index: SQLExpr::Value(value),
        }) => match &value.value {
            Value::SingleQuotedString(field) => Some(Ident::with_quote('`', field)),
            _ => None,
        },
        _ => None,
    }
}

/// Checks that a lambda body refers only to fields of its parameter.
struct BodyChecker<'a> {
    function: &'a str,
    param: &'a Ident,
    /// The next expression visited is the root of a `param[...]` access that
    /// was already checked.
    is_param_root_next: bool,
}

impl Visitor for BodyChecker<'_> {
    type Break = Error;

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<Error> {
        if std::mem::take(&mut self.is_param_root_next) {
            return ControlFlow::Continue(());
        }
        let param = &self.param.value;
        let problem = match expr {
            SQLExpr::Lambda(_) => Some(
                "a lambda inside the predicate is not supported; use any_element for a nested list"
                    .to_string(),
            ),
            SQLExpr::CompoundIdentifier(path) if starts_with_param(self.param, path) => None,
            SQLExpr::CompoundFieldAccess { root, access_chain } if is_param(self.param, root) => {
                self.is_param_root_next = true;
                match access_chain.first().and_then(accessed_field) {
                    Some(_) => None,
                    None => Some(format!(
                        "`{expr}` must name a field of `{param}`, such as {param}['label']"
                    )),
                }
            }
            SQLExpr::Identifier(name) if name.value == *param => Some(format!(
                "refer to the fields of `{param}`, such as {param}.label, not to `{param}` itself"
            )),
            // Lance reads a double-quoted identifier as a string literal.
            SQLExpr::Identifier(name) if name.quote_style == Some('"') => None,
            SQLExpr::Identifier(_) | SQLExpr::CompoundIdentifier(_) => Some(format!(
                "the predicate may only refer to fields of `{param}`, but `{expr}` is outside the \
                 element; put conditions on other columns outside the call"
            )),
            _ => None,
        };
        match problem {
            Some(problem) => ControlFlow::Break(Error::invalid_input(format!(
                "{}: {problem}",
                self.function
            ))),
            None => ControlFlow::Continue(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_schema::{Field, Fields, Schema};
    use rstest::rstest;

    use crate::planner::Planner;
    use crate::sql::parse_sql_filter;

    fn planner() -> Planner {
        let element = Fields::from(vec![
            Field::new("label", DataType::Utf8, true),
            Field::new("confidence", DataType::Float32, true),
            Field::new(
                "details",
                DataType::Struct(Fields::from(vec![Field::new("kind", DataType::Utf8, true)])),
                true,
            ),
            Field::new("my field", DataType::Int32, true),
        ]);
        Planner::new(Arc::new(Schema::new(vec![
            Field::new("width", DataType::Int32, false),
            Field::new(
                "detections",
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::Struct(element),
                    true,
                ))),
                true,
            ),
        ])))
    }

    fn rewritten(filter: &str) -> Result<String> {
        let mut expr = parse_sql_filter(filter)?;
        planner().rewrite_filter_lambdas(&mut expr)?;
        Ok(expr.to_string())
    }

    #[rstest]
    #[case::same_element(
        "array_any_match(detections, d -> d.label = 'car' AND d.confidence > 0.9)",
        "any_element(detections, 'label = ''car'' AND confidence > 0.9')"
    )]
    #[case::alias_and_parenthesized_parameter(
        "ANY_MATCH(detections, (x) -> x.label IN ('person', 'truck'))",
        "any_element(detections, 'label IN (''person'', ''truck'')')"
    )]
    #[case::nested_field(
        "array_any_match(detections, d -> d.details.kind = 'suv')",
        "any_element(detections, 'details.kind = ''suv''')"
    )]
    #[case::subscript(
        "array_any_match(detections, d -> d['my field'] > 3 AND d['details']['kind'] IS NULL)",
        "any_element(detections, '`my field` > 3 AND `details`[''kind''] IS NULL')"
    )]
    #[case::double_quoted_string(
        "array_any_match(detections, d -> d.label = \"car\")",
        "any_element(detections, 'label = \"car\"')"
    )]
    #[case::inside_other_expressions(
        "NOT array_any_match(detections, d -> d.confidence < 0.5) AND width > 1920",
        "NOT any_element(detections, 'confidence < 0.5') AND width > 1920"
    )]
    #[case::inside_a_function_call(
        "coalesce(array_any_match(detections, d -> d.label = 'car'), false)",
        "coalesce(any_element(detections, 'label = ''car'''), false)"
    )]
    fn test_rewrite(#[case] filter: &str, #[case] expected: &str) {
        assert_eq!(rewritten(filter).unwrap(), expected);
    }

    /// A lambda at the far end of a long `AND`/`OR` chain is still rewritten.
    /// The chain is balanced first, so this walks its leaves with an explicit
    /// stack rather than assuming the parsed left-deep shape.
    #[rstest]
    fn test_rewrite_deep_logical_filter(#[values("AND", "OR")] op: &str) {
        let mut conditions = vec!["width > 1"; 1000];
        conditions.push("array_any_match(detections, d -> d.label = 'car')");
        let mut expr = parse_sql_filter(&conditions.join(&format!(" {op} "))).unwrap();
        planner().rewrite_filter_lambdas(&mut expr).unwrap();

        let mut stack = vec![&expr];
        let mut leaves = Vec::new();
        while let Some(expr) = stack.pop() {
            match expr {
                SQLExpr::BinaryOp {
                    left,
                    op: BinaryOperator::And | BinaryOperator::Or,
                    right,
                } => {
                    stack.push(right.as_ref());
                    stack.push(left.as_ref());
                }
                leaf => leaves.push(leaf),
            }
        }
        assert_eq!(leaves.len(), 1001);
        assert_eq!(
            leaves.last().unwrap().to_string(),
            "any_element(detections, 'label = ''car''')"
        );

        let mut has_lambda = false;
        let _ = visit_expressions(&expr, |expr| {
            has_lambda |= matches!(expr, SQLExpr::Lambda(_));
            ControlFlow::<()>::Continue(())
        });
        assert!(!has_lambda, "a lambda was left unrewritten");
    }

    #[rstest]
    #[case::outside_column(
        "array_any_match(detections, d -> d.label = 'car' AND width > 1)",
        "`width` is outside the element"
    )]
    #[case::column_without_parameter(
        "array_any_match(detections, d -> label = 'car')",
        "`label` is outside the element"
    )]
    #[case::whole_element("array_any_match(detections, d -> d IS NOT NULL)", "not to `d` itself")]
    #[case::numeric_subscript(
        "array_any_match(detections, d -> d[1] IS NULL)",
        "must name a field"
    )]
    #[case::two_parameters(
        "array_any_match(detections, (a, b) -> a.label = 'car')",
        "has 2 parameters"
    )]
    #[case::nested_lambda(
        "array_any_match(detections, d -> array_any_match(d.parts, p -> p.x > 1))",
        "use any_element for a nested list"
    )]
    #[case::not_a_list_of_structs("array_any_match(width, x -> x.y > 1)", "`width` is Int32")]
    #[case::other_function(
        "cardinality(array_filter(detections, d -> d.confidence > 0.5)) > 0",
        "only accept lambdas as array_any_match"
    )]
    #[case::bare_lambda("x -> x > 1", "only accept lambdas as array_any_match")]
    fn test_rewrite_errors_are_clear(#[case] filter: &str, #[case] message: &str) {
        let err = rewritten(filter).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
        assert!(err.to_string().contains(message), "{filter}: {err}");
    }
}
