// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Extends logical expression.

use std::sync::Arc;

use arrow_schema::DataType;

use crate::expr::safe_coerce_scalar;
use crate::numeric_coercion::rewrite as rewrite_numeric;
use datafusion::common::DFSchema;
use datafusion::logical_expr::{Between, ScalarUDFImpl};
use datafusion::logical_expr::{BinaryExpr, Operator};
use datafusion::optimizer::simplify_expressions::{ExprSimplifier, SimplifyContext};
use datafusion::prelude::*;
use datafusion::scalar::ScalarValue;
use datafusion_functions::core::expr_ext::FieldAccessor;
use datafusion_functions::core::getfield::GetFieldFunc;
use lance_arrow::DataTypeExt;

use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
/// Resolve a Value
fn resolve_value(expr: &Expr, data_type: &DataType) -> Result<Expr> {
    match expr {
        Expr::Literal(scalar_value, metadata) => {
            Ok(Expr::Literal(safe_coerce_scalar(scalar_value, data_type).ok_or_else(|| Error::invalid_input(format!("Received literal {expr} and could not convert to literal of type '{data_type:?}'")))?, metadata.clone()))
        }
        _ => Err(Error::invalid_input(format!("Expected a literal of type '{data_type:?}' but received: {expr}"))),
    }
}

/// A simple helper function that interprets an Expr as a string scalar
/// or returns None if it is not.
pub fn get_as_string_scalar_opt(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _) => Some(s),
        _ => None,
    }
}

/// Given a column or nested `get_field` expression, get the data type of the referenced
/// field in the schema.
///
/// If the column is not found in the schema, return None. If the expression is
/// not a field reference, also returns None.
pub fn resolve_column_type(expr: &Expr, schema: &Schema) -> Option<DataType> {
    let mut field_path = Vec::new();
    let mut current_expr = expr;
    // We are looping from outer-most reference to inner-most.
    loop {
        match current_expr {
            Expr::Column(c) => {
                field_path.push(c.name.as_str());
                break;
            }
            Expr::ScalarFunction(udf) if udf.name() == GetFieldFunc::default().name() => {
                let name = get_as_string_scalar_opt(&udf.args[1])?;
                field_path.push(name);
                current_expr = &udf.args[0];
            }
            _ => return None,
        }
    }

    let mut path_iter = field_path.iter().rev();
    let mut field = schema.field(path_iter.next()?)?;
    for name in path_iter {
        if field.data_type().is_struct() {
            field = field.children.iter().find(|f| &f.name == name)?;
        } else {
            return None;
        }
    }
    Some(field.data_type())
}

fn is_literal_only_same_type_expr(expr: &Expr) -> bool {
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            (op.is_numerical_operators()
                || matches!(
                    op,
                    Operator::BitwiseShiftLeft
                        | Operator::BitwiseShiftRight
                        | Operator::StringConcat
                ))
                && is_literal_only_same_type_expr(left)
                && is_literal_only_same_type_expr(right)
        }
        Expr::Literal(..) => true,
        _ => false,
    }
}

/// Whether `expr` is arithmetic on literals only, such as `100 + 100`.
fn is_constant_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::BinaryExpr(_)) && is_literal_only_same_type_expr(expr)
}

/// Folds a constant expression to a literal, or returns `None` when it does not fold.
fn simplify_constant(expr: Expr) -> Option<ScalarValue> {
    let schema = Arc::new(DFSchema::empty());
    let context = SimplifyContext::builder().with_schema(schema.clone());
    let simplifier = ExprSimplifier::new(context.build());
    let folded = simplifier
        .coerce(expr, &schema)
        .and_then(|expr| simplifier.simplify(expr))
        .ok()?;
    match folded {
        Expr::Literal(value, _) => Some(value),
        _ => None,
    }
}

/// The exact value of `+`, `-`, `*`, `/`, `%`, `<<` and `>>` on integer
/// literals, or `None` for any other operator or operand, a division by zero,
/// a shift by 64 or more, or a value beyond `i128`.
fn exact_integer(expr: &Expr) -> Option<i128> {
    // Beyond 63 the operand types mask the shift, so the exact value would differ.
    let shift = |by: i128| u32::try_from(by).ok().filter(|by| *by < 64);
    match expr {
        Expr::Literal(value, _) => match value {
            ScalarValue::Int8(Some(v)) => Some((*v).into()),
            ScalarValue::Int16(Some(v)) => Some((*v).into()),
            ScalarValue::Int32(Some(v)) => Some((*v).into()),
            ScalarValue::Int64(Some(v)) => Some((*v).into()),
            ScalarValue::UInt8(Some(v)) => Some((*v).into()),
            ScalarValue::UInt16(Some(v)) => Some((*v).into()),
            ScalarValue::UInt32(Some(v)) => Some((*v).into()),
            ScalarValue::UInt64(Some(v)) => Some((*v).into()),
            _ => None,
        },
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            let (left, right) = (exact_integer(left)?, exact_integer(right)?);
            match op {
                Operator::Plus => left.checked_add(right),
                Operator::Minus => left.checked_sub(right),
                Operator::Multiply => left.checked_mul(right),
                Operator::Divide => left.checked_div(right),
                Operator::Modulo => left.checked_rem(right),
                Operator::BitwiseShiftLeft => left.checked_mul(1 << shift(right)?),
                Operator::BitwiseShiftRight => Some(left >> shift(right)?),
                _ => None,
            }
        }
        _ => None,
    }
}

/// `value` as a scalar of the integer type `data_type`, when that type holds it.
fn integer_scalar(value: i128, data_type: &DataType) -> Option<ScalarValue> {
    Some(match data_type {
        DataType::Int8 => ScalarValue::Int8(Some(value.try_into().ok()?)),
        DataType::Int16 => ScalarValue::Int16(Some(value.try_into().ok()?)),
        DataType::Int32 => ScalarValue::Int32(Some(value.try_into().ok()?)),
        DataType::Int64 => ScalarValue::Int64(Some(value.try_into().ok()?)),
        DataType::UInt8 => ScalarValue::UInt8(Some(value.try_into().ok()?)),
        DataType::UInt16 => ScalarValue::UInt16(Some(value.try_into().ok()?)),
        DataType::UInt32 => ScalarValue::UInt32(Some(value.try_into().ok()?)),
        DataType::UInt64 => ScalarValue::UInt64(Some(value.try_into().ok()?)),
        _ => return None,
    })
}

/// Folds a constant expression compared with a column of type `data_type`.
///
/// The expression is evaluated as written, in its literals' own types, so it
/// keeps its arithmetic, NULL and errors: `(1 << 7) > i` compares with 128 on
/// an `Int8` column, and `7 / 0 = f` is left to fail as before. Only the result
/// is then resolved like any literal compared with the column, which narrows it
/// to the column type when it fits. Integer arithmetic whose exact result the
/// column's integer type holds folds to that result, so `u = (1 << 63)` matches
/// 2^63 on a `UInt64` column although `Int64` wraps it.
fn fold_constant(expr: &Expr, data_type: Option<&DataType>) -> Option<Expr> {
    if !is_constant_expr(expr) {
        return None;
    }
    if let Some(value) =
        data_type.and_then(|data_type| integer_scalar(exact_integer(expr)?, data_type))
    {
        return Some(lit(value));
    }
    let value = simplify_constant(expr.clone())?;
    // A NULL result compares as NULL whatever its type.
    let value = match data_type {
        Some(data_type) if value.is_null() => ScalarValue::try_from(data_type).unwrap_or(value),
        _ => value,
    };
    Some(lit(value))
}

/// Whether `op` is one of the comparisons `numeric_coercion::rewrite` resolves.
/// Only these narrow a constant operand to the other side's column type;
/// arithmetic keeps DataFusion's promotion, so a projection such as
/// `(25 * 2) + i` is not narrowed.
fn is_comparison(op: &Operator) -> bool {
    matches!(
        op,
        Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
    )
}

/// Folds the constant operands of a comparison, `BETWEEN` or `IN` list, which
/// are then resolved like any other literal. Returns `None` when no operand folds.
fn fold_constant_operands(expr: &Expr, schema: &Schema) -> Option<Expr> {
    let fold = |operand: &Expr, column: &Expr| {
        fold_constant(operand, resolve_column_type(column, schema).as_ref())
            .unwrap_or_else(|| operand.clone())
    };
    let folded = match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right })
            if is_comparison(op) && (is_constant_expr(left) || is_constant_expr(right)) =>
        {
            Expr::BinaryExpr(BinaryExpr::new(
                Box::new(fold(left, right)),
                *op,
                Box::new(fold(right, left)),
            ))
        }
        Expr::Between(between)
            if is_constant_expr(&between.low) || is_constant_expr(&between.high) =>
        {
            Expr::Between(Between {
                low: Box::new(fold(&between.low, &between.expr)),
                high: Box::new(fold(&between.high, &between.expr)),
                ..between.clone()
            })
        }
        Expr::InList(in_list) if in_list.list.iter().any(is_constant_expr) => Expr::in_list(
            in_list.expr.as_ref().clone(),
            in_list
                .list
                .iter()
                .map(|item| fold(item, &in_list.expr))
                .collect(),
            in_list.negated,
        ),
        _ => return None,
    };
    (folded != *expr).then_some(folded)
}

/// Resolve logical expression `expr`.
///
/// Parameters
///
/// - *expr*: a datafusion logical expression
/// - *schema*: lance schema.
pub fn resolve_expr(expr: &Expr, schema: &Schema) -> Result<Expr> {
    if let Some(folded) = fold_constant_operands(expr, schema) {
        return resolve_expr(&folded, schema);
    }
    if let Some(rewritten) = rewrite_numeric(expr, schema) {
        return Ok(rewritten);
    }
    match expr {
        // A predicate wrapped in NOT or IS [NOT] TRUE/FALSE compares the same
        // literal, so it has to be narrowed the same way.
        Expr::Not(inner) => Ok(Expr::Not(Box::new(resolve_expr(inner.as_ref(), schema)?))),
        Expr::IsTrue(inner) => Ok(Expr::IsTrue(Box::new(resolve_expr(inner, schema)?))),
        Expr::IsFalse(inner) => Ok(Expr::IsFalse(Box::new(resolve_expr(inner, schema)?))),
        Expr::IsNotTrue(inner) => Ok(Expr::IsNotTrue(Box::new(resolve_expr(inner, schema)?))),
        Expr::IsNotFalse(inner) => Ok(Expr::IsNotFalse(Box::new(resolve_expr(inner, schema)?))),
        Expr::Between(Between {
            expr: inner_expr,
            low,
            high,
            negated,
        }) => {
            if let Some(inner_expr_type) = resolve_column_type(inner_expr.as_ref(), schema) {
                Ok(Expr::Between(Between {
                    expr: inner_expr.clone(),
                    low: Box::new(coerce_expr(low.as_ref(), &inner_expr_type)?),
                    high: Box::new(coerce_expr(high.as_ref(), &inner_expr_type)?),
                    negated: *negated,
                }))
            } else {
                Ok(expr.clone())
            }
        }
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            if matches!(op, Operator::And | Operator::Or) {
                Ok(Expr::BinaryExpr(BinaryExpr {
                    left: Box::new(resolve_expr(left.as_ref(), schema)?),
                    op: *op,
                    right: Box::new(resolve_expr(right.as_ref(), schema)?),
                }))
            } else if let Some(left_type) = resolve_column_type(left.as_ref(), schema) {
                match right.as_ref() {
                    Expr::Literal(..) | Expr::Cast(_) => Ok(Expr::BinaryExpr(BinaryExpr {
                        left: left.clone(),
                        op: *op,
                        right: Box::new(coerce_expr(right.as_ref(), &left_type)?),
                    })),
                    // Let DataFusion coerce other expressions. Recursively resolving them
                    // can narrow literals to an inner column's type before arithmetic promotes
                    // its operands, for example the 0.5 in `float_col = int_col + 0.5`.
                    _ => Ok(expr.clone()),
                }
            } else if let Some(right_type) = resolve_column_type(right.as_ref(), schema) {
                match left.as_ref() {
                    Expr::Literal(..) | Expr::Cast(_) => Ok(Expr::BinaryExpr(BinaryExpr {
                        left: Box::new(coerce_expr(left.as_ref(), &right_type)?),
                        op: *op,
                        right: right.clone(),
                    })),
                    _ => Ok(expr.clone()),
                }
            } else {
                Ok(expr.clone())
            }
        }
        Expr::InList(in_list) => {
            if matches!(in_list.expr.as_ref(), Expr::Column(_)) {
                if let Some(resolved_type) = resolve_column_type(in_list.expr.as_ref(), schema) {
                    let resolved_values = in_list
                        .list
                        .iter()
                        .map(|val| coerce_expr(val, &resolved_type))
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Expr::in_list(
                        in_list.expr.as_ref().clone(),
                        resolved_values,
                        in_list.negated,
                    ))
                } else {
                    Ok(expr.clone())
                }
            } else {
                Ok(expr.clone())
            }
        }
        _ => {
            // Passthrough
            Ok(expr.clone())
        }
    }
}

/// Coerce expression of literals to column type.
///
/// Parameters
///
/// - *expr*: a datafusion logical expression
/// - *dtype*: a lance data type
pub fn coerce_expr(expr: &Expr, dtype: &DataType) -> Result<Expr> {
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => Ok(Expr::BinaryExpr(BinaryExpr {
            left: Box::new(coerce_expr(left, dtype)?),
            op: *op,
            right: Box::new(coerce_expr(right, dtype)?),
        })),
        literal_expr @ Expr::Literal(..) => Ok(resolve_value(literal_expr, dtype)?),
        Expr::Cast(_) => Ok(narrow_double_literal(expr, dtype).unwrap_or_else(|| expr.clone())),
        _ => Ok(expr.clone()),
    }
}

/// Narrows a double literal such as `CAST(0.5 AS double)` to a `Float32` column's
/// type when the value converts exactly, so the column is not cast and stays
/// indexable.
fn narrow_double_literal(expr: &Expr, dtype: &DataType) -> Option<Expr> {
    let Expr::Cast(cast) = expr else {
        return None;
    };
    let Expr::Literal(value, _) = cast.expr.as_ref() else {
        return None;
    };
    if *dtype != DataType::Float32 || *cast.field.data_type() != DataType::Float64 {
        return None;
    }
    let ScalarValue::Float64(value) = value.cast_to(&DataType::Float64).ok()? else {
        return None;
    };
    let narrowed = value.map(|value| value as f32);
    // `total_cmp` also tells NaN signs and signed zeros apart.
    let is_exact = value
        .zip(narrowed)
        .is_none_or(|(value, narrowed)| f64::from(narrowed).total_cmp(&value).is_eq());
    is_exact.then(|| lit(ScalarValue::Float32(narrowed)))
}

/// Coerce logical expression for filters to boolean.
///
/// Parameters
///
/// - *expr*: a datafusion logical expression
pub fn coerce_filter_type_to_boolean(expr: Expr) -> Expr {
    match expr {
        // Coerce regexp_match to boolean by checking for non-null
        Expr::ScalarFunction(sf) if sf.func.name() == "regexp_match" => {
            log::warn!(
                "regexp_match now is coerced to boolean, this may be changed in the future, please use `regexp_like` instead"
            );
            Expr::IsNotNull(Box::new(Expr::ScalarFunction(sf)))
        }

        // Only boolean operands need coercion. Null checks and comparisons must
        // preserve regexp_match's nullable list result.
        Expr::BinaryExpr(BinaryExpr { left, op, right })
            if matches!(op, Operator::And | Operator::Or) =>
        {
            Expr::BinaryExpr(BinaryExpr {
                left: Box::new(coerce_filter_type_to_boolean(*left)),
                op,
                right: Box::new(coerce_filter_type_to_boolean(*right)),
            })
        }
        Expr::Not(inner) => Expr::Not(Box::new(coerce_filter_type_to_boolean(*inner))),

        // Pass-through for all other nodes
        other => other,
    }
}

/// Convert a field path string into a DataFusion expression.
///
/// This function handles:
/// - Simple column names: "column"
/// - Nested paths: "parent.child" or "parent.child.grandchild"
/// - Backtick-escaped field names: "parent.`field.with.dots`"
///
/// # Arguments
///
/// * `field_path` - The field path to convert. Supports simple columns, nested paths,
///   and backtick-escaped field names.
///
/// # Returns
///
/// Returns `Result<Expr>` - Ok with the DataFusion expression, or Err if the path
/// could not be parsed.
///
/// # Example
///
/// ```
/// use lance_datafusion::logical_expr::field_path_to_expr;
///
/// // Simple column
/// let expr = field_path_to_expr("column_name").unwrap();
///
/// // Nested field
/// let expr = field_path_to_expr("parent.child").unwrap();
///
/// // Backtick-escaped field with dots
/// let expr = field_path_to_expr("parent.`field.with.dots`").unwrap();
/// ```
pub fn field_path_to_expr(field_path: &str) -> Result<Expr> {
    // Parse the field path to handle nested fields and backtick-escaped names
    let parts = lance_core::datatypes::parse_field_path(field_path)?;

    if parts.is_empty() {
        return Err(Error::invalid_input(format!(
            "Invalid empty field path: {}",
            field_path
        )));
    }

    // Build the column expression, handling nested fields.
    let mut expr = Expr::Column(datafusion::common::Column::new_unqualified(
        parts[0].clone(),
    ));
    for part in &parts[1..] {
        expr = expr.field(part.as_str());
    }

    Ok(expr)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    use arrow_schema::{Field, Schema as ArrowSchema};
    use datafusion::common::Column;

    #[test]
    fn test_field_path_to_expr_preserves_case_sensitive_root_column() {
        let expr = field_path_to_expr("VECTOR").unwrap();

        assert_eq!(expr, Expr::Column(Column::new_unqualified("VECTOR")));
    }

    #[rstest::rstest]
    #[case::nested("Parent.Child", "Parent", vec!["Child"])]
    #[case::deeply_nested("Parent.Child.Leaf", "Parent", vec!["Child", "Leaf"])]
    #[case::escaped_child("Parent.`Child.With.Dot`", "Parent", vec!["Child.With.Dot"])]
    #[case::escaped_root("`Parent.With.Dot`.Child", "Parent.With.Dot", vec!["Child"])]
    #[case::escaped_nested(
        "`Parent.With.Dot`.`Child.With.Dot`.Leaf",
        "Parent.With.Dot",
        vec!["Child.With.Dot", "Leaf"]
    )]
    fn test_field_path_to_expr_preserves_nested_names(
        #[case] path: &str,
        #[case] root: &str,
        #[case] children: Vec<&str>,
    ) {
        let expr = field_path_to_expr(path).unwrap();
        let mut expected = Expr::Column(Column::new_unqualified(root));
        for name in children {
            expected = get_field(expected, name);
        }

        assert_eq!(expr, expected);
    }

    #[test]
    fn test_resolve_large_utf8() {
        let arrow_schema = ArrowSchema::new(vec![Field::new("a", DataType::LargeUtf8, false)]);
        let expr = Expr::BinaryExpr(BinaryExpr {
            left: Box::new(Expr::Column("a".to_string().into())),
            op: Operator::Eq,
            right: Box::new(Expr::Literal(
                ScalarValue::Utf8(Some("a".to_string())),
                None,
            )),
        });

        let resolved = resolve_expr(&expr, &Schema::try_from(&arrow_schema).unwrap()).unwrap();
        match resolved {
            Expr::BinaryExpr(be) => {
                assert_eq!(
                    be.right.as_ref(),
                    &Expr::Literal(ScalarValue::LargeUtf8(Some("a".to_string())), None)
                )
            }
            _ => unreachable!("Expected BinaryExpr"),
        };
    }

    #[test]
    fn test_resolve_binary_expr_on_right() {
        let arrow_schema = ArrowSchema::new(vec![Field::new("a", DataType::Float64, false)]);
        let expr = Expr::BinaryExpr(BinaryExpr {
            left: Box::new(Expr::Column("a".to_string().into())),
            op: Operator::Eq,
            right: Box::new(Expr::BinaryExpr(BinaryExpr {
                left: Box::new(Expr::Literal(ScalarValue::Int64(Some(2)), None)),
                op: Operator::Minus,
                right: Box::new(Expr::Literal(ScalarValue::Int64(Some(-1)), None)),
            })),
        });
        let resolved = resolve_expr(&expr, &Schema::try_from(&arrow_schema).unwrap()).unwrap();

        // The constant is folded first, then the result is coerced to the column type.
        assert_eq!(resolved, col("a").eq(lit(3.0_f64)));
    }

    #[test]
    fn test_resolve_in_expr() {
        // Type coercion should apply for `A IN (0)` or `A NOT IN (0)`
        let arrow_schema = ArrowSchema::new(vec![Field::new("a", DataType::Float32, false)]);
        let expr = Expr::in_list(
            Expr::Column("a".to_string().into()),
            vec![Expr::Literal(ScalarValue::Float64(Some(0.0)), None)],
            false,
        );
        let resolved = resolve_expr(&expr, &Schema::try_from(&arrow_schema).unwrap()).unwrap();
        let expected = Expr::in_list(
            Expr::Column("a".to_string().into()),
            vec![Expr::Literal(ScalarValue::Float32(Some(0.0)), None)],
            false,
        );
        assert_eq!(resolved, expected);

        let expr = Expr::in_list(
            Expr::Column("a".to_string().into()),
            vec![Expr::Literal(ScalarValue::Float64(Some(0.0)), None)],
            true,
        );
        let resolved = resolve_expr(&expr, &Schema::try_from(&arrow_schema).unwrap()).unwrap();
        let expected = Expr::in_list(
            Expr::Column("a".to_string().into()),
            vec![Expr::Literal(ScalarValue::Float32(Some(0.0)), None)],
            true,
        );
        assert_eq!(resolved, expected);
    }

    #[rstest::rstest]
    #[case::half("x > CAST(0.5 AS DOUBLE)", col("x").gt(lit(0.5_f32)))]
    #[case::reversed("CAST(0.5 AS DOUBLE) < x", lit(0.5_f32).lt(col("x")))]
    #[case::negative_infinity("x < CAST('-inf' AS DOUBLE)", col("x").lt(lit(f32::NEG_INFINITY)))]
    #[case::null("x > CAST(NULL AS DOUBLE)", col("x").gt(lit(ScalarValue::Float32(None))))]
    #[case::between(
        "x BETWEEN 0.0 AND CAST(0.5 AS DOUBLE)",
        col("x").between(lit(0.0_f32), lit(0.5_f32))
    )]
    #[case::in_list("x IN (CAST(0.5 AS DOUBLE))", col("x").in_list(vec![lit(0.5_f32)], false))]
    #[case::inexact(
        "x = CAST(0.1 AS DOUBLE)",
        col("x").eq(cast(lit(0.1), DataType::Float64))
    )]
    #[case::overflow(
        "x > CAST(1e300 AS DOUBLE)",
        col("x").gt(cast(lit(1e300), DataType::Float64))
    )]
    #[case::underflow(
        "x > CAST(1e-300 AS DOUBLE)",
        col("x").gt(cast(lit(1e-300), DataType::Float64))
    )]
    fn test_resolve_double_literal_on_float32(#[case] sql: &str, #[case] expected: Expr) {
        let schema = ArrowSchema::new(vec![Field::new("x", DataType::Float32, true)]);
        let planner = crate::planner::Planner::new(Arc::new(schema));
        assert_eq!(planner.parse_filter(sql).unwrap(), expected, "{sql}");
    }

    #[test]
    fn test_resolve_column_type() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("int", DataType::Int32, true),
            Field::new(
                "st",
                DataType::Struct(
                    vec![
                        Field::new("str", DataType::Utf8, true),
                        Field::new("Child.With.Dot", DataType::Int32, true),
                        Field::new(
                            "st",
                            DataType::Struct(
                                vec![Field::new("float", DataType::Float64, true)].into(),
                            ),
                            true,
                        ),
                    ]
                    .into(),
                ),
                true,
            ),
        ]));
        let schema = Schema::try_from(schema.as_ref()).unwrap();

        assert_eq!(
            resolve_column_type(&col("int"), &schema),
            Some(DataType::Int32)
        );
        assert_eq!(
            resolve_column_type(&col("st").field("str"), &schema),
            Some(DataType::Utf8)
        );
        assert_eq!(
            resolve_column_type(&col("st").field("st").field("float"), &schema),
            Some(DataType::Float64)
        );

        assert_eq!(
            resolve_column_type(&field_path_to_expr("st.`Child.With.Dot`").unwrap(), &schema),
            Some(DataType::Int32)
        );

        assert_eq!(resolve_column_type(&col("x"), &schema), None);
        assert_eq!(resolve_column_type(&col("str"), &schema), None);
        assert_eq!(resolve_column_type(&col("float"), &schema), None);
        assert_eq!(
            resolve_column_type(&col("st").field("str").eq(lit("x")), &schema),
            None
        );
    }

    #[test]
    fn test_resolve_utf8view_literal_against_utf8_column() {
        // Simulates DataFusion 43+ producing a Utf8View literal (e.g. from md5())
        // being compared against a Utf8 column stored in Lance.
        let arrow_schema = ArrowSchema::new(vec![Field::new("hash", DataType::Utf8, false)]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let expr = Expr::BinaryExpr(BinaryExpr {
            left: Box::new(Expr::Column("hash".to_string().into())),
            op: Operator::Eq,
            right: Box::new(Expr::Literal(
                ScalarValue::Utf8View(Some("abc".to_string())),
                None,
            )),
        });

        let resolved = resolve_expr(&expr, &schema).unwrap();
        match resolved {
            Expr::BinaryExpr(be) => {
                assert_eq!(
                    be.right.as_ref(),
                    &Expr::Literal(ScalarValue::Utf8(Some("abc".to_string())), None)
                )
            }
            _ => unreachable!("Expected BinaryExpr"),
        }
    }

    #[test]
    fn test_resolve_typed_null_against_dictionary_column() {
        // A dictionary-encoded string column, e.g. a categorical field.
        let dict_ty = DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8));
        let arrow_schema = ArrowSchema::new(vec![Field::new("etld", dict_ty, true)]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        // A typed null must be wrapped in the dictionary type, not left as a bare
        // `Utf8(None)` literal sitting next to a `Dictionary(...)` column.
        let expected_null = Expr::Literal(
            ScalarValue::Dictionary(Box::new(DataType::Int16), Box::new(ScalarValue::Utf8(None))),
            None,
        );

        // `etld = <typed null>` built directly via the API, as opposed to coming
        // through SQL parsing.
        let expr = Expr::BinaryExpr(BinaryExpr {
            left: Box::new(Expr::Column("etld".to_string().into())),
            op: Operator::Eq,
            right: Box::new(Expr::Literal(ScalarValue::Utf8(None), None)),
        });
        match resolve_expr(&expr, &schema).unwrap() {
            Expr::BinaryExpr(be) => assert_eq!(be.right.as_ref(), &expected_null),
            other => unreachable!("Expected BinaryExpr, got {other:?}"),
        }

        // `etld IN ('a', <typed null>)` — a typed value mixed with a typed null,
        // both already typed as Utf8. Every list element is wrapped in the
        // dictionary type.
        let expr = Expr::in_list(
            Expr::Column("etld".to_string().into()),
            vec![
                Expr::Literal(ScalarValue::Utf8(Some("a".to_string())), None),
                Expr::Literal(ScalarValue::Utf8(None), None),
            ],
            false,
        );
        let expected = Expr::in_list(
            Expr::Column("etld".to_string().into()),
            vec![
                Expr::Literal(
                    ScalarValue::Dictionary(
                        Box::new(DataType::Int16),
                        Box::new(ScalarValue::Utf8(Some("a".to_string()))),
                    ),
                    None,
                ),
                expected_null,
            ],
            false,
        );
        assert_eq!(resolve_expr(&expr, &schema).unwrap(), expected);
    }

    fn constant_expression_batch() -> arrow_array::RecordBatch {
        use arrow_array::{Float64Array, Int8Array, UInt8Array};
        arrow_array::RecordBatch::try_from_iter(vec![
            (
                "i",
                Arc::new(Int8Array::from(vec![Some(-56), Some(50), Some(100), None]))
                    as arrow_array::ArrayRef,
            ),
            (
                "u",
                Arc::new(UInt8Array::from(vec![Some(0), Some(10), Some(255), None])),
            ),
            (
                "f",
                Arc::new(Float64Array::from(vec![
                    Some(3.0),
                    Some(3.5),
                    Some(1.0),
                    None,
                ])),
            ),
        ])
        .unwrap()
    }

    /// Plans the filter `sql` on [`constant_expression_batch`] and evaluates it,
    /// returning the first planning or evaluation error as text.
    fn evaluate_filter(sql: &str) -> std::result::Result<arrow_array::ArrayRef, String> {
        let batch = constant_expression_batch();
        let planner = crate::planner::Planner::new(batch.schema());
        let expr = planner.parse_filter(sql).map_err(|e| e.to_string())?;
        let expr = planner.optimize_expr(expr).map_err(|e| e.to_string())?;
        let expr = planner
            .create_physical_expr(&expr)
            .map_err(|e| e.to_string())?;
        let value = expr.evaluate(&batch).map_err(|e| e.to_string())?;
        value
            .into_array(batch.num_rows())
            .map_err(|e| e.to_string())
    }

    /// A constant expression keeps its own value whichever side of the
    /// comparison it is on: `i < 100 + 100` matches the same rows as
    /// `100 + 100 > i` and `i < 200`, rather than adding the literals as `Int8`.
    /// It is evaluated as written, in its literals' own types, before only its
    /// result is narrowed to the column type.
    #[rstest::rstest]
    #[case::sum_above_type("i < 100 + 100", [Some(true), Some(true), Some(true), None])]
    #[case::sum_above_type_reversed("100 + 100 > i", [Some(true), Some(true), Some(true), None])]
    #[case::gt_sum_above_type("i > 100 + 100", [Some(false), Some(false), Some(false), None])]
    #[case::eq_sum_above_type("i = 100 + 100", [Some(false), Some(false), Some(false), None])]
    #[case::in_sum_above_type("i IN (100 + 100)", [Some(false), Some(false), Some(false), None])]
    #[case::between_sum_above_type(
        "i BETWEEN 100 + 100 AND 127",
        [Some(false), Some(false), Some(false), None]
    )]
    #[case::max_plus_one("i < 127 + 1", [Some(true), Some(true), Some(true), None])]
    #[case::operand_above_type("i < 200 - 100", [Some(true), Some(true), Some(false), None])]
    #[case::float_operand("i > 1.5 + 1", [Some(false), Some(true), Some(true), None])]
    #[case::in_range("i = 25 * 2", [Some(false), Some(true), Some(false), None])]
    #[case::shift_above_type("(1 << 7) > i", [Some(true), Some(true), Some(true), None])]
    #[case::shift_above_type_column_first("i < (1 << 7)", [Some(true), Some(true), Some(true), None])]
    #[case::unsigned_below_zero("u > 0 - 1", [Some(true), Some(true), Some(true), None])]
    #[case::unsigned_difference_below_zero(
        "u < 5 - 10",
        [Some(false), Some(false), Some(false), None]
    )]
    #[case::integers_before_float(
        "9007199254740993 - 9007199254740992 = f",
        [Some(false), Some(false), Some(true), None]
    )]
    #[case::integers_before_float_column_first(
        "f = 9007199254740993 - 9007199254740992",
        [Some(false), Some(false), Some(true), None]
    )]
    #[case::integer_division("7 / 2 = f", [Some(true), Some(false), Some(false), None])]
    #[case::integer_division_column_first("f = 7 / 2", [Some(true), Some(false), Some(false), None])]
    #[case::float_division("f = 7.0 / 2.0", [Some(false), Some(true), Some(false), None])]
    #[case::distinct_from_sum_above_type(
        "i IS DISTINCT FROM 100 + 100",
        [Some(true), Some(true), Some(true), Some(true)]
    )]
    #[case::null_operand("i = 1 + NULL", [None, None, None, None])]
    #[case::null_operand_reversed("1 + NULL = i", [None, None, None, None])]
    fn constant_expressions_keep_their_value(
        #[case] sql: &str,
        #[case] expected: [Option<bool>; 4],
    ) {
        use arrow_array::{BooleanArray, cast::AsArray};

        let result = evaluate_filter(sql).unwrap_or_else(|error| panic!("{sql}: {error}"));
        assert_eq!(
            result.as_boolean(),
            &BooleanArray::from(expected.to_vec()),
            "{sql}"
        );
    }

    /// A constant expression that fails, such as a division by zero, still
    /// fails instead of being evaluated in the column type.
    #[rstest::rstest]
    #[case::constant_first("7 / 0 = f")]
    #[case::column_first("f = 7 / 0")]
    #[case::integer_column("i = 7 / 0")]
    #[case::remainder("i = 7 % 0")]
    fn constant_expressions_keep_their_errors(#[case] sql: &str) {
        let error = evaluate_filter(sql).expect_err(sql);
        assert!(error.contains("Divide by zero"), "{sql}: {error}");
    }

    /// A folded constant that fits the column type is still narrowed to it, so
    /// the comparison stays on the bare column and a scalar index applies.
    #[rstest::rstest]
    #[case::sum("i = 25 * 2", col("i").eq(lit(50_i8)))]
    #[case::reversed("25 * 2 = i", lit(50_i8).eq(col("i")))]
    #[case::in_list("i IN (25 * 2)", col("i").in_list(vec![lit(50_i8)], false))]
    fn constant_expressions_plan_on_column(#[case] sql: &str, #[case] expected: Expr) {
        let batch = constant_expression_batch();
        let planner = crate::planner::Planner::new(batch.schema());
        assert_eq!(planner.parse_filter(sql).unwrap(), expected, "{sql}");
    }

    /// A shift whose exact result a `UInt64` column holds compares with that
    /// result on the bare column, although it wraps as `Int64`.
    #[rstest::rstest]
    #[case::column_first("w = (1 << 63)", col("w").eq(lit(1_u64 << 63)))]
    #[case::constant_first("(1 << 63) = w", lit(1_u64 << 63).eq(col("w")))]
    fn exact_shift_plans_on_unsigned_column(#[case] sql: &str, #[case] expected: Expr) {
        let batch = arrow_array::record_batch!(("w", UInt64, [1_u64 << 63, 3])).unwrap();
        let planner = crate::planner::Planner::new(batch.schema());
        assert_eq!(planner.parse_filter(sql).unwrap(), expected, "{sql}");
    }

    /// Every integer operator a constant expression may hold keeps its exact
    /// value before the result is narrowed, so `(1 << 63) / 2` compares with
    /// 2^62 on a `UInt64` column rather than with a negative `Int64`.
    #[rstest::rstest]
    #[case::divide("w >= (1 << 63) / 2", [Some(false), Some(false), Some(true), Some(true), None])]
    #[case::divide_reversed("(1 << 63) / 2 <= w", [Some(false), Some(false), Some(true), Some(true), None])]
    #[case::right_shift("w >= (1 << 63) >> 1", [Some(false), Some(false), Some(true), Some(true), None])]
    #[case::remainder(
        "w = (1 << 63) % 9223372036854775807",
        [Some(false), Some(true), Some(false), Some(false), None]
    )]
    #[case::between(
        "w BETWEEN (1 << 63) >> 1 AND (1 << 63) / 1",
        [Some(false), Some(false), Some(true), Some(true), None]
    )]
    fn exact_integer_arithmetic_on_unsigned_column(
        #[case] sql: &str,
        #[case] expected: [Option<bool>; 5],
    ) {
        use arrow_array::{BooleanArray, cast::AsArray};

        let batch = arrow_array::record_batch!((
            "w",
            UInt64,
            [Some(0), Some(1), Some(1_u64 << 62), Some(1_u64 << 63), None]
        ))
        .unwrap();
        let planner = crate::planner::Planner::new(batch.schema());
        let expr = planner
            .optimize_expr(planner.parse_filter(sql).unwrap())
            .unwrap();
        let result = planner
            .create_physical_expr(&expr)
            .unwrap()
            .evaluate(&batch)
            .unwrap()
            .into_array(batch.num_rows())
            .unwrap();
        assert_eq!(
            result.as_boolean(),
            &BooleanArray::from(expected.to_vec()),
            "{sql}"
        );
    }

    /// Arithmetic with a constant operand is not narrowed to the column type,
    /// so a projection keeps DataFusion's promotion instead of wrapping.
    #[rstest::rstest]
    #[case::sum("(25 * 2) + i", "150")]
    #[case::product("(1 + 1) * u", "510")]
    fn constant_operands_do_not_narrow_arithmetic(#[case] sql: &str, #[case] expected: &str) {
        let batch = arrow_array::record_batch!(("i", Int8, [100]), ("u", UInt8, [255])).unwrap();
        let planner = crate::planner::Planner::new(batch.schema());
        let expr = planner
            .optimize_expr(planner.parse_expr(sql).unwrap())
            .unwrap();
        let result = planner
            .create_physical_expr(&expr)
            .unwrap()
            .evaluate(&batch)
            .unwrap()
            .into_array(1)
            .unwrap();
        let value = ScalarValue::try_from_array(result.as_ref(), 0).unwrap();
        assert_eq!(value.to_string(), expected, "{sql}: {value:?}");
    }
}
