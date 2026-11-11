// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `any_element(list, predicate)`: a predicate on the elements of a
//! `list<struct>` column, written as an ordinary scalar function call.
//!
//! ```sql
//! any_element(detections, 'label = ''car'' AND confidence > 0.9')
//! ```
//!
//! is TRUE for a row when some element of `detections` satisfies the predicate,
//! with every condition evaluated on that one element. Two separate conditions
//! on the list (`array_has(...) AND ...`) cannot express this: they may each be
//! satisfied by a different element.
//!
//! The predicate is a SQL string literal over the element struct's fields,
//! written as plain column names (`label`, not `d.label`). It is planned with
//! Lance's filter planner against a schema made of those fields, so it accepts
//! what a Lance filter accepts, and it may not name columns outside the element.
//! Because it is a string literal, the call needs no lambda support in the SQL
//! dialect or in Substrait.
//!
//! The result follows SQL three-valued logic, like
//! `array_any_match(list, x -> predicate)`:
//!
//! - NULL when the list is NULL;
//! - TRUE when the predicate is TRUE on at least one element;
//! - otherwise NULL when it is NULL on at least one element (a NULL element
//!   reads as NULL in every field);
//! - otherwise FALSE, which includes the empty list.
//!
//! The predicate is compiled once per (predicate, element type) and cached for
//! the process, so it may only call immutable functions: a call to `now()` or
//! `random()` is rejected when the query is planned.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use arrow::compute::{is_null, nullif};
use arrow_array::builder::BooleanBuilder;
use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BooleanArray, GenericListArray, OffsetSizeTrait, RecordBatch,
    RecordBatchOptions, StructArray,
};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion::physical_plan::PhysicalExpr;
use datafusion::prelude::Expr;
use datafusion_common::{DataFusionError, Result as DFResult, ScalarValue, exec_err, plan_err};
use lance_core::{Error, Result};

use crate::planner::Planner;

mod lambda;

pub use lambda::rewrite_element_lambdas;

const ANY_ELEMENT: &str = "any_element";

/// Compiled predicates kept per process; the cache is cleared when it is full.
const MAX_COMPILED: usize = 1024;

/// The fields of the struct elements of `list_type` (`List` or `LargeList` of
/// `Struct`), as the schema an `any_element` predicate is planned against.
///
/// Every field is nullable there, because a NULL element reads as NULL in each
/// of its fields.
fn element_schema(list_type: &DataType) -> Result<SchemaRef> {
    let element = match list_type {
        DataType::List(field) | DataType::LargeList(field) => field.data_type(),
        other => {
            return Err(Error::invalid_input(format!(
                "{ANY_ELEMENT} needs a list<struct> column, got {other}"
            )));
        }
    };
    let DataType::Struct(fields) = element else {
        return Err(Error::invalid_input(format!(
            "{ANY_ELEMENT} needs a list<struct> column, got a list of {element}"
        )));
    };
    Ok(Arc::new(Schema::new(
        fields
            .iter()
            .map(|field| field.as_ref().clone().with_nullable(true))
            .collect::<Vec<Field>>(),
    )))
}

/// Parse and optimize an `any_element` predicate over the elements of
/// `list_type`, the way Lance plans a filter.
///
/// Compiled predicates are reused across queries, and optimizing folds a stable
/// function such as `now()` into a constant. So the predicate is checked for
/// functions that are not immutable before it is optimized.
fn parse_element_predicate(planner: &Planner, predicate: &str) -> Result<Expr> {
    let cannot_plan = |err: Error| {
        Error::invalid_input(format!(
            "{ANY_ELEMENT}: cannot plan the predicate '{predicate}' over the element fields: {err}"
        ))
    };
    let expr = planner.parse_filter(predicate).map_err(cannot_plan)?;
    if let Some(function) = first_non_immutable_function(&expr)? {
        return Err(Error::invalid_input(format!(
            "{ANY_ELEMENT}: the predicate '{predicate}' calls {function}(), which is not \
             immutable; {ANY_ELEMENT} predicates may only call immutable functions"
        )));
    }
    planner.optimize_expr(expr).map_err(cannot_plan)
}

/// The first function `expr` calls whose result can change between calls with
/// the same arguments, such as `now()` or `random()`.
fn first_non_immutable_function(expr: &Expr) -> Result<Option<String>> {
    let mut found = None;
    expr.apply(|node| {
        if let Expr::ScalarFunction(call) = node
            && call.func.signature().volatility != Volatility::Immutable
        {
            found = Some(call.func.name().to_string());
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

fn predicate_str(value: &ScalarValue) -> Option<&str> {
    match value {
        ScalarValue::Utf8(Some(predicate))
        | ScalarValue::LargeUtf8(Some(predicate))
        | ScalarValue::Utf8View(Some(predicate)) => Some(predicate),
        _ => None,
    }
}

type Compiled = (SchemaRef, Arc<dyn PhysicalExpr>);

static COMPILED: LazyLock<Mutex<HashMap<(String, DataType), Compiled>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn compile(predicate: &str, list_type: &DataType) -> Result<Compiled> {
    let key = (predicate.to_string(), list_type.clone());
    // The cache only holds finished entries, so a panic while it was locked
    // cannot leave it inconsistent.
    if let Some(compiled) = COMPILED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return Ok(compiled.clone());
    }
    let schema = element_schema(list_type)?;
    let planner = Planner::new(schema.clone());
    let expr = parse_element_predicate(&planner, predicate)?;
    let physical = planner.create_physical_expr(&expr)?;
    let data_type = physical.data_type(schema.as_ref())?;
    if data_type != DataType::Boolean {
        return Err(Error::invalid_input(format!(
            "{ANY_ELEMENT}: the predicate '{predicate}' is {data_type}, not boolean"
        )));
    }
    let compiled = (schema, physical);
    let mut cache = COMPILED.lock().unwrap_or_else(PoisonError::into_inner);
    if cache.len() >= MAX_COMPILED {
        cache.clear();
    }
    cache.insert(key, compiled.clone());
    Ok(compiled)
}

/// `array` with `null_mask` (TRUE where NULL) applied, and every struct's nulls
/// pushed down into its children, so a field read below a NULL struct is NULL.
///
/// Arrow allows any values under a NULL slot, and DataFusion's `get_field`
/// returns a struct's child as stored, without the struct's own nulls.
fn with_nulls_pushed_down(
    array: &ArrayRef,
    null_mask: Option<&BooleanArray>,
) -> DFResult<ArrayRef> {
    let array = match null_mask {
        Some(mask) => nullif(array, mask)?,
        None => array.clone(),
    };
    let DataType::Struct(fields) = array.data_type() else {
        return Ok(array);
    };
    if fields.is_empty() {
        return Ok(array);
    }
    let parent = array.as_struct();
    let child_mask = parent
        .nulls()
        .filter(|nulls| nulls.null_count() > 0)
        .map(|nulls| BooleanArray::new(!nulls.inner(), None));
    let children = parent
        .columns()
        .iter()
        .map(|child| with_nulls_pushed_down(child, child_mask.as_ref()))
        .collect::<DFResult<Vec<ArrayRef>>>()?;
    Ok(Arc::new(StructArray::try_new(
        fields.clone(),
        children,
        parent.nulls().cloned(),
    )?))
}

/// Evaluate `predicate` on every element of `list` and fold the element
/// results into one value per list, in SQL three-valued logic.
fn any_element<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
    schema: &SchemaRef,
    predicate: &Arc<dyn PhysicalExpr>,
) -> DFResult<BooleanArray> {
    let offsets = list.value_offsets();
    let start = offsets[0].as_usize();
    let end = offsets[list.len()].as_usize();
    let elements = list.values().slice(start, end - start);
    let elements = elements.as_struct();
    // A NULL element reads as NULL in every field, at every nesting level,
    // whatever its child slots hold.
    let element_is_null = (elements.null_count() > 0)
        .then(|| is_null(elements))
        .transpose()?;
    let columns = elements
        .columns()
        .iter()
        .map(|column| with_nulls_pushed_down(column, element_is_null.as_ref()))
        .collect::<DFResult<Vec<ArrayRef>>>()?;
    let batch = RecordBatch::try_new_with_options(
        schema.clone(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(end - start)),
    )?;
    let matched = predicate.evaluate(&batch)?.into_array(batch.num_rows())?;
    let matched = matched.as_boolean();

    let mut out = BooleanBuilder::with_capacity(list.len());
    for row in 0..list.len() {
        if list.is_null(row) {
            out.append_null();
            continue;
        }
        let (first, last) = (
            offsets[row].as_usize() - start,
            offsets[row + 1].as_usize() - start,
        );
        let mut is_unknown = false;
        let mut is_match = false;
        for element in first..last {
            if matched.is_null(element) {
                is_unknown = true;
            } else if matched.value(element) {
                is_match = true;
                break;
            }
        }
        out.append_option(if is_match {
            Some(true)
        } else if is_unknown {
            None
        } else {
            Some(false)
        });
    }
    Ok(out.finish())
}

/// `any_element(list, predicate)`; see the [module documentation](self).
#[derive(Debug, PartialEq, Eq, Hash)]
struct AnyElement {
    signature: Signature,
}

impl Default for AnyElement {
    fn default() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for AnyElement {
    fn name(&self) -> &str {
        ANY_ELEMENT
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> DFResult<DataType> {
        let [list, predicate] = arg_types else {
            return plan_err!("{ANY_ELEMENT} takes (list<struct>, predicate string)");
        };
        if let Err(err) = element_schema(list) {
            return plan_err!("{err}");
        }
        if !matches!(
            predicate,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        ) {
            return plan_err!(
                "{ANY_ELEMENT}: the predicate must be a string literal, got {predicate}"
            );
        }
        Ok(DataType::Boolean)
    }

    // Planning passes the predicate literal here, so a bad predicate fails when
    // the query is planned, not in the middle of a scan. When the literal is
    // not available, `invoke_with_args` checks it instead.
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> DFResult<FieldRef> {
        let [list, predicate] = args.arg_fields else {
            return plan_err!("{ANY_ELEMENT} takes (list<struct>, predicate string)");
        };
        let return_type =
            self.return_type(&[list.data_type().clone(), predicate.data_type().clone()])?;
        if let Some(Some(literal)) = args.scalar_arguments.get(1) {
            let Some(predicate) = predicate_str(literal) else {
                return plan_err!("{ANY_ELEMENT}: the predicate must be a non-null string literal");
            };
            compile(predicate, list.data_type()).map_err(DataFusionError::from)?;
        }
        Ok(Arc::new(Field::new(ANY_ELEMENT, return_type, true)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let [list, predicate] = args.args.as_slice() else {
            return exec_err!("{ANY_ELEMENT} takes (list<struct>, predicate string)");
        };
        let ColumnarValue::Scalar(predicate) = predicate else {
            return exec_err!("{ANY_ELEMENT}: the predicate must be a non-null string literal");
        };
        let Some(predicate) = predicate_str(predicate) else {
            return exec_err!("{ANY_ELEMENT}: the predicate must be a non-null string literal");
        };
        let list = list.to_array(args.number_rows)?;
        let (schema, physical) =
            compile(predicate, list.data_type()).map_err(DataFusionError::from)?;
        let result = match list.data_type() {
            DataType::List(_) => any_element(list.as_list::<i32>(), &schema, &physical)?,
            DataType::LargeList(_) => any_element(list.as_list::<i64>(), &schema, &physical)?,
            other => return exec_err!("{ANY_ELEMENT} needs a list<struct> column, got {other}"),
        };
        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

/// The `any_element` function, registered by [`super::register_functions`].
pub fn any_element_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(AnyElement::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::compute::cast;
    use arrow_array::{Float32Array, Int32Array, ListArray, StringArray};
    use arrow_buffer::{NullBuffer, OffsetBuffer};
    use arrow_schema::Fields;
    use rstest::rstest;

    /// label, confidence, area
    type Det = (Option<&'static str>, Option<f32>, Option<f32>);

    fn fields() -> Fields {
        Fields::from(vec![
            Field::new("label", DataType::Utf8, true),
            Field::new("confidence", DataType::Float32, true),
            Field::new("area", DataType::Float32, true),
        ])
    }

    fn list_type(is_large: bool) -> DataType {
        let item = Arc::new(Field::new("item", DataType::Struct(fields()), true));
        if is_large {
            DataType::LargeList(item)
        } else {
            DataType::List(item)
        }
    }

    fn schema(is_large: bool) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("width", DataType::Int32, false),
            Field::new("detections", list_type(is_large), true),
        ]))
    }

    /// A `None` list is a null list. A `None` element is a null detection whose
    /// child slots hold a matching car, since Arrow allows any values under a
    /// null slot.
    fn batch(rows: &[(i32, Option<Vec<Option<Det>>>)], is_large: bool) -> RecordBatch {
        let (mut labels, mut confidences, mut areas) = (vec![], vec![], vec![]);
        let (mut valid, mut lengths, mut list_valid, mut widths) = (vec![], vec![], vec![], vec![]);
        for (width, dets) in rows {
            widths.push(*width);
            list_valid.push(dets.is_some());
            let dets = dets.clone().unwrap_or_default();
            lengths.push(dets.len());
            for det in dets {
                valid.push(det.is_some());
                let (l, c, a) = det.unwrap_or((Some("car"), Some(0.99), Some(99999.0)));
                labels.push(l);
                confidences.push(c);
                areas.push(a);
            }
        }
        let elements = StructArray::try_new(
            fields(),
            vec![
                Arc::new(StringArray::from(labels)) as ArrayRef,
                Arc::new(Float32Array::from(confidences)),
                Arc::new(Float32Array::from(areas)),
            ],
            Some(NullBuffer::from(valid)),
        )
        .unwrap();
        let list: ArrayRef = Arc::new(
            ListArray::try_new(
                Arc::new(Field::new("item", DataType::Struct(fields()), true)),
                OffsetBuffer::from_lengths(lengths),
                Arc::new(elements),
                Some(NullBuffer::from(list_valid)),
            )
            .unwrap(),
        );
        let list = cast(&list, &list_type(is_large)).unwrap();
        RecordBatch::try_new(
            schema(is_large),
            vec![Arc::new(Int32Array::from(widths)), list],
        )
        .unwrap()
    }

    fn car(c: f32, a: f32) -> Option<Det> {
        Some((Some("car"), Some(c), Some(a)))
    }

    fn person(c: f32, a: f32) -> Option<Det> {
        Some((Some("person"), Some(c), Some(a)))
    }

    fn rows() -> Vec<(i32, Option<Vec<Option<Det>>>)> {
        vec![
            // 0: the trap: a low-confidence car next to a high-confidence person.
            (1280, Some(vec![car(0.30, 20000.0), person(0.95, 5000.0)])),
            // 1: a matching car.
            (1280, Some(vec![car(0.95, 500.0)])),
            // 2: empty list.
            (3840, Some(vec![])),
            // 3: null list.
            (3840, None),
            // 4: a null detection next to a matching car.
            (3840, Some(vec![None, car(0.95, 20000.0)])),
            // 5: only a null detection (its hidden children would match).
            (1280, Some(vec![None])),
            // 6: a car with a null confidence.
            (1280, Some(vec![Some((Some("car"), None, Some(20000.0)))])),
            // 7: confidence exactly at the boundary.
            (3840, Some(vec![car(0.90, 20000.0)])),
            // 8: the area holds on another detection.
            (2560, Some(vec![car(0.95, 9999.0), person(0.10, 20000.0)])),
        ]
    }

    fn try_eval(filter: &str, is_large: bool) -> Result<BooleanArray> {
        let planner = Planner::new(schema(is_large));
        let expr = planner.optimize_expr(planner.parse_filter(filter)?)?;
        let physical = planner.create_physical_expr(&expr)?;
        let out = physical
            .evaluate(&batch(&rows(), is_large))?
            .into_array(rows().len())?;
        Ok(out.as_boolean().clone())
    }

    fn eval(filter: &str, is_large: bool) -> BooleanArray {
        try_eval(filter, is_large).unwrap()
    }

    const T: Option<bool> = Some(true);
    const F: Option<bool> = Some(false);
    const N: Option<bool> = None;

    const Q1: &str = "any_element(detections, 'label = ''car'' AND confidence > 0.9')";

    #[rstest]
    #[case::list(false)]
    #[case::large_list(true)]
    fn test_same_element_semantics_and_nulls(#[case] is_large: bool) {
        assert_eq!(
            eval(Q1, is_large),
            BooleanArray::from(vec![F, T, F, N, T, N, N, F, T])
        );
        assert_eq!(
            eval(&format!("NOT {Q1}"), is_large),
            BooleanArray::from(vec![T, F, T, N, F, N, N, T, F])
        );
        // A third condition binds to the same element as the others.
        assert_eq!(
            eval(
                "any_element(detections, 'label = ''car'' AND confidence > 0.9 AND area > 10000')",
                is_large
            ),
            BooleanArray::from(vec![F, F, F, N, T, N, N, F, F])
        );
        // NULL AND FALSE is FALSE: rows 5 and 6 are narrow.
        assert_eq!(
            eval(&format!("{Q1} AND width > 1920"), is_large),
            BooleanArray::from(vec![F, F, F, N, T, F, F, F, T])
        );
    }

    /// The lambda form plans to the same `any_element` call, so it gets the
    /// same answers, literal handling and Substrait encoding.
    #[rstest]
    #[case::same_element(
        "array_any_match(detections, d -> d.label = 'car' AND d.confidence > 0.9)",
        Q1
    )]
    #[case::inclusive_float_bound(
        "array_any_match(detections, d -> d.label = 'car' AND d.confidence >= 0.9)",
        "any_element(detections, 'label = ''car'' AND confidence >= 0.9')"
    )]
    #[case::alias_and_null_test(
        "any_match(detections, (x) -> x.confidence IS NULL)",
        "any_element(detections, 'confidence IS NULL')"
    )]
    #[case::negated_with_row_column(
        "NOT array_any_match(detections, d -> d['label'] IN ('person', 'truck')) AND width > 1920",
        "NOT any_element(detections, '`label` IN (''person'', ''truck'')') AND width > 1920"
    )]
    fn test_lambda_form_is_any_element(#[case] lambda: &str, #[case] call: &str) {
        let planner = Planner::new(schema(false));
        assert_eq!(
            planner.parse_filter(lambda).unwrap(),
            planner.parse_filter(call).unwrap()
        );
        assert_eq!(eval(lambda, false), eval(call, false));
    }

    /// A deep chain below `NOT` is balanced before the recursive visitors
    /// reach it, and its lambda calls are still rewritten: if they were not,
    /// the filter could not be parsed.
    #[test]
    fn test_deep_filter_with_lambda_calls() {
        let lambda = "array_any_match(detections, d -> d.label = 'car' AND d.confidence > 0.9)";
        let chain = std::iter::repeat_n(lambda, 1000)
            .collect::<Vec<_>>()
            .join(" OR ");
        assert_eq!(
            eval(&format!("NOT ({chain})"), false),
            BooleanArray::from(vec![T, F, T, N, F, N, N, T, F])
        );
    }

    /// A deep predicate inside the lambda body is balanced before the body is
    /// cloned, checked and printed, all of which recurse over the AST.
    #[test]
    fn test_deep_lambda_body() {
        let predicate = std::iter::repeat_n("d.label = 'car'", 1000)
            .collect::<Vec<_>>()
            .join(" OR ");
        assert_eq!(
            eval(
                &format!("array_any_match(detections, d -> {predicate})"),
                false
            ),
            BooleanArray::from(vec![T, T, F, N, T, N, T, T, T])
        );
    }

    #[test]
    fn test_predicate_forms() {
        // >=, IN, BETWEEN, IS NULL and OR inside the element predicate.
        let cases = [
            (
                "any_element(detections, 'label = ''car'' AND confidence >= 0.9')",
                vec![F, T, F, N, T, N, N, T, T],
            ),
            (
                "any_element(detections, 'label IN (''person'', ''truck'') AND confidence > 0.9')",
                vec![T, F, F, N, N, N, F, F, F],
            ),
            (
                "any_element(detections, 'confidence BETWEEN 0.25 AND 0.35')",
                vec![T, F, F, N, N, N, N, F, F],
            ),
            (
                "any_element(detections, 'confidence IS NULL')",
                vec![F, F, F, N, T, T, T, F, F],
            ),
            (
                "any_element(detections, 'label = ''person'' OR area < 1000')",
                vec![T, T, F, N, N, N, F, F, T],
            ),
        ];
        for (filter, want) in cases {
            assert_eq!(eval(filter, false), BooleanArray::from(want), "{filter}");
        }
    }

    /// A field read below a NULL struct is NULL, whether the NULL is the
    /// element itself or a struct field of the element.
    #[test]
    fn test_nested_fields_below_a_null_struct_are_null() {
        let label = Field::new("label", DataType::Utf8, true);
        let element_fields = Fields::from(vec![Field::new(
            "details",
            DataType::Struct(Fields::from(vec![label.clone()])),
            true,
        )]);
        // Every hidden label is 'car': under a NULL element (row 0) and under a
        // NULL `details` (row 1). Rows 2 and 3 hold a car and a truck.
        let details = StructArray::try_new(
            Fields::from(vec![label]),
            vec![Arc::new(StringArray::from(vec!["car", "car", "car", "truck"])) as ArrayRef],
            Some(NullBuffer::from(vec![true, false, true, true])),
        )
        .unwrap();
        let elements = StructArray::try_new(
            element_fields.clone(),
            vec![Arc::new(details) as ArrayRef],
            Some(NullBuffer::from(vec![false, true, true, true])),
        )
        .unwrap();
        let item = Arc::new(Field::new("item", DataType::Struct(element_fields), true));
        let list = ListArray::try_new(
            item.clone(),
            OffsetBuffer::from_lengths([1, 1, 1, 1]),
            Arc::new(elements),
            None,
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "detections",
            DataType::List(item),
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(list)]).unwrap();
        let planner = Planner::new(schema);
        let eval = |filter: &str| {
            let expr = planner
                .optimize_expr(planner.parse_filter(filter).unwrap())
                .unwrap();
            let physical = planner.create_physical_expr(&expr).unwrap();
            physical
                .evaluate(&batch)
                .unwrap()
                .into_array(batch.num_rows())
                .unwrap()
                .as_boolean()
                .clone()
        };
        assert_eq!(
            eval("any_element(detections, 'details.label = ''car''')"),
            BooleanArray::from(vec![N, N, T, F])
        );
        assert_eq!(
            eval("any_element(detections, 'details.label IS NULL')"),
            BooleanArray::from(vec![T, T, F, F])
        );
        assert_eq!(
            eval("array_any_match(detections, d -> d.details.label = 'car')"),
            BooleanArray::from(vec![N, N, T, F])
        );
    }

    #[test]
    fn test_bad_predicate_fails_at_planning() {
        let planner = Planner::new(schema(false));
        let planned = planner
            .parse_filter("any_element(detections, 'width > 1')")
            .and_then(|expr| planner.optimize_expr(expr))
            .and_then(|expr| planner.create_physical_expr(&expr));
        let err = planned.expect_err("planning should reject the predicate");
        assert!(err.to_string().contains("width"), "{err}");
    }

    #[rstest]
    #[case::outside_column("any_element(detections, 'width > 1')", "width")]
    #[case::not_a_list("any_element(width, 'x > 1')", "list<struct>")]
    #[case::not_boolean("any_element(detections, 'area + 1')", "not boolean")]
    #[case::stable_function("any_element(detections, 'now() IS NOT NULL')", "calls now()")]
    #[case::volatile_function("any_element(detections, 'confidence < random()')", "calls random()")]
    fn test_predicate_errors_are_clear(#[case] filter: &str, #[case] message: &str) {
        let err = try_eval(filter, false).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains(ANY_ELEMENT) && text.contains(message),
            "{filter}: {text}"
        );
    }
}
