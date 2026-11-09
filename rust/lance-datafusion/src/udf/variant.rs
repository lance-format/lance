// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SQL access to standard Variant values, independent of their shredding layout.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BinaryArray, StructArray, new_empty_array};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion::prelude::SessionContext;
use parquet_variant::{Variant, VariantPath};
use parquet_variant_compute::{
    GetOptions, VariantArray, VariantType, VariantValueArrayBuilder, unshred_variant, variant_get,
    variant_to_json,
};

#[derive(Debug, PartialEq, Eq, Hash)]
enum Output {
    Variant,
    Json,
    Typed(DataType),
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VariantUdf {
    name: &'static str,
    output: Output,
    signature: Signature,
}

pub(super) fn register_functions(ctx: &SessionContext) {
    for (name, output) in [
        ("variant_get", Output::Variant),
        ("variant_to_json", Output::Json),
        ("variant_get_int", Output::Typed(DataType::Int64)),
        ("variant_get_float", Output::Typed(DataType::Float64)),
        ("variant_get_string", Output::Typed(DataType::Utf8)),
        ("variant_get_bool", Output::Typed(DataType::Boolean)),
    ] {
        let arity = if output == Output::Json { 1 } else { 2 };
        ctx.register_udf(ScalarUDF::new_from_impl(VariantUdf {
            name,
            output,
            signature: Signature::any(arity, Volatility::Immutable),
        }));
    }
}

// A stable return layout lets expressions compose across different input layouts.
fn variant_field() -> Field {
    Field::new(
        "variant",
        DataType::Struct(
            vec![
                Field::new("metadata", DataType::Binary, false),
                Field::new("value", DataType::Binary, true),
            ]
            .into(),
        ),
        true,
    )
    .with_extension_type(VariantType)
}

fn literal_path(value: Option<&ScalarValue>) -> Result<VariantPath<'_>> {
    let Some(
        ScalarValue::Utf8(Some(path))
        | ScalarValue::LargeUtf8(Some(path))
        | ScalarValue::Utf8View(Some(path)),
    ) = value
    else {
        return Err(DataFusionError::Plan(
            "Variant path must be a non-null constant string".into(),
        ));
    };
    VariantPath::try_from(path.as_str())
        .map_err(|error| DataFusionError::Plan(format!("Invalid Variant path {path:?}: {error}")))
}

impl ScalarUDFImpl for VariantUdf {
    fn name(&self) -> &str {
        self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(match &self.output {
            Output::Variant => variant_field().data_type().clone(),
            Output::Json => DataType::Utf8,
            Output::Typed(data_type) => data_type.clone(),
        })
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = args.arg_fields.first().ok_or_else(|| {
            DataFusionError::Plan(format!("{} requires a Variant argument", self.name))
        })?;
        if !input.has_valid_extension_type::<VariantType>() {
            return Err(DataFusionError::Plan(format!(
                "{} requires an arrow.parquet.variant column, got {} with metadata {:?}",
                self.name,
                input.data_type(),
                input.metadata()
            )));
        }
        VariantArray::try_new(new_empty_array(input.data_type()).as_ref())?;
        if self.output != Output::Json {
            literal_path(args.scalar_arguments.get(1).copied().flatten())?;
        }
        Ok(Arc::new(if self.output == Output::Variant {
            variant_field()
        } else {
            Field::new(self.name, self.return_type(&[])?, true)
        }))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let input = args.args.first().ok_or_else(|| {
            DataFusionError::Execution(format!("{} requires a Variant argument", self.name))
        })?;
        let is_scalar = matches!(input, ColumnarValue::Scalar(_));
        let array = input
            .clone()
            .into_array(if is_scalar { 1 } else { args.number_rows })?;
        let variant = VariantArray::try_new(array.as_ref())?;
        // Ancestor residuals can contain an entire object when a selected child
        // cannot be shredded. Following typed children alone would lose those rows.
        // Reconstruct before path traversal until residual-aware pushdown is available.
        let reconstructed = arrow_cast::cast(
            unshred_variant(&variant)?.inner(),
            variant_field().data_type(),
        )?;
        let children = reconstructed.as_struct();
        let values = children.column(1).as_binary::<i32>();
        let reconstructed = if values.null_count() > 0 {
            // The Variant read rules interpret absent value/typed_value as a
            // Variant null, distinct from the parent struct's SQL null bitmap.
            let mut null_builder = VariantValueArrayBuilder::new(1);
            null_builder.append_value(Variant::Null);
            let null_value = null_builder.build()?;
            let values = BinaryArray::from_iter(
                values
                    .iter()
                    .map(|value| Some(value.unwrap_or(null_value.value(0)))),
            );
            Arc::new(StructArray::try_new(
                children.fields().clone(),
                vec![children.column(0).clone(), Arc::new(values)],
                children.nulls().cloned(),
            )?) as ArrayRef
        } else {
            reconstructed
        };
        let children = reconstructed.as_struct();
        let metadata = children.column(0).as_binary::<i32>();
        let values = children.column(1).as_binary::<i32>();
        for row in 0..reconstructed.len() {
            if reconstructed.is_null(row) {
                continue;
            }
            if metadata.is_null(row) || values.is_null(row) {
                return Err(DataFusionError::Execution(format!(
                    "{}: non-null Variant row {row} requires metadata and value",
                    self.name
                )));
            }
            // Upstream query kernels use accessors that assume valid binary input.
            Variant::try_new(metadata.value(row), values.value(row)).map_err(|error| {
                DataFusionError::Execution(format!(
                    "{}: invalid Variant at row {row}: {error}",
                    self.name
                ))
            })?;
        }
        let result: ArrayRef = if self.output == Output::Json {
            Arc::new(variant_to_json(&reconstructed)?)
        } else {
            let path_arg = match args.args.get(1) {
                Some(ColumnarValue::Scalar(value)) => Some(value),
                _ => None,
            };
            let path = literal_path(path_arg)?;
            let options =
                GetOptions::new_with_path(path).with_as_type(Some(args.return_field.clone()));
            let result = variant_get(&reconstructed, options)?;
            // Upstream Variant kernels may return binary views even when the
            // requested extension field has binary storage.
            arrow_cast::cast(&result, args.return_type())?
        };
        if is_scalar {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        } else {
            Ok(ColumnarValue::Array(result))
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::Schema;
    use parquet_variant_compute::{json_to_variant, shred_variant};
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::unshredded(false)]
    #[case::shredded(true)]
    #[tokio::test]
    async fn variant_sql_paths(#[case] is_shredded: bool) {
        let json: ArrayRef = Arc::new(StringArray::from(vec![
            Some(r#"{"id":42,"nested":{"items":[5,null]},"name":"alice","active":true}"#),
            Some(r#"{"id":"43","nested":{"items":[6]},"name":"bob","active":false}"#),
            Some(r#"{"id":null,"nested":null}"#),
            Some("{}"),
            Some("null"),
            None,
            Some(r#"{"id":[],"nested":{"items":[]}}"#),
        ]));
        let variant = json_to_variant(&json).unwrap();
        let variant = if is_shredded {
            shred_variant(
                &variant,
                &DataType::Struct(vec![Field::new("id", DataType::Int64, true)].into()),
            )
            .unwrap()
        } else {
            variant
        };
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![variant.field("event")])),
            vec![Arc::new(variant.into_inner())],
        )
        .unwrap();
        let ctx = SessionContext::new();
        register_functions(&ctx);
        ctx.register_batch("events", batch).unwrap();
        let result = ctx
            .sql(
                "SELECT variant_get_int(event, 'id') AS id,
                    variant_get_float(event, 'nested.items[0]') AS first,
                    variant_get_string(event, 'name') AS name,
                    variant_get_bool(event, 'active') AS active,
                    variant_to_json(variant_get(event, 'nested.items[1]')) AS second,
                    variant_to_json(variant_get(event, 'absent')) AS absent,
                    variant_get(event, 'id') AS raw
             FROM events",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let result = &result[0];
        assert_eq!(
            result["id"].as_ref(),
            &Int64Array::from(vec![Some(42), None, None, None, None, None, None])
        );
        assert_eq!(
            result["first"].as_ref(),
            &Float64Array::from(vec![Some(5.0), Some(6.0), None, None, None, None, None])
        );
        assert_eq!(
            result["name"].as_ref(),
            &StringArray::from(vec![
                Some("alice"),
                Some("bob"),
                None,
                None,
                None,
                None,
                None
            ])
        );
        assert_eq!(
            result["active"].as_ref(),
            &BooleanArray::from(vec![Some(true), Some(false), None, None, None, None, None])
        );
        assert_eq!(
            result["second"].as_ref(),
            &StringArray::from(vec![Some("null"), None, None, None, None, None, None])
        );
        assert_eq!(result["absent"].null_count(), 7);
        assert!(
            result
                .schema()
                .field_with_name("raw")
                .unwrap()
                .has_valid_extension_type::<VariantType>()
        );
        // Missing and SQL null produce SQL null; an explicit Variant null is valid.
        assert!(result["raw"].is_valid(2));
        assert!(result["raw"].is_null(3));
        assert!(result["raw"].is_null(5));
    }

    #[tokio::test]
    async fn variant_query_integer_conversion() {
        let strings: ArrayRef = Arc::new(StringArray::from(vec!["1.5", "true", "\"42\"", "null"]));
        let variant = json_to_variant(&strings).unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![variant.field("event")])),
            vec![Arc::new(variant.into_inner())],
        )
        .unwrap();
        let ctx = SessionContext::new();
        register_functions(&ctx);
        ctx.register_batch("events", batch).unwrap();
        let result = ctx
            .sql("SELECT variant_get_int(event, '') AS id FROM events")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            result[0]["id"].as_ref(),
            &Int64Array::from(vec![Some(1), Some(1), None, None])
        );
    }

    #[tokio::test]
    async fn variant_query_absent_value_is_variant_null() {
        let strings: ArrayRef = Arc::new(StringArray::from(vec![Some("null"), None]));
        let variant = json_to_variant(&strings).unwrap();
        let field = variant_field().with_name("event");
        let canonical = arrow_cast::cast(variant.inner(), field.data_type()).unwrap();
        let children = canonical.as_struct();
        let array = StructArray::new(
            children.fields().clone(),
            vec![
                children.column(0).clone(),
                Arc::new(BinaryArray::new_null(2)),
            ],
            children.nulls().cloned(),
        );
        let ctx = SessionContext::new();
        register_functions(&ctx);
        ctx.register_batch(
            "events",
            RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![Arc::new(array)])
                .unwrap(),
        )
        .unwrap();
        let result = ctx.sql("SELECT variant_to_json(event) AS json, variant_to_json(variant_get(event, '')) AS root FROM events")
            .await.unwrap().collect().await.unwrap();
        let expected = StringArray::from(vec![Some("null"), None]);
        assert_eq!(result[0]["json"].as_ref(), &expected);
        assert_eq!(result[0]["root"].as_ref(), &expected);
    }

    #[rstest]
    #[case::empty(false, 0)]
    #[case::sliced(false, 1)]
    #[case::scalar(true, 1)]
    #[test]
    fn variant_query_shapes(#[case] is_scalar: bool, #[case] len: usize) {
        let strings: ArrayRef = Arc::new(StringArray::from(vec![
            Some(r#"{"id":1}"#),
            Some(r#"{"id":2}"#),
            None,
        ]));
        let variant = json_to_variant(&strings).unwrap().slice(1, len);
        let field = Arc::new(variant.field("event"));
        let array: ArrayRef = Arc::new(variant.into_inner());
        let input = if is_scalar {
            ColumnarValue::Scalar(ScalarValue::try_from_array(&array, 0).unwrap())
        } else {
            ColumnarValue::Array(array)
        };
        let udf = VariantUdf {
            name: "variant_get_int",
            output: Output::Typed(DataType::Int64),
            signature: Signature::any(2, Volatility::Immutable),
        };
        let result = udf
            .invoke_with_args(ScalarFunctionArgs {
                args: vec![
                    input,
                    ColumnarValue::Scalar(ScalarValue::Utf8(Some("id".into()))),
                ],
                arg_fields: vec![field, Arc::new(Field::new("path", DataType::Utf8, false))],
                number_rows: len,
                return_field: Arc::new(Field::new("id", DataType::Int64, true)),
                config_options: Arc::default(),
            })
            .unwrap();
        assert_eq!(matches!(result, ColumnarValue::Scalar(_)), is_scalar);
        let result = result.into_array(len).unwrap();
        assert_eq!(result.as_ref(), &Int64Array::from(vec![2; len]));
    }

    #[rstest]
    #[case::json("variant_to_json(event)")]
    #[case::path("variant_get(event, '')")]
    #[tokio::test]
    async fn variant_query_invalid_binary(#[case] expression: &str) {
        let schema = Arc::new(Schema::new(vec![variant_field().with_name("event")]));
        let fields = match variant_field().data_type() {
            DataType::Struct(fields) => fields.clone(),
            _ => unreachable!(),
        };
        let array = arrow_array::StructArray::new(
            fields,
            vec![
                Arc::new(arrow_array::BinaryArray::from(vec![&[1, 0, 0][..]])),
                Arc::new(arrow_array::BinaryArray::from(vec![&[255][..]])),
            ],
            None,
        );
        let ctx = SessionContext::new();
        register_functions(&ctx);
        ctx.register_batch(
            "events",
            RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap(),
        )
        .unwrap();
        let error = ctx
            .sql(&format!("SELECT {expression} FROM events"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap_err();
        assert!(
            matches!(
                error,
                DataFusionError::Execution(_) | DataFusionError::Context(_, _)
            ),
            "{error}"
        );
        assert!(
            error.to_string().contains("invalid Variant at row 0"),
            "{error}"
        );
    }

    #[rstest]
    #[case::plain_struct("variant_get_int(plain, 'id')", "arrow.parquet.variant")]
    #[case::numeric_input("variant_to_json(1)", "arrow.parquet.variant")]
    #[case::dynamic_path("variant_get(event, path)", "constant string")]
    #[case::null_path("variant_get(event, NULL)", "constant string")]
    #[case::numeric_path("variant_get(event, 1)", "constant string")]
    #[case::invalid_path("variant_get(event, 'items[')", "Invalid Variant path")]
    #[tokio::test]
    async fn variant_sql_invalid_arguments(#[case] expression: &str, #[case] message: &str) {
        let strings: ArrayRef = Arc::new(StringArray::from(vec![r#"{"id":1}"#]));
        let variant = json_to_variant(&strings).unwrap();
        let schema = Arc::new(Schema::new(vec![
            variant.field("event"),
            Field::new("plain", variant.data_type().clone(), true),
            Field::new("path", DataType::Utf8, false),
        ]));
        let array: ArrayRef = Arc::new(variant.into_inner());
        let batch = RecordBatch::try_new(
            schema,
            vec![
                array.clone(),
                array,
                Arc::new(StringArray::from(vec!["id"])),
            ],
        )
        .unwrap();
        let ctx = SessionContext::new();
        register_functions(&ctx);
        ctx.register_batch("events", batch).unwrap();
        let error = ctx
            .sql(&format!("SELECT {expression} FROM events"))
            .await
            .unwrap_err();
        assert!(
            matches!(
                error,
                DataFusionError::Plan(_) | DataFusionError::Context(_, _)
            ),
            "{error}"
        );
        assert!(error.to_string().contains(message), "{error}");
    }
}
