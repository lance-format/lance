// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Explicit Variant shredding for the current insert path.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, StructArray, new_empty_array};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Schema as ArrowSchema, SchemaRef};
use datafusion::error::DataFusionError;
use datafusion::physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use futures::TryStreamExt;
use lance_arrow::ARROW_EXT_NAME_KEY;
use lance_core::datatypes::Schema;
use lance_core::utils::tokio::spawn_cpu;
use parquet_variant::Variant;
use parquet_variant_compute::{VariantArray, VariantType, shred_variant};

use crate::{Error, Result};

struct ShreddingPlan {
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    columns: Vec<(usize, DataType)>,
}

pub(super) fn shred_stream(
    stream: SendableRecordBatchStream,
    targets: &[(String, DataType)],
    existing_schema: Option<&Schema>,
) -> Result<SendableRecordBatchStream> {
    let input_schema = stream.schema();
    let mut fields = input_schema.fields().to_vec();
    let mut columns = Vec::with_capacity(targets.len());
    for (column, target) in targets {
        let index = input_schema.index_of(column).map_err(|_| {
            Error::invalid_input(format!(
                "Variant shredding column '{column}' is not a top-level input field"
            ))
        })?;
        if columns.iter().any(|(previous, _)| *previous == index) {
            return Err(Error::invalid_input(format!(
                "Duplicate Variant shredding column '{column}'"
            )));
        }
        let field = &fields[index];
        if !field.has_valid_extension_type::<VariantType>() {
            return Err(Error::invalid_input(format!(
                "Variant shredding column '{column}' must have the arrow.parquet.variant extension"
            )));
        }
        validate_target(target).map_err(|error| {
            Error::invalid_input(format!(
                "Invalid Variant shredding target for column '{column}': {error}"
            ))
        })?;
        // An empty array resolves the output layout without consuming source data.
        let empty = new_empty_array(field.data_type());
        let output = VariantArray::try_new(empty.as_ref())
            .and_then(|array| shred_variant(&array, target))
            .map_err(|error| {
                Error::invalid_input(format!(
                    "Cannot shred Variant column '{column}' as {target}: {error}"
                ))
            })?;
        fields[index] = Arc::new(
            field
                .as_ref()
                .clone()
                .with_data_type(output.inner().data_type().clone()),
        );
        columns.push((index, target.clone()));
    }
    let output_schema = Arc::new(ArrowSchema::new_with_metadata(
        fields,
        input_schema.metadata().clone(),
    ));
    let physical_schema = Schema::try_from(output_schema.as_ref())?;
    if let Some(existing) = existing_schema {
        for (index, _) in &columns {
            let field = &physical_schema.fields[*index];
            let stored = existing
                .fields
                .iter()
                .find(|stored| stored.name == field.name);
            if !stored.is_some_and(|stored| {
                stored.data_type() == field.data_type()
                    && stored.metadata.get(ARROW_EXT_NAME_KEY)
                        == field.metadata.get(ARROW_EXT_NAME_KEY)
            }) {
                return Err(Error::invalid_input(format!(
                    "Variant shredding layout for column '{}' does not match the existing dataset; append requires the same layout (requested: {}, stored: {:?})",
                    field.name,
                    field.data_type(),
                    stored.map(|field| field.data_type())
                )));
            }
        }
    }
    let plan = Arc::new(ShreddingPlan {
        input_schema,
        output_schema: output_schema.clone(),
        columns,
    });
    let transformed = stream.and_then(move |batch| {
        let plan = plan.clone();
        async move {
            spawn_cpu(move || plan.shred_batch(batch))
                .await
                .map_err(|error| DataFusionError::External(Box::new(error)))
        }
    });
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        output_schema,
        transformed,
    )))
}

impl ShreddingPlan {
    fn shred_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        if batch.schema() != self.input_schema {
            return Err(Error::invalid_input(format!(
                "Variant shredding batch schema differs from the declared input schema: expected {:?}, got {:?}",
                self.input_schema,
                batch.schema()
            )));
        }
        let mut arrays = batch.columns().to_vec();
        for (index, target) in &self.columns {
            let column = self.input_schema.field(*index).name();
            let input = VariantArray::try_new(arrays[*index].as_ref())?;
            arrays[*index] = shred_losslessly(&input, target).map_err(|error| {
                Error::invalid_input(format!("Cannot shred Variant column '{column}': {error}"))
            })?;
        }
        Ok(RecordBatch::try_new(self.output_schema.clone(), arrays)?)
    }
}

fn validate_target(target: &DataType) -> Result<()> {
    match target {
        DataType::Struct(fields) => {
            for (index, field) in fields.iter().enumerate() {
                if fields[..index]
                    .iter()
                    .any(|previous| previous.name() == field.name())
                {
                    return Err(Error::invalid_input(format!(
                        "Duplicate target field '{}'",
                        field.name()
                    )));
                }
                validate_target(field.data_type())?;
            }
            Ok(())
        }
        DataType::List(field) | DataType::LargeList(field) => validate_target(field.data_type()),
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Float32
        | DataType::Float64
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView => Ok(()),
        _ => Err(Error::invalid_input(format!(
            "Unsupported Variant shredding target type {target}"
        ))),
    }
}

// Upstream shredding uses cast kernels (e.g. true -> 1, 1.5 -> 1). Only pass
// rows whose selected values fit without coercion. Whole-row residual fallback
// avoids implementing a second Variant codec while retaining every original value.
fn fits_without_coercion(value: &Variant<'_, '_>, target: &DataType) -> bool {
    if matches!(value, Variant::Null) {
        return true;
    }
    match (value, target) {
        (Variant::Object(object), DataType::Struct(fields)) => fields.iter().all(|field| {
            object
                .get(field.name())
                .is_none_or(|value| fits_without_coercion(&value, field.data_type()))
        }),
        (Variant::List(list), DataType::List(field) | DataType::LargeList(field)) => list
            .iter()
            .all(|value| fits_without_coercion(&value, field.data_type())),
        (Variant::BooleanTrue | Variant::BooleanFalse, DataType::Boolean)
        | (Variant::Float(_), DataType::Float32)
        | (Variant::Double(_), DataType::Float64)
        | (
            Variant::String(_) | Variant::ShortString(_),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View,
        )
        | (Variant::Binary(_), DataType::Binary | DataType::LargeBinary | DataType::BinaryView) => {
            true
        }
        (Variant::Int8(_) | Variant::Int16(_) | Variant::Int32(_) | Variant::Int64(_), target) => {
            match target {
                DataType::Int8 => value.as_int8().is_some(),
                DataType::Int16 => value.as_int16().is_some(),
                DataType::Int32 => value.as_int32().is_some(),
                DataType::Int64 => true,
                _ => false,
            }
        }
        _ => false,
    }
}

fn shred_losslessly(input: &VariantArray, target: &DataType) -> Result<ArrayRef> {
    let metadata = arrow_cast::cast(input.metadata_column(), &DataType::BinaryView)?;
    let value = arrow_cast::cast(input.value_column(), &DataType::BinaryView)?;
    let metadata_bytes = metadata.as_binary_view();
    let value_bytes = value.as_binary_view();
    let mut shreddable_rows = Vec::with_capacity(input.len());
    for row in 0..input.len() {
        if input.is_null(row) {
            shreddable_rows.push(false);
            continue;
        }
        if metadata.is_null(row) || value.is_null(row) {
            return Err(Error::invalid_input(format!(
                "Non-null Variant row {row} requires non-null metadata and value"
            )));
        }
        // Validate bytes before calling upstream accessors that assume valid input.
        let variant = Variant::try_new(metadata_bytes.value(row), value_bytes.value(row)).map_err(
            |error| Error::invalid_input(format!("Invalid Variant at row {row}: {error}")),
        )?;
        shreddable_rows.push(fits_without_coercion(&variant, target));
    }
    let shreddable_rows = BooleanArray::from(shreddable_rows);
    let masked = StructArray::try_new(
        input.inner().fields().clone(),
        input.inner().columns().to_vec(),
        Some(NullBuffer::new(shreddable_rows.values().clone())),
    )?;
    let shredded = shred_variant(&VariantArray::try_new(&masked)?, target)?;
    let residual = arrow_select::zip::zip(&shreddable_rows, shredded.value_column(), &value)?;
    let mut columns = shredded.inner().columns().to_vec();
    let value_index = shredded
        .inner()
        .fields()
        .iter()
        .position(|field| field.name() == "value")
        .ok_or_else(|| Error::invalid_input("Shredded Variant is missing its value field"))?;
    columns[value_index] = residual;
    Ok(Arc::new(StructArray::try_new(
        shredded.inner().fields().clone(),
        columns,
        input.nulls().cloned(),
    )?))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::{BinaryArray, Int64Array, RecordBatchIterator, StringArray};
    use arrow_schema::Field;
    use futures::{StreamExt, stream};
    use lance_file::version::LanceFileVersion;
    use parquet_variant_compute::{json_to_variant, unshred_variant, variant_to_json};
    use rstest::rstest;

    use super::*;
    use crate::dataset::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};

    fn variant_batch(documents: Vec<Option<&str>>) -> RecordBatch {
        let strings: ArrayRef = Arc::new(StringArray::from(documents));
        let variant = json_to_variant(&strings).unwrap();
        let mut field = variant.field("event");
        field.metadata_mut().insert("source".into(), "test".into());
        let schema = Arc::new(ArrowSchema::new_with_metadata(
            vec![Field::new("row", DataType::Int64, false), field],
            [("dataset".into(), "variant-test".into())].into(),
        ));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(0..variant.len() as i64)),
                Arc::new(variant.into_inner()),
            ],
        )
        .unwrap()
    }

    fn object_target() -> DataType {
        DataType::Struct(
            vec![
                Field::new("id", DataType::Int64, true),
                Field::new("name", DataType::Utf8View, true),
                Field::new(
                    "items",
                    DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                    true,
                ),
            ]
            .into(),
        )
    }

    fn assert_documents(batch: &RecordBatch, documents: &[Option<&str>]) {
        let stored = VariantArray::try_new(batch["event"].as_ref()).unwrap();
        let unshredded = unshred_variant(&stored).unwrap();
        let binary_type = DataType::Struct(
            vec![
                Field::new("metadata", DataType::Binary, false),
                Field::new("value", DataType::Binary, true),
            ]
            .into(),
        );
        let unshredded = arrow_cast::cast(unshredded.inner(), &binary_type).unwrap();
        let json = variant_to_json(&unshredded).unwrap();
        assert_eq!(json.len(), documents.len());
        for (row, (actual, expected)) in json.iter().zip(documents).enumerate() {
            assert_eq!(
                actual.map(|s| serde_json::from_str::<serde_json::Value>(s).unwrap()),
                expected.map(|s| serde_json::from_str::<serde_json::Value>(s).unwrap()),
                "row {row}",
            );
        }
    }

    #[rstest]
    #[case::unshredded(false)]
    #[case::shredded(true)]
    #[tokio::test]
    async fn variant_query_residuals(#[case] is_shredded: bool) {
        let documents = vec![
            Some(r#"{"id":42,"name":"typed","items":[5,null],"extra":{"x":7}}"#),
            // This entire object stays in the root residual: id does not fit int64.
            Some(r#"{"id":1.5,"name":"residual","items":[6],"extra":{"x":8}}"#),
            Some(r#"{"id":null,"items":[]}"#),
            Some("{}"),
            Some("null"),
            None,
            Some(r#"{"items":[false,2]}"#),
            Some(r#"{"id":43,"name":"other","items":[9]}"#),
        ];
        let batch = variant_batch(documents.clone());
        let params = WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        };
        let builder = InsertBuilder::new("memory://").with_params(&params);
        let builder = if is_shredded {
            builder.with_variant_shredding("event", object_target())
        } else {
            builder
        };
        let dataset = builder.execute(vec![batch]).await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 4);
        let result = dataset
            .scan()
            .project_with_transform(&[
                ("id", "variant_get_float(event, 'id')"),
                ("name", "variant_get_string(event, 'name')"),
                ("extra", "variant_get_int(event, 'extra.x')"),
                ("second", "variant_to_json(variant_get(event, 'items[1]'))"),
                ("raw", "variant_to_json(event)"),
                ("event", "variant_get(event, '')"),
            ])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            result["id"].as_ref(),
            &arrow_array::Float64Array::from(vec![
                Some(42.0),
                Some(1.5),
                None,
                None,
                None,
                None,
                None,
                Some(43.0)
            ])
        );
        assert_eq!(
            result["name"].as_ref(),
            &StringArray::from(vec![
                Some("typed"),
                Some("residual"),
                None,
                None,
                None,
                None,
                None,
                Some("other")
            ])
        );
        assert_eq!(
            result["extra"].as_ref(),
            &Int64Array::from(vec![Some(7), Some(8), None, None, None, None, None, None])
        );
        assert_eq!(
            result["second"].as_ref(),
            &StringArray::from(vec![
                Some("null"),
                None,
                None,
                None,
                None,
                None,
                Some("2"),
                None
            ])
        );
        assert_documents(&result, &documents);
        for (actual, expected) in result["raw"].as_string::<i32>().iter().zip(&documents) {
            assert_eq!(
                actual.map(|v| serde_json::from_str::<serde_json::Value>(v).unwrap()),
                expected.map(|v| serde_json::from_str::<serde_json::Value>(v).unwrap()),
            );
        }
        let filtered = dataset
            .scan()
            .filter("variant_get_string(event, 'name') = 'residual'")
            .unwrap()
            .project(&["row"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(filtered["row"].as_ref(), &Int64Array::from(vec![1]));
        let results = dataset
            .sql("SELECT SUM(variant_get_float(event, 'id')) AS total FROM events")
            .table_name("events")
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        assert_eq!(
            results[0]["total"].as_ref(),
            &arrow_array::Float64Array::from(vec![86.5])
        );
    }

    #[rstest]
    #[case::batches(false, false)]
    #[case::stream(true, false)]
    #[case::uncommitted_batches(false, true)]
    #[case::uncommitted_stream(true, true)]
    #[tokio::test]
    async fn variant_insert_roundtrip(#[case] is_stream: bool, #[case] is_uncommitted: bool) {
        let documents = vec![
            None,
            Some("null"),
            Some("{}"),
            Some(
                r#"{"id":42,"name":"a long view string that is not inline","items":[1,null,3],"other":{"keep":true}}"#,
            ),
            Some(r#"{"id":"42","name":false,"items":["mixed",2]}"#),
            Some(r#"{"id":null,"items":[]}"#),
            Some(r#"[1,{"nested":true},null]"#),
            Some("[]"),
            Some("17"),
            Some(r#"{"id":true}"#),
            Some(r#"{"id":1.5}"#),
            Some(r#"{"items":[false,2]}"#),
        ];
        let batch = variant_batch(documents.clone());
        let params = WriteParams {
            max_rows_per_file: 3,
            max_rows_per_group: 3,
            ..Default::default()
        };
        let dataset = InsertBuilder::new("memory://")
            .with_params(&params)
            .with_variant_shredding("event", object_target())
            .execute(vec![batch.slice(0, 3), batch.slice(3, 3)])
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        let dataset = Arc::new(dataset);
        let params = WriteParams {
            mode: WriteMode::Append,
            ..params
        };
        let builder = InsertBuilder::new(dataset.clone())
            .with_params(&params)
            .with_variant_shredding("event", object_target());
        let batches = vec![batch.slice(6, 3), batch.slice(9, 3)];
        let dataset = if is_uncommitted {
            let transaction = if is_stream {
                builder
                    .execute_uncommitted_stream(RecordBatchIterator::new(
                        batches.into_iter().map(Ok),
                        batch.schema(),
                    ))
                    .await
                    .unwrap()
            } else {
                builder.execute_uncommitted(batches).await.unwrap()
            };
            assert_eq!(dataset.count_rows(None).await.unwrap(), 6);
            CommitBuilder::new(dataset)
                .execute(transaction)
                .await
                .unwrap()
        } else if is_stream {
            builder
                .execute_stream(RecordBatchIterator::new(
                    batches.into_iter().map(Ok),
                    batch.schema(),
                ))
                .await
                .unwrap()
        } else {
            builder.execute(batches).await.unwrap()
        };
        assert_eq!(dataset.get_fragments().len(), 4);
        let result = dataset.scan().try_into_batch().await.unwrap();
        assert_documents(&result, &documents);
        assert_eq!(result["row"].as_ref(), batch["row"].as_ref());
        assert_eq!(
            result.schema().field(1).metadata(),
            batch.schema().field(1).metadata()
        );
        assert_eq!(
            dataset.schema().metadata.get("dataset").unwrap(),
            "variant-test"
        );
        let stored = VariantArray::try_new(result["event"].as_ref()).unwrap();
        assert!(stored.is_null(0));
        assert!(stored.is_valid(1));
        let projected = dataset
            .scan()
            .project(&["event.typed_value.id.typed_value"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let mut expected = vec![None; documents.len()];
        expected[3] = Some(42);
        assert_eq!(projected.column(0).as_ref(), &Int64Array::from(expected));
    }

    #[rstest]
    #[case::integer(DataType::Int8, vec![Some("127"), Some("128"), Some("-129"), Some("true"), Some("1.5"), Some(r#""42""#)])]
    #[case::float(DataType::Float32, vec![Some("16777217.25"), Some("2.5"), Some("16777217"), Some("false")])]
    #[case::nested(DataType::Struct(vec![Field::new("nested", object_target(), true)].into()), vec![Some(r#"{"nested":{"id":1.5}}"#), Some(r#"{"nested":{"id":2},"extra":3}"#)])]
    #[case::list(DataType::List(Arc::new(Field::new("item", DataType::Int8, true))), vec![Some("[1,null,2]"), Some("[true,2]"), Some("[128]"), Some("[]")])]
    #[case::large_list(DataType::LargeList(Arc::new(Field::new("item", DataType::Utf8View, true))), vec![Some(r#"["hello",null]"#), Some("[1]"), Some("[]")])]
    #[tokio::test]
    async fn variant_insert_lossless_types(
        #[case] target: DataType,
        #[case] documents: Vec<Option<&str>>,
    ) {
        let dataset = InsertBuilder::new("memory://")
            .with_variant_shredding("event", target)
            .with_params(&WriteParams {
                max_rows_per_file: 1,
                max_rows_per_group: 1,
                ..Default::default()
            })
            .execute(vec![variant_batch(documents.clone())])
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), documents.len());
        assert_documents(&dataset.scan().try_into_batch().await.unwrap(), &documents);
    }

    #[rstest]
    #[case::float32(DataType::Float32, Variant::Float(1.25))]
    #[case::float64(DataType::Float64, Variant::Double(-0.0))]
    #[case::binary(DataType::Binary, Variant::Binary(b"\x00\xff"))]
    #[case::large_binary(DataType::LargeBinary, Variant::Binary(b"\x00\xff"))]
    #[case::binary_view(DataType::BinaryView, Variant::Binary(b"\x00\xff"))]
    #[tokio::test]
    async fn variant_insert_native_values(
        #[case] target: DataType,
        #[case] value: Variant<'static, 'static>,
    ) {
        let mut builder = parquet_variant_compute::VariantArrayBuilder::new(3);
        builder.append_variant(value.clone());
        builder.append_variant(Variant::BooleanTrue);
        builder.append_null();
        let variant = builder.build();
        let schema = Arc::new(ArrowSchema::new(vec![variant.field("event")]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(variant.into_inner())]).unwrap();
        let dataset = InsertBuilder::new("memory://")
            .with_variant_shredding("event", target)
            .execute(vec![batch])
            .await
            .unwrap();
        let result = dataset.scan().try_into_batch().await.unwrap();
        let stored = VariantArray::try_new(result["event"].as_ref()).unwrap();
        let typed = stored.typed_value_column().unwrap();
        assert!(typed.is_valid(0));
        assert!(typed.is_null(1));
        let restored = unshred_variant(&stored).unwrap();
        assert_eq!(restored.try_value(0).unwrap(), value);
        assert_eq!(restored.try_value(1).unwrap(), Variant::BooleanTrue);
        assert!(restored.is_null(2));
    }

    #[rstest]
    #[case::missing("missing", DataType::Int64, false, "not a top-level input field")]
    #[case::non_variant("row", DataType::Int64, false, "arrow.parquet.variant")]
    #[case::unsupported(
        "event",
        DataType::UInt64,
        false,
        "Unsupported Variant shredding target type UInt64"
    )]
    #[case::duplicate("event", DataType::Int64, true, "Duplicate Variant shredding column")]
    #[case::duplicate_fields("event", DataType::Struct(vec![Field::new("id", DataType::Int64, true), Field::new("id", DataType::Utf8, true)].into()), false, "Duplicate target field 'id'")]
    #[tokio::test]
    async fn variant_insert_invalid_options(
        #[case] column: &str,
        #[case] target: DataType,
        #[case] duplicate: bool,
        #[case] message: &str,
    ) {
        let mut builder =
            InsertBuilder::new("memory://").with_variant_shredding(column, target.clone());
        if duplicate {
            builder = builder.with_variant_shredding(column, target);
        }
        let error = builder
            .execute(vec![variant_batch(vec![Some("1")])])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(message), "{error}");
    }

    #[rstest]
    #[case::different_type(DataType::Int64)]
    #[case::missing_selected_field(DataType::Struct(vec![Field::new("id", DataType::Int64, true)].into()))]
    #[tokio::test]
    async fn variant_insert_rejects_append_layout(#[case] target: DataType) {
        let batch = variant_batch(vec![Some(r#"{"id":1}"#)]);
        let dataset = Arc::new(
            InsertBuilder::new("memory://")
                .with_variant_shredding("event", object_target())
                .execute(vec![batch.clone()])
                .await
                .unwrap(),
        );
        let error = InsertBuilder::new(dataset.clone())
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .with_variant_shredding("event", target)
            .execute(vec![batch])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("append requires the same layout"),
            "{error}"
        );
        assert_eq!(dataset.count_rows(None).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn variant_insert_rejects_already_shredded_and_legacy() {
        let batch = variant_batch(vec![Some("1")]);
        let variant = VariantArray::try_new(batch["event"].as_ref()).unwrap();
        let shredded = shred_variant(&variant, &DataType::Int64).unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![shredded.field("event")]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(shredded.into_inner())]).unwrap();
        let error = InsertBuilder::new("memory://")
            .with_variant_shredding("event", DataType::Int64)
            .execute(vec![batch])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("already shredded"), "{error}");
        let error = InsertBuilder::new("memory://")
            .with_variant_shredding("event", DataType::Int64)
            .with_params(&WriteParams {
                data_storage_version: Some(LanceFileVersion::Legacy),
                ..Default::default()
            })
            .execute(vec![variant_batch(vec![Some("1")])])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("current V2 writer"), "{error}");
    }

    #[tokio::test]
    async fn variant_shredding_stream_is_lazy_and_preserves_errors() {
        let batch = variant_batch(vec![Some("1")]);
        let schema = batch.schema();
        let polls = Arc::new(AtomicUsize::new(0));
        let observed = polls.clone();
        let source = stream::iter(vec![
            Ok(batch),
            Err(DataFusionError::Execution("source failed".into())),
        ])
        .inspect(move |_| {
            observed.fetch_add(1, Ordering::SeqCst);
        });
        let source = Box::pin(RecordBatchStreamAdapter::new(schema, source));
        let mut transformed =
            shred_stream(source, &[("event".into(), DataType::Int64)], None).unwrap();
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert_eq!(transformed.try_next().await.unwrap().unwrap().num_rows(), 1);
        assert_eq!(polls.load(Ordering::SeqCst), 1);
        let error = transformed.try_next().await.unwrap_err();
        assert!(matches!(error, DataFusionError::Execution(_)));
        assert!(error.to_string().contains("source failed"));
    }

    #[rstest]
    #[case::no_batches(false)]
    #[case::empty_batch(true)]
    #[tokio::test]
    async fn variant_insert_empty(#[case] has_batch: bool) {
        let batch = variant_batch(vec![]);
        let reader =
            RecordBatchIterator::new(has_batch.then_some(Ok(batch.clone())), batch.schema());
        let dataset = InsertBuilder::new("memory://")
            .with_variant_shredding("event", object_target())
            .execute_stream(reader)
            .await
            .unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 0);
        let DataType::Struct(fields) = dataset.schema().field("event").unwrap().data_type() else {
            panic!("expected struct")
        };
        assert!(fields.iter().any(|field| field.name() == "typed_value"));
    }

    #[tokio::test]
    async fn variant_insert_multiple_columns() {
        let batch = variant_batch(vec![Some("1"), Some("2")]);
        let schema = Arc::new(ArrowSchema::new(vec![
            batch.schema().field(0).clone(),
            batch.schema().field(1).clone(),
            batch.schema().field(1).clone().with_name("other"),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                batch["row"].clone(),
                batch["event"].clone(),
                batch["event"].clone(),
            ],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://")
            .with_variant_shredding("other", DataType::Utf8)
            .with_variant_shredding("event", DataType::Int64)
            .execute(vec![batch])
            .await
            .unwrap();
        let result = dataset.scan().try_into_batch().await.unwrap();
        let event = VariantArray::try_new(result["event"].as_ref()).unwrap();
        assert_eq!(
            event.typed_value_column().unwrap().as_ref(),
            &Int64Array::from(vec![1, 2])
        );
        let other = VariantArray::try_new(result["other"].as_ref()).unwrap();
        assert_eq!(other.typed_value_column().unwrap().null_count(), 2);
        assert_eq!(other.value_column().null_count(), 0);
    }

    #[rstest]
    #[case::source_error(false)]
    #[case::schema_drift(true)]
    #[tokio::test]
    async fn variant_insert_failed_stream_does_not_commit(#[case] schema_drift: bool) {
        let batch = variant_batch(vec![Some("1")]);
        let dataset = Arc::new(
            InsertBuilder::new("memory://")
                .with_variant_shredding("event", DataType::Int64)
                .execute(vec![batch.clone()])
                .await
                .unwrap(),
        );
        let second = if schema_drift {
            let schema = Arc::new(ArrowSchema::new(vec![batch.schema().field(1).clone()]));
            Ok(RecordBatch::try_new(schema, vec![batch["event"].clone()]).unwrap())
        } else {
            Err(DataFusionError::Execution("source failed".into()))
        };
        let schema = batch.schema();
        let source: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![Ok(batch), second]),
        ));
        let error = InsertBuilder::new(dataset.clone())
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: 1,
                max_rows_per_group: 1,
                ..Default::default()
            })
            .with_variant_shredding("event", DataType::Int64)
            .execute_stream(source)
            .await
            .unwrap_err();
        if schema_drift {
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(
                error.to_string().contains("batch schema differs"),
                "{error}"
            );
        } else {
            assert!(matches!(error, Error::Execution { .. }), "{error}");
            assert!(error.to_string().contains("source failed"), "{error}");
        }
        let mut latest = dataset.as_ref().clone();
        latest.checkout_latest().await.unwrap();
        assert_eq!(latest.version_id(), dataset.version_id());
        assert_eq!(latest.count_rows(None).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn variant_insert_rejects_malformed_bytes() {
        let fields = vec![
            Field::new("metadata", DataType::Binary, false),
            Field::new("value", DataType::Binary, true),
        ];
        let array = StructArray::new(
            fields.into(),
            vec![
                Arc::new(BinaryArray::from(vec![b"".as_slice()])),
                Arc::new(BinaryArray::from(vec![b"".as_slice()])),
            ],
            None,
        );
        let field =
            Field::new("event", array.data_type().clone(), true).with_extension_type(VariantType);
        let schema = Arc::new(ArrowSchema::new(vec![field]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
        let error = InsertBuilder::new("memory://")
            .with_variant_shredding("event", DataType::Int64)
            .execute(vec![batch])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("Invalid Variant at row 0"),
            "{error}"
        );
    }
}
