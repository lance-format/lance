// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream};
use futures::Stream;
use lance_arrow::json::{JsonEncoding, JsonValues};
use lance_core::{Error, Result};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Presents a JSON column as JSON text documents, whichever encoding the
/// column arrives in.
pub struct JsonTextStream {
    inner: SendableRecordBatchStream,
    json_col: String,
    schema: SchemaRef,
}

impl JsonTextStream {
    /// Fails when `json_col` is missing or does not hold JSON.
    pub fn try_new(inner: SendableRecordBatchStream, json_col: String) -> Result<Self> {
        let input_schema = inner.schema();
        let fields = input_schema
            .fields()
            .iter()
            .map(|field| {
                if field.name() != &json_col {
                    return Ok(field.as_ref().clone());
                }
                match JsonEncoding::of_field(field) {
                    Some(JsonEncoding::Jsonb) => Ok(Field::new(
                        &json_col,
                        DataType::LargeUtf8,
                        field.is_nullable(),
                    )),
                    Some(JsonEncoding::Text) => Ok(field.as_ref().clone()),
                    None => Err(Error::invalid_input(format!(
                        "column {json_col} is not json"
                    ))),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inner,
            json_col,
            schema: Arc::new(Schema::new_with_metadata(
                fields,
                input_schema.metadata().clone(),
            )),
        })
    }

    fn to_text(&self, batch: RecordBatch) -> datafusion_common::Result<RecordBatch> {
        let columns = batch
            .schema()
            .fields()
            .iter()
            .zip(batch.columns())
            .map(|(field, column)| match JsonValues::try_new(field, column) {
                Some(values) if field.name() == &self.json_col => values.to_text(),
                _ => column.clone(),
            })
            .collect();
        Ok(RecordBatch::try_new(self.schema.clone(), columns)?)
    }
}

impl Stream for JsonTextStream {
    type Item = datafusion_common::Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => Poll::Ready(Some(self.to_text(batch))),
            other => other,
        }
    }
}

impl RecordBatchStream for JsonTextStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::scalar::inverted::json::JsonTextStream;
    use arrow_array::builder::UInt64Builder;
    use arrow_array::cast::AsArray;
    use arrow_array::{ArrayRef, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::execution::RecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::{TryStreamExt, stream};
    use lance_arrow::ARROW_EXT_NAME_KEY;
    use lance_arrow::json::{ARROW_JSON_EXT_NAME, JsonArray, json_field};
    use rstest::rstest;
    use serde_json::Value;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// JSON documents read as the same text whether they arrive as stored
    /// JSONB or as Arrow JSON text.
    #[rstest]
    #[case::jsonb(true)]
    #[case::arrow_json_text(false)]
    #[tokio::test]
    async fn test_json_text_stream(#[case] is_jsonb: bool) {
        let json_strings = [
            r#"{"a": 1, "b": "hello"}"#,
            r#"{"c": [1, 2, 3], "d": {"e": true}}"#,
            r#"{"f": null}"#,
        ];
        let text = StringArray::from(json_strings.to_vec());
        let (json_field, json_values): (Field, ArrayRef) = if is_jsonb {
            (
                json_field("json_col", true),
                Arc::new(JsonArray::try_from(&text).unwrap().into_inner()),
            )
        } else {
            (
                Field::new("json_col", DataType::Utf8, true).with_metadata(HashMap::from([(
                    ARROW_EXT_NAME_KEY.to_string(),
                    ARROW_JSON_EXT_NAME.to_string(),
                )])),
                Arc::new(text),
            )
        };
        let mut rowid_builder = UInt64Builder::new();
        rowid_builder.append_slice(&[0, 1, 2]);

        let schema = Arc::new(Schema::new(vec![
            json_field,
            Field::new("rowid", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![json_values, Arc::new(rowid_builder.finish()) as ArrayRef],
        )
        .unwrap();

        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::once(async { Ok(batch) }),
        ));

        let json_text_stream = JsonTextStream::try_new(stream, "json_col".to_string()).unwrap();
        let stream_schema = json_text_stream.schema();
        let result_batches: Vec<RecordBatch> = json_text_stream.try_collect().await.unwrap();
        assert_eq!(result_batches.len(), 1);
        let result_batch = &result_batches[0];
        assert_eq!(result_batch.schema(), stream_schema);

        let json_text_col = result_batch.column_by_name("json_col").unwrap();
        let json_texts: Vec<&str> = if is_jsonb {
            json_text_col.as_string::<i64>().iter().flatten().collect()
        } else {
            json_text_col.as_string::<i32>().iter().flatten().collect()
        };
        for (original_json_str, converted_json_str) in json_strings.iter().zip(json_texts) {
            let original_value: Value = serde_json::from_str(original_json_str).unwrap();
            let converted_value: Value = serde_json::from_str(converted_json_str).unwrap();
            assert_eq!(original_value, converted_value);
        }
    }

    #[test]
    fn test_json_text_stream_rejects_non_json_column() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "json_col",
            DataType::LargeBinary,
            true,
        )]));
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream::empty()));
        let error = JsonTextStream::try_new(stream, "json_col".to_string())
            .err()
            .unwrap();
        assert!(
            error.to_string().contains("json_col is not json"),
            "{error}"
        );
    }
}
