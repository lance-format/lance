// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_schema::{Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use datafusion::physical_plan::SendableRecordBatchStream as DataFusionRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter as DataFusionStreamAdapter;
use futures::StreamExt;
use lance_arrow::json::{convert_lance_json_to_arrow, has_json_fields, lance_json_to_arrow_json};
pub use lance_table_runtime::rowids::{CapturedRowIds, make_rowid_capture_stream};
pub use lance_table_runtime::write::SchemaAdapter;

fn requires_logical_conversion(schema: &ArrowSchemaRef) -> bool {
    schema.fields().iter().any(|field| has_json_fields(field))
}

/// Convert a physical stream into a logical stream.
pub fn to_logical_stream(stream: DataFusionRecordBatchStream) -> DataFusionRecordBatchStream {
    if !requires_logical_conversion(&stream.schema()) {
        return stream;
    }

    let arrow_schema = stream.schema();
    let mut new_fields = Vec::with_capacity(arrow_schema.fields().len());
    for field in arrow_schema.fields() {
        if has_json_fields(field) {
            new_fields.push(lance_json_to_arrow_json(field));
        } else {
            new_fields.push(field.as_ref().clone());
        }
    }
    let converted_schema = Arc::new(ArrowSchema::new_with_metadata(
        new_fields,
        arrow_schema.metadata().clone(),
    ));

    let converted_stream = stream.map(move |batch_result| {
        batch_result.and_then(|batch| {
            convert_lance_json_to_arrow(&batch).map_err(|e| {
                datafusion::error::DataFusionError::ArrowError(
                    Box::new(arrow_schema::ArrowError::InvalidArgumentError(
                        e.to_string(),
                    )),
                    None,
                )
            })
        })
    });

    Box::pin(DataFusionStreamAdapter::new(
        converted_schema,
        converted_stream,
    ))
}
