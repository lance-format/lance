// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stream utilities for sorting update references before materializing wide payload columns.

use std::sync::Arc;

use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, FieldRef, Schema as ArrowSchema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{StreamExt, stream};
use lance_arrow::interleave_batches;
use lance_core::ROW_ADDR;

pub(super) fn unique_column_name(schema: &SchemaRef, base: &str) -> String {
    let contains = |name: &str| schema.fields().iter().any(|field| field.name() == name);
    if !contains(base) {
        return base.to_string();
    }
    for suffix in 1..=schema.fields().len() {
        let candidate = format!("{base}_{suffix}");
        if !contains(&candidate) {
            return candidate;
        }
    }
    unreachable!("a schema with N fields cannot contain more than N distinct candidate names")
}

pub(super) fn enumerate_stream(
    input: SendableRecordBatchStream,
    row_id_name: String,
) -> SendableRecordBatchStream {
    let input_schema = input.schema();
    let mut fields = input_schema.fields().to_vec();
    fields.push(Arc::new(Field::new(row_id_name, DataType::UInt64, false)));
    let schema = Arc::new(ArrowSchema::new_with_metadata(
        fields,
        input_schema.metadata().clone(),
    ));
    let state = EnumerateState {
        input,
        schema: schema.clone(),
        next_row_id: 0,
    };
    let output = stream::try_unfold(state, |mut state| async move {
        let Some(batch) = state.input.next().await else {
            return Ok(None);
        };
        let batch = batch?;
        let row_count = u64::try_from(batch.num_rows()).map_err(|_| {
            DataFusionError::Execution(format!(
                "RHS update batch row count {} does not fit in UInt64",
                batch.num_rows()
            ))
        })?;
        let end_row_id = state.next_row_id.checked_add(row_count).ok_or_else(|| {
            DataFusionError::Execution(format!(
                "RHS update row identifier overflowed UInt64 at {} rows",
                state.next_row_id
            ))
        })?;
        let row_ids = UInt64Array::from((state.next_row_id..end_row_id).collect::<Vec<_>>());
        let mut columns = batch.columns().to_vec();
        columns.push(Arc::new(row_ids));
        let batch = RecordBatch::try_new(state.schema.clone(), columns)?;
        state.next_row_id = end_row_id;
        Ok(Some((batch, state)))
    });
    Box::pin(RecordBatchStreamAdapter::new(schema, output))
}

pub(super) fn project_stream(
    input: SendableRecordBatchStream,
    indices: Vec<usize>,
) -> DataFusionResult<SendableRecordBatchStream> {
    let schema = Arc::new(input.schema().project(&indices)?);
    let output = input.map(move |batch| Ok(batch?.project(&indices)?));
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
}

pub(super) fn materialize_payloads(
    mappings: SendableRecordBatchStream,
    payloads: SendableRecordBatchStream,
    row_id_name: &str,
    payload_schema: SchemaRef,
) -> DataFusionResult<SendableRecordBatchStream> {
    let mapping_schema = mappings.schema();
    let address_index = mapping_schema.index_of(ROW_ADDR)?;
    let mapping_row_id_index = mapping_schema.index_of(row_id_name)?;
    validate_uint64_field(&mapping_schema, address_index, ROW_ADDR)?;
    validate_uint64_field(&mapping_schema, mapping_row_id_index, row_id_name)?;

    let payload_input_schema = payloads.schema();
    let payload_row_id_index = payload_input_schema.index_of(row_id_name)?;
    validate_uint64_field(&payload_input_schema, payload_row_id_index, row_id_name)?;
    let payload_indices = payload_schema
        .fields()
        .iter()
        .map(|field| payload_input_schema.index_of(field.name()))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let mut output_fields: Vec<FieldRef> = Vec::with_capacity(payload_schema.fields().len() + 1);
    output_fields.push(mapping_schema.fields()[address_index].clone());
    output_fields.extend(payload_schema.fields().iter().cloned());
    let output_schema = Arc::new(ArrowSchema::new_with_metadata(
        output_fields,
        payload_schema.metadata().clone(),
    ));
    let state = PayloadMaterializationState {
        mappings,
        payloads,
        output_schema: output_schema.clone(),
        address_index,
        mapping_row_id_index,
        payload_row_id_index,
        payload_indices,
        current_payload: None,
        current_payload_position: 0,
        current_payload_batch_id: 0,
        previous_mapping_row_id: None,
        previous_payload_row_id: None,
    };
    let output = stream::try_unfold(state, |mut state| async move {
        loop {
            let Some(mapping) = state.mappings.next().await else {
                return Ok(None);
            };
            let mapping = mapping?;
            if mapping.num_rows() == 0 {
                continue;
            }
            let batch = state.materialize_mapping(mapping).await?;
            return Ok(Some((batch, state)));
        }
    });
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        output_schema,
        output,
    )))
}

fn validate_uint64_field(
    schema: &SchemaRef,
    field_index: usize,
    field_name: &str,
) -> DataFusionResult<()> {
    let data_type = schema.field(field_index).data_type();
    if data_type != &DataType::UInt64 {
        return Err(DataFusionError::Execution(format!(
            "Late materialization column '{field_name}' must be UInt64, got {data_type}"
        )));
    }
    Ok(())
}

struct EnumerateState {
    input: SendableRecordBatchStream,
    schema: SchemaRef,
    next_row_id: u64,
}

struct PayloadMaterializationState {
    mappings: SendableRecordBatchStream,
    payloads: SendableRecordBatchStream,
    output_schema: SchemaRef,
    address_index: usize,
    mapping_row_id_index: usize,
    payload_row_id_index: usize,
    payload_indices: Vec<usize>,
    current_payload: Option<RecordBatch>,
    current_payload_position: usize,
    current_payload_batch_id: u64,
    previous_mapping_row_id: Option<u64>,
    previous_payload_row_id: Option<u64>,
}

impl PayloadMaterializationState {
    async fn materialize_mapping(&mut self, mapping: RecordBatch) -> DataFusionResult<RecordBatch> {
        let mapping_row_ids = mapping
            .column(self.mapping_row_id_index)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                DataFusionError::Execution(
                    "Late materialization mapping row IDs are not UInt64".to_string(),
                )
            })?;
        let mut requested_row_ids = Vec::with_capacity(mapping.num_rows());
        for row_index in 0..mapping.num_rows() {
            if mapping_row_ids.is_null(row_index) {
                return Err(DataFusionError::Execution(format!(
                    "Late materialization mapping row ID is null at row {row_index}"
                )));
            }
            let row_id = mapping_row_ids.value(row_index);
            if let Some(previous) = self.previous_mapping_row_id
                && row_id < previous
            {
                return Err(DataFusionError::Execution(format!(
                    "Late materialization mapping row IDs are out of order: {row_id} follows {previous}"
                )));
            }
            self.previous_mapping_row_id = Some(row_id);
            requested_row_ids.push(row_id);
        }

        let mut source_batches = Vec::new();
        let mut source_indices = Vec::with_capacity(requested_row_ids.len());
        let mut local_source = None;
        for requested_row_id in requested_row_ids {
            loop {
                if self.current_payload.is_none() {
                    self.load_next_payload().await?;
                }
                let Some(payload) = self.current_payload.as_ref() else {
                    return Err(DataFusionError::Execution(format!(
                        "Late materialization payload ended before RHS row ID {requested_row_id}"
                    )));
                };
                let payload_row_ids = payload
                    .column(self.payload_row_id_index)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| {
                        DataFusionError::Execution(
                            "Late materialization payload row IDs are not UInt64".to_string(),
                        )
                    })?;
                while self.current_payload_position < payload.num_rows()
                    && payload_row_ids.value(self.current_payload_position) < requested_row_id
                {
                    self.current_payload_position += 1;
                }
                if self.current_payload_position == payload.num_rows() {
                    self.current_payload = None;
                    self.current_payload_position = 0;
                    continue;
                }

                let payload_row_id = payload_row_ids.value(self.current_payload_position);
                if payload_row_id != requested_row_id {
                    return Err(DataFusionError::Execution(format!(
                        "Late materialization payload is missing RHS row ID {requested_row_id}; next row ID is {payload_row_id}"
                    )));
                }
                let source_index = match local_source {
                    Some((batch_id, source_index)) if batch_id == self.current_payload_batch_id => {
                        source_index
                    }
                    _ => {
                        source_batches.push(payload.project(&self.payload_indices)?);
                        let source_index = source_batches.len() - 1;
                        local_source = Some((self.current_payload_batch_id, source_index));
                        source_index
                    }
                };
                source_indices.push((source_index, self.current_payload_position));
                break;
            }
        }

        let payload = interleave_batches(&source_batches, &source_indices)?;
        let mut columns = Vec::with_capacity(payload.num_columns() + 1);
        columns.push(mapping.column(self.address_index).clone());
        columns.extend(payload.columns().iter().cloned());
        Ok(RecordBatch::try_new(self.output_schema.clone(), columns)?)
    }

    async fn load_next_payload(&mut self) -> DataFusionResult<()> {
        loop {
            let Some(payload) = self.payloads.next().await else {
                self.current_payload = None;
                return Ok(());
            };
            let payload = payload?;
            if payload.num_rows() == 0 {
                continue;
            }
            let row_ids = payload
                .column(self.payload_row_id_index)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    DataFusionError::Execution(
                        "Late materialization payload row IDs are not UInt64".to_string(),
                    )
                })?;
            for row_index in 0..payload.num_rows() {
                if row_ids.is_null(row_index) {
                    return Err(DataFusionError::Execution(format!(
                        "Late materialization payload row ID is null at row {row_index}"
                    )));
                }
                let row_id = row_ids.value(row_index);
                if let Some(previous) = self.previous_payload_row_id
                    && row_id <= previous
                {
                    return Err(DataFusionError::Execution(format!(
                        "Late materialization payload row IDs must be strictly increasing: {row_id} follows {previous}"
                    )));
                }
                self.previous_payload_row_id = Some(row_id);
            }
            self.current_payload_batch_id = self
                .current_payload_batch_id
                .checked_add(1)
                .ok_or_else(|| {
                    DataFusionError::Execution(
                        "Late materialization payload batch identifier overflowed UInt64"
                            .to_string(),
                    )
                })?;
            self.current_payload = Some(payload);
            self.current_payload_position = 0;
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int32Array, record_batch};
    use futures::TryStreamExt;

    use super::*;

    fn batches(schema: SchemaRef, batches: Vec<RecordBatch>) -> SendableRecordBatchStream {
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(batches.into_iter().map(Ok)),
        ))
    }

    #[test]
    fn test_unique_column_name_avoids_existing_fields() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("internal", DataType::Int32, false),
            Field::new("internal_1", DataType::Int32, false),
        ]));
        assert_eq!(unique_column_name(&schema, "internal"), "internal_2");
        assert_eq!(unique_column_name(&schema, "unused"), "unused");
    }

    #[tokio::test]
    async fn test_enumerate_and_project_stream() {
        let first = record_batch!(("key", Int32, [4, 5]), ("value", Int32, [40, 50])).unwrap();
        let second = record_batch!(("key", Int32, [6]), ("value", Int32, [60])).unwrap();
        let enumerated = enumerate_stream(
            batches(first.schema(), vec![first, second]),
            "row_id".to_string(),
        );
        let projected = project_stream(enumerated, vec![0, 2])
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(
            projected[0].column_by_name("row_id").unwrap().as_ref(),
            &UInt64Array::from(vec![0, 1])
        );
        assert_eq!(
            projected[1].column_by_name("row_id").unwrap().as_ref(),
            &UInt64Array::from(vec![2])
        );
    }

    #[tokio::test]
    async fn test_materializes_repeated_payloads_across_batches() {
        let payload_first = record_batch!(
            ("value", Int32, [Some(10), Some(11)]),
            ("row_id", UInt64, [0, 1])
        )
        .unwrap();
        let payload_second = record_batch!(
            ("value", Int32, [None, Some(30)]),
            ("row_id", UInt64, [2, 3])
        )
        .unwrap();
        let mapping_first = record_batch!(
            (ROW_ADDR, UInt64, [100, 102, 101]),
            ("row_id", UInt64, [0, 2, 2])
        )
        .unwrap();
        let mapping_second =
            record_batch!((ROW_ADDR, UInt64, [103]), ("row_id", UInt64, [3])).unwrap();
        let payload_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "value",
            DataType::Int32,
            true,
        )]));
        let output = materialize_payloads(
            batches(mapping_first.schema(), vec![mapping_first, mapping_second]),
            batches(payload_first.schema(), vec![payload_first, payload_second]),
            "row_id",
            payload_schema,
        )
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
        let output = arrow::compute::concat_batches(&output[0].schema(), &output).unwrap();
        assert_eq!(
            output.column_by_name(ROW_ADDR).unwrap().as_ref(),
            &UInt64Array::from(vec![100, 102, 101, 103])
        );
        assert_eq!(
            output.column_by_name("value").unwrap().as_ref(),
            &Int32Array::from(vec![Some(10), None, None, Some(30)])
        );
    }

    #[tokio::test]
    async fn test_rejects_missing_payload_row_id() {
        let payload = record_batch!(("value", Int32, [10]), ("row_id", UInt64, [0])).unwrap();
        let mapping = record_batch!((ROW_ADDR, UInt64, [100]), ("row_id", UInt64, [1])).unwrap();
        let payload_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let error = materialize_payloads(
            batches(mapping.schema(), vec![mapping]),
            batches(payload.schema(), vec![payload]),
            "row_id",
            payload_schema,
        )
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap_err();
        assert!(error.to_string().contains("ended before RHS row ID 1"));
    }
}
