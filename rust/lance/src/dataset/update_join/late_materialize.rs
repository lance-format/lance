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

use crate::io::exec::take::batch::array_size;

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
    max_batch_bytes: usize,
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
        max_batch_bytes,
        current_payload: None,
        current_payload_row_sizes: Vec::new(),
        current_payload_position: 0,
        current_payload_batch_id: 0,
        previous_mapping_row_id: None,
        previous_payload_row_id: None,
        current_mapping: None,
        current_mapping_position: 0,
        chunk: PayloadChunk::default(),
        pending_output: None,
    };
    let output = stream::try_unfold(state, |mut state| async move {
        match state.next_output().await? {
            Some(batch) => Ok(Some((batch, state))),
            None => Ok(None),
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
    max_batch_bytes: usize,
    current_payload: Option<RecordBatch>,
    current_payload_row_sizes: Vec<usize>,
    current_payload_position: usize,
    current_payload_batch_id: u64,
    previous_mapping_row_id: Option<u64>,
    previous_payload_row_id: Option<u64>,
    current_mapping: Option<RecordBatch>,
    current_mapping_position: usize,
    chunk: PayloadChunk,
    pending_output: Option<RecordBatch>,
}

#[derive(Default)]
struct PayloadChunk {
    mapping_start: usize,
    source_batches: Vec<RecordBatch>,
    source_indices: Vec<(usize, usize)>,
    source_batch_bytes: usize,
    output_bytes: usize,
    local_source: Option<(u64, usize)>,
}

impl PayloadChunk {
    fn is_empty(&self) -> bool {
        self.source_indices.is_empty()
    }
}

impl PayloadMaterializationState {
    async fn next_output(&mut self) -> DataFusionResult<Option<RecordBatch>> {
        loop {
            if let Some(batch) = self.pending_output.take() {
                return Ok(Some(batch));
            }

            if self.current_mapping.is_some() {
                let mapping_rows = self
                    .current_mapping
                    .as_ref()
                    .map(RecordBatch::num_rows)
                    .unwrap_or(0);
                if self.current_mapping_position < mapping_rows {
                    self.add_current_mapping_row().await?;
                    continue;
                }

                if !self.chunk.is_empty() {
                    let batch = self.flush_chunk()?;
                    self.current_mapping = None;
                    self.current_mapping_position = 0;
                    return Ok(Some(batch));
                }
                self.current_mapping = None;
                self.current_mapping_position = 0;
                continue;
            }

            let Some(mapping) = self.mappings.next().await else {
                return Ok(None);
            };
            let mapping = mapping?;
            if mapping.num_rows() == 0 {
                continue;
            }
            self.current_mapping = Some(mapping);
            self.current_mapping_position = 0;
            self.chunk = PayloadChunk::default();
        }
    }

    async fn add_current_mapping_row(&mut self) -> DataFusionResult<()> {
        let row_index = self.current_mapping_position;
        let row_id = {
            let mapping = self.current_mapping.as_ref().ok_or_else(|| {
                DataFusionError::Internal("Missing update mapping batch".to_string())
            })?;
            let mapping_row_ids = mapping
                .column(self.mapping_row_id_index)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    DataFusionError::Execution(
                        "Late materialization mapping row IDs are not UInt64".to_string(),
                    )
                })?;
            if mapping_row_ids.is_null(row_index) {
                return Err(DataFusionError::Execution(format!(
                    "Late materialization mapping row ID is null at row {row_index}"
                )));
            }
            mapping_row_ids.value(row_index)
        };
        if let Some(previous) = self.previous_mapping_row_id
            && row_id < previous
        {
            return Err(DataFusionError::Execution(format!(
                "Late materialization mapping row IDs are out of order: {row_id} follows {previous}"
            )));
        }
        self.previous_mapping_row_id = Some(row_id);

        let (payload_batch_id, payload_position, row_size) =
            self.locate_payload_row(row_id).await?;
        let output_bytes = row_size
            .checked_add(std::mem::size_of::<u64>())
            .ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "Update payload row size for RHS row ID {row_id} overflowed"
                ))
            })?;
        if output_bytes > self.max_batch_bytes {
            return Err(DataFusionError::ResourcesExhausted(format!(
                "Update payload row for RHS row ID {row_id} requires {output_bytes} bytes, \
                 exceeding the materialization batch budget of {} bytes",
                self.max_batch_bytes
            )));
        }

        let (new_source_bytes, source_index) = match self.chunk.local_source {
            Some((batch_id, source_index)) if batch_id == payload_batch_id => (0, source_index),
            _ => {
                let payload = self.current_payload.as_ref().ok_or_else(|| {
                    DataFusionError::Internal(
                        "Update payload batch disappeared during materialization".to_string(),
                    )
                })?;
                let projected = payload.project(&self.payload_indices)?;
                // A projected payload batch is the smallest retainable unit. It
                // may exceed the per-chunk source budget when it is a slice that
                // shares a larger buffer, so allow one such batch and flush
                // before retaining another.
                let bytes = projected.get_array_memory_size();
                self.chunk.source_batches.push(projected);
                let source_index = self.chunk.source_batches.len() - 1;
                self.chunk.source_batch_bytes = self.chunk.source_batch_bytes.saturating_add(bytes);
                self.chunk.local_source = Some((payload_batch_id, source_index));
                (bytes, source_index)
            }
        };

        if !self.chunk.is_empty()
            && (self.chunk.output_bytes.saturating_add(output_bytes) > self.max_batch_bytes
                || (new_source_bytes > 0
                    && self
                        .chunk
                        .source_batch_bytes
                        .saturating_add(new_source_bytes)
                        > self.max_batch_bytes))
        {
            self.pending_output = Some(self.flush_chunk()?);
            return Ok(());
        }

        self.chunk
            .source_indices
            .push((source_index, payload_position));
        self.chunk.output_bytes = self.chunk.output_bytes.saturating_add(output_bytes);
        self.current_mapping_position += 1;
        Ok(())
    }

    async fn locate_payload_row(
        &mut self,
        requested_row_id: u64,
    ) -> DataFusionResult<(u64, usize, usize)> {
        loop {
            if self.current_payload.is_none() {
                self.load_next_payload().await?;
            }
            let Some(payload) = self.current_payload.as_ref() else {
                return Err(DataFusionError::Execution(format!(
                    "Late materialization payload ended before RHS row ID {requested_row_id}"
                )));
            };
            let payload = payload.clone();
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
            let position = self.current_payload_position;
            let row_size = if self.current_payload_row_sizes[position] == 0 {
                let size = payload_row_size(&payload, position)?;
                self.current_payload_row_sizes[position] = size;
                size
            } else {
                self.current_payload_row_sizes[position]
            };
            return Ok((self.current_payload_batch_id, position, row_size));
        }
    }

    fn flush_chunk(&mut self) -> DataFusionResult<RecordBatch> {
        let mapping_end = self.current_mapping_position;
        let mapping_start = self.chunk.mapping_start;
        if mapping_start >= mapping_end {
            return Err(DataFusionError::Internal(format!(
                "Cannot materialize an empty update payload chunk: start={mapping_start}, end={mapping_end}"
            )));
        }
        let addresses = self
            .current_mapping
            .as_ref()
            .ok_or_else(|| {
                DataFusionError::Internal("Missing update mapping batch during flush".to_string())
            })?
            .column(self.address_index)
            .slice(mapping_start, mapping_end - mapping_start);
        let payload = interleave_batches(&self.chunk.source_batches, &self.chunk.source_indices)?;
        let mut columns = Vec::with_capacity(payload.num_columns() + 1);
        columns.push(addresses);
        columns.extend(payload.columns().iter().cloned());
        let batch = RecordBatch::try_new(self.output_schema.clone(), columns)?;
        self.chunk = PayloadChunk {
            mapping_start: mapping_end,
            ..Default::default()
        };
        Ok(batch)
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
            self.current_payload_row_sizes = vec![0; payload.num_rows()];
            self.current_payload = Some(payload);
            self.current_payload_position = 0;
            return Ok(());
        }
    }
}

fn payload_row_size(batch: &RecordBatch, row: usize) -> DataFusionResult<usize> {
    batch.columns().iter().try_fold(0usize, |total, column| {
        let size = array_size(&column.slice(row, 1))?;
        Ok(total.saturating_add(size.max(1)))
    })
}

#[cfg(test)]
mod tests {
    use arrow_array::{FixedSizeListArray, Int32Array, Int64Array, record_batch};
    use futures::TryStreamExt;
    use lance_core::utils::address::RowAddress;

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
            usize::MAX,
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
            usize::MAX,
        )
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap_err();
        assert!(error.to_string().contains("ended before RHS row ID 1"));
    }

    fn wide_payload_batch(row_ids: &[u64], values: i64) -> RecordBatch {
        const WIDTH: i32 = 32;
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new(
                "payload",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Int64, false)),
                    WIDTH,
                ),
                false,
            ),
            Field::new("row_id", DataType::UInt64, false),
        ]));
        let values = Int64Array::from_iter_values(std::iter::repeat_n(
            values,
            row_ids.len() * WIDTH as usize,
        ));
        let payload = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Int64, false)),
            WIDTH,
            Arc::new(values),
            None,
        )
        .unwrap();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(payload),
                Arc::new(UInt64Array::from(row_ids.to_vec())),
            ],
        )
        .unwrap()
    }

    fn mapping_batch(row_ids: &[u64]) -> RecordBatch {
        let addresses = (0..row_ids.len() as u64)
            .map(|offset| {
                u64::from(RowAddress::new_from_parts(
                    7,
                    u32::try_from(offset).unwrap(),
                ))
            })
            .collect::<Vec<_>>();
        record_batch!(
            (ROW_ADDR, UInt64, addresses),
            ("row_id", UInt64, row_ids.to_vec())
        )
        .unwrap()
    }

    fn payload_schema() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![Field::new(
            "payload",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int64, false)), 32),
            false,
        )]))
    }

    fn logical_batch_size(batch: &RecordBatch) -> usize {
        batch
            .columns()
            .iter()
            .map(|column| array_size(column.as_ref()).unwrap())
            .sum()
    }

    #[tokio::test]
    async fn test_materialization_bounds_repeated_wide_rows() {
        const MAX_BYTES: usize = 1024;
        let payload = wide_payload_batch(&[0], 7);
        let mapping = mapping_batch(&[0; 16]);
        let output = materialize_payloads(
            batches(mapping.schema(), vec![mapping]),
            batches(payload.schema(), vec![payload]),
            "row_id",
            payload_schema(),
            MAX_BYTES,
        )
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        assert!(output.len() > 1, "repeated wide payloads must be split");
        assert_eq!(output.iter().map(RecordBatch::num_rows).sum::<usize>(), 16);
        for batch in &output {
            let bytes = logical_batch_size(batch);
            assert!(
                bytes <= MAX_BYTES,
                "output batch uses {bytes} logical bytes, expected at most {MAX_BYTES}"
            );
            let addresses = batch
                .column_by_name(ROW_ADDR)
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert!(addresses.iter().all(|address| address.is_some()));
        }
    }

    #[tokio::test]
    async fn test_materialization_bounds_sparse_source_batches() {
        const MAX_BYTES: usize = 32 * 1024;
        const PAYLOAD_ROWS: usize = 100;
        const BATCHES: usize = 10;
        let payload_batches = (0..BATCHES)
            .map(|batch_index| {
                let start = (batch_index * PAYLOAD_ROWS) as u64;
                wide_payload_batch(&(start..start + PAYLOAD_ROWS as u64).collect::<Vec<_>>(), 9)
            })
            .collect::<Vec<_>>();
        let mapping = mapping_batch(
            &(0..BATCHES)
                .map(|batch_index| (batch_index * PAYLOAD_ROWS) as u64)
                .collect::<Vec<_>>(),
        );
        let output = materialize_payloads(
            batches(mapping.schema(), vec![mapping]),
            batches(payload_batches[0].schema(), payload_batches),
            "row_id",
            payload_schema(),
            MAX_BYTES,
        )
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        assert_eq!(
            output.len(),
            BATCHES,
            "each sparse source batch must be released before retaining the next"
        );
        assert!(output.iter().all(|batch| batch.num_rows() == 1));
    }

    #[tokio::test]
    async fn test_materialization_rejects_indivisible_row_over_budget() {
        const MAX_BYTES: usize = 128;
        let payload = wide_payload_batch(&[0], 11);
        let mapping = mapping_batch(&[0]);
        let error = materialize_payloads(
            batches(mapping.schema(), vec![mapping]),
            batches(payload.schema(), vec![payload]),
            "row_id",
            payload_schema(),
            MAX_BYTES,
        )
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap_err();
        assert!(error.to_string().contains("requires"));
        assert!(
            error
                .to_string()
                .contains("exceeding the materialization batch budget")
        );
    }
}
