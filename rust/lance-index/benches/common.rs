// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Common utilities and data generation for scalar index benchmarks.
use std::sync::Arc;

use arrow::datatypes::{Int64Type, UInt64Type};
use arrow_array::{Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::physical_plan::SendableRecordBatchStream;
use lance_datafusion::datagen::DatafusionDatagenExt;
use lance_datagen::{BatchCount, RowCount, array, gen_batch};

/// Total number of rows in the dataset
pub const TOTAL_ROWS: u64 = 1_000_000;

/// Number of unique values for low cardinality tests
pub const LOW_CARDINALITY_COUNT: usize = 100;

/// Batch size for streaming data
pub const BATCH_SIZE: u64 = 10_000;

/// Number of batches in the dataset
pub const NUM_BATCHES: u64 = TOTAL_ROWS / BATCH_SIZE;

/// Number of fragments the many-fragment dataset is spread over
#[allow(dead_code)] // only the btree bench uses the many-fragment dataset
pub const NUM_FRAGMENTS: u64 = 1_000;

/// Total rows of the many-fragment dataset. Defaults to [`TOTAL_ROWS`]; set
/// `LANCE_BTREE_BENCH_MANY_FRAG_ROWS` to scale it (e.g. down for a quick run,
/// up to make every range query touch thousands of pages).
#[allow(dead_code)] // only the btree bench uses the many-fragment dataset
pub fn many_fragment_total_rows() -> u64 {
    std::env::var("LANCE_BTREE_BENCH_MANY_FRAG_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TOTAL_ROWS)
}

/// Row address of the `i`-th value of the many-fragment dataset: consecutive
/// values round-robin across fragments, so every btree page (rows sorted by
/// value) spans all [`NUM_FRAGMENTS`] fragments.
#[allow(dead_code)] // only the btree bench uses the many-fragment dataset
pub fn many_fragment_row_addr(i: u64) -> u64 {
    ((i % NUM_FRAGMENTS) << 32) | (i / NUM_FRAGMENTS)
}

/// Generate a stream of int64 data with unique sequential values whose row
/// ids are fragment-style addresses interleaved across [`NUM_FRAGMENTS`]
/// fragments (see [`many_fragment_row_addr`]).
#[allow(dead_code)] // only the btree bench uses the many-fragment dataset
pub fn generate_int_many_fragment_stream() -> SendableRecordBatchStream {
    let total_rows = many_fragment_total_rows();
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int64, false),
        Field::new("_rowid", DataType::UInt64, false),
    ]));

    let mut batches = Vec::new();
    let mut current_row = 0u64;
    while current_row < total_rows {
        let batch_end = (current_row + BATCH_SIZE).min(total_rows);
        let values: Vec<i64> = (current_row..batch_end).map(|i| i as i64).collect();
        let row_ids: Vec<u64> = (current_row..batch_end)
            .map(many_fragment_row_addr)
            .collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(values)),
                Arc::new(UInt64Array::from(row_ids)),
            ],
        )
        .unwrap();
        batches.push(Ok(batch));
        current_row = batch_end;
    }

    let stream = futures::stream::iter(batches);
    Box::pin(datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream))
}

/// Generate a stream of int64 data with unique values (sequential)
pub fn generate_int_unique_stream() -> SendableRecordBatchStream {
    gen_batch()
        .col("value", array::step::<Int64Type>())
        .col("_rowid", array::step::<UInt64Type>())
        .into_df_stream(
            RowCount::from(BATCH_SIZE),
            BatchCount::from(NUM_BATCHES as u32),
        )
}

/// Generate sorted int64 data with low cardinality (100 unique values)
/// Each value appears 10,000 times consecutively
pub fn generate_int_low_cardinality_stream() -> SendableRecordBatchStream {
    let rows_per_value = TOTAL_ROWS / LOW_CARDINALITY_COUNT as u64;
    let mut batches = Vec::new();
    let mut current_row = 0u64;

    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int64, false),
        Field::new("_rowid", DataType::UInt64, false),
    ]));

    for value_idx in 0..LOW_CARDINALITY_COUNT {
        let value = value_idx as i64;
        let value_end_row = current_row + rows_per_value;

        while current_row < value_end_row {
            let batch_end = (current_row + BATCH_SIZE).min(value_end_row);
            let batch_size = (batch_end - current_row) as usize;

            // Manually create arrays with proper row IDs
            let values = vec![value; batch_size];
            let row_ids: Vec<u64> = (current_row..batch_end).collect();

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(values)),
                    Arc::new(UInt64Array::from(row_ids)),
                ],
            )
            .unwrap();

            batches.push(Ok(batch));
            current_row = batch_end;
        }
    }

    let stream = futures::stream::iter(batches);
    Box::pin(datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream))
}

/// Generate a stream of string data with unique values
/// Strings are zero-padded to 10 digits for proper lexicographic sorting
pub fn generate_string_unique_stream() -> SendableRecordBatchStream {
    let mut batches = Vec::new();
    let mut current_row = 0u64;

    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Utf8, false),
        Field::new("_rowid", DataType::UInt64, false),
    ]));

    while current_row < TOTAL_ROWS {
        let batch_end = (current_row + BATCH_SIZE).min(TOTAL_ROWS);

        // Generate zero-padded strings for proper lexicographic sorting
        let values: Vec<String> = (current_row..batch_end)
            .map(|i| format!("string_{:010}", i))
            .collect();
        let row_ids: Vec<u64> = (current_row..batch_end).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(values)),
                Arc::new(UInt64Array::from(row_ids)),
            ],
        )
        .unwrap();

        batches.push(Ok(batch));
        current_row = batch_end;
    }

    let stream = futures::stream::iter(batches);
    Box::pin(datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream))
}

/// Generate sorted string data with low cardinality (100 unique values)
pub fn generate_string_low_cardinality_stream() -> SendableRecordBatchStream {
    let rows_per_value = TOTAL_ROWS / LOW_CARDINALITY_COUNT as u64;
    let mut batches = Vec::new();
    let mut current_row = 0u64;

    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Utf8, false),
        Field::new("_rowid", DataType::UInt64, false),
    ]));

    for value_idx in 0..LOW_CARDINALITY_COUNT {
        let value = format!("value_{:03}", value_idx);
        let value_end_row = current_row + rows_per_value;

        while current_row < value_end_row {
            let batch_end = (current_row + BATCH_SIZE).min(value_end_row);
            let batch_size = (batch_end - current_row) as usize;

            // Manually create arrays with proper row IDs
            let values = vec![value.as_str(); batch_size];
            let row_ids: Vec<u64> = (current_row..batch_end).collect();

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(values)),
                    Arc::new(UInt64Array::from(row_ids)),
                ],
            )
            .unwrap();

            batches.push(Ok(batch));
            current_row = batch_end;
        }
    }

    let stream = futures::stream::iter(batches);
    Box::pin(datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream))
}
