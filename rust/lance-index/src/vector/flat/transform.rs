// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatch;
use arrow_schema::Field;
use lance_arrow::RecordBatchExt;
use lance_core::Error;
use tracing::instrument;

use crate::vector::transform::Transformer;

use super::storage::FLAT_COLUMN;

#[derive(Debug)]
pub struct FlatTransformer {
    input_column: String,
}

impl FlatTransformer {
    pub fn new(input_column: impl AsRef<str>) -> Self {
        Self {
            input_column: input_column.as_ref().to_owned(),
        }
    }
}

impl Transformer for FlatTransformer {
    #[instrument(name = "FlatTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> lance_core::Result<RecordBatch> {
        let input_arr = batch
            .column_by_name(&self.input_column)
            .ok_or(Error::index(format!(
                "FlatTransform: column {} not found in batch",
                self.input_column
            )))?;
        let field = Field::new(
            FLAT_COLUMN,
            input_arr.data_type().clone(),
            input_arr.is_nullable(),
        );
        // rename the column to FLAT_COLUMN
        let batch = batch
            .drop_column(&self.input_column)?
            .try_with_column(field, input_arr.clone())?;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow::buffer::NullBuffer;
    use arrow_array::{Array, FixedSizeListArray, Float32Array, Int32Array};
    use arrow_schema::{DataType, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use rstest::rstest;

    const DIM: i32 = 4;
    const ROWS: usize = 8;

    fn batch() -> RecordBatch {
        let values = Float32Array::from_iter((0..(DIM as usize * ROWS)).map(|v| v as f32));
        let vectors = Arc::new(FixedSizeListArray::try_new_from_values(values, DIM).unwrap());
        let schema = Schema::new(vec![
            Field::new("vec", vectors.data_type().clone(), true),
            Field::new("other", DataType::Int32, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                vectors,
                Arc::new(Int32Array::from_iter_values(0..ROWS as i32)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_flat_transform_renames_vector_column() {
        let input = batch();
        let output = FlatTransformer::new("vec").transform(&input).unwrap();

        assert!(output.column_by_name("vec").is_none());
        assert!(
            output.column_by_name("other").is_some(),
            "unrelated columns should survive"
        );
        assert_eq!(output.num_rows(), ROWS);

        let flat = output.column_by_name(FLAT_COLUMN).unwrap();
        assert_eq!(
            flat.data_type(),
            input.column_by_name("vec").unwrap().data_type()
        );
        assert_eq!(flat.as_ref(), input.column_by_name("vec").unwrap().as_ref());
    }

    /// Builds a batch whose vector column holds `nulls`, one entry per row.
    fn batch_with_nulls(nulls: &[bool]) -> RecordBatch {
        let rows = nulls.len();
        let values = Float32Array::from_iter((0..(DIM as usize * rows)).map(|v| v as f32));
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let validity = NullBuffer::from(nulls.iter().map(|n| !n).collect::<Vec<bool>>());
        let vectors = Arc::new(
            FixedSizeListArray::try_new(item, DIM, Arc::new(values), Some(validity)).unwrap(),
        );
        let schema = Schema::new(vec![
            Field::new("vec", vectors.data_type().clone(), true),
            Field::new("other", DataType::Int32, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                vectors,
                Arc::new(Int32Array::from_iter_values(0..rows as i32)),
            ],
        )
        .unwrap()
    }

    /// The renamed field takes its nullability from `Array::is_nullable`, which
    /// reports whether the column *currently holds* nulls rather than what the
    /// source schema declared. Both directions are pinned so the flag cannot be
    /// hardcoded either way.
    #[rstest]
    #[case::no_nulls(&[false, false], false)]
    #[case::some_nulls(&[false, true], true)]
    fn test_flat_transform_nullability_follows_the_data(
        #[case] nulls: &[bool],
        #[case] expected: bool,
    ) {
        let input = batch_with_nulls(nulls);
        let output = FlatTransformer::new("vec").transform(&input).unwrap();
        let field = output
            .schema()
            .field_with_name(FLAT_COLUMN)
            .unwrap()
            .clone();
        assert_eq!(field.is_nullable(), expected);
    }

    #[test]
    fn test_flat_transform_reports_missing_column() {
        let message = FlatTransformer::new("absent")
            .transform(&batch())
            .unwrap_err()
            .to_string();
        assert!(message.contains("column absent"), "{message}");
    }
}
