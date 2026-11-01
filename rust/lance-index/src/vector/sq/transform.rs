// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    fmt::{Debug, Formatter},
    sync::Arc,
};

use arrow::array::AsArray;
use arrow_array::{
    RecordBatch,
    types::{Float16Type, Float32Type, Float64Type},
};
use arrow_schema::{DataType, Field};
use tracing::instrument;

use crate::vector::transform::Transformer;

use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};

use super::ScalarQuantizer;

pub struct SQTransformer {
    quantizer: ScalarQuantizer,
    input_column: String,
    output_column: String,
}

impl SQTransformer {
    pub fn new(quantizer: ScalarQuantizer, input_column: String, output_column: String) -> Self {
        Self {
            quantizer,
            input_column,
            output_column,
        }
    }
}

impl Debug for SQTransformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SQTransformer(input={}, output={})",
            self.input_column, self.output_column
        )
    }
}

impl Transformer for SQTransformer {
    #[instrument(name = "SQTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let input = batch
            .column_by_name(&self.input_column)
            .ok_or(Error::index(format!(
                "SQ Transform: column {} not found in batch",
                self.input_column
            )))?;
        let fsl = input.as_fixed_size_list_opt().ok_or_else(|| {
            Error::index(format!(
                "SQ Transform: column {} is not a fixed size list vector: {}",
                self.input_column,
                input.data_type()
            ))
        })?;
        let sq_code = match fsl.value_type() {
            DataType::Float16 => self.quantizer.transform::<Float16Type>(input)?,
            DataType::Float32 => self.quantizer.transform::<Float32Type>(input)?,
            DataType::Float64 => self.quantizer.transform::<Float64Type>(input)?,
            _ => {
                return Err(Error::index(format!(
                    "SQ Transform: column {} has unsupported value type: {}",
                    self.input_column,
                    fsl.value_type()
                )));
            }
        };

        let sq_field = Field::new(&self.output_column, sq_code.data_type().clone(), false);
        let batch = batch
            .try_with_column(sq_field, Arc::new(sq_code))?
            .drop_column(&self.input_column)?;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Array, FixedSizeListArray, Float32Array, Int32Array};
    use arrow_schema::Schema;
    use lance_arrow::FixedSizeListArrayExt;

    const SQ_COLUMN: &str = "sq";
    const VECTOR_COLUMN: &str = "v";

    fn transformer(dim: usize) -> SQTransformer {
        let mut quantizer = ScalarQuantizer::new(8, dim);
        quantizer.metadata.bounds = 0.0..1.0;
        SQTransformer::new(quantizer, VECTOR_COLUMN.into(), SQ_COLUMN.into())
    }

    fn vector_batch(values: Float32Array, dim: i32) -> RecordBatch {
        let fsl = FixedSizeListArray::try_new_from_values(values, dim).unwrap();
        let schema = Schema::new(vec![Field::new(
            VECTOR_COLUMN,
            fsl.data_type().clone(),
            true,
        )]);
        RecordBatch::try_new(schema.into(), vec![Arc::new(fsl)]).unwrap()
    }

    #[test]
    fn test_sq_transform_replaces_vector_with_code_column() {
        let batch = vector_batch(Float32Array::from(vec![0.0, 0.5, 1.0, 0.25]), 2);
        let output = transformer(2).transform(&batch).unwrap();

        assert!(
            output.column_by_name(VECTOR_COLUMN).is_none(),
            "input column should be dropped"
        );
        let codes = output
            .column_by_name(SQ_COLUMN)
            .unwrap()
            .as_fixed_size_list();
        assert_eq!(codes.len(), 2);
        assert_eq!(codes.value_length(), 2);
    }

    #[test]
    fn test_sq_transform_reports_missing_column() {
        let batch = vector_batch(Float32Array::from(vec![0.0, 1.0]), 2);
        let transformer = SQTransformer::new(
            ScalarQuantizer::new(8, 2),
            "absent".into(),
            SQ_COLUMN.into(),
        );
        let message = transformer.transform(&batch).unwrap_err().to_string();
        assert!(message.contains("column absent"), "{message}");
    }

    #[test]
    fn test_sq_transform_reports_non_vector_column() {
        let batch = RecordBatch::try_new(
            Schema::new(vec![Field::new(VECTOR_COLUMN, DataType::Int32, false)]).into(),
            vec![Arc::new(Int32Array::from(vec![1, 2]))],
        )
        .unwrap();
        let message = transformer(2).transform(&batch).unwrap_err().to_string();
        assert!(message.contains("column v"), "{message}");
        assert!(message.contains("Int32"), "{message}");
    }

    #[test]
    fn test_sq_transform_reports_unsupported_value_type() {
        let values = Int32Array::from(vec![1, 2, 3, 4]);
        let fsl = FixedSizeListArray::try_new_from_values(values, 2).unwrap();
        let schema = Schema::new(vec![Field::new(
            VECTOR_COLUMN,
            fsl.data_type().clone(),
            true,
        )]);
        let batch = RecordBatch::try_new(schema.into(), vec![Arc::new(fsl)]).unwrap();
        let message = transformer(2).transform(&batch).unwrap_err().to_string();
        assert!(message.contains("column v"), "{message}");
        assert!(message.contains("Int32"), "{message}");
    }
}
