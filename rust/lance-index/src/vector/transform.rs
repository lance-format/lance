// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Transforms
//!

use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, ArrowPrimitiveType, RecordBatch, UInt32Array, cast::AsArray};
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::RecordBatchExt;
use num_traits::Float;

use lance_core::{Error, Result};
use lance_linalg::kernels::normalize_fsl;
use tracing::instrument;

use super::{CENTROID_DIST_COLUMN, PART_ID_COLUMN};

/// Transform of a Vector Matrix.
///
///
pub trait Transformer: Debug + Send + Sync {
    /// Transform a [`RecordBatch`] of vectors
    ///
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch>;
}

/// Normalize Transformer
///
/// L2 Normalize each vector.
#[derive(Debug)]
pub struct NormalizeTransformer {
    input_column: String,
    output_column: Option<String>,
}

impl NormalizeTransformer {
    pub fn new(column: impl AsRef<str>) -> Self {
        Self {
            input_column: column.as_ref().to_owned(),
            output_column: None,
        }
    }

    /// Create Normalize output transform that will be stored in a different column.
    ///
    pub fn new_with_output(input_column: impl AsRef<str>, output_column: impl AsRef<str>) -> Self {
        Self {
            input_column: input_column.as_ref().to_owned(),
            output_column: Some(output_column.as_ref().to_owned()),
        }
    }
}

impl Transformer for NormalizeTransformer {
    #[instrument(name = "NormalizeTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let arr = batch.column_by_name(&self.input_column).ok_or_else(|| {
            Error::index(format!(
                "Normalize Transform: column {} not found in RecordBatch {}",
                self.input_column,
                batch.schema(),
            ))
        })?;

        let data = arr.as_fixed_size_list();
        let norm = normalize_fsl(data)?;
        let transformed = Arc::new(norm);

        if let Some(output_column) = &self.output_column {
            let field = Field::new(output_column, transformed.data_type().clone(), true);
            Ok(batch.try_with_column(field, transformed)?)
        } else {
            Ok(batch.replace_column_by_name(&self.input_column, transformed)?)
        }
    }
}

/// Only keep the vectors that is finite number, filter out NaN and Inf.
#[derive(Debug)]
pub(crate) struct KeepFiniteVectors {
    column: String,
}

impl KeepFiniteVectors {
    pub fn new(column: impl AsRef<str>) -> Self {
        Self {
            column: column.as_ref().to_owned(),
        }
    }
}

fn is_all_finite<T: ArrowPrimitiveType>(arr: &dyn Array) -> bool
where
    T::Native: Float,
{
    arr.null_count() == 0
        && !arr
            .as_primitive::<T>()
            .values()
            .iter()
            .any(|&v| !v.is_finite())
}

impl Transformer for KeepFiniteVectors {
    #[instrument(name = "KeepFiniteVectors::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let Some(arr) = batch.column_by_name(&self.column) else {
            return Ok(batch.clone());
        };

        let data = match arr.data_type() {
            DataType::FixedSizeList(_, _) => arr.as_fixed_size_list(),
            DataType::List(_) => arr.as_list::<i32>().values().as_fixed_size_list(),
            _ => {
                return Err(Error::index(format!(
                    "KeepFiniteVectors: column {} is not a fixed size list: {}",
                    self.column,
                    arr.data_type()
                )));
            }
        };

        let mut valid = Vec::with_capacity(batch.num_rows());
        data.iter().enumerate().for_each(|(idx, arr)| {
            if let Some(data) = arr {
                let is_valid = match data.data_type() {
                    // f16 vectors are computed in f32 space, so they will not overflow.
                    DataType::Float16 => is_all_finite::<Float16Type>(&data),
                    // f32 vectors must be bounded to avoid overflow in distance computation.
                    DataType::Float32 => is_all_finite::<Float32Type>(&data),
                    // f32 vectors are computed in f32 space, so they have the same limit as f64.
                    DataType::Float64 => is_all_finite::<Float64Type>(&data),
                    DataType::UInt8 => data.null_count() == 0,
                    DataType::Int8 => data.null_count() == 0,
                    _ => false,
                };
                if is_valid {
                    valid.push(idx as u32);
                }
            };
        });
        if valid.len() < batch.num_rows() {
            let indices = UInt32Array::from(valid);
            Ok(batch.take(&indices)?)
        } else {
            Ok(batch.clone())
        }
    }
}

#[derive(Debug)]
pub struct DropColumn {
    column: String,
}

impl DropColumn {
    pub fn new(column: &str) -> Self {
        Self {
            column: column.to_owned(),
        }
    }
}

impl Transformer for DropColumn {
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        Ok(batch.drop_column(&self.column)?)
    }
}

#[derive(Debug)]
pub struct Flatten {
    column: String,
}

impl Flatten {
    pub fn new(column: &str) -> Self {
        Self {
            column: column.to_owned(),
        }
    }
}

impl Transformer for Flatten {
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let Some(arr) = batch.column_by_name(&self.column) else {
            // this case is that we have precomputed buffers,
            // so we don't need to flatten the original vectors.
            return Ok(batch.clone());
        };
        match arr.data_type() {
            DataType::FixedSizeList(_, _) => Ok(batch.clone()),
            DataType::List(_) => {
                let vectors = arr.as_list::<i32>();
                // Each source row expands into one output row per vector in its list.
                // Replicate the row id AND every other column (e.g. covering / "included"
                // columns) across the expansion via a gather; dropping them here would
                // silently lose covering data on the multivector build/split path.
                //
                // The IVF partition-assignment columns are the exception and must be
                // DROPPED, not replicated: they describe the source *row*, and after the
                // expansion each sub-vector needs its own nearest centroid. Carrying them
                // through makes `PartitionTransformer` see its output column already
                // present and early-return, silently collapsing every sub-vector of a row
                // into that row's single partition -- no error, just degraded recall.
                let take_indices =
                    UInt32Array::from_iter_values((0..vectors.len()).flat_map(|i| {
                        let n = if vectors.is_valid(i) {
                            vectors.value(i).len()
                        } else {
                            0
                        };
                        std::iter::repeat_n(i as u32, n)
                    }));
                let flat_vectors = vectors.values().as_fixed_size_list().clone();
                let mut fields: Vec<Arc<Field>> = Vec::with_capacity(batch.num_columns());
                let mut columns: Vec<arrow_array::ArrayRef> =
                    Vec::with_capacity(batch.num_columns());
                for (i, field) in batch.schema().fields().iter().enumerate() {
                    if field.name() == &self.column {
                        fields.push(Arc::new(Field::new(
                            self.column.as_str(),
                            flat_vectors.data_type().clone(),
                            true,
                        )));
                        columns.push(Arc::new(flat_vectors.clone()));
                    } else if field.name() == PART_ID_COLUMN || field.name() == CENTROID_DIST_COLUMN
                    {
                        continue;
                    } else {
                        fields.push(field.clone());
                        columns.push(arrow::compute::take(batch.column(i), &take_indices, None)?);
                    }
                }
                let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
                Ok(batch)
            }
            _ => Err(Error::index(format!(
                "Flatten: column {} is not a vector: {}",
                self.column,
                arr.data_type()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use arrow::buffer::NullBuffer;
    use arrow_array::{FixedSizeListArray, Float16Array, Float32Array, Int32Array};
    use arrow_schema::Schema;
    use half::f16;
    use lance_arrow::*;
    use lance_linalg::distance::L2;

    #[test]
    fn test_flatten_preserves_extra_columns_for_multivector() {
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::UInt64Type;
        use arrow_array::{ListArray, UInt64Array};
        use lance_core::{ROW_ID, ROW_ID_FIELD};

        // row 10 has 2 vectors, row 20 has 1 vector; a covering column "meta" must be
        // replicated across each row's expanded vectors, not dropped.
        let values = Float32Array::from(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let fsl = FixedSizeListArray::try_new_from_values(values, 2).unwrap();
        let offsets = OffsetBuffer::new(vec![0i32, 2, 3].into());
        let item = Arc::new(Field::new("item", fsl.data_type().clone(), true));
        let list = ListArray::new(item, offsets, Arc::new(fsl), None);

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                ROW_ID_FIELD.clone(),
                Field::new("vector", list.data_type().clone(), true),
                Field::new("meta", DataType::Int32, true),
            ])),
            vec![
                Arc::new(UInt64Array::from(vec![10u64, 20])),
                Arc::new(list),
                Arc::new(Int32Array::from(vec![100, 200])),
            ],
        )
        .unwrap();

        let out = Flatten::new("vector").transform(&batch).unwrap();

        assert_eq!(out.num_rows(), 3, "2 + 1 vectors expand to 3 rows");
        assert_eq!(
            out[ROW_ID].as_primitive::<UInt64Type>().values(),
            &[10, 10, 20]
        );
        let meta = out
            .column_by_name("meta")
            .expect("covering column must survive Flatten")
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(
            meta.values(),
            &[100, 100, 200],
            "covering column replicated per expanded vector"
        );
        assert!(matches!(
            out.column_by_name("vector").unwrap().data_type(),
            DataType::FixedSizeList(_, 2)
        ));
    }

    /// The IVF partition-assignment columns describe the source *row*. Replicating them
    /// across the multivector expansion makes `PartitionTransformer` see its output column
    /// already present and early-return, so every sub-vector inherits its row's single
    /// partition instead of being assigned to its own nearest centroid -- silent recall
    /// loss with no error. They must be dropped even though covering columns are kept.
    #[test]
    fn test_flatten_drops_partition_columns_for_multivector() {
        use arrow::buffer::OffsetBuffer;
        use arrow_array::{ListArray, UInt32Array as U32, UInt64Array};
        use lance_core::ROW_ID_FIELD;

        let values = Float32Array::from(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let fsl = FixedSizeListArray::try_new_from_values(values, 2).unwrap();
        let offsets = OffsetBuffer::new(vec![0i32, 2, 3].into());
        let item = Arc::new(Field::new("item", fsl.data_type().clone(), true));
        let list = ListArray::new(item, offsets, Arc::new(fsl), None);

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                ROW_ID_FIELD.clone(),
                Field::new("vector", list.data_type().clone(), true),
                Field::new("meta", DataType::Int32, true),
                Field::new(PART_ID_COLUMN, DataType::UInt32, true),
                Field::new(CENTROID_DIST_COLUMN, DataType::Float32, true),
            ])),
            vec![
                Arc::new(UInt64Array::from(vec![10u64, 20])),
                Arc::new(list),
                Arc::new(Int32Array::from(vec![100, 200])),
                Arc::new(U32::from(vec![7u32, 9])),
                Arc::new(Float32Array::from(vec![0.5f32, 0.25])),
            ],
        )
        .unwrap();

        let out = Flatten::new("vector").transform(&batch).unwrap();

        assert_eq!(out.num_rows(), 3);
        assert!(
            out.column_by_name(PART_ID_COLUMN).is_none(),
            "a row's partition assignment must not survive the multivector expansion, or \
             PartitionTransformer early-returns and every sub-vector shares one partition"
        );
        assert!(
            out.column_by_name(CENTROID_DIST_COLUMN).is_none(),
            "the row's centroid distance is likewise meaningless per sub-vector"
        );
        // ...while genuine covering columns are still replicated.
        assert_eq!(
            out.column_by_name("meta")
                .expect("covering column must survive")
                .as_primitive::<arrow_array::types::Int32Type>()
                .values(),
            &[100, 100, 200]
        );
    }

    #[tokio::test]
    async fn test_normalize_transformer_f32() {
        let data = Float32Array::from_iter_values([1.0, 1.0, 2.0, 2.0].into_iter());
        let fsl = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            true,
        )]);
        let batch = RecordBatch::try_new(schema.into(), vec![Arc::new(fsl)]).unwrap();
        let transformer = NormalizeTransformer::new("v");
        let output = transformer.transform(&batch).unwrap();
        let actual = output.column_by_name("v").unwrap();
        let act_fsl = actual.as_fixed_size_list();
        assert_eq!(act_fsl.len(), 2);
        assert_relative_eq!(
            act_fsl.value(0).as_primitive::<Float32Type>().values()[..],
            [1.0 / 2.0_f32.sqrt(); 2]
        );
        assert_relative_eq!(
            act_fsl.value(1).as_primitive::<Float32Type>().values()[..],
            [2.0 / 8.0_f32.sqrt(); 2]
        );
    }

    #[tokio::test]
    async fn test_normalize_transformer_16() {
        let data =
            Float16Array::from_iter_values([1.0_f32, 1.0, 2.0, 2.0].into_iter().map(f16::from_f32));
        let fsl = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float16, true)), 2),
            true,
        )]);
        let batch = RecordBatch::try_new(schema.into(), vec![Arc::new(fsl)]).unwrap();
        let transformer = NormalizeTransformer::new("v");
        let output = transformer.transform(&batch).unwrap();
        let actual = output.column_by_name("v").unwrap();
        let act_fsl = actual.as_fixed_size_list();
        assert_eq!(act_fsl.len(), 2);
        let expect_1 = [f16::from_f32_const(1.0) / f16::from_f32_const(2.0).sqrt(); 2];
        act_fsl
            .value(0)
            .as_primitive::<Float16Type>()
            .values()
            .iter()
            .zip(expect_1.iter())
            .for_each(|(a, b)| assert!(a - b <= f16::epsilon()));

        let expect_2 = [f16::from_f32_const(2.0) / f16::from_f32_const(8.0).sqrt(); 2];
        act_fsl
            .value(1)
            .as_primitive::<Float16Type>()
            .values()
            .iter()
            .zip(expect_2.iter())
            .for_each(|(a, b)| assert!(a - b <= f16::epsilon()));
    }

    #[tokio::test]
    async fn test_normalize_transformer_with_output_column() {
        let data = Float32Array::from_iter_values([1.0, 1.0, 2.0, 2.0].into_iter());
        let fsl = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            true,
        )]);
        let batch = RecordBatch::try_new(schema.into(), vec![Arc::new(fsl.clone())]).unwrap();
        let transformer = NormalizeTransformer::new_with_output("v", "o");
        let output = transformer.transform(&batch).unwrap();
        let input = output.column_by_name("v").unwrap();
        assert_eq!(input.as_ref(), &fsl);
        let actual = output.column_by_name("o").unwrap();
        let act_fsl = actual.as_fixed_size_list();
        assert_eq!(act_fsl.len(), 2);
        assert_relative_eq!(
            act_fsl.value(0).as_primitive::<Float32Type>().values()[..],
            [1.0 / 2.0_f32.sqrt(); 2]
        );
        assert_relative_eq!(
            act_fsl.value(1).as_primitive::<Float32Type>().values()[..],
            [2.0 / 8.0_f32.sqrt(); 2]
        );
    }

    #[tokio::test]
    async fn test_drop_column() {
        let i32_array = Int32Array::from_iter_values([1, 2].into_iter());
        let data = Float32Array::from_iter_values([1.0, 1.0, 2.0, 2.0].into_iter());
        let fsl = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        let schema = Schema::new(vec![
            Field::new("i32", DataType::Int32, false),
            Field::new(
                "v",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
                true,
            ),
        ]);
        let batch =
            RecordBatch::try_new(schema.into(), vec![Arc::new(i32_array), Arc::new(fsl)]).unwrap();
        let transformer = DropColumn::new("v");
        let output = transformer.transform(&batch).unwrap();
        assert!(output.column_by_name("v").is_none());

        let dup_drop_result = transformer.transform(&output);
        assert!(dup_drop_result.is_ok());
    }

    /// Builds a 2-dim FSL column with one row per `rows` entry. `None` means a
    /// null vector; otherwise the two f32 values are used as-is.
    fn fsl_batch(rows: &[Option<[f32; 2]>]) -> RecordBatch {
        let values = Float32Array::from(
            rows.iter()
                .flat_map(|row| row.unwrap_or([0.0, 0.0]))
                .collect::<Vec<f32>>(),
        );
        let nulls = NullBuffer::from(rows.iter().map(Option::is_some).collect::<Vec<bool>>());
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let fsl =
            FixedSizeListArray::try_new(item.clone(), 2, Arc::new(values), Some(nulls)).unwrap();
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(item, 2),
            true,
        )]);
        RecordBatch::try_new(schema.into(), vec![Arc::new(fsl)]).unwrap()
    }

    #[test]
    fn test_keep_finite_vectors_drops_null_and_non_finite_rows() {
        let batch = fsl_batch(&[
            Some([1.0, 2.0]),
            None,
            Some([f32::NAN, 1.0]),
            Some([f32::INFINITY, 1.0]),
            Some([f32::NEG_INFINITY, 1.0]),
            Some([1e20, -1e20]),
            Some([3.0, 4.0]),
        ]);
        let output = KeepFiniteVectors::new("v").transform(&batch).unwrap();

        let kept = output.column_by_name("v").unwrap().as_fixed_size_list();
        assert_eq!(kept.len(), 3, "only finite rows survive");
        assert_eq!(kept.null_count(), 0);
        assert_eq!(
            kept.values().as_primitive::<Float32Type>().values(),
            &[1.0, 2.0, 1e20, -1e20, 3.0, 4.0]
        );
    }

    #[test]
    fn test_keep_finite_vectors_on_all_null_and_empty_batches() {
        let all_null = fsl_batch(&[None, None]);
        assert_eq!(
            KeepFiniteVectors::new("v")
                .transform(&all_null)
                .unwrap()
                .num_rows(),
            0
        );

        let empty = fsl_batch(&[]);
        assert_eq!(
            KeepFiniteVectors::new("v")
                .transform(&empty)
                .unwrap()
                .num_rows(),
            0
        );
    }

    #[test]
    fn test_keep_finite_vectors_passes_through_missing_column() {
        // A batch without the configured column is returned untouched rather
        // than erroring, so the transform is a no-op on unrelated batches.
        let batch = fsl_batch(&[Some([1.0, 2.0])]);
        let output = KeepFiniteVectors::new("other").transform(&batch).unwrap();
        assert_eq!(output.num_rows(), 1);
    }

    #[test]
    fn test_keep_finite_vectors_rejects_non_list_column() {
        let batch = RecordBatch::try_new(
            Schema::new(vec![Field::new("v", DataType::Int32, false)]).into(),
            vec![Arc::new(Int32Array::from(vec![1, 2]))],
        )
        .unwrap();
        let error = KeepFiniteVectors::new("v").transform(&batch).unwrap_err();
        assert!(matches!(error, Error::Index { .. }), "{error:?}");
        let message = error.to_string();
        assert!(message.contains("column v"), "{message}");
        assert!(message.contains("Int32"), "{message}");
    }

    #[test]
    fn test_is_all_finite() {
        let array = Float32Array::from(vec![1.0, 2.0]);
        assert!(is_all_finite::<Float32Type>(&array));

        let failure_values = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        for &v in &failure_values {
            let array = Float32Array::from(vec![1.0, v]);
            assert!(
                !is_all_finite::<Float32Type>(&array),
                "value {} should fail is_all_finite",
                v
            );
        }
    }

    #[test]
    fn test_finite_f16() {
        let v1 = vec![f16::MAX; 10_000];
        let v2 = vec![f16::MAX - f16::from_f32_const(1.0); 10_000];
        let distance = f16::l2(&v1, &v2);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_finite_f32() {
        let v1 = vec![f32::MAX; 10_000];
        let v2 = vec![f32::MAX - 1.0; 10_000];
        let distance = f32::l2(&v1, &v2);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_finite_f64() {
        let v1 = vec![f64::MAX; 10_000];
        let v2 = vec![f64::MAX - 1.0; 10_000];
        let distance = f64::l2(&v1, &v2);
        assert!(distance.is_finite());
    }
}
