// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::{AddAssign, DivAssign};
use std::sync::Arc;
use std::{iter, ops::MulAssign};

use crate::vector::kmeans::{KMeansAlgoFloat, MaybeF16, compute_partitions};
use arrow_array::ArrowNumericType;
use arrow_array::{
    Array, FixedSizeListArray, PrimitiveArray, RecordBatch, UInt32Array,
    cast::AsArray,
    types::{Float16Type, Float32Type, Float64Type, UInt32Type},
};
use arrow_schema::DataType;
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::{Error, Result};
use lance_linalg::distance::{DistanceType, Dot, L2};
use num_traits::{Float, FromPrimitive, Num};
use tracing::instrument;

use super::{PQ_CODE_COLUMN, transform::Transformer};

/// Compute the residual vector of a Vector Matrix to their centroids.
///
/// The residual vector is the difference between the original vector and the centroid.
///
#[derive(Clone)]
pub struct ResidualTransform {
    /// Flattened centroids.
    centroids: FixedSizeListArray,

    /// Partition Column
    part_col: String,

    /// Vector Column
    vec_col: String,
}

impl std::fmt::Debug for ResidualTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ResidualTransform")
    }
}

impl ResidualTransform {
    pub fn new(centroids: FixedSizeListArray, part_col: &str, column: &str) -> Self {
        Self {
            centroids,
            part_col: part_col.to_owned(),
            vec_col: column.to_owned(),
        }
    }
}

fn do_compute_residual<T: ArrowNumericType + MaybeF16>(
    centroids: &FixedSizeListArray,
    vectors: &FixedSizeListArray,
    distance_type: Option<DistanceType>,
    partitions: Option<&UInt32Array>,
) -> Result<FixedSizeListArray>
where
    T::Native: Num + Float + L2 + Dot + MulAssign + DivAssign + AddAssign + FromPrimitive,
    PrimitiveArray<T>: From<Vec<T::Native>>,
{
    let dimension = centroids.value_length() as usize;
    let centroids = centroids.values().as_primitive::<T>();
    let vectors = vectors.values().as_primitive::<T>();

    let part_ids = partitions.cloned().unwrap_or_else(|| {
        compute_partitions::<T, KMeansAlgoFloat<T>>(
            centroids,
            vectors,
            dimension,
            distance_type.expect("provide either partitions or distance type"),
        )
        .0
        .into()
    });
    let part_ids = part_ids.values();

    let vectors_slice = vectors.values();
    let centroids_slice = centroids.values();
    let mut residuals = Vec::with_capacity(vectors.len());
    for (idx, vector) in vectors_slice.chunks_exact(dimension).enumerate() {
        let part_id = part_ids[idx] as usize;
        let c = &centroids_slice[part_id * dimension..(part_id + 1) * dimension];
        residuals.extend(iter::zip(vector, c).map(|(v, cent)| *v - *cent));
    }
    debug_assert_eq!(residuals.len(), vectors.len());
    let residual_arr = PrimitiveArray::<T>::from_iter_values(residuals);
    debug_assert_eq!(residual_arr.len(), vectors.len());
    Ok(FixedSizeListArray::try_new_from_values(
        residual_arr,
        dimension as i32,
    )?)
}

/// Compute residual vectors from the original vectors and centroids.
///
/// ## Parameter
/// - `centroids`: The KMeans centroids.
/// - `vectors`: The original vectors to compute residual vectors.
/// - `distance_type`: The distance type to compute the residual vector.
/// - `partitions`: The partition ID for each vector, if present.
pub(crate) fn compute_residual(
    centroids: &FixedSizeListArray,
    vectors: &FixedSizeListArray,
    distance_type: Option<DistanceType>,
    partitions: Option<&UInt32Array>,
) -> Result<FixedSizeListArray> {
    if centroids.value_length() != vectors.value_length() {
        return Err(Error::index(format!(
            "Compute residual vector: centroid and vector length mismatch: centroid: {}, vector: {}",
            centroids.value_length(),
            vectors.value_length(),
        )));
    }
    // TODO: Bf16 is not supported yet.
    match (centroids.value_type(), vectors.value_type()) {
        (DataType::Float16, DataType::Float16) => {
            do_compute_residual::<Float16Type>(centroids, vectors, distance_type, partitions)
        }
        (DataType::Float32, DataType::Float32) => {
            do_compute_residual::<Float32Type>(centroids, vectors, distance_type, partitions)
        }
        (DataType::Float64, DataType::Float64) => {
            do_compute_residual::<Float64Type>(centroids, vectors, distance_type, partitions)
        }
        (DataType::Float32, DataType::Int8) => do_compute_residual::<Float32Type>(
            centroids,
            &vectors.convert_to_floating_point()?,
            distance_type,
            partitions,
        ),
        _ => Err(Error::index(format!(
            "Compute residual vector: centroids and vector type mismatch: centroid: {}, vector: {}",
            centroids.value_type(),
            vectors.value_type(),
        ))),
    }
}

impl Transformer for ResidualTransform {
    /// Replace the original vector in the [`RecordBatch`] to residual vectors.
    ///
    /// The new [`RecordBatch`] will have a new column named `RESIDUAL_COLUMN`.
    #[instrument(name = "ResidualTransform::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        if batch.column_by_name(PQ_CODE_COLUMN).is_some() {
            // If the PQ code column is present, we don't need to compute residual vectors.
            return Ok(batch.clone());
        }

        let part_ids = batch
            .column_by_name(&self.part_col)
            .ok_or(Error::index(format!(
                "Compute residual vector: partition id column not found: {}",
                self.part_col
            )))?;
        let original = batch
            .column_by_name(&self.vec_col)
            .ok_or(Error::index(format!(
                "Compute residual vector: original vector column {} not found in batch {}",
                self.vec_col,
                batch.schema(),
            )))?;
        let original_vectors = original
            .as_fixed_size_list_opt()
            .ok_or(Error::index(format!(
                "Compute residual vector: original vector column {} is not fixed size list: {}",
                self.vec_col,
                original.data_type(),
            )))?;

        let part_ids_ref = part_ids.as_primitive::<UInt32Type>();
        let residual_arr =
            compute_residual(&self.centroids, original_vectors, None, Some(part_ids_ref))?;

        let batch = if residual_arr.data_type() != original.data_type() {
            batch.replace_column_schema_by_name(
                &self.vec_col,
                residual_arr.data_type().clone(),
                Arc::new(residual_arr),
            )?
        } else {
            batch.replace_column_by_name(&self.vec_col, Arc::new(residual_arr))?
        };

        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{ArrayRef, Float16Array, Float32Array, Float64Array, Int8Array, Int32Array};
    use arrow_schema::{Field, Schema};
    use half::f16;
    use lance_arrow::FixedSizeListArrayExt;

    const PART_COLUMN: &str = "part_id";
    const VECTOR_COLUMN: &str = "v";

    fn fsl<T: Array + 'static>(values: T, dim: i32) -> FixedSizeListArray {
        FixedSizeListArray::try_new_from_values(values, dim).unwrap()
    }

    fn f32_values(arr: &FixedSizeListArray) -> Vec<f32> {
        arr.values().as_primitive::<Float32Type>().values().to_vec()
    }

    /// A batch holding a vector column plus the partition ids the transform reads.
    fn batch(vectors: ArrayRef, part_ids: Vec<u32>) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new(VECTOR_COLUMN, vectors.data_type().clone(), true),
            Field::new(PART_COLUMN, DataType::UInt32, false),
        ]);
        RecordBatch::try_new(
            schema.into(),
            vec![vectors, Arc::new(UInt32Array::from(part_ids))],
        )
        .unwrap()
    }

    fn transform_of(centroids: FixedSizeListArray) -> ResidualTransform {
        ResidualTransform::new(centroids, PART_COLUMN, VECTOR_COLUMN)
    }

    /// The whole point of the file: each vector loses the centroid of *its own*
    /// partition, not the first one. Two partitions with distinct centroids and
    /// interleaved partition ids are what make a row/centroid mix-up visible.
    #[test]
    fn test_compute_residual_subtracts_the_assigned_centroid() {
        let centroids = fsl(Float32Array::from(vec![0.0, 0.0, 10.0, 20.0]), 2);
        let vectors = fsl(Float32Array::from(vec![1.0, 2.0, 11.0, 23.0, 3.0, 4.0]), 2);
        let part_ids = UInt32Array::from(vec![0, 1, 0]);

        let residual = compute_residual(&centroids, &vectors, None, Some(&part_ids)).unwrap();

        assert_eq!(residual.value_length(), 2);
        assert_eq!(f32_values(&residual), vec![1.0, 2.0, 1.0, 3.0, 3.0, 4.0]);
    }

    /// Mismatched widths would slice the centroid row out of bounds, so this is
    /// rejected up front. The message has to name both widths to be actionable.
    #[test]
    fn test_compute_residual_rejects_dimension_mismatch() {
        let centroids = fsl(Float32Array::from(vec![0.0, 0.0]), 2);
        let vectors = fsl(Float32Array::from(vec![1.0, 2.0, 3.0]), 3);

        let message = compute_residual(&centroids, &vectors, None, None)
            .unwrap_err()
            .to_string();

        assert!(message.contains("centroid: 2"), "{message}");
        assert!(message.contains("vector: 3"), "{message}");
    }

    /// Only the four pairs listed in `compute_residual` are dispatched; anything
    /// else must report both value types rather than silently picking one.
    #[test]
    fn test_compute_residual_rejects_type_mismatch() {
        let centroids = fsl(Float32Array::from(vec![0.0, 0.0]), 2);
        let vectors = fsl(Float64Array::from(vec![1.0, 2.0]), 2);

        let message = compute_residual(&centroids, &vectors, None, None)
            .unwrap_err()
            .to_string();

        assert!(message.contains("Float32"), "{message}");
        assert!(message.contains("Float64"), "{message}");
    }

    #[test]
    fn test_compute_residual_float16() {
        let centroids = fsl(
            Float16Array::from(vec![f16::from_f32(1.0), f16::from_f32(2.0)]),
            2,
        );
        let vectors = fsl(
            Float16Array::from(vec![f16::from_f32(4.0), f16::from_f32(8.0)]),
            2,
        );
        let part_ids = UInt32Array::from(vec![0]);

        let residual = compute_residual(&centroids, &vectors, None, Some(&part_ids)).unwrap();

        let values = residual.values().as_primitive::<Float16Type>().values();
        assert_eq!(values, &[f16::from_f32(3.0), f16::from_f32(6.0)]);
    }

    #[test]
    fn test_compute_residual_float64() {
        let centroids = fsl(Float64Array::from(vec![1.0, 2.0]), 2);
        let vectors = fsl(Float64Array::from(vec![4.0, 8.0]), 2);
        let part_ids = UInt32Array::from(vec![0]);

        let residual = compute_residual(&centroids, &vectors, None, Some(&part_ids)).unwrap();

        let values = residual.values().as_primitive::<Float64Type>().values();
        assert_eq!(values, &[3.0, 6.0]);
    }

    /// Int8 vectors are the one asymmetric arm: they are widened to Float32
    /// before subtraction, so the residual comes back wider than the input.
    #[test]
    fn test_compute_residual_widens_int8_vectors_to_float32() {
        let centroids = fsl(Float32Array::from(vec![0.5, 1.5]), 2);
        let vectors = fsl(Int8Array::from(vec![4i8, 9]), 2);
        let part_ids = UInt32Array::from(vec![0]);

        let residual = compute_residual(&centroids, &vectors, None, Some(&part_ids)).unwrap();

        assert_eq!(residual.value_type(), DataType::Float32);
        assert_eq!(f32_values(&residual), vec![3.5, 7.5]);
    }

    /// With no partition ids the transform falls back to assigning them from the
    /// distance type. Centroids are far apart so the assignment is unambiguous
    /// regardless of accumulation precision.
    #[test]
    fn test_compute_residual_assigns_partitions_when_absent() {
        let centroids = fsl(Float32Array::from(vec![0.0, 0.0, 100.0, 100.0]), 2);
        let vectors = fsl(Float32Array::from(vec![99.0, 101.0, 1.0, -2.0]), 2);

        let residual =
            compute_residual(&centroids, &vectors, Some(DistanceType::L2), None).unwrap();

        assert_eq!(f32_values(&residual), vec![-1.0, 1.0, 1.0, -2.0]);
    }

    /// An already-quantized batch has nothing to subtract, and recomputing would
    /// corrupt the codes, so the batch must pass through untouched.
    #[test]
    fn test_transform_is_noop_when_pq_code_present() {
        let vectors = fsl(Float32Array::from(vec![1.0, 2.0]), 2);
        let schema = Schema::new(vec![
            Field::new(VECTOR_COLUMN, vectors.data_type().clone(), true),
            Field::new(PART_COLUMN, DataType::UInt32, false),
            Field::new(PQ_CODE_COLUMN, DataType::Int32, false),
        ]);
        let input = RecordBatch::try_new(
            schema.into(),
            vec![
                Arc::new(vectors),
                Arc::new(UInt32Array::from(vec![0])),
                Arc::new(Int32Array::from(vec![7])),
            ],
        )
        .unwrap();

        let centroids = fsl(Float32Array::from(vec![9.0, 9.0]), 2);
        let output = transform_of(centroids).transform(&input).unwrap();

        assert_eq!(output, input);
    }

    #[test]
    fn test_transform_reports_missing_partition_column() {
        let vectors = fsl(Float32Array::from(vec![1.0, 2.0]), 2);
        let schema = Schema::new(vec![Field::new(
            VECTOR_COLUMN,
            vectors.data_type().clone(),
            true,
        )]);
        let input = RecordBatch::try_new(schema.into(), vec![Arc::new(vectors)]).unwrap();

        let centroids = fsl(Float32Array::from(vec![0.0, 0.0]), 2);
        let message = transform_of(centroids)
            .transform(&input)
            .unwrap_err()
            .to_string();

        assert!(message.contains(PART_COLUMN), "{message}");
    }

    #[test]
    fn test_transform_reports_missing_vector_column() {
        let input = RecordBatch::try_new(
            Schema::new(vec![Field::new(PART_COLUMN, DataType::UInt32, false)]).into(),
            vec![Arc::new(UInt32Array::from(vec![0]))],
        )
        .unwrap();

        let centroids = fsl(Float32Array::from(vec![0.0, 0.0]), 2);
        let message = transform_of(centroids)
            .transform(&input)
            .unwrap_err()
            .to_string();

        assert!(message.contains(VECTOR_COLUMN), "{message}");
    }

    #[test]
    fn test_transform_reports_non_vector_column() {
        let input = batch(Arc::new(Int32Array::from(vec![1, 2])), vec![0, 0]);

        let centroids = fsl(Float32Array::from(vec![0.0, 0.0]), 2);
        let message = transform_of(centroids)
            .transform(&input)
            .unwrap_err()
            .to_string();

        assert!(message.contains("is not fixed size list"), "{message}");
        assert!(message.contains("Int32"), "{message}");
    }

    /// Same-width residuals replace the column in place: the name and schema stay
    /// put and only the values change, so downstream projections keep working.
    #[test]
    fn test_transform_replaces_vector_column_in_place() {
        let vectors = fsl(Float32Array::from(vec![1.0, 2.0, 11.0, 22.0]), 2);
        let input = batch(Arc::new(vectors), vec![0, 1]);
        let centroids = fsl(Float32Array::from(vec![1.0, 1.0, 10.0, 20.0]), 2);

        let output = transform_of(centroids).transform(&input).unwrap();

        assert_eq!(output.schema(), input.schema());
        let residual = output
            .column_by_name(VECTOR_COLUMN)
            .unwrap()
            .as_fixed_size_list();
        assert_eq!(f32_values(residual), vec![0.0, 1.0, 1.0, 2.0]);
    }

    /// Int8 input widens to Float32, so replacing the column alone would leave the
    /// schema claiming Int8 while the data is Float32. This is the branch that has
    /// to rewrite the field type as well.
    #[test]
    fn test_transform_rewrites_schema_when_residual_widens() {
        let input = batch(Arc::new(fsl(Int8Array::from(vec![4i8, 9]), 2)), vec![0]);
        let centroids = fsl(Float32Array::from(vec![0.5, 1.5]), 2);

        let output = transform_of(centroids).transform(&input).unwrap();

        let field = output
            .schema()
            .field_with_name(VECTOR_COLUMN)
            .unwrap()
            .clone();
        let residual = output
            .column_by_name(VECTOR_COLUMN)
            .unwrap()
            .as_fixed_size_list();
        assert_eq!(field.data_type(), residual.data_type());
        assert_eq!(residual.value_type(), DataType::Float32);
        assert_eq!(f32_values(residual), vec![3.5, 7.5]);
    }
}
