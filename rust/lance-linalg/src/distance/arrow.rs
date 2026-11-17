// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared validation, support table, and dtype dispatch for Arrow vectors.

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type, Int8Type, UInt8Type};
use arrow_array::{Array, ArrowPrimitiveType, FixedSizeListArray, Float32Array, ListArray};
use arrow_schema::{ArrowError, DataType};
use lance_arrow::FixedSizeListArrayExt;
use num_traits::AsPrimitive;

use super::{
    Cosine, DistanceFunc, DistanceType, Dot, L2, Normalize, cosine_distance_batch_into,
    dot_distance_batch_into, hamming, l2_distance_batch_into, norm_l2,
};
use crate::Result;

#[derive(Clone, Copy)]
pub(super) enum Norm {
    L2,
    Squared,
}

#[derive(Clone, Copy)]
pub(super) enum Operation<'a> {
    Batch(DistanceType),
    Multivec(DistanceType, &'a ListArray),
    Norm(Norm),
}

impl Operation<'_> {
    // Keep the accepted metric × dtype pairs here, including the deliberate
    // distinction between Int8 batches and multivectors.
    fn supports(self, dtype: &DataType) -> bool {
        matches!(
            (self, dtype),
            (
                Self::Batch(DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot)
                    | Self::Multivec(
                        DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot,
                        _
                    )
                    | Self::Norm(_),
                DataType::Float16 | DataType::Float32 | DataType::Float64,
            ) | (
                Self::Batch(DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot),
                DataType::Int8
            ) | (
                Self::Batch(DistanceType::Hamming) | Self::Multivec(DistanceType::Hamming, _),
                DataType::UInt8
            )
        )
    }

    fn supports_int8_query(self, dtype: &DataType) -> bool {
        matches!(
            (self, dtype),
            (
                Self::Batch(DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot),
                DataType::Float32 | DataType::Int8 | DataType::Int16 | DataType::Int32,
            )
        )
    }

    fn error(self, message: impl std::fmt::Display) -> ArrowError {
        let name = match self {
            Self::Batch(metric) => format!("{metric} batch"),
            Self::Multivec(metric, _) => format!("{metric} multivec"),
            Self::Norm(Norm::L2) => "L2 norm".to_string(),
            Self::Norm(Norm::Squared) => "squared L2 norm".to_string(),
        };
        ArrowError::InvalidArgumentError(format!("{name}: {message}"))
    }

    fn numeric<T>(self, query: &[T], values: &[T], dim: usize) -> Result<Vec<f32>>
    where
        T: L2 + Dot + Cosine + Normalize + AsPrimitive<f32> + AsPrimitive<f64>,
    {
        match self {
            Self::Batch(metric) => {
                let mut distances = vec![0.0; values.len() / dim];
                match metric {
                    DistanceType::L2 => l2_distance_batch_into(query, values, dim, &mut distances),
                    DistanceType::Cosine => {
                        cosine_distance_batch_into(query, values, dim, &mut distances)
                    }
                    DistanceType::Dot => {
                        dot_distance_batch_into(query, values, dim, &mut distances)
                    }
                    _ => return Err(self.error("Hamming requires UInt8 values")),
                }
                Ok(distances)
            }
            Self::Multivec(metric, rows) => match metric {
                DistanceType::Cosine => {
                    // `cosine` is defined as `cosine_fast(x, norm_l2(x), other)`,
                    // so hoisting each query sub-vector's norm out of the
                    // pairwise loop is bit-identical while dropping one O(dim)
                    // query pass per (row, stored sub-vector) pair.
                    let query_norms: Vec<f32> = query.chunks_exact(dim).map(norm_l2).collect();
                    Ok(multivec(query, values, dim, rows, |query_index, q, v| {
                        T::cosine_fast(q, query_norms[query_index], v)
                    }))
                }
                _ => {
                    let distance = metric.try_func()?;
                    Ok(multivec(query, values, dim, rows, |_, q, v| distance(q, v)))
                }
            },
            Self::Norm(Norm::L2) => Ok(values.chunks_exact(dim).map(norm_l2).collect()),
            Self::Norm(Norm::Squared) => Ok(values
                .chunks_exact(dim)
                .map(|v| {
                    // Preserve f64 accumulation for f64, and widen f16 before
                    // squaring so large and small half values remain finite.
                    if std::mem::size_of::<T>() > std::mem::size_of::<f32>() {
                        v.iter()
                            .map(|&v| {
                                let v: f64 = v.as_();
                                v * v
                            })
                            .sum::<f64>() as f32
                    } else {
                        v.iter()
                            .map(|&v| {
                                let v: f32 = v.as_();
                                v * v
                            })
                            .sum()
                    }
                })
                .collect()),
        }
    }

    fn binary(self, query: &[u8], values: &[u8], dim: usize) -> Result<Vec<f32>> {
        match self {
            Self::Batch(DistanceType::Hamming) => {
                let mut distances = vec![0.0; values.len() / dim];
                hamming::hamming_distance_batch_into(query, values, dim, &mut distances);
                Ok(distances)
            }
            Self::Multivec(DistanceType::Hamming, rows) => {
                Ok(multivec(query, values, dim, rows, |_, q, v| {
                    hamming::hamming(q, v)
                }))
            }
            _ => Err(self.error("UInt8 values require Hamming distance")),
        }
    }
}

/// Validate before the only Arrow dtype dispatch used by distances and norms.
pub(super) fn dispatch(
    operation: Operation<'_>,
    query: Option<&dyn Array>,
    vectors: &FixedSizeListArray,
) -> Result<Vec<f32>> {
    let dtype = vectors.value_type();
    let int8_query = query.is_some_and(|query| query.data_type() == &DataType::Int8)
        && operation.supports_int8_query(&dtype);
    if !operation.supports(&dtype) && !int8_query {
        return Err(operation.error(format!("unsupported vector element type {dtype}")));
    }
    if let Some(query) = query {
        if !int8_query && query.data_type() != &dtype {
            return Err(operation.error(format!(
                "query type {} does not match the stored vector type {dtype}",
                query.data_type()
            )));
        }
        if query.null_count() > 0 {
            return Err(operation.error(format!(
                "query must not contain nulls, found {} in {} values",
                query.null_count(),
                query.len()
            )));
        }
    }
    let dim = vectors.value_length() as usize;
    if dim == 0 {
        return Err(operation.error("stored vectors have dimension 0"));
    }
    if let Some(query) = query {
        match operation {
            Operation::Batch(_) if query.len() != dim => {
                return Err(operation.error(format!(
                    "query length {} must equal the vector dimension {dim}",
                    query.len()
                )));
            }
            Operation::Multivec(_, _) if query.is_empty() || !query.len().is_multiple_of(dim) => {
                return Err(operation.error(format!(
                    "query length {} must be a positive multiple of the vector dimension {dim}",
                    query.len()
                )));
            }
            _ => {}
        }
    }

    match dtype {
        DataType::Float16 => numeric::<Float16Type>(operation, query, vectors, dim),
        DataType::Float32 if int8_query => numeric_int8_query(operation, query, vectors, dim),
        DataType::Float32 => numeric::<Float32Type>(operation, query, vectors, dim),
        DataType::Float64 => numeric::<Float64Type>(operation, query, vectors, dim),
        DataType::Int8 | DataType::Int16 | DataType::Int32 => {
            numeric_int8_query(operation, query, vectors, dim)
        }
        DataType::UInt8 => operation.binary(
            query.map_or(&[], |q| q.as_primitive::<UInt8Type>().values().as_ref()),
            vectors.values().as_primitive::<UInt8Type>().values(),
            dim,
        ),
        _ => Err(operation.error(format!("unsupported vector element type {dtype}"))),
    }
}

fn numeric_int8_query(
    operation: Operation<'_>,
    query: Option<&dyn Array>,
    vectors: &FixedSizeListArray,
    dim: usize,
) -> Result<Vec<f32>> {
    let query = query.ok_or_else(|| operation.error("Int8 values require a query"))?;
    let query = query.as_primitive::<Int8Type>();
    let query = Float32Array::from(query.values().iter().map(|&v| v as f32).collect::<Vec<_>>());
    let vectors = vectors.convert_to_floating_point()?;
    numeric::<Float32Type>(operation, Some(&query), &vectors, dim)
}

fn numeric<T>(
    operation: Operation<'_>,
    query: Option<&dyn Array>,
    vectors: &FixedSizeListArray,
    dim: usize,
) -> Result<Vec<f32>>
where
    T: ArrowPrimitiveType,
    T::Native: L2 + Dot + Cosine + Normalize + AsPrimitive<f32> + AsPrimitive<f64>,
{
    operation.numeric(
        query.map_or(&[], |q| q.as_primitive::<T>().values().as_ref()),
        vectors.values().as_primitive::<T>().values(),
        dim,
    )
}

fn multivec<T, F>(query: &[T], values: &[T], dim: usize, rows: &ListArray, distance: F) -> Vec<f32>
where
    F: Fn(usize, &[T], &[T]) -> f32,
{
    rows.value_offsets()
        .windows(2)
        .enumerate()
        .map(|(row, offsets)| {
            if rows.is_null(row) || offsets[0] == offsets[1] {
                return f32::NAN;
            }
            let values = &values[offsets[0] as usize * dim..offsets[1] as usize * dim];
            query
                .chunks_exact(dim)
                .enumerate()
                .map(|(query_index, q)| {
                    values
                        .chunks_exact(dim)
                        .map(|v| distance(query_index, q, v))
                        .min_by(f32::total_cmp)
                        .unwrap_or(f32::NAN)
                })
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::Field;
    use num_traits::{Num, One, Zero};
    use std::cell::Cell;
    use std::ops::{Add, Div, Mul, Rem, Sub};
    use std::sync::Arc;

    thread_local! {
        /// `CountedF32` norm computations on this thread.
        static NORM_CALLS: Cell<usize> = const { Cell::new(0) };
    }

    /// Scalar test type whose norm computations are counted, so tests can
    /// observe how often a distance path recomputes a query norm.
    ///
    /// The kernels are plain scalar math; `Cosine` uses every default method,
    /// which is exactly the combination the multivector path relies on: the
    /// default `cosine` derives the norm through `Normalize::norm_l2`, while a
    /// caller that already knows the norm goes straight to `cosine_fast`.
    #[derive(Clone, Copy, PartialEq)]
    struct CountedF32(f32);

    impl Zero for CountedF32 {
        fn zero() -> Self {
            Self(0.0)
        }
        fn is_zero(&self) -> bool {
            self.0 == 0.0
        }
    }

    impl One for CountedF32 {
        fn one() -> Self {
            Self(1.0)
        }
    }

    impl Add for CountedF32 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(self.0 + rhs.0)
        }
    }

    impl Sub for CountedF32 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self(self.0 - rhs.0)
        }
    }

    impl Mul for CountedF32 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0)
        }
    }

    impl Div for CountedF32 {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            Self(self.0 / rhs.0)
        }
    }

    impl Rem for CountedF32 {
        type Output = Self;
        fn rem(self, rhs: Self) -> Self {
            Self(self.0 % rhs.0)
        }
    }

    impl Num for CountedF32 {
        type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;
        fn from_str_radix(
            str: &str,
            radix: u32,
        ) -> std::result::Result<Self, Self::FromStrRadixErr> {
            <f32 as Num>::from_str_radix(str, radix).map(Self)
        }
    }

    impl L2 for CountedF32 {
        fn l2(x: &[Self], y: &[Self]) -> f32 {
            x.iter().zip(y).map(|(a, b)| (a.0 - b.0).powi(2)).sum()
        }
    }

    impl Dot for CountedF32 {
        fn dot(x: &[Self], y: &[Self]) -> f32 {
            x.iter().zip(y).map(|(a, b)| a.0 * b.0).sum()
        }
    }

    impl Normalize for CountedF32 {
        fn norm_l2(vector: &[Self]) -> f32 {
            NORM_CALLS.with(|count| count.set(count.get() + 1));
            vector.iter().map(|v| v.0 * v.0).sum::<f32>().sqrt()
        }
    }

    impl AsPrimitive<f32> for CountedF32 {
        fn as_(self) -> f32 {
            self.0
        }
    }

    impl AsPrimitive<f64> for CountedF32 {
        fn as_(self) -> f64 {
            self.0 as f64
        }
    }

    impl Cosine for CountedF32 {}

    /// Two `List<FixedSizeList<Float32, 2>>` rows of two sub-vectors each,
    /// plus the matching flattened values `Operation::numeric` indexes into.
    fn rows_of_two_subvectors() -> (ListArray, Vec<f32>) {
        let values = vec![1.0, 0.0, 0.0, 1.0, 3.0, 4.0, 0.0, 2.0];
        let fsl =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values.clone()), 2).unwrap();
        let field = Arc::new(Field::new("item", fsl.data_type().clone(), true));
        let offsets = OffsetBuffer::from_lengths([2_usize, 2]);
        let rows = ListArray::try_new(field, offsets, Arc::new(fsl), None).unwrap();
        (rows, values)
    }

    /// Cosine multivector search must norm each query sub-vector exactly once
    /// rather than once per (row, stored sub-vector) pair: with 2 query
    /// sub-vectors, 2 rows and 2 stored sub-vectors per row the pairwise path
    /// computes 8 norms where 2 suffice. The distances themselves must not
    /// change: `cosine` is defined as `cosine_fast(x, norm_l2(x), other)`.
    #[test]
    fn multivec_cosine_norms_each_query_sub_vector_once() {
        let (rows, values) = rows_of_two_subvectors();
        let query: Vec<CountedF32> = [1.0, 0.0, 0.0, 2.0].map(CountedF32).to_vec();
        let values: Vec<CountedF32> = values.into_iter().map(CountedF32).collect();

        NORM_CALLS.with(|count| count.set(0));
        let distances = Operation::Multivec(DistanceType::Cosine, &rows)
            .numeric::<CountedF32>(&query, &values, 2)
            .unwrap();
        let norm_calls = NORM_CALLS.with(Cell::get);

        assert_eq!(
            norm_calls, 2,
            "cosine multivec must norm each of the 2 query sub-vectors exactly once, \
             got {norm_calls} norm computations"
        );

        // Row 0 ([1,0], [0,1]): both query sub-vectors match exactly, sum 0.
        // Row 1 ([3,4], [0,2]): q0=[1,0] is closest to [3,4] at 1-3/5, and
        // q1=[0,2] matches [0,2] exactly, sum 0.4.
        assert_eq!(distances.len(), 2, "{distances:?}");
        assert!(
            (distances[0] - 0.0).abs() < 1e-6,
            "row 0 distance: {distances:?}"
        );
        assert!(
            (distances[1] - 0.4).abs() < 1e-6,
            "row 1 distance: {distances:?}"
        );
    }
}
