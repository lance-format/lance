// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Distance metrics
//!
//! This module provides distance metrics for vectors.
//!
//! - `bf16, f16, f32, f64` types are supported.
//! - SIMD is used when available, on `x86_64`, `aarch64` and `loongarch64`
//!   architectures.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type, Int8Type, UInt8Type};
use arrow_array::{
    Array, ArrowPrimitiveType, FixedSizeListArray, Float32Array, ListArray, PrimitiveArray,
};
use arrow_schema::{ArrowError, DataType};

pub mod cosine;
pub mod cosine_u8;
pub mod dot;
pub mod dot_f16;
pub mod dot_u8;
pub mod hamming;
pub mod l2;
pub mod l2_u8;
pub mod norm_l2;

/// Widens an `Int8` query vector to `f32`, rejecting nulls.
///
/// The three `_arrow_batch` entry points that accept `Int8` take the query as a
/// `&dyn Array`, so it has to be widened before it reaches a kernel. A null
/// element has no distance to compute, so it is rejected rather than widened.
fn int8_query_to_f32(query: &PrimitiveArray<Int8Type>) -> Result<Float32Array> {
    if query.null_count() > 0 {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Int8 query vector `from` must not contain nulls, found {} in {} values",
            query.null_count(),
            query.len()
        )));
    }
    Ok(Float32Array::from(
        query.values().iter().map(|&v| v as f32).collect::<Vec<_>>(),
    ))
}

#[inline]
fn assert_equal_lengths(left_len: usize, right_len: usize) {
    assert_eq!(
        left_len, right_len,
        "distance inputs must have equal lengths: left={left_len}, right={right_len}"
    );
}

#[inline]
fn assert_batch_layout(vector_len: usize, batch_len: usize, dimension: usize) {
    assert!(
        dimension > 0,
        "distance dimension must be greater than zero"
    );
    assert_eq!(
        vector_len, dimension,
        "distance vector length must match dimension: vector={vector_len}, dimension={dimension}"
    );
    assert_eq!(
        batch_len % dimension,
        0,
        "distance batch length must be divisible by dimension: batch={batch_len}, dimension={dimension}"
    );
}

/// Largest number of maximal u8 product terms whose sum fits in a u32.
const U8_U32_ACCUMULATOR_MAX_LEN: usize = u32::MAX as usize / (u8::MAX as usize * u8::MAX as usize);

/// Number of distances computed per call into a runtime-selected batch kernel.
///
/// Keeping a small output buffer amortizes the `#[target_feature]` call while
/// avoiding the allocation and full-batch materialization that dominate the
/// common dimension-8 case.
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
const BATCH_BUFFER_SIZE: usize = 64;

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
pub(crate) type BatchKernel = unsafe fn(&[f32], &[f32], usize, &mut [f32]);

/// Runtime-selected target-feature tier for a batch kernel.
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
#[derive(Clone, Copy)]
pub(crate) enum BatchKind {
    Scalar,
    Avx,
    AvxFma,
    Avx512,
}

/// Per-vector operations used when a batch consumer can be folded directly.
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
pub(crate) trait BatchOperation {
    fn fold_scalar<B, F>(key: &[f32], batch: &[f32], dimension: usize, init: B, f: F) -> B
    where
        F: FnMut(B, f32) -> B;

    unsafe fn fold_avx<B, F>(key: &[f32], batch: &[f32], dimension: usize, init: B, f: F) -> B
    where
        F: FnMut(B, f32) -> B;

    unsafe fn fold_avx_fma<B, F>(key: &[f32], batch: &[f32], dimension: usize, init: B, f: F) -> B
    where
        F: FnMut(B, f32) -> B;

    unsafe fn fold_avx512<B, F>(key: &[f32], batch: &[f32], dimension: usize, init: B, f: F) -> B
    where
        F: FnMut(B, f32) -> B;
}

/// Allocation-free iterator over a runtime-selected f32 distance kernel.
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
pub(crate) struct BatchIter<'a, O> {
    key: &'a [f32],
    batch: &'a [f32],
    dimension: usize,
    kernel: BatchKernel,
    kind: BatchKind,
    buffer: [f32; BATCH_BUFFER_SIZE],
    buffer_index: usize,
    buffer_len: usize,
    operation: std::marker::PhantomData<O>,
}

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
impl<'a, O> BatchIter<'a, O> {
    /// Creates an iterator after the caller has verified `kernel`'s CPU feature
    /// requirements.
    ///
    /// # Safety
    /// The host must support every target feature required by `kernel`.
    #[inline]
    pub(crate) unsafe fn new(
        key: &'a [f32],
        batch: &'a [f32],
        dimension: usize,
        kernel: BatchKernel,
        kind: BatchKind,
    ) -> Self {
        // Match `chunks_exact` validation before the buffered iterator performs
        // division by the dimension.
        let _ = batch.chunks_exact(dimension);
        Self {
            key,
            batch,
            dimension,
            kernel,
            kind,
            buffer: [0.0; BATCH_BUFFER_SIZE],
            buffer_index: 0,
            buffer_len: 0,
            operation: std::marker::PhantomData,
        }
    }

    #[inline]
    fn refill(&mut self) -> bool {
        let num_vectors = (self.batch.len() / self.dimension).min(BATCH_BUFFER_SIZE);
        if num_vectors == 0 {
            return false;
        }

        let num_values = num_vectors * self.dimension;
        let (input, remaining) = self.batch.split_at(num_values);
        // SAFETY: `new` requires the caller to verify the selected kernel's
        // target features before constructing the iterator.
        unsafe {
            (self.kernel)(
                self.key,
                input,
                self.dimension,
                &mut self.buffer[..num_vectors],
            );
        }
        self.batch = remaining;
        self.buffer_index = 0;
        self.buffer_len = num_vectors;
        true
    }
}

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
impl<O: BatchOperation> Iterator for BatchIter<'_, O> {
    type Item = f32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer_index == self.buffer_len && !self.refill() {
            return None;
        }
        let value = self.buffer[self.buffer_index];
        self.buffer_index += 1;
        Some(value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    /// Processes each filled buffer directly so consumers such as `sum` do not
    /// pay a refill check for every distance.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let accumulator = self.buffer[self.buffer_index..self.buffer_len]
            .iter()
            .copied()
            .fold(init, &mut f);

        // SAFETY: `new` requires the caller to verify the selected tier's
        // target features. Each helper runs the complete remaining loop in
        // that target-feature context, avoiding intermediate output writes.
        match self.kind {
            BatchKind::Scalar => {
                O::fold_scalar(self.key, self.batch, self.dimension, accumulator, f)
            }
            BatchKind::Avx => unsafe {
                O::fold_avx(self.key, self.batch, self.dimension, accumulator, f)
            },
            BatchKind::AvxFma => unsafe {
                O::fold_avx_fma(self.key, self.batch, self.dimension, accumulator, f)
            },
            BatchKind::Avx512 => unsafe {
                O::fold_avx512(self.key, self.batch, self.dimension, accumulator, f)
            },
        }
    }

    #[inline]
    fn for_each<F>(self, mut f: F)
    where
        F: FnMut(Self::Item),
    {
        self.fold((), |(), value| f(value));
    }
}

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
impl<O: BatchOperation> ExactSizeIterator for BatchIter<'_, O> {
    #[inline]
    fn len(&self) -> usize {
        self.buffer_len - self.buffer_index + self.batch.len() / self.dimension
    }
}

pub use cosine::*;
pub use dot::*;
pub use hamming::{
    BinaryHashValues, Cluster, ClusteringResult, PairwiseResult, UnionFind, cluster_edges,
    cluster_pairwise_result, extract_binary_hashes_from_fixed_list, extract_hashes_from_fixed_list,
    hamming_distance_arrow_batch, hamming_u64, pairwise_hamming_distance,
    pairwise_hamming_distance_binary, pairwise_hamming_distance_binary_parallel,
    pairwise_hamming_distance_parallel,
};
pub use l2::*;
use lance_core::deepsize::DeepSizeOf;
pub use norm_l2::*;

use crate::Result;

/// Distance metrics type.
#[derive(Debug, Copy, Clone, PartialEq, DeepSizeOf)]
pub enum DistanceType {
    L2,
    Cosine,
    /// Dot Product
    Dot,
    /// Hamming Distance
    Hamming,
}

/// For backwards compatibility.
pub type MetricType = DistanceType;

pub type DistanceFunc<T> = fn(&[T], &[T]) -> f32;
pub type BatchDistanceFunc = fn(&[f32], &[f32], usize) -> Arc<Float32Array>;
pub type ArrowBatchDistanceFunc = fn(&dyn Array, &FixedSizeListArray) -> Result<Arc<Float32Array>>;

impl DistanceType {
    /// Compute the distance from one vector to a batch of vectors.
    ///
    /// This propagates nulls to the output.
    pub fn arrow_batch_func(&self) -> ArrowBatchDistanceFunc {
        match self {
            Self::L2 => l2_distance_arrow_batch,
            Self::Cosine => cosine_distance_arrow_batch,
            Self::Dot => dot_distance_arrow_batch,
            Self::Hamming => hamming_distance_arrow_batch,
        }
    }

    /// Returns the distance function between two vectors.
    pub fn func<T: L2 + Cosine + Dot>(&self) -> DistanceFunc<T> {
        match self {
            Self::L2 => l2,
            Self::Cosine => cosine_distance,
            Self::Dot => dot_distance,
            Self::Hamming => todo!(),
        }
    }
}

impl std::fmt::Display for DistanceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::L2 => "l2",
                Self::Cosine => "cosine",
                Self::Dot => "dot",
                Self::Hamming => "hamming",
            }
        )
    }
}

impl TryFrom<&str> for DistanceType {
    type Error = ArrowError;

    fn try_from(s: &str) -> std::result::Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Self::L2),
            "cosine" => Ok(Self::Cosine),
            "dot" => Ok(Self::Dot),
            "hamming" => Ok(Self::Hamming),
            _ => Err(ArrowError::InvalidArgumentError(format!(
                "Metric type '{s}' is not supported"
            ))),
        }
    }
}

/// Computes the additive late-interaction distance from a multivector query.
///
/// For each query sub-vector, this finds the minimum distance to any stored
/// sub-vector in the row, then sums those minimum distances. Null or empty
/// stored rows produce `NaN`.
pub fn multivec_distance(
    query: &dyn Array,
    vectors: &ListArray,
    distance_type: DistanceType,
) -> Result<Vec<f32>> {
    let (element_type, dim) = match vectors.value_type() {
        DataType::FixedSizeList(field, dim) => (field.data_type().clone(), dim as usize),
        _ => {
            return Err(ArrowError::InvalidArgumentError(
                "vectors must be a list of fixed size list".to_string(),
            ));
        }
    };

    // Validate the query once, up front, rather than per vector. The type and
    // metric checks below prevent an arrow downcast panic or the `unreachable!`
    // dispatch arm — the dispatch picks its kernel type from the query's dtype
    // and then downcasts the *stored* values to that same type. The dim, null
    // and length checks prevent a `chunks_exact` panic and, worse, silently
    // wrong results: a short query yields no sub-vectors and scores every row
    // `0.0`, and a null slot is scored from whatever the values buffer holds.
    let query_type = query.data_type();
    // Which element types have a kernel here at all. `Int8` is a valid vector
    // element type elsewhere in the stack (`l2_distance_arrow_batch` and its
    // siblings have an `Int8` arm) but has no multivector kernel, so it is
    // rejected for the type, not the metric.
    let type_supported = matches!(
        query_type,
        DataType::UInt8 | DataType::Float16 | DataType::Float32 | DataType::Float64
    );
    if !type_supported {
        return Err(ArrowError::InvalidArgumentError(format!(
            "multivec_distance: unsupported vector element type {query_type}"
        )));
    }
    let metric_supported = match query_type {
        DataType::UInt8 => distance_type == DistanceType::Hamming,
        _ => matches!(
            distance_type,
            DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot
        ),
    };
    if !metric_supported {
        return Err(ArrowError::InvalidArgumentError(format!(
            "multivec_distance: distance type {distance_type} does not support query type {query_type}"
        )));
    }
    if *query_type != element_type {
        return Err(ArrowError::InvalidArgumentError(format!(
            "multivec_distance: query type {query_type} does not match the stored vector type {element_type}"
        )));
    }
    if dim == 0 {
        return Err(ArrowError::InvalidArgumentError(
            "multivec_distance: stored vectors have dimension 0".to_string(),
        ));
    }
    if query.null_count() > 0 {
        return Err(ArrowError::InvalidArgumentError(format!(
            "multivec_distance: query must not contain nulls, got {} null(s)",
            query.null_count()
        )));
    }
    if query.is_empty() || !query.len().is_multiple_of(dim) {
        return Err(ArrowError::InvalidArgumentError(format!(
            "multivec_distance: query length {} must be a positive multiple of the vector dimension {dim}",
            query.len()
        )));
    }

    let mut dists = Vec::with_capacity(vectors.len());
    for v in vectors.iter() {
        match v {
            None => dists.push(f32::NAN),
            Some(v) => {
                let multivector = v.as_fixed_size_list();
                if multivector.len() == 0 {
                    dists.push(f32::NAN);
                    continue;
                }

                let distance = match distance_type {
                    DistanceType::Hamming => multivec_distance_impl::<UInt8Type>(
                        query,
                        multivector,
                        dim,
                        hamming::hamming,
                    ),
                    _ => match query.data_type() {
                        DataType::Float16 => multivec_distance_impl::<Float16Type>(
                            query,
                            multivector,
                            dim,
                            distance_type.func(),
                        ),
                        DataType::Float32 => multivec_distance_impl::<Float32Type>(
                            query,
                            multivector,
                            dim,
                            distance_type.func(),
                        ),
                        DataType::Float64 => multivec_distance_impl::<Float64Type>(
                            query,
                            multivector,
                            dim,
                            distance_type.func(),
                        ),
                        _ => unreachable!("missed to check query type"),
                    },
                };

                dists.push(distance);
            }
        }
    }
    Ok(dists)
}

fn multivec_distance_impl<T: ArrowPrimitiveType>(
    query: &dyn Array,
    multivector: &FixedSizeListArray,
    dim: usize,
    distance_func: DistanceFunc<T::Native>,
) -> f32 {
    let query = query.as_primitive::<T>().values();
    query
        .chunks_exact(dim)
        .map(|q| {
            multivector
                .values()
                .as_primitive::<T>()
                .values()
                .chunks_exact(dim)
                .map(|v| distance_func(q, v))
                .min_by(|a, b| a.total_cmp(b))
                .unwrap()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    use std::io::Write;
    use std::sync::Arc;

    use arrow_array::types::{Float16Type, Float32Type, Int8Type};
    use arrow_array::{
        Float32Array, Float64Array, Int8Array, Int32Array, ListArray, PrimitiveArray, UInt8Array,
    };
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_schema::Field;
    use half::f16;
    use lance_arrow::FixedSizeListArrayExt;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_x86_runtime_feature_report() {
        // Write directly to stderr so this remains visible when libtest captures
        // ordinary output from passing tests.
        writeln!(
            std::io::stderr().lock(),
            "lance-linalg x86 runtime features: avx={}, fma={}, avx2={}, avx512f={}, avx512bw={}, avx512vnni={}, avx512vpopcntdq={}",
            std::is_x86_feature_detected!("avx"),
            std::is_x86_feature_detected!("fma"),
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("avx512f"),
            std::is_x86_feature_detected!("avx512bw"),
            std::is_x86_feature_detected!("avx512vnni"),
            std::is_x86_feature_detected!("avx512vpopcntdq"),
        )
        .expect("write x86 runtime feature report");
    }

    #[test]
    fn test_arrow_batch_type_errors_identify_the_argument() {
        let float32_targets =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![1.0, 2.0]), 2).unwrap();
        let unsupported_query = Int32Array::from(vec![1, 2]);

        for distance_type in [DistanceType::L2, DistanceType::Cosine, DistanceType::Dot] {
            let error =
                distance_type.arrow_batch_func()(&unsupported_query, &float32_targets).unwrap_err();
            assert!(
                matches!(error, ArrowError::InvalidArgumentError(_)),
                "{distance_type} returned a different error variant: {error}"
            );
        }

        let unsupported_from_error =
            cosine_distance_arrow_batch(&unsupported_query, &float32_targets).unwrap_err();
        assert!(
            matches!(&unsupported_from_error, ArrowError::InvalidArgumentError(message)
                if message == "`from` has unsupported data type Int32"),
            "unexpected unsupported `from` error: {unsupported_from_error}"
        );

        let float32_query = Float32Array::from(vec![1.0, 2.0]);
        let float64_targets =
            FixedSizeListArray::try_new_from_values(Float64Array::from(vec![1.0, 2.0]), 2).unwrap();
        let mismatched_to_error =
            cosine_distance_arrow_batch(&float32_query, &float64_targets).unwrap_err();
        assert!(
            matches!(&mismatched_to_error, ArrowError::InvalidArgumentError(message)
                if message == "`to` values have data type Float64, expected Float32 to match `from`"),
            "unexpected mismatched `to` error: {mismatched_to_error}"
        );
    }

    /// Build `List<FixedSizeList<T, dim>>` rows from flattened sub-vector values.
    fn multivecs_of<T: ArrowPrimitiveType>(rows: Vec<Vec<T::Native>>, dim: i32) -> ListArray {
        let lengths = rows
            .iter()
            .map(|row| {
                assert_eq!(row.len() % dim as usize, 0);
                row.len() / dim as usize
            })
            .collect::<Vec<_>>();
        let values = ScalarBuffer::from(rows.into_iter().flatten().collect::<Vec<_>>());
        let inner = PrimitiveArray::<T>::new(values, None);
        let fsl = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", T::DATA_TYPE, true)),
            dim,
            Arc::new(inner),
            None,
        )
        .unwrap();
        let offsets = OffsetBuffer::from_lengths(lengths);
        let field = Arc::new(Field::new("item", fsl.data_type().clone(), true));
        ListArray::try_new(field, offsets, Arc::new(fsl), None).unwrap()
    }

    /// Build one `List<FixedSizeList<T, dim>>` row.
    fn multivec_of<T: ArrowPrimitiveType>(values: Vec<T::Native>, dim: i32) -> ListArray {
        multivecs_of::<T>(vec![values], dim)
    }

    /// The `(query dtype, distance type)` pre-check and the dispatch must agree.
    /// `UInt8` is only valid with Hamming, and the float types only with the
    /// float metrics; a mismatch must be an error rather than a panic in the
    /// dispatch arm or inside an arrow downcast.
    #[test]
    fn test_multivec_distance_rejects_dtype_metric_mismatch() {
        let f32_vectors = multivec_of::<Float32Type>(vec![1.0, 2.0], 2);
        let u8_vectors = multivec_of::<UInt8Type>(vec![1, 2], 2);

        let u8_query: Arc<dyn Array> = Arc::new(UInt8Array::from(vec![1_u8, 2]));
        let f32_query: Arc<dyn Array> = Arc::new(Float32Array::from(vec![1.0_f32, 2.0]));

        // Query and stored types MATCH in each case, so only the metric is wrong
        // — otherwise the element-type check would reject these first and this
        // test would pass with the metric guard deleted.
        for dt in [DistanceType::L2, DistanceType::Cosine, DistanceType::Dot] {
            let err = multivec_distance(u8_query.as_ref(), &u8_vectors, dt).unwrap_err();
            assert!(
                matches!(&err, ArrowError::InvalidArgumentError(m) if m.contains("does not support query type")),
                "UInt8 query with {dt} must be rejected for the metric, got: {err}"
            );
        }

        let err =
            multivec_distance(f32_query.as_ref(), &f32_vectors, DistanceType::Hamming).unwrap_err();
        assert!(
            matches!(&err, ArrowError::InvalidArgumentError(m) if m.contains("does not support query type")),
            "Float32 query with hamming must be rejected for the metric, got: {err}"
        );
    }

    /// The `_arrow_batch` entry points that accept `Int8` widen the query element
    /// by element.
    /// A null there used to reach an `unwrap`, so a query column with a null in
    /// its values panicked instead of returning an error, on the metrics that
    /// accept `Int8`.
    #[test]
    fn test_arrow_batch_rejects_null_int8_query() {
        let targets =
            FixedSizeListArray::try_new_from_values(Int8Array::from(vec![1_i8, 2, 3, 4]), 2)
                .unwrap();
        let query: Arc<dyn Array> = Arc::new(Int8Array::from(vec![Some(1_i8), None]));

        for dt in [DistanceType::L2, DistanceType::Cosine, DistanceType::Dot] {
            let err = dt.arrow_batch_func()(query.as_ref(), &targets).unwrap_err();
            assert!(
                matches!(&err, ArrowError::InvalidArgumentError(m)
                    if m.contains("Int8 query vector `from`") && m.contains("found 1 in 2 values")),
                "{dt} accepted a null Int8 query element, got: {err}"
            );
        }

        // The same query without nulls goes through, so the guard is not
        // rejecting every `Int8` query.
        let query: Arc<dyn Array> = Arc::new(Int8Array::from(vec![1_i8, 2]));
        for dt in [DistanceType::L2, DistanceType::Cosine, DistanceType::Dot] {
            assert_eq!(
                dt.arrow_batch_func()(query.as_ref(), &targets)
                    .unwrap()
                    .len(),
                2,
                "{dt} rejected a well-formed Int8 query"
            );
        }

        // A sliced query reads through `values()`, which has to follow the slice:
        // the window here holds no nulls while the full buffer does. The L2
        // distances are asserted literally rather than against a second call,
        // since computing the expected values by the same route would hide a bug
        // that transformed both alike. Query [3, 4] against [[1, 2], [3, 4]]
        // gives (3-1)^2 + (4-2)^2 = 8 and 0.
        let sliced = Int8Array::from(vec![None, Some(3_i8), Some(4), None]).slice(1, 2);
        let query: Arc<dyn Array> = Arc::new(sliced);
        let got = DistanceType::L2.arrow_batch_func()(query.as_ref(), &targets).unwrap();
        assert_eq!(
            got.values(),
            &[8.0_f32, 0.0],
            "L2 did not follow the query slice"
        );
        for dt in [DistanceType::Cosine, DistanceType::Dot] {
            let got = dt.arrow_batch_func()(query.as_ref(), &targets).unwrap();
            let want =
                dt.arrow_batch_func()(Arc::new(Int8Array::from(vec![3_i8, 4])).as_ref(), &targets)
                    .unwrap();
            assert_eq!(got, want, "{dt} did not follow the query slice");
        }
    }

    /// An input that is both a length mismatch and a null query must reach the
    /// null error on all three metrics. `dot` used to carry a second
    /// `debug_assert_eq!` on the dimension in its public entry point, ahead of
    /// the `Int8` arm's null guard.
    #[test]
    fn test_arrow_batch_null_and_length_mismatch_agree() {
        let targets =
            FixedSizeListArray::try_new_from_values(Int8Array::from(vec![1_i8, 2, 3, 4]), 2)
                .unwrap();
        let query: Arc<dyn Array> = Arc::new(Int8Array::from(vec![Some(1_i8), None, Some(2)]));

        for dt in [DistanceType::L2, DistanceType::Cosine, DistanceType::Dot] {
            let err = dt.arrow_batch_func()(query.as_ref(), &targets).unwrap_err();
            assert!(
                matches!(&err, ArrowError::InvalidArgumentError(m) if m.contains("must not contain nulls")),
                "{dt} did not report the null query, got: {err}"
            );
        }
    }

    /// `Int8` is a valid vector element type elsewhere in the crate but has no
    /// multivector kernel, so it must be rejected for the type, not the metric.
    #[test]
    fn test_multivec_distance_rejects_unsupported_element_type() {
        let i8_vectors = multivec_of::<Int8Type>(vec![1, 2], 2);
        let i8_query: Arc<dyn Array> = Arc::new(Int8Array::from(vec![1_i8, 2]));

        let err = multivec_distance(i8_query.as_ref(), &i8_vectors, DistanceType::L2).unwrap_err();
        assert!(
            matches!(&err, ArrowError::InvalidArgumentError(m) if m.contains("unsupported vector element type")),
            "Int8 must be rejected for the element type, got: {err}"
        );
    }

    /// The query's element type must match the stored vectors': the dispatch
    /// picks `T` from the query and then downcasts the stored array to the same
    /// `T` without checking it.
    #[test]
    fn test_multivec_distance_rejects_element_type_mismatch() {
        let f16_vectors =
            multivec_of::<Float16Type>(vec![f16::from_f32(1.0), f16::from_f32(2.0)], 2);
        let f32_query: Arc<dyn Array> = Arc::new(Float32Array::from(vec![1.0_f32, 2.0]));

        let err =
            multivec_distance(f32_query.as_ref(), &f16_vectors, DistanceType::L2).unwrap_err();
        assert!(
            matches!(&err, ArrowError::InvalidArgumentError(m) if m.contains("does not match the stored vector type")),
            "Float32 query against a Float16 column must be rejected, got: {err}"
        );
    }

    /// A query length that is not a positive multiple of `dim` is structurally
    /// invalid: `chunks_exact` would silently drop the tail, and a query shorter
    /// than `dim` would yield no sub-vectors at all and score every row `0.0`.
    #[test]
    fn test_multivec_distance_rejects_bad_query_length() {
        let vectors = multivec_of::<Float32Type>(vec![1.0, 2.0], 2);

        for bad in [vec![7.0_f32], vec![7.0, 7.0, 999.0], vec![]] {
            let len = bad.len();
            let query: Arc<dyn Array> = Arc::new(Float32Array::from(bad));
            let err = multivec_distance(query.as_ref(), &vectors, DistanceType::L2).unwrap_err();
            assert!(
                matches!(&err, ArrowError::InvalidArgumentError(m) if m.contains("must be a positive multiple")),
                "query of length {len} against dim 2 must be rejected, got: {err}"
            );
        }
    }

    /// A zero-dimension column would panic in `chunks_exact(0)`; it gets its own
    /// message rather than blaming the query's length.
    #[test]
    fn test_multivec_distance_rejects_zero_dim() {
        let values = Float32Array::from(Vec::<f32>::new());
        let fsl = FixedSizeListArray::try_new_with_length(
            Arc::new(Field::new("item", DataType::Float32, true)),
            0,
            Arc::new(values),
            None,
            1,
        )
        .unwrap();
        let field = Arc::new(Field::new("item", fsl.data_type().clone(), true));
        let vectors = ListArray::try_new(
            field,
            OffsetBuffer::from_lengths([1_usize]),
            Arc::new(fsl),
            None,
        )
        .unwrap();
        let query: Arc<dyn Array> = Arc::new(Float32Array::from(vec![1.0_f32, 2.0]));

        let err = multivec_distance(query.as_ref(), &vectors, DistanceType::L2).unwrap_err();
        assert!(
            matches!(&err, ArrowError::InvalidArgumentError(m)
                if m.contains("stored vectors have dimension 0")
                    && !m.contains("positive multiple")),
            "a zero-dim column must be rejected on its own terms, got: {err}"
        );
    }

    /// A null query slot is read from the raw values buffer, so it would be
    /// silently scored as whatever the buffer holds.
    #[test]
    fn test_multivec_distance_rejects_null_query() {
        let vectors = multivec_of::<Float32Type>(vec![1.0, 2.0], 2);
        let query: Arc<dyn Array> = Arc::new(Float32Array::from(vec![Some(1.0_f32), None]));

        let err = multivec_distance(query.as_ref(), &vectors, DistanceType::L2).unwrap_err();
        assert!(
            matches!(&err, ArrowError::InvalidArgumentError(m) if m.contains("must not contain nulls")),
            "a query with nulls must be rejected, got: {err}"
        );
    }

    /// Each query sub-vector contributes its minimum Hamming distance to the
    /// row total.
    #[test]
    fn test_multivec_distance_hamming() {
        let vectors =
            multivecs_of::<UInt8Type>(vec![vec![0b0000_0000, 0b0000_1111], vec![0b0000_0011]], 1);
        let query: Arc<dyn Array> = Arc::new(UInt8Array::from(vec![0b0000_0000_u8, 0b0000_1111]));

        let dists = multivec_distance(query.as_ref(), &vectors, DistanceType::Hamming).unwrap();

        assert_eq!(dists, vec![0.0, 4.0]);
    }

    #[rstest::rstest]
    #[case::l2_perfect(
        DistanceType::L2,
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        0.0
    )]
    #[case::cosine_perfect(
        DistanceType::Cosine,
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        0.0
    )]
    #[case::dot_perfect(
        DistanceType::Dot,
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        0.0
    )]
    #[case::cosine_repeated_query(
        DistanceType::Cosine,
        vec![0.6, 0.8],
        vec![1.0, 0.0, 1.0, 0.0],
        0.8
    )]
    #[case::cosine_single_query(
        DistanceType::Cosine,
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        1.0
    )]
    fn test_multivec_distance_float(
        #[case] distance_type: DistanceType,
        #[case] vectors: Vec<f32>,
        #[case] query: Vec<f32>,
        #[case] expected: f32,
    ) {
        let vectors = multivec_of::<Float32Type>(vectors, 2);
        let query: Arc<dyn Array> = Arc::new(Float32Array::from(query));

        let dists = multivec_distance(query.as_ref(), &vectors, distance_type).unwrap();

        assert!((dists[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_multivec_distance_empty_row_is_nan() {
        let query: Arc<dyn Array> = Arc::new(Float32Array::from_iter_values([1.0_f32, 2.0]));

        let dim = 2;
        let values = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![Some(vec![Some(1.0_f32), Some(2.0)])],
            dim,
        );

        // Two rows: first is empty list, second has one sub-vector.
        let offsets = OffsetBuffer::from_lengths([0_usize, 1]);
        let field = Arc::new(Field::new("item", values.data_type().clone(), true));
        let vectors = ListArray::try_new(field, offsets, Arc::new(values), None).unwrap();

        let dists = multivec_distance(query.as_ref(), &vectors, DistanceType::Dot).unwrap();
        assert_eq!(dists.len(), 2);
        assert!(dists[0].is_nan());
        assert_eq!(dists[1], -4.0);
    }
}
