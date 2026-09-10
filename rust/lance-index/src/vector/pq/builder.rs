// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Product Quantizer Builder
//!

use std::sync::Arc;

use crate::vector::quantizer::QuantizerBuildParams;
use arrow::array::PrimitiveBuilder;
use arrow_array::types::{Float16Type, Float64Type};
use arrow_array::{Array, ArrayRef, cast::AsArray, types::Float32Type};
use arrow_array::{ArrowNumericType, FixedSizeListArray, PrimitiveArray};
use arrow_schema::DataType;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use lance_linalg::distance::{Dot, L2, Normalize};

use super::ProductQuantizer;
use super::utils::divide_to_subvectors;
use crate::vector::kmeans::{KMeansParams, train_kmeans};

/// Parameters for building product quantizer.
#[derive(Debug, Clone)]
pub struct PQBuildParams {
    /// Number of sub-vectors to build PQ code
    pub num_sub_vectors: usize,

    /// The number of bits to present one PQ centroid.
    pub num_bits: usize,

    /// The max number of iterations for kmeans training.
    pub max_iters: usize,

    /// Run kmeans `REDOS` times and take the best result.
    /// Default to 1.
    pub kmeans_redos: usize,

    /// User provided codebook.
    pub codebook: Option<ArrayRef>,

    /// Sample rate to train PQ codebook.
    pub sample_rate: usize,
}

impl From<&PQBuildParams> for crate::pb::vector_index_details::ProductQuantization {
    fn from(params: &PQBuildParams) -> Self {
        Self {
            num_bits: params.num_bits as u32,
            num_sub_vectors: params.num_sub_vectors as u32,
        }
    }
}

impl Default for PQBuildParams {
    fn default() -> Self {
        Self {
            num_sub_vectors: 16,
            num_bits: 8,
            max_iters: 50,
            kmeans_redos: 1,
            codebook: None,
            sample_rate: 256,
        }
    }
}

impl QuantizerBuildParams for PQBuildParams {
    fn sample_size(&self) -> usize {
        self.training_sample_size()
            .expect("PQ training sample size must fit in usize")
    }

    fn try_sample_size(&self) -> Result<usize> {
        self.training_sample_size()
    }

    fn use_residual(distance_type: DistanceType) -> bool {
        matches!(distance_type, DistanceType::L2 | DistanceType::Cosine)
    }
}

impl PQBuildParams {
    pub fn new(num_sub_vectors: usize, num_bits: usize) -> Self {
        Self {
            num_sub_vectors,
            num_bits,
            ..Default::default()
        }
    }

    pub fn with_codebook(num_sub_vectors: usize, num_bits: usize, codebook: ArrayRef) -> Self {
        Self {
            num_sub_vectors,
            num_bits,
            codebook: Some(codebook),
            ..Default::default()
        }
    }

    fn num_centroids(&self) -> Result<usize> {
        u32::try_from(self.num_bits)
            .ok()
            .and_then(|num_bits| 1_usize.checked_shl(num_bits))
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "PQ centroid count overflows: num_bits={}, usize_bits={}",
                    self.num_bits,
                    usize::BITS
                ))
            })
    }

    fn training_sample_size(&self) -> Result<usize> {
        let num_centroids = self.num_centroids()?;
        self.sample_rate.checked_mul(num_centroids).ok_or_else(|| {
            Error::invalid_input(format!(
                "PQ training sample size overflows: sample_rate={}, num_centroids={num_centroids}",
                self.sample_rate
            ))
        })
    }

    fn build_from_fsl<T: ArrowNumericType>(
        &self,
        data: &FixedSizeListArray,
        distance_type: DistanceType,
    ) -> Result<ProductQuantizer>
    where
        T::Native: Dot + L2 + Normalize,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        assert_ne!(
            distance_type,
            DistanceType::Cosine,
            "PQ code does not support cosine"
        );

        let num_centroids = self.num_centroids()?;
        let max_training_rows = self.try_sample_size()?;
        let training_rows = data.len().min(max_training_rows);
        let sub_vectors = divide_to_subvectors::<T>(data, self.num_sub_vectors, training_rows)?;
        let dimension = data.value_length() as usize;
        let sub_vector_dimension = dimension / self.num_sub_vectors;

        let d = sub_vectors
            .into_iter()
            .enumerate()
            .map(|(sub_vec_idx, sub_vec)| {
                let params = KMeansParams::new(
                    self.codebook.as_ref().map(|cb| {
                        let sub_vec_centroids = FixedSizeListArray::try_new_from_values(
                            cb.as_fixed_size_list().values().as_primitive::<T>().slice(
                                sub_vec_idx * num_centroids * sub_vector_dimension,
                                num_centroids * sub_vector_dimension,
                            ),
                            sub_vector_dimension as i32,
                        )
                        .unwrap();
                        Arc::new(sub_vec_centroids)
                    }),
                    self.max_iters as u32,
                    self.kmeans_redos,
                    distance_type,
                );
                train_kmeans::<T>(
                    &sub_vec,
                    params,
                    sub_vector_dimension,
                    num_centroids,
                    self.sample_rate,
                )
                .map(|kmeans| kmeans.centroids)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut codebook_builder = PrimitiveBuilder::<T>::with_capacity(num_centroids * dimension);
        for centroid in d.iter() {
            let c = centroid
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .expect("failed to downcast to PrimitiveArray");
            codebook_builder.append_slice(c.values());
        }

        let pd_centroids = codebook_builder.finish();

        Ok(ProductQuantizer::new(
            self.num_sub_vectors,
            self.num_bits as u32,
            dimension,
            FixedSizeListArray::try_new_from_values(pd_centroids, dimension as i32)?,
            distance_type,
        ))
    }

    /// Build a [ProductQuantizer] from the given data.
    ///
    /// If the [`DistanceType`] is [`DistanceType::Cosine`], the input data will be normalized.
    pub fn build(&self, data: &dyn Array, distance_type: DistanceType) -> Result<ProductQuantizer> {
        assert_eq!(data.null_count(), 0);
        let fsl = data.as_fixed_size_list_opt().ok_or(Error::index(format!(
            "PQ builder: input is not a FixedSizeList: {}",
            data.data_type()
        )))?;

        let num_centroids = self.num_centroids()?;
        if data.len() < num_centroids {
            return Err(Error::unprocessable(format!(
                "Not enough rows to train PQ. Requires {num_centroids} rows but only {} available",
                data.len()
            )));
        }

        // TODO: support bf16 later.
        match fsl.value_type() {
            DataType::Float16 => self.build_from_fsl::<Float16Type>(fsl, distance_type),
            DataType::Float32 => self.build_from_fsl::<Float32Type>(fsl, distance_type),
            DataType::Float64 => self.build_from_fsl::<Float64Type>(fsl, distance_type),
            _ => Err(Error::index(format!(
                "PQ builder: unsupported data type: {}",
                fsl.value_type()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float32Array;

    #[test]
    fn test_build_samples_before_materializing_subvectors() {
        const N: usize = 4096;
        const DIM: usize = 8;
        const NUM_SUB_VECTORS: usize = 2;
        const NUM_BITS: usize = 2;
        const K: usize = 1 << NUM_BITS;
        const SUB_DIM: usize = DIM / NUM_SUB_VECTORS;

        // The 256 * K sample cap is smaller than N. Initial centroids make
        // training deterministic so the optimized and reference paths can be
        // compared exactly.
        let values = Float32Array::from_iter((0..N).flat_map(|row| {
            let cluster = row % K;
            (0..DIM).map(move |col| (cluster * 1000 + col) as f32 + row as f32 * 1e-4)
        }));
        let fsl = FixedSizeListArray::try_new_from_values(values.clone(), DIM as i32).unwrap();

        let init_values: Vec<f32> = (0..NUM_SUB_VECTORS * K)
            .flat_map(|i| (0..SUB_DIM).map(move |col| (i % K * 1000 + col) as f32))
            .collect();
        let init_codebook: ArrayRef = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(init_values.clone()),
                DIM as i32,
            )
            .unwrap(),
        );

        let pq = PQBuildParams::with_codebook(NUM_SUB_VECTORS, NUM_BITS, init_codebook)
            .build(&fsl, DistanceType::L2)
            .unwrap();

        let mut expected = Vec::with_capacity(K * DIM);
        for sub_idx in 0..NUM_SUB_VECTORS {
            let mut sub_values = Vec::with_capacity(N * SUB_DIM);
            for row in values.values().chunks(DIM) {
                sub_values.extend_from_slice(&row[sub_idx * SUB_DIM..(sub_idx + 1) * SUB_DIM]);
            }
            let sub_init = FixedSizeListArray::try_new_from_values(
                Float32Array::from(
                    init_values[sub_idx * K * SUB_DIM..(sub_idx + 1) * K * SUB_DIM].to_vec(),
                ),
                SUB_DIM as i32,
            )
            .unwrap();
            let params = KMeansParams::new(Some(Arc::new(sub_init)), 50, 1, DistanceType::L2);
            let kmeans = train_kmeans::<Float32Type>(
                &Float32Array::from(sub_values),
                params,
                SUB_DIM,
                K,
                256,
            )
            .unwrap();
            expected.extend_from_slice(kmeans.centroids.as_primitive::<Float32Type>().values());
        }

        assert_eq!(
            pq.codebook.values().as_primitive::<Float32Type>().values(),
            expected.as_slice()
        );
    }

    #[test]
    fn test_try_sample_size_rejects_overflow() {
        let mut params = PQBuildParams::new(2, 2);
        params.sample_rate = usize::MAX;

        let error = params.try_sample_size().unwrap_err();
        let expected = format!(
            "PQ training sample size overflows: sample_rate={}, num_centroids=4",
            usize::MAX
        );
        assert!(matches!(&error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(&expected), "{error}");
    }

    #[test]
    fn test_try_sample_size_rejects_centroid_count_overflow() {
        let params = PQBuildParams::new(2, usize::BITS as usize);

        let error = params.try_sample_size().unwrap_err();
        let expected = format!(
            "PQ centroid count overflows: num_bits={}, usize_bits={}",
            usize::BITS,
            usize::BITS
        );
        assert!(matches!(&error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(&expected), "{error}");
    }

    #[test]
    fn test_build_rejects_sample_size_overflow() {
        let values = Float32Array::from_iter((0..4 * 8).map(|v| v as f32));
        let fsl = FixedSizeListArray::try_new_from_values(values, 8).unwrap();
        let mut params = PQBuildParams::new(2, 2);
        params.sample_rate = usize::MAX;

        let error = params.build(&fsl, DistanceType::L2).unwrap_err();
        let expected = format!(
            "PQ training sample size overflows: sample_rate={}, num_centroids=4",
            usize::MAX
        );
        assert!(matches!(&error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(&expected), "{error}");
    }
}
