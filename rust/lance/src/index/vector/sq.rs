// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Training reusable scalar quantization models for distributed index builds.
use super::utils::{
    filter_finite_training_data, get_vector_dim, get_vector_type, maybe_sample_training_data,
};
use crate::Dataset;
use arrow_array::Array;
use arrow_schema::DataType;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_index::vector::{
    quantizer::{Quantization, QuantizerBuildParams},
    sq::{ScalarQuantizer, builder::SQBuildParams, storage::ScalarQuantizationMetadata},
};
use lance_linalg::distance::DistanceType;

/// A trained SQ8 quantizer and the metric used to choose its input space.
///
/// Cosine encodes normalized vectors. L2 and dot product encode raw vectors
/// and may share the same SQ model. A physical merge also requires shared IVF
/// centroids and a common distance metric.
#[derive(Debug, Clone)]
pub struct SqModel {
    metadata: ScalarQuantizationMetadata,
    distance_type: DistanceType,
}

impl SqModel {
    /// Restore trained bounds and their distance metric.
    ///
    /// ```
    /// use lance::index::vector::sq::SqModel;
    /// use lance_index::vector::sq::storage::ScalarQuantizationMetadata;
    /// use lance_linalg::distance::DistanceType;
    /// let model = SqModel::try_new(
    ///     ScalarQuantizationMetadata { dim: 32, num_bits: 8, bounds: -1.0..1.0 },
    ///     DistanceType::L2,
    /// )?;
    /// assert_eq!(model.metadata().dim, 32);
    /// # Ok::<(), lance_core::Error>(())
    /// ```
    pub fn try_new(
        metadata: ScalarQuantizationMetadata,
        distance_type: DistanceType,
    ) -> Result<Self> {
        validate_sq_metric(distance_type)?;
        if metadata.dim == 0 || metadata.dim > i32::MAX as usize {
            return Err(Error::invalid_input(format!(
                "SQ model dimension must be in 1..={}, got {}",
                i32::MAX,
                metadata.dim
            )));
        }
        if metadata.num_bits != 8 {
            return Err(Error::invalid_input(format!(
                "SQ model num_bits must be 8, got {}",
                metadata.num_bits
            )));
        }
        if !metadata.bounds.start.is_finite()
            || !metadata.bounds.end.is_finite()
            || metadata.bounds.start > metadata.bounds.end
            || !(metadata.bounds.end - metadata.bounds.start).is_finite()
        {
            return Err(Error::invalid_input(format!(
                "SQ model bounds must be finite and ordered with a finite range, got {:?}",
                metadata.bounds
            )));
        }
        Ok(Self {
            metadata,
            distance_type,
        })
    }

    /// Return the dimension, bit width and bounds used to encode vectors.
    pub fn metadata(&self) -> &ScalarQuantizationMetadata {
        &self.metadata
    }

    /// Return the metric used at training time.
    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    /// Check that this model can encode the given column with the requested settings.
    ///
    /// See [`Self::try_new`] for constructing a model and [`build_sq_model`] for training one.
    pub fn validate_for_build(
        &self,
        schema: &Schema,
        column: &str,
        distance_type: DistanceType,
        params: &SQBuildParams,
    ) -> Result<()> {
        validate_sq_metric(distance_type)?;
        if (self.distance_type == DistanceType::Cosine) != (distance_type == DistanceType::Cosine) {
            return Err(Error::invalid_input(format!(
                "SQ model was trained with {} and cannot encode vectors for {}",
                self.distance_type, distance_type,
            )));
        }
        let (_, element_type) = get_vector_type(schema, column)?;
        if !matches!(
            element_type,
            DataType::Float16 | DataType::Float32 | DataType::Float64
        ) {
            return Err(Error::invalid_input(format!(
                "SQ requires floating-point vectors, got {element_type} for column {column}"
            )));
        }
        let dimension = get_vector_dim(schema, column)?;
        if self.metadata.dim != dimension || self.metadata.num_bits != params.num_bits {
            return Err(Error::invalid_input(format!(
                "SQ model dimension {} and num_bits {} do not match requested dimension {dimension} and num_bits {}",
                self.metadata.dim, self.metadata.num_bits, params.num_bits,
            )));
        }
        Ok(())
    }

    pub(crate) fn quantizer(&self) -> ScalarQuantizer {
        ScalarQuantizer::with_bounds(
            self.metadata.num_bits,
            self.metadata.dim,
            self.metadata.bounds.clone(),
        )
    }
}

fn validate_sq_metric(distance_type: DistanceType) -> Result<()> {
    if distance_type == DistanceType::Hamming {
        return Err(Error::invalid_input(
            "SQ does not support the hamming metric",
        ));
    }
    Ok(())
}

/// Train an SQ model with the same preprocessing as the IVF builder.
/// Null and non-finite vectors are excluded. An empty usable sample is rejected.
///
/// ```no_run
/// # use lance::{Dataset, Result, index::DatasetIndexExt};
/// # use lance_index::{IndexType, vector::sq::builder::SQBuildParams};
/// # use lance_linalg::distance::DistanceType;
/// # async fn example(dataset: &mut Dataset) -> Result<()> {
/// use lance::index::vector::{VectorIndexParams, sq::build_sq_model};
/// let sq_params = SQBuildParams::default();
/// let model = build_sq_model(dataset, "vector", DistanceType::L2, &sq_params).await?;
/// let params = VectorIndexParams::with_ivf_sq_params(DistanceType::L2, Default::default(), sq_params);
/// dataset.create_index_builder(&["vector"], IndexType::Vector, &params)
///     .with_sq_model(model)
///     .await?;
/// # Ok(()) }
/// ```
pub async fn build_sq_model(
    dataset: &Dataset,
    column: &str,
    distance_type: DistanceType,
    params: &SQBuildParams,
) -> Result<SqModel> {
    validate_sq_metric(distance_type)?;
    if params.num_bits != 8 {
        return Err(Error::invalid_input(format!(
            "SQ num_bits must be 8, got {}",
            params.num_bits
        )));
    }
    if params.sample_rate == 0 {
        return Err(Error::invalid_input(
            "SQ sample_rate must be greater than zero",
        ));
    }
    let data = maybe_sample_training_data(dataset, column, params.try_sample_size()?, None).await?;
    let data = if distance_type == DistanceType::Cosine {
        lance_linalg::kernels::normalize_fsl_owned(data)?
    } else {
        data
    };
    let data = filter_finite_training_data(data)?;
    if data.is_empty() {
        return Err(Error::invalid_input(
            "SQ training requires at least one finite, non-null vector",
        ));
    }
    let quantizer = ScalarQuantizer::build(&data, DistanceType::L2, params)?;
    SqModel::try_new(quantizer.metadata(None), distance_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::WriteParams;
    use crate::index::{DatasetIndexExt, vector::VectorIndexParams};
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::Float32Type;
    use arrow_array::{FixedSizeListArray, Float32Array};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_datagen::{ArrayGeneratorExt, BatchCount, Dimension, RowCount, array, gen_batch};
    use lance_index::{IndexType, vector::ivf::IvfBuildParams};
    use std::sync::Arc;

    #[rstest::rstest]
    #[case::dimension(0, 8, 0.0, 1.0, "dimension")]
    #[case::bits(2, 4, 0.0, 1.0, "num_bits")]
    #[case::reversed(2, 8, 2.0, 1.0, "bounds")]
    #[case::nan(2, 8, f64::NAN, 1.0, "bounds")]
    #[case::infinity(2, 8, 0.0, f64::INFINITY, "bounds")]
    #[case::overflow(2, 8, -f64::MAX, f64::MAX, "bounds")]
    fn test_invalid_sq_model(
        #[case] dim: usize,
        #[case] num_bits: u16,
        #[case] lower: f64,
        #[case] upper: f64,
        #[case] message: &str,
    ) {
        let metadata = ScalarQuantizationMetadata {
            dim,
            num_bits,
            bounds: lower..upper,
        };
        let err = SqModel::try_new(metadata, DistanceType::L2).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains(message));
    }

    #[test]
    fn test_constant_sq_model() {
        let metadata = ScalarQuantizationMetadata {
            dim: 2,
            num_bits: 8,
            bounds: 1.0..1.0,
        };
        SqModel::try_new(metadata, DistanceType::L2).unwrap();
    }

    #[rstest::rstest]
    #[case::l2(DistanceType::L2, 3.0, 8.0)]
    #[case::dot(DistanceType::Dot, 3.0, 8.0)]
    #[case::cosine(DistanceType::Cosine, 0.6, 0.8)]
    #[tokio::test]
    async fn test_sq_training_space(
        #[case] metric: DistanceType,
        #[case] lower: f64,
        #[case] upper: f64,
    ) {
        // A null list and a non-finite vector must not expand the trained bounds.
        let dataset = gen_batch()
            .col(
                "vector",
                array::cycle_vec(
                    array::cycle::<Float32Type>(vec![
                        3.0,
                        4.0,
                        -100.0,
                        100.0,
                        f32::NAN,
                        f32::INFINITY,
                        6.0,
                        8.0,
                    ]),
                    Dimension::from(2),
                )
                .with_nulls(&[false, true, false, false]),
            )
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(2))
            .await
            .unwrap();
        let sq = build_sq_model(&dataset, "vector", metric, &SQBuildParams::default())
            .await
            .unwrap();
        let metadata = sq.metadata();
        assert!((metadata.bounds.start - lower).abs() < 1e-6);
        assert!((metadata.bounds.end - upper).abs() < 1e-6);
        assert_eq!(metadata.dim, 2);
        assert_eq!(sq.distance_type(), metric);
    }

    #[tokio::test]
    async fn test_build_sq_shard_without_valid_vectors() {
        let reader = gen_batch()
            .col(
                "vector",
                array::cycle_vec(
                    array::cycle::<Float32Type>(vec![3.0, 4.0]),
                    Dimension::from(2),
                )
                .with_nulls(&[true, true, false, false]),
            )
            .into_reader_rows(RowCount::from(4), BatchCount::from(1));
        let mut dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        let fragment_id = fragments[0].id() as u32;
        let centroids =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![3.0, 4.0]), 2).unwrap();
        let params = VectorIndexParams::with_ivf_sq_params(
            DistanceType::L2,
            IvfBuildParams::try_with_centroids(1, Arc::new(centroids)).unwrap(),
            SQBuildParams {
                num_bits: 8,
                sample_rate: 256,
            },
        );
        let index = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .fragments(vec![fragment_id])
            .execute_uncommitted()
            .await
            .unwrap();
        assert_eq!(
            index.fragment_bitmap.unwrap().iter().collect::<Vec<_>>(),
            [fragment_id]
        );
        assert_eq!(index.files.unwrap().len(), 2);
    }

    #[rstest::rstest]
    #[case::empty(0, false)]
    #[case::all_null(2, true)]
    #[case::non_finite(2, false)]
    #[tokio::test]
    async fn test_sq_empty_training(#[case] rows: usize, #[case] all_null: bool) {
        let reader = gen_batch()
            .col(
                "vector",
                array::cycle_vec(
                    array::cycle::<Float32Type>(vec![f32::NAN]),
                    Dimension::from(2),
                )
                .with_nulls(&[all_null]),
            )
            .into_reader_rows(RowCount::from(rows as u64), BatchCount::from(1));
        let dataset = Dataset::write(reader, "memory://", None).await.unwrap();
        let err = build_sq_model(
            &dataset,
            "vector",
            DistanceType::L2,
            &SQBuildParams::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(
            err.to_string()
                .contains("at least one finite, non-null vector"),
            "{err}"
        );
    }
}
