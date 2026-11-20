// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Training reusable scalar quantization models for distributed index builds.
use super::utils::{
    get_vector_type, infer_vector_dim, maybe_sample_training_data, prepare_quantizer_training_data,
};
use crate::Dataset;
use arrow_array::Array;
use lance_core::{Error, Result};
use lance_index::vector::{
    quantizer::{Quantization, QuantizerBuildParams},
    sq::{ScalarQuantizer, builder::SQBuildParams},
};
use lance_linalg::distance::DistanceType;

pub use lance_index::vector::sq::builder::{SqInputSpace, SqModel};

/// Train an SQ model with the same preprocessing as the IVF builder.
/// Null and non-finite vectors are excluded. An empty usable sample is rejected.
/// A supplied model is validated against the column and returned directly.
///
/// ```no_run
/// # use lance::{Dataset, Result, index::DatasetIndexExt};
/// # use lance_index::{IndexType, vector::sq::builder::SQBuildParams};
/// # use lance_linalg::distance::DistanceType;
/// # async fn example(dataset: &mut Dataset) -> Result<()> {
/// use lance::index::vector::{VectorIndexParams, sq::build_sq_model};
/// let sq_params = SQBuildParams::default();
/// let model = build_sq_model(dataset, "vector", DistanceType::L2, &sq_params).await?;
/// let params = VectorIndexParams::with_ivf_sq_params(
///     DistanceType::L2, Default::default(), SQBuildParams { model: Some(model), ..sq_params },
/// );
/// dataset.create_index_builder(&["vector"], IndexType::Vector, &params).await?;
/// # Ok(()) }
/// ```
pub async fn build_sq_model(
    dataset: &Dataset,
    column: &str,
    distance_type: DistanceType,
    params: &SQBuildParams,
) -> Result<SqModel> {
    if let Some(model) = &params.model {
        let (vector_type, element_type) = get_vector_type(dataset.schema(), column)?;
        model.validate_for_build(
            infer_vector_dim(&vector_type)?,
            &element_type,
            distance_type,
            params.num_bits,
        )?;
        return Ok(model.clone());
    }

    let input_space = SqInputSpace::try_from(distance_type)?;
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
    let data = prepare_quantizer_training_data(data, distance_type)?;
    if data.is_empty() {
        return Err(Error::invalid_input(
            "SQ training requires at least one finite, non-null vector",
        ));
    }
    let quantizer = ScalarQuantizer::build(&data, DistanceType::L2, params)?;
    SqModel::try_new(quantizer.metadata(None), input_space)
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
    use lance_index::{
        IndexType,
        vector::{ivf::IvfBuildParams, sq::storage::ScalarQuantizationMetadata},
    };
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
        let err = SqModel::try_new(metadata, SqInputSpace::Raw).unwrap_err();
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
        SqModel::try_new(metadata, SqInputSpace::Raw).unwrap();
    }

    #[rstest::rstest]
    #[case::l2(DistanceType::L2, SqInputSpace::Raw, 3.0, 8.0)]
    #[case::dot(DistanceType::Dot, SqInputSpace::Raw, 3.0, 8.0)]
    #[case::cosine(DistanceType::Cosine, SqInputSpace::L2Normalized, 0.6, 0.8)]
    #[tokio::test]
    async fn test_sq_training_space(
        #[case] metric: DistanceType,
        #[case] input_space: SqInputSpace,
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
        assert_eq!(sq.input_space(), input_space);
    }

    #[rstest::rstest]
    #[case::local_training(false)]
    #[case::shared_model(true)]
    #[tokio::test]
    async fn test_build_sq_shard_without_valid_vectors(#[case] reuse_model: bool) {
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
        let sq_params = if reuse_model {
            let model = build_sq_model(
                &dataset,
                "vector",
                DistanceType::L2,
                &SQBuildParams::default(),
            )
            .await
            .unwrap();
            // Computing a training sample size would overflow if the model were ignored.
            SQBuildParams {
                sample_rate: usize::MAX,
                model: Some(model),
                ..Default::default()
            }
        } else {
            SQBuildParams::default()
        };
        let params = VectorIndexParams::with_ivf_sq_params(
            DistanceType::L2,
            IvfBuildParams::try_with_centroids(1, Arc::new(centroids)).unwrap(),
            sq_params,
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
    #[case::empty(0, false, 1.0)]
    #[case::all_null(2, true, 1.0)]
    #[case::non_finite(2, false, f32::NAN)]
    #[tokio::test]
    async fn test_sq_empty_training(
        #[case] rows: usize,
        #[case] all_null: bool,
        #[case] value: f32,
    ) {
        let reader = gen_batch()
            .col(
                "vector",
                array::cycle_vec(array::cycle::<Float32Type>(vec![value]), Dimension::from(2))
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
