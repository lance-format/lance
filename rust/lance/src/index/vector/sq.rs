// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Training reusable scalar quantization models for distributed index builds.
use super::utils::{filter_finite_training_data, get_vector_dim, maybe_sample_training_data};
use crate::Dataset;
use lance_core::{Error, Result};
use lance_index::vector::{
    quantizer::{Quantization, QuantizerBuildParams},
    sq::{ScalarQuantizer, builder::SQBuildParams},
};
use lance_linalg::distance::DistanceType;

/// Train or restore an SQ model using the same preprocessing as the IVF builder.
/// Cosine training normalizes vectors; L2 and dot use the original vectors.
/// Null and non-finite vectors are excluded. An empty training sample is rejected.
/// Pass the resulting metadata in [`SQBuildParams::model`] on every worker,
/// together with shared IVF centroids and the same distance metric.
///
/// ```no_run
/// # use lance::{Dataset, Result};
/// # use lance_index::vector::{sq::builder::SQBuildParams, quantizer::Quantization};
/// # use lance_linalg::distance::DistanceType;
/// # async fn example(dataset: &Dataset) -> Result<()> {
/// let mut params = SQBuildParams::default();
/// let model = lance::index::vector::sq::build_sq_model(dataset, "vector", DistanceType::L2, &params).await?;
/// params.model = Some(model.metadata(None));
/// # Ok(()) }
/// ```
pub async fn build_sq_model(
    dataset: &Dataset,
    column: &str,
    distance_type: DistanceType,
    params: &SQBuildParams,
) -> Result<ScalarQuantizer> {
    if distance_type == DistanceType::Hamming {
        return Err(Error::invalid_input(
            "SQ does not support the hamming metric",
        ));
    }
    params.validate(get_vector_dim(dataset.schema(), column)?)?;
    let data = maybe_sample_training_data(dataset, column, params.try_sample_size()?, None).await?;
    let data = if distance_type == DistanceType::Cosine {
        lance_linalg::kernels::normalize_fsl_owned(data)?
    } else {
        data
    };
    let data = filter_finite_training_data(data)?;
    ScalarQuantizer::build(&data, DistanceType::L2, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::Float32Type;
    use lance_datagen::{ArrayGeneratorExt, BatchCount, Dimension, RowCount, array, gen_batch};

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
        assert!((sq.bounds().start - lower).abs() < 1e-6);
        assert!((sq.bounds().end - upper).abs() < 1e-6);
        assert_eq!(sq.metadata(None).dim, 2);
        let params = SQBuildParams {
            model: Some(sq.metadata(None)),
            sample_rate: 0,
            ..Default::default()
        };
        assert_eq!(
            build_sq_model(&dataset, "vector", metric, &params)
                .await
                .unwrap()
                .metadata(None),
            sq.metadata(None)
        );
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
