// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::FixedSizeListArray;
use arrow_schema::DataType;
use async_trait::async_trait;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::Result;
use lance_linalg::distance::MetricType;

use crate::dataset::Dataset;

use super::super::{
    FixedIvfTrainingRanges, FixedIvfTrainingSampler, KMeansProgressCallback,
    StreamingFloatKMeansMetricPolicy, StreamingKMeansMetricPolicy, WeightedCoreset,
    WeightedHierarchicalKMeansParams, append_local_coreset, f32_fsl_from_values,
    refine_streaming_f32_kmeans_with_resampling, refine_streaming_f32_kmeans_with_sampler,
    refine_weighted_f32_kmeans, train_weighted_hierarchical_f32_kmeans,
};
use super::StreamingCoresetAlgorithm;

pub(in crate::index::vector::ivf) struct FloatCoresetAlgorithm<'a> {
    metric_policy: &'a dyn StreamingFloatKMeansMetricPolicy,
}

impl<'a> FloatCoresetAlgorithm<'a> {
    pub(in crate::index::vector::ivf) fn new(
        metric_policy: &'a dyn StreamingFloatKMeansMetricPolicy,
    ) -> Self {
        Self { metric_policy }
    }
}

#[async_trait]
impl StreamingCoresetAlgorithm for FloatCoresetAlgorithm<'_> {
    type Coreset = WeightedCoreset;

    fn metric_policy(&self) -> &dyn StreamingKMeansMetricPolicy {
        self.metric_policy
    }

    fn prepare_sample(
        &self,
        training_data: FixedSizeListArray,
        sampled_metric: MetricType,
    ) -> Result<FixedSizeListArray> {
        self.metric_policy.validate_sample_metric(sampled_metric)?;
        if training_data.value_type() == DataType::Float32 {
            Ok(training_data)
        } else {
            Ok(training_data.convert_to_floating_point()?)
        }
    }

    fn new_coreset(&self, dimension: usize, capacity: usize) -> Self::Coreset {
        WeightedCoreset::new(dimension, capacity)
    }

    fn append_local_coreset(
        &self,
        coreset: &mut Self::Coreset,
        data: &FixedSizeListArray,
        local_k: usize,
        max_iters: usize,
        on_progress: KMeansProgressCallback,
    ) -> Result<()> {
        append_local_coreset(
            coreset,
            data,
            self.metric_policy,
            local_k,
            max_iters,
            on_progress,
        )
    }

    fn append_coreset(&self, coreset: &mut Self::Coreset, other: Self::Coreset) -> Result<()> {
        coreset.append(other);
        Ok(())
    }

    fn reduce_coreset(
        &self,
        coreset: &mut Self::Coreset,
        dimension: usize,
        budget: usize,
    ) -> Result<()> {
        coreset.reduce_to_budget(dimension, budget, self.metric_policy)
    }

    fn coreset_len(&self, coreset: &Self::Coreset) -> usize {
        coreset.len()
    }

    fn train_centroids(
        &self,
        coreset: Self::Coreset,
        dimension: usize,
        num_partitions: usize,
        max_iters: usize,
        on_progress: KMeansProgressCallback,
    ) -> Result<FixedSizeListArray> {
        let (coreset_data, coreset_weights, coreset_losses) = coreset.into_fsl_parts(dimension)?;
        let mut centroids = {
            let params = WeightedHierarchicalKMeansParams {
                dimension,
                target_k: num_partitions,
                metric_policy: self.metric_policy,
                max_iters,
                on_progress: on_progress.clone(),
            };
            train_weighted_hierarchical_f32_kmeans(
                &coreset_data,
                &coreset_weights,
                &coreset_losses,
                &params,
            )?
        };
        let refined = refine_weighted_f32_kmeans(
            &coreset_data,
            &coreset_weights,
            &coreset_losses,
            &centroids,
            self.metric_policy,
            3,
            on_progress,
        )?;
        centroids = f32_fsl_from_values(refined.centroids, dimension)?;
        Ok(centroids)
    }

    async fn refine_with_sampler(
        &self,
        sampler: &FixedIvfTrainingSampler<'_>,
        streaming_sample_size: usize,
        sample_ranges: &FixedIvfTrainingRanges,
        initial_centroids: &FixedSizeListArray,
        passes: usize,
        on_progress: KMeansProgressCallback,
    ) -> Result<FixedSizeListArray> {
        refine_streaming_f32_kmeans_with_sampler(
            sampler,
            self.metric_policy.metric_type(),
            streaming_sample_size,
            sample_ranges,
            initial_centroids,
            passes,
            on_progress,
        )
        .await
    }

    async fn refine_with_resampling(
        &self,
        dataset: &Dataset,
        column: &str,
        total_sample_rate: usize,
        streaming_sample_rate: usize,
        num_partitions: usize,
        initial_centroids: &FixedSizeListArray,
        fragment_ids: Option<&[u32]>,
        passes: usize,
        on_progress: KMeansProgressCallback,
    ) -> Result<FixedSizeListArray> {
        refine_streaming_f32_kmeans_with_resampling(
            dataset,
            column,
            self.metric_policy.metric_type(),
            total_sample_rate,
            streaming_sample_rate,
            num_partitions,
            initial_centroids,
            fragment_ids,
            passes,
            on_progress,
        )
        .await
    }
}
