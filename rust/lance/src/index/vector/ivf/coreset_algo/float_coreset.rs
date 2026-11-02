// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::FixedSizeListArray;
use async_trait::async_trait;
use lance_core::Result;
use lance_linalg::distance::MetricType;

use crate::dataset::Dataset;

use super::super::{
    FixedIvfTrainingRanges, FixedIvfTrainingSampler, KMeansProgressCallback,
    StreamingFloatKMeansMetricPolicy, StreamingKMeansArrowType, StreamingKMeansMetricPolicy,
    WeightedCoreset, WeightedHierarchicalKMeansParams, append_local_coreset,
    prepare_streaming_coreset_sample, refine_streaming_kmeans_with_resampling,
    refine_streaming_kmeans_with_sampler, refine_weighted_kmeans, streaming_kmeans_fsl_from_values,
    train_weighted_hierarchical_kmeans,
};
use super::StreamingCoresetAlgorithm;

pub(in crate::index::vector::ivf) struct FloatCoresetAlgorithm<'a, T: StreamingKMeansArrowType> {
    metric_policy: &'a dyn StreamingFloatKMeansMetricPolicy<T>,
}

impl<'a, T: StreamingKMeansArrowType> FloatCoresetAlgorithm<'a, T> {
    pub(in crate::index::vector::ivf) fn new(
        metric_policy: &'a dyn StreamingFloatKMeansMetricPolicy<T>,
    ) -> Self {
        Self { metric_policy }
    }
}

#[async_trait]
impl<T: StreamingKMeansArrowType> StreamingCoresetAlgorithm for FloatCoresetAlgorithm<'_, T> {
    type Coreset = WeightedCoreset<T>;

    fn metric_policy(&self) -> &dyn StreamingKMeansMetricPolicy {
        self.metric_policy
    }

    fn prepare_sample(
        &self,
        training_data: FixedSizeListArray,
        sampled_metric: MetricType,
    ) -> Result<FixedSizeListArray> {
        self.metric_policy.validate_sample_metric(sampled_metric)?;
        prepare_streaming_coreset_sample::<T>(training_data)
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
        append_local_coreset::<T>(
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
            train_weighted_hierarchical_kmeans::<T>(
                &coreset_data,
                &coreset_weights,
                &coreset_losses,
                &params,
            )?
        };
        let refined = refine_weighted_kmeans::<T>(
            &coreset_data,
            &coreset_weights,
            &coreset_losses,
            &centroids,
            self.metric_policy,
            3,
            on_progress,
        )?;
        centroids = streaming_kmeans_fsl_from_values::<T>(refined.centroids, dimension)?;
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
        refine_streaming_kmeans_with_sampler::<T>(
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
        refine_streaming_kmeans_with_resampling::<T>(
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
