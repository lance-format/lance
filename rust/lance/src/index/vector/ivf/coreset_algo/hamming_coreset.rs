// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::FixedSizeListArray;
use arrow_schema::DataType;
use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_linalg::distance::MetricType;

use crate::dataset::Dataset;

use super::super::{
    FixedIvfTrainingRanges, FixedIvfTrainingSampler, HAMMING_METRIC_POLICY, KMeansProgressCallback,
    StreamingKMeansMetricPolicy, refine_streaming_hamming_kmodes_with_resampling,
    refine_streaming_hamming_kmodes_with_sampler, streaming_hamming,
};
use super::StreamingCoresetAlgorithm;

pub(in crate::index::vector::ivf) struct HammingCoresetAlgorithm;

#[async_trait]
impl StreamingCoresetAlgorithm for HammingCoresetAlgorithm {
    type Coreset = streaming_hamming::HammingCoreset;

    fn metric_policy(&self) -> &dyn StreamingKMeansMetricPolicy {
        &HAMMING_METRIC_POLICY
    }

    fn prepare_sample(
        &self,
        training_data: FixedSizeListArray,
        sampled_metric: MetricType,
    ) -> Result<FixedSizeListArray> {
        self.metric_policy()
            .validate_sample_metric(sampled_metric)?;
        if training_data.value_type() == DataType::UInt8 {
            Ok(training_data)
        } else {
            Err(Error::invalid_input(format!(
                "streaming Hamming k-modes requires UInt8 vectors, got {}",
                training_data.value_type()
            )))
        }
    }

    fn new_coreset(&self, dimension: usize, capacity: usize) -> Self::Coreset {
        streaming_hamming::HammingCoreset::new(dimension, capacity)
    }

    fn append_local_coreset(
        &self,
        coreset: &mut Self::Coreset,
        data: &FixedSizeListArray,
        local_k: usize,
        max_iters: usize,
        on_progress: KMeansProgressCallback,
    ) -> Result<()> {
        streaming_hamming::append_local_coreset(coreset, data, local_k, max_iters, on_progress)
    }

    fn append_coreset(&self, coreset: &mut Self::Coreset, other: Self::Coreset) -> Result<()> {
        coreset.append(other)
    }

    fn reduce_coreset(
        &self,
        coreset: &mut Self::Coreset,
        _dimension: usize,
        budget: usize,
    ) -> Result<()> {
        coreset.reduce_to_budget(budget)
    }

    fn coreset_len(&self, coreset: &Self::Coreset) -> usize {
        coreset.len()
    }

    fn train_centroids(
        &self,
        coreset: Self::Coreset,
        _dimension: usize,
        num_partitions: usize,
        max_iters: usize,
        on_progress: KMeansProgressCallback,
    ) -> Result<FixedSizeListArray> {
        let centroids = streaming_hamming::train_hierarchical(
            &coreset,
            num_partitions,
            max_iters,
            on_progress.clone(),
        )?;
        streaming_hamming::refine_weighted(&coreset, &centroids, 3, on_progress)
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
        refine_streaming_hamming_kmodes_with_sampler(
            sampler,
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
        refine_streaming_hamming_kmodes_with_resampling(
            dataset,
            column,
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
