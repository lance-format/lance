// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Metric-specific adapters for streaming coreset IVF training.

mod float_coreset;
mod hamming_coreset;

use arrow_array::FixedSizeListArray;
use async_trait::async_trait;
use lance_core::Result;
use lance_linalg::distance::MetricType;

use crate::dataset::Dataset;

use super::{
    FixedIvfTrainingRanges, FixedIvfTrainingSampler, KMeansProgressCallback,
    StreamingKMeansMetricPolicy,
};

pub(in crate::index::vector::ivf) use float_coreset::FloatCoresetAlgorithm;
pub(in crate::index::vector::ivf) use hamming_coreset::HammingCoresetAlgorithm;

#[async_trait]
pub(in crate::index::vector::ivf) trait StreamingCoresetAlgorithm:
    Sync
{
    type Coreset;

    fn metric_policy(&self) -> &dyn StreamingKMeansMetricPolicy;

    fn prepare_sample(
        &self,
        training_data: FixedSizeListArray,
        sampled_metric: MetricType,
    ) -> Result<FixedSizeListArray>;

    fn new_coreset(&self, dimension: usize, capacity: usize) -> Self::Coreset;

    fn append_local_coreset(
        &self,
        coreset: &mut Self::Coreset,
        data: &FixedSizeListArray,
        local_k: usize,
        max_iters: usize,
        on_progress: KMeansProgressCallback,
    ) -> Result<()>;

    fn append_coreset(&self, coreset: &mut Self::Coreset, other: Self::Coreset) -> Result<()>;

    fn reduce_coreset(
        &self,
        coreset: &mut Self::Coreset,
        dimension: usize,
        budget: usize,
    ) -> Result<()>;

    fn coreset_len(&self, coreset: &Self::Coreset) -> usize;

    fn train_centroids(
        &self,
        coreset: Self::Coreset,
        dimension: usize,
        num_partitions: usize,
        max_iters: usize,
        on_progress: KMeansProgressCallback,
    ) -> Result<FixedSizeListArray>;

    async fn refine_with_sampler(
        &self,
        sampler: &FixedIvfTrainingSampler<'_>,
        streaming_sample_size: usize,
        sample_ranges: &FixedIvfTrainingRanges,
        initial_centroids: &FixedSizeListArray,
        passes: usize,
        on_progress: KMeansProgressCallback,
    ) -> Result<FixedSizeListArray>;

    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<FixedSizeListArray>;
}
