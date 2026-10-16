// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Build IVF model

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, FixedSizeListArray, UInt32Array, UInt64Array};
use futures::TryStreamExt;
use object_store::path::Path;

use lance_core::error::{Error, Result};
use lance_io::stream::RecordBatchStream;

/// Parameters to build IVF partitions
#[derive(Debug, Clone)]
pub struct IvfBuildParams {
    /// Deprecated: use `target_partition_size` instead.
    /// Number of partitions to build.
    pub num_partitions: Option<usize>,

    /// Target partition size.
    /// If set, the number of partitions will be computed based on the target partition size.
    /// Otherwise, the `target_partition_size` will be set by index type.
    pub target_partition_size: Option<usize>,

    // ---- kmeans parameters
    /// Max number of iterations to train kmeans.
    pub max_iters: usize,

    /// Use provided IVF centroids.
    pub centroids: Option<Arc<FixedSizeListArray>>,

    /// Retrain centroids.
    /// If true, the centroids will be retrained based on provided `centroids`.
    pub retrain: bool,

    pub sample_rate: usize,

    /// Optional per-step sample rate for streaming IVF kmeans training.
    ///
    /// When set, IVF training loads at most `num_partitions * streaming_sample_rate`
    /// vectors at a time. For `num_partitions > 256`, each chunk is compressed into
    /// a weighted coreset and final centroids are trained with weighted hierarchical
    /// kmeans over the coreset. The coreset budget is also bounded by this rate by
    /// default so large partition counts can control peak memory by lowering
    /// `streaming_sample_rate`. The total number of sampled vectors remains bounded
    /// by `num_partitions * sample_rate`.
    pub streaming_sample_rate: Option<usize>,

    /// Optional coreset rate for streaming IVF kmeans training.
    ///
    /// When set, the final weighted coreset budget is
    /// `num_partitions * streaming_coreset_rate`, independent of
    /// `streaming_sample_rate`. The streaming chunk size is still controlled by
    /// `streaming_sample_rate`.
    pub streaming_coreset_rate: Option<usize>,

    /// Number of extra streaming Lloyd refinement passes to run after streaming
    /// coreset training.
    ///
    /// Each pass reuses the same sampled vectors and only loads
    /// `num_partitions * streaming_sample_rate` raw vectors at a time.  This is
    /// experimental and defaults to 0 to preserve existing behavior.
    pub streaming_refine_passes: usize,

    /// Precomputed partitions file (row_id -> partition_id)
    /// mutually exclusive with `precomputed_shuffle_buffers`
    pub precomputed_partitions_file: Option<String>,

    /// Precomputed shuffle buffers (row_id -> partition_id, pq_code)
    /// mutually exclusive with `precomputed_partitions_file`
    /// requires `centroids` to be set
    ///
    /// The input is expected to be (/dir/to/buffers, [buffer1.lance, buffer2.lance, ...])
    pub precomputed_shuffle_buffers: Option<(Path, Vec<String>)>,

    pub shuffle_partition_batches: usize,

    pub shuffle_partition_concurrency: usize,

    /// Storage options used to load precomputed partitions.
    pub storage_options: Option<HashMap<String, String>>,
}

impl Default for IvfBuildParams {
    fn default() -> Self {
        Self {
            num_partitions: None,
            target_partition_size: None,
            max_iters: 50,
            centroids: None,
            retrain: false,
            sample_rate: 256, // See faiss
            streaming_sample_rate: None,
            streaming_coreset_rate: None,
            streaming_refine_passes: 0,
            precomputed_partitions_file: None,
            precomputed_shuffle_buffers: None,
            shuffle_partition_batches: 1024 * 10,
            shuffle_partition_concurrency: 2,
            storage_options: None,
        }
    }
}

impl IvfBuildParams {
    /// Create a new instance of `IvfBuildParams`.
    pub fn new(num_partitions: usize) -> Self {
        Self {
            num_partitions: Some(num_partitions),
            ..Default::default()
        }
    }

    pub fn with_target_partition_size(target_partition_size: usize) -> Self {
        Self {
            target_partition_size: Some(target_partition_size),
            ..Default::default()
        }
    }

    /// Create a new instance of [`IvfBuildParams`] with centroids.
    pub fn try_with_centroids(
        num_partitions: usize,
        centroids: Arc<FixedSizeListArray>,
    ) -> Result<Self> {
        if num_partitions != centroids.len() {
            return Err(Error::index(format!(
                "IvfBuildParams::try_with_centroids: num_partitions {} != centroids.len() {}",
                num_partitions,
                centroids.len()
            )));
        }
        Ok(Self {
            num_partitions: Some(num_partitions),
            centroids: Some(centroids),
            ..Default::default()
        })
    }

    /// Check the field combinations the precomputed inputs above declare.
    ///
    /// Both precomputed inputs carry partition ids that were assigned against
    /// one particular set of centroids, so neither says anything about an index
    /// whose centroids are trained from the data instead, and supplying both at
    /// once means one of the two assignments is silently unused.
    pub fn validate(&self) -> Result<()> {
        if self.precomputed_shuffle_buffers.is_some() && self.precomputed_partitions_file.is_some()
        {
            return Err(Error::invalid_input(
                "precomputed_shuffle_buffers and precomputed_partitions_file are mutually \
                 exclusive, but both were set",
            ));
        }
        if self.centroids.is_none() {
            if self.precomputed_shuffle_buffers.is_some() {
                return Err(Error::invalid_input(
                    "precomputed_shuffle_buffers requires centroids to be set: the buffers hold \
                     partition ids assigned against the centroids they were built with",
                ));
            }
            if self.precomputed_partitions_file.is_some() {
                return Err(Error::invalid_input(
                    "precomputed_partitions_file requires centroids to be set: the file holds \
                     partition ids assigned against the centroids it was built with",
                ));
            }
        }
        Ok(())
    }
}

pub fn recommended_num_partitions(num_rows: usize, target_partition_size: usize) -> usize {
    // The maximum number of partitions is 4096 to avoid slow KMeans clustering,
    // bump it once we have better clustering algorithms.
    const MAX_PARTITIONS: usize = 4096;
    (num_rows / target_partition_size).clamp(1, MAX_PARTITIONS)
}

/// Load precomputed partitions from disk.
///
/// Currently, because `Dataset` is not cleanly refactored from `lance` to `lance-core`,
/// we have to use `RecordBatchStream` as parameter.
pub async fn load_precomputed_partitions(
    stream: impl RecordBatchStream + Unpin + 'static,
    size_hint: usize,
) -> Result<HashMap<u64, u32>> {
    let partition_lookup = stream
        .try_fold(HashMap::with_capacity(size_hint), |mut lookup, batch| {
            let row_ids: &UInt64Array = batch
                .column_by_name("row_id")
                .expect("malformed partition file: missing row_id column")
                .as_primitive();
            let partitions: &UInt32Array = batch
                .column_by_name("partition")
                .expect("malformed partition file: missing partition column")
                .as_primitive();
            row_ids
                .values()
                .iter()
                .zip(partitions.values().iter())
                .for_each(|(row_id, partition)| {
                    lookup.insert(*row_id, *partition);
                });
            async move { Ok(lookup) }
        })
        .await?;

    Ok(partition_lookup)
}

#[cfg(test)]
mod tests {
    use arrow_array::Float32Array;
    use lance_arrow::FixedSizeListArrayExt;
    use rstest::rstest;

    use super::*;

    fn centroids(num_partitions: usize) -> Arc<FixedSizeListArray> {
        let values = Float32Array::from(vec![0.0_f32; num_partitions * 2]);
        Arc::new(FixedSizeListArray::try_new_from_values(values, 2).unwrap())
    }

    fn buffers() -> (Path, Vec<String>) {
        (Path::from("buffers/data"), vec!["buffer1.lance".to_owned()])
    }

    #[rstest]
    #[case::buffers_and_partitions_file(true, true, true, "mutually exclusive")]
    #[case::buffers_without_centroids(
        true,
        false,
        false,
        "precomputed_shuffle_buffers requires centroids"
    )]
    #[case::partitions_file_without_centroids(
        false,
        true,
        false,
        "precomputed_partitions_file requires centroids"
    )]
    fn test_validate_rejects_precomputed_inputs(
        #[case] with_buffers: bool,
        #[case] with_partitions_file: bool,
        #[case] with_centroids: bool,
        #[case] expected: &str,
    ) {
        let mut params = IvfBuildParams::new(2);
        if with_buffers {
            params.precomputed_shuffle_buffers = Some(buffers());
        }
        if with_partitions_file {
            params.precomputed_partitions_file = Some("partitions.lance".to_owned());
        }
        if with_centroids {
            params.centroids = Some(centroids(2));
        }

        let err = params.validate().unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
        assert!(
            err.to_string().contains(expected),
            "expected the message to mention {expected:?}, got: {err}"
        );
    }

    #[rstest]
    #[case::nothing_precomputed(false, false)]
    #[case::buffers_with_centroids(true, false)]
    #[case::partitions_file_with_centroids(false, true)]
    fn test_validate_accepts_supported_combinations(
        #[case] with_buffers: bool,
        #[case] with_partitions_file: bool,
    ) {
        let mut params = IvfBuildParams::try_with_centroids(2, centroids(2)).unwrap();
        if with_buffers {
            params.precomputed_shuffle_buffers = Some(buffers());
        }
        if with_partitions_file {
            params.precomputed_partitions_file = Some("partitions.lance".to_owned());
        }

        params.validate().unwrap();
    }
}
