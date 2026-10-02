// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Build IVF model

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt32Type, UInt64Type};
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
}

pub fn recommended_num_partitions(num_rows: usize, target_partition_size: usize) -> usize {
    // The maximum number of partitions is 4096 to avoid slow KMeans clustering,
    // bump it once we have better clustering algorithms.
    const MAX_PARTITIONS: usize = 4096;
    (num_rows / target_partition_size).clamp(1, MAX_PARTITIONS)
}

/// The error for a precomputed partitions file whose `name` column is absent or
/// has the wrong physical type, naming what the file has instead.
fn bad_column(schema: &arrow_schema::Schema, name: &str, expected: &str) -> Error {
    let found = match schema.field_with_name(name) {
        Ok(field) => format!("found {}", field.data_type()),
        Err(_) => "no such column".to_string(),
    };
    Error::invalid_input(format!(
        "malformed precomputed partitions file: expected a {expected} '{name}' column, {found}"
    ))
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
        .try_fold(
            HashMap::with_capacity(size_hint),
            |mut lookup, batch| async move {
                let row_ids: &UInt64Array = batch
                    .column_by_name("row_id")
                    .and_then(|col| col.as_primitive_opt::<UInt64Type>())
                    .ok_or_else(|| bad_column(batch.schema_ref(), "row_id", "UInt64"))?;
                let partitions: &UInt32Array = batch
                    .column_by_name("partition")
                    .and_then(|col| col.as_primitive_opt::<UInt32Type>())
                    .ok_or_else(|| bad_column(batch.schema_ref(), "partition", "UInt32"))?;
                // Both columns are read through their values buffer, which
                // ignores validity, so a null would be taken for whatever the
                // buffer happens to hold there. The writer this file comes from
                // marks the fields nullable without using nulls, so reject them
                // rather than assign a row to an arbitrary partition.
                if row_ids.null_count() > 0 || partitions.null_count() > 0 {
                    return Err(Error::invalid_input(format!(
                        "malformed precomputed partitions file: nulls are not allowed \
                         ('row_id' has {}, 'partition' has {})",
                        row_ids.null_count(),
                        partitions.null_count()
                    )));
                }
                lookup.extend(
                    row_ids
                        .values()
                        .iter()
                        .copied()
                        .zip(partitions.values().iter().copied()),
                );
                Ok(lookup)
            },
        )
        .await?;

    Ok(partition_lookup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::RecordBatch;
    use arrow_array::{Float32Array, Int64Array, UInt32Array, UInt64Array};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_io::stream::RecordBatchStreamAdapter;

    fn stream_of(batches: Vec<RecordBatch>) -> impl RecordBatchStream + Unpin {
        let schema = batches[0].schema();
        RecordBatchStreamAdapter::new(schema, futures::stream::iter(batches.into_iter().map(Ok)))
    }

    /// A well-formed batch: `row_id` UInt64, `partition` UInt32, no nulls.
    fn good_batch(row_ids: Vec<u64>, partitions: Vec<u32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("partition", DataType::UInt32, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(row_ids)),
                Arc::new(UInt32Array::from(partitions)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_load_precomputed_partitions_merges_batches() {
        let lookup = load_precomputed_partitions(
            stream_of(vec![
                good_batch(vec![7, 9], vec![1, 2]),
                good_batch(vec![4], vec![3]),
            ]),
            4,
        )
        .await
        .unwrap();
        assert_eq!(lookup.len(), 3);
        assert_eq!(lookup[&7], 1);
        assert_eq!(lookup[&9], 2);
        assert_eq!(lookup[&4], 3);
    }

    #[tokio::test]
    async fn test_load_precomputed_partitions_rejects_wrong_row_id_type() {
        // A file whose row_id column is Int64 used to panic in the unchecked
        // downcast; it must surface as an input error.
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "row_id",
            DataType::Int64,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1i64]))]).unwrap();
        let err = load_precomputed_partitions(stream_of(vec![batch]), 4)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "got: {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("row_id") && msg.contains("Int64"),
            "got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_load_precomputed_partitions_rejects_missing_partition_column() {
        // The branch that replaced `.expect("… missing partition column")`.
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "row_id",
            DataType::UInt64,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(vec![1u64]))]).unwrap();
        let err = load_precomputed_partitions(stream_of(vec![batch]), 4)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "got: {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("partition") && msg.contains("no such column"),
            "got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_load_precomputed_partitions_rejects_wrong_partition_type() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("partition", DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![1u64])),
                Arc::new(Float32Array::from(vec![0.0f32])),
            ],
        )
        .unwrap();
        let err = load_precomputed_partitions(stream_of(vec![batch]), 4)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "got: {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("partition") && msg.contains("Float32"),
            "got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_load_precomputed_partitions_rejects_nulls() {
        // Nulls are read through the values buffer, so a null partition would
        // silently land the row in whatever partition the buffer holds.
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("row_id", DataType::UInt64, true),
            Field::new("partition", DataType::UInt32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![Some(1u64), Some(2)])),
                Arc::new(UInt32Array::from(vec![Some(0u32), None])),
            ],
        )
        .unwrap();
        let err = load_precomputed_partitions(stream_of(vec![batch]), 4)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "got: {err:?}");
        assert!(err.to_string().contains("null"), "got: {err}");
    }
}
