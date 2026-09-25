// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use super::utils::AllocTracker;
use all_asserts::assert_le;
use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchIterator, types::Float32Type};
use arrow_schema::{DataType, Field, Schema};
use lance::Dataset;
use lance::dataset::{InsertBuilder, WriteMode, WriteParams};
use lance::index::vector::VectorIndexParams;
use lance::index::vector::utils::maybe_sample_training_data;
use lance::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::IndexType;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::optimize::OptimizeOptions;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_io::object_store::ObjectStoreParams;
use lance_linalg::distance::MetricType;

#[tokio::test]
async fn test_nullable_fragment_sampling_memory_stays_bounded() {
    let dim = 1024;
    let num_fragments = 4;
    let rows_per_fragment = 8_192;
    let sample_size = 32;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vec",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
        true,
    )]));

    let batches = (0..num_fragments)
        .map(|seed| {
            let values = lance_testing::datagen::generate_random_array_with_seed::<Float32Type>(
                rows_per_fragment * dim as usize,
                [seed as u8; 32],
            );
            let vectors = Arc::new(
                arrow_array::FixedSizeListArray::try_new_from_values(values, dim).unwrap(),
            ) as ArrayRef;
            RecordBatch::try_new(schema.clone(), vec![vectors]).unwrap()
        })
        .collect::<Vec<_>>();

    let tmp_dir = tempfile::tempdir().unwrap();
    let uri = tmp_dir.path().to_str().unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new(batches.into_iter().map(Ok), schema),
        uri,
        Some(WriteParams {
            max_rows_per_file: rows_per_fragment,
            max_rows_per_group: rows_per_fragment,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let fragment_ids = dataset
        .get_fragments()
        .into_iter()
        .take(2)
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();

    let alloc_tracker = AllocTracker::new();
    let training_data = {
        let _guard = alloc_tracker.enter();
        maybe_sample_training_data(&dataset, "vec", sample_size, Some(&fragment_ids))
            .await
            .unwrap()
    };
    let stats = alloc_tracker.stats();

    assert_eq!(training_data.len(), sample_size);

    // A full scan of the selected fragments would need at least:
    // 2 fragments * 8192 rows * 1024 dims * 4 bytes = 64 MiB
    // Keep a generous ceiling well below that lower bound so the test remains
    // stable while still catching regressions back to eager materialization.
    assert_le!(
        stats.max_bytes_allocated,
        24 * 1024 * 1024,
        "nullable fragment sampling allocated too much memory: {:?}",
        stats
    );
}

/// Vectors of `dim` f32 values, `num_rows` of them, seeded by `seed`.
fn random_vector_batch(schema: &Arc<Schema>, num_rows: usize, dim: i32, seed: u8) -> RecordBatch {
    let values = lance_testing::datagen::generate_random_array_with_seed::<Float32Type>(
        num_rows * dim as usize,
        [seed; 32],
    );
    let vectors =
        Arc::new(arrow_array::FixedSizeListArray::try_new_from_values(values, dim).unwrap())
            as ArrayRef;
    RecordBatch::try_new(schema.clone(), vec![vectors]).unwrap()
}

#[tokio::test]
async fn test_ivf_split_reshuffle_memory_stays_bounded() {
    // A split re-reads the raw vectors of every partition it touches. With two
    // partitions of ~64Ki rows each and a target of 8Ki rows, both are split
    // and every one of the 128Ki rows (512 MiB of vectors) is re-read.
    let dim = 1024;
    let rows_per_batch = 8_192;
    let num_batches = 16;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vec",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
        false,
    )]));
    // Cloud stores report a 64 KiB block size; the re-read must not size its
    // batches from it.
    let store_params = ObjectStoreParams {
        block_size: Some(64 * 1024),
        ..Default::default()
    };

    let tmp_dir = tempfile::tempdir().unwrap();
    let uri = tmp_dir.path().to_str().unwrap();
    let batches = (0..num_batches).map(|seed| {
        Ok(random_vector_batch(
            &schema,
            rows_per_batch,
            dim,
            seed as u8,
        ))
    });
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(batches, schema.clone()),
        uri,
        Some(WriteParams {
            store_params: Some(store_params.clone()),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let mut ivf_params = IvfBuildParams::new(2);
    ivf_params.target_partition_size = Some(8_192);
    // 4-bit codes keep PQ training and encoding cheap in unoptimized builds.
    let pq_params = PQBuildParams {
        num_sub_vectors: 16,
        num_bits: 4,
        max_iters: 2,
        sample_rate: 16,
        ..Default::default()
    };
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("idx".into()),
            &VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params),
            false,
        )
        .await
        .unwrap();
    // New rows give the default optimize something to merge, which is when it
    // rebalances oversized partitions.
    let mut dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            store_params: Some(store_params),
            ..Default::default()
        })
        .execute(vec![random_vector_batch(&schema, 16, dim, u8::MAX)])
        .await
        .unwrap();

    let alloc_tracker = AllocTracker::new();
    {
        let _guard = alloc_tracker.enter();
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
    }
    let stats = alloc_tracker.stats();

    let indices = dataset.load_indices_by_name("idx").await.unwrap();
    let index = dataset
        .open_vector_index("vec", &indices[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        index.ivf_model().num_partitions() > 2,
        "the optimize must split the oversized partitions"
    );

    // Holding every re-read row at once needs 512 MiB. Keep the ceiling well
    // below that so a regression to unbounded re-reads fails.
    assert_le!(
        stats.max_bytes_allocated,
        256 * 1024 * 1024,
        "split reshuffle allocated too much memory: {:?}",
        stats
    );
}
