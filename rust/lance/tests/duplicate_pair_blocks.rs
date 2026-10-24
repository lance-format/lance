// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch, RecordBatchIterator, UInt64Array, record_batch};
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::TryStreamExt;
use lance::Dataset;
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance::index::vector::dedup::{DuplicatePairsOptions, find_duplicate_pairs_with_options};
use lance_index::IndexType;
use lance_linalg::distance::DistanceType;
use rstest::rstest;

async fn pairs(mut stream: SendableRecordBatchStream) -> Vec<(u64, u64, u32)> {
    let mut pairs = Vec::new();
    while let Some(batch) = stream.try_next().await.unwrap() {
        let a = batch["row_id_a"]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let b = batch["row_id_b"]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let d = batch["distance"]
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        pairs.extend((0..batch.num_rows()).map(|i| (a.value(i), b.value(i), d.value(i).to_bits())));
    }
    pairs.sort_unstable();
    pairs
}

#[rstest]
#[case::flat_l2(false, DistanceType::L2)]
#[case::rq_l2(true, DistanceType::L2)]
#[case::rq_cosine(true, DistanceType::Cosine)]
#[tokio::test]
async fn blocks_preserve_cross_block_pairs_and_deletions(
    #[case] quantized: bool,
    #[case] metric: DistanceType,
) {
    let ids = record_batch!(("id", Int32, (0..17).collect::<Vec<_>>())).unwrap();
    let vectors = arrow_array::FixedSizeListArray::from_iter_primitive::<
        arrow_array::types::Float32Type,
        _,
        _,
    >(
        (0..17).map(|i| {
            let mut vector = vec![Some(1.0); 8];
            vector[0] = Some((i % 5) as f32);
            Some(vector)
        }),
        8,
    );
    let batch = RecordBatch::try_from_iter([
        ("id", ids.column(0).clone()),
        ("vector", Arc::new(vectors) as arrow_array::ArrayRef),
    ])
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let params = if quantized {
        VectorIndexParams::ivf_rq(1, 1, metric)
    } else {
        VectorIndexParams::ivf_flat(1, metric)
    };
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();
    dataset.delete("id = 3 OR id = 12").await.unwrap();
    let dataset = Arc::new(dataset);
    let expected = pairs(
        find_duplicate_pairs_with_options(
            dataset.clone(),
            "vector",
            0.0,
            DuplicatePairsOptions::default(),
        )
        .await
        .unwrap(),
    )
    .await;
    assert!(!expected.is_empty());
    for memory in [0, 1024 * 1024] {
        for rows in [3, 8, 16] {
            let mut actual = Vec::new();
            for worker in 0..4 {
                actual.extend(
                    pairs(
                        find_duplicate_pairs_with_options(
                            dataset.clone(),
                            "vector",
                            0.0,
                            DuplicatePairsOptions::default()
                                .with_memory_limit(memory)
                                .with_max_concurrency(2)
                                .with_blocking(rows, worker, 4),
                        )
                        .await
                        .unwrap(),
                    )
                    .await,
                );
            }
            actual.sort_unstable();
            assert_eq!(actual, expected, "memory={memory} rows={rows}");
        }
    }
    for (rows, index, count) in [(0, 0, 1), (32, 0, 0), (32, 2, 2)] {
        let error = find_duplicate_pairs_with_options(
            dataset.clone(),
            "vector",
            0.0,
            DuplicatePairsOptions::default().with_blocking(rows, index, count),
        )
        .await
        .err()
        .unwrap();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(error.to_string().contains("block"));
    }
}

#[tokio::test]
async fn large_comparison_blocks_cross_internal_batch_boundaries() {
    let ids = record_batch!(("id", Int32, (0..8195).collect::<Vec<_>>())).unwrap();
    let vectors = arrow_array::FixedSizeListArray::from_iter_primitive::<
        arrow_array::types::Float32Type,
        _,
        _,
    >((0..8195).map(|_| Some(vec![Some(1.0); 8])), 8);
    let batch = RecordBatch::try_from_iter([
        ("id", ids.column(0).clone()),
        ("vector", Arc::new(vectors) as arrow_array::ArrayRef),
    ])
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            None,
            &VectorIndexParams::ivf_flat(1, DistanceType::L2),
            false,
        )
        .await
        .unwrap();
    dataset
        .delete("id NOT IN (0, 8191, 8192, 8194)")
        .await
        .unwrap();
    let dataset = Arc::new(dataset);
    let expected = pairs(
        find_duplicate_pairs_with_options(
            dataset.clone(),
            "vector",
            0.0,
            DuplicatePairsOptions::default(),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_eq!(expected.len(), 6);
    for rows in [8193, 100_000] {
        for memory in [0, 1024 * 1024] {
            let mut actual = Vec::new();
            for worker in 0..3 {
                actual.extend(
                    pairs(
                        find_duplicate_pairs_with_options(
                            dataset.clone(),
                            "vector",
                            0.0,
                            DuplicatePairsOptions::default()
                                .with_memory_limit(memory)
                                .with_blocking(rows, worker, 3),
                        )
                        .await
                        .unwrap(),
                    )
                    .await,
                );
            }
            actual.sort_unstable();
            assert_eq!(actual, expected);
        }
    }
}
