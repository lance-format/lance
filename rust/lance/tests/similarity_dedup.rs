// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    StructArray, UInt64Array,
};
use futures::TryStreamExt;
use lance::Dataset;
use lance::index::vector::similarity_dedup::{
    SimilaritySearchBlocking, find_similar_pairs_in_partition_with_blocking,
};
use lance::index::vector::similarity_dedup::{
    SimilaritySearchOptions, find_similar_pairs_in_partition,
};
use lance::index::{DatasetIndexExt, DatasetIndexInternalExt, vector::VectorIndexParams};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::{IndexType, vector::ivf::IvfBuildParams};
use lance_linalg::distance::DistanceType;

async fn dataset(values: Vec<f32>, centers: Vec<f32>) -> Arc<Dataset> {
    let vectors = FixedSizeListArray::try_new_from_values(Float32Array::from(values), 1).unwrap();
    let batch = RecordBatch::try_from_iter([("vector", Arc::new(vectors) as ArrayRef)]).unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let centers =
        FixedSizeListArray::try_new_from_values(Float32Array::from(centers.clone()), 1).unwrap();
    let ivf = IvfBuildParams::try_with_centroids(centers.len(), Arc::new(centers)).unwrap();
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            None,
            &VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf),
            false,
        )
        .await
        .unwrap();
    Arc::new(dataset)
}

async fn pairs(
    dataset: Arc<Dataset>,
    options: SimilaritySearchOptions,
    threshold: f32,
) -> Vec<(u64, u64)> {
    let meta = dataset.load_indices().await.unwrap();
    let segment = meta[0].uuid;
    let count = if dataset.count_rows(None).await.unwrap() > 100 {
        1
    } else {
        2
    };
    let mut pairs = Vec::new();
    for partition in 0..count {
        let mut stream = find_similar_pairs_in_partition(
            dataset.clone(),
            "vector",
            segment,
            partition,
            threshold,
            options,
        )
        .await
        .unwrap();
        while let Some(batch) = stream.try_next().await.unwrap() {
            let a = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let b = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            pairs.extend(a.values().iter().zip(b.values()).map(|(&a, &b)| (a, b)));
        }
    }
    pairs
}

#[tokio::test]
async fn auto_probes_cross_centroid_boundary() {
    let dataset = dataset(vec![-0.01, 0.01, -1.0, 1.0], vec![-1.0, 1.0]).await;
    let fixed = pairs(
        dataset.clone(),
        SimilaritySearchOptions {
            nprobes: Some(1),
            max_nprobes: None,
        },
        0.001,
    )
    .await;
    assert!(fixed.is_empty());
    let auto = pairs(dataset, SimilaritySearchOptions::default(), 0.001).await;
    assert_eq!(
        auto.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from([(0, 1)])
    );
}

#[tokio::test]
async fn radius_has_no_top_k_truncation() {
    let dataset = dataset(vec![1.0; 151], vec![1.0]).await;
    let pairs = pairs(dataset, SimilaritySearchOptions::default(), 0.0).await;
    assert_eq!(pairs.len(), 151 * 150);
    assert_eq!(
        pairs.into_iter().collect::<BTreeSet<_>>().len(),
        151 * 150 / 2
    );
    assert!(
        SimilaritySearchOptions {
            nprobes: Some(0),
            max_nprobes: None
        }
        .validate()
        .is_err()
    );
    assert!(
        SimilaritySearchOptions {
            nprobes: Some(3),
            max_nprobes: Some(2)
        }
        .validate()
        .is_err()
    );
}

#[tokio::test]
async fn source_and_target_blocks_preserve_all_cross_block_edges() {
    for quantization in ["flat", "rq", "pq"] {
        let vectors =
            FixedSizeListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
                (0..19).map(|i| {
                    let mut vector = vec![Some(1.0); 8];
                    vector[0] = Some((i % 5) as f32);
                    Some(vector)
                }),
                8,
            );
        let batch =
            RecordBatch::try_from_iter([("vector", Arc::new(vectors) as ArrayRef)]).unwrap();
        let mut ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            "memory://",
            None,
        )
        .await
        .unwrap();
        let params = match quantization {
            "rq" => VectorIndexParams::ivf_rq(1, 4, DistanceType::L2),
            "pq" => VectorIndexParams::ivf_pq(1, 4, 4, DistanceType::L2, 2),
            _ => VectorIndexParams::ivf_flat(1, DistanceType::L2),
        };
        ds.create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();
        let ds = Arc::new(ds);
        let segment = ds.load_indices().await.unwrap()[0].uuid;
        let index = ds
            .open_vector_index(
                "vector",
                &segment,
                &lance_index::metrics::NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let mut query = lance_index::vector::Query {
            column: "vector".into(),
            key: Arc::new(Float32Array::from(vec![
                2.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ])),
            k: 1,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            search_effort: 0.5,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::L2),
            use_index: true,
            query_parallelism: 1,
            dist_q_c: 0.0,
            approx_mode: lance_index::vector::ApproxMode::Accurate,
        };
        query.dist_q_c = index.find_partitions(&query).unwrap().1.value(0);
        let measured = |batch: RecordBatch| {
            let ids = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let distances = batch
                .column(0)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();
            ids.values()
                .iter()
                .zip(distances.values())
                .map(|(&id, &distance)| (id, distance.to_bits()))
                .collect::<Vec<_>>()
        };
        let full = index
            .range_search_in_partition(
                0,
                &query,
                1000.0,
                Arc::new(lance_index::prefilter::NoFilter),
                &lance_index::metrics::NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let mut full = measured(full);
        full.sort_unstable();
        for rows in [1, 3, 8] {
            let mut actual = Vec::new();
            for start in (0..19).step_by(rows) {
                let batch = index
                    .range_search_in_partition_range(
                        0,
                        start..(start + rows).min(19),
                        &query,
                        1000.0,
                        Arc::new(lance_index::prefilter::NoFilter),
                        &lance_index::metrics::NoOpMetricsCollector,
                    )
                    .await
                    .unwrap();
                actual.extend(measured(batch));
            }
            actual.sort_unstable();
            assert_eq!(actual, full, "quantization={quantization},rows={rows}");
        }
        let mut expected = Vec::new();
        let mut stream = find_similar_pairs_in_partition(
            ds.clone(),
            "vector",
            segment,
            0,
            1000.0,
            SimilaritySearchOptions::default(),
        )
        .await
        .unwrap();
        while let Some(batch) = stream.try_next().await.unwrap() {
            let a = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let b = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            expected.extend(a.values().iter().zip(b.values()).map(|(&a, &b)| (a, b)));
        }
        expected.sort_unstable();
        assert_eq!(expected.len(), 19 * 18);
        for rows in [1, 3, 8] {
            let mut actual = Vec::new();
            for worker in 0..4 {
                let mut stream = find_similar_pairs_in_partition_with_blocking(
                    ds.clone(),
                    "vector",
                    segment,
                    0,
                    1000.0,
                    SimilaritySearchOptions::default(),
                    SimilaritySearchBlocking {
                        rows_per_block: rows,
                        worker_index: worker,
                        worker_count: 4,
                    },
                )
                .await
                .unwrap();
                while let Some(batch) = stream.try_next().await.unwrap() {
                    let a = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap();
                    let b = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap();
                    actual.extend(a.values().iter().zip(b.values()).map(|(&a, &b)| (a, b)));
                }
            }
            actual.sort_unstable();
            assert_eq!(actual, expected, "quantization={quantization},rows={rows}");
        }
        assert!(
            SimilaritySearchBlocking {
                rows_per_block: 0,
                worker_index: 0,
                worker_count: 1
            }
            .validate()
            .is_err()
        );
        assert!(
            SimilaritySearchBlocking {
                rows_per_block: 8,
                worker_index: 4,
                worker_count: 4
            }
            .validate()
            .is_err()
        );
    }
}

#[tokio::test]
async fn nested_vector_projection_is_resolved_for_both_endpoints() {
    let vectors = FixedSizeListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
        [
            Some(vec![Some(1.0), Some(1.0)]),
            Some(vec![Some(1.0), Some(1.0)]),
            Some(vec![Some(4.0), Some(1.0)]),
        ],
        2,
    );
    let nested = StructArray::from(vec![(
        Arc::new(arrow_schema::Field::new(
            "vector",
            vectors.data_type().clone(),
            true,
        )),
        Arc::new(vectors) as ArrayRef,
    )]);
    let batch = RecordBatch::try_from_iter([("payload", Arc::new(nested) as ArrayRef)]).unwrap();
    let mut ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    ds.create_index(
        &["payload.vector"],
        IndexType::Vector,
        None,
        &VectorIndexParams::ivf_flat(1, DistanceType::L2),
        false,
    )
    .await
    .unwrap();
    let segment = ds.load_indices().await.unwrap()[0].uuid;
    let mut output = find_similar_pairs_in_partition(
        Arc::new(ds),
        "payload.vector",
        segment,
        0,
        0.0,
        SimilaritySearchOptions::default(),
    )
    .await
    .unwrap();
    let mut rows = 0;
    while let Some(batch) = output.try_next().await.unwrap() {
        rows += batch.num_rows();
    }
    assert_eq!(rows, 2);
}

#[tokio::test]
async fn automatic_probe_budget_is_bounded_and_can_be_raised() {
    let ds = dataset(vec![-1.0, 1.0, 3.0], vec![-1.0, 1.0, 3.0]).await;
    let segment = ds.load_indices().await.unwrap()[0].uuid;
    for (maximum, expect_outer_pair) in [(None, false), (Some(3), true)] {
        let mut outer_pair = false;
        for partition in 0..3 {
            let mut stream = find_similar_pairs_in_partition(
                ds.clone(),
                "vector",
                segment,
                partition,
                16.0,
                SimilaritySearchOptions {
                    nprobes: None,
                    max_nprobes: maximum,
                },
            )
            .await
            .unwrap();
            while let Some(batch) = stream.try_next().await.unwrap() {
                let a = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                let b = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                outer_pair |= a
                    .values()
                    .iter()
                    .zip(b.values())
                    .any(|(&a, &b)| a == 0 && b == 2);
            }
        }
        assert_eq!(outer_pair, expect_outer_pair);
    }
}
