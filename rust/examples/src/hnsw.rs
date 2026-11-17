// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! HNSW is a graph based algorithm for approximate neighbor search in high-dimensional spaces.
//! In this example, we will demonstrate how to build HNSW vector indexing against a Lance dataset.
//! run with `cargo run -v --package lance-examples --example hnsw``
// linked to `docs/examples/Rust/hnsw.rst`
#![allow(clippy::print_stdout)]
use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{AsArray, FixedSizeListBuilder, Float32Builder, types::UInt64Type};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::record_batch::RecordBatchIterator;
use lance::Dataset;
use lance::dataset::ProjectionRequest;
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_core::ROW_ID;
use lance_index::{
    IndexType,
    vector::{hnsw::builder::HnswBuildParams, ivf::IvfBuildParams, sq::builder::SQBuildParams},
};
use lance_linalg::distance::MetricType;

pub async fn create_test_vector_dataset(output: &str, num_rows: usize, dim: i32) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
        false,
    )]));

    let mut batches = Vec::new();

    // Create a few batches
    for _ in 0..2 {
        let v_builder = Float32Builder::new();
        let mut list_builder = FixedSizeListBuilder::new(v_builder, dim);

        for _ in 0..num_rows {
            for _ in 0..dim {
                list_builder.values().append_value(rand::random::<f32>());
            }
            list_builder.append(true);
        }
        let array = Arc::new(list_builder.finish());
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        batches.push(batch);
    }
    let batch_reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    println!("Writing dataset to {}", output);
    Dataset::write(batch_reader, output, None).await.unwrap();
}

#[tokio::main]
async fn main() {
    let uri: Option<String> = None; // None means generate test data
    let column = "vector";
    let ef = 100;
    let max_edges = 30;
    let max_level = 7;
    let num_partitions = 2;

    // 1. Generate a synthetic test data of specified dimensions
    let mut dataset = match uri.as_deref() {
        None => {
            println!("No uri is provided, generating test dataset...");
            let output = "test_vectors.lance";
            create_test_vector_dataset(output, 1000, 64).await;
            Dataset::open(output).await.expect("Failed to open dataset")
        }
        Some(uri) => Dataset::open(uri).await.expect("Failed to open dataset"),
    };

    println!("Dataset schema: {:#?}", dataset.schema());
    let first_row = dataset
        .take(
            &[0],
            ProjectionRequest::from_columns([column], dataset.schema()),
        )
        .await
        .unwrap();
    let q = first_row
        .column_by_name(column)
        .unwrap()
        .as_fixed_size_list()
        .value(0);
    let k = 10;

    // Compute exact L2 ground truth with a brute-force scan (`use_index(false)`).
    let gt_batch = dataset
        .scan()
        .project::<String>(&[])
        .unwrap()
        .with_row_id()
        .nearest(column, q.as_ref(), k)
        .unwrap()
        .use_index(false)
        .try_into_batch()
        .await
        .unwrap();
    let gt: HashSet<u64> = gt_batch
        .column_by_name(ROW_ID)
        .unwrap()
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect();

    for ef_construction in [30, 50, 100] {
        let now = std::time::Instant::now();
        // 2. Build a persistent IVF_HNSW_SQ vector index on the dataset
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            MetricType::L2,
            IvfBuildParams::new(num_partitions),
            HnswBuildParams::default()
                .max_level(max_level)
                .num_edges(max_edges)
                .ef_construction(ef_construction),
            SQBuildParams::default(),
        );
        dataset
            .create_index(&[column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        let construct_time = now.elapsed().as_secs_f32();
        let now = std::time::Instant::now();
        // 3. Query the dataset through Scanner::nearest with explicit HNSW `ef`
        let result_batch = dataset
            .scan()
            .project::<String>(&[])
            .unwrap()
            .with_row_id()
            .nearest(column, q.as_ref(), k)
            .unwrap()
            .ef(ef)
            .minimum_nprobes(num_partitions)
            .try_into_batch()
            .await
            .unwrap();
        let results: HashSet<u64> = result_batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect();
        let search_time = now.elapsed().as_micros();
        println!(
            "level={}, ef_construct={}, ef={} recall={}: construct={:.3}s search={:.3} us",
            max_level,
            ef_construction,
            ef,
            results.intersection(&gt).count() as f32 / k as f32,
            construct_time,
            search_time
        );
    }
}
