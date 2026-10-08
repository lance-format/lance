// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tiny `IVF_HNSW_RQ` example. Writes a toy dataset, builds the index, searches.
//!
//! Build (any supported `num_bits`, including 1), search, print `IVF_HNSW_RQ`, exit 0:
//! ```text
//! cargo run --release -p lance-examples --example ivf_hnsw_rq
//! cargo run --release -p lance-examples --example ivf_hnsw_rq -- --num-bits 1
//! ```
//!
//! Python: `ds.create_index("vector", "IVF_HNSW_RQ", num_partitions=1, num_bits=2)`.

#![allow(clippy::print_stdout)]

use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, FieldRef, Schema};
use clap::Parser;
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::vector::bq::{RQBuildParams, RQRotationType};
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_linalg::distance::MetricType;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

const N: usize = 64;
const DIM: usize = 16;

#[derive(Parser, Debug)]
#[command(about = "Build and search a tiny IVF_HNSW_RQ index")]
struct Args {
    /// RQ bits per dimension. 1-bit is supported (no ex-code rerank layer).
    #[arg(long, default_value_t = 2)]
    num_bits: u8,
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

async fn run() -> lance::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let tmp = TempStrDir::default();
    let uri = format!("{}/ivf_hnsw_rq", tmp.as_str());

    let mut rng = StdRng::seed_from_u64(20260916);
    let vectors: Vec<f32> = (0..N * DIM)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let mut dataset = write_dataset(&uri, &vectors).await?;

    let params = VectorIndexParams::with_ivf_hnsw_rq_params(
        MetricType::L2,
        IvfBuildParams {
            num_partitions: Some(1),
            ..Default::default()
        },
        HnswBuildParams::default()
            .max_level(2)
            .num_edges(4)
            .ef_construction(16),
        RQBuildParams::with_rotation_type(args.num_bits, RQRotationType::Fast),
    );

    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("ivf_hnsw_rq".to_string()),
            &params,
            false,
        )
        .await?;

    println!("index_type={}", IndexType::IvfHnswRq);

    let stats = dataset.index_statistics("ivf_hnsw_rq").await?;
    println!("{stats}");

    let query = Float32Array::from(vectors[..DIM].to_vec());
    let results = dataset
        .scan()
        .nearest("vector", &query, 1)?
        .nprobes(1)
        .ef(16)
        .try_into_batch()
        .await?;
    println!("hits={}", results.num_rows());
    Ok(())
}

async fn write_dataset(uri: &str, vectors: &[f32]) -> lance::Result<Dataset> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            FieldRef::new(Field::new("item", DataType::Float32, true)),
            DIM as i32,
        ),
        false,
    )]));
    let fsl =
        FixedSizeListArray::try_new_from_values(Float32Array::from(vectors.to_vec()), DIM as i32)?;
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)])?;
    let reader = RecordBatchIterator::new(std::iter::once(Ok(batch)), schema);
    Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: N,
            mode: WriteMode::Create,
            ..Default::default()
        }),
    )
    .await
}
