// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One-shot benchmark for batch and streaming Hamming IVF training.
//!
//! The benchmark intentionally runs one operation per process so `/usr/bin/time -v`
//! can measure peak RSS without Criterion or dataset generation in the process.

#![allow(clippy::print_stdout)]
#![recursion_limit = "256"]

use std::error::Error;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt8Type;
use arrow_array::{Array, FixedSizeListArray, RecordBatch, RecordBatchIterator, UInt8Array};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Schema};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::index::vector::ivf::build_ivf_model;
use lance_arrow::FixedSizeListArrayExt;
use lance_index::progress::noop_progress;
use lance_index::vector::ivf::IvfBuildParams;
use lance_linalg::distance::MetricType;
use lance_linalg::distance::hamming::hamming;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const DEFAULT_DIMENSION_BITS: usize = 256;
const SAMPLE_RATE: usize = 256;
const STREAMING_SAMPLE_RATE: usize = 64;
const STREAMING_CORESET_RATE: usize = 16;
const BATCH_ROWS: usize = 65_536;
const MAX_ROWS_PER_FILE: usize = 10_000_000;

fn dimension_bytes(dimension_bits: usize) -> Result<usize, Box<dyn Error>> {
    if dimension_bits == 0 || !dimension_bits.is_multiple_of(8) {
        return Err(format!(
            "dimension_bits must be a positive multiple of 8, got {dimension_bits}"
        )
        .into());
    }
    let dimension_bytes = dimension_bits / 8;
    if i32::try_from(dimension_bytes).is_err() {
        return Err(
            format!("dimension_bits is too large for FixedSizeList: {dimension_bits}").into(),
        );
    }
    Ok(dimension_bytes)
}

fn schema(dimension_bytes: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            FieldRef::new(Field::new("item", DataType::UInt8, true)),
            dimension_bytes as i32,
        ),
        false,
    )]))
}

fn validate_dataset_dimension(
    dataset: &Dataset,
    expected_dimension_bytes: usize,
) -> Result<(), Box<dyn Error>> {
    let field = dataset
        .schema()
        .field("vector")
        .ok_or("dataset does not contain a vector column")?;
    match field.data_type() {
        DataType::FixedSizeList(item, dimension)
            if item.data_type() == &DataType::UInt8
                && usize::try_from(dimension) == Ok(expected_dimension_bytes) =>
        {
            Ok(())
        }
        data_type => Err(format!(
            "vector column must be FixedSizeList<UInt8, {expected_dimension_bytes}>, got {data_type}"
        )
        .into()),
    }
}

fn generate_batch(
    prototypes: &[u8],
    num_prototypes: usize,
    dimension_bytes: usize,
    flip_probability: f64,
    num_rows: usize,
    rng: &mut SmallRng,
    schema: &Arc<Schema>,
) -> Result<RecordBatch, ArrowError> {
    let value_capacity = num_rows.checked_mul(dimension_bytes).ok_or_else(|| {
        ArrowError::InvalidArgumentError(format!(
            "num_rows * dimension_bytes overflow: {num_rows} * {dimension_bytes}"
        ))
    })?;
    let mut values = Vec::with_capacity(value_capacity);
    for _ in 0..num_rows {
        let prototype = rng.random_range(0..num_prototypes);
        let start = prototype * dimension_bytes;
        for &prototype_byte in &prototypes[start..start + dimension_bytes] {
            let mut flip_mask = 0_u8;
            for bit in 0..8 {
                if rng.random_bool(flip_probability) {
                    flip_mask |= 1 << bit;
                }
            }
            values.push(prototype_byte ^ flip_mask);
        }
    }
    let vectors =
        FixedSizeListArray::try_new_from_values(UInt8Array::from(values), dimension_bytes as i32)?;
    RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)])
}

async fn generate_dataset(
    uri: &str,
    num_rows: usize,
    num_prototypes: usize,
    flip_probability: f64,
    seed: u64,
    dimension_bytes: usize,
) -> Result<(), Box<dyn Error>> {
    if Path::new(uri).exists() {
        return Err(format!("dataset path already exists: {uri}").into());
    }
    if num_rows == 0 {
        return Err("num_rows must be greater than zero".into());
    }
    if num_prototypes == 0 {
        return Err("num_prototypes must be greater than zero".into());
    }
    if !(0.0..=1.0).contains(&flip_probability) || !flip_probability.is_finite() {
        return Err(format!(
            "flip_probability must be finite and in [0, 1], got {flip_probability}"
        )
        .into());
    }

    let prototype_values = num_prototypes.checked_mul(dimension_bytes).ok_or_else(|| {
        format!("num_prototypes * dimension_bytes overflow: {num_prototypes} * {dimension_bytes}")
    })?;
    let mut rng = SmallRng::seed_from_u64(seed);
    let prototypes = (0..prototype_values)
        .map(|_| rng.random::<u8>())
        .collect::<Vec<_>>();
    let schema = schema(dimension_bytes);
    let batch_schema = schema.clone();
    let mut remaining_rows = num_rows;
    let batches = std::iter::from_fn(move || {
        if remaining_rows == 0 {
            return None;
        }
        let rows = remaining_rows.min(BATCH_ROWS);
        remaining_rows -= rows;
        Some(generate_batch(
            &prototypes,
            num_prototypes,
            dimension_bytes,
            flip_probability,
            rows,
            &mut rng,
            &batch_schema,
        ))
    });

    let reader = RecordBatchIterator::new(batches, schema);
    let dataset = Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: num_rows.min(MAX_ROWS_PER_FILE),
            max_rows_per_group: BATCH_ROWS,
            ..Default::default()
        }),
    )
    .await?;
    let written_rows = dataset.count_rows(None).await?;
    if written_rows != num_rows {
        return Err(format!(
            "generated row count mismatch: expected {num_rows}, got {written_rows}"
        )
        .into());
    }
    println!(
        "generated uri={uri} rows={num_rows} dimension_bytes={dimension_bytes} prototypes={num_prototypes} flip_probability={flip_probability} seed={seed}"
    );
    Ok(())
}

async fn train(
    uri: &str,
    mode: &str,
    num_partitions: usize,
    refine_passes: usize,
    centroids_path: &str,
    dimension_bytes: usize,
) -> Result<(), Box<dyn Error>> {
    if num_partitions == 0 {
        return Err("num_partitions must be greater than zero".into());
    }
    let dataset = Dataset::open(uri).await?;
    validate_dataset_dimension(&dataset, dimension_bytes)?;
    let mut params = IvfBuildParams::new(num_partitions);
    params.sample_rate = SAMPLE_RATE;
    if mode == "stream" {
        params.streaming_sample_rate = Some(STREAMING_SAMPLE_RATE);
        params.streaming_coreset_rate = Some(STREAMING_CORESET_RATE);
        params.streaming_refine_passes = refine_passes;
    } else if mode != "batch" {
        return Err(format!("unknown training mode: {mode}").into());
    } else if refine_passes != 0 {
        return Err("batch mode does not support streaming refinement passes".into());
    }

    let start = Instant::now();
    let model = build_ivf_model(
        &dataset,
        "vector",
        dimension_bytes,
        MetricType::Hamming,
        &params,
        None,
        noop_progress(),
    )
    .await?;
    let train_seconds = start.elapsed().as_secs_f64();
    let centroids = model
        .centroids_array()
        .ok_or("trained IVF model has no centroids")?;
    let values = centroids.values().as_primitive::<UInt8Type>();
    fs::write(centroids_path, values.values())?;
    println!(
        "mode={mode} k={num_partitions} refine_passes={refine_passes} status=ok train_seconds={train_seconds:.6} centroids={} dimension_bytes={}",
        centroids.len(),
        centroids.value_length()
    );
    Ok(())
}

async fn exact_loss(
    uri: &str,
    num_partitions: usize,
    evaluation_rows: usize,
    centroids_path: &str,
    dimension_bytes: usize,
) -> Result<(), Box<dyn Error>> {
    if num_partitions == 0 {
        return Err("num_partitions must be greater than zero".into());
    }
    if evaluation_rows == 0 {
        return Err("evaluation_rows must be greater than zero".into());
    }
    let dataset = Dataset::open(uri).await?;
    validate_dataset_dimension(&dataset, dimension_bytes)?;
    let mut scanner = dataset.scan();
    scanner.project(&["vector"])?;
    scanner.limit(Some(i64::try_from(evaluation_rows)?), None)?;
    let batch = scanner.try_into_batch().await?;
    let vectors = batch["vector"].as_fixed_size_list();
    let vector_values = vectors.values().as_primitive::<UInt8Type>();
    let centroids = fs::read(centroids_path)?;
    let expected_centroid_bytes = num_partitions.checked_mul(dimension_bytes).ok_or_else(|| {
        format!("num_partitions * dimension_bytes overflow: {num_partitions} * {dimension_bytes}")
    })?;
    if centroids.len() != expected_centroid_bytes {
        return Err(format!(
            "centroid file has {} bytes, expected {}",
            centroids.len(),
            expected_centroid_bytes
        )
        .into());
    }

    let start = Instant::now();
    let loss = vector_values
        .values()
        .par_chunks_exact(dimension_bytes)
        .map(|vector| {
            centroids
                .chunks_exact(dimension_bytes)
                .fold(u64::MAX, |nearest, centroid| {
                    nearest.min(hamming(vector, centroid) as u64)
                })
        })
        .sum::<u64>();
    println!(
        "status=ok k={num_partitions} rows={} exact_loss={loss} mean_loss={:.6} eval_seconds={:.6}",
        vectors.len(),
        loss as f64 / vectors.len() as f64,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn usage() -> &'static str {
    "usage: hamming_ivf_training generate <dataset> <rows> <prototypes> <flip_probability> <seed> [dimension_bits] | train <dataset> <stream|batch> <k> <refine_passes> <centroids> [dimension_bits] | loss <dataset> <k> <evaluation_rows> <centroids> [dimension_bits]"
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let runtime = tokio::runtime::Runtime::new()?;
    let default_dimension_bytes = dimension_bytes(DEFAULT_DIMENSION_BITS)?;
    match args.as_slice() {
        [_, command, uri, rows, prototypes, flip_probability, seed] if command == "generate" => {
            runtime.block_on(generate_dataset(
                uri,
                rows.parse()?,
                prototypes.parse()?,
                flip_probability.parse()?,
                seed.parse()?,
                default_dimension_bytes,
            ))
        }
        [
            _,
            command,
            uri,
            rows,
            prototypes,
            flip_probability,
            seed,
            dimension_bits,
        ] if command == "generate" => runtime.block_on(generate_dataset(
            uri,
            rows.parse()?,
            prototypes.parse()?,
            flip_probability.parse()?,
            seed.parse()?,
            dimension_bytes(dimension_bits.parse()?)?,
        )),
        [
            _,
            command,
            uri,
            mode,
            num_partitions,
            refine_passes,
            centroids,
        ] if command == "train" => runtime.block_on(train(
            uri,
            mode,
            num_partitions.parse()?,
            refine_passes.parse()?,
            centroids,
            default_dimension_bytes,
        )),
        [
            _,
            command,
            uri,
            mode,
            num_partitions,
            refine_passes,
            centroids,
            dimension_bits,
        ] if command == "train" => runtime.block_on(train(
            uri,
            mode,
            num_partitions.parse()?,
            refine_passes.parse()?,
            centroids,
            dimension_bytes(dimension_bits.parse()?)?,
        )),
        [_, command, uri, num_partitions, evaluation_rows, centroids] if command == "loss" => {
            runtime.block_on(exact_loss(
                uri,
                num_partitions.parse()?,
                evaluation_rows.parse()?,
                centroids,
                default_dimension_bytes,
            ))
        }
        [
            _,
            command,
            uri,
            num_partitions,
            evaluation_rows,
            centroids,
            dimension_bits,
        ] if command == "loss" => runtime.block_on(exact_loss(
            uri,
            num_partitions.parse()?,
            evaluation_rows.parse()?,
            centroids,
            dimension_bytes(dimension_bits.parse()?)?,
        )),
        _ => Err(usage().into()),
    }
}
