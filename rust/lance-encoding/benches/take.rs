// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{Field, Schema};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::StreamExt;
use lance_core::cache::LanceCache;
use lance_encoding::{
    BufferScheduler, EncodingsIo,
    decoder::{
        DecodeBatchScheduler, DecoderConfig, DecoderPlugins, EncodedBatchLayout, FilterExpression,
        create_decode_stream,
    },
    encoder::{EncodingOptions, encode_batch},
};
use tokio::sync::mpsc::unbounded_channel;

pub mod common;
use common::encoding_strategy;

const NUM_ROWS: usize = 262_144;
const NUM_TAKES: usize = 512;

fn metadata(bss_on: bool) -> HashMap<String, String> {
    let mut metadata = HashMap::from([
        (
            "lance-encoding:structural-encoding".to_string(),
            "miniblock".to_string(),
        ),
        // Disable dictionary encoding so the bench exercises the primitive
        // value pages (bitpack, RLE, BSS, raw) selected by the encoder.
        (
            "lance-encoding:dict-divisor".to_string(),
            "100000".to_string(),
        ),
        // Explicit compression is required for the miniblock path to wrap
        // pages in a general-purpose codec (see compression.rs BSS gating).
        ("lance-encoding:compression".to_string(), "zstd".to_string()),
        (
            "lance-encoding:compression-level".to_string(),
            "3".to_string(),
        ),
    ]);
    // Automatic BSS would hide the integer bitpacking and RLE paths.
    metadata.insert(
        "lance-encoding:bss".to_string(),
        if bss_on { "on" } else { "off" }.to_string(),
    );
    metadata
}

fn corpora() -> Vec<(&'static str, ArrayRef, bool)> {
    vec![
        (
            "i64_narrow",
            Arc::new(Int64Array::from_iter_values(
                (0..NUM_ROWS).map(|i| (i % 65_536) as i64),
            )),
            false,
        ),
        (
            "i64_runs",
            Arc::new(Int64Array::from_iter_values(
                (0..NUM_ROWS).map(|i| (i / 64) as i64),
            )),
            false,
        ),
        (
            "i64_random",
            Arc::new(Int64Array::from_iter_values(
                (0..NUM_ROWS).map(|_| rand::random::<i64>()),
            )),
            false,
        ),
        (
            "f64_clustered",
            Arc::new(Float64Array::from_iter_values(
                (0..NUM_ROWS).map(|i| 1.0 + (i % 1024) as f64 * 1e-9),
            )),
            true,
        ),
        (
            "f64_random",
            Arc::new(Float64Array::from_iter_values(
                (0..NUM_ROWS).map(|_| rand::random::<f64>()),
            )),
            false,
        ),
    ]
}

async fn decode_take(
    encoded: &lance_encoding::encoder::EncodedBatch,
    indices: &[u64],
    cache: Arc<LanceCache>,
) -> RecordBatch {
    let io_scheduler = Arc::new(BufferScheduler::new(encoded.data.clone())) as Arc<dyn EncodingsIo>;
    let filter = FilterExpression::no_filter();
    let mut decode_scheduler = DecodeBatchScheduler::try_new(
        encoded.schema.as_ref(),
        &encoded.top_level_columns,
        &encoded.page_table,
        &vec![],
        encoded.num_rows,
        Arc::<DecoderPlugins>::default(),
        io_scheduler.clone(),
        cache,
        &filter,
        &DecoderConfig::default(),
    )
    .await
    .unwrap();
    let (tx, rx) = unbounded_channel();
    decode_scheduler.schedule_take(indices, &filter, tx, io_scheduler);
    let mut stream = create_decode_stream(
        &encoded.schema,
        indices.len() as u64,
        indices.len() as u32,
        true,
        false,
        true,
        rx,
        None,
    )
    .unwrap();
    stream.next().await.unwrap().task.await.unwrap()
}

fn bench_take(c: &mut Criterion) {
    let mut take_indices = (0..NUM_TAKES)
        .map(|index| (index as u64).wrapping_mul(104_729).wrapping_add(8_191) % NUM_ROWS as u64)
        .collect::<Vec<_>>();
    take_indices.sort_unstable();
    take_indices.dedup();

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("primitive_take");

    for (corpus, array, bss_on) in corpora() {
        let field =
            Field::new("value", array.data_type().clone(), false).with_metadata(metadata(bss_on));
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        let lance_schema =
            Arc::new(lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap());
        let encoded = rt
            .block_on(encode_batch(
                &batch,
                lance_schema,
                encoding_strategy(common::BenchEncoding::StructuralU32).as_ref(),
                &EncodingOptions::default(),
            ))
            .unwrap();
        let encoded_bytes = encoded.data.len();
        let suffix = format!("{corpus}/{encoded_bytes}B");

        group.throughput(Throughput::Elements(NUM_ROWS as u64));
        group.bench_function(format!("scan/{suffix}"), |bencher| {
            bencher.iter(|| {
                let decoded = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        EncodedBatchLayout::Structural,
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(decoded.num_rows(), NUM_ROWS);
            })
        });

        for cache_mode in ["cold", "warm"] {
            let cache = if cache_mode == "cold" {
                Arc::new(LanceCache::no_cache())
            } else {
                Arc::new(LanceCache::with_capacity(64 * 1024 * 1024))
            };
            group.throughput(Throughput::Elements(take_indices.len() as u64));
            group.bench_function(format!("take/{suffix}/{cache_mode}"), |bencher| {
                bencher.iter(|| {
                    let decoded = rt.block_on(decode_take(&encoded, &take_indices, cache.clone()));
                    assert_eq!(decoded.num_rows(), take_indices.len());
                })
            });
        }
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(lance_testing::pprof::PProfProfiler::new(100, lance_testing::pprof::Output::Flamegraph(None)));
    targets = bench_take);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_take);

criterion_main!(benches);
