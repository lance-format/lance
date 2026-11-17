// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, hint::black_box, sync::Arc};

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lance_core::cache::LanceCache;
use lance_encoding::{
    decoder::{DecoderPlugins, EncodedBatchLayout, FilterExpression, decode_batch},
    encoder::{EncodingOptions, encode_batch},
};

pub mod common;
use common::{BenchEncoding, encoding_strategy};

const NUM_ROWS: usize = 1_000_000;

/// zstd level 0 is the library default (level 3). The matrix keeps lower
/// levels visible so the fast tier used for small miniblocks can be compared.
const ZSTD_LEVELS: &[i32] = &[0, 1, 3, 9];
/// lz4 does not currently honor `compression-level` in the miniblock path;
/// the levels are still exercised so ignored settings stay visible in output.
const LZ4_LEVELS: &[i32] = &[0, 1, 9, 12];
/// Minichunk sizes span the small (4-8KiB) and large (>32KiB) regimes.
const MINICHUNK_SIZES: &[i64] = &[4 * 1024, 8 * 1024, 32 * 1024, 64 * 1024];

fn metadata_for(compression: &str, level: i32, minichunk_size: i64) -> HashMap<String, String> {
    HashMap::from([
        (
            "lance-encoding:structural-encoding".to_string(),
            "miniblock".to_string(),
        ),
        (
            "lance-encoding:dict-divisor".to_string(),
            "100000".to_string(),
        ),
        (
            "lance-encoding:compression".to_string(),
            compression.to_string(),
        ),
        (
            "lance-encoding:compression-level".to_string(),
            level.to_string(),
        ),
        (
            "lance-encoding:minichunk-size".to_string(),
            minichunk_size.to_string(),
        ),
    ])
}

fn compressible_string_batch(compression: &str, level: i32, minichunk_size: i64) -> RecordBatch {
    // High-cardinality but compressible strings: unique values with a repeated
    // prefix, so dictionary encoding stays disabled and the compression codec
    // is the only size factor.
    let array: ArrayRef = Arc::new(StringArray::from_iter_values(
        (0..NUM_ROWS).map(|i| format!("prefix_that_compresses_well_{i:09}")),
    ));
    let field = Field::new("value", DataType::Utf8, false).with_metadata(metadata_for(
        compression,
        level,
        minichunk_size,
    ));
    let schema = Arc::new(Schema::new(vec![field]));
    RecordBatch::try_new(schema, vec![array]).unwrap()
}

fn bench_encode_compression_levels(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("compression_levels_encode");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));

    let strategy = encoding_strategy(BenchEncoding::StructuralU32);
    let options = EncodingOptions::default();
    for (compression, levels) in [("zstd", ZSTD_LEVELS), ("lz4", LZ4_LEVELS)] {
        for level in levels {
            for minichunk_size in MINICHUNK_SIZES {
                let data = compressible_string_batch(compression, *level, *minichunk_size);
                let lance_schema = Arc::new(
                    lance_core::datatypes::Schema::try_from(data.schema().as_ref()).unwrap(),
                );
                let encoded_bytes = rt
                    .block_on(encode_batch(
                        &data,
                        lance_schema.clone(),
                        strategy.as_ref(),
                        &options,
                    ))
                    .unwrap()
                    .data
                    .len();
                let chunk_kib = minichunk_size / 1024;
                group.bench_function(
                    format!("{compression}_level{level}/{chunk_kib}Ki/{encoded_bytes}B"),
                    |bencher| {
                        bencher.iter(|| {
                            black_box(rt.block_on(encode_batch(
                                &data,
                                lance_schema.clone(),
                                strategy.as_ref(),
                                &options,
                            )))
                        })
                    },
                );
            }
        }
    }
}

fn bench_decode_compression_levels(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("compression_levels_decode");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));

    let strategy = encoding_strategy(BenchEncoding::StructuralU32);
    let options = EncodingOptions::default();
    for (compression, levels) in [("zstd", ZSTD_LEVELS), ("lz4", LZ4_LEVELS)] {
        for level in levels {
            for minichunk_size in MINICHUNK_SIZES {
                let data = compressible_string_batch(compression, *level, *minichunk_size);
                let lance_schema = Arc::new(
                    lance_core::datatypes::Schema::try_from(data.schema().as_ref()).unwrap(),
                );
                let encoded = rt
                    .block_on(encode_batch(
                        &data,
                        lance_schema,
                        strategy.as_ref(),
                        &options,
                    ))
                    .unwrap();
                let encoded_bytes = encoded.data.len();
                let chunk_kib = minichunk_size / 1024;
                let cache = Arc::new(LanceCache::no_cache());
                group.bench_function(
                    format!("{compression}_level{level}/{chunk_kib}Ki/{encoded_bytes}B"),
                    |bencher| {
                        bencher.iter(|| {
                            let batch = rt
                                .block_on(decode_batch(
                                    &encoded,
                                    &FilterExpression::no_filter(),
                                    Arc::<DecoderPlugins>::default(),
                                    false,
                                    EncodedBatchLayout::Structural,
                                    Some(cache.clone()),
                                ))
                                .unwrap();
                            assert_eq!(batch.num_rows(), NUM_ROWS);
                        })
                    },
                );
            }
        }
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(lance_testing::pprof::PProfProfiler::new(100, lance_testing::pprof::Output::Flamegraph(None)));
    targets = bench_encode_compression_levels, bench_decode_compression_levels);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_encode_compression_levels, bench_decode_compression_levels);

criterion_main!(benches);
