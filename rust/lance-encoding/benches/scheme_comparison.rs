// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, hint::black_box, sync::Arc};

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{Field, Schema};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lance_core::{Error, Result, datatypes::Field as LanceField};
use lance_encoding::{
    compression::{
        BlockCompressor, CompressionStrategy, finalize_miniblock_compressor,
        try_bitpacking_miniblock, try_byte_stream_split_miniblock, try_fixed_u8_rle_miniblock,
        try_raw_block, try_raw_fixed_width_miniblock, try_raw_per_value,
    },
    compression_config::{BssMode, CompressionFieldParams},
    data::DataBlock,
    decoder::{DecoderPlugins, EncodedBatchLayout, FilterExpression, decode_batch},
    encoder::{EncodingOptions, encode_batch},
    encodings::logical::primitive::fullzip::PerValueCompressor,
    encodings::logical::primitive::miniblock::MiniBlockCompressor,
};

pub mod common;
use common::{BenchEncoding, encoding_strategy_with_compression};

#[derive(Clone, Copy, Debug)]
enum Scheme {
    /// Uncompressed values stored at full width.
    Raw,
    /// Fixed-width run-length encoding.
    Rle,
    /// Inline bitpacking at 1024-value chunk granularity.
    Bitpack,
    /// Byte-stream split followed by zstd level 3 (the real BSS pipeline).
    BssZstd,
    /// Flat values wrapped in zstd level 3.
    Zstd,
    /// Flat values wrapped in lz4.
    Lz4,
}

impl Scheme {
    fn name(&self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Rle => "rle",
            Self::Bitpack => "bitpack",
            Self::BssZstd => "bss_zstd",
            Self::Zstd => "zstd",
            Self::Lz4 => "lz4",
        }
    }
}

const ALL_SCHEMES: &[Scheme] = &[
    Scheme::Raw,
    Scheme::Rle,
    Scheme::Bitpack,
    Scheme::BssZstd,
    Scheme::Zstd,
    Scheme::Lz4,
];

fn compressor_for(scheme: Scheme, data: &DataBlock) -> Option<Box<dyn MiniBlockCompressor>> {
    match scheme {
        Scheme::Raw => try_raw_fixed_width_miniblock(data),
        Scheme::Rle => {
            // A threshold of 1.0 keeps RLE in the running for data with any
            // run; the size model still rejects it when it cannot beat raw or
            // bitpacking.
            let params = CompressionFieldParams {
                rle_threshold: Some(1.0),
                ..Default::default()
            };
            try_fixed_u8_rle_miniblock(data, &params)
        }
        Scheme::Bitpack => try_bitpacking_miniblock(data),
        Scheme::BssZstd => {
            let params = CompressionFieldParams {
                compression: Some("zstd".to_string()),
                compression_level: Some(3),
                bss: Some(BssMode::On),
                ..Default::default()
            };
            let inner = try_byte_stream_split_miniblock(data, &params)?;
            finalize_miniblock_compressor(data, inner, &params).ok()
        }
        Scheme::Zstd => {
            let params = CompressionFieldParams {
                compression: Some("zstd".to_string()),
                compression_level: Some(3),
                ..Default::default()
            };
            let inner = try_raw_fixed_width_miniblock(data)?;
            finalize_miniblock_compressor(data, inner, &params).ok()
        }
        Scheme::Lz4 => {
            let params = CompressionFieldParams {
                compression: Some("lz4".to_string()),
                compression_level: Some(0),
                ..Default::default()
            };
            let inner = try_raw_fixed_width_miniblock(data)?;
            finalize_miniblock_compressor(data, inner, &params).ok()
        }
    }
}

/// Forced-scheme strategy: `create_miniblock_compressor` returns exactly the
/// requested scheme for every page that qualifies; other page shapes fall back
/// to raw so they never silently substitute a different codec.
#[derive(Clone, Debug)]
struct SchemeCompressionStrategy {
    scheme: Scheme,
}

impl CompressionStrategy for SchemeCompressionStrategy {
    fn create_per_value(
        &self,
        _field: &LanceField,
        data: &DataBlock,
    ) -> Result<Box<dyn PerValueCompressor>> {
        try_raw_per_value(data).ok_or_else(|| {
            Error::not_supported_source(
                "scheme bench only supports raw per-value"
                    .to_string()
                    .into(),
            )
        })
    }

    fn create_block_compressor(
        &self,
        _field: &LanceField,
        data: &DataBlock,
    ) -> Result<Box<dyn BlockCompressor>> {
        try_raw_block(data).ok_or_else(|| {
            Error::not_supported_source(
                "scheme bench only supports raw block compression"
                    .to_string()
                    .into(),
            )
        })
    }

    fn create_miniblock_compressor(
        &self,
        _field: &LanceField,
        data: &DataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        compressor_for(self.scheme, data).ok_or_else(|| {
            Error::not_supported_source(
                format!(
                    "scheme {} is not applicable to this data",
                    self.scheme.name()
                )
                .into(),
            )
        })
    }
}

fn corpora() -> Vec<(&'static str, ArrayRef)> {
    vec![
        (
            "i64_narrow",
            Arc::new(Int64Array::from_iter_values(
                (0..8192).map(|i| (i * 7919) % 1_000_000),
            )),
        ),
        (
            "i64_runs",
            Arc::new(Int64Array::from_iter_values(
                (0..8192).map(|i| (i / 32) % 256),
            )),
        ),
        (
            "i64_random",
            Arc::new(Int64Array::from_iter_values(
                (0..8192).map(|_| rand::random::<i64>()),
            )),
        ),
        (
            "f64_clustered",
            Arc::new(Float64Array::from_iter_values(
                (0..8192).map(|i| 1.0 + (i % 1000) as f64 * 1e-9),
            )),
        ),
        (
            "f64_random",
            Arc::new(Float64Array::from_iter_values(
                (0..8192).map(|_| rand::random::<f64>()),
            )),
        ),
    ]
}

fn batch_for(array: ArrayRef) -> RecordBatch {
    let metadata = HashMap::from([
        (
            "lance-encoding:structural-encoding".to_string(),
            "miniblock".to_string(),
        ),
        // Disable dictionary encoding so the comparison isolates the value
        // codec rather than a shared dictionary layer.
        (
            "lance-encoding:dict-divisor".to_string(),
            "100000".to_string(),
        ),
    ]);
    let field = Field::new("value", array.data_type().clone(), false).with_metadata(metadata);
    let schema = Arc::new(Schema::new(vec![field]));
    RecordBatch::try_new(schema, vec![array]).unwrap()
}

fn bench_scheme_encode(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let options = EncodingOptions::default();
    let mut encode_group = c.benchmark_group("scheme_comparison_encode");

    for (corpus, array) in corpora() {
        let batch = batch_for(array);
        let num_values = batch.num_rows() as u64;
        let lance_schema =
            Arc::new(lance_core::datatypes::Schema::try_from(batch.schema().as_ref()).unwrap());
        encode_group.throughput(Throughput::Elements(num_values));
        for scheme in ALL_SCHEMES {
            let strategy = encoding_strategy_with_compression(
                BenchEncoding::StructuralU32,
                Arc::new(SchemeCompressionStrategy { scheme: *scheme }),
            );
            // Probe so inapplicable schemes (e.g. RLE on random data) are
            // skipped instead of failing the whole bench run.
            let Ok(probe) = rt.block_on(encode_batch(
                &batch,
                lance_schema.clone(),
                strategy.as_ref(),
                &options,
            )) else {
                continue;
            };
            let encoded_bytes = probe.data.len();
            encode_group.bench_function(
                format!("{corpus}/{}/{encoded_bytes}B", scheme.name()),
                |bencher| {
                    bencher.iter(|| {
                        black_box(rt.block_on(encode_batch(
                            &batch,
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

fn bench_scheme_decode(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let options = EncodingOptions::default();
    let cache = Arc::new(lance_core::cache::LanceCache::no_cache());
    let mut decode_group = c.benchmark_group("scheme_comparison_decode");

    for (corpus, array) in corpora() {
        let batch = batch_for(array);
        let num_values = batch.num_rows() as u64;
        let lance_schema =
            Arc::new(lance_core::datatypes::Schema::try_from(batch.schema().as_ref()).unwrap());
        decode_group.throughput(Throughput::Elements(num_values));
        for scheme in ALL_SCHEMES {
            let strategy = encoding_strategy_with_compression(
                BenchEncoding::StructuralU32,
                Arc::new(SchemeCompressionStrategy { scheme: *scheme }),
            );
            // Probe so inapplicable schemes are skipped consistently with the
            // encode group.
            let Ok(probe) = rt.block_on(encode_batch(
                &batch,
                lance_schema.clone(),
                strategy.as_ref(),
                &options,
            )) else {
                continue;
            };
            let encoded_bytes = probe.data.len();
            decode_group.bench_function(
                format!("{corpus}/{}/{encoded_bytes}B", scheme.name()),
                |bencher| {
                    bencher.iter(|| {
                        let decoded = rt
                            .block_on(decode_batch(
                                &probe,
                                &FilterExpression::no_filter(),
                                Arc::<DecoderPlugins>::default(),
                                false,
                                EncodedBatchLayout::Structural,
                                Some(cache.clone()),
                            ))
                            .unwrap();
                        assert_eq!(decoded.num_rows(), num_values as usize);
                    })
                },
            );
        }
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(lance_testing::pprof::PProfProfiler::new(100, lance_testing::pprof::Output::Flamegraph(None)));
    targets = bench_scheme_encode, bench_scheme_decode);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_scheme_encode, bench_scheme_decode);

criterion_main!(benches);
