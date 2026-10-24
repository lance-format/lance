// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, hint::black_box, sync::Arc};

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{Field, Schema};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lance_core::cache::LanceCache;
use lance_encoding::{
    decoder::{DecoderPlugins, EncodedBatchLayout, FilterExpression},
    encoder::{EncodingOptions, encode_batch},
};

pub mod common;
use common::{BenchEncoding, encoding_strategy};

const NUM_VALUES: usize = 262_144;

fn field_metadata(bss_off: bool) -> HashMap<String, String> {
    let mut metadata = HashMap::from([
        (
            "lance-encoding:structural-encoding".to_string(),
            "miniblock".to_string(),
        ),
        (
            "lance-encoding:dict-divisor".to_string(),
            "100000".to_string(),
        ),
        // Explicit compression activates the miniblock codec selection so the
        // bench observes the production heuristics (including BSS) on each
        // synthetic distribution.
        ("lance-encoding:compression".to_string(), "zstd".to_string()),
        (
            "lance-encoding:compression-level".to_string(),
            "3".to_string(),
        ),
    ]);
    if bss_off {
        metadata.insert("lance-encoding:bss".to_string(), "off".to_string());
    }
    metadata
}

fn corpora() -> Vec<(&'static str, ArrayRef)> {
    vec![
        // A few extreme values widen the bit width for bitpacking and the byte
        // streams for BSS, which is the classic outlier failure mode.
        (
            "f64_outliers",
            Arc::new(Float64Array::from_iter_values((0..NUM_VALUES).map(|i| {
                if i % 4096 == 0 {
                    1e300
                } else {
                    1.0 + (i % 1024) as f64 * 1e-9
                }
            }))),
        ),
        // NaN shares payload bits with valid floats in a byte stream, so sign
        // and exponent handling can change the chosen scheme.
        (
            "f64_nan",
            Arc::new(Float64Array::from_iter_values((0..NUM_VALUES).map(|i| {
                if i % 100 == 0 {
                    f64::NAN
                } else {
                    1.0 + (i % 1024) as f64 * 1e-9
                }
            }))),
        ),
        (
            "f64_all_zero",
            Arc::new(Float64Array::from_iter_values(std::iter::repeat_n(
                0.0, NUM_VALUES,
            ))),
        ),
        // Constant high bytes plus noisy low mantissa: BSS separates the
        // constant prefix into empty streams while plain zstd also compresses
        // it away.
        (
            "f64_constant_prefix_noise",
            Arc::new(Float64Array::from_iter_values(
                (0..NUM_VALUES).map(|i| 1.0 + (i as u64 % 65_536) as f64 * 1e-12),
            )),
        ),
        (
            "i64_outliers",
            Arc::new(Int64Array::from_iter_values((0..NUM_VALUES).map(|i| {
                if i % 4096 == 0 {
                    i64::MAX
                } else {
                    (i % 65_536) as i64
                }
            }))),
        ),
    ]
}

fn bench_synthetic_distributions_encode(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let options = EncodingOptions::default();
    let mut encode_group = c.benchmark_group("synthetic_distributions_encode");

    for (corpus, array) in corpora() {
        for (mode, bss_off) in [("auto", false), ("no_bss", true)] {
            let strategy = encoding_strategy(BenchEncoding::StructuralU32);
            let field = Field::new("value", array.data_type().clone(), false)
                .with_metadata(field_metadata(bss_off));
            let schema = Arc::new(Schema::new(vec![field]));
            let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();
            let lance_schema =
                Arc::new(lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap());
            let encoded = rt
                .block_on(encode_batch(
                    &batch,
                    lance_schema.clone(),
                    strategy.as_ref(),
                    &options,
                ))
                .unwrap();
            let encoded_bytes = encoded.data.len();
            let suffix = format!("{corpus}/{mode}/{encoded_bytes}B");

            encode_group.throughput(Throughput::Elements(NUM_VALUES as u64));
            encode_group.bench_function(&suffix, |bencher| {
                bencher.iter(|| {
                    black_box(rt.block_on(encode_batch(
                        &batch,
                        lance_schema.clone(),
                        strategy.as_ref(),
                        &options,
                    )))
                })
            });
        }
    }
}

fn bench_synthetic_distributions_decode(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let options = EncodingOptions::default();
    let mut decode_group = c.benchmark_group("synthetic_distributions_decode");

    for (corpus, array) in corpora() {
        for (mode, bss_off) in [("auto", false), ("no_bss", true)] {
            let strategy = encoding_strategy(BenchEncoding::StructuralU32);
            let field = Field::new("value", array.data_type().clone(), false)
                .with_metadata(field_metadata(bss_off));
            let schema = Arc::new(Schema::new(vec![field]));
            let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();
            let lance_schema =
                Arc::new(lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap());
            let encoded = rt
                .block_on(encode_batch(
                    &batch,
                    lance_schema,
                    strategy.as_ref(),
                    &options,
                ))
                .unwrap();
            let encoded_bytes = encoded.data.len();
            let suffix = format!("{corpus}/{mode}/{encoded_bytes}B");

            decode_group.throughput(Throughput::Elements(NUM_VALUES as u64));
            decode_group.bench_function(&suffix, |bencher| {
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
                    assert_eq!(decoded.num_rows(), NUM_VALUES);
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
    targets = bench_synthetic_distributions_encode, bench_synthetic_distributions_decode);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_synthetic_distributions_encode, bench_synthetic_distributions_decode);

criterion_main!(benches);
