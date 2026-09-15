// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::{sync::Arc, time::Duration};

use arrow_array::{RecordBatch, StringArray, UInt64Array};
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use itertools::Itertools;
use lance_core::ROW_ADDR;
use lance_index::scalar::bloomfilter::{BloomFilterIndexBuilder, BloomFilterIndexBuilderParams};
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};

/// Training on a high-cardinality string column across many fragments — the
/// canonical bloom filter workload (values too scattered for zonemap pruning).
/// Mirrors the zonemap string benchmark's data shape so the two index types'
/// training costs can be compared directly.
fn bench_bloomfilter_train(c: &mut Criterion) {
    const TOTAL: usize = 8_000_000;
    const NUM_FRAGMENTS: usize = 64;
    const BATCH_SIZE: usize = 8192;

    let rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();

    // 125000 rows per fragment is deliberately not a multiple of the batch size
    // so fragment boundaries fall mid-batch.
    let rows_per_fragment = TOTAL / NUM_FRAGMENTS;
    let data_col = StringArray::from_iter_values(
        (0..TOTAL as u64)
            .map(|i| format!("value-{:012}", i.wrapping_mul(2654435761) % TOTAL as u64)),
    );
    let row_addr_col = UInt64Array::from_iter_values((0..TOTAL as u64).map(|i| {
        let fragment_id = i / rows_per_fragment as u64;
        let offset = i % rows_per_fragment as u64;
        (fragment_id << 32) | offset
    }));

    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("values", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new(ROW_ADDR, arrow_schema::DataType::UInt64, false),
        ])
        .into(),
        vec![Arc::new(data_col), Arc::new(row_addr_col)],
    )
    .unwrap();

    let batches = (0..TOTAL.div_ceil(BATCH_SIZE))
        .map(|i| batch.slice(i * BATCH_SIZE, BATCH_SIZE.min(TOTAL - i * BATCH_SIZE)))
        .collect_vec();

    let mut group = c.benchmark_group("train");
    group.sample_size(10);
    group.bench_function(
        format!("bloomfilter_train_string({TOTAL}x{NUM_FRAGMENTS}frags)").as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let stream = RecordBatchStreamAdapter::new(
                    batch.schema(),
                    stream::iter(batches.clone().into_iter().map(Ok)),
                );
                let mut builder =
                    BloomFilterIndexBuilder::try_new(BloomFilterIndexBuilderParams::default())
                        .unwrap();
                builder.train(Box::pin(stream)).await.unwrap();
                black_box(&builder);
            });
        },
    );
    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_bloomfilter_train);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10);
    targets = bench_bloomfilter_train);

criterion_main!(benches);
