// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::{sync::Arc, time::Duration};

use arrow_array::{Int32Array, RecordBatch, StringArray, UInt64Array};
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::scalar::ScalarValue;
use futures::stream;
use itertools::Itertools;
use lance_core::ROW_ADDR;
use lance_core::cache::LanceCache;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::pbold;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::zonemap::{
    ZoneMapIndexBuilder, ZoneMapIndexBuilderParams, ZoneMapIndexPlugin,
};
use lance_index::scalar::{SargableQuery, registry::ScalarIndexPlugin};
use lance_io::object_store::ObjectStore;
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};
use object_store::path::Path;

fn bench_zonemap(c: &mut Criterion) {
    const TOTAL: usize = 1_000_000;

    env_logger::init();

    let rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();

    let tempdir = tempfile::tempdir().unwrap();
    let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
    let store = rt.block_on(async {
        Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            index_dir,
            Arc::new(LanceCache::no_cache()),
        ))
    });

    // Generate sequential integers for the zonemap index
    let data_col = arrow_array::Int32Array::from_iter_values(0..TOTAL as i32);

    let row_addr_col = Arc::new(UInt64Array::from(
        (0..TOTAL).map(|i| i as u64).collect_vec(),
    ));

    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("values", arrow_schema::DataType::Int32, false),
            arrow_schema::Field::new(ROW_ADDR, arrow_schema::DataType::UInt64, false),
        ])
        .into(),
        vec![Arc::new(data_col), row_addr_col],
    )
    .unwrap();

    let batches = (0..1000).map(|i| batch.slice(i * 1000, 1000)).collect_vec();

    let mut group = c.benchmark_group("train");

    group.sample_size(10);
    group.bench_function(format!("zonemap_train({TOTAL})").as_str(), |b| {
        b.to_async(&rt).iter(|| async {
            let stream = RecordBatchStreamAdapter::new(
                batch.schema(),
                stream::iter(batches.clone().into_iter().map(Ok)),
            );

            let mut builder = ZoneMapIndexBuilder::try_new(
                ZoneMapIndexBuilderParams::default(),
                batch.schema().field(0).data_type().clone(),
            )
            .unwrap();

            builder.train(Box::pin(stream)).await.unwrap();
            builder.write_index(store.as_ref()).await.unwrap();
        })
    });

    drop(group);

    let mut group = c.benchmark_group("search");

    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    // Write the index explicitly instead of relying on the train benchmark
    // above having run: criterion filters (e.g. `-- zonemap_search`) can skip
    // it, which would leave nothing on disk to load.
    rt.block_on(async {
        let mut builder = ZoneMapIndexBuilder::try_new(
            ZoneMapIndexBuilderParams::default(),
            batch.schema().field(0).data_type().clone(),
        )
        .unwrap();
        let stream = RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::iter(batches.clone().into_iter().map(Ok)),
        );
        builder.train(Box::pin(stream)).await.unwrap();
        builder.write_index(store.as_ref()).await.unwrap();
    });
    let details = prost_types::Any::from_msg(&pbold::ZoneMapIndexDetails::default()).unwrap();
    let index = rt
        .block_on(ZoneMapIndexPlugin.load_index(store, &details, None, &LanceCache::no_cache()))
        .unwrap();
    group.bench_function(format!("zonemap_search({TOTAL})").as_str(), |b| {
        b.to_async(&rt).iter(|| async {
            let sample_idx = rand::random_range(0..batch.num_rows());
            let sample_value = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(sample_idx);
            let query = SargableQuery::Equals(ScalarValue::Int32(Some(sample_value)));
            black_box(index.search(&query, &NoOpMetricsCollector).await.unwrap());
        })
    });
}

/// Training on a string column across many fragments. This is the shape where
/// training throughput matters most (per-zone min/max over variable-length
/// values); the Int32 benchmark above guards per-zone dispatch overhead on
/// cheap fixed-width types instead.
///
/// Times training only — `write_index` is deliberately left out so the numbers
/// stay comparable with the bloom filter training benchmark.
fn bench_zonemap_string_multifragment(c: &mut Criterion) {
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
        format!("zonemap_train_string({TOTAL}x{NUM_FRAGMENTS}frags)").as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let stream = RecordBatchStreamAdapter::new(
                    batch.schema(),
                    stream::iter(batches.clone().into_iter().map(Ok)),
                );

                let mut builder = ZoneMapIndexBuilder::try_new(
                    ZoneMapIndexBuilderParams::default(),
                    batch.schema().field(0).data_type().clone(),
                )
                .unwrap();

                builder.train(Box::pin(stream)).await.unwrap();
                black_box(&builder);
            })
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
    targets = bench_zonemap, bench_zonemap_string_multifragment);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10);
    targets = bench_zonemap, bench_zonemap_string_multifragment);

criterion_main!(benches);
