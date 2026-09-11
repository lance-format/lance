// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
//! Timing benchmarks for the sparse structural read path.
//!
//! Three groups, because a change to the cached chunk index can move them independently:
//!
//! * `sparse_init` — scheduler construction against a cold cache. This is the metadata
//!   parse plus chunk-index build, paid once per page per open dataset.
//! * `sparse_scan` — full sequential decode with a warm cache, the throughput case.
//! * `sparse_take` — scattered ascending row take with a warm cache. Each index lands in a
//!   different chunk, so this is the path that does the most per-chunk lookups and is
//!   where a more compact chunk index could plausibly cost time.
//!
//! Run:
//!
//! ```text
//! cargo bench -p lance-encoding --bench sparse_decode
//! cargo bench -p lance-encoding --bench sparse_decode -- sparse_take
//! ```

use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use futures::StreamExt;
use lance_core::cache::LanceCache;
use lance_encoding::{
    decoder::{
        DecodeBatchScheduler, DecoderConfig, DecoderPlugins, FilterExpression, create_decode_stream,
    },
    encoder::EncodedBatch,
};
use tokio::sync::mpsc::unbounded_channel;

#[path = "sparse/cases.rs"]
mod cases;

use cases::{Case, cases, encode, scattered_indices};

const BATCH_SIZE: u32 = 8192;
/// Enough scattered indices to spread across many chunks without the take degenerating
/// into a full scan.
const TAKE_COUNT: u64 = 4096;

/// Build a scheduler over an already-encoded batch.
///
/// The cache is supplied by the caller so that the init benchmark can pass a cold cache
/// per iteration while the scan and take benchmarks reuse a warm one.
async fn scheduler(
    encoded: &EncodedBatch,
    cache: Arc<LanceCache>,
    io: Arc<dyn lance_encoding::EncodingsIo>,
) -> DecodeBatchScheduler {
    DecodeBatchScheduler::try_new(
        encoded.schema.as_ref(),
        &encoded.top_level_columns,
        &encoded.page_table,
        &vec![],
        encoded.num_rows,
        Arc::<DecoderPlugins>::default(),
        io,
        cache,
        &FilterExpression::no_filter(),
        &DecoderConfig::default(),
    )
    .await
    .expect("scheduler")
}

/// Drive the decoder to completion, returning the row count so the work cannot be
/// optimised away.
async fn drain(encoded: &EncodedBatch, cache: Arc<LanceCache>, indices: Option<&[u64]>) -> usize {
    let io = Arc::new(lance_encoding::BufferScheduler::new(encoded.data.clone()))
        as Arc<dyn lance_encoding::EncodingsIo>;
    let filter = FilterExpression::no_filter();
    let mut sched = scheduler(encoded, cache, io.clone()).await;

    let (tx, rx) = unbounded_channel();
    let expected = match indices {
        Some(indices) => {
            sched.schedule_take(indices, &filter, tx, io);
            indices.len() as u64
        }
        None => {
            sched.schedule_range(0..encoded.num_rows, &filter, tx, io);
            encoded.num_rows
        }
    };

    let stream = create_decode_stream(
        &encoded.schema,
        expected,
        BATCH_SIZE,
        /*is_structural=*/ true,
        /*should_validate=*/ false,
        /*spawn_structural_batch_decode_tasks=*/ false,
        rx,
        None,
    )
    .expect("decode stream");

    let mut rows = 0;
    let mut stream = stream.map(|task| task.task).buffered(1);
    while let Some(batch) = stream.next().await {
        rows += batch.expect("decoded batch").num_rows();
    }
    rows
}

/// Cases worth timing. The degenerate shapes are covered by the footprint report; timing
/// them adds Criterion runtime without adding signal, except for the two that actually
/// stress chunk lookups.
fn timed_cases() -> Vec<Case> {
    cases()
        .into_iter()
        .filter(|c| {
            matches!(
                c.name,
                "degenerate/single_value"
                    | "uniform/many_chunks"
                    | "non_uniform/many_chunks"
                    | "nested/list_of_list"
                    | "wide/32_columns"
                    | "automatic/skewed_lists"
            )
        })
        .collect()
}

/// Criterion ids use `_` where the case name uses `/`, so that `uniform/many_chunks` and
/// `non_uniform/many_chunks` stay distinct within a group.
fn bench_id(name: &str) -> String {
    name.replace('/', "_")
}

fn bench_sparse(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    for case in timed_cases() {
        let encoded = Arc::new(encode(&case));
        if encoded.num_rows == 0 {
            continue;
        }
        let indices = scattered_indices(encoded.num_rows, TAKE_COUNT);
        let id = bench_id(case.name);

        // A cold cache per iteration: this is the once-per-page cost, so a warm cache
        // would measure nothing.
        let mut group = c.benchmark_group("sparse_init");
        group.bench_function(&id, |b| {
            b.iter(|| {
                let cache = Arc::new(LanceCache::with_capacity(1024 * 1024 * 1024));
                let io = Arc::new(lance_encoding::BufferScheduler::new(encoded.data.clone()))
                    as Arc<dyn lance_encoding::EncodingsIo>;
                rt.block_on(scheduler(&encoded, cache, io));
            })
        });
        group.finish();

        // Warm cache, shared across iterations, so the scan measures decode rather than
        // repeated metadata parsing.
        let warm = Arc::new(LanceCache::with_capacity(1024 * 1024 * 1024));
        rt.block_on(drain(&encoded, warm.clone(), None));

        let mut group = c.benchmark_group("sparse_scan");
        group.throughput(criterion::Throughput::Elements(encoded.num_rows));
        group.bench_function(&id, |b| {
            b.iter(|| {
                let rows = rt.block_on(drain(&encoded, warm.clone(), None));
                assert_eq!(rows as u64, encoded.num_rows);
            })
        });
        group.finish();

        if indices.is_empty() {
            continue;
        }
        let mut group = c.benchmark_group("sparse_take");
        group.throughput(criterion::Throughput::Elements(indices.len() as u64));
        group.bench_function(&id, |b| {
            b.iter(|| {
                let rows = rt.block_on(drain(&encoded, warm.clone(), Some(&indices)));
                assert_eq!(rows, indices.len());
            })
        });
        group.finish();
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(lance_testing::pprof::PProfProfiler::new(100, lance_testing::pprof::Output::Flamegraph(None)));
    targets = bench_sparse
);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_sparse
);

criterion_main!(benches);
