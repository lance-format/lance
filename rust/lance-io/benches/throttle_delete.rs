// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use futures::{StreamExt, TryStreamExt, stream};
use lance_core::utils::aimd::AimdConfig;
use lance_io::object_store::throttle::{AimdThrottleConfig, AimdThrottledStore};
use object_store::{ObjectStore, memory::InMemory, path::Path};

fn bench_generic_delete(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let target: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let config = AimdThrottleConfig::default()
        .with_burst_capacity(1_000)
        .with_delete_aimd(
            AimdConfig::default()
                .with_initial_rate(1_000_000.0)
                .with_max_rate(1_000_000.0),
        );
    let throttled = AimdThrottledStore::new(Arc::clone(&target), config).unwrap();
    let paths: Vec<_> = (0..256)
        .map(|index| Path::from(format!("fragments/{index}")))
        .collect();

    let mut group = c.benchmark_group("delete_256_paths");
    group.bench_function("raw_memory", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let results = target
                    .delete_stream(stream::iter(paths.clone().into_iter().map(Ok)).boxed())
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
                assert_eq!(results.len(), paths.len());
            });
        });
    });
    group.bench_function("generic_throttled", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let results = throttled
                    .delete_stream(stream::iter(paths.clone().into_iter().map(Ok)).boxed())
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
                assert_eq!(results.len(), paths.len());
            });
        });
    });
    group.finish();
}

criterion_group!(benches, bench_generic_delete);
criterion_main!(benches);
