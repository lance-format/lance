// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Comparable fixtures for cache instrumentation, independent of legacy keys.

use std::hint::black_box;
use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, Throughput};
use lance_core::cache::{
    CacheBackend, LanceCache, MokaCacheBackend, QuickCacheBackend, WeakLanceCache,
};

use super::PageKey;

#[cfg(feature = "metrics")]
mod recorder;

const CAPACITY: usize = 1 << 20;
const RESIDENT_KEYS: u64 = 128;
const READER_BATCH: u64 = 4096;

#[derive(Clone, Copy)]
enum Backend {
    Quick,
    Moka,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Moka => "moka",
        }
    }

    fn build(self, capacity: usize) -> Arc<dyn CacheBackend> {
        match self {
            Self::Quick => Arc::new(QuickCacheBackend::with_capacity(capacity)),
            Self::Moka => Arc::new(MokaCacheBackend::with_capacity(capacity)),
        }
    }
}

fn key(id: u64) -> PageKey {
    PageKey {
        column_index: 17,
        page_index: id,
    }
}

async fn prefill(cache: &LanceCache, entries: u64) {
    let value = Arc::new(vec![1_u8; 32]);
    for id in 0..entries {
        cache.insert_with_key(&key(id), value.clone()).await;
    }
    assert_eq!(cache.size().await, entries as usize);
    for id in 0..entries {
        assert!(cache.get_with_key(&key(id)).await.is_some());
    }
}

pub fn benchmark(c: &mut Criterion) {
    let mode = std::env::var("LANCE_CACHE_BENCH_MODE").unwrap_or_else(|_| "no_recorder".into());
    match mode.as_str() {
        "no_recorder" => {}
        #[cfg(feature = "metrics")]
        "aggregate" => recorder::install(),
        _ => panic!(
            "unsupported LANCE_CACHE_BENCH_MODE={mode}; aggregate requires --features metrics"
        ),
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for backend in [Backend::Quick, Backend::Moka] {
        let physical = backend.build(CAPACITY);
        let cache = LanceCache::with_backend(physical.clone());
        runtime.block_on(prefill(&cache, RESIDENT_KEYS));
        let weak = WeakLanceCache::from(&cache);
        let keys: Vec<_> = (0..RESIDENT_KEYS).map(key).collect();
        let mut group = c.benchmark_group(format!("cache_metrics_hits/{}", backend.name()));
        for (pattern, mask) in [("hot", 0), ("rotating", RESIDENT_KEYS - 1)] {
            let mut sequence = 0;
            group.bench_function(BenchmarkId::new("strong", pattern), |b| {
                b.to_async(&runtime).iter(|| {
                    let key = &keys[(sequence & mask) as usize];
                    sequence += 1;
                    async { black_box(cache.get_with_key(black_box(key)).await) }
                });
            });
            group.bench_function(BenchmarkId::new("weak", pattern), |b| {
                b.to_async(&runtime).iter(|| {
                    let key = &keys[(sequence & mask) as usize];
                    sequence += 1;
                    async { black_box(weak.get_with_key(black_box(key)).await) }
                });
            });
            group.bench_function(BenchmarkId::new("loader_skipped", pattern), |b| {
                b.to_async(&runtime).iter(|| {
                    let key = key(sequence & mask);
                    sequence += 1;
                    cache.get_or_insert_with_key_hit(key, || async {
                        panic!("a warmed-hit benchmark executed its loader")
                    })
                });
            });
        }
        group.finish();

        let mut group = c.benchmark_group(format!("cache_metrics_writes/{}", backend.name()));
        let value = Arc::new(vec![2_u8; 32]);
        group.bench_function("replacement", |b| {
            b.to_async(&runtime)
                .iter(|| cache.insert_with_key(&keys[0], value.clone()));
        });
        let churn = LanceCache::with_backend(backend.build(4096));
        let mut sequence = 0;
        group.bench_function("bounded_churn", |b| {
            b.to_async(&runtime).iter(|| {
                let id = sequence % 512;
                sequence += 1;
                let value = value.clone();
                let churn = &churn;
                async move { churn.insert_with_key(&key(id), value).await }
            });
        });
        group.throughput(Throughput::Elements(128));
        group.bench_function("churn_with_maintenance", |b| {
            b.to_async(&runtime).iter(|| async {
                for id in 0..128 {
                    churn.insert_with_key(&key(id), value.clone()).await;
                }
                black_box(churn.size_bytes().await);
            });
        });
        group.finish();

        let disabled = LanceCache::with_backend(backend.build(0));
        c.bench_function(
            &format!("cache_metrics_loads/{}/disabled", backend.name()),
            |b| {
                b.to_async(&runtime).iter(|| {
                    disabled.get_or_insert_with_key(key(0), || async { Ok(vec![1_u8; 32]) })
                });
            },
        );

        for entries in [0, 128, 4096, 65536] {
            let physical = backend.build(16 << 20);
            let cache = LanceCache::with_backend(physical.clone());
            runtime.block_on(prefill(&cache, entries));
            c.bench_function(
                &format!(
                    "cache_metrics_native_diagnostics/{}/{entries}/approximate",
                    backend.name()
                ),
                |b| b.iter(|| black_box(cache.diagnostics())),
            );
            c.bench_function(
                &format!(
                    "cache_metrics_collection/{}/{entries}/approximate",
                    backend.name()
                ),
                |b| {
                    b.iter(|| {
                        black_box((
                            physical.approx_num_entries(),
                            physical.approx_size_bytes(),
                            physical.capacity_bytes(),
                        ))
                    });
                },
            );
            c.bench_function(
                &format!(
                    "cache_metrics_collection/{}/{entries}/refreshed",
                    backend.name()
                ),
                |b| {
                    b.to_async(&runtime).iter(|| cache.stats());
                },
            );
        }

        concurrent_readers(c, backend);
    }
}

/// Workers live across all samples. Dispatch and barriers are amortized across
/// 4096 reads per worker; elapsed/read describes throughput, not request latency.
fn concurrent_readers(c: &mut Criterion, backend: Backend) {
    let available = thread::available_parallelism().unwrap().get();
    let setup_runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for readers in [1, 2, 4, 8].into_iter().filter(|n| *n <= available) {
        for sharing in ["clones", "wrappers", "independent"] {
            for pattern in ["hot", "disjoint"] {
                let physical = backend.build(CAPACITY);
                let shared = LanceCache::with_backend(physical.clone());
                setup_runtime.block_on(prefill(&shared, RESIDENT_KEYS));
                let barrier = Arc::new(Barrier::new(readers + 1));
                let mut senders = Vec::with_capacity(readers);
                let mut workers = Vec::with_capacity(readers);
                for worker in 0..readers {
                    let cache = match sharing {
                        "clones" => shared.clone(),
                        "wrappers" => LanceCache::with_backend(physical.clone()),
                        _ => {
                            let cache = LanceCache::with_backend(backend.build(CAPACITY));
                            setup_runtime.block_on(prefill(&cache, RESIDENT_KEYS));
                            cache
                        }
                    };
                    let barrier = barrier.clone();
                    let (tx, rx) = mpsc::channel::<u64>();
                    senders.push(tx);
                    workers.push(thread::spawn(move || {
                        let runtime = tokio::runtime::Builder::new_current_thread()
                            .build()
                            .unwrap();
                        let key = key(if pattern == "hot" { 0 } else { worker as u64 });
                        while let Ok(batches) = rx.recv() {
                            barrier.wait();
                            runtime.block_on(async {
                                for _ in 0..batches * READER_BATCH {
                                    black_box(cache.get_with_key(&key).await);
                                }
                            });
                            barrier.wait();
                        }
                    }));
                }
                let mut group = c.benchmark_group(format!(
                    "cache_metrics_readers/{}/{sharing}/{pattern}",
                    backend.name()
                ));
                group.throughput(Throughput::Elements(readers as u64 * READER_BATCH));
                group.bench_function(readers.to_string(), |b| {
                    b.iter_custom(|batches| {
                        for tx in &senders {
                            tx.send(batches).unwrap();
                        }
                        let start = Instant::now();
                        barrier.wait();
                        barrier.wait();
                        start.elapsed()
                    })
                });
                group.finish();
                drop(senders);
                for worker in workers {
                    worker.join().unwrap();
                }
            }
        }
    }
}
