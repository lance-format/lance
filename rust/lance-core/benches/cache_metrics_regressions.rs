// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Causal controls for native cache instrumentation costs.

use std::borrow::Cow;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier, mpsc};
use std::task::Poll;
use std::thread;
use std::time::Instant;

use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use lance_core::cache::{
    CacheBackend, CacheEntry, CacheKey, CacheKeySchema, InternalCacheKey, KeyBuilder, LanceCache,
    MokaCacheBackend, QuickCacheBackend,
};
use moka::notification::RemovalCause;

const CAPACITY: usize = 1 << 20;
const READER_BATCH: u64 = 4096;

struct PageKey(u64);

impl CacheKey for PageKey {
    type ValueType = Vec<u8>;

    fn key(&self) -> Cow<'_, str> {
        Cow::Borrowed("unused")
    }

    fn type_name() -> &'static str {
        "bench.Page"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("bench.page-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(17);
        builder.write_u64(self.0);
    }
}

#[derive(Clone)]
struct Entry {
    value: CacheEntry,
    size_bytes: usize,
}

#[derive(Default)]
struct WriteCounters {
    count: AtomicU64,
    bytes: AtomicU64,
}

impl WriteCounters {
    fn record(&self, bytes: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        let _ = self
            .bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                Some(n.saturating_add(bytes))
            });
    }
}

fn raw_moka(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut variants = vec![
        "plain",
        "writes",
        "noop_listener",
        "sync_listener",
        "zst_listener",
    ];
    if std::env::var_os("LANCE_CACHE_BENCH_REVERSE").is_some() {
        variants.reverse();
    }
    for variant in variants {
        let writes = Arc::new(WriteCounters::default());
        let removals = Arc::new(WriteCounters::default());
        let listener_removals = removals.clone();
        let builder = moka::future::Cache::builder()
            .max_capacity(CAPACITY as u64)
            .weigher(|_: &InternalCacheKey, value: &Entry| (16 + value.size_bytes) as u32);
        let cache = match variant {
            "noop_listener" => builder.eviction_listener(|_, _, _| {}).build(),
            "sync_listener" => builder
                .eviction_listener(move |_, entry, cause| {
                    if cause == RemovalCause::Size {
                        listener_removals.record((16 + entry.size_bytes) as u64);
                    }
                })
                .build(),
            "zst_listener" => builder
                .async_eviction_listener(move |_, entry, cause| {
                    if cause == RemovalCause::Size {
                        listener_removals.record((16 + entry.size_bytes) as u64);
                    }
                    // A zero-sized future needs no heap allocation. The callback
                    // still runs synchronously, as in Moka's sync adapter.
                    Box::pin(std::future::poll_fn(|_| Poll::Ready(())))
                })
                .build(),
            _ => builder.build(),
        };
        let value = Entry {
            value: Arc::new(vec![1_u8; 32]),
            size_bytes: 56,
        };
        let key = InternalCacheKey::from_bytes([0; 16]);
        runtime.block_on(async {
            cache.insert(key, value.clone()).await;
            cache.run_pending_tasks().await;
            assert!(cache.get(&key).await.is_some());
        });
        let record_writes = matches!(variant, "writes" | "sync_listener" | "zst_listener");
        let mut group = c.benchmark_group("cache_regressions_moka_replacement");
        group.bench_function(variant, |b| {
            b.to_async(&runtime).iter(|| async {
                if record_writes {
                    writes.record(72);
                }
                cache.insert(black_box(key), value.clone()).await;
            });
        });
        group.finish();
        let mut group = c.benchmark_group("cache_regressions_moka_replacement_drained");
        group.throughput(Throughput::Elements(128));
        group.bench_function(variant, |b| {
            b.to_async(&runtime).iter(|| async {
                for _ in 0..128 {
                    if record_writes {
                        writes.record(72);
                    }
                    cache.insert(black_box(key), value.clone()).await;
                }
                cache.run_pending_tasks().await;
                black_box(cache.weighted_size());
            });
        });
        group.finish();
        // Every measured operation replaces one resident entry; the listener
        // must not count these as size removals.
        assert_eq!(removals.count.load(Ordering::Relaxed), 0);
        black_box((
            writes.count.load(Ordering::Relaxed),
            writes.bytes.load(Ordering::Relaxed),
        ));
        black_box(removals.bytes.load(Ordering::Relaxed));
        black_box(&value.value);
    }
}

fn physical(backend: &str, capacity: usize) -> Arc<dyn CacheBackend> {
    match backend {
        "quick" => Arc::new(QuickCacheBackend::with_capacity(capacity)),
        "moka" => Arc::new(MokaCacheBackend::with_capacity(capacity)),
        _ => unreachable!(),
    }
}

async fn prefill(cache: &LanceCache, common: Option<&Arc<Vec<u8>>>) {
    for id in 0..128 {
        let value = common.cloned().unwrap_or_else(|| Arc::new(vec![1_u8; 32]));
        cache.insert_with_key(&PageKey(id), value).await;
    }
    assert_eq!(cache.size().await, 128);
}

fn wrappers(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for backend in ["quick", "moka"] {
        let cache = LanceCache::with_backend(physical(backend, CAPACITY));
        runtime.block_on(prefill(&cache, None));
        let mut group = c.benchmark_group(format!("cache_regressions_hits/{backend}"));
        for (pattern, mask) in [("hot", 0), ("rotating", 127)] {
            let mut sequence = 0;
            group.bench_function(BenchmarkId::new("strong", pattern), |b| {
                b.to_async(&runtime).iter(|| {
                    let key = PageKey(sequence & mask);
                    sequence += 1;
                    let cache = &cache;
                    async move { black_box(cache.get_with_key(&key).await.unwrap()) }
                });
            });
            group.bench_function(BenchmarkId::new("loader_skipped", pattern), |b| {
                b.to_async(&runtime).iter(|| {
                    let key = PageKey(sequence & mask);
                    sequence += 1;
                    cache
                        .get_or_insert_with_key_hit(key, || async { panic!("resident loader ran") })
                });
            });
        }
        group.finish();
        black_box(&cache);
        let cache = LanceCache::with_backend(physical(backend, 0));
        c.bench_function(
            &format!("cache_regressions_loads/{backend}/disabled"),
            |b| {
                b.to_async(&runtime).iter(|| {
                    cache.get_or_insert_with_key(PageKey(0), || async { Ok(vec![1_u8; 32]) })
                });
            },
        );
        black_box(&cache);
    }
}

// Linux-only CPU binding is opt-in and applies to this investigation fixture.
// Keeping the coordinator on a separate CPU avoids migrating workers at barriers.
#[cfg(target_os = "linux")]
fn pin_worker(worker: usize) {
    let Some(cpus) = std::env::var_os("LANCE_CACHE_READER_CPUS") else {
        return;
    };
    let cpus: Vec<usize> = cpus
        .to_str()
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let cpu = cpus[worker];
    assert!(cpu < libc::CPU_SETSIZE as usize);
    // SAFETY: the initialized set has the platform's required size, cpu was
    // bounds checked, and pid zero changes only the calling thread's affinity.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        assert_eq!(
            libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set),
            0
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_worker(_: usize) {
    assert!(
        std::env::var_os("LANCE_CACHE_READER_CPUS").is_none(),
        "CPU binding requires Linux"
    );
}

fn readers(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for (backend_kind, loader_skipped, hot_key) in [
        ("quick", false, false),
        ("moka", true, true),
        ("moka", true, false),
    ] {
        for values in ["shared_value", "distinct_values"] {
            let common_value = Arc::new(vec![1_u8; 32]);
            for sharing in ["clones", "wrappers", "independent"] {
                let backend = physical(backend_kind, CAPACITY);
                let shared = LanceCache::with_backend(backend.clone());
                let common = (values == "shared_value").then_some(&common_value);
                runtime.block_on(prefill(&shared, common));
                let barrier = Arc::new(Barrier::new(5));
                let mut senders = Vec::with_capacity(4);
                let mut workers = Vec::with_capacity(4);
                for worker in 0..4 {
                    let cache = match sharing {
                        "clones" => shared.clone(),
                        "wrappers" => LanceCache::with_backend(backend.clone()),
                        _ => {
                            let cache = LanceCache::with_backend(physical(backend_kind, CAPACITY));
                            runtime.block_on(prefill(&cache, common));
                            cache
                        }
                    };
                    let barrier = barrier.clone();
                    let (tx, rx) = mpsc::channel::<u64>();
                    senders.push(tx);
                    workers.push(thread::spawn(move || {
                        pin_worker(worker);
                        let runtime = tokio::runtime::Builder::new_current_thread()
                            .build()
                            .unwrap();
                        while let Ok(batches) = rx.recv() {
                            barrier.wait();
                            runtime.block_on(async {
                                let key = PageKey(if hot_key { 0 } else { worker as u64 });
                                if loader_skipped {
                                    for _ in 0..batches * READER_BATCH {
                                        let (value, was_cached) = cache
                                            .get_or_insert_with_key_hit(PageKey(key.0), || async {
                                                panic!("resident loader ran")
                                            })
                                            .await
                                            .unwrap();
                                        assert!(was_cached);
                                        black_box(value);
                                    }
                                } else {
                                    for _ in 0..batches * READER_BATCH {
                                        black_box(cache.get_with_key(&key).await.unwrap());
                                    }
                                }
                            });
                            barrier.wait();
                        }
                        black_box(&cache);
                    }));
                }
                let group_name = if loader_skipped {
                    let pattern = if hot_key { "hot" } else { "disjoint" };
                    format!("cache_regressions_readers/moka/{pattern}/{values}")
                } else {
                    format!("cache_regressions_readers/quick/{values}")
                };
                let mut group = c.benchmark_group(group_name);
                group.throughput(Throughput::Elements(4 * READER_BATCH));
                group.sampling_mode(SamplingMode::Flat);
                group.bench_function(sharing, |b| {
                    b.iter_custom(|batches| {
                        for tx in &senders {
                            tx.send(batches).unwrap();
                        }
                        let start = Instant::now();
                        barrier.wait();
                        barrier.wait();
                        start.elapsed()
                    });
                });
                group.finish();
                drop(senders);
                for worker in workers {
                    worker.join().unwrap();
                }
                black_box((&shared, &backend));
            }
        }
    }
}

fn primitives(c: &mut Criterion) {
    let counters: [AtomicU64; 5] = std::array::from_fn(|_| AtomicU64::new(0));
    c.bench_function("cache_regressions_primitives/two_clock_reads", |b| {
        b.iter(|| {
            let start = Instant::now();
            black_box(start.elapsed());
        })
    });
    c.bench_function("cache_regressions_primitives/five_atomics", |b| {
        b.iter(|| {
            for counter in &counters {
                black_box(counter.fetch_add(1, Ordering::Relaxed));
            }
        })
    });
    c.bench_function("cache_regressions_primitives/clock_and_five_atomics", |b| {
        b.iter(|| {
            counters[0].fetch_add(1, Ordering::Relaxed);
            counters[1].fetch_add(1, Ordering::Relaxed);
            let start = Instant::now();
            let elapsed = start.elapsed().as_nanos() as u64;
            counters[2].fetch_add(1, Ordering::Relaxed);
            counters[3].fetch_add(elapsed, Ordering::Relaxed);
            counters[1].fetch_sub(1, Ordering::Relaxed);
        })
    });
}

criterion_group!(benches, raw_moka, wrappers, readers, primitives);
criterion_main!(benches);
