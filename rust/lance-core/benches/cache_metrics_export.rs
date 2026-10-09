// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Representative costs for emitted cache metrics with and without a recorder.

#[cfg(not(feature = "metrics"))]
compile_error!("cache_metrics_export requires the `metrics` feature");

use std::borrow::Cow;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use criterion::{Criterion, criterion_group, criterion_main};
use lance_core::Error;
use lance_core::cache::{
    CacheBackend, CacheKey, CacheKeySchema, CacheLoadOrigin, CacheSnapshotMode, KeyBuilder,
    LanceCache, MokaCacheBackend, QuickCacheBackend, refresh_metrics,
};

#[path = "cache_metrics/recorder.rs"]
mod recorder;

const CAPACITY: usize = 64 << 20;
const SMALL_CAPACITY: usize = 4096;

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
        builder.write_u64(self.0);
    }
}

struct TypeKey<const N: u8>(u64);

impl<const N: u8> CacheKey for TypeKey<N> {
    type ValueType = Vec<u8>;

    fn key(&self) -> Cow<'_, str> {
        Cow::Borrowed("unused")
    }

    fn type_name() -> &'static str {
        match N {
            0 => "bench.Type00",
            1 => "bench.Type01",
            2 => "bench.Type02",
            3 => "bench.Type03",
            4 => "bench.Type04",
            5 => "bench.Type05",
            6 => "bench.Type06",
            7 => "bench.Type07",
            8 => "bench.Type08",
            9 => "bench.Type09",
            10 => "bench.Type10",
            11 => "bench.Type11",
            12 => "bench.Type12",
            13 => "bench.Type13",
            14 => "bench.Type14",
            15 => "bench.Type15",
            16 => "bench.Type16",
            17 => "bench.Type17",
            18 => "bench.Type18",
            19 => "bench.Type19",
            20 => "bench.Type20",
            21 => "bench.Type21",
            22 => "bench.Type22",
            23 => "bench.Type23",
            24 => "bench.Type24",
            25 => "bench.Type25",
            26 => "bench.Type26",
            27 => "bench.Type27",
            28 => "bench.Type28",
            29 => "bench.Type29",
            30 => "bench.Type30",
            31 => "bench.Type31",
            32 => "bench.Type32",
            33 => "bench.Type33",
            34 => "bench.Type34",
            35 => "bench.Type35",
            36 => "bench.Type36",
            37 => "bench.Type37",
            38 => "bench.Type38",
            39 => "bench.Type39",
            40 => "bench.Type40",
            41 => "bench.Type41",
            42 => "bench.Type42",
            43 => "bench.Type43",
            44 => "bench.Type44",
            45 => "bench.Type45",
            46 => "bench.Type46",
            47 => "bench.Type47",
            48 => "bench.Type48",
            49 => "bench.Type49",
            50 => "bench.Type50",
            51 => "bench.Type51",
            52 => "bench.Type52",
            53 => "bench.Type53",
            54 => "bench.Type54",
            55 => "bench.Type55",
            56 => "bench.Type56",
            57 => "bench.Type57",
            58 => "bench.Type58",
            59 => "bench.Type59",
            60 => "bench.Type60",
            61 => "bench.Type61",
            62 => "bench.Type62",
            63 => "bench.Type63Overflow",
            _ => "bench.InvalidType",
        }
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("bench.type-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.0);
    }
}

fn backend(kind: &str, capacity: usize) -> Arc<dyn CacheBackend> {
    match kind {
        "quick" => Arc::new(QuickCacheBackend::with_capacity(capacity)),
        "moka" => Arc::new(MokaCacheBackend::with_capacity(capacity)),
        _ => unreachable!(),
    }
}

fn moka_backend(capacity: usize, with_size_removal_metrics: bool) -> Arc<dyn CacheBackend> {
    let builder = MokaCacheBackend::builder(capacity);
    let builder = if with_size_removal_metrics {
        builder.with_size_removal_metrics()
    } else {
        builder
    };
    Arc::new(builder.build())
}

async fn register_overflow_types(cache: &LanceCache) {
    macro_rules! register {
        ($($index:literal),* $(,)?) => {
            $(
                black_box(cache.get_with_key(&TypeKey::<$index>(0)).await);
            )*
        };
    }
    register!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62
    );
}

fn benchmark_gauges(c: &mut Criterion, runtime: &tokio::runtime::Runtime, kind: &str) {
    let value = Arc::new(vec![1_u8; 32]);
    for pool_count in [1, 16] {
        let caches: Vec<_> = (0..pool_count)
            .map(|_| LanceCache::with_backend(backend(kind, CAPACITY)))
            .collect();
        runtime.block_on(async {
            for (pool_index, cache) in caches.iter().enumerate() {
                for entry_index in 0..8 {
                    cache
                        .insert_with_key(
                            &PageKey((pool_index * 8 + entry_index) as u64),
                            value.clone(),
                        )
                        .await;
                }
                cache.stats().await;
            }
        });
        refresh_metrics();
        let label = if pool_count == 1 {
            "one_live_pool"
        } else {
            "sixteen_live_pools"
        };
        c.bench_function(&format!("cache_export/gauges/{kind}/{label}"), |b| {
            b.iter(|| {
                refresh_metrics();
                black_box(&caches);
            });
        });
    }
}

fn benchmark(c: &mut Criterion) {
    match std::env::var("LANCE_CACHE_BENCH_MODE").as_deref() {
        Ok("no_recorder") => {}
        Ok("aggregate") => recorder::install(),
        _ => panic!("set LANCE_CACHE_BENCH_MODE to no_recorder or aggregate"),
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let value = Arc::new(vec![1_u8; 32]);

    for kind in ["quick", "moka"] {
        let cache = LanceCache::with_backend(backend(kind, CAPACITY));
        runtime.block_on(async {
            cache.insert_with_key(&PageKey(0), value.clone()).await;
        });
        c.bench_function(&format!("cache_export/lookup/{kind}/hit"), |b| {
            b.to_async(&runtime)
                .iter(|| cache.get_with_key(black_box(&PageKey(0))));
        });

        let load_cache = LanceCache::with_backend(backend(kind, CAPACITY));
        let sequence = AtomicU64::new(0);
        c.bench_function(&format!("cache_export/load/{kind}/success"), |b| {
            b.to_async(&runtime).iter(|| {
                let id = sequence.fetch_add(1, Ordering::Relaxed);
                load_cache.get_or_insert_with_key(PageKey(id), || async { Ok(vec![1_u8; 32]) })
            });
        });

        let error_cache = LanceCache::with_backend(backend(kind, CAPACITY));
        c.bench_function(&format!("cache_export/load/{kind}/error"), |b| {
            b.to_async(&runtime).iter(|| {
                error_cache.get_or_insert_with_key(PageKey(0), || async {
                    Err::<Vec<u8>, _>(Error::timeout("benchmark loader failure"))
                })
            });
        });

        let write_cache = LanceCache::with_backend(backend(kind, CAPACITY));
        runtime.block_on(async {
            write_cache
                .insert_with_key(&PageKey(0), value.clone())
                .await;
        });
        c.bench_function(&format!("cache_export/write/{kind}/replacement"), |b| {
            b.to_async(&runtime)
                .iter(|| write_cache.insert_with_key(&PageKey(0), value.clone()));
        });

        // A zero-capacity insertion is the only ordinary write path that emits
        // the positively identified write-rejection event.
        let rejected_cache = LanceCache::with_backend(backend(kind, 0));
        c.bench_function(&format!("cache_export/write/{kind}/disabled"), |b| {
            b.to_async(&runtime)
                .iter(|| rejected_cache.insert_with_key(&PageKey(0), value.clone()));
        });

        let warm = cache.with_load_origin(CacheLoadOrigin::Warm);
        c.bench_function(&format!("cache_export/warm/{kind}/hit"), |b| {
            b.to_async(&runtime).iter(|| {
                warm.get_or_insert_with_key(PageKey(0), || async {
                    panic!("warm hit unexpectedly ran its loader")
                })
            });
        });

        let warm_load = LanceCache::with_backend(backend(kind, CAPACITY))
            .with_load_origin(CacheLoadOrigin::Warm);
        let warm_sequence = AtomicU64::new(0);
        c.bench_function(&format!("cache_export/warm/{kind}/load"), |b| {
            b.to_async(&runtime).iter(|| {
                let id = warm_sequence.fetch_add(1, Ordering::Relaxed);
                warm_load.get_or_insert_with_key(PageKey(id), || async { Ok(vec![1_u8; 32]) })
            });
        });

        let warm_error = LanceCache::with_backend(backend(kind, CAPACITY))
            .with_load_origin(CacheLoadOrigin::Warm);
        c.bench_function(&format!("cache_export/warm/{kind}/error"), |b| {
            b.to_async(&runtime).iter(|| {
                warm_error.get_or_insert_with_key(PageKey(0), || async {
                    Err::<Vec<u8>, _>(Error::timeout("benchmark warm loader failure"))
                })
            });
        });
    }

    for kind in ["quick", "moka"] {
        benchmark_gauges(c, &runtime, kind);
    }

    let quick_backend = backend("quick", SMALL_CAPACITY);
    let quick_churn = LanceCache::with_backend(quick_backend.clone());
    let quick_runtime = &runtime;
    quick_runtime.block_on(async {
        for id in 0..128 {
            quick_churn
                .insert_with_key(&PageKey(id), value.clone())
                .await;
        }
        let snapshot = quick_churn
            .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
            .await;
        assert!(snapshot.backend.size_removals.unwrap() > 0);
    });
    let sequence = AtomicU64::new(128);
    c.bench_function("cache_export/size_removal/quick/churn", |b| {
        b.to_async(&runtime).iter(|| {
            let id = sequence.fetch_add(1, Ordering::Relaxed) % 512;
            let key = PageKey(id);
            let value = value.clone();
            let cache = &quick_churn;
            async move { cache.insert_with_key(&key, value).await }
        });
    });

    for enabled in [false, true] {
        let mode = if enabled {
            "listener_enabled"
        } else {
            "listener_disabled"
        };
        let replacement_backend = moka_backend(CAPACITY, enabled);
        let replacement_cache = LanceCache::with_backend(replacement_backend);
        runtime.block_on(async {
            replacement_cache
                .insert_with_key(&PageKey(0), value.clone())
                .await;
        });
        c.bench_function(
            &format!("cache_export/size_removal/moka/{mode}/replacement"),
            |b| {
                b.to_async(&runtime)
                    .iter(|| replacement_cache.insert_with_key(&PageKey(0), value.clone()));
            },
        );

        let churn_backend = moka_backend(SMALL_CAPACITY, enabled);
        let churn_cache = LanceCache::with_backend(churn_backend.clone());
        let sequence = AtomicU64::new(128);
        runtime.block_on(async {
            for id in 0..128 {
                churn_cache
                    .insert_with_key(&PageKey(id), value.clone())
                    .await;
            }
            let snapshot = churn_cache
                .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
                .await;
            if enabled {
                assert!(snapshot.backend.size_removals.unwrap() > 0);
            } else {
                assert_eq!(snapshot.backend.size_removals, None);
            }
        });
        c.bench_function(
            &format!("cache_export/size_removal/moka/{mode}/churn"),
            |b| {
                b.to_async(&runtime).iter(|| {
                    let id = sequence.fetch_add(1, Ordering::Relaxed) % 512;
                    let key = PageKey(id);
                    let value = value.clone();
                    let cache = &churn_cache;
                    async move { cache.insert_with_key(&key, value).await }
                });
            },
        );
        let snapshot = runtime.block_on(async {
            churn_cache
                .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
                .await
        });
        if enabled {
            assert!(snapshot.backend.size_removals.unwrap() > 0);
            assert!(snapshot.backend.size_removed_bytes.unwrap() > 0);
        } else {
            assert_eq!(snapshot.backend.size_removals, None);
            assert_eq!(snapshot.backend.size_removed_bytes, None);
        }
    }

    // This is last so no prior case consumes the bounded 64-name type budget.
    let overflow_cache = LanceCache::with_backend(backend("quick", CAPACITY));
    runtime.block_on(async {
        black_box(overflow_cache.get_with_key(&PageKey(0)).await);
        register_overflow_types(&overflow_cache).await;
        black_box(overflow_cache.get_with_key(&TypeKey::<63>(0)).await);
        overflow_cache
            .insert_with_key(&TypeKey::<63>(0), value.clone())
            .await;
        assert!(
            overflow_cache
                .get_with_key(&TypeKey::<63>(0))
                .await
                .is_some()
        );
    });
    c.bench_function("cache_export/type_overflow/quick/hit", |b| {
        b.to_async(&runtime)
            .iter(|| overflow_cache.get_with_key(&TypeKey::<63>(0)));
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
