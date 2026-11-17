// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Run with `cargo bench -p lance-core --bench cache_metrics --features metrics
//! --profile release-with-debug`. Criterion reports to stdout by default.

use std::borrow::Cow;
use std::hint::black_box;
use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::Instant;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lance_core::cache::{CacheBackend, CacheKey, LanceCache, MokaCacheBackend, QuickCacheBackend};

struct TypedKey<const TYPE: usize>;

impl<const TYPE: usize> CacheKey for TypedKey<TYPE> {
    type ValueType = u64;

    fn key(&self) -> Cow<'_, str> {
        Cow::Borrowed("resident")
    }

    fn type_name() -> &'static str {
        std::any::type_name::<Self>()
    }
}

#[derive(Clone, Default)]
struct Recorder {
    #[cfg(feature = "metrics")]
    inner: Option<Arc<metrics_util::debugging::DebuggingRecorder>>,
}

impl Recorder {
    fn with<T>(&self, f: impl FnOnce() -> T) -> T {
        #[cfg(feature = "metrics")]
        if let Some(recorder) = &self.inner {
            return metrics::with_local_recorder(recorder.as_ref(), f);
        }
        f()
    }
}

async fn prefill(cache: &LanceCache, types: usize) {
    macro_rules! register {
        ($($id:literal),*) => {$({
            if $id < types {
                cache.insert_with_key(&TypedKey::<$id>, Arc::new(42)).await;
                assert!(cache.get_with_key(&TypedKey::<$id>).await.is_some());
            }
        })*};
    }
    register!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31
    );
}

fn backend(kind: &str, capacity: usize) -> Arc<dyn CacheBackend> {
    match kind {
        "quick" => Arc::new(QuickCacheBackend::with_capacity(capacity)),
        "moka" => Arc::new(MokaCacheBackend::with_capacity(capacity)),
        _ => unreachable!(),
    }
}

fn hits<const TYPE: usize>(
    c: &mut Criterion,
    name: &str,
    cache: &LanceCache,
    runtime: &tokio::runtime::Runtime,
) {
    c.bench_function(&format!("{name}/hit"), |b| {
        b.to_async(runtime)
            .iter(|| async { black_box(cache.get_with_key(&TypedKey::<TYPE>).await.unwrap()) });
    });
    c.bench_function(&format!("{name}/loader_skipped"), |b| {
        b.to_async(runtime).iter(|| {
            cache.get_or_insert_with_key(TypedKey::<TYPE>, || async {
                panic!("resident loader executed")
            })
        });
    });
}

// Persistent workers amortize dispatch over 1024 reads. Results measure
// throughput per read, rather than latency of an individual request.
fn readers<const TYPE: usize>(
    c: &mut Criterion,
    name: &str,
    cache: &LanceCache,
    recorder: &Recorder,
) {
    const BATCH: u64 = 1024;
    const READERS: usize = 4;
    let barrier = Arc::new(Barrier::new(READERS + 1));
    let mut senders = Vec::with_capacity(READERS);
    let mut workers = Vec::with_capacity(READERS);
    for _ in 0..READERS {
        let (tx, rx) = mpsc::channel::<u64>();
        senders.push(tx);
        let barrier = barrier.clone();
        let cache = cache.clone();
        let recorder = recorder.clone();
        workers.push(thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            recorder.with(|| {
                while let Ok(batches) = rx.recv() {
                    barrier.wait();
                    runtime.block_on(async {
                        for _ in 0..batches * BATCH {
                            black_box(cache.get_with_key(&TypedKey::<TYPE>).await.unwrap());
                        }
                    });
                    barrier.wait();
                }
            });
        }));
    }
    let mut group = c.benchmark_group(format!("{name}/readers"));
    group.throughput(Throughput::Elements(READERS as u64 * BATCH));
    group.bench_function(READERS.to_string(), |b| {
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
}

fn benchmark(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for installed in [false, true]
        .into_iter()
        .filter(|v| !v || cfg!(feature = "metrics"))
    {
        let recorder = Recorder {
            #[cfg(feature = "metrics")]
            inner: installed.then(|| Arc::new(metrics_util::debugging::DebuggingRecorder::new())),
        };
        let mode = if installed { "recorder" } else { "no_recorder" };
        recorder.with(|| {
            for kind in ["quick", "moka"] {
                for types in [1, 32] {
                    let cache = LanceCache::with_backend(backend(kind, 1 << 20));
                    runtime.block_on(prefill(&cache, types));
                    let name = format!("cache_metrics/{mode}/{kind}/{types}_types");
                    if types == 1 {
                        hits::<0>(c, &name, &cache, &runtime);
                        readers::<0>(c, &name, &cache, &recorder);
                    } else {
                        hits::<31>(c, &name, &cache, &runtime);
                        readers::<31>(c, &name, &cache, &recorder);
                    }
                }
                let cache = LanceCache::with_backend(backend(kind, 0));
                c.bench_function(&format!("cache_metrics/{mode}/{kind}/load"), |b| {
                    b.to_async(&runtime)
                        .iter(|| cache.get_or_insert_with_key(TypedKey::<0>, || async { Ok(42) }));
                });
            }
        });
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
