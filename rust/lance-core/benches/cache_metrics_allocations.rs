// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Untimed allocation probe. Run separately from the timing benchmark.

use std::alloc::{GlobalAlloc, Layout, System};
use std::borrow::Cow;
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;
use std::task::Poll;

use lance_core::cache::{
    CacheBackend, CacheEntry, CacheKey, CacheKeySchema, InternalCacheKey, KeyBuilder, LanceCache,
    MokaCacheBackend, QuickCacheBackend,
};

#[derive(Clone, Copy, Default)]
struct Allocations {
    count: u64,
    bytes: u64,
    sizes: [usize; 32],
    counts: [u64; 32],
}

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<Allocations> = Cell::new(Allocations::default());
}

fn record(size: usize) {
    if TRACK.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|state| {
            let mut value = state.get();
            value.count += 1;
            value.bytes += size as u64;
            if let Some(i) = value
                .sizes
                .iter()
                .zip(value.counts)
                .position(|(s, n)| n == 0 || *s == size)
            {
                value.sizes[i] = size;
                value.counts[i] += 1;
            }
            state.set(value);
        });
    }
}

struct TrackingAllocator;

// SAFETY: allocation and deallocation retain System's layout and ownership
// contracts. Tracking uses only fixed-size thread-local state and never allocates.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

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

fn measure(name: String, operation: impl FnOnce()) -> serde_json::Value {
    ALLOCATIONS.with(|state| state.set(Allocations::default()));
    TRACK.with(|state| state.set(true));
    operation();
    TRACK.with(|state| state.set(false));
    let value = ALLOCATIONS.with(Cell::get);
    let sizes: Vec<_> = value
        .sizes
        .into_iter()
        .zip(value.counts)
        .filter(|(_, count)| *count > 0)
        .map(|(bytes, count)| serde_json::json!({"bytes": bytes, "count": count}))
        .collect();
    serde_json::json!({"operation": name, "iterations": 128, "allocations": value.count, "requested_bytes": value.bytes, "layouts": sizes})
}

#[derive(Clone)]
struct Entry {
    value: CacheEntry,
    size_bytes: usize,
}

fn main() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut results = Vec::new();
    for backend in ["quick", "moka"] {
        for capacity in [0, 1 << 20] {
            let physical: Arc<dyn CacheBackend> = match backend {
                "quick" => Arc::new(QuickCacheBackend::with_capacity(capacity)),
                _ => Arc::new(MokaCacheBackend::with_capacity(capacity)),
            };
            let cache = LanceCache::with_backend(physical);
            runtime.block_on(async {
                cache
                    .insert_with_key(&PageKey(0), Arc::new(vec![1_u8; 32]))
                    .await;
                cache.size().await;
                cache
                    .get_or_insert_with_key(PageKey(0), || async { Ok(vec![1_u8; 32]) })
                    .await
                    .unwrap();
            });
            results.push(measure(
                format!(
                    "{backend}/loader/{}",
                    if capacity == 0 { "disabled" } else { "skipped" }
                ),
                || {
                    runtime.block_on(async {
                        for _ in 0..128 {
                            black_box(
                                cache
                                    .get_or_insert_with_key(PageKey(0), || async {
                                        Ok(vec![1_u8; 32])
                                    })
                                    .await
                                    .unwrap(),
                            );
                        }
                    });
                },
            ));
            if capacity != 0 {
                let value = Arc::new(vec![1_u8; 32]);
                results.push(measure(format!("{backend}/replacement"), || {
                    runtime.block_on(async {
                        for _ in 0..128 {
                            cache.insert_with_key(&PageKey(0), value.clone()).await;
                        }
                    });
                }));
            }
        }
    }
    for variant in ["plain", "sync_listener", "zst_listener"] {
        let builder = moka::future::Cache::builder()
            .max_capacity(1 << 20)
            .weigher(|_: &InternalCacheKey, entry: &Entry| (16 + entry.size_bytes) as u32);
        let cache = match variant {
            "sync_listener" => builder.eviction_listener(|_, _, _| {}).build(),
            "zst_listener" => builder
                .async_eviction_listener(|_, _, _| {
                    Box::pin(std::future::poll_fn(|_| Poll::Ready(())))
                })
                .build(),
            _ => builder.build(),
        };
        let key = InternalCacheKey::from_bytes([0; 16]);
        let value = Entry {
            value: Arc::new(vec![1_u8; 32]),
            size_bytes: 56,
        };
        runtime.block_on(async {
            cache.insert(key, value.clone()).await;
            cache.run_pending_tasks().await;
        });
        results.push(measure(format!("raw_moka/{variant}/replacement"), || {
            runtime.block_on(async {
                for _ in 0..128 {
                    cache.insert(key, value.clone()).await;
                }
            });
        }));
        black_box(&value.value);
    }
    let output = std::env::var_os("LANCE_CACHE_ALLOCATION_OUTPUT")
        .expect("set LANCE_CACHE_ALLOCATION_OUTPUT to the JSON output path");
    std::fs::write(output, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}
