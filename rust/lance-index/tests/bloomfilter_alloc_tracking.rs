// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Allocation-mechanism regression tests for the bloom-filter index hot paths.
//!
//! These live in their own integration-test binary rather than the lib's unit
//! tests because they need to define the binary's `#[global_allocator]`; the
//! lib test binary already defines one for other index types.
//!
//! Two behavior-preserving performance fixes are guarded here by allocation
//! accounting (their returned values are identical, so value assertions alone
//! cannot catch a regression):
//!
//! 1. Deep-sizing a cached bloom-filter index must not serialize the zones:
//!    `Sbbf::size_bytes()` returns the same number as `to_bytes().len()` in
//!    O(1), so a deep-size pass must not allocate anything as large as one
//!    zone's payload.
//! 2. `load_index_with_remapping` only attaches a batch remapper to a freshly
//!    loaded, uniquely-owned index, so it must not deep-copy every zone's
//!    filter: its big-allocation footprint must match a plain load.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion_common::ScalarValue;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_core::cache::LanceCache;
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::bloomfilter::sbbf::Sbbf;
use lance_core::utils::tempfile::TempObjDir;
use lance_core::{ROW_ADDR, Result};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::bloomfilter::{
    BloomFilterIndexBuilder, BloomFilterIndexBuilderParams, BloomFilterIndexPlugin,
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::{ScalarIndexPlugin, VALUE_COLUMN_NAME};
use lance_index::scalar::{BloomFilterQuery, ScalarIndex};
use lance_index_core::remapping::BatchRowIdRemapper;
use lance_io::object_store::ObjectStore;

// The tracking allocator is inert unless the current thread arms it, and only
// the arming thread is counted, so tests running in parallel on other threads
// never contaminate the counts (same pattern as lance-core's
// cache_key_allocations.rs integration test).
struct TrackingAllocator;

#[derive(Clone, Copy)]
struct AllocTrackerState {
    enabled: bool,
    threshold: usize,
    count: usize,
    bytes: usize,
}

thread_local! {
    static ALLOC_TRACKER: Cell<AllocTrackerState> = const {
        Cell::new(AllocTrackerState {
            enabled: false,
            threshold: usize::MAX,
            count: 0,
            bytes: 0,
        })
    };
}

fn record_allocation(size: usize) {
    ALLOC_TRACKER
        .try_with(|tracker| {
            let mut state = tracker.get();
            if state.enabled && size >= state.threshold {
                state.count += 1;
                state.bytes += size;
            }
            tracker.set(state);
        })
        .ok();
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

/// Count allocations of at least `threshold` bytes made by this thread until
/// [`disarm_big_allocation_tracking`] is called.
fn arm_big_allocation_tracking(threshold: usize) {
    ALLOC_TRACKER.with(|tracker| {
        let mut state = tracker.get();
        state.enabled = true;
        state.threshold = threshold;
        state.count = 0;
        state.bytes = 0;
        tracker.set(state);
    });
}

/// Stop counting and return `(count, bytes)` of the big allocations seen.
fn disarm_big_allocation_tracking() -> (usize, usize) {
    ALLOC_TRACKER.with(|tracker| {
        let mut state = tracker.get();
        state.enabled = false;
        (state.count, state.bytes)
    })
}

// Identity batch remapper: keeps every row address unchanged.
#[derive(Debug)]
struct NoopBatchRemapper;

#[async_trait]
impl BatchRowIdRemapper for NoopBatchRemapper {
    async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
        Ok(row_ids.iter().map(|row_id| Some(*row_id)).collect())
    }
}

const ZONE_COUNT: usize = 8;

/// Train and write a bloom-filter index with `ZONE_COUNT` zones of default
/// parameters (a serialized filter of tens of KiB per zone), returning its
/// temp dir, store, and per-zone payload size. Keep the returned `TempObjDir`
/// alive for as long as the store is used.
async fn write_bloomfilter_fixture() -> (TempObjDir, Arc<LanceIndexStore>, usize) {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    // Default parameters: 8192 items at p=0.00057 per zone.
    let zone_payload = Sbbf::with_ndv_fpp(8192, 0.00057).unwrap().size_bytes();
    assert!(
        zone_payload >= 8 * 1024,
        "fixture too small to be meaningful: {zone_payload} bytes"
    );

    let row_count = ZONE_COUNT * 8192;
    let values = Int64Array::from_iter_values(0..row_count as i64);
    let row_addrs = UInt64Array::from_iter_values(0..row_count as u64);
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, DataType::Int64, false),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(values) as _, Arc::new(row_addrs) as _],
    )
    .unwrap();
    let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        stream::once(std::future::ready(Ok(batch))),
    ));

    let mut builder =
        BloomFilterIndexBuilder::try_new(BloomFilterIndexBuilderParams::default()).unwrap();
    builder.train(data_stream).await.unwrap();
    builder.write_index(store.as_ref()).await.unwrap();

    (tmpdir, store, zone_payload)
}

async fn load_index(store: &Arc<LanceIndexStore>, cache: &LanceCache) -> Arc<dyn ScalarIndex> {
    BloomFilterIndexPlugin
        .load_index(store.clone(), &prost_types::Any::default(), 0, None, cache)
        .await
        .unwrap()
}

// The cache-weigher path (`LanceCache::entry_size` on a loaded bloom-filter
// index) must not allocate one full serialized filter per zone.
#[tokio::test]
async fn deep_size_of_loaded_index_allocates_no_zone_payloads() {
    let (_tmpdir, store, zone_payload) = write_bloomfilter_fixture().await;
    let cache = LanceCache::no_cache();

    let index = load_index(&store, &cache).await;
    assert_eq!(
        index.statistics().unwrap()["num_blocks"].as_u64(),
        Some(ZONE_COUNT as u64),
        "fixture must hold {ZONE_COUNT} zones"
    );

    arm_big_allocation_tracking(zone_payload / 2);
    let _ = black_box(index.deep_size_of());
    let (count, bytes) = disarm_big_allocation_tracking();
    assert_eq!(
        count,
        0,
        "deep_size_of performed {count} allocations of at least {} bytes \
         ({bytes} bytes total); sizing must not serialize the zones",
        zone_payload / 2
    );
}

// Attaching a batch remapper to a freshly loaded index must not deep-copy the
// zones: the remapped load's big-allocation footprint has to match a plain
// load of the same index.
#[tokio::test]
async fn load_with_remapping_does_not_copy_zone_payload() {
    let (_tmpdir, store, zone_payload) = write_bloomfilter_fixture().await;
    let cache = LanceCache::no_cache();

    // Warm every lazy path so the two measured loads allocate identically.
    drop(load_index(&store, &cache).await);

    arm_big_allocation_tracking(zone_payload / 2);
    let plain = load_index(&store, &cache).await;
    let (plain_count, plain_bytes) = disarm_big_allocation_tracking();

    arm_big_allocation_tracking(zone_payload / 2);
    let remapped = BloomFilterIndexPlugin
        .load_index_with_remapping(
            store.clone(),
            &prost_types::Any::default(),
            0,
            Some(Arc::new(NoopBatchRemapper)),
            &cache,
        )
        .await
        .unwrap();
    let (remapped_count, remapped_bytes) = disarm_big_allocation_tracking();

    assert_eq!(
        remapped.statistics().unwrap()["num_blocks"].as_u64(),
        Some(ZONE_COUNT as u64)
    );
    assert!(
        remapped_bytes < plain_bytes + zone_payload / 2,
        "remapped load made {remapped_count} big allocations ({remapped_bytes} bytes) vs \
         {plain_count} ({plain_bytes} bytes) for a plain load of the same {ZONE_COUNT}-zone \
         index (zone payload {zone_payload} bytes); attaching the remapper must not \
         deep-copy the zones"
    );

    // The remapped index is fully functional and returns the plain load's
    // results through the batch-remapping search path.
    let query = BloomFilterQuery::Equals(ScalarValue::Int64(Some(1)));
    let plain_result = plain.search(&query, &NoOpMetricsCollector).await.unwrap();
    let remapped_result = remapped
        .search(&query, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(remapped_result, plain_result);
}
