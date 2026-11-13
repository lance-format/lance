// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `LANCE_MEMORY_OBSERVE` reports the bytes that the I/O schedulers, the
//! DataFusion memory pools and the MemWAL writers hold.
//!
//! This file must stay a single-test integration binary: Lance reads the
//! switch once per process and the recorder is process-global, so a sibling
//! test could read the switch before the `set_var` below or record into the
//! same recorder.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::execution::memory_pool::MemoryConsumer;
use lance::dataset::mem_wal::{ShardWriter, ShardWriterConfig};
use lance::memory::{METRIC_COMPONENT, describe_metrics, refresh_metrics};
use lance_datafusion::exec::{LanceExecutionOptions, get_session_context, new_session_context};
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use metrics::Unit;
use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
use object_store::path::Path;
use uuid::Uuid;

/// Refreshes the gauges and returns the component gauges, by label value.
fn components(snapshotter: &Snapshotter) -> BTreeMap<String, f64> {
    refresh_metrics();
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(key, _unit, _description, value)| {
            let key = key.key();
            if key.name() != METRIC_COMPONENT {
                return None;
            }
            let component = key.labels().find(|label| label.key() == "component")?;
            match value {
                DebugValue::Gauge(value) => Some((component.value().to_string(), value.0)),
                _ => None,
            }
        })
        .collect()
}

/// Refreshes the gauges until `component` reads `expected`. Reads finish on
/// the runtime's threads and writers release their memory when their tasks
/// end, so a reading can lag behind the step that changes it.
async fn wait_for(snapshotter: &Snapshotter, component: &str, expected: f64) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let read = components(snapshotter).get(component).copied();
        if read == Some(expected) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{component} reads {read:?}, expected {expected}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[test]
fn test_components_report_the_memory_they_hold() {
    // SAFETY: this is the only test in the binary and no other thread runs
    // yet: the runtime that runs Lance is built below.
    unsafe { std::env::set_var("LANCE_MEMORY_OBSERVE", "1") };
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    recorder.install().unwrap();
    describe_metrics();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(check_components(&snapshotter));

    let (unit, description) = snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .find_map(|(key, unit, description, _)| {
            (key.key().name() == METRIC_COMPONENT).then_some((unit, description))
        })
        .unwrap();
    assert_eq!(unit, Some(Unit::Bytes));
    assert!(description.is_some());
}

async fn check_components(snapshotter: &Snapshotter) {
    // A component is not reported before Lance creates one.
    assert_eq!(components(snapshotter), BTreeMap::new());

    // I/O schedulers: the reads they admitted and their callers have not
    // received yet.
    let store = Arc::new(ObjectStore::memory());
    let path = Path::from("data");
    store.put(&path, &[0_u8; 1000]).await.unwrap();
    let config = |use_lite_scheduler| SchedulerConfig {
        io_buffer_size_bytes: 1 << 20,
        use_lite_scheduler: Some(use_lite_scheduler),
    };
    let standard = ScanScheduler::new(store.clone(), config(false));
    let lite = ScanScheduler::new(store, config(true));
    let size = CachedFileSize::unknown();
    let standard_file = standard.open_file(&path, &size).await.unwrap();
    let lite_file = lite.open_file(&path, &size).await.unwrap();
    let standard_read = standard_file.submit_single(0..300, 0);
    // Through the scheduler, so that the read does not borrow `lite_file`,
    // which is dropped below while the read is pending.
    let lite_read = lite.submit_request(lite_file.reader().clone(), vec![0..100], 0, false);
    wait_for(snapshotter, "io_scheduler", 400.0).await;
    drop(standard_read);
    wait_for(snapshotter, "io_scheduler", 100.0).await;
    // A lite scheduler that closes with a read pending leaves the read's bytes
    // reserved in its queue until the read is dropped; the scheduler stops
    // counting when it closes.
    drop((lite_file, lite));
    wait_for(snapshotter, "io_scheduler", 0.0).await;
    drop(lite_read);

    // DataFusion pools: the reservations in the pools Lance builds, with and
    // without spilling.
    let spilling_options = LanceExecutionOptions {
        use_spilling: true,
        mem_pool_size: Some(64 << 20),
        ..Default::default()
    };
    let spilling = new_session_context(&spilling_options);
    let unbounded = get_session_context(&LanceExecutionOptions {
        use_spilling: false,
        ..Default::default()
    });
    let spilling_reservation =
        MemoryConsumer::new("spilling").register(&spilling.runtime_env().memory_pool);
    let unbounded_reservation =
        MemoryConsumer::new("unbounded").register(&unbounded.runtime_env().memory_pool);
    spilling_reservation.try_grow(4096).unwrap();
    unbounded_reservation.try_grow(1000).unwrap();
    wait_for(snapshotter, "datafusion_pool", 5096.0).await;
    drop((
        spilling_reservation,
        unbounded_reservation,
        spilling,
        unbounded,
    ));
    wait_for(snapshotter, "datafusion_pool", 0.0).await;

    // A pool keeps counting after its environment is gone, while a
    // reservation holds it.
    let context = new_session_context(&spilling_options);
    let pool = context.runtime_env().memory_pool.clone();
    drop(context);
    let reservation = MemoryConsumer::new("outlives").register(&pool);
    reservation.try_grow(2048).unwrap();
    wait_for(snapshotter, "datafusion_pool", 2048.0).await;
    drop(reservation);
    wait_for(snapshotter, "datafusion_pool", 0.0).await;

    // MemWAL writers: a memtable, and the pending batches of a WAL-only
    // writer that has no reason to append them yet.
    let uri = "memory://mem_wal";
    let (store, base_path) = ObjectStore::from_uri(uri).await.unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..100))],
    )
    .unwrap();
    let memtable_writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        uri,
        ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            durable_write: false,
            ..Default::default()
        },
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    let wal_only_writer = ShardWriter::open(
        store,
        base_path,
        uri,
        ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            durable_write: false,
            enable_memtable: false,
            max_wal_flush_interval: None,
            ..Default::default()
        },
        schema,
        vec![],
    )
    .await
    .unwrap();
    let empty_memtable_bytes = memtable_writer.memory().retained_bytes();
    memtable_writer.put(vec![batch.clone()]).await.unwrap();
    wal_only_writer.put(vec![batch]).await.unwrap();
    let memtable_bytes = memtable_writer.memory().retained_bytes();
    let pending_bytes = wal_only_writer.memory().retained_bytes();
    assert!(memtable_bytes > empty_memtable_bytes && pending_bytes > 0);
    wait_for(
        snapshotter,
        "mem_wal",
        (memtable_bytes + pending_bytes) as f64,
    )
    .await;
    memtable_writer.close().await.unwrap();
    wal_only_writer.close().await.unwrap();
    wait_for(snapshotter, "mem_wal", 0.0).await;

    let all_released = ["datafusion_pool", "io_scheduler", "mem_wal"]
        .map(|component| (component.to_string(), 0.0))
        .into();
    assert_eq!(components(snapshotter), all_released);
}
