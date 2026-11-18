// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Write, WAL, memtable, and observer statistics.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use uuid::Uuid;

use super::*;
use crate::dataset::mem_wal::index::MemIndexSpec;
use crate::dataset::mem_wal::observer::WalObserver;
use crate::dataset::mem_wal::write::test_support::*;
use crate::dataset::mem_wal::write::{ShardWriter, ShardWriterConfig};

#[tokio::test]
async fn test_memtable_stats_frozen_count_outlives_frozen_bytes() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 1024, // small enough to freeze within the loop below
        frozen_memtable_grace: Duration::from_secs(600),
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    let fresh = writer.memtable_stats().await.unwrap();
    assert_eq!(fresh.frozen_count, 0);
    assert_eq!(writer.memory().frozen_bytes(), 0);

    for i in 0..20 {
        let batch = create_test_batch(&schema, i * 10, 10);
        writer.put(vec![batch]).await.unwrap();
    }
    writer.wait_for_flush_drain().await.unwrap();

    let stats = writer.memtable_stats().await.unwrap();
    assert!(
        stats.frozen_count > 0,
        "flushed memtables must stay in the read view for the grace window"
    );
    let memory = writer.memory();
    assert_eq!(
        memory.frozen_bytes(),
        0,
        "every freeze flushed, so nothing is owed to flush"
    );

    // Owing nothing to flush is not the same as holding nothing. Those
    // generations stay resident for the whole grace window, and a
    // process-wide budget has to see them even though the per-shard flush
    // valve deliberately does not meter them.
    assert!(
        memory.grace_bytes() > 0,
        "flushed-but-retained generations hold real memory"
    );
    assert_eq!(
        memory.retained_bytes(),
        memory.unflushed_bytes() + memory.grace_bytes(),
        "retained is the whole footprint; unflushed is only what a flush can reclaim"
    );

    writer.close().await.unwrap();
}

#[test]
fn test_record_put() {
    let stats = WriteStats::new();
    stats.record_put(Duration::from_millis(10));
    stats.record_put(Duration::from_millis(20));

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.put_count, 2);
    assert_eq!(snapshot.put_time, Duration::from_millis(30));
    assert_eq!(snapshot.avg_put_latency(), Some(Duration::from_millis(15)));
}

#[test]
fn test_record_wal_flush() {
    let stats = WriteStats::new();
    stats.record_wal_flush(Duration::from_millis(100), 1024);
    stats.record_wal_flush(Duration::from_millis(200), 2048);

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.wal_flush_count, 2);
    assert_eq!(snapshot.wal_flush_time, Duration::from_millis(300));
    assert_eq!(snapshot.wal_flush_bytes, 3072);
    assert_eq!(snapshot.avg_wal_flush_bytes(), Some(1536));
}

#[test]
fn test_record_memtable_flush() {
    let stats = WriteStats::new();
    stats.record_memtable_flush(Duration::from_secs(1), 10000);

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.memtable_flush_count, 1);
    assert_eq!(snapshot.memtable_flush_time, Duration::from_secs(1));
    assert_eq!(snapshot.memtable_flush_rows, 10000);
}

#[test]
fn test_stats_reset() {
    let stats = WriteStats::new();
    stats.record_put(Duration::from_millis(10));
    stats.record_wal_flush(Duration::from_millis(100), 1024);

    stats.reset();

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.put_count, 0);
    assert_eq!(snapshot.wal_flush_count, 0);
}

#[tokio::test]
async fn test_wal_only_stats_no_memtable_flush() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        wal_only_config(Uuid::new_v4()),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    writer
        .put(vec![create_test_batch(&schema, 0, 1)])
        .await
        .unwrap();

    let stats_handle = writer.stats_handle();
    writer.close().await.unwrap();

    let snapshot = stats_handle.snapshot();
    assert!(snapshot.put_count >= 1, "expected at least one put");
    assert!(
        snapshot.wal_flush_count >= 1,
        "expected at least one WAL flush"
    );
    assert_eq!(
        snapshot.memtable_flush_count, 0,
        "WAL-only mode must never trigger a memtable flush"
    );
    assert_eq!(
        snapshot.index_update_count, 0,
        "WAL-only mode must never trigger an index update"
    );
}

#[tokio::test]
async fn test_memtable_stats_record_index_update() {
    // MemTable mode with a BTree index: index application runs on its own
    // task and must record an index-update stat. Regression for the stat
    // silently reading zero after index apply moved off the WAL-flush path.
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_pk_test_schema();
    let index_specs = vec![MemIndexSpec::btree("id_idx", 0, "id")];

    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        flush_test_config(Uuid::new_v4()),
        schema.clone(),
        index_specs,
    )
    .await
    .unwrap();
    writer
        .put(vec![create_test_batch(&schema, 0, 3)])
        .await
        .unwrap();

    let stats_handle = writer.stats_handle();
    // `close()` drains the index-apply task, so the apply is settled here.
    writer.close().await.unwrap();

    let snapshot = stats_handle.snapshot();
    assert!(
        snapshot.index_update_count >= 1,
        "the index apply must record an index-update stat, got {}",
        snapshot.index_update_count
    );
    assert_eq!(
        snapshot.index_update_rows, 3,
        "every indexed row must be counted exactly once, got {}",
        snapshot.index_update_rows
    );
    assert!(
        snapshot.avg_index_update_latency().is_some(),
        "a recorded index update must expose an average latency"
    );
}

/// A durable put returns only once its WAL flush landed, and the freeze
/// fence resolves only once the frozen memtable became an SSTable — so both
/// callbacks have fired by the time this asserts, without sleeping.
#[tokio::test]
async fn test_observer_sees_both_flush_kinds() {
    #[derive(Debug, Default)]
    struct CountingObserver {
        wal_flushes: AtomicU64,
        wal_bytes: AtomicU64,
        memtable_flushes: AtomicU64,
        memtable_rows: AtomicU64,
    }

    impl WalObserver for CountingObserver {
        fn on_wal_flush(&self, _duration: Duration, bytes: usize) {
            self.wal_flushes.fetch_add(1, Ordering::Relaxed);
            self.wal_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        }

        fn on_memtable_flush(&self, _duration: Duration, rows: usize) {
            self.memtable_flushes.fetch_add(1, Ordering::Relaxed);
            self.memtable_rows.fetch_add(rows as u64, Ordering::Relaxed);
        }
    }

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    let observer = Arc::new(CountingObserver::default());
    let sink: Arc<dyn WalObserver> = observer.clone();
    let config = ShardWriterConfig {
        observer: Some(sink),
        ..freeze_fence_test_config(Uuid::new_v4())
    };

    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    writer
        .put(vec![create_test_batch(&schema, 0, 10)])
        .await
        .unwrap();
    writer
        .force_freeze_active()
        .await
        .unwrap()
        .wait()
        .await
        .unwrap();

    assert!(observer.wal_flushes.load(Ordering::Relaxed) > 0);
    assert!(observer.wal_bytes.load(Ordering::Relaxed) > 0);
    assert_eq!(observer.memtable_flushes.load(Ordering::Relaxed), 1);
    assert_eq!(observer.memtable_rows.load(Ordering::Relaxed), 10);

    writer.close().await.unwrap();
}
