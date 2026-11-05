// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixtures that need access to ShardWriter and WriterState internals.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Wake, Waker};
use std::time::Duration;

use tokio::sync::mpsc;
use uuid::Uuid;

use super::super::{ShardWriter, WriterMode, in_memory_ref};
use crate::dataset::mem_wal::memtable::flush::{MemTableFlusher, TriggerMemTableFlush};
use crate::dataset::mem_wal::wal::{WalFlushSource, WriterCursors};
use crate::dataset::mem_wal::write::ShardWriterConfig;
use crate::dataset::mem_wal::write::handlers::MemTableFlushHandler;
use crate::dataset::mem_wal::write::schema::Incoming;
use crate::dataset::mem_wal::write::test_support::{
    create_local_store, create_pk_test_schema, create_test_batch,
};
use tempfile::TempDir;

/// Prepare a real frozen memtable with its WAL appended but its PK index
/// unapplied. The tests drive the handlers themselves, so no background
/// dispatcher can apply the index before the flush reaches its barrier.
#[cfg_attr(coverage, coverage(off))]
pub(super) async fn pending_pk_flush() -> (
    ShardWriter,
    MemTableFlushHandler,
    TriggerMemTableFlush,
    TempDir,
) {
    let (store, base_path, base_uri, temp) = create_local_store().await;
    let schema = create_pk_test_schema();
    let mut writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            max_wal_flush_interval: Some(Duration::from_secs(60)),
            ..Default::default()
        },
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    writer.task_executor.shutdown_all().await.unwrap();

    let WriterMode::MemTable {
        state,
        writer_state,
        ..
    } = &mut writer.mode
    else {
        unreachable!("the test enables memtables");
    };
    let writer_state = Arc::get_mut(writer_state).unwrap();
    let (index_tx, _index_rx) = mpsc::unbounded_channel();
    let (flush_tx, mut flush_rx) = mpsc::unbounded_channel();
    writer_state.index_apply_tx = index_tx;
    writer_state.memtable_flush_tx = flush_tx;
    let mut state_guard = state.write().await;
    let batch = Incoming::Rows(vec![create_test_batch(&schema, 1, 2)])
        .shape(&state_guard.schema)
        .unwrap()
        .pop()
        .unwrap();
    state_guard.memtable.insert_batch_only(batch).await.unwrap();
    let batch_store = state_guard.memtable.batch_store();
    let wal_result = writer
        .wal_flusher
        .flush(
            &WalFlushSource::BatchStore {
                batch_store: batch_store.clone(),
            },
            batch_store.len(),
        )
        .await
        .unwrap();
    let wal_position = wal_result.entry.unwrap().position;
    state_guard.last_flushed_wal_entry_position = wal_position;
    writer_state.freeze_memtable(&mut state_guard).unwrap();
    drop(state_guard);
    let message = flush_rx.try_recv().unwrap();

    let handler = MemTableFlushHandler::new(
        state.clone(),
        writer_state.memory.clone(),
        Arc::new(MemTableFlusher::new(
            store,
            base_path,
            base_uri,
            "",
            writer.config.shard_id,
            writer.manifest_store.clone(),
        )),
        writer.wal_flusher.clone(),
        writer.epoch,
        writer.stats.clone(),
        None,
        Duration::ZERO,
        writer_state.stop_flush_retries.clone(),
    );
    (writer, handler, message, temp)
}

/// Poll into the cursor wait and prove that its notification wakes this
/// future. Ignore wakes from blocking I/O threads so an unrelated pending
/// storage operation cannot satisfy the assertion.
#[cfg_attr(coverage, coverage(off))]
pub(super) fn assert_waiting_for_cursor(future: Pin<&mut impl Future>, cursors: &WriterCursors) {
    struct CursorWake {
        thread: std::thread::ThreadId,
        wakes: AtomicUsize,
    }

    impl Wake for CursorWake {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            if std::thread::current().id() == self.thread {
                self.wakes.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    let counter = Arc::new(CursorWake {
        thread: std::thread::current().id(),
        wakes: AtomicUsize::new(0),
    });
    let waker = Waker::from(counter.clone());
    assert!(future.poll(&mut Context::from_waker(&waker)).is_pending());
    counter.wakes.store(0, Ordering::Relaxed);
    cursors.wake();
    assert!(
        counter.wakes.load(Ordering::Relaxed) > 0,
        "the flush must be waiting on the writer cursor before index application"
    );
}

/// Recompute what the shard is holding straight from `WriterState`, the
/// way [`ShardWriter::memtable_stats`] reads it — independent of the
/// publish mechanism, which is the thing that can drift.
pub(super) async fn ground_truth(writer: &ShardWriter) -> (usize, usize) {
    let state = writer.memtable_state_lock().unwrap().read().await;
    let active = in_memory_ref(&state.memtable).resident_bytes();
    let frozen = state
        .frozen_memtables
        .iter()
        .filter(|frozen| frozen.flushed_at_ms.is_none())
        .map(|frozen| in_memory_ref(&frozen.memtable).resident_bytes())
        .sum();
    (active, frozen)
}

pub(super) async fn assert_no_drift(writer: &ShardWriter, after: &str) {
    let (active, frozen) = ground_truth(writer).await;
    let memory = writer.memory();
    assert_eq!(
        (memory.active_bytes(), memory.frozen_bytes()),
        (active, frozen),
        "published memory drifted from the writer state after {after}"
    );
    assert_eq!(
        memory.unflushed_bytes(),
        active + frozen,
        "unflushed must be the sum of the two terms after {after}"
    );
}

/// Append the active memtable's un-appended batches to the WAL without
/// going through the flush handler, so the writer state does not learn the
/// new entry position: the window between an append and its bookkeeping.
pub(super) async fn append_active_without_state_update(writer: &ShardWriter) {
    let batch_store = writer
        .memtable_state_lock()
        .unwrap()
        .read()
        .await
        .memtable
        .batch_store();
    let end = batch_store.len();
    let result = writer
        .wal_flusher
        .flush(&WalFlushSource::BatchStore { batch_store }, end)
        .await
        .unwrap();
    assert!(result.entry.is_some(), "the append must write an entry");
}
