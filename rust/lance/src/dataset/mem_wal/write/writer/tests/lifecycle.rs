// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Auto-flush, close, freeze fences, grace retention, and row-headroom limits.

use super::super::*;
use super::support::*;
use crate::Dataset;
use crate::dataset::mem_wal::memtable::CacheConfig;
use crate::dataset::mem_wal::test_util::{failing_memory_store, observable_store_params};
use crate::dataset::mem_wal::wal::apply_index_range;
use crate::dataset::mem_wal::write::backpressure::{
    Drain, DurabilityCell, DurabilityResult, LocalBackpressureController,
};
use crate::dataset::mem_wal::write::tasks::MessageHandler;
use crate::dataset::mem_wal::write::test_support::*;
use arrow_array::{Array, FixedSizeListArray, Float32Array, Int32Array, StringArray};
use arrow_schema::{DataType, Field};
use lance_core::FenceReason;
use lance_io::object_store::ObjectStoreRegistry;
use rstest::rstest;

/// A PK-only flush must wait for the asynchronous index apply before it
/// publishes the SSTable. Otherwise the manifest can name a generation
/// whose mandatory PK sidecar has not been written yet.
#[tokio::test]
async fn test_pk_only_flush_waits_for_index_apply() {
    let (writer, mut handler, message, temp) = pending_pk_flush().await;
    let TriggerMemTableFlush::Flush { memtable, .. } = &message;
    let memtable = memtable.clone();
    let schema = create_pk_test_schema();
    let indexes = memtable.indexes_arc().unwrap();
    assert_eq!(indexes.indexed_count(), 0);
    let mut flush = Box::pin(handler.handle(message));
    assert_waiting_for_cursor(flush.as_mut(), writer.wal_flusher.cursors());
    assert!(
        writer
            .manifest()
            .await
            .unwrap()
            .unwrap()
            .sstables
            .is_empty()
    );

    apply_index_range(
        writer.wal_flusher.cursors(),
        TriggerIndexApply {
            batch_store: memtable.batch_store(),
            indexes,
            end_batch_position: memtable.batch_count(),
        },
    )
    .await
    .unwrap();
    tokio::time::timeout(Duration::from_secs(1), flush)
        .await
        .expect("flush did not finish after applying its PK index")
        .unwrap();
    let base_uri = format!("file://{}", temp.path().display());
    assert_eq!(
        read_sstable_ids_via_lsm(&writer, schema, &base_uri, writer.config.shard_id, None,).await,
        vec![1, 2],
        "the published PK-only SSTable must be readable through the LSM scanner"
    );
}

/// Abort must wake an in-flight flush before cancellation discards the
/// queued index apply that would otherwise advance its cursor.
#[tokio::test]
async fn test_abort_wakes_flush_waiting_for_index_apply() {
    struct NotifyWhenEntered {
        handler: crate::dataset::mem_wal::write::handlers::MemTableFlushHandler,
        started: Arc<tokio::sync::Notify>,
    }

    #[async_trait::async_trait]
    impl MessageHandler<TriggerMemTableFlush> for NotifyWhenEntered {
        async fn handle(&mut self, message: TriggerMemTableFlush) -> Result<()> {
            let pending = self.handler.handle(message);
            self.started.notify_one();
            pending.await
        }
    }

    let (mut writer, handler, message, _temp) = pending_pk_flush().await;
    let TriggerMemTableFlush::Flush { memtable, .. } = &message;
    let memtable = Arc::clone(memtable);
    writer.task_executor = Arc::new(TaskExecutor::new());
    let indexes = memtable.indexes_arc().unwrap();
    let mut completion = memtable.create_memtable_flush_completion();
    let (flush_tx, flush_rx) = mpsc::unbounded_channel();
    let started = Arc::new(tokio::sync::Notify::new());
    writer
        .task_executor
        .add_handler(
            "memtable_flusher".to_string(),
            Box::new(NotifyWhenEntered {
                handler,
                started: started.clone(),
            }),
            flush_rx,
        )
        .unwrap();
    flush_tx.send(message).unwrap();
    started.notified().await;

    let (index_tx, index_rx) = mpsc::unbounded_channel();
    index_tx
        .send(TriggerIndexApply {
            batch_store: memtable.batch_store(),
            indexes: indexes.clone(),
            end_batch_position: memtable.batch_count(),
        })
        .unwrap();
    writer
        .task_executor
        .add_handler(
            "index_applier".to_string(),
            Box::new(IndexApplyHandler {
                cursors: writer.wal_flusher.cursors().clone(),
                wal_flusher: writer.wal_flusher.clone(),
                stats: writer.stats.clone(),
            }),
            index_rx,
        )
        .unwrap();

    // On this single-thread runtime neither spawned dispatcher can run
    // before abort poisons the cursors and cancels their shared token.
    tokio::time::timeout(Duration::from_secs(1), writer.abort())
        .await
        .expect("abort stranded the flush waiting for the cancelled index apply")
        .unwrap();
    assert_eq!(
        indexes.indexed_count(),
        0,
        "abort must discard the queued apply"
    );
    let Some(DurabilityResult::Failed(message)) = completion.await_value().await else {
        panic!("the pending flush must report abort's terminal failure");
    };
    assert!(message.contains("ShardWriter aborted"), "{message}");
    assert!(
        writer
            .manifest()
            .await
            .unwrap()
            .unwrap()
            .sstables
            .is_empty()
    );
}

/// Regression for #6713 against memory://, which is closer to S3
/// semantics (conditional PUT and list-prefix consistency).
#[tokio::test]
async fn test_shard_writer_auto_flush_repeatedly_memory_store() {
    let base_uri = "memory:///bench_test_flush";
    let (store, base_path) = ObjectStore::from_uri(base_uri).await.unwrap();
    let base_uri = base_uri.to_string();
    let schema = create_test_schema();

    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: true,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    let initial_gen = writer.memtable_stats().await.unwrap().generation;

    // The bug appeared on the second generation because repeated flushes
    // reused the generation-1 path. Queue several flushes before draining
    // so both path uniqueness and background sequencing are exercised.
    for i in 0..8 {
        let batch = create_test_batch(&schema, i * 10, 10);
        writer.put(vec![batch]).await.unwrap();
    }

    writer.wait_for_flush_drain().await.unwrap();

    let stats = writer.memtable_stats().await.unwrap();
    assert!(
        stats.generation >= initial_gen + 3,
        "expected repeated successful flushes; generation went {} → {}",
        initial_gen,
        stats.generation
    );

    writer.close().await.unwrap();
}

/// Regression for #6713: with durable_write=true and a memtable
/// size threshold that fires frequently, the memtable flush task
/// hit "Dataset already exists: …_gen_1" once the second flush
/// started.
#[tokio::test]
async fn test_shard_writer_auto_flush_repeatedly_local_store() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: true,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        // Tiny size threshold — every batch crosses it.
        max_memtable_size: 64,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    let initial_gen = writer.memtable_stats().await.unwrap().generation;

    // Queue multiple generations before waiting. The original failure was
    // deterministic on the second flush, so three committed generations
    // are sufficient to prove that paths and generation IDs advance.
    for i in 0..8 {
        let batch = create_test_batch(&schema, i * 10, 10);
        writer.put(vec![batch]).await.unwrap();
    }

    writer.wait_for_flush_drain().await.unwrap();

    let stats = writer.memtable_stats().await.unwrap();
    assert!(
        stats.generation >= initial_gen + 3,
        "expected repeated successful flushes; generation went {} → {}",
        initial_gen,
        stats.generation
    );

    writer.close().await.unwrap();
}

/// Regression: `close()` must flush the active memtable (not just
/// drain the WAL). Earlier, with a `max_memtable_size` set well
/// above the workload, no auto-flush would fire and `close()`
/// would return without producing a Lance fragment — data was
/// durable in the WAL but no LSM-level generation existed,
/// surprising callers and making flush-cost benchmarks impossible.
///
/// The test verifies, end-to-end:
/// 1. close() returns Ok (a freeze/flush error must propagate, not
///    be silently dropped).
/// 2. The persisted shard manifest's `current_generation` has
///    advanced past the initial generation — direct evidence that
///    a MemTable flush + manifest commit happened during close()
///    rather than the active memtable being dropped on the floor.
/// (Verifying replay's post-flush behavior is tangled with
/// independent replay logic and is exercised by the dedicated
/// `test_memtable_replay_*` tests.)
#[tokio::test]
async fn test_close_flushes_active_memtable() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    // Reuse the same shard_id when reopening so we observe state
    // produced by the first writer's `close()`.
    let shard_id = Uuid::new_v4();
    // Huge size threshold so puts never trigger an auto-flush.
    let config = ShardWriterConfig {
        shard_id,
        shard_spec_id: 0,
        durable_write: true,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: usize::MAX,
        max_unflushed_memtable_bytes: usize::MAX,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        config.clone(),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    let initial_gen = writer.memtable_stats().await.unwrap().generation;

    for i in 0..50 {
        let batch = create_test_batch(&schema, i * 10, 10);
        writer.put(vec![batch]).await.unwrap();
    }

    // Pre-close sanity: no auto-flush should have fired.
    let stats_before = writer.memtable_stats().await.unwrap();
    assert_eq!(
        stats_before.generation, initial_gen,
        "no flush should have fired during puts (size threshold is usize::MAX)"
    );
    assert!(
        stats_before.row_count > 0,
        "memtable should hold the rows we just inserted"
    );

    // close() must succeed; any freeze/flush error must propagate.
    writer
        .close()
        .await
        .expect("close() must succeed and propagate any freeze/flush error");

    // Reopen the same shard and read the persisted manifest. The
    // active memtable from the closed writer was frozen + flushed
    // inside close(), which must have committed a new manifest
    // recording the advanced generation.
    let reopened = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();
    let manifest = reopened
        .manifest()
        .await
        .unwrap()
        .expect("reopened shard must have a persisted manifest");
    assert!(
        manifest.current_generation > initial_gen,
        "expected manifest current_generation to advance past {} after close() flushed the active memtable; got {}",
        initial_gen,
        manifest.current_generation,
    );

    reopened.close().await.unwrap();
}

#[rstest]
#[case::memtable(true)]
#[case::wal_only(false)]
#[tokio::test]
async fn test_close_propagates_final_wal_persistence_failure(#[case] enable_memtable: bool) {
    let (store, base_path, controls) = failing_memory_store().await;
    let base_uri = "memory:///";
    let schema = create_test_schema();
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        enable_memtable,
        max_wal_buffer_size: usize::MAX,
        max_wal_flush_interval: None,
        max_wal_persist_retries: 0,
        max_memtable_size: usize::MAX,
        max_unflushed_memtable_bytes: usize::MAX,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };
    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();
    let task_executor = writer.task_executor.clone();

    writer
        .put(vec![create_test_batch(&schema, 0, 10)])
        .await
        .unwrap();
    controls.fail_wal_puts(usize::MAX);

    let error = writer
        .close()
        .await
        .expect_err("close must propagate the final WAL persistence failure");
    assert_eq!(error.fence_reason(), Some(FenceReason::PersistenceFailure));
    assert!(
        error
            .to_string()
            .contains("injected transient WAL put failure"),
        "unexpected error: {error}"
    );
    task_executor.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn test_force_freeze_active_and_wait_for_flush_drain() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    // Thresholds high enough that auto-flush won't fire; the freeze is
    // the only thing that should rotate the memtable.
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 64 * 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    let initial_gen = writer.memtable_stats().await.unwrap().generation;
    let flushed_before = writer
        .manifest()
        .await
        .unwrap()
        .map(|m| m.sstables.len())
        .unwrap_or(0);

    writer
        .put(vec![create_test_batch(&schema, 0, 10)])
        .await
        .unwrap();
    let fence = writer.force_freeze_active().await.unwrap();
    assert_eq!(fence.frozen_generation(), Some(initial_gen));
    fence.wait().await.unwrap();

    let stats = writer.memtable_stats().await.unwrap();
    assert_eq!(stats.generation, initial_gen + 1);
    assert_eq!(stats.batch_count, 0);

    let manifest = writer
        .manifest()
        .await
        .unwrap()
        .expect("manifest should exist after flush");
    assert_eq!(manifest.sstables.len(), flushed_before + 1);

    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_freeze_fence_observes_poison_without_flush_completion() {
    let completion = DurabilityCell::new();
    let cursors = Arc::new(WriterCursors::new(true));
    let fence = FreezeFence {
        frozen_generation: Some(1),
        watchers: vec![completion.reader()],
        cursors: cursors.clone(),
    };
    let waiting = fence.wait();
    tokio::pin!(waiting);
    assert!(futures::poll!(&mut waiting).is_pending());

    // No flush completion arrives, and its sender stays alive. The
    // terminal state itself must wake an already waiting fence.
    cursors.mark_terminal_failure(&Error::writer_poisoned("SSTable flush failed"));
    let error = tokio::time::timeout(Duration::from_secs(1), waiting)
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(error.fence_reason(), Some(FenceReason::PersistenceFailure));
    assert!(error.to_string().contains("SSTable flush failed"));
}

/// An empty active memtable does not mean every pre-call write is in
/// SSTable: a size/interval trigger swaps generation N for an empty N+1
/// while N's flush is still in flight. The freeze must fence that
/// generation anyway — reporting "nothing frozen, nothing to wait for"
/// lets a caller acknowledge a flush that has not happened.
#[tokio::test]
async fn test_force_freeze_active_fences_pending_generation_when_active_is_empty() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        freeze_fence_test_config(Uuid::new_v4()),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    writer
        .put(vec![create_test_batch(&schema, 0, 10)])
        .await
        .unwrap();

    // Stop the flush tasks, then freeze by hand: this is the post-rotation
    // state held still — generation N frozen and unflushed, N+1 active and
    // empty.
    writer.task_executor.shutdown_all().await.unwrap();
    let pending_generation = match &writer.mode {
        WriterMode::MemTable {
            state,
            writer_state,
            ..
        } => {
            let mut state = state.write().await;
            writer_state.freeze_memtable(&mut state).unwrap() - 1
        }
        WriterMode::WalOnly { .. } => unreachable!("opened in memtable mode"),
    };

    let fence = writer.force_freeze_active().await.unwrap();
    assert_eq!(
        fence.frozen_generation(),
        None,
        "the active memtable was empty, so this freeze froze nothing"
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(200), fence.wait())
            .await
            .is_err(),
        "generation {pending_generation} is still awaiting flush; the fence must not be satisfied"
    );
}

/// The fence is tied to each flush's own outcome, independently of changes
/// to `ShardManifest::current_generation` made by other manifest users.
#[tokio::test]
async fn test_force_freeze_active_fence_ignores_manifest_generation_advance() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        freeze_fence_test_config(Uuid::new_v4()),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    writer
        .put(vec![create_test_batch(&schema, 0, 10)])
        .await
        .unwrap();

    // No flush task, so the frozen generation never becomes an SSTable.
    writer.task_executor.shutdown_all().await.unwrap();
    let fence = writer.force_freeze_active().await.unwrap();
    let frozen = fence
        .frozen_generation()
        .expect("the active memtable held rows");

    // Simulate a manifest change that did not flush this writer's memtable.
    writer
        .manifest_store
        .commit_update(writer.epoch(), |current| ShardManifest {
            version: current.next_version(),
            current_generation: frozen + 2,
            ..current.clone()
        })
        .await
        .unwrap();

    assert!(
        tokio::time::timeout(Duration::from_millis(200), fence.wait())
            .await
            .is_err(),
        "generation {frozen} never became an SSTable; a manifest advance past it must not satisfy its fence"
    );
}

/// `abort` tears down the background flush tasks WITHOUT flushing —
/// buffered memtable rows are discarded, not frozen into an SSTable
/// generation the way `close` would. Idempotent on a second call.
#[tokio::test]
async fn test_abort_discards_without_flushing_and_is_idempotent() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    // Thresholds high enough that nothing auto-flushes; the rows stay
    // in the active memtable until abort discards them.
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 64 * 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    writer
        .put(vec![create_test_batch(&schema, 0, 10)])
        .await
        .unwrap();
    let flushed_before = writer
        .manifest()
        .await
        .unwrap()
        .map(|m| m.sstables.len())
        .unwrap_or(0);

    writer.abort().await.unwrap();

    // No generation was frozen — contrast with `close`, which flushes
    // the 10 buffered rows into a new SSTable.
    let flushed_after = writer
        .manifest()
        .await
        .unwrap()
        .map(|m| m.sstables.len())
        .unwrap_or(0);
    assert_eq!(
        flushed_after, flushed_before,
        "abort must not flush a new SSTable"
    );

    // Idempotent: re-cancels the already-cancelled token, joins an
    // already-emptied task set.
    writer.abort().await.unwrap();
}

/// On a successful flush commit the frozen generation's rows land in the
/// manifest immediately, but the in-memory handle is NOT dropped — it
/// lingers for `frozen_memtable_grace` (so in-flight as-of reads keep
/// batch-resolved membership), then is swept by the grace sweep.
#[tokio::test]
async fn test_frozen_retained_during_grace_then_swept() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 64 * 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        // Short grace so the sweep is observable without a slow test.
        frozen_memtable_grace: Duration::from_millis(50),
        ..Default::default()
    };
    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    let initial_gen = writer.memtable_stats().await.unwrap().generation;
    writer
        .put(vec![create_test_batch(&schema, 0, 10)])
        .await
        .unwrap();
    writer.force_freeze_active().await.unwrap();
    writer.wait_for_flush_drain().await.unwrap();

    // Recorded in the manifest at commit time.
    let manifest = writer.manifest().await.unwrap().expect("manifest exists");
    assert!(
        manifest
            .sstables
            .iter()
            .any(|g| g.generation == initial_gen),
        "SSTable must be recorded in the manifest"
    );

    // Still queryable in memory immediately after commit (within grace).
    let refs = writer.in_memory_memtable_refs().await.unwrap();
    assert_eq!(refs.active.generation, initial_gen + 1);
    assert!(
        refs.frozen.iter().any(|f| f.generation == initial_gen),
        "SSTable must stay queryable during the grace window"
    );

    // After the grace elapses (plus a sweep tick) the handle is evicted.
    tokio::time::sleep(Duration::from_millis(250)).await;
    let refs = writer.in_memory_memtable_refs().await.unwrap();
    assert!(
        refs.frozen.is_empty(),
        "frozen handle must be swept once the grace elapses"
    );

    writer.close().await.unwrap();
}

/// With zero grace (the default) a frozen handle is evicted synchronously on
/// flush commit — no sweep tick, no lingering window.
#[tokio::test]
async fn test_frozen_evicted_immediately_with_zero_grace() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 64 * 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        frozen_memtable_grace: Duration::ZERO,
        ..Default::default()
    };
    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    let initial_gen = writer.memtable_stats().await.unwrap().generation;
    writer
        .put(vec![create_test_batch(&schema, 0, 10)])
        .await
        .unwrap();
    writer.force_freeze_active().await.unwrap();
    writer.wait_for_flush_drain().await.unwrap();

    // Rows are durably in the manifest...
    let manifest = writer.manifest().await.unwrap().expect("manifest exists");
    assert!(
        manifest
            .sstables
            .iter()
            .any(|g| g.generation == initial_gen),
        "SSTable must be recorded in the manifest"
    );

    // ...and the in-memory handle is already gone, no sweep tick needed.
    let refs = writer.in_memory_memtable_refs().await.unwrap();
    assert!(
        refs.frozen.is_empty(),
        "frozen handle must be evicted on commit when grace is zero"
    );

    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_close_propagates_frozen_memtable_flush_failure() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();
    let writer_a = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        memtable_config_with_pk(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    writer_a
        .put(vec![create_test_batch(&schema, 0, 10)])
        .await
        .unwrap();

    let writer_b = ShardWriter::open(
        store,
        base_path,
        base_uri,
        memtable_config_with_pk(shard_id),
        schema,
        vec![],
    )
    .await
    .unwrap();
    assert!(writer_b.epoch() > writer_a.epoch());

    let error = writer_a
        .close()
        .await
        .expect_err("close must propagate the fenced MemTable flush");
    assert_eq!(error.fence_reason(), Some(FenceReason::PeerClaimedEpoch));
    assert!(
        error.to_string().contains("Writer fenced"),
        "unexpected error: {error}"
    );

    writer_b.close().await.unwrap();
}

#[tokio::test]
async fn test_close_shuts_down_after_sstable_failure_during_index_drain() {
    let (_store, _base_path, base_uri, _temp) = create_local_store().await;
    let (params, controls) = observable_store_params();
    let (store, base_path) = ObjectStore::from_uri_and_params(
        Arc::new(ObjectStoreRegistry::default()),
        &base_uri,
        &params,
    )
    .await
    .unwrap();
    let schema = schema_with_pk();
    let config = ShardWriterConfig {
        store_params: Some(params),
        ..memtable_config_with_pk(Uuid::new_v4())
    };
    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();
    writer
        .put(vec![create_test_batch(&schema, 0, 1)])
        .await
        .unwrap();
    controls.hold_sstable_puts();
    controls.fail_sstable_puts(1);
    let _first_fence = writer.force_freeze_active().await.unwrap();
    writer
        .put_no_wait(vec![create_test_batch(&schema, 1, 1)])
        .await
        .unwrap();
    let executor = writer.task_executor.clone();
    let cancellation = executor.child_token();
    let cursors = writer.wal_flusher.cursors().clone();
    let state = Arc::downgrade(writer.memtable_state_lock().unwrap());

    let error = {
        let closing = writer.close();
        tokio::pin!(closing);
        // Enter the index drain before allowing the frozen generation to fail.
        assert!(futures::poll!(&mut closing).is_pending());
        controls.release_sstable_puts();
        tokio::time::timeout(Duration::from_secs(1), cursors.wait_for_failure())
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), closing)
            .await
            .unwrap()
            .unwrap_err()
    };
    assert_eq!(error.fence_reason(), Some(FenceReason::PersistenceFailure));
    assert!(error.to_string().contains("injected SSTable write failure"));

    assert!(
        cancellation.is_cancelled(),
        "close must cancel its background tasks"
    );
    let has_retained_state = state.upgrade().is_some();
    // Clean up even when checking the regression against the broken close path.
    executor.shutdown_all().await.unwrap();
    assert!(state.upgrade().is_none());
    assert!(
        !has_retained_state,
        "close must join tasks and release writer state before returning"
    );
}

/// A later generation cannot commit after a flush gives up retrying, even
/// after storage heals. Both generations must remain recoverable from the WAL.
#[tokio::test]
async fn test_terminal_sstable_failure_is_sticky_and_replay_recovers_later_generations() {
    let (_store, _base_path, base_uri, _temp) = create_local_store().await;
    let (params, controls) = observable_store_params();
    let (store, base_path) = ObjectStore::from_uri_and_params(
        Arc::new(ObjectStoreRegistry::default()),
        &base_uri,
        &params,
    )
    .await
    .unwrap();
    let schema = create_test_schema();
    let config = ShardWriterConfig {
        store_params: Some(params),
        ..freeze_fence_test_config(Uuid::new_v4())
    };
    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        config.clone(),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    writer
        .put(vec![create_test_batch(&schema, 0, 1)])
        .await
        .unwrap();
    writer
        .force_freeze_active()
        .await
        .unwrap()
        .wait()
        .await
        .unwrap();
    let before = writer
        .manifest_store
        .refresh_latest()
        .await
        .unwrap()
        .unwrap();
    let attempts_before = controls.sstable_attempts();

    controls.hold_sstable_puts();
    controls.fail_sstable_puts(1);
    writer
        .put(vec![create_test_batch(&schema, 1, 1)])
        .await
        .unwrap();
    let first = writer.force_freeze_active().await.unwrap();
    writer
        .put(vec![create_test_batch(&schema, 2, 1)])
        .await
        .unwrap();
    let second = writer.force_freeze_active().await.unwrap();
    let mut last_flush = second.watchers.last().unwrap().clone();

    // Closing stops retries. Exercise the resulting terminal failure with
    // both generations queued, so healthy storage cannot publish past it.
    let WriterMode::MemTable { writer_state, .. } = &writer.mode else {
        unreachable!("opened in memtable mode")
    };
    writer_state.stop_flush_retries.cancel();
    controls.release_sstable_puts();
    tokio::time::timeout(Duration::from_secs(1), last_flush.await_value())
        .await
        .unwrap()
        .unwrap();
    let after = writer
        .manifest_store
        .refresh_latest()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(after.version, before.version);
    assert_eq!(after.current_generation, before.current_generation);
    assert_eq!(
        after.replay_after_wal_entry_position,
        before.replay_after_wal_entry_position
    );
    assert_eq!(after.sstables.len(), 1);
    for result in [
        first.wait().await,
        second.wait().await,
        writer.wait_for_flush_drain().await,
        writer.wait_for_flush_drain().await,
        writer.force_freeze_active().await.map(|_| ()),
        writer
            .put(vec![create_test_batch(&schema, 3, 1)])
            .await
            .map(|_| ()),
        writer.scan().await.map(|_| ()),
    ] {
        let error = result.unwrap_err();
        assert_eq!(error.fence_reason(), Some(FenceReason::PersistenceFailure));
        assert!(error.to_string().contains("injected SSTable write failure"));
    }
    assert_eq!(controls.sstable_attempts(), attempts_before + 1);
    assert_eq!(writer.memtable_stats().await.unwrap().frozen_count, 2);

    // Watchers have retired, but close must still report the typed failure
    // and finish promptly. Recovery replays both uncommitted generations.
    let error = tokio::time::timeout(Duration::from_secs(1), writer.close())
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(error.fence_reason(), Some(FenceReason::PersistenceFailure));
    let shard_id = config.shard_id;
    let reopened = ShardWriter::open(store, base_path, base_uri.clone(), config, schema, vec![])
        .await
        .unwrap();
    // Replay preserves generation boundaries: generation 2 is published
    // while opening, and generation 3 remains in the active memtable.
    let recovered = reopened
        .manifest_store
        .refresh_latest()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(recovered.sstables.len(), 2);
    for (sstable, id) in recovered.sstables.iter().zip(0..2) {
        let uri = format!("{base_uri}/_mem_wal/{shard_id}/{}", sstable.path);
        let rows = Dataset::open(&uri)
            .await
            .unwrap()
            .scan()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            rows["id"].as_ref(),
            &Int32Array::from(vec![id]) as &dyn Array
        );
    }
    assert_eq!(reopened.memtable_stats().await.unwrap().row_count, 1);
    let rows = reopened
        .scan()
        .await
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(rows.num_rows(), 1);
    assert_eq!(
        rows["id"].as_ref(),
        &Int32Array::from(vec![2]) as &dyn Array
    );
    let manifest_store = reopened.manifest_store.clone();
    reopened.close().await.unwrap();
    let committed = manifest_store.refresh_latest().await.unwrap().unwrap();
    assert_eq!(committed.sstables.len(), 3);
    assert_eq!(
        committed
            .sstables
            .iter()
            .map(|s| s.physical_rows.unwrap())
            .sum::<u64>(),
        3
    );
}

/// A failed flush retains its rows and their memory charge for recovery,
/// while reads fail fast on the poisoned writer.
/// Failure is induced deterministically by fencing the writer with a
/// successor before the freeze, so the flush's `check_fenced` rejects it.
#[tokio::test]
async fn test_frozen_retained_after_failed_flush() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let shard_id = Uuid::new_v4();

    let writer_a = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        memtable_config_with_pk(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    let initial_gen = writer_a.memtable_stats().await.unwrap().generation;
    writer_a
        .put(vec![create_test_batch(&schema, 0, 10)])
        .await
        .unwrap();

    // Successor claims a higher epoch, fencing A.
    let writer_b = ShardWriter::open(
        store,
        base_path,
        base_uri,
        memtable_config_with_pk(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    assert!(writer_b.epoch() > writer_a.epoch());

    // `force_freeze_active` would reject up-front on a fenced writer;
    // freeze directly so the failure surfaces at flush-commit time —
    // exactly the freeze/flush race the fix guards.
    match &writer_a.mode {
        WriterMode::MemTable {
            state,
            writer_state,
            ..
        } => {
            let mut st = state.write().await;
            writer_state.freeze_memtable(&mut st).unwrap();
        }
        WriterMode::WalOnly { .. } => unreachable!("opened in memtable mode"),
    }

    // The fenced flush fails; the drain surfaces that error.
    assert!(
        writer_a.wait_for_flush_drain().await.is_err(),
        "fenced flush should fail the drain"
    );

    let error = writer_a.in_memory_memtable_refs().await.err().unwrap();
    assert_eq!(error.fence_reason(), Some(FenceReason::PeerClaimedEpoch));
    let WriterMode::MemTable { writer_state, .. } = &writer_a.mode else {
        unreachable!();
    };
    let refs = writer_state.memory.load_full();
    assert_eq!(refs.frozen.len(), 1, "frozen generation must be retained");
    assert_eq!(refs.frozen[0].generation, initial_gen);
    assert!(
        !refs.frozen[0].batch_store.is_empty(),
        "retained frozen memtable must still hold its rows"
    );
    assert_eq!(refs.active.as_ref().unwrap().generation, initial_gen + 1);

    // Nor did it vanish from the accounting. A failed flush leaves its
    // memtable un-stamped, so it stays in the owed-to-flush set and keeps
    // metering — those bytes are resident and only another flush reclaims
    // them. This is also what an operator reads to decide whether to evict.
    let stats = writer_a.memtable_stats().await.unwrap();
    assert_eq!(stats.frozen_count, 1);
    let frozen_bytes = writer_a.memory().frozen_bytes();
    assert!(
        frozen_bytes >= refs.frozen[0].batch_store.row_bytes(),
        "a failed flush must keep owing its resident bytes, got {frozen_bytes}"
    );

    // Charging those bytes must not become a trap. Nothing retries the
    // failed generation, and its watcher came off the queue when the flush
    // reported, so there is no event left that would drain the pool — while
    // the valve itself is what keeps the puts that could freeze a new
    // generation from arriving. Waiting here would never end, so the
    // controller refuses instead.
    assert!(
        matches!(writer_a.memory().drain(), Drain::Stalled),
        "a failed flush leaves nothing outstanding to wait on"
    );
    let controller = LocalBackpressureController::new(&ShardWriterConfig {
        max_unflushed_memtable_bytes: writer_a.memory().unflushed_bytes(),
        ..Default::default()
    });
    let refused = tokio::time::timeout(
        Duration::from_secs(10),
        controller.maybe_apply_backpressure(writer_a.memory()),
    )
    .await
    .expect("a shard nothing can drain must refuse, not park the writer");
    assert!(
        refused.is_err_and(|e| e.is_backpressure()),
        "the refusal must be the retryable backpressure signal"
    );

    writer_b.close().await.unwrap();
}

/// A one-row slice pins its whole parent, so a memtable can hold megabytes
/// while its row window reads a few dozen bytes. With only the row arm the
/// shard crossed its ceiling with nothing frozen and no freeze reachable — the
/// valve then found `Drain::Stalled` and refused every put, permanently.
/// The resident arm is what makes that shard drain instead.
#[tokio::test]
async fn test_pinned_parents_freeze_before_the_ceiling_traps_the_writer() {
    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]));

    let chunk = 1_000_000; // ~4MB parent per put
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        // Far above anything the one-row windows will ever sum to, so the row
        // arm cannot be what freezes here.
        max_memtable_size: 1024 * 1024,
        // Never filled, so the capacity arm cannot be it either.
        max_memtable_batches: 1024,
        max_unflushed_memtable_bytes: 8 * 1024 * 1024,
        ..Default::default()
    };
    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    for i in 0..6i32 {
        let parent = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(
                (0..chunk).map(|v| v + i).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        tokio::time::timeout(
            Duration::from_secs(10),
            writer.put(vec![parent.slice(0, 1)]),
        )
        .await
        .expect("a shard the resident arm can freeze must not park forever")
        .unwrap_or_else(|e| panic!("put {i} was refused, so the ceiling is still a trap: {e}"));
    }

    let stats = writer.memtable_stats().await.unwrap();
    assert!(
        stats.generation > 0,
        "the pinned parents must have frozen a generation; row bytes never \
         came close to max_memtable_size"
    );
    assert!(
        writer.memory().row_bytes() < 1024,
        "the row window must still be tiny — otherwise the row arm did the \
         freezing and this proves nothing"
    );

    writer.close().await.unwrap();
}

/// Index memory is not charged to `max_memtable_size`, so it is the resident
/// arm that has to notice a memtable whose indexes rather than its rows are
/// filling the ceiling.
#[tokio::test]
async fn test_resident_arm_freezes_on_index_memory() {
    let schema = create_test_schema();
    let mut memtable =
        MemTable::with_capacity(schema.clone(), 1, vec![], CacheConfig::default(), 64).unwrap();
    memtable
        .insert(create_test_batch(&schema, 0, 50))
        .await
        .unwrap();

    let resident = memtable_resident_bytes(&memtable);
    let rows = memtable.batch_store().row_bytes();
    assert!(
        resident > rows,
        "the fixture needs non-row memory to be measuring anything: \
         resident {resident} vs rows {rows}"
    );

    // Row arm way out of range; only the resident arm can fire.
    assert!(
        memtable_reached_flush_threshold(&memtable, usize::MAX, usize::MAX, resident, 1, 1),
        "resident bytes at the ceiling must freeze"
    );
    assert!(
        !memtable_reached_flush_threshold(&memtable, usize::MAX, usize::MAX, resident + 1, 1, 1),
        "and must not freeze below it"
    );
}

/// The row arm answers for the rows about to arrive, not the rows already
/// inserted: the cap is a hard index capacity.
#[tokio::test]
async fn test_row_arm_freezes_on_max_memtable_rows() {
    let schema = create_test_schema();
    let mut memtable =
        MemTable::with_capacity(schema.clone(), 1, vec![], CacheConfig::default(), 64).unwrap();
    memtable
        .insert(create_test_batch(&schema, 0, 50))
        .await
        .unwrap();

    // Byte and resident arms out of range; only the row arm can fire.
    let row_arm = |cap, incoming| {
        memtable_reached_flush_threshold(&memtable, usize::MAX, cap, usize::MAX, 1, incoming)
    };
    assert!(!row_arm(50, 0), "50 rows under a cap of 50 must not freeze");
    assert!(row_arm(50, 1), "no room for one more row must freeze");
    assert!(
        !row_arm(60, 10),
        "a put that exactly fills the cap must not freeze"
    );
    assert!(
        row_arm(60, 11),
        "a put that would overflow the cap must freeze"
    );
}

/// A memtable stays within `max_memtable_rows` with the byte and batch arms
/// far out of reach, so only the row arm can freeze it.
#[tokio::test]
async fn test_put_freezes_on_max_memtable_rows() {
    let (store, base_path, base_uri, _t) = create_local_store().await;
    let schema = create_test_schema();
    let cap = 64;
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        durable_write: false,
        max_memtable_rows: cap,
        // Both far out of reach, so only the row arm can freeze.
        max_memtable_size: 64 * 1024 * 1024,
        max_memtable_batches: 8_000,
        ..Default::default()
    };
    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    for round in 0..10i32 {
        writer
            .put(vec![create_test_batch(&schema, round * 10, 10)])
            .await
            .unwrap();
        let stats = writer.memtable_stats().await.unwrap();
        assert!(
            stats.row_count <= cap,
            "the active memtable holds {} rows, past the cap of {cap}",
            stats.row_count
        );
    }

    assert!(
        writer.memtable_stats().await.unwrap().generation > 1,
        "100 rows under a cap of {cap} must have rotated at least once"
    );

    writer.close().await.unwrap();
}

/// A put is never split across memtables, so one larger than the cap is
/// rejected as invalid input rather than overflowing a fresh memtable.
#[tokio::test]
async fn test_put_rejects_more_rows_than_a_memtable_holds() {
    let (store, base_path, base_uri, _t) = create_local_store().await;
    let schema = create_test_schema();
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        durable_write: false,
        max_memtable_rows: 8,
        ..Default::default()
    };
    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    // Split across two batches: the cap is on the put, not on one batch.
    let err = writer
        .put(vec![
            create_test_batch(&schema, 0, 5),
            create_test_batch(&schema, 5, 4),
        ])
        .await
        .expect_err("a put of 9 rows under a cap of 8 must be rejected");
    assert!(
        matches!(err, Error::InvalidInput { .. }),
        "an oversized put is caller error, not a writer fault: {err}"
    );
    assert!(
        err.to_string().contains("max_memtable_rows=8"),
        "the error must name the knob and its value, got: {err}"
    );

    // Exactly the cap still goes through, and the writer is unharmed.
    writer
        .put(vec![create_test_batch(&schema, 0, 8)])
        .await
        .unwrap();
    writer.close().await.unwrap();
}

/// An HNSW graph is sized to `max_memtable_rows`, so a shard that writes past
/// the cap has to keep freezing for the graph to never see a row it cannot
/// hold.
#[tokio::test]
async fn test_hnsw_index_survives_a_shard_that_outgrows_the_row_cap() {
    use lance_arrow::FixedSizeListArrayExt;

    let (store, base_path, base_uri, _t) = create_local_store().await;
    let dim = 8;
    let schema = hnsw_schema(dim);
    let cap = 64;
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        durable_write: false,
        max_memtable_rows: cap,
        max_memtable_size: 64 * 1024 * 1024,
        max_memtable_batches: 8_000,
        ..Default::default()
    };
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        config,
        schema.clone(),
        hnsw_specs(),
    )
    .await
    .unwrap();

    for round in 0..20i32 {
        let ids: Vec<i32> = (0..10).map(|v| v + round * 10).collect();
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(
                (0..10 * dim as usize)
                    .map(|v| v as f32 * 0.01)
                    .collect::<Vec<_>>(),
            ),
            dim,
        )
        .unwrap();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(ids)), Arc::new(vectors)],
        )
        .unwrap();
        writer
            .put(vec![batch])
            .await
            .unwrap_or_else(|e| panic!("put {round} was refused: {e}"));
    }

    // The index apply runs outside the put, so a poisoned writer surfaces here
    // even when every put returned Ok.
    writer.close().await.unwrap();
}

/// The post-insert freeze check runs inside the writer lock; the index apply
/// that follows it runs outside. So a put's index growth is invisible to the
/// only check that put makes, and the *next* put is gated by the valve before
/// it can insert and check again — leaving a shard over its ceiling with
/// nothing frozen and every write refused. Replay reaches the same state by
/// building its final memtable's indexes after its last check.
#[tokio::test]
async fn test_index_growth_after_the_freeze_check_still_drains() {
    let (store, base_path, base_uri, _t) = create_local_store().await;
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, true),
    ]));
    // Keys long enough to spill out of the skiplist nodes, so the index heap
    // grows with the column instead of staying a fixed reservation.
    let btree = vec![MemIndexSpec::btree("text_idx", 1, "text")];

    let rows = 2_000usize;
    let width = 512usize;
    let payload = rows * width;
    let ceiling = payload + payload / 2;

    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        // As high as the open-time check allows, keeping the row arm out of
        // reach of the rows below so only the resident arm can freeze.
        max_memtable_size: ceiling - 64 * 1024,
        max_memtable_batches: 1024,
        // Above the rows alone, below rows plus the index heap they build.
        max_unflushed_memtable_bytes: ceiling,
        ..Default::default()
    };

    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), btree)
        .await
        .unwrap();

    for i in 0..3i32 {
        let ids: Vec<i32> = (0..rows as i32).map(|v| v + i * rows as i32).collect();
        let texts: Vec<String> = ids
            .iter()
            .map(|v| format!("{v:07}{}", "z".repeat(width - 7)))
            .collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(
                    texts.iter().map(|t| Some(t.as_str())).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();

        tokio::time::timeout(Duration::from_secs(10), writer.put(vec![batch]))
            .await
            .expect("a shard with a freezable memtable must not park forever")
            .unwrap_or_else(|e| {
                panic!("put {i} was refused; index growth outran the freeze check: {e}")
            });
    }

    assert!(
        writer.memory().index_bytes() > writer.memory().row_bytes(),
        "the fixture must be index-dominated, or it is not testing this path"
    );

    writer.close().await.unwrap();
}

/// `max_memtable_size` is the headroom the reservation check reserves for
/// rows. At zero it reserves none, so a fresh memtable's index reservation
/// could equal the ceiling exactly — admissible by the check, yet over budget
/// before its first row, with an empty memtable offering nothing to freeze. The
/// writer refused its first write and never recovered.
#[tokio::test]
async fn test_open_rejects_a_zero_row_headroom() {
    let sizing = ShardWriterConfig {
        max_memtable_rows: 2_000,
        ..Default::default()
    };
    let reserved = reserved_index_bytes(&sizing, &hnsw_specs());
    let (store, base_path, base_uri, _t) = create_local_store().await;
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        max_memtable_rows: sizing.max_memtable_rows,
        max_memtable_size: 0,
        // Exactly the fresh reservation: the boundary the old `>` admitted.
        max_unflushed_memtable_bytes: reserved,
        ..Default::default()
    };

    let err = ShardWriter::open(
        store,
        base_path,
        base_uri,
        config,
        hnsw_schema(8),
        hnsw_specs(),
    )
    .await
    .err()
    .expect("a zero row headroom must be rejected at open");

    assert!(
        err.to_string()
            .contains("max_memtable_size must be greater than zero"),
        "the error must name the knob, got: {err}"
    );
}

/// An HNSW graph is charged from `max_memtable_rows` before the first
/// insert, while only row bytes freeze a memtable. Sized past the ceiling it
/// would put the shard over budget at zero rows with nothing to freeze and so
/// nothing to flush — every put stalling, then failing as `Backpressure`,
/// which means "retry later" and never comes true. That is a config error,
/// so it has to land at `open`, not on put #1.
#[tokio::test]
async fn test_open_rejects_indexes_that_cannot_fit_under_the_ceiling() {
    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        max_memtable_rows: 100_000,
        max_memtable_size: 1024 * 1024,
        max_unflushed_memtable_bytes: 1024 * 1024,
        ..Default::default()
    };
    let reserved = reserved_index_bytes(&config, &hnsw_specs());
    assert!(
        reserved > config.max_unflushed_memtable_bytes,
        "the fixture must actually over-subscribe the ceiling, got {reserved}"
    );

    let err = ShardWriter::open(
        store,
        base_path,
        base_uri,
        config,
        hnsw_schema(32),
        hnsw_specs(),
    )
    .await
    .err()
    .expect("an over-subscribed ceiling must be rejected at open");

    let message = err.to_string();
    for fragment in [
        "in-memory indexes reserve",
        "max_unflushed_memtable_bytes",
        "stalling every write",
    ] {
        assert!(
            message.contains(fragment),
            "the error must name {fragment}, got: {message}"
        );
    }
    assert!(
        !err.is_backpressure(),
        "a config error must not masquerade as the retryable busy signal"
    );
}

/// A held-back freeze pins the active memtable at its cap and refuses the
/// put that could only have landed after the freeze.
#[tokio::test]
async fn test_a_blocked_freeze_pins_the_memtable_and_refuses_the_put() {
    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let schema = create_pk_test_schema();
    let blocker = Arc::new(FreezeBlocker {
        blocked: std::sync::atomic::AtomicBool::new(true),
    });
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        durable_write: false,
        // One put of this size carries row bytes past the window, so the
        // put after it is one that can only land after a freeze.
        max_memtable_size: 1024,
        backpressure: Some(blocker.clone()),
        ..Default::default()
    };
    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    let generation = writer.memtable_stats().await.unwrap().generation;
    writer
        .put(vec![create_test_batch(&schema, 0, 200)])
        .await
        .expect("the first put fits and needs no freeze");

    let rows_before = writer.memtable_stats().await.unwrap().row_count;
    let err = writer
        .put(vec![create_test_batch(&schema, 200, 200)])
        .await
        .expect_err("a put that can only land after a blocked freeze must be refused");
    assert!(
        err.is_backpressure(),
        "the refusal must be the retryable signal, not a hard failure: {err}"
    );

    let stats = writer.memtable_stats().await.unwrap();
    assert_eq!(
        stats.generation, generation,
        "a blocked freeze must not rotate the memtable"
    );
    assert_eq!(
        stats.row_count, rows_before,
        "a refused put must not have landed any rows"
    );

    // Room again: the block is a stall, not a wedge.
    blocker
        .blocked
        .store(false, std::sync::atomic::Ordering::Relaxed);
    writer
        .put(vec![create_test_batch(&schema, 200, 200)])
        .await
        .expect("the put admitted once the freeze is allowed");
    // Not `+ 1`: at this memtable size the put freezes on the way in and
    // rotates again on the way out.
    assert!(
        writer.memtable_stats().await.unwrap().generation > generation,
        "the freeze the block was holding back must happen once it lifts"
    );

    writer.close().await.unwrap();
}

/// The other side of the gate: a ceiling with room for the reservation *and*
/// a full memtable of rows on top opens, and keeps taking writes across a
/// freeze — the frozen generation gives the valve a flush to park on, so the
/// wait ends instead of refusing.
#[tokio::test]
async fn test_writer_with_indexes_under_the_ceiling_keeps_accepting_writes() {
    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let dim = 8;
    let sizing = ShardWriterConfig {
        max_memtable_rows: 2_000,
        ..Default::default()
    };
    let reserved = reserved_index_bytes(&sizing, &hnsw_specs());
    let max_memtable_size = 4 * 1024;

    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        max_memtable_rows: sizing.max_memtable_rows,
        max_memtable_size,
        // Two generations' worth, so a freeze does not immediately re-park the
        // writer on a ceiling it cannot clear.
        max_unflushed_memtable_bytes: 2 * (reserved + max_memtable_size),
        ..Default::default()
    };

    let schema = hnsw_schema(dim);
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        config,
        schema.clone(),
        hnsw_specs(),
    )
    .await
    .expect("a reservation that leaves room under the ceiling must open");

    // Enough rows to carry row bytes past `max_memtable_size` several times
    // over, so the run spans freezes rather than sitting in one memtable.
    for round in 0..8i32 {
        let rows = 64;
        let ids: Vec<i32> = (0..rows).map(|i| round * rows + i).collect();
        let values: Vec<f32> = (0..rows * dim).map(|i| i as f32).collect();
        let vectors = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim,
            Arc::new(Float32Array::from(values)),
            None,
        )
        .unwrap();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(ids)), Arc::new(vectors)],
        )
        .unwrap();

        tokio::time::timeout(Duration::from_secs(10), writer.put(vec![batch]))
            .await
            .expect("a drainable shard must not park the writer indefinitely")
            .unwrap_or_else(|e| panic!("put in round {round} was refused: {e}"));
    }

    // Without a freeze this would only prove one memtable fits, which is not
    // the case that stalls.
    assert!(
        writer.memtable_stats().await.unwrap().generation > 0,
        "the run must cross a freeze for the drain path to have been exercised"
    );

    writer.close().await.unwrap();
}
