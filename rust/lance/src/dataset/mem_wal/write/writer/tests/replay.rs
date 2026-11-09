// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Open-time replay, generation recovery, and writer poisoning.

use super::super::*;
use super::support::*;
use crate::Dataset;
use crate::dataset::mem_wal::observer::WalObserver;
use crate::dataset::mem_wal::test_util::failing_memory_store;
use crate::dataset::mem_wal::write::test_support::*;
use arrow_array::{FixedSizeListArray, Float32Array, Int32Array};
use lance_core::FenceReason;
use rstest::rstest;

#[tokio::test]
async fn test_writer_poisons_on_persistence_failure_and_recovers_on_reopen() {
    let (store, base_path, controls) = failing_memory_store().await;
    let base_uri = "memory:///";
    let shard_id = Uuid::new_v4();
    let schema = schema_with_pk();
    controls.fail_wal_puts(usize::MAX);

    let config = ShardWriterConfig {
        max_wal_persist_retries: 1,
        wal_persist_retry_base_delay: Duration::from_millis(1),
        ..memtable_config_with_pk(shard_id)
    };

    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri,
        config.clone(),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    // The durable put waits on the flush, which poisons after its retries.
    let err = writer
        .put(vec![create_test_batch(&schema, 0, 1)])
        .await
        .unwrap_err();
    assert_eq!(err.fence_reason(), Some(FenceReason::PersistenceFailure));

    // A poisoned writer rejects further writes fast, same typed reason.
    let err = writer
        .put(vec![create_test_batch(&schema, 1, 1)])
        .await
        .unwrap_err();
    assert_eq!(err.fence_reason(), Some(FenceReason::PersistenceFailure));

    // ...and rejects *reads* too. Batch 0 was committed to the BatchStore
    // before its WAL PUT failed, so a poisoned writer that still served
    // reads would hand out a row that is not durable and that replay will
    // not reproduce — a divergent snapshot. Mirrors SlateDB's
    // `check_closed()` at the top of every read.
    for reason in [
        writer.scan().await.err().and_then(|e| e.fence_reason()),
        writer
            .active_memtable_ref()
            .await
            .err()
            .and_then(|e| e.fence_reason()),
        writer
            .in_memory_memtable_refs()
            .await
            .err()
            .and_then(|e| e.fence_reason()),
    ] {
        assert_eq!(reason, Some(FenceReason::PersistenceFailure));
    }

    // Stats stay readable: this is what an operator (and the eviction path)
    // inspects to decide what to do about the poisoned shard.
    writer.memtable_stats().await.unwrap();
    drop(writer);

    // Storage heals: reopening replays the WAL and accepts writes again.
    controls.recover();
    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();
    writer
        .put(vec![create_test_batch(&schema, 2, 1)])
        .await
        .unwrap();
}

/// A doomed open must fail on local validation *before* it claims the epoch,
/// so it cannot fence the healthy writer already serving the shard. The
/// index-config check used to run *after* `claim_epoch` (and, for a successor,
/// after `write_fence_sentinel`), so a rejected open still bumped the stored
/// epoch and fenced the incumbent.
#[tokio::test]
async fn test_doomed_open_does_not_fence_incumbent() {
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
        .put(vec![create_test_batch(&schema, 0, 1)])
        .await
        .unwrap();

    // An index spec that disagrees with the schema (FTS on the Int32 `id`
    // column) is rejected on local validation. On the old path this rejection
    // landed only after the epoch had already been claimed.
    let bad_fts = MemIndexSpec::fts("bad_fts", 0, "id");
    let err = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        memtable_config_with_pk(shard_id),
        schema.clone(),
        vec![bad_fts],
    )
    .await
    .map(|_| ())
    .expect_err("an FTS index on a non-Utf8 column must be rejected");
    assert!(
        err.to_string().contains("bad_fts") && err.to_string().contains("Utf8"),
        "unexpected error: {err}"
    );

    // The incumbent is untouched: not fenced, still accepting writes.
    writer_a.check_fenced().await.unwrap();
    writer_a
        .put(vec![create_test_batch(&schema, 1, 1)])
        .await
        .unwrap();
    writer_a.close().await.unwrap();
}

/// A failed dispatch during `freeze_memtable` must not drop the outgoing
/// table's rows from the read view. The active memtable is replaced before
/// the WAL-flush and index-apply sends; a send that failed (background tasks
/// gone) used to return before the outgoing table was retained in
/// `frozen_memtables`, so its accepted rows silently vanished — a scan
/// returned 0 rows with no error. The writer must instead retain the table
/// and poison, so reads fail fast rather than serve a divergent snapshot.
#[tokio::test]
async fn test_freeze_dispatch_failure_retains_rows_and_poisons() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();

    // Non-durable + no ticker: the put is read-your-writes (waits for its
    // index apply) but nothing is WAL-flushed, so the freeze below still owes
    // a WAL append.
    let config = ShardWriterConfig {
        durable_write: false,
        max_wal_flush_interval: None,
        ..memtable_config_with_pk(shard_id)
    };
    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    writer
        .put(vec![create_test_batch(&schema, 0, 10)])
        .await
        .unwrap();
    assert_eq!(writer.memtable_stats().await.unwrap().row_count, 10);

    // Tear the background tasks down without using `abort`, which poisons
    // the writer before shutdown. This test specifically exercises the
    // freeze dispatch failure as the first terminal error.
    writer.task_executor.shutdown_all().await.unwrap();

    let err = writer
        .force_freeze_active()
        .await
        .expect_err("force_freeze_active must surface the failed dispatch");
    assert!(
        err.to_string().contains("channel closed"),
        "unexpected error: {err}"
    );

    // The failure poisoned the writer: reads fail fast instead of returning a
    // silent zero-row snapshot of a shard whose rows were dropped.
    assert!(
        writer.scan().await.is_err(),
        "a poisoned writer must reject reads, not serve a divergent snapshot"
    );
    assert!(writer.in_memory_memtable_refs().await.is_err());
}

/// A WAL holding more batches than one memtable's capacity must reopen.
///
/// WAL-only entries name no generation, so only this size rule splits them.
#[tokio::test]
async fn test_replay_rotates_when_wal_exceeds_one_memtable() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();

    const N: i32 = 8;

    // Writer A is WAL-only, so its eight entries name no generation.
    let writer_a_config = ShardWriterConfig {
        enable_memtable: false,
        ..memtable_config_with_pk(shard_id)
    };
    // Writer B has a *two-batch* capacity, so replaying that eight-entry WAL is
    // exactly what must rotate. Keeping the configs distinct isolates replay
    // rotation from the live rotation writer A would otherwise do concurrently.
    let config = ShardWriterConfig {
        max_memtable_batches: 2,
        ..memtable_config_with_pk(shard_id)
    };

    {
        let writer_a = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            writer_a_config,
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        for id in 0..N {
            writer_a
                .put(vec![create_test_batch(&schema, id, 1)])
                .await
                .unwrap();
        }
        // Drop without close: only the WAL survives, and it holds more batches
        // than writer B's memtable can.
    }

    // Total rows across the active memtable plus every SSTable.
    // Distinct ids, so no cross-generation dedup — a plain sum is exact.
    async fn total_rows(writer: &ShardWriter, base_uri: &str, shard_id: Uuid) -> usize {
        let mut rows = writer.memtable_stats().await.unwrap().row_count;
        let manifest = writer.manifest().await.unwrap().unwrap();
        for sstable in &manifest.sstables {
            let gen_uri = format!("{}/_mem_wal/{}/{}", base_uri, shard_id, sstable.path);
            let dataset = crate::Dataset::open(&gen_uri).await.unwrap();
            rows += dataset.count_rows(None).await.unwrap();
        }
        rows
    }

    // Reopen. This used to fail with a full-batch-store error.
    let writer_b = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        config.clone(),
        schema.clone(),
        vec![],
    )
    .await
    .expect("a WAL larger than one memtable must still reopen");

    // Rotation produced frozen memtables, and replay flushed each to a Lance
    // generation rather than holding it in memory or leaving it in the WAL.
    let manifest = writer_b.manifest().await.unwrap().unwrap();
    assert!(
        !manifest.sstables.is_empty(),
        "replay must have frozen and flushed at least one full memtable"
    );

    // Every row survived, split between the SSTables and the
    // active (partial) memtable.
    assert_eq!(
        total_rows(&writer_b, &base_uri, shard_id).await as i32,
        N,
        "every replayed row must be durable, across generations and the active memtable"
    );
    writer_b.close().await.unwrap();

    // Because the frozen memtables were flushed, the manifest's replay cursor
    // advanced past their WAL entries — so a second reopen replays only the
    // tail and still accounts for every row. The WAL truncates across reopens
    // rather than growing without bound.
    let writer_c = ShardWriter::open(store, base_path, base_uri.clone(), config, schema, vec![])
        .await
        .unwrap();
    assert_eq!(total_rows(&writer_c, &base_uri, shard_id).await as i32, N);
    writer_c.close().await.unwrap();
}

/// The same rotation, driven by `max_memtable_rows` instead of the batch cap.
///
/// Replay sizes each memtable's HNSW graph to that cap, so a WAL holding more
/// rows than one memtable's capacity has to rotate for `open()` to succeed.
#[tokio::test]
async fn test_replay_rotates_when_wal_exceeds_the_row_cap() {
    use lance_arrow::FixedSizeListArrayExt;

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let dim = 8;
    let schema = hnsw_schema(dim);
    let shard_id = Uuid::new_v4();
    let cap = 8;

    let vector_batch = |start: i32, rows: usize| {
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(
                (0..rows * dim as usize)
                    .map(|v| v as f32 * 0.01)
                    .collect::<Vec<_>>(),
            ),
            dim,
        )
        .unwrap();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(
                    (start..start + rows as i32).collect::<Vec<_>>(),
                )),
                Arc::new(vectors),
            ],
        )
        .unwrap()
    };

    // Writer A is WAL-only, so its entries name no generation.
    let writer_a_config = ShardWriterConfig {
        enable_memtable: false,
        ..memtable_config_with_pk(shard_id)
    };
    // Writer B caps a memtable at 8 rows — and sizes its HNSW graph to match.
    let config = ShardWriterConfig {
        max_memtable_rows: cap,
        ..memtable_config_with_pk(shard_id)
    };

    {
        let writer_a = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            writer_a_config,
            schema.clone(),
            // WAL-only mode builds no index; writer B carries the HNSW specs.
            vec![],
        )
        .await
        .unwrap();
        for round in 0..8i32 {
            writer_a
                .put(vec![vector_batch(round * 4, 4)])
                .await
                .unwrap();
        }
    }

    // Replay has 32 rows to place into memtables capped at 8.
    let writer_b = ShardWriter::open(store, base_path, base_uri, config, schema, hnsw_specs())
        .await
        .expect("a WAL holding more rows than the cap must still reopen");

    let manifest = writer_b.manifest().await.unwrap().unwrap();
    assert!(
        !manifest.sstables.is_empty(),
        "replay must have frozen and flushed the memtables it filled"
    );
    let stats = writer_b.memtable_stats().await.unwrap();
    assert!(
        stats.row_count <= cap,
        "replay left {} rows in a memtable capped at {cap}",
        stats.row_count
    );

    writer_b.close().await.unwrap();
}

/// Replay-on-open recovers durable WAL entries that were never flushed
/// to a Lance generation. Setup: writer A durably writes batches, drops
/// without close (so MemTable freeze never runs); writer B reopens and
/// must see A's rows in its MemTable scan.
#[tokio::test]
async fn test_memtable_replay_recovers_unflushed_writes() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();

    // Writer A: write two durable batches, then drop without close.
    // The WAL files persist; the in-memory MemTable does not.
    {
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
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();
        writer_a
            .put(vec![create_test_batch(&schema, 100, 3)])
            .await
            .unwrap();
        // intentionally drop without close()
    }

    // Writer B reopens. Replay must rehydrate A's two batches into the
    // active MemTable.
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

    let stats = writer_b.memtable_stats().await.unwrap();
    assert_eq!(
        stats.row_count, 8,
        "expected replay to insert 5 + 3 = 8 rows, got {}",
        stats.row_count
    );
    assert_eq!(
        stats.batch_count, 2,
        "expected replay to insert 2 batches, got {}",
        stats.batch_count
    );

    writer_b.close().await.unwrap();
}

/// A failing flush is retried before any later generation publishes, so no
/// accepted row is lost to a crash once it lands.
#[tokio::test]
async fn a_failing_flush_is_retried_before_any_later_generation() {
    let (store, base_path, controls) = failing_memory_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();
    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        "memory:///",
        memtable_config_with_pk(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    writer
        .put(vec![create_test_batch(&schema, 0, 5)])
        .await
        .unwrap();
    controls.fail_writes_under("_gen_1/");
    let first = writer.force_freeze_active().await.unwrap();
    writer
        .put(vec![create_test_batch(&schema, 100, 3)])
        .await
        .unwrap();
    let second = writer.force_freeze_active().await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let manifest = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2)
        .latest()
        .await
        .unwrap()
        .unwrap();
    assert!(
        manifest.sstables.is_empty(),
        "nothing publishes while generation 1 fails"
    );

    controls.clear_write_failures();
    first.wait().await.unwrap();
    second.wait().await.unwrap();
    // Crash: drop without close.
    drop(writer);

    let manifest = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2)
        .latest()
        .await
        .unwrap()
        .unwrap();
    let generations: Vec<u64> = manifest.sstables.iter().map(|s| s.generation).collect();
    assert_eq!(generations, vec![1, 2]);
    assert_eq!(
        rows_after_reopen(&store, &base_path, shard_id, &schema).await,
        5 + 3
    );
}

/// Each failed flush attempt is counted in the stats and reported to the
/// observer, so a stuck generation is visible before backpressure builds.
#[tokio::test]
async fn failed_flush_attempts_are_counted_and_observed() {
    #[derive(Debug, Default)]
    struct FailureObserver {
        failed_generations: std::sync::Mutex<Vec<u64>>,
    }

    impl WalObserver for FailureObserver {
        fn on_memtable_flush_failure(&self, generation: u64, _error: &Error) {
            self.failed_generations.lock().unwrap().push(generation);
        }
    }

    let (store, base_path, controls) = failing_memory_store().await;
    let schema = schema_with_pk();
    let observer = Arc::new(FailureObserver::default());
    let sink: Arc<dyn WalObserver> = observer.clone();
    let config = ShardWriterConfig {
        observer: Some(sink),
        ..memtable_config_with_pk(Uuid::new_v4())
    };
    let writer = ShardWriter::open(
        store,
        base_path,
        "memory:///",
        config,
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    writer
        .put(vec![create_test_batch(&schema, 0, 5)])
        .await
        .unwrap();
    controls.fail_writes_under("_gen_1/");
    let _first = writer.force_freeze_active().await.unwrap();

    tokio::time::timeout(Duration::from_secs(5), async {
        while writer.stats().memtable_flush_failures == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the failed flush attempt should be counted");
    assert_eq!(writer.stats().memtable_flush_count, 0);
    let failed_generations = observer.failed_generations.lock().unwrap().clone();
    assert!(!failed_generations.is_empty());
    assert!(failed_generations.iter().all(|&generation| generation == 1));

    assert!(writer.close().await.is_err());
}

/// Close stops retrying a failing flush without publishing a later
/// generation, so the next open replays every row.
#[tokio::test]
async fn close_gives_up_on_a_failing_flush_and_replay_recovers_it() {
    let (store, base_path, controls) = failing_memory_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();
    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        "memory:///",
        memtable_config_with_pk(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    writer
        .put(vec![create_test_batch(&schema, 0, 5)])
        .await
        .unwrap();
    controls.fail_writes_under("_gen_1/");
    let _first = writer.force_freeze_active().await.unwrap();
    writer
        .put(vec![create_test_batch(&schema, 100, 3)])
        .await
        .unwrap();
    assert!(writer.close().await.is_err());

    controls.clear_write_failures();
    let manifest = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2)
        .latest()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(manifest.current_generation, 1);
    assert!(manifest.sstables.is_empty());
    assert_eq!(
        rows_after_reopen(&store, &base_path, shard_id, &schema).await,
        5 + 3
    );
}

/// A close that fails while draining the active memtable's indexes still
/// stops retrying a failing flush.
#[tokio::test]
async fn close_failing_before_the_flush_stops_flush_retries() {
    let (store, base_path, controls) = failing_memory_store().await;
    let schema = schema_with_pk();
    let writer = ShardWriter::open(
        store,
        base_path,
        "memory:///",
        memtable_config_with_pk(Uuid::new_v4()),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    writer
        .put(vec![create_test_batch(&schema, 0, 5)])
        .await
        .unwrap();
    controls.fail_writes_under("_gen_1/");
    let _first = writer.force_freeze_active().await.unwrap();
    writer
        .put_no_wait(vec![create_test_batch(&schema, 100, 3)])
        .await
        .unwrap();
    let stop_flush_retries = match &writer.mode {
        WriterMode::MemTable { writer_state, .. } => writer_state.stop_flush_retries.clone(),
        WriterMode::WalOnly { .. } => unreachable!("opened in memtable mode"),
    };
    writer
        .wal_flusher
        .poison(&Error::io("injected index apply failure"));

    let error = writer.close().await.unwrap_err();
    assert!(
        error.to_string().contains("injected index apply failure"),
        "{error}"
    );
    assert!(stop_flush_retries.is_cancelled());
}

/// Replayed batches are already WAL-durable, so the first flush after a
/// reopen must not re-append them to the WAL or re-insert them into the
/// indexes. Before replay stamped the durability cursor it stayed at
/// "nothing flushed", so the next flush re-covered `[0, end)`: it appended
/// the already-durable rows a second time *and* re-indexed them. None of
/// the in-memory indexes is idempotent, so an indexed PK lookup returned
/// the row twice while a full scan still looked healthy.
#[tokio::test]
async fn test_replay_does_not_reappend_or_reindex() {
    use crate::dataset::mem_wal::wal::WalTailer;

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();

    // Writer A: two durable batches (5 + 3 = 8 rows), dropped without close.
    {
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
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();
        writer_a
            .put(vec![create_test_batch(&schema, 100, 3)])
            .await
            .unwrap();
    }

    // Writer B reopens and replays A's two batches.
    let writer_b = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri,
        memtable_config_with_pk(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    let stats = writer_b.memtable_stats().await.unwrap();
    assert_eq!(stats.batch_count, 2);
    assert_eq!(
        stats.durable_batch_count, 2,
        "replayed batches came from the WAL, so the durability cursor must already cover them"
    );

    // One more durable put. Its flush must cover only the new batch.
    writer_b
        .put(vec![create_test_batch(&schema, 200, 2)])
        .await
        .unwrap();

    let tailer = WalTailer::new(store, base_path, "", shard_id);
    let first = tailer.first_position().await.unwrap();
    let next = tailer.next_position().await.unwrap();
    let mut wal_rows = 0;
    for position in first..next {
        if let Some(entry) = tailer.read_entry(position).await.unwrap() {
            wal_rows += entry.batches.iter().map(|b| b.num_rows()).sum::<usize>();
        }
    }
    assert_eq!(
        wal_rows, 10,
        "WAL must hold 8 replayed + 2 new rows; a re-covering flush re-appends the replayed 8"
    );

    // The indexed arm must not see a replayed row twice.
    let mut scanner = writer_b.scan().await.unwrap();
    scanner.filter("id = 0").unwrap();
    let hit = scanner.try_into_batch().await.unwrap();
    assert_eq!(
        hit.num_rows(),
        1,
        "indexed PK lookup returned the replayed row more than once"
    );

    let all = writer_b
        .scan()
        .await
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(all.num_rows(), 10);

    writer_b.close().await.unwrap();
}

/// A durable writer with no flush interval cannot guarantee progress in
/// either mode: a small put may not cross the size threshold, and without an
/// interval neither the write path nor ticker drives its WAL append.
#[rstest]
#[case::memtable(true)]
#[case::wal_only(false)]
#[tokio::test]
async fn test_open_rejects_durable_write_without_a_ticker(#[case] enable_memtable: bool) {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();

    let config = ShardWriterConfig {
        durable_write: true,
        enable_memtable,
        max_wal_flush_interval: None,
        ..memtable_config_with_pk(shard_id)
    };

    let Err(err) = ShardWriter::open(store, base_path, base_uri, config, schema, vec![]).await
    else {
        panic!("durable_write with no ticker must be rejected");
    };
    assert!(
        err.to_string().contains("max_wal_flush_interval"),
        "the error must name the knob, got: {err}"
    );
}

/// WAL entries must be appended in global batch-position order across a
/// memtable rotation, because append order *is* primary-key recency order.
///
/// `WalAppender::append` assigns each entry's position from its own counter,
/// in call order. Replay walks those positions ascending and assigns row
/// positions in that order. Primary-key recency is "newest visible row
/// position wins". So an out-of-order append silently inverts dedup after a
/// crash — the stale row wins — and a full scan cannot see it.
///
/// The hazard is the background ticker. If it resolved to "whatever memtable
/// is active" rather than to the oldest store still owing an append, a tick
/// enqueued before a freeze but handled after it would append the *incoming*
/// memtable's batches ahead of the outgoing memtable's tail. So the target is
/// resolved from the durability cursor, not from wall-clock timing.
#[tokio::test]
async fn test_wal_append_order_preserves_pk_recency_across_rotation() {
    use crate::dataset::mem_wal::wal::WalTailer;

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();

    // Two batches per memtable, so the second put fills it and rotates.
    let config = ShardWriterConfig {
        max_memtable_batches: 2,
        ..memtable_config_with_pk(shard_id)
    };

    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri,
        config,
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    // Memtable 1: ids 7 then 20 (the second fills it and triggers the freeze).
    writer
        .put(vec![create_test_batch(&schema, 7, 1)])
        .await
        .unwrap();
    writer
        .put(vec![create_test_batch(&schema, 20, 1)])
        .await
        .unwrap();
    // Memtable 2 overwrites id=7 with a newer row. Its append must land in the
    // WAL *after* memtable 1's, or replay would resolve id=7 to the stale copy.
    writer
        .put(vec![create_test_batch(&schema, 30, 1)])
        .await
        .unwrap();
    writer.close().await.unwrap();

    // Walk the WAL in entry order and collect the ids as replay would see them.
    let tailer = WalTailer::new(store, base_path, "", shard_id);
    let first = tailer.first_position().await.unwrap();
    let next = tailer.next_position().await.unwrap();
    let mut ids: Vec<i32> = Vec::new();
    for position in first..next {
        let Some(entry) = tailer.read_entry(position).await.unwrap() else {
            continue;
        };
        for batch in &entry.batches {
            let column = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            ids.extend((0..column.len()).map(|i| column.value(i)));
        }
    }

    assert_eq!(
        ids,
        vec![7, 20, 30],
        "WAL entries must follow global batch-position order; memtable 1's rows must \
         precede memtable 2's, or replay inverts primary-key recency"
    );
}

/// An index spec that disagrees with the schema fails `open()` outright.
/// It must not be allowed to accept writes: the insert would fail
/// deterministically on every batch, including batches replayed from the
/// WAL, so once a row was durable the shard could never reopen. Before this
/// check, an FTS index on a non-Utf8 column silently indexed nothing and the
/// shard reported healthy.
#[tokio::test]
async fn test_open_rejects_index_config_that_disagrees_with_schema() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();

    // `id` is Int32, not a string column.
    let bad_fts = MemIndexSpec::fts("bad_fts", 0, "id");

    let Err(err) = ShardWriter::open(
        store,
        base_path,
        base_uri,
        memtable_config_with_pk(shard_id),
        schema,
        vec![bad_fts],
    )
    .await
    else {
        panic!("open must reject an FTS index on a non-Utf8 column");
    };

    let message = err.to_string();
    assert!(
        message.contains("bad_fts") && message.contains("Utf8"),
        "error must name the index and the constraint, got: {message}"
    );
}

/// Replay is a no-op on a fresh shard: the MemTable starts empty.
#[tokio::test]
async fn test_memtable_replay_no_op_on_fresh_shard() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();

    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        memtable_config_with_pk(shard_id),
        schema,
        vec![],
    )
    .await
    .unwrap();
    let stats = writer.memtable_stats().await.unwrap();
    assert_eq!(stats.row_count, 0);
    assert_eq!(stats.batch_count, 0);
    writer.close().await.unwrap();
}

/// Regression for the OSS-WAL compactor-drain bug: after a flush
/// records its generation in the manifest and an external compactor
/// later drains `sstables` back to empty (the legitimate
/// outcome after compacting the SSTable into the base table), reopening
/// the writer must not re-replay the already-flushed WAL entry into
/// the active memtable.
///
/// Under the pre-fix logic, replay disambiguated "fresh shard" from
/// "flushed-then-compacted" with `sstables.is_empty()`,
/// which collapsed both cases into start-at-0. With 1-based WAL
/// positions and a default cursor of 0 meaning "no flush stamped",
/// the flush-then-drain sequence leaves `replay_after_wal_entry_position`
/// pinned at the flushed position, so replay correctly starts past it.
#[tokio::test]
async fn test_memtable_replay_skips_entries_after_external_compaction() {
    use crate::dataset::mem_wal::ShardManifestStore;

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();

    // Writer A: write 5 rows, close (forces a flush of the active
    // memtable). The manifest now records an SSTable and
    // pins `replay_after_wal_entry_position` to the covered WAL entry.
    {
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
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();
        writer_a.close().await.unwrap();
    }

    // Simulate an external compactor compacting the SSTable.
    // into the base table: drain `sstables` to empty via a
    // direct manifest commit. The cursor stays where the flush put it.
    let manifest_store = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2);
    let pre = manifest_store.latest().await.unwrap().unwrap();
    assert!(
        !pre.sstables.is_empty(),
        "writer A's close() should have stamped an SSTable"
    );
    let cursor_at_flush = pre.replay_after_wal_entry_position;
    let old_generation_uri = format!(
        "{}/_mem_wal/{}/{}",
        base_uri.trim_end_matches('/'),
        shard_id,
        pre.sstables[0].path
    );
    assert!(Dataset::open(&old_generation_uri).await.is_ok());
    assert!(
        cursor_at_flush >= 1,
        "expected cursor to land on a 1-based WAL position after flush, got {cursor_at_flush}"
    );
    // Bump the epoch (claim_epoch) so we can commit_update without
    // being fenced; this also mirrors how a compactor process would
    // hold its own writer claim.
    let (compactor_epoch, _) = manifest_store.claim_epoch(pre.shard_spec_id).await.unwrap();
    manifest_store
        .commit_update(compactor_epoch, |current| ShardManifest {
            version: current.next_version(),
            sstables: vec![],
            ..current.clone()
        })
        .await
        .unwrap();
    let post = manifest_store.latest().await.unwrap().unwrap();
    assert!(
        post.sstables.is_empty(),
        "compactor drain should have left sstables empty"
    );
    assert_eq!(
        post.replay_after_wal_entry_position, cursor_at_flush,
        "compactor must not touch the replay cursor"
    );

    // Writer B reopens. Pre-fix: replay saw sstables empty,
    // restarted at WAL position 0, and re-inserted writer A's rows.
    // Post-fix: replay starts at cursor + 1, finds no entry, and the
    // memtable stays empty.
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
    assert!(
        Dataset::open(&old_generation_uri).await.is_ok(),
        "writer reopen removed an SSTable retained by the older shard manifest"
    );
    let stats = writer_b.memtable_stats().await.unwrap();
    assert_eq!(
        stats.row_count, 0,
        "memtable must not re-replay compacted WAL entries; got {} rows",
        stats.row_count
    );
    assert_eq!(stats.batch_count, 0);
    writer_b.close().await.unwrap();
}

/// A memtable frozen after its batches were appended, but before the
/// writer state recorded that append, flushes with a covered position
/// short of the append. Reopen replays that entry and skips it by its
/// generation, restoring only the generation after it.
#[tokio::test]
async fn test_reopen_after_flush_with_stale_covered_position() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();
    // Appends happen only when the test drives them.
    let config = ShardWriterConfig {
        durable_write: false,
        max_wal_flush_interval: None,
        max_memtable_rows: COVERED_TEST_ROW_CAP,
        ..memtable_config_with_pk(shard_id)
    };

    let writer_a = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        config.clone(),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    writer_a
        .put(vec![create_test_batch(&schema, 0, 4)])
        .await
        .unwrap();
    append_active_without_state_update(&writer_a).await;
    // The freeze finds nothing left to append, so it triggers no append
    // whose position it could take.
    writer_a.force_freeze_active().await.unwrap();
    writer_a.wait_for_flush_drain().await.unwrap();

    let manifest = writer_a.manifest().await.unwrap().unwrap();
    assert_eq!(
        manifest.replay_after_wal_entry_position, 0,
        "the covered position must miss WAL entry 1 for reopen to replay it"
    );

    // Fill the next generation to just under the cap, then crash.
    writer_a
        .put(vec![create_test_batch(&schema, 100, 14)])
        .await
        .unwrap();
    append_active_without_state_update(&writer_a).await;
    writer_a.abort().await.unwrap();
    drop(writer_a);

    let writer_b = ShardWriter::open(
        store,
        base_path,
        base_uri.clone(),
        config,
        schema.clone(),
        vec![],
    )
    .await
    .expect("a shard reopened with the config it was written under must open");

    assert_eq!(
        writer_b.memtable_stats().await.unwrap().row_count,
        14,
        "replay must restore only the unflushed generation"
    );
    assert_eq!(
        read_sstable_ids_via_lsm(&writer_b, schema, &base_uri, shard_id, None).await,
        (0..4).collect::<Vec<_>>()
    );
    writer_b.close().await.unwrap();
}

/// Past the generation replay starts from, WAL generations only move
/// forward; an entry whose generation goes backwards fails the open.
#[tokio::test]
async fn test_replay_rejects_backwards_generation() {
    use crate::dataset::mem_wal::ShardManifestStore;

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();
    let config = memtable_config_with_pk(shard_id);

    let writer_a = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        config.clone(),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    let manifest = writer_a.manifest().await.unwrap().unwrap();
    let base_generation = manifest.current_generation;
    let epoch = writer_a.epoch;
    writer_a.abort().await.unwrap();
    drop(writer_a);

    let manifest_store = Arc::new(ShardManifestStore::new(
        store.clone(),
        &base_path,
        "",
        shard_id,
        2,
    ));
    let appender = WalAppender::with_claimed_epoch(
        store.clone(),
        base_path.clone(),
        "",
        shard_id,
        manifest_store,
        epoch,
        0,
        WalRetryConfig::default(),
    );
    for (start_id, generation) in [(0, base_generation + 1), (10, base_generation)] {
        appender
            .append_for_target(
                vec![create_test_batch(&schema, start_id, 2)],
                None,
                Some(generation),
            )
            .await
            .unwrap();
    }

    let err = ShardWriter::open(store, base_path, base_uri, config, schema, vec![])
        .await
        .map(|_| ())
        .expect_err("a WAL generation going backwards must fail the open");
    assert!(
        err.to_string().contains("below the generation"),
        "unexpected error: {err}"
    );
}

/// Replay aborts the open with a clear fence error if it encounters a
/// WAL entry written with an epoch strictly greater than ours. Simulate
/// the race where another writer wrote an entry with a higher epoch
/// between our `claim_epoch` and our replay by injecting a high-epoch
/// entry directly via `WalAppender::with_claimed_epoch` (which
/// bypasses `claim_epoch` and so does not bump the manifest).
#[tokio::test]
async fn test_memtable_replay_fenced_aborts_open() {
    use crate::dataset::mem_wal::ShardManifestStore;

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();

    // Writer A: write one durable batch (claims epoch 1, writes entry at position 1).
    {
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
            .put(vec![create_test_batch(&schema, 0, 1)])
            .await
            .unwrap();
        // drop without close
    }

    // Inject a WAL entry written with epoch 100 — far above whatever
    // claim_epoch will hand the next opener. The manifest is not
    // updated since we use `with_claimed_epoch` directly.
    let manifest_store = Arc::new(ShardManifestStore::new(
        store.clone(),
        &base_path,
        "",
        shard_id,
        2,
    ));
    let high_epoch_appender = WalAppender::with_claimed_epoch(
        store.clone(),
        base_path.clone(),
        "",
        shard_id,
        manifest_store,
        100,
        // hint seed irrelevant; the real position counter is discovered
        // lazily on the first append.
        0,
        WalRetryConfig::default(),
    );
    high_epoch_appender
        .append(vec![create_test_batch(&schema, 999, 1)])
        .await
        .unwrap();

    // Writer B opens. claim_epoch returns 2 (manifest's writer_epoch
    // was 1 before this open). Replay reads the injected entry, sees
    // epoch 100 > 2, and aborts with a fence error.
    let result = ShardWriter::open(
        store,
        base_path,
        base_uri,
        memtable_config_with_pk(shard_id),
        schema,
        vec![],
    )
    .await;
    let Err(err) = result else {
        panic!("expected open to fail with fence error during replay");
    };
    // Assert the *typed* fence reason, not just the message: a regression
    // reverting this to `Error::io` would still carry a message containing
    // "fenced" and slip past a string check, but must not report a
    // `FenceReason`.
    assert_eq!(
        err.fence_reason(),
        Some(FenceReason::PeerClaimedEpoch),
        "replay must abort with a typed peer-fence error, got: {err}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("WAL replay aborted") && msg.contains("fenced"),
        "unexpected error: {msg}"
    );
}

/// Regression: `wal_stats().next_wal_entry_position` must reflect the
/// post-recovery cursor immediately on reopen, not 0 until the first
/// append discovers the tip. Pre-fix the appender's hint was seeded at
/// 0 and only updated after the first successful append, so external
/// monitors saw 0 between open and first put on a shard with prior
/// entries.
#[tokio::test]
async fn test_wal_stats_seeded_from_manifest_on_reopen() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let shard_id = Uuid::new_v4();

    // First writer creates a shard, writes one entry, closes.
    let writer1 = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        wal_only_config(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    writer1
        .put(vec![create_test_batch(&schema, 0, 1)])
        .await
        .unwrap();
    writer1.close().await.unwrap();

    // Reopen: stats must reflect the post-recovery cursor immediately,
    // before any put has happened on this writer.
    let writer2 = ShardWriter::open(
        store,
        base_path,
        base_uri,
        wal_only_config(shard_id),
        schema,
        vec![],
    )
    .await
    .unwrap();
    let next = writer2.wal_stats().next_wal_entry_position;
    assert!(
        next >= 1,
        "expected wal_stats to reflect post-recovery cursor (>= 1) on reopen, got {next}"
    );

    writer2.close().await.unwrap();
}
