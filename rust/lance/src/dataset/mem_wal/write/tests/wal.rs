// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Public WAL-only mode and WAL ordering behavior.

use std::time::Duration;

use uuid::Uuid;

use super::super::test_support::*;
use super::super::{ShardWriter, ShardWriterConfig};
use crate::dataset::mem_wal::index::MemIndexSpec;

#[tokio::test]
async fn test_wal_only_durable_round_trip() {
    use crate::dataset::mem_wal::wal::WalTailer;

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    let shard_id = Uuid::new_v4();
    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri,
        wal_only_config(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    // Two durable puts → two WAL entries. Each `put` awaits its own append
    // (driven by the background ticker) before returning, so the second
    // batch is only pushed after the first is durable — they never coalesce.
    let r1 = writer
        .put(vec![create_test_batch(&schema, 0, 4)])
        .await
        .unwrap();
    let r2 = writer
        .put(vec![create_test_batch(&schema, 100, 2)])
        .await
        .unwrap();
    assert_eq!(r1.batch_positions, 0..1);
    assert_eq!(r2.batch_positions, 1..2);

    writer.close().await.unwrap();

    // Read back via WalTailer. WAL positions are 1-based, so two
    // entries from a fresh shard land at 1 and 2.
    let tailer = WalTailer::new(store, base_path, "", shard_id);
    assert_eq!(tailer.first_position().await.unwrap(), 1);
    assert_eq!(tailer.next_position().await.unwrap(), 3);

    let e0 = tailer.read_entry(1).await.unwrap().unwrap();
    let e1 = tailer.read_entry(2).await.unwrap().unwrap();
    assert_eq!(e0.batches.len(), 1);
    assert_eq!(e0.batches[0].num_rows(), 4);
    assert_eq!(e1.batches.len(), 1);
    assert_eq!(e1.batches[0].num_rows(), 2);
    // Both entries from the same writer should have the same epoch.
    assert_eq!(e0.writer_epoch, e1.writer_epoch);
    assert!(e0.writer_epoch >= 1);
}

/// A durable WAL-only put is driven by the background ticker, not an inline
/// per-put flush: `put().await` must not return until the ticker's append
/// advances the durability watermark it waits on. With a long interval and a
/// buffer the write cannot cross, the ticker is the *only* thing that can
/// make the put durable — so if the WAL entry is present the instant `put`
/// returns (before any `close()`), the watermark-wait is doing its job.
#[tokio::test]
async fn test_wal_only_durable_put_waits_for_ticker_append() {
    use crate::dataset::mem_wal::wal::WalTailer;

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let shard_id = Uuid::new_v4();

    let mut config = wal_only_config(shard_id);
    config.max_wal_flush_interval = Some(Duration::from_millis(100));
    config.max_wal_buffer_size = 100 * 1024 * 1024; // never crossed

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

    writer
        .put(vec![create_test_batch(&schema, 0, 4)])
        .await
        .unwrap();

    // `put` returned, so the batch must already be durable — read the WAL
    // directly, without closing the writer.
    let tailer = WalTailer::new(store, base_path, "", shard_id);
    assert_eq!(tailer.next_position().await.unwrap(), 2);
    let entry = tailer.read_entry(1).await.unwrap().unwrap();
    assert_eq!(entry.batches.len(), 1);
    assert_eq!(entry.batches[0].num_rows(), 4);

    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_wal_only_rejects_index_specs() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let index_specs = vec![MemIndexSpec::btree("id_idx", 0, "id")];

    let err = ShardWriter::open(
        store,
        base_path,
        base_uri,
        wal_only_config(Uuid::new_v4()),
        schema,
        index_specs,
    )
    .await
    .err()
    .expect("expected invalid_input");
    assert!(
        err.to_string().contains("indexes require enable_memtable"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn test_wal_only_rejects_empty_batches() {
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

    // Empty list.
    let err = writer.put(vec![]).await.err().unwrap();
    assert!(err.to_string().contains("empty batch list"));

    // Single empty batch.
    let zero = arrow_array::RecordBatch::new_empty(schema);
    let err = writer.put(vec![zero]).await.err().unwrap();
    assert!(err.to_string().contains("Batch 0 is empty"));

    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_wal_only_memtable_accessors_error() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        wal_only_config(Uuid::new_v4()),
        schema,
        vec![],
    )
    .await
    .unwrap();

    let err = writer.memtable_stats().await.err().unwrap();
    assert!(err.to_string().contains("WAL-only mode"));
    let err = writer.scan().await.err().unwrap();
    assert!(err.to_string().contains("WAL-only mode"));
    let err = writer.active_memtable_ref().await.err().unwrap();
    assert!(err.to_string().contains("WAL-only mode"));
    let err = writer.in_memory_memtable_refs().await.err().unwrap();
    assert!(err.to_string().contains("WAL-only mode"));

    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_wal_only_async_batches_multiple_puts() {
    use crate::dataset::mem_wal::wal::WalTailer;

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    let mut config = wal_only_config(Uuid::new_v4());
    let shard_id = config.shard_id;
    // Non-durable: puts should accumulate in the pending queue until
    // close() drains them with a single WAL entry.
    config.durable_write = false;
    config.max_wal_flush_interval = None;
    config.max_wal_buffer_size = 100 * 1024 * 1024; // never crossed

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

    for i in 0..3 {
        writer
            .put(vec![create_test_batch(&schema, i * 10, 10)])
            .await
            .unwrap();
    }

    writer.close().await.unwrap();

    // All three puts should be in a single WAL entry at position 1
    // (WAL positions are 1-based).
    let tailer = WalTailer::new(store, base_path, "", shard_id);
    assert_eq!(tailer.next_position().await.unwrap(), 2);
    let entry = tailer.read_entry(1).await.unwrap().unwrap();
    assert_eq!(entry.batches.len(), 3);
    for (i, batch) in entry.batches.iter().enumerate() {
        assert_eq!(batch.num_rows(), 10, "batch {i}");
    }
}

#[tokio::test]
async fn test_wal_only_fencing() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let shard_id = Uuid::new_v4();

    let writer_a = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        wal_only_config(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    writer_a
        .put(vec![create_test_batch(&schema, 0, 1)])
        .await
        .unwrap();

    // Writer B claims a higher epoch and writes — fences A.
    let writer_b = ShardWriter::open(
        store,
        base_path,
        base_uri,
        wal_only_config(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    assert!(writer_b.epoch() > writer_a.epoch());
    writer_b
        .put(vec![create_test_batch(&schema, 1, 1)])
        .await
        .unwrap();

    // A's next durable put must fail with a fence error (the underlying
    // WalAppender::append surfaces it via atomic_put).
    let err = writer_a
        .put(vec![create_test_batch(&schema, 2, 1)])
        .await
        .expect_err("expected fence error");
    assert!(
        err.to_string().contains("Writer fenced"),
        "unexpected error: {err}"
    );

    writer_b.close().await.unwrap();
}

#[tokio::test]
async fn test_check_fenced_detects_successor_claim() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let shard_id = Uuid::new_v4();

    let writer_a = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        wal_only_config(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    // Not yet fenced.
    writer_a.check_fenced().await.unwrap();

    // Successor claims a higher epoch.
    let writer_b = ShardWriter::open(
        store,
        base_path,
        base_uri,
        wal_only_config(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    assert!(writer_b.epoch() > writer_a.epoch());

    // A's check_fenced surfaces the fence without needing a put round-trip.
    let err = writer_a
        .check_fenced()
        .await
        .expect_err("expected fence error");
    assert!(
        err.to_string().contains("Writer fenced"),
        "unexpected error: {err}"
    );

    // B is the current writer and is not fenced.
    writer_b.check_fenced().await.unwrap();
    writer_b.close().await.unwrap();
}

/// A non-durable put is readable through the **index-backed** arms the moment
/// it returns, not just through a full scan.
///
/// This is what splitting the index apply off the WAL flush buys. Before, the
/// index apply only ran as one arm of the flush, so with `durable_write:
/// false` nothing triggered it on the put path at all: the row sat in the
/// batch store, unindexed, until some later flush happened along. A full scan
/// (which reads the batch store directly) could still find it, while every
/// index-accelerated query could not — the tiers disagreed. Now the apply is
/// triggered per-put in both modes, so `put` returning means "indexed".
#[tokio::test]
async fn test_non_durable_put_is_visible_through_the_index() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();

    let config = ShardWriterConfig {
        durable_write: false,
        ..memtable_config_with_pk(shard_id)
    };
    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    writer
        .put(vec![create_test_batch(&schema, 0, 5)])
        .await
        .unwrap();

    // The indexed PK lookup must find it — this is the arm that saw nothing
    // before, because the index apply had never run.
    let mut scanner = writer.scan().await.unwrap();
    scanner.filter("id = 3").unwrap();
    let hit = scanner.try_into_batch().await.unwrap();
    assert_eq!(
        hit.num_rows(),
        1,
        "an index-backed lookup must see a non-durable put as soon as it returns"
    );

    // ...and so must the unindexed full scan, i.e. the tiers agree.
    let all = writer.scan().await.unwrap().try_into_batch().await.unwrap();
    assert_eq!(all.num_rows(), 5);

    writer.close().await.unwrap();
}

/// The durability cursor is writer-global, so a put into a *post-rotation*
/// memtable must still wait for its own WAL append.
///
/// Batch positions restart at 0 in every memtable, but the durability watch
/// channel spans the writer's whole life and is never reset. When the put
/// path targeted a memtable-local position, the first N puts into every
/// memtable after the first were already "satisfied" by the *previous*
/// memtable's N appends: they acked instantly, with no WAL append, and
/// `durable_write: true` silently degraded to non-durable. Worse, the next
/// append then sent a *smaller* value, walking the watermark backwards and
/// hanging any watcher still waiting on the old, higher one.
///
/// The cursor is now a writer-global exclusive count and every target is
/// lifted through the store's `global_offset`, so it only ever moves forward
/// and a post-rotation put can only be acked by its own append.
#[tokio::test]
async fn test_durable_ack_after_rotation_requires_its_own_wal_append() {
    use crate::dataset::mem_wal::wal::WalTailer;

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = schema_with_pk();
    let shard_id = Uuid::new_v4();

    // A two-batch memtable, so the third put forces a freeze + rotation.
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

    // Fill and rotate the first memtable.
    for i in 0..3 {
        writer
            .put(vec![create_test_batch(&schema, i * 5, 5)])
            .await
            .unwrap();
    }
    let stats = writer.memtable_stats().await.unwrap();
    assert!(
        stats.global_offset > 0,
        "expected a rotation; the active memtable is still the writer's first"
    );

    // Every batch acked so far must be genuinely durable, and the cursor must
    // cover the active memtable's entire prefix rather than lagging inside it.
    let stats = writer.memtable_stats().await.unwrap();
    assert!(
        stats.durable_batch_count >= stats.global_offset + stats.batch_count,
        "durable_write acked a put the WAL never received: durable={} but the active \
         memtable spans [{}, {})",
        stats.durable_batch_count,
        stats.global_offset,
        stats.global_offset + stats.batch_count
    );

    // And the WAL really holds every row we acked (3 puts x 5 rows).
    writer.close().await.unwrap();
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
        wal_rows, 15,
        "every acked row must be in the WAL; a post-rotation put that acked without an \
         append would leave rows missing"
    );
}

/// Regression test for the size-based trigger after a drain.
///
/// Earlier the WAL-only size trigger used a monotonic counter
/// (`wal_flush_trigger_count`) which never reset across drains. After
/// the first crossing the counter was >= 1 and `pending_bytes / threshold`
/// could never grow past 1 (because pending_bytes resets on drain), so
/// the size trigger silently stopped firing. This test pushes batches
/// to cross the size threshold multiple times across drains and asserts
/// every crossing produces a WAL entry.
#[tokio::test]
async fn test_wal_only_size_trigger_fires_repeatedly() {
    use crate::dataset::mem_wal::wal::WalTailer;

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    let mut config = wal_only_config(Uuid::new_v4());
    let shard_id = config.shard_id;
    // Non-durable so puts don't auto-flush. Time trigger off so only
    // the size trigger drives flushes.
    config.durable_write = false;
    config.max_wal_flush_interval = None;
    // Pick a tiny threshold so a single batch crosses it.
    config.max_wal_buffer_size = 1;

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

    // Three puts, each large enough to cross the (1-byte) threshold.
    // Without the fix, only the first would trigger; the rest would
    // sit in the queue until close().
    for i in 0..3 {
        writer
            .put(vec![create_test_batch(&schema, i * 10, 10)])
            .await
            .unwrap();
        // Yield, then sleep, so the background flush handler can
        // drain the trigger queue before the next push — otherwise
        // multiple pending triggers can coalesce into a single drain.
        // 50ms historically failed on slow Windows CI runners; 250ms
        // gives a comfortable margin without making the suite slow.
        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }

    writer.close().await.unwrap();

    // Each put should have produced its own WAL entry — three crossings,
    // three entries. Without the regression fix, all three batches end
    // up in a single entry written by `close()`.
    let tailer = WalTailer::new(store, base_path, "", shard_id);
    let next = tailer.next_position().await.unwrap();
    assert!(
        next >= 3,
        "expected at least 3 WAL entries (one per crossing), got next_position = {next}"
    );
}

/// Regression test for concurrent durable WAL-only puts on a fenced
/// writer. Earlier `flush_from_wal_only` did a destructive `drain()`
/// before calling `wal_appender.append`. If the append failed (e.g.
/// fence), the drained batches were dropped — the next concurrent put
/// would then see an empty pending queue and spuriously return Ok,
/// hiding the data loss. With the snapshot/commit fix, the failed flush
/// leaves the batches in the queue for retry, and — because the flush is
/// terminal — it poisons the writer, waking *both* parked durability
/// waiters with the typed fence error instead of either one hanging.
#[tokio::test]
async fn test_wal_only_fenced_concurrent_puts_do_not_silently_succeed() {
    use std::sync::Arc;

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let shard_id = Uuid::new_v4();

    // Writer A claims epoch 1, writes one entry (takes WAL position 1,
    // caches its next-position as 2 internally).
    let writer_a = Arc::new(
        ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap(),
    );
    writer_a
        .put(vec![create_test_batch(&schema, 0, 1)])
        .await
        .unwrap();

    // Writer B claims epoch 2 and writes its own entry (takes WAL
    // position 1). A is now fenced: A's next put will attempt WAL
    // position 1 (its cached next), collide with B's entry, and
    // surface a "Writer fenced" error from `check_fenced`.
    let writer_b = ShardWriter::open(
        store,
        base_path,
        base_uri,
        wal_only_config(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    writer_b
        .put(vec![create_test_batch(&schema, 1, 1)])
        .await
        .unwrap();

    // Two concurrent durable puts on the (now-fenced) writer A. With
    // the destructive-drain bug, the first flush would consume both
    // pending batches into a failing append; the second flush would
    // see an empty queue and return spurious success, silently losing
    // the second put's data. Now both puts park on the durability
    // watermark; the ticker's append fails with the fence, poisons the
    // writer, and both waiters wake with the fence error — batches intact
    // in the queue.
    let a1 = writer_a.clone();
    let a2 = writer_a.clone();
    let schema1 = schema.clone();
    let schema2 = schema.clone();
    let h1 = tokio::spawn(async move { a1.put(vec![create_test_batch(&schema1, 2, 1)]).await });
    let h2 = tokio::spawn(async move { a2.put(vec![create_test_batch(&schema2, 3, 1)]).await });

    let r1 = h1.await.unwrap();
    let r2 = h2.await.unwrap();

    assert!(
        r1.is_err() && r2.is_err(),
        "expected both concurrent puts on a fenced writer to fail, got r1={r1:?} r2={r2:?}",
    );

    writer_b.close().await.unwrap();
}
