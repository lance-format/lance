// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Writer admission and poisoning across the state lock.

use super::super::*;
use super::support::*;
use crate::dataset::mem_wal::memtable::CacheConfig;
use crate::dataset::mem_wal::write::backpressure::Drain;
use crate::dataset::mem_wal::write::test_support::*;
use lance_core::FenceReason;
use rstest::rstest;

#[rstest]
#[case::parked_in_controller(false)]
#[case::waiting_for_state_lock(true)]
#[tokio::test]
async fn test_poison_interrupts_write_admission(#[case] release_admission: bool) {
    #[derive(Debug, Default)]
    struct ParkedController {
        entered: tokio::sync::Notify,
        release: tokio::sync::Notify,
        exited: tokio::sync::Notify,
    }

    #[async_trait::async_trait]
    impl BackpressureController for ParkedController {
        async fn maybe_apply_backpressure(&self, _shard: ShardMemory) -> Result<()> {
            self.entered.notify_one();
            self.release.notified().await;
            self.exited.notify_one();
            Ok(())
        }
    }

    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let controller = Arc::new(ParkedController::default());
    let schema = create_test_schema();
    let config = freeze_fence_test_config(Uuid::new_v4()).with_backpressure(controller.clone());
    let writer = Arc::new(
        ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap(),
    );
    let pending_writer = writer.clone();
    let pending = tokio::spawn(async move {
        pending_writer
            .put(vec![create_test_batch(&schema, 0, 1)])
            .await
    });
    controller.entered.notified().await;
    let WriterMode::MemTable { state, .. } = &writer.mode else {
        unreachable!();
    };
    let locked = state.write().await;
    if release_admission {
        controller.release.notify_one();
        controller.exited.notified().await;
    }
    writer
        .wal_flusher
        .poison(&Error::writer_poisoned("SSTable flush failed"));
    drop(locked);

    let error = tokio::time::timeout(Duration::from_secs(1), pending)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert_eq!(error.fence_reason(), Some(FenceReason::PersistenceFailure));
    assert!(error.to_string().contains("SSTable flush failed"));
    assert_eq!(writer.memtable_stats().await.unwrap().row_count, 0);
    writer.abort().await.unwrap();
}

/// The keystone invariant: the published snapshot is **derived** from
/// `WriterState`, never adjusted, so it cannot drift from it — across every
/// event that changes the memtable set.
///
/// This is what replaced a pair of incremented counters. A counter has no
/// way back: one missed decrement is permanent, and the pod eventually
/// refuses every write with a memtable that reads empty. So this walks the
/// writer through open, puts, a freeze, a flush commit, and puts into the
/// fresh memtable, checking the invariant after each.
#[rstest]
#[case::grace_keeps_handles(Duration::from_secs(600))]
#[case::zero_grace_evicts_on_commit(Duration::ZERO)]
#[tokio::test]
async fn test_memory_snapshot_never_drifts_from_writer_state(#[case] grace: Duration) {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        durable_write: false,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        // Small enough that the loop below freezes several times.
        max_memtable_size: 2048,
        frozen_memtable_grace: grace,
        ..Default::default()
    };
    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    // Seeded at open, before anything has been written.
    assert_no_drift(&writer, "open").await;
    let initial_generation = writer.memtable_stats().await.unwrap().generation;

    for i in 0..24 {
        writer
            .put(vec![create_test_batch(&schema, i * 10, 10)])
            .await
            .unwrap();
        assert_no_drift(&writer, &format!("put {i}")).await;
    }

    // Freezes happened above; make sure at least one did, or this test proves
    // nothing about the freeze path. Generation is monotonic, unlike the
    // current frozen handle count, which zero grace removes on flush commit.
    let after_puts = writer.memtable_stats().await.unwrap();
    assert!(
        after_puts.generation > initial_generation,
        "the loop must have advanced the active generation to cover freeze: \
         {initial_generation} -> {}",
        after_puts.generation
    );

    writer.wait_for_flush_drain().await.unwrap();
    assert_no_drift(&writer, "flush drain").await;
    let after_drain = writer.memtable_stats().await.unwrap();

    // A long grace keeps the flushed handles in the read view and a zero
    // grace evicts them on commit — two different publish paths. Either way
    // they are reclaimable, so neither may keep metering.
    assert_eq!(
        writer.memory().frozen_bytes(),
        0,
        "flushed memtables are reclaimable and must stop metering (grace {grace:?})"
    );
    if grace.is_zero() {
        assert_eq!(
            after_drain.frozen_count, 0,
            "zero grace must evict flushed handles on commit"
        );
    } else {
        assert!(
            after_drain.frozen_count > 0,
            "long grace must retain flushed handles in the read view"
        );
    }

    for i in 24..32 {
        writer
            .put(vec![create_test_batch(&schema, i * 10, 10)])
            .await
            .unwrap();
        assert_no_drift(&writer, &format!("post-flush put {i}")).await;
    }

    writer.close().await.unwrap();
}

/// The bug the whole design exists to avoid: the old accounting read
/// `WriterState` through `try_read()`, which fails while a writer holds the
/// lock — and, because tokio's `RwLock` is write-preferring, also while one
/// is merely queued. It therefore reported **zero** on exactly the shards
/// taking writes, so a pod under load looked empty.
///
/// Holding the write lock outright is the deterministic version of that
/// race.
#[tokio::test]
async fn test_memory_is_readable_while_a_writer_holds_the_lock() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
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

    writer
        .put(vec![create_test_batch(&schema, 0, 100)])
        .await
        .unwrap();
    let unlocked = writer.memory().unflushed_bytes();
    assert!(unlocked > 0, "rows are resident, so this cannot be zero");

    let state_lock = writer.memtable_state_lock().unwrap().clone();
    let held = state_lock.write().await;
    assert_eq!(
        writer.memory().unflushed_bytes(),
        unlocked,
        "a writer holding the lock must not zero the memory view"
    );
    // The drain classification comes off the same published snapshot, so it
    // answers under the lock as well. Nothing is frozen here, so the honest
    // answer is that no flush is outstanding.
    assert!(
        matches!(writer.memory().drain(), Drain::Stalled),
        "the drain classification is published too, so it reads under the lock"
    );
    drop(held);

    writer.close().await.unwrap();
}

/// Two freeze predicates exist — `MemTable::should_flush` and
/// `memtable_reached_flush_threshold` — and their *row-window* arms must trip
/// at the same byte. They did not: one counted the PK bloom filter and the
/// other did not, so they disagreed by a fixed offset on every memtable.
///
/// Only that arm is shared. `memtable_reached_flush_threshold` also freezes on
/// resident bytes and row count, which `should_flush` knows nothing about, so
/// both are held out of range below to compare like with like.
#[tokio::test]
async fn test_both_freeze_predicates_share_one_byte_arm() {
    let schema = create_test_schema();
    let mut memtable =
        MemTable::with_capacity(schema.clone(), 1, vec![], CacheConfig::default(), 64).unwrap();
    memtable
        .insert(create_test_batch(&schema, 0, 50))
        .await
        .unwrap();

    let at = memtable.batch_store().row_bytes();
    assert!(at > 0, "the memtable must hold something to be a test");
    // Owned batches, so the pinned footprint sits just under the window sum
    // (no per-batch header in the allocation walk). Asserted rather than
    // assumed: if it ever exceeded `at`, the retained arm would fire and this
    // would be comparing something other than the row arm.

    // `incoming_batches` of 1 against a capacity of 64 keeps the batch-count
    // arm out of it, and the two `usize::MAX` limits keep the row-count and
    // resident arms out, so only the row-window arms are being compared.
    for (bytes, expected) in [(at, true), (at + 1, false)] {
        assert_eq!(
            memtable.should_flush(bytes),
            expected,
            "should_flush at {bytes}"
        );
        assert_eq!(
            memtable_reached_flush_threshold(&memtable, bytes, usize::MAX, usize::MAX, 1, 1),
            expected,
            "the two freeze predicates disagree at {bytes}; a bloom-sized offset \
             between them makes every memtable freeze early on one path"
        );
    }
}

/// A handle taken before any rows exist must still see them arrive: the
/// memory view is read without the writer lock, so a handle that captured a
/// value instead of a live counter would gate writes on a stale reading.
/// Index heap is counted, row bytes are not inflated by it.
#[tokio::test]
async fn test_memory_handle_tracks_the_live_memtable() {
    let schema = create_test_schema();
    let mut memtable =
        MemTable::with_capacity(schema.clone(), 1, vec![], CacheConfig::default(), 8).unwrap();

    let handle = in_memory_ref(&memtable);
    let empty_rows = handle.row_bytes();
    for _ in 0..3 {
        memtable
            .insert(create_test_batch(&schema, 0, 10))
            .await
            .unwrap();
    }

    assert!(
        handle.row_bytes() > empty_rows,
        "a handle taken before the writes must still see them"
    );
    assert_eq!(
        handle.row_bytes(),
        in_memory_ref(&memtable).row_bytes(),
        "a handle and a freshly taken one must agree"
    );
    // The bloom filter is fixed-size and lives in the index term, so it
    // never moves the flush unit.
    assert_eq!(
        handle.row_bytes(),
        memtable.batch_store().row_bytes(),
        "row bytes are the flush unit: batches only"
    );
    assert_eq!(
        handle.index_bytes(),
        crate::dataset::mem_wal::memtable::pk_bloom_filter_bytes(),
        "an unindexed memtable still holds its PK bloom filter"
    );
    // The ceiling is built on what the batches keep alive, not on the flush
    // unit. The two row measures are close but never identical even for
    // owned batches — buffer capacity is padded, and `row_bytes` charges a
    // `RecordBatch` header per batch that an allocation walk does not see.
    // How far they diverge for zero-copy slices, which is the case that
    // matters, is pinned in `BatchStore`'s own tests.
    assert_eq!(
        handle.resident_bytes(),
        handle.retained_row_bytes() + handle.index_bytes()
    );
}
