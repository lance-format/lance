// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Memory snapshots, admission control, and backpressure waits.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock as StdRwLock};
use std::time::{Duration, Instant};

use lance_core::{Error, Result};
use uuid::Uuid;

use super::*;
use crate::dataset::mem_wal::write::ShardWriter;
use crate::dataset::mem_wal::write::test_support::*;

/// The controller sits in a private `WriterMode` variant, reachable only
/// through the writer. Both modes must answer.
#[rstest::rstest]
#[case::memtable(true)]
#[case::wal_only(false)]
#[tokio::test]
async fn test_backpressure_stats_reachable_in_both_modes(#[case] enable_memtable: bool) {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        enable_memtable,
        ..Default::default()
    };

    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    // A writer well under the threshold has never throttled.
    let stats = writer.backpressure_stats();
    assert_eq!(stats.total_count, 0);
    assert_eq!(stats.total_wait_ms, 0);
    assert_eq!(stats.active_count, 0);

    writer.close().await.unwrap();
}
#[tokio::test]
async fn test_no_backpressure_when_under_threshold() {
    let config = ShardWriterConfig::default().with_max_unflushed_memtable_bytes(1024 * 1024); // 1MB

    let controller = LocalBackpressureController::new(&config);

    // Should return immediately - well under threshold (100 bytes < 1MB)
    controller
        .maybe_apply_backpressure(fixed_memory(100))
        .await
        .unwrap();

    assert_eq!(controller.stats().count(), 0);
}

#[tokio::test]
async fn test_backpressure_loops_until_under_threshold() {
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    let config = ShardWriterConfig::default()
        .with_max_unflushed_memtable_bytes(100) // Very low threshold
        .with_backpressure_log_interval(Duration::from_millis(50));

    // Simulate: starts at 1000 bytes, drops by 400 each call (simulating memtable flushes)
    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_clone = call_count.clone();

    let controller = LocalBackpressureController::new(&config);
    let draining = fake_memory(move || {
        let count = call_count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // 1000 -> 600 -> 200 -> under threshold (need 3 iterations)
        1000usize.saturating_sub(count * 400)
    });

    controller.maybe_apply_backpressure(draining).await.unwrap();

    // Should have read the shard 4 times (initial + 3 waits until under 100)
    assert_eq!(call_count.load(std::sync::atomic::Ordering::Relaxed), 4);
    // Should have recorded backpressure wait time (waited 3 times)
    assert_eq!(controller.stats().count(), 1);
    assert_eq!(
        controller.stats().snapshot().active_count,
        0,
        "the wait is over, so nobody is parked"
    );
}

/// The totals only move when a wait ends, so a first stall would be
/// invisible to a poller if `active_count` did not report it live.
#[tokio::test]
async fn test_backpressure_in_progress_wait_is_observable() {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering as AtomicOrdering;
    use std::time::Duration;

    let config = ShardWriterConfig::default()
        .with_max_unflushed_memtable_bytes(100)
        .with_backpressure_log_interval(Duration::from_millis(50));

    let unflushed = Arc::new(AtomicUsize::new(1000));
    let release = unflushed.clone();

    let controller = LocalBackpressureController::new(&config);
    let stats = controller.stats().clone();

    let parked = controller
        .maybe_apply_backpressure(fake_memory(move || unflushed.load(AtomicOrdering::Relaxed)));

    let observer = async {
        // Bounded so a regression that never publishes the park fails here
        // instead of hanging the suite.
        let deadline = Instant::now() + Duration::from_secs(5);
        while stats.snapshot().active_count == 0 {
            assert!(
                Instant::now() < deadline,
                "an ongoing wait was never published"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let mid = stats.snapshot();
        assert_eq!(mid.active_count, 1);
        assert_eq!(
            mid.total_count, 0,
            "a wait still in progress is not a completed one"
        );
        assert_eq!(mid.total_wait_ms, 0);

        release.store(0, AtomicOrdering::Relaxed);
    };

    let (result, ()) = tokio::join!(parked, observer);
    result.unwrap();

    let after = stats.snapshot();
    assert_eq!(after.active_count, 0, "the guard drops when the wait ends");
    assert_eq!(after.total_count, 1);
}

/// A `put` whose caller times out drops the wait future mid-park. The gauge
/// must come back down, or a cancelled writer is throttled forever on paper.
#[tokio::test]
async fn test_backpressure_cancelled_wait_does_not_leak_active_count() {
    use std::time::Duration;

    let config = ShardWriterConfig::default()
        .with_max_unflushed_memtable_bytes(100)
        .with_backpressure_log_interval(Duration::from_millis(50));

    let controller = LocalBackpressureController::new(&config);

    // Never drops below the threshold, so the timeout is what ends the wait.
    assert!(
        tokio::time::timeout(
            Duration::from_millis(50),
            controller.maybe_apply_backpressure(fixed_memory(1000)),
        )
        .await
        .is_err()
    );

    let after = controller.stats().snapshot();
    assert_eq!(after.active_count, 0, "cancellation must release the guard");
    assert_eq!(
        after.total_count, 0,
        "a cancelled wait never completed, so it is not a completed wait"
    );
}

/// An injected controller *replaces* the built-in valve rather than
/// stacking on it, and is handed what the calling shard holds, split the
/// way relief cares about — everything it needs to own the whole policy,
/// per-shard rules included.
#[tokio::test]
async fn test_injected_controller_replaces_default_and_sees_shard_memory() {
    let seen = Arc::new(StdRwLock::new(Vec::new()));
    let spy = Arc::new(SpyController {
        seen: seen.clone(),
        reject: false,
    });
    // A budget the built-in valve would trip on instantly, to prove it is
    // not the thing being consulted.
    let config = ShardWriterConfig::default()
        .with_max_unflushed_memtable_bytes(1)
        .with_backpressure(spy);

    let controller = resolve_backpressure(&config);
    controller
        .maybe_apply_backpressure(fixed_memory(1000))
        .await
        .unwrap();

    assert_eq!(
        *seen.read().unwrap(),
        vec![(1000, 0)],
        "the injected controller ran, and the built-in valve did not park the write"
    );
}

/// `ShardMemory` must stay live across polls: a controller that delays is
/// waiting for these bytes to fall, so a captured copy would spin forever.
#[tokio::test]
async fn test_shard_memory_reflects_drain_across_polls() {
    #[derive(Debug)]
    struct DrainWaiter {
        polls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl BackpressureController for DrainWaiter {
        async fn maybe_apply_backpressure(&self, shard: ShardMemory) -> Result<()> {
            while shard.unflushed_bytes() > 0 {
                self.polls.fetch_add(1, Ordering::Relaxed);
                tokio::task::yield_now().await;
            }
            Ok(())
        }
    }

    let resident = Arc::new(AtomicUsize::new(4_096));
    let drain = resident.clone();
    let view = fake_memory(move || resident.load(Ordering::Relaxed));

    let controller = Arc::new(DrainWaiter {
        polls: AtomicUsize::new(0),
    });
    let gate = controller.clone();
    let waiting = tokio::spawn(async move { gate.maybe_apply_backpressure(view).await });

    // Let the gate observe the full pool, then drain it as a flush commit
    // would. A stale copy would never see this and the task would hang.
    tokio::task::yield_now().await;
    drain.store(0, Ordering::Relaxed);

    waiting.await.unwrap().unwrap();
    assert!(controller.polls.load(Ordering::Relaxed) > 0);
}

/// With nothing injected, lance keeps its own per-shard valve.
#[tokio::test]
async fn test_default_backpressure_is_used_when_none_injected() {
    let config = ShardWriterConfig::default().with_max_unflushed_memtable_bytes(100);
    let controller = resolve_backpressure(&config);

    let polls = Arc::new(AtomicUsize::new(0));
    let polls_clone = polls.clone();
    controller
        .maybe_apply_backpressure(fake_memory(move || {
            polls_clone.fetch_add(1, Ordering::Relaxed);
            0
        }))
        .await
        .unwrap();

    assert_eq!(polls.load(Ordering::Relaxed), 1);
}

/// A rejecting controller surfaces as `Error::Backpressure`, which is
/// distinguishable from a real failure without matching on the message.
#[tokio::test]
async fn test_injected_controller_rejects_with_backpressure_error() {
    let spy = Arc::new(SpyController {
        seen: Arc::new(StdRwLock::new(Vec::new())),
        reject: true,
    });
    let config = ShardWriterConfig::default().with_backpressure(spy);
    let controller = resolve_backpressure(&config);

    let err = controller
        .maybe_apply_backpressure(empty_shard_memory())
        .await
        .unwrap_err();
    assert!(err.is_backpressure(), "expected backpressure, got {err:?}");
}

/// A `ShardMemory` backed by a closure instead of a live writer, re-read on
/// every poll exactly as the real one is.
fn fake_memory(read: impl Fn() -> usize + Send + Sync + 'static) -> ShardMemory {
    ShardMemory {
        source: ShardMemorySource::Fake(Arc::new(read)),
        freeze_required: false,
    }
}

fn fixed_memory(unflushed: usize) -> ShardMemory {
    fake_memory(move || unflushed)
}

fn empty_shard_memory() -> ShardMemory {
    fixed_memory(0)
}

/// Records what it saw and answers with a fixed verdict.
#[derive(Debug)]
struct SpyController {
    seen: Arc<StdRwLock<Vec<(usize, usize)>>>,
    reject: bool,
}

#[async_trait::async_trait]
impl BackpressureController for SpyController {
    async fn maybe_apply_backpressure(&self, shard: ShardMemory) -> Result<()> {
        self.seen
            .write()
            .unwrap()
            .push((shard.active_bytes(), shard.frozen_bytes()));
        if self.reject {
            return Err(Error::backpressure("full"));
        }
        Ok(())
    }
}
