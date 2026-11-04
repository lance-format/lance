// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Background dispatcher error handling and shutdown.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use lance_core::{Error, Result};
use tokio::sync::mpsc;

use super::*;

/// `MemTableStats::frozen_count` and `ShardMemory::frozen_bytes` count
/// different things on purpose: bytes drop on flush commit, the handle
/// lingers for `frozen_memtable_grace`. A long grace therefore leaves count
/// non-zero with bytes back at zero — and pins that the two surfaces are
/// answering different questions, not disagreeing about one.
/// A saturated handler starves its own ticker; `add_periodic` does not.
///
/// `TaskDispatcher` biases a handler's mailbox ahead of its ticker, and the
/// `select!` is not evaluated while `handle()` awaits, so a mailbox that
/// never empties leaves the ticker no instant to win. Both halves are
/// asserted so the distinction holds.
#[tokio::test(start_paused = true)]
async fn test_saturated_handler_starves_its_ticker_but_not_a_periodic_task() {
    #[derive(Debug)]
    enum Msg {
        Work,
        Tick,
    }

    #[derive(Debug)]
    struct SlowHandler {
        worked: Arc<AtomicUsize>,
        ticked: Arc<AtomicUsize>,
        work: Duration,
        tick: Duration,
    }

    #[async_trait]
    impl MessageHandler<Msg> for SlowHandler {
        fn tickers(&mut self) -> Vec<(Duration, MessageFactory<Msg>)> {
            vec![(self.tick, Box::new(|| Msg::Tick))]
        }

        async fn handle(&mut self, message: Msg) -> Result<()> {
            match message {
                Msg::Work => {
                    self.worked.fetch_add(1, Ordering::Relaxed);
                    tokio::time::sleep(self.work).await;
                }
                Msg::Tick => {
                    self.ticked.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(())
        }
    }

    // One unit of work outlasts the tick interval: that is what starves a
    // ticker sharing the handler's loop.
    let work = Duration::from_millis(100);
    let tick = Duration::from_millis(10);
    let run = Duration::from_millis(1000);
    let expected_ticks = (run.as_millis() / tick.as_millis()) as usize;

    let executor = TaskExecutor::new();
    let worked = Arc::new(AtomicUsize::new(0));
    let ticked = Arc::new(AtomicUsize::new(0));
    let swept = Arc::new(AtomicUsize::new(0));

    let (tx, rx) = mpsc::unbounded_channel();
    executor
        .add_handler(
            "slow".to_string(),
            Box::new(SlowHandler {
                worked: worked.clone(),
                ticked: ticked.clone(),
                work,
                tick,
            }),
            rx,
        )
        .unwrap();

    // The same cadence, on its own task rather than the handler's loop.
    let swept_by_task = swept.clone();
    executor
        .add_periodic("sweeper".to_string(), tick, move || {
            let swept = swept_by_task.clone();
            async move {
                swept.fetch_add(1, Ordering::Relaxed);
            }
        })
        .unwrap();

    // Enough queued work that the mailbox never empties during the run.
    for _ in 0..(run.as_millis() / work.as_millis()) + 2 {
        tx.send(Msg::Work).unwrap();
    }
    tokio::time::sleep(run).await;

    let (worked, ticked, swept) = (
        worked.load(Ordering::Relaxed),
        ticked.load(Ordering::Relaxed),
        swept.load(Ordering::Relaxed),
    );
    assert!(worked > 0, "the handler must have been busy");
    assert!(
        ticked * 10 < expected_ticks,
        "a ticker sharing a saturated handler's loop should be starved, but \
         it ran {ticked} of ~{expected_ticks}"
    );
    assert!(
        swept * 2 >= expected_ticks,
        "periodic task ran {swept} of ~{expected_ticks}"
    );

    executor.shutdown_all().await.ok();
}

/// Regression for #6713: a single failing `handle()` must not kill
/// the dispatcher. Earlier the loop would `break Err(e)` on the
/// first message error, dropping the rx side and stranding
/// subsequent senders. The flusher tasks need to survive transient
/// errors so the writer keeps making forward progress.
#[tokio::test]
async fn test_task_dispatcher_survives_handle_error() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FlakyHandler {
        call_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl MessageHandler<u32> for FlakyHandler {
        async fn handle(&mut self, message: u32) -> Result<()> {
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                Err(Error::io("first message intentionally fails"))
            } else {
                let _ = message;
                Ok(())
            }
        }
    }

    let executor = TaskExecutor::new();
    let call_count = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::unbounded_channel::<u32>();
    executor
        .add_handler(
            "flaky".to_string(),
            Box::new(FlakyHandler {
                call_count: call_count.clone(),
            }),
            rx,
        )
        .unwrap();

    // Send three messages: the first errors, the next two should
    // still be delivered to the (still-alive) handler.
    tx.send(1).unwrap();
    tx.send(2).unwrap();
    tx.send(3).unwrap();

    for _ in 0..50 {
        if call_count.load(Ordering::SeqCst) >= 3 {
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
    }

    assert!(
        call_count.load(Ordering::SeqCst) >= 3,
        "dispatcher exited after the first error; only {} message(s) were handled",
        call_count.load(Ordering::SeqCst)
    );

    executor
        .shutdown_all()
        .await
        .expect("dispatcher should shut down successfully");
}

#[tokio::test]
async fn test_task_executor_shutdown_propagates_cleanup_error_and_joins_all_tasks() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CleanupHandler {
        cleanup_count: Arc<AtomicUsize>,
        error_message: Option<&'static str>,
    }

    #[async_trait]
    impl MessageHandler<u32> for CleanupHandler {
        async fn handle(&mut self, _message: u32) -> Result<()> {
            Ok(())
        }

        async fn cleanup(&mut self, _shutdown_ok: bool) -> Result<()> {
            self.cleanup_count.fetch_add(1, Ordering::SeqCst);
            match self.error_message {
                Some(message) => Err(Error::io(message)),
                None => Ok(()),
            }
        }
    }

    let executor = TaskExecutor::new();
    let cleanup_count = Arc::new(AtomicUsize::new(0));
    let (_failing_tx, failing_rx) = mpsc::unbounded_channel::<u32>();
    executor
        .add_handler(
            "failing-cleanup".to_string(),
            Box::new(CleanupHandler {
                cleanup_count: cleanup_count.clone(),
                error_message: Some("intentional cleanup failure"),
            }),
            failing_rx,
        )
        .unwrap();
    let (_successful_tx, successful_rx) = mpsc::unbounded_channel::<u32>();
    executor
        .add_handler(
            "successful-cleanup".to_string(),
            Box::new(CleanupHandler {
                cleanup_count: cleanup_count.clone(),
                error_message: None,
            }),
            successful_rx,
        )
        .unwrap();

    let error = executor
        .shutdown_all()
        .await
        .expect_err("shutdown must propagate the handler cleanup failure");
    assert!(matches!(&error, Error::IO { .. }));
    assert!(
        error.to_string().contains("intentional cleanup failure"),
        "unexpected error: {error}"
    );
    assert_eq!(
        cleanup_count.load(Ordering::SeqCst),
        2,
        "shutdown must join and clean up every task after the first failure"
    );
    assert!(executor.tasks.read().unwrap().is_empty());
}

#[tokio::test]
async fn test_task_executor_shutdown_propagates_task_panic() {
    struct PanickingCleanupHandler;

    #[async_trait]
    impl MessageHandler<u32> for PanickingCleanupHandler {
        async fn handle(&mut self, _message: u32) -> Result<()> {
            Ok(())
        }

        async fn cleanup(&mut self, _shutdown_ok: bool) -> Result<()> {
            panic!("intentional cleanup panic");
        }
    }

    let executor = TaskExecutor::new();
    let (_tx, rx) = mpsc::unbounded_channel::<u32>();
    executor
        .add_handler(
            "panicking-cleanup".to_string(),
            Box::new(PanickingCleanupHandler),
            rx,
        )
        .unwrap();

    let error = executor
        .shutdown_all()
        .await
        .expect_err("shutdown must propagate the task panic");
    assert!(matches!(&error, Error::Internal { .. }));
    assert!(
        error.to_string().contains("panicking-cleanup")
            && error.to_string().contains("panicked during shutdown"),
        "unexpected error: {error}"
    );
}
