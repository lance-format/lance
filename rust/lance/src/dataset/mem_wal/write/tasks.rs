// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Background task dispatch and shutdown lifecycle.

use std::fmt::Debug;
use std::sync::RwLock as StdRwLock;
use std::time::Duration;

use async_trait::async_trait;
use lance_core::utils::tokio::JoinHandle;
use lance_core::{Error, Result};
use log::{debug, error, info, warn};
use tokio::sync::mpsc;
use tokio::time::{Interval, interval_at};
use tokio_util::sync::CancellationToken;

/// Factory function for creating ticker messages.
pub(super) type MessageFactory<T> = Box<dyn Fn() -> T + Send + Sync>;

/// Handler trait for processing messages in a background task.
#[async_trait]
pub trait MessageHandler<T: Send + Debug + 'static>: Send {
    /// Define periodic tickers that generate messages.
    fn tickers(&mut self) -> Vec<(Duration, MessageFactory<T>)> {
        vec![]
    }

    /// Handle a single message.
    async fn handle(&mut self, message: T) -> Result<()>;

    /// Cleanup on shutdown.
    async fn cleanup(&mut self, _shutdown_ok: bool) -> Result<()> {
        Ok(())
    }
}

/// Dispatcher that runs the event loop for a single message handler.
struct TaskDispatcher<T: Send + Debug> {
    handler: Box<dyn MessageHandler<T>>,
    rx: mpsc::UnboundedReceiver<T>,
    cancellation_token: CancellationToken,
    name: String,
}

impl<T: Send + Debug + 'static> TaskDispatcher<T> {
    async fn run(mut self) -> Result<()> {
        let tickers = self.handler.tickers();
        let mut ticker_intervals: Vec<(Interval, MessageFactory<T>)> = tickers
            .into_iter()
            .map(|(duration, factory)| {
                let mut interval = interval_at(tokio::time::Instant::now() + duration, duration);
                // `Burst` (the default) replays every tick missed while `handle()`
                // was running. A WAL append can easily outlast its own interval, so
                // the missed ticks pile up, the ticker arm below is always ready,
                // and — being `biased` — it starves `rx` indefinitely: freeze
                // completion cells and `close()`'s final append never get handled.
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                (interval, factory)
            })
            .collect();

        // A single failing message must not bring down the dispatcher:
        // the WAL flusher and MemTable flusher both run on this loop,
        // and dropping their channels deadlocks all subsequent puts
        // (and panics any task waiting on the corresponding watch).
        // Log and keep draining. Handlers latch terminal failures so later
        // messages cannot publish past a failed generation; reopen replays WAL.
        let result = loop {
            if ticker_intervals.is_empty() {
                tokio::select! {
                    biased;
                    _ = self.cancellation_token.cancelled() => {
                        debug!("Task '{}' received cancellation", self.name);
                        break Ok(());
                    }
                    msg = self.rx.recv() => {
                        match msg {
                            Some(message) => {
                                if let Err(e) = self.handler.handle(message).await {
                                    error!("Task '{}' error handling message: {}", self.name, e);
                                }
                            }
                            None => {
                                debug!("Task '{}' channel closed", self.name);
                                break Ok(());
                            }
                        }
                    }
                }
            } else {
                let first_ticker = ticker_intervals.first_mut().unwrap();
                let first_interval = &mut first_ticker.0;

                tokio::select! {
                    biased;
                    _ = self.cancellation_token.cancelled() => {
                        debug!("Task '{}' received cancellation", self.name);
                        break Ok(());
                    }
                    // Explicit messages outrank the ticker. A tick is a backstop
                    // — it only ever *adds* an append that a real trigger would
                    // have made anyway — whereas a message may be a freeze's
                    // completion cell or `close()`'s final append, which nothing
                    // else will deliver. Polling the ticker first (as this did)
                    // lets a ready tick starve them.
                    msg = self.rx.recv() => {
                        match msg {
                            Some(message) => {
                                if let Err(e) = self.handler.handle(message).await {
                                    error!("Task '{}' error handling message: {}", self.name, e);
                                }
                            }
                            None => {
                                debug!("Task '{}' channel closed", self.name);
                                break Ok(());
                            }
                        }
                    }
                    _ = first_interval.tick() => {
                        let message = (ticker_intervals[0].1)();
                        if let Err(e) = self.handler.handle(message).await {
                            error!("Task '{}' error handling ticker message: {}", self.name, e);
                        }
                    }
                }
            }
        };

        let cleanup_ok = result.is_ok();
        self.handler.cleanup(cleanup_ok).await?;

        info!("Task dispatcher '{}' stopped", self.name);
        result
    }
}

/// Executor that manages multiple background tasks.
pub struct TaskExecutor {
    tasks: StdRwLock<Vec<(String, JoinHandle<Result<()>>)>>,
    cancellation_token: CancellationToken,
}

impl TaskExecutor {
    pub fn new() -> Self {
        Self {
            tasks: StdRwLock::new(Vec::new()),
            cancellation_token: CancellationToken::new(),
        }
    }

    pub fn add_handler<T: Send + Debug + 'static>(
        &self,
        name: String,
        handler: Box<dyn MessageHandler<T>>,
        rx: mpsc::UnboundedReceiver<T>,
    ) -> Result<()> {
        let dispatcher = TaskDispatcher {
            handler,
            rx,
            cancellation_token: self.cancellation_token.clone(),
            name: name.clone(),
        };

        let handle = lance_core::utils::tokio::spawn(async move { dispatcher.run().await });
        self.tasks.write().unwrap().push((name, handle));
        Ok(())
    }

    /// A token cancelled when every task is, which its holder may cancel sooner.
    pub(super) fn child_token(&self) -> CancellationToken {
        self.cancellation_token.child_token()
    }

    /// Register periodic work that is **not** driven by a message.
    ///
    /// Runs `work` on its own task every `every`. Use it for work only the tick
    /// performs: a ticker registered through [`MessageHandler::tickers`] shares
    /// the handler's message loop, where a mailbox that never empties can keep
    /// it from running.
    ///
    /// Cancellation and shutdown match [`Self::add_handler`]: the task observes
    /// the same token and is joined by [`Self::shutdown_all`].
    pub fn add_periodic<F, Fut>(&self, name: String, every: Duration, mut work: F) -> Result<()>
    where
        F: FnMut() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send,
    {
        let cancellation_token = self.cancellation_token.clone();
        let task_name = name.clone();
        let handle = lance_core::utils::tokio::spawn(async move {
            let mut interval = interval_at(tokio::time::Instant::now() + every, every);
            // Same reason as the dispatcher: `Burst` would replay every tick
            // missed while `work` ran, which for a slow `work` means the timer
            // is permanently ready and the cancellation arm never wins.
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    biased;
                    _ = cancellation_token.cancelled() => {
                        debug!("Periodic task '{}' received cancellation", task_name);
                        return Ok(());
                    }
                    _ = interval.tick() => work().await,
                }
            }
        });
        self.tasks.write().unwrap().push((name, handle));
        Ok(())
    }

    /// Cancel and join every handler registered by [`Self::add_handler`].
    ///
    /// Cancellation causes each handler's dispatcher to stop accepting messages and call
    /// [`MessageHandler::cleanup`]. This method waits for every dispatcher to finish, even if
    /// cleanup fails or a dispatcher panics, and then returns the first such failure. It returns
    /// `Ok(())` only after every registered handler has been cleaned up successfully.
    pub async fn shutdown_all(&self) -> Result<()> {
        info!("Shutting down all tasks");
        self.cancellation_token.cancel();

        let tasks = std::mem::take(&mut *self.tasks.write().unwrap());
        let mut first_error = None;
        for (name, handle) in tasks {
            match handle.await {
                Ok(Ok(())) => debug!("Task '{}' completed successfully", name),
                Ok(Err(e)) => {
                    warn!("Task '{}' completed with error: {}", name, e);
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
                Err(e) => {
                    error!("Task '{}' panicked: {}", name, e);
                    if first_error.is_none() {
                        first_error = Some(Error::internal(format!(
                            "Task '{name}' panicked during shutdown: {e}"
                        )));
                    }
                }
            }
        }

        first_error.map_or(Ok(()), Err)
    }
}

impl Default for TaskExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
