// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Detached local I/O completes and releases its owners before join notification.

use std::{panic::resume_unwind, sync::Arc};

use futures::future::BoxFuture;

use crate::{Error, Result, utils::memory::MemoryReserver};

use super::global_cpu_runtime;

/// Dispatch blocking I/O on the caller's I/O runtime, retaining completion
/// ownership independently of cancellation or shutdown of the awaiting runtime.
/// The completion driver runs async code; pure CPU workers never wait for I/O.
/// Admit the payload separately and retain its charge inside `work` and its
/// result. This function admits the additional driver/channel control first.
///
/// ```
/// # use std::sync::Arc;
/// # use lance_core::{Result, utils::{memory::MemoryReserver, tokio::io_task_with_resources}};
/// # async fn example(pool: Arc<dyn MemoryReserver>) -> Result<()> {
/// let metadata = io_task_with_resources(pool, "file metadata", || {
///     std::fs::metadata(".")
/// })?.await??;
/// assert!(metadata.is_dir());
/// # Ok(())
/// # }
/// ```
pub fn io_task_with_resources<T: Send + 'static, F: FnOnce() -> T + Send + 'static>(
    pool: Arc<dyn MemoryReserver>,
    stage: &'static str,
    work: F,
) -> Result<BoxFuture<'static, Result<T>>> {
    let bytes = size_of::<F>()
        .checked_mul(2)
        .and_then(|bytes| {
            size_of::<T>()
                .checked_mul(2)
                .and_then(|output| bytes.checked_add(output))
        })
        .and_then(|bytes| bytes.checked_add(4096))
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "I/O completion control size overflow: stage={stage}"
            ))
        })?;
    let control = pool.reserve("I/O completion driver control", bytes)?;
    let running = pool
        .task_group()
        .map(|tasks| tasks.start(stage))
        .transpose()?;
    // Dispatch before entering the independent completion driver, using the
    // caller's I/O pool. The returned await may be dropped without stopping it.
    let native = tokio::task::spawn_blocking(work);
    let (completed, outcome) = tokio::sync::oneshot::channel();
    let driver = async move {
        let result = native.await;
        let _ = completed.send(result);
        drop(control);
        drop(running);
    };
    let driver = global_cpu_runtime().spawn(driver);
    Ok(Box::pin(async move {
        if let Err(error) = driver.await {
            match error.try_into_panic() {
                Ok(panic) => resume_unwind(panic),
                Err(error) => {
                    return Err(Error::io(format!(
                        "I/O completion driver failed: stage={stage}, {error}"
                    )));
                }
            }
        }
        match outcome.await.map_err(|error| {
            Error::io(format!(
                "I/O completion outcome lost: stage={stage}, {error}"
            ))
        })? {
            Ok(result) => Ok(result),
            Err(error) => match error.try_into_panic() {
                Ok(panic) => resume_unwind(panic),
                Err(error) => Err(Error::io(format!(
                    "I/O task failed: stage={stage}, {error}"
                ))),
            },
        }
    }))
}
