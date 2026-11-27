// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Experimental, unstable cross-crate resource API.
//!
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
#[doc(hidden)]
pub fn io_task_with_resources<T: Send + 'static, F: FnOnce() -> T + Send + 'static>(
    pool: Arc<dyn MemoryReserver>,
    stage: &'static str,
    work: F,
) -> Result<BoxFuture<'static, Result<T>>> {
    // Native closure storage and the channel's result slot are explicit;
    // runtime task headers and channel bookkeeping use a fixed surcharge.
    // Work and output each occupy one admitted box. Runtime/channel frames
    // contain pointers rather than copies of arbitrarily large inline values.
    // The surcharge covers those frames and their allocation headers; this is
    // cooperative accounting, not a stable promise about Tokio's internals.
    let bytes = size_of::<F>()
        .checked_add(size_of::<T>())
        .and_then(|bytes| bytes.checked_add(4096))
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "I/O completion control size overflow: stage={stage}"
            ))
        })?;
    let control = Arc::new(pool.reserve("I/O completion driver control", bytes)?);
    let work = Box::new(work);
    let running = pool
        .task_group()
        .map(|tasks| tasks.start(stage))
        .transpose()?;
    // Dispatch before entering the independent completion driver, using the
    // caller's I/O pool. The returned await may be dropped without stopping it.
    let runtime = tokio::runtime::Handle::current();
    let (completed, outcome) = tokio::sync::oneshot::channel();
    let driver = async move {
        let work_control = control.clone();
        let result = runtime
            .spawn_blocking(move || Box::new((work(), work_control)))
            .await;
        let _ = completed.send(result);
        drop(control);
    };
    let driver_control = pool.reserve("I/O completion future", size_of_val(&driver))?;
    // Build and admit the public await before dispatch. Transfer the actual
    // handle only after all reservations succeed, so a refused write has no
    // side effects and can safely be retried by its caller.
    let (dispatch, dispatched) = tokio::sync::oneshot::channel::<tokio::task::JoinHandle<()>>();
    let future = async move {
        let driver = dispatched.await.map_err(|error| {
            Error::io(format!(
                "I/O driver was not dispatched: stage={stage}, {error}"
            ))
        })?;
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
            Ok(result) => {
                let (value, _control) = *result;
                Ok(value)
            }
            Err(error) => match error.try_into_panic() {
                Ok(panic) => resume_unwind(panic),
                Err(error) => Err(Error::io(format!(
                    "I/O task failed: stage={stage}, {error}"
                ))),
            },
        }
    };
    let await_control = pool.reserve("I/O await future", size_of_val(&future))?;
    let driver = global_cpu_runtime().spawn(async move {
        // Tuple field order also preserves cleanup-before-completion on unwind.
        let completion = (driver_control, running);
        driver.await;
        drop(completion);
    });
    // The receiver is owned by `future` and cannot have been dropped or polled.
    let _ = dispatch.send(driver);
    Ok(Box::pin(async move {
        let result = future.await;
        drop(await_control);
        result
    }))
}
