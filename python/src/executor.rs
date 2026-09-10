// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::mpsc::RecvTimeoutError;

use futures::Future;
use pyo3::{PyResult, Python, exceptions::PyRuntimeError};

pub const SIGNAL_CHECK_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

fn is_python314_or_later(py: Option<Python<'_>>) -> bool {
    py.is_some_and(|py| py.version_info() >= (3, 14))
}

/// A wrapper around tokio runtime.
///
/// This is used to spawn tasks in the background and wait synchronously for them
/// to complete. This is important for cases where we want to avoid nested
/// block_on() calls.
///
/// The methods also make sure that the GIL is released before spawning the task.
pub struct BackgroundExecutor {
    pub runtime: tokio::runtime::Runtime,
}

impl BackgroundExecutor {
    pub fn get_runtime_handle(&self) -> Option<tokio::runtime::Handle> {
        Some(self.runtime.handle().clone())
    }
    /// Creates a tokio runtime and spawns a thread to run it.
    pub fn new() -> Self {
        // Create a new Runtime to run tasks
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("lance_background_thread")
            .build()
            .expect("Creating Tokio runtime");

        Self { runtime }
    }

    /// Spawn a task and wait for it to complete.
    ///
    /// This method is safe to use with inputs that may reference a Rust async
    /// runtime.
    pub fn spawn<T>(&self, py: Option<Python<'_>>, task: T) -> PyResult<T::Output>
    where
        T: Future + Send + 'static,
        T::Output: Send + 'static,
    {
        if let Some(py) = py {
            py.detach(|| self.spawn_impl(task))
        } else {
            let mut task = Some(task);
            if let Some(result) = Python::try_attach(|py| {
                let task = task.take().expect("task should not be taken");
                py.detach(|| self.spawn_impl(task))
            }) {
                result
            } else {
                self.spawn_impl(task.expect("task should still be available"))
            }
        }
    }

    fn spawn_impl<T>(&self, task: T) -> PyResult<T::Output>
    where
        T: Future + Send + 'static,
        T::Output: Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel::<T::Output>();

        let fut = Box::pin(async move {
            let task_output = task.await;
            tokio::task::spawn_blocking(move || {
                tx.send(task_output).ok();
            })
            .await
            .unwrap();
        });

        let handle = self.runtime.spawn(fut);

        loop {
            // Check for keyboard interrupts
            let signal_check = match Python::try_attach(|py| py.check_signals()) {
                Some(result) => result,
                // Python may be finalizing or unavailable. In this state we can't
                // observe KeyboardInterrupt reliably, but we should not panic.
                None => Ok(()),
            };
            match signal_check {
                Ok(_) => {}
                Err(err) => {
                    handle.abort();
                    return Err(err);
                }
            }
            // Wait for 100ms before checking signals again
            match rx.recv_timeout(SIGNAL_CHECK_INTERVAL) {
                Ok(output) => return Ok(output),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    handle.abort();
                    return Err(PyRuntimeError::new_err("Task was aborted"));
                }
            }
        }
    }

    /// Spawn a task in the background
    pub fn spawn_background<T>(&self, py: Option<Python<'_>>, task: T)
    where
        T: Future + Send + 'static,
        T::Output: Send + 'static,
    {
        if let Some(py) = py {
            py.detach(|| {
                self.runtime.spawn(task);
            })
        } else {
            let mut task = Some(task);
            if Python::try_attach(|py| {
                let task = task.take().expect("task should not be taken");
                py.detach(|| {
                    self.runtime.spawn(task);
                })
            })
            .is_none()
            {
                self.runtime
                    .spawn(task.expect("task should still be available"));
            }
        }
    }

    /// Block on a future and wait for it to complete.
    ///
    /// This helper method also frees the GIL before blocking.
    ///
    /// This method is NOT safe to use with inputs that may reference a Rust async
    /// runtime. If the future references an async runtime, it will panic on an
    /// error: "Cannot start a runtime from within a runtime."
    pub fn block_on<F: Future + Send>(
        &self,
        py: Option<Python<'_>>,
        future: F,
    ) -> PyResult<F::Output>
    where
        F::Output: Send,
    {
        let future = Self::result_or_interrupt(future);
        if let Some(py) = py {
            py.detach(move || self.runtime.block_on(future))
        } else {
            let mut future = Some(future);
            if let Some(result) = Python::try_attach(|py| {
                let future = future.take().expect("future should not be taken");
                py.detach(|| self.runtime.block_on(future))
            }) {
                result
            } else {
                self.runtime
                    .block_on(future.expect("future should still be available"))
            }
        }
    }

    /// Block on a future, periodically draining a caller-provided pump.
    ///
    /// This is intended for progress callbacks where the future emits events from
    /// background tasks and the caller needs to run Python code while waiting for
    /// the future to complete.
    ///
    /// Unlike [`block_on`], this method does **not** wrap the future in
    /// `result_or_interrupt`, so keyboard interrupts are only checked on the
    /// main thread between poll intervals (`SIGNAL_CHECK_INTERVAL`).  This is
    /// acceptable because index operations yield frequently to emit progress
    /// events.
    pub fn block_on_pumping<F, P>(
        &self,
        py: Option<Python<'_>>,
        future: F,
        mut pump: P,
    ) -> PyResult<F::Output>
    where
        F: Future + Send,
        F::Output: Send,
        P: FnMut() -> PyResult<()>,
    {
        let should_propagate_on_completion = is_python314_or_later(py);

        let mut future = std::pin::pin!(future);

        loop {
            pump()?;

            let signal_check = match Python::try_attach(|py| py.check_signals()) {
                Some(result) => result,
                None => Ok(()),
            };
            signal_check?;

            let maybe_output = if let Some(py) = py {
                py.detach(|| {
                    self.runtime.block_on(async {
                        tokio::select! {
                            result = &mut future => Some(result),
                            _ = tokio::time::sleep(SIGNAL_CHECK_INTERVAL) => None,
                        }
                    })
                })
            } else if let Some(result) = Python::try_attach(|py| {
                py.detach(|| {
                    self.runtime.block_on(async {
                        tokio::select! {
                            result = &mut future => Some(result),
                            _ = tokio::time::sleep(SIGNAL_CHECK_INTERVAL) => None,
                        }
                    })
                })
            }) {
                result
            } else {
                self.runtime.block_on(async {
                    tokio::select! {
                        result = &mut future => Some(result),
                        _ = tokio::time::sleep(SIGNAL_CHECK_INTERVAL) => None,
                    }
                })
            };

            if let Some(output) = maybe_output {
                // When the index build finishes so fast that no pump cycles
                // occurred during execution, pending events sit in the channel
                // buffer and get drained after completion. Python ≥ 3.14 changed
                // GIL/async scheduling timing such that callback invocations may
                // only be visible in this post-completion drain, so we propagate
                // errors for 3.14+. For ≤ 3.13 we keep the old tolerant behavior
                // to avoid spurious failures when scheduling shifts slightly.
                if should_propagate_on_completion {
                    pump()?;
                } else {
                    if let Err(err) = pump() {
                        log::warn!(
                            "Ignoring progress callback error after operation completed successfully: {}",
                            err
                        );
                    }
                }
                return Ok(output);
            }
        }
    }

    async fn result_or_interrupt<F>(future: F) -> PyResult<F::Output>
    where
        F: Future + Send,
        F::Output: Send,
    {
        let interrupt_future = async {
            loop {
                // Check for keyboard interrupts
                let signal_check = match Python::try_attach(|py| py.check_signals()) {
                    Some(result) => result,
                    // Python may be finalizing or unavailable. In this state we can't
                    // observe KeyboardInterrupt reliably, but we should not panic.
                    None => Ok(()),
                };
                match signal_check {
                    Ok(_) => {
                        // Wait for 100ms before checking signals again
                        tokio::time::sleep(SIGNAL_CHECK_INTERVAL).await;
                    }
                    Err(err) => return Err(err),
                }
            }
        };

        tokio::select! {
            result = future => Ok(result),
            err = interrupt_future => err,
        }
    }
}

impl Default for BackgroundExecutor {
    fn default() -> Self {
        Self::new()
    }
}
