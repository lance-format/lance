// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cancellation and completion of actual resource-owned work.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use tokio::sync::Notify;

use crate::{Error, Result};

const CANCELLED: usize = 1 << (usize::BITS - 1);
const ACTIVE: usize = CANCELLED - 1;

/// One operation's actual CPU/I/O tasks, independently of their public awaits.
///
/// A cancelled group rejects new work. Keep its task guards in async completion
/// drivers or I/O workers, never inside pure CPU closures: completion can wake
/// async waiters. Cancelling cannot interrupt an already running native call.
///
/// ```
/// # use std::sync::Arc;
/// # use lance_core::utils::tasks::ResourceTaskGroup;
/// let tasks = Arc::new(ResourceTaskGroup::default());
/// let running = tasks.start("encoding")?;
/// tasks.cancel();
/// assert!(tasks.is_cancelled());
/// drop(running);
/// # Ok::<(), lance_core::Error>(())
/// ```
#[derive(Debug, Default)]
pub struct ResourceTaskGroup {
    state: AtomicUsize,
    peak: AtomicUsize,
    completed: Notify,
}

impl ResourceTaskGroup {
    /// Register before dispatch, atomically refusing work after cancellation.
    pub fn start(self: &Arc<Self>, stage: &str) -> Result<ResourceTask> {
        let previous = self
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                (state & CANCELLED == 0 && state & ACTIVE != ACTIVE).then(|| state + 1)
            })
            .map_err(|state| {
                if state & CANCELLED != 0 {
                    Error::execution(format!(
                        "resource task cancelled before dispatch: stage={stage}"
                    ))
                } else {
                    Error::internal(format!("resource task count overflow: stage={stage}"))
                }
            })?;
        self.peak
            .fetch_max((previous & ACTIVE) + 1, Ordering::Relaxed);
        Ok(ResourceTask {
            group: self.clone(),
        })
    }

    /// Signal cooperative cancellation and prevent subsequent task admission.
    pub fn cancel(&self) {
        self.state.fetch_or(CANCELLED, Ordering::AcqRel);
        self.completed.notify_waiters();
    }

    /// Read the cancellation signal without locking or waiting.
    pub fn is_cancelled(&self) -> bool {
        self.state.load(Ordering::Acquire) & CANCELLED != 0
    }

    /// Number of dispatched tasks whose actual completion is still pending.
    pub fn active(&self) -> usize {
        self.state.load(Ordering::Acquire) & ACTIVE
    }

    /// Highest simultaneous number of dispatched tasks, including queued work.
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Acquire).max(self.active())
    }

    /// Await actual completion after signalling cancellation and dropping the
    /// pipeline futures. This does not wait for ordinary model/data ownership.
    pub async fn wait(&self) {
        loop {
            let completed = self.completed.notified();
            tokio::pin!(completed);
            completed.as_mut().enable();
            if self.active() == 0 {
                return;
            }
            completed.await;
        }
    }
}

/// Completion ownership for one task. Drop only after its actual work stops.
pub struct ResourceTask {
    group: Arc<ResourceTaskGroup>,
}

impl Drop for ResourceTask {
    fn drop(&mut self) {
        let previous = self.group.state.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous & ACTIVE != 0, "resource task count underflow");
        if previous & ACTIVE == 1 {
            self.group.completed.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;

    #[tokio::test]
    async fn resource_tasks_cancel_and_join_actual_completion() {
        let tasks = Arc::new(ResourceTaskGroup::default());
        let first = tasks.start("first").unwrap();
        let second = tasks.start("second").unwrap();
        assert_eq!(tasks.active(), 2);
        assert_eq!(tasks.peak(), 2);
        tasks.cancel();
        let error = tasks.start("queued").err().unwrap();
        assert!(matches!(error, Error::Execution { .. }));
        assert!(error.to_string().contains("stage=queued"));
        drop(first);
        assert_eq!(tasks.active(), 1);
        assert!(tasks.wait().now_or_never().is_none());
        drop(second);
        tasks.wait().await;
        assert_eq!(tasks.active(), 0);
        assert!(tasks.is_cancelled());
        assert_eq!(tasks.peak(), 2);
    }
}
