// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Optional, file-scoped observations of bulk copy futures.
//!
//! Attach a callback with [`CopyObserver::operation_observer`](super::CopyObserver::operation_observer).
//! Each invocation of the factory scopes a copy of one source to one destination. Join events
//! by operation ID within a process. Tokio task IDs can be absent outside spawned tasks, and
//! the read and write branches of a copy may share a task. Wakes are attributed to the waker
//! registered at poll time. A later poll can register a different waker.
//!
//! HTTP events describe the Lance native connector boundary. One operation may dispatch several
//! attempts. A response ID can be joined to service logs by the consumer. Client-side events
//! alone cannot establish server receipt. Backoff and permit events cover Lance-owned waits.
//! Credential lock events cover the AWS SDK credential adapter. External transports and
//! credential providers can correlate their own observations with [`CopyOperationId::current`].

use std::future::{Future, poll_fn};
use std::pin::pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Wake, Waker};
use std::time::Instant;

use tokio::task::Id;

/// One Lance operation. Range ends are exclusive and part indices start at zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CopyOperation {
    SourceMetadata,
    ReadRange { start: usize, end: usize },
    OpenStream,
    ReadStream { offset: usize },
    Write { offset: usize },
    Shutdown,
    SinglePut,
    CreateMultipart,
    UploadPart { index: u16, bytes: usize },
    CompleteMultipart,
    NativeCopy,
    DestinationMetadata,
}

/// A process-local identifier for joining a copy observation to lower-level diagnostics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CopyOperationId(u64);

impl CopyOperationId {
    /// The current operation while Lance polls an observed future. A transport or
    /// credential provider spawning a task can capture this ID before spawning.
    ///
    /// ```
    /// use lance_io::object_store::copy_diagnostics::CopyOperationId;
    /// let operation = CopyOperationId::current().map(CopyOperationId::get);
    /// ```
    pub fn current() -> Option<Self> {
        CURRENT.try_with(|(operation, _)| operation.id).ok()
    }

    /// Numeric identifier for structured diagnostics.
    pub fn get(self) -> u64 {
        self.0
    }
}

tokio::task_local! { static CURRENT: (Operation, CopyOperationObserver); }

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
pub(crate) fn start_http_attempt() -> Option<u64> {
    CURRENT
        .try_with(|(operation, observer)| {
            static NEXT_ATTEMPT: AtomicU64 = AtomicU64::new(1);
            let attempt = NEXT_ATTEMPT.fetch_add(1, Ordering::Relaxed);
            operation.report(
                observer,
                CopyOperationEvent::HttpDispatch { attempt },
                tokio::task::try_id(),
            );
            attempt
        })
        .ok()
}

/// Captures the operation before suspension so dropping a guard on cancellation
/// can report the resource release even outside the original task-local scope.
#[cfg(feature = "aws")]
pub(crate) fn credential_lock_acquired(cache: u64) -> impl Drop {
    struct Ownership(Option<(Operation, CopyOperationObserver)>, u64);
    impl Drop for Ownership {
        fn drop(&mut self) {
            if let Some((operation, observer)) = &self.0 {
                operation.report(
                    observer,
                    CopyOperationEvent::CredentialLockReleasing { cache: self.1 },
                    tokio::task::try_id(),
                );
            }
        }
    }
    let scope = CURRENT
        .try_with(|(operation, observer)| {
            operation.report(
                observer,
                CopyOperationEvent::CredentialLockAcquired { cache },
                tokio::task::try_id(),
            );
            (*operation, observer.clone())
        })
        .ok();
    Ownership(scope, cache)
}

pub(crate) fn record_current(event: impl FnOnce() -> CopyOperationEvent) {
    let _ = CURRENT.try_with(|(operation, observer)| {
        operation.report(observer, event(), tokio::task::try_id());
    });
}

/// Client-side future lifecycle. Polls and wakes indicate scheduling activity only.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CopyOperationEvent {
    Submitted,
    Poll,
    /// The waker supplied to this future was invoked. Other sources can also wake its task.
    Wake,
    Returned {
        success: bool,
    },
    /// The writer reaped this part's result from its join set.
    Collected,
    /// An acquisition of a Lance AIMD token began or returned.
    PermitWaiting,
    PermitAcquired,
    /// A Lance retry loop is about to sleep before its next attempt.
    Backoff {
        duration: std::time::Duration,
    },
    /// Lance handed one request to its configured native HTTP transport.
    HttpDispatch {
        attempt: u64,
    },
    /// The transport returned response headers. Only a provider request ID is retained.
    HttpResponse {
        attempt: u64,
        status: u16,
        request_id: Option<String>,
    },
    HttpError {
        attempt: u64,
    },
    /// The native multipart retry layer finished collecting this response body.
    HttpBodyReturned {
        attempt: u64,
        success: bool,
    },
    /// A mandatory refresh is acquiring the AWS adapter's serialization lock.
    CredentialLockWaiting {
        cache: u64,
    },
    CredentialLockAcquired {
        cache: u64,
    },
    /// The owner is releasing this adapter's refresh lock, including on cancellation.
    CredentialLockReleasing {
        cache: u64,
    },
    CredentialRefreshStarted {
        cache: u64,
    },
    /// The credential provider returned while the adapter owned the refresh lock.
    CredentialRefreshReturned {
        cache: u64,
        success: bool,
    },
}

/// A single event. `task` identifies the polling, submitting or collecting Tokio task
/// when available. For a wake it identifies the task that registered the waker.
#[derive(Clone, Debug)]
pub struct CopyObservation {
    pub id: CopyOperationId,
    pub operation: CopyOperation,
    pub event: CopyOperationEvent,
    pub at: Instant,
    pub task: Option<Id>,
}

/// A callback for a single source/destination pair. Invocations from independent tasks
/// can overlap. Implementations must return promptly, including on the wake path.
pub type CopyOperationObserver = Arc<dyn Fn(CopyObservation) + Send + Sync>;

#[derive(Clone, Default)]
pub(crate) struct CopyDiagnostics(pub(crate) Option<CopyOperationObserver>);

#[derive(Clone, Copy)]
pub(crate) struct Operation {
    id: CopyOperationId,
    kind: CopyOperation,
}

impl Operation {
    fn report(self, observer: &CopyOperationObserver, event: CopyOperationEvent, task: Option<Id>) {
        observer(CopyObservation {
            id: self.id,
            operation: self.kind,
            event,
            at: Instant::now(),
            task,
        });
    }
}

struct OperationWaker {
    parent: Waker,
    task: Option<Id>,
    operation: Operation,
    observer: CopyOperationObserver,
}

impl Wake for OperationWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.operation
            .report(&self.observer, CopyOperationEvent::Wake, self.task);
        self.parent.wake_by_ref();
    }
}

impl CopyDiagnostics {
    #[cfg(test)]
    pub(crate) fn recording() -> (Self, Arc<std::sync::Mutex<Vec<CopyObservation>>>) {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = events.clone();
        (
            Self(Some(Arc::new(move |event| {
                captured.lock().unwrap().push(event)
            }))),
            events,
        )
    }

    pub(crate) fn submit(&self, kind: CopyOperation) -> Option<Operation> {
        let observer = self.0.as_ref()?;
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        let operation = Operation {
            id: CopyOperationId(NEXT_ID.fetch_add(1, Ordering::Relaxed)),
            kind,
        };
        operation.report(
            observer,
            CopyOperationEvent::Submitted,
            tokio::task::try_id(),
        );
        Some(operation)
    }

    pub(crate) fn in_scope<T>(&self, operation: Option<Operation>, f: impl FnOnce() -> T) -> T {
        match (&self.0, operation) {
            (Some(observer), Some(operation)) => {
                CURRENT.sync_scope((operation, observer.clone()), f)
            }
            _ => f(),
        }
    }

    pub(crate) fn collected(&self, operation: Option<Operation>) {
        if let (Some(observer), Some(operation)) = (&self.0, operation) {
            operation.report(
                observer,
                CopyOperationEvent::Collected,
                tokio::task::try_id(),
            );
        }
    }

    pub(crate) async fn run<T, E>(
        &self,
        kind: CopyOperation,
        future: impl Future<Output = Result<T, E>>,
    ) -> Result<T, E> {
        self.run_submitted(self.submit(kind), future).await
    }

    pub(crate) async fn run_submitted<T, E>(
        &self,
        operation: Option<Operation>,
        future: impl Future<Output = Result<T, E>>,
    ) -> Result<T, E> {
        let (Some(observer), Some(operation)) = (&self.0, operation) else {
            return future.await;
        };
        let mut future = pin!(future);
        let mut registered: Option<Arc<OperationWaker>> = None;
        CURRENT
            .scope(
                (operation, observer.clone()),
                poll_fn(|cx| {
                    let task = tokio::task::try_id();
                    operation.report(observer, CopyOperationEvent::Poll, task);
                    if registered.as_ref().is_some_and(|proxy| {
                        proxy.task != task || !proxy.parent.will_wake(cx.waker())
                    }) {
                        registered = None;
                    }
                    let proxy = registered.get_or_insert_with(|| {
                        Arc::new(OperationWaker {
                            parent: cx.waker().clone(),
                            task,
                            operation,
                            observer: observer.clone(),
                        })
                    });
                    let waker = Waker::from(proxy.clone());
                    let result = future.as_mut().poll(&mut Context::from_waker(&waker));
                    if let std::task::Poll::Ready(value) = &result {
                        operation.report(
                            observer,
                            CopyOperationEvent::Returned {
                                success: value.is_ok(),
                            },
                            task,
                        );
                    }
                    result
                }),
            )
            .await
    }
}
