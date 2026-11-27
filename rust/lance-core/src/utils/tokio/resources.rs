// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Experimental, unstable cross-crate resource API.
//!
//! Task ownership for cooperative memory admission.

use std::convert::Infallible;
use std::mem::{size_of, size_of_val};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;

use futures::{FutureExt, future::BoxFuture};
use tracing::Instrument;

use crate::{
    Error, Result,
    utils::memory::{MemoryLease, MemoryReserver},
};

use super::{global_cpu_runtime, spawn_cpu};

/// Extra control space per pending task, beyond closure and future sizes. Covers
/// runtime task handles, pool registration, and reservation owners inside the
/// operation's pool rather than as an unbounded per-task runtime allowance.
const TASK_CONTROL_HEADROOM_BYTES: usize = 4096;
/// Inline values need pool registration and owner headers, without the runtime
/// task handles and pending-future storage reserved by dispatched work.
const INLINE_CONTROL_HEADROOM_BYTES: usize = 1024;

struct TaskMemory {
    _input_owner: Arc<dyn Send + Sync>,
    _control: Box<dyn Send + Sync>,
}

/// Memory owners transferred with a CPU task's result. Keep them until that
/// result's backing allocations and retained slices are freed or transferred to
/// another accounted owner.
#[doc(hidden)]
pub struct CpuTaskResources {
    _workspace: Box<dyn MemoryLease>,
    workspace_bytes: usize,
    reserved_bytes: usize,
    // Keep task control charged through workspace/owner destruction.
    _input_and_control: Arc<TaskMemory>,
}

/// A value and the memory reservations that accompany its backing allocations.
/// Field drop order frees the value before returning its allowance.
#[doc(hidden)]
pub struct MemoryOwned<T> {
    value: T,
    resources: CpuTaskResources,
}

impl<T> MemoryOwned<T> {
    /// Prepare a small value inline, admitting control space and its backing
    /// allocations before calling `create`. Use [`cpu_task_with_resources`] for
    /// substantial CPU work; the same ownership rules apply to both paths.
    ///
    /// `backing_bytes` must cover the creation's peak temporary and result backing
    /// allocations. Existing inputs need separate owners if the result retains
    /// them. The callback should do short CPU work without I/O or waits.
    ///
    /// ```
    /// # use lance_core::{Result, utils::{memory::MemoryReserver, tokio::MemoryOwned}};
    /// fn allocate(pool: &dyn MemoryReserver) -> Result<MemoryOwned<Vec<u8>>> {
    ///     MemoryOwned::try_new(pool, "index metadata", 128, || Ok(vec![0_u8; 128]))
    /// }
    /// ```
    #[doc(hidden)]
    pub fn try_new(
        pool: &dyn MemoryReserver,
        stage: &str,
        backing_bytes: usize,
        create: impl FnOnce() -> Result<T>,
    ) -> Result<Self> {
        let control_bytes = size_of::<T>()
            .checked_add(INLINE_CONTROL_HEADROOM_BYTES)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "inline allocation control size overflow: stage={stage}, value_bytes={}",
                    size_of::<T>(),
                ))
            })?;
        let reserved_bytes = control_bytes.checked_add(backing_bytes).ok_or_else(|| {
            Error::invalid_input(format!("inline allocation total size overflow: control_bytes={control_bytes}, backing_bytes={backing_bytes}"))
        })?;
        let control = pool.reserve("inline allocation control", control_bytes)?;
        let workspace = pool.reserve_lease(stage, backing_bytes)?;
        let value = create()?;
        Ok(Self {
            value,
            resources: CpuTaskResources {
                _input_and_control: Arc::new(TaskMemory {
                    _input_owner: Arc::new(()),
                    _control: control,
                }),
                _workspace: workspace,
                workspace_bytes: backing_bytes,
                reserved_bytes,
            },
        })
    }

    /// Fill a pre-admitted output on the CPU pool, retaining its reservations
    /// outside the callback through errors, panics, and cancelled awaits.
    ///
    /// `temporary_bytes` admits additional workspace that can be freed when the
    /// callback finishes. The returned value keeps this output's original
    /// workspace and control, and releases temporary/input owners on
    /// the async driver. The output's inline envelope cannot grow. The initial
    /// value and callback captures must contain pure CPU data; retain native
    /// I/O and reservation owners through `input_owner` instead.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance_core::{Result, utils::{memory::MemoryReserver, tokio::MemoryOwned}};
    /// # async fn example(pool: Arc<dyn MemoryReserver>) -> Result<()> {
    /// let output = MemoryOwned::<Option<Vec<u8>>>::try_new(
    ///     pool.as_ref(), "copied output", 1024, || Ok(None))?;
    /// let output = output.into_cpu_task(pool, "copy", 0, Arc::new(()),
    ///     |_| Ok(vec![7_u8; 1024]))?.await?;
    /// assert_eq!(output.value().len(), 1024);
    /// # Ok(())
    /// # }
    /// ```
    #[doc(hidden)]
    pub fn into_cpu_task<U, F>(
        self,
        pool: Arc<dyn MemoryReserver>,
        stage: &'static str,
        temporary_bytes: usize,
        input_owner: Arc<dyn Send + Sync>,
        work: F,
    ) -> Result<ResourceCpuTask<U>>
    where
        T: Send + 'static,
        U: Send + 'static,
        F: FnOnce(T) -> Result<U> + Send + 'static,
    {
        if size_of::<MemoryOwned<U>>() > size_of::<Self>() {
            return Err(Error::invalid_input(format!(
                "memory-owned CPU conversion increases inline size: input_bytes={}, output_bytes={}",
                size_of::<Self>(),
                size_of::<MemoryOwned<U>>(),
            )));
        }
        let (value, resources) = self.into_parts();
        // The existing inline-control allowance covers these two small Arc
        // headers. Task admission separately covers closure/future storage.
        let resources = Arc::new(resources);
        let owners = Arc::new((input_owner, resources.clone()));
        let task =
            cpu_task_with_resources(pool, stage, temporary_bytes, owners, move || work(value))?;
        Ok(Box::pin(async move {
            let result = task.await?;
            let (value, temporary) = result.into_parts();
            drop(temporary);
            MemoryOwned::from_shared_resources(value, resources)
        }))
    }

    fn from_shared_resources(value: T, resources: Arc<CpuTaskResources>) -> Result<Self> {
        match Arc::try_unwrap(resources) {
            Ok(resources) => Ok(Self { value, resources }),
            Err(resources) => {
                drop(value);
                drop(resources);
                Err(Error::internal(
                    "completed CPU output still has temporary reservation owners",
                ))
            }
        }
    }

    /// This value's control and workspace reservations, excluding separately
    /// retained input/schema/model owners. Count an owner only once when sharing
    /// it with `Arc`. This measures reserved allowance, not physical allocation.
    ///
    /// ```
    /// # use lance_core::utils::tokio::MemoryOwned;
    /// fn retained_allowance<T>(value: &MemoryOwned<T>) -> usize {
    ///     value.reserved_bytes()
    /// }
    /// ```
    #[doc(hidden)]
    pub fn reserved_bytes(&self) -> usize {
        self.resources.reserved_bytes
    }

    /// Release temporary workspace in the async driver after CPU completion.
    /// `backing_bytes` must cover every backing allocation and slice still
    /// dependent on this owner. Control/input/schema owners remain charged.
    /// Return `false` if the backend retains the larger original allowance.
    /// Growing is rejected without changing the value or its reservations.
    ///
    /// ```
    /// # use lance_core::{Result, utils::tokio::MemoryOwned};
    /// fn retain(mut value: MemoryOwned<Vec<u8>>) -> Result<MemoryOwned<Vec<u8>>> {
    ///     value.try_shrink_workspace(value.value().capacity())?;
    ///     Ok(value)
    /// }
    /// ```
    #[doc(hidden)]
    pub fn try_shrink_workspace(&mut self, backing_bytes: usize) -> Result<bool> {
        let released = self.resources.workspace_bytes.checked_sub(backing_bytes)
            .ok_or_else(|| Error::invalid_input(format!(
                "memory-owned workspace cannot grow: requested_bytes={backing_bytes}, reserved_bytes={}",
                self.resources.workspace_bytes,
            )))?;
        let reserved_bytes = self
            .resources
            .reserved_bytes
            .checked_sub(released)
            .ok_or_else(|| Error::internal("memory-owned reservation accounting underflow"))?;
        if self.resources._workspace.try_shrink(backing_bytes)? {
            self.resources.workspace_bytes = backing_bytes;
            self.resources.reserved_bytes = reserved_bytes;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Borrow the result while retaining all of its reservations. Any derived
    /// slice or clone must keep this owner alive; use [`Self::into_parts`] when
    /// transferring backing allocations to another consumer.
    #[doc(hidden)]
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Transfer to a consumer that keeps `resources` until `value`'s backing
    /// allocations and any retained slices are freed or transferred.
    ///
    /// ```
    /// # use lance_core::utils::tokio::MemoryOwned;
    /// fn consume(result: MemoryOwned<Vec<u8>>) {
    ///     let (bytes, resources) = result.into_parts();
    ///     drop(bytes);
    ///     drop(resources);
    /// }
    /// ```
    #[doc(hidden)]
    pub fn into_parts(self) -> (T, CpuTaskResources) {
        (self.value, self.resources)
    }
}

/// A lazy CPU task whose result retains its accounting owners.
#[doc(hidden)]
pub type ResourceCpuTask<T> = BoxFuture<'static, Result<MemoryOwned<T>>>;

struct ChargedWork<F> {
    // Also preserve ownership if a queued CPU closure is dropped before running.
    work: F,
    resources: CpuTaskResources,
}

struct CpuTaskOutcome<T> {
    value: std::thread::Result<Result<T>>,
    resources: CpuTaskResources,
}

impl<T, F: FnOnce() -> Result<T>> ChargedWork<F> {
    fn execute(self) -> CpuTaskOutcome<T> {
        // Hand owners back even on a fallible result or panic. Pool release and
        // panic propagation belong to the surrounding async task, after the
        // CPU closure has returned. Cancellation also leaves this result owned.
        let value = catch_unwind(AssertUnwindSafe(self.work));
        CpuTaskOutcome {
            value,
            resources: self.resources,
        }
    }
}

struct PendingWork<F> {
    work: F,
    pool: Arc<dyn MemoryReserver>,
    stage: &'static str,
    workspace_bytes: usize,
    reserved_bytes: usize,
    memory: Arc<TaskMemory>,
}

impl<T: Send + 'static, F: FnOnce() -> Result<T> + Send + 'static> PendingWork<F> {
    async fn run(self) -> Result<MemoryOwned<T>> {
        let group = self.pool.task_group();
        let running = group
            .as_ref()
            .map(|group| group.start(self.stage))
            .transpose()?;
        // Acquire outside the CPU closure. No worker waits for memory or for
        // another pipeline stage that might require the same worker to run.
        let workspace = self.pool.reserve_lease(self.stage, self.workspace_bytes)?;
        let charged = ChargedWork {
            work: self.work,
            resources: CpuTaskResources {
                _input_and_control: self.memory,
                _workspace: workspace,
                workspace_bytes: self.workspace_bytes,
                reserved_bytes: self.reserved_bytes,
            },
        };
        // A detached blocking task drops an unclaimed result on its worker.
        // Keep an independent async driver awaiting that result even when this
        // caller is cancelled or its runtime shuts down. Native reservations
        // and retained I/O owners are consequently released outside the CPU
        // closure and its blocking-task completion path.
        let (completed, outcome) = tokio::sync::oneshot::channel();
        let driver = async move {
            let outcome = match spawn_cpu(move || {
                let outcome = if group.as_ref().is_some_and(|group| group.is_cancelled()) {
                    // Do not enter queued native work after its operation stops.
                    // Its owners still return to the async completion driver.
                    CpuTaskOutcome {
                        value: Ok(Err(Error::execution(
                            "resource CPU task cancelled before execution",
                        ))),
                        resources: charged.resources,
                    }
                } else {
                    charged.execute()
                };
                Ok::<_, Infallible>(outcome)
            })
            .await
            {
                Ok(outcome) => outcome,
                Err(never) => match never {},
            };
            // A cancelled receiver drops its unclaimed backing here, before
            // announcing completion. The channel is only used by this async
            // driver; the pure CPU closure never sends or waits.
            let _ = completed.send(outcome);
        }
        .instrument(tracing::Span::current());
        // Charge the concrete completion future separately before Tokio stores it.
        // The fixed surcharge covers opaque runtime/channel bookkeeping.
        let driver_control = self
            .pool
            .reserve("CPU completion future", size_of_val(&driver))?;
        let driver = async move {
            // Task completion includes disposition and completion-only control.
            let completion = (driver_control, running);
            driver.await;
            drop(completion);
        };
        global_cpu_runtime().spawn(driver).await.map_err(|source| {
            Error::internal(format!(
                "CPU completion driver failed: stage={}, {source}",
                self.stage
            ))
        })?;
        let outcome = outcome.await.map_err(|source| {
            Error::internal(format!(
                "CPU completion driver lost its outcome: stage={}, {source}",
                self.stage
            ))
        })?;
        let CpuTaskOutcome { value, resources } = outcome;
        let value = match value {
            Ok(result) => result?,
            Err(panic) => resume_unwind(panic),
        };
        Ok(MemoryOwned { value, resources })
    }
}

/// Prepare pure CPU work under an operation's memory pool, dispatching only when
/// polled. Admit control memory before boxing the future and workspace before
/// spawning work. Ownership survives a cancelled await and follows pending output.
///
/// `workspace_bytes` must cover audited peak temporary and result backing
/// allocations; inline task and result layout is charged separately.
/// `input_owner` must already own charges for all retained inputs; share it when
/// several tasks reference one backing. Poll tasks sequentially
/// when the operation leaves workspace for only one result at a time.
///
/// Like [`spawn_cpu`], `work` must never do I/O, wait, lock, or use channels.
///
/// ```
/// # use std::sync::Arc;
/// # use lance_core::{Result, utils::{memory::MemoryReserver, tokio::cpu_task_with_resources}};
/// # async fn example(pool: Arc<dyn MemoryReserver>) -> Result<()> {
/// let task = cpu_task_with_resources(pool, "encoded page", 1024, Arc::new(()),
///     || Ok(vec![0_u8; 1024]))?;
/// let result = task.await?;
/// assert_eq!(result.value().len(), 1024);
/// # Ok(())
/// # }
/// ```
#[doc(hidden)]
pub fn cpu_task_with_resources<T, F>(
    pool: Arc<dyn MemoryReserver>,
    stage: &'static str,
    workspace_bytes: usize,
    input_owner: Arc<dyn Send + Sync>,
    work: F,
) -> Result<ResourceCpuTask<T>>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    // Pending futures, the independent completion driver, blocking-task storage
    // and their inline results can coexist. Count their control allocations and
    // consumer names before allocating any of those task objects.
    // Account the concrete submitted-work and result envelopes; the lazy
    // future and completion driver receive their own measured reservations.
    let control_bytes = size_of::<ChargedWork<F>>()
        .checked_add(size_of::<CpuTaskOutcome<T>>())
        // The native task result and completion channel can coexist.
        .and_then(|bytes| bytes.checked_add(size_of::<std::thread::Result<Result<T>>>()))
        .and_then(|bytes| bytes.checked_add(stage.len()))
        .and_then(|bytes| bytes.checked_add(TASK_CONTROL_HEADROOM_BYTES))
        .ok_or_else(|| Error::invalid_input("CPU task control size overflow"))?;
    let reserved_bytes = control_bytes.checked_add(workspace_bytes).ok_or_else(|| {
        Error::invalid_input(format!("CPU task total size overflow: stage={stage}, control_bytes={control_bytes}, workspace_bytes={workspace_bytes}"))
    })?;
    let control = pool.reserve("CPU task control", control_bytes)?;
    let memory = Arc::new(TaskMemory {
        _input_owner: input_owner,
        _control: control,
    });
    let pool_for_future = pool.clone();
    let future = PendingWork {
        work,
        pool,
        stage,
        workspace_bytes,
        reserved_bytes,
        memory,
    }
    .run();
    let future_control = pool_for_future.reserve("CPU task control", size_of_val(&future))?;
    Ok(async move {
        let result = future.await;
        drop(future_control);
        result
    }
    .boxed())
}

#[cfg(test)]
mod tests {
    use super::super::tests::CpuTestPool;
    use super::*;
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    struct DropProbe {
        other_owner: Arc<Mutex<Option<Arc<CpuTaskResources>>>>,
        used: Arc<AtomicUsize>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            // Simulate another task releasing the last competing owner while
            // this value's destructor still needs its admitted workspace.
            drop(self.other_owner.lock().unwrap().take());
            assert!(self.used.load(Ordering::Relaxed) > 0);
        }
    }

    #[test]
    fn shared_cpu_result_drops_value_before_error_owner() {
        let pool = CpuTestPool {
            used: Arc::default(),
            limit: 16 << 10,
        };
        let other_owner = Arc::new(Mutex::new(None));
        let output = MemoryOwned::try_new(&pool, "test output", 1024, || {
            Ok(DropProbe {
                other_owner: other_owner.clone(),
                used: pool.used.clone(),
            })
        })
        .unwrap();
        let (value, resources) = output.into_parts();
        let resources = Arc::new(resources);
        *other_owner.lock().unwrap() = Some(resources.clone());
        let error = MemoryOwned::from_shared_resources(value, resources)
            .err()
            .unwrap();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(error.to_string().contains("temporary reservation owners"));
        assert_eq!(pool.used.load(Ordering::Relaxed), 0);
    }
}
