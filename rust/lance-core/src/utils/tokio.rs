// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cell::UnsafeCell;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock, atomic};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::{Future, FutureExt};
use tokio::runtime::{Builder, Runtime};
use tracing::Span;

/// We cache the call to num_cpus::get() because:
///
/// 1. It shouldn't change during the lifetime of the program
/// 2. It's a relatively expensive call (requires opening several files and examining them)
static NUM_COMPUTE_INTENSIVE_CPUS: LazyLock<usize> =
    LazyLock::new(calculate_num_compute_intensive_cpus);

pub fn get_num_compute_intensive_cpus() -> usize {
    *NUM_COMPUTE_INTENSIVE_CPUS
}

fn calculate_num_compute_intensive_cpus() -> usize {
    if let Ok(raw) = std::env::var("LANCE_CPU_THREADS") {
        return parse_env_usize("LANCE_CPU_THREADS", &raw, 1).unwrap_or_else(|e| panic!("{e}"));
    }

    let cpus = num_cpus::get();

    if cpus <= *IO_CORE_RESERVATION {
        // If the user is not setting a custom value for LANCE_IO_CORE_RESERVATION then we don't emit
        // a warning because they're just on a small machine and there isn't much they can do about it.
        if cpus > 2 {
            log::warn!(
                "Number of CPUs is less than or equal to the number of IO core reservations. \
                This is not a supported configuration. using 1 CPU for compute intensive tasks."
            );
        }
        return 1;
    }

    num_cpus::get() - *IO_CORE_RESERVATION
}

/// Parse an integer environment variable, rejecting values below `min`.
///
/// The error names the variable, so a bad value is diagnosable instead of
/// surfacing as a bare `ParseIntError` or, for `LANCE_CPU_THREADS=0`, a panic
/// deep inside tokio's `max_blocking_threads`.
fn parse_env_usize(name: &str, raw: &str, min: usize) -> Result<usize, String> {
    let value: usize = raw
        .trim()
        .parse()
        .map_err(|e| format!("environment variable {name} must be an integer, got {raw:?}: {e}"))?;
    if value < min {
        return Err(format!(
            "environment variable {name} must be at least {min}, got {value}"
        ));
    }
    Ok(value)
}

/// Number of CPU cores held back for I/O and control tasks.
///
/// Overridable via the `LANCE_IO_CORE_RESERVATION` environment variable;
/// defaults to `2` when unset. `0` is allowed (reserve nothing);
/// [`get_num_compute_intensive_cpus`] subtracts this from the core count to
/// size the compute pool. A non-integer value panics on first access with an
/// error naming the variable.
pub static IO_CORE_RESERVATION: LazyLock<usize> =
    LazyLock::new(|| match std::env::var("LANCE_IO_CORE_RESERVATION") {
        Ok(raw) => {
            parse_env_usize("LANCE_IO_CORE_RESERVATION", &raw, 0).unwrap_or_else(|e| panic!("{e}"))
        }
        Err(_) => 2,
    });

fn create_runtime() -> Runtime {
    Builder::new_multi_thread()
        .thread_name("lance-cpu")
        .max_blocking_threads(get_num_compute_intensive_cpus())
        .worker_threads(1)
        // keep the thread alive "forever"
        .thread_keep_alive(Duration::from_secs(u64::MAX))
        .build()
        .unwrap()
}

static CPU_RUNTIME: atomic::AtomicPtr<Runtime> = atomic::AtomicPtr::new(std::ptr::null_mut());

static RUNTIME_INSTALLED: atomic::AtomicBool = atomic::AtomicBool::new(false);

static ATFORK_INSTALLED: atomic::AtomicBool = atomic::AtomicBool::new(false);

fn global_cpu_runtime() -> &'static Runtime {
    loop {
        let ptr = CPU_RUNTIME.load(Ordering::SeqCst);
        if !ptr.is_null() {
            // SAFETY: `ptr` was produced by `Box::into_raw` below and is only ever
            // reset to null by `atfork_tokio_child` in the forked child (single-
            // threaded, async-signal context). The `Box` is never reclaimed, so the
            // `Runtime` lives for the rest of the process.
            return unsafe { &*ptr };
        }
        if !RUNTIME_INSTALLED.fetch_or(true, Ordering::SeqCst) {
            break;
        }
        std::thread::yield_now();
    }
    if !ATFORK_INSTALLED.fetch_or(true, Ordering::SeqCst) {
        install_atfork();
    }
    let new_ptr = Box::into_raw(Box::new(create_runtime()));
    CPU_RUNTIME.store(new_ptr, Ordering::SeqCst);
    // SAFETY: `new_ptr` was just obtained from `Box::into_raw`, so it is non-null,
    // aligned, and points to a live `Runtime` that is never reclaimed.
    unsafe { &*new_ptr }
}

/// After a fork() operation, force re-creation of the BackgroundExecutor. Note: this function
/// runs in "async-signal context" which means that we can't (safely) do much here.
extern "C" fn atfork_tokio_child() {
    CPU_RUNTIME.store(std::ptr::null_mut(), Ordering::SeqCst);
    RUNTIME_INSTALLED.store(false, Ordering::SeqCst);
}

#[cfg(not(windows))]
fn install_atfork() {
    unsafe { libc::pthread_atfork(None, None, Some(atfork_tokio_child)) };
}

#[cfg(windows)]
fn install_atfork() {}

/// Spawn a future onto the current tokio runtime, like [`tokio::spawn`].
///
/// Prefer this over [`tokio::spawn`] in library code. `tokio::spawn` instantiates
/// tokio's task machinery (task cell, harness, vtable, `catch_unwind`, drop glue,
/// about 1.7K LLVM IR lines) once per future type, and every `async` block is its
/// own type. This hands tokio a single non-generic future type instead, so that
/// machinery is instantiated once for the whole program. The future and its
/// output share one extra allocation.
///
/// The returned handle mirrors [`tokio::task::JoinHandle`]: awaiting it yields
/// the task's output or a [`tokio::task::JoinError`], and it offers the same
/// `abort`/`abort_handle`/`is_finished` controls.
///
/// ```
/// # use lance_core::utils::tokio::spawn;
/// # async fn example() -> Result<(), tokio::task::JoinError> {
/// let output = spawn(async { 42 }).await?;
/// assert_eq!(output, 42);
/// # Ok(())
/// # }
/// ```
#[track_caller]
pub fn spawn<T: Send + 'static>(future: impl Future<Output = T> + Send + 'static) -> JoinHandle<T> {
    let task = Arc::new(SpawnedTask {
        stage: UnsafeCell::new(Stage::Running(future)),
    });
    JoinHandle {
        handle: spawn_erased_task(ErasedTask(task.clone())),
        task,
    }
}

/// Keep tokio's generic `spawn` instantiated for one concrete future type, in
/// this one place, instead of at every `spawn` call site.
#[track_caller]
fn spawn_erased_task(task: ErasedTask) -> tokio::task::JoinHandle<()> {
    tokio::spawn(task)
}

/// Handle to a task started with [`spawn`]. Awaiting it behaves like awaiting a
/// [`tokio::task::JoinHandle`].
pub struct JoinHandle<T> {
    handle: tokio::task::JoinHandle<()>,
    task: Arc<dyn TakeOutput<T>>,
}

impl<T> JoinHandle<T> {
    /// See [`tokio::task::JoinHandle::abort`].
    pub fn abort(&self) {
        self.handle.abort();
    }

    /// See [`tokio::task::JoinHandle::is_finished`].
    pub fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    /// See [`tokio::task::JoinHandle::abort_handle`].
    pub fn abort_handle(&self) -> tokio::task::AbortHandle {
        self.handle.abort_handle()
    }
}

impl<T> std::fmt::Debug for JoinHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.handle.fmt(f)
    }
}

impl<T> Future for JoinHandle<T> {
    type Output = Result<T, tokio::task::JoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.handle.poll_unpin(cx).map_ok(|()| {
            // SAFETY: `Ok` means tokio dropped the `ErasedTask` before it
            // reported completion (see `SpawnedTask`), so this is the only
            // remaining access to the stage.
            unsafe { self.task.take_output() }
                .expect("spawned task completed without storing its output")
        })
    }
}

/// A spawned future and, once it completes, its output.
///
/// The stage is shared between the task and its [`JoinHandle`] without locks.
/// Access is sequenced by the task lifecycle rather than synchronized: while the
/// task is alive, only tokio touches the stage, through the task's exclusive
/// `ErasedTask` (polled or dropped by one thread at a time).
///
/// The handle reads the stage only after its [`tokio::task::JoinHandle`] resolved
/// with `Ok`, and tokio drops the future before it reports that: the poll that
/// returns `Ready` drops the `ErasedTask`, the task then transitions to complete
/// with a release store, and the handle reads the output only after an acquire
/// load of that state. So both the poll that wrote the stage and the drop that
/// may touch it again happen before `take_output`.
struct SpawnedTask<F: Future> {
    stage: UnsafeCell<Stage<F, F::Output>>,
}

enum Stage<F, T> {
    Running(F),
    Finished(T),
    Consumed,
}

// SAFETY: the stage is never accessed concurrently (see `SpawnedTask`), so
// sharing it across threads only requires its contents to be `Send`.
unsafe impl<F: Future + Send> Send for SpawnedTask<F> where F::Output: Send {}
unsafe impl<F: Future + Send> Sync for SpawnedTask<F> where F::Output: Send {}

trait PollErased: Send + Sync {
    /// # Safety
    ///
    /// Must only be called by the owning `ErasedTask`.
    unsafe fn poll_erased(&self, cx: &mut Context<'_>) -> Poll<()>;

    /// # Safety
    ///
    /// Must only be called by the owning `ErasedTask`.
    unsafe fn drop_future(&self);
}

trait TakeOutput<T>: Send + Sync {
    /// # Safety
    ///
    /// Must only be called after the task's `ErasedTask` has been dropped.
    unsafe fn take_output(&self) -> Option<T>;
}

impl<F: Future + Send> PollErased for SpawnedTask<F>
where
    F::Output: Send,
{
    unsafe fn poll_erased(&self, cx: &mut Context<'_>) -> Poll<()> {
        // SAFETY: the caller has exclusive access to the stage.
        let stage = unsafe { &mut *self.stage.get() };
        let Stage::Running(future) = stage else {
            panic!("spawned task polled after completion");
        };
        // SAFETY: the future lives inside an `Arc` allocation and is only ever
        // dropped in place, never moved, so it stays pinned.
        let output = std::task::ready!(unsafe { Pin::new_unchecked(future) }.poll(cx));
        *stage = Stage::Finished(output);
        Poll::Ready(())
    }

    unsafe fn drop_future(&self) {
        // SAFETY: the caller has exclusive access to the stage.
        let stage = unsafe { &mut *self.stage.get() };
        if matches!(stage, Stage::Running(_)) {
            // Drop an unfinished future now, as tokio would on abort or
            // shutdown, rather than when the last handle goes away.
            *stage = Stage::Consumed;
        }
    }
}

impl<F: Future + Send> TakeOutput<F::Output> for SpawnedTask<F>
where
    F::Output: Send,
{
    unsafe fn take_output(&self) -> Option<F::Output> {
        // SAFETY: the caller guarantees the task side is gone.
        match std::mem::replace(unsafe { &mut *self.stage.get() }, Stage::Consumed) {
            Stage::Finished(output) => Some(output),
            _ => None,
        }
    }
}

/// The single future type handed to `tokio::spawn`.
struct ErasedTask(Arc<dyn PollErased>);

impl Future for ErasedTask {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        // SAFETY: `ErasedTask` is the task side's only access to the stage, and
        // tokio polls a task from one thread at a time.
        unsafe { self.0.poll_erased(cx) }
    }
}

impl Drop for ErasedTask {
    fn drop(&mut self) {
        // SAFETY: as in `poll`; the handle does not read the stage until this
        // drop has happened.
        unsafe { self.0.drop_future() }
    }
}

/// Spawn a CPU intensive task
///
/// This task will be put onto a thread pool dedicated for CPU-intensive work
/// This keeps the tokio thread pool free so that we can always be ready to service
/// cheap I/O & control requests.
///
/// This can also be used to convert a big chunk of synchronous work into a future
/// so that it can be run in parallel with something like StreamExt::buffered()
///
/// # Only hand over substantial CPU work
///
/// Dispatching to the pool has real overhead (a `spawn_blocking` hop plus a oneshot
/// channel round trip). As a rule of thumb the closure should be expected to do at
/// least ~100µs of CPU work; below that the thread-pool overhead is likely to
/// outweigh any parallelism benefit, and the work is better left inline.
///
/// # The task must never wait on anything
///
/// The CPU pool is sized to [`get_num_compute_intensive_cpus`], which is
/// `max(1, num_cpus - LANCE_IO_CORE_RESERVATION)`. On a big host that is plenty of
/// workers (e.g. 62 on a 64-core box), but in resource-constrained environments it can
/// collapse to a **single blocking thread** — on machines with `<= 3` visible CPUs
/// (1-vCPU VMs, CI runners, CPU-limited Kubernetes pods) the pool has exactly one
/// worker. A closure passed to `spawn_cpu` occupies one of these threads for its entire
/// lifetime, including any time it spends *parked*. So the closure must only consume
/// CPU and return; it must
/// **never** block, wait, or park. Concretely, the closure must not, directly or
/// transitively:
///
/// * **No channels** — no blocking send/recv (`send_blocking`, blocking `recv`, etc.).
///   A full/empty channel parks the thread, and whatever would drain/fill the channel
///   may need the same pool to run.
/// * **No I/O** — no file, network, or object-store reads/writes, and no disk spills.
///   I/O parks the thread while making no progress on CPU work.
/// * **No locks** — no acquiring a contended lock (or any lock that is held across an
///   `.await` elsewhere). Waiting for the lock parks the thread.
/// * **No `block_on` / `.blocking_*`** — never drive or wait on another async task
///   from inside the closure.
///
/// If any of these hold, the parked thread can starve the exact work that would
/// unblock it, deadlocking the whole pool with no timeout and no error — a silent
/// hang at 0% CPU. (See <https://github.com/lancedb/lance/pull/7423>.) When work
/// needs to wait on a channel/lock/I/O, keep the waiting in an async task and only
/// hand the pure-CPU portion to `spawn_cpu`, e.g. build each batch with `spawn_cpu`
/// and dispatch it with `tx.send(batch).await` in the surrounding async code.
pub fn spawn_cpu<
    E: std::error::Error + Send + 'static,
    F: FnOnce() -> std::result::Result<R, E> + Send + 'static,
    R: Send + 'static,
>(
    func: F,
) -> impl Future<Output = std::result::Result<R, E>> {
    // The closure and its output share one allocation, and only the non-generic
    // `spawn_erased` touches tokio, so tokio's blocking-task machinery (task
    // cell, harness, vtable, drop glue) is instantiated once rather than once
    // per closure type in every calling crate.
    let task = Arc::new(CpuTask {
        func: UnsafeCell::new(Some(func)),
        output: UnsafeCell::new(None),
    });
    spawn_erased(task.clone()).map(move |res| match res {
        // SAFETY: the join handle resolved, so `run` has returned on the CPU
        // thread and tokio's task-completion handoff orders its write to
        // `output` before this read. Nothing else touches `output`.
        Ok(()) => unsafe { (*task.output.get()).take() }
            .expect("spawn_cpu task completed without storing its output"),
        // Awaited through the join handle, not a result channel: a panic in
        // `func` arrives as a `JoinError` still carrying its payload, so
        // resuming it re-raises the original panic in the caller.
        Err(join_error) => match join_error.try_into_panic() {
            Ok(panic) => std::panic::resume_unwind(panic),
            // The CPU runtime outlives every caller, so its tasks are not
            // cancelled out from under one.
            Err(join_error) => panic!("spawn_cpu task failed: {join_error}"),
        },
    })
}

/// A closure and the slot for its output, shared between the caller and the
/// CPU thread without locks. Access is sequenced by the task lifecycle rather
/// than synchronized per field: `func` is only touched by `run`, and `output`
/// is written by `run` and read by the caller only after the task completes.
struct CpuTask<F, T> {
    func: UnsafeCell<Option<F>>,
    output: UnsafeCell<Option<T>>,
}

// SAFETY: the fields are never accessed concurrently (see `CpuTask`), so
// sharing across threads only requires the contents to be `Send`.
unsafe impl<F: Send, T: Send> Sync for CpuTask<F, T> {}

trait RunOnce: Send + Sync {
    /// # Safety
    ///
    /// Must be called at most once, and `output` must not be read until it
    /// returns.
    unsafe fn run(&self);
}

impl<F: FnOnce() -> T + Send, T: Send> RunOnce for CpuTask<F, T> {
    unsafe fn run(&self) {
        // SAFETY: upheld by the caller of `run`; see `RunOnce::run`.
        let func = unsafe { (*self.func.get()).take() }.expect("spawn_cpu task runs once");
        let output = func();
        unsafe { *self.output.get() = Some(output) };
    }
}

fn spawn_erased(task: Arc<dyn RunOnce>) -> tokio::task::JoinHandle<()> {
    // Propagate the current span into the task
    let span = Span::current();
    global_cpu_runtime().spawn_blocking(move || {
        let _span_guard = span.enter();
        // SAFETY: tokio runs a blocking task exactly once, and the caller in
        // `spawn_cpu` reads `output` only after this task's join handle resolves.
        unsafe { task.run() };
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The successful path must hand back the closure's own output.
    ///
    /// The output travels through an erased, type-independent task, so this
    /// pins the typed contract: the caller receives the exact value the
    /// closure returned, moved out of the task rather than re-created.
    #[tokio::test]
    async fn spawn_cpu_returns_the_closure_output() {
        let result = spawn_cpu(|| -> std::result::Result<(u32, String), std::io::Error> {
            Ok((7, String::from("done")))
        })
        .await;

        assert_eq!(
            result.expect("the closure succeeded"),
            (7, "done".to_string())
        );
    }

    /// A panic in the closure must reach the caller intact.
    ///
    /// Reporting the closure's outcome over a channel loses it: the sender
    /// drops unsent and the caller can only see an opaque receive error, so
    /// every panic in every `spawn_cpu` closure looks the same.
    #[tokio::test]
    async fn spawn_cpu_reraises_the_closure_panic() {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let joined = tokio::spawn(async {
            spawn_cpu(|| -> std::result::Result<(), std::io::Error> {
                panic!("the original message")
            })
            .await
        })
        .await;
        std::panic::set_hook(hook);

        let payload = joined
            .expect_err("the closure's panic propagates to the caller")
            .into_panic();
        let message = payload
            .downcast_ref::<&str>()
            .copied()
            .expect("the original payload survives");
        assert_eq!(message, "the original message");
    }

    #[tokio::test]
    async fn spawn_returns_the_future_output() {
        let output = spawn(async { String::from("done") }).await.unwrap();
        assert_eq!(output, "done");
    }

    /// The stage has to survive across polls: a future that is pending first
    /// and ready later still hands back its output.
    #[tokio::test]
    async fn spawn_returns_the_output_after_a_pending_poll() {
        let output = spawn(async {
            tokio::task::yield_now().await;
            String::from("done")
        })
        .await
        .unwrap();
        assert_eq!(output, "done");
    }

    #[tokio::test]
    async fn spawn_reports_a_panic_with_its_payload() {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = spawn(async { panic!("the original message") }).await;
        std::panic::set_hook(hook);

        let payload = result.expect_err("the panic is reported").into_panic();
        assert_eq!(
            payload.downcast_ref::<&str>().copied(),
            Some("the original message")
        );
    }

    /// Aborting must drop the in-flight future right away, as `tokio::spawn`
    /// does, even though the handle keeps the shared allocation alive.
    #[tokio::test]
    async fn spawn_abort_drops_the_future_before_the_handle() {
        struct SetOnDrop(Arc<atomic::AtomicBool>);
        impl Drop for SetOnDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let dropped = Arc::new(atomic::AtomicBool::new(false));
        let guard = SetOnDrop(dropped.clone());
        let mut handle = spawn(async move {
            let _guard = guard;
            futures::future::pending::<()>().await;
        });
        handle.abort();
        let err = (&mut handle)
            .await
            .expect_err("an aborted task is cancelled");
        assert!(err.is_cancelled());
        assert!(dropped.load(Ordering::SeqCst));
        drop(handle);
    }

    // The env vars feed process-global `LazyLock`s that read once and are read
    // in parallel by other tests, so the pure parser is tested directly rather
    // than by mutating the environment.

    #[test]
    fn parses_valid_value_and_trims_surrounding_whitespace() {
        assert_eq!(parse_env_usize("VAR", "8", 1).unwrap(), 8);
        assert_eq!(parse_env_usize("VAR", " 8 ", 1).unwrap(), 8);
    }

    #[test]
    fn rejects_non_integer_naming_the_variable() {
        let err = parse_env_usize("LANCE_CPU_THREADS", "abc", 1).unwrap_err();
        assert!(err.contains("LANCE_CPU_THREADS"), "{err}");
        assert!(err.contains("must be an integer"), "{err}");
    }

    #[test]
    fn rejects_value_below_minimum() {
        // LANCE_CPU_THREADS=0 parses fine but would panic in tokio's
        // max_blocking_threads(0); the minimum stops it at the boundary.
        let err = parse_env_usize("LANCE_CPU_THREADS", "0", 1).unwrap_err();
        assert!(err.contains("at least 1"), "{err}");
    }

    #[test]
    fn allows_zero_when_minimum_is_zero() {
        // LANCE_IO_CORE_RESERVATION=0 is valid: no cores reserved for IO.
        assert_eq!(parse_env_usize("VAR", "0", 0).unwrap(), 0);
    }
}
