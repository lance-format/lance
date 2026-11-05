// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The switch that turns memory observation on, and the sampler that tracks
//! peak memory over windows of time.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, Once, PoisonError, Weak};
use std::time::Duration;

use super::proc::{ProcessMemory, read_process_memory};
use crate::utils::parse::str_to_bool;
use crate::{Error, Result};

/// Environment variable that turns memory observation on.
///
/// It takes a boolean such as `1`/`0`, `true`/`false`, `on`/`off` or
/// `yes`/`no`, in any case; unset or empty means off. Any other value fails
/// the operation that would be observed with an invalid-input error. The
/// variable is read once per process, when Lance first needs it.
pub const OBSERVE_ENV_VAR: &str = "LANCE_MEMORY_OBSERVE";

/// How often the process-wide sampler reads memory while a window is open.
const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

static OBSERVE_FLAG: LazyLock<std::result::Result<bool, String>> =
    LazyLock::new(|| match std::env::var(OBSERVE_ENV_VAR) {
        Ok(value) => parse_observe_flag(Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(value)) => Err(invalid_flag_message(&value)),
    });

fn parse_observe_flag(value: Option<&str>) -> std::result::Result<bool, String> {
    let Some(value) = value else {
        return Ok(false);
    };
    if value.trim().is_empty() {
        return Ok(false);
    }
    str_to_bool(value.trim()).ok_or_else(|| invalid_flag_message(value))
}

fn invalid_flag_message(value: &(impl fmt::Debug + ?Sized)) -> String {
    format!("{OBSERVE_ENV_VAR}={value:?} is invalid, expected a boolean such as 1/0 or true/false")
}

/// Whether [`OBSERVE_ENV_VAR`] turns observation on, read once per process.
pub(super) fn observe_flag() -> Result<bool> {
    OBSERVE_FLAG.clone().map_err(Error::invalid_input)
}

/// Logs once that the switch is on where observation cannot run.
pub(super) fn warn_unsupported() {
    static UNSUPPORTED: Once = Once::new();
    UNSUPPORTED.call_once(|| {
        tracing::warn!(
            "{OBSERVE_ENV_VAR} is set, but memory observation needs Linux and the `metrics` \
             feature; it stays off"
        );
    });
}

type ReadMemory = Box<dyn Fn() -> Option<ProcessMemory> + Send + Sync>;

/// The largest RSS read while a window was open.
#[derive(Debug)]
struct WindowState {
    max_rss_bytes: AtomicU64,
}

#[derive(Debug, Default)]
struct SamplerState {
    windows: Vec<Weak<WindowState>>,
    stopped: bool,
}

/// What the sampler thread shares with the observer that started it.
struct Shared {
    read: ReadMemory,
    state: Mutex<SamplerState>,
    wake: Condvar,
    ticks: AtomicU64,
    parked: AtomicBool,
}

impl fmt::Debug for Shared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Shared")
            .field("state", &self.state)
            .field("ticks", &self.ticks)
            .field("parked", &self.parked)
            .finish_non_exhaustive()
    }
}

impl Shared {
    fn lock_state(&self) -> MutexGuard<'_, SamplerState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Reads memory once and raises the maximum of every window open at the
    /// time. Every reading counts this way, including the ones taken when a
    /// window opens or closes, so a window is never below one nested in it.
    fn sample(&self) -> Option<ProcessMemory> {
        // The windows are collected before the read, so that a reading never
        // counts for a window opened after it, and the read happens outside
        // the lock, so that opening a window never waits behind it.
        let open: Vec<Arc<WindowState>> = {
            let mut state = self.lock_state();
            state.windows.retain(|window| window.strong_count() > 0);
            state.windows.iter().filter_map(Weak::upgrade).collect()
        };
        let memory = (self.read)();
        if let Some(memory) = memory {
            for window in &open {
                window
                    .max_rss_bytes
                    .fetch_max(memory.rss_bytes, Ordering::Relaxed);
            }
        }
        memory
    }

    fn tick(&self) {
        self.sample();
        self.ticks.fetch_add(1, Ordering::Relaxed);
    }

    fn run(&self, interval: Duration) {
        let mut state = self.lock_state();
        loop {
            if state.stopped {
                return;
            }
            state.windows.retain(|window| window.strong_count() > 0);
            if state.windows.is_empty() {
                self.parked.store(true, Ordering::Release);
                state = self
                    .wake
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
                self.parked.store(false, Ordering::Release);
                continue;
            }
            drop(state);
            self.tick();
            state = self.lock_state();
            if !state.stopped {
                state = self
                    .wake
                    .wait_timeout(state, interval)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
        }
    }
}

/// Tracks the peak resident set size (RSS) of the process over windows of
/// time, such as the stages of an index build.
///
/// One background thread samples memory while at least one window is open and
/// sleeps otherwise. It reads `/proc`, so windows report a peak on Linux only.
/// Public for the `lance` crate, which observes its index builds with it.
#[doc(hidden)]
#[derive(Debug)]
pub struct MemoryObserver {
    shared: Arc<Shared>,
    /// `None` leaves sampling to the caller, which only tests do.
    interval: Option<Duration>,
    /// Whether the sampler thread runs; a failed start is retried with the
    /// next window.
    sampler_running: Mutex<bool>,
}

impl MemoryObserver {
    /// The process-wide observer, or `None` unless [`OBSERVE_ENV_VAR`] turns
    /// observation on.
    ///
    /// Observation also needs Linux and the `metrics` feature, the only
    /// consumer so far; without them this logs a warning once and returns
    /// `None`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when [`OBSERVE_ENV_VAR`] holds something
    /// other than a boolean.
    ///
    /// ```
    /// # use lance_core::memory::MemoryObserver;
    /// if let Some(observer) = MemoryObserver::global().unwrap() {
    ///     let window = observer.open_window();
    ///     // ... the work to observe ...
    ///     let _peak_bytes = window.close();
    /// }
    /// ```
    pub fn global() -> Result<Option<Arc<Self>>> {
        if !observe_flag()? {
            return Ok(None);
        }
        if !cfg!(all(feature = "metrics", target_os = "linux")) {
            warn_unsupported();
            return Ok(None);
        }
        Ok(Some(observer_for(&PROCESS_OBSERVER, std::process::id())))
    }

    /// An observer of this process that samples memory every `interval` while
    /// a window is open.
    ///
    /// # Panics
    ///
    /// Panics if `interval` is zero.
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use lance_core::memory::MemoryObserver;
    /// let observer = MemoryObserver::new(Duration::from_millis(10));
    /// let window = observer.open_window();
    /// // ... the work to observe ...
    /// let peak_bytes = window.close();
    /// # #[cfg(target_os = "linux")]
    /// # assert!(peak_bytes.is_some());
    /// ```
    pub fn new(interval: Duration) -> Self {
        assert!(
            !interval.is_zero(),
            "the memory sampling interval must be positive"
        );
        Self::with_reader(Box::new(read_process_memory), Some(interval))
    }

    fn with_reader(read: ReadMemory, interval: Option<Duration>) -> Self {
        Self {
            shared: Arc::new(Shared {
                read,
                state: Mutex::new(SamplerState::default()),
                wake: Condvar::new(),
                ticks: AtomicU64::new(0),
                parked: AtomicBool::new(false),
            }),
            interval,
            sampler_running: Mutex::new(false),
        }
    }

    /// Opens a window of time whose peak [`StageWindow::close`] reports.
    pub fn open_window(&self) -> StageWindow {
        let start = self.shared.sample();
        let window = Arc::new(WindowState {
            max_rss_bytes: AtomicU64::new(start.map_or(0, |memory| memory.rss_bytes)),
        });
        self.shared
            .lock_state()
            .windows
            .push(Arc::downgrade(&window));
        self.shared.wake.notify_one();
        self.start_sampler();
        StageWindow {
            window,
            shared: Arc::clone(&self.shared),
            start_peak_rss_bytes: start.map(|memory| memory.peak_rss_bytes),
        }
    }

    fn start_sampler(&self) {
        let Some(interval) = self.interval else {
            return;
        };
        let mut running = self
            .sampler_running
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if *running {
            return;
        }
        let shared = Arc::clone(&self.shared);
        match std::thread::Builder::new()
            .name("lance-memory-sampler".to_string())
            .spawn(move || shared.run(interval))
        {
            Ok(_) => *running = true,
            Err(err) => {
                static SPAWN_FAILED: Once = Once::new();
                SPAWN_FAILED.call_once(|| {
                    tracing::warn!(
                        "Could not start the memory sampler thread, retrying with the next \
                         stage; until then stage peaks come from stage boundaries only: {err}"
                    );
                });
            }
        }
    }
}

impl Drop for MemoryObserver {
    fn drop(&mut self) {
        self.shared.lock_state().stopped = true;
        self.shared.wake.notify_all();
    }
}

/// A window of time opened by [`MemoryObserver::open_window`].
#[doc(hidden)]
#[derive(Debug)]
pub struct StageWindow {
    window: Arc<WindowState>,
    shared: Arc<Shared>,
    start_peak_rss_bytes: Option<u64>,
}

impl StageWindow {
    /// The largest RSS of the process while the window was open, in bytes, or
    /// `None` where the platform reports no memory.
    ///
    /// The value is a lower bound: a spike between two readings is missed
    /// unless it raised the peak RSS the kernel keeps for the process
    /// (`VmHWM`). It covers the whole process, so work running beside the
    /// window counts too.
    pub fn close(self) -> Option<u64> {
        // This reading also counts for the windows that stay open.
        let end = self.shared.sample();
        let mut peak = self.window.max_rss_bytes.load(Ordering::Relaxed);
        // A peak that rose while the window was open happened inside it.
        if let Some(end) = end
            && self
                .start_peak_rss_bytes
                .is_some_and(|start| end.peak_rss_bytes > start)
        {
            peak = peak.max(end.peak_rss_bytes);
        }
        (peak > 0).then_some(peak)
    }
}

/// The process-wide observer and the process it was created in.
struct ProcessObserver {
    pid: u32,
    observer: Arc<MemoryObserver>,
}

static PROCESS_OBSERVER: AtomicPtr<ProcessObserver> = AtomicPtr::new(std::ptr::null_mut());

/// The observer of process `pid`, created on first use.
///
/// A child forked after the observer started inherits it without its sampler
/// thread, and possibly with the sampler's lock held at the moment of the
/// fork. The child therefore installs an observer of its own and never
/// touches the inherited one, which is leaked; the slot is a raw pointer
/// rather than a lock for the same reason.
fn observer_for(slot: &AtomicPtr<ProcessObserver>, pid: u32) -> Arc<MemoryObserver> {
    loop {
        let current = slot.load(Ordering::Acquire);
        // SAFETY: non-null values of the slot come from `Box::into_raw` below
        // and are never freed, so a loaded pointer stays valid.
        if let Some(installed) = unsafe { current.as_ref() }
            && installed.pid == pid
        {
            return Arc::clone(&installed.observer);
        }
        let fresh = Box::into_raw(Box::new(ProcessObserver {
            pid,
            observer: Arc::new(MemoryObserver::new(SAMPLE_INTERVAL)),
        }));
        match slot.compare_exchange(current, fresh, Ordering::AcqRel, Ordering::Acquire) {
            // SAFETY: `fresh` came from `Box::into_raw` above and the slot now
            // owns it for the rest of the process.
            Ok(_) => return Arc::clone(&unsafe { &*fresh }.observer),
            // SAFETY: another thread installed its observer first, so `fresh`
            // was never shared and can be freed here.
            Err(_) => drop(unsafe { Box::from_raw(fresh) }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::time::Instant;

    /// Memory readings a test sets by hand in place of `/proc`.
    #[derive(Debug, Default)]
    struct ScriptedMemory {
        rss_bytes: AtomicU64,
        peak_rss_bytes: AtomicU64,
    }

    impl ScriptedMemory {
        fn set_rss(&self, rss_bytes: u64) {
            self.rss_bytes.store(rss_bytes, Ordering::Relaxed);
            self.peak_rss_bytes.fetch_max(rss_bytes, Ordering::Relaxed);
        }

        /// A spike between two samples: only the all-time peak remembers it.
        fn spike(&self, rss_bytes: u64) {
            self.peak_rss_bytes.fetch_max(rss_bytes, Ordering::Relaxed);
        }
    }

    fn scripted_observer(
        memory: &Arc<ScriptedMemory>,
        interval: Option<Duration>,
    ) -> MemoryObserver {
        let memory = Arc::clone(memory);
        MemoryObserver::with_reader(
            Box::new(move || {
                Some(ProcessMemory {
                    rss_bytes: memory.rss_bytes.load(Ordering::Relaxed),
                    peak_rss_bytes: memory.peak_rss_bytes.load(Ordering::Relaxed),
                })
            }),
            interval,
        )
    }

    /// Scripted memory whose all-time peak was set before any window opens,
    /// so that only the readings see what happens inside a window.
    fn memory_after_an_earlier_peak() -> Arc<ScriptedMemory> {
        let memory = Arc::new(ScriptedMemory::default());
        memory.set_rss(1000);
        memory.set_rss(100);
        memory
    }

    fn wait_until(what: &str, condition: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !condition() {
            assert!(Instant::now() < deadline, "timed out waiting until {what}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[rstest]
    #[case::unset(None, Ok(false))]
    #[case::empty(Some(""), Ok(false))]
    #[case::one(Some("1"), Ok(true))]
    #[case::word(Some("true"), Ok(true))]
    #[case::upper_case(Some("ON"), Ok(true))]
    #[case::padded(Some(" yes "), Ok(true))]
    #[case::zero(Some("0"), Ok(false))]
    #[case::off(Some("off"), Ok(false))]
    #[case::not_a_boolean(
        Some("maybe"),
        Err(r#"LANCE_MEMORY_OBSERVE="maybe" is invalid, expected a boolean such as 1/0 or true/false"#)
    )]
    fn test_parse_observe_flag(
        #[case] value: Option<&str>,
        #[case] expected: std::result::Result<bool, &str>,
    ) {
        assert_eq!(
            parse_observe_flag(value),
            expected.map_err(ToString::to_string)
        );
    }

    #[test]
    fn test_observation_is_off_unless_switched_on() {
        assert!(
            std::env::var_os(OBSERVE_ENV_VAR).is_none(),
            "this test needs {OBSERVE_ENV_VAR} unset"
        );
        assert!(MemoryObserver::global().unwrap().is_none());
    }

    #[test]
    fn test_peak_is_the_largest_sample() {
        let memory = memory_after_an_earlier_peak();
        let observer = scripted_observer(&memory, None);
        let window = observer.open_window();
        memory.set_rss(300);
        observer.shared.tick();
        memory.set_rss(200);
        observer.shared.tick();
        memory.set_rss(150);
        assert_eq!(window.close(), Some(300));
    }

    #[test]
    fn test_peak_starts_from_the_rss_at_open() {
        let memory = Arc::new(ScriptedMemory::default());
        memory.set_rss(500);
        let observer = scripted_observer(&memory, None);
        let window = observer.open_window();
        memory.set_rss(100);
        observer.shared.tick();
        assert_eq!(window.close(), Some(500));
    }

    #[test]
    fn test_peak_catches_a_spike_between_samples() {
        let memory = Arc::new(ScriptedMemory::default());
        memory.set_rss(100);
        let observer = scripted_observer(&memory, None);
        let window = observer.open_window();
        memory.spike(900);
        memory.set_rss(120);
        assert_eq!(window.close(), Some(900));
    }

    #[test]
    fn test_peak_ignores_a_spike_before_the_window() {
        let memory = Arc::new(ScriptedMemory::default());
        memory.set_rss(100);
        memory.spike(900);
        let observer = scripted_observer(&memory, None);
        let window = observer.open_window();
        memory.set_rss(200);
        assert_eq!(window.close(), Some(200));
    }

    #[test]
    fn test_windows_track_their_own_peaks() {
        let memory = Arc::new(ScriptedMemory::default());
        memory.set_rss(100);
        let observer = scripted_observer(&memory, None);
        let first = observer.open_window();
        memory.set_rss(400);
        observer.shared.tick();
        memory.set_rss(100);
        let second = observer.open_window();
        observer.shared.tick();
        assert_eq!(second.close(), Some(100));
        assert_eq!(first.close(), Some(400));
    }

    #[test]
    fn test_every_reading_counts_for_every_open_window() {
        let memory = memory_after_an_earlier_peak();
        let observer = scripted_observer(&memory, None);
        let build = observer.open_window();
        memory.set_rss(300);
        let first = observer.open_window();
        memory.set_rss(100);
        assert_eq!(first.close(), Some(300));
        memory.set_rss(500);
        let second = observer.open_window();
        memory.set_rss(700);
        assert_eq!(second.close(), Some(700));
        memory.set_rss(100);
        // No sample was taken, yet the build saw what its stages saw.
        assert_eq!(build.close(), Some(700));
    }

    #[test]
    fn test_dropped_windows_are_pruned() {
        let memory = Arc::new(ScriptedMemory::default());
        memory.set_rss(100);
        let observer = scripted_observer(&memory, None);
        drop(observer.open_window());
        let open = observer.open_window();
        assert_eq!(observer.shared.lock_state().windows.len(), 1);
        drop(open);
        observer.shared.tick();
        assert!(observer.shared.lock_state().windows.is_empty());
    }

    #[test]
    fn test_no_readings_means_no_peak() {
        let observer = MemoryObserver::with_reader(Box::new(|| None), None);
        let window = observer.open_window();
        observer.shared.tick();
        assert_eq!(window.close(), None);
    }

    #[test]
    fn test_sampler_runs_only_while_a_window_is_open() {
        let memory = Arc::new(ScriptedMemory::default());
        memory.set_rss(100);
        let observer = scripted_observer(&memory, Some(Duration::from_millis(1)));
        // The second round checks that a window wakes the parked sampler.
        for _ in 0..2 {
            let window = observer.open_window();
            let ticks = observer.shared.ticks.load(Ordering::Relaxed);
            wait_until("the sampler ticks twice", || {
                observer.shared.ticks.load(Ordering::Relaxed) >= ticks + 2
            });
            assert_eq!(window.close(), Some(100));
            wait_until("the sampler parks", || {
                observer.shared.parked.load(Ordering::Acquire)
            });
        }
    }

    #[test]
    #[should_panic(expected = "must be positive")]
    fn test_a_zero_interval_is_rejected() {
        MemoryObserver::new(Duration::ZERO);
    }

    #[test]
    fn test_forked_child_replaces_the_inherited_observer() {
        let slot = AtomicPtr::new(std::ptr::null_mut());
        let parent = observer_for(&slot, 1);
        assert_eq!(parent.interval, Some(SAMPLE_INTERVAL));
        assert!(Arc::ptr_eq(&parent, &observer_for(&slot, 1)));
        let child = observer_for(&slot, 2);
        assert!(!Arc::ptr_eq(&parent, &child));
        assert!(Arc::ptr_eq(&child, &observer_for(&slot, 2)));
    }
}
