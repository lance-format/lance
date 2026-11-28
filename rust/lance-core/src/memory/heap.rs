// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Memory statistics supplied by the allocator of a Lance binding.

/// Memory that an allocator currently manages for the process.
///
/// The values are allocator statistics, not a decomposition of the process's
/// resident set size (RSS).
///
/// ```
/// let memory = lance_core::memory::HeapMemory {
///     in_use_bytes: 1024,
///     free_bytes: 512,
/// };
/// assert_eq!(memory.in_use_bytes, 1024);
/// assert_eq!(memory.free_bytes, 512);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeapMemory {
    /// Bytes the allocator reports as in use, including freed blocks in glibc tcache.
    pub in_use_bytes: u64,
    /// Bytes the allocator reports as free and retains, excluding glibc tcache.
    pub free_bytes: u64,
}

/// Reads the current memory statistics of an allocator.
///
/// Return `None` when the allocator cannot report them on this platform. The
/// Python and Java bindings use [`read_glibc_heap`].
///
/// ```
/// let _provider: lance_core::memory::HeapProvider = lance_core::memory::read_glibc_heap;
/// ```
pub type HeapProvider = fn() -> Option<HeapMemory>;

/// Reads the current glibc heap statistics.
///
/// Returns `None` outside GNU/Linux and on glibc versions before 2.33, where
/// `mallinfo2` is unavailable. The symbol is resolved at runtime so binaries
/// built for older glibc versions retain their existing compatibility. In-use
/// bytes include both arena allocations (`uordblks`) and mmap-backed
/// allocations (`hblkhd`); free bytes are those retained in arenas
/// (`fordblks`). Freed blocks in glibc's per-thread cache (tcache) can remain
/// counted as in use, so free bytes do not cover all allocator retention.
///
/// This function locks allocator arenas while glibc collects their statistics.
/// Use it from metric collection, not a memory sampler or an operation's hot
/// path, through [`super::refresh_metrics_with_heap`].
///
/// ```
/// let _reading = lance_core::memory::read_glibc_heap();
/// ```
pub fn read_glibc_heap() -> Option<HeapMemory> {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        glibc::read()
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        None
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod glibc {
    use std::ffi::{CStr, c_void};
    use std::sync::atomic::{AtomicPtr, Ordering};

    use super::HeapMemory;

    type Mallinfo2 = unsafe extern "C" fn() -> libc::mallinfo2;
    type Lookup = fn(&CStr) -> Option<*mut c_void>;

    const UNAVAILABLE: *mut c_void = std::ptr::dangling_mut::<c_void>();

    static MALLINFO2: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

    pub(super) fn read() -> Option<HeapMemory> {
        read_with_lookup(&MALLINFO2, lookup)
    }

    pub(super) fn read_with_lookup(
        cache: &AtomicPtr<c_void>,
        lookup: Lookup,
    ) -> Option<HeapMemory> {
        let address = match cache.load(Ordering::Acquire) {
            address if address.is_null() => {
                // Concurrent first lookups are harmless: `dlsym` is
                // thread-safe, and only a final address is published. An
                // uninitialized cache inherited through fork never waits for
                // the thread that disappeared.
                let address = lookup(c"mallinfo2").unwrap_or(UNAVAILABLE);
                cache.store(address, Ordering::Release);
                address
            }
            address => address,
        };
        if address == UNAVAILABLE {
            return None;
        }
        // SAFETY: every non-sentinel cached address came from resolving
        // `mallinfo2` with `dlsym`, whose signature matches `Mallinfo2`.
        let mallinfo2 = unsafe { std::mem::transmute::<*mut c_void, Mallinfo2>(address) };
        // SAFETY: `mallinfo2` was resolved above. The returned struct is owned.
        let info = unsafe { mallinfo2() };
        from_mallinfo2(info)
    }

    pub(super) fn lookup(symbol: &CStr) -> Option<*mut c_void> {
        // SAFETY: `symbol` is NUL-terminated and `RTLD_DEFAULT` asks the
        // dynamic loader for the process-global symbol.
        let address = unsafe { libc::dlsym(libc::RTLD_DEFAULT, symbol.as_ptr()) };
        (!address.is_null()).then_some(address)
    }

    pub(super) fn from_mallinfo2(info: libc::mallinfo2) -> Option<HeapMemory> {
        let arena_in_use = u64::try_from(info.uordblks).ok()?;
        let mmap_in_use = u64::try_from(info.hblkhd).ok()?;
        Some(HeapMemory {
            in_use_bytes: arena_in_use.checked_add(mmap_in_use)?,
            free_bytes: u64::try_from(info.fordblks).ok()?,
        })
    }
}

#[cfg(test)]
mod tests {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

    use super::*;

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    fn mallinfo2(uordblks: usize, hblkhd: usize, fordblks: usize) -> libc::mallinfo2 {
        libc::mallinfo2 {
            arena: 4096,
            ordblks: 1,
            smblks: 0,
            hblks: 1,
            hblkhd,
            usmblks: 0,
            fsmblks: 0,
            uordblks,
            fordblks,
            keepcost: 128,
        }
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn test_glibc_reading_includes_arena_and_mmap_allocations() {
        let memory = glibc::from_mallinfo2(mallinfo2(1000, 2000, 300)).unwrap();
        assert_eq!(memory.in_use_bytes, 3000);
        assert_eq!(memory.free_bytes, 300);
    }

    #[cfg(all(target_os = "linux", target_env = "gnu", target_pointer_width = "64"))]
    #[test]
    fn test_glibc_reading_rejects_overflow() {
        assert!(glibc::from_mallinfo2(mallinfo2(usize::MAX, 1, 0)).is_none());
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn test_glibc_lookup_reports_missing_symbols() {
        assert!(glibc::lookup(c"lance_missing_mallinfo2_symbol").is_none());
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn test_glibc_lookup_caches_missing_symbol() {
        static LOOKUPS: AtomicUsize = AtomicUsize::new(0);

        fn missing_symbol(symbol: &std::ffi::CStr) -> Option<*mut std::ffi::c_void> {
            assert_eq!(symbol, c"mallinfo2");
            LOOKUPS.fetch_add(1, Ordering::Relaxed);
            None
        }

        let cache = AtomicPtr::new(std::ptr::null_mut());
        assert_eq!(glibc::read_with_lookup(&cache, missing_symbol), None);
        assert_eq!(glibc::read_with_lookup(&cache, missing_symbol), None);
        assert_eq!(LOOKUPS.load(Ordering::Relaxed), 1);
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    fn glibc_supports_mallinfo2() -> bool {
        // SAFETY: glibc returns a process-lifetime, NUL-terminated version
        // string which must not be freed.
        let version = unsafe {
            let version = libc::gnu_get_libc_version();
            assert!(!version.is_null());
            std::ffi::CStr::from_ptr(version)
        };
        let version = version.to_str().unwrap();
        let mut components = version.split('.');
        let major = components.next().unwrap().parse::<u32>().unwrap();
        let minor = components.next().unwrap().parse::<u32>().unwrap();
        (major, minor) >= (2, 33)
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn test_read_glibc_heap_matches_symbol_availability() {
        let reading = read_glibc_heap();
        if glibc_supports_mallinfo2() {
            assert!(reading.is_some_and(|memory| memory.in_use_bytes > 0));
        } else {
            assert!(reading.is_none());
        }
    }

    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    #[test]
    fn test_read_glibc_heap_is_unavailable_outside_gnu_linux() {
        assert!(read_glibc_heap().is_none());
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    const FORK_TEST_CHILD: &str = "LANCE_HEAP_FORK_TEST_CHILD";

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    fn assert_fork_child_can_read(cache: &std::sync::atomic::AtomicPtr<std::ffi::c_void>) {
        use std::time::{Duration, Instant};

        // SAFETY: this test runs in an isolated subprocess. The parent waits
        // for the child and the child exits without returning to the harness.
        let pid = unsafe { libc::fork() };
        match pid {
            -1 => panic!("fork failed: {}", std::io::Error::last_os_error()),
            0 => {
                let available = glibc_supports_mallinfo2();
                let reading = glibc::read_with_lookup(cache, glibc::lookup);
                let exit_code = if available {
                    usize::from(!reading.is_some_and(|memory| memory.in_use_bytes > 0))
                } else {
                    usize::from(reading.is_some())
                };
                // SAFETY: the child must not run any test-harness cleanup.
                unsafe { libc::_exit(exit_code as i32) }
            }
            _ => {
                let mut status = 0;
                let deadline = Instant::now() + Duration::from_secs(5);
                loop {
                    // SAFETY: `status` points to writable memory and `pid` is
                    // the child returned by fork above.
                    let waited = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
                    assert!(waited >= 0, "waitpid failed: {waited}");
                    if waited == pid {
                        break;
                    }
                    if Instant::now() >= deadline {
                        // SAFETY: `pid` is still the child on this timeout
                        // path. Kill and reap it so the outer command's pipes
                        // close instead of hanging the test harness.
                        unsafe {
                            libc::kill(pid, libc::SIGKILL);
                            libc::waitpid(pid, &mut status, 0);
                        }
                        panic!("child did not finish within {deadline:?}");
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                assert!(libc::WIFEXITED(status), "child status: {status}");
                assert_eq!(libc::WEXITSTATUS(status), 0, "child status: {status}");
            }
        }
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn test_forked_children_can_read_heap() {
        use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

        static LOOKUP_STARTED: AtomicBool = AtomicBool::new(false);
        static RELEASE_LOOKUP: AtomicBool = AtomicBool::new(false);
        struct ReleaseLookup;
        impl Drop for ReleaseLookup {
            fn drop(&mut self) {
                RELEASE_LOOKUP.store(true, Ordering::Release);
            }
        }
        fn paused_lookup(_: &std::ffi::CStr) -> Option<*mut std::ffi::c_void> {
            LOOKUP_STARTED.store(true, Ordering::Release);
            while !RELEASE_LOOKUP.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            glibc::lookup(c"mallinfo2")
        }

        if std::env::var_os(FORK_TEST_CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "memory::heap::tests::test_forked_children_can_read_heap",
                ])
                .env(FORK_TEST_CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "child process failed: {output:?}");
            assert!(
                String::from_utf8_lossy(&output.stdout).contains("test result: ok. 1 passed;"),
                "the child test did not run: {output:?}"
            );
            return;
        }

        let cache = AtomicPtr::new(std::ptr::null_mut());
        LOOKUP_STARTED.store(false, Ordering::Release);
        RELEASE_LOOKUP.store(false, Ordering::Release);
        std::thread::scope(|scope| {
            let _release = ReleaseLookup;
            scope.spawn(|| glibc::read_with_lookup(&cache, paused_lookup));
            while !LOOKUP_STARTED.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            assert_fork_child_can_read(&cache);
        });

        let initialized = AtomicPtr::new(std::ptr::null_mut());
        let initial = glibc::read_with_lookup(&initialized, glibc::lookup);
        assert_eq!(initial.is_some(), glibc_supports_mallinfo2());
        assert_fork_child_can_read(&initialized);
    }

    #[cfg(not(feature = "metrics"))]
    #[test]
    fn test_refresh_without_metrics_does_not_call_provider() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        const CHILD: &str = "LANCE_HEAP_NO_METRICS_TEST_CHILD";
        static PROVIDER_CALLS: AtomicUsize = AtomicUsize::new(0);
        fn counted_provider() -> Option<HeapMemory> {
            PROVIDER_CALLS.fetch_add(1, Ordering::Relaxed);
            Some(HeapMemory {
                in_use_bytes: 100,
                free_bytes: 50,
            })
        }

        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "memory::heap::tests::test_refresh_without_metrics_does_not_call_provider",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "child process failed: {output:?}");
            assert!(
                String::from_utf8_lossy(&output.stdout).contains("test result: ok. 1 passed;"),
                "the child test did not run: {output:?}"
            );
            return;
        }

        crate::memory::set_observation(true);
        PROVIDER_CALLS.store(0, Ordering::Relaxed);
        crate::memory::refresh_metrics_with_heap(counted_provider);
        assert_eq!(PROVIDER_CALLS.load(Ordering::Relaxed), 0);
    }
}
