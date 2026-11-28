// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Memory events logged on the `lance::memory` tracing target.

use std::cell::LazyCell;

use crate::utils::tracing::TRACE_MEMORY;

/// Logs one event of observed work with the id of the process and the current
/// memory of the process and of its cgroup. A field without a value, such as
/// the stage of an event about a whole operation, is left out of the event.
pub(super) fn emit(
    event: &'static str,
    operation_id: u64,
    operation: &'static str,
    index_type: Option<&str>,
    stage: Option<&str>,
    peak_bytes: Option<u64>,
) {
    // Reading the memory takes several file reads at every stage boundary.
    // Tracing evaluates the fields only for an event that a subscriber takes,
    // or that it hands to `log` when no subscriber is set.
    let memory = LazyCell::new(read_memory);
    tracing::info!(
        target: TRACE_MEMORY,
        event,
        pid = std::process::id(),
        operation_id,
        operation,
        index_type,
        stage,
        peak_bytes,
        rss_bytes = memory.rss_bytes,
        cgroup_bytes = memory.cgroup_bytes,
        cgroup_limit_bytes = memory.cgroup_limit_bytes,
    );
}

/// The memory of the process and of its cgroup when an event is logged.
struct Readings {
    rss_bytes: Option<u64>,
    /// Memory charged to the cgroup that sets the tightest limit on the process.
    cgroup_bytes: Option<u64>,
    cgroup_limit_bytes: Option<u64>,
}

#[cfg(test)]
thread_local! {
    /// How often this thread read the memory for an event.
    static READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn read_memory() -> Readings {
    #[cfg(test)]
    READS.with(|reads| reads.set(reads.get() + 1));
    let (cgroup_bytes, cgroup_limit_bytes) = cgroup_memory();
    Readings {
        rss_bytes: super::proc::read_process_memory().map(|memory| memory.rss_bytes),
        cgroup_bytes,
        cgroup_limit_bytes,
    }
}

/// The memory charged to the cgroup that sets the tightest limit on the
/// process, and that limit.
fn cgroup_memory() -> (Option<u64>, Option<u64>) {
    #[cfg(target_os = "linux")]
    {
        super::cgroup::read_cgroup_memory().map_or((None, None), |cgroup| {
            (cgroup.current_bytes, cgroup.limit_bytes)
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        (None, None)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, PoisonError};

    use tracing::span::{Attributes, Id, Record};
    use tracing::subscriber::Interest;
    use tracing::{Event, Metadata};

    use super::*;
    use crate::memory::EMITTING;

    /// Takes the memory events, or none of them.
    struct Taker {
        takes_memory_events: bool,
    }

    impl tracing::Subscriber for Taker {
        fn register_callsite(&self, _: &'static Metadata<'static>) -> Interest {
            // Asked on every event, whatever earlier tests cached.
            Interest::sometimes()
        }

        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            self.takes_memory_events && metadata.target() == TRACE_MEMORY
        }

        fn new_span(&self, _: &Attributes<'_>) -> Id {
            Id::from_u64(1)
        }

        fn record(&self, _: &Id, _: &Record<'_>) {}
        fn record_follows_from(&self, _: &Id, _: &Id) {}
        fn event(&self, _: &Event<'_>) {}
        fn enter(&self, _: &Id) {}
        fn exit(&self, _: &Id) {}
    }

    #[test]
    fn test_memory_is_read_only_for_an_event_that_is_taken() {
        let _emitting = EMITTING.lock().unwrap_or_else(PoisonError::into_inner);
        let reads_while = |takes_memory_events| {
            let before = READS.with(std::cell::Cell::get);
            tracing::subscriber::with_default(
                Taker {
                    takes_memory_events,
                },
                || emit("operation_start", 1, "write", None, None, None),
            );
            READS.with(std::cell::Cell::get) - before
        };
        assert_eq!(reads_while(false), 0);
        // One reading serves all the memory fields of the event.
        assert_eq!(reads_while(true), 1);
    }

    const LOG_CHILD: &str = "LANCE_MEMORY_LOG_TEST_CHILD";

    #[test]
    fn test_memory_events_reach_log_without_a_tracing_subscriber() {
        // The logger and tracing dispatcher are process-wide, so isolate the
        // same log-only setup that the Java binding uses in a child process.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "memory::events::tests::test_memory_log_child"])
            .env(LOG_CHILD, "1")
            .output()
            .unwrap();
        assert!(output.status.success(), "child process failed: {output:?}");
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("test result: ok. 1 passed;"),
            "the child test did not run: {output:?}"
        );
    }

    struct LogRecorder(Mutex<Vec<String>>);

    impl log::Log for LogRecorder {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.target() == TRACE_MEMORY
        }

        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                self.0.lock().unwrap().push(record.args().to_string());
            }
        }

        fn flush(&self) {}
    }

    #[test]
    fn test_memory_log_child() {
        if std::env::var_os(LOG_CHILD).is_none() {
            return;
        }
        static LOGGER: LogRecorder = LogRecorder(Mutex::new(Vec::new()));
        assert!(!tracing::dispatcher::has_been_set());
        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Off);
        let before = READS.with(std::cell::Cell::get);
        emit("operation_start", 1, "write", None, None, None);
        assert_eq!(READS.with(std::cell::Cell::get), before);
        assert!(LOGGER.0.lock().unwrap().is_empty());

        log::set_max_level(log::LevelFilter::Info);
        emit("operation_start", 1, "write", None, None, None);
        assert_eq!(READS.with(std::cell::Cell::get), before + 1);
        let records = LOGGER.0.lock().unwrap();
        assert_eq!(records.len(), 1, "{records:?}");
        assert!(records[0].contains("operation_start"), "{records:?}");
        assert!(records[0].contains("write"), "{records:?}");
    }
}
