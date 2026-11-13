// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Memory events logged on the `lance::memory` tracing target.

use crate::utils::tracing::TRACE_MEMORY;

/// Logs one event of observed work with the id of the process and the current
/// memory of the process and of its cgroup. A field without a value, such as
/// the stage of an event about a whole operation, is left out of the event.
pub(super) fn emit(
    event: &'static str,
    operation_id: u64,
    operation: &'static str,
    index_type: &str,
    stage: Option<&str>,
    peak_bytes: Option<u64>,
) {
    let rss_bytes = super::proc::read_process_memory().map(|memory| memory.rss_bytes);
    let (cgroup_bytes, cgroup_limit_bytes) = cgroup_memory();
    tracing::info!(
        target: TRACE_MEMORY,
        event,
        pid = std::process::id(),
        operation_id,
        operation,
        index_type,
        stage,
        peak_bytes,
        rss_bytes,
        cgroup_bytes,
        cgroup_limit_bytes,
    );
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
