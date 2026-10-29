// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Resident memory of the current process, read from `/proc/self/status`.

/// Resident set size of the current process, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ProcessMemory {
    pub(super) rss_bytes: u64,
    /// The largest RSS since the process started (`VmHWM`).
    pub(super) peak_rss_bytes: u64,
}

/// Current and peak RSS, or `None` where the platform does not report them.
pub(super) fn read_process_memory() -> Option<ProcessMemory> {
    #[cfg(target_os = "linux")]
    {
        parse_status(&std::fs::read_to_string("/proc/self/status").ok()?)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(any(test, target_os = "linux"))]
fn parse_status(status: &str) -> Option<ProcessMemory> {
    let mut rss_bytes = None;
    let mut peak_rss_bytes = None;
    for line in status.lines() {
        if let Some(value) = line.strip_prefix("VmRSS:") {
            rss_bytes = parse_kib(value);
        } else if let Some(value) = line.strip_prefix("VmHWM:") {
            peak_rss_bytes = parse_kib(value);
        }
    }
    Some(ProcessMemory {
        rss_bytes: rss_bytes?,
        peak_rss_bytes: peak_rss_bytes?,
    })
}

#[cfg(any(test, target_os = "linux"))]
fn parse_kib(value: &str) -> Option<u64> {
    let mut parts = value.split_whitespace();
    match (parts.next(), parts.next()) {
        (Some(number), Some("kB")) => number.parse::<u64>().ok()?.checked_mul(1024),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::typical(
        "Name:\tlance\nVmPeak:\t    4096 kB\nVmHWM:\t    1536 kB\nVmRSS:\t    1024 kB\nThreads:\t8\n",
        Some((1024 * 1024, 1536 * 1024))
    )]
    #[case::missing_peak("VmRSS:\t1024 kB\n", None)]
    #[case::not_a_number("VmHWM:\t1536 kB\nVmRSS:\tlots kB\n", None)]
    #[case::unexpected_unit("VmHWM:\t1536 kB\nVmRSS:\t1 GB\n", None)]
    fn test_parse_status(#[case] status: &str, #[case] expected: Option<(u64, u64)>) {
        let parsed = parse_status(status).map(|memory| (memory.rss_bytes, memory.peak_rss_bytes));
        assert_eq!(parsed, expected);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_read_process_memory_reports_this_process() {
        let memory = read_process_memory().expect("/proc/self/status is readable on Linux");
        assert!(memory.rss_bytes > 0, "{memory:?}");
        assert!(memory.peak_rss_bytes >= memory.rss_bytes, "{memory:?}");
    }
}
