// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Opt-in observation of the process's memory.
//!
//! Observation is off unless [`OBSERVE_ENV_VAR`] (`LANCE_MEMORY_OBSERVE`) is
//! set to a true value such as `1`. While it is on, Lance reads the process's
//! resident set size (RSS) during the stages of an index build and reports the
//! peak of every stage and of the whole build. Together with gauges for the
//! process and its cgroup, these are published through the `metrics` crate
//! facade when Lance is built with the `metrics` feature. Readings come from
//! `/proc` and cgroupfs, so observation is available on Linux only.
//!
//! An application that installs its own recorder registers the metrics once
//! and refreshes the gauges from its collection loop; the Python and Java
//! OpenTelemetry bridges do both automatically:
//!
//! ```
//! lance_core::memory::describe_metrics();
//! // ... on every collection:
//! lance_core::memory::refresh_metrics();
//! ```

#[cfg(any(test, all(feature = "metrics", target_os = "linux")))]
mod cgroup;
mod observer;
mod proc;
mod telemetry;

pub use observer::{MemoryObserver, OBSERVE_ENV_VAR, StageWindow};
pub use telemetry::{
    METRIC_CGROUP, METRIC_CGROUP_LIMIT, METRIC_OPERATION_PEAK, METRIC_RSS, METRIC_RSS_PEAK,
    METRIC_STAGE_PEAK, PEAK_BOUNDS, describe_metrics, histogram_bounds, record_operation_peak,
    record_stage_peak, refresh_metrics,
};
