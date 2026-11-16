// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Opt-in observation of the process's memory.
//!
//! Observation is off unless [`set_observation`] turns it on or, before its
//! first call, [`OBSERVE_ENV_VAR`] (`LANCE_MEMORY_OBSERVE`) is
//! set to a true value such as `1`. While it is on, Lance reads the process's
//! resident set size (RSS) during the operations it observes, such as index
//! builds, compaction, `optimize_indices`, MemWAL flushes and writes, and
//! reports every stage and the whole operation in two ways:
//!
//! - as trace events on the [`TRACE_MEMORY`] target, one when the stage or
//!   operation starts and one with its peak when it completes, each carrying
//!   the id of the operation and the current memory of the process and of its
//!   cgroup;
//! - through the `metrics` crate facade when Lance is built with the `metrics`
//!   feature, as the peak of every stage and of the whole operation next to
//!   gauges for the process and its cgroup.
//!
//! Readings come from `/proc` and cgroupfs, so observation is available on
//! Linux only.
//!
//! On every platform, the same switch and feature also report how many bytes
//! the I/O schedulers, the DataFusion memory pools that Lance builds from its
//! execution options and the MemWAL writers of the process hold, from their
//! own accounting.
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
//!
//! [`TRACE_MEMORY`]: crate::utils::tracing::TRACE_MEMORY

#[cfg(any(test, target_os = "linux"))]
mod cgroup;
mod components;
mod events;
mod observer;
mod proc;
mod report;
mod scope;
mod telemetry;

pub use components::{Component, register_component};
pub use observer::{MemoryObserver, OBSERVE_ENV_VAR, StageWindow, set_observation};
pub use report::OperationReport;
pub use scope::{ObservedOperation, OperationScope, Stage, StageReporter};
pub use telemetry::{
    METRIC_CGROUP, METRIC_CGROUP_LIMIT, METRIC_COMPONENT, METRIC_OPERATION_PEAK, METRIC_RSS,
    METRIC_RSS_PEAK, METRIC_STAGE_PEAK, PEAK_BOUNDS, describe_metrics, histogram_bounds,
    refresh_metrics,
};

/// Taken by every test that emits memory events. Tracing caches a callsite's
/// interest when it is first hit, and while only one scoped subscriber exists
/// it asks just the thread that hits it: a test emitting without a subscriber
/// could then turn the callsite off for the subscriber of a test running in
/// parallel.
#[cfg(test)]
static EMITTING: std::sync::Mutex<()> = std::sync::Mutex::new(());
