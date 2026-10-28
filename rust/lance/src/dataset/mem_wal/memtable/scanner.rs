// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read path for MemTable.
//!
//! This module provides query execution over MemTable data using DataFusion.
//!
//! ## Architecture
//!
//! ```text
//!                     MemTableScanner (Builder)
//!                            |
//!                     create_plan()
//!                            |
//!               +------------+------------+
//!               |                         |
//!          Full Scan                 Index Query
//!               |                         |
//!               v                         v
//!         MemTableScanExec            IndexExec
//!               |                         |
//!               +------------+------------+
//!                            |
//!                     DataFusion Execution
//!                            |
//!                            v
//!                   SendableRecordBatchStream
//! ```
//!
//! ## Key Features
//!
//! - **MVCC Visibility**: All scans respect visibility sequence numbers
//! - **Index Support**: whichever indexes the memtable maintains, chosen by
//!   what they can answer
//! - **DataFusion Integration**: Full ExecutionPlan compatibility

mod builder;
mod exec;

pub use builder::MemTableScanner;
pub(crate) use builder::local_fts_query;

/// The matches a plan checked for being their key's newest version: zero when
/// every visible row was read instead.
#[cfg(test)]
pub(crate) fn newest_checks(
    plan: &std::sync::Arc<dyn datafusion::physical_plan::ExecutionPlan>,
) -> usize {
    metric_sum(plan, exec::NEWEST_CHECKS_METRIC)
}

/// How many times a plan read every visible row because its indexes matched
/// too many.
#[cfg(test)]
pub(crate) fn fallback_reads(
    plan: &std::sync::Arc<dyn datafusion::physical_plan::ExecutionPlan>,
) -> usize {
    metric_sum(plan, exec::FALLBACK_READS_METRIC)
}

#[cfg(test)]
fn metric_sum(
    plan: &std::sync::Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    name: &str,
) -> usize {
    plan.metrics()
        .and_then(|metrics| metrics.sum_by_name(name))
        .map_or(0, |value| value.as_usize())
        + plan
            .children()
            .into_iter()
            .map(|child| metric_sum(child, name))
            .sum::<usize>()
}
pub use exec::{FtsIndexExec, MemTableScanExec, ScalarMemIndexExec, VectorIndexExec};
