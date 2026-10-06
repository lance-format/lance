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
//! - **Index Support**: whichever indexes the memtable maintains, asked by
//!   what they can answer
//! - **DataFusion Integration**: Full ExecutionPlan compatibility

mod builder;

mod exec;

pub use builder::MemTableScanner;
pub(crate) use builder::local_fts_query;
pub use exec::{FtsIndexExec, MemTableScanExec, ScalarIndexExec, VectorIndexExec};
