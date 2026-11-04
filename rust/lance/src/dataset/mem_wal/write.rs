// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemWAL write path.
//!
//! [`ShardWriter`] coordinates configuration, WAL durability, optional
//! memtables, index application, and generation flushing for one shard.
//! Configuration and validation live next to the data they govern, while
//! background task ownership and write coordination stay separate.

mod backpressure;
mod config;
mod handlers;
mod replay;
mod schema;
pub mod stats;
pub mod tasks;
mod writer;

#[cfg(test)]
mod test_support;

#[cfg(test)]
mod tests;

pub use backpressure::{
    BackpressureController, BackpressureStats, BackpressureStatsSnapshot, BackpressureWaitGuard,
    Drain, DurabilityCell, DurabilityResult, DurabilityWatcher, LocalBackpressureController,
    ShardMemory,
};
pub use config::ShardWriterConfig;
pub use stats::{MemTableStats, WalStats, WriteStats, WriteStatsSnapshot};
pub use writer::{FreezeFence, ShardWriter, WriteResult};
