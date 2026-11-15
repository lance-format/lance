// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Secondary index APIs, lifecycle coordination, loading, and maintenance.
//!
//! Index algorithms and storage remain in `lance-index`. This module coordinates
//! Dataset metadata: `lifecycle` publishes and maintains logical index
//! segments, `provenance` decides whether stored coverage is still fresh,
//! `loading` opens index implementations and owns their cache identity, and
//! `statistics` reports persisted metadata, including the existing legacy
//! auto-migration needed to recompute incomplete fragment statistics.

use lance_index::IndexType;

mod api;
pub(crate) mod append;
mod create;
pub mod frag_reuse;
pub mod frag_reuse_reader;
mod frag_reuse_remapping;
mod frag_reuse_v0_cache;
pub(crate) mod frag_reuse_with_stable_row_ids;
mod lifecycle;
pub(crate) mod loading;
pub mod mem_wal;
pub(crate) mod memory_progress;
pub mod optimize;
pub mod prefilter;
mod prewarm;
pub(crate) mod provenance;
pub(crate) mod remap;
pub mod scalar;
pub(crate) mod scalar_logical;
pub(crate) mod segment;
pub(crate) mod segment_coverage;
mod statistics;
pub mod vector;
pub(crate) mod visibility;

pub use crate::index::api::{
    DatasetIndexExt, IndexSegment, IndexSegmentStatistics, IntoIndexSegment,
};
pub use crate::index::loading::DatasetIndexInternalExt;
pub use crate::index::prefilter::{FilterLoader, PreFilter};
pub use crate::index::vector::{LogicalIvfView, LogicalVectorIndex};
pub use create::CreateIndexBuilder;
pub use lance_index::IndexDescription;
pub use lance_index::IndexParams;
pub use lance_index::progress::{IndexBuildProgress, NoopIndexBuildProgress};
pub use lance_index::registry::register_scalar_index_plugin;

/// Builds an index.
#[async_trait::async_trait]
pub trait IndexBuilder {
    /// Return the concrete index type.
    fn index_type() -> IndexType;

    /// Build the index.
    async fn build(&self) -> crate::Result<()>;
}

#[cfg(test)]
mod tests;
