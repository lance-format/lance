// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vamana (DiskANN) graph as an IVF sub-index.
//!
//! Each IVF partition holds one flat Vamana graph over the partition's storage:
//! vertex `i` is storage position `i`. The graph is built with the two-pass
//! algorithm of the DiskANN paper, kept up to date through compaction with
//! FreshVamana's delete consolidation, and extended in place when new rows are
//! merged into a partition.
//!
//! This module exists only in the Vamana fork of Lance.

pub(crate) mod batch;
pub mod build;
pub mod consolidate;
pub mod entry_points;
pub mod graph;
pub mod incremental;
pub mod index;
pub mod insert;
pub mod params;
pub mod remap;
pub mod search;
pub(crate) mod shards;
#[cfg(test)]
pub(crate) mod testing;
pub mod walk;

pub use index::Vamana;
pub use params::{VamanaBuildParams, VamanaMetadata, VamanaQueryParams};

pub const VAMANA_TYPE: &str = "VAMANA";
pub const VAMANA_METADATA_KEY: &str = "lance:vamana";
pub const ROW_ID_COL: &str = "__row_id";
pub const NEIGHBORS_COL: &str = "__neighbors";

/// The seed every build, insertion and entry-point training draws from, so a
/// partition rebuilt from the same rows comes out as the same graph.
pub const VAMANA_SEED: u64 = 42;
