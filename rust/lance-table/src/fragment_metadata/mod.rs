// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Immutable fragment metadata and buffered state changes.
//!
//! Native transactions own validation and rebase. This module prepares tree
//! objects and a descriptor. The Version Manifest is the publication authority.
//!
//! This API and its on-disk layout are experimental and may change at any time.

pub mod action;
pub mod commit;
mod content;
mod context;
pub mod layout;
mod leaf;
mod node;
mod store;
mod tree;
mod validation;

#[cfg(any(test, feature = "test-util"))]
pub mod support;

pub use commit::{TouchedFragments, ValidatedCommit, data_replacement};
pub use content::apply_actions;
#[cfg(any(test, feature = "test-util"))]
pub use content::{apply_verified, squash_buffer};
pub use context::TreeContext;
pub use layout::{MANIFEST_LAYOUT_FLAT, MANIFEST_LAYOUT_KEY, MANIFEST_LAYOUT_TREE, ManifestLayout};
pub use node::{DEFAULT_MAX_LEAF_BYTES, DEFAULT_MAX_NODE_BYTES, FragmentTreeConfig};
#[cfg(any(test, feature = "test-util"))]
pub use store::root_path;
pub use store::{LeafLocation, SingleFlightLeafCache, StoredFragments};
pub use tree::{
    BootstrapStats, CommitMode, CommitStats, FragmentTree, InventoryRetention, LeafFragments,
    LeafSummaries, OffsetResolution, SnapshotPolicy, TreeCommit,
};

#[cfg(any(test, feature = "test-util"))]
pub use tree::ShapeReport;
