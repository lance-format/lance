// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Immutable fragment metadata and buffered state changes.
//!
//! Native transactions own validation and rebase. This module prepares tree
//! objects and a descriptor. The Version Manifest is the publication authority.
//!
//! This API and its on-disk layout are experimental and may change at any time.

pub mod action;
mod bulk;
pub mod commit;
pub mod layout;
mod leaf;
pub mod node;
pub mod store;
pub mod tree;
mod validation;

#[cfg(any(test, feature = "test-util"))]
pub mod support;

pub use commit::{TouchedFragments, ValidatedCommit, data_replacement};
pub use layout::{MANIFEST_LAYOUT_FLAT, MANIFEST_LAYOUT_KEY, MANIFEST_LAYOUT_TREE, ManifestLayout};
pub use node::{DEFAULT_MAX_LEAF_BYTES, DEFAULT_MAX_NODE_BYTES, FragmentTreeConfig};
pub use tree::{BootstrapStats, CommitStats, FragmentTree, OffsetResolution, SnapshotPolicy};
