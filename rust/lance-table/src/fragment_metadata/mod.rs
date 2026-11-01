// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Immutable fragment metadata and buffered state changes.
//!
//! The [Fragment Metadata Tree](https://lance.org/format/table/fragment_metadata/)
//! page is the spec for the stored objects and fields. This header maps them to
//! the names the code uses.
//!
//! ```text
//! Version Manifest
//! └─ descriptor, pb::FragmentTree
//!    ├─ root, inline or _bt/root/{uuid}.root
//!    │  ├─ buffer
//!    │  └─ children ordered by fence
//!    │     ├─ interior, _bt/node/{uuid}.node, with children and a buffer
//!    │     └─ leaf, _bt/leaf/{uuid}.lance
//!    └─ suffix, mutations_since_root
//! ```
//!
//! A commit buffers its changes near the root, and pressure drains them toward
//! the leaves in batches worth a rewrite.
//!
//! - A descriptor is the `FragmentTree` message a Version Manifest publishes.
//! - The suffix is its `mutations_since_root`, the changes an external root lacks.
//! - A checkpoint publishes a root with an empty suffix.
//! - A fence is a child's `min_key`, the only routing bound between siblings.
//! - A root's frontier is its `next_action_sequence`. Every action sequence an
//!   interior below it names stays under that bound.
//! - A retained inventory is a snapshot's complete fragment list in memory.
//!   Any other snapshot reads leaves on demand.
//!
//! `tree` holds the snapshot, its reads and the rewrite a commit runs. `store`
//! reads, writes and caches objects, and `leaf` encodes leaf files. `node` owns
//! routing and byte budgets, `content` replays and squashes mutations, `commit`
//! derives what a commit changes, and `validation` checks what storage returns.
//!
//! A [`Transaction`](crate::transaction::Transaction) is validated and rebased
//! before it reaches this module, which prepares tree objects and a descriptor.
//! The Version Manifest is the publication authority.
//!
//! ```
//! # use lance_table::format::{Fragment, Manifest};
//! # use lance_table::fragment_metadata::{
//! #     CommitMode, FragmentChanges, FragmentTree, FragmentTreeConfig, InventoryRetention,
//! #     PublicationPolicy, TreeContext, action,
//! # };
//! # async fn example(
//! #     context: TreeContext,
//! #     fragments: Vec<Fragment>,
//! #     appended: &Fragment,
//! #     committed: &Manifest,
//! # ) -> lance_core::Result<()> {
//! let config = FragmentTreeConfig::default();
//! let policy = PublicationPolicy::default();
//! let tree = FragmentTree::build(
//!     context.clone(),
//!     config.clone(),
//!     fragments,
//!     1,
//!     policy,
//!     InventoryRetention::Discard,
//! )
//! .await?;
//!
//! let changes = FragmentChanges::new(vec![action::upsert_fragment(appended)?]);
//! let prepared = tree
//!     .begin_commit(CommitMode::Incremental)
//!     .prepare(changes, policy)
//!     .await?;
//!
//! // Nothing is visible yet. The caller commits a Version Manifest carrying
//! // `prepared.descriptor()` and `prepared.next_fragment_id()`, and only then
//! // binds the snapshot to it.
//! let published = prepared.into_published(committed, config.clone(), context.clone())?;
//!
//! let reopened = FragmentTree::open(context, committed, config).await?;
//! assert_eq!(reopened.count_fragments(), published.count_fragments());
//! # Ok(())
//! # }
//! ```
//!
//! This API and its on-disk layout are experimental and may change at any time.

pub mod action;
mod commit;
mod content;
mod context;
mod layout;
mod leaf;
mod node;
mod store;
mod tree;
mod validation;

#[cfg(any(test, feature = "test-util"))]
pub mod support;

pub use commit::{FragmentChanges, TouchedFragments, data_replacement_actions};
pub use content::apply_actions;
#[cfg(any(test, feature = "test-util"))]
pub use content::{apply_verified, squash_buffer};
pub use context::TreeContext;
pub use layout::{MANIFEST_LAYOUT_FLAT, MANIFEST_LAYOUT_KEY, MANIFEST_LAYOUT_TREE, ManifestLayout};
pub use node::FragmentTreeConfig;
#[cfg(any(test, feature = "test-util"))]
pub use store::root_path;
pub use store::{LeafLocation, SingleFlightLeafCache, StoredFragments, TREE_DIR};
pub use tree::{
    CommitMode, FragmentTree, InventoryRetention, LeafFragments, LeafSummaries, OffsetResolution,
    PublicationPolicy, TreeCommit,
};

#[cfg(any(test, feature = "test-util"))]
pub use tree::ShapeReport;
