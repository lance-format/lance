// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Transaction definitions for updating datasets
//!
//! Prior to creating a new manifest, a transaction must be created representing
//! the changes being made to the dataset. By representing them as incremental
//! changes, we can detect whether concurrent operations are compatible with
//! one another. We can also rebuild manifests when retrying committing a
//! manifest.
//!
//! For more details please refer to the
//! [Transaction Specification](https://lance.org/format/table/transaction/#transaction-types).
//!
//! The work splits along these lines:
//!
//! ```text
//! builder            Transaction: an operation plus the version it was based on
//! operation          the vocabulary of changes an operation can describe
//! action             the finer-grained Transaction V2 vocabulary (draft)
//! update_map         incremental edits to the manifest's string maps
//! validate           pre-commit checks against the manifest being replaced
//! manifest_build     applying an operation to produce the next manifest
//! index_maintenance  how that narrows or drops index metadata
//! row_version        how it assigns row ids and per-row version metadata
//! conflicts          whether two operations collide, for the commit retry path
//! proto              the persisted protobuf encoding of all of the above
//! ```

pub mod action;
mod builder;
mod conflicts;
mod index_maintenance;
mod manifest_build;
mod operation;
mod proto;
mod row_version;
mod update_map;
mod validate;

#[cfg(test)]
pub(crate) mod test_support;

pub use builder::{Transaction, TransactionBuilder};
pub use operation::{
    DataOverlayGroup, DataReplacementGroup, Operation, RewriteGroup, RewrittenIndex, UpdateMode,
    UpdatedFragmentOffsets,
};
pub use update_map::{
    UpdateMap, UpdateMapEntry, translate_config_updates, translate_schema_metadata_updates,
};
pub use validate::validate_operation;

use crate::format::{IndexMetadata, Manifest};
use roaring::RoaringBitmap;
use std::collections::BTreeMap;
use uuid::Uuid;

/// Non-system logical index name -> its physical segments, ordered by UUID.
///
/// Whole segment metadata rather than UUIDs alone: operations such as `Rewrite`
/// prune a segment's fragment bitmap while keeping its UUID, so a UUID-only
/// comparison would keep coverage for an index that no longer spans the same
/// base fragments.
pub type LogicalIndexSegments = BTreeMap<String, Vec<CoverageIdentity>>;

/// What one physical index segment contributes to coverage.
///
/// Deliberately not the whole [`IndexMetadata`]. It rests on one contract:
/// changing an index's physical contents mints a new UUID. Of the mutations
/// sanctioned under an existing UUID, only the fragment bitmap changes which
/// rows the index answers for -- an `Update` prunes it in place, and
/// `migrate_indices` recalculates it -- so the UUID alone is not enough and the
/// bitmap has to be compared too. The rest of the metadata, file lists and
/// timestamps and inferred details, is filled in by migrations routinely;
/// comparing it would withdraw coverage for no reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageIdentity {
    uuid: Uuid,
    fragment_bitmap: Option<RoaringBitmap>,
}

/// The version a transaction read, as the coverage derivation needs it.
///
/// An index covering every fragment live at this version holds every row
/// compaction had copied into the base table by then, so it is caught up to
/// that version's `compacted_sstables`. That is the only proof available:
/// nothing maps a compaction generation to the fragments its rows landed in.
///
/// `read_version` is fixed for the life of a transaction and survives rebase,
/// so the credit a commit can prove is stable across attempts. The recorded
/// result may still differ between attempts, because a rebased attempt sees a
/// different head: other commits move the compacted generations and the
/// positions already recorded.
#[derive(Debug, Clone, Copy)]
pub struct ReadVersionState<'a> {
    pub manifest: &'a Manifest,
    pub indices: &'a [IndexMetadata],
}
