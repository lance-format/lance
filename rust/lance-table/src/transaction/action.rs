// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Action-based transactions.
//!
//! A transaction is a flat ordered list of typed [`Action`]s. Conflict
//! detection, rebasing, and manifest construction all work from `Action` type
//! and payload alone — no enclosing `Operation` semantics required at the
//! resolver layer. See `rust/lance-table/design/action_based_conflict_resolution.md`
//! for the full design; this module comment captures the principles every
//! action must obey.
//!
//! # Layers
//!
//! * [`Action`] — a single granular change to the manifest. Implements
//!   `apply` / `validate`. The conflict-resolution half (`reads` / `writes`
//!   masks and `rebase`) lives on a separate branch (PR B); PR A commits
//!   through the legacy
//!   [`TransactionRebase`](crate::io::commit::conflict_resolver) resolver.
//! * [`UserAction`] — a description plus an ordered list of `Action`s.
//!   Preserves "the user did X" structure across a compound transaction
//!   (e.g. "append batch" + "rebuild index").
//! * [`UserOperation`] — a UUID, read version, and ordered list of
//!   `UserAction`s. The normal top-level transaction variant; `Restore` and
//!   `Clone` remain separate top-level kinds.
//!
//! # Design principles
//!
//! Two patterns govern action payload shape. Both exist to avoid committing
//! too early to concrete values that may not survive a concurrent commit.
//!
//! ## 1. Criterion-based targeting
//!
//! Some actions act on an *open set* — entities a writer may not be able to
//! enumerate when the transaction is constructed, because a concurrent
//! writer can introduce new matching entities before commit. Such actions
//! carry a **predicate**, not an identity list. At `apply` time the
//! predicate is evaluated against the *current* manifest and every matching
//! entity is acted on.
//!
//! Example: [`InvalidateIndexCoverage`] carries `{ fragment_ids, field_ids }`,
//! not index UUIDs. If a concurrent writer creates a new index over one of
//! the listed fields, the apply against the rebased manifest still catches
//! it — using UUIDs would have left the new index covering stale data.
//!
//! ## 2. Late binding of action output
//!
//! Actions that *produce* new entities (fragments, row IDs, base ids) carry
//! the *intent* to produce, not a concrete identity. Identities are
//! allocated at `apply` time from manifest counters.
//!
//! Example: [`AddFragments`] carries `Fragment`s whose `id` is a placeholder;
//! `apply` allocates the concrete id from `manifest.max_fragment_id`.
//! [`AddBases`] does the same for base-path ids.
//!
//! Late binding makes `rebase` a no-op for these fields — there is nothing
//! to rewrite. Contrast with a "rebasable IDs" design where every concurrent
//! commit would force the serialized transaction to mutate.
//!
//! Late binding applies to **outputs** the action produces. It does *not*
//! apply to **inputs** the action depends on (e.g. which rows it intends to
//! mark deleted) — those are mutated by `rebase` when concurrent changes
//! shift them.
//!
//! ## 3. Symbolic intra-transaction references
//!
//! A direct consequence of late binding. If action *i* produces a fragment
//! and action *j > i* needs to reference it, *j* cannot use a concrete ID.
//! References are symbolic, resolved as actions are applied in order
//! through the driver's context. This pins the compound-transaction
//! dependency model to an **ordered list**, not a DAG.
//!
//! Today the catalog needs no explicit symbolic references: an
//! `AddIndex`'s criterion is re-resolved against the working manifest after
//! an earlier `AddFragments` applies, which covers the compound
//! "append + create index" case implicitly. The runtime representation
//! will be pinned down when the first action that needs it lands.
//!
//! # Translation
//!
//! Legacy `Operation` variants are translated to action lists by
//! [`actions_from_operation`] (or [`actions_from_operation_with_manifest`]
//! when the translation needs to diff against the prior manifest, as
//! `Merge` does). Translation is forward-only for `Delete`, `Merge`, and a
//! `CreateIndex` that removes indices, because their actions intentionally
//! drop fields needed to rebuild the original `Operation`. The reverse
//! [`operation_from_actions`] reconstructs only operations whose actions
//! carry every field of the legacy variant.

use std::collections::HashSet;
use std::sync::Arc;

use crate::format::{IndexMetadata, Manifest};
use lance_core::datatypes::Field;

mod action_trait;
mod config;
mod fragment;
mod index;
mod row_version;
mod schema;
#[cfg(any(test, feature = "test-util"))]
pub mod test_support;
mod user_op;

pub use action_trait::Action;
pub use config::*;
pub use fragment::*;
pub use index::*;
pub use row_version::*;
pub use schema::*;
pub use user_op::*;

/// Drop `fragment_ids` from the `fragment_bitmap` of every index that covers
/// any of `field_ids`.
///
/// Shared by [`InvalidateIndexCoverage`] and
/// [`ReplaceFragmentColumns`]: replacing or rewriting a covered
/// field's data leaves the index entry stale for the modified fragments, so
/// they are removed from its coverage bitmap.
pub fn invalidate_index_coverage(
    indices: &mut [IndexMetadata],
    fragment_ids: &[u64],
    field_ids: &[i32],
) {
    let fields: HashSet<i32> = field_ids.iter().copied().collect();
    for index in indices.iter_mut() {
        if index.fields.iter().any(|id| fields.contains(id))
            && let Some(bitmap) = &mut index.fragment_bitmap
        {
            for fragment_id in fragment_ids {
                bitmap.remove(*fragment_id as u32);
            }
        }
    }
}

/// Collect every field id in `fields` into `ids`, recursing into nested
/// children so a struct's child ids are honored too.
pub fn collect_field_ids(fields: &[Field], ids: &mut HashSet<i32>) {
    for field in fields {
        ids.insert(field.id);
        collect_field_ids(&field.children, ids);
    }
}

/// Drop data files that, after a schema change, reference no field still in
/// the schema.
///
/// A file that still covers at least one live field is kept verbatim, even if
/// it also lists a now-absent field id. Shared by [`DropFields`] and
/// [`ChangeSchema`]; mirrors the file pruning the legacy
/// [`Operation::Project`](super::Operation::Project) path performs.
pub fn prune_dead_data_files(manifest: &mut Manifest) {
    let live_ids: HashSet<i32> = manifest.schema.fields_pre_order().map(|f| f.id).collect();
    let mut new_fragments = (*manifest.fragments).clone();
    for fragment in new_fragments.iter_mut() {
        fragment
            .files
            .retain(|file| file.fields.iter().any(|id| live_ids.contains(id)));
    }
    manifest.fragments = Arc::new(new_fragments);
}

/// Remove every field whose id is in `drop` from `fields`, recursing into
/// nested children so a dropped struct child is honored too.
pub fn remove_field_ids(fields: &mut Vec<Field>, drop: &HashSet<i32>) {
    fields.retain(|field| !drop.contains(&field.id));
    for field in fields.iter_mut() {
        remove_field_ids(&mut field.children, drop);
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    #[test]
    fn action_validate_accepts_actions_without_structural_preconditions() {
        // Actions whose well-formedness is intrinsic validate against any
        // manifest, even one that does not contain their targets — their
        // `apply` is what validates against concrete manifest state.
        let manifest = empty_manifest();
        let actions: Vec<Box<dyn Action>> = vec![
            Box::new(AddFragments {
                fragments: vec![sample_fragment(0)],
                inserted_rows_filter: None,
            }),
            Box::new(RemoveFragments {
                selector: FragmentSelector::Ids(vec![7]),
            }),
            Box::new(udv(7, None)),
            Box::new(ReserveFragmentIds { num_fragments: 3 }),
        ];
        for action in actions {
            action
                .validate(&manifest)
                .unwrap_or_else(|e| panic!("{action:?} should validate: {e}"));
        }
    }

    #[test]
    fn replace_fragment_columns_apply_invalidates_covering_index() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![fragment_with_files(
            0,
            vec![data_file("b.lance", vec![1])],
        )]);
        let covering = index_meta("covers-1", &[1], Some(&[0, 1]));
        let other = index_meta("covers-0", &[0], Some(&[0, 1]));
        let mut indices = vec![covering, other];

        replace_columns(0, "b-new.lance", &[1])
            .apply(&mut manifest, &mut indices)
            .unwrap();

        // The index over the replaced field 1 loses fragment 0; the index
        // over field 0 is untouched.
        assert_eq!(indices[0].fragment_bitmap, Some(bitmap(&[1])));
        assert_eq!(indices[1].fragment_bitmap, Some(bitmap(&[0, 1])));
    }
}
