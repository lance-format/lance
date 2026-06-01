// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bidirectional translation between [`Operation`](super::super::Operation) and
//! [`UserAction`].
//!
//! This module is the parity surface between the legacy per-`Operation`
//! semantics and the action-based pipeline: every concurrent-write subtlety
//! the old resolver knew about must be re-expressed here as an explicit
//! action list, or it's lost.
//!
//! # Two entry points
//!
//! * [`actions_from_operation`] — pure, takes only the `Operation`. Works
//!   for translations whose action list can be computed from the operation
//!   payload alone.
//! * [`actions_from_operation_with_manifest`] — takes the prior manifest in
//!   addition. Required for operations whose action list depends on the
//!   current schema/fragment/index state (notably `Merge`, which diffs new
//!   fields against existing ones, and stable-row-id `Update`/`Merge` which
//!   need to emit a trailing [`RefreshRowVersionMetadata`]).
//!
//! Either entry point returns `Option<UserAction>`. `None` is a **decline**:
//! the translator does not handle this payload shape, and the caller falls
//! back to the legacy per-`Operation` `build_manifest` arm. Declines are
//! audited to error-on-both-paths so the fallback is defensive, not
//! load-bearing.
//!
//! # Round-trip contract
//!
//! Some translations are forward-only:
//!
//! * `Delete` — [`UpdateDeletionVector`] drops the rest of the `Fragment`,
//!   so a `Delete` cannot be rebuilt from its actions.
//! * `Merge` — [`AddFields`] keeps only the *added* fields and files; the
//!   prior manifest is needed to rebuild a full `Merge`.
//! * `CreateIndex` that removed indices — [`RemoveIndex`] keeps only a UUID,
//!   so the original `CreateIndex` cannot be rebuilt.
//!
//! All other operations round-trip both ways via [`operation_from_actions`],
//! since their actions carry the entire legacy payload.
//!
//! # Ordering and decomposition rules
//!
//! The op→action decomposition is in design doc §4. A few non-obvious
//! orderings are load-bearing and live here:
//!
//! * Overwrite emits `RemoveFragments → UpdateConfig? → ChangeSchema → AddFragments`.
//!   `UpdateConfig` must precede `ChangeSchema` so the schema validator sees
//!   the new config; `AddFragments` must follow `ChangeSchema` so its
//!   inline validate sees the new schema.
//! * Stable-row-id `Update::RewriteRows` orders `RemoveFragments` *last*, so
//!   the trailing `RefreshRowVersionMetadata` can still trace `created_at`
//!   through the about-to-be-emptied source fragments.
//!
//! When adding a translation, the rule of thumb: actions whose `apply`
//! depends on state another action in the same transaction produces must
//! come later in the list (the driver's per-action validate runs against
//! the in-progress manifest — see [`conflict`](super::super::conflict)).

use std::collections::{HashMap, HashSet};

use lance_core::datatypes::{Field, Schema};
use lance_index::mem_wal::MergedGeneration;
use lance_table::format::{DataFile, Fragment, IndexMetadata, Manifest};

use super::super::{
    DataReplacementGroup, Operation, RewriteGroup, RewrittenIndex, UpdateMode,
    UpdatedFragmentOffsets, is_pure_rewrite_fragment, recalculate_fragment_bitmap,
    translate_config_updates,
};
use super::{
    Action, AddBases, AddFields, AddFragments, AddIndex, ChangeSchema, ColumnReplacement,
    FragmentCoverage, FragmentSelector, IndexCoverageRemap, InvalidateIndexCoverage,
    RebindIndexCoverage, RefreshRowVersionMetadata, RegisterFragReuse, RemoveFragments,
    RemoveIndex, ReplaceFragmentColumns, ReserveFragmentIds, RewriteFragments, RewriteGroupRemap,
    RewriteIndex, RowVersionRefresh, UpdateConfig, UpdateDeletionVector, UpdateMergedGenerations,
    UpdateSchemaMetadata, UserAction, collect_field_ids,
};
use crate::dataset::write::merge_insert::inserted_rows::KeyExistenceFilter;

/// Translate an old [`Operation`] into a [`UserAction`].
///
/// The `description` field on the returned [`UserAction`] carries any
/// operation-level metadata that has no native action representation yet
/// (e.g. [`Operation::Delete::predicate`]). This keeps the translation
/// lossless even though the action vocabulary is still being built out.
///
/// Returns `None` for variants that have not yet been ported. Callers
/// should fall back to the old per-Operation code path in that case.
pub fn actions_from_operation(operation: &Operation) -> Option<UserAction> {
    match operation {
        Operation::Append { fragments } => Some(UserAction::new(vec![Box::new(AddFragments {
            fragments: fragments.clone(),
            inserted_rows_filter: None,
        })])),
        Operation::Delete {
            updated_fragments,
            deleted_fragment_ids,
            predicate,
        } => {
            let mut actions: Vec<Box<dyn Action>> = Vec::with_capacity(updated_fragments.len() + 1);
            for fragment in updated_fragments {
                // A delete that leaves rows in a fragment must attach a new
                // deletion file to it. A missing deletion file means the
                // operation is malformed for this translation — return `None`
                // so the caller falls back to the legacy per-Operation path.
                let new_deletion_file = fragment.deletion_file.clone()?;
                actions.push(Box::new(UpdateDeletionVector {
                    fragment_id: fragment.id,
                    new_deletion_file,
                    affected_rows: None,
                }));
            }
            if !deleted_fragment_ids.is_empty() {
                actions.push(Box::new(RemoveFragments {
                    selector: FragmentSelector::Ids(deleted_fragment_ids.clone()),
                }));
            }
            Some(UserAction::new(actions).with_description(predicate.clone()))
        }
        Operation::Project { schema } => {
            // Project narrows the schema; `ChangeSchema` records the
            // wholesale-replace intent that last-writer-wins relies on. The
            // dead-data-file pruning Project performs is reproduced by
            // `ChangeSchema::apply`.
            Some(UserAction::new(vec![Box::new(ChangeSchema {
                schema: schema.clone(),
            })]))
        }
        Operation::DataReplacement { replacements } => {
            // One `ReplaceFragmentColumns` per `DataReplacementGroup`. The
            // replaced field set is inferred from the swapped `DataFile`;
            // `ReplaceFragmentColumns::validate` resolves the swap against the
            // current manifest at apply time, so no prior state is needed here.
            let actions: Vec<Box<dyn Action>> = replacements
                .iter()
                .map(
                    |DataReplacementGroup(fragment_id, new_file)| -> Box<dyn Action> {
                        Box::new(ReplaceFragmentColumns {
                            fragment_id: *fragment_id,
                            field_ids: new_file.fields.to_vec(),
                            new_data_files: vec![new_file.clone()],
                            updated_row_offsets: None,
                            replacement: ColumnReplacement::InPlace,
                        })
                    },
                )
                .collect();
            Some(UserAction::new(actions))
        }
        Operation::CreateIndex {
            new_indices,
            removed_indices,
        } => {
            // `build_manifest` drops every index whose UUID is in the removed
            // *or* the new set, then appends the new ones. `RemoveIndex` must
            // therefore come before `AddIndex` so a UUID present in both ends
            // up added, not removed.
            let mut actions: Vec<Box<dyn Action>> =
                Vec::with_capacity(removed_indices.len() + new_indices.len());
            for removed in removed_indices {
                actions.push(Box::new(RemoveIndex { uuid: removed.uuid }));
            }
            for new_index in new_indices {
                // The legacy `CreateIndex` freezes coverage into the index's
                // `fragment_bitmap`. A regular column index without one is a
                // legacy on-disk shape we cannot express; decline. A
                // *system* index without a bitmap is the inline-coverage
                // shape (the MemWAL system index uses this — it covers no
                // data fragments), so emit `Unbound` to preserve the `None`
                // bitmap through `apply`.
                let coverage = match new_index.fragment_bitmap.clone() {
                    Some(bitmap) => FragmentCoverage::Bitmap(bitmap),
                    None if lance_index::is_system_index(new_index) => FragmentCoverage::Unbound,
                    None => return None,
                };
                actions.push(Box::new(AddIndex {
                    index: new_index.clone(),
                    coverage,
                    // Translation is sync and cannot load the details file;
                    // PR B's action resolver will populate this via an async
                    // enrichment step before rebase runs.
                    frag_reuse_versions: None,
                }));
            }
            Some(UserAction::new(actions))
        }
        Operation::UpdateMemWalState { merged_generations } => {
            Some(UserAction::new(vec![Box::new(UpdateMergedGenerations {
                merged_generations: merged_generations.clone(),
            })]))
        }
        Operation::Rewrite { .. } => {
            // `Rewrite`'s index half needs the dataset's stable-row-id flag
            // and the pre-rewrite coverage bitmaps — neither is derivable
            // from the `Operation` alone, so the translation lives in
            // `actions_from_operation_with_manifest`.
            None
        }
        Operation::Update { .. } => {
            // `Update` needs the dataset's stable-row-id flag to decide
            // whether it can be expressed as actions at all — the translation
            // lives in `actions_from_operation_with_manifest`.
            None
        }
        Operation::UpdateConfig {
            config_updates,
            table_metadata_updates,
            schema_metadata_updates,
            field_metadata_updates,
        } => {
            // Always emit an `UpdateConfig` action — even when both halves are
            // empty — so the action list has the fixed `[UpdateConfig]` /
            // `[UpdateConfig, UpdateSchemaMetadata]` shape `operation_from_actions`
            // matches on. The schema- and field-metadata halves split onto an
            // `UpdateSchemaMetadata` action, emitted only when at least one is
            // non-empty.
            let mut actions: Vec<Box<dyn Action>> = vec![Box::new(UpdateConfig {
                config_updates: config_updates.clone(),
                table_metadata_updates: table_metadata_updates.clone(),
            })];
            if schema_metadata_updates.is_some() || !field_metadata_updates.is_empty() {
                actions.push(Box::new(UpdateSchemaMetadata {
                    metadata_updates: schema_metadata_updates.clone(),
                    field_metadata_updates: field_metadata_updates.clone(),
                }));
            }
            Some(UserAction::new(actions))
        }
        Operation::UpdateBases { new_bases } => Some(UserAction::new(vec![Box::new(AddBases {
            bases: new_bases.clone(),
        })])),
        Operation::ReserveFragments { num_fragments } => {
            Some(UserAction::new(vec![Box::new(ReserveFragmentIds {
                num_fragments: *num_fragments,
            })]))
        }
        Operation::Overwrite {
            fragments,
            schema,
            config_upsert_values,
            initial_bases,
        } => {
            // `initial_bases` is populated only when an Overwrite *creates* a
            // dataset; that CREATE path stays on the legacy `build_manifest`
            // arm, so an action-routed Overwrite always has `None` here. Fall
            // back to the legacy path if it is set.
            if initial_bases.is_some() {
                return None;
            }
            // §4 decomposition: wipe every current fragment, apply the config
            // upsert, replace the schema wholesale, then add the new fragments.
            // `AllCurrent` re-resolves at apply time, so a fragment a concurrent
            // `AddFragments` committed is wiped too (spike #6448 §9). The
            // decomposition's `AddFields?` arm is never emitted — a wholesale
            // `ChangeSchema` already carries the new schema, and Overwrite
            // removes every fragment first, so there is no existing fragment
            // for `AddFields` to add column data to.
            //
            // `UpdateConfig` is emitted *before* `ChangeSchema` deliberately.
            // The `commit` rebase driver short-circuits at the first
            // conflicting action; `ChangeSchema::rebase` always reports a
            // concurrent `Overwrite` retryable, so an `UpdateConfig` placed
            // after it would never be rebased. Ordering it first lets a
            // config-key collision surface as the incompatible conflict the
            // legacy `check_overwrite_txn` raises.
            let mut actions: Vec<Box<dyn Action>> = vec![Box::new(RemoveFragments {
                selector: FragmentSelector::AllCurrent,
            })];
            // `config_upsert_values` is a plain key upsert (the legacy arm
            // does `config_mut().extend`); translate it to an incremental
            // `UpdateConfig`.
            if let Some(config_upsert_values) = config_upsert_values {
                actions.push(Box::new(UpdateConfig {
                    config_updates: Some(translate_config_updates(config_upsert_values, &[])),
                    table_metadata_updates: None,
                }));
            }
            actions.push(Box::new(ChangeSchema {
                schema: schema.clone(),
            }));
            actions.push(Box::new(AddFragments {
                fragments: fragments.clone(),
                inserted_rows_filter: None,
            }));
            Some(UserAction::new(actions))
        }
        // `Merge` needs the prior manifest to diff the added fields and data
        // files; see `actions_from_operation_with_manifest`.
        _ => None,
    }
}

/// Translate an [`Operation`] into a [`UserAction`], using the prior `manifest`
/// and `indices` as the state for operations whose action diff cannot be
/// computed from the [`Operation`] alone.
///
/// [`Operation::Merge`] carries the *final* schema and fragment list, while the
/// [`AddFields`] action carries only the *added* fields and the *new*
/// per-fragment data files — so the diff is taken against `manifest`.
/// [`Operation::Rewrite`] and [`Operation::Update`] need the stable-row-id flag
/// from `manifest` to decide whether they can be expressed as actions, and
/// `Rewrite` needs the pre-rewrite coverage bitmaps from `indices`. Operations
/// that need neither are delegated to [`actions_from_operation`].
///
/// Returns `None` when [`actions_from_operation`] would, when a `Merge` is
/// not a pure column add — it removes or modifies a field, rewrites or removes
/// a data file, adds or drops a fragment, or changes schema metadata — for
/// the `Rewrite` cases [`rewrite_to_actions`] declines, and for the `Update`
/// cases [`update_to_actions`] declines. Such an operation cannot be expressed
/// as actions, so the caller falls back to the legacy per-Operation path.
pub fn actions_from_operation_with_manifest(
    operation: &Operation,
    manifest: &Manifest,
    indices: &[IndexMetadata],
) -> Option<UserAction> {
    match operation {
        Operation::Merge { fragments, schema } => merge_to_add_fields(fragments, schema, manifest),
        Operation::Rewrite {
            groups,
            rewritten_indices,
            frag_reuse_index,
        } => rewrite_to_actions(
            groups,
            rewritten_indices,
            frag_reuse_index,
            manifest,
            indices,
        ),
        Operation::Update {
            removed_fragment_ids,
            updated_fragments,
            new_fragments,
            fields_modified,
            merged_generations,
            fields_for_preserving_frag_bitmap,
            update_mode,
            inserted_rows_filter,
            updated_fragment_offsets,
        } => update_to_actions(
            removed_fragment_ids,
            updated_fragments,
            new_fragments,
            fields_modified,
            merged_generations,
            fields_for_preserving_frag_bitmap,
            update_mode,
            inserted_rows_filter,
            updated_fragment_offsets,
            manifest,
        ),
        other => actions_from_operation(other),
    }
}

/// Translate an [`Operation::Rewrite`] into a [`UserAction`].
///
/// The decomposition (spike #6448 §4) is one [`RewriteFragments`] per rewrite
/// group, followed by the index half:
///
/// * on a stable-row-id dataset the rewrite preserves row ids, so index data
///   stays valid and only the fragment ids move — a single
///   [`RebindIndexCoverage`] re-points every index's coverage bitmap;
/// * otherwise each rewritten index is replaced by a [`RewriteIndex`] carrying
///   the coverage bitmap recalculated against the rewrite groups.
///
/// A trailing [`RegisterFragReuse`] is emitted when the rewrite registers a
/// fragment-reuse index.
///
/// Returns `None` — deferring to the legacy `build_manifest` arm — when the
/// rewrite has no groups, when a stable-row-id rewrite carries
/// `rewritten_indices` (the legacy arm asserts it does not), or when a
/// rewritten index has no stored coverage bitmap to recalculate.
fn rewrite_to_actions(
    groups: &[RewriteGroup],
    rewritten_indices: &[RewrittenIndex],
    frag_reuse_index: &Option<IndexMetadata>,
    manifest: &Manifest,
    indices: &[IndexMetadata],
) -> Option<UserAction> {
    // A rewrite with no groups changes no fragments — nothing to translate.
    if groups.is_empty() {
        return None;
    }

    let preserves_row_ids = manifest.uses_stable_row_ids();

    let mut actions: Vec<Box<dyn Action>> =
        Vec::with_capacity(groups.len() + rewritten_indices.len() + 1);
    for group in groups {
        actions.push(Box::new(RewriteFragments {
            old_fragment_ids: group.old_fragments.iter().map(|f| f.id).collect(),
            new_fragments: group.new_fragments.clone(),
            preserves_row_ids,
        }));
    }

    if preserves_row_ids {
        // A stable-row-id rewrite reuses index data wholesale; the legacy
        // `build_manifest` arm asserts it carries no `rewritten_indices`.
        if !rewritten_indices.is_empty() {
            return None;
        }
        actions.push(Box::new(RebindIndexCoverage {
            remap: IndexCoverageRemap::RewriteGroups(rewrite_groups_remap(groups)),
            // Compaction rewrites the physical layout, not any column's data,
            // so no field is modified and every index is rebound.
            modified_field_ids: Vec::new(),
        }));
    } else {
        for rewritten in rewritten_indices {
            // An index missing from `indices` was dropped concurrently; the
            // legacy `handle_rewrite_indices` skips it and `RewriteIndex::apply`
            // does too, so emit nothing for it.
            let Some(old_index) = indices.iter().find(|idx| idx.uuid == rewritten.old_id) else {
                continue;
            };
            // An index with no stored coverage bitmap cannot be rewritten;
            // defer to the legacy arm, which raises the descriptive error.
            let old_bitmap = old_index.fragment_bitmap.as_ref()?;
            let new_fragment_bitmap = recalculate_fragment_bitmap(old_bitmap, groups).ok()?;
            actions.push(Box::new(RewriteIndex {
                old_uuid: rewritten.old_id,
                new_uuid: rewritten.new_id,
                new_index_files: rewritten.new_index_files.clone(),
                new_fragment_bitmap,
            }));
        }
    }

    if let Some(frag_reuse_index) = frag_reuse_index {
        actions.push(Box::new(RegisterFragReuse {
            frag_reuse_index: frag_reuse_index.clone(),
            // PR B's action resolver populates this via an async enrichment
            // step before rebase runs.
            frag_reuse_versions: None,
        }));
    }

    Some(UserAction::new(actions))
}

/// Build the per-group fragment-id remap for the [`RebindIndexCoverage`] that
/// follows a stable-row-id rewrite.
///
/// Each [`RewriteGroupRemap`] keeps the group's old and new fragment ids
/// verbatim; [`Action::apply`] then performs the all-or-nothing recalculation
/// [`Transaction::recalculate_fragment_bitmap`] does. Because the remap stores
/// both id sets in full it expresses arbitrary cardinality — a group that
/// splits one fragment into many, or compacts a fully-deleted fragment into
/// none — so this never declines.
fn rewrite_groups_remap(groups: &[RewriteGroup]) -> Vec<RewriteGroupRemap> {
    groups
        .iter()
        .map(|group| RewriteGroupRemap {
            old_fragment_ids: group.old_fragments.iter().map(|f| f.id).collect(),
            new_fragment_ids: group.new_fragments.iter().map(|f| f.id).collect(),
        })
        .collect()
}

/// Translate an [`Operation::Update`] into a [`UserAction`].
///
/// Both `update_mode`s are handled (spike #6448 §4, #6843); the mode is encoded
/// structurally — which actions are emitted — rather than carried on any action.
///
/// A `RewriteRows` update deletes rows from existing fragments and rewrites them
/// into new ones, so the action list is
///
/// * `RemoveFragments(Ids)` for the fully-emptied `removed_fragment_ids`;
/// * one `UpdateDeletionVector` per `updated_fragments` entry — the leftover
///   partial delete that emptied some, but not all, of a fragment's rows;
/// * `AddFragments` carrying `new_fragments` and the `inserted_rows_filter`
///   merge-insert conflict metadata;
/// * `InvalidateIndexCoverage` dropping the updated fragments from every index
///   that covers a modified field;
/// * on a stable-row-id dataset, a `RebindIndexCoverage` in
///   [`IndexCoverageRemap::InsertPureRewrite`] mode registering the
///   pure-rewrite fragments into the coverage of every index that already
///   spanned all the rewritten rows (legacy
///   `register_pure_rewrite_rows_update_frags_in_indices`);
/// * a trailing `UpdateMergedGenerations` when MemWAL generations merged.
///
/// A `RewriteColumns` update rewrites whole columns in place, so each
/// `updated_fragments` entry becomes one `ReplaceFragmentColumns` carrying the
/// data files that differ from the manifest. `ReplaceFragmentColumns::apply`
/// invalidates the covering index entries itself — its `field_ids` is set to
/// `fields_modified`, so the union over every emitted action reproduces
/// `prune_updated_fields_from_indices` without a separate
/// `InvalidateIndexCoverage`. `RemoveFragments` / `AddFragments` /
/// `UpdateMergedGenerations` are emitted as for `RewriteRows`.
///
/// On a stable-row-id dataset the update also refreshes per-row version
/// metadata; a trailing [`RefreshRowVersionMetadata`] reproduces the legacy
/// `build_manifest` refresh ([`RowVersionRefresh::UpdatedRows`] for the freshly
/// written fragments, [`RowVersionRefresh::RewrittenColumns`] for the partially
/// rewritten ones). The `RemoveFragments` is then emitted *last* — after the
/// refresh — because [`RowVersionRefresh::UpdatedRows`] traces each rewritten
/// row's `created_at` through the emptied source fragments, which must still be
/// in the manifest when it applies.
///
/// Returns `None` — deferring to the legacy `build_manifest` arm — for a
/// malformed `RewriteRows` update whose `updated_fragments` carry no deletion
/// file (the same guard as the `Delete` translation), for a `RewriteColumns`
/// update whose fragment diff is not a pure column swap/append (see
/// [`rewrite_columns_replace_for_fragment`]), and for a stable-row-id
/// `RewriteRows` update whose `new_fragments` lack `physical_rows` (the
/// pure-rewrite classification cannot run).
#[allow(clippy::too_many_arguments)]
fn update_to_actions(
    removed_fragment_ids: &[u64],
    updated_fragments: &[Fragment],
    new_fragments: &[Fragment],
    fields_modified: &[u32],
    merged_generations: &[MergedGeneration],
    fields_for_preserving_frag_bitmap: &[u32],
    update_mode: &Option<UpdateMode>,
    inserted_rows_filter: &Option<KeyExistenceFilter>,
    updated_fragment_offsets: &Option<UpdatedFragmentOffsets>,
    manifest: &Manifest,
) -> Option<UserAction> {
    let stable_row_ids = manifest.uses_stable_row_ids();
    let rewrite_columns = matches!(update_mode, Some(UpdateMode::RewriteColumns));
    let rewrite_rows = matches!(update_mode, Some(UpdateMode::RewriteRows));

    let mut actions: Vec<Box<dyn Action>> = Vec::with_capacity(updated_fragments.len() + 5);
    // The `RemoveFragments` for the emptied source fragments. On a stable-row-id
    // dataset it is emitted last (the `UpdatedRows` refresh traces `created_at`
    // through those fragments); otherwise it is dropped up front, as the legacy
    // arm does.
    let make_remove_fragments = || {
        Box::new(RemoveFragments {
            selector: FragmentSelector::Ids(removed_fragment_ids.to_vec()),
        })
    };
    if !stable_row_ids && !removed_fragment_ids.is_empty() {
        actions.push(make_remove_fragments());
    }
    for fragment in updated_fragments {
        // A rebased Update can list the same fragment in both
        // `removed_fragment_ids` and `updated_fragments`: when a concurrent
        // commit's deletes empty a fragment outright, the conflict resolver
        // appends it to `removed_fragment_ids` without pruning the now-stale
        // `updated_fragments` entry. The legacy `build_manifest` arm checks
        // `removed_fragment_ids` first, so removal wins; mirror that by
        // skipping the stale entry — its `RemoveFragments` is already queued.
        if removed_fragment_ids.contains(&fragment.id) {
            continue;
        }
        if rewrite_columns {
            actions.push(rewrite_columns_replace_for_fragment(
                fragment,
                fields_modified,
                manifest,
            )?);
        } else {
            // A `RewriteRows` update only deletes rows from an existing
            // fragment, so every updated fragment must carry a fresh deletion
            // file. A missing one means the operation is malformed for this
            // translation — return `None` and fall back to the legacy path.
            let new_deletion_file = fragment.deletion_file.clone()?;
            actions.push(Box::new(UpdateDeletionVector {
                fragment_id: fragment.id,
                new_deletion_file,
                affected_rows: None,
            }));
        }
    }
    if !new_fragments.is_empty() {
        actions.push(Box::new(AddFragments {
            fragments: new_fragments.to_vec(),
            inserted_rows_filter: inserted_rows_filter.clone(),
        }));
    }
    // `AddFragments::apply` allocates the id-0 `new_fragments` from the
    // manifest's fragment-id high-water mark. Mirror that allocation here so the
    // later actions can name the freshly written fragments by their committed
    // ids — valid because `build_manifest_via_actions` runs this translation and
    // `apply` in one pass off the same manifest, with no action removing a
    // fragment before `AddFragments`.
    let new_fragment_ids: Vec<u64> = {
        let mut next_id = manifest.max_fragment_id().map_or(0, |id| id + 1);
        new_fragments
            .iter()
            .map(|fragment| {
                if fragment.id == 0 {
                    let id = next_id;
                    next_id += 1;
                    id
                } else {
                    fragment.id
                }
            })
            .collect()
    };
    // `RewriteColumns` invalidates index coverage inside each
    // `ReplaceFragmentColumns::apply`; `RewriteRows` needs an explicit action.
    if !rewrite_columns && !fields_modified.is_empty() {
        // Mirror `Transaction::prune_updated_fields_from_indices`: drop every
        // updated fragment from the bitmap of each index over a modified field.
        actions.push(Box::new(InvalidateIndexCoverage {
            fragment_ids: updated_fragments.iter().map(|f| f.id).collect(),
            field_ids: fields_modified.iter().map(|&id| id as i32).collect(),
        }));
    }
    // A stable-row-id `RewriteRows` update re-points index coverage onto the
    // pure-rewrite fragments it produced, reproducing the legacy
    // `register_pure_rewrite_rows_update_frags_in_indices`. Only fragments all
    // of whose rows carry a stable row id qualify (the merge-insert fragments
    // hold freshly inserted, unindexed rows); a missing `physical_rows` makes
    // that classification impossible, so the whole update defers to the legacy
    // arm.
    if stable_row_ids && rewrite_rows {
        // Of the freshly written fragments, the pure-rewrite ones (every row
        // carries a stable id) get index coverage re-pointed onto them.
        let mut pure_rewrite_ids = Vec::new();
        for (fragment, &assigned_id) in new_fragments.iter().zip(&new_fragment_ids) {
            if is_pure_rewrite_fragment(fragment).ok()? {
                pure_rewrite_ids.push(assigned_id);
            }
        }
        if !pure_rewrite_ids.is_empty() {
            // The gate: every original fragment (the emptied `removed` plus the
            // still-present `updated`) whose rows were rewritten.
            let original_fragment_ids: Vec<u64> = removed_fragment_ids
                .iter()
                .chain(updated_fragments.iter().map(|f| &f.id))
                .copied()
                .collect();
            actions.push(Box::new(RebindIndexCoverage {
                remap: IndexCoverageRemap::InsertPureRewrite {
                    new_fragment_ids: pure_rewrite_ids,
                    original_fragment_ids,
                },
                modified_field_ids: fields_for_preserving_frag_bitmap
                    .iter()
                    .map(|&id| id as i32)
                    .collect(),
            }));
        }
    }
    if !merged_generations.is_empty() {
        actions.push(Box::new(UpdateMergedGenerations {
            merged_generations: merged_generations.to_vec(),
        }));
    }
    // On a stable-row-id dataset, refresh per-row version metadata.
    // `UpdatedRows` traces `created_at` from the source fragments — including
    // the emptied `removed_fragment_ids`, whose `RemoveFragments` is therefore
    // deferred to the end of the action list. `RewrittenColumns` is a
    // `RewriteColumns`-only refresh; its targets are disjoint from
    // `UpdatedRows`' freshly written ones, so the two refreshes commute.
    if stable_row_ids {
        if !new_fragments.is_empty() {
            actions.push(Box::new(RefreshRowVersionMetadata {
                mode: RowVersionRefresh::UpdatedRows { new_fragment_ids },
            }));
        }
        if rewrite_columns
            && let Some(UpdatedFragmentOffsets(off_map)) = updated_fragment_offsets
            && !off_map.is_empty()
        {
            // Sort by fragment id so the action is deterministic regardless of
            // the source `HashMap`'s iteration order.
            let mut touched_offsets: Vec<(u64, Vec<u64>)> = off_map
                .iter()
                .map(|(id, bitmap)| (*id, bitmap.iter().map(u64::from).collect()))
                .collect();
            touched_offsets.sort_by_key(|(id, _)| *id);
            actions.push(Box::new(RefreshRowVersionMetadata {
                mode: RowVersionRefresh::RewrittenColumns { touched_offsets },
            }));
        }
    }
    if stable_row_ids && !removed_fragment_ids.is_empty() {
        actions.push(make_remove_fragments());
    }
    Some(UserAction::new(actions))
}

/// Diff one `RewriteColumns`-mode `updated_fragments` entry against the manifest
/// fragment of the same id, producing the [`ReplaceFragmentColumns`]
/// that reproduces the column rewrite.
///
/// A `RewriteColumns` update has two file shapes:
///
/// * [`ColumnReplacement::InPlace`] — each rewritten field already lives in its
///   own data file, so the rewrite swaps that file in place (and, for an
///   all-NULL column gaining real data, appends a new file). The action
///   carries exactly the data files that differ from the manifest.
/// * [`ColumnReplacement::Tombstone`] — any other layout: a rewritten field is
///   tombstoned out of a data file it shared with an untouched one. The action
///   carries the fragment's complete post-rewrite file list, which `apply`
///   installs verbatim — re-deriving the layout is not possible because the
///   `RewriteColumns` producers disagree on whether a fully-tombstoned file is
///   dropped or kept.
///
/// Either way `field_ids` is `fields_modified`, so `ReplaceFragmentColumns::apply`
/// invalidates the same index entries the legacy
/// `prune_updated_fields_from_indices` does — even when a listed fragment had
/// no file change, the action is still emitted so its coverage is dropped.
///
/// Returns `None` — so the caller falls back to the legacy path — when the
/// fragment is missing from the manifest, or when it changed in any way beyond
/// its data files (a deletion vector or row-id change is not a column rewrite).
fn rewrite_columns_replace_for_fragment(
    updated: &Fragment,
    fields_modified: &[u32],
    manifest: &Manifest,
) -> Option<Box<dyn Action>> {
    let prior = manifest.fragments.iter().find(|f| f.id == updated.id)?;
    // Apart from its data files, the fragment must be untouched — a deletion
    // vector or row-id change is not a column rewrite.
    let mut without_files = updated.clone();
    without_files.files = prior.files.clone();
    if &without_files != prior {
        return None;
    }

    let field_ids: Vec<i32> = fields_modified.iter().map(|&id| id as i32).collect();
    // Prefer the in-place swap (it round-trips back to `DataReplacement`); fall
    // back to carrying the whole post-rewrite file list for any other layout.
    let (replacement, new_data_files) = match in_place_column_diff(updated, prior) {
        Some(files) => (ColumnReplacement::InPlace, files),
        None => (ColumnReplacement::Tombstone, updated.files.clone()),
    };

    Some(Box::new(ReplaceFragmentColumns {
        fragment_id: updated.id,
        field_ids,
        new_data_files,
        // The legacy non-stable `RewriteColumns` path ignores
        // `updated_fragment_offsets`, and row-level conflict resolution does
        // not consume `updated_row_offsets` yet; leave it `None` as the
        // `Operation::DataReplacement` translation does.
        updated_row_offsets: None,
        replacement,
    }))
}

/// The [`ColumnReplacement::InPlace`] half of [`rewrite_columns_replace_for_fragment`]:
/// the rewrite is a positional file swap plus disjoint all-NULL appends.
///
/// Returns the data files that differ from the manifest, or `None` when the
/// diff is not a pure swap/append — a removed or reordered file, a change to a
/// file's field layout or format version, or a trailing file overlapping an
/// already-covered field.
fn in_place_column_diff(updated: &Fragment, prior: &Fragment) -> Option<Vec<DataFile>> {
    // A dropped file cannot be expressed as an in-place swap.
    if updated.files.len() < prior.files.len() {
        return None;
    }
    let mut new_data_files = Vec::new();
    // The leading files line up positionally with the manifest's. A swapped
    // file differs only by `path`/`file_size_bytes`; a change to its field
    // layout or format version cannot be reproduced by a swap.
    for (new_file, old_file) in updated.files.iter().zip(prior.files.iter()) {
        if new_file.fields != old_file.fields
            || new_file.column_indices != old_file.column_indices
            || new_file.file_major_version != old_file.file_major_version
            || new_file.file_minor_version != old_file.file_minor_version
            || new_file.base_id != old_file.base_id
        {
            return None;
        }
        if new_file != old_file {
            new_data_files.push(new_file.clone());
        }
    }
    // Trailing files are appended — an all-NULL column gaining real data — so
    // they must reference only fields no existing data file covers.
    let covered: HashSet<i32> = prior
        .files
        .iter()
        .flat_map(|f| f.fields.iter().copied())
        .collect();
    for appended in &updated.files[prior.files.len()..] {
        if appended.fields.iter().any(|id| covered.contains(id)) {
            return None;
        }
        new_data_files.push(appended.clone());
    }
    Some(new_data_files)
}

/// Translate an [`Operation::Merge`] into the equivalent action list.
///
/// A pure column-add merge becomes a single [`AddFields`] (see
/// [`merge_pure_column_add`]). The `alter_columns` data-type-cast shape — a
/// prior field replaced by a recast version under a fresh field id, its data
/// rewritten into new files, and the now-dead files dropped — becomes
/// `[AddFields, ChangeSchema]` (see [`merge_recast_columns`]). On a
/// stable-row-id dataset either is followed by a [`RefreshRowVersionMetadata`].
///
/// Returns `None` for any merge neither path can express.
fn merge_to_add_fields(
    fragments: &[Fragment],
    schema: &Schema,
    manifest: &Manifest,
) -> Option<UserAction> {
    merge_pure_column_add(fragments, schema, manifest)
        .or_else(|| merge_recast_columns(fragments, schema, manifest))
}

/// Diff a pure column-add [`Operation::Merge`]'s final `schema` + `fragments`
/// against `manifest` to build the equivalent [`AddFields`].
///
/// On a stable-row-id dataset a trailing [`RefreshRowVersionMetadata`] with
/// [`RowVersionRefresh::MergedColumns`] reproduces the legacy `Operation::Merge`
/// refresh of `last_updated_at_version_meta` for every fragment that gained a
/// data file.
///
/// Returns `None` for any merge that is not a pure append of new top-level
/// columns: a removed or modified prior field, changed schema metadata, a
/// changed fragment set, a rewritten or removed data file, or a new data file
/// that references a field other than the ones being added. The `alter_columns`
/// type-cast shape is caught by [`merge_recast_columns`] instead.
fn merge_pure_column_add(
    fragments: &[Fragment],
    schema: &Schema,
    manifest: &Manifest,
) -> Option<UserAction> {
    // The merge must keep every prior field untouched and only append new
    // top-level fields; schema metadata must be unchanged.
    if !schema.fields.starts_with(&manifest.schema.fields)
        || schema.metadata != manifest.schema.metadata
    {
        return None;
    }
    let new_fields = schema.fields[manifest.schema.fields.len()..].to_vec();
    let mut new_field_ids = HashSet::new();
    collect_field_ids(&new_fields, &mut new_field_ids);

    // A pure column add neither adds nor removes fragments.
    if fragments.len() != manifest.fragments.len() {
        return None;
    }
    let prior_by_id: HashMap<u64, &Fragment> =
        manifest.fragments.iter().map(|f| (f.id, f)).collect();

    let mut fragment_files = Vec::new();
    for fragment in fragments {
        let prior = prior_by_id.get(&fragment.id)?;
        // Every prior data file must survive verbatim; the merge may only
        // append new files after them.
        if !fragment.files.starts_with(&prior.files) {
            return None;
        }
        // Apart from the appended files, the fragment must be untouched.
        let mut truncated = fragment.clone();
        truncated.files = prior.files.clone();
        if &truncated != *prior {
            return None;
        }
        let new_files = fragment.files[prior.files.len()..].to_vec();
        // New data files may only carry the freshly added fields.
        if new_files
            .iter()
            .flat_map(|file| file.fields.iter())
            .any(|id| !new_field_ids.contains(id))
        {
            return None;
        }
        if !new_files.is_empty() {
            fragment_files.push((fragment.id, new_files));
        }
    }

    // On a stable-row-id dataset the merge also bumps `last_updated` for every
    // fragment that gained a data file (`merge_fragment_physically_rewritten`
    // is true exactly when the file count grew, i.e. for the entries in
    // `fragment_files`). A pure column add introduces no brand-new fragments,
    // so only `last_updated` is refreshed — no `created_at` branch.
    let refreshed_fragment_ids: Vec<u64> = if manifest.uses_stable_row_ids() {
        fragment_files.iter().map(|(id, _)| *id).collect()
    } else {
        Vec::new()
    };

    let mut actions: Vec<Box<dyn Action>> = vec![Box::new(AddFields {
        new_fields,
        fragment_files,
    })];
    if !refreshed_fragment_ids.is_empty() {
        actions.push(Box::new(RefreshRowVersionMetadata {
            mode: RowVersionRefresh::MergedColumns {
                fragment_ids: refreshed_fragment_ids,
            },
        }));
    }
    Some(UserAction::new(actions))
}

/// Translate the `alter_columns` data-type-cast [`Operation::Merge`] into
/// `[AddFields, ChangeSchema]`.
///
/// A type-cast `alter_columns` is not a pure column add: it replaces a field
/// with a recast version under a fresh field id, writes the recast data into
/// new per-fragment files, and drops data files left covering no live field —
/// so [`merge_pure_column_add`]'s `starts_with` checks reject it. The decomposition:
///
/// * [`AddFields`] appends the recast columns' new data files. Its
///   `new_fields` only has to satisfy [`AddFields::validate`] — the schema
///   `ChangeSchema` installs next overwrites it — so it carries one flat
///   [`Field`] per new id the new files reference.
/// * [`ChangeSchema`] installs the merge's final schema (recast fields
///   at their original positions, old ids gone) and prunes the now-dead files,
///   exactly mirroring the legacy `Merge` arm's wholesale schema + fragment
///   replacement followed by `remove_tombstoned_data_files`.
///
/// On a stable-row-id dataset a trailing [`RefreshRowVersionMetadata`] bumps
/// `last_updated` for every fragment whose data files changed, reproducing the
/// legacy `merge_fragment_physically_rewritten` refresh.
///
/// Returns `None` when the merge is not this shape: no field added, a changed
/// fragment set, or a prior data file modified rather than kept-or-dropped.
fn merge_recast_columns(
    fragments: &[Fragment],
    schema: &Schema,
    manifest: &Manifest,
) -> Option<UserAction> {
    // `alter_columns` rewrites the dataset's own fragments — adding and
    // dropping files but never adding or removing a fragment.
    if fragments.len() != manifest.fragments.len() {
        return None;
    }
    let prior_by_id: HashMap<u64, &Fragment> =
        manifest.fragments.iter().map(|f| (f.id, f)).collect();

    let prior_field_ids: HashSet<i32> = manifest.schema.fields_pre_order().map(|f| f.id).collect();
    let added_ids: HashSet<i32> = schema
        .fields_pre_order()
        .map(|f| f.id)
        .filter(|id| !prior_field_ids.contains(id))
        .collect();
    // A merge that adds no field is a pure schema narrowing, not the recast
    // shape; `alter_columns` emits `Operation::Project` for those, so leave it
    // to the legacy path.
    if added_ids.is_empty() {
        return None;
    }

    let mut fragment_files: Vec<(u64, Vec<DataFile>)> = Vec::new();
    let mut refreshed_fragment_ids: Vec<u64> = Vec::new();
    for fragment in fragments {
        let prior = prior_by_id.get(&fragment.id)?;
        // Split the merged file list into files carried over verbatim and
        // brand-new files. A new file may only carry recast (added) fields — a
        // "new" file touching an existing field id would be an in-place data
        // rewrite, which no action can express.
        let mut new_files: Vec<DataFile> = Vec::new();
        let mut kept: Vec<DataFile> = Vec::new();
        for file in &fragment.files {
            if prior.files.contains(file) {
                kept.push(file.clone());
            } else if file.fields.iter().all(|id| added_ids.contains(id)) {
                new_files.push(file.clone());
            } else {
                return None;
            }
        }
        // The merged files must be the surviving prior files, in their original
        // order, followed by the appended new files — exactly the layout
        // `AddFields` then `ChangeSchema`'s dead-file pruning reproduce.
        let expected: Vec<DataFile> = prior
            .files
            .iter()
            .filter(|f| kept.contains(f))
            .chain(new_files.iter())
            .cloned()
            .collect();
        if fragment.files != expected {
            return None;
        }
        // Apart from its files, the fragment must be untouched.
        let mut truncated = fragment.clone();
        truncated.files = prior.files.clone();
        if &truncated != *prior {
            return None;
        }
        // The legacy `Merge` arm refreshes `last_updated` for every fragment
        // whose data files changed (`merge_fragment_physically_rewritten`):
        // here, one that gained a new file or dropped a prior one.
        if !new_files.is_empty() || kept.len() != prior.files.len() {
            refreshed_fragment_ids.push(fragment.id);
        }
        if !new_files.is_empty() {
            fragment_files.push((fragment.id, new_files));
        }
    }

    // `AddFields::validate` requires every new data file's field ids to be
    // declared in `new_fields`. The schema content is irrelevant — `ChangeSchema`
    // overwrites it — so carry one flat field per referenced new id.
    let mut referenced_new_ids: Vec<i32> = fragment_files
        .iter()
        .flat_map(|(_, files)| files.iter().flat_map(|file| file.fields.iter().copied()))
        .collect();
    referenced_new_ids.sort_unstable();
    referenced_new_ids.dedup();
    let mut new_fields: Vec<Field> = Vec::with_capacity(referenced_new_ids.len());
    for id in referenced_new_ids {
        let mut field = schema.field_by_id(id)?.clone();
        // Flatten so each entry contributes exactly its own id.
        field.children.clear();
        new_fields.push(field);
    }

    let mut actions: Vec<Box<dyn Action>> = vec![
        Box::new(AddFields {
            new_fields,
            fragment_files,
        }),
        Box::new(ChangeSchema {
            schema: schema.clone(),
        }),
    ];
    if manifest.uses_stable_row_ids() && !refreshed_fragment_ids.is_empty() {
        actions.push(Box::new(RefreshRowVersionMetadata {
            mode: RowVersionRefresh::MergedColumns {
                fragment_ids: refreshed_fragment_ids,
            },
        }));
    }
    Some(UserAction::new(actions))
}

/// Translate a [`UserAction`] back into an old [`Operation`].
///
/// Only action lists whose payloads carry every field of the target
/// [`Operation`] are supported. [`Operation::Append`] round-trips because
/// [`AddFragments`] keeps the whole [`Fragment`]; [`Operation::Delete`] does
/// **not**, because [`UpdateDeletionVector`] deliberately drops everything
/// but the deletion file (spike #6448). Rebuilding a `Delete` would require
/// the manifest the actions were applied against, which this function does
/// not take — verify `Delete` translation by applying the actions instead.
///
/// `CreateIndex` round-trips only when it has no `removed_indices`:
/// [`RemoveIndex`] keeps only a UUID, so a removed entry's [`IndexMetadata`]
/// cannot be rebuilt — verify a `CreateIndex` with removals by applying the
/// actions instead. `UpdateMemWalState` round-trips both ways, since
/// [`UpdateMergedGenerations`] carries its whole payload.
///
/// `Rewrite` does **not** round-trip: [`RewriteFragments`] keeps only the old
/// fragment *ids*, not the full [`Fragment`] payloads a [`RewriteGroup`] holds,
/// and [`RewriteIndex`] drops the rewritten index's details and version — so a
/// `Rewrite` cannot be reconstructed. Verify its translation by applying the
/// actions instead.
///
/// `Update` round-trips only in its pure-insert shape — a lone `AddFragments`
/// carrying an `inserted_rows_filter`. A general `Update` decomposes to
/// `UpdateDeletionVector` / `InvalidateIndexCoverage`, which keep only fragment
/// *ids*, so its `updated_fragments` payloads cannot be rebuilt; verify those
/// by applying the actions instead.
///
/// Returns `None` when the list does not match a supported shape.
pub fn operation_from_actions(user_action: &UserAction) -> Option<Operation> {
    let actions = user_action.actions.as_slice();
    // Per-position downcasts replace the old `enum Action` slice patterns.
    // `cast::<T>(i)` returns `Some(&T)` only when the `i`th action's concrete
    // payload type is `T`; the cascading `match` selects the unique shape that
    // every legacy `Operation` decomposes to.
    let cast = |i: usize| -> Option<&dyn Action> { actions.get(i).map(|a| a.as_ref()) };
    let downcast = |i: usize| -> Option<&dyn std::any::Any> { cast(i).map(|a| a.as_any()) };
    let as_t = |i: usize| -> Option<&dyn std::any::Any> { downcast(i) };

    match actions.len() {
        1 => {
            if let Some(add) = as_t(0).and_then(|a| a.downcast_ref::<AddFragments>()) {
                // A merge-insert's `inserted_rows_filter` rides on `AddFragments`
                // only inside an `Operation::Update` decomposition; a lone
                // `AddFragments` carrying one is the `RewriteRows` decomposition
                // of a pure-insert merge-insert `Update`. Without it, it's an
                // `Append`.
                return if add.inserted_rows_filter.is_none() {
                    Some(Operation::Append {
                        fragments: add.fragments.clone(),
                    })
                } else {
                    Some(Operation::Update {
                        removed_fragment_ids: Vec::new(),
                        updated_fragments: Vec::new(),
                        new_fragments: add.fragments.clone(),
                        fields_modified: Vec::new(),
                        merged_generations: Vec::new(),
                        fields_for_preserving_frag_bitmap: Vec::new(),
                        update_mode: None,
                        inserted_rows_filter: add.inserted_rows_filter.clone(),
                        updated_fragment_offsets: None,
                    })
                };
            }
            // A lone `ChangeSchema` carries the whole replacement schema, so it
            // rebuilds `Operation::Project` exactly. `Overwrite` also emits a
            // `ChangeSchema`, but always alongside fragment actions.
            if let Some(change) = as_t(0).and_then(|a| a.downcast_ref::<ChangeSchema>()) {
                return Some(Operation::Project {
                    schema: change.schema.clone(),
                });
            }
            if let Some(merged) = as_t(0).and_then(|a| a.downcast_ref::<UpdateMergedGenerations>())
            {
                return Some(Operation::UpdateMemWalState {
                    merged_generations: merged.merged_generations.clone(),
                });
            }
            if let Some(cfg) = as_t(0).and_then(|a| a.downcast_ref::<UpdateConfig>()) {
                return Some(Operation::UpdateConfig {
                    config_updates: cfg.config_updates.clone(),
                    table_metadata_updates: cfg.table_metadata_updates.clone(),
                    schema_metadata_updates: None,
                    field_metadata_updates: HashMap::new(),
                });
            }
            if let Some(bases) = as_t(0).and_then(|a| a.downcast_ref::<AddBases>()) {
                return Some(Operation::UpdateBases {
                    new_bases: bases.bases.clone(),
                });
            }
            if let Some(reserve) = as_t(0).and_then(|a| a.downcast_ref::<ReserveFragmentIds>()) {
                return Some(Operation::ReserveFragments {
                    num_fragments: reserve.num_fragments,
                });
            }
        }
        2 => {
            // `UpdateConfig` translation emits a leading `UpdateConfig` action,
            // optionally followed by an `UpdateSchemaMetadata` carrying the
            // schema- and field-metadata halves.
            if let (Some(cfg), Some(schema_meta)) = (
                as_t(0).and_then(|a| a.downcast_ref::<UpdateConfig>()),
                as_t(1).and_then(|a| a.downcast_ref::<UpdateSchemaMetadata>()),
            ) {
                return Some(Operation::UpdateConfig {
                    config_updates: cfg.config_updates.clone(),
                    table_metadata_updates: cfg.table_metadata_updates.clone(),
                    schema_metadata_updates: schema_meta.metadata_updates.clone(),
                    field_metadata_updates: schema_meta.field_metadata_updates.clone(),
                });
            }
        }
        3 => {
            // `Overwrite` without an upsert: `RemoveFragments(AllCurrent)` head
            // + `[ChangeSchema, AddFragments]` tail. The leading
            // `RemoveFragments(AllCurrent)` makes the shape unambiguous — no
            // other operation emits one. `initial_bases` is always `None`
            // because the CREATE path that sets it never routes through
            // actions.
            if let (Some(remove), Some(change), Some(add)) = (
                as_t(0).and_then(|a| a.downcast_ref::<RemoveFragments>()),
                as_t(1).and_then(|a| a.downcast_ref::<ChangeSchema>()),
                as_t(2).and_then(|a| a.downcast_ref::<AddFragments>()),
            ) && remove.selector == FragmentSelector::AllCurrent
                && add.inserted_rows_filter.is_none()
            {
                return Some(Operation::Overwrite {
                    fragments: add.fragments.clone(),
                    schema: change.schema.clone(),
                    config_upsert_values: None,
                    initial_bases: None,
                });
            }
        }
        4 => {
            // `Overwrite` with an upsert injects an `UpdateConfig` between the
            // `RemoveFragments(AllCurrent)` head and the schema/fragments tail.
            if let (Some(remove), Some(cfg), Some(change), Some(add)) = (
                as_t(0).and_then(|a| a.downcast_ref::<RemoveFragments>()),
                as_t(1).and_then(|a| a.downcast_ref::<UpdateConfig>()),
                as_t(2).and_then(|a| a.downcast_ref::<ChangeSchema>()),
                as_t(3).and_then(|a| a.downcast_ref::<AddFragments>()),
            ) && remove.selector == FragmentSelector::AllCurrent
                && cfg.table_metadata_updates.is_none()
                && add.inserted_rows_filter.is_none()
                && let Some(config_updates) = cfg.config_updates.as_ref()
            {
                return Some(Operation::Overwrite {
                    fragments: add.fragments.clone(),
                    schema: change.schema.clone(),
                    // Overwrite only ever emits value-bearing upserts, so
                    // every entry rebuilds a `config_upsert_values` pair.
                    config_upsert_values: Some(
                        config_updates
                            .update_entries
                            .iter()
                            .filter_map(|entry| {
                                entry.value.clone().map(|value| (entry.key.clone(), value))
                            })
                            .collect(),
                    ),
                    initial_bases: None,
                });
            }
        }
        _ => {}
    }

    // A non-empty run of nothing but `AddIndex` is a `CreateIndex` with no
    // removals; each `AddIndex` carries the full `IndexMetadata`. A mixed list
    // containing a `RemoveIndex` is not reconstructible.
    if !actions.is_empty()
        && actions
            .iter()
            .all(|action| action.as_any().is::<AddIndex>())
    {
        let new_indices = actions
            .iter()
            .map(|action| {
                action
                    .as_any()
                    .downcast_ref::<AddIndex>()
                    .expect("guarded by the all-AddIndex check above")
                    .index
                    .clone()
            })
            .collect();
        return Some(Operation::CreateIndex {
            new_indices,
            removed_indices: Vec::new(),
        });
    }

    // A non-empty run of nothing but `ReplaceFragmentColumns` rebuilds a
    // `DataReplacement` — one `DataReplacementGroup` per action. Each group
    // carries a single swapped `DataFile`, so an action holding anything other
    // than exactly one data file (only reachable post-rebase) is not
    // reconstructible.
    if !actions.is_empty()
        && actions
            .iter()
            .all(|action| action.as_any().is::<ReplaceFragmentColumns>())
    {
        return actions
            .iter()
            .map(|action| {
                let replace = action
                    .as_any()
                    .downcast_ref::<ReplaceFragmentColumns>()
                    .expect("guarded by the all-ReplaceFragmentColumns check above");
                match replace.new_data_files.as_slice() {
                    // A `DataReplacementGroup` is always an in-place swap of
                    // one file; a tombstone replacement has no
                    // `DataReplacement` form.
                    [new_file] if replace.replacement == ColumnReplacement::InPlace => {
                        Some(DataReplacementGroup(replace.fragment_id, new_file.clone()))
                    }
                    _ => None,
                }
            })
            .collect::<Option<Vec<_>>>()
            .map(|replacements| Operation::DataReplacement { replacements });
    }

    None
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use lance_index::mem_wal::{MEM_WAL_INDEX_NAME, MergedGeneration};
    use lance_table::format::Fragment;
    use roaring::RoaringBitmap;
    use uuid::Uuid;

    use std::sync::Arc;

    use lance_table::transaction::action::test_support::*;

    use super::super::*;
    use super::*;
    use crate::dataset::transaction::{
        DataReplacementGroup, Operation, UpdateMode, UpdatedFragmentOffsets,
    };
    use crate::dataset::write::merge_insert::inserted_rows::KeyExistenceFilter;

    /// Downcast helper used in place of the removed `enum Action` slice
    /// patterns. Panics with the expected concrete type name on a mismatch.
    fn cast<T: Action + 'static>(action: &dyn Action) -> &T {
        action
            .as_any()
            .downcast_ref::<T>()
            .unwrap_or_else(|| panic!("expected action of type {}", std::any::type_name::<T>()))
    }

    #[test]
    fn append_round_trips_through_actions() {
        let original = Operation::Append {
            fragments: vec![sample_fragment(0), sample_fragment(0)],
        };

        let user_action = actions_from_operation(&original).expect("Append should translate");
        assert_eq!(user_action.actions.len(), 1);
        assert!(user_action.actions[0].as_any().is::<AddFragments>());

        let round_tripped = operation_from_actions(&user_action)
            .expect("single AddFragments should translate back");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn empty_append_round_trips() {
        let original = Operation::Append { fragments: vec![] };
        let user_action = actions_from_operation(&original).unwrap();
        let round_tripped = operation_from_actions(&user_action).unwrap();
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn add_fragments_with_inserted_rows_filter_round_trips_as_update() {
        use crate::dataset::write::merge_insert::inserted_rows::FilterType;

        // An `AddFragments` carrying an `inserted_rows_filter` comes from a
        // merge-insert `Update`, never an `Append`; `operation_from_actions`
        // must rebuild it as an `Update` so the filter is not dropped.
        let filter = KeyExistenceFilter {
            field_ids: vec![0],
            filter: FilterType::ExactSet(HashSet::from([7])),
        };
        let with_filter = UserAction::new(vec![Box::new(AddFragments {
            fragments: vec![sample_fragment(0)],
            inserted_rows_filter: Some(filter.clone()),
        })]);
        match operation_from_actions(&with_filter).expect("rebuilds as Update") {
            Operation::Update {
                new_fragments,
                inserted_rows_filter: Some(rebuilt),
                removed_fragment_ids,
                updated_fragments,
                ..
            } => {
                assert_eq!(new_fragments, vec![sample_fragment(0)]);
                assert_eq!(rebuilt, filter);
                assert!(removed_fragment_ids.is_empty());
                assert!(updated_fragments.is_empty());
            }
            other => panic!("expected Update, got {other:?}"),
        }
    }

    #[test]
    fn add_fragments_apply_ignores_inserted_rows_filter() {
        use crate::dataset::write::merge_insert::inserted_rows::FilterType;

        // The filter is conflict-detection metadata only — `apply` produces the
        // same manifest whether or not it is present.
        let mut bare = empty_manifest();
        let mut filtered = empty_manifest();
        AddFragments {
            fragments: vec![sample_fragment(0)],
            inserted_rows_filter: None,
        }
        .apply(&mut bare, &mut vec![])
        .unwrap();
        AddFragments {
            fragments: vec![sample_fragment(0)],
            inserted_rows_filter: Some(KeyExistenceFilter {
                field_ids: vec![0],
                filter: FilterType::ExactSet(HashSet::from([7])),
            }),
        }
        .apply(&mut filtered, &mut vec![])
        .unwrap();
        assert_eq!(bare.fragments, filtered.fragments);
        assert_eq!(bare.max_fragment_id, filtered.max_fragment_id);
    }

    #[test]
    fn unknown_operation_translation_returns_none() {
        // Rewrite has no action mapping yet — its fragment half needs the
        // not-yet-added `RewriteFragments` action.
        let op = Operation::Rewrite {
            groups: vec![],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        };
        assert!(actions_from_operation(&op).is_none());
    }

    #[test]
    fn unknown_action_list_translation_returns_none() {
        // Empty action list cannot be represented as an old Operation
        // (Append { fragments: vec![] } is *not* empty — it has one action).
        assert!(operation_from_actions(&UserAction::new(vec![])).is_none());
    }

    #[test]
    fn rewrite_translates_fragment_half_and_rewrite_index() {
        // Non-stable dataset: the rewrite replaces its indices outright.
        let manifest = manifest_with_fragments(&[1, 2, 3]);
        let old_index = index_meta("idx", &[0], Some(&[1, 2]));
        let new_uuid = Uuid::new_v4();
        let operation = Operation::Rewrite {
            groups: vec![rewrite_group(&[1, 2], &[10])],
            rewritten_indices: vec![rewritten_index(old_index.uuid, new_uuid)],
            frag_reuse_index: None,
        };

        let user_action = actions_from_operation_with_manifest(
            &operation,
            &manifest,
            std::slice::from_ref(&old_index),
        )
        .expect("a non-stable Rewrite with a stored coverage bitmap translates");

        let actions = user_action.actions.as_slice();
        let [a0, a1] = actions else {
            panic!("expected [RewriteFragments, RewriteIndex], got {actions:?}");
        };
        let rewrite = cast::<RewriteFragments>(a0.as_ref());
        let index = cast::<RewriteIndex>(a1.as_ref());
        assert_eq!(rewrite.old_fragment_ids, vec![1, 2]);
        assert_eq!(rewrite.new_fragments, vec![Fragment::new(10)]);
        assert!(!rewrite.preserves_row_ids);
        assert_eq!(index.old_uuid, old_index.uuid);
        assert_eq!(index.new_uuid, new_uuid);
        // Coverage {1, 2} is re-pointed onto the rewritten fragment 10.
        assert_eq!(index.new_fragment_bitmap, bitmap(&[10]));
    }

    #[test]
    fn rewrite_stable_row_ids_translates_to_rebind_index_coverage() {
        let mut manifest = manifest_with_fragments(&[1, 2, 3]);
        manifest.reader_feature_flags |= lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
        let operation = Operation::Rewrite {
            groups: vec![rewrite_group(&[1, 2], &[10]), rewrite_group(&[3], &[11])],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        };

        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a stable-row-id Rewrite translates");

        let actions = user_action.actions.as_slice();
        let [a0, a1, a2] = actions else {
            panic!("expected 3 actions, got {actions:?}");
        };
        let first = cast::<RewriteFragments>(a0.as_ref());
        let second = cast::<RewriteFragments>(a1.as_ref());
        let rebind = cast::<RebindIndexCoverage>(a2.as_ref());
        assert!(first.preserves_row_ids && second.preserves_row_ids);
        // Each group keeps its old and new ids verbatim.
        assert_eq!(
            rebind.remap,
            IndexCoverageRemap::RewriteGroups(vec![
                group_remap(&[1, 2], &[10]),
                group_remap(&[3], &[11]),
            ])
        );
        assert!(rebind.modified_field_ids.is_empty());
    }

    #[test]
    fn rewrite_translates_frag_reuse_index_to_trailing_action() {
        let manifest = manifest_with_fragments(&[1, 2]);
        let operation = Operation::Rewrite {
            groups: vec![rewrite_group(&[1, 2], &[10])],
            rewritten_indices: vec![],
            frag_reuse_index: Some(index_meta("frag_reuse", &[], None)),
        };

        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a Rewrite registering a frag-reuse index translates");

        let actions = user_action.actions.as_slice();
        assert!(
            actions.len() == 2
                && actions[0].as_any().is::<RewriteFragments>()
                && actions[1].as_any().is::<RegisterFragReuse>(),
            "unexpected action list: {actions:?}"
        );
    }

    #[test]
    fn rewrite_with_no_groups_is_not_translated() {
        let manifest = manifest_with_fragments(&[1]);
        let operation = Operation::Rewrite {
            groups: vec![],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        };
        assert!(actions_from_operation_with_manifest(&operation, &manifest, &[]).is_none());
    }

    #[test]
    fn rewrite_stable_with_rewritten_indices_is_not_translated() {
        // A stable-row-id rewrite must not carry rewritten indices; such an
        // operation defers to the legacy arm.
        let mut manifest = manifest_with_fragments(&[1]);
        manifest.reader_feature_flags |= lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
        let operation = Operation::Rewrite {
            groups: vec![rewrite_group(&[1], &[10])],
            rewritten_indices: vec![rewritten_index(Uuid::new_v4(), Uuid::new_v4())],
            frag_reuse_index: None,
        };
        assert!(actions_from_operation_with_manifest(&operation, &manifest, &[]).is_none());
    }

    #[test]
    fn rewrite_stable_cardinality_changing_group_translates() {
        // A stable-row-id compaction group may split one fragment into many
        // (a large deletion-heavy fragment) or compact a fully-deleted one
        // into none. The per-group remap expresses both.
        let mut manifest = manifest_with_fragments(&[1, 2]);
        manifest.reader_feature_flags |= lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
        let operation = Operation::Rewrite {
            // Group one grows 1 -> {10, 11}; group two empties 2 -> {}.
            groups: vec![rewrite_group(&[1], &[10, 11]), rewrite_group(&[2], &[])],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        };

        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a stable-row-id Rewrite with cardinality-changing groups translates");

        let actions = user_action.actions.as_slice();
        let [a0, a1, a2] = actions else {
            panic!("expected 3 actions, got {actions:?}");
        };
        assert!(a0.as_any().is::<RewriteFragments>());
        assert!(a1.as_any().is::<RewriteFragments>());
        let rebind = cast::<RebindIndexCoverage>(a2.as_ref());
        assert_eq!(
            rebind.remap,
            IndexCoverageRemap::RewriteGroups(vec![
                group_remap(&[1], &[10, 11]),
                group_remap(&[2], &[]),
            ])
        );
    }

    #[test]
    fn rewrite_non_stable_skips_index_missing_from_manifest() {
        // A rewritten index dropped by a concurrent commit is not emitted.
        let manifest = manifest_with_fragments(&[1, 2]);
        let operation = Operation::Rewrite {
            groups: vec![rewrite_group(&[1, 2], &[10])],
            rewritten_indices: vec![rewritten_index(Uuid::new_v4(), Uuid::new_v4())],
            frag_reuse_index: None,
        };

        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a Rewrite whose index vanished still translates its fragment half");
        let actions = user_action.actions.as_slice();
        assert!(
            actions.len() == 1 && actions[0].as_any().is::<RewriteFragments>(),
            "unexpected action list: {actions:?}"
        );
    }

    #[test]
    fn rewrite_non_stable_index_without_bitmap_is_not_translated() {
        // An index with no stored coverage bitmap cannot be recalculated; the
        // operation defers to the legacy arm.
        let manifest = manifest_with_fragments(&[1, 2]);
        let old_index = index_meta("idx", &[0], None);
        let operation = Operation::Rewrite {
            groups: vec![rewrite_group(&[1, 2], &[10])],
            rewritten_indices: vec![rewritten_index(old_index.uuid, Uuid::new_v4())],
            frag_reuse_index: None,
        };
        assert!(
            actions_from_operation_with_manifest(
                &operation,
                &manifest,
                std::slice::from_ref(&old_index)
            )
            .is_none()
        );
    }

    fn merged_generation() -> MergedGeneration {
        MergedGeneration {
            shard_id: Uuid::new_v4(),
            generation: 7,
        }
    }

    fn rewrite_rows_update() -> Operation {
        Operation::Update {
            removed_fragment_ids: vec![3],
            updated_fragments: vec![fragment_with_deletion(1, deletion_file(1))],
            new_fragments: vec![sample_fragment(0)],
            fields_modified: vec![0],
            merged_generations: vec![merged_generation()],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteRows),
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        }
    }

    #[test]
    fn update_rewrite_rows_translates_to_actions() {
        let manifest = manifest_with_fragments(&[1, 2, 3]);
        let user_action =
            actions_from_operation_with_manifest(&rewrite_rows_update(), &manifest, &[])
                .expect("a non-stable RewriteRows update translates");

        let actions = user_action.actions.as_slice();
        let [a0, a1, a2, a3, a4] = actions else {
            panic!("expected 5 actions, got {actions:?}");
        };
        let remove = cast::<RemoveFragments>(a0.as_ref());
        let FragmentSelector::Ids(removed) = &remove.selector else {
            panic!("expected RemoveFragments::Ids, got {remove:?}");
        };
        let udv = cast::<UpdateDeletionVector>(a1.as_ref());
        let add = cast::<AddFragments>(a2.as_ref());
        let invalidate = cast::<InvalidateIndexCoverage>(a3.as_ref());
        assert!(a4.as_any().is::<UpdateMergedGenerations>());
        assert_eq!(removed, &[3]);
        assert_eq!(udv.fragment_id, 1);
        assert_eq!(add.fragments, vec![sample_fragment(0)]);
        assert!(add.inserted_rows_filter.is_none());
        assert_eq!(invalidate.fragment_ids, vec![1]);
        assert_eq!(invalidate.field_ids, vec![0]);
    }

    #[test]
    fn update_rewrite_rows_applies_like_the_legacy_arm() {
        let mut manifest = manifest_with_fragments(&[1, 2, 3]);
        let mut indices = vec![index_meta("idx", &[0], Some(&[1, 2, 3]))];

        let user_action =
            actions_from_operation_with_manifest(&rewrite_rows_update(), &manifest, &[])
                .expect("a non-stable RewriteRows update translates");
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut indices).unwrap();
        }

        let ids: Vec<u64> = manifest.fragments.iter().map(|f| f.id).collect();
        // Fragment 3 removed, 1 and 2 kept, the new fragment appended with a
        // freshly allocated id.
        assert_eq!(ids, vec![1, 2, 4]);
        let updated = manifest.fragments.iter().find(|f| f.id == 1).unwrap();
        assert!(updated.deletion_file.is_some());
        // The index over the modified field drops only the updated fragment;
        // the removed fragment is cleaned up by query-time bitmap intersection.
        assert_eq!(indices[0].fragment_bitmap, Some(bitmap(&[2, 3])));
    }

    #[test]
    fn update_rewrite_rows_skips_fragment_in_both_removed_and_updated() {
        // A rebased Update can list a fully-emptied fragment in both
        // `removed_fragment_ids` and `updated_fragments` — the conflict
        // resolver appends it to the former without pruning the latter. The
        // translation must emit only `RemoveFragments` for it, not a stale
        // `UpdateDeletionVector` targeting a fragment `RemoveFragments` drops.
        let mut manifest = manifest_with_fragments(&[1, 2]);
        let operation = Operation::Update {
            removed_fragment_ids: vec![1],
            updated_fragments: vec![fragment_with_deletion(1, deletion_file(1))],
            new_fragments: vec![],
            fields_modified: vec![0],
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteRows),
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        };
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a rebased RewriteRows update translates");
        assert!(
            !user_action
                .actions
                .iter()
                .any(|a| a.as_any().is::<UpdateDeletionVector>()),
            "the fragment is removed, not partially deleted",
        );

        // Applying must not error on the dropped fragment.
        let mut indices = vec![index_meta("idx", &[0], Some(&[1, 2]))];
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut indices).unwrap();
        }
        let ids: Vec<u64> = manifest.fragments.iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![2]);
    }

    #[test]
    fn update_with_inserted_rows_filter_carries_it_on_add_fragments() {
        use crate::dataset::write::merge_insert::inserted_rows::FilterType;

        let filter = KeyExistenceFilter {
            field_ids: vec![0],
            filter: FilterType::ExactSet(HashSet::from([7])),
        };
        let operation = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![],
            new_fragments: vec![sample_fragment(0)],
            fields_modified: vec![],
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteRows),
            inserted_rows_filter: Some(filter.clone()),
            updated_fragment_offsets: None,
        };
        let manifest = empty_manifest();
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a pure-insert update translates");
        let actions = user_action.actions.as_slice();
        let [a0] = actions else {
            panic!("expected [AddFragments], got {actions:?}");
        };
        let add = cast::<AddFragments>(a0.as_ref());
        assert_eq!(add.inserted_rows_filter.as_ref(), Some(&filter));
    }

    #[test]
    fn update_pure_insert_round_trips_through_actions() {
        use crate::dataset::write::merge_insert::inserted_rows::FilterType;

        let original = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![],
            new_fragments: vec![sample_fragment(0), sample_fragment(0)],
            fields_modified: vec![],
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: None,
            inserted_rows_filter: Some(KeyExistenceFilter {
                field_ids: vec![0],
                filter: FilterType::ExactSet(HashSet::from([7])),
            }),
            updated_fragment_offsets: None,
        };
        let manifest = empty_manifest();
        let user_action = actions_from_operation_with_manifest(&original, &manifest, &[])
            .expect("a pure-insert update translates");
        let round_tripped =
            operation_from_actions(&user_action).expect("a filter-bearing AddFragments rebuilds");
        assert_eq!(original, round_tripped);
    }

    fn rewrite_columns_update(
        updated_fragments: Vec<Fragment>,
        fields_modified: Vec<u32>,
    ) -> Operation {
        Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments,
            new_fragments: vec![],
            fields_modified,
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        }
    }

    fn manifest_with_two_column_fragments() -> Manifest {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![
            fragment_with_files(
                1,
                vec![
                    data_file("f1_a.lance", vec![0]),
                    data_file("f1_b.lance", vec![1]),
                ],
            ),
            fragment_with_files(
                2,
                vec![
                    data_file("f2_a.lance", vec![0]),
                    data_file("f2_b.lance", vec![1]),
                ],
            ),
        ]);
        manifest.max_fragment_id = Some(2);
        manifest
    }

    #[test]
    fn update_rewrite_columns_translates_to_replace_actions() {
        let manifest = manifest_with_two_column_fragments();
        // Fragment 1's `b` column is rewritten in place.
        let updated = fragment_with_files(
            1,
            vec![
                data_file("f1_a.lance", vec![0]),
                data_file("f1_b_v2.lance", vec![1]),
            ],
        );
        let operation = rewrite_columns_update(vec![updated], vec![1]);
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a non-stable RewriteColumns update translates");

        assert_eq!(user_action.actions.len(), 1);
        let replace = cast::<ReplaceFragmentColumns>(user_action.actions[0].as_ref());
        assert_eq!(replace.fragment_id, 1);
        assert_eq!(replace.field_ids, vec![1]);
        // Only the rewritten `b` file is carried; the untouched `a` file is
        // left for `apply` to keep from the manifest.
        assert_eq!(
            replace.new_data_files,
            vec![data_file("f1_b_v2.lance", vec![1])]
        );
        assert!(replace.updated_row_offsets.is_none());
    }

    #[test]
    fn update_rewrite_columns_applies_like_the_legacy_arm() {
        let mut manifest = manifest_with_two_column_fragments();
        let mut indices = vec![index_meta("idx", &[1], Some(&[1, 2]))];

        let updated = fragment_with_files(
            1,
            vec![
                data_file("f1_a.lance", vec![0]),
                data_file("f1_b_v2.lance", vec![1]),
            ],
        );
        let operation = rewrite_columns_update(vec![updated.clone()], vec![1]);
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a non-stable RewriteColumns update translates");
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut indices).unwrap();
        }

        // Fragment 1 carries the rewritten `b` file; fragment 2 is untouched.
        assert_eq!(manifest.fragments[0], updated);
        assert_eq!(manifest.fragments[1].files[1].path, "f2_b.lance");
        // The index over the rewritten field drops the updated fragment.
        assert_eq!(indices[0].fragment_bitmap, Some(bitmap(&[2])));
    }

    #[test]
    fn update_rewrite_columns_translates_all_null_column_fill() {
        let manifest = manifest_with_two_column_fragments();
        // Fragment 1 gains real data for a previously-uncovered field.
        let updated = fragment_with_files(
            1,
            vec![
                data_file("f1_a.lance", vec![0]),
                data_file("f1_b.lance", vec![1]),
                data_file("f1_c.lance", vec![2]),
            ],
        );
        let operation = rewrite_columns_update(vec![updated], vec![2]);
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("an all-NULL column fill translates");
        assert_eq!(user_action.actions.len(), 1);
        let replace = cast::<ReplaceFragmentColumns>(user_action.actions[0].as_ref());
        // The appended file is the only one carried.
        assert_eq!(
            replace.new_data_files,
            vec![data_file("f1_c.lance", vec![2])]
        );
    }

    #[test]
    fn update_rewrite_columns_with_dropped_file_translates_verbatim() {
        // A `RewriteColumns` update that drops a data file is not a pure column
        // swap/append, so it is carried as a `Tombstone` replacement holding
        // the fragment's whole post-rewrite file list.
        let mut manifest = manifest_with_two_column_fragments();
        let updated = fragment_with_files(1, vec![data_file("f1_a.lance", vec![0])]);
        let operation = rewrite_columns_update(vec![updated.clone()], vec![1]);
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a dropped-file RewriteColumns update translates");
        assert_eq!(user_action.actions.len(), 1);
        let replace = cast::<ReplaceFragmentColumns>(user_action.actions[0].as_ref());
        assert_eq!(replace.replacement, ColumnReplacement::Tombstone);
        assert_eq!(replace.new_data_files, updated.files);
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut vec![]).unwrap();
        }
        assert_eq!(manifest.fragments[0], updated);
    }

    #[test]
    fn update_rewrite_columns_with_relayout_translates_verbatim() {
        // A leading file whose field layout changed cannot be reproduced by an
        // in-place swap, so it is carried verbatim as a `Tombstone` replacement.
        let mut manifest = manifest_with_two_column_fragments();
        let updated = fragment_with_files(
            1,
            vec![
                data_file("f1_a.lance", vec![0]),
                data_file("f1_b.lance", vec![1, 2]),
            ],
        );
        let operation = rewrite_columns_update(vec![updated.clone()], vec![1]);
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a relayout RewriteColumns update translates");
        assert_eq!(user_action.actions.len(), 1);
        let replace = cast::<ReplaceFragmentColumns>(user_action.actions[0].as_ref());
        assert_eq!(replace.replacement, ColumnReplacement::Tombstone);
        assert_eq!(replace.new_data_files, updated.files);
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut vec![]).unwrap();
        }
        assert_eq!(manifest.fragments[0], updated);
    }

    fn manifest_with_one_two_field_file() -> Manifest {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![fragment_with_files(
            1,
            vec![data_file("f1.lance", vec![0, 1])],
        )]);
        manifest
    }

    #[test]
    fn update_rewrite_columns_partial_field_translates_to_tombstone() {
        // Field 1 shares `f1.lance` with the untouched field 0, so rewriting
        // it tombstones field 1 in that file and writes a trailing `[1]` file
        // — the `ColumnReplacement::Tombstone` shape.
        let manifest = manifest_with_one_two_field_file();
        let updated = fragment_with_files(
            1,
            vec![
                data_file("f1.lance", vec![0, -2]),
                data_file("f1_b_v2.lance", vec![1]),
            ],
        );
        let operation = rewrite_columns_update(vec![updated.clone()], vec![1]);
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a partial-field RewriteColumns update translates");
        assert_eq!(user_action.actions.len(), 1);
        let replace = cast::<ReplaceFragmentColumns>(user_action.actions[0].as_ref());
        assert_eq!(replace.replacement, ColumnReplacement::Tombstone);
        assert_eq!(replace.fragment_id, 1);
        assert_eq!(replace.field_ids, vec![1]);
        // The action carries the fragment's whole post-rewrite file list;
        // `apply` installs it verbatim.
        assert_eq!(replace.new_data_files, updated.files);
    }

    #[test]
    fn update_rewrite_columns_keeps_a_fully_tombstoned_file() {
        // The `merge_insert` partial-schema path tombstones the rewritten
        // fields in place but does not drop the file it empties — so a kept
        // `[-2, -2]` file lands between the untouched leading file and the
        // freshly written one. The translation carries that layout verbatim.
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![fragment_with_files(
            1,
            vec![
                data_file("untouched.lance", vec![-2, 1, -2, 3]),
                data_file("shared.lance", vec![0, 2]),
            ],
        )]);
        let updated = fragment_with_files(
            1,
            vec![
                data_file("untouched.lance", vec![-2, 1, -2, 3]),
                data_file("shared.lance", vec![-2, -2]),
                data_file("rewritten.lance", vec![0, 2]),
            ],
        );
        let operation = rewrite_columns_update(vec![updated.clone()], vec![0, 2]);
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a kept-tombstoned-file RewriteColumns update translates");
        assert_eq!(user_action.actions.len(), 1);
        let replace = cast::<ReplaceFragmentColumns>(user_action.actions[0].as_ref());
        assert_eq!(replace.replacement, ColumnReplacement::Tombstone);
        assert_eq!(replace.new_data_files, updated.files);
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut vec![]).unwrap();
        }
        // The action path leaves the fragment exactly as the legacy arm does.
        assert_eq!(manifest.fragments[0], updated);
    }

    #[test]
    fn update_rewrite_columns_partial_field_applies_like_the_legacy_arm() {
        // The action path leaves the fragment exactly as the legacy
        // `build_manifest` arm does: the `updated_fragments` entry verbatim.
        let mut manifest = manifest_with_one_two_field_file();
        let mut indices = vec![index_meta("idx", &[1], Some(&[1]))];
        let updated = fragment_with_files(
            1,
            vec![
                data_file("f1.lance", vec![0, -2]),
                data_file("f1_b_v2.lance", vec![1]),
            ],
        );
        let operation = rewrite_columns_update(vec![updated.clone()], vec![1]);
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a partial-field RewriteColumns update translates");
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut indices).unwrap();
        }
        assert_eq!(manifest.fragments[0], updated);
        // The index over the rewritten field drops the updated fragment.
        assert_eq!(indices[0].fragment_bitmap, Some(bitmap(&[])));
    }

    #[test]
    fn update_rewrite_columns_partial_field_translation_round_trips_via_apply() {
        // A second partial rewrite of a fragment that already carries a `-2`
        // tombstone from an earlier rewrite still translates and applies.
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![fragment_with_files(
            1,
            vec![
                data_file("f1.lance", vec![0, -2]),
                data_file("f1_b_v1.lance", vec![1]),
            ],
        )]);
        let updated = fragment_with_files(
            1,
            vec![
                data_file("f1.lance", vec![0, -2]),
                data_file("f1_b_v2.lance", vec![1]),
            ],
        );
        let operation = rewrite_columns_update(vec![updated.clone()], vec![1]);
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("re-rewriting a partially tombstoned fragment translates");
        // Field 1 already lives in its own file, so this is an in-place swap.
        assert_eq!(user_action.actions.len(), 1);
        let replace = cast::<ReplaceFragmentColumns>(user_action.actions[0].as_ref());
        assert_eq!(replace.replacement, ColumnReplacement::InPlace);
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut vec![]).unwrap();
        }
        assert_eq!(manifest.fragments[0], updated);
    }

    fn stable_rewrite_rows_update(fields_modified: Vec<u32>) -> Operation {
        let mut updated = rowid_fragment(1, &[0, 1, 2], Some(2));
        updated.deletion_file = Some(deletion_file(1));
        Operation::Update {
            removed_fragment_ids: vec![3],
            updated_fragments: vec![updated],
            new_fragments: vec![rowid_fragment(0, &[6, 7], None)],
            fields_modified,
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteRows),
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        }
    }

    fn stable_manifest_with_rowid_fragments() -> Manifest {
        let mut updated = rowid_fragment(1, &[0, 1, 2], Some(2));
        updated.deletion_file = Some(deletion_file(1));
        let mut manifest = manifest_at_version(
            5,
            vec![
                updated,
                rowid_fragment(2, &[3, 4, 5], Some(2)),
                rowid_fragment(3, &[6, 7, 8], Some(2)),
            ],
        );
        manifest.reader_feature_flags |= lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
        manifest.max_fragment_id = Some(3);
        manifest.next_row_id = 9;
        manifest
    }

    #[test]
    fn update_rewrite_rows_on_stable_row_id_dataset_translates() {
        // A stable-row-id `RewriteRows` update translates: the deletion-vector
        // and add half, an `InvalidateIndexCoverage` for the modified field, a
        // `RebindIndexCoverage` registering the pure-rewrite fragment, the
        // version-metadata refresh, and finally `RemoveFragments` — emitted
        // last so the refresh can still trace `created_at` through fragment 3.
        let manifest = stable_manifest_with_rowid_fragments();
        let user_action = actions_from_operation_with_manifest(
            &stable_rewrite_rows_update(vec![5]),
            &manifest,
            &[],
        )
        .expect("a stable-row-id RewriteRows update translates");
        let actions = user_action.actions.as_slice();
        let [a0, a1, a2, a3, a4, a5] = actions else {
            panic!("expected 6 actions, got {actions:?}");
        };
        let udv = cast::<UpdateDeletionVector>(a0.as_ref());
        assert!(a1.as_any().is::<AddFragments>());
        let invalidate = cast::<InvalidateIndexCoverage>(a2.as_ref());
        let rebind = cast::<RebindIndexCoverage>(a3.as_ref());
        let IndexCoverageRemap::InsertPureRewrite {
            new_fragment_ids,
            original_fragment_ids,
        } = &rebind.remap
        else {
            panic!("expected InsertPureRewrite, got {:?}", rebind.remap);
        };
        let modified_field_ids = &rebind.modified_field_ids;
        let refresh = cast::<RefreshRowVersionMetadata>(a4.as_ref());
        // The refresh names the freshly written fragment (allocated id 4).
        assert!(matches!(
            &refresh.mode,
            RowVersionRefresh::UpdatedRows { new_fragment_ids } if new_fragment_ids == &[4]
        ));
        let remove = cast::<RemoveFragments>(a5.as_ref());
        let FragmentSelector::Ids(removed) = &remove.selector else {
            panic!("expected RemoveFragments::Ids, got {remove:?}");
        };
        assert_eq!(udv.fragment_id, 1);
        assert_eq!(invalidate.fragment_ids, vec![1]);
        // The new fragment is allocated id 4 (high-water mark was 3).
        assert_eq!(new_fragment_ids, &[4]);
        // The gate is every original fragment: emptied 3, plus updated 1.
        assert_eq!(original_fragment_ids, &[3, 1]);
        // `fields_for_preserving_frag_bitmap` is empty here.
        assert!(modified_field_ids.is_empty());
        assert_eq!(removed, &[3]);
    }

    #[test]
    fn update_rewrite_rows_registers_pure_rewrite_fragment_when_fully_covered() {
        // The index covers every original fragment (1, 2, 3) over an
        // unmodified field, so the pure-rewrite fragment 4 is registered.
        let mut manifest = stable_manifest_with_rowid_fragments();
        let mut indices = vec![index_meta("idx", &[7], Some(&[1, 2, 3]))];
        let user_action = actions_from_operation_with_manifest(
            &stable_rewrite_rows_update(vec![]),
            &manifest,
            &indices,
        )
        .expect("a stable-row-id RewriteRows update translates");
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut indices).unwrap();
        }
        assert!(
            indices[0].fragment_bitmap.as_ref().unwrap().contains(4),
            "pure-rewrite fragment 4 should be registered: {:?}",
            indices[0].fragment_bitmap,
        );
        assert!(!manifest.fragments.iter().any(|f| f.id == 3));
    }

    #[test]
    fn update_rewrite_rows_pure_rewrite_registration_gated_on_full_coverage() {
        // The index misses original fragment 3, so not every rewritten row was
        // indexed; the pure-rewrite fragment must not be registered.
        let mut manifest = stable_manifest_with_rowid_fragments();
        let mut indices = vec![index_meta("idx", &[7], Some(&[1, 2]))];
        let user_action = actions_from_operation_with_manifest(
            &stable_rewrite_rows_update(vec![]),
            &manifest,
            &indices,
        )
        .expect("a stable-row-id RewriteRows update translates");
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut indices).unwrap();
        }
        assert!(
            !indices[0].fragment_bitmap.as_ref().unwrap().contains(4),
            "fragment 4 must stay unindexed when coverage is incomplete: {:?}",
            indices[0].fragment_bitmap,
        );
    }

    #[test]
    fn update_unspecified_mode_on_stable_row_id_dataset_emits_no_rebind() {
        // Only `RewriteRows` registers pure-rewrite fragments; an
        // unspecified-mode stable update refreshes version metadata but emits
        // no `RebindIndexCoverage`.
        let manifest = stable_manifest_with_rowid_fragments();
        let mut operation = stable_rewrite_rows_update(vec![]);
        if let Operation::Update { update_mode, .. } = &mut operation {
            *update_mode = None;
        }
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("an unspecified-mode stable update translates");
        assert!(
            !user_action
                .actions
                .iter()
                .any(|a| a.as_any().is::<RebindIndexCoverage>()),
            "no RebindIndexCoverage for an unspecified-mode update",
        );
        assert!(
            user_action.actions.iter().any(|a| matches!(
                a.as_any().downcast_ref::<RefreshRowVersionMetadata>(),
                Some(RefreshRowVersionMetadata {
                    mode: RowVersionRefresh::UpdatedRows { .. },
                })
            )),
            "version metadata is still refreshed",
        );
    }

    #[test]
    fn update_rewrite_columns_on_stable_row_id_dataset_refreshes_version_metadata() {
        // A stable-row-id `RewriteColumns` update translates fully: the column
        // swap plus a trailing `RefreshRowVersionMetadata` for the partially
        // rewritten rows.
        let mut manifest = manifest_with_two_column_fragments();
        manifest.reader_feature_flags |= lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
        let updated = fragment_with_files(
            1,
            vec![
                data_file("f1_a.lance", vec![0]),
                data_file("f1_b_v2.lance", vec![1]),
            ],
        );
        let mut operation = rewrite_columns_update(vec![updated], vec![1]);
        if let Operation::Update {
            updated_fragment_offsets,
            ..
        } = &mut operation
        {
            *updated_fragment_offsets = Some(UpdatedFragmentOffsets(HashMap::from([(
                1,
                RoaringBitmap::from_iter([0u32, 2]),
            )])));
        }
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a stable-row-id RewriteColumns update translates");
        let actions = user_action.actions.as_slice();
        let [a0, a1] = actions else {
            panic!("expected 2 actions, got {actions:?}");
        };
        let replace = cast::<ReplaceFragmentColumns>(a0.as_ref());
        let refresh = cast::<RefreshRowVersionMetadata>(a1.as_ref());
        let RowVersionRefresh::RewrittenColumns { touched_offsets } = &refresh.mode else {
            panic!("expected RewrittenColumns, got {:?}", refresh.mode);
        };
        assert_eq!(replace.fragment_id, 1);
        assert_eq!(touched_offsets, &vec![(1u64, vec![0u64, 2])]);
    }

    #[test]
    fn update_rewrite_columns_on_stable_row_id_dataset_without_offsets_emits_no_refresh() {
        // With no matched offsets there is nothing to refresh — the partial
        // path is a no-op, mirroring the legacy `build_manifest` skip.
        let mut manifest = manifest_with_two_column_fragments();
        manifest.reader_feature_flags |= lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
        let updated = fragment_with_files(
            1,
            vec![
                data_file("f1_a.lance", vec![0]),
                data_file("f1_b_v2.lance", vec![1]),
            ],
        );
        let operation = rewrite_columns_update(vec![updated], vec![1]);
        let user_action = actions_from_operation_with_manifest(&operation, &manifest, &[])
            .expect("a stable-row-id RewriteColumns update translates");
        assert!(
            !user_action
                .actions
                .iter()
                .any(|a| a.as_any().is::<RefreshRowVersionMetadata>()),
            "no offsets means no refresh action",
        );
    }

    #[test]
    fn update_with_missing_deletion_file_is_not_translated() {
        // A RewriteRows update whose updated fragment carries no deletion file
        // is malformed for this translation — same guard as Delete.
        let mut operation = rewrite_rows_update();
        if let Operation::Update {
            updated_fragments, ..
        } = &mut operation
        {
            *updated_fragments = vec![sample_fragment(1)];
        }
        let manifest = manifest_with_fragments(&[1, 2, 3]);
        assert!(actions_from_operation_with_manifest(&operation, &manifest, &[]).is_none());
    }

    #[test]
    fn delete_translates_to_update_deletion_vector_and_remove() {
        let file = deletion_file(100);
        let original = Operation::Delete {
            updated_fragments: vec![fragment_with_deletion(1, file.clone())],
            deleted_fragment_ids: vec![2, 3],
            predicate: "id < 10".into(),
        };

        let user_action = actions_from_operation(&original).expect("Delete translates");
        // One UpdateDeletionVector + one RemoveFragments, in that order.
        assert_eq!(user_action.actions.len(), 2);
        let udv = user_action.actions[0]
            .as_any()
            .downcast_ref::<UpdateDeletionVector>()
            .expect("first action is UpdateDeletionVector");
        assert_eq!(udv.fragment_id, 1);
        assert_eq!(udv.new_deletion_file, file);
        assert!(udv.affected_rows.is_none());
        assert!(user_action.actions[1].as_any().is::<RemoveFragments>());
        assert_eq!(user_action.description.as_deref(), Some("id < 10"));

        // Applying the actions reproduces the Delete's manifest effect.
        let result = apply_all(&user_action.actions, &[1, 2, 3]);
        let ids: Vec<u64> = result.iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![1]);
        assert_eq!(result[0].deletion_file, Some(file));
    }

    #[test]
    fn delete_with_no_removed_fragments_translates() {
        let file = deletion_file(7);
        let original = Operation::Delete {
            updated_fragments: vec![fragment_with_deletion(1, file.clone())],
            deleted_fragment_ids: vec![],
            predicate: "x".into(),
        };
        let user_action = actions_from_operation(&original).unwrap();
        // No RemoveFragments action when nothing is fully removed.
        assert_eq!(user_action.actions.len(), 1);
        assert!(user_action.actions[0].as_any().is::<UpdateDeletionVector>());

        let result = apply_all(&user_action.actions, &[1]);
        assert_eq!(result[0].deletion_file, Some(file));
    }

    #[test]
    fn delete_with_no_updated_fragments_translates() {
        let original = Operation::Delete {
            updated_fragments: vec![],
            deleted_fragment_ids: vec![5, 6],
            predicate: "everything".into(),
        };
        let user_action = actions_from_operation(&original).unwrap();
        assert_eq!(user_action.actions.len(), 1);
        assert!(user_action.actions[0].as_any().is::<RemoveFragments>());

        let result = apply_all(&user_action.actions, &[5, 6, 7]);
        let ids: Vec<u64> = result.iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![7]);
    }

    #[test]
    fn delete_with_updated_fragment_missing_deletion_file_falls_back() {
        // An updated fragment with no deletion file is malformed for the
        // action translation, so the operation falls back to the legacy path.
        let original = Operation::Delete {
            updated_fragments: vec![sample_fragment(1)],
            deleted_fragment_ids: vec![],
            predicate: "x".into(),
        };
        assert!(actions_from_operation(&original).is_none());
    }

    #[test]
    fn project_round_trips_through_actions() {
        let original = Operation::Project {
            schema: schema_with(&[(0, "a")]),
        };

        let user_action = actions_from_operation(&original).expect("Project translates");
        assert_eq!(user_action.actions.len(), 1);
        assert!(user_action.actions[0].as_any().is::<ChangeSchema>());

        let round_tripped =
            operation_from_actions(&user_action).expect("ChangeSchema translates back to Project");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn project_translation_apply_narrows_schema_and_prunes_files() {
        // A Project that drops field 1, applied via its translated action,
        // reproduces the legacy Project manifest effect (schema narrowed,
        // dead data files pruned).
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a"), (1, "b")]);
        manifest.fragments = Arc::new(vec![fragment_with_files(
            1,
            vec![data_file("a.lance", vec![0]), data_file("b.lance", vec![1])],
        )]);

        let operation = Operation::Project {
            schema: schema_with(&[(0, "a")]),
        };
        let user_action = actions_from_operation(&operation).unwrap();
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut vec![]).unwrap();
        }

        assert_eq!(manifest.schema, schema_with(&[(0, "a")]));
        assert_eq!(manifest.fragments[0].files.len(), 1);
        assert_eq!(manifest.fragments[0].files[0].path, "a.lance");
    }

    #[test]
    fn overwrite_round_trips_through_actions() {
        let original = Operation::Overwrite {
            fragments: vec![sample_fragment(0), sample_fragment(0)],
            schema: schema_with(&[(0, "a"), (1, "b")]),
            config_upsert_values: None,
            initial_bases: None,
        };

        let user_action = actions_from_operation(&original).expect("Overwrite translates");
        let actions = user_action.actions.as_slice();
        let [a0, a1, a2] = actions else {
            panic!("expected 3 actions, got {actions:?}");
        };
        let remove = cast::<RemoveFragments>(a0.as_ref());
        assert_eq!(remove.selector, FragmentSelector::AllCurrent);
        assert!(a1.as_any().is::<ChangeSchema>());
        assert!(a2.as_any().is::<AddFragments>());

        let round_tripped = operation_from_actions(&user_action)
            .expect("the Overwrite decomposition translates back");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn overwrite_with_config_round_trips_through_actions() {
        let original = Operation::Overwrite {
            fragments: vec![sample_fragment(0)],
            schema: schema_with(&[(0, "a")]),
            config_upsert_values: Some(HashMap::from([("k".to_string(), "v".to_string())])),
            initial_bases: None,
        };

        let user_action = actions_from_operation(&original).expect("Overwrite translates");
        // `UpdateConfig` is emitted before `ChangeSchema` so a config-key
        // collision is detected ahead of the schema retry — see
        // `actions_from_operation`.
        let actions = user_action.actions.as_slice();
        let [a0, a1, a2, a3] = actions else {
            panic!("expected 4 actions, got {actions:?}");
        };
        let remove = cast::<RemoveFragments>(a0.as_ref());
        assert_eq!(remove.selector, FragmentSelector::AllCurrent);
        let cfg = cast::<UpdateConfig>(a1.as_ref());
        assert!(cfg.table_metadata_updates.is_none());
        assert!(a2.as_any().is::<ChangeSchema>());
        assert!(a3.as_any().is::<AddFragments>());

        let round_tripped = operation_from_actions(&user_action)
            .expect("the Overwrite decomposition translates back");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn overwrite_with_initial_bases_falls_back() {
        // `initial_bases` is set only by the CREATE path, which the legacy
        // `build_manifest` arm still owns, so translation declines it.
        let original = Operation::Overwrite {
            fragments: vec![sample_fragment(0)],
            schema: schema_with(&[(0, "a")]),
            config_upsert_values: None,
            initial_bases: Some(vec![]),
        };
        assert!(actions_from_operation(&original).is_none());
    }

    #[test]
    fn overwrite_translation_apply_replaces_fragments_schema_and_config() {
        // Applying the translated actions reproduces the legacy Overwrite
        // manifest effect: every prior fragment gone, schema replaced, new
        // fragments appended (ids restarting from 0, since the `AllCurrent`
        // removal resets the high-water mark), and the config upsert merged
        // in.
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a")]);
        manifest.fragments = Arc::new(vec![sample_fragment(1), sample_fragment(2)]);
        manifest.max_fragment_id = Some(2);
        manifest
            .config
            .insert("keep".to_string(), "old".to_string());

        let operation = Operation::Overwrite {
            fragments: vec![sample_fragment(0), sample_fragment(0)],
            schema: schema_with(&[(0, "x"), (1, "y")]),
            config_upsert_values: Some(HashMap::from([("new".to_string(), "val".to_string())])),
            initial_bases: None,
        };
        let user_action = actions_from_operation(&operation).unwrap();
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut vec![]).unwrap();
        }

        assert_eq!(manifest.schema, schema_with(&[(0, "x"), (1, "y")]));
        // Prior fragment ids 1/2 are gone; the new fragments restart at 0
        // because the `AllCurrent` removal cleared the high-water mark.
        let ids: Vec<u64> = manifest.fragments.iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(manifest.max_fragment_id, Some(1));
        assert_eq!(manifest.config.get("keep"), Some(&"old".to_string()));
        assert_eq!(manifest.config.get("new"), Some(&"val".to_string()));
    }

    fn merge_prior_manifest() -> Manifest {
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a")]);
        manifest.fragments = Arc::new(vec![
            fragment_with_files(1, vec![data_file("a.lance", vec![0])]),
            fragment_with_files(2, vec![data_file("a2.lance", vec![0])]),
        ]);
        manifest.max_fragment_id = Some(2);
        manifest
    }

    #[test]
    fn merge_translates_to_add_fields() {
        let prior = merge_prior_manifest();
        // The merge appends column `b` (field id 1) to both fragments.
        let merge_schema = schema_with(&[(0, "a"), (1, "b")]);
        let operation = Operation::Merge {
            schema: merge_schema.clone(),
            fragments: vec![
                fragment_with_files(
                    1,
                    vec![data_file("a.lance", vec![0]), data_file("b.lance", vec![1])],
                ),
                fragment_with_files(
                    2,
                    vec![
                        data_file("a2.lance", vec![0]),
                        data_file("b2.lance", vec![1]),
                    ],
                ),
            ],
        };

        let user_action = actions_from_operation_with_manifest(&operation, &prior, &[])
            .expect("column-add Merge translates");
        assert_eq!(user_action.actions.len(), 1);
        let add = user_action.actions[0]
            .as_any()
            .downcast_ref::<AddFields>()
            .expect("expected AddFields");
        assert_eq!(add.new_fields, vec![field(1, "b")]);
        assert_eq!(add.fragment_files.len(), 2);
        assert_eq!(add.fragment_files[0].0, 1);
        assert_eq!(add.fragment_files[0].1[0].path, "b.lance");
        assert_eq!(add.fragment_files[1].0, 2);
        assert_eq!(add.fragment_files[1].1[0].path, "b2.lance");

        // Applying the translated action reproduces the Merge's manifest.
        let mut manifest = prior;
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut vec![]).unwrap();
        }
        assert_eq!(manifest.schema, merge_schema);
        let Operation::Merge { fragments, .. } = &operation else {
            unreachable!()
        };
        assert_eq!(manifest.fragments.as_ref(), fragments);
    }

    #[test]
    fn merge_with_schema_only_change_translates() {
        // A Merge that adds a field but no per-fragment data (e.g. an
        // all-fragments-empty dataset) yields an AddFields with no files.
        let mut prior = merge_prior_manifest();
        prior.fragments = Arc::new(vec![]);

        let operation = Operation::Merge {
            schema: schema_with(&[(0, "a"), (1, "b")]),
            fragments: vec![],
        };
        let user_action = actions_from_operation_with_manifest(&operation, &prior, &[]).unwrap();
        let add = user_action.actions[0]
            .as_any()
            .downcast_ref::<AddFields>()
            .expect("expected AddFields");
        assert_eq!(add.new_fields, vec![field(1, "b")]);
        assert!(add.fragment_files.is_empty());
    }

    #[test]
    fn merge_that_drops_a_field_does_not_translate() {
        let mut prior = merge_prior_manifest();
        prior.schema = schema_with(&[(0, "a"), (1, "b")]);
        // The merge schema drops field 1 — not a pure column add.
        let operation = Operation::Merge {
            schema: schema_with(&[(0, "a")]),
            fragments: prior.fragments.as_ref().clone(),
        };
        assert!(actions_from_operation_with_manifest(&operation, &prior, &[]).is_none());
    }

    #[test]
    fn merge_that_rewrites_a_data_file_does_not_translate() {
        let prior = merge_prior_manifest();
        // Fragment 1's existing file is replaced rather than appended to.
        let operation = Operation::Merge {
            schema: schema_with(&[(0, "a"), (1, "b")]),
            fragments: vec![
                fragment_with_files(
                    1,
                    vec![
                        data_file("a_rewritten.lance", vec![0]),
                        data_file("b.lance", vec![1]),
                    ],
                ),
                fragment_with_files(
                    2,
                    vec![
                        data_file("a2.lance", vec![0]),
                        data_file("b2.lance", vec![1]),
                    ],
                ),
            ],
        };
        assert!(actions_from_operation_with_manifest(&operation, &prior, &[]).is_none());
    }

    #[test]
    fn merge_that_changes_the_fragment_set_does_not_translate() {
        let prior = merge_prior_manifest();
        // The merge introduces a brand-new fragment 3 — AddFields cannot add
        // fragments.
        let operation = Operation::Merge {
            schema: schema_with(&[(0, "a"), (1, "b")]),
            fragments: vec![
                fragment_with_files(
                    1,
                    vec![data_file("a.lance", vec![0]), data_file("b.lance", vec![1])],
                ),
                fragment_with_files(
                    2,
                    vec![
                        data_file("a2.lance", vec![0]),
                        data_file("b2.lance", vec![1]),
                    ],
                ),
                fragment_with_files(3, vec![data_file("b3.lance", vec![1])]),
            ],
        };
        assert!(actions_from_operation_with_manifest(&operation, &prior, &[]).is_none());
    }

    #[test]
    fn merge_on_stable_row_id_dataset_refreshes_version_metadata() {
        // A column-add Merge on a stable-row-id dataset translates fully:
        // `AddFields` plus a trailing `RefreshRowVersionMetadata` bumping
        // `last_updated` for every fragment that gained a data file.
        let mut prior = merge_prior_manifest();
        prior.reader_feature_flags |= lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
        let operation = Operation::Merge {
            schema: schema_with(&[(0, "a"), (1, "b")]),
            fragments: vec![
                fragment_with_files(
                    1,
                    vec![data_file("a.lance", vec![0]), data_file("b.lance", vec![1])],
                ),
                fragment_with_files(
                    2,
                    vec![
                        data_file("a2.lance", vec![0]),
                        data_file("b2.lance", vec![1]),
                    ],
                ),
            ],
        };
        let user_action = actions_from_operation_with_manifest(&operation, &prior, &[])
            .expect("a stable-row-id column-add Merge translates");
        let actions = user_action.actions.as_slice();
        let [a0, a1] = actions else {
            panic!("expected 2 actions, got {actions:?}");
        };
        assert!(a0.as_any().is::<AddFields>());
        let refresh = cast::<RefreshRowVersionMetadata>(a1.as_ref());
        let RowVersionRefresh::MergedColumns { fragment_ids } = &refresh.mode else {
            panic!("expected MergedColumns, got {:?}", refresh.mode);
        };
        assert_eq!(fragment_ids, &vec![1u64, 2]);
    }

    #[test]
    fn merge_on_stable_row_id_dataset_with_no_new_files_emits_no_refresh() {
        // A Merge that adds a field but writes no per-fragment data has no
        // fragment to refresh, so only `AddFields` is emitted.
        let mut prior = merge_prior_manifest();
        prior.reader_feature_flags |= lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
        prior.fragments = Arc::new(vec![]);
        let operation = Operation::Merge {
            schema: schema_with(&[(0, "a"), (1, "b")]),
            fragments: vec![],
        };
        let user_action = actions_from_operation_with_manifest(&operation, &prior, &[])
            .expect("a stable-row-id schema-only Merge translates");
        let actions = user_action.actions.as_slice();
        assert!(
            actions.len() == 1 && actions[0].as_any().is::<AddFields>(),
            "unexpected action list: {actions:?}"
        );
    }

    fn recast_prior_manifest() -> Manifest {
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a"), (1, "b")]);
        manifest.fragments = Arc::new(vec![
            fragment_with_files(
                1,
                vec![data_file("a.lance", vec![0]), data_file("b.lance", vec![1])],
            ),
            fragment_with_files(
                2,
                vec![
                    data_file("a2.lance", vec![0]),
                    data_file("b2.lance", vec![1]),
                ],
            ),
        ]);
        manifest.max_fragment_id = Some(2);
        manifest
    }

    #[test]
    fn merge_recast_column_translates() {
        // `alter_columns` casting `b`'s type replaces field 1 with field 2 at
        // the same position: `b`'s old data file is dropped and the recast file
        // appended. The merge decomposes into `[AddFields, ChangeSchema]`.
        let prior = recast_prior_manifest();
        let recast_schema = schema_with(&[(0, "a"), (2, "b")]);
        let operation = Operation::Merge {
            schema: recast_schema.clone(),
            fragments: vec![
                fragment_with_files(
                    1,
                    vec![
                        data_file("a.lance", vec![0]),
                        data_file("b_cast.lance", vec![2]),
                    ],
                ),
                fragment_with_files(
                    2,
                    vec![
                        data_file("a2.lance", vec![0]),
                        data_file("b2_cast.lance", vec![2]),
                    ],
                ),
            ],
        };

        let user_action = actions_from_operation_with_manifest(&operation, &prior, &[])
            .expect("a type-cast Merge translates");
        let actions = user_action.actions.as_slice();
        let [a0, a1] = actions else {
            panic!("expected [AddFields, ChangeSchema], got {actions:?}");
        };
        let add = cast::<AddFields>(a0.as_ref());
        let change = cast::<ChangeSchema>(a1.as_ref());
        assert_eq!(add.new_fields, vec![field(2, "b")]);
        assert_eq!(
            add.fragment_files,
            vec![
                (1u64, vec![data_file("b_cast.lance", vec![2])]),
                (2u64, vec![data_file("b2_cast.lance", vec![2])]),
            ]
        );
        assert_eq!(change.schema, recast_schema);

        // Applying the translated actions reproduces the Merge's manifest.
        let mut manifest = prior;
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut vec![]).unwrap();
        }
        assert_eq!(manifest.schema, recast_schema);
        let Operation::Merge { fragments, .. } = &operation else {
            unreachable!()
        };
        assert_eq!(manifest.fragments.as_ref(), fragments);
    }

    #[test]
    fn merge_recast_only_column_drops_emptied_file() {
        // Casting the sole column leaves its old data file covering no live
        // field; `ChangeSchema`'s pruning drops it, mirroring the legacy
        // `remove_tombstoned_data_files`.
        let mut prior = empty_manifest();
        prior.schema = schema_with(&[(0, "a")]);
        prior.fragments = Arc::new(vec![fragment_with_files(
            1,
            vec![data_file("a.lance", vec![0])],
        )]);
        prior.max_fragment_id = Some(1);

        let recast_schema = schema_with(&[(1, "a")]);
        let operation = Operation::Merge {
            schema: recast_schema.clone(),
            fragments: vec![fragment_with_files(
                1,
                vec![data_file("a_cast.lance", vec![1])],
            )],
        };

        let user_action = actions_from_operation_with_manifest(&operation, &prior, &[])
            .expect("a sole-column type-cast Merge translates");
        let mut manifest = prior;
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut vec![]).unwrap();
        }
        assert_eq!(manifest.schema, recast_schema);
        assert_eq!(
            manifest.fragments[0].files,
            vec![data_file("a_cast.lance", vec![1])]
        );
    }

    #[test]
    fn merge_recast_on_stable_row_id_dataset_refreshes_version_metadata() {
        // A type-cast Merge on a stable-row-id dataset adds a trailing
        // `RefreshRowVersionMetadata` for every fragment whose files changed.
        let mut prior = recast_prior_manifest();
        prior.reader_feature_flags |= lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
        let operation = Operation::Merge {
            schema: schema_with(&[(0, "a"), (2, "b")]),
            fragments: vec![
                fragment_with_files(
                    1,
                    vec![
                        data_file("a.lance", vec![0]),
                        data_file("b_cast.lance", vec![2]),
                    ],
                ),
                fragment_with_files(
                    2,
                    vec![
                        data_file("a2.lance", vec![0]),
                        data_file("b2_cast.lance", vec![2]),
                    ],
                ),
            ],
        };
        let user_action = actions_from_operation_with_manifest(&operation, &prior, &[])
            .expect("a stable-row-id type-cast Merge translates");
        let actions = user_action.actions.as_slice();
        let [a0, a1, a2] = actions else {
            panic!("expected 3 actions, got {actions:?}");
        };
        assert!(a0.as_any().is::<AddFields>());
        assert!(a1.as_any().is::<ChangeSchema>());
        let refresh = cast::<RefreshRowVersionMetadata>(a2.as_ref());
        let RowVersionRefresh::MergedColumns { fragment_ids } = &refresh.mode else {
            panic!("expected MergedColumns, got {:?}", refresh.mode);
        };
        assert_eq!(fragment_ids, &vec![1u64, 2]);
    }

    #[test]
    fn actions_from_operation_with_manifest_delegates_non_merge() {
        // Non-Merge operations ignore the manifest and go through the pure
        // translation.
        let prior = merge_prior_manifest();
        let operation = Operation::Append {
            fragments: vec![sample_fragment(0)],
        };
        let user_action = actions_from_operation_with_manifest(&operation, &prior, &[]).unwrap();
        assert!(user_action.actions[0].as_any().is::<AddFragments>());
    }

    #[test]
    fn data_replacement_translates_one_action_per_group() {
        let operation = Operation::DataReplacement {
            replacements: vec![
                DataReplacementGroup(0, data_file("frag0-new.lance", vec![1, 2])),
                DataReplacementGroup(3, data_file("frag3-new.lance", vec![1, 2])),
            ],
        };

        let user_action =
            actions_from_operation(&operation).expect("DataReplacement should translate");
        assert_eq!(user_action.actions.len(), 2);

        let first = user_action.actions[0]
            .as_any()
            .downcast_ref::<ReplaceFragmentColumns>()
            .expect("expected ReplaceFragmentColumns");
        assert_eq!(first.fragment_id, 0);
        // The replaced field set is inferred from the swapped data file.
        assert_eq!(first.field_ids, vec![1, 2]);
        assert_eq!(first.new_data_files.len(), 1);
        assert_eq!(first.new_data_files[0].path, "frag0-new.lance");
        // Translation from the old format records no row offsets.
        assert!(first.updated_row_offsets.is_none());

        let second = user_action.actions[1]
            .as_any()
            .downcast_ref::<ReplaceFragmentColumns>()
            .expect("expected ReplaceFragmentColumns");
        assert_eq!(second.fragment_id, 3);
    }

    #[test]
    fn data_replacement_round_trips_through_actions() {
        let original = Operation::DataReplacement {
            replacements: vec![
                DataReplacementGroup(0, data_file("frag0-new.lance", vec![1, 2])),
                DataReplacementGroup(1, data_file("frag1-new.lance", vec![1, 2])),
            ],
        };

        let user_action =
            actions_from_operation(&original).expect("DataReplacement should translate");
        let round_tripped = operation_from_actions(&user_action)
            .expect("a run of ReplaceFragmentColumns should translate back");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn empty_data_replacement_translates_to_empty_action_list() {
        // An empty DataReplacement is a degenerate no-op; it still translates,
        // producing an empty action list (which `operation_from_actions`
        // cannot reconstruct — see `unknown_action_list_translation_returns_none`).
        let operation = Operation::DataReplacement {
            replacements: vec![],
        };
        let user_action =
            actions_from_operation(&operation).expect("DataReplacement should translate");
        assert!(user_action.actions.is_empty());
        assert!(operation_from_actions(&user_action).is_none());
    }

    #[test]
    fn multi_file_replace_fragment_columns_does_not_round_trip() {
        // A `ReplaceFragmentColumns` carrying more than one data file (only
        // reachable post-rebase) has no single `DataReplacementGroup` form.
        let user_action = UserAction::new(vec![Box::new(ReplaceFragmentColumns {
            fragment_id: 0,
            field_ids: vec![1, 2],
            new_data_files: vec![data_file("a.lance", vec![1]), data_file("b.lance", vec![2])],
            updated_row_offsets: None,
            replacement: ColumnReplacement::InPlace,
        })]);
        assert!(operation_from_actions(&user_action).is_none());
    }

    #[test]
    fn create_index_translates_to_remove_then_add_actions() {
        let removed = index_meta("old", &[0], Some(&[1]));
        let new_a = index_meta("a", &[0], Some(&[1, 2]));
        let new_b = index_meta("b", &[1], Some(&[3]));
        let original = Operation::CreateIndex {
            new_indices: vec![new_a.clone(), new_b.clone()],
            removed_indices: vec![removed.clone()],
        };

        let user_action = actions_from_operation(&original).expect("CreateIndex should translate");
        // RemoveIndex precedes AddIndex so a UUID in both sets ends up added.
        let actions = user_action.actions.as_slice();
        let [a0, a1, a2] = actions else {
            panic!("expected 3 actions, got {actions:?}");
        };
        let ri = cast::<RemoveIndex>(a0.as_ref());
        let ai_a = cast::<AddIndex>(a1.as_ref());
        let ai_b = cast::<AddIndex>(a2.as_ref());
        assert_eq!(ri.uuid, removed.uuid);
        assert_eq!(ai_a.index, new_a);
        assert_eq!(ai_a.coverage, FragmentCoverage::Bitmap(bitmap(&[1, 2])));
        assert_eq!(ai_b.index, new_b);
        assert_eq!(ai_b.coverage, FragmentCoverage::Bitmap(bitmap(&[3])));
    }

    #[test]
    fn create_index_without_removals_round_trips() {
        let new_a = index_meta("a", &[0], Some(&[1, 2]));
        let new_b = index_meta("b", &[1], Some(&[3]));
        let original = Operation::CreateIndex {
            new_indices: vec![new_a, new_b],
            removed_indices: vec![],
        };

        let user_action = actions_from_operation(&original).unwrap();
        let round_tripped = operation_from_actions(&user_action)
            .expect("a pure AddIndex list rebuilds CreateIndex");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn create_index_with_removals_is_forward_only() {
        // RemoveIndex keeps only a UUID, so a CreateIndex carrying removed
        // indices cannot be rebuilt from its action list.
        let original = Operation::CreateIndex {
            new_indices: vec![index_meta("a", &[0], Some(&[1]))],
            removed_indices: vec![index_meta("old", &[0], Some(&[1]))],
        };
        let user_action = actions_from_operation(&original).unwrap();
        assert!(operation_from_actions(&user_action).is_none());
    }

    #[test]
    fn create_index_without_fragment_bitmap_does_not_translate() {
        // A regular column index with no frozen coverage is the legacy
        // on-disk shape and cannot become an `AddIndex` criterion.
        let original = Operation::CreateIndex {
            new_indices: vec![index_meta("a", &[0], None)],
            removed_indices: vec![],
        };
        assert!(actions_from_operation(&original).is_none());
    }

    #[test]
    fn create_index_translates_system_index_without_bitmap_as_unbound() {
        // System indices (the inline MemWAL state, for example) carry
        // `fragment_bitmap = None` by design — they cover no data fragments.
        // Translation must succeed with `FragmentCoverage::Unbound` so the
        // action-resolver path is reachable for MemWAL init.
        let mem_wal = index_meta(MEM_WAL_INDEX_NAME, &[], None);
        let original = Operation::CreateIndex {
            new_indices: vec![mem_wal.clone()],
            removed_indices: vec![],
        };

        let user_action =
            actions_from_operation(&original).expect("system-index CreateIndex translates");
        assert_eq!(user_action.actions.len(), 1);
        let add = cast::<AddIndex>(user_action.actions[0].as_ref());
        assert_eq!(add.coverage, FragmentCoverage::Unbound);
        assert_eq!(add.index, mem_wal);

        // Round-trip through `Action::apply` preserves the `None` bitmap.
        let mut manifest = empty_manifest();
        let mut indices = vec![];
        for action in &user_action.actions {
            action.apply(&mut manifest, &mut indices).unwrap();
        }
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, MEM_WAL_INDEX_NAME);
        assert_eq!(indices[0].fragment_bitmap, None);

        // And `operation_from_actions` reconstructs the original op.
        let round_tripped = operation_from_actions(&user_action)
            .expect("system-index CreateIndex rebuilds from actions");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn update_mem_wal_state_round_trips_through_actions() {
        let shard = Uuid::new_v4();
        let original = Operation::UpdateMemWalState {
            merged_generations: vec![MergedGeneration::new(shard, 5)],
        };

        let user_action = actions_from_operation(&original).unwrap();
        let actions = user_action.actions.as_slice();
        assert!(
            actions.len() == 1 && actions[0].as_any().is::<UpdateMergedGenerations>(),
            "unexpected action list: {actions:?}"
        );
        let round_tripped = operation_from_actions(&user_action)
            .expect("UpdateMergedGenerations rebuilds UpdateMemWalState");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn rewrite_is_deferred_to_the_legacy_path() {
        // Rewrite's fragment half needs the not-yet-defined RewriteFragments
        // action, so the whole operation stays untranslated for now.
        let operation = Operation::Rewrite {
            groups: vec![],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        };
        assert!(actions_from_operation(&operation).is_none());
    }

    #[test]
    fn update_config_round_trips_through_actions() {
        let original = Operation::UpdateConfig {
            config_updates: Some(umap(&[("lance.a", Some("1"))], false)),
            table_metadata_updates: Some(umap(&[("k", Some("v"))], false)),
            schema_metadata_updates: None,
            field_metadata_updates: HashMap::new(),
        };
        let user_action = actions_from_operation(&original).expect("UpdateConfig translates");
        let actions = user_action.actions.as_slice();
        assert!(
            actions.len() == 1 && actions[0].as_any().is::<UpdateConfig>(),
            "unexpected action list: {actions:?}"
        );
        let round_tripped = operation_from_actions(&user_action).expect("UpdateConfig rebuilds");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn update_config_with_schema_metadata_round_trips() {
        let original = Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: None,
            schema_metadata_updates: Some(umap(&[("s", Some("1"))], false)),
            field_metadata_updates: HashMap::new(),
        };
        let user_action = actions_from_operation(&original).expect("UpdateConfig translates");
        let actions = user_action.actions.as_slice();
        assert!(
            actions.len() == 2
                && actions[0].as_any().is::<UpdateConfig>()
                && actions[1].as_any().is::<UpdateSchemaMetadata>(),
            "unexpected action list: {actions:?}"
        );
        let round_tripped = operation_from_actions(&user_action).expect("UpdateConfig rebuilds");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn update_config_with_field_metadata_round_trips() {
        // Per-field metadata rides on the `UpdateSchemaMetadata` action
        // alongside the schema-level metadata half.
        let mut field_metadata_updates = HashMap::new();
        field_metadata_updates.insert(0, umap(&[("f", Some("1"))], false));
        field_metadata_updates.insert(2, umap(&[("g", None)], true));
        let original = Operation::UpdateConfig {
            config_updates: Some(umap(&[("lance.a", Some("1"))], false)),
            table_metadata_updates: None,
            schema_metadata_updates: Some(umap(&[("s", Some("1"))], false)),
            field_metadata_updates,
        };
        let user_action = actions_from_operation(&original).expect("UpdateConfig translates");
        let actions = user_action.actions.as_slice();
        assert!(
            actions.len() == 2
                && actions[0].as_any().is::<UpdateConfig>()
                && actions[1].as_any().is::<UpdateSchemaMetadata>(),
            "unexpected action list: {actions:?}"
        );
        let round_tripped = operation_from_actions(&user_action).expect("UpdateConfig rebuilds");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn update_config_with_only_field_metadata_emits_schema_metadata_action() {
        // An `UpdateConfig` whose sole non-empty half is per-field metadata
        // still emits the `UpdateSchemaMetadata` action (with no schema-level
        // metadata) so the schema-side conflict is not silently lost.
        let mut field_metadata_updates = HashMap::new();
        field_metadata_updates.insert(0, umap(&[("f", Some("1"))], false));
        let original = Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: None,
            schema_metadata_updates: None,
            field_metadata_updates,
        };
        let user_action = actions_from_operation(&original).expect("UpdateConfig translates");
        let actions = user_action.actions.as_slice();
        let [a0, a1] = actions else {
            panic!("expected 2 actions, got {actions:?}");
        };
        assert!(a0.as_any().is::<UpdateConfig>());
        let meta = cast::<UpdateSchemaMetadata>(a1.as_ref());
        assert!(meta.metadata_updates.is_none());
        assert!(meta.field_metadata_updates.contains_key(&0));
        let round_tripped = operation_from_actions(&user_action).expect("UpdateConfig rebuilds");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn add_bases_round_trips_through_actions() {
        let original = Operation::UpdateBases {
            new_bases: vec![base_path(0, Some("a"), "/a")],
        };
        let user_action = actions_from_operation(&original).expect("UpdateBases translates");
        let actions = user_action.actions.as_slice();
        assert!(
            actions.len() == 1 && actions[0].as_any().is::<AddBases>(),
            "unexpected action list: {actions:?}"
        );
        let round_tripped = operation_from_actions(&user_action).expect("AddBases rebuilds");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn reserve_fragments_round_trips_through_actions() {
        let original = Operation::ReserveFragments { num_fragments: 6 };
        let user_action = actions_from_operation(&original).expect("ReserveFragments translates");
        let actions = user_action.actions.as_slice();
        assert!(
            actions.len() == 1 && actions[0].as_any().is::<ReserveFragmentIds>(),
            "unexpected action list: {actions:?}"
        );
        let round_tripped =
            operation_from_actions(&user_action).expect("ReserveFragmentIds rebuilds");
        assert_eq!(original, round_tripped);
    }

    /// Systematic Op↔Actions roundtrip and decline-contract tests.
    ///
    /// The ad-hoc tests above each pin one variant's translation in detail
    /// (action shape, apply effect, edge cases). This submodule covers the
    /// *parity surface* itself: every lossless `Operation` shape round-trips
    /// through `actions_from_operation → operation_from_actions`, and every
    /// forward-only shape (`Delete`, `Merge`, `CreateIndex` with removals,
    /// `Rewrite`, general `Update`) is declined explicitly. Adding a new
    /// `Operation` variant should add a case here too, so the decline
    /// contract `ActionRebase`'s splice pass depends on does not silently
    /// regress.
    mod roundtrip {
        use rstest::rstest;

        use super::*;
        use crate::dataset::write::merge_insert::inserted_rows::FilterType;

        /// Op→Actions→Op symmetry for variants the translator declares
        /// lossless. Cases are kept minimal — the ad-hoc tests above cover
        /// payload edge cases per variant; this one just proves the round-trip
        /// holds across the full set.
        #[rstest]
        #[case::append_empty(Operation::Append { fragments: vec![] })]
        #[case::append_single(Operation::Append { fragments: vec![sample_fragment(0)] })]
        #[case::append_multi(Operation::Append {
            fragments: vec![sample_fragment(0), sample_fragment(1), sample_fragment(2)],
        })]
        #[case::project(Operation::Project { schema: schema_with(&[(0, "a")]) })]
        #[case::overwrite_no_config(Operation::Overwrite {
            fragments: vec![sample_fragment(0), sample_fragment(1)],
            schema: schema_with(&[(0, "a")]),
            config_upsert_values: None,
            initial_bases: None,
        })]
        #[case::overwrite_with_config(Operation::Overwrite {
            fragments: vec![sample_fragment(0)],
            schema: schema_with(&[(0, "a")]),
            config_upsert_values: Some(HashMap::from([("k".to_string(), "v".to_string())])),
            initial_bases: None,
        })]
        #[case::update_config_plain(Operation::UpdateConfig {
            config_updates: Some(metadata_update(&[("k", Some("v"))], false)),
            table_metadata_updates: None,
            schema_metadata_updates: None,
            field_metadata_updates: HashMap::new(),
        })]
        #[case::update_config_with_schema_metadata(Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: None,
            schema_metadata_updates: Some(metadata_update(&[("sk", Some("sv"))], false)),
            field_metadata_updates: HashMap::new(),
        })]
        #[case::reserve_fragments(Operation::ReserveFragments { num_fragments: 4 })]
        #[case::update_mem_wal_state(Operation::UpdateMemWalState {
            merged_generations: vec![MergedGeneration {
                shard_id: Uuid::nil(),
                generation: 1,
            }],
        })]
        #[case::update_bases(Operation::UpdateBases {
            new_bases: vec![base_path(0, Some("b"), "/tmp/b")],
        })]
        #[case::data_replacement_multi(Operation::DataReplacement {
            replacements: vec![
                DataReplacementGroup(1, data_file("f1.lance", vec![0])),
                DataReplacementGroup(2, data_file("f2.lance", vec![0])),
            ],
        })]
        #[case::create_index_no_removals(Operation::CreateIndex {
            new_indices: vec![
                index_meta("idx_a", &[0], Some(&[1, 2])),
                index_meta("idx_b", &[0], Some(&[3])),
            ],
            removed_indices: vec![],
        })]
        fn lossless_variants_round_trip(#[case] original: Operation) {
            let actions = actions_from_operation(&original)
                .unwrap_or_else(|| panic!("variant must translate: {original:?}"));
            let round_tripped = operation_from_actions(&actions)
                .unwrap_or_else(|| panic!("variant must rebuild: {original:?}"));
            assert_eq!(original, round_tripped);
        }

        /// The reverse-only direction: a lone `AddFragments` carrying an
        /// `inserted_rows_filter` rebuilds an `Update`. The forward direction
        /// (`Update → AddFragments`) is *not* lossless even for a pure
        /// merge-insert — `update_to_actions` running on a stable-row-id
        /// dataset always emits the `RefreshRowVersionMetadata` (and, when
        /// physical rows are known, the `RebindIndexCoverage`) the legacy
        /// `build_manifest` arm performed inline, so the action list is no
        /// longer the lone-`AddFragments` shape `operation_from_actions`
        /// recognises. The detailed forward-direction Update tests above
        /// cover the action-shape contract.
        #[test]
        fn add_fragments_with_filter_rebuilds_as_pure_insert_update() {
            let filter = KeyExistenceFilter {
                field_ids: vec![0],
                filter: FilterType::ExactSet(HashSet::from([1, 2])),
            };
            let actions = UserAction::new(vec![Box::new(AddFragments {
                fragments: vec![sample_fragment(0), sample_fragment(1)],
                inserted_rows_filter: Some(filter.clone()),
            })]);
            match operation_from_actions(&actions).expect("rebuilds as Update") {
                Operation::Update {
                    new_fragments,
                    inserted_rows_filter: Some(rebuilt),
                    removed_fragment_ids,
                    updated_fragments,
                    ..
                } => {
                    assert_eq!(new_fragments.len(), 2);
                    assert_eq!(rebuilt, filter);
                    assert!(removed_fragment_ids.is_empty());
                    assert!(updated_fragments.is_empty());
                }
                other => panic!("expected Update, got {other:?}"),
            }
        }

        /// Forward-only shapes: `actions_from_operation` produces actions
        /// whose `operation_from_actions` is `None`. This is the decline
        /// contract the legacy `build_manifest` fallback depends on — if any
        /// of these starts round-tripping, the action vocabulary has gained a
        /// capability and the corresponding fallback can be retired.
        #[rstest]
        // `Delete` action shape (UpdateDeletionVector + RemoveFragments) drops
        // the source `Fragment` payloads, so a `Delete` cannot be rebuilt.
        #[case::delete(Operation::Delete {
            updated_fragments: vec![fragment_with_deletion(1, deletion_file(1))],
            deleted_fragment_ids: vec![2],
            predicate: "x = 1".into(),
        })]
        // `CreateIndex` with `removed_indices` emits a `RemoveIndex` that keeps
        // only the UUID, so the original `IndexMetadata` cannot be rebuilt.
        #[case::create_index_with_removals(Operation::CreateIndex {
            new_indices: vec![index_meta("idx_new", &[0], Some(&[1]))],
            removed_indices: vec![index_meta("idx_old", &[0], Some(&[2]))],
        })]
        fn forward_only_variants_decline_reverse_translation(#[case] op: Operation) {
            let actions = actions_from_operation(&op)
                .unwrap_or_else(|| panic!("forward-only variant must translate: {op:?}"));
            assert!(
                operation_from_actions(&actions).is_none(),
                "forward-only variant unexpectedly rebuilt: {op:?}",
            );
        }

        /// An empty `UserAction` cannot be rebuilt — every legacy `Operation`
        /// decomposes to at least one action, so an empty list is not a
        /// recognised shape. The ad-hoc test above pins this; the rstest
        /// version here keeps the decline contract together in one place.
        #[test]
        fn empty_action_list_declines() {
            assert!(operation_from_actions(&UserAction::new(vec![])).is_none());
        }

        /// Variants that have no action decomposition at all: the forward
        /// translator returns `None`, which the commit path reads as "route
        /// this through the legacy `build_manifest` arm". Pinning this here
        /// guards the legacy-fallback contract — if any of these starts
        /// translating, the action vocabulary has grown to cover it and the
        /// corresponding legacy arm can be retired.
        #[rstest]
        #[case::restore(Operation::Restore { version: 7 })]
        #[case::clone_op(Operation::Clone {
            is_shallow: true,
            ref_name: Some("main".into()),
            ref_version: 3,
            ref_path: "/tmp/src".into(),
            branch_name: None,
        })]
        fn forward_translation_declines(#[case] op: Operation) {
            assert!(
                actions_from_operation(&op).is_none(),
                "variant must decline forward translation: {op:?}",
            );
            // `actions_from_operation_with_manifest` delegates the non-special
            // cases to `actions_from_operation`, so the decline must hold
            // through both entry points.
            let manifest = manifest_with_fragments(&[]);
            assert!(
                actions_from_operation_with_manifest(&op, &manifest, &[]).is_none(),
                "variant must decline forward translation via _with_manifest: {op:?}",
            );
        }

        /// `Merge` decomposes to `AddFields` (+ optional `ChangeSchema` /
        /// `RefreshRowVersionMetadata`), which keep only the *added* fields
        /// and files — the prior manifest is needed to rebuild the *full*
        /// `Merge::fragments` payload, so the reverse direction declines.
        #[test]
        fn merge_action_shape_declines() {
            let prior = merge_prior_manifest();
            let merge_schema = schema_with(&[(0, "a"), (1, "b")]);
            let merge = Operation::Merge {
                schema: merge_schema,
                fragments: vec![
                    fragment_with_files(
                        1,
                        vec![data_file("a.lance", vec![0]), data_file("b.lance", vec![1])],
                    ),
                    fragment_with_files(
                        2,
                        vec![
                            data_file("a2.lance", vec![0]),
                            data_file("b2.lance", vec![1]),
                        ],
                    ),
                ],
            };
            let actions = actions_from_operation_with_manifest(&merge, &prior, &[])
                .expect("Merge translates");
            assert!(operation_from_actions(&actions).is_none());
        }

        /// `Rewrite` decomposes to `RewriteFragments` (+ index half), which
        /// keeps only old fragment *ids* and drops the rewritten index's
        /// details — the original `RewriteGroup` and `RewrittenIndex`
        /// payloads cannot be rebuilt, so the reverse direction declines.
        #[test]
        fn rewrite_action_shape_declines() {
            let manifest = manifest_with_fragments(&[1, 2, 3]);
            let old_index = index_meta("idx", &[0], Some(&[1, 2]));
            let new_uuid = Uuid::new_v4();
            let rewrite = Operation::Rewrite {
                groups: vec![rewrite_group(&[1, 2], &[10])],
                rewritten_indices: vec![rewritten_index(old_index.uuid, new_uuid)],
                frag_reuse_index: None,
            };
            let actions = actions_from_operation_with_manifest(
                &rewrite,
                &manifest,
                std::slice::from_ref(&old_index),
            )
            .expect("Rewrite translates");
            assert!(operation_from_actions(&actions).is_none());
        }
    }
}
