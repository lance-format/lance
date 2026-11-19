// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Footprints of the named [`Operation`]s.
//!
//! A named operation has no actions to ask, so its footprint is read straight
//! off the operation. Several carry post-images -- a fragment restated whole,
//! as `Delete` and `Update` do, or the whole schema, as `Merge` and `Project`
//! do -- and applying one replaces everything it names, not just the part the
//! operation meant to change. So whatever an operation restates, it reads
//! whole: any change to it since conflicts.
//!
//! Committed, only the facts are read (see [`Footprint`]), and a named
//! operation states those the same way an action set does.

use super::{ConfigMap, FieldRef, Footprint, IndexWrite, Part, Ref, field_refs};
use crate::format::key_existence::KeyExistenceFilter;
use crate::format::{Fragment, IndexMetadata, Manifest, RowIdMeta};
use crate::rowids::read_row_ids;
use crate::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use crate::system_index::mem_wal::{CompactedSsTable, MEM_WAL_INDEX_NAME};
use crate::transaction::action::{AddOverlays, ConfigUpdate};
use crate::transaction::{Operation, RewriteGroup, RewrittenIndex, Transaction};
use lance_core::datatypes::Schema;
use std::collections::HashSet;

impl Footprint {
    /// What `transaction`, which read `read`, reads and writes. An action
    /// set's footprint is its actions'.
    pub fn of(transaction: &Transaction, read: &Manifest) -> Self {
        Self::of_operation(&transaction.operation, read)
    }

    fn of_operation(operation: &Operation, read: &Manifest) -> Self {
        let mut footprint = Self::default();
        match operation {
            Operation::CompositeOperation(composite_operation) => {
                return Self::from(composite_operation);
            }
            Operation::Append { fragments } => footprint.append(fragments),
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                ..
            } => {
                for fragment in updated_fragments {
                    footprint.restate_with_deletions_added(fragment.id);
                }
                for fragment in deleted_fragment_ids {
                    footprint.remove_fragment(*fragment);
                }
            }
            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified,
                compacted_sstables,
                inserted_rows_filter,
                ..
            } => {
                // With new fragments, the rows deleted here are taken to have
                // moved there, taking the values as they were.
                let moves_rows = !new_fragments.is_empty();
                for fragment in removed_fragment_ids {
                    footprint.remove_fragment(*fragment);
                    if moves_rows {
                        footprint.relocate_rows_out_of(*fragment);
                    }
                }
                for fragment in updated_fragments {
                    footprint.restate_with_deletions_added(fragment.id);
                    if moves_rows {
                        footprint.relocate_rows_out_of(fragment.id);
                    }
                    footprint.write_field_data(
                        Ref::Committed(fragment.id),
                        fields_modified
                            .iter()
                            .map(|field| FieldRef::Committed(*field as i32)),
                        true,
                    );
                }
                footprint.append(new_fragments);
                footprint.update_facts(inserted_rows_filter.as_ref(), compacted_sstables);
            }
            // The replacement swaps files in the fragment as it is at commit,
            // not in a post-image of it.
            Operation::DataReplacement { replacements, .. } => {
                for replacement in replacements {
                    footprint.write_field_data(
                        Ref::Committed(replacement.0),
                        field_refs(&replacement.1.fields),
                        true,
                    );
                }
            }
            // Overlays are appended to the fragment as it is at commit, so this
            // is the same edit as the action's.
            Operation::DataOverlay { groups } => {
                for group in groups {
                    AddOverlays::new_unchecked(
                        Ref::Committed(group.fragment_id),
                        group.overlays.clone(),
                    )
                    .footprint(&mut footprint);
                }
            }
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            } => footprint.index_changes(new_indices, removed_indices),
            Operation::Rewrite {
                groups,
                rewritten_indices,
                frag_reuse_index,
            } => {
                for fragment in groups.iter().flat_map(|group| &group.old_fragments) {
                    footprint.move_rows_out_of(fragment.id);
                }
                footprint.rewrite_facts(groups, rewritten_indices, frag_reuse_index.is_some());
            }
            Operation::Overwrite {
                config_upsert_values,
                initial_bases,
                ..
            } => {
                footprint.reset_table();
                for key in config_upsert_values.iter().flat_map(|values| values.keys()) {
                    footprint.read(Part::MapEntry {
                        map: ConfigMap::Config,
                        key: key.clone(),
                    });
                }
                for base in initial_bases.iter().flatten() {
                    footprint.add_base(base);
                }
            }
            // A restore reinstates an older schema, and with it that schema's
            // metadata. A concurrent metadata edit it lands over is lost, and
            // legacy treats that pair as incompatible rather than letting the
            // restore win, so the restore reads every map it puts back.
            Operation::Restore { .. } => {
                footprint.reset_table();
                footprint.read(Part::Map(ConfigMap::TableMetadata));
                footprint.read(Part::Map(ConfigMap::SchemaMetadata));
                for field in read.schema.fields_pre_order() {
                    footprint.read(Part::Map(ConfigMap::Field(field.id)));
                }
            }
            Operation::UpdateConfig {
                config_updates,
                table_metadata_updates,
                schema_metadata_updates,
                field_metadata_updates,
            } => {
                // The same edit as an action set's, key for key, so the two
                // forms get the same verdict.
                let mut update = ConfigUpdate::default();
                if let Some(updates) = config_updates {
                    update = update.with_config(updates.clone());
                }
                if let Some(updates) = table_metadata_updates {
                    update = update.with_table_metadata(updates.clone());
                }
                if let Some(updates) = schema_metadata_updates {
                    update = update.with_schema_metadata(updates.clone());
                }
                for (field, updates) in field_metadata_updates {
                    update =
                        update.with_field_metadata(FieldRef::Committed(*field), updates.clone());
                }
                update.footprint(&mut footprint);
            }
            Operation::UpdateBases { new_bases } => {
                for base in new_bases {
                    footprint.add_base(base);
                }
            }
            // Ids are counters a reservation only advances; whatever later
            // claims a reserved id claims it as a part of its own.
            Operation::ReserveFragments { .. } => {}
            // A clone commits into a dataset of its own, and legacy finds it
            // compatible with every operation.
            Operation::Clone { .. } => {}
            // A merge installs the schema and the fragment list it carries,
            // built from what it read: every fragment it read is restated,
            // and one added since is dropped, as is every field added since.
            // Committed, it wrote the values of every field it rebound to a
            // new file in a fragment it read.
            Operation::Merge {
                fragments, schema, ..
            } => {
                footprint.read(Part::Schema);
                footprint.read(Part::Fragments);
                footprint.add_required_column();
                footprint.tightenings(&read.schema, schema);
                footprint.append(&merged_fragments_added(fragments, read));
                footprint
                    .did
                    .wrote_values
                    .extend(merged_values_written(fragments, read));
                footprint
                    .did
                    .replaced_fields
                    .extend(replaced_fields(&read.schema, schema));
            }
            // A projection installs the schema it carries, but prunes the files
            // of the fields it drops from the fragments as they are at commit.
            Operation::Project { schema, .. } => {
                footprint.read(Part::Schema);
                footprint.tightenings(&read.schema, schema);
            }
            // The MemWAL state is checked against the data it compacted, which
            // no part describes yet. An unknown operation may touch anything.
            Operation::UpdateMemWalState { .. } | Operation::Unknown { .. } => {
                footprint.mark_opaque()
            }
        }
        footprint
    }

    /// A fragment a `Delete` or `Update` restates whole from its read version,
    /// with deletions added. Any change to it since conflicts, a concurrent
    /// deletion included.
    fn restate_with_deletions_added(&mut self, fragment: u64) {
        self.read_whole_fragment(fragment);
    }

    /// Fragments added with rows that were not in the dataset before. Their
    /// ids are assigned at commit, so no concurrent writer can name one. Row
    /// ids they arrive carrying are claimed as an `AddFragment`'s are, and
    /// ids that do not decode make the footprint opaque.
    fn append(&mut self, fragments: &[Fragment]) {
        if fragments.is_empty() {
            return;
        }
        self.add_rows(true);
        for fragment in fragments {
            if let Some(RowIdMeta::Inline(data)) = &fragment.row_id_meta {
                match read_row_ids(data) {
                    Ok(sequence) => self.supply_row_ids(&sequence.to_treemap()),
                    Err(_) => self.mark_opaque(),
                }
            }
            for file in &fragment.files {
                self.write_field_data(Ref::Local(0), field_refs(&file.fields), true);
            }
        }
    }

    /// The fields a schema installed over `read` tightens to non-nullable. A
    /// tightening was checked against the values the operation read, so it
    /// holds only if none were written since.
    fn tightenings(&mut self, read: &Schema, installed: &Schema) {
        for field in read.fields_pre_order() {
            if let Some(kept) = installed.field_by_id(field.id)
                && field.nullable
                && !kept.nullable
            {
                self.require_no_values_written(FieldRef::Committed(field.id));
                self.add_required_column();
            }
        }
    }

    /// The key assertion and MemWAL compaction an `Update` carries.
    fn update_facts(
        &mut self,
        inserted_rows_filter: Option<&KeyExistenceFilter>,
        compacted_sstables: &[CompactedSsTable],
    ) {
        if let Some(filter) = inserted_rows_filter {
            self.assert_unique_keys(field_refs(&filter.field_ids).collect(), filter.clone());
        }
        if !compacted_sstables.is_empty() {
            self.rewrite_index(MEM_WAL_INDEX_NAME.to_string());
        }
    }

    /// The segments a `CreateIndex` adds and removes. Legacy replaces a
    /// same-named index rather than adding to it, so each new segment
    /// replaces the index, bringing what its bitmap names under it.
    fn index_changes(&mut self, new_indices: &[IndexMetadata], removed_indices: &[IndexMetadata]) {
        for index in new_indices {
            self.write_index(IndexWrite {
                name: index.name.clone(),
                identity: None,
                adds: index
                    .fragment_bitmap
                    .as_ref()
                    .map(|bitmap| bitmap.iter().map(u64::from).collect()),
                replaces: true,
            });
            for field in field_refs(&index.fields).chain(field_refs(&index.covering_fields)) {
                self.require_field(field);
            }
        }
        for index in removed_indices {
            self.write_segment(index.uuid);
            self.rewrite_index(index.name.clone());
        }
    }

    /// What a `Rewrite` does to the indices over the rows it moves.
    fn rewrite_facts(
        &mut self,
        groups: &[RewriteGroup],
        rewritten_indices: &[RewrittenIndex],
        writes_frag_reuse_index: bool,
    ) {
        self.did.compacted_fragments.extend(
            groups
                .iter()
                .flat_map(|group| &group.old_fragments)
                .map(|fragment| fragment.id),
        );
        self.remap_indices_over(
            groups
                .iter()
                .flat_map(|group| &group.old_fragments)
                .map(|fragment| fragment.id)
                .collect(),
        );
        for index in rewritten_indices {
            self.write_segment(index.old_id);
        }
        if writes_frag_reuse_index {
            self.rewrite_index(FRAG_REUSE_INDEX_NAME.to_string());
        }
    }
}

/// The fields the read version holds whose values a merge rewrote in a
/// fragment it read: those backed by a file the fragment did not have. The
/// first occurrence of a read id is that fragment, as in
/// [`merged_fragments_added`].
fn merged_values_written(fragments: &[Fragment], read: &Manifest) -> HashSet<i32> {
    let mut seen = HashSet::new();
    fragments
        .iter()
        .filter(|merged| seen.insert(merged.id))
        .filter_map(|merged| Some((merged, read.fragment_by_id(merged.id)?)))
        .flat_map(|(merged, was)| {
            merged
                .files
                .iter()
                .filter(move |file| !was.files.contains(file))
        })
        .flat_map(|file| file.fields.iter().copied())
        .filter(|field| *field >= 0 && read.schema.field_by_id(*field).is_some())
        .collect()
}

/// The fields of `read` that `installed` replaces by a field of the same
/// path with another id, as a cast does.
fn replaced_fields(read: &Schema, installed: &Schema) -> Vec<i32> {
    read.fields_pre_order()
        .filter(|field| installed.field_by_id(field.id).is_none())
        .filter(|field| {
            read.field_path(field.id)
                .is_ok_and(|path| installed.field(&path).is_some())
        })
        .map(|field| field.id)
        .collect()
}

/// The fragments in a merge's list that are not the merged form of one it
/// read. As the apply decides it: the first occurrence of a read id is that
/// fragment, and everything else is new.
fn merged_fragments_added(fragments: &[Fragment], read: &Manifest) -> Vec<Fragment> {
    let read_ids: HashSet<u64> = read.fragments.iter().map(|fragment| fragment.id).collect();
    let mut seen = HashSet::new();
    fragments
        .iter()
        .filter(|fragment| !read_ids.contains(&fragment.id) || !seen.insert(fragment.id))
        .cloned()
        .collect()
}
