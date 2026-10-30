// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Footprints of the named [`Operation`]s.
//!
//! A named operation has no actions to ask. Pending, its footprint is read
//! straight off the operation, and the mapping is coarse on purpose. Several
//! operations carry post-images -- a fragment restated whole, as `Delete` and
//! `Update` do, or the whole schema, as `Merge` and `Project` do -- and
//! applying one replaces everything it names, not just the part the operation
//! meant to change. So whatever an operation restates is claimed whole.
//!
//! Committed, the post-image is installed, and the manifests either side of it
//! say exactly what it changed (see `diff`). The operation supplies only what
//! the manifests cannot: the keys it inserted, whether the rows it added are
//! new, and the indices it touched.
//!
//! Lowering an operation to actions instead would under-claim on the pending
//! side for exactly that reason: the actions describe the intended delta, the
//! apply installs the post-image.

use super::{ConfigMap, Coordinate, FieldFacet, FieldRef, Footprint, Ref, field_refs};
use crate::format::key_existence::KeyExistenceFilter;
use crate::format::{BasePath, Fragment, IndexMetadata, Manifest};
use crate::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use crate::system_index::mem_wal::{CompactedSsTable, MEM_WAL_INDEX_NAME};
use crate::transaction::action::{AddBase, AddOverlays, ConfigUpdate, FieldMetadataUpdate};
use crate::transaction::{Operation, RewriteGroup, RewrittenIndex};
use lance_core::datatypes::Schema;
use std::collections::HashSet;

impl Footprint {
    /// The footprint of `operation`, pending against `read`. An action set's
    /// is its actions'.
    pub(super) fn of_operation(operation: &Operation, read: &Manifest) -> Self {
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
                    footprint.replace_fragment(fragment.id);
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
                for fragment in removed_fragment_ids {
                    footprint.remove_fragment(*fragment);
                }
                for fragment in updated_fragments {
                    footprint.replace_fragment(fragment.id);
                    // Already inside the fragment's region, but named so a key
                    // assertion over one of these columns sees the rewrite.
                    if !fields_modified.is_empty() {
                        footprint.write_field_data(
                            Ref::Committed(fragment.id),
                            fields_modified
                                .iter()
                                .map(|field| FieldRef::Committed(*field as i32)),
                            true,
                        );
                    }
                }
                footprint.append(new_fragments);
                footprint.update_facts(inserted_rows_filter.as_ref(), compacted_sstables);
            }
            Operation::DataReplacement { replacements } => {
                for replacement in replacements {
                    footprint.replace_fragment(replacement.0);
                    footprint.write_field_data(
                        Ref::Committed(replacement.0),
                        field_refs(&replacement.1.fields),
                        true,
                    );
                }
            }
            // Overlays are appended to the fragment as it is at commit, not
            // installed from a post-image, so this is the same edit as the
            // action's.
            Operation::DataOverlay { groups } => {
                for group in groups {
                    AddOverlays {
                        fragment: Ref::Committed(group.fragment_id),
                        overlays: group.overlays.clone(),
                        data_change: true,
                    }
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
                    footprint.remove_fragment(fragment.id);
                }
                // The rows are moved, not written, so the values are not new;
                // but they are still laid out in the types they were read in.
                for file in groups
                    .iter()
                    .flat_map(|group| &group.new_fragments)
                    .flat_map(|fragment| &fragment.files)
                {
                    for field in field_refs(&file.fields) {
                        footprint.require_field_facet(field, FieldFacet::Type);
                    }
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
                    footprint.write(Coordinate::ConfigEntry {
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
            // restore win, so the restore claims every map it puts back. The
            // read schema names every committed field a concurrent writer can.
            Operation::Restore { .. } => {
                footprint.reset_table();
                footprint.write_map(ConfigMap::SchemaMetadata);
                for field in read.schema.fields_pre_order() {
                    footprint.write_map(ConfigMap::Field(field.id));
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
                ConfigUpdate {
                    config: config_updates.clone(),
                    table_metadata: table_metadata_updates.clone(),
                    schema_metadata: schema_metadata_updates.clone(),
                    field_metadata: field_metadata_updates
                        .iter()
                        .map(|(field, updates)| FieldMetadataUpdate {
                            field: FieldRef::Committed(*field),
                            updates: updates.clone(),
                        })
                        .collect(),
                }
                .footprint(&mut footprint);
            }
            Operation::UpdateBases { new_bases } => {
                for base in new_bases {
                    footprint.add_base(base);
                }
            }
            // Ids are counters a reservation only advances; whatever later
            // claims a reserved id claims it as a coordinate of its own.
            Operation::ReserveFragments { .. } => {}
            // A clone commits into a dataset of its own, and legacy finds it
            // compatible with every operation.
            Operation::Clone { .. } => {}
            // A merge installs the schema and the fragment list it carries,
            // built from what it read. Every fragment it read is restated, and
            // one added since is dropped, as is every field added since.
            Operation::Merge {
                fragments, schema, ..
            } => {
                footprint.replace_schema();
                footprint.schema_changes(&read.schema, schema);
                for fragment in read.fragments.iter() {
                    footprint.replace_fragment(fragment.id);
                }
                footprint.require_no_rows_added();
                footprint.append(&merged_fragments_added(fragments, read));
            }
            // A projection installs the schema it carries, but prunes the files
            // of the fields it drops from the fragments as they are at commit.
            Operation::Project { schema, .. } => {
                footprint.replace_schema();
                footprint.schema_changes(&read.schema, schema);
            }
            // The MemWAL state is checked against the data it compacted, which
            // no coordinate describes yet. An unknown operation may touch
            // anything.
            Operation::UpdateMemWalState { .. } | Operation::Unknown { .. } => {
                footprint.opaque = true
            }
        }
        footprint
    }

    /// The footprint of `operation`, which committed `after` on top of
    /// `before`.
    pub(super) fn of_committed_operation(
        operation: &Operation,
        before: &Manifest,
        after: &Manifest,
    ) -> Self {
        match operation {
            Operation::CompositeOperation(composite_operation) => Self::from(composite_operation),
            // A reset conflicts with every set that read before it, so what
            // else it changed adds nothing. The rest change nothing the
            // manifests show, or cannot be described by them.
            Operation::Overwrite { .. }
            | Operation::Restore { .. }
            | Operation::Clone { .. }
            | Operation::ReserveFragments { .. }
            | Operation::UpdateMemWalState { .. }
            | Operation::Unknown { .. } => Self::of_operation(operation, before),
            _ => {
                let moves_rows = matches!(operation, Operation::Rewrite { .. });
                let mut footprint = Self::of_manifest_change(before, after, moves_rows);
                footprint.operation_facts(operation);
                footprint
            }
        }
    }

    /// What a committed operation did that the manifests either side of it
    /// do not show.
    fn operation_facts(&mut self, operation: &Operation) {
        match operation {
            Operation::Append { fragments } => {
                if !fragments.is_empty() {
                    self.insert_rows();
                }
            }
            Operation::Update {
                updated_fragments,
                new_fragments,
                fields_modified,
                inserted_rows_filter,
                compacted_sstables,
                ..
            } => {
                // The columns it rewrote, named whether or not the rewrite
                // changed which file backs them, so a key assertion over one
                // sees the new values.
                if !fields_modified.is_empty() {
                    for fragment in updated_fragments {
                        self.write_field_data(
                            Ref::Committed(fragment.id),
                            fields_modified
                                .iter()
                                .map(|field| FieldRef::Committed(*field as i32)),
                            true,
                        );
                    }
                }
                if !new_fragments.is_empty() {
                    self.insert_rows();
                    // A row-moving update deletes rows from these fragments and
                    // writes them anew elsewhere, taking the values as they
                    // were. A column written into one of them since it read
                    // misses the moved rows, so the move writes the whole of
                    // what the fragment's files hold.
                    for fragment in updated_fragments {
                        self.write(Coordinate::FragmentFiles(fragment.id));
                    }
                }
                self.update_facts(inserted_rows_filter.as_ref(), compacted_sstables);
            }
            Operation::Merge { .. } => {
                if self.rows_added {
                    self.insert_rows();
                }
            }
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            } => self.index_changes(new_indices, removed_indices),
            Operation::Rewrite {
                groups,
                rewritten_indices,
                frag_reuse_index,
            } => self.rewrite_facts(groups, rewritten_indices, frag_reuse_index.is_some()),
            _ => {}
        }
    }

    /// Fragments added with rows that were not in the dataset before. Their
    /// ids are assigned at commit, so no concurrent writer can name one.
    fn append(&mut self, fragments: &[Fragment]) {
        if fragments.is_empty() {
            return;
        }
        self.add_rows();
        self.insert_rows();
        for file in fragments.iter().flat_map(|fragment| &fragment.files) {
            self.write_field_data(Ref::Local(0), field_refs(&file.fields), true);
        }
    }

    /// The fields a schema installed over `read` drops, and the ones it
    /// tightens to non-nullable. A tightening was checked against the values
    /// the operation read, so it holds only if none were written since.
    fn schema_changes(&mut self, read: &Schema, installed: &Schema) {
        for field in read.fields_pre_order() {
            let reference = FieldRef::Committed(field.id);
            match installed.field_by_id(field.id) {
                None => self.remove_field(reference),
                Some(kept) if field.nullable && !kept.nullable => {
                    self.require_no_values_written(reference);
                    self.require_no_rows_added();
                }
                Some(_) => {}
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

    /// The segments a `CreateIndex` adds and removes.
    fn index_changes(&mut self, new_indices: &[IndexMetadata], removed_indices: &[IndexMetadata]) {
        for index in new_indices {
            self.create_index(index);
        }
        for index in removed_indices {
            self.write(Coordinate::IndexSegment(index.uuid));
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
        for fragment in groups.iter().flat_map(|group| &group.old_fragments) {
            self.remaps_indices_over.insert(fragment.id);
        }
        for index in rewritten_indices {
            self.write(Coordinate::IndexSegment(index.old_id));
        }
        if writes_frag_reuse_index {
            self.rewrite_index(FRAG_REUSE_INDEX_NAME.to_string());
        }
    }

    /// An index segment built over `index.fields` in the fragments its bitmap
    /// covers. Legacy replaces a same-named index rather than adding to it, so
    /// its reach on the index is unstated.
    fn create_index(&mut self, index: &IndexMetadata) {
        self.rewrite_index(index.name.clone());
        let fields: Vec<FieldRef> = field_refs(&index.fields)
            .chain(field_refs(&index.covering_fields))
            .collect();
        for field in fields.iter().copied() {
            self.require_field_facet(field, FieldFacet::Existence);
            self.require_field_facet(field, FieldFacet::Type);
        }
        for fragment in index.fragment_bitmap.iter().flatten() {
            self.require_field_data_unless_removed(u64::from(fragment), fields.iter().copied());
        }
    }

    pub(super) fn add_base(&mut self, base: &BasePath) {
        AddBase {
            local: 0,
            base: base.clone(),
        }
        .footprint(self);
    }
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

#[cfg(test)]
mod tests {
    use super::super::Mode;
    use super::*;
    use crate::format::key_existence::{FilterType, KeyExistenceFilter};
    use crate::format::{DataFile, DataStorageFormat, DeletionFile, DeletionFileType, Manifest};
    use crate::transaction::action::test_support::added_field;
    use crate::transaction::action::{
        Action, AddDataFile, AddField, AddFragment, AddIndexSegment, AlterField, AssertUniqueKeys,
        CompositeOperation, TombstoneFieldData, UserStep, Verdict,
    };
    use crate::transaction::test_support::{
        default_build_config, overlay_with_field, sample_index_metadata, sample_manifest,
    };
    use crate::transaction::{
        DataOverlayGroup, DataReplacementGroup, RewriteGroup, Transaction, UpdateMap,
        UpdateMapEntry, UpdateMode,
    };
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_file::version::ConcreteFileVersion;
    use rstest::rstest;
    use std::collections::HashMap;
    use std::sync::Arc;
    use uuid::Uuid;

    fn fragment(id: u64) -> Fragment {
        let mut fragment = Fragment::new(id);
        fragment.physical_rows = Some(10);
        fragment.files.push(data_file());
        fragment
    }

    fn data_file() -> DataFile {
        DataFile::new(
            "data/x.lance",
            vec![0],
            vec![0],
            ConcreteFileVersion::V2_0,
            None,
            None,
        )
    }

    fn update(
        updated: Vec<Fragment>,
        new: Vec<Fragment>,
        fields_modified: Vec<u32>,
        inserted_rows_filter: Option<KeyExistenceFilter>,
    ) -> Operation {
        Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: updated,
            new_fragments: new,
            fields_modified,
            compacted_sstables: vec![],
            mem_wal_dir_id: None,
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteRows),
            inserted_rows_filter,
            updated_fragment_offsets: None,
        }
    }

    fn update_map(key: &str) -> UpdateMap {
        UpdateMap {
            update_entries: vec![UpdateMapEntry::from((key, Some("v")))],
            replace: false,
        }
    }

    fn update_config(schema_key: Option<&str>, table_key: Option<&str>) -> Operation {
        Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: table_key.map(update_map),
            schema_metadata_updates: schema_key.map(update_map),
            field_metadata_updates: HashMap::new(),
        }
    }

    fn config_action(schema_key: Option<&str>, table_key: Option<&str>) -> Action {
        Action::ConfigUpdate(ConfigUpdate {
            schema_metadata: schema_key.map(update_map),
            table_metadata: table_key.map(update_map),
            ..Default::default()
        })
    }

    fn append_actions() -> Vec<Action> {
        vec![
            Action::AddFragment(AddFragment {
                id: Ref::Local(0),
                physical_rows: 10,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(0),
                file: without_fields(data_file()),
                field_ids: vec![FieldRef::Committed(0)],
                data_change: true,
                base: None,
            }),
        ]
    }

    fn key_filter(hashes: &[u64]) -> KeyExistenceFilter {
        KeyExistenceFilter {
            field_ids: vec![0],
            filter: FilterType::ExactSet(hashes.iter().copied().collect()),
        }
    }

    fn key_assertion(hashes: &[u64]) -> Action {
        Action::AssertUniqueKeys(AssertUniqueKeys {
            key_fields: vec![FieldRef::Committed(0)],
            filter: key_filter(hashes),
        })
    }

    fn actions(actions: Vec<Action>) -> Operation {
        Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
            "step", actions,
        )]))
    }

    /// Whether `ours`, read at the sample manifest, conflicts with `theirs`,
    /// which committed on top of it.
    fn conflicts(ours: &Operation, theirs: &Operation) -> bool {
        let mut before = sample_manifest();
        before.fragments = Arc::new(vec![fragment(0), fragment(1)]);
        let pending = Footprint::pending(&Transaction::new(1, ours.clone(), None), &before);
        // These never build a manifest here: a restore is built from the one
        // it restores, a clone commits elsewhere, a MemWAL state update needs
        // a MemWAL the sample lacks, and an unknown operation cannot be built
        // at all. None of their footprints look at one.
        let committed = if matches!(
            theirs,
            Operation::Restore { .. }
                | Operation::Clone { .. }
                | Operation::UpdateMemWalState { .. }
                | Operation::Unknown { .. }
        ) {
            Footprint::committed(&Transaction::new(1, theirs.clone(), None), &before, &before)
        } else {
            let (transaction, after) = commit(&before, theirs.clone());
            Footprint::committed(&transaction, &before, &after)
        };
        match pending.compare(&committed) {
            Verdict::Compatible => false,
            Verdict::Conflict => true,
            Verdict::Rebase(overlaps) => {
                assert!(overlaps.is_empty(), "no overlap can be rebased past yet");
                false
            }
        }
    }

    /// An action set doing what a legacy `Append` and a legacy `UpdateConfig`
    /// do, in one commit, against every named operation in both orders. It is
    /// at least as permissive as those two legacy operations would be, except
    /// that a projection landing second conflicts with the schema-metadata
    /// edit its read schema would discard, where legacy discards it.
    #[rstest]
    #[case::append(Operation::Append { fragments: vec![fragment(0)] }, false, false)]
    #[case::delete(
        Operation::Delete {
            updated_fragments: vec![fragment(0)],
            deleted_fragment_ids: vec![1],
            predicate: "true".into(),
        },
        false,
        false
    )]
    #[case::update_rewrite_rows(
        update(vec![fragment(0)], vec![fragment(0)], vec![], None),
        false,
        false
    )]
    #[case::update_rewrite_columns(update(vec![fragment(0)], vec![], vec![0], None), false, false)]
    #[case::data_overlay(
        Operation::DataOverlay {
            groups: vec![DataOverlayGroup {
                fragment_id: 0,
                overlays: vec![overlay_with_field(0, 1)],
            }],
        },
        false,
        false
    )]
    #[case::data_replacement(
        Operation::DataReplacement { replacements: vec![DataReplacementGroup(0, field_file("replacement", 0))] },
        false,
        false
    )]
    #[case::create_index(
        Operation::CreateIndex {
            new_indices: vec![sample_index_metadata("idx")],
            removed_indices: vec![],
        },
        false,
        false
    )]
    #[case::rewrite(
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: vec![fragment(0)],
                new_fragments: vec![fragment(0)],
            }],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        },
        false,
        false
    )]
    // A reset that landed first leaves nothing of what the set read; one that
    // lands second wins.
    #[case::overwrite(
        Operation::Overwrite {
            fragments: vec![fragment(0)],
            schema: sample_manifest().schema,
            config_upsert_values: None,
            initial_bases: None,
        },
        true,
        false
    )]
    // A restore puts back the schema metadata the set edits.
    #[case::restore(Operation::Restore { version: 1 }, true, true)]
    #[case::reserve_fragments(Operation::ReserveFragments { num_fragments: 2 }, false, false)]
    #[case::update_config_on_the_same_key(update_config(Some("k"), None), true, true)]
    #[case::update_config_on_other_keys(update_config(Some("other"), Some("t")), false, false)]
    #[case::update_bases(
        Operation::UpdateBases {
            new_bases: vec![BasePath::new(1, "s3://bucket/base".into(), Some("base".into()), false)],
        },
        false,
        false
    )]
    #[case::clone(
        Operation::Clone {
            is_shallow: true,
            ref_name: None,
            ref_version: 1,
            ref_path: "memory://source".into(),
            branch_name: None,
        },
        false,
        false
    )]
    #[case::merge(
        Operation::Merge {
            fragments: vec![fragment(0), fragment(1)],
            schema: sample_manifest().schema,
            preserves_nullability: true,
        },
        false,
        true
    )]
    #[case::project(
        Operation::Project { schema: sample_manifest().schema, preserves_nullability: true },
        false,
        true
    )]
    #[case::update_mem_wal_state(
        Operation::UpdateMemWalState {
            compacted_sstables: vec![],
            dir_id: String::new(),
        },
        true,
        true
    )]
    #[case::unknown(Operation::Unknown {}, true, true)]
    fn test_append_and_config_against_each_named_operation(
        #[case] named: Operation,
        #[case] actions_after_named: bool,
        #[case] named_after_actions: bool,
    ) {
        let mut set = append_actions();
        set.push(config_action(Some("k"), None));
        let set = actions(set);

        assert_eq!(conflicts(&set, &named), actions_after_named);
        assert_eq!(conflicts(&named, &set), named_after_actions);
    }

    /// A uniqueness assertion is a precondition on its own commit, as legacy
    /// merge insert's inserted-key filter is: rows a named operation inserted
    /// before it fail it unless that operation says which keys it inserted,
    /// and nothing that lands after it is held to it.
    #[rstest]
    #[case::an_append(Operation::Append { fragments: vec![fragment(0)] }, true)]
    #[case::an_update_inserting_rows(update(vec![], vec![fragment(0)], vec![], None), true)]
    #[case::an_update_inserting_other_keys(
        update(vec![], vec![fragment(0)], vec![], Some(key_filter(&[2]))),
        false
    )]
    #[case::an_update_rewriting_the_key_column(
        update(vec![fragment(0)], vec![], vec![0], None),
        true
    )]
    #[case::a_delete(
        Operation::Delete {
            updated_fragments: vec![fragment(0)],
            deleted_fragment_ids: vec![],
            predicate: "true".into(),
        },
        false
    )]
    fn test_key_assertions_against_named_operations(
        #[case] named: Operation,
        #[case] assertion_after_named: bool,
    ) {
        let mut set = append_actions();
        set.push(key_assertion(&[1]));
        let set = actions(set);

        assert_eq!(conflicts(&set, &named), assertion_after_named);
        assert!(!conflicts(&named, &set));
    }

    /// A legacy merge insert's inserted-key filter is an assertion of its own,
    /// held to an action set's inserts the same way, and compared with an
    /// action set's assertion in either order.
    #[test]
    fn test_a_named_update_asserts_its_inserted_keys() {
        let merge_insert =
            |key| update(vec![], vec![fragment(0)], vec![], Some(key_filter(&[key])));
        let asserting = |key| {
            let mut set = append_actions();
            set.push(key_assertion(&[key]));
            actions(set)
        };

        assert!(conflicts(&merge_insert(1), &actions(append_actions())));
        assert!(!conflicts(&merge_insert(1), &asserting(2)));
        assert!(!conflicts(&asserting(2), &merge_insert(1)));
        assert!(conflicts(&merge_insert(1), &asserting(1)));
        assert!(conflicts(&asserting(1), &merge_insert(1)));
    }

    /// Two edits of the manifest's string maps get the same verdict whichever
    /// form each side is in.
    #[rstest]
    #[case::disjoint_keys(Some("a"), Some("b"), false)]
    #[case::the_same_key(Some("a"), Some("a"), true)]
    fn test_table_metadata_edits_agree_across_forms(
        #[case] ours: Option<&str>,
        #[case] theirs: Option<&str>,
        #[case] expected: bool,
    ) {
        let named = |key| update_config(None, key);
        let set = |key| actions(vec![config_action(None, key)]);

        assert_eq!(named(ours).modifies_same_metadata(&named(theirs)), expected);
        assert_eq!(conflicts(&named(ours), &set(theirs)), expected);
        assert_eq!(conflicts(&set(ours), &named(theirs)), expected);
        assert_eq!(conflicts(&set(ours), &set(theirs)), expected);
    }

    /// Two fragments of three rows over `a` (id 0) and `c` (id 1), each field in
    /// a data file of its own.
    fn two_column_manifest() -> Manifest {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("c", DataType::Int32, true),
        ]);
        let fragments = (0..2).map(two_column_fragment).collect();
        let mut manifest = Manifest::new(
            Schema::try_from(&schema).unwrap(),
            Arc::new(fragments),
            DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );
        manifest.version = 2;
        manifest
    }

    fn two_column_fragment(id: u64) -> Fragment {
        let mut fragment = Fragment::new(id);
        fragment.physical_rows = Some(3);
        fragment.files = vec![
            field_file(&format!("a-{id}"), 0),
            field_file(&format!("c-{id}"), 1),
        ];
        fragment
    }

    /// `file` as an [`AddDataFile`] carries it: its field list is the action's.
    fn without_fields(file: DataFile) -> DataFile {
        DataFile {
            fields: Default::default(),
            ..file
        }
    }

    fn field_file(name: &str, field: i32) -> DataFile {
        DataFile::new(
            format!("data/{name}.lance"),
            vec![field],
            vec![0],
            ConcreteFileVersion::V2_0,
            None,
            None,
        )
    }

    /// `operation`, read at `before` and committed on top of it.
    fn commit(before: &Manifest, operation: Operation) -> (Transaction, Manifest) {
        let transaction = Transaction::new(before.version, operation, None);
        let (after, _) = transaction
            .build_manifest(Some(before), vec![], "tx.txn", &default_build_config())
            .unwrap();
        (transaction, after)
    }

    /// Whether `ours`, read at `before`, conflicts with `theirs`, which committed
    /// `before` -> the manifest `theirs` builds.
    fn conflicts_with_committed(ours: &Operation, before: &Manifest, theirs: Operation) -> bool {
        let (transaction, after) = commit(before, theirs);
        let pending = Footprint::pending(&Transaction::new(1, ours.clone(), None), before);
        let committed = Footprint::committed(&transaction, before, &after);
        matches!(pending.compare(&committed), Verdict::Conflict)
    }

    fn project_without_c(manifest: &Manifest) -> Operation {
        Operation::Project {
            schema: manifest.schema.project(&["a"]).unwrap(),
            preserves_nullability: true,
        }
    }

    fn replace_c(fragment: u64, data_change: bool) -> Vec<Action> {
        vec![
            Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(fragment),
                field_ids: vec![FieldRef::Committed(1)],
                data_change,
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Committed(fragment),
                file: without_fields(field_file("c-new", 1)),
                field_ids: vec![FieldRef::Committed(1)],
                data_change,
                base: None,
            }),
        ]
    }

    fn rename_c() -> Action {
        Action::AlterField(AlterField {
            field: FieldRef::Committed(1),
            name: Some("c_renamed".into()),
            nullable: None,
        })
    }

    fn add_column_e() -> Vec<Action> {
        vec![
            Action::AddField(AddField {
                local: 0,
                parent: None,
                def: added_field("e"),
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Committed(0),
                file: without_fields(field_file("e-0", 2)),
                field_ids: vec![FieldRef::Local(0)],
                data_change: true,
                base: None,
            }),
        ]
    }

    fn append_a_and_c() -> Vec<Action> {
        vec![
            Action::AddFragment(AddFragment {
                id: Ref::Local(0),
                physical_rows: 3,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(0),
                file: without_fields(field_file("new", 0)),
                field_ids: vec![FieldRef::Committed(0), FieldRef::Committed(1)],
                data_change: true,
                base: None,
            }),
        ]
    }

    /// A projection or merge installs the schema it read. Whatever a concurrent
    /// set did to the schema is lost if it lands second, but data written into
    /// the fields it keeps is not.
    #[rstest]
    #[case::an_append(append_a_and_c(), false)]
    #[case::a_column_replacement(replace_c(0, true), false)]
    #[case::an_added_column(add_column_e(), true)]
    #[case::a_rename(vec![rename_c()], true)]
    #[case::a_schema_metadata_edit(vec![config_action(Some("k"), None)], true)]
    fn test_a_pending_project_against_a_committed_set(
        #[case] committed: Vec<Action>,
        #[case] expected: bool,
    ) {
        let before = two_column_manifest();
        let pending = Footprint::pending(
            &Transaction::new(2, project_without_c(&before), None),
            &before,
        );
        let committed = Footprint::committed(
            &Transaction::new(2, actions(committed), None),
            &before,
            &before,
        );

        assert_eq!(
            matches!(pending.compare(&committed), Verdict::Conflict),
            expected
        );
    }

    /// A merge restates every fragment it read and keeps only those, so it
    /// cannot land over rows added since; an index over data it leaves alone
    /// is another matter.
    #[test]
    fn test_a_pending_merge_against_committed_sets() {
        let before = two_column_manifest();
        let merge = Operation::Merge {
            fragments: before.fragments.as_ref().clone(),
            schema: before.schema.clone(),
            preserves_nullability: true,
        };
        let pending = Footprint::pending(&Transaction::new(2, merge, None), &before);
        let committed =
            |set| Footprint::committed(&Transaction::new(2, actions(set), None), &before, &before);

        assert!(matches!(
            pending.compare(&committed(append_a_and_c())),
            Verdict::Conflict
        ));
        let segment = vec![Action::AddIndexSegment(AddIndexSegment {
            uuid: Uuid::new_v4(),
            name: "idx".into(),
            fields: vec![FieldRef::Committed(0)],
            covering_fields: vec![],
            index_details: None,
            index_version: 1,
            covered_fragments: Some(vec![Ref::Committed(0)]),
            files: vec![],
            base: None,
            created_at: None,
            dataset_version: None,
            data_change: true,
        })];
        assert!(matches!(
            pending.compare(&committed(segment)),
            Verdict::Compatible
        ));
    }

    /// A legacy Update installs its post-image of the fragment, so it would put
    /// back a column a concurrent set tombstoned.
    #[test]
    fn test_a_pending_update_conflicts_with_a_committed_tombstone() {
        let before = two_column_manifest();
        let mut restated = before.fragments[0].clone();
        restated.deletion_file = Some(deletion_file());
        let update = update(vec![restated], vec![], vec![], None);
        let tombstone = actions(vec![Action::TombstoneFieldData(TombstoneFieldData {
            fragment: Ref::Committed(0),
            field_ids: vec![FieldRef::Committed(1)],
            data_change: true,
        })]);

        let pending = Footprint::pending(&Transaction::new(2, update, None), &before);
        let committed =
            Footprint::committed(&Transaction::new(2, tombstone, None), &before, &before);
        assert!(matches!(pending.compare(&committed), Verdict::Conflict));
    }

    fn deletion_file() -> DeletionFile {
        DeletionFile {
            read_version: 2,
            id: 1,
            file_type: DeletionFileType::Array,
            num_deleted_rows: Some(1),
            base_id: None,
        }
    }

    /// A committed named operation is held to what its commit changed, not to
    /// everything it restated.
    #[rstest]
    // Legacy lands an append over a projection, leaving the dropped column in
    // the new fragment's file unread. An action append is rebased to the same
    // file, binding only the fields that are left.
    #[case::an_append_over_a_project(append_a_and_c(), project_without_c(&two_column_manifest()), false)]
    #[case::a_rename_over_an_update_restating_the_fragment(
        vec![rename_c()],
        update(vec![{
            let mut restated = two_column_fragment(0);
            restated.deletion_file = Some(deletion_file());
            restated
        }], vec![], vec![], None),
        false
    )]
    #[case::a_column_replacement_over_a_delete(
        replace_c(0, true),
        Operation::Delete {
            updated_fragments: vec![{
                let mut restated = two_column_fragment(0);
                restated.deletion_file = Some(deletion_file());
                restated
            }],
            deleted_fragment_ids: vec![],
            predicate: "a = 0".into(),
        },
        false
    )]
    // The rows a merge insert moved out of the fragment take the values the
    // rewrite never saw with them.
    #[case::a_column_replacement_over_an_update_moving_rows(
        replace_c(0, true),
        update(vec![{
            let mut restated = two_column_fragment(0);
            restated.deletion_file = Some(deletion_file());
            restated
        }], vec![two_column_fragment(0)], vec![], None),
        true
    )]
    #[case::a_column_replacement_over_an_overlay(
        replace_c(0, true),
        Operation::DataOverlay {
            groups: vec![DataOverlayGroup {
                fragment_id: 0,
                overlays: vec![overlay_with_field(1, 0)],
            }],
        },
        false
    )]
    #[case::a_column_replacement_over_a_replacement(
        replace_c(0, true),
        Operation::DataReplacement {
            replacements: vec![DataReplacementGroup(0, field_file("c-other", 1))],
        },
        true
    )]
    #[case::a_rename_over_a_project_dropping_the_field(
        vec![rename_c()],
        project_without_c(&two_column_manifest()),
        true
    )]
    #[case::an_added_column_over_a_compaction_of_its_fragment(
        add_column_e(),
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: vec![two_column_fragment(0)],
                new_fragments: vec![two_column_fragment(0)],
            }],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        },
        true
    )]
    fn test_a_committed_operation_by_what_it_changed(
        #[case] pending: Vec<Action>,
        #[case] committed: Operation,
        #[case] expected: bool,
    ) {
        let before = two_column_manifest();
        assert_eq!(
            conflicts_with_committed(&actions(pending), &before, committed),
            expected
        );
    }

    /// The footprint read from a commit's manifests never claims less than the
    /// operation itself names: every coordinate it writes, fragment it removes,
    /// field it writes values for, row it adds, key it asserts and index it
    /// touches.
    ///
    /// Only the post-image regions of the operation's own footprint are left
    /// out. They describe what a pending operation would overwrite, which is
    /// more than a committed one changed.
    #[rstest]
    #[case::append(Operation::Append { fragments: vec![two_column_fragment(0)] })]
    #[case::delete(Operation::Delete {
        updated_fragments: vec![{
            let mut restated = two_column_fragment(0);
            restated.deletion_file = Some(deletion_file());
            restated
        }],
        deleted_fragment_ids: vec![1],
        predicate: "a = 0".into(),
    })]
    #[case::update_rewrite_rows(update(
        vec![{
            let mut restated = two_column_fragment(0);
            restated.deletion_file = Some(deletion_file());
            restated
        }],
        vec![two_column_fragment(0)],
        vec![],
        Some(key_filter(&[1])),
    ))]
    #[case::update_rewrite_columns(update(
        vec![{
            let mut restated = two_column_fragment(0);
            restated.files[1] = field_file("c-new", 1);
            restated
        }],
        vec![],
        vec![1],
        None,
    ))]
    #[case::data_overlay(Operation::DataOverlay {
        groups: vec![DataOverlayGroup {
            fragment_id: 0,
            overlays: vec![overlay_with_field(1, 0)],
        }],
    })]
    #[case::data_replacement(Operation::DataReplacement {
        replacements: vec![DataReplacementGroup(0, field_file("c-new", 1))],
    })]
    #[case::project(project_without_c(&two_column_manifest()))]
    #[case::merge(Operation::Merge {
        fragments: (0..2)
            .map(|id| {
                let mut fragment = two_column_fragment(id);
                fragment.files.truncate(1);
                fragment
            })
            .collect(),
        schema: two_column_manifest().schema.project(&["a"]).unwrap(),
        preserves_nullability: true,
    })]
    #[case::create_index(Operation::CreateIndex {
        new_indices: vec![sample_index_metadata("idx")],
        removed_indices: vec![],
    })]
    #[case::rewrite(Operation::Rewrite {
        groups: vec![RewriteGroup {
            old_fragments: vec![two_column_fragment(0)],
            new_fragments: vec![two_column_fragment(0)],
        }],
        rewritten_indices: vec![],
        frag_reuse_index: None,
    })]
    #[case::update_config(update_config(Some("k"), Some("t")))]
    #[case::update_field_metadata(Operation::UpdateConfig {
        config_updates: Some(update_map("c")),
        table_metadata_updates: None,
        schema_metadata_updates: None,
        field_metadata_updates: HashMap::from([(1, update_map("f"))]),
    })]
    #[case::update_bases(Operation::UpdateBases {
        new_bases: vec![BasePath::new(1, "s3://bucket/base".into(), Some("base".into()), false)],
    })]
    #[case::reserve_fragments(Operation::ReserveFragments { num_fragments: 2 })]
    fn test_a_committed_footprint_covers_the_operation(#[case] operation: Operation) {
        let before = two_column_manifest();
        let (transaction, after) = commit(&before, operation);

        let named = Footprint::pending(&transaction, &before);
        let changed = Footprint::committed(&transaction, &before, &after);

        for (coordinate, mode) in named
            .claims
            .iter()
            .filter(|(_, mode)| **mode >= Mode::WritesPart)
        {
            let claimed = changed
                .claims
                .get(coordinate)
                .is_some_and(|changed| changed >= mode)
                || changed
                    .regions
                    .iter()
                    .any(|(region, changed)| region.contains(coordinate) && changed >= mode);
            assert!(
                claimed,
                "{coordinate:?} is named {mode:?} but not claimed by the diff {changed:#?}"
            );
        }
        assert!(named.values_written.is_subset(&changed.values_written));
        assert!(
            named
                .required_fragments
                .is_subset(&changed.required_fragments)
        );
        assert!(
            named
                .remaps_indices_over
                .is_subset(&changed.remaps_indices_over)
        );
        assert!(!named.rows_added || changed.rows_added);
        assert!(!named.inserts_rows || changed.inserts_rows);
        assert!(!named.resets_table || changed.resets_table);
        assert!(!named.opaque || changed.opaque);
        assert_eq!(named.key_assertions, changed.key_assertions);
        assert_eq!(named.index_claims, changed.index_claims);
    }

    /// A compaction carries the indices over the rows it moves through a remap
    /// computed at its read version, so a segment built over those rows since
    /// is left describing rows that moved.
    #[test]
    fn test_a_rewrite_conflicts_with_an_index_segment_over_the_rows_it_moves() {
        let rewrite = |old: u64| Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: vec![fragment(old)],
                new_fragments: vec![fragment(0)],
            }],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        };
        let segment = actions(vec![Action::AddIndexSegment(AddIndexSegment {
            uuid: Uuid::new_v4(),
            name: "idx".into(),
            fields: vec![FieldRef::Committed(0)],
            covering_fields: vec![],
            index_details: None,
            index_version: 1,
            covered_fragments: Some(vec![Ref::Committed(0)]),
            files: vec![],
            base: None,
            created_at: None,
            dataset_version: None,
            data_change: true,
        })]);

        assert!(conflicts(&rewrite(0), &segment));
        assert!(!conflicts(&rewrite(1), &segment));
    }
}
