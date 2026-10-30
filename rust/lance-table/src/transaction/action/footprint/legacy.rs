// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Footprints of the named [`Operation`]s.
//!
//! A named operation has no actions to ask, so its footprint is read straight
//! off the operation. The mapping is coarse on purpose. Several operations
//! carry post-images -- a fragment restated whole, as `Delete` and `Update` do
//! -- and applying one replaces everything in the fragment it names, not just
//! the part the operation meant to change. So a fragment an operation restates
//! is claimed whole, and an operation whose post-image is the whole schema is
//! not described at all and fails closed.
//!
//! Lowering an operation to actions instead would under-claim for exactly that
//! reason: the actions describe the intended delta, the apply installs the
//! post-image.

use super::{ConfigMap, Coordinate, FieldRef, Footprint, Ref};
use crate::format::{BasePath, Fragment, IndexMetadata};
use crate::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use crate::system_index::mem_wal::MEM_WAL_INDEX_NAME;
use crate::transaction::Operation;
use crate::transaction::action::{AddBase, ConfigUpdate, FieldMetadataUpdate};
use lance_core::datatypes::Schema;

impl Footprint {
    /// The footprint of `operation`, which read `read_schema`. An action set's
    /// is its actions'.
    pub(super) fn of_operation(operation: &Operation, read_schema: &Schema) -> Self {
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
                    footprint.write_field_data(
                        Ref::Committed(fragment.id),
                        fields_modified
                            .iter()
                            .map(|field| FieldRef::Committed(*field as i32)),
                    );
                }
                footprint.append(new_fragments);
                if let Some(filter) = inserted_rows_filter {
                    footprint.assert_unique_keys(
                        field_refs(&filter.field_ids).collect(),
                        filter.clone(),
                    );
                }
                if !compacted_sstables.is_empty() {
                    footprint.rewrite_index(MEM_WAL_INDEX_NAME.to_string());
                }
            }
            Operation::DataReplacement { replacements } => {
                for replacement in replacements {
                    footprint.replace_fragment(replacement.0);
                    footprint.write_field_data(
                        Ref::Committed(replacement.0),
                        field_refs(&replacement.1.fields),
                    );
                }
            }
            Operation::DataOverlay { groups } => {
                for group in groups {
                    footprint.replace_fragment(group.fragment_id);
                    for overlay in &group.overlays {
                        footprint.write_partial_field_data(
                            Ref::Committed(group.fragment_id),
                            field_refs(&overlay.data_file.fields),
                        );
                    }
                }
            }
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            } => {
                for index in new_indices {
                    footprint.create_index(index);
                }
                for index in removed_indices {
                    footprint.write(Coordinate::IndexSegment(index.uuid));
                    footprint.rewrite_index(index.name.clone());
                }
            }
            Operation::Rewrite {
                groups,
                rewritten_indices,
                frag_reuse_index,
            } => {
                for group in groups {
                    for fragment in &group.old_fragments {
                        footprint.remove_fragment(fragment.id);
                        footprint.remaps_indices_over.insert(fragment.id);
                    }
                    // The rows are moved, not written, so the values are not
                    // new; but they are still laid out in the types they were
                    // read in.
                    for file in group.new_fragments.iter().flat_map(|f| &f.files) {
                        for field in field_refs(&file.fields) {
                            footprint.require_field_definition(field);
                        }
                    }
                }
                for index in rewritten_indices {
                    footprint.write(Coordinate::IndexSegment(index.old_id));
                }
                if frag_reuse_index.is_some() {
                    footprint.rewrite_index(FRAG_REUSE_INDEX_NAME.to_string());
                }
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
                for field in read_schema.fields_pre_order() {
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
            // A merge or projection restates the whole schema, and the MemWAL
            // state is checked against the data it compacted; neither is
            // described by coordinates yet. An unknown operation may touch
            // anything.
            Operation::Merge { .. }
            | Operation::Project { .. }
            | Operation::UpdateMemWalState { .. }
            | Operation::Unknown { .. } => footprint.opaque = true,
        }
        footprint
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
            self.write_field_data(Ref::Local(0), field_refs(&file.fields));
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
            self.require_field_definition(field);
        }
        for fragment in index.fragment_bitmap.iter().flatten() {
            self.require_field_data_unless_removed(u64::from(fragment), fields.iter().copied());
        }
    }

    fn add_base(&mut self, base: &BasePath) {
        AddBase {
            local: 0,
            base: base.clone(),
        }
        .footprint(self);
    }
}

/// The committed fields among `fields`. A negative id is a tombstone, which
/// names no field.
fn field_refs(fields: &[i32]) -> impl Iterator<Item = FieldRef> + '_ {
    fields
        .iter()
        .filter(|field| **field >= 0)
        .map(|field| FieldRef::Committed(*field))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::format::key_existence::{FilterType, KeyExistenceFilter};
    use crate::transaction::action::{
        Action, AddDataFile, AddFragment, AddIndexSegment, AssertUniqueKeys, CompositeOperation,
        UserStep, Verdict,
    };
    use crate::transaction::test_support::{
        overlay_with_field, sample_index_metadata, sample_manifest,
    };
    use crate::transaction::{
        DataOverlayGroup, DataReplacementGroup, RewriteGroup, Transaction, UpdateMap,
        UpdateMapEntry, UpdateMode,
    };
    use lance_file::version::ConcreteFileVersion;
    use rstest::rstest;
    use std::collections::HashMap;
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
                file: data_file(),
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
        let manifest = sample_manifest();
        let pending = Footprint::pending(&Transaction::new(1, ours.clone(), None), &manifest);
        let committed = Footprint::committed(
            &Transaction::new(1, theirs.clone(), None),
            &manifest,
            &manifest,
        );
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
    /// against the operations that restate the whole schema, which no
    /// coordinate describes yet.
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
        Operation::DataReplacement { replacements: vec![DataReplacementGroup(0, data_file())] },
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
            fragments: vec![],
            schema: sample_manifest().schema,
            preserves_nullability: true,
        },
        true,
        true
    )]
    #[case::project(
        Operation::Project { schema: sample_manifest().schema, preserves_nullability: true },
        true,
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
