// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The set of coordinates an action set writes.
//!
//! Two concurrent action sets can both commit when neither writes anything the
//! other writes. This is a structural test over coordinates rather than a
//! matrix over operation pairs, so it stays a single rule as the vocabulary
//! grows -- adding an action means saying which coordinates it writes, not
//! extending an N-by-N table.
//!
//! Footprints are derived from the actions at conflict time and never
//! serialized. A writer cannot pin down what a reader considers a conflict, and
//! the rule can be tightened in a later release without a format change.
//!
//! Which coordinates an action writes is decided by that action, in its own
//! module. This module holds the coordinate space and the comparison.

use super::{Ref, UserOperation};
use crate::transaction::UpdateMap;
use std::collections::HashSet;

/// One thing an action set writes.
///
/// Only committed coordinates appear. A minted fragment, field, or base has no
/// id in the read version, so no concurrent writer can be naming the same
/// thing; relocation re-resolves it against whatever version wins.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Coordinate {
    /// Whether a committed fragment is still part of the dataset.
    FragmentExistence(u64),
    /// A committed fragment's deletion file.
    FragmentDeletions(u64),
    /// The data backing one field within one committed fragment.
    FieldData { fragment: u64, field: i32 },
    /// A field's definition in the schema.
    FieldDefinition(i32),
    /// A base path's name, which the manifest requires to be unique.
    BaseName(Option<String>),
    /// A base path's location, which the manifest requires to be unique.
    BaseLocation(String),
    /// One key in one of the manifest's string maps.
    ConfigEntry { map: ConfigMap, key: String },
}

/// One of the string maps a manifest carries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConfigMap {
    /// Dataset config.
    Config,
    /// Table metadata.
    TableMetadata,
    /// Schema-level metadata.
    SchemaMetadata,
    /// One field's metadata, by field id.
    Field(i32),
}

impl Coordinate {
    /// The fragment this coordinate lives in, if it is fragment-scoped.
    fn fragment(&self) -> Option<u64> {
        match self {
            Self::FragmentExistence(id) | Self::FragmentDeletions(id) => Some(*id),
            Self::FieldData { fragment, .. } => Some(*fragment),
            Self::FieldDefinition(_)
            | Self::BaseName(_)
            | Self::BaseLocation(_)
            | Self::ConfigEntry { .. } => None,
        }
    }

    /// The field this coordinate belongs to, if it is field-scoped.
    fn field(&self) -> Option<i32> {
        match self {
            Self::FieldData { field, .. } => Some(*field),
            Self::FieldDefinition(id) => Some(*id),
            // A field's metadata goes with the field, so dropping the field
            // writes over a concurrent update to its metadata.
            Self::ConfigEntry {
                map: ConfigMap::Field(id),
                ..
            } => Some(*id),
            Self::FragmentExistence(_)
            | Self::FragmentDeletions(_)
            | Self::BaseName(_)
            | Self::BaseLocation(_)
            | Self::ConfigEntry { .. } => None,
        }
    }

    /// The string map this coordinate is a key in, if it is one.
    fn config_map(&self) -> Option<&ConfigMap> {
        match self {
            Self::ConfigEntry { map, .. } => Some(map),
            _ => None,
        }
    }
}

/// Everything an action set writes, in one set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Footprint {
    writes: HashSet<Coordinate>,
    /// Fragments this set removes outright. Removing a fragment writes every
    /// coordinate inside it, which cannot be enumerated, so it is tracked
    /// separately and matched against the other set by fragment id.
    removed_fragments: HashSet<u64>,
    /// Fields this set drops from the schema. Like a fragment removal, this
    /// writes every coordinate belonging to the field -- its definition and its
    /// data in every fragment -- so it is matched by field id.
    ///
    /// Only the named field, not its descendants: a footprint has no schema to
    /// expand a struct with. A concurrent write to a child of a dropped struct
    /// is therefore not caught here and fails when it is applied against the
    /// version where the child no longer exists.
    removed_fields: HashSet<i32>,
    /// String maps this set replaces outright rather than merging into. Like a
    /// fragment removal, this writes every key in the map, including keys it
    /// does not name, so it is matched by map rather than by key.
    replaced_maps: HashSet<ConfigMap>,
    /// Whether this set rewrites the table wholesale. Such a set writes every
    /// coordinate there is, including ones a concurrent set would only mint, so
    /// it is tracked as a flag rather than enumerated.
    exclusive: bool,
}

impl Footprint {
    pub fn conflicts_with(&self, other: &Self) -> bool {
        // A wholesale rewrite leaves nothing for a concurrent set to land on --
        // not even an append, whose rows the reset would discard or resurrect
        // depending on which commit won.
        if self.exclusive || other.exclusive {
            return true;
        }
        if !self.writes.is_disjoint(&other.writes) {
            return true;
        }
        // Two sets replacing the same map collide even when neither names a
        // key, since clearing a map is a replacement with no entries.
        if !self.replaced_maps.is_disjoint(&other.replaced_maps) {
            return true;
        }
        self.removes_something_touched_by(other) || other.removes_something_touched_by(self)
    }

    /// Whether this set wipes out something -- a fragment, a field, a whole
    /// string map -- that `other` also writes to.
    fn removes_something_touched_by(&self, other: &Self) -> bool {
        other.writes.iter().any(|coordinate| {
            coordinate
                .fragment()
                .is_some_and(|id| self.removed_fragments.contains(&id))
                || coordinate
                    .field()
                    .is_some_and(|id| self.removed_fields.contains(&id))
                || coordinate
                    .config_map()
                    .is_some_and(|map| self.replaced_maps.contains(map))
        })
    }

    pub(super) fn add(&mut self, coordinate: Coordinate) {
        self.writes.insert(coordinate);
    }

    /// The data of each field within `fragment`. A fragment minted in this same
    /// operation records nothing: no concurrent writer can be naming it.
    pub(super) fn add_field_data(&mut self, fragment: Ref, fields: impl IntoIterator<Item = i32>) {
        let Some(fragment) = fragment.committed() else {
            return;
        };
        for field in fields {
            self.add(Coordinate::FieldData { fragment, field });
        }
    }

    pub(super) fn remove_fragment(&mut self, fragment: u64) {
        self.add(Coordinate::FragmentExistence(fragment));
        self.removed_fragments.insert(fragment);
    }

    /// Record an edit to one of the manifest's string maps: the keys it names,
    /// or the whole map when it replaces rather than merges.
    pub(super) fn add_map_update(&mut self, map: ConfigMap, update: &UpdateMap) {
        if update.replace {
            self.replaced_maps.insert(map);
            return;
        }
        for entry in &update.update_entries {
            self.add(Coordinate::ConfigEntry {
                map: map.clone(),
                key: entry.key.clone(),
            });
        }
    }

    /// Mark this set as rewriting the whole table, conflicting with any
    /// concurrent set whatsoever.
    pub(super) fn take_exclusive(&mut self) {
        self.exclusive = true;
    }

    pub(super) fn remove_field(&mut self, field: i32) {
        self.add(Coordinate::FieldDefinition(field));
        self.removed_fields.insert(field);
    }
}

impl From<&UserOperation> for Footprint {
    fn from(user_operation: &UserOperation) -> Self {
        let mut footprint = Self::default();
        for action in user_operation.iter_actions() {
            action.footprint(&mut footprint);
        }
        footprint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{BasePath, DataFile};
    use crate::transaction::action::{
        Action, AddBase, AddDataFile, AddField, AddFragment, AlterField, DropField, RemoveFragment,
        SetDeletionFile, TombstoneFieldData, UserAction,
    };
    use arrow_schema::{DataType, Field as ArrowField};
    use lance_core::datatypes::Field;
    use rstest::rstest;

    fn footprint(actions: Vec<Action>) -> Footprint {
        Footprint::from(&UserOperation::new(
            "test",
            vec![UserAction::new("step", actions)],
        ))
    }

    fn add_fragment(local: u32) -> Action {
        Action::AddFragment(AddFragment {
            local,
            physical_rows: 10,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        })
    }

    fn add_data_file(fragment: Ref, fields: &[i32]) -> Action {
        Action::AddDataFile(AddDataFile {
            fragment,
            file: DataFile::new_unstarted("data/x.lance", 2, 0),
            field_ids: fields
                .iter()
                .map(|field| Ref::Committed(*field as u64))
                .collect(),
            data_change: true,
        })
    }

    fn tombstone(fragment: u64, fields: &[i32]) -> Action {
        Action::TombstoneFieldData(TombstoneFieldData {
            fragment: Ref::Committed(fragment),
            field_ids: fields.to_vec(),
            data_change: true,
        })
    }

    fn remove_fragment(fragment: u64) -> Action {
        Action::RemoveFragment(RemoveFragment {
            fragment: Ref::Committed(fragment),
            data_change: true,
        })
    }

    fn set_deletion_file(fragment: u64) -> Action {
        Action::SetDeletionFile(SetDeletionFile {
            fragment,
            deletion_file: None,
            data_change: true,
        })
    }

    fn add_base(local: u32, name: &str, path: &str) -> Action {
        Action::AddBase(AddBase {
            local,
            base: BasePath::new(0, path.into(), Some(name.into()), false),
        })
    }

    #[test]
    fn test_minting_actions_write_nothing() {
        let minting = footprint(vec![
            add_fragment(0),
            Action::AddField(AddField {
                local: 1,
                parent: None,
                def: Field::try_from(ArrowField::new("new", DataType::Int32, true)).unwrap(),
            }),
            add_data_file(Ref::Local(0), &[0]),
        ]);

        // Two writers appending at the same time never collide.
        assert!(!minting.conflicts_with(&minting.clone()));
    }

    #[rstest]
    #[case::same_field_in_same_fragment(
        vec![tombstone(0, &[1])],
        vec![add_data_file(Ref::Committed(0), &[1])],
        true,
    )]
    #[case::different_fields_in_same_fragment(
        vec![tombstone(0, &[1])],
        vec![add_data_file(Ref::Committed(0), &[2])],
        false,
    )]
    #[case::same_field_in_different_fragments(
        vec![tombstone(0, &[1])],
        vec![tombstone(1, &[1])],
        false,
    )]
    #[case::deletions_do_not_collide_with_field_data(
        vec![set_deletion_file(0)],
        vec![tombstone(0, &[1])],
        false,
    )]
    #[case::concurrent_deletes_of_one_fragment(
        vec![set_deletion_file(0)],
        vec![set_deletion_file(0)],
        true,
    )]
    #[case::removal_swallows_the_whole_fragment(
        vec![remove_fragment(0)],
        vec![tombstone(0, &[1])],
        true,
    )]
    #[case::removal_leaves_other_fragments_alone(
        vec![remove_fragment(0)],
        vec![tombstone(1, &[1])],
        false,
    )]
    #[case::same_field_definition(
        vec![Action::AlterField(AlterField { field: 1, name: Some("a".into()), ..Default::default() })],
        vec![Action::AlterField(AlterField { field: 1, nullable: Some(true), ..Default::default() })],
        true,
    )]
    #[case::different_field_definitions(
        vec![Action::AlterField(AlterField { field: 1, ..Default::default() })],
        vec![Action::AlterField(AlterField { field: 2, ..Default::default() })],
        false,
    )]
    #[case::dropping_a_field_collides_with_altering_it(
        vec![Action::DropField(DropField { field: 1 })],
        vec![Action::AlterField(AlterField { field: 1, nullable: Some(true), ..Default::default() })],
        true,
    )]
    #[case::dropping_a_field_collides_with_rewriting_its_data(
        vec![Action::DropField(DropField { field: 1 })],
        vec![tombstone(0, &[1])],
        true,
    )]
    #[case::dropping_a_field_leaves_other_fields_alone(
        vec![Action::DropField(DropField { field: 1 })],
        vec![tombstone(0, &[2])],
        false,
    )]
    #[case::dropping_a_field_leaves_deletions_alone(
        vec![Action::DropField(DropField { field: 1 })],
        vec![set_deletion_file(0)],
        false,
    )]
    #[case::bases_with_the_same_name(
        vec![add_base(0, "a", "s3://bucket/one")],
        vec![add_base(0, "a", "s3://bucket/two")],
        true,
    )]
    #[case::bases_with_the_same_location(
        vec![add_base(0, "a", "s3://bucket/one")],
        vec![add_base(0, "b", "s3://bucket/one")],
        true,
    )]
    #[case::unrelated_bases(
        vec![add_base(0, "a", "s3://bucket/one")],
        vec![add_base(0, "b", "s3://bucket/two")],
        false,
    )]
    fn test_conflicts(
        #[case] ours: Vec<Action>,
        #[case] theirs: Vec<Action>,
        #[case] expected: bool,
    ) {
        let ours = footprint(ours);
        let theirs = footprint(theirs);
        assert_eq!(ours.conflicts_with(&theirs), expected);
        // The relation has to hold whichever side is asking.
        assert_eq!(theirs.conflicts_with(&ours), expected);
    }
}
