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

use super::{Action, Ref, UserOperation};
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
}

impl Coordinate {
    /// The fragment this coordinate lives in, if it is fragment-scoped.
    fn fragment(&self) -> Option<u64> {
        match self {
            Self::FragmentExistence(id) | Self::FragmentDeletions(id) => Some(*id),
            Self::FieldData { fragment, .. } => Some(*fragment),
            Self::FieldDefinition(_) | Self::BaseName(_) | Self::BaseLocation(_) => None,
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
}

impl Footprint {
    pub fn conflicts_with(&self, other: &Self) -> bool {
        if !self.writes.is_disjoint(&other.writes) {
            return true;
        }
        self.removes_a_fragment_touched_by(other) || other.removes_a_fragment_touched_by(self)
    }

    fn removes_a_fragment_touched_by(&self, other: &Self) -> bool {
        self.removed_fragments.iter().any(|removed| {
            other
                .writes
                .iter()
                .any(|coordinate| coordinate.fragment() == Some(*removed))
        })
    }

    fn add(&mut self, coordinate: Coordinate) {
        self.writes.insert(coordinate);
    }

    fn add_field_data(&mut self, fragment: Ref, fields: impl IntoIterator<Item = i32>) {
        let Some(fragment) = fragment.committed() else {
            return;
        };
        for field in fields {
            self.add(Coordinate::FieldData { fragment, field });
        }
    }
}

impl From<&UserOperation> for Footprint {
    fn from(user_operation: &UserOperation) -> Self {
        let mut footprint = Self::default();
        for action in user_operation.iter_actions() {
            match action {
                // Minting actions name nothing that exists in the read version.
                Action::AddFragment(_) | Action::AddField(_) => {}
                Action::AddBase(action) => {
                    footprint.add(Coordinate::BaseName(action.base.name.clone()));
                    footprint.add(Coordinate::BaseLocation(action.base.path.clone()));
                }
                Action::AddDataFile(action) => footprint.add_field_data(
                    action.fragment,
                    action.field_ids.iter().filter_map(|field| {
                        field.committed().and_then(|id| i32::try_from(id).ok())
                    }),
                ),
                Action::TombstoneFieldData(action) => {
                    footprint.add_field_data(action.fragment, action.field_ids.iter().copied())
                }
                Action::RemoveFragment(action) => {
                    if let Some(id) = action.fragment.committed() {
                        footprint.add(Coordinate::FragmentExistence(id));
                        footprint.removed_fragments.insert(id);
                    }
                }
                Action::SetDeletionFile(action) => {
                    footprint.add(Coordinate::FragmentDeletions(action.fragment))
                }
                Action::AlterField(action) => {
                    footprint.add(Coordinate::FieldDefinition(action.field))
                }
            }
        }
        footprint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{BasePath, DataFile};
    use crate::transaction::action::{
        AddBase, AddDataFile, AddField, AddFragment, AlterField, RemoveFragment, SetDeletionFile,
        TombstoneFieldData, UserAction,
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
