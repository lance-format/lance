// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Assert that the cells an operation read are unchanged.

use super::apply::ApplyState;
use super::row_selection::row_selections_from_proto;
use super::{FieldRef, Footprint, Ref, RowSelection};
use crate::format::pb_actions;
use lance_core::datatypes::Field;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// A precondition rather than a delta: the declared cells -- every field in
/// [`Self::fields`] at every row in [`Self::rows`] -- hold the values they held
/// at the version the operation read.
///
/// Conflict detection otherwise compares only what two operations write, which
/// is snapshot isolation: `UPDATE t SET a = b` and `UPDATE t SET b = a` from one
/// snapshot write disjoint columns, so both commit and the columns end up
/// swapped -- a result neither serial order produces, since either leaves the
/// two columns equal. A writer whose values depend on what it read declares
/// the read here, and a concurrent commit that changed those cells makes it
/// conflict.
///
/// "Changed" follows the cell preconditions proposed for the legacy
/// transaction (#9121): a field dropped or redefined, a selected fragment
/// removed, a data file carrying a declared field replaced, or an overlay
/// written over it. A change to the deletion vector does not count: a deleted
/// row's value is invisible. The check may over-reject but never under-reject;
/// for now it works at fragment granularity, so a row selection narrower than
/// the whole fragment is honored as the whole fragment.
///
/// Only top-level leaf columns, and packed structs, can be declared. A
/// footprint has no schema to relate a field to its descendants, so a write to
/// a struct child or a list item would go unnoticed by an assertion over its
/// parent, and the reverse.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AssertCellsUnchanged {
    /// The fields read. Committed only: a field minted in the same operation
    /// has no value at the version read. Empty declares no cells.
    pub fields: Vec<FieldRef>,
    /// The rows read, one selection per committed fragment. Empty declares no
    /// cells.
    pub rows: Vec<RowSelection>,
}

impl AssertCellsUnchanged {
    /// Nothing to apply -- the assertion is checked when two operations are
    /// compared. What it names is still checked against the version it lands
    /// on, so an assertion over something that is not there fails rather than
    /// silently guarding nothing.
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        for field in &self.fields {
            if let FieldRef::Local(token) = field {
                return Err(Error::invalid_input(format!(
                    "AssertCellsUnchanged names a field minted in the same operation (local \
                     token {token}), which has no value at the version read"
                )));
            }
            let field_id = state.resolve_schema_field(*field, "AssertCellsUnchanged")?;
            let field = state.schema().field_by_id(field_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "AssertCellsUnchanged names field {field_id}, which is not in the schema"
                ))
            })?;
            check_assertable(field, "AssertCellsUnchanged")?;
        }
        for selection in &self.rows {
            let fragment_id =
                state.resolve_fragment(Ref::Committed(u64::from(selection.fragment_id)))?;
            state.fragment_mut(fragment_id, "AssertCellsUnchanged")?;
        }
        Ok(())
    }

    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    /// A precondition on what a concurrent set wrote, at fragment granularity:
    /// the declared fields in the selected fragments. A minted field names
    /// nothing a concurrent set could write, and apply rejects it anyway.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.assert_cells_unchanged(
            self.fields
                .iter()
                .filter_map(|field| field.committed())
                .collect(),
            self.rows
                .iter()
                .map(|selection| u64::from(selection.fragment_id))
                .collect(),
        );
    }
}

/// Footprints match the fields an assertion names to the field ids a write
/// records, exactly and without a schema. A write to a nested value is
/// recorded under its descendants' ids -- a V2.1 list overlay under its item's,
/// a struct rewrite under its children's -- so an assertion over a field with
/// children, or over a child, would not see it. Only a top-level leaf column,
/// or a packed struct, whose values live under its own id, can be asserted.
pub(super) fn check_assertable(field: &Field, action: &str) -> Result<()> {
    let has_own_values = field.children.is_empty() || field.is_packed_struct();
    if field.parent_id < 0 && has_own_values {
        return Ok(());
    }
    Err(Error::invalid_input(format!(
        "{action} names field {} ('{}'), which is nested or holds nested fields; only \
         top-level leaf columns and packed structs can be asserted, because conflict \
         detection matches field ids exactly and a write to a nested value is recorded \
         under its descendants' ids",
        field.id, field.name
    )))
}

impl From<&AssertCellsUnchanged> for pb_actions::AssertCellsUnchanged {
    fn from(value: &AssertCellsUnchanged) -> Self {
        Self {
            fields: value.fields.iter().map(|field| (*field).into()).collect(),
            rows: value.rows.iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<pb_actions::AssertCellsUnchanged> for AssertCellsUnchanged {
    type Error = Error;

    fn try_from(message: pb_actions::AssertCellsUnchanged) -> Result<Self> {
        Ok(Self {
            fields: message
                .fields
                .into_iter()
                .map(FieldRef::try_from)
                .collect::<Result<Vec<_>>>()?,
            rows: row_selections_from_proto(message.rows, "AssertCellsUnchanged")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::format::overlay::{DataOverlayFile, OverlayCoverage};
    use crate::transaction::action::SelectedRows;
    use crate::transaction::action::test_support::{
        added_field, apply, backed_manifest, footprint,
    };
    use crate::transaction::action::{
        Action, AddDataFile, AddField, AddOverlays, AlterField, DropField, Ref, RemoveFragment,
        SetDeletionFile, TombstoneFieldData,
    };
    use arrow_schema::{DataType, Field as ArrowField, Fields};
    use lance_file::version::ConcreteFileVersion;
    use roaring::RoaringBitmap;
    use rstest::rstest;
    use std::sync::Arc;

    fn read(fields: &[i32], fragments: &[u32]) -> Action {
        Action::AssertCellsUnchanged(AssertCellsUnchanged {
            fields: fields.iter().copied().map(FieldRef::Committed).collect(),
            rows: fragments
                .iter()
                .map(|fragment_id| RowSelection {
                    fragment_id: *fragment_id,
                    rows: SelectedRows::Full,
                })
                .collect(),
        })
    }

    fn rewrite(fragment: u64, field: i32) -> Vec<Action> {
        vec![
            Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(fragment),
                field_ids: vec![FieldRef::Committed(field)],
                data_change: true,
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Committed(fragment),
                file: DataFile::new_unstarted("data/new.lance", ConcreteFileVersion::V2_0),
                field_ids: vec![FieldRef::Committed(field)],
                data_change: true,
                base: None,
            }),
        ]
    }

    /// `UPDATE t SET a = b`: read b, write a.
    fn set_from(written: i32, read_field: i32) -> Vec<Action> {
        let mut actions = rewrite(0, written);
        actions.push(read(&[read_field], &[0]));
        actions
    }

    fn overlay(fragment: u64, field: i32) -> Action {
        let mut data_file =
            DataFile::new_unstarted("data/overlay.lance", ConcreteFileVersion::V2_0);
        data_file.fields = vec![field].into();
        data_file.column_indices = vec![0].into();
        Action::AddOverlays(AddOverlays {
            fragment: Ref::Committed(fragment),
            overlays: vec![DataOverlayFile {
                data_file,
                coverage: OverlayCoverage::Shared(Arc::new(RoaringBitmap::from_iter([0u32]))),
                committed_version: 0,
            }],
            data_change: true,
        })
    }

    /// `a = b` and `b = a` from one snapshot write disjoint columns, so
    /// without the assertions both commit and swap the columns, which neither
    /// serial order does. Each read the column the other wrote, so whichever
    /// lands second conflicts.
    #[test]
    fn test_swapping_two_columns_concurrently_conflicts() {
        let a_from_b = footprint(set_from(0, 1));
        let b_from_a = footprint(set_from(1, 0));

        assert!(b_from_a.conflicts_with(&a_from_b));
        assert!(a_from_b.conflicts_with(&b_from_a));

        // Without the reads, the same writes commute.
        assert!(!footprint(rewrite(0, 1)).conflicts_with(&footprint(rewrite(0, 0))));
    }

    /// What a committed set may have done to the cells the asserting set read.
    /// The assertion reads field 1 in fragment 0.
    #[rstest]
    #[case::rewrote_the_field_there(rewrite(0, 1), true)]
    #[case::overlaid_the_field_there(vec![overlay(0, 1)], true)]
    #[case::removed_the_fragment(
        vec![Action::RemoveFragment(RemoveFragment { fragment: Ref::Committed(0), data_change: true })],
        true
    )]
    #[case::dropped_the_field(vec![Action::DropField(DropField { field: FieldRef::Committed(1) })], true)]
    #[case::redefined_the_field(
        vec![Action::AlterField(AlterField { field: FieldRef::Committed(1), name: None, nullable: Some(true) })],
        true
    )]
    #[case::rewrote_the_field_elsewhere(rewrite(1, 1), false)]
    #[case::rewrote_another_field_there(rewrite(0, 2), false)]
    #[case::deleted_rows_there(
        vec![Action::SetDeletionFile(SetDeletionFile {
            fragment: 0,
            deletion_file: None,
            data_change: true,
            base: None,
        })],
        false
    )]
    fn test_an_assertion_conflicts_with_a_change_to_its_cells(
        #[case] committed: Vec<Action>,
        #[case] expected: bool,
    ) {
        let asserting = footprint(vec![read(&[1], &[0])]);

        assert_eq!(asserting.conflicts_with(&footprint(committed)), expected);
    }

    /// The assertion is a precondition on the set that carries it: a set that
    /// landed first was valid when it landed, whatever it read.
    #[test]
    fn test_an_assertion_constrains_only_the_set_that_carries_it() {
        let asserting = footprint(vec![read(&[1], &[0])]);

        assert!(!footprint(rewrite(0, 1)).conflicts_with(&asserting));
    }

    fn nested_manifest() -> crate::format::Manifest {
        let mut manifest = backed_manifest();
        manifest.schema =
            lance_core::datatypes::Schema::try_from(&arrow_schema::Schema::new(vec![
                ArrowField::new("id", DataType::Int32, false),
                ArrowField::new(
                    "s",
                    DataType::Struct(Fields::from(vec![ArrowField::new(
                        "x",
                        DataType::Int32,
                        true,
                    )])),
                    true,
                ),
                ArrowField::new(
                    "tags",
                    DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true))),
                    true,
                ),
            ]))
            .unwrap();
        manifest
    }

    /// What an assertion names is checked against the version it lands on.
    /// `nested_manifest` holds fragment 0, a top-level `id` (0), an unpacked
    /// struct `s` (1) with child `x` (2), and a list `tags` (3) with item (4).
    #[rstest]
    #[case::a_top_level_column(read(&[0], &[0]), None)]
    #[case::no_cells(read(&[], &[]), None)]
    #[case::a_field_beyond_the_read_version(read(&[9], &[0]), Some("names field 9 by a Committed ref"))]
    #[case::a_fragment_beyond_the_read_version(read(&[0], &[7]), Some("fragment 7 by a Committed ref"))]
    #[case::an_unpacked_struct(read(&[1], &[0]), Some("names field 1 ('s'), which is nested or holds nested fields"))]
    #[case::a_struct_child(read(&[2], &[0]), Some("names field 2 ('x'), which is nested or holds nested fields"))]
    #[case::a_list(read(&[3], &[0]), Some("names field 3 ('tags'), which is nested or holds nested fields"))]
    #[case::a_list_item(read(&[4], &[0]), Some("names field 4 ('item'), which is nested or holds nested fields"))]
    #[case::a_reserved_id(read(&[-3], &[0]), Some("names reserved field id -3"))]
    #[case::a_field_minted_in_the_operation(
        Action::AssertCellsUnchanged(AssertCellsUnchanged {
            fields: vec![FieldRef::Local(0)],
            rows: vec![RowSelection { fragment_id: 0, rows: SelectedRows::Full }],
        }),
        Some("names a field minted in the same operation")
    )]
    fn test_an_assertion_names_cells_that_exist(
        #[case] assertion: Action,
        #[case] expected_error: Option<&str>,
    ) {
        let actions = vec![
            Action::AddField(AddField {
                local: 0,
                parent: None,
                def: added_field("fresh"),
            }),
            assertion,
        ];
        let result = apply(&nested_manifest(), actions);
        match expected_error {
            None => {
                result.unwrap();
            }
            Some(message) => {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
                assert!(
                    error.to_string().contains(message),
                    "unexpected error: {error}"
                );
            }
        }
    }

    #[rstest]
    #[case::repeated_fragment(
        vec![
            pb_actions::RowSelection {
                fragment_id: 0,
                coverage: Some(pb_actions::row_selection::Coverage::Full(())),
            },
            pb_actions::RowSelection {
                fragment_id: 0,
                coverage: Some(pb_actions::row_selection::Coverage::Full(())),
            },
        ],
        "selects rows of fragment 0 more than once"
    )]
    #[case::bad_bitmap(
        vec![pb_actions::RowSelection {
            fragment_id: 0,
            coverage: Some(pb_actions::row_selection::Coverage::Bitmap(vec![1, 2, 3])),
        }],
        "not a portable Roaring bitmap"
    )]
    #[case::no_coverage(
        vec![pb_actions::RowSelection { fragment_id: 0, coverage: None }],
        "without saying which"
    )]
    fn test_decoding_rejects_a_malformed_row_selection(
        #[case] rows: Vec<pb_actions::RowSelection>,
        #[case] expected: &str,
    ) {
        let error = AssertCellsUnchanged::try_from(pb_actions::AssertCellsUnchanged {
            fields: vec![FieldRef::Committed(0).into()],
            rows,
        })
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains(expected),
            "unexpected error: {error}"
        );
    }
}
