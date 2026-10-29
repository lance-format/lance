// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Add a data file to a fragment.

use super::apply::ApplyState;
use super::proto::{data_change_from_wire, data_change_to_wire, required};
use super::{FieldRef, Footprint, Ref};
use crate::format::{DataFile, pb, pb_actions};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use std::collections::HashMap;

/// Add a data file to a fragment.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddDataFile {
    /// The fragment to add the file to: committed, or a fragment minted earlier
    /// in the same operation.
    pub fragment: Ref,
    /// The file, with `fields` left empty: apply stamps them from `field_ids`,
    /// the only field list the action carries.
    pub file: DataFile,
    /// One entry per column in `file`, in column order.
    pub field_ids: Vec<FieldRef>,
    /// See [`AddFragment::data_change`](super::AddFragment::data_change).
    pub data_change: bool,
    /// The base path the file lives under: committed, or one minted earlier in
    /// the same operation. `None` for the dataset's own data directory.
    pub base: Option<Ref>,
}

impl AddDataFile {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let fragment_id = state.resolve_fragment(self.fragment)?;
        let resolved = self
            .field_ids
            .iter()
            .map(|field| state.resolve_field(*field))
            .collect::<Result<Vec<_>>>()?;
        let field_ids: Vec<i32> = resolved.iter().map(|id| id.get()).collect();
        let spilled_lineage = state
            .fragment_mut(fragment_id, "AddDataFile")?
            .spilled_row_lineage_field_ids();
        for id in &resolved {
            if id.is_schema() {
                // A column for a field the schema does not have is data no
                // reader can see, such as one dropped earlier in the operation.
                if state.schema().field_by_id(id.get()).is_none() {
                    return Err(Error::invalid_input(format!(
                        "AddDataFile for fragment {fragment_id} names field {id}, which does \
                         not exist (field ids {field_ids:?})"
                    )));
                }
            } else if id.is_row_lineage() {
                // The column only means something to a fragment whose metadata
                // points at it, and only a fragment added here can still be
                // given one: a committed fragment's sequences are settled.
                if !state.is_new_fragment(fragment_id) || !spilled_lineage.contains(&id.get()) {
                    return Err(Error::invalid_input(format!(
                        "AddDataFile for fragment {fragment_id} binds row lineage field {id}, \
                         but the fragment is not one this operation adds with that sequence \
                         spilled (field ids {field_ids:?})"
                    )));
                }
            } else {
                return Err(Error::invalid_input(format!(
                    "AddDataFile for fragment {fragment_id} names reserved field id {id}; a data \
                     file binds schema fields and its fragment's spilled row lineage columns \
                     only (field ids {field_ids:?})"
                )));
            }
        }

        if !self.file.fields.is_empty() {
            return Err(Error::invalid_input(format!(
                "AddDataFile for fragment {fragment_id} carries file.fields {:?}; \
                 file.fields must be empty, since field_ids is the file's field list",
                self.file.fields
            )));
        }
        if let Some(base_id) = self.file.base_id {
            return Err(Error::invalid_input(format!(
                "AddDataFile for fragment {fragment_id} carries file.base_id {base_id}; \
                 file.base_id must be unset, since the action's base names where the file lives"
            )));
        }
        let mut file = self.file.clone();
        file.base_id = self.base.map(|base| state.resolve_base(base)).transpose()?;
        if !file.column_indices.is_empty() && file.column_indices.len() != field_ids.len() {
            return Err(Error::invalid_input(format!(
                "AddDataFile for fragment {fragment_id} lists {} field ids but the file has {} \
                 columns",
                field_ids.len(),
                file.column_indices.len()
            )));
        }
        reject_repeat(fragment_id, "field", &field_ids, |_| true)?;
        // -1 marks a field with no top-level column of its own, so it may repeat.
        reject_repeat(fragment_id, "column", &file.column_indices, |column| {
            *column >= 0
        })?;
        // A field's values live in exactly one file, so a second file for the
        // same field is ambiguous. A struct's own id is the exception: a file
        // adding a child under a committed struct lists the parent too, and the
        // parent holds no values of its own unless it is packed. Re-encoding a
        // column goes through TombstoneFieldData first, which frees the slot.
        let exclusive_fields: Vec<i32> = field_ids
            .iter()
            .copied()
            .filter(|id| {
                state
                    .schema()
                    .field_by_id(*id)
                    .is_some_and(|field| field.children.is_empty() || field.is_packed_struct())
            })
            .collect();
        file.fields = field_ids.into();

        let fragment = state.fragment_mut(fragment_id, "AddDataFile")?;
        for field_id in exclusive_fields {
            if let Some(existing) = fragment
                .files
                .iter()
                .find(|existing| existing.fields.contains(&field_id))
            {
                return Err(Error::invalid_input(format!(
                    "AddDataFile for fragment {fragment_id} backs field {field_id}, which is \
                     already backed by {}; tombstone the field's data with TombstoneFieldData \
                     first to replace it",
                    existing.path
                )));
            }
        }
        let bound_fields = file.fields.clone();
        fragment.files.push(file);
        for field_id in bound_fields.iter() {
            state.note_rebind(fragment_id, *field_id, self.data_change)?;
        }
        Ok(())
    }

    pub(super) fn is_data_change(&self) -> bool {
        self.data_change
    }

    /// The data of every committed field the file backs, in the fragment it is
    /// attached to. A file backing only minted fields writes nothing.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        if self.data_change {
            footprint.write_field_data(self.fragment, self.field_ids.iter().copied());
        } else {
            footprint.move_field_data(self.fragment, self.field_ids.iter().copied());
        }
    }
}

/// Reject a value that appears twice among the `values` that `counts`, naming
/// both positions so the writer can find the pair.
fn reject_repeat(
    fragment_id: u64,
    what: &str,
    values: &[i32],
    counts: impl Fn(&i32) -> bool,
) -> Result<()> {
    let mut first_seen = HashMap::with_capacity(values.len());
    for (position, value) in values.iter().enumerate() {
        if !counts(value) {
            continue;
        }
        if let Some(first) = first_seen.insert(*value, position) {
            return Err(Error::invalid_input(format!(
                "AddDataFile for fragment {fragment_id} binds {what} {value} at positions {first} \
                 and {position}; each {what} is bound at most once per file"
            )));
        }
    }
    Ok(())
}

impl From<&AddDataFile> for pb_actions::AddDataFile {
    fn from(value: &AddDataFile) -> Self {
        Self {
            fragment: Some(value.fragment.into()),
            file: Some(pb::DataFile::from(&value.file)),
            field_ids: value.field_ids.iter().map(|id| (*id).into()).collect(),
            data_change: data_change_to_wire(value.data_change),
            base: value.base.map(Into::into),
        }
    }
}

impl TryFrom<pb_actions::AddDataFile> for AddDataFile {
    type Error = Error;

    fn try_from(message: pb_actions::AddDataFile) -> Result<Self> {
        Ok(Self {
            fragment: required(message.fragment, "AddDataFile.fragment")?.try_into()?,
            file: DataFile::try_from(required(message.file, "AddDataFile.file")?)?,
            field_ids: message
                .field_ids
                .into_iter()
                .map(FieldRef::try_from)
                .collect::<Result<Vec<_>>>()?,
            data_change: data_change_from_wire(message.data_change),
            base: message.base.map(Ref::try_from).transpose()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::RowIdMeta;
    use crate::rowids::write_row_ids;
    use crate::transaction::action::test_support::{
        added_field, apply, backed_manifest, manifest_with_base, mint_base, one_column_file,
        unbacked_manifest,
    };
    use crate::transaction::action::{Action, AddField, AddFragment, TombstoneFieldData};
    use crate::transaction::test_support::make_stable_row_id_manifest;
    use lance_core::ROW_ID_FIELD_ID;
    use lance_file::version::ConcreteFileVersion;
    use rstest::rstest;

    #[test]
    fn test_add_data_file_rejects_an_unbound_local_token() {
        let manifest = unbacked_manifest();
        let error = apply(
            &manifest,
            vec![Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(3),
                file: one_column_file("data/x.lance"),
                field_ids: vec![FieldRef::Committed(0)],
                data_change: true,
                base: None,
            })],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("local fragment token 3"),
            "unexpected error: {error}"
        );
    }

    /// `field_ids` is the only field list on the wire; a second one in the
    /// embedded file could only disagree with it.
    #[test]
    fn test_add_data_file_rejects_a_file_carrying_its_own_field_ids() {
        let mut file = one_column_file("data/x.lance");
        file.fields = vec![0].into();
        file.column_indices = vec![0].into();

        let error = apply(
            &unbacked_manifest(),
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
                    file,
                    field_ids: vec![FieldRef::Committed(0)],
                    data_change: true,
                    base: None,
                }),
            ],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("file.fields must be empty"),
            "unexpected error: {error}"
        );
    }

    /// `base` is where the file lives; the embedded `base_id` is stamped from
    /// it, so a writer that sets that too has said it twice.
    #[rstest]
    #[case::own_directory(None, None, false, Ok(None))]
    #[case::committed_base(None, Some(Ref::Committed(1)), false, Ok(Some(1)))]
    #[case::base_minted_earlier(None, Some(Ref::Local(0)), true, Ok(Some(2)))]
    #[case::embedded_base_id(Some(1), None, false, Err("file.base_id must be unset"))]
    #[case::embedded_base_id_beside_base(
        Some(1),
        Some(Ref::Committed(1)),
        false,
        Err("file.base_id must be unset")
    )]
    #[case::committed_base_the_dataset_lacks(
        None,
        Some(Ref::Committed(0)),
        false,
        Err("base id 0 in an action names no base path of the dataset")
    )]
    #[case::base_not_minted(
        None,
        Some(Ref::Local(0)),
        false,
        Err("local base token 0, which no earlier action in this operation minted")
    )]
    fn test_add_data_file_stamps_its_base(
        #[case] embedded_base_id: Option<u32>,
        #[case] base: Option<Ref>,
        #[case] mint_base_first: bool,
        #[case] expected: std::result::Result<Option<u32>, &str>,
    ) {
        let mut file = one_column_file("data/x.lance");
        file.base_id = embedded_base_id;
        let mut actions = Vec::new();
        if mint_base_first {
            actions.push(mint_base());
        }
        actions.push(Action::AddField(AddField {
            local: 0,
            parent: None,
            def: added_field("added"),
        }));
        actions.push(Action::AddDataFile(AddDataFile {
            fragment: Ref::Committed(0),
            file,
            field_ids: vec![FieldRef::Local(0)],
            data_change: true,
            base,
        }));

        let result = apply(&manifest_with_base(), actions);

        match expected {
            Ok(base_id) => {
                let next = result.unwrap();
                assert_eq!(next.fragments[0].files.last().unwrap().base_id, base_id);
            }
            Err(message) => {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
                assert!(
                    error.to_string().contains(message),
                    "unexpected error: {error}"
                );
            }
        }
    }

    fn new_fragment(row_id_meta: Option<RowIdMeta>) -> Action {
        Action::AddFragment(AddFragment {
            id: Ref::Local(0),
            physical_rows: 10,
            row_id_meta,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        })
    }

    fn file_binding(fragment: Ref, field_ids: &[i32]) -> Action {
        Action::AddDataFile(AddDataFile {
            fragment,
            file: DataFile {
                column_indices: (0..).take(field_ids.len()).collect(),
                ..DataFile::new_unstarted("data/x.lance", ConcreteFileVersion::V2_0)
            },
            field_ids: field_ids.iter().copied().map(FieldRef::Committed).collect(),
            data_change: true,
            base: None,
        })
    }

    /// A file binds schema fields, plus the hidden lineage column of a
    /// sequence its own fragment declared spilled in this operation. Every
    /// other negative id is a sentinel of the stored format, not a column.
    #[rstest]
    #[case::unassigned_placeholder(
        vec![new_fragment(None), file_binding(Ref::Local(0), &[0, -1])],
        Some("reserved field id -1")
    )]
    #[case::tombstone(
        vec![new_fragment(None), file_binding(Ref::Local(0), &[0, -2])],
        Some("reserved field id -2")
    )]
    #[case::unknown_negative(
        vec![new_fragment(None), file_binding(Ref::Local(0), &[0, -6])],
        Some("reserved field id -6")
    )]
    #[case::lineage_in_a_committed_fragment(
        vec![file_binding(Ref::Committed(0), &[ROW_ID_FIELD_ID])],
        Some("row lineage field -3")
    )]
    #[case::lineage_the_fragment_keeps_inline(
        vec![
            new_fragment(Some(RowIdMeta::Inline(write_row_ids(&(10..20).into()).into()))),
            file_binding(Ref::Local(0), &[0, ROW_ID_FIELD_ID]),
        ],
        Some("row lineage field -3")
    )]
    #[case::lineage_the_fragment_spills(
        vec![
            new_fragment(Some(RowIdMeta::Column)),
            file_binding(Ref::Local(0), &[0, ROW_ID_FIELD_ID]),
        ],
        None
    )]
    #[case::lineage_in_a_file_of_its_own(
        vec![
            new_fragment(Some(RowIdMeta::Column)),
            file_binding(Ref::Local(0), &[0]),
            file_binding(Ref::Local(0), &[ROW_ID_FIELD_ID]),
        ],
        None
    )]
    #[case::lineage_bound_twice_in_one_file(
        vec![
            new_fragment(Some(RowIdMeta::Column)),
            file_binding(Ref::Local(0), &[0, ROW_ID_FIELD_ID, ROW_ID_FIELD_ID]),
        ],
        Some("binds field -3 at positions 1 and 2")
    )]
    #[case::spilled_lineage_no_file_carries(
        vec![new_fragment(Some(RowIdMeta::Column)), file_binding(Ref::Local(0), &[0])],
        Some("spills row lineage field -3 but no AddDataFile binds it")
    )]
    #[case::spilled_lineage_two_files_carry(
        vec![
            new_fragment(Some(RowIdMeta::Column)),
            file_binding(Ref::Local(0), &[0, ROW_ID_FIELD_ID]),
            file_binding(Ref::Local(0), &[ROW_ID_FIELD_ID]),
        ],
        Some("binds row lineage field -3 in more than one data file")
    )]
    fn test_add_data_file_binds_only_schema_and_declared_lineage_fields(
        #[case] actions: Vec<Action>,
        #[case] expected_error: Option<&str>,
    ) {
        // Spilling row ids takes a dataset that has them.
        let mut fragment = backed_manifest().fragments[0].clone();
        fragment.row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&(0..10).into()).into()));
        let manifest = make_stable_row_id_manifest(vec![fragment]);

        let result = apply(&manifest, actions);
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

    /// A data file maps each field to one column and each column to one field,
    /// so a repeat in either list leaves a reader two answers. A column index
    /// of -1 (a field with no top-level column) is not a column, and a v1 file
    /// has no column indices at all.
    #[rstest]
    #[case::distinct(&[0, 1], &[0, 1], None)]
    #[case::repeated_field(&[0, 0], &[0, 1], Some("binds field 0 at positions 0 and 1"))]
    #[case::repeated_column(&[0, 1], &[2, 2], Some("binds column 2 at positions 0 and 1"))]
    #[case::repeated_unplaced_column(&[0, 1], &[-1, -1], None)]
    #[case::no_column_indices(&[0, 1], &[], None)]
    fn test_add_data_file_rejects_a_repeated_binding(
        #[case] field_ids: &[i32],
        #[case] column_indices: &[i32],
        #[case] expected_error: Option<&str>,
    ) {
        let mut manifest = backed_manifest();
        manifest.schema =
            lance_core::datatypes::Schema::try_from(&arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("a", arrow_schema::DataType::Int32, true),
                arrow_schema::Field::new("b", arrow_schema::DataType::Int32, true),
            ]))
            .unwrap();
        let mut file = one_column_file("data/x.lance");
        file.column_indices = column_indices.into();

        let result = apply(
            &manifest,
            vec![
                new_fragment(None),
                Action::AddDataFile(AddDataFile {
                    fragment: Ref::Local(0),
                    file,
                    field_ids: field_ids.iter().copied().map(FieldRef::Committed).collect(),
                    data_change: true,
                    base: None,
                }),
            ],
        );

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

    #[test]
    fn test_add_data_file_rejects_a_column_count_mismatch() {
        let manifest = unbacked_manifest();
        let mut file = one_column_file("data/x.lance");
        file.column_indices = vec![0, 1].into();

        let error = apply(
            &manifest,
            vec![Action::AddDataFile(AddDataFile {
                fragment: Ref::Committed(0),
                file,
                field_ids: vec![FieldRef::Committed(0)],
                data_change: true,
                base: None,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("1 field ids"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_add_data_file_rejects_a_missing_fragment() {
        let manifest = unbacked_manifest();
        let error = apply(
            &manifest,
            vec![Action::AddDataFile(AddDataFile {
                fragment: Ref::Committed(7),
                file: one_column_file("data/x.lance"),
                field_ids: vec![FieldRef::Committed(0)],
                data_change: true,
                base: None,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("fragment 7"), "{error}");
    }

    fn add_file_for_field_zero() -> Action {
        Action::AddDataFile(AddDataFile {
            fragment: Ref::Committed(0),
            file: one_column_file("data/new.lance"),
            field_ids: vec![FieldRef::Committed(0)],
            data_change: true,
            base: None,
        })
    }

    #[test]
    fn test_add_data_file_rejects_a_field_another_file_already_backs() {
        let error = apply(&backed_manifest(), vec![add_file_for_field_zero()]).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("already backed by data/0.lance"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_add_data_file_may_back_a_field_tombstoned_earlier_in_the_set() {
        let next = apply(
            &backed_manifest(),
            vec![
                Action::TombstoneFieldData(TombstoneFieldData {
                    fragment: Ref::Committed(0),
                    field_ids: vec![FieldRef::Committed(0)],
                    data_change: true,
                }),
                add_file_for_field_zero(),
            ],
        )
        .unwrap();

        let backing: Vec<&str> = next.fragments[0]
            .files
            .iter()
            .filter(|file| file.fields.contains(&0))
            .map(|file| file.path.as_str())
            .collect();
        assert_eq!(backing, ["data/new.lance"]);
    }
}
