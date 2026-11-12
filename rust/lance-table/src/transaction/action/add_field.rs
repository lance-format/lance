// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Mint a new schema field.

use super::apply::ApplyState;
use super::proto::required;
use super::{Coordinate, FieldRef, Footprint};
use crate::format::pb_actions;
use lance_core::datatypes::Field;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Mint a new schema field.
///
/// A nested column that introduces several fields is several ordered
/// `AddField`s -- parent first, each child naming its parent's local token.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddField {
    local: u32,
    parent: Option<FieldRef>,
    def: Field,
}

impl AddField {
    /// Mint a top-level field defined by `def` under the token `local`, which
    /// stands in for the field id until it is allocated at apply.
    ///
    /// The definition's `id`, `parent_id`, and `children` are ignored: `local`
    /// and [`with_parent`](Self::with_parent) carry that structure, and each
    /// child is its own action.
    pub fn new(local: u32, def: Field) -> Self {
        Self {
            local,
            parent: None,
            def,
        }
    }

    /// Nest the field under `parent`: committed, or minted earlier in the same
    /// operation.
    pub fn with_parent(mut self, parent: FieldRef) -> Self {
        self.parent = Some(parent);
        self
    }

    /// The token later actions name the field by.
    pub fn local(&self) -> u32 {
        self.local
    }

    /// The parent field, or `None` for a top-level column.
    pub fn parent(&self) -> Option<FieldRef> {
        self.parent
    }

    /// The field definition, with its structural fields ignored.
    pub fn def(&self) -> &Field {
        &self.def
    }

    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let id = state.mint_field(self.local)?;
        let parent_id = self
            .parent
            .map(|parent| state.resolve_schema_field(parent, "AddField"))
            .transpose()?;
        state.check_field_name(parent_id, &self.def.name, "AddField")?;

        // The definition's own id, parent id, and children are ignored: the
        // minted id and the parent reference carry that structure, and each
        // child column arrives as its own action.
        let field = Field {
            id,
            parent_id: parent_id.unwrap_or(-1),
            children: Vec::new(),
            ..self.def.clone()
        };

        match parent_id {
            None => state.schema_mut().fields.push(field),
            Some(parent_id) => {
                let parent = state
                    .schema_mut()
                    .field_by_id_mut(parent_id)
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "AddField names parent field {parent_id}, which does not exist"
                        ))
                    })?;
                parent.children.push(field);
            }
        }
        Ok(())
    }

    /// A new field starts empty; an
    /// [`AddDataFile`](super::AddDataFile) is what gives it values.
    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    /// The field does not exist in the read version, so its definition is not
    /// a coordinate. Its name is: two writers adding a column by the same name
    /// would otherwise both land. Attaching it under a committed parent does
    /// not rewrite the parent's definition, but it does need the parent to
    /// still be there.
    ///
    /// A non-nullable field is also a promise about rows: each must carry a
    /// value for it. Its data only covers the fragments the writer saw, so
    /// rows added by a concurrent writer, in either order, would break it.
    ///
    /// A child under a committed parent changes the shape of the parent's
    /// rows, and a stale writer may write the parent packed without the child,
    /// so it fences new rows too. This is coarse: it does not ask whether the
    /// parent is packed.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.write(Coordinate::FieldName(self.def.name.clone()));
        if !self.def.nullable {
            footprint.add_required_column();
        }
        if let Some(parent) = self.parent {
            footprint.require_field_definition(parent);
            footprint.add_child(parent);
            if parent.committed().is_some() {
                footprint.write(Coordinate::NewRowSchema);
            }
        }
    }
}

impl From<&AddField> for pb_actions::AddField {
    fn from(value: &AddField) -> Self {
        Self {
            local: value.local,
            parent: value.parent.map(Into::into),
            def: Some(lance_file::format::pb::Field::from(&value.def)),
        }
    }
}

impl TryFrom<pb_actions::AddField> for AddField {
    type Error = Error;

    fn try_from(message: pb_actions::AddField) -> Result<Self> {
        let field = Self::new(
            message.local,
            Field::from(&required(message.def, "AddField.def")?),
        );
        Ok(match message.parent {
            Some(parent) => field.with_parent(parent.try_into()?),
            None => field,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::action::test_support::{
        added_field, apply, one_column_file, unbacked_manifest,
    };
    use crate::transaction::action::{Action, AddDataFile, Ref};
    use arrow_schema::{DataType, Field as ArrowField};

    #[test]
    fn test_two_add_fields_mint_distinct_ids() {
        let manifest = unbacked_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddField(AddField::new(0, added_field("a"))),
                Action::AddField(AddField::new(1, added_field("b"))),
            ],
        )
        .unwrap();

        let ids = next
            .schema
            .fields
            .iter()
            .map(|f| (f.name.as_str(), f.id))
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![("id", 0), ("a", 1), ("b", 2)]);
    }

    #[test]
    fn test_add_field_then_add_its_data_file() {
        // The add-column shape: mint the field, then write the file that backs
        // it, naming the field by the token the mint has not resolved yet.
        let manifest = unbacked_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddField(AddField::new(7, added_field("added"))),
                Action::AddDataFile(
                    AddDataFile::try_new_unchecked(
                        Ref::Committed(0),
                        one_column_file("data/added.lance"),
                        vec![FieldRef::Local(7)],
                        None,
                        true,
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();

        let field_id = next.schema.field("added").unwrap().id;
        assert_eq!(field_id, 1);
        let fragment = next.fragments.iter().find(|f| f.id == 0).unwrap();
        assert_eq!(fragment.files.last().unwrap().fields.as_ref(), &[field_id]);
    }

    #[test]
    fn test_add_field_under_a_parent() {
        let manifest = unbacked_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddField(AddField::new(
                    0,
                    Field::try_from(ArrowField::new(
                        "nested",
                        DataType::Struct(Default::default()),
                        true,
                    ))
                    .unwrap(),
                )),
                Action::AddField(
                    AddField::new(1, added_field("child")).with_parent(FieldRef::Local(0)),
                ),
            ],
        )
        .unwrap();

        let parent = next.schema.field("nested").unwrap();
        assert_eq!(parent.children.len(), 1);
        assert_eq!(parent.children[0].name, "child");
        assert_eq!(parent.children[0].parent_id, parent.id);
    }

    #[test]
    fn test_add_field_rejects_a_missing_parent() {
        let manifest = unbacked_manifest();
        let error = apply(
            &manifest,
            vec![Action::AddField(
                AddField::new(0, added_field("orphan")).with_parent(FieldRef::Committed(7)),
            )],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        // Above the read version's field mark, so it cannot name a parent.
        assert!(error.to_string().contains("field 7"), "{error}");
    }

    #[test]
    fn test_add_field_rejects_a_name_a_sibling_already_has() {
        // `sample_manifest` already has a top-level field called "id".
        let manifest = unbacked_manifest();
        let error = apply(
            &manifest,
            vec![Action::AddField(AddField::new(0, added_field("id")))],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("unique among siblings"),
            "{error}"
        );

        // Under a parent, only that parent's children are siblings, so the
        // top-level name is free again.
        let nested = Field::try_from(ArrowField::new(
            "nested",
            DataType::Struct(Default::default()),
            true,
        ))
        .unwrap();
        let next = apply(
            &manifest,
            vec![
                Action::AddField(AddField::new(0, nested.clone())),
                Action::AddField(
                    AddField::new(1, added_field("id")).with_parent(FieldRef::Local(0)),
                ),
            ],
        )
        .unwrap();
        assert_eq!(next.schema.field("nested").unwrap().children[0].name, "id");

        let error = apply(
            &manifest,
            vec![
                Action::AddField(AddField::new(0, nested)),
                Action::AddField(
                    AddField::new(1, added_field("child")).with_parent(FieldRef::Local(0)),
                ),
                Action::AddField(
                    AddField::new(2, added_field("child")).with_parent(FieldRef::Local(0)),
                ),
            ],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("unique among siblings"),
            "{error}"
        );
    }

    #[test]
    fn test_duplicate_local_token_is_rejected() {
        let manifest = unbacked_manifest();
        let error = apply(
            &manifest,
            vec![
                Action::AddField(AddField::new(0, added_field("a"))),
                Action::AddField(AddField::new(0, added_field("b"))),
            ],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("minted more than once"),
            "unexpected error: {error}"
        );
    }
}
