// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Mint a new schema field.

use super::apply::ApplyState;
use super::proto::required;
use super::{Footprint, Ref};
use crate::format::pb;
use lance_core::datatypes::Field;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Mint a new schema field.
///
/// A nested column that introduces several fields is several ordered
/// `AddField`s -- parent first, each child naming its parent's local token.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddField {
    /// Token standing in for the field id until it is allocated at apply.
    pub local: u32,
    /// The parent field, or `None` for a top-level column.
    pub parent: Option<Ref>,
    /// The field definition. Its `id`, `parent_id`, and `children` are ignored:
    /// `local` and `parent` carry that structure, and each child is its own
    /// action.
    pub def: Field,
}

impl AddField {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let id = state.mint_field(self.local)?;
        let parent_id = self
            .parent
            .map(|parent| state.resolve_field(parent))
            .transpose()?;

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

    /// Nothing: the field does not exist in the read version. Attaching it
    /// under a committed parent does not rewrite the parent's definition.
    pub(super) fn footprint(&self, _footprint: &mut Footprint) {}
}

impl From<&AddField> for pb::AddField {
    fn from(value: &AddField) -> Self {
        Self {
            local: value.local,
            parent: value.parent.map(Into::into),
            def: Some(lance_file::format::pb::Field::from(&value.def)),
        }
    }
}

impl TryFrom<pb::AddField> for AddField {
    type Error = Error;

    fn try_from(message: pb::AddField) -> Result<Self> {
        Ok(Self {
            local: message.local,
            parent: message.parent.map(Ref::try_from).transpose()?,
            def: Field::from(&required(message.def, "AddField.def")?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::transaction::action::test_support::{added_field, apply};
    use crate::transaction::action::{Action, AddDataFile};
    use crate::transaction::test_support::sample_manifest;
    use arrow_schema::{DataType, Field as ArrowField};
    use lance_file::version::ConcreteFileVersion;

    #[test]
    fn test_two_add_fields_mint_distinct_ids() {
        let manifest = sample_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("a"),
                }),
                Action::AddField(AddField {
                    local: 1,
                    parent: None,
                    def: added_field("b"),
                }),
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
        let manifest = sample_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddField(AddField {
                    local: 7,
                    parent: None,
                    def: added_field("added"),
                }),
                Action::AddDataFile(AddDataFile {
                    fragment: Ref::Committed(0),
                    file: DataFile::new_unstarted("data/added.lance", ConcreteFileVersion::V2_0),
                    field_ids: vec![Ref::Local(7)],
                    data_change: true,
                }),
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
        let manifest = sample_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: Field::try_from(ArrowField::new(
                        "nested",
                        DataType::Struct(Default::default()),
                        true,
                    ))
                    .unwrap(),
                }),
                Action::AddField(AddField {
                    local: 1,
                    parent: Some(Ref::Local(0)),
                    def: added_field("child"),
                }),
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
        let manifest = sample_manifest();
        let error = apply(
            &manifest,
            vec![Action::AddField(AddField {
                local: 0,
                parent: Some(Ref::Committed(7)),
                def: added_field("orphan"),
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("parent field 7"), "{error}");
    }

    #[test]
    fn test_duplicate_local_token_is_rejected() {
        let manifest = sample_manifest();
        let error = apply(
            &manifest,
            vec![
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("a"),
                }),
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("b"),
                }),
            ],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("minted more than once"),
            "unexpected error: {error}"
        );
    }
}
