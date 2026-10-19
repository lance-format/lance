// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Alter facets of an existing field in place.

use super::apply::ApplyState;
use super::proto::required;
use super::{Coordinate, Footprint, Ref};
use crate::format::pb_transaction;
use lance_core::datatypes::LogicalType;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Alter facets of an existing field in place, preserving its id.
///
/// Each facet is independently optional -- present means "change this", absent
/// means "leave it alone".
///
/// `logical_type` is on the wire but rejected when applied: a cast is a
/// [`DropField`](super::DropField), an [`AddField`](super::AddField) under the
/// same name, and the data.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AlterField {
    pub field: Ref,
    pub name: Option<String>,
    /// The new type. Rejected at apply; see the type's docs.
    ///
    /// A [`LogicalType`], Lance's own string encoding of an Arrow type and
    /// what [`Field::logical_type`](lance_core::datatypes::Field::logical_type)
    /// holds, not an [`arrow_schema::DataType`]. The schema persists the string
    /// form, so the action names the same thing the manifest stores rather than
    /// a type that would have to be converted at both ends.
    pub logical_type: Option<LogicalType>,
    pub nullable: Option<bool>,
}

impl AlterField {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        if let Some(logical_type) = &self.logical_type {
            // A cast that rewrites values must not keep the field's id: the
            // id is durable identity, and once ids are non-reusable it may
            // not name a different type in a later version. Only a promotion
            // that leaves stored values valid could keep it, and none is
            // supported yet. A cast is spelled as a DropField, an AddField
            // with the same name and the new type, and the data. See the
            // lineage issue (`derived_from`) for the explicit form.
            return Err(Error::not_supported(format!(
                "AlterField cannot change field {:?}'s type to {logical_type} in place; \
                 use DropField, then AddField with the new type, then AddDataFile for its data",
                self.field
            )));
        }
        let field_id = state.resolve_field(self.field)?;
        let missing = || {
            Error::invalid_input(format!(
                "AlterField names field {field_id}, which does not exist"
            ))
        };
        if let Some(name) = &self.name {
            let field = state.schema().field_by_id(field_id).ok_or_else(missing)?;
            if *name != field.name {
                let parent_id = (field.parent_id >= 0).then_some(field.parent_id);
                state.reject_duplicate_sibling_name(parent_id, name, "AlterField")?;
            }
        }
        let field = state
            .schema_mut()
            .field_by_id_mut(field_id)
            .ok_or_else(missing)?;
        if let Some(name) = &self.name {
            field.name.clone_from(name);
        }
        if let Some(nullable) = self.nullable {
            field.nullable = nullable;
        }
        Ok(())
    }

    /// Renaming or changing nullability leaves the values alone.
    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    /// The field's definition, and the name a rename takes -- a name is a
    /// coordinate whether a field is added under it or moved to it.
    ///
    /// Tightening to non-nullable is only valid against the values it read,
    /// and applying it does not look at any. Values written since, or rows
    /// added that leave the field unset, may hold nulls. Landing first, it
    /// makes the field required like a non-nullable add does, which fences a
    /// stale append whether or not that append names the field -- including a
    /// field minted in this set, which no other writer can name.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.write_field_definition(self.field);
        if self.nullable == Some(false) {
            footprint.require_no_values_written(self.field);
            footprint.add_required_column();
        }
        if let Some(name) = &self.name {
            footprint.write(Coordinate::FieldName(name.clone()));
        }
    }
}

impl From<&AlterField> for pb_transaction::AlterField {
    fn from(value: &AlterField) -> Self {
        Self {
            field: Some(value.field.into()),
            name: value.name.clone(),
            logical_type: value
                .logical_type
                .as_ref()
                .map(|logical_type| logical_type.to_string()),
            nullable: value.nullable,
        }
    }
}

impl TryFrom<pb_transaction::AlterField> for AlterField {
    type Error = Error;

    fn try_from(message: pb_transaction::AlterField) -> Result<Self> {
        Ok(Self {
            field: required(message.field, "AlterField.field")?.try_into()?,
            name: message.name,
            logical_type: message.logical_type.as_deref().map(LogicalType::from),
            nullable: message.nullable,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::action::Action;
    use crate::transaction::action::test_support::{apply, apply_with_indices, backed_manifest};
    use crate::transaction::test_support::sample_index_metadata;

    #[test]
    fn test_alter_field_renames_without_touching_indices() {
        let (next, indices) = apply_with_indices(
            &backed_manifest(),
            vec![Action::AlterField(AlterField {
                field: Ref::Committed(0),
                name: Some("renamed".into()),
                logical_type: None,
                nullable: Some(true),
            })],
            vec![sample_index_metadata("idx")],
        )
        .unwrap();

        let field = next.schema.field_by_id(0).unwrap();
        assert_eq!(field.name, "renamed");
        assert!(field.nullable);
        // A rename does not change the values the index recorded.
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().contains(0));
    }

    #[test]
    fn test_alter_field_rejects_a_retype() {
        let error = apply(
            &backed_manifest(),
            vec![Action::AlterField(AlterField {
                field: Ref::Committed(0),
                name: None,
                logical_type: Some("int64".into()),
                nullable: None,
            })],
        )
        .unwrap_err();

        assert!(
            matches!(error, Error::NotSupported { .. }),
            "unexpected error: {error}"
        );
        assert!(
            error.to_string().contains("DropField"),
            "the error should point at drop + add: {error}"
        );
    }

    #[test]
    fn test_alter_field_rejects_a_missing_field() {
        let error = apply(
            &backed_manifest(),
            vec![Action::AlterField(AlterField {
                field: Ref::Committed(7),
                name: Some("nope".into()),
                logical_type: None,
                nullable: None,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("field 7"), "{error}");
    }

    #[test]
    fn test_alter_field_rejects_renaming_onto_a_sibling() {
        use crate::transaction::action::AddField;
        use crate::transaction::action::test_support::added_field;

        let taken = Action::AddField(AddField {
            local: 0,
            parent: None,
            def: added_field("taken"),
        });
        let rename = |name: &str| {
            Action::AlterField(AlterField {
                field: Ref::Committed(0),
                name: Some(name.into()),
                logical_type: None,
                nullable: None,
            })
        };

        let error = apply(&backed_manifest(), vec![taken.clone(), rename("taken")]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("unique among siblings"),
            "{error}"
        );

        // Restating a field's own name is not a collision with itself.
        let next = apply(&backed_manifest(), vec![taken, rename("id")]).unwrap();
        assert_eq!(next.schema.field_by_id(0).unwrap().name, "id");
    }

    #[test]
    fn test_alter_field_can_name_a_field_minted_in_the_same_operation() {
        // A squash of "add column" and "rename column" lowers to exactly this,
        // and has no committed id to name the field by.
        use crate::transaction::action::AddField;
        use crate::transaction::action::test_support::added_field;

        let next = apply(
            &backed_manifest(),
            vec![
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("before"),
                }),
                Action::AlterField(AlterField {
                    field: Ref::Local(0),
                    name: Some("after".into()),
                    logical_type: None,
                    nullable: None,
                }),
            ],
        )
        .unwrap();

        assert!(next.schema.field("before").is_none());
        assert!(next.schema.field("after").is_some());
    }
}
