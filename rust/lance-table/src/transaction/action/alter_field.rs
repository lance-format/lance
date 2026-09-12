// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Alter facets of an existing field in place.

use super::apply::ApplyState;
use super::proto::required;
use super::{Footprint, Ref};
use crate::format::pb;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Alter facets of an existing field in place, preserving its id.
///
/// Each facet is independently optional -- present means "change this", absent
/// means "leave it alone" -- so a widening cast and a nullability relaxation on
/// the same field commute. A cast additionally needs a
/// [`TombstoneFieldData`](super::TombstoneFieldData) plus a fresh
/// [`AddDataFile`](super::AddDataFile) to rewrite the data.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AlterField {
    pub field: Ref,
    pub name: Option<String>,
    /// The new Arrow logical type. The cast.
    pub logical_type: Option<String>,
    pub nullable: Option<bool>,
}

impl AlterField {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let field_id = state.resolve_field(self.field)?;
        let field = state
            .schema_mut()
            .field_by_id_mut(field_id)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "AlterField names field {field_id}, which does not exist"
                ))
            })?;
        if let Some(name) = &self.name {
            field.name.clone_from(name);
        }
        if let Some(nullable) = self.nullable {
            field.nullable = nullable;
        }
        if let Some(logical_type) = &self.logical_type {
            field.logical_type = logical_type.as_str().into();
            // The cast leaves any index on the field describing the old type.
            // The data rewrite itself is separate actions; this only records
            // that every fragment's view of the field changed.
            state.rebind_field_everywhere(field_id);
        }
        Ok(())
    }

    /// Renaming or relaxing a field leaves the values alone, and the
    /// rewrite a cast needs is separate actions that answer for themselves.
    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    /// The field's definition. The data rewrite a cast needs is separate
    /// actions, which record their own coordinates.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.add_field_definition(self.field);
    }
}

impl From<&AlterField> for pb::AlterField {
    fn from(value: &AlterField) -> Self {
        Self {
            field: Some(value.field.into()),
            name: value.name.clone(),
            logical_type: value.logical_type.clone(),
            nullable: value.nullable,
        }
    }
}

impl TryFrom<pb::AlterField> for AlterField {
    type Error = Error;

    fn try_from(message: pb::AlterField) -> Result<Self> {
        Ok(Self {
            field: required(message.field, "AlterField.field")?.try_into()?,
            name: message.name,
            logical_type: message.logical_type,
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
    fn test_alter_field_retype_prunes_covering_indices() {
        let (next, indices) = apply_with_indices(
            &backed_manifest(),
            vec![Action::AlterField(AlterField {
                field: Ref::Committed(0),
                name: None,
                logical_type: Some("int64".into()),
                nullable: None,
            })],
            vec![sample_index_metadata("idx")],
        )
        .unwrap();

        assert_eq!(
            next.schema.field_by_id(0).unwrap().logical_type.to_string(),
            "int64"
        );
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().is_empty());
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
