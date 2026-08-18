// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Alter facets of an existing field in place.

use super::apply::ApplyState;
use super::proto::{field_id_from_wire, field_id_to_wire, required};
use super::{Coordinate, Footprint};
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
#[derive(Debug, Clone, PartialEq, DeepSizeOf, Default)]
pub struct AlterField {
    pub field: i32,
    pub name: Option<String>,
    /// The new Arrow logical type. The cast.
    pub logical_type: Option<String>,
    pub nullable: Option<bool>,
}

impl AlterField {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let field = state
            .schema_mut()
            .field_by_id_mut(self.field)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "AlterField names field {}, which does not exist",
                    self.field
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
            state.rebind_field_everywhere(self.field);
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
        footprint.add(Coordinate::FieldDefinition(self.field));
    }
}

impl From<&AlterField> for pb::AlterField {
    fn from(value: &AlterField) -> Self {
        Self {
            field: Some(field_id_to_wire(value.field)),
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
            field: field_id_from_wire(required(message.field, "AlterField.field")?)?,
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
                field: 0,
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
                field: 0,
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
                field: 7,
                name: Some("nope".into()),
                ..Default::default()
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("field 7"), "{error}");
    }
}
