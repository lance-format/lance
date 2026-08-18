// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Remove a field from the schema.

use super::apply::{ApplyState, TOMBSTONED_FIELD};
use super::proto::required;
use super::{Footprint, Ref};
use crate::format::pb;
use lance_core::datatypes::Field;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use std::collections::HashSet;

/// Remove a field from the schema.
///
/// The field's descendants go with it, since a struct's children cannot outlive
/// it. At apply, any data file left backing no live field is dropped, and any
/// index over a removed field is discarded.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct DropField {
    pub field: Ref,
}

impl DropField {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let field_id = state.resolve_field(self.field)?;
        let field = state.schema().field_by_id(field_id).ok_or_else(|| {
            Error::invalid_input(format!(
                "DropField names field {field_id}, which does not exist"
            ))
        })?;
        // A struct's children cannot outlive it, so the whole subtree goes.
        let mut dropped = HashSet::new();
        collect_subtree_ids(field, &mut dropped);
        remove_field(&mut state.schema_mut().fields, field_id);

        // The fields are gone from the schema, so the slots that backed them in
        // each data file are dead. Tombstoning rather than rewriting the field
        // list keeps a file's remaining columns at the positions they were
        // written at; a file left with nothing live is pruned during
        // normalization.
        for fragment in state.fragments_mut() {
            for file in fragment.files.iter_mut() {
                if !file.fields.iter().any(|id| dropped.contains(id)) {
                    continue;
                }
                let fields = file
                    .fields
                    .iter()
                    .map(|id| {
                        if dropped.contains(id) {
                            TOMBSTONED_FIELD
                        } else {
                            *id
                        }
                    })
                    .collect::<Vec<_>>();
                file.fields = fields.into();
            }

            let overlaid: Vec<u32> = dropped
                .iter()
                .filter_map(|id| u32::try_from(*id).ok())
                .collect();
            crate::format::overlay::tombstone_overlay_fields(&mut fragment.overlays, &overlaid);
        }

        // Indices over a field that no longer exists are discarded wholesale by
        // `retain_relevant_indices`, so there is nothing to record here.
        Ok(())
    }

    /// Dropping a field discards every value it held.
    pub(super) fn is_data_change(&self) -> bool {
        true
    }

    /// The field's definition and all of its data, which cannot be enumerated,
    /// so the removal is recorded as such and matched by field id.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.remove_field(self.field);
    }
}

fn collect_subtree_ids(field: &Field, out: &mut HashSet<i32>) {
    out.insert(field.id);
    for child in &field.children {
        collect_subtree_ids(child, out);
    }
}

/// Remove the field with `field_id` from `fields`, at whatever depth it sits.
fn remove_field(fields: &mut Vec<Field>, field_id: i32) {
    let before = fields.len();
    fields.retain(|field| field.id != field_id);
    if fields.len() != before {
        return;
    }
    for field in fields.iter_mut() {
        remove_field(&mut field.children, field_id);
    }
}

impl From<&DropField> for pb::DropField {
    fn from(value: &DropField) -> Self {
        Self {
            field: Some(value.field.into()),
        }
    }
}

impl TryFrom<pb::DropField> for DropField {
    type Error = Error;

    fn try_from(message: pb::DropField) -> Result<Self> {
        Ok(Self {
            field: required(message.field, "DropField.field")?.try_into()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::transaction::action::test_support::{
        added_field, apply, apply_with_indices, backed_manifest,
    };
    use crate::transaction::action::{Action, AddField};
    use crate::transaction::test_support::sample_index_metadata;
    use arrow_schema::{DataType, Field as ArrowField};
    use std::sync::Arc;

    #[test]
    fn test_drop_field_removes_it_and_its_data() {
        let (next, indices) = apply_with_indices(
            &backed_manifest(),
            vec![Action::DropField(DropField {
                field: Ref::Committed(0),
            })],
            vec![sample_index_metadata("idx")],
        )
        .unwrap();

        assert!(next.schema.field_by_id(0).is_none());
        // The file backed only the dropped field, so nothing is left of it.
        assert!(next.fragments[0].files.is_empty());
        // An index over a field that no longer exists is discarded outright.
        assert!(indices.is_empty());
    }

    #[test]
    fn test_drop_field_keeps_a_file_with_a_surviving_field() {
        let mut manifest = backed_manifest();
        let mut schema_field = added_field("keep");
        schema_field.id = 1;
        manifest.schema.fields.push(schema_field);
        let mut fragment = manifest.fragments[0].clone();
        fragment.files[0] = DataFile::new("data/0.lance", vec![0, 1], vec![0, 1], 2, 0, None, None);
        manifest.fragments = Arc::new(vec![fragment]);

        let next = apply(
            &manifest,
            vec![Action::DropField(DropField {
                field: Ref::Committed(0),
            })],
        )
        .unwrap();

        assert!(next.schema.field_by_id(0).is_none());
        assert!(next.schema.field_by_id(1).is_some());
        // The surviving field stays at the position it was written at.
        assert_eq!(
            next.fragments[0].files[0].fields.as_ref(),
            &[TOMBSTONED_FIELD, 1]
        );
    }

    #[test]
    fn test_drop_field_takes_the_whole_subtree() {
        let mut manifest = backed_manifest();
        let mut parent = Field::try_from(ArrowField::new("parent", DataType::Int32, true)).unwrap();
        parent.id = 1;
        let mut child = added_field("child");
        child.id = 2;
        child.parent_id = 1;
        parent.children.push(child);
        manifest.schema.fields.push(parent);

        let next = apply(
            &manifest,
            vec![Action::DropField(DropField {
                field: Ref::Committed(1),
            })],
        )
        .unwrap();

        assert!(next.schema.field_by_id(1).is_none());
        assert!(
            next.schema.field_by_id(2).is_none(),
            "a struct's children cannot outlive it"
        );
    }

    #[test]
    fn test_drop_field_rejects_a_missing_field() {
        let error = apply(
            &backed_manifest(),
            vec![Action::DropField(DropField {
                field: Ref::Committed(7),
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("field 7"), "{error}");
    }

    #[test]
    fn test_drop_field_then_add_a_field_reuses_no_id() {
        let next = apply(
            &backed_manifest(),
            vec![
                Action::DropField(DropField {
                    field: Ref::Committed(0),
                }),
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("replacement"),
                }),
            ],
        )
        .unwrap();

        // Field ids come from a monotonic counter, never from the freed id --
        // an old data file naming id 0 must not be read as the new field.
        let ids = next
            .schema
            .fields_pre_order()
            .map(|field| field.id)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn test_drop_field_rejects_a_field_id_out_of_range() {
        let error = apply(
            &backed_manifest(),
            vec![Action::DropField(DropField {
                field: Ref::Committed(u64::from(u32::MAX) + 1),
            })],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("out of range"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn test_drop_field_can_name_a_field_minted_in_the_same_operation() {
        // A squash of "add column" and "drop column" lowers to exactly this,
        // and has no committed id to name the field by.
        let next = apply(
            &backed_manifest(),
            vec![
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("transient"),
                }),
                Action::DropField(DropField {
                    field: Ref::Local(0),
                }),
            ],
        )
        .unwrap();

        assert!(next.schema.field("transient").is_none());
    }
}
