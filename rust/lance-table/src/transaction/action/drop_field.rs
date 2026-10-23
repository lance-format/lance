// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Remove a field from the schema.

use super::apply::{ApplyState, TOMBSTONED_FIELD};
use super::proto::required;
use super::{FieldRef, Footprint};
use crate::format::pb_actions;
use lance_core::datatypes::Field;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Remove a field from the schema.
///
/// The field must have no children left: a struct or list is dropped by one
/// `DropField` per descendant, deepest first, then the parent, all in the same
/// operation. Naming every removed field is what makes the footprint complete,
/// since a footprint has no schema to expand a subtree with. At apply, any data
/// file left backing no live field is dropped, and any index over a removed
/// field is discarded.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct DropField {
    pub field: FieldRef,
}

impl DropField {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let field_id = state.resolve_schema_field(self.field, "DropField")?;
        let field = state.schema().field_by_id(field_id).ok_or_else(|| {
            Error::invalid_input(format!(
                "DropField names field {field_id}, which does not exist"
            ))
        })?;
        if !field.children.is_empty() {
            let children = field
                .children
                .iter()
                .map(|child| child.id.to_string())
                .collect::<Vec<_>>();
            return Err(Error::invalid_input(format!(
                "DropField names field {field_id}, which still has children [{}]; drop a \
                 field's children first, deepest first, in the same operation",
                children.join(", ")
            )));
        }
        remove_field(&mut state.schema_mut().fields, field_id);

        // The fields are gone from the schema, so the slots that backed them in
        // each data file are dead. Tombstoning rather than rewriting the field
        // list keeps a file's remaining columns at the positions they were
        // written at; a file left with nothing live is pruned during
        // normalization.
        for fragment in state.fragments_mut() {
            for file in fragment.files.iter_mut() {
                if !file.fields.contains(&field_id) {
                    continue;
                }
                let fields = file
                    .fields
                    .iter()
                    .map(|id| {
                        if *id == field_id {
                            TOMBSTONED_FIELD
                        } else {
                            *id
                        }
                    })
                    .collect::<Vec<_>>();
                file.fields = fields.into();
            }

            if let Ok(overlaid) = u32::try_from(field_id) {
                crate::format::overlay::tombstone_overlay_fields(
                    &mut fragment.overlays,
                    &[overlaid],
                );
            }
        }

        // Indices over a field that no longer exists are discarded wholesale by
        // `retain_relevant_indices`, so there is nothing to record here.
        Ok(())
    }

    /// Dropping a field discards every value it held.
    pub(super) fn is_data_change(&self) -> bool {
        true
    }

    /// The field's definition, and nothing else. Anything that depends on the
    /// field -- data written for it, its metadata, a child added under it --
    /// requires that definition, so it collides with the drop as `Requires`
    /// against `Writes`.
    ///
    /// That is complete for a struct too, because apply only drops a field
    /// with no children left: the children were each dropped by their own
    /// `DropField` in the same operation, which writes their definitions.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.write_field_definition(self.field);
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

impl From<&DropField> for pb_actions::DropField {
    fn from(value: &DropField) -> Self {
        Self {
            field: Some(value.field.into()),
        }
    }
}

impl TryFrom<pb_actions::DropField> for DropField {
    type Error = Error;

    fn try_from(message: pb_actions::DropField) -> Result<Self> {
        Ok(Self {
            field: required(message.field, "DropField.field")?.try_into()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::format::Manifest;
    use crate::transaction::action::test_support::{
        added_field, apply, apply_with_indices, backed_manifest,
    };
    use crate::transaction::action::{Action, AddField};
    use crate::transaction::test_support::sample_index_metadata;
    use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
    use lance_core::datatypes::Schema;
    use lance_file::version::ConcreteFileVersion;
    use rstest::rstest;
    use std::sync::Arc;

    #[test]
    fn test_drop_field_removes_it_and_its_data() {
        let (next, indices) = apply_with_indices(
            &backed_manifest(),
            vec![Action::DropField(DropField {
                field: FieldRef::Committed(0),
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
        fragment.files[0] = DataFile::new(
            "data/0.lance",
            vec![0, 1],
            vec![0, 1],
            ConcreteFileVersion::V2_0,
            None,
            None,
        );
        manifest.fragments = Arc::new(vec![fragment]);

        let next = apply(
            &manifest,
            vec![Action::DropField(DropField {
                field: FieldRef::Committed(0),
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

    /// `id` at 0, then `s: {a, t: {b}}` at 1-4, with fragment 0's second file
    /// backing the whole struct.
    fn manifest_with_struct() -> Manifest {
        let arrow = ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new(
                "s",
                DataType::Struct(Fields::from(vec![
                    ArrowField::new("a", DataType::Int32, true),
                    ArrowField::new(
                        "t",
                        DataType::Struct(Fields::from(vec![ArrowField::new(
                            "b",
                            DataType::Int32,
                            true,
                        )])),
                        true,
                    ),
                ])),
                true,
            ),
        ]);
        let mut manifest = backed_manifest();
        manifest.schema = Schema::try_from(&arrow).unwrap();
        let mut fragment = manifest.fragments[0].clone();
        fragment.files.push(DataFile::new(
            "data/0-s.lance",
            vec![1, 2, 3, 4],
            vec![0, 1, 2, 3],
            ConcreteFileVersion::V2_0,
            None,
            None,
        ));
        manifest.fragments = Arc::new(vec![fragment]);
        manifest
    }

    fn drop_field(id: i32) -> Action {
        Action::DropField(DropField {
            field: FieldRef::Committed(id),
        })
    }

    /// A drop names exactly one field, so that every field it removes is in
    /// its footprint; a struct with children still in place would take them
    /// with it unnamed.
    #[rstest]
    #[case::struct_with_children(vec![drop_field(1)], "field 1")]
    #[case::nested_struct_before_its_child(
        vec![drop_field(2), drop_field(3), drop_field(4), drop_field(1)],
        "field 3"
    )]
    fn test_drop_field_rejects_a_field_with_live_children(
        #[case] actions: Vec<Action>,
        #[case] named: &str,
    ) {
        let error = apply(&manifest_with_struct(), actions).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        let message = error.to_string();
        assert!(message.contains(named), "{message}");
        assert!(
            message.contains("children first, deepest first"),
            "{message}"
        );
    }

    #[test]
    fn test_drop_field_takes_a_struct_children_first() {
        let next = apply(
            &manifest_with_struct(),
            vec![drop_field(4), drop_field(3), drop_field(2), drop_field(1)],
        )
        .unwrap();

        assert_eq!(
            next.schema
                .fields_pre_order()
                .map(|field| field.id)
                .collect::<Vec<_>>(),
            vec![0]
        );
        // The struct's file backed nothing else, so it is pruned.
        assert_eq!(next.fragments[0].files.len(), 1);
        assert_eq!(next.fragments[0].files[0].path, "data/0.lance");
    }

    /// A drop that lands after a concurrent write to a child still applies:
    /// it tombstones the child out of every file the version it lands on
    /// holds, the concurrent one included.
    #[test]
    fn test_drop_field_landing_after_a_write_to_a_child() {
        let mut manifest = manifest_with_struct();
        let mut concurrent = manifest.fragments[0].clone();
        concurrent.id = 1;
        concurrent.files = vec![DataFile::new(
            "data/1-a.lance",
            vec![2],
            vec![0],
            ConcreteFileVersion::V2_0,
            None,
            None,
        )];
        let mut fragments = manifest.fragments.as_ref().clone();
        fragments.push(concurrent);
        manifest.fragments = Arc::new(fragments);

        let next = apply(
            &manifest,
            vec![drop_field(4), drop_field(3), drop_field(2), drop_field(1)],
        )
        .unwrap();

        let concurrent = next.fragments.iter().find(|f| f.id == 1).unwrap();
        assert!(concurrent.files.is_empty(), "{:?}", concurrent.files);
    }

    #[test]
    fn test_drop_field_rejects_a_missing_field() {
        let error = apply(
            &backed_manifest(),
            vec![Action::DropField(DropField {
                field: FieldRef::Committed(7),
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
                    field: FieldRef::Committed(0),
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
                    field: FieldRef::Local(0),
                }),
            ],
        )
        .unwrap();

        assert!(next.schema.field("transient").is_none());
    }
}
