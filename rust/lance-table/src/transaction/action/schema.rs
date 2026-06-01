// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Schema-domain action payloads.
//!
//! Extracted from `action.rs` to keep that module focused on the `Action`
//! trait and dispatch. Items here are re-exported from `action.rs`, so the
//! public surface of `crate::transaction::action::*` is unchanged.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::format::{DataFile, IndexMetadata, Manifest};
use deepsize::DeepSizeOf;
use lance_core::datatypes::{Field, Schema};
use lance_core::{Error, Result};

use crate::impl_dyn_action;
use crate::transaction::update_map::UpdateMap;
use crate::transaction::update_map::apply_update_map;

use super::{Action, collect_field_ids, prune_dead_data_files, remove_field_ids};

/// Payload for [`AddFields`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddFields {
    /// Fields appended to the schema. Their `id` fields are already assigned
    /// by the caller (translation allocates them from the schema's
    /// `max_field_id` high-water mark).
    pub new_fields: Vec<Field>,
    /// New per-fragment data files holding the added columns' data, keyed by
    /// the target fragment id. Each fragment must already exist in the
    /// manifest at apply time. The data files are expected to align
    /// row-for-row with their target fragment (see [`AddFields::validate`]).
    pub fragment_files: Vec<(u64, Vec<DataFile>)>,
}

impl Action for AddFields {
    impl_dyn_action!(AddFields);

    fn apply(&self, manifest: &mut Manifest, _indices: &mut Vec<IndexMetadata>) -> Result<()> {
        self.validate(manifest)?;
        manifest
            .schema
            .fields
            .extend(self.new_fields.iter().cloned());
        if !self.fragment_files.is_empty() {
            let mut new_fragments = (*manifest.fragments).clone();
            for (fragment_id, files) in &self.fragment_files {
                let slot = new_fragments
                    .iter_mut()
                    .find(|f| f.id == *fragment_id)
                    .expect("AddFields::validate confirmed every target fragment id exists");
                slot.files.extend(files.iter().cloned());
            }
            manifest.fragments = Arc::new(new_fragments);
        }
        Ok(())
    }

    /// `AddFields` *appends* new data files to existing fragments rather than
    /// replacing them, so a fragment's row count is preserved structurally —
    /// the legacy [`Operation::Merge`] carried whole replacement fragments and
    /// needed an explicit row-count-drift check (`merge_fragments_valid` in
    /// `transaction.rs`). What this method enforces instead are the
    /// preconditions for the new column data to land row-for-row on real
    /// fragments without corrupting the schema:
    ///
    /// * every fragment id in [`fragment_files`](Self::fragment_files) exists
    ///   in the manifest, and none is targeted more than once;
    /// * every [`new_fields`](Self::new_fields) id (including nested children)
    ///   is fresh — not already used by a field in the manifest schema;
    /// * each new data file references only field ids that are being added,
    ///   so it cannot silently shadow an existing column.
    ///
    /// [`apply`](Self::apply) calls this first, so applying an `AddFields`
    /// action either validates cleanly or fails without mutating the manifest.
    fn validate(&self, manifest: &Manifest) -> Result<()> {
        let existing_fragment_ids: HashSet<u64> = manifest.fragments.iter().map(|f| f.id).collect();
        let mut targeted = HashSet::with_capacity(self.fragment_files.len());
        for (fragment_id, _) in &self.fragment_files {
            if !existing_fragment_ids.contains(fragment_id) {
                return Err(Error::invalid_input(format!(
                    "AddFields targets fragment id {fragment_id} which is not present in the manifest"
                )));
            }
            if !targeted.insert(*fragment_id) {
                return Err(Error::invalid_input(format!(
                    "AddFields targets fragment id {fragment_id} more than once"
                )));
            }
        }

        let existing_field_ids: HashSet<i32> =
            manifest.schema.fields_pre_order().map(|f| f.id).collect();
        let mut new_field_ids = HashSet::new();
        collect_field_ids(&self.new_fields, &mut new_field_ids);
        for id in &new_field_ids {
            if existing_field_ids.contains(id) {
                return Err(Error::invalid_input(format!(
                    "AddFields adds a field with id {id} which already exists in the schema"
                )));
            }
        }

        for (fragment_id, files) in &self.fragment_files {
            for file in files {
                for field_id in file.fields.iter() {
                    if !new_field_ids.contains(field_id) {
                        return Err(Error::invalid_input(format!(
                            "AddFields data file {} on fragment {fragment_id} references \
                             field id {field_id}, which is not one of the fields being added",
                            file.path
                        )));
                    }
                }
            }
        }

        Ok(())
    }
}

/// Payload for [`DropFields`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct DropFields {
    /// IDs of fields to drop from the schema. Nested (child) field IDs are
    /// honored as well.
    pub field_ids: Vec<i32>,
}

impl Action for DropFields {
    impl_dyn_action!(DropFields);

    fn apply(&self, manifest: &mut Manifest, _indices: &mut Vec<IndexMetadata>) -> Result<()> {
        let drop: HashSet<i32> = self.field_ids.iter().copied().collect();
        remove_field_ids(&mut manifest.schema.fields, &drop);
        prune_dead_data_files(manifest);
        Ok(())
    }
}

/// Payload for [`UpdateSchemaMetadata`].
///
/// Carries both halves of an [`Operation::UpdateConfig`]'s schema-side edits:
/// the schema-level metadata map and the per-field metadata maps. At least one
/// half is non-empty whenever the action is emitted.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct UpdateSchemaMetadata {
    /// Changes to apply to the schema-level metadata map, or `None` when only
    /// per-field metadata changes. Honors the `replace` flag and key deletion,
    /// like config / table metadata updates.
    pub metadata_updates: Option<UpdateMap>,
    /// Changes to apply to individual fields' metadata maps, keyed by field id.
    /// Each update honors the `replace` flag and key deletion.
    pub field_metadata_updates: HashMap<i32, UpdateMap>,
}

impl Action for UpdateSchemaMetadata {
    impl_dyn_action!(UpdateSchemaMetadata);

    fn apply(&self, manifest: &mut Manifest, _indices: &mut Vec<IndexMetadata>) -> Result<()> {
        if let Some(metadata_updates) = &self.metadata_updates {
            apply_update_map(&mut manifest.schema.metadata, metadata_updates);
        }
        for (field_id, field_update) in &self.field_metadata_updates {
            let field = manifest.schema.field_by_id_mut(*field_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "UpdateSchemaMetadata targets field id {field_id}, which does not \
                     exist in the manifest"
                ))
            })?;
            apply_update_map(&mut field.metadata, field_update);
            // Keep the derived unenforced-primary-key flag in sync with the
            // updated field metadata.
            field.refresh_unenforced_primary_key_position();
        }
        Ok(())
    }
}

/// Payload for [`ChangeSchema`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct ChangeSchema {
    /// The replacement schema.
    pub schema: Schema,
}

impl Action for ChangeSchema {
    impl_dyn_action!(ChangeSchema);

    fn apply(&self, manifest: &mut Manifest, _indices: &mut Vec<IndexMetadata>) -> Result<()> {
        manifest.schema = self.schema.clone();
        // Replacing the schema can leave data files that reference only
        // now-absent fields; prune them so the manifest stays consistent,
        // exactly as the legacy `Operation::Project` path does in
        // `build_manifest`.
        prune_dead_data_files(manifest);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use lance_core::datatypes::Schema;

    use super::super::test_support::*;
    use super::super::*;
    use super::*;

    /// Build an `AddFields` payload that adds one field and one data file to
    /// the given fragment.
    fn add_one_field(field_id: i32, name: &str, fragment_id: u64) -> AddFields {
        AddFields {
            new_fields: vec![field(field_id, name)],
            fragment_files: vec![(
                fragment_id,
                vec![data_file(&format!("{name}.lance"), vec![field_id])],
            )],
        }
    }

    fn manifest_with_one_field_and_fragment() -> Manifest {
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a")]);
        manifest.fragments = Arc::new(vec![fragment_with_files(
            1,
            vec![data_file("a.lance", vec![0])],
        )]);
        manifest.max_fragment_id = Some(1);
        manifest
    }

    fn schema_metadata(
        metadata: Option<UpdateMap>,
        field_metadata: &[(i32, UpdateMap)],
    ) -> UpdateSchemaMetadata {
        UpdateSchemaMetadata {
            metadata_updates: metadata,
            field_metadata_updates: field_metadata.iter().cloned().collect(),
        }
    }

    #[test]
    fn add_fields_apply_appends_fields_and_files() {
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a")]);
        manifest.fragments = Arc::new(vec![
            fragment_with_files(1, vec![data_file("a.lance", vec![0])]),
            fragment_with_files(2, vec![data_file("a2.lance", vec![0])]),
        ]);
        manifest.max_fragment_id = Some(2);

        let action = AddFields {
            new_fields: vec![field(1, "b")],
            fragment_files: vec![
                (1, vec![data_file("b.lance", vec![1])]),
                (2, vec![data_file("b2.lance", vec![1])]),
            ],
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        let field_ids: Vec<i32> = manifest.schema.fields.iter().map(|f| f.id).collect();
        assert_eq!(field_ids, vec![0, 1]);
        assert_eq!(manifest.fragments[0].files.len(), 2);
        assert_eq!(manifest.fragments[0].files[1].path, "b.lance");
        assert_eq!(manifest.fragments[1].files[1].path, "b2.lance");
    }

    #[test]
    fn add_fields_apply_without_fragment_files_is_schema_only() {
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a")]);

        let action = AddFields {
            new_fields: vec![field(1, "b")],
            fragment_files: vec![],
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        let field_ids: Vec<i32> = manifest.schema.fields.iter().map(|f| f.id).collect();
        assert_eq!(field_ids, vec![0, 1]);
    }

    #[test]
    fn add_fields_apply_rejects_unknown_fragment_id() {
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a")]);
        manifest.fragments = Arc::new(vec![sample_fragment(1)]);

        let action = AddFields {
            new_fields: vec![field(1, "b")],
            fragment_files: vec![(99, vec![data_file("b.lance", vec![1])])],
        };
        let err = action.apply(&mut manifest, &mut vec![]).unwrap_err();
        assert!(format!("{err}").contains("99"), "got: {err}");
    }

    #[test]
    fn add_fields_validate_accepts_well_formed_action() {
        let manifest = manifest_with_one_field_and_fragment();
        add_one_field(1, "b", 1).validate(&manifest).unwrap();
    }

    #[test]
    fn add_fields_validate_accepts_schema_only_action() {
        let manifest = manifest_with_one_field_and_fragment();
        let add = AddFields {
            new_fields: vec![field(1, "b")],
            fragment_files: vec![],
        };
        add.validate(&manifest).unwrap();
    }

    #[test]
    fn add_fields_validate_rejects_unknown_fragment() {
        let manifest = manifest_with_one_field_and_fragment();
        let err = add_one_field(1, "b", 99).validate(&manifest).unwrap_err();
        assert!(format!("{err}").contains("99"), "got: {err}");
    }

    #[test]
    fn add_fields_validate_rejects_duplicate_fragment() {
        let manifest = manifest_with_one_field_and_fragment();
        let add = AddFields {
            new_fields: vec![field(1, "b")],
            fragment_files: vec![
                (1, vec![data_file("b.lance", vec![1])]),
                (1, vec![data_file("b2.lance", vec![1])]),
            ],
        };
        let err = add.validate(&manifest).unwrap_err();
        assert!(format!("{err}").contains("more than once"), "got: {err}");
    }

    #[test]
    fn add_fields_validate_rejects_colliding_field_id() {
        let manifest = manifest_with_one_field_and_fragment();
        // Field id 0 already exists in the schema.
        let err = add_one_field(0, "b", 1).validate(&manifest).unwrap_err();
        assert!(format!("{err}").contains("already exists"), "got: {err}");
    }

    #[test]
    fn add_fields_validate_rejects_colliding_nested_field_id() {
        let manifest = manifest_with_one_field_and_fragment();
        let mut parent = field(1, "s");
        // A nested child reuses the existing top-level field id 0.
        parent.children = vec![field(0, "x")];
        let add = AddFields {
            new_fields: vec![parent],
            fragment_files: vec![],
        };
        let err = add.validate(&manifest).unwrap_err();
        assert!(format!("{err}").contains("already exists"), "got: {err}");
    }

    #[test]
    fn add_fields_validate_rejects_data_file_referencing_existing_field() {
        let manifest = manifest_with_one_field_and_fragment();
        let add = AddFields {
            new_fields: vec![field(1, "b")],
            // The data file claims field id 0, an existing column, not the
            // freshly added field 1.
            fragment_files: vec![(1, vec![data_file("b.lance", vec![0])])],
        };
        let err = add.validate(&manifest).unwrap_err();
        assert!(
            format!("{err}").contains("not one of the fields being added"),
            "got: {err}"
        );
    }

    #[test]
    fn add_fields_apply_rejects_colliding_field_id() {
        // apply runs validate first, so a bad action fails without mutating.
        let mut manifest = manifest_with_one_field_and_fragment();
        let action = add_one_field(0, "b", 1);
        let err = action.apply(&mut manifest, &mut vec![]).unwrap_err();
        assert!(format!("{err}").contains("already exists"), "got: {err}");
        // Schema untouched by the failed apply.
        assert_eq!(manifest.schema.fields.len(), 1);
    }

    #[test]
    fn action_validate_dispatches_to_add_fields() {
        let manifest = manifest_with_one_field_and_fragment();
        // A well-formed AddFields passes through Action::validate.
        add_one_field(1, "b", 1).validate(&manifest).unwrap();
        // A malformed one surfaces the same AddFields error.
        let err = add_one_field(1, "b", 99).validate(&manifest).unwrap_err();
        assert!(format!("{err}").contains("99"), "got: {err}");
    }

    #[test]
    fn drop_fields_apply_removes_fields_and_dead_files() {
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a"), (1, "b")]);
        manifest.fragments = Arc::new(vec![fragment_with_files(
            1,
            vec![data_file("a.lance", vec![0]), data_file("b.lance", vec![1])],
        )]);

        let action = DropFields { field_ids: vec![1] };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        let field_ids: Vec<i32> = manifest.schema.fields.iter().map(|f| f.id).collect();
        assert_eq!(field_ids, vec![0]);
        // The file covering only the dropped field is pruned.
        assert_eq!(manifest.fragments[0].files.len(), 1);
        assert_eq!(manifest.fragments[0].files[0].path, "a.lance");
    }

    #[test]
    fn drop_fields_apply_keeps_files_with_a_live_field() {
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a"), (1, "b")]);
        // One file covers both fields; dropping one field still leaves a live
        // field, so the file is kept verbatim (dropped id still listed).
        manifest.fragments = Arc::new(vec![fragment_with_files(
            1,
            vec![data_file("ab.lance", vec![0, 1])],
        )]);

        let action = DropFields { field_ids: vec![1] };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        assert_eq!(manifest.fragments[0].files.len(), 1);
        assert_eq!(&*manifest.fragments[0].files[0].fields, &[0, 1]);
    }

    #[test]
    fn drop_fields_apply_removes_nested_children() {
        let mut manifest = empty_manifest();
        let mut parent = field(0, "s");
        parent.children = vec![field(1, "x"), field(2, "y")];
        manifest.schema = Schema {
            fields: vec![parent],
            metadata: HashMap::new(),
        };

        let action = DropFields { field_ids: vec![1] };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        let remaining: Vec<i32> = manifest.schema.fields_pre_order().map(|f| f.id).collect();
        assert_eq!(remaining, vec![0, 2]);
    }

    #[test]
    fn update_schema_metadata_apply_merges_and_deletes() {
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a")]);
        manifest.schema.metadata = HashMap::from([
            ("keep".to_string(), "1".to_string()),
            ("drop".to_string(), "2".to_string()),
        ]);

        let action = UpdateSchemaMetadata {
            metadata_updates: Some(metadata_update(
                &[("drop", None), ("new", Some("3"))],
                false,
            )),
            field_metadata_updates: HashMap::new(),
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        assert_eq!(
            manifest.schema.metadata.get("keep").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            manifest.schema.metadata.get("new").map(String::as_str),
            Some("3")
        );
        assert!(!manifest.schema.metadata.contains_key("drop"));
    }

    #[test]
    fn update_schema_metadata_apply_replaces() {
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a")]);
        manifest.schema.metadata = HashMap::from([("old".to_string(), "1".to_string())]);

        let action = UpdateSchemaMetadata {
            metadata_updates: Some(metadata_update(&[("only", Some("v"))], true)),
            field_metadata_updates: HashMap::new(),
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        assert_eq!(manifest.schema.metadata.len(), 1);
        assert_eq!(
            manifest.schema.metadata.get("only").map(String::as_str),
            Some("v")
        );
    }

    #[test]
    fn update_schema_metadata_apply_updates_field_metadata() {
        use lance_core::datatypes::LANCE_UNENFORCED_PRIMARY_KEY_POSITION;
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a"), (1, "b")]);

        let action = schema_metadata(
            None,
            &[(
                0,
                metadata_update(&[(LANCE_UNENFORCED_PRIMARY_KEY_POSITION, Some("0"))], false),
            )],
        );
        action.apply(&mut manifest, &mut vec![]).unwrap();

        let field = manifest.schema.field_by_id_mut(0).unwrap();
        assert_eq!(
            field
                .metadata
                .get(LANCE_UNENFORCED_PRIMARY_KEY_POSITION)
                .map(String::as_str),
            Some("0")
        );
        // The derived flag is refreshed from the updated field metadata.
        assert_eq!(field.unenforced_primary_key_position, Some(0));
    }

    #[test]
    fn update_schema_metadata_apply_rejects_missing_field() {
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a")]);

        let action = schema_metadata(None, &[(7, metadata_update(&[("k", Some("v"))], false))]);
        let err = action.apply(&mut manifest, &mut vec![]).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "got: {err}");
    }

    #[test]
    fn change_schema_apply_replaces_schema_wholesale() {
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a"), (1, "b")]);

        let replacement = schema_with(&[(0, "a")]);
        let action = ChangeSchema {
            schema: replacement.clone(),
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        assert_eq!(manifest.schema, replacement);
    }

    #[test]
    fn change_schema_apply_prunes_dead_data_files() {
        let mut manifest = empty_manifest();
        manifest.schema = schema_with(&[(0, "a"), (1, "b")]);
        manifest.fragments = Arc::new(vec![fragment_with_files(
            1,
            vec![data_file("a.lance", vec![0]), data_file("b.lance", vec![1])],
        )]);

        // Replace the schema with one that keeps only field 0; the file that
        // covers only the now-absent field 1 must be pruned.
        let action = ChangeSchema {
            schema: schema_with(&[(0, "a")]),
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        assert_eq!(manifest.fragments[0].files.len(), 1);
        assert_eq!(manifest.fragments[0].files[0].path, "a.lance");
    }
}
