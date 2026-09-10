// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Pre-commit validation of an operation against the manifest it applies to.
//!
//! These checks reject transactions that could not produce a coherent manifest —
//! a fragment list that disagrees with the schema, a merge that silently dropped
//! or rewrote data files — before any manifest is written.

use crate::format::{Fragment, Manifest};
use crate::io::deletion::relative_deletion_file_path;
use crate::transaction::{Operation, UpdateMode, UpdatedFragmentOffsets};
use lance_core::datatypes::{Field, Schema};
use lance_core::{Error, Result};
use lance_file::version::ConcreteFileVersion;
use std::collections::{HashMap, HashSet};

/// Validate the operation is valid for the given manifest.
pub fn validate_operation(manifest: Option<&Manifest>, operation: &Operation) -> Result<()> {
    let manifest = match (manifest, operation) {
        (
            None,
            Operation::Overwrite {
                fragments, schema, ..
            },
        ) => {
            // Validate here because we are going to return early.
            overwrite_fragments_valid(fragments)?;
            schema_fragments_valid(None, schema, fragments)?;

            return Ok(());
        }
        (None, Operation::Clone { .. }) => return Ok(()),
        (Some(manifest), _) => manifest,
        (None, _) => {
            return Err(Error::invalid_input(format!(
                "Cannot apply operation {} to non-existent dataset",
                operation.name()
            )));
        }
    };

    match operation {
        Operation::Append { fragments } => {
            // Fragments must contain all fields in the schema
            schema_fragments_valid(Some(manifest), &manifest.schema, fragments)
        }
        Operation::Project { schema, .. } => {
            schema_fragments_valid(Some(manifest), schema, manifest.fragments.as_ref())
        }
        Operation::Merge {
            fragments, schema, ..
        } => {
            merge_fragments_valid(manifest, fragments)?;
            merge_schema_valid(manifest, schema, fragments)?;
            schema_fragments_valid(Some(manifest), schema, fragments)
        }
        Operation::Overwrite {
            fragments, schema, ..
        } => {
            overwrite_fragments_valid(fragments)?;
            // Pass None for manifest because Overwrite replaces all fragments.
            // The old manifest's storage format is irrelevant for validating
            // the new fragments (e.g., LEGACY→STABLE transitions).
            schema_fragments_valid(None, schema, fragments)
        }
        Operation::Update {
            updated_fragments,
            new_fragments,
            updated_fragment_offsets,
            update_mode,
            ..
        } => {
            schema_fragments_valid(Some(manifest), &manifest.schema, updated_fragments)?;
            schema_fragments_valid(Some(manifest), &manifest.schema, new_fragments)?;
            // Key-presence check only applies to RewriteColumns: that is the only
            // mode where build_manifest stamps version metadata using off_map keys,
            // so a stray key can corrupt an unrelated fragment's metadata.
            // Other modes (e.g. rewrite_rows) may supply offsets for fragments
            // outside updated_fragments for their own purposes.
            if matches!(update_mode, Some(UpdateMode::RewriteColumns))
                && let Some(UpdatedFragmentOffsets(off_map)) = updated_fragment_offsets
            {
                let updated_ids: HashSet<u64> = updated_fragments.iter().map(|f| f.id).collect();
                for &frag_id in off_map.keys() {
                    if !updated_ids.contains(&frag_id) {
                        return Err(Error::invalid_input(format!(
                            "updatedFragmentOffsets key {} is not in updated_fragments; \
                             offsets must reference only fragments being rewritten",
                            frag_id
                        )));
                    }
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

// An overwrite's fragments are newly written, so they are given fresh ids at
// commit time. A deletion file cannot come along for that ride: its path embeds
// the fragment id, so renumbering the fragment would orphan the deletion vector
// and silently resurrect deleted rows.
fn overwrite_fragments_valid(fragments: &[Fragment]) -> Result<()> {
    for fragment in fragments {
        if let Some(deletion_file) = &fragment.deletion_file {
            return Err(Error::invalid_input(format!(
                "Overwrite fragments must be newly written, but fragment {} carries \
                 deletion file {}. Use Delete to commit deletions against existing \
                 fragments, or Merge to change their schema.",
                fragment.id,
                relative_deletion_file_path(fragment.id, deletion_file)
            )));
        }
    }
    Ok(())
}

fn schema_fragments_valid(
    manifest: Option<&Manifest>,
    schema: &Schema,
    fragments: &[Fragment],
) -> Result<()> {
    if let Some(manifest) = manifest {
        return match manifest.data_storage_format.lance_file_format() {
            ConcreteFileVersion::V1 => schema_fragments_legacy_valid(schema, fragments),
            ConcreteFileVersion::V2_0
            | ConcreteFileVersion::V2_1
            | ConcreteFileVersion::V2_2
            | ConcreteFileVersion::V2_3 => schema_fragments_modern_valid(schema, fragments),
        };
    }
    schema_fragments_modern_valid(schema, fragments)
}

pub fn schema_fragments_modern_valid(_schema: &Schema, fragments: &[Fragment]) -> Result<()> {
    // validate that each data file at least contains one field.
    for fragment in fragments {
        for data_file in &fragment.files {
            if data_file.fields.iter().len() == 0 {
                return Err(Error::invalid_input(format!(
                    "Datafile {} does not contain any fields",
                    data_file.path
                )));
            }
        }
    }
    Ok(())
}

/// Check that each fragment contains all fields in the schema.
/// It is not required that the schema contains all fields in the fragment.
/// There may be masked fields.
pub fn schema_fragments_legacy_valid(schema: &Schema, fragments: &[Fragment]) -> Result<()> {
    // TODO: add additional validation. Consider consolidating with various
    // validate() methods in the codebase.
    for fragment in fragments {
        for field in schema.fields_pre_order() {
            if !fragment
                .files
                .iter()
                .flat_map(|f| f.fields.iter())
                .any(|f_id| f_id == &field.id)
            {
                return Err(Error::invalid_input(format!(
                    "Fragment {} does not contain field {:?}",
                    fragment.id, field
                )));
            }
        }
    }
    Ok(())
}

/// Returns true if Operation::Merge rewrote this fragment's column data files (Fragment::files
/// changed versus the previous manifest). Used to bump last_updated_at_version_meta only when
/// new column values were materialized to disk.
///
/// Deletion file changes alone are not treated as rewrites: tombstones remove rows but
/// survivors did not receive new column bytes; stamping last_updated for those rows would be
/// incorrect for CDF.
#[inline]
pub(super) fn merge_fragment_physically_rewritten(prev: &Fragment, merged: &Fragment) -> bool {
    debug_assert_eq!(prev.id, merged.id);
    if prev.files.len() != merged.files.len() {
        return true;
    }
    // Compare identity fields only. file_size_bytes is an AtomicU64 cache that
    // concurrent scans can populate in place on the manifest's DataFile, so it
    // must not be part of the rewrite check.
    prev.files.iter().zip(merged.files.iter()).any(|(p, m)| {
        p.path != m.path
            || p.fields != m.fields
            || p.column_indices != m.column_indices
            || p.file_major_version != m.file_major_version
            || p.file_minor_version != m.file_minor_version
            || p.base_id != m.base_id
    })
}

/// Validate that Merge operations preserve all original fragments.
/// Merge operations should only add columns or rows, not reduce fragments.
/// This ensures fragments correspond at one-to-one with the original fragment list.
fn merge_fragments_valid(manifest: &Manifest, new_fragments: &[Fragment]) -> Result<()> {
    let original_fragments = manifest.fragments.as_ref();

    // Additional validation: ensure we're not accidentally reducing the fragment count
    if new_fragments.len() < original_fragments.len() {
        return Err(Error::invalid_input(format!(
            "Merge operation reduced fragment count from {} to {}. \
             Merge operations should only add columns, not reduce fragments.",
            original_fragments.len(),
            new_fragments.len()
        )));
    }

    // Collect new fragment IDs
    let new_fragment_map: HashMap<u64, &Fragment> =
        new_fragments.iter().map(|f| (f.id, f)).collect();

    // Check that all original fragments are preserved in the new fragments list
    // Validate that each original fragment's metadata is preserved
    let mut missing_fragments: Vec<u64> = Vec::new();
    for original_fragment in original_fragments {
        if let Some(new_fragment) = new_fragment_map.get(&original_fragment.id) {
            // Validate physical_rows (row count) hasn't changed
            if original_fragment.physical_rows != new_fragment.physical_rows {
                return Err(Error::invalid_input(format!(
                    "Merge operation changed row count for fragment {}. \
                     Original: {:?}, New: {:?}. \
                     Merge operations should preserve fragment row counts and only add new columns.",
                    original_fragment.id,
                    original_fragment.physical_rows,
                    new_fragment.physical_rows
                )));
            }
        } else {
            missing_fragments.push(original_fragment.id);
        }
    }

    if !missing_fragments.is_empty() {
        return Err(Error::invalid_input(format!(
            "Merge operation is missing original fragments: {:?}. \
             Merge operations should preserve all original fragments and only add new columns. \
             Expected fragments: {:?}, but got: {:?}",
            missing_fragments,
            original_fragments.iter().map(|f| f.id).collect::<Vec<_>>(),
            new_fragment_map.keys().copied().collect::<Vec<_>>()
        )));
    }

    Ok(())
}

/// Validate that a Merge schema preserves the dataset's field id bindings.
///
/// Readers resolve columns by field id (name -> schema id -> DataFile::fields
/// position), so renumbered ids silently rebind live columns to other columns'
/// bytes. Shared ids must keep their field path. Their logical type,
/// nullability, storage encoding, and dictionary may change only when every
/// existing base or overlay file carrying the id is replaced and every
/// proposed fragment materializes the id in a base data file. New ids must
/// exceed the manifest's max so a dropped field's id is never reused. An
/// existing path may move to a fresh id only when every proposed fragment
/// materializes that id in a base data file (the `alter_columns` cast path).
/// Omitting a field (dropping it) and updating field metadata remain legal.
fn merge_schema_valid(
    manifest: &Manifest,
    new_schema: &Schema,
    fragments: &[Fragment],
) -> Result<()> {
    let prior_schema = &manifest.schema;
    let new_fragment_map: HashMap<u64, &Fragment> = fragments
        .iter()
        .map(|fragment| (fragment.id, fragment))
        .collect();

    // Remap and semantic errors first: a renumbered schema usually violates
    // both the shared-id and new-id clauses.
    for field in new_schema.fields_pre_order() {
        let Some(prior_field) = prior_schema.field_by_id(field.id) else {
            continue;
        };
        let prior_path = prior_schema.field_path(field.id)?;
        let new_path = new_schema.field_path(field.id)?;
        if prior_path != new_path {
            return Err(Error::invalid_input(format!(
                "Merge operation remaps field id {} from \"{}\" to \"{}\". \
                 Merge must preserve the dataset's field ids: derive the new schema \
                 from the dataset's current schema instead of renumbering fields.",
                field.id, prior_path, new_path
            )));
        }
        if let Some(changes) = shared_field_binding_changes(prior_field, field)
            && !is_field_binding_fully_rewritten(manifest, &new_fragment_map, field.id)
        {
            return Err(Error::invalid_input(format!(
                "Merge operation changes field id {} (\"{}\") without rewriting it in \
                 every existing fragment: {}. Merge must preserve each existing field's \
                 logical type, nullability, storage encoding, and dictionary unless all \
                 existing base and overlay files carrying that field are replaced.",
                field.id, new_path, changes
            )));
        }
    }

    let max_field_id = manifest.max_field_id();
    for field in new_schema.fields_pre_order() {
        if prior_schema.field_by_id(field.id).is_none() && field.id <= max_field_id {
            let next_id_msg = match max_field_id.checked_add(1) {
                Some(next_id) => format!("New fields must use ids of at least {}.", next_id),
                None => {
                    "No further field id can be allocated because ids are exhausted.".to_string()
                }
            };
            return Err(Error::invalid_input(format!(
                "Merge operation assigns id {} to new field \"{}\", but ids up to {} are \
                 already used by current or dropped fields. {}",
                field.id,
                new_schema.field_path(field.id)?,
                max_field_id,
                next_id_msg
            )));
        }
    }

    let mut prior_paths = HashMap::with_capacity(prior_schema.fields_pre_order().count());
    for field in prior_schema.fields_pre_order() {
        prior_paths.insert(prior_schema.field_path(field.id)?, field);
    }
    for field in new_schema.fields_pre_order() {
        if prior_schema.field_by_id(field.id).is_some() {
            continue;
        }
        let new_path = new_schema.field_path(field.id)?;
        let Some(prior_field) = prior_paths.get(&new_path) else {
            continue;
        };
        let materialized = fragments.iter().all(|fragment| {
            fragment
                .files
                .iter()
                .any(|file| file.fields.contains(&field.id))
        });
        if !materialized {
            return Err(Error::invalid_input(format!(
                "Merge operation remaps existing field \"{}\" from id {} to id {} without \
                 rewriting its data. Every proposed fragment must materialize the new field \
                 id in a base data file.",
                new_path, prior_field.id, field.id
            )));
        }
    }

    Ok(())
}

fn is_field_binding_fully_rewritten(
    manifest: &Manifest,
    new_fragment_map: &HashMap<u64, &Fragment>,
    field_id: i32,
) -> bool {
    manifest.fragments.iter().all(|prior_fragment| {
        let Some(new_fragment) = new_fragment_map.get(&prior_fragment.id) else {
            return false;
        };

        let is_materialized = new_fragment
            .files
            .iter()
            .any(|file| file.fields.contains(&field_id));
        if !is_materialized {
            return false;
        }

        prior_fragment
            .referenced_lance_files()
            .filter(|file| file.fields.contains(&field_id))
            .all(|prior_file| {
                !new_fragment.referenced_lance_files().any(|new_file| {
                    new_file.fields.contains(&field_id)
                        && new_file.base_id == prior_file.base_id
                        && new_file.path == prior_file.path
                })
            })
    })
}

fn shared_field_binding_changes(prior: &Field, new: &Field) -> Option<String> {
    let mut changes = Vec::with_capacity(4);
    if prior.logical_type != new.logical_type {
        changes.push(format!(
            "logical type {} -> {}",
            prior.logical_type, new.logical_type
        ));
    }
    if prior.nullable != new.nullable {
        changes.push(format!("nullable {} -> {}", prior.nullable, new.nullable));
    }
    if prior.encoding != new.encoding {
        changes.push(format!(
            "storage encoding {:?} -> {:?}",
            prior.encoding, new.encoding
        ));
    }
    if prior.dictionary != new.dictionary {
        changes.push("dictionary".to_string());
    }
    if changes.is_empty() {
        None
    } else {
        Some(changes.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::overlay::{DataOverlayFile, OverlayCoverage};
    use crate::format::{DataFile, DataStorageFormat};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::{Field as LanceCoreField, LogicalType, Schema as LanceSchema};
    use roaring::RoaringBitmap;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_merge_fragments_valid() {
        // Create a simple schema for testing
        let schema = ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("name", DataType::Utf8, false),
        ]);

        // Create original fragments
        let original_fragments = vec![Fragment::new(1), Fragment::new(2), Fragment::new(3)];

        // Create a manifest with original fragments
        let manifest = Manifest::new(
            LanceSchema::try_from(&schema).unwrap(),
            Arc::new(original_fragments),
            DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );

        // Test 1: Empty fragments should fail
        let empty_fragments = vec![];
        let result = merge_fragments_valid(&manifest, &empty_fragments);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("reduced fragment count")
        );

        // Test 2: Missing original fragments should fail
        let missing_fragments = vec![
            Fragment::new(1),
            Fragment::new(2),
            // Fragment 3 is missing
            Fragment::new(4), // New fragment
        ];
        let result = merge_fragments_valid(&manifest, &missing_fragments);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("missing original fragments")
        );

        // Test 3: Reduced fragment count should fail
        let reduced_fragments = vec![
            Fragment::new(1),
            Fragment::new(2),
            // Fragment 3 is missing, no new fragments added
        ];
        let result = merge_fragments_valid(&manifest, &reduced_fragments);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("reduced fragment count")
        );

        // Test 4: Valid merge with all original fragments plus new ones should succeed
        let valid_fragments = vec![
            Fragment::new(1),
            Fragment::new(2),
            Fragment::new(3),
            Fragment::new(4), // New fragment
            Fragment::new(5), // Another new fragment
        ];
        let result = merge_fragments_valid(&manifest, &valid_fragments);
        assert!(result.is_ok());

        // Test 5: Same fragments (no new ones) should succeed
        let same_fragments = vec![Fragment::new(1), Fragment::new(2), Fragment::new(3)];
        let result = merge_fragments_valid(&manifest, &same_fragments);
        assert!(result.is_ok());
    }

    fn one_field_schema() -> LanceSchema {
        LanceSchema::try_from(&ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            true,
        )]))
        .unwrap()
    }

    fn fragment_with_file_fields(id: u64, path: &str, fields: Vec<i32>) -> Fragment {
        let mut fragment = Fragment::new(id);
        fragment
            .files
            .push(DataFile::new_legacy_from_fields(path, fields, None));
        fragment
    }

    fn manifest_with_file_fields(schema: LanceSchema, fields: Vec<i32>) -> Manifest {
        Manifest::new(
            schema,
            Arc::new(vec![fragment_with_file_fields(0, "f.lance", fields)]),
            DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        )
    }

    #[rstest::rstest]
    #[case::logical_type(DataType::Float32, true)]
    #[case::nullability(DataType::Int32, false)]
    #[test]
    fn test_merge_shared_id_change_requires_full_rewrite(
        #[case] data_type: DataType,
        #[case] nullable: bool,
    ) {
        let schema = one_field_schema();
        let prior_fragments = vec![
            fragment_with_file_fields(0, "old-0.lance", vec![0]),
            fragment_with_file_fields(1, "old-1.lance", vec![0]),
        ];
        let manifest = Manifest::new(
            schema.clone(),
            Arc::new(prior_fragments.clone()),
            DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );
        let mut new_schema = schema;
        new_schema.fields[0].logical_type = LogicalType::try_from(&data_type).unwrap();
        new_schema.fields[0].nullable = nullable;

        let rewritten_fragments = vec![
            fragment_with_file_fields(0, "new-0.lance", vec![0]),
            fragment_with_file_fields(1, "new-1.lance", vec![0]),
        ];
        merge_schema_valid(&manifest, &new_schema, &rewritten_fragments).unwrap();

        let partially_rewritten = vec![rewritten_fragments[0].clone(), prior_fragments[1].clone()];
        let err = merge_schema_valid(&manifest, &new_schema, &partially_rewritten).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "got {:?}", err);
        assert!(
            err.to_string()
                .contains("without rewriting it in every existing fragment"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_merge_shared_id_change_rejects_retained_overlay() {
        let schema = one_field_schema();
        let mut prior_fragment = fragment_with_file_fields(0, "old.lance", vec![0]);
        prior_fragment.overlays.push(DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("old-overlay.lance", vec![0], None),
            coverage: OverlayCoverage::Shared(Arc::new(RoaringBitmap::from_iter([0_u32]))),
            committed_version: 1,
        });
        let manifest = Manifest::new(
            schema.clone(),
            Arc::new(vec![prior_fragment.clone()]),
            DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );
        let mut new_schema = schema;
        new_schema.fields[0].nullable = false;

        let mut rewritten = fragment_with_file_fields(0, "new.lance", vec![0]);
        rewritten.overlays = prior_fragment.overlays.clone();
        let err = merge_schema_valid(&manifest, &new_schema, &[rewritten]).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "got {:?}", err);
        assert!(
            err.to_string()
                .contains("without rewriting it in every existing fragment"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_merge_allows_rewritten_fresh_field_id() {
        let schema = one_field_schema();
        let manifest = manifest_with_file_fields(schema.clone(), vec![0]);
        let mut rewritten_schema = schema;
        rewritten_schema.fields[0].id = 1;
        let mut rewritten = manifest.fragments[0].clone();
        rewritten.files[0] = DataFile::new_legacy_from_fields("rewritten.lance", vec![1], None);
        merge_schema_valid(&manifest, &rewritten_schema, &[rewritten]).unwrap();
    }

    #[test]
    fn test_merge_rejects_max_field_id_overflow() {
        let schema = one_field_schema();
        let manifest = manifest_with_file_fields(schema.clone(), vec![0, i32::MAX]);
        assert_eq!(manifest.max_field_id(), i32::MAX);

        let mut new_schema = schema;
        let mut extra =
            LanceCoreField::try_from(&ArrowField::new("b", DataType::Int32, true)).unwrap();
        extra.id = 1;
        new_schema.fields.push(extra);

        let err = merge_schema_valid(&manifest, &new_schema, &manifest.fragments).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "got {:?}", err);
        let message = err.to_string();
        assert!(
            message.contains("assigns id 1 to new field \"b\"") && message.contains("exhausted"),
            "unexpected error: {}",
            message
        );
    }

    /// Regression test for https://github.com/lance-format/lance/issues/6417
    ///
    /// When overwriting a LEGACY dataset with STABLE-format fragments, the
    /// validation should not use the old manifest's format. STABLE fragments
    /// omit struct parent fields, which the strict legacy check rejects.
    #[test]
    fn test_overwrite_legacy_to_stable_with_struct_fields() {
        use arrow_schema::Fields;

        // Schema: id (field 0), name (field 1), address (field 2, struct parent),
        //   city (field 3), country (field 4)
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("name", DataType::Utf8, false),
            ArrowField::new(
                "address",
                DataType::Struct(Fields::from(vec![
                    ArrowField::new("city", DataType::Utf8, false),
                    ArrowField::new("country", DataType::Utf8, false),
                ])),
                false,
            ),
        ]);
        let schema = LanceSchema::try_from(&arrow_schema).unwrap();

        // Old manifest is LEGACY format
        let legacy_manifest = Manifest::new(
            schema.clone(),
            Arc::new(vec![Fragment::new(0)]),
            DataStorageFormat::new(ConcreteFileVersion::V1),
            HashMap::new(),
        );

        // New fragments in STABLE format omit struct parent field (id=2),
        // only including leaf fields: id=0, name=1, city=3, country=4
        let stable_fragment = Fragment {
            id: 0,
            files: vec![DataFile::new(
                "data.lance",
                vec![0, 1, 3, 4], // no field 2 (struct parent)
                vec![0, 1, 2, 3],
                ConcreteFileVersion::V1,
                None,
                None,
            )],
            physical_rows: Some(10),
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };

        let operation = Operation::Overwrite {
            fragments: vec![stable_fragment],
            schema,
            config_upsert_values: None,
            initial_bases: None,
        };

        // This should succeed — the old manifest's LEGACY format should not
        // cause strict validation of the new STABLE fragments.
        validate_operation(Some(&legacy_manifest), &operation).unwrap();
    }
}
