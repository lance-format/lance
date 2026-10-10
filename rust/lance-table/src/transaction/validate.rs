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

type DataFileIdentity = (Option<u32>, String);

/// Resolve an Arrow-derived operation against the dataset it was read from.
///
/// Overwrite assigns new IDs to every field. Merge matches fields by name and
/// type, not positional Arrow IDs.
/// Project may use explicit IDs for renames; its conversion must leave missing
/// IDs unassigned. New file mappings follow the resolved schema; retained files
/// are unchanged. Call this at the conversion boundary, before committing the
/// resulting Lance operation. Commit-time allocation still handles retries.
///
/// ```no_run
/// # use lance_table::{format::Manifest, transaction::{Operation, resolve_arrow_field_ids}};
/// # fn convert(manifest: &Manifest, operation: &mut Operation) -> lance_core::Result<()> {
/// resolve_arrow_field_ids(Some(manifest), operation)?;
/// # Ok(())
/// # }
/// ```
pub fn resolve_arrow_field_ids(
    manifest: Option<&Manifest>,
    operation: &mut Operation,
) -> Result<()> {
    if manifest.is_some_and(|manifest| !manifest.uses_non_reusable_field_ids()) {
        match operation {
            Operation::Overwrite { schema, .. }
            | Operation::Project { schema, .. }
            | Operation::Merge { schema, .. } => {
                schema.try_set_field_id(None)?;
                schema.validate()?;
                schema.verify_primary_key()?;
            }
            _ => {}
        }
        return Ok(());
    }

    match operation {
        Operation::Overwrite {
            schema, fragments, ..
        } => {
            let field_id_remap = canonicalize_schema(manifest, schema, None, true)?;
            resolve_fragment_field_ids(fragments, &field_id_remap, &HashSet::new(), schema)?;
        }
        Operation::Project { schema, .. } => {
            let Some(manifest) = manifest else {
                return Ok(());
            };
            canonicalize_raw_project_schema(manifest, schema)?;
        }
        Operation::Merge {
            schema, fragments, ..
        } => {
            let Some(manifest) = manifest else {
                return Ok(());
            };
            let retained_files = manifest
                .fragments
                .iter()
                .flat_map(|fragment| fragment.referenced_lance_files())
                .map(|file| (file.base_id, file.path.clone()))
                .collect();
            let field_id_remap =
                canonicalize_schema(Some(manifest), schema, Some(&manifest.schema), true)?;
            resolve_fragment_field_ids(fragments, &field_id_remap, &retained_files, schema)?;
        }
        _ => {}
    }
    Ok(())
}

/// Assign IDs and update new file mappings before committing a Lance operation.
///
/// `manifest` is the latest version. Overwrite assigns new IDs to every field.
/// For Merge, `read_schema` identifies existing fields when the transaction was
/// prepared; `None` uses the manifest's schema. New fields are allocated above
/// the latest high-water mark, without binding their provisional IDs to fields introduced
/// by a concurrent commit. Arrow inputs must first use [`resolve_arrow_field_ids`].
///
/// ```no_run
/// # use lance_table::{format::Manifest, transaction::{Operation, canonicalize_non_reusable_field_ids}};
/// # fn commit(read: &Manifest, latest: &Manifest, operation: &mut Operation) -> lance_core::Result<()> {
/// canonicalize_non_reusable_field_ids(Some(latest), operation, Some(&read.schema))?;
/// # Ok(())
/// # }
/// ```
pub fn canonicalize_non_reusable_field_ids(
    manifest: Option<&Manifest>,
    operation: &mut Operation,
    read_schema: Option<&Schema>,
) -> Result<()> {
    if manifest.is_some_and(|manifest| !manifest.uses_non_reusable_field_ids()) {
        return Ok(());
    }
    match operation {
        Operation::Overwrite {
            schema, fragments, ..
        } => {
            let remap = canonicalize_schema(manifest, schema, None, false)?;
            remap_fragment_field_ids(fragments, &remap, &HashSet::new());
        }
        Operation::Merge {
            schema, fragments, ..
        } => {
            if let Some(manifest) = manifest {
                let retained_files = manifest
                    .fragments
                    .iter()
                    .flat_map(|fragment| fragment.referenced_lance_files())
                    .map(|file| (file.base_id, file.path.clone()))
                    .collect();
                canonicalize_merge_replacements(
                    manifest,
                    schema,
                    fragments,
                    &retained_files,
                    read_schema.unwrap_or(&manifest.schema),
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn canonicalize_merge_replacements(
    manifest: &Manifest,
    schema: &mut Schema,
    fragments: &mut [Fragment],
    retained_files: &HashSet<DataFileIdentity>,
    read_schema: &Schema,
) -> Result<()> {
    let mut replaced_field_ids = HashSet::new();
    for fragment in fragments.iter() {
        let retained_field_ids = fragment
            .referenced_lance_files()
            .filter(|file| retained_files.contains(&(file.base_id, file.path.clone())))
            .flat_map(|file| file.fields.iter().copied())
            .collect::<HashSet<_>>();
        replaced_field_ids.extend(
            fragment
                .referenced_lance_files()
                .filter(|file| !retained_files.contains(&(file.base_id, file.path.clone())))
                .flat_map(|file| file.fields.iter().copied())
                .filter(|field_id| retained_field_ids.contains(field_id)),
        );
    }
    let original = schema.clone();
    for field in &mut schema.fields {
        clear_replaced_and_new_field_ids(field, read_schema, &replaced_field_ids);
    }
    schema.try_set_field_id(Some(manifest.max_field_id()))?;
    schema.validate()?;
    schema.verify_primary_key()?;

    let field_id_remap = original
        .fields_pre_order()
        .zip(schema.fields_pre_order())
        .filter(|(original, _)| original.id >= 0)
        .map(|(original, canonical)| (original.id, canonical.id))
        .collect();
    remap_fragment_field_ids(fragments, &field_id_remap, retained_files);
    Ok(())
}

fn clear_replaced_and_new_field_ids(
    field: &mut Field,
    base_schema: &Schema,
    replaced_field_ids: &HashSet<i32>,
) {
    if base_schema.field_by_id(field.id).is_none() || replaced_field_ids.contains(&field.id) {
        clear_field_ids(field);
        return;
    }
    for child in &mut field.children {
        clear_replaced_and_new_field_ids(child, base_schema, replaced_field_ids);
    }
}

fn canonicalize_raw_project_schema(manifest: &Manifest, schema: &mut Schema) -> Result<()> {
    let mut unmatched_fields = Vec::new();
    for field in &mut schema.fields {
        if !canonicalize_field(field, -1, &manifest.schema, None, Some(&manifest.schema)) {
            unmatched_fields.push(field.name.clone());
        }
    }
    if !unmatched_fields.is_empty() {
        return Err(Error::invalid_input(format!(
            "Raw Arrow Project fields [{}] do not match existing field identities; Project cannot allocate new identities because it writes no data",
            unmatched_fields.join(", ")
        )));
    }
    schema.validate()?;
    schema.verify_primary_key()?;
    Ok(())
}

fn canonicalize_schema(
    manifest: Option<&Manifest>,
    schema: &mut Schema,
    matching_schema: Option<&Schema>,
    remap_raw_source_ids: bool,
) -> Result<HashMap<i32, i32>> {
    let mut original = schema.clone();
    if remap_raw_source_ids {
        original.try_set_field_id(None)?;
    }

    let max_existing_id = manifest.map(Manifest::max_field_id);
    if let Some(matching_schema) = matching_schema {
        for field in &mut schema.fields {
            canonicalize_field(field, -1, matching_schema, None, None);
        }
        schema.try_set_field_id(max_existing_id)?;
    } else {
        schema.try_reassign_field_ids(max_existing_id)?;
    }
    schema.validate()?;
    schema.verify_primary_key()?;

    Ok(original
        .fields_pre_order()
        .zip(schema.fields_pre_order())
        .filter(|(original, _)| original.id >= 0)
        .map(|(original, canonical)| (original.id, canonical.id))
        .collect())
}

fn canonicalize_field(
    field: &mut Field,
    parent_id: i32,
    base_schema: &Schema,
    base_parent: Option<&Field>,
    identity_schema: Option<&Schema>,
) -> bool {
    let same_name = match base_parent {
        Some(parent) => parent.children.iter().find(|base| base.name == field.name),
        None => base_schema
            .fields
            .iter()
            .find(|base| base.name == field.name),
    };
    let allow_id_binding =
        identity_schema.is_some_and(|schema| schema.field_by_id(field.id).is_some());
    let by_id = (allow_id_binding && field.id >= 0)
        .then(|| base_schema.field_by_id(field.id))
        .flatten()
        .filter(|base| base.parent_id == parent_id);
    let base_field = if allow_id_binding && field.id >= 0 {
        by_id
            .filter(|base| base.logical_type == field.logical_type)
            .or_else(|| same_name.filter(|base| base.logical_type == field.logical_type))
    } else {
        same_name.filter(|base| base.logical_type == field.logical_type)
    };

    let Some(base_field) = base_field else {
        clear_field_ids(field);
        return false;
    };

    field.id = base_field.id;
    field.parent_id = parent_id;
    let mut all_children_match = true;
    for child in &mut field.children {
        if !canonicalize_field(
            child,
            field.id,
            base_schema,
            Some(base_field),
            identity_schema,
        ) {
            all_children_match = false;
        }
    }
    all_children_match
}

fn clear_field_ids(field: &mut Field) {
    field.id = -1;
    field.parent_id = -1;
    for child in &mut field.children {
        clear_field_ids(child);
    }
}

fn resolve_fragment_field_ids(
    fragments: &mut [Fragment],
    field_id_remap: &HashMap<i32, i32>,
    retained_files: &HashSet<DataFileIdentity>,
    schema: &Schema,
) -> Result<()> {
    let canonical_ids = schema.field_ids();
    for fragment in fragments {
        // Raw Arrow overwrite fragments may have been written either by a
        // standalone writer using the source schema IDs or by a
        // dataset-aware writer using canonical IDs. Resolve that namespace
        // once for the whole fragment so split files cannot disagree. If
        // both interpretations are possible and produce different
        // identities then there is no safe mapping without provenance.
        let fragment_field_ids = fragment
            .referenced_lance_files()
            .filter(|file| !retained_files.contains(&(file.base_id, file.path.clone())))
            .flat_map(|file| file.fields.iter().copied())
            .filter(|field_id| *field_id >= 0)
            .collect::<HashSet<_>>();
        let canonical_source = fragment_field_ids
            .iter()
            .all(|field_id| canonical_ids.contains(field_id));
        let raw_source = fragment_field_ids
            .iter()
            .all(|field_id| field_id_remap.contains_key(field_id));
        let raw_changes_identity = fragment_field_ids
            .iter()
            .any(|field_id| field_id_remap.get(field_id) != Some(field_id));

        match (canonical_source, raw_source, raw_changes_identity) {
            (true, true, true) => {
                return Err(Error::invalid_input(format!(
                    "Fragment {} has ambiguous raw Arrow field IDs; its file mappings can be interpreted as either source or canonical identities",
                    fragment.id
                )));
            }
            (true, _, _) => continue,
            (false, true, _) => {}
            (false, false, _) => {
                return Err(Error::invalid_input(format!(
                    "Fragment {} field IDs do not match either the raw Arrow source schema or the canonical replacement schema",
                    fragment.id
                )));
            }
        }
        remap_fragment_field_ids(
            std::slice::from_mut(fragment),
            field_id_remap,
            retained_files,
        );
    }
    Ok(())
}

fn remap_fragment_field_ids(
    fragments: &mut [Fragment],
    field_id_remap: &HashMap<i32, i32>,
    retained_files: &HashSet<DataFileIdentity>,
) {
    for fragment in fragments {
        for file in fragment.referenced_lance_files_mut() {
            if retained_files.contains(&(file.base_id, file.path.clone())) {
                continue;
            }
            for field_id in std::sync::Arc::make_mut(&mut file.fields) {
                if let Some(canonical_id) = field_id_remap.get(field_id) {
                    *field_id = *canonical_id;
                }
            }
        }
    }
}

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

    let result = match operation {
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
        // The offsets stamp version metadata by fragment id, so a key outside
        // the replacements would corrupt an unrelated fragment's metadata.
        Operation::DataReplacement {
            replacements,
            replaced_offsets: Some(UpdatedFragmentOffsets(off_map)),
            ..
        } => {
            let replaced_ids: HashSet<u64> = replacements.iter().map(|r| r.0).collect();
            for &frag_id in off_map.keys() {
                if !replaced_ids.contains(&frag_id) {
                    return Err(Error::invalid_input(format!(
                        "replacedOffsets key {frag_id} is not in replacements; \
                         offsets must reference only fragments being replaced"
                    )));
                }
            }
            Ok(())
        }
        _ => Ok(()),
    };
    result?;
    validate_non_reusable_field_id_operation(manifest, operation)
}

/// Validate non-reusable-field-ID invariants that are independent of one operation.
fn validate_non_reusable_field_id_manifest(manifest: &Manifest) -> Result<()> {
    let Some(max_allocated_field_id) = manifest.max_allocated_field_id else {
        return Ok(());
    };
    let max_referenced_field_id = manifest.max_referenced_field_id();
    if max_allocated_field_id < max_referenced_field_id {
        return Err(Error::invalid_input(format!(
            "Non-reusable field-ID high-water mark {} is below referenced field ID {}",
            max_allocated_field_id, max_referenced_field_id
        )));
    }
    Ok(())
}

/// Validate newly referenced field IDs before advancing the high-water mark.
///
/// A data-only operation must not manufacture allocator state by putting an
/// otherwise unknown ID in a data-file or overlay mapping. IDs above the parent
/// high-water mark are legal only when the canonical successor schema contains
/// that newly allocated identity. Overwrite has no retained physical state, so
/// every non-negative reference must belong to its replacement schema.
pub fn validate_non_reusable_field_id_transition(
    parent: &Manifest,
    successor: &Manifest,
    operation: &Operation,
) -> Result<()> {
    let Some(parent_max_field_id) = parent.max_allocated_field_id else {
        return Ok(());
    };
    let Some(successor_max_field_id) = successor.max_allocated_field_id else {
        return Err(Error::invalid_input(
            "Non-reusable field-ID activation marker is missing from the successor manifest",
        ));
    };
    if successor_max_field_id < parent_max_field_id {
        return Err(Error::invalid_input(format!(
            "Non-reusable field-ID high-water mark decreases from {parent_max_field_id} to {successor_max_field_id}"
        )));
    }
    let successor_schema_ids = successor
        .schema
        .fields_pre_order()
        .map(|field| field.id)
        .collect::<HashSet<_>>();

    if !matches!(operation, Operation::Restore { .. }) {
        validate_new_field_ids(
            parent,
            successor.schema.fields_pre_order().filter(|field| {
                matches!(operation, Operation::Overwrite { .. })
                    || parent.schema.field_by_id(field.id).is_none()
            }),
        )?;
    }

    for field_id in successor
        .fragments
        .iter()
        .flat_map(|fragment| fragment.referenced_lance_files())
        .flat_map(|file| file.fields.iter())
        .copied()
        .filter(|field_id| *field_id >= 0)
    {
        if matches!(operation, Operation::Overwrite { .. })
            && !successor_schema_ids.contains(&field_id)
        {
            return Err(Error::invalid_input(format!(
                "Overwrite references field ID {field_id}, which is not in its replacement schema"
            )));
        }
        if field_id > parent_max_field_id && !successor_schema_ids.contains(&field_id) {
            return Err(Error::invalid_input(format!(
                "Data file or overlay references new field ID {field_id}, but the canonical successor schema does not contain that identity"
            )));
        }
    }
    Ok(())
}

fn validate_new_field_ids<'a>(
    manifest: &Manifest,
    new_fields: impl Iterator<Item = &'a Field>,
) -> Result<()> {
    let max_allocated_field_id = manifest.max_field_id();
    for field in new_fields {
        if field.id <= max_allocated_field_id {
            return Err(Error::invalid_input(format!(
                "New field '{}' has ID {}, but non-reusable field IDs must be greater than the high-water mark {}",
                field.name, field.id, max_allocated_field_id
            )));
        }
    }
    Ok(())
}

fn validate_non_reusable_field_id_operation(
    manifest: &Manifest,
    operation: &Operation,
) -> Result<()> {
    if !manifest.uses_non_reusable_field_ids() {
        return Ok(());
    }
    validate_non_reusable_field_id_manifest(manifest)?;

    let (Operation::Overwrite { schema, .. }
    | Operation::Merge { schema, .. }
    | Operation::Project { schema, .. }) = operation
    else {
        return Ok(());
    };
    schema.validate()?;

    if matches!(operation, Operation::Overwrite { .. }) {
        return validate_new_field_ids(manifest, schema.fields_pre_order());
    }

    for field in schema.fields_pre_order() {
        let Some(prior_field) = manifest.schema.field_by_id(field.id) else {
            continue;
        };
        if field.parent_id != prior_field.parent_id {
            return Err(Error::invalid_input(format!(
                "Field ID {} moves from parent {} to parent {}; non-reusable field identity cannot move between parents",
                field.id, prior_field.parent_id, field.parent_id
            )));
        }
        if field.logical_type != prior_field.logical_type {
            return Err(Error::invalid_input(format!(
                "Field ID {} changes logical type from {} to {}; type replacement must allocate a new field ID",
                field.id, prior_field.logical_type, field.logical_type
            )));
        }
    }

    validate_new_field_ids(
        manifest,
        schema
            .fields_pre_order()
            .filter(|field| manifest.schema.field_by_id(field.id).is_none()),
    )
}

/// Reject detached schema changes once non-reusable field identity is active.
pub fn validate_detached_non_reusable_field_ids(
    manifest: &Manifest,
    operation: &Operation,
) -> Result<()> {
    if !manifest.uses_non_reusable_field_ids() {
        return Ok(());
    }
    match operation {
        Operation::Merge { schema, .. } if schema == &manifest.schema => Ok(()),
        Operation::Merge { .. }
        | Operation::Project { .. }
        | Operation::Overwrite { .. }
        | Operation::Restore { .. } => Err(Error::invalid_input(
            "Detached commits cannot change schema after non-reusable field IDs are activated",
        )),
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

    // Id 0 means unassigned (assigned at commit, like Append); a non-zero id
    // unknown to the manifest was pre-reserved. The first occurrence of each
    // previous id is the existing fragment (mirrors build_manifest), so staged
    // id-0 fragments must follow the existing ones.
    let mut new_fragment_map: HashMap<u64, &Fragment> = HashMap::new();
    for fragment in new_fragments {
        if fragment.id != 0 && new_fragment_map.contains_key(&fragment.id) {
            return Err(Error::invalid_input(format!(
                "Merge operation contains duplicate fragment id {}. \
                 New fragments must use id 0 (assigned at commit time) or a unique reserved id.",
                fragment.id
            )));
        }
        new_fragment_map.entry(fragment.id).or_insert(fragment);
    }

    // build_manifest treats the FIRST occurrence of each previous id as the
    // merged existing fragment, so a staged id-0 fragment listed before the
    // fragment it collides with would silently take that fragment's place.
    // The row-count and row-id-metadata checks below only catch that swap
    // when the two fragments differ; enforce the order structurally instead:
    // every fragment classified as existing must precede every new one.
    let previous_ids: HashSet<u64> = original_fragments.iter().map(|f| f.id).collect();
    let mut seen_previous: HashSet<u64> = HashSet::new();
    let mut first_new_id: Option<u64> = None;
    for fragment in new_fragments {
        let is_existing = previous_ids.contains(&fragment.id) && seen_previous.insert(fragment.id);
        if !is_existing {
            first_new_id.get_or_insert(fragment.id);
        } else if let Some(new_id) = first_new_id {
            return Err(Error::invalid_input(format!(
                "Merge operation lists existing fragment {} after a new fragment \
                 (id {}). New fragments must be listed after every existing fragment.",
                fragment.id, new_id
            )));
        }
    }

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
            // A previous-id fragment without row id metadata on a stable
            // dataset is most likely a staged fragment listed too early.
            if manifest.uses_stable_row_ids() && new_fragment.row_id_meta.is_none() {
                return Err(Error::invalid_input(format!(
                    "Merge operation dropped row id metadata for existing fragment {}. \
                     New fragments (id 0) must be listed after the existing fragments; \
                     they are assigned row ids at commit time.",
                    original_fragment.id
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
/// bytes. Shared ids may be renamed to unused names within the same parent,
/// but must not take another sibling's name or move between parents. Parent ids
/// must agree with the actual schema tree. Their logical type, nullability,
/// storage encoding, and dictionary may change only when every
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
    // First occurrence wins: a later staged fragment with placeholder id 0 is
    // not fragment 0's merged version.
    let mut new_fragment_map: HashMap<u64, &Fragment> = HashMap::new();
    for fragment in fragments {
        new_fragment_map.entry(fragment.id).or_insert(fragment);
    }

    // Remap and semantic errors first: a renumbered schema usually violates
    // both the shared-id and new-id clauses.
    for field in new_schema.fields_pre_order() {
        let Some(prior_field) = prior_schema.field_by_id(field.id) else {
            continue;
        };
        // A cloned field can keep a stale parent_id after moving in the schema tree.
        let prior_parent = prior_schema
            .field_ancestry_by_id(field.id)
            .ok_or_else(|| {
                Error::internal(format!(
                    "Missing prior schema ancestry for field id {}",
                    field.id
                ))
            })?
            .into_iter()
            .rev()
            .nth(1);
        let new_parent = new_schema
            .field_ancestry_by_id(field.id)
            .ok_or_else(|| {
                Error::internal(format!(
                    "Missing proposed schema ancestry for field id {}",
                    field.id
                ))
            })?
            .into_iter()
            .rev()
            .nth(1);
        let prior_parent_id = prior_parent.map_or(-1, |parent| parent.id);
        let new_parent_id = new_parent.map_or(-1, |parent| parent.id);
        if prior_field.parent_id != prior_parent_id || field.parent_id != new_parent_id {
            return Err(Error::invalid_input(format!(
                "Merge operation has inconsistent parent ids for field id {} (\"{}\"): \
                 prior field declares {} but its schema ancestry has {}; proposed field \
                 declares {} but its schema ancestry has {}.",
                field.id,
                new_schema.field_path(field.id)?,
                prior_field.parent_id,
                prior_parent_id,
                field.parent_id,
                new_parent_id
            )));
        }
        // Match siblings by parent id so an ancestor rename does not hide renumbering.
        let prior_siblings = prior_parent.map_or(prior_schema.fields.as_slice(), |parent| {
            parent.children.as_slice()
        });
        if new_parent_id != prior_parent_id
            || prior_siblings
                .iter()
                .any(|sibling| sibling.name == field.name && sibling.id != field.id)
        {
            return Err(Error::invalid_input(format!(
                "Merge operation remaps field id {} from \"{}\" to \"{}\". \
                 Merge must preserve the dataset's field ids: derive the new schema \
                 from the dataset's current schema instead of renumbering fields.",
                field.id,
                prior_schema.field_path(field.id)?,
                new_schema.field_path(field.id)?
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
                field.id,
                new_schema.field_path(field.id)?,
                changes
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
                .any(|file| file_materializes_field(file.fields.as_ref(), field))
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

fn file_materializes_field(file_field_ids: &[i32], field: &Field) -> bool {
    file_field_ids.contains(&field.id)
        || field
            .children
            .iter()
            .any(|child| file_materializes_field(file_field_ids, child))
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

    /// An offsets key outside the replacements would stamp an unrelated
    /// fragment's version metadata.
    #[test]
    fn test_data_replacement_offsets_must_name_a_replaced_fragment() {
        use crate::format::DataFile;
        use crate::transaction::DataReplacementGroup;
        use roaring::RoaringBitmap;
        use std::collections::HashMap;
        let replacement = DataReplacementGroup(
            3,
            DataFile::new_legacy_from_fields("replacement.lance", vec![0], None),
        );
        let manifest =
            crate::transaction::test_support::make_stable_row_id_manifest(vec![Fragment::new(3)]);
        let with_offsets = |fragment_id: u64| Operation::DataReplacement {
            replacements: vec![replacement.clone()],
            replaced_offsets: Some(UpdatedFragmentOffsets(HashMap::from([(
                fragment_id,
                RoaringBitmap::from_iter([0u32]),
            )]))),
        };
        assert!(validate_operation(Some(&manifest), &with_offsets(3)).is_ok());
        let error = validate_operation(Some(&manifest), &with_offsets(4)).unwrap_err();
        assert!(
            error.to_string().contains("is not in replacements"),
            "{error}"
        );
    }

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

    fn activated_manifest() -> Manifest {
        let schema = one_field_schema();
        let mut manifest = manifest_with_file_fields(schema, vec![0]);
        manifest.activate_non_reusable_field_ids();
        manifest
    }

    #[test]
    fn non_reusable_field_ids_allow_reserved_ids_above_high_water_mark() {
        let mut manifest = activated_manifest();
        manifest.max_allocated_field_id = Some(5);
        let mut schema = manifest.schema.clone();
        let mut new_field =
            LanceCoreField::try_from(&ArrowField::new("b", DataType::Int32, true)).unwrap();
        new_field.id = 6;
        schema.fields.push(new_field);
        let valid = Operation::Project {
            schema: schema.clone(),
            preserves_nullability: true,
        };
        validate_operation(Some(&manifest), &valid).unwrap();

        schema.fields.last_mut().unwrap().id = 7;
        let reserved = Operation::Project {
            schema: schema.clone(),
            preserves_nullability: true,
        };
        validate_operation(Some(&manifest), &reserved).unwrap();

        schema.fields.last_mut().unwrap().id = 5;
        let reused = Operation::Project {
            schema,
            preserves_nullability: true,
        };
        let err = validate_operation(Some(&manifest), &reused).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(
            err.to_string()
                .contains("greater than the high-water mark 5"),
            "{err}"
        );
    }

    #[test]
    fn non_reusable_field_ids_require_fresh_identity_for_type_replacement() {
        let manifest = activated_manifest();
        let mut schema = manifest.schema.clone();
        schema.fields[0].logical_type = LogicalType::try_from(&DataType::Float32).unwrap();
        let operation = Operation::Project {
            schema,
            preserves_nullability: true,
        };

        let err = validate_operation(Some(&manifest), &operation).unwrap_err();

        assert!(err.to_string().contains("type replacement"), "{err}");
    }

    #[test]
    fn non_reusable_field_ids_reject_overwrite_with_existing_ids() {
        let manifest = activated_manifest();
        let schema = manifest.schema.clone();
        let operation = Operation::Overwrite {
            fragments: vec![fragment_with_file_fields(0, "new.lance", vec![0])],
            schema,
            config_upsert_values: None,
            initial_bases: None,
        };

        for err in [
            validate_operation(Some(&manifest), &operation).unwrap_err(),
            validate_non_reusable_field_id_transition(&manifest, &manifest, &operation)
                .unwrap_err(),
        ] {
            assert!(matches!(err, Error::InvalidInput { .. }));
            assert!(
                err.to_string()
                    .contains("greater than the high-water mark 0")
            );
        }
    }

    #[test]
    fn canonicalize_overwrite_remaps_hostile_arrow_field_ids() {
        let manifest = activated_manifest();
        let mut schema = one_field_schema();
        schema.fields[0].id = i32::MAX;
        let mut operation = Operation::Overwrite {
            fragments: vec![fragment_with_file_fields(0, "new.lance", vec![i32::MAX])],
            schema,
            config_upsert_values: None,
            initial_bases: None,
        };

        canonicalize_non_reusable_field_ids(Some(&manifest), &mut operation, None).unwrap();

        let Operation::Overwrite {
            schema, fragments, ..
        } = operation
        else {
            unreachable!();
        };
        assert_eq!(schema.fields[0].id, 1);
        assert_eq!(fragments[0].files[0].fields.as_ref(), &[1]);
    }

    #[rstest::rstest]
    #[case::last_id(i32::MAX - 1)]
    #[case::exhausted(i32::MAX)]
    fn overwrite_cannot_keep_existing_ids_to_avoid_exhaustion(#[case] max_id: i32) {
        let mut manifest = activated_manifest();
        manifest.max_allocated_field_id = Some(max_id);
        let mut operation = Operation::Overwrite {
            schema: manifest.schema.clone(),
            fragments: vec![],
            config_upsert_values: None,
            initial_bases: None,
        };
        let result = canonicalize_non_reusable_field_ids(Some(&manifest), &mut operation, None);
        if max_id == i32::MAX {
            let err = result.unwrap_err();
            assert!(matches!(err, Error::InvalidInput { .. }));
            assert!(err.to_string().contains("IDs are exhausted"));
        } else {
            result.unwrap();
            let Operation::Overwrite { schema, .. } = operation else {
                unreachable!();
            };
            assert_eq!(schema.fields[0].id, i32::MAX);
        }
    }

    #[test]
    fn canonicalize_overwrite_remaps_standalone_fragment_field_ids() {
        let manifest = activated_manifest();
        let mut schema = one_field_schema();
        schema.fields[0].id = -1;
        let mut operation = Operation::Overwrite {
            fragments: vec![fragment_with_file_fields(0, "new.lance", vec![0])],
            schema,
            config_upsert_values: None,
            initial_bases: None,
        };

        resolve_arrow_field_ids(Some(&manifest), &mut operation).unwrap();

        let Operation::Overwrite {
            schema, fragments, ..
        } = operation
        else {
            unreachable!();
        };
        assert_eq!(schema.fields[0].id, 1);
        assert_eq!(fragments[0].files[0].fields.as_ref(), &[1]);
    }

    #[rstest::rstest]
    fn canonicalize_overwrite_retry_assigns_all_ids_above_latest_mark(
        #[values(false, true)] replace: bool,
    ) {
        let read = activated_manifest();
        let mut staged = read.schema.clone();
        let mut new_field = Field::new_arrow("new_column", DataType::Int32, true).unwrap();
        new_field.id = 1;
        staged.fields.push(new_field.clone());
        let mut latest_schema = read.schema.clone();
        if replace {
            latest_schema.fields.clear();
        }
        new_field.name = "concurrent_column".to_string();
        latest_schema.fields.push(new_field);
        let latest_ids = latest_schema.field_ids().into_iter().collect();
        let mut latest = manifest_with_file_fields(latest_schema, latest_ids);
        latest.activate_non_reusable_field_ids();
        let mut operation = Operation::Overwrite {
            schema: staged,
            fragments: vec![fragment_with_file_fields(0, "new.lance", vec![0, 1])],
            config_upsert_values: None,
            initial_bases: None,
        };

        canonicalize_non_reusable_field_ids(Some(&latest), &mut operation, Some(&read.schema))
            .unwrap();
        validate_operation(Some(&latest), &operation).unwrap();

        let Operation::Overwrite {
            schema, fragments, ..
        } = operation
        else {
            unreachable!();
        };
        assert_eq!(
            schema
                .fields
                .iter()
                .map(|field| field.id)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(fragments[0].files[0].fields.as_ref(), &[2, 3]);
        assert_eq!(schema.fields[1].name, "new_column");
    }

    #[test]
    fn canonicalize_raw_arrow_overwrite_assigns_new_ids_in_schema_order() {
        let schema = LanceSchema::try_from(&ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("b", DataType::Int32, true),
        ]))
        .unwrap();
        let mut manifest = manifest_with_file_fields(schema, vec![0, 1]);
        manifest.activate_non_reusable_field_ids();
        let mut raw_schema = LanceSchema::try_from(&ArrowSchema::new(vec![
            ArrowField::new("b", DataType::Int32, true),
            ArrowField::new("a", DataType::Int32, true),
        ]))
        .unwrap();
        raw_schema
            .metadata
            .insert("source".to_string(), "user metadata".to_string());
        let mut operation = Operation::Overwrite {
            fragments: vec![],
            schema: raw_schema,
            config_upsert_values: None,
            initial_bases: None,
        };

        resolve_arrow_field_ids(Some(&manifest), &mut operation).unwrap();

        let Operation::Overwrite { schema, .. } = operation else {
            unreachable!();
        };
        assert_eq!(schema.field("b").unwrap().id, 2);
        assert_eq!(schema.field("a").unwrap().id, 3);
        assert_eq!(schema.metadata.get("source").unwrap(), "user metadata");
    }

    #[rstest::rstest]
    #[case::canonical_ids([1, 2])]
    #[case::duplicate_arrow_metadata([42, 42])]
    fn canonicalize_overwrite_preserves_already_canonical_fragment_field_ids(
        #[case] input_ids: [i32; 2],
    ) {
        let manifest = activated_manifest();
        let mut schema = LanceSchema::try_from(&ArrowSchema::new(vec![
            ArrowField::new("b", DataType::Int32, true),
            ArrowField::new("c", DataType::Int32, true),
        ]))
        .unwrap();
        schema.fields[0].id = input_ids[0];
        schema.fields[1].id = input_ids[1];
        let mut operation = Operation::Overwrite {
            fragments: vec![fragment_with_file_fields(0, "new.lance", vec![1, 2])],
            schema,
            config_upsert_values: None,
            initial_bases: None,
        };

        resolve_arrow_field_ids(Some(&manifest), &mut operation).unwrap();

        let Operation::Overwrite {
            schema, fragments, ..
        } = operation
        else {
            unreachable!();
        };
        assert_eq!(
            schema
                .fields_pre_order()
                .map(|field| field.id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(fragments[0].files[0].fields.as_ref(), &[1, 2]);
    }

    #[test]
    fn canonicalize_overwrite_uses_one_raw_namespace_for_split_files() {
        let manifest = activated_manifest();
        let mut schema = LanceSchema::try_from(&ArrowSchema::new(vec![
            ArrowField::new("b", DataType::Int32, true),
            ArrowField::new("c", DataType::Int32, true),
        ]))
        .unwrap();
        for field in &mut schema.fields {
            field.id = -1;
        }
        let mut fragment = fragment_with_file_fields(0, "b.lance", vec![0]);
        fragment
            .files
            .push(DataFile::new_legacy_from_fields("c.lance", vec![1], None));
        let mut operation = Operation::Overwrite {
            fragments: vec![fragment],
            schema,
            config_upsert_values: None,
            initial_bases: None,
        };

        resolve_arrow_field_ids(Some(&manifest), &mut operation).unwrap();

        let Operation::Overwrite {
            schema, fragments, ..
        } = operation
        else {
            unreachable!();
        };
        assert_eq!(
            schema
                .fields_pre_order()
                .map(|field| field.id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(fragments[0].files[0].fields.as_ref(), &[1]);
        assert_eq!(fragments[0].files[1].fields.as_ref(), &[2]);
    }

    #[test]
    fn canonicalize_overwrite_rejects_ambiguous_raw_field_ids() {
        let manifest = activated_manifest();
        let mut schema = LanceSchema::try_from(&ArrowSchema::new(vec![
            ArrowField::new("b", DataType::Int32, true),
            ArrowField::new("c", DataType::Int32, true),
        ]))
        .unwrap();
        schema.fields[0].id = 2;
        schema.fields[1].id = 1;
        let mut operation = Operation::Overwrite {
            fragments: vec![fragment_with_file_fields(0, "new.lance", vec![2, 1])],
            schema,
            config_upsert_values: None,
            initial_bases: None,
        };

        let err = resolve_arrow_field_ids(Some(&manifest), &mut operation).unwrap_err();

        assert!(err.to_string().contains("ambiguous raw Arrow field IDs"));
    }

    #[test]
    fn canonicalize_raw_arrow_project_rejects_unmatched_field() {
        let manifest = activated_manifest();
        let mut schema = one_field_schema();
        schema.fields[0].name = "renamed".to_string();
        schema.fields[0].id = -1;
        let mut operation = Operation::Project {
            schema,
            preserves_nullability: true,
        };

        let err = resolve_arrow_field_ids(Some(&manifest), &mut operation).unwrap_err();

        assert!(err.to_string().contains("writes no data"), "{err}");
    }

    #[test]
    fn canonicalize_raw_arrow_project_preserves_explicit_existing_identity() {
        let manifest = activated_manifest();
        let mut schema = one_field_schema();
        schema.fields[0].name = "renamed".to_string();
        let mut operation = Operation::Project {
            schema,
            preserves_nullability: true,
        };

        resolve_arrow_field_ids(Some(&manifest), &mut operation).unwrap();

        let Operation::Project { schema, .. } = operation else {
            unreachable!();
        };
        assert_eq!(schema.fields[0].name, "renamed");
        assert_eq!(schema.fields[0].id, 0);
    }

    #[test]
    fn canonicalize_raw_arrow_schema_for_legacy_dataset() {
        let manifest = manifest_with_file_fields(one_field_schema(), vec![0]);
        let mut schema = one_field_schema();
        schema.fields[0].id = -1;
        let mut operation = Operation::Project {
            schema,
            preserves_nullability: true,
        };

        resolve_arrow_field_ids(Some(&manifest), &mut operation).unwrap();

        let Operation::Project { schema, .. } = operation else {
            unreachable!();
        };
        assert_eq!(schema.fields[0].id, 0);
        assert!(schema.metadata.is_empty());
    }

    #[test]
    fn canonicalize_merge_preserves_identity_for_physical_rewrite() {
        let mut manifest = activated_manifest();
        Arc::make_mut(&mut manifest.fragments).push(fragment_with_file_fields(
            1,
            "retained.lance",
            vec![0],
        ));
        let rewritten_fragment = fragment_with_file_fields(0, "replacement.lance", vec![0]);
        let mut operation = Operation::Merge {
            fragments: vec![rewritten_fragment, manifest.fragments[1].clone()],
            schema: manifest.schema.clone(),
            preserves_nullability: true,
        };

        canonicalize_non_reusable_field_ids(Some(&manifest), &mut operation, None).unwrap();

        let Operation::Merge {
            schema, fragments, ..
        } = operation
        else {
            unreachable!();
        };
        assert_eq!(schema.fields[0].id, 0);
        assert_eq!(fragments[0].files[0].fields.as_ref(), &[0]);
        assert_eq!(fragments[1].files[0].fields.as_ref(), &[0]);
        validate_operation(
            Some(&manifest),
            &Operation::Merge {
                fragments,
                schema,
                preserves_nullability: true,
            },
        )
        .unwrap();
    }

    #[test]
    fn canonicalize_merge_allocates_fresh_identity_for_overlaid_column() {
        let manifest = activated_manifest();
        let mut merged_fragment = manifest.fragments[0].clone();
        merged_fragment.files.push(DataFile::new_legacy_from_fields(
            "replacement.lance",
            vec![0],
            None,
        ));
        let mut operation = Operation::Merge {
            fragments: vec![merged_fragment],
            schema: manifest.schema.clone(),
            preserves_nullability: true,
        };

        canonicalize_non_reusable_field_ids(Some(&manifest), &mut operation, None).unwrap();

        let Operation::Merge {
            schema, fragments, ..
        } = operation
        else {
            unreachable!();
        };
        assert_eq!(schema.fields[0].id, 1);
        assert_eq!(fragments[0].files[0].fields.as_ref(), &[0]);
        assert_eq!(fragments[0].files[1].fields.as_ref(), &[1]);
    }

    #[test]
    fn canonicalize_merge_rejects_ambiguous_raw_field_ids() {
        let manifest = activated_manifest();
        let mut schema = LanceSchema::try_from(&ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("b", DataType::Int32, true),
            ArrowField::new("c", DataType::Int32, true),
        ]))
        .unwrap();
        schema.fields[0].id = 0;
        schema.fields[1].id = 2;
        schema.fields[2].id = 1;
        let mut merged_fragment = manifest.fragments[0].clone();
        merged_fragment.files.push(DataFile::new_legacy_from_fields(
            "new.lance",
            vec![1, 2],
            None,
        ));
        let mut operation = Operation::Merge {
            fragments: vec![merged_fragment],
            schema,
            preserves_nullability: true,
        };

        let err = resolve_arrow_field_ids(Some(&manifest), &mut operation).unwrap_err();

        assert!(err.to_string().contains("ambiguous raw Arrow field IDs"));
    }

    #[test]
    fn non_reusable_field_id_manifest_rejects_high_water_mark_below_overlay_reference() {
        let mut manifest = activated_manifest();
        Arc::make_mut(&mut manifest.fragments)[0]
            .overlays
            .push(DataOverlayFile {
                data_file: DataFile::new_legacy_from_fields("overlay.lance", vec![7], None),
                coverage: OverlayCoverage::Shared(Arc::new(RoaringBitmap::from_iter([0_u32]))),
                committed_version: 1,
            });

        let err = validate_non_reusable_field_id_manifest(&manifest).unwrap_err();

        assert!(
            err.to_string().contains("below referenced field ID 7"),
            "{err}"
        );
    }

    #[test]
    fn non_reusable_field_id_transition_rejects_file_only_allocator_advance() {
        let manifest = activated_manifest();
        let mut successor = Manifest::new_from_previous(
            &manifest,
            manifest.schema.clone(),
            Arc::new(vec![fragment_with_file_fields(0, "new.lance", vec![0, 1])]),
        );
        successor.max_allocated_field_id = manifest.max_allocated_field_id;
        let operation = Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: None,
            schema_metadata_updates: None,
            field_metadata_updates: HashMap::new(),
        };

        let err = validate_non_reusable_field_id_transition(&manifest, &successor, &operation)
            .unwrap_err();

        assert!(
            err.to_string().contains("canonical successor schema"),
            "{err}"
        );
    }

    #[test]
    fn non_reusable_field_id_transition_rejects_decreasing_high_water_mark() {
        let manifest = activated_manifest();
        let mut successor = Manifest::new_from_previous(
            &manifest,
            manifest.schema.clone(),
            manifest.fragments.clone(),
        );
        successor.max_allocated_field_id = Some(manifest.max_field_id() - 1);
        let operation = Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: None,
            schema_metadata_updates: None,
            field_metadata_updates: HashMap::new(),
        };

        let err = validate_non_reusable_field_id_transition(&manifest, &successor, &operation)
            .unwrap_err();

        assert!(
            err.to_string().contains("high-water mark decreases"),
            "{err}"
        );
    }

    #[test]
    fn detached_non_reusable_field_ids_allow_data_only_merge_and_reject_schema_change() {
        let manifest = activated_manifest();
        let data_only = Operation::Merge {
            fragments: manifest.fragments.as_ref().clone(),
            schema: manifest.schema.clone(),
            preserves_nullability: true,
        };
        validate_detached_non_reusable_field_ids(&manifest, &data_only).unwrap();

        let mut changed_schema = manifest.schema.clone();
        changed_schema.fields[0].name = "renamed".to_string();
        let schema_change = Operation::Project {
            schema: changed_schema,
            preserves_nullability: true,
        };
        let err = validate_detached_non_reusable_field_ids(&manifest, &schema_change).unwrap_err();
        assert!(
            err.to_string()
                .contains("Detached commits cannot change schema"),
            "{err}"
        );
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
    fn test_merge_preserved_dictionary_not_flagged() {
        // The manifest schema stores a dictionary as offset/length with no
        // attached values, so an unchanged dictionary must not count as a field
        // change (which would demand a full rewrite). The moved-offset half
        // passes under the old comparison too, where every dictionary looked
        // changed; it is here to pin that offset stays part of the identity.
        let dictionary = lance_core::datatypes::Dictionary {
            offset: 100,
            length: 5,
            values: None,
        };
        let schema = LanceSchema::try_from(&ArrowSchema::new(vec![ArrowField::new(
            "d",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        )]))
        .unwrap();
        let mut prior_schema = schema.clone();
        prior_schema.fields[0].dictionary = Some(dictionary.clone());
        let manifest = Manifest::new(
            prior_schema,
            Arc::new(vec![fragment_with_file_fields(0, "old.lance", vec![0])]),
            DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );

        let mut unchanged = schema.clone();
        unchanged.fields[0].dictionary = Some(dictionary);
        let retained = vec![fragment_with_file_fields(0, "old.lance", vec![0])];
        assert_eq!(
            shared_field_binding_changes(&manifest.schema.fields[0], &unchanged.fields[0]),
            None
        );
        merge_schema_valid(&manifest, &unchanged, &retained).unwrap();

        let mut moved = schema;
        moved.fields[0].dictionary = Some(lance_core::datatypes::Dictionary {
            offset: 200,
            length: 5,
            values: None,
        });
        assert_eq!(
            shared_field_binding_changes(&manifest.schema.fields[0], &moved.fields[0]),
            Some("dictionary".to_string())
        );
        let err = merge_schema_valid(&manifest, &moved, &retained).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "got {err:?}");
        assert!(
            err.to_string()
                .contains(r#"changes field id 0 ("d") without rewriting it in every existing fragment: dictionary."#),
            "unexpected error: {err}"
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

    #[rstest::rstest]
    #[case::parent("s")]
    #[case::child("s.x")]
    #[test]
    fn test_merge_allows_id_preserving_rename(#[case] path: &str) {
        let schema = LanceSchema::try_from(&ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(
                vec![
                    ArrowField::new("x", DataType::Int32, true),
                    ArrowField::new("y", DataType::Int32, true),
                ]
                .into(),
            ),
            true,
        )]))
        .unwrap();
        let manifest = manifest_with_file_fields(schema.clone(), vec![1, 2]);
        let mut renamed = schema;
        let field_id = renamed.field(path).unwrap().id;
        renamed.mut_field_by_id(field_id).unwrap().name = "renamed".into();
        merge_schema_valid(&manifest, &renamed, &manifest.fragments).unwrap();

        // Renaming an ancestor must not hide an accidental remapping of its children.
        renamed.fields[0].children.swap(0, 1);
        renamed.fields[0].children[0].id = 1;
        renamed.fields[0].children[1].id = 2;
        let err = merge_schema_valid(&manifest, &renamed, &manifest.fragments).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(err.to_string().contains("remaps field id 1"), "{err}");
    }

    #[rstest::rstest]
    #[case::dropped_parent_stale(false, false)]
    #[case::retained_parent_stale(true, false)]
    #[case::dropped_parent_updated(false, true)]
    #[case::retained_parent_updated(true, true)]
    #[test]
    fn test_merge_rejects_moving_field_to_unused_path(
        #[case] retain_parent: bool,
        #[case] update_parent_id: bool,
    ) {
        let schema = LanceSchema::try_from(&ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(
                vec![
                    ArrowField::new("x", DataType::Int32, true),
                    ArrowField::new("y", DataType::Int32, true),
                ]
                .into(),
            ),
            true,
        )]))
        .unwrap();
        let manifest = manifest_with_file_fields(schema.clone(), vec![1, 2]);
        let mut moved = schema;
        let mut child = moved.fields[0].children.remove(0);
        if !retain_parent {
            moved.fields.clear();
        }
        child.name = "renamed".into();
        if update_parent_id {
            child.parent_id = -1;
        }
        moved.fields.push(child);
        moved.validate().unwrap();

        let operation = Operation::Merge {
            schema: moved,
            fragments: manifest.fragments.as_ref().clone(),
            preserves_nullability: true,
        };
        let err = validate_operation(Some(&manifest), &operation).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        let expected = if update_parent_id {
            "remaps field id 1 from \"s.x\" to \"renamed\""
        } else {
            "inconsistent parent ids for field id 1"
        };
        assert!(err.to_string().contains(expected), "{err}");
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
