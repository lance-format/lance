// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};

use lance_core::{Error, Result, datatypes::Field};
use lance_file::version::ConcreteFileVersion;

use super::{DataFile, DataStorageFormat, Manifest};
use crate::feature_flags::FLAG_MIXED_DATA_FILE_VERSIONS;

/// Validate the file versions and column mappings of an opened manifest.
pub fn check_manifest_storage_version(manifest: &mut Manifest) -> Result<()> {
    check_manifest_storage_contract(manifest, StorageContractMode::Read)
}

/// Validate a candidate manifest before finalizing its storage flags.
pub fn check_manifest_storage_version_for_commit(manifest: &mut Manifest) -> Result<()> {
    check_manifest_storage_contract(manifest, StorageContractMode::Commit)
}

/// Validate a manifest and finalize its default version and mixed-version flags.
pub fn finalize_manifest_storage_version(manifest: &mut Manifest) -> Result<()> {
    check_manifest_storage_contract(manifest, StorageContractMode::Finalize)
}

#[derive(Clone, Copy)]
enum StorageContractMode {
    Read,
    Commit,
    Finalize,
}

fn check_manifest_storage_contract(
    manifest: &mut Manifest,
    mode: StorageContractMode,
) -> Result<()> {
    let default_version = manifest.data_storage_format.lance_file_format();
    let mixed_enabled = manifest.reader_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS != 0
        && manifest.writer_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS != 0;

    if mixed_enabled && default_version == ConcreteFileVersion::V1 {
        return Err(Error::invalid_input(
            "Dataset has mixed data-file-version capability enabled, which requires a V2 default, but the manifest default is V1",
        ));
    }

    let mut saw_v1 = false;
    let mut saw_v2 = false;
    let mut first_file_version = None;
    let mut first_mismatch = None;
    let mut first_non_default = None;
    let fields_by_id = field_column_requirements(manifest);
    let mut validated_lists = HashSet::new();

    for fragment in manifest.fragments.iter() {
        for data_file in fragment.referenced_lance_files() {
            let file_version = data_file.file_version()?;
            match file_version {
                ConcreteFileVersion::V1 => saw_v1 = true,
                ConcreteFileVersion::V2_0
                | ConcreteFileVersion::V2_1
                | ConcreteFileVersion::V2_2
                | ConcreteFileVersion::V2_3 => saw_v2 = true,
            }

            match first_file_version {
                None => first_file_version = Some(file_version),
                Some(first_version)
                    if first_version != file_version && first_mismatch.is_none() =>
                {
                    first_mismatch = Some((first_version, file_version));
                }
                Some(_) => {}
            }

            if !mixed_enabled && file_version != default_version && first_non_default.is_none() {
                first_non_default = Some((data_file.path.clone(), fragment.id, file_version));
            }

            validate_file_column_indices(
                &fields_by_id,
                &mut validated_lists,
                fragment.id,
                data_file,
                file_version,
            )?;
        }
    }

    // Released Lance 0.16 could persist a V1 default while referencing both V1
    // and V2 files. Keep those snapshots readable, but never publish a new
    // manifest with that state.
    if matches!(mode, StorageContractMode::Read)
        && default_version == ConcreteFileVersion::V1
        && !mixed_enabled
        && saw_v1
        && saw_v2
    {
        return Ok(());
    }

    let mut effective_version = default_version;
    if default_version == ConcreteFileVersion::V1 {
        if let Some((first_version, other_version)) = first_mismatch {
            return Err(Error::internal(format!(
                "The dataset contains a mixture of file versions. You will need to rollback to an earlier version: All data files must have the same version. Detected both {first_version} and {other_version}"
            )));
        }
        if let Some(actual) = first_file_version
            && actual != ConcreteFileVersion::V1
        {
            effective_version = actual;
            first_non_default = None;
            if matches!(mode, StorageContractMode::Finalize) {
                log::warn!(
                    "Data storage version {} is less than the actual file version {}. This has been automatically updated.",
                    default_version,
                    actual
                );
                manifest.data_storage_format = DataStorageFormat::new(actual);
            }
        }
    }

    if saw_v1 && saw_v2 {
        return Err(Error::invalid_input(
            "Dataset snapshot mixes V1 and V2 data files",
        ));
    }

    if mixed_enabled && saw_v1 {
        return Err(Error::invalid_input(
            "Dataset has mixed data-file-version capability enabled but references V1 data files",
        ));
    }
    if let Some((path, fragment_id, file_version)) = first_non_default {
        if file_version == ConcreteFileVersion::V1 || effective_version == ConcreteFileVersion::V1 {
            return Err(Error::invalid_input(format!(
                "Data file '{path}' in fragment {fragment_id} has version {file_version}, but the manifest default is {effective_version}; V1 and V2 storage versions cannot be mixed"
            )));
        }
        match mode {
            StorageContractMode::Read => {
                return Err(Error::invalid_input(format!(
                    "Data file '{path}' in fragment {fragment_id} has version {file_version}, but the manifest default is {effective_version} and mixed data-file-version capability is not enabled"
                )));
            }
            StorageContractMode::Commit => {}
            StorageContractMode::Finalize => {
                manifest.reader_feature_flags |= FLAG_MIXED_DATA_FILE_VERSIONS;
                manifest.writer_feature_flags |= FLAG_MIXED_DATA_FILE_VERSIONS;
            }
        }
    }

    Ok(())
}

/// Validate data-file column mappings independently of the manifest default version.
pub fn validate_column_indices(manifest: &Manifest) -> Result<()> {
    let fields_by_id = field_column_requirements(manifest);
    let mut validated_lists = HashSet::new();
    for fragment in manifest.fragments.iter() {
        for data_file in fragment.referenced_lance_files() {
            validate_file_column_indices(
                &fields_by_id,
                &mut validated_lists,
                fragment.id,
                data_file,
                data_file.file_version()?,
            )?;
        }
    }
    Ok(())
}

fn field_column_requirements(manifest: &Manifest) -> HashMap<i32, (&Field, bool)> {
    let mut fields_by_id = HashMap::new();
    for field in manifest.schema.fields_pre_order() {
        let needs_column = field.is_leaf() || field.is_packed_struct() || field.is_blob();
        fields_by_id
            .entry(field.id)
            .or_insert((field, needs_column));
    }
    fields_by_id
}

fn validate_file_column_indices(
    fields_by_id: &HashMap<i32, (&Field, bool)>,
    validated_lists: &mut HashSet<(usize, usize)>,
    fragment_id: u64,
    data_file: &DataFile,
    file_version: ConcreteFileVersion,
) -> Result<()> {
    if file_version == ConcreteFileVersion::V1 || data_file.column_indices.is_empty() {
        return Ok(());
    }
    if data_file.fields.len() != data_file.column_indices.len() {
        return Err(Error::invalid_input(format!(
            "Data file '{}' (fragment {}) has {} field ids but {} column indices. These must be the same length.",
            data_file.path,
            fragment_id,
            data_file.fields.len(),
            data_file.column_indices.len()
        )));
    }
    if file_version == ConcreteFileVersion::V2_0 {
        return Ok(());
    }
    let list_key = (
        data_file.fields.as_ptr() as usize,
        data_file.column_indices.as_ptr() as usize,
    );
    if !validated_lists.insert(list_key) {
        return Ok(());
    }
    for (field_id, column_index) in data_file.fields.iter().zip(data_file.column_indices.iter()) {
        let Some((field, needs_column)) = fields_by_id.get(field_id).copied() else {
            continue;
        };
        if needs_column && *column_index == -1 {
            return Err(Error::invalid_input(format!(
                "Field '{}' (id={}) in data file '{}' (fragment {}) has column_index=-1, but leaf fields, packed structs, and blob fields must have a valid column index in file format 2.1+.",
                field.name, field_id, data_file.path, fragment_id
            )));
        }
        if !needs_column && *column_index != -1 {
            return Err(Error::invalid_input(format!(
                "Non-leaf field '{}' (id={}) in data file '{}' (fragment {}) has column_index={}, but non-leaf fields should have column_index=-1 in file format 2.1+.",
                field.name, field_id, data_file.path, fragment_id, column_index
            )));
        }
    }
    Ok(())
}
