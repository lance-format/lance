// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Storage-action constructors and routing keys.
//!
//! These constructors do not validate actions. Tree preparation checks them
//! against the resolved fragment state before storing them.

use crate::format::pb::{self, fragment_action::Action};
use crate::format::{DataFile, DeletionFile, DeletionFileType, Fragment};

/// Attach `file` to fragment `frag_id` (the validated add-column case).
pub fn add_data_file(frag_id: u64, file: &DataFile) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::AddDataFile(pb::AddDataFile {
            frag_id,
            file: Some(pb::DataFile::from(file)),
        })),
    }
}

/// Drop the data file at `path` from `frag_id`.
pub fn remove_data_file(frag_id: u64, path: impl Into<String>) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::RemoveDataFile(pb::RemoveDataFile {
            frag_id,
            path: path.into(),
        })),
    }
}

/// Swap the data file at `expected_path` on `frag_id` for `file`. Only
/// [`crate::fragment_metadata::commit::data_replacement`] should build this, after
/// checking the fragment's current state.
pub fn replace_data_file(frag_id: u64, expected_path: &str, file: &DataFile) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::ReplaceDataFile(pb::ReplaceDataFile {
            frag_id,
            expected_path: expected_path.to_string(),
            path: file.path.clone(),
            file_size_bytes: file.file_size_bytes.get().map_or(0, |v| v.get()),
            base_id: file.base_id,
        })),
    }
}

/// Insert a fragment, or replace the whole record of an existing one.
pub fn upsert_fragment(fragment: &Fragment) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::UpsertFragment(pb::DataFragment::from(fragment))),
    }
}

/// Remove a whole fragment (tombstone).
pub fn remove_fragment(frag_id: u64) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::RemoveFragment(frag_id)),
    }
}

/// Attach or replace the deletion file for `frag_id`.
pub fn add_deletion_file(frag_id: u64, deletion_file: &DeletionFile) -> pb::FragmentAction {
    let file_type = match deletion_file.file_type {
        DeletionFileType::Array => pb::deletion_file::DeletionFileType::ArrowArray,
        DeletionFileType::Bitmap => pb::deletion_file::DeletionFileType::Bitmap,
    };
    pb::FragmentAction {
        action: Some(Action::AddDeletionFile(pb::AddDeletionFile {
            frag_id,
            deletion_file: Some(pb::DeletionFile {
                read_version: deletion_file.read_version,
                id: deletion_file.id,
                file_type: file_type.into(),
                num_deleted_rows: deletion_file.num_deleted_rows.unwrap_or_default() as u64,
                base_id: deletion_file.base_id,
            }),
        })),
    }
}

/// Clear the deletion file attached to `frag_id`.
pub fn clear_deletion_file(frag_id: u64) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::ClearDeletionFile(pb::ClearDeletionFile { frag_id })),
    }
}

/// The fragment id an action targets, used to route it to the owning child.
pub fn target_frag_id(action: &pb::FragmentAction) -> Option<u64> {
    match action.action.as_ref()? {
        Action::UpsertFragment(f) => Some(f.id),
        Action::RemoveFragment(id) => Some(*id),
        Action::AddDataFile(a) => Some(a.frag_id),
        Action::RemoveDataFile(a) => Some(a.frag_id),
        Action::ReplaceDataFile(a) => Some(a.frag_id),
        Action::AddDeletionFile(a) => Some(a.frag_id),
        Action::ClearDeletionFile(a) => Some(a.frag_id),
    }
}
