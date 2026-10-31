// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Storage-action constructors and routing keys.
//!
//! Constructors refuse only what an action cannot encode, an unknown row or
//! deleted-row count. Tree preparation checks every action against the
//! resolved fragment state before storing it.

use lance_core::{Error, Result};

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
pub fn upsert_fragment(fragment: &Fragment) -> Result<pb::FragmentAction> {
    require_known_counts(fragment)?;
    Ok(pb::FragmentAction {
        action: Some(Action::UpsertFragment(pb::DataFragment::from(fragment))),
    })
}

/// Remove a whole fragment (tombstone).
pub fn remove_fragment(frag_id: u64) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::RemoveFragment(frag_id)),
    }
}

/// Attach or replace the deletion file for `frag_id`.
pub fn add_deletion_file(frag_id: u64, deletion_file: &DeletionFile) -> Result<pb::FragmentAction> {
    let num_deleted_rows = known_deleted_rows(frag_id, deletion_file)?;
    let file_type = match deletion_file.file_type {
        DeletionFileType::Array => pb::deletion_file::DeletionFileType::ArrowArray,
        DeletionFileType::Bitmap => pb::deletion_file::DeletionFileType::Bitmap,
    };
    Ok(pb::FragmentAction {
        action: Some(Action::AddDeletionFile(pb::AddDeletionFile {
            frag_id,
            deletion_file: Some(pb::DeletionFile {
                read_version: deletion_file.read_version,
                id: deletion_file.id,
                file_type: file_type.into(),
                num_deleted_rows: num_deleted_rows as u64,
                base_id: deletion_file.base_id,
            }),
        })),
    })
}

/// Every count in the tree format is known, and the protobuf encoding
/// writes an unknown count as zero, so a record must state both counts
/// before it is encoded.
pub(super) fn require_known_counts(fragment: &Fragment) -> Result<()> {
    if fragment.physical_rows.is_none() {
        return Err(Error::invalid_input(format!(
            "fragment {} has no physical row count; the fragment metadata tree requires known counts",
            fragment.id
        )));
    }
    match &fragment.deletion_file {
        Some(deletion_file) => known_deleted_rows(fragment.id, deletion_file).map(|_| ()),
        None => Ok(()),
    }
}

fn known_deleted_rows(frag_id: u64, deletion_file: &DeletionFile) -> Result<usize> {
    deletion_file.num_deleted_rows.ok_or_else(|| {
        Error::invalid_input(format!(
            "fragment {frag_id} has a deletion file without a deleted-row count; the fragment \
             metadata tree requires known counts"
        ))
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment_metadata::support::make_fragment;
    use rstest::rstest;

    #[derive(Clone, Copy, Debug)]
    enum UnknownCount {
        UpsertPhysical,
        UpsertDeleted,
        AddedDeletionFile,
    }

    /// The protobuf encoding would store an unknown count as a known zero,
    /// so the constructors refuse it before encoding.
    #[rstest]
    #[case::upserted_physical_rows(UnknownCount::UpsertPhysical, "no physical row count")]
    #[case::upserted_deleted_rows(UnknownCount::UpsertDeleted, "without a deleted-row count")]
    #[case::added_deleted_rows(UnknownCount::AddedDeletionFile, "without a deleted-row count")]
    fn constructors_refuse_unknown_counts(#[case] unknown: UnknownCount, #[case] expected: &str) {
        let deletion_file = DeletionFile {
            read_version: 1,
            id: 1,
            file_type: DeletionFileType::Bitmap,
            num_deleted_rows: None,
            base_id: None,
        };
        let mut fragment = make_fragment(2);
        let error = match unknown {
            UnknownCount::UpsertPhysical => {
                fragment.physical_rows = None;
                upsert_fragment(&fragment)
            }
            UnknownCount::UpsertDeleted => {
                fragment.deletion_file = Some(deletion_file);
                upsert_fragment(&fragment)
            }
            UnknownCount::AddedDeletionFile => add_deletion_file(2, &deletion_file),
        }
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        let message = error.to_string();
        assert!(message.contains("fragment 2"), "{message}");
        assert!(message.contains(expected), "{message}");
        assert!(message.contains("requires known counts"), "{message}");
    }
}
