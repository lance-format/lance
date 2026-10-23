// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Set (replace) a fragment's deletion file.

use super::apply::ApplyState;
use super::proto::{data_change_from_wire, data_change_to_wire};
use super::{Coordinate, Footprint, Ref};
use crate::format::{DeletionFile, pb, pb_actions};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Set (replace) a fragment's deletion file.
///
/// This is reference-stable rather than a delta: the fragment id is committed
/// and physical row offsets never move, so the post-image is unambiguous.
///
/// TODO: derive the newly-deleted rows by diffing against the read version, so
/// rebase and conflict detection can merge two deletes on one fragment. Until
/// then the whole fragment's deletions are one coordinate, and two deletes on
/// the same fragment always conflict.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct SetDeletionFile {
    /// The fragment, by committed id.
    ///
    /// Unlike its sibling fragment actions this takes no [`Ref`](super::Ref).
    /// A deletion file's path is `{fragment_id}-{read_version}-{id}.{suffix}`
    /// (see [`deletion_file_path`](crate::io::deletion::deletion_file_path)),
    /// so the writer has to know the committed fragment id before it can write
    /// the file at all, and a minted id does not exist until apply.
    pub fragment: u64,
    /// The new deletion file, or `None` to clear the fragment's deletions.
    pub deletion_file: Option<DeletionFile>,
    /// See [`AddFragment::data_change`](super::AddFragment::data_change).
    pub data_change: bool,
    /// The base path the deletion file lives under: committed, or one minted
    /// earlier in the same operation. `None` for the dataset's own directory.
    pub base: Option<Ref>,
}

impl SetDeletionFile {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let deletion_file = match (&self.deletion_file, self.base) {
            (None, Some(base)) => {
                return Err(Error::invalid_input(format!(
                    "SetDeletionFile for fragment {} clears its deletion file but names base \
                     {base:?}; a base only says where a deletion file lives",
                    self.fragment
                )));
            }
            (None, None) => None,
            (Some(file), base) => {
                if let Some(base_id) = file.base_id {
                    return Err(Error::invalid_input(format!(
                        "SetDeletionFile for fragment {} carries deletion_file.base_id \
                         {base_id}; deletion_file.base_id must be unset, since the action's \
                         base names where the file lives",
                        self.fragment
                    )));
                }
                Some(DeletionFile {
                    base_id: base.map(|base| state.resolve_base(base)).transpose()?,
                    ..file.clone()
                })
            }
        };
        let fragment_id = state.resolve_fragment(Ref::Committed(self.fragment))?;
        state
            .fragment_mut(fragment_id, "SetDeletionFile")?
            .deletion_file = deletion_file;
        Ok(())
    }

    pub(super) fn is_data_change(&self) -> bool {
        self.data_change
    }

    /// The fragment's deletions, which is a distinct coordinate from the data of
    /// any field in it: deleting rows and re-encoding a column commute.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.write(Coordinate::FragmentDeletions(self.fragment));
    }
}

impl From<&SetDeletionFile> for pb_actions::SetDeletionFile {
    fn from(value: &SetDeletionFile) -> Self {
        Self {
            fragment: value.fragment,
            deletion_file: value.deletion_file.as_ref().map(pb::DeletionFile::from),
            data_change: data_change_to_wire(value.data_change),
            base: value.base.map(Into::into),
        }
    }
}

impl TryFrom<pb_actions::SetDeletionFile> for SetDeletionFile {
    type Error = Error;

    fn try_from(message: pb_actions::SetDeletionFile) -> Result<Self> {
        Ok(Self {
            fragment: message.fragment,
            deletion_file: message
                .deletion_file
                .map(DeletionFile::try_from)
                .transpose()?,
            data_change: data_change_from_wire(message.data_change),
            base: message.base.map(Ref::try_from).transpose()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DeletionFileType;
    use crate::transaction::action::test_support::{
        apply, backed_manifest, manifest_with_base, mint_base,
    };
    use crate::transaction::action::{Action, Ref};
    use rstest::rstest;

    #[test]
    fn test_set_deletion_file_sets_and_clears() {
        let manifest = backed_manifest();
        let deletion_file = DeletionFile {
            read_version: manifest.version,
            id: 3,
            file_type: DeletionFileType::Array,
            num_deleted_rows: Some(2),
            base_id: None,
        };
        let next = apply(
            &manifest,
            vec![Action::SetDeletionFile(SetDeletionFile {
                fragment: 0,
                deletion_file: Some(deletion_file.clone()),
                data_change: true,
                base: None,
            })],
        )
        .unwrap();
        assert_eq!(next.fragments[0].deletion_file, Some(deletion_file));

        // An absent deletion file is a request to clear it, not a no-op.
        let cleared = apply(
            &next,
            vec![Action::SetDeletionFile(SetDeletionFile {
                fragment: 0,
                deletion_file: None,
                data_change: true,
                base: None,
            })],
        )
        .unwrap();
        assert_eq!(cleared.fragments[0].deletion_file, None);
    }

    #[test]
    fn test_set_deletion_file_rejects_a_missing_fragment() {
        let error = apply(
            &backed_manifest(),
            vec![Action::SetDeletionFile(SetDeletionFile {
                fragment: 7,
                deletion_file: None,
                data_change: true,
                base: None,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
    }

    fn deletion_file(base_id: Option<u32>) -> DeletionFile {
        DeletionFile {
            read_version: 1,
            id: 3,
            file_type: DeletionFileType::Array,
            num_deleted_rows: Some(2),
            base_id,
        }
    }

    /// The same contract as [`AddDataFile`](super::super::AddDataFile)'s: `base`
    /// says where the file lives and apply stamps the embedded `base_id`.
    #[rstest]
    #[case::own_directory(Some(deletion_file(None)), None, false, Ok(None))]
    #[case::committed_base(Some(deletion_file(None)), Some(Ref::Committed(1)), false, Ok(Some(1)))]
    #[case::base_minted_earlier(Some(deletion_file(None)), Some(Ref::Local(0)), true, Ok(Some(2)))]
    #[case::embedded_base_id(
        Some(deletion_file(Some(1))),
        None,
        false,
        Err("deletion_file.base_id must be unset")
    )]
    #[case::committed_base_the_dataset_lacks(
        Some(deletion_file(None)),
        Some(Ref::Committed(0)),
        false,
        Err("base id 0 in an action names no base path of the dataset")
    )]
    #[case::base_not_minted(
        Some(deletion_file(None)),
        Some(Ref::Local(0)),
        false,
        Err("local base token 0, which no earlier action in this operation minted")
    )]
    #[case::base_for_a_cleared_deletion_file(
        None,
        Some(Ref::Committed(1)),
        false,
        Err("names base")
    )]
    fn test_set_deletion_file_stamps_its_base(
        #[case] file: Option<DeletionFile>,
        #[case] base: Option<Ref>,
        #[case] mint_base_first: bool,
        #[case] expected: std::result::Result<Option<u32>, &str>,
    ) {
        let mut actions = Vec::new();
        if mint_base_first {
            actions.push(mint_base());
        }
        actions.push(Action::SetDeletionFile(SetDeletionFile {
            fragment: 0,
            deletion_file: file,
            data_change: true,
            base,
        }));

        let result = apply(&manifest_with_base(), actions);

        match expected {
            Ok(base_id) => {
                let next = result.unwrap();
                let stamped = next.fragments[0].deletion_file.as_ref().unwrap();
                assert_eq!(stamped.base_id, base_id);
            }
            Err(message) => {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
                assert!(
                    error.to_string().contains(message),
                    "unexpected error: {error}"
                );
            }
        }
    }
}
