// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Set (replace) a fragment's deletion file.

use super::apply::ApplyState;
use super::proto::required;
use super::{Coordinate, Footprint, Ref};
use crate::format::{DeletionFile, pb, pb_actions};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Set (replace) a fragment's deletion file.
///
/// Physical row offsets within a fragment never move, so the post-image is
/// unambiguous.
///
/// A fragment's deletions only grow. Indices built while rows were deleted
/// never saw them, so bringing rows back would leave those indices silently
/// missing them. A deletion file therefore cannot be cleared, and a
/// replacement known to delete fewer rows than the one it replaces is
/// rejected.
///
/// TODO: derive the newly-deleted rows by diffing against the read version, so
/// rebase and conflict detection can merge two deletes on one fragment. Until
/// then the whole fragment's deletions are one coordinate, and two deletes on
/// the same fragment always conflict.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct SetDeletionFile {
    /// The target fragment, either committed or minted earlier in this
    /// operation.
    ///
    /// When [`Ref::Local`], `deletion_file` must store its vector inline or at
    /// an explicit [`DeletionFile::path`](crate::format::DeletionFile::path),
    /// because a minted fragment's id is assigned at apply and can change on
    /// rebase.
    pub fragment: Ref,
    /// The new deletion file.
    pub deletion_file: DeletionFile,
    /// See [`AddFragment::data_change`](super::AddFragment::data_change).
    pub data_change: bool,
    /// The base path the deletion file lives under: committed, or one minted
    /// earlier in the same operation. `None` for the dataset's own directory.
    pub base: Option<Ref>,
}

impl SetDeletionFile {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        self.deletion_file.validate_location()?;
        let fragment_id = state.resolve_fragment(self.fragment)?;
        if self.fragment.local().is_some() && self.deletion_file.uses_derived_path() {
            return Err(Error::invalid_input(format!(
                "SetDeletionFile for local fragment {fragment_id} cannot use a derived path; \
                 a deletion file on a minted fragment must set an explicit path or inline vector"
            )));
        }
        if let Some(base_id) = self.deletion_file.base_id {
            return Err(Error::invalid_input(format!(
                "SetDeletionFile for fragment {fragment_id} carries deletion_file.base_id {base_id}; \
                 deletion_file.base_id must be unset, since the action's base names where the \
                 file lives"
            )));
        }
        let deletion_file = DeletionFile {
            base_id: self.base.map(|base| state.resolve_base(base)).transpose()?,
            ..self.deletion_file.clone()
        };
        let fragment = state.fragment_mut(fragment_id, "SetDeletionFile")?;
        let deleted_before = fragment
            .deletion_file
            .as_ref()
            .and_then(|previous| previous.num_deleted_rows);
        if let (Some(before), Some(after)) = (deleted_before, deletion_file.num_deleted_rows)
            && after < before
        {
            return Err(Error::invalid_input(format!(
                "SetDeletionFile for fragment {fragment_id} deletes {after} rows, fewer than \
                 the {before} its current deletion file deletes; deleted rows cannot be \
                 restored"
            )));
        }
        fragment.deletion_file = Some(deletion_file);
        Ok(())
    }

    pub(super) fn is_data_change(&self) -> bool {
        self.data_change
    }

    /// The fragment's deletions, which is a distinct coordinate from the data of
    /// any field in it: deleting rows and re-encoding a column commute.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        if let Some(fragment) = self.fragment.committed() {
            footprint.write(Coordinate::FragmentDeletions(fragment));
        }
    }
}

impl From<&SetDeletionFile> for pb_actions::SetDeletionFile {
    fn from(value: &SetDeletionFile) -> Self {
        Self {
            fragment: Some(value.fragment.into()),
            deletion_file: Some(pb::DeletionFile::from(&value.deletion_file)),
            data_unchanged: !value.data_change,
            base: value.base.map(Into::into),
        }
    }
}

impl TryFrom<pb_actions::SetDeletionFile> for SetDeletionFile {
    type Error = Error;

    fn try_from(message: pb_actions::SetDeletionFile) -> Result<Self> {
        Ok(Self {
            fragment: required(message.fragment, "SetDeletionFile.fragment")?.try_into()?,
            deletion_file: DeletionFile::try_from(required(
                message.deletion_file,
                "SetDeletionFile.deletion_file",
            )?)?,
            data_change: !message.data_unchanged,
            base: message.base.map(Ref::try_from).transpose()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_flags::{FLAG_DELETION_FILE_PATHS, FLAG_INLINE_DELETIONS};
    use crate::format::DeletionFileType;
    use crate::transaction::action::test_support::{
        apply, backed_manifest, manifest_with_base, mint_base,
    };
    use crate::transaction::action::{Action, AddFragment, Ref};
    use bytes::Bytes;
    use rstest::rstest;

    fn set_deletion_file(file: DeletionFile) -> Action {
        Action::SetDeletionFile(SetDeletionFile {
            fragment: Ref::Committed(0),
            deletion_file: file,
            data_change: true,
            base: None,
        })
    }

    fn deleting(num_deleted_rows: Option<usize>) -> DeletionFile {
        DeletionFile {
            num_deleted_rows,
            ..deletion_file(None)
        }
    }

    #[test]
    fn test_set_deletion_file_sets_the_file() {
        let next = apply(
            &backed_manifest(),
            vec![set_deletion_file(deleting(Some(2)))],
        )
        .unwrap();

        assert_eq!(next.fragments[0].deletion_file, Some(deleting(Some(2))));
    }

    /// The protobuf field is a message, so absence is representable on the
    /// wire; it does not mean "clear".
    #[test]
    fn test_set_deletion_file_requires_a_file_on_the_wire() {
        let message = pb_actions::SetDeletionFile {
            fragment: Some(Ref::Committed(0).into()),
            deletion_file: None,
            data_unchanged: false,
            base: None,
        };

        let error = SetDeletionFile::try_from(message).unwrap_err();

        assert!(
            error.to_string().contains("SetDeletionFile.deletion_file"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_set_deletion_file_requires_a_fragment_on_the_wire() {
        let message = pb_actions::SetDeletionFile {
            fragment: None,
            deletion_file: Some(pb::DeletionFile::from(&deletion_file(None))),
            data_unchanged: false,
            base: None,
        };

        let error = SetDeletionFile::try_from(message).unwrap_err();

        assert!(
            error.to_string().contains("SetDeletionFile.fragment"),
            "unexpected error: {error}"
        );
    }

    /// Fewer deleted rows would bring rows back that indices never saw. A
    /// count either side does not record cannot be compared, and is let
    /// through.
    #[rstest]
    #[case::more(Some(2), Some(3), true)]
    #[case::same(Some(2), Some(2), true)]
    #[case::fewer(Some(2), Some(1), false)]
    #[case::unknown_before(None, Some(1), true)]
    #[case::unknown_after(Some(2), None, true)]
    fn test_set_deletion_file_rejects_restoring_deleted_rows(
        #[case] before: Option<usize>,
        #[case] after: Option<usize>,
        #[case] accepted: bool,
    ) {
        let manifest = apply(
            &backed_manifest(),
            vec![set_deletion_file(deleting(before))],
        )
        .unwrap();

        let result = apply(&manifest, vec![set_deletion_file(deleting(after))]);

        if accepted {
            result.unwrap();
        } else {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
            assert!(
                error
                    .to_string()
                    .contains("deleted rows cannot be restored"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn test_set_deletion_file_rejects_a_missing_fragment() {
        let error = apply(
            &backed_manifest(),
            vec![Action::SetDeletionFile(SetDeletionFile {
                fragment: Ref::Committed(7),
                deletion_file: deletion_file(None),
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
            path: None,
            inline: None,
        }
    }

    #[rstest]
    #[case::explicit_path(
        DeletionFile {
            file_type: DeletionFileType::Bitmap,
            path: Some("_deletions/0b2f8a3c-6d4e-4f10-9a7b-1c2d3e4f5a6b.bin".to_string()),
            ..deletion_file(None)
        },
        Ok(FLAG_DELETION_FILE_PATHS),
    )]
    #[case::inline_vector(
        DeletionFile {
            file_type: DeletionFileType::Bitmap,
            inline: Some(Bytes::from_static(&[1, 2, 3])),
            ..deletion_file(None)
        },
        Ok(FLAG_INLINE_DELETIONS),
    )]
    #[case::derived_path(deletion_file(None), Err("derived path"))]
    fn test_set_deletion_file_local_fragment_requires_explicit_path_or_inline(
        #[case] file: DeletionFile,
        #[case] expected: std::result::Result<u64, &str>,
    ) {
        let actions = vec![
            Action::AddFragment(AddFragment {
                id: Ref::Local(0),
                physical_rows: 10,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            }),
            Action::SetDeletionFile(SetDeletionFile {
                fragment: Ref::Local(0),
                deletion_file: file.clone(),
                data_change: true,
                base: None,
            }),
        ];

        let result = apply(&backed_manifest(), actions);

        match expected {
            Ok(flag) => {
                let next = result.unwrap();
                assert_eq!(next.fragments[1].id, 1);
                assert_eq!(next.fragments[1].deletion_file.as_ref(), Some(&file));
                assert_ne!(next.reader_feature_flags & flag, 0);
                assert_ne!(next.writer_feature_flags & flag, 0);
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

    /// The same contract as [`AddDataFile`](super::super::AddDataFile)'s: `base`
    /// says where the file lives and apply stamps the embedded `base_id`.
    #[rstest]
    #[case::own_directory(deletion_file(None), None, false, Ok(None))]
    #[case::committed_base(deletion_file(None), Some(Ref::Committed(1)), false, Ok(Some(1)))]
    #[case::base_minted_earlier(deletion_file(None), Some(Ref::Local(0)), true, Ok(Some(2)))]
    #[case::embedded_base_id(
        deletion_file(Some(1)),
        None,
        false,
        Err("deletion_file.base_id must be unset")
    )]
    #[case::committed_base_the_dataset_lacks(
        deletion_file(None),
        Some(Ref::Committed(0)),
        false,
        Err("base id 0 in an action names no base path of the dataset")
    )]
    #[case::base_not_minted(
        deletion_file(None),
        Some(Ref::Local(0)),
        false,
        Err("local base token 0, which no earlier action in this operation minted")
    )]
    fn test_set_deletion_file_stamps_its_base(
        #[case] file: DeletionFile,
        #[case] base: Option<Ref>,
        #[case] mint_base_first: bool,
        #[case] expected: std::result::Result<Option<u32>, &str>,
    ) {
        let mut actions = Vec::new();
        if mint_base_first {
            actions.push(mint_base());
        }
        actions.push(Action::SetDeletionFile(SetDeletionFile {
            fragment: Ref::Committed(0),
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
