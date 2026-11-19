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
/// This is reference-stable rather than a delta: the fragment id is committed
/// and physical row offsets never move, so the post-image is unambiguous.
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
    fragment: u64,
    deletion_file: DeletionFile,
    data_change: bool,
    base: Option<Ref>,
}

impl SetDeletionFile {
    /// Replace the deletion file of committed fragment `fragment` with
    /// `deletion_file`.
    ///
    /// * `fragment` is a committed id, unlike its sibling fragment actions: a
    ///   deletion file's path is `{fragment_id}-{read_version}-{id}.{suffix}`
    ///   (see [`deletion_file_path`](crate::io::deletion::deletion_file_path)),
    ///   so the writer has to know the committed fragment id before it can
    ///   write the file at all, and a minted id does not exist until apply.
    /// * `deletion_file` has `base_id` unset: apply stamps it from `base`.
    /// * `base` is the base path the file lives under: committed, or one minted
    ///   earlier in the same operation. `None` for the dataset's own directory.
    /// * `data_change` says whether a reader sees rows leave. See
    ///   [`AddFragment`](super::AddFragment).
    ///
    /// # Contract
    ///
    /// The caller certifies, and nothing downstream checks, that:
    ///
    /// * The file exists under `base`, is readable as the type
    ///   `deletion_file` declares, and names only offsets below the fragment's
    ///   physical row count.
    /// * `deletion_file.num_deleted_rows`, when set, is the number of offsets
    ///   the file names.
    /// * If `data_change` is `false`, the file deletes exactly the rows the
    ///   fragment already treats as deleted.
    ///
    /// Breaking the contract is not undefined behavior, but it commits a table
    /// whose reads fail or return the wrong rows.
    ///
    /// # Errors
    ///
    /// Rejects a `deletion_file` whose `base_id` is set.
    pub fn try_new_unchecked(
        fragment: u64,
        deletion_file: DeletionFile,
        base: Option<Ref>,
        data_change: bool,
    ) -> Result<Self> {
        if let Some(base_id) = deletion_file.base_id {
            return Err(Error::invalid_input(format!(
                "SetDeletionFile for fragment {fragment} carries deletion_file.base_id \
                 {base_id}; deletion_file.base_id must be unset, since the action's base names \
                 where the file lives"
            )));
        }
        Ok(Self {
            fragment,
            deletion_file,
            data_change,
            base,
        })
    }

    /// The committed fragment whose deletions are set.
    pub fn fragment(&self) -> u64 {
        self.fragment
    }

    /// The new deletion file, with `base_id` unset.
    pub fn deletion_file(&self) -> &DeletionFile {
        &self.deletion_file
    }

    /// The base path the deletion file lives under, or `None` for the
    /// dataset's own directory.
    pub fn base(&self) -> Option<Ref> {
        self.base
    }

    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let deletion_file = DeletionFile {
            base_id: self.base.map(|base| state.resolve_base(base)).transpose()?,
            ..self.deletion_file.clone()
        };
        let fragment_id = state.resolve_fragment(Ref::Committed(self.fragment))?;
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

    /// Whether a reader sees rows leave.
    pub fn is_data_change(&self) -> bool {
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
            deletion_file: Some(pb::DeletionFile::from(&value.deletion_file)),
            data_unchanged: !value.data_change,
            base: value.base.map(Into::into),
        }
    }
}

impl TryFrom<pb_actions::SetDeletionFile> for SetDeletionFile {
    type Error = Error;

    fn try_from(message: pb_actions::SetDeletionFile) -> Result<Self> {
        // A committed transaction file carries the certification of the
        // writer that staged it.
        Self::try_new_unchecked(
            message.fragment,
            DeletionFile::try_from(required(
                message.deletion_file,
                "SetDeletionFile.deletion_file",
            )?)?,
            message.base.map(Ref::try_from).transpose()?,
            !message.data_unchanged,
        )
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

    fn set_deletion_file(file: DeletionFile) -> Action {
        Action::SetDeletionFile(SetDeletionFile::try_new_unchecked(0, file, None, true).unwrap())
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
            fragment: 0,
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
            vec![Action::SetDeletionFile(
                SetDeletionFile::try_new_unchecked(7, deletion_file(None), None, true).unwrap(),
            )],
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
            inline: None,
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
        // An embedded base_id is rejected when the action is built, the rest
        // when it is applied.
        let result = SetDeletionFile::try_new_unchecked(0, file, base, true).and_then(|action| {
            actions.push(Action::SetDeletionFile(action));
            apply(&manifest_with_base(), actions)
        });

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
