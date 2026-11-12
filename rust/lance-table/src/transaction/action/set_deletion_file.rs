// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Set (replace) a fragment's deletion file.

use super::apply::ApplyState;
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
    fragment: u64,
    deletion_file: Option<DeletionFile>,
    data_change: bool,
    base: Option<Ref>,
}

impl SetDeletionFile {
    /// Clear the deletions of committed fragment `fragment`, bringing back
    /// every row it had deleted.
    pub fn clear(fragment: u64) -> Self {
        Self {
            fragment,
            deletion_file: None,
            data_change: true,
            base: None,
        }
    }

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
            deletion_file: Some(deletion_file),
            data_change,
            base,
        })
    }

    /// The committed fragment whose deletions are set.
    pub fn fragment(&self) -> u64 {
        self.fragment
    }

    /// The new deletion file, with `base_id` unset, or `None` when the action
    /// clears the fragment's deletions.
    pub fn deletion_file(&self) -> Option<&DeletionFile> {
        self.deletion_file.as_ref()
    }

    /// The base path the deletion file lives under, or `None` for the
    /// dataset's own directory.
    pub fn base(&self) -> Option<Ref> {
        self.base
    }

    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let deletion_file = self
            .deletion_file
            .as_ref()
            .map(|file| {
                Ok::<_, Error>(DeletionFile {
                    base_id: self.base.map(|base| state.resolve_base(base)).transpose()?,
                    ..file.clone()
                })
            })
            .transpose()?;
        let fragment_id = state.resolve_fragment(Ref::Committed(self.fragment))?;
        state
            .fragment_mut(fragment_id, "SetDeletionFile")?
            .deletion_file = deletion_file;
        Ok(())
    }

    /// Whether a reader sees rows leave or come back.
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
            deletion_file: value.deletion_file.as_ref().map(pb::DeletionFile::from),
            data_unchanged: !value.data_change,
            base: value.base.map(Into::into),
        }
    }
}

impl TryFrom<pb_actions::SetDeletionFile> for SetDeletionFile {
    type Error = Error;

    fn try_from(message: pb_actions::SetDeletionFile) -> Result<Self> {
        let base = message.base.map(Ref::try_from).transpose()?;
        // A committed transaction file carries the certification of the
        // writer that staged it.
        match message.deletion_file {
            Some(file) => Self::try_new_unchecked(
                message.fragment,
                DeletionFile::try_from(file)?,
                base,
                !message.data_unchanged,
            ),
            None if base.is_some() => Err(Error::invalid_input(format!(
                "SetDeletionFile for fragment {} clears its deletion file but names base \
                 {base:?}; a base only says where a deletion file lives",
                message.fragment
            ))),
            None if message.data_unchanged => Err(Error::invalid_input(format!(
                "SetDeletionFile for fragment {} clears its deletion file with data_change \
                     = false; clearing brings deleted rows back, so it is always a data change",
                message.fragment
            ))),
            None => Ok(Self::clear(message.fragment)),
        }
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
            inline: None,
        };
        let next = apply(
            &manifest,
            vec![Action::SetDeletionFile(
                SetDeletionFile::try_new_unchecked(0, deletion_file.clone(), None, true).unwrap(),
            )],
        )
        .unwrap();
        assert_eq!(next.fragments[0].deletion_file, Some(deletion_file));

        // An absent deletion file is a request to clear it, not a no-op.
        let cleared = apply(
            &next,
            vec![Action::SetDeletionFile(SetDeletionFile::clear(0))],
        )
        .unwrap();
        assert_eq!(cleared.fragments[0].deletion_file, None);
    }

    #[test]
    fn test_set_deletion_file_rejects_a_missing_fragment() {
        let error = apply(
            &backed_manifest(),
            vec![Action::SetDeletionFile(SetDeletionFile::clear(7))],
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

    /// Clearing a deletion file brings rows back, and a base only says where a
    /// deletion file lives. The action cannot be built either way, so only a
    /// transaction file can say it.
    #[rstest]
    #[case::base_for_a_cleared_deletion_file(
        Some(pb_actions::Ref::from(Ref::Committed(1))),
        false,
        "names base"
    )]
    #[case::cleared_without_a_data_change(None, true, "always a data change")]
    fn test_a_cleared_deletion_file_is_rejected_on_decode(
        #[case] base: Option<pb_actions::Ref>,
        #[case] data_unchanged: bool,
        #[case] expected: &str,
    ) {
        let message = pb_actions::SetDeletionFile {
            fragment: 0,
            deletion_file: None,
            data_unchanged,
            base,
        };

        let error = SetDeletionFile::try_from(message).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains(expected),
            "unexpected error: {error}"
        );
    }
}
