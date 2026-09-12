// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Set (replace) a fragment's deletion file.

use super::apply::ApplyState;
use super::proto::{data_change_from_wire, data_change_to_wire};
use super::{Coordinate, Footprint};
use crate::format::{DeletionFile, pb};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Set (replace) a fragment's deletion file.
///
/// This is reference-stable rather than a delta: the fragment id is committed
/// and physical row offsets never move, so the post-image is unambiguous. The
/// newly-deleted rows -- the delta rebase and conflict detection need -- are
/// derived by diffing against the read version rather than serialized.
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
}

impl SetDeletionFile {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        state
            .fragment_mut(self.fragment, "SetDeletionFile")?
            .deletion_file = self.deletion_file.clone();
        Ok(())
    }

    pub(super) fn is_data_change(&self) -> bool {
        self.data_change
    }

    /// The fragment's deletions, which is a distinct coordinate from the data of
    /// any field in it: deleting rows and re-encoding a column commute.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.add(Coordinate::FragmentDeletions(self.fragment));
    }
}

impl From<&SetDeletionFile> for pb::SetDeletionFile {
    fn from(value: &SetDeletionFile) -> Self {
        Self {
            fragment: value.fragment,
            deletion_file: value.deletion_file.as_ref().map(pb::DeletionFile::from),
            data_change: data_change_to_wire(value.data_change),
        }
    }
}

impl TryFrom<pb::SetDeletionFile> for SetDeletionFile {
    type Error = Error;

    fn try_from(message: pb::SetDeletionFile) -> Result<Self> {
        Ok(Self {
            fragment: message.fragment,
            deletion_file: message
                .deletion_file
                .map(DeletionFile::try_from)
                .transpose()?,
            data_change: data_change_from_wire(message.data_change),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DeletionFileType;
    use crate::transaction::action::Action;
    use crate::transaction::action::test_support::{apply, backed_manifest};

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
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
    }
}
