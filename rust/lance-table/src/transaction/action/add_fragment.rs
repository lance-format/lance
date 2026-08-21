// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Mint a new, empty fragment.

use super::Footprint;
use super::apply::ApplyState;
use super::proto::{data_change_from_wire, data_change_to_wire};
use crate::format::{ExternalFile, Fragment, RowIdMeta, pb};
use crate::rowids::version::RowDatasetVersionMeta;
use lance_core::Result;
use lance_core::deepsize::DeepSizeOf;

/// Mint a new, empty fragment.
///
/// Its data files arrive via [`AddDataFile`](super::AddDataFile) actions naming
/// this fragment's local token. A freshly-minted fragment has no deletion
/// vector: it has no committed rows to delete yet.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddFragment {
    /// Token standing in for the fragment id until it is allocated at apply.
    pub local: u32,
    /// Physical rows in the fragment, including rows later tombstoned.
    pub physical_rows: u64,
    /// Stable row id sequence. `None` on datasets without stable row ids, and
    /// on datasets that have them but where the ids are assigned at apply.
    pub row_id_meta: Option<RowIdMeta>,
    /// Per-row version metadata, carried exactly as on
    /// [`Fragment`](crate::format::Fragment).
    ///
    /// `None` means "stamp at apply", which fills both with a uniform sequence
    /// at the commit version. That is right for an append, but not for the two
    /// producers whose rows carry versions from before this commit, so they set
    /// the fields explicitly: an update resolves each row's `created_at` from
    /// the fragment the row came from, and a compaction rechunks both sequences
    /// off the fragments it merged, since moving a row does not update it.
    pub last_updated_at_version_meta: Option<RowDatasetVersionMeta>,
    pub created_at_version_meta: Option<RowDatasetVersionMeta>,
    /// `false` marks a pure rearrangement, e.g. a compaction rewrite.
    pub data_change: bool,
}

impl AddFragment {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let id = state.mint_fragment(self.local)?;
        state.push_fragment(Fragment {
            id,
            files: Vec::new(),
            overlays: Vec::new(),
            deletion_file: None,
            row_id_meta: self.row_id_meta.clone(),
            physical_rows: Some(self.physical_rows as usize),
            last_updated_at_version_meta: self.last_updated_at_version_meta.clone(),
            created_at_version_meta: self.created_at_version_meta.clone(),
        });
        Ok(())
    }

    pub(super) fn is_data_change(&self) -> bool {
        self.data_change
    }

    /// Nothing: the fragment does not exist in the read version, so no
    /// concurrent writer can be naming it.
    pub(super) fn footprint(&self, _footprint: &mut Footprint) {}
}

fn external_file_to_wire(file: &ExternalFile) -> pb::ExternalFile {
    pb::ExternalFile {
        path: file.path.clone(),
        offset: file.offset,
        size: file.size,
    }
}

fn external_file_from_wire(file: pb::ExternalFile) -> ExternalFile {
    ExternalFile {
        path: file.path,
        offset: file.offset,
        size: file.size,
    }
}

impl From<&AddFragment> for pb::AddFragment {
    fn from(value: &AddFragment) -> Self {
        Self {
            local: value.local,
            physical_rows: value.physical_rows,
            row_id_sequence: value.row_id_meta.as_ref().map(|meta| match meta {
                RowIdMeta::Inline(data) => {
                    pb::add_fragment::RowIdSequence::InlineRowIds(data.to_vec())
                }
                RowIdMeta::External(file) => {
                    pb::add_fragment::RowIdSequence::ExternalRowIds(external_file_to_wire(file))
                }
            }),
            last_updated_at_version_sequence: value
                .last_updated_at_version_meta
                .as_ref()
                .map(|meta| match meta {
                    RowDatasetVersionMeta::Inline(data) => {
                        pb::add_fragment::LastUpdatedAtVersionSequence::InlineLastUpdatedAtVersions(
                            data.to_vec(),
                        )
                    }
                    RowDatasetVersionMeta::External(file) => {
                        pb::add_fragment::LastUpdatedAtVersionSequence::ExternalLastUpdatedAtVersions(
                            external_file_to_wire(file),
                        )
                    }
                }),
            created_at_version_sequence: value
                .created_at_version_meta
                .as_ref()
                .map(|meta| match meta {
                    RowDatasetVersionMeta::Inline(data) => {
                        pb::add_fragment::CreatedAtVersionSequence::InlineCreatedAtVersions(
                            data.to_vec(),
                        )
                    }
                    RowDatasetVersionMeta::External(file) => {
                        pb::add_fragment::CreatedAtVersionSequence::ExternalCreatedAtVersions(
                            external_file_to_wire(file),
                        )
                    }
                }),
            data_change: data_change_to_wire(value.data_change),
        }
    }
}

impl TryFrom<pb::AddFragment> for AddFragment {
    type Error = lance_core::Error;

    fn try_from(message: pb::AddFragment) -> Result<Self> {
        Ok(Self {
            local: message.local,
            physical_rows: message.physical_rows,
            row_id_meta: message.row_id_sequence.map(|sequence| match sequence {
                pb::add_fragment::RowIdSequence::InlineRowIds(data) => {
                    RowIdMeta::Inline(data.into())
                }
                pb::add_fragment::RowIdSequence::ExternalRowIds(file) => {
                    RowIdMeta::External(external_file_from_wire(file))
                }
            }),
            last_updated_at_version_meta: message.last_updated_at_version_sequence.map(
                |sequence| {
                    match sequence {
                    pb::add_fragment::LastUpdatedAtVersionSequence::InlineLastUpdatedAtVersions(
                        data,
                    ) => RowDatasetVersionMeta::Inline(data.into()),
                    pb::add_fragment::LastUpdatedAtVersionSequence::ExternalLastUpdatedAtVersions(
                        file,
                    ) => RowDatasetVersionMeta::External(external_file_from_wire(file)),
                }
                },
            ),
            created_at_version_meta: message.created_at_version_sequence.map(|sequence| {
                match sequence {
                    pb::add_fragment::CreatedAtVersionSequence::InlineCreatedAtVersions(data) => {
                        RowDatasetVersionMeta::Inline(data.into())
                    }
                    pb::add_fragment::CreatedAtVersionSequence::ExternalCreatedAtVersions(file) => {
                        RowDatasetVersionMeta::External(external_file_from_wire(file))
                    }
                }
            }),
            data_change: data_change_from_wire(message.data_change),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::transaction::action::test_support::apply;
    use crate::transaction::action::{Action, AddDataFile, Ref};
    use crate::transaction::test_support::sample_manifest;
    use lance_file::version::ConcreteFileVersion;

    #[test]
    fn test_add_fragment_and_data_file_mint_ids() {
        let manifest = sample_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddFragment(AddFragment {
                    local: 0,
                    physical_rows: 10,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                }),
                Action::AddDataFile(AddDataFile {
                    fragment: Ref::Local(0),
                    file: DataFile::new_unstarted("data/new.lance", ConcreteFileVersion::V2_0),
                    field_ids: vec![Ref::Committed(0)],
                    data_change: true,
                }),
            ],
        )
        .unwrap();

        // sample_manifest already holds fragment 0, so the mint lands on 1.
        assert_eq!(
            next.fragments.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec![0, 1]
        );
        let minted = next.fragments.iter().find(|f| f.id == 1).unwrap();
        assert_eq!(minted.physical_rows, Some(10));
        assert_eq!(minted.files.len(), 1);
        // The file's field list is stamped in from the action's refs.
        assert_eq!(minted.files[0].fields.as_ref(), &[0]);
        assert_eq!(next.max_fragment_id(), Some(1));
    }
}
