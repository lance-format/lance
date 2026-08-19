// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Add a new, empty fragment.

use super::apply::ApplyState;
use super::proto::{data_change_from_wire, data_change_to_wire, required};
use super::{Coordinate, Footprint, Ref};
use crate::format::{ExternalFile, Fragment, RowIdMeta, pb};
use crate::rowids::version::RowDatasetVersionMeta;
use lance_core::Result;
use lance_core::deepsize::DeepSizeOf;

/// Add a new, empty fragment.
///
/// Its data files arrive via [`AddDataFile`](super::AddDataFile) actions naming
/// this fragment by the same [`Ref`]. A newly added fragment has no deletion
/// vector: it has no committed rows to delete yet.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddFragment {
    /// The id to add the fragment at.
    ///
    /// [`Ref::Local`] is the usual form: a token standing in for an id
    /// allocated at apply, which re-allocates when the operation is replayed
    /// onto a newer version.
    ///
    /// [`Ref::Committed`] names an id out of a range an earlier commit reserved
    /// with [`ReserveFragmentIds`](super::ReserveFragmentIds). It exists for a
    /// writer that has to know the id before it commits, because it bakes row
    /// addresses into an index it writes in the same commit. Such an id is the
    /// writer's alone and stays valid at every later version, so it does not
    /// move under replay -- but it is also not the writer's to choose freely:
    /// apply rejects an id that is not reserved or is already in use.
    pub id: Ref,
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
        let id = state.add_fragment_id(self.id)?;
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

    /// No coordinate for a local token: the fragment does not exist in the read
    /// version, so no concurrent writer can be naming it.
    ///
    /// A committed id does write one. A reservation is meant to be one writer's
    /// alone, but nothing in the format enforces that, so two operations handed
    /// the same range would otherwise each add a fragment at the same id and
    /// the second commit would silently win.
    ///
    /// Either form records that rows arrive, which is the one thing about an
    /// added fragment a concurrent
    /// [`AssertUniqueKeys`](super::AssertUniqueKeys) has to know. A fragment
    /// that is not a data change holds rows that were already in the dataset --
    /// a compaction rewrite -- and brings in no new key.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        if let Some(id) = self.id.committed() {
            footprint.add(Coordinate::FragmentExistence(id));
        }
        if self.data_change {
            footprint.insert_rows();
        }
    }
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
            id: Some(value.id.into()),
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
            id: required(message.id, "AddFragment.id")?.try_into()?,
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
    use crate::transaction::action::{Action, AddDataFile, ReserveFragmentIds};
    use crate::transaction::test_support::sample_manifest;
    use lance_file::version::ConcreteFileVersion;

    fn add_fragment(id: Ref) -> Action {
        Action::AddFragment(AddFragment {
            id,
            physical_rows: 10,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        })
    }

    #[test]
    fn test_add_fragment_and_data_file_mint_ids() {
        let manifest = sample_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddFragment(AddFragment {
                    id: Ref::Local(0),
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

    #[test]
    fn test_a_fragment_can_be_added_at_a_reserved_id() {
        // What a writer that needs the id up front does: reserve in one commit,
        // read the high-water mark back, then name the id in the next.
        let reserved = apply(
            &sample_manifest(),
            vec![Action::ReserveFragmentIds(ReserveFragmentIds { count: 2 })],
        )
        .unwrap();
        assert_eq!(reserved.max_fragment_id(), Some(2));

        let next = apply(&reserved, vec![add_fragment(Ref::Committed(1))]).unwrap();

        assert_eq!(
            next.fragments.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec![0, 1]
        );
        // The rest of the reservation stays reserved: the high-water mark does
        // not fall back to the ids actually in use.
        assert_eq!(next.max_fragment_id(), Some(2));
    }

    #[test]
    fn test_an_unreserved_id_is_rejected() {
        // sample_manifest holds fragment 0 and nothing is reserved, so 1 is the
        // next id a mint would hand out -- claiming it would collide.
        let error = apply(&sample_manifest(), vec![add_fragment(Ref::Committed(1))]).unwrap_err();

        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(
            error.to_string().contains("no commit has reserved"),
            "{error}"
        );
    }

    #[test]
    fn test_an_id_already_in_use_is_rejected() {
        let reserved = apply(
            &sample_manifest(),
            vec![Action::ReserveFragmentIds(ReserveFragmentIds { count: 2 })],
        )
        .unwrap();

        // Reserved, but claimed twice in the one operation.
        let error = apply(
            &reserved,
            vec![
                add_fragment(Ref::Committed(1)),
                add_fragment(Ref::Committed(1)),
            ],
        )
        .unwrap_err();

        assert!(error.to_string().contains("already exists"), "{error}");
    }

    #[test]
    fn test_a_reserved_id_can_back_a_data_file_in_the_same_operation() {
        let reserved = apply(
            &sample_manifest(),
            vec![Action::ReserveFragmentIds(ReserveFragmentIds { count: 1 })],
        )
        .unwrap();

        let next = apply(
            &reserved,
            vec![
                add_fragment(Ref::Committed(1)),
                Action::AddDataFile(AddDataFile {
                    // The same reference the AddFragment carried, so a writer
                    // does not need a local token alongside the reserved id.
                    fragment: Ref::Committed(1),
                    file: DataFile::new_unstarted("data/new.lance", ConcreteFileVersion::V2_0),
                    field_ids: vec![Ref::Committed(0)],
                    data_change: true,
                }),
            ],
        )
        .unwrap();

        let added = next.fragments.iter().find(|f| f.id == 1).unwrap();
        assert_eq!(added.files.len(), 1);
        assert_eq!(added.files[0].fields.as_ref(), &[0]);
    }
}
