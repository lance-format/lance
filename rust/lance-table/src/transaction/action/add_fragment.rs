// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Add a new, empty fragment.

use super::apply::ApplyState;
use super::proto::{data_change_from_wire, data_change_to_wire, required};
use super::{Coordinate, Footprint, Ref};
use crate::format::{Fragment, RowIdMeta, pb, pb_actions};
use crate::rowids::read_row_ids;
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
    /// [`Ref::Committed`] names an id out of a range a commit at or before the
    /// read version reserved with [`ReserveFragmentIds`](super::ReserveFragmentIds). It exists for a
    /// writer that has to know the id before it commits, because it bakes row
    /// addresses into an index it writes in the same commit. Such an id is the
    /// writer's alone and stays valid at every later version, so it does not
    /// move under replay -- but it is also not the writer's to choose freely:
    /// apply rejects an id that is not reserved or is already in use, and one
    /// an index segment's coverage still lists.
    ///
    /// Known limitation: an id claimed, then removed, then claimed again
    /// within one operation is not caught. Nothing from the first claim
    /// outlives the operation, so the result is still consistent.
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

    /// Nothing for a local token: the fragment does not exist in the read
    /// version, so no concurrent writer can be naming it.
    ///
    /// A committed id does write one coordinate. A reservation is meant to be
    /// one writer's alone, but nothing in the format enforces that, so two
    /// operations handed the same range would otherwise each add a fragment at
    /// the same id and the second commit would silently win.
    ///
    /// Row ids the fragment arrives carrying are claimed as an exact set, so
    /// two operations spending one reservation collide. A sequence that does
    /// not decode takes the table exclusively: apply rejects it anyway, and a
    /// footprint must not under-report. A spilled sequence lives in a file
    /// this cannot read, and goes unclaimed.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.add_rows();
        if let Some(id) = self.id.committed() {
            footprint.write(Coordinate::FragmentExistence(id));
        }
        if let Some(RowIdMeta::Inline(data)) = &self.row_id_meta {
            match read_row_ids(data) {
                Ok(sequence) => footprint.supply_row_ids(&sequence.to_treemap()),
                Err(_) => footprint.take_exclusive(),
            }
        }
    }
}

impl From<&AddFragment> for pb_actions::AddFragment {
    fn from(value: &AddFragment) -> Self {
        Self {
            id: Some(value.id.into()),
            physical_rows: value.physical_rows,
            row_id_sequence: value.row_id_meta.as_ref().map(|meta| match meta {
                RowIdMeta::Inline(data) => {
                    pb_actions::add_fragment::RowIdSequence::InlineRowIds(data.to_vec())
                }
                RowIdMeta::Column => pb_actions::add_fragment::RowIdSequence::ColumnRowIds(
                    pb::RowLineageColumn {},
                ),
            }),
            last_updated_at_version_sequence: value
                .last_updated_at_version_meta
                .as_ref()
                .map(|meta| match meta {
                    RowDatasetVersionMeta::Inline(data) => {
                        pb_actions::add_fragment::LastUpdatedAtVersionSequence::InlineLastUpdatedAtVersions(
                            data.to_vec(),
                        )
                    }
                    RowDatasetVersionMeta::Column => {
                        pb_actions::add_fragment::LastUpdatedAtVersionSequence::ColumnLastUpdatedAtVersions(
                            pb::RowLineageColumn {},
                        )
                    }
                }),
            created_at_version_sequence: value
                .created_at_version_meta
                .as_ref()
                .map(|meta| match meta {
                    RowDatasetVersionMeta::Inline(data) => {
                        pb_actions::add_fragment::CreatedAtVersionSequence::InlineCreatedAtVersions(
                            data.to_vec(),
                        )
                    }
                    RowDatasetVersionMeta::Column => {
                        pb_actions::add_fragment::CreatedAtVersionSequence::ColumnCreatedAtVersions(
                            pb::RowLineageColumn {},
                        )
                    }
                }),
            data_change: data_change_to_wire(value.data_change),
        }
    }
}

impl TryFrom<pb_actions::AddFragment> for AddFragment {
    type Error = lance_core::Error;

    fn try_from(message: pb_actions::AddFragment) -> Result<Self> {
        Ok(Self {
            id: required(message.id, "AddFragment.id")?.try_into()?,
            physical_rows: message.physical_rows,
            row_id_meta: message.row_id_sequence.map(|sequence| match sequence {
                pb_actions::add_fragment::RowIdSequence::InlineRowIds(data) => {
                    RowIdMeta::Inline(data.into())
                }
                pb_actions::add_fragment::RowIdSequence::ColumnRowIds(_) => RowIdMeta::Column,
            }),
            last_updated_at_version_meta: message.last_updated_at_version_sequence.map(
                |sequence| {
                    match sequence {
                    pb_actions::add_fragment::LastUpdatedAtVersionSequence::InlineLastUpdatedAtVersions(
                        data,
                    ) => RowDatasetVersionMeta::Inline(data.into()),
                    pb_actions::add_fragment::LastUpdatedAtVersionSequence::ColumnLastUpdatedAtVersions(
                        _,
                    ) => RowDatasetVersionMeta::Column,
                }
                },
            ),
            created_at_version_meta: message.created_at_version_sequence.map(|sequence| {
                match sequence {
                    pb_actions::add_fragment::CreatedAtVersionSequence::InlineCreatedAtVersions(data) => {
                        RowDatasetVersionMeta::Inline(data.into())
                    }
                    pb_actions::add_fragment::CreatedAtVersionSequence::ColumnCreatedAtVersions(_) => {
                        RowDatasetVersionMeta::Column
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
    use crate::format::IndexMetadata;
    use crate::rowids::version::RowDatasetVersionSequence;
    use crate::rowids::write_row_ids;
    use crate::transaction::action::test_support::{apply, apply_with_indices};
    use crate::transaction::action::{Action, AddDataFile, FieldRef, ReserveFragmentIds};
    use crate::transaction::test_support::{
        make_stable_row_id_manifest, sample_index_metadata, sample_manifest,
    };
    use lance_file::version::ConcreteFileVersion;
    use roaring::RoaringBitmap;
    use rstest::rstest;

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
                    field_ids: vec![FieldRef::Committed(0)],
                    data_change: true,
                    base: None,
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

    /// A reservation is taken after the operation's own mints, so an id it
    /// reserves is not yet reserved for the actions beside it.
    #[rstest]
    #[case::reserve_then_claim(vec![
        Action::ReserveFragmentIds(ReserveFragmentIds { count: 2 }),
        add_fragment(Ref::Committed(1)),
    ])]
    #[case::claim_then_reserve(vec![
        add_fragment(Ref::Committed(1)),
        Action::ReserveFragmentIds(ReserveFragmentIds { count: 2 }),
    ])]
    fn test_an_id_reserved_in_the_same_operation_cannot_be_claimed(#[case] actions: Vec<Action>) {
        let error = apply(&sample_manifest(), actions).unwrap_err();

        assert!(
            error.to_string().contains("no commit has reserved"),
            "{error}"
        );
    }

    /// Index coverage bitmaps are not pruned when a fragment is removed, and
    /// readers intersect them with the fragments that exist. A reserved id
    /// some segment still lists would read as indexed the moment it is
    /// claimed, though no index has seen its rows.
    #[rstest]
    #[case::listed_by_an_index(1, Some("index 'idx' segment"))]
    #[case::not_listed(2, None)]
    fn test_a_reserved_id_an_index_still_lists_cannot_be_claimed(
        #[case] id: u64,
        #[case] expected_error: Option<&str>,
    ) {
        let reserved = apply(
            &sample_manifest(),
            vec![Action::ReserveFragmentIds(ReserveFragmentIds { count: 2 })],
        )
        .unwrap();
        let stale_index = IndexMetadata {
            fragment_bitmap: Some(RoaringBitmap::from_iter([0u32, 1])),
            ..sample_index_metadata("idx")
        };

        let result = apply_with_indices(
            &reserved,
            vec![add_fragment(Ref::Committed(id))],
            vec![stale_index],
        );

        match expected_error {
            None => {
                result.unwrap();
            }
            Some(message) => {
                let error = result.unwrap_err();
                assert!(
                    matches!(error, lance_core::Error::InvalidInput { .. }),
                    "{error:?}"
                );
                assert!(error.to_string().contains(message), "{error}");
            }
        }
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
                    field_ids: vec![FieldRef::Committed(0)],
                    data_change: true,
                    base: None,
                }),
            ],
        )
        .unwrap();

        let added = next.fragments.iter().find(|f| f.id == 1).unwrap();
        assert_eq!(added.files.len(), 1);
        assert_eq!(added.files[0].fields.as_ref(), &[0]);
    }

    fn fragment_carrying(
        data_change: bool,
        row_ids: bool,
        last_updated: bool,
        created: bool,
    ) -> Action {
        let versions = || {
            RowDatasetVersionMeta::from_sequence(
                &RowDatasetVersionSequence::from_uniform_row_count(10, 1),
            )
            .unwrap()
        };
        Action::AddFragment(AddFragment {
            id: Ref::Local(0),
            physical_rows: 10,
            row_id_meta: row_ids.then(|| RowIdMeta::Inline(write_row_ids(&(0..10).into()).into())),
            last_updated_at_version_meta: last_updated.then(versions),
            created_at_version_meta: created.then(versions),
            data_change,
        })
    }

    /// A fragment that only moves rows (compaction) has to keep their
    /// identity: with a sequence left out, apply would mint fresh row ids or
    /// stamp the commit version, and the moved rows would read as new ones.
    /// A data change is stamped instead, and a dataset without stable row ids
    /// has no lineage to keep.
    #[rstest]
    #[case::move_with_every_sequence(true, fragment_carrying(false, true, true, true), None)]
    #[case::move_without_row_ids(
        true,
        fragment_carrying(false, false, true, true),
        Some("row ids")
    )]
    #[case::move_without_last_updated_versions(
        true,
        fragment_carrying(false, true, false, true),
        Some("last_updated_at versions")
    )]
    #[case::move_without_created_versions(
        true,
        fragment_carrying(false, true, true, false),
        Some("created_at versions")
    )]
    #[case::data_change_without_sequences(true, fragment_carrying(true, false, false, false), None)]
    #[case::move_without_stable_row_ids(false, fragment_carrying(false, false, false, false), None)]
    fn test_a_move_on_a_stable_row_id_dataset_carries_its_lineage(
        #[case] stable_row_ids: bool,
        #[case] action: Action,
        #[case] missing: Option<&str>,
    ) {
        let manifest = if stable_row_ids {
            make_stable_row_id_manifest(vec![])
        } else {
            sample_manifest()
        };

        let result = apply(&manifest, vec![action]);

        let Some(missing) = missing else {
            let next = result.unwrap();
            let added = next.fragments.iter().max_by_key(|f| f.id).unwrap();
            assert_eq!(added.row_id_meta.is_some(), stable_row_ids);
            assert_eq!(added.last_updated_at_version_meta.is_some(), stable_row_ids);
            assert_eq!(added.created_at_version_meta.is_some(), stable_row_ids);
            return;
        };
        let error = result.unwrap_err();
        assert!(
            matches!(error, lance_core::Error::InvalidInput { .. }),
            "{error:?}"
        );
        assert!(error.to_string().contains("data_change = false"), "{error}");
        assert!(error.to_string().contains(missing), "{error}");
    }
}
