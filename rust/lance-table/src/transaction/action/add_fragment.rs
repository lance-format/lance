// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Add a new, empty fragment.

use super::apply::ApplyState;
use super::proto::required;
use super::{Coordinate, Footprint, Ref};
use crate::format::{Fragment, RowIdMeta, pb, pb_actions};
use crate::rowids::read_row_ids;
use crate::rowids::version::RowDatasetVersionMeta;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// The contents of an [`AddFragment`], before its caller certifies them with
/// [`AddFragment::new_unchecked`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddFragmentParts {
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
    /// `false` marks a pure rearrangement, e.g. a compaction rewrite. On a
    /// dataset with stable row ids such a fragment must carry all three
    /// lineage sequences, since its rows keep their identity. No action in the
    /// same operation may change such a fragment's values.
    pub data_change: bool,
}

/// Add a new, empty fragment.
///
/// Its data files arrive via [`AddDataFile`](super::AddDataFile) actions naming
/// this fragment by the same [`Ref`]. A newly added fragment has no deletion
/// vector: it has no committed rows to delete yet.
///
/// The fragment's row count and lineage are claims about the files the
/// operation binds to it, so it can only be built with
/// [`new_unchecked`](Self::new_unchecked).
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddFragment {
    parts: AddFragmentParts,
}

impl AddFragment {
    /// Add a fragment described by `parts`.
    ///
    /// # Contract
    ///
    /// The caller certifies, and nothing downstream checks, that:
    ///
    /// * Every data file the operation binds to the fragment holds exactly
    ///   `physical_rows` rows.
    /// * A lineage sequence given as [`Column`](RowIdMeta::Column) is held by
    ///   the file the operation binds that column to, one entry per row.
    /// * If `data_change` is `false`, every row is one already in the dataset,
    ///   moved here unchanged, and any lineage sequence given is the one the
    ///   row already had.
    ///
    /// Breaking the contract is not undefined behavior, but it commits a table
    /// whose reads fail or return the wrong rows.
    pub fn new_unchecked(parts: AddFragmentParts) -> Self {
        Self { parts }
    }

    /// What the fragment was built from.
    pub fn parts(&self) -> &AddFragmentParts {
        &self.parts
    }

    /// Take the parts back, e.g. to change one and certify the result again
    /// with [`new_unchecked`](Self::new_unchecked).
    pub fn into_parts(self) -> AddFragmentParts {
        self.parts
    }

    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        if !self.parts.data_change && state.uses_stable_row_ids() {
            self.check_carries_lineage()?;
        }
        let id = state.add_fragment_id(self.parts.id)?;
        if !self.parts.data_change {
            state.note_rearranged(id);
        }
        state.push_fragment(Fragment {
            id,
            files: Vec::new(),
            overlays: Vec::new(),
            deletion_file: None,
            row_id_meta: self.parts.row_id_meta.clone(),
            physical_rows: Some(self.parts.physical_rows as usize),
            last_updated_at_version_meta: self.parts.last_updated_at_version_meta.clone(),
            created_at_version_meta: self.parts.created_at_version_meta.clone(),
            clustering_id: None,
            metadata: Default::default(),
        });
        Ok(())
    }

    /// A fragment that only moves rows keeps their identity, so it has to
    /// carry every lineage sequence: apply stamps whatever is left out as if
    /// the rows were new, with fresh row ids and this commit's version.
    fn check_carries_lineage(&self) -> Result<()> {
        let missing = [
            (self.parts.row_id_meta.is_none(), "row ids"),
            (
                self.parts.last_updated_at_version_meta.is_none(),
                "last_updated_at versions",
            ),
            (
                self.parts.created_at_version_meta.is_none(),
                "created_at versions",
            ),
        ]
        .into_iter()
        .filter_map(|(is_missing, sequence)| is_missing.then_some(sequence))
        .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(());
        }
        Err(Error::invalid_input(format!(
            "AddFragment {:?} has data_change = false on a dataset with stable row ids but does \
             not carry its {}; a fragment that only moves rows must carry their row id \
             sequence and both row version sequences, inline or as a column, or be marked as \
             a data change",
            self.parts.id,
            missing.join(", ")
        )))
    }

    /// See [`AddFragmentParts::data_change`].
    pub fn is_data_change(&self) -> bool {
        self.parts.data_change
    }

    /// No coordinate for a local token: the fragment does not exist in the read
    /// version, so no concurrent writer can be naming it.
    ///
    /// A committed id does write one. A reservation is meant to be one writer's
    /// alone, but nothing in the format enforces that, so two operations handed
    /// the same range would otherwise each add a fragment at the same id and
    /// the second commit would silently win.
    ///
    /// Row ids the fragment arrives carrying are claimed as an exact set, so
    /// two operations spending one reservation collide. A sequence that does
    /// not decode makes the footprint opaque: apply rejects it anyway, and a
    /// footprint must not under-report. A spilled sequence lives in a file
    /// this cannot read, and goes unclaimed.
    ///
    /// Either form records that rows arrive, which is the one thing about an
    /// added fragment a concurrent
    /// [`AssertUniqueKeys`](super::AssertUniqueKeys) has to know. A fragment
    /// that is not a data change holds rows that were already in the dataset --
    /// a compaction rewrite -- and brings in no new key.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.add_rows();
        if let Some(id) = self.parts.id.committed() {
            footprint.write(Coordinate::FragmentExistence(id));
        }
        if let Some(RowIdMeta::Inline(data)) = &self.parts.row_id_meta {
            match read_row_ids(data) {
                Ok(sequence) => footprint.supply_row_ids(&sequence.to_treemap()),
                Err(_) => footprint.mark_opaque(),
            }
        }
        if self.parts.data_change {
            footprint.insert_rows();
        }
    }
}

impl From<&AddFragment> for pb_actions::AddFragment {
    fn from(value: &AddFragment) -> Self {
        Self {
            id: Some(value.parts.id.into()),
            physical_rows: value.parts.physical_rows,
            row_id_sequence: value.parts.row_id_meta.as_ref().map(|meta| match meta {
                RowIdMeta::Inline(data) => {
                    pb_actions::add_fragment::RowIdSequence::InlineRowIds(data.to_vec())
                }
                RowIdMeta::Column => pb_actions::add_fragment::RowIdSequence::ColumnRowIds(
                    pb::RowLineageColumn {},
                ),
            }),
            last_updated_at_version_sequence: value
                .parts
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
                .parts
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
            data_unchanged: !value.parts.data_change,
        }
    }
}

impl TryFrom<pb_actions::AddFragment> for AddFragment {
    type Error = lance_core::Error;

    fn try_from(message: pb_actions::AddFragment) -> Result<Self> {
        // A committed transaction file carries the certification of the
        // writer that staged it.
        Ok(Self::new_unchecked(AddFragmentParts {
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
            data_change: !message.data_unchanged,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::IndexMetadata;
    use crate::rowids::version::RowDatasetVersionSequence;
    use crate::rowids::write_row_ids;
    use crate::transaction::action::test_support::{
        apply, apply_with_indices, one_column_file, unbacked_manifest,
    };
    use crate::transaction::action::{Action, AddDataFile, FieldRef, ReserveFragmentIds};
    use crate::transaction::test_support::{make_stable_row_id_manifest, sample_index_metadata};
    use roaring::RoaringBitmap;
    use rstest::rstest;

    fn add_fragment(id: Ref) -> Action {
        Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
            id,
            physical_rows: 10,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        }))
    }

    #[test]
    fn test_add_fragment_and_data_file_mint_ids() {
        let manifest = unbacked_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
                    id: Ref::Local(0),
                    physical_rows: 10,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                })),
                Action::AddDataFile(
                    AddDataFile::try_new_unchecked(
                        Ref::Local(0),
                        one_column_file("data/new.lance"),
                        vec![FieldRef::Committed(0)],
                        None,
                        true,
                    )
                    .unwrap(),
                ),
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
            &unbacked_manifest(),
            vec![Action::ReserveFragmentIds(ReserveFragmentIds::new(2))],
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
        let error = apply(&unbacked_manifest(), vec![add_fragment(Ref::Committed(1))]).unwrap_err();

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
        Action::ReserveFragmentIds(ReserveFragmentIds::new(2)),
        add_fragment(Ref::Committed(1)),
    ])]
    #[case::claim_then_reserve(vec![
        add_fragment(Ref::Committed(1)),
        Action::ReserveFragmentIds(ReserveFragmentIds::new(2)),
    ])]
    fn test_an_id_reserved_in_the_same_operation_cannot_be_claimed(#[case] actions: Vec<Action>) {
        let error = apply(&unbacked_manifest(), actions).unwrap_err();

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
            &unbacked_manifest(),
            vec![Action::ReserveFragmentIds(ReserveFragmentIds::new(2))],
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
            &unbacked_manifest(),
            vec![Action::ReserveFragmentIds(ReserveFragmentIds::new(2))],
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
            &unbacked_manifest(),
            vec![Action::ReserveFragmentIds(ReserveFragmentIds::new(1))],
        )
        .unwrap();

        let next = apply(
            &reserved,
            vec![
                add_fragment(Ref::Committed(1)),
                Action::AddDataFile(
                    // The same reference the AddFragment carried, so a writer
                    // does not need a local token alongside the reserved id.
                    AddDataFile::try_new_unchecked(
                        Ref::Committed(1),
                        one_column_file("data/new.lance"),
                        vec![FieldRef::Committed(0)],
                        None,
                        true,
                    )
                    .unwrap(),
                ),
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
        Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
            id: Ref::Local(0),
            physical_rows: 10,
            row_id_meta: row_ids.then(|| RowIdMeta::Inline(write_row_ids(&(0..10).into()).into())),
            last_updated_at_version_meta: last_updated.then(versions),
            created_at_version_meta: created.then(versions),
            data_change,
        }))
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
            let mut manifest = make_stable_row_id_manifest(vec![]);
            // The moved fragment carries no data, which a required column forbids.
            manifest.schema.field_by_id_mut(0).unwrap().nullable = true;
            manifest
        } else {
            unbacked_manifest()
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

    fn rearranged_data_file(data_change: bool) -> Action {
        Action::AddDataFile(
            AddDataFile::try_new_unchecked(
                Ref::Local(0),
                one_column_file("data/moved.lance"),
                vec![FieldRef::Committed(0)],
                None,
                data_change,
            )
            .unwrap(),
        )
    }

    /// A fragment added with `data_change = false` holds rows that were
    /// already in the table, carrying their old row versions, and adds no row
    /// a key assertion has to rule out. A write in the same operation that
    /// changes those rows' values would hide behind both: an incremental
    /// reader would miss the change, and a concurrent key assertion would not
    /// see the changed key.
    #[rstest]
    #[case::an_overlay(vec![
        rearranged_data_file(false),
        Action::AddOverlays(crate::transaction::action::AddOverlays::new_unchecked(
            Ref::Local(0),
            vec![crate::format::overlay::DataOverlayFile {
                data_file: crate::format::DataFile::new(
                    "data/overlay.lance",
                    vec![0],
                    vec![0],
                    lance_file::version::ConcreteFileVersion::V2_0,
                    None,
                    None,
                ),
                coverage: crate::format::overlay::OverlayCoverage::dense(
                    RoaringBitmap::from_iter([0u32]),
                ),
                committed_version: 0,
            }],
        )),
    ], "AddOverlays")]
    #[case::a_changed_data_file(vec![rearranged_data_file(true)], "AddDataFile")]
    #[case::a_changing_tombstone(vec![
        rearranged_data_file(false),
        Action::TombstoneFieldData(crate::transaction::action::TombstoneFieldData::new(
            Ref::Local(0),
            vec![FieldRef::Committed(0)],
        )),
    ], "TombstoneFieldData")]
    fn test_a_rearranged_fragment_takes_no_data_change(
        #[case] writes: Vec<Action>,
        #[case] action: &str,
    ) {
        let mut add = add_fragment(Ref::Local(0));
        if let Action::AddFragment(fragment) = &mut add {
            *fragment = AddFragment::new_unchecked(AddFragmentParts {
                data_change: false,
                ..fragment.parts().clone()
            });
        }
        let mut actions = vec![add];
        actions.extend(writes);

        let error = apply(&unbacked_manifest(), actions).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        let message = error.to_string();
        assert!(message.contains(action), "{message}");
        assert!(
            message.contains("which this operation adds with data_change = false"),
            "{message}"
        );
    }
}
