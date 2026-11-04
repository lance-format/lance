// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Restamp the row-version metadata of rows rewritten in place.

use super::apply::ApplyState;
use super::row_selection::row_selections_from_proto;
use super::{Coordinate, Footprint, Ref, RowSelection, SelectedRows};
use crate::format::pb_actions;
use crate::rowids::version::{
    refresh_row_latest_update_meta_for_full_frag_rewrite_cols,
    refresh_row_latest_update_meta_for_partial_frag_rewrite_cols,
};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Record that the selected rows were updated by this operation.
///
/// What it writes: each selected row's `last_updated_at_version` becomes the
/// commit version. A [`SelectedRows::Full`] selection replaces the fragment's
/// sequence, inline or spilled to a column, with an inline uniform one. A
/// [`SelectedRows::Offsets`] selection keeps every other row's version and
/// writes the result inline. `created_at_version` is untouched: the rows are
/// the same rows, and a row minted by this operation gets both stamps from the
/// [`AddFragment`](super::AddFragment) that minted it.
///
/// A fragment with no last-updated sequence, such as one migrated to stable
/// row ids, gets one from an offsets selection, with every unselected row at
/// version 1, the version scans read for missing lineage. The legacy partial
/// update skips such a fragment instead, leaving its restamped rows
/// indistinguishable from the rest; [`AddOverlays`](super::AddOverlays) fills
/// the same way this does.
///
/// Why it is an action rather than implied by the data files the operation
/// writes: how many rows a column rewrite restamps is not a function of the
/// file. An update in rewrite-columns mode, and merge-insert's partial-fragment
/// path, write a whole-fragment column file but restamp only the matched rows;
/// a key-only merge-insert match restamps rows with no file at all; a `Merge`
/// restamps every row on removing a file or changing its path.
/// [`AddOverlays`](super::AddOverlays) can imply its restamp only because its
/// coverage bitmap is exactly the set of changed rows.
///
/// Fragments are named by committed id. A fragment this operation minted has
/// nothing to restamp.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct RefreshRowVersionMetadata {
    /// The rows to restamp, one selection per committed fragment. An offsets
    /// selection must be non-empty and below the fragment's physical row count.
    pub rows: Vec<RowSelection>,
}

impl RefreshRowVersionMetadata {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        if !self.rows.is_empty() && !state.uses_stable_row_ids() {
            return Err(Error::invalid_input(
                "RefreshRowVersionMetadata restamps the per-row version sequences, which only \
                 exist on a dataset using stable row ids",
            ));
        }
        let new_version = state.new_version();
        let spilled = state.spilled_row_lineage();
        for selection in &self.rows {
            let fragment_id =
                state.resolve_fragment(Ref::Committed(u64::from(selection.fragment_id)))?;
            let fragment = state.fragment_mut(fragment_id, "RefreshRowVersionMetadata")?;
            match &selection.rows {
                SelectedRows::Full => {
                    refresh_row_latest_update_meta_for_full_frag_rewrite_cols(
                        fragment,
                        new_version,
                    )?;
                }
                SelectedRows::Offsets(offsets) => {
                    let Some(max_offset) = offsets.max() else {
                        return Err(Error::invalid_input(format!(
                            "RefreshRowVersionMetadata selects no rows of fragment \
                             {fragment_id}; select the full fragment or at least one row offset"
                        )));
                    };
                    let physical_rows = fragment.physical_rows.ok_or_else(|| {
                        Error::invalid_input(format!(
                            "RefreshRowVersionMetadata selects row offsets of fragment \
                             {fragment_id}, which does not record its physical row count, so \
                             the offsets cannot be checked; select the full fragment instead"
                        ))
                    })?;
                    if max_offset as usize >= physical_rows {
                        return Err(Error::invalid_input(format!(
                            "RefreshRowVersionMetadata selects row offset {max_offset} of \
                             fragment {fragment_id}, which has {physical_rows} physical rows"
                        )));
                    }
                    let offsets = offsets
                        .iter()
                        .map(|offset| offset as usize)
                        .collect::<Vec<_>>();
                    refresh_row_latest_update_meta_for_partial_frag_rewrite_cols(
                        fragment,
                        &offsets,
                        new_version,
                        // The version scans read for a row with no sequence.
                        1,
                        spilled,
                    )?;
                }
            }
            state.note_refreshed(fragment_id);
        }
        Ok(())
    }

    /// The rows themselves are restated elsewhere in the operation -- by the
    /// data files it writes -- so this is bookkeeping about that change, not a
    /// change of its own.
    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    /// Each selected fragment's version sequence, whole, whatever rows the
    /// selection names. The sequence is one value per fragment, rewritten at
    /// once, so two sets refreshing the same fragment collide even when they
    /// select disjoint rows, or the column rewrites that prompted them touched
    /// different fields. Over-strict, and cheap: the alternative is a
    /// sequence that records which columns each row's version speaks for.
    ///
    /// An [`AddOverlays`](super::AddOverlays) restamps the same sequence
    /// without claiming it, so the two do not collide: the overlay sets only
    /// the rows it covers and keeps whatever the rest hold when it applies,
    /// which composes with a refresh in either order. Two refreshes would
    /// compose too -- each stamps only its own rows, and the later version
    /// wins where they overlap -- so their conflict is part of the same
    /// over-strictness, tracked in
    /// <https://github.com/lance-format/lance/issues/10189>.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        for selection in &self.rows {
            footprint.write(Coordinate::FragmentRowVersions(u64::from(
                selection.fragment_id,
            )));
        }
    }
}

impl From<&RefreshRowVersionMetadata> for pb_actions::RefreshRowVersionMetadata {
    fn from(value: &RefreshRowVersionMetadata) -> Self {
        Self {
            rows: value.rows.iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<pb_actions::RefreshRowVersionMetadata> for RefreshRowVersionMetadata {
    type Error = Error;

    fn try_from(message: pb_actions::RefreshRowVersionMetadata) -> Result<Self> {
        Ok(Self {
            rows: row_selections_from_proto(message.rows, "RefreshRowVersionMetadata")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{
        DataFile, Fragment, Manifest, ROW_LAST_UPDATED_AT_VERSION_FIELD_ID, RowIdMeta,
    };
    use crate::rowids::version::{
        LoadedRowLineage, RowDatasetVersionMeta, RowDatasetVersionSequence, SpilledRowLineage,
    };
    use crate::rowids::{RowIdSequence, write_row_ids};
    use crate::transaction::action::test_support::{
        apply, backed_manifest, footprint, one_column_file,
    };
    use crate::transaction::action::{
        Action, AddDataFile, AddFragment, CompositeOperation, FieldRef, Ref, RemoveFragment,
        TombstoneFieldData, UserStep,
    };
    use crate::transaction::test_support::{default_build_config, make_stable_row_id_manifest};
    use crate::transaction::{Operation, Transaction};
    use lance_file::version::ConcreteFileVersion;
    use roaring::RoaringBitmap;
    use rstest::rstest;
    use std::sync::Arc;

    /// Stands for the version the commit produces in an expected sequence.
    const NEW: u64 = 0;

    /// Refresh every row of each fragment.
    fn refresh(fragment_ids: &[u32]) -> Action {
        Action::RefreshRowVersionMetadata(RefreshRowVersionMetadata {
            rows: fragment_ids
                .iter()
                .map(|fragment_id| RowSelection {
                    fragment_id: *fragment_id,
                    rows: SelectedRows::Full,
                })
                .collect(),
        })
    }

    /// Refresh the rows of fragment `fragment_id` at `offsets`.
    fn refresh_rows(fragment_id: u32, offsets: &[u32]) -> Action {
        Action::RefreshRowVersionMetadata(RefreshRowVersionMetadata {
            rows: vec![RowSelection {
                fragment_id,
                rows: SelectedRows::Offsets(offsets.iter().copied().collect()),
            }],
        })
    }

    /// A stable-row-id manifest whose fragment 0 has `rows` rows, all last
    /// updated at version 1.
    fn manifest_with_rows(rows: usize) -> Manifest {
        let row_ids = RowIdSequence::from((0..rows as u64).collect::<Vec<_>>().as_slice());
        let mut fragment = Fragment {
            id: 0,
            files: vec![DataFile::new(
                "data.lance",
                vec![0],
                vec![0],
                ConcreteFileVersion::V2_0,
                None,
                None,
            )],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&row_ids).into())),
            physical_rows: Some(rows),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            clustering_id: None,
            metadata: Default::default(),
        };
        refresh_row_latest_update_meta_for_full_frag_rewrite_cols(&mut fragment, 1).unwrap();
        make_stable_row_id_manifest(vec![fragment])
    }

    /// `manifest_with_rows(3)`, with fragment 0 last updated at versions 2, 3
    /// and 4, or with no last-updated sequence at all.
    fn manifest_with_versions(has_sequence: bool) -> Manifest {
        let mut manifest = manifest_with_rows(3);
        let mut fragment = manifest.fragments[0].clone();
        fragment.last_updated_at_version_meta = has_sequence.then(|| {
            RowDatasetVersionMeta::from_sequence(&RowDatasetVersionSequence::from_versions(&[
                2, 3, 4,
            ]))
            .unwrap()
        });
        manifest.fragments = Arc::new(vec![fragment]);
        manifest
    }

    /// `expected`, with [`NEW`] replaced by the version `manifest`'s commit
    /// produces.
    fn with_new_version(manifest: &Manifest, expected: &[u64]) -> Vec<u64> {
        expected
            .iter()
            .map(|&version| {
                if version == NEW {
                    manifest.version + 1
                } else {
                    version
                }
            })
            .collect()
    }

    fn last_updated_versions(fragment: &Fragment) -> Vec<u64> {
        let meta = fragment
            .last_updated_at_version_meta
            .as_ref()
            .expect("the fragment should carry a last-updated sequence");
        let sequence = meta.load_sequence().unwrap();
        (0..fragment.physical_rows.unwrap())
            .map(|offset| sequence.version_at(offset).unwrap())
            .collect()
    }

    #[test]
    fn test_refresh_stamps_every_row_with_the_version_the_commit_produces() {
        let manifest = manifest_with_rows(3);
        let expected = manifest.version + 1;

        let out = apply(&manifest, vec![refresh(&[0])]).unwrap();

        assert_eq!(last_updated_versions(&out.fragments[0]), vec![expected; 3]);
    }

    /// A selection restamps its rows and leaves every other row at the version
    /// it had. A fragment with no sequence reads as version 1, so its
    /// unselected rows get exactly that.
    #[rstest]
    #[case::full(true, SelectedRows::Full, &[NEW, NEW, NEW])]
    #[case::one_row(true, SelectedRows::Offsets(RoaringBitmap::from_iter([1u32])), &[2, NEW, 4])]
    #[case::two_rows(
        true,
        SelectedRows::Offsets(RoaringBitmap::from_iter([0u32, 2])),
        &[NEW, 3, NEW]
    )]
    #[case::full_without_a_sequence(false, SelectedRows::Full, &[NEW, NEW, NEW])]
    #[case::one_row_without_a_sequence(
        false,
        SelectedRows::Offsets(RoaringBitmap::from_iter([1u32])),
        &[1, NEW, 1]
    )]
    fn test_refresh_stamps_only_the_selected_rows(
        #[case] has_sequence: bool,
        #[case] rows: SelectedRows,
        #[case] expected: &[u64],
    ) {
        let manifest = manifest_with_versions(has_sequence);
        let action = Action::RefreshRowVersionMetadata(RefreshRowVersionMetadata {
            rows: vec![RowSelection {
                fragment_id: 0,
                rows,
            }],
        });

        let out = apply(&manifest, vec![action]).unwrap();

        assert_eq!(
            last_updated_versions(&out.fragments[0]),
            with_new_version(&manifest, expected)
        );
    }

    /// A fragment whose last-updated versions are spilled to a data file has
    /// them read ahead of the commit; a partial restamp keeps the loaded
    /// versions of the rows it does not select, and refuses rather than
    /// guessing without them. A full restamp needs none of them.
    #[rstest]
    #[case::offsets_loaded(refresh_rows(0, &[0]), true, Ok(&[NEW, 3, 4][..]))]
    #[case::offsets_not_loaded(refresh_rows(0, &[0]), false, Err("not loaded ahead of the commit"))]
    #[case::full_not_loaded(refresh(&[0]), false, Ok(&[NEW, NEW, NEW][..]))]
    fn test_refresh_restamps_lineage_spilled_out_of_the_manifest(
        #[case] action: Action,
        #[case] is_loaded: bool,
        #[case] expected: std::result::Result<&[u64], &str>,
    ) {
        let mut manifest = manifest_with_rows(3);
        let mut fragment = manifest.fragments[0].clone();
        fragment.last_updated_at_version_meta = Some(RowDatasetVersionMeta::Column);
        fragment.files.push(DataFile::new(
            "lineage.lance",
            vec![ROW_LAST_UPDATED_AT_VERSION_FIELD_ID],
            vec![1],
            ConcreteFileVersion::V2_0,
            None,
            None,
        ));
        manifest.fragments = Arc::new(vec![fragment]);
        let mut config = default_build_config();
        if is_loaded {
            let loaded = LoadedRowLineage {
                last_updated_at: Some(Arc::new(RowDatasetVersionSequence::from_versions(&[
                    2, 3, 4,
                ]))),
                ..Default::default()
            };
            config.spilled_row_lineage = Arc::new(SpilledRowLineage::from([(0, loaded)]));
        }
        let transaction = Transaction::new(
            manifest.version,
            Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
                "step",
                vec![action],
            )])),
            None,
        );

        let result = transaction.build_manifest(Some(&manifest), Vec::new(), "tx.txn", &config);

        match expected {
            Ok(expected) => {
                let (out, _) = result.unwrap();
                assert_eq!(
                    last_updated_versions(&out.fragments[0]),
                    with_new_version(&manifest, expected)
                );
            }
            Err(message) => {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::NotSupported { .. }), "{error:?}");
                assert!(error.to_string().contains(message), "{error}");
            }
        }
    }

    #[rstest]
    #[case::offset_past_the_end(
        Some(3),
        refresh_rows(0, &[1, 3]),
        "selects row offset 3 of fragment 0, which has 3 physical rows"
    )]
    #[case::no_rows(Some(3), refresh_rows(0, &[]), "selects no rows of fragment 0")]
    #[case::unknown_row_count(
        None,
        refresh_rows(0, &[1]),
        "fragment 0, which does not record its physical row count"
    )]
    fn test_refreshing_an_invalid_row_selection_is_rejected(
        #[case] physical_rows: Option<usize>,
        #[case] action: Action,
        #[case] expected: &str,
    ) {
        let mut manifest = manifest_with_rows(3);
        let mut fragment = manifest.fragments[0].clone();
        fragment.physical_rows = physical_rows;
        manifest.fragments = Arc::new(vec![fragment]);

        let error = apply(&manifest, vec![action]).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        let message = error.to_string();
        assert!(message.contains(expected), "{message}");
        assert!(message.contains("RefreshRowVersionMetadata"), "{message}");
    }

    #[test]
    fn test_refresh_leaves_the_created_at_sequence_alone() {
        let mut manifest = manifest_with_rows(3);
        let mut fragment = manifest.fragments[0].clone();
        fragment.created_at_version_meta = fragment.last_updated_at_version_meta.clone();
        let created_at = fragment.created_at_version_meta.clone();
        manifest.fragments = Arc::new(vec![fragment]);

        let out = apply(&manifest, vec![refresh_rows(0, &[1])]).unwrap();

        assert_eq!(out.fragments[0].created_at_version_meta, created_at);
    }

    #[test]
    fn test_refreshing_no_fragments_is_allowed_on_any_dataset() {
        // An empty list is what a writer emits when nothing was rewritten in
        // place, so it must not depend on stable row ids.
        apply(&backed_manifest(), vec![refresh(&[])]).unwrap();
    }

    #[test]
    fn test_refreshing_without_stable_row_ids_is_rejected() {
        let error = apply(&backed_manifest(), vec![refresh_rows(0, &[0])]).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("stable row ids"),
            "unexpected error: {error}"
        );
    }

    /// A fragment id the read version handed out but no longer holds, and
    /// one it never handed out, which may since name another writer's
    /// fragment.
    #[rstest]
    #[case::removed(
        Some(7),
        "RefreshRowVersionMetadata targets fragment 7, which does not exist"
    )]
    #[case::beyond_the_read_version(None, "fragment 7 by a Committed ref")]
    fn test_refreshing_a_fragment_that_is_not_there_is_rejected(
        #[case] fragment_mark: Option<u32>,
        #[case] expected: &str,
    ) {
        let mut manifest = manifest_with_rows(3);
        manifest.max_fragment_id = fragment_mark;

        let error = apply(&manifest, vec![refresh(&[7])]).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains(expected),
            "unexpected error: {error}"
        );
    }

    /// The whole sequence is claimed whatever the selection, so even disjoint
    /// rows of one fragment collide.
    #[rstest]
    #[case::full_and_full(refresh(&[0]), refresh(&[0, 1]), true)]
    #[case::full_and_rows(refresh(&[0]), refresh_rows(0, &[1]), true)]
    #[case::disjoint_rows(refresh_rows(0, &[0]), refresh_rows(0, &[1]), true)]
    #[case::other_fragment_full(refresh(&[0]), refresh(&[1]), false)]
    #[case::other_fragment_rows(refresh_rows(0, &[0]), refresh_rows(1, &[0]), false)]
    fn test_two_writers_refreshing_the_same_fragment_conflict(
        #[case] ours: Action,
        #[case] theirs: Action,
        #[case] is_conflict: bool,
    ) {
        let ours = footprint(vec![ours]);
        let theirs = footprint(vec![theirs]);

        assert_eq!(ours.conflicts_with(&theirs), is_conflict);
        assert_eq!(theirs.conflicts_with(&ours), is_conflict);
    }

    /// The version sequence lives in the fragment, so a concurrent removal of
    /// it leaves nothing to restamp, in either order.
    #[test]
    fn test_refreshing_a_fragment_a_concurrent_writer_removes_conflicts() {
        let ours = footprint(vec![refresh_rows(0, &[1])]);
        let removal = |fragment: u64| {
            footprint(vec![Action::RemoveFragment(RemoveFragment {
                fragment: Ref::Committed(fragment),
                data_change: true,
            })])
        };

        assert!(ours.conflicts_with(&removal(0)));
        assert!(removal(0).conflicts_with(&ours));
        assert!(!ours.conflicts_with(&removal(1)));
        assert!(!removal(1).conflicts_with(&ours));
    }

    fn rewrite_field_zero(fragment: Ref, data_change: bool) -> Vec<Action> {
        let mut actions = Vec::new();
        if fragment.committed().is_some() {
            actions.push(Action::TombstoneFieldData(TombstoneFieldData {
                fragment,
                field_ids: vec![FieldRef::Committed(0)],
                data_change,
            }));
        }
        actions.push(Action::AddDataFile(AddDataFile {
            fragment,
            file: one_column_file("data/rewritten.lance"),
            field_ids: vec![FieldRef::Committed(0)],
            data_change,
            base: None,
        }));
        actions
    }

    fn append() -> Vec<Action> {
        let mut actions = vec![Action::AddFragment(AddFragment {
            id: Ref::Local(0),
            physical_rows: 3,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        })];
        actions.extend(rewrite_field_zero(Ref::Local(0), true));
        actions
    }

    /// Which rows a change restamps is not implied by the file it writes, so
    /// apply does not derive it: a data change written into a fragment that
    /// was already there has to say so with a refresh, or its rows keep the
    /// versions from before the change and a reader of changes since then
    /// misses them. A fragment the operation adds is stamped as a whole
    /// anyway.
    #[rstest]
    #[case::rewrite_without_refresh(
        true,
        rewrite_field_zero(Ref::Committed(0), true),
        Some("fragment 0")
    )]
    #[case::rewrite_with_refresh(
        true,
        [rewrite_field_zero(Ref::Committed(0), true), vec![refresh(&[0])]].concat(),
        None
    )]
    #[case::rewrite_with_partial_refresh(
        true,
        [rewrite_field_zero(Ref::Committed(0), true), vec![refresh_rows(0, &[1])]].concat(),
        None
    )]
    #[case::refresh_before_the_rewrite(
        true,
        [vec![refresh(&[0])], rewrite_field_zero(Ref::Committed(0), true)].concat(),
        None
    )]
    #[case::rearrangement(true, rewrite_field_zero(Ref::Committed(0), false), None)]
    #[case::fragment_added_in_the_operation(true, append(), None)]
    #[case::no_stable_row_ids(false, rewrite_field_zero(Ref::Committed(0), true), None)]
    fn test_a_data_change_to_an_existing_fragment_needs_a_refresh(
        #[case] stable_row_ids: bool,
        #[case] actions: Vec<Action>,
        #[case] expected_error: Option<&str>,
    ) {
        let manifest = if stable_row_ids {
            manifest_with_rows(3)
        } else {
            backed_manifest()
        };

        let result = apply(&manifest, actions);

        let Some(expected_error) = expected_error else {
            result.unwrap();
            return;
        };
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        let message = error.to_string();
        assert!(message.contains(expected_error), "{message}");
        assert!(message.contains("RefreshRowVersionMetadata"), "{message}");
    }
}
