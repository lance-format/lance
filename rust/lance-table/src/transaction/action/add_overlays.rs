// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Append overlay files to a fragment.

use super::apply::ApplyState;
use super::proto::{data_change_from_wire, data_change_to_wire, required};
use super::{FieldRef, Footprint, Ref};
use crate::format::overlay::DataOverlayFile;
use crate::format::pb;
use crate::format::pb_actions;
use crate::rowids::version::refresh_row_latest_update_meta_for_partial_frag_rewrite_cols;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;

/// Append overlay files to one fragment, supplying new values for a subset of
/// its `(row offset, field)` cells without rewriting its base data files.
///
/// Overlays are appended after the ones already on the fragment rather than
/// replacing them, so overlays a concurrent writer added survive, and later
/// entries take precedence.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddOverlays {
    /// The fragment to overlay: committed, or one minted earlier in the same
    /// operation.
    pub fragment: Ref,
    /// The overlays to append, in precedence order: where two cover the same
    /// cell, the later one wins. Each one's `committed_version` is ignored and
    /// stamped with the version this commit produces, so replaying the action
    /// onto a newer version re-stamps rather than backdates.
    pub overlays: Vec<DataOverlayFile>,
    /// Must be `true`; `false` is rejected at apply, since an overlay always
    /// changes values. See [`AddFragment::data_change`](super::AddFragment::data_change).
    pub data_change: bool,
}

impl AddOverlays {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        if !self.data_change {
            return Err(Error::invalid_input(
                "AddOverlays cannot set data_change = false: an overlay supplies values that \
                 differ from the ones it covers, so there is no restatement to describe, and \
                 the flag would only hide the change from consumers that skip non-data commits",
            ));
        }
        // The same refusal the legacy DataOverlay meets on such a table: the
        // read path does not yet mask overlaid cells out of segments that
        // translate into the fragment.
        if state.carries_tagged_history() {
            return Err(Error::not_supported(
                "AddOverlays is not supported on a table with a tagged fragment reuse history \
                 yet; trim the history or rebuild the indices that translate through it first",
            ));
        }
        let fragment_id = state.resolve_fragment(self.fragment)?;
        let committed_version = state.new_version();
        let refreshes_lineage = state.uses_stable_row_ids() && self.fragment.committed().is_some();
        let spilled = state.spilled_row_lineage();
        let fragment = state.fragment_mut(fragment_id, "AddOverlays")?;

        let mut covered = RoaringBitmap::new();
        for overlay in &self.overlays {
            for position in 0..overlay.data_file.fields.len() {
                covered |= overlay.coverage_for_field(position)?.as_ref();
            }
        }
        // Only checkable when the fragment records its row count, which
        // fragments written by older Lance versions may not.
        if let (Some(max_offset), Some(physical_rows)) = (covered.max(), fragment.physical_rows)
            && max_offset as usize >= physical_rows
        {
            return Err(Error::invalid_input(format!(
                "AddOverlays covers row offset {max_offset} of fragment {fragment_id}, which has \
                 {physical_rows} rows"
            )));
        }

        fragment
            .overlays
            .extend(self.overlays.iter().cloned().map(|mut overlay| {
                overlay.committed_version = committed_version;
                overlay
            }));

        // The overlaid cells changed at this version, which incremental
        // consumers find through the rows' last-updated versions. A fragment
        // this operation minted takes its lineage from the `AddFragment` that
        // minted it.
        if !refreshes_lineage {
            return Ok(());
        }
        let offsets = covered
            .iter()
            .map(|offset| offset as usize)
            .collect::<Vec<_>>();
        refresh_row_latest_update_meta_for_partial_frag_rewrite_cols(
            fragment,
            &offsets,
            committed_version,
            // A fragment with no sequence, such as one migrated to stable row
            // ids, gets one. Scans read missing lineage as version 1, so that
            // keeps every uncovered row where it was, as the legacy path does.
            1,
            spilled,
        )
    }

    /// An overlay supplies new cell values, so it changes what a reader sees.
    /// `data_change = false` is rejected at apply; see there.
    pub(super) fn is_data_change(&self) -> bool {
        self.data_change
    }

    /// A partial write of each overlaid field's data in the fragment, a
    /// requirement on each of those fields' definitions, and a partial write of
    /// the fragment's deletions.
    ///
    /// Partial, because two concurrent overlays over the same cells both land
    /// and the newer `committed_version` decides which value wins. A full
    /// rewrite of the same column is another matter: it replaces every cell
    /// from a snapshot that never saw this overlay, and
    /// [`TombstoneFieldData`](super::TombstoneFieldData) tombstones the overlay
    /// as it applies, so the overlay's values would be lost without a trace.
    /// That pair conflicts in either order, as the legacy Update-vs-DataOverlay
    /// check already does.
    ///
    /// The definition is required for the same reason a data file requires it:
    /// the overlay's values are encoded in the type the schema names, and a
    /// concurrent drop landing first would leave them over a field the manifest
    /// no longer has. A fragment this operation mints still requires it, since
    /// the overlay may supply a field the fragment's base files do not.
    ///
    /// The overlay leaves the fragment's deletions alone, but marks them as a
    /// conflict point (see [`Footprint::write_part_of_deletions`]): the marker
    /// collides with any full deletion write on the fragment, in either order,
    /// and commutes with other overlays. It exists for the row-moving update
    /// race. An update deletes rows from the fragment and re-adds them in a new
    /// fragment built from the pre-overlay base, so the overlay's values for
    /// those rows are lost whichever commits first. A plain delete conflicts
    /// too, because the footprint cannot tell it from a row-moving update.
    ///
    /// TODO(https://github.com/lance-format/lance/issues/10189): that is an
    /// over-conflict next to the legacy resolver's row-level check of moved
    /// rows against overlay coverage; restore it as part of conflict parity.
    ///
    /// A fragment this operation mints has no cells or deletions a concurrent
    /// writer can name.
    ///
    /// The rows' last-updated versions are restamped without a claim on
    /// [`Coordinate::FragmentRowVersions`](super::Coordinate::FragmentRowVersions);
    /// see [`RefreshRowVersionMetadata`](super::RefreshRowVersionMetadata)'s
    /// footprint for why.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        let fields = self
            .overlays
            .iter()
            .flat_map(|overlay| overlay.data_file.fields.iter().copied())
            // A negative id marks a tombstoned column in the overlay's file,
            // which holds no values a reader sees.
            .filter(|field| *field >= 0)
            .map(FieldRef::Committed);
        footprint.write_partial_field_data(self.fragment, fields);
        if let Some(fragment) = self.fragment.committed() {
            footprint.write_part_of_deletions(fragment);
        }
    }
}

impl From<&AddOverlays> for pb_actions::AddOverlays {
    fn from(value: &AddOverlays) -> Self {
        Self {
            fragment: Some(value.fragment.into()),
            overlays: value
                .overlays
                .iter()
                .map(pb::DataOverlayFile::from)
                .collect(),
            data_change: data_change_to_wire(value.data_change),
        }
    }
}

impl TryFrom<pb_actions::AddOverlays> for AddOverlays {
    type Error = Error;

    fn try_from(message: pb_actions::AddOverlays) -> Result<Self> {
        Ok(Self {
            fragment: required(message.fragment, "AddOverlays.fragment")?.try_into()?,
            overlays: message
                .overlays
                .into_iter()
                .map(DataOverlayFile::try_from)
                .collect::<Result<Vec<_>>>()?,
            data_change: data_change_from_wire(message.data_change),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::overlay::OverlayCoverage;
    use crate::format::{DataFile, Manifest, ROW_LAST_UPDATED_AT_VERSION_FIELD_ID, RowIdMeta};
    use crate::rowids::version::{
        LoadedRowLineage, RowDatasetVersionMeta, RowDatasetVersionSequence, SpilledRowLineage,
        refresh_row_latest_update_meta_for_full_frag_rewrite_cols,
    };
    use crate::rowids::{RowIdSequence, write_row_ids};
    use crate::transaction::action::test_support::{
        added_field, apply, backed_manifest, footprint,
    };
    use crate::transaction::action::{
        Action, AddField, AddFragment, AddIndexSegment, AdjustIndexCoverage, AlterField,
        CompositeOperation, DropField, FieldRef, RemoveFragment, SetDeletionFile,
        TombstoneFieldData, UserStep,
    };
    use crate::transaction::test_support::{default_build_config, make_stable_row_id_manifest};
    use crate::transaction::{Operation, Transaction};
    use lance_file::version::ConcreteFileVersion;
    use roaring::RoaringBitmap;
    use std::sync::Arc;

    fn overlay(path: &str, offsets: &[u32]) -> DataOverlayFile {
        DataOverlayFile {
            data_file: DataFile::new(
                path,
                vec![0],
                vec![0],
                ConcreteFileVersion::V2_0,
                None,
                None,
            ),
            coverage: OverlayCoverage::Shared(Arc::new(
                offsets.iter().copied().collect::<RoaringBitmap>(),
            )),
            // Whatever the writer left here is overwritten at apply.
            committed_version: 0,
        }
    }

    fn add(fragment: Ref, overlays: Vec<DataOverlayFile>) -> Action {
        Action::AddOverlays(AddOverlays {
            fragment,
            overlays,
            data_change: true,
        })
    }

    fn mint(physical_rows: u64) -> Action {
        Action::AddFragment(AddFragment {
            id: Ref::Local(0),
            physical_rows,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        })
    }

    /// A stable-row-id manifest whose one fragment holds three rows, with or
    /// without a last-updated sequence (a fragment migrated to stable row ids
    /// has none).
    fn stable_row_id_manifest(has_sequence: bool) -> Manifest {
        let mut fragment = backed_manifest().fragments[0].clone();
        fragment.physical_rows = Some(3);
        let ids = RowIdSequence::from([0u64, 1, 2].as_slice());
        fragment.row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&ids).into()));
        if has_sequence {
            refresh_row_latest_update_meta_for_full_frag_rewrite_cols(&mut fragment, 1).unwrap();
        }
        make_stable_row_id_manifest(vec![fragment])
    }

    #[test]
    fn test_overlays_are_stamped_with_the_version_the_commit_produces() {
        let manifest = backed_manifest();
        let expected = manifest.version + 1;

        let out = apply(
            &manifest,
            vec![add(Ref::Committed(0), vec![overlay("a.lance", &[1, 2])])],
        )
        .unwrap();

        let overlays = &out.fragments[0].overlays;
        assert_eq!(overlays.len(), 1);
        assert_eq!(overlays[0].committed_version, expected);
    }

    #[test]
    fn test_overlays_are_appended_to_the_ones_already_there() {
        let mut manifest = backed_manifest();
        let mut fragment = manifest.fragments[0].clone();
        fragment.overlays.push(overlay("old.lance", &[0]));
        manifest.fragments = Arc::new(vec![fragment]);

        let out = apply(
            &manifest,
            vec![add(Ref::Committed(0), vec![overlay("new.lance", &[1])])],
        )
        .unwrap();

        let paths = out.fragments[0]
            .overlays
            .iter()
            .map(|overlay| overlay.data_file.path.as_str())
            .collect::<Vec<_>>();
        assert_eq!(paths, vec!["old.lance", "new.lance"]);
    }

    #[test]
    fn test_several_overlays_keep_the_order_they_were_given_in() {
        let out = apply(
            &backed_manifest(),
            vec![add(
                Ref::Committed(0),
                vec![overlay("first.lance", &[0]), overlay("second.lance", &[0])],
            )],
        )
        .unwrap();

        let paths = out.fragments[0]
            .overlays
            .iter()
            .map(|overlay| overlay.data_file.path.as_str())
            .collect::<Vec<_>>();
        assert_eq!(paths, vec!["first.lance", "second.lance"]);
    }

    #[test]
    fn test_an_overlay_can_target_a_fragment_this_operation_minted() {
        let out = apply(
            &backed_manifest(),
            vec![
                Action::AddFragment(AddFragment {
                    id: Ref::Local(0),
                    physical_rows: 4,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                }),
                add(Ref::Local(0), vec![overlay("a.lance", &[0])]),
            ],
        )
        .unwrap();

        let minted = out.fragments.iter().find(|f| f.id == 1).unwrap();
        assert_eq!(minted.overlays.len(), 1);
    }

    #[test]
    fn test_overlaying_a_fragment_that_is_not_there_is_rejected() {
        // Handed out, so a committed ref may name it, but not in the dataset.
        let mut manifest = backed_manifest();
        manifest.max_fragment_id = Some(42);
        let error = apply(
            &manifest,
            vec![add(Ref::Committed(42), vec![overlay("a.lance", &[0])])],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error
                .to_string()
                .contains("AddOverlays targets fragment 42, which does not exist"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_two_writers_overlaying_the_same_fragment_do_not_conflict() {
        let ours = footprint(vec![add(Ref::Committed(0), vec![overlay("a.lance", &[0])])]);
        let theirs = footprint(vec![add(Ref::Committed(0), vec![overlay("b.lance", &[0])])]);

        assert!(!ours.conflicts_with(&theirs));
    }

    /// A full rewrite of the overlaid column replaces every cell from a
    /// snapshot that never saw the overlay, and tombstones the overlay as it
    /// applies; the overlay's values would be gone without a trace. Either
    /// order loses, so the pair is symmetric.
    #[test]
    fn test_an_overlay_and_a_rewrite_of_the_same_column_conflict() {
        let overlaid = footprint(vec![add(Ref::Committed(0), vec![overlay("a.lance", &[0])])]);
        let rewrite = |field: i32| {
            footprint(vec![Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(0),
                field_ids: vec![FieldRef::Committed(field)],
                data_change: true,
            })])
        };

        assert!(overlaid.conflicts_with(&rewrite(0)));
        assert!(rewrite(0).conflicts_with(&overlaid));
        // Another column of the same fragment is not touched by the overlay.
        assert!(!overlaid.conflicts_with(&rewrite(1)));
        assert!(!rewrite(1).conflicts_with(&overlaid));
    }

    /// An overlay landing after its field was altered or dropped would carry
    /// values checked against a definition the schema no longer holds, or for
    /// a field that is gone. Landing first, the alter covers the overlay's
    /// values like any others and the drop tombstones the overlay, so those
    /// orders are fine.
    #[test]
    fn test_an_overlay_needs_its_field_defined_as_it_was() {
        let overlaid = footprint(vec![add(Ref::Committed(0), vec![overlay("a.lance", &[0])])]);
        let altered = footprint(vec![Action::AlterField(AlterField {
            field: FieldRef::Committed(0),
            name: Some("renamed".into()),
            nullable: None,
        })]);
        let dropped = footprint(vec![Action::DropField(DropField {
            field: FieldRef::Committed(0),
        })]);

        assert!(overlaid.conflicts_with(&altered));
        assert!(!altered.conflicts_with(&overlaid));
        assert!(overlaid.conflicts_with(&dropped));
        assert!(!dropped.conflicts_with(&overlaid));
    }

    /// An index built over a column an overlay landed on first is accepted:
    /// the segment is stamped with the version it read, the overlay carries
    /// the later version it committed at, and the read path masks the
    /// overlaid cells out of any segment older than the overlay
    /// (`collect_overlay_stale_frags`). A requirement therefore does not
    /// collide with a committed partial write, only with a full one.
    #[test]
    fn test_an_index_may_be_built_over_a_committed_overlay() {
        let build = footprint(vec![Action::AddIndexSegment(AddIndexSegment {
            uuid: uuid::Uuid::from_u128(1),
            name: "by_a".into(),
            fields: vec![FieldRef::Committed(0)],
            covering_fields: Vec::new(),
            index_details: None,
            index_version: 1,
            covered_fragments: Some(vec![Ref::Committed(0)]),
            files: Vec::new(),
            base: None,
            created_at: None,
            dataset_version: None,
            data_change: false,
        })]);
        let overlaid = footprint(vec![add(Ref::Committed(0), vec![overlay("a.lance", &[0])])]);

        assert!(!build.conflicts_with(&overlaid));
        assert!(!overlaid.conflicts_with(&build));
    }

    /// One set may overlay a cell and then rewrite the whole column in the
    /// same operation (an overlay being materialized). The full write is what
    /// a concurrent set sees: another overlay of the column no longer
    /// commutes with it, while a rewrite of a different column still does.
    #[test]
    fn test_a_full_write_in_the_same_set_dominates_a_partial_one() {
        let materialize = footprint(vec![
            add(Ref::Committed(0), vec![overlay("a.lance", &[0])]),
            Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(0),
                field_ids: vec![FieldRef::Committed(0)],
                data_change: true,
            }),
        ]);
        let other_overlay = footprint(vec![add(Ref::Committed(0), vec![overlay("b.lance", &[0])])]);
        let other_column = footprint(vec![Action::TombstoneFieldData(TombstoneFieldData {
            fragment: Ref::Committed(0),
            field_ids: vec![FieldRef::Committed(1)],
            data_change: true,
        })]);

        assert!(materialize.conflicts_with(&other_overlay));
        assert!(other_overlay.conflicts_with(&materialize));
        assert!(!materialize.conflicts_with(&other_column));
        assert!(!other_column.conflicts_with(&materialize));
    }

    /// Overlaid cells have changed since the version a consumer last read, so
    /// the rows' last-updated versions have to say so, leaving every row the
    /// overlay does not cover at the version it already had. A fragment with no
    /// sequence, such as one migrated to stable row ids, reads as version 1.
    #[rstest::rstest]
    #[case::with_a_sequence(true)]
    #[case::without_a_sequence(false)]
    fn test_overlaid_rows_are_restamped_with_the_version_the_commit_produces(
        #[case] has_sequence: bool,
    ) {
        let manifest = stable_row_id_manifest(has_sequence);
        let expected = manifest.version + 1;

        let out = apply(
            &manifest,
            vec![add(Ref::Committed(0), vec![overlay("new.lance", &[0])])],
        )
        .unwrap();

        let versions = out.fragments[0]
            .last_updated_at_version_meta
            .as_ref()
            .unwrap()
            .load_sequence()
            .unwrap();
        assert_eq!(
            versions.versions().collect::<Vec<_>>(),
            vec![expected, 1, 1]
        );
    }

    #[test]
    fn test_an_overlay_leaves_lineage_alone_without_stable_row_ids() {
        let out = apply(
            &backed_manifest(),
            vec![add(Ref::Committed(0), vec![overlay("a.lance", &[0])])],
        )
        .unwrap();

        assert_eq!(out.fragments[0].last_updated_at_version_meta, None);
    }

    /// A fragment this operation mints takes its lineage from the
    /// `AddFragment` that minted it, which may carry versions from before this
    /// commit -- an update's rewritten rows do. An overlay in the same
    /// operation leaves that alone.
    #[test]
    fn test_an_overlay_keeps_the_lineage_a_minted_fragment_was_given() {
        let minted_with =
            RowDatasetVersionMeta::from_sequence(&RowDatasetVersionSequence::from_versions(&[
                2, 2, 2,
            ]))
            .unwrap();
        let Action::AddFragment(mut minted) = mint(3) else {
            unreachable!()
        };
        minted.last_updated_at_version_meta = Some(minted_with.clone());
        minted.created_at_version_meta = Some(minted_with.clone());

        let out = apply(
            &stable_row_id_manifest(true),
            vec![
                Action::AddFragment(minted),
                add(Ref::Local(0), vec![overlay("a.lance", &[0])]),
            ],
        )
        .unwrap();

        let minted = out.fragments.iter().find(|f| f.id != 0).unwrap();
        assert_eq!(minted.last_updated_at_version_meta, Some(minted_with));
    }

    /// A fragment whose last-updated versions are spilled to a data file has
    /// them read ahead of the commit; the restamp keeps the loaded versions of
    /// the rows it does not cover. Without them it refuses rather than
    /// guessing.
    #[rstest::rstest]
    #[case::loaded(true)]
    #[case::not_loaded(false)]
    fn test_an_overlay_restamps_lineage_spilled_out_of_the_manifest(#[case] is_loaded: bool) {
        let mut manifest = stable_row_id_manifest(false);
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
        let expected = manifest.version + 1;
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
                vec![add(Ref::Committed(0), vec![overlay("new.lance", &[0])])],
            )])),
            None,
        );

        let result = transaction.build_manifest(Some(&manifest), Vec::new(), "tx.txn", &config);

        if is_loaded {
            let (out, _) = result.unwrap();
            let versions = out.fragments[0]
                .last_updated_at_version_meta
                .as_ref()
                .unwrap()
                .load_sequence()
                .unwrap();
            assert_eq!(
                versions.versions().collect::<Vec<_>>(),
                vec![expected, 3, 4]
            );
        } else {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::NotSupported { .. }), "{error:?}");
            assert!(
                error.to_string().contains("not loaded ahead of the commit"),
                "{error}"
            );
        }
    }

    /// A negative id marks a tombstoned column in the overlay's file. It holds
    /// nothing a reader sees, so it claims nothing: only the live column
    /// collides with a rewrite.
    #[test]
    fn test_a_tombstoned_column_in_an_overlay_claims_nothing() {
        let mut partly_tombstoned = overlay("a.lance", &[0]);
        partly_tombstoned.data_file.fields = vec![-1, 1].into();
        partly_tombstoned.data_file.column_indices = vec![0, 1].into();
        let overlaid = footprint(vec![add(Ref::Committed(0), vec![partly_tombstoned])]);
        let rewrite = |field: i32| {
            footprint(vec![Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(0),
                field_ids: vec![FieldRef::Committed(field)],
                data_change: true,
            })])
        };
        let drop_field_zero = footprint(vec![Action::DropField(DropField {
            field: FieldRef::Committed(0),
        })]);

        assert!(overlaid.conflicts_with(&rewrite(1)));
        assert!(rewrite(1).conflicts_with(&overlaid));
        for other in [rewrite(0), drop_field_zero] {
            assert!(!overlaid.conflicts_with(&other));
            assert!(!other.conflicts_with(&overlaid));
        }
    }

    /// An overlay covering a row past the end of its fragment supplies values
    /// for a row that does not exist, whatever the dataset and whether the
    /// fragment is committed or minted in the same operation.
    #[rstest::rstest]
    #[case::without_stable_row_ids(backed_manifest(), Ref::Committed(0), 10)]
    #[case::with_stable_row_ids(stable_row_id_manifest(true), Ref::Committed(0), 3)]
    #[case::minted_fragment(backed_manifest(), Ref::Local(0), 4)]
    fn test_an_overlay_past_the_end_of_its_fragment_is_rejected(
        #[case] manifest: Manifest,
        #[case] fragment: Ref,
        #[case] offset: u32,
    ) {
        let mut actions = Vec::new();
        if fragment.committed().is_none() {
            actions.push(mint(4));
        }
        actions.push(add(fragment, vec![overlay("a.lance", &[0, offset])]));

        let error = apply(&manifest, actions).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error
                .to_string()
                .contains(&format!("covers row offset {offset}")),
            "{error}"
        );
    }

    /// An overlay always supplies values that differ from what was there, so
    /// there is no restatement for `data_change = false` to describe; it would
    /// only hide the change from consumers that skip non-data commits.
    #[test]
    fn test_an_overlay_cannot_be_marked_as_no_data_change() {
        let error = apply(
            &backed_manifest(),
            vec![Action::AddOverlays(AddOverlays {
                fragment: Ref::Committed(0),
                overlays: vec![overlay("a.lance", &[0])],
                data_change: false,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("data_change"), "{error}");
    }

    /// The overlay's values are encoded in the type the schema names, whether or
    /// not the fragment it lands in is one this operation minted: a stale append
    /// carrying an overlay for a field a concurrent set dropped would leave
    /// values for a field the schema no longer has.
    ///
    /// Landing second, a drop tombstones the field out of the new fragment, but
    /// a replacement (a cast) is only valid against the rows it read.
    #[rstest::rstest]
    #[case::replacement(
        vec![
            Action::DropField(DropField { field: FieldRef::Committed(0) }),
            Action::AddField(AddField {
                local: 0,
                parent: None,
                def: added_field("id"),
            }),
        ],
        true,
    )]
    #[case::drop(vec![Action::DropField(DropField { field: FieldRef::Committed(0) })], false)]
    fn test_an_overlay_on_a_minted_fragment_needs_its_field_defined_as_it_was(
        #[case] schema_change: Vec<Action>,
        #[case] schema_change_conflicts_landing_second: bool,
    ) {
        let overlaid = footprint(vec![
            Action::AddFragment(AddFragment {
                id: Ref::Local(0),
                physical_rows: 4,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            }),
            add(Ref::Local(0), vec![overlay("new.lance", &[0])]),
        ]);
        let schema_change = footprint(schema_change);

        assert!(overlaid.conflicts_with(&schema_change));
        assert_eq!(
            schema_change.conflicts_with(&overlaid),
            schema_change_conflicts_landing_second
        );
    }

    /// A concurrent row-moving update deletes rows from the fragment and
    /// re-creates them in a new one built from the pre-overlay base, so the
    /// overlay's values for them would be lost. The footprint cannot tell a
    /// move from a plain delete, so any deletion-file change on the fragment
    /// collides, in both orders -- not a requirement, because the update
    /// landing second is the lossy order. Two overlays still commute.
    #[test]
    fn test_an_overlay_conflicts_with_a_deletion_of_the_same_fragment() {
        let delete = |fragment: u64| {
            footprint(vec![Action::SetDeletionFile(SetDeletionFile {
                fragment,
                deletion_file: None,
                data_change: true,
                base: None,
            })])
        };
        let overlaid = footprint(vec![add(Ref::Committed(0), vec![overlay("a.lance", &[0])])]);

        assert!(overlaid.conflicts_with(&delete(0)));
        assert!(delete(0).conflicts_with(&overlaid));
        assert!(!overlaid.conflicts_with(&delete(1)));
        assert!(!delete(1).conflicts_with(&overlaid));
    }

    #[test]
    fn test_an_overlay_on_a_minted_fragment_does_not_conflict_with_a_deletion() {
        let overlaid = footprint(vec![
            Action::AddFragment(AddFragment {
                id: Ref::Local(0),
                physical_rows: 4,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            }),
            add(Ref::Local(0), vec![overlay("new.lance", &[0])]),
        ]);
        let delete = footprint(vec![Action::SetDeletionFile(SetDeletionFile {
            fragment: 0,
            deletion_file: None,
            data_change: true,
            base: None,
        })]);

        assert!(!overlaid.conflicts_with(&delete));
        assert!(!delete.conflicts_with(&overlaid));
    }

    /// An overlay writes values for its field, on any fragment, so a set that
    /// is valid only while the field holds no new values cannot land after it.
    #[rstest::rstest]
    #[case::committed_fragment(Ref::Committed(0))]
    #[case::minted_fragment(Ref::Local(0))]
    fn test_a_tightening_cannot_land_after_an_overlay_of_its_field(#[case] fragment: Ref) {
        let mut actions = Vec::new();
        if fragment.committed().is_none() {
            actions.push(Action::AddFragment(AddFragment {
                id: fragment,
                physical_rows: 4,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            }));
        }
        actions.push(add(fragment, vec![overlay("a.lance", &[0])]));
        let overlaid = footprint(actions);
        let tighten = footprint(vec![Action::AlterField(AlterField {
            field: FieldRef::Committed(0),
            name: None,
            nullable: Some(false),
        })]);

        assert!(tighten.conflicts_with(&overlaid));
    }

    #[test]
    fn test_overlaying_a_fragment_a_concurrent_writer_removes_conflicts() {
        let ours = footprint(vec![add(Ref::Committed(0), vec![overlay("a.lance", &[0])])]);
        let theirs = footprint(vec![Action::RemoveFragment(RemoveFragment {
            fragment: Ref::Committed(0),
            data_change: true,
        })]);

        assert!(ours.conflicts_with(&theirs));
        assert!(theirs.conflicts_with(&ours));
    }

    /// Unlike a rewrite, an overlay does not invalidate a widened segment: a
    /// read masks an overlay committed after the segment's `dataset_version`,
    /// so the index never answers from the values it replaced.
    #[test]
    fn test_an_overlay_does_not_conflict_with_a_widening_onto_its_fragment() {
        let widen = footprint(vec![Action::AdjustIndexCoverage(AdjustIndexCoverage {
            uuid: uuid::Uuid::from_u128(1),
            name: "by_a".into(),
            add_fragments: vec![Ref::Committed(7)],
            remove_fragments: Vec::new(),
            fields: vec![FieldRef::Committed(0)],
            covering_fields: Vec::new(),
        })]);
        let overlay = footprint(vec![add(Ref::Committed(7), vec![overlay("a.lance", &[0])])]);

        assert!(!widen.conflicts_with(&overlay));
        assert!(!overlay.conflicts_with(&widen));
    }
}
