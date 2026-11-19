// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Append overlay files to a fragment.

use super::apply::ApplyState;
use super::proto::required;
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
    fragment: Ref,
    overlays: Vec<DataOverlayFile>,
}

impl AddOverlays {
    /// Append `overlays` to `fragment`.
    ///
    /// * `fragment` is committed, or one minted earlier in the same operation
    ///   with `data_change = true`.
    /// * `overlays` are in precedence order: where two cover the same cell, the
    ///   later one wins. Each one's `committed_version` is ignored and stamped
    ///   with the version this commit produces, so replaying the action onto a
    ///   newer version re-stamps rather than backdates.
    ///
    /// An overlay always changes values, so the action is always a data change.
    ///
    /// # Contract
    ///
    /// The caller certifies, and nothing downstream checks, that each overlay's
    /// file exists, is readable, and holds a value of its field's type for
    /// every cell its coverage names. Apply checks the coverage stays within
    /// the fragment's rows, not what the file holds.
    ///
    /// Breaking the contract is not undefined behavior, but it commits a table
    /// whose reads fail or return the wrong rows.
    pub fn new_unchecked(fragment: Ref, overlays: Vec<DataOverlayFile>) -> Self {
        Self { fragment, overlays }
    }

    /// The fragment overlaid.
    pub fn fragment(&self) -> Ref {
        self.fragment
    }

    /// The overlays appended, in precedence order.
    pub fn overlays(&self) -> &[DataOverlayFile] {
        &self.overlays
    }

    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
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
        state.reject_data_change_to_rearranged(fragment_id, "AddOverlays")?;
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
    pub fn is_data_change(&self) -> bool {
        true
    }

    /// The fragment, which must still be there with its rows where they were,
    /// and each overlaid field, which must still exist.
    ///
    /// Nothing else, because two concurrent overlays over the same cells both
    /// land and the newer `committed_version` decides which value wins. A
    /// re-encode of the column commutes with it too, since it changes no
    /// value. A data-changing rewrite of the same column does not: its new
    /// base values tombstone every overlay on the column as they land, so the
    /// two conflict in either order (see
    /// [`Footprint::supersede_overlays`](super::Footprint::supersede_overlays)).
    ///
    /// A concurrent drop landing first would leave the values over a field the
    /// manifest no longer has. A fragment this operation mints still requires
    /// the field, since the overlay may supply one the base files do not.
    ///
    /// A plain delete of overlaid rows leaves the overlay inert over them, so
    /// the two commute. A concurrent row-moving update does not: it re-creates
    /// the rows it moves in another fragment, from the base the overlay never
    /// touched, losing the overlay's values for them. Which rows moved is not
    /// checked here, so any move out of the fragment conflicts.
    ///
    /// The rows' last-updated versions are restamped at apply, setting only
    /// the rows the overlay covers, which composes with a refresh in either
    /// order.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        let fields = self
            .overlays
            .iter()
            .flat_map(|overlay| overlay.data_file.fields.iter().copied())
            // A negative id marks a tombstoned column in the overlay's file,
            // which holds no values a reader sees.
            .filter(|field| *field >= 0)
            .map(FieldRef::Committed);
        footprint.write_overlays(self.fragment, fields);
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
            data_unchanged: false,
        }
    }
}

impl TryFrom<pb_actions::AddOverlays> for AddOverlays {
    type Error = Error;

    fn try_from(message: pb_actions::AddOverlays) -> Result<Self> {
        if message.data_unchanged {
            return Err(Error::invalid_input(
                "AddOverlays cannot set data_change = false: an overlay supplies values that \
                 differ from the ones it covers, so there is no restatement to describe, and \
                 the flag would only hide the change from consumers that skip non-data commits",
            ));
        }
        // A committed transaction file carries the certification of the
        // writer that staged it.
        Ok(Self::new_unchecked(
            required(message.fragment, "AddOverlays.fragment")?.try_into()?,
            message
                .overlays
                .into_iter()
                .map(DataOverlayFile::try_from)
                .collect::<Result<Vec<_>>>()?,
        ))
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
    use crate::transaction::action::test_support::{apply, backed_manifest};
    use crate::transaction::action::{
        Action, AddFragment, AddFragmentParts, CompositeOperation, UserStep,
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
        Action::AddOverlays(AddOverlays::new_unchecked(fragment, overlays))
    }

    fn mint(physical_rows: u64) -> Action {
        Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
            id: Ref::Local(0),
            physical_rows,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        }))
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
                Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
                    id: Ref::Local(0),
                    physical_rows: 4,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                })),
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
        let Action::AddFragment(minted) = mint(3) else {
            unreachable!()
        };
        let minted = AddFragment::new_unchecked(AddFragmentParts {
            last_updated_at_version_meta: Some(minted_with.clone()),
            created_at_version_meta: Some(minted_with.clone()),
            ..minted.into_parts()
        });

        let mut manifest = stable_row_id_manifest(true);
        // The overlay is the minted fragment's only data, which a required
        // column forbids.
        manifest.schema.field_by_id_mut(0).unwrap().nullable = true;
        let out = apply(
            &manifest,
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
    /// only hide the change from consumers that skip non-data commits. The
    /// action cannot be built that way, so only a transaction file can say it.
    #[test]
    fn test_an_overlay_cannot_be_marked_as_no_data_change() {
        let message = pb_actions::AddOverlays {
            data_unchanged: true,
            ..pb_actions::AddOverlays::from(&AddOverlays::new_unchecked(
                Ref::Committed(0),
                vec![overlay("a.lance", &[0])],
            ))
        };
        let error = AddOverlays::try_from(message).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("data_change"), "{error}");
    }
}
