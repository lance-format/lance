// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Append overlay files to a fragment.

use super::apply::ApplyState;
use super::proto::{data_change_from_wire, data_change_to_wire, required};
use super::{Coordinate, Footprint, Ref};
use crate::format::overlay::DataOverlayFile;
use crate::format::pb;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Append overlay files to one fragment, supplying new values for a subset of
/// its `(row offset, field)` cells without rewriting its base data files.
///
/// Overlays are appended rather than replaced, so overlays a concurrent writer
/// added survive. Within the fragment they are ordered newest-last, which is
/// what appending at the version this commit produces gives.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddOverlays {
    pub fragment: Ref,
    /// The overlays to append, oldest first. Each one's `committed_version` is
    /// ignored and stamped with the version this commit produces, so replaying
    /// the action onto a newer version re-stamps rather than backdates.
    pub overlays: Vec<DataOverlayFile>,
    pub data_change: bool,
}

impl AddOverlays {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let fragment_id = state.resolve_fragment(self.fragment)?;
        let committed_version = state.new_version();
        let fragment = state.fragment_mut(fragment_id, "AddOverlays")?;
        fragment
            .overlays
            .extend(self.overlays.iter().cloned().map(|mut overlay| {
                overlay.committed_version = committed_version;
                overlay
            }));
        Ok(())
    }

    /// An overlay supplies new cell values, so by default it changes what a
    /// reader sees. The writer can still mark a restatement of existing values
    /// as no change.
    pub(super) fn is_data_change(&self) -> bool {
        self.data_change
    }

    /// A partial write of each overlaid field's data in the fragment, and a
    /// requirement on each of those fields' definitions.
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
    /// concurrent cast or drop landing first would leave them described as
    /// something they are not, or over a field the manifest no longer has.
    ///
    /// A fragment this operation mints records nothing; no concurrent writer
    /// can name a cell inside it.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        let Some(fragment) = self.fragment.committed() else {
            return;
        };
        for field in self
            .overlays
            .iter()
            .flat_map(|overlay| overlay.data_file.fields.iter().copied())
            .filter(|field| *field >= 0)
        {
            footprint.write_part(Coordinate::FieldData { fragment, field });
            footprint.require(Coordinate::FieldDefinition(field));
        }
    }
}

impl From<&AddOverlays> for pb::AddOverlays {
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

impl TryFrom<pb::AddOverlays> for AddOverlays {
    type Error = Error;

    fn try_from(message: pb::AddOverlays) -> Result<Self> {
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
    use crate::format::DataFile;
    use crate::format::overlay::OverlayCoverage;
    use crate::transaction::action::test_support::{apply, backed_manifest, footprint};
    use crate::transaction::action::{
        Action, AddFragment, AddIndexSegment, AlterField, DropField, RemoveFragment,
        TombstoneFieldData,
    };
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
        let error = apply(
            &backed_manifest(),
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
                field_ids: vec![Ref::Committed(field as u64)],
                data_change: true,
            })])
        };

        assert!(overlaid.conflicts_with(&rewrite(0)));
        assert!(rewrite(0).conflicts_with(&overlaid));
        // Another column of the same fragment is not touched by the overlay.
        assert!(!overlaid.conflicts_with(&rewrite(1)));
        assert!(!rewrite(1).conflicts_with(&overlaid));
    }

    /// An overlay landing after its field was cast or dropped would carry
    /// values in a type the schema no longer names, or for a field that is
    /// gone. Landing first, the cast rebinds the field everywhere and the drop
    /// tombstones the overlay, so those orders are fine.
    #[test]
    fn test_an_overlay_needs_its_field_defined_as_it_was() {
        let overlaid = footprint(vec![add(Ref::Committed(0), vec![overlay("a.lance", &[0])])]);
        let cast = footprint(vec![Action::AlterField(AlterField {
            field: Ref::Committed(0),
            name: None,
            logical_type: Some("int64".into()),
            nullable: None,
        })]);
        let dropped = footprint(vec![Action::DropField(DropField {
            field: Ref::Committed(0),
        })]);

        assert!(overlaid.conflicts_with(&cast));
        assert!(!cast.conflicts_with(&overlaid));
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
            fields: vec![Ref::Committed(0)],
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
                field_ids: vec![Ref::Committed(0)],
                data_change: true,
            }),
        ]);
        let other_overlay = footprint(vec![add(Ref::Committed(0), vec![overlay("b.lance", &[0])])]);
        let other_column = footprint(vec![Action::TombstoneFieldData(TombstoneFieldData {
            fragment: Ref::Committed(0),
            field_ids: vec![Ref::Committed(1)],
            data_change: true,
        })]);

        assert!(materialize.conflicts_with(&other_overlay));
        assert!(other_overlay.conflicts_with(&materialize));
        assert!(!materialize.conflicts_with(&other_column));
        assert!(!other_column.conflicts_with(&materialize));
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
}
