// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tombstone the data-file binding of committed fields within one fragment.

use super::apply::{ApplyState, TOMBSTONED_FIELD};
use super::proto::{data_change_from_wire, data_change_to_wire, required};
use super::{FieldRef, Footprint, Ref};
use crate::format::pb_actions;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Tombstone the data-file binding of committed fields within one fragment.
///
/// Each field's slot in whatever file currently backs it is marked tombstoned,
/// and a file left with no live field is pruned at apply. Data files have no id
/// of their own and a live field is backed by exactly one file, so this is how a
/// column's data is dropped or superseded: re-encoding a column is a tombstone
/// followed by an [`AddDataFile`](super::AddDataFile) for the same field.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct TombstoneFieldData {
    pub fragment: Ref,
    /// The fields whose current backing is tombstoned.
    pub field_ids: Vec<FieldRef>,
    /// `false` marks one half of a re-encode: every tombstoned field is
    /// re-bound in this fragment by an [`AddDataFile`](super::AddDataFile) in
    /// the same operation that is also `false`, holding the same base values.
    /// Such a tombstone keeps the fields' overlays and index coverage, the
    /// way a `DataReplacement` with `data_change` false does. `true` tombstones
    /// the fields out of the fragment's overlays and withdraws every index on
    /// them from the fragment.
    pub data_change: bool,
}

impl TombstoneFieldData {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let fragment_id = state.resolve_fragment(self.fragment)?;
        let field_ids = self
            .field_ids
            .iter()
            .map(|field| state.resolve_schema_field(*field, "TombstoneFieldData"))
            .collect::<Result<Vec<_>>>()?;
        let fragment = state.fragment_mut(fragment_id, "TombstoneFieldData")?;

        for &field_id in &field_ids {
            let mut found = false;
            for file in fragment.files.iter_mut() {
                let Some(position) = file.fields.iter().position(|id| *id == field_id) else {
                    continue;
                };
                let mut fields = file.fields.to_vec();
                fields[position] = TOMBSTONED_FIELD;
                file.fields = fields.into();
                found = true;
            }
            if !found {
                return Err(Error::invalid_input(format!(
                    "TombstoneFieldData names field {field_id}, which no data file in fragment \
                     {fragment_id} backs"
                )));
            }
        }

        if self.data_change {
            // New values for these fields supersede any overlay still shadowing
            // them, so the drop is not silently masked by stale overlay cells.
            let overlaid: Vec<u32> = field_ids
                .iter()
                .filter_map(|id| u32::try_from(*id).ok())
                .collect();
            crate::format::overlay::tombstone_overlay_fields(&mut fragment.overlays, &overlaid);
            state.rebind_fields(fragment_id, field_ids.iter().copied());
        }
        // A re-encode leaves both alone: the overlays are newer than the base
        // values it moves, and an index over the fields still describes them.
        for field_id in field_ids {
            state.note_tombstone(fragment_id, field_id, self.data_change);
        }
        Ok(())
    }

    pub(super) fn is_data_change(&self) -> bool {
        self.data_change
    }

    /// The data of each named field in this fragment, and nothing else: another
    /// field's data in the same fragment is untouched.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        if self.data_change {
            footprint.write_field_data(self.fragment, self.field_ids.iter().copied());
        } else {
            footprint.move_field_data(self.fragment, self.field_ids.iter().copied());
        }
    }
}

impl From<&TombstoneFieldData> for pb_actions::TombstoneFieldData {
    fn from(value: &TombstoneFieldData) -> Self {
        Self {
            fragment: Some(value.fragment.into()),
            field_ids: value.field_ids.iter().map(|id| (*id).into()).collect(),
            data_change: data_change_to_wire(value.data_change),
        }
    }
}

impl TryFrom<pb_actions::TombstoneFieldData> for TombstoneFieldData {
    type Error = Error;

    fn try_from(message: pb_actions::TombstoneFieldData) -> Result<Self> {
        Ok(Self {
            fragment: required(message.fragment, "TombstoneFieldData.fragment")?.try_into()?,
            field_ids: message
                .field_ids
                .into_iter()
                .map(FieldRef::try_from)
                .collect::<Result<Vec<_>>>()?,
            data_change: data_change_from_wire(message.data_change),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::transaction::action::test_support::{
        added_field, apply, apply_with_indices, backed_manifest, one_column_file,
    };
    use crate::transaction::action::{Action, AddDataFile, AddField};
    use crate::transaction::test_support::{overlay_with_field, sample_index_metadata};
    use lance_file::version::ConcreteFileVersion;
    use rstest::rstest;
    use std::sync::Arc;

    fn tombstone_field_zero(data_change: bool) -> Action {
        Action::TombstoneFieldData(TombstoneFieldData {
            fragment: Ref::Committed(0),
            field_ids: vec![FieldRef::Committed(0)],
            data_change,
        })
    }

    fn rebind_field_zero(data_change: bool) -> Action {
        Action::AddDataFile(AddDataFile {
            fragment: Ref::Committed(0),
            file: one_column_file("data/reencoded.lance"),
            field_ids: vec![FieldRef::Committed(0)],
            data_change,
            base: None,
        })
    }

    #[test]
    fn test_tombstone_field_data_drops_the_backing_file() {
        let next = apply(&backed_manifest(), vec![tombstone_field_zero(true)]).unwrap();

        // The file backed only field 0, so tombstoning it leaves nothing behind.
        assert!(next.fragments[0].files.is_empty());
    }

    #[test]
    fn test_tombstone_field_data_keeps_a_file_with_a_live_field() {
        let mut manifest = backed_manifest();
        let mut fragment = manifest.fragments[0].clone();
        fragment.files[0] = DataFile::new(
            "data/0.lance",
            vec![0, 1],
            vec![0, 1],
            ConcreteFileVersion::V2_0,
            None,
            None,
        );
        manifest.fragments = Arc::new(vec![fragment]);

        let next = apply(&manifest, vec![tombstone_field_zero(true)]).unwrap();

        assert_eq!(
            next.fragments[0].files[0].fields.as_ref(),
            &[TOMBSTONED_FIELD, 1]
        );
    }

    #[test]
    fn test_tombstone_field_data_prunes_the_fragment_from_covering_indices() {
        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![tombstone_field_zero(true)],
            vec![sample_index_metadata("idx")],
        )
        .unwrap();

        // The index covers field 0, whose data in fragment 0 is now gone.
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().is_empty());
    }

    #[rstest]
    #[case::moving_the_values(false, true)]
    #[case::changing_the_values(true, false)]
    fn test_a_reencode_keeps_overlays_and_index_coverage_only_if_values_do_not_change(
        #[case] data_change: bool,
        #[case] kept: bool,
    ) {
        let mut manifest = backed_manifest();
        let mut fragment = manifest.fragments[0].clone();
        fragment.overlays = vec![overlay_with_field(0, manifest.version)];
        manifest.fragments = Arc::new(vec![fragment]);

        let (next, indices) = apply_with_indices(
            &manifest,
            vec![
                tombstone_field_zero(data_change),
                rebind_field_zero(data_change),
            ],
            vec![sample_index_metadata("idx")],
        )
        .unwrap();

        // Only the overlay's field is tombstoned, which leaves it with none
        // live, so a data change drops it outright.
        assert_eq!(next.fragments[0].overlays.len(), usize::from(kept));
        assert_eq!(
            indices[0].fragment_bitmap.as_ref().unwrap().contains(0),
            kept
        );
    }

    #[rstest]
    #[case::a_tombstone_left_unbound(
        vec![tombstone_field_zero(false)],
        "without a data file"
    )]
    #[case::a_rebind_that_changes_data(
        vec![tombstone_field_zero(false), rebind_field_zero(true)],
        "must agree"
    )]
    #[case::a_rebind_of_a_data_changing_tombstone(
        vec![tombstone_field_zero(true), rebind_field_zero(false)],
        "must agree"
    )]
    fn test_a_reencode_without_a_data_change_must_rebind_what_it_tombstones(
        #[case] actions: Vec<Action>,
        #[case] message: &str,
    ) {
        let error = apply(&backed_manifest(), actions).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains(message), "{error}");
    }

    #[test]
    fn test_tombstone_field_data_rejects_a_field_no_file_backs() {
        let error = apply(
            &backed_manifest(),
            vec![Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(0),
                field_ids: vec![FieldRef::Committed(7)],
                data_change: true,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("field 7"), "{error}");
    }

    #[test]
    fn test_tombstone_field_data_can_name_a_field_minted_in_the_same_operation() {
        // A squash of "add column with data" and "re-encode that column" lowers
        // to exactly this, and has no committed id to name the field by.
        let next = apply(
            &backed_manifest(),
            vec![
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("fresh"),
                }),
                Action::AddDataFile(AddDataFile {
                    fragment: Ref::Committed(0),
                    file: DataFile::new_unstarted("data/fresh.lance", ConcreteFileVersion::V1),
                    field_ids: vec![FieldRef::Local(0)],
                    data_change: true,
                    base: None,
                }),
                Action::TombstoneFieldData(TombstoneFieldData {
                    fragment: Ref::Committed(0),
                    field_ids: vec![FieldRef::Local(0)],
                    data_change: true,
                }),
            ],
        )
        .unwrap();

        // The file that backed only the minted field is left backing nothing
        // and is pruned, so the fragment keeps just its original file.
        assert_eq!(next.fragments[0].files.len(), 1);
        assert!(
            !next.fragments[0]
                .files
                .iter()
                .any(|file| file.path == "data/fresh.lance")
        );
    }
}
