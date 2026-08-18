// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tombstone the data-file binding of committed fields within one fragment.

use super::apply::{ApplyState, TOMBSTONED_FIELD};
use super::proto::{data_change_from_wire, data_change_to_wire, required};
use super::{Footprint, Ref};
use crate::format::pb;
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
    pub field_ids: Vec<Ref>,
    /// `false` marks a tombstone whose data is re-added by an
    /// [`AddDataFile`](super::AddDataFile) for the same fields in the same
    /// operation, i.e. a re-encode that moves the bytes without changing any
    /// row's value. A tombstone with no matching re-add nulls the column out
    /// and is a data change.
    pub data_change: bool,
}

impl TombstoneFieldData {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let fragment_id = state.resolve_fragment(self.fragment)?;
        let field_ids = self
            .field_ids
            .iter()
            .map(|field| state.resolve_field(*field))
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

        // New values for these fields supersede any overlay still shadowing
        // them, so the drop is not silently masked by stale overlay cells.
        let overlaid: Vec<u32> = field_ids
            .iter()
            .filter_map(|id| u32::try_from(*id).ok())
            .collect();
        crate::format::overlay::tombstone_overlay_fields(&mut fragment.overlays, &overlaid);

        state.rebind_fields(fragment_id, field_ids.iter().copied());
        Ok(())
    }

    pub(super) fn is_data_change(&self) -> bool {
        self.data_change
    }

    /// The data of each named field in this fragment, and nothing else: another
    /// field's data in the same fragment is untouched.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.add_field_data(self.fragment, self.field_ids.iter().copied());
    }
}

impl From<&TombstoneFieldData> for pb::TombstoneFieldData {
    fn from(value: &TombstoneFieldData) -> Self {
        Self {
            fragment: Some(value.fragment.into()),
            field_ids: value.field_ids.iter().map(|id| (*id).into()).collect(),
            data_change: data_change_to_wire(value.data_change),
        }
    }
}

impl TryFrom<pb::TombstoneFieldData> for TombstoneFieldData {
    type Error = Error;

    fn try_from(message: pb::TombstoneFieldData) -> Result<Self> {
        Ok(Self {
            fragment: required(message.fragment, "TombstoneFieldData.fragment")?.try_into()?,
            field_ids: message
                .field_ids
                .into_iter()
                .map(Ref::try_from)
                .collect::<Result<Vec<_>>>()?,
            data_change: data_change_from_wire(message.data_change),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::transaction::action::Action;
    use crate::transaction::action::test_support::{apply, apply_with_indices, backed_manifest};
    use crate::transaction::test_support::sample_index_metadata;
    use std::sync::Arc;

    fn tombstone_field_zero() -> Action {
        Action::TombstoneFieldData(TombstoneFieldData {
            fragment: Ref::Committed(0),
            field_ids: vec![Ref::Committed(0)],
            data_change: true,
        })
    }

    #[test]
    fn test_tombstone_field_data_drops_the_backing_file() {
        let next = apply(&backed_manifest(), vec![tombstone_field_zero()]).unwrap();

        // The file backed only field 0, so tombstoning it leaves nothing behind.
        assert!(next.fragments[0].files.is_empty());
    }

    #[test]
    fn test_tombstone_field_data_keeps_a_file_with_a_live_field() {
        let mut manifest = backed_manifest();
        let mut fragment = manifest.fragments[0].clone();
        fragment.files[0] = DataFile::new("data/0.lance", vec![0, 1], vec![0, 1], 2, 0, None, None);
        manifest.fragments = Arc::new(vec![fragment]);

        let next = apply(&manifest, vec![tombstone_field_zero()]).unwrap();

        assert_eq!(
            next.fragments[0].files[0].fields.as_ref(),
            &[TOMBSTONED_FIELD, 1]
        );
    }

    #[test]
    fn test_tombstone_field_data_prunes_the_fragment_from_covering_indices() {
        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![tombstone_field_zero()],
            vec![sample_index_metadata("idx")],
        )
        .unwrap();

        // The index covers field 0, whose data in fragment 0 is now gone.
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().is_empty());
    }

    #[test]
    fn test_tombstone_field_data_rejects_a_field_no_file_backs() {
        let error = apply(
            &backed_manifest(),
            vec![Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(0),
                field_ids: vec![Ref::Committed(7)],
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
        use crate::transaction::action::test_support::added_field;
        use crate::transaction::action::{AddDataFile, AddField};

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
                    file: DataFile::new_unstarted("data/fresh.lance", 1, 0),
                    field_ids: vec![Ref::Local(0)],
                    data_change: true,
                }),
                Action::TombstoneFieldData(TombstoneFieldData {
                    fragment: Ref::Committed(0),
                    field_ids: vec![Ref::Local(0)],
                    data_change: false,
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
