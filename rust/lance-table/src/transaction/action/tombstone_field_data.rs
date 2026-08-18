// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tombstone the data-file binding of committed fields within one fragment.

use super::apply::{ApplyState, TOMBSTONED_FIELD};
use super::proto::{
    data_change_from_wire, data_change_to_wire, field_id_from_wire, field_id_to_wire,
    required,
};
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
    /// Committed field ids whose current backing is tombstoned.
    pub field_ids: Vec<i32>,
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
        let fragment = state.fragment_mut(fragment_id, "TombstoneFieldData")?;

        for &field_id in &self.field_ids {
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
        let overlaid: Vec<u32> = self
            .field_ids
            .iter()
            .filter_map(|id| u32::try_from(*id).ok())
            .collect();
        crate::format::overlay::tombstone_overlay_fields(&mut fragment.overlays, &overlaid);

        state.rebind_fields(fragment_id, self.field_ids.iter().copied());
        Ok(())
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
            field_ids: value.field_ids.iter().map(|id| field_id_to_wire(*id)).collect(),
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
                .map(field_id_from_wire)
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
            field_ids: vec![0],
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
                field_ids: vec![7],
                data_change: true,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("field 7"), "{error}");
    }
}
