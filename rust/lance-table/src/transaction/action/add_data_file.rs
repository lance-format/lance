// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Add a data file to a fragment.

use super::apply::ApplyState;
use super::proto::{data_change_from_wire, data_change_to_wire, required};
use super::{Footprint, Ref};
use crate::format::{DataFile, pb};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Add a data file to a fragment.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddDataFile {
    /// The fragment to add the file to: committed, or a fragment minted earlier
    /// in the same operation.
    pub fragment: Ref,
    /// The file. Its `fields` are placeholders and are stamped in at apply from
    /// `field_ids`, which is the authority for the column -> field mapping.
    pub file: DataFile,
    /// One entry per column in `file`, in column order.
    pub field_ids: Vec<Ref>,
    /// See [`AddFragment::data_change`](super::AddFragment::data_change).
    pub data_change: bool,
}

impl AddDataFile {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let fragment_id = state.resolve_fragment(self.fragment)?;
        let field_ids = self
            .field_ids
            .iter()
            .map(|field| state.resolve_field(*field))
            .collect::<Result<Vec<_>>>()?;

        let mut file = self.file.clone();
        if !file.column_indices.is_empty() && file.column_indices.len() != field_ids.len() {
            return Err(Error::invalid_input(format!(
                "AddDataFile for fragment {fragment_id} lists {} field ids but the file has {} \
                 columns",
                field_ids.len(),
                file.column_indices.len()
            )));
        }
        file.fields = field_ids.into();

        state
            .fragment_mut(fragment_id, "AddDataFile")?
            .files
            .push(file);
        Ok(())
    }

    pub(super) fn is_data_change(&self) -> bool {
        self.data_change
    }

    /// The data of every committed field the file backs, in the fragment it is
    /// attached to. A file backing only minted fields writes nothing.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.add_field_data(self.fragment, self.field_ids.iter().copied());
    }
}

impl From<&AddDataFile> for pb::AddDataFile {
    fn from(value: &AddDataFile) -> Self {
        Self {
            fragment: Some(value.fragment.into()),
            file: Some(pb::DataFile::from(&value.file)),
            field_ids: value.field_ids.iter().map(|id| (*id).into()).collect(),
            data_change: data_change_to_wire(value.data_change),
        }
    }
}

impl TryFrom<pb::AddDataFile> for AddDataFile {
    type Error = Error;

    fn try_from(message: pb::AddDataFile) -> Result<Self> {
        Ok(Self {
            fragment: required(message.fragment, "AddDataFile.fragment")?.try_into()?,
            file: DataFile::try_from(required(message.file, "AddDataFile.file")?)?,
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
    use crate::transaction::action::Action;
    use crate::transaction::action::test_support::apply;
    use crate::transaction::test_support::sample_manifest;

    #[test]
    fn test_add_data_file_rejects_an_unbound_local_token() {
        let manifest = sample_manifest();
        let error = apply(
            &manifest,
            vec![Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(3),
                file: DataFile::new_unstarted("data/x.lance", 2, 0),
                field_ids: vec![Ref::Committed(0)],
                data_change: true,
            })],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("local fragment token 3"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_add_data_file_rejects_a_column_count_mismatch() {
        let manifest = sample_manifest();
        let mut file = DataFile::new_unstarted("data/x.lance", 2, 0);
        file.column_indices = vec![0, 1].into();

        let error = apply(
            &manifest,
            vec![Action::AddDataFile(AddDataFile {
                fragment: Ref::Committed(0),
                file,
                field_ids: vec![Ref::Committed(0)],
                data_change: true,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("1 field ids"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_add_data_file_rejects_a_missing_fragment() {
        let manifest = sample_manifest();
        let error = apply(
            &manifest,
            vec![Action::AddDataFile(AddDataFile {
                fragment: Ref::Committed(7),
                file: DataFile::new_unstarted("data/x.lance", 2, 0),
                field_ids: vec![Ref::Committed(0)],
                data_change: true,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("fragment 7"), "{error}");
    }
}
