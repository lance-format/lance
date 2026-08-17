// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Conversions between the action vocabulary and its protobuf encoding.
//!
//! Reading is fail-closed: an action this build does not implement is an error,
//! never a silently skipped element. The commit path collects concurrent
//! transactions with `try_collect`, so a transaction carrying an unknown action
//! must abort the commit rather than be treated as a no-op.

use super::{
    Action, AddBase, AddDataFile, AddField, AddFragment, AlterField, DropField, Ref,
    RemoveFragment, SetDeletionFile, TombstoneFieldData, UserAction, UserOperation,
};
use crate::format::pb;
use crate::format::{BasePath, DataFile, DeletionFile, ExternalFile, RowIdMeta};
use crate::rowids::version::RowDatasetVersionMeta;
use lance_core::datatypes::Field;
use lance_core::{Error, Result};

/// A field on the wire is a [`Ref`], so an action can name a field the same
/// operation is minting. The actions that resolve one against the manifest come
/// later; until then only a committed reference means anything here.
fn field_id_from_wire(field: pb::Ref) -> Result<i32> {
    match Ref::try_from(field)? {
        Ref::Committed(id) => i32::try_from(id).map_err(|_| {
            Error::invalid_input(format!(
                "field id {id} in an action exceeds the maximum field id ({})",
                i32::MAX
            ))
        }),
        Ref::Local(token) => Err(Error::invalid_input(format!(
            "an action names field {token}, which the same operation is minting; this build \
             resolves only committed field references"
        ))),
    }
}

fn field_id_to_wire(id: i32) -> pb::Ref {
    Ref::Committed(id as u64).into()
}

impl From<Ref> for pb::Ref {
    fn from(value: Ref) -> Self {
        let kind = match value {
            Ref::Committed(id) => pb::r#ref::Kind::Committed(id),
            Ref::Local(token) => pb::r#ref::Kind::Local(token),
        };
        Self { kind: Some(kind) }
    }
}

impl TryFrom<pb::Ref> for Ref {
    type Error = Error;

    fn try_from(message: pb::Ref) -> Result<Self> {
        match message.kind {
            Some(pb::r#ref::Kind::Committed(id)) => Ok(Self::Committed(id)),
            Some(pb::r#ref::Kind::Local(token)) => Ok(Self::Local(token)),
            None => Err(Error::invalid_input(
                "a Ref in an action was empty; it must be either committed or local",
            )),
        }
    }
}

/// `data_change` is absent-means-true on the wire, so only the `false` case is
/// written out.
fn data_change_to_wire(data_change: bool) -> Option<bool> {
    (!data_change).then_some(false)
}

fn data_change_from_wire(data_change: Option<bool>) -> bool {
    data_change.unwrap_or(true)
}

impl From<&UserOperation> for pb::CompositeOperation {
    fn from(value: &UserOperation) -> Self {
        Self {
            // uuid and read_version mirror the enclosing Transaction and are
            // stamped in by its conversion.
            uuid: String::new(),
            read_version: 0,
            actions: value.actions.iter().map(pb::UserAction::from).collect(),
        }
    }
}

impl TryFrom<pb::CompositeOperation> for UserOperation {
    type Error = Error;

    fn try_from(message: pb::CompositeOperation) -> Result<Self> {
        Ok(Self {
            description: String::new(),
            actions: message
                .actions
                .into_iter()
                .map(UserAction::try_from)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl From<&UserAction> for pb::UserAction {
    fn from(value: &UserAction) -> Self {
        Self {
            description: value.description.clone(),
            actions: value.actions.iter().map(pb::Action::from).collect(),
        }
    }
}

impl TryFrom<pb::UserAction> for UserAction {
    type Error = Error;

    fn try_from(message: pb::UserAction) -> Result<Self> {
        Ok(Self {
            description: message.description,
            actions: message
                .actions
                .into_iter()
                .map(Action::try_from)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl From<&Action> for pb::Action {
    fn from(value: &Action) -> Self {
        let action = match value {
            Action::AddFragment(action) => pb::action::Action::AddFragment(action.into()),
            Action::AddDataFile(action) => pb::action::Action::AddDataFile(action.into()),
            Action::AddField(action) => pb::action::Action::AddField(action.into()),
            Action::AddBase(action) => pb::action::Action::AddBase(action.into()),
            Action::TombstoneFieldData(action) => {
                pb::action::Action::TombstoneFieldData(action.into())
            }
            Action::RemoveFragment(action) => pb::action::Action::RemoveFragment(action.into()),
            Action::SetDeletionFile(action) => pb::action::Action::SetDeletionFile(action.into()),
            Action::AlterField(action) => pb::action::Action::AlterField(action.into()),
            Action::DropField(action) => pb::action::Action::DropField(action.into()),
        };
        Self {
            action: Some(action),
        }
    }
}

impl TryFrom<pb::Action> for Action {
    type Error = Error;

    fn try_from(message: pb::Action) -> Result<Self> {
        match message.action {
            Some(pb::action::Action::AddFragment(action)) => {
                Ok(Self::AddFragment(action.try_into()?))
            }
            Some(pb::action::Action::AddDataFile(action)) => {
                Ok(Self::AddDataFile(action.try_into()?))
            }
            Some(pb::action::Action::AddField(action)) => Ok(Self::AddField(action.try_into()?)),
            Some(pb::action::Action::AddBase(action)) => Ok(Self::AddBase(action.try_into()?)),
            Some(pb::action::Action::TombstoneFieldData(action)) => {
                Ok(Self::TombstoneFieldData(action.try_into()?))
            }
            Some(pb::action::Action::RemoveFragment(action)) => {
                Ok(Self::RemoveFragment(action.try_into()?))
            }
            Some(pb::action::Action::SetDeletionFile(action)) => {
                Ok(Self::SetDeletionFile(action.try_into()?))
            }
            Some(pb::action::Action::AlterField(action)) => {
                Ok(Self::AlterField(action.try_into()?))
            }
            Some(pb::action::Action::DropField(action)) => Ok(Self::DropField(action.try_into()?)),
            // The drafted vocabulary is larger than what is implemented. Reject
            // rather than skip: silently dropping an action would apply a
            // partial transaction.
            Some(other) => Err(Error::not_supported(format!(
                "the action-based transaction uses action {other:?}, which is drafted but not \
                 implemented by this version of Lance",
            ))),
            None => Err(Error::invalid_input(
                "an Action in a user operation was empty",
            )),
        }
    }
}

impl From<&AddFragment> for pb::AddFragment {
    fn from(value: &AddFragment) -> Self {
        Self {
            local: value.local,
            physical_rows: value.physical_rows,
            row_id_sequence: value.row_id_meta.as_ref().map(|meta| match meta {
                RowIdMeta::Inline(data) => {
                    pb::add_fragment::RowIdSequence::InlineRowIds(data.clone())
                }
                RowIdMeta::External(file) => {
                    pb::add_fragment::RowIdSequence::ExternalRowIds(external_file_to_wire(file))
                }
            }),
            last_updated_at_version_sequence: value
                .last_updated_at_version_meta
                .as_ref()
                .map(|meta| match meta {
                    RowDatasetVersionMeta::Inline(data) => {
                        pb::add_fragment::LastUpdatedAtVersionSequence::InlineLastUpdatedAtVersions(
                            data.to_vec(),
                        )
                    }
                    RowDatasetVersionMeta::External(file) => {
                        pb::add_fragment::LastUpdatedAtVersionSequence::ExternalLastUpdatedAtVersions(
                            external_file_to_wire(file),
                        )
                    }
                }),
            created_at_version_sequence: value
                .created_at_version_meta
                .as_ref()
                .map(|meta| match meta {
                    RowDatasetVersionMeta::Inline(data) => {
                        pb::add_fragment::CreatedAtVersionSequence::InlineCreatedAtVersions(
                            data.to_vec(),
                        )
                    }
                    RowDatasetVersionMeta::External(file) => {
                        pb::add_fragment::CreatedAtVersionSequence::ExternalCreatedAtVersions(
                            external_file_to_wire(file),
                        )
                    }
                }),
            data_change: data_change_to_wire(value.data_change),
        }
    }
}

impl TryFrom<pb::AddFragment> for AddFragment {
    type Error = Error;

    fn try_from(message: pb::AddFragment) -> Result<Self> {
        Ok(Self {
            local: message.local,
            physical_rows: message.physical_rows,
            row_id_meta: message.row_id_sequence.map(|sequence| match sequence {
                pb::add_fragment::RowIdSequence::InlineRowIds(data) => RowIdMeta::Inline(data),
                pb::add_fragment::RowIdSequence::ExternalRowIds(file) => {
                    RowIdMeta::External(external_file_from_wire(file))
                }
            }),
            last_updated_at_version_meta: message.last_updated_at_version_sequence.map(
                |sequence| {
                    match sequence {
                    pb::add_fragment::LastUpdatedAtVersionSequence::InlineLastUpdatedAtVersions(
                        data,
                    ) => RowDatasetVersionMeta::Inline(data.into()),
                    pb::add_fragment::LastUpdatedAtVersionSequence::ExternalLastUpdatedAtVersions(
                        file,
                    ) => RowDatasetVersionMeta::External(external_file_from_wire(file)),
                }
                },
            ),
            created_at_version_meta: message.created_at_version_sequence.map(|sequence| {
                match sequence {
                    pb::add_fragment::CreatedAtVersionSequence::InlineCreatedAtVersions(data) => {
                        RowDatasetVersionMeta::Inline(data.into())
                    }
                    pb::add_fragment::CreatedAtVersionSequence::ExternalCreatedAtVersions(file) => {
                        RowDatasetVersionMeta::External(external_file_from_wire(file))
                    }
                }
            }),
            data_change: data_change_from_wire(message.data_change),
        })
    }
}

fn external_file_to_wire(file: &ExternalFile) -> pb::ExternalFile {
    pb::ExternalFile {
        path: file.path.clone(),
        offset: file.offset,
        size: file.size,
    }
}

fn external_file_from_wire(file: pb::ExternalFile) -> ExternalFile {
    ExternalFile {
        path: file.path,
        offset: file.offset,
        size: file.size,
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

impl From<&AddField> for pb::AddField {
    fn from(value: &AddField) -> Self {
        Self {
            local: value.local,
            parent: value.parent.map(Into::into),
            def: Some(lance_file::format::pb::Field::from(&value.def)),
        }
    }
}

impl TryFrom<pb::AddField> for AddField {
    type Error = Error;

    fn try_from(message: pb::AddField) -> Result<Self> {
        Ok(Self {
            local: message.local,
            parent: message.parent.map(Ref::try_from).transpose()?,
            def: Field::from(&required(message.def, "AddField.def")?),
        })
    }
}

impl From<&AddBase> for pb::AddBase {
    fn from(value: &AddBase) -> Self {
        Self {
            local: value.local,
            base: Some(pb::BasePath::from(value.base.clone())),
        }
    }
}

impl TryFrom<pb::AddBase> for AddBase {
    type Error = Error;

    fn try_from(message: pb::AddBase) -> Result<Self> {
        Ok(Self {
            local: message.local,
            base: BasePath::from(required(message.base, "AddBase.base")?),
        })
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

impl From<&RemoveFragment> for pb::RemoveFragment {
    fn from(value: &RemoveFragment) -> Self {
        Self {
            fragment: Some(value.fragment.into()),
            data_change: data_change_to_wire(value.data_change),
        }
    }
}

impl TryFrom<pb::RemoveFragment> for RemoveFragment {
    type Error = Error;

    fn try_from(message: pb::RemoveFragment) -> Result<Self> {
        Ok(Self {
            fragment: required(message.fragment, "RemoveFragment.fragment")?.try_into()?,
            data_change: data_change_from_wire(message.data_change),
        })
    }
}

impl From<&SetDeletionFile> for pb::SetDeletionFile {
    fn from(value: &SetDeletionFile) -> Self {
        Self {
            fragment: value.fragment,
            deletion_file: value.deletion_file.as_ref().map(pb::DeletionFile::from),
            data_change: data_change_to_wire(value.data_change),
        }
    }
}

impl TryFrom<pb::SetDeletionFile> for SetDeletionFile {
    type Error = Error;

    fn try_from(message: pb::SetDeletionFile) -> Result<Self> {
        Ok(Self {
            fragment: message.fragment,
            deletion_file: message
                .deletion_file
                .map(DeletionFile::try_from)
                .transpose()?,
            data_change: data_change_from_wire(message.data_change),
        })
    }
}

impl From<&AlterField> for pb::AlterField {
    fn from(value: &AlterField) -> Self {
        Self {
            field: Some(field_id_to_wire(value.field)),
            name: value.name.clone(),
            logical_type: value.logical_type.clone(),
            nullable: value.nullable,
        }
    }
}

impl TryFrom<pb::AlterField> for AlterField {
    type Error = Error;

    fn try_from(message: pb::AlterField) -> Result<Self> {
        Ok(Self {
            field: field_id_from_wire(required(message.field, "AlterField.field")?)?,
            name: message.name,
            logical_type: message.logical_type,
            nullable: message.nullable,
        })
    }
}

impl From<&DropField> for pb::DropField {
    fn from(value: &DropField) -> Self {
        Self {
            field: value.field as u64,
        }
    }
}

impl TryFrom<pb::DropField> for DropField {
    type Error = Error;

    fn try_from(message: pb::DropField) -> Result<Self> {
        Ok(Self {
            field: field_id_from_wire(message.field)?,
        })
    }
}

fn required<T>(value: Option<T>, what: &str) -> Result<T> {
    value.ok_or_else(|| Error::invalid_input(format!("{what} is required but was not set")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{DeletionFileType, pb};
    use arrow_schema::{DataType, Field as ArrowField};
    use std::sync::Arc;

    fn sample_data_file() -> DataFile {
        DataFile::new_unstarted("data/1.lance", 2, 0)
    }

    fn all_actions() -> Vec<Action> {
        vec![
            Action::AddFragment(AddFragment {
                local: 0,
                physical_rows: 10,
                row_id_meta: Some(RowIdMeta::Inline(vec![1, 2, 3])),
                last_updated_at_version_meta: Some(RowDatasetVersionMeta::Inline(Arc::from(
                    [4u8, 5].as_slice(),
                ))),
                created_at_version_meta: None,
                data_change: true,
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(0),
                file: sample_data_file(),
                field_ids: vec![Ref::Committed(1), Ref::Local(3)],
                data_change: false,
            }),
            Action::AddField(AddField {
                local: 3,
                parent: Some(Ref::Committed(1)),
                def: Field::try_from(ArrowField::new("added", DataType::Int32, true)).unwrap(),
            }),
            Action::AddBase(AddBase {
                local: 1,
                base: BasePath::new(0, "s3://bucket/x".into(), Some("other".into()), false),
            }),
            Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(4),
                field_ids: vec![7, 8],
                data_change: true,
            }),
            Action::RemoveFragment(RemoveFragment {
                fragment: Ref::Committed(5),
                data_change: true,
            }),
            Action::SetDeletionFile(SetDeletionFile {
                fragment: 6,
                deletion_file: Some(DeletionFile {
                    read_version: 3,
                    id: 9,
                    file_type: DeletionFileType::Bitmap,
                    num_deleted_rows: Some(4),
                    base_id: None,
                }),
                data_change: true,
            }),
            Action::AlterField(AlterField {
                field: 2,
                name: Some("renamed".into()),
                logical_type: Some("int64".into()),
                nullable: Some(false),
            }),
            Action::DropField(DropField { field: 3 }),
        ]
    }

    #[test]
    fn test_user_operation_round_trips() {
        let operation = UserOperation::new(
            "compound commit",
            vec![
                UserAction::new("everything", all_actions()),
                UserAction::new("nothing", vec![]),
            ],
        );

        let message = pb::CompositeOperation::from(&operation);
        let round_tripped = UserOperation::try_from(message).unwrap();

        assert_eq!(round_tripped, operation);
    }

    #[test]
    fn test_data_change_is_absent_when_true() {
        // Absent means "real change" on the wire, so the common case costs no
        // bytes and an old field-less writer is read correctly.
        let message = pb::RemoveFragment::from(&RemoveFragment {
            fragment: Ref::Committed(1),
            data_change: true,
        });
        assert_eq!(message.data_change, None);
        assert!(RemoveFragment::try_from(message).unwrap().data_change);
    }

    #[test]
    fn test_unimplemented_action_is_rejected() {
        let message = pb::Action {
            action: Some(pb::action::Action::ResetTable(pb::ResetTable {})),
        };
        let error = Action::try_from(message).unwrap_err();
        assert!(
            matches!(error, Error::NotSupported { .. }),
            "expected NotSupported, got {error:?}"
        );
        assert!(
            error.to_string().contains("not implemented"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn test_empty_action_is_rejected() {
        let error = Action::try_from(pb::Action { action: None }).unwrap_err();
        assert!(
            error.to_string().contains("was empty"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn test_empty_ref_is_rejected() {
        let error = Ref::try_from(pb::Ref { kind: None }).unwrap_err();
        assert!(
            error.to_string().contains("committed or local"),
            "unexpected message: {error}"
        );
    }
}
