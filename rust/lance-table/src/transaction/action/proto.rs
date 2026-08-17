// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The envelope around the per-action protobuf encodings.
//!
//! Each action encodes itself, in its own module; this module carries the
//! [`Ref`], [`UserOperation`], and [`UserAction`] wrappers, the dispatch over
//! the `oneof`, and the helpers the per-action conversions share.
//!
//! Reading is fail-closed: an action this build does not implement is an error,
//! never a silently skipped element. The commit path collects concurrent
//! transactions with `try_collect`, so a transaction carrying an unknown action
//! must abort the commit rather than be treated as a no-op.

use super::{Action, Ref, UserAction, UserOperation};
use crate::format::pb;
use lance_core::{Error, Result};

/// A field on the wire is a [`Ref`], so an action can name a field the same
/// operation is minting. The actions that resolve one against the manifest come
/// later; until then only a committed reference means anything here.
pub(super) fn field_id_from_wire(field: pb::Ref) -> Result<i32> {
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

pub(super) fn field_id_to_wire(id: i32) -> pb::Ref {
    Ref::Committed(id as u64).into()
}

/// `data_change` is absent-means-true on the wire, so only the `false` case is
/// written out.
pub(super) fn data_change_to_wire(data_change: bool) -> Option<bool> {
    (!data_change).then_some(false)
}

pub(super) fn data_change_from_wire(data_change: Option<bool>) -> bool {
    data_change.unwrap_or(true)
}

pub(super) fn required<T>(value: Option<T>, what: &str) -> Result<T> {
    value.ok_or_else(|| Error::invalid_input(format!("{what} is required but was not set")))
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
            Action::ReserveFragmentIds(action) => {
                pb::action::Action::ReserveFragmentIds(action.into())
            }
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
            Some(pb::action::Action::ReserveFragmentIds(action)) => {
                Ok(Self::ReserveFragmentIds(action.try_into()?))
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{BasePath, DataFile, DeletionFile, DeletionFileType, RowIdMeta, pb};
    use crate::rowids::version::RowDatasetVersionMeta;
    use crate::transaction::action::{
        AddBase, AddDataFile, AddField, AddFragment, AlterField, DropField, RemoveFragment,
        ReserveFragmentIds, SetDeletionFile, TombstoneFieldData,
    };
    use arrow_schema::{DataType, Field as ArrowField};
    use lance_core::datatypes::Field;
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
            Action::ReserveFragmentIds(ReserveFragmentIds { count: 4 }),
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

    #[test]
    fn test_field_id_out_of_range_is_rejected() {
        let error = field_id_from_wire(u64::from(u32::MAX) + 1).unwrap_err();
        assert!(
            error.to_string().contains("exceeds the maximum field id"),
            "unexpected message: {error}"
        );
    }
}
