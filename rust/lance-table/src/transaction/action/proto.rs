// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The envelope around the per-action protobuf encodings.
//!
//! Each action encodes itself, in its own module; this module carries the
//! [`Ref`], [`CompositeOperation`], and [`UserAction`] wrappers, the dispatch over
//! the `oneof`, and the helpers the per-action conversions share.
//!
//! Reading is fail-closed: an action this build does not recognize is an error,
//! never a silently skipped element. The commit path collects concurrent
//! transactions with `try_collect`, so a transaction carrying an unknown action
//! must abort the commit rather than be treated as a no-op. Every drafted action
//! is implemented, so an unrecognized one can only come from a newer Lance --
//! which protobuf decodes as no variant at all, since it drops the field it does
//! not know.

use super::{Action, CompositeOperation, Ref, UserAction};
use crate::format::pb;
use lance_core::{Error, Result};

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

/// A required string field. Protobuf gives an unset string the empty value, so
/// the two cases cannot be told apart and neither is usable.
pub(super) fn non_empty(value: String, what: &str) -> Result<String> {
    if value.is_empty() {
        return Err(Error::invalid_input(format!(
            "{what} is required but was empty"
        )));
    }
    Ok(value)
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

impl From<&CompositeOperation> for pb::CompositeOperation {
    fn from(value: &CompositeOperation) -> Self {
        Self {
            // uuid and read_version mirror the enclosing Transaction and are
            // stamped in by its conversion.
            uuid: String::new(),
            read_version: 0,
            actions: value.actions.iter().map(pb::UserAction::from).collect(),
        }
    }
}

impl TryFrom<pb::CompositeOperation> for CompositeOperation {
    type Error = Error;

    fn try_from(message: pb::CompositeOperation) -> Result<Self> {
        Ok(Self {
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

macro_rules! define_action_proto {
    ($($variant:ident,)*) => {
        impl From<&Action> for pb::Action {
            fn from(value: &Action) -> Self {
                let action = match value {
                    $(Action::$variant(action) => pb::action::Action::$variant(action.into()),)*
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
                    $(Some(pb::action::Action::$variant(action)) => {
                        Ok(Self::$variant(action.try_into()?))
                    })*
                    // An action written by a newer Lance decodes to no known
                    // variant, because protobuf drops the field it does not
                    // know. Reject rather than skip: silently dropping an
                    // action would apply a partial transaction.
                    None => Err(Error::not_supported(
                        "an Action in a user operation carried no change this version of Lance \
                         understands; it was either empty or written by a newer version",
                    )),
                }
            }
        }
    };
}

for_each_action!(define_action_proto);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::key_existence::{FilterType, KeyExistenceFilter};
    use crate::format::overlay::{DataOverlayFile, OverlayCoverage};
    use crate::format::{
        BasePath, DataFile, DeletionFile, DeletionFileType, IndexFile, RowIdMeta, pb,
    };
    use crate::rowids::version::RowDatasetVersionMeta;
    use crate::system_index::mem_wal::CompactedSsTable;
    use crate::transaction::UpdateMap;
    use crate::transaction::action::{
        AddBase, AddDataFile, AddField, AddFragment, AddIndexSegment, AddOverlays,
        AdjustIndexCoverage, AlterField, AssertUniqueKeys, ConfigUpdate, DropField,
        FieldMetadataUpdate, RefreshRowVersionMetadata, RemoveFragment, RemoveIndexSegment,
        ReserveFragmentIds, ResetTable, SetDeletionFile, TombstoneFieldData,
        UpdateCompactedSsTables,
    };
    use arrow_schema::{DataType, Field as ArrowField};
    use chrono::DateTime;
    use lance_core::datatypes::Field;
    use lance_file::version::ConcreteFileVersion;
    use roaring::RoaringBitmap;
    use rstest::rstest;
    use std::sync::Arc;
    use uuid::Uuid;

    fn sample_data_file() -> DataFile {
        DataFile::new_unstarted("data/1.lance", ConcreteFileVersion::V2_0)
    }

    fn all_actions() -> Vec<Action> {
        vec![
            Action::AddFragment(AddFragment {
                local: 0,
                physical_rows: 10,
                row_id_meta: Some(RowIdMeta::Inline(vec![1, 2, 3].into())),
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
                field_ids: vec![Ref::Committed(7), Ref::Committed(8)],
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
            Action::AddOverlays(AddOverlays {
                fragment: Ref::Committed(6),
                overlays: vec![DataOverlayFile {
                    data_file: sample_data_file(),
                    coverage: OverlayCoverage::PerField(vec![Arc::new(
                        [1u32, 4].into_iter().collect::<RoaringBitmap>(),
                    )]),
                    committed_version: 11,
                }],
                data_change: true,
            }),
            Action::RefreshRowVersionMetadata(RefreshRowVersionMetadata {
                fragment_ids: vec![4, 6],
            }),
            Action::AlterField(AlterField {
                field: Ref::Committed(2),
                name: Some("renamed".into()),
                logical_type: Some("int64".into()),
                nullable: Some(false),
            }),
            Action::DropField(DropField {
                field: Ref::Committed(3),
            }),
            Action::AddIndexSegment(AddIndexSegment {
                uuid: Uuid::from_u128(7),
                name: "by_a".into(),
                fields: vec![Ref::Committed(1), Ref::Local(3)],
                index_details: Some(Arc::new(prost_types::Any {
                    type_url: "type.googleapis.com/lance.table.MemWalIndexDetails".into(),
                    value: vec![1, 2, 3],
                })),
                index_version: 2,
                covered_fragments: Some(vec![Ref::Committed(4), Ref::Local(0)]),
                files: vec![IndexFile {
                    path: "index.idx".into(),
                    size_bytes: 512,
                }],
                base: Some(Ref::Local(1)),
                created_at: DateTime::from_timestamp_millis(1_700_000_000_000),
                dataset_version: Some(3),
                data_change: false,
            }),
            Action::RemoveIndexSegment(RemoveIndexSegment {
                uuid: Uuid::from_u128(8),
                name: "by_a".into(),
                data_change: true,
            }),
            Action::AdjustIndexCoverage(AdjustIndexCoverage {
                uuid: Uuid::from_u128(9),
                name: "by_b".into(),
                add_fragments: vec![Ref::Committed(1), Ref::Local(0)],
                remove_fragments: vec![2, 3],
            }),
            Action::UpdateCompactedSsTables(UpdateCompactedSsTables {
                compacted_sstables: vec![
                    CompactedSsTable::new(Uuid::from_u128(10), 2),
                    CompactedSsTable::new(Uuid::from_u128(11), 5),
                ],
            }),
            Action::AssertUniqueKeys(AssertUniqueKeys {
                key_fields: vec![Ref::Committed(1), Ref::Local(3)],
                filter: KeyExistenceFilter {
                    field_ids: Vec::new(),
                    filter: FilterType::ExactSet([7u64, 9].into_iter().collect()),
                },
            }),
            Action::ReserveFragmentIds(ReserveFragmentIds { count: 4 }),
            Action::ResetTable(ResetTable),
            Action::ConfigUpdate(ConfigUpdate {
                config: Some(UpdateMap {
                    update_entries: vec![("a", "1").into(), ("b", None).into()],
                    replace: false,
                }),
                table_metadata: None,
                schema_metadata: Some(UpdateMap {
                    update_entries: vec![("c", "2").into()],
                    replace: true,
                }),
                field_metadata: vec![FieldMetadataUpdate {
                    field: Ref::Local(3),
                    updates: UpdateMap {
                        update_entries: vec![("d", "3").into()],
                        replace: false,
                    },
                }],
            }),
        ]
    }

    #[test]
    fn test_composite_operation_round_trips() {
        let operation = CompositeOperation::new(vec![
            UserAction::new("everything", all_actions()),
            UserAction::new("nothing", vec![]),
        ]);

        let message = pb::CompositeOperation::from(&operation);
        let round_tripped = CompositeOperation::try_from(message).unwrap();

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
    fn test_an_unrecognized_action_is_rejected() {
        // An empty action and one a newer Lance wrote decode the same way:
        // protobuf drops the field this build does not know.
        let error = Action::try_from(pb::Action { action: None }).unwrap_err();
        assert!(
            matches!(error, Error::NotSupported { .. }),
            "expected NotSupported, got {error:?}"
        );
        assert!(
            error.to_string().contains("written by a newer version"),
            "unexpected message: {error}"
        );
    }

    #[rstest]
    #[case::removal(pb::Action {
        action: Some(pb::action::Action::RemoveIndexSegment(pb::RemoveIndexSegment {
            uuid: Some((&Uuid::from_u128(1)).into()),
            name: String::new(),
            data_change: None,
        })),
    }, "RemoveIndexSegment.name")]
    #[case::coverage(pb::Action {
        action: Some(pb::action::Action::AdjustIndexCoverage(pb::AdjustIndexCoverage {
            uuid: Some((&Uuid::from_u128(1)).into()),
            name: String::new(),
            add_fragments: Vec::new(),
            remove_fragments: Vec::new(),
        })),
    }, "AdjustIndexCoverage.name")]
    fn test_an_index_action_without_a_name_is_rejected(
        #[case] message: pb::Action,
        #[case] expected: &str,
    ) {
        // Protobuf cannot tell an unset string from an empty one, and an index
        // action that does not say which index it edits cannot be checked for
        // conflicts.
        let error = Action::try_from(message).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains(expected),
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
