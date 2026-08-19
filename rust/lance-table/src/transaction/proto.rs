// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Conversions between the transaction types and their protobuf encoding.
//!
//! A transaction is persisted as a `pb::Transaction` alongside the manifest it
//! produced, so these conversions are the format contract for everything in this
//! module: a field added to an `Operation` is only durable once it round-trips
//! here.

use crate::format::key_existence::KeyExistenceFilter;
use crate::format::pb;
use crate::format::pb_actions;
use crate::format::{BasePath, Fragment, IndexFile, IndexMetadata, overlay::DataOverlayFile};
use crate::system_index::mem_wal::CompactedSsTable;
use crate::transaction::action::CompositeOperation;
use crate::transaction::{
    DataOverlayGroup, DataReplacementGroup, Operation, RewriteGroup, RewrittenIndex, Transaction,
    UpdateMap, UpdateMapEntry, UpdateMode, UpdatedFragmentOffsets, translate_config_updates,
    translate_schema_metadata_updates,
};
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_file::datatypes::Fields;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::num::NonZero;
use std::sync::Arc;
use uuid::Uuid;

impl From<&DataReplacementGroup> for pb::transaction::DataReplacementGroup {
    fn from(DataReplacementGroup(fragment_id, new_file): &DataReplacementGroup) -> Self {
        Self {
            fragment_id: *fragment_id,
            new_file: Some(new_file.into()),
        }
    }
}

/// Convert a protobug DataReplacementGroup to a rust native DataReplacementGroup
/// this is unfortunately TryFrom instead of From because of the Option in the pb::DataReplacementGroup
impl TryFrom<pb::transaction::DataReplacementGroup> for DataReplacementGroup {
    type Error = Error;

    fn try_from(message: pb::transaction::DataReplacementGroup) -> Result<Self> {
        Ok(Self(
            message.fragment_id,
            message
                .new_file
                .ok_or(Error::invalid_input(
                    "DataReplacementGroup must have a new_file",
                ))?
                .try_into()?,
        ))
    }
}

impl From<&DataOverlayGroup> for pb::transaction::DataOverlayGroup {
    fn from(group: &DataOverlayGroup) -> Self {
        Self {
            fragment_id: group.fragment_id,
            overlays: group
                .overlays
                .iter()
                .map(pb::DataOverlayFile::from)
                .collect(),
        }
    }
}

impl TryFrom<pb::transaction::DataOverlayGroup> for DataOverlayGroup {
    type Error = Error;

    fn try_from(message: pb::transaction::DataOverlayGroup) -> Result<Self> {
        Ok(Self {
            fragment_id: message.fragment_id,
            overlays: message
                .overlays
                .into_iter()
                .map(DataOverlayFile::try_from)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl TryFrom<pb::Transaction> for Transaction {
    type Error = Error;

    fn try_from(message: pb::Transaction) -> Result<Self> {
        let operation = match message.operation {
            Some(pb::transaction::Operation::Append(pb::transaction::Append { fragments })) => {
                Operation::Append {
                    fragments: fragments
                        .into_iter()
                        .map(Fragment::try_from)
                        .collect::<Result<Vec<_>>>()?,
                }
            }
            Some(pb::transaction::Operation::Clone(pb::transaction::Clone {
                is_shallow,
                ref_name,
                ref_version,
                ref_path,
                branch_name,
            })) => Operation::Clone {
                is_shallow,
                ref_name,
                ref_version,
                ref_path,
                branch_name,
            },
            Some(pb::transaction::Operation::Delete(pb::transaction::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            })) => Operation::Delete {
                updated_fragments: updated_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                deleted_fragment_ids,
                predicate,
            },
            Some(pb::transaction::Operation::Overwrite(pb::transaction::Overwrite {
                fragments,
                schema,
                schema_metadata: _schema_metadata, // TODO: handle metadata
                config_upsert_values,
                initial_bases,
            })) => {
                let config_upsert_option = if config_upsert_values.is_empty() {
                    None
                } else {
                    Some(config_upsert_values)
                };

                Operation::Overwrite {
                    fragments: fragments
                        .into_iter()
                        .map(Fragment::try_from)
                        .collect::<Result<Vec<_>>>()?,
                    schema: Schema::try_from(&Fields(schema))?,
                    config_upsert_values: config_upsert_option,
                    initial_bases: if initial_bases.is_empty() {
                        None
                    } else {
                        Some(initial_bases.into_iter().map(BasePath::from).collect())
                    },
                }
            }
            Some(pb::transaction::Operation::ReserveFragments(
                pb::transaction::ReserveFragments { num_fragments },
            )) => Operation::ReserveFragments { num_fragments },
            Some(pb::transaction::Operation::Rewrite(pb::transaction::Rewrite {
                old_fragments,
                new_fragments,
                groups,
                rewritten_indices,
            })) => {
                let groups = if !groups.is_empty() {
                    groups
                        .into_iter()
                        .map(RewriteGroup::try_from)
                        .collect::<Result<_>>()?
                } else {
                    vec![RewriteGroup {
                        old_fragments: old_fragments
                            .into_iter()
                            .map(Fragment::try_from)
                            .collect::<Result<Vec<_>>>()?,
                        new_fragments: new_fragments
                            .into_iter()
                            .map(Fragment::try_from)
                            .collect::<Result<Vec<_>>>()?,
                    }]
                };
                let rewritten_indices = rewritten_indices
                    .iter()
                    .map(RewrittenIndex::try_from)
                    .collect::<Result<_>>()?;

                Operation::Rewrite {
                    groups,
                    rewritten_indices,
                    frag_reuse_index: None,
                }
            }
            Some(pb::transaction::Operation::CreateIndex(pb::transaction::CreateIndex {
                new_indices,
                removed_indices,
            })) => Operation::CreateIndex {
                new_indices: new_indices
                    .into_iter()
                    .map(IndexMetadata::try_from)
                    .collect::<Result<_>>()?,
                removed_indices: removed_indices
                    .into_iter()
                    .map(IndexMetadata::try_from)
                    .collect::<Result<_>>()?,
            },
            Some(pb::transaction::Operation::Merge(pb::transaction::Merge {
                fragments,
                schema,
                schema_metadata: _schema_metadata, // TODO: handle metadata
                preserves_nullability,
            })) => Operation::Merge {
                fragments: fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                schema: Schema::try_from(&Fields(schema))?,
                // False for a writer that predates the field: no assertion, so
                // a legacy required-field merge still conflicts and a legacy
                // nullable merge over-conflicts, which only retries.
                preserves_nullability,
            },
            Some(pb::transaction::Operation::Restore(pb::transaction::Restore { version })) => {
                Operation::Restore { version }
            }
            Some(pb::transaction::Operation::Update(pb::transaction::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified,
                compacted_sstables,
                fields_for_preserving_frag_bitmap,
                update_mode,
                inserted_rows,
                updated_fragment_offsets,
                updated_fragment_offset_bitmaps,
                mem_wal_dir_id,
            })) => Operation::Update {
                removed_fragment_ids,
                updated_fragments: updated_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                new_fragments: new_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                fields_modified,
                compacted_sstables: compacted_sstables
                    .into_iter()
                    .map(|m| CompactedSsTable::try_from(m).unwrap())
                    .collect(),
                mem_wal_dir_id,
                fields_for_preserving_frag_bitmap,
                update_mode: match update_mode {
                    0 => Some(UpdateMode::RewriteRows),
                    1 => Some(UpdateMode::RewriteColumns),
                    _ => Some(UpdateMode::RewriteRows),
                },
                inserted_rows_filter: inserted_rows
                    .map(|ik| KeyExistenceFilter::try_from(&ik))
                    .transpose()?,
                updated_fragment_offsets: {
                    // Prefer field 10 (RoaringBitmap bytes); fall back to field 9 (UInt32List)
                    // for manifests written before this change.
                    let m: HashMap<u64, RoaringBitmap> =
                        if !updated_fragment_offset_bitmaps.is_empty() {
                            updated_fragment_offset_bitmaps
                                .into_iter()
                                .filter(|(_, bytes)| !bytes.is_empty())
                                .map(|(id, bytes)| {
                                    let bitmap = RoaringBitmap::deserialize_from(bytes.as_slice())
                                        .map_err(|e| {
                                            Error::invalid_input(format!(
                                                "invalid updated_fragment_offset_bitmaps \
                                                     for fragment {id}: {e}"
                                            ))
                                        })?;
                                    Ok((id, bitmap))
                                })
                                .collect::<Result<HashMap<_, _>>>()?
                        } else {
                            updated_fragment_offsets
                                .into_iter()
                                .filter(|(_, list)| !list.values.is_empty())
                                .map(|(id, list)| (id, RoaringBitmap::from_iter(list.values)))
                                .collect()
                        };
                    if m.is_empty() {
                        None
                    } else {
                        Some(UpdatedFragmentOffsets(m))
                    }
                },
            },
            Some(pb::transaction::Operation::Project(pb::transaction::Project {
                schema,
                preserves_nullability,
            })) => Operation::Project {
                schema: Schema::try_from(&Fields(schema))?,
                // False for a writer that predates the field: no assertion, so
                // a legacy tightening still conflicts and a legacy rename
                // over-conflicts, which only retries.
                preserves_nullability,
            },
            Some(pb::transaction::Operation::UpdateConfig(update_config)) => {
                // Check if new-style fields are present
                let has_new_fields = update_config.config_updates.is_some()
                    || update_config.table_metadata_updates.is_some()
                    || update_config.schema_metadata_updates.is_some()
                    || !update_config.field_metadata_updates.is_empty();

                // Check if old-style fields are present
                let has_old_fields = !update_config.upsert_values.is_empty()
                    || !update_config.delete_keys.is_empty()
                    || !update_config.schema_metadata.is_empty()
                    || !update_config.field_metadata.is_empty();

                // Error if both are present
                if has_new_fields && has_old_fields {
                    return Err(Error::invalid_input_source(
                        "Cannot mix old and new style UpdateConfig fields".into(),
                    ));
                }

                if has_old_fields {
                    // Translate old-style to new-style
                    let config_updates = if !update_config.upsert_values.is_empty()
                        || !update_config.delete_keys.is_empty()
                    {
                        Some(translate_config_updates(
                            &update_config.upsert_values,
                            &update_config.delete_keys,
                        ))
                    } else {
                        None
                    };

                    let schema_metadata_updates = if !update_config.schema_metadata.is_empty() {
                        Some(translate_schema_metadata_updates(
                            &update_config.schema_metadata,
                        ))
                    } else {
                        None
                    };

                    let field_metadata_updates = update_config
                        .field_metadata
                        .into_iter()
                        .map(|(field_id, field_meta_update)| {
                            (
                                field_id as i32,
                                translate_schema_metadata_updates(&field_meta_update.metadata),
                            )
                        })
                        .collect();

                    Operation::UpdateConfig {
                        config_updates,
                        table_metadata_updates: None,
                        schema_metadata_updates,
                        field_metadata_updates,
                    }
                } else {
                    // Use new-style fields directly (convert from protobuf)
                    Operation::UpdateConfig {
                        config_updates: update_config.config_updates.as_ref().map(UpdateMap::from),
                        table_metadata_updates: update_config
                            .table_metadata_updates
                            .as_ref()
                            .map(UpdateMap::from),
                        schema_metadata_updates: update_config
                            .schema_metadata_updates
                            .as_ref()
                            .map(UpdateMap::from),
                        field_metadata_updates: update_config
                            .field_metadata_updates
                            .iter()
                            .map(|(field_id, pb_update_map)| {
                                (*field_id, UpdateMap::from(pb_update_map))
                            })
                            .collect(),
                    }
                }
            }
            // `data_change` is not acted on yet: every replacement is still
            // applied as a data change.
            Some(pb::transaction::Operation::DataReplacement(
                pb::transaction::DataReplacement { replacements, .. },
            )) => Operation::DataReplacement {
                replacements: replacements
                    .into_iter()
                    .map(DataReplacementGroup::try_from)
                    .collect::<Result<Vec<_>>>()?,
            },
            Some(pb::transaction::Operation::UpdateMemWalState(
                pb::transaction::UpdateMemWalState {
                    compacted_sstables,
                    dir_id,
                },
            )) => Operation::UpdateMemWalState {
                compacted_sstables: compacted_sstables
                    .into_iter()
                    .map(CompactedSsTable::try_from)
                    .collect::<Result<_>>()?,
                dir_id,
            },
            Some(pb::transaction::Operation::UpdateBases(pb::transaction::UpdateBases {
                new_bases,
            })) => Operation::UpdateBases {
                new_bases: new_bases.into_iter().map(BasePath::from).collect(),
            },
            Some(pb::transaction::Operation::DataOverlay(pb::transaction::DataOverlay {
                groups,
            })) => Operation::DataOverlay {
                groups: groups
                    .into_iter()
                    .map(DataOverlayGroup::try_from)
                    .collect::<Result<Vec<_>>>()?,
            },
            Some(pb::transaction::Operation::CompositeOperation(composite_operation)) => {
                // Action-based transactions (Transaction V2) are an experimental
                // wire format (discussion #5960). Parsing is fail-closed: an action this build
                // does not implement is an error, never a skipped element.
                // load_and_sort_new_transactions collects concurrent transactions
                // with try_collect, so such a transaction aborts the commit rather
                // than being silently treated as a no-op. Do NOT make this lenient.
                Operation::CompositeOperation(CompositeOperation::try_from(composite_operation)?)
            }
            // prost drops unrecognized oneof fields, so an operation from a
            // newer writer is indistinguishable from a missing one.
            None => Operation::Unknown {},
        };
        Ok(Self {
            read_version: message.read_version,
            uuid: message.uuid.clone(),
            operation,
            tag: if message.tag.is_empty() {
                None
            } else {
                Some(message.tag.clone())
            },
            transaction_properties: if message.transaction_properties.is_empty() {
                None
            } else {
                Some(Arc::new(message.transaction_properties))
            },
        })
    }
}

impl TryFrom<&pb::transaction::rewrite::RewrittenIndex> for RewrittenIndex {
    type Error = Error;

    fn try_from(message: &pb::transaction::rewrite::RewrittenIndex) -> Result<Self> {
        Ok(Self {
            old_id: message
                .old_id
                .as_ref()
                .map(Uuid::try_from)
                .ok_or_else(|| {
                    Error::invalid_input("required field (old_id) missing from message".to_string())
                })??,
            new_id: message
                .new_id
                .as_ref()
                .map(Uuid::try_from)
                .ok_or_else(|| {
                    Error::invalid_input("required field (new_id) missing from message".to_string())
                })??,
            new_index_details: message
                .new_index_details
                .as_ref()
                .ok_or_else(|| {
                    Error::invalid_input("new_index_details is a required field".to_string())
                })?
                .clone(),
            new_index_version: message.new_index_version,
            new_index_files: if message.new_index_files.is_empty() {
                None
            } else {
                Some(
                    message
                        .new_index_files
                        .iter()
                        .map(|f| IndexFile {
                            path: f.path.clone(),
                            size_bytes: f.size_bytes,
                            file_metadata_size_bytes: NonZero::new(f.file_metadata_size_bytes),
                        })
                        .collect(),
                )
            },
        })
    }
}

impl TryFrom<pb::transaction::rewrite::RewriteGroup> for RewriteGroup {
    type Error = Error;

    fn try_from(message: pb::transaction::rewrite::RewriteGroup) -> Result<Self> {
        Ok(Self {
            old_fragments: message
                .old_fragments
                .into_iter()
                .map(Fragment::try_from)
                .collect::<Result<Vec<_>>>()?,
            new_fragments: message
                .new_fragments
                .into_iter()
                .map(Fragment::try_from)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl TryFrom<&Transaction> for pb::Transaction {
    type Error = Error;

    fn try_from(value: &Transaction) -> Result<Self> {
        let operation = match &value.operation {
            Operation::Append { fragments } => {
                pb::transaction::Operation::Append(pb::transaction::Append {
                    fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                })
            }
            Operation::Clone {
                is_shallow,
                ref_name,
                ref_version,
                ref_path,
                branch_name,
            } => pb::transaction::Operation::Clone(pb::transaction::Clone {
                is_shallow: *is_shallow,
                ref_name: ref_name.clone(),
                ref_version: *ref_version,
                ref_path: ref_path.clone(),
                branch_name: branch_name.clone(),
            }),
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            } => pb::transaction::Operation::Delete(pb::transaction::Delete {
                updated_fragments: updated_fragments
                    .iter()
                    .map(pb::DataFragment::from)
                    .collect(),
                deleted_fragment_ids: deleted_fragment_ids.clone(),
                predicate: predicate.clone(),
            }),
            Operation::Overwrite {
                fragments,
                schema,
                config_upsert_values,
                initial_bases,
            } => {
                pb::transaction::Operation::Overwrite(pb::transaction::Overwrite {
                    fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                    schema: Fields::from(schema).0,
                    schema_metadata: Default::default(), // TODO: handle metadata
                    config_upsert_values: config_upsert_values
                        .clone()
                        .unwrap_or(Default::default()),
                    initial_bases: initial_bases
                        .as_ref()
                        .map(|paths| {
                            paths
                                .iter()
                                .cloned()
                                .map(|bp: BasePath| -> pb::BasePath { bp.into() })
                                .collect::<Vec<pb::BasePath>>()
                        })
                        .unwrap_or_default(),
                })
            }
            Operation::ReserveFragments { num_fragments } => {
                pb::transaction::Operation::ReserveFragments(pb::transaction::ReserveFragments {
                    num_fragments: *num_fragments,
                })
            }
            Operation::Rewrite {
                groups,
                rewritten_indices,
                frag_reuse_index: _,
            } => pb::transaction::Operation::Rewrite(pb::transaction::Rewrite {
                groups: groups
                    .iter()
                    .map(pb::transaction::rewrite::RewriteGroup::from)
                    .collect(),
                rewritten_indices: rewritten_indices
                    .iter()
                    .map(|rewritten| rewritten.into())
                    .collect(),
                ..Default::default()
            }),
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            } => pb::transaction::Operation::CreateIndex(pb::transaction::CreateIndex {
                new_indices: new_indices.iter().map(pb::IndexMetadata::from).collect(),
                removed_indices: removed_indices
                    .iter()
                    .map(pb::IndexMetadata::from)
                    .collect(),
            }),
            Operation::Merge {
                fragments,
                schema,
                preserves_nullability,
            } => pb::transaction::Operation::Merge(pb::transaction::Merge {
                fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                schema: Fields::from(schema).0,
                schema_metadata: Default::default(), // TODO: handle metadata
                preserves_nullability: *preserves_nullability,
            }),
            Operation::Restore { version } => {
                pb::transaction::Operation::Restore(pb::transaction::Restore { version: *version })
            }
            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified,
                compacted_sstables,
                mem_wal_dir_id,
                fields_for_preserving_frag_bitmap,
                update_mode,
                inserted_rows_filter,
                updated_fragment_offsets,
            } => pb::transaction::Operation::Update(pb::transaction::Update {
                removed_fragment_ids: removed_fragment_ids.clone(),
                updated_fragments: updated_fragments
                    .iter()
                    .map(pb::DataFragment::from)
                    .collect(),
                new_fragments: new_fragments.iter().map(pb::DataFragment::from).collect(),
                fields_modified: fields_modified.clone(),
                compacted_sstables: compacted_sstables
                    .iter()
                    .map(pb::CompactedSsTable::from)
                    .collect(),
                mem_wal_dir_id: mem_wal_dir_id.clone(),
                fields_for_preserving_frag_bitmap: fields_for_preserving_frag_bitmap.clone(),
                update_mode: update_mode
                    .as_ref()
                    .map(|mode| match mode {
                        UpdateMode::RewriteRows => 0,
                        UpdateMode::RewriteColumns => 1,
                    })
                    .unwrap_or(0),
                inserted_rows: inserted_rows_filter.as_ref().map(|ik| ik.into()),
                // Field 9: no longer written; kept empty for forward compat.
                updated_fragment_offsets: HashMap::new(),
                // Field 10: RoaringBitmap bytes.
                updated_fragment_offset_bitmaps: updated_fragment_offsets
                    .as_ref()
                    .map(|UpdatedFragmentOffsets(m)| {
                        m.iter()
                            .filter(|(_, b)| !b.is_empty())
                            .map(|(frag_id, b)| {
                                let mut buf = Vec::new();
                                b.serialize_into(&mut buf)
                                    .expect("RoaringBitmap serialization cannot fail");
                                (*frag_id, buf)
                            })
                            .collect::<HashMap<_, _>>()
                    })
                    .unwrap_or_default(),
            }),
            Operation::Project {
                schema,
                preserves_nullability,
            } => pb::transaction::Operation::Project(pb::transaction::Project {
                schema: Fields::from(schema).0,
                preserves_nullability: *preserves_nullability,
            }),
            Operation::UpdateConfig {
                config_updates,
                table_metadata_updates,
                schema_metadata_updates,
                field_metadata_updates,
            } => pb::transaction::Operation::UpdateConfig(pb::transaction::UpdateConfig {
                config_updates: config_updates.as_ref().map(pb_actions::UpdateMap::from),
                table_metadata_updates: table_metadata_updates
                    .as_ref()
                    .map(pb_actions::UpdateMap::from),
                schema_metadata_updates: schema_metadata_updates
                    .as_ref()
                    .map(pb_actions::UpdateMap::from),
                field_metadata_updates: field_metadata_updates
                    .iter()
                    .map(|(field_id, update_map)| {
                        (*field_id, pb_actions::UpdateMap::from(update_map))
                    })
                    .collect(),
                // Leave old fields empty - we only write new-style fields
                upsert_values: Default::default(),
                delete_keys: Default::default(),
                schema_metadata: Default::default(),
                field_metadata: Default::default(),
            }),
            Operation::DataReplacement { replacements } => {
                pb::transaction::Operation::DataReplacement(pb::transaction::DataReplacement {
                    replacements: replacements
                        .iter()
                        .map(pb::transaction::DataReplacementGroup::from)
                        .collect(),
                    data_change: None,
                })
            }
            Operation::DataOverlay { groups } => {
                pb::transaction::Operation::DataOverlay(pb::transaction::DataOverlay {
                    groups: groups
                        .iter()
                        .map(pb::transaction::DataOverlayGroup::from)
                        .collect(),
                })
            }
            Operation::UpdateMemWalState {
                compacted_sstables,
                dir_id,
            } => {
                pb::transaction::Operation::UpdateMemWalState(pb::transaction::UpdateMemWalState {
                    compacted_sstables: compacted_sstables
                        .iter()
                        .map(pb::CompactedSsTable::from)
                        .collect::<Vec<_>>(),
                    dir_id: dir_id.clone(),
                })
            }
            Operation::UpdateBases { new_bases } => {
                pb::transaction::Operation::UpdateBases(pb::transaction::UpdateBases {
                    new_bases: new_bases
                        .iter()
                        .cloned()
                        .map(|bp: BasePath| -> pb::BasePath { bp.into() })
                        .collect::<Vec<pb::BasePath>>(),
                })
            }
            Operation::Unknown { .. } => {
                return Err(Error::not_supported(format!(
                    "Transaction {} has an operation written by a newer version of Lance \
                     and cannot be re-encoded by this version",
                    value.uuid
                )));
            }
            Operation::CompositeOperation(composite_operation) => {
                pb::transaction::Operation::CompositeOperation(
                    pb_actions::CompositeOperation::from(composite_operation),
                )
            }
        };

        let transaction_properties = value
            .transaction_properties
            .as_ref()
            .map(|arc| arc.as_ref().clone())
            .unwrap_or_default();
        Ok(Self {
            read_version: value.read_version,
            uuid: value.uuid.clone(),
            operation: Some(operation),
            tag: value.tag.clone().unwrap_or("".to_string()),
            transaction_properties,
        })
    }
}

impl From<&RewrittenIndex> for pb::transaction::rewrite::RewrittenIndex {
    fn from(value: &RewrittenIndex) -> Self {
        Self {
            old_id: Some((&value.old_id).into()),
            new_id: Some((&value.new_id).into()),
            new_index_details: Some(value.new_index_details.clone()),
            new_index_version: value.new_index_version,
            new_index_files: value
                .new_index_files
                .as_ref()
                .map(|files| {
                    files
                        .iter()
                        .map(|f| pb::IndexFile {
                            path: f.path.clone(),
                            size_bytes: f.size_bytes,
                            file_metadata_size_bytes: f
                                .file_metadata_size_bytes
                                .map_or(0, NonZero::get),
                        })
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

impl From<&RewriteGroup> for pb::transaction::rewrite::RewriteGroup {
    fn from(value: &RewriteGroup) -> Self {
        Self {
            old_fragments: value
                .old_fragments
                .iter()
                .map(pb::DataFragment::from)
                .collect(),
            new_fragments: value
                .new_fragments
                .iter()
                .map(pb::DataFragment::from)
                .collect(),
        }
    }
}

impl From<&UpdateMap> for pb_actions::UpdateMap {
    fn from(update_map: &UpdateMap) -> Self {
        Self {
            update_entries: update_map
                .update_entries
                .iter()
                .map(|entry| pb_actions::UpdateMapEntry {
                    key: entry.key.clone(),
                    value: entry.value.clone(),
                })
                .collect(),
            replace: update_map.replace,
        }
    }
}

impl From<&pb_actions::UpdateMap> for UpdateMap {
    fn from(pb_update_map: &pb_actions::UpdateMap) -> Self {
        Self {
            update_entries: pb_update_map
                .update_entries
                .iter()
                .map(|entry| UpdateMapEntry {
                    key: entry.key.clone(),
                    value: entry.value.clone(),
                })
                .collect(),
            replace: pb_update_map.replace,
        }
    }
}

impl TryFrom<&Transaction> for crate::format::Transaction {
    type Error = Error;

    fn try_from(value: &Transaction) -> Result<Self> {
        Ok(Self {
            inner: pb::Transaction::try_from(value)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::format::overlay::OverlayCoverage;
    use crate::transaction::action::{
        Action, AddDataFile, AddFragment, FieldRef, Ref, RemoveFragment, UserStep,
    };
    use crate::transaction::operation::RewriteGroup;
    use lance_file::version::ConcreteFileVersion;
    use prost::Message;
    use rstest::rstest;

    #[test]
    fn test_rewrite_frag_reuse_update_stays_in_memory() {
        // The fragment reuse update never enters the transaction file:
        // other writers' conflict decisions only need the fragment sets in
        // `groups`.
        let entry = crate::transaction::test_support::sample_index_metadata(
            crate::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME,
        );
        let transaction = Transaction::new(
            1,
            Operation::Rewrite {
                groups: vec![],
                rewritten_indices: vec![],
                frag_reuse_index: Some(entry),
            },
            None,
        );
        let decoded =
            Transaction::try_from(pb::Transaction::try_from(&transaction).unwrap()).unwrap();
        match decoded.operation {
            Operation::Rewrite {
                frag_reuse_index, ..
            } => {
                assert!(frag_reuse_index.is_none());
            }
            other => panic!("expected Rewrite, got {other:?}"),
        }
    }

    #[test]
    fn test_data_overlay_operation_roundtrips() {
        // A DataOverlay operation survives the protobuf round-trip, preserving
        // the target fragment, the overlay's coverage, and its committed_version.
        let mut bitmap = roaring::RoaringBitmap::new();
        bitmap.insert(1);
        bitmap.insert(4);
        let overlay = DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("overlay-0.lance", vec![3], None),
            coverage: OverlayCoverage::dense(bitmap.clone()),
            committed_version: 6,
        };
        let pb_overlay = pb::DataOverlayFile::from(&overlay);

        let message = pb::Transaction {
            read_version: 1,
            uuid: Uuid::new_v4().to_string(),
            operation: Some(pb::transaction::Operation::DataOverlay(
                pb::transaction::DataOverlay {
                    groups: vec![pb::transaction::DataOverlayGroup {
                        fragment_id: 7,
                        overlays: vec![pb_overlay],
                    }],
                },
            )),
            ..Default::default()
        };

        let txn = Transaction::try_from(message).unwrap();
        match txn.operation {
            Operation::DataOverlay { groups } => {
                assert_eq!(groups.len(), 1);
                assert_eq!(groups[0].fragment_id, 7);
                assert_eq!(groups[0].overlays.len(), 1);
                assert_eq!(groups[0].overlays[0].committed_version, 6);
                assert_eq!(
                    *groups[0].overlays[0].coverage_for_field(0).unwrap(),
                    bitmap
                );
            }
            other => panic!("expected DataOverlay, got {other:?}"),
        }
    }

    #[test]
    fn test_rewritten_index_roundtrips() {
        let rewritten_index = RewrittenIndex {
            old_id: Uuid::new_v4(),
            new_id: Uuid::new_v4(),
            new_index_details: prost_types::Any::default(),
            new_index_version: 1,
            new_index_files: Some(vec![
                IndexFile {
                    path: "index.idx".to_string(),
                    size_bytes: 4_096,
                    file_metadata_size_bytes: NonZero::new(256),
                },
                IndexFile {
                    path: "auxiliary.idx".to_string(),
                    size_bytes: 1_024,
                    file_metadata_size_bytes: None,
                },
            ]),
        };

        let message = pb::transaction::rewrite::RewrittenIndex::from(&rewritten_index);
        assert_eq!(RewrittenIndex::try_from(&message).unwrap(), rewritten_index);
    }

    #[test]
    fn test_composite_operation_round_trips_through_transaction() {
        let uuid = Uuid::new_v4().to_string();
        let transaction = Transaction {
            read_version: 4,
            uuid: uuid.clone(),
            operation: Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
                "append batch",
                vec![Action::AddFragment(AddFragment {
                    id: Ref::Local(0),
                    physical_rows: 1,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                })],
            )])),
            tag: None,
            transaction_properties: None,
        };

        let message = pb::Transaction::try_from(&transaction).unwrap();
        // The identity lives on the envelope alone. The operation carries the
        // steps, and their descriptions are what keeps history readable.
        assert_eq!(message.uuid, uuid);
        assert_eq!(message.read_version, 4);
        match &message.operation {
            Some(pb::transaction::Operation::CompositeOperation(composite_operation)) => {
                let descriptions: Vec<&str> = composite_operation
                    .steps
                    .iter()
                    .map(|step| step.description.as_str())
                    .collect();
                assert_eq!(descriptions, ["append batch"]);
            }
            other => panic!("expected CompositeOperation, got {other:?}"),
        }

        assert_eq!(Transaction::try_from(message).unwrap(), transaction);
    }

    #[test]
    fn test_unrecognized_action_fails_closed_on_load() {
        // An action a newer Lance wrote decodes to no known variant, because
        // protobuf drops the field this build does not know. Loading it must
        // fail rather than parse leniently: load_and_sort_new_transactions
        // collects concurrent transactions with try_collect, so this aborts an
        // in-flight commit instead of letting it proceed against a change it
        // cannot see.
        let message = pb::Transaction {
            read_version: 1,
            uuid: Uuid::new_v4().to_string(),
            operation: Some(pb::transaction::Operation::CompositeOperation(
                pb_actions::CompositeOperation {
                    steps: vec![pb_actions::UserStep {
                        description: "something from the future".to_string(),
                        actions: vec![pb_actions::Action { action: None }],
                    }],
                },
            )),
            ..Default::default()
        };

        let err = Transaction::try_from(message).unwrap_err();
        assert!(
            matches!(err, Error::NotSupported { .. }),
            "expected NotSupported, got: {err:?}"
        );
    }

    const WIDE_COLUMNS: i32 = 200;

    fn wide_data_file(index: usize, fields: Vec<i32>) -> DataFile {
        DataFile::new(
            format!("data/{:032x}.lance", index),
            fields,
            (0..WIDE_COLUMNS).collect(),
            ConcreteFileVersion::V2_0,
            std::num::NonZero::new(64 << 20),
            None,
        )
    }

    fn wide_fragment(id: u64) -> Fragment {
        let mut fragment = Fragment::new(id);
        fragment.physical_rows = Some(1_000_000);
        fragment.files = vec![wide_data_file(id as usize, (0..WIDE_COLUMNS).collect())];
        fragment
    }

    /// The composite spelling of minting `count` wide fragments: `file.fields`
    /// empty and `field_ids` the file's only field list.
    fn mint_wide_fragments(count: usize, data_change: bool) -> Vec<Action> {
        (0..count)
            .flat_map(|index| {
                let token = index as u32;
                [
                    Action::AddFragment(AddFragment {
                        id: Ref::Local(token),
                        physical_rows: 1_000_000,
                        row_id_meta: None,
                        last_updated_at_version_meta: None,
                        created_at_version_meta: None,
                        data_change,
                    }),
                    Action::AddDataFile(AddDataFile {
                        fragment: Ref::Local(token),
                        file: wide_data_file(index, Vec::new()),
                        field_ids: (0..WIDE_COLUMNS).map(FieldRef::Committed).collect(),
                        data_change,
                        base: None,
                    }),
                ]
            })
            .collect()
    }

    fn encoded_len(operation: Operation) -> usize {
        let transaction = Transaction::new(1, operation, None);
        pb::Transaction::try_from(&transaction)
            .unwrap()
            .encoded_len()
    }

    fn composite(actions: Vec<Action>) -> Operation {
        Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
            "step", actions,
        )]))
    }

    /// A wide table makes the per-column cost of an action set visible: 100
    /// fragments of 200 columns each, appended, and compacted 10 to 1.
    ///
    /// Each bound is what the format costs today plus ~10%, so it only catches
    /// growth. What remains of the append's overhead is `field_ids`: a
    /// `FieldRef` per column costs a few bytes more than a packed `int32`.
    #[rstest]
    #[case::append(
        Operation::Append { fragments: (0..100).map(wide_fragment).collect() },
        composite(mint_wide_fragments(100, true)),
        2.3,
    )]
    #[case::compaction(
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: (0..100).map(wide_fragment).collect(),
                new_fragments: (100..110).map(wide_fragment).collect(),
            }],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        },
        composite(
            (0..100)
                .map(|id| {
                    Action::RemoveFragment(RemoveFragment {
                        fragment: Ref::Committed(id),
                        data_change: false,
                    })
                })
                .chain(mint_wide_fragments(10, false))
                .collect(),
        ),
        0.23,
    )]
    fn test_a_wide_action_set_stays_within_a_bound_of_the_legacy_size(
        #[case] legacy: Operation,
        #[case] actions: Operation,
        #[case] max_ratio: f64,
    ) {
        let legacy = encoded_len(legacy);
        let actions = encoded_len(actions);
        assert!(
            actions as f64 <= legacy as f64 * max_ratio,
            "the action set encodes to {actions} bytes, more than {max_ratio}x the legacy \
             operation's {legacy}"
        );
    }
}
