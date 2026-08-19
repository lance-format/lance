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
            Some(pb::transaction::Operation::DataReplacement(
                pb::transaction::DataReplacement { replacements },
            )) => Operation::DataReplacement {
                replacements: replacements
                    .into_iter()
                    .map(DataReplacementGroup::try_from)
                    .collect::<Result<Vec<_>>>()?,
            },
            Some(pb::transaction::Operation::UpdateMemWalState(
                pb::transaction::UpdateMemWalState {
                    compacted_sstables,
                    require_index_catchup,
                },
            )) => Operation::UpdateMemWalState {
                compacted_sstables: compacted_sstables
                    .into_iter()
                    .map(CompactedSsTable::try_from)
                    .collect::<Result<_>>()?,
                // Absent is an ordinary progress update. Explicit `false` is
                // refused rather than read as absent, so a caller cannot express
                // "deactivate" -- the migration is one-way.
                require_index_catchup: match require_index_catchup {
                    Some(false) => {
                        return Err(Error::invalid_input(
                            "require_index_catchup cannot be false: MemWAL index catch-up \
                             cannot stop being required once it is",
                        ));
                    }
                    other => other.unwrap_or(false),
                },
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
                // Action-based transactions (Transaction V2) are a draft wire
                // format (OSS-1530). Parsing is fail-closed: an action this build
                // does not implement is an error, never a skipped element.
                // load_and_sort_new_transactions collects concurrent transactions
                // with try_collect, so such a transaction aborts the commit rather
                // than being silently treated as a no-op. Do NOT make this lenient.
                Operation::CompositeOperation(CompositeOperation::try_from(composite_operation)?)
            }
            None => {
                return Err(Error::internal(
                    "Transaction message did not contain an operation".to_string(),
                ));
            }
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

impl From<&Transaction> for pb::Transaction {
    fn from(value: &Transaction) -> Self {
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
                config_updates: config_updates.as_ref().map(pb::UpdateMap::from),
                table_metadata_updates: table_metadata_updates.as_ref().map(pb::UpdateMap::from),
                schema_metadata_updates: schema_metadata_updates.as_ref().map(pb::UpdateMap::from),
                field_metadata_updates: field_metadata_updates
                    .iter()
                    .map(|(field_id, update_map)| (*field_id, pb::UpdateMap::from(update_map)))
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
                require_index_catchup,
            } => {
                pb::transaction::Operation::UpdateMemWalState(pb::transaction::UpdateMemWalState {
                    compacted_sstables: compacted_sstables
                        .iter()
                        .map(pb::CompactedSsTable::from)
                        .collect::<Vec<_>>(),
                    // Written only when requesting activation, so an ordinary
                    // progress update stays byte-identical to before.
                    require_index_catchup: require_index_catchup.then_some(true),
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
            Operation::CompositeOperation(composite_operation) => {
                let mut message = pb::CompositeOperation::from(composite_operation);
                // The operation's identity and read version are the enclosing
                // transaction's; the wire carries them in both places so a
                // squashed operation keeps its own provenance.
                message.uuid = value.uuid.clone();
                message.read_version = value.read_version;
                pb::transaction::Operation::CompositeOperation(message)
            }
        };

        let transaction_properties = value
            .transaction_properties
            .as_ref()
            .map(|arc| arc.as_ref().clone())
            .unwrap_or_default();
        Self {
            read_version: value.read_version,
            uuid: value.uuid.clone(),
            operation: Some(operation),
            tag: value.tag.clone().unwrap_or("".to_string()),
            transaction_properties,
        }
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

impl From<&UpdateMap> for pb::UpdateMap {
    fn from(update_map: &UpdateMap) -> Self {
        Self {
            update_entries: update_map
                .update_entries
                .iter()
                .map(|entry| pb::UpdateMapEntry {
                    key: entry.key.clone(),
                    value: entry.value.clone(),
                })
                .collect(),
            replace: update_map.replace,
        }
    }
}

impl From<&pb::UpdateMap> for UpdateMap {
    fn from(pb_update_map: &pb::UpdateMap) -> Self {
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

impl From<&Transaction> for crate::format::Transaction {
    fn from(value: &Transaction) -> Self {
        let pb_transaction: pb::Transaction = value.into();
        Self {
            inner: pb_transaction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::format::overlay::OverlayCoverage;
    use crate::transaction::action::{Action, AddFragment, UserAction};

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
    fn test_composite_operation_round_trips_through_transaction() {
        let uuid = Uuid::new_v4().to_string();
        let transaction = Transaction {
            read_version: 4,
            uuid: uuid.clone(),
            operation: Operation::CompositeOperation(CompositeOperation::new(vec![
                UserAction::new(
                    "append batch",
                    vec![Action::AddFragment(AddFragment {
                        local: 0,
                        physical_rows: 1,
                        row_id_meta: None,
                        last_updated_at_version_meta: None,
                        created_at_version_meta: None,
                        data_change: true,
                    })],
                ),
            ])),
            tag: None,
            transaction_properties: None,
        };

        let message = pb::Transaction::from(&transaction);
        // The operation repeats the envelope's identity so a squashed operation
        // keeps the provenance of the commit it came from.
        match &message.operation {
            Some(pb::transaction::Operation::CompositeOperation(composite_operation)) => {
                assert_eq!(composite_operation.uuid, uuid);
                assert_eq!(composite_operation.read_version, 4);
            }
            other => panic!("expected CompositeOperation, got {other:?}"),
        }

        assert_eq!(Transaction::try_from(message).unwrap(), transaction);
    }

    #[test]
    fn test_unimplemented_action_fails_closed_on_load() {
        // The drafted vocabulary is larger than what is implemented. Loading a
        // transaction that uses an unimplemented action must fail rather than
        // parse leniently: load_and_sort_new_transactions collects concurrent
        // transactions with try_collect, so this aborts an in-flight commit
        // instead of letting it proceed against a change it cannot see.
        let message = pb::Transaction {
            read_version: 1,
            uuid: Uuid::new_v4().to_string(),
            operation: Some(pb::transaction::Operation::CompositeOperation(
                pb::CompositeOperation {
                    uuid: Uuid::new_v4().to_string(),
                    read_version: 1,
                    actions: vec![pb::UserAction {
                        description: "assert unique keys".to_string(),
                        actions: vec![pb::Action {
                            action: Some(pb::action::Action::AssertUniqueKeys(
                                pb::AssertUniqueKeys {
                                    key_fields: vec![],
                                    filter: None,
                                },
                            )),
                        }],
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
}
