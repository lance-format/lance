// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Update-map data types used by the [`UpdateConfig`](crate::Action) family of
//! actions and the legacy `Operation::UpdateConfig` / `Operation::Update`
//! variants.

use std::collections::HashMap;

use crate::format::pb;
use deepsize::DeepSizeOf;
use roaring::RoaringBitmap;

/// An entry for a map update. If value is None, the key will be removed from the map.
#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct UpdateMapEntry {
    /// The key of the map entry to update.
    pub key: String,
    /// The value to set for the key.
    pub value: Option<String>,
}

impl From<(String, Option<String>)> for UpdateMapEntry {
    fn from((key, value): (String, Option<String>)) -> Self {
        Self { key, value }
    }
}

impl From<(String, String)> for UpdateMapEntry {
    fn from((key, value): (String, String)) -> Self {
        Self::from((key, Some(value)))
    }
}

impl From<(&str, Option<&str>)> for UpdateMapEntry {
    fn from((key, value): (&str, Option<&str>)) -> Self {
        Self {
            key: key.to_string(),
            value: value.map(str::to_owned),
        }
    }
}

impl From<(&str, &str)> for UpdateMapEntry {
    fn from((key, value): (&str, &str)) -> Self {
        Self::from((key, Some(value)))
    }
}

/// Represents updates to a map (either incremental or replacement)
#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct UpdateMap {
    pub update_entries: Vec<UpdateMapEntry>,
    /// If true, the map will be replaced entirely with the new entries.
    /// If false, the new entries will be merged with the existing map.
    pub replace: bool,
}

#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub enum UpdateMode {
    /// rows are deleted in current fragments and rewritten in new fragments.
    /// This is most optimal when the majority of columns are being rewritten
    /// or only a few rows are being updated.
    RewriteRows,

    /// within each fragment, columns are fully rewritten and inserted as new data files.
    /// Old versions of columns are tombstoned. This is most optimal when most rows are affected
    /// but a small subset of columns are affected.
    RewriteColumns,
}

/// Matched physical row offsets per fragment for a partial [`UpdateMode::RewriteColumns`] update.
///
/// Used with stable row IDs so `build_manifest` can refresh row-level version
/// metadata only for rows that were rewritten.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UpdatedFragmentOffsets(pub HashMap<u64, RoaringBitmap>);

impl DeepSizeOf for UpdatedFragmentOffsets {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.0.iter().fold(0_usize, |acc, (frag_id, bitmap)| {
            acc + frag_id.deep_size_of_children(context)
                + (bitmap.len() as usize).saturating_mul(std::mem::size_of::<u32>())
        })
    }
}

/// Helper function to apply UpdateMap changes to a HashMap<String, String>
pub fn apply_update_map(target: &mut HashMap<String, String>, update_map: &UpdateMap) {
    if update_map.replace {
        // Full replacement - clear existing and replace with new entries that have values
        target.clear();
        for entry in &update_map.update_entries {
            if let Some(value) = &entry.value {
                target.insert(entry.key.clone(), value.clone());
            }
        }
    } else {
        // Incremental update - merge entries
        for entry in &update_map.update_entries {
            if let Some(value) = &entry.value {
                target.insert(entry.key.clone(), value.clone());
            } else {
                target.remove(&entry.key);
            }
        }
    }
}

/// Helper function to translate old-style config updates to new UpdateMap format
pub fn translate_config_updates(
    upsert_values: &HashMap<String, String>,
    delete_keys: &[String],
) -> UpdateMap {
    let mut update_entries = Vec::new();

    // Add upsert entries (with values)
    for (key, value) in upsert_values {
        update_entries.push(UpdateMapEntry {
            key: key.clone(),
            value: Some(value.clone()),
        });
    }

    // Add delete entries (without values)
    for key in delete_keys {
        update_entries.push(UpdateMapEntry {
            key: key.clone(),
            value: None,
        });
    }

    UpdateMap {
        update_entries,
        replace: false, // Old style was always incremental
    }
}

/// Helper function to translate old-style schema metadata to new UpdateMap format
pub fn translate_schema_metadata_updates(schema_metadata: &HashMap<String, String>) -> UpdateMap {
    let update_entries = schema_metadata
        .iter()
        .map(|(key, value)| UpdateMapEntry {
            key: key.clone(),
            value: Some(value.clone()),
        })
        .collect();

    UpdateMap {
        update_entries,
        replace: true, // Old style schema metadata was full replacement
    }
}

impl From<&UpdateMap> for pb::transaction::UpdateMap {
    fn from(update_map: &UpdateMap) -> Self {
        Self {
            update_entries: update_map
                .update_entries
                .iter()
                .map(|entry| pb::transaction::UpdateMapEntry {
                    key: entry.key.clone(),
                    value: entry.value.clone(),
                })
                .collect(),
            replace: update_map.replace,
        }
    }
}

impl From<&pb::transaction::UpdateMap> for UpdateMap {
    fn from(pb_update_map: &pb::transaction::UpdateMap) -> Self {
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
