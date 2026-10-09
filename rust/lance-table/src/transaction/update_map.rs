// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Incremental edits to the string maps a manifest carries.
//!
//! Dataset config, table metadata, schema metadata and per-field metadata are all
//! `HashMap<String, String>`, and all four are updated the same way: a list of
//! entries where a `None` value means delete the key, plus a flag choosing between
//! merging into the existing map and replacing it outright.

use lance_core::deepsize::DeepSizeOf;

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

/// Helper function to apply UpdateMap changes to a HashMap<String, String>
pub(super) fn apply_update_map(
    target: &mut std::collections::HashMap<String, String>,
    update_map: &UpdateMap,
) {
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
    upsert_values: &std::collections::HashMap<String, String>,
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
pub fn translate_schema_metadata_updates(
    schema_metadata: &std::collections::HashMap<String, String>,
) -> UpdateMap {
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
