// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Table-level data file layout settings shared by writes and compaction.
//!
//! The settings live in the manifest config under [`LAYOUT_CONFIG_PREFIX`] so a
//! table carries one description of how its data files should be sized, instead
//! of each writer or maintenance job choosing independently.

use std::collections::HashMap;

use lance_core::{Error, Result};
use tracing::warn;

/// Config key prefix for table layout settings stored in the manifest config.
pub const LAYOUT_CONFIG_PREFIX: &str = "lance.layout.";

/// Data file layout for a table, as stored under the `lance.layout.` config
/// prefix.
///
/// Every field is optional so that "no opinion" stays distinct from an explicit
/// value that happens to equal a default. Callers combine instances by
/// precedence rather than by comparing against defaults, so an explicit
/// default-valued setting is still honored.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TableLayoutConfig {
    /// Target number of rows per data file.
    ///
    /// Writes use it as the per-file row limit; compaction uses it as the row
    /// target for the files it rewrites. Must be in `1..=u32::MAX`, the row
    /// capacity of one fragment.
    pub target_rows_per_fragment: Option<usize>,

    /// Soft bound on the size of one data file, in bytes.
    ///
    /// Writes and compaction check the bound after writing each row group, so a
    /// file can exceed it by one row group plus the footer. It is not a limit on
    /// total data size: a fragment that a task must keep as one output can still
    /// be written as a single larger file.
    pub max_bytes_per_file: Option<usize>,
}

impl TableLayoutConfig {
    /// Read the layout settings from a manifest config map.
    ///
    /// Keys outside [`LAYOUT_CONFIG_PREFIX`] are ignored, so a caller can pass a
    /// whole manifest config.
    pub fn from_dataset_config(config: &HashMap<String, String>) -> Result<Self> {
        let mut layout = Self::default();
        layout.apply_dataset_config(config)?;
        Ok(layout)
    }

    /// Apply the `lance.layout.` entries of a manifest config map to this
    /// instance.
    ///
    /// Only fields with a corresponding key are modified; other fields keep the
    /// value they already hold.
    pub fn apply_dataset_config(&mut self, config: &HashMap<String, String>) -> Result<()> {
        for (key, value) in config {
            let Some(field) = key.strip_prefix(LAYOUT_CONFIG_PREFIX) else {
                continue;
            };
            match field {
                "target_rows_per_fragment" => {
                    let rows: usize = value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?;
                    // A zero row target cannot be honored and a fragment cannot
                    // hold more than `u32::MAX` rows, so reject both here rather
                    // than at the first write that tries to use them.
                    if rows == 0 || rows > u32::MAX as usize {
                        return Err(Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected 1 to {}, the row capacity of one fragment)",
                            key,
                            value,
                            u32::MAX
                        )));
                    }
                    self.target_rows_per_fragment = Some(rows);
                }
                "max_bytes_per_file" => {
                    let bytes: usize = value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?;
                    if bytes == 0 {
                        return Err(Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a positive integer)",
                            key, value
                        )));
                    }
                    self.max_bytes_per_file = Some(bytes);
                }
                _ => {
                    warn!("Ignoring unknown table layout config key: {}", key);
                }
            }
        }
        Ok(())
    }

    /// Fill every unset field from `fallback`.
    ///
    /// Fields set on `self` win, including values that equal a default, because
    /// presence is what carries precedence.
    pub fn fallback_to(self, fallback: Self) -> Self {
        Self {
            target_rows_per_fragment: self
                .target_rows_per_fragment
                .or(fallback.target_rows_per_fragment),
            max_bytes_per_file: self.max_bytes_per_file.or(fallback.max_bytes_per_file),
        }
    }

    /// The manifest config entries for the fields that are set.
    ///
    /// Used to persist the layout of a new table in the same transaction as its
    /// first data files.
    pub fn config_values(&self) -> HashMap<String, String> {
        let mut values = HashMap::new();
        if let Some(rows) = self.target_rows_per_fragment {
            values.insert(
                format!("{LAYOUT_CONFIG_PREFIX}target_rows_per_fragment"),
                rows.to_string(),
            );
        }
        if let Some(bytes) = self.max_bytes_per_file {
            values.insert(
                format!("{LAYOUT_CONFIG_PREFIX}max_bytes_per_file"),
                bytes.to_string(),
            );
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn config(entries: &[(&str, &str)]) -> HashMap<String, String> {
        entries
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    #[test]
    fn parses_layout_keys_and_ignores_others() {
        let layout = TableLayoutConfig::from_dataset_config(&config(&[
            ("lance.layout.target_rows_per_fragment", "4096"),
            ("lance.layout.max_bytes_per_file", "1048576"),
            ("lance.compaction.target_rows_per_fragment", "8"),
            ("some.other.key", "value"),
        ]))
        .unwrap();

        assert_eq!(
            layout,
            TableLayoutConfig {
                target_rows_per_fragment: Some(4096),
                max_bytes_per_file: Some(1_048_576),
            }
        );
        // The write path must not pick up compaction-only keys.
        assert_eq!(
            TableLayoutConfig::from_dataset_config(&config(&[(
                "lance.compaction.target_rows_per_fragment",
                "8"
            )]))
            .unwrap(),
            TableLayoutConfig::default()
        );
    }

    #[test]
    fn parses_partial_layout() {
        let layout = TableLayoutConfig::from_dataset_config(&config(&[(
            "lance.layout.target_rows_per_fragment",
            "16",
        )]))
        .unwrap();

        assert_eq!(layout.target_rows_per_fragment, Some(16));
        assert_eq!(layout.max_bytes_per_file, None);
        assert_eq!(layout.config_values().len(), 1);
        assert!(TableLayoutConfig::default().config_values().is_empty());
    }

    #[rstest]
    #[case::non_numeric("target_rows_per_fragment", "many")]
    #[case::zero_rows("target_rows_per_fragment", "0")]
    #[case::rows_above_fragment_capacity("target_rows_per_fragment", "4294967296")]
    #[case::negative_rows("target_rows_per_fragment", "-1")]
    #[case::non_numeric_bytes("max_bytes_per_file", "big")]
    #[case::zero_bytes("max_bytes_per_file", "0")]
    fn rejects_invalid_layout_value(#[case] field: &str, #[case] value: &str) {
        let key = format!("{LAYOUT_CONFIG_PREFIX}{field}");
        let error = TableLayoutConfig::from_dataset_config(&config(&[(&key, value)])).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        let message = error.to_string();
        assert!(message.contains(&key), "{message}");
        assert!(message.contains(value), "{message}");
    }

    #[test]
    fn explicit_values_win_regardless_of_value() {
        let explicit = TableLayoutConfig {
            // Equal to `WriteParams::default().max_rows_per_file` on purpose:
            // presence, not the value, decides precedence.
            target_rows_per_fragment: Some(1024 * 1024),
            max_bytes_per_file: None,
        };
        let persisted = TableLayoutConfig {
            target_rows_per_fragment: Some(50),
            max_bytes_per_file: Some(1024),
        };

        let resolved = explicit.fallback_to(persisted);
        assert_eq!(resolved.target_rows_per_fragment, Some(1024 * 1024));
        assert_eq!(resolved.max_bytes_per_file, Some(1024));
    }

    #[test]
    fn config_values_round_trip() {
        let layout = TableLayoutConfig {
            target_rows_per_fragment: Some(4096),
            max_bytes_per_file: Some(1_048_576),
        };
        let values = layout.config_values();

        assert_eq!(
            TableLayoutConfig::from_dataset_config(&values).unwrap(),
            layout
        );
        assert_eq!(values.len(), 2);
    }
}
