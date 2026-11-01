// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Creation-time layout selection. Manifest feature flags govern compatibility.
//! This setting chooses which writer to use for a new dataset.

use std::collections::HashMap;

use lance_core::{Error, Result};

/// Table config key selecting a manifest layout.
pub const MANIFEST_LAYOUT_KEY: &str = "lance.manifest.layout";
/// Config value selecting today's flat manifest, also the default.
pub const MANIFEST_LAYOUT_FLAT: &str = "flat";
/// Config value selecting the buffered fragment tree layout.
pub const MANIFEST_LAYOUT_TREE: &str = "tree";

/// The manifest layout a table's config selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestLayout {
    /// Single-protobuf manifest. The default when the key is unset.
    Flat,
    /// Buffered fragment metadata, selected by `lance.manifest.layout=tree`.
    Tree,
}

impl ManifestLayout {
    /// Select a layout from table config, treating an unset key as flat.
    ///
    /// Rejects unknown values instead of falling back, so a typo cannot
    /// silently create a flat dataset.
    pub fn from_config(config: &HashMap<String, String>) -> Result<Self> {
        match config.get(MANIFEST_LAYOUT_KEY).map(String::as_str) {
            None | Some(MANIFEST_LAYOUT_FLAT) => Ok(Self::Flat),
            Some(MANIFEST_LAYOUT_TREE) => Ok(Self::Tree),
            Some(other) => Err(Error::invalid_input(format!(
                "unsupported {MANIFEST_LAYOUT_KEY} value: layout={other:?}, \
                 expected={MANIFEST_LAYOUT_FLAT:?} or {MANIFEST_LAYOUT_TREE:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_defaults_to_flat_and_rejects_unknown_values() {
        assert_eq!(
            ManifestLayout::from_config(&HashMap::new()).unwrap(),
            ManifestLayout::Flat
        );
        let flat = HashMap::from([(MANIFEST_LAYOUT_KEY.to_string(), "flat".to_string())]);
        assert_eq!(
            ManifestLayout::from_config(&flat).unwrap(),
            ManifestLayout::Flat
        );
        let tree = HashMap::from([(MANIFEST_LAYOUT_KEY.to_string(), "tree".to_string())]);
        assert_eq!(
            ManifestLayout::from_config(&tree).unwrap(),
            ManifestLayout::Tree
        );
        let unknown = HashMap::from([(MANIFEST_LAYOUT_KEY.to_string(), "tiered".to_string())]);
        let error = ManifestLayout::from_config(&unknown).unwrap_err();
        assert!(error.to_string().contains("layout=\"tiered\""), "{error}");
    }
}
