// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use lance_core::{Error, Result};

#[cfg(feature = "geo")]
use crate::scalar::rtree::RTreeIndexPlugin;
use crate::{
    pb, pbold,
    scalar::{
        bitmap::BitmapIndexPlugin, bloomfilter::BloomFilterIndexPlugin, btree::BTreeIndexPlugin,
        fmindex::FMIndexPlugin, inverted::InvertedIndexPlugin, json::JsonIndexPlugin,
        label_list::LabelListIndexPlugin, ngram::NGramIndexPlugin, registry::ScalarIndexPlugin,
        zonemap::ZoneMapIndexPlugin,
    },
};

/// Scalar detail package emitted by Lance 0.36 before the messages moved back
/// to `lance.table` for forward compatibility.
const V036_SCALAR_DETAILS_PACKAGE: &str = "lance.index.pb";

/// Derive the scalar index plugin name from a details type URL.
///
/// Takes the last `.`-separated segment, lowercases it, and strips any trailing
/// `"indexdetails"` suffix so the result matches the plugin name used in
/// [`IndexPluginRegistry`]. For example, `/lance.index.pb.ZoneMapIndexDetails`
/// yields `"zonemap"`.
pub fn plugin_name_from_details_url(type_url: &str) -> String {
    let segment = type_url.split('.').next_back().unwrap_or(type_url);
    let lower = segment.to_lowercase();
    lower
        .strip_suffix("indexdetails")
        .map(|s| s.to_string())
        .unwrap_or(lower)
}

/// Derive a human-readable index type name from a details type URL.
///
/// The display name is the final `.`-separated segment of the type URL with any
/// trailing `IndexDetails` removed. For example, `/lance.index.pb.VectorIndexDetails`
/// yields `Vector`. Used as a best-effort fallback when no plugin is registered
/// for the type URL, so the index type is never reported as opaque "Unknown"
/// while valid index details exist.
pub fn display_type_from_url(type_url: &str) -> &str {
    let segment = type_url.rsplit('.').next().unwrap_or(type_url);
    segment
        .strip_suffix("IndexDetails")
        .filter(|stripped| !stripped.is_empty())
        .unwrap_or(segment)
}

/// A registry of index plugins
pub struct IndexPluginRegistry {
    plugins: HashMap<String, Box<dyn ScalarIndexPlugin>>,
    details_type_names: HashSet<String>,
}

impl IndexPluginRegistry {
    fn normalize_plugin_name(name: &str) -> String {
        name.to_lowercase()
    }

    fn get_plugin_name_from_details_name(&self, details_name: &str) -> String {
        plugin_name_from_details_url(details_name)
    }

    /// Adds a plugin to the registry, using the name of the details message to determine
    /// the plugin name.
    ///
    /// The plugin name will be the lowercased name of the details message with any trailing
    /// "indexdetails" removed.
    ///
    /// For example, if the details message is `BTreeIndexDetails`, the plugin name will be
    /// `btree`.
    pub fn add_plugin<
        DetailsType: prost::Message + prost::Name,
        PluginType: ScalarIndexPlugin + std::default::Default + 'static,
    >(
        &mut self,
    ) {
        let plugin_name = self.get_plugin_name_from_details_name(DetailsType::NAME);
        self.details_type_names
            .insert(DetailsType::full_name().to_ascii_lowercase());
        self.plugins
            .insert(plugin_name, Box::new(PluginType::default()));
    }

    fn add_details_type_alias<DetailsType: prost::Name>(&mut self, package: &str) {
        self.details_type_names
            .insert(format!("{}.{}", package, DetailsType::NAME).to_ascii_lowercase());
    }

    /// Create a registry with the default plugins
    pub fn with_default_plugins() -> Arc<Self> {
        let mut registry = Self {
            plugins: HashMap::new(),
            details_type_names: HashSet::new(),
        };
        registry.add_plugin::<pbold::BTreeIndexDetails, BTreeIndexPlugin>();
        registry.add_plugin::<pbold::BitmapIndexDetails, BitmapIndexPlugin>();
        registry.add_plugin::<pbold::LabelListIndexDetails, LabelListIndexPlugin>();
        registry.add_plugin::<pbold::NGramIndexDetails, NGramIndexPlugin>();
        registry.add_plugin::<pbold::ZoneMapIndexDetails, ZoneMapIndexPlugin>();
        registry.add_plugin::<pb::BloomFilterIndexDetails, BloomFilterIndexPlugin>();
        registry.add_plugin::<pbold::InvertedIndexDetails, InvertedIndexPlugin>();
        registry.add_plugin::<pb::JsonIndexDetails, JsonIndexPlugin>();
        registry.add_plugin::<pb::FmIndexDetails, FMIndexPlugin>();
        #[cfg(feature = "geo")]
        registry.add_plugin::<pb::RTreeIndexDetails, RTreeIndexPlugin>();

        // Lance 0.36 released these scalar detail messages in the index package.
        // Register only those historical identities, not arbitrary packages
        // carrying the same terminal message names.
        registry.add_details_type_alias::<pbold::BTreeIndexDetails>(V036_SCALAR_DETAILS_PACKAGE);
        registry.add_details_type_alias::<pbold::BitmapIndexDetails>(V036_SCALAR_DETAILS_PACKAGE);
        registry
            .add_details_type_alias::<pbold::LabelListIndexDetails>(V036_SCALAR_DETAILS_PACKAGE);
        registry.add_details_type_alias::<pbold::NGramIndexDetails>(V036_SCALAR_DETAILS_PACKAGE);
        registry.add_details_type_alias::<pbold::ZoneMapIndexDetails>(V036_SCALAR_DETAILS_PACKAGE);
        registry.add_details_type_alias::<pbold::InvertedIndexDetails>(V036_SCALAR_DETAILS_PACKAGE);

        let registry = Arc::new(registry);
        for plugin in registry.plugins.values() {
            plugin.attach_registry(registry.clone());
        }

        registry
    }

    /// Returns whether the complete protobuf type name in `details` belongs to
    /// a registered scalar index reader.
    ///
    /// Type URL authorities may vary, so matching uses the fully qualified
    /// message name after the final slash. The table format requires index type
    /// URL comparisons to be case-insensitive.
    pub fn supports_details(&self, details: &prost_types::Any) -> bool {
        let Some((_, details_type_name)) = details.type_url.rsplit_once('/') else {
            return false;
        };
        if details_type_name.is_empty() || details_type_name.starts_with('.') {
            return false;
        }
        self.details_type_names
            .contains(&details_type_name.to_ascii_lowercase())
    }

    /// Get an index plugin suitable for training an index with the given parameters
    pub fn get_plugin_by_name(&self, name: &str) -> Result<&dyn ScalarIndexPlugin> {
        let plugin_name = Self::normalize_plugin_name(name);
        self.plugins
            .get(&plugin_name)
            .map(|plugin| plugin.as_ref())
            .ok_or_else(|| {
                let hint = if plugin_name == "rtree" {
                    ". The 'rtree' index requires the `geo` feature. \
                     Rebuild with `--features geo` to enable geospatial support"
                } else {
                    ""
                };
                Error::invalid_input_source(
                    format!("No scalar index plugin found for name '{name}'{hint}").into(),
                )
            })
    }

    pub fn get_plugin_by_details(
        &self,
        details: &prost_types::Any,
    ) -> Result<&dyn ScalarIndexPlugin> {
        let details_name = details.type_url.split('.').next_back().unwrap();
        let plugin_name = self.get_plugin_name_from_details_name(details_name);
        self.get_plugin_by_name(&plugin_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_type_from_url() {
        assert_eq!(
            display_type_from_url("/lance.index.pb.VectorIndexDetails"),
            "Vector"
        );
        assert_eq!(display_type_from_url("BTreeIndexDetails"), "BTree");
        // Segment without the IndexDetails suffix is returned verbatim.
        assert_eq!(
            display_type_from_url("/lance.pb.SomethingElse"),
            "SomethingElse"
        );
        // A bare "IndexDetails" segment has nothing left after stripping, so it
        // is returned as-is rather than an empty string.
        assert_eq!(display_type_from_url("IndexDetails"), "IndexDetails");
        assert_eq!(display_type_from_url(""), "");
    }

    #[test]
    fn test_get_plugin_by_name_accepts_case_insensitive_builtin_names() {
        let registry = IndexPluginRegistry::with_default_plugins();

        for (requested_name, expected_name) in [
            ("BTREE", "BTree"),
            ("Bitmap", "Bitmap"),
            ("INVERTED", "Inverted"),
            ("NGRAM", "NGram"),
            ("ZONEMAP", "ZoneMap"),
            ("BLOOMFILTER", "BloomFilter"),
            ("FM", "Fm"),
            ("JSON", "Json"),
        ] {
            let plugin = registry.get_plugin_by_name(requested_name).unwrap();
            assert_eq!(plugin.name(), expected_name);
        }
    }

    #[test]
    fn test_supports_details_matches_complete_type_name_case_insensitively() {
        let registry = IndexPluginRegistry::with_default_plugins();

        for type_url in [
            "/lance.table.BTreeIndexDetails",
            "type.googleapis.com/LANCE.TABLE.BTREEINDEXDETAILS",
            "/lance.index.pb.BTreeIndexDetails",
            "/lance.index.pb.BitmapIndexDetails",
            "/lance.index.pb.LabelListIndexDetails",
            "/lance.index.pb.NGramIndexDetails",
            "/lance.index.pb.ZoneMapIndexDetails",
            "/lance.index.pb.InvertedIndexDetails",
        ] {
            assert!(registry.supports_details(&prost_types::Any {
                type_url: type_url.to_string(),
                value: Vec::new(),
            }));
        }

        for type_url in [
            "type.googleapis.com/example.BTreeIndexDetails",
            "BTreeIndexDetails",
            "/.lance.table.BTreeIndexDetails",
        ] {
            assert!(!registry.supports_details(&prost_types::Any {
                type_url: type_url.to_string(),
                value: Vec::new(),
            }));
        }
    }
}
