// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Config-domain action payloads.
//!
//! Extracted from `action.rs` to keep that module focused on the `Action`
//! trait and dispatch. Items here are re-exported from `action.rs`, so the
//! public surface of `crate::transaction::action::*` is unchanged.

use crate::format::{BasePath, IndexMetadata, Manifest};
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};

use super::Action;
use crate::impl_dyn_action;
use crate::transaction::update_map::{UpdateMap, apply_update_map};

/// Payload for [`UpdateConfig`](super::UpdateConfig).
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct UpdateConfig {
    /// Changes to apply to the table config (`lance.*` keys). Honors the
    /// `replace` flag and key deletion.
    pub config_updates: Option<UpdateMap>,
    /// Changes to apply to the arbitrary table-metadata map. Honors the
    /// `replace` flag and key deletion.
    pub table_metadata_updates: Option<UpdateMap>,
}

impl Action for UpdateConfig {
    impl_dyn_action!(UpdateConfig);

    fn apply(&self, manifest: &mut Manifest, _indices: &mut Vec<IndexMetadata>) -> Result<()> {
        if let Some(config_updates) = &self.config_updates {
            apply_update_map(&mut manifest.config, config_updates);
        }
        if let Some(table_metadata_updates) = &self.table_metadata_updates {
            apply_update_map(&mut manifest.table_metadata, table_metadata_updates);
        }
        Ok(())
    }
}

/// Payload for [`AddBases`](super::AddBases).
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddBases {
    /// Base paths to append. A base whose `id` is `0` is treated as
    /// unassigned and allocated at apply time.
    pub bases: Vec<BasePath>,
}

impl Action for AddBases {
    impl_dyn_action!(AddBases);

    fn apply(&self, manifest: &mut Manifest, _indices: &mut Vec<IndexMetadata>) -> Result<()> {
        for base in &self.bases {
            if let Some(existing) = manifest
                .base_paths
                .values()
                .find(|bp| bp.name == base.name || bp.path == base.path)
            {
                return Err(Error::invalid_input(format!(
                    "AddBases conflicts with an existing base path: new name={:?} path={} \
                     collides with existing name={:?} path={}",
                    base.name, base.path, existing.name, existing.path
                )));
            }
            // An unassigned id (0) is allocated from the high-water mark,
            // mirroring the legacy `Operation::UpdateBases` path.
            let mut base = base.clone();
            if base.id == 0 {
                base.id = manifest.base_paths.keys().max().map_or(1, |&id| id + 1);
            }
            manifest.base_paths.insert(base.id, base);
        }
        Ok(())
    }
}

/// Payload for [`ReserveFragmentIds`](super::ReserveFragmentIds).
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct ReserveFragmentIds {
    /// How many fragment ids to reserve — `max_fragment_id` advances by this.
    pub num_fragments: u32,
}

impl Action for ReserveFragmentIds {
    impl_dyn_action!(ReserveFragmentIds);

    fn apply(&self, manifest: &mut Manifest, _indices: &mut Vec<IndexMetadata>) -> Result<()> {
        // Resolve the high-water mark via `max_fragment_id()` so a manifest
        // whose `max_fragment_id` field is not yet populated still reserves
        // above its existing fragments.
        let current = manifest.max_fragment_id().unwrap_or(0) as u32;
        manifest.max_fragment_id = Some(current + self.num_fragments);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::test_support::*;
    use super::super::*;
    use super::*;

    fn update_config(config: Option<UpdateMap>, table_metadata: Option<UpdateMap>) -> UpdateConfig {
        UpdateConfig {
            config_updates: config,
            table_metadata_updates: table_metadata,
        }
    }

    #[test]
    fn update_config_apply_merges_config_and_table_metadata() {
        let mut manifest = empty_manifest();
        manifest.config.insert("keep".into(), "1".into());
        manifest.table_metadata.insert("old".into(), "v".into());

        let action = update_config(
            Some(umap(&[("lance.x", Some("on")), ("keep", None)], false)),
            Some(umap(&[("owner", Some("ml"))], false)),
        );
        action.apply(&mut manifest, &mut vec![]).unwrap();

        assert_eq!(
            manifest.config.get("lance.x").map(String::as_str),
            Some("on")
        );
        assert!(!manifest.config.contains_key("keep"));
        assert_eq!(
            manifest.table_metadata.get("owner").map(String::as_str),
            Some("ml")
        );
        assert_eq!(
            manifest.table_metadata.get("old").map(String::as_str),
            Some("v")
        );
    }

    #[test]
    fn update_config_apply_replace_clears_existing() {
        let mut manifest = empty_manifest();
        manifest.config.insert("stale".into(), "1".into());

        let action = update_config(Some(umap(&[("fresh", Some("2"))], true)), None);
        action.apply(&mut manifest, &mut vec![]).unwrap();

        assert!(!manifest.config.contains_key("stale"));
        assert_eq!(manifest.config.get("fresh").map(String::as_str), Some("2"));
    }

    #[test]
    fn add_bases_apply_assigns_unset_ids() {
        let mut manifest = empty_manifest();
        manifest
            .base_paths
            .insert(1, base_path(1, Some("root"), "/root"));

        let action = AddBases {
            bases: vec![base_path(0, Some("a"), "/a"), base_path(0, Some("b"), "/b")],
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        let mut ids: Vec<u32> = manifest.base_paths.keys().copied().collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn add_bases_apply_keeps_explicit_ids() {
        let mut manifest = empty_manifest();
        let action = AddBases {
            bases: vec![base_path(7, Some("a"), "/a")],
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();
        assert!(manifest.base_paths.contains_key(&7));
    }

    #[test]
    fn add_bases_apply_rejects_duplicate_path() {
        let mut manifest = empty_manifest();
        manifest
            .base_paths
            .insert(1, base_path(1, Some("root"), "/root"));

        let action = AddBases {
            bases: vec![base_path(0, Some("other"), "/root")],
        };
        let err = action.apply(&mut manifest, &mut vec![]).unwrap_err();
        assert!(format!("{err}").contains("/root"), "got: {err}");
    }

    #[test]
    fn reserve_fragment_ids_apply_advances_max_fragment_id() {
        // Empty manifest: reservation starts from zero.
        let mut manifest = empty_manifest();
        ReserveFragmentIds { num_fragments: 4 }
            .apply(&mut manifest, &mut vec![])
            .unwrap();
        assert_eq!(manifest.max_fragment_id, Some(4));

        // With existing fragments the reservation is above the high-water mark.
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![sample_fragment(9)]);
        manifest.max_fragment_id = Some(9);
        ReserveFragmentIds { num_fragments: 3 }
            .apply(&mut manifest, &mut vec![])
            .unwrap();
        assert_eq!(manifest.max_fragment_id, Some(12));
    }
}
