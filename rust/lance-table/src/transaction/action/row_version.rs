// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Row-version-metadata refresh action and its apply logic.
//!
//! Extracted from `action.rs` to keep that module focused on the `Action`
//! trait and dispatch. Items here are re-exported from `action.rs`, so the
//! public surface of `crate::transaction::action::*` is unchanged.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::format::{Fragment, IndexMetadata, Manifest};
use deepsize::DeepSizeOf;
use lance_core::Result;

use super::Action;
use crate::impl_dyn_action;
use crate::transaction::row_ids::assign_row_ids;
use crate::transaction::row_version::resolve_update_version_metadata;

/// Payload for [`RefreshRowVersionMetadata`](super::RefreshRowVersionMetadata).
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct RefreshRowVersionMetadata {
    /// Which of the three legacy refresh paths to reproduce.
    pub mode: RowVersionRefresh,
}

impl Action for RefreshRowVersionMetadata {
    impl_dyn_action!(RefreshRowVersionMetadata);

    fn apply(&self, manifest: &mut Manifest, _indices: &mut Vec<IndexMetadata>) -> Result<()> {
        refresh_row_version_metadata(manifest, &self.mode)
    }
}

/// The per-row version-metadata refresh [`RefreshRowVersionMetadata`](super::RefreshRowVersionMetadata)
/// performs — one variant per legacy `build_manifest` refresh path (#6914).
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub enum RowVersionRefresh {
    /// A stable-row-id [`Operation::Update`](super::super::Operation::Update)'s newly
    /// written fragments — reproduces `resolve_update_version_metadata`.
    ///
    /// `new_fragment_ids` names the freshly written fragments by their committed
    /// ids (allocated the same way [`AddFragments`](super::AddFragments) allocates
    /// them). Every other manifest fragment is a carried-over source: `apply`
    /// assigns the new fragments' row ids, then sets their
    /// `created_at_version_meta` by tracing each row back to the source fragment
    /// it originated in and `last_updated_at_version_meta` to the new version.
    ///
    /// Identifying the new fragments explicitly (rather than by absence of
    /// `created_at`) keeps a source fragment that predates `created_at` tracking
    /// in the source pool, matching the legacy path which passes every existing
    /// fragment as the source.
    UpdatedRows {
        /// Committed ids of the update's freshly written fragments.
        new_fragment_ids: Vec<u64>,
    },

    /// A stable-row-id `Update` in `RewriteColumns` mode that rewrote only some
    /// rows of a fragment — reproduces
    /// `refresh_row_latest_update_meta_for_partial_frag_rewrite_cols`.
    ///
    /// Each `(fragment_id, offsets)` pair bumps `last_updated_at_version_meta`
    /// to the new version at exactly those local row offsets, preserving the
    /// rest. A fragment id absent from the manifest, or one with no existing
    /// `last_updated_at_version_meta`, is skipped.
    RewrittenColumns {
        /// `fragment_id -> touched local row offsets`.
        touched_offsets: Vec<(u64, Vec<u64>)>,
    },

    /// A stable-row-id [`Operation::Merge`](super::super::Operation::Merge) that
    /// materialized new column data into fragments — reproduces
    /// `refresh_row_latest_update_meta_for_full_frag_rewrite_cols`.
    ///
    /// Every row of each listed fragment gets `last_updated_at_version_meta`
    /// set uniformly to the new version. A fragment id absent from the manifest
    /// is skipped.
    MergedColumns {
        /// Fragment ids that gained data files in the merge.
        fragment_ids: Vec<u64>,
    },
}

/// Apply a [`RowVersionRefresh`] to `manifest` — the body of
/// [`RefreshRowVersionMetadata`](super::RefreshRowVersionMetadata)'s `apply`.
///
/// The new dataset version is `manifest.version + 1`, the same trick
/// [`UpdateMergedGenerations`](super::UpdateMergedGenerations) uses; everything else the refresh needs
/// is read from the fragments already in `manifest`.
pub(super) fn refresh_row_version_metadata(
    manifest: &mut Manifest,
    mode: &RowVersionRefresh,
) -> Result<()> {
    use crate::rowids::version::{
        refresh_row_latest_update_meta_for_full_frag_rewrite_cols,
        refresh_row_latest_update_meta_for_partial_frag_rewrite_cols,
    };

    let new_version = manifest.version + 1;
    let mut fragments = (*manifest.fragments).clone();
    match mode {
        RowVersionRefresh::UpdatedRows { new_fragment_ids } => {
            // The freshly written fragments are named explicitly; every other
            // fragment in the manifest is a carried-over source whose inline row
            // ids still list any moved rows at their original offsets. A
            // `RewriteRows` new fragment also keeps `row_id_meta`, so row ids
            // cannot tell the two apart — but the explicit id list can, and
            // unlike the absence of `created_at` it does not misclassify a
            // source fragment that predates `created_at` tracking.
            let fresh_ids: HashSet<u64> = new_fragment_ids.iter().copied().collect();
            let mut fresh: Vec<Fragment> = Vec::new();
            let mut source: Vec<Fragment> = Vec::new();
            for fragment in &fragments {
                if fresh_ids.contains(&fragment.id) {
                    fresh.push(fragment.clone());
                } else {
                    source.push(fragment.clone());
                }
            }
            if fresh.is_empty() {
                return Ok(());
            }
            // Assign row ids before the trace: `resolve_update_version_metadata`
            // maps each new row's stable id back to its source fragment. This
            // is the legacy `assign_row_ids` + `resolve_update_version_metadata`
            // pairing; doing it here means `build_manifest_via_actions`'s
            // post-processing skips these fragments (they now carry row ids).
            let mut next_row_id = manifest.next_row_id;
            assign_row_ids(&mut next_row_id, &mut fresh)?;
            resolve_update_version_metadata(&source, &mut fresh, new_version)?;
            manifest.next_row_id = next_row_id;

            let refreshed: HashMap<u64, Fragment> = fresh.into_iter().map(|f| (f.id, f)).collect();
            for fragment in fragments.iter_mut() {
                if let Some(updated) = refreshed.get(&fragment.id) {
                    *fragment = updated.clone();
                }
            }
        }
        RowVersionRefresh::RewrittenColumns { touched_offsets } => {
            let prev_version = manifest.version;
            for (fragment_id, offsets) in touched_offsets {
                let Some(fragment) = fragments.iter_mut().find(|f| f.id == *fragment_id) else {
                    continue;
                };
                // Mirror the legacy partial path: an empty offset set is a
                // no-op, and a fragment with no existing `last_updated` meta is
                // skipped so its unmatched rows are not stamped `prev_version`.
                if offsets.is_empty() || fragment.last_updated_at_version_meta.is_none() {
                    continue;
                }
                let offsets: Vec<usize> = offsets.iter().map(|&o| o as usize).collect();
                refresh_row_latest_update_meta_for_partial_frag_rewrite_cols(
                    fragment,
                    &offsets,
                    new_version,
                    prev_version,
                )?;
            }
        }
        RowVersionRefresh::MergedColumns { fragment_ids } => {
            for fragment_id in fragment_ids {
                let Some(fragment) = fragments.iter_mut().find(|f| f.id == *fragment_id) else {
                    continue;
                };
                refresh_row_latest_update_meta_for_full_frag_rewrite_cols(fragment, new_version)?;
            }
        }
    }
    manifest.fragments = Arc::new(fragments);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::format::Fragment;

    use super::super::test_support::*;
    use super::super::*;
    use super::*;

    fn refresh(mode: RowVersionRefresh) -> RefreshRowVersionMetadata {
        RefreshRowVersionMetadata { mode }
    }

    /// Assert every one of `rows` positions in `meta` carries `expected`.
    fn assert_all_version(
        meta: &Option<crate::format::RowDatasetVersionMeta>,
        rows: usize,
        expected: u64,
    ) {
        let seq = meta
            .as_ref()
            .expect("version metadata present")
            .load_sequence()
            .unwrap();
        for offset in 0..rows {
            assert_eq!(seq.version_at(offset), Some(expected), "row {offset}");
        }
    }

    #[test]
    fn merged_columns_refreshes_listed_fragments_uniformly() {
        let mut manifest = manifest_at_version(
            5,
            vec![
                rowid_fragment(0, &[0, 1, 2], Some(3)),
                rowid_fragment(1, &[3, 4, 5], Some(3)),
            ],
        );
        refresh(RowVersionRefresh::MergedColumns {
            fragment_ids: vec![0],
        })
        .apply(&mut manifest, &mut vec![])
        .unwrap();

        // Fragment 0's last_updated jumps to the new version; created_at and
        // the untargeted fragment 1 are left alone.
        assert_all_version(&manifest.fragments[0].last_updated_at_version_meta, 3, 6);
        assert_all_version(&manifest.fragments[0].created_at_version_meta, 3, 3);
        assert_all_version(&manifest.fragments[1].last_updated_at_version_meta, 3, 3);
    }

    #[test]
    fn merged_columns_skips_missing_fragment_id() {
        let mut manifest = manifest_at_version(5, vec![rowid_fragment(0, &[0, 1, 2], Some(3))]);
        let before = manifest.fragments.clone();
        refresh(RowVersionRefresh::MergedColumns {
            fragment_ids: vec![99],
        })
        .apply(&mut manifest, &mut vec![])
        .unwrap();
        assert_eq!(manifest.fragments, before);
    }

    #[test]
    fn rewritten_columns_bumps_only_touched_offsets() {
        let mut manifest = manifest_at_version(5, vec![rowid_fragment(0, &[0, 1, 2, 3], Some(3))]);
        refresh(RowVersionRefresh::RewrittenColumns {
            touched_offsets: vec![(0, vec![1, 3])],
        })
        .apply(&mut manifest, &mut vec![])
        .unwrap();

        let seq = manifest.fragments[0]
            .last_updated_at_version_meta
            .as_ref()
            .unwrap()
            .load_sequence()
            .unwrap();
        assert_eq!(seq.version_at(0), Some(3));
        assert_eq!(seq.version_at(1), Some(6));
        assert_eq!(seq.version_at(2), Some(3));
        assert_eq!(seq.version_at(3), Some(6));
    }

    #[test]
    fn rewritten_columns_skips_fragment_without_version_meta() {
        // The legacy partial path skips a fragment with no existing
        // `last_updated` meta rather than fabricating one.
        let mut manifest = manifest_at_version(5, vec![rowid_fragment(0, &[0, 1, 2, 3], None)]);
        refresh(RowVersionRefresh::RewrittenColumns {
            touched_offsets: vec![(0, vec![1])],
        })
        .apply(&mut manifest, &mut vec![])
        .unwrap();
        assert!(manifest.fragments[0].last_updated_at_version_meta.is_none());
    }

    #[test]
    fn updated_rows_traces_created_at_from_source_fragments() {
        // A `RewriteRows` update: the source fragment 0 holds rows created at
        // version 2; fragment 1 is freshly written, has no version metadata,
        // and keeps the preserved stable ids of the two rewritten rows.
        let source = rowid_fragment(0, &[10, 11, 12], Some(2));
        let mut fresh = Fragment::new(1);
        fresh.physical_rows = Some(2);
        fresh.row_id_meta = Some(inline_row_ids(&[10, 12]));
        let mut manifest = manifest_at_version(5, vec![source, fresh]);

        refresh(RowVersionRefresh::UpdatedRows {
            new_fragment_ids: vec![1],
        })
        .apply(&mut manifest, &mut vec![])
        .unwrap();

        let refreshed = manifest.fragments.iter().find(|f| f.id == 1).unwrap();
        // created_at is traced back to the rows' original creation version.
        assert_all_version(&refreshed.created_at_version_meta, 2, 2);
        // last_updated is stamped at the new version.
        assert_all_version(&refreshed.last_updated_at_version_meta, 2, 6);
        // The source fragment is untouched.
        let source = manifest.fragments.iter().find(|f| f.id == 0).unwrap();
        assert_all_version(&source.created_at_version_meta, 3, 2);
        assert_all_version(&source.last_updated_at_version_meta, 3, 2);
    }

    #[test]
    fn updated_rows_assigns_row_ids_to_inserted_fragments() {
        // An inserted-rows fragment has neither version metadata nor row ids;
        // `apply` assigns row ids from `next_row_id` and stamps last_updated.
        let source = rowid_fragment(0, &[0, 1], Some(2));
        let mut inserted = Fragment::new(1);
        inserted.physical_rows = Some(2);
        let mut manifest = manifest_at_version(5, vec![source, inserted]);
        manifest.next_row_id = 100;

        refresh(RowVersionRefresh::UpdatedRows {
            new_fragment_ids: vec![1],
        })
        .apply(&mut manifest, &mut vec![])
        .unwrap();

        let refreshed = manifest.fragments.iter().find(|f| f.id == 1).unwrap();
        assert!(refreshed.row_id_meta.is_some());
        assert_eq!(manifest.next_row_id, 102);
        assert_all_version(&refreshed.last_updated_at_version_meta, 2, 6);
    }

    #[test]
    fn refresh_row_version_metadata_action_is_well_formed() {
        let action = refresh(RowVersionRefresh::UpdatedRows {
            new_fragment_ids: vec![],
        });
        // Stateless payload — validates against any manifest.
        assert!(action.validate(&empty_manifest()).is_ok());
    }
}
