// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fragment-domain action payloads.
//!
//! Extracted from `action.rs` to keep that module focused on the `Action`
//! trait and dispatch. Items here are re-exported from `action.rs`, so the
//! public surface of `crate::transaction::action::*` is unchanged.

use std::collections::HashSet;
use std::sync::Arc;

use crate::format::{DataFile, DeletionFile, Fragment, IndexMetadata, Manifest};
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;

use super::{Action, invalidate_index_coverage};
use crate::impl_dyn_action;
use crate::transaction::inserted_rows::KeyExistenceFilter;

/// Payload for [`AddFragments`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddFragments {
    /// Fragments to add. Their `id` fields are unassigned (treated as
    /// placeholders) and will be filled in at apply time.
    pub fragments: Vec<Fragment>,
    /// Bloom filter over the primary keys inserted by these fragments, when the
    /// fragments come from a merge-insert. Used only for conflict detection: a
    /// concurrent insert whose keys intersect this filter conflicts (spike
    /// #6448 Q1, design §6). `apply` ignores it. `None` for a plain append or
    /// any writer that did not record inserted keys.
    pub inserted_rows_filter: Option<KeyExistenceFilter>,
}

impl Action for AddFragments {
    impl_dyn_action!(AddFragments);

    fn apply(&self, manifest: &mut Manifest, _indices: &mut Vec<IndexMetadata>) -> Result<()> {
        let mut next_id = manifest.max_fragment_id().map_or(0, |id| id + 1);
        let mut new_fragments = (*manifest.fragments).clone();
        new_fragments.reserve(self.fragments.len());
        for fragment in &self.fragments {
            let mut fragment = fragment.clone();
            // Mirror `Transaction::fragments_with_ids`: id 0 means
            // "unassigned", so allocate from the high-water mark.
            if fragment.id == 0 {
                fragment.id = next_id;
                next_id += 1;
            }
            new_fragments.push(fragment);
        }
        manifest.fragments = Arc::new(new_fragments);
        manifest.update_max_fragment_id();
        Ok(())
    }
}

/// Which fragments an [`RemoveFragments`] targets.
///
/// `Delete` removes a fixed set of emptied fragments and translates to
/// [`FragmentSelector::Ids`]. `Overwrite` wipes the dataset and translates to
/// [`FragmentSelector::AllCurrent`]: the criterion is resolved at apply time —
/// after rebase — so an overwrite also removes a fragment a concurrent
/// `AddFragments` committed in the meantime (spike #6448 Q3, design §9).
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub enum FragmentSelector {
    /// Exactly the fragment ids in this list.
    Ids(Vec<u64>),
    /// Every fragment present in the manifest at apply time.
    AllCurrent,
}

impl FragmentSelector {
    /// The concrete set of fragment ids removed when applied to `manifest`.
    pub fn resolve(&self, manifest: &Manifest) -> Vec<u64> {
        match self {
            Self::Ids(ids) => ids.clone(),
            Self::AllCurrent => manifest.fragments.iter().map(|f| f.id).collect(),
        }
    }

    /// Whether this selector removes `fragment_id`.
    ///
    /// `AllCurrent` removes every fragment that exists at apply time, so for
    /// conflict detection — which screens action pairs without a manifest — it
    /// conservatively targets any id.
    pub fn targets(&self, fragment_id: u64) -> bool {
        match self {
            Self::Ids(ids) => ids.contains(&fragment_id),
            Self::AllCurrent => true,
        }
    }
}

/// Payload for [`RemoveFragments`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct RemoveFragments {
    /// Which fragments to remove from the manifest.
    pub selector: FragmentSelector,
}

impl Action for RemoveFragments {
    impl_dyn_action!(RemoveFragments);

    fn apply(&self, manifest: &mut Manifest, indices: &mut Vec<IndexMetadata>) -> Result<()> {
        // `AllCurrent` is resolved here, against the post-rebase manifest, so
        // an `Overwrite` also wipes a fragment a concurrent `AddFragments`
        // committed (spike #6448 §9).
        let drop: HashSet<u64> = self.selector.resolve(manifest).into_iter().collect();
        let new_fragments: Vec<Fragment> = manifest
            .fragments
            .iter()
            .filter(|f| !drop.contains(&f.id))
            .cloned()
            .collect();
        manifest.fragments = Arc::new(new_fragments);
        match &self.selector {
            // An `AllCurrent` removal wipes the dataset — it only ever comes
            // from an `Overwrite`. Fragment ids restart from 0, matching the
            // legacy `Overwrite` arm, so the high-water mark resets too. The
            // index list is wiped too: an Overwrite's freshly written
            // fragments may reuse the old fragment ids (the counter is reset),
            // so without an explicit drop, downstream pruning would mistake
            // the old indices' fragment bitmaps for current coverage and
            // retain them — leaving cleanup unable to reclaim the index files
            // (and matching the legacy `Operation::Overwrite` arm, which
            // unconditionally clears `final_indices`).
            FragmentSelector::AllCurrent => {
                manifest.max_fragment_id = None;
                indices.clear();
            }
            // An `Ids` removal keeps `max_fragment_id` monotonic — it does
            // NOT shrink after removal so future fragment ids stay unique
            // across history.
            FragmentSelector::Ids(_) => {}
        }
        Ok(())
    }
}

/// Payload for [`UpdateDeletionVector`].
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateDeletionVector {
    /// ID of the existing fragment whose deletion vector is replaced. Must
    /// reference a fragment present in the manifest at apply time.
    pub fragment_id: u64,
    /// The replacement deletion file.
    pub new_deletion_file: DeletionFile,
    /// Rows this update marks deleted, when known. Used only for conflict
    /// detection: two updates to the same fragment commute when both
    /// enumerate disjoint row sets. Translation from [`Operation::Delete`]
    /// leaves this `None` — the old format does not record affected rows.
    pub affected_rows: Option<RoaringBitmap>,
}

impl DeepSizeOf for UpdateDeletionVector {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        // `RoaringBitmap` has no `DeepSizeOf` impl; approximate it by its
        // cardinality, mirroring `UpdatedFragmentOffsets` in `transaction.rs`.
        self.new_deletion_file.deep_size_of_children(context)
            + self.affected_rows.as_ref().map_or(0, |rows| {
                (rows.len() as usize).saturating_mul(std::mem::size_of::<u32>())
            })
    }
}

impl Action for UpdateDeletionVector {
    impl_dyn_action!(UpdateDeletionVector);

    fn apply(&self, manifest: &mut Manifest, _indices: &mut Vec<IndexMetadata>) -> Result<()> {
        let mut new_fragments = (*manifest.fragments).clone();
        let slot = new_fragments
            .iter_mut()
            .find(|f| f.id == self.fragment_id)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "UpdateDeletionVector targets fragment id {} which is not present in the manifest",
                    self.fragment_id
                ))
            })?;
        slot.deletion_file = Some(self.new_deletion_file.clone());
        manifest.fragments = Arc::new(new_fragments);
        Ok(())
    }
}

/// How a [`ReplaceFragmentColumns`]'s `new_data_files` fold into the target
/// fragment's current data files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnReplacement {
    /// Each new file either swaps an existing data file with the same field
    /// set and file format version in place, or — for a field no data file
    /// covers yet — is appended. This is the legacy
    /// [`Operation::DataReplacement`](super::Operation::DataReplacement) shape
    /// and the non-tombstoning `RewriteColumns` shape, where each rewritten
    /// field already lives in its own data file.
    InPlace,
    /// A rewritten field was tombstoned (rewritten to the `-2` sentinel) out of
    /// a data file it shared with an untouched field, so the rewrite is not a
    /// positional swap. `new_data_files` holds the fragment's complete
    /// post-rewrite file list and `apply` installs it verbatim. This is the
    /// shape a partial `RewriteColumns` update produces.
    Tombstone,
}

/// Payload for [`ReplaceFragmentColumns`].
#[derive(Debug, Clone, PartialEq)]
pub struct ReplaceFragmentColumns {
    /// ID of the existing fragment whose column data is replaced. Must
    /// reference a fragment present in the manifest at apply time.
    pub fragment_id: u64,
    /// The field ids whose backing data is replaced. Used as the conflict key
    /// (a concurrent replacement of an overlapping field set on the same
    /// fragment conflicts) and to invalidate covering index entries.
    pub field_ids: Vec<i32>,
    /// The replacement data files. For [`ColumnReplacement::InPlace`] each must
    /// align with the target fragment's current layout (see
    /// [`ReplaceFragmentColumns::validate`]); for [`ColumnReplacement::Tombstone`]
    /// this is the fragment's complete post-rewrite file list.
    pub new_data_files: Vec<DataFile>,
    /// Row offsets this replacement rewrites, when known. Reserved for
    /// row-level conflict resolution; translation from
    /// [`Operation::DataReplacement`](super::Operation::DataReplacement)
    /// leaves this `None` — the old format replaces a whole column and does
    /// not record row offsets.
    pub updated_row_offsets: Option<RoaringBitmap>,
    /// How `new_data_files` fold into the fragment's current layout.
    pub replacement: ColumnReplacement,
}

impl Action for ReplaceFragmentColumns {
    impl_dyn_action!(ReplaceFragmentColumns);

    fn apply(&self, manifest: &mut Manifest, indices: &mut Vec<IndexMetadata>) -> Result<()> {
        self.validate(manifest)?;
        let mut new_fragments = (*manifest.fragments).clone();
        let fragment = new_fragments
            .iter_mut()
            .find(|f| f.id == self.fragment_id)
            .expect("ReplaceFragmentColumns::validate confirmed the fragment exists");
        match self.replacement {
            ColumnReplacement::InPlace => {
                for new_file in &self.new_data_files {
                    let covered: HashSet<i32> = fragment
                        .files
                        .iter()
                        .flat_map(|f| f.fields.iter().copied())
                        .collect();
                    let mut swapped = false;
                    for file in &mut fragment.files {
                        if file.fields == new_file.fields
                            && file.file_major_version == new_file.file_major_version
                            && file.file_minor_version == new_file.file_minor_version
                        {
                            file.path = new_file.path.clone();
                            file.file_size_bytes = new_file.file_size_bytes.clone();
                            swapped = true;
                        }
                    }
                    // An all-NULL column being filled with real data has no
                    // existing file to swap; append instead.
                    if !swapped && new_file.fields.iter().all(|id| !covered.contains(id)) {
                        fragment.files.push(new_file.clone());
                    }
                }
            }
            ColumnReplacement::Tombstone => {
                // A tombstone replacement carries the fragment's complete
                // post-rewrite file list — the rewritten fields tombstoned out
                // of the files that shared them plus the fresh files
                // re-supplying them — so install it directly. Re-deriving the
                // layout is not possible: the `RewriteColumns` producers
                // disagree on whether a fully-tombstoned file is dropped or
                // kept.
                fragment.files = self.new_data_files.clone();
            }
        }
        manifest.fragments = Arc::new(new_fragments);
        // Replacing a covered field's data invalidates the index entries built
        // over it; drop the modified fragment from their coverage bitmaps, as
        // the legacy `DataReplacement` path does.
        invalidate_index_coverage(indices, &[self.fragment_id], &self.field_ids);
        Ok(())
    }

    /// Enforces the same-layout rule the legacy `Operation::DataReplacement`
    /// path checks: the target fragment must exist, and every replacement data
    /// file must either
    ///
    /// * match an existing data file on the fragment by field-id set and file
    ///   format version — an in-place column swap; or
    /// * carry fields disjoint from every field the fragment's data files
    ///   already cover — the all-NULL-column → real-data case, where the new
    ///   file is appended rather than swapped.
    ///
    /// A file that partially overlaps the fragment's layout matches no
    /// existing file and would leave the fragment unchanged; that is rejected
    /// with the same error the legacy path raises. Applied to every
    /// `ReplaceFragmentColumns` translated from one `DataReplacement`, this
    /// per-fragment check collectively enforces that the targeted fragments
    /// have isomorphic layouts along the replaced columns.
    ///
    /// A [`ColumnReplacement::Tombstone`] replacement carries the fragment's
    /// whole post-rewrite file list, so it only requires that list to be
    /// non-empty.
    fn validate(&self, manifest: &Manifest) -> Result<()> {
        let fragment = manifest
            .fragments
            .iter()
            .find(|f| f.id == self.fragment_id)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "ReplaceFragmentColumns targets fragment id {} which is not present in the manifest",
                    self.fragment_id
                ))
            })?;
        match self.replacement {
            ColumnReplacement::InPlace => {
                let covered: HashSet<i32> = fragment
                    .files
                    .iter()
                    .flat_map(|f| f.fields.iter().copied())
                    .collect();
                for new_file in &self.new_data_files {
                    let matches_existing = fragment.files.iter().any(|f| {
                        f.fields == new_file.fields
                            && f.file_major_version == new_file.file_major_version
                            && f.file_minor_version == new_file.file_minor_version
                    });
                    let is_all_null_add = new_file.fields.iter().all(|id| !covered.contains(id));
                    if !matches_existing && !is_all_null_add {
                        return Err(Error::invalid_input(format!(
                            "Expected to modify the fragment but no changes were made. The new \
                             data file for fragment {} does not align with any existing data \
                             file. Check that the new data file's field ids and file format \
                             version match an existing file.",
                            self.fragment_id
                        )));
                    }
                }
            }
            ColumnReplacement::Tombstone => {
                // `new_data_files` is the fragment's complete post-rewrite file
                // list; a fragment must keep at least one data file.
                if self.new_data_files.is_empty() {
                    return Err(Error::invalid_input(format!(
                        "ReplaceFragmentColumns tombstone replacement for fragment {} \
                         carries an empty data file list",
                        self.fragment_id
                    )));
                }
            }
        }
        Ok(())
    }
}

impl DeepSizeOf for ReplaceFragmentColumns {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        // `RoaringBitmap` has no `DeepSizeOf` impl; approximate it by its
        // cardinality, mirroring `UpdateDeletionVector`.
        self.field_ids.deep_size_of_children(context)
            + self.new_data_files.deep_size_of_children(context)
            + self.updated_row_offsets.as_ref().map_or(0, |rows| {
                (rows.len() as usize).saturating_mul(std::mem::size_of::<u32>())
            })
    }
}

/// Payload for [`RewriteFragments`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct RewriteFragments {
    /// Ids of the existing fragments this rewrite replaces. They must all be
    /// present in the manifest at apply time and, for the contiguous-splice
    /// fast path, are expected to appear in this order in the fragment list.
    pub old_fragment_ids: Vec<u64>,
    /// The data-equivalent replacement fragments. An `id` of `0` is treated as
    /// a placeholder and allocated from the high-water mark at apply time.
    pub new_fragments: Vec<Fragment>,
    /// Whether the rewrite preserved stable row ids. The `Rewrite`
    /// decomposition pairs a `preserves_row_ids` rewrite with a
    /// `RebindIndexCoverage` and a non-preserving one with a `RewriteIndex`
    /// per affected index.
    pub preserves_row_ids: bool,
}

impl Action for RewriteFragments {
    impl_dyn_action!(RewriteFragments);

    fn apply(&self, manifest: &mut Manifest, _indices: &mut Vec<IndexMetadata>) -> Result<()> {
        self.validate(manifest)?;
        let mut next_id = manifest.max_fragment_id().map_or(0, |id| id + 1);
        let new_fragments: Vec<Fragment> = self
            .new_fragments
            .iter()
            .map(|fragment| {
                let mut fragment = fragment.clone();
                // Mirror `Transaction::fragments_with_ids`: id 0 means
                // "unassigned", so allocate from the high-water mark.
                if fragment.id == 0 {
                    fragment.id = next_id;
                    next_id += 1;
                }
                fragment
            })
            .collect();

        let mut fragments = (*manifest.fragments).clone();
        // Fast path: the old fragments occupy a contiguous run, in order, so
        // splice the new ones into that range. This mirrors the legacy
        // `Transaction::handle_rewrite_fragments`.
        let contiguous_range = fragments
            .iter()
            .position(|f| f.id == self.old_fragment_ids[0])
            .filter(|&start| {
                self.old_fragment_ids
                    .iter()
                    .enumerate()
                    .all(|(offset, id)| fragments.get(start + offset).map(|f| f.id) == Some(*id))
            })
            .map(|start| start..start + self.old_fragment_ids.len());
        if let Some(range) = contiguous_range {
            fragments.splice(range, new_fragments);
        } else {
            let drop: HashSet<u64> = self.old_fragment_ids.iter().copied().collect();
            fragments.retain(|f| !drop.contains(&f.id));
            fragments.extend(new_fragments);
        }
        manifest.fragments = Arc::new(fragments);
        manifest.update_max_fragment_id();
        Ok(())
    }

    /// A `RewriteFragments` must replace at least one existing fragment, and
    /// every `old_fragment_id` must be present in the manifest. The rewrite
    /// must also be *data equivalent* — it changes the physical layout of the
    /// rows, not the rows themselves — so an index covering the old fragments
    /// can have its coverage salvaged onto the new fragments. When every
    /// fragment involved reports its physical row count, this asserts the new
    /// fragments hold exactly as many live (non-deleted) rows as the old ones.
    fn validate(&self, manifest: &Manifest) -> Result<()> {
        if self.old_fragment_ids.is_empty() {
            return Err(Error::invalid_input(
                "RewriteFragments must replace at least one fragment".to_string(),
            ));
        }
        let mut old_fragments = Vec::with_capacity(self.old_fragment_ids.len());
        for id in &self.old_fragment_ids {
            let fragment = manifest
                .fragments
                .iter()
                .find(|f| f.id == *id)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "RewriteFragments targets fragment id {id} which is not present in the manifest"
                    ))
                })?;
            old_fragments.push(fragment);
        }

        // Data-equivalence check: only enforced when every involved fragment
        // reports its physical row count. Compaction output always carries it;
        // a translated rewrite from an older writer may not, and is skipped.
        let old_live: Option<u64> = old_fragments.iter().map(|f| live_row_count(f)).sum();
        let new_live: Option<u64> = self.new_fragments.iter().map(live_row_count).sum();
        if let (Some(old_live), Some(new_live)) = (old_live, new_live)
            && old_live != new_live
        {
            return Err(Error::invalid_input(format!(
                "RewriteFragments is not data equivalent: the old fragments hold {old_live} \
                 live rows but the new fragments hold {new_live}"
            )));
        }
        Ok(())
    }
}

/// The number of live (non-deleted) rows in a fragment, or `None` when the
/// fragment does not report its physical row count.
fn live_row_count(fragment: &Fragment) -> Option<u64> {
    let physical = fragment.physical_rows? as u64;
    let deleted = fragment
        .deletion_file
        .as_ref()
        .and_then(|df| df.num_deleted_rows)
        .unwrap_or(0) as u64;
    Some(physical.saturating_sub(deleted))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::format::{DeletionFile, DeletionFileType};

    use super::super::test_support::*;
    use super::super::*;
    use super::*;

    #[test]
    fn add_fragments_apply_assigns_ids_from_high_water_mark() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![sample_fragment(5)]);
        manifest.max_fragment_id = Some(5);

        let action = AddFragments {
            fragments: vec![sample_fragment(0), sample_fragment(0)],
            inserted_rows_filter: None,
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        let ids: Vec<u64> = manifest.fragments.iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![5, 6, 7]);
        assert_eq!(manifest.max_fragment_id, Some(7));
    }

    #[test]
    fn add_fragments_apply_into_empty_manifest_starts_at_zero() {
        let mut manifest = empty_manifest();
        let action = AddFragments {
            fragments: vec![sample_fragment(0), sample_fragment(0)],
            inserted_rows_filter: None,
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();
        let ids: Vec<u64> = manifest.fragments.iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(manifest.max_fragment_id, Some(1));
    }

    #[test]
    fn add_fragments_apply_preserves_explicit_ids() {
        // A non-zero id should be left alone — useful for compaction-like
        // actions that bring pre-assigned fragments back in.
        let mut manifest = empty_manifest();
        let action = AddFragments {
            fragments: vec![sample_fragment(42)],
            inserted_rows_filter: None,
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();
        assert_eq!(manifest.fragments[0].id, 42);
        assert_eq!(manifest.max_fragment_id, Some(42));
    }

    #[test]
    fn remove_fragments_apply_drops_matching_ids() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![
            sample_fragment(1),
            sample_fragment(2),
            sample_fragment(3),
        ]);
        manifest.max_fragment_id = Some(3);

        let action = RemoveFragments {
            selector: FragmentSelector::Ids(vec![2]),
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        let ids: Vec<u64> = manifest.fragments.iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![1, 3]);
        // max_fragment_id is monotonic and must not shrink.
        assert_eq!(manifest.max_fragment_id, Some(3));
    }

    #[test]
    fn remove_fragments_all_current_apply_clears_every_fragment() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![
            sample_fragment(1),
            sample_fragment(2),
            sample_fragment(3),
        ]);
        manifest.max_fragment_id = Some(3);

        let action = RemoveFragments {
            selector: FragmentSelector::AllCurrent,
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        assert!(manifest.fragments.is_empty());
        // An `AllCurrent` removal wipes the dataset, so the high-water mark
        // resets — fragment ids restart from 0 on the following `Overwrite`.
        assert_eq!(manifest.max_fragment_id, None);
    }

    #[test]
    fn remove_fragments_all_current_apply_clears_indices() {
        // An `Overwrite` translates to `RemoveFragments(AllCurrent)` + new
        // fragments. The new fragments restart from id 0, so a stale index
        // whose bitmap happened to cover the same ids would be retained by
        // the downstream coverage check and leak its on-disk files (cleanup
        // saw `index_files_removed = 0` before this drop). Clearing here
        // matches the legacy `Operation::Overwrite` arm in `build_manifest`.
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![sample_fragment(1), sample_fragment(2)]);
        manifest.max_fragment_id = Some(2);
        let mut indices = vec![index_meta("idx", &[0], Some(&[1, 2]))];

        RemoveFragments {
            selector: FragmentSelector::AllCurrent,
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();

        assert!(indices.is_empty(), "AllCurrent removal must wipe indices");
    }

    #[test]
    fn remove_fragments_ids_apply_leaves_indices_alone() {
        // A plain `Delete`-style removal targets specific fragment ids and
        // does NOT wipe indices — `retain_relevant_indices` later prunes
        // those whose effective coverage has emptied.
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![sample_fragment(1), sample_fragment(2)]);
        manifest.max_fragment_id = Some(2);
        let mut indices = vec![index_meta("idx", &[0], Some(&[1, 2]))];

        RemoveFragments {
            selector: FragmentSelector::Ids(vec![1]),
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();

        assert_eq!(indices.len(), 1);
    }

    #[test]
    fn remove_fragments_apply_ignores_unknown_ids() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![sample_fragment(1)]);
        manifest.max_fragment_id = Some(1);

        let action = RemoveFragments {
            selector: FragmentSelector::Ids(vec![42]),
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();
        assert_eq!(manifest.fragments.len(), 1);
    }

    #[test]
    fn update_deletion_vector_apply_sets_deletion_file() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![sample_fragment(7), sample_fragment(8)]);
        manifest.max_fragment_id = Some(8);

        let file = deletion_file(42);
        let action = UpdateDeletionVector {
            fragment_id: 8,
            new_deletion_file: file.clone(),
            affected_rows: None,
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        assert_eq!(manifest.fragments.len(), 2);
        assert_eq!(manifest.fragments[0].id, 7);
        assert!(manifest.fragments[0].deletion_file.is_none());
        assert_eq!(manifest.fragments[1].id, 8);
        assert_eq!(manifest.fragments[1].deletion_file, Some(file));
    }

    #[test]
    fn update_deletion_vector_apply_rejects_unknown_id() {
        let mut manifest = empty_manifest();
        let action = UpdateDeletionVector {
            fragment_id: 99,
            new_deletion_file: deletion_file(1),
            affected_rows: None,
        };
        let err = action.apply(&mut manifest, &mut vec![]).unwrap_err();
        assert!(format!("{err}").contains("99"), "got: {err}");
    }

    /// A fragment carrying a physical row count, for `RewriteFragments`
    /// data-equivalence tests.
    #[test]
    fn rewrite_fragments_apply_splices_contiguous_range() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![
            fragment_with_rows(1, 10),
            fragment_with_rows(2, 10),
            fragment_with_rows(3, 10),
            fragment_with_rows(4, 10),
        ]);
        manifest.max_fragment_id = Some(4);

        // Rewrite the contiguous run [2, 3] into a single placeholder fragment.
        let action = RewriteFragments {
            old_fragment_ids: vec![2, 3],
            new_fragments: vec![fragment_with_rows(0, 20)],
            preserves_row_ids: false,
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        let ids: Vec<u64> = manifest.fragments.iter().map(|f| f.id).collect();
        // The new fragment lands in place of [2, 3] and takes a fresh id.
        assert_eq!(ids, vec![1, 5, 4]);
        assert_eq!(manifest.max_fragment_id, Some(5));
    }

    #[test]
    fn rewrite_fragments_apply_handles_non_contiguous_old_ids() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![
            fragment_with_rows(1, 10),
            fragment_with_rows(2, 10),
            fragment_with_rows(3, 10),
        ]);
        manifest.max_fragment_id = Some(3);

        // Old ids [1, 3] are not contiguous: the retain+extend path drops them
        // and appends the new fragment.
        let action = RewriteFragments {
            old_fragment_ids: vec![1, 3],
            new_fragments: vec![fragment_with_rows(0, 20)],
            preserves_row_ids: false,
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        let ids: Vec<u64> = manifest.fragments.iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![2, 4]);
        assert_eq!(manifest.max_fragment_id, Some(4));
    }

    #[test]
    fn rewrite_fragments_apply_preserves_explicit_new_ids() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![fragment_with_rows(1, 10)]);
        manifest.max_fragment_id = Some(1);

        let action = RewriteFragments {
            old_fragment_ids: vec![1],
            new_fragments: vec![fragment_with_rows(99, 10)],
            preserves_row_ids: true,
        };
        action.apply(&mut manifest, &mut vec![]).unwrap();

        let ids: Vec<u64> = manifest.fragments.iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![99]);
        assert_eq!(manifest.max_fragment_id, Some(99));
    }

    #[test]
    fn rewrite_fragments_validate_rejects_missing_fragment() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![fragment_with_rows(1, 10)]);

        let action = RewriteFragments {
            old_fragment_ids: vec![1, 42],
            new_fragments: vec![fragment_with_rows(0, 10)],
            preserves_row_ids: false,
        };
        let err = action.apply(&mut manifest, &mut vec![]).unwrap_err();
        assert!(format!("{err}").contains("42"), "got: {err}");
    }

    #[test]
    fn rewrite_fragments_validate_rejects_empty_old_ids() {
        let manifest = empty_manifest();
        let rewrite = RewriteFragments {
            old_fragment_ids: vec![],
            new_fragments: vec![fragment_with_rows(0, 10)],
            preserves_row_ids: false,
        };
        let err = rewrite.validate(&manifest).unwrap_err();
        assert!(
            format!("{err}").contains("at least one fragment"),
            "got: {err}"
        );
    }

    #[test]
    fn rewrite_fragments_validate_rejects_row_count_mismatch() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![fragment_with_rows(1, 10), fragment_with_rows(2, 10)]);

        // The new fragment holds fewer rows than the two it replaces — the
        // rewrite is not data equivalent.
        let rewrite = RewriteFragments {
            old_fragment_ids: vec![1, 2],
            new_fragments: vec![fragment_with_rows(0, 15)],
            preserves_row_ids: false,
        };
        let err = rewrite.validate(&manifest).unwrap_err();
        assert!(format!("{err}").contains("data equivalent"), "got: {err}");
    }

    #[test]
    fn rewrite_fragments_validate_accounts_for_deleted_rows() {
        let mut manifest = empty_manifest();
        // Fragment 1 has 10 physical rows, 4 of them deleted — 6 live.
        let mut deleted = fragment_with_rows(1, 10);
        deleted.deletion_file = Some(DeletionFile {
            read_version: 1,
            id: 1,
            file_type: DeletionFileType::Array,
            num_deleted_rows: Some(4),
            base_id: None,
        });
        manifest.fragments = Arc::new(vec![deleted, fragment_with_rows(2, 10)]);

        // 6 live + 10 live = 16; the compacted fragment holds exactly 16.
        let rewrite = RewriteFragments {
            old_fragment_ids: vec![1, 2],
            new_fragments: vec![fragment_with_rows(0, 16)],
            preserves_row_ids: false,
        };
        rewrite.validate(&manifest).unwrap();
    }

    #[test]
    fn replace_fragment_columns_apply_swaps_matching_data_file() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![fragment_with_files(
            0,
            vec![data_file("a.lance", vec![0]), data_file("b.lance", vec![1])],
        )]);

        replace_columns(0, "b-new.lance", &[1])
            .apply(&mut manifest, &mut vec![])
            .unwrap();

        let files = &manifest.fragments[0].files;
        // Only the file backing field 1 is swapped; field 0's file is intact.
        assert_eq!(files[0].path, "a.lance");
        assert_eq!(files[1].path, "b-new.lance");
        assert_eq!(*files[1].fields, [1]);
    }

    #[test]
    fn replace_fragment_columns_apply_adds_all_null_column() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![fragment_with_files(
            0,
            vec![data_file("a.lance", vec![0])],
        )]);

        // Field 1 is covered by no existing file — the all-NULL-column case,
        // so the new file is appended rather than swapped.
        replace_columns(0, "b.lance", &[1])
            .apply(&mut manifest, &mut vec![])
            .unwrap();

        let files = &manifest.fragments[0].files;
        assert_eq!(files.len(), 2);
        assert_eq!(files[1].path, "b.lance");
        assert_eq!(*files[1].fields, [1]);
    }

    #[test]
    fn replace_fragment_columns_validate_rejects_missing_fragment() {
        let manifest = empty_manifest();
        let err = replace_columns(7, "b.lance", &[1])
            .validate(&manifest)
            .unwrap_err();
        assert!(
            err.to_string().contains("fragment id 7"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn replace_fragment_columns_validate_rejects_misaligned_file() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![fragment_with_files(
            0,
            vec![data_file("a.lance", vec![0])],
        )]);

        // Fields [0, 1]: field 0 is covered so this is not an all-NULL add,
        // and no existing file has exactly those fields so it is not a swap.
        let action = ReplaceFragmentColumns {
            fragment_id: 0,
            field_ids: vec![0, 1],
            new_data_files: vec![data_file("c.lance", vec![0, 1])],
            updated_row_offsets: None,
            replacement: ColumnReplacement::InPlace,
        };
        let err = action.validate(&manifest).unwrap_err();
        assert!(
            err.to_string()
                .contains("Expected to modify the fragment but no changes were made"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn replace_fragment_columns_apply_tombstone_installs_the_carried_file_list() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![fragment_with_files(
            0,
            vec![data_file("f0.lance", vec![0, 1])],
        )]);
        let mut indices = vec![index_meta("covers-1", &[1], Some(&[0]))];

        // The action carries the fragment's complete post-rewrite file list:
        // field 1 tombstoned out of its original file, plus the fresh file.
        ReplaceFragmentColumns {
            fragment_id: 0,
            field_ids: vec![1],
            new_data_files: vec![
                data_file("f0.lance", vec![0, -2]),
                data_file("f0_b_v2.lance", vec![1]),
            ],
            updated_row_offsets: None,
            replacement: ColumnReplacement::Tombstone,
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();

        let files = &manifest.fragments[0].files;
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path, "f0.lance");
        assert_eq!(*files[0].fields, [0, -2]);
        assert_eq!(files[1].path, "f0_b_v2.lance");
        assert_eq!(*files[1].fields, [1]);
        // The index over the replaced field drops the modified fragment.
        assert_eq!(indices[0].fragment_bitmap, Some(bitmap(&[])));
    }

    #[test]
    fn replace_fragment_columns_apply_tombstone_keeps_a_fully_tombstoned_file() {
        // The `merge_insert` partial-schema producer tombstones a rewritten
        // field in place without dropping the file it emptied — so a kept
        // `[-2, -2]` file appears mid-list. `apply` must reproduce it verbatim
        // rather than re-deriving (and dropping) the emptied file.
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![fragment_with_files(
            0,
            vec![data_file("shared.lance", vec![0, 2])],
        )]);

        ReplaceFragmentColumns {
            fragment_id: 0,
            field_ids: vec![0, 2],
            new_data_files: vec![
                data_file("shared.lance", vec![-2, -2]),
                data_file("rewritten.lance", vec![0, 2]),
            ],
            updated_row_offsets: None,
            replacement: ColumnReplacement::Tombstone,
        }
        .apply(&mut manifest, &mut vec![])
        .unwrap();

        let files = &manifest.fragments[0].files;
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path, "shared.lance");
        assert_eq!(*files[0].fields, [-2, -2]);
        assert_eq!(files[1].path, "rewritten.lance");
        assert_eq!(*files[1].fields, [0, 2]);
    }

    #[test]
    fn replace_fragment_columns_tombstone_validate_rejects_empty_file_list() {
        let mut manifest = empty_manifest();
        manifest.fragments = Arc::new(vec![fragment_with_files(
            0,
            vec![data_file("f0.lance", vec![0, 1])],
        )]);

        // A tombstone replacement carries the whole post-rewrite file list, so
        // an empty list would leave the fragment with no data files.
        let err = ReplaceFragmentColumns {
            fragment_id: 0,
            field_ids: vec![1],
            new_data_files: vec![],
            updated_row_offsets: None,
            replacement: ColumnReplacement::Tombstone,
        }
        .validate(&manifest)
        .unwrap_err();
        assert!(
            err.to_string().contains("empty data file list"),
            "unexpected error: {err}"
        );
    }
}
