// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keeping index metadata honest about what the new fragment list contains.
//!
//! An index entry claims coverage of a set of fragments and fields. Any operation
//! that rewrites data can invalidate part of that claim, so a commit has to either
//! narrow the entry's fragment bitmap, drop the fields it no longer describes, or
//! drop the index. Getting this wrong does not fail the commit -- it silently
//! returns stale rows from the index -- so each rule here is paired with a test.

use crate::format::overlay::staleness::collect_overlay_stale_frags;
use crate::format::{Fragment, IndexMetadata};
use crate::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use crate::system_index::is_system_index;
use crate::transaction::{RewriteGroup, RewrittenIndex, Transaction};
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use std::collections::{HashMap, HashSet};

/// Insert `field`'s id and every descendant's id into `field_ids`.
///
/// A private copy of `lance::index::collect_subtree_field_ids`: the walk is four
/// lines and the alternative is exporting a lance-table helper solely so the lance
/// crate's copy could delegate to it.
fn collect_subtree_field_ids(field: &lance_core::datatypes::Field, field_ids: &mut HashSet<i32>) {
    field_ids.insert(field.id);
    for child in &field.children {
        collect_subtree_field_ids(child, field_ids);
    }
}

impl Transaction {
    pub(super) fn register_pure_rewrite_rows_update_frags_in_indices(
        indices: &mut [IndexMetadata],
        pure_update_frag_ids: &[u64],
        original_fragment_ids: &[u64],
        fields_for_preserving_frag_bitmap: &[u32],
        original_overlaid_frags: &HashMap<u32, &Fragment>,
        schema: &Schema,
    ) -> Result<()> {
        if pure_update_frag_ids.is_empty() {
            return Ok(());
        }

        // Read the parameter name as the inverse of what it does. Despite
        // "for_preserving_frag_bitmap", this list is the set of fields whose VALUES were
        // updated by the rewrite, and an index depending on any of them is *excluded* from
        // the bitmap extension below -- i.e. passing a field here withholds preservation
        // rather than granting it. A caller that cannot prove any field survived the rewrite
        // unchanged therefore passes *every* field id, which correctly extends no bitmap at
        // all. The name is kept only because renaming it would churn shared callers.
        let value_updated_field_set = fields_for_preserving_frag_bitmap
            .iter()
            .collect::<HashSet<_>>();

        for index in indices.iter_mut() {
            // Physical row addresses cannot follow moved rows into a new fragment.
            // Leave that fragment uncovered so the scanner reads it directly.
            if index.results_are_row_addrs() {
                continue;
            }
            // Covering (`covering_fields`) values are materialized in the index
            // storage just like the indexed `fields`, so a rewrite of either
            // makes this fragment's entries stale. Reusing them would serve the
            // pre-update value, so treat both as "modified field" here.
            //
            // Expand to the leaf subtree, exactly as every sibling prune does:
            // `covering_fields` can only ever hold TOP-LEVEL ids (index creation
            // resolves whole columns and rejects dotted names), while the caller
            // may report the individual leaves it rewrote. Comparing raw ids would
            // miss a covered struct whose child was rewritten and wrongly extend
            // this index's bitmap onto the moved fragment, serving the pre-update
            // subfield value from index storage.
            let index_covers_modified_field = Self::index_dependent_field_ids(index, schema)
                .iter()
                .any(|field_id| value_updated_field_set.contains(field_id));
            if index_covers_modified_field {
                continue;
            }
            let Some(fragment_bitmap) = index.fragment_bitmap.as_ref() else {
                continue;
            };

            // Check that all the original fragments containing the updated rows are covered by
            // the index. If not, some updated rows were not indexed, so we cannot index them.
            let index_covers_all_original_fragments = original_fragment_ids
                .iter()
                .all(|&fragment_id| fragment_bitmap.contains(fragment_id as u32));
            if !index_covers_all_original_fragments {
                continue;
            }

            // A rewrite materializes overlays.  If any of those overlays touched the
            // column being indexed then the rewrite will modify that column.  As a
            // result, that index will no longer cover the fragment and it does not
            // count as a pure rewrite and we must exclude it from the index's fragment
            // bitmap.
            let mut overlay_stale = RoaringBitmap::new();
            collect_overlay_stale_frags(
                index,
                original_overlaid_frags,
                &mut overlay_stale,
                schema,
            )?;
            if !overlay_stale.is_empty() {
                continue;
            }

            if let Some(fragment_bitmap) = index.fragment_bitmap.as_mut() {
                for fragment_id in pure_update_frag_ids.iter().map(|f| *f as u32) {
                    fragment_bitmap.insert(fragment_id);
                }
            }
        }
        Ok(())
    }

    /// The full set of leaf field ids an index depends on: its indexed key fields plus
    /// every covered (included) field, each expanded to its whole subtree. `fields`
    /// already lists `covering_fields` as its trailing entries (see
    /// [`IndexMetadata::covering_fields`]), so walking `fields` alone covers both; a
    /// modified/overlaid data file lists leaf ids, so a covered struct's parent id must
    /// still be expanded to recognize a change to one of its subfields. Returned as raw
    /// `i32` (the field id space of `DataFile.fields` / overlay fields). Shared by the
    /// freshness prunes and the conflict-rebase checks.
    pub fn index_dependent_leaf_ids(index: &IndexMetadata, schema: &Schema) -> HashSet<i32> {
        let mut ids: HashSet<i32> = HashSet::new();
        for &id in index.fields.iter() {
            match schema.field_by_id(id) {
                Some(field) => collect_subtree_field_ids(field, &mut ids),
                None => {
                    ids.insert(id);
                }
            }
        }
        ids
    }

    fn index_dependent_field_ids(index: &IndexMetadata, schema: &Schema) -> HashSet<u32> {
        Self::index_dependent_leaf_ids(index, schema)
            .into_iter()
            .filter_map(|id| u32::try_from(id).ok())
            .collect()
    }

    /// Drop `updated_fragments` from the coverage of any index that depends on a field
    /// listed in `fields_modified`. Covered struct fields are expanded to their leaf
    /// subtree: `covering_fields` records the parent struct id while a modified data
    /// file lists leaf ids, so exact-id matching would miss an update to a covered
    /// struct's subfield and serve its stale value from the index. Every caller must
    /// therefore supply the schema (the conflict-rebase path captures it at the read
    /// version).
    pub fn prune_updated_fields_from_indices(
        indices: &mut [IndexMetadata],
        updated_fragments: &[Fragment],
        fields_modified: &[u32],
        schema: &Schema,
    ) {
        if fields_modified.is_empty() {
            return;
        }

        let deps = Self::index_dependent_field_ids_for_each(indices, schema);
        Self::prune_updated_fields_with_deps(indices, updated_fragments, fields_modified, &deps);
    }

    /// The dependent-leaf-id set for each index, positionally parallel to `indices`.
    ///
    /// Resolving these walks the schema and allocates per index, so callers that prune
    /// fragment-by-fragment must compute this once and reuse it rather than paying it per
    /// fragment -- on a compaction rewriting N fragments across M indices the per-fragment
    /// form is N*M schema walks on the commit path.
    fn index_dependent_field_ids_for_each(
        indices: &[IndexMetadata],
        schema: &Schema,
    ) -> Vec<HashSet<u32>> {
        indices
            .iter()
            .map(|index| Self::index_dependent_field_ids(index, schema))
            .collect()
    }

    /// [`Self::prune_updated_fields_from_indices`] with the per-index dependent-id sets
    /// already resolved. `deps` must be positionally parallel to `indices`.
    fn prune_updated_fields_with_deps(
        indices: &mut [IndexMetadata],
        updated_fragments: &[Fragment],
        fields_modified: &[u32],
        deps: &[HashSet<u32>],
    ) {
        if fields_modified.is_empty() {
            return;
        }
        debug_assert_eq!(indices.len(), deps.len());

        // If we modified any fields in the fragments then we need to remove those fragments
        // from the index if the index covers one of those modified fields.
        let fields_modified_set = fields_modified.iter().copied().collect::<HashSet<u32>>();
        for (index, dependent_ids) in indices.iter_mut().zip(deps) {
            let touches_index = dependent_ids
                .iter()
                .any(|field_id| fields_modified_set.contains(field_id));
            if touches_index && let Some(fragment_bitmap) = &mut index.fragment_bitmap {
                for fragment_id in updated_fragments.iter().map(|f| f.id as u32) {
                    fragment_bitmap.remove(fragment_id);
                }
            }
        }
    }

    /// Map each (non-tombstoned) field id in a fragment to the path of the data
    /// file that backs it.
    // `pub`: the conflict-rebase path in the `lance` crate resolves per-fragment
    // field ownership with this too.
    pub fn fragment_field_paths(frag: &Fragment) -> HashMap<i32, &str> {
        let mut map = HashMap::new();
        for file in &frag.files {
            for &field_id in file.fields.iter() {
                if field_id >= 0 {
                    map.insert(field_id, file.path.as_str());
                }
            }
        }
        map
    }

    /// A `Merge` can rewrite a column's data *in place* -- the field stays in the
    /// schema but its backing data file changes (the overlay fragment carries a new
    /// file for the field and tombstones its old field id). `retain_relevant_indices`
    /// only drops indices for *removed* fields, so without this the index keeps
    /// covering the rewritten fragments with stale entries. Remove each such fragment
    /// from any index covering a field whose backing data file changed.
    pub(super) fn prune_merge_rewritten_fields_from_indices(
        indices: &mut [IndexMetadata],
        prev_fragments: &[Fragment],
        new_fragments: &[Fragment],
        schema: &Schema,
    ) {
        let prev_by_id: HashMap<u64, &Fragment> =
            prev_fragments.iter().map(|f| (f.id, f)).collect();
        // Resolved once for all fragments: this is the commit path, and the per-fragment
        // form costs a schema walk and two HashSet allocations per index per fragment.
        let deps = Self::index_dependent_field_ids_for_each(indices, schema);
        for new_frag in new_fragments {
            let Some(prev) = prev_by_id.get(&new_frag.id) else {
                continue; // brand-new fragment: nothing stale to prune
            };
            let prev_paths = Self::fragment_field_paths(prev);
            let new_paths = Self::fragment_field_paths(new_frag);
            // Fields still present whose backing file path changed == rewritten data.
            let changed: Vec<u32> = prev_paths
                .iter()
                .filter(|(field_id, prev_path)| {
                    new_paths
                        .get(*field_id)
                        .is_some_and(|new_path| new_path != *prev_path)
                })
                .map(|(field_id, _)| *field_id as u32)
                .collect();
            if changed.is_empty() {
                continue;
            }
            Self::prune_updated_fields_with_deps(
                indices,
                std::slice::from_ref(new_frag),
                &changed,
                &deps,
            );
        }
    }

    /// After a `Rewrite` fully compacts a fragment, its data overlays are baked
    /// into the new fragment's base data. An index built *before* one of those
    /// overlays (`overlay.committed_version > index.dataset_version`) indexed the
    /// stale pre-overlay values -- and unlike a live overlay, the compacted
    /// fragment no longer signals that staleness to the query path. Drop each
    /// rewritten (new) fragment from the coverage of any index covering a field
    /// such an overlay supplied, so those rows fall back to a flat scan.
    pub(super) fn prune_overlay_stale_fields_from_indices(
        indices: &mut [IndexMetadata],
        groups: &[RewriteGroup],
        schema: &Schema,
    ) {
        // Resolved once for all groups, as in `prune_merge_rewritten_fields_from_indices`:
        // this is the commit path, and the per-index form costs a schema walk and two
        // HashSet allocations for every (group, index) pair.
        let deps = Self::index_dependent_field_ids_for_each(indices, schema);
        for group in groups {
            // field id -> newest overlay committed_version supplying that field
            let mut overlaid_field_versions: HashMap<u32, u64> = HashMap::new();
            for old_frag in &group.old_fragments {
                for overlay in &old_frag.overlays {
                    for &field_id in overlay.data_file.fields.iter() {
                        // Tombstoned (obsolete) overlay fields (< 0) supply nothing.
                        let Ok(field_id) = u32::try_from(field_id) else {
                            continue;
                        };
                        let entry = overlaid_field_versions.entry(field_id).or_insert(0);
                        *entry = (*entry).max(overlay.committed_version);
                    }
                }
            }
            if overlaid_field_versions.is_empty() {
                continue;
            }

            let new_fragment_ids = group
                .new_fragments
                .iter()
                .map(|f| f.id as u32)
                .collect::<Vec<_>>();
            for (index, dependent_field_ids) in indices.iter_mut().zip(deps.iter()) {
                // A covered (included) field an overlay supplied makes the index just as
                // stale as an indexed field would -- the index storage still holds the
                // pre-overlay copy of that column. `deps` expands covered structs to their
                // leaf subtree, since the overlay lists leaf ids.
                let is_stale = dependent_field_ids.iter().any(|field_id| {
                    overlaid_field_versions
                        .get(field_id)
                        .is_some_and(|&overlay_version| overlay_version > index.dataset_version)
                });
                if is_stale && let Some(fragment_bitmap) = &mut index.fragment_bitmap {
                    for new_id in &new_fragment_ids {
                        fragment_bitmap.remove(*new_id);
                    }
                }
            }
        }
    }

    /// Reject a commit that would drop, rename, retype, or otherwise change the subtree of
    /// a field an index *covers* (an "included" column). A covered field's physical payload
    /// schema is fixed at index build time, so an index whose key survives but whose covered
    /// column's subtree changed would keep emitting the old payload schema while covered ANN
    /// queries declare the new one. This fires for two commit operations:
    /// - `Project`, which can drop/rename/retype a covered field (public drop/alter APIs
    ///   preflight this, but a raw `Operation::Project` reaches `build_manifest` directly);
    /// - `Merge` (how `add_columns` commits), which can grow a covered *struct* by adding a
    ///   child -- even an AllNulls, metadata-only child that writes no data file, so the
    ///   file-path-keyed coverage prune never sees it.
    ///
    /// `old_schema` is the pre-commit schema (needed to detect renames/retypes/child-adds,
    /// which keep the covered field's id). `op` names the operation for the error message.
    pub(super) fn reject_covered_field_subtree_change(
        indices: &[IndexMetadata],
        old_schema: Option<&Schema>,
        new_schema: &Schema,
        op: &str,
    ) -> Result<()> {
        let new_ids: HashSet<i32> = new_schema.fields_pre_order().map(|f| f.id).collect();
        for index in indices {
            // Skip only if a KEYED field left the schema: `retain_relevant_indices` drops
            // the whole index once any of `fields` is gone, so a dropped key field means
            // the index is on its way out anyway and there is no dangling covering to
            // protect. Checking the *keyed* prefix specifically (not all of `fields`)
            // matters because `fields` also lists the covering suffix (see
            // `IndexMetadata::covering_fields`): a commit that drops *only* a covering
            // field, leaving the key intact, must NOT be skipped here. If it were, this
            // guard would let the drop through uncontested, and `retain_relevant_indices`
            // -- which keys its own retention on the very same full `fields` list -- would
            // then silently delete the entire index (not just desync its covering) with no
            // error. Below, that same dropped covering id is caught by
            // `covered_field_subtree_changed`'s "removed" case and rejected instead.
            // (`FLAG_COVERED_INDEX_METADATA` fences pre-covering builds off the whole
            // dataset, so a declaration erased by an old writer -- once this guard's
            // other concern -- cannot arise.)
            if !index.keyed_fields().iter().all(|id| new_ids.contains(id)) {
                continue;
            }
            // A field the index is also KEYED on gets NO exemption: carried refine vectors
            // (`store_vectors_for_refine`) declare the indexed column in `covering_fields`,
            // and index storage holds that copy under the column's build-time name. A rename
            // preserves the field id but not that name, leaving storage the rebuild paths
            // cannot resolve, so it is refused here exactly as it is in the `schema_evolution`
            // preflight and in `prune_stale_segment_coverage`. A change that *removes* the id
            // (a drop, or a cast, which reassigns a fresh one) never reaches this loop: the
            // keyed-field check above skipped the index and `retain_relevant_indices` drops it.
            for &covered_id in index.covering_fields.iter() {
                let changed = match old_schema {
                    Some(old) => Self::covered_field_subtree_changed(old, new_schema, covered_id),
                    // Without the pre-commit schema we can only detect a drop.
                    None => new_schema.field_by_id(covered_id).is_none(),
                };
                if changed {
                    return Err(Error::invalid_input(format!(
                        "{op} would drop or alter covered (included) field id {covered_id} \
                         (its subtree changed), still used by index '{}'. Drop the index with \
                         drop_index() before changing the column.",
                        index.name
                    )));
                }
            }
        }
        Ok(())
    }

    /// Whether a covered ("included") field's subtree differs between `old_schema` and
    /// `new_schema` -- dropped, renamed, retyped, a nullability flip, or any child change.
    /// Such a change means the index's stored physical payload no longer matches the
    /// schema a query declares. `data_type()` is recursive for structs, so it also covers
    /// a child's name/type/nullability change.
    pub fn covered_field_subtree_changed(
        old_schema: &Schema,
        new_schema: &Schema,
        covered_id: i32,
    ) -> bool {
        let Some(old) = old_schema.field_by_id(covered_id) else {
            return false;
        };
        match new_schema.field_by_id(covered_id) {
            None => true,
            Some(new) => {
                // Metadata is compared for the same reason name, type and nullability are:
                // this guard exists to turn a covering desync into an explicit error, and
                // the read path's acceptance test (`covering_fields_match` in
                // `io::exec::knn`) compares metadata too. A difference this guard ignores
                // but that one does not is not a caught error -- it is a covered index
                // that silently stops serving its columns and falls back to a base-table
                // read, with no way for the user to learn why.
                old.name != new.name
                    || old.nullable != new.nullable
                    || old.data_type() != new.data_type()
                    || old.metadata != new.metadata
            }
        }
    }

    pub(crate) fn retain_relevant_indices(
        indices: &mut Vec<IndexMetadata>,
        schema: &Schema,
        fragments: &[Fragment],
    ) {
        let field_ids = schema
            .fields_pre_order()
            .map(|f| f.id)
            .collect::<HashSet<_>>();

        // Remove indices for fields no longer in schema
        indices.retain(|existing_index| {
            existing_index
                .fields
                .iter()
                .all(|field_id| field_ids.contains(field_id))
                || is_system_index(existing_index)
        });

        let mut indices_by_name: std::collections::HashMap<String, Vec<&IndexMetadata>> =
            std::collections::HashMap::new();

        for index in indices.iter() {
            if index.name != FRAG_REUSE_INDEX_NAME {
                indices_by_name
                    .entry(index.name.clone())
                    .or_default()
                    .push(index);
            }
        }

        let mut uuids_to_keep = std::collections::HashSet::new();

        let existing_fragments = fragments
            .iter()
            .map(|f| f.id as u32)
            .collect::<RoaringBitmap>();

        for (_, same_name_indices) in indices_by_name {
            // Unknown coverage is not empty coverage: a segment whose bitmap is
            // missing has never been measured, and dropping it deletes an index
            // that migration could not open yet.
            let (unknown_coverage, same_name_indices): (Vec<_>, Vec<_>) = same_name_indices
                .into_iter()
                .partition(|index| index.fragment_bitmap.is_none());
            for index in unknown_coverage {
                uuids_to_keep.insert(index.uuid);
            }

            if same_name_indices.len() > 1 {
                let (empty_indices, non_empty_indices): (Vec<_>, Vec<_>) =
                    same_name_indices.iter().partition(|index| {
                        index
                            .effective_fragment_bitmap(&existing_fragments)
                            .as_ref()
                            .is_none_or(|bitmap| bitmap.is_empty())
                    });

                if non_empty_indices.is_empty() {
                    // All indices are empty -- keep only the oldest definition.
                    //
                    // An empty index definition is still correct: the scanner
                    // falls back to scanning unindexed fragments, and normal
                    // index maintenance rebuilds coverage once rows accrue.
                    // Dropping the definition instead would silently lose the
                    // index whenever an operation replaces every fragment it
                    // covered (e.g. a full table rewrite), leaving the dataset
                    // without its declared index.
                    let mut sorted_indices = empty_indices;
                    sorted_indices.sort_by_key(|index: &&IndexMetadata| index.dataset_version);

                    if let Some(oldest) = sorted_indices.first() {
                        uuids_to_keep.insert(oldest.uuid);
                    }
                } else {
                    for index in non_empty_indices {
                        uuids_to_keep.insert(index.uuid);
                    }
                }
            } else {
                // Single index whose column is still in schema: keep it, even
                // when its coverage is empty (see the all-empty note above).
                if let Some(index) = same_name_indices.first() {
                    uuids_to_keep.insert(index.uuid);
                }
            }
        }

        indices.retain(|index| {
            index.name == FRAG_REUSE_INDEX_NAME || uuids_to_keep.contains(&index.uuid)
        });
    }

    pub(super) fn recalculate_fragment_bitmap(
        old: &RoaringBitmap,
        groups: &[RewriteGroup],
    ) -> Result<RoaringBitmap> {
        let mut new_bitmap = old.clone();
        for group in groups {
            let any_in_index = group
                .old_fragments
                .iter()
                .any(|frag| old.contains(frag.id as u32));
            let all_in_index = group
                .old_fragments
                .iter()
                .all(|frag| old.contains(frag.id as u32));
            // Any rewrite group may or may not be covered by the index.  However, if any fragment
            // in a rewrite group was previously covered by the index then all fragments in the rewrite
            // group must have been previously covered by the index.  plan_compaction takes care of
            // this for us so this should be safe to assume.
            if any_in_index {
                if all_in_index {
                    for frag_id in group.old_fragments.iter().map(|frag| frag.id as u32) {
                        new_bitmap.remove(frag_id);
                    }
                    new_bitmap.extend(group.new_fragments.iter().map(|frag| frag.id as u32));
                } else {
                    return Err(Error::invalid_input(
                        "The compaction plan included a rewrite group that was a split of indexed and non-indexed data",
                    ));
                }
            }
        }
        Ok(new_bitmap)
    }

    pub(super) fn handle_rewrite_indices(
        indices: &mut [IndexMetadata],
        rewritten_indices: &[RewrittenIndex],
        groups: &[RewriteGroup],
    ) -> Result<()> {
        let mut modified_indices = HashSet::new();

        for rewritten_index in rewritten_indices {
            if !modified_indices.insert(rewritten_index.old_id) {
                return Err(Error::invalid_input(format!(
                    "An invalid compaction plan must have been generated because multiple tasks modified the same index: {}",
                    rewritten_index.old_id
                )));
            }

            // Skip indices that no longer exist (may have been removed by concurrent operation)
            let Some(index) = indices
                .iter_mut()
                .find(|idx| idx.uuid == rewritten_index.old_id)
            else {
                continue;
            };

            index.fragment_bitmap = Some(Self::recalculate_fragment_bitmap(
                index.fragment_bitmap.as_ref().ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Cannot rewrite index {} which did not store fragment bitmap",
                        index.uuid
                    ))
                })?,
                groups,
            )?);
            index.uuid = rewritten_index.new_id;
            // Update file sizes to match the new index files. When not available
            // (e.g., from older writers), clear the old file sizes to avoid
            // using stale sizes from the pre-remap index.
            index.files = rewritten_index.new_index_files.clone();
        }
        Ok(())
    }

    pub(super) fn handle_rewrite_fragments(
        final_fragments: &mut Vec<Fragment>,
        groups: &[RewriteGroup],
        fragment_id: &mut u64,
        version: u64,
        _next_row_id: Option<&u64>,
    ) -> Result<()> {
        for group in groups {
            // If the old fragments are contiguous, find the range
            let replace_range = {
                let start = final_fragments
                    .iter()
                    .enumerate()
                    .find(|(_, f)| f.id == group.old_fragments[0].id)
                    .ok_or_else(|| {
                        Error::commit_conflict_source(
                            version,
                            format!(
                                "dataset does not contain a fragment a rewrite operation wants to replace: id={}",
                                group.old_fragments[0].id
                            )
                            .into(),
                        )
                    })?
                    .0;

                // Verify old_fragments matches contiguous range
                let mut i = 1;
                loop {
                    if i == group.old_fragments.len() {
                        break Some(start..start + i);
                    }
                    if final_fragments[start + i].id != group.old_fragments[i].id {
                        break None;
                    }
                    i += 1;
                }
            };

            let new_fragments = Self::fragments_with_ids(group.new_fragments.clone(), fragment_id)
                .collect::<Vec<_>>();

            // Version metadata for rewritten fragments is handled by the compaction code
            // (recalc_versions_for_rewritten_fragments) which preserves version information
            // from the original fragments. We don't modify it here.

            if let Some(replace_range) = replace_range {
                // Efficiently path using slice
                final_fragments.splice(replace_range, new_fragments);
            } else {
                // Slower path for non-contiguous ranges
                for fragment in group.old_fragments.iter() {
                    final_fragments.retain(|f| f.id != fragment.id);
                }
                final_fragments.extend(new_fragments);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::test_support::overlay_with_field;
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::Schema as LanceSchema;
    use uuid::Uuid;

    #[test]
    fn test_rewrite_fragments() {
        let existing_fragments: Vec<Fragment> = (0..10).map(Fragment::new).collect();

        let mut final_fragments = existing_fragments;
        let rewrite_groups = vec![
            // Since these are contiguous, they will be put in the same location
            // as 1 and 2.
            RewriteGroup {
                old_fragments: vec![Fragment::new(1), Fragment::new(2)],
                // These two fragments were previously reserved
                new_fragments: vec![Fragment::new(15), Fragment::new(16)],
            },
            // These are not contiguous, so they will be inserted at the end.
            RewriteGroup {
                old_fragments: vec![Fragment::new(5), Fragment::new(8)],
                // We pretend this id was not reserved.  Does not happen in practice today
                // but we want to leave the door open.
                new_fragments: vec![Fragment::new(0)],
            },
        ];

        let mut fragment_id = 20;
        let version = 0;

        Transaction::handle_rewrite_fragments(
            &mut final_fragments,
            &rewrite_groups,
            &mut fragment_id,
            version,
            None,
        )
        .unwrap();

        assert_eq!(fragment_id, 21);

        let expected_fragments: Vec<Fragment> = vec![
            Fragment::new(0),
            Fragment::new(15),
            Fragment::new(16),
            Fragment::new(3),
            Fragment::new(4),
            Fragment::new(6),
            Fragment::new(7),
            Fragment::new(9),
            Fragment::new(20),
        ];

        assert_eq!(final_fragments, expected_fragments);
    }

    #[test]
    fn test_retain_indices_removes_missing_fields() {
        let schema = create_test_schema(&[1, 2]);
        let fragments = vec![Fragment::new(1), Fragment::new(2)];

        let mut indices = vec![
            create_test_index("idx1", 1, 1, Some(RoaringBitmap::from_iter([1])), false),
            create_test_index("idx2", 2, 1, Some(RoaringBitmap::from_iter([1])), false),
            create_test_index("idx3", 99, 1, Some(RoaringBitmap::from_iter([1])), false), // Field doesn't exist
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        assert_eq!(indices.len(), 2);
        assert!(indices.iter().all(|idx| idx.fields[0] != 99));
    }

    #[test]
    fn test_retain_indices_keeps_system_indices() {
        use crate::system_index::mem_wal::MEM_WAL_INDEX_NAME;

        let schema = create_test_schema(&[1, 2]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_system_index(FRAG_REUSE_INDEX_NAME, 99), // Field doesn't exist but should be kept
            create_system_index(MEM_WAL_INDEX_NAME, 99), // Field doesn't exist but should be kept
            create_test_index("regular_idx", 99, 1, Some(RoaringBitmap::new()), false), // Should be removed
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        assert_eq!(indices.len(), 2);
        assert!(indices.iter().any(|idx| idx.name == FRAG_REUSE_INDEX_NAME));
        assert!(indices.iter().any(|idx| idx.name == MEM_WAL_INDEX_NAME));
    }

    #[test]
    fn test_retain_indices_keeps_fragment_reuse_index() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_system_index(FRAG_REUSE_INDEX_NAME, 1),
            create_test_index("other_idx", 1, 1, Some(RoaringBitmap::new()), false),
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Fragment reuse index should always be kept
        assert!(indices.iter().any(|idx| idx.name == FRAG_REUSE_INDEX_NAME));
    }

    #[test]
    fn test_retain_single_empty_scalar_index() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![create_test_index(
            "scalar_idx",
            1,
            1,
            Some(RoaringBitmap::new()), // Empty bitmap
            false,
        )];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Single empty scalar index should be kept
        assert_eq!(indices.len(), 1);
    }

    #[test]
    fn test_retain_single_empty_vector_index_is_kept() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![create_test_index(
            "vector_idx",
            1,
            1,
            Some(RoaringBitmap::new()), // Empty bitmap
            true,
        )];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // The empty definition is retained: coverage is empty but the index
        // declaration must survive operations that replace every fragment.
        assert_eq!(indices.len(), 1);
    }

    #[test]
    fn test_retain_single_nonempty_index() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut scalar_indices = vec![create_test_index(
            "scalar_idx",
            1,
            1,
            Some(RoaringBitmap::from_iter([1])),
            false,
        )];

        let mut vector_indices = vec![create_test_index(
            "vector_idx",
            1,
            1,
            Some(RoaringBitmap::from_iter([1])),
            true,
        )];

        Transaction::retain_relevant_indices(&mut scalar_indices, &schema, &fragments);
        Transaction::retain_relevant_indices(&mut vector_indices, &schema, &fragments);

        // Both should be kept
        assert_eq!(scalar_indices.len(), 1);
        assert_eq!(vector_indices.len(), 1);
    }

    #[test]
    fn test_retain_single_index_with_none_bitmap() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut scalar_indices = vec![create_test_index("scalar_idx", 1, 1, None, false)];
        let mut vector_indices = vec![create_test_index("vector_idx", 1, 1, None, true)];

        Transaction::retain_relevant_indices(&mut scalar_indices, &schema, &fragments);
        Transaction::retain_relevant_indices(&mut vector_indices, &schema, &fragments);

        // Both kept: a None bitmap is unknown coverage, not empty coverage, and
        // an unmeasured segment is retained regardless of index type.
        assert_eq!(scalar_indices.len(), 1);
        assert_eq!(vector_indices.len(), 1);
    }

    #[test]
    fn test_retain_unknown_coverage_alongside_nonempty_sibling() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1), Fragment::new(2)];

        let mut indices = vec![
            create_test_index("idx", 1, 1, None, false), // Coverage never measured
            create_test_index("idx", 1, 2, Some(RoaringBitmap::from_iter([2])), false),
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // The unmeasured segment must survive its non-empty sibling: its bitmap
        // is missing because migration could not open the index, and deleting
        // the segment would take the only record of it with it.
        assert_eq!(indices.len(), 2);
        assert!(indices.iter().any(|idx| idx.fragment_bitmap.is_none()));
    }

    #[test]
    fn test_retain_multiple_empty_scalar_indices_keeps_oldest() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_test_index("idx", 1, 3, Some(RoaringBitmap::new()), false),
            create_test_index("idx", 1, 1, Some(RoaringBitmap::new()), false), // Oldest
            create_test_index("idx", 1, 2, Some(RoaringBitmap::new()), false),
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Should keep only the oldest (dataset_version = 1)
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].dataset_version, 1);
    }

    #[test]
    fn test_retain_multiple_empty_vector_indices_keeps_oldest() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_test_index("vec_idx", 1, 1, Some(RoaringBitmap::new()), true),
            create_test_index("vec_idx", 1, 2, Some(RoaringBitmap::new()), true),
            create_test_index("vec_idx", 1, 3, Some(RoaringBitmap::new()), true),
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Same as the scalar case: all deltas are empty, so only the oldest
        // definition survives.
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].dataset_version, 1);
    }

    #[test]
    fn test_retain_mixed_empty_nonempty_keeps_nonempty() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_test_index("idx", 1, 1, Some(RoaringBitmap::new()), false), // Empty
            create_test_index("idx", 1, 2, Some(RoaringBitmap::from_iter([1])), false), // Non-empty
            create_test_index("idx", 1, 3, Some(RoaringBitmap::new()), false), // Empty
            create_test_index("idx", 1, 4, Some(RoaringBitmap::from_iter([1])), false), // Non-empty
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Should keep only non-empty indices
        assert_eq!(indices.len(), 2);
        assert!(
            indices
                .iter()
                .all(|idx| idx.dataset_version == 2 || idx.dataset_version == 4)
        );
    }

    #[test]
    fn test_retain_mixed_empty_nonempty_vector_keeps_nonempty() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_test_index("vec_idx", 1, 1, Some(RoaringBitmap::new()), true), // Empty
            create_test_index("vec_idx", 1, 2, Some(RoaringBitmap::from_iter([1])), true), // Non-empty
            create_test_index("vec_idx", 1, 3, Some(RoaringBitmap::new()), true),          // Empty
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Should keep only non-empty index
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].dataset_version, 2);
    }

    #[test]
    fn test_retain_fragment_bitmap_with_nonexistent_fragments() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1), Fragment::new(2)]; // Only fragments 1 and 2 exist

        let mut indices = vec![create_test_index(
            "idx",
            1,
            1,
            Some(RoaringBitmap::from_iter([1, 2, 3, 4])), // References non-existent fragments 3, 4
            false,
        )];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Should still keep the index (effective bitmap will be intersection with existing)
        assert_eq!(indices.len(), 1);
        // Original bitmap should be unchanged
        assert_eq!(
            indices[0].fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([1, 2, 3, 4])
        );
    }

    #[test]
    fn test_retain_effective_empty_bitmap_single_index() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(5), Fragment::new(6)];

        // Bitmap references fragments that don't exist, so effective bitmap is empty
        let mut scalar_indices = vec![create_test_index(
            "scalar_idx",
            1,
            1,
            Some(RoaringBitmap::from_iter([1, 2, 3])),
            false,
        )];

        let mut vector_indices = vec![create_test_index(
            "vector_idx",
            1,
            1,
            Some(RoaringBitmap::from_iter([1, 2, 3])),
            true,
        )];

        Transaction::retain_relevant_indices(&mut scalar_indices, &schema, &fragments);
        Transaction::retain_relevant_indices(&mut vector_indices, &schema, &fragments);

        // Both kept: a single index whose column is still in schema is
        // retained even when its effective coverage is empty.
        assert_eq!(scalar_indices.len(), 1);
        assert_eq!(vector_indices.len(), 1);
    }

    #[test]
    fn test_retain_different_index_names() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_test_index("idx_a", 1, 1, Some(RoaringBitmap::new()), false),
            create_test_index("idx_b", 1, 1, Some(RoaringBitmap::new()), true),
            create_test_index("idx_c", 1, 1, Some(RoaringBitmap::from_iter([1])), false),
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // All three kept: empty definitions are retained for scalar and
        // vector indexes alike.
        assert_eq!(indices.len(), 3);
        assert!(indices.iter().any(|idx| idx.name == "idx_a"));
        assert!(indices.iter().any(|idx| idx.name == "idx_b"));
        assert!(indices.iter().any(|idx| idx.name == "idx_c"));
    }

    #[test]
    fn test_retain_empty_indices_vec() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices: Vec<IndexMetadata> = vec![];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn test_retain_all_indices_removed() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_test_index("vec1", 1, 1, Some(RoaringBitmap::new()), true),
            create_test_index("vec2", 1, 1, Some(RoaringBitmap::new()), true),
            create_test_index("idx3", 99, 1, Some(RoaringBitmap::from_iter([1])), false), // Bad field
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Only the bad-field index is dropped; the empty vector definitions
        // are retained.
        assert_eq!(indices.len(), 2);
        assert!(!indices.iter().any(|idx| idx.name == "idx3"));
    }

    #[test]
    fn test_retain_complex_scenario() {
        let schema = create_test_schema(&[1, 2]);
        let fragments = vec![Fragment::new(1), Fragment::new(2)];

        let mut indices = vec![
            // System index - should always be kept
            create_system_index(FRAG_REUSE_INDEX_NAME, 1),
            // Group "idx_a" - all empty scalars, keep oldest
            create_test_index("idx_a", 1, 3, Some(RoaringBitmap::new()), false),
            create_test_index("idx_a", 1, 1, Some(RoaringBitmap::new()), false), // Oldest
            create_test_index("idx_a", 1, 2, Some(RoaringBitmap::new()), false),
            // Group "vec_b" - all empty vectors, keep oldest definition
            create_test_index("vec_b", 1, 1, Some(RoaringBitmap::new()), true),
            create_test_index("vec_b", 1, 2, Some(RoaringBitmap::new()), true),
            // Group "idx_c" - mixed empty/non-empty, keep non-empty
            create_test_index("idx_c", 2, 1, Some(RoaringBitmap::new()), false),
            create_test_index("idx_c", 2, 2, Some(RoaringBitmap::from_iter([1])), false), // Keep
            create_test_index("idx_c", 2, 3, Some(RoaringBitmap::from_iter([2])), false), // Keep
            // Single non-empty - keep
            create_test_index("idx_d", 1, 1, Some(RoaringBitmap::from_iter([1, 2])), false),
            // Index with bad field - remove
            create_test_index("idx_e", 99, 1, Some(RoaringBitmap::from_iter([1])), false),
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Expected: frag_reuse, idx_a (oldest), vec_b (oldest), idx_c (2
        // non-empty), idx_d = 6 total
        assert_eq!(indices.len(), 6);

        // Verify system index kept
        assert!(indices.iter().any(|idx| idx.name == FRAG_REUSE_INDEX_NAME));

        // Verify idx_a kept oldest only
        let idx_a_indices: Vec<_> = indices.iter().filter(|idx| idx.name == "idx_a").collect();
        assert_eq!(idx_a_indices.len(), 1);
        assert_eq!(idx_a_indices[0].dataset_version, 1);

        // Verify vec_b kept oldest definition only
        let vec_b_indices: Vec<_> = indices.iter().filter(|idx| idx.name == "vec_b").collect();
        assert_eq!(vec_b_indices.len(), 1);
        assert_eq!(vec_b_indices[0].dataset_version, 1);

        // Verify idx_c kept non-empty only
        let idx_c_indices: Vec<_> = indices.iter().filter(|idx| idx.name == "idx_c").collect();
        assert_eq!(idx_c_indices.len(), 2);
        assert!(
            idx_c_indices
                .iter()
                .all(|idx| idx.dataset_version == 2 || idx.dataset_version == 3)
        );

        // Verify idx_d kept
        assert!(indices.iter().any(|idx| idx.name == "idx_d"));

        // Verify idx_e removed (bad field)
        assert!(!indices.iter().any(|idx| idx.name == "idx_e"));
    }

    #[test]
    fn test_handle_rewrite_indices_skips_missing_index() {
        // Create an empty indices list
        let mut indices = vec![];

        // Create rewritten_indices referring to a non-existent index
        let rewritten_indices = vec![RewrittenIndex {
            old_id: Uuid::new_v4(),
            new_id: Uuid::new_v4(),
            new_index_details: prost_types::Any {
                type_url: String::new(),
                value: vec![],
            },
            new_index_version: 1,
            new_index_files: None,
        }];

        // Should succeed (skip missing index) instead of error
        let result = Transaction::handle_rewrite_indices(&mut indices, &rewritten_indices, &[]);
        assert!(result.is_ok());
        assert!(indices.is_empty());
    }

    /// A covered column whose *only* change is field metadata must be rejected here.
    ///
    /// The read path intersects a declaration against each segment's physical columns with
    /// `covering_fields_match`, which compares name, type, nullability AND metadata. If
    /// this guard compared any less, a metadata-only edit would commit cleanly and then
    /// silently withdraw covering at query time -- exactly the silent degradation the
    /// guard is here to convert into an error.
    #[test]
    fn test_covered_field_subtree_change_notices_metadata_only_edit() {
        let with_metadata = |value: Option<&str>| {
            let mut field = ArrowField::new("carried", DataType::Int32, false);
            if let Some(value) = value {
                field = field.with_metadata(
                    [("unit".to_string(), value.to_string())]
                        .into_iter()
                        .collect(),
                );
            }
            LanceSchema::try_from(&ArrowSchema::new(vec![
                ArrowField::new("key", DataType::Int32, false),
                field,
            ]))
            .unwrap()
        };

        let old_schema = with_metadata(Some("metres"));
        let carried_id = old_schema.field("carried").unwrap().id;

        assert!(
            Transaction::covered_field_subtree_changed(
                &old_schema,
                &with_metadata(Some("feet")),
                carried_id
            ),
            "a changed metadata value must count as a covering change"
        );
        assert!(
            Transaction::covered_field_subtree_changed(
                &old_schema,
                &with_metadata(None),
                carried_id
            ),
            "dropping metadata entirely must count as a covering change"
        );
        assert!(
            !Transaction::covered_field_subtree_changed(
                &old_schema,
                &with_metadata(Some("metres")),
                carried_id
            ),
            "an unchanged field must not be reported as changed"
        );
    }

    #[test]
    fn test_prune_overlay_stale_fields_from_indices() {
        // Fragment 0 carried an overlay on field 1 committed at v5, and was
        // fully compacted into new fragment 7.
        let mut old_frag = Fragment::new(0);
        old_frag.overlays = vec![overlay_with_field(1, 5)];
        let groups = vec![RewriteGroup {
            old_fragments: vec![old_frag],
            new_fragments: vec![Fragment::new(7)],
        }];

        // Post-remap state: every index already covers the new fragment (7).
        let covering = || Some(RoaringBitmap::from_iter([7u32]));
        // Indexes an un-overlaid field (3) but *covers* the overlaid field 1 as an
        // included column; built (v2) before the overlay, so its stored copy of
        // field 1 is stale and the rewritten fragment must be dropped.
        let mut covered_stale = create_test_index("covered_stale", 3, 2, covering(), false);
        // `covering_fields` is always the trailing entries of `fields`.
        covered_stale.fields = vec![3, 1];
        covered_stale.covering_fields = vec![1];
        let mut indices = vec![
            // Stale: covers the overlaid field 1, built (v2) before the overlay.
            create_test_index("stale", 1, 2, covering(), false),
            // Not stale: covers field 1 but built at the overlay's version (v5);
            // `committed_version > dataset_version` is false at equality.
            create_test_index("fresh", 1, 5, covering(), false),
            // Unrelated: covers field 2, which the overlay never touched.
            create_test_index("unrelated", 2, 2, covering(), false),
            covered_stale,
        ];

        // Flat field ids with no struct nesting: an empty schema makes subtree
        // expansion a no-op (each id maps to itself), exercising the id-level logic.
        Transaction::prune_overlay_stale_fields_from_indices(
            &mut indices,
            &groups,
            &LanceSchema::default(),
        );

        assert!(
            !indices[0].fragment_bitmap.as_ref().unwrap().contains(7),
            "stale index must drop the rewritten fragment from its coverage"
        );
        assert!(
            indices[1].fragment_bitmap.as_ref().unwrap().contains(7),
            "an index built at/after the overlay is not stale"
        );
        assert!(
            indices[2].fragment_bitmap.as_ref().unwrap().contains(7),
            "an index on an un-overlaid field is unaffected"
        );
        assert!(
            !indices[3].fragment_bitmap.as_ref().unwrap().contains(7),
            "an index whose *covered* field was overlaid is stale too"
        );
    }

    /// `covering_fields` records a covered struct's parent id, but a modified data file
    /// lists leaf ids. Schema-aware pruning must expand the covered struct to its subtree
    /// so an update to a subfield still invalidates the fragment's coverage.
    #[test]
    fn test_prune_updated_fields_expands_covered_struct_subtree() {
        use arrow_schema::Fields as ArrowFields;

        let arrow = ArrowSchema::new(vec![
            ArrowField::new(
                "s",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("a", DataType::Int32, true),
                    ArrowField::new("b", DataType::Int32, true),
                ])),
                true,
            ),
            ArrowField::new("x", DataType::Int32, true),
        ]);
        let schema = LanceSchema::try_from(&arrow).unwrap();
        let s_id = schema.field("s").unwrap().id;
        let a_id = u32::try_from(schema.field("s.a").unwrap().id).unwrap();
        let x_id = schema.field("x").unwrap().id;

        // Index keyed on scalar `x`, covering the whole struct `s`, over fragment 0.
        let mut idx = create_test_index(
            "cov",
            x_id,
            1,
            Some(RoaringBitmap::from_iter([0u32])),
            false,
        );
        // `covering_fields` is always the trailing entries of `fields`.
        idx.fields = vec![x_id, s_id];
        idx.covering_fields = vec![s_id];

        // Updating leaf `s.a` prunes the covered struct's fragment: the subtree
        // expansion bridges the parent-id-vs-leaf-id mismatch that exact-id matching
        // would miss.
        let mut indices = vec![idx];
        Transaction::prune_updated_fields_from_indices(
            &mut indices,
            &[Fragment::new(0)],
            &[a_id],
            &schema,
        );
        assert!(
            !indices[0].fragment_bitmap.as_ref().unwrap().contains(0),
            "updating a covered struct's subfield must drop the fragment from coverage"
        );
    }

    /// A Project that drops or renames a covered (included) column must be rejected at
    /// the commit boundary, even when the index's key column survives.
    #[test]
    fn test_reject_projected_covered_fields() {
        let arrow = ArrowSchema::new(vec![
            ArrowField::new("vec", DataType::Int32, false),
            ArrowField::new("meta", DataType::Utf8, true),
            ArrowField::new("other", DataType::Int32, true),
        ]);
        let old_schema = LanceSchema::try_from(&arrow).unwrap();
        let vec_id = old_schema.field("vec").unwrap().id;
        let meta_id = old_schema.field("meta").unwrap().id;

        // Index keyed on `vec`, covering `meta`.
        let mut idx = create_test_index(
            "cov",
            vec_id,
            1,
            Some(RoaringBitmap::from_iter([0u32])),
            true,
        );
        // `covering_fields` is always the trailing entries of `fields`.
        idx.fields = vec![vec_id, meta_id];
        idx.covering_fields = vec![meta_id];
        let indices = vec![idx];

        // Keeping every field -> Ok.
        assert!(
            Transaction::reject_covered_field_subtree_change(
                &indices,
                Some(&old_schema),
                &old_schema,
                "Project"
            )
            .is_ok()
        );

        // Dropping the covered `meta` while keeping the key -> rejected.
        let dropped = old_schema.project(&["vec", "other"]).unwrap();
        let err = Transaction::reject_covered_field_subtree_change(
            &indices,
            Some(&old_schema),
            &dropped,
            "Project",
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("drop or alter covered"),
            "expected a covered-subtree rejection, got: {err}"
        );

        // Dropping the key too (the index would be dropped) -> not our concern.
        let drop_all = old_schema.project(&["other"]).unwrap();
        assert!(
            Transaction::reject_covered_field_subtree_change(
                &indices,
                Some(&old_schema),
                &drop_all,
                "Project"
            )
            .is_ok()
        );

        // Renaming the covered field (id stays, name changes) -> rejected.
        let mut renamed = old_schema.clone();
        renamed.field_by_id_mut(meta_id).unwrap().name = "meta_renamed".to_string();
        assert!(
            Transaction::reject_covered_field_subtree_change(
                &indices,
                Some(&old_schema),
                &renamed,
                "Project"
            )
            .is_err(),
            "renaming a covered field must be rejected"
        );

        // Flipping the covered field's nullability (id + name stay) -> rejected.
        let mut retyped_null = old_schema.clone();
        retyped_null.field_by_id_mut(meta_id).unwrap().nullable = false;
        assert!(
            Transaction::reject_covered_field_subtree_change(
                &indices,
                Some(&old_schema),
                &retyped_null,
                "Project"
            )
            .is_err(),
            "a covered field nullability change must be rejected"
        );
    }

    /// Growing a covered *struct*'s subtree (adding a child) is what `add_columns`/Merge
    /// does; the covered field id is unchanged but its `data_type()` differs, so the commit
    /// boundary must reject it just like the Project drop/alter cases above.
    #[test]
    fn test_reject_covered_struct_child_add() {
        let child = ArrowField::new("a", DataType::Int32, false);
        let arrow = ArrowSchema::new(vec![
            ArrowField::new("vec", DataType::Int32, false),
            ArrowField::new("meta", DataType::Struct(vec![child.clone()].into()), true),
        ]);
        let old_schema = LanceSchema::try_from(&arrow).unwrap();
        let vec_id = old_schema.field("vec").unwrap().id;
        let meta_id = old_schema.field("meta").unwrap().id;

        let mut idx = create_test_index(
            "cov",
            vec_id,
            1,
            Some(RoaringBitmap::from_iter([0u32])),
            true,
        );
        // `covering_fields` is always the trailing entries of `fields`.
        idx.fields = vec![vec_id, meta_id];
        idx.covering_fields = vec![meta_id]; // covers the struct's parent id
        let indices = vec![idx];

        // New schema grows `meta` with a second child -> the struct's data_type changes.
        let grown_arrow = ArrowSchema::new(vec![
            ArrowField::new("vec", DataType::Int32, false),
            ArrowField::new(
                "meta",
                DataType::Struct(vec![child, ArrowField::new("b", DataType::Int32, true)].into()),
                true,
            ),
        ]);
        // Preorder id assignment gives `meta` the same id (1) in both schemas, so the
        // covered id resolves to the struct on each side and only its data_type differs.
        let grown = LanceSchema::try_from(&grown_arrow).unwrap();
        assert_eq!(grown.field("meta").unwrap().id, meta_id);

        let err = Transaction::reject_covered_field_subtree_change(
            &indices,
            Some(&old_schema),
            &grown,
            "add_columns (Merge)",
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("drop or alter covered")
                && err.to_string().contains("add_columns"),
            "expected a covered-struct-growth rejection, got: {err}"
        );
    }

    // Helper functions for retain_relevant_indices tests
    fn create_test_index(
        name: &str,
        field_id: i32,
        dataset_version: u64,
        fragment_bitmap: Option<RoaringBitmap>,
        is_vector: bool,
    ) -> IndexMetadata {
        use prost_types::Any;
        use std::sync::Arc;

        let index_details = if is_vector {
            Some(Arc::new(Any {
                type_url: "type.googleapis.com/lance.index.VectorIndexDetails".to_string(),
                value: vec![],
            }))
        } else {
            Some(Arc::new(Any {
                type_url: "type.googleapis.com/lance.index.ScalarIndexDetails".to_string(),
                value: vec![],
            }))
        };

        IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: vec![field_id],
            covering_fields: vec![],
            name: name.to_string(),
            dataset_version,
            fragment_bitmap,
            index_details,
            index_version: 1,
            created_at: None,
            base_id: None,
            files: None,
        }
    }

    fn create_system_index(name: &str, field_id: i32) -> IndexMetadata {
        use prost_types::Any;
        use std::sync::Arc;

        IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: vec![field_id],
            covering_fields: vec![],
            name: name.to_string(),
            dataset_version: 1,
            fragment_bitmap: Some(RoaringBitmap::from_iter([1, 2])),
            index_details: Some(Arc::new(Any {
                type_url: "type.googleapis.com/lance.index.SystemIndexDetails".to_string(),
                value: vec![],
            })),
            index_version: 1,
            created_at: None,
            base_id: None,
            files: None,
        }
    }

    fn create_test_schema(field_ids: &[i32]) -> Schema {
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use lance_core::datatypes::Schema as LanceSchema;

        let fields: Vec<ArrowField> = field_ids
            .iter()
            .map(|id| ArrowField::new(format!("field_{}", id), DataType::Int32, false))
            .collect();

        let arrow_schema = ArrowSchema::new(fields);
        let mut lance_schema = LanceSchema::try_from(&arrow_schema).unwrap();

        // Assign field IDs
        for (i, field_id) in field_ids.iter().enumerate() {
            lance_schema.mut_field_by_id(i as i32).unwrap().id = *field_id;
        }

        lance_schema
    }
}
