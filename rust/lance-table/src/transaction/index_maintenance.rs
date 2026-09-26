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
use crate::system_index::frag_reuse::lineage::TaggedLineage;
use crate::system_index::frag_reuse::metadata::is_tagged;
use crate::system_index::is_system_index;
use crate::transaction::{
    DataReplacementGroup, Operation, RewriteGroup, RewrittenIndex, Transaction,
};
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

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

        let value_updated_field_set = fields_for_preserving_frag_bitmap
            .iter()
            .collect::<HashSet<_>>();

        for index in indices.iter_mut() {
            // Physical row addresses cannot follow moved rows into a new fragment.
            // Leave that fragment uncovered so the scanner reads it directly.
            if index.results_are_row_addrs() {
                continue;
            }
            let index_covers_modified_field = index.fields.iter().any(|field_id| {
                value_updated_field_set.contains(&u32::try_from(*field_id).unwrap())
            });
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

    /// Withdraw what an in-place column rewrite invalidates from every index
    /// on a table with a tagged fragment reuse history.
    ///
    /// The physical columns `operation` rewrites in place, per live fragment
    /// id, unexpanded: an update with `fields_modified` rewrites those
    /// fields in every updated fragment; a merge rewrites, in each fragment
    /// present in `previous_fragments`, the fields whose backing data file
    /// changed (`merge_rewritten_fields`); a data replacement rewrites the
    /// fields its new files carry, read through `schema`. Any other
    /// operation rewrites nothing. `previous_fragments` is the caller's
    /// "before" list: the current manifest's for a commit, the read
    /// version's for a rebase.
    pub fn rewritten_physical_columns(
        operation: &Operation,
        schema: &Schema,
        previous_fragments: &[Fragment],
    ) -> Vec<(u64, Vec<u32>)> {
        match operation {
            Operation::Update {
                updated_fragments,
                fields_modified,
                ..
            } if !fields_modified.is_empty() => updated_fragments
                .iter()
                .map(|fragment| (fragment.id, fields_modified.clone()))
                .collect(),
            Operation::Merge { fragments, .. } => {
                Self::merge_rewritten_fields(previous_fragments, fragments)
            }
            Operation::DataReplacement { replacements } => replacements
                .iter()
                .map(|DataReplacementGroup(fragment_id, new_file)| {
                    let mut fields: Vec<u32> = new_file
                        .schema(schema)
                        .field_ids()
                        .into_iter()
                        .chain(new_file.fields.iter().copied())
                        .filter_map(|id| u32::try_from(id).ok())
                        .collect();
                    fields.sort_unstable();
                    fields.dedup();
                    (*fragment_id, fields)
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The one implementation of an in-place column rewrite's effect on
    /// index coverage, shared by the commit preparation (on the list read
    /// from the current manifest) and by the conflict resolver rebasing a
    /// new index over a committed rewrite (on the index being created).
    ///
    /// `rewrites` comes from [`Self::rewritten_physical_columns`]. With a
    /// walkable tagged history (`lineage`), every rewritten field id is
    /// expanded to its descendants through `schema` (a packed struct is one
    /// physical column while an index on `s.x` records the child's id) and
    /// the coverage is withdrawn along the lineage
    /// ([`Self::withdraw_rewritten_coverage`]). Without a tagged history the
    /// rewritten fragments are pruned from the covering indices by id, the
    /// ordinary rule, exactly as before.
    pub fn withdraw_in_place_rewrites(
        indices: &mut Vec<IndexMetadata>,
        rewrites: Vec<(u64, Vec<u32>)>,
        schema: &Schema,
        live: &RoaringBitmap,
        lineage: Option<&TaggedLineage>,
    ) {
        if rewrites.is_empty() {
            return;
        }
        match lineage {
            Some(lineage) => {
                let rewrites = Self::expand_rewritten_fields(schema, rewrites);
                Self::withdraw_rewritten_coverage(indices, Some(lineage), live, &rewrites);
            }
            None => {
                for (fragment, fields) in rewrites {
                    Self::prune_updated_fields_from_indices(
                        indices,
                        &[Fragment::new(fragment)],
                        &fields,
                    );
                }
            }
        }
    }

    /// Expand every rewritten field id in `rewrites` to the field and all
    /// its descendants through `schema`, the form
    /// `withdraw_rewritten_coverage` compares against index fields.
    fn expand_rewritten_fields(
        schema: &Schema,
        rewrites: Vec<(u64, Vec<u32>)>,
    ) -> Vec<(u64, Vec<u32>)> {
        rewrites
            .into_iter()
            .map(|(fragment, fields)| (fragment, Self::with_descendants(schema, fields)))
            .collect()
    }

    /// `fields` plus every field nested under them, deduplicated.
    fn with_descendants(schema: &Schema, fields: Vec<u32>) -> Vec<u32> {
        let mut expanded = Vec::with_capacity(fields.len());
        for id in fields {
            expanded.push(id);
            let Ok(field_id) = i32::try_from(id) else {
                continue;
            };
            let mut pending: Vec<&lance_core::datatypes::Field> = schema
                .field_by_id(field_id)
                .map(|field| field.children.iter().collect())
                .unwrap_or_default();
            while let Some(child) = pending.pop() {
                if let Ok(child_id) = u32::try_from(child.id) {
                    expanded.push(child_id);
                }
                pending.extend(child.children.iter());
            }
        }
        expanded.sort_unstable();
        expanded.dedup();
        expanded
    }

    /// `rewrites` lists each rewritten live fragment with the fields
    /// rewritten in it (expanded to their descendants, see
    /// [`Self::withdraw_in_place_rewrites`]). For every
    /// non-system segment that indexes one of those fields (keyed or
    /// carried), with bitmap `B`:
    /// - a rewritten fragment named in `B` is removed, as on an untagged
    ///   table;
    /// - every retired fragment `r` of `B` (not in `live`) whose rows the
    ///   history moves into the rewritten fragment (`destinations_of`,
    ///   transitively, so mixed provenance holding intermediates is
    ///   covered) is removed. The sources of one transition all reach the
    ///   same destinations, so the transition's sources go whole: the
    ///   segment derives no coverage of its destinations and those rows are
    ///   scanned until `optimize_indices` rebuilds them;
    /// - without a walkable history every retired fragment of `B` goes;
    /// - a segment without a bitmap is untouched: its coverage is unknown,
    ///   not empty, and the reader already excludes it.
    ///
    /// What a withdrawal leaves is then settled per logical index name:
    /// - coverage partly left: the segment stays and serves what is left,
    ///   provided the reader will still translate it. The file keeps the
    ///   withdrawn rows; the translating load drops them (they resolve
    ///   outside the derived coverage), but a segment whose remaining
    ///   bitmap is live-only and disjoint from the history is loaded as it
    ///   is, and would serve the withdrawn retired rows raw. So when a
    ///   RETIRED fragment was withdrawn and the remainder would be loaded
    ///   as it is, the remainder is withdrawn too (the segment empties and
    ///   is rebuilt). Withdrawing only live fragments leaves the remainder
    ///   alone: rows of fragments that still exist are masked by the
    ///   ordinary per-segment ownership filter;
    /// - coverage emptied while a same-name sibling still has coverage (or
    ///   an unknown one): the segment's metadata leaves the manifest in this
    ///   commit; its files are reclaimed by the ordinary index cleanup once
    ///   no retained version references them;
    /// - coverage emptied and it is the last segment of its name: it stays
    ///   as the record of what index to build (queries never use it, since
    ///   the reader lists nothing for it; maintenance rebuilds it from the
    ///   live fragments).
    ///
    /// Every withdrawal is logged at warn level.
    pub fn withdraw_rewritten_coverage(
        indices: &mut Vec<IndexMetadata>,
        lineage: Option<&TaggedLineage>,
        live: &RoaringBitmap,
        rewrites: &[(u64, Vec<u32>)],
    ) {
        let mut emptied = Vec::new();
        for index in indices.iter_mut().filter(|index| !is_system_index(index)) {
            let indexed: HashSet<u32> = index
                .fields
                .iter()
                .filter_map(|field| u32::try_from(*field).ok())
                .collect();
            let Some(bitmap) = index.fragment_bitmap.as_mut() else {
                continue;
            };
            let retired = &*bitmap - live;
            let mut withdrawn = RoaringBitmap::new();
            for (fragment, fields) in rewrites {
                if !fields.iter().any(|field| indexed.contains(field)) {
                    continue;
                }
                let Ok(rewritten) = u32::try_from(*fragment) else {
                    continue;
                };
                if bitmap.contains(rewritten) {
                    withdrawn.insert(rewritten);
                }
                // Every retired source whose rows reach the rewritten
                // fragment goes; the sources of one transition all reach the
                // same destinations, so the transition is withdrawn whole.
                for source in retired.iter() {
                    match lineage.and_then(|lineage| lineage.destinations_of(source)) {
                        Some(destinations) if destinations.contains(&rewritten) => {
                            withdrawn.insert(source);
                        }
                        Some(_) => {}
                        None => {
                            withdrawn.insert(source);
                        }
                    }
                }
            }
            if withdrawn.is_empty() {
                continue;
            }
            *bitmap -= &withdrawn;
            let withdrew_retired = !(&withdrawn & &retired).is_empty();
            if withdrew_retired && !bitmap.is_empty() {
                // Would the reader still translate what is left? Only a
                // bitmap naming a retired fragment or one the history
                // mentions is translated (and filtered to its derived
                // coverage); a live-only remainder the history does not
                // mention is loaded as it is, which would serve the
                // withdrawn rows the file still holds. A history that
                // cannot be walked holds transitions this build cannot
                // read, and the reader then translates every segment, so
                // the remainder is safe in that case.
                let reader_translates = !bitmap.is_subset(live)
                    || bitmap.iter().any(|fragment| {
                        lineage
                            .and_then(|lineage| lineage.mentions(fragment))
                            .unwrap_or(true)
                    });
                if !reader_translates {
                    log::warn!(
                        "index {} (segment {}): its remaining coverage {:?} would be loaded \
                         as it is while its file still holds rows of the withdrawn retired \
                         fragments; withdrawing it whole so the segment is rebuilt",
                        index.name,
                        index.uuid,
                        bitmap.iter().collect::<Vec<_>>()
                    );
                    withdrawn |= &*bitmap;
                    bitmap.clear();
                }
            }
            if bitmap.is_empty() {
                emptied.push(index.uuid);
            }
            log::warn!(
                "index {} (segment {}): withdrew {} fragment(s) of coverage after an in-place \
                 rewrite of an indexed column; the rows they covered are scanned until the index \
                 is optimized",
                index.name,
                index.uuid,
                withdrawn.len()
            );
        }
        Self::retire_emptied_segments(indices, &emptied);
    }

    /// Drop the metadata of every segment in `emptied` whose logical index
    /// still has another segment with coverage (or with unknown coverage).
    /// The last segment of a name is kept, empty, as the record of what to
    /// rebuild.
    fn retire_emptied_segments(indices: &mut Vec<IndexMetadata>, emptied: &[Uuid]) {
        let removable: Vec<Uuid> = emptied
            .iter()
            .copied()
            .filter(|uuid| {
                let Some(name) = indices
                    .iter()
                    .find(|index| index.uuid == *uuid)
                    .map(|index| index.name.as_str())
                else {
                    return false;
                };
                indices.iter().any(|sibling| {
                    sibling.name == name
                        && sibling.uuid != *uuid
                        && sibling
                            .fragment_bitmap
                            .as_ref()
                            .is_none_or(|bitmap| !bitmap.is_empty())
                })
            })
            .collect();
        if removable.is_empty() {
            return;
        }
        indices.retain(|index| {
            let retired = removable.contains(&index.uuid);
            if retired {
                log::info!(
                    "index {} (segment {}): every fragment of its coverage was withdrawn and a \
                     sibling segment still serves the index; the segment leaves the manifest and \
                     its files are reclaimed by index cleanup",
                    index.name,
                    index.uuid
                );
            }
            !retired
        });
    }

    /// If an operation modifies one or more fields in a fragment then we need to remove
    /// that fragment from any indices that cover one of the modified fields.
    pub fn prune_updated_fields_from_indices(
        indices: &mut [IndexMetadata],
        updated_fragments: &[Fragment],
        fields_modified: &[u32],
    ) {
        if fields_modified.is_empty() {
            return;
        }

        // If we modified any fields in the fragments then we need to remove those fragments
        // from the index if the index covers one of those modified fields.
        let fields_modified_set = fields_modified.iter().collect::<HashSet<_>>();
        for index in indices.iter_mut() {
            if index
                .fields
                .iter()
                .any(|field_id| fields_modified_set.contains(&u32::try_from(*field_id).unwrap()))
                && let Some(fragment_bitmap) = &mut index.fragment_bitmap
            {
                for fragment_id in updated_fragments.iter().map(|f| f.id as u32) {
                    fragment_bitmap.remove(fragment_id);
                }
            }
        }
    }

    /// Map each (non-tombstoned) field id in a fragment to the path of the data
    /// file that backs it.
    fn fragment_field_paths(frag: &Fragment) -> HashMap<i32, &str> {
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

    /// The columns a `Merge` rewrote in place: for each fragment present in
    /// both lists, the fields still present whose backing data file path
    /// changed. Brand-new fragments carry nothing stale.
    pub fn merge_rewritten_fields(
        prev_fragments: &[Fragment],
        new_fragments: &[Fragment],
    ) -> Vec<(u64, Vec<u32>)> {
        let prev_by_id: HashMap<u64, &Fragment> =
            prev_fragments.iter().map(|f| (f.id, f)).collect();
        new_fragments
            .iter()
            .filter_map(|new_frag| {
                let prev = prev_by_id.get(&new_frag.id)?;
                let prev_paths = Self::fragment_field_paths(prev);
                let new_paths = Self::fragment_field_paths(new_frag);
                let mut changed: Vec<u32> = prev_paths
                    .iter()
                    .filter(|(field_id, prev_path)| {
                        new_paths
                            .get(*field_id)
                            .is_some_and(|new_path| new_path != *prev_path)
                    })
                    .map(|(field_id, _)| *field_id as u32)
                    .collect();
                changed.sort_unstable();
                (!changed.is_empty()).then_some((new_frag.id, changed))
            })
            .collect()
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
    ) {
        for group in groups {
            // field id -> newest overlay committed_version supplying that field
            let mut overlaid_field_versions: HashMap<i32, u64> = HashMap::new();
            for old_frag in &group.old_fragments {
                for overlay in &old_frag.overlays {
                    for &field_id in overlay.data_file.fields.iter() {
                        if field_id < 0 {
                            // Tombstoned (obsolete) overlay field: supplies nothing.
                            continue;
                        }
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
            for index in indices.iter_mut() {
                let is_stale = index.fields.iter().any(|field_id| {
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

        // Under a tagged fragment reuse history a segment's stored bitmap is
        // its provenance (the retired source fragments it was built from),
        // not the fragments it serves: its live coverage is derived at query
        // time by translating through the history, and is empty against the
        // live fragments by construction once its sources were rewritten.
        // Measuring such a segment here would drop it. Every segment is kept
        // instead; superseded segments are pruned by the tagged maintenance
        // path, which reasons on derived coverage.
        let tagged = indices.iter().any(is_tagged);

        for (_, same_name_indices) in indices_by_name {
            if tagged {
                for index in same_name_indices {
                    uuids_to_keep.insert(index.uuid);
                }
                continue;
            }
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

    /// The rewrite groups whose index bitmaps follow the rewrite (source ids
    /// swapped for destination ids). Groups covered by the stable-partition
    /// transitions' sources are excluded: they redistribute rows, so their
    /// bitmaps keep the retired source ids as provenance and the tagged
    /// fragment reuse index entry records the row-level translation. A group
    /// must be entirely reordered or entirely order-preserving.
    pub(super) fn ordered_rewrite_groups(
        groups: &[RewriteGroup],
        reordered_sources: Option<&RoaringBitmap>,
    ) -> Result<Vec<RewriteGroup>> {
        let Some(sources) = reordered_sources else {
            return Ok(groups.to_vec());
        };
        let mut ordered = Vec::new();
        for group in groups {
            let covered = group
                .old_fragments
                .iter()
                .filter(|frag| sources.contains(frag.id as u32))
                .count();
            if covered == 0 {
                ordered.push(group.clone());
            } else if covered != group.old_fragments.len() {
                return Err(Error::invalid_input(
                    "a rewrite group mixes transition-covered and order-preserving source fragments",
                ));
            }
        }
        Ok(ordered)
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
    use uuid::Uuid;

    #[test]
    fn test_ordered_rewrite_groups_split_and_mixed() {
        use crate::format::pb::fragment_reuse_index_details as pb_fri;
        let group = |old_ids: &[u64], new_ids: &[u64]| RewriteGroup {
            old_fragments: old_ids.iter().map(|&id| Fragment::new(id)).collect(),
            new_fragments: new_ids.iter().map(|&id| Fragment::new(id)).collect(),
        };
        let groups = vec![group(&[0, 1], &[10]), group(&[2], &[11])];

        // No stable partition: every group takes part in bitmap maintenance.
        assert_eq!(
            Transaction::ordered_rewrite_groups(&groups, None)
                .unwrap()
                .len(),
            2
        );

        // Group [0, 1] is reordered: only group [2] remains ordered.
        let digest = |id: u64| pb_fri::FragmentDigest {
            id,
            physical_rows: 4,
            num_deleted_rows: 0,
        };
        let reordered = crate::transaction::reordered_sources(&[pb_fri::Transition {
            sources: vec![digest(0), digest(1)],
            destinations: vec![digest(10), digest(10)],
            mapping: None,
        }])
        .unwrap();
        let ordered = Transaction::ordered_rewrite_groups(&groups, Some(&reordered)).unwrap();
        assert_eq!(ordered.len(), 1);
        assert_eq!(ordered[0].old_fragments[0].id, 2);

        // A group straddling reordered and order-preserving sources is
        // rejected.
        let mixed = vec![group(&[1, 2], &[10])];
        assert!(Transaction::ordered_rewrite_groups(&mixed, Some(&reordered)).is_err());
    }

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
            legacy_frag_reuse_index(99), // Field doesn't exist but should be kept
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
            legacy_frag_reuse_index(1),
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
            legacy_frag_reuse_index(1),
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
        let mut indices = vec![
            // Stale: covers the overlaid field 1, built (v2) before the overlay.
            create_test_index("stale", 1, 2, covering(), false),
            // Not stale: covers field 1 but built at the overlay's version (v5);
            // `committed_version > dataset_version` is false at equality.
            create_test_index("fresh", 1, 5, covering(), false),
            // Unrelated: covers field 2, which the overlay never touched.
            create_test_index("unrelated", 2, 2, covering(), false),
        ];

        Transaction::prune_overlay_stale_fields_from_indices(&mut indices, &groups);

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

    /// A version-0 fragment reuse entry: the pre-tagged history whose
    /// presence leaves coverage-based retention as it always was.
    fn legacy_frag_reuse_index(field_id: i32) -> IndexMetadata {
        let mut index = create_system_index(FRAG_REUSE_INDEX_NAME, field_id);
        index.index_version = 0;
        index
    }

    /// Under a tagged history a translating segment's stored bitmap is its
    /// retired provenance, empty against the live fragments by construction,
    /// so retention must keep every segment: only a segment on a field that
    /// left the schema is dropped.
    #[test]
    fn test_retain_keeps_every_segment_under_tagged_history() {
        let schema = create_test_schema(&[1, 2]);
        let fragments = vec![Fragment::new(10), Fragment::new(11)];
        let mut indices = vec![
            create_system_index(FRAG_REUSE_INDEX_NAME, 1),
            // Two translating segments: provenance {0} and {1} are both dead.
            create_test_index("idx_a", 1, 1, Some(RoaringBitmap::from_iter([0])), false),
            create_test_index("idx_a", 1, 2, Some(RoaringBitmap::from_iter([1])), false),
            // A rebuilt sibling directly covering a live destination.
            create_test_index("idx_a", 1, 3, Some(RoaringBitmap::from_iter([10])), false),
            // An all-dead group and an all-empty group: every member kept.
            create_test_index("vec_b", 1, 1, Some(RoaringBitmap::from_iter([0])), true),
            create_test_index("vec_b", 1, 2, Some(RoaringBitmap::from_iter([1])), true),
            create_test_index("idx_c", 2, 1, Some(RoaringBitmap::new()), false),
            create_test_index("idx_c", 2, 2, Some(RoaringBitmap::new()), false),
            // Field left the schema: dropped as always.
            create_test_index("idx_e", 99, 1, Some(RoaringBitmap::from_iter([10])), false),
        ];
        assert!(indices.iter().any(is_tagged));

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        assert_eq!(indices.len(), 8);
        assert!(!indices.iter().any(|idx| idx.name == "idx_e"));
        assert_eq!(indices.iter().filter(|idx| idx.name == "idx_a").count(), 3);
        assert_eq!(indices.iter().filter(|idx| idx.name == "vec_b").count(), 2);
        assert_eq!(indices.iter().filter(|idx| idx.name == "idx_c").count(), 2);
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

#[cfg(test)]
mod withdrawal_tests {
    use super::*;
    use crate::transaction::test_support::{sample_index_metadata, tagged_entry};

    fn segment(bitmap: Option<&[u32]>, fields: Vec<i32>) -> IndexMetadata {
        let mut segment = sample_index_metadata("idx");
        segment.fields = fields;
        segment.fragment_bitmap = bitmap.map(|ids| ids.iter().copied().collect());
        segment
    }

    fn lineage(entry: &IndexMetadata) -> TaggedLineage {
        TaggedLineage::from_indices_with_ledger(std::slice::from_ref(entry), None)
            .unwrap()
            .unwrap()
    }

    /// F1, F2 -> F3 then F3 -> F4 (F4 live), plus an unrelated F6 -> F8.
    fn history() -> IndexMetadata {
        tagged_entry(&[(&[1, 2], &[3]), (&[3], &[4]), (&[6], &[8])])
    }

    #[test]
    fn withdraws_the_whole_transition_reaching_the_rewritten_fragment() {
        let entry = history();
        let live: RoaringBitmap = [4u32, 8, 9].into_iter().collect();
        // Provenance {1, 2} reaches F4 through F3; F6 (retired, on the
        // unrelated F6 -> F8 lineage) is untouched and keeps the segment on
        // the translating path, so it survives with {6}.
        let mut indices = vec![segment(Some(&[1, 2, 6]), vec![0]), entry.clone()];
        Transaction::withdraw_rewritten_coverage(
            &mut indices,
            Some(&lineage(&entry)),
            &live,
            &[(4, vec![0])],
        );
        assert_eq!(
            indices[0].fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([6u32])
        );
        // The entry itself is never touched.
        assert_eq!(indices[1].fragment_bitmap, entry.fragment_bitmap);
    }

    /// The file of a segment whose retired sources were withdrawn still
    /// holds their rows. A remainder that is live-only and unknown to the
    /// history would be loaded as it is and serve those rows raw, so it is
    /// withdrawn whole (the segment empties and is rebuilt).
    #[test]
    fn live_only_remainder_after_withdrawing_retired_sources_goes_whole() {
        let entry = history();
        let live: RoaringBitmap = [4u32, 8, 9].into_iter().collect();
        let mut indices = vec![segment(Some(&[1, 2, 9]), vec![0]), entry.clone()];
        Transaction::withdraw_rewritten_coverage(
            &mut indices,
            Some(&lineage(&entry)),
            &live,
            &[(4, vec![0])],
        );
        assert!(
            indices[0].fragment_bitmap.as_ref().unwrap().is_empty(),
            "{:?}",
            indices[0].fragment_bitmap
        );
    }

    /// A remainder the history mentions (F8 is a destination) is translated
    /// and filtered by the reader, so partial survival is safe there.
    #[test]
    fn remainder_the_history_mentions_survives() {
        let entry = history();
        let live: RoaringBitmap = [4u32, 8, 9].into_iter().collect();
        let mut indices = vec![segment(Some(&[1, 2, 8]), vec![0])];
        Transaction::withdraw_rewritten_coverage(
            &mut indices,
            Some(&lineage(&entry)),
            &live,
            &[(4, vec![0])],
        );
        assert_eq!(
            indices[0].fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([8u32])
        );
    }

    /// Withdrawing only a live fragment (an index built after the rewrite
    /// that produced F4) leaves the remainder alone: rows of fragments that
    /// still exist are masked by the ordinary per-segment ownership filter.
    #[test]
    fn withdrawing_only_a_live_fragment_keeps_the_remainder() {
        let entry = history();
        let live: RoaringBitmap = [4u32, 8, 9].into_iter().collect();
        let mut indices = vec![segment(Some(&[4, 9]), vec![0])];
        Transaction::withdraw_rewritten_coverage(
            &mut indices,
            Some(&lineage(&entry)),
            &live,
            &[(4, vec![0])],
        );
        assert_eq!(
            indices[0].fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([9u32])
        );
    }

    #[test]
    fn mixed_provenance_loses_the_direct_id_and_the_sources() {
        let entry = history();
        let live: RoaringBitmap = [4u32, 8].into_iter().collect();
        // {3} is an intermediate the merge kept beside the live F4.
        let mut indices = vec![segment(Some(&[3, 4]), vec![0])];
        Transaction::withdraw_rewritten_coverage(
            &mut indices,
            Some(&lineage(&entry)),
            &live,
            &[(4, vec![0])],
        );
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().is_empty());
    }

    #[test]
    fn unrelated_lineage_unindexed_field_and_missing_bitmap_are_untouched() {
        let entry = history();
        let live: RoaringBitmap = [4u32, 8].into_iter().collect();
        let mut indices = vec![
            segment(Some(&[6]), vec![0]),
            segment(Some(&[1, 2]), vec![1]),
            segment(None, vec![0]),
        ];
        Transaction::withdraw_rewritten_coverage(
            &mut indices,
            Some(&lineage(&entry)),
            &live,
            &[(4, vec![0])],
        );
        assert_eq!(
            indices[0].fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([6u32])
        );
        assert_eq!(
            indices[1].fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([1u32, 2])
        );
        assert!(indices[2].fragment_bitmap.is_none());
    }

    #[test]
    fn without_a_walkable_history_every_retired_fragment_goes() {
        let entry = history();
        let live: RoaringBitmap = [4u32, 8, 9].into_iter().collect();
        let mut indices = vec![segment(Some(&[6, 9]), vec![0])];
        Transaction::withdraw_rewritten_coverage(
            &mut indices,
            Some(&TaggedLineage::new(&entry, None)),
            &live,
            &[(4, vec![0])],
        );
        assert_eq!(
            indices[0].fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([9u32])
        );
    }
}
