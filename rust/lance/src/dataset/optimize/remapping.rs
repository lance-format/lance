// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for remapping row ids. Necessary before stable row ids.
//!

use crate::Result;
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::DatasetIndexExt;
use crate::index::frag_reuse::{
    decode_frag_reuse_ledger, load_frag_reuse_index_details, open_frag_reuse_index,
};
use crate::{Dataset, index};
use async_trait::async_trait;
use lance_core::Error;
use lance_core::utils::address::RowAddress;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_index::frag_reuse::{FRAG_REUSE_INDEX_NAME, FragDigest};
use lance_table::format::{Fragment, IndexFile, IndexMetadata};
use lance_table::io::manifest::read_manifest_indexes;
use lance_table::system_index::frag_reuse::ledger::{FragReuseLedger, Mapping};
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// The result of remapping an index
#[derive(Debug, Clone, PartialEq)]
pub enum RemapResult {
    // Index could not be remapped, drop it
    Drop,
    // No remapping is needed, keep the index as-is
    Keep(Uuid),
    // Index was remapped, return the new index
    Remapped(RemappedIndex),
}

/// A remapped index
#[derive(Debug, Clone, PartialEq)]
pub struct RemappedIndex {
    pub old_id: Uuid,
    pub new_id: Uuid,
    pub index_details: prost_types::Any,
    pub index_version: u32,
    /// List of files in the index with their sizes.
    pub files: Option<Vec<IndexFile>>,
}

/// When compaction runs the row ids will change.  This typically means that
/// indices will need to be remapped.  The details of how this happens are not
/// a part of the compaction process and so a trait is defined here to allow
/// for inversion of control.
#[async_trait]
pub trait IndexRemapper: Send + Sync {
    async fn remap_indices(
        &self,
        index_map: RowAddrRemap,
        affected_fragment_ids: &[u64],
    ) -> Result<Vec<RemappedIndex>>;
}

/// Options for creating an [IndexRemapper]
///
/// Currently we don't have any options but we may need options in the future and so we
/// want to keep a placeholder
#[async_trait]
pub trait IndexRemapperOptions: Send + Sync {
    /// Creates a remapper when the dataset has indices that need row address remapping.
    ///
    /// Returns `None` when no remappable indices exist, allowing compaction to avoid
    /// materializing an unused row address map.
    async fn create_remapper(&self, dataset: &Dataset) -> Result<Option<Box<dyn IndexRemapper>>>;
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct IgnoreRemap {}

#[async_trait]
impl IndexRemapper for IgnoreRemap {
    async fn remap_indices(&self, _: RowAddrRemap, _: &[u64]) -> Result<Vec<RemappedIndex>> {
        Ok(Vec::new())
    }
}

#[async_trait]
impl IndexRemapperOptions for IgnoreRemap {
    async fn create_remapper(&self, _: &Dataset) -> Result<Option<Box<dyn IndexRemapper>>> {
        Ok(None)
    }
}

/// Iterator that yields row_addrs that are in the given fragments but not in
/// the given row_addrs iterator.
struct MissingAddrs<'a, I: Iterator<Item = u64>> {
    row_addrs: I,
    expected_row_addr: u64,
    current_fragment_idx: usize,
    last: Option<u64>,
    fragments: &'a Vec<FragDigest>,
}

impl<'a, I: Iterator<Item = u64>> MissingAddrs<'a, I> {
    /// row_addrs must be sorted in the same order in which the rows would be
    /// found by scanning fragments in the order they are presented in.
    /// fragments is not guaranteed to be sorted by id.
    fn new(row_addrs: I, fragments: &'a Vec<FragDigest>) -> Self {
        assert!(!fragments.is_empty());
        let first_frag = &fragments[0];
        Self {
            row_addrs,
            expected_row_addr: first_frag.id * RowAddress::FRAGMENT_SIZE,
            current_fragment_idx: 0,
            last: None,
            fragments,
        }
    }
}

impl<I: Iterator<Item = u64>> Iterator for MissingAddrs<'_, I> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_fragment_idx >= self.fragments.len() {
                return None;
            }
            let val = if let Some(last) = self.last {
                self.last = None;
                last
            } else {
                // If we've exhausted row_addrs but we aren't done then use 0 which
                // is guaranteed to not match because that would mean that row_addrs
                // was empty and we check for that earlier.
                self.row_addrs.next().unwrap_or(0)
            };

            let current_fragment = &self.fragments[self.current_fragment_idx];
            let frag = val / RowAddress::FRAGMENT_SIZE;
            let expected_row_addr = self.expected_row_addr;
            self.expected_row_addr += 1;

            let current_physical_rows = current_fragment.physical_rows;
            if (self.expected_row_addr % RowAddress::FRAGMENT_SIZE) == current_physical_rows as u64
            {
                self.current_fragment_idx += 1;
                if self.current_fragment_idx < self.fragments.len() {
                    self.expected_row_addr =
                        self.fragments[self.current_fragment_idx].id * RowAddress::FRAGMENT_SIZE;
                }
            }
            if frag != current_fragment.id {
                self.last = Some(val);
                return Some(expected_row_addr);
            }
            if val != expected_row_addr {
                self.last = Some(val);
                return Some(expected_row_addr);
            }
        }
    }
}

pub fn transpose_row_addrs(
    row_addrs: RoaringTreemap,
    old_fragments: &[Fragment],
    new_fragments: &[Fragment],
) -> HashMap<u64, Option<u64>> {
    let old_frag_digests: Vec<FragDigest> = old_fragments.iter().map(|frag| frag.into()).collect();
    let new_frag_digests: Vec<FragDigest> = new_fragments.iter().map(|frag| frag.into()).collect();
    transpose_row_ids_from_digest(row_addrs, &old_frag_digests, &new_frag_digests)
}

pub fn transpose_row_ids_from_digest(
    row_addrs: RoaringTreemap,
    old_fragments: &Vec<FragDigest>,
    new_fragments: &[FragDigest],
) -> HashMap<u64, Option<u64>> {
    let new_addrs = new_fragments.iter().flat_map(|frag| {
        (0..frag.physical_rows as u32).map(|offset| {
            Some(u64::from(RowAddress::new_from_parts(
                frag.id as u32,
                offset,
            )))
        })
    });
    // The hashmap will have an entry for each row addr to map plus all rows that
    // were deleted.
    let expected_size = row_addrs.len() as usize
        + old_fragments
            .iter()
            .map(|frag| frag.num_deleted_rows)
            .sum::<usize>();
    // We expect row addrs to be unique, so we should already not get many collisions.
    // The default hasher is designed to be resistance to DoS attacks, which is
    // more than we need for this use case.
    let mut mapping: HashMap<u64, Option<u64>> = HashMap::with_capacity(expected_size);
    mapping.extend(row_addrs.iter().zip(new_addrs));
    MissingAddrs::new(row_addrs.into_iter(), old_fragments).for_each(|addr| {
        mapping.insert(addr, None);
    });
    mapping
}

/// Remap a given index using the fragment reuse index if possible.
/// If the frag reuse index does not exist, the operation fails with [Error::NotSupported]
/// If the frag reuse index exists but is empty, the operation succeeds without a commit.
async fn remap_index(dataset: &mut Dataset, index_id: &Uuid) -> Result<()> {
    let indices = dataset.load_indices().await?;
    let frag_reuse_index_meta = match indices.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME) {
        None => Err(Error::not_supported_source(
            "Fragment reuse index not found, cannot remap an index post compaction".into(),
        )),
        Some(frag_reuse_index_meta) => Ok(frag_reuse_index_meta),
    }?;

    // Hard fork by index_version: a tagged history remaps per-segment through
    // compaction-only chains in fresh code below, while the v0 path stays
    // exactly as it was.
    if frag_reuse_index_meta.index_version != 0 {
        return remap_index_tagged(dataset, index_id).await;
    }

    let frag_reuse_details = load_frag_reuse_index_details(dataset, frag_reuse_index_meta).await?;
    let frag_reuse_index =
        open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref()).await?;

    if frag_reuse_index.is_empty() {
        return Ok(());
    }

    // Read the index's on-disk metadata once. Its stored row addresses are at
    // this baseline; we compose all reuse versions into a single remap so the
    // index file is rebuilt and committed exactly once, rather than once per
    // version (the reuse index can accumulate many versions before remap runs).
    let curr_index_meta = read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await?
    .into_iter()
    .find(|idx| idx.uuid == *index_id)
    .ok_or_else(|| {
        Error::index(format!(
            "index {index_id} not found in manifest; it may have been concurrently dropped"
        ))
    })?;

    // Compose the coverage (fragment bitmap) remap across every reuse version in
    // one pass. Chaining is automatic: a version inserts its new fragments,
    // which a later version then sees as its old fragments. `data_predates_version`
    // is evaluated against the fixed baseline (there are no intermediate
    // commits), and the new-fragment branch handles a bitmap that was already
    // coverage-remapped + persisted before the data was remapped (e.g. while
    // remapping a *sibling* index).
    let baseline_version = curr_index_meta.dataset_version;
    let has_unknown_coverage = curr_index_meta.fragment_bitmap.is_none();
    let (should_remap, mut bitmap_after_remap) = match curr_index_meta.fragment_bitmap.clone() {
        Some(mut index_frag_bitmap) => {
            let mut should_remap = false;
            for version in frag_reuse_index.details.versions.iter() {
                let data_predates_version = baseline_version < version.dataset_version;
                for group in version.groups.iter() {
                    let mut old_frag_in_index = 0;
                    for old_frag in group.old_frags.iter() {
                        if index_frag_bitmap.remove(old_frag.id as u32) {
                            old_frag_in_index += 1;
                        }
                    }

                    if old_frag_in_index > 0 {
                        if old_frag_in_index != group.old_frags.len() {
                            // this should never happen because we always commit a full rewrite group
                            // and we always reindex either the entire group or nothing.
                            // We use invalid input to be consistent with
                            // dataset::transaction::recalculate_fragment_bitmap
                            return Err(Error::invalid_input(format!(
                                "The compaction plan included a rewrite group that was a split of indexed and non-indexed data: {:?}",
                                group.old_frags
                            )));
                        }
                        index_frag_bitmap.extend(group.new_frags.iter().map(|f| f.id as u32));
                        should_remap = true;
                    } else if data_predates_version
                        && group
                            .new_frags
                            .iter()
                            .any(|new_frag| index_frag_bitmap.contains(new_frag.id as u32))
                    {
                        // The bitmap was already coverage-remapped onto this
                        // group's new fragments and persisted before the data was
                        // remapped, so the old fragments are gone from the bitmap
                        // but the index data still needs remapping.
                        should_remap = true;
                    }
                }
            }
            (should_remap, Some(index_frag_bitmap))
        }
        None => (true, None),
    };

    if !should_remap {
        return Ok(());
    }

    // Apply the compact version chain directly while rebuilding the index. The
    // remapper passes intermediate moved addresses into later FRI versions and
    // leaves missing mappings unchanged, so no composed per-row map is needed.
    // This also handles the sibling-coverage-remap case: remapping is driven by
    // the row addresses stored in the index, not by its already-advanced bitmap.
    let remap_result =
        index::remap_index(dataset, index_id, frag_reuse_index.row_addr_remap()).await?;

    // Remapping advances the index watermark for fragment-reuse cleanup, but it
    // does not incorporate overlays committed after the source index was built.
    // Exclude those fragments so queries scan their current values instead.
    if let Some(fragment_bitmap) = &mut bitmap_after_remap {
        for fragment in dataset.manifest.fragments.iter() {
            let has_newer_indexed_overlay = fragment.overlays.iter().any(|overlay| {
                overlay.committed_version > curr_index_meta.dataset_version
                    && overlay
                        .data_file
                        .fields
                        .iter()
                        .any(|field_id| curr_index_meta.fields.contains(field_id))
            });
            if has_newer_indexed_overlay {
                fragment_bitmap.remove(fragment.id as u32);
            }
        }
    }
    let new_dataset_version = if has_unknown_coverage {
        curr_index_meta.dataset_version
    } else {
        dataset.manifest.version
    };

    let new_index_meta = match remap_result {
        // Nothing to commit: either the composed remap emptied the index (every
        // row deleted), matching the prior per-version behavior, or
        // `index::remap_index` withdrew a covered index it cannot carry payload
        // through. Either way the existing entry is left untouched.
        //
        // The withdrawal case is unreachable here today: the only caller is
        // `remap_column_index`, which refuses a covered index first. Compaction
        // reaches that withdrawal through `DatasetIndexRemapper`, which handles
        // `RemapResult::Drop` in `dataset/index.rs` rather than through here.
        RemapResult::Drop => return Ok(()),
        RemapResult::Keep(new_id) => IndexMetadata {
            uuid: new_id,
            name: curr_index_meta.name.clone(),
            fields: curr_index_meta.fields.clone(),
            covering_fields: curr_index_meta.covering_fields.clone(),
            dataset_version: new_dataset_version,
            fragment_bitmap: bitmap_after_remap,
            index_details: curr_index_meta.index_details.clone(),
            index_version: curr_index_meta.index_version,
            created_at: curr_index_meta.created_at,
            base_id: None,
            files: curr_index_meta.files.clone(),
        },
        RemapResult::Remapped(remapped_index) => IndexMetadata {
            uuid: remapped_index.new_id,
            name: curr_index_meta.name.clone(),
            fields: curr_index_meta.fields.clone(),
            covering_fields: curr_index_meta.covering_fields.clone(),
            dataset_version: new_dataset_version,
            fragment_bitmap: bitmap_after_remap,
            index_details: Some(Arc::new(remapped_index.index_details)),
            index_version: remapped_index.index_version as i32,
            created_at: curr_index_meta.created_at,
            base_id: None,
            files: remapped_index.files,
        },
    };

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![new_index_meta],
            removed_indices: vec![curr_index_meta],
        },
        None,
    );

    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;

    Ok(())
}

/// The outcome of walking a segment's stored provenance forward through a
/// tagged history's consumer edges.
#[derive(Debug)]
enum TaggedRemapPlan {
    /// No transition consumes any covered fragment; the segment is current.
    Identity,
    /// The walk reached a stable-partition transition: its row movement is a
    /// reorder no address arithmetic reproduces, so the segment cannot be
    /// remapped and is skipped cleanly (queries keep translating through the
    /// ledger; a rebuild catches the segment up).
    Blocked { fragment: u32 },
    /// A compaction-only chain: the ordered hops composed into one remap and
    /// the coverage the swapped bitmap must claim.
    Remap {
        remap: RowAddrRemap,
        coverage: RoaringBitmap,
    },
}

/// Walk FORWARD from the segment's provenance along consumer edges. Every hop
/// must be an ordered compaction with `sources ⊆ coverage`; coverage then
/// evolves to the hop's destinations. A partial overlap (the segment covers
/// only some of a hop's sources) takes the v0 straddle fallback: that
/// coverage is dropped from the bitmap (sources removed, destinations NOT
/// added) and the walk continues; the dropped rows are served by scan.
///
/// Iron law (asserted): a swapped bitmap only ever claims destination
/// fragments the segment fully owns -- destinations are added only by a hop
/// applied with ALL of its sources covered, so no destination mixing rows
/// from uncovered sources can be claimed.
fn plan_tagged_remap(ledger: &FragReuseLedger, provenance: &RoaringBitmap) -> TaggedRemapPlan {
    let mut coverage = provenance.clone();
    let mut hops: Vec<RowAddrRemap> = Vec::new();
    let mut applied = vec![false; ledger.transitions().len()];
    let mut changed = false;
    // Transitions are in lineage order (producers before consumers), so one
    // pass reaches a fixpoint.
    for (position, transition) in ledger.transitions().iter().enumerate() {
        let sources: RoaringBitmap = transition
            .sources()
            .iter()
            .map(|digest| digest.id as u32)
            .collect();
        let overlap = &sources & &coverage;
        if overlap.is_empty() {
            continue;
        }
        match transition.mapping() {
            Mapping::StablePartition(_) => {
                return TaggedRemapPlan::Blocked {
                    fragment: overlap.min().unwrap(),
                };
            }
            Mapping::OrderedCompaction(remap) => {
                if overlap == sources {
                    coverage -= &sources;
                    coverage.extend(
                        transition
                            .destinations()
                            .iter()
                            .map(|digest| digest.id as u32),
                    );
                    hops.push(remap.as_ref().clone());
                    applied[position] = true;
                } else {
                    coverage -= &overlap;
                }
                changed = true;
            }
        }
    }
    // Iron law: any destination fragment the final bitmap claims comes from
    // an applied (fully-covered) hop or was directly covered to begin with.
    debug_assert!(
        ledger
            .transitions()
            .iter()
            .enumerate()
            .all(|(position, transition)| {
                applied[position]
                    || transition.destinations().iter().all(|digest| {
                        !coverage.contains(digest.id as u32)
                            || provenance.contains(digest.id as u32)
                    })
            }),
        "a remapped fragment bitmap may only claim destinations the segment fully owns"
    );
    if !changed {
        TaggedRemapPlan::Identity
    } else {
        TaggedRemapPlan::Remap {
            remap: RowAddrRemap::chained(hops),
            coverage,
        }
    }
}

/// Remap one segment of a user index through a tagged history's
/// compaction-only chain. Stable-partition-touching segments are skipped
/// cleanly (logged, not an error) so a maintenance pipeline proceeds to the
/// other segments and indices.
async fn remap_index_tagged(dataset: &mut Dataset, index_id: &Uuid) -> Result<()> {
    // Stored metadata: the segment's provenance bitmap and the entry, never
    // the query-rewritten listing `load_indices` produces on tagged tables.
    let stored = read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await?;
    let entry = stored
        .iter()
        .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
        .ok_or_else(|| {
            Error::index("the fragment reuse index entry disappeared during remap".to_string())
        })?;
    let ledger = decode_frag_reuse_ledger(dataset, entry).await?;
    if ledger.has_unsupported_transitions() {
        return Err(Error::not_supported(
            "the tagged FRI history carries transitions this client cannot interpret; \
             upgrade to a newer version of Lance before remapping",
        ));
    }
    let curr_index_meta = stored
        .iter()
        .find(|idx| idx.uuid == *index_id)
        .cloned()
        .ok_or_else(|| {
            Error::index(format!(
                "index {index_id} not found in manifest; it may have been concurrently dropped"
            ))
        })?;
    let Some(provenance) = curr_index_meta.fragment_bitmap.clone() else {
        log::warn!(
            "Index {} ({}) has no stored fragment bitmap; its lineage through the tagged \
             history cannot be determined, skipping remap. Consider rebuilding the index",
            curr_index_meta.name,
            curr_index_meta.uuid
        );
        return Ok(());
    };

    let (remap, mut coverage) = match plan_tagged_remap(&ledger, &provenance) {
        TaggedRemapPlan::Identity => return Ok(()),
        TaggedRemapPlan::Blocked { fragment } => {
            log::info!(
                "Skipping remap of index {} ({}): its coverage reaches a stable-partition \
                 transition at fragment {}. Queries keep translating through the reuse \
                 index; rebuild the index to catch it up",
                curr_index_meta.name,
                curr_index_meta.uuid,
                fragment
            );
            return Ok(());
        }
        TaggedRemapPlan::Remap { remap, coverage } => (remap, coverage),
    };

    // Straddle-only outcome: coverage was dropped but no address moves, so
    // the index files stay as they are and only the swapped bitmap (plus the
    // advanced dataset_version) is committed. This also avoids opening the
    // index, which the query planner may refuse for a straddled segment.
    let remap_result = if remap.is_empty() {
        RemapResult::Keep(*index_id)
    } else {
        index::remap_index(dataset, index_id, &remap).await?
    };

    // Overlays committed after the source index was built are not
    // incorporated by the remap; exclude those fragments so queries scan
    // their current values instead (mirrors the v0 path).
    for fragment in dataset.manifest.fragments.iter() {
        let has_newer_indexed_overlay = fragment.overlays.iter().any(|overlay| {
            overlay.committed_version > curr_index_meta.dataset_version
                && overlay
                    .data_file
                    .fields
                    .iter()
                    .any(|field_id| curr_index_meta.fields.contains(field_id))
        });
        if has_newer_indexed_overlay {
            coverage.remove(fragment.id as u32);
        }
    }

    let new_index_meta = match remap_result {
        RemapResult::Drop => return Ok(()),
        RemapResult::Keep(new_id) => IndexMetadata {
            uuid: new_id,
            name: curr_index_meta.name.clone(),
            fields: curr_index_meta.fields.clone(),
            covering_fields: curr_index_meta.covering_fields.clone(),
            dataset_version: dataset.manifest.version,
            fragment_bitmap: Some(coverage),
            index_details: curr_index_meta.index_details.clone(),
            index_version: curr_index_meta.index_version,
            created_at: curr_index_meta.created_at,
            base_id: None,
            files: curr_index_meta.files.clone(),
        },
        RemapResult::Remapped(remapped_index) => IndexMetadata {
            uuid: remapped_index.new_id,
            name: curr_index_meta.name.clone(),
            fields: curr_index_meta.fields.clone(),
            covering_fields: curr_index_meta.covering_fields.clone(),
            dataset_version: dataset.manifest.version,
            fragment_bitmap: Some(coverage),
            index_details: Some(Arc::new(remapped_index.index_details)),
            index_version: remapped_index.index_version as i32,
            created_at: curr_index_meta.created_at,
            base_id: None,
            files: remapped_index.files,
        },
    };

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![new_index_meta],
            removed_indices: vec![curr_index_meta],
        },
        None,
    );

    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;

    Ok(())
}

pub async fn remap_column_index(
    dataset: &mut Dataset,
    columns: &[&str],
    name: Option<String>,
) -> Result<()> {
    if columns.len() != 1 {
        return Err(Error::index(
            "Only support remapping index on 1 column at the moment".to_string(),
        ));
    }

    let column = columns[0];
    let Some(field) = dataset.schema().field(column) else {
        return Err(Error::index(format!(
            "RemapIndex: column '{column}' does not exist"
        )));
    };

    let index_name = name.unwrap_or(format!("{column}_idx"));

    // On a tagged table the named segment is resolved from the STORED
    // metadata: the query-filtered listing may exclude exactly the segments
    // maintenance must reach (e.g. a straddled segment left with no query
    // coverage). The v0 path below keeps its filtered lookup untouched.
    let stored = crate::index::load_all_indices(dataset).await?;
    if stored
        .iter()
        .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME && idx.index_version != 0)
    {
        let index = stored
            .iter()
            .find(|idx| idx.name == index_name)
            .ok_or_else(|| Error::index(format!("Index with name {} not found", index_name)))?;
        if index.keyed_field() != Some(field.id) {
            return Err(Error::index(format!(
                "Index name {} already exists with fields {:?} (carried fields {:?}); \
                 expected a single keyed field {}",
                index_name, index.fields, index.covering_fields, field.id
            )));
        }
        if !index.covering_fields.is_empty() {
            return Err(Error::index(format!(
                "Remapping index '{}' is not supported: it declares covering \
                 fields {:?}, which no index builder writes or preserves yet",
                index_name, index.covering_fields,
            )));
        }
        return remap_index_tagged(dataset, &index.uuid).await;
    }

    let indices = dataset.load_indices().await?;
    let index = match indices.iter().find(|i| i.name == index_name) {
        None => {
            return Err(Error::index(format!(
                "Index with name {} not found",
                index_name
            )));
        }
        Some(index) => {
            // The real question is "does this index belong to this column",
            // i.e. its one keyed field is `field.id`. Carried fields are
            // irrelevant here, same as in `index::remap_index`.
            if index.keyed_field() != Some(field.id) {
                Err(Error::index(format!(
                    "Index name {} already exists with fields {:?} (carried fields {:?}); \
                     expected a single keyed field {}",
                    index_name, index.fields, index.covering_fields, field.id
                )))
            } else if !index.covering_fields.is_empty() {
                // Same rule as `optimize_indices`, and for the same reason: no
                // index type carries the declared payload through a remap, so the
                // result would still claim values its storage does not hold. The
                // caller named this index, so refuse out loud -- compaction
                // withdraws instead only because it must not block a table-level
                // operation over one index it cannot remap.
                Err(Error::index(format!(
                    "Remapping index '{}' is not supported: it declares covering \
                     fields {:?}, which no index builder writes or preserves yet",
                    index_name, index.covering_fields,
                )))
            } else {
                Ok(index)
            }
        }
    }?;

    remap_index(dataset, &index.uuid).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A-rows: per-segment eligibility of the tagged remap walk.
    mod tagged_remap_plan {
        use super::*;
        use lance_table::format::pb::fragment_reuse_index_details as pb_fri;
        use prost::Message;

        fn digest(id: u64, rows: u64, deleted: u64) -> pb_fri::FragmentDigest {
            pb_fri::FragmentDigest {
                id,
                physical_rows: rows,
                num_deleted_rows: deleted,
            }
        }

        fn addr(fragment: u32, offset: u32) -> u64 {
            RowAddress::new_from_parts(fragment, offset).into()
        }

        /// An ordered-compaction transition moving every (live) row of
        /// `sources` into `destinations`, in scan order.
        fn ordered(sources: &[u64], destinations: &[u64]) -> pb_fri::Transition {
            let rows_per_fragment = 4u64;
            let mut addrs = RoaringTreemap::new();
            for source in sources {
                for offset in 0..rows_per_fragment as u32 {
                    addrs.insert(addr(*source as u32, offset));
                }
            }
            let mut changed_row_addrs = Vec::new();
            addrs.serialize_into(&mut changed_row_addrs).unwrap();
            let total = rows_per_fragment * sources.len() as u64;
            let per_destination = total / destinations.len() as u64;
            pb_fri::Transition {
                sources: sources
                    .iter()
                    .map(|id| digest(*id, rows_per_fragment, 0))
                    .collect(),
                destinations: destinations
                    .iter()
                    .map(|id| digest(*id, per_destination, 0))
                    .collect(),
                mapping: Some(pb_fri::transition::Mapping::OrderedCompaction(
                    pb_fri::OrderedCompaction { changed_row_addrs },
                )),
            }
        }

        fn partition(sources: &[u64], destinations: &[u64]) -> pb_fri::Transition {
            let rows_per_fragment = 4u64;
            let total = rows_per_fragment * sources.len() as u64;
            let per_destination = total / destinations.len() as u64;
            pb_fri::Transition {
                sources: sources
                    .iter()
                    .map(|id| digest(*id, rows_per_fragment, 0))
                    .collect(),
                destinations: destinations
                    .iter()
                    .map(|id| digest(*id, per_destination, 0))
                    .collect(),
                mapping: Some(pb_fri::transition::Mapping::StablePartition(
                    pb_fri::StablePartition {
                        map_id: Uuid::new_v4().to_string(),
                        map_size_bytes: 1,
                        base_id: None,
                    },
                )),
            }
        }

        async fn ledger(transitions: Vec<pb_fri::Transition>) -> FragReuseLedger {
            let content = pb_fri::InlineContent {
                legacy_versions: vec![],
                transitions,
            }
            .encode_to_vec();
            crate::index::frag_reuse::decode_frag_reuse_ledger_from_content(1, &content)
                .await
                .unwrap()
        }

        fn coverage(fragments: &[u32]) -> RoaringBitmap {
            fragments.iter().copied().collect()
        }

        /// A1: coverage disjoint from the ledger is already current.
        #[tokio::test]
        async fn all_live_is_identity() {
            let ledger = ledger(vec![ordered(&[1], &[2])]).await;
            assert!(matches!(
                plan_tagged_remap(&ledger, &coverage(&[7])),
                TaggedRemapPlan::Identity
            ));
        }

        /// A2: a single ordered hop with sources within coverage remaps and
        /// swaps the bitmap onto the destinations.
        #[tokio::test]
        async fn single_hop_compaction_remaps() {
            let ledger = ledger(vec![ordered(&[1, 2], &[5, 6])]).await;
            let TaggedRemapPlan::Remap { remap, coverage } =
                plan_tagged_remap(&ledger, &coverage(&[1, 2]))
            else {
                panic!("expected a remap plan");
            };
            assert_eq!(coverage, RoaringBitmap::from_iter([5u32, 6]));
            assert_eq!(remap.get(addr(1, 0)), Some(Some(addr(5, 0))));
            assert_eq!(remap.get(addr(2, 3)), Some(Some(addr(6, 3))));
            assert_eq!(remap.get(addr(9, 0)), None);
        }

        /// A3: multi-hop ordered chains compose into one remap.
        #[tokio::test]
        async fn multi_hop_compaction_composes() {
            let ledger = ledger(vec![ordered(&[1], &[2]), ordered(&[2], &[3])]).await;
            let TaggedRemapPlan::Remap { remap, coverage } =
                plan_tagged_remap(&ledger, &coverage(&[1]))
            else {
                panic!("expected a remap plan");
            };
            assert_eq!(coverage, RoaringBitmap::from_iter([3u32]));
            assert_eq!(remap.get(addr(1, 2)), Some(Some(addr(3, 2))));
        }

        /// A4: a stable-partition first hop blocks the whole segment.
        #[tokio::test]
        async fn first_hop_stable_partition_blocks() {
            let ledger = ledger(vec![partition(&[1], &[2])]).await;
            assert!(matches!(
                plan_tagged_remap(&ledger, &coverage(&[1])),
                TaggedRemapPlan::Blocked { fragment: 1 }
            ));
        }

        /// A5: a downstream stable-partition hop blocks too, even though the
        /// first hop is an ordered compaction.
        #[tokio::test]
        async fn downstream_stable_partition_blocks() {
            let ledger = ledger(vec![ordered(&[1], &[2]), partition(&[2], &[3])]).await;
            assert!(matches!(
                plan_tagged_remap(&ledger, &coverage(&[1])),
                TaggedRemapPlan::Blocked { fragment: 2 }
            ));
        }

        /// A6: mixed provenance where one lineage hits a stable partition
        /// skips the whole segment (no partial remap).
        #[tokio::test]
        async fn partially_blocked_segment_is_skipped() {
            let ledger = ledger(vec![ordered(&[1], &[2]), partition(&[5], &[6])]).await;
            assert!(matches!(
                plan_tagged_remap(&ledger, &coverage(&[1, 5])),
                TaggedRemapPlan::Blocked { fragment: 5 }
            ));
        }

        /// A8: the owner's example. A segment covering only {3} of the chain
        /// 3,4 -> 5,6 -> 7,8 straddles the first hop: the group's coverage is
        /// dropped from the bitmap (sources removed, destinations NOT added)
        /// and nothing downstream applies, leaving an empty coverage and no
        /// address movement -- the rows are served by scan (A9).
        #[tokio::test]
        async fn straddled_first_hop_drops_group_coverage() {
            let ledger = ledger(vec![ordered(&[3, 4], &[5, 6]), ordered(&[5, 6], &[7, 8])]).await;
            let TaggedRemapPlan::Remap { remap, coverage } =
                plan_tagged_remap(&ledger, &coverage(&[3]))
            else {
                panic!("expected a (coverage-only) remap plan");
            };
            assert!(coverage.is_empty());
            assert_eq!(remap.get(addr(3, 0)), None, "dropped rows are not moved");

            // The same chain with full coverage composes both hops (A10: the
            // swapped bitmap claims exactly the fully-owned destinations).
            let full = plan_tagged_remap(
                &ledger,
                &[3u32, 4].iter().copied().collect::<RoaringBitmap>(),
            );
            let TaggedRemapPlan::Remap { remap, coverage } = full else {
                panic!("expected a remap plan");
            };
            assert_eq!(coverage, RoaringBitmap::from_iter([7u32, 8]));
            assert_eq!(remap.get(addr(3, 0)), Some(Some(addr(7, 0))));
        }

        /// Straddling mid-chain drops only that hop's coverage while the
        /// other lineage keeps composing.
        #[tokio::test]
        async fn straddle_drop_is_per_group() {
            let ledger = ledger(vec![ordered(&[1], &[2]), ordered(&[5, 6], &[7, 8])]).await;
            let TaggedRemapPlan::Remap { remap, coverage } =
                plan_tagged_remap(&ledger, &coverage(&[1, 5]))
            else {
                panic!("expected a remap plan");
            };
            // Fragment 5's coverage dropped (straddle), fragment 1 remapped.
            assert_eq!(coverage, RoaringBitmap::from_iter([2u32]));
            assert_eq!(remap.get(addr(1, 0)), Some(Some(addr(2, 0))));
            assert_eq!(remap.get(addr(5, 0)), None);
        }
    }

    /// A/C-rows end to end: eligibility, clean skips, and the remap-then-trim
    /// pipeline on a real tagged table.
    mod tagged_remap_integration {
        use super::*;
        use crate::dataset::index::frag_reuse::cleanup_frag_reuse_index;
        use crate::dataset::optimize::{CompactionOptions, compact_files};
        use crate::dataset::write::CommitBuilder;
        use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
        use crate::index::DatasetIndexExt;
        use crate::index::frag_reuse::decode_frag_reuse_ledger;
        use crate::index::frag_reuse_reader::tests as reader_tests;
        use arrow_array::cast::AsArray;
        use arrow_array::types::Int32Type;
        use lance_index::IndexType;
        use lance_index::scalar::ScalarIndexParams;
        use lance_table::transaction::{FragmentReuseRewrite, RewriteGroup};

        async fn reserve_fragments(dataset: &mut Dataset, num_fragments: u32) {
            dataset
                .apply_commit(
                    Transaction::new(
                        dataset.manifest.version,
                        Operation::ReserveFragments { num_fragments },
                        None,
                    ),
                    &Default::default(),
                    &Default::default(),
                )
                .await
                .unwrap();
        }

        async fn commit_stable_partition(
            dataset: Dataset,
            source_ids: &[u64],
            dest_base_id: u64,
        ) -> Dataset {
            let old_fragments: Vec<Fragment> = source_ids
                .iter()
                .map(|id| {
                    dataset
                        .fragments()
                        .iter()
                        .find(|f| f.id == *id)
                        .unwrap()
                        .clone()
                })
                .collect();
            let (transition, destinations) =
                reader_tests::prepare_partition(&dataset, source_ids, dest_base_id).await;
            let read_version = dataset.manifest.version;
            CommitBuilder::new(Arc::new(dataset))
                .execute(Transaction::new(
                    read_version,
                    Operation::Rewrite {
                        groups: vec![RewriteGroup {
                            old_fragments,
                            new_fragments: destinations,
                        }],
                        rewritten_indices: vec![],
                        frag_reuse_index: None,
                        frag_reuse_rewrite: Some(FragmentReuseRewrite {
                            transitions: vec![transition],
                            base_entry_version: None,
                        }),
                    },
                    None,
                ))
                .await
                .unwrap()
        }

        async fn stored_index(dataset: &Dataset, name: &str) -> IndexMetadata {
            read_manifest_indexes(
                &dataset.object_store,
                &dataset.manifest_location,
                &dataset.manifest,
            )
            .await
            .unwrap()
            .into_iter()
            .find(|idx| idx.name == name)
            .unwrap()
        }

        /// Replace an index's stored bitmap with an under-claim; the segment
        /// then only answers for `covered` and everything else is scanned.
        async fn narrow_index_bitmap(dataset: &mut Dataset, name: &str, covered: &[u32]) {
            let original = stored_index(dataset, name).await;
            let mut narrowed = original.clone();
            narrowed.fragment_bitmap = Some(covered.iter().copied().collect());
            dataset
                .apply_commit(
                    Transaction::new(
                        dataset.manifest.version,
                        Operation::CreateIndex {
                            new_indices: vec![narrowed],
                            removed_indices: vec![original],
                        },
                        None,
                    ),
                    &Default::default(),
                    &Default::default(),
                )
                .await
                .unwrap();
        }

        async fn append_two_fragments(dataset: Dataset) -> Dataset {
            let batch = lance_datagen::gen_batch()
                .col("i", lance_datagen::array::step_custom::<Int32Type>(8, 1))
                .into_batch_rows(lance_datagen::RowCount::from(8))
                .unwrap();
            InsertBuilder::new(Arc::new(dataset))
                .with_params(&WriteParams {
                    mode: WriteMode::Append,
                    max_rows_per_file: 4,
                    ..Default::default()
                })
                .execute(vec![batch])
                .await
                .unwrap()
        }

        async fn sorted_values(dataset: &Dataset, predicate: Option<&str>) -> Vec<i32> {
            let mut scan = dataset.scan();
            if let Some(predicate) = predicate {
                scan.filter(predicate).unwrap();
            }
            let batch = scan.try_into_batch().await.unwrap();
            let mut values: Vec<i32> = batch["i"]
                .as_primitive::<Int32Type>()
                .iter()
                .map(|value| value.unwrap())
                .collect();
            values.sort_unstable();
            values
        }

        /// C2 + the pipeline: on one tagged table, a compaction-only segment
        /// remaps (bitmap swapped, version advanced), an SP-touching segment
        /// skips cleanly, and the following cleanup trims exactly the
        /// compaction transition no index needs anymore while the
        /// stable-partition transition with un-drained consumers stays.
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn remap_skip_and_trim_pipeline() {
            // Fragments {0,1} indexed by i_idx; {2,3} appended after, with a
            // second index narrowed onto them.
            let dataset = reader_tests::fixture().await;
            let mut dataset = append_two_fragments(dataset).await;
            dataset
                .create_index(
                    &["i"],
                    IndexType::Scalar,
                    Some("b_idx".into()),
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
            narrow_index_bitmap(&mut dataset, "b_idx", &[2, 3]).await;
            let all_values: Vec<i32> = (0..16).collect();
            assert_eq!(sorted_values(&dataset, None).await, all_values);

            // The table becomes tagged: {0,1} -> {10,11} by stable partition.
            reserve_fragments(&mut dataset, 40).await;
            let mut dataset = commit_stable_partition(dataset, &[0, 1], 10).await;

            // A deferred tagged compaction rewrites the small fragments in
            // two groups: the SP destinations and the appended pair.
            compact_files(
                &mut dataset,
                CompactionOptions {
                    target_rows_per_fragment: 8,
                    defer_index_remap: true,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
            let entry = stored_index(&dataset, FRAG_REUSE_INDEX_NAME).await;
            let ledger = decode_frag_reuse_ledger(&dataset, &entry).await.unwrap();
            assert_eq!(ledger.transitions().len(), 3, "one SP and two OC groups");

            // i_idx's lineage passes through the stable partition: clean
            // skip, nothing committed, metadata untouched.
            let i_before = stored_index(&dataset, "i_idx").await;
            let version_before = dataset.manifest.version;
            remap_column_index(&mut dataset, &["i"], Some("i_idx".into()))
                .await
                .unwrap();
            assert_eq!(dataset.manifest.version, version_before);
            let i_after = stored_index(&dataset, "i_idx").await;
            assert_eq!(i_after.uuid, i_before.uuid);
            assert_eq!(i_after.fragment_bitmap, i_before.fragment_bitmap);

            // b_idx's lineage is compaction-only: remapped, bitmap swapped to
            // the destination, dataset_version advanced.
            let b_before = stored_index(&dataset, "b_idx").await;
            remap_column_index(&mut dataset, &["i"], Some("b_idx".into()))
                .await
                .unwrap();
            let b_after = stored_index(&dataset, "b_idx").await;
            assert_ne!(b_after.uuid, b_before.uuid);
            // Advanced to the version the remap was built on (the commit
            // itself is one version later), same as the v0 path.
            assert_eq!(b_after.dataset_version, dataset.manifest.version - 1);
            assert!(b_after.dataset_version > b_before.dataset_version);
            let destination = ledger.transitions()[ledger
                .consumer(2)
                .expect("fragment 2 must be consumed by a compaction")]
            .destinations()[0]
                .id as u32;
            assert_eq!(
                b_after.fragment_bitmap.as_ref().unwrap(),
                &RoaringBitmap::from_iter([destination])
            );

            // Queries still return the exact values through every path.
            assert_eq!(sorted_values(&dataset, None).await, all_values);
            assert_eq!(
                sorted_values(&dataset, Some("i >= 8 AND i < 12")).await,
                (8..12).collect::<Vec<_>>()
            );
            assert_eq!(sorted_values(&dataset, Some("i = 3")).await, vec![3]);

            // The pipeline's release: cleanup trims the {2,3} compaction (no
            // index needs it after the remap) while the stable partition and
            // its downstream compaction stay for the un-drained i_idx.
            cleanup_frag_reuse_index(&mut dataset).await.unwrap();
            let entry = stored_index(&dataset, FRAG_REUSE_INDEX_NAME).await;
            let ledger = decode_frag_reuse_ledger(&dataset, &entry).await.unwrap();
            assert_eq!(ledger.transitions().len(), 2);
            assert!(ledger.consumer(2).is_none(), "the drained OC is trimmed");
            assert!(
                ledger.consumer(0).is_some(),
                "the SP with un-drained consumers stays"
            );
            assert!(
                ledger.consumer(10).is_some(),
                "the OC downstream of the retained SP stays"
            );
            assert_eq!(sorted_values(&dataset, None).await, all_values);
        }

        /// A8/A9 integration: a segment covering only part of a compaction
        /// group takes the straddle fallback -- the group's coverage is
        /// dropped from the swapped bitmap -- and the affected rows are
        /// served by scan with correct results.
        ///
        /// The compaction planner never bins indexed and unindexed fragments
        /// together, so this state cannot arise from a planned compaction;
        /// the fallback is defensive. Recreate it by narrowing the segment's
        /// stored bitmap (a legal under-claim) after the compaction.
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn straddled_segment_drops_coverage_and_scans() {
            let dataset = reader_tests::fixture().await;
            let mut dataset = append_two_fragments(dataset).await;
            reserve_fragments(&mut dataset, 40).await;
            // Tag the table via a stable partition on the OTHER fragments.
            let mut dataset = commit_stable_partition(dataset, &[2, 3], 10).await;
            compact_files(
                &mut dataset,
                CompactionOptions {
                    target_rows_per_fragment: 8,
                    defer_index_remap: true,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
            // i_idx now claims only fragment 0 of the compacted {0,1} group.
            narrow_index_bitmap(&mut dataset, "i_idx", &[0]).await;

            let before = stored_index(&dataset, "i_idx").await;
            let version_before = dataset.manifest.version;
            remap_column_index(&mut dataset, &["i"], Some("i_idx".into()))
                .await
                .unwrap();
            // The drop is committed (with no address movement to apply, the
            // segment may keep its uuid through the empty-remap shortcut).
            assert_eq!(dataset.manifest.version, version_before + 1);
            let after = stored_index(&dataset, "i_idx").await;
            assert!(
                after.fragment_bitmap.as_ref().unwrap().is_empty(),
                "the straddled group's coverage is dropped, not partially claimed"
            );
            assert!(after.dataset_version > before.dataset_version);

            // The dropped rows come back correct through scans.
            assert_eq!(
                sorted_values(&dataset, Some("i < 4")).await,
                (0..4).collect::<Vec<_>>()
            );
            assert_eq!(
                sorted_values(&dataset, None).await,
                (0..16).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_compact_matches_transpose() {
        use lance_core::utils::row_addr_remap::GroupInputWithLayout;
        // Ascending old fragments (compaction's scan order), with deletions.
        let old = vec![
            FragDigest {
                id: 0,
                physical_rows: 5,
                num_deleted_rows: 2,
            },
            FragDigest {
                id: 1,
                physical_rows: 4,
                num_deleted_rows: 1,
            },
            FragDigest {
                id: 3,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
        ];
        // 9 rewritten rows (offsets that survived in each old fragment).
        let rewritten = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 0),
            (1, 1),
            (1, 3),
            (3, 0),
            (3, 1),
            (3, 2),
        ];
        let addrs = RoaringTreemap::from_iter(
            rewritten
                .iter()
                .map(|(f, o)| u64::from(RowAddress::new_from_parts(*f, *o))),
        );
        // 9 rewritten rows split across two new fragments.
        let new = vec![
            FragDigest {
                id: 10,
                physical_rows: 4,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 11,
                physical_rows: 5,
                num_deleted_rows: 0,
            },
        ];

        let expected = transpose_row_ids_from_digest(addrs.clone(), &old, &new);
        let compact = RowAddrRemap::compact_with_layout([GroupInputWithLayout {
            rewritten_old_row_addrs: addrs,
            old_frags: old
                .iter()
                .map(|f| (f.id as u32, f.physical_rows as u32))
                .collect(),
            new_frags: new
                .iter()
                .map(|f| (f.id as u32, f.physical_rows as u32))
                .collect(),
        }])
        .unwrap();

        // Every real address in the old fragments must map identically.
        for f in &old {
            for o in 0..f.physical_rows as u32 {
                let a = u64::from(RowAddress::new_from_parts(f.id as u32, o));
                assert_eq!(
                    compact.get(a),
                    expected.get(&a).copied(),
                    "mismatch at ({}, {})",
                    f.id,
                    o
                );
            }
        }
        // A fragment outside the group is unaffected by both.
        let outside = u64::from(RowAddress::new_from_parts(99, 0));
        assert_eq!(compact.get(outside), expected.get(&outside).copied());
    }

    #[test]
    fn test_missing_indices() {
        // Sanity test to make sure MissingIds works.  Does not test actual functionality so
        // feel free to remove if it becomes inconvenient
        let frags = vec![
            FragDigest {
                id: 0,
                physical_rows: 5,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 3,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
        ];
        let rows = [(0, 1), (0, 3), (0, 4), (3, 0), (3, 2)]
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into());

        let missing = MissingAddrs::new(rows, &frags).collect::<Vec<_>>();
        let expected_missing = [(0, 0), (0, 2), (3, 1)]
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into())
            .collect::<Vec<u64>>();
        assert_eq!(missing, expected_missing);
    }

    #[test]
    fn test_missing_ids() {
        // test with missing first row
        // test with missing last row
        // test fragment ids out of order

        let fragments = vec![
            FragDigest {
                id: 0,
                physical_rows: 5,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 3,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 1,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
        ];

        // Written as pairs of (fragment_id, offset)
        let row_addrs = vec![
            (0, 1),
            (0, 3),
            (0, 4),
            (3, 0),
            (3, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        ];
        let row_addrs = row_addrs
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into());
        let result = MissingAddrs::new(row_addrs, &fragments).collect::<Vec<_>>();

        let expected = vec![(0, 0), (0, 2), (3, 1)];
        let expected = expected
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into())
            .collect::<Vec<u64>>();
        assert_eq!(result, expected);
    }

    /// A *physical* remap through the real production entry point,
    /// `remap_column_index`, must refuse a covered index rather than quietly do
    /// nothing. No index type carries the declared payload through a remap, and
    /// the caller named this index, so a silent no-op would hand back an index
    /// that covers nothing with no indication why.
    ///
    /// Asserting on committed metadata here would prove nothing: the flow is
    /// `remap_column_index` -> the private `remap_index` -> `index::remap_index`,
    /// which withdraws a covered index before the fully-deleted `Keep` check, and
    /// the `Drop` arm returns without committing. The `Keep`/`Remapped` arms are
    /// therefore unreachable for a covered index, so a "declaration preserved"
    /// assertion would pass even if those arms stopped preserving it. Assert the
    /// refusal instead.
    #[tokio::test]
    async fn test_remap_column_index_refuses_a_covered_index() {
        use crate::dataset::index::DatasetIndexRemapperOptions;
        use crate::dataset::optimize::{
            CompactionOptions, commit_compaction, plan_compaction, rewrite_files,
        };
        use crate::utils::test::covering;
        use lance_core::utils::tempfile::TempStrDir;
        use std::borrow::Cow;

        let test_uri = TempStrDir::default();
        // Two fragments, so compaction has something to merge.
        let mut dataset = covering::write_vector_payload_dataset(&test_uri).await;
        covering::append_vector_payload_rows(&mut dataset, covering::ROWS_PER_FRAGMENT).await;
        covering::create_ivf_pq_index(&mut dataset, "vec").await;

        let (_, id_field_id) = covering::declare_covering(&mut dataset, "vec", "payload").await;
        let index_name = dataset.load_indices().await.unwrap()[0].name.clone();

        // Delete some (not all) rows so the remap has real work to do and does
        // not take the all-fragments-deleted `RemapResult::Keep` shortcut.
        dataset.delete("payload < 100").await.unwrap();

        let options = CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert!(
            !plan.tasks().is_empty(),
            "compaction plan must have work to do, or this test proves nothing"
        );
        for task in plan.tasks().iter() {
            let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), task.clone(), &options)
                .await
                .unwrap();
            commit_compaction(
                &mut dataset,
                Vec::from([rewrite_result]),
                Arc::new(DatasetIndexRemapperOptions::default()),
                &options,
            )
            .await
            .unwrap();
        }

        let before = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name == index_name)
            .cloned()
            .expect("precondition: the covered index exists");

        // `remap_column_index` is user-directed -- the caller named this index --
        // so it must refuse rather than no-op. Refusing is also what makes this
        // test meaningful: the `Keep`/`Remapped` arms below are unreachable for a
        // covered index, so asserting on the committed metadata instead would
        // pass whether or not those arms preserved `covering_fields`.
        let error = remap_column_index(&mut dataset, &["vec"], Some(index_name.clone()))
            .await
            .expect_err("remapping a covered index must be refused");
        assert!(
            error.to_string().contains("declares covering fields"),
            "unexpected message: {error}"
        );

        // Refused, not half-applied.
        let after = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name == index_name)
            .cloned()
            .expect("a refused remap must leave the index in place");
        assert_eq!(
            after.uuid, before.uuid,
            "a refused remap replaced the index"
        );
        assert_eq!(after.covering_fields, vec![id_field_id]);
        assert_eq!(after.fragment_bitmap, before.fragment_bitmap);
    }
}
