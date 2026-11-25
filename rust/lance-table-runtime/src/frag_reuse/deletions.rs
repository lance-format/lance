// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use lance_core::utils::fragment_reuse::{MappingReader, OrderedCompactionMapping};
use lance_core::{Error, Result};
use lance_table::format::Fragment;
use lance_table::system_index::frag_reuse::stable_partition::{
    MAPPING_FILE, StablePartitionMapping,
};
use lance_table::transaction::RewriteGroup;
use roaring::{RoaringBitmap, RoaringTreemap};

use crate::io::deletion::read_table_deletion_file;
use crate::paths::FRAG_REUSE_DIR;
use crate::{FileFragment, Table};
use lance_core::utils::address::RowAddress;

/// The source fragment's deleted row count as the manifest sees it right
/// now. Materialized metadata is free; a deletion file whose count was never
/// materialized is read once (rare -- the delete path materializes counts).
pub(super) async fn current_deleted_rows(
    dataset: &Table,
    frag: &Fragment,
) -> lance_core::Result<u64> {
    match &frag.deletion_file {
        None => Ok(0),
        Some(deletion) => match deletion.num_deleted_rows {
            Some(count) => Ok(count as u64),
            None => Ok(
                crate::io::deletion::read_table_deletion_file(dataset, frag.id, deletion)
                    .await?
                    .len() as u64,
            ),
        },
    }
}

/// Validate job-side deletion folding with exact accounting.
///
/// A concurrent delete landed on a source after the rewrite job snapshotted
/// it; instead of recomputing, the job translated each newly deleted source
/// row through the transition's own mapping and wrote destination deletion
/// vectors for exactly those positions. Three equations, all equalities:
///
/// 1. per source, the rows of its CURRENT deletion vector that were already
///    dead at rewrite time (a null row-map label for stable partition, an
///    address absent from the survivor bitmap for ordered compaction) must
///    number exactly the digest's rewrite-time deleted count -- this also
///    pins the arithmetic that a delta row can never predate the mapping,
///    and implies the remaining rows number exactly the delta;
/// 2. the destination deletion vectors must hold exactly the summed delta
///    rows -- no extras, no misses;
/// 3. per destination, the translated delta positions must equal the
///    supplied deletion vector as a SET -- a count match with a position
///    mismatch is one row wrongly dead plus one row resurrected.
///
/// AUTHORITY: source deletion state is read from the CURRENT MANIFEST
/// fragments, never from the job-supplied rewrite group, which establishes
/// group identity and ordering only. Trusting the group here would let a
/// crafted group deletion vector resurrect one row while wrongly deleting
/// another with all counts matching.
///
/// The translation is the reader stack's own [`MappingReader`]
/// (`OrderedCompactionMapping` over the transition's survivor bitmap, or
/// `StablePartitionMapping` over the row map file, opened exactly as the
/// reader opens it). Deleted offsets are translated in bounded batches, and
/// the expected destination positions accumulate as one `RoaringBitmap` per
/// destination, so memory stays bounded by the batch size plus the bitmaps.
/// IO: none at all for ordered compaction (the bitmap is already in the
/// transition details); for stable partition, only the label blocks each
/// batch touches, plus the reader's one tail read for the counts matrix.
pub(super) async fn validate_folded_deletions(
    dataset: &Table,
    group: &RewriteGroup,
    transition: &lance_table::format::pb::fragment_reuse_index_details::Transition,
    manifest_by_id: &HashMap<u64, &Fragment>,
    source_deltas: &[u64],
) -> lance_core::Result<()> {
    use lance_table::format::pb::fragment_reuse_index_details::transition::Mapping;

    /// Deleted offsets translated per request; bounds the translation
    /// buffers and, for stable partition, the label blocks in flight.
    const FOLDING_BATCH_ROWS: usize = 16 * 1024;

    // One common reader over either mapping kind.
    let mapping: Arc<dyn MappingReader> = match &transition.mapping {
        Some(Mapping::OrderedCompaction(ordered)) => {
            let mut cursor = std::io::Cursor::new(&ordered.changed_row_addrs);
            let bitmap = roaring::RoaringTreemap::deserialize_from(&mut cursor)
                .map_err(|e| Error::invalid_input(e.to_string()))?;
            let layout =
                |digests: &[lance_table::format::pb::fragment_reuse_index_details::FragmentDigest]| {
                    digests
                        .iter()
                        .map(|digest| (digest.id as u32, digest.physical_rows as u32))
                        .collect()
                };
            let remap = lance_core::utils::row_addr_remap::RowAddrRemap::compact_with_layout([
                lance_core::utils::row_addr_remap::GroupInputWithLayout {
                    rewritten_old_row_addrs: bitmap,
                    old_frags: layout(&transition.sources),
                    new_frags: layout(&transition.destinations),
                },
            ])
            .map_err(|e| Error::invalid_input(e.to_string()))?;
            Arc::new(OrderedCompactionMapping::new(Arc::new(remap)))
        }
        Some(Mapping::StablePartition(reference)) => {
            // Open the row map exactly as the reader does (base-aware).
            let base = match reference.base_id {
                None => dataset.state.base.clone(),
                Some(id) => dataset
                    .manifest
                    .base_paths
                    .get(&id)
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "mapping {} references missing base {id}",
                            reference.map_id
                        ))
                    })?
                    .extract_path(dataset.state.session.store_registry())?,
            };
            let directory = base.join(FRAG_REUSE_DIR).join(reference.map_id.as_str());
            let store = dataset.object_store(reference.base_id).await?;
            let metadata_cache = dataset.state.metadata_cache.file_metadata_cache(&directory);
            Arc::new(StablePartitionMapping::try_from_file(
                store,
                directory.join(MAPPING_FILE),
                reference.map_size_bytes,
                metadata_cache,
                transition.sources.clone(),
                transition.destinations.clone(),
                None,
            )?)
        }
        None => {
            return Err(Error::invalid_input(
                "a transition carrying folded deletions has no mapping",
            ));
        }
    };

    // Expected destination deletions, one bitmap per destination.
    let destination_slot: HashMap<u64, usize> = transition
        .destinations
        .iter()
        .enumerate()
        .map(|(slot, digest)| (digest.id, slot))
        .collect();
    let mut expected: Vec<RoaringBitmap> =
        vec![RoaringBitmap::new(); transition.destinations.len()];
    let mut expected_total = 0u64;

    for digest in transition.sources.iter() {
        // AUTHORITY: the current manifest fragment, not the group copy.
        let Some(frag) = manifest_by_id.get(&digest.id) else {
            return Err(Error::invalid_input(format!(
                "source fragment {} is no longer in the current dataset",
                digest.id
            )));
        };
        let mut offsets: Vec<u32> = match &frag.deletion_file {
            None => Vec::new(),
            Some(deletion) => {
                crate::io::deletion::read_table_deletion_file(dataset, frag.id, deletion)
                    .await?
                    .iter()
                    .collect()
            }
        };
        // Sorted for block locality in the stable-partition reads.
        offsets.sort_unstable();
        let mut rewrite_time_deleted = 0u64;
        for chunk in offsets.chunks(FOLDING_BATCH_ROWS) {
            let mut addresses = Vec::with_capacity(chunk.len());
            for &offset in chunk {
                if u64::from(offset) >= digest.physical_rows {
                    return Err(Error::invalid_input(format!(
                        "source fragment {} has a deleted row offset {offset} beyond its \
                         {} physical rows",
                        frag.id, digest.physical_rows
                    )));
                }
                addresses.push((digest.id << 32) | u64::from(offset));
            }
            for translation in mapping.remap_row_ids(&addresses).await? {
                match translation {
                    None => rewrite_time_deleted += 1,
                    Some(address) => {
                        let slot = destination_slot.get(&(address >> 32)).ok_or_else(|| {
                            Error::invalid_input(format!(
                                "the mapping translates to fragment {} which is not a \
                                     destination of this transition",
                                address >> 32
                            ))
                        })?;
                        expected[*slot].insert(address as u32);
                        expected_total += 1;
                    }
                }
            }
        }
        // Equation 1: rows dead at rewrite time, exactly the digest's count.
        if rewrite_time_deleted != digest.num_deleted_rows {
            return Err(Error::invalid_input(format!(
                "scan-time deletion accounting failed for source fragment {}: \
                 {rewrite_time_deleted} deleted rows predate the rewrite mapping but the \
                 digest records {} rewrite-time deletions",
                digest.id, digest.num_deleted_rows
            )));
        }
    }

    // The supplied destination deletion vectors, per destination.
    let mut supplied: Vec<RoaringBitmap> = Vec::with_capacity(group.new_fragments.len());
    let mut supplied_total = 0u64;
    for frag in group.new_fragments.iter() {
        let bitmap: RoaringBitmap = match &frag.deletion_file {
            None => RoaringBitmap::new(),
            Some(deletion) => {
                let bitmap: RoaringBitmap =
                    crate::io::deletion::read_table_deletion_file(dataset, frag.id, deletion)
                        .await?
                        .iter()
                        .collect();
                // The metadata count is consumed downstream (e.g. row
                // counting) without re-reading the file, so a lying count
                // must be rejected here even though the positions themselves
                // are validated from the file below.
                if let Some(num_deleted_rows) = deletion.num_deleted_rows
                    && num_deleted_rows as u64 != bitmap.len()
                {
                    return Err(Error::invalid_input(format!(
                        "destination fragment {} records {num_deleted_rows} deleted rows \
                         in its deletion file metadata but the deletion file holds {} \
                         positions",
                        frag.id,
                        bitmap.len()
                    )));
                }
                bitmap
            }
        };
        supplied_total += bitmap.len();
        supplied.push(bitmap);
    }
    // Equation 2: cardinality, exactly the summed deltas.
    let expected_delta: u64 = source_deltas.iter().sum();
    if supplied_total != expected_delta {
        return Err(Error::invalid_input(format!(
            "destination deletion vectors hold {supplied_total} rows but the sources \
             gained {expected_delta} deletions since the scan"
        )));
    }
    debug_assert_eq!(expected_total, expected_delta);
    // Equation 3: positions, per destination, as sets.
    for (slot, digest) in transition.destinations.iter().enumerate() {
        if expected[slot] != supplied[slot] {
            let missing = (&expected[slot] - &supplied[slot]).iter().next();
            let extra = (&supplied[slot] - &expected[slot]).iter().next();
            return Err(Error::invalid_input(format!(
                "folded deletions do not land on the translated positions for destination \
                 fragment {}: missing translated offset {missing:?}, unexpected deletion \
                 offset {extra:?}",
                digest.id
            )));
        }
    }
    Ok(())
}

/// Every live row address of `fragments`: their physical rows minus deletions,
/// read from fragment metadata and deletion files.
pub async fn live_source_row_addrs(
    dataset: &Table,
    fragments: &[Fragment],
) -> Result<RoaringTreemap> {
    let mut addrs = RoaringTreemap::new();
    for fragment in fragments {
        let fragment_id = u32::try_from(fragment.id).map_err(|_| {
            Error::internal(format!(
                "Fragment id {} is outside the row-address range",
                fragment.id
            ))
        })?;
        let physical_rows = fragment.physical_rows.ok_or_else(|| {
            Error::internal(format!(
                "Fragment {} is missing physical_rows after migration",
                fragment.id
            ))
        })?;
        let start = u64::from(RowAddress::first_row(fragment_id));
        addrs.insert_range(start..start + physical_rows as u64);
        if let Some(deletion_file) = &fragment.deletion_file {
            let deletions = read_table_deletion_file(dataset, fragment.id, deletion_file).await?;
            for offset in deletions.to_sorted_iter() {
                addrs.remove(u64::from(RowAddress::new_from_parts(fragment_id, offset)));
            }
        }
    }
    Ok(addrs)
}

/// Materialize the counts the fragment reuse digests and conservation
/// validation depend on, without changing the fragment list. Unlike
/// `migrate_fragments`, fully-deleted fragments are KEPT: the rewrite
/// group must still remove them from the manifest and the recorded
/// transition must still consume them (they contribute zero live rows), so
/// a silently shortened list would desynchronize the digests, the group and
/// the manifest.
pub async fn normalize_source_fragments(
    dataset: &Table,
    fragments: &[Fragment],
) -> Result<Vec<Fragment>> {
    let mut normalized = Vec::with_capacity(fragments.len());
    for fragment in fragments {
        let mut fragment = fragment.clone();
        if fragment.physical_rows.is_none() {
            let file_fragment = FileFragment::new(Arc::new(dataset.clone()), fragment.clone());
            fragment.physical_rows = Some(file_fragment.physical_rows().await?);
        }
        if fragment
            .deletion_file
            .as_ref()
            .is_some_and(|deletion| deletion.num_deleted_rows.is_none())
        {
            let deletion_file = fragment.deletion_file.as_ref().unwrap();
            let count = read_table_deletion_file(dataset, fragment.id, deletion_file)
                .await?
                .len();
            fragment.deletion_file.as_mut().unwrap().num_deleted_rows = Some(count);
        }
        normalized.push(fragment);
    }
    Ok(normalized)
}
