// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compaction commit coordination, index remapping, and fragment publication.
//!
//! Committing any subset of completed tasks is valid. Once the rewrite commit
//! is attempted, failures leave staged files for dataset GC because an
//! ambiguous success may already reference them. Tests live in
//! `optimize/tests/commit.rs` and `optimize/tests/frag_reuse.rs`.

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use super::execution::{CompactionMetrics, RewriteResult};
use super::planning::{
    CompactionOptions, IndexRemapMode, has_eager_remappable_index,
    reject_eager_remap_over_undrained_fri, reject_unremappable_rewrite,
    validate_fragment_groups_preserve_index_boundaries,
};
use super::remapping::{self, IndexRemapperOptions};
use crate::Dataset;
use crate::dataset::fragment::FileFragment;
use crate::dataset::transaction::{
    Operation, RewriteGroup, RewrittenIndex, Transaction, TransactionBuilder,
};
use crate::index::frag_reuse::build_new_frag_reuse_index;
use crate::index::frag_reuse_with_stable_row_ids::ensure_frag_reuse_recording_allowed;
use crate::index::loading::load_all_indices;
use crate::io::commit::{commit_transaction, default_commit_retry_timeout};
use crate::io::deletion::read_dataset_deletion_file;
use lance_core::utils::row_addr_remap::{GroupInput, RowAddrRemap};
use lance_core::{Error, Result};
use lance_index::frag_reuse::{FRAG_REUSE_INDEX_NAME, FragReuseGroup};
use lance_table::format::Fragment;
use roaring::{RoaringBitmap, RoaringTreemap};

/// Returns the version the reservation committed at; `dataset` is not moved to it.
pub(super) async fn reserve_fragment_ids(
    dataset: &Dataset,
    fragments: impl ExactSizeIterator<Item = &mut Fragment>,
) -> Result<u64> {
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::ReserveFragments {
            num_fragments: fragments.len() as u32,
        },
        None,
    );

    let (manifest, _) = commit_transaction(
        dataset,
        dataset.object_store.as_ref(),
        dataset.commit_handler.as_ref(),
        &transaction,
        &Default::default(),
        &Default::default(),
        default_commit_retry_timeout(),
        dataset.manifest_location.naming_scheme,
        None,
        None,
    )
    .await?;

    // Need +1 since max_fragment_id is inclusive in this case and ranges are exclusive
    let new_max_exclusive = manifest.max_fragment_id.unwrap_or(0) + 1;
    let reserved_ids = (new_max_exclusive - fragments.len() as u32)..(new_max_exclusive);

    for (fragment, new_id) in fragments.zip(reserved_ids) {
        fragment.id = new_id as u64;
    }

    Ok(manifest.version)
}

/// The tasks carry the recording intent, so those that recorded moves keep them even
/// when the commit options omit `defer_index_remap`.
///
/// `remapping_eagerly` is true when this commit is about to remap an
/// address-domain index immediately with these recorded addresses (no
/// deferral, no FRI). That consumption never writes or extends a fragment
/// reuse index, so it is kept out of the FRI-with-stable-row-ids gate, and
/// the caller never folds it into a fragment-reuse record.
async fn records_stable_row_id_moves(
    dataset: &Dataset,
    completed_tasks: &[RewriteResult],
    options: &CompactionOptions,
    remapping_eagerly: bool,
) -> Result<bool> {
    let recorded = completed_tasks
        .iter()
        .filter(|task| task.row_addrs.is_some())
        .count();
    if recorded != 0 && recorded != completed_tasks.len() {
        return Err(Error::invalid_input(format!(
            "Cannot commit compaction tasks of which {recorded} of {} recorded row-address \
             moves: run every task with the same defer_index_remap",
            completed_tasks.len()
        )));
    }
    if remapping_eagerly {
        return Ok(false);
    }
    if !options.defer_index_remap && recorded == 0 {
        return Ok(false);
    }
    if recorded == 0 {
        return Err(Error::invalid_input(
            "Cannot commit with defer_index_remap: the tasks ran without recording row-address \
             moves; run the compaction with defer_index_remap",
        ));
    }
    ensure_frag_reuse_recording_allowed(&dataset.manifest, &load_all_indices(dataset).await?)?;
    Ok(true)
}

/// Commit the results of file compaction.
///
/// It is not required that all tasks are passed to this method. If some failed,
/// they can be omitted and the successful tasks can be committed. However, once
/// some of the tasks have been committed, the remainder of the tasks will not
/// be able to be committed and should be considered cancelled.
///
/// With stable row IDs, tasks that recorded their row-address moves (see
/// [`CompactionOptions::defer_index_remap`]) commit them even if `options` does not
/// set `defer_index_remap`; either all tasks or none must have recorded them. Tasks
/// that did not record commit without recording, also on a table that has a fragment
/// reuse index.
pub async fn commit_compaction(
    dataset: &mut Dataset,
    completed_tasks: Vec<RewriteResult>,
    remap_options: Arc<dyn IndexRemapperOptions>,
    options: &CompactionOptions,
) -> Result<CompactionMetrics> {
    if completed_tasks.is_empty() {
        return Ok(CompactionMetrics::default());
    }

    // A tagged FRI history stores row translation in the manifest entry and
    // depends on index bitmaps keeping their retired provenance, so a
    // compaction on a tagged table can only commit by appending transitions
    // (the deferred path below). Refuse the eager path up front, before any
    // remapping work runs into the commit gate.
    let tagged_fri = load_all_indices(dataset)
        .await?
        .iter()
        .any(lance_table::system_index::frag_reuse::metadata::is_tagged);
    if tagged_fri && !options.defer_index_remap {
        return Err(Error::not_supported(
            "Compacting a table with a tagged fragment reuse history requires \
             defer_index_remap; set defer_index_remap, or upgrade to a writer that \
             remaps tagged histories eagerly",
        ));
    }

    // Before anything is written or committed. The condition mirrors the
    // planner's: see `unremappable_index_coverage` for why this only applies
    // without stable row ids.
    let uses_stable_row_ids = dataset.manifest.uses_stable_row_ids();
    if !uses_stable_row_ids && !options.defer_index_remap {
        reject_unremappable_rewrite(dataset, &completed_tasks).await?;
    }

    let has_address_style = completed_tasks.iter().any(|t| t.row_addrs.is_some());
    // Address-style results require immediate index remapping unless it is
    // deferred. On a stable-row-id table, only an index whose type actually
    // supports an eager remap (today, BTree) makes this worth attempting:
    // ZoneMap/BloomFilter/FM always report `RemapResult::Drop`, so creating a
    // remapper for them would materialize an old-to-new move for every
    // rewritten row only to discard it -- on a large compaction, table-scale
    // memory the rewrite files have already been written to produce.
    // `rewrite_files`'s capture decision already applies this same test
    // (`has_eager_remappable_index`), so this mostly re-confirms it; it
    // matters on its own in distributed mode, where a task can record moves
    // under `defer_index_remap` at plan time and then commit under
    // different, eager options with no remappable index at all -- there
    // `needs_remapping` must stay false so the recorded moves fall through
    // to the fragment-reuse index below instead of being silently discarded.
    let needs_remapping = !options.defer_index_remap
        && has_address_style
        && (!uses_stable_row_ids || has_eager_remappable_index(dataset).await?);

    if needs_remapping {
        reject_eager_remap_over_undrained_fri(dataset).await?;
    }

    let records_row_addrs = if uses_stable_row_ids {
        records_stable_row_id_moves(dataset, &completed_tasks, options, needs_remapping).await?
    } else {
        options.defer_index_remap
    };

    // Confirm there is a remapper before materializing the potentially very large row address map.
    let index_remapper = if needs_remapping {
        remap_options.create_remapper(dataset).await?
    } else {
        None
    };
    // Determine the earliest version at which compaction tasks were planned/executed.
    //
    // In distributed mode (e.g. Spark) the caller opens *two separate* Dataset
    // handles: one for `plan_compaction` (at version V) and a fresh one for
    // `commit_compaction` (at the latest version V+N).  Using `dataset.manifest.version`
    // (= V+N) as the transaction's `read_version` would cause the conflict checker to
    // scan only versions after V+N — finding nothing — and therefore silently skip any
    // concurrent DELETE/UPDATE that landed between V and V+N, resurrecting deleted rows.
    //
    // By anchoring `read_version` to the minimum version carried in the RewriteResults
    // we ensure the conflict checker covers the full range [V, V+N] and will reject the
    // commit with a retryable conflict error if a concurrent write touched the same
    // fragments.
    let tasks_read_version = completed_tasks
        .iter()
        .map(|t| t.read_version)
        .min()
        .unwrap_or(dataset.manifest.version);

    if index_remapper.is_none() || !remap_options.supports_vector_segment_replacement() {
        let read_snapshot = if dataset.manifest.version == tasks_read_version {
            Cow::Borrowed(&*dataset)
        } else {
            Cow::Owned(dataset.checkout_version(tasks_read_version).await?)
        };
        validate_fragment_groups_preserve_index_boundaries(
            read_snapshot.as_ref(),
            completed_tasks
                .iter()
                .map(|task| task.original_fragments.as_slice()),
        )
        .await?;
    }

    let mut completed_tasks = completed_tasks;

    let new_fragment_count: usize = completed_tasks.iter().map(|t| t.new_fragments.len()).sum();

    // Single reserve_fragment_ids for all address-style tasks
    let mut reserved_version = None;
    if has_address_style {
        let frags: Vec<&mut Fragment> = completed_tasks
            .iter_mut()
            .filter(|t| t.row_addrs.is_some())
            .flat_map(|t| t.new_fragments.iter_mut())
            .collect();
        // A replayed result shares file paths with its earlier commit, so a failed
        // reservation cannot prove the files are unreferenced. Leave them for GC.
        reserved_version = Some(reserve_fragment_ids(dataset, frags.into_iter()).await?);
    }

    let mut rewrite_groups = Vec::with_capacity(completed_tasks.len());
    let mut metrics = CompactionMetrics::default();

    let mut remap_group_inputs: Vec<GroupInput> = Vec::new();
    let mut direct_row_id_map: HashMap<u64, Option<u64>> = HashMap::default();
    let mut frag_reuse_groups: Vec<FragReuseGroup> = Vec::new();
    let mut new_fragment_bitmap: RoaringBitmap = RoaringBitmap::new();

    // A deferred compaction records every group, including groups no committed
    // index covers: an index build in flight may have staged a segment over
    // those fragments, and only a recorded group lets that segment commit
    // past this rewrite (a bare rewrite of a covered fragment preempts it).
    // Trim reclaims such records once every index is disjoint from the chain
    // or postdates the rewrite.

    for task in completed_tasks {
        metrics += task.metrics;
        // One source of truth: the deferred branch normalizes the source
        // metadata (materialized physical and deleted row counts) and BOTH
        // the rewrite group and the reuse digests are built from it, so the
        // commit-side binding validation compares like with like.
        let old_fragments = if index_remapper.is_none() && records_row_addrs {
            normalize_source_fragments(dataset, &task.original_fragments).await?
        } else {
            task.original_fragments.clone()
        };
        let rewrite_group = RewriteGroup {
            old_fragments: old_fragments.clone(),
            new_fragments: task.new_fragments.clone(),
        };

        if index_remapper.is_some() {
            if let Some(row_addrs_bytes) = task.row_addrs {
                let row_addrs =
                    RoaringTreemap::deserialize_from(&mut Cursor::new(&row_addrs_bytes))?;
                match options.index_remap_mode {
                    IndexRemapMode::Direct => {
                        let transposed = remapping::transpose_row_addrs(
                            row_addrs,
                            &task.original_fragments,
                            &task.new_fragments,
                        );
                        direct_row_id_map.extend(transposed);
                    }
                    IndexRemapMode::Compact => {
                        let new_frags = task
                            .new_fragments
                            .iter()
                            .map(|f| {
                                let physical_rows = f.physical_rows.ok_or_else(|| {
                                    Error::invalid_input(format!(
                                        "compacted fragment {} is missing physical_rows",
                                        f.id
                                    ))
                                })?;
                                let fragment_id = u32::try_from(f.id).map_err(|_| {
                                    Error::invalid_input(format!(
                                        "compacted fragment id {} is outside the row-address range",
                                        f.id
                                    ))
                                })?;
                                let physical_rows = u32::try_from(physical_rows).map_err(|_| {
                                    Error::invalid_input(format!(
                                        "compacted fragment {} has physical_rows={} outside the row-address range",
                                        f.id, physical_rows
                                    ))
                                })?;
                                Ok((fragment_id, physical_rows))
                            })
                            .collect::<Result<Vec<_>>>()?;

                        remap_group_inputs.push(GroupInput {
                            rewritten_old_row_addrs: row_addrs,
                            old_frag_ids: task
                                .original_fragments
                                .iter()
                                .map(|f| {
                                    u32::try_from(f.id).map_err(|_| {
                                        Error::invalid_input(format!(
                                            "compacted source fragment id {} is outside the row-address range",
                                            f.id
                                        ))
                                    })
                                })
                                .collect::<Result<Vec<_>>>()?,
                            new_frags,
                        });
                    }
                }
            }
        } else if records_row_addrs {
            let changed_row_addrs = task.row_addrs.ok_or_else(|| {
                Error::internal(
                    "defer_index_remap requires row_addrs but none were provided".to_string(),
                )
            })?;
            // Digest construction must not panic on incomplete metadata (the
            // tasks may be replayed results planned elsewhere); the counts
            // were materialized by `normalize_source_fragments` above, from
            // the same list the rewrite group carries.
            let digest =
                |frag: &Fragment| -> Result<lance_table::system_index::frag_reuse::FragDigest> {
                    Ok(lance_table::system_index::frag_reuse::FragDigest {
                        id: frag.id,
                        physical_rows: frag.physical_rows.ok_or_else(|| {
                            Error::invalid_input(format!(
                                "fragment {} has no physical row count for the fragment reuse record",
                                frag.id
                            ))
                        })?,
                        num_deleted_rows: frag
                            .deletion_file
                            .as_ref()
                            .map(|deletion| {
                                deletion.num_deleted_rows.ok_or_else(|| {
                                    Error::invalid_input(format!(
                                        "fragment {} has a deletion file without a materialized row count",
                                        frag.id
                                    ))
                                })
                            })
                            .transpose()?
                            .unwrap_or(0),
                    })
                };
            frag_reuse_groups.push(FragReuseGroup {
                changed_row_addrs,
                old_frags: old_fragments.iter().map(digest).collect::<Result<_>>()?,
                new_frags: task
                    .new_fragments
                    .iter()
                    .map(digest)
                    .collect::<Result<_>>()?,
            });
            for frag in task.new_fragments.iter() {
                new_fragment_bitmap.insert(u32::try_from(frag.id).map_err(|_| {
                    Error::invalid_input(format!(
                        "compacted fragment id {} is outside the row-address range",
                        frag.id
                    ))
                })?);
            }
        }
        rewrite_groups.push(rewrite_group);
    }

    let rewritten_indices = if let Some(index_remapper) = index_remapper {
        let affected_ids = rewrite_groups
            .iter()
            .flat_map(|group| group.old_fragments.iter().map(|frag| frag.id))
            .collect::<Vec<_>>();

        let remap = match options.index_remap_mode {
            IndexRemapMode::Direct => RowAddrRemap::direct(direct_row_id_map),
            IndexRemapMode::Compact => RowAddrRemap::compact(remap_group_inputs)?,
        };
        let remapped_indices = index_remapper
            .remap_indices_with_context(remap, &affected_ids, &rewrite_groups)
            .await?;
        remapped_indices
            .into_iter()
            .map(|rewritten| RewrittenIndex {
                old_id: rewritten.old_id,
                new_id: rewritten.new_id,
                new_index_details: rewritten.index_details,
                new_index_version: rewritten.index_version,
                new_index_files: rewritten.files,
            })
            .collect()
    } else if !options.defer_index_remap && !has_address_style {
        // We need to reserve fragment ids here so that the fragment bitmap
        // can be updated for each index. Only needed for stable row IDs
        // since address-style IDs were already reserved above.
        let new_fragments = rewrite_groups
            .iter_mut()
            .flat_map(|group| group.new_fragments.iter_mut())
            .collect::<Vec<_>>();
        // As above, reservation failure does not make replayed files safe to delete.
        reserve_fragment_ids(dataset, new_fragments.into_iter()).await?;
        Vec::new()
    } else {
        Vec::new()
    };

    // A stable-row-id table always keeps a v0 record: recording is refused
    // before any rewrite where the table requires a tagged history.
    //
    // The record's FORM -- a v0 entry, or tagged
    // transitions riding the in-memory rewrite intent that
    // `build_frag_reuse_rewrite_entry` assembles onto the entry at every
    // commit attempt -- is decided HERE, at commit time, from the freshest
    // state this handle has (the id reservations above do not advance it), not
    // from a sample taken when the compaction started: another writer may
    // have tagged the table in between. The record's BASE, however, is the
    // snapshot at `tasks_read_version`: the commit path works out what the
    // entry adds by diffing it against the entry at the transaction's read
    // version, so an entry built from this later handle would fold every
    // transition other writers appended in between into this rewrite's own
    // additions and fail the commit for good instead of merging the
    // histories (`rewrite_reuse_state` refuses such an entry outright). The
    // read version itself stays at V: raising it to the latest version would
    // skip the concurrency checks against the writes that landed in between.
    // The remaining race onto the manifest CAS itself is handled by the
    // rebase, which converts a v0-shaped intent into tagged transitions when
    // the current entry turns out tagged (see `finish_rewrite`); the commit
    // gate still rejects a v0 entry spliced by a writer without that
    // conversion.
    let frag_reuse_index = if records_row_addrs {
        // Once a table is v1, it stays v1: the sticky
        // FLAG_FRAGMENT_REUSE_INDEX is the final authority
        // (`uses_tagged_fri`), never the entry's own index_version. With NO
        // entry -- for example a fully drained history whose entry a trim
        // deleted -- the flag alone decides, so a tagged table never
        // restarts its history in the v0 format; with a v0 entry UNDER the
        // flag (a legacy entry a concurrent upgrade left pending) the
        // tagged assembly lifts that entry byte-verbatim instead of
        // extending it in place. A table that never was tagged has neither
        // flag nor tagged entry, and keeps creating the v0 entry byte
        // identically.
        let stored = load_all_indices(dataset).await?;
        let tagged_at_commit = lance_table::system_index::frag_reuse::metadata::uses_tagged_fri(
            &dataset.manifest,
            stored.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME),
        );
        if tagged_at_commit {
            // Materializing a data overlay breaks the reuse premise that a
            // rewrite moves addresses, never values; on a tagged table the
            // stale-index handling that v0 applies (dropping destination
            // ids from swapped bitmaps) is a no-op because provenance never
            // contains the destinations. Refuse rather than record a
            // transition; see `build_frag_reuse_rewrite_entry` for the same
            // rule on the stable-partition path.
            if let Some(overlaid) = rewrite_groups
                .iter()
                .flat_map(|group| group.old_fragments.iter())
                .find(|frag| !frag.overlays.is_empty())
            {
                return Err(Error::not_supported(format!(
                    "source fragment {} carries data overlay files; deferred compaction on a \
                     tagged fragment reuse table would materialize the overlaid values while \
                     indices keep translated coverage over the old addresses. Compact the \
                     overlays away or rebuild the covering indices eagerly first",
                    overlaid.id
                )));
            }
            use lance_table::format::pb::fragment_reuse_index_details as pb_fri;
            let transitions = frag_reuse_groups
                .into_iter()
                .map(|group| pb_fri::Transition {
                    sources: group
                        .old_frags
                        .iter()
                        .map(pb_fri::FragmentDigest::from)
                        .collect(),
                    destinations: group
                        .new_frags
                        .iter()
                        .map(pb_fri::FragmentDigest::from)
                        .collect(),
                    mapping: Some(pb_fri::transition::Mapping::OrderedCompaction(
                        pb_fri::OrderedCompaction {
                            changed_row_addrs: group.changed_row_addrs,
                        },
                    )),
                })
                .collect();
            // The complete entry: the history at the read version plus this
            // compaction's own transitions, built from the same snapshot the
            // commit path diffs it against (see above). The commit path
            // merges the additions onto the entry current at commit.
            let read_snapshot = if dataset.manifest.version == tasks_read_version {
                Cow::Borrowed(&*dataset)
            } else {
                // A cleaned-up read version fails here, explicitly, rather
                // than guessing the base from a later snapshot.
                Cow::Owned(dataset.checkout_version(tasks_read_version).await?)
            };
            Some(
                crate::index::frag_reuse::frag_reuse_entry_appending(&read_snapshot, transitions)
                    .await?,
            )
        } else {
            // The rewrite publishes right after the reservation unless another commit
            // lands in between, in which case `finish_rewrite` restamps it.
            let dataset_version = reserved_version.unwrap_or(dataset.manifest.version);
            Some(
                build_new_frag_reuse_index(
                    dataset,
                    frag_reuse_groups,
                    new_fragment_bitmap,
                    dataset_version,
                )
                .await?,
            )
        }
    } else {
        if records_row_addrs {
            log::debug!(
                "skipping fragment reuse record: no rewritten fragments are covered by an index or the reuse lineage"
            );
        }
        None
    };

    let transaction = TransactionBuilder::new(
        // Use the version at which the compaction tasks were *planned*, not the
        // version of the dataset handle passed to this function.  In distributed
        // mode the caller may open a fresh dataset at a later version (V+N), but
        // the tasks were executed against an older snapshot (V).  Anchoring the
        // transaction to V ensures the OCC conflict checker scans all writes that
        // landed between V and the commit point, detecting concurrent DELETE
        // transactions that would otherwise cause deleted rows to reappear.
        tasks_read_version,
        Operation::Rewrite {
            groups: rewrite_groups,
            rewritten_indices,
            frag_reuse_index,
        },
    )
    .transaction_properties(options.transaction_properties.clone())
    .build();

    if let Err(e) = dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
    {
        // RewriteResult is serializable and may be retried after an earlier
        // ambiguous success. A conflict on this call therefore does not prove
        // that the rewritten files are unreferenced. Leave them for dataset GC.
        log::warn!(
            "Compaction commit failed; leaving {} rewritten fragment(s) in place for GC: {}",
            new_fragment_count,
            e
        );
        return Err(e);
    }

    Ok(metrics)
}

/// Materialize the counts the fragment reuse digests and conservation
/// validation depend on, without changing the fragment list. Unlike
/// `migrate_fragments`, fully-deleted fragments are KEPT: the rewrite
/// group must still remove them from the manifest and the recorded
/// transition must still consume them (they contribute zero live rows), so
/// a silently shortened list would desynchronize the digests, the group and
/// the manifest.
pub async fn normalize_source_fragments(
    dataset: &Dataset,
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
            let count = read_dataset_deletion_file(dataset, fragment.id, deletion_file)
                .await?
                .len();
            fragment.deletion_file.as_mut().unwrap().num_deleted_rows = Some(count);
        }
        normalized.push(fragment);
    }
    Ok(normalized)
}
