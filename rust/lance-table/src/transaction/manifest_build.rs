// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Applying an operation to produce the next manifest.
//!
//! [`Transaction::build_manifest`] is the centre of this module and of the
//! transaction machinery generally: given the current manifest and index list, it
//! decides the new fragment list, the surviving indices and the next row id, then
//! assembles the manifest. Everything else in `super` exists to serve it -- the
//! operation vocabulary it matches on, the index rules it applies, the row version
//! metadata it stamps, the validation that runs before it.

use crate::feature_flags::{
    FLAG_MEM_WAL_INDEX_CATCHUP, FLAG_STABLE_ROW_IDS, apply_feature_flags,
    inherit_mem_wal_index_catchup, validate_mem_wal_index_catchup_flags,
};
use crate::format::overlay::TOMBSTONE_FIELD_ID;
use crate::format::{
    DataFile, DataStorageFormat, Fragment, IndexMetadata, Manifest, ManifestBuildConfig,
    overlay::DataOverlayFile,
};
use crate::io::{
    commit::CommitHandler,
    manifest::{read_manifest, read_manifest_indexes},
};
use crate::rowids::version::build_version_meta;
use crate::system_index::is_system_index;
use crate::system_index::mem_wal::{
    CompactedSsTable, IndexCatchupProgress, MEM_WAL_INDEX_NAME, load_mem_wal_index_details,
    new_mem_wal_index_meta, update_mem_wal_index_compacted_sstables,
};
use crate::transaction::UpdateMode::{RewriteColumns, RewriteRows};
use crate::transaction::row_version::resolve_update_version_metadata;
use crate::transaction::update_map::apply_update_map;
use crate::transaction::validate::merge_fragment_physically_rewritten;
use crate::transaction::{
    CoverageIdentity, DataReplacementGroup, LogicalIndexSegments, Operation, ReadVersionState,
    RewriteGroup, Transaction, UpdatedFragmentOffsets,
};
use lance_core::datatypes::{
    LANCE_UNENFORCED_CLUSTERING_KEY_POSITION, LANCE_UNENFORCED_PRIMARY_KEY,
    LANCE_UNENFORCED_PRIMARY_KEY_POSITION,
};
use lance_core::{Error, Result};
use lance_file::version::ConcreteFileVersion;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use roaring::RoaringBitmap;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

impl Transaction {
    pub(super) fn fragments_with_ids<'a, T>(
        new_fragments: T,
        fragment_id: &'a mut u64,
    ) -> impl Iterator<Item = Fragment> + 'a
    where
        T: IntoIterator<Item = Fragment> + 'a,
    {
        new_fragments.into_iter().map(move |mut f| {
            if f.id == 0 {
                f.id = *fragment_id;
                *fragment_id += 1;
            }
            f
        })
    }

    fn data_storage_format_from_files(
        fragments: &[Fragment],
        user_requested: Option<ConcreteFileVersion>,
    ) -> Result<DataStorageFormat> {
        if let Some(file_version) = Fragment::try_infer_version(fragments)? {
            // Ensure user-requested matches data files
            if let Some(user_requested) = user_requested
                && user_requested != file_version
            {
                return Err(Error::invalid_input(format!(
                    "User requested data storage version ({}) does not match version in data files ({})",
                    user_requested, file_version
                )));
            }
            Ok(DataStorageFormat::new(file_version))
        } else {
            // If no files use user-requested or default
            Ok(user_requested
                .map(DataStorageFormat::new)
                .unwrap_or_default())
        }
    }

    pub async fn restore_old_manifest(
        object_store: &ObjectStore,
        commit_handler: &dyn CommitHandler,
        base_path: &Path,
        version: u64,
        config: &ManifestBuildConfig,
        tx_path: &str,
        current_manifest: &Manifest,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        let location = commit_handler
            .resolve_version_location(base_path, version, &object_store.inner)
            .await?;
        let mut manifest = read_manifest(object_store, &location.path, location.size).await?;
        // Read below the reader validation boundary, so nothing else refuses a
        // half-set manifest here: the flag reset would quietly drop the lone bit
        // and republish an undefined state as legacy.
        validate_mem_wal_index_catchup_flags(&manifest)?;
        manifest.set_timestamp(config.timestamp_nanos);
        manifest.transaction_file = Some(tx_path.to_string());
        let indices = read_manifest_indexes(object_store, &location, &manifest).await?;
        manifest.max_fragment_id = manifest
            .max_fragment_id
            .max(current_manifest.max_fragment_id);
        // A version from before catch-up was required carries MemWAL state this
        // protocol never validated -- catch-up values activation deliberately
        // cleared, or compaction progress it deliberately refused to trust.
        // Keeping the bit would republish those as if this protocol had recorded
        // them. Refuse instead: sanitizing is not possible here, because both
        // fields would have to be re-derived from data Lance cannot see.
        let current_requires = current_manifest.reader_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP
            != 0
            || current_manifest.writer_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP != 0;
        let restored_requires = manifest.reader_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP != 0
            && manifest.writer_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP != 0;
        if current_requires && !restored_requires {
            return Err(Error::invalid_input(format!(
                "Cannot restore version {version}: this table requires MemWAL index \
                 catch-up and that version predates it, so its recorded catch-up and \
                 compaction progress were never validated by this protocol"
            )));
        }
        inherit_mem_wal_index_catchup(&mut manifest, current_manifest)?;
        Ok((manifest, indices))
    }

    /// Require index catch-up on a table that has never required it.
    ///
    /// One-way, because returning to legacy semantics -- where a missing
    /// coverage entry reads as "fully caught up" -- is unsafe once any SSTable
    /// has been retired against a recorded catch-up position.
    fn require_index_catchup(final_indices: &mut [IndexMetadata], new_version: u64) -> Result<()> {
        let Some(pos) = final_indices
            .iter()
            .position(|idx| idx.name == MEM_WAL_INDEX_NAME)
        else {
            return Err(Error::invalid_input(format!(
                "Cannot require MemWAL index catch-up: the {} system index does \
                 not exist on this table",
                MEM_WAL_INDEX_NAME
            )));
        };

        let mut details = load_mem_wal_index_details(final_indices[pos].clone())?;

        // The beta protocol wrote compaction progress that was never an active
        // retirement record, and Lance cannot check those numbers against WAL
        // shard manifests. Trusting them would let the first trim after
        // activation delete SSTables no commit copied in, so a table carrying
        // them must be drained through an explicit migration instead.
        if !details.compacted_sstables.is_empty() {
            return Err(Error::invalid_input(
                "Cannot require MemWAL index catch-up: the table already records \
                 SSTable compaction progress from the beta protocol, which cannot \
                 be validated. Drain or reset the table first.",
            ));
        }

        // Beta coverage was written under rules this protocol does not enforce,
        // so it is not trustworthy. Left in place, a later compaction would find it
        // already satisfied and could retire an SSTable that no index covers.
        if details.index_catchup.is_empty() {
            return Ok(());
        }
        details.index_catchup.clear();
        final_indices[pos] = new_mem_wal_index_meta(new_version, details)?;
        Ok(())
    }

    /// Every non-system logical index, mapped to what determines its coverage.
    ///
    /// A logical index may be backed by several physical segments, so "did this
    /// index change" is a question about the whole set. Sorted by UUID so the
    /// two sides compare positionally.
    pub fn logical_index_segments(indices: &[IndexMetadata]) -> LogicalIndexSegments {
        let mut by_name: LogicalIndexSegments = BTreeMap::new();
        for idx in indices.iter().filter(|idx| !is_system_index(idx)) {
            by_name
                .entry(idx.name.clone())
                .or_default()
                .push(CoverageIdentity {
                    uuid: idx.uuid,
                    fragment_bitmap: idx.fragment_bitmap.clone(),
                });
        }
        for segments in by_name.values_mut() {
            segments.sort_unstable_by_key(|segment| segment.uuid);
        }
        by_name
    }

    /// Apply MemWAL index-coverage rules once the final index list is known.
    ///
    /// Coverage records that a base-table index contains the rows a compaction
    /// copied in, and the WAL pod retires SSTables against it.
    ///
    /// It is derived, not reported. An index covering every fragment live at the
    /// transaction's read version holds every row compaction had copied in by
    /// then, so it is caught up to that version's `compacted_sstables`. That is
    /// the only proof available: nothing maps a generation to the fragments its
    /// rows landed in, so covering the table as the transaction read it is how
    /// an index shows it covered those rows. Fragments appended since are a
    /// later gap.
    ///
    /// Deriving rather than transmitting means no claim can go stale between
    /// inspection and commit, the answer survives rebase (`read_version` is
    /// fixed for a transaction's life), and any operation can earn coverage --
    /// an ordinary reindex that fully covers no longer has to throw its work
    /// away and wait for a repair.
    ///
    /// Only meaningful once catch-up is required, where a missing entry means
    /// "not caught up" and the SSTables stay. A legacy table reads a missing
    /// entry as "fully caught up", so this leaves it untouched rather than
    /// making the table look more covered than it is.
    pub fn apply_mem_wal_index_coverage(
        final_indices: &mut [IndexMetadata],
        segments_before: &LogicalIndexSegments,
        read_version_state: Option<ReadVersionState<'_>>,
        index_catchup_required: bool,
        new_version: u64,
    ) -> Result<()> {
        if !index_catchup_required {
            return Ok(());
        }

        let Some(pos) = final_indices
            .iter()
            .position(|idx| idx.name == MEM_WAL_INDEX_NAME)
        else {
            // The system index went away with this transaction (MemWAL disable,
            // or an overwrite). There is no coverage left to maintain.
            return Ok(());
        };

        let mut details = load_mem_wal_index_details(final_indices[pos].clone())?;

        // Nothing has ever been compacted, so no index can be behind and there
        // is no coverage to invalidate.
        if details.compacted_sstables.is_empty() && details.index_catchup.is_empty() {
            return Ok(());
        }

        let segments_after = Self::logical_index_segments(final_indices);
        let catchup_before = std::mem::take(&mut details.index_catchup);

        // Per shard: what this commit records as compacted, and the most the
        // read version may credit. Generations compacted after that read landed
        // in fragments no index under consideration has seen; the committed
        // value caps it in turn, so a read version since rolled back cannot
        // retire SSTables no live commit copied in.
        let read_details = read_version_state
            .map(|state| {
                state
                    .indices
                    .iter()
                    .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
                    .cloned()
                    .map(load_mem_wal_index_details)
                    .transpose()
            })
            .transpose()?
            .flatten();
        let shards: Vec<(Uuid, u64, u64)> = details
            .compacted_sstables
            .iter()
            .map(|committed| {
                let at_read = read_details
                    .as_ref()
                    .and_then(|read| {
                        read.compacted_sstables
                            .iter()
                            .find(|s| s.shard_id == committed.shard_id)
                    })
                    .map_or(0, |s| s.generation);
                (
                    committed.shard_id,
                    committed.generation,
                    at_read.min(committed.generation),
                )
            })
            .collect();

        // Every fragment live when the transaction read the table. An index
        // spanning all of them holds every row compacted by then.
        let read_fragments: Option<RoaringBitmap> = read_version_state.map(|state| {
            state
                .manifest
                .fragments
                .iter()
                .map(|fragment| fragment.id as u32)
                .collect()
        });

        let covers_read_version = |segments: &[CoverageIdentity]| -> bool {
            let Some(required) = read_fragments.as_ref() else {
                return false;
            };
            if required.is_empty() {
                // Subset-of-empty is trivially true, so this would credit every
                // index on a table with no fragments. Refused because an empty
                // fragment list is not only what an emptied table looks like:
                // it is also what a manifest written before #8438 looks like,
                // where UpdateMemWalState published no fragments at all. On
                // such a table the SSTables are the last copy of those rows,
                // and crediting coverage would retire them. The cost is that a
                // genuinely emptied table keeps its SSTables.
                return false;
            }
            let mut covered = RoaringBitmap::new();
            for segment in segments {
                match segment.fragment_bitmap.as_ref() {
                    Some(bitmap) => covered |= bitmap,
                    // An unknown bitmap cannot be shown to cover anything.
                    None => return false,
                }
            }
            required.is_subset(&covered)
        };

        let mut rebuilt: Vec<IndexCatchupProgress> = Vec::new();
        for (name, after) in segments_after.iter() {
            // Compared by [`CoverageIdentity`], not segment UUID: an Update
            // that touches an indexed field prunes a segment's fragment bitmap
            // in place while keeping its UUID, so a UUID-only comparison would
            // carry a position forward that the index no longer earns.
            let unchanged = segments_before.get(name) == Some(after);
            let carried = unchanged
                .then(|| catchup_before.iter().find(|e| e.index_name == *name))
                .flatten();
            let proven = covers_read_version(after);

            if carried.is_none() && !proven {
                // Changed, and nothing shows the new index covers the read
                // version. No entry: a missing one reads as "not caught up".
                continue;
            }

            let generations = shards
                .iter()
                .map(|&(shard_id, committed, creditable)| {
                    let prior = carried
                        .and_then(|entry| entry.caught_up_generation_for_shard(&shard_id))
                        .unwrap_or(0);
                    let credited = if proven { creditable } else { 0 };
                    // Takes the better of what this commit proves and what an
                    // unchanged index already held, so a commit reading an older
                    // version does not lower a position it cannot re-prove. The
                    // clamp is the exception: a position above what this commit
                    // records as compacted describes rows no live commit copied
                    // in.
                    CompactedSsTable::new(shard_id, prior.max(credited).min(committed))
                })
                .collect::<Vec<_>>();
            if generations.iter().all(|g| g.generation == 0) {
                continue;
            }
            rebuilt.push(IndexCatchupProgress::new(name.clone(), generations));
        }
        rebuilt.sort_by(|a, b| a.index_name.cmp(&b.index_name));

        let mut before_sorted = catchup_before;
        before_sorted.sort_by(|a, b| a.index_name.cmp(&b.index_name));
        if rebuilt == before_sorted {
            return Ok(());
        }

        let dropped: Vec<&str> = before_sorted
            .iter()
            .map(|e| e.index_name.as_str())
            .filter(|name| !rebuilt.iter().any(|kept| kept.index_name == *name))
            .collect();
        if !dropped.is_empty() {
            // The first thing to check when SSTables stop becoming trimmable.
            log::info!(
                "MemWAL index catch-up invalidated at version {new_version} for {dropped:?}: \
                 these indices changed and no longer cover the version this commit read"
            );
        }

        details.index_catchup = rebuilt;
        final_indices[pos] = new_mem_wal_index_meta(new_version, details)?;
        Ok(())
    }

    /// Drop coverage for indices a post-`build_manifest` step narrowed.
    ///
    /// The derivation runs while the manifest is being built, but the index list
    /// is not final there: `migrate_indices` can recalculate a segment's
    /// fragment bitmap and keep its UUID, so an index can narrow after its
    /// position was decided. It reports which ones it touched rather than the
    /// caller re-snapshotting every bitmap to find out. Only ever removes.
    pub fn withdraw_coverage_invalidated_after_build(
        indices: &mut [IndexMetadata],
        changed: &[String],
        new_version: u64,
    ) -> Result<()> {
        if changed.is_empty() {
            return Ok(());
        }
        let Some(pos) = indices
            .iter()
            .position(|idx| idx.name == MEM_WAL_INDEX_NAME)
        else {
            return Ok(());
        };
        let mut details = load_mem_wal_index_details(indices[pos].clone())?;
        let before = details.index_catchup.len();
        details
            .index_catchup
            .retain(|entry| !changed.contains(&entry.index_name));
        if details.index_catchup.len() == before {
            return Ok(());
        }
        log::info!(
            "MemWAL index catch-up withdrawn at version {new_version} for {changed:?}: \
             these indices were recalculated after their coverage was derived"
        );
        indices[pos] = new_mem_wal_index_meta(new_version, details)?;
        Ok(())
    }

    /// Create a new manifest from the current manifest and the transaction.
    ///
    /// `current_manifest` should only be None if the dataset does not yet exist.
    pub fn build_manifest(
        &self,
        current_manifest: Option<&Manifest>,
        current_indices: Vec<IndexMetadata>,
        transaction_file_path: &str,
        config: &ManifestBuildConfig,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        self.build_manifest_with_read_version(
            current_manifest,
            current_indices,
            transaction_file_path,
            config,
            None,
        )
    }

    /// [`Self::build_manifest`] with the version this transaction read.
    ///
    /// Supplied by the commit path, which already materializes that version.
    /// `None` where there is none to read -- dataset creation and detached
    /// commits -- in which case no index can be shown to cover it and coverage
    /// is left as the invalidation rules put it.
    pub fn build_manifest_with_read_version(
        &self,
        current_manifest: Option<&Manifest>,
        current_indices: Vec<IndexMetadata>,
        transaction_file_path: &str,
        config: &ManifestBuildConfig,
        read_version_state: Option<ReadVersionState<'_>>,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        if config.use_stable_row_ids
            && config.migration_next_row_id.is_none()
            && current_manifest
                .map(|m| !m.uses_stable_row_ids())
                .unwrap_or_default()
        {
            return Err(Error::not_supported_source(
                "This dataset was not created with the stable row ids feature.  Please run `migrate_to_stable_row_ids` before attempting to use stable row ids".into(),
            ));
        }

        if config.migration_next_row_id.is_some() && !current_indices.is_empty() {
            let names: Vec<&str> = current_indices
                .iter()
                .map(|idx| idx.name.as_str())
                .collect();
            return Err(Error::invalid_input(format!(
                "Cannot migrate to stable row IDs while indexes exist on the dataset. \
                 Drop the following indexes first, then re-run the migration, and \
                 recreate them afterwards: {}",
                names.join(", ")
            )));
        }
        let mut reference_paths = match current_manifest {
            Some(m) => m.base_paths.clone(),
            None => HashMap::new(),
        };

        if let Operation::Overwrite {
            initial_bases: Some(initial_bases),
            ..
        } = &self.operation
        {
            if current_manifest.is_none() {
                // CREATE mode: registering base paths
                // Base IDs should have been assigned during write operation
                // Validate uniqueness and insert them into the manifest
                for base_path in initial_bases.iter() {
                    if reference_paths.contains_key(&base_path.id) {
                        return Err(Error::invalid_input(format!(
                            "Duplicate base path ID {} detected. Base path IDs must be unique.",
                            base_path.id
                        )));
                    }
                    reference_paths.insert(base_path.id, base_path.clone());
                }
            } else {
                // OVERWRITE mode with initial_bases should have been rejected by validation
                // This branch should never be reached
                return Err(Error::invalid_input(
                    "OVERWRITE mode cannot register new bases. This should have been caught by validation.",
                ));
            }
        }

        // Get the schema and the final fragment list
        let schema = match self.operation {
            Operation::Overwrite { ref schema, .. } => schema.clone(),
            Operation::Merge { ref schema, .. } => schema.clone(),
            Operation::Project { ref schema, .. } => schema.clone(),
            _ => {
                if let Some(current_manifest) = current_manifest {
                    current_manifest.schema.clone()
                } else {
                    return Err(Error::internal(
                        "Cannot create a new dataset without a schema".to_string(),
                    ));
                }
            }
        };

        // Fragment ids are a high water mark for the whole dataset history: an id
        // must never name two different sets of rows, or per-fragment state keyed
        // by id (caches, deletion files, row addresses) can be attributed to the
        // wrong rows.
        let mut fragment_id = current_manifest
            .and_then(|m| m.max_fragment_id())
            .map(|id| id + 1)
            .unwrap_or(0);
        let mut final_fragments = Vec::new();
        let mut final_indices = current_indices;

        // Both words must agree: a reader that keeps legacy semantics would read a
        // missing entry as "fully caught up", so a half-set state is not safe mode.
        let index_catchup_required = current_manifest
            .map(|m| {
                m.reader_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP != 0
                    && m.writer_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP != 0
            })
            .unwrap_or(false);

        // Snapshot taken before the operation rewrites the list, so coverage can
        // be compared against what each logical index looked like going in. Only
        // tables in safe mode maintain coverage, so every other commit -- and the
        // segment clones this costs -- pays nothing.
        let mem_wal_segments_before = (index_catchup_required
            && final_indices
                .iter()
                .any(|idx| idx.name == MEM_WAL_INDEX_NAME))
        .then(|| Self::logical_index_segments(&final_indices));

        let mut next_row_id = {
            // Only use row ids if the feature flag is set already, or this is
            // a migration activation that explicitly provides the next_row_id.
            match (current_manifest, config.use_stable_row_ids) {
                (Some(manifest), _) if manifest.reader_feature_flags & FLAG_STABLE_ROW_IDS != 0 => {
                    Some(manifest.next_row_id)
                }
                (None, true) => Some(0),
                (_, false) => None,
                (Some(_), true) => {
                    // Migration activation: use the provided next_row_id.
                    if let Some(migration_nri) = config.migration_next_row_id {
                        Some(migration_nri)
                    } else {
                        return Err(Error::not_supported_source(
                            "This dataset was not created with the stable row ids feature.  Please run `migrate_to_stable_row_ids` before attempting to use stable row ids".into(),
                        ));
                    }
                }
            }
        };

        let maybe_existing_fragments =
            current_manifest
                .map(|m| m.fragments.as_ref())
                .ok_or_else(|| {
                    Error::internal(format!(
                        "No current manifest was provided while building manifest for operation {}",
                        self.operation.name()
                    ))
                });

        let new_version = current_manifest.map_or(1, |m| m.version + 1);

        match &self.operation {
            Operation::Clone { .. } => {
                return Err(Error::internal(
                    "Clone operation should not enter build_manifest.".to_string(),
                ));
            }
            Operation::Append { fragments } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
                let mut new_fragments =
                    Self::fragments_with_ids(fragments.clone(), &mut fragment_id)
                        .collect::<Vec<_>>();
                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                    // Add version metadata for all new fragments
                    for fragment in new_fragments.iter_mut() {
                        let version_meta = build_version_meta(fragment, new_version);
                        fragment.last_updated_at_version_meta = version_meta.clone();
                        fragment.created_at_version_meta = version_meta;
                    }
                }
                final_fragments.extend(new_fragments);
            }
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                ..
            } => {
                // Remove the deleted fragments
                // Hash lookups keep this linear on tables with many fragments.
                let deleted_ids: HashSet<u64> = deleted_fragment_ids.iter().copied().collect();
                let updated_by_id: HashMap<u64, &Fragment> =
                    updated_fragments.iter().map(|f| (f.id, f)).collect();
                final_fragments.extend(maybe_existing_fragments?.clone());
                final_fragments.retain(|f| !deleted_ids.contains(&f.id));
                final_fragments.iter_mut().for_each(|f| {
                    if let Some(updated) = updated_by_id.get(&f.id) {
                        *f = (*updated).clone();
                    }
                });
                Self::retain_relevant_indices(&mut final_indices, &schema, &final_fragments)
            }
            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified,
                compacted_sstables,
                fields_for_preserving_frag_bitmap,
                update_mode,
                updated_fragment_offsets,
                ..
            } => {
                // Extract existing fragments once for reuse
                let existing_fragments = maybe_existing_fragments?;

                // Apply updates to existing fragments
                // Hash lookups keep this linear on tables with many fragments.
                let removed_ids: HashSet<u64> = removed_fragment_ids.iter().copied().collect();
                let mut updated_by_id: HashMap<u64, &Fragment> =
                    HashMap::with_capacity(updated_fragments.len());
                for fragment in updated_fragments {
                    updated_by_id.entry(fragment.id).or_insert(fragment);
                }
                let updated_frags: Vec<Fragment> = existing_fragments
                    .iter()
                    .filter_map(|f| {
                        if removed_ids.contains(&f.id) {
                            return None;
                        }
                        if let Some(&updated) = updated_by_id.get(&f.id) {
                            let mut updated = updated.clone();
                            // Carry forward the fragment's current overlays (which
                            // may include ones added by a concurrent commit). An
                            // in-place column rewrite then tombstones the overlaid
                            // fields it rewrote, since the fresh base values
                            // supersede them.
                            updated.overlays = f.overlays.clone();
                            if matches!(update_mode, Some(RewriteColumns)) {
                                crate::format::overlay::tombstone_overlay_fields(
                                    &mut updated.overlays,
                                    fields_modified,
                                );
                            }
                            Some(updated)
                        } else {
                            Some(f.clone())
                        }
                    })
                    .collect();

                // Update version metadata for updated fragments if stable row IDs are enabled
                // Note: We don't update version metadata for fragments with deletion vectors
                // because the version sequences are indexed by physical row position, not logical position.
                // Version metadata for deleted rows will be filtered out during scan using the deletion vector.
                if next_row_id.is_some() {
                    // Version metadata will be properly set during compaction when deletions are materialized
                }

                final_fragments.extend(updated_frags);

                if next_row_id.is_some()
                    && matches!(update_mode, Some(RewriteColumns))
                    && let Some(UpdatedFragmentOffsets(off_map)) = updated_fragment_offsets
                    && !off_map.is_empty()
                {
                    let prev_version = current_manifest.map(|m| m.version).unwrap_or(0);
                    for fragment in final_fragments.iter_mut() {
                        let Some(bitmap) = off_map.get(&fragment.id) else {
                            continue;
                        };
                        // Defense-in-depth: only stamp fragments that were actually
                        // rewritten. validate_operation enforces this invariant before
                        // build_manifest is called; this guard catches any path that
                        // bypasses validation.
                        if !updated_by_id.contains_key(&fragment.id) {
                            continue;
                        }
                        if bitmap.is_empty() {
                            continue;
                        }
                        // Skip fragments with no existing version metadata: the helper
                        // would fill unmatched rows with prev_version, fabricating a
                        // last_updated stamp for rows that never had one.
                        if fragment.last_updated_at_version_meta.is_none() {
                            continue;
                        }
                        let max_allowed = existing_fragments
                            .iter()
                            .find(|f| f.id == fragment.id)
                            .and_then(|f| f.physical_rows)
                            .unwrap_or(1 << 24);
                        if bitmap.len() as usize > max_allowed {
                            return Err(Error::invalid_input(format!(
                                "updatedFragmentOffsets cardinality {} exceeds fragment {} limit {}",
                                bitmap.len(),
                                fragment.id,
                                max_allowed
                            )));
                        }
                        if let Some(max_off) = bitmap.max()
                            && max_off as usize >= max_allowed
                        {
                            return Err(Error::invalid_input(format!(
                                "updatedFragmentOffsets max offset {} exceeds fragment {} limit {}",
                                max_off, fragment.id, max_allowed
                            )));
                        }
                        let offsets: Vec<usize> = bitmap.iter().map(|o| o as usize).collect();
                        crate::rowids::version::refresh_row_latest_update_meta_for_partial_frag_rewrite_cols(
                            fragment,
                            &offsets,
                            new_version,
                            prev_version,
                        )?;
                    }
                }

                // If we updated any fields, remove those fragments from indices covering those fields
                Self::prune_updated_fields_from_indices(
                    &mut final_indices,
                    updated_fragments,
                    fields_modified,
                );

                let mut new_fragments =
                    Self::fragments_with_ids(new_fragments.clone(), &mut fragment_id)
                        .collect::<Vec<_>>();

                // Assign row IDs to any fragments that don't have them yet
                // (e.g., inserted rows from merge_insert operations)
                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                }

                if next_row_id.is_some() {
                    resolve_update_version_metadata(
                        existing_fragments,
                        new_fragments.as_mut_slice(),
                        new_version,
                    )?;
                }

                if config.use_stable_row_ids
                    && update_mode.is_some()
                    && *update_mode == Some(RewriteRows)
                {
                    let pure_updated_frag_ids =
                        Self::collect_pure_rewrite_row_update_frags_ids(&new_fragments)?;

                    // collect all the original frag ids that contains the updated rows
                    let original_fragment_ids: Vec<u64> = removed_fragment_ids
                        .iter()
                        .chain(updated_fragments.iter().map(|f| &f.id))
                        .copied()
                        .collect();

                    // The original fragments that carried an overlay: their moved rows may have a
                    // stale index entry (see `register_pure_rewrite_rows_update_frags_in_indices`).
                    // Reuse the hash lookups built above instead of scanning
                    // `original_fragment_ids` per fragment.
                    let original_overlaid_frags: HashMap<u32, &Fragment> = existing_fragments
                        .iter()
                        .filter(|f| {
                            (removed_ids.contains(&f.id) || updated_by_id.contains_key(&f.id))
                                && !f.overlays.is_empty()
                        })
                        .map(|f| (f.id as u32, f))
                        .collect();

                    Self::register_pure_rewrite_rows_update_frags_in_indices(
                        &mut final_indices,
                        &pure_updated_frag_ids,
                        &original_fragment_ids,
                        fields_for_preserving_frag_bitmap,
                        &original_overlaid_frags,
                        &schema,
                    )?;
                }

                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                    // Note: Version metadata is already set above (lines 1627-1755)
                    // for Update operations, preserving created_at from original fragments.
                    // Don't overwrite it here.
                }
                // Identify fragments that were updated or newly created in this update
                let mut target_ids: HashSet<u64> = HashSet::new();
                target_ids.extend(new_fragments.iter().map(|f| f.id));
                final_fragments.extend(new_fragments);
                Self::retain_relevant_indices(&mut final_indices, &schema, &final_fragments);

                if !compacted_sstables.is_empty() {
                    update_mem_wal_index_compacted_sstables(
                        &mut final_indices,
                        new_version,
                        compacted_sstables.clone(),
                    )?;
                }
            }
            Operation::Overwrite { fragments, .. } => {
                // Every fragment in an overwrite is newly written, so all of them
                // take fresh ids regardless of the id they arrive with. Fragments
                // carried over from the dataset being replaced are rejected by
                // `validate_operation`, which is what makes ignoring the incoming
                // id safe here.
                let mut new_fragments = fragments.clone();
                for fragment in new_fragments.iter_mut() {
                    fragment.id = fragment_id;
                    fragment_id += 1;
                }
                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                    // Add version metadata for all new fragments
                    for fragment in new_fragments.iter_mut() {
                        let version_meta = build_version_meta(fragment, new_version);
                        fragment.last_updated_at_version_meta = version_meta.clone();
                        fragment.created_at_version_meta = version_meta;
                    }
                }
                final_fragments.extend(new_fragments);
                final_indices = Vec::new();
            }
            Operation::Rewrite {
                groups,
                rewritten_indices,
                frag_reuse_index,
            } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
                let current_version = current_manifest.map(|m| m.version).unwrap_or_default();
                Self::handle_rewrite_fragments(
                    &mut final_fragments,
                    groups,
                    &mut fragment_id,
                    current_version,
                    next_row_id.as_ref(),
                )?;

                if next_row_id.is_some() {
                    // We can re-use indices, but need to rewrite the fragment bitmaps
                    debug_assert!(rewritten_indices.is_empty());
                    for index in final_indices.iter_mut() {
                        let results_are_row_addrs = index.results_are_row_addrs();
                        if let Some(fragment_bitmap) = &mut index.fragment_bitmap {
                            *fragment_bitmap = if results_are_row_addrs {
                                // Stable row ids survive a rewrite, so a row-id-domain index
                                // can simply follow its data to the new fragments. An
                                // address-domain index cannot: its stored addresses point into
                                // the fragments the rewrite dropped. Claiming coverage of the
                                // new fragments would make it answer queries with addresses
                                // that no longer resolve, so drop the rewritten fragments from
                                // its coverage instead and let the scanner fall back to a full
                                // scan for them.
                                Self::drop_rewritten_fragments(fragment_bitmap, groups)
                            } else {
                                Self::recalculate_fragment_bitmap(fragment_bitmap, groups)?
                            };
                        }
                    }
                } else {
                    Self::handle_rewrite_indices(&mut final_indices, rewritten_indices, groups)?;
                }

                // A full compaction materializes a fragment's overlays into fresh
                // base data. Any index older than one of those overlays was built on
                // the pre-overlay values, so drop the rewritten fragment from its
                // coverage to keep it from serving stale values.
                Self::prune_overlay_stale_fields_from_indices(&mut final_indices, groups);

                if let Some(frag_reuse_index) = frag_reuse_index {
                    final_indices.retain(|idx| idx.name != frag_reuse_index.name);
                    final_indices.push(frag_reuse_index.clone());
                }
            }
            Operation::CreateIndex {
                new_indices,
                removed_indices,
                ..
            } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
                let removed_uuids = removed_indices
                    .iter()
                    .map(|old_index| old_index.uuid)
                    .collect::<HashSet<_>>();
                let new_uuids = new_indices
                    .iter()
                    .map(|new_index| new_index.uuid)
                    .collect::<HashSet<_>>();
                final_indices.retain(|existing_index| {
                    !removed_uuids.contains(&existing_index.uuid)
                        && !new_uuids.contains(&existing_index.uuid)
                });
                final_indices.extend(new_indices.clone());
            }
            Operation::ReserveFragments { .. } | Operation::UpdateConfig { .. } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
            }
            Operation::Merge { fragments, .. } => {
                let existing_fragments = maybe_existing_fragments?;
                let mut merged_fragments = fragments.clone();
                if next_row_id.is_some() {
                    let prev_by_id: HashMap<u64, &Fragment> =
                        existing_fragments.iter().map(|f| (f.id, f)).collect();
                    for fragment in merged_fragments.iter_mut() {
                        match prev_by_id.get(&fragment.id) {
                            Some(prev) => {
                                if merge_fragment_physically_rewritten(prev, fragment) {
                                    crate::rowids::version::refresh_row_latest_update_meta_for_full_frag_rewrite_cols(
                                        fragment,
                                        new_version,
                                    )?;
                                }
                            }
                            None => {
                                // Brand-new fragment ID not present in the previous manifest.
                                // Set both last_updated and created version meta, consistent
                                // with Append/Overwrite for genuinely new fragments.
                                crate::rowids::version::refresh_row_latest_update_meta_for_full_frag_rewrite_cols(
                                    fragment,
                                    new_version,
                                )?;
                                fragment.created_at_version_meta =
                                    fragment.last_updated_at_version_meta.clone();
                            }
                        }
                    }
                }
                final_fragments.extend(merged_fragments);

                // A Merge can rewrite a column's data file in place; the field stays
                // in the schema, so the index is retained -- prune its now-stale
                // entries for the rewritten fragments.
                Self::prune_merge_rewritten_fields_from_indices(
                    &mut final_indices,
                    existing_fragments,
                    fragments,
                );

                // Some fields that have indices may have been removed, so we should
                // remove those indices as well.
                Self::retain_relevant_indices(&mut final_indices, &schema, &final_fragments)
            }
            Operation::Project { .. } => {
                final_fragments.extend(maybe_existing_fragments?.clone());

                // We might have removed all fields for certain data files, so
                // we should remove the data files that are no longer relevant.
                let remaining_field_ids = schema
                    .fields_pre_order()
                    .map(|f| f.id)
                    .collect::<HashSet<_>>();
                for fragment in final_fragments.iter_mut() {
                    fragment.files.retain(|file| {
                        file.fields
                            .iter()
                            .any(|field_id| remaining_field_ids.contains(field_id))
                    });
                }

                // Some fields that have indices may have been removed, so we should
                // remove those indices as well.
                Self::retain_relevant_indices(&mut final_indices, &schema, &final_fragments)
            }
            Operation::Restore { .. } => {
                unreachable!()
            }
            Operation::DataReplacement { replacements } => {
                log::warn!(
                    "Building manifest with DataReplacement operation. This operation is not stable yet, please use with caution."
                );

                let (old_fragment_ids, new_datafiles): (Vec<&u64>, Vec<&DataFile>) = replacements
                    .iter()
                    .map(|DataReplacementGroup(fragment_id, new_file)| (fragment_id, new_file))
                    .unzip();

                // 1. make sure the new files all have the same fields / or empty
                // NOTE: arguably this requirement could be relaxed in the future
                // for the sake of simplicity, we require the new files to have the same fields
                if new_datafiles
                    .iter()
                    .map(|f| f.fields.clone())
                    .collect::<HashSet<_>>()
                    .len()
                    > 1
                {
                    let field_info = new_datafiles
                        .iter()
                        .enumerate()
                        .map(|(id, f)| (id, f.fields.clone()))
                        .fold("".to_string(), |acc, (id, fields)| {
                            format!("{}File {}: {:?}\n", acc, id, fields)
                        });

                    return Err(Error::invalid_input(format!(
                        "All new data files must have the same fields, but found different fields:\n{field_info}"
                    )));
                }

                let existing_fragments = maybe_existing_fragments?;

                // Collect replaced field IDs before consuming new_datafiles
                let replaced_fields: Vec<u32> = new_datafiles
                    .first()
                    .map(|f| {
                        f.fields
                            .iter()
                            .filter(|&&id| id >= 0)
                            .map(|&id| id as u32)
                            .collect()
                    })
                    .unwrap_or_default();

                // 2. check that the fragments being modified have isomorphic layouts along the columns being replaced
                // 3. add modified fragments to final_fragments
                for (frag_id, new_file) in old_fragment_ids.iter().zip(new_datafiles) {
                    let frag = existing_fragments
                        .iter()
                        .find(|f| f.id == **frag_id)
                        .ok_or_else(|| {
                            Error::invalid_input(
                                "Fragment being replaced not found in existing fragments",
                            )
                        })?;
                    let mut new_frag = frag.clone();

                    // TODO(rmeng): check new file and fragment are the same length

                    let mut columns_covered = HashSet::new();
                    // Set when an existing file covers exactly the replaced
                    // fields, so the whole file swaps rather than part of it.
                    let mut replaced_in_place = false;
                    for file in &mut new_frag.files {
                        if file.fields == new_file.fields
                            && file.file_major_version == new_file.file_major_version
                            && file.file_minor_version == new_file.file_minor_version
                        {
                            // assign the new file path / size / base to the fragment
                            file.path = new_file.path.clone();
                            file.file_size_bytes = new_file.file_size_bytes.clone();
                            file.base_id = new_file.base_id;
                            replaced_in_place = true;
                        }
                        columns_covered.extend(file.fields.iter());
                    }
                    // Reject a file whose version does not decode before any
                    // arm publishes it.
                    new_file.file_version()?;

                    // SPECIAL CASE: if the column(s) being replaced are not covered by the fragment
                    // Then it means it's a all-NULL column that is being replaced with real data
                    // just add it to the final fragments. Push the DataFile as
                    // given so every field (including base_id) is preserved.
                    if columns_covered.is_disjoint(&new_file.fields.iter().collect()) {
                        new_frag.files.push(new_file.clone());
                    } else if !replaced_in_place
                        && new_file.fields.iter().all(|field| {
                            let mut covering = new_frag
                                .files
                                .iter()
                                .filter(|file| file.fields.contains(field))
                                .peekable();
                            // Covered by something, and by nothing we cannot
                            // tombstone. A field no file covers leaves the
                            // mixed layout the error below reports.
                            covering.peek().is_some()
                                && covering.all(|file| {
                                    file.file_version()
                                        .is_ok_and(|version| version != ConcreteFileVersion::V1)
                                })
                        })
                    {
                        // Tombstone the replaced fields where they live and
                        // append the new file to answer for them, the idiom
                        // `update_columns` uses. Compaction decides that layout,
                        // so the fields may sit in one wider file or span
                        // several.
                        //
                        // Legacy V1 is excluded: its reader derives the page table
                        // offset from the first field in the metadata, so
                        // tombstoning one field leaves its siblings decoding from
                        // the wrong pages. A field a V1 file covers keeps
                        // exact-match replacement.
                        for file in &mut new_frag.files {
                            // Same reason as the guard above.
                            if file.file_version()? == ConcreteFileVersion::V1 {
                                continue;
                            }
                            file.fields = file
                                .fields
                                .iter()
                                .map(|field| {
                                    if new_file.fields.contains(field) {
                                        TOMBSTONE_FIELD_ID
                                    } else {
                                        *field
                                    }
                                })
                                .collect::<Vec<_>>()
                                .into();
                        }
                        // Every data file must share at least one field with
                        // the dataset schema: a file kept alive only by
                        // tombstones or by ids the schema no longer defines is
                        // unreachable to readers, uncollectable by cleanup,
                        // and reported corrupt by validate().
                        let live_ids = schema
                            .fields_pre_order()
                            .map(|field| field.id)
                            .collect::<HashSet<i32>>();
                        new_frag
                            .files
                            .retain(|file| file.fields.iter().any(|f| live_ids.contains(f)));
                        new_frag.files.push(new_file.clone());
                    }

                    // Nothing changed in the current fragment, which is not expected -- error out
                    if &new_frag == frag {
                        return Err(Error::invalid_input(
                            "Expected to modify the fragment but no changes were made. This means the new data files does not align with any exiting datafiles. Please check if the schema of the new data files matches the schema of the old data files including the file major and minor versions",
                        ));
                    }

                    // New base values supersede any overlay still shadowing
                    // them, so tombstone the overlaid fields. An overlay
                    // committed after this transaction's snapshot is the newer
                    // value though -- the conflict resolver rebases these two
                    // precisely because the overlay wins -- so it stays, and
                    // being newer it stays last, preserving the ordering.
                    let (mut superseded, newer): (Vec<_>, Vec<_>) = new_frag
                        .overlays
                        .drain(..)
                        .partition(|overlay| overlay.committed_version <= self.read_version);
                    crate::format::overlay::tombstone_overlay_fields(
                        &mut superseded,
                        &replaced_fields,
                    );
                    superseded.extend(newer);
                    new_frag.overlays = superseded;

                    final_fragments.push(new_frag);
                }

                let fragments_changed = old_fragment_ids
                    .iter()
                    .cloned()
                    .cloned()
                    .collect::<HashSet<_>>();

                // 4. push fragments that didn't change back to final_fragments
                let unmodified_fragments = existing_fragments
                    .iter()
                    .filter(|f| !fragments_changed.contains(&f.id))
                    .cloned()
                    .collect::<Vec<_>>();

                final_fragments.extend(unmodified_fragments);

                // 5. Invalidate index bitmaps for replaced fields
                let modified_fragments: Vec<Fragment> = final_fragments
                    .iter()
                    .filter(|f| fragments_changed.contains(&f.id))
                    .cloned()
                    .collect();

                // A replacement changes what its rows read as, so stamp them
                // updated. Without this, get_updated_rows never reports them and
                // an incremental consumer skips them for good.
                if next_row_id.is_some() {
                    let new_version = current_manifest.map_or(1, |m| m.version + 1);
                    for fragment in final_fragments
                        .iter_mut()
                        .filter(|f| fragments_changed.contains(&f.id))
                    {
                        crate::rowids::version::refresh_row_latest_update_meta_for_full_frag_rewrite_cols(
                            fragment,
                            new_version,
                        )?;
                    }
                }

                Self::prune_updated_fields_from_indices(
                    &mut final_indices,
                    &modified_fragments,
                    &replaced_fields,
                );
            }
            Operation::DataOverlay { groups } => {
                // Stamp each overlay with the version this commit is producing.
                // build_manifest re-runs on every retry with an updated
                // current_manifest, so this is naturally re-stamped on retry.
                let new_version = current_manifest.map_or(1, |m| m.version + 1);

                let existing_fragments = maybe_existing_fragments?;
                // Multiple groups may target the same fragment; merge them in
                // order rather than letting a HashMap collapse drop all but the
                // last group's overlays.
                let mut overlays_by_fragment: HashMap<u64, Vec<&DataOverlayFile>> = HashMap::new();
                for group in groups {
                    overlays_by_fragment
                        .entry(group.fragment_id)
                        .or_default()
                        .extend(group.overlays.iter());
                }

                // Every group must target an existing fragment. Build a set of
                // existing ids once so this is O(groups + fragments) rather than
                // O(groups * fragments).
                let existing_fragment_ids: HashSet<u64> =
                    existing_fragments.iter().map(|f| f.id).collect();
                for fragment_id in overlays_by_fragment.keys() {
                    if !existing_fragment_ids.contains(fragment_id) {
                        return Err(Error::invalid_input(format!(
                            "DataOverlay targets fragment {fragment_id}, which does not exist"
                        )));
                    }
                }

                for fragment in existing_fragments {
                    let mut fragment = fragment.clone();
                    if let Some(new_overlays) = overlays_by_fragment.get(&fragment.id) {
                        // Appended (not replaced) so concurrently-written overlays
                        // survive; later entries are newer.
                        fragment
                            .overlays
                            .extend(new_overlays.iter().map(|&overlay| {
                                let mut overlay = overlay.clone();
                                overlay.committed_version = new_version;
                                overlay
                            }));
                    }
                    final_fragments.push(fragment);
                }
            }
            Operation::UpdateMemWalState {
                compacted_sstables, ..
            } => {
                // Updates the MemWAL index only; the fragments are unchanged.
                final_fragments.extend(maybe_existing_fragments?.clone());
                update_mem_wal_index_compacted_sstables(
                    &mut final_indices,
                    new_version,
                    compacted_sstables.clone(),
                )?;
            }
            Operation::UpdateBases { .. } => {
                // UpdateBases operation doesn't modify fragments or indices
                // Base paths are handled in the manifest creation section below
                final_fragments.extend(maybe_existing_fragments?.clone());
            }
            Operation::UserOperation(_) => {
                return Err(Error::not_supported(
                    "applying an action-based transaction is not implemented yet",
                ));
            }
        };

        // If a fragment was reserved then it may not belong at the end of the fragments list.
        final_fragments.sort_by_key(|frag| frag.id);

        // Clean up data files that only contain tombstoned fields
        Self::remove_tombstoned_data_files(&mut final_fragments);

        // Enforce the newest-last overlay ordering invariant at the write
        // boundary. Load normalizes with a sort; this rejects any commit path
        // that assembled a fragment's overlays out of order.
        for fragment in &final_fragments {
            if !fragment.overlays.is_empty() {
                crate::format::overlay::verify_overlays_newest_last(&fragment.overlays)?;
            }
        }

        let user_requested_version = match (&config.storage_format, config.use_legacy_format) {
            (Some(storage_format), _) => Some(storage_format.lance_file_format()),
            (None, Some(true)) => Some(ConcreteFileVersion::V1),
            (None, Some(false)) => Some(ConcreteFileVersion::V2_0),
            (None, None) => None,
        };

        // Applied once the final index list is known, so it sees exactly the
        // indices this commit publishes rather than what any one operation arm
        // intended.
        if mem_wal_segments_before.is_some() {
            let empty_segments = LogicalIndexSegments::new();
            Self::apply_mem_wal_index_coverage(
                &mut final_indices,
                mem_wal_segments_before.as_ref().unwrap_or(&empty_segments),
                read_version_state,
                index_catchup_required,
                new_version,
            )?;
        }

        let mut manifest = if let Some(current_manifest) = current_manifest {
            // OVERWRITE with initial_bases on existing dataset is not allowed (caught by validation)
            // So we always use new_from_previous which preserves base_paths
            let mut prev_manifest =
                Manifest::new_from_previous(current_manifest, schema, Arc::new(final_fragments));

            if let (Some(user_requested_version), Operation::Overwrite { .. }) =
                (user_requested_version, &self.operation)
            {
                // If this is an overwrite operation and the user has requested a specific version
                // then overwrite with that version.  Otherwise, if the user didn't request a specific
                // version, then overwrite with whatever version we had before.
                prev_manifest.data_storage_format = DataStorageFormat::new(user_requested_version);
            }

            prev_manifest
        } else {
            let data_storage_format =
                Self::data_storage_format_from_files(&final_fragments, user_requested_version)?;
            Manifest::new(
                schema,
                Arc::new(final_fragments),
                data_storage_format,
                reference_paths,
            )
        };

        manifest.tag.clone_from(&self.tag);

        if config.auto_set_feature_flags {
            // Internal operations (e.g. CreateIndex) build with the default config,
            // which has use_stable_row_ids = false. Without inheriting from the previous
            // manifest, apply_feature_flags would clear FLAG_STABLE_ROW_IDS.
            let inherited = current_manifest
                .map(|m| m.uses_stable_row_ids())
                .unwrap_or(false);
            let use_stable_row_ids = config.use_stable_row_ids || inherited;
            apply_feature_flags(
                &mut manifest,
                use_stable_row_ids,
                config.disable_transaction_file,
            )?;
        }
        // Carried from the manifest this one is derived from. `new_from_previous`
        // zeroes both feature words, so `apply_feature_flags` cannot see the
        // previous state and every ordinary commit would otherwise drop the bit.
        if let Some(current_manifest) = current_manifest {
            inherit_mem_wal_index_catchup(&mut manifest, current_manifest)?;
        }

        // Set after apply_feature_flags, which resets both flag words: activation
        // is the one place the bit is turned on, and it must survive that reset.
        if let Operation::UpdateMemWalState {
            require_index_catchup: true,
            ..
        } = &self.operation
        {
            let reader_set = current_manifest
                .map(|m| m.reader_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP != 0)
                .unwrap_or(false);
            let writer_set = current_manifest
                .map(|m| m.writer_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP != 0)
                .unwrap_or(false);
            match (reader_set, writer_set) {
                (false, false) => {
                    Self::require_index_catchup(&mut final_indices, new_version)?;
                    log::info!(
                        "MemWAL index catch-up is now required at version {new_version}; a \
                         missing catch-up entry means an index is behind, not caught up. \
                         This is one-way."
                    );
                }
                // Already active. A retry whose first attempt landed but lost its
                // response must not clear coverage repaired since, so this keeps
                // every recorded generation.
                (true, true) => {}
                _ => {
                    return Err(Error::invalid_input(
                        "Cannot require MemWAL index catch-up: the table has only one of \
                         the reader and writer feature bits set, so its catch-up \
                         semantics are undefined",
                    ));
                }
            }
            manifest.reader_feature_flags |= FLAG_MEM_WAL_INDEX_CATCHUP;
            manifest.writer_feature_flags |= FLAG_MEM_WAL_INDEX_CATCHUP;
        }

        manifest.set_timestamp(config.timestamp_nanos);

        manifest.update_max_fragment_id();

        match &self.operation {
            Operation::Overwrite {
                config_upsert_values: Some(tm),
                ..
            } => {
                manifest.config_mut().extend(tm.clone());
            }
            Operation::UpdateConfig {
                config_updates,
                table_metadata_updates,
                schema_metadata_updates,
                field_metadata_updates,
            } => {
                if let Some(config_updates) = config_updates {
                    let mut config = manifest.config.clone();
                    apply_update_map(&mut config, config_updates);
                    manifest.config = config;
                }
                if let Some(table_metadata_updates) = table_metadata_updates {
                    let mut table_metadata = manifest.table_metadata.clone();
                    apply_update_map(&mut table_metadata, table_metadata_updates);
                    manifest.table_metadata = table_metadata;
                }
                if let Some(schema_metadata_updates) = schema_metadata_updates {
                    let mut schema_metadata = manifest.schema.metadata.clone();
                    apply_update_map(&mut schema_metadata, schema_metadata_updates);
                    manifest.schema.metadata = schema_metadata;
                }
                // The unenforced primary and clustering keys are reserved
                // schema properties: each is immutable once set, and its
                // reserved metadata keys cannot be written with an invalid
                // value. Capture the prior keys, and whether this transaction
                // writes a reserved key, before applying the updates so
                // violations can be rejected below. This runs on every apply,
                // including conflict-rebase, so it also rejects the
                // concurrent-writer race.
                let primary_key_before: Vec<i32> = manifest
                    .schema
                    .unenforced_primary_key()
                    .iter()
                    .map(|field| field.id)
                    .collect();
                let writes_primary_key = field_metadata_updates.values().any(|update| {
                    update.update_entries.iter().any(|entry| {
                        entry.key == LANCE_UNENFORCED_PRIMARY_KEY
                            || entry.key == LANCE_UNENFORCED_PRIMARY_KEY_POSITION
                    })
                });
                let clustering_key_before: Vec<i32> = manifest
                    .schema
                    .unenforced_clustering_key()
                    .iter()
                    .map(|field| field.id)
                    .collect();
                let writes_clustering_key = field_metadata_updates.values().any(|update| {
                    update
                        .update_entries
                        .iter()
                        .any(|entry| entry.key == LANCE_UNENFORCED_CLUSTERING_KEY_POSITION)
                });
                for (field_id, field_metadata_update) in field_metadata_updates {
                    if let Some(field) = manifest.schema.field_by_id_mut(*field_id) {
                        apply_update_map(&mut field.metadata, field_metadata_update);
                        // Also set unenforced primary key based on updated field metadata.
                        field.unenforced_primary_key_position = field
                            .metadata
                            .get(LANCE_UNENFORCED_PRIMARY_KEY_POSITION)
                            .and_then(|s| s.parse::<u32>().ok())
                            .or_else(|| {
                                field
                                    .metadata
                                    .get(LANCE_UNENFORCED_PRIMARY_KEY)
                                    .filter(|s| {
                                        matches!(s.to_lowercase().as_str(), "true" | "1" | "yes")
                                    })
                                    .map(|_| 0)
                            });
                        // Also set unenforced clustering key based on updated
                        // field metadata.
                        field.unenforced_clustering_key_position = field
                            .metadata
                            .get(LANCE_UNENFORCED_CLUSTERING_KEY_POSITION)
                            .and_then(|s| s.parse::<u32>().ok());
                    } else {
                        return Err(Error::invalid_input_source(
                            format!("Field with id {} does not exist", field_id).into(),
                        ));
                    }
                }
                let primary_key_after: Vec<i32> = manifest
                    .schema
                    .unenforced_primary_key()
                    .iter()
                    .map(|field| field.id)
                    .collect();
                if !primary_key_before.is_empty() {
                    // The primary key is already set: reject any change to it,
                    // and any write that touches a reserved primary key.
                    if writes_primary_key || primary_key_after != primary_key_before {
                        return Err(Error::invalid_input(
                            "the unenforced primary key is a reserved key and cannot be changed once set",
                        ));
                    }
                } else if writes_primary_key && primary_key_after.is_empty() {
                    // A reserved primary key was written but did not install a
                    // valid primary key (e.g. a non-marker flag value or a
                    // non-numeric position).
                    return Err(Error::invalid_input(
                        "the unenforced primary key is a reserved key and cannot be set to an invalid value",
                    ));
                }
                let clustering_key_after: Vec<i32> = manifest
                    .schema
                    .unenforced_clustering_key()
                    .iter()
                    .map(|field| field.id)
                    .collect();
                if !clustering_key_before.is_empty() {
                    // The clustering key is already set: reject any change to
                    // it, and any write that touches the reserved key.
                    if writes_clustering_key || clustering_key_after != clustering_key_before {
                        return Err(Error::invalid_input(
                            "the unenforced clustering key is a reserved key and cannot be changed once set",
                        ));
                    }
                } else if writes_clustering_key && clustering_key_after.is_empty() {
                    // The reserved clustering key was written but did not
                    // install a valid clustering key (e.g. a non-numeric
                    // position value).
                    return Err(Error::invalid_input(
                        "the unenforced clustering key is a reserved key and cannot be set to an invalid value",
                    ));
                }
            }
            _ => {}
        }

        // Handle UpdateBases operation to update manifest base_paths
        if let Operation::UpdateBases { new_bases } = &self.operation {
            // Validate and add new base paths to the manifest
            for new_base in new_bases {
                // Check for conflicts with existing base paths
                if let Some(existing_base) = manifest
                    .base_paths
                    .values()
                    .find(|bp| bp.name == new_base.name || bp.path == new_base.path)
                {
                    return Err(Error::invalid_input(format!(
                        "Conflict detected: Base path with name '{:?}' or path '{}' already exists. Existing: name='{:?}', path='{}'",
                        new_base.name, new_base.path, existing_base.name, existing_base.path
                    )));
                }

                // Assign a new ID if not already assigned
                let mut base_to_add = new_base.clone();
                if base_to_add.id == 0 {
                    let next_id = manifest
                        .base_paths
                        .keys()
                        .max()
                        .map(|&id| id + 1)
                        .unwrap_or(1);
                    base_to_add.id = next_id;
                }

                manifest.base_paths.insert(base_to_add.id, base_to_add);
            }
        }

        if let Operation::ReserveFragments { num_fragments } = self.operation {
            manifest.max_fragment_id = Some(manifest.max_fragment_id.unwrap_or(0) + num_fragments);
        }

        manifest.transaction_file = Some(transaction_file_path.to_string());

        if let Some(next_row_id) = next_row_id {
            manifest.next_row_id = next_row_id;
        }

        Ok((manifest, final_indices))
    }

    /// Remove data files that only contain tombstoned fields (-2)
    /// These files no longer contain any live data and can be safely dropped
    fn remove_tombstoned_data_files(fragments: &mut [Fragment]) {
        for fragment in fragments {
            fragment.files.retain(|file| {
                // Keep file if it has at least one non-tombstoned field
                file.fields.iter().any(|&field_id| field_id != -2)
            });
        }
    }
    /// Coverage of an index that a rewrite invalidates: the rewritten fragments are
    /// removed and the fragments they became are *not* added.
    fn drop_rewritten_fragments(old: &RoaringBitmap, groups: &[RewriteGroup]) -> RoaringBitmap {
        let mut new_bitmap = old.clone();
        for group in groups {
            for old_fragment in &group.old_fragments {
                new_bitmap.remove(old_fragment.id as u32);
            }
        }
        new_bitmap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::overlay::OverlayCoverage;
    use crate::format::pb;
    use crate::format::{RowDatasetVersionMeta, RowDatasetVersionSequence, RowIdMeta};
    use crate::rowids::{RowIdSequence, write_row_ids};
    use crate::transaction::test_support::{
        default_build_config, make_stable_row_id_manifest, overlay_with_field,
        sample_index_metadata, sample_manifest,
    };
    use crate::transaction::{DataOverlayGroup, UpdateMode, validate_operation};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::Schema as LanceSchema;
    use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
    use lance_io::utils::CachedFileSize;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn sample_manifest_with_fragments(ids: std::ops::Range<u64>) -> Manifest {
        let schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        Manifest::new(
            LanceSchema::try_from(&schema).unwrap(),
            Arc::new(ids.map(Fragment::new).collect()),
            DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        )
    }

    #[test]
    fn test_create_index_build_manifest_keeps_unremoved_same_name_indices() {
        let manifest = sample_manifest();
        let first_index = sample_index_metadata("vector_idx");
        let second_index = sample_index_metadata("vector_idx");
        let third_index = sample_index_metadata("vector_idx");

        let transaction = Transaction::new(
            manifest.version,
            Operation::CreateIndex {
                new_indices: vec![third_index.clone()],
                removed_indices: vec![second_index.clone()],
            },
            None,
        );

        let (_, final_indices) = transaction
            .build_manifest(
                Some(&manifest),
                vec![first_index.clone(), second_index.clone()],
                "txn",
                &default_build_config(),
            )
            .unwrap();

        assert_eq!(final_indices.len(), 2);
        assert!(final_indices.iter().any(|idx| idx.uuid == first_index.uuid));
        assert!(final_indices.iter().any(|idx| idx.uuid == third_index.uuid));
        assert!(
            !final_indices
                .iter()
                .any(|idx| idx.uuid == second_index.uuid)
        );
    }

    #[test]
    fn test_create_index_build_manifest_deduplicates_relisted_indices_by_uuid() {
        let manifest = sample_manifest();
        let first_index = sample_index_metadata("vector_idx");
        let second_index = sample_index_metadata("vector_idx");
        let third_index = sample_index_metadata("vector_idx");

        let transaction = Transaction::new(
            manifest.version,
            Operation::CreateIndex {
                new_indices: vec![first_index.clone(), third_index.clone()],
                removed_indices: vec![second_index.clone()],
            },
            None,
        );

        let (_, final_indices) = transaction
            .build_manifest(
                Some(&manifest),
                vec![first_index.clone(), second_index.clone()],
                "txn",
                &default_build_config(),
            )
            .unwrap();

        assert_eq!(final_indices.len(), 2);
        assert_eq!(
            final_indices
                .iter()
                .filter(|idx| idx.uuid == first_index.uuid)
                .count(),
            1
        );
        assert!(final_indices.iter().any(|idx| idx.uuid == third_index.uuid));
        assert!(
            !final_indices
                .iter()
                .any(|idx| idx.uuid == second_index.uuid)
        );
    }

    #[test]
    fn test_update_build_manifest_replaces_and_removes_fragments() {
        let manifest = sample_manifest_with_fragments(0..5);

        let mut updated2 = Fragment::new(2);
        updated2.physical_rows = Some(42);
        let mut updated4 = Fragment::new(4);
        updated4.physical_rows = Some(43);

        let transaction = Transaction::new(
            manifest.version,
            Operation::Update {
                removed_fragment_ids: vec![1],
                // Fragment 99 does not exist in the dataset; it must be ignored,
                // not appended.
                updated_fragments: vec![updated2, updated4, Fragment::new(99)],
                new_fragments: vec![],
                fields_modified: vec![],
                compacted_sstables: vec![],
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
            None,
        );

        let (new_manifest, _) = transaction
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        let ids: Vec<u64> = new_manifest.fragments.iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![0, 2, 3, 4]);
        let rows: Vec<Option<usize>> = new_manifest
            .fragments
            .iter()
            .map(|f| f.physical_rows)
            .collect();
        assert_eq!(rows, vec![None, Some(42), None, Some(43)]);
    }

    #[test]
    fn test_delete_build_manifest_replaces_and_removes_fragments() {
        let manifest = sample_manifest_with_fragments(0..5);

        let mut updated2 = Fragment::new(2);
        updated2.physical_rows = Some(42);

        let transaction = Transaction::new(
            manifest.version,
            Operation::Delete {
                updated_fragments: vec![updated2],
                deleted_fragment_ids: vec![1, 3],
                predicate: "id > 0".to_string(),
            },
            None,
        );

        let (new_manifest, _) = transaction
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        let ids: Vec<u64> = new_manifest.fragments.iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![0, 2, 4]);
        let rows: Vec<Option<usize>> = new_manifest
            .fragments
            .iter()
            .map(|f| f.physical_rows)
            .collect();
        assert_eq!(rows, vec![None, Some(42), None]);
    }

    #[test]
    fn test_remove_tombstoned_data_files() {
        // Create a fragment with mixed data files: some normal, some fully tombstoned
        let mut fragment = Fragment::new(1);

        // Add a normal data file with valid field IDs
        fragment.files.push(DataFile {
            path: "normal.lance".to_string(),
            fields: Arc::from([1, 2, 3]),
            column_indices: Arc::from([]),
            file_major_version: 2,
            file_minor_version: 0,
            file_size_bytes: CachedFileSize::new(1000),
            base_id: None,
        });

        // Add a data file with all fields tombstoned
        fragment.files.push(DataFile {
            path: "all_tombstoned.lance".to_string(),
            fields: Arc::from([-2, -2, -2]),
            column_indices: Arc::from([]),
            file_major_version: 2,
            file_minor_version: 0,
            file_size_bytes: CachedFileSize::new(500),
            base_id: None,
        });

        // Add a data file with mixed tombstoned and valid fields
        fragment.files.push(DataFile {
            path: "mixed.lance".to_string(),
            fields: Arc::from([4, -2, 5]),
            column_indices: Arc::from([]),
            file_major_version: 2,
            file_minor_version: 0,
            file_size_bytes: CachedFileSize::new(750),
            base_id: None,
        });

        // Add another fully tombstoned file
        fragment.files.push(DataFile {
            path: "another_tombstoned.lance".to_string(),
            fields: Arc::from([-2_i32]),
            column_indices: Arc::from([]),
            file_major_version: 2,
            file_minor_version: 0,
            file_size_bytes: CachedFileSize::new(250),
            base_id: None,
        });

        let mut fragments = vec![fragment];

        // Apply the cleanup
        Transaction::remove_tombstoned_data_files(&mut fragments);

        // Should have removed the two fully tombstoned files
        assert_eq!(fragments[0].files.len(), 2);
        assert_eq!(fragments[0].files[0].path, "normal.lance");
        assert_eq!(fragments[0].files[1].path, "mixed.lance");
    }

    /// When a fragment has no existing last_updated_at_version_meta (None), a
    /// partial RewriteColumns refresh must leave it as None rather than fabricating
    /// prev_version for unmatched rows.
    #[test]
    fn test_partial_rewrite_skips_fragment_with_no_version_meta() {
        let row_ids = RowIdSequence::from([10u64, 11, 12, 13, 14].as_slice());
        let row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&row_ids).into()));

        let data_file = DataFile::new(
            "data.lance",
            vec![0],
            vec![0],
            LanceFileVersion::Stable.resolve(),
            None,
            None,
        );

        let fragment = Fragment {
            id: 1,
            files: vec![data_file],
            overlays: vec![],
            deletion_file: None,
            row_id_meta,
            physical_rows: Some(5),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };

        let manifest = make_stable_row_id_manifest(vec![fragment.clone()]);

        // Simulate a RewriteColumns update that matched offsets 1 and 3
        let off_map = HashMap::from([(1u64, RoaringBitmap::from_iter([1u32, 3]))]);
        let tx = Transaction::new(
            manifest.version,
            Operation::Update {
                removed_fragment_ids: vec![],
                updated_fragments: vec![fragment],
                new_fragments: vec![],
                fields_modified: vec![],
                compacted_sstables: vec![],
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(UpdateMode::RewriteColumns),
                inserted_rows_filter: None,
                updated_fragment_offsets: Some(UpdatedFragmentOffsets(off_map)),
            },
            None,
        );

        let (out, _) = tx
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        assert!(
            out.fragments[0].last_updated_at_version_meta.is_none(),
            "fragment with no prior version metadata must not have fabricated prev_version stamped on unmatched rows"
        );
    }

    #[test]
    fn test_bitmap_cardinality_exceeds_physical_rows() {
        let row_ids = RowIdSequence::from([10u64, 11, 12, 13, 14].as_slice());
        let row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&row_ids).into()));

        let data_file = DataFile::new(
            "data.lance",
            vec![0],
            vec![0],
            LanceFileVersion::Stable.resolve(),
            None,
            None,
        );

        let version_seq = RowDatasetVersionSequence::from_uniform_row_count(5, 1);
        let version_meta = RowDatasetVersionMeta::from_sequence(&version_seq).unwrap();

        let fragment = Fragment {
            id: 1,
            files: vec![data_file],
            overlays: vec![],
            deletion_file: None,
            row_id_meta,
            physical_rows: Some(5),
            last_updated_at_version_meta: Some(version_meta.clone()),
            created_at_version_meta: Some(version_meta),
        };

        let manifest = make_stable_row_id_manifest(vec![fragment.clone()]);

        // Bitmap with 10 offsets but fragment only has 5 physical rows.
        let off_map = HashMap::from([(1u64, RoaringBitmap::from_iter(0u32..10))]);
        let tx = Transaction::new(
            manifest.version,
            Operation::Update {
                removed_fragment_ids: vec![],
                updated_fragments: vec![fragment],
                new_fragments: vec![],
                fields_modified: vec![],
                compacted_sstables: vec![],
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(UpdateMode::RewriteColumns),
                inserted_rows_filter: None,
                updated_fragment_offsets: Some(UpdatedFragmentOffsets(off_map)),
            },
            None,
        );

        let result = tx.build_manifest(Some(&manifest), vec![], "txn", &default_build_config());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("cardinality"),
            "expected cardinality error, got: {msg}"
        );
    }

    #[test]
    fn test_bitmap_max_offset_exceeds_physical_rows() {
        let row_ids = RowIdSequence::from([10u64, 11, 12, 13, 14].as_slice());
        let row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&row_ids).into()));

        let data_file = DataFile::new(
            "data.lance",
            vec![0],
            vec![0],
            LanceFileVersion::Stable.resolve(),
            None,
            None,
        );

        let version_seq = RowDatasetVersionSequence::from_uniform_row_count(5, 1);
        let version_meta = RowDatasetVersionMeta::from_sequence(&version_seq).unwrap();

        let fragment = Fragment {
            id: 1,
            files: vec![data_file],
            overlays: vec![],
            deletion_file: None,
            row_id_meta,
            physical_rows: Some(5),
            last_updated_at_version_meta: Some(version_meta.clone()),
            created_at_version_meta: Some(version_meta),
        };

        let manifest = make_stable_row_id_manifest(vec![fragment.clone()]);

        // Only 2 offsets (within cardinality) but max offset 100 exceeds physical_rows 5.
        let off_map = HashMap::from([(1u64, RoaringBitmap::from_iter([0u32, 100]))]);
        let tx = Transaction::new(
            manifest.version,
            Operation::Update {
                removed_fragment_ids: vec![],
                updated_fragments: vec![fragment],
                new_fragments: vec![],
                fields_modified: vec![],
                compacted_sstables: vec![],
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(UpdateMode::RewriteColumns),
                inserted_rows_filter: None,
                updated_fragment_offsets: Some(UpdatedFragmentOffsets(off_map)),
            },
            None,
        );

        let result = tx.build_manifest(Some(&manifest), vec![], "txn", &default_build_config());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("max offset"),
            "expected max offset error, got: {msg}"
        );
    }

    #[test]
    fn test_bitmap_at_exact_physical_rows_boundary_succeeds() {
        let row_ids = RowIdSequence::from([10u64, 11, 12, 13, 14].as_slice());
        let row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&row_ids).into()));

        let data_file = DataFile::new(
            "data.lance",
            vec![0],
            vec![0],
            LanceFileVersion::Stable.resolve(),
            None,
            None,
        );

        let version_seq = RowDatasetVersionSequence::from_uniform_row_count(5, 1);
        let version_meta = RowDatasetVersionMeta::from_sequence(&version_seq).unwrap();

        let fragment = Fragment {
            id: 1,
            files: vec![data_file],
            overlays: vec![],
            deletion_file: None,
            row_id_meta,
            physical_rows: Some(5),
            last_updated_at_version_meta: Some(version_meta.clone()),
            created_at_version_meta: Some(version_meta),
        };

        let manifest = make_stable_row_id_manifest(vec![fragment.clone()]);

        // All 5 offsets on a 5-row fragment — exactly at the boundary, should succeed.
        let off_map = HashMap::from([(1u64, RoaringBitmap::from_iter(0u32..5))]);
        let tx = Transaction::new(
            manifest.version,
            Operation::Update {
                removed_fragment_ids: vec![],
                updated_fragments: vec![fragment],
                new_fragments: vec![],
                fields_modified: vec![],
                compacted_sstables: vec![],
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(UpdateMode::RewriteColumns),
                inserted_rows_filter: None,
                updated_fragment_offsets: Some(UpdatedFragmentOffsets(off_map)),
            },
            None,
        );

        tx.build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .expect("bitmap at exact physical_rows boundary should succeed");
    }

    #[test]
    fn test_updated_fragment_offsets_key_not_in_updated_fragments_is_rejected() {
        // Fragment A is being rewritten; fragment B exists in the manifest but is
        // NOT in updated_fragments. Supplying an offset key for B must be rejected
        // so that B's version metadata cannot be stamped by an unrelated commit.
        let make_fragment = |id: u64| {
            let row_ids = RowIdSequence::from([id * 10].as_slice());
            let row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&row_ids).into()));
            Fragment {
                id,
                files: vec![DataFile::new(
                    format!("{id}.lance"),
                    vec![0],
                    vec![0],
                    LanceFileVersion::Stable.resolve(),
                    None,
                    None,
                )],
                overlays: vec![],
                deletion_file: None,
                row_id_meta,
                physical_rows: Some(5),
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
            }
        };

        let frag_a = make_fragment(1);
        let frag_b = make_fragment(2);
        let manifest = make_stable_row_id_manifest(vec![frag_a.clone(), frag_b.clone()]);

        // updated_fragments contains only A; offsets are keyed to B — must fail.
        let off_map = HashMap::from([(frag_b.id, RoaringBitmap::from_iter([0u32, 1, 2]))]);
        let operation = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![frag_a],
            new_fragments: vec![],
            fields_modified: vec![],
            compacted_sstables: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: Some(UpdatedFragmentOffsets(off_map)),
        };

        let err = validate_operation(Some(&manifest), &operation).unwrap_err();
        assert!(
            err.to_string().contains("not in updated_fragments"),
            "expected key-presence error, got: {err}"
        );
    }

    #[test]
    fn test_proto_round_trip_field_10() {
        let off_map = HashMap::from([
            (1u64, RoaringBitmap::from_iter([1u32, 3, 5])),
            (2u64, RoaringBitmap::from_iter([0u32, 2, 4, 6])),
        ]);
        let tx = Transaction::new(
            1,
            Operation::Update {
                removed_fragment_ids: vec![],
                updated_fragments: vec![],
                new_fragments: vec![],
                fields_modified: vec![],
                compacted_sstables: vec![],
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(UpdateMode::RewriteColumns),
                inserted_rows_filter: None,
                updated_fragment_offsets: Some(UpdatedFragmentOffsets(off_map.clone())),
            },
            None,
        );

        let pb_tx: pb::Transaction = pb::Transaction::from(&tx);

        // Field 9 must be empty; field 10 must be populated.
        if let Some(pb::transaction::Operation::Update(ref update)) = pb_tx.operation {
            assert!(
                update.updated_fragment_offsets.is_empty(),
                "field 9 should be empty"
            );
            assert_eq!(update.updated_fragment_offset_bitmaps.len(), 2);
        } else {
            panic!("expected Update operation");
        }

        let tx2 = Transaction::try_from(pb_tx).unwrap();
        if let Operation::Update {
            updated_fragment_offsets: Some(UpdatedFragmentOffsets(m)),
            ..
        } = &tx2.operation
        {
            assert_eq!(m.len(), 2);
            assert_eq!(*m.get(&1).unwrap(), off_map[&1]);
            assert_eq!(*m.get(&2).unwrap(), off_map[&2]);
        } else {
            panic!("expected Update with offsets");
        }
    }

    #[test]
    fn test_proto_legacy_field_9_read() {
        // Simulate a manifest written by old Lance: only field 9, no field 10.
        let pb_tx = pb::Transaction {
            read_version: 1,
            uuid: "test".to_string(),
            tag: String::new(),
            transaction_properties: HashMap::new(),
            operation: Some(pb::transaction::Operation::Update(
                pb::transaction::Update {
                    removed_fragment_ids: vec![],
                    updated_fragments: vec![],
                    new_fragments: vec![],
                    fields_modified: vec![],
                    compacted_sstables: vec![],
                    fields_for_preserving_frag_bitmap: vec![],
                    update_mode: 1,
                    inserted_rows: None,
                    updated_fragment_offsets: HashMap::from([(
                        1u64,
                        pb::transaction::UInt32List {
                            values: vec![1, 3, 5],
                        },
                    )]),
                    updated_fragment_offset_bitmaps: HashMap::new(),
                },
            )),
        };

        let tx = Transaction::try_from(pb_tx).unwrap();
        if let Operation::Update {
            updated_fragment_offsets: Some(UpdatedFragmentOffsets(m)),
            ..
        } = &tx.operation
        {
            assert_eq!(m.len(), 1);
            let bitmap = m.get(&1).unwrap();
            let offsets: Vec<u32> = bitmap.iter().collect();
            assert_eq!(offsets, vec![1, 3, 5]);
        } else {
            panic!("expected Update with offsets from legacy field 9");
        }
    }

    #[test]
    fn test_proto_field_10_takes_precedence_over_field_9() {
        // When both fields present, field 10 wins.
        let mut bitmap_bytes = Vec::new();
        RoaringBitmap::from_iter([10u32, 20, 30])
            .serialize_into(&mut bitmap_bytes)
            .unwrap();

        let pb_tx = pb::Transaction {
            read_version: 1,
            uuid: "test".to_string(),
            tag: String::new(),
            transaction_properties: HashMap::new(),
            operation: Some(pb::transaction::Operation::Update(
                pb::transaction::Update {
                    removed_fragment_ids: vec![],
                    updated_fragments: vec![],
                    new_fragments: vec![],
                    fields_modified: vec![],
                    compacted_sstables: vec![],
                    fields_for_preserving_frag_bitmap: vec![],
                    update_mode: 1,
                    inserted_rows: None,
                    // Field 9 has different values than field 10.
                    updated_fragment_offsets: HashMap::from([(
                        1u64,
                        pb::transaction::UInt32List {
                            values: vec![99, 100],
                        },
                    )]),
                    updated_fragment_offset_bitmaps: HashMap::from([(1u64, bitmap_bytes)]),
                },
            )),
        };

        let tx = Transaction::try_from(pb_tx).unwrap();
        if let Operation::Update {
            updated_fragment_offsets: Some(UpdatedFragmentOffsets(m)),
            ..
        } = &tx.operation
        {
            let offsets: Vec<u32> = m.get(&1).unwrap().iter().collect();
            assert_eq!(offsets, vec![10, 20, 30], "field 10 should take precedence");
        } else {
            panic!("expected Update with offsets from field 10");
        }
    }

    #[test]
    fn merge_build_manifest_refreshes_last_updated_when_data_files_change_stable_row_ids() {
        use crate::feature_flags::FLAG_STABLE_ROW_IDS;
        use lance_file::version::LanceFileVersion;

        let mk_file = |path: &str| {
            DataFile::new(
                path,
                vec![0],
                vec![0],
                LanceFileVersion::Stable.resolve(),
                None,
                None,
            )
        };

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        let lance_schema = LanceSchema::try_from(&arrow_schema).unwrap();

        let row_ids = RowIdSequence::from([100u64, 101, 102, 103, 104].as_slice());
        let row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&row_ids).into()));

        let prev_fragment = Fragment {
            id: 0,
            files: vec![mk_file("before.lance")],
            overlays: vec![],
            deletion_file: None,
            row_id_meta,
            physical_rows: Some(5),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };

        let mut manifest = Manifest::new(
            lance_schema.clone(),
            Arc::new(vec![prev_fragment.clone()]),
            DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );
        manifest.reader_feature_flags |= FLAG_STABLE_ROW_IDS;
        manifest.next_row_id = 100;

        let merged_fragment = Fragment {
            files: vec![mk_file("after.lance")],
            ..prev_fragment
        };

        let tx = Transaction::new(
            manifest.version,
            Operation::Merge {
                fragments: vec![merged_fragment],
                schema: lance_schema,
                preserves_nullability: true,
            },
            None,
        );

        let (out, _) = tx
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        assert_eq!(out.version, 2);
        let frag = &out.fragments[0];
        let seq = frag
            .last_updated_at_version_meta
            .as_ref()
            .unwrap()
            .load_sequence()
            .unwrap();
        assert_eq!(seq.version_at(0).unwrap(), 2);
        assert_eq!(seq.version_at(4).unwrap(), 2);
    }

    #[test]
    fn merge_build_manifest_skips_refresh_when_carry_forward_stable_row_ids() {
        use crate::feature_flags::FLAG_STABLE_ROW_IDS;
        use crate::rowids::version::{RowDatasetVersionMeta, RowDatasetVersionSequence};
        use lance_file::version::LanceFileVersion;

        let data_file = DataFile::new(
            "same.lance",
            vec![0],
            vec![0],
            LanceFileVersion::Stable.resolve(),
            None,
            None,
        );

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        let lance_schema = LanceSchema::try_from(&arrow_schema).unwrap();

        let row_ids = RowIdSequence::from([200u64, 201, 202, 203, 204].as_slice());
        let row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&row_ids).into()));

        let uniform_v1 = RowDatasetVersionSequence::from_uniform_row_count(5, 1);
        let meta_v1 = RowDatasetVersionMeta::from_sequence(&uniform_v1).unwrap();

        let prev_fragment = Fragment {
            id: 0,
            files: vec![data_file.clone()],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: row_id_meta.clone(),
            physical_rows: Some(5),
            last_updated_at_version_meta: Some(meta_v1.clone()),
            created_at_version_meta: None,
        };

        let mut manifest = Manifest::new(
            lance_schema.clone(),
            Arc::new(vec![prev_fragment]),
            DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );
        manifest.reader_feature_flags |= FLAG_STABLE_ROW_IDS;
        manifest.next_row_id = 100;

        let merged_fragment = Fragment {
            id: 0,
            files: vec![data_file],
            overlays: vec![],
            deletion_file: None,
            row_id_meta,
            physical_rows: Some(5),
            last_updated_at_version_meta: Some(meta_v1),
            created_at_version_meta: None,
        };

        let tx = Transaction::new(
            manifest.version,
            Operation::Merge {
                fragments: vec![merged_fragment],
                schema: lance_schema,
                preserves_nullability: true,
            },
            None,
        );

        let (out, _) = tx
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        let seq = out.fragments[0]
            .last_updated_at_version_meta
            .as_ref()
            .unwrap()
            .load_sequence()
            .unwrap();
        assert_eq!(seq.version_at(0).unwrap(), 1);
        assert_eq!(seq.version_at(4).unwrap(), 1);
    }

    #[test]
    fn merge_build_manifest_no_last_updated_refresh_without_stable_row_ids() {
        use crate::feature_flags::FLAG_STABLE_ROW_IDS;
        use lance_file::version::LanceFileVersion;

        let mk_file = |path: &str| {
            DataFile::new(
                path,
                vec![0],
                vec![0],
                LanceFileVersion::Stable.resolve(),
                None,
                None,
            )
        };

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        let lance_schema = LanceSchema::try_from(&arrow_schema).unwrap();

        let prev_fragment = Fragment {
            id: 0,
            files: vec![mk_file("before.lance")],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(5),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };

        let manifest = Manifest::new(
            lance_schema.clone(),
            Arc::new(vec![prev_fragment.clone()]),
            DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );
        assert_eq!(
            manifest.reader_feature_flags & FLAG_STABLE_ROW_IDS,
            0,
            "manifest must not use stable row IDs for this guard test"
        );

        let merged_fragment = Fragment {
            files: vec![mk_file("after.lance")],
            ..prev_fragment
        };

        let tx = Transaction::new(
            manifest.version,
            Operation::Merge {
                fragments: vec![merged_fragment],
                schema: lance_schema,
                preserves_nullability: true,
            },
            None,
        );

        let (out, _) = tx
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        assert!(
            out.fragments[0].last_updated_at_version_meta.is_none(),
            "without stable row IDs, Merge must not populate per-row last_updated metadata"
        );
    }

    #[test]
    fn merge_build_manifest_sets_both_version_meta_for_new_fragment_id_stable_row_ids() {
        use crate::feature_flags::FLAG_STABLE_ROW_IDS;
        use lance_file::version::LanceFileVersion;

        let mk_file = |path: &str| {
            DataFile::new(
                path,
                vec![0],
                vec![0],
                LanceFileVersion::Stable.resolve(),
                None,
                None,
            )
        };

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        let lance_schema = LanceSchema::try_from(&arrow_schema).unwrap();

        // Existing fragment (id=0) with stable row IDs
        let row_ids_0 = RowIdSequence::from([10u64, 11, 12].as_slice());
        let existing_fragment = Fragment {
            id: 0,
            files: vec![mk_file("existing.lance")],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&row_ids_0).into())),
            physical_rows: Some(3),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };

        let mut manifest = Manifest::new(
            lance_schema.clone(),
            Arc::new(vec![existing_fragment.clone()]),
            DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );
        manifest.reader_feature_flags |= FLAG_STABLE_ROW_IDS;
        manifest.next_row_id = 100;
        manifest.version = 1;

        // New fragment (id=1) not present in prev manifest — exercises the None branch
        let row_ids_1 = RowIdSequence::from([20u64, 21, 22, 23].as_slice());
        let new_fragment = Fragment {
            id: 1,
            files: vec![mk_file("new.lance")],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&row_ids_1).into())),
            physical_rows: Some(4),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };

        let tx = Transaction::new(
            manifest.version,
            Operation::Merge {
                fragments: vec![existing_fragment, new_fragment],
                schema: lance_schema,
                preserves_nullability: true,
            },
            None,
        );

        let (out, _) = tx
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        assert_eq!(out.version, 2);

        let new_frag = out.fragments.iter().find(|f| f.id == 1).unwrap();

        // last_updated_at_version must be set to the commit version
        let last_updated_seq = new_frag
            .last_updated_at_version_meta
            .as_ref()
            .expect("new fragment must have last_updated_at_version_meta")
            .load_sequence()
            .unwrap();
        assert_eq!(last_updated_seq.version_at(0).unwrap(), 2);
        assert_eq!(last_updated_seq.version_at(3).unwrap(), 2);

        // created_at_version must also be set — must not be None
        let created_seq = new_frag
            .created_at_version_meta
            .as_ref()
            .expect("new fragment must have created_at_version_meta")
            .load_sequence()
            .unwrap();
        assert_eq!(created_seq.version_at(0).unwrap(), 2);
        assert_eq!(created_seq.version_at(3).unwrap(), 2);
    }

    #[test]
    fn test_data_overlay_build_manifest_multi_fragment() {
        // Overlays targeting two distinct fragments are each applied and stamped.
        // A targeted fragment already carrying an overlay (committed at v3) gets
        // the new overlay appended and stamped while its existing overlay is
        // preserved, and a fragment the operation does not target is passed
        // through with its existing overlays untouched.
        let mut frag0 = Fragment::new(0);
        frag0.overlays = vec![overlay_with_field(5, 3)]; // targeted, pre-existing at v3
        let frag1 = Fragment::new(1);
        let mut frag2 = Fragment::new(2);
        frag2.overlays = vec![overlay_with_field(9, 3)]; // untargeted, committed at v3
        let schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        let mut manifest = Manifest::new(
            LanceSchema::try_from(&schema).unwrap(),
            Arc::new(vec![frag0, frag1, frag2]),
            crate::format::DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );
        // The pre-existing overlays were committed at v3, so the current
        // manifest must be at least that version; the new commit then stamps
        // its overlay at v4, keeping the fragment's overlays newest-last.
        manifest.version = 3;

        let txn = Transaction::new(
            manifest.version,
            Operation::DataOverlay {
                groups: vec![
                    DataOverlayGroup {
                        fragment_id: 0,
                        overlays: vec![overlay_with_field(1, 0)],
                    },
                    DataOverlayGroup {
                        fragment_id: 1,
                        overlays: vec![overlay_with_field(2, 0)],
                    },
                ],
            },
            None,
        );

        let (result, _) = txn
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        let frag = |id: u64| {
            result
                .fragments
                .iter()
                .find(|f| f.id == id)
                .unwrap_or_else(|| panic!("fragment {id} missing from result"))
        };
        // The already-overlaid target keeps its v3 overlay and appends the new
        // one, stamped to the new version.
        assert_eq!(frag(0).overlays.len(), 2);
        assert_eq!(frag(0).overlays[0].committed_version, 3);
        assert_eq!(frag(0).overlays[1].committed_version, result.version);
        // The fresh target gets its overlay, stamped to the new version.
        assert_eq!(frag(1).overlays.len(), 1);
        assert_eq!(frag(1).overlays[0].committed_version, result.version);
        // The untargeted fragment is unchanged: same overlay, original version.
        assert_eq!(frag(2).overlays.len(), 1);
        assert_eq!(frag(2).overlays[0].committed_version, 3);
        assert!(result.version > manifest.version);
    }

    #[test]
    fn test_data_replacement_tombstones_overlaid_fields() {
        // A DataReplacement writing new base values for field 5 must stop any
        // overlay already shadowing those cells: field 5 is tombstoned in place
        // (preserving the overlay's field 3), and an overlay covering only field
        // 5 is dropped entirely. Both overlays predate the transaction's read
        // version, which is what makes the replacement the newer value.
        let mut fragment = Fragment::new(0);
        fragment.files = vec![
            DataFile::new_legacy_from_fields("f3.lance", vec![3], None),
            DataFile::new_legacy_from_fields("f5.lance", vec![5], None),
        ];
        fragment.overlays = vec![
            DataOverlayFile {
                data_file: DataFile::new_legacy_from_fields("o35.lance", vec![3, 5], None),
                coverage: OverlayCoverage::sparse(vec![
                    roaring::RoaringBitmap::from_iter([0u32]),
                    roaring::RoaringBitmap::from_iter([0u32]),
                ]),
                committed_version: 1,
            },
            DataOverlayFile {
                data_file: DataFile::new_legacy_from_fields("o5.lance", vec![5], None),
                coverage: OverlayCoverage::dense(roaring::RoaringBitmap::from_iter([0u32])),
                committed_version: 1,
            },
        ];

        let schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        let manifest = Manifest::new(
            LanceSchema::try_from(&schema).unwrap(),
            Arc::new(vec![fragment]),
            crate::format::DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );

        let txn = Transaction::new(
            manifest.version,
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(
                    0,
                    DataFile::new_legacy_from_fields("f5-new.lance", vec![5], None),
                )],
            },
            None,
        );

        let (result, _) = txn
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        let frag = &result.fragments[0];
        // The base data file for field 5 was swapped in.
        assert!(frag.files.iter().any(|f| f.path == "f5-new.lance"));
        // The [3, 5] overlay keeps field 3 and tombstones field 5; the [5]-only
        // overlay is dropped.
        assert_eq!(frag.overlays.len(), 1);
        assert_eq!(frag.overlays[0].data_file.fields.as_ref(), &[3, -2]);
    }

    /// Replace `fields` in `fragment` at `read_version`, against a manifest
    /// at `manifest_version` whose schema declares field ids 3 ("x"), 4 ("a"),
    /// 5 ("v") and 6 ("y").
    fn replace_fields(
        fragment: Fragment,
        fields: Vec<i32>,
        manifest_version: u64,
        read_version: u64,
    ) -> Result<Fragment> {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("x", DataType::Int32, true),
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("v", DataType::Int32, true),
            ArrowField::new("y", DataType::Int32, true),
        ]);
        let mut lance_schema = LanceSchema::try_from(&schema).unwrap();
        lance_schema.fields[0].id = 3;
        lance_schema.fields[1].id = 4;
        lance_schema.fields[2].id = 5;
        lance_schema.fields[3].id = 6;
        let mut manifest = Manifest::new(
            lance_schema,
            Arc::new(vec![fragment]),
            crate::format::DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );
        manifest.version = manifest_version;

        let column_indices = (0..fields.len() as i32).collect();
        let txn = Transaction::new(
            read_version,
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(
                    0,
                    DataFile::new(
                        "v-new.lance",
                        fields,
                        column_indices,
                        ConcreteFileVersion::V2_0,
                        None,
                        None,
                    ),
                )],
            },
            None,
        );
        txn.build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .map(|(manifest, _)| manifest.fragments[0].clone())
    }

    /// Replace field 5 in `fragment` at `read_version`, against a manifest at
    /// `manifest_version`.
    fn replace_field_5(
        fragment: Fragment,
        manifest_version: u64,
        read_version: u64,
    ) -> Result<Fragment> {
        replace_fields(fragment, vec![5], manifest_version, read_version)
    }

    #[test]
    fn test_data_replacement_rejects_subset_of_legacy_file() {
        // The V1 reader derives its page table offset from the first field in
        // the file metadata, so turning `[4, 5]` into `[-2, 5]` would leave
        // field 4 decoding from field 5's pages. With no exact match to swap,
        // the replacement must be rejected rather than corrupting the sibling.
        let mut fragment = Fragment::new(0);
        fragment.files = vec![DataFile::new_legacy_from_fields(
            "wide.lance",
            vec![4, 5],
            None,
        )];

        let result = replace_field_5(fragment, 1, 1);
        assert!(
            result.is_err(),
            "legacy subset replacement must be rejected, got: {:?}",
            result.map(|fragment| fragment.files)
        );
    }

    #[test]
    fn test_data_replacement_tombstones_fields_spanning_files() {
        // The replaced fields sit in two different wider files. Each file is
        // tombstoned for the field it holds and survives on its remaining
        // live one, with the new file answering for both.
        let mut fragment = Fragment::new(0);
        fragment.files = vec![
            DataFile::new(
                "ab.lance",
                vec![3, 4],
                vec![0, 1],
                ConcreteFileVersion::V2_0,
                None,
                None,
            ),
            DataFile::new(
                "cd.lance",
                vec![5, 6],
                vec![0, 1],
                ConcreteFileVersion::V2_0,
                None,
                None,
            ),
        ];

        let fragment = replace_fields(fragment, vec![4, 5], 1, 1).unwrap();
        let file = |path| {
            fragment
                .files
                .iter()
                .find(|file| file.path == path)
                .unwrap_or_else(|| panic!("{path} survives on its live field"))
        };
        assert_eq!(file("ab.lance").fields.as_ref(), &[3, TOMBSTONE_FIELD_ID]);
        assert_eq!(file("cd.lance").fields.as_ref(), &[TOMBSTONE_FIELD_ID, 6]);
        assert!(fragment.files.iter().any(|file| file.path == "v-new.lance"));
    }

    #[test]
    fn test_data_replacement_rejects_fields_spanning_a_legacy_file() {
        // Spanning is only resolvable while every covering file can be
        // tombstoned. A V1 file holding one of the replaced fields cannot,
        // so the replacement must be rejected rather than half applied.
        let mut fragment = Fragment::new(0);
        fragment.files = vec![
            DataFile::new(
                "ab.lance",
                vec![3, 4],
                vec![0, 1],
                ConcreteFileVersion::V2_0,
                None,
                None,
            ),
            DataFile::new_legacy_from_fields("cd.lance", vec![5, 6], None),
        ];

        let result = replace_fields(fragment, vec![4, 5], 1, 1);
        assert!(
            result.is_err(),
            "spanning a legacy file must be rejected, got: {:?}",
            result.map(|fragment| fragment.files)
        );
    }

    #[test]
    fn test_data_replacement_retombstones_wider_file() {
        // A wider file carrying a tombstone from an earlier round is
        // tombstoned again for the newly replaced field and survives on its
        // remaining live field.
        let mut fragment = Fragment::new(0);
        fragment.files = vec![DataFile::new(
            "wide.lance",
            vec![4, TOMBSTONE_FIELD_ID, 5],
            vec![0, 1, 2],
            ConcreteFileVersion::V2_0,
            None,
            None,
        )];

        let fragment = replace_fields(fragment, vec![5], 1, 1).unwrap();
        let wide = fragment
            .files
            .iter()
            .find(|file| file.path == "wide.lance")
            .expect("wider file survives on its live field");
        assert_eq!(
            wide.fields.as_ref(),
            &[4, TOMBSTONE_FIELD_ID, TOMBSTONE_FIELD_ID]
        );
        assert!(fragment.files.iter().any(|file| file.path == "v-new.lance"));
    }

    #[test]
    fn test_data_replacement_preserves_overlay_newer_than_snapshot() {
        // An overlay committed after this transaction read its snapshot holds
        // the newer value; the conflict resolver rebases the two precisely
        // because the overlay wins. Tombstoning it would discard a committed
        // write, so only overlays the transaction could have seen are superseded.
        let mut fragment = Fragment::new(0);
        // One wider file, so the replacement takes the tombstone-and-append path.
        fragment.files = vec![DataFile::new(
            "wide.lance",
            vec![4, 5],
            vec![0, 1],
            ConcreteFileVersion::V2_0,
            None,
            None,
        )];
        fragment.overlays = vec![DataOverlayFile {
            data_file: DataFile::new(
                "newer.lance",
                vec![5],
                vec![0],
                ConcreteFileVersion::V2_0,
                None,
                None,
            ),
            coverage: OverlayCoverage::dense(roaring::RoaringBitmap::from_iter([0u32])),
            committed_version: 7,
        }];

        // Staged against version 6, i.e. before the overlay landed.
        let fragment = replace_field_5(fragment, 7, 6).unwrap();
        assert!(fragment.files.iter().any(|f| f.path == "v-new.lance"));
        assert_eq!(
            fragment.overlays.len(),
            1,
            "overlay committed after the snapshot must survive"
        );
        assert_eq!(fragment.overlays[0].data_file.fields.as_ref(), &[5]);
    }

    #[test]
    fn test_data_overlay_build_manifest_merges_duplicate_groups() {
        // Two groups targeting the same fragment must both survive (a HashMap
        // collapse would have dropped the first).
        let manifest = sample_manifest();
        let txn = Transaction::new(
            manifest.version,
            Operation::DataOverlay {
                groups: vec![
                    DataOverlayGroup {
                        fragment_id: 0,
                        overlays: vec![overlay_with_field(1, 0)],
                    },
                    DataOverlayGroup {
                        fragment_id: 0,
                        overlays: vec![overlay_with_field(2, 0)],
                    },
                ],
            },
            None,
        );

        let (result, _) = txn
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        let overlays = &result.fragments[0].overlays;
        assert_eq!(overlays.len(), 2);
        assert_eq!(overlays[0].data_file.fields.as_ref(), [1i32].as_slice());
        assert_eq!(overlays[1].data_file.fields.as_ref(), [2i32].as_slice());
    }

    #[test]
    fn test_data_overlay_build_manifest_rejects_unknown_fragment() {
        let manifest = sample_manifest();
        let txn = Transaction::new(
            manifest.version,
            Operation::DataOverlay {
                groups: vec![DataOverlayGroup {
                    fragment_id: 99,
                    overlays: vec![overlay_with_field(1, 0)],
                }],
            },
            None,
        );
        let err = txn
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap_err();
        assert!(err.to_string().contains("does not exist"), "{err}");
    }

    #[test]
    fn test_nullability_assertion_defaults_conservative() {
        // A writer that predates the field encodes nothing, which decodes as
        // false: no assertion, so a legacy tightening or required-field merge
        // still conflicts. Only an explicit true skips the barrier.
        for encoded in [false, true] {
            let txn = Transaction::try_from(pb::Transaction {
                read_version: 1,
                uuid: "test".to_string(),
                operation: Some(pb::transaction::Operation::Project(
                    pb::transaction::Project {
                        schema: vec![],
                        preserves_nullability: encoded,
                    },
                )),
                ..Default::default()
            })
            .unwrap();
            assert!(
                matches!(txn.operation, Operation::Project { preserves_nullability, .. } if preserves_nullability == encoded),
                "encoded={encoded:?}"
            );

            let txn = Transaction::try_from(pb::Transaction {
                read_version: 1,
                uuid: "test".to_string(),
                operation: Some(pb::transaction::Operation::Merge(pb::transaction::Merge {
                    fragments: vec![],
                    schema: vec![],
                    schema_metadata: Default::default(),
                    preserves_nullability: encoded,
                })),
                ..Default::default()
            })
            .unwrap();
            assert!(
                matches!(txn.operation, Operation::Merge { preserves_nullability, .. } if preserves_nullability == encoded),
                "encoded={encoded:?}"
            );
        }
    }

    mod mem_wal_index_coverage {
        use super::*;
        use crate::feature_flags::FLAG_MEM_WAL_INDEX_CATCHUP;
        use crate::system_index::mem_wal::{
            CompactedSsTable, IndexCatchupProgress, MEM_WAL_INDEX_NAME, MemWalIndexDetails,
        };

        fn user_index(name: &str, uuid: Uuid, frags: &[u32]) -> IndexMetadata {
            IndexMetadata {
                uuid,
                name: name.to_string(),
                fields: vec![0],
                dataset_version: 1,
                fragment_bitmap: Some(RoaringBitmap::from_iter(frags.iter().copied())),
                index_details: None,
                index_version: 0,
                created_at: None,
                base_id: None,
                files: None,
            }
        }

        fn mem_wal_index(details: MemWalIndexDetails) -> IndexMetadata {
            crate::system_index::mem_wal::new_mem_wal_index_meta(1, details).unwrap()
        }

        fn coverage_for(indices: &[IndexMetadata], name: &str) -> Option<Vec<CompactedSsTable>> {
            let meta = indices
                .iter()
                .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
                .expect("mem wal index present");
            load_mem_wal_index_details(meta.clone())
                .unwrap()
                .index_catchup
                .into_iter()
                .find(|entry| entry.index_name == name)
                .map(|entry| entry.caught_up_generations)
        }

        fn compacted(shard: Uuid, generation: u64) -> Vec<CompactedSsTable> {
            vec![CompactedSsTable::new(shard, generation)]
        }

        /// A manifest carrying exactly `frags`, standing in for the version a
        /// transaction read.
        fn manifest_with(frags: &[u32]) -> Manifest {
            let fragments: Vec<Fragment> =
                frags.iter().map(|id| Fragment::new(*id as u64)).collect();
            Manifest::new(
                LanceSchema::default(),
                Arc::new(fragments),
                DataStorageFormat::default(),
                Default::default(),
            )
        }

        /// Drives the production path, so these exercise the real derivation.
        fn apply(
            after: &mut [IndexMetadata],
            before: &[IndexMetadata],
            read_frags: &[u32],
            read_indices: &[IndexMetadata],
            required: bool,
        ) -> Result<()> {
            let manifest = manifest_with(read_frags);
            let segments_before = Transaction::logical_index_segments(before);
            Transaction::apply_mem_wal_index_coverage(
                after,
                &segments_before,
                Some(ReadVersionState {
                    manifest: &manifest,
                    indices: read_indices,
                }),
                required,
                2,
            )
        }

        fn table(idx_frags: &[u32], uuid: Uuid, details: MemWalIndexDetails) -> Vec<IndexMetadata> {
            vec![user_index("idx", uuid, idx_frags), mem_wal_index(details)]
        }

        fn progress(shard: Uuid, generation: u64) -> MemWalIndexDetails {
            MemWalIndexDetails {
                compacted_sstables: compacted(shard, generation),
                ..Default::default()
            }
        }

        fn progress_with_catchup(shard: Uuid, generation: u64, caught: u64) -> MemWalIndexDetails {
            MemWalIndexDetails {
                compacted_sstables: compacted(shard, generation),
                index_catchup: vec![IndexCatchupProgress::new(
                    "idx".to_string(),
                    compacted(shard, caught),
                )],
                ..Default::default()
            }
        }

        /// An index spanning every fragment the transaction read is credited
        /// with what that version had compacted.
        #[test]
        fn an_index_covering_the_read_version_is_credited() {
            let shard = Uuid::new_v4();
            let read = table(&[0, 1], Uuid::new_v4(), progress(shard, 5));
            let mut after = table(&[0, 1], Uuid::new_v4(), progress(shard, 5));
            apply(&mut after, &read, &[0, 1], &read, true).unwrap();
            assert_eq!(coverage_for(&after, "idx"), Some(compacted(shard, 5)));
        }

        /// An index short of the read version proves nothing, so it gets no
        /// entry -- absence reads as "not caught up".
        #[test]
        fn an_index_short_of_the_read_version_is_not_credited() {
            let shard = Uuid::new_v4();
            let read = table(&[0], Uuid::new_v4(), progress(shard, 5));
            let mut after = table(&[0], Uuid::new_v4(), progress(shard, 5));
            apply(&mut after, &read, &[0, 1], &read, true).unwrap();
            assert_eq!(coverage_for(&after, "idx"), None);
        }

        /// The hazard that makes the comparison use whole metadata.
        ///
        /// `Operation::Update` prunes a segment's fragment bitmap in place when
        /// it touches an indexed field, keeping the same UUID. A UUID-only
        /// "unchanged" test carries the old position forward while the index
        /// covers fewer fragments, and the WAL pod then trims on a position the
        /// index no longer earns. Reachable from the ordinary SSTable merge.
        #[test]
        fn a_bitmap_pruned_in_place_does_not_keep_its_position() {
            let shard = Uuid::new_v4();
            let uuid = Uuid::new_v4();
            let before = table(&[0, 1], uuid, progress_with_catchup(shard, 5, 5));
            // Same UUID, fragment 1 pruned away.
            let mut after = table(&[0], uuid, progress_with_catchup(shard, 5, 5));
            apply(&mut after, &before, &[0, 1], &before, true).unwrap();
            assert_eq!(
                coverage_for(&after, "idx"),
                None,
                "a shrunken index kept a position it no longer earns"
            );
        }

        /// Carrying a position forward is not the same as extending it. An
        /// index that has not moved still only holds the generations it caught
        /// up to; the compaction that has landed since is in fragments it does
        /// not span.
        #[test]
        fn an_unchanged_index_is_not_raised_beyond_what_it_proves() {
            let shard = Uuid::new_v4();
            let uuid = Uuid::new_v4();
            // Recorded at generation 2; generation 5 has since been folded in.
            let before = table(&[0], uuid, progress_with_catchup(shard, 5, 2));
            let mut after = before.clone();
            // Fragment 1 arrived with that compaction and this index lacks it.
            apply(&mut after, &before, &[0, 1], &before, true).unwrap();
            assert_eq!(coverage_for(&after, "idx"), Some(compacted(shard, 2)));
        }

        /// A recorded position above what this commit says was compacted is
        /// clamped down. Nothing should produce one, but a position the base
        /// table cannot back would retire SSTables whose rows are nowhere.
        #[test]
        fn a_carried_position_cannot_exceed_the_committed_progress() {
            let shard = Uuid::new_v4();
            let uuid = Uuid::new_v4();
            let before = table(&[0], uuid, progress_with_catchup(shard, 3, 9));
            let mut after = before.clone();
            apply(&mut after, &before, &[0], &before, true).unwrap();
            assert_eq!(coverage_for(&after, "idx"), Some(compacted(shard, 3)));
        }

        /// An unchanged index keeps what it recorded even when this commit's
        /// own snapshot cannot prove as much.
        #[test]
        fn an_unchanged_index_is_never_lowered() {
            let shard = Uuid::new_v4();
            let uuid = Uuid::new_v4();
            let before = table(&[0], uuid, progress_with_catchup(shard, 9, 9));
            let mut after = before.clone();
            apply(&mut after, &before, &[0, 1], &before, true).unwrap();
            assert_eq!(coverage_for(&after, "idx"), Some(compacted(shard, 9)));
        }

        /// Credit never exceeds what this commit records as compacted, so a
        /// read version since rolled back cannot retire SSTables no live commit
        /// copied in.
        #[test]
        fn credit_is_capped_by_the_committed_progress() {
            let shard = Uuid::new_v4();
            let read = table(&[0], Uuid::new_v4(), progress(shard, 9));
            let mut after = table(&[0], Uuid::new_v4(), progress(shard, 3));
            apply(&mut after, &read, &[0], &read, true).unwrap();
            assert_eq!(coverage_for(&after, "idx"), Some(compacted(shard, 3)));
        }

        /// The cap is the read version's progress, not this commit's. A
        /// compaction that landed while the index was being built put its rows
        /// in fragments this transaction never inspected, so covering
        /// everything it *did* read earns only what had been folded in by then.
        #[test]
        fn credit_never_reaches_past_the_read_version() {
            let shard = Uuid::new_v4();
            // Read at generation 2; generation 5 landed while this ran.
            let read = table(&[0], Uuid::new_v4(), progress(shard, 2));
            let mut after = table(&[0], Uuid::new_v4(), progress(shard, 5));
            apply(&mut after, &read, &[0], &read, true).unwrap();
            assert_eq!(coverage_for(&after, "idx"), Some(compacted(shard, 2)));
        }

        /// One segment with an unknown bitmap makes the whole index unproven,
        /// even when its siblings happen to span everything. Coverage that
        /// cannot be read is not coverage that can be relied on.
        #[test]
        fn an_index_with_an_unknown_segment_is_not_credited() {
            let shard = Uuid::new_v4();
            let mut unknown = user_index("idx", Uuid::new_v4(), &[]);
            unknown.fragment_bitmap = None;
            let read = vec![
                user_index("idx", Uuid::new_v4(), &[0, 1]),
                unknown,
                mem_wal_index(progress(shard, 5)),
            ];
            let mut after = read.clone();
            apply(&mut after, &read, &[0, 1], &read, true).unwrap();
            assert_eq!(coverage_for(&after, "idx"), None);
        }

        /// A dropped index has no coverage left to gate anything.
        #[test]
        fn a_dropped_index_loses_its_entry() {
            let shard = Uuid::new_v4();
            let before = table(&[0], Uuid::new_v4(), progress_with_catchup(shard, 5, 5));
            let mut after = vec![mem_wal_index(progress_with_catchup(shard, 5, 5))];
            apply(&mut after, &before, &[0], &before, true).unwrap();
            assert_eq!(coverage_for(&after, "idx"), None);
        }

        /// An index created by this commit is credited if it spans the read
        /// version -- it was built over those fragments, so it holds their
        /// rows. This is what the advance model could not express: an ordinary
        /// build that fully covers had to throw its work away and wait.
        #[test]
        fn a_new_index_covering_the_read_version_is_credited() {
            let shard = Uuid::new_v4();
            let before = vec![mem_wal_index(progress(shard, 5))];
            let mut after = table(&[0], Uuid::new_v4(), progress(shard, 5));
            // Covers the read version, but was not there when it was read.
            apply(&mut after, &before, &[0], &before, true).unwrap();
            assert_eq!(coverage_for(&after, "idx"), Some(compacted(shard, 5)));
        }

        /// A legacy table reads a missing entry as "fully caught up", so this
        /// must leave it alone rather than make it look more covered.
        #[test]
        fn a_legacy_table_is_untouched() {
            let shard = Uuid::new_v4();
            let before = table(&[0], Uuid::new_v4(), progress(shard, 5));
            let mut after = before.clone();
            let untouched = after.clone();
            apply(&mut after, &before, &[0], &before, false).unwrap();
            assert_eq!(after, untouched);
        }

        /// Two shards, only one of them compacted.
        #[test]
        fn each_shard_is_credited_independently() {
            let merged = Uuid::new_v4();
            let idle = Uuid::new_v4();
            let details = MemWalIndexDetails {
                compacted_sstables: vec![
                    CompactedSsTable::new(merged, 4),
                    CompactedSsTable::new(idle, 0),
                ],
                ..Default::default()
            };
            let read = table(&[0], Uuid::new_v4(), details.clone());
            let mut after = table(&[0], Uuid::new_v4(), details);
            apply(&mut after, &read, &[0], &read, true).unwrap();
            let coverage = coverage_for(&after, "idx").expect("credited");
            assert_eq!(
                coverage
                    .iter()
                    .find(|g| g.shard_id == merged)
                    .map(|g| g.generation),
                Some(4)
            );
            assert_eq!(
                coverage
                    .iter()
                    .find(|g| g.shard_id == idle)
                    .map(|g| g.generation),
                Some(0)
            );
        }

        /// Two indexes advance independently: one covering, one behind.
        #[test]
        fn indexes_are_credited_independently() {
            let shard = Uuid::new_v4();
            let read = vec![
                user_index("fast", Uuid::new_v4(), &[0, 1]),
                user_index("slow", Uuid::new_v4(), &[0]),
                mem_wal_index(progress(shard, 6)),
            ];
            let mut after = read.clone();
            apply(&mut after, &read, &[0, 1], &read, true).unwrap();
            assert_eq!(coverage_for(&after, "fast"), Some(compacted(shard, 6)));
            assert_eq!(coverage_for(&after, "slow"), None);
        }

        /// An index whose coverage is unknown cannot be shown to cover anything.
        #[test]
        fn an_index_without_a_bitmap_is_not_credited() {
            let shard = Uuid::new_v4();
            let mut idx = user_index("idx", Uuid::new_v4(), &[0]);
            idx.fragment_bitmap = None;
            let read = vec![idx, mem_wal_index(progress(shard, 5))];
            let mut after = read.clone();
            apply(&mut after, &read, &[0], &read, true).unwrap();
            assert_eq!(coverage_for(&after, "idx"), None);
        }

        /// Nothing compacted means nothing to be behind on.
        #[test]
        fn no_compaction_progress_writes_no_entries() {
            let before = table(&[0], Uuid::new_v4(), MemWalIndexDetails::default());
            let mut after = before.clone();
            let untouched = after.clone();
            apply(&mut after, &before, &[0], &before, true).unwrap();
            assert_eq!(after, untouched);
        }

        /// No MemWAL system index: nothing to maintain, and no error.
        #[test]
        fn a_table_without_mem_wal_is_a_no_op() {
            let before = vec![user_index("idx", Uuid::new_v4(), &[0])];
            let mut after = before.clone();
            let untouched = after.clone();
            apply(&mut after, &before, &[0], &before, true).unwrap();
            assert_eq!(after, untouched);
        }

        /// No read version -- dataset creation, detached commits -- credits
        /// nothing and lowers nothing.
        #[test]
        fn without_a_read_version_nothing_changes() {
            let shard = Uuid::new_v4();
            let uuid = Uuid::new_v4();
            let before = table(&[0], uuid, progress_with_catchup(shard, 5, 5));
            let mut after = before.clone();
            let segments_before = Transaction::logical_index_segments(&before);
            Transaction::apply_mem_wal_index_coverage(&mut after, &segments_before, None, true, 2)
                .unwrap();
            assert_eq!(coverage_for(&after, "idx"), Some(compacted(shard, 5)));
        }

        /// An untrained index covers nothing that exists, so a sibling's work
        /// is no evidence for it.
        #[test]
        fn an_untrained_index_earns_nothing() {
            let shard = Uuid::new_v4();
            let read = vec![
                user_index("untrained", Uuid::new_v4(), &[]),
                user_index("trained", Uuid::new_v4(), &[0]),
                mem_wal_index(progress(shard, 10)),
            ];
            let mut after = read.clone();
            apply(&mut after, &read, &[0], &read, true).unwrap();
            assert_eq!(coverage_for(&after, "untrained"), None);
            assert_eq!(coverage_for(&after, "trained"), Some(compacted(shard, 10)));
        }

        /// Shards move independently within one index: one advances on this
        /// commit's proof while another keeps the position it already had.
        #[test]
        fn a_shard_keeps_its_position_while_another_advances() {
            let (advancing, quiet) = (Uuid::new_v4(), Uuid::new_v4());
            let uuid = Uuid::new_v4();
            let details = |advancing_gen: u64| MemWalIndexDetails {
                compacted_sstables: vec![
                    CompactedSsTable::new(advancing, advancing_gen),
                    CompactedSsTable::new(quiet, 10),
                ],
                index_catchup: vec![IndexCatchupProgress::new(
                    "idx".to_string(),
                    vec![CompactedSsTable::new(quiet, 7)],
                )],
                ..Default::default()
            };
            // The quiet shard was never compacted as of the read, so nothing
            // this commit proves reaches it -- it keeps its recorded 7.
            let read = vec![
                user_index("idx", uuid, &[0]),
                mem_wal_index(MemWalIndexDetails {
                    compacted_sstables: vec![CompactedSsTable::new(advancing, 9)],
                    ..details(9)
                }),
            ];
            let mut after = vec![user_index("idx", uuid, &[0]), mem_wal_index(details(10))];
            apply(&mut after, &read, &[0], &read, true).unwrap();

            let mut coverage = coverage_for(&after, "idx").expect("credited");
            coverage.sort_unstable_by_key(|sstable| sstable.shard_id);
            let mut expected = vec![
                CompactedSsTable::new(advancing, 9),
                CompactedSsTable::new(quiet, 7),
            ];
            expected.sort_unstable_by_key(|sstable| sstable.shard_id);
            assert_eq!(coverage, expected);
        }

        /// The derivation drops coverage an index no longer earns, but it never
        /// rejects the commit -- an ordinary index job must not be blocked by
        /// a protocol it knows nothing about.
        #[test]
        fn an_ordinary_index_job_is_never_blocked() {
            let shard = Uuid::new_v4();
            let before = table(&[0, 1], Uuid::new_v4(), progress_with_catchup(shard, 5, 5));
            // Rebuilt over a subset -- the shape a partial reindex leaves.
            let mut after = table(&[0], Uuid::new_v4(), progress_with_catchup(shard, 5, 5));
            apply(&mut after, &before, &[0, 1], &before, true).unwrap();
            assert_eq!(coverage_for(&after, "idx"), None);
        }

        /// A reader's rule is that a missing entry means "not caught up", so an
        /// index caught up to nothing must be absent rather than present at
        /// generation zero -- otherwise it reads as known-and-covered.
        #[test]
        fn an_index_caught_up_to_nothing_gets_no_entry() {
            let shard = Uuid::new_v4();
            let uuid = Uuid::new_v4();
            let before = table(&[0], uuid, progress_with_catchup(shard, 5, 0));
            let mut after = before.clone();
            // Does not span the read version, so nothing lifts it off zero.
            apply(&mut after, &before, &[0, 1], &before, true).unwrap();
            assert_eq!(coverage_for(&after, "idx"), None);
        }

        /// Each shard carries its own position. Collapsing them to one value
        /// would credit a lagging shard with a busier shard's progress.
        #[test]
        fn carried_positions_do_not_leak_between_shards() {
            let (ahead, behind) = (Uuid::new_v4(), Uuid::new_v4());
            let uuid = Uuid::new_v4();
            let details = MemWalIndexDetails {
                compacted_sstables: vec![
                    CompactedSsTable::new(ahead, 10),
                    CompactedSsTable::new(behind, 10),
                ],
                index_catchup: vec![IndexCatchupProgress::new(
                    "idx".to_string(),
                    vec![
                        CompactedSsTable::new(ahead, 8),
                        CompactedSsTable::new(behind, 2),
                    ],
                )],
                ..Default::default()
            };
            let before = vec![user_index("idx", uuid, &[0]), mem_wal_index(details)];
            let mut after = before.clone();
            // Unchanged and unproven: both shards keep exactly what they had.
            apply(&mut after, &before, &[0, 1], &before, true).unwrap();

            let mut coverage = coverage_for(&after, "idx").expect("carried");
            coverage.sort_unstable_by_key(|sstable| sstable.shard_id);
            let mut expected = vec![
                CompactedSsTable::new(ahead, 8),
                CompactedSsTable::new(behind, 2),
            ];
            expected.sort_unstable_by_key(|sstable| sstable.shard_id);
            assert_eq!(coverage, expected);
        }

        /// The derivation runs while the manifest is being built, but the
        /// index list is not final there: `migrate_indices` recalculates a
        /// segment's fragment bitmap and keeps its UUID. A position decided
        /// before that must not survive the narrowing, or the WAL pod trims
        /// against an index that no longer covers those rows.
        #[test]
        fn a_bitmap_narrowed_after_the_build_loses_its_position() {
            let shard = Uuid::new_v4();
            let uuid = Uuid::new_v4();
            // What migrate_indices leaves behind: same UUID, fewer fragments,
            // and it says so.
            let mut migrated = table(&[0], uuid, progress_with_catchup(shard, 5, 5));

            Transaction::withdraw_coverage_invalidated_after_build(
                &mut migrated,
                &["idx".to_string()],
                3,
            )
            .unwrap();

            assert_eq!(coverage_for(&migrated, "idx"), None);
        }

        /// Migration routinely fills in file lists and inferred details. Those
        /// do not change which rows an index answers for, so withdrawing on
        /// them would drop coverage every commit for no reason.
        #[test]
        fn metadata_migration_that_does_not_narrow_keeps_its_position() {
            let shard = Uuid::new_v4();
            let uuid = Uuid::new_v4();
            let mut migrated = table(&[0, 1], uuid, progress_with_catchup(shard, 5, 5));
            migrated[0].files = Some(Vec::new());
            migrated[0].created_at = Some(chrono::Utc::now());

            // Nothing narrowed, so migration reports nothing.
            Transaction::withdraw_coverage_invalidated_after_build(&mut migrated, &[], 3).unwrap();

            assert_eq!(coverage_for(&migrated, "idx"), Some(compacted(shard, 5)));
        }

        /// A commit that changes nothing must not churn the system index: a new
        /// UUID on every append would invalidate its cache entry fleet-wide.
        #[test]
        fn an_unchanged_commit_does_not_rewrite_the_system_index() {
            let shard = Uuid::new_v4();
            let uuid = Uuid::new_v4();
            let before = table(&[0], uuid, progress_with_catchup(shard, 5, 5));
            let mut after = before.clone();
            apply(&mut after, &before, &[0], &before, true).unwrap();

            let system_uuid = |indices: &[IndexMetadata]| {
                indices
                    .iter()
                    .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
                    .unwrap()
                    .uuid
            };
            assert_eq!(system_uuid(&after), system_uuid(&before));
        }

        /// Activation is what puts a table on the protocol. A table that has
        /// never compacted is clean.
        #[test]
        fn activation_accepts_a_clean_table() {
            let mut indices = vec![mem_wal_index(MemWalIndexDetails::default())];
            Transaction::require_index_catchup(&mut indices, 2).unwrap();
        }

        /// There is nothing to put on the protocol.
        #[test]
        fn activation_requires_the_mem_wal_index() {
            let err = Transaction::require_index_catchup(&mut [], 2).unwrap_err();
            assert!(err.to_string().contains("does not exist"), "{err}");
        }

        /// Coverage recorded under the beta rules was written to a different
        /// contract; keeping it would let the first trim run unchecked.
        #[test]
        fn activation_clears_beta_coverage() {
            let shard = Uuid::new_v4();
            let mut indices = vec![mem_wal_index(MemWalIndexDetails {
                index_catchup: vec![IndexCatchupProgress::new(
                    "idx".to_string(),
                    compacted(shard, 100),
                )],
                ..Default::default()
            })];

            Transaction::require_index_catchup(&mut indices, 2).unwrap();

            assert!(
                load_mem_wal_index_details(indices[0].clone())
                    .unwrap()
                    .index_catchup
                    .is_empty()
            );
        }

        /// Beta compaction progress means SSTables were folded in without any
        /// coverage rule. No later commit can prove which indexes hold them.
        #[test]
        fn activation_rejects_pre_existing_beta_compaction_progress() {
            let mut indices = vec![mem_wal_index(progress(Uuid::new_v4(), 4))];
            let err = Transaction::require_index_catchup(&mut indices, 2).unwrap_err();
            assert!(err.to_string().contains("beta protocol"), "{err}");
        }

        fn config_transaction(current: &Manifest) -> Transaction {
            Transaction::new(
                current.version,
                Operation::UpdateConfig {
                    config_updates: None,
                    table_metadata_updates: None,
                    schema_metadata_updates: None,
                    field_metadata_updates: HashMap::new(),
                },
                None,
            )
        }

        /// One bit without the other is a manifest no writer should produce:
        /// a reader-only bit lets an unaware writer trim, a writer-only bit
        /// lets an unaware reader serve rows no index holds.
        #[test]
        fn a_half_set_feature_bit_is_refused() {
            for (reader, writer) in [
                (FLAG_MEM_WAL_INDEX_CATCHUP, 0),
                (0, FLAG_MEM_WAL_INDEX_CATCHUP),
            ] {
                let mut current = sample_manifest_with_fragments(0..1);
                current.reader_feature_flags = reader;
                current.writer_feature_flags = writer;

                let err = config_transaction(&current)
                    .build_manifest(
                        Some(&current),
                        vec![mem_wal_index(MemWalIndexDetails::default())],
                        "txn",
                        &default_build_config(),
                    )
                    .unwrap_err();

                assert!(err.to_string().contains("only one of"), "{err}");
            }
        }

        /// A writer that knows nothing about catch-up must not silently take a
        /// table off the protocol.
        #[test]
        fn an_ordinary_commit_keeps_the_feature_bit() {
            let mut current = sample_manifest_with_fragments(0..1);
            current.reader_feature_flags = FLAG_MEM_WAL_INDEX_CATCHUP;
            current.writer_feature_flags = FLAG_MEM_WAL_INDEX_CATCHUP;

            let (next, _) = config_transaction(&current)
                .build_manifest(
                    Some(&current),
                    vec![mem_wal_index(MemWalIndexDetails::default())],
                    "txn",
                    &default_build_config(),
                )
                .unwrap();

            assert_ne!(next.reader_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP, 0);
            assert_ne!(next.writer_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP, 0);
        }

        /// A commit with no read version still withdraws. It can prove nothing,
        /// so an index it changed keeps no position -- the alternative leaves a
        /// position describing an index that no longer exists.
        #[test]
        fn without_a_read_version_a_changed_index_still_loses_its_position() {
            let shard = Uuid::new_v4();
            let before = table(&[0, 1], Uuid::new_v4(), progress_with_catchup(shard, 5, 5));
            let mut after = table(&[0], Uuid::new_v4(), progress_with_catchup(shard, 5, 5));
            let segments_before = Transaction::logical_index_segments(&before);
            Transaction::apply_mem_wal_index_coverage(&mut after, &segments_before, None, true, 2)
                .unwrap();
            assert_eq!(coverage_for(&after, "idx"), None);
        }

        /// Two attempts against the same read version agree, which is what makes
        /// a rebase safe: `read_version` is fixed for a transaction's life.
        #[test]
        fn the_derivation_is_stable_across_attempts() {
            let shard = Uuid::new_v4();
            let read = table(&[0, 1], Uuid::new_v4(), progress(shard, 5));
            let mut first = table(&[0, 1], Uuid::new_v4(), progress(shard, 5));
            let mut second = first.clone();
            apply(&mut first, &read, &[0, 1], &read, true).unwrap();
            apply(&mut second, &read, &[0, 1], &read, true).unwrap();
            assert_eq!(coverage_for(&first, "idx"), coverage_for(&second, "idx"));
        }
    }
}
