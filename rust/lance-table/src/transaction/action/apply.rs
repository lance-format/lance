// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Applying an action set to produce the next manifest.
//!
//! Each action is applied in order against a working copy of the version being
//! committed on top of. Minting actions allocate an id from the target's counters as they are
//! reached and record it against their local token, so every later reference to
//! that token -- in this same operation -- resolves to the id this apply chose.
//! Replaying the same action set against a different version therefore produces
//! different ids without any of the actions changing.
//!
//! An action naming an id an earlier commit reserved is the exception: that id
//! is already allocated, so apply checks the claim rather than making one, and
//! replay leaves it alone.
//!
//! Reservations are the one exception to applying in order: whatever their
//! position, they are summed and taken off the counters after the operation's
//! own mints. A reservation creates nothing another action can name, so the
//! order has no other visible effect.
//!
//! [`ApplyState`] is that working copy, and its methods are the API the action
//! modules program against. What each action does with it lives in that action's
//! own module.

use super::{CompositeOperation, FieldId, FieldRef, Ref, UserStep};
use crate::feature_flags::{
    FLAG_MANAGED_BLOBS, apply_index_object_metadata_flag, inherit_sticky_feature_flags,
    managed_blobs_enabled, validate_auto_cleanup_config,
};
use crate::format::overlay::staleness::field_affects_index;
use crate::format::{
    BasePath, DataFile, Fragment, IndexMetadata, Manifest, ManifestBuildConfig, RowIdMeta,
};
use crate::rowids::read_row_ids;
use crate::rowids::version::{LoadedRowLineage, SpilledRowLineage, build_version_meta};
use crate::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use crate::system_index::frag_reuse::FragReuseActionInputs;
use crate::system_index::is_system_index;
use crate::system_index::mem_wal::{
    CompactedSsTable, MEM_WAL_INDEX_NAME, update_mem_wal_index_compacted_sstables,
};
use crate::transaction::manifest_build::{apply_range_segments, check_blob_nesting};
use crate::transaction::row_version::advance_row_id_counter;
use crate::transaction::validate::schema_fragments_valid;
use crate::transaction::{FragReuseUpdate, LogicalIndexSegments, ReadVersionState, Transaction};
use lance_core::datatypes::{Field, Schema};
use lance_core::{Error, Result, is_system_column};
use roaring::RoaringTreemap;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;
use working_fragments::{WorkingFragments, fresh_row_id_count};

mod frag_reuse_history;
mod working_fragments;

/// The field id written into a data file's field list once the file no longer
/// backs that field. A file whose every slot is tombstoned is dropped.
pub(super) const TOMBSTONED_FIELD: i32 = -2;

/// The largest fragment id a manifest can record: its high-water mark is a
/// `u32`, so an id past it could be handed out again.
const MAX_FRAGMENT_ID: u64 = u32::MAX as u64;

impl Transaction {
    /// Build the next manifest by applying an action set.
    ///
    /// Takes the current manifest by reference, unlike the legacy path: an
    /// action set is a delta, so there is nothing for it to be a delta against
    /// when the dataset does not exist yet. The caller turns the absent case
    /// into an error, which is where the `Option` it holds actually lives.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::transaction) fn build_manifest_from_actions(
        &self,
        composite_operation: &CompositeOperation,
        current_manifest: &Manifest,
        current_indices: Vec<IndexMetadata>,
        transaction_file_path: &str,
        config: &ManifestBuildConfig,
        read_version_state: Option<ReadVersionState<'_>>,
        frag_reuse: &FragReuseUpdate,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        reject_unsupported_build_config(config, current_manifest)?;

        // Committed ids are checked against the version the writer read, not
        // the one applied onto: an id minted in between belongs to whichever
        // concurrent commit minted it. Without the read version that check
        // has nothing to run against, so only an apply onto the read version
        // itself -- a detached commit, or a build with no history -- proceeds.
        let read_manifest = match &read_version_state {
            Some(read) => read.manifest,
            None if current_manifest.version == self.read_version => current_manifest,
            None => {
                return Err(Error::invalid_input(format!(
                    "an action-based transaction that read version {} is being applied onto \
                     version {} without the read version's manifest, so its committed ids \
                     cannot be checked against what it read",
                    self.read_version, current_manifest.version
                )));
            }
        };

        let mut state = ApplyState::new(
            Arc::new(current_manifest.clone()),
            current_indices,
            config,
            self.read_version,
            ReadMarks::of(read_manifest),
        )?;
        if let FragReuseUpdate::Actions(inputs) = frag_reuse {
            state.frag_reuse_inputs = inputs.clone();
        }
        for step in &composite_operation.steps {
            state.apply_step(step)?;
        }

        state.finish(
            self.tag.as_deref(),
            Some(transaction_file_path),
            read_version_state,
        )
    }
}

/// The id high-water marks of the version the operation read. A committed id
/// above them names something the writer cannot have seen.
#[derive(Clone)]
struct ReadMarks {
    fragment: Option<u64>,
    field: i32,
    base: u32,
    /// The read version's row id watermark; `None` when it has no stable row
    /// ids, so no row id could have been reserved.
    next_row_id: Option<u64>,
}

impl ReadMarks {
    fn of(manifest: &Manifest) -> Self {
        Self {
            fragment: manifest.max_fragment_id(),
            field: manifest.max_field_id(),
            base: max_base_id(manifest),
            next_row_id: manifest
                .uses_stable_row_ids()
                .then_some(manifest.next_row_id),
        }
    }
}

fn max_base_id(manifest: &Manifest) -> u32 {
    manifest.base_paths.keys().max().copied().unwrap_or(0)
}

fn reject_unsupported_build_config(
    config: &ManifestBuildConfig,
    current_manifest: &Manifest,
) -> Result<()> {
    if config.clustering_definitions.is_some() {
        return Err(Error::invalid_input(
            "a complete clustering definitions list can only be committed with UpdateConfig",
        ));
    }
    // No `migration_next_row_id` escape, unlike the legacy guard in
    // `build_manifest_with_read_version`: `migrate_to_stable_row_ids`
    // activates the flag with an `Operation::Merge`, so an action set
    // never carries that watermark.
    if config.use_stable_row_ids && !current_manifest.uses_stable_row_ids() {
        return Err(Error::not_supported_source(
            "This dataset was not created with the stable row ids feature.  Please run \
             `migrate_to_stable_row_ids` before attempting to use stable row ids"
                .into(),
        ));
    }
    Ok(())
}

fn merge_loaded_row_lineage(
    existing: &LoadedRowLineage,
    incoming: &LoadedRowLineage,
    fragment_id: u64,
) -> Result<(LoadedRowLineage, bool)> {
    fn merge_field<T: PartialEq>(
        existing: &Option<Arc<T>>,
        incoming: &Option<Arc<T>>,
        fragment_id: u64,
        field_name: &str,
    ) -> Result<(Option<Arc<T>>, bool)> {
        match (existing, incoming) {
            (Some(a), Some(b)) => {
                if !Arc::ptr_eq(a, b) && a != b {
                    return Err(Error::invalid_input(format!(
                        "spilled {field_name} lineage for fragment {fragment_id} conflicts with \
                         lineage already loaded for that fragment"
                    )));
                }
                Ok((Some(Arc::clone(a)), false))
            }
            (Some(a), None) => Ok((Some(Arc::clone(a)), false)),
            (None, Some(b)) => Ok((Some(Arc::clone(b)), true)),
            (None, None) => Ok((None, false)),
        }
    }

    let (row_ids, added_ids) =
        merge_field(&existing.row_ids, &incoming.row_ids, fragment_id, "row id")?;
    let (created_at, added_created) = merge_field(
        &existing.created_at,
        &incoming.created_at,
        fragment_id,
        "created_at",
    )?;
    let (last_updated_at, added_updated) = merge_field(
        &existing.last_updated_at,
        &incoming.last_updated_at,
        fragment_id,
        "last_updated_at",
    )?;
    Ok((
        LoadedRowLineage {
            row_ids,
            created_at,
            last_updated_at,
        },
        added_ids || added_created || added_updated,
    ))
}

/// The state of the version being committed on top of, plus the id
/// allocations made so far.
///
/// The state owns everything it works on, so it can outlive one build. A
/// writer that stages its steps one at a time keeps one state between steps.
/// It applies and validates each step on a clone and keeps the clone only when
/// both succeed. [`Self::build_manifest`] reads the table the steps produce so
/// far without consuming the kept state.
///
/// While staging on `read_version`, [`Self::build_manifest`] assigns the same
/// fragment ids and fresh row ids that committing the steps so far onto
/// `read_version` produces, and removing an earlier staged fragment keeps its
/// row-id slot so later staged fragments do not shift ids between snapshots.
/// Replaying the steps onto a newer committed version renumbers local fragment
/// ids and fresh row ids from that version's watermarks, so callers that expose
/// intermediate ids or stage steps that depend on them must require the commit
/// to land on `read_version` rather than rebase.
///
/// ```
/// # use std::sync::Arc;
/// # use lance_core::Result;
/// # use lance_table::format::{IndexMetadata, Manifest, ManifestBuildConfig};
/// # use lance_table::transaction::action::{ApplyState, UserStep};
/// # fn example(
/// #     base: Arc<Manifest>,
/// #     indices: Vec<IndexMetadata>,
/// #     config: &ManifestBuildConfig,
/// #     steps: &[UserStep],
/// # ) -> Result<()> {
/// let mut state = ApplyState::at_read_version(base, indices, config)?;
/// for step in steps {
///     state = state.apply_step_checked(step)?;
/// }
/// let (manifest, indices) = state.build_manifest()?;
/// # let _ = (manifest, indices);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct ApplyState {
    /// The version being committed on top of: the read version on a first
    /// attempt, the latest committed version on a retry.
    current_manifest: Arc<Manifest>,
    read_marks: ReadMarks,
    /// Options controlling how manifests are assembled from this state. Fixed
    /// at creation; `spilled_row_lineage` is tracked separately below so later
    /// steps can extend it without cloning the config.
    build_config: Arc<ManifestBuildConfig>,
    /// Row lineage the caller read ahead of time for fragments whose lineage
    /// lives in a data file, which applying cannot read itself. Seeded from
    /// `build_config.spilled_row_lineage` and grown by
    /// [`Self::extend_spilled_row_lineage`].
    spilled_row_lineage: Arc<SpilledRowLineage>,
    /// What each logical index looked like before any action ran, which MemWAL
    /// coverage is compared against. Only a table with a MemWAL index
    /// maintains coverage, so every other state holds nothing.
    mem_wal_segments_before: Option<Arc<LogicalIndexSegments>>,
    schema: Schema,
    /// Copy-on-write fragment working set over the manifest's base fragments,
    /// with lazy masking of fields dropped by [`DropField`](super::DropField).
    fragments: WorkingFragments,
    added_fragment_order: Arc<Vec<AddedFragment>>,
    /// Running tally of fresh row ids the added fragments consume, removed
    /// ones included. Exact while no added fragment supplies inline row ids.
    fresh_row_ids_added: u64,
    /// An added fragment supplied inline row ids or could not be counted, so
    /// row id counter validation must number copies of the added fragments.
    row_ids_need_dry_run: bool,
    /// An added fragment carried row id metadata on a table without a stable
    /// row id counter, so feature-flag validation must inspect all fragments.
    added_row_ids_without_counter: bool,
    /// Changes accumulated across unvalidated [`Self::apply_step`] calls.
    unvalidated: UnvalidatedScope,
    /// Base paths minted by this operation. Kept apart from the manifest's own
    /// base paths, which the manifest assembly inherits from the version being
    /// committed on top of.
    new_bases: Arc<Vec<BasePath>>,
    existing_base_paths: Arc<HashMap<u32, BasePath>>,
    /// The index segments, as the actions have left them so far. Index
    /// actions edit this list; the assembly then prunes whatever the data
    /// actions invalidated.
    indices: Arc<Vec<IndexMetadata>>,
    /// The manifest's string maps. Unlike the schema and the fragment list,
    /// these are inherited wholesale by the manifest assembly, so an edit has
    /// to be written back over the assembled manifest.
    config: Arc<HashMap<String, String>>,
    table_metadata: Arc<HashMap<String, String>>,

    /// The highest id handed out so far in each space, by the version being
    /// committed on top of or by this operation. Kept as marks rather than
    /// next ids so a counter at its maximum is representable, and the next
    /// mint fails rather than wraps. No fragment id has been handed out while
    /// the fragment mark is `None`; base ids start at 1.
    fragment_high_water: Option<u64>,
    field_high_water: i32,
    base_high_water: u32,

    /// Local token -> the id minted for it, one map per id space.
    fragment_tokens: Arc<HashMap<u32, u64>>,
    field_tokens: Arc<HashMap<u32, i32>>,
    base_tokens: Arc<HashMap<u32, u32>>,

    /// Ids of the fragments this operation added, whether minted from the
    /// counter or claimed out of a reservation.
    new_fragments: Arc<HashSet<u64>>,
    /// The new fragments an [`AddFragment`](super::AddFragment) added with
    /// `data_change = false`: rows moved from elsewhere in the table, whose
    /// values no action in the operation may change.
    rearranged_fragments: Arc<HashSet<u64>>,

    /// How many fragment ids this operation reserved for a later writer. Like
    /// [`Self::reserved_row_ids`], taken off the counter after this
    /// operation's own fragments are numbered.
    reserved_fragment_ids: u64,

    /// How many row ids this operation reserved for a later writer. Taken off
    /// the counter after this operation's own fragments are numbered, so the
    /// reserved range is always the top `count` ids of the resulting watermark.
    reserved_row_ids: u64,

    /// Names of the logical indices this operation added a segment to or
    /// adjusted, whose segment sets are validated once the operation has run.
    touched_index_names: Arc<BTreeSet<String>>,

    /// Every field a [`TombstoneFieldData`](super::TombstoneFieldData) freed,
    /// by fragment and field, so a re-encode can be checked to re-bind what it
    /// tombstoned.
    tombstoned_fields: Arc<HashMap<(u64, i32), TombstonedField>>,

    /// Committed fragments whose values a data-changing action changed in
    /// place, which on a stable row id table leaves their row versions stale,
    /// and the committed fragments a
    /// [`RefreshRowVersionMetadata`](super::RefreshRowVersionMetadata)
    /// restamped, wholly or in part. Every value-changed fragment has to be
    /// restamped.
    value_changed_fragments: Arc<BTreeSet<u64>>,
    restamped_fragments: Arc<BTreeSet<u64>>,

    /// Fragments a [`RemoveFragment`](super::RemoveFragment) removed: a
    /// fragment reuse transition may only consume a fragment a compaction
    /// removed as it was committed.
    removed_fragments: Arc<HashMap<u64, RemovedFragment>>,
    /// Committed fields whose values a data-changing tombstone replaced, by
    /// fragment. On a tagged fragment reuse history the coverage of segments
    /// translating into the fragment is withdrawn along the lineage.
    rebound_fields: Arc<Vec<(u64, Vec<u32>)>>,
    /// What the operation asks of the fragment reuse history, settled once
    /// the whole operation has run.
    frag_reuse: frag_reuse_history::FragReuseChanges,
    /// What the commit attempt supplied for the history.
    frag_reuse_inputs: Arc<FragReuseActionInputs>,

    /// The version the transaction was built against. Not the current
    /// manifest's version: on a retry the set is replayed onto something newer,
    /// and what an action saw is still the version it read.
    read_version: u64,
}

/// Pending changes made by [`ApplyState::apply_step`] since the state was
/// created or last validated by [`ApplyState::validate_step_invariants`].
#[derive(Clone, Default)]
struct UnvalidatedScope {
    dirty_fragments: Vec<u64>,
    changed_schema: bool,
    changed_config: bool,
}

/// One fragment this operation added, in the order it was added.
///
/// A removed fragment keeps its slot because an intermediate manifest build may
/// already have exposed the row ids of fragments added after it.
#[derive(Clone)]
enum AddedFragment {
    Present(u64),
    Removed { fresh_row_ids: u64 },
}

/// A field a [`TombstoneFieldData`](super::TombstoneFieldData) freed.
#[derive(Debug, Clone, Copy)]
struct TombstonedField {
    data_change: bool,
    /// Whether an [`AddDataFile`](super::AddDataFile) has bound it again.
    rebound: bool,
}

/// A fragment a [`RemoveFragment`](super::RemoveFragment) removed.
#[derive(Debug, Clone, Copy)]
struct RemovedFragment {
    data_change: bool,
    /// Whether an earlier action in the operation changed it from the
    /// committed fragment.
    modified: bool,
}

impl ApplyState {
    /// Start a state on `base`, the version the transaction read. The read
    /// marks come from that same manifest.
    pub fn at_read_version(
        base: Arc<Manifest>,
        indices: Vec<IndexMetadata>,
        config: &ManifestBuildConfig,
    ) -> Result<Self> {
        reject_unsupported_build_config(config, &base)?;
        let read_version = base.version;
        let read_marks = ReadMarks::of(&base);
        Self::new(base, indices, config, read_version, read_marks)
    }

    fn new(
        manifest: Arc<Manifest>,
        indices: Vec<IndexMetadata>,
        build_config: &ManifestBuildConfig,
        read_version: u64,
        read_marks: ReadMarks,
    ) -> Result<Self> {
        let fragments = WorkingFragments::from_manifest(&manifest)?;
        let fragment_high_water = match (manifest.max_fragment_id(), fragments.max_base_id()) {
            (Some(mark), Some(base_max)) => Some(mark.max(base_max)),
            (mark, base_max) => mark.or(base_max),
        };
        Ok(Self {
            read_version,
            read_marks,
            spilled_row_lineage: Arc::clone(&build_config.spilled_row_lineage),
            build_config: Arc::new(build_config.clone()),
            mem_wal_segments_before: indices
                .iter()
                .any(|index| index.name == MEM_WAL_INDEX_NAME)
                .then(|| Arc::new(Transaction::logical_index_segments(&indices))),
            schema: manifest.schema.clone(),
            fragments,
            added_fragment_order: Default::default(),
            fresh_row_ids_added: 0,
            row_ids_need_dry_run: false,
            added_row_ids_without_counter: false,
            unvalidated: UnvalidatedScope::default(),
            indices: Arc::new(indices),
            new_bases: Default::default(),
            existing_base_paths: Arc::new(manifest.base_paths.clone()),
            config: Arc::new(manifest.config.clone()),
            table_metadata: Arc::new(manifest.table_metadata.clone()),
            fragment_high_water,
            field_high_water: manifest.max_field_id(),
            base_high_water: max_base_id(&manifest),
            fragment_tokens: Default::default(),
            field_tokens: Default::default(),
            base_tokens: Default::default(),
            new_fragments: Default::default(),
            rearranged_fragments: Default::default(),
            reserved_fragment_ids: 0,
            reserved_row_ids: 0,
            touched_index_names: Default::default(),
            tombstoned_fields: Default::default(),
            value_changed_fragments: Default::default(),
            restamped_fragments: Default::default(),
            removed_fragments: Default::default(),
            rebound_fields: Default::default(),
            frag_reuse: Default::default(),
            frag_reuse_inputs: Default::default(),
            current_manifest: manifest,
        })
    }

    /// Apply one step's actions in order.
    ///
    /// A step that fails leaves the state partly changed, and changes applied
    /// here remain in the pending validation scope until
    /// [`Self::validate_step_invariants`] succeeds. Apply to a clone or use
    /// [`Self::apply_step_checked`] when the previous state must survive a
    /// rejected step.
    pub(crate) fn apply_step(&mut self, step: &UserStep) -> Result<()> {
        for action in &step.actions {
            action.apply(self)?;
        }
        Ok(())
    }

    /// Merge spilled row lineage loaded for fragments a later step touches.
    ///
    /// Missing sequence slots on an existing fragment entry are filled in, and
    /// conflicting sequences for the same fragment id are rejected.
    pub fn extend_spilled_row_lineage(&mut self, loaded: &Arc<SpilledRowLineage>) -> Result<()> {
        if loaded.is_empty() || Arc::ptr_eq(&self.spilled_row_lineage, loaded) {
            return Ok(());
        }
        if self.spilled_row_lineage.is_empty() {
            self.spilled_row_lineage = Arc::clone(loaded);
            return Ok(());
        }
        let mut updates = Vec::new();
        for (&fragment_id, incoming) in loaded.iter() {
            match self.spilled_row_lineage.get(&fragment_id) {
                Some(existing) => {
                    let (merged, added) =
                        merge_loaded_row_lineage(existing, incoming, fragment_id)?;
                    if added {
                        updates.push((fragment_id, merged));
                    }
                }
                None => updates.push((fragment_id, incoming.clone())),
            }
        }
        if !updates.is_empty() {
            let map = Arc::make_mut(&mut self.spilled_row_lineage);
            for (fragment_id, merged) in updates {
                map.insert(fragment_id, merged);
            }
        }
        Ok(())
    }

    /// Spilled row lineage entry loaded for `fragment_id`, if any.
    ///
    /// An entry may carry only a subset of sequences, so callers that load
    /// individual sequences should inspect the fields they require.
    pub fn spilled_row_lineage_for(&self, fragment_id: u64) -> Option<&LoadedRowLineage> {
        self.spilled_row_lineage.get(&fragment_id)
    }

    /// Clone this state, apply `step`, and validate all pending invariants
    /// before returning the new state. Leaves `self` unchanged on failure.
    pub fn apply_step_checked(&self, step: &UserStep) -> Result<Self> {
        let mut next = self.clone();
        next.apply_step(step)?;
        next.validate_step_invariants()?;
        Ok(next)
    }

    /// Build the manifest and index list that the steps applied so far produce
    /// on the read version.
    ///
    /// Finishing stamps row ids and takes reservations once per operation, so
    /// a clone is finished and the kept state accepts more steps. The manifest
    /// names no transaction file because an uncommitted snapshot is not a
    /// published commit.
    pub fn build_manifest(&self) -> Result<(Manifest, Vec<IndexMetadata>)> {
        self.clone().into_manifest()
    }

    /// Consume this state and assemble the uncommitted manifest and index list
    /// for the read version without cloning the state.
    pub fn into_manifest(self) -> Result<(Manifest, Vec<IndexMetadata>)> {
        debug_assert_eq!(self.current_manifest.version, self.read_version);
        self.finish(None, None, None)
    }

    /// Validate all changes accumulated since creation or the last successful
    /// validation against [`Self::build_manifest`] invariants.
    ///
    /// Steps touching only fragments, index segments, or id reservations are
    /// validated against the pending dirty fragments without assembling a full
    /// manifest. Unvalidated schema, config, fragment reuse history, MemWAL,
    /// legacy unpromoted blob, or uncounted row-id changes fall back to
    /// [`Self::build_manifest`], as does any state the incremental check
    /// refuses so the returned error matches commit-time validation. Clears
    /// the pending validation scope only on success.
    pub(crate) fn validate_step_invariants(&mut self) -> Result<()> {
        let dirty_fragments = self.unvalidated_dirty_fragments()?;
        if self.needs_manifest_build_for_validation(&dirty_fragments)
            || self.check_step_incrementally(dirty_fragments).is_err()
        {
            self.build_manifest()?;
        }
        self.unvalidated = UnvalidatedScope::default();
        Ok(())
    }

    fn unvalidated_dirty_fragments(&self) -> Result<Vec<Fragment>> {
        let mut ids = self.unvalidated.dirty_fragments.clone();
        ids.sort_unstable();
        ids.dedup();
        let mut fragments = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(fragment) = self.live_fragment(id)? {
                fragments.push(fragment.into_owned());
            }
        }
        Ok(fragments)
    }

    fn needs_manifest_build_for_validation(&self, dirty_fragments: &[Fragment]) -> bool {
        let may_promote_a_blob = !dirty_fragments.is_empty()
            && self
                .schema
                .fields_pre_order()
                .any(|field| field.is_blob() && !field.is_blob_v2());
        let rewrote_the_mem_wal_entry = self.mem_wal_segments_before.is_some()
            && self.touched_index_names.contains(MEM_WAL_INDEX_NAME);
        self.unvalidated.changed_schema
            || self.unvalidated.changed_config
            || self.has_frag_reuse_history_work()
            || rewrote_the_mem_wal_entry
            || may_promote_a_blob
            || self.added_row_ids_without_counter
    }

    fn check_step_incrementally(&self, mut dirty_fragments: Vec<Fragment>) -> Result<()> {
        self.check_spilled_lineage_is_bound()?;
        self.check_reencodes_are_rebound()?;
        self.check_value_changes_are_restamped()?;
        self.check_row_id_counter()?;
        self.fragment_high_water_after_reservation()?;
        validate_index_segment_sets(&self.indices, &self.touched_index_names)?;
        Transaction::normalize_fragments(&mut dirty_fragments)?;
        schema_fragments_valid(Some(&self.current_manifest), &self.schema, &dirty_fragments)
    }

    /// Assemble the next manifest from the state the actions left behind.
    fn finish(
        mut self,
        tag: Option<&str>,
        transaction_file_path: Option<&str>,
        read_version_state: Option<ReadVersionState<'_>>,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        let current_manifest = Arc::clone(&self.current_manifest);
        let current_manifest = current_manifest.as_ref();
        let new_version = current_manifest.version + 1;

        self.check_spilled_lineage_is_bound()?;
        self.check_reencodes_are_rebound()?;
        self.check_value_changes_are_restamped()?;
        let publishes_frag_reuse = self.finish_frag_reuse_history()?;

        let mut next_row_id = current_manifest
            .uses_stable_row_ids()
            .then_some(current_manifest.next_row_id);
        self.assign_row_ids_to_new_fragments(&mut next_row_id, new_version)?;
        self.take_row_id_reservation(&mut next_row_id)?;
        self.take_fragment_id_reservation()?;
        let mut fragments = self.build_fragments()?;

        let Self {
            schema,
            indices,
            touched_index_names,
            new_bases,
            fragment_high_water,
            field_high_water,
            config: dataset_config,
            table_metadata,
            mem_wal_segments_before,
            build_config,
            ..
        } = self;

        // Actions build the schema one field at a time, so nothing guarantees
        // what building it from Arrow would. Checked before anything reads a
        // field's data type, which panics on a list with no child. A table
        // whose schema already failed stays committable.
        if check_schema(&current_manifest.schema).is_ok() {
            check_schema(&schema)?;
        }

        let mut indices = Arc::unwrap_or_clone(indices);
        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);
        validate_index_segment_sets(&indices, &touched_index_names)?;

        Transaction::normalize_fragments(&mut fragments)?;
        check_required_fields_are_stored(current_manifest, &schema, &fragments)?;

        // Applied once the final index list is known, so it sees exactly the
        // indices this commit publishes rather than what any one action
        // intended. Same position as on the legacy path: after the index list
        // has been pruned, before the manifest is assembled.
        if let Some(segments_before) = mem_wal_segments_before.as_deref() {
            Transaction::apply_mem_wal_index_coverage(
                &mut indices,
                segments_before,
                read_version_state,
                new_version,
            )?;
        }

        let mut manifest = Transaction::assemble_manifest(
            tag,
            Some(current_manifest),
            schema,
            fragments,
            &indices,
            HashMap::new(),
            false,
            // The actions do not yet say which fragments they leave untouched.
            None,
            &build_config,
        )?;
        // After assembly, which may promote a blob to v2.
        check_blob_nesting(&manifest, Some(current_manifest))?;
        // Checked on the final schema because several actions shape the key: a
        // ConfigUpdate declares it, an AlterField can loosen a key column's
        // nullability. A table whose key was invalid before this check existed
        // stays committable.
        if current_manifest.schema.verify_primary_key().is_ok() {
            manifest.schema.verify_primary_key()?;
        }
        inherit_sticky_feature_flags(&mut manifest, current_manifest)?;
        apply_index_object_metadata_flag(&mut manifest, &indices);
        if publishes_frag_reuse {
            frag_reuse_history::set_feature_flags(&mut manifest, &indices);
        }

        for base in Arc::unwrap_or_clone(new_bases) {
            manifest.base_paths.insert(base.id, base);
        }

        manifest.config = Arc::unwrap_or_clone(dataset_config);
        manifest.table_metadata = Arc::unwrap_or_clone(table_metadata);
        // Only a change is validated: settings inherited as they were, even
        // invalid ones, must not block an unrelated commit.
        if auto_cleanup_settings(&current_manifest.config)
            != auto_cleanup_settings(&manifest.config)
        {
            validate_auto_cleanup_config(&manifest)?;
        }
        // Both read the final schema and config, so they come after those land.
        apply_range_segments(&mut manifest, Some(current_manifest))?;
        if managed_blobs_enabled(&manifest.schema)? {
            manifest.reader_feature_flags |= FLAG_MANAGED_BLOBS;
            manifest.writer_feature_flags |= FLAG_MANAGED_BLOBS;
        }

        // The assembly derives the marks from what the manifest still holds,
        // which misses ids this operation reserved or minted and then
        // discarded (a reset), and ids the version being committed on top of
        // handed out before a reset emptied it. Raise them to cover every id
        // handed out, so none is handed out twice.
        if let Some(high_water) = fragment_high_water {
            let high_water = u32::try_from(high_water)
                .map_err(|_| exhausted_err("fragment id", MAX_FRAGMENT_ID))?;
            manifest.max_fragment_id = Some(
                manifest
                    .max_fragment_id
                    .map_or(high_water, |current| current.max(high_water)),
            );
        }
        // Only a table under the non-reusable field id contract records a
        // field mark; elsewhere field ids are derived from the schema and
        // files and are reusable by that table's own rules.
        if let Some(current) = manifest.max_allocated_field_id.as_mut() {
            *current = (*current).max(field_high_water);
        }

        manifest.transaction_file = transaction_file_path.map(str::to_owned);
        if let Some(next_row_id) = next_row_id {
            manifest.next_row_id = next_row_id;
        }

        Ok((manifest, indices))
    }

    pub(super) fn uses_stable_row_ids(&self) -> bool {
        self.current_manifest.uses_stable_row_ids()
    }

    /// The version the transaction read, which is the newest data an index
    /// segment added by this operation can have been built from.
    ///
    /// Deliberately not the current manifest's version. A set that loses a race
    /// is replayed against whatever won, and a segment built before that still
    /// reflects only what its writer could see -- stamping it with the newer
    /// version would claim coverage of rows it never read.
    pub(super) fn read_version(&self) -> u64 {
        self.read_version
    }

    /// The version this delta produces, for the actions that stamp it into what
    /// they write. Distinct from [`Self::read_version`]: a retry against a newer
    /// manifest re-runs the apply, so anything stamped with this is re-stamped
    /// rather than carried over.
    pub(super) fn new_version(&self) -> u64 {
        self.current_manifest.version + 1
    }

    /// Add an index segment. A segment's uuid identifies it, so re-adding one
    /// that is already there is a mistake rather than a replacement -- swapping
    /// a segment out is a [`RemoveIndexSegment`](super::RemoveIndexSegment)
    /// followed by an add.
    pub(super) fn add_index_segment(&mut self, segment: IndexMetadata) -> Result<()> {
        if self.indices.iter().any(|index| index.uuid == segment.uuid) {
            return Err(Error::invalid_input(format!(
                "index segment {} is already part of the dataset; a segment is added once",
                segment.uuid
            )));
        }
        Arc::make_mut(&mut self.touched_index_names).insert(segment.name.clone());
        Arc::make_mut(&mut self.indices).push(segment);
        Ok(())
    }

    /// Drop an index segment by uuid. A removal that names a segment the
    /// dataset does not have is a mistake rather than a no-op: it means the
    /// operation was planned against a different set of segments.
    pub(super) fn remove_index_segment(&mut self, uuid: Uuid, name: &str) -> Result<()> {
        let Some(position) = self.index_segment_position(uuid, name)? else {
            return Err(Error::invalid_input(format!(
                "index segment {uuid} is not part of the dataset, so it cannot be removed"
            )));
        };
        Arc::make_mut(&mut self.indices).remove(position);
        Ok(())
    }

    /// Remove every segment of the index `name`.
    pub(super) fn drop_index(&mut self, name: &str) -> Result<()> {
        let before = self.indices.len();
        Arc::make_mut(&mut self.indices).retain(|index| index.name != name);
        if self.indices.len() == before {
            return Err(Error::invalid_input(format!(
                "DropIndex names index '{name}', which has no segment in the dataset"
            )));
        }
        Ok(())
    }

    /// Record MemWAL SSTable compaction progress against the table's existing
    /// MemWAL index, which has to already be there -- progress against an index
    /// no one built is rejected rather than inventing the metadata.
    pub(super) fn update_compacted_sstables(
        &mut self,
        compacted_sstables: Vec<CompactedSsTable>,
    ) -> Result<()> {
        let new_version = self.new_version();
        update_mem_wal_index_compacted_sstables(
            Arc::make_mut(&mut self.indices).as_mut_slice(),
            new_version,
            compacted_sstables,
        )
    }

    /// Where the segment `uuid` lives, checking on the way that it really
    /// belongs to `name`. A mismatch means the operation was planned against a
    /// different set of segments, the same way a missing segment does.
    fn index_segment_position(&self, uuid: Uuid, name: &str) -> Result<Option<usize>> {
        let Some(position) = self.indices.iter().position(|index| index.uuid == uuid) else {
            return Ok(None);
        };
        let found = &self.indices[position].name;
        if found != name {
            return Err(Error::invalid_input(format!(
                "index segment {uuid} belongs to index '{found}', but the action names it as part \
                 of index '{name}'"
            )));
        }
        Ok(Some(position))
    }

    pub(super) fn index_segment_mut(
        &mut self,
        uuid: Uuid,
        name: &str,
    ) -> Result<&mut IndexMetadata> {
        let Some(position) = self.index_segment_position(uuid, name)? else {
            return Err(Error::invalid_input(format!(
                "index segment {uuid} is not part of the dataset, so it cannot be adjusted"
            )));
        };
        Arc::make_mut(&mut self.touched_index_names).insert(name.to_string());
        Ok(&mut Arc::make_mut(&mut self.indices)[position])
    }

    /// A shared handle, because callers hold the lineage across a mutable
    /// borrow of a fragment.
    pub(super) fn spilled_row_lineage(&self) -> Arc<SpilledRowLineage> {
        Arc::clone(&self.spilled_row_lineage)
    }

    pub(super) fn schema(&self) -> &Schema {
        &self.schema
    }

    pub(super) fn schema_mut(&mut self) -> &mut Schema {
        self.unvalidated.changed_schema = true;
        &mut self.schema
    }

    /// Check that a field may take `name` under `parent_id` (or at the top
    /// level, for `None`). No field there may have it already: names are
    /// unique among siblings, and this is the one place the schema is at hand
    /// to check it -- the footprint keeps two concurrent writers from taking
    /// one name, but a name already committed is only visible here.
    ///
    /// A top-level name may also not be a system column's: the scanner
    /// injects those, so a stored one would collide with it on read. This is
    /// the check the insert path makes.
    pub(super) fn check_field_name(
        &self,
        parent_id: Option<i32>,
        name: &str,
        action: &str,
    ) -> Result<()> {
        if parent_id.is_none() && is_system_column(name) {
            return Err(Error::invalid_input(format!(
                "The column {name} is a reserved name and cannot be used in a Lance dataset"
            )));
        }
        let siblings = match parent_id {
            None => &self.schema.fields,
            Some(parent_id) => {
                &self
                    .schema
                    .field_by_id(parent_id)
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "{action} names parent field {parent_id}, which does not exist"
                        ))
                    })?
                    .children
            }
        };
        if let Some(existing) = siblings.iter().find(|field| field.name == name) {
            return Err(Error::invalid_input(format!(
                "{action} names a field {name:?}, but field {} already has that name at the same \
                 level; field names are unique among siblings",
                existing.id
            )));
        }
        Ok(())
    }

    pub(super) fn config_mut(&mut self) -> &mut HashMap<String, String> {
        self.unvalidated.changed_config = true;
        Arc::make_mut(&mut self.config)
    }

    pub(super) fn table_metadata_mut(&mut self) -> &mut HashMap<String, String> {
        Arc::make_mut(&mut self.table_metadata)
    }

    pub(super) fn tombstone_dropped_field(&mut self, field_id: i32) -> Result<()> {
        self.fragments
            .tombstone_dropped_field(field_id, &self.schema)
    }

    pub(super) fn base_fragment(&self, fragment_id: u64) -> Option<&Fragment> {
        self.fragments.base_fragment(fragment_id)
    }

    pub(super) fn has_live_fragment(&self, fragment_id: u64) -> bool {
        self.fragments.contains(fragment_id)
    }

    pub(super) fn live_fragment(&self, fragment_id: u64) -> Result<Option<Cow<'_, Fragment>>> {
        self.fragments.fragment(fragment_id, &self.schema)
    }

    pub(super) fn live_fragment_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.fragments.live_ids()
    }

    fn added_fragments(&self) -> impl Iterator<Item = &Fragment> {
        self.added_fragment_order
            .iter()
            .filter_map(|slot| match slot {
                AddedFragment::Present(id) => self.fragments.patched.get(id)?.as_deref(),
                AddedFragment::Removed { .. } => None,
            })
    }

    fn build_fragments(&self) -> Result<Vec<Fragment>> {
        self.fragments.build_fragments(&self.schema)
    }

    pub(super) fn fragment_mut(&mut self, fragment_id: u64, action: &str) -> Result<&mut Fragment> {
        let fragment = self.fragments.get_mut(fragment_id, &self.schema, action)?;
        self.unvalidated.dirty_fragments.push(fragment_id);
        Ok(fragment)
    }

    pub(super) fn push_fragment(&mut self, fragment: Fragment) {
        if self.uses_stable_row_ids() {
            let supplies_row_ids = matches!(fragment.row_id_meta, Some(RowIdMeta::Inline(_)));
            let tally = fresh_row_id_count(&fragment)
                .ok()
                .and_then(|fresh| self.fresh_row_ids_added.checked_add(fresh));
            match tally {
                Some(tally) => {
                    self.fresh_row_ids_added = tally;
                    self.row_ids_need_dry_run |= supplies_row_ids;
                }
                None => self.row_ids_need_dry_run = true,
            }
        } else if fragment.row_id_meta.is_some() {
            self.added_row_ids_without_counter = true;
        }
        self.unvalidated.dirty_fragments.push(fragment.id);
        Arc::make_mut(&mut self.added_fragment_order).push(AddedFragment::Present(fragment.id));
        self.fragments.insert(fragment);
    }

    /// Drop a fragment and forget everything recorded about it. `false` if no
    /// such fragment was present.
    pub(super) fn remove_fragment(&mut self, fragment_id: u64, data_change: bool) -> Result<bool> {
        let Some(fragment) = self.fragments.fragment(fragment_id, &self.schema)? else {
            return Ok(false);
        };
        let burned_row_ids =
            if self.new_fragments.contains(&fragment_id) && self.uses_stable_row_ids() {
                fresh_row_id_count(&fragment)?
            } else {
                0
            };
        let modified = self
            .fragments
            .base_fragment(fragment_id)
            .is_some_and(|committed| {
                !std::ptr::eq(committed, fragment.as_ref()) && *committed != *fragment
            });
        self.fragments.remove(fragment_id);
        Arc::make_mut(&mut self.removed_fragments).insert(
            fragment_id,
            RemovedFragment {
                data_change,
                modified,
            },
        );
        if Arc::make_mut(&mut self.new_fragments).remove(&fragment_id)
            && let Some(slot_pos) = self
                .added_fragment_order
                .iter()
                .rposition(|slot| matches!(slot, AddedFragment::Present(id) if *id == fragment_id))
        {
            if burned_row_ids > 0 {
                Arc::make_mut(&mut self.added_fragment_order)[slot_pos] = AddedFragment::Removed {
                    fresh_row_ids: burned_row_ids,
                };
            } else {
                Arc::make_mut(&mut self.added_fragment_order).remove(slot_pos);
            }
        }
        Arc::make_mut(&mut self.rearranged_fragments).remove(&fragment_id);
        Arc::make_mut(&mut self.tombstoned_fields)
            .retain(|(fragment, _), _| *fragment != fragment_id);
        Arc::make_mut(&mut self.value_changed_fragments).remove(&fragment_id);
        Arc::make_mut(&mut self.restamped_fragments).remove(&fragment_id);
        Ok(true)
    }

    /// Empty the table: no schema, no fragments, no indices. The id counters
    /// keep going, so a field or fragment added afterwards never reuses an id an
    /// old file might still name.
    pub(super) fn reset(&mut self) -> Result<()> {
        let mut burned = Vec::new();
        let mut burned_total: Option<u64> = Some(0);
        if self.uses_stable_row_ids() {
            for slot in self.added_fragment_order.iter() {
                let fresh_row_ids = match slot {
                    AddedFragment::Removed { fresh_row_ids } => *fresh_row_ids,
                    AddedFragment::Present(id) => match self.fragments.get_unmasked(*id) {
                        Some(fragment) => fresh_row_id_count(fragment)?,
                        None => 0,
                    },
                };
                if fresh_row_ids > 0 {
                    burned_total = burned_total.and_then(|sum| sum.checked_add(fresh_row_ids));
                    burned.push(AddedFragment::Removed { fresh_row_ids });
                }
            }
        }
        match burned_total {
            Some(total) => {
                self.fresh_row_ids_added = total;
                self.row_ids_need_dry_run = false;
            }
            None => self.row_ids_need_dry_run = true,
        }
        self.schema.fields.clear();
        self.schema.metadata.clear();
        self.added_row_ids_without_counter = false;
        self.unvalidated.changed_schema = true;
        self.fragments.clear();
        self.added_fragment_order = Arc::new(burned);
        Arc::make_mut(&mut self.indices).clear();
        self.frag_reuse = Default::default();
        Arc::make_mut(&mut self.rebound_fields).clear();
        Arc::make_mut(&mut self.new_fragments).clear();
        Arc::make_mut(&mut self.rearranged_fragments).clear();
        Arc::make_mut(&mut self.tombstoned_fields).clear();
        Arc::make_mut(&mut self.value_changed_fragments).clear();
        Arc::make_mut(&mut self.restamped_fragments).clear();
        Ok(())
    }

    /// The base paths this apply can see: the current manifest's, plus the ones
    /// earlier actions in this operation minted.
    pub(super) fn bases(&self) -> impl Iterator<Item = &BasePath> {
        self.existing_base_paths
            .values()
            .chain(self.new_bases.iter())
    }

    pub(super) fn push_base(&mut self, base: BasePath) {
        Arc::make_mut(&mut self.new_bases).push(base);
    }

    /// The id to add a new fragment at: minted off the counter for a local
    /// token, or claimed out of a reservation for a committed id.
    pub(super) fn add_fragment_id(&mut self, reference: Ref) -> Result<u64> {
        match reference {
            Ref::Local(token) => self.mint_fragment(token),
            Ref::Committed(id) => self.claim_reserved_fragment_id(id),
        }
    }

    fn mint_fragment(&mut self, token: u32) -> Result<u64> {
        if self.fragment_tokens.contains_key(&token) {
            return Err(duplicate_token_err("fragment", token));
        }
        let id = match self.fragment_high_water {
            None => 0,
            Some(mark) => mark
                .checked_add(1)
                .filter(|id| *id <= MAX_FRAGMENT_ID)
                .ok_or_else(|| exhausted_err("fragment id", MAX_FRAGMENT_ID))?,
        };
        self.fragment_high_water = Some(id);
        Arc::make_mut(&mut self.fragment_tokens).insert(token, id);
        Arc::make_mut(&mut self.new_fragments).insert(id);
        Ok(id)
    }

    /// Claim an id an earlier commit set aside with
    /// [`ReserveFragmentIds`](super::ReserveFragmentIds).
    ///
    /// Only ids at or below the read version's high-water mark can be
    /// claimed. Anything above it was not reserved when the writer read the
    /// table: it is an id a later mint would hand out, or one a concurrent
    /// commit reserved for its own writer. A writer that guessed rather than
    /// reserved is rejected here instead of colliding later.
    fn claim_reserved_fragment_id(&mut self, id: u64) -> Result<u64> {
        let unreserved = |mark: Option<u64>| mark.is_none_or(|mark| id > mark);
        if unreserved(self.read_marks.fragment) || unreserved(self.fragment_high_water) {
            return Err(Error::invalid_input(format!(
                "AddFragment adds a fragment at id {id}, which no commit has reserved as of the \
                 read version: its fragment id high-water mark is {}. Reserve a range with \
                 ReserveFragmentIds and read the committed manifest's high-water mark back \
                 before naming an id, or name a new fragment by a Local ref.",
                display_mark(self.read_marks.fragment)
            )));
        }
        if self.has_live_fragment(id) {
            return Err(Error::invalid_input(format!(
                "AddFragment adds a fragment at id {id}, which already exists"
            )));
        }
        // Coverage bitmaps are not pruned when a fragment is removed, and
        // readers intersect them with the fragments that exist, so an id a
        // segment still lists would read as indexed once claimed.
        if let Ok(bitmap_id) = u32::try_from(id)
            && let Some(index) = self.indices.iter().find(|index| {
                index
                    .fragment_bitmap
                    .as_ref()
                    .is_some_and(|bitmap| bitmap.contains(bitmap_id))
            })
        {
            return Err(Error::invalid_input(format!(
                "AddFragment adds a fragment at id {id}, which index '{}' segment {} still lists \
                 in its coverage; reserve a fresh range rather than reuse this id",
                index.name, index.uuid
            )));
        }
        Arc::make_mut(&mut self.new_fragments).insert(id);
        Ok(id)
    }

    /// Set `count` fragment ids aside for a later writer, on top of any this
    /// operation already reserved. A later writer names them as
    /// [`Ref::Committed`].
    pub(super) fn reserve_fragment_ids(&mut self, count: u32) -> Result<()> {
        self.reserved_fragment_ids = self
            .reserved_fragment_ids
            .checked_add(u64::from(count))
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "reserving {count} more fragment ids on top of {} already reserved in this \
                     operation overflows the fragment id counter",
                    self.reserved_fragment_ids
                ))
            })?;
        Ok(())
    }

    /// Set `count` row ids aside for a later writer, on top of any this
    /// operation already reserved.
    pub(super) fn reserve_row_ids(&mut self, count: u64) -> Result<()> {
        self.reserved_row_ids = self.reserved_row_ids.checked_add(count).ok_or_else(|| {
            Error::invalid_input(format!(
                "reserving {count} more row ids on top of {} already reserved in this operation \
                 overflows the row id counter",
                self.reserved_row_ids
            ))
        })?;
        Ok(())
    }

    pub(super) fn mint_field(&mut self, token: u32) -> Result<i32> {
        if self.field_tokens.contains_key(&token) {
            return Err(duplicate_token_err("field", token));
        }
        let id = self
            .field_high_water
            .checked_add(1)
            .ok_or_else(|| exhausted_err("field id", i32::MAX))?;
        self.field_high_water = id;
        Arc::make_mut(&mut self.field_tokens).insert(token, id);
        Ok(id)
    }

    pub(super) fn mint_base(&mut self, token: u32) -> Result<u32> {
        if self.base_tokens.contains_key(&token) {
            return Err(duplicate_token_err("base", token));
        }
        let id = self
            .base_high_water
            .checked_add(1)
            .ok_or_else(|| exhausted_err("base id", u32::MAX))?;
        self.base_high_water = id;
        Arc::make_mut(&mut self.base_tokens).insert(token, id);
        Ok(id)
    }

    /// Resolve a committed or operation-local fragment reference against this state.
    pub fn resolve_fragment(&self, reference: Ref) -> Result<u64> {
        match reference {
            Ref::Committed(id) => {
                if self.read_marks.fragment.is_none_or(|mark| id > mark) {
                    return Err(beyond_read_mark_err(
                        "fragment",
                        id,
                        display_mark(self.read_marks.fragment),
                    ));
                }
                Ok(id)
            }
            Ref::Local(token) => self
                .fragment_tokens
                .get(&token)
                .copied()
                .ok_or_else(|| unbound_token_err("fragment", token)),
        }
    }

    /// The id of a base path: a committed one the version being committed on
    /// top of has, or one an earlier action in this operation minted.
    pub(super) fn resolve_base(&self, reference: Ref) -> Result<u32> {
        match reference {
            Ref::Committed(id) if id > u64::from(self.read_marks.base) => Err(
                beyond_read_mark_err("base", id, self.read_marks.base.to_string()),
            ),
            Ref::Committed(id) => u32::try_from(id)
                .ok()
                .filter(|id| self.existing_base_paths.contains_key(id))
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "base id {id} in an action names no base path of the dataset"
                    ))
                }),
            Ref::Local(token) => self
                .base_tokens
                .get(&token)
                .copied()
                .ok_or_else(|| unbound_token_err("base", token)),
        }
    }

    /// The reserved negative ids are not drawn from the field counter, so only
    /// a schema field id is held to the read version's mark.
    pub(super) fn resolve_field(&self, reference: FieldRef) -> Result<FieldId> {
        match reference {
            FieldRef::Committed(id) if id > self.read_marks.field => Err(beyond_read_mark_err(
                "field",
                id,
                self.read_marks.field.to_string(),
            )),
            FieldRef::Committed(id) => Ok(FieldId(id)),
            FieldRef::Local(token) => self
                .field_tokens
                .get(&token)
                .copied()
                .map(FieldId)
                .ok_or_else(|| unbound_token_err("field", token)),
        }
    }

    /// Resolve a reference that has to name a schema field. Only
    /// [`AddDataFile`](super::AddDataFile) binds the reserved ids, which name a
    /// column of one data file rather than a field.
    pub(super) fn resolve_schema_field(&self, reference: FieldRef, action: &str) -> Result<i32> {
        let id = self.resolve_field(reference)?;
        if !id.is_schema() {
            return Err(Error::invalid_input(format!(
                "{action} names reserved field id {id}, which is not a schema field; only \
                 AddDataFile binds a reserved id, for a row lineage column"
            )));
        }
        Ok(id.get())
    }

    /// Whether this operation added the fragment, as opposed to finding it in
    /// the version being committed on top of.
    pub(super) fn is_new_fragment(&self, fragment_id: u64) -> bool {
        self.new_fragments.contains(&fragment_id)
    }

    pub(super) fn note_rearranged(&mut self, fragment_id: u64) {
        Arc::make_mut(&mut self.rearranged_fragments).insert(fragment_id);
    }

    /// Whether `fragment_id` is one this operation added with
    /// `data_change = false`.
    pub(super) fn is_rearranged(&self, fragment_id: u64) -> bool {
        self.rearranged_fragments.contains(&fragment_id)
    }

    /// Refuse a write that changes the values of a fragment this operation
    /// added with `data_change = false`.
    ///
    /// Such a fragment's rows were already in the table: they keep their row
    /// versions, and the footprint records no inserted rows and no write to a
    /// committed fragment. A value change in the same operation would hide
    /// behind all three -- an incremental reader would miss the changed rows,
    /// a concurrent key assertion would not see a changed key, and an index
    /// carried onto the fragment through the fragment reuse history would
    /// answer from the old values.
    pub(super) fn reject_data_change_to_rearranged(
        &self,
        fragment_id: u64,
        action: &str,
    ) -> Result<()> {
        if !self.is_rearranged(fragment_id) {
            return Ok(());
        }
        Err(Error::invalid_input(format!(
            "{action} changes the values of fragment {fragment_id}, which this operation adds \
             with data_change = false; a fragment that only moves rows cannot also change \
             them, so add it with data_change = true, or change the rows in a later operation"
        )))
    }

    /// Record that a [`RefreshRowVersionMetadata`](super::RefreshRowVersionMetadata)
    /// restamped some or all of this fragment's rows.
    pub(super) fn note_restamp(&mut self, fragment_id: u64) {
        Arc::make_mut(&mut self.restamped_fragments).insert(fragment_id);
    }

    /// Record that these fields' data in this fragment no longer matches what
    /// an index built over them recorded.
    ///
    /// Prunes the segments present now rather than at assembly: a segment an
    /// later action in this operation adds was built over the new data, so the
    /// rebind does not invalidate it.
    pub(super) fn rebind_fields(
        &mut self,
        fragment_id: u64,
        fields: impl IntoIterator<Item = i32>,
    ) -> Result<()> {
        let fields: Vec<i32> = fields.into_iter().collect();
        Arc::make_mut(&mut self.rebound_fields).push((
            fragment_id,
            fields
                .iter()
                .filter_map(|field| u32::try_from(*field).ok())
                .collect(),
        ));
        prune_rebound_fields_from_indices(
            Arc::make_mut(&mut self.indices).as_mut_slice(),
            fragment_id,
            &fields,
            &self.current_manifest.schema,
            &self.schema,
        )
    }

    /// Record that a data-changing action changed values stored in this
    /// fragment. A fragment this operation adds is stamped whole at assembly.
    pub(super) fn note_value_change(&mut self, fragment_id: u64) {
        if !self.is_new_fragment(fragment_id) {
            Arc::make_mut(&mut self.value_changed_fragments).insert(fragment_id);
        }
    }

    /// Record that a [`TombstoneFieldData`](super::TombstoneFieldData) freed
    /// `field_id` in `fragment_id`.
    pub(super) fn note_tombstone(&mut self, fragment_id: u64, field_id: i32, data_change: bool) {
        Arc::make_mut(&mut self.tombstoned_fields).insert(
            (fragment_id, field_id),
            TombstonedField {
                data_change,
                rebound: false,
            },
        );
    }

    /// Record that an [`AddDataFile`](super::AddDataFile) bound `field_id` in
    /// `fragment_id`. Re-binding a field this operation tombstoned has to agree
    /// with the tombstone on whether the values change: a re-encode that is
    /// half data change would keep index coverage of values that moved, or
    /// withdraw it from values that did not.
    pub(super) fn note_rebind(
        &mut self,
        fragment_id: u64,
        field_id: i32,
        data_change: bool,
    ) -> Result<()> {
        let Some(tombstoned) =
            Arc::make_mut(&mut self.tombstoned_fields).get_mut(&(fragment_id, field_id))
        else {
            return Ok(());
        };
        if tombstoned.data_change != data_change {
            return Err(Error::invalid_input(format!(
                "AddDataFile for fragment {fragment_id} re-binds field {field_id} with \
                 data_change {data_change}, but the TombstoneFieldData that freed it has \
                 data_change {}; a re-encode's tombstone and re-add must agree",
                tombstoned.data_change
            )));
        }
        tombstoned.rebound = true;
        Ok(())
    }

    /// A tombstone with `data_change` false promises the values come back
    /// unchanged, which is what lets it keep the fields' overlays and index
    /// coverage. Without the re-add the field would read as null, a change
    /// those would still claim to describe.
    fn check_reencodes_are_rebound(&self) -> Result<()> {
        let mut unbound = self
            .tombstoned_fields
            .iter()
            .filter(|(_, tombstoned)| !tombstoned.data_change && !tombstoned.rebound)
            .map(|(key, _)| *key)
            .collect::<Vec<_>>();
        if unbound.is_empty() {
            return Ok(());
        }
        unbound.sort_unstable();
        let unbound = unbound
            .iter()
            .map(|(fragment, field)| format!("field {field} in fragment {fragment}"))
            .collect::<Vec<_>>()
            .join(", ");
        Err(Error::invalid_input(format!(
            "TombstoneFieldData with data_change false leaves {unbound} without a data file; \
             a re-encode must re-bind every field it tombstones with an AddDataFile that is \
             also data_change false, or tombstone it with data_change true"
        )))
    }

    /// On a stable row id table, `get_updated_rows()` reads which rows changed
    /// from `last_updated_at_version_meta`. Apply cannot infer which rows of a
    /// committed fragment a data file or tombstone changed (a column rewrite
    /// writes the whole fragment but may change only some rows), so the
    /// writer has to say with a
    /// [`RefreshRowVersionMetadata`](super::RefreshRowVersionMetadata), or the
    /// change would publish with stale row versions. Checked once every action
    /// has been seen, because the refresh may come before or after the change.
    ///
    /// Any selection of the fragment's rows satisfies it, down to one row:
    /// which rows changed is exactly what the files cannot say, so the
    /// writer's selection is taken as given.
    fn check_value_changes_are_restamped(&self) -> Result<()> {
        if !self.uses_stable_row_ids() {
            return Ok(());
        }
        let unrestamped = self
            .value_changed_fragments
            .difference(&self.restamped_fragments)
            .map(|fragment| format!("fragment {fragment}"))
            .collect::<Vec<_>>();
        if unrestamped.is_empty() {
            return Ok(());
        }
        let unrestamped = unrestamped.join(", ");
        Err(Error::invalid_input(format!(
            "the table has stable row ids, and the operation changes values in place in \
             committed {unrestamped}, which would leave last_updated_at_version_meta stale; add \
             a RefreshRowVersionMetadata covering the changed rows of {unrestamped}"
        )))
    }

    /// Every row lineage sequence a fragment added here spills has to be bound
    /// by exactly one of its data files, or the sequence has no column to be
    /// read from. Checked once the operation has run, because the
    /// [`AddDataFile`](super::AddDataFile) carrying it follows the
    /// [`AddFragment`](super::AddFragment) that declares it.
    fn check_spilled_lineage_is_bound(&self) -> Result<()> {
        for fragment in self.added_fragments() {
            for field_id in fragment.spilled_row_lineage_field_ids() {
                let carriers = fragment
                    .files
                    .iter()
                    .filter(|file| file.fields.contains(&field_id))
                    .count();
                if carriers == 0 {
                    return Err(Error::invalid_input(format!(
                        "fragment {} spills row lineage field {field_id} but no AddDataFile \
                         binds it",
                        fragment.id
                    )));
                }
                if carriers > 1 {
                    return Err(Error::invalid_input(format!(
                        "fragment {} binds row lineage field {field_id} in more than one data \
                         file",
                        fragment.id
                    )));
                }
            }
        }
        Ok(())
    }

    /// Stamp row ids and version metadata onto the fragments this operation
    /// minted, matching what an Append does for its new fragments.
    fn assign_row_ids_to_new_fragments(
        &mut self,
        next_row_id: &mut Option<u64>,
        new_version: u64,
    ) -> Result<()> {
        let Some(next_row_id) = next_row_id.as_mut() else {
            return Ok(());
        };
        let mut updated = self.added_fragments_numbered(next_row_id)?;
        for fragment in updated.iter_mut() {
            let version_meta = build_version_meta(fragment, new_version);
            if fragment.last_updated_at_version_meta.is_none() {
                fragment.last_updated_at_version_meta = version_meta.clone();
            }
            if fragment.created_at_version_meta.is_none() {
                fragment.created_at_version_meta = version_meta;
            }
        }
        for fragment in updated {
            self.fragments.insert(fragment);
        }
        Ok(())
    }

    fn added_fragments_numbered(&self, next_row_id: &mut u64) -> Result<Vec<Fragment>> {
        self.validate_supplied_row_ids()?;
        let mut numbered = Vec::new();
        for slot in self.added_fragment_order.iter() {
            match slot {
                AddedFragment::Removed { fresh_row_ids } => {
                    *next_row_id = advance_row_id_counter(*next_row_id, *fresh_row_ids)?;
                }
                AddedFragment::Present(id) => {
                    let Some(mut fragment) = self.fragments.get_unmasked(*id).cloned() else {
                        continue;
                    };
                    Transaction::assign_row_ids(next_row_id, std::slice::from_mut(&mut fragment))?;
                    numbered.push(fragment);
                }
            }
        }
        Ok(numbered)
    }

    /// Reject row ids a new fragment arrived with that no commit reserved.
    ///
    /// [`Transaction::assign_row_ids`] honors a sequence a fragment carries and
    /// leaves the counter where it found it, which is right for a compaction:
    /// its rows keep ids that are already below the watermark. A writer
    /// supplying *fresh* ids must have set them aside with
    /// [`ReserveRowIds`](super::ReserveRowIds), which moved the watermark past
    /// them. An id at or above the watermark was never reserved, and the next
    /// append would hand it out a second time.
    ///
    /// The watermark is the read version's, not the one applied onto: a
    /// concurrent append mints ids above the read watermark without claiming
    /// them in its footprint, so an id the writer guessed there may already
    /// name one of that append's rows.
    ///
    /// A row id names one row, so an id carried twice -- by two fragments of
    /// this operation, or twice within one fragment -- is rejected too.
    fn validate_supplied_row_ids(&self) -> Result<()> {
        let reserved_below = self.read_marks.next_row_id.unwrap_or(0);
        let mut supplied: Vec<(u64, RoaringTreemap)> = Vec::new();
        for fragment in self.added_fragments() {
            // A spilled sequence lives in a file, which this synchronous path
            // has no way to read; it goes unchecked.
            let Some(RowIdMeta::Inline(data)) = &fragment.row_id_meta else {
                continue;
            };
            let sequence = read_row_ids(data)?;
            let row_ids = sequence.to_treemap();
            // The treemap collapses repeats, so only a shorter treemap means a
            // repeat worth paying a second pass to name.
            let mut seen = RoaringTreemap::new();
            if row_ids.len() != sequence.len()
                && let Some(duplicate) = sequence.iter().find(|row_id| !seen.insert(*row_id))
            {
                return Err(Error::invalid_input(format!(
                    "fragment {} carries row id {duplicate} more than once; a row id names one \
                     row",
                    fragment.id
                )));
            }
            for (other, other_row_ids) in &supplied {
                if let Some(shared) = (&row_ids & other_row_ids).min() {
                    return Err(Error::invalid_input(format!(
                        "fragment {} carries row id {shared}, which fragment {other} also added in \
                         this operation carries; a row id names one row",
                        fragment.id
                    )));
                }
            }
            let Some(range) = sequence.row_id_range() else {
                continue;
            };
            supplied.push((fragment.id, row_ids));
            if *range.end() >= reserved_below {
                return Err(Error::invalid_input(format!(
                    "fragment {} is added carrying row id {}, which no commit has reserved as of \
                     the read version: only ids below {reserved_below} are reserved. Reserve a \
                     range with ReserveRowIds and read the committed manifest's next_row_id back \
                     before supplying row ids.",
                    fragment.id,
                    range.end()
                )));
            }
        }
        Ok(())
    }

    /// Take this operation's fragment id reservation off the counter, after its
    /// own fragments have been numbered, so the reserved range is always the
    /// top `count` ids of the committed high-water mark.
    fn take_fragment_id_reservation(&mut self) -> Result<()> {
        self.fragment_high_water = self.fragment_high_water_after_reservation()?;
        Ok(())
    }

    fn fragment_high_water_after_reservation(&self) -> Result<Option<u64>> {
        if self.reserved_fragment_ids == 0 {
            return Ok(self.fragment_high_water);
        }
        let high_water = match self.fragment_high_water {
            None => Some(self.reserved_fragment_ids - 1),
            Some(mark) => mark.checked_add(self.reserved_fragment_ids),
        }
        .and_then(|high_water| u32::try_from(high_water).ok())
        .ok_or_else(|| exhausted_err("fragment id", MAX_FRAGMENT_ID))?;
        Ok(Some(u64::from(high_water)))
    }

    /// Take this operation's row id reservation off the counter, after its own
    /// fragments have been numbered, so the reserved range is always the top
    /// `count` ids of the committed watermark.
    fn take_row_id_reservation(&self, next_row_id: &mut Option<u64>) -> Result<()> {
        if self.reserved_row_ids == 0 {
            return Ok(());
        }
        let Some(next_row_id) = next_row_id.as_mut() else {
            return Err(Error::invalid_input(
                "ReserveRowIds needs a row id counter to reserve from, and this dataset does \
                 not use stable row ids",
            ));
        };
        let current = *next_row_id;
        *next_row_id = current.checked_add(self.reserved_row_ids).ok_or_else(|| {
            Error::invalid_input(format!(
                "reserving {} row ids overflows the row id counter at {current}",
                self.reserved_row_ids
            ))
        })?;
        Ok(())
    }

    fn check_row_id_counter(&self) -> Result<()> {
        let mut next_row_id = self
            .uses_stable_row_ids()
            .then_some(self.current_manifest.next_row_id);
        if let Some(next_row_id) = next_row_id.as_mut() {
            if self.row_ids_need_dry_run {
                self.added_fragments_numbered(next_row_id)?;
            } else {
                *next_row_id = advance_row_id_counter(*next_row_id, self.fresh_row_ids_added)?;
            }
        }
        self.take_row_id_reservation(&mut next_row_id)
    }
}

/// Reject a fragment that stores no data for a non-nullable field: it would
/// read back as nulls, which the schema forbids.
///
/// Like the legacy path, only fragments whose data files this operation set
/// are checked, so a fragment committed before the check existed does not
/// block an unrelated commit. Making a field required is a claim about every
/// row, so then every fragment is checked.
fn check_required_fields_are_stored(
    current_manifest: &Manifest,
    schema: &Schema,
    fragments: &[Fragment],
) -> Result<()> {
    if !required_field_ids(schema).is_subset(&required_field_ids(&current_manifest.schema)) {
        return schema_fragments_valid(Some(current_manifest), schema, fragments);
    }
    let files_before: HashMap<u64, &[DataFile]> = current_manifest
        .fragments
        .iter()
        .map(|fragment| (fragment.id, fragment.files.as_slice()))
        .collect();
    let written: Vec<Fragment> = fragments
        .iter()
        .filter(|fragment| files_before.get(&fragment.id) != Some(&fragment.files.as_slice()))
        .cloned()
        .collect();
    schema_fragments_valid(Some(current_manifest), schema, &written)
}

/// The ids of the fields that can never be null: non-nullable, under
/// non-nullable ancestors only.
fn required_field_ids(schema: &Schema) -> HashSet<i32> {
    fn collect(fields: &[Field], ids: &mut HashSet<i32>) {
        for field in fields.iter().filter(|field| !field.nullable) {
            ids.insert(field.id);
            collect(&field.children, ids);
        }
    }
    let mut ids = HashSet::new();
    collect(&schema.fields, &mut ids);
    ids
}

/// Drop the fragments whose data no longer matches what an index recorded.
///
/// An index built over a field describes the values that were in that field
/// when it was built. Rebinding the field's data in a fragment invalidates the
/// index for that fragment only, so the fragment leaves the bitmap rather than
/// the whole index being discarded.
///
/// Rebinding a field also invalidates an index on any field it contains or is
/// nested in: a packed struct is stored under the parent's id while an index may
/// be keyed on a child. A field counts if its ancestry relates it to the index
/// in either `base_schema`, the schema of the version being committed on top
/// of, or `working_schema`, the one being built: a field this operation drops
/// or moves has already left the working schema, and a child it adds is not
/// yet in the base one.
///
/// A user index with no bitmap -- one built before indices recorded their
/// coverage -- has no fragment to prune, so a rewrite under it is refused
/// rather than left to serve stale rows. System indices carry no bitmap by
/// design and are keyed on no column, so a rewrite never affects them.
fn prune_rebound_fields_from_indices(
    indices: &mut [IndexMetadata],
    fragment_id: u64,
    fields: &[i32],
    base_schema: &Schema,
    working_schema: &Schema,
) -> Result<()> {
    let Ok(fragment_id) = u32::try_from(fragment_id) else {
        return Ok(());
    };
    for index in indices.iter_mut() {
        let affected_field = fields.iter().copied().find(|field| {
            field_affects_index(*field, &index.fields, base_schema)
                || field_affects_index(*field, &index.fields, working_schema)
        });
        let Some(bitmap) = index.fragment_bitmap.as_mut() else {
            if !is_system_index(index)
                && let Some(field) = affected_field
            {
                return Err(Error::invalid_input(format!(
                    "index '{}' has unknown coverage; migrate or rebuild it before rewriting \
                     field {field}",
                    index.name
                )));
            }
            continue;
        };
        if affected_field.is_some() {
            bitmap.remove(fragment_id);
        }
    }
    Ok(())
}

/// [`Schema::validate`], preceded by the shape [`Field::data_type`] assumes: a
/// list or map has exactly one child, a struct at least one (dropping a
/// struct's last child drops the struct on the legacy path), and any other
/// type none.
fn check_schema(schema: &Schema) -> Result<()> {
    for field in schema.fields_pre_order() {
        let logical_type = &field.logical_type;
        let (fits, expected) = if logical_type.is_list()
            || logical_type.is_large_list()
            || logical_type.is_fixed_size_list_struct()
            || logical_type.is_map()
        {
            (field.children.len() == 1, "exactly one child")
        } else if logical_type.is_struct() {
            (!field.children.is_empty(), "at least one child")
        } else {
            (field.children.is_empty(), "no children")
        };
        if !fits {
            return Err(Error::invalid_input(format!(
                "field '{}' (id {}) of type {} has {} children, but its type takes {expected}",
                field.name,
                field.id,
                logical_type,
                field.children.len()
            )));
        }
    }
    schema.validate()
}

fn auto_cleanup_settings(config: &HashMap<String, String>) -> BTreeMap<&str, &str> {
    config
        .iter()
        .filter(|(key, _)| key.starts_with("lance.auto_cleanup."))
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect()
}

/// Refuse an index action naming the fragment reuse history. The history is
/// not an index on the action path -- its own actions append to and trim it,
/// merged onto the current history at every attempt -- so an index action
/// naming it could only splice in a snapshot or drop records a concurrent
/// writer appended.
pub(super) fn reject_frag_reuse_history(action: &str, name: &str) -> Result<()> {
    if name == FRAG_REUSE_INDEX_NAME {
        return Err(Error::invalid_input(format!(
            "{action} names '{FRAG_REUSE_INDEX_NAME}', the fragment reuse history, which is not \
             an index an index action may change; use AppendFragReuseTransition or \
             TrimFragReuse"
        )));
    }
    Ok(())
}

fn beyond_read_mark_err(space: &str, id: impl std::fmt::Display, mark: String) -> Error {
    Error::invalid_input(format!(
        "an action names {space} {id} by a Committed ref, but the read version's high-water mark \
         for {space} ids is {mark}: the id did not exist when the operation read the table, and \
         a concurrent commit may since have minted it. Name a {space} this operation mints by \
         its Local ref."
    ))
}

fn display_mark(mark: Option<u64>) -> String {
    mark.map_or_else(
        || "unset (no id handed out)".to_string(),
        |mark| mark.to_string(),
    )
}

/// Check that the segments left under each touched index name still form one
/// logical index.
///
/// The query path unions a logical index's segments, so they must agree on
/// what the index is: the same keyed and carried fields, resolved to ids (a
/// `Local` token means a different field in each operation that mints one), and
/// the same index type. And they must not describe a fragment twice, which
/// would double-count its rows. The check runs on the final set rather than the
/// actions, so a segment added by an earlier commit is compared too.
///
/// The rest of the index details is not compared, unlike the conflict check.
/// A rejection here is permanent, and segments of one index legitimately carry
/// different details: training knobs chosen per build, map fields with no fixed
/// encoding order, and datasets that already hold such segments. Whether a
/// vector query can mix segments is checked by `lance` when it opens them.
///
/// Only names this operation touched are checked, so an operation is not blamed
/// for an inconsistency it did not introduce.
fn validate_index_segment_sets(
    indices: &[IndexMetadata],
    touched: &BTreeSet<String>,
) -> Result<()> {
    for name in touched {
        let mut segments = indices.iter().filter(|index| &index.name == name);
        let Some(first) = segments.next() else {
            continue;
        };
        let type_url = |index: &IndexMetadata| {
            index
                .index_details
                .as_ref()
                .map(|details| details.type_url.clone())
        };
        let mut covered = first.fragment_bitmap.clone().unwrap_or_default();
        for segment in segments {
            if segment.fields != first.fields || segment.covering_fields != first.covering_fields {
                return Err(Error::invalid_input(format!(
                    "index '{name}' has segment {} over fields {:?} (carried {:?}) beside segment {} \
                     over fields {:?} (carried {:?}); the segments of one index must agree",
                    segment.uuid,
                    segment.fields,
                    segment.covering_fields,
                    first.uuid,
                    first.fields,
                    first.covering_fields
                )));
            }
            if type_url(segment) != type_url(first) {
                return Err(Error::invalid_input(format!(
                    "index '{name}' has segment {} of type {:?} beside segment {} of type {:?}; \
                     the segments of one index must be the same type",
                    segment.uuid,
                    type_url(segment),
                    first.uuid,
                    type_url(first)
                )));
            }
            if segment.results_are_row_addrs() != first.results_are_row_addrs() {
                return Err(Error::invalid_input(format!(
                    "Index '{name}' mixes row-ID and row-address segments {} (version {}) \
                     and {} (version {}); rebuild every segment in one commit",
                    first.uuid, first.index_version, segment.uuid, segment.index_version,
                )));
            }
            let Some(bitmap) = &segment.fragment_bitmap else {
                continue;
            };
            if let Some(fragment) = (&covered & bitmap).min() {
                return Err(Error::invalid_input(format!(
                    "index '{name}' has segment {} covering fragment {fragment}, which another \
                     segment of the index already covers; its rows would be counted twice",
                    segment.uuid
                )));
            }
            covered |= bitmap;
        }
    }
    Ok(())
}

fn unbound_token_err(space: &str, token: u32) -> Error {
    Error::invalid_input(format!(
        "an action references local {space} token {token}, which no earlier action in this \
         operation minted"
    ))
}

fn exhausted_err(counter: &str, max: impl std::fmt::Display) -> Error {
    Error::invalid_input(format!(
        "the {counter} counter is exhausted: this operation needs an id past the maximum, {max}"
    ))
}

fn duplicate_token_err(space: &str, token: u32) -> Error {
    Error::invalid_input(format!(
        "local {space} token {token} is minted more than once in this operation; tokens must be \
         distinct within an operation"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_flags::{
        FLAG_MIXED_DATA_FILE_VERSIONS, FLAG_NON_REUSABLE_FIELD_IDS, FLAG_OBJECT_METADATA,
        MANAGED_BLOBS_META_KEY,
    };
    use crate::format::{DataFile, ROW_LAST_UPDATED_AT_VERSION_FIELD_ID};
    use crate::rowids::version::{
        LoadedRowLineage, RowDatasetVersionMeta, RowDatasetVersionSequence,
    };
    use crate::rowids::write_row_ids;
    use crate::system_index::mem_wal::{
        CompactedSsTable, IndexCatchupProgress, MemWalIndexDetails, load_mem_wal_index_details,
        new_mem_wal_index_meta,
    };
    use crate::transaction::action::test_support::{
        added_field, any_deletion_file, apply, apply_after_reading, apply_with_indices,
        backed_manifest, frag_reuse, last_updated_versions, manifest_with_base, mint_base,
        one_column_file, refresh, refresh_rows, replace_field_data,
    };
    use crate::transaction::action::{
        Action, AddBase, AddDataFile, AddField, AddFragment, AddFragmentParts, AddIndexSegment,
        AddIndexSegmentParts, AdjustIndexCoverage, AdjustIndexCoverageParts, AlterField,
        AssertCellsUnchanged, CompositeOperation, ConfigUpdate, DropField, DropIndex,
        RemoveFragment, RemoveIndexSegment, ReserveFragmentIds, ReserveRowIds, ResetTable,
        RowSelection, SelectedRows, SetDeletionFile, TombstoneFieldData, UserStep,
    };
    use crate::transaction::test_support::{
        default_build_config, make_stable_row_id_manifest, sample_index_metadata, sample_manifest,
    };
    use crate::transaction::{DataReplacementGroup, Operation, UpdateMap};
    use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
    use lance_core::{ROW_ADDR, ROW_ID, ROW_ID_FIELD_ID};
    use lance_file::version::ConcreteFileVersion;
    use roaring::RoaringBitmap;
    use rstest::rstest;
    use std::sync::Arc;
    use uuid::Uuid;

    /// MemWAL coverage invalidation is derived while the manifest is built, so
    /// it has to run on the action path as well as the legacy one. A commit
    /// that narrows an index below the version it read must lose that index's
    /// catch-up entry, or the WAL pod goes on retiring SSTables against a
    /// position the index no longer holds.
    #[rstest]
    #[case::legacy(false)]
    #[case::actions(true)]
    fn test_narrowing_an_index_invalidates_mem_wal_catchup_on_both_paths(
        #[case] via_actions: bool,
    ) {
        let mut manifest = backed_manifest();
        let mut second = manifest.fragments[0].clone();
        second.id = 1;
        second.files[0].path = "data/1.lance".into();
        manifest.fragments = Arc::new(vec![manifest.fragments[0].clone(), second]);

        let shard = Uuid::from_u128(7);
        let existing = vec![
            IndexMetadata {
                fragment_bitmap: Some(RoaringBitmap::from_iter([0u32, 1])),
                ..sample_index_metadata("idx")
            },
            new_mem_wal_index_meta(
                manifest.version,
                MemWalIndexDetails {
                    compacted_sstables: vec![CompactedSsTable::new(shard, 5)],
                    index_catchup: vec![IndexCatchupProgress::new(
                        "idx".to_string(),
                        vec![CompactedSsTable::new(shard, 5)],
                    )],
                    ..Default::default()
                },
            )
            .unwrap(),
        ];

        // Replacing field 0's data in fragment 0 prunes that fragment from the
        // index, which then no longer spans the version this commit read. The
        // two forms below are the same change said twice.
        let replacement = DataFile::new(
            "data/0-new.lance",
            vec![0],
            vec![0],
            ConcreteFileVersion::V2_0,
            None,
            None,
        );
        let operation = if via_actions {
            Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
                "replace field 0",
                vec![
                    Action::TombstoneFieldData(TombstoneFieldData::new(
                        Ref::Committed(0),
                        vec![FieldRef::Committed(0)],
                    )),
                    Action::AddDataFile(
                        AddDataFile::try_new_unchecked(
                            Ref::Committed(0),
                            DataFile {
                                fields: Arc::from([]),
                                ..replacement
                            },
                            vec![FieldRef::Committed(0)],
                            None,
                            true,
                        )
                        .unwrap(),
                    ),
                ],
            )]))
        } else {
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, replacement)],
                data_change: true,
            }
        };

        let (_, indices) = Transaction::new(manifest.version, operation, None)
            .build_manifest_with_read_version(
                Some(&manifest),
                existing.clone(),
                "tx.txn",
                &default_build_config(),
                Some(ReadVersionState {
                    manifest: &manifest,
                    indices: &existing,
                }),
            )
            .unwrap();

        let narrowed = indices.iter().find(|index| index.name == "idx").unwrap();
        assert_eq!(
            narrowed.fragment_bitmap,
            Some(RoaringBitmap::from_iter([1u32])),
            "the replacement should have pruned fragment 0 from the index"
        );

        let details = load_mem_wal_index_details(
            indices
                .iter()
                .find(|index| index.name == MEM_WAL_INDEX_NAME)
                .expect("the MemWAL index should still be there")
                .clone(),
        )
        .unwrap();
        assert!(
            details.index_catchup.is_empty(),
            "an index that no longer covers the read version must lose its catch-up entry, \
             but it kept {:?}",
            details.index_catchup
        );
    }

    #[test]
    fn test_an_action_set_relocates_onto_a_newer_version() {
        let actions = vec![
            Action::AddField(AddField::new(0, added_field("added"))),
            Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
                id: Ref::Local(0),
                physical_rows: 10,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            })),
            Action::AddDataFile(
                AddDataFile::try_new_unchecked(
                    Ref::Local(0),
                    one_column_file("data/new.lance"),
                    vec![FieldRef::Local(0)],
                    None,
                    true,
                )
                .unwrap(),
            ),
        ];

        // Replaying the same actions against the manifest the first run
        // produced re-resolves both local tokens against the newer counters.
        // Only the field's name differs, since the first run took "added".
        let first = apply(&backed_manifest(), actions.clone()).unwrap();
        let mut replayed = actions;
        let Action::AddField(add_field) = &mut replayed[0] else {
            unreachable!("the first action is the AddField built above");
        };
        let mut def = add_field.def().clone();
        def.name = "added_again".into();
        *add_field = AddField::new(add_field.local(), def);
        let second = apply(&first, replayed).unwrap();

        assert_eq!(
            second.fragments.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            second
                .schema
                .fields_pre_order()
                .map(|field| field.id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        // The second run's data file points at the field the second run minted,
        // not at the one the first run did.
        let relocated = second.fragments.iter().find(|f| f.id == 2).unwrap();
        assert_eq!(relocated.files[0].fields.as_ref(), &[2]);
    }

    #[test]
    fn test_action_set_cannot_create_a_dataset() {
        let transaction = Transaction::new(
            0,
            Operation::CompositeOperation(CompositeOperation::new(vec![])),
            None,
        );
        let error = transaction
            .build_manifest(None, Vec::new(), "tx.txn", &default_build_config())
            .unwrap_err();
        assert!(
            error.to_string().contains("cannot create one"),
            "unexpected error: {error}"
        );
    }

    /// An index is keyed to one field and the data is stored under another: a
    /// packed struct is stored under the parent's id while an index may be
    /// built on a child, and the reverse for an unpacked struct. Either way the
    /// rewrite changes what the index recorded, so the exact ids differing is
    /// not a reason to keep the fragment covered.
    #[rstest]
    #[case::packed_parent_rewritten_under_a_child_index(0, 1)]
    #[case::child_rewritten_under_a_parent_index(1, 0)]
    fn test_rewriting_a_field_prunes_indices_on_its_ancestor_or_descendant(
        #[case] rewritten: i32,
        #[case] indexed: i32,
    ) {
        let arrow = ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(Fields::from(vec![ArrowField::new(
                "x",
                DataType::Int32,
                true,
            )])),
            true,
        )]);
        let mut manifest = backed_manifest();
        manifest.schema = lance_core::datatypes::Schema::try_from(&arrow).unwrap();
        let mut fragment = manifest.fragments[0].clone();
        fragment.files[0].fields = vec![rewritten].into();
        manifest.fragments = Arc::new(vec![fragment]);

        let (_, indices) = apply_with_indices(
            &manifest,
            vec![
                Action::TombstoneFieldData(TombstoneFieldData::new(
                    Ref::Committed(0),
                    vec![FieldRef::Committed(rewritten)],
                )),
                Action::AddDataFile(
                    AddDataFile::try_new_unchecked(
                        Ref::Committed(0),
                        DataFile::new(
                            "data/0-new.lance",
                            Vec::<i32>::new(),
                            vec![0],
                            ConcreteFileVersion::V2_0,
                            None,
                            None,
                        ),
                        vec![FieldRef::Committed(rewritten)],
                        None,
                        true,
                    )
                    .unwrap(),
                ),
            ],
            vec![IndexMetadata {
                fields: vec![indexed],
                ..sample_index_metadata("idx")
            }],
        )
        .unwrap();

        assert!(indices[0].fragment_bitmap.as_ref().unwrap().is_empty());
    }

    /// A child the operation itself adds is not in the schema it builds on, so
    /// its ancestry has to come from the schema being built for a rewrite of
    /// its parent to reach an index on it.
    #[test]
    fn test_rewriting_a_struct_prunes_an_index_on_a_child_added_in_the_same_operation() {
        let mut manifest = backed_manifest();
        manifest.schema = Schema::try_from(&ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(Fields::from(vec![ArrowField::new(
                "x",
                DataType::Int32,
                true,
            )])),
            true,
        )]))
        .unwrap();

        let mut actions = vec![
            Action::AddField(
                AddField::new(0, added_field("z")).with_parent(FieldRef::Committed(0)),
            ),
            Action::AddIndexSegment(
                AddIndexSegment::try_new_unchecked(AddIndexSegmentParts {
                    uuid: Uuid::from_u128(1),
                    name: "idx".into(),
                    fields: vec![FieldRef::Local(0)],
                    covering_fields: Vec::new(),
                    index_details: None,
                    index_version: 1,
                    covered_fragments: Some(vec![Ref::Committed(0)]),
                    files: Vec::new(),
                    base: None,
                    created_at: None,
                    dataset_version: None,
                    data_change: false,
                })
                .unwrap(),
            ),
        ];
        actions.extend(replace_field_data(0, 0, true));

        let (_, indices) = apply_with_indices(&manifest, actions, Vec::new()).unwrap();

        let index = indices.iter().find(|index| index.name == "idx").unwrap();
        assert!(
            !index.fragment_bitmap.as_ref().unwrap().contains(0),
            "rewriting s in fragment 0 should prune it from the index on s.z, got {:?}",
            index.fragment_bitmap
        );
    }

    /// An index from before fragment bitmaps has no record of what it covers,
    /// so there is no fragment to prune out of it when data under it is
    /// rewritten. Keeping it would serve stale results, and dropping it
    /// silently would lose an index the user built, so the rewrite is refused.
    #[rstest]
    #[case::indexed_field(0, true)]
    #[case::other_field(1, false)]
    fn test_rewriting_under_an_index_of_unknown_coverage_is_refused(
        #[case] indexed: i32,
        #[case] rejected: bool,
    ) {
        let index = IndexMetadata {
            fields: vec![indexed],
            fragment_bitmap: None,
            ..sample_index_metadata("idx")
        };

        let result = apply_with_indices(
            &backed_manifest(),
            vec![
                Action::TombstoneFieldData(TombstoneFieldData::new(
                    Ref::Committed(0),
                    vec![FieldRef::Committed(0)],
                )),
                Action::AddDataFile(
                    AddDataFile::try_new_unchecked(
                        Ref::Committed(0),
                        DataFile::new(
                            "data/0-new.lance",
                            vec![],
                            vec![0],
                            ConcreteFileVersion::V2_0,
                            None,
                            None,
                        ),
                        vec![FieldRef::Committed(0)],
                        None,
                        true,
                    )
                    .unwrap(),
                ),
            ],
            vec![index],
        );

        if !rejected {
            result.unwrap();
            return;
        }
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains(
                "index 'idx' has unknown coverage; migrate or rebuild it before rewriting field 0"
            ),
            "unexpected error: {error}"
        );
    }

    pub(super) fn apply_steps(manifest: &Manifest, steps: Vec<Vec<Action>>) -> Result<Manifest> {
        let steps = steps
            .into_iter()
            .enumerate()
            .map(|(index, actions)| UserStep::new(format!("step {index}"), actions))
            .collect();
        Transaction::new(
            manifest.version,
            Operation::CompositeOperation(CompositeOperation::new(steps)),
            None,
        )
        .build_manifest(
            Some(manifest),
            Vec::new(),
            "tx.txn",
            &default_build_config(),
        )
        .map(|(manifest, _)| manifest)
    }

    fn mint_fragment(token: u32) -> Action {
        Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
            id: Ref::Local(token),
            physical_rows: 10,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        }))
    }

    fn write_data(fragment: Ref, field: FieldRef) -> Action {
        Action::AddDataFile(
            AddDataFile::try_new_unchecked(
                fragment,
                one_column_file("data/new.lance"),
                vec![field],
                None,
                true,
            )
            .unwrap(),
        )
    }

    fn mint_field(token: u32, name: &str, parent: Option<FieldRef>) -> Action {
        let field = AddField::new(token, added_field(name));
        Action::AddField(match parent {
            Some(parent) => field.with_parent(parent),
            None => field,
        })
    }

    pub(super) fn remove_fragment_zero() -> Action {
        Action::RemoveFragment(RemoveFragment::new(Ref::Committed(0)))
    }

    fn drop_field_zero() -> Action {
        Action::DropField(DropField::new(FieldRef::Committed(0)))
    }

    /// Actions apply in order, across steps as within one, so a reference is
    /// only valid between what mints it and what removes it. `expected_error`
    /// is a fragment of the rejection, or `None` for a sequence that applies.
    #[rstest]
    #[case::use_before_mint(
        vec![vec![write_data(Ref::Local(0), FieldRef::Committed(0)), mint_fragment(0)]],
        Some("references local fragment token 0, which no earlier action in this operation minted"),
    )]
    #[case::child_before_its_local_parent(
        vec![vec![mint_field(1, "child", Some(FieldRef::Local(0))), mint_field(0, "parent", None)]],
        Some("references local field token 0, which no earlier action in this operation minted"),
    )]
    #[case::data_file_after_remove(
        vec![vec![remove_fragment_zero(), write_data(Ref::Committed(0), FieldRef::Committed(0))]],
        Some("AddDataFile targets fragment 0, which does not exist"),
    )]
    #[case::deletion_file_after_remove(
        vec![vec![
            remove_fragment_zero(),
            Action::SetDeletionFile(
                SetDeletionFile::try_new_unchecked(0, any_deletion_file(), None, true).unwrap(),
            ),
        ]],
        Some("SetDeletionFile targets fragment 0, which does not exist"),
    )]
    #[case::tombstone_after_remove(
        vec![vec![
            remove_fragment_zero(),
            Action::TombstoneFieldData(TombstoneFieldData::new(
                Ref::Committed(0),
                vec![FieldRef::Committed(0)],
            )),
        ]],
        Some("TombstoneFieldData targets fragment 0, which does not exist"),
    )]
    #[case::alter_after_drop(
        vec![vec![
            drop_field_zero(),
            Action::AlterField(AlterField::new(FieldRef::Committed(0)).with_name("renamed")),
        ]],
        Some("AlterField names field 0, which does not exist"),
    )]
    #[case::data_file_after_drop(
        vec![vec![drop_field_zero(), write_data(Ref::Committed(0), FieldRef::Committed(0))]],
        Some("AddDataFile for fragment 0 names field 0, which does not exist"),
    )]
    #[case::child_after_its_parent_is_dropped(
        vec![vec![drop_field_zero(), mint_field(0, "child", Some(FieldRef::Committed(0)))]],
        Some("AddField names parent field 0, which does not exist"),
    )]
    #[case::double_remove(
        vec![vec![remove_fragment_zero(), remove_fragment_zero()]],
        Some("RemoveFragment targets fragment 0, which does not exist"),
    )]
    #[case::double_drop(
        vec![vec![drop_field_zero(), drop_field_zero()]],
        Some("DropField names field 0, which does not exist"),
    )]
    #[case::token_used_a_step_before_it_is_minted(
        vec![
            vec![write_data(Ref::Local(0), FieldRef::Committed(0))],
            vec![mint_fragment(0)],
        ],
        Some("references local fragment token 0, which no earlier action in this operation minted"),
    )]
    #[case::token_minted_in_two_steps(
        vec![
            vec![mint_fragment(0), write_data(Ref::Local(0), FieldRef::Committed(0))],
            vec![mint_fragment(0), write_data(Ref::Local(0), FieldRef::Committed(0))],
        ],
        Some("local fragment token 0 is minted more than once in this operation"),
    )]
    #[case::committed_ref_to_a_fragment_from_before_a_reset(
        vec![vec![
            Action::ResetTable(ResetTable),
            mint_field(0, "x", None),
            write_data(Ref::Committed(0), FieldRef::Local(0)),
        ]],
        Some("AddDataFile targets fragment 0, which does not exist"),
    )]
    #[case::token_minted_a_step_before_it_is_used(
        vec![
            vec![mint_fragment(0)],
            vec![write_data(Ref::Local(0), FieldRef::Committed(0))],
        ],
        None,
    )]
    #[case::actions_before_a_reset(
        vec![vec![
            mint_fragment(0),
            write_data(Ref::Local(0), FieldRef::Committed(0)),
            Action::ResetTable(ResetTable),
            mint_field(1, "x", None),
            mint_fragment(1),
            write_data(Ref::Local(1), FieldRef::Local(1)),
        ]],
        None,
    )]
    fn test_an_action_sequence_is_checked_in_order(
        #[case] steps: Vec<Vec<Action>>,
        #[case] expected_error: Option<&str>,
    ) {
        let result = apply_steps(&backed_manifest(), steps);
        match expected_error {
            None => {
                result.unwrap();
            }
            Some(expected) => {
                let error = result.unwrap_err().to_string();
                assert!(error.contains(expected), "unexpected error: {error}");
            }
        }
    }

    /// A reserved id names a column of one data file, never a field of the
    /// schema, so only the action that binds a file's columns accepts one.
    #[rstest]
    #[case::tombstone(Action::TombstoneFieldData(TombstoneFieldData::new(
        Ref::Committed(0),
        vec![FieldRef::Committed(ROW_ID_FIELD_ID)],
    )))]
    #[case::drop(Action::DropField(DropField::new(FieldRef::Committed(ROW_ID_FIELD_ID))))]
    #[case::alter(Action::AlterField(
        AlterField::new(FieldRef::Committed(ROW_ID_FIELD_ID)).with_name("renamed")
    ))]
    #[case::parent(mint_field(0, "child", Some(FieldRef::Committed(ROW_ID_FIELD_ID))))]
    #[case::field_metadata(Action::ConfigUpdate(ConfigUpdate::default().with_field_metadata(
        FieldRef::Committed(ROW_ID_FIELD_ID),
        UpdateMap {
            update_entries: vec![],
            replace: false,
        },
    )))]
    fn test_a_reserved_field_id_names_no_schema_field(#[case] action: Action) {
        let error = apply(&backed_manifest(), vec![action]).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("names reserved field id -3"),
            "unexpected error: {error}"
        );
    }

    fn struct_field(name: &str) -> lance_core::datatypes::Field {
        lance_core::datatypes::Field::try_from(ArrowField::new(
            name,
            DataType::Struct(Fields::empty()),
            true,
        ))
        .unwrap()
    }

    /// The scanner injects the system columns at the top level, so a stored
    /// column by one of those names would collide with it on read. Nested
    /// fields are addressed through their parent and cannot collide.
    #[rstest]
    #[case::add_top_level(
        vec![mint_field(0, ROW_ID, None)],
        true
    )]
    #[case::add_nested(
        vec![
            Action::AddField(AddField::new(0, struct_field("s"))),
            mint_field(1, ROW_ID, Some(FieldRef::Local(0))),
        ],
        false
    )]
    #[case::rename_top_level(
        vec![Action::AlterField(AlterField::new(FieldRef::Committed(0)).with_name(ROW_ADDR))],
        true
    )]
    #[case::rename_nested(
        vec![
            Action::AddField(AddField::new(0, struct_field("s"))),
            mint_field(1, "child", Some(FieldRef::Local(0))),
            Action::AlterField(AlterField::new(FieldRef::Local(1)).with_name(ROW_ADDR)),
        ],
        false
    )]
    fn test_a_top_level_field_may_not_take_a_system_column_name(
        #[case] actions: Vec<Action>,
        #[case] rejected: bool,
    ) {
        let result = apply(&backed_manifest(), actions);
        if !rejected {
            result.unwrap();
            return;
        }
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error
                .to_string()
                .contains("is a reserved name and cannot be used in a Lance dataset"),
            "unexpected error: {error}"
        );
    }

    pub(super) fn config_update() -> Action {
        Action::ConfigUpdate(ConfigUpdate::default().with_config(UpdateMap {
            update_entries: vec![("k", "v").into()],
            replace: false,
        }))
    }

    fn fragment_counter_at_max() -> Manifest {
        let mut manifest = backed_manifest();
        manifest.max_fragment_id = Some(u32::MAX);
        manifest
    }

    fn field_counter_at_max() -> Manifest {
        let mut manifest = backed_manifest();
        manifest.max_allocated_field_id = Some(i32::MAX);
        manifest.writer_feature_flags |= FLAG_NON_REUSABLE_FIELD_IDS;
        manifest
    }

    fn base_counter_at_max() -> Manifest {
        let mut manifest = backed_manifest();
        manifest.base_paths.insert(
            u32::MAX,
            BasePath::new(u32::MAX, "s3://bucket/last".into(), None, false),
        );
        manifest
    }

    fn row_id_counter_at_max() -> Manifest {
        let mut fragment = backed_manifest().fragments[0].clone();
        fragment.row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&(0..10).into()).into()));
        let mut manifest = make_stable_row_id_manifest(vec![fragment]);
        manifest.next_row_id = u64::MAX - 5;
        manifest
    }

    /// Every id counter is a high-water mark, and a mint past its maximum is
    /// an error naming the counter, never a wrap. An operation that mints
    /// nothing still commits on a table whose counter is spent.
    #[rstest]
    #[case::fragment(fragment_counter_at_max(), mint_fragment(0), "fragment id")]
    #[case::fragment_reservation(
        fragment_counter_at_max(),
        Action::ReserveFragmentIds(ReserveFragmentIds::new(1)),
        "fragment id"
    )]
    #[case::field(field_counter_at_max(), mint_field(0, "added", None), "field id")]
    #[case::base(
        base_counter_at_max(),
        Action::AddBase(AddBase::new(0, BasePath::new(0, "s3://bucket/new".into(), None, false))),
        "base id"
    )]
    #[case::row_id(row_id_counter_at_max(), mint_fragment(0), "row id")]
    fn test_an_exhausted_counter_refuses_a_mint(
        #[case] manifest: Manifest,
        #[case] mint: Action,
        #[case] counter: &str,
    ) {
        apply(&manifest, vec![config_update()]).unwrap();

        let error = apply(&manifest, vec![mint]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error
                .to_string()
                .contains(&format!("{counter} counter is exhausted")),
            "unexpected error: {error}"
        );
    }

    fn rename_field_zero() -> Action {
        Action::AlterField(AlterField::new(FieldRef::Committed(0)).with_name("renamed"))
    }

    #[test]
    fn test_an_action_set_keeps_sticky_feature_flags() {
        let mut manifest = backed_manifest();
        manifest.reader_feature_flags |= FLAG_MIXED_DATA_FILE_VERSIONS;
        manifest.writer_feature_flags |= FLAG_MIXED_DATA_FILE_VERSIONS;
        manifest.activate_non_reusable_field_ids();
        manifest.writer_feature_flags |= FLAG_NON_REUSABLE_FIELD_IDS;

        let next = apply(&manifest, vec![rename_field_zero()]).unwrap();

        assert_ne!(next.reader_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS, 0);
        assert_ne!(next.writer_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS, 0);
        assert_ne!(next.writer_feature_flags & FLAG_NON_REUSABLE_FIELD_IDS, 0);
        assert_eq!(next.reader_feature_flags & FLAG_NON_REUSABLE_FIELD_IDS, 0);
    }

    /// The sticky bits are copied forward, so a source whose flags are
    /// inconsistent has to be rejected rather than masked into a valid-looking
    /// manifest, as the legacy path does.
    #[rstest]
    #[case::sticky_flag_on_the_reader_side_only(FLAG_MIXED_DATA_FILE_VERSIONS, 0, false)]
    #[case::non_reusable_field_ids_on_the_reader_side(
        FLAG_NON_REUSABLE_FIELD_IDS,
        FLAG_NON_REUSABLE_FIELD_IDS,
        true
    )]
    #[case::non_reusable_field_ids_without_its_high_water_mark(
        0,
        FLAG_NON_REUSABLE_FIELD_IDS,
        false
    )]
    fn test_an_action_set_rejects_inconsistent_sticky_flags(
        #[case] reader_flags: u64,
        #[case] writer_flags: u64,
        #[case] has_high_water_mark: bool,
    ) {
        let mut manifest = backed_manifest();
        manifest.reader_feature_flags |= reader_flags;
        manifest.writer_feature_flags |= writer_flags;
        if has_high_water_mark {
            manifest.activate_non_reusable_field_ids();
        }

        let error = apply(&manifest, vec![rename_field_zero()]).unwrap_err();

        assert!(
            matches!(error, Error::CorruptFile { .. }),
            "unexpected error: {error:?}"
        );
    }

    fn assert_rejected_beyond_read_mark(result: Result<Manifest>, named: &str) {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        let message = error.to_string();
        assert!(message.contains(named), "{message}");
        assert!(
            message.contains("read version's high-water mark"),
            "{message}"
        );
        assert!(message.contains("Local"), "{message}");
    }

    fn data_file_under_base(fragment: Ref, base: Ref) -> Action {
        Action::AddDataFile(
            AddDataFile::try_new_unchecked(
                fragment,
                one_column_file("data/new.lance"),
                vec![FieldRef::Committed(0)],
                Some(base),
                true,
            )
            .unwrap(),
        )
    }

    /// A committed id above the read version's mark names nothing the writer
    /// could have read. Even when this operation is what mints it, a replay
    /// onto a version where a concurrent commit took that id first would
    /// silently target the other writer's object, so the writer must name it
    /// by its local token.
    #[rstest]
    #[case::deletion_file_on_a_minted_fragment(
        vec![
            mint_fragment(0),
            Action::SetDeletionFile(
                SetDeletionFile::try_new_unchecked(5, any_deletion_file(), None, true).unwrap(),
            ),
        ],
        "fragment 5",
    )]
    #[case::data_file_for_a_minted_field(
        vec![
            mint_field(0, "added", None),
            mint_fragment(0),
            write_data(Ref::Local(0), FieldRef::Committed(1)),
        ],
        "field 1",
    )]
    #[case::data_file_under_a_minted_base(
        vec![
            mint_base(),
            mint_fragment(0),
            data_file_under_base(Ref::Local(0), Ref::Committed(2)),
        ],
        "base 2",
    )]
    fn test_a_committed_ref_to_an_id_minted_in_the_same_operation_is_rejected(
        #[case] actions: Vec<Action>,
        #[case] named: &str,
    ) {
        // Fragments 1-4 are reserved, so the fragment minted here is 5.
        let mut read = manifest_with_base();
        read.max_fragment_id = Some(4);

        assert_rejected_beyond_read_mark(apply(&read, actions), named);
    }

    /// The retry case the check exists for: the id exists by the time this
    /// operation applies, but a concurrent commit minted it for its own
    /// object.
    #[rstest]
    #[case::deletion_file(
        Action::SetDeletionFile(
            SetDeletionFile::try_new_unchecked(1, any_deletion_file(), None, true).unwrap()
        ),
        "fragment 1",
    )]
    #[case::remove_fragment(
        Action::RemoveFragment(RemoveFragment::new(Ref::Committed(1))),
        "fragment 1"
    )]
    #[case::data_file_for_a_field(write_data(Ref::Committed(0), FieldRef::Committed(1)), "field 1")]
    #[case::data_file_under_a_base(
        data_file_under_base(Ref::Local(0), Ref::Committed(2)),
        "base 2"
    )]
    fn test_a_committed_ref_to_an_id_a_concurrent_commit_minted_is_rejected(
        #[case] action: Action,
        #[case] named: &str,
    ) {
        let read = manifest_with_base();
        let current = apply(
            &read,
            vec![
                mint_field(0, "concurrent", None),
                mint_base(),
                mint_fragment(0),
                write_data(Ref::Local(0), FieldRef::Committed(0)),
            ],
        )
        .unwrap();

        assert_rejected_beyond_read_mark(
            apply_after_reading(&read, &current, vec![mint_fragment(0), action]),
            named,
        );
    }

    /// A reservation the operation could have read back is claimable on a
    /// retry; one a concurrent commit took after the read is not, even though
    /// the version applied onto has it.
    #[rstest]
    #[case::reserved_before_the_read(3, None)]
    #[case::reserved_concurrently(6, Some("no commit has reserved"))]
    fn test_a_fragment_claim_must_come_from_a_reservation_at_the_read_version(
        #[case] id: u64,
        #[case] expected_error: Option<&str>,
    ) {
        let mut read = backed_manifest();
        read.max_fragment_id = Some(4);
        let current = apply(
            &read,
            vec![Action::ReserveFragmentIds(ReserveFragmentIds::new(2))],
        )
        .unwrap();

        let result = apply_after_reading(
            &read,
            &current,
            vec![Action::AddFragment(AddFragment::new_unchecked(
                AddFragmentParts {
                    id: Ref::Committed(id),
                    physical_rows: 10,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                },
            ))],
        );

        match expected_error {
            None => {
                let next = result.unwrap();
                assert!(next.fragments.iter().any(|fragment| fragment.id == id));
            }
            Some(message) => {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
                assert!(error.to_string().contains(message), "{error}");
                assert!(error.to_string().contains("read version"), "{error}");
            }
        }
    }

    /// Without the read version there is nothing to check a committed id
    /// against, so an apply onto any other version fails closed.
    #[test]
    fn test_an_apply_without_the_read_version_fails_closed() {
        let read = backed_manifest();
        let current = apply(&read, vec![mint_fragment(0)]).unwrap();

        let error = Transaction::new(
            read.version,
            Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
                "step",
                vec![remove_fragment_zero()],
            )])),
            None,
        )
        .build_manifest(
            Some(&current),
            Vec::new(),
            "tx.txn",
            &default_build_config(),
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("read version"), "{error}");
    }

    /// `backed_manifest` with its stored field 0 required, plus a nullable
    /// field 1 that fragment 0 stores no data for, which is allowed while the
    /// field is nullable.
    fn manifest_with_an_unstored_nullable_field() -> Manifest {
        let mut manifest = backed_manifest();
        manifest.schema.field_by_id_mut(0).unwrap().nullable = false;
        apply(
            &manifest,
            vec![Action::AddField(AddField::new(0, added_field("loose")))],
        )
        .unwrap()
    }

    /// A fragment that stores no data for a non-nullable field reads it back
    /// as nulls, which the schema forbids. The legacy path checks every
    /// fragment an operation writes (#10211). Making a field required is a
    /// claim about every row, so it checks every fragment.
    #[rstest]
    #[case::minted_fragment_without_its_data(vec![mint_fragment(0)])]
    #[case::required_field_without_data(vec![Action::AddField(AddField::new(
        0,
        lance_core::datatypes::Field::try_from(ArrowField::new("strict", DataType::Int32, false))
            .unwrap(),
    ))])]
    #[case::tightened_field_without_data(vec![Action::AlterField(
        AlterField::new(FieldRef::Committed(1)).with_nullable(false)
    )])]
    fn test_a_fragment_must_store_every_required_field(#[case] actions: Vec<Action>) {
        let error = apply(&manifest_with_an_unstored_nullable_field(), actions).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("non-nullable field"), "{error}");
    }

    /// Only the fragments an operation writes are checked, so a fragment
    /// committed before the check existed does not block an unrelated commit.
    #[test]
    fn test_an_unwritten_fragment_missing_a_required_field_does_not_block_a_commit() {
        // sample_manifest's fragment 0 has no data files at all.
        let manifest = sample_manifest();
        let next = apply(
            &manifest,
            vec![
                mint_fragment(0),
                write_data(Ref::Local(0), FieldRef::Committed(0)),
            ],
        );
        assert!(next.is_ok(), "{next:?}");
    }

    /// The object-metadata writer bit is derived from the index list on every
    /// commit, so an action set has to set it too, or it would drop the fence
    /// for segments it carries over unchanged.
    #[test]
    fn test_an_action_set_keeps_the_object_metadata_fence() {
        let mut index = sample_index_metadata("idx");
        index
            .metadata
            .insert("org.example.test".into(), "opaque".into());

        let (manifest, _) = apply_with_indices(
            &backed_manifest(),
            vec![
                mint_fragment(0),
                write_data(Ref::Local(0), FieldRef::Committed(0)),
            ],
            vec![index],
        )
        .unwrap();

        assert_ne!(manifest.writer_feature_flags & FLAG_OBJECT_METADATA, 0);
    }

    fn add_top_level(def: lance_core::datatypes::Field) -> Action {
        Action::AddField(AddField::new(0, def))
    }

    fn arrow_def(name: &str, data_type: DataType) -> lance_core::datatypes::Field {
        lance_core::datatypes::Field::try_from(ArrowField::new(name, data_type, true)).unwrap()
    }

    /// `backed_manifest` with a nullable `s: struct<x: int32>` beside `id`.
    fn struct_manifest() -> Manifest {
        let mut manifest = backed_manifest();
        manifest.schema = Schema::try_from(&ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, true),
            ArrowField::new(
                "s",
                DataType::Struct(Fields::from(vec![ArrowField::new(
                    "x",
                    DataType::Int32,
                    true,
                )])),
                true,
            ),
        ]))
        .unwrap();
        manifest
    }

    /// Actions build the schema a field at a time, so apply checks what
    /// building it from Arrow would have guaranteed.
    #[rstest]
    #[case::dotted_top_level_name(
        vec![add_top_level(added_field("a.b"))],
        "cannot contain `.`"
    )]
    #[case::renamed_to_a_dotted_name(
        vec![Action::AlterField(AlterField::new(FieldRef::Committed(0)).with_name("x.y"))],
        "cannot contain `.`"
    )]
    #[case::zero_dimension_fixed_size_list(
        vec![add_top_level(arrow_def(
            "v",
            DataType::FixedSizeList(Arc::new(ArrowField::new("item", DataType::Int32, true)), 0),
        ))],
        "dimension 0"
    )]
    #[case::child_under_a_primitive(
        vec![Action::AddField(
            AddField::new(0, added_field("child")).with_parent(FieldRef::Committed(0))
        )],
        "takes no children"
    )]
    #[case::list_without_its_item(
        vec![add_top_level(arrow_def(
            "l",
            DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true))),
        ))],
        "takes exactly one child"
    )]
    #[case::struct_left_empty(
        vec![Action::DropField(DropField::new(FieldRef::Committed(2)))],
        "takes at least one child"
    )]
    fn test_apply_rejects_a_malformed_schema(#[case] actions: Vec<Action>, #[case] expected: &str) {
        let error = apply(&struct_manifest(), actions).unwrap_err();

        assert!(
            error.to_string().contains(expected),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_a_schema_malformed_before_the_commit_does_not_block_it() {
        let mut manifest = backed_manifest();
        manifest.schema.field_by_id_mut(0).unwrap().name = "a.b".into();

        apply(
            &manifest,
            vec![Action::ConfigUpdate(
                ConfigUpdate::default().with_schema_metadata(UpdateMap {
                    update_entries: vec![("note", Some("unrelated")).into()],
                    replace: false,
                }),
            )],
        )
        .unwrap();
    }

    fn set_schema_metadata(key: &str, value: &str) -> Action {
        Action::ConfigUpdate(ConfigUpdate::default().with_schema_metadata(UpdateMap {
            update_entries: vec![(key, Some(value)).into()],
            replace: false,
        }))
    }

    /// Readers that predate managed blobs must be fenced off once the schema
    /// opts in, as the legacy path does on every commit.
    #[test]
    fn test_opting_into_managed_blobs_sets_the_feature_flag() {
        let manifest = apply(
            &backed_manifest(),
            vec![set_schema_metadata(MANAGED_BLOBS_META_KEY, "true")],
        )
        .unwrap();

        assert_ne!(manifest.reader_feature_flags & FLAG_MANAGED_BLOBS, 0);
        assert_ne!(manifest.writer_feature_flags & FLAG_MANAGED_BLOBS, 0);

        let error = apply(
            &backed_manifest(),
            vec![set_schema_metadata(MANAGED_BLOBS_META_KEY, "maybe")],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("maybe"),
            "unexpected error: {error}"
        );
    }

    /// Turning range segments on re-encodes every fragment once, breaking
    /// `Range` runs a reader would otherwise fold into one interval across a
    /// wide gap.
    #[test]
    fn test_enabling_range_segments_reencodes_row_ids() {
        let mut row_ids = crate::rowids::RowIdSequence::from(0u64..100);
        row_ids.extend(crate::rowids::RowIdSequence::from(1_000_000u64..1_000_100));
        let mut fragment = Fragment::new(0);
        fragment.row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&row_ids).into()));
        fragment.physical_rows = Some(200);
        let manifest = make_stable_row_id_manifest(vec![fragment]);

        let next = apply(
            &manifest,
            vec![Action::ConfigUpdate(ConfigUpdate::default().with_config(
                UpdateMap {
                    update_entries: vec![
                        (crate::transaction::RANGE_SEGMENTS_CONFIG_KEY, Some("true")).into(),
                    ],
                    replace: false,
                },
            ))],
        )
        .unwrap();

        let Some(RowIdMeta::Inline(data)) = &next.fragments[0].row_id_meta else {
            panic!("expected inline row ids");
        };
        assert!(!read_row_ids(data).unwrap().range_runs_span_a_gap());
    }

    /// Fragments 0 and 1 of a stable row id table, each backed by a file for a
    /// nullable field 0, so its data can be tombstoned without a replacement.
    /// Every row was last updated at version 2.
    fn stable_row_id_backed_manifest() -> Manifest {
        let fragments = (0..2u64)
            .map(|id| {
                let mut fragment = backed_manifest().fragments[0].clone();
                fragment.id = id;
                fragment.files[0].path = format!("data/{id}.lance");
                fragment.row_id_meta = Some(RowIdMeta::Inline(
                    write_row_ids(&(id * 10..id * 10 + 10).into()).into(),
                ));
                fragment.last_updated_at_version_meta = Some(
                    RowDatasetVersionMeta::from_sequence(
                        &RowDatasetVersionSequence::from_uniform_row_count(10, 2),
                    )
                    .unwrap(),
                );
                fragment
            })
            .collect();
        let mut manifest = make_stable_row_id_manifest(fragments);
        manifest.schema.field_by_id_mut(0).unwrap().nullable = true;
        manifest
    }

    fn tombstone_field_zero(data_change: bool) -> Action {
        Action::TombstoneFieldData(TombstoneFieldData::new_unchecked(
            Ref::Committed(0),
            vec![FieldRef::Committed(0)],
            data_change,
        ))
    }

    fn reencode_field_zero(data_change: bool) -> Action {
        Action::AddDataFile(
            AddDataFile::try_new_unchecked(
                Ref::Committed(0),
                one_column_file("data/reencoded.lance"),
                vec![FieldRef::Committed(0)],
                None,
                data_change,
            )
            .unwrap(),
        )
    }

    fn new_column() -> Vec<Action> {
        vec![
            mint_field(0, "added", None),
            write_data(Ref::Committed(0), FieldRef::Local(0)),
        ]
    }

    fn dropped_column() -> Vec<Action> {
        vec![tombstone_field_zero(true)]
    }

    fn rewritten_column() -> Vec<Action> {
        vec![tombstone_field_zero(true), reencode_field_zero(true)]
    }

    /// `get_updated_rows()` reads `last_updated_at_version_meta`, which only a
    /// restamp of the changed rows keeps current. Apply cannot tell which rows
    /// a file for an existing fragment changed, so a value change to a
    /// committed fragment of a stable row id table needs a
    /// `RefreshRowVersionMetadata` covering that fragment, before or after the
    /// change. The writer's selection is taken as given, down to one row.
    #[rstest]
    #[case::not_restamped(vec![], vec![], None)]
    #[case::restamped_whole(vec![], vec![refresh(&[0])], Some(&[true; 10][..]))]
    #[case::restamped_rows(
        vec![],
        vec![refresh_rows(0, &[1, 3])],
        Some(&[false, true, false, true, false, false, false, false, false, false][..])
    )]
    #[case::restamped_before_the_change(vec![refresh(&[0])], vec![], Some(&[true; 10][..]))]
    #[case::other_fragment_restamped(vec![], vec![refresh(&[1])], None)]
    fn test_a_value_change_to_a_committed_fragment_needs_a_row_version_restamp(
        #[values(new_column, dropped_column, rewritten_column)] change: fn() -> Vec<Action>,
        #[case] before: Vec<Action>,
        #[case] after: Vec<Action>,
        #[case] restamped_rows: Option<&[bool]>,
    ) {
        let manifest = stable_row_id_backed_manifest();
        let actions = [before, change(), after].concat();

        let result = apply(&manifest, actions);

        match restamped_rows {
            Some(restamped_rows) => {
                let next = result.unwrap();
                let expected: Vec<u64> = restamped_rows
                    .iter()
                    .map(|&restamped| if restamped { manifest.version + 1 } else { 2 })
                    .collect();
                assert_eq!(last_updated_versions(&next.fragments[0]), expected);
            }
            None => {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
                let message = error.to_string();
                assert!(message.contains("fragment 0"), "{message}");
                assert!(!message.contains("fragment 1"), "{message}");
                assert!(
                    message.contains("last_updated_at_version_meta"),
                    "{message}"
                );
                assert!(message.contains("RefreshRowVersionMetadata"), "{message}");
                // Without stable row ids there is no row version to keep.
                let mut manifest = backed_manifest();
                manifest.schema.field_by_id_mut(0).unwrap().nullable = true;
                apply(&manifest, change()).unwrap();
            }
        }
    }

    /// Moving values, writing new fragments, and deleting rows leave no row's
    /// value changed in place, so none of them needs a restamp.
    #[rstest]
    #[case::reencode_of_a_committed_fragment(vec![
        tombstone_field_zero(false),
        reencode_field_zero(false),
    ])]
    #[case::new_fragment(vec![
        mint_fragment(0),
        write_data(Ref::Local(0), FieldRef::Committed(0)),
    ])]
    #[case::deletion_in_a_committed_fragment(vec![Action::SetDeletionFile(
        SetDeletionFile::try_new_unchecked(0, any_deletion_file(), None, true).unwrap(),
    )])]
    #[case::changed_fragment_removed_afterwards(vec![
        tombstone_field_zero(true),
        Action::RemoveFragment(RemoveFragment::new(Ref::Committed(0))),
    ])]
    fn test_a_change_leaving_no_value_changed_in_place_needs_no_restamp(
        #[case] actions: Vec<Action>,
    ) {
        apply(&stable_row_id_backed_manifest(), actions).unwrap();
    }

    /// The fragment reuse history is not an index on the action path: no
    /// index action may name it, so it changes only through its own actions.
    #[rstest]
    #[case::add_segment(|uuid: Uuid| Action::AddIndexSegment(
        AddIndexSegment::try_new_unchecked(AddIndexSegmentParts {
            uuid: Uuid::from_u128(uuid.as_u128() + 1),
            name: FRAG_REUSE_INDEX_NAME.into(),
            fields: Vec::new(),
            covering_fields: Vec::new(),
            index_details: None,
            index_version: 1,
            covered_fragments: Some(Vec::new()),
            files: Vec::new(),
            base: None,
            created_at: None,
            dataset_version: None,
            data_change: false,
        })
        .unwrap(),
    ))]
    #[case::remove_segment(|uuid| Action::RemoveIndexSegment(
        RemoveIndexSegment::try_new_unchecked(uuid, FRAG_REUSE_INDEX_NAME, false).unwrap(),
    ))]
    #[case::adjust_coverage(|uuid| Action::AdjustIndexCoverage(
        AdjustIndexCoverage::try_new_unchecked(AdjustIndexCoverageParts {
            uuid,
            name: FRAG_REUSE_INDEX_NAME.into(),
            add_fragments: Vec::new(),
            remove_fragments: vec![0],
            fields: Vec::new(),
            covering_fields: Vec::new(),
        })
        .unwrap(),
    ))]
    #[case::drop_index(|_| Action::DropIndex(DropIndex::try_new(FRAG_REUSE_INDEX_NAME).unwrap()))]
    fn test_an_index_action_may_not_name_the_fragment_reuse_history(
        #[case] action: fn(Uuid) -> Action,
    ) {
        let mut entry = sample_index_metadata(FRAG_REUSE_INDEX_NAME);
        entry.fields.clear();
        entry.index_version = 0;
        let error = apply_with_indices(&backed_manifest(), vec![action(entry.uuid)], vec![entry])
            .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains(FRAG_REUSE_INDEX_NAME), "{error}");
    }

    pub(super) fn whole_fragment(token: u32) -> Vec<Action> {
        vec![
            mint_fragment(token),
            write_data(Ref::Local(token), FieldRef::Committed(0)),
        ]
    }

    fn inline_row_ids(fragment: &Fragment) -> Vec<u64> {
        let Some(RowIdMeta::Inline(row_ids)) = &fragment.row_id_meta else {
            panic!("fragment {} has no inline row ids", fragment.id);
        };
        read_row_ids(row_ids).unwrap().iter().collect()
    }

    fn build_without_transaction_file(
        base: &Manifest,
        indices: &[IndexMetadata],
        steps: &[UserStep],
        config: &ManifestBuildConfig,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        Transaction::new(
            base.version,
            Operation::CompositeOperation(CompositeOperation::new(steps.to_vec())),
            None,
        )
        .build_manifest(Some(base), indices.to_vec(), "tx.txn", config)
        .map(|(mut manifest, indices)| {
            manifest.transaction_file = None;
            (manifest, indices)
        })
    }

    #[rstest]
    #[case::two_separate_appends(vec![whole_fragment(0), whole_fragment(1)])]
    #[case::three_separate_appends(vec![whole_fragment(0), whole_fragment(1), whole_fragment(2)])]
    #[case::file_in_a_later_step(vec![
        vec![mint_fragment(0)],
        vec![write_data(Ref::Local(0), FieldRef::Committed(0))],
    ])]
    #[case::add_base_then_fragment_in_later_step(vec![
        vec![Action::AddBase(AddBase::new(
            0,
            BasePath::new(0, "s3://bucket/ext".into(), Some("ext".into()), false),
        ))],
        vec![
            mint_fragment(0),
            Action::AddDataFile(
                AddDataFile::try_new_unchecked(
                    Ref::Local(0),
                    one_column_file("data/ext.lance"),
                    vec![FieldRef::Committed(0)],
                    Some(Ref::Local(0)),
                    true,
                )
                .unwrap(),
            ),
        ],
    ])]
    #[case::reservations_before_a_fragment(vec![
        vec![
            Action::ReserveFragmentIds(ReserveFragmentIds::new(5)),
            Action::ReserveRowIds(ReserveRowIds::new(7)),
        ],
        whole_fragment(0),
    ])]
    #[case::staged_fragment_removed_later(vec![
        whole_fragment(0),
        whole_fragment(1),
        vec![Action::RemoveFragment(RemoveFragment::new(Ref::Local(0)))],
    ])]
    fn test_a_snapshot_after_each_step_matches_one_build_of_the_steps_so_far(
        #[case] steps: Vec<Vec<Action>>,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        use crate::transaction::action::test_support::frag_reuse::table;

        let base = table(2, stable_row_ids);
        let indices = vec![sample_index_metadata("idx")];
        let config = default_build_config();
        let reserves_row_ids = steps
            .iter()
            .flatten()
            .any(|action| matches!(action, Action::ReserveRowIds(_)));
        let steps: Vec<UserStep> = steps
            .into_iter()
            .map(|actions| UserStep::new("step", actions))
            .collect();
        let outcome = |built: Result<(Manifest, Vec<IndexMetadata>)>| {
            built.map_err(|error| error.to_string())
        };

        let mut working =
            ApplyState::at_read_version(Arc::new(base.clone()), indices.clone(), &config).unwrap();
        let mut last_snapshot = None;
        for applied in 1..=steps.len() {
            let mut next = working.clone();
            next.apply_step(&steps[applied - 1]).unwrap();
            let validation = next
                .validate_step_invariants()
                .map_err(|error| error.to_string());
            working = next;
            let snapshot = outcome(working.build_manifest());
            assert_eq!(
                validation.err(),
                snapshot.as_ref().err().cloned(),
                "after step {applied}"
            );
            assert_eq!(
                snapshot,
                outcome(build_without_transaction_file(
                    &base,
                    &indices,
                    &steps[..applied],
                    &config,
                )),
                "after step {applied}"
            );
            if let Ok((manifest, _)) = &snapshot {
                assert_eq!(manifest.transaction_file, None, "after step {applied}");
            }
            last_snapshot = Some(snapshot);
        }
        let last_snapshot = last_snapshot.unwrap();
        assert_eq!(
            last_snapshot.is_ok(),
            stable_row_ids || !reserves_row_ids,
            "{last_snapshot:?}"
        );
        if stable_row_ids {
            let (manifest, _) = last_snapshot.unwrap();
            assert!(manifest.next_row_id > base.next_row_id);
        }
    }

    #[test]
    fn test_add_base_followed_by_add_fragment_in_later_step_binds_base_id() {
        use crate::transaction::action::test_support::frag_reuse::table;

        let base = table(1, true);
        let config = default_build_config();
        let mut state = ApplyState::at_read_version(Arc::new(base), Vec::new(), &config).unwrap();

        state
            .apply_step(&UserStep::new(
                "add_base",
                vec![Action::AddBase(AddBase::new(
                    0,
                    BasePath::new(0, "s3://bucket/ext".into(), Some("ext".into()), false),
                ))],
            ))
            .unwrap();
        let (intermediate, _) = state.build_manifest().unwrap();
        assert_eq!(intermediate.base_paths[&1].path, "s3://bucket/ext");

        state
            .apply_step(&UserStep::new(
                "append_on_base",
                vec![
                    mint_fragment(0),
                    Action::AddDataFile(
                        AddDataFile::try_new_unchecked(
                            Ref::Local(0),
                            one_column_file("data/ext.lance"),
                            vec![FieldRef::Committed(0)],
                            Some(Ref::Local(0)),
                            true,
                        )
                        .unwrap(),
                    ),
                ],
            ))
            .unwrap();

        let (manifest, _) = state.build_manifest().unwrap();
        assert_eq!(manifest.base_paths[&1].path, "s3://bucket/ext");
        let added = manifest.fragments.iter().find(|f| f.id == 1).unwrap();
        assert_eq!(added.files[0].base_id, Some(1));
    }

    #[test]
    fn test_stable_row_ids_stay_fixed_across_fragment_removal_and_reset() {
        use crate::transaction::action::test_support::frag_reuse::table;

        let base = table(1, true);
        let config = default_build_config();
        let mut state = ApplyState::at_read_version(Arc::new(base), Vec::new(), &config).unwrap();

        state
            .apply_step(&UserStep::new("append_a", whole_fragment(0)))
            .unwrap();
        state
            .apply_step(&UserStep::new("append_b", whole_fragment(1)))
            .unwrap();
        let (before_remove, _) = state.build_manifest().unwrap();
        let frag_b_before = before_remove.fragments.iter().find(|f| f.id == 2).unwrap();
        assert_eq!(
            inline_row_ids(frag_b_before),
            (1010..1020).collect::<Vec<_>>()
        );
        assert_eq!(before_remove.next_row_id, 1020);

        state
            .apply_step(&UserStep::new(
                "remove_a",
                vec![Action::RemoveFragment(RemoveFragment::new(Ref::Local(0)))],
            ))
            .unwrap();
        let (after_remove, _) = state.build_manifest().unwrap();
        let frag_b_after = after_remove.fragments.iter().find(|f| f.id == 2).unwrap();
        assert_eq!(
            inline_row_ids(frag_b_after),
            (1010..1020).collect::<Vec<_>>()
        );
        assert_eq!(after_remove.next_row_id, 1020);

        state
            .apply_step(&UserStep::new(
                "reset_and_rebuild",
                vec![
                    Action::ResetTable(ResetTable),
                    mint_field(0, "fresh", None),
                    mint_fragment(2),
                    write_data(Ref::Local(2), FieldRef::Local(0)),
                ],
            ))
            .unwrap();
        let (after_reset, _) = state.build_manifest().unwrap();
        assert_eq!(after_reset.fragments.len(), 1);
        assert_eq!(after_reset.fragments[0].id, 3);
        assert_eq!(
            inline_row_ids(&after_reset.fragments[0]),
            (1020..1030).collect::<Vec<_>>()
        );
        assert_eq!(after_reset.next_row_id, 1030);
    }

    #[test]
    fn test_repeated_manifest_builds_do_not_consume_reservations_or_double_apply_updates() {
        use crate::transaction::action::test_support::frag_reuse::table;

        let base = table(2, true);
        let shard = Uuid::from_u128(19);
        let indices = vec![
            IndexMetadata {
                fragment_bitmap: Some(RoaringBitmap::from_iter([0u32, 1])),
                ..sample_index_metadata("idx")
            },
            new_mem_wal_index_meta(
                base.version,
                MemWalIndexDetails {
                    compacted_sstables: vec![CompactedSsTable::new(shard, 5)],
                    index_catchup: vec![IndexCatchupProgress::new(
                        "idx".to_string(),
                        vec![CompactedSsTable::new(shard, 5)],
                    )],
                    ..Default::default()
                },
            )
            .unwrap(),
        ];
        let config = default_build_config();
        let step_one = UserStep::new(
            "step_one",
            vec![
                mint_fragment(0),
                write_data(Ref::Local(0), FieldRef::Committed(0)),
                Action::ReserveFragmentIds(ReserveFragmentIds::new(3)),
                Action::ReserveRowIds(ReserveRowIds::new(5)),
                tombstone_field_zero(true),
                reencode_field_zero(true),
                refresh(&[0]),
            ],
        );
        let step_two = UserStep::new("step_two", whole_fragment(1));

        let mut state =
            ApplyState::at_read_version(Arc::new(base.clone()), indices.clone(), &config).unwrap();
        state.apply_step(&step_one).unwrap();

        let first_build = state.build_manifest().unwrap();
        let second_build = state.build_manifest().unwrap();
        assert_eq!(first_build.0, second_build.0);
        assert_eq!(first_build.1[0], second_build.1[0]);
        assert_eq!(
            load_mem_wal_index_details(first_build.1[1].clone()).unwrap(),
            load_mem_wal_index_details(second_build.1[1].clone()).unwrap()
        );
        assert!(
            load_mem_wal_index_details(first_build.1[1].clone())
                .unwrap()
                .index_catchup
                .is_empty()
        );
        assert_eq!(first_build.0.max_fragment_id, Some(5));
        assert_eq!(first_build.0.next_row_id, base.next_row_id + 15);
        assert_eq!(first_build.0.transaction_file, None);

        state.apply_step(&step_two).unwrap();
        let final_build = state.into_manifest().unwrap();
        let expected =
            build_without_transaction_file(&base, &indices, &[step_one, step_two], &config)
                .unwrap();
        assert_eq!(final_build.0, expected.0);
        assert_eq!(final_build.1[0], expected.1[0]);
        assert_eq!(
            load_mem_wal_index_details(final_build.1[1].clone()).unwrap(),
            load_mem_wal_index_details(expected.1[1].clone()).unwrap()
        );
        assert_eq!(final_build.0.max_fragment_id, Some(6));
        assert_eq!(final_build.0.next_row_id, base.next_row_id + 25);
    }

    #[test]
    fn test_rebase_renumbers_local_fragments_and_checks_read_marks() {
        use crate::transaction::action::test_support::frag_reuse::table;

        let read_manifest = table(2, true);
        let config = default_build_config();
        let step_one = UserStep::new("append_a", whole_fragment(0));
        let step_two = UserStep::new("append_b", whole_fragment(1));

        let mut staged =
            ApplyState::at_read_version(Arc::new(read_manifest.clone()), Vec::new(), &config)
                .unwrap();
        staged.apply_step(&step_one).unwrap();
        staged.apply_step(&step_two).unwrap();
        let (staged_manifest, _) = staged.build_manifest().unwrap();
        let staged_ids: Vec<u64> = staged_manifest.fragments.iter().map(|f| f.id).collect();
        assert_eq!(staged_ids, vec![0, 1, 2, 3]);

        let (current_manifest, current_indices) = Transaction::new(
            read_manifest.version,
            Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
                "concurrent_append",
                whole_fragment(0),
            )])),
            None,
        )
        .build_manifest(Some(&read_manifest), Vec::new(), "concurrent.txn", &config)
        .unwrap();
        assert_eq!(current_manifest.max_fragment_id(), Some(2));
        assert_eq!(current_manifest.next_row_id, 1010);

        let (rebased, _) = Transaction::new(
            read_manifest.version,
            Operation::CompositeOperation(CompositeOperation::new(vec![step_one, step_two])),
            None,
        )
        .build_manifest_with_read_version(
            Some(&current_manifest),
            current_indices.clone(),
            "rebased.txn",
            &config,
            Some(ReadVersionState {
                manifest: &read_manifest,
                indices: &[],
            }),
        )
        .unwrap();
        let rebased_ids: Vec<u64> = rebased.fragments.iter().map(|f| f.id).collect();
        assert_eq!(rebased_ids, vec![0, 1, 2, 3, 4]);
        assert_eq!(
            inline_row_ids(rebased.fragments.iter().find(|f| f.id == 3).unwrap()),
            (1010..1020).collect::<Vec<_>>()
        );
        assert_eq!(
            inline_row_ids(rebased.fragments.iter().find(|f| f.id == 4).unwrap()),
            (1020..1030).collect::<Vec<_>>()
        );
        assert_eq!(rebased.next_row_id, 1030);

        let illegal = Transaction::new(
            read_manifest.version,
            Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
                "touch_concurrent",
                vec![Action::RemoveFragment(RemoveFragment::new(Ref::Committed(
                    2,
                )))],
            )])),
            None,
        )
        .build_manifest_with_read_version(
            Some(&current_manifest),
            current_indices,
            "illegal.txn",
            &config,
            Some(ReadVersionState {
                manifest: &read_manifest,
                indices: &[],
            }),
        )
        .unwrap_err();
        assert!(
            illegal
                .to_string()
                .contains("high-water mark for fragment ids is 1"),
            "{illegal}"
        );
    }

    #[test]
    fn test_index_coverage_across_multi_step_additions_rewrites_and_removals() {
        use crate::transaction::action::test_support::frag_reuse::table;

        let base = table(2, true);
        let config = default_build_config();
        let seg_uuid = Uuid::from_u128(42);
        let initial_index = IndexMetadata {
            uuid: seg_uuid,
            fragment_bitmap: Some(RoaringBitmap::from_iter([0u32, 1])),
            ..sample_index_metadata("idx")
        };

        let steps = vec![
            UserStep::new("append_frag_2", whole_fragment(0)),
            UserStep::new(
                "cover_frag_2",
                vec![Action::AdjustIndexCoverage(
                    AdjustIndexCoverage::try_new_unchecked(AdjustIndexCoverageParts {
                        uuid: seg_uuid,
                        name: "idx".into(),
                        add_fragments: vec![Ref::Local(0)],
                        remove_fragments: Vec::new(),
                        fields: vec![FieldRef::Committed(0)],
                        covering_fields: Vec::new(),
                    })
                    .unwrap(),
                )],
            ),
            UserStep::new(
                "rewrite_frag_0",
                vec![
                    tombstone_field_zero(true),
                    reencode_field_zero(true),
                    refresh(&[0]),
                ],
            ),
            UserStep::new(
                "remove_frag_1",
                vec![Action::RemoveFragment(RemoveFragment::new(Ref::Committed(
                    1,
                )))],
            ),
        ];

        let mut state = ApplyState::at_read_version(
            Arc::new(base.clone()),
            vec![initial_index.clone()],
            &config,
        )
        .unwrap();
        for step in &steps {
            state.apply_step(step).unwrap();
        }
        let (manifest, indices) = state.build_manifest().unwrap();
        assert_eq!(
            manifest.fragments.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec![0, 2]
        );
        let live_bitmap: RoaringBitmap = manifest.fragments.iter().map(|f| f.id as u32).collect();
        assert_eq!(
            indices[0].effective_fragment_bitmap(&live_bitmap),
            Some(RoaringBitmap::from_iter([2u32]))
        );
        assert_eq!(
            (manifest, indices),
            build_without_transaction_file(&base, &[initial_index], &steps, &config).unwrap()
        );
    }

    #[test]
    fn test_rewriting_a_newly_added_field_prunes_an_index_built_in_an_earlier_step() {
        use crate::transaction::action::test_support::frag_reuse::table;

        let base = table(1, true);
        let config = default_build_config();
        let seg_uuid = Uuid::from_u128(77);
        let mut state = ApplyState::at_read_version(Arc::new(base), Vec::new(), &config).unwrap();

        state
            .apply_step(&UserStep::new(
                "add_field_and_index",
                vec![
                    mint_field(0, "added", None),
                    write_data(Ref::Committed(0), FieldRef::Local(0)),
                    refresh(&[0]),
                    Action::AddIndexSegment(
                        AddIndexSegment::try_new_unchecked(AddIndexSegmentParts {
                            uuid: seg_uuid,
                            name: "added_idx".into(),
                            fields: vec![FieldRef::Local(0)],
                            covering_fields: Vec::new(),
                            index_details: None,
                            index_version: 1,
                            covered_fragments: Some(vec![Ref::Committed(0)]),
                            files: Vec::new(),
                            base: None,
                            created_at: None,
                            dataset_version: None,
                            data_change: false,
                        })
                        .unwrap(),
                    ),
                ],
            ))
            .unwrap();

        state
            .apply_step(&UserStep::new(
                "rewrite_added_field",
                vec![
                    Action::TombstoneFieldData(TombstoneFieldData::new(
                        Ref::Committed(0),
                        vec![FieldRef::Local(0)],
                    )),
                    write_data(Ref::Committed(0), FieldRef::Local(0)),
                    refresh(&[0]),
                ],
            ))
            .unwrap();

        let (_, indices) = state.build_manifest().unwrap();
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().is_empty());
    }

    #[test]
    fn test_a_failed_step_on_a_clone_leaves_the_kept_state_unchanged() {
        use crate::transaction::action::test_support::frag_reuse::table;

        let base = table(2, true);
        let config = default_build_config();
        let mut working =
            ApplyState::at_read_version(Arc::new(base.clone()), Vec::new(), &config).unwrap();
        let first = UserStep::new("step", whole_fragment(0));
        working = working.apply_step_checked(&first).unwrap();
        let before = working.build_manifest().unwrap();

        let error = working
            .apply_step_checked(&UserStep::new(
                "step",
                vec![mint_fragment(1), mint_fragment(0)],
            ))
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("local fragment token 0 is minted more than once in this operation"),
            "{error}"
        );
        assert_eq!(working.build_manifest().unwrap(), before);

        let error = working
            .apply_step_checked(&UserStep::new("step", vec![mint_fragment(1)]))
            .err()
            .unwrap();
        assert!(error.to_string().contains("non-nullable field"), "{error}");
        assert_eq!(working.build_manifest().unwrap(), before);
        working.validate_step_invariants().unwrap();

        let second = UserStep::new("step", whole_fragment(1));
        working = working.apply_step_checked(&second).unwrap();
        let (manifest, indices) = working.build_manifest().unwrap();
        let fragment_ids: Vec<u64> = manifest.fragments.iter().map(|f| f.id).collect();
        assert_eq!(fragment_ids, vec![0, 1, 2, 3]);
        assert_eq!(
            (manifest, indices),
            build_without_transaction_file(&base, &[], &[first, second], &config).unwrap()
        );
    }

    #[test]
    fn test_row_ids_follow_the_order_fragments_were_added() {
        use crate::transaction::action::test_support::frag_reuse::table;

        let mut base = table(1, true);
        base.max_fragment_id = Some(2);
        let claim_reserved = Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
            id: Ref::Committed(1),
            physical_rows: 10,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        }));

        let next = apply_steps(
            &base,
            vec![vec![
                mint_fragment(0),
                write_data(Ref::Local(0), FieldRef::Committed(0)),
                claim_reserved,
                write_data(Ref::Committed(1), FieldRef::Committed(0)),
            ]],
        )
        .unwrap();

        let frag_3 = next.fragments.iter().find(|f| f.id == 3).unwrap();
        let frag_1 = next.fragments.iter().find(|f| f.id == 1).unwrap();
        assert_eq!(inline_row_ids(frag_3), (1000..1010).collect::<Vec<_>>());
        assert_eq!(inline_row_ids(frag_1), (1010..1020).collect::<Vec<_>>());
        assert_eq!(next.next_row_id, 1020);
    }

    fn strict_field() -> Action {
        Action::AddField(AddField::new(
            0,
            lance_core::datatypes::Field::try_from(ArrowField::new(
                "strict",
                DataType::Int32,
                false,
            ))
            .unwrap(),
        ))
    }

    fn whole_fragment_with_row_ids() -> Vec<Action> {
        vec![
            Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
                id: Ref::Local(0),
                physical_rows: 10,
                row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&(100..110).into()).into())),
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            })),
            write_data(Ref::Local(0), FieldRef::Committed(0)),
        ]
    }

    #[rstest]
    #[case::whole_fragment(frag_reuse::table(2, true), whole_fragment(0), None)]
    #[case::removed_fragment(frag_reuse::table(2, true), vec![remove_fragment_zero()], None)]
    #[case::renamed_field(frag_reuse::table(2, true), vec![rename_field_zero()], None)]
    #[case::config(frag_reuse::table(2, true), vec![config_update()], None)]
    #[case::deletion_on_a_fragment_missing_a_required_field(
        sample_manifest(),
        vec![Action::SetDeletionFile(
            SetDeletionFile::try_new_unchecked(0, any_deletion_file(), None, true).unwrap(),
        )],
        None
    )]
    #[case::fragment_without_its_data(
        frag_reuse::table(2, true),
        vec![mint_fragment(0)],
        Some("non-nullable field")
    )]
    #[case::data_change_without_a_refresh(
        frag_reuse::table(2, true),
        replace_field_data(0, 0, true),
        Some("RefreshRowVersionMetadata")
    )]
    #[case::reencode_without_its_rebind(
        frag_reuse::table(2, true),
        replace_field_data(0, 0, false).into_iter().take(1).collect(),
        Some("without a data file")
    )]
    #[case::fragment_reservation_past_the_counter(
        frag_reuse::table(2, true),
        vec![Action::ReserveFragmentIds(ReserveFragmentIds::new(u32::MAX))],
        Some("fragment id counter is exhausted")
    )]
    #[case::row_id_mint_past_the_counter(
        row_id_counter_at_max(),
        vec![mint_fragment(0)],
        Some("row id counter is exhausted")
    )]
    #[case::row_id_reservation_without_a_counter(
        frag_reuse::table(2, false),
        vec![Action::ReserveRowIds(ReserveRowIds::new(7))],
        Some("ReserveRowIds needs a row id counter")
    )]
    #[case::supplied_row_ids_without_a_counter(
        frag_reuse::table(2, false),
        whole_fragment_with_row_ids(),
        Some("All fragments must have row ids")
    )]
    #[case::required_field_no_fragment_stores(
        frag_reuse::table(2, true),
        vec![strict_field()],
        Some("non-nullable field")
    )]
    fn test_step_validation_gives_the_verdict_of_a_snapshot_build(
        #[case] base: Manifest,
        #[case] step: Vec<Action>,
        #[case] rejected_with: Option<&str>,
    ) {
        let config = default_build_config();
        let mut stepped = ApplyState::at_read_version(
            Arc::new(base),
            vec![sample_index_metadata("idx")],
            &config,
        )
        .unwrap();
        stepped.apply_step(&UserStep::new("step", step)).unwrap();

        let validation = stepped
            .validate_step_invariants()
            .map_err(|error| error.to_string());
        let snapshot = stepped
            .build_manifest()
            .map(|_| ())
            .map_err(|error| error.to_string());
        assert_eq!(validation, snapshot);
        match rejected_with {
            None => validation.unwrap(),
            Some(expected) => {
                let error = validation.unwrap_err();
                assert!(error.contains(expected), "{error}");
            }
        }
    }

    fn table_with_a_legacy_blob_field() -> Manifest {
        let mut manifest = frag_reuse::table(2, true);
        manifest.schema.fields[0]
            .metadata
            .insert(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string());
        manifest
    }

    fn mem_wal_index() -> IndexMetadata {
        new_mem_wal_index_meta(1, MemWalIndexDetails::default()).unwrap()
    }

    fn mem_wal_segment() -> Vec<Action> {
        vec![Action::AddIndexSegment(
            AddIndexSegment::try_new_unchecked(AddIndexSegmentParts {
                uuid: Uuid::from_u128(11),
                name: MEM_WAL_INDEX_NAME.into(),
                fields: Vec::new(),
                covering_fields: Vec::new(),
                index_details: None,
                index_version: 1,
                covered_fragments: None,
                files: Vec::new(),
                base: None,
                created_at: None,
                dataset_version: None,
                data_change: true,
            })
            .unwrap(),
        )]
    }

    #[rstest]
    #[case::whole_fragment(frag_reuse::table(2, true), Vec::new(), whole_fragment(0), false)]
    #[case::removed_fragment(
        frag_reuse::table(2, true),
        Vec::new(),
        vec![remove_fragment_zero()],
        false
    )]
    #[case::fragment_reservation(
        frag_reuse::table(2, true),
        Vec::new(),
        vec![Action::ReserveFragmentIds(ReserveFragmentIds::new(5))],
        false
    )]
    #[case::renamed_field(frag_reuse::table(2, true), Vec::new(), vec![rename_field_zero()], true)]
    #[case::added_field(
        frag_reuse::table(2, true),
        Vec::new(),
        vec![mint_field(0, "added", None)],
        true
    )]
    #[case::config(frag_reuse::table(2, true), Vec::new(), vec![config_update()], true)]
    #[case::reset(
        frag_reuse::table(2, true),
        Vec::new(),
        vec![Action::ResetTable(ResetTable)],
        true
    )]
    #[case::compaction_with_a_transition(
        frag_reuse::table(2, true),
        Vec::new(),
        frag_reuse::compaction(
            &[0, 1],
            0,
            true,
            Some(frag_reuse::ordered(frag_reuse::survivors(&[0, 1], &[])))
        ),
        true
    )]
    #[case::fragment_on_a_legacy_blob_schema(
        table_with_a_legacy_blob_field(),
        Vec::new(),
        whole_fragment(0),
        true
    )]
    #[case::supplied_row_ids_without_a_counter(
        frag_reuse::table(2, false),
        Vec::new(),
        whole_fragment_with_row_ids(),
        true
    )]
    #[case::mem_wal_segment(
        frag_reuse::table(2, true),
        vec![mem_wal_index()],
        mem_wal_segment(),
        true
    )]
    fn test_only_a_step_its_fragments_cannot_judge_asks_for_a_snapshot_build(
        #[case] base: Manifest,
        #[case] indices: Vec<IndexMetadata>,
        #[case] step: Vec<Action>,
        #[case] needs_build: bool,
    ) {
        let config = default_build_config();
        let mut stepped = ApplyState::at_read_version(Arc::new(base), indices, &config).unwrap();
        stepped.apply_step(&UserStep::new("step", step)).unwrap();

        let dirty = stepped.unvalidated_dirty_fragments().unwrap();
        assert_eq!(
            stepped.needs_manifest_build_for_validation(&dirty),
            needs_build
        );
        if !needs_build {
            stepped.check_step_incrementally(dirty).unwrap();
        }
    }

    #[test]
    fn test_unvalidated_scope_accumulates_across_unchecked_steps_and_clears_on_validation() {
        let config = default_build_config();
        let mut working =
            ApplyState::at_read_version(Arc::new(frag_reuse::table(4, false)), Vec::new(), &config)
                .unwrap();
        let dirty_ids = |state: &ApplyState| -> Vec<u64> {
            state
                .unvalidated_dirty_fragments()
                .unwrap()
                .iter()
                .map(|fragment| fragment.id)
                .collect()
        };

        working
            .apply_step(&UserStep::new("step", whole_fragment(0)))
            .unwrap();
        assert_eq!(dirty_ids(&working), vec![4]);
        assert!(!working.unvalidated.changed_schema);
        assert!(!working.unvalidated.changed_config);

        working
            .apply_step(&UserStep::new("step", vec![rename_field_zero()]))
            .unwrap();
        assert_eq!(dirty_ids(&working), vec![4]);
        assert!(working.unvalidated.changed_schema);
        assert!(!working.unvalidated.changed_config);

        working.validate_step_invariants().unwrap();
        assert!(working.unvalidated.dirty_fragments.is_empty());
        assert!(!working.unvalidated.changed_schema);
        assert!(!working.unvalidated.changed_config);

        working
            .apply_step(&UserStep::new("step", vec![config_update()]))
            .unwrap();
        assert!(!working.unvalidated.changed_schema);
        assert!(working.unvalidated.changed_config);
        working.validate_step_invariants().unwrap();
        assert!(!working.unvalidated.changed_config);

        let assertion = AssertCellsUnchanged::try_new(
            vec![FieldRef::Committed(0)],
            vec![RowSelection {
                fragment_id: 1,
                rows: SelectedRows::Full,
            }],
        )
        .unwrap();
        working
            .apply_step(&UserStep::new(
                "step",
                vec![Action::AssertCellsUnchanged(assertion)],
            ))
            .unwrap();
        assert!(working.unvalidated.dirty_fragments.is_empty());
        assert!(!working.fragments.patched.contains_key(&1));
    }

    #[rstest]
    #[case::unchecked_schema_violation(vec![strict_field()], "non-nullable field")]
    #[case::unchecked_fragment_violation(vec![mint_fragment(0)], "non-nullable field")]
    #[case::unchecked_config_violation(
        vec![Action::ConfigUpdate(ConfigUpdate::default().with_config(UpdateMap {
            update_entries: vec![("lance.auto_cleanup.policy_interval", "1d").into()],
            replace: false,
        }))],
        "policy_interval"
    )]
    fn test_checked_step_after_unchecked_violation_rejects_pending_scope(
        #[case] bad_unchecked: Vec<Action>,
        #[case] expected_error: &str,
    ) {
        let config = default_build_config();
        let mut working =
            ApplyState::at_read_version(Arc::new(frag_reuse::table(2, true)), Vec::new(), &config)
                .unwrap();
        working
            .apply_step(&UserStep::new("unchecked", bad_unchecked))
            .unwrap();

        let direct_err = working.validate_step_invariants().unwrap_err();
        assert!(
            direct_err.to_string().contains(expected_error),
            "{direct_err}"
        );

        let valid_follow_up = vec![
            mint_fragment(1),
            write_data(Ref::Local(1), FieldRef::Committed(0)),
        ];
        let checked_err = working
            .apply_step_checked(&UserStep::new("valid_follow_up", valid_follow_up))
            .err()
            .unwrap();
        assert!(
            checked_err.to_string().contains(expected_error),
            "{checked_err}"
        );
    }

    #[test]
    fn test_later_steps_accept_newly_loaded_spilled_row_lineage() {
        let mut base = frag_reuse::table(1, true);
        let mut fragment = base.fragments[0].clone();
        fragment.last_updated_at_version_meta = Some(RowDatasetVersionMeta::Column);
        fragment.files.push(DataFile::new(
            "data/0-lineage.lance",
            vec![ROW_LAST_UPDATED_AT_VERSION_FIELD_ID],
            vec![0],
            ConcreteFileVersion::V2_0,
            None,
            None,
        ));
        base.fragments = Arc::new(vec![fragment]);

        let empty_config = default_build_config();
        let state =
            ApplyState::at_read_version(Arc::new(base.clone()), Vec::new(), &empty_config).unwrap();
        assert!(state.spilled_row_lineage_for(0).is_none());

        let after_append = state
            .apply_step_checked(&UserStep::new("append", whole_fragment(0)))
            .unwrap();
        assert!(after_append.spilled_row_lineage_for(0).is_none());

        let missing_err = after_append
            .apply_step_checked(&UserStep::new(
                "rewrite_without_lineage",
                vec![
                    tombstone_field_zero(true),
                    reencode_field_zero(true),
                    refresh_rows(0, &[1]),
                ],
            ))
            .err()
            .unwrap();
        assert!(
            missing_err
                .to_string()
                .contains("not loaded ahead of the commit"),
            "{missing_err}"
        );

        let loaded = LoadedRowLineage {
            last_updated_at: Some(Arc::new(RowDatasetVersionSequence::from_versions(&[2; 10]))),
            ..Default::default()
        };
        let lineage = Arc::new(SpilledRowLineage::from([(0, loaded)]));

        let mut bad_with_lineage = after_append.clone();
        bad_with_lineage
            .extend_spilled_row_lineage(&lineage)
            .unwrap();
        let rejected = bad_with_lineage
            .apply_step_checked(&UserStep::new(
                "bad_step_with_lineage",
                vec![mint_fragment(0)],
            ))
            .err()
            .unwrap();
        assert!(
            rejected
                .to_string()
                .contains("local fragment token 0 is minted more than once"),
            "{rejected}"
        );
        assert!(after_append.spilled_row_lineage_for(0).is_none());

        let rewrite_step = UserStep::new(
            "rewrite_with_lineage",
            vec![
                tombstone_field_zero(true),
                reencode_field_zero(true),
                refresh_rows(0, &[1, 3]),
            ],
        );
        let mut with_lineage = after_append;
        with_lineage.extend_spilled_row_lineage(&lineage).unwrap();
        let after_rewrite = with_lineage.apply_step_checked(&rewrite_step).unwrap();
        assert!(after_rewrite.spilled_row_lineage_for(0).is_some());

        let (manifest, _) = after_rewrite.build_manifest().unwrap();
        let new_version = base.version + 1;
        assert_eq!(
            last_updated_versions(&manifest.fragments[0]),
            vec![2, new_version, 2, new_version, 2, 2, 2, 2, 2, 2]
        );

        let second_rewrite = UserStep::new(
            "second_rewrite_using_persisted_lineage",
            vec![
                tombstone_field_zero(true),
                reencode_field_zero(true),
                refresh_rows(0, &[5]),
            ],
        );
        let after_second_rewrite = after_rewrite.apply_step_checked(&second_rewrite).unwrap();
        let (manifest_two, _) = after_second_rewrite.build_manifest().unwrap();
        assert_eq!(
            last_updated_versions(&manifest_two.fragments[0]),
            vec![2, new_version, 2, new_version, 2, new_version, 2, 2, 2, 2]
        );

        let mut partial_row_ids_only = after_second_rewrite.clone();
        partial_row_ids_only
            .extend_spilled_row_lineage(&Arc::new(SpilledRowLineage::from([(
                0,
                LoadedRowLineage {
                    row_ids: Some(Arc::new((0..10).into())),
                    ..Default::default()
                },
            )])))
            .unwrap();
        let merged = partial_row_ids_only.spilled_row_lineage_for(0).unwrap();
        assert!(merged.last_updated_at.is_some());
        assert!(merged.row_ids.is_some());
        assert!(merged.created_at.is_none());

        let conflicting_lineage = Arc::new(SpilledRowLineage::from([(
            0,
            LoadedRowLineage {
                last_updated_at: Some(Arc::new(RowDatasetVersionSequence::from_versions(&[9; 10]))),
                ..Default::default()
            },
        )]));
        let mut with_conflict = after_second_rewrite;
        let conflict_err = with_conflict
            .extend_spilled_row_lineage(&conflicting_lineage)
            .err()
            .unwrap();
        assert!(
            conflict_err
                .to_string()
                .contains("conflicts with lineage already loaded"),
            "{conflict_err}"
        );
    }

    #[test]
    fn test_validation_handles_build_config_and_cross_step_row_id_mismatch() {
        let mut empty_base = frag_reuse::table(1, false);
        empty_base.fragments = Arc::new(Vec::new());
        empty_base.max_fragment_id = None;

        let config_flags_on = default_build_config();
        let mut config_flags_off = default_build_config();
        config_flags_off.auto_set_feature_flags = false;

        let state = ApplyState::at_read_version(Arc::new(empty_base), Vec::new(), &config_flags_on)
            .unwrap();
        let step_one = state
            .apply_step_checked(&UserStep::new(
                "with_row_ids",
                whole_fragment_with_row_ids(),
            ))
            .unwrap();
        let mismatch_err = step_one
            .apply_step_checked(&UserStep::new("without_row_ids", whole_fragment(1)))
            .err()
            .unwrap();
        assert!(
            mismatch_err
                .to_string()
                .contains("All fragments must have row ids"),
            "{mismatch_err}"
        );

        let non_empty_base = frag_reuse::table(2, false);
        let state = ApplyState::at_read_version(
            Arc::new(non_empty_base.clone()),
            Vec::new(),
            &config_flags_off,
        )
        .unwrap();
        let stepped = state
            .apply_step_checked(&UserStep::new(
                "deferred_flag_check",
                whole_fragment_with_row_ids(),
            ))
            .unwrap();
        assert_eq!(stepped.build_manifest().unwrap().0.fragments.len(), 3);

        let mut clustering_config = default_build_config();
        clustering_config.clustering_definitions = Some(Vec::new());
        let clustering_err =
            ApplyState::at_read_version(Arc::new(non_empty_base), Vec::new(), &clustering_config)
                .err()
                .unwrap();
        assert!(
            clustering_err
                .to_string()
                .contains("clustering definitions list can only be committed with UpdateConfig"),
            "{clustering_err}"
        );
    }

    #[test]
    fn test_reset_recomputes_row_id_tally_and_preserves_overflow_check() {
        let mut base = frag_reuse::table(2, true);
        base.next_row_id = 1_000;
        let config = default_build_config();

        let mut state = ApplyState::at_read_version(Arc::new(base), Vec::new(), &config).unwrap();
        state
            .apply_step(&UserStep::new(
                "carried_row_ids",
                vec![
                    Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
                        id: Ref::Local(0),
                        physical_rows: 10,
                        row_id_meta: Some(RowIdMeta::Inline(
                            write_row_ids(&(100..104).into()).into(),
                        )),
                        last_updated_at_version_meta: None,
                        created_at_version_meta: None,
                        data_change: true,
                    })),
                    write_data(Ref::Local(0), FieldRef::Committed(0)),
                ],
            ))
            .unwrap();
        assert!(state.row_ids_need_dry_run);

        state
            .apply_step(&UserStep::new(
                "reset",
                vec![Action::ResetTable(ResetTable)],
            ))
            .unwrap();
        assert!(!state.row_ids_need_dry_run);
        assert_eq!(state.fresh_row_ids_added, 6);

        let mut overflow_base = frag_reuse::table(1, true);
        overflow_base.next_row_id = 0;
        let mut overflow_state =
            ApplyState::at_read_version(Arc::new(overflow_base), Vec::new(), &config).unwrap();
        let huge_frag = |token| {
            Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
                id: Ref::Local(token),
                physical_rows: u64::MAX,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            }))
        };
        overflow_state
            .apply_step(&UserStep::new(
                "overflow_then_reset",
                vec![huge_frag(0), huge_frag(1), Action::ResetTable(ResetTable)],
            ))
            .unwrap();
        assert!(overflow_state.row_ids_need_dry_run);
        let err = overflow_state.validate_step_invariants().unwrap_err();
        assert!(
            err.to_string().contains("row id counter is exhausted"),
            "{err}"
        );
    }
}
