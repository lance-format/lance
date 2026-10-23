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
//! [`ApplyState`] is that working copy, and its methods are the API the action
//! modules program against. What each action does with it lives in that action's
//! own module.

use super::{CompositeOperation, FieldId, FieldRef, Ref};
use crate::feature_flags::inherit_sticky_feature_flags;
use crate::format::overlay::staleness::field_affects_index;
use crate::format::{BasePath, Fragment, IndexMetadata, Manifest, ManifestBuildConfig, RowIdMeta};
use crate::rowids::read_row_ids;
use crate::rowids::version::build_version_meta;
use crate::system_index::mem_wal::MEM_WAL_INDEX_NAME;
use crate::transaction::{LogicalIndexSegments, ReadVersionState, Transaction};
use lance_core::datatypes::Schema;
use lance_core::{Error, Result, is_system_column};
use roaring::RoaringTreemap;
use std::collections::{HashMap, HashSet};

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
    pub(in crate::transaction) fn build_manifest_from_actions(
        &self,
        composite_operation: &CompositeOperation,
        current_manifest: &Manifest,
        current_indices: Vec<IndexMetadata>,
        transaction_file_path: &str,
        config: &ManifestBuildConfig,
        read_version_state: Option<ReadVersionState<'_>>,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
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

        // Snapshot taken before the actions rewrite the list, so coverage can
        // be compared against what each logical index looked like going in --
        // the same point the legacy path snapshots at, before its operation
        // arm runs. Only tables with a MemWAL index maintain coverage, so
        // every other commit pays nothing.
        let mem_wal_segments_before = current_indices
            .iter()
            .any(|index| index.name == MEM_WAL_INDEX_NAME)
            .then(|| Self::logical_index_segments(&current_indices));

        let mut state = ApplyState::new(current_manifest);
        for action in composite_operation.iter_actions() {
            action.apply(&mut state)?;
        }

        state.into_manifest(
            self,
            current_indices,
            transaction_file_path,
            config,
            mem_wal_segments_before.as_ref(),
            read_version_state,
        )
    }
}

/// The state of the version being committed on top of, plus the id
/// allocations made so far.
pub(super) struct ApplyState<'a> {
    /// The version being committed on top of: the read version on a first
    /// attempt, the latest committed version on a retry.
    current_manifest: &'a Manifest,
    schema: Schema,
    fragments: Vec<Fragment>,
    /// Base paths minted by this operation. Kept apart from the manifest's own
    /// base paths, which the manifest assembly inherits from the version being
    /// committed on top of.
    new_bases: Vec<BasePath>,
    existing_base_paths: HashMap<u32, BasePath>,
    /// The manifest's string maps. Unlike the schema and the fragment list,
    /// these are inherited wholesale by the manifest assembly, so an edit has
    /// to be written back over the assembled manifest.
    config: HashMap<String, String>,
    table_metadata: HashMap<String, String>,

    /// The highest id handed out so far in each space, by the version being
    /// committed on top of or by this operation. Kept as marks rather than
    /// next ids so a counter at its maximum is representable, and the next
    /// mint fails rather than wraps. No fragment id has been handed out while
    /// the fragment mark is `None`; base ids start at 1.
    fragment_high_water: Option<u64>,
    field_high_water: i32,
    base_high_water: u32,

    /// Local token -> the id minted for it, one map per id space.
    fragment_tokens: HashMap<u32, u64>,
    field_tokens: HashMap<u32, i32>,
    base_tokens: HashMap<u32, u32>,

    /// Ids of the fragments this operation added, whether minted from the
    /// counter or claimed out of a reservation.
    new_fragments: HashSet<u64>,

    /// How many fragment ids this operation reserved for a later writer. Like
    /// [`Self::reserved_row_ids`], taken off the counter after this
    /// operation's own fragments are numbered.
    reserved_fragment_ids: u64,

    /// How many row ids this operation reserved for a later writer. Taken off
    /// the counter after this operation's own fragments are numbered, so the
    /// reserved range is always the top `count` ids of the resulting watermark.
    reserved_row_ids: u64,

    /// Fields whose backing data changed, per fragment. An index covering such
    /// a field no longer describes that fragment's contents.
    rebound_fields: HashMap<u64, HashSet<i32>>,

    /// Whether the table was reset, which discards every index outright rather
    /// than pruning fragments out of them.
    reset: bool,
}

impl<'a> ApplyState<'a> {
    fn new(manifest: &'a Manifest) -> Self {
        Self {
            current_manifest: manifest,
            schema: manifest.schema.clone(),
            fragments: manifest.fragments.as_ref().clone(),
            new_bases: Vec::new(),
            existing_base_paths: manifest.base_paths.clone(),
            config: manifest.config.clone(),
            table_metadata: manifest.table_metadata.clone(),
            fragment_high_water: manifest.max_fragment_id(),
            field_high_water: manifest.max_field_id(),
            base_high_water: manifest.base_paths.keys().max().copied().unwrap_or(0),
            fragment_tokens: HashMap::new(),
            field_tokens: HashMap::new(),
            base_tokens: HashMap::new(),
            new_fragments: HashSet::new(),
            reserved_fragment_ids: 0,
            reserved_row_ids: 0,
            rebound_fields: HashMap::new(),
            reset: false,
        }
    }

    /// Assemble the next manifest from the state the actions left behind.
    fn into_manifest(
        mut self,
        transaction: &Transaction,
        current_indices: Vec<IndexMetadata>,
        transaction_file_path: &str,
        config: &ManifestBuildConfig,
        mem_wal_segments_before: Option<&LogicalIndexSegments>,
        read_version_state: Option<ReadVersionState<'_>>,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        let current_manifest = self.current_manifest;
        let new_version = current_manifest.version + 1;

        self.check_spilled_lineage_is_bound()?;

        let mut next_row_id = current_manifest
            .uses_stable_row_ids()
            .then_some(current_manifest.next_row_id);
        self.assign_row_ids_to_new_fragments(&mut next_row_id, new_version)?;
        self.take_row_id_reservation(&mut next_row_id)?;
        self.take_fragment_id_reservation()?;

        let ApplyState {
            schema,
            mut fragments,
            new_bases,
            rebound_fields,
            reset,
            fragment_high_water,
            field_high_water,
            config: dataset_config,
            table_metadata,
            ..
        } = self;

        let mut indices = current_indices;
        if reset {
            indices.clear();
        }
        prune_rebound_fields_from_indices(&mut indices, &rebound_fields, &current_manifest.schema);
        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        Transaction::normalize_fragments(&mut fragments)?;

        // Applied once the final index list is known, so it sees exactly the
        // indices this commit publishes rather than what any one action
        // intended. Same position as on the legacy path: after the index list
        // has been pruned, before the manifest is assembled.
        if let Some(segments_before) = mem_wal_segments_before {
            Transaction::apply_mem_wal_index_coverage(
                &mut indices,
                segments_before,
                read_version_state,
                new_version,
            )?;
        }

        let mut manifest = transaction.assemble_manifest(
            Some(current_manifest),
            schema,
            fragments,
            &indices,
            HashMap::new(),
            false,
            config,
        )?;
        inherit_sticky_feature_flags(&mut manifest, current_manifest)?;

        for base in new_bases {
            manifest.base_paths.insert(base.id, base);
        }

        manifest.config = dataset_config;
        manifest.table_metadata = table_metadata;

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

        manifest.transaction_file = Some(transaction_file_path.to_string());
        if let Some(next_row_id) = next_row_id {
            manifest.next_row_id = next_row_id;
        }

        Ok((manifest, indices))
    }

    pub(super) fn schema(&self) -> &Schema {
        &self.schema
    }

    pub(super) fn schema_mut(&mut self) -> &mut Schema {
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
        &mut self.config
    }

    pub(super) fn table_metadata_mut(&mut self) -> &mut HashMap<String, String> {
        &mut self.table_metadata
    }

    pub(super) fn fragments_mut(&mut self) -> &mut [Fragment] {
        &mut self.fragments
    }

    pub(super) fn fragment_mut(&mut self, fragment_id: u64, action: &str) -> Result<&mut Fragment> {
        self.fragments
            .iter_mut()
            .find(|fragment| fragment.id == fragment_id)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "{action} targets fragment {fragment_id}, which does not exist"
                ))
            })
    }

    pub(super) fn push_fragment(&mut self, fragment: Fragment) {
        self.fragments.push(fragment);
    }

    /// Drop a fragment and forget everything recorded about it. `false` if no
    /// such fragment was present.
    pub(super) fn remove_fragment(&mut self, fragment_id: u64) -> bool {
        let before = self.fragments.len();
        self.fragments.retain(|fragment| fragment.id != fragment_id);
        if self.fragments.len() == before {
            return false;
        }
        self.new_fragments.remove(&fragment_id);
        self.rebound_fields.remove(&fragment_id);
        true
    }

    /// Empty the table: no schema, no fragments, no indices. The id counters
    /// keep going, so a field or fragment added afterwards never reuses an id an
    /// old file might still name.
    pub(super) fn reset(&mut self) {
        self.schema.fields.clear();
        self.schema.metadata.clear();
        self.fragments.clear();
        self.new_fragments.clear();
        self.rebound_fields.clear();
        self.reset = true;
    }

    /// The base paths this apply can see: the current manifest's, plus the ones
    /// earlier actions in this operation minted.
    pub(super) fn bases(&self) -> impl Iterator<Item = &BasePath> {
        self.existing_base_paths
            .values()
            .chain(self.new_bases.iter())
    }

    pub(super) fn push_base(&mut self, base: BasePath) {
        self.new_bases.push(base);
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
        self.fragment_tokens.insert(token, id);
        self.new_fragments.insert(id);
        Ok(id)
    }

    /// Claim an id an earlier commit set aside with
    /// [`ReserveFragmentIds`](super::ReserveFragmentIds).
    ///
    /// Only ids at or below the high-water mark can be claimed. Anything above
    /// it is an id a later mint would hand out again, so a writer that guessed
    /// rather than reserved is rejected here instead of colliding later.
    fn claim_reserved_fragment_id(&mut self, id: u64) -> Result<u64> {
        if self.fragment_high_water.is_none_or(|mark| id > mark) {
            return Err(Error::invalid_input(format!(
                "AddFragment adds a fragment at id {id}, which no commit has reserved: the \
                 fragment id high-water mark is {:?}. Reserve a range with ReserveFragmentIds \
                 and read the committed manifest's high-water mark back before naming an id.",
                self.fragment_high_water
            )));
        }
        if self.fragments.iter().any(|fragment| fragment.id == id) {
            return Err(Error::invalid_input(format!(
                "AddFragment adds a fragment at id {id}, which already exists"
            )));
        }
        self.new_fragments.insert(id);
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
        self.field_tokens.insert(token, id);
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
        self.base_tokens.insert(token, id);
        Ok(id)
    }

    pub(super) fn resolve_fragment(&self, reference: Ref) -> Result<u64> {
        match reference {
            Ref::Committed(id) => Ok(id),
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

    pub(super) fn resolve_field(&self, reference: FieldRef) -> Result<FieldId> {
        match reference {
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

    /// Record that these fields' data in this fragment no longer matches what
    /// an index built over them recorded.
    pub(super) fn rebind_fields(
        &mut self,
        fragment_id: u64,
        fields: impl IntoIterator<Item = i32>,
    ) {
        self.rebound_fields
            .entry(fragment_id)
            .or_default()
            .extend(fields);
    }

    /// Every row lineage sequence a fragment added here spills has to be bound
    /// by exactly one of its data files, or the sequence has no column to be
    /// read from. Checked once the operation has run, because the
    /// [`AddDataFile`](super::AddDataFile) carrying it follows the
    /// [`AddFragment`](super::AddFragment) that declares it.
    fn check_spilled_lineage_is_bound(&self) -> Result<()> {
        for fragment in &self.fragments {
            if !self.new_fragments.contains(&fragment.id) {
                continue;
            }
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
        self.validate_supplied_row_ids(*next_row_id)?;
        let minted_ids = std::mem::take(&mut self.new_fragments);
        // The manifest assembly sorts fragments by id, so partitioning them here
        // does not disturb the final order.
        let (mut minted, existing): (Vec<Fragment>, Vec<Fragment>) = self
            .fragments
            .drain(..)
            .partition(|fragment| minted_ids.contains(&fragment.id));

        Transaction::assign_row_ids(next_row_id, minted.as_mut_slice())?;
        for fragment in minted.iter_mut() {
            // An action may carry its own sequences (a squashed operation
            // does); only stamp the ones it left for apply to fill.
            let version_meta = build_version_meta(fragment, new_version);
            if fragment.last_updated_at_version_meta.is_none() {
                fragment.last_updated_at_version_meta = version_meta.clone();
            }
            if fragment.created_at_version_meta.is_none() {
                fragment.created_at_version_meta = version_meta;
            }
        }

        self.fragments = existing;
        self.fragments.extend(minted);
        self.new_fragments = minted_ids;
        Ok(())
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
    /// Two fragments of this operation carrying the same id are rejected too:
    /// a row id names one row.
    fn validate_supplied_row_ids(&self, next_row_id: u64) -> Result<()> {
        let mut supplied: Vec<(u64, RoaringTreemap)> = Vec::new();
        for fragment in &self.fragments {
            if !self.new_fragments.contains(&fragment.id) {
                continue;
            }
            // A spilled sequence lives in a file, which this synchronous path
            // has no way to read; it goes unchecked.
            let Some(RowIdMeta::Inline(data)) = &fragment.row_id_meta else {
                continue;
            };
            let sequence = read_row_ids(data)?;
            let row_ids = sequence.to_treemap();
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
            if *range.end() >= next_row_id {
                return Err(Error::invalid_input(format!(
                    "fragment {} is added carrying row id {}, which no commit has reserved: \
                     only ids below {next_row_id} are reserved. Reserve a range with \
                     ReserveRowIds and read the committed manifest's next_row_id back before \
                     supplying row ids.",
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
        if self.reserved_fragment_ids == 0 {
            return Ok(());
        }
        let high_water = match self.fragment_high_water {
            None => Some(self.reserved_fragment_ids - 1),
            Some(mark) => mark.checked_add(self.reserved_fragment_ids),
        }
        .and_then(|high_water| u32::try_from(high_water).ok())
        .ok_or_else(|| exhausted_err("fragment id", MAX_FRAGMENT_ID))?;
        self.fragment_high_water = Some(u64::from(high_water));
        Ok(())
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
/// be keyed on a child. Ancestry is read from `base_schema`, the schema of the
/// version being committed on top of, because a field this operation drops or
/// moves has already left the schema being built.
fn prune_rebound_fields_from_indices(
    indices: &mut [IndexMetadata],
    rebound: &HashMap<u64, HashSet<i32>>,
    base_schema: &Schema,
) {
    if rebound.is_empty() {
        return;
    }
    for index in indices.iter_mut() {
        let Some(bitmap) = index.fragment_bitmap.as_mut() else {
            continue;
        };
        for (fragment_id, fields) in rebound {
            if fields
                .iter()
                .any(|field| field_affects_index(*field, &index.fields, base_schema))
                && let Ok(fragment_id) = u32::try_from(*fragment_id)
            {
                bitmap.remove(fragment_id);
            }
        }
    }
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
    use crate::feature_flags::{FLAG_MIXED_DATA_FILE_VERSIONS, FLAG_NON_REUSABLE_FIELD_IDS};
    use crate::format::DataFile;
    use crate::rowids::write_row_ids;
    use crate::system_index::mem_wal::{
        CompactedSsTable, IndexCatchupProgress, MemWalIndexDetails, load_mem_wal_index_details,
        new_mem_wal_index_meta,
    };
    use crate::transaction::action::test_support::{
        added_field, apply, apply_with_indices, backed_manifest,
    };
    use crate::transaction::action::{
        Action, AddBase, AddDataFile, AddField, AddFragment, AlterField, CompositeOperation,
        ConfigUpdate, DropField, FieldMetadataUpdate, RemoveFragment, ReserveFragmentIds,
        ResetTable, SetDeletionFile, TombstoneFieldData, UserStep,
    };
    use crate::transaction::test_support::{
        default_build_config, make_stable_row_id_manifest, sample_index_metadata,
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
                    Action::TombstoneFieldData(TombstoneFieldData {
                        fragment: Ref::Committed(0),
                        field_ids: vec![FieldRef::Committed(0)],
                        data_change: true,
                    }),
                    Action::AddDataFile(AddDataFile {
                        fragment: Ref::Committed(0),
                        file: DataFile {
                            fields: Arc::from([]),
                            ..replacement
                        },
                        field_ids: vec![FieldRef::Committed(0)],
                        data_change: true,
                        base: None,
                    }),
                ],
            )]))
        } else {
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, replacement)],
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
            Action::AddField(AddField {
                local: 0,
                parent: None,
                def: added_field("added"),
            }),
            Action::AddFragment(AddFragment {
                id: Ref::Local(0),
                physical_rows: 10,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(0),
                file: DataFile::new_unstarted("data/new.lance", ConcreteFileVersion::V2_0),
                field_ids: vec![FieldRef::Local(0)],
                data_change: true,
                base: None,
            }),
        ];

        // Replaying the same actions against the manifest the first run
        // produced re-resolves both local tokens against the newer counters.
        // Only the field's name differs, since the first run took "added".
        let first = apply(&backed_manifest(), actions.clone()).unwrap();
        let mut replayed = actions;
        let Action::AddField(add_field) = &mut replayed[0] else {
            unreachable!("the first action is the AddField built above");
        };
        add_field.def.name = "added_again".into();
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
                false,
            )])),
            false,
        )]);
        let mut manifest = backed_manifest();
        manifest.schema = lance_core::datatypes::Schema::try_from(&arrow).unwrap();
        let mut fragment = manifest.fragments[0].clone();
        fragment.files[0].fields = vec![rewritten].into();
        manifest.fragments = Arc::new(vec![fragment]);

        let (_, indices) = apply_with_indices(
            &manifest,
            vec![
                Action::TombstoneFieldData(TombstoneFieldData {
                    fragment: Ref::Committed(0),
                    field_ids: vec![FieldRef::Committed(rewritten)],
                    data_change: true,
                }),
                Action::AddDataFile(AddDataFile {
                    fragment: Ref::Committed(0),
                    file: DataFile::new(
                        "data/0-new.lance",
                        Vec::<i32>::new(),
                        vec![0],
                        ConcreteFileVersion::V2_0,
                        None,
                        None,
                    ),
                    field_ids: vec![FieldRef::Committed(rewritten)],
                    data_change: true,
                    base: None,
                }),
            ],
            vec![IndexMetadata {
                fields: vec![indexed],
                ..sample_index_metadata("idx")
            }],
        )
        .unwrap();

        assert!(indices[0].fragment_bitmap.as_ref().unwrap().is_empty());
    }

    fn apply_steps(manifest: &Manifest, steps: Vec<Vec<Action>>) -> Result<Manifest> {
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
        Action::AddFragment(AddFragment {
            id: Ref::Local(token),
            physical_rows: 10,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        })
    }

    fn write_data(fragment: Ref, field: FieldRef) -> Action {
        Action::AddDataFile(AddDataFile {
            fragment,
            file: DataFile::new_unstarted("data/new.lance", ConcreteFileVersion::V2_0),
            field_ids: vec![field],
            data_change: true,
            base: None,
        })
    }

    fn mint_field(token: u32, name: &str, parent: Option<FieldRef>) -> Action {
        Action::AddField(AddField {
            local: token,
            parent,
            def: added_field(name),
        })
    }

    fn remove_fragment_zero() -> Action {
        Action::RemoveFragment(RemoveFragment {
            fragment: Ref::Committed(0),
            data_change: true,
        })
    }

    fn drop_field_zero() -> Action {
        Action::DropField(DropField {
            field: FieldRef::Committed(0),
        })
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
            Action::SetDeletionFile(SetDeletionFile {
                fragment: 0,
                deletion_file: None,
                data_change: true,
                base: None,
            }),
        ]],
        Some("SetDeletionFile targets fragment 0, which does not exist"),
    )]
    #[case::tombstone_after_remove(
        vec![vec![
            remove_fragment_zero(),
            Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(0),
                field_ids: vec![FieldRef::Committed(0)],
                data_change: true,
            }),
        ]],
        Some("TombstoneFieldData targets fragment 0, which does not exist"),
    )]
    #[case::alter_after_drop(
        vec![vec![
            drop_field_zero(),
            Action::AlterField(AlterField {
                field: FieldRef::Committed(0),
                name: Some("renamed".into()),
                nullable: None,
            }),
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
    #[case::tombstone(Action::TombstoneFieldData(TombstoneFieldData {
        fragment: Ref::Committed(0),
        field_ids: vec![FieldRef::Committed(ROW_ID_FIELD_ID)],
        data_change: true,
    }))]
    #[case::drop(Action::DropField(DropField {
        field: FieldRef::Committed(ROW_ID_FIELD_ID),
    }))]
    #[case::alter(Action::AlterField(AlterField {
        field: FieldRef::Committed(ROW_ID_FIELD_ID),
        name: Some("renamed".into()),
        nullable: None,
    }))]
    #[case::parent(mint_field(0, "child", Some(FieldRef::Committed(ROW_ID_FIELD_ID))))]
    #[case::field_metadata(Action::ConfigUpdate(ConfigUpdate {
        field_metadata: vec![FieldMetadataUpdate {
            field: FieldRef::Committed(ROW_ID_FIELD_ID),
            updates: UpdateMap {
                update_entries: vec![],
                replace: false,
            },
        }],
        ..Default::default()
    }))]
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
            Action::AddField(AddField { local: 0, parent: None, def: struct_field("s") }),
            mint_field(1, ROW_ID, Some(FieldRef::Local(0))),
        ],
        false
    )]
    #[case::rename_top_level(
        vec![Action::AlterField(AlterField {
            field: FieldRef::Committed(0),
            name: Some(ROW_ADDR.into()),
            nullable: None,
        })],
        true
    )]
    #[case::rename_nested(
        vec![
            Action::AddField(AddField { local: 0, parent: None, def: struct_field("s") }),
            mint_field(1, "child", Some(FieldRef::Local(0))),
            Action::AlterField(AlterField {
                field: FieldRef::Local(1),
                name: Some(ROW_ADDR.into()),
                nullable: None,
            }),
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

    fn config_update() -> Action {
        Action::ConfigUpdate(ConfigUpdate {
            config: Some(UpdateMap {
                update_entries: vec![("k", "v").into()],
                replace: false,
            }),
            ..Default::default()
        })
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
        Action::ReserveFragmentIds(ReserveFragmentIds { count: 1 }),
        "fragment id"
    )]
    #[case::field(field_counter_at_max(), mint_field(0, "added", None), "field id")]
    #[case::base(
        base_counter_at_max(),
        Action::AddBase(AddBase {
            local: 0,
            base: BasePath::new(0, "s3://bucket/new".into(), None, false),
        }),
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
        Action::AlterField(AlterField {
            field: FieldRef::Committed(0),
            name: Some("renamed".into()),
            nullable: None,
        })
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
}
