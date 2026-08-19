// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Applying an action set to produce the next manifest.
//!
//! Each action is applied in order against a working copy of the read-version
//! state. Minting actions allocate an id from the target's counters as they are
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

use super::{CompositeOperation, Ref};
use crate::format::{BasePath, Fragment, IndexMetadata, Manifest, ManifestBuildConfig, RowIdMeta};
use crate::rowids::read_row_ids;
use crate::rowids::version::build_version_meta;
use crate::system_index::mem_wal::{CompactedSsTable, update_mem_wal_index_compacted_sstables};
use crate::transaction::Transaction;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// The field id written into a data file's field list once the file no longer
/// backs that field. A file whose every slot is tombstoned is dropped.
pub(super) const TOMBSTONED_FIELD: i32 = -2;

impl Transaction {
    /// Build the next manifest by applying an action set.
    ///
    /// Unlike the legacy path this always requires a current manifest: an action
    /// set describes a delta, so there is nothing for it to be a delta against
    /// when the dataset does not exist yet.
    pub(in crate::transaction) fn build_manifest_from_actions(
        &self,
        composite_operation: &CompositeOperation,
        current_manifest: Option<&Manifest>,
        current_indices: Vec<IndexMetadata>,
        transaction_file_path: &str,
        config: &ManifestBuildConfig,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        let current_manifest = current_manifest.ok_or_else(|| {
            Error::invalid_input(
                "an action-based transaction describes a change to an existing dataset; \
                 it cannot create one",
            )
        })?;
        if config.use_stable_row_ids && !current_manifest.uses_stable_row_ids() {
            return Err(Error::not_supported_source(
                "Cannot enable stable row ids on existing dataset".into(),
            ));
        }

        let mut state = ApplyState::new(current_manifest, current_indices, config);
        for action in composite_operation.iter_actions() {
            action.apply(&mut state)?;
        }

        state.into_manifest(self, transaction_file_path)
    }
}

/// The read-version state an action set is applied against, plus the id
/// allocations made so far.
pub(super) struct ApplyState<'a> {
    /// The read version this delta applies to.
    current_manifest: &'a Manifest,
    /// How the manifest this delta produces is to be assembled.
    build_config: &'a ManifestBuildConfig,
    schema: Schema,
    fragments: Vec<Fragment>,
    /// Base paths minted by this operation. Kept apart from the manifest's own
    /// base paths, which the manifest assembly inherits from the read version.
    new_bases: Vec<BasePath>,
    existing_base_paths: HashMap<u32, BasePath>,
    /// The index segments, as the actions have left them so far. Index
    /// actions edit this list; the assembly then prunes whatever the data
    /// actions invalidated.
    indices: Vec<IndexMetadata>,
    /// The manifest's string maps. Unlike the schema and the fragment list,
    /// these are inherited wholesale by the manifest assembly, so an edit has
    /// to be written back over the assembled manifest.
    config: HashMap<String, String>,
    table_metadata: HashMap<String, String>,

    next_fragment_id: u64,
    next_field_id: i32,
    next_base_id: u32,

    /// Local token -> the id minted for it, one map per id space.
    fragment_tokens: HashMap<u32, u64>,
    field_tokens: HashMap<u32, i32>,
    base_tokens: HashMap<u32, u32>,

    /// Ids of the fragments this operation added, whether minted from the
    /// counter or claimed out of a reservation.
    new_fragments: HashSet<u64>,

    /// The highest fragment id this operation reserved for a later writer, if
    /// it reserved any. No fragment backs it, so the manifest assembly cannot
    /// infer it from the fragment list.
    reserved_fragment_ids: Option<u64>,

    /// How many row ids this operation reserved for a later writer. Taken off
    /// the counter after this operation's own fragments are numbered, so the
    /// reserved range is always the top `count` ids of the resulting watermark.
    reserved_row_ids: u64,

    /// Fields whose backing data changed, per fragment. An index covering such
    /// a field no longer describes that fragment's contents.
    rebound_fields: HashMap<u64, HashSet<i32>>,
}

impl<'a> ApplyState<'a> {
    fn new(
        manifest: &'a Manifest,
        indices: Vec<IndexMetadata>,
        build_config: &'a ManifestBuildConfig,
    ) -> Self {
        Self {
            current_manifest: manifest,
            build_config,
            schema: manifest.schema.clone(),
            fragments: manifest.fragments.as_ref().clone(),
            indices,
            new_bases: Vec::new(),
            existing_base_paths: manifest.base_paths.clone(),
            config: manifest.config.clone(),
            table_metadata: manifest.table_metadata.clone(),
            next_fragment_id: manifest.max_fragment_id().map(|id| id + 1).unwrap_or(0),
            next_field_id: manifest.max_field_id() + 1,
            next_base_id: manifest
                .base_paths
                .keys()
                .max()
                .map(|id| id + 1)
                .unwrap_or(1),
            fragment_tokens: HashMap::new(),
            field_tokens: HashMap::new(),
            base_tokens: HashMap::new(),
            new_fragments: HashSet::new(),
            reserved_fragment_ids: None,
            reserved_row_ids: 0,
            rebound_fields: HashMap::new(),
        }
    }

    /// Assemble the next manifest from the state the actions left behind.
    fn into_manifest(
        mut self,
        transaction: &Transaction,
        transaction_file_path: &str,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        let current_manifest = self.current_manifest;
        let config = self.build_config;
        let new_version = current_manifest.version + 1;

        let mut next_row_id = current_manifest
            .uses_stable_row_ids()
            .then_some(current_manifest.next_row_id);
        self.assign_row_ids_to_new_fragments(&mut next_row_id, new_version)?;
        self.take_row_id_reservation(&mut next_row_id)?;

        let ApplyState {
            schema,
            mut fragments,
            mut indices,
            new_bases,
            rebound_fields,
            reserved_fragment_ids,
            config: dataset_config,
            table_metadata,
            ..
        } = self;

        prune_rebound_fields_from_indices(&mut indices, &rebound_fields);
        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        Transaction::normalize_fragments(&mut fragments)?;
        let mut manifest = transaction.assemble_manifest(
            Some(current_manifest),
            schema,
            fragments,
            HashMap::new(),
            false,
            config,
        )?;

        for base in new_bases {
            manifest.base_paths.insert(base.id, base);
        }

        manifest.config = dataset_config;
        manifest.table_metadata = table_metadata;

        // A reserved id backs no fragment, so the manifest assembly cannot
        // derive it from the fragment list; raise the high-water mark to cover
        // the range so a later writer's ids are not handed out twice.
        if let Some(high_water) = reserved_fragment_ids {
            let high_water = u32::try_from(high_water).map_err(|_| {
                Error::invalid_input(format!(
                    "reserving fragment ids up to {high_water} exceeds the maximum fragment id \
                     ({})",
                    u32::MAX
                ))
            })?;
            manifest.max_fragment_id = Some(
                manifest
                    .max_fragment_id
                    .map_or(high_water, |current| current.max(high_water)),
            );
        }

        manifest.transaction_file = Some(transaction_file_path.to_string());
        if let Some(next_row_id) = next_row_id {
            manifest.next_row_id = next_row_id;
        }

        Ok((manifest, indices))
    }

    /// The version this delta applies to, which is the newest data an index
    /// segment added by this operation can have been built from.
    pub(super) fn read_version(&self) -> u64 {
        self.current_manifest.version
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
        self.indices.push(segment);
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
        self.indices.remove(position);
        Ok(())
    }

    /// Record MemWAL SSTable compaction progress, creating the MemWAL index
    /// when the dataset does not have one yet.
    pub(super) fn update_compacted_sstables(
        &mut self,
        compacted_sstables: Vec<CompactedSsTable>,
    ) -> Result<()> {
        let new_version = self.new_version();
        update_mem_wal_index_compacted_sstables(&mut self.indices, new_version, compacted_sstables)
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
        Ok(&mut self.indices[position])
    }

    /// Whether the dataset carries stable row ids, and with them the per-row
    /// version sequences.
    pub(super) fn uses_stable_row_ids(&self) -> bool {
        self.current_manifest.uses_stable_row_ids()
    }

    pub(super) fn schema(&self) -> &Schema {
        &self.schema
    }

    pub(super) fn schema_mut(&mut self) -> &mut Schema {
        &mut self.schema
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
        self.indices.clear();
        self.new_fragments.clear();
        self.rebound_fields.clear();
    }

    /// The base paths this apply can see: the read version's, plus the ones
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
        let id = self.next_fragment_id;
        self.next_fragment_id += 1;
        self.fragment_tokens.insert(token, id);
        self.new_fragments.insert(id);
        Ok(id)
    }

    /// Claim an id an earlier commit set aside with
    /// [`ReserveFragmentIds`](super::ReserveFragmentIds).
    ///
    /// Only ids already below the counter can be claimed. Anything at or above
    /// it is an id a later mint would hand out again, so a writer that guessed
    /// rather than reserved is rejected here instead of colliding later.
    fn claim_reserved_fragment_id(&mut self, id: u64) -> Result<u64> {
        if id >= self.next_fragment_id {
            return Err(Error::invalid_input(format!(
                "AddFragment adds a fragment at id {id}, which no commit has reserved: only ids \
                 below {} are reserved. Reserve a range with ReserveFragmentIds and read the \
                 committed manifest's high-water mark back before naming an id.",
                self.next_fragment_id
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

    /// Take `count` ids off the fragment counter without minting fragments for
    /// them. The reserved ids are `[next, next + count)`; a later writer names
    /// them as [`Ref::Committed`].
    pub(super) fn reserve_fragment_ids(&mut self, count: u32) {
        if count == 0 {
            return;
        }
        self.next_fragment_id += u64::from(count);
        self.reserved_fragment_ids = Some(self.next_fragment_id - 1);
    }

    /// Set `count` row ids aside for a later writer, on top of any this
    /// operation already reserved.
    pub(super) fn reserve_row_ids(&mut self, count: u64) {
        self.reserved_row_ids += count;
    }

    pub(super) fn mint_field(&mut self, token: u32) -> Result<i32> {
        if self.field_tokens.contains_key(&token) {
            return Err(duplicate_token_err("field", token));
        }
        let id = self.next_field_id;
        self.next_field_id += 1;
        self.field_tokens.insert(token, id);
        Ok(id)
    }

    pub(super) fn mint_base(&mut self, token: u32) -> Result<u32> {
        if self.base_tokens.contains_key(&token) {
            return Err(duplicate_token_err("base", token));
        }
        let id = self.next_base_id;
        self.next_base_id += 1;
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

    pub(super) fn resolve_base(&self, reference: Ref) -> Result<u32> {
        match reference {
            Ref::Committed(id) => u32::try_from(id).map_err(|_| {
                Error::invalid_input(format!("base id {id} in an action is out of range"))
            }),
            Ref::Local(token) => self
                .base_tokens
                .get(&token)
                .copied()
                .ok_or_else(|| unbound_token_err("base", token)),
        }
    }

    pub(super) fn resolve_field(&self, reference: Ref) -> Result<i32> {
        match reference {
            Ref::Committed(id) => i32::try_from(id).map_err(|_| {
                Error::invalid_input(format!("field id {id} in an action is out of range"))
            }),
            Ref::Local(token) => self
                .field_tokens
                .get(&token)
                .copied()
                .ok_or_else(|| unbound_token_err("field", token)),
        }
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

    /// As [`Self::rebind_fields`], for a change that invalidates the field
    /// across every fragment at once.
    pub(super) fn rebind_field_everywhere(&mut self, field: i32) {
        let fragment_ids = self
            .fragments
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>();
        for fragment_id in fragment_ids {
            self.rebound_fields
                .entry(fragment_id)
                .or_default()
                .insert(field);
        }
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
    fn validate_supplied_row_ids(&self, next_row_id: u64) -> Result<()> {
        for fragment in &self.fragments {
            if !self.new_fragments.contains(&fragment.id) {
                continue;
            }
            // An external sequence lives in a file, which this synchronous path
            // has no way to read; it goes unchecked.
            let Some(RowIdMeta::Inline(data)) = &fragment.row_id_meta else {
                continue;
            };
            let Some(range) = read_row_ids(data)?.row_id_range() else {
                continue;
            };
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
fn prune_rebound_fields_from_indices(
    indices: &mut [IndexMetadata],
    rebound: &HashMap<u64, HashSet<i32>>,
) {
    if rebound.is_empty() {
        return;
    }
    for index in indices.iter_mut() {
        let Some(bitmap) = index.fragment_bitmap.as_mut() else {
            continue;
        };
        for (fragment_id, fields) in rebound {
            if index.fields.iter().any(|field| fields.contains(field))
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

fn duplicate_token_err(space: &str, token: u32) -> Error {
    Error::invalid_input(format!(
        "local {space} token {token} is minted more than once in this operation; tokens must be \
         distinct within an operation"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::transaction::Operation;
    use crate::transaction::action::test_support::{added_field, apply, backed_manifest};
    use crate::transaction::action::{Action, AddDataFile, AddField, AddFragment};
    use crate::transaction::test_support::default_build_config;
    use lance_file::version::ConcreteFileVersion;

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
                field_ids: vec![Ref::Local(0)],
                data_change: true,
            }),
        ];

        // Replaying the very same actions against the manifest the first run
        // produced re-resolves both local tokens against the newer counters.
        let first = apply(&backed_manifest(), actions.clone()).unwrap();
        let second = apply(&first, actions).unwrap();

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
}
