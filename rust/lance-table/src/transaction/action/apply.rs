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
//! [`ApplyState`] is that working copy, and its methods are the API the action
//! modules program against. What each action does with it lives in that action's
//! own module.

use super::{Ref, UserOperation};
use crate::format::{BasePath, Fragment, IndexMetadata, Manifest, ManifestBuildConfig};
use crate::rowids::version::build_version_meta;
use crate::transaction::Transaction;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use std::collections::{HashMap, HashSet};

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
        user_operation: &UserOperation,
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

        let new_version = current_manifest.version + 1;
        let mut state = ApplyState::new(current_manifest);
        for action in user_operation.iter_actions() {
            action.apply(&mut state)?;
        }

        let mut next_row_id = current_manifest
            .uses_stable_row_ids()
            .then_some(current_manifest.next_row_id);
        state.assign_row_ids_to_minted_fragments(&mut next_row_id, new_version)?;

        let ApplyState {
            schema,
            mut fragments,
            new_bases,
            rebound_fields,
            ..
        } = state;

        let mut indices = current_indices;
        prune_rebound_fields_from_indices(&mut indices, &rebound_fields);
        Self::retain_relevant_indices(&mut indices, &schema, &fragments);

        Self::normalize_fragments(&mut fragments)?;
        let mut manifest = self.assemble_manifest(
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

        manifest.transaction_file = Some(transaction_file_path.to_string());
        if let Some(next_row_id) = next_row_id {
            manifest.next_row_id = next_row_id;
        }

        Ok((manifest, indices))
    }
}

/// The read-version state an action set is applied against, plus the id
/// allocations made so far.
pub(super) struct ApplyState {
    schema: Schema,
    fragments: Vec<Fragment>,
    /// Base paths minted by this operation. Kept apart from the manifest's own
    /// base paths, which the manifest assembly inherits from the read version.
    new_bases: Vec<BasePath>,
    existing_base_paths: HashMap<u32, BasePath>,

    next_fragment_id: u64,
    next_field_id: i32,
    next_base_id: u32,

    /// Local token -> the id minted for it, one map per id space.
    fragment_tokens: HashMap<u32, u64>,
    field_tokens: HashMap<u32, i32>,
    base_tokens: HashMap<u32, u32>,

    /// Ids of the fragments this operation minted.
    minted_fragments: HashSet<u64>,

    /// Fields whose backing data changed, per fragment. An index covering such
    /// a field no longer describes that fragment's contents.
    rebound_fields: HashMap<u64, HashSet<i32>>,
}

impl ApplyState {
    fn new(manifest: &Manifest) -> Self {
        Self {
            schema: manifest.schema.clone(),
            fragments: manifest.fragments.as_ref().clone(),
            new_bases: Vec::new(),
            existing_base_paths: manifest.base_paths.clone(),
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
            minted_fragments: HashSet::new(),
            rebound_fields: HashMap::new(),
        }
    }

    pub(super) fn schema(&self) -> &Schema {
        &self.schema
    }

    pub(super) fn schema_mut(&mut self) -> &mut Schema {
        &mut self.schema
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
        self.minted_fragments.remove(&fragment_id);
        self.rebound_fields.remove(&fragment_id);
        true
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

    pub(super) fn mint_fragment(&mut self, token: u32) -> Result<u64> {
        if self.fragment_tokens.contains_key(&token) {
            return Err(duplicate_token_err("fragment", token));
        }
        let id = self.next_fragment_id;
        self.next_fragment_id += 1;
        self.fragment_tokens.insert(token, id);
        self.minted_fragments.insert(id);
        Ok(id)
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
    fn assign_row_ids_to_minted_fragments(
        &mut self,
        next_row_id: &mut Option<u64>,
        new_version: u64,
    ) -> Result<()> {
        let Some(next_row_id) = next_row_id.as_mut() else {
            return Ok(());
        };
        let minted_ids = std::mem::take(&mut self.minted_fragments);
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
        self.minted_fragments = minted_ids;
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

    #[test]
    fn test_an_action_set_relocates_onto_a_newer_version() {
        let actions = vec![
            Action::AddField(AddField {
                local: 0,
                parent: None,
                def: added_field("added"),
            }),
            Action::AddFragment(AddFragment {
                local: 0,
                physical_rows: 10,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(0),
                file: DataFile::new_unstarted("data/new.lance", 2, 0),
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
            Operation::UserOperation(UserOperation::new("test", vec![])),
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
