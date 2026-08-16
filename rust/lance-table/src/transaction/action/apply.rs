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

use super::{
    Action, AddBase, AddDataFile, AddField, AddFragment, AlterField, Ref, RemoveFragment,
    SetDeletionFile, TombstoneFieldData, UserOperation,
};
use crate::format::{BasePath, Fragment, IndexMetadata, Manifest, ManifestBuildConfig};
use crate::rowids::version::build_version_meta;
use crate::transaction::Transaction;
use lance_core::datatypes::{Field, Schema};
use lance_core::{Error, Result};
use std::collections::{HashMap, HashSet};

/// The field id written into a data file's field list once the file no longer
/// backs that field. A file whose every slot is tombstoned is dropped.
const TOMBSTONED_FIELD: i32 = -2;

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
            state.apply(action)?;
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
struct ApplyState {
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

    fn apply(&mut self, action: &Action) -> Result<()> {
        match action {
            Action::AddFragment(action) => self.add_fragment(action),
            Action::AddDataFile(action) => self.add_data_file(action),
            Action::AddField(action) => self.add_field(action),
            Action::AddBase(action) => self.add_base(action),
            Action::TombstoneFieldData(action) => self.tombstone_field_data(action),
            Action::RemoveFragment(action) => self.remove_fragment(action),
            Action::SetDeletionFile(action) => self.set_deletion_file(action),
            Action::AlterField(action) => self.alter_field(action),
        }
    }

    fn add_fragment(&mut self, action: &AddFragment) -> Result<()> {
        if self.fragment_tokens.contains_key(&action.local) {
            return Err(duplicate_token_err("fragment", action.local));
        }
        let id = self.next_fragment_id;
        self.next_fragment_id += 1;
        self.fragment_tokens.insert(action.local, id);
        self.minted_fragments.insert(id);

        self.fragments.push(Fragment {
            id,
            files: Vec::new(),
            overlays: Vec::new(),
            deletion_file: None,
            row_id_meta: action.row_id_meta.clone(),
            physical_rows: Some(action.physical_rows as usize),
            last_updated_at_version_meta: action.last_updated_at_version_meta.clone(),
            created_at_version_meta: action.created_at_version_meta.clone(),
        });
        Ok(())
    }

    fn add_data_file(&mut self, action: &AddDataFile) -> Result<()> {
        let fragment_id = self.resolve_fragment(action.fragment)?;
        let field_ids = action
            .field_ids
            .iter()
            .map(|field| self.resolve_field(*field))
            .collect::<Result<Vec<_>>>()?;

        let mut file = action.file.clone();
        if !file.column_indices.is_empty() && file.column_indices.len() != field_ids.len() {
            return Err(Error::invalid_input(format!(
                "AddDataFile for fragment {fragment_id} lists {} field ids but the file has {} \
                 columns",
                field_ids.len(),
                file.column_indices.len()
            )));
        }
        file.fields = field_ids.into();

        let fragment = self
            .fragments
            .iter_mut()
            .find(|fragment| fragment.id == fragment_id)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "AddDataFile targets fragment {fragment_id}, which does not exist"
                ))
            })?;
        fragment.files.push(file);
        Ok(())
    }

    fn add_field(&mut self, action: &AddField) -> Result<()> {
        let id = self.next_field_id;
        self.next_field_id += 1;
        if self.field_tokens.contains_key(&action.local) {
            return Err(duplicate_token_err("field", action.local));
        }
        self.field_tokens.insert(action.local, id);

        let parent_id = action
            .parent
            .map(|parent| self.resolve_field(parent))
            .transpose()?;

        // The definition's own id, parent id, and children are ignored: the
        // minted id and the parent reference carry that structure, and each
        // child column arrives as its own action.
        let field = Field {
            id,
            parent_id: parent_id.unwrap_or(-1),
            children: Vec::new(),
            ..action.def.clone()
        };

        match parent_id {
            None => self.schema.fields.push(field),
            Some(parent_id) => {
                let parent = self.schema.field_by_id_mut(parent_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "AddField names parent field {parent_id}, which does not exist"
                    ))
                })?;
                parent.children.push(field);
            }
        }
        Ok(())
    }

    fn add_base(&mut self, action: &AddBase) -> Result<()> {
        let id = self.next_base_id;
        self.next_base_id += 1;
        if self.base_tokens.contains_key(&action.local) {
            return Err(duplicate_token_err("base", action.local));
        }
        self.base_tokens.insert(action.local, id);

        let conflicting = self
            .existing_base_paths
            .values()
            .chain(self.new_bases.iter())
            .find(|base| base.name == action.base.name || base.path == action.base.path);
        if let Some(conflicting) = conflicting {
            return Err(Error::invalid_input(format!(
                "Conflict detected: Base path with name '{:?}' or path '{}' already exists. \
                 Existing: name='{:?}', path='{}'",
                action.base.name, action.base.path, conflicting.name, conflicting.path
            )));
        }

        let mut base = action.base.clone();
        base.id = id;
        self.new_bases.push(base);
        Ok(())
    }

    fn tombstone_field_data(&mut self, action: &TombstoneFieldData) -> Result<()> {
        let fragment_id = self.resolve_fragment(action.fragment)?;
        let fragment = fragment_mut(&mut self.fragments, fragment_id, "TombstoneFieldData")?;

        for &field_id in &action.field_ids {
            let mut found = false;
            for file in fragment.files.iter_mut() {
                let Some(position) = file.fields.iter().position(|id| *id == field_id) else {
                    continue;
                };
                let mut fields = file.fields.to_vec();
                fields[position] = TOMBSTONED_FIELD;
                file.fields = fields.into();
                found = true;
            }
            if !found {
                return Err(Error::invalid_input(format!(
                    "TombstoneFieldData names field {field_id}, which no data file in fragment \
                     {fragment_id} backs"
                )));
            }
        }

        // New values for these fields supersede any overlay still shadowing
        // them, so the drop is not silently masked by stale overlay cells.
        let overlaid: Vec<u32> = action
            .field_ids
            .iter()
            .filter_map(|id| u32::try_from(*id).ok())
            .collect();
        crate::format::overlay::tombstone_overlay_fields(&mut fragment.overlays, &overlaid);

        self.rebound_fields
            .entry(fragment_id)
            .or_default()
            .extend(action.field_ids.iter().copied());
        Ok(())
    }

    fn remove_fragment(&mut self, action: &RemoveFragment) -> Result<()> {
        let fragment_id = self.resolve_fragment(action.fragment)?;
        let before = self.fragments.len();
        self.fragments.retain(|fragment| fragment.id != fragment_id);
        if self.fragments.len() == before {
            return Err(Error::invalid_input(format!(
                "RemoveFragment targets fragment {fragment_id}, which does not exist"
            )));
        }
        self.minted_fragments.remove(&fragment_id);
        self.rebound_fields.remove(&fragment_id);
        Ok(())
    }

    fn set_deletion_file(&mut self, action: &SetDeletionFile) -> Result<()> {
        let fragment = fragment_mut(&mut self.fragments, action.fragment, "SetDeletionFile")?;
        fragment.deletion_file = action.deletion_file.clone();
        Ok(())
    }

    fn alter_field(&mut self, action: &AlterField) -> Result<()> {
        let field = self.schema.field_by_id_mut(action.field).ok_or_else(|| {
            Error::invalid_input(format!(
                "AlterField names field {}, which does not exist",
                action.field
            ))
        })?;
        if let Some(name) = &action.name {
            field.name.clone_from(name);
        }
        if let Some(nullable) = action.nullable {
            field.nullable = nullable;
        }
        if let Some(logical_type) = &action.logical_type {
            field.logical_type = logical_type.as_str().into();
            // The cast leaves any index on the field describing the old type.
            // The data rewrite itself is separate actions; this only records
            // that every fragment's view of the field changed.
            for fragment in &self.fragments {
                self.rebound_fields
                    .entry(fragment.id)
                    .or_default()
                    .insert(action.field);
            }
        }
        Ok(())
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

    fn resolve_fragment(&self, reference: Ref) -> Result<u64> {
        match reference {
            Ref::Committed(id) => Ok(id),
            Ref::Local(token) => self
                .fragment_tokens
                .get(&token)
                .copied()
                .ok_or_else(|| unbound_token_err("fragment", token)),
        }
    }

    fn resolve_field(&self, reference: Ref) -> Result<i32> {
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
}

fn fragment_mut<'a>(
    fragments: &'a mut [Fragment],
    fragment_id: u64,
    action: &str,
) -> Result<&'a mut Fragment> {
    fragments
        .iter_mut()
        .find(|fragment| fragment.id == fragment_id)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "{action} targets fragment {fragment_id}, which does not exist"
            ))
        })
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
    use crate::format::{DataFile, DeletionFile, DeletionFileType};
    use crate::transaction::Operation;
    use crate::transaction::action::UserAction;
    use crate::transaction::test_support::{
        default_build_config, sample_index_metadata, sample_manifest,
    };
    use arrow_schema::{DataType, Field as ArrowField};
    use std::sync::Arc;

    fn apply(manifest: &Manifest, actions: Vec<Action>) -> Result<Manifest> {
        apply_with_indices(manifest, actions, Vec::new()).map(|(manifest, _)| manifest)
    }

    fn apply_with_indices(
        manifest: &Manifest,
        actions: Vec<Action>,
        indices: Vec<IndexMetadata>,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        let transaction = Transaction::new(
            manifest.version,
            Operation::UserOperation(UserOperation::new(
                "test",
                vec![UserAction::new("step", actions)],
            )),
            None,
        );
        transaction.build_manifest(Some(manifest), indices, "tx.txn", &default_build_config())
    }

    fn added_field(name: &str) -> Field {
        Field::try_from(ArrowField::new(name, DataType::Int32, true)).unwrap()
    }

    /// `sample_manifest` with fragment 0 actually backed by a data file, so the
    /// reference-stable actions have something committed to point at.
    fn backed_manifest() -> Manifest {
        let mut manifest = sample_manifest();
        let mut fragment = Fragment::new(0);
        fragment.physical_rows = Some(10);
        fragment.files.push(DataFile::new(
            "data/0.lance",
            vec![0],
            vec![0],
            2,
            0,
            None,
            None,
        ));
        manifest.fragments = Arc::new(vec![fragment]);
        manifest
    }

    #[test]
    fn test_tombstone_field_data_drops_the_backing_file() {
        let manifest = backed_manifest();
        let next = apply(
            &manifest,
            vec![Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(0),
                field_ids: vec![0],
                data_change: true,
            })],
        )
        .unwrap();

        // The file backed only field 0, so tombstoning it leaves nothing behind.
        assert!(next.fragments[0].files.is_empty());
    }

    #[test]
    fn test_tombstone_field_data_keeps_a_file_with_a_live_field() {
        let mut manifest = backed_manifest();
        let mut fragment = manifest.fragments[0].clone();
        fragment.files[0] = DataFile::new("data/0.lance", vec![0, 1], vec![0, 1], 2, 0, None, None);
        manifest.fragments = Arc::new(vec![fragment]);

        let next = apply(
            &manifest,
            vec![Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(0),
                field_ids: vec![0],
                data_change: true,
            })],
        )
        .unwrap();

        assert_eq!(
            next.fragments[0].files[0].fields.as_ref(),
            &[TOMBSTONED_FIELD, 1]
        );
    }

    #[test]
    fn test_tombstone_field_data_prunes_the_fragment_from_covering_indices() {
        let manifest = backed_manifest();
        let (_, indices) = apply_with_indices(
            &manifest,
            vec![Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(0),
                field_ids: vec![0],
                data_change: true,
            })],
            vec![sample_index_metadata("idx")],
        )
        .unwrap();

        // The index covers field 0, whose data in fragment 0 is now gone.
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().is_empty());
    }

    #[test]
    fn test_tombstone_field_data_rejects_a_field_no_file_backs() {
        let manifest = backed_manifest();
        let error = apply(
            &manifest,
            vec![Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(0),
                field_ids: vec![7],
                data_change: true,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("field 7"), "{error}");
    }

    #[test]
    fn test_remove_fragment() {
        let manifest = backed_manifest();
        let next = apply(
            &manifest,
            vec![Action::RemoveFragment(RemoveFragment {
                fragment: Ref::Committed(0),
                data_change: true,
            })],
        )
        .unwrap();

        assert!(next.fragments.is_empty());
    }

    #[test]
    fn test_remove_fragment_can_drop_one_minted_in_the_same_operation() {
        let manifest = backed_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddFragment(AddFragment {
                    local: 0,
                    physical_rows: 10,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                }),
                Action::RemoveFragment(RemoveFragment {
                    fragment: Ref::Local(0),
                    data_change: true,
                }),
            ],
        )
        .unwrap();

        assert_eq!(
            next.fragments.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec![0]
        );
    }

    #[test]
    fn test_remove_fragment_rejects_a_missing_fragment() {
        let manifest = backed_manifest();
        let error = apply(
            &manifest,
            vec![Action::RemoveFragment(RemoveFragment {
                fragment: Ref::Committed(7),
                data_change: true,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("fragment 7"), "{error}");
    }

    #[test]
    fn test_set_deletion_file_sets_and_clears() {
        let manifest = backed_manifest();
        let deletion_file = DeletionFile {
            read_version: manifest.version,
            id: 3,
            file_type: DeletionFileType::Array,
            num_deleted_rows: Some(2),
            base_id: None,
        };
        let next = apply(
            &manifest,
            vec![Action::SetDeletionFile(SetDeletionFile {
                fragment: 0,
                deletion_file: Some(deletion_file.clone()),
                data_change: true,
            })],
        )
        .unwrap();
        assert_eq!(next.fragments[0].deletion_file, Some(deletion_file));

        // An absent deletion file is a request to clear it, not a no-op.
        let cleared = apply(
            &next,
            vec![Action::SetDeletionFile(SetDeletionFile {
                fragment: 0,
                deletion_file: None,
                data_change: true,
            })],
        )
        .unwrap();
        assert_eq!(cleared.fragments[0].deletion_file, None);
    }

    #[test]
    fn test_set_deletion_file_rejects_a_missing_fragment() {
        let manifest = backed_manifest();
        let error = apply(
            &manifest,
            vec![Action::SetDeletionFile(SetDeletionFile {
                fragment: 7,
                deletion_file: None,
                data_change: true,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
    }

    #[test]
    fn test_alter_field_renames_without_touching_indices() {
        let manifest = backed_manifest();
        let (next, indices) = apply_with_indices(
            &manifest,
            vec![Action::AlterField(AlterField {
                field: 0,
                name: Some("renamed".into()),
                logical_type: None,
                nullable: Some(true),
            })],
            vec![sample_index_metadata("idx")],
        )
        .unwrap();

        let field = next.schema.field_by_id(0).unwrap();
        assert_eq!(field.name, "renamed");
        assert!(field.nullable);
        // A rename does not change the values the index recorded.
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().contains(0));
    }

    #[test]
    fn test_alter_field_retype_prunes_covering_indices() {
        let manifest = backed_manifest();
        let (next, indices) = apply_with_indices(
            &manifest,
            vec![Action::AlterField(AlterField {
                field: 0,
                name: None,
                logical_type: Some("int64".into()),
                nullable: None,
            })],
            vec![sample_index_metadata("idx")],
        )
        .unwrap();

        assert_eq!(
            next.schema.field_by_id(0).unwrap().logical_type.to_string(),
            "int64"
        );
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().is_empty());
    }

    #[test]
    fn test_alter_field_rejects_a_missing_field() {
        let manifest = backed_manifest();
        let error = apply(
            &manifest,
            vec![Action::AlterField(AlterField {
                field: 7,
                name: Some("nope".into()),
                ..Default::default()
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("field 7"), "{error}");
    }

    #[test]
    fn test_add_fragment_and_data_file_mint_ids() {
        let manifest = sample_manifest();
        let next = apply(
            &manifest,
            vec![
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
                    field_ids: vec![Ref::Committed(0)],
                    data_change: true,
                }),
            ],
        )
        .unwrap();

        // sample_manifest already holds fragment 0, so the mint lands on 1.
        assert_eq!(
            next.fragments.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec![0, 1]
        );
        let minted = next.fragments.iter().find(|f| f.id == 1).unwrap();
        assert_eq!(minted.physical_rows, Some(10));
        assert_eq!(minted.files.len(), 1);
        // The file's field list is stamped in from the action's refs.
        assert_eq!(minted.files[0].fields.as_ref(), &[0]);
        assert_eq!(next.max_fragment_id(), Some(1));
    }

    #[test]
    fn test_two_add_fields_mint_distinct_ids() {
        let manifest = sample_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("a"),
                }),
                Action::AddField(AddField {
                    local: 1,
                    parent: None,
                    def: added_field("b"),
                }),
            ],
        )
        .unwrap();

        let ids = next
            .schema
            .fields
            .iter()
            .map(|f| (f.name.as_str(), f.id))
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![("id", 0), ("a", 1), ("b", 2)]);
    }

    #[test]
    fn test_add_field_then_add_its_data_file() {
        // The add-column shape: mint the field, then write the file that backs
        // it, naming the field by the token the mint has not resolved yet.
        let manifest = sample_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddField(AddField {
                    local: 7,
                    parent: None,
                    def: added_field("added"),
                }),
                Action::AddDataFile(AddDataFile {
                    fragment: Ref::Committed(0),
                    file: DataFile::new_unstarted("data/added.lance", 2, 0),
                    field_ids: vec![Ref::Local(7)],
                    data_change: true,
                }),
            ],
        )
        .unwrap();

        let field_id = next.schema.field("added").unwrap().id;
        assert_eq!(field_id, 1);
        let fragment = next.fragments.iter().find(|f| f.id == 0).unwrap();
        assert_eq!(fragment.files.last().unwrap().fields.as_ref(), &[field_id]);
    }

    #[test]
    fn test_add_field_under_a_parent() {
        let manifest = sample_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: Field::try_from(ArrowField::new(
                        "nested",
                        DataType::Struct(Default::default()),
                        true,
                    ))
                    .unwrap(),
                }),
                Action::AddField(AddField {
                    local: 1,
                    parent: Some(Ref::Local(0)),
                    def: added_field("child"),
                }),
            ],
        )
        .unwrap();

        let parent = next.schema.field("nested").unwrap();
        assert_eq!(parent.children.len(), 1);
        assert_eq!(parent.children[0].name, "child");
        assert_eq!(parent.children[0].parent_id, parent.id);
    }

    #[test]
    fn test_add_base_mints_an_id_and_rejects_duplicates() {
        let manifest = sample_manifest();
        let next = apply(
            &manifest,
            vec![Action::AddBase(AddBase {
                local: 0,
                base: BasePath::new(0, "s3://bucket/a".into(), Some("a".into()), false),
            })],
        )
        .unwrap();
        assert_eq!(next.base_paths.len(), 1);
        assert_eq!(next.base_paths[&1].path, "s3://bucket/a");

        let error = apply(
            &next,
            vec![Action::AddBase(AddBase {
                local: 0,
                base: BasePath::new(0, "s3://bucket/a".into(), Some("other".into()), false),
            })],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("already exists"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_unbound_local_token_is_rejected() {
        let manifest = sample_manifest();
        let error = apply(
            &manifest,
            vec![Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(3),
                file: DataFile::new_unstarted("data/x.lance", 2, 0),
                field_ids: vec![Ref::Committed(0)],
                data_change: true,
            })],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("local fragment token 3"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_duplicate_local_token_is_rejected() {
        let manifest = sample_manifest();
        let error = apply(
            &manifest,
            vec![
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("a"),
                }),
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("b"),
                }),
            ],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("minted more than once"),
            "unexpected error: {error}"
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
