// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Apply config and metadata updates.

use super::apply::ApplyState;
use super::footprint::UnenforcedKey;
use super::proto::required;
use super::{ConfigMap, Coordinate, FieldRef, Footprint};
use crate::format::pb_actions;
use crate::transaction::UpdateMap;
use crate::transaction::update_map::apply_update_map;
use lance_core::datatypes::{
    Field, LANCE_UNENFORCED_CLUSTERING_KEY_POSITION, LANCE_UNENFORCED_PRIMARY_KEY,
    LANCE_UNENFORCED_PRIMARY_KEY_POSITION,
};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Apply config and metadata updates.
///
/// Each of the four string maps a manifest carries is edited by an
/// [`UpdateMap`]: a list of keys to set or delete, or a wholesale replacement.
/// Absent means "leave this map alone", which is why every field is optional.
///
/// This is reference-stable rather than a delta -- config keys and field ids are
/// stable coordinates -- so two operations editing different keys commute.
///
/// Every update is safe to build: apply checks it against the table it lands
/// on, including the reserved key rules. Start from
/// [`ConfigUpdate::default`], which changes nothing, and add the maps to edit.
///
/// ```
/// # use lance_table::transaction::UpdateMap;
/// # use lance_table::transaction::action::{ConfigUpdate, FieldRef};
/// let set_interval = UpdateMap {
///     update_entries: vec![("lance.auto_cleanup.interval", "20").into()],
///     replace: false,
/// };
/// let drop_comment = UpdateMap {
///     update_entries: vec![("comment", None).into()],
///     replace: false,
/// };
/// let update = ConfigUpdate::default()
///     .with_config(set_interval)
///     .with_field_metadata(FieldRef::Committed(0), drop_comment);
/// assert!(update.table_metadata().is_none());
/// ```
#[derive(Debug, Clone, PartialEq, DeepSizeOf, Default)]
pub struct ConfigUpdate {
    config: Option<UpdateMap>,
    table_metadata: Option<UpdateMap>,
    schema_metadata: Option<UpdateMap>,
    field_metadata: Vec<FieldMetadataUpdate>,
}

/// One field's metadata within a [`ConfigUpdate`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct FieldMetadataUpdate {
    field: FieldRef,
    updates: UpdateMap,
}

impl FieldMetadataUpdate {
    /// The field whose metadata is edited.
    pub fn field(&self) -> FieldRef {
        self.field
    }

    /// The edit to the field's metadata.
    pub fn updates(&self) -> &UpdateMap {
        &self.updates
    }
}

impl ConfigUpdate {
    /// Edit the dataset config, replacing any edit set before.
    pub fn with_config(mut self, updates: UpdateMap) -> Self {
        self.config = Some(updates);
        self
    }

    /// Edit the table metadata, replacing any edit set before.
    pub fn with_table_metadata(mut self, updates: UpdateMap) -> Self {
        self.table_metadata = Some(updates);
        self
    }

    /// Edit the schema-level metadata, replacing any edit set before.
    pub fn with_schema_metadata(mut self, updates: UpdateMap) -> Self {
        self.schema_metadata = Some(updates);
        self
    }

    /// Edit one field's metadata, after any field edits added before. `field`
    /// may be minted earlier in the same operation.
    pub fn with_field_metadata(mut self, field: FieldRef, updates: UpdateMap) -> Self {
        self.field_metadata
            .push(FieldMetadataUpdate { field, updates });
        self
    }

    /// The edit to the dataset config, or `None` to leave it alone.
    pub fn config(&self) -> Option<&UpdateMap> {
        self.config.as_ref()
    }

    /// The edit to the table metadata, or `None` to leave it alone.
    pub fn table_metadata(&self) -> Option<&UpdateMap> {
        self.table_metadata.as_ref()
    }

    /// The edit to the schema-level metadata, or `None` to leave it alone.
    pub fn schema_metadata(&self) -> Option<&UpdateMap> {
        self.schema_metadata.as_ref()
    }

    /// The per-field metadata edits, in application order.
    pub fn field_metadata(&self) -> &[FieldMetadataUpdate] {
        &self.field_metadata
    }

    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        if let Some(updates) = &self.config {
            apply_update_map(state.config_mut(), updates);
        }
        if let Some(updates) = &self.table_metadata {
            apply_update_map(state.table_metadata_mut(), updates);
        }
        if let Some(updates) = &self.schema_metadata {
            apply_update_map(&mut state.schema_mut().metadata, updates);
        }
        if self.field_metadata.is_empty() {
            return Ok(());
        }

        // The unenforced primary and clustering keys are reserved schema
        // properties: each is immutable once set, and its reserved keys cannot
        // be written with an invalid value. Capture what they were before the
        // updates land so a violation can be rejected below. This runs on every
        // apply, including a conflict rebase, so it also catches the
        // concurrent-writer race.
        let primary_key_before = unenforced_primary_key(state);
        let clustering_key_before = unenforced_clustering_key(state);
        let writes_primary_key = self.writes_any(&[
            LANCE_UNENFORCED_PRIMARY_KEY,
            LANCE_UNENFORCED_PRIMARY_KEY_POSITION,
        ]);
        let writes_clustering_key = self.writes_any(&[LANCE_UNENFORCED_CLUSTERING_KEY_POSITION]);

        for update in &self.field_metadata {
            let field_id = state.resolve_schema_field(update.field, "ConfigUpdate")?;
            let field = state
                .schema_mut()
                .field_by_id_mut(field_id)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "ConfigUpdate names field {field_id}, which does not exist"
                    ))
                })?;
            apply_update_map(&mut field.metadata, &update.updates);
            refresh_reserved_key_positions(field);
        }

        reject_reserved_key_change(
            "primary",
            &primary_key_before,
            &unenforced_primary_key(state),
            writes_primary_key,
        )?;
        reject_reserved_key_change(
            "clustering",
            &clustering_key_before,
            &unenforced_clustering_key(state),
            writes_clustering_key,
        )
    }

    /// The config and metadata maps hold no row values.
    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    /// The keys this update names, or the whole map when it replaces one. A
    /// field's metadata belongs to the field, so an update to it needs the
    /// field to still be there when it lands.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        for (map, update) in [
            (ConfigMap::Config, &self.config),
            (ConfigMap::TableMetadata, &self.table_metadata),
            (ConfigMap::SchemaMetadata, &self.schema_metadata),
        ] {
            if let Some(update) = update {
                footprint.write_map_update(map, update);
            }
        }
        for update in &self.field_metadata {
            // A field minted in this operation has no committed id, so no
            // concurrent writer can be naming it. A committed field's metadata
            // goes with the field: writing it needs the field to still be
            // there, and is lost with it if the field is dropped afterwards.
            if let Some(id) = update.field.committed() {
                footprint.write_map_update(ConfigMap::Field(id), &update.updates);
                footprint.require_field_definition(update.field);
            }
        }
        // The reserved keys are declared per field but held once per table, so
        // two writers declaring one on different fields collide on the key
        // itself, not on either field's metadata.
        if self.writes_any(&[
            LANCE_UNENFORCED_PRIMARY_KEY,
            LANCE_UNENFORCED_PRIMARY_KEY_POSITION,
        ]) {
            footprint.write(Coordinate::UnenforcedKey(UnenforcedKey::Primary));
        }
        if self.writes_any(&[LANCE_UNENFORCED_CLUSTERING_KEY_POSITION]) {
            footprint.write(Coordinate::UnenforcedKey(UnenforcedKey::Clustering));
        }
    }

    fn writes_any(&self, keys: &[&str]) -> bool {
        self.field_metadata.iter().any(|update| {
            update
                .updates
                .update_entries
                .iter()
                .any(|entry| keys.contains(&entry.key.as_str()))
        })
    }
}

fn unenforced_primary_key(state: &ApplyState) -> Vec<i32> {
    state
        .schema()
        .unenforced_primary_key()
        .iter()
        .map(|field| field.id)
        .collect()
}

fn unenforced_clustering_key(state: &ApplyState) -> Vec<i32> {
    state
        .schema()
        .unenforced_clustering_key()
        .iter()
        .map(|field| field.id)
        .collect()
}

/// A field caches its reserved-key positions alongside the metadata they are
/// parsed from, so the cache has to be rebuilt whenever the metadata changes.
fn refresh_reserved_key_positions(field: &mut Field) {
    field.unenforced_primary_key_position = field
        .metadata
        .get(LANCE_UNENFORCED_PRIMARY_KEY_POSITION)
        .and_then(|value| value.parse::<u32>().ok())
        .or_else(|| {
            field
                .metadata
                .get(LANCE_UNENFORCED_PRIMARY_KEY)
                .filter(|value| matches!(value.to_lowercase().as_str(), "true" | "1" | "yes"))
                .map(|_| 0)
        });
    field.unenforced_clustering_key_position = field
        .metadata
        .get(LANCE_UNENFORCED_CLUSTERING_KEY_POSITION)
        .and_then(|value| value.parse::<u32>().ok());
}

fn reject_reserved_key_change(
    which: &str,
    before: &[i32],
    after: &[i32],
    writes_reserved_key: bool,
) -> Result<()> {
    if !before.is_empty() {
        if writes_reserved_key || after != before {
            return Err(Error::invalid_input(format!(
                "the unenforced {which} key is a reserved key and cannot be changed once set"
            )));
        }
    } else if writes_reserved_key && after.is_empty() {
        // A reserved key was written but installed no valid key, e.g. a
        // non-marker flag value or a non-numeric position.
        return Err(Error::invalid_input(format!(
            "the unenforced {which} key is a reserved key and cannot be set to an invalid value"
        )));
    }
    Ok(())
}

impl From<&ConfigUpdate> for pb_actions::ConfigUpdate {
    fn from(value: &ConfigUpdate) -> Self {
        Self {
            config: value.config.as_ref().map(pb_actions::UpdateMap::from),
            table_metadata: value
                .table_metadata
                .as_ref()
                .map(pb_actions::UpdateMap::from),
            schema_metadata: value
                .schema_metadata
                .as_ref()
                .map(pb_actions::UpdateMap::from),
            field_metadata: value
                .field_metadata
                .iter()
                .map(|update| pb_actions::config_update::FieldMetadata {
                    field: Some(update.field.into()),
                    updates: Some(pb_actions::UpdateMap::from(&update.updates)),
                })
                .collect(),
        }
    }
}

impl TryFrom<pb_actions::ConfigUpdate> for ConfigUpdate {
    type Error = Error;

    fn try_from(message: pb_actions::ConfigUpdate) -> Result<Self> {
        let mut update = Self::default();
        if let Some(updates) = &message.config {
            update = update.with_config(updates.into());
        }
        if let Some(updates) = &message.table_metadata {
            update = update.with_table_metadata(updates.into());
        }
        if let Some(updates) = &message.schema_metadata {
            update = update.with_schema_metadata(updates.into());
        }
        for field_update in message.field_metadata {
            update = update.with_field_metadata(
                required(field_update.field, "ConfigUpdate.field_metadata.field")?.try_into()?,
                UpdateMap::from(&required(
                    field_update.updates,
                    "ConfigUpdate.field_metadata.updates",
                )?),
            );
        }
        Ok(update)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::Manifest;
    use crate::transaction::action::test_support::{
        added_field, apply, backed_manifest, footprint,
    };
    use crate::transaction::action::{Action, AddField, DropField};
    use crate::transaction::update_map::UpdateMapEntry;
    use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_V2_EXT_NAME};
    use lance_core::datatypes::Schema;
    use std::sync::Arc;

    fn merge(entries: &[(&str, Option<&str>)]) -> UpdateMap {
        UpdateMap {
            update_entries: entries.iter().map(|entry| (*entry).into()).collect(),
            replace: false,
        }
    }

    fn replace(entries: &[(&str, &str)]) -> UpdateMap {
        UpdateMap {
            update_entries: entries
                .iter()
                .map(|(key, value)| UpdateMapEntry {
                    key: (*key).into(),
                    value: Some((*value).into()),
                })
                .collect(),
            replace: true,
        }
    }

    #[test]
    fn test_config_update_merges_and_deletes() {
        let mut manifest = backed_manifest();
        manifest.config.insert("keep".into(), "yes".into());
        manifest.config.insert("drop".into(), "yes".into());

        let next = apply(
            &manifest,
            vec![Action::ConfigUpdate(
                ConfigUpdate::default().with_config(merge(&[("drop", None), ("added", Some("1"))])),
            )],
        )
        .unwrap();

        assert_eq!(next.config.get("keep"), Some(&"yes".to_string()));
        assert_eq!(next.config.get("added"), Some(&"1".to_string()));
        assert!(!next.config.contains_key("drop"));
    }

    #[test]
    fn test_config_update_replaces_a_whole_map() {
        let mut manifest = backed_manifest();
        manifest.table_metadata.insert("old".into(), "yes".into());

        let next = apply(
            &manifest,
            vec![Action::ConfigUpdate(
                ConfigUpdate::default().with_table_metadata(replace(&[("new", "1")])),
            )],
        )
        .unwrap();

        assert!(!next.table_metadata.contains_key("old"));
        assert_eq!(next.table_metadata.get("new"), Some(&"1".to_string()));
    }

    #[test]
    fn test_config_update_edits_schema_and_field_metadata() {
        let next = apply(
            &backed_manifest(),
            vec![Action::ConfigUpdate(
                ConfigUpdate::default()
                    .with_schema_metadata(merge(&[("schema", Some("yes"))]))
                    .with_field_metadata(
                        FieldRef::Committed(0),
                        merge(&[("comment", Some("the id"))]),
                    ),
            )],
        )
        .unwrap();

        assert_eq!(next.schema.metadata.get("schema"), Some(&"yes".to_string()));
        let field = next.schema.field_by_id(0).unwrap();
        assert_eq!(field.metadata.get("comment"), Some(&"the id".to_string()));
    }

    #[test]
    fn test_config_update_can_name_a_field_minted_in_the_same_operation() {
        let next = apply(
            &backed_manifest(),
            vec![
                Action::AddField(AddField::new(0, added_field("fresh"))),
                Action::ConfigUpdate(ConfigUpdate::default().with_field_metadata(
                    FieldRef::Local(0),
                    merge(&[("comment", Some("brand new"))]),
                )),
            ],
        )
        .unwrap();

        let field = next.schema.field("fresh").unwrap();
        assert_eq!(
            field.metadata.get("comment"),
            Some(&"brand new".to_string())
        );
    }

    /// `attrs: Map<Utf8, Binary>` with no fragments, and the id of its value
    /// field.
    fn map_manifest() -> (Manifest, i32) {
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        let entries = ArrowField::new(
            "entries",
            DataType::Struct(
                vec![
                    ArrowField::new("key", DataType::Utf8, false),
                    ArrowField::new("value", DataType::Binary, true),
                ]
                .into(),
            ),
            false,
        );
        let attrs = ArrowField::new("attrs", DataType::Map(Arc::new(entries), false), true);
        let mut manifest = backed_manifest();
        manifest.schema = Schema::try_from(&ArrowSchema::new(vec![attrs])).unwrap();
        manifest.fragments = Arc::new(Vec::new());
        let value = manifest.schema.field("attrs.entries.value").unwrap().id;
        (manifest, value)
    }

    fn promote_to_blob_v2(field: i32) -> Action {
        Action::ConfigUpdate(ConfigUpdate {
            field_metadata: vec![FieldMetadataUpdate {
                field: FieldRef::Committed(field),
                updates: merge(&[(ARROW_EXT_NAME_KEY, Some(BLOB_V2_EXT_NAME))]),
            }],
            ..Default::default()
        })
    }

    /// Readers cannot scan a Blob v2 field nested in a Map, so the action path
    /// must refuse to publish one, as the named metadata update does.
    #[test]
    fn test_config_update_rejects_a_blob_nested_in_a_map() {
        let (manifest, value) = map_manifest();

        let error = apply(&manifest, vec![promote_to_blob_v2(value)]).unwrap_err();

        assert!(matches!(error, Error::NotSupported { .. }), "{error:?}");
        assert!(
            error.to_string().contains("nested in Map field 'attrs'"),
            "{error}"
        );
    }

    /// A table that already has such a field can still be maintained.
    #[test]
    fn test_config_update_keeps_a_blob_already_nested_in_a_map() {
        let (mut manifest, value) = map_manifest();
        manifest
            .schema
            .mut_field_by_id(value)
            .unwrap()
            .metadata
            .insert(ARROW_EXT_NAME_KEY.into(), BLOB_V2_EXT_NAME.into());

        let next = apply(
            &manifest,
            vec![Action::ConfigUpdate(ConfigUpdate {
                schema_metadata: Some(merge(&[("note", Some("kept"))])),
                ..Default::default()
            })],
        )
        .unwrap();

        assert!(next.schema.field_by_id(value).unwrap().is_blob_v2());
    }

    #[test]
    fn test_config_update_rejects_a_missing_field() {
        let error = apply(
            &backed_manifest(),
            vec![Action::ConfigUpdate(
                ConfigUpdate::default().with_field_metadata(
                    FieldRef::Committed(7),
                    merge(&[("comment", Some("nope"))]),
                ),
            )],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("field 7"), "{error}");
    }

    #[test]
    fn test_config_update_sets_then_refuses_to_change_the_primary_key() {
        let set_key = |field: i32| {
            Action::ConfigUpdate(ConfigUpdate::default().with_field_metadata(
                FieldRef::Committed(field),
                merge(&[(LANCE_UNENFORCED_PRIMARY_KEY, Some("true"))]),
            ))
        };

        let next = apply(&backed_manifest(), vec![set_key(0)]).unwrap();
        assert_eq!(
            next.schema
                .unenforced_primary_key()
                .iter()
                .map(|field| field.id)
                .collect::<Vec<_>>(),
            vec![0]
        );

        let error = apply(&next, vec![set_key(0)]).unwrap_err();
        assert!(
            error.to_string().contains("cannot be changed once set"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_config_update_rejects_an_invalid_primary_key() {
        let error = apply(
            &backed_manifest(),
            vec![Action::ConfigUpdate(
                ConfigUpdate::default().with_field_metadata(
                    FieldRef::Committed(0),
                    merge(&[(LANCE_UNENFORCED_PRIMARY_KEY_POSITION, Some("not a number"))]),
                ),
            )],
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("cannot be set to an invalid value"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_edits_to_different_keys_commute() {
        let update = |key: &str| {
            vec![Action::ConfigUpdate(
                ConfigUpdate::default().with_config(merge(&[(key, Some("1"))])),
            )]
        };

        assert!(!footprint(update("a")).conflicts_with(&footprint(update("b"))));
        assert!(footprint(update("a")).conflicts_with(&footprint(update("a"))));
    }

    #[test]
    fn test_a_replacement_collides_with_any_edit_to_the_same_map() {
        let replaced = footprint(vec![Action::ConfigUpdate(
            ConfigUpdate::default().with_config(replace(&[("a", "1")])),
        )]);
        let merged = footprint(vec![Action::ConfigUpdate(
            ConfigUpdate::default().with_config(merge(&[("untouched-by-name", Some("1"))])),
        )]);
        let other_map = footprint(vec![Action::ConfigUpdate(
            ConfigUpdate::default().with_table_metadata(merge(&[("a", Some("1"))])),
        )]);

        assert!(replaced.conflicts_with(&merged));
        // Two clears of the same map name no key at all, and still collide.
        assert!(replaced.conflicts_with(&replaced.clone()));
        // A different map is a different coordinate space.
        assert!(!replaced.conflicts_with(&other_map));
    }

    #[test]
    fn test_dropping_a_field_collides_with_updating_its_metadata() {
        let metadata = footprint(vec![Action::ConfigUpdate(
            ConfigUpdate::default()
                .with_field_metadata(FieldRef::Committed(1), merge(&[("comment", Some("x"))])),
        )]);
        let dropped = footprint(vec![Action::DropField(DropField::new(
            FieldRef::Committed(1),
        ))]);
        let other = footprint(vec![Action::DropField(DropField::new(
            FieldRef::Committed(2),
        ))]);

        // Metadata landing after the drop names a field that is gone; the
        // drop landing after the metadata takes the metadata with the field.
        assert!(metadata.conflicts_with(&dropped));
        assert!(!dropped.conflicts_with(&metadata));
        assert!(!other.conflicts_with(&metadata));
        assert!(!metadata.conflicts_with(&other));
    }

    #[test]
    fn test_declaring_the_primary_key_on_different_fields_collides() {
        let primary_key = |field: i32| {
            footprint(vec![Action::ConfigUpdate(
                ConfigUpdate::default().with_field_metadata(
                    FieldRef::Committed(field),
                    merge(&[(LANCE_UNENFORCED_PRIMARY_KEY, Some("true"))]),
                ),
            )])
        };
        let clustering_key = footprint(vec![Action::ConfigUpdate(
            ConfigUpdate::default().with_field_metadata(
                FieldRef::Committed(3),
                merge(&[(LANCE_UNENFORCED_CLUSTERING_KEY_POSITION, Some("0"))]),
            ),
        )]);
        let comment = footprint(vec![Action::ConfigUpdate(
            ConfigUpdate::default()
                .with_field_metadata(FieldRef::Committed(3), merge(&[("comment", Some("x"))])),
        )]);

        // The key is one per table, so the fields being different is no help.
        assert!(primary_key(1).conflicts_with(&primary_key(2)));
        // A different reserved key, or plain metadata, is a different thing.
        assert!(!primary_key(1).conflicts_with(&clustering_key));
        assert!(!primary_key(1).conflicts_with(&comment));
    }
}
