// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a commit changed, read from the manifests either side of it.
//!
//! A named operation that has committed installed whatever post-image it
//! carried, and the difference between the manifest before and after says
//! exactly what that changed. Claiming that, rather than everything the
//! operation restated, is what lets a set that read before it land over a
//! fragment or schema it restated but did not change.

use super::{
    ConfigMap, Coordinate, FieldFacet, FieldRef, Footprint, Ref, UnenforcedKey, field_refs,
};
use crate::format::overlay::DataOverlayFile;
use crate::format::{Fragment, Manifest};
use crate::transaction::action::AddOverlays;
use lance_core::datatypes::{
    LANCE_UNENFORCED_CLUSTERING_KEY_POSITION, LANCE_UNENFORCED_PRIMARY_KEY,
    LANCE_UNENFORCED_PRIMARY_KEY_POSITION, Schema,
};
use std::collections::{HashMap, HashSet};

impl Footprint {
    /// What the commit from `before` to `after` changed.
    ///
    /// `moves_rows` says the fragments the commit adds hold rows it moved from
    /// the ones it removes, as a compaction's do, rather than rows that are
    /// new. Which is the case is a fact about the operation, not the manifests.
    pub(super) fn of_manifest_change(
        before: &Manifest,
        after: &Manifest,
        moves_rows: bool,
    ) -> Self {
        let mut footprint = Self::default();
        footprint.fragment_changes(before, after, moves_rows);
        footprint.schema_diff(&before.schema, &after.schema);
        footprint.map_changes(ConfigMap::Config, &before.config, &after.config);
        footprint.map_changes(
            ConfigMap::TableMetadata,
            &before.table_metadata,
            &after.table_metadata,
        );
        footprint.map_changes(
            ConfigMap::SchemaMetadata,
            &before.schema.metadata,
            &after.schema.metadata,
        );
        for (id, base) in &after.base_paths {
            if !before.base_paths.contains_key(id) {
                footprint.add_base(base);
            }
        }
        footprint
    }

    fn fragment_changes(&mut self, before: &Manifest, after: &Manifest, moves_rows: bool) {
        let after_by_id: HashMap<u64, &Fragment> = after
            .fragments
            .iter()
            .map(|fragment| (fragment.id, fragment))
            .collect();
        for fragment in before.fragments.iter() {
            match after_by_id.get(&fragment.id) {
                None => self.remove_fragment(fragment.id),
                Some(changed) => self.changes_within(&before.schema, fragment, changed),
            }
        }

        let before_ids: HashSet<u64> = before.fragments.iter().map(|f| f.id).collect();
        let added = after
            .fragments
            .iter()
            .filter(|fragment| !before_ids.contains(&fragment.id));
        for fragment in added {
            let fields = fragment
                .files
                .iter()
                .flat_map(|file| field_refs(&file.fields))
                .filter(|field| is_committed(&before.schema, *field));
            if moves_rows {
                for field in fields {
                    self.require_field_facet(field, FieldFacet::Type);
                }
            } else {
                self.add_rows();
                self.write_field_data(Ref::Local(0), fields, true);
            }
        }
    }

    /// What changed inside one fragment that is in both manifests.
    fn changes_within(&mut self, schema: &Schema, before: &Fragment, after: &Fragment) {
        let id = before.id;
        if before.deletion_file != after.deletion_file {
            self.write(Coordinate::FragmentDeletions(id));
        }
        // Added overlays restamp the rows they cover, as an `AddOverlays` does
        // without claiming the sequence: the restamp composes with a refresh in
        // either order, since each sets the rows it stamps and the later
        // version wins.
        let restamped = before.last_updated_at_version_meta != after.last_updated_at_version_meta;
        if before.row_id_meta != after.row_id_meta
            || before.created_at_version_meta != after.created_at_version_meta
            || (restamped && !adds_overlays(before, after))
        {
            self.write(Coordinate::FragmentRowVersions(id));
        }

        let (before_backing, after_backing) = (backing(before), backing(after));
        let changed: HashSet<i32> = before_backing
            .keys()
            .chain(after_backing.keys())
            .filter(|field| before_backing.get(field) != after_backing.get(field))
            .copied()
            .collect();
        if !changed.is_empty() || before.files.len() != after.files.len() {
            self.write_part(Coordinate::FragmentFiles(id));
        }
        for field in changed {
            let field = FieldRef::Committed(field);
            if is_committed(schema, field) {
                self.write_field_data(Ref::Committed(id), [field], true);
            } else {
                // A field minted by this commit: no one else can name its
                // data, but it is lost with the fragment.
                self.require_fragment(id);
            }
        }

        self.overlay_changes(id, &before.overlays, &after.overlays);
    }

    /// Overlays added to a fragment are the same edit as an
    /// [`AddOverlays`]'s. Overlays removed, or left without some field, have
    /// had that field's overlays replaced.
    fn overlay_changes(
        &mut self,
        fragment: u64,
        before: &[DataOverlayFile],
        after: &[DataOverlayFile],
    ) {
        let key = overlay_key;
        let before_by_key: HashMap<_, &DataOverlayFile> = before
            .iter()
            .map(|overlay| (key(overlay), overlay))
            .collect();
        let after_by_key: HashMap<_, &DataOverlayFile> = after
            .iter()
            .map(|overlay| (key(overlay), overlay))
            .collect();

        let added: Vec<DataOverlayFile> = after
            .iter()
            .filter(|overlay| !before_by_key.contains_key(&key(overlay)))
            .cloned()
            .collect();
        if !added.is_empty() {
            AddOverlays {
                fragment: Ref::Committed(fragment),
                overlays: added,
                data_change: true,
            }
            .footprint(self);
        }

        for overlay in before {
            let kept: HashSet<i32> = after_by_key
                .get(&key(overlay))
                .map(|kept| kept.data_file.fields.iter().copied().collect())
                .unwrap_or_default();
            for field in overlay.data_file.fields.iter() {
                if *field >= 0 && !kept.contains(field) {
                    self.write(Coordinate::FieldOverlays {
                        fragment,
                        field: *field,
                    });
                }
            }
        }
    }

    /// What changed in the schema's fields: each dropped field's existence,
    /// each changed facet, each name taken, and each field's metadata.
    fn schema_diff(&mut self, before: &Schema, after: &Schema) {
        let added: Vec<_> = after
            .fields_pre_order()
            .filter(|field| before.field_by_id(field.id).is_none())
            .collect();
        for field in before.fields_pre_order() {
            let reference = FieldRef::Committed(field.id);
            let Some(kept) = after.field_by_id(field.id) else {
                self.remove_field(reference);
                if added.iter().any(|new| new.name == field.name) {
                    self.replace_field(reference);
                }
                continue;
            };
            if kept.logical_type != field.logical_type {
                self.write_field_facet(reference, FieldFacet::Type);
            }
            if kept.nullable != field.nullable {
                self.write_field_facet(reference, FieldFacet::Nullability);
            }
            if kept.name != field.name {
                self.write_field_facet(reference, FieldFacet::Name);
                self.write(Coordinate::FieldName(kept.name.clone()));
            }
            self.map_changes(ConfigMap::Field(field.id), &field.metadata, &kept.metadata);
        }
        for field in added {
            self.write(Coordinate::FieldName(field.name.clone()));
            if !field.nullable {
                self.write(Coordinate::NewRowSchema);
            }
        }
    }

    /// Every key of one string map whose value changed, was added or was
    /// removed. In a field's metadata, a reserved key also declares one of the
    /// table-wide keys.
    fn map_changes(
        &mut self,
        map: ConfigMap,
        before: &HashMap<String, String>,
        after: &HashMap<String, String>,
    ) {
        let changed: Vec<&String> = before
            .keys()
            .chain(after.keys())
            .filter(|key| before.get(*key) != after.get(*key))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        if matches!(map, ConfigMap::Field(_)) {
            let declares = |keys: &[&str]| changed.iter().any(|key| keys.contains(&key.as_str()));
            if declares(&[
                LANCE_UNENFORCED_PRIMARY_KEY,
                LANCE_UNENFORCED_PRIMARY_KEY_POSITION,
            ]) {
                self.write(Coordinate::UnenforcedKey(UnenforcedKey::Primary));
            }
            if declares(&[LANCE_UNENFORCED_CLUSTERING_KEY_POSITION]) {
                self.write(Coordinate::UnenforcedKey(UnenforcedKey::Clustering));
            }
        }
        for key in changed {
            self.write(Coordinate::ConfigEntry {
                map: map.clone(),
                key: key.clone(),
            });
        }
    }
}

/// Which file, and which column in it, backs each live field of `fragment`.
fn backing(fragment: &Fragment) -> HashMap<i32, (&str, Option<i32>)> {
    fragment
        .files
        .iter()
        .flat_map(|file| {
            file.fields
                .iter()
                .enumerate()
                .filter(|(_, field)| **field >= 0)
                .map(|(position, field)| {
                    let column = file.column_indices.get(position).copied();
                    (*field, (file.path.as_str(), column))
                })
        })
        .collect()
}

/// What tells one overlay of a fragment from another across two manifests.
fn overlay_key(overlay: &DataOverlayFile) -> (&str, u64) {
    (overlay.data_file.path.as_str(), overlay.committed_version)
}

/// Whether `after` holds an overlay `before` did not.
fn adds_overlays(before: &Fragment, after: &Fragment) -> bool {
    let before_keys: HashSet<_> = before.overlays.iter().map(overlay_key).collect();
    after
        .overlays
        .iter()
        .any(|overlay| !before_keys.contains(&overlay_key(overlay)))
}

/// Whether `field` was in the schema the commit started from, and so is a
/// field a concurrent writer can name.
fn is_committed(schema: &Schema, field: FieldRef) -> bool {
    field
        .committed()
        .is_some_and(|id| schema.field_by_id(id).is_some())
}
