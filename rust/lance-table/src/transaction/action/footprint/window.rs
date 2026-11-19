// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The commits that landed after a pending transaction read, seen as one
//! change of state, and the check of a pending [`Footprint`] against it.

use super::{ConfigMap, Footprint, Overlap, Part, Verdict};
use crate::format::overlay::DataOverlayFile;
use crate::format::{DataFile, DeletionFile, Fragment, Manifest, RowDatasetVersionMeta, RowIdMeta};
use lance_core::datatypes::{Field, LogicalType};
use std::cell::Cell;
use std::collections::HashMap;

/// What changed between the manifest a pending transaction read and the
/// latest one, and the footprints of the transactions committed in between.
///
/// Parts are looked up by key on demand, never by diffing the manifests.
pub struct Window<'a> {
    read: &'a Manifest,
    latest: &'a Manifest,
    committed: Vec<Footprint>,
    /// How many fragments the check looked up, so a test can pin that the
    /// cost follows what the transactions touch rather than the table.
    fragment_lookups: Cell<usize>,
}

impl<'a> Window<'a> {
    /// The change from `read` to `latest`, made by the transactions whose
    /// footprints are `committed`.
    pub fn new(read: &'a Manifest, latest: &'a Manifest, committed: Vec<Footprint>) -> Self {
        Self {
            read,
            latest,
            committed,
            fragment_lookups: Cell::new(0),
        }
    }

    /// Fragment `id` in `manifest`, counted in `fragment_lookups`.
    fn fragment<'m>(&self, manifest: &'m Manifest, id: u64) -> Option<&'m Fragment> {
        self.fragment_lookups.set(self.fragment_lookups.get() + 1);
        manifest.fragment_by_id(id)
    }

    #[cfg(test)]
    pub(super) fn fragment_lookups(&self) -> usize {
        self.fragment_lookups.get()
    }

    /// Whether `part` changed, and if it changed in a way that can be rebased
    /// past, how.
    fn admits(&self, part: &Part) -> Result<Option<Overlap>, String> {
        let conflict = || Err(format!("{part:?} changed"));
        if let Part::MergedDeletions(id) = part {
            let (Some(read), Some(latest)) = (
                self.fragment(self.read, *id),
                self.fragment(self.latest, *id),
            ) else {
                return conflict();
            };
            return if read.deletion_file == latest.deletion_file {
                Ok(None)
            } else if adds_deletions(read, latest) {
                Ok(Some(Overlap::Deletions { fragment: *id }))
            } else {
                conflict()
            };
        }
        if self.changed(part) {
            conflict()
        } else {
            Ok(None)
        }
    }

    /// Whether `part` differs between the read and the latest manifest.
    fn changed(&self, part: &Part) -> bool {
        let (read, latest) = (self.read, self.latest);
        match part {
            Part::FragmentExists(id) => {
                self.fragment(read, *id).is_some() != self.fragment(latest, *id).is_some()
            }
            Part::DataFiles(id)
            | Part::Overlays(id)
            | Part::RowVersions(id)
            | Part::Deletions(id)
            | Part::MergedDeletions(id) => self.piece_changed(*id, part),
            Part::FieldData { fragment, field } => {
                self.fragment(read, *fragment).map(|f| backing(f, *field))
                    != self.fragment(latest, *fragment).map(|f| backing(f, *field))
            }
            Part::Cells { fragment, field } => {
                self.fragment(read, *fragment)
                    .map(|f| (backing(f, *field), overlays_over(f, *field)))
                    != self
                        .fragment(latest, *fragment)
                        .map(|f| (backing(f, *field), overlays_over(f, *field)))
            }
            Part::FieldExists(id) => {
                read.schema.field_by_id(*id).is_some() != latest.schema.field_by_id(*id).is_some()
            }
            Part::FieldDefinition(id) => {
                read.schema.field_by_id(*id).map(definition)
                    != latest.schema.field_by_id(*id).map(definition)
            }
            Part::NewRowSchema => {
                let was: HashMap<i32, &Field> = read
                    .schema
                    .fields_pre_order()
                    .map(|field| (field.id, field))
                    .collect();
                latest
                    .schema
                    .fields_pre_order()
                    .any(|field| match was.get(&field.id) {
                        Some(was) => was.nullable && !field.nullable,
                        None => !field.nullable || was.contains_key(&field.parent_id),
                    })
            }
            // Schema equality compares fields, their metadata included, but
            // not the schema's own metadata.
            Part::Schema => {
                read.schema != latest.schema || read.schema.metadata != latest.schema.metadata
            }
            Part::Fragments => {
                !std::sync::Arc::ptr_eq(&read.fragments, &latest.fragments)
                    && read.fragments != latest.fragments
            }
            Part::MapEntry { map, key } => {
                map_of(read, map).and_then(|entries| entries.get(key))
                    != map_of(latest, map).and_then(|entries| entries.get(key))
            }
            Part::Map(map) => map_of(read, map) != map_of(latest, map),
            Part::BaseName(name) => {
                let named = |manifest: &Manifest| {
                    manifest
                        .base_paths
                        .values()
                        .any(|base| base.name.as_ref() == Some(name))
                };
                named(read) != named(latest)
            }
            Part::BaseLocation(path) => {
                let located = |manifest: &Manifest| {
                    manifest.base_paths.values().any(|base| &base.path == path)
                };
                located(read) != located(latest)
            }
        }
    }

    /// `fragment` without what the latest schema no longer has: a drop
    /// tombstones the dropped fields in every fragment's files and overlays,
    /// and prunes those left with nothing live, which
    /// is part of the drop rather than a write of the fragment. No reader
    /// sees those columns either way.
    fn live(&self, fragment: &Fragment) -> Fragment {
        let schema = &self.latest.schema;
        let is_live = |field: &i32| *field >= 0 && schema.field_by_id(*field).is_some();
        let tombstone_dead = |fields: &[i32]| -> Vec<i32> {
            fields
                .iter()
                .map(|field| if is_live(field) { *field } else { -2 })
                .collect()
        };
        let mut live = fragment.clone();
        for file in &mut live.files {
            file.fields = tombstone_dead(&file.fields).into();
        }
        live.files.retain(|file| file.fields.iter().any(is_live));
        for overlay in &mut live.overlays {
            overlay.data_file.fields = tombstone_dead(&overlay.data_file.fields).into();
        }
        live.overlays
            .retain(|overlay| overlay.data_file.fields.iter().any(is_live));
        live
    }

    /// Whether the piece of fragment `id` that `part` names differs between
    /// the read and the latest manifest, a fragment that is there in only one
    /// of them included.
    fn piece_changed(&self, id: u64, part: &Part) -> bool {
        let live = |manifest: &Manifest| {
            self.fragment(manifest, id)
                .map(|fragment| self.live(fragment))
        };
        let (read, latest) = (live(self.read), live(self.latest));
        let (Some(read), Some(latest)) = (&read, &latest) else {
            return read.is_some() != latest.is_some();
        };
        let (read, latest) = (Pieces::of(read), Pieces::of(latest));
        match part {
            Part::DataFiles(_) => read.data_files != latest.data_files,
            Part::Overlays(_) => read.overlays != latest.overlays,
            Part::RowVersions(_) => read.row_versions != latest.row_versions,
            Part::Deletions(_) | Part::MergedDeletions(_) => read.deletions != latest.deletions,
            _ => unreachable!("{part:?} is not a piece of a fragment"),
        }
    }
}

/// A fragment split into what each of the fragment parts compares.
/// Destructured without `..`, so a field added to [`Fragment`] has to be
/// placed in one.
struct Pieces<'f> {
    data_files: (
        &'f [DataFile],
        Option<usize>,
        Option<u32>,
        &'f HashMap<String, String>,
    ),
    overlays: &'f [DataOverlayFile],
    row_versions: (
        &'f Option<RowIdMeta>,
        &'f Option<RowDatasetVersionMeta>,
        &'f Option<RowDatasetVersionMeta>,
    ),
    deletions: &'f Option<DeletionFile>,
}

impl<'f> Pieces<'f> {
    fn of(fragment: &'f Fragment) -> Self {
        let Fragment {
            id: _,
            files,
            overlays,
            deletion_file,
            row_id_meta,
            physical_rows,
            last_updated_at_version_meta,
            created_at_version_meta,
            clustering_id,
            metadata,
        } = fragment;
        Self {
            data_files: (files, *physical_rows, *clustering_id, metadata),
            overlays,
            row_versions: (
                row_id_meta,
                created_at_version_meta,
                last_updated_at_version_meta,
            ),
            deletions: deletion_file,
        }
    }
}

impl Footprint {
    /// What it takes for this transaction, pending, to land at the end of
    /// `window`.
    pub fn check(&self, window: &Window<'_>) -> Verdict {
        match self.overlaps_in(window) {
            Ok(overlaps) if overlaps.is_empty() => Verdict::Compatible,
            Ok(overlaps) => Verdict::Rebase(overlaps),
            Err(reason) => Verdict::Conflict(reason),
        }
    }

    /// The changes in `window` this transaction can be rebased past, or why
    /// it cannot land at all.
    fn overlaps_in(&self, window: &Window<'_>) -> Result<Vec<Overlap>, String> {
        if self.did.opaque {
            return Err("what this transaction touches cannot be stated".into());
        }
        let committed = &window.committed;
        for theirs in committed {
            if theirs.did.opaque {
                return Err("a concurrent transaction touched what cannot be stated".into());
            }
            if theirs.did.reset_table {
                return Err("a concurrent transaction reset the table".into());
            }
        }

        let mut overlaps = Vec::new();
        for part in &self.parts {
            overlaps.extend(window.admits(part)?);
        }
        overlaps.sort_unstable();

        for theirs in committed {
            if let Some(forbid) = self
                .forbids
                .iter()
                .find(|forbid| forbid.occurs_in(&theirs.did, &window.read.schema))
            {
                return Err(forbid.describe());
            }
        }
        Ok(overlaps)
    }
}

/// Which file, in which base, and which column in it, backs `field` in
/// `fragment`. A path is relative to its base, so the same path in another
/// base is other bytes.
fn backing(fragment: &Fragment, field: i32) -> Option<(Option<u32>, &str, Option<i32>)> {
    fragment.files.iter().find_map(|file| {
        let position = file
            .fields
            .iter()
            .position(|candidate| *candidate == field)?;
        Some((
            file.base_id,
            file.path.as_str(),
            file.column_indices.get(position).copied(),
        ))
    })
}

/// The overlays over `field` in `fragment`, by path and version.
fn overlays_over(fragment: &Fragment, field: i32) -> Vec<(&str, u64)> {
    fragment
        .overlays
        .iter()
        .filter(|overlay| overlay.data_file.fields.contains(&field))
        .map(|overlay| (overlay.data_file.path.as_str(), overlay.committed_version))
        .collect()
}

/// What a field's definition is, as [`Part::FieldDefinition`] compares it.
fn definition(field: &Field) -> (&str, bool, &LogicalType, Vec<i32>) {
    (
        field.name.as_str(),
        field.nullable,
        &field.logical_type,
        field.children.iter().map(|child| child.id).collect(),
    )
}

fn map_of<'m>(manifest: &'m Manifest, map: &ConfigMap) -> Option<&'m HashMap<String, String>> {
    match map {
        ConfigMap::Config => Some(&manifest.config),
        ConfigMap::TableMetadata => Some(&manifest.table_metadata),
        ConfigMap::SchemaMetadata => Some(&manifest.schema.metadata),
        ConfigMap::Field(id) => manifest
            .schema
            .field_by_id(*id)
            .map(|field| &field.metadata),
    }
}

/// Whether `latest`'s deletions can only have added to `read`'s, as far as
/// the manifests tell: there is a deletion file, and it counts no fewer rows.
/// The rebase that merges them reads both files to make sure.
fn adds_deletions(read: &Fragment, latest: &Fragment) -> bool {
    let deleted = |fragment: &Fragment| {
        fragment
            .deletion_file
            .as_ref()
            .map(|file| file.num_deleted_rows)
    };
    match (deleted(read), deleted(latest)) {
        (_, None) => false,
        (Some(Some(read)), Some(Some(latest))) => latest >= read,
        _ => true,
    }
}
