// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use lance_core::datatypes::Schema;
use lance_file::reader::ReaderProjection;
use lance_table::format::{DataFile, Manifest};

/// The part of opening a current-format data file that depends only on the
/// file's layout (the field ids it holds, the column each maps to and its
/// format version), the table schema and the read projection.
#[derive(Debug)]
pub(super) struct FileOpenState {
    pub(super) fields: Arc<[i32]>,
    pub(super) column_indices: Arc<[i32]>,
    pub(super) file_major_version: u32,
    pub(super) file_minor_version: u32,
    /// The table schema restricted to the fields the file holds.
    pub(super) data_file_schema: Arc<Schema>,
    /// `data_file_schema` restricted to the read projection.
    pub(super) schema_per_file: Arc<Schema>,
    pub(super) field_id_to_column_idx: Arc<BTreeMap<u32, u32>>,
    /// `None` when `schema_per_file` is empty, or when the file's footer
    /// decides which blob columns use the legacy layout (2.2 files).
    pub(super) physical: Option<PhysicalProjection>,
}

#[derive(Debug, PartialEq)]
pub(super) struct PhysicalProjection {
    pub(super) legacy_blobs: Arc<HashSet<i32>>,
    pub(super) reader_projection: ReaderProjection,
}

impl FileOpenState {
    /// Data files decoded from one manifest share interned `fields` and
    /// `column_indices` arrays, so a hit is usually a pointer comparison.
    fn has_layout(&self, data_file: &DataFile) -> bool {
        self.file_major_version == data_file.file_major_version
            && self.file_minor_version == data_file.file_minor_version
            && (Arc::ptr_eq(&self.fields, &data_file.fields) || self.fields == data_file.fields)
            && (Arc::ptr_eq(&self.column_indices, &data_file.column_indices)
                || self.column_indices == data_file.column_indices)
    }

    fn same_layout(&self, other: &Self) -> bool {
        self.file_major_version == other.file_major_version
            && self.file_minor_version == other.file_minor_version
            && self.fields == other.fields
            && self.column_indices == other.column_indices
    }
}

/// A per-scan cache of the projection-derived state for opening data files.
///
/// Every fragment of a scan otherwise rebuilds the per-file schema, the
/// field-to-column map and the reader projection for each of its data files,
/// although files sharing a layout produce identical state.
///
/// Entries apply only to opens against the same `Arc<Manifest>` and the same
/// projection `Arc<Schema>`, compared by address; the cache holds both, so
/// neither address can be reused while it is alive.
#[derive(Debug)]
pub struct FragmentOpenCache {
    manifest: Arc<Manifest>,
    projection: Arc<Schema>,
    entries: Mutex<Vec<Arc<FileOpenState>>>,
}

impl FragmentOpenCache {
    /// A scan rarely sees more than a few layouts; the cap only bounds a
    /// pathological schema-evolution history, whose extra layouts are
    /// computed uncached.
    pub(super) const MAX_ENTRIES: usize = 16;

    /// Create an empty cache for opens of `manifest`'s files with `projection`.
    pub fn new(manifest: Arc<Manifest>, projection: Arc<Schema>) -> Self {
        Self {
            manifest,
            projection,
            entries: Mutex::new(Vec::new()),
        }
    }

    pub(super) fn applies_to(&self, manifest: &Arc<Manifest>, projection: &Schema) -> bool {
        Arc::ptr_eq(&self.manifest, manifest) && std::ptr::eq(self.projection.as_ref(), projection)
    }

    pub(super) fn get(&self, data_file: &DataFile) -> Option<Arc<FileOpenState>> {
        let entries = self.entries.lock().unwrap();
        entries
            .iter()
            .find(|state| state.has_layout(data_file))
            .cloned()
    }

    /// Returns the existing entry when a concurrent open inserted the same
    /// layout first, so parallel misses keep one entry per layout.
    pub(super) fn insert(&self, state: Arc<FileOpenState>) -> Arc<FileOpenState> {
        let mut entries = self.entries.lock().unwrap();
        if let Some(existing) = entries.iter().find(|existing| existing.same_layout(&state)) {
            return existing.clone();
        }
        if entries.len() < Self::MAX_ENTRIES {
            entries.push(state.clone());
        }
        state
    }

    /// Number of layouts cached.
    #[doc(hidden)]
    pub fn num_entries(&self) -> usize {
        self.entries.lock().unwrap().len()
    }
}
