// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared test fixtures used by the action submodules' test suites.

use std::collections::HashMap;
use std::sync::Arc;

use crate::format::{
    BasePath, DataFile, DeletionFile, DeletionFileType, Fragment, IndexMetadata, Manifest,
};
use arrow_schema::DataType;
use lance_core::datatypes::{Field, Schema};
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::transaction::rewrite::{RewriteGroup, RewrittenIndex};
use crate::transaction::update_map::{UpdateMap, UpdateMapEntry};

use super::*;

pub fn sample_fragment(id: u64) -> Fragment {
    Fragment::new(id)
}

pub fn deletion_file(id: u64) -> DeletionFile {
    DeletionFile {
        read_version: 1,
        id,
        file_type: DeletionFileType::Array,
        num_deleted_rows: Some(3),
        base_id: None,
    }
}

pub fn fragment_with_deletion(id: u64, file: DeletionFile) -> Fragment {
    let mut f = Fragment::new(id);
    f.deletion_file = Some(file);
    f
}

/// Build an `UpdateDeletionVector` action; `affected_rows` of `None`
/// mirrors translation from the old format, `Some` enables row-level
/// conflict reasoning.
pub fn udv(fragment_id: u64, affected_rows: Option<&[u32]>) -> UpdateDeletionVector {
    UpdateDeletionVector {
        fragment_id,
        new_deletion_file: deletion_file(fragment_id),
        affected_rows: affected_rows.map(|r| r.iter().copied().collect()),
    }
}

pub fn empty_manifest() -> Manifest {
    use crate::format::DataStorageFormat;
    Manifest::new(
        Schema::default(),
        Arc::new(Vec::new()),
        DataStorageFormat::default(),
        std::collections::HashMap::new(),
    )
}

pub fn fragment_with_rows(id: u64, rows: usize) -> Fragment {
    let mut f = Fragment::new(id);
    f.physical_rows = Some(rows);
    f
}

pub fn rewrite_group(old_ids: &[u64], new_ids: &[u64]) -> RewriteGroup {
    RewriteGroup {
        old_fragments: old_ids.iter().copied().map(Fragment::new).collect(),
        new_fragments: new_ids.iter().copied().map(Fragment::new).collect(),
    }
}

pub fn rewritten_index(old_id: Uuid, new_id: Uuid) -> RewrittenIndex {
    RewrittenIndex {
        old_id,
        new_id,
        new_index_details: prost_types::Any::default(),
        new_index_version: 0,
        new_index_files: None,
    }
}

pub fn apply_all(actions: &[Box<dyn Action>], present_ids: &[u64]) -> Vec<Fragment> {
    let mut manifest = empty_manifest();
    manifest.fragments = Arc::new(present_ids.iter().copied().map(sample_fragment).collect());
    manifest.max_fragment_id = present_ids.iter().copied().max().map(|id| id as u32);
    for action in actions {
        action.apply(&mut manifest, &mut vec![]).unwrap();
    }
    manifest.fragments.as_ref().clone()
}

/// Build a top-level schema field with an explicit id.
pub fn field(id: i32, name: &str) -> Field {
    let mut f = Field::new_arrow(name, DataType::Int32, true).unwrap();
    f.id = id;
    f
}

/// Build a schema from `(id, name)` field pairs.
pub fn schema_with(fields: &[(i32, &str)]) -> Schema {
    Schema {
        fields: fields.iter().map(|(id, name)| field(*id, name)).collect(),
        metadata: HashMap::new(),
    }
}

/// Build a `DataFile` that covers `field_ids`.
pub fn data_file(path: &str, field_ids: Vec<i32>) -> DataFile {
    let column_indices = (0..field_ids.len() as i32).collect();
    DataFile::new(path, field_ids, column_indices, 2, 0, None, None)
}

pub fn fragment_with_files(id: u64, files: Vec<DataFile>) -> Fragment {
    let mut f = Fragment::new(id);
    f.files = files;
    f
}

pub fn metadata_update(entries: &[(&str, Option<&str>)], replace: bool) -> UpdateMap {
    UpdateMap {
        update_entries: entries
            .iter()
            .map(|(key, value)| UpdateMapEntry {
                key: (*key).to_string(),
                value: value.map(str::to_string),
            })
            .collect(),
        replace,
    }
}

pub fn bitmap(ids: &[u32]) -> RoaringBitmap {
    ids.iter().copied().collect()
}

/// Build an [`IndexMetadata`] with a fresh UUID.
pub fn index_meta(name: &str, fields: &[i32], fragment_bitmap: Option<&[u32]>) -> IndexMetadata {
    IndexMetadata {
        uuid: Uuid::new_v4(),
        fields: fields.to_vec(),
        name: name.to_string(),
        dataset_version: 1,
        fragment_bitmap: fragment_bitmap.map(bitmap),
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
    }
}

/// Manifest seeded with fragments carrying the given ids.
pub fn manifest_with_fragments(ids: &[u64]) -> Manifest {
    let mut manifest = empty_manifest();
    manifest.fragments = Arc::new(ids.iter().copied().map(sample_fragment).collect());
    manifest.max_fragment_id = ids.iter().copied().max().map(|id| id as u32);
    manifest
}

/// Build a `ReplaceFragmentColumns` action whose new file carries `fields`.
pub fn replace_columns(fragment_id: u64, path: &str, fields: &[i32]) -> ReplaceFragmentColumns {
    ReplaceFragmentColumns {
        fragment_id,
        field_ids: fields.to_vec(),
        new_data_files: vec![data_file(path, fields.to_vec())],
        updated_row_offsets: None,
        replacement: ColumnReplacement::InPlace,
    }
}

pub fn group_remap(old_ids: &[u64], new_ids: &[u64]) -> RewriteGroupRemap {
    RewriteGroupRemap {
        old_fragment_ids: old_ids.to_vec(),
        new_fragment_ids: new_ids.to_vec(),
    }
}

pub fn umap(entries: &[(&str, Option<&str>)], replace: bool) -> UpdateMap {
    UpdateMap {
        update_entries: entries.iter().map(|&pair| pair.into()).collect(),
        replace,
    }
}

pub fn base_path(id: u32, name: Option<&str>, path: &str) -> BasePath {
    BasePath::new(id, path.to_string(), name.map(str::to_string), false)
}

pub fn inline_row_ids(ids: &[u64]) -> crate::format::RowIdMeta {
    use crate::rowids::{RowIdSequence, write_row_ids};
    crate::format::RowIdMeta::Inline(write_row_ids(&RowIdSequence::from(ids)))
}

pub fn uniform_meta(rows: u64, version: u64) -> crate::format::RowDatasetVersionMeta {
    use crate::format::{RowDatasetVersionMeta, RowDatasetVersionSequence};
    RowDatasetVersionMeta::from_sequence(&RowDatasetVersionSequence::from_uniform_row_count(
        rows, version,
    ))
    .unwrap()
}

/// A fragment with inline stable row ids; `version` stamps both
/// `created_at` and `last_updated` uniformly when `Some`.
pub fn rowid_fragment(id: u64, row_ids: &[u64], version: Option<u64>) -> Fragment {
    let mut fragment = Fragment::new(id);
    fragment.physical_rows = Some(row_ids.len());
    fragment.row_id_meta = Some(inline_row_ids(row_ids));
    if let Some(version) = version {
        let meta = uniform_meta(row_ids.len() as u64, version);
        fragment.created_at_version_meta = Some(meta.clone());
        fragment.last_updated_at_version_meta = Some(meta);
    }
    fragment
}

pub fn manifest_at_version(version: u64, fragments: Vec<Fragment>) -> Manifest {
    let mut manifest = empty_manifest();
    manifest.version = version;
    manifest.fragments = Arc::new(fragments);
    manifest
}
