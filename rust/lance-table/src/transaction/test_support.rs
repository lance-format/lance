// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixtures shared between the tests of several submodules.

use crate::feature_flags::FLAG_STABLE_ROW_IDS;
use crate::format::overlay::{DataOverlayFile, OverlayCoverage};
use crate::format::{
    DataFile, DataStorageFormat, Fragment, IndexMetadata, Manifest, ManifestBuildConfig,
};
use crate::transaction::{Operation, Transaction};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use chrono::Utc;
use lance_core::datatypes::Schema as LanceSchema;
use lance_file::version::ConcreteFileVersion;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// The build config that `lance`'s `ManifestWriteConfig::default()` resolves to.
pub fn default_build_config() -> ManifestBuildConfig {
    ManifestBuildConfig {
        auto_set_feature_flags: true,
        timestamp_nanos: std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        use_stable_row_ids: false,
        use_legacy_format: None,
        storage_format: None,
        disable_transaction_file: false,
        migration_next_row_id: None,
    }
}

pub fn sample_manifest() -> Manifest {
    let schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
    Manifest::new(
        LanceSchema::try_from(&schema).unwrap(),
        Arc::new(vec![Fragment::new(0)]),
        DataStorageFormat::new(ConcreteFileVersion::V2_0),
        HashMap::new(),
    )
}

pub fn sample_index_metadata(name: &str) -> IndexMetadata {
    IndexMetadata {
        uuid: Uuid::new_v4(),
        fields: vec![0],
        covering_fields: vec![],
        name: name.to_string(),
        dataset_version: 0,
        fragment_bitmap: Some([0].into_iter().collect()),
        index_details: None,
        index_version: 1,
        created_at: Some(Utc::now()),
        base_id: None,
        files: None,
    }
}

pub fn overlay_with_field(field: i32, committed_version: u64) -> DataOverlayFile {
    DataOverlayFile {
        data_file: DataFile::new_legacy_from_fields("o.lance", vec![field], None),
        coverage: OverlayCoverage::dense(roaring::RoaringBitmap::from_iter([0u32])),
        committed_version,
    }
}

/// Existing fragments use id >= 1 to avoid collision with `Fragment::new(0)`
/// used by `sample_manifest`. New (updated) fragments use id = 10.
pub fn make_stable_row_id_manifest(fragments: Vec<Fragment>) -> Manifest {
    let schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
    let mut manifest = Manifest::new(
        LanceSchema::try_from(&schema).unwrap(),
        Arc::new(fragments),
        DataStorageFormat::new(ConcreteFileVersion::V2_0),
        HashMap::new(),
    );
    manifest.reader_feature_flags = FLAG_STABLE_ROW_IDS;
    manifest.next_row_id = 1000;
    manifest.version = 4;
    manifest
}

pub fn update_txn(new_fragments: Vec<Fragment>) -> Transaction {
    Transaction::new(
        4,
        Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![],
            new_fragments,
            fields_modified: vec![],
            compacted_sstables: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: None,
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        },
        None,
    )
}

pub fn created_at_versions(manifest: &Manifest, frag_id: u64) -> Vec<u64> {
    let frag = manifest.fragments.iter().find(|f| f.id == frag_id).unwrap();
    let seq = frag
        .created_at_version_meta
        .as_ref()
        .unwrap()
        .load_sequence()
        .unwrap();
    seq.versions().collect()
}

pub fn last_updated_at_versions(manifest: &Manifest, frag_id: u64) -> Vec<u64> {
    let frag = manifest.fragments.iter().find(|f| f.id == frag_id).unwrap();
    let seq = frag
        .last_updated_at_version_meta
        .as_ref()
        .unwrap()
        .load_sequence()
        .unwrap();
    seq.versions().collect()
}
