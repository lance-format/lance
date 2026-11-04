// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixtures shared by the per-action test modules.

use super::{Action, CompositeOperation, FieldRef, Footprint, Ref, UserStep};
use crate::format::{BasePath, DataFile, Fragment, IndexMetadata, Manifest};
use crate::transaction::test_support::{default_build_config, sample_manifest};
use crate::transaction::{Operation, ReadVersionState, Transaction};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::Result;
use lance_core::datatypes::{Field, Schema};
use lance_file::version::ConcreteFileVersion;
use std::sync::Arc;

pub(super) fn apply(manifest: &Manifest, actions: Vec<Action>) -> Result<Manifest> {
    apply_with_indices(manifest, actions, Vec::new()).map(|(manifest, _)| manifest)
}

pub(super) fn apply_with_indices(
    manifest: &Manifest,
    actions: Vec<Action>,
    indices: Vec<IndexMetadata>,
) -> Result<(Manifest, Vec<IndexMetadata>)> {
    let transaction = Transaction::new(
        manifest.version,
        Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
            "step", actions,
        )])),
        None,
    );
    transaction.build_manifest(Some(manifest), indices, "tx.txn", &default_build_config())
}

/// Apply `actions` on top of `current` as an operation that read `read`,
/// the shape of a retry after a concurrent commit.
pub(super) fn apply_after_reading(
    read: &Manifest,
    current: &Manifest,
    actions: Vec<Action>,
) -> Result<Manifest> {
    Transaction::new(
        read.version,
        Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
            "step", actions,
        )])),
        None,
    )
    .build_manifest_with_read_version(
        Some(current),
        Vec::new(),
        "tx.txn",
        &default_build_config(),
        Some(ReadVersionState {
            manifest: read,
            indices: &[],
        }),
    )
    .map(|(manifest, _)| manifest)
}

/// The footprint of one action set, for pair tests. Every conflict test in the
/// action modules builds its two sides with this, so a pair reads as two lists
/// of actions and an expected answer.
pub(super) fn footprint(actions: Vec<Action>) -> Footprint {
    Footprint::from(&CompositeOperation::new(vec![UserStep::new(
        "step", actions,
    )]))
}

pub(super) fn added_field(name: &str) -> Field {
    Field::try_from(ArrowField::new(name, DataType::Int32, true)).unwrap()
}

/// A one-column table whose fragment 0 has no data files. The column is
/// nullable, so a fragment may be added without data for it, which lets tests
/// about ids and reservations skip writing any.
pub(super) fn unbacked_manifest() -> Manifest {
    let mut manifest = sample_manifest();
    manifest.schema.field_by_id_mut(0).unwrap().nullable = true;
    manifest
}

/// `unbacked_manifest` with fragment 0 actually backed by a data file, so the
/// reference-stable actions have something committed to point at.
pub(super) fn backed_manifest() -> Manifest {
    let mut manifest = unbacked_manifest();
    let mut fragment = Fragment::new(0);
    fragment.physical_rows = Some(10);
    fragment.files.push(DataFile::new(
        "data/0.lance",
        vec![0],
        vec![0],
        ConcreteFileVersion::V2_0,
        None,
        None,
    ));
    manifest.fragments = Arc::new(vec![fragment]);
    manifest
}

/// `backed_manifest` with one base path besides the dataset's own, at id 1,
/// so an action can name a committed base. A base minted on top of it gets
/// id 2.
pub(super) fn manifest_with_base() -> Manifest {
    let mut manifest = backed_manifest();
    manifest.base_paths.insert(
        1,
        BasePath::new(1, "s3://bucket/other".into(), Some("other".into()), false),
    );
    manifest
}

/// `id`, then a struct `s` holding `x` and `y`: field ids 0, 1, 2, 3. For
/// tests relating a field to the fields it is nested in or contains.
pub(super) fn nested_schema() -> Schema {
    let schema = Schema::try_from(&ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new(
            "s",
            DataType::Struct(
                vec![
                    ArrowField::new("x", DataType::Int32, true),
                    ArrowField::new("y", DataType::Int32, true),
                ]
                .into(),
            ),
            true,
        ),
    ]))
    .unwrap();
    assert_eq!(schema.field_ids(), vec![0, 1, 2, 3]);
    schema
}

/// Tombstone `field` in `fragment` and bind a new file for it, both carrying
/// `data_change`: a rewrite when true, a re-encode when false.
pub(super) fn replace_field_data(fragment: u64, field: i32, data_change: bool) -> Vec<Action> {
    let field_ids = vec![FieldRef::Committed(field)];
    vec![
        Action::TombstoneFieldData(super::TombstoneFieldData {
            fragment: Ref::Committed(fragment),
            field_ids: field_ids.clone(),
            data_change,
        }),
        Action::AddDataFile(super::AddDataFile {
            fragment: Ref::Committed(fragment),
            file: DataFile::new_unstarted("data/replaced.lance", ConcreteFileVersion::V2_0),
            field_ids,
            data_change,
            base: None,
        }),
    ]
}

/// An action minting base token 0 at a location no fixture uses.
pub(super) fn mint_base() -> Action {
    Action::AddBase(super::AddBase {
        local: 0,
        base: BasePath::new(0, "s3://bucket/minted".into(), None, false),
    })
}

/// A v2 data file with one top-level column, as a writer hands it to
/// `AddDataFile`. A v2 file with no column indices stores nothing a reader can
/// find, so it would not back the field it names.
pub(super) fn one_column_file(path: &str) -> DataFile {
    DataFile {
        column_indices: Arc::from([0]),
        ..DataFile::new_unstarted(path, ConcreteFileVersion::V2_0)
    }
}
