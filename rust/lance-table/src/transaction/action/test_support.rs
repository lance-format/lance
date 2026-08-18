// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixtures shared by the per-action test modules.

use super::{Action, CompositeOperation, UserAction};
use crate::format::{DataFile, Fragment, IndexMetadata, Manifest};
use crate::transaction::test_support::{default_build_config, sample_manifest};
use crate::transaction::{Operation, Transaction};
use arrow_schema::{DataType, Field as ArrowField};
use lance_core::Result;
use lance_core::datatypes::Field;
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
        Operation::CompositeOperation(CompositeOperation::new(vec![UserAction::new(
            "step", actions,
        )])),
        None,
    );
    transaction.build_manifest(Some(manifest), indices, "tx.txn", &default_build_config())
}

pub(super) fn added_field(name: &str) -> Field {
    Field::try_from(ArrowField::new(name, DataType::Int32, true)).unwrap()
}

/// `sample_manifest` with fragment 0 actually backed by a data file, so the
/// reference-stable actions have something committed to point at.
pub(super) fn backed_manifest() -> Manifest {
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
