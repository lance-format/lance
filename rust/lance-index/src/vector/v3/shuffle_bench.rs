// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark-only support for reopening a frozen two-file shuffle fixture.

use std::sync::Arc;

use lance_core::Result;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use serde::{Deserialize, Serialize};

use super::shuffler::{ShuffleReader, TwoFileShuffleReader};
/// The path-independent metadata needed to reopen a two-file shuffle fixture.
///
/// This is public only so the external Criterion benchmark can serialize the
/// manifest on one revision and reopen the same data from another revision.
#[doc(hidden)]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TwoFileShuffleFixtureManifest {
    pub num_partitions: usize,
    pub num_flush_groups: u64,
    pub partition_counts: Vec<u64>,
    pub total_loss: f64,
}

/// Reopen a frozen local two-file shuffle fixture.
///
/// `output_dir` is deliberately separate from the manifest so a fixture can be
/// copied without embedding a machine-specific absolute path.
#[doc(hidden)]
pub async fn open_two_file_shuffle_fixture(
    output_dir: Path,
    manifest: &TwoFileShuffleFixtureManifest,
) -> Result<Box<dyn ShuffleReader>> {
    TwoFileShuffleReader::try_new(
        Arc::new(ObjectStore::local()),
        output_dir,
        manifest.num_partitions,
        manifest.num_flush_groups,
        manifest.partition_counts.clone(),
        manifest.total_loss,
    )
    .await
}
