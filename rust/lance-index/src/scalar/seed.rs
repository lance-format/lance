// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index seed writers — compact per-fragment summaries embedded in data files.
//!
//! A seed writer observes column values as they are written to a data file,
//! accumulates compact statistics in memory, and serializes them to a byte
//! buffer that is embedded in the data file footer as a global buffer.
//!
//! The buffer can later be read back during index updates to reconstruct index
//! statistics without re-scanning the column data.

use bytes::Bytes;

pub use lance_table::format::index_seed::{IndexSeedWriter, SEED_META_KEY_PREFIX};

/// A pre-harvested seed buffer from a single fragment's data file.
#[derive(Debug, Clone)]
pub struct FragmentSeed {
    pub fragment_id: u64,
    /// The fragment's physical row count: the rows the seed must cover
    /// completely for its statistics to stand in for a scan.
    pub physical_rows: u64,
    pub bytes: Bytes,
    /// The raw value that was stored in the data file's schema metadata under
    /// the seed key (i.e. the output of [`IndexSeedWriter::schema_metadata_value`]).
    /// Plugins can inspect this to validate that the seed is compatible with the
    /// current index configuration before consuming `bytes`.
    pub metadata_value: String,
}
