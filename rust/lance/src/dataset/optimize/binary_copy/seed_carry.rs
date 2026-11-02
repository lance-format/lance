// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Carrying zone write seeds through binary-copy compaction.
//!
//! Binary copy moves page bytes verbatim and regenerates the output footer,
//! so input seeds (extra global buffers) would otherwise be lost. Each input
//! file's seeds are validated here, per output file and per column: when every
//! contributing input carries a compatible, complete seed for a column, the
//! zone records are concatenated in copy order into the output file's seed;
//! otherwise (a missing seed, a version this build cannot interpret, or a
//! version 0 seed) that output column carries no seed and the data still
//! binary copies. Nothing is rescanned and no statistics are recomputed.

use std::collections::HashMap;

use bytes::Bytes;
use lance_core::Error;
use lance_file::reader::CachedFileMetadata;
use lance_file::versions::data_file_columns;
use lance_index::scalar::seed::SEED_META_KEY_PREFIX;
use lance_index::scalar::zonemap::{
    CarriedSeed, MergeableSeed, merge_seeds_for_binary_copy, parse_seed_for_binary_copy,
};
use lance_io::scheduler::FileScheduler;
use lance_table::format::DataFile;

use crate::Dataset;
use crate::Result;
use crate::datatypes::Schema;

/// One input column's recognized seed.
pub enum SourceColumnSeed {
    /// Validated and ready for merging. `bytes` is the original buffer,
    /// carried verbatim when one source covers an entire output file.
    Supported { seed: MergeableSeed, bytes: Bytes },
    /// A seed version this build cannot interpret: the column copies
    /// without a seed.
    UnsupportedVersion(String),
}

/// The `rows_per_zone` an output column's locator declares: the current
/// index specification for the column when one exists, else the writer
/// default. It is write-time configuration; zones are located by length.
pub fn output_rows_per_zone(index_rows_per_zone: Option<u64>) -> u64 {
    index_rows_per_zone.unwrap_or(lance_index::scalar::zonemap::ROWS_PER_ZONE_DEFAULT)
}

/// The configured `rows_per_zone` of each column's current usable ZoneMap
/// index, by field id.
pub async fn index_rows_per_zone(dataset: &Dataset) -> Result<HashMap<i32, u64>> {
    use crate::index::scalar::fetch_index_details;
    use crate::index::{index_is_usable, load_all_indices};
    use lance_index::pbold::ZoneMapIndexDetails;

    let mut rows_per_zone = HashMap::new();
    for index in load_all_indices(dataset)
        .await?
        .iter()
        .filter(|index| index_is_usable(index))
    {
        let Some(field_id) = index.keyed_field() else {
            continue;
        };
        let Some(field) = dataset
            .schema()
            .fields
            .iter()
            .find(|field| field.id == field_id)
        else {
            continue;
        };
        let Ok(details) = fetch_index_details(dataset, &field.name, index).await else {
            continue;
        };
        if let Ok(zone_map) = details.to_msg::<ZoneMapIndexDetails>()
            && let Some(configured) = zone_map.rows_per_zone
        {
            rows_per_zone.insert(field_id, configured);
        }
    }
    Ok(rows_per_zone)
}

/// The validated seeds of one input file, keyed by field id. An input with no
/// seeds contributes an empty map, which clears every column for the outputs
/// it feeds.
pub struct SourceSeeds {
    seeds: HashMap<i32, SourceColumnSeed>,
}

/// Read and validate the extra global buffers of one binary-copy input.
///
/// Every extra buffer (beyond the schema descriptor at index 0) must be
/// claimed by exactly one `lance.seed.<column>` schema metadata entry whose
/// locator parses, resolves to a stored field, and whose buffer is a zone
/// map seed: either of a supported version covering exactly the file's
/// physical rows, or of a version this build does not interpret (recognized,
/// carried as [`SourceColumnSeed::UnsupportedVersion`]). Anything else is an
/// error: the caller then refuses binary copy for the task, preserving the
/// existing protection against unknown global buffers (a key prefix alone
/// does not prove the content is a zone map seed).
pub async fn extract_source_seeds(
    file_meta: &CachedFileMetadata,
    file_scheduler: &FileScheduler,
    data_file: &DataFile,
) -> Result<SourceSeeds> {
    let file_schema = &file_meta.file_schema;
    let num_rows = file_meta.num_rows;
    let extra_buffers = file_meta.file_buffers.len().saturating_sub(1);

    let mut seeds = HashMap::new();
    let mut claimed_buffers: HashMap<u32, String> = HashMap::new();
    let mut recognized = 0usize;
    let (stored_ids, stored_columns) = data_file_columns(file_meta.version, file_schema);
    if stored_columns.len() != file_meta.column_metadatas.len() {
        return Err(Error::invalid_input(format!(
            "data file '{}': stored columns cannot be mapped to its column metadata",
            data_file.path
        )));
    }
    for (key, value) in file_schema.metadata.iter() {
        let Some(stored_name) = key.strip_prefix(SEED_META_KEY_PREFIX) else {
            continue;
        };
        // The locator is `<buffer_index>:<rows_per_zone>`; the buffer's own
        // version metadata decides which schema rules apply.
        let parsed = value.split_once(':').and_then(|(index, rows_per_zone)| {
            index
                .parse::<u32>()
                .ok()
                .zip(rows_per_zone.parse::<u64>().ok())
        });
        let Some((buf_index, locator_rows_per_zone)) = parsed else {
            return Err(Error::invalid_input(format!(
                "data file '{}': seed metadata '{}' has an unparsable locator '{}'",
                data_file.path, key, value
            )));
        };
        let Some(buffer) = (buf_index as usize)
            .checked_sub(1)
            .and_then(|_| file_meta.file_buffers.get(buf_index as usize))
        else {
            return Err(Error::invalid_input(format!(
                "data file '{}': seed '{}' references global buffer {} outside 1..{}",
                data_file.path,
                key,
                buf_index,
                file_meta.file_buffers.len()
            )));
        };
        if let Some(previous) = claimed_buffers.insert(buf_index, key.clone()) {
            return Err(Error::invalid_input(format!(
                "data file '{}': seeds '{}' and '{}' claim the same global buffer {}",
                data_file.path, previous, key, buf_index
            )));
        }
        // Seed keys keep their write-time names. Resolve that stored field
        // first, then translate its physical column to the current field id.
        let Some(stored_field) = file_schema
            .fields
            .iter()
            .find(|field| field.name == stored_name)
        else {
            return Err(Error::invalid_input(format!(
                "data file '{}': seed '{}' names no stored field",
                data_file.path, key
            )));
        };

        let bytes = file_scheduler
            .submit_single(buffer.position..buffer.position + buffer.size, 0)
            .await?;
        let seed = match parse_seed_for_binary_copy(&bytes, locator_rows_per_zone)? {
            CarriedSeed::Supported(seed) => {
                if seed.total_rows() != num_rows {
                    return Err(Error::invalid_input(format!(
                        "data file '{}': seed '{}' covers {} rows but the file holds {}",
                        data_file.path,
                        key,
                        seed.total_rows(),
                        num_rows
                    )));
                }
                SourceColumnSeed::Supported { seed, bytes }
            }
            CarriedSeed::UnsupportedVersion(version) => {
                SourceColumnSeed::UnsupportedVersion(version)
            }
        };
        recognized += 1;
        // An imported file may remap field ids in `DataFile.fields` while its
        // footer keeps the original ids, so the destination identity comes
        // from the physical column, as the seed harvest resolves it. A stored
        // column no destination field maps to (dropped) carries no seed.
        let Some((_, stored_column)) = stored_ids
            .iter()
            .zip(&stored_columns)
            .find(|(id, _)| **id == stored_field.id)
        else {
            return Err(Error::invalid_input(format!(
                "data file '{}': seed '{}' names a field without a physical column",
                data_file.path, key
            )));
        };
        if let Some((field_id, _)) = data_file
            .fields
            .iter()
            .zip(data_file.column_indices.iter())
            .find(|(_, column_index)| **column_index >= 0 && **column_index == *stored_column)
        {
            seeds.insert(*field_id, seed);
        }
    }

    if recognized != extra_buffers {
        return Err(Error::invalid_input(format!(
            "data file '{}': {} extra global buffers but {} recognized seeds",
            data_file.path, extra_buffers, recognized
        )));
    }
    Ok(SourceSeeds { seeds })
}

/// A finished seed for one output column, ready to embed in the footer.
pub struct OutputSeed {
    pub column_name: String,
    pub rows_per_zone: u64,
    pub bytes: Bytes,
}

/// Accumulates the inputs feeding the current output file, in copy order, and
/// produces the output's per-column seeds. State is reset for each output.
#[derive(Default)]
pub struct OutputSeedAccumulator {
    sources: Vec<SourceSeeds>,
}

impl OutputSeedAccumulator {
    pub fn push(&mut self, source: SourceSeeds) {
        self.sources.push(source);
    }

    /// Merge the accumulated inputs' seeds and reset for the next output.
    ///
    /// A column is kept only when every input carries a version 1 seed of one
    /// value type; zones are located by length, so differing collection
    /// granularities concatenate freely and the output locator declares the
    /// column's current `rows_per_zone` from `index_rows_per_zone` (or the
    /// writer default). Each column is decided on its own: an input without
    /// a seed, with a version this build cannot interpret, or with a version
    /// 0 seed drops that column (no conversion or relabeling), and the data
    /// still copies.
    pub fn finish(
        &mut self,
        current_schema: &Schema,
        index_rows_per_zone: &HashMap<i32, u64>,
    ) -> Result<Vec<OutputSeed>> {
        let sources = std::mem::take(&mut self.sources);
        let Some(first) = sources.first() else {
            return Ok(Vec::new());
        };

        let mut output = Vec::new();
        for field_id in first.seeds.keys() {
            // The current schema names the output column; a field the schema
            // no longer lists top-level has no consumer for its seed.
            let Some(current_field) = current_schema
                .fields
                .iter()
                .find(|field| field.id == *field_id)
            else {
                continue;
            };
            let Some(column_seeds) = sources
                .iter()
                .map(|source| source.seeds.get(field_id))
                .collect::<Option<Vec<_>>>()
            else {
                // Some input lacks this column's seed: partial statistics must
                // not impersonate complete coverage of the output.
                log::debug!(
                    "Not carrying seed for column {}: an input has none",
                    current_field.name
                );
                continue;
            };
            let mut supported = Vec::with_capacity(column_seeds.len());
            let mut unsupported = None;
            for column_seed in &column_seeds {
                match column_seed {
                    SourceColumnSeed::Supported { seed, bytes } => supported.push((seed, bytes)),
                    SourceColumnSeed::UnsupportedVersion(version) => {
                        unsupported.get_or_insert(version.as_str());
                    }
                }
            }
            if let Some(version) = unsupported {
                log::debug!(
                    "Not carrying seed for column {}: seed version '{}' is not supported by \
                     this build and seed versions are not merged across versions",
                    current_field.name,
                    version
                );
                continue;
            }
            if supported.iter().any(|(seed, _)| !seed.is_version_1()) {
                // Version 0 sources are never carried: converting them would
                // fabricate sign counts, and their fixed-stride layout does
                // not concatenate.
                log::debug!(
                    "Not carrying seed for column {}: a version 0 seed is not carried; the \
                     next index update rescans these rows",
                    current_field.name
                );
                continue;
            }

            let bytes = if supported.len() == 1 {
                // One input covers the whole output: its seed is already the
                // output's seed, byte for byte.
                supported[0].1.clone()
            } else {
                let merged: Vec<&MergeableSeed> = supported.iter().map(|(seed, _)| *seed).collect();
                match merge_seeds_for_binary_copy(&merged)? {
                    Some(bytes) => bytes,
                    None => continue,
                }
            };
            output.push(OutputSeed {
                column_name: current_field.name.clone(),
                rows_per_zone: output_rows_per_zone(index_rows_per_zone.get(field_id).copied()),
                bytes,
            });
        }
        // Deterministic footer metadata order.
        output.sort_by(|a, b| a.column_name.cmp(&b.column_name));
        Ok(output)
    }
}
