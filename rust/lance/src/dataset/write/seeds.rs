// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Write seed collection shared by every writer that produces a data file.
//!
//! A [`SeedCollector`] decides which columns of a data file get a seed, feeds
//! each written batch to the seed writers, and embeds the finished seeds in the
//! file before it is closed. Writers that create whole fragments and writers
//! that add or replace a column inside an existing fragment all go through it,
//! so a seeded column keeps its seed in whichever data file serves it.
//!
//! Seed formats that readers must opt into declare dataset feature flags
//! ([`IndexSeedWriter::required_feature_flags`]). The collector embeds such a
//! seed only when the writer's snapshot already carries the flags, and the
//! commit path sets them for every index that requests such a seed
//! ([`apply_seed_feature_flags`]), so no manifest references a seed its flags
//! do not announce.

use std::collections::{HashMap, HashSet};

use arrow_array::RecordBatch;
use lance_core::Result;
use lance_core::datatypes::Schema;
use lance_file::version::ConcreteFileVersion;
use lance_index::pbold;
use lance_index::scalar::seed::IndexSeedWriter;
use lance_index::scalar::zonemap::ZoneMapSeedWriter;
use lance_table::format::{IndexMetadata, Manifest};

use super::GenericWriter;
use crate::Dataset;
use crate::index::scalar::zonemap::stored_null_bitmap_policy;
use crate::index::scalar::{IndexDetails, fetch_index_details};
use crate::index::{index_is_usable, load_all_indices};

/// The unresolved same-column conflict is a configuration fact, so it is
/// reported once per process and at debug level afterwards.
static SEED_CONFLICT_WARNED: std::sync::Once = std::sync::Once::new();

/// Collects write seeds for the columns of one data file.
#[derive(Debug, Default)]
pub struct SeedCollector {
    writers: Vec<Box<dyn IndexSeedWriter>>,
}

impl SeedCollector {
    /// A collector that writes no seeds.
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Build the seed writers for the top-level columns of `write_schema`.
    ///
    /// A column gets a seed writer when an existing, usable index on it asks
    /// for one through its plugin and the dataset already carries the feature
    /// flags that seed format requires. `consult_indices` is false for writes
    /// that discard the dataset's indices (overwrite), where seeds have no
    /// consumer.
    pub async fn for_write(
        version: ConcreteFileVersion,
        dataset: Option<&Dataset>,
        write_schema: &Schema,
        consult_indices: bool,
    ) -> Result<Self> {
        // Legacy files have no global buffers to hold a seed.
        if version == ConcreteFileVersion::V1 || !consult_indices {
            return Ok(Self::disabled());
        }
        let Some(dataset) = dataset else {
            return Ok(Self::disabled());
        };

        let indices = load_all_indices(dataset).await?;
        let manifest = dataset.manifest.as_ref();
        let writers = seed_writers_for(dataset, &indices, write_schema, SeedWriterPurpose::Collect)
            .await?
            .into_iter()
            .filter(|writer| {
                let required = writer.required_feature_flags();
                let activated = manifest.reader_feature_flags & required == required
                    && manifest.writer_feature_flags & required == required;
                if !activated {
                    // Readers could not be told about this seed yet; the next
                    // commit activates the flags and later writes embed it.
                    log::debug!(
                        "skipping write seed for column {}: dataset version {} does not \
                         carry feature flags {:#x}",
                        writer.column_name(),
                        manifest.version,
                        required
                    );
                }
                activated
            })
            .collect();
        Ok(Self { writers })
    }

    /// Number of columns that will receive a seed.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.writers.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.writers.is_empty()
    }

    /// Feed one written batch to every seed writer whose column it contains.
    pub fn observe(&mut self, batch: &RecordBatch) -> Result<()> {
        for writer in self.writers.iter_mut() {
            if let Some(column) = batch.column_by_name(writer.column_name()) {
                writer.observe_batch(column)?;
            }
        }
        Ok(())
    }

    /// Embed the finished seeds in `writer` before its `finish()` and reset
    /// the seed writers for the next data file.
    pub async fn flush(&mut self, writer: &mut dyn GenericWriter) -> Result<()> {
        for seed_writer in self.writers.iter_mut() {
            if let Some(bytes) = seed_writer.finish()? {
                let buf_index = writer.add_global_buffer(bytes).await?;
                let key = seed_writer.schema_metadata_key();
                let value = seed_writer.schema_metadata_value(buf_index);
                writer.add_schema_metadata(key, value);
            }
        }
        Ok(())
    }
}

/// The seed writers that the usable indices in `indices` request for the
/// top-level columns of `schema`.
///
/// `dataset` resolves index details that are not inline in the metadata; it
/// only has to share its index store with `indices`, which may belong to a
/// manifest that is still being built.
/// Why seed writers are being selected.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SeedWriterPurpose {
    /// Embedding seeds in a file being written: each writer is fully
    /// configured, which for a zone map includes the null-position policy
    /// stored in its index file.
    Collect,
    /// Computing the feature flags the seed formats require. The flags do
    /// not depend on the null-position policy, so no index file is read.
    FeatureFlags,
}

async fn seed_writers_for(
    dataset: &Dataset,
    indices: &[IndexMetadata],
    schema: &Schema,
    purpose: SeedWriterPurpose,
) -> Result<Vec<Box<dyn IndexSeedWriter>>> {
    let mut writers: Vec<Box<dyn IndexSeedWriter>> = Vec::new();
    // Zone map requirements per column: every ZoneMap index on the column
    // that collects seeds, as (index name, rows_per_zone, null positions).
    let mut zone_map_columns: HashMap<i32, Vec<(String, u64, bool)>> = HashMap::new();
    let mut other_columns: HashSet<i32> = HashSet::new();
    for index in indices.iter().filter(|index| index_is_usable(index)) {
        // A covered index lists its carried columns in `fields` too; the
        // seed writer keys on the single keyed column. System indices
        // commit no fields at all, so this also skips them.
        let Some(field_id) = index.keyed_field() else {
            continue;
        };
        // Seeds are collected for top-level columns of the file being
        // written. Batches are observed by column name, which is the field
        // path for a top-level field.
        let Some(field) = schema.fields.iter().find(|f| f.id == field_id) else {
            continue;
        };
        let Ok(index_details) = fetch_index_details(dataset, &field.name, index).await else {
            continue;
        };
        if let Ok(zone_map) = index_details.to_msg::<pbold::ZoneMapIndexDetails>() {
            // Seeds follow the index's configuration: its zone size from
            // the details and its null-position policy from the index
            // file (one cached footer read; see `stored_null_bitmap_policy`).
            if !zone_map.use_seeds.unwrap_or(false) {
                continue;
            }
            let Some(rows_per_zone) = zone_map.rows_per_zone else {
                continue;
            };
            // The policy is optional metadata: a segment that cannot be read
            // right now costs this file its seed for that index, never the
            // write itself.
            let positions = match purpose {
                SeedWriterPurpose::FeatureFlags => false,
                SeedWriterPurpose::Collect => {
                    match stored_null_bitmap_policy(dataset, index).await {
                        Ok(positions) => positions,
                        Err(err) => {
                            log::warn!(
                                "column {}: zone map {} could not be read ({err}); no zone map \
                                 seed is written for it in this file",
                                field.name,
                                index.name
                            );
                            continue;
                        }
                    }
                }
            };
            zone_map_columns.entry(field_id).or_default().push((
                index.name.clone(),
                rows_per_zone,
                positions,
            ));
            continue;
        }
        let details = IndexDetails(index_details.clone());
        let Ok(plugin) = details.get_plugin() else {
            continue;
        };
        if let Some(writer) = plugin
            .create_seed_writer(&field.name, &field.data_type(), &index_details)
            .await?
        {
            other_columns.insert(field_id);
            writers.push(writer);
        }
    }
    for (field_id, consumers) in zone_map_columns {
        let Some(field) = schema.fields.iter().find(|f| f.id == field_id) else {
            continue;
        };
        if other_columns.contains(&field_id) {
            // One seed key per column; a zone map and another plugin's
            // seed cannot both be served. Left unresolved rather than
            // guessing: the zone maps scan their new fragments.
            let message = format!(
                "column {} has a zone map and another seeded index: no zone map seed is written for it",
                field.name
            );
            if SEED_CONFLICT_WARNED.is_completed() {
                log::debug!("{message}");
            } else {
                SEED_CONFLICT_WARNED.call_once(|| log::warn!("{message}"));
            }
            continue;
        }
        // Several zone maps on one column share one seed: the finest zone
        // size serves every coarser index through whole-record reuse, and
        // positions are collected if any of them keeps a bitmap. Each
        // index still cuts its own zones by its own configuration.
        let Some(rows_per_zone) = consumers.iter().map(|c| c.1).min() else {
            continue;
        };
        let positions = consumers.iter().any(|c| c.2);
        if purpose == SeedWriterPurpose::Collect
            && consumers
                .iter()
                .any(|c| c.1 != rows_per_zone || c.2 != positions)
        {
            log::info!(
                "column {}: zone maps {:?} share one seed of {rows_per_zone} rows per record{}",
                field.name,
                consumers.iter().map(|c| c.0.as_str()).collect::<Vec<_>>(),
                if positions {
                    " with null positions"
                } else {
                    ""
                }
            );
        }
        writers.push(Box::new(
            ZoneMapSeedWriter::new(&field.name, rows_per_zone, field.data_type())?
                .with_null_positions(positions),
        ));
    }
    Ok(writers)
}

/// The feature flags the seeds of `writers` require, combined.
fn required_seed_feature_flags(writers: &[Box<dyn IndexSeedWriter>]) -> u64 {
    writers
        .iter()
        .fold(0, |flags, writer| flags | writer.required_feature_flags())
}

/// Set, in both feature words of a manifest being built, the flags required
/// by every seed its indices request.
///
/// `indices` is the prepared index list of that manifest. An index that
/// requests a flagged seed format activates the flags in the commit that
/// publishes it (`CreateIndex` of a seeded ZoneMap), before any file carries
/// such a seed; an index left behind by a build that did not set them
/// activates them in the next commit of any kind. Until then the write path
/// embeds no such seed ([`SeedCollector::for_write`]), and afterwards the
/// flags are sticky, so no manifest references a seed its flags do not
/// announce. Nothing is read from data files or index files; `dataset`
/// only serves as the index store for details that are not inline.
pub async fn apply_seed_feature_flags(
    manifest: &mut Manifest,
    indices: &[IndexMetadata],
    dataset: &Dataset,
) -> Result<()> {
    if indices.is_empty() {
        return Ok(());
    }
    let writers = seed_writers_for(
        dataset,
        indices,
        &manifest.schema,
        SeedWriterPurpose::FeatureFlags,
    )
    .await?;
    let required = required_seed_feature_flags(&writers);
    manifest.reader_feature_flags |= required;
    manifest.writer_feature_flags |= required;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::ArrayRef;
    use arrow_schema::DataType;
    use bytes::Bytes;
    use lance_index::scalar::zonemap::ZoneMapSeedWriter;
    use lance_table::feature_flags::FLAG_ZONE_WRITE_SEEDS;

    /// A seed format that needs no reader support.
    #[derive(Debug)]
    struct PlainSeedWriter;

    impl IndexSeedWriter for PlainSeedWriter {
        fn column_name(&self) -> &str {
            "plain"
        }

        fn observe_batch(&mut self, _values: &ArrayRef) -> Result<()> {
            Ok(())
        }

        fn finish(&mut self) -> Result<Option<Bytes>> {
            Ok(None)
        }

        fn schema_metadata_key(&self) -> String {
            "lance.seed.plain".to_string()
        }

        fn schema_metadata_value(&self, buf_index: u32) -> String {
            buf_index.to_string()
        }
    }

    #[test]
    fn test_required_flags_follow_the_seed_format() {
        let plain: Vec<Box<dyn IndexSeedWriter>> = vec![Box::new(PlainSeedWriter)];
        assert_eq!(required_seed_feature_flags(&plain), 0);
        assert_eq!(required_seed_feature_flags(&[]), 0);

        let zone_map: Vec<Box<dyn IndexSeedWriter>> = vec![
            Box::new(PlainSeedWriter),
            Box::new(ZoneMapSeedWriter::new("val", 8, DataType::Int32).unwrap()),
        ];
        assert_eq!(
            required_seed_feature_flags(&zone_map),
            FLAG_ZONE_WRITE_SEEDS
        );
    }
}
