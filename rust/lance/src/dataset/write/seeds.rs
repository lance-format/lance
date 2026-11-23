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
//!
//! Columns without a seeded index get seeds only when the dataset config
//! asks for them ([`ZoneWriteSeedsConfig`]):
//!
//! - `lance.zone_write_seeds.enabled`: `"true"` collects a zone map seed for
//!   every top-level column of a supported scalar type that no seeded
//!   ZoneMap index covers, so a ZoneMap index created later can be built
//!   from seeds instead of scanning. Nested, vector, dictionary, view,
//!   extension and blob columns are never seeded by the config. Absent
//!   or `"false"` disables this; any other value is treated as disabled with
//!   a warning.
//! - `lance.zone_write_seeds.rows_per_zone`: rows per seed zone for those
//!   columns, a positive integer. Absent, zero or unparsable values use
//!   [`ROWS_PER_ZONE_DEFAULT`] (the latter two with a warning).
//!
//! A column with a seeded ZoneMap index keeps following that index's
//! `rows_per_zone`. Enable the setting with [`Dataset::update_config`]; the
//! commit that sets it also sets the seed feature flags, and writes based on
//! later versions embed the seeds.

use std::collections::{HashMap, HashSet};

use arrow_array::RecordBatch;
use arrow_schema::DataType;
use lance_core::Result;
use lance_core::datatypes::{Field, Schema};
use lance_file::version::ConcreteFileVersion;
use lance_index::pbold;
use lance_index::scalar::seed::IndexSeedWriter;
use lance_index::scalar::zonemap::{ROWS_PER_ZONE_DEFAULT, ZoneMapSeedWriter};
use lance_table::format::{IndexMetadata, Manifest};

use super::GenericWriter;
use crate::Dataset;
use crate::index::scalar::zonemap::stored_null_bitmap_policy;
use crate::index::scalar::{IndexDetails, fetch_index_details};
use crate::index::{loading::load_all_indices, segment::index_is_usable};

/// The unresolved same-column conflict is a configuration fact, so it is
/// reported once per process and at debug level afterwards.
static SEED_CONFLICT_WARNED: std::sync::Once = std::sync::Once::new();

/// Dataset config key that enables zone map seeds for columns without a
/// seeded ZoneMap index (`"true"` / `"false"`).
pub const ZONE_WRITE_SEEDS_ENABLED_KEY: &str = "lance.zone_write_seeds.enabled";
/// Dataset config key for the rows per seed zone of those columns.
pub const ZONE_WRITE_SEEDS_ROWS_PER_ZONE_KEY: &str = "lance.zone_write_seeds.rows_per_zone";

/// The dataset-level zone write seed setting, parsed from the dataset config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZoneWriteSeedsConfig {
    /// Rows per seed zone for columns without a seeded ZoneMap index.
    pub rows_per_zone: u64,
}

impl ZoneWriteSeedsConfig {
    /// The setting in `config`, or `None` when it is disabled.
    ///
    /// Invalid values never fail a write: an unparsable `enabled` disables
    /// the setting and an invalid `rows_per_zone` falls back to
    /// [`ROWS_PER_ZONE_DEFAULT`], both with a warning.
    pub fn from_config(config: &HashMap<String, String>) -> Option<Self> {
        let enabled = config.get(ZONE_WRITE_SEEDS_ENABLED_KEY)?;
        match enabled.parse::<bool>() {
            Ok(true) => {}
            Ok(false) => return None,
            Err(err) => {
                log::warn!(
                    "ignoring {ZONE_WRITE_SEEDS_ENABLED_KEY}={enabled:?} ({err}); zone write \
                     seeds stay disabled"
                );
                return None;
            }
        }
        let rows_per_zone = match config.get(ZONE_WRITE_SEEDS_ROWS_PER_ZONE_KEY) {
            None => ROWS_PER_ZONE_DEFAULT,
            Some(value) => match value.parse::<u64>() {
                Ok(rows) if rows > 0 => rows,
                _ => {
                    log::warn!(
                        "ignoring {ZONE_WRITE_SEEDS_ROWS_PER_ZONE_KEY}={value:?}: expected a \
                         positive integer; using {ROWS_PER_ZONE_DEFAULT}"
                    );
                    ROWS_PER_ZONE_DEFAULT
                }
            },
        };
        Some(Self { rows_per_zone })
    }
}

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
    /// flags that seed format requires. With zone write seeds enabled in the
    /// dataset config ([`ZoneWriteSeedsConfig`]), every other column the zone
    /// map supports gets a zone map seed under the same flag rule.
    /// `consult_indices` is false for writes that discard the dataset's
    /// indices (overwrite), where seeds have no consumer.
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
        let config = ZoneWriteSeedsConfig::from_config(&manifest.config);
        let writers = seed_writers_for(
            dataset,
            &indices,
            write_schema,
            config,
            SeedWriterPurpose::Collect,
        )
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

/// The seed writers that the usable indices in `indices` request for the
/// top-level columns of `schema`, plus, when `config` is set, a zone map seed
/// of the configured size for every other column the zone map supports.
///
/// `dataset` resolves index details that are not inline in the metadata; it
/// only has to share its index store with `indices`, which may belong to a
/// manifest that is still being built.
async fn seed_writers_for(
    dataset: &Dataset,
    indices: &[IndexMetadata],
    schema: &Schema,
    config: Option<ZoneWriteSeedsConfig>,
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
    // A column served by a seeded index follows that index; the configured
    // size only applies to the rest.
    let config_columns: Vec<_> = config
        .map(|config| {
            schema
                .fields
                .iter()
                .filter(|field| {
                    !zone_map_columns.contains_key(&field.id)
                        && !other_columns.contains(&field.id)
                        && config_seed_supports(field)
                })
                .map(|field| (field, config.rows_per_zone))
                .collect()
        })
        .unwrap_or_default();
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
    // Configured seeds have no index whose null bitmap they could feed. A
    // writer that cannot be created is skipped rather than failing the write.
    for (field, rows_per_zone) in config_columns {
        match ZoneMapSeedWriter::new(&field.name, rows_per_zone, field.data_type()) {
            Ok(writer) => writers.push(Box::new(writer.with_null_positions(false))),
            Err(err) => log::debug!(
                "column {}: no configured zone write seed ({err})",
                field.name
            ),
        }
    }
    Ok(writers)
}

/// Whether the dataset config seeds `field`: flat scalar columns only, whose
/// zone statistics are cheap to collect on every write. Blob and
/// extension-typed columns are excluded because the written values are not
/// the column's logical values.
fn config_seed_supports(field: &Field) -> bool {
    if field.is_blob() || field.is_blob_v2() || field.extension_name().is_some() {
        return false;
    }
    matches!(
        field.data_type(),
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
    )
}

/// The feature flags the seeds of `writers` require, combined.
fn required_seed_feature_flags(writers: &[Box<dyn IndexSeedWriter>]) -> u64 {
    writers
        .iter()
        .fold(0, |flags, writer| flags | writer.required_feature_flags())
}

/// Set, in both feature words of a manifest being built, the flags required
/// by every seed its indices or its zone write seed config request.
///
/// `indices` is the prepared index list of that manifest. An index that
/// requests a flagged seed format activates the flags in the commit that
/// publishes it (`CreateIndex` of a seeded ZoneMap), before any file carries
/// such a seed; a commit whose config enables zone write seeds activates
/// them in that same commit; an index left behind by a build that did not set them
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
    let config = ZoneWriteSeedsConfig::from_config(&manifest.config);
    if indices.is_empty() && config.is_none() {
        return Ok(());
    }
    let writers = seed_writers_for(
        dataset,
        indices,
        &manifest.schema,
        config,
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
    use std::sync::Arc;

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

    #[rstest::rstest]
    #[case::absent(&[], None)]
    #[case::disabled(&[(ZONE_WRITE_SEEDS_ENABLED_KEY, "false")], None)]
    #[case::size_without_enabled(&[(ZONE_WRITE_SEEDS_ROWS_PER_ZONE_KEY, "16")], None)]
    #[case::unparsable_enabled(
        &[(ZONE_WRITE_SEEDS_ENABLED_KEY, "yes"), (ZONE_WRITE_SEEDS_ROWS_PER_ZONE_KEY, "16")],
        None
    )]
    #[case::default_size(&[(ZONE_WRITE_SEEDS_ENABLED_KEY, "true")], Some(ROWS_PER_ZONE_DEFAULT))]
    #[case::explicit_size(
        &[(ZONE_WRITE_SEEDS_ENABLED_KEY, "true"), (ZONE_WRITE_SEEDS_ROWS_PER_ZONE_KEY, "16")],
        Some(16)
    )]
    #[case::zero_size(
        &[(ZONE_WRITE_SEEDS_ENABLED_KEY, "true"), (ZONE_WRITE_SEEDS_ROWS_PER_ZONE_KEY, "0")],
        Some(ROWS_PER_ZONE_DEFAULT)
    )]
    #[case::negative_size(
        &[(ZONE_WRITE_SEEDS_ENABLED_KEY, "true"), (ZONE_WRITE_SEEDS_ROWS_PER_ZONE_KEY, "-4")],
        Some(ROWS_PER_ZONE_DEFAULT)
    )]
    #[case::unparsable_size(
        &[(ZONE_WRITE_SEEDS_ENABLED_KEY, "true"), (ZONE_WRITE_SEEDS_ROWS_PER_ZONE_KEY, "many")],
        Some(ROWS_PER_ZONE_DEFAULT)
    )]
    fn test_zone_write_seeds_config_from_config(
        #[case] entries: &[(&str, &str)],
        #[case] expected_rows_per_zone: Option<u64>,
    ) {
        let config: HashMap<String, String> = entries
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        assert_eq!(
            ZoneWriteSeedsConfig::from_config(&config),
            expected_rows_per_zone.map(|rows_per_zone| ZoneWriteSeedsConfig { rows_per_zone })
        );
    }

    #[rstest::rstest]
    #[case::boolean(DataType::Boolean, true)]
    #[case::int64(DataType::Int64, true)]
    #[case::uint8(DataType::UInt8, true)]
    #[case::float16(DataType::Float16, true)]
    #[case::date32(DataType::Date32, true)]
    #[case::time64(DataType::Time64(arrow_schema::TimeUnit::Nanosecond), true)]
    #[case::timestamp_tz(
        DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, Some("UTC".into())),
        true
    )]
    #[case::duration(DataType::Duration(arrow_schema::TimeUnit::Second), true)]
    #[case::utf8(DataType::Utf8, true)]
    #[case::large_binary(DataType::LargeBinary, true)]
    #[case::fixed_size_binary(DataType::FixedSizeBinary(16), true)]
    #[case::decimal128(DataType::Decimal128(10, 2), true)]
    #[case::utf8_view(DataType::Utf8View, false)]
    #[case::null(DataType::Null, false)]
    #[case::fixed_size_list(
        DataType::FixedSizeList(
            Arc::new(arrow_schema::Field::new("item", DataType::Float32, true)),
            4
        ),
        false
    )]
    #[case::list(
        DataType::List(Arc::new(arrow_schema::Field::new("item", DataType::Int32, true))),
        false
    )]
    #[case::struct_(
        DataType::Struct(vec![arrow_schema::Field::new("a", DataType::Int32, true)].into()),
        false
    )]
    #[case::dictionary(
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        false
    )]
    fn test_config_seed_supports_type(#[case] data_type: DataType, #[case] expected: bool) {
        let field = Field::try_from(&arrow_schema::Field::new("col", data_type, true)).unwrap();
        assert_eq!(config_seed_supports(&field), expected);
    }

    #[test]
    fn test_config_seed_skips_blob_columns() {
        let blob_v2 = Field::try_from(&crate::blob::blob_field("blob", true)).unwrap();
        assert!(!config_seed_supports(&blob_v2));
        let legacy_blob = Field::try_from(
            &arrow_schema::Field::new("blob", DataType::LargeBinary, true).with_metadata(
                HashMap::from([(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string())]),
            ),
        )
        .unwrap();
        assert!(!config_seed_supports(&legacy_blob));
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
