// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Write seed collection shared by every writer that produces a data file.
//!
//! A [`SeedCollector`] decides which columns of a data file get a seed, feeds
//! each written batch to the seed writers, and embeds the finished seeds in the
//! file before it is closed. Writers that create whole fragments and writers
//! that add or replace a column inside an existing fragment all go through it,
//! so a seeded column keeps its seed in whichever data file serves it.

use arrow_array::RecordBatch;
use lance_core::Result;
use lance_core::datatypes::Schema;
use lance_file::version::ConcreteFileVersion;
use lance_index::scalar::seed::IndexSeedWriter;

use arrow_schema::DataType;
use lance_index::scalar::zonemap::ZoneMapSeedWriter;

use super::GenericWriter;
use crate::Dataset;
use crate::index::scalar::{IndexDetails, fetch_index_details};
use crate::index::{index_is_usable, load_all_indices};

/// Zone size for seeds collected by default (no index asked for them).
/// Matches the zone map plugin's default, so a later index build from these
/// seeds sees the granularity it would have requested.
pub const DEFAULT_SEED_ROWS_PER_ZONE: u64 = 8192;

/// Whether a column of this type gets a seed when collection is on by
/// default. Mirrors the types whose statistics carry usable bounds; columns
/// outside it would pay the footer bytes for null-only records.
fn column_gets_default_seed(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
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
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
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
    )
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
    /// for one through its plugin (`consult_indices` is false for writes that
    /// discard the dataset's indices, where those seeds have no consumer), or
    /// when `collect_by_default` is set, for every supported column type: the
    /// seed then serves whatever index is created later, without rereading
    /// the column.
    pub async fn for_write(
        version: ConcreteFileVersion,
        dataset: Option<&Dataset>,
        write_schema: &Schema,
        consult_indices: bool,
        collect_by_default: bool,
    ) -> Result<Self> {
        // Legacy files have no global buffers to hold a seed.
        if version == ConcreteFileVersion::V1 {
            return Ok(Self::disabled());
        }
        let mut writers: Vec<Box<dyn IndexSeedWriter>> = Vec::new();
        if let (Some(dataset), true) = (dataset, consult_indices) {
            Self::index_driven_writers(dataset, write_schema, &mut writers).await?;
        }
        if collect_by_default {
            for field in write_schema.fields.iter() {
                if writers.iter().any(|w| w.column_name() == field.name) {
                    continue;
                }
                let data_type = field.data_type();
                if !column_gets_default_seed(&data_type) || field.is_blob_v2() {
                    continue;
                }
                writers.push(Box::new(ZoneMapSeedWriter::new(
                    &field.name,
                    DEFAULT_SEED_ROWS_PER_ZONE,
                    data_type,
                )?));
            }
        }
        Ok(Self { writers })
    }

    /// The index-driven writers: columns whose existing, usable index asks
    /// for a seed through its plugin.
    async fn index_driven_writers(
        dataset: &Dataset,
        write_schema: &Schema,
        writers: &mut Vec<Box<dyn IndexSeedWriter>>,
    ) -> Result<()> {
        let indices = load_all_indices(dataset).await?;
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
            let Some(field) = write_schema.fields.iter().find(|f| f.id == field_id) else {
                continue;
            };
            let Ok(index_details) = fetch_index_details(dataset, &field.name, index).await else {
                continue;
            };
            let details = IndexDetails(index_details.clone());
            let Ok(plugin) = details.get_plugin() else {
                continue;
            };
            if let Some(writer) = plugin
                .create_seed_writer(&field.name, &field.data_type(), &index_details)
                .await?
            {
                writers.push(writer);
            }
        }
        Ok(())
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

    /// Whether any column of this write receives a seed.
    pub(crate) fn has_writers(&self) -> bool {
        !self.writers.is_empty()
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

/// Whether this transaction's data writes collected version 1 seeds, by the
/// same rule the writers used: the commit must then carry the paired
/// `FLAG_ZONE_WRITE_SEEDS` bits (sticky thereafter), so builds that cannot
/// read version 1 seeds refuse the dataset instead of misreading them.
///
/// Mirrors `SeedCollector::for_write` over the dataset schema, which can only
/// over-approximate the written schema: a transaction that happened not to
/// touch the seeded column still sets the bits, which is safe (the flag
/// promises nothing about any particular file). `collect_by_default` is true
/// here even though a write can opt out through `WriteParams`: the commit
/// cannot see the write's params, and over-approximating keeps the unsafe
/// direction (seeded files without the flag) impossible. Overwrite discards
/// every index, so its writes consulted none and collected nothing.
pub async fn transaction_writes_v1_seeds(
    dataset: Option<&Dataset>,
    operation: &crate::dataset::transaction::Operation,
) -> Result<bool> {
    use crate::dataset::transaction::Operation;
    let adds_data_files = matches!(
        operation,
        Operation::Append { .. }
            | Operation::Update { .. }
            | Operation::Merge { .. }
            | Operation::DataReplacement { .. }
    );
    if !adds_data_files {
        return Ok(false);
    }
    let Some(dataset) = dataset else {
        return Ok(false);
    };
    let version = dataset.manifest.data_storage_format.lance_file_format();
    if version == ConcreteFileVersion::V1 {
        return Ok(false);
    }
    Ok(
        SeedCollector::for_write(version, Some(dataset), dataset.schema(), true, true)
            .await?
            .has_writers(),
    )
}
