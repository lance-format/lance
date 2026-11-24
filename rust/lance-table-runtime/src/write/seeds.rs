// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatch;
use async_trait::async_trait;
use lance_core::{Result, datatypes::Schema};
use lance_file::version::ConcreteFileVersion;
use lance_table::format::index_seed::IndexSeedWriter;

use crate::Table;

use super::file::GenericWriter;

/// Supplies index seed writers for the columns of a data file.
///
/// The runtime checks the writers' required feature flags and embeds their
/// output in the finished file. The hook chooses writers using the host's
/// index registry and the existing table snapshot.
#[async_trait]
pub trait WriteSeedHook: Send + Sync {
    /// Return writers for the given schema and exact data file version.
    async fn seed_writers(
        &self,
        table: &Table,
        version: ConcreteFileVersion,
        schema: &Schema,
    ) -> Result<Vec<Box<dyn IndexSeedWriter>>>;
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
    /// Seeds require a current-format file, an existing table, an active
    /// hook, and the feature flags declared by each returned writer.
    pub async fn for_write(
        version: ConcreteFileVersion,
        table: Option<&Table>,
        write_schema: &Schema,
        hook: Option<&dyn WriteSeedHook>,
    ) -> Result<Self> {
        // Legacy files have no global buffers to hold a seed.
        if version == ConcreteFileVersion::V1 {
            return Ok(Self::disabled());
        }
        let (Some(table), Some(hook)) = (table, hook) else {
            return Ok(Self::disabled());
        };

        let manifest = table.manifest();
        let writers = hook
            .seed_writers(table, version, write_schema)
            .await?
            .into_iter()
            .filter(|writer| {
                let required = writer.required_feature_flags();
                let activated = manifest.reader_feature_flags & required == required
                    && manifest.writer_feature_flags & required == required;
                if !activated {
                    // Never embed a seed whose required feature flags are not
                    // already active on the snapshot.
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
    #[doc(hidden)]
    pub fn len(&self) -> usize {
        self.writers.len()
    }

    #[doc(hidden)]
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
