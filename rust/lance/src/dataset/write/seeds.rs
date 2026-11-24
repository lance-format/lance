// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};

use std::sync::Arc;

use async_trait::async_trait;
use lance_core::Result;
use lance_core::datatypes::Schema;
use lance_file::version::ConcreteFileVersion;
use lance_index::pbold;
use lance_index::scalar::zonemap::{ZoneMapSeedVersion, ZoneMapSeedWriter};
use lance_table::feature_flags::write_v1_zone_write_seeds;
use lance_table::format::IndexMetadata;
use lance_table::format::index_seed::IndexSeedWriter;

use crate::index::scalar::zonemap::stored_null_bitmap_policy;
use crate::index::scalar::{IndexDetails, fetch_index_details};
use crate::index::{loading::load_all_indices, segment::index_is_usable};
use crate::{Dataset, session::Session};
use lance_table_runtime::write::SeedCollector;
use lance_table_runtime::{Table, WriteSeedHook};

/// The unresolved same-column conflict is a configuration fact, so it is
/// reported once per process and at debug level afterwards.
static SEED_CONFLICT_WARNED: std::sync::Once = std::sync::Once::new();

/// Selects seed writers using Lance's scalar index registry.
pub struct LanceWriteSeedHook {
    session: Arc<Session>,
}

impl LanceWriteSeedHook {
    pub(crate) fn new(session: Arc<Session>) -> Self {
        Self { session }
    }
}

#[async_trait]
impl WriteSeedHook for LanceWriteSeedHook {
    async fn seed_writers(
        &self,
        table: &Table,
        _: ConcreteFileVersion,
        schema: &Schema,
    ) -> Result<Vec<Box<dyn IndexSeedWriter>>> {
        let dataset = Dataset::from_committed_table(table.clone(), self.session.clone());
        let indices = load_all_indices(&dataset).await?;
        let zone_map_seed_version = if write_v1_zone_write_seeds(table.manifest()) {
            ZoneMapSeedVersion::V1
        } else {
            ZoneMapSeedVersion::V0
        };
        seed_writers_for(&dataset, &indices, schema, zone_map_seed_version).await
    }
}

pub async fn collect_write_seeds(
    version: ConcreteFileVersion,
    dataset: &Dataset,
    schema: &Schema,
) -> Result<SeedCollector> {
    let hook = LanceWriteSeedHook::new(dataset.session.clone());
    let table = dataset.table_view().to_table();
    SeedCollector::for_write(version, Some(&table), schema, Some(&hook)).await
}

/// The seed writers that the usable indices in `indices` request for the
/// top-level columns of `schema`.
///
/// `dataset` resolves index details that are not inline in the metadata; it
/// only has to share its index store with `indices`.
async fn seed_writers_for(
    dataset: &Dataset,
    indices: &[IndexMetadata],
    schema: &Schema,
    zone_map_seed_version: ZoneMapSeedVersion,
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
            let positions = match stored_null_bitmap_policy(dataset, index).await {
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
        if consumers
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
                .with_version(zone_map_seed_version)
                .with_null_positions(positions),
        ));
    }
    Ok(writers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::ArrayRef;
    use arrow_schema::DataType;
    use bytes::Bytes;
    use lance_index::scalar::zonemap::{ZoneMapSeedVersion, ZoneMapSeedWriter};
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
        assert_eq!(PlainSeedWriter.required_feature_flags(), 0);
        let v0 = ZoneMapSeedWriter::new("val", 8, DataType::Int32)
            .unwrap()
            .with_version(ZoneMapSeedVersion::V0);
        assert_eq!(v0.required_feature_flags(), 0);

        let v1 = ZoneMapSeedWriter::new("val", 8, DataType::Int32)
            .unwrap()
            .with_version(ZoneMapSeedVersion::V1);
        assert_eq!(v1.required_feature_flags(), FLAG_ZONE_WRITE_SEEDS);
    }
}
