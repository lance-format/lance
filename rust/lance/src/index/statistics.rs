// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index descriptions and serialized index statistics.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use lance_core::utils::parse::parse_env_as_bool;
use lance_index::frag_reuse::CompactFragReuseIndexHandle;
use lance_index::mem_wal::MemWalIndexHandle;
use lance_index::registry::display_type_from_url;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::{
    Index, IndexDescription, IndexFragmentCoverage, IndexType, is_system_index,
    metrics::NoOpMetricsCollector,
};
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;
use serde_json::json;

use crate::dataset::index::LanceIndexStoreExt;
use crate::index::api::{DatasetIndexExt, IndexSegmentStatistics};
use crate::index::loading::{DatasetIndexInternalExt, find_frag_reuse_index_meta};
use crate::index::scalar::{self, IndexDetails};
use crate::index::segment::segment_has_vector_details;
use crate::index::vector::details::{derive_vector_index_type, vector_details_as_json};
use crate::{Dataset, Error, Result};

fn sum_indexed_rows_per_delta(indexed_fragments_per_delta: &[Vec<Fragment>]) -> Result<Vec<usize>> {
    let mut rows_per_delta = Vec::with_capacity(indexed_fragments_per_delta.len());
    for frags in indexed_fragments_per_delta {
        let mut sum = 0usize;
        for frag in frags {
            sum += frag.num_rows().ok_or_else(|| {
                Error::internal(
                    "Fragment should have row counts, please upgrade lance and \
                                  trigger a single write to fix this"
                        .to_string(),
                )
            })?;
        }
        rows_per_delta.push(sum);
    }
    Ok(rows_per_delta)
}

fn unique_indexed_fragment_count(indexed_fragments_per_delta: &[Vec<Fragment>]) -> Option<usize> {
    let mut fragment_ids = HashSet::new();
    for frags in indexed_fragments_per_delta {
        for frag in frags {
            if !fragment_ids.insert(frag.id) {
                return None;
            }
        }
    }
    Some(fragment_ids.len())
}

// Whether to auto-migrate a dataset when we encounter corruption.
fn auto_migrate_corruption() -> bool {
    static LANCE_AUTO_MIGRATION: OnceLock<bool> = OnceLock::new();
    *LANCE_AUTO_MIGRATION.get_or_init(|| parse_env_as_bool("LANCE_AUTO_MIGRATION", true))
}

/// Derive a friendly (but not necessarily unique) type name from a type URL.
/// Extract a human-friendly type name from a type URL.
///
/// Strips prefixes like `type.googleapis.com/` and package names, then removes
/// trailing `IndexDetails` / `Index` so callers get a concise display name.
fn type_name_from_uri(index_uri: &str) -> String {
    let type_name = index_uri.rsplit('/').next().unwrap_or(index_uri);
    let type_name = type_name.rsplit('.').next().unwrap_or(type_name);
    type_name.trim_end_matches("IndexDetails").to_string()
}

/// Legacy mapping from type URL to the old IndexType string for backwards compatibility.
/// Legacy mapping from type URL to the old IndexType string for backwards compatibility.
///
/// If `index_type_hint` is provided (e.g. parsed from the index statistics of a concrete
/// index instance), it takes precedence so callers can surface the exact index type even
/// when the type URL alone is too generic (such as VectorIndexDetails).
fn legacy_type_name(index_uri: &str, index_type_hint: Option<&str>) -> String {
    if let Some(hint) = index_type_hint {
        return hint.to_string();
    }

    let base = type_name_from_uri(index_uri);

    match base.as_str() {
        "BTree" => IndexType::BTree.to_string(),
        "Bitmap" => IndexType::Bitmap.to_string(),
        "LabelList" => IndexType::LabelList.to_string(),
        "NGram" => IndexType::NGram.to_string(),
        "ZoneMap" => IndexType::ZoneMap.to_string(),
        "BloomFilter" => IndexType::BloomFilter.to_string(),
        "RTree" => IndexType::RTree.to_string(),
        "Inverted" => IndexType::Inverted.to_string(),
        "FMIndex" | "FM" => IndexType::Fm.to_string(),
        "MinHashLsh" => IndexType::MinHashLsh.to_string(),
        "Json" => IndexType::Scalar.to_string(),
        "Flat" | "Vector" => IndexType::Vector.to_string(),
        other if other.contains("Vector") => IndexType::Vector.to_string(),
        _ => scalar::SCALAR_INDEX_PLUGIN_REGISTRY
            .get_plugin_by_details(&prost_types::Any {
                type_url: index_uri.to_string(),
                value: Vec::new(),
            })
            .map(|plugin| plugin.name().to_string())
            .unwrap_or_else(|_| "N/A".to_string()),
    }
}

pub(super) struct IndexDescriptionImpl {
    name: String,
    field_ids: Vec<u32>,
    segments: Vec<IndexMetadata>,
    index_type: String,
    /// Index details, or `None` for indices created before details were
    /// persisted in the manifest. Such indices are still described on a
    /// best-effort basis rather than rejected.
    details: Option<IndexDetails>,
    rows_indexed: u64,
    fragment_coverage: Option<IndexFragmentCoverage>,
}

fn is_externally_maintained_index(metadata: &IndexMetadata) -> bool {
    metadata.index_details.as_ref().is_some_and(|details| {
        scalar::SCALAR_INDEX_PLUGIN_REGISTRY
            .get_plugin_by_details(details)
            .is_ok_and(|plugin| !plugin.supports_optimization())
    })
}

fn empty_fragment_bitmap_has_unknown_coverage(metadata: &IndexMetadata) -> bool {
    let Some(details) = metadata.index_details.as_ref() else {
        // Preserve best-effort behavior for legacy descriptions without details.
        return false;
    };
    if segment_has_vector_details(metadata) {
        return false;
    }
    match scalar::SCALAR_INDEX_PLUGIN_REGISTRY.get_plugin_by_details(details) {
        Ok(plugin) => !plugin.supports_optimization(),
        // A description can still use a type URL whose plugin is unavailable.
        // In that case an empty bitmap is not evidence of known-zero coverage.
        Err(_) => true,
    }
}

impl IndexDescriptionImpl {
    pub(super) async fn try_new(segments: Vec<IndexMetadata>, dataset: &Dataset) -> Result<Self> {
        if segments.is_empty() {
            return Err(Error::index("Index metadata is empty".to_string()));
        }

        // We assume the type URL and details are the same for all segments
        let example_metadata = &segments[0];

        let name = example_metadata.name.clone();
        if !segments.iter().all(|shard| shard.name == name) {
            return Err(Error::index(
                "Index name should be identical across all segments".to_string(),
            ));
        }

        let field_ids = &example_metadata.fields;
        if !segments.iter().all(|shard| shard.fields == *field_ids) {
            return Err(Error::index(
                "Index fields should be identical across all segments".to_string(),
            ));
        }
        // Only the keyed prefix. `fields` also lists the columns the index merely
        // carries values for, which it cannot be searched on, and this list is what
        // every binding advertises as the index's columns -- Python's
        // `_default_vector_index_for_column` matches on membership of it, so
        // including a carried column here would hand back the keyed column's IVF
        // model for a query about the carried one.
        let field_ids_vec: Vec<u32> = example_metadata
            .keyed_fields()
            .iter()
            .map(|id| *id as u32)
            .collect();

        // Index details may be absent on indices created before details were
        // persisted in the manifest. We describe such indices on a best-effort
        // basis rather than erroring, so callers can still see they exist.
        let details = example_metadata.index_details.clone().map(IndexDetails);
        if let Some(details) = details.as_ref() {
            let type_url = &details.0.type_url;
            if !segments.iter().all(|shard| {
                shard
                    .index_details
                    .as_ref()
                    .map(|d| d.type_url == *type_url)
                    .unwrap_or(false)
            }) {
                return Err(Error::index(
                    "Index type URL should be present and identical across all segments"
                        .to_string(),
                ));
            }
        }

        let index_type =
            if let Some(system_type) = lance_index::infer_system_index_type(example_metadata) {
                // System indices (frag-reuse, mem-wal) are identified by name, not
                // by index details, so this must be checked before the plugin lookup.
                system_type.to_string()
            } else if let Some(details) = details.as_ref() {
                if details.is_vector() {
                    derive_vector_index_type(&details.0)
                } else {
                    // Fall back to a name derived from the type URL when no plugin
                    // is registered, so a known type URL is never reported as the
                    // opaque "Unknown".
                    details
                        .get_plugin()
                        .map(|p| p.name().to_string())
                        .unwrap_or_else(|_| {
                            display_type_from_url(details.0.type_url.as_str()).to_string()
                        })
                }
            } else if segment_has_vector_details(example_metadata) {
                // Legacy vector indices predate VectorIndexDetails and are
                // recognized by their monolithic index file name.
                "Vector".to_string()
            } else {
                "Unknown".to_string()
            };

        let mut fragment_rows = HashMap::with_capacity(dataset.manifest.fragments.len());
        for fragment in dataset.iter_fragments() {
            fragment_rows.insert(
                fragment.id as u32,
                fragment_logical_rows_from_metadata(fragment)?,
            );
        }
        let mut rows_indexed = 0;
        let mut indexed_fragment_refs = 0u64;
        let mut missing_fragment_refs = 0u64;
        let mut indexed_fragments = RoaringBitmap::new();
        let mut fragment_bitmap_size_bytes = 0u64;
        let mut fragment_coverage_unknown = false;
        // System-index bitmaps have index-specific meanings (for example,
        // fragment-reuse participation), not data-index coverage.
        let reports_fragment_coverage = !is_system_index(example_metadata);

        for shard in &segments {
            let Some(fragment_bitmap) = shard.fragment_bitmap.as_ref() else {
                // A system index (e.g. __mem_wal) indexes no fragments, so a
                // missing bitmap means zero indexed rows. For a data index it
                // means unknown coverage — reject rather than fabricate a count.
                if is_system_index(shard) {
                    continue;
                }
                return Err(Error::index("Fragment bitmap is required for index description.  This index must be retrained to support this method.".to_string()));
            };

            if reports_fragment_coverage {
                // Externally maintained indices can keep coverage in their own
                // payload and deliberately declare an empty bitmap. A genuinely
                // empty built-in index still has known zero coverage.
                if fragment_bitmap.is_empty() && empty_fragment_bitmap_has_unknown_coverage(shard) {
                    fragment_coverage_unknown = true;
                }
                indexed_fragments |= fragment_bitmap;
                let serialized_size = shard.fragment_bitmap_serialized_size().ok_or_else(|| {
                    Error::index("Fragment bitmap disappeared while describing index".to_string())
                })?;
                let serialized_size = u64::try_from(serialized_size)
                    .map_err(|_| Error::index("Fragment bitmap size overflow".to_string()))?;
                fragment_bitmap_size_bytes = fragment_bitmap_size_bytes
                    .checked_add(serialized_size)
                    .ok_or_else(|| Error::index("Fragment bitmap size overflow".to_string()))?;
            }
            indexed_fragment_refs += fragment_bitmap.len();
            for fragment_id in fragment_bitmap.iter() {
                if let Some(fragment_rows) = fragment_rows.get(&fragment_id) {
                    rows_indexed += *fragment_rows;
                } else {
                    missing_fragment_refs += 1;
                }
            }
        }
        tracing::debug!(
            index_name = name.as_str(),
            index_type = index_type.as_str(),
            segment_count = segments.len(),
            dataset_fragment_count = fragment_rows.len(),
            indexed_fragment_refs,
            missing_fragment_refs,
            rows_indexed,
            "described index row coverage from fragment metadata"
        );

        let fragment_coverage = if reports_fragment_coverage && !fragment_coverage_unknown {
            let current_fragments = fragment_rows.keys().copied().collect::<RoaringBitmap>();
            Some(IndexFragmentCoverage {
                covered_fragment_count: indexed_fragments.intersection_len(&current_fragments),
                current_fragment_count: current_fragments.len(),
                missing_fragment_count: current_fragments.difference_len(&indexed_fragments),
                stale_fragment_count: indexed_fragments.difference_len(&current_fragments),
                fragment_bitmap_size_bytes,
            })
        } else {
            None
        };

        Ok(Self {
            name,
            field_ids: field_ids_vec,
            index_type,
            segments,
            details,
            rows_indexed,
            fragment_coverage,
        })
    }
}

fn fragment_logical_rows_from_metadata(fragment: &Fragment) -> Result<u64> {
    fragment.num_rows().map(|rows| rows as u64).ok_or_else(|| {
        Error::internal(format!(
            "Index description requires physical row count and deletion count in fragment metadata. To add this, make a write to this dataset using this library version. Fragment id: {}",
            fragment.id
        ))
    })
}

impl IndexDescription for IndexDescriptionImpl {
    fn name(&self) -> &str {
        &self.name
    }

    fn field_ids(&self) -> &[u32] {
        &self.field_ids
    }

    fn index_type(&self) -> &str {
        &self.index_type
    }

    fn metadata(&self) -> &[IndexMetadata] {
        &self.segments
    }

    fn type_url(&self) -> &str {
        self.details
            .as_ref()
            .map(|d| d.0.type_url.as_str())
            .unwrap_or("")
    }

    fn rows_indexed(&self) -> u64 {
        self.rows_indexed
    }

    fn details(&self) -> Result<String> {
        let Some(details) = self.details.as_ref() else {
            return Ok("{}".to_string());
        };
        if details.is_vector() {
            vector_details_as_json(&details.0)
        } else {
            let plugin = details.get_plugin()?;
            plugin.details_as_json(&details.0).map(|v| v.to_string())
        }
    }

    fn total_size_bytes(&self) -> Option<u64> {
        let mut total = 0u64;
        for segment in &self.segments {
            // If any segment is missing file info, return None for backward compatibility
            let files = segment.files.as_ref()?;
            for file in files {
                total += file.size_bytes;
            }
        }
        Some(total)
    }

    fn fragment_coverage(&self) -> Option<&IndexFragmentCoverage> {
        self.fragment_coverage.as_ref()
    }
}

fn serialize_index_statistics(stats: &serde_json::Value) -> Result<String> {
    serde_json::to_string(stats)
        .map_err(|e| Error::index(format!("Failed to serialize index statistics: {}", e)))
}

async fn migrate_and_recompute_index_statistics(ds: &Dataset, index_name: &str) -> Result<String> {
    let mut ds = ds.clone();
    log::warn!(
        "Detecting out-dated fragment metadata, migrating dataset. \
                        To disable migration, set LANCE_AUTO_MIGRATION=false"
    );
    ds.delete("false").await.map(|_| ()).map_err(|err| {
        Error::execution(format!(
            "Failed to migrate dataset while calculating index statistics. \
                        To disable migration, set LANCE_AUTO_MIGRATION=false. Original error: {}",
            err
        ))
    })?;
    ds.index_statistics(index_name).await
}

pub(super) async fn index_statistics_frag_reuse(ds: &Dataset) -> Result<String> {
    if let Some(fri) = find_frag_reuse_index_meta(ds).await?
        && fri.index_version != 0
    {
        return Err(Error::not_supported(
            "FRI index_version 1 statistics are not implemented. Please upgrade to a supporting version",
        ));
    }
    let index = ds
        .open_frag_reuse_index(&NoOpMetricsCollector)
        .await?
        .expect("FragmentReuse index does not exist");
    serialize_index_statistics(&CompactFragReuseIndexHandle(index).statistics()?)
}

pub(super) async fn index_statistics_mem_wal(ds: &Dataset) -> Result<String> {
    let index = ds
        .open_mem_wal_index(&NoOpMetricsCollector)
        .await?
        .expect("MemWal index does not exist");
    serialize_index_statistics(&MemWalIndexHandle(index).statistics()?)
}

pub(super) async fn index_statistics_scalar(
    ds: &Dataset,
    index_name: &str,
    metadatas: Vec<IndexMetadata>,
) -> Result<String> {
    let field_id = metadatas[0].fields[0];
    let field_path = ds.schema().field_path(field_id)?;

    let segment_statistics =
        collect_regular_indices_statistics(ds, &metadatas, &field_path).await?;
    match serialize_regular_index_statistics(ds, index_name, &metadatas, segment_statistics).await?
    {
        Some(statistics) => Ok(statistics),
        None => migrate_and_recompute_index_statistics(ds, index_name).await,
    }
}

pub(super) async fn serialize_regular_index_statistics(
    ds: &Dataset,
    index_name: &str,
    metadatas: &[IndexMetadata],
    segment_statistics: Vec<IndexSegmentStatistics>,
) -> Result<Option<String>> {
    if segment_statistics.len() != metadatas.len() {
        return Err(Error::invalid_input(format!(
            "Expected {} segment statistics for index {index_name}, got {}",
            metadatas.len(),
            segment_statistics.len()
        )));
    }
    let expected_uuids: HashSet<_> = metadatas.iter().map(|metadata| metadata.uuid).collect();
    let mut statistics_by_uuid = HashMap::with_capacity(segment_statistics.len());
    for statistics in segment_statistics {
        if statistics.read_version != ds.manifest.version {
            return Err(Error::invalid_input(format!(
                "Index segment {} statistics are from dataset version {}, expected {}",
                statistics.index_uuid, statistics.read_version, ds.manifest.version
            )));
        }
        if !expected_uuids.contains(&statistics.index_uuid) {
            return Err(Error::invalid_input(format!(
                "Segment {} does not belong to index {index_name} at this snapshot",
                statistics.index_uuid
            )));
        }
        if statistics_by_uuid
            .insert(statistics.index_uuid, statistics)
            .is_some()
        {
            return Err(Error::invalid_input(format!(
                "Duplicate segment statistics for index {index_name}"
            )));
        }
    }
    let mut indices_stats = Vec::with_capacity(metadatas.len());
    let mut index_uri = None;
    for metadata in metadatas {
        let statistics = statistics_by_uuid.remove(&metadata.uuid).ok_or_else(|| {
            Error::invalid_input(format!(
                "Missing statistics for index segment {}",
                metadata.uuid
            ))
        })?;
        if !statistics.index_type_uri.is_empty() {
            index_uri.get_or_insert(statistics.index_type_uri);
        }
        indices_stats.push(statistics.statistics);
    }
    let index_uri = index_uri.unwrap_or_else(|| "unknown".to_string());

    let index_type_hint = indices_stats
        .first()
        .and_then(|stats| stats.get("index_type"))
        .and_then(|v| v.as_str());
    let index_type = legacy_type_name(&index_uri, index_type_hint);

    let Some((
        num_indexed_rows_per_delta,
        num_indexed_fragments,
        num_unindexed_fragments,
        num_indexed_rows,
        num_unindexed_rows,
    )) = gather_fragment_statistics(ds, index_name).await?
    else {
        return Ok(None);
    };

    let num_indices = metadatas.len();
    let updated_at = metadatas
        .iter()
        .filter_map(|metadata| metadata.created_at)
        .max()
        .map(|timestamp| timestamp.timestamp_millis() as u64);
    let stats = json!({
        "index_type": index_type,
        "name": index_name,
        "num_indices": num_indices,
        "num_segments": num_indices,
        "indices": indices_stats.clone(),
        "segments": indices_stats,
        "num_indexed_fragments": num_indexed_fragments,
        "num_indexed_rows": num_indexed_rows,
        "num_unindexed_fragments": num_unindexed_fragments,
        "num_unindexed_rows": num_unindexed_rows,
        "num_indexed_rows_per_delta": num_indexed_rows_per_delta,
        "updated_at_timestamp_ms": updated_at,
    });

    serialize_index_statistics(&stats).map(Some)
}

async fn collect_regular_indices_statistics(
    ds: &Dataset,
    metadatas: &[IndexMetadata],
    field_path: &str,
) -> Result<Vec<IndexSegmentStatistics>> {
    let mut statistics = Vec::with_capacity(metadatas.len());
    for metadata in metadatas {
        statistics.push(collect_index_segment_statistics(ds, metadata, field_path).await?);
    }
    Ok(statistics)
}

pub(super) async fn collect_index_segment_statistics(
    ds: &Dataset,
    metadata: &IndexMetadata,
    field_path: &str,
) -> Result<IndexSegmentStatistics> {
    // Externally maintained indices can keep coverage in their own payload and
    // deliberately declare an empty bitmap. Other empty indices carry only a
    // definition until there is enough data to train them.
    if !is_externally_maintained_index(metadata)
        && metadata
            .fragment_bitmap
            .as_ref()
            .is_some_and(RoaringBitmap::is_empty)
    {
        return Ok(IndexSegmentStatistics {
            read_version: ds.manifest.version,
            index_uuid: metadata.uuid,
            index_type_uri: String::new(),
            statistics: json!({}),
        });
    }
    let index_store = Arc::new(LanceIndexStore::from_dataset_for_existing(ds, metadata).await?);
    let index_details = scalar::fetch_index_details(ds, field_path, metadata).await?;
    let index_details_wrapper = scalar::IndexDetails(index_details.clone());
    let stored_statistics = match index_details_wrapper.get_plugin() {
        Ok(plugin) => {
            plugin
                .load_statistics(index_store, index_details.as_ref())
                .await?
        }
        Err(_) => None,
    };
    let statistics = match stored_statistics {
        Some(statistics) => statistics,
        None => {
            let index = ds
                .open_generic_index(field_path, &metadata.uuid, &NoOpMetricsCollector)
                .await?;
            index.statistics()?
        }
    };
    Ok(IndexSegmentStatistics {
        read_version: ds.manifest.version,
        index_uuid: metadata.uuid,
        index_type_uri: index_details.type_url.clone(),
        statistics,
    })
}

async fn gather_fragment_statistics(
    ds: &Dataset,
    index_name: &str,
) -> Result<Option<(Vec<usize>, usize, usize, usize, usize)>> {
    let indexed_fragments_per_delta = ds.indexed_fragments(index_name).await?;

    let num_indexed_rows_per_delta = match sum_indexed_rows_per_delta(&indexed_fragments_per_delta)
    {
        Ok(rows) => rows,
        Err(Error::Internal { message, .. })
            if auto_migrate_corruption() && message.contains("trigger a single write") =>
        {
            return Ok(None);
        }
        Err(e) => return Err(e),
    };

    let Some(num_indexed_fragments) = unique_indexed_fragment_count(&indexed_fragments_per_delta)
    else {
        if auto_migrate_corruption() {
            return Ok(None);
        }
        return Err(Error::internal(
            "Overlap in indexed fragments. Please upgrade to lance >= 0.23.0 \
                              and trigger a single write to fix this"
                .to_string(),
        ));
    };

    let num_unindexed_fragments = ds.fragments().len() - num_indexed_fragments;
    let num_indexed_rows: usize = num_indexed_rows_per_delta.iter().sum();

    drop(indexed_fragments_per_delta);
    let total_rows = ds.count_rows(None).await?;
    let num_unindexed_rows = total_rows - num_indexed_rows;

    Ok(Some((
        num_indexed_rows_per_delta,
        num_indexed_fragments,
        num_unindexed_fragments,
        num_indexed_rows,
        num_unindexed_rows,
    )))
}
