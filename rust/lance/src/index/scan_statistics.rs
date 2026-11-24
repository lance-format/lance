// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::fmt::Display;

use chrono::{DateTime, Utc};
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::pb::JsonIndexDetails;
use lance_index::pbold::{BTreeIndexDetails, InvertedIndexDetails, ZoneMapIndexDetails};
use lance_index::scalar::registry::{QueryDomain, ScalarIndexPlugin};
use lance_index::scalar::{
    bitmap::BITMAP_LOOKUP_NAME,
    bloomfilter::BLOOMFILTER_FILENAME,
    btree::BTREE_LOOKUP_NAME,
    inverted::{INVERT_LIST_FILE, METADATA_FILE},
    ngram::POSTINGS_FILENAME,
    zonemap::ZONEMAP_INDEX_FILENAME,
};
use lance_index::{INDEX_FILE_NAME, metrics::NoOpMetricsCollector};
use lance_linalg::distance::DistanceType;
use lance_table::format::overlay::staleness::{
    collect_overlay_stale_rows_for_segment, overlaid_fragments,
};
use lance_table::format::{Fragment, IndexMetadata};
use lance_table::io::manifest::read_manifest_indexes;
use lance_table::system_index::is_system_index;
use prost::Message;
use roaring::RoaringBitmap;
use serde_json::Value;
use uuid::Uuid;

use super::append::is_definition_only_segment;
use super::loading::DatasetIndexInternalExt;
use super::loading::recover_scalar_query_domain;
use super::scalar::{IndexDetails, SCALAR_INDEX_PLUGIN_REGISTRY};
use super::segment::{
    index_is_usable, index_type_is_known, segment_has_vector_details, unsupported_index_version,
};
use super::statistics::empty_fragment_bitmap_has_unknown_coverage;
use super::vector::details::{
    decode_vector_details, decoded_vector_index_type, recorded_vector_metric, vector_details_value,
};
use super::vector::utils::infer_vector_element_type;
use crate::{Dataset, Error, Result};

/// Controls how much storage inspection is allowed when collecting index scan statistics.
///
/// See [Dataset::index_scan_statistics] for an example.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexScanStatisticsOptions {
    include_segment_stats: bool,
}

impl IndexScanStatisticsOptions {
    /// Allow fragment-reuse resolution, legacy JSON coverage verification,
    /// and IVF model inspection.
    ///
    /// This may read index files and open a handle per eligible segment.
    /// The layout occupies space proportional to the sum of their partition counts.
    pub fn with_segment_stats(mut self, include_segment_stats: bool) -> Self {
        self.include_segment_stats = include_segment_stats;
        self
    }
}

/// Recorded facts about one named index and the dataset snapshot containing it.
///
/// See [Dataset::index_scan_statistics] for an example.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IndexScanStatistics {
    /// Version of the dataset handle used for the inspection.
    pub dataset_version: u64,
    /// Number of current fragments.
    pub fragment_count: u64,
    /// Live rows in all current fragments, if their counts are known.
    pub live_rows: Option<u64>,
    /// Deleted slots in current fragments, if their counts are known.
    pub deleted_rows: Option<u64>,
    /// Recorded whole-file bytes for current base and overlay data, if complete.
    pub total_data_bytes: Option<u64>,
    /// Logical index name.
    pub index_name: String,
    /// Field IDs the index searches on, in key order.
    pub keyed_field_ids: Vec<i32>,
    /// Field IDs whose values the index carries.
    pub covering_field_ids: Vec<i32>,
    /// Persisted details type URL, when present and shared by every segment.
    pub type_url: Option<String>,
    /// Recorded index type, when known and shared by every segment.
    pub index_type: Option<String>,
    /// Recorded vector metric, when known and shared by every segment.
    pub metric_type: Option<DistanceType>,
    /// Recorded configuration, when known and shared by every segment.
    pub details: Option<Value>,
    /// Whole-file bytes for all registered segments, if their file lists are known.
    pub total_index_bytes: Option<u64>,
    /// Current whole-fragment coverage, if interpretable for every segment.
    pub coverage: Option<IndexScanCoverage>,
    /// All registered segments in manifest order.
    pub segments: Vec<IndexScanSegmentStatistics>,
}

/// Whole-fragment coverage and overlay replay for one index scan.
///
/// See [Dataset::index_scan_statistics] for an example.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IndexScanCoverage {
    /// Number of distinct current fragments referenced by the segments.
    pub covered_fragment_count: u64,
    /// Current fragments outside the union of segment coverage.
    pub uncovered_fragment_ids: RoaringBitmap,
    /// Live rows in the covered fragment union, if their counts are known.
    pub covered_fragment_live_rows: Option<u64>,
    /// Live rows in uncovered fragments, if their counts are known.
    pub uncovered_fragment_live_rows: Option<u64>,
    /// Recorded whole-file base and overlay data bytes in uncovered fragments, if complete.
    pub uncovered_data_bytes: Option<u64>,
    /// Fraction of live rows in covered fragments, if known. Zero for a known empty dataset.
    pub covered_live_row_ratio: Option<f64>,
    /// Distinct physical slots requiring overlay replay in covered fragments, including deleted slots.
    pub overlay_stale_row_slots: u64,
}

/// Statistics for one registered segment of a logical index.
///
/// See [Dataset::index_scan_statistics] for an example.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IndexScanSegmentStatistics {
    /// Segment identity.
    pub uuid: Uuid,
    /// Segment format version.
    pub index_version: i32,
    /// Recorded segment creation time, if present.
    pub created_at: Option<DateTime<Utc>>,
    /// Explicit storage base ID, if set.
    pub base_id: Option<u32>,
    /// Dataset version recorded as the segment's maintenance watermark.
    pub dataset_version_at_last_update: u64,
    /// Recorded whole-file bytes, if the segment has a file list.
    pub size_bytes: Option<u64>,
    /// Recorded index type, if known.
    pub index_type: Option<String>,
    /// Recorded vector metric, if known.
    pub metric_type: Option<DistanceType>,
    /// Recorded configuration, if understood by the reader.
    pub details: Option<Value>,
    /// Current fragment membership, if interpretable.
    pub covered_fragment_ids: Option<RoaringBitmap>,
    /// Live rows in those fragments, if known. Segments may overlap.
    pub covered_fragment_live_rows: Option<u64>,
    /// Deleted slots in those fragments, if known. Segments may overlap.
    pub covered_fragment_deleted_rows: Option<u64>,
    /// Physical IVF layout, when requested and available.
    pub ivf: Option<IndexScanIvfStatistics>,
}

/// Physical entry counts from a materialized IVF segment.
///
/// See [Dataset::index_scan_statistics] for an example.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IndexScanIvfStatistics {
    /// Stored vector entries, which can include deleted rows or multiple entries per row.
    pub indexed_vector_entries: u64,
    /// Number of partitions in the segment.
    pub num_partitions: u64,
    /// Stored vector entries per partition in partition-ID order.
    pub partition_entry_counts: Vec<u64>,
}

impl Dataset {
    /// Inspect all registered segments of a named data index at this snapshot.
    ///
    /// By default, only the pinned manifest may be read. Missing facts remain
    /// absent. This includes current coverage on fragment-reuse snapshots
    /// and legacy JSON coverage that requires verification against index files.
    /// [IndexScanStatisticsOptions::with_segment_stats] permits additional
    /// query-visible metadata and index reads. Both modes are read-only.
    /// JSON details include `target_index_type` and, when recorded,
    /// `target_index_version` for each resolved wrapper.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// use lance::index::{IndexScanStatistics, IndexScanStatisticsOptions};
    ///
    /// async fn inspect(dataset: &Dataset, name: &str) -> Result<IndexScanStatistics> {
    ///     dataset
    ///         .index_scan_statistics(name, IndexScanStatisticsOptions::default())
    ///         .await
    /// }
    /// ```
    pub async fn index_scan_statistics(
        &self,
        index_name: &str,
        options: IndexScanStatisticsOptions,
    ) -> Result<IndexScanStatistics> {
        // The query cache can contain inferred details and remapped bitmaps.
        let registered = read_manifest_indexes(
            self.table_state.object_store.as_ref(),
            &self.table_state.manifest_location,
            self.manifest(),
        )
        .await?;
        let segments: Vec<_> = registered
            .iter()
            .filter(|segment| segment.name == index_name)
            .collect();
        let first = segments
            .first()
            .ok_or_else(|| Error::index_not_found(format!("name={index_name}")))?;
        if is_system_index(first) {
            return Err(Error::not_supported(format!(
                "Index scan statistics are unavailable for system index '{index_name}'"
            )));
        }

        let version = self.manifest().version;
        let context = format!("snapshot {version}, index '{index_name}'");
        let mut uuids = HashSet::with_capacity(segments.len());
        for segment in &segments {
            if !uuids.insert(segment.uuid) {
                return Err(Error::index(format!(
                    "{context}: duplicate segment {}",
                    segment.uuid
                )));
            }
            if !first.declares_same_fields(segment) {
                return Err(Error::index(format!(
                    "{context}: segments {} and {} declare different keyed or carried fields",
                    first.uuid, segment.uuid
                )));
            }
        }

        let has_reuse = registered
            .iter()
            .any(|segment| segment.name == FRAG_REUSE_INDEX_NAME);
        let listing = if options.include_segment_stats && has_reuse {
            Some(self.derived_index_listing().await?)
        } else {
            None
        };
        let mut query_segments = HashMap::with_capacity(segments.len());
        if let Some(listing) = &listing {
            query_segments.extend(
                listing
                    .indices
                    .iter()
                    .filter(|segment| segment.name == index_name)
                    .map(|segment| (segment.uuid, segment)),
            );
        } else if options.include_segment_stats && !has_reuse {
            query_segments.extend(
                segments
                    .iter()
                    .copied()
                    .filter(|segment| index_is_usable(segment))
                    .map(|segment| (segment.uuid, segment)),
            );
        }

        // Fragment reuse requires its own coverage derivation. An empty map
        // for a reuse snapshot leaves all current coverage unknown.
        let no_coverage = HashMap::new();
        let reuse_coverage = has_reuse.then_some(
            if listing
                .as_ref()
                .is_some_and(|listing| listing.unverifiable.contains(index_name))
            {
                &no_coverage
            } else {
                &query_segments
            },
        );
        let mut stats = collect_manifest_statistics(
            self,
            first,
            &segments,
            reuse_coverage,
            options.include_segment_stats,
            &context,
        )
        .await?;
        if options.include_segment_stats {
            let mut layouts = inspect_ivf(
                self,
                index_name,
                segments.iter().filter_map(|stored| {
                    // Legacy inference can fill a name's query metadata. Keep
                    // the persisted reader and family authoritative when present.
                    if stored.index_details.is_some()
                        && (!index_is_usable(stored) || !segment_has_vector_details(stored))
                    {
                        return None;
                    }
                    query_segments
                        .get(&stored.uuid)
                        .map(|query| (*stored, *query))
                }),
            )
            .await?;
            for segment in &mut stats.segments {
                segment.ivf = layouts.remove(&segment.uuid);
            }
        }
        Ok(stats)
    }
}

async fn collect_manifest_statistics(
    dataset: &Dataset,
    first: &IndexMetadata,
    segments: &[&IndexMetadata],
    reuse_coverage: Option<&HashMap<Uuid, &IndexMetadata>>,
    include_segment_stats: bool,
    context: &str,
) -> Result<IndexScanStatistics> {
    let version = dataset.manifest().version;
    let index_name = &first.name;
    let fragments = FragmentFacts::new(dataset)?;
    let overlaid = overlaid_fragments(&dataset.manifest().fragments);
    let mut result_segments = Vec::with_capacity(segments.len());
    let mut identities = Vec::with_capacity(segments.len());
    let mut json_domains = Vec::with_capacity(segments.len());
    let mut has_json = false;
    let mut covered = RoaringBitmap::new();
    let mut stale = HashMap::new();
    for stored in segments {
        let segment_context = format!("{context}, segment {}", stored.uuid);
        let configuration = recorded_configuration(stored)
            .map_err(|error| Error::index(format!("{segment_context}: {error}")))?;
        let is_json = matches!(
            configuration.identity,
            Some(ConfigurationIdentity::Json(..))
        );
        has_json |= is_json;
        identities.push(configuration.identity);
        let mut json_domain = configuration.json_domain;
        let coverage_metadata = match reuse_coverage {
            None => Some(*stored),
            Some(segments) if stored.index_details.is_none() || index_is_usable(stored) => {
                segments.get(&stored.uuid).copied()
            }
            Some(_) => None,
        };
        let coverage_metadata = coverage_metadata.filter(|metadata| {
            !configuration.coverage_blocked
                && index_is_usable(metadata)
                && metadata.fragment_bitmap.as_ref().is_some_and(|bitmap| {
                    !bitmap.is_empty()
                        || (metadata.index_details.is_some()
                            && !empty_fragment_bitmap_has_unknown_coverage(metadata))
                })
        });
        let registered = coverage_metadata
            .and_then(|segment| segment.effective_fragment_bitmap(dataset.fragment_bitmap()));
        let needs_type_recovery = is_json
            && json_domain.is_none()
            && registered.as_ref().is_some_and(|bitmap| !bitmap.is_empty());
        if include_segment_stats
            && needs_type_recovery
            && let Some(details) = &stored.index_details
        {
            let details = IndexDetails(details.clone());
            let plugin = details.get_plugin()?;
            match recover_scalar_query_domain(dataset, stored, &details, plugin).await {
                Ok(recovered) => json_domain = recovered.map(|(_, domain)| domain),
                Err(error) => {
                    log::warn!("Could not verify the value type of {segment_context}: {error}")
                }
            }
        }
        let bitmap = if needs_type_recovery && json_domain.is_none() {
            None
        } else if dataset.manifest().uses_stable_row_ids()
            && stored.legacy_json_coverage_requires_verification()
            && registered.as_ref().is_some_and(|bitmap| !bitmap.is_empty())
        {
            if include_segment_stats && reuse_coverage.is_none() {
                dataset.verified_legacy_json_query_fragments(stored).await
            } else {
                None
            }
        } else {
            registered
        };
        json_domains.push(json_domain);
        let counts = if let (Some(metadata), Some(bitmap)) = (coverage_metadata, &bitmap) {
            covered |= bitmap;
            if !overlaid.is_empty() && !bitmap.is_empty() {
                collect_overlay_stale_rows_for_segment(
                    metadata,
                    &overlaid,
                    &mut stale,
                    dataset.schema(),
                )?;
            }
            Some(fragments.sum_rows(bitmap, &segment_context)?)
        } else {
            None
        };
        let size_bytes = stored
            .files
            .as_ref()
            .map(|files| {
                checked_sum(
                    files.iter().map(|file| Some(file.size_bytes)),
                    &segment_context,
                    "index bytes",
                )
            })
            .transpose()?
            .flatten();
        result_segments.push(IndexScanSegmentStatistics {
            uuid: stored.uuid,
            index_version: stored.index_version,
            created_at: stored.created_at,
            base_id: stored.base_id,
            dataset_version_at_last_update: stored.dataset_version,
            size_bytes,
            index_type: configuration.index_type,
            metric_type: configuration.metric_type,
            details: configuration.details,
            covered_fragment_ids: bitmap,
            covered_fragment_live_rows: counts.and_then(|counts| counts.live),
            covered_fragment_deleted_rows: counts.and_then(|counts| counts.deleted),
            ivf: None,
        });
    }

    let current = dataset.fragment_bitmap();
    let totals = fragments.sum_rows(current, context)?;
    // The scanner routes a logical JSON index only when all of its segments
    // prove one query domain, including any type recovered from storage.
    let agrees_on_json_domain = !has_json
        || segments.len() == 1
        || common_value(json_domains.iter().map(Option::as_ref)).is_some();
    let coverage = if agrees_on_json_domain
        && result_segments
            .iter()
            .all(|segment| segment.covered_fragment_ids.is_some())
    {
        let uncovered = current.as_ref() - &covered;
        let covered_live = fragments.sum_live(&covered, context)?;
        Some(IndexScanCoverage {
            covered_fragment_count: covered.len(),
            covered_fragment_live_rows: covered_live,
            uncovered_fragment_live_rows: fragments.sum_live(&uncovered, context)?,
            uncovered_data_bytes: fragments.sum_bytes(&uncovered, context)?,
            uncovered_fragment_ids: uncovered,
            covered_live_row_ratio: covered_live.zip(totals.live).map(|(covered, total)| {
                if total == 0 {
                    0.0
                } else {
                    covered as f64 / total as f64
                }
            }),
            overlay_stale_row_slots: checked_sum(
                stale.values().map(|bitmap| Some(bitmap.len())),
                context,
                "stale row slots",
            )?
            .unwrap_or_default(),
        })
    } else {
        None
    };

    let type_url = common_value(segments.iter().map(|segment| {
        segment
            .index_details
            .as_ref()
            .map(|details| &details.type_url)
    }))
    .cloned();
    let details = common_value(identities.iter().map(Option::as_ref)).and_then(|_| {
        common_value(
            result_segments
                .iter()
                .map(|segment| segment.details.as_ref()),
        )
        .cloned()
    });
    Ok(IndexScanStatistics {
        dataset_version: version,
        fragment_count: current.len(),
        live_rows: totals.live,
        deleted_rows: totals.deleted,
        total_data_bytes: fragments.sum_bytes(current, context)?,
        index_name: index_name.to_string(),
        keyed_field_ids: first.keyed_fields().to_vec(),
        covering_field_ids: first.covering_fields.clone(),
        type_url,
        index_type: common_value(
            result_segments
                .iter()
                .map(|segment| segment.index_type.as_ref()),
        )
        .cloned(),
        metric_type: common_value(
            result_segments
                .iter()
                .map(|segment| segment.metric_type.as_ref()),
        )
        .copied(),
        details,
        total_index_bytes: checked_sum(
            result_segments.iter().map(|segment| segment.size_bytes),
            context,
            "index bytes",
        )?,
        coverage,
        segments: result_segments,
    })
}

fn common_value<'a, T: PartialEq>(
    mut values: impl Iterator<Item = Option<&'a T>>,
) -> Option<&'a T> {
    let first = values.next().flatten()?;
    values.all(|value| value == Some(first)).then_some(first)
}

#[derive(Default)]
struct RecordedConfiguration {
    index_type: Option<String>,
    metric_type: Option<DistanceType>,
    details: Option<Value>,
    identity: Option<ConfigurationIdentity>,
    coverage_blocked: bool,
    json_domain: Option<QueryDomain>,
}

#[derive(PartialEq)]
enum ConfigurationIdentity {
    Vector,
    Scalar(String, Vec<u8>),
    Json(Option<u32>, Box<Self>),
}

fn recorded_configuration(segment: &IndexMetadata) -> Result<RecordedConfiguration> {
    let Some(details) = segment.index_details.clone().map(IndexDetails) else {
        return Ok(RecordedConfiguration::default());
    };
    if !index_type_is_known(segment) {
        return Ok(RecordedConfiguration::default());
    }
    if details.is_vector() {
        let decoded = decode_vector_details(&details.0);
        if unsupported_index_version(segment).is_some() {
            return Ok(RecordedConfiguration {
                metric_type: decoded
                    .ok()
                    .flatten()
                    .as_ref()
                    .and_then(recorded_vector_metric),
                ..Default::default()
            });
        }
        let Some(decoded) = decoded? else {
            return Ok(RecordedConfiguration {
                index_type: Some("Vector".to_string()),
                ..Default::default()
            });
        };
        Ok(RecordedConfiguration {
            index_type: Some(decoded_vector_index_type(&decoded)),
            metric_type: recorded_vector_metric(&decoded),
            details: Some(vector_details_value(decoded)?),
            identity: Some(ConfigurationIdentity::Vector),
            ..Default::default()
        })
    } else {
        let plugin = details.get_plugin()?;
        let index_type = Some(plugin.name().to_string());
        if unsupported_index_version(segment).is_some() {
            return Ok(RecordedConfiguration {
                index_type,
                ..Default::default()
            });
        }
        let mut recorded = RecordedConfiguration {
            index_type,
            ..Default::default()
        };
        if let Some(configuration) = scalar_configuration(
            &details.0,
            plugin,
            segment.index_version as u32,
            &segment.name,
        )? {
            recorded.identity = Some(configuration.identity);
            recorded.details = Some(configuration.details);
            recorded.coverage_blocked = configuration.coverage_blocked;
            recorded.json_domain = configuration.json_domain;
        } else {
            recorded.coverage_blocked = true;
        }
        Ok(recorded)
    }
}

struct ScalarConfiguration {
    identity: ConfigurationIdentity,
    details: Value,
    coverage_blocked: bool,
    json_domain: Option<QueryDomain>,
}

fn scalar_configuration(
    details: &prost_types::Any,
    plugin: &dyn ScalarIndexPlugin,
    index_version: u32,
    index_name: &str,
) -> Result<Option<ScalarConfiguration>> {
    let payload = details.value.as_slice();
    // The registry resolves aliases by message name. Distinct registered
    // messages can expose the same display name.
    let message_name = details
        .type_url
        .rsplit('.')
        .next()
        .unwrap_or(&details.type_url)
        .to_ascii_lowercase();
    let (identity, json) = match message_name.as_str() {
        "jsonindexdetails" => {
            let wrapper = JsonIndexDetails::decode(payload)?;
            let Some((target, target_plugin)) =
                wrapper.resolve_target(&SCALAR_INDEX_PLUGIN_REGISTRY)?
            else {
                return Ok(None);
            };
            let Some(configuration) = scalar_configuration(
                target,
                target_plugin,
                wrapper.effective_target_version(target),
                index_name,
            )?
            else {
                return Ok(None);
            };
            // A nested wrapper may itself need recovery before the outer
            // parser can establish a domain.
            let nested_needs_recovery =
                matches!(configuration.identity, ConfigurationIdentity::Json(..))
                    && configuration.json_domain.is_none();
            let coverage_blocked = configuration.coverage_blocked
                || !wrapper.target_allows_query(index_name, target, target_plugin);
            let json_domain = (!coverage_blocked && !nested_needs_recovery)
                .then(|| {
                    wrapper.recorded_domain(target, &SCALAR_INDEX_PLUGIN_REGISTRY, index_version)
                })
                .flatten();
            let mut details = wrapper.render_details(configuration.details);
            details["target_index_type"] = Value::from(target_plugin.name());
            if let Some(version) = wrapper.target_index_version {
                details["target_index_version"] = Value::from(version);
            }
            return Ok(Some(ScalarConfiguration {
                identity: ConfigurationIdentity::Json(
                    wrapper.target_index_version,
                    Box::new(configuration.identity),
                ),
                details,
                coverage_blocked,
                json_domain,
            }));
        }
        "zonemapindexdetails" => {
            let settings = ZoneMapIndexDetails::decode(payload)?;
            let mut json = serde_json::Map::new();
            for (key, value) in [
                ("rows_per_zone", settings.rows_per_zone.map(Value::from)),
                ("use_seeds", settings.use_seeds.map(Value::from)),
                ("has_null_bitmap", settings.has_null_bitmap.map(Value::from)),
            ] {
                if let Some(value) = value {
                    json.insert(key.to_string(), value);
                }
            }
            (
                ConfigurationIdentity::Scalar(message_name, settings.encode_to_vec()),
                Value::Object(json),
            )
        }
        name => {
            // Decode settings with known schemas. A display renderer can omit
            // settings, so opaque plugin payloads must also match byte for byte.
            let normalized = match name {
                "btreeindexdetails" => BTreeIndexDetails::decode(payload)?.encode_to_vec(),
                "invertedindexdetails" => InvertedIndexDetails::decode(payload)?.encode_to_vec(),
                _ => {
                    <()>::decode(payload)?;
                    payload.to_vec()
                }
            };
            (
                ConfigurationIdentity::Scalar(message_name, normalized),
                plugin.details_as_json(details)?,
            )
        }
    };
    Ok(Some(ScalarConfiguration {
        identity,
        details: json,
        coverage_blocked: false,
        json_domain: None,
    }))
}

async fn inspect_ivf<'a>(
    dataset: &Dataset,
    name: &str,
    metadata: impl Iterator<Item = (&'a IndexMetadata, &'a IndexMetadata)>,
) -> Result<HashMap<Uuid, IndexScanIvfStatistics>> {
    let mut layouts = HashMap::with_capacity(metadata.size_hint().0);
    for (stored, query) in metadata {
        if !index_is_usable(query) || is_definition_only_segment(stored) {
            continue;
        }
        let open_error =
            |error| Error::index(format!("Index '{name}', segment {}: {error}", stored.uuid));
        let is_vector = if stored.index_details.is_some() || stored.files.is_some() {
            segment_has_vector_details(stored)
        } else {
            legacy_segment_is_vector(dataset, stored)
                .await
                .map_err(open_error)?
        };
        if !is_vector {
            continue;
        }
        let field = stored.keyed_field().ok_or_else(|| {
            Error::index(format!(
                "Index '{name}', segment {}: expected a single keyed vector field",
                stored.uuid
            ))
        })?;
        let column = dataset.schema().field_path(field).map_err(open_error)?;
        let handle = dataset
            .open_vector_index_from_metadata(&column, query, &NoOpMetricsCollector)
            .await
            .map_err(open_error)?;
        let Some(partition_entry_counts) =
            handle.ivf_partition_entry_counts().map_err(open_error)?
        else {
            continue;
        };
        let indexed_vector_entries =
            partition_entry_counts
                .iter()
                .try_fold(0u64, |total, &count| {
                    total.checked_add(count).ok_or_else(|| {
                        Error::index(format!(
                            "Index '{name}', segment {}: stored vector entry count exceeds u64",
                            stored.uuid
                        ))
                    })
                })?;
        layouts.insert(
            stored.uuid,
            IndexScanIvfStatistics {
                indexed_vector_entries,
                num_partitions: partition_entry_counts.len() as u64,
                partition_entry_counts,
            },
        );
    }
    Ok(layouts)
}

async fn legacy_segment_is_vector(dataset: &Dataset, segment: &IndexMetadata) -> Result<bool> {
    if dataset
        .legacy_index_has_file(segment, &[INDEX_FILE_NAME])
        .await?
    {
        return Ok(true);
    }
    let is_vector_column = segment
        .keyed_field()
        .and_then(|field_id| dataset.schema().field_by_id(field_id))
        .is_some_and(|field| infer_vector_element_type(&field.data_type()).is_ok());
    if !is_vector_column {
        return Ok(false);
    }
    // An unavailable vector file is ambiguous. Only a positive scalar marker
    // can rule out IVF on a vector column without attempting the query open.
    Ok(!dataset
        .legacy_index_has_file(
            segment,
            &[
                BTREE_LOOKUP_NAME,
                BITMAP_LOOKUP_NAME,
                ZONEMAP_INDEX_FILENAME,
                BLOOMFILTER_FILENAME,
                INVERT_LIST_FILE,
                METADATA_FILE,
                POSTINGS_FILENAME,
            ],
        )
        .await?)
}

#[derive(Clone, Copy)]
struct Rows {
    live: Option<u64>,
    deleted: Option<u64>,
}

struct FragmentFact {
    rows: Rows,
    data_bytes: Option<u64>,
}

struct FragmentFacts {
    by_id: HashMap<u32, FragmentFact>,
}

impl FragmentFacts {
    fn new(dataset: &Dataset) -> Result<Self> {
        let manifest = dataset.manifest();
        let trust_rows = manifest.writer_version.is_some();
        let mut by_id = HashMap::with_capacity(manifest.fragments.len());
        for fragment in manifest.fragments.iter() {
            let version = manifest.version;
            let id = fragment.id;
            let fragment_id = u32::try_from(id).map_err(|_| {
                Error::index(format!("snapshot {version}: fragment ID {id} exceeds u32"))
            })?;
            let deleted = fragment
                .deletion_file
                .as_ref()
                .map_or(Some(0), |file| file.num_deleted_rows)
                .map(|rows| rows as u64);
            let physical = fragment
                .physical_rows
                .filter(|_| trust_rows)
                .map(|rows| rows as u64);
            let live = physical
                .zip(deleted)
                .map(|(physical, deleted)| {
                    physical.checked_sub(deleted).ok_or_else(|| {
                        Error::index(format!(
                            "snapshot {version}, fragment {id}: deleted rows ({deleted}) exceed physical rows ({physical})"
                        ))
                    })
                })
                .transpose()?;
            let data_bytes =
                fragment_bytes(fragment, format_args!("snapshot {version}, fragment {id}"))?;
            by_id.insert(
                fragment_id,
                FragmentFact {
                    rows: Rows { live, deleted },
                    data_bytes,
                },
            );
        }
        Ok(Self { by_id })
    }

    fn sum_rows(&self, ids: &RoaringBitmap, context: &str) -> Result<Rows> {
        Ok(Rows {
            live: self.sum_live(ids, context)?,
            deleted: checked_sum(
                ids.iter().map(|id| self.by_id[&id].rows.deleted),
                context,
                "deleted rows",
            )?,
        })
    }

    fn sum_live(&self, ids: &RoaringBitmap, context: &str) -> Result<Option<u64>> {
        checked_sum(
            ids.iter().map(|id| self.by_id[&id].rows.live),
            context,
            "live rows",
        )
    }

    fn sum_bytes(&self, ids: &RoaringBitmap, context: &str) -> Result<Option<u64>> {
        checked_sum(
            ids.iter().map(|id| self.by_id[&id].data_bytes),
            context,
            "data bytes",
        )
    }
}

fn fragment_bytes(fragment: &Fragment, context: impl Display) -> Result<Option<u64>> {
    if fragment.overlays.is_empty() && fragment.files.len() <= 1 {
        return Ok(fragment
            .files
            .first()
            .map_or(Some(0), |file| file.file_size_bytes.get().map(u64::from)));
    }
    let mut files = HashMap::new();
    for file in fragment.referenced_lance_files() {
        let size = file.file_size_bytes.get().map(u64::from);
        let existing = files
            .entry((file.base_id, file.path.as_str()))
            .or_insert(size);
        if existing.is_none() {
            *existing = size;
        } else if size.is_some() && *existing != size {
            return Err(Error::index(format!(
                "{context}: inconsistent recorded sizes for {}",
                file.path
            )));
        }
    }
    checked_sum(files.into_values(), context, "data bytes")
}

fn checked_sum(
    values: impl IntoIterator<Item = Option<u64>>,
    context: impl Display,
    units: &str,
) -> Result<Option<u64>> {
    let mut total = 0u64;
    let mut complete = true;
    for value in values {
        if let Some(value) = value {
            total = total.checked_add(value).ok_or_else(|| {
                Error::index(format!(
                    "{context}: overflow summing {units} ({total} + {value})"
                ))
            })?;
        } else {
            complete = false;
        }
    }
    Ok(complete.then_some(total))
}
