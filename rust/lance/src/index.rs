// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Secondary Index
//!

use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use arrow_schema::DataType;
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use futures::{FutureExt, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::datatypes::Field;
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::utils::parse::parse_env_as_bool;
use lance_core::utils::tracing::{
    IO_TYPE_OPEN_FRAG_REUSE, IO_TYPE_OPEN_MEM_WAL, IO_TYPE_OPEN_VECTOR, TRACE_IO_EVENTS,
};
use lance_file::reader::FileReaderOptions;
use lance_file::versions::v1::reader::FileReader as V1FileReader;
use lance_index::INDEX_METADATA_SCHEMA_KEY;
pub use lance_index::IndexParams;
use lance_index::frag_reuse::{
    CompactFragReuseIndex, CompactFragReuseIndexHandle, FRAG_REUSE_INDEX_NAME,
};
use lance_index::mem_wal::{MEM_WAL_INDEX_NAME, MemWalIndex, MemWalIndexHandle};
use lance_index::optimize::OptimizeOptions;
use lance_index::pb::index::Implementation;
pub use lance_index::progress::{IndexBuildProgress, NoopIndexBuildProgress};
use lance_index::scalar::expression::{IndexInformationProvider, MultiQueryParser};
use lance_index::scalar::inverted::{InvertedIndex, InvertedIndexPlugin};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::{TrainingCriteria, TrainingOrdering};
use lance_index::scalar::{CreatedIndex, ScalarIndex, index_files_to_table, table_files_to_index};
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::flat::index::{FlatBinQuantizer, FlatIndex, FlatQuantizer};
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::quantizer::Quantization;
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_index::{
    FtsPrewarmDiagnostics, FtsPrewarmOptions, FtsPrewarmResult, FtsPrewarmSegmentStatus,
    INDEX_FILE_NAME, Index, IndexType, PrewarmOptions, pb, vector::VectorIndex,
};
use lance_index::{
    IndexCriteria, is_system_index,
    metrics::{MetricsCollector, NoOpMetricsCollector},
    registry::display_type_from_url,
    scalar::btree::BTREE_LOOKUP_NAME,
};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::traits::Reader;
use lance_io::utils::{
    CachedFileSize, read_last_block, read_message, read_message_from_buf, read_metadata_offset,
    read_version,
};
use lance_table::format::{DataFile, Fragment, SelfDescribingFileReader};
use lance_table::format::{IndexFile, IndexMetadata, list_index_files_with_sizes};
use lance_table::io::manifest::read_manifest_indexes;
use roaring::RoaringBitmap;
use scalar::index_matches_criteria;
use serde_json::json;
use tracing::{info, instrument, warn};
use uuid::Uuid;
use vector::details::{
    derive_vector_index_type, infer_missing_vector_details, needs_vector_details_inference,
    vector_details_as_json,
};
pub(crate) use vector::details::{vector_index_details, vector_index_details_default};
use vector::ivf::v2::{IVFIndex, IvfStateEntryBox};
use vector::utils::get_vector_type;

mod api;
pub(crate) mod append;
pub(crate) mod covering;
mod create;
pub mod frag_reuse;
pub mod mem_wal;
pub mod prefilter;
pub mod scalar;
pub(crate) mod scalar_logical;
pub mod vector;

use self::append::merge_indices;
use self::vector::remap_vector_index;
use crate::dataset::index::LanceIndexStoreExt;
use crate::dataset::optimize::RemappedIndex;
use crate::dataset::optimize::remapping::RemapResult;
use crate::dataset::transaction::{Operation, ReadVersionState, Transaction, TransactionBuilder};
pub use crate::index::api::{DatasetIndexExt, IndexSegment, IntoIndexSegment};
use crate::index::frag_reuse::{load_frag_reuse_index_details, open_frag_reuse_index};
use crate::index::mem_wal::open_mem_wal_index;
pub use crate::index::prefilter::{FilterLoader, PreFilter};
use crate::index::scalar::{IndexDetails, fetch_index_details, load_training_data};
pub use crate::index::vector::{LogicalIvfView, LogicalVectorIndex};
use crate::session::index_caches::{FragReuseIndexKey, IndexMetadataKey, write_index_identity};
use crate::{Error, Result, dataset::Dataset};
pub use create::CreateIndexBuilder;
pub use lance_index::IndexDescription;

fn validate_segment_metadata(index_name: &str, segments: &[IndexMetadata]) -> Result<()> {
    if segments.is_empty() {
        return Err(Error::invalid_input(
            "CreateIndex: at least one index segment is required".to_string(),
        ));
    }

    let mut seen_segment_ids = HashSet::with_capacity(segments.len());
    let mut covered_fragments = RoaringBitmap::new();
    for segment in segments {
        if !seen_segment_ids.insert(segment.uuid) {
            return Err(Error::invalid_input(format!(
                "CreateIndex: duplicate segment uuid {} for index '{}'",
                segment.uuid, index_name
            )));
        }
        let fragment_bitmap = segment.fragment_bitmap.as_ref().ok_or_else(|| {
            Error::invalid_input(format!(
                "CreateIndex: segment {} is missing fragment coverage",
                segment.uuid
            ))
        })?;
        if !covered_fragments.is_disjoint(fragment_bitmap) {
            return Err(Error::invalid_input(format!(
                "CreateIndex: overlapping fragment coverage in segment set for index '{}'",
                index_name
            )));
        }
        covered_fragments |= fragment_bitmap.clone();
    }

    Ok(())
}

pub(crate) fn collect_subtree_field_ids(field: &Field, field_ids: &mut HashSet<i32>) {
    field_ids.insert(field.id);
    for child in &field.children {
        collect_subtree_field_ids(child, field_ids);
    }
}

/// Stable identity fields for a physical data file.
///
/// This mirrors transaction rewrite validation, additionally resolves a
/// registered base to its physical binding, and deliberately excludes
/// `file_size_bytes`, which is a mutable cache rather than file identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhysicalBaseBinding<'a> {
    Primary,
    Registered {
        path: &'a str,
        is_dataset_root: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PhysicalDataFileIdentity<'a> {
    base_id: Option<u32>,
    base_binding: PhysicalBaseBinding<'a>,
    path: &'a str,
    // Deliberately NOT the file's `fields` list. This identity is built per field, but
    // `fields` is file-wide, so a tombstone of any *other* field in a shared file (an
    // unrelated `RewriteColumns` turns its id negative) would change every field's
    // identity and prune coverage for indexes whose data did not move. A tombstone of
    // *this* field is still caught: its id stops matching and it drops out of the map
    // `fragment_field_files` builds. `column_indices` covers an actual layout change.
    //
    // Upstream cannot hit this without a covering producer -- with no payload written,
    // two indexes never share a file with different field sets -- which is why the guard
    // held until this commit introduced one.
    column_indices: &'a [i32],
    file_major_version: u32,
    file_minor_version: u32,
}

impl<'a> PhysicalDataFileIdentity<'a> {
    fn try_new(dataset: &'a Dataset, file: &'a DataFile) -> Option<Self> {
        let base_binding = match file.base_id {
            Some(base_id) => {
                let base = dataset.manifest.base_paths.get(&base_id)?;
                PhysicalBaseBinding::Registered {
                    path: &base.path,
                    is_dataset_root: base.is_dataset_root,
                }
            }
            None => PhysicalBaseBinding::Primary,
        };
        Some(Self {
            base_id: file.base_id,
            base_binding,
            path: &file.path,
            column_indices: file.column_indices.as_ref(),
            file_major_version: file.file_major_version,
            file_minor_version: file.file_minor_version,
        })
    }
}

fn fragment_field_files<'a>(
    dataset: &'a Dataset,
    fragment: &'a Fragment,
    indexed_field_ids: &HashSet<i32>,
) -> Option<HashMap<i32, PhysicalDataFileIdentity<'a>>> {
    fragment
        .files
        .iter()
        .flat_map(|file| {
            file.fields
                .iter()
                .filter(|field_id| indexed_field_ids.contains(field_id))
                .map(|field_id| {
                    PhysicalDataFileIdentity::try_new(dataset, file)
                        .map(|identity| (*field_id, identity))
                })
        })
        .collect()
}

/// Resolve the field ids a segment's staleness check must consider: the subtree of
/// every field the segment declares, keyed and carried alike (see
/// [`IndexSegment::fields`] and [`IndexSegment::covering_fields`]). A covered
/// segment's carried columns can go stale independently of its keyed column, so
/// checking only the keyed subtree would leave a fragment covered after a carried
/// column was rewritten, and the segment would answer with the obsolete value.
fn segment_indexed_field_ids(dataset: &Dataset, segment: &IndexSegment) -> Result<HashSet<i32>> {
    let mut indexed_field_ids = HashSet::new();
    for field_id in segment.fields() {
        let field = dataset.schema().field_by_id(*field_id).ok_or_else(|| {
            Error::invalid_input(format!(
                "CreateIndex: field id {field_id} does not exist in the current schema"
            ))
        })?;
        collect_subtree_field_ids(field, &mut indexed_field_ids);
    }
    Ok(indexed_field_ids)
}

async fn prune_stale_segment_coverage(
    dataset: &Dataset,
    segments: &mut [IndexSegment],
    prune_historically_missing: bool,
    prune_newer_overlays: bool,
) -> Result<()> {
    let current_fragments = dataset
        .fragments()
        .iter()
        .map(|fragment| (fragment.id as u32, fragment))
        .collect::<HashMap<_, _>>();
    let historical_versions = segments
        .iter()
        .map(IndexSegment::dataset_version)
        .filter(|version| *version < dataset.manifest.version)
        .collect::<HashSet<_>>();

    for version in historical_versions {
        let historical = dataset.checkout_version(version).await.map_err(|error| {
            Error::invalid_input(format!(
                "CreateIndex: cannot validate segment coverage built at dataset version {version}: {error}"
            ))
        })?;
        let historical_fragments = historical
            .fragments()
            .iter()
            .map(|fragment| (fragment.id as u32, fragment))
            .collect::<HashMap<_, _>>();

        for segment in segments
            .iter_mut()
            .filter(|segment| segment.dataset_version() == version)
        {
            // `segment_indexed_field_ids` already walks every entry of `segment.fields()`,
            // keyed and carried alike -- `covering_fields` is always the trailing slice of
            // `fields` -- so no separate expansion over `covering_fields()` is needed here.
            let relevant_field_ids = segment_indexed_field_ids(dataset, segment)?;
            for included_id in segment.covering_fields() {
                // No exemption for a field the segment is also keyed on: carried refine
                // vectors declare the indexed column here, and index storage holds that copy
                // under its build-time name, so a rename leaves storage the rebuild paths
                // cannot resolve. The `alter_columns` preflight (`schema_evolution`) and the
                // commit-boundary guard (`reject_covered_field_subtree_change`) refuse it for
                // the same reason; all three must agree or one of them forbids what another
                // accepts.
                //
                // A covered field whose subtree changed between the segment's build version
                // and now -- e.g. a struct that gained a child via `add_columns` (which
                // commits while this segment is still uncommitted, so the Merge-commit guard
                // never saw it) -- makes the segment's stored covering payload type-
                // incompatible with the current schema for *every* row it covers. No
                // per-fragment file-path prune below can repair a global type change, so
                // reject the commit; the segment must be rebuilt against the current schema.
                if Transaction::covered_field_subtree_changed(
                    historical.schema(),
                    dataset.schema(),
                    *included_id,
                ) {
                    return Err(Error::invalid_input(format!(
                        "CreateIndex: covered (included) field id {included_id} changed its \
                         subtree between the segment's build version {version} and the current \
                         dataset version {}; the segment's stored covering payload no longer \
                         matches the schema. Rebuild the segment against the current schema.",
                        dataset.manifest.version
                    )));
                }
            }
            let stale_fragments = segment
                .fragment_bitmap()
                .iter()
                .filter(|fragment_id| {
                    let Some(historical_fragment) = historical_fragments.get(fragment_id) else {
                        return prune_historically_missing;
                    };
                    let Some(current_fragment) = current_fragments.get(fragment_id) else {
                        return true;
                    };
                    let historical_files =
                        fragment_field_files(&historical, historical_fragment, &relevant_field_ids);
                    let current_files =
                        fragment_field_files(dataset, current_fragment, &relevant_field_ids);
                    let changed_files =
                        historical_files.is_none() || historical_files != current_files;
                    let changed_overlays = prune_newer_overlays
                        && current_fragment.overlays.iter().any(|overlay| {
                            overlay.committed_version > version
                                && overlay
                                    .data_file
                                    .fields
                                    .iter()
                                    .any(|field_id| relevant_field_ids.contains(field_id))
                        });
                    changed_files || changed_overlays
                })
                .collect::<Vec<_>>();
            for fragment_id in stale_fragments {
                segment.fragment_bitmap_mut().remove(fragment_id);
            }
        }
    }
    Ok(())
}

pub(crate) async fn build_index_metadata_from_segments(
    dataset: &Dataset,
    index_name: &str,
    field_id: i32,
    mut segments: Vec<IndexSegment>,
) -> Result<Vec<IndexMetadata>> {
    if segments.is_empty() {
        return Err(Error::invalid_input(
            "CreateIndex: at least one index segment is required".to_string(),
        ));
    }

    let mut seen_segment_ids = HashSet::with_capacity(segments.len());
    let mut covered_fragments = RoaringBitmap::new();
    // One logical index needs one declaration. The per-segment rules below only
    // pin the *keyed* field and its count, so segments that disagree on the
    // carried columns -- `fields = [k, a]` beside `fields = [k]`, say -- each pass
    // individually. Committing that pair produces an index whose own description
    // path refuses it: `IndexDescriptionImpl::try_new` requires `fields` to be
    // identical across segments, so `describe_indices` would error on metadata
    // this function just wrote. Same rule, and same reason, as
    // `merge_existing_index_segments`.
    let expected_fields = segments[0].fields().to_vec();
    let expected_covering_fields = segments[0].covering_fields().to_vec();
    for segment in &segments {
        if segment.dataset_version() > dataset.manifest.version {
            return Err(Error::invalid_input(format!(
                "CreateIndex: segment {} was built at future dataset version {} (current version {})",
                segment.uuid(),
                segment.dataset_version(),
                dataset.manifest.version
            )));
        }
        if segment.keyed_field() != Some(field_id) {
            return Err(Error::invalid_input(format!(
                "CreateIndex: segment {} was built for fields {:?} (carried {:?}), \
                 expected keyed field [{}]",
                segment.uuid(),
                segment.fields(),
                segment.covering_fields(),
                field_id
            )));
        }
        // Cardinality alone does not prove `covering_fields` is really the
        // trailing slice of `fields`; delegate that to the same rule a
        // committed `IndexMetadata` is held to, rather than re-deriving it.
        IndexMetadata {
            uuid: segment.uuid(),
            fields: segment.fields().to_vec(),
            covering_fields: segment.covering_fields().to_vec(),
            name: index_name.to_string(),
            dataset_version: segment.dataset_version(),
            fragment_bitmap: None,
            index_details: None,
            index_version: segment.index_version(),
            created_at: None,
            base_id: None,
            files: None,
        }
        .validate_covering_fields()?;
        if segment.fields() != expected_fields.as_slice()
            || segment.covering_fields() != expected_covering_fields.as_slice()
        {
            return Err(Error::invalid_input(format!(
                "CreateIndex: segment {} declares fields {:?} (carried {:?}) but index \
                 '{}' declares fields {:?} (carried {:?}); every segment of one index \
                 must declare the same columns",
                segment.uuid(),
                segment.fields(),
                segment.covering_fields(),
                index_name,
                expected_fields,
                expected_covering_fields,
            )));
        }
        if !seen_segment_ids.insert(segment.uuid()) {
            return Err(Error::invalid_input(format!(
                "CreateIndex: duplicate segment uuid {} for index '{}'",
                segment.uuid(),
                index_name
            )));
        }
        if !covered_fragments.is_disjoint(segment.fragment_bitmap()) {
            return Err(Error::invalid_input(format!(
                "CreateIndex: overlapping fragment coverage in segment set for index '{}'",
                index_name
            )));
        }
        covered_fragments |= segment.fragment_bitmap().clone();
    }

    // The declaration is a dependency list, so validate it against the dataset schema and
    // prune stale fragment coverage before committing. Physical covering payload is not a
    // commit precondition: readers independently verify every segment's storage capability
    // and fall back to a base-table take for any declared field that is absent.
    prune_stale_segment_coverage(dataset, &mut segments, false, false).await?;

    let new_indices = futures::stream::iter(segments.into_iter().map(|segment| async move {
        let (
            uuid,
            fragment_bitmap,
            fields,
            covering_fields,
            index_details,
            index_version,
            dataset_version,
        ) = segment.into_parts();
        let is_inverted_index = index_details.type_url.ends_with("InvertedIndexDetails");
        if is_inverted_index {
            let metadata = IndexMetadata {
                uuid,
                name: index_name.to_string(),
                fields: fields.clone(),
                covering_fields: covering_fields.clone(),
                dataset_version,
                fragment_bitmap: Some(fragment_bitmap.clone()),
                index_details: Some(index_details.clone()),
                index_version,
                created_at: Some(chrono::Utc::now()),
                base_id: None,
                files: None,
            };
            crate::index::scalar::inverted::finalize_segment_files_if_needed(dataset, &metadata)
                .await?;
        }
        let index_dir = dataset.indices_dir().join(uuid.to_string());
        let mut files = list_index_files_with_sizes(&dataset.object_store, &index_dir).await?;
        if is_inverted_index {
            retain_committed_inverted_files(&mut files);
        }
        Ok::<_, Error>(IndexMetadata {
            uuid,
            name: index_name.to_string(),
            fields,
            covering_fields,
            dataset_version,
            fragment_bitmap: Some(fragment_bitmap),
            index_details: Some(index_details),
            index_version,
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            files: Some(files),
        })
    }))
    .buffered(dataset.object_store.io_parallelism())
    .try_collect::<Vec<_>>()
    .await?;

    Ok(new_indices)
}

fn retain_committed_inverted_files(files: &mut Vec<IndexFile>) {
    files.retain(|file| !file.path.starts_with("staging/"));
}

async fn prewarm_opened_index(
    index: Arc<dyn Index>,
    options: Option<&PrewarmOptions>,
) -> Result<()> {
    match options {
        None => index.prewarm().await,
        Some(PrewarmOptions::Fts(fts_options)) => {
            let inverted = index
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "FTS prewarm options are only supported for inverted indices, got {:?}",
                        index.index_type()
                    ))
                })?;
            inverted.prewarm_with_options(fts_options).await
        }
        Some(_) => Err(Error::not_supported(
            "unsupported prewarm options for this lance version".to_owned(),
        )),
    }
}

struct SegmentPrewarmResult {
    partition_count: usize,
}

struct OpenedSegmentPrewarmResult {
    index_uuid: Uuid,
    index: Arc<dyn Index>,
    partition_count: usize,
}

async fn prewarm_opened_index_result(
    index: Arc<dyn Index>,
    options: &PrewarmOptions,
) -> Result<SegmentPrewarmResult> {
    match options {
        PrewarmOptions::Fts(fts_options) => {
            let inverted = index
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "FTS prewarm options are only supported for inverted indices, got {:?}",
                        index.index_type()
                    ))
                })?;
            inverted.prewarm_with_options_result(fts_options).await?;
            Ok(SegmentPrewarmResult {
                partition_count: inverted.partition_count(),
            })
        }
        _ => Err(Error::not_supported(
            "unsupported prewarm options for this lance version".to_owned(),
        )),
    }
}

async fn aggregate_fts_prewarm_results(
    dataset: &Dataset,
    segment_results: Vec<OpenedSegmentPrewarmResult>,
    options: Option<&PrewarmOptions>,
) -> Result<FtsPrewarmResult> {
    let partition_count = segment_results
        .iter()
        .map(|result| result.partition_count)
        .sum();
    let mut failing_segments = Vec::new();
    let mut failing_partitions = Vec::new();
    let mut best_effort = false;

    for segment in segment_results {
        let segment_id = segment.index_uuid.to_string();
        let prewarmed_inverted = segment.index.as_any().downcast_ref::<InvertedIndex>();
        let Some(fts_options) = (match options {
            Some(PrewarmOptions::Fts(fts_options)) => Some(fts_options.clone()),
            None if prewarmed_inverted.is_some() => Some(FtsPrewarmOptions::default()),
            _ => None,
        }) else {
            continue;
        };
        best_effort |= fts_options.mode.is_best_effort();

        let cached_index =
            scalar::cached_scalar_index_container(dataset, &segment.index_uuid).await;
        let cached_inverted = cached_index
            .as_ref()
            .and_then(|index| index.as_any().downcast_ref::<InvertedIndex>());
        let container_resident = cached_inverted.is_some();
        let container_matches_prewarmed = match (prewarmed_inverted, cached_inverted) {
            (Some(prewarmed), Some(cached)) => std::ptr::addr_eq(prewarmed, cached),
            _ => false,
        };

        if !container_resident || !container_matches_prewarmed {
            failing_segments.push(FtsPrewarmSegmentStatus {
                segment_id: segment_id.clone(),
                scalar_index_container_resident: container_resident,
                scalar_index_container_matches_prewarmed: container_matches_prewarmed,
            });
        }

        if let Some(cached_inverted) = cached_inverted
            && let Some(mut diagnostics) = cached_inverted
                .prewarm_residency_result(fts_options.with_position)
                .await
                .diagnostics
        {
            failing_segments.append(&mut diagnostics.failing_segments);
            failing_partitions.extend(diagnostics.failing_partitions.drain(..).map(
                |mut partition| {
                    partition.segment_id = Some(segment_id.clone());
                    partition
                },
            ));
        }
    }

    let diagnostics = FtsPrewarmDiagnostics {
        partition_count,
        failing_segments,
        failing_partitions,
    };

    if diagnostics.fully_resident() {
        Ok(FtsPrewarmResult::fully_resident())
    } else if best_effort {
        Ok(FtsPrewarmResult::partial(diagnostics))
    } else {
        Err(Error::internal(diagnostics.to_string()))
    }
}

fn total_index_segment_size_bytes(indices: &[IndexMetadata]) -> Option<u64> {
    let mut total = 0u64;
    for index_meta in indices {
        total += index_meta.total_size_bytes()?;
    }
    Some(total)
}

fn cache_size_delta(after: usize, before: usize) -> i64 {
    after.saturating_sub(before) as i64 - before.saturating_sub(after) as i64
}

fn prewarm_options_fields(options: Option<&PrewarmOptions>) -> (&'static str, bool) {
    match options {
        None => ("default", false),
        Some(PrewarmOptions::Fts(fts_options)) => ("fts", fts_options.with_position),
        Some(_) => ("unsupported", false),
    }
}

async fn prewarm_index_segments_by_metadata(
    dataset: &Dataset,
    name: &str,
    indices: Vec<IndexMetadata>,
    options: Option<&PrewarmOptions>,
    available_segment_count: usize,
    requested_segment_count: Option<usize>,
) -> Result<FtsPrewarmResult> {
    let request_started = Instant::now();
    let selected_segment_count = indices.len();
    let selected_size_bytes = total_index_segment_size_bytes(&indices);
    let (prewarm_options, fts_with_position) = prewarm_options_fields(options);
    let cache_stats_before = dataset.session.index_cache_stats().await;
    info!(
        index_name = name,
        selected_segment_count,
        available_segment_count,
        requested_segment_count = requested_segment_count.unwrap_or(0),
        segment_filter = requested_segment_count.is_some(),
        selected_size_bytes = selected_size_bytes.unwrap_or(0),
        selected_size_bytes_known = selected_size_bytes.is_some(),
        index_cache_entries_before = cache_stats_before.num_entries,
        index_cache_size_bytes_before = cache_stats_before.size_bytes,
        prewarm_options,
        fts_with_position,
        "prewarm index segments started"
    );

    let result = futures::future::try_join_all(indices.into_iter().map(|index_meta| async move {
        let index_uuid = index_meta.uuid;
        let size_bytes = index_meta.total_size_bytes();
        let fragment_count = index_meta
            .fragment_bitmap
            .as_ref()
            .map(|bitmap| bitmap.len());
        let segment_started = Instant::now();
        info!(
            index_name = name,
            %index_uuid,
            index_version = index_meta.index_version,
            dataset_version = index_meta.dataset_version,
            fragment_count = fragment_count.unwrap_or(0),
            fragment_count_known = fragment_count.is_some(),
            size_bytes = size_bytes.unwrap_or(0),
            size_bytes_known = size_bytes.is_some(),
            "prewarm index segment started"
        );

        let index = match dataset
            .open_generic_index(name, &index_uuid, &NoOpMetricsCollector)
            .await
        {
            Ok(index) => {
                info!(
                    index_name = name,
                    %index_uuid,
                    index_type = ?index.index_type(),
                    elapsed_ms = segment_started.elapsed().as_millis() as u64,
                    "opened index segment for prewarm"
                );
                index
            }
            Err(err) => {
                warn!(
                    index_name = name,
                    %index_uuid,
                    error = %err,
                    elapsed_ms = segment_started.elapsed().as_millis() as u64,
                    "failed to open index segment for prewarm"
                );
                return Err(err);
            }
        };

        let segment_result = match options {
            Some(options) => match prewarm_opened_index_result(index.clone(), options).await {
                Ok(result) => result,
                Err(err) => {
                    warn!(
                        index_name = name,
                        %index_uuid,
                        error = %err,
                        elapsed_ms = segment_started.elapsed().as_millis() as u64,
                        "prewarm index segment failed"
                    );
                    return Err(err);
                }
            },
            None => {
                if let Err(err) = prewarm_opened_index(index.clone(), None).await {
                    warn!(
                        index_name = name,
                        %index_uuid,
                        error = %err,
                        elapsed_ms = segment_started.elapsed().as_millis() as u64,
                        "prewarm index segment failed"
                    );
                    return Err(err);
                }
                SegmentPrewarmResult { partition_count: 0 }
            }
        };

        info!(
            index_name = name,
            %index_uuid,
            elapsed_ms = segment_started.elapsed().as_millis() as u64,
            "prewarm index segment finished"
        );
        Ok(OpenedSegmentPrewarmResult {
            index_uuid,
            index,
            partition_count: segment_result.partition_count,
        })
    }))
    .await;

    match result {
        Ok(segment_results) => {
            let cache_stats_after = dataset.session.index_cache_stats().await;
            let result = aggregate_fts_prewarm_results(dataset, segment_results, options).await?;
            info!(
                index_name = name,
                selected_segment_count,
                fts_fully_resident = result.fully_resident,
                index_cache_entries_after = cache_stats_after.num_entries,
                index_cache_entries_delta = cache_size_delta(
                    cache_stats_after.num_entries,
                    cache_stats_before.num_entries,
                ),
                index_cache_size_bytes_after = cache_stats_after.size_bytes,
                index_cache_size_bytes_delta =
                    cache_size_delta(cache_stats_after.size_bytes, cache_stats_before.size_bytes,),
                elapsed_ms = request_started.elapsed().as_millis() as u64,
                "prewarm index segments finished"
            );
            Ok(result)
        }
        Err(err) => {
            let cache_stats_after = dataset.session.index_cache_stats().await;
            warn!(
                index_name = name,
                selected_segment_count,
                index_cache_entries_after = cache_stats_after.num_entries,
                index_cache_entries_delta = cache_size_delta(
                    cache_stats_after.num_entries,
                    cache_stats_before.num_entries,
                ),
                index_cache_size_bytes_after = cache_stats_after.size_bytes,
                index_cache_size_bytes_delta = cache_size_delta(
                    cache_stats_after.size_bytes,
                    cache_stats_before.size_bytes,
                ),
                error = %err,
                elapsed_ms = request_started.elapsed().as_millis() as u64,
                "prewarm index segments failed"
            );
            Err(err)
        }
    }
}

fn filter_index_segments_by_ids(
    name: &str,
    indices: Vec<IndexMetadata>,
    segment_ids: &[Uuid],
) -> Result<Vec<IndexMetadata>> {
    if segment_ids.is_empty() {
        return Ok(Vec::new());
    }

    let requested = segment_ids.iter().copied().collect::<HashSet<_>>();
    let mut matched = HashSet::new();
    let filtered = indices
        .into_iter()
        .filter(|index_meta| {
            if requested.contains(&index_meta.uuid) {
                matched.insert(index_meta.uuid);
                true
            } else {
                false
            }
        })
        .collect::<Vec<_>>();

    if matched.len() != requested.len() {
        let mut missing = requested
            .difference(&matched)
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        missing.sort();
        return Err(Error::index_not_found(format!(
            "name={}, segment_ids=[{}]",
            name,
            missing.join(", ")
        )));
    }

    Ok(filtered)
}

fn validate_segment_index_details(index_name: &str, segments: &[IndexMetadata]) -> Result<()> {
    let mut type_url = None::<&str>;
    for segment in segments {
        let segment_type_url = segment.index_details.as_ref().ok_or_else(|| {
            Error::invalid_input(format!(
                "CreateIndex: segment {} is missing index details",
                segment.uuid
            ))
        })?;
        match type_url {
            Some(expected) if expected != segment_type_url.type_url => {
                return Err(Error::invalid_input(format!(
                    "CreateIndex: segment set for index '{}' mixes incompatible index detail types",
                    index_name
                )));
            }
            None => type_url = Some(segment_type_url.type_url.as_str()),
            Some(_) => {}
        }
    }

    Ok(())
}

/// Detect vector segments while preserving the legacy pre-details fallback.
///
/// Older vector segments may not have `VectorIndexDetails` in the manifest, so
/// we also recognize them by the legacy monolithic index file name.
fn segment_has_vector_details(segment: &IndexMetadata) -> bool {
    segment.index_details.as_ref().map_or_else(
        || {
            segment
                .files
                .as_ref()
                .is_some_and(|files| files.iter().any(|file| file.path == INDEX_FILE_NAME))
        },
        |details| details.type_url.ends_with("VectorIndexDetails"),
    )
}

/// Whether this build has a reader for the index's declared type.
///
/// Segments without details predate type URLs and remain readable through the
/// legacy file-based detection in the index open paths.
pub(crate) fn index_type_is_known(index: &IndexMetadata) -> bool {
    is_system_index(index)
        || index
            .index_details
            .as_ref()
            .is_none_or(|details| IndexDetails(details.clone()).has_reader())
}

/// Detect FTS / inverted segments from manifest details.
///
/// Unlike vector, inverted segment support was added after index details were
/// part of the segment contract, so there is no legacy file-name fallback here.
fn segment_has_inverted_details(segment: &IndexMetadata) -> bool {
    segment
        .index_details
        .as_ref()
        .is_some_and(|details| details.type_url.ends_with("InvertedIndexDetails"))
}

fn segment_has_bitmap_details(segment: &IndexMetadata) -> bool {
    segment
        .index_details
        .as_ref()
        .is_some_and(|details| details.type_url.ends_with("BitmapIndexDetails"))
}

fn segment_has_bloomfilter_details(segment: &IndexMetadata) -> bool {
    segment
        .index_details
        .as_ref()
        .is_some_and(|details| details.type_url.ends_with("BloomFilterIndexDetails"))
}

/// Detect BTree segments, preserving a legacy pre-details fallback.
fn segment_has_btree_details(segment: &IndexMetadata) -> bool {
    segment.index_details.as_ref().map_or_else(
        || {
            segment
                .files
                .as_ref()
                .is_some_and(|files| files.iter().any(|file| file.path == BTREE_LOOKUP_NAME))
        },
        |details| details.type_url.ends_with("BTreeIndexDetails"),
    )
}

fn segment_has_zonemap_details(segment: &IndexMetadata) -> bool {
    segment
        .index_details
        .as_ref()
        .is_some_and(|details| details.type_url.ends_with("ZoneMapIndexDetails"))
}

fn segment_has_fmindex_details(segment: &IndexMetadata) -> bool {
    segment
        .index_details
        .as_ref()
        .is_some_and(|details| details.type_url.ends_with("FMIndexDetails"))
}

fn segment_has_label_list_details(segment: &IndexMetadata) -> bool {
    segment
        .index_details
        .as_ref()
        .is_some_and(|details| details.type_url.ends_with("LabelListIndexDetails"))
}

fn segment_has_rtree_details(segment: &IndexMetadata) -> bool {
    segment
        .index_details
        .as_ref()
        .is_some_and(|details| details.type_url.ends_with("RTreeIndexDetails"))
}

fn segment_has_ngram_details(segment: &IndexMetadata) -> bool {
    segment
        .index_details
        .as_ref()
        .is_some_and(|details| details.type_url.ends_with("NGramIndexDetails"))
}

// Cache keys for different index types
#[derive(Debug, Clone)]
pub(crate) struct LegacyVectorIndexCacheKey<'a> {
    uuid: &'a Uuid,
    fri_uuid: Option<&'a Uuid>,
}

impl<'a> LegacyVectorIndexCacheKey<'a> {
    fn new(uuid: &'a Uuid, fri_uuid: Option<&'a Uuid>) -> Self {
        Self { uuid, fri_uuid }
    }
}

impl CacheKey for LegacyVectorIndexCacheKey<'_> {
    type ValueType = CachedLegacyVectorIndex;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        if let Some(fri_uuid) = self.fri_uuid {
            format!("{}-{}", self.uuid, fri_uuid).into()
        } else {
            self.uuid.to_string().into()
        }
    }

    fn type_name() -> &'static str {
        "LegacyVectorIndex"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.legacy-vector-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        write_index_identity(builder, self.uuid, self.fri_uuid);
    }
}

/// Sized cache key for `IvfIndexState`.
///
/// Used for v0.3+ indices that support serialization. This key has a codec,
/// so custom cache backends can serialize the state to disk/Redis/etc.
/// Legacy indices use `LegacyVectorIndexCacheKey` instead (in-memory only).
///
/// Note: legacy entries hold live readers bound to the object store that
/// opened them, so they are only valid for credential setups that refresh
/// internally (e.g. a credentials provider). Deployments that pass fresh
/// static credentials per dataset open should use v0.3+ index formats.
#[derive(Debug, Clone)]
pub(crate) struct IvfIndexStateCacheKey<'a> {
    uuid: &'a Uuid,
    fri_uuid: Option<&'a Uuid>,
}

impl<'a> IvfIndexStateCacheKey<'a> {
    fn new(uuid: &'a Uuid, fri_uuid: Option<&'a Uuid>) -> Self {
        Self { uuid, fri_uuid }
    }
}

impl CacheKey for IvfIndexStateCacheKey<'_> {
    type ValueType = IvfStateEntryBox;

    fn type_name() -> &'static str {
        "IvfIndexState"
    }

    fn key(&self) -> std::borrow::Cow<'_, str> {
        if let Some(fri_uuid) = self.fri_uuid {
            format!("{}-{}", self.uuid, fri_uuid).into()
        } else {
            self.uuid.to_string().into()
        }
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.ivf-state-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        write_index_identity(builder, self.uuid, self.fri_uuid);
    }

    fn codec() -> Option<lance_core::cache::CacheCodec> {
        Some(lance_core::cache::CacheCodec::from_impl::<IvfStateEntryBox>())
    }
}

/// Wrapper that stores a live VectorIndex in the cache.
/// Used for v0.1/v0.2 indices that don't support serializable caching.
#[derive(Debug, lance_core::deepsize::DeepSizeOf)]
pub(crate) struct CachedLegacyVectorIndex(Arc<dyn VectorIndex>);

#[derive(Debug, Clone)]
pub struct FragReuseIndexCacheKey<'a> {
    pub uuid: &'a Uuid,
    pub fri_uuid: Option<&'a Uuid>,
}

impl<'a> FragReuseIndexCacheKey<'a> {
    pub fn new(uuid: &'a Uuid, fri_uuid: Option<&'a Uuid>) -> Self {
        Self { uuid, fri_uuid }
    }
}

impl CacheKey for FragReuseIndexCacheKey<'_> {
    type ValueType = CompactFragReuseIndex;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        if let Some(fri_uuid) = self.fri_uuid {
            format!("{}-{}", self.uuid, fri_uuid).into()
        } else {
            self.uuid.to_string().into()
        }
    }

    fn type_name() -> &'static str {
        "FragReuseIndex"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fragment-reuse-cache-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        write_index_identity(builder, self.uuid, self.fri_uuid);
    }
}

#[derive(Debug, Clone)]
pub struct MemWalCacheKey<'a> {
    pub uuid: &'a Uuid,
    pub fri_uuid: Option<&'a Uuid>,
}

impl<'a> MemWalCacheKey<'a> {
    pub fn new(uuid: &'a Uuid, fri_uuid: Option<&'a Uuid>) -> Self {
        Self { uuid, fri_uuid }
    }
}

impl CacheKey for MemWalCacheKey<'_> {
    type ValueType = MemWalIndex;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        if let Some(fri_uuid) = self.fri_uuid {
            format!("{}-{}", self.uuid, fri_uuid).into()
        } else {
            self.uuid.to_string().into()
        }
    }

    fn type_name() -> &'static str {
        "MemWalIndex"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.mem-wal-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        write_index_identity(builder, self.uuid, self.fri_uuid);
    }
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
        "Json" => IndexType::Scalar.to_string(),
        "Flat" | "Vector" => IndexType::Vector.to_string(),
        other if other.contains("Vector") => IndexType::Vector.to_string(),
        _ => "N/A".to_string(),
    }
}

/// Builds index.
#[async_trait]
pub trait IndexBuilder {
    fn index_type() -> IndexType;

    async fn build(&self) -> Result<()>;
}

pub(crate) async fn remap_index(
    dataset: &Dataset,
    index_id: &Uuid,
    row_id_map: &RowAddrRemap,
) -> Result<RemapResult> {
    // Load indices from the disk.
    let indices = dataset.load_indices().await?;
    let matched = indices
        .iter()
        .find(|i| i.uuid == *index_id)
        .ok_or_else(|| Error::index(format!("Index with id {} does not exist", index_id)))?;

    // Corrupt metadata fails closed before anything else: a declaration that is
    // not a valid suffix of `fields` cannot be reasoned about at all, and the
    // withdrawal below would otherwise swallow it as an ordinary covered index.
    matched.validate_covering_fields()?;

    // Remap handles a single keyed field. Carried fields are irrelevant to the
    // remap itself: the index is opened by its keyed field, and each storage
    // format's `remap` preserves its covering columns row-aligned (the format
    // commit dropped covered indexes here because nothing wrote carried values
    // yet; the builders in this commit do). Only a genuinely composite index is
    // rejected -- `keyed_fields` fails closed on malformed metadata, but the
    // validation above already rejected that.
    if matched.keyed_fields().len() > 1 {
        return Err(Error::index(format!(
            "Remapping index '{}' is not supported: it has {} keyed fields {:?} \
             (carried fields {:?}); only one keyed field is supported",
            matched.name,
            matched.keyed_fields().len(),
            matched.fields,
            matched.covering_fields,
        )));
    }

    if let Some(deleted_bitmap) = row_id_map.fully_deleted_fragments()
        && Some(deleted_bitmap) == matched.fragment_bitmap
    {
        // If remap deleted all rows, we can just return the same index ID.
        // This can happen if there is a bug where the index is covering empty
        // fragment that haven't been cleaned up. They should be cleaned up
        // outside of this function.
        return Ok(RemapResult::Keep(*index_id));
    }

    let field_id = matched
        .fields
        .first()
        .expect("An index existed with no fields");
    let field_path = dataset.schema().field_path(*field_id)?;

    let new_id = Uuid::new_v4();

    let generic = match dataset
        .open_generic_index(&field_path, index_id, &NoOpMetricsCollector)
        .await
    {
        Ok(g) => g,
        Err(e) => {
            log::warn!(
                "Cannot open index '{}' on '{}': {}. \
                 Index will be dropped during compaction.",
                index_id,
                field_path,
                e
            );
            return Ok(RemapResult::Drop);
        }
    };

    let created_index = match generic.index_type() {
        it if it.is_scalar() => {
            // Withdraw rather than remap a covered scalar index. The vector storages'
            // `remap` carries their covering columns through row-aligned, which is what
            // lets a covered vector index be remapped in place; `BTreeIndex::remap` and
            // friends rewrite only `(value, row_id)`. Remapping would republish a
            // declaration the new files do not back -- and because carried fields live in
            // `fields`, the index would then shed fragment coverage on every update to a
            // column it never held. Withdrawing follows the same shape as the
            // `can_remap()` check below and the open failure above: a table-level
            // compaction degrades one index rather than failing outright.
            if !matched.covering_fields.is_empty() {
                log::warn!(
                    "Withdrawing covered index '{}' during remap: its storage format does \
                     not carry covering columns through a remap, so the declaration would \
                     outlive the payload.",
                    matched.name
                );
                return Ok(RemapResult::Drop);
            }
            let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_id)?;

            let scalar_index = dataset
                .open_scalar_index(&field_path, index_id, &NoOpMetricsCollector)
                .await?;
            if !scalar_index.can_remap() {
                return Ok(RemapResult::Drop);
            }

            match scalar_index.index_type() {
                IndexType::Inverted => {
                    let inverted_index = scalar_index
                        .as_any()
                        .downcast_ref::<lance_index::scalar::inverted::InvertedIndex>()
                        .ok_or(Error::index("expected inverted index".to_string()))?;
                    if inverted_index.is_legacy() {
                        log::warn!(
                            "reindex because of legacy format, index_type: {}, index_id: {}, field: {}",
                            scalar_index.index_type(),
                            index_id,
                            field_path
                        );
                        let training_data = load_training_data(
                            dataset,
                            &field_path,
                            &TrainingCriteria::new(TrainingOrdering::None),
                            None,
                            true, // Legacy reindexing should always train
                            None,
                        )
                        .await?;
                        InvertedIndexPlugin::train_inverted_index(
                            training_data,
                            &new_store,
                            inverted_index.params().clone(),
                            None,
                            Arc::new(NoopIndexBuildProgress),
                        )
                        .await?
                    } else {
                        scalar_index.remap(row_id_map, &new_store).await?
                    }
                }
                _ => scalar_index.remap(row_id_map, &new_store).await?,
            }
        }
        it if it.is_vector() => {
            let index_version = u32::try_from(matched.index_version).map_err(|_| {
                Error::index(format!(
                    "Invalid vector index version {} on index {}",
                    matched.index_version, matched.name
                ))
            })?;
            let files = remap_vector_index(
                Arc::new(dataset.clone()),
                &field_path,
                index_id,
                &new_id,
                matched,
                row_id_map,
            )
            .await?;

            CreatedIndex {
                index_details: prost_types::Any::from_msg(
                    &lance_index::pb::VectorIndexDetails::default(),
                )
                .unwrap(),
                index_version,
                files: table_files_to_index(files),
            }
        }
        _ => {
            return Err(Error::index(format!(
                "Index type {} is not supported",
                generic.index_type()
            )));
        }
    };

    Ok(RemapResult::Remapped(RemappedIndex {
        old_id: *index_id,
        new_id,
        index_details: created_index.index_details,
        index_version: created_index.index_version,
        files: Some(index_files_to_table(created_index.files)),
    }))
}

/// Snapshot of every scalar index on a dataset, captured at planning time
/// and consumed by the scalar/aggregate pushdown machinery.
///
/// Built once per planner invocation by walking the manifest's `IndexMetadata`
/// entries; thereafter all lookups are synchronous, so optimizer rules and the
/// filter parser can interrogate it without needing an async context.
#[derive(Debug)]
pub struct ScalarIndexInfo {
    /// Per-column dispatch table for [`apply_scalar_indices`]: keyed by the
    /// full dotted field path (e.g. `"x"`, `"metadata.status.code"`), the same
    /// string callers use when referring to columns in filter expressions.
    ///
    /// The value pairs the column's data type with a [`MultiQueryParser`]
    /// that fans out to every per-index parser registered for that column.
    /// When a column carries more than one index (e.g. BTree + bitmap), the
    /// `MultiQueryParser` tries each in order and the first match wins; the
    /// resulting [`crate::scalar::expression::ScalarIndexSearch`] records
    /// which specific index was chosen. So *which* index served the query is
    /// an output of parsing, not an input — that's why this map is keyed only
    /// by column.
    ///
    /// `fragment_bitmaps`, by contrast, *is* keyed by `(column, index_name)`,
    /// because by the time the optimizer needs the bitmap the index name is
    /// already pinned in the parsed leaf.
    indexed_columns: HashMap<String, (DataType, Box<MultiQueryParser>)>,
    /// `(column, index_name) → fragment_bitmap` taken straight off each
    /// [`IndexMetadata`] at construction time. Used by the optimizer rule for
    /// aggregate pushdown to reason about index coverage synchronously.
    /// Indices that omit `fragment_bitmap` (legacy or unsupported) simply
    /// don't appear here and so report coverage as unknown.
    fragment_bitmaps: HashMap<(String, String), RoaringBitmap>,
}

impl IndexInformationProvider for ScalarIndexInfo {
    fn get_index(&self, col: &str) -> Option<(&DataType, &MultiQueryParser)> {
        self.indexed_columns
            .get(col)
            .map(|(ty, parser)| (ty, parser.as_ref()))
    }

    fn fragment_bitmap(&self, column: &str, index_name: &str) -> Option<RoaringBitmap> {
        self.fragment_bitmaps
            .get(&(column.to_string(), index_name.to_string()))
            .cloned()
    }
}

async fn open_index_proto(reader: &dyn Reader) -> Result<pb::Index> {
    let file_size = reader.size().await?;
    let tail_bytes = read_last_block(reader).await?;
    let metadata_pos = read_metadata_offset(&tail_bytes)?;
    let proto: pb::Index = if metadata_pos < file_size - tail_bytes.len() {
        // We have not read the metadata bytes yet.
        read_message(reader, metadata_pos).await?
    } else {
        let offset = tail_bytes.len() - (file_size - metadata_pos);
        read_message_from_buf(&tail_bytes.slice(offset..))?
    };
    Ok(proto)
}

struct IndexDescriptionImpl {
    name: String,
    field_ids: Vec<u32>,
    segments: Vec<IndexMetadata>,
    index_type: String,
    /// Index details, or `None` for indices created before details were
    /// persisted in the manifest. Such indices are still described on a
    /// best-effort basis rather than rejected.
    details: Option<IndexDetails>,
    rows_indexed: u64,
}

impl IndexDescriptionImpl {
    async fn try_new(segments: Vec<IndexMetadata>, dataset: &Dataset) -> Result<Self> {
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

        Ok(Self {
            name,
            field_ids: field_ids_vec,
            index_type,
            segments,
            details,
            rows_indexed,
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
}

impl Dataset {
    /// Whether an otherwise empty commit would record a new MemWAL catch-up
    /// position.
    ///
    /// Dry-runs the derivation rather than restating its conditions: a second
    /// copy of "is this index behind" would be one more place to keep in step
    /// with the real rule. A no-work optimize publishes no new segment, so the
    /// index list it would commit is the one already loaded, and the version it
    /// would read is the current one -- which makes the speculative answer the
    /// same one the commit reaches.
    fn mem_wal_catch_up_would_advance(&self, indices: &[IndexMetadata]) -> Result<bool> {
        if !indices.iter().any(|index| index.name == MEM_WAL_INDEX_NAME) {
            return Ok(false);
        }
        let catchup_of = |indices: &[IndexMetadata]| -> Result<Option<Vec<_>>> {
            indices
                .iter()
                .find(|index| index.name == MEM_WAL_INDEX_NAME)
                .cloned()
                .map(|index| {
                    crate::index::mem_wal::load_mem_wal_index_details(index)
                        .map(|details| details.index_catchup)
                })
                .transpose()
        };

        let mut speculative = indices.to_vec();
        Transaction::apply_mem_wal_index_coverage(
            &mut speculative,
            &Transaction::logical_index_segments(indices),
            Some(ReadVersionState {
                manifest: &self.manifest,
                indices,
            }),
            self.manifest.version + 1,
        )?;
        Ok(catchup_of(&speculative)? != catchup_of(indices)?)
    }
}

#[async_trait]
impl DatasetIndexExt for Dataset {
    type IndexBuilder<'a> = CreateIndexBuilder<'a>;
    /// Create a builder for creating an index on columns.
    ///
    /// This returns a builder that can be configured with additional options
    /// before awaiting to execute.
    ///
    /// # Examples
    ///
    /// Create a scalar BTREE index:
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::index::DatasetIndexExt;
    /// # use lance_index::{IndexType, scalar::ScalarIndexParams};
    /// # async fn example(dataset: &mut Dataset) -> Result<()> {
    /// let params = ScalarIndexParams::default();
    /// dataset
    ///     .create_index_builder(&["id"], IndexType::BTree, &params)
    ///     .name("id_index".to_string())
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Create an empty index that will be populated later:
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::index::DatasetIndexExt;
    /// # use lance_index::{IndexType, scalar::ScalarIndexParams};
    /// # async fn example(dataset: &mut Dataset) -> Result<()> {
    /// let params = ScalarIndexParams::default();
    /// dataset
    ///     .create_index_builder(&["category"], IndexType::Bitmap, &params)
    ///     .train(false)  // Create empty index
    ///     .replace(true)  // Replace if exists
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    fn create_index_builder<'a>(
        &'a mut self,
        columns: &[&str],
        index_type: IndexType,
        params: &'a dyn IndexParams,
    ) -> CreateIndexBuilder<'a> {
        CreateIndexBuilder::new(self, columns, index_type, params)
    }

    #[instrument(skip_all)]
    async fn create_index(
        &mut self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
        replace: bool,
    ) -> Result<IndexMetadata> {
        // Use the builder pattern with default train=true for backward compatibility
        let mut builder = self.create_index_builder(columns, index_type, params);

        if let Some(name) = name {
            builder = builder.name(name);
        }

        builder.replace(replace).await
    }

    async fn drop_index(&mut self, name: &str) -> Result<()> {
        // Removal never opens the index, so an index this build cannot read is
        // still droppable - and has to be, since it is otherwise unremovable.
        let indices = load_all_indices_by_name(self, name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![],
                removed_indices: indices.clone(),
            },
            None,
        );

        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        Ok(())
    }

    async fn prewarm_index(&self, name: &str) -> Result<()> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }

        let available_segment_count = indices.len();
        prewarm_index_segments_by_metadata(self, name, indices, None, available_segment_count, None)
            .await
            .map(|_| ())
    }

    async fn prewarm_index_with_options(&self, name: &str, options: &PrewarmOptions) -> Result<()> {
        self.prewarm_index_with_options_result(name, options)
            .await
            .map(|_| ())
    }

    async fn prewarm_index_with_options_result(
        &self,
        name: &str,
        options: &PrewarmOptions,
    ) -> Result<FtsPrewarmResult> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }

        let available_segment_count = indices.len();
        prewarm_index_segments_by_metadata(
            self,
            name,
            indices,
            Some(options),
            available_segment_count,
            None,
        )
        .await
    }

    async fn prewarm_index_segments(&self, name: &str, segment_ids: &[Uuid]) -> Result<()> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }
        let available_segment_count = indices.len();
        let indices = filter_index_segments_by_ids(name, indices, segment_ids)?;

        prewarm_index_segments_by_metadata(
            self,
            name,
            indices,
            None,
            available_segment_count,
            Some(segment_ids.len()),
        )
        .await
        .map(|_| ())
    }

    async fn prewarm_index_segments_with_options(
        &self,
        name: &str,
        segment_ids: &[Uuid],
        options: &PrewarmOptions,
    ) -> Result<()> {
        self.prewarm_index_segments_with_options_result(name, segment_ids, options)
            .await
            .map(|_| ())
    }

    async fn prewarm_index_segments_with_options_result(
        &self,
        name: &str,
        segment_ids: &[Uuid],
        options: &PrewarmOptions,
    ) -> Result<FtsPrewarmResult> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }
        let available_segment_count = indices.len();
        let indices = filter_index_segments_by_ids(name, indices, segment_ids)?;

        prewarm_index_segments_by_metadata(
            self,
            name,
            indices,
            Some(options),
            available_segment_count,
            Some(segment_ids.len()),
        )
        .await
    }

    async fn describe_indices<'a, 'b>(
        &'a self,
        criteria: Option<IndexCriteria<'b>>,
    ) -> Result<Vec<Arc<dyn IndexDescription>>> {
        let indices = self.load_indices().await?;
        let mut indices = if let Some(criteria) = criteria {
            indices.iter().filter(|idx| {
                if idx.index_details.is_none() {
                    log::warn!("The method describe_indices does not support indexes without index details.  Please retrain the index {}", idx.name);
                    return false;
                }
                // Only the keyed prefix. `index_matches_criteria` compares this
                // slice against the single column named by `for_column`, so
                // passing the carried columns too would push its length past one
                // and silently drop every covered index from a filtered
                // describe.
                let fields = idx
                    .keyed_fields()
                    .iter()
                    .filter_map(|id| self.schema().field_by_id(*id))
                    .collect::<Vec<_>>();
                match index_matches_criteria(idx, &criteria, &fields, false, self.schema()) {
                    Ok(matched) => matched,
                    Err(err) => {
                        log::warn!("Could not describe index {}: {}", idx.name, err);
                        false
                    }
                }
            }).collect::<Vec<_>>()
        } else {
            indices.iter().collect::<Vec<_>>()
        };
        indices.sort_by_key(|idx| &idx.name);

        let grouped: Vec<Vec<IndexMetadata>> = indices
            .into_iter()
            .chunk_by(|idx| idx.name.clone())
            .into_iter()
            .map(|(_, segments)| segments.cloned().collect::<Vec<_>>())
            .collect();

        let mut results = Vec::with_capacity(grouped.len());
        for segments in grouped {
            let desc = IndexDescriptionImpl::try_new(segments, self).await?;
            results.push(Arc::new(desc) as Arc<dyn IndexDescription>);
        }
        Ok(results)
    }

    async fn load_indices(&self) -> Result<Arc<Vec<IndexMetadata>>> {
        let indices = load_all_indices(self).await?;
        if indices.iter().all(index_is_usable) {
            return Ok(indices);
        }
        Ok(Arc::new(
            indices
                .iter()
                .filter(|idx| index_is_usable(idx))
                .cloned()
                .collect(),
        ))
    }

    async fn merge_existing_index_segments(
        &self,
        mut source_segments: Vec<IndexMetadata>,
    ) -> Result<IndexMetadata> {
        validate_segment_metadata("uncommitted", &source_segments)?;
        if let Some(segment) = source_segments
            .iter()
            .find(|segment| segment.dataset_version > self.manifest.version)
        {
            return Err(Error::invalid_input(format!(
                "merge_existing_index_segments: segment {} was built at future dataset version {} (current version {})",
                segment.uuid, segment.dataset_version, self.manifest.version
            )));
        }
        let source_dataset_version = source_segments
            .iter()
            .map(|segment| segment.dataset_version)
            .min()
            .unwrap_or(self.manifest.version);
        let field_id = *source_segments[0].fields.first().ok_or_else(|| {
            Error::invalid_input(format!(
                "CreateIndex: segment {} is missing field ids",
                source_segments[0].uuid
            ))
        })?;
        let expected_covering_fields = source_segments[0].covering_fields.clone();
        for segment in &source_segments {
            // Same rule as `build_index_metadata_from_segments`: only the
            // keyed-field count matters here, not the exact `fields` vector,
            // so a covered segment is not rejected outright.
            if segment.keyed_field() != Some(field_id) {
                return Err(Error::invalid_input(format!(
                    "merge_existing_index_segments: segment {} was built for fields {:?} \
                     (carried {:?}), expected keyed field [{}]",
                    segment.uuid, segment.fields, segment.covering_fields, field_id
                )));
            }
            segment.validate_covering_fields()?;
            // Merging requires one coherent output declaration, so every
            // input must carry the same columns (not merely the same count).
            if segment.covering_fields != expected_covering_fields {
                return Err(Error::invalid_input(
                    "merge_existing_index_segments requires segments with identical fields"
                        .to_string(),
                ));
            }
        }
        let all_vector = source_segments.iter().all(segment_has_vector_details);
        let all_inverted = source_segments.iter().all(segment_has_inverted_details);
        let all_bitmap = source_segments.iter().all(segment_has_bitmap_details);
        let all_bloomfilter = source_segments.iter().all(segment_has_bloomfilter_details);
        let all_btree = source_segments.iter().all(segment_has_btree_details);
        let all_fmindex = source_segments.iter().all(segment_has_fmindex_details);
        let all_zonemap = source_segments.iter().all(segment_has_zonemap_details);
        let all_label_list = source_segments.iter().all(segment_has_label_list_details);
        let all_rtree = source_segments.iter().all(segment_has_rtree_details);
        let all_ngram = source_segments.iter().all(segment_has_ngram_details);
        if !all_vector
            && !all_inverted
            && !all_bitmap
            && !all_bloomfilter
            && !all_btree
            && !all_fmindex
            && !all_zonemap
            && !all_label_list
            && !all_rtree
            && !all_ngram
        {
            return Err(Error::invalid_input(
                "merge_existing_index_segments requires all segments to have the same supported index type"
                    .to_string(),
            ));
        }

        let merged_dataset_version = if all_rtree {
            let mut source_coverage = source_segments
                .iter()
                .cloned()
                .map(IntoIndexSegment::into_index_segment)
                .collect::<Result<Vec<_>>>()?;
            prune_stale_segment_coverage(self, &mut source_coverage, true, true).await?;
            for (source, coverage) in source_segments.iter_mut().zip(source_coverage) {
                source.fragment_bitmap = Some(coverage.fragment_bitmap().clone());
            }
            self.manifest.version
        } else {
            source_dataset_version
        };

        let mut merged_segment = if all_vector {
            crate::index::vector::ivf::merge_segments(self, source_segments).await?
        } else if all_inverted {
            crate::index::scalar::inverted::merge_segments(self, source_segments).await?
        } else if all_fmindex {
            crate::index::scalar::fmindex::merge_segments(self, source_segments).await?
        } else if all_bitmap {
            crate::index::scalar::bitmap::merge_segments(self, source_segments).await?
        } else if all_bloomfilter {
            crate::index::scalar::bloomfilter::merge_segments(self, source_segments).await?
        } else if all_label_list {
            crate::index::scalar::label_list::merge_segments(self, source_segments).await?
        } else if all_zonemap {
            crate::index::scalar::zonemap::merge_segments(self, source_segments).await?
        } else if all_ngram {
            crate::index::scalar::ngram::merge_segments(self, source_segments).await?
        } else if all_rtree {
            #[cfg(feature = "geo")]
            {
                crate::index::scalar::rtree::merge_segments(self, source_segments).await?
            }
            #[cfg(not(feature = "geo"))]
            return Err(Error::not_supported(
                "RTree segment merge requires the `geo` feature".to_string(),
            ));
        } else {
            crate::index::scalar::btree::merge_segments(self, source_segments).await?
        };
        if !all_ngram && !all_fmindex {
            merged_segment.dataset_version = merged_dataset_version;
        }
        Ok(merged_segment)
    }

    async fn commit_existing_index_segments(
        &mut self,
        index_name: &str,
        column: &str,
        segments: Vec<impl IntoIndexSegment + Send>,
    ) -> Result<()> {
        let Some(field) = self.schema().field(column) else {
            return Err(Error::index(format!(
                "CreateIndex: column '{column}' does not exist"
            )));
        };

        let segments = segments
            .into_iter()
            .map(IntoIndexSegment::into_index_segment)
            .collect::<Result<Vec<_>>>()?;
        let dataset_fragments = self.fragment_bitmap.as_ref().clone();
        if segments.first().is_some_and(|segment| {
            segment
                .index_details()
                .type_url
                .ends_with("NGramIndexDetails")
        }) {
            let has_retired_coverage = segments
                .iter()
                .any(|segment| !(segment.fragment_bitmap() - &dataset_fragments).is_empty());
            let frag_reuse_index = self.open_frag_reuse_index(&NoOpMetricsCollector).await?;
            let requires_rebuild = frag_reuse_index.as_ref().is_some_and(|frag_reuse_index| {
                segments.iter().any(|segment| {
                    append::fragment_reuse_affects_segment(
                        frag_reuse_index,
                        segment.fragment_bitmap(),
                        segment.dataset_version(),
                    )
                })
            });
            if has_retired_coverage || requires_rebuild {
                return Err(Error::invalid_input(
                    "CreateIndex: NGram segments built before compaction must be rebuilt or merged \
                     with merge_existing_index_segments before commit"
                        .to_string(),
                ));
            }
        }
        let new_indices =
            build_index_metadata_from_segments(self, index_name, field.id, segments).await?;
        validate_segment_metadata(index_name, &new_indices)?;
        validate_segment_index_details(index_name, &new_indices)?;

        let incoming_type_url = new_indices[0]
            .index_details
            .as_ref()
            .map(|details| details.type_url.clone());
        let mut incoming_fragments = RoaringBitmap::new();
        for segment in &new_indices {
            // Mirrors `build_index_metadata_from_segments`'s guard, which
            // already validated these exact segments. Kept in the same shape
            // so this second look does not silently re-reject what that guard
            // just accepted.
            if segment.keyed_field() != Some(field.id) {
                return Err(Error::invalid_input(format!(
                    "CreateIndex: segment {} was built for fields {:?} (carried {:?}), \
                     expected keyed field [{}]",
                    segment.uuid, segment.fields, segment.covering_fields, field.id
                )));
            }
            segment.validate_covering_fields()?;
            if let Some(fragment_bitmap) = &segment.fragment_bitmap {
                incoming_fragments |= fragment_bitmap.clone();
            }
        }

        let existing_named_indices = load_all_indices_by_name(self, index_name).await?;
        if existing_named_indices.iter().any(|idx| {
            // Same name-collision rule as `CreateIndexBuilder`'s default-name
            // loop in create.rs.
            idx.keyed_field() != Some(field.id)
        }) {
            return Err(Error::index(format!(
                "Index name '{index_name}' already exists with different fields, \
                please specify a different name"
            )));
        }
        let existing_different_type_url = existing_named_indices.iter().find_map(|idx| {
            // Legacy metadata may omit index details. Its type cannot be proven
            // compatible, so only a full replacement is safe.
            let Some(existing_details) = idx.index_details.as_ref() else {
                return Some("<unknown>".to_owned());
            };
            let existing_type_url = existing_details.type_url.as_str();
            (Some(existing_type_url) != incoming_type_url.as_deref())
                .then(|| existing_type_url.to_owned())
        });
        let missing_fragments = &dataset_fragments - &incoming_fragments;
        if let Some(existing_type_url) = &existing_different_type_url
            && !missing_fragments.is_empty()
        {
            return Err(Error::invalid_input(format!(
                "CreateIndex: cannot change index '{}' from type '{}' to type '{}' with partial fragment coverage; incoming segments are missing current fragments {:?}",
                index_name,
                existing_type_url,
                incoming_type_url.as_deref().unwrap_or("<missing>"),
                missing_fragments.iter().collect::<Vec<_>>()
            )));
        }

        let is_index_type_change = existing_different_type_url.is_some();
        // What a retained sibling has to agree with. Every incoming segment
        // already carries the same pair: `build_index_metadata_from_segments`
        // compares them against each other before this point.
        let expected_fields = new_indices[0].fields.clone();
        let expected_covering_fields = new_indices[0].covering_fields.clone();
        let removed_indices = existing_named_indices
            .into_iter()
            .map(|idx| -> Result<Option<IndexMetadata>> {
                // A logical index cannot combine segment types. Full current-fragment
                // coverage was verified above, so a type change replaces every segment.
                if is_index_type_change {
                    return Ok(Some(idx));
                }

                let Some(existing_fragments) = idx.effective_fragment_bitmap(&dataset_fragments)
                else {
                    if incoming_fragments != dataset_fragments {
                        return Err(Error::invalid_input(format!(
                            "CreateIndex: cannot replace legacy index segment {} for '{}' with partial fragment coverage; rebuild all fragments in one commit",
                            idx.uuid, index_name
                        )));
                    }
                    return Ok(Some(idx));
                };

                // A zero-fragment segment can be used to create an index while
                // deferring the actual build. Such a segment is disjoint from every
                // other segment but should still be removed.
                if existing_fragments.is_empty() {
                    return Ok(Some(idx));
                }

                if existing_fragments.is_disjoint(&incoming_fragments) {
                    // Retained, so its declaration outlives this commit and has
                    // to match what is being written: `IndexDescriptionImpl::try_new`
                    // requires `fields` to be identical across the segments of one
                    // logical index. Nothing above catches a disagreement, because
                    // `keyed_fields` is the prefix left after the carried ones -- a
                    // segment carrying columns keys on exactly what a plain one
                    // keys on, and both pass the keyed-field guard.
                    if idx.fields != expected_fields
                        || idx.covering_fields != expected_covering_fields
                    {
                        return Err(Error::invalid_input(format!(
                            "CreateIndex: incoming segments for '{}' declare fields {:?} and covering_fields {:?}, \
                             but retained segment {} declares fields {:?} and covering_fields {:?}; \
                             a logical index cannot mix declarations - rebuild every segment in one commit",
                            index_name,
                            expected_fields,
                            expected_covering_fields,
                            idx.uuid,
                            idx.fields,
                            idx.covering_fields
                        )));
                    }
                    return Ok(None);
                }

                let uncovered = existing_fragments - &incoming_fragments;
                if !uncovered.is_empty() {
                    return Err(Error::invalid_input(format!(
                        "CreateIndex: incoming segments for '{}' would orphan fragments {:?} from existing segment {}",
                        index_name,
                        uncovered.iter().collect::<Vec<_>>(),
                        idx.uuid
                    )));
                }

                Ok(Some(idx))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            },
            None,
        );

        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        Ok(())
    }

    async fn load_scalar_index<'a, 'b>(
        &'a self,
        criteria: IndexCriteria<'b>,
    ) -> Result<Option<IndexMetadata>> {
        let indices = self.load_indices().await?;

        let mut indices = indices
            .iter()
            .filter(|idx| {
                // We shouldn't have any indices with empty fields, but just in case, log an error
                // but don't fail the operation (we might not be using that index)
                if idx.fields.is_empty() {
                    if idx.name != FRAG_REUSE_INDEX_NAME {
                        log::error!("Index {} has no fields", idx.name);
                    }
                    false
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();
        // This sorting & chunking is only needed to calculate if there are multiple indexes on the same
        // field.  This fact is only needed for backwards compatibility behavior for indexes that don't have
        // index details.  At some point we should deprecate indexes without index details.
        //
        // TODO: At some point we should just fail if the index details are missing and ask the user to
        // retrain the index.
        indices.sort_by_key(|idx| idx.fields[0]);
        let indice_by_field = indices.into_iter().chunk_by(|idx| idx.fields[0]);
        for (field_id, indices) in &indice_by_field {
            let indices = indices.collect::<Vec<_>>();
            let has_multiple = indices.len() > 1;
            for idx in indices {
                let field = self.schema().field_by_id(field_id);
                if let Some(field) = field
                    && index_matches_criteria(
                        idx,
                        &criteria,
                        &[field],
                        has_multiple,
                        self.schema(),
                    )?
                {
                    let non_empty = idx.fragment_bitmap.as_ref().is_some_and(|bitmap| {
                        bitmap.intersection_len(self.fragment_bitmap.as_ref()) > 0
                    });
                    let is_fts_index = if let Some(details) = &idx.index_details {
                        IndexDetails(details.clone()).supports_fts()
                    } else {
                        false
                    };
                    // FTS indices must always be returned even if empty, because FTS queries
                    // require an index to exist. The query execution will handle the empty
                    // bitmap appropriately and fall back to scanning unindexed data.
                    // Other index types can be skipped if empty since they're optional optimizations.
                    if non_empty || is_fts_index {
                        return Ok(Some(idx.clone()));
                    }
                }
            }
        }
        return Ok(None);
    }

    #[instrument(skip_all)]

    async fn optimize_indices(&mut self, options: &OptimizeOptions) -> Result<()> {
        let dataset = Arc::new(self.clone());
        // Grouped from the complete list so a name's segments are all accounted
        // for. A segment this build cannot read is still coverage, and merging
        // against a group whose coverage is only partly visible would commit a
        // new segment claiming fragments an existing one already holds.
        let indices = load_all_indices(self).await?;

        let indices_to_optimize = options
            .index_names
            .as_ref()
            .map(|names| names.iter().collect::<HashSet<_>>());
        let name_to_indices = indices
            .iter()
            .filter(|idx| {
                indices_to_optimize
                    .as_ref()
                    .is_none_or(|names| names.contains(&idx.name))
                    && !is_system_index(idx)
            })
            .map(|idx| (idx.name.clone(), idx))
            .into_group_map();

        let mut new_indices = vec![];
        let mut removed_indices = vec![];
        for (name, deltas) in name_to_indices.iter() {
            if let Some(index) = deltas.iter().find(|idx| !index_type_is_known(idx)) {
                let type_url = index
                    .index_details
                    .as_ref()
                    .map(|details| details.type_url.as_str())
                    .unwrap_or("<legacy>");
                log::warn!(
                    "Skipping optimization of index '{}' because this build does not recognize index type '{}'",
                    index.name,
                    type_url
                );
                continue;
            }

            // Optimizing a name means replacing its segments with one that
            // covers their union, which this build cannot compute when it
            // cannot read one of them: the merged segment would overlap the
            // segment left behind, and `Dataset::validate` calls that
            // corruption. Leave the whole name to a build that can read it.
            if let Some(max_supported_version) =
                deltas.iter().find_map(|idx| unsupported_index_version(idx))
            {
                log::warn!(
                    "Index {} has a segment newer than version {}, which this build cannot read; \
                     skipping its optimization",
                    name,
                    max_supported_version,
                );
                continue;
            }
            // Scalar indices have no rebalance concept, so skip them entirely
            // when every fragment is already covered and the caller hasn't
            // asked for retrain or an explicit delta merge. Vector indices
            // fall through and use a rebalance-aware no-op check inside
            // merge_indices_with_unindexed_frags.
            if !options.retrain
                && options.num_indices_to_merge.is_none_or(|n| n == 0)
                && index_group_is_scalar(self, deltas)
                && index_group_has_no_unindexed(self, deltas)
            {
                continue;
            }

            let Some(res) = merge_indices(dataset.clone(), deltas.as_slice(), options).await?
            else {
                continue;
            };

            let last_idx = deltas.last().expect("Delta indices should not be empty");
            let new_idx = IndexMetadata {
                uuid: res.new_uuid,
                name: last_idx.name.clone(), // Keep the same name
                fields: last_idx.fields.clone(),
                covering_fields: last_idx.covering_fields.clone(),
                dataset_version: res.new_dataset_version,
                fragment_bitmap: Some(res.new_fragment_bitmap),
                index_details: Some(Arc::new(res.new_index_details)),
                index_version: res.new_index_version,
                created_at: Some(chrono::Utc::now()),
                base_id: None, // New merged index file locates in the cloned dataset.
                files: Some(res.files),
            };
            removed_indices.extend(res.removed_indices.iter().map(|&idx| idx.clone()));
            new_indices.push(new_idx);
        }

        // A no-work optimize still has to commit on a table that requires
        // catch-up. Coverage is derived at commit time, so an index that
        // already spans the table records its position only if there is a
        // commit to record it on -- and that is the ordinary case after a
        // remap or a compaction that advanced a generation without changing
        // fragments. Returning early there leaves the position missing forever
        // and the repair rescheduling itself.
        if new_indices.is_empty() && !self.mem_wal_catch_up_would_advance(&indices)? {
            return Ok(());
        }

        let transaction = TransactionBuilder::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            },
        )
        .transaction_properties(options.transaction_properties.clone())
        .build();

        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        Ok(())
    }

    async fn index_statistics(&self, index_name: &str) -> Result<String> {
        let metadatas = self.load_indices_by_name(index_name).await?;
        if metadatas.is_empty() {
            return Err(Error::index_not_found(format!("name={}", index_name)));
        }

        if index_name == FRAG_REUSE_INDEX_NAME {
            return index_statistics_frag_reuse(self).boxed().await;
        }

        if index_name == MEM_WAL_INDEX_NAME {
            return index_statistics_mem_wal(self).boxed().await;
        }

        index_statistics_scalar(self, index_name, metadatas)
            .boxed()
            .await
    }

    async fn read_index_partition(
        &self,
        index_name: &str,
        partition_id: usize,
        with_vector: bool,
    ) -> Result<SendableRecordBatchStream> {
        let indices = self.load_indices_by_name(index_name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", index_name)));
        }
        let field_id = indices[0].fields[0];
        let field_path = self.schema().field_path(field_id)?;
        let logical_index = self
            .open_logical_vector_index(&field_path, index_name)
            .await?;
        logical_index
            .as_ivf()?
            .read_partition(partition_id, with_vector)
            .await
    }
}

fn index_group_is_scalar(dataset: &Dataset, deltas: &[&IndexMetadata]) -> bool {
    let Some(field_id) = deltas.first().and_then(|d| d.fields.first()) else {
        return false;
    };
    match dataset.schema().field_by_id(*field_id) {
        Some(field) => !is_vector_field(field.data_type()),
        None => false,
    }
}

fn index_group_has_no_unindexed(dataset: &Dataset, deltas: &[&IndexMetadata]) -> bool {
    let mut indexed = RoaringBitmap::new();
    for idx in deltas {
        if let Some(bitmap) = idx.fragment_bitmap.as_ref() {
            indexed |= bitmap;
        } else {
            // Pre-0.8 indices have no fragment bitmap; treat as needing optimize.
            return false;
        }
    }
    dataset
        .fragments()
        .iter()
        .all(|frag| indexed.contains(frag.id as u32))
}

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

async fn index_statistics_frag_reuse(ds: &Dataset) -> Result<String> {
    let index = ds
        .open_frag_reuse_index(&NoOpMetricsCollector)
        .await?
        .expect("FragmentReuse index does not exist");
    serialize_index_statistics(&CompactFragReuseIndexHandle(index).statistics()?)
}

async fn index_statistics_mem_wal(ds: &Dataset) -> Result<String> {
    let index = ds
        .open_mem_wal_index(&NoOpMetricsCollector)
        .await?
        .expect("MemWal index does not exist");
    serialize_index_statistics(&MemWalIndexHandle(index).statistics()?)
}

async fn index_statistics_scalar(
    ds: &Dataset,
    index_name: &str,
    metadatas: Vec<IndexMetadata>,
) -> Result<String> {
    let field_id = metadatas[0].fields[0];
    let field_path = ds.schema().field_path(field_id)?;

    let (indices_stats, index_uri, num_indices, updated_at) =
        collect_regular_indices_statistics(ds, metadatas, &field_path).await?;

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
        return migrate_and_recompute_index_statistics(ds, index_name).await;
    };

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

    serialize_index_statistics(&stats)
}

async fn collect_regular_indices_statistics(
    ds: &Dataset,
    metadatas: Vec<IndexMetadata>,
    field_path: &str,
) -> Result<(Vec<serde_json::Value>, String, usize, Option<u64>)> {
    let num_indices = metadatas.len();
    let updated_at = metadatas
        .iter()
        .filter_map(|m| m.created_at)
        .max()
        .map(|dt| dt.timestamp_millis() as u64);

    let mut indices_stats = Vec::with_capacity(num_indices);
    let mut index_uri: Option<String> = None;

    for meta in metadatas.iter() {
        let index_store = Arc::new(LanceIndexStore::from_dataset_for_existing(ds, meta).await?);
        let index_details = scalar::fetch_index_details(ds, field_path, meta).await?;
        if index_uri.is_none() {
            index_uri = Some(index_details.type_url.clone());
        }

        let index_details_wrapper = scalar::IndexDetails(index_details.clone());
        if let Ok(plugin) = index_details_wrapper.get_plugin()
            && let Some(stats) = plugin
                .load_statistics(index_store.clone(), index_details.as_ref())
                .await?
        {
            indices_stats.push(stats);
            continue;
        }

        let index = ds
            .open_generic_index(field_path, &meta.uuid, &NoOpMetricsCollector)
            .await?;

        indices_stats.push(index.statistics()?);
    }

    Ok((
        indices_stats,
        index_uri.unwrap_or_else(|| "unknown".to_string()),
        num_indices,
        updated_at,
    ))
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

/// `None` when this build supports the index's version, otherwise the highest
/// version it does support.
///
/// Only a version bump of a type this build already has a plugin for is caught.
/// Reader availability is checked separately by [`index_is_usable`], because
/// an unknown type has no meaningful maximum version in this build.
pub(crate) fn unsupported_index_version(index: &IndexMetadata) -> Option<u32> {
    let max_supported_version = index
        .index_details
        .as_ref()
        .map(|details| {
            IndexDetails(details.clone())
                .index_version()
                .unwrap_or(i32::MAX as u32)
        })
        .unwrap_or_default();
    (index.index_version > max_supported_version as i32).then_some(max_supported_version)
}

/// Whether this build may expose an index through the usable-index view.
///
/// System indices have dedicated readers rather than scalar plugins. Ordinary
/// indices need both a reader for their exact declared type and a supported
/// format version.
pub(crate) fn index_is_usable(index: &IndexMetadata) -> bool {
    index_type_is_known(index) && unsupported_index_version(index).is_none()
}

/// Name the indices this build has no reader for, once per manifest read.
///
/// Deliberately not inside the filter in [`DatasetIndexExt::load_indices`]: that
/// runs on every call, and `load_indices` sits on the query-planning path and on
/// merge_insert's per-batch path. Warning there would cost an operator one line
/// per hidden index per query for as long as the dataset carries one.
pub(crate) fn warn_about_unsupported_indices(indices: &[IndexMetadata]) {
    for idx in indices {
        if !index_type_is_known(idx) {
            let type_url = idx
                .index_details
                .as_ref()
                .map(|details| details.type_url.as_str())
                .unwrap_or("<legacy>");
            log::warn!(
                "Index {} has unrecognized type {}, ignoring it",
                idx.name,
                type_url,
            );
        } else if let Some(max_supported_version) = unsupported_index_version(idx) {
            log::warn!(
                "Index {} has version {}, which is not supported (<={}), ignoring it",
                idx.name,
                idx.index_version,
                max_supported_version,
            );
        }
    }
}

/// Every index the manifest names, including any this build has no reader for.
///
/// Separate from [`DatasetIndexExt::load_indices`] because the two answer
/// different questions. A reader asks which indices it may *use*, and an index
/// it cannot decode is rightly absent. Everything that decides what the *next*
/// manifest looks like asks a different question, and there the same omission is
/// not a filter but an erasure. `build_manifest` seeds the new index list from
/// what it is handed, so an index left out disappears from the dataset for every
/// build, including the one that could have read it. Index bookkeeping - name
/// reservation, replace and removal selection, explicit drop - answers that
/// second question too: a name it cannot see is a name it will hand out twice.
pub(crate) async fn load_all_indices(dataset: &Dataset) -> Result<Arc<Vec<IndexMetadata>>> {
    let metadata_key = IndexMetadataKey {
        version: dataset.version().version,
        store_identity: &dataset.object_store.store_prefix,
        e_tag: dataset.manifest_location.e_tag.as_deref(),
    };
    let mut indices = dataset
        .index_cache
        .get_or_insert_with_key(metadata_key, || async {
            let loaded = read_manifest_indexes(
                &dataset.object_store,
                &dataset.manifest_location,
                &dataset.manifest,
            )
            .await?;
            warn_about_unsupported_indices(&loaded);
            Ok(loaded)
        })
        .await?;

    // Infer details for legacy vector indices (once per index name, concurrently).
    // This may run on indices that were opportunistically cached during Dataset::open
    // before the full Dataset was available for inference.
    {
        let schema = dataset.schema();
        if indices
            .iter()
            .any(|idx| needs_vector_details_inference(idx, schema))
        {
            let mut updated = indices.as_ref().clone();
            infer_missing_vector_details(dataset, &mut updated).await;
            if updated != *indices {
                indices = Arc::new(updated);
                dataset
                    .index_cache
                    .insert_with_key(&metadata_key, indices.clone())
                    .await;
            }
        }
    }

    if let Some(frag_reuse_index_meta) =
        indices.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
    {
        let fri_key = FragReuseIndexKey {
            uuid: &frag_reuse_index_meta.uuid,
        };
        let frag_reuse_index = dataset
            .index_cache
            .get_or_insert_with_key(fri_key, || async move {
                let index_details =
                    load_frag_reuse_index_details(dataset, frag_reuse_index_meta).await?;
                open_frag_reuse_index(frag_reuse_index_meta.uuid, index_details.as_ref()).await
            })
            .await?;
        let mut indices = indices.as_ref().clone();
        for idx in indices.iter_mut() {
            if let Some(bitmap) = idx.fragment_bitmap.as_mut() {
                frag_reuse_index.remap_fragment_bitmap(bitmap)?;
            }
        }
        Ok(Arc::new(indices))
    } else {
        Ok(indices)
    }
}

/// The segments named `name`, including any this build has no reader for.
///
/// The bookkeeping counterpart to [`DatasetIndexExt::load_indices_by_name`]. See
/// [`load_all_indices`] for which of the two a call site wants.
pub(crate) async fn load_all_indices_by_name(
    dataset: &Dataset,
    name: &str,
) -> Result<Vec<IndexMetadata>> {
    Ok(load_all_indices(dataset)
        .await?
        .iter()
        .filter(|idx| idx.name == name)
        .cloned()
        .collect())
}

/// A trait for internal dataset utilities
///
/// Internal use only. No API stability guarantees.
#[async_trait]
pub trait DatasetIndexInternalExt: DatasetIndexExt {
    /// Opens an index (scalar or vector) as a generic index
    async fn open_generic_index(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn Index>>;
    /// Opens the requested scalar index
    async fn open_scalar_index(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>>;
    /// Opens the requested vector index
    async fn open_vector_index(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndex>>;
    /// Opens all segments for one logical vector index and returns a materialized snapshot.
    async fn open_logical_vector_index(
        &self,
        column: &str,
        name: &str,
    ) -> Result<LogicalVectorIndex>;

    /// Opens the fragment reuse index
    async fn open_frag_reuse_index(
        &self,
        metrics: &dyn MetricsCollector,
    ) -> Result<Option<Arc<CompactFragReuseIndex>>>;

    /// Opens the MemWAL index
    async fn open_mem_wal_index(
        &self,
        metrics: &dyn MetricsCollector,
    ) -> Result<Option<Arc<MemWalIndex>>>;

    /// Gets the fragment reuse index UUID from the current manifest, if it exists
    async fn frag_reuse_index_uuid(&self) -> Option<Uuid>;

    /// Loads information about all the available scalar indices on the dataset
    async fn scalar_index_info(&self) -> Result<ScalarIndexInfo>;

    /// Return the fragments that are not covered by any of the deltas of the index.
    async fn unindexed_fragments(&self, idx_name: &str) -> Result<Vec<Fragment>>;

    /// Return the fragments that are covered by each of the deltas of the index.
    async fn indexed_fragments(&self, idx_name: &str) -> Result<Vec<Vec<Fragment>>>;

    /// Initialize a specific index on this dataset based on an index from a source dataset.
    async fn initialize_index(&mut self, source_dataset: &Dataset, index_name: &str) -> Result<()>;

    /// Initialize all indices on this dataset based on indices from a source dataset.
    /// This will call `initialize_index` for each non-system index in the source dataset.
    async fn initialize_indices(&mut self, source_dataset: &Dataset) -> Result<()>;
}

#[async_trait]
impl DatasetIndexInternalExt for Dataset {
    async fn open_generic_index(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn Index>> {
        // Checking for cache existence is cheap so we just check the vector caches.
        // Scalar indices cache themselves inside `open_scalar_index` (the cache
        // key is a plugin detail), so there is no cheap scalar check here.
        let frag_reuse_uuid = self.frag_reuse_index_uuid().await;

        // Check sized cache for IvfIndexState (v2+ indices).
        let state_key = IvfIndexStateCacheKey::new(uuid, frag_reuse_uuid.as_ref());
        if self.index_cache.get_with_key(&state_key).await.is_some() {
            // Reconstruct via open_vector_index which will hit the same sized key.
            let index = self.open_vector_index(column, uuid, metrics).await?;
            return Ok(index.as_index());
        }

        // Fallback: in-memory cache for legacy indices.
        let vector_cache_key = LegacyVectorIndexCacheKey::new(uuid, frag_reuse_uuid.as_ref());
        if let Some(cached) = self.index_cache.get_with_key(&vector_cache_key).await {
            return Ok(cached.0.clone().as_index());
        }

        let frag_reuse_cache_key = FragReuseIndexCacheKey::new(uuid, frag_reuse_uuid.as_ref());
        if let Some(index) = self.index_cache.get_with_key(&frag_reuse_cache_key).await {
            return Ok(Arc::new(CompactFragReuseIndexHandle(index)).as_index());
        }

        // Sometimes we want to open an index and we don't care if it is a scalar or vector index.
        // For example, we might want to get statistics for an index, regardless of type.
        //
        // We determine if this is a vector index by checking if INDEX_FILE_NAME exists in the
        // file list (available since file sizes tracking was added). If the file list is not
        // available (older indices), we fall back to checking file existence via HEAD request.
        let index_meta = self
            .load_index(uuid)
            .await?
            .ok_or_else(|| Error::index(format!("Index with id {} does not exist", uuid)))?;

        // Check if this is a vector index by looking at the files list
        let is_vector_index = if let Some(files) = &index_meta.files {
            // If we have file metadata, check if INDEX_FILE_NAME is in the list
            files.iter().any(|f| f.path == INDEX_FILE_NAME)
        } else {
            // Fall back to file existence check for older indices without file metadata
            let index_dir = self.indice_files_dir(&index_meta)?;
            let index_file = index_dir
                .clone()
                .join(uuid.to_string())
                .join(INDEX_FILE_NAME);
            let object_store = self.object_store_for_index(&index_meta).await?;
            object_store.exists(&index_file).await?
        };

        if is_vector_index {
            let index = self.open_vector_index(column, uuid, metrics).await?;
            Ok(index.as_index())
        } else {
            let index = self.open_scalar_index(column, uuid, metrics).await?;
            Ok(index.as_index())
        }
    }

    #[instrument(level = "debug", skip_all)]
    async fn open_scalar_index(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        // Caching (including the choice of in-memory vs. serializable state) is
        // a plugin implementation detail handled inside `scalar::open_scalar_index`.
        let index_meta = self
            .load_index(uuid)
            .await?
            .ok_or_else(|| Error::index(format!("Index with id {} does not exist", uuid)))?;

        scalar::open_scalar_index(self, column, &index_meta, metrics).await
    }

    async fn open_vector_index(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndex>> {
        let frag_reuse_uuid = self.frag_reuse_index_uuid().await;
        let index_meta = self
            .load_index(uuid)
            .await?
            .ok_or_else(|| Error::index(format!("Index with id {} does not exist", uuid)))?;
        let object_store = self.object_store_for_index(&index_meta).await?;

        // Check sized cache first (v2+ indices with serializable state).
        let state_key = IvfIndexStateCacheKey::new(uuid, frag_reuse_uuid.as_ref());
        if let Some(entry) = self.index_cache.get_with_key(&state_key).await {
            log::debug!("Found IvfIndexState in cache uuid: {}", uuid);
            let partition_cache = self.index_cache.for_index(uuid, frag_reuse_uuid.as_ref());
            let frag_reuse_index = self.open_frag_reuse_index(metrics).await?;
            return entry
                .0
                .reconstruct(
                    object_store,
                    self.metadata_cache.as_ref(),
                    partition_cache,
                    frag_reuse_index,
                )
                .await;
        }

        // Fallback: in-memory cache for legacy indices.
        let cache_key = LegacyVectorIndexCacheKey::new(uuid, frag_reuse_uuid.as_ref());
        if let Some(cached) = self.index_cache.get_with_key(&cache_key).await {
            return Ok(cached.0.clone());
        }

        let frag_reuse_index = self.open_frag_reuse_index(metrics).await?;
        let index_dir = self.indice_files_dir(&index_meta)?;
        let index_file = index_dir
            .clone()
            .join(uuid.to_string())
            .join(INDEX_FILE_NAME);
        let file_sizes = index_meta.file_size_map();
        let reader: Arc<dyn Reader> = vector::open_index_file(
            object_store.as_ref(),
            &index_file,
            INDEX_FILE_NAME,
            &file_sizes,
        )
        .await?
        .into();

        let tailing_bytes = read_last_block(reader.as_ref()).await?;
        let (major_version, minor_version) = read_version(&tailing_bytes)?;

        // Namespace the index cache by the UUID of the index. v2+ partition
        // entries are store-free and remain reusable across object-store
        // generations alongside their serializable state.
        let index_cache = self.index_cache.for_index(uuid, frag_reuse_uuid.as_ref());

        // Extract the cacheable state before type-erasing to Arc<dyn VectorIndex>.
        fn wrap_ivf<S: IvfSubIndex + 'static, Q: Quantization + 'static>(
            ivf: IVFIndex<S, Q>,
        ) -> (Arc<dyn VectorIndex>, Option<IvfStateEntryBox>) {
            let entry = ivf.to_state_entry();
            (Arc::new(ivf), Some(entry))
        }

        // the index file is in lance format since version (0,2)
        // TODO: we need to change the legacy IVF_PQ to be in lance format
        let result: Result<(Arc<dyn VectorIndex>, Option<IvfStateEntryBox>)> = match (
            major_version,
            minor_version,
        ) {
            (0, 1) | (0, 0) => {
                info!(target: TRACE_IO_EVENTS, index_uuid=%uuid, r#type=IO_TYPE_OPEN_VECTOR, version="0.1", index_type="IVF_PQ");
                let proto = open_index_proto(reader.as_ref()).await?;
                match &proto.implementation {
                    Some(Implementation::VectorIndex(vector_index)) => {
                        let dataset = Arc::new(self.clone());
                        let idx = vector::open_vector_index(
                            dataset,
                            uuid,
                            vector_index,
                            reader,
                            frag_reuse_index,
                        )
                        .await?;
                        Ok((idx, None::<IvfStateEntryBox>))
                    }
                    None => Err(Error::internal(
                        "Index proto was missing implementation field",
                    )),
                }
            }

            (0, 2) => {
                info!(target: TRACE_IO_EVENTS, index_uuid=%uuid, r#type=IO_TYPE_OPEN_VECTOR, version="0.2", index_type="IVF_PQ");
                let reader = V1FileReader::try_new_self_described_from_reader(
                    reader.clone(),
                    Some(&self.metadata_cache.file_metadata_cache(&index_file)),
                )
                .await?;
                let idx = vector::open_vector_index_v2(
                    Arc::new(self.clone()),
                    column,
                    uuid,
                    reader,
                    frag_reuse_index,
                )
                .await?;
                Ok((idx, None::<IvfStateEntryBox>))
            }

            (0, 3) | (2, _) => {
                let scheduler = ScanScheduler::new(
                    object_store.clone(),
                    SchedulerConfig::max_bandwidth(&object_store),
                );
                let cached_size = file_sizes
                    .get(INDEX_FILE_NAME)
                    .map(|&size| CachedFileSize::new(size))
                    .unwrap_or_else(CachedFileSize::unknown);
                let file = scheduler.open_file(&index_file, &cached_size).await?;
                let reader = lance_file::reader::FileReader::try_open(
                    file,
                    None,
                    Default::default(),
                    &self.metadata_cache.file_metadata_cache(&index_file),
                    FileReaderOptions::default(),
                )
                .await?;
                let index_metadata = reader
                    .schema()
                    .metadata
                    .get(INDEX_METADATA_SCHEMA_KEY)
                    .ok_or(Error::index("Index Metadata not found".to_owned()))?;
                let index_metadata: lance_index::IndexMetadata =
                    serde_json::from_str(index_metadata)?;

                // Resolve the column name and field
                let (field_path, field) = resolve_index_column(self.schema(), &index_meta, column)?;

                let (_, element_type) = get_vector_type(self.schema(), &field_path)?;

                info!(target: TRACE_IO_EVENTS, index_uuid=%uuid, r#type=IO_TYPE_OPEN_VECTOR, version="0.3", index_type=index_metadata.index_type);

                match index_metadata.index_type.as_str() {
                    "IVF_FLAT" => match element_type {
                        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                            let ivf = IVFIndex::<FlatIndex, FlatQuantizer>::try_new(
                                object_store.clone(),
                                index_dir,
                                uuid.to_owned(),
                                frag_reuse_index,
                                self.metadata_cache.as_ref(),
                                index_cache,
                                file_sizes,
                            )
                            .await?;
                            Ok(wrap_ivf(ivf))
                        }
                        DataType::UInt8 => {
                            let ivf = IVFIndex::<FlatIndex, FlatBinQuantizer>::try_new(
                                object_store.clone(),
                                index_dir,
                                uuid.to_owned(),
                                frag_reuse_index,
                                self.metadata_cache.as_ref(),
                                index_cache,
                                file_sizes,
                            )
                            .await?;
                            Ok(wrap_ivf(ivf))
                        }
                        _ => Err(Error::index(format!(
                            "the field type {} is not supported for FLAT index",
                            field.data_type()
                        ))),
                    },

                    "IVF_PQ" => {
                        let ivf = IVFIndex::<FlatIndex, ProductQuantizer>::try_new(
                            object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            frag_reuse_index,
                            self.metadata_cache.as_ref(),
                            index_cache,
                            file_sizes,
                        )
                        .await?;
                        Ok(wrap_ivf(ivf))
                    }

                    "IVF_SQ" => {
                        let ivf = IVFIndex::<FlatIndex, ScalarQuantizer>::try_new(
                            object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            frag_reuse_index,
                            self.metadata_cache.as_ref(),
                            index_cache,
                            file_sizes,
                        )
                        .await?;
                        Ok(wrap_ivf(ivf))
                    }

                    "IVF_RQ" => {
                        let ivf = IVFIndex::<FlatIndex, RabitQuantizer>::try_new(
                            object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            frag_reuse_index,
                            self.metadata_cache.as_ref(),
                            index_cache,
                            file_sizes,
                        )
                        .await?;
                        Ok(wrap_ivf(ivf))
                    }

                    "IVF_HNSW_FLAT" => match element_type {
                        DataType::UInt8 => {
                            let ivf = IVFIndex::<HNSW, FlatBinQuantizer>::try_new(
                                object_store.clone(),
                                index_dir,
                                uuid.to_owned(),
                                frag_reuse_index,
                                self.metadata_cache.as_ref(),
                                index_cache,
                                file_sizes,
                            )
                            .await?;
                            Ok(wrap_ivf(ivf))
                        }
                        _ => {
                            let ivf = IVFIndex::<HNSW, FlatQuantizer>::try_new(
                                object_store.clone(),
                                index_dir,
                                uuid.to_owned(),
                                frag_reuse_index,
                                self.metadata_cache.as_ref(),
                                index_cache,
                                file_sizes,
                            )
                            .await?;
                            Ok(wrap_ivf(ivf))
                        }
                    },

                    "IVF_HNSW_SQ" => {
                        let ivf = IVFIndex::<HNSW, ScalarQuantizer>::try_new(
                            object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            frag_reuse_index,
                            self.metadata_cache.as_ref(),
                            index_cache,
                            file_sizes,
                        )
                        .await?;
                        Ok(wrap_ivf(ivf))
                    }

                    "IVF_HNSW_PQ" => {
                        let ivf = IVFIndex::<HNSW, ProductQuantizer>::try_new(
                            object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            frag_reuse_index,
                            self.metadata_cache.as_ref(),
                            index_cache,
                            file_sizes,
                        )
                        .await?;
                        Ok(wrap_ivf(ivf))
                    }

                    _ => Err(Error::index(format!(
                        "Unsupported index type: {}",
                        index_metadata.index_type
                    ))),
                }
            }

            _ => Err(Error::index(
                "unsupported index version (maybe need to upgrade your lance version)".to_owned(),
            )),
        };
        let (index, ivf_entry) = result?;
        metrics.record_index_load();
        // Attribute the one-time index-open I/O (file footers, IVF centroids,
        // quantization metadata) to this query's metrics.  This runs only on a
        // real open; cache hits return earlier, so a warm query reports zero
        // index-open I/O.
        if let Some(io_stats) = metrics.io_stats()
            && let Some(open_stats) = index.open_io_stats()
        {
            io_stats.add_scan_stats(&open_stats);
        }
        if let Some(ivf_entry) = ivf_entry {
            self.index_cache
                .insert_with_key(&state_key, Arc::new(ivf_entry))
                .await;
        } else {
            self.index_cache
                .insert_with_key(&cache_key, Arc::new(CachedLegacyVectorIndex(index.clone())))
                .await;
        }
        Ok(index)
    }

    async fn open_logical_vector_index(
        &self,
        column: &str,
        name: &str,
    ) -> Result<LogicalVectorIndex> {
        let metadatas = self.load_indices_by_name(name).await?;
        if metadatas.is_empty() {
            return Err(Error::index_not_found(format!("name={name}")));
        }

        let field_id = self.schema().field_id(column)?;
        if let Some(invalid_metadata) = metadatas
            .iter()
            .find(|metadata| metadata.fields.first() != Some(&field_id))
        {
            return Err(Error::invalid_input(format!(
                "Logical vector index '{}' contains segment {} that does not belong to column '{}'",
                name, invalid_metadata.uuid, column
            )));
        }

        let mut segments = Vec::with_capacity(metadatas.len());
        for metadata in metadatas {
            let index = self
                .open_vector_index(column, &metadata.uuid, &NoOpMetricsCollector)
                .await?;
            segments.push((metadata, index));
        }

        LogicalVectorIndex::try_new(name.to_string(), column.to_string(), segments)
    }

    async fn open_frag_reuse_index(
        &self,
        metrics: &dyn MetricsCollector,
    ) -> Result<Option<Arc<CompactFragReuseIndex>>> {
        if let Some(frag_reuse_index_meta) = self.load_index_by_name(FRAG_REUSE_INDEX_NAME).await? {
            let frag_reuse_uuid = frag_reuse_index_meta.uuid;
            let frag_reuse_key = FragReuseIndexKey {
                uuid: &frag_reuse_uuid,
            };

            let index = self
                .index_cache
                .get_or_insert_with_key(frag_reuse_key, || async move {
                    let index_details =
                        load_frag_reuse_index_details(self, &frag_reuse_index_meta).await?;
                    let index =
                        open_frag_reuse_index(frag_reuse_index_meta.uuid, index_details.as_ref())
                            .await?;

                    info!(target: TRACE_IO_EVENTS, index_uuid=%frag_reuse_uuid, r#type=IO_TYPE_OPEN_FRAG_REUSE);
                    metrics.record_index_load();

                    Ok(index)
                })
                .await?;

            Ok(Some(index))
        } else {
            Ok(None)
        }
    }

    async fn open_mem_wal_index(
        &self,
        metrics: &dyn MetricsCollector,
    ) -> Result<Option<Arc<MemWalIndex>>> {
        let Some(mem_wal_meta) = self.load_index_by_name(MEM_WAL_INDEX_NAME).await? else {
            return Ok(None);
        };

        let frag_reuse_uuid = self.frag_reuse_index_uuid().await;
        let cache_key = MemWalCacheKey::new(&mem_wal_meta.uuid, frag_reuse_uuid.as_ref());
        if let Some(index) = self.index_cache.get_with_key(&cache_key).await {
            log::debug!("Found MemWAL index in cache uuid: {}", mem_wal_meta.uuid);
            return Ok(Some(index));
        }

        let uuid = mem_wal_meta.uuid;

        let index_meta = self
            .load_index(&uuid)
            .await?
            .ok_or_else(|| Error::index(format!("Index with id {} does not exist", uuid)))?;
        let index = open_mem_wal_index(index_meta)?;

        info!(target: TRACE_IO_EVENTS, index_uuid=%uuid, r#type=IO_TYPE_OPEN_MEM_WAL);
        metrics.record_index_load();

        self.index_cache
            .insert_with_key(&cache_key, index.clone())
            .await;
        Ok(Some(index))
    }

    async fn frag_reuse_index_uuid(&self) -> Option<Uuid> {
        if let Ok(indices) = self.load_indices().await {
            indices
                .iter()
                .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
                .map(|idx| idx.uuid)
        } else {
            None
        }
    }

    #[instrument(level = "trace", skip_all)]
    async fn scalar_index_info(&self) -> Result<ScalarIndexInfo> {
        let indices = self.load_indices().await?;
        let schema = self.schema();
        let mut indexed_fields = Vec::new();
        // (column, index_name) → union of every contributing IndexMetadata's
        // fragment_bitmap. Multiple entries can land here for delta-merged
        // indices that share a name. We only insert when every contributing
        // entry has a bitmap; if any are missing, we leave the entry absent
        // so the optimizer treats coverage as unknown.
        let mut fragment_bitmaps: HashMap<(String, String), Option<RoaringBitmap>> = HashMap::new();
        for index in indices.iter().filter(|idx| {
            // Check if this is an FTS index by looking at index details
            let is_fts_index = if let Some(details) = &idx.index_details {
                IndexDetails(details.clone()).supports_fts()
            } else {
                false
            };

            // Only include indices with non-empty fragment bitmaps, except for FTS indices
            // which need to be discoverable even when empty
            let has_non_empty_bitmap = idx.fragment_bitmap.as_ref().is_some_and(|bitmap| {
                !bitmap.is_empty() && !(bitmap & self.fragment_bitmap.as_ref()).is_empty()
            });

            // Same name-collision rationale as `CreateIndexBuilder`'s
            // default-name loop in create.rs: only the keyed prefix decides
            // which column this index answers for.
            idx.keyed_field().is_some() && (has_non_empty_bitmap || is_fts_index)
        }) {
            let field = index.fields[0];
            let field = schema.field_by_id(field).ok_or_else(|| {
                Error::internal(format!(
                    "Index referenced a field with id {field} which did not exist in the schema"
                ))
            })?;

            // Build the full field path for nested fields
            let field_path = if let Some(ancestors) = schema.field_ancestry_by_id(field.id) {
                let field_refs: Vec<&str> = ancestors.iter().map(|f| f.name.as_str()).collect();
                lance_core::datatypes::format_field_path(&field_refs)
            } else {
                field.name.clone()
            };

            let index_details = IndexDetails(fetch_index_details(self, &field_path, index).await?);
            if index_details.is_vector() {
                continue;
            }

            let plugin = match index_details.get_plugin() {
                Ok(plugin) => plugin,
                Err(e) => {
                    log::warn!(
                        "Skipping index '{}' on column '{}': {}. \
                         Queries on this column will fall back to a full scan.",
                        index.name,
                        field_path,
                        e
                    );
                    continue;
                }
            };
            let query_parser = plugin.new_query_parser(index.name.clone(), &index_details.0);

            if let Some(query_parser) = query_parser {
                // Union the per-segment fragment bitmap into this
                // (column, index_name) entry. If any segment is missing a
                // bitmap, downgrade the entry to None so callers know
                // coverage is partial/unknown.
                let key = (field_path.clone(), index.name.clone());
                fragment_bitmaps
                    .entry(key)
                    .and_modify(|entry| {
                        if let (Some(acc), Some(seg)) =
                            (entry.as_mut(), index.fragment_bitmap.as_ref())
                        {
                            *acc |= seg;
                        } else {
                            *entry = None;
                        }
                    })
                    .or_insert_with(|| index.fragment_bitmap.clone());
                indexed_fields.push((field_path, (field.data_type(), query_parser)));
            }
        }
        let mut index_info_map = HashMap::with_capacity(indexed_fields.len());
        for indexed_field in indexed_fields {
            // Need to wrap in an option here because we know that only one of and_modify and or_insert will be called
            // but the rust compiler does not.
            let mut parser = Some(indexed_field.1.1);
            let parser = &mut parser;
            index_info_map
                .entry(indexed_field.0)
                .and_modify(|existing: &mut (DataType, Box<MultiQueryParser>)| {
                    // If there are two indices on the same column, they must have the same type
                    debug_assert_eq!(existing.0, indexed_field.1.0);

                    existing.1.add(parser.take().unwrap());
                })
                .or_insert_with(|| {
                    (
                        indexed_field.1.0,
                        Box::new(MultiQueryParser::single(parser.take().unwrap())),
                    )
                });
        }
        // Drop entries we couldn't pin to a known bitmap.
        let fragment_bitmaps = fragment_bitmaps
            .into_iter()
            .filter_map(|(k, v)| v.map(|bm| (k, bm)))
            .collect();
        Ok(ScalarIndexInfo {
            indexed_columns: index_info_map,
            fragment_bitmaps,
        })
    }

    async fn unindexed_fragments(&self, name: &str) -> Result<Vec<Fragment>> {
        let indices = self.load_indices_by_name(name).await?;
        let mut total_fragment_bitmap = RoaringBitmap::new();
        for idx in indices.iter() {
            total_fragment_bitmap |= idx.fragment_bitmap.as_ref().ok_or(Error::index(
                "Please upgrade lance to 0.8+ to use this function".to_string(),
            ))?;
        }
        Ok(self
            .fragments()
            .iter()
            .filter(|f| !total_fragment_bitmap.contains(f.id as u32))
            .cloned()
            .collect())
    }

    async fn indexed_fragments(&self, name: &str) -> Result<Vec<Vec<Fragment>>> {
        let indices = self.load_indices_by_name(name).await?;
        indices
            .iter()
            .map(|index| {
                let fragment_bitmap = index.fragment_bitmap.as_ref().ok_or(Error::index(
                    "Please upgrade lance to 0.8+ to use this function".to_string(),
                ))?;
                let mut indexed_frags = Vec::with_capacity(fragment_bitmap.len() as usize);
                for frag in self.fragments().iter() {
                    if fragment_bitmap.contains(frag.id as u32) {
                        indexed_frags.push(frag.clone());
                    }
                }
                Ok(indexed_frags)
            })
            .collect()
    }

    async fn initialize_index(&mut self, source_dataset: &Dataset, index_name: &str) -> Result<()> {
        let source_indices = source_dataset.load_indices_by_name(index_name).await?;

        if source_indices.is_empty() {
            return Err(Error::index(format!(
                "Index '{}' not found in source dataset",
                index_name
            )));
        }

        let source_index = source_indices
            .iter()
            .min_by_key(|idx| idx.created_at)
            .ok_or_else(|| {
                Error::index(format!(
                    "Could not determine oldest index for '{}'",
                    index_name
                ))
            })?;

        // Only the keyed prefix matters here. The rebuild below writes fresh,
        // non-covered metadata and never reads the carried columns, so requiring
        // them to exist and type-match would reject a target that can hold this
        // index perfectly well.
        let keyed_fields = source_index.keyed_fields();
        let mut field_paths = Vec::with_capacity(keyed_fields.len());
        for field_id in keyed_fields.iter() {
            let source_field = source_dataset
                .schema()
                .field_by_id(*field_id)
                .ok_or_else(|| {
                    Error::index(format!(
                        "Field with id {} not found in source dataset",
                        field_id
                    ))
                })?;
            let source_field_path = source_dataset.schema().field_path(*field_id)?;

            let target_field = self.schema().field(&source_field_path).ok_or_else(|| {
                Error::index(format!(
                    "Field '{}' required by index '{}' not found in target dataset",
                    source_field_path, index_name
                ))
            })?;

            if source_field.data_type() != target_field.data_type() {
                return Err(Error::index(format!(
                    "Field '{}' has different types in source ({:?}) and target ({:?}) datasets",
                    source_field_path,
                    source_field.data_type(),
                    target_field.data_type()
                )));
            }

            field_paths.push(source_field_path);
        }

        if field_paths.is_empty() {
            return Err(Error::index(format!(
                "Index '{}' has no fields",
                index_name
            )));
        }

        // `field_paths` is already the keyed prefix -- it was built from `keyed_fields()`
        // above -- so a carried column is deliberately NOT resolved or type-checked here.
        // That is the point: the rebuild writes fresh, non-covered metadata and never reads
        // the carried payload, so demanding those columns exist in the target would reject
        // a target that can hold this index perfectly well.
        //
        // `initialize_vector_index` then re-resolves the carried columns itself, by name,
        // against the target schema. `initialize_scalar_index` does NOT -- it recreates
        // with default params and never reads `covering_fields`, so a covered scalar index
        // would land on the target with covering silently dropped. Unreachable today:
        // `covering_columns` exists only on `VectorIndexParams`, so nothing produces a
        // covered scalar index. Whoever adds a scalar producer must fix that path too.
        let field_names = field_paths.iter().map(String::as_str).collect::<Vec<_>>();
        if let Some(index_details) = &source_index.index_details {
            let index_details_wrapper = IndexDetails(index_details.clone());

            if index_details_wrapper.is_vector() {
                vector::initialize_vector_index(self, source_dataset, source_index, &field_names)
                    .await?;
            } else {
                scalar::initialize_scalar_index(self, source_dataset, source_index, &field_names)
                    .await?;
            }
        } else {
            log::warn!(
                "Index '{}' has no index_details, skipping",
                source_index.name
            );
        }

        Ok(())
    }

    async fn initialize_indices(&mut self, source_dataset: &Dataset) -> Result<()> {
        let source_indices = source_dataset.load_indices().await?;
        let non_system_indices: Vec<_> = source_indices
            .iter()
            .filter(|idx| !lance_index::is_system_index(idx))
            .collect();

        if non_system_indices.is_empty() {
            return Ok(());
        }

        let mut unique_index_names = HashSet::new();
        for index in non_system_indices.iter() {
            unique_index_names.insert(index.name.clone());
        }

        for index_name in unique_index_names {
            self.initialize_index(source_dataset, &index_name).await?;
        }

        Ok(())
    }
}

/// Resolves the column name and field for an index operation.
///
/// This function handles the case where the caller passes an index name instead of a column name.
/// It returns the full field path and the field reference.
fn resolve_index_column(
    schema: &LanceSchema,
    index_meta: &IndexMetadata,
    column_arg: &str,
) -> Result<(String, Arc<Field>)> {
    // First, try to find the column directly in the schema
    if let Some(field) = schema.field(column_arg) {
        // Column exists in schema, use it
        return Ok((column_arg.to_string(), Arc::new(field.clone())));
    }

    // Column doesn't exist in schema, check if it's the index name
    if column_arg == index_meta.name {
        // Get the actual column from index metadata
        if let Some(field_id) = index_meta.fields.first() {
            let field = schema.field_by_id(*field_id).ok_or_else(|| {
                Error::index(format!(
                    "Index '{}' references field with id {} which does not exist in schema",
                    index_meta.name, field_id
                ))
            })?;
            let field_path = schema.field_path(*field_id)?;
            return Ok((field_path, Arc::new(field.clone())));
        } else {
            return Err(Error::index(format!(
                "Index '{}' has no fields",
                index_meta.name
            )));
        }
    }

    // Column doesn't exist and is not the index name
    Err(Error::index(format!(
        "Column '{}' does not exist in the schema",
        column_arg
    )))
}

fn is_vector_field(data_type: DataType) -> bool {
    match data_type {
        DataType::FixedSizeList(_, _) => true,
        DataType::List(inner) => {
            // If the inner type is a fixed size list, then it is a multivector field
            matches!(inner.data_type(), DataType::FixedSizeList(_, _))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::vector::VectorIndexParams;
    use crate::session::Session;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount, copy_test_data_to_tmp};
    use arrow::array::AsArray;
    use arrow::datatypes::{Float32Type, Int32Type, Int64Type};
    use arrow_array::Int32Array;
    use arrow_array::{
        FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema};
    use futures::stream::TryStreamExt;
    use lance_arrow::*;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::gen_batch;
    use lance_datagen::{BatchCount, ByteCount, Dimension, RowCount, array};
    use lance_index::pbold::{BTreeIndexDetails, InvertedIndexDetails};
    use lance_index::scalar::bitmap::BITMAP_LOOKUP_NAME;
    use lance_index::scalar::inverted::query::{FtsQuery, PhraseQuery};
    use lance_index::scalar::inverted::{
        INVERTED_INDEX_VERSION_V1, INVERTED_INDEX_VERSION_V2, INVERTED_INDEX_VERSION_V3,
    };
    use lance_index::scalar::{
        BuiltinIndexType, FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams,
    };
    use lance_index::vector::{
        hnsw::builder::HnswBuildParams,
        ivf::IvfBuildParams,
        kmeans::{KMeansParams, train_kmeans},
        sq::builder::SQBuildParams,
    };
    use lance_io::{assert_io_eq, assert_io_lt, utils::tracking_store::IoStats};
    use lance_linalg::distance::{DistanceType, MetricType};
    use lance_testing::datagen::generate_random_array;
    use object_store::ObjectStoreExt;
    use rstest::rstest;
    use std::collections::{HashMap, HashSet};

    async fn write_vector_segment_metadata(
        dataset: &Dataset,
        index_name: &str,
        field_id: i32,
        uuid: Uuid,
        fragment_bitmap: impl IntoIterator<Item = u32>,
        payload: &[u8],
    ) -> IndexMetadata {
        let index_path = dataset
            .indices_dir()
            .join(uuid.to_string())
            .join(INDEX_FILE_NAME);
        dataset
            .object_store
            .as_ref()
            .put(&index_path, payload)
            .await
            .unwrap();
        IndexMetadata {
            uuid,
            name: index_name.to_string(),
            fields: vec![field_id],
            covering_fields: vec![],
            dataset_version: dataset.manifest.version,
            fragment_bitmap: Some(fragment_bitmap.into_iter().collect()),
            index_details: Some(Arc::new(vector_index_details_default())),
            index_version: IndexType::Vector.version(),
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            files: Some(vec![lance_table::format::IndexFile {
                path: INDEX_FILE_NAME.to_string(),
                size_bytes: payload.len() as u64,
            }]),
        }
    }

    fn list_io_stats(stats: &IoStats) -> IoStats {
        let requests = stats
            .requests
            .iter()
            .filter(|request| request.method == "list")
            .cloned()
            .collect::<Vec<_>>();
        IoStats {
            read_iops: requests.len() as u64,
            requests,
            ..Default::default()
        }
    }

    fn segment_from_metadata(metadata: &IndexMetadata) -> IndexSegment {
        metadata
            .clone()
            .into_index_segment()
            .expect("test segment metadata should convert to an index segment")
    }

    async fn write_fragmented_vector_dataset(uri: &str, dimension: i32) -> Dataset {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension,
                ),
                false,
            ),
        ]));
        let batches = (0..5)
            .map(|i| {
                let vector_values: Float32Array = (0..dimension * 80)
                    .map(|value| value as f32 + (i * 1000) as f32)
                    .collect();
                let vectors =
                    FixedSizeListArray::try_new_from_values(vector_values, dimension).unwrap();
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(i * 80..(i + 1) * 80)),
                        Arc::new(vectors),
                    ],
                )
            })
            .collect::<std::result::Result<Vec<_>, arrow_schema::ArrowError>>()
            .unwrap();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                max_rows_per_group: 10,
                max_rows_per_file: 80,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    async fn write_skewed_fragmented_vector_dataset(uri: &str, dimension: i32) -> Dataset {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension,
                ),
                false,
            ),
        ]));
        let first_fragment_rows = 20_000;
        let second_fragment_rows = 100;
        let first_values = vec![0.0f32; first_fragment_rows * dimension as usize];
        let second_values = vec![100.0f32; second_fragment_rows * dimension as usize];
        let first_vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(first_values), dimension)
                .unwrap();
        let second_vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(second_values), dimension)
                .unwrap();
        let batches = vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..first_fragment_rows as i32)),
                    Arc::new(first_vectors),
                ],
            )
            .unwrap(),
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(
                        first_fragment_rows as i32
                            ..(first_fragment_rows + second_fragment_rows) as i32,
                    )),
                    Arc::new(second_vectors),
                ],
            )
            .unwrap(),
        ];
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                max_rows_per_group: 1024,
                max_rows_per_file: first_fragment_rows,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    async fn create_segmented_vector_index(
        dataset: &mut Dataset,
        index_name: &str,
        column: &str,
        dimension: i32,
    ) -> Vec<Uuid> {
        let batch = dataset
            .scan()
            .project(&[column])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = batch
            .column_by_name(column)
            .expect("vector column should exist")
            .as_fixed_size_list();
        let values = vectors.values().as_primitive::<Float32Type>();
        let centroids = train_kmeans::<Float32Type>(
            values,
            KMeansParams::new(None, 10, 1, DistanceType::L2),
            dimension as usize,
            2,
            2,
        )
        .unwrap()
        .centroids
        .as_primitive::<Float32Type>()
        .clone();
        let centroids =
            Arc::new(FixedSizeListArray::try_new_from_values(centroids, dimension).unwrap());
        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            IvfBuildParams::try_with_centroids(2, centroids).unwrap(),
        );
        let fragment_ids = dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<Vec<_>>();
        let columns = [column];

        let mut segments = Vec::with_capacity(fragment_ids.len());
        for fragment_id in fragment_ids {
            let mut builder = dataset.create_index_builder(&columns, IndexType::Vector, &params);
            builder = builder
                .name(index_name.to_string())
                .fragments(vec![fragment_id]);
            segments.push(builder.execute_uncommitted().await.unwrap());
        }

        let segment_ids = segments
            .iter()
            .map(|segment| segment.uuid)
            .collect::<Vec<_>>();
        dataset
            .commit_existing_index_segments(index_name, column, segments)
            .await
            .unwrap();
        segment_ids
    }

    /// Premise guard, not a regression test: confirms that the broken shape
    /// (`fields` truncated to the keyed field while `covering_fields` is
    /// inherited unchanged) is actually rejected by `validate_covering_fields`,
    /// and that the fixed shape (both fields carried through) is accepted.
    /// Both shapes are hand-built here, so this exercises no `merge_segments`
    /// code and would keep passing even if every impl regressed to the broken
    /// shape. The real regression coverage is
    /// `test_merge_existing_index_segments_preserves_covering_bitmap`,
    /// `_btree`, `_vector`, and `_rtree` below, which merge real segments
    /// through the actual `merge_segments` implementations.
    #[test]
    fn test_merged_covered_metadata_is_committable() {
        let source = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "covered".to_string(),
            fields: vec![7, 11],
            covering_fields: vec![11],
            dataset_version: 1,
            fragment_bitmap: Some(RoaringBitmap::from_iter([0u32])),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
        };

        // The shape every merge_segments impl builds: keyed field only, carried
        // inherited from the source.
        let field_id = *source.fields.first().unwrap();
        let merged = IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: vec![field_id],
            ..source.clone()
        };
        assert!(
            merged.validate_covering_fields().is_err(),
            "this is the broken shape; if it validates, the premise has changed"
        );

        // What the fix must produce instead.
        let fixed = IndexMetadata {
            uuid: Uuid::new_v4(),
            ..source
        };
        fixed
            .validate_covering_fields()
            .expect("merged covered metadata must be committable");
        assert_eq!(fixed.fields, vec![7, 11]);
        assert_eq!(fixed.covering_fields, vec![11]);
    }

    /// Two-fragment, two-int-column dataset for the scalar-segment covering
    /// tests below. Column "id" is keyed, "carried" plays the covered column.
    async fn write_two_int_column_dataset(uri: &str, rows_per_fragment: i32) -> Dataset {
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("carried", array::step::<Int32Type>())
            .into_reader_rows(
                RowCount::from(rows_per_fragment as u64),
                BatchCount::from(2),
            );
        Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                max_rows_per_file: rows_per_fragment as usize,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    /// Nothing writes a covering declaration yet, so the declaration below is
    /// hand-constructed on top of real single-field segments: merge plumbing
    /// never reads covered-column storage, so a declaration naming a real,
    /// uninvolved column is a faithful stand-in for a genuinely covered segment.
    fn make_covered(
        segment: IndexMetadata,
        keyed_field_id: i32,
        carried_field_id: i32,
    ) -> IndexMetadata {
        IndexMetadata {
            fields: vec![keyed_field_id, carried_field_id],
            covering_fields: vec![carried_field_id],
            ..segment
        }
    }

    /// Build one covered segment per named fragment, merge them, and assert the
    /// merged declaration survived and is committable.
    ///
    /// The per-index-type cases below differ only in how their dataset and
    /// params are built, so the merge and the assertions live here.
    async fn assert_merge_preserves_covering(
        dataset: &mut Dataset,
        column: &str,
        index_type: IndexType,
        params: &dyn IndexParams,
        fragment_ids: Vec<u32>,
        keyed_field_id: i32,
        carried_field_id: i32,
    ) {
        let index_name = format!("covered_{column}");
        let mut covered_segments = Vec::new();
        for fragment_id in fragment_ids {
            let segment = dataset
                .create_index_builder(&[column], index_type, params)
                .name(index_name.clone())
                .fragments(vec![fragment_id])
                .execute_uncommitted()
                .await
                .unwrap();
            covered_segments.push(make_covered(segment, keyed_field_id, carried_field_id));
        }

        let merged = dataset
            .merge_existing_index_segments(covered_segments)
            .await
            .unwrap();

        merged
            .validate_covering_fields()
            .expect("merged covering declaration must be committable");
        assert_eq!(merged.fields, vec![keyed_field_id, carried_field_id]);
        assert_eq!(
            merged.covering_fields,
            vec![carried_field_id],
            "merge_segments must neither drop the covering declaration nor \
             produce an uncommittable one"
        );
    }

    fn all_fragment_ids(dataset: &Dataset) -> Vec<u32> {
        dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect()
    }

    /// Two shapes of the same bug, one per `merge_segments` family:
    ///
    ///   - **struct update** (bitmap, inverted, zonemap, label_list,
    ///     bloomfilter, fmindex): `fields` was truncated to `vec![field_id]`
    ///     while `..segments[0].clone()` re-supplied `covering_fields`,
    ///     producing an *uncommittable* merged segment.
    ///   - **explicit** (btree, ngram): `fields: vec![field_id],
    ///     covering_fields: vec![]` was built outright, *silently dropping* the
    ///     declaration instead.
    #[rstest]
    #[case::struct_update(BuiltinIndexType::Bitmap, IndexType::Bitmap)]
    #[case::explicit(BuiltinIndexType::BTree, IndexType::BTree)]
    #[tokio::test]
    async fn test_merge_existing_index_segments_preserves_covering_scalar(
        #[case] builtin: BuiltinIndexType,
        #[case] index_type: IndexType,
    ) {
        let test_dir = TempStrDir::default();
        let mut dataset = write_two_int_column_dataset(&test_dir, 20).await;
        let id_field_id = dataset.schema().field("id").unwrap().id;
        let carried_field_id = dataset.schema().field("carried").unwrap().id;
        let fragment_ids = all_fragment_ids(&dataset);

        let params = ScalarIndexParams::for_builtin(builtin);
        assert_merge_preserves_covering(
            &mut dataset,
            "id",
            index_type,
            &params,
            fragment_ids,
            id_field_id,
            carried_field_id,
        )
        .await;
    }

    /// Segments that disagree on which columns they carry cannot be folded
    /// into one coherent output declaration: there is no single
    /// `covering_fields` that would describe the merged segment truthfully.
    #[tokio::test]
    async fn test_merge_existing_index_segments_rejects_mismatched_covering_fields() {
        let test_dir = TempStrDir::default();
        let mut dataset = write_two_int_column_dataset(&test_dir, 20).await;
        let id_field_id = dataset.schema().field("id").unwrap().id;
        let carried_field_id = dataset.schema().field("carried").unwrap().id;

        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
        let mut mismatched_segments = Vec::new();
        for (i, fragment) in dataset.get_fragments().into_iter().enumerate() {
            let segment = dataset
                .create_index_builder(&["id"], IndexType::Bitmap, &params)
                .name("mismatched_bitmap".to_string())
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap();
            // First segment declares `carried` covered; the second declares
            // nothing covered -- a genuine disagreement, not just a
            // reordering.
            if i == 0 {
                mismatched_segments.push(make_covered(segment, id_field_id, carried_field_id));
            } else {
                mismatched_segments.push(segment);
            }
        }

        let err = dataset
            .merge_existing_index_segments(mismatched_segments)
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("requires segments with identical fields"),
            "unexpected error: {err}"
        );
    }

    /// Vector shape: `crate::index::vector::ivf::merge_segments` already
    /// preserves both `fields` and `covering_fields` via struct update, but
    /// `merge_existing_index_segments`'s own tail used to unconditionally
    /// overwrite `fields` to `vec![field_id]` afterwards, reproducing the
    /// struct-update bug for every index type it dispatches to.
    #[tokio::test]
    async fn test_merge_existing_index_segments_preserves_covering_vector() {
        const DIMENSION: i32 = 4;
        let test_dir = TempStrDir::default();
        let mut dataset = write_fragmented_vector_dataset(&test_dir, DIMENSION).await;
        let vector_field_id = dataset.schema().field("vector").unwrap().id;
        let id_field_id = dataset.schema().field("id").unwrap().id;

        let batch = dataset
            .scan()
            .project(&["vector"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = batch
            .column_by_name("vector")
            .expect("vector column should exist")
            .as_fixed_size_list();
        let values = vectors.values().as_primitive::<Float32Type>();
        let centroids = train_kmeans::<Float32Type>(
            values,
            KMeansParams::new(None, 10, 1, DistanceType::L2),
            DIMENSION as usize,
            2,
            2,
        )
        .unwrap()
        .centroids
        .as_primitive::<Float32Type>()
        .clone();
        let centroids =
            Arc::new(FixedSizeListArray::try_new_from_values(centroids, DIMENSION).unwrap());
        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            IvfBuildParams::try_with_centroids(2, centroids).unwrap(),
        );

        let fragment_ids = all_fragment_ids(&dataset).into_iter().take(2).collect();

        assert_merge_preserves_covering(
            &mut dataset,
            "vector",
            IndexType::Vector,
            &params,
            fragment_ids,
            vector_field_id,
            id_field_id,
        )
        .await;
    }

    /// Struct-update shape, sweep-found in `rtree.rs`: same bug as the
    /// bitmap/inverted/zonemap/label_list/bloomfilter/fmindex family, not
    /// called out in the original brief but reachable from the same
    /// `all_rtree` dispatch branch this function tests.
    #[cfg(feature = "geo")]
    #[tokio::test]
    async fn test_merge_existing_index_segments_preserves_covering_rtree() {
        use geo_types::line_string;
        use geoarrow_array::GeoArrowArray;
        use geoarrow_array::builder::LineStringBuilder;
        use geoarrow_schema::{Dimension, LineStringType};

        const ROWS_PER_FRAGMENT: i32 = 20;
        let line_string_type = LineStringType::new(Dimension::XY, Default::default());
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            line_string_type.clone().to_field("geometry", true),
        ]));

        let batches = (0..2)
            .map(|fragment: i32| {
                let mut builder = LineStringBuilder::new(line_string_type.clone());
                for row in 0..ROWS_PER_FRAGMENT {
                    let x = (fragment * ROWS_PER_FRAGMENT + row) as f64;
                    builder
                        .push_line_string(Some(&line_string![
                            (x: x, y: x),
                            (x: x + 1.0, y: x + 1.0)
                        ]))
                        .unwrap();
                }
                let ids = Int32Array::from_iter_values(
                    fragment * ROWS_PER_FRAGMENT..(fragment + 1) * ROWS_PER_FRAGMENT,
                );
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(ids), builder.finish().to_array_ref()],
                )
            })
            .collect::<std::result::Result<Vec<_>, arrow_schema::ArrowError>>()
            .unwrap();

        let test_dir = TempStrDir::default();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(
            reader,
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: ROWS_PER_FRAGMENT as usize,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let id_field_id = dataset.schema().field("id").unwrap().id;
        let geometry_field_id = dataset.schema().field("geometry").unwrap().id;
        let fragment_ids = all_fragment_ids(&dataset);

        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::RTree);
        assert_merge_preserves_covering(
            &mut dataset,
            "geometry",
            IndexType::RTree,
            &params,
            fragment_ids,
            geometry_field_id,
            id_field_id,
        )
        .await;
    }

    #[tokio::test]
    async fn test_open_logical_vector_index_single_segment_quality_apis() {
        const DIMENSION: i32 = 8;

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = write_fragmented_vector_dataset(test_uri, DIMENSION).await;
        let params =
            VectorIndexParams::with_ivf_flat_params(DistanceType::L2, IvfBuildParams::new(2));

        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let logical_index = dataset
            .open_logical_vector_index("vector", "vector_idx")
            .await
            .unwrap();

        assert_eq!(logical_index.name(), "vector_idx");
        assert_eq!(logical_index.column(), "vector");
        assert_eq!(logical_index.num_segments(), 1);
        assert_eq!(logical_index.metadatas().len(), 1);

        let rows_per_segment = logical_index.num_rows_per_segment();
        assert_eq!(rows_per_segment.len(), 1);
        assert_eq!(rows_per_segment[0].1, 400);

        let ivf_view = logical_index.as_ivf().unwrap();
        let partitions_per_segment = ivf_view.num_partitions_per_segment();
        assert_eq!(partitions_per_segment, vec![(rows_per_segment[0].0, 2)]);

        let partition_sizes = ivf_view.partition_sizes();
        assert_eq!(partition_sizes.len(), 1);
        assert_eq!(partition_sizes[0].1.len(), 2);
        assert_eq!(
            partition_sizes[0].1.iter().sum::<usize>(),
            rows_per_segment[0].1 as usize
        );
    }

    #[tokio::test]
    async fn test_open_logical_vector_index_segmented_quality_apis() {
        const DIMENSION: i32 = 8;

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = write_fragmented_vector_dataset(test_uri, DIMENSION).await;
        let segment_ids =
            create_segmented_vector_index(&mut dataset, "vector_idx", "vector", DIMENSION).await;

        let logical_index = dataset
            .open_logical_vector_index("vector", "vector_idx")
            .await
            .unwrap();

        assert_eq!(logical_index.name(), "vector_idx");
        assert_eq!(logical_index.column(), "vector");
        assert_eq!(logical_index.num_segments(), segment_ids.len());

        let metadata_ids = logical_index
            .metadatas()
            .map(|metadata| metadata.uuid)
            .collect::<HashSet<_>>();
        assert_eq!(
            metadata_ids,
            segment_ids.into_iter().collect::<HashSet<_>>()
        );

        let rows_per_segment = logical_index.num_rows_per_segment();
        assert_eq!(rows_per_segment.len(), logical_index.num_segments());
        assert_eq!(
            rows_per_segment
                .iter()
                .map(|(_, num_rows)| *num_rows)
                .sum::<u64>(),
            400
        );
        assert!(
            rows_per_segment.iter().all(|(_, num_rows)| *num_rows > 0),
            "each segment should contain indexed rows"
        );

        let ivf_view = logical_index.as_ivf().unwrap();
        let partitions_per_segment = ivf_view.num_partitions_per_segment();
        assert!(
            partitions_per_segment
                .iter()
                .all(|(_, num_partitions)| *num_partitions == 2)
        );

        let row_count_by_segment = rows_per_segment.into_iter().collect::<HashMap<_, _>>();
        let partition_sizes = ivf_view.partition_sizes();
        assert_eq!(partition_sizes.len(), logical_index.num_segments());
        for (segment_id, sizes) in partition_sizes {
            assert_eq!(sizes.len(), 2);
            assert_eq!(
                sizes.iter().sum::<usize>(),
                row_count_by_segment[&segment_id] as usize
            );
        }
    }

    #[tokio::test]
    async fn test_open_logical_vector_index_rejects_wrong_column() {
        const DIMENSION: i32 = 8;

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = write_fragmented_vector_dataset(test_uri, DIMENSION).await;
        let params =
            VectorIndexParams::with_ivf_flat_params(DistanceType::L2, IvfBuildParams::new(2));

        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let err = dataset
            .open_logical_vector_index("id", "vector_idx")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("does not belong to column 'id'"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_segmented_optimize_rebalances_only_one_segment() {
        const DIMENSION: i32 = 8;

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = write_skewed_fragmented_vector_dataset(test_uri, DIMENSION).await;
        create_segmented_vector_index(&mut dataset, "vector_idx", "vector", DIMENSION).await;

        let before_segments = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(before_segments.len(), 2);
        let before_by_fragment = before_segments
            .iter()
            .map(|metadata| {
                let fragments = metadata
                    .fragment_bitmap
                    .as_ref()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>();
                (fragments, metadata.uuid)
            })
            .collect::<HashMap<_, _>>();
        assert_eq!(before_by_fragment.len(), 2);

        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();

        let after_segments = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(after_segments.len(), 2);
        let after_by_fragment = after_segments
            .iter()
            .map(|metadata| {
                let fragments = metadata
                    .fragment_bitmap
                    .as_ref()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>();
                (fragments, metadata.uuid)
            })
            .collect::<HashMap<_, _>>();
        assert_eq!(after_by_fragment.len(), 2);

        assert_ne!(
            before_by_fragment[&vec![0]],
            after_by_fragment[&vec![0]],
            "expected optimize to replace the oversized segment"
        );
        assert_eq!(
            before_by_fragment[&vec![1]],
            after_by_fragment[&vec![1]],
            "expected optimize to leave the smaller segment untouched"
        );

        let logical_index = dataset
            .open_logical_vector_index("vector", "vector_idx")
            .await
            .unwrap();
        let partitions_per_segment = logical_index
            .as_ivf()
            .unwrap()
            .num_partitions_per_segment()
            .into_iter()
            .collect::<HashMap<_, _>>();
        assert_eq!(partitions_per_segment[&after_by_fragment[&vec![0]]], 3);
        assert_eq!(partitions_per_segment[&after_by_fragment[&vec![1]]], 2);
    }

    #[tokio::test]
    async fn test_recreate_index() {
        const DIM: i32 = 8;
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "v",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
                true,
            ),
            Field::new(
                "o",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
                true,
            ),
        ]));
        let data = generate_random_array(2048 * DIM as usize);
        let batches: Vec<RecordBatch> = vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(FixedSizeListArray::try_new_from_values(data.clone(), DIM).unwrap()),
                    Arc::new(FixedSizeListArray::try_new_from_values(data, DIM).unwrap()),
                ],
            )
            .unwrap(),
        ];

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 2);
        dataset
            .create_index(&["v"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        dataset
            .create_index(&["o"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        // Create index again
        dataset
            .create_index(&["v"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        // Can not overwrite an index on different columns.
        assert!(
            dataset
                .create_index(
                    &["v"],
                    IndexType::Vector,
                    Some("o_idx".to_string()),
                    &params,
                    true,
                )
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_bitmap_index_statistics_minimal_io_via_dataset() {
        const NUM_ROWS: usize = 500_000;
        let test_dir = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "status",
            DataType::Int32,
            false,
        )]));
        let values: Vec<i32> = (0..NUM_ROWS as i32).collect();
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(values))]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let mut dataset = Dataset::write(reader, &test_dir, None).await.unwrap();
        let io_tracker = dataset.object_store.as_ref().io_tracker().clone();

        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
        dataset
            .create_index(
                &["status"],
                IndexType::Bitmap,
                Some("status_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        let index_meta = indices
            .iter()
            .find(|idx| idx.name == "status_idx")
            .expect("status_idx should exist");
        let lookup_path = dataset
            .indice_files_dir(index_meta)
            .unwrap()
            .join(index_meta.uuid.to_string())
            .join(BITMAP_LOOKUP_NAME);
        let meta = dataset.object_store.inner.head(&lookup_path).await.unwrap();
        assert!(
            meta.size >= 1_000_000,
            "bitmap index should be large enough to fail without metadata path, size={} bytes",
            meta.size
        );

        // Reset stats collected during index creation
        io_tracker.incremental_stats();

        dataset.index_statistics("status_idx").await.unwrap();

        let stats = io_tracker.incremental_stats();
        assert_io_eq!(
            stats,
            read_bytes,
            4096,
            "index_statistics should only read the index footer; got {} bytes",
            stats.read_bytes
        );
        assert_io_lt!(
            stats,
            read_iops,
            3,
            "index_statistics should only require a head plus one range read; got {} ops",
            stats.read_iops
        );
        assert_io_eq!(
            stats,
            written_bytes,
            0,
            "index_statistics should not perform writes"
        );
    }

    fn sample_vector_field() -> Field {
        let dimensions = 16;
        let column_name = "vec";
        Field::new(
            column_name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        )
    }

    #[tokio::test]
    async fn test_drop_index() {
        let test_dir = TempStrDir::default();
        let schema = Schema::new(vec![
            sample_vector_field(),
            Field::new("ints", DataType::Int32, false),
        ]);
        let mut dataset = lance_datagen::rand(&schema)
            .into_dataset(
                &test_dir,
                FragmentCount::from(1),
                FragmentRowCount::from(256),
            )
            .await
            .unwrap();

        let idx_name = "name".to_string();
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some(idx_name.clone()),
                &VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 10),
                true,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["ints"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        assert_eq!(dataset.load_indices().await.unwrap().len(), 2);

        dataset.drop_index(&idx_name).await.unwrap();

        assert_eq!(dataset.load_indices().await.unwrap().len(), 1);

        // Even though we didn't give the scalar index a name it still has an auto-generated one we can use
        let scalar_idx_name = &dataset.load_indices().await.unwrap()[0].name;
        dataset.drop_index(scalar_idx_name).await.unwrap();

        assert_eq!(dataset.load_indices().await.unwrap().len(), 0);

        // Make sure it returns an error if the index doesn't exist
        assert!(dataset.drop_index(scalar_idx_name).await.is_err());
    }

    #[tokio::test]
    async fn test_count_index_rows() {
        let test_dir = TempStrDir::default();
        let dimensions = 16;
        let column_name = "vec";
        let field = sample_vector_field();
        let schema = Arc::new(Schema::new(vec![field]));

        let float_arr = generate_random_array(512 * dimensions as usize);

        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());
        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
        dataset.validate().await.unwrap();

        // Make sure it returns None if there's no index with the passed identifier
        assert!(dataset.index_statistics("bad_id").await.is_err());
        // Create an index
        let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 10);
        dataset
            .create_index(
                &[column_name],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 512);

        // Now we'll append some rows which shouldn't be indexed and see the
        // count change
        let float_arr = generate_random_array(512 * dimensions as usize);
        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

        let reader = RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema);
        dataset.append(reader, None).await.unwrap();

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
    }

    #[tokio::test]
    async fn test_v036_scalar_details_are_still_known() {
        let test_dir = copy_test_data_to_tmp("0.36.0/btree_in_index_pkg.lance").unwrap();
        let dataset = Dataset::open(&test_dir.path_str()).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let details = indices[0].index_details.clone().unwrap();

        assert_eq!(details.type_url, "/lance.index.pb.BTreeIndexDetails");
        assert_eq!(IndexDetails(details).get_plugin().unwrap().name(), "BTree");
        assert!(index_type_is_known(&indices[0]));
    }

    #[tokio::test]
    async fn test_unknown_index_type_does_not_block_queries_or_optimization() {
        let reader = gen_batch()
            .col("vector", array::rand_vec::<Float32Type>(Dimension::from(8)))
            .col("number", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(64), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
        dataset
            .create_index(
                &["number"],
                IndexType::BTree,
                Some("number_idx".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        let appended = gen_batch()
            .col("vector", array::rand_vec::<Float32Type>(Dimension::from(8)))
            .col("number", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(32), BatchCount::from(1));
        dataset.append(appended, None).await.unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("number_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 32);

        let field_id = dataset.schema().field("vector").unwrap().id;
        let fragment_ids = dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<Vec<_>>();
        let mut foreign_segment = write_vector_segment_metadata(
            &dataset,
            "foreign_idx",
            field_id,
            Uuid::new_v4(),
            fragment_ids,
            b"opaque external index",
        )
        .await;
        foreign_segment.index_details = Some(Arc::new(prost_types::Any {
            type_url: "type.googleapis.com/example.MyVectorIndexDetails".to_string(),
            value: Vec::new(),
        }));
        foreign_segment.index_version = 1;
        dataset
            .commit_existing_index_segments("foreign_idx", "vector", vec![foreign_segment])
            .await
            .unwrap();

        assert!(
            dataset
                .load_indices_by_name("foreign_idx")
                .await
                .unwrap()
                .is_empty(),
            "an index with no reader must not enter the usable-index view"
        );
        assert_eq!(
            load_all_indices_by_name(&dataset, "foreign_idx")
                .await
                .unwrap()
                .len(),
            1,
            "hiding an unusable index must not erase its manifest metadata"
        );

        let query = Float32Array::from(vec![0.5_f32; 8]);
        let mut scanner = dataset.scan();
        scanner.nearest("vector", &query, 5).unwrap();
        assert_eq!(scanner.try_into_batch().await.unwrap().num_rows(), 5);

        dataset.optimize_indices(&Default::default()).await.unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("number_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(
            load_all_indices_by_name(&dataset, "foreign_idx")
                .await
                .unwrap()
                .len(),
            1,
            "optimizing supported indices must preserve the opaque segment"
        );
    }

    #[tokio::test]
    async fn test_optimize_delta_indices() {
        let dimensions = 16;
        let column_name = "vec";
        let vec_field = Field::new(
            column_name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        );
        let other_column_name = "other_vec";
        let other_vec_field = Field::new(
            other_column_name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        );
        let schema = Arc::new(Schema::new(vec![vec_field, other_vec_field]));

        let float_arr = generate_random_array(512 * dimensions as usize);

        let vectors = Arc::new(
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap(),
        );

        let record_batch =
            RecordBatch::try_new(schema.clone(), vec![vectors.clone(), vectors.clone()]).unwrap();

        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );

        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
        let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 10);
        dataset
            .create_index(
                &[column_name],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &[other_column_name],
                IndexType::Vector,
                Some("other_vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        async fn get_stats(dataset: &Dataset, name: &str) -> serde_json::Value {
            serde_json::from_str(&dataset.index_statistics(name).await.unwrap()).unwrap()
        }
        async fn get_meta(dataset: &Dataset, name: &str) -> Vec<IndexMetadata> {
            dataset
                .load_indices()
                .await
                .unwrap()
                .iter()
                .filter(|m| m.name == name)
                .cloned()
                .collect()
        }
        fn get_bitmap(meta: &IndexMetadata) -> Vec<u32> {
            meta.fragment_bitmap.as_ref().unwrap().iter().collect()
        }
        fn assert_segment_aliases(stats: &serde_json::Value) {
            assert_eq!(stats["num_segments"], stats["num_indices"]);
            assert_eq!(stats["segments"], stats["indices"]);
        }

        let stats = get_stats(&dataset, "vec_idx").await;
        assert_segment_aliases(&stats);
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);
        let meta = get_meta(&dataset, "vec_idx").await;
        assert_eq!(meta.len(), 1);
        assert_eq!(get_bitmap(&meta[0]), vec![0]);

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());
        dataset.append(reader, None).await.unwrap();
        let stats = get_stats(&dataset, "vec_idx").await;
        assert_segment_aliases(&stats);
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);
        let meta = get_meta(&dataset, "vec_idx").await;
        assert_eq!(meta.len(), 1);
        assert_eq!(get_bitmap(&meta[0]), vec![0]);

        dataset
            .optimize_indices(&OptimizeOptions::append().index_names(vec![])) // Does nothing because no index name is passed
            .await
            .unwrap();
        let stats = get_stats(&dataset, "vec_idx").await;
        assert_segment_aliases(&stats);
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);
        let meta = get_meta(&dataset, "vec_idx").await;
        assert_eq!(meta.len(), 1);
        assert_eq!(get_bitmap(&meta[0]), vec![0]);

        // optimize the other index
        dataset
            .optimize_indices(
                &OptimizeOptions::append().index_names(vec!["other_vec_idx".to_owned()]),
            )
            .await
            .unwrap();
        let stats = get_stats(&dataset, "vec_idx").await;
        assert_segment_aliases(&stats);
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);
        let meta = get_meta(&dataset, "vec_idx").await;
        assert_eq!(meta.len(), 1);
        assert_eq!(get_bitmap(&meta[0]), vec![0]);

        let stats = get_stats(&dataset, "other_vec_idx").await;
        assert_segment_aliases(&stats);
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 1024);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        assert_eq!(stats["num_indices"], 2);
        let meta = get_meta(&dataset, "other_vec_idx").await;
        assert_eq!(meta.len(), 2);
        assert_eq!(get_bitmap(&meta[0]), vec![0]);
        assert_eq!(get_bitmap(&meta[1]), vec![1]);

        dataset
            .optimize_indices(&OptimizeOptions::retrain())
            .await
            .unwrap();

        let stats = get_stats(&dataset, "vec_idx").await;
        assert_segment_aliases(&stats);
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 1024);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        assert_eq!(stats["num_indices"], 1);
        let meta = get_meta(&dataset, "vec_idx").await;
        assert_eq!(meta.len(), 1);
        assert_eq!(get_bitmap(&meta[0]), vec![0, 1]);

        dataset
            .optimize_indices(&OptimizeOptions::retrain())
            .await
            .unwrap();
        let stats = get_stats(&dataset, "other_vec_idx").await;
        assert_segment_aliases(&stats);
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 1024);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        assert_eq!(stats["num_indices"], 1);
        let meta = get_meta(&dataset, "other_vec_idx").await;
        assert_eq!(meta.len(), 1);
        assert_eq!(get_bitmap(&meta[0]), vec![0, 1]);
    }

    #[tokio::test]
    async fn test_optimize_ivf_hnsw_sq_delta_indices() {
        let test_dir = TempStrDir::default();
        let dimensions = 16;
        let column_name = "vec";
        let field = Field::new(
            column_name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        );
        let schema = Arc::new(Schema::new(vec![field]));

        let float_arr = generate_random_array(512 * dimensions as usize);

        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );

        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let ivf_params = IvfBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let sq_params = SQBuildParams::default();
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            MetricType::L2,
            ivf_params,
            hnsw_params,
            sq_params,
        );
        dataset
            .create_index(
                &[column_name],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());
        dataset.append(reader, None).await.unwrap();
        let mut dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);

        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 1024);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        assert_eq!(stats["num_indices"], 2);

        dataset
            .optimize_indices(&OptimizeOptions::retrain())
            .await
            .unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 1024);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        assert_eq!(stats["num_indices"], 1);
    }

    #[rstest]
    #[case::v1("v3.0.1/fts_v1", INVERTED_INDEX_VERSION_V1)]
    #[case::v2("v4.0.1/fts_v2", INVERTED_INDEX_VERSION_V2)]
    #[tokio::test]
    async fn test_read_fts_format_fixture(
        #[case] fixture_path: &str,
        #[case] expected_version: u32,
    ) {
        async fn search_ids(dataset: &Dataset, query: FullTextSearchQuery) -> Vec<i64> {
            let result = dataset
                .scan()
                .project(&["id"])
                .unwrap()
                .full_text_search(query)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            let mut ids = result["id"].as_primitive::<Int64Type>().values().to_vec();
            ids.sort_unstable();
            ids
        }

        let test_dir = copy_test_data_to_tmp(fixture_path).unwrap();
        let dataset = Dataset::open(&test_dir.path_str()).await.unwrap();

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].index_version, expected_version as i32);

        let match_ids = search_ids(
            &dataset,
            FullTextSearchQuery::new("compatibility".to_string()),
        )
        .await;
        assert_eq!(match_ids, (0..300).collect::<Vec<_>>());

        let phrase =
            PhraseQuery::new("lance database".to_string()).with_column(Some("text".to_string()));
        let phrase_ids = search_ids(
            &dataset,
            FullTextSearchQuery::new_query(FtsQuery::Phrase(phrase)),
        )
        .await;
        assert_eq!(phrase_ids, (0..300).step_by(3).collect::<Vec<_>>());
    }

    #[rstest]
    #[tokio::test]
    async fn test_optimize_fts(#[values(false, true)] with_position: bool) {
        let words = ["apple", "banana", "cherry", "date"];

        let dir = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let data = StringArray::from_iter_values(words.iter().map(|s| s.to_string()));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(data)]).unwrap();
        let batch_iterator = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let mut dataset = Dataset::write(batch_iterator, &dir, None).await.unwrap();

        let params = InvertedIndexParams::default()
            .lower_case(false)
            .with_position(with_position);
        dataset
            .create_index(&["text"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();

        async fn assert_indexed_rows(dataset: &Dataset, expected_indexed_rows: usize) {
            let stats = dataset.index_statistics("text_idx").await.unwrap();
            let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
            let indexed_rows = stats["num_indexed_rows"].as_u64().unwrap() as usize;
            let unindexed_rows = stats["num_unindexed_rows"].as_u64().unwrap() as usize;
            let num_rows = dataset.count_all_rows().await.unwrap();
            assert_eq!(indexed_rows, expected_indexed_rows);
            assert_eq!(unindexed_rows, num_rows - expected_indexed_rows);
        }

        let num_rows = dataset.count_all_rows().await.unwrap();
        assert_indexed_rows(&dataset, num_rows).await;

        let new_words = ["elephant", "fig", "grape", "honeydew"];
        let new_data = StringArray::from_iter_values(new_words.iter().map(|s| s.to_string()));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(new_data)]).unwrap();
        let batch_iter = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        dataset.append(batch_iter, None).await.unwrap();
        assert_indexed_rows(&dataset, num_rows).await;

        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let num_rows = dataset.count_all_rows().await.unwrap();
        assert_indexed_rows(&dataset, num_rows).await;

        for &word in words.iter().chain(new_words.iter()) {
            let query_result = dataset
                .scan()
                .project(&["text"])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new(word.to_string()))
                .unwrap()
                .limit(Some(10), None)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            let texts = query_result["text"]
                .as_string::<i32>()
                .iter()
                .map(|v| match v {
                    None => "".to_string(),
                    Some(v) => v.to_string(),
                })
                .collect::<Vec<String>>();

            assert_eq!(texts.len(), 1);
            assert_eq!(texts[0], word);
        }

        let uppercase_words = ["Apple", "Banana", "Cherry", "Date"];
        for &word in uppercase_words.iter() {
            let query_result = dataset
                .scan()
                .project(&["text"])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new(word.to_string()))
                .unwrap()
                .limit(Some(10), None)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            let texts = query_result["text"]
                .as_string::<i32>()
                .iter()
                .map(|v| match v {
                    None => "".to_string(),
                    Some(v) => v.to_string(),
                })
                .collect::<Vec<String>>();

            assert_eq!(texts.len(), 0);
        }
        let new_data = StringArray::from_iter_values(uppercase_words.iter().map(|s| s.to_string()));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(new_data)]).unwrap();
        let batch_iter = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        dataset.append(batch_iter, None).await.unwrap();
        assert_indexed_rows(&dataset, num_rows).await;

        // we should be able to query the new words
        for &word in uppercase_words.iter() {
            let query_result = dataset
                .scan()
                .project(&["text"])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new(word.to_string()))
                .unwrap()
                .limit(Some(10), None)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            let texts = query_result["text"]
                .as_string::<i32>()
                .iter()
                .map(|v| match v {
                    None => "".to_string(),
                    Some(v) => v.to_string(),
                })
                .collect::<Vec<String>>();

            assert_eq!(texts.len(), 1, "query: {}, texts: {:?}", word, texts);
            assert_eq!(texts[0], word, "query: {}, texts: {:?}", word, texts);
        }

        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let num_rows = dataset.count_all_rows().await.unwrap();
        assert_indexed_rows(&dataset, num_rows).await;

        // we should be able to query the new words after optimization
        for &word in uppercase_words.iter() {
            let query_result = dataset
                .scan()
                .project(&["text"])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new(word.to_string()))
                .unwrap()
                .limit(Some(10), None)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            let texts = query_result["text"]
                .as_string::<i32>()
                .iter()
                .map(|v| match v {
                    None => "".to_string(),
                    Some(v) => v.to_string(),
                })
                .collect::<Vec<String>>();

            assert_eq!(texts.len(), 1, "query: {}, texts: {:?}", word, texts);
            assert_eq!(texts[0], word, "query: {}, texts: {:?}", word, texts);

            // we should be able to query the new words after compaction
            compact_files(&mut dataset, CompactionOptions::default(), None)
                .await
                .unwrap();
            for &word in uppercase_words.iter() {
                let query_result = dataset
                    .scan()
                    .project(&["text"])
                    .unwrap()
                    .full_text_search(FullTextSearchQuery::new(word.to_string()))
                    .unwrap()
                    .try_into_batch()
                    .await
                    .unwrap();
                let texts = query_result["text"]
                    .as_string::<i32>()
                    .iter()
                    .map(|v| match v {
                        None => "".to_string(),
                        Some(v) => v.to_string(),
                    })
                    .collect::<Vec<String>>();
                assert_eq!(texts.len(), 1, "query: {}, texts: {:?}", word, texts);
                assert_eq!(texts[0], word, "query: {}, texts: {:?}", word, texts);
            }
            assert_indexed_rows(&dataset, num_rows).await;
        }
    }

    #[tokio::test]
    async fn test_optimize_fts_respects_num_indices_to_merge() {
        use lance_index::scalar::inverted::query::PhraseQuery;

        async fn append_texts(dataset: &mut Dataset, schema: Arc<Schema>, rows: &[(i32, &str)]) {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(rows.iter().map(|(id, _)| *id))),
                    Arc::new(StringArray::from_iter_values(
                        rows.iter().map(|(_, text)| *text),
                    )),
                ],
            )
            .unwrap();
            let batch_iter = RecordBatchIterator::new(vec![Ok(batch)], schema);
            dataset.append(batch_iter, None).await.unwrap();
        }

        async fn num_fts_segments(dataset: &Dataset) -> usize {
            let stats = dataset.index_statistics("text_idx").await.unwrap();
            let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
            stats["num_indices"].as_u64().unwrap() as usize
        }

        async fn assert_fts_ids(
            dataset: &Dataset,
            query: FullTextSearchQuery,
            expected_ids: &[i32],
        ) {
            let result = dataset
                .scan()
                .project(&["id"])
                .unwrap()
                .full_text_search(query)
                .unwrap()
                .limit(Some(20), None)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            let mut ids = result["id"].as_primitive::<Int32Type>().values().to_vec();
            ids.sort_unstable();
            assert_eq!(ids, expected_ids);
        }

        let dir = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values([0, 1])),
                Arc::new(StringArray::from_iter_values([
                    "alpha base phrase",
                    "beta base phrase",
                ])),
            ],
        )
        .unwrap();
        let batch_iterator = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let mut dataset = Dataset::write(batch_iterator, &dir, None).await.unwrap();
        let params = InvertedIndexParams::default().with_position(true);
        dataset
            .create_index(&["text"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();
        assert_eq!(num_fts_segments(&dataset).await, 1);

        append_texts(&mut dataset, schema.clone(), &[(2, "gamma delta phrase")]).await;
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        assert_eq!(num_fts_segments(&dataset).await, 2);

        append_texts(&mut dataset, schema.clone(), &[(3, "epsilon zeta phrase")]).await;
        dataset
            .optimize_indices(&OptimizeOptions::merge(1))
            .await
            .unwrap();
        assert_eq!(num_fts_segments(&dataset).await, 2);
        assert_fts_ids(
            &dataset,
            FullTextSearchQuery::new("epsilon".to_owned()),
            &[3],
        )
        .await;

        append_texts(&mut dataset, schema.clone(), &[(4, "eta theta phrase")]).await;
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        assert_eq!(num_fts_segments(&dataset).await, 3);

        dataset
            .optimize_indices(&OptimizeOptions::merge(2))
            .await
            .unwrap();
        assert_eq!(num_fts_segments(&dataset).await, 2);
        assert_fts_ids(
            &dataset,
            FullTextSearchQuery::new_query(PhraseQuery::new("eta theta".to_owned()).into()),
            &[4],
        )
        .await;

        append_texts(&mut dataset, schema, &[(5, "iota kappa phrase")]).await;
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        assert_eq!(num_fts_segments(&dataset).await, 3);

        dataset
            .optimize_indices(&OptimizeOptions::merge(10))
            .await
            .unwrap();
        assert_eq!(num_fts_segments(&dataset).await, 1);
        assert_fts_ids(&dataset, FullTextSearchQuery::new("alpha".to_owned()), &[0]).await;
        assert_fts_ids(&dataset, FullTextSearchQuery::new("iota".to_owned()), &[5]).await;
    }

    #[tokio::test]
    async fn test_create_index_too_small_for_pq() {
        let test_dir = TempStrDir::default();
        let dimensions = 1536;

        let field = Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        );

        let schema = Arc::new(Schema::new(vec![field]));
        let float_arr = generate_random_array(100 * dimensions as usize);

        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();
        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );

        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_pq(1, 8, 96, DistanceType::L2, 1);
        let result = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await;

        assert!(matches!(result, Err(Error::Unprocessable { .. })));
        if let Error::Unprocessable { message, .. } = result.unwrap_err() {
            assert_eq!(
                message,
                "Not enough rows to train PQ. Requires 256 rows but only 100 available",
            )
        }
    }

    #[tokio::test]
    async fn test_create_bitmap_index() {
        let test_dir = TempStrDir::default();
        let field = Field::new("tag", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![field]));
        let array = StringArray::from_iter_values((0..128).map(|i| ["a", "b", "c"][i % 3]));
        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );

        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
        dataset
            .create_index(
                &["tag"],
                IndexType::Bitmap,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let index = dataset
            .open_generic_index("tag", &indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(index.index_type(), IndexType::Bitmap);
    }

    // #[tokio::test]
    #[lance_test_macros::test(tokio::test)]
    async fn test_load_indices() {
        let session = Arc::new(Session::default());
        let write_params = WriteParams {
            session: Some(session.clone()),
            ..Default::default()
        };

        let test_dir = TempStrDir::default();
        let field = Field::new("tag", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![field]));
        let array = StringArray::from_iter_values((0..128).map(|i| ["a", "b", "c"][i % 3]));
        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );

        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();
        dataset
            .create_index(
                &["tag"],
                IndexType::Bitmap,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        dataset.object_store.as_ref().io_stats_incremental(); // Reset

        let indices = dataset.load_indices().await.unwrap();
        let stats = dataset.object_store.as_ref().io_stats_incremental();
        // We should already have this cached since we just wrote it.
        assert_io_eq!(stats, read_iops, 0);
        assert_io_eq!(stats, read_bytes, 0);
        assert_eq!(indices.len(), 1);

        // Clear the cache
        session.index_cache.clear().await;
        session.file_metadata_cache().clear().await;

        let dataset2 = DatasetBuilder::from_uri(test_uri)
            .with_session(session.clone())
            .load()
            .await
            .unwrap();
        let stats = dataset2.object_store.as_ref().io_stats_incremental(); // Reset
        assert_io_lt!(stats, read_bytes, 64 * 1024);

        // Because the manifest is so small, we should have opportunistically
        // cached the indices in memory already.
        let indices2 = dataset2.load_indices().await.unwrap();
        let stats = dataset2.object_store.as_ref().io_stats_incremental();
        assert_io_eq!(stats, read_iops, 0);
        assert_io_eq!(stats, read_bytes, 0);
        assert_eq!(indices2.len(), 1);
    }

    #[tokio::test]
    async fn test_index_metadata_cache_does_not_survive_drop_recreate_same_uri() {
        fn batch_with_schema_metadata(metadata_value: String) -> RecordBatch {
            let field = Field::new("tag", DataType::Utf8, false);
            let schema = Arc::new(Schema::new_with_metadata(
                vec![field],
                HashMap::from([("large_metadata".to_string(), metadata_value)]),
            ));
            let array = StringArray::from_iter_values((0..128).map(|i| ["a", "b", "c"][i % 3]));
            RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap()
        }

        async fn write_indexed_dataset(uri: &str, session: Arc<Session>, metadata_value: String) {
            let batch = batch_with_schema_metadata(metadata_value);
            let schema = batch.schema();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
            let write_params = WriteParams {
                session: Some(session),
                ..Default::default()
            };
            let mut dataset = Dataset::write(reader, uri, Some(write_params))
                .await
                .unwrap();
            dataset
                .create_index(
                    &["tag"],
                    IndexType::Bitmap,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
        }

        let qn_session = Arc::new(Session::default());
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        write_indexed_dataset(test_uri, qn_session.clone(), "old".to_string()).await;
        let first_dataset = DatasetBuilder::from_uri(test_uri)
            .with_session(qn_session.clone())
            .load()
            .await
            .unwrap();
        let first_indices = first_dataset.load_indices().await.unwrap();
        let first_uuid = first_indices[0].uuid;
        assert_eq!(first_dataset.version().version, 2);
        drop(first_dataset);

        std::fs::remove_dir_all(test_uri).unwrap();

        // Use a different writer session so the QN session keeps the previous
        // incarnation's index metadata cache entry. The large schema metadata
        // keeps the manifest index section outside the final read block during
        // open, so the fresh manifest load cannot opportunistically overwrite
        // the stale index metadata entry before load_indices().
        write_indexed_dataset(
            test_uri,
            Arc::new(Session::default()),
            "x".repeat(128 * 1024),
        )
        .await;
        let second_dataset = DatasetBuilder::from_uri(test_uri)
            .with_session(qn_session)
            .load()
            .await
            .unwrap();
        assert_eq!(second_dataset.version().version, 2);

        let raw_second_indices = read_manifest_indexes(
            second_dataset.object_store.as_ref(),
            &second_dataset.manifest_location,
            second_dataset.manifest(),
        )
        .await
        .unwrap();
        let raw_second_uuid = raw_second_indices[0].uuid;
        assert_ne!(
            raw_second_uuid, first_uuid,
            "the recreated dataset should commit a new physical index UUID"
        );

        let cached_second_indices = second_dataset.load_indices().await.unwrap();
        assert_eq!(
            cached_second_indices[0].uuid, raw_second_uuid,
            "load_indices should return index metadata from the recreated dataset, not the previous same-URI incarnation"
        );
    }

    #[tokio::test]
    async fn test_load_indices_singleflights_concurrent_cache_misses() {
        let session = Arc::new(Session::default());
        let write_params = WriteParams {
            session: Some(session.clone()),
            ..Default::default()
        };
        let test_dir = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![Field::new("tag", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec!["a", "b", "c"]))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &test_dir,
            Some(write_params),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["tag"],
                IndexType::Bitmap,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        session.index_cache.clear().await;
        let before = session.index_cache_stats().await;
        let results = futures::future::join_all((0..32).map(|_| dataset.load_indices())).await;
        assert!(results.iter().all(|result| result.is_ok()));
        let after = session.index_cache_stats().await;

        assert_eq!(after.misses - before.misses, 1);
        assert_eq!(after.hits - before.hits, 31);
    }

    #[tokio::test]
    async fn test_open_frag_reuse_index_with_zero_capacity_cache() {
        let test_dir = TempStrDir::default();
        let data = gen_batch()
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(400), BatchCount::from(1));
        let mut dataset = Dataset::write(
            data,
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 100,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".to_owned()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 200,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(
            dataset
                .load_index_by_name(FRAG_REUSE_INDEX_NAME)
                .await
                .unwrap()
                .is_some()
        );

        let session = Arc::new(Session::with_index_cache_backend(
            Arc::new(lance_core::cache::MokaCacheBackend::no_cache()),
            128 * 1024 * 1024,
            Default::default(),
        ));
        let dataset = DatasetBuilder::from_uri(&test_dir)
            .with_session(session)
            .load()
            .await
            .unwrap();

        let frag_reuse_index = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            dataset.open_frag_reuse_index(&NoOpMetricsCollector),
        )
        .await
        .expect("opening the fragment reuse index deadlocked")
        .unwrap();
        assert!(frag_reuse_index.is_some());
    }

    #[tokio::test]
    async fn test_remap_empty_chain() {
        let data = gen_batch()
            .col("int", array::step::<Int32Type>())
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(16)),
            )
            .into_reader_rows(RowCount::from(256), BatchCount::from(1));
        let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();

        let params = VectorIndexParams::ivf_pq(1, 8, 1, DistanceType::L2, 1);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let index_uuid = dataset.load_indices().await.unwrap()[0].uuid;
        let row_count = dataset.count_all_rows().await.unwrap();
        let first_half = (0..row_count / 2)
            .map(|i| (i as u64, None))
            .collect::<HashMap<_, _>>();
        let second_half = (row_count / 2..row_count)
            .map(|i| (i as u64, None))
            .collect::<HashMap<_, _>>();
        let remap_to_empty = RowAddrRemap::chained([
            RowAddrRemap::direct(first_half),
            RowAddrRemap::direct(second_half),
        ]);
        let new_uuid = remap_index(&dataset, &index_uuid, &remap_to_empty)
            .await
            .unwrap();
        assert_eq!(new_uuid, RemapResult::Keep(index_uuid));
    }

    /// The `fields.len() > 1` rejection in `remap_index`, which had no dedicated
    /// coverage. A covered index never reaches it: the withdrawal above returns
    /// `RemapResult::Drop` first, which is what
    /// `test_compaction_withdraws_a_covered_index_without_failing` in
    /// `dataset/tests/dataset_index.rs` exercises, and
    /// `test_remap_column_index_refuses_a_covered_index` in
    /// `dataset/optimize/remapping.rs` never reaches `remap_index` at all since
    /// `remap_column_index` refuses first. So only a genuinely composite index --
    /// two keyed fields, no covering declaration -- arrives here, and it must
    /// still be rejected.
    #[tokio::test]
    async fn test_remap_index_rejects_composite_index() {
        let data = gen_batch()
            .col("a", array::step::<Int32Type>())
            .col("b", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(20), BatchCount::from(1));
        let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();
        dataset
            .create_index(
                &["a"],
                IndexType::BTree,
                None,
                &lance_index::scalar::ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();

        let a_id = dataset.schema().field("a").unwrap().id;
        let b_id = dataset.schema().field("b").unwrap().id;
        let current = dataset.load_indices().await.unwrap();
        let mut composite = current[0].clone();
        // No covering declaration at all -- genuinely composite, not covered.
        composite.fields = vec![a_id, b_id];
        composite.covering_fields = vec![];

        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![composite],
                removed_indices: current.to_vec(),
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let index_uuid = dataset.load_indices().await.unwrap()[0].uuid;
        let error = remap_index(&dataset, &index_uuid, &RowAddrRemap::empty())
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("only one keyed field is supported"),
            "{error}"
        );
    }

    /// Decoding a manifest validates the declaration, so `matched` cannot be
    /// malformed by that route -- but it is not the only route: metadata
    /// constructed in-process (segment conversion, a distributed build's output)
    /// reaches this guard without ever passing through
    /// `TryFrom<pb::IndexMetadata>`. This guard is therefore load-bearing on its
    /// own, and must produce a named error rather than either a panic (plain
    /// subtraction on `usize` underflows) or a silent
    /// saturation that lets corrupt metadata through as an ordinary covered index
    /// and gets it quietly withdrawn. The test reaches it by seeding the index
    /// cache, the same way the migration-path test in `dataset/index.rs` does.
    #[tokio::test]
    async fn test_remap_index_rejects_malformed_covering_fields() {
        let data = gen_batch()
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(20), BatchCount::from(1));
        let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                None,
                &lance_index::scalar::ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();

        let index_uuid = dataset.load_indices().await.unwrap()[0].uuid;

        // Malform the cached metadata directly: more carried fields than fields
        // at all. No commit and no manifest decode can produce this -- both run
        // `validate_covering_fields` -- which is exactly why the guard has to
        // stand on its own for metadata that reached it by neither route.
        let mut indices = dataset.load_indices().await.unwrap().as_ref().clone();
        for idx in &mut indices {
            if idx.uuid == index_uuid {
                idx.covering_fields = idx
                    .fields
                    .iter()
                    .copied()
                    .chain(std::iter::once(999))
                    .collect();
            }
        }
        assert!(
            indices
                .iter()
                .any(|idx| idx.uuid == index_uuid && idx.covering_fields.len() > idx.fields.len()),
            "test setup should have produced a malformed entry"
        );
        let metadata_key = crate::session::index_caches::IndexMetadataKey {
            version: dataset.version().version,
            store_identity: &dataset.object_store.store_prefix,
            e_tag: dataset.manifest_location.e_tag.as_deref(),
        };
        dataset
            .index_cache
            .insert_with_key(&metadata_key, Arc::new(indices))
            .await;

        // The validation runs first in `remap_index`, ahead of the withdrawal and
        // of every row-map-dependent branch, so an empty remap reaches it.
        let error = remap_index(&dataset, &index_uuid, &RowAddrRemap::empty())
            .await
            .expect_err("malformed covering metadata must not be silently accepted");
        assert!(
            error.to_string().contains("are not among its fields"),
            "expected the validator's message, got: {error}"
        );
    }

    /// A covered scalar index still answers for its keyed column: only the
    /// keyed prefix of `fields` must be a single field for `scalar_index_info`
    /// to make it eligible for filter pushdown, not the full vector including
    /// carried columns. Before this fix, a covered index was silently excluded
    /// -- no error, no failing query, just a plan that quietly stopped using it.
    #[tokio::test]
    async fn test_scalar_index_info_includes_covered_index() {
        let data = gen_batch()
            .col("a", array::step::<Int32Type>())
            .col("b", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(20), BatchCount::from(1));
        let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();
        dataset
            .create_index(
                &["a"],
                IndexType::BTree,
                None,
                &lance_index::scalar::ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();

        crate::utils::test::covering::declare_covering(&mut dataset, "a", "b").await;

        let index_info = dataset.scalar_index_info().await.unwrap();
        assert!(
            index_info.get_index("a").is_some(),
            "a covered index must still be eligible for filter pushdown on its keyed column"
        );
    }

    #[tokio::test]
    async fn test_optimize_ivf_pq_up_to_date() {
        // https://github.com/lance-format/lance/issues/4016
        let nrows = 256;
        let dimensions = 16;
        let column_name = "vector";
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                column_name,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimensions,
                ),
                false,
            ),
        ]));

        let float_arr = generate_random_array(nrows * dimensions as usize);
        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();
        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow_array::Int32Array::from_iter_values(0..nrows as i32)),
                Arc::new(vectors),
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        let params = VectorIndexParams::ivf_pq(1, 8, 2, MetricType::L2, 2);
        dataset
            .create_index(&[column_name], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        let query_vector = generate_random_array(dimensions as usize);

        let nearest = dataset
            .scan()
            .nearest(column_name, &query_vector, 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let ids = nearest["id"].as_primitive::<Int32Type>();
        let mut seen = HashSet::new();
        for id in ids.values() {
            assert!(seen.insert(*id), "Duplicate id found: {}", id);
        }

        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();

        dataset.validate().await.unwrap();

        let nearest_after = dataset
            .scan()
            .nearest(column_name, &query_vector, 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let ids = nearest_after["id"].as_primitive::<Int32Type>();
        let mut seen = HashSet::new();
        for id in ids.values() {
            assert!(seen.insert(*id), "Duplicate id found: {}", id);
        }
    }

    #[tokio::test]
    async fn test_index_created_at_timestamp() {
        // Test that created_at is set when creating an index
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("values", DataType::Utf8, false),
        ]));

        let values = StringArray::from_iter_values(["hello", "world", "foo", "bar"]);
        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..4)),
                Arc::new(values),
            ],
        )
        .unwrap();

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        // Record time before creating index
        let before_index = chrono::Utc::now();

        // Create a scalar index
        dataset
            .create_index(
                &["values"],
                IndexType::Scalar,
                Some("test_idx".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Record time after creating index
        let after_index = chrono::Utc::now();

        // Get index metadata
        let indices = dataset.load_indices().await.unwrap();
        let test_index = indices.iter().find(|idx| idx.name == "test_idx").unwrap();

        // Verify created_at is set and within reasonable bounds
        assert!(test_index.created_at.is_some());
        let created_at = test_index.created_at.unwrap();
        assert!(created_at >= before_index);
        assert!(created_at <= after_index);
    }

    #[tokio::test]
    async fn test_index_statistics_updated_at() {
        // Test that updated_at appears in index statistics
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("values", DataType::Utf8, false),
        ]));

        let values = StringArray::from_iter_values(["hello", "world", "foo", "bar"]);
        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..4)),
                Arc::new(values),
            ],
        )
        .unwrap();

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        // Create a scalar index
        dataset
            .create_index(
                &["values"],
                IndexType::Scalar,
                Some("test_idx".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Get index statistics
        let stats_str = dataset.index_statistics("test_idx").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats_str).unwrap();

        // Verify updated_at_timestamp_ms field exists in statistics
        assert!(stats["updated_at_timestamp_ms"].is_number());
        let updated_at = stats["updated_at_timestamp_ms"].as_u64().unwrap();

        // Get the index metadata to compare with created_at
        let indices = dataset.load_indices().await.unwrap();
        let test_index = indices.iter().find(|idx| idx.name == "test_idx").unwrap();
        let created_at = test_index.created_at.unwrap().timestamp_millis() as u64;

        // For a single index, updated_at should equal created_at
        assert_eq!(updated_at, created_at);
    }

    #[tokio::test]
    async fn test_index_statistics_updated_at_multiple_deltas() {
        // Test that updated_at reflects max(created_at) across multiple delta indices
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                false,
            ),
        ]));

        // Create initial dataset (need more rows for PQ training)
        let num_rows = 300;
        let float_arr = generate_random_array(4 * num_rows);
        let vectors = FixedSizeListArray::try_new_from_values(float_arr, 4).unwrap();
        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..num_rows as i32)),
                Arc::new(vectors),
            ],
        )
        .unwrap();

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        // Create vector index
        let params = VectorIndexParams::ivf_pq(1, 8, 2, MetricType::L2, 2);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("test_vec_idx".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Get initial statistics
        let stats_str_1 = dataset.index_statistics("test_vec_idx").await.unwrap();
        let stats_1: serde_json::Value = serde_json::from_str(&stats_str_1).unwrap();
        let initial_updated_at = stats_1["updated_at_timestamp_ms"].as_u64().unwrap();

        // Add more data to create additional delta indices
        std::thread::sleep(std::time::Duration::from_millis(10)); // Ensure different timestamp

        let num_rows_2 = 50;
        let float_arr_2 = generate_random_array(4 * num_rows_2);
        let vectors_2 = FixedSizeListArray::try_new_from_values(float_arr_2, 4).unwrap();
        let record_batch_2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    num_rows as i32..(num_rows + num_rows_2) as i32,
                )),
                Arc::new(vectors_2),
            ],
        )
        .unwrap();

        let reader_2 =
            RecordBatchIterator::new(vec![record_batch_2].into_iter().map(Ok), schema.clone());

        dataset.append(reader_2, None).await.unwrap();

        // Update the index
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("test_vec_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Get updated statistics
        let stats_str_2 = dataset.index_statistics("test_vec_idx").await.unwrap();
        let stats_2: serde_json::Value = serde_json::from_str(&stats_str_2).unwrap();
        let final_updated_at = stats_2["updated_at_timestamp_ms"].as_u64().unwrap();

        // The updated_at should be newer than the initial one
        assert!(final_updated_at >= initial_updated_at);
    }

    #[tokio::test]
    async fn test_index_statistics_updated_at_none_when_no_created_at() {
        // Test backward compatibility: when indices were created before the created_at field,
        // updated_at_timestamp_ms should be null

        // Use test data created with Lance 0.29.0 (before created_at field was added)
        let test_dir =
            copy_test_data_to_tmp("v0.30.0_pre_created_at/index_without_created_at").unwrap();
        let test_uri = test_dir.path_str();
        let test_uri = &test_uri;

        let dataset = Dataset::open(test_uri).await.unwrap();

        // Get the index metadata to verify created_at is None
        let indices = dataset.load_indices().await.unwrap();
        assert!(!indices.is_empty(), "Test dataset should have indices");

        // Verify that the index created with old version has no created_at
        for index in indices.iter() {
            assert!(
                index.created_at.is_none(),
                "Index from old version should have created_at = None"
            );
        }

        // Get index statistics - should work even with old indices
        let index_name = &indices[0].name;
        let stats_str = dataset.index_statistics(index_name).await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats_str).unwrap();

        // Verify updated_at_timestamp_ms field is null when no indices have created_at
        assert!(
            stats["updated_at_timestamp_ms"].is_null(),
            "updated_at_timestamp_ms should be null when no indices have created_at timestamps"
        );
    }

    #[tokio::test]
    async fn test_legacy_vector_index_details_inferred_on_load_and_migration() {
        use lance_linalg::distance::DistanceType;

        // Create a fresh dataset with IVF_HNSW_SQ so inference produces non-default
        // details (HNSW config + SQ compression) that survive proto serialization.
        let test_dir = lance_core::utils::tempfile::TempDir::default();
        let test_uri = test_dir.path_str();
        let data = gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("vec", array::rand_vec::<Float32Type>(16.into()))
            .into_reader_rows(RowCount::from(1024), BatchCount::from(1));
        let mut dataset = Dataset::write(data, &test_uri, None).await.unwrap();

        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            DistanceType::Cosine,
            IvfBuildParams {
                num_partitions: Some(2),
                ..Default::default()
            },
            HnswBuildParams::default(),
            SQBuildParams::default(),
        );
        dataset
            .create_index(&["vec"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        // Verify the index has populated details.
        let descriptions = dataset.describe_indices(None).await.unwrap();
        assert_eq!(descriptions.len(), 1);
        assert_eq!(descriptions[0].index_type(), "IVF_HNSW_SQ");

        // Simulate a legacy dataset by clearing details from the manifest.
        // Write a new manifest with empty VectorIndexDetails value bytes.
        let mut indices = dataset.load_indices().await.unwrap().as_ref().clone();
        for idx in &mut indices {
            if let Some(details) = idx.index_details.as_ref()
                && details.type_url.ends_with("VectorIndexDetails")
            {
                idx.index_details = Some(Arc::new(vector_index_details_default()));
            }
        }
        // Write back via a no-op commit that carries the cleared indices.
        // We commit by doing a delete("false") after replacing the cached indices.
        let metadata_key = crate::session::index_caches::IndexMetadataKey {
            version: dataset.version().version,
            store_identity: &dataset.object_store.store_prefix,
            e_tag: dataset.manifest_location.e_tag.as_deref(),
        };
        dataset
            .index_cache
            .insert_with_key(&metadata_key, Arc::new(indices))
            .await;
        dataset.delete("false").await.unwrap();

        // -- Part 1: Inference on load --
        // Open with a fresh session so nothing is cached.
        let dataset = DatasetBuilder::from_uri(&test_uri)
            .with_session(Arc::new(Session::default()))
            .load()
            .await
            .unwrap();

        // load_indices should detect empty details and infer from index files.
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        let details = indices[0].index_details.as_ref().unwrap();
        assert!(
            !details.value.is_empty(),
            "Details should have been inferred from index files"
        );

        // describe_indices should return a real type (not generic "Vector").
        let descriptions = dataset.describe_indices(None).await.unwrap();
        assert_eq!(descriptions.len(), 1);
        assert_ne!(
            descriptions[0].index_type(),
            "Vector",
            "Should have inferred a specific index type"
        );
        assert_eq!(
            descriptions[0].index_type(),
            "IVF_HNSW_SQ",
            "Inferred type should match the originally-built index"
        );
        let inferred_type = descriptions[0].index_type().to_string();
        let details_json: serde_json::Value =
            serde_json::from_str(&descriptions[0].details().unwrap()).unwrap();
        assert_eq!(details_json["metric_type"], "COSINE");
        assert_eq!(details_json["compression"]["type"], "sq");
        assert!(
            details_json["hnsw"]["max_connections"].is_number(),
            "Inferred HNSW config should have max_connections; got {details_json}"
        );
        assert!(details_json["hnsw"]["construction_ef"].is_number());
        assert!(details_json["hnsw"]["max_level"].is_number());

        // -- Part 2: Migration persists inferred details --
        let mut dataset = dataset;
        dataset.delete("false").await.unwrap();

        // Open with yet another fresh session.
        let dataset = DatasetBuilder::from_uri(&test_uri)
            .with_session(Arc::new(Session::default()))
            .load()
            .await
            .unwrap();

        // The migrated manifest should have non-empty details without
        // needing to read index files again.
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert!(
            !indices[0].index_details.as_ref().unwrap().value.is_empty(),
            "Migrated manifest should have non-empty details"
        );
        let descriptions = dataset.describe_indices(None).await.unwrap();
        assert_eq!(descriptions[0].index_type(), inferred_type);
    }

    #[tokio::test]
    async fn test_describe_indices_tolerates_missing_index_details() {
        // An index whose manifest entry has no index details (e.g. created
        // before details were persisted) is still described on a best-effort
        // basis rather than causing describe_indices to error.
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let dataset = Dataset::write(reader, test_uri, None).await.unwrap();
        let field_id = dataset.schema().field("id").unwrap().id;

        let metadata = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "mystery_idx".to_string(),
            fields: vec![field_id],
            covering_fields: vec![],
            dataset_version: dataset.manifest.version,
            fragment_bitmap: Some(std::iter::once(0_u32).collect()),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
        };

        let desc = IndexDescriptionImpl::try_new(vec![metadata], &dataset)
            .await
            .unwrap();
        assert_eq!(desc.index_type(), "Unknown");
        assert_eq!(desc.type_url(), "");
        assert_eq!(desc.details().unwrap(), "{}");
        assert_eq!(desc.rows_indexed(), 10);
    }

    /// `describe_indices` with a column criteria must still find a covered index.
    ///
    /// `index_matches_criteria` is documented to receive only the column named by
    /// `for_column` -- it rejects anything else with `fields.len() != 1`. A covered
    /// index lists its carried columns in `fields` too, so resolving all of `fields`
    /// silently fails that check and drops the index from the description with no
    /// error: `describe_indices(for_column)` reports "no index on this column" for a
    /// column that is indexed.
    #[tokio::test]
    async fn test_describe_indices_for_column_finds_covered_index() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(20), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let mut params = crate::index::vector::VectorIndexParams::ivf_flat(
            1,
            lance_linalg::distance::MetricType::L2,
        );
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("covered_idx".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();
        let covered = dataset.load_indices_by_name("covered_idx").await.unwrap();
        assert_eq!(covered[0].fields.len(), 2, "index should be covered");

        let descriptions = dataset
            .describe_indices(Some(IndexCriteria::default().for_column("vector")))
            .await
            .unwrap();
        assert_eq!(
            descriptions.len(),
            1,
            "a covered index must still be described for its keyed column"
        );
        assert_eq!(descriptions[0].name(), "covered_idx");

        // The carried column is not the indexed column, so it must not match.
        let by_carried = dataset
            .describe_indices(Some(IndexCriteria::default().for_column("id")))
            .await
            .unwrap();
        assert!(
            by_carried.is_empty(),
            "a carried column is not indexed and must not match `for_column`"
        );

        // `field_ids()` is a capability statement -- "the fields the index is built on" --
        // so it must report the keyed column only. Reporting the carried column here tells
        // every consumer (Python `describe_indices().fields`, Java `getFieldIds()`, the
        // namespace REST `columns` list) that this index can serve `id`; Python's
        // `_default_vector_index_for_column` then hands back `vec`'s IVF centroids for the
        // int32 column `id` instead of raising.
        let vector_field_id = dataset.schema().field("vector").unwrap().id as u32;
        let id_field_id = dataset.schema().field("id").unwrap().id as u32;
        let desc = &descriptions[0];
        assert_eq!(
            desc.field_ids(),
            &[vector_field_id],
            "field_ids() must report the keyed column only, not the carried one"
        );
        // The carried set is not lost -- it stays reachable per segment.
        assert_eq!(
            desc.metadata()[0].covering_fields,
            vec![id_field_id as i32],
            "the carried column must remain reachable via metadata()[i].covering_fields"
        );
    }

    #[tokio::test]
    async fn test_describe_indices_derives_type_from_url_without_plugin() {
        // When index details exist but no plugin is registered for the type
        // URL, the index type is derived from the type URL rather than being
        // reported as the opaque "Unknown".
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let dataset = Dataset::write(reader, test_uri, None).await.unwrap();
        let field_id = dataset.schema().field("id").unwrap().id;

        let metadata = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "mystery_idx".to_string(),
            fields: vec![field_id],
            covering_fields: vec![],
            dataset_version: dataset.manifest.version,
            fragment_bitmap: Some(std::iter::once(0_u32).collect()),
            index_details: Some(Arc::new(prost_types::Any {
                type_url: "/lance.index.pb.MysteryIndexDetails".to_string(),
                value: Vec::new(),
            })),
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
        };

        let desc = IndexDescriptionImpl::try_new(vec![metadata], &dataset)
            .await
            .unwrap();
        assert_eq!(desc.index_type(), "Mystery");
        assert_eq!(desc.type_url(), "/lance.index.pb.MysteryIndexDetails");
    }

    #[rstest]
    #[case::btree("i", IndexType::BTree, Box::new(ScalarIndexParams::default()))]
    #[case::bitmap("i", IndexType::Bitmap, Box::new(ScalarIndexParams::default()))]
    #[case::inverted("text", IndexType::Inverted, Box::new(InvertedIndexParams::default()))]
    #[tokio::test]
    async fn test_create_empty_scalar_index(
        #[case] column_name: &str,
        #[case] index_type: IndexType,
        #[case] params: Box<dyn IndexParams>,
    ) {
        use lance_datagen::{BatchCount, ByteCount, RowCount, array};

        // Create dataset with scalar and text columns (no vector column needed)
        let reader = lance_datagen::gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("text", array::rand_utf8(ByteCount::from(10), false))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, "memory://test", None).await.unwrap();

        // Create an empty index with train=false
        // Test using IntoFuture - can await directly without calling .execute()
        dataset
            .create_index_builder(&[column_name], index_type, params.as_ref())
            .name("index".to_string())
            .train(false)
            .await
            .unwrap();

        // Verify we can get index statistics
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"], 0,
            "Empty index should have zero indexed rows"
        );

        // Append new data using lance_datagen
        let append_reader = lance_datagen::gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("text", array::rand_utf8(ByteCount::from(10), false))
            .into_reader_rows(RowCount::from(50), BatchCount::from(1));

        dataset.append(append_reader, None).await.unwrap();

        // Critical test: Verify the empty index is still present after append
        let indices_after_append = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_append.len(),
            1,
            "Index should be retained after append for index type {:?}",
            index_type
        );

        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"], 0,
            "Empty index should still have zero indexed rows after append"
        );

        // Test optimize_indices with empty index
        dataset.optimize_indices(&Default::default()).await.unwrap();

        // Verify the index still exists after optimization
        let indices_after_optimize = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_optimize.len(),
            1,
            "Index should still exist after optimization"
        );

        // Check index statistics after optimization
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_unindexed_rows"], 0,
            "Empty index should indexed all rows"
        );
    }

    #[rstest]
    #[case::block_size_128(128)]
    #[case::block_size_256(256)]
    #[tokio::test]
    async fn test_optimize_empty_code_fts_index_preserves_params(#[case] block_size: usize) {
        let dir = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("code", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values([0, 1])),
                Arc::new(StringArray::from_iter_values([
                    "fn GetUserEmail() {}",
                    "fn ParseConfig() {}",
                ])),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, &dir, None).await.unwrap();

        let params = InvertedIndexParams::default()
            .analyzer("code")
            .unwrap()
            .block_size(block_size)
            .unwrap()
            .split_identifiers(true)
            .preserve_original(false);
        dataset
            .create_index_builder(&["code"], IndexType::Inverted, &params)
            .name("code_idx".to_string())
            .train(false)
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].index_version, INVERTED_INDEX_VERSION_V3 as i32);

        dataset.optimize_indices(&Default::default()).await.unwrap();

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].index_version, INVERTED_INDEX_VERSION_V3 as i32);
        let details = indices[0]
            .index_details
            .as_ref()
            .expect("optimized FTS index should retain index details")
            .to_msg::<InvertedIndexDetails>()
            .unwrap();
        assert_eq!(details.base_tokenizer.as_deref(), Some("code"));
        assert_eq!(details.block_size, Some(block_size as u32));
        let code_config = details
            .code_config
            .expect("optimized code FTS index should retain code configuration");
        assert!(code_config.split_identifiers);
        assert_eq!(code_config.preserve_original, Some(false));

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("code_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);

        let result = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("email".to_string()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result["id"].as_primitive::<Int32Type>().values(), &[0]);
    }

    /// Helper function to check if an index is being used in a query plan
    fn assert_index_usage(plan: &str, column_name: &str, should_use_index: bool, context: &str) {
        let index_used = if column_name == "text" {
            // For inverted index, look for MatchQuery which indicates FTS index usage
            plan.contains("MatchQuery")
        } else {
            // For btree/bitmap, look for MaterializeIndex which indicates scalar index usage
            plan.contains("ScalarIndexQuery")
        };

        if should_use_index {
            assert!(
                index_used,
                "Query plan should use index {}: {}",
                context, plan
            );
        } else {
            assert!(
                !index_used,
                "Query plan should NOT use index {}: {}",
                context, plan
            );
        }
    }

    /// Test that scalar indices are retained after deleting all data from a table.
    ///
    /// This test verifies that when we:
    /// 1. Create a table with data
    /// 2. Add a scalar index with train=true
    /// 3. Delete all data in the table
    /// The index remains available on the table.
    #[rstest]
    #[case::btree("i", IndexType::BTree, Box::new(ScalarIndexParams::default()))]
    #[case::bitmap("i", IndexType::Bitmap, Box::new(ScalarIndexParams::default()))]
    #[case::inverted("text", IndexType::Inverted, Box::new(InvertedIndexParams::default()))]
    #[tokio::test]
    async fn test_scalar_index_retained_after_delete_all(
        #[case] column_name: &str,
        #[case] index_type: IndexType,
        #[case] params: Box<dyn IndexParams>,
    ) {
        use lance_datagen::{BatchCount, ByteCount, RowCount, array};

        // Create dataset with initial data
        let reader = lance_datagen::gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("text", array::rand_utf8(ByteCount::from(10), false))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, "memory://test", None).await.unwrap();

        // Create index with train=true (normal index with data)
        dataset
            .create_index_builder(&[column_name], index_type, params.as_ref())
            .name("index".to_string())
            .train(true)
            .await
            .unwrap();

        // Verify index was created and has indexed rows
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"], 100,
            "Index should have indexed all 100 rows"
        );

        // Verify index is being used in queries before delete
        let plan = if column_name == "text" {
            // Use full-text search for inverted index
            dataset
                .scan()
                .full_text_search(FullTextSearchQuery::new("test".to_string()))
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap()
        } else {
            // Use equality filter for btree/bitmap indices
            dataset
                .scan()
                .filter(format!("{} = 50", column_name).as_str())
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap()
        };
        // Verify index is being used before delete
        assert_index_usage(&plan, column_name, true, "before delete");

        let indexes = dataset.load_indices().await.unwrap();
        let original_index = indexes[0].clone();

        // Delete all rows from the table
        dataset.delete("true").await.unwrap();

        // Verify table is empty
        let row_count = dataset.count_rows(None).await.unwrap();
        assert_eq!(row_count, 0, "Table should be empty after delete all");

        // Critical test: Verify the index still exists after deleting all data
        let indices_after_delete = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_delete.len(),
            1,
            "Index should be retained after deleting all data"
        );
        assert_eq!(
            indices_after_delete[0].name, "index",
            "Index name should remain the same after delete"
        );

        // Critical test: Verify the fragment bitmap is empty after delete
        let index_after_delete = &indices_after_delete[0];
        let effective_bitmap = index_after_delete
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .unwrap();
        assert!(
            effective_bitmap.is_empty(),
            "Effective bitmap should be empty after deleting all data"
        );
        assert_eq!(
            index_after_delete.fragment_bitmap, original_index.fragment_bitmap,
            "Fragment bitmap should remain the same after delete"
        );

        // Verify we can still get index statistics
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"], 0,
            "Index should now report zero indexed rows after delete all"
        );
        assert_eq!(
            stats["num_unindexed_rows"], 0,
            "Index should report zero unindexed rows after delete all"
        );
        assert_eq!(
            stats["num_indexed_fragments"], 0,
            "Index should report zero indexed fragments after delete all"
        );
        assert_eq!(
            stats["num_unindexed_fragments"], 0,
            "Index should report zero unindexed fragments after delete all"
        );

        // Verify index is NOT being used in queries after delete (empty bitmap)
        if column_name == "text" {
            // Inverted indexes will still appear to be used in FTS queries.
            // TODO: once metrics are working on FTS queries, we can check the
            // analyze plan output instead for index usage.
            let _plan_after_delete = dataset
                .scan()
                .project(&[column_name])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new("test".to_string()))
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap();
            assert_index_usage(
                &_plan_after_delete,
                column_name,
                true,
                "after delete (empty bitmap)",
            );
        } else {
            // Use equality filter for btree/bitmap indices
            let _plan_after_delete = dataset
                .scan()
                .filter(format!("{} = 50", column_name).as_str())
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap();
            // Verify index is NOT being used after delete (empty bitmap)
            assert_index_usage(
                &_plan_after_delete,
                column_name,
                false,
                "after delete (empty bitmap)",
            );
        }

        // Test that we can append new data and the index is still there
        let append_reader = lance_datagen::gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("text", array::rand_utf8(ByteCount::from(10), false))
            .into_reader_rows(RowCount::from(50), BatchCount::from(1));

        dataset.append(append_reader, None).await.unwrap();

        // Verify index still exists after append
        let indices_after_append = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_append.len(),
            1,
            "Index should still exist after appending to empty table"
        );
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"], 0,
            "Index should now report zero indexed rows after data is added"
        );

        // Test optimize_indices after delete all
        dataset.optimize_indices(&Default::default()).await.unwrap();

        // Verify index still exists after optimization
        let indices_after_optimize = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_optimize.len(),
            1,
            "Index should still exist after optimization following delete all"
        );

        // Verify we can still get index statistics after optimization
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"],
            dataset.count_rows(None).await.unwrap(),
            "Index should now cover all newly added rows after optimization"
        );
    }

    /// Test that scalar indices are retained after updating rows in a table.
    ///
    /// This test verifies that when we:
    /// 1. Create a table with data
    /// 2. Add a scalar index with train=true
    /// 3. Update rows in the table
    /// The index remains available on the table.
    #[rstest]
    #[case::btree("i", IndexType::BTree, Box::new(ScalarIndexParams::default()))]
    #[case::bitmap("i", IndexType::Bitmap, Box::new(ScalarIndexParams::default()))]
    #[case::inverted("text", IndexType::Inverted, Box::new(InvertedIndexParams::default()))]
    #[tokio::test]
    async fn test_scalar_index_retained_after_update(
        #[case] column_name: &str,
        #[case] index_type: IndexType,
        #[case] params: Box<dyn IndexParams>,
    ) {
        use crate::dataset::UpdateBuilder;
        use lance_datagen::{BatchCount, ByteCount, RowCount, array};

        // Create dataset with initial data
        let reader = lance_datagen::gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("text", array::rand_utf8(ByteCount::from(10), false))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, "memory://test", None).await.unwrap();

        // Create index with train=true (normal index with data)
        dataset
            .create_index_builder(&[column_name], index_type, params.as_ref())
            .name("index".to_string())
            .train(true)
            .await
            .unwrap();

        // Verify index was created and has indexed rows
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"], 100,
            "Index should have indexed all 100 rows"
        );

        // Verify index is being used in queries before update
        let plan = if column_name == "text" {
            // Use full-text search for inverted index
            dataset
                .scan()
                .project(&[column_name])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new("test".to_string()))
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap()
        } else {
            // Use equality filter for btree/bitmap indices
            dataset
                .scan()
                .filter(format!("{} = 50", column_name).as_str())
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap()
        };
        // Verify index is being used before update
        assert_index_usage(&plan, column_name, true, "before update");

        // Update some rows - update first 50 rows
        let update_result = UpdateBuilder::new(Arc::new(dataset))
            .set("i", "i + 1000")
            .unwrap()
            .set("text", "'updated_' || text")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let mut dataset = update_result.new_dataset.as_ref().clone();

        // Verify row count remains the same
        let row_count = dataset.count_rows(None).await.unwrap();
        assert_eq!(row_count, 100, "Row count should remain 100 after update");

        // Critical test: Verify the index still exists after updating data
        let indices_after_update = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_update.len(),
            1,
            "Index should be retained after updating rows"
        );

        // Critical test: Verify the effective fragment bitmap is empty after update
        let indices = dataset.load_indices().await.unwrap();
        let index = &indices[0];
        let effective_bitmap = index
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .unwrap();
        assert!(
            effective_bitmap.is_empty(),
            "Effective fragment bitmap should be empty after updating all data"
        );

        // Verify we can still get index statistics
        let stats_after_update = dataset.index_statistics("index").await.unwrap();
        let stats_after_update: serde_json::Value =
            serde_json::from_str(&stats_after_update).unwrap();

        // The index should still be available
        assert_eq!(
            stats_after_update["num_indexed_rows"], 0,
            "Index statistics should be zero after update, as it is not re-trained"
        );

        // Verify index behavior in queries after update (empty bitmap)
        if column_name == "text" {
            // Inverted indexes will still appear to be used in FTS queries even with empty bitmaps.
            // This is because FTS queries require an index to exist, and the query execution
            // will handle the empty bitmap appropriately by falling back to scanning unindexed data.
            // TODO: once metrics are working on FTS queries, we can check the
            // analyze plan output instead for actual index usage statistics.
            let _plan_after_update = dataset
                .scan()
                .project(&[column_name])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new("test".to_string()))
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap();
            assert_index_usage(
                &_plan_after_update,
                column_name,
                true, // FTS indices always appear in the plan, even with empty bitmaps
                "after update (empty bitmap)",
            );
        } else {
            // Use equality filter for btree/bitmap indices
            let _plan_after_update = dataset
                .scan()
                .filter(format!("{} = 50", column_name).as_str())
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap();
            // With immutable bitmaps, index is still used even with empty effective bitmap
            // The prefilter will handle non-existent fragments
            assert_index_usage(
                &_plan_after_update,
                column_name,
                false,
                "after update (empty effective bitmap)",
            );
        }

        // Test that we can optimize indices after update
        dataset.optimize_indices(&Default::default()).await.unwrap();

        // Verify index still exists after optimization
        let indices_after_optimize = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_optimize.len(),
            1,
            "Index should still exist after optimization following update"
        );

        let stats_after_optimization = dataset.index_statistics("index").await.unwrap();
        let stats_after_optimization: serde_json::Value =
            serde_json::from_str(&stats_after_optimization).unwrap();

        // The index should still be available
        assert_eq!(
            stats_after_optimization["num_unindexed_rows"], 0,
            "Index should have zero unindexed rows after optimization"
        );
    }

    // Helper function to validate indices after each clone iteration
    async fn validate_indices_after_clone(
        dataset: &Dataset,
        round: usize,
        expected_scalar_rows: usize,
        dimensions: u32,
    ) {
        // Verify cloned dataset has indices
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices.len(),
            2,
            "Round {}: Cloned dataset should have 2 indices",
            round
        );
        let index_names: HashSet<String> = indices.iter().map(|idx| idx.name.clone()).collect();
        assert!(
            index_names.contains("vector_idx"),
            "Round {}: Should contain vector_idx",
            round
        );
        assert!(
            index_names.contains("category_idx"),
            "Round {}: Should contain category_idx",
            round
        );

        // Test basic data access without using indices for now
        // This ensures the dataset is accessible and data is intact
        // In chain cloning, each round adds 50 rows to the previous dataset
        // Round 1: 300 (original), Round 2: 350 (300 + 50 from round 1), Round 3: 400 (350 + 50 from round 2)
        let expected_total_rows = 300 + (round - 1) * 50;
        let total_rows = dataset.count_rows(None).await.unwrap();
        assert_eq!(
            total_rows, expected_total_rows,
            "Round {}: Should have {} rows after clone (chain cloning accumulates data)",
            round, expected_total_rows
        );

        // Verify vector search
        let query_vector = generate_random_array(dimensions as usize);
        let search_results = dataset
            .scan()
            .nearest("vector", &query_vector, 5)
            .unwrap()
            .limit(Some(5), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(
            search_results.num_rows() > 0,
            "Round {}: Vector search should return results immediately after clone",
            round
        );

        // Test basic scalar query to verify data integrity
        let scalar_results = dataset
            .scan()
            .filter("category = 'category_0'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(
            expected_scalar_rows,
            scalar_results.num_rows(),
            "Round {}: Scalar query should return {} results",
            round,
            expected_scalar_rows
        );
    }

    #[tokio::test]
    async fn test_shallow_clone_with_index() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Create a schema with both vector and scalar columns
        let dimensions = 16u32;
        // Generate test data using lance_datagen (300 rows to satisfy PQ training requirements)
        let data = gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("category", array::fill_utf8("category_0".to_string()))
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(dimensions)),
            )
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));

        // Create initial dataset
        let mut dataset = Dataset::write(data, test_uri, None).await.unwrap();
        // Create vector index (IVF_PQ)
        let vector_params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 10);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &vector_params,
                true,
            )
            .await
            .unwrap();

        // Create scalar index (BTree)
        dataset
            .create_index(
                &["category"],
                IndexType::BTree,
                Some("category_idx".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Verify indices were created
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 2, "Should have 2 indices");
        let index_names: HashSet<String> = indices.iter().map(|idx| idx.name.clone()).collect();
        assert!(index_names.contains("vector_idx"));
        assert!(index_names.contains("category_idx"));

        // Test scalar query on source dataset
        let scalar_results = dataset
            .scan()
            .filter("category = 'category_0'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let source_scalar_query_rows = scalar_results.num_rows();
        assert!(
            scalar_results.num_rows() > 0,
            "Scalar query should return results"
        );

        // Multiple shallow clone iterations test with chain cloning
        let clone_rounds = 3;
        let mut current_dataset = dataset;

        for round in 1..=clone_rounds {
            let round_clone_dir = format!("{}/clone_round_{}", test_dir, round);
            let round_cloned_uri = &round_clone_dir;
            let tag_name = format!("shallow_clone_test_{}", round);

            // Create tag for this round (use current dataset for chain cloning)
            let current_version = current_dataset.version().version;
            current_dataset
                .tags()
                .create(&tag_name, current_version)
                .await
                .unwrap();

            // Perform shallow clone for this round (chain cloning from current dataset)
            let mut round_cloned_dataset = current_dataset
                .shallow_clone(round_cloned_uri, tag_name.as_str(), None)
                .await
                .unwrap();

            // Immediately validate indices after each clone
            validate_indices_after_clone(
                &round_cloned_dataset,
                round,
                source_scalar_query_rows,
                dimensions,
            )
            .await;

            // Complete validation cycle after each clone: append data, optimize index, validate
            // Append new data to the cloned dataset
            let new_data = gen_batch()
                .col(
                    "id",
                    array::step_custom::<Int32Type>(300 + (round * 50) as i32, 1),
                )
                .col("category", array::fill_utf8(format!("category_{}", round)))
                .col(
                    "vector",
                    array::rand_vec::<Float32Type>(Dimension::from(dimensions)),
                )
                .into_reader_rows(RowCount::from(50), BatchCount::from(1));

            round_cloned_dataset = Dataset::write(
                new_data,
                round_cloned_uri,
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

            // Verify row count increased
            let expected_rows = 300 + round * 50;
            let total_rows = round_cloned_dataset.count_rows(None).await.unwrap();
            assert_eq!(
                total_rows, expected_rows,
                "Round {}: Should have {} rows after append",
                round, expected_rows
            );

            let indices_before_optimize = round_cloned_dataset.load_indices().await.unwrap();
            let vector_idx_before = indices_before_optimize
                .iter()
                .find(|idx| idx.name == "vector_idx")
                .unwrap();
            let category_idx_before = indices_before_optimize
                .iter()
                .find(|idx| idx.name == "category_idx")
                .unwrap();

            // Optimize indices
            round_cloned_dataset
                .optimize_indices(&OptimizeOptions::merge(indices_before_optimize.len()))
                .await
                .unwrap();

            // Verify index UUID has changed
            let optimized_indices = round_cloned_dataset.load_indices().await.unwrap();
            let new_vector_idx = optimized_indices
                .iter()
                .find(|idx| idx.name == "vector_idx")
                .unwrap();
            let new_category_idx = optimized_indices
                .iter()
                .find(|idx| idx.name == "category_idx")
                .unwrap();

            assert_ne!(
                new_vector_idx.uuid, vector_idx_before.uuid,
                "Round {}: Vector index should have a new UUID after optimization",
                round
            );
            assert_ne!(
                new_category_idx.uuid, category_idx_before.uuid,
                "Round {}: Category index should have a new UUID after optimization",
                round
            );

            // Verify the location of index files
            use std::path::PathBuf;
            let clone_indices_dir = PathBuf::from(round_cloned_uri).join("_indices");
            let vector_index_dir = clone_indices_dir.join(new_vector_idx.uuid.to_string());
            let category_index_dir = clone_indices_dir.join(new_category_idx.uuid.to_string());

            assert!(
                vector_index_dir.exists(),
                "Round {}: New vector index directory should exist in cloned dataset location: {:?}",
                round,
                vector_index_dir
            );
            assert!(
                category_index_dir.exists(),
                "Round {}: New category index directory should exist in cloned dataset location: {:?}",
                round,
                category_index_dir
            );

            // Verify base id
            assert!(
                new_vector_idx.base_id.is_none(),
                "Round {}: New vector index should not have base_id after optimization in cloned dataset",
                round
            );
            assert!(
                new_category_idx.base_id.is_none(),
                "Round {}: New category index should not have base_id after optimization in cloned dataset",
                round
            );

            // Verify the source location does NOT contain new data
            let original_indices_dir = PathBuf::from(current_dataset.uri()).join("_indices");
            let wrong_vector_dir = original_indices_dir.join(new_vector_idx.uuid.to_string());
            let wrong_category_dir = original_indices_dir.join(new_category_idx.uuid.to_string());

            assert!(
                !wrong_vector_dir.exists(),
                "Round {}: New vector index should NOT be in original dataset location: {:?}",
                round,
                wrong_vector_dir
            );
            assert!(
                !wrong_category_dir.exists(),
                "Round {}: New category index should NOT be in original dataset location: {:?}",
                round,
                wrong_category_dir
            );

            // Validate data integrity and index functionality after optimization
            let old_category_results = round_cloned_dataset
                .scan()
                .filter("category = 'category_0'")
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            let new_category_results = round_cloned_dataset
                .scan()
                .filter(&format!("category = 'category_{}'", round))
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            assert_eq!(
                source_scalar_query_rows,
                old_category_results.num_rows(),
                "Round {}: Should find old category data with {} rows",
                round,
                source_scalar_query_rows
            );
            assert!(
                new_category_results.num_rows() > 0,
                "Round {}: Should find new category data",
                round
            );

            // Test vector search functionality
            let query_vector = generate_random_array(dimensions as usize);
            let search_results = round_cloned_dataset
                .scan()
                .nearest("vector", &query_vector, 10)
                .unwrap()
                .limit(Some(10), None)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            assert!(
                search_results.num_rows() > 0,
                "Round {}: Vector search should return results after optimization",
                round
            );

            // Verify statistic of indexes
            let vector_stats: serde_json::Value = serde_json::from_str(
                &round_cloned_dataset
                    .index_statistics("vector_idx")
                    .await
                    .unwrap(),
            )
            .unwrap();
            let category_stats: serde_json::Value = serde_json::from_str(
                &round_cloned_dataset
                    .index_statistics("category_idx")
                    .await
                    .unwrap(),
            )
            .unwrap();

            assert_eq!(
                vector_stats["num_indexed_rows"].as_u64().unwrap(),
                expected_rows as u64,
                "Round {}: Vector index should have {} indexed rows",
                round,
                expected_rows
            );
            assert_eq!(
                category_stats["num_indexed_rows"].as_u64().unwrap(),
                expected_rows as u64,
                "Round {}: Category index should have {} indexed rows",
                round,
                expected_rows
            );

            // Prepare for next round: use the cloned dataset as the source for next clone
            current_dataset = round_cloned_dataset;
        }

        // Use the final cloned dataset for any remaining tests
        let final_cloned_dataset = current_dataset;

        // Verify cloned dataset has indices
        let cloned_indices = final_cloned_dataset.load_indices().await.unwrap();
        assert_eq!(
            cloned_indices.len(),
            2,
            "Final cloned dataset should have 2 indices"
        );
        let cloned_index_names: HashSet<String> =
            cloned_indices.iter().map(|idx| idx.name.clone()).collect();
        assert!(cloned_index_names.contains("vector_idx"));
        assert!(cloned_index_names.contains("category_idx"));

        // Test vector search on final cloned dataset
        let query_vector = generate_random_array(dimensions as usize);
        let search_results = final_cloned_dataset
            .scan()
            .nearest("vector", &query_vector, 5)
            .unwrap()
            .limit(Some(5), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert!(
            search_results.num_rows() > 0,
            "Vector search should return results on final dataset"
        );

        // Test scalar query on final cloned dataset
        let scalar_results = final_cloned_dataset
            .scan()
            .filter("category = 'category_0'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(
            source_scalar_query_rows,
            scalar_results.num_rows(),
            "Scalar query should return results on final dataset"
        );
    }

    #[tokio::test]
    async fn test_initialize_indices() {
        use crate::dataset::Dataset;
        use arrow_array::types::Float32Type;
        use lance_core::utils::tempfile::TempStrDir;
        use lance_datagen::{BatchCount, RowCount, array};
        use lance_index::scalar::{InvertedIndexParams, ScalarIndexParams};
        use lance_linalg::distance::MetricType;
        use std::collections::HashSet;

        // Create source dataset with various index types
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/{}", test_dir, "source");
        let target_uri = format!("{}/{}", test_dir, "target");

        // Generate test data using lance_datagen (need at least 256 rows for PQ training)
        let source_reader = lance_datagen::gen_batch()
            .col("vector", array::rand_vec::<Float32Type>(8.into()))
            .col(
                "text",
                array::cycle_utf8_literals(&["hello world", "foo bar", "test data"]),
            )
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));

        // Create source dataset
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create indices on source dataset
        // 1. Vector index
        let vector_params = VectorIndexParams::ivf_pq(4, 8, 2, MetricType::L2, 10);
        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vec_idx".to_string()),
                &vector_params,
                false,
            )
            .await
            .unwrap();

        // 2. FTS index
        let fts_params = InvertedIndexParams::default();
        source_dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                Some("text_idx".to_string()),
                &fts_params,
                false,
            )
            .await
            .unwrap();

        // 3. Scalar index
        let scalar_params = ScalarIndexParams::default();
        source_dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".to_string()),
                &scalar_params,
                false,
            )
            .await
            .unwrap();

        // Reload source dataset to get updated index metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();

        // Verify source has 3 indices
        let source_indices = source_dataset.load_indices().await.unwrap();
        assert_eq!(
            source_indices.len(),
            3,
            "Source dataset should have 3 indices"
        );

        // Create target dataset with same schema but different data (need at least 256 rows for PQ)
        let target_reader = lance_datagen::gen_batch()
            .col("vector", array::rand_vec::<Float32Type>(8.into()))
            .col(
                "text",
                array::cycle_utf8_literals(&["foo bar", "test data", "hello world"]),
            )
            .col("id", array::step_custom::<Int32Type>(100, 1))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize indices from source dataset
        target_dataset
            .initialize_indices(&source_dataset)
            .await
            .unwrap();

        // Verify target has same indices
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(
            target_indices.len(),
            3,
            "Target dataset should have 3 indices after initialization"
        );

        // Check index names match
        let source_names: HashSet<String> =
            source_indices.iter().map(|idx| idx.name.clone()).collect();
        let target_names: HashSet<String> =
            target_indices.iter().map(|idx| idx.name.clone()).collect();
        assert_eq!(
            source_names, target_names,
            "Index names should match between source and target"
        );

        // Verify indices are functional by running queries
        // 1. Test vector index
        let query_vector = generate_random_array(8);
        let search_results = target_dataset
            .scan()
            .nearest("vector", &query_vector, 5)
            .unwrap()
            .limit(Some(5), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(
            search_results.num_rows() > 0,
            "Vector index should be functional"
        );

        // 2. Test scalar index
        let scalar_results = target_dataset
            .scan()
            .filter("id = 125")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            scalar_results.num_rows(),
            1,
            "Scalar index should find exact match"
        );
    }

    #[tokio::test]
    async fn test_initialize_indices_with_missing_field() {
        use crate::dataset::Dataset;
        use arrow_array::types::Int32Type;
        use lance_core::utils::tempfile::TempStrDir;
        use lance_datagen::{BatchCount, RowCount, array};
        use lance_index::scalar::ScalarIndexParams;

        // Test that initialize_indices handles missing fields gracefully
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/{}", test_dir, "source");
        let target_uri = format!("{}/{}", test_dir, "target");

        // Create source dataset with extra field
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("extra", array::cycle_utf8_literals(&["test"]))
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create index on extra field in source
        source_dataset
            .create_index(
                &["extra"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Create target dataset without extra field
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step_custom::<Int32Type>(10, 1))
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize indices should skip the index on missing field with an error
        let result = target_dataset.initialize_indices(&source_dataset).await;

        // Should fail when field is missing
        assert!(result.is_err(), "Should error when field is missing");
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not found in target dataset")
        );
    }

    /// `initialize_index` must depend only on the keyed prefix of the source
    /// index. The rebuild it drives writes fresh, non-covered metadata and never
    /// reads the carried columns, so requiring them to exist and type-match in
    /// the target rejects a target that can hold the index perfectly well.
    #[tokio::test]
    async fn test_initialize_index_ignores_carried_columns() {
        use crate::dataset::Dataset;
        use lance_index::scalar::ScalarIndexParams;

        let source_data = gen_batch()
            .col("a", array::step::<Int32Type>())
            .col("carried", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(20), BatchCount::from(1));
        let mut source = Dataset::write(source_data, "memory://source", None)
            .await
            .unwrap();
        source
            .create_index(
                &["a"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // No creation API writes a covering declaration yet, so commit one.
        crate::utils::test::covering::declare_covering(&mut source, "a", "carried").await;
        let index_name = source.load_indices().await.unwrap()[0].name.clone();

        // The target holds the keyed column only -- exactly what the rebuild
        // reads.
        let target_data = gen_batch()
            .col("a", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(20), BatchCount::from(1));
        let mut target = Dataset::write(target_data, "memory://target", None)
            .await
            .unwrap();

        target.initialize_index(&source, &index_name).await.unwrap();

        let initialized = target.load_indices().await.unwrap();
        assert_eq!(initialized.len(), 1);
        assert_eq!(
            initialized[0].fields,
            vec![target.schema().field("a").unwrap().id]
        );
        assert!(
            initialized[0].covering_fields.is_empty(),
            "the rebuild carries nothing, so it must not claim to"
        );
    }

    #[tokio::test]
    async fn test_initialize_single_index() {
        use crate::dataset::Dataset;
        use crate::index::vector::VectorIndexParams;
        use arrow_array::types::{Float32Type, Int32Type};
        use lance_core::utils::tempfile::TempStrDir;
        use lance_datagen::{BatchCount, RowCount, array};
        use lance_index::scalar::ScalarIndexParams;
        use lance_linalg::distance::MetricType;

        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/{}", test_dir, "source");
        let target_uri = format!("{}/{}", test_dir, "target");

        // Create source dataset (need at least 256 rows for PQ training)
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("name", array::rand_utf8(4.into(), false))
            .col("vector", array::rand_vec::<Float32Type>(8.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create multiple indices on source
        let scalar_params = ScalarIndexParams::default();
        source_dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_index".to_string()),
                &scalar_params,
                false,
            )
            .await
            .unwrap();

        let vector_params = VectorIndexParams::ivf_pq(16, 8, 4, MetricType::L2, 50);
        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_index".to_string()),
                &vector_params,
                false,
            )
            .await
            .unwrap();

        // Reload source dataset to get updated index metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("name", array::rand_utf8(4.into(), false))
            .col("vector", array::rand_vec::<Float32Type>(8.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize only the vector index
        target_dataset
            .initialize_index(&source_dataset, "vector_index")
            .await
            .unwrap();

        // Verify only vector index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Should have only 1 index");
        assert_eq!(
            target_indices[0].name, "vector_index",
            "Should have the vector index"
        );

        // Initialize the scalar index
        target_dataset
            .initialize_index(&source_dataset, "id_index")
            .await
            .unwrap();

        // Verify both indices now exist
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 2, "Should have 2 indices");

        let index_names: HashSet<String> =
            target_indices.iter().map(|idx| idx.name.clone()).collect();
        assert!(
            index_names.contains("vector_index"),
            "Should have vector index"
        );
        assert!(index_names.contains("id_index"), "Should have id index");

        // Test error case - non-existent index
        let result = target_dataset
            .initialize_index(&source_dataset, "non_existent")
            .await;
        assert!(result.is_err(), "Should error for non-existent index");
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not found in source dataset")
        );
    }

    /// `initialize_index` must copy a covered index, not reject it.
    ///
    /// It derives the column names it initializes from `source_index.fields`, and a
    /// covered index lists its carried columns there too. Both initializers accept
    /// exactly one name (they re-resolve the carried columns themselves from
    /// `covering_fields`), so passing all of `fields` makes every covered index
    /// un-copyable -- and `initialize_indices` then fails the whole dataset copy.
    #[tokio::test]
    async fn test_initialize_index_copies_covered_index() {
        use arrow_array::types::Float32Type;
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/{}", test_dir, "source");
        let target_uri = format!("{}/{}", test_dir, "target");

        let make_reader = || {
            lance_datagen::gen_batch()
                .col("id", array::step::<arrow_array::types::Int32Type>())
                .col("vector", array::rand_vec::<Float32Type>(8.into()))
                .into_reader_rows(RowCount::from(300), BatchCount::from(1))
        };

        let mut source_dataset = Dataset::write(make_reader(), &source_uri, None)
            .await
            .unwrap();
        let mut params = crate::index::vector::VectorIndexParams::ivf_flat(
            4,
            lance_linalg::distance::MetricType::L2,
        );
        params.covering_columns(vec!["id".to_string()]);
        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("covered_idx".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();
        let source_dataset = Dataset::open(&source_uri).await.unwrap();

        let mut target_dataset = Dataset::write(make_reader(), &target_uri, None)
            .await
            .unwrap();
        target_dataset
            .initialize_index(&source_dataset, "covered_idx")
            .await
            .unwrap();

        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1);
        let id_field_id = target_dataset.schema().field("id").unwrap().id;
        assert_eq!(
            target_indices[0].covering_fields,
            vec![id_field_id],
            "the copied index must keep its covering declaration"
        );
        target_indices[0].validate_covering_fields().unwrap();
    }

    #[rstest]
    #[case::simple("value", "data.value")]
    #[case::quoted("value.with.dot", "data.`value.with.dot`")]
    #[tokio::test]
    async fn test_initialize_index_on_nested_field(
        #[case] nested_field_name: &str,
        #[case] nested_field_path: &str,
    ) {
        let nested_field = Arc::new(Field::new(nested_field_name, DataType::Int32, false));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Struct(vec![nested_field.clone()].into()),
            false,
        )]));
        let nested_values = arrow_array::StructArray::from(vec![(
            nested_field,
            Arc::new(Int32Array::from_iter_values(0..10)) as Arc<dyn arrow_array::Array>,
        )]);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(nested_values)]).unwrap();

        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir);
        let target_uri = format!("{}/target", test_dir);

        let source_reader =
            RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), schema.clone());
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();
        source_dataset
            .create_index(
                &[nested_field_path],
                IndexType::BTree,
                Some("nested_idx".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let source_dataset = Dataset::open(&source_uri).await.unwrap();

        let target_reader =
            RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();
        target_dataset
            .initialize_index(&source_dataset, "nested_idx")
            .await
            .unwrap();

        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1);
        assert_eq!(
            target_dataset
                .schema()
                .field_path(target_indices[0].fields[0])
                .unwrap(),
            nested_field_path
        );
    }

    #[tokio::test]
    async fn test_vector_index_on_nested_field_with_dots() {
        let dimensions = 16;
        let num_rows = 256;

        // Create schema with nested field containing dots in the name
        let struct_field = Field::new(
            "embedding_data",
            DataType::Struct(
                vec![
                    Field::new(
                        "vector.v1", // Field name with dot
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::Float32, true)),
                            dimensions,
                        ),
                        false,
                    ),
                    Field::new(
                        "vector.v2", // Another field name with dot
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::Float32, true)),
                            dimensions,
                        ),
                        false,
                    ),
                    Field::new("metadata", DataType::Utf8, false),
                ]
                .into(),
            ),
            false,
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            struct_field,
        ]));

        // Generate test data
        let float_arr_v1 = generate_random_array(num_rows * dimensions as usize);
        let vectors_v1 = FixedSizeListArray::try_new_from_values(float_arr_v1, dimensions).unwrap();

        let float_arr_v2 = generate_random_array(num_rows * dimensions as usize);
        let vectors_v2 = FixedSizeListArray::try_new_from_values(float_arr_v2, dimensions).unwrap();

        let ids = Int32Array::from_iter_values(0..num_rows as i32);
        let metadata = StringArray::from_iter_values((0..num_rows).map(|i| format!("meta_{}", i)));

        let struct_array = arrow_array::StructArray::from(vec![
            (
                Arc::new(Field::new(
                    "vector.v1",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dimensions,
                    ),
                    false,
                )),
                Arc::new(vectors_v1) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new(
                    "vector.v2",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dimensions,
                    ),
                    false,
                )),
                Arc::new(vectors_v2) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("metadata", DataType::Utf8, false)),
                Arc::new(metadata) as Arc<dyn arrow_array::Array>,
            ),
        ]);

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)])
                .unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Test creating index on nested field with dots using quoted syntax
        let nested_column_path_v1 = "embedding_data.`vector.v1`";
        let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 10);

        dataset
            .create_index(
                &[nested_column_path_v1],
                IndexType::Vector,
                Some("vec_v1_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Verify index was created
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "vec_v1_idx");

        // Verify the correct field was indexed
        let field_id = indices[0].fields[0];
        let field_path = dataset.schema().field_path(field_id).unwrap();
        assert_eq!(field_path, "embedding_data.`vector.v1`");

        // Test creating index on the second vector field with dots
        let nested_column_path_v2 = "embedding_data.`vector.v2`";
        dataset
            .create_index(
                &[nested_column_path_v2],
                IndexType::Vector,
                Some("vec_v2_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Verify both indices exist
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 2);

        // Verify we can query using the indexed fields and check the plan
        let query_vector = generate_random_array(dimensions as usize);

        // Check the query plan for the first vector field
        let plan_v1 = dataset
            .scan()
            .nearest(nested_column_path_v1, &query_vector, 5)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();

        // Verify the vector index is being used (should show ANNSubIndex or ANNIvfPartition)
        assert!(
            plan_v1.contains("ANNSubIndex") || plan_v1.contains("ANNIvfPartition"),
            "Query plan should use vector index for nested field with dots. Plan: {}",
            plan_v1
        );

        let search_results_v1 = dataset
            .scan()
            .nearest(nested_column_path_v1, &query_vector, 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(search_results_v1.num_rows(), 5);

        // Check the query plan for the second vector field
        let plan_v2 = dataset
            .scan()
            .nearest(nested_column_path_v2, &query_vector, 5)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();

        // Verify the vector index is being used
        assert!(
            plan_v2.contains("ANNSubIndex") || plan_v2.contains("ANNIvfPartition"),
            "Query plan should use vector index for second nested field with dots. Plan: {}",
            plan_v2
        );

        let search_results_v2 = dataset
            .scan()
            .nearest(nested_column_path_v2, &query_vector, 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(search_results_v2.num_rows(), 5);
    }

    #[tokio::test]
    async fn test_vector_index_on_simple_nested_field() {
        // This test reproduces the Python test scenario from test_nested_field_vector_index
        // where the nested field path is simple (data.embedding) without dots in field names
        let dimensions = 16;
        let num_rows = 256;

        // Create schema with simple nested field (no dots in field names)
        let struct_field = Field::new(
            "data",
            DataType::Struct(
                vec![
                    Field::new(
                        "embedding",
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::Float32, true)),
                            dimensions,
                        ),
                        false,
                    ),
                    Field::new("label", DataType::Utf8, false),
                ]
                .into(),
            ),
            false,
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            struct_field,
        ]));

        // Generate test data
        let float_arr = generate_random_array(num_rows * dimensions as usize);
        let vectors = FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

        let ids = Int32Array::from_iter_values(0..num_rows as i32);
        let labels = StringArray::from_iter_values((0..num_rows).map(|i| format!("label_{}", i)));

        let struct_array = arrow_array::StructArray::from(vec![
            (
                Arc::new(Field::new(
                    "embedding",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dimensions,
                    ),
                    false,
                )),
                Arc::new(vectors) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("label", DataType::Utf8, false)),
                Arc::new(labels) as Arc<dyn arrow_array::Array>,
            ),
        ]);

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)])
                .unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Test creating index on nested field
        let nested_column_path = "data.embedding";
        let params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 10);

        dataset
            .create_index(
                &[nested_column_path],
                IndexType::Vector,
                Some("vec_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Verify index was created
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "vec_idx");

        // Verify the correct field was indexed
        let field_id = indices[0].fields[0];
        let field_path = dataset.schema().field_path(field_id).unwrap();
        assert_eq!(field_path, "data.embedding");

        // Test querying with the index
        let query_vector = generate_random_array(dimensions as usize);

        let plan = dataset
            .scan()
            .nearest(nested_column_path, &query_vector, 5)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();

        // Verify the vector index is being used
        assert!(
            plan.contains("ANNSubIndex") || plan.contains("ANNIvfPartition"),
            "Query plan should use vector index for nested field. Plan: {}",
            plan
        );

        let search_results = dataset
            .scan()
            .nearest(nested_column_path, &query_vector, 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(search_results.num_rows(), 5);
    }

    #[tokio::test]
    async fn test_btree_index_on_nested_field_with_dots() {
        // Test creating BTree index on nested field with dots in the name
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Create schema with nested field containing dots
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "data",
                DataType::Struct(
                    vec![
                        Field::new("value.v1", DataType::Int32, false),
                        Field::new("value.v2", DataType::Float32, false),
                        Field::new("text", DataType::Utf8, false),
                    ]
                    .into(),
                ),
                false,
            ),
        ]));

        // Generate test data
        let num_rows = 1000;
        let ids = Int32Array::from_iter_values(0..num_rows);
        let values_v1 = Int32Array::from_iter_values((0..num_rows).map(|i| i % 100));
        let values_v2 = Float32Array::from_iter_values((0..num_rows).map(|i| (i as f32) * 0.1));
        let texts = StringArray::from_iter_values((0..num_rows).map(|i| format!("text_{}", i)));

        let struct_array = arrow_array::StructArray::from(vec![
            (
                Arc::new(Field::new("value.v1", DataType::Int32, false)),
                Arc::new(values_v1) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("value.v2", DataType::Float32, false)),
                Arc::new(values_v2) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("text", DataType::Utf8, false)),
                Arc::new(texts) as Arc<dyn arrow_array::Array>,
            ),
        ]);

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)])
                .unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Create BTree index on nested field with dots
        let nested_column_path = "data.`value.v1`";
        let params = ScalarIndexParams::default();

        dataset
            .create_index(
                &[nested_column_path],
                IndexType::BTree,
                Some("btree_v1_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Reload dataset to ensure index is loaded
        dataset = Dataset::open(test_uri).await.unwrap();

        // Verify index was created
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "btree_v1_idx");

        // Verify the correct field was indexed
        let field_id = indices[0].fields[0];
        let field_path = dataset.schema().field_path(field_id).unwrap();
        assert_eq!(field_path, "data.`value.v1`");

        // Test querying with the index and verify it's being used
        let plan = dataset
            .scan()
            .filter("data.`value.v1` = 42")
            .unwrap()
            .prefilter(true)
            .explain_plan(false)
            .await
            .unwrap();

        // Verify the query plan (scalar indices on nested fields may use optimized filters)
        // The index may be used internally even if not shown as ScalarIndexQuery
        assert!(
            plan.contains("ScalarIndexQuery"),
            "Query plan should show optimized read. Plan: {}",
            plan
        );

        // Also test that the query returns results
        let results = dataset
            .scan()
            .filter("data.`value.v1` = 42")
            .unwrap()
            .prefilter(true)
            .try_into_batch()
            .await
            .unwrap();

        assert!(results.num_rows() > 0);
    }

    #[tokio::test]
    async fn test_bitmap_index_on_nested_field_with_dots() {
        // Test creating Bitmap index on nested field with dots in the name
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Create schema with nested field containing dots - using low cardinality for bitmap
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "metadata",
                DataType::Struct(
                    vec![
                        Field::new("status.code", DataType::Int32, false),
                        Field::new("category.name", DataType::Utf8, false),
                    ]
                    .into(),
                ),
                false,
            ),
        ]));

        // Generate test data with low cardinality (good for bitmap index)
        let num_rows = 1000;
        let ids = Int32Array::from_iter_values(0..num_rows);
        // Only 10 unique status codes
        let status_codes = Int32Array::from_iter_values((0..num_rows).map(|i| i % 10));
        // Only 5 unique categories
        let categories =
            StringArray::from_iter_values((0..num_rows).map(|i| format!("category_{}", i % 5)));

        let struct_array = arrow_array::StructArray::from(vec![
            (
                Arc::new(Field::new("status.code", DataType::Int32, false)),
                Arc::new(status_codes) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("category.name", DataType::Utf8, false)),
                Arc::new(categories) as Arc<dyn arrow_array::Array>,
            ),
        ]);

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)])
                .unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Create Bitmap index on nested field with dots
        let nested_column_path = "metadata.`status.code`";
        let params = ScalarIndexParams::default();

        dataset
            .create_index(
                &[nested_column_path],
                IndexType::Bitmap,
                Some("bitmap_status_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Reload dataset to ensure index is loaded
        dataset = Dataset::open(test_uri).await.unwrap();

        // Verify index was created
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "bitmap_status_idx");

        // Verify the correct field was indexed
        let field_id = indices[0].fields[0];
        let field_path = dataset.schema().field_path(field_id).unwrap();
        assert_eq!(field_path, "metadata.`status.code`");

        // Test querying with the index and verify it's being used
        let plan = dataset
            .scan()
            .filter("metadata.`status.code` = 5")
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();

        // Verify the query plan (scalar indices on nested fields may use optimized filters)
        // The index may be used internally even if not shown as ScalarIndexQuery
        assert!(
            plan.contains("ScalarIndexQuery"),
            "Query plan should show optimized read. Plan: {}",
            plan
        );

        // Also test that the query returns results
        let results = dataset
            .scan()
            .filter("metadata.`status.code` = 5")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        // Should have ~100 rows with status code 5
        assert!(results.num_rows() > 0);
        assert_eq!(results.num_rows(), 100);
    }

    #[tokio::test]
    async fn test_inverted_index_on_nested_field_with_dots() {
        use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;

        // Test creating Inverted index on nested text field with dots in the name
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Create schema with nested text field containing dots
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "document",
                DataType::Struct(
                    vec![
                        Field::new("content.text", DataType::Utf8, false),
                        Field::new("content.summary", DataType::Utf8, false),
                    ]
                    .into(),
                ),
                false,
            ),
        ]));

        // Generate test data with text content
        let num_rows = 100;
        let ids = Int32Array::from_iter_values(0..num_rows as i32);
        let content_texts = StringArray::from_iter_values((0..num_rows).map(|i| match i % 3 {
            0 => format!("The quick brown fox jumps over the lazy dog {}", i),
            1 => format!(
                "Machine learning and artificial intelligence document {}",
                i
            ),
            _ => format!("Data science and analytics content piece {}", i),
        }));
        let summaries = StringArray::from_iter_values(
            (0..num_rows).map(|i| format!("Summary of document {}", i)),
        );

        let struct_array = arrow_array::StructArray::from(vec![
            (
                Arc::new(Field::new("content.text", DataType::Utf8, false)),
                Arc::new(content_texts) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("content.summary", DataType::Utf8, false)),
                Arc::new(summaries) as Arc<dyn arrow_array::Array>,
            ),
        ]);

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)])
                .unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Create Inverted index on nested text field with dots
        let nested_column_path = "document.`content.text`";
        let params = InvertedIndexParams::default();

        dataset
            .create_index(
                &[nested_column_path],
                IndexType::Inverted,
                Some("inverted_content_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Reload the dataset to ensure the index is loaded
        dataset = Dataset::open(test_uri).await.unwrap();

        // Verify index was created
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "inverted_content_idx");

        // Verify the correct field was indexed
        let field_id = indices[0].fields[0];
        let field_path = dataset.schema().field_path(field_id).unwrap();
        assert_eq!(field_path, "document.`content.text`");

        // Test full-text search on the nested field with dots
        // Use the field_path that the index reports
        let query = FullTextSearchQuery::new("machine learning".to_string())
            .with_column(field_path.clone())
            .unwrap();

        // Check the query plan uses the inverted index
        let plan = dataset
            .scan()
            .full_text_search(query.clone())
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();

        // Verify the inverted index is being used
        assert!(
            plan.contains("MatchQuery") || plan.contains("PhraseQuery"),
            "Query plan should use inverted index for nested field with dots. Plan: {}",
            plan
        );

        let results = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        // Verify we get results from the full-text search
        assert!(
            !results.is_empty(),
            "Full-text search should return results"
        );

        // Check that we found documents containing "machine learning"
        let mut found_count = 0;
        for batch in results {
            found_count += batch.num_rows();
        }
        // We expect to find approximately 1/3 of documents (those with i % 3 == 1)
        assert!(
            found_count > 0,
            "Should find at least some documents with 'machine learning'"
        );
        assert!(found_count < num_rows, "Should not match all documents");
    }

    #[tokio::test]
    async fn test_resolve_index_column() {
        use lance_datagen::{BatchCount, RowCount, array};

        // Create a test dataset with a vector column
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(32.into()),
            )
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));

        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Create an index with a custom name
        let params = crate::index::vector::VectorIndexParams::ivf_flat(
            4,
            lance_linalg::distance::MetricType::L2,
        );
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("my_vector_index".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload dataset to get the index metadata
        let dataset = Dataset::open(test_uri).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        let index_meta = &indices[0];

        // Test 1: Pass the actual column name
        let (field_path, field) =
            resolve_index_column(dataset.schema(), index_meta, "vector").unwrap();
        assert_eq!(field_path, "vector");
        assert_eq!(field.name, "vector");

        // Test 2: Pass the index name (should resolve to the actual column)
        let (field_path2, field2) =
            resolve_index_column(dataset.schema(), index_meta, "my_vector_index").unwrap();
        assert_eq!(field_path2, "vector");
        assert_eq!(field2.name, "vector");

        // Test 3: Pass a non-existent column name (should fail)
        let result = resolve_index_column(dataset.schema(), index_meta, "nonexistent");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("does not exist in the schema")
        );
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_commits_multiple_segments() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(20), BatchCount::from(2));

        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 10,
                max_rows_per_group: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let field_id = dataset.schema().field("vector").unwrap().id;
        let seg0 = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            field_id,
            Uuid::new_v4(),
            [0_u32],
            b"seg0",
        )
        .await;
        let seg1 = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            field_id,
            Uuid::new_v4(),
            [1_u32],
            b"seg1",
        )
        .await;

        dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&seg0), segment_from_metadata(&seg1)],
            )
            .await
            .unwrap();

        let committed = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(committed.len(), 2);
        let committed_uuids = committed.iter().map(|idx| idx.uuid).collect::<HashSet<_>>();
        assert_eq!(
            committed_uuids,
            HashSet::from([seg0.uuid, seg1.uuid]),
            "all committed segment uuids should be preserved"
        );
        assert_eq!(
            committed
                .iter()
                .map(|idx| idx
                    .fragment_bitmap
                    .as_ref()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>())
                .collect::<HashSet<_>>(),
            HashSet::from([vec![0], vec![1]]),
            "each committed segment should preserve its fragment coverage"
        );
        assert!(
            committed
                .iter()
                .all(|idx| idx.files.as_ref().is_some_and(|files| !files.is_empty())),
            "committed segment metadata should capture on-disk file info"
        );
    }

    /// The logical covering declaration and the physical payload are independent at commit.
    /// Transitional segments may declare a dependency before their writer emits the values,
    /// while physical payload without a declaration remains an unused implementation detail.
    #[tokio::test]
    async fn test_commit_existing_index_segments_preserves_covering_declaration_independently() {
        use crate::index::vector::VectorIndexParams;
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(256), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
        let id_field_id = dataset.schema().field("id").unwrap().id;

        // Direction 1: storage built WITHOUT covering, declaration fabricated.
        let params = VectorIndexParams::ivf_pq(2, 8, 4, lance_linalg::distance::MetricType::L2, 2);
        let mut plain_segment = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vector_idx".to_string())
            .execute_uncommitted()
            .await
            .unwrap();
        assert!(plain_segment.covering_fields.is_empty());
        // Carried columns live at the end of `fields`, so fabricating a declaration means
        // extending both lists, not just `covering_fields`.
        plain_segment.fields.push(id_field_id);
        plain_segment.covering_fields = vec![id_field_id];
        dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&plain_segment)],
            )
            .await
            .unwrap();
        let committed = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(committed[0].covering_fields, vec![id_field_id]);

        // Direction 2: storage built WITH covering, declaration dropped.
        let mut covered_params =
            VectorIndexParams::ivf_pq(2, 8, 4, lance_linalg::distance::MetricType::L2, 2);
        covered_params.covering_columns(vec!["id".to_string()]);
        let mut covered_segment = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &covered_params)
            .name("vector_idx".to_string())
            .replace(true)
            .execute_uncommitted()
            .await
            .unwrap();
        assert_eq!(covered_segment.covering_fields, vec![id_field_id]);
        // Dropping the declaration drops the carried suffix from `fields` too, leaving a
        // segment that claims to key on `vector` alone while its storage still holds `id`.
        covered_segment.fields.pop();
        covered_segment.covering_fields = Vec::new();
        dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&covered_segment)],
            )
            .await
            .unwrap();
        let committed = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert!(committed[0].covering_fields.is_empty());

        // Sanity: the honest covered segment commits fine, and the declaration survives the
        // round trip. That matters beyond bookkeeping -- the segment files still hold the
        // payload, so losing the declaration would silently drop covered queries back to a
        // base-table take.
        covered_segment.fields.push(id_field_id);
        covered_segment.covering_fields = vec![id_field_id];
        dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&covered_segment)],
            )
            .await
            .unwrap();
        let committed = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(committed.len(), 1);
        assert_eq!(committed[0].covering_fields, vec![id_field_id]);
    }

    /// A second commit whose incoming segment covers a different column set than an
    /// existing, disjoint segment we would *retain* must be rejected: all segments of one
    /// logical index must agree on their declaration so they expose one dependency set and
    /// one declaration-ordered query schema. Their physical capabilities may still differ.
    #[tokio::test]
    async fn test_commit_existing_index_segments_rejects_inconsistent_retained_covering() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(20), BatchCount::from(1));

        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 10,
                max_rows_per_group: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let id_field_id = dataset.schema().field("id").unwrap().id;

        // First commit: a real segment covering `id`, over fragment 0 only.
        let mut covered_params = crate::index::vector::VectorIndexParams::ivf_flat(
            1,
            lance_linalg::distance::MetricType::L2,
        );
        covered_params.covering_columns(vec!["id".to_string()]);
        let seg0 = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &covered_params)
            .name("vector_idx".to_string())
            .fragments(vec![0])
            .execute_uncommitted()
            .await
            .unwrap();
        assert_eq!(seg0.covering_fields, vec![id_field_id]);
        dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&seg0)],
            )
            .await
            .unwrap();

        // Second commit: a disjoint segment (fragment 1) that covers *nothing*. It would
        // be retained alongside seg0, whose covering is `[id]` -- an inconsistency.
        let plain_params = crate::index::vector::VectorIndexParams::ivf_flat(
            1,
            lance_linalg::distance::MetricType::L2,
        );
        let seg1 = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &plain_params)
            .name("vector_idx".to_string())
            // The name already exists (seg0 was committed); replace(true) only skips
            // the duplicate-name check for this uncommitted build.
            .replace(true)
            .fragments(vec![1])
            .execute_uncommitted()
            .await
            .unwrap();
        assert!(seg1.covering_fields.is_empty());
        let result = dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&seg1)],
            )
            .await;
        let err =
            result.expect_err("mismatched covering across retained segments must be rejected");
        assert!(
            err.to_string()
                .contains("a logical index cannot mix declarations")
                && err.to_string().contains("covering_fields"),
            "expected a covering-consistency rejection, got: {err}"
        );
    }

    /// When a segment covers a column whose data file was rewritten in place after the
    /// segment was built, `prune_stale_segment_coverage` must drop the affected fragment
    /// from the segment's coverage -- even though the *indexed* column's file is untouched.
    /// Otherwise a covered query reads the segment's stale copy of that column.
    #[tokio::test]
    async fn test_commit_existing_index_segments_prunes_stale_covered_fragment() {
        use crate::dataset::{MergeInsertBuilder, WhenMatched, WhenNotMatched};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let dim = 8i32;
        let n = 20i32;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("tag", DataType::Int32, true),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                false,
            ),
        ]));
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from((0..n * dim).map(|v| v as f32).collect::<Vec<_>>()),
            dim,
        )
        .unwrap();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from((0..n).collect::<Vec<_>>())),
                Arc::new(Int32Array::from((0..n).collect::<Vec<_>>())),
                Arc::new(vectors),
            ],
        )
        .unwrap();
        // Two fragments (ids 0..10 -> frag 0, 10..20 -> frag 1).
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema.clone()),
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 10,
                max_rows_per_group: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // A scalar index on the merge key routes the partial-schema update through the
        // in-place column-rewrite path (RewriteColumns) instead of a row-move, so the
        // covered column's file changes while the fragment id is preserved.
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        let tag_field_id = dataset.schema().field("tag").unwrap().id;

        // Build a real covering segment BEFORE the update, so it is stamped at the
        // pre-update version and covers `tag`, whose files still hold the old values.
        let mut params = crate::index::vector::VectorIndexParams::ivf_flat(
            1,
            lance_linalg::distance::MetricType::L2,
        );
        params.covering_columns(vec!["tag".to_string()]);
        let seg = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vector_idx".to_string())
            .execute_uncommitted()
            .await
            .unwrap();
        assert_eq!(seg.covering_fields, vec![tag_field_id]);

        // In-place rewrite of the covered `tag` column for fragment 0 only (no covering
        // index committed yet, so this patches the column file in place and keeps the
        // fragment id). The indexed `vector` column's file is untouched.
        let sub_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("tag", DataType::Int32, true),
        ]));
        let update = RecordBatch::try_new(
            sub_schema.clone(),
            vec![
                Arc::new(Int32Array::from((0..5).collect::<Vec<_>>())),
                Arc::new(Int32Array::from((100..105).collect::<Vec<_>>())),
            ],
        )
        .unwrap();
        let (updated, _) = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new([Ok(update)], sub_schema)))
            .await
            .unwrap();
        let mut dataset = Arc::try_unwrap(updated).unwrap_or_else(|arc| (*arc).clone());

        dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&seg)],
            )
            .await
            .unwrap();

        let committed = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(committed.len(), 1);
        let bitmap = committed[0].fragment_bitmap.as_ref().unwrap();
        assert!(
            !bitmap.contains(0),
            "fragment 0's covered column was rewritten; its stale coverage must be pruned"
        );
        assert!(
            bitmap.contains(1),
            "fragment 1's covered column was untouched; it stays covered"
        );
    }

    /// A covered *struct* that grows a child (via `add_columns`) after a segment was built
    /// makes the segment's stored covering payload type-incompatible for every covered row --
    /// a global change no per-fragment prune can repair. Since the covering index is not
    /// committed while the segment is staged, the Merge-commit guard never sees the add;
    /// `prune_stale_segment_coverage` must reject the segment commit instead.
    #[tokio::test]
    async fn test_commit_existing_index_segments_rejects_grown_covered_struct() {
        use arrow_array::StructArray;
        use arrow_schema::Fields;
        use lance_file::version::LanceFileVersion;

        use crate::dataset::NewColumnTransform;

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let dim = 8i32;
        let n = 10i32;
        let meta_fields = Fields::from(vec![Field::new("a", DataType::Int32, false)]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("meta", DataType::Struct(meta_fields.clone()), true),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                false,
            ),
        ]));
        let a = Arc::new(Int32Array::from((0..n).collect::<Vec<_>>()));
        let meta = Arc::new(StructArray::new(meta_fields, vec![a], None));
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from((0..n * dim).map(|v| v as f32).collect::<Vec<_>>()),
            dim,
        )
        .unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![meta, Arc::new(vectors)]).unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema.clone()),
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let meta_field_id = dataset.schema().field("meta").unwrap().id;

        // Build a real covering segment stamped at the current (pre-add) version,
        // covering the struct column `meta`.
        let mut params = crate::index::vector::VectorIndexParams::ivf_flat(
            1,
            lance_linalg::distance::MetricType::L2,
        );
        params.covering_columns(vec!["meta".to_string()]);
        let seg = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vector_idx".to_string())
            .execute_uncommitted()
            .await
            .unwrap();
        assert_eq!(seg.covering_fields, vec![meta_field_id]);

        // Grow the covered struct AFTER the segment was built. No covering index is
        // committed yet, so the Merge-commit guard does not fire and the add succeeds.
        let add =
            NewColumnTransform::AllNulls(Arc::new(arrow_schema::Schema::new(vec![Field::new(
                "meta",
                DataType::Struct(Fields::from(vec![Field::new("b", DataType::Int32, true)])),
                true,
            )])));
        dataset.add_columns(add, None, None).await.unwrap();

        // Committing the now-stale segment must be rejected: its stored covering payload
        // still holds the pre-child struct type.
        let err = dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&seg)],
            )
            .await
            .expect_err("committing a segment covering a since-grown struct must be rejected");
        assert!(
            err.to_string().contains("changed its subtree"),
            "expected a covered-subtree-change rejection, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_rejects_duplicate_segment_ids() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));

        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let field_id = dataset.schema().field("vector").unwrap().id;
        let base = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            field_id,
            Uuid::new_v4(),
            [0_u32],
            b"base",
        )
        .await;

        let err = dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![
                    segment_from_metadata(&base),
                    segment_from_metadata(&IndexMetadata {
                        fragment_bitmap: Some(std::iter::once(1_u32).collect()),
                        ..base
                    }),
                ],
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("duplicate segment uuid"));
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_rejects_empty_segments() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));

        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let err = dataset
            .commit_existing_index_segments("vector_idx", "vector", Vec::<IndexSegment>::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("at least one index segment"));
    }

    /// A segment may carry extra columns; `build_index_metadata_from_segments`
    /// must still commit it, keyed on the field the index is being built for.
    /// Calls the guarded function directly (not through
    /// `commit_existing_index_segments`) so this test's outcome depends only
    /// on this guard, not on the independent duplicate a few lines into
    /// `commit_existing_index_segments`.
    #[tokio::test]
    async fn test_build_index_metadata_from_segments_accepts_carried_fields() {
        use crate::index::vector::VectorIndexParams;
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(256), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

        let vector_field_id = dataset.schema().field("vector").unwrap().id;
        let id_field_id = dataset.schema().field("id").unwrap().id;
        // A genuinely covered segment, not a hand-edited declaration: the commit boundary
        // now cross-checks `covering_fields` against the payload in the segment's auxiliary
        // storage, so `id` has to really be there.
        let mut params =
            VectorIndexParams::ivf_pq(2, 8, 4, lance_linalg::distance::MetricType::L2, 2);
        params.covering_columns(vec!["id".to_string()]);
        let metadata = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vector_idx".to_string())
            .execute_uncommitted()
            .await
            .unwrap();
        // Carried `id` sits at the end of `fields`, behind the keyed `vector` field.
        assert_eq!(metadata.fields, vec![vector_field_id, id_field_id]);
        assert_eq!(metadata.covering_fields, vec![id_field_id]);

        let new_indices = build_index_metadata_from_segments(
            &dataset,
            "vector_idx",
            vector_field_id,
            vec![segment_from_metadata(&metadata)],
        )
        .await
        .unwrap();

        assert_eq!(new_indices.len(), 1);
        assert_eq!(new_indices[0].fields, vec![vector_field_id, id_field_id]);
        assert_eq!(new_indices[0].covering_fields, vec![id_field_id]);
    }

    /// A segment's carried columns can go stale independently of its keyed column,
    /// so the staleness check has to walk every entry of `fields`, not just the
    /// keyed subtree. Here an in-place update rewrites the carried column's data
    /// file after the segment was built, so the segment's stored copy is stale and
    /// its coverage of that fragment must be pruned. Walking only the keyed subtree
    /// sees no change (the vector file is untouched) and leaves the fragment
    /// covered, which is the bug. Both segments are genuinely built (the commit
    /// boundary cross-checks a covering declaration against the auxiliary storage,
    /// so hand-doctored metadata cannot reach the prune).
    #[tokio::test]
    async fn test_prune_stale_coverage_notices_a_changed_carried_column() {
        use crate::index::vector::VectorIndexParams;
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col("payload", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(256), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

        let vector_field_id = dataset.schema().field("vector").unwrap().id;

        // A real covered segment carrying `payload`, and a plain control segment,
        // both built against the current file layout.
        let mut covered_params =
            VectorIndexParams::ivf_pq(2, 8, 4, lance_linalg::distance::MetricType::L2, 2);
        covered_params.covering_columns(vec!["payload".to_string()]);
        let covered = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &covered_params)
            .name("vector_idx".to_string())
            .execute_uncommitted()
            .await
            .unwrap();
        let plain_params =
            VectorIndexParams::ivf_pq(2, 8, 4, lance_linalg::distance::MetricType::L2, 2);
        let plain = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &plain_params)
            .name("vector_idx".to_string())
            .execute_uncommitted()
            .await
            .unwrap();

        // Rewrite `payload`'s backing file IN PLACE (same fragment id, new data
        // file for `payload` only); `vector`'s file is untouched. A row-moving
        // update would replace the fragment and prune BOTH segments' coverage,
        // which the control below would catch.
        let update_schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Int32, false),
            arrow_schema::Field::new("payload", arrow_schema::DataType::Int32, false),
        ]));
        let update_batch = RecordBatch::try_new(
            update_schema.clone(),
            vec![
                Arc::new(arrow_array::Int32Array::from(vec![2, 4, 6])),
                Arc::new(arrow_array::Int32Array::from(vec![99, 99, 99])),
            ],
        )
        .unwrap();
        let right: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(arrow_array::RecordBatchIterator::new(
                vec![Ok(update_batch)].into_iter(),
                update_schema,
            ));
        let mut frag0 = dataset.get_fragment(0).unwrap();
        let (updated_fragment, fields_modified) =
            frag0.update_columns(right, "id", "id").await.unwrap();
        let op = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![updated_fragment],
            new_fragments: vec![],
            fields_modified,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(crate::dataset::transaction::UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        };
        let read_version = dataset.manifest.version;
        let mut dataset = dataset;
        dataset
            .apply_commit(
                Transaction::new(read_version, op, None),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();

        let new_indices = build_index_metadata_from_segments(
            &dataset,
            "vector_idx",
            vector_field_id,
            vec![segment_from_metadata(&covered)],
        )
        .await
        .unwrap();

        assert_eq!(new_indices.len(), 1);
        assert!(
            new_indices[0].fragment_bitmap.as_ref().unwrap().is_empty(),
            "coverage of a fragment whose carried column changed must be pruned, got {:?}",
            new_indices[0].fragment_bitmap
        );

        // Control: with nothing carried, the same segment over the same fragment
        // keeps its coverage -- so the assertion above is about the carried column,
        // not about the update invalidating everything.
        let plain_indices = build_index_metadata_from_segments(
            &dataset,
            "vector_idx",
            vector_field_id,
            vec![segment_from_metadata(&plain)],
        )
        .await
        .unwrap();
        assert!(
            !plain_indices[0]
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .is_empty(),
            "a non-covering segment must keep its coverage across the same update"
        );
    }

    /// The per-segment rules only pin the keyed field and its count, so two
    /// segments can disagree about the carried columns and each still pass. That
    /// pair must be refused at commit, because `IndexDescriptionImpl::try_new`
    /// requires `fields` to be identical across segments -- committing it would
    /// leave `describe_indices` erroring on metadata this call just wrote.
    #[tokio::test]
    async fn test_build_index_metadata_from_segments_rejects_mixed_covering_declarations() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

        let vector_field_id = dataset.schema().field("vector").unwrap().id;
        let id_field_id = dataset.schema().field("id").unwrap().id;

        let mut covered = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            vector_field_id,
            Uuid::new_v4(),
            [0_u32],
            b"covered",
        )
        .await;
        covered.fields = vec![vector_field_id, id_field_id];
        covered.covering_fields = vec![id_field_id];

        // Plain: one keyed field, nothing carried. Passes every per-segment rule
        // on its own -- `keyed == 1` and `fields[0]` is the keyed field -- and so
        // does `covered`; only comparing them catches the disagreement.
        let plain = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            vector_field_id,
            Uuid::new_v4(),
            [1_u32],
            b"plain",
        )
        .await;
        assert_eq!(plain.fields, vec![vector_field_id]);
        assert!(plain.covering_fields.is_empty());

        let err = build_index_metadata_from_segments(
            &dataset,
            "vector_idx",
            vector_field_id,
            vec![
                segment_from_metadata(&covered),
                segment_from_metadata(&plain),
            ],
        )
        .await
        .expect_err("segments disagreeing about carried columns must not commit");
        assert!(
            err.to_string().contains("must declare the same columns"),
            "unexpected message: {err}"
        );
    }

    /// `fields` lists the carried columns too, but a logical description must
    /// advertise only what the index can be searched on. Every binding reads this
    /// list as "the index's columns" -- Python's `_default_vector_index_for_column`
    /// matches on membership -- so a carried column here hands back the keyed
    /// column's model for a query about the carried one.
    #[tokio::test]
    async fn test_index_description_reports_only_keyed_fields() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

        let vector_field_id = dataset.schema().field("vector").unwrap().id;
        let id_field_id = dataset.schema().field("id").unwrap().id;
        let mut metadata = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            vector_field_id,
            Uuid::new_v4(),
            [0_u32],
            b"segment",
        )
        .await;
        metadata.fields = vec![vector_field_id, id_field_id];
        metadata.covering_fields = vec![id_field_id];

        let description = IndexDescriptionImpl::try_new(vec![metadata], &dataset)
            .await
            .unwrap();

        assert_eq!(
            description.field_ids(),
            &[vector_field_id as u32],
            "a covered index must advertise only its keyed column"
        );
    }

    /// `covering_fields` is a format-level declaration shared by every index type. A
    /// multi-field `fields` with a carried suffix must therefore remain legal even when
    /// that index implementation has no physical covering capability yet.
    /// A covered SCALAR index must be withdrawn by remap, not remapped.
    ///
    /// `remap_index` stopped withdrawing covered indexes on the grounds that "each storage
    /// format's `remap` preserves its covering columns row-aligned". That holds for the four
    /// vector storages; `BTreeIndex::remap` rewrites only `(value, row_id)`. Remapping one
    /// republishes a declaration nothing backs, and because carried fields live in `fields`
    /// the index then loses fragment coverage on every update to a column it never held.
    #[tokio::test]
    async fn test_remap_withdraws_a_covered_scalar_index() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col("other", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(64), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

        let id_field_id = dataset.schema().field("id").unwrap().id;
        let other_field_id = dataset.schema().field("other").unwrap().id;

        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".into()),
                &btree_params,
                true,
            )
            .await
            .unwrap();

        // Actually COMMIT the covered declaration -- mutating a loaded copy proves nothing,
        // because `remap_index` re-loads the indices from the dataset itself.
        crate::utils::test::covering::declare_covering(&mut dataset, "id", "other").await;
        let indices = dataset.load_indices().await.unwrap();
        let committed = indices.iter().find(|i| i.name == "id_idx").unwrap();
        assert_eq!(
            committed.covering_fields,
            vec![other_field_id],
            "precondition: the covered declaration must be committed, or this test is vacuous"
        );
        assert_eq!(committed.fields, vec![id_field_id, other_field_id]);
        let index_id = committed.uuid;

        let result = remap_index(&dataset, &index_id, &RowAddrRemap::empty()).await;
        assert!(
            matches!(result, Ok(RemapResult::Drop)),
            "a covered scalar index must be withdrawn rather than remapped without its \
             payload; got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_build_index_metadata_from_segments_accepts_carried_fields_on_scalar() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col("other", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(64), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

        let id_field_id = dataset.schema().field("id").unwrap().id;
        let other_field_id = dataset.schema().field("other").unwrap().id;

        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        let mut metadata = dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .name("id_idx".to_string())
            .execute_uncommitted()
            .await
            .unwrap();
        // No producer builds covered scalar indexes (`covering_columns` lives only on
        // `VectorIndexParams`), so the declaration is hand-constructed on a real BTree
        // segment. That is the shape a distributed caller can hand to the public
        // `commit_existing_index_segments`, which is what this guard has to tolerate.
        metadata.fields = vec![id_field_id, other_field_id];
        metadata.covering_fields = vec![other_field_id];

        let new_indices = build_index_metadata_from_segments(
            &dataset,
            "id_idx",
            id_field_id,
            vec![segment_from_metadata(&metadata)],
        )
        .await
        .expect("a carried-field declaration must not require physical payload at commit");

        assert_eq!(new_indices.len(), 1);
        assert_eq!(new_indices[0].fields, vec![id_field_id, other_field_id]);
        assert_eq!(new_indices[0].covering_fields, vec![other_field_id]);
    }

    /// A segment keyed on the wrong field must still be rejected. Calls
    /// `build_index_metadata_from_segments` directly so the assertion is
    /// evidence for *this* guard specifically -- going through
    /// `commit_existing_index_segments` would let the untouched-looking
    /// duplicate check there mask this guard's removal.
    #[tokio::test]
    async fn test_build_index_metadata_from_segments_rejects_wrong_field_provenance() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

        let vector_field_id = dataset.schema().field("vector").unwrap().id;
        let id_field_id = dataset.schema().field("id").unwrap().id;
        let mut metadata = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            vector_field_id,
            Uuid::new_v4(),
            [0_u32],
            b"segment",
        )
        .await;
        metadata.fields = vec![id_field_id];

        let error = build_index_metadata_from_segments(
            &dataset,
            "vector_idx",
            vector_field_id,
            vec![segment_from_metadata(&metadata)],
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("expected keyed field"));
    }

    /// Matching cardinality is not enough: `covering_fields` must actually be
    /// the trailing slice of `fields`. `fields = [vector, id]`,
    /// `covering_fields = [999]` has exactly one keyed field but carries a
    /// column that is not even present in `fields`, let alone its suffix.
    #[tokio::test]
    async fn test_build_index_metadata_from_segments_rejects_non_suffix_covering_fields() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

        let vector_field_id = dataset.schema().field("vector").unwrap().id;
        let id_field_id = dataset.schema().field("id").unwrap().id;
        let mut metadata = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            vector_field_id,
            Uuid::new_v4(),
            [0_u32],
            b"segment",
        )
        .await;
        metadata.fields = vec![vector_field_id, id_field_id];
        metadata.covering_fields = vec![999];

        let error = build_index_metadata_from_segments(
            &dataset,
            "vector_idx",
            vector_field_id,
            vec![segment_from_metadata(&metadata)],
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("are not among its fields"));
    }

    /// Two segments passed to a single `build_index_metadata_from_segments`
    /// call must agree on `covering_fields`: the read path (`knn.rs`,
    /// `scanner.rs`) derives its declared exec output schema from a single
    /// segment and assumes every sibling of the same logical index agrees.
    /// Each segment individually satisfies the per-segment suffix rule
    /// (`validate_covering_fields`), so only a cross-segment comparison
    /// catches the disagreement. Calls `build_index_metadata_from_segments`
    /// directly, not through `commit_existing_index_segments`, so the
    /// assertion is evidence for this guard specifically.
    #[tokio::test]
    async fn test_build_index_metadata_from_segments_rejects_covering_fields_disagreement() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

        let vector_field_id = dataset.schema().field("vector").unwrap().id;
        let id_field_id = dataset.schema().field("id").unwrap().id;

        let mut covered = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            vector_field_id,
            Uuid::new_v4(),
            [0_u32],
            b"seg0",
        )
        .await;
        covered.fields = vec![vector_field_id, id_field_id];
        covered.covering_fields = vec![id_field_id];

        let uncovered = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            vector_field_id,
            Uuid::new_v4(),
            [1_u32],
            b"seg1",
        )
        .await;

        let error = build_index_metadata_from_segments(
            &dataset,
            "vector_idx",
            vector_field_id,
            vec![
                segment_from_metadata(&covered),
                segment_from_metadata(&uncovered),
            ],
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("every segment of one index must declare the same columns"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_rejects_wrong_field_provenance() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

        let vector_field_id = dataset.schema().field("vector").unwrap().id;
        let mut metadata = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            vector_field_id,
            Uuid::new_v4(),
            [0_u32],
            b"segment",
        )
        .await;
        metadata.fields = vec![dataset.schema().field("id").unwrap().id];

        let error = dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&metadata)],
            )
            .await
            .unwrap_err();
        // This end-to-end path is guarded twice (`build_index_metadata_from_segments`
        // and a duplicate check in `commit_existing_index_segments` itself), so
        // this assertion alone does not prove which one fired -- see
        // `test_build_index_metadata_from_segments_rejects_wrong_field_provenance`
        // for that. It still proves the public `commit_existing_index_segments`
        // entry point rejects this input end-to-end.
        assert!(error.to_string().contains("expected keyed field"));
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_rejects_future_dataset_version() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

        let field_id = dataset.schema().field("vector").unwrap().id;
        let mut metadata = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            field_id,
            Uuid::new_v4(),
            [0_u32],
            b"segment",
        )
        .await;
        metadata.dataset_version = dataset.manifest.version + 1;

        let error = dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&metadata)],
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("future dataset version"));
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_rejects_overlapping_fragment_coverage() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(20), BatchCount::from(2));

        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let field_id = dataset.schema().field("vector").unwrap().id;
        let seg0 = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            field_id,
            Uuid::new_v4(),
            [0_u32, 1_u32],
            b"seg0",
        )
        .await;
        let seg1 = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            field_id,
            Uuid::new_v4(),
            [1_u32],
            b"seg1",
        )
        .await;

        let err = dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&seg0), segment_from_metadata(&seg1)],
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("overlapping fragment coverage"));
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_rejects_mixed_index_detail_types() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(20), BatchCount::from(2));

        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let field_id = dataset.schema().field("vector").unwrap().id;
        let seg0 = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            field_id,
            Uuid::new_v4(),
            [0_u32],
            b"seg0",
        )
        .await;
        let seg1 = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "vector_idx".to_string(),
            fields: vec![field_id],
            covering_fields: vec![],
            dataset_version: dataset.manifest.version,
            fragment_bitmap: Some(std::iter::once(1_u32).collect()),
            index_details: Some(Arc::new(
                prost_types::Any::from_msg(&BTreeIndexDetails::default()).unwrap(),
            )),
            index_version: IndexType::BTree.version(),
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            files: seg0.files.clone(),
        };

        let err = dataset
            .commit_existing_index_segments("vector_idx", "vector", vec![seg0, seg1])
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("mixes incompatible index detail types")
        );
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_rejects_partial_replacement_of_wider_segment() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(20), BatchCount::from(2));

        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 20,
                max_rows_per_group: 20,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);

        let field_id = dataset.schema().field("vector").unwrap().id;
        let original = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            field_id,
            Uuid::new_v4(),
            [0_u32, 1_u32],
            b"original",
        )
        .await;
        dataset
            .commit_existing_index_segments("vector_idx", "vector", vec![original])
            .await
            .unwrap();

        let replacement = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            field_id,
            Uuid::new_v4(),
            [0_u32],
            b"replacement",
        )
        .await;

        let err = dataset
            .commit_existing_index_segments("vector_idx", "vector", vec![replacement])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("would orphan fragments"));
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_replaces_different_index_type() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

        let index_name = "shared_name";
        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        let original = dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .name("staged_btree".to_string())
            .execute_uncommitted()
            .await
            .unwrap();
        dataset
            .commit_existing_index_segments(index_name, "id", vec![original])
            .await
            .unwrap();

        let bitmap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
        let replacement = dataset
            .create_index_builder(&["id"], IndexType::Bitmap, &bitmap_params)
            .name("staged_bitmap".to_string())
            .execute_uncommitted()
            .await
            .unwrap();
        let replacement_uuid = replacement.uuid;
        let replacement_type_url = replacement.index_details.as_ref().unwrap().type_url.clone();

        dataset
            .commit_existing_index_segments(index_name, "id", vec![replacement])
            .await
            .unwrap();

        let committed = dataset
            .load_index_by_name(index_name)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(committed.uuid, replacement_uuid);
        assert_eq!(
            committed.index_details.unwrap().type_url,
            replacement_type_url
        );
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_rejects_partial_index_type_change() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(20), BatchCount::from(2));
        let mut dataset = Dataset::write(
            reader,
            test_dir.path().to_str().unwrap(),
            Some(WriteParams {
                max_rows_per_file: 20,
                max_rows_per_group: 20,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let index_name = "shared_name";
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        let mut original_segments = Vec::with_capacity(fragments.len());
        for fragment in &fragments {
            original_segments.push(
                dataset
                    .create_index_builder(&["id"], IndexType::BTree, &btree_params)
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        dataset
            .commit_existing_index_segments(index_name, "id", original_segments)
            .await
            .unwrap();
        let committed_version = dataset.manifest.version;

        let bitmap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
        let replacement = dataset
            .create_index_builder(&["id"], IndexType::Bitmap, &bitmap_params)
            .fragments(vec![fragments[0].id() as u32])
            .execute_uncommitted()
            .await
            .unwrap();
        let error = dataset
            .commit_existing_index_segments(index_name, "id", vec![replacement])
            .await
            .unwrap_err();

        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected invalid input error, got: {error}"
        );
        assert!(
            error
                .to_string()
                .contains("cannot change index 'shared_name'")
        );
        assert!(error.to_string().contains("missing current fragments [1]"));
        assert_eq!(dataset.manifest.version, committed_version);

        let committed =
            crate::index::scalar_logical::load_named_scalar_segments(&dataset, "id", index_name)
                .await
                .unwrap();
        assert_eq!(committed.len(), 2);
        assert!(committed.iter().all(|segment| {
            segment
                .index_details
                .as_ref()
                .is_some_and(|details| details.type_url.ends_with("BTreeIndexDetails"))
        }));
    }

    /// A covered index still names the same column by its keyed prefix. The
    /// name-collision guard in `commit_existing_index_segments` must recognize
    /// that, not reject on the full `fields` vector including the carried
    /// column -- which would misfire "already exists with different fields"
    /// before ever reaching the deeper type-compatibility check this test
    /// otherwise shares with
    /// `test_commit_existing_index_segments_rejects_partial_index_type_change`.
    #[tokio::test]
    async fn test_commit_existing_index_segments_recognizes_covered_index_by_name() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col("extra", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(20), BatchCount::from(2));
        let mut dataset = Dataset::write(
            reader,
            test_dir.path().to_str().unwrap(),
            Some(WriteParams {
                max_rows_per_file: 20,
                max_rows_per_group: 20,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let index_name = "shared_name";
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        let mut original_segments = Vec::with_capacity(fragments.len());
        for fragment in &fragments {
            original_segments.push(
                dataset
                    .create_index_builder(&["id"], IndexType::BTree, &btree_params)
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        dataset
            .commit_existing_index_segments(index_name, "id", original_segments)
            .await
            .unwrap();

        // Hand-declare the committed index as covering an extra carried column
        // -- there is no producer in this phase, so this is done directly.
        let id_field = dataset.schema().field("id").unwrap().id;
        let extra_field = dataset.schema().field("extra").unwrap().id;
        let current = dataset.load_indices_by_name(index_name).await.unwrap();
        let covered_segments = current
            .iter()
            .cloned()
            .map(|mut idx| {
                idx.fields = vec![id_field, extra_field];
                idx.covering_fields = vec![extra_field];
                idx
            })
            .collect::<Vec<_>>();
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: covered_segments,
                removed_indices: current.to_vec(),
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let bitmap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
        let replacement = dataset
            .create_index_builder(&["id"], IndexType::Bitmap, &bitmap_params)
            .fragments(vec![fragments[0].id() as u32])
            .execute_uncommitted()
            .await
            .unwrap();
        let error = dataset
            .commit_existing_index_segments(index_name, "id", vec![replacement])
            .await
            .unwrap_err();

        // Must reach the deeper type-compatibility error, not misfire the
        // shallow "already exists with different fields" guard this test
        // targets.
        assert!(
            error
                .to_string()
                .contains("cannot change index 'shared_name'"),
            "{error}"
        );
        assert!(
            error.to_string().contains("missing current fragments"),
            "{error}"
        );
        assert!(
            !error
                .to_string()
                .contains("already exists with different fields"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn test_partial_type_change_with_legacy_missing_details_is_rejected() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let index_name = "shared_name";
        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        let original = dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .execute_uncommitted()
            .await
            .unwrap();
        dataset
            .commit_existing_index_segments(index_name, "id", vec![original])
            .await
            .unwrap();

        let current = dataset.load_indices_by_name(index_name).await.unwrap();
        assert_eq!(current.len(), 1);
        let original_uuid = current[0].uuid;
        let mut legacy = current.clone();
        legacy[0].index_details = None;
        legacy[0].index_version = 0;
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: legacy,
                removed_indices: current,
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let append_reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let mut dataset = Dataset::write(
            append_reader,
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);

        let bitmap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
        let replacement = dataset
            .create_index_builder(&["id"], IndexType::Bitmap, &bitmap_params)
            .fragments(vec![fragments[1].id() as u32])
            .execute_uncommitted()
            .await
            .unwrap();
        let version_before = dataset.manifest.version;
        let error = dataset
            .commit_existing_index_segments(index_name, "id", vec![replacement])
            .await
            .unwrap_err();

        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected invalid input error, got: {error}"
        );
        assert!(error.to_string().contains("from type '<unknown>'"));
        assert!(error.to_string().contains("partial fragment coverage"));
        assert!(error.to_string().contains("missing current fragments [0]"));
        assert_eq!(dataset.manifest.version, version_before);

        let committed = dataset.load_indices_by_name(index_name).await.unwrap();
        assert_eq!(committed.len(), 1);
        assert_eq!(committed[0].uuid, original_uuid);
        assert!(committed[0].index_details.is_none());
        assert_eq!(committed[0].index_version, 0);

        let field_id = dataset.schema().field("id").unwrap().id;
        let sentinel_segment = IndexSegment::new(
            Uuid::new_v4(),
            [fragments[1].id() as u32],
            [field_id],
            Arc::new(prost_types::Any {
                type_url: "<unknown>".to_string(),
                value: Vec::new(),
            }),
            0,
            dataset.manifest.version,
            vec![],
        );
        let error = dataset
            .commit_existing_index_segments(index_name, "id", vec![sentinel_segment])
            .await
            .unwrap_err();

        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected invalid input error, got: {error}"
        );
        assert!(error.to_string().contains("from type '<unknown>'"));
        assert!(error.to_string().contains("to type '<unknown>'"));
        assert!(error.to_string().contains("missing current fragments [0]"));
        assert_eq!(dataset.manifest.version, version_before);

        let committed = dataset.load_indices_by_name(index_name).await.unwrap();
        assert_eq!(committed.len(), 1);
        assert_eq!(committed[0].uuid, original_uuid);
        assert!(committed[0].index_details.is_none());
        assert_eq!(committed[0].index_version, 0);
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_removes_empty_segment() {
        use lance_datagen::{BatchCount, RowCount, array};

        let test_dir = tempfile::tempdir().unwrap();
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
            )
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();
        let field_id = dataset.schema().field("vector").unwrap().id;
        let uuid = Uuid::new_v4();

        // Commit a 0-fragment segment, then a real segment covering the dataset.
        let empty = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            field_id,
            uuid,
            std::iter::empty::<u32>(),
            b"empty",
        )
        .await;
        dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&empty)],
            )
            .await
            .unwrap();
        let seg = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            field_id,
            Uuid::new_v4(),
            [0_u32],
            b"seg",
        )
        .await;
        dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&seg)],
            )
            .await
            .unwrap();

        // The real segment covers the dataset, so the redundant empty one is removed.
        let committed = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(
            committed.iter().map(|i| i.uuid).collect::<HashSet<_>>(),
            HashSet::from([seg.uuid]),
            "empty segment should be removed once a real segment covers the dataset",
        );
    }

    #[tokio::test]
    async fn test_resolve_index_column_error_cases() {
        use lance_datagen::{BatchCount, RowCount, array};

        // Create a test dataset
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(32.into()),
            )
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));

        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Create an index
        let params = crate::index::vector::VectorIndexParams::ivf_flat(
            4,
            lance_linalg::distance::MetricType::L2,
        );
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("my_index".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload dataset
        let dataset = Dataset::open(test_uri).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let index_meta = &indices[0];

        // Test: Pass a column that doesn't exist and is not the index name
        let result = resolve_index_column(dataset.schema(), index_meta, "nonexistent_column");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("does not exist in the schema"),
            "Error message should mention column doesn't exist, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_resolve_index_column_nested_field() {
        use arrow_array::{RecordBatch, StructArray};
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};

        // Create a test dataset with nested struct manually
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        // Create schema with nested structure: data.vector
        let vector_field = ArrowField::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                8,
            ),
            false,
        );
        let struct_field = ArrowField::new(
            "data",
            DataType::Struct(vec![vector_field.clone()].into()),
            false,
        );
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            struct_field,
        ]));

        // Create data
        let id_array = arrow_array::Int32Array::from(vec![1, 2, 3, 4, 5]);

        // Create nested vector data
        let mut vector_values = Vec::new();
        for _ in 0..5 {
            for _ in 0..8 {
                vector_values.push(rand::random::<f32>());
            }
        }
        let vector_array = arrow_array::FixedSizeListArray::try_new_from_values(
            arrow_array::Float32Array::from(vector_values),
            8,
        )
        .unwrap();

        let struct_array = StructArray::from(vec![(
            Arc::new(vector_field),
            Arc::new(vector_array) as arrow_array::ArrayRef,
        )]);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(id_array), Arc::new(struct_array)],
        )
        .unwrap();

        let reader = Box::new(arrow_array::RecordBatchIterator::new(
            vec![Ok(batch)],
            schema,
        ));

        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Create an index on the nested field
        let params = crate::index::vector::VectorIndexParams::ivf_flat(
            2,
            lance_linalg::distance::MetricType::L2,
        );
        dataset
            .create_index(
                &["data.vector"],
                IndexType::Vector,
                Some("nested_vector_index".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload dataset to get the index metadata
        let dataset = Dataset::open(test_uri).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        let index_meta = &indices[0];

        // Test 1: Pass the nested field path directly
        let (field_path, field) =
            resolve_index_column(dataset.schema(), index_meta, "data.vector").unwrap();
        assert_eq!(field_path, "data.vector");
        assert_eq!(field.name, "vector");

        // Test 2: Pass the index name, should resolve to the nested field path
        let (field_path2, field2) =
            resolve_index_column(dataset.schema(), index_meta, "nested_vector_index").unwrap();
        assert_eq!(field_path2, "data.vector");
        assert_eq!(field2.name, "vector");

        // Verify the field path is correct for nested access
        assert!(
            field_path2.contains('.'),
            "Field path should contain '.' for nested field"
        );
    }

    #[tokio::test]
    async fn test_scalar_index_file_sizes_captured() {
        // Test that file sizes are captured when creating a scalar index
        let reader = gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col("values", array::rand_utf8(ByteCount::from(10), false))
            .into_reader_rows(RowCount::from(4), BatchCount::from(1));

        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        // Create a scalar index
        dataset
            .create_index(
                &["values"],
                IndexType::Scalar,
                Some("test_idx".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Get index metadata and verify files are populated
        let indices = dataset.load_indices().await.unwrap();
        let test_index = indices.iter().find(|idx| idx.name == "test_idx").unwrap();

        assert!(
            test_index.files.is_some(),
            "Index should have files populated"
        );
        let files = test_index.files.as_ref().unwrap();
        assert!(!files.is_empty(), "Index should have at least one file");

        // Verify each file has a positive size
        for file in files {
            assert!(
                file.size_bytes > 0,
                "File {} should have positive size",
                file.path
            );
        }

        // Verify total_size_bytes works
        let total_size = test_index.total_size_bytes();
        assert!(total_size.is_some(), "total_size_bytes should return Some");
        assert!(total_size.unwrap() > 0, "Total size should be positive");
    }

    #[tokio::test]
    async fn test_vector_index_file_sizes_captured() {
        // Test that file sizes are captured when creating a vector index
        let reader = gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(4.into()),
            )
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));

        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        // Create vector index
        let params = VectorIndexParams::ivf_pq(1, 8, 2, MetricType::L2, 2);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("test_vec_idx".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Get index metadata and verify files are populated
        let indices = dataset.load_indices().await.unwrap();
        let test_index = indices
            .iter()
            .find(|idx| idx.name == "test_vec_idx")
            .unwrap();

        assert!(
            test_index.files.is_some(),
            "Index should have files populated"
        );
        let files = test_index.files.as_ref().unwrap();
        assert!(!files.is_empty(), "Index should have at least one file");

        // Verify each file has a positive size
        for file in files {
            assert!(
                file.size_bytes > 0,
                "File {} should have positive size",
                file.path
            );
        }

        // Verify total_size_bytes works
        let total_size = test_index.total_size_bytes();
        assert!(total_size.is_some(), "total_size_bytes should return Some");
        assert!(total_size.unwrap() > 0, "Total size should be positive");
    }

    #[tokio::test]
    async fn test_describe_indices_total_size() {
        // Test that describe_indices returns total_size_bytes
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("values", DataType::Utf8, false),
        ]));

        let values = StringArray::from_iter_values(["hello", "world", "foo", "bar"]);
        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..4)),
                Arc::new(values),
            ],
        )
        .unwrap();

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        // Create a scalar index
        dataset
            .create_index(
                &["values"],
                IndexType::Scalar,
                Some("test_idx".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Use describe_indices to get index info
        let descriptions = dataset.describe_indices(None).await.unwrap();
        assert_eq!(descriptions.len(), 1);

        let desc = &descriptions[0];
        assert_eq!(desc.name(), "test_idx");
        assert_eq!(desc.rows_indexed(), 4);

        // Verify total_size_bytes is available
        let total_size = desc.total_size_bytes();
        assert!(total_size.is_some(), "total_size_bytes should be Some");
        assert!(total_size.unwrap() > 0, "Total size should be positive");
    }

    #[tokio::test]
    async fn test_describe_indices_rows_indexed_multi_fragment() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("values", array::rand_utf8(ByteCount::from(8), false))
            .into_reader_rows(RowCount::from(10), BatchCount::from(3));

        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 10,
                max_rows_per_group: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments().len(), 3);

        dataset
            .create_index(
                &["values"],
                IndexType::Scalar,
                Some("multi_frag_idx".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        let descriptions = dataset.describe_indices(None).await.unwrap();
        assert_eq!(descriptions.len(), 1);
        assert_eq!(descriptions[0].name(), "multi_frag_idx");
        assert_eq!(
            descriptions[0].rows_indexed(),
            30,
            "rows_indexed should sum logical rows across all indexed fragments"
        );
    }

    #[tokio::test]
    async fn test_describe_indices_rows_indexed_with_deletions() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("values", array::rand_utf8(ByteCount::from(8), false))
            .into_reader_rows(RowCount::from(20), BatchCount::from(1));

        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
        dataset.delete("id >= 15").await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 15);

        dataset
            .create_index(
                &["values"],
                IndexType::Scalar,
                Some("deleted_rows_idx".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        let descriptions = dataset.describe_indices(None).await.unwrap();
        assert_eq!(descriptions.len(), 1);
        assert_eq!(descriptions[0].name(), "deleted_rows_idx");
        assert_eq!(
            descriptions[0].rows_indexed(),
            15,
            "rows_indexed should use logical rows (physical_rows - num_deleted_rows)"
        );
    }

    #[tokio::test]
    async fn test_describe_indices_rows_indexed_stale_bitmap_fragment() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(8.into()))
            .into_reader_rows(RowCount::from(10), BatchCount::from(2));

        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 10,
                max_rows_per_group: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments().len(), 2);

        let field_id = dataset.schema().field("vector").unwrap().id;
        let stale_segment = write_vector_segment_metadata(
            &dataset,
            "vector_idx",
            field_id,
            Uuid::new_v4(),
            [0_u32, 1_u32, 999_u32],
            b"stale-bitmap-segment",
        )
        .await;

        dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![segment_from_metadata(&stale_segment)],
            )
            .await
            .unwrap();

        let descriptions = dataset.describe_indices(None).await.unwrap();
        assert_eq!(descriptions.len(), 1);
        assert_eq!(descriptions[0].name(), "vector_idx");
        assert_eq!(
            descriptions[0].rows_indexed(),
            20,
            "stale bitmap entries for missing fragments should be skipped without failing"
        );
    }

    /// Helper to assert that all indices have file sizes populated
    async fn assert_all_indices_have_files(dataset: &Dataset, context: &str) {
        let indices = dataset.load_indices().await.unwrap();
        for index in indices.iter() {
            // Skip system indices (mem_wal, frag_reuse) which don't have files
            if index.name == lance_index::mem_wal::MEM_WAL_INDEX_NAME
                || index.name == lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME
            {
                continue;
            }
            assert!(
                index.files.is_some(),
                "{}: Index '{}' should have files field populated",
                context,
                index.name
            );
            let files = index.files.as_ref().unwrap();
            assert!(
                !files.is_empty(),
                "{}: Index '{}' should have at least one file",
                context,
                index.name
            );
            for file in files {
                assert!(
                    file.size_bytes > 0,
                    "{}: Index '{}' file '{}' should have positive size",
                    context,
                    index.name,
                    file.path
                );
            }
        }
    }

    #[tokio::test]
    async fn test_scalar_index_create_does_not_list_files() {
        let test_dir = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("category", DataType::Int32, false),
        ]));
        let ids = Int32Array::from_iter_values(0..128);
        let categories = Int32Array::from_iter_values((0..128).map(|value| value % 8));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(categories)])
            .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, test_dir.as_str(), None)
            .await
            .unwrap();
        let io_tracker = dataset.object_store.as_ref().io_tracker().clone();

        io_tracker.incremental_stats();
        dataset
            .create_index(
                &["category"],
                IndexType::Bitmap,
                Some("category_bitmap".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        let stats = io_tracker.incremental_stats();
        let list_stats = list_io_stats(&stats);
        assert_io_eq!(
            list_stats,
            read_iops,
            0,
            "new scalar index files should be reported by writer return values"
        );
    }

    #[tokio::test]
    async fn test_vector_index_create_does_not_list_files() {
        let test_dir = TempStrDir::default();
        let dimension = 8;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension,
                ),
                false,
            ),
        ]));
        let ids = Int32Array::from_iter_values(0..256);
        let vectors = (0..256)
            .map(|row| {
                Some(
                    (0..dimension)
                        .map(|dim| Some((row * dimension + dim) as f32))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let vector_array =
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, dimension);
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vector_array)])
                .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, test_dir.as_str(), None)
            .await
            .unwrap();
        let io_tracker = dataset.object_store.as_ref().io_tracker().clone();

        io_tracker.incremental_stats();
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_ivf_flat".to_string()),
                &VectorIndexParams::ivf_flat(4, MetricType::L2),
                true,
            )
            .await
            .unwrap();

        let stats = io_tracker.incremental_stats();
        let list_stats = list_io_stats(&stats);
        assert_io_eq!(
            list_stats,
            read_iops,
            0,
            "new V3 vector index files should be reported by builder return values"
        );
    }

    #[tokio::test]
    async fn test_index_file_sizes_through_lifecycle() {
        use crate::dataset::WriteDestination;
        use crate::dataset::optimize::{CompactionOptions, compact_files, remapping};
        use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;

        // Create initial dataset with columns for different index types
        let data = gen_batch()
            .col("int_col", array::step::<Int32Type>())
            .col("str_col", array::rand_utf8(8.into(), false))
            .col(
                "vec_col",
                array::rand_vec::<Float32Type>(Dimension::from(32)),
            )
            .into_reader_rows(RowCount::from(1000), BatchCount::from(1));

        let test_dir = TempStrDir::default();
        let mut dataset = Dataset::write(
            data,
            test_dir.as_str(),
            Some(WriteParams {
                max_rows_per_file: 200, // Multiple fragments for compaction
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create BTree index
        dataset
            .create_index(
                &["int_col"],
                IndexType::BTree,
                Some("btree_idx".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Create Bitmap index
        dataset
            .create_index(
                &["int_col"],
                IndexType::Bitmap,
                Some("bitmap_idx".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Create Inverted index for text search
        dataset
            .create_index(
                &["str_col"],
                IndexType::Inverted,
                Some("inverted_idx".to_string()),
                &InvertedIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Validate files are populated after creation
        assert_all_indices_have_files(&dataset, "after initial creation").await;

        // Append more data
        let more_data = gen_batch()
            .col("int_col", array::step::<Int32Type>())
            .col("str_col", array::rand_utf8(8.into(), false))
            .col(
                "vec_col",
                array::rand_vec::<Float32Type>(Dimension::from(32)),
            )
            .into_reader_rows(RowCount::from(500), BatchCount::from(1));

        Dataset::write(
            more_data,
            WriteDestination::Dataset(Arc::new(dataset.clone())),
            Some(WriteParams {
                max_rows_per_file: 200,
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset = DatasetBuilder::from_uri(test_dir.as_str())
            .load()
            .await
            .unwrap();

        // Optimize indices (triggers update/merge)
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();

        // Validate files are still populated after optimize
        assert_all_indices_have_files(&dataset, "after optimize_indices").await;

        // Run compaction with deferred remap
        let options = CompactionOptions {
            target_rows_per_fragment: 500,
            defer_index_remap: true,
            ..Default::default()
        };

        compact_files(&mut dataset, options.clone(), None)
            .await
            .unwrap();

        // Check if frag reuse index exists (indicates remap is needed)
        if dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some()
        {
            // Remap each index
            remapping::remap_column_index(
                &mut dataset,
                &["int_col"],
                Some("btree_idx".to_string()),
            )
            .await
            .unwrap();

            remapping::remap_column_index(
                &mut dataset,
                &["int_col"],
                Some("bitmap_idx".to_string()),
            )
            .await
            .unwrap();

            remapping::remap_column_index(
                &mut dataset,
                &["str_col"],
                Some("inverted_idx".to_string()),
            )
            .await
            .unwrap();

            // Validate files are populated after remap
            assert_all_indices_have_files(&dataset, "after remap").await;
        }
    }

    #[tokio::test]
    async fn test_btree_index_iops() {
        // Test that querying a BTree index uses minimal IOPs (no HEAD requests)
        let test_dir = TempStrDir::default();

        // Create dataset with a column suitable for BTree index
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let num_rows = 1000;
        let ids = Int32Array::from_iter_values(0..num_rows);
        let values = Int32Array::from_iter_values((0..num_rows).map(|i| i % 100));

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(values)]).unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_dir.as_str(), None)
            .await
            .unwrap();

        // Create BTree index
        dataset
            .create_index(
                &["value"],
                IndexType::BTree,
                Some("btree_idx".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Re-open dataset fresh to avoid cached state
        let dataset = DatasetBuilder::from_uri(test_dir.as_str())
            .load()
            .await
            .unwrap();

        // Reset IO stats before query
        let _ = dataset.object_store.as_ref().io_stats_incremental();

        // Query using the BTree index
        let results = dataset
            .scan()
            .filter("value = 50")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(results.num_rows() > 0);

        // Verify IOPs - should be minimal (no HEAD requests)
        let stats = dataset.object_store.as_ref().io_stats_incremental();
        // We expect reads for: index metadata + index pages + data files
        // The key assertion is that we don't have extra HEAD requests
        assert_io_lt!(
            stats,
            read_iops,
            10,
            "BTree index query should use minimal IOPs"
        );
    }

    #[tokio::test]
    async fn test_bitmap_index_iops() {
        // Test that querying a Bitmap index uses minimal IOPs (no HEAD requests)
        let test_dir = TempStrDir::default();

        // Create dataset with low-cardinality column for Bitmap index
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("category", DataType::Int32, false),
        ]));

        let num_rows = 1000;
        let ids = Int32Array::from_iter_values(0..num_rows);
        // Low cardinality - only 10 unique values
        let categories = Int32Array::from_iter_values((0..num_rows).map(|i| i % 10));

        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(categories)])
            .unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_dir.as_str(), None)
            .await
            .unwrap();

        // Create Bitmap index
        dataset
            .create_index(
                &["category"],
                IndexType::Bitmap,
                Some("bitmap_idx".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Re-open dataset fresh
        let dataset = DatasetBuilder::from_uri(test_dir.as_str())
            .load()
            .await
            .unwrap();

        // Reset IO stats before query
        let _ = dataset.object_store.as_ref().io_stats_incremental();

        // Query using the Bitmap index
        let results = dataset
            .scan()
            .filter("category = 5")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(results.num_rows() > 0);

        // Verify IOPs
        let stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_lt!(
            stats,
            read_iops,
            10,
            "Bitmap index query should use minimal IOPs"
        );
    }

    #[tokio::test]
    async fn test_inverted_index_iops() {
        // Test that querying an Inverted (FTS) index uses minimal IOPs
        let test_dir = TempStrDir::default();

        // Create dataset with text column for Inverted index
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, false),
        ]));

        let num_rows = 100;
        let ids = Int32Array::from_iter_values(0..num_rows);
        let texts = StringArray::from_iter_values((0..num_rows).map(|i| {
            if i % 3 == 0 {
                format!("hello world document {}", i)
            } else if i % 3 == 1 {
                format!("goodbye universe text {}", i)
            } else {
                format!("random content item {}", i)
            }
        }));

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(texts)]).unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_dir.as_str(), None)
            .await
            .unwrap();

        // Create Inverted index
        let params = InvertedIndexParams::default();
        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                Some("inverted_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Re-open dataset fresh
        let dataset = DatasetBuilder::from_uri(test_dir.as_str())
            .load()
            .await
            .unwrap();

        // Reset IO stats before query
        let _ = dataset.object_store.as_ref().io_stats_incremental();

        // Query using the Inverted index (full-text search)
        let results = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("hello".to_string()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(results.num_rows() > 0);

        // Verify IOPs. The deferred DocSet loads per-doc num_tokens/row_ids on
        // first use rather than eagerly at index open, so a cold (un-prewarmed)
        // query opens the docs file on demand — a couple more IOPs than the
        // eager path, but constant and only on the first query (prewarm or a
        // warm cache serve it with zero IO).
        let stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_lt!(
            stats,
            read_iops,
            18,
            "Inverted index query should use minimal IOPs"
        );
    }

    #[tokio::test]
    async fn test_ivf_pq_index_iops() {
        // Test that querying an IVF_PQ vector index uses minimal IOPs
        let test_dir = TempStrDir::default();

        // Create dataset with vector column
        let dimension = 32;
        let num_rows = 1000;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension,
                ),
                false,
            ),
        ]));

        let ids = Int32Array::from_iter_values(0..num_rows);
        let vectors: Vec<Option<Vec<Option<f32>>>> = (0..num_rows)
            .map(|i| {
                Some(
                    (0..dimension)
                        .map(|j| Some((i * dimension + j) as f32 / 1000.0))
                        .collect(),
                )
            })
            .collect();
        let vector_array =
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, dimension);

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vector_array)])
                .unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_dir.as_str(), None)
            .await
            .unwrap();

        // Create IVF_PQ index
        let params = VectorIndexParams::ivf_pq(4, 8, 4, MetricType::L2, 50);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("ivf_pq_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Re-open dataset fresh
        let dataset = DatasetBuilder::from_uri(test_dir.as_str())
            .load()
            .await
            .unwrap();

        // Do a full scan to warm up data file metadata
        let _ = dataset.scan().try_into_batch().await.unwrap();

        // Reset IO stats before query
        let _ = dataset.object_store.as_ref().io_stats_incremental();

        // Query using the IVF_PQ index (KNN search)
        let query_vector: Vec<f32> = (0..dimension).map(|i| i as f32 / 1000.0).collect();
        let results = dataset
            .scan()
            .nearest("vector", &Float32Array::from(query_vector), 10)
            .unwrap()
            .nprobes(2)
            .try_into_batch()
            .await
            .unwrap();
        assert!(results.num_rows() > 0);

        // Verify IOPs
        let stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_lt!(
            stats,
            read_iops,
            17,
            "IVF_PQ index query should use minimal IOPs"
        );
    }

    #[tokio::test]
    async fn test_describe_indices_returns_correct_vector_index_type() {
        const DIM: i32 = 8;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
                true,
            ),
        ]));

        let data = generate_random_array(256 * DIM as usize);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..256)),
                Arc::new(FixedSizeListArray::try_new_from_values(data, DIM).unwrap()),
            ],
        )
        .unwrap();

        let test_dir = TempStrDir::default();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(reader, &test_dir, None).await.unwrap();

        // Create IVF_FLAT index
        let params = VectorIndexParams::ivf_flat(2, MetricType::L2);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Reload dataset and call describe_indices
        let dataset = Dataset::open(&test_dir).await.unwrap();
        let descriptions = dataset.describe_indices(None).await.unwrap();

        assert_eq!(descriptions.len(), 1);
        let desc = &descriptions[0];
        assert_eq!(desc.name(), "vector_idx");
        // This should be "IVF_FLAT", not "Unknown"
        assert_eq!(desc.index_type(), "IVF_FLAT");
        assert!(!desc.field_ids().is_empty());
    }

    /// FRI-straddle corruption (PR #6610) used to panic in `load_indices`.
    /// The fixture is a pre-#6610 dataset where a user index's
    /// `fragment_bitmap` only partially covers a rewrite group. After the
    /// tolerant-load fix `load_indices` returns Ok; affected old-frag IDs
    /// are dropped, no new-frag IDs are inserted, and `validate()` succeeds.
    #[tokio::test]
    async fn test_load_indices_tolerates_fri_straddle() {
        let tmp = copy_test_data_to_tmp("fri_straddle_pre_6610/fri_straddle_dataset").unwrap();
        let uri = format!("file://{}", tmp.std_path().display());
        let dataset = Dataset::open(&uri).await.unwrap();

        let indices = dataset.load_indices().await.unwrap();
        assert!(!indices.is_empty());
        dataset.validate().await.unwrap();
    }

    /// Any commit reseeds indices via `load_indices` → `build_manifest`,
    /// so a single no-op write persists the cleaned bitmap to disk.
    #[tokio::test]
    async fn test_auto_heal_persists_cleaned_bitmap() {
        use lance_table::io::manifest::read_manifest_indexes;

        let tmp = copy_test_data_to_tmp("fri_straddle_pre_6610/fri_straddle_dataset").unwrap();
        let uri = format!("file://{}", tmp.std_path().display());

        // Sanity: the on-disk fixture has at least one straddling segment.
        let pre = Dataset::open(&uri).await.unwrap();
        let raw_pre =
            read_manifest_indexes(&pre.object_store, &pre.manifest_location, &pre.manifest)
                .await
                .unwrap();
        let cleaned = pre.load_indices().await.unwrap();
        let any_changed = raw_pre
            .iter()
            .zip(cleaned.iter())
            .any(|(r, c)| r.fragment_bitmap != c.fragment_bitmap);
        assert!(
            any_changed,
            "fixture should have at least one segment whose bitmap is cleaned at load"
        );
        drop(pre);

        // No-op delete commits a fresh manifest seeded from cleaned indices.
        let mut dataset = Dataset::open(&uri).await.unwrap();
        dataset.delete("false").await.unwrap();

        // Reopen and read raw manifest indices: cleaned bitmaps now persisted.
        let healed = Dataset::open(&uri).await.unwrap();
        let raw_post = read_manifest_indexes(
            &healed.object_store,
            &healed.manifest_location,
            &healed.manifest,
        )
        .await
        .unwrap();
        let cleaned_post = healed.load_indices().await.unwrap();
        for (r, c) in raw_post.iter().zip(cleaned_post.iter()) {
            assert_eq!(
                r.fragment_bitmap, c.fragment_bitmap,
                "after auto-heal, raw on-disk bitmap should match cleaned bitmap"
            );
        }
    }

    fn two_column_reader() -> impl arrow_array::RecordBatchReader + Send + 'static {
        lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("payload", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(10), BatchCount::from(1))
    }

    #[derive(Debug, Clone, Copy)]
    enum UnreadableIndexKind {
        NewerVersion,
        UnknownType,
    }

    /// Make `index_name` unreadable to this build, and give it the full fragment
    /// coverage a real index of that name would have.
    async fn hide_index_as(dataset: &mut Dataset, index_name: &str, kind: UnreadableIndexKind) {
        let current = dataset.load_indices_by_name(index_name).await.unwrap();
        assert_eq!(current.len(), 1);
        let mut unreadable = current.clone();
        match kind {
            UnreadableIndexKind::NewerVersion => {
                unreadable[0].index_version = current[0].index_version + 1;
            }
            UnreadableIndexKind::UnknownType => {
                unreadable[0].index_details = Some(Arc::new(prost_types::Any {
                    type_url: "type.googleapis.com/example.ForeignIndexDetails".to_string(),
                    value: Vec::new(),
                }));
            }
        }
        unreadable[0].fragment_bitmap = Some(dataset.fragment_bitmap.as_ref().clone());
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: unreadable,
                removed_indices: current,
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();
    }

    /// Raise `index_name` past the version this build can read.
    async fn hide_index_from_this_build(dataset: &mut Dataset, index_name: &str) {
        hide_index_as(dataset, index_name, UnreadableIndexKind::NewerVersion).await;
    }

    /// The readable companion every fixture below carries over `payload`.
    const READABLE_INDEX: &str = "payload_idx";

    /// A dataset carrying an unreadable BTree index over `id` beside an ordinary
    /// readable BTree index over `payload`.
    ///
    /// The readable companion is what makes the filter's selectivity visible:
    /// with a single entry, "hid the one it cannot read" and "hid everything"
    /// produce the same answer to every assertion in this module.
    ///
    /// Both indices are committed untrained and given their coverage by hand.
    /// Nothing here ever reads them, and training one would take a non-spillable
    /// 40 MB reservation out of the session's shared 150 MB pool to sort ten
    /// rows - three of those in flight at once is all the pool has room for.
    async fn dataset_with_an_unreadable_index(
        uri: &str,
        index_name: &str,
        kind: UnreadableIndexKind,
    ) -> Dataset {
        let mut dataset = Dataset::write(two_column_reader(), uri, None)
            .await
            .unwrap();

        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .name(index_name.to_string())
            .train(false)
            .await
            .unwrap();
        hide_index_as(&mut dataset, index_name, kind).await;

        dataset
            .create_index_builder(&["payload"], IndexType::BTree, &btree_params)
            .name(READABLE_INDEX.to_string())
            .train(false)
            .await
            .unwrap();
        dataset
    }

    /// A dataset carrying an index whose version is newer than this build.
    async fn dataset_with_an_index_from_a_newer_build(uri: &str, index_name: &str) -> Dataset {
        dataset_with_an_unreadable_index(uri, index_name, UnreadableIndexKind::NewerVersion).await
    }

    /// Indices the manifest itself carries, bypassing the usable-index filter.
    async fn raw_manifest_indices(dataset: &Dataset) -> Vec<IndexMetadata> {
        lance_table::io::manifest::read_manifest_indexes(
            &dataset.object_store,
            &dataset.manifest_location,
            &dataset.manifest,
        )
        .await
        .unwrap()
    }

    /// The manifest entry named `name`, whole. Compare these, not names: a
    /// carried-forward index that kept its name but lost its coverage or gained
    /// a new uuid is exactly the corruption this suite exists to catch.
    async fn manifest_index(dataset: &Dataset, name: &str) -> IndexMetadata {
        raw_manifest_indices(dataset)
            .await
            .into_iter()
            .find(|idx| idx.name == name)
            .unwrap_or_else(|| panic!("no index named {name} in the manifest"))
    }

    /// Sorted: a commit that replaces an entry appends the replacement, so
    /// comparing in manifest order would break on which operation ran rather
    /// than on what it did. Order is not meaningless in general - delta merging
    /// selects a suffix of it - but no test here asserts on it.
    async fn manifest_index_names(dataset: &Dataset) -> Vec<String> {
        let mut names = raw_manifest_indices(dataset)
            .await
            .into_iter()
            .map(|idx| idx.name)
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    /// An index this build cannot read must be hidden, not erased.
    ///
    /// Every commit rebuilds the index list from what it is handed, so filtering
    /// the version there turns "ignore it" into "delete it", and the build that
    /// could have read the index never gets the chance.
    #[tokio::test]
    async fn test_unsupported_index_survives_an_unrelated_commit() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(
            dataset
                .load_indices()
                .await
                .unwrap()
                .iter()
                .map(|idx| idx.name.as_str())
                .collect::<Vec<_>>(),
            [READABLE_INDEX],
            "the filter must hide the index this build cannot read, and only it"
        );
        assert_eq!(
            manifest_index_names(&dataset).await,
            ["id_idx", READABLE_INDEX]
        );
        let before = manifest_index(&dataset, "id_idx").await;

        // An unrelated append. Nothing about the index is part of this operation.
        let dataset = Dataset::write(
            two_column_reader(),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(
            manifest_index(&dataset, "id_idx").await,
            before,
            "an unrelated append changed an index this build merely could not read"
        );
    }

    /// Carrying an unreadable index forward is not the same as keeping it
    /// forever: dropping the column it covers still removes it.
    #[tokio::test]
    async fn test_unsupported_index_is_dropped_with_its_column() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

        let mut dataset = Dataset::open(test_uri).await.unwrap();
        let readable_before = manifest_index(&dataset, READABLE_INDEX).await;

        dataset.drop_columns(&["id"]).await.unwrap();

        assert_eq!(
            manifest_index_names(&dataset).await,
            [READABLE_INDEX],
            "dropping `id` must remove the index over it, and only that one"
        );
        assert_eq!(
            manifest_index(&dataset, READABLE_INDEX).await,
            readable_before,
            "the index over the surviving column was rewritten"
        );
    }

    /// Carrying it forward must also not drag it through index migration.
    ///
    /// `migrate_indices` recalculates a missing `fragment_bitmap` by opening the
    /// index, which is precisely what this build cannot do - so an unreadable
    /// index would fail every later commit instead of riding along.
    #[rstest]
    #[case::newer_version(UnreadableIndexKind::NewerVersion)]
    #[case::unknown_type(UnreadableIndexKind::UnknownType)]
    #[tokio::test]
    async fn test_unsupported_index_without_a_bitmap_does_not_fail_later_commits(
        #[case] kind: UnreadableIndexKind,
    ) {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = dataset_with_an_unreadable_index(test_uri, "id_idx", kind).await;

        // Drop the coverage too, so migration would want to rebuild it.
        let hidden = manifest_index(&dataset, "id_idx").await;
        let without_bitmap = IndexMetadata {
            fragment_bitmap: None,
            ..hidden.clone()
        };
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![without_bitmap.clone()],
                removed_indices: vec![hidden],
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let mut dataset = Dataset::open(test_uri).await.unwrap();
        dataset.delete("false").await.unwrap();

        assert_eq!(
            manifest_index(&dataset, "id_idx").await,
            without_bitmap,
            "a commit rewrote an index it cannot open instead of carrying it through"
        );
    }

    /// A name an unreadable index already owns cannot be handed out again.
    ///
    /// Nothing in the format stops two entries from sharing a name, and the
    /// build that can read both would take them for segments of one index.
    #[tokio::test]
    async fn test_unsupported_index_name_is_still_taken() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

        let mut dataset = Dataset::open(test_uri).await.unwrap();
        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        let err = dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .name("id_idx".to_string())
            .await
            .expect_err("a name is taken by an index this build cannot read");
        assert!(
            err.to_string().contains("already exists"),
            "expected a name collision, got: {err}"
        );
        assert_eq!(
            manifest_index_names(&dataset).await,
            ["id_idx", READABLE_INDEX]
        );
    }

    /// The generated-name loop reads the same view the collision check does, so
    /// a name an unreadable index holds is skipped rather than reused.
    #[tokio::test]
    async fn test_an_auto_generated_name_skips_an_unsupported_index() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

        // A different index kind on the same column: the loop steps past the
        // taken name instead of stopping at the collision check.
        let mut dataset = Dataset::open(test_uri).await.unwrap();
        let bitmap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
        dataset
            .create_index_builder(&["id"], IndexType::Bitmap, &bitmap_params)
            .train(false)
            .await
            .unwrap();

        assert_eq!(
            manifest_index_names(&dataset).await,
            ["id_idx", "id_idx_2", READABLE_INDEX],
            "the generated name reused one an unreadable index already holds"
        );
    }

    /// The multi-segment FM-Index builder reserves names on its own, so it needs
    /// the same complete view as the single-segment path.
    ///
    /// The hidden index is a BTree and the new one an FM index, which is what
    /// makes this test specific to the multi-segment builder: its name loop
    /// (`index/create.rs`) only steps past a taken name when the *fields* differ,
    /// where the single-segment loop also steps past a different index kind. So
    /// the single-segment path would quietly settle on `text_idx_2` and succeed;
    /// only the multi-segment path keeps `text_idx` and hits the collision.
    #[tokio::test]
    async fn test_multi_segment_fmindex_respects_an_unsupported_index_name() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"]))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
            test_uri,
            None,
        )
        .await
        .unwrap();

        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index_builder(&["text"], IndexType::BTree, &btree_params)
            .name("text_idx".to_string())
            .train(false)
            .await
            .unwrap();
        hide_index_from_this_build(&mut dataset, "text_idx").await;

        let mut dataset = Dataset::open(test_uri).await.unwrap();
        let multi_segment_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Fm)
            .with_params(&serde_json::json!({ "num_segments": 2 }));
        let err = dataset
            .create_index_builder(&["text"], IndexType::Fm, &multi_segment_params)
            .train(false)
            .await
            .expect_err("a name is taken by an index this build cannot read");
        assert!(
            err.to_string().contains("already exists"),
            "expected a name collision, got: {err}"
        );
        assert_eq!(manifest_index_names(&dataset).await, ["text_idx"]);
    }

    /// Being unreadable must not make an index unremovable.
    #[tokio::test]
    async fn test_unsupported_index_can_be_dropped_by_name() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

        let mut dataset = Dataset::open(test_uri).await.unwrap();
        dataset.drop_index("id_idx").await.unwrap();

        assert_eq!(
            manifest_index_names(&dataset).await,
            [READABLE_INDEX],
            "drop_index removed the wrong set of indices"
        );
    }

    /// `replace` has to select the index it replaces from the same complete view
    /// the name was reserved against, or it adds a twin instead of replacing.
    #[tokio::test]
    async fn test_replacing_an_unsupported_index_does_not_duplicate_it() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

        let mut dataset = Dataset::open(test_uri).await.unwrap();
        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .name("id_idx".to_string())
            .replace(true)
            .train(false)
            .await
            .unwrap();

        assert_eq!(
            manifest_index_names(&dataset).await,
            ["id_idx", READABLE_INDEX],
            "replace committed a second index under a name already taken"
        );
        assert!(
            unsupported_index_version(&manifest_index(&dataset, "id_idx").await).is_none(),
            "replace kept the unreadable index and discarded the new one"
        );
    }

    /// The same selection, through the segment-commit path rather than the
    /// builder: full coverage replaces the segments already under that name.
    #[tokio::test]
    async fn test_committing_a_segment_beside_an_unsupported_index_replaces_it() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

        let mut dataset = Dataset::open(test_uri).await.unwrap();
        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        let mut segment = dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .name("id_idx".to_string())
            .replace(true)
            .train(false)
            .execute_uncommitted()
            .await
            .unwrap();
        // Full coverage, so the removal decision goes through the fragment
        // overlap branch rather than the empty-bitmap shortcut. Set by hand
        // because training it would take 40 MB of the shared pool to sort ten
        // rows - see `dataset_with_an_index_from_a_newer_build`.
        segment.fragment_bitmap = Some(dataset.fragment_bitmap.as_ref().clone());
        dataset
            .commit_existing_index_segments("id_idx", "id", vec![segment])
            .await
            .unwrap();

        assert_eq!(
            manifest_index_names(&dataset).await,
            ["id_idx", READABLE_INDEX],
            "the incoming segment was committed beside the unreadable one"
        );
        assert!(
            unsupported_index_version(&manifest_index(&dataset, "id_idx").await).is_none(),
            "the incoming segment did not replace the unreadable one"
        );
    }

    /// The retention path itself: a segment on fragments the existing one does
    /// not cover is kept beside it, and agreeing declarations are what makes
    /// that legal. This is the case the rejection below must not swallow -
    /// partial-coverage builds depend on it.
    #[tokio::test]
    async fn test_committing_a_segment_on_disjoint_fragments_keeps_the_existing_one() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let mut dataset = Dataset::write(two_column_reader(), test_uri, None)
            .await
            .unwrap();
        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .name("id_idx".to_string())
            .train(false)
            .await
            .unwrap();
        // Given by hand: an untrained segment commits with an empty bitmap, which
        // takes the zero-coverage removal branch rather than the disjoint one.
        // See `dataset_with_an_index_from_a_newer_build` for why nothing here trains.
        let untrained = manifest_index(&dataset, "id_idx").await;
        let mut existing = untrained.clone();
        existing.fragment_bitmap = Some(dataset.fragment_bitmap.as_ref().clone());
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![existing.clone()],
                        removed_indices: vec![untrained],
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        let covered = existing.fragment_bitmap.clone().unwrap();
        assert!(!covered.is_empty());

        let mut dataset = Dataset::write(
            two_column_reader(),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let appended = dataset.fragment_bitmap.as_ref() - &covered;
        assert!(!appended.is_empty());

        let mut segment = dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .name("id_idx".to_string())
            .replace(true)
            .fragments(appended.iter().collect())
            .train(false)
            .execute_uncommitted()
            .await
            .unwrap();
        segment.fragment_bitmap = Some(appended);
        assert_eq!(segment.fields, existing.fields);
        assert_eq!(segment.covering_fields, existing.covering_fields);

        dataset
            .commit_existing_index_segments("id_idx", "id", vec![segment])
            .await
            .unwrap();

        let uuids = raw_manifest_indices(&dataset)
            .await
            .into_iter()
            .filter(|idx| idx.name == "id_idx")
            .map(|idx| idx.uuid)
            .collect::<Vec<_>>();
        assert_eq!(uuids.len(), 2, "the disjoint existing segment was dropped");
        assert!(uuids.contains(&existing.uuid));
    }

    /// One logical index needs one declaration, and the complete view is what
    /// makes the disagreement reachable: a segment this build cannot read may
    /// carry columns, and a plain segment committed beside it on disjoint
    /// fragments is retained rather than replaced. Both pass the per-segment
    /// rules - `keyed_fields` is the prefix left after the carried ones, so
    /// `[id, payload]` carrying `[payload]` keys on `id` exactly as `[id]` does.
    /// Committing the pair would leave `describe_indices` erroring on metadata
    /// this call just wrote, the same failure
    /// `test_build_index_metadata_from_segments_rejects_mixed_covering_declarations`
    /// pins for the incoming side.
    #[tokio::test]
    async fn test_committing_a_plain_segment_beside_a_covered_unsupported_one_is_rejected() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

        // Give the hidden segment a carried column. A newer build is exactly
        // where a covering segment would come from.
        let hidden = manifest_index(&dataset, "id_idx").await;
        let payload_id = dataset.schema().field("payload").unwrap().id;
        let mut hidden_covered = hidden.clone();
        hidden_covered.fields.push(payload_id);
        hidden_covered.covering_fields = vec![payload_id];
        assert_eq!(hidden_covered.keyed_field(), hidden.keyed_field());
        let covered_fragments = hidden_covered.fragment_bitmap.clone().unwrap();
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![hidden_covered],
                        removed_indices: vec![hidden],
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();

        // Fragments the hidden segment does not cover, so the incoming segment
        // takes the disjoint branch and the hidden one is retained.
        let mut dataset = Dataset::write(
            two_column_reader(),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let appended = dataset.fragment_bitmap.as_ref() - &covered_fragments;
        assert!(!appended.is_empty());

        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        let mut plain = dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .name("id_idx".to_string())
            .replace(true)
            .fragments(appended.iter().collect())
            .train(false)
            .execute_uncommitted()
            .await
            .unwrap();
        plain.fragment_bitmap = Some(appended);
        assert!(plain.covering_fields.is_empty());

        let err = dataset
            .commit_existing_index_segments("id_idx", "id", vec![plain])
            .await
            .expect_err("a logical index cannot mix covered and plain segment declarations");
        assert!(
            err.to_string().contains("covering_fields"),
            "unexpected error: {err}"
        );
    }

    /// A cast reassigns the field id, so no index on that column can be carried
    /// forward. The guard that makes that explicit has to see the hidden ones
    /// too, or they get exactly the silent drop it exists to abolish.
    #[tokio::test]
    async fn test_casting_a_column_with_an_unsupported_index_is_rejected() {
        use crate::dataset::ColumnAlteration;

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

        let mut dataset = Dataset::open(test_uri).await.unwrap();
        let err = dataset
            .alter_columns(&[ColumnAlteration::new("id".into()).cast_to(DataType::Int64)])
            .await
            .expect_err("a cast must not silently erase an index it cannot read");
        assert!(
            err.to_string().contains("id_idx"),
            "the error should name the index, got: {err}"
        );
        assert_eq!(
            manifest_index_names(&dataset).await,
            ["id_idx", READABLE_INDEX],
            "the rejected cast still changed the manifest"
        );
    }

    /// A cache entry written under the key's previous meaning must cold-miss.
    ///
    /// v1 of `lance.index.metadata-key` held only the indices the writing build
    /// could read. The key fields are identical, so on a persistent backend
    /// shared with such a build nothing but the schema version stops this one
    /// from reading that filtered list as the complete one.
    #[tokio::test]
    async fn test_a_pre_rotation_cache_entry_is_not_consulted() {
        use lance_core::cache::{CacheCodec, CacheKey, CacheKeySchema, KeyBuilder};
        use std::borrow::Cow;

        struct PreRotationIndexMetadataKey<'a> {
            version: u64,
            store_identity: &'a str,
        }

        impl CacheKey for PreRotationIndexMetadataKey<'_> {
            type ValueType = Vec<IndexMetadata>;

            fn key(&self) -> Cow<'_, str> {
                Cow::Owned(format!(
                    "{}:{}/{}",
                    self.store_identity.len(),
                    self.store_identity,
                    self.version
                ))
            }

            fn type_name() -> &'static str {
                "Vec<IndexMetadata>"
            }

            fn schema() -> CacheKeySchema {
                CacheKeySchema::new("lance.index.metadata-key", 1)
            }

            fn write_key(&self, builder: &mut KeyBuilder) {
                builder.write_str(self.store_identity);
                builder.write_u64(self.version);
            }

            fn codec() -> Option<CacheCodec> {
                Some(lance_table::format::index_metadata_codec())
            }
        }

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

        let mut dataset = Dataset::open(test_uri).await.unwrap();
        let complete = raw_manifest_indices(&dataset).await;
        let as_a_released_build_would_cache_it = complete
            .iter()
            .filter(|idx| unsupported_index_version(idx).is_none())
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            as_a_released_build_would_cache_it.len() < complete.len(),
            "the fixture must give the two key versions different values to cache"
        );
        dataset
            .index_cache
            .insert_with_key(
                &PreRotationIndexMetadataKey {
                    version: dataset.version().version,
                    store_identity: &dataset.object_store.store_prefix,
                },
                Arc::new(as_a_released_build_would_cache_it),
            )
            .await;

        dataset.delete("false").await.unwrap();

        assert_eq!(
            manifest_index_names(&dataset).await,
            ["id_idx", READABLE_INDEX],
            "a commit read a cache entry written under the key's previous meaning"
        );
    }

    /// `validate` checks the manifest, so it has to see all of it. An index this
    /// build cannot read is no less corrupt for being unreadable, and now that
    /// such an index is carried forward the corrupt state is durable rather than
    /// gone at the next commit.
    #[tokio::test]
    async fn test_validate_sees_an_unsupported_index() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;
        dataset.validate().await.unwrap();

        // A second segment under the same name covering the same fragments. Only
        // `detect_overlapping_fragments` over the complete list can see it.
        let hidden = manifest_index(&dataset, "id_idx").await;
        let overlapping = IndexMetadata {
            uuid: Uuid::new_v4(),
            ..hidden
        };
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![overlapping],
                removed_indices: vec![],
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let err = dataset
            .validate()
            .await
            .expect_err("two segments of one name covering the same fragments is corrupt");
        assert!(
            err.to_string().contains("id_idx"),
            "the error should name the index, got: {err}"
        );
    }

    /// A detached commit builds its manifest through its own code path, and it
    /// carries the index list forward exactly as an attached one does.
    #[tokio::test]
    async fn test_a_detached_commit_does_not_erase_an_unsupported_index() {
        use crate::dataset::InsertBuilder;
        use crate::dataset::write::CommitBuilder;

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

        let batches = two_column_reader()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let dataset = Arc::new(Dataset::open(test_uri).await.unwrap());
        let before = manifest_index(&dataset, "id_idx").await;
        let transaction = InsertBuilder::new(dataset.clone())
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute_uncommitted(batches)
            .await
            .unwrap();
        let detached = CommitBuilder::new(dataset.clone())
            .with_detached(true)
            .execute(transaction)
            .await
            .unwrap();

        assert_eq!(
            manifest_index_names(&detached).await,
            ["id_idx", READABLE_INDEX],
            "a detached commit erased an index this build merely could not read"
        );
        assert_eq!(
            manifest_index(&detached, "id_idx").await,
            before,
            "a detached commit rewrote an index this build merely could not read"
        );
        assert!(lance_table::format::is_detached_version(
            detached.manifest.version
        ));
        assert_eq!(detached.count_rows(None).await.unwrap(), 20);
    }

    /// Compaction bins fragments so that no rewrite group splits an index's
    /// coverage, and `recalculate_fragment_bitmap` rejects the plan if one does.
    /// Both sides therefore have to count the same indices: planning from the
    /// filtered view while the commit carries the complete one makes compaction
    /// fail outright on a dataset holding an index from a newer build.
    #[tokio::test]
    async fn test_compaction_survives_an_unsupported_index() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 5,
            ..Default::default()
        };
        let mut dataset = Dataset::write(two_column_reader(), test_uri, Some(write_params.clone()))
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);

        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .name("id_idx".to_string())
            .train(false)
            .await
            .unwrap();
        hide_index_from_this_build(&mut dataset, "id_idx").await;

        // A fragment the index does not cover, so a bin holding it together with
        // the covered ones would split the index's coverage.
        let mut dataset = Dataset::write(
            two_column_reader(),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..write_params
            }),
        )
        .await
        .unwrap();
        let before = manifest_index(&dataset, "id_idx").await;

        let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();

        // Without a real rewrite the assertions below hold vacuously: an empty
        // plan commits nothing and re-reads the manifest it started from.
        assert_eq!(metrics.fragments_removed, 4);
        assert_eq!(dataset.get_fragments().len(), 2);

        let after = manifest_index(&dataset, "id_idx").await;
        assert_eq!(after.uuid, before.uuid);
        assert_eq!(after.index_version, before.index_version);
        let live = dataset
            .get_fragments()
            .iter()
            .map(|f| f.id() as u32)
            .collect::<RoaringBitmap>();
        assert!(
            !after.fragment_bitmap.unwrap().is_disjoint(&live),
            "the surviving index covers only fragments the rewrite deleted"
        );
    }

    /// Without stable row ids a rewrite moves every row address, so each index
    /// over the rewritten fragments has to be remapped - and remapping one means
    /// opening it. A build with no reader for an index cannot remap it, so
    /// compacting the fragments it covers would leave it addressing rows that no
    /// longer exist. Those fragments are held back from the plan instead; the
    /// rest of the table still compacts.
    ///
    /// The stable-row-id case is the test above: there the fragment-reuse index
    /// repairs the coverage afterwards, so nothing has to be held back.
    #[rstest]
    #[case::newer_version(UnreadableIndexKind::NewerVersion)]
    #[case::unknown_type(UnreadableIndexKind::UnknownType)]
    #[tokio::test]
    async fn test_compaction_defers_fragments_an_unsupported_index_covers(
        #[case] kind: UnreadableIndexKind,
    ) {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let write_params = WriteParams {
            enable_stable_row_ids: false,
            max_rows_per_file: 5,
            ..Default::default()
        };
        let mut dataset = Dataset::write(two_column_reader(), test_uri, Some(write_params.clone()))
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);

        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .name("id_idx".to_string())
            .train(false)
            .await
            .unwrap();
        hide_index_as(&mut dataset, "id_idx", kind).await;
        let covered = manifest_index(&dataset, "id_idx")
            .await
            .fragment_bitmap
            .unwrap();

        // Two more fragments the hidden index does not cover: they are the ones
        // compaction is still free to rewrite.
        let mut dataset = Dataset::write(
            two_column_reader(),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..write_params
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 4);
        let before = manifest_index(&dataset, "id_idx").await;

        let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();

        // The two uncovered fragments coalesce; the two the hidden index covers
        // are left alone. Both halves matter: no rewrite at all would satisfy the
        // coverage assertion below for the wrong reason.
        assert_eq!(metrics.fragments_removed, 2);
        assert_eq!(metrics.fragments_added, 1);
        let live = dataset
            .get_fragments()
            .iter()
            .map(|f| f.id() as u32)
            .collect::<RoaringBitmap>();
        assert!(
            covered.is_subset(&live),
            "a fragment the unreadable index covers was rewritten: covered {covered:?}, live {live:?}"
        );

        let after = manifest_index(&dataset, "id_idx").await;
        assert_eq!(after.uuid, before.uuid);
        assert_eq!(after.fragment_bitmap, before.fragment_bitmap);

        // Nothing compactable is left outside the held-back set, so the plan is
        // empty. That has to be an ordinary no-op: on a table the index covers
        // whole - the usual shape - every compaction takes this path.
        let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        assert_eq!(metrics.fragments_removed, 0);
        assert_eq!(metrics.fragments_added, 0);
        assert_eq!(manifest_index(&dataset, "id_idx").await.uuid, before.uuid);
    }

    /// Holding back the fragments an unreadable index covers is an optimization
    /// in the planner, not the rule: `compact_files_with_planner` takes any
    /// planner, `CompactionPlan` is public and serializable, and a distributed
    /// driver hands `commit_compaction` results planned on another machine. This
    /// takes that last route, so the refusal is pinned to the commit boundary.
    ///
    /// The other half - that the boundary does not refuse a rewrite the index
    /// does not cover - is the test above, which compacts the uncovered
    /// fragments of this same shape through `compact_files`.
    #[tokio::test]
    async fn test_committing_a_compaction_an_unsupported_index_covers_is_rejected() {
        use crate::dataset::index::DatasetIndexRemapperOptions;
        use crate::dataset::optimize::{CompactionPlan, TaskData, commit_compaction};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let write_params = WriteParams {
            enable_stable_row_ids: false,
            max_rows_per_file: 5,
            ..Default::default()
        };
        let mut dataset = Dataset::write(two_column_reader(), test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .name("id_idx".to_string())
            .train(false)
            .await
            .unwrap();
        hide_index_from_this_build(&mut dataset, "id_idx").await;
        let covered = manifest_index(&dataset, "id_idx")
            .await
            .fragment_bitmap
            .unwrap();

        // Two more fragments the hidden index does not cover, so the plan below
        // is a genuine selection rather than "every fragment there is".
        let mut dataset = Dataset::write(
            two_column_reader(),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..write_params
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 4);
        let before = manifest_index(&dataset, "id_idx").await;

        let plan = CompactionPlan {
            tasks: vec![TaskData {
                fragments: dataset
                    .fragments()
                    .iter()
                    .filter(|fragment| covered.contains(fragment.id as u32))
                    .cloned()
                    .collect(),
            }],
            read_version: dataset.version().version,
            options: CompactionOptions::default(),
        };
        assert_eq!(plan.tasks[0].fragments.len(), 2);

        let mut rewrites = Vec::new();
        for task in plan.compaction_tasks() {
            rewrites.push(task.execute(&dataset).await.unwrap());
        }

        let err = commit_compaction(
            &mut dataset,
            rewrites,
            Arc::new(DatasetIndexRemapperOptions::default()),
            &plan.options,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("id_idx"),
            "the refusal has to name the index that blocks the rewrite: {err}"
        );

        // Nothing was committed: the fragments the plan named are still there,
        // and the index still covers them.
        assert_eq!(dataset.get_fragments().len(), 4);
        let after = manifest_index(&dataset, "id_idx").await;
        assert_eq!(after.uuid, before.uuid);
        assert_eq!(after.fragment_bitmap, before.fragment_bitmap);
    }

    /// An index old enough to predate the fragment bitmap has its coverage
    /// reconstructed from the version it was written against, so it arrives in
    /// that version's fragment-id space. `load_all_indices` moves a *stored*
    /// bitmap into the current space through the fragment-reuse index and leaves
    /// a `None` one alone, so the reconstruction has to make that move itself.
    ///
    /// Without it, a deferred compaction is enough to defeat both guards: the
    /// coverage still names the fragments the rows moved out of, which no later
    /// rewrite can intersect, so the planner stops holding anything back and the
    /// commit boundary waves the rewrite through.
    #[tokio::test]
    async fn test_reconstructed_coverage_follows_a_deferred_compaction() {
        use crate::dataset::index::DatasetIndexRemapperOptions;
        use crate::dataset::optimize::{CompactionPlan, TaskData, commit_compaction};

        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let write_params = WriteParams {
            enable_stable_row_ids: false,
            max_rows_per_file: 5,
            ..Default::default()
        };
        let mut dataset = Dataset::write(two_column_reader(), test_uri, Some(write_params.clone()))
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);

        let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index_builder(&["id"], IndexType::BTree, &btree_params)
            .name("id_idx".to_string())
            .train(false)
            .await
            .unwrap();
        hide_index_from_this_build(&mut dataset, "id_idx").await;

        // Drop the bitmap, which is what an index written before it existed
        // looks like: coverage has to be reconstructed from `dataset_version`.
        let hidden = manifest_index(&dataset, "id_idx").await;
        let legacy = IndexMetadata {
            fragment_bitmap: None,
            ..hidden.clone()
        };
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![legacy],
                removed_indices: vec![hidden],
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        // Deferred remap is the one compaction a build that cannot read the
        // index may run: it hands the repair on through a fragment-reuse index.
        // Fragments 0 and 1 become fragment 2, and the index still covers those
        // rows - now under a different id.
        compact_files(
            &mut dataset,
            CompactionOptions {
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let after_defer = dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<Vec<_>>();
        assert_eq!(after_defer, vec![2]);

        // The commit-boundary half, taken first: the planner half below rewrites
        // fragment 2 when the remap is missing, which would leave this nothing
        // to ask about and hide whether the guard is sensitive on its own.
        let plan = CompactionPlan {
            tasks: vec![TaskData {
                fragments: dataset.fragments().as_ref().clone(),
            }],
            read_version: dataset.version().version,
            options: CompactionOptions::default(),
        };
        let mut rewrites = Vec::new();
        for task in plan.compaction_tasks() {
            rewrites.push(task.execute(&dataset).await.unwrap());
        }
        let err = commit_compaction(
            &mut dataset,
            rewrites,
            Arc::new(DatasetIndexRemapperOptions::default()),
            &plan.options,
        )
        .await
        .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("id_idx") && message.contains("[2]"),
            "the refusal has to name the index and the fragment in current ids: {message}"
        );

        // Two fragments the index never covered, so the planner half has
        // something to compact and cannot pass by finding nothing to do.
        let mut dataset = Dataset::write(
            two_column_reader(),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..write_params
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);

        // The planner half: the appended pair coalesces, fragment 2 is held back.
        let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        assert_eq!(metrics.fragments_removed, 2);
        assert_eq!(metrics.fragments_added, 1);
        assert!(
            dataset
                .get_fragments()
                .iter()
                .any(|fragment| fragment.id() == 2),
            "the fragment the reconstructed coverage maps to was rewritten"
        );
    }

    /// Optimizing a name whose segments this build cannot all read would commit
    /// a merged segment overlapping the one it left behind - a state
    /// `Dataset::validate` reports as corruption and no later commit can heal.
    #[tokio::test]
    async fn test_optimize_skips_an_index_with_an_unsupported_segment() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

        // A readable segment beside the hidden one, under the same name, so the
        // group is exactly the mixed case: `id_idx` is now partly readable.
        let hidden = manifest_index(&dataset, "id_idx").await;
        let readable_sibling = IndexMetadata {
            uuid: Uuid::new_v4(),
            index_version: hidden.index_version - 1,
            fragment_bitmap: Some(RoaringBitmap::new()),
            ..hidden
        };
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![readable_sibling],
                removed_indices: vec![],
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();
        let segments_of = |indices: Vec<IndexMetadata>| {
            indices
                .into_iter()
                .filter(|idx| idx.name == "id_idx")
                .collect::<Vec<_>>()
        };
        let before = segments_of(raw_manifest_indices(&dataset).await);
        assert_eq!(before.len(), 2, "the fixture must build the mixed group");

        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();

        assert_eq!(
            segments_of(raw_manifest_indices(&dataset).await),
            before,
            "optimize touched a name carrying a segment this build cannot read"
        );
        dataset.validate().await.unwrap();
    }

    #[tokio::test]
    async fn test_optimize_rebuilds_dormant_vector_index_instead_of_merging_stale_rows() {
        use crate::dataset::UpdateBuilder;

        let mut dataset = gen_batch()
            .col("vec", array::rand_vec::<Float32Type>(Dimension::from(4)))
            .into_ram_dataset_with_params(
                FragmentCount::from(2),
                FragmentRowCount::from(3),
                Some(WriteParams {
                    max_rows_per_file: 3,
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        let original = dataset
            .scan()
            .project(&["vec"])
            .unwrap()
            .limit(Some(1), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let query = original["vec"].as_fixed_size_list().value(0);

        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("vec_idx".to_string()),
                &VectorIndexParams::ivf_flat(1, MetricType::L2),
                true,
            )
            .await
            .unwrap();

        // A full rewrite replaces every covered fragment; the definition is
        // retained as a dormant segment with empty effective coverage.
        let mut updated = UpdateBuilder::new(Arc::new(dataset))
            .set("vec", "array[100.0, 100.0, 100.0, 100.0]")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset
            .as_ref()
            .clone();
        let retained = updated.load_indices_by_name("vec_idx").await.unwrap();
        assert_eq!(retained.len(), 1);
        assert!(
            retained[0]
                .effective_fragment_bitmap(&updated.fragment_bitmap)
                .unwrap()
                .is_empty()
        );

        // Optimize must rebuild from live data and replace the dormant
        // segment, not merge its stale postings back in.
        updated.optimize_indices(&Default::default()).await.unwrap();
        let rebuilt = updated.load_indices_by_name("vec_idx").await.unwrap();
        assert_eq!(rebuilt.len(), 1);
        assert!(
            !rebuilt[0]
                .effective_fragment_bitmap(&updated.fragment_bitmap)
                .unwrap()
                .is_empty()
        );

        let result = updated
            .scan()
            .nearest("vec", query.as_ref(), 1)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let distance = result["_distance"].as_primitive::<Float32Type>().value(0);
        assert!(
            distance > 1_000.0,
            "nearest distance {distance} should reflect the rewritten vectors, not stale postings"
        );
    }
}
