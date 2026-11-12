// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Loading, opening, and caching software for Dataset index metadata.

// These helpers stay crate-visible so callers do not depend on private parameter types.
#![allow(clippy::redundant_pub_crate)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_schema::DataType;
use async_trait::async_trait;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::datatypes::{Field, Schema as LanceSchema};
use lance_core::utils::tracing::{
    IO_TYPE_OPEN_FRAG_REUSE, IO_TYPE_OPEN_MEM_WAL, IO_TYPE_OPEN_VECTOR, TRACE_IO_EVENTS,
};
use lance_file::reader::FileReaderOptions;
use lance_file::versions::v1::reader::FileReader as V1FileReader;
use lance_index::scalar::ScalarIndex;
use lance_index::scalar::expression::{IndexInformationProvider, MultiQueryParser};
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::flat::index::{FlatBinQuantizer, FlatIndex, FlatQuantizer};
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::quantizer::Quantization;
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::storage::ResidentColumns;
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_index::{
    INDEX_FILE_NAME, INDEX_METADATA_SCHEMA_KEY, Index,
    frag_reuse::{CompactFragReuseIndex, CompactFragReuseIndexHandle, FRAG_REUSE_INDEX_NAME},
    is_system_index,
    mem_wal::{MEM_WAL_INDEX_NAME, MemWalIndex},
    metrics::{MetricsCollector, NoOpMetricsCollector},
    pb::{self, index::Implementation},
    vector::VectorIndex,
};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::traits::Reader;
use lance_io::utils::{
    CachedFileSize, read_last_block, read_message, read_message_from_buf, read_metadata_offset,
    read_version,
};
use lance_table::format::{Fragment, IndexMetadata, SelfDescribingFileReader};
use lance_table::io::manifest::read_manifest_indexes;
use roaring::RoaringBitmap;
use tracing::{info, instrument};
use uuid::Uuid;

use crate::index::api::DatasetIndexExt;
use crate::index::frag_reuse::{self, load_frag_reuse_index_details, open_frag_reuse_index};
use crate::index::frag_reuse_with_stable_row_ids::warn_about_indices_hidden_by_frag_reuse;
use crate::index::mem_wal::open_mem_wal_index;
use crate::index::scalar::{self, IndexDetails, fetch_index_details};
use crate::index::segment::{
    index_is_usable, index_type_is_known, segment_has_vector_details, unsupported_index_version,
};
use crate::index::vector;
use crate::index::vector::{
    LogicalVectorIndex,
    details::{infer_missing_vector_details, needs_vector_details_inference},
    ivf::v2::{IVFIndex, IvfOpenContext, IvfStateEntryBox, aux_file_key},
    utils::get_vector_type,
};
use crate::session::index_caches::{
    DerivedIndexListingKey, FragReuseIndexKey, IndexMetadataKey, write_index_identity,
};
use crate::{Dataset, Error, Result};

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
pub(crate) struct FragReuseIndexCacheKey<'a> {
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
pub(crate) struct MemWalCacheKey<'a> {
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

pub(crate) async fn open_index_proto(reader: &dyn Reader) -> Result<pb::Index> {
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
        } else if !index_is_usable(idx) {
            log::warn!(
                "Index {} has unsupported field dependencies {:?}, ignoring it",
                idx.name,
                idx.dependency_fields(),
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
        version: dataset.manifest.version,
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
            warn_about_indices_hidden_by_frag_reuse(&dataset.manifest, &loaded);
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

    // Legacy FRI index version 0 is handled below by directly remapping fragment coverage.
    // For version 1 and above, load_indices handles version checks and filters index
    // segments for query use; keep their metadata unchanged here.
    if indices
        .iter()
        .any(lance_table::system_index::frag_reuse::metadata::is_tagged)
    {
        if indices
            .iter()
            .filter(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
            .count()
            != 1
        {
            return Err(Error::corrupt_file_named(
                "FRI metadata",
                "tagged history requires a single FRI entry",
            ));
        }
        // Commit bookkeeping carries the original Any unchanged. Only query
        // loading or an operation consuming FRI needs to interpret its encoding.
        return Ok(indices);
    }
    if let Some(frag_reuse_index_meta) =
        indices.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
    {
        // Query planning loads the listing several times per scan, and each
        // remap walks the whole reuse history.
        let remapped_key = DerivedIndexListingKey {
            version: metadata_key.version,
            store_identity: metadata_key.store_identity,
            e_tag: metadata_key.e_tag,
        };
        if let Some(remapped) = dataset.index_cache.get_with_key(&remapped_key).await {
            return Ok(remapped);
        }
        let identity = frag_reuse::v0_cache_identity(dataset, frag_reuse_index_meta);
        let fri_key = FragReuseIndexKey { uuid: &identity };
        let frag_reuse_index = dataset
            .index_cache
            .get_or_insert_with_key(fri_key, || async move {
                let index_details =
                    load_frag_reuse_index_details(dataset, frag_reuse_index_meta).await?;
                open_frag_reuse_index(frag_reuse_index_meta.uuid, index_details.as_ref()).await
            })
            .await?;
        let mut remapped = indices.as_ref().clone();
        for idx in remapped.iter_mut() {
            if let Some(bitmap) = idx.fragment_bitmap.as_mut() {
                frag_reuse_index.remap_fragment_bitmap(bitmap)?;
            }
        }
        let remapped = Arc::new(remapped);
        // A later call may still infer the missing details; don't pin this list.
        if !indices
            .iter()
            .any(|idx| needs_vector_details_inference(idx, dataset.schema()))
        {
            dataset
                .index_cache
                .insert_with_key(&remapped_key, remapped.clone())
                .await;
        }
        Ok(remapped)
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

/// Find the FRI entry from the raw manifest listing (bookkeeping path).
///
/// This is a metadata-only lookup: it must not drive the query reader's coverage
/// rewrite, parse the mapping, or fail on a corrupt ledger. Callers that then use
/// the mapping (for example the v0 reader inside `open_frag_reuse_index`) still
/// validate it and surface corruption; only the by-name lookup skips that work.
pub(crate) async fn find_frag_reuse_index_meta(dataset: &Dataset) -> Result<Option<IndexMetadata>> {
    Ok(load_all_indices(dataset)
        .await?
        .iter()
        .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
        .cloned())
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
    /// Opens a built vector segment without requiring it to be committed to the manifest.
    async fn open_vector_index_from_metadata(
        &self,
        column: &str,
        index_meta: &IndexMetadata,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndex>>;
    /// [`Self::open_generic_index`] for maintenance (merge, rebuild, remap):
    /// a registered segment the tagged reader excludes from the listing opens
    /// as contributing nothing instead of failing. Never for queries.
    async fn open_generic_index_for_maintenance(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn Index>>;
    /// [`Self::open_scalar_index`] for maintenance; see
    /// [`Self::open_generic_index_for_maintenance`].
    async fn open_scalar_index_for_maintenance(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>>;
    /// [`Self::open_vector_index`] for maintenance; see
    /// [`Self::open_generic_index_for_maintenance`].
    async fn open_vector_index_for_maintenance(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndex>>;
    /// [`Self::open_logical_vector_index`] for maintenance: every segment of
    /// the name from the manifest, one the tagged reader excludes from the
    /// listing included (it opens as contributing nothing, to be rebuilt).
    async fn open_logical_vector_index_for_maintenance(
        &self,
        column: &str,
        name: &str,
    ) -> Result<LogicalVectorIndex>;
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

    /// Loads scalar-index planning information from an explicitly selected metadata set.
    async fn scalar_index_info_for_segments(
        &self,
        segments: &[IndexMetadata],
    ) -> Result<ScalarIndexInfo>;

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

/// The FRI identity that belongs in the vector cache keys before any remapping
/// is resolved: only a v0 history, whose remapper is applied while the index
/// is decoded, identifies cached content (see [`frag_reuse::fri_cache_id`]).
async fn v0_frag_reuse_cache_id(dataset: &Dataset) -> Option<Uuid> {
    load_all_indices(dataset)
        .await
        .ok()?
        .iter()
        .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME && idx.index_version == 0)
        .map(|idx| frag_reuse::v0_cache_identity(dataset, idx))
}

/// The FRI part of a cache key for content read alongside the dataset's
/// history: [`frag_reuse::v0_cache_identity`] for `index_version = 0`, or the
/// entry UUID for tagged history, which every change to that history replaces.
pub(crate) async fn frag_reuse_cache_scope(dataset: &Dataset) -> Option<Uuid> {
    load_all_indices(dataset)
        .await
        .ok()?
        .iter()
        .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
        .map(|idx| match idx.index_version {
            0 => frag_reuse::v0_cache_identity(dataset, idx),
            _ => idx.uuid,
        })
}

impl Dataset {
    #[instrument(level = "trace", skip_all)]
    pub(crate) async fn scalar_index_info_ignoring(
        &self,
        ignored_index_names: &HashSet<String>,
    ) -> Result<ScalarIndexInfo> {
        let indices = self.load_indices().await?;
        if !ignored_index_names.is_empty() && log::log_enabled!(log::Level::Debug) {
            let available_names: HashSet<&str> =
                indices.iter().map(|index| index.name.as_str()).collect();
            let mut unmatched_names: Vec<&str> = ignored_index_names
                .iter()
                .map(String::as_str)
                .filter(|name| !available_names.contains(name))
                .collect();
            if !unmatched_names.is_empty() {
                unmatched_names.sort_unstable();
                log::debug!("Ignored index names not found: {unmatched_names:?}");
            }
        }
        self.scalar_index_info_for_segments_ignoring(indices.as_ref(), ignored_index_names)
            .await
    }

    #[instrument(level = "trace", skip_all)]
    pub(crate) async fn scalar_index_info_for_segments_ignoring(
        &self,
        indices: &[IndexMetadata],
        ignored_index_names: &HashSet<String>,
    ) -> Result<ScalarIndexInfo> {
        let schema = self.schema();
        let mut indexed_fields = Vec::new();
        // (column, index_name) → union of every contributing IndexMetadata's
        // fragment_bitmap. Multiple entries can land here for delta-merged
        // indices that share a name. We only insert when every contributing
        // entry has a bitmap; if any are missing, we leave the entry absent
        // so the optimizer treats coverage as unknown.
        let mut fragment_bitmaps: HashMap<(String, String), Option<RoaringBitmap>> = HashMap::new();
        for index in indices.iter().filter(|idx| {
            if ignored_index_names.contains(&idx.name) {
                return false;
            }

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

    /// The segment's metadata for the given purpose. A query sees the derived
    /// listing only. Maintenance also reaches a registered segment the tagged
    /// reader left out of the listing (its coverage was withdrawn by an
    /// in-place rewrite, or its translation cannot be honored): it gets the
    /// stored metadata and opens the segment as contributing nothing, to
    /// rebuild or replace it.
    pub(crate) async fn load_index_with_purpose(
        &self,
        uuid: &Uuid,
        purpose: frag_reuse::OpenPurpose,
    ) -> Result<Option<IndexMetadata>> {
        if let Some(index) = self
            .load_indices()
            .await?
            .iter()
            .find(|idx| idx.uuid == *uuid)
        {
            return Ok(Some(index.clone()));
        }
        if purpose == frag_reuse::OpenPurpose::Query {
            return Ok(None);
        }
        Ok(load_all_indices(self)
            .await?
            .iter()
            .find(|idx| idx.uuid == *uuid && !is_system_index(idx) && index_is_usable(idx))
            .cloned())
    }

    pub(crate) async fn open_generic_index_with_purpose(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
        purpose: frag_reuse::OpenPurpose,
    ) -> Result<Arc<dyn Index>> {
        // Checking for cache existence is cheap so we just check the vector caches.
        // Scalar indices cache themselves inside `open_scalar_index` (the cache
        // key is a plugin detail), so there is no cheap scalar check here.
        let frag_reuse_uuid = v0_frag_reuse_cache_id(self).await;

        // Check sized cache for IvfIndexState (v2+ indices).
        let state_key = IvfIndexStateCacheKey::new(uuid, frag_reuse_uuid.as_ref());
        if self.index_cache.get_with_key(&state_key).await.is_some() {
            // Reconstruct via open_vector_index which will hit the same sized key.
            let index = self
                .open_vector_index_with_purpose(column, uuid, metrics, purpose)
                .await?;
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
            .load_index_with_purpose(uuid, purpose)
            .await?
            .ok_or_else(|| Error::index(format!("Index with id {} does not exist", uuid)))?;

        // Declared type first, and the legacy file name only for segments that
        // predate details: an index awaiting training declares itself a vector
        // index and has no file, so a file-based answer would send it to the
        // scalar reader.
        let is_vector_index = if index_meta.index_details.is_some() {
            segment_has_vector_details(&index_meta)
        } else if let Some(files) = &index_meta.files {
            files.iter().any(|f| f.path == INDEX_FILE_NAME)
        } else {
            let index_dir = self.indice_files_dir(&index_meta)?;
            let index_file = index_dir
                .clone()
                .join(uuid.to_string())
                .join(INDEX_FILE_NAME);
            let object_store = self.object_store_for_index(&index_meta).await?;
            object_store.exists(&index_file).await?
        };

        if is_vector_index {
            let index = self
                .open_vector_index_with_purpose(column, uuid, metrics, purpose)
                .await?;
            Ok(index.as_index())
        } else {
            let index = self
                .open_scalar_index_with_purpose(column, uuid, metrics, purpose)
                .await?;
            Ok(index.as_index())
        }
    }

    #[instrument(level = "debug", skip_all)]
    pub(crate) async fn open_scalar_index_with_purpose(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
        purpose: frag_reuse::OpenPurpose,
    ) -> Result<Arc<dyn ScalarIndex>> {
        // Caching (including the choice of in-memory vs. serializable state) is
        // a plugin implementation detail handled inside `scalar::open_scalar_index`.
        let index_meta = self
            .load_index_with_purpose(uuid, purpose)
            .await?
            .ok_or_else(|| Error::index(format!("Index with id {} does not exist", uuid)))?;

        scalar::open_scalar_index_with_plan(self, column, &index_meta, None, purpose, metrics).await
    }

    pub(crate) async fn open_vector_index_with_purpose(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
        purpose: frag_reuse::OpenPurpose,
    ) -> Result<Arc<dyn VectorIndex>> {
        let index_meta = self
            .load_index_with_purpose(uuid, purpose)
            .await?
            .ok_or_else(|| Error::index(format!("Index with id {} does not exist", uuid)))?;
        self.open_vector_index_from_metadata_with_purpose(column, &index_meta, metrics, purpose)
            .await
    }

    pub(crate) async fn open_vector_index_from_metadata_with_purpose(
        &self,
        column: &str,
        index_meta: &IndexMetadata,
        metrics: &dyn MetricsCollector,
        purpose: frag_reuse::OpenPurpose,
    ) -> Result<Arc<dyn VectorIndex>> {
        self.open_vector_index_from_metadata_with_plan(column, index_meta, None, purpose, metrics)
            .await
    }

    /// [`Self::open_vector_index_from_metadata`] with the reuse translation
    /// settled by the caller: `staged` is the plan of a segment the manifest
    /// does not list (its merge planned it), `None` looks the segment up in
    /// the snapshot plan.
    pub(crate) async fn open_vector_index_from_metadata_with_plan(
        &self,
        column: &str,
        index_meta: &IndexMetadata,
        staged: Option<&frag_reuse::SegmentRemappingPlan>,
        purpose: frag_reuse::OpenPurpose,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndex>> {
        let uuid = &index_meta.uuid;
        let object_store = self.object_store_for_index(index_meta).await?;
        let index_dir = self.indice_files_dir(index_meta)?;
        // Hold the shared allocation before cache accesses can evict it.
        // The handle is dropped if this index does not use resident columns.
        // Physical file keys also let shallow clones share immutable columns.
        // Translated partitions remain scoped to the dataset and FRI below.
        let file_cache = self.session.index_cache.with_key_prefix("ivf-file");
        let resident_columns_handle = Some(ResidentColumns::shared(&aux_file_key(
            &object_store,
            &index_dir,
            uuid,
        )));
        // What the session declares about the index, for whichever IVF open
        // or reconstruction below runs, and where a reconstruction adds the
        // I/O of a resident store load.
        let open_context = IvfOpenContext {
            rabit_scan_factory: self.session.rabit_scan_factory.clone(),
            origin_latency_hint: self.session.index_origin_latency(),
            resident_columns: self.session.index_resident_columns(),
            file_cache: Some(file_cache),
            resident_columns_handle,
            io_stats: metrics.io_stats(),
        };
        let resolved =
            frag_reuse::open_row_id_remapping_with_plan(self, index_meta, staged, purpose, metrics)
                .await?;
        // The index state (paths, model, quantizer metadata) and a legacy
        // whole-index entry embed no translated rows: they live in the plain
        // per-index namespace and stay warm across appends and unrelated
        // rewrites. Partitions decoded through a translating remapper live in
        // the translated namespace of `query_cache`.
        let frag_reuse_uuid = frag_reuse::fri_cache_id(&resolved).copied();
        let query_cache = frag_reuse::scoped_index_cache(self, &resolved);
        let remapping = resolved.map(|(_, remapping)| remapping);

        // Check sized cache first (v2+ indices with serializable state).
        let state_key = IvfIndexStateCacheKey::new(uuid, frag_reuse_uuid.as_ref());
        if let Some(entry) = self.index_cache.get_with_key(&state_key).await {
            log::debug!("Found IvfIndexState in cache uuid: {}", uuid);
            let partition_cache = query_cache.for_index(uuid, frag_reuse_uuid.as_ref());
            return entry
                .0
                .reconstruct(
                    object_store,
                    self.metadata_cache.as_ref(),
                    partition_cache,
                    remapping,
                    open_context,
                )
                .await;
        }

        // Fallback: in-memory cache for legacy indices.
        let cache_key = LegacyVectorIndexCacheKey::new(uuid, frag_reuse_uuid.as_ref());
        if let Some(cached) = self.index_cache.get_with_key(&cache_key).await {
            return Ok(cached.0.clone());
        }

        // Only legacy vector file readers consume the V1 handle. A tagged
        // history is handled by the shared remapper above, including identity.
        let has_tagged_history = load_all_indices(self)
            .await?
            .iter()
            .any(|index| index.name == FRAG_REUSE_INDEX_NAME && index.index_version != 0);
        let frag_reuse_index = if has_tagged_history {
            None
        } else {
            self.open_frag_reuse_index(metrics).await?
        };
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

        // Legacy vector indexes are cached whole and can outlive the files they
        // were opened from: cleanup unlinks old index files while the cache still
        // holds an entry for them. Anything stored in the cached index must
        // therefore reopen the file per read instead of pinning a descriptor.
        let lazy_reader = vector::open_cache_reader(
            object_store.clone(),
            &index_file,
            INDEX_FILE_NAME,
            &file_sizes,
        )
        .await?;

        // Namespace the index cache by the UUID of the index. v2+ partition
        // entries are store-free and remain reusable across object-store
        // generations alongside their serializable state.
        let index_cache = query_cache.for_index(uuid, frag_reuse_uuid.as_ref());

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
                            lazy_reader,
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
                    lazy_reader,
                    Some(&self.metadata_cache.file_metadata_cache(&index_file)),
                )
                .await?;
                let idx = vector::open_vector_index_v2(
                    Arc::new(self.clone()),
                    column,
                    index_meta,
                    reader,
                    frag_reuse_index,
                )
                .await?;
                Ok((idx, None::<IvfStateEntryBox>))
            }

            (0, 3) | (2, _) => {
                let index_metadata = self
                    .read_vector_index_metadata(&object_store, &index_file, &file_sizes)
                    .await?;

                // Resolve the column name and field
                let (field_path, field) = resolve_index_column(self.schema(), index_meta, column)?;

                let (_, element_type) = get_vector_type(self.schema(), &field_path)?;

                info!(target: TRACE_IO_EVENTS, index_uuid=%uuid, r#type=IO_TYPE_OPEN_VECTOR, version="0.3", index_type=index_metadata.index_type);

                match index_metadata.index_type.as_str() {
                    "IVF_FLAT" => match element_type {
                        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                            let ivf = IVFIndex::<FlatIndex, FlatQuantizer>::try_new(
                                object_store.clone(),
                                index_dir,
                                uuid.to_owned(),
                                remapping,
                                self.metadata_cache.as_ref(),
                                index_cache,
                                file_sizes,
                                open_context,
                            )
                            .await?;
                            Ok(wrap_ivf(ivf))
                        }
                        DataType::UInt8 => {
                            let ivf = IVFIndex::<FlatIndex, FlatBinQuantizer>::try_new(
                                object_store.clone(),
                                index_dir,
                                uuid.to_owned(),
                                remapping,
                                self.metadata_cache.as_ref(),
                                index_cache,
                                file_sizes,
                                open_context,
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
                            remapping,
                            self.metadata_cache.as_ref(),
                            index_cache,
                            file_sizes,
                            open_context,
                        )
                        .await?;
                        Ok(wrap_ivf(ivf))
                    }

                    "IVF_SQ" => {
                        let ivf = IVFIndex::<FlatIndex, ScalarQuantizer>::try_new(
                            object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            remapping,
                            self.metadata_cache.as_ref(),
                            index_cache,
                            file_sizes,
                            open_context,
                        )
                        .await?;
                        Ok(wrap_ivf(ivf))
                    }

                    "IVF_RQ" => {
                        let ivf = IVFIndex::<FlatIndex, RabitQuantizer>::try_new(
                            object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            remapping,
                            self.metadata_cache.as_ref(),
                            index_cache,
                            file_sizes,
                            open_context,
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
                                remapping,
                                self.metadata_cache.as_ref(),
                                index_cache,
                                file_sizes,
                                open_context,
                            )
                            .await?;
                            Ok(wrap_ivf(ivf))
                        }
                        _ => {
                            let ivf = IVFIndex::<HNSW, FlatQuantizer>::try_new(
                                object_store.clone(),
                                index_dir,
                                uuid.to_owned(),
                                remapping,
                                self.metadata_cache.as_ref(),
                                index_cache,
                                file_sizes,
                                open_context,
                            )
                            .await?;
                            Ok(wrap_ivf(ivf))
                        }
                    },

                    "IVF_HNSW_RQ" => {
                        let ivf = IVFIndex::<HNSW, RabitQuantizer>::try_new(
                            object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            remapping,
                            self.metadata_cache.as_ref(),
                            index_cache,
                            file_sizes,
                            open_context,
                        )
                        .await?;
                        Ok(wrap_ivf(ivf))
                    }

                    "IVF_HNSW_SQ" => {
                        let ivf = IVFIndex::<HNSW, ScalarQuantizer>::try_new(
                            object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            remapping,
                            self.metadata_cache.as_ref(),
                            index_cache,
                            file_sizes,
                            open_context,
                        )
                        .await?;
                        Ok(wrap_ivf(ivf))
                    }

                    "IVF_HNSW_PQ" => {
                        let ivf = IVFIndex::<HNSW, ProductQuantizer>::try_new(
                            object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            remapping,
                            self.metadata_cache.as_ref(),
                            index_cache,
                            file_sizes,
                            open_context,
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

    /// The index metadata in the schema of a vector index file of the
    /// current format, `(0, 3)` or `(2, _)`: the type the file opens as and
    /// its distance type, both read before any centroid or quantizer.
    pub(crate) async fn read_vector_index_metadata(
        &self,
        object_store: &Arc<lance_io::object_store::ObjectStore>,
        index_file: &object_store::path::Path,
        file_sizes: &HashMap<String, u64>,
    ) -> Result<lance_index::IndexMetadata> {
        let scheduler = ScanScheduler::new(
            object_store.clone(),
            SchedulerConfig::max_bandwidth(object_store),
        );
        let cached_size = file_sizes
            .get(INDEX_FILE_NAME)
            .map(|&size| CachedFileSize::new(size))
            .unwrap_or_else(CachedFileSize::unknown);
        let file = scheduler.open_file(index_file, &cached_size).await?;
        let reader = lance_file::reader::FileReader::try_open(
            file,
            None,
            Default::default(),
            &self.metadata_cache.file_metadata_cache(index_file),
            FileReaderOptions::default(),
        )
        .await?;
        let index_metadata = reader
            .schema()
            .metadata
            .get(INDEX_METADATA_SCHEMA_KEY)
            .ok_or(Error::index("Index Metadata not found".to_owned()))?;
        Ok(serde_json::from_str(index_metadata)?)
    }
}

#[async_trait]
impl DatasetIndexInternalExt for Dataset {
    async fn open_generic_index(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn Index>> {
        self.open_generic_index_with_purpose(column, uuid, metrics, frag_reuse::OpenPurpose::Query)
            .await
    }

    async fn open_scalar_index(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        self.open_scalar_index_with_purpose(column, uuid, metrics, frag_reuse::OpenPurpose::Query)
            .await
    }

    async fn open_vector_index(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndex>> {
        self.open_vector_index_with_purpose(column, uuid, metrics, frag_reuse::OpenPurpose::Query)
            .await
    }

    async fn open_vector_index_from_metadata(
        &self,
        column: &str,
        index_meta: &IndexMetadata,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndex>> {
        self.open_vector_index_from_metadata_with_purpose(
            column,
            index_meta,
            metrics,
            frag_reuse::OpenPurpose::Query,
        )
        .await
    }

    async fn open_generic_index_for_maintenance(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn Index>> {
        self.open_generic_index_with_purpose(
            column,
            uuid,
            metrics,
            frag_reuse::OpenPurpose::Maintenance,
        )
        .await
    }

    async fn open_scalar_index_for_maintenance(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        self.open_scalar_index_with_purpose(
            column,
            uuid,
            metrics,
            frag_reuse::OpenPurpose::Maintenance,
        )
        .await
    }

    async fn open_vector_index_for_maintenance(
        &self,
        column: &str,
        uuid: &Uuid,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndex>> {
        self.open_vector_index_with_purpose(
            column,
            uuid,
            metrics,
            frag_reuse::OpenPurpose::Maintenance,
        )
        .await
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

    async fn open_logical_vector_index_for_maintenance(
        &self,
        column: &str,
        name: &str,
    ) -> Result<LogicalVectorIndex> {
        // Every segment of the name the manifest registers, each with the
        // metadata its open sees: the listed one (derived coverage) for a
        // listed segment, the stored one for a segment the reader excludes.
        let registered: Vec<Uuid> = load_all_indices(self)
            .await?
            .iter()
            .filter(|metadata| {
                metadata.name == name && !is_system_index(metadata) && index_is_usable(metadata)
            })
            .map(|metadata| metadata.uuid)
            .collect();
        let mut metadatas = Vec::with_capacity(registered.len());
        for uuid in registered {
            if let Some(metadata) = self
                .load_index_with_purpose(&uuid, frag_reuse::OpenPurpose::Maintenance)
                .await?
            {
                metadatas.push(metadata);
            }
        }
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
                .open_vector_index_for_maintenance(column, &metadata.uuid, &NoOpMetricsCollector)
                .await?;
            segments.push((metadata, index));
        }
        LogicalVectorIndex::try_new(name.to_string(), column.to_string(), segments)
    }

    async fn open_frag_reuse_index(
        &self,
        metrics: &dyn MetricsCollector,
    ) -> Result<Option<Arc<CompactFragReuseIndex>>> {
        if let Some(frag_reuse_index_meta) = find_frag_reuse_index_meta(self).await? {
            if frag_reuse_index_meta.index_version != 0 {
                // Version-1 consumers are installed separately. The planner excludes
                // affected segments; independent segments need no legacy remapper.
                // Maintenance is rejected before entering the legacy write path.
                return match frag_reuse_index_meta.index_version {
                    // None means this legacy API cannot provide a reader for FRI index
                    // version 1; it does not mean the dataset has no FRI.
                    // Callers must not use this result to authorize index maintenance.
                    1 => Ok(None),
                    version => Err(Error::not_supported(format!(
                        "FRI index_version {version} is unsupported. Please upgrade to a newer version",
                    ))),
                };
            }
            let frag_reuse_uuid = frag_reuse_index_meta.uuid;
            let identity = frag_reuse::v0_cache_identity(self, &frag_reuse_index_meta);
            let frag_reuse_key = FragReuseIndexKey { uuid: &identity };

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

        let frag_reuse_uuid = frag_reuse_cache_scope(self).await;
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
        // Bookkeeping-only lookup: the uuid comes off the raw manifest listing,
        // so it must not drive the query reader's coverage rewrite (nor parse the
        // mapping, nor fail on a corrupt ledger). Use load_all_indices; the FRI
        // entry itself is identical in the raw and derived listings.
        if let Ok(indices) = load_all_indices(self).await {
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
        self.scalar_index_info_ignoring(&HashSet::new()).await
    }

    #[instrument(level = "trace", skip_all)]
    async fn scalar_index_info_for_segments(
        &self,
        segments: &[IndexMetadata],
    ) -> Result<ScalarIndexInfo> {
        self.scalar_index_info_for_segments_ignoring(segments, &HashSet::new())
            .await
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

#[cfg(test)]
mod tests;
