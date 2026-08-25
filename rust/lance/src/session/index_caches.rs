// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Caches for Lance indices. They are organized in a hierarchical manner to
//! avoid collisions.
//!
//!  GlobalIndexCache
//!     │
//!     ├─► DSIndexCache (prefixed by dataset URI)
//!     │    │
//!     └────┴──► Index-specific cache (prefixed by index UUID and FRI UUID)

use std::{borrow::Cow, collections::HashSet, ops::Deref, sync::Arc};

use arrow_array::{
    cast::AsArray,
    types::{Float16Type, Float32Type, Float64Type},
};
use arrow_schema::DataType;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_index::frag_reuse::{CompactFragReuseIndex, FragReuseIndexDetails};
use lance_index::vector::quantizer::QuantizationType;
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::vector::{ApproxMode, Query};
use lance_linalg::distance::DistanceType;
use lance_select::RowAddrMask;
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use crate::{Error, Result};

/// A type-safe wrapper around a LanceCache that enforces namespaces for index data.
pub struct GlobalIndexCache(pub(super) LanceCache);

impl GlobalIndexCache {
    pub fn for_dataset(&self, uri: &str) -> DSIndexCache {
        // Create a sub-cache for the dataset by adding the URI as a key prefix.
        // This prevents collisions between different datasets.
        DSIndexCache(self.0.with_key_prefix(uri))
    }
}

impl Clone for GlobalIndexCache {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Deref for GlobalIndexCache {
    type Target = LanceCache;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DeepSizeOf for GlobalIndexCache {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}

/// A type-safe wrapper around a LanceCache that enforces namespaces and keys
/// for dataset-specific index data.
pub struct DSIndexCache(pub(crate) LanceCache);

impl Deref for DSIndexCache {
    type Target = LanceCache;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DSIndexCache {
    /// Create an index-specific cache with the given UUID prefix.
    pub fn for_index(&self, uuid: &Uuid, fri_uuid: Option<&Uuid>) -> LanceCache {
        let mut uuid_buffer = Uuid::encode_buffer();
        let cache = self
            .0
            .with_key_prefix(uuid.as_hyphenated().encode_lower(&mut uuid_buffer));
        if let Some(fri_uuid) = fri_uuid {
            // If a FRI UUID is provided, use it to create a more specific cache key.
            let mut fri_uuid_buffer = Uuid::encode_buffer();
            cache.with_key_prefix(fri_uuid.as_hyphenated().encode_lower(&mut fri_uuid_buffer))
        } else {
            // Otherwise, just use the index UUID as the key prefix.
            cache
        }
    }
}

pub(crate) fn write_index_identity(builder: &mut KeyBuilder, uuid: &Uuid, fri_uuid: Option<&Uuid>) {
    builder.write_fixed_bytes(uuid.as_bytes());
    if let Some(fri_uuid) = fri_uuid {
        builder.write_some();
        builder.write_fixed_bytes(fri_uuid.as_bytes());
    } else {
        builder.write_none();
    }
}

/// One candidate stored by the experimental vector-results cache.
///
/// The offset is meaningful only inside `partition_id` of the exact index
/// segment identified by [`VectorResultsCacheIdentity`]. It must still be
/// bounds-checked against the loaded partition before use.
#[derive(Clone, Copy, Debug, DeepSizeOf, Eq, Hash, PartialEq)]
pub struct CachedVectorCandidate {
    partition_id: u32,
    offset_in_partition: u32,
}

impl CachedVectorCandidate {
    /// Create a partition-local candidate identity.
    pub fn new(partition_id: u32, offset_in_partition: u32) -> Self {
        Self {
            partition_id,
            offset_in_partition,
        }
    }

    /// Return the IVF partition containing this candidate.
    pub fn partition_id(&self) -> u32 {
        self.partition_id
    }

    /// Return the candidate's local offset inside its IVF partition.
    pub fn offset_in_partition(&self) -> u32 {
        self.offset_in_partition
    }
}

/// Domain-separates exact query fingerprints from other BLAKE3 uses.
const VECTOR_RESULTS_QUERY_FINGERPRINT_CONTEXT: &str =
    "lance.vector-results-cache-query-fingerprint.v1";

/// Complete compatibility identity for one reusable vector candidate pool.
///
/// The exact query values are represented by a cryptographic fingerprint. A
/// candidate pool can therefore be reused only by the same query bit pattern;
/// the ordered IVF partition list remains in the identity to bind the pool to
/// its search shape. Unsupported query forms are rejected by [`Self::try_new`]
/// instead of being represented by a partial key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VectorResultsCacheIdentity {
    store_identity: String,
    dataset_read_identity: String,
    dataset_version: u64,
    index_uuid: Uuid,
    frag_reuse_uuid: Option<Uuid>,
    index_dataset_version: u64,
    index_version: i32,
    index_base_id: Option<u32>,
    index_fields: Vec<i32>,
    column: String,
    metric_variant: u32,
    sub_index_variant: u32,
    quantization_variant: u32,
    approx_mode_variant: u32,
    vector_type_variant: u32,
    dimension: u32,
    query_fingerprint: [u8; 32],
    nprobes: u32,
    partition_ids: Vec<u32>,
    result_limit: u32,
    candidate_limit: u32,
}

impl DeepSizeOf for VectorResultsCacheIdentity {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.store_identity.deep_size_of_children(context)
            + self.dataset_read_identity.deep_size_of_children(context)
            + self.index_fields.deep_size_of_children(context)
            + self.column.deep_size_of_children(context)
            + self.partition_ids.deep_size_of_children(context)
    }
}

impl VectorResultsCacheIdentity {
    /// Build a fail-closed cache identity for the initial supported query scope.
    ///
    /// `dataset_read_identity` must identify the exact manifest being read, not
    /// merely the dataset URI. `has_prefilter` represents a user/scalar-index
    /// prefilter; ordinary deletion filtering is safe because the exact dataset
    /// read identity and version are included in the key.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        store_identity: &str,
        dataset_read_identity: &str,
        dataset_version: u64,
        index: &IndexMetadata,
        frag_reuse_uuid: Option<Uuid>,
        query: &Query,
        index_metric: DistanceType,
        sub_index_type: SubIndexType,
        quantization_type: QuantizationType,
        partition_ids: &[u32],
        candidate_limit: usize,
        has_prefilter: bool,
        has_overlay: bool,
    ) -> Result<Self> {
        if store_identity.is_empty() {
            return Err(Error::invalid_input(
                "vector results cache requires a non-empty object-store identity",
            ));
        }
        if dataset_read_identity.is_empty() {
            return Err(Error::invalid_input(
                "vector results cache requires an exact dataset read identity",
            ));
        }
        if index.fragment_bitmap.is_none() {
            return Err(Error::not_supported(format!(
                "vector results cache requires known fragment coverage for index {}",
                index.uuid
            )));
        }
        if index.index_details.is_none() {
            return Err(Error::not_supported(format!(
                "vector results cache requires current index details for index {}",
                index.uuid
            )));
        }
        if index.fields.is_empty() {
            return Err(Error::invalid_input(format!(
                "vector results cache cannot identify an index with no fields: {}",
                index.uuid
            )));
        }
        if has_prefilter {
            return Err(Error::not_supported(
                "vector results cache does not support prefiltered queries",
            ));
        }
        if has_overlay {
            return Err(Error::not_supported(
                "vector results cache does not support data overlays",
            ));
        }
        if query.lower_bound.is_some() || query.upper_bound.is_some() {
            return Err(Error::not_supported(
                "vector results cache does not support distance bounds",
            ));
        }
        if query.ef.is_some() {
            return Err(Error::not_supported(
                "vector results cache does not support HNSW query parameters",
            ));
        }
        if !query.use_index {
            return Err(Error::invalid_input(
                "vector results cache requires index search to be enabled",
            ));
        }
        if query.column.is_empty() {
            return Err(Error::invalid_input(
                "vector results cache requires a non-empty vector column",
            ));
        }
        if query.key.is_empty() || query.key.null_count() != 0 {
            return Err(Error::invalid_input(format!(
                "vector results cache requires a non-empty, non-null query vector, got length {} with {} nulls",
                query.key.len(),
                query.key.null_count()
            )));
        }

        let mut query_hasher =
            blake3::Hasher::new_derive_key(VECTOR_RESULTS_QUERY_FINGERPRINT_CONTEXT);
        let (vector_type_variant, is_finite) = match query.key.data_type() {
            DataType::Float16 => {
                let values = query.key.as_primitive::<Float16Type>().values();
                query_hasher.update(values.inner().as_slice());
                (0, values.iter().all(|value| value.is_finite()))
            }
            DataType::Float32 => {
                let values = query.key.as_primitive::<Float32Type>().values();
                query_hasher.update(values.inner().as_slice());
                (1, values.iter().all(|value| value.is_finite()))
            }
            DataType::Float64 => {
                let values = query.key.as_primitive::<Float64Type>().values();
                query_hasher.update(values.inner().as_slice());
                (2, values.iter().all(|value| value.is_finite()))
            }
            data_type => {
                return Err(Error::not_supported(format!(
                    "vector results cache supports one float vector, got query type {data_type}"
                )));
            }
        };
        if !is_finite {
            return Err(Error::invalid_input(
                "vector results cache requires a finite query vector",
            ));
        }
        let dimension = u32::try_from(query.key.len()).map_err(|_| {
            Error::invalid_input(format!(
                "query vector dimension {} exceeds the vector results cache limit",
                query.key.len()
            ))
        })?;
        let query_fingerprint = *query_hasher.finalize().as_bytes();

        let metric_type = query.metric_type.unwrap_or(index_metric);
        if metric_type != index_metric {
            return Err(Error::invalid_input(format!(
                "query metric {metric_type} does not match index metric {index_metric} for vector results cache"
            )));
        }
        let metric_variant = match metric_type {
            DistanceType::L2 => 0,
            DistanceType::Cosine => 1,
            DistanceType::Dot => 2,
            DistanceType::Hamming => {
                return Err(Error::not_supported(
                    "vector results cache does not support Hamming distance",
                ));
            }
        };
        let sub_index_variant = match sub_index_type {
            SubIndexType::Flat => 0,
            SubIndexType::Hnsw => {
                return Err(Error::not_supported(
                    "vector results cache initially supports only flat IVF sub-indices",
                ));
            }
        };
        let quantization_variant = match quantization_type {
            QuantizationType::Scalar => 0,
            QuantizationType::Rabit => {
                if query.approx_mode != ApproxMode::Accurate {
                    return Err(Error::not_supported(format!(
                        "vector results cache supports RQ only with ApproxMode::Accurate; got {:?}",
                        query.approx_mode
                    )));
                }
                1
            }
            other => {
                return Err(Error::not_supported(format!(
                    "vector results cache does not support {other} quantization"
                )));
            }
        };
        let approx_mode_variant = match query.approx_mode {
            ApproxMode::Fast => 0,
            ApproxMode::Normal => 1,
            ApproxMode::Accurate => 2,
        };

        let Some(maximum_nprobes) = query.maximum_nprobes else {
            return Err(Error::not_supported(
                "vector results cache requires a fixed maximum_nprobes",
            ));
        };
        if query.minimum_nprobes != maximum_nprobes {
            return Err(Error::not_supported(format!(
                "vector results cache requires fixed nprobes, got minimum_nprobes={} and maximum_nprobes={maximum_nprobes}",
                query.minimum_nprobes
            )));
        }
        if maximum_nprobes == 0 || maximum_nprobes != partition_ids.len() {
            return Err(Error::invalid_input(format!(
                "vector results cache nprobes {maximum_nprobes} does not match {} searched partitions",
                partition_ids.len()
            )));
        }
        if partition_ids.iter().copied().collect::<HashSet<_>>().len() != partition_ids.len() {
            return Err(Error::invalid_input(format!(
                "vector results cache requires unique partition ids, got {partition_ids:?}"
            )));
        }
        let nprobes = u32::try_from(maximum_nprobes).map_err(|_| {
            Error::invalid_input(format!(
                "nprobes {maximum_nprobes} exceeds the vector results cache limit"
            ))
        })?;

        let refine_factor = query.refine_factor.unwrap_or(1) as usize;
        let result_limit = query.k.checked_mul(refine_factor).ok_or_else(|| {
            Error::invalid_input(format!(
                "vector results cache result limit overflows: k={} refine_factor={refine_factor}",
                query.k
            ))
        })?;
        if result_limit == 0 || candidate_limit < result_limit {
            return Err(Error::invalid_input(format!(
                "vector results cache candidate limit {candidate_limit} must be at least the requested result limit {result_limit}"
            )));
        }
        let result_limit = u32::try_from(result_limit).map_err(|_| {
            Error::invalid_input(format!(
                "result limit {result_limit} exceeds the vector results cache limit"
            ))
        })?;
        let candidate_limit = u32::try_from(candidate_limit).map_err(|_| {
            Error::invalid_input(format!(
                "candidate limit {candidate_limit} exceeds the vector results cache limit"
            ))
        })?;

        Ok(Self {
            store_identity: store_identity.to_owned(),
            dataset_read_identity: dataset_read_identity.to_owned(),
            dataset_version,
            index_uuid: index.uuid,
            frag_reuse_uuid,
            index_dataset_version: index.dataset_version,
            index_version: index.index_version,
            index_base_id: index.base_id,
            index_fields: index.fields.clone(),
            column: query.column.clone(),
            metric_variant,
            sub_index_variant,
            quantization_variant,
            approx_mode_variant,
            vector_type_variant,
            dimension,
            query_fingerprint,
            nprobes,
            partition_ids: partition_ids.to_vec(),
            result_limit,
            candidate_limit,
        })
    }

    /// Return the exact ordered partition list represented by this cache key.
    pub fn partition_ids(&self) -> &[u32] {
        &self.partition_ids
    }

    /// Return the maximum number of candidates stored in a compatible entry.
    pub fn candidate_limit(&self) -> usize {
        self.candidate_limit as usize
    }

    /// Return the number of scored candidates emitted for this query shape.
    pub fn result_limit(&self) -> usize {
        self.result_limit as usize
    }
}

/// In-memory candidate pool stored under a [`VectorResultsCacheIdentity`].
///
/// No codec is registered yet, so this does not create a persistent cache or a
/// file-format compatibility surface.
#[derive(Clone, Debug, DeepSizeOf)]
pub struct VectorResultsCacheEntry {
    identity: VectorResultsCacheIdentity,
    candidates: Vec<CachedVectorCandidate>,
}

impl VectorResultsCacheEntry {
    /// Create an entry after validating its candidate count and partitions.
    pub fn try_new(
        identity: VectorResultsCacheIdentity,
        candidates: Vec<CachedVectorCandidate>,
    ) -> Result<Self> {
        let entry = Self {
            identity,
            candidates,
        };
        if !entry.has_valid_shape() {
            return Err(Error::invalid_input(
                "vector results cache entry contains incompatible or duplicate candidates",
            ));
        }
        Ok(entry)
    }

    /// Return true only if the entry exactly matches the requested identity and
    /// all candidates satisfy the identity's structural constraints.
    pub fn is_compatible_with(&self, identity: &VectorResultsCacheIdentity) -> bool {
        self.identity == *identity && self.has_valid_shape()
    }

    /// Return the validated partition-local candidates.
    pub fn candidates(&self) -> &[CachedVectorCandidate] {
        &self.candidates
    }

    fn has_valid_shape(&self) -> bool {
        if self.candidates.len() > self.identity.candidate_limit() {
            return false;
        }
        let valid_partitions = self
            .identity
            .partition_ids()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let unique_candidates = self.candidates.iter().copied().collect::<HashSet<_>>();
        unique_candidates.len() == self.candidates.len()
            && self
                .candidates
                .iter()
                .all(|candidate| valid_partitions.contains(&candidate.partition_id()))
    }
}

impl CacheKey for VectorResultsCacheIdentity {
    type ValueType = VectorResultsCacheEntry;

    fn key(&self) -> Cow<'_, str> {
        let partition_ids = self
            .partition_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let base_id = self.index_base_id.map_or(-1, i64::from);
        Cow::Owned(format!(
            "{store_len}:{store}/{read_len}:{read}/{dataset_version}/{index_uuid}/{frag_reuse_uuid:?}/{index_dataset_version}/{index_version}/{base_id}/{index_fields:?}/{column_len}:{column}/{metric_variant}/{sub_index_variant}/{quantization_variant}/{approx_mode_variant}/{vector_type_variant}/{dimension}/{query_fingerprint:?}/{nprobes}/{partition_ids}/{result_limit}/{candidate_limit}",
            store_len = self.store_identity.len(),
            store = self.store_identity,
            read_len = self.dataset_read_identity.len(),
            read = self.dataset_read_identity,
            dataset_version = self.dataset_version,
            index_uuid = self.index_uuid,
            frag_reuse_uuid = self.frag_reuse_uuid,
            index_dataset_version = self.index_dataset_version,
            index_version = self.index_version,
            index_fields = self.index_fields,
            column_len = self.column.len(),
            column = self.column,
            metric_variant = self.metric_variant,
            sub_index_variant = self.sub_index_variant,
            quantization_variant = self.quantization_variant,
            approx_mode_variant = self.approx_mode_variant,
            vector_type_variant = self.vector_type_variant,
            dimension = self.dimension,
            query_fingerprint = self.query_fingerprint,
            nprobes = self.nprobes,
            result_limit = self.result_limit,
            candidate_limit = self.candidate_limit,
        ))
    }

    fn type_name() -> &'static str {
        "VectorResultsCacheEntry"
    }

    fn stable_type_id() -> &'static str {
        "lance.VectorResultsCacheEntry"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.vector-results-cache-key", 2)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(&self.store_identity);
        builder.write_str(&self.dataset_read_identity);
        builder.write_u64(self.dataset_version);
        write_index_identity(builder, &self.index_uuid, self.frag_reuse_uuid.as_ref());
        builder.write_u64(self.index_dataset_version);
        builder.write_i32(self.index_version);
        if let Some(base_id) = self.index_base_id {
            builder.write_some();
            builder.write_u32(base_id);
        } else {
            builder.write_none();
        }
        builder.write_sequence_len(self.index_fields.len() as u64);
        for field_id in &self.index_fields {
            builder.write_i32(*field_id);
        }
        builder.write_str(&self.column);
        builder.write_variant(self.metric_variant);
        builder.write_variant(self.sub_index_variant);
        builder.write_variant(self.quantization_variant);
        builder.write_variant(self.approx_mode_variant);
        builder.write_variant(self.vector_type_variant);
        builder.write_u32(self.dimension);
        builder.write_fixed_bytes(&self.query_fingerprint);
        builder.write_u32(self.nprobes);
        builder.write_sequence_len(self.partition_ids.len() as u64);
        for partition_id in &self.partition_ids {
            builder.write_u32(*partition_id);
        }
        builder.write_u32(self.result_limit);
        builder.write_u32(self.candidate_limit);
    }
}

// Cache key types for type-safe cache access

#[derive(Debug)]
pub struct FragReuseIndexKey<'a> {
    pub uuid: &'a Uuid,
}

impl CacheKey for FragReuseIndexKey<'_> {
    type ValueType = CompactFragReuseIndex;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("frag_reuse/{}", self.uuid))
    }

    fn type_name() -> &'static str {
        "FragReuseIndex"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fragment-reuse-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.uuid.as_bytes());
    }
}

/// Decoded fragment reuse details of an external `details.binpb`, keyed by where the
/// file lives: a shallow clone's entry resolves to the source dataset's file. Each file
/// is written once, under a fresh entry UUID. A hit does not check the file still exists.
#[derive(Debug)]
pub struct FragReuseDetailsKey<'a> {
    pub store_identity: &'a str,
    pub path: &'a object_store::path::Path,
    pub offset: u64,
    pub size: u64,
}

impl CacheKey for FragReuseDetailsKey<'_> {
    type ValueType = FragReuseIndexDetails;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "frag_reuse_details/{}:{}/{}/{}/{}",
            self.store_identity.len(),
            self.store_identity,
            self.path,
            self.offset,
            self.size
        ))
    }

    fn type_name() -> &'static str {
        "FragReuseIndexDetails"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fragment-reuse-details-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(self.store_identity);
        builder.write_str(self.path.as_ref());
        builder.write_u64(self.offset);
        builder.write_u64(self.size);
    }
}

/// Restricted deletion mask of a vector index segment at one dataset version
/// (`None`: the segment needs no restriction). Building it walks every
/// fragment of the dataset, hence the cache entry.
#[derive(Debug)]
pub struct SegmentOwnershipMaskKey<'a> {
    pub uuid: &'a Uuid,
    pub version: u64,
    pub store_identity: &'a str,
    pub e_tag: Option<&'a str>,
}

impl CacheKey for SegmentOwnershipMaskKey<'_> {
    type ValueType = Option<Arc<RowAddrMask>>;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "segment_mask/{}/{}:{}/{}/{}",
            self.uuid,
            self.store_identity.len(),
            self.store_identity,
            self.version,
            self.e_tag.unwrap_or("")
        ))
    }

    fn type_name() -> &'static str {
        "SegmentOwnershipMask"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.segment-ownership-mask-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.uuid.as_bytes());
        builder.write_str(self.store_identity);
        builder.write_u64(self.version);
        match self.e_tag {
            Some(e_tag) => {
                builder.write_some();
                builder.write_str(e_tag);
            }
            None => builder.write_none(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IndexMetadataKey<'a> {
    pub version: u64,
    pub store_identity: &'a str,
    pub e_tag: Option<&'a str>,
}

impl CacheKey for IndexMetadataKey<'_> {
    type ValueType = Vec<IndexMetadata>;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "{}:{}/{}/{}",
            self.store_identity.len(),
            self.store_identity,
            self.version,
            self.e_tag.unwrap_or("")
        ))
    }

    fn type_name() -> &'static str {
        "Vec<IndexMetadata>"
    }

    fn schema() -> CacheKeySchema {
        // v2 holds every index the manifest names; v1 held only the ones the
        // writing build could read. The fields are identical, so on a persistent
        // backend shared with another release nothing but this version stops each
        // build from reading the other's entry as its own meaning.
        CacheKeySchema::new("lance.index.metadata-key", 2)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(self.store_identity);
        builder.write_u64(self.version);
        match self.e_tag {
            Some(e_tag) => {
                builder.write_some();
                builder.write_str(e_tag);
            }
            None => builder.write_none(),
        }
    }

    fn codec() -> Option<lance_core::cache::CacheCodec> {
        Some(lance_table::format::index_metadata_codec())
    }
}

/// Cache key for the query-time index listing DERIVED from a v1 (tagged) FRI:
/// the coverage-rewritten, filtered listing produced by `frag_reuse_reader`.
///
/// Separate namespace from [`IndexMetadataKey`], which caches the raw manifest
/// listing. This one caches the per-snapshot result of the coverage backtrack so
/// `load_indices` does not recompute `segment_coverage` and the per-index bitmap
/// rewrites on every query-planning / merge_insert call. Keyed by the same
/// snapshot identity: a different table (`store_identity`), a different snapshot
/// (`version`), or any index add/replace (which commits a new manifest and bumps
/// `version`/`e_tag`) yields a different key, so a stale derived listing is never
/// reused.
///
/// This cache is MEMORY-ONLY (no `codec`): the derived listing depends on the
/// READER's translation capability, not only on the snapshot. Different builds
/// admit different segments for the same snapshot (a build that cannot translate
/// a segment excludes it; one that can includes it), so a persisted entry
/// reused across builds could serve a listing the current reader must not use.
/// Snapshot identity does not capture reader capability, so we never persist
/// this value; it is cheap to recompute per process.
#[derive(Clone, Copy, Debug)]
pub struct DerivedIndexListingKey<'a> {
    pub version: u64,
    pub store_identity: &'a str,
    pub e_tag: Option<&'a str>,
}

impl CacheKey for DerivedIndexListingKey<'_> {
    type ValueType = Vec<IndexMetadata>;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "{}:{}/{}/{}",
            self.store_identity.len(),
            self.store_identity,
            self.version,
            self.e_tag.unwrap_or("")
        ))
    }

    fn type_name() -> &'static str {
        "Vec<IndexMetadata>"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.derived-listing-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(self.store_identity);
        builder.write_u64(self.version);
        match self.e_tag {
            Some(e_tag) => {
                builder.write_some();
                builder.write_str(e_tag);
            }
            None => builder.write_none(),
        }
    }

    // No `codec`: memory-only. See the type doc: the derived listing is
    // reader-capability dependent and must not be persisted across builds.
}

pub struct ProstAny(pub Arc<prost_types::Any>);

impl DeepSizeOf for ProstAny {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.type_url.deep_size_of_children(context) + self.0.value.deep_size_of_children(context)
    }
}

/// Cache key for scalar index details
///
/// Typically we don't use the cache for scalar index details because they are stored
/// in the manifest and readily available.  However, old versions of Lance didn't store
/// details in the manifest, and we have to perform an expensive inference process to determine
/// what they are.  These we cache.
#[derive(Debug)]
pub struct ScalarIndexDetailsKey<'a> {
    pub uuid: &'a Uuid,
}

impl CacheKey for ScalarIndexDetailsKey<'_> {
    type ValueType = ProstAny;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("type/{}", self.uuid))
    }

    fn type_name() -> &'static str {
        "ScalarIndexDetails"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.scalar-details-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.uuid.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Float32Array};
    use lance_core::cache::{CacheNamespace, InternalCacheKey};
    use lance_index::vector::DEFAULT_QUERY_PARALLELISM;

    fn vector_index_metadata() -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::from_u128(0x11111111_2222_3333_4444_555555555555),
            fields: vec![7],
            covering_fields: vec![],
            name: "vector_idx".to_string(),
            dataset_version: 5,
            fragment_bitmap: Some([1_u32, 3, 8].into_iter().collect()),
            index_details: Some(Arc::new(prost_types::Any {
                type_url: "type.googleapis.com/lance.table.VectorIndexDetails".to_string(),
                value: vec![1],
            })),
            index_version: 3,
            created_at: None,
            base_id: Some(2),
            files: None,
        }
    }

    fn vector_query() -> Query {
        Query {
            column: "vector".to_string(),
            key: Arc::new(Float32Array::from(vec![0.25, 0.5, 0.75])) as ArrayRef,
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 3,
            maximum_nprobes: Some(3),
            ef: None,
            refine_factor: Some(2),
            metric_type: Some(DistanceType::Cosine),
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: f32::NAN,
            approx_mode: ApproxMode::Accurate,
        }
    }

    fn vector_results_identity_for(query: &Query) -> VectorResultsCacheIdentity {
        VectorResultsCacheIdentity::try_new(
            "s3$account-a",
            "_versions/17.manifest",
            17,
            &vector_index_metadata(),
            Some(Uuid::from_u128(0xaaaaaaaa_bbbb_cccc_dddd_eeeeeeeeeeee)),
            query,
            DistanceType::Cosine,
            SubIndexType::Flat,
            QuantizationType::Rabit,
            &[2, 5, 9],
            100,
            false,
            false,
        )
        .unwrap()
    }

    fn vector_results_identity() -> VectorResultsCacheIdentity {
        vector_results_identity_for(&vector_query())
    }

    fn physical_key(identity: &VectorResultsCacheIdentity) -> InternalCacheKey {
        let mut builder = KeyBuilder::new(
            CacheNamespace::root(),
            VectorResultsCacheIdentity::stable_type_id(),
            VectorResultsCacheIdentity::schema(),
        );
        identity.write_key(&mut builder);
        builder.finish()
    }

    fn assert_identity_field_isolated(
        base: &VectorResultsCacheIdentity,
        change: impl FnOnce(&mut VectorResultsCacheIdentity),
    ) {
        let mut changed = base.clone();
        change(&mut changed);
        assert_ne!(physical_key(base), physical_key(&changed));
    }

    #[test]
    fn index_metadata_key_isolates_object_store_identity() {
        let first = IndexMetadataKey {
            version: 7,
            store_identity: "s3$first-options",
            e_tag: Some("manifest-etag"),
        };
        let second = IndexMetadataKey {
            version: 7,
            store_identity: "s3$second-options",
            e_tag: Some("manifest-etag"),
        };

        assert_ne!(first.key(), second.key());
    }

    #[test]
    fn index_metadata_key_isolates_manifest_generation() {
        let first = IndexMetadataKey {
            version: 7,
            store_identity: "s3$options",
            e_tag: Some("first-etag"),
        };
        let second = IndexMetadataKey {
            version: 7,
            store_identity: "s3$options",
            e_tag: Some("second-etag"),
        };

        assert_ne!(first.key(), second.key());
    }

    #[test]
    fn derived_listing_key_isolates_table_snapshot_and_generation() {
        let base = DerivedIndexListingKey {
            version: 7,
            store_identity: "s3$options",
            e_tag: Some("etag"),
        };
        // Different table (object store identity).
        let other_table = DerivedIndexListingKey {
            store_identity: "s3$other-options",
            ..base
        };
        // Different snapshot version (an index add/replace commits a new version).
        let other_version = DerivedIndexListingKey { version: 8, ..base };
        // Different manifest generation (e_tag).
        let other_generation = DerivedIndexListingKey {
            e_tag: Some("other-etag"),
            ..base
        };

        assert_ne!(base.key(), other_table.key());
        assert_ne!(base.key(), other_version.key());
        assert_ne!(base.key(), other_generation.key());
    }

    #[test]
    fn derived_listing_key_does_not_collide_with_raw_metadata_key() {
        // The derived (coverage-rewritten) listing and the raw manifest listing
        // share the same snapshot identity but must live in separate cache
        // namespaces, so the derived listing is never served as the raw one.
        assert_ne!(
            DerivedIndexListingKey::schema().id(),
            IndexMetadataKey::schema().id()
        );
    }
    #[test]
    fn vector_results_cache_key_isolates_every_identity_axis() {
        let base = vector_results_identity();

        assert_identity_field_isolated(&base, |identity| {
            identity.store_identity.push_str("-rotated")
        });
        assert_identity_field_isolated(&base, |identity| {
            identity.dataset_read_identity.push_str("-detached")
        });
        assert_identity_field_isolated(&base, |identity| identity.dataset_version += 1);
        assert_identity_field_isolated(&base, |identity| identity.index_uuid = Uuid::new_v4());
        assert_identity_field_isolated(&base, |identity| identity.frag_reuse_uuid = None);
        assert_identity_field_isolated(&base, |identity| identity.index_dataset_version += 1);
        assert_identity_field_isolated(&base, |identity| identity.index_version += 1);
        assert_identity_field_isolated(&base, |identity| identity.index_base_id = None);
        assert_identity_field_isolated(&base, |identity| identity.index_fields.push(8));
        assert_identity_field_isolated(&base, |identity| identity.column.push_str("_new"));
        assert_identity_field_isolated(&base, |identity| identity.metric_variant += 1);
        assert_identity_field_isolated(&base, |identity| identity.sub_index_variant += 1);
        assert_identity_field_isolated(&base, |identity| identity.quantization_variant += 1);
        assert_identity_field_isolated(&base, |identity| identity.approx_mode_variant += 1);
        assert_identity_field_isolated(&base, |identity| identity.vector_type_variant += 1);
        assert_identity_field_isolated(&base, |identity| identity.dimension += 1);
        assert_identity_field_isolated(&base, |identity| identity.query_fingerprint[0] ^= 1);
        assert_identity_field_isolated(&base, |identity| identity.nprobes += 1);
        assert_identity_field_isolated(&base, |identity| identity.partition_ids.swap(0, 1));
        assert_identity_field_isolated(&base, |identity| identity.result_limit += 1);
        assert_identity_field_isolated(&base, |identity| identity.candidate_limit += 1);
    }

    #[test]
    fn vector_results_cache_key_isolates_exact_query_values() {
        let first_query = vector_query();
        let mut second_query = first_query.clone();
        second_query.key = Arc::new(Float32Array::from(vec![0.25, 0.5, 0.76]));

        let first = vector_results_identity_for(&first_query);
        let second = vector_results_identity_for(&second_query);

        assert_eq!(first.partition_ids(), second.partition_ids());
        assert_ne!(first.query_fingerprint, second.query_fingerprint);
        assert_ne!(physical_key(&first), physical_key(&second));
    }

    #[test]
    fn vector_results_cache_identity_rejects_unsafe_query_forms() {
        let index = vector_index_metadata();
        let make_identity = |query: &Query,
                             index: &IndexMetadata,
                             sub_index_type,
                             quantization_type,
                             partition_ids: &[u32],
                             has_prefilter,
                             has_overlay| {
            VectorResultsCacheIdentity::try_new(
                "s3$account-a",
                "_versions/17.manifest",
                17,
                index,
                None,
                query,
                DistanceType::Cosine,
                sub_index_type,
                quantization_type,
                partition_ids,
                100,
                has_prefilter,
                has_overlay,
            )
        };

        let query = vector_query();
        assert!(
            make_identity(
                &query,
                &index,
                SubIndexType::Flat,
                QuantizationType::Rabit,
                &[2, 5, 9],
                true,
                false,
            )
            .is_err()
        );
        assert!(
            make_identity(
                &query,
                &index,
                SubIndexType::Flat,
                QuantizationType::Rabit,
                &[2, 5, 9],
                false,
                true,
            )
            .is_err()
        );

        let mut bounded = query.clone();
        bounded.lower_bound = Some(0.1);
        let mut adaptive = query.clone();
        adaptive.maximum_nprobes = None;
        let mut mismatched_metric = query.clone();
        mismatched_metric.metric_type = Some(DistanceType::Dot);
        let mut hnsw_query = query.clone();
        hnsw_query.ef = Some(64);
        let mut empty_query = query.clone();
        empty_query.key = Arc::new(Float32Array::from(Vec::<f32>::new()));
        let mut null_query = query.clone();
        null_query.key = Arc::new(Float32Array::from(vec![Some(0.25), None, Some(0.75)]));
        let mut nan_query = query.clone();
        nan_query.key = Arc::new(Float32Array::from(vec![0.25, f32::NAN, 0.75]));
        let mut infinite_query = query.clone();
        infinite_query.key = Arc::new(Float32Array::from(vec![0.25, f32::INFINITY, 0.75]));
        for unsupported_query in [
            bounded,
            adaptive,
            mismatched_metric,
            hnsw_query,
            empty_query,
            null_query,
            nan_query,
            infinite_query,
        ] {
            assert!(
                make_identity(
                    &unsupported_query,
                    &index,
                    SubIndexType::Flat,
                    QuantizationType::Rabit,
                    &[2, 5, 9],
                    false,
                    false,
                )
                .is_err()
            );
        }

        for approx_mode in [ApproxMode::Fast, ApproxMode::Normal] {
            let mut rq_query = query.clone();
            rq_query.approx_mode = approx_mode;
            let error = make_identity(
                &rq_query,
                &index,
                SubIndexType::Flat,
                QuantizationType::Rabit,
                &[2, 5, 9],
                false,
                false,
            )
            .unwrap_err();
            assert!(matches!(error, Error::NotSupported { .. }));
            assert!(
                error
                    .to_string()
                    .contains("supports RQ only with ApproxMode::Accurate")
            );
            assert!(
                make_identity(
                    &rq_query,
                    &index,
                    SubIndexType::Flat,
                    QuantizationType::Scalar,
                    &[2, 5, 9],
                    false,
                    false,
                )
                .is_ok(),
                "SQ ignores the RQ-specific approximation policy"
            );
        }

        assert!(
            make_identity(
                &query,
                &index,
                SubIndexType::Hnsw,
                QuantizationType::Scalar,
                &[2, 5, 9],
                false,
                false,
            )
            .is_err()
        );
        assert!(
            make_identity(
                &query,
                &index,
                SubIndexType::Flat,
                QuantizationType::Product,
                &[2, 5, 9],
                false,
                false,
            )
            .is_err()
        );
        assert!(
            make_identity(
                &query,
                &index,
                SubIndexType::Flat,
                QuantizationType::Rabit,
                &[2, 2, 9],
                false,
                false,
            )
            .is_err()
        );

        let mut unknown_coverage = index;
        unknown_coverage.fragment_bitmap = None;
        assert!(
            make_identity(
                &query,
                &unknown_coverage,
                SubIndexType::Flat,
                QuantizationType::Rabit,
                &[2, 5, 9],
                false,
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn vector_results_cache_entry_validates_candidate_shape() {
        let identity = vector_results_identity();
        let candidates = vec![
            CachedVectorCandidate::new(2, 10),
            CachedVectorCandidate::new(5, 20),
            CachedVectorCandidate::new(9, 30),
        ];
        let entry = VectorResultsCacheEntry::try_new(identity.clone(), candidates).unwrap();
        assert!(entry.is_compatible_with(&identity));
        assert_eq!(entry.candidates()[1].partition_id(), 5);
        assert_eq!(entry.candidates()[1].offset_in_partition(), 20);

        let mut different_identity = identity.clone();
        different_identity.dataset_version += 1;
        assert!(!entry.is_compatible_with(&different_identity));
        assert!(
            VectorResultsCacheEntry::try_new(
                identity.clone(),
                vec![CachedVectorCandidate::new(7, 10)],
            )
            .is_err()
        );
        assert!(
            VectorResultsCacheEntry::try_new(
                identity,
                vec![
                    CachedVectorCandidate::new(2, 10),
                    CachedVectorCandidate::new(2, 10),
                ],
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn vector_results_cache_identity_produces_typed_cache_misses() {
        let cache = LanceCache::with_capacity(4096);
        let identity = vector_results_identity();
        let entry = Arc::new(
            VectorResultsCacheEntry::try_new(
                identity.clone(),
                vec![CachedVectorCandidate::new(2, 10)],
            )
            .unwrap(),
        );
        cache.insert_with_key(&identity, entry.clone()).await;
        let cached = cache.get_with_key(&identity).await.unwrap();
        assert!(cached.is_compatible_with(&identity));

        let mut other_version = identity;
        other_version.dataset_version += 1;
        assert!(cache.get_with_key(&other_version).await.is_none());
    }
}
