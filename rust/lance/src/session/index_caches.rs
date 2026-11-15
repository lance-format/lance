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

use std::{borrow::Cow, ops::Deref, sync::Arc};

use arrow_array::{
    Array, Float32Array, RecordBatch, UInt64Array,
    cast::AsArray,
    types::{Float32Type, UInt64Type},
};
use arrow_schema::DataType;
use lance_core::ROW_ID;
use lance_core::cache::{
    CacheKey, CacheKeySchema, CacheNamespace, InternalCacheKey, KeyBuilder, LanceCache,
};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_index::frag_reuse::{CompactFragReuseIndex, FragReuseIndexDetails};
use lance_index::vector::{ApproxMode, DIST_COL, Query, VECTOR_RESULT_SCHEMA};
use lance_linalg::distance::DistanceType;
use lance_select::RowAddrMask;
use lance_table::{format::IndexMetadata, rowids::RowIdIndex};
use uuid::Uuid;

/// Owned partition results only: no resident buffers, prefilters or dataset handles.
/// No codec is provided, so persistent backends keep these entries in memory.
pub(crate) struct QuantizedRefinementResult(pub RecordBatch);

impl QuantizedRefinementResult {
    pub fn copy_from(batch: &RecordBatch) -> crate::Result<Self> {
        // Accurate range search can return a tiny slice of a whole partition.
        // Rebuild only these two columns so admission charges everything retained.
        let distances = batch
            .column_by_name(DIST_COL)
            .and_then(|column| column.as_primitive_opt::<Float32Type>())
            .ok_or_else(|| {
                crate::Error::internal("quantized refinement result requires Float32 _distance")
            })?
            .iter()
            .collect::<Float32Array>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .and_then(|column| column.as_primitive_opt::<UInt64Type>())
            .ok_or_else(|| {
                crate::Error::internal("quantized refinement result requires UInt64 _rowid")
            })?
            .iter()
            .collect::<UInt64Array>();
        Ok(Self(RecordBatch::try_new(
            VECTOR_RESULT_SCHEMA.clone(),
            vec![Arc::new(distances), Arc::new(row_ids)],
        )?))
    }
}

impl DeepSizeOf for QuantizedRefinementResult {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.0.get_array_memory_size()
    }
}

/// The original result contract, before the planner expands k for exact refinement.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QuantizedRefinementOptions {
    pub result_count: usize,
    pub exact_factor: Option<u32>,
}

/// Snapshot and physical segment identity, shared by one query's partition keys.
#[derive(Clone, Debug)]
pub(crate) struct QuantizedRefinementCache {
    pub cache: LanceCache,
    pub identity: InternalCacheKey,
}

impl QuantizedRefinementCache {
    pub async fn for_segment(
        dataset: &crate::Dataset,
        index: &IndexMetadata,
        original_query: &Query,
        options: QuantizedRefinementOptions,
    ) -> crate::Result<Option<Self>> {
        let location = dataset.manifest_location();
        let manifest = dataset.manifest();
        // Transactions have unique names even after overwrite/recreation. A
        // manifest path/version alone cannot establish the dataset incarnation.
        if dataset.index_cache.capacity_bytes() == Some(0)
            || (manifest
                .transaction_file
                .as_deref()
                .is_none_or(str::is_empty)
                && location.e_tag.as_deref().is_none_or(str::is_empty))
        {
            return Ok(None);
        }
        let mut builder = KeyBuilder::new(
            CacheNamespace::root(),
            QuantizedRefinementKey::type_name(),
            QuantizedRefinementKey::schema(),
        );
        builder.write_str(dataset.uri());
        builder.write_str(&dataset.object_store(None).await?.store_prefix);
        builder.write_str(location.path.as_ref());
        builder.write_u64(manifest.version);
        builder.write_u64(options.result_count as u64);
        match options.exact_factor {
            Some(factor) => {
                builder.write_some();
                builder.write_u32(factor);
            }
            None => builder.write_none(),
        }
        for value in [
            manifest.branch.as_deref(),
            manifest.transaction_file.as_deref(),
            location.e_tag.as_deref(),
        ] {
            match value {
                Some(value) => {
                    builder.write_some();
                    builder.write_str(value);
                }
                None => builder.write_none(),
            }
        }
        let store = dataset.object_store_for_index(index).await?;
        builder.write_str(&store.store_prefix);
        builder.write_str(dataset.indice_files_dir(index)?.as_ref());
        write_index_identity(&mut builder, &index.uuid, None);
        // The authoritative manifest generation binds both FRI contents and
        // the segment's effective coverage. Never reuse rankings across snapshots,
        // even when the native immutable-data/translation caches remain reusable.
        if !write_query_vector(&mut builder, original_query) {
            return Ok(None);
        }
        Ok(Some(Self {
            cache: dataset.index_cache.0.clone(),
            identity: builder.finish(),
        }))
    }
}

pub(crate) struct QuantizedRefinementKey {
    pub identity: InternalCacheKey,
    pub partition_id: usize,
    pub dist_q_c: f32,
}

fn write_query_vector(builder: &mut KeyBuilder, query: &Query) -> bool {
    if query.key.null_count() != 0 {
        return false;
    }
    builder.write_u64(query.key.len() as u64);
    match query.key.data_type() {
        DataType::Float16 => {
            builder.write_u8(16);
            for value in query
                .key
                .as_primitive::<arrow_array::types::Float16Type>()
                .values()
            {
                if !value.is_finite() {
                    return false;
                }
                builder.write_u16(value.to_bits());
            }
        }
        DataType::Float32 => {
            builder.write_u8(32);
            for value in query
                .key
                .as_primitive::<arrow_array::types::Float32Type>()
                .values()
            {
                if !value.is_finite() {
                    return false;
                }
                builder.write_u32(value.to_bits());
            }
        }
        DataType::Float64 => {
            builder.write_u8(64);
            for value in query
                .key
                .as_primitive::<arrow_array::types::Float64Type>()
                .values()
            {
                if !value.is_finite() {
                    return false;
                }
                builder.write_u64(value.to_bits());
            }
        }
        _ => return false,
    }
    true
}

impl QuantizedRefinementKey {
    /// Bind resolved controls once per segment, after normalization and probe
    /// policy resolution. Original vector bits are already bound by `identity`;
    /// centroid distance varies per partition and belongs in `write_key`.
    pub fn query_identity(
        identity: InternalCacheKey,
        query: &Query,
        metric: DistanceType,
        factor: u32,
    ) -> InternalCacheKey {
        let mut builder =
            KeyBuilder::new(CacheNamespace::root(), Self::type_name(), Self::schema());
        builder.write_fixed_bytes(identity.as_bytes());
        builder.write_str(&query.column);
        builder.write_str(&metric.to_string());
        // Include every resolved control, including values overridden by this
        // stage, to make changes to the query contract conservatively miss.
        let Query {
            k,
            lower_bound,
            upper_bound,
            minimum_nprobes,
            maximum_nprobes,
            search_effort,
            ef,
            refine_factor,
            metric_type,
            use_index,
            query_parallelism,
            dist_q_c: _,
            approx_mode,
            column: _,
            key: _,
        } = query;
        builder.write_u64(*k as u64);
        builder.write_u32(factor);
        builder.write_u64(*minimum_nprobes as u64);
        for value in [
            maximum_nprobes.map(|v| v as u64),
            ef.map(|v| v as u64),
            refine_factor.map(u64::from),
            lower_bound.map(|v| u64::from(v.to_bits())),
            upper_bound.map(|v| u64::from(v.to_bits())),
        ] {
            match value {
                Some(value) => {
                    builder.write_some();
                    builder.write_u64(value);
                }
                None => builder.write_none(),
            }
        }
        builder.write_u64(search_effort.to_bits());
        builder.write_i32(*query_parallelism);
        builder.write_bool(*use_index);
        builder.write_str(&format!("{metric_type:?}"));
        builder.write_u8(match approx_mode {
            ApproxMode::Fast => 0,
            ApproxMode::Normal => 1,
            ApproxMode::Accurate => 2,
        });
        builder.finish()
    }
}

impl CacheKey for QuantizedRefinementKey {
    type ValueType = QuantizedRefinementResult;

    fn key(&self) -> Cow<'_, str> {
        let mut builder =
            KeyBuilder::new(CacheNamespace::root(), Self::type_name(), Self::schema());
        self.write_key(&mut builder);
        Cow::Owned(format!("{:?}", builder.finish()))
    }

    fn type_name() -> &'static str {
        "QuantizedRefinementResult"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.quantized-refinement-key", 2)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.identity.as_bytes());
        builder.write_u64(self.partition_id as u64);
        builder.write_u32(self.dist_q_c.to_bits());
    }
}

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

// Cache key types for type-safe cache access

/// Whole-dataset row-id lookup for one manifest and deletion mode.
#[derive(Debug)]
pub(crate) struct RowIdIndexKey<'a> {
    pub version: u64,
    /// Whether the index knows the deleted rows (it then maps them to `None`).
    /// An index built without deletion vectors reads no deletion file.
    pub with_deletions: bool,
    /// A dataset dropped and recreated at the same URI restarts its version
    /// history at 1, so a long-lived session's cache can otherwise return the
    /// previous incarnation's index for a version number the new incarnation
    /// now also holds. The e-tag disambiguates generations the same way
    /// [`super::caches::ManifestKey::e_tag`] does. Callers without one must not
    /// share the cached index at all (see `get_row_id_index`).
    pub e_tag: Option<&'a str>,
}

impl CacheKey for RowIdIndexKey<'_> {
    type ValueType = RowIdIndex;
    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "row_id_index/{}/{}/{}",
            self.version,
            self.with_deletions,
            self.e_tag.unwrap_or("")
        ))
    }
    fn type_name() -> &'static str {
        "RowIdIndex"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.row-id-index-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.version);
        builder.write_u64(self.with_deletions as u64);
        match self.e_tag {
            Some(e_tag) => {
                builder.write_some();
                builder.write_str(e_tag);
            }
            None => builder.write_none(),
        }
    }
}

/// Decoded fragment reuse history with `index_version = 0`, keyed by
/// `index::frag_reuse::v0_cache_identity`.
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
        // v3 entries carry each segment's clustering reference and metadata;
        // a v2 entry, written by a build that did not know those fields, holds
        // every segment unlabelled, and a writer trusting it would republish
        // the index section without the labels. v2 in turn holds every index
        // the manifest names where v1 held only the readable ones. On a
        // persistent backend shared with another release nothing but this
        // version stops each build from reading the other's entry as its own
        // meaning.
        CacheKeySchema::new("lance.index.metadata-key", 3)
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

/// Cache key for the listing `load_all_indices` remaps through a v0 FRI: the
/// fragment bitmaps rewritten so query planning sees live fragments. A snapshot
/// carries one FRI entry, so one key per snapshot.
///
/// Separate namespace from [`IndexMetadataKey`], which caches the raw manifest
/// listing. Query loading records its own value under `QueryIndexListingKey`,
/// which also tracks whether an incomplete listing needs another inference or
/// format-verification attempt. Keyed by the same snapshot identity: a different
/// table (`store_identity`), a different snapshot (`version`), or any index
/// add/replace (which commits a new manifest and bumps `version`/`e_tag`) yields
/// a different key, so a stale remapped listing is never reused.
///
/// This cache is MEMORY-ONLY (no `codec`): the remapped listing depends on the
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

/// Query-visible metadata and whether a later call should retry derivation.
#[derive(Debug)]
pub(crate) struct QueryIndexListing {
    pub(crate) indices: Arc<Vec<IndexMetadata>>,
    pub(crate) retry_derivation: bool,
}

impl DeepSizeOf for QueryIndexListing {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.indices.deep_size_of_children(context)
    }
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

/// Cache the complete query listing using the same snapshot identity as the
/// legacy derived listing. A separate key preserves its public value type and
/// prevents cached values with different concrete types from sharing a namespace.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QueryIndexListingKey<'a>(pub(crate) DerivedIndexListingKey<'a>);

impl CacheKey for QueryIndexListingKey<'_> {
    type ValueType = QueryIndexListing;

    fn key(&self) -> Cow<'_, str> {
        self.0.key()
    }

    fn type_name() -> &'static str {
        "QueryIndexListing"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.query-listing-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        self.0.write_key(builder);
    }

    // Reader compatibility depends on the current build, so keep this memory-only.
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

    use arrow_array::record_batch;
    use lance_core::cache::{MokaCacheBackend, QuickCacheBackend};
    use rstest::rstest;

    #[rstest::rstest]
    #[case::generation(5, true, "second-etag")]
    #[case::version(6, true, "first-etag")]
    #[case::deletions(5, false, "first-etag")]
    #[tokio::test]
    async fn row_id_index_key_separates_snapshots(
        #[case] version: u64,
        #[case] with_deletions: bool,
        #[case] e_tag: &str,
    ) {
        let global = GlobalIndexCache(LanceCache::with_capacity(4096));
        let cache = global.for_dataset("first");
        let key = RowIdIndexKey {
            version: 5,
            with_deletions: true,
            e_tag: Some("first-etag"),
        };
        cache
            .insert_with_key(&key, Arc::new(RowIdIndex::new(&[]).unwrap()))
            .await;
        assert!(cache.get_with_key(&key).await.is_some());

        assert!(
            cache
                .get_with_key(&RowIdIndexKey {
                    version,
                    with_deletions,
                    e_tag: Some(e_tag),
                })
                .await
                .is_none()
        );
        assert!(
            global
                .for_dataset("second")
                .get_with_key(&key)
                .await
                .is_none()
        );
    }

    fn refinement_query() -> Query {
        Query {
            column: "vec".to_string(),
            key: Arc::new(Float32Array::from(vec![0.0, 1.0])),
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            search_effort: 0.5,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::L2),
            use_index: true,
            query_parallelism: 0,
            dist_q_c: 0.0,
            approx_mode: ApproxMode::Normal,
        }
    }

    fn refinement_identity(query: &Query) -> InternalCacheKey {
        let mut builder = KeyBuilder::new(
            CacheNamespace::root(),
            QuantizedRefinementKey::type_name(),
            QuantizedRefinementKey::schema(),
        );
        assert!(write_query_vector(&mut builder, query));
        builder.finish()
    }

    fn refinement_key(query: &Query) -> QuantizedRefinementKey {
        QuantizedRefinementKey {
            identity: QuantizedRefinementKey::query_identity(
                refinement_identity(query),
                query,
                DistanceType::L2,
                4,
            ),
            partition_id: 0,
            dist_q_c: query.dist_q_c,
        }
    }

    #[rstest]
    #[case::values(|q: &mut Query| q.key = Arc::new(Float32Array::from(vec![0.0, 2.0])))]
    #[case::signed_zero(|q: &mut Query| q.key = Arc::new(Float32Array::from(vec![-0.0, 1.0])))]
    #[case::element_type(|q: &mut Query| q.key = Arc::new(arrow_array::Float64Array::from(vec![0.0, 1.0])))]
    #[case::dimension(|q: &mut Query| q.key = Arc::new(Float32Array::from(vec![0.0, 1.0, 2.0])))]
    #[case::column(|q: &mut Query| q.column = "other".to_string())]
    #[case::k(|q: &mut Query| q.k = 11)]
    #[case::lower_bound(|q: &mut Query| q.lower_bound = Some(-0.0))]
    #[case::upper_bound(|q: &mut Query| q.upper_bound = Some(1.0))]
    #[case::minimum_probes(|q: &mut Query| q.minimum_nprobes = 2)]
    #[case::maximum_probes(|q: &mut Query| q.maximum_nprobes = Some(2))]
    #[case::effort(|q: &mut Query| q.search_effort = 1.0)]
    #[case::ef(|q: &mut Query| q.ef = Some(20))]
    #[case::exact_factor(|q: &mut Query| q.refine_factor = Some(2))]
    #[case::metric(|q: &mut Query| q.metric_type = Some(DistanceType::Cosine))]
    #[case::parallelism(|q: &mut Query| q.query_parallelism = 2)]
    #[case::centroid_distance(|q: &mut Query| q.dist_q_c = -0.0)]
    #[case::scoring_mode(|q: &mut Query| q.approx_mode = ApproxMode::Accurate)]
    #[test]
    fn refinement_key_binds_query_controls(#[case] change: fn(&mut Query)) {
        let original = refinement_query();
        let mut changed = original.clone();
        change(&mut changed);
        assert_ne!(
            refinement_key(&original).key(),
            refinement_key(&changed).key()
        );
        // Arrow slices are keyed by their logical values, without buffer padding.
        let mut sliced = original.clone();
        sliced.key = Arc::new(Float32Array::from(vec![99.0, 0.0, 1.0, 99.0]).slice(1, 2));
        assert_eq!(
            refinement_key(&original).key(),
            refinement_key(&sliced).key()
        );
    }

    #[test]
    fn refinement_key_binds_partition_metric_and_factor() {
        let query = refinement_query();
        let original = refinement_key(&query).key().into_owned();
        for key in [
            QuantizedRefinementKey {
                partition_id: 1,
                ..refinement_key(&query)
            },
            QuantizedRefinementKey {
                identity: QuantizedRefinementKey::query_identity(
                    refinement_identity(&query),
                    &query,
                    DistanceType::L2,
                    5,
                ),
                ..refinement_key(&query)
            },
            QuantizedRefinementKey {
                identity: QuantizedRefinementKey::query_identity(
                    refinement_identity(&query),
                    &query,
                    DistanceType::Dot,
                    4,
                ),
                ..refinement_key(&query)
            },
        ] {
            assert_ne!(original, key.key());
        }
        assert!(QuantizedRefinementKey::codec().is_none());
    }

    #[test]
    fn refinement_query_identity_is_shared_across_partitions() {
        let query = refinement_query();
        let first = refinement_key(&query);
        let mut other_partition = query;
        other_partition.dist_q_c = -0.0;
        let second = refinement_key(&other_partition);
        assert_eq!(first.identity, second.identity);
        assert_ne!(first.key(), second.key());
    }

    #[test]
    fn refinement_query_identity_preserves_original_vectors_after_normalization() {
        let original = refinement_query();
        let mut scaled = original.clone();
        scaled.key = Arc::new(Float32Array::from(vec![0.0, 2.0]));
        // Both originals normalize to the same vector. They must still trigger
        // independent candidate discovery under the identical-query contract.
        let resolved = original.clone();
        let first = QuantizedRefinementKey::query_identity(
            refinement_identity(&original),
            &resolved,
            DistanceType::Cosine,
            4,
        );
        let second = QuantizedRefinementKey::query_identity(
            refinement_identity(&scaled),
            &resolved,
            DistanceType::Cosine,
            4,
        );
        assert_ne!(first, second);
    }

    #[rstest]
    #[case::null(vec![Some(1.0), None])]
    #[case::nan(vec![Some(f32::NAN)])]
    #[case::infinite(vec![Some(f32::INFINITY)])]
    #[test]
    fn refinement_key_rejects_unsupported_vectors(#[case] values: Vec<Option<f32>>) {
        let mut query = refinement_query();
        query.key = Arc::new(Float32Array::from(values));
        let mut builder = KeyBuilder::new(
            CacheNamespace::root(),
            QuantizedRefinementKey::type_name(),
            QuantizedRefinementKey::schema(),
        );
        assert!(!write_query_vector(&mut builder, &query));
    }

    #[test]
    fn refinement_result_does_not_pin_partition_slices_or_resident_columns() {
        let batch = record_batch!(
            ("_distance", Float32, vec![Some(1.0); 4096]),
            ("_rowid", UInt64, vec![Some(7u64); 4096]),
            ("resident", UInt64, vec![Some(99u64); 4096])
        )
        .unwrap();
        let slice = batch.slice(123, 1);
        let owned = QuantizedRefinementResult::copy_from(&slice).unwrap();
        assert_eq!(owned.0.num_columns(), 2);
        assert_eq!(owned.0["_rowid"].as_primitive::<UInt64Type>().value(0), 7);
        assert!(slice.get_array_memory_size() > 4096 * 12);
        assert!(owned.deep_size_of() < 1024);
        assert_ne!(
            owned.0["_rowid"]
                .as_primitive::<UInt64Type>()
                .values()
                .as_ptr(),
            slice["_rowid"]
                .as_primitive::<UInt64Type>()
                .values()
                .as_ptr()
        );
    }

    fn owned_refinement_result() -> QuantizedRefinementResult {
        QuantizedRefinementResult(
            record_batch!(("_distance", Float32, [1.0]), ("_rowid", UInt64, [7u64])).unwrap(),
        )
    }

    #[rstest]
    #[case::moka(false)]
    #[case::quick(true)]
    #[tokio::test]
    async fn refinement_cache_load_lifecycle(#[case] is_quick: bool) {
        let cache = if is_quick {
            LanceCache::with_backend(Arc::new(QuickCacheBackend::with_capacity(4096)))
        } else {
            LanceCache::with_backend(Arc::new(MokaCacheBackend::with_capacity(4096)))
        };
        let query = refinement_query();
        // Dropped and failed computations never publish a completed result.
        let mut pending = Box::pin(cache.get_or_insert_with_key(
            refinement_key(&query),
            || async {
                futures::future::pending::<crate::Result<QuantizedRefinementResult>>().await
            },
        ));
        assert!(futures::poll!(pending.as_mut()).is_pending());
        drop(pending);
        assert!(cache.get_with_key(&refinement_key(&query)).await.is_none());
        let error = cache
            .get_or_insert_with_key(refinement_key(&query), || async {
                Err(crate::Error::invalid_input("controlled scoring failure"))
            })
            .await
            .err()
            .unwrap();
        let mut source = &error;
        while let crate::Error::Shared { source: inner } = source {
            source = inner;
        }
        assert!(matches!(source, crate::Error::InvalidInput { .. }));
        assert!(error.to_string().contains("controlled scoring failure"));
        assert!(cache.get_with_key(&refinement_key(&query)).await.is_none());

        let (sender, receiver) = tokio::sync::oneshot::channel();
        let mut first = Box::pin(
            cache.get_or_insert_with_key(refinement_key(&query), || async {
                receiver.await.unwrap();
                Ok(owned_refinement_result())
            }),
        );
        assert!(futures::poll!(first.as_mut()).is_pending());
        let mut second = Box::pin(
            cache.get_or_insert_with_key(refinement_key(&query), || async {
                panic!("identical concurrent result loads must coalesce")
            }),
        );
        assert!(futures::poll!(second.as_mut()).is_pending());
        sender.send(()).unwrap();
        let (first, second) = futures::join!(first, second);
        assert!(Arc::ptr_eq(&first.unwrap(), &second.unwrap()));
        let warm = cache
            .get_or_insert_with_key(refinement_key(&query), || async {
                panic!("resident result should skip scoring")
            })
            .await
            .unwrap();
        assert_eq!(
            warm.0["_rowid"]
                .as_primitive::<arrow_array::types::UInt64Type>()
                .value(0),
            7
        );
        cache.clear().await;
        assert!(cache.get_with_key(&refinement_key(&query)).await.is_none());
    }

    #[tokio::test]
    async fn refinement_cache_reloads_owned_results_under_pressure() {
        let cache = LanceCache::with_backend(Arc::new(QuickCacheBackend::with_shard_policy(
            4096,
            lance_core::cache::QuickCacheShardPolicy::Single,
        )));
        let query = refinement_query();
        for partition_id in 0..64 {
            let key = QuantizedRefinementKey {
                partition_id,
                ..refinement_key(&query)
            };
            cache
                .get_or_insert_with_key(key, || async { Ok(owned_refinement_result()) })
                .await
                .unwrap();
        }
        // Admission may protect the oldest entry and reject a newer one.
        // Check a nonresident partition rather than assuming a particular victim.
        let mut missing = None;
        for partition_id in 0..64 {
            let key = QuantizedRefinementKey {
                partition_id,
                ..refinement_key(&query)
            };
            if cache.get_with_key(&key).await.is_none() {
                missing = Some(partition_id);
                break;
            }
        }
        let key = QuantizedRefinementKey {
            partition_id: missing.expect("cache pressure must leave some results nonresident"),
            ..refinement_key(&query)
        };
        let before = cache.diagnostics().activity.loads_started;
        let result = cache
            .get_or_insert_with_key(key, || async { Ok(owned_refinement_result()) })
            .await
            .unwrap();
        assert_eq!(result.0, owned_refinement_result().0);
        assert_eq!(cache.diagnostics().activity.loads_started, before + 1);
        let snapshot = cache
            .diagnostics_with_mode(lance_core::cache::CacheSnapshotMode::Refreshed)
            .await;
        assert!(snapshot.backend.size_bytes.unwrap() <= 4096);
    }

    #[rstest]
    #[case::disabled(0)]
    #[case::oversized(1)]
    #[tokio::test]
    async fn refinement_cache_respects_budget(#[case] capacity: usize) {
        let cache = LanceCache::with_capacity(capacity);
        let query = refinement_query();
        for _ in 0..2 {
            let result = cache
                .get_or_insert_with_key(refinement_key(&query), || async {
                    Ok(owned_refinement_result())
                })
                .await
                .unwrap();
            assert_eq!(result.0.num_rows(), 1);
            // Moka applies its weighted admission policy in deferred maintenance.
            let snapshot = cache
                .diagnostics_with_mode(lance_core::cache::CacheSnapshotMode::Refreshed)
                .await;
            assert!(snapshot.backend.size_bytes.unwrap() <= capacity as u64);
            assert!(cache.get_with_key(&refinement_key(&query)).await.is_none());
        }
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
        assert_ne!(
            QueryIndexListingKey(base).key(),
            QueryIndexListingKey(other_table).key()
        );
        assert_ne!(
            QueryIndexListingKey(base).key(),
            QueryIndexListingKey(other_version).key()
        );
        assert_ne!(
            QueryIndexListingKey(base).key(),
            QueryIndexListingKey(other_generation).key()
        );
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
        assert_ne!(
            QueryIndexListingKey::schema().id(),
            IndexMetadataKey::schema().id()
        );
        assert_ne!(
            QueryIndexListingKey::schema().id(),
            DerivedIndexListingKey::schema().id()
        );
    }
}
