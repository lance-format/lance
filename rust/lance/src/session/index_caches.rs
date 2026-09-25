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

use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_index::frag_reuse::CompactFragReuseIndex;
use lance_table::format::IndexMetadata;
use uuid::Uuid;

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
}
