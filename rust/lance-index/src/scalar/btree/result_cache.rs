// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    borrow::Cow,
    hash::{DefaultHasher, Hash, Hasher},
    mem::size_of,
    ops::Bound,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use datafusion_common::ScalarValue;
use lance_core::{
    cache::{CacheKey, CacheKeySchema, CacheNamespace, InternalCacheKey, KeyBuilder},
    deepsize::DeepSizeOf,
};
use lance_select::NullableRowAddrSet;

use crate::scalar::{SargableQuery, SearchOptions};

/// Below this page count, the native search avoids the extra cache access.
pub(super) const MIN_MATCHED_PAGES: usize = 128;
/// Limit key construction for wide string literals. A range has at most two.
const MAX_LITERAL_BYTES: usize = 4 * 1024;
/// Keep an individual result from displacing a large share of the page cache.
const MAX_RESULT_BYTES: usize = 16 * 1024 * 1024;
/// Keep 512 bytes of recent filter fingerprints per resident BTree state.
const RECENT_FILTER_SLOTS: usize = 64;

/// Shared by warm opens of the same immutable BTree state.
#[derive(Debug)]
pub(super) struct ResultCacheState {
    pub(super) identity: Arc<()>,
    recent_filters: [AtomicU64; RECENT_FILTER_SLOTS],
}

impl Default for ResultCacheState {
    fn default() -> Self {
        Self {
            identity: Arc::new(()),
            recent_filters: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl DeepSizeOf for ResultCacheState {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.identity.deep_size_of_children(context)
    }
}

impl ResultCacheState {
    /// Preserve page-cache capacity by admitting only recently repeated filters.
    /// This is an admission hint: collisions can affect insertion, never reuse.
    pub(super) fn should_admit(&self, key: &QueryResultKey<'_>) -> bool {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let fingerprint = hasher.finish();
        let slot = (fingerprint as usize) % RECENT_FILTER_SLOTS;
        let marker = fingerprint | 1; // Zero denotes a slot that has never been written.
        self.recent_filters[slot].swap(marker, Ordering::Relaxed) == marker
    }
}

#[derive(Clone, Copy, Hash)]
pub(super) struct QueryResultKey<'a> {
    // The index namespace isolates segments. A resident state shares this
    // token across opens, while another physical load gets a new token.
    identity: usize,
    track_nulls: bool,
    predicate: Predicate<'a>,
}

#[derive(Clone, Copy, Hash)]
enum Predicate<'a> {
    Equals(Literal<'a>),
    Range(Bound<Literal<'a>>, Bound<Literal<'a>>),
}

#[derive(Clone, Copy, Hash)]
enum Literal<'a> {
    Int32(i32),
    Int64(i64),
    UInt32(u32),
    UInt64(u64),
    Utf8(&'a str),
    LargeUtf8(&'a str),
}

impl<'a> Literal<'a> {
    fn read(value: &'a ScalarValue) -> Option<Self> {
        match value {
            ScalarValue::Int32(Some(value)) => Some(Self::Int32(*value)),
            ScalarValue::Int64(Some(value)) => Some(Self::Int64(*value)),
            ScalarValue::UInt32(Some(value)) => Some(Self::UInt32(*value)),
            ScalarValue::UInt64(Some(value)) => Some(Self::UInt64(*value)),
            ScalarValue::Utf8(Some(value)) if value.len() <= MAX_LITERAL_BYTES => {
                Some(Self::Utf8(value))
            }
            ScalarValue::LargeUtf8(Some(value)) if value.len() <= MAX_LITERAL_BYTES => {
                Some(Self::LargeUtf8(value))
            }
            _ => None,
        }
    }

    fn write(&self, builder: &mut KeyBuilder) {
        match self {
            Self::Int32(value) => builder.write_i32(*value),
            Self::Int64(value) => builder.write_i64(*value),
            Self::UInt32(value) => builder.write_u32(*value),
            Self::UInt64(value) => builder.write_u64(*value),
            Self::Utf8(value) | Self::LargeUtf8(value) => {
                builder.write_bool(matches!(self, Self::LargeUtf8(_)));
                builder.write_str(value);
            }
        }
    }
}

fn read_bound(bound: &Bound<ScalarValue>) -> Option<Bound<Literal<'_>>> {
    match bound {
        Bound::Included(value) => Some(Bound::Included(Literal::read(value)?)),
        Bound::Excluded(value) => Some(Bound::Excluded(Literal::read(value)?)),
        Bound::Unbounded => Some(Bound::Unbounded),
    }
}

fn write_bound(bound: &Bound<Literal<'_>>, builder: &mut KeyBuilder) {
    match bound {
        Bound::Included(value) => {
            builder.write_u8(1);
            value.write(builder);
        }
        Bound::Excluded(value) => {
            builder.write_u8(2);
            value.write(builder);
        }
        Bound::Unbounded => builder.write_u8(0),
    }
}

impl<'a> QueryResultKey<'a> {
    pub(super) fn new(
        identity: &Arc<()>,
        query: &'a SargableQuery,
        options: SearchOptions,
    ) -> Option<Self> {
        let predicate = match query {
            SargableQuery::Equals(value) => Predicate::Equals(Literal::read(value)?),
            SargableQuery::Range(lower, upper) => {
                Predicate::Range(read_bound(lower)?, read_bound(upper)?)
            }
            _ => return None,
        };
        Some(Self {
            identity: Arc::as_ptr(identity) as usize,
            track_nulls: options.track_nulls(),
            predicate,
        })
    }
}

impl CacheKey for QueryResultKey<'_> {
    type ValueType = CachedResult;

    fn key(&self) -> Cow<'_, str> {
        let mut builder = KeyBuilder::new(
            CacheNamespace::root(),
            Self::stable_type_id(),
            Self::schema(),
        );
        self.write_key(&mut builder);
        format!("{:?}", builder.finish()).into()
    }

    fn type_name() -> &'static str {
        "BTreeQueryResult"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.btree-query-result-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.identity as u64);
        builder.write_bool(self.track_nulls);
        match &self.predicate {
            Predicate::Equals(value) => {
                builder.write_u8(1);
                value.write(builder);
            }
            Predicate::Range(lower, upper) => {
                builder.write_u8(2);
                write_bound(lower, builder);
                write_bound(upper, builder);
            }
        }
    }

    // Process-local identity deliberately uses the default (no disk codec).
}

#[derive(DeepSizeOf)]
pub(super) struct CachedResult {
    pub(super) rows: NullableRowAddrSet,
    // Keep the token alive for the lifetime of the cache entry so its address
    // cannot be assigned to a later load.
    pub(super) identity: Arc<()>,
}

impl CachedResult {
    pub(super) fn fits(rows: &NullableRowAddrSet, capacity: usize) -> bool {
        let overhead =
            size_of::<Arc<()>>() + 4 * size_of::<usize>() + size_of::<InternalCacheKey>();
        // Scale down admission for deployments with a small index cache.
        rows.deep_size_of().saturating_add(overhead) <= MAX_RESULT_BYTES.min(capacity / 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_preserves_literal_types_bounds_and_null_tracking() {
        let identity = Arc::new(());
        let options = SearchOptions::default();
        let key = |query: &SargableQuery, options| {
            QueryResultKey::new(&identity, query, options)
                .unwrap()
                .key()
                .into_owned()
        };
        let int32 = SargableQuery::Equals(ScalarValue::Int32(Some(42)));
        assert_ne!(
            key(&int32, options),
            key(&int32, options.with_track_nulls(false))
        );
        assert_ne!(
            key(&int32, options),
            key(
                &SargableQuery::Equals(ScalarValue::UInt32(Some(42))),
                options
            ),
        );
        let text = ScalarValue::Utf8(Some("same text".to_string()));
        assert_ne!(
            key(
                &SargableQuery::Range(Bound::Unbounded, Bound::Included(text.clone())),
                options
            ),
            key(
                &SargableQuery::Range(Bound::Unbounded, Bound::Excluded(text.clone())),
                options
            ),
        );
        assert_ne!(
            key(&SargableQuery::Equals(text), options),
            key(
                &SargableQuery::Equals(ScalarValue::LargeUtf8(Some("same text".to_string()))),
                options
            ),
        );
        assert!(
            QueryResultKey::new(
                &identity,
                &SargableQuery::Equals(ScalarValue::Utf8(Some("a".repeat(MAX_LITERAL_BYTES + 1)))),
                options,
            )
            .is_none()
        );
    }
}
