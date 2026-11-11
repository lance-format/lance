// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Sole cache ownership of prepared allocations shared by snapshot directories.

use std::collections::HashMap;
use std::sync::Arc;

use lance_core::cache::{CacheKey, CacheNamespace, KeyBuilder};
use lance_table::rowids::{RowIdIndex, RowIdIndexPart};

use crate::session::index_caches::PreparedRowIdPartKey;
use crate::{Dataset, Error, Result};

use super::cache::CachedFragment;

pub(super) async fn prepare(
    dataset: &Dataset,
    index: &RowIdIndex,
    fragments: &[CachedFragment],
    retained: &[Arc<RowIdIndexPart>],
) -> Result<(Vec<PreparedRowIdPartKey>, Vec<Arc<RowIdIndexPart>>)> {
    let sources: HashMap<_, _> = fragments
        .iter()
        .map(|cached| (cached.fragment_id, cached))
        .collect();
    let mut keys = Vec::new();
    let mut owned = Vec::new();
    for part in index.prepared_parts() {
        let mut ids = part.fragment_ids();
        ids.sort_unstable();
        ids.dedup();
        let mut builder = KeyBuilder::new(
            CacheNamespace::root(),
            "lance.dataset.row-id-part-sources",
            PreparedRowIdPartKey::schema(),
        );
        builder.write_str(&dataset.uri);
        for id in ids {
            let source = sources.get(&id).ok_or_else(|| {
                Error::internal(format!(
                    "prepared row id part references fragment {id} outside snapshot {}",
                    dataset.manifest.version
                ))
            })?;
            builder.write_u32(id);
            builder.write_fixed_bytes(source.identity.as_bytes());
            if let Some(deletion) = source.deletion_identity {
                builder.write_some();
                builder.write_fixed_bytes(deletion.as_bytes());
            } else {
                builder.write_none();
            }
        }
        let (merged, allocation) = part.allocation_identity();
        let key = PreparedRowIdPartKey {
            sources: builder.finish(),
            merged,
            allocation,
        };
        let existing = dataset.index_cache.get_with_key(&key).await.or_else(|| {
            retained
                .iter()
                .find(|existing| existing.allocation_identity() == (merged, allocation))
                .cloned()
        });
        let part = if let Some(existing) = existing {
            Arc::new(existing.try_union(&part)?)
        } else {
            Arc::new(part)
        };
        keys.push(key);
        owned.push(part);
    }
    Ok((keys, owned))
}

/// Refresh cache access for the graph owners and retain them for this caller.
/// A weak view can additionally refer to an actual active reader's allocations.
pub(super) async fn acquire(
    dataset: &Dataset,
    keys: &[PreparedRowIdPartKey],
) -> Vec<Arc<RowIdIndexPart>> {
    let mut parts = Vec::with_capacity(keys.len());
    for key in keys {
        if let Some(part) = dataset.index_cache.get_with_key(key).await {
            parts.push(part);
        }
    }
    parts
}

/// Admit owners only after the caller has checked the complete working set.
pub(super) async fn admit(
    dataset: &Dataset,
    keys: &[PreparedRowIdPartKey],
    parts: &[Arc<RowIdIndexPart>],
) {
    for (key, part) in keys.iter().zip(parts) {
        dataset.index_cache.insert_with_key(key, part.clone()).await;
    }
}
