// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache};
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_table::format::RowIdMeta;
use lance_table::rowids::{read_row_ids, segment::U64Segment};

use super::get_row_id_index_with_cache;
use crate::Table;
use crate::caches::RowIdIndexKey;
use crate::io::deletion::read_table_deletion_file;

/// A dense fragment maps an ID to a physical offset by subtraction. Cache only
/// its endpoints and manifest slot. Prefix maxima let a point lookup skip
/// disjoint ranges and still find old and new copies after an update.
#[derive(DeepSizeOf)]
struct InlineRowIdRange {
    start: u64,
    end: u64,
    prefix_end: u64,
    slot: usize,
}

struct InlineRowIdRangesKey<'a> {
    version: u64,
    e_tag: &'a str,
}

impl CacheKey for InlineRowIdRangesKey<'_> {
    type ValueType = Option<Vec<InlineRowIdRange>>;

    fn key(&self) -> Cow<'_, str> {
        format!("inline_row_id_ranges/{}/{}", self.version, self.e_tag).into()
    }

    fn type_name() -> &'static str {
        "InlineRowIdRanges"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.inline-row-id-ranges", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.version);
        builder.write_str(self.e_tag);
    }
}

fn inline_row_id_ranges(table: &Table) -> Result<Option<Vec<InlineRowIdRange>>> {
    let mut ranges = Vec::new();
    for (slot, fragment) in table.iter_fragments().enumerate() {
        let Some(RowIdMeta::Inline(data)) = &fragment.row_id_meta else {
            return Ok(None);
        };
        // A protobuf sequence with one Range (two u64 varints) fits in 32
        // bytes. Avoid decoding an arbitrarily large inline array to reject it.
        if data.len() > 32 {
            return Ok(None);
        }
        let sequence = read_row_ids(data)?;
        let [U64Segment::Range(range)] = sequence.segments() else {
            return Ok(None);
        };
        // Leave legacy or unrepresentable physical addresses to the existing index.
        if u32::try_from(fragment.id).is_err()
            || range.end.saturating_sub(range.start) > u64::from(u32::MAX) + 1
            || slot > 0 && table.manifest.fragments[slot - 1].id >= fragment.id
        {
            return Ok(None);
        }
        if range.is_empty() {
            continue;
        }
        ranges.push(InlineRowIdRange {
            start: range.start,
            end: range.end,
            prefix_end: 0,
            slot,
        });
    }
    ranges.sort_unstable_by_key(|range| range.start);
    let mut prefix_end = 0;
    for range in &mut ranges {
        prefix_end = prefix_end.max(range.end);
        range.prefix_end = prefix_end;
    }
    Ok(Some(ranges))
}

/// Resolve one stable ID using only a dense inline snapshot and its candidate
/// deletion files. The outer None means the snapshot needs the general index;
/// the inner None means the ID is missing or deleted.
async fn lookup_inline_row_id(
    table: &Table,
    row_id: u64,
    with_deletions: bool,
    cache: &LanceCache,
) -> Result<Option<Option<RowAddress>>> {
    let ranges = if let Some(e_tag) = table.state.manifest_location.e_tag.as_deref() {
        let key = InlineRowIdRangesKey {
            version: table.manifest.version,
            e_tag,
        };
        cache
            .get_or_insert_with_key(key, || async { inline_row_id_ranges(table) })
            .await?
    } else {
        // As with RowIdIndex, an unversioned cache identity cannot be shared
        // across a drop and recreate at the same URI.
        Arc::new(inline_row_id_ranges(table)?)
    };
    let Some(ranges) = ranges.as_ref() else {
        return Ok(None);
    };
    let lower = ranges.partition_point(|range| range.prefix_end <= row_id);
    let upper = ranges.partition_point(|range| range.start <= row_id);
    let mut lookups = Vec::new();
    for idx in lower..upper {
        let range = &ranges[idx];
        if range.end <= row_id {
            continue;
        }
        let fragment = &table.manifest.fragments[range.slot];
        let address = RowAddress::new_from_parts(fragment.id as u32, (row_id - range.start) as u32);
        // The general ignore-deletions index also applies tombstones to a
        // fragment whose span overlaps another, even for its unique IDs.
        let overlaps = idx > 0 && ranges[idx - 1].prefix_end > range.start
            || ranges
                .get(idx + 1)
                .is_some_and(|next| next.start < range.end);
        lookups.push(async move {
            if (with_deletions || overlaps)
                && let Some(file) = &fragment.deletion_file
                && read_table_deletion_file(table, fragment.id, file)
                    .await?
                    .contains(address.row_offset())
            {
                return Ok(None);
            }
            Ok::<_, Error>(Some(address))
        });
    }
    let mut live = None;
    let mut candidates =
        futures::stream::iter(lookups).buffer_unordered(table.state.object_store.io_parallelism());
    while let Some(candidate) = candidates.try_next().await? {
        if let Some(address) = candidate
            && live.replace(address).is_some()
        {
            return Err(Error::internal(format!(
                "row id index corrupt: stable row id {row_id} is live in multiple fragments"
            )));
        }
    }
    Ok(Some(live))
}

#[doc(hidden)]
pub async fn try_resolve_point_take_row_id(
    table: &Table,
    row_id: u64,
    with_deletions: bool,
) -> Result<Option<Option<RowAddress>>> {
    let cache = table
        .state
        .session
        .index_cache()
        .with_key_prefix(table.uri());
    if let Some(e_tag) = table.state.manifest_location.e_tag.as_deref()
        && let Some(index) = cache
            .get_with_key(&RowIdIndexKey {
                version: table.manifest.version,
                with_deletions,
                e_tag: Some(e_tag),
            })
            .await
    {
        return Ok(Some(index.get(row_id)?));
    }
    lookup_inline_row_id(table, row_id, with_deletions, &cache).await
}

pub async fn resolve_take_row_ids(
    table: &Table,
    row_ids: &[u64],
    with_deletions: bool,
) -> Result<Option<Vec<Option<RowAddress>>>> {
    if !table.manifest.uses_stable_row_ids() {
        return Ok(None);
    }
    if let [id] = row_ids
        && let Some(address) = try_resolve_point_take_row_id(table, *id, with_deletions).await?
    {
        return Ok(Some(vec![address]));
    }
    let cache = table
        .state
        .session
        .index_cache()
        .with_key_prefix(table.uri());
    get_row_id_index_with_cache(table, with_deletions, &cache)
        .await?
        .map(|index| index.get_many(row_ids))
        .transpose()
}
