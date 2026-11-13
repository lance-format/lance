// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Resolve requested IDs while preserving complete-index lookup semantics.

use std::ops::{Range, RangeInclusive};
use std::sync::{Arc, Weak};

use futures::{StreamExt, TryStreamExt};
use lance_core::cache::InternalCacheKey;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::DeletionVector;
use lance_core::utils::tokio::spawn_cpu;
use lance_table::format::{Fragment, RowIdMeta};
use lance_table::rowids::{FragmentRowIdIndex, RowIdIndex, segment::U64Segment};
use roaring::RoaringBitmap;

use crate::session::index_caches::{
    PreparedRowIdFragmentKey, RowIdFragmentAliasKey, RowIdFragmentDeletionKey, RowIdFragmentSpanKey,
};
use crate::{Dataset, Error, Result};

use super::cache::{deletion_identity, load_deletions, load_source, sequence_identity};
use super::{get_row_id_index_with, read_row_id_sequence};

/// Small encoded inline sources need no CPU-pool dispatch to prepare a lookup.
const INLINE_PREPARE_BYTES: usize = 16 * 1024;
/// Bound decoded source rows while allowing small fragments to retain remote
/// I/O concurrency. This schedules work; it is not an RSS limit.
const PREPARE_ROW_BUDGET: usize = 192 * 1024 * 1024;
/// A cached empty span proves the immutable source has no IDs. The metadata
/// entry holds no sequence, deletion vector or prepared lookup allocation.
#[derive(Debug)]
pub struct FragmentSpan(Option<RangeInclusive<u64>>);

impl DeepSizeOf for FragmentSpan {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

/// One cache-owned fragment lookup. Active callers keep the same allocation
/// alive through eviction; preparation registries retain only weak references.
#[derive(Debug)]
pub struct PreparedFragment {
    index: Option<Arc<RowIdIndex>>,
    span: Option<RangeInclusive<u64>>,
    /// Deletions already applied to the lookup, not necessarily the current snapshot.
    deletion: Option<InternalCacheKey>,
    source_rows: u64,
    is_range_only: bool,
    /// Physical offsets retained by a pruned lookup. This proves whether a later
    /// snapshot can reuse it without retaining the original row-ID sequence.
    retained: Option<RoaringBitmap>,
}

/// A validated view does not retain or charge a second copy of its mapping.
#[derive(Debug)]
pub struct FragmentAlias(Weak<PreparedFragment>);

impl DeepSizeOf for FragmentAlias {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

impl DeepSizeOf for PreparedFragment {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.index.deep_size_of_children(context)
            + self
                .retained
                .as_ref()
                .map_or(0, RoaringBitmap::serialized_size)
    }
}

impl PreparedFragment {
    fn should_rebuild(&self, deleted_rows: usize) -> bool {
        let retained = self
            .retained
            .as_ref()
            .map_or(self.source_rows, RoaringBitmap::len);
        let live = self.source_rows.saturating_sub(deleted_rows as u64);
        // Reclaim at least half the remaining entries before paying another
        // source read/build. Range lookups are already compact without pruning.
        !self.is_range_only && live < retained && live <= retained / 2
    }

    fn usable_without_validation(
        &self,
        fragment: &Fragment,
        deletion: Option<InternalCacheKey>,
    ) -> bool {
        if self.deletion == deletion {
            return true;
        }
        self.deletion.is_none()
            && (self.is_range_only
                || fragment
                    .deletion_file
                    .as_ref()
                    .and_then(|file| file.num_deleted_rows)
                    .is_some_and(|count| !self.should_rebuild(count)))
    }

    fn covers_snapshot(&self, deleted: &DeletionVector) -> bool {
        let Some(retained) = &self.retained else {
            return true;
        };
        let removed = self.source_rows - retained.len();
        if removed == 0 {
            return true;
        }
        let outside_retained = match deleted {
            DeletionVector::NoDeletions => 0,
            DeletionVector::Set(set) => set
                .iter()
                .filter(|offset| {
                    u64::from(**offset) < self.source_rows && !retained.contains(**offset)
                })
                .count() as u64,
            DeletionVector::Bitmap(bitmap) => {
                // Retained offsets are in the source domain; ignore any invalid
                // out-of-domain deletions rather than using them as reuse proof.
                let Some(last) = self
                    .source_rows
                    .checked_sub(1)
                    .and_then(|n| u32::try_from(n).ok())
                else {
                    return false;
                };
                bitmap.range_cardinality(..=last) - bitmap.intersection_len(retained)
            }
        };
        outside_retained == removed
    }
}

/// Resolve all unknown candidates; cached fragment lookups answer different IDs
/// without keeping original row-ID arrays or caching individual query answers.
pub async fn resolve_row_ids(
    dataset: &Dataset,
    row_ids: &[u64],
    with_deletions: bool,
) -> Result<Option<Vec<Option<RowAddress>>>> {
    if !dataset.manifest.uses_stable_row_ids() {
        return Ok(None);
    }
    // Allocation uses an exclusive high-water mark, preserved by clone/restore.
    // It cannot route lower IDs or prove that a partially searched ID is absent.
    if row_ids.is_empty() {
        return Ok(Some(Vec::new()));
    }
    if with_deletions && row_ids.iter().all(|id| *id >= dataset.manifest.next_row_id) {
        return Ok(Some(vec![None; row_ids.len()]));
    }
    if !with_deletions || dataset.manifest_location.e_tag.is_none() {
        return get_row_id_index_with(dataset, with_deletions)
            .await?
            .map(|index| index.get_many(row_ids))
            .transpose();
    }
    // Sort once for fragment-bound slicing, but retain every original position.
    // Only the exclusive allocation bound can discard IDs without inspecting
    // candidate sources; a match never ends the search for another live copy.
    let mut sorted: Vec<_> = row_ids
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, id)| *id < dataset.manifest.next_row_id)
        .map(|(position, id)| (id, position))
        .collect();
    sorted.sort_unstable_by_key(|&(id, _)| id);
    let mut unique_ids: Vec<_> = sorted.iter().map(|&(id, _)| id).collect();
    unique_ids.dedup();
    let mut found = vec![None; unique_ids.len()];
    let mut misses = Vec::with_capacity(dataset.manifest.fragments.len());
    let mut cached_fragments = 0;
    let mut excluded_fragments = 0;
    let mut resident = Vec::new();
    // Consume every hit before admitting misses: a scan under cache pressure
    // must not evict a useful resident entry before this request has queried it.
    for fragment in dataset.manifest.fragments.iter() {
        let key = PreparedRowIdFragmentKey {
            source: sequence_identity(dataset, fragment)?,
            deletion: deletion_identity(dataset, fragment, true)?,
        };
        let mut range = 0..unique_ids.len();
        if let Some(cached) = cached_fragment(dataset, fragment, &key).await {
            range = candidate_range(&cached.span, &unique_ids);
            if range.is_empty() {
                excluded_fragments += 1;
                continue;
            }
            resident.push((fragment, key.deletion, cached, range));
            cached_fragments += 1;
            continue;
        }
        let span_key = RowIdFragmentSpanKey { source: key.source };
        if let Some(span) = dataset.metadata_cache.get_with_key(&span_key).await {
            range = candidate_range(&span.0, &unique_ids);
            if range.is_empty() {
                excluded_fragments += 1;
                continue;
            }
        }
        misses.push((fragment, key, range));
    }
    // Consume every resident hit before admitting any source miss.
    let mut resident_queries = Vec::with_capacity(resident.len());
    for (fragment, deletion, prepared, range) in resident {
        let ids = &unique_ids;
        resident_queries.push(async move {
            let addresses =
                lookup_fragment(dataset, fragment, deletion, &prepared, &ids[range.clone()])
                    .await?;
            Ok::<_, Error>((range, addresses))
        });
    }
    let mut resident_loads = futures::stream::iter(resident_queries)
        .buffer_unordered(dataset.object_store.io_parallelism());
    while let Some((range, addresses)) = resident_loads.try_next().await? {
        record_matches(
            &unique_ids[range.clone()],
            &mut found[range.clone()],
            addresses,
        )?;
    }
    let parallelism = preparation_parallelism(
        dataset,
        misses.iter().map(|(fragment, _, _)| fragment.physical_rows),
    );
    let mut loads = Vec::with_capacity(misses.len());
    for (fragment, key, range) in misses {
        let ids = &unique_ids;
        loads.push(async move {
            let prepared = prepare_shared(dataset, fragment, &key, &ids[range]).await?;
            // A newly read source can narrow the candidate slice further than
            // the metadata available when the request was first scheduled.
            let range = candidate_range(&prepared.span, ids);
            let addresses = lookup_fragment(
                dataset,
                fragment,
                key.deletion,
                &prepared,
                &ids[range.clone()],
            )
            .await?;
            Ok::<_, Error>((prepared, range, addresses))
        });
    }
    let mut loads = futures::stream::iter(loads).buffer_unordered(parallelism);
    let mut loaded_fragments = 0;
    let mut prepared_fragments = 0;
    while let Some((prepared, range, addresses)) = loads.try_next().await? {
        loaded_fragments += 1;
        if prepared.index.is_some() {
            record_matches(&unique_ids[range.clone()], &mut found[range], addresses)?;
            prepared_fragments += 1;
        }
    }
    tracing::debug!(
        event = "row_id_fragment_resolution",
        total_fragments = dataset.manifest.fragments.len(),
        cached_fragments,
        loaded_fragments,
        prepared_fragments,
        excluded_fragments,
        preparation_parallelism = parallelism,
        requested_ids = row_ids.len(),
        unique_requested_ids = unique_ids.len(),
        fragment_batch_calls = cached_fragments + prepared_fragments,
        "resolved row ids with fragment-only cache"
    );
    let mut result = vec![None; row_ids.len()];
    let mut unique_position = 0;
    for (id, original_position) in sorted {
        if unique_ids[unique_position] != id {
            unique_position += 1;
        }
        result[original_position] = found[unique_position];
    }
    Ok(Some(result))
}

/// Min/max bounds are inclusive and can only exclude candidates, including
/// for unsorted or holed sources. Partitioning avoids overflowing an upper ID.
fn candidate_range(span: &Option<RangeInclusive<u64>>, ids: &[u64]) -> Range<usize> {
    let Some(span) = span else {
        return 0..0;
    };
    let start = ids.partition_point(|id| *id < *span.start());
    let end = ids.partition_point(|id| *id <= *span.end());
    start..end
}

fn preparation_parallelism(dataset: &Dataset, rows: impl Iterator<Item = Option<usize>>) -> usize {
    let largest = rows
        .map(|rows| rows.unwrap_or(PREPARE_ROW_BUDGET))
        .max()
        .unwrap_or(PREPARE_ROW_BUDGET)
        .max(1);
    (PREPARE_ROW_BUDGET / largest)
        .max(1)
        .min(dataset.object_store.io_parallelism())
}

async fn prepare_shared(
    dataset: &Dataset,
    fragment: &Fragment,
    key: &PreparedRowIdFragmentKey,
    row_ids: &[u64],
) -> Result<Arc<PreparedFragment>> {
    let source_key = PreparedRowIdFragmentKey {
        source: key.source,
        deletion: None,
    };
    let build = dataset
        .session()
        .row_id_index_builds
        .for_fragment(&dataset.uri, &source_key)?;
    let mut slot = build.lock().await;
    if let Some(active) = slot.as_ref()
        && active.usable_without_validation(fragment, key.deletion)
        && (active.index.is_some() || candidate_range(&active.span, row_ids).is_empty())
    {
        return Ok(active.clone());
    }
    if let Some(cached) = cached_fragment(dataset, fragment, key).await {
        *slot = Some(cached.clone());
        return Ok(cached);
    }
    let previous = match slot.as_ref() {
        Some(active) if active.index.is_some() => Some(active.clone()),
        _ => match dataset
            .index_cache
            .get_with_key(&RowIdFragmentAliasKey(source_key))
            .await
            .and_then(|alias| alias.0.upgrade())
        {
            Some(previous) => Some(previous),
            None => dataset.index_cache.get_with_key(&source_key).await,
        },
    };
    let mut deletions = None;
    let mut reusable = None;
    if let Some(previous) = previous
        && !fragment
            .deletion_file
            .as_ref()
            .and_then(|file| file.num_deleted_rows)
            .is_some_and(|count| previous.should_rebuild(count))
    {
        let deleted = match key.deletion {
            Some(identity) => load_deletions(dataset, fragment, identity).await?,
            None => Arc::new(DeletionVector::default()),
        };
        if previous.index.is_some()
            && previous.covers_snapshot(&deleted)
            && !previous.should_rebuild(deleted.len())
        {
            reusable = Some(previous);
        }
        deletions = Some(deleted);
    }
    let active = match reusable {
        Some(prepared) => prepared,
        None => Arc::new(prepare_fragment(dataset, fragment, key, row_ids, deletions).await?),
    };
    if active.index.is_some() {
        // One charged entry owns each distinct mapping. Latest-source and
        // validated-snapshot aliases are weak, so sharing never multiplies its
        // cache weight or keeps an evicted mapping alive.
        let prepared_key = PreparedRowIdFragmentKey {
            source: key.source,
            deletion: active.deletion,
        };
        dataset
            .index_cache
            .insert_with_key(&prepared_key, active.clone())
            .await;
        dataset
            .index_cache
            .insert_with_key(
                &RowIdFragmentAliasKey(source_key),
                Arc::new(FragmentAlias(Arc::downgrade(&active))),
            )
            .await;
        if key.deletion != active.deletion {
            dataset
                .index_cache
                .insert_with_key(
                    &RowIdFragmentAliasKey(*key),
                    Arc::new(FragmentAlias(Arc::downgrade(&active))),
                )
                .await;
        }
    }
    dataset
        .session()
        .row_id_index_builds
        .prepared(&dataset.uri, &source_key, &active)?;
    *slot = Some(active.clone());
    Ok(active)
}

async fn cached_fragment(
    dataset: &Dataset,
    fragment: &Fragment,
    key: &PreparedRowIdFragmentKey,
) -> Option<Arc<PreparedFragment>> {
    let source_key = PreparedRowIdFragmentKey {
        source: key.source,
        deletion: None,
    };
    if let Some(prepared) = dataset.index_cache.get_with_key(key).await {
        return Some(prepared);
    }
    if key.deletion.is_some()
        && let Some(prepared) = dataset
            .index_cache
            .get_with_key(&RowIdFragmentAliasKey(*key))
            .await
            .and_then(|alias| alias.0.upgrade())
    {
        return Some(prepared);
    }
    if let Some(prepared) = dataset
        .index_cache
        .get_with_key(&RowIdFragmentAliasKey(source_key))
        .await
        .and_then(|alias| alias.0.upgrade())
        && prepared.usable_without_validation(fragment, key.deletion)
    {
        return Some(prepared);
    }
    // An older snapshot can still use the resident full source map. The same
    // threshold check prevents it from bypassing a needed deletion rebuild.
    dataset
        .index_cache
        .get_with_key(&source_key)
        .await
        .filter(|prepared| prepared.usable_without_validation(fragment, key.deletion))
}

async fn lookup_fragment(
    dataset: &Dataset,
    fragment: &Fragment,
    deletion: Option<InternalCacheKey>,
    prepared: &PreparedFragment,
    row_ids: &[u64],
) -> Result<Vec<Option<RowAddress>>> {
    let Some(index) = &prepared.index else {
        return Ok(Vec::new());
    };
    if prepared.deletion != deletion
        && let Some(deletion) = deletion
    {
        let deleted = load_deletions(dataset, fragment, deletion).await?;
        // The filter runs inside the probe, before duplicate validation. An old
        // deleted copy and a new live copy may occur in different source segments.
        return index
            .get_many_with_filter(row_ids, |address| !deleted.contains(address.row_offset()));
    }
    index.get_many(row_ids)
}

async fn prepare_fragment(
    dataset: &Dataset,
    fragment: &Fragment,
    key: &PreparedRowIdFragmentKey,
    row_ids: &[u64],
    mut deleted: Option<Arc<DeletionVector>>,
) -> Result<PreparedFragment> {
    let sequence = Arc::new(read_row_id_sequence(dataset, fragment).await?);
    let span = sequence.row_id_range();
    let source_rows = sequence.len();
    let is_range_only = sequence
        .segments()
        .iter()
        .all(|segment| matches!(segment, U64Segment::Range(_)));
    dataset
        .metadata_cache
        .insert_with_key(
            &RowIdFragmentSpanKey { source: key.source },
            Arc::new(FragmentSpan(span.clone())),
        )
        .await;
    if candidate_range(&span, row_ids).is_empty() {
        return Ok(PreparedFragment {
            index: None,
            span,
            deletion: None,
            source_rows,
            is_range_only,
            retained: None,
        });
    }
    let is_small_inline = matches!(&fragment.row_id_meta,
        Some(RowIdMeta::Inline(bytes)) if bytes.len() <= INLINE_PREPARE_BYTES);
    let build = |deletion_vector| {
        let source = FragmentRowIdIndex {
            fragment_id: fragment.id as u32,
            row_id_sequence: sequence.clone(),
            deletion_vector,
        };
        async move {
            let prepare = move || RowIdIndex::new(&[source]);
            if is_small_inline {
                prepare()
            } else {
                spawn_cpu(prepare).await
            }
        }
    };
    if !is_range_only && deleted.is_none() && key.deletion.is_some() {
        deleted = Some(
            load_source(dataset, fragment, sequence.clone(), true)
                .await?
                .deletion_vector,
        );
    }
    let mut deletion = key.deletion.filter(|_| {
        !is_range_only
            && deleted
                .as_ref()
                .is_some_and(|deleted| deleted.len() as u64 >= source_rows.div_ceil(2))
    });
    let mut pruned = match (deletion, &deleted) {
        (Some(_), Some(deleted)) => deleted.clone(),
        _ => Arc::new(DeletionVector::default()),
    };
    let index = match build(pruned.clone()).await {
        Ok(index) => index,
        // A merged map cannot represent repeated raw IDs. Preserve the previous
        // snapshot-aware construction for such sources, using the loaded sequence.
        Err(_) if deletion.is_none() && key.deletion.is_some() => {
            pruned = match &deleted {
                Some(deleted) => deleted.clone(),
                None => {
                    load_source(dataset, fragment, sequence.clone(), true)
                        .await?
                        .deletion_vector
                }
            };
            deletion = key.deletion;
            build(pruned.clone()).await?
        }
        Err(error) => return Err(error),
    };
    // A folded deletion vector is no longer needed by a merged lookup. Cache
    // visibility only when queries still need it, avoiding a second large mask
    // alongside a small, already-pruned map under cache pressure.
    if deletion.is_none()
        && let (Some(identity), Some(deleted)) = (key.deletion, deleted)
    {
        dataset
            .index_cache
            .insert_with_key(&RowIdFragmentDeletionKey { deletion: identity }, deleted)
            .await;
    }
    let retained = if deletion.is_some() {
        let mut retained = RoaringBitmap::new();
        if let Some(last) = source_rows.checked_sub(1) {
            let last = u32::try_from(last).map_err(|_| {
                Error::corrupt_file_named(
                    &fragment.id.to_string(),
                    "row-ID source exceeds the physical offset domain".to_owned(),
                )
            })?;
            retained.insert_range(..=last);
        }
        match pruned.as_ref() {
            DeletionVector::Bitmap(bitmap) => retained -= bitmap,
            DeletionVector::Set(set) => {
                for &offset in set {
                    retained.remove(offset);
                }
            }
            DeletionVector::NoDeletions => {}
        }
        Some(retained)
    } else {
        None
    };
    Ok(PreparedFragment {
        index: Some(Arc::new(index)),
        span,
        deletion,
        source_rows,
        is_range_only,
        retained,
    })
}

fn record_matches(
    row_ids: &[u64],
    found: &mut [Option<RowAddress>],
    addresses: Vec<Option<RowAddress>>,
) -> Result<()> {
    debug_assert_eq!(row_ids.len(), found.len(), "candidate output slice length");
    debug_assert_eq!(
        row_ids.len(),
        addresses.len(),
        "fragment lookup result length"
    );
    for ((row_id, found), address) in row_ids.iter().zip(found).zip(addresses) {
        if let Some(address) = address
            && found.replace(address).is_some()
        {
            return Err(Error::internal(format!(
                "row id index corrupt: stable row id {row_id} is live in multiple fragments"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, UInt64Type};
    use lance_core::cache::CacheKey;
    use lance_table::rowids::RowIdSequence;
    use lance_table::rowids::version::RowDatasetVersionSequence;
    use lance_table::transaction::{Operation, Transaction};
    use rstest::rstest;

    use super::*;
    use crate::dataset::rowids::{RowLineage, get_row_id_index, place_row_lineage};
    use crate::dataset::write::CommitBuilder;
    use crate::dataset::{ProjectionRequest, WriteParams};
    use crate::session::Session;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

    async fn scan_address(dataset: &Dataset, value: i32) -> RowAddress {
        let mut scan = dataset.scan();
        scan.with_row_address()
            .filter(&format!("i = {value}"))
            .unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 1);
        RowAddress::from(
            batch[lance_core::ROW_ADDR]
                .as_primitive::<UInt64Type>()
                .value(0),
        )
    }

    fn cold_dataset(mut dataset: Dataset, capacity: usize, metadata_capacity: usize) -> Dataset {
        let session = Arc::new(Session::new(
            capacity,
            metadata_capacity,
            dataset.session().store_registry(),
        ));
        dataset.index_cache = Arc::new(session.index_cache.for_dataset(&dataset.uri));
        dataset.metadata_cache = Arc::new(session.metadata_cache.for_dataset(&dataset.uri));
        dataset.session = session;
        dataset
    }

    async fn shuffled_dataset(spilled: bool, overlapping: bool) -> Dataset {
        let mut dataset = super::super::cache::tests::dataset(false).await;
        if spilled {
            dataset
                .update_config([
                    (super::super::SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                    (super::super::INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
                ])
                .await
                .unwrap();
        }
        let mut fragments = dataset.manifest.fragments.as_ref().clone();
        for (number, fragment) in fragments.iter_mut().enumerate() {
            let lineage = RowLineage {
                row_ids: RowIdSequence::try_from_iter((0..10).map(|offset| {
                    if overlapping {
                        ((number * 10 + offset) as u64 * 7) % 40
                    } else {
                        (number * 10 + (offset * 3) % 10) as u64
                    }
                }))
                .unwrap(),
                created_at: RowDatasetVersionSequence::from_uniform_row_count(10, 1),
                last_updated_at: RowDatasetVersionSequence::from_uniform_row_count(10, 1),
            };
            place_row_lineage(&dataset, &lineage)
                .await
                .unwrap()
                .apply(fragment);
        }
        let transaction = Transaction::new_from_version(
            dataset.version().version,
            Operation::Merge {
                fragments,
                schema: dataset.schema().clone(),
                preserves_nullability: true,
            },
        );
        CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await
            .unwrap()
    }

    async fn large_shuffled_dataset() -> Dataset {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset_with_params(
                FragmentCount::from(1),
                FragmentRowCount::from(4096),
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: 4096,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        dataset
            .update_config([
                (super::super::SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                (super::super::INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
            ])
            .await
            .unwrap();
        let lineage = RowLineage {
            row_ids: RowIdSequence::try_from_iter((0..4096).map(|i| (i * 2053) % 4096)).unwrap(),
            created_at: RowDatasetVersionSequence::from_uniform_row_count(4096, 1),
            last_updated_at: RowDatasetVersionSequence::from_uniform_row_count(4096, 1),
        };
        let mut fragments = dataset.manifest.fragments.as_ref().clone();
        place_row_lineage(&dataset, &lineage)
            .await
            .unwrap()
            .apply(&mut fragments[0]);
        let transaction = Transaction::new_from_version(
            dataset.version().version,
            Operation::Merge {
                fragments,
                schema: dataset.schema().clone(),
                preserves_nullability: true,
            },
        );
        dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await
            .unwrap();
        dataset
    }

    #[tokio::test]
    async fn warm_deleted_fragment_rebuilds_and_reuses_the_smaller_lookup() {
        let mut dataset = cold_dataset(large_shuffled_dataset().await, 1 << 20, 0);
        let old = dataset.clone();
        let fragment_id = dataset.manifest.fragments[0].id as u32;
        let key = PreparedRowIdFragmentKey {
            source: sequence_identity(&dataset, &dataset.manifest.fragments[0]).unwrap(),
            deletion: None,
        };
        let ids = [0, 80, 2053, 4096, 0];
        assert_eq!(
            resolve_row_ids(&dataset, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            [Some(0), Some(16), Some(1), None, Some(0)]
                .map(|offset| offset.map(|offset| RowAddress::new_from_parts(fragment_id, offset)))
        );
        let initial = dataset.index_cache.get_with_key(&key).await.unwrap();
        let initial_bytes = initial.index.as_ref().unwrap().deep_size_of();
        dataset.delete("i = 3").await.unwrap();
        resolve_row_ids(&dataset, &ids, true).await.unwrap();
        let lightly_deleted = dataset.clone();
        dataset.delete("i % 16 != 0").await.unwrap();
        assert_eq!(
            resolve_row_ids(&dataset, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            [Some(0), Some(16), None, None, Some(0)]
                .map(|offset| offset.map(|offset| RowAddress::new_from_parts(fragment_id, offset)))
        );
        let pruned_key = PreparedRowIdFragmentKey {
            deletion: deletion_identity(&dataset, &dataset.manifest.fragments[0], true).unwrap(),
            ..key
        };
        let compacted = dataset.index_cache.get_with_key(&pruned_key).await.unwrap();
        assert!(
            compacted.index.as_ref().unwrap().deep_size_of() * 4 < initial_bytes,
            "a resident lookup must release most deleted mappings"
        );
        let first_deleted = dataset.clone();
        dataset.delete("i = 16").await.unwrap();
        let fragment = &dataset.manifest.fragments[0];
        load_deletions(
            &dataset,
            fragment,
            deletion_identity(&dataset, fragment, true)
                .unwrap()
                .unwrap(),
        )
        .await
        .unwrap();
        let charged_before = dataset.index_cache.size_bytes().await;
        let before = dataset.object_store.io_stats_snapshot();
        assert_eq!(
            resolve_row_ids(&dataset, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            [Some(0), None, None, None, Some(0)]
                .map(|offset| offset.map(|offset| RowAddress::new_from_parts(fragment_id, offset)))
        );
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_bytes,
            before.read_bytes,
            "one more deletion must not reread the source after a major rebuild"
        );
        let next = cached_fragment(
            &dataset,
            fragment,
            &PreparedRowIdFragmentKey {
                deletion: deletion_identity(&dataset, fragment, true).unwrap(),
                ..key
            },
        )
        .await
        .unwrap();
        assert!(Arc::ptr_eq(
            compacted.index.as_ref().unwrap(),
            next.index.as_ref().unwrap()
        ));
        let alias_charge = dataset.index_cache.size_bytes().await - charged_before;
        assert!(
            alias_charge < 256,
            "snapshot aliases must not charge the shared mapping twice"
        );
        dataset.delete("i % 32 != 0").await.unwrap();
        let fragment = &dataset.manifest.fragments[0];
        let final_key = PreparedRowIdFragmentKey {
            deletion: deletion_identity(&dataset, fragment, true).unwrap(),
            ..key
        };
        resolve_row_ids(&dataset, &[0, 160], true).await.unwrap();
        let final_map = dataset.index_cache.get_with_key(&final_key).await.unwrap();
        assert!(!Arc::ptr_eq(&compacted, &final_map));
        assert_eq!(final_map.retained.as_ref().unwrap().len(), 128);
        let before = dataset.object_store.io_stats_snapshot();
        assert_eq!(
            resolve_row_ids(&lightly_deleted, &[2053], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(RowAddress::new_from_parts(fragment_id, 1))]
        );
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_bytes,
            before.read_bytes,
            "an older snapshot must reuse a compatible resident source mapping"
        );
        assert_eq!(
            resolve_row_ids(&first_deleted, &[80], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(RowAddress::new_from_parts(fragment_id, 16))]
        );
        assert_eq!(
            resolve_row_ids(&old, &[2053], true).await.unwrap().unwrap(),
            vec![Some(RowAddress::new_from_parts(fragment_id, 1))]
        );
        let reclaimed = Arc::downgrade(&final_map);
        drop(final_map);
        dataset.index_cache.clear().await;
        assert!(
            reclaimed.upgrade().is_none(),
            "weak aliases must not retain evicted mappings"
        );
    }

    #[tokio::test]
    async fn cold_deleted_fragment_fits_the_live_mapping_budget() {
        let mut dataset = large_shuffled_dataset().await;
        dataset.delete("i % 16 != 0").await.unwrap();
        let fragment = dataset.manifest.fragments[0].id as u32;
        let dataset = cold_dataset(dataset, 4096, 0);
        let ids = [0, 80, 4096, 2053, 0];
        let expected = [
            Some(RowAddress::new_from_parts(fragment, 0)),
            Some(RowAddress::new_from_parts(fragment, 16)),
            None,
            None,
            Some(RowAddress::new_from_parts(fragment, 0)),
        ];
        assert_eq!(
            resolve_row_ids(&dataset, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            expected
        );
        let before = dataset.object_store.io_stats_snapshot();
        for _ in 0..3 {
            assert_eq!(
                resolve_row_ids(&dataset, &ids, true)
                    .await
                    .unwrap()
                    .unwrap(),
                expected
            );
        }
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_bytes,
            before.read_bytes,
            "a cold deleted fragment must retain its live lookup in the small cache"
        );
        assert!(dataset.index_cache.size_bytes().await <= 4096);
    }

    #[rstest]
    #[case::inline(false)]
    #[case::spilled(true)]
    #[tokio::test]
    async fn pruned_lookup_does_not_hide_rows_after_restore(#[case] spilled: bool) {
        let mut dataset = cold_dataset(shuffled_dataset(spilled, true).await, 1 << 20, 0);
        let mut restored = dataset.clone();
        resolve_row_ids(&dataset, &[0, 7, 17, 27, 37], true)
            .await
            .unwrap();
        dataset.delete("i % 10 != 0").await.unwrap();
        assert_eq!(
            resolve_row_ids(&dataset, &[0, 7, 17, 27, 37], true)
                .await
                .unwrap()
                .unwrap(),
            vec![
                Some(RowAddress::new_from_parts(0, 0)),
                None,
                None,
                None,
                None
            ]
        );
        restored.restore().await.unwrap();
        restored.delete("i % 10 != 1").await.unwrap();
        let ids = [7, 37, 27, 17, 7, 0, u64::MAX];
        let expected = [Some(0), Some(1), Some(2), Some(3), Some(0), None, None]
            .map(|fragment| fragment.map(|fragment| RowAddress::new_from_parts(fragment, 1)));
        for _ in 0..2 {
            let (before, after) = tokio::join!(
                resolve_row_ids(&dataset, &ids, true),
                resolve_row_ids(&restored, &ids, true)
            );
            assert_eq!(after.unwrap().unwrap(), expected);
            assert_eq!(
                before.unwrap().unwrap(),
                [
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(RowAddress::new_from_parts(0, 0)),
                    None
                ]
            );
        }
        let batch = restored
            .take_rows(
                &ids,
                ProjectionRequest::from_columns(["i"], restored.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            &[1, 11, 21, 31, 1]
        );
    }

    #[rstest]
    #[case::inline(false)]
    #[case::spilled(true)]
    #[tokio::test]
    async fn distinct_ids_and_batches_reuse_fragment_lookups(#[case] spilled: bool) {
        let dataset = cold_dataset(shuffled_dataset(spilled, false).await, 1 << 20, 1 << 20);
        let expected_7 = scan_address(&dataset, 9).await;
        let expected_31 = scan_address(&dataset, 37).await;
        let before = dataset.object_store.io_stats_snapshot();
        assert_eq!(
            resolve_row_ids(&dataset, &[7], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(expected_7)]
        );
        let after = dataset.object_store.io_stats_snapshot();
        if spilled {
            assert!(after.read_bytes > before.read_bytes);
        }
        let first_key = PreparedRowIdFragmentKey {
            source: sequence_identity(&dataset, &dataset.manifest.fragments[0]).unwrap(),
            deletion: None,
        };
        let first = dataset.index_cache.get_with_key(&first_key).await.unwrap();
        assert_eq!(
            resolve_row_ids(&dataset, &[4], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(RowAddress::new_from_parts(
                dataset.manifest.fragments[0].id as u32,
                8
            ))]
        );
        assert!(Arc::ptr_eq(
            &first,
            &dataset.index_cache.get_with_key(&first_key).await.unwrap()
        ));
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_bytes,
            after.read_bytes
        );
        assert_eq!(
            resolve_row_ids(&dataset, &[31], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(expected_31)]
        );
        assert!(
            dataset
                .index_cache
                .get_with_key(&crate::session::index_caches::RowIdIndexKey {
                    version: dataset.manifest.version,
                    with_deletions: true,
                    e_tag: dataset.manifest_location.e_tag.as_deref(),
                })
                .await
                .is_none()
        );
        let batch = dataset
            .take_rows(
                &[31, 7, 31],
                ProjectionRequest::from_columns(["i"], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            &[37, 9, 37]
        );
        assert!(
            dataset
                .index_cache
                .get_with_key(&crate::session::index_caches::RowIdIndexKey {
                    version: dataset.manifest.version,
                    with_deletions: true,
                    e_tag: dataset.manifest_location.e_tag.as_deref(),
                })
                .await
                .is_none(),
            "a batch must reuse fragment lookups without forcing a complete map"
        );
    }

    #[rstest]
    #[case::two(2)]
    #[case::lane(32)]
    #[case::large(1000)]
    #[tokio::test]
    async fn batches_preserve_physical_addresses_order_and_missing(
        #[case] count: usize,
        #[values(false, true)] spilled: bool,
        #[values(false, true)] overlapping: bool,
    ) {
        let dataset = cold_dataset(
            shuffled_dataset(spilled, overlapping).await,
            1 << 20,
            1 << 20,
        );
        let ids: Vec<_> = (0..count)
            .map(|position| (position * 17 % 41) as u64)
            .collect();
        // Invert the fixture's mathematical permutation, independently of the
        // reverse index and without a scan that could warm its metadata.
        let expected: Vec<_> = ids
            .iter()
            .map(|&id| {
                if id >= 40 {
                    return None;
                }
                let physical = if overlapping {
                    (id * 23) % 40
                } else {
                    (id / 10) * 10 + (id % 10 * 7) % 10
                };
                Some(RowAddress::new_from_parts(
                    dataset.manifest.fragments[physical as usize / 10].id as u32,
                    physical as u32 % 10,
                ))
            })
            .collect();
        assert_eq!(
            resolve_row_ids(&dataset, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            expected
        );
        assert_eq!(
            resolve_row_ids(&dataset, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            expected
        );
        // Repeated batches reuse the prepared fragment working set.
        // Subsequent requests must reuse either that view or fragment state.
        let before = dataset.object_store.io_stats_snapshot();
        assert_eq!(
            resolve_row_ids(&dataset, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            expected
        );
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_bytes,
            before.read_bytes
        );
        let batch = dataset
            .take_rows(
                &ids,
                ProjectionRequest::from_columns(["i"], dataset.schema()),
            )
            .await
            .unwrap();
        let physical: Vec<_> = expected
            .into_iter()
            .flatten()
            .map(|address| address.fragment_id() as i32 * 10 + address.row_offset() as i32)
            .collect();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            physical
        );
    }

    #[rstest]
    #[case::between_representations(false)]
    #[case::admission_boundary(true)]
    #[tokio::test]
    async fn fragment_working_set_serves_batches_below_complete_index_capacity(
        #[case] admission_boundary: bool,
    ) {
        const ROWS: usize = 1024;
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset_with_params(
                FragmentCount::from(4),
                FragmentRowCount::from(ROWS as u32),
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: ROWS,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        dataset
            .update_config([
                (super::super::SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                (super::super::INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
            ])
            .await
            .unwrap();
        let mut fragments = dataset.manifest.fragments.as_ref().clone();
        for (number, fragment) in fragments.iter_mut().enumerate() {
            let lineage = RowLineage {
                row_ids: RowIdSequence::try_from_iter(
                    (0..ROWS).map(|offset| (offset * 4 + number) as u64),
                )
                .unwrap(),
                created_at: RowDatasetVersionSequence::from_uniform_row_count(ROWS as u64, 1),
                last_updated_at: RowDatasetVersionSequence::from_uniform_row_count(ROWS as u64, 1),
            };
            place_row_lineage(&dataset, &lineage)
                .await
                .unwrap()
                .apply(fragment);
        }
        let transaction = Transaction::new_from_version(
            dataset.manifest.version,
            Operation::Merge {
                fragments,
                schema: dataset.schema().clone(),
                preserves_nullability: true,
            },
        );
        dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await
            .unwrap();
        let ids = [7, 17, 31, 1000];
        let expected = ids.map(|id| {
            Some(RowAddress::new_from_parts(
                dataset.manifest.fragments[id as usize % 4].id as u32,
                id as u32 / 4,
            ))
        });
        let whole = cold_dataset(dataset.clone(), 1 << 20, 1 << 20);
        let complete = get_row_id_index(&whole).await.unwrap().unwrap();
        let full_bytes = whole.index_cache.size_bytes().await;
        assert!(full_bytes > 0);
        drop(complete);
        let fragments = cold_dataset(dataset.clone(), 1 << 20, 1 << 20);
        assert_eq!(
            resolve_row_ids(&fragments, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            expected
        );
        let fragment_bytes = fragments.index_cache.size_bytes().await;
        // Sorted per-fragment IDs have contiguous physical offsets. Their
        // dense union needs interleaved addresses, so its representation grows.
        assert!(
            fragment_bytes < full_bytes,
            "fixture must have a larger complete view"
        );
        let budget = if admission_boundary {
            full_bytes - 1
        } else {
            (fragment_bytes + full_bytes) / 2
        };
        let dataset = cold_dataset(dataset, budget, 1 << 20);
        for _ in 0..4 {
            assert_eq!(
                resolve_row_ids(&dataset, &ids, true)
                    .await
                    .unwrap()
                    .unwrap(),
                expected
            );
        }
        let before = dataset.object_store.io_stats_snapshot();
        assert_eq!(
            resolve_row_ids(&dataset, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            expected
        );
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_bytes,
            before.read_bytes,
            "a complete view that cannot enter the cache must not cause repeated source reconstruction"
        );
    }

    #[rstest]
    #[case::inline(false)]
    #[case::spilled(true)]
    #[tokio::test]
    async fn repeated_overlapping_batch_keeps_fragment_only_working_set(#[case] spilled: bool) {
        let dataset = cold_dataset(shuffled_dataset(spilled, true).await, 1 << 20, 1 << 20);
        let ids = [7, 17, 31, 7, u64::MAX];
        let expected = ids.map(|id| {
            (id < 40).then(|| {
                let physical = id * 23 % 40;
                RowAddress::new_from_parts(
                    dataset.manifest.fragments[physical as usize / 10].id as u32,
                    physical as u32 % 10,
                )
            })
        });
        for _ in 0..2 {
            assert_eq!(
                resolve_row_ids(&dataset, &ids, true)
                    .await
                    .unwrap()
                    .unwrap(),
                expected
            );
        }
        assert!(
            dataset
                .index_cache
                .get_with_key(&crate::session::index_caches::RowIdIndexKey {
                    version: dataset.manifest.version,
                    with_deletions: true,
                    e_tag: dataset.manifest_location.e_tag.as_deref(),
                })
                .await
                .is_none()
        );
        let before = dataset.object_store.io_stats_snapshot();
        let results =
            futures::future::join_all((0..4).map(|_| resolve_row_ids(&dataset, &ids, true))).await;
        for result in results {
            assert_eq!(result.unwrap().unwrap(), expected);
        }
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_bytes,
            before.read_bytes
        );
    }

    #[rstest]
    #[case::inline(false)]
    #[case::spilled(true)]
    #[tokio::test]
    async fn cold_same_fragment_batch_prepares_only_its_candidate(#[case] spilled: bool) {
        let dataset = cold_dataset(shuffled_dataset(spilled, false).await, 1 << 20, 1 << 20);
        let ids = [7, 4, 7, 9, u64::MAX, 2];
        let fragment_id = dataset.manifest.fragments[0].id as u32;
        let expected = [Some(9), Some(8), Some(9), Some(3), None, Some(4)]
            .map(|offset| offset.map(|offset| RowAddress::new_from_parts(fragment_id, offset)));
        assert_eq!(
            resolve_row_ids(&dataset, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            expected
        );
        for (position, fragment) in dataset.manifest.fragments.iter().enumerate() {
            let key = PreparedRowIdFragmentKey {
                source: sequence_identity(&dataset, fragment).unwrap(),
                deletion: None,
            };
            assert_eq!(
                dataset.index_cache.get_with_key(&key).await.is_some(),
                position == 0
            );
        }
        assert!(
            dataset
                .index_cache
                .get_with_key(&crate::session::index_caches::RowIdIndexKey {
                    version: dataset.manifest.version,
                    with_deletions: true,
                    e_tag: dataset.manifest_location.e_tag.as_deref(),
                })
                .await
                .is_none()
        );
        let before = dataset.object_store.io_stats_snapshot();
        assert_eq!(
            resolve_row_ids(&dataset, &[9, 4, 2], true)
                .await
                .unwrap()
                .unwrap(),
            vec![expected[3], expected[1], expected[5]]
        );
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_bytes,
            before.read_bytes
        );
    }

    #[tokio::test]
    async fn bounds_survive_graph_eviction_without_retaining_prepared_state() {
        let dataset = cold_dataset(shuffled_dataset(true, false).await, 1 << 20, 1 << 20);
        let expected = RowAddress::new_from_parts(dataset.manifest.fragments[0].id as u32, 9);
        assert_eq!(
            resolve_row_ids(&dataset, &[7], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(expected)]
        );
        let key = PreparedRowIdFragmentKey {
            source: sequence_identity(&dataset, &dataset.manifest.fragments[0]).unwrap(),
            deletion: None,
        };
        let cached = dataset.index_cache.get_with_key(&key).await.unwrap();
        let view = Arc::downgrade(cached.index.as_ref().unwrap());
        assert!(view.upgrade().is_some());
        drop(cached);
        let mut spans = Vec::new();
        for fragment in dataset.manifest.fragments.iter() {
            let span_key = RowIdFragmentSpanKey {
                source: sequence_identity(&dataset, fragment).unwrap(),
            };
            spans.push((
                span_key,
                dataset
                    .metadata_cache
                    .get_with_key(&span_key)
                    .await
                    .unwrap(),
            ));
        }
        // Measure the next fragment with no cached file footers, then retain
        // only bounds. Small spilled arrays can otherwise hide in footers.
        let uncached = cold_dataset(dataset.clone(), 0, 0);
        let before = dataset.object_store.io_stats_snapshot();
        read_row_id_sequence(&uncached, &dataset.manifest.fragments[3])
            .await
            .unwrap();
        let source_bytes = dataset.object_store.io_stats_snapshot().read_bytes - before.read_bytes;
        assert!(source_bytes > 0);
        dataset.metadata_cache.clear().await;
        for (key, span) in spans {
            dataset.metadata_cache.insert_with_key(&key, span).await;
        }
        dataset.index_cache.clear().await;
        assert!(view.upgrade().is_none());
        let before = dataset.object_store.io_stats_snapshot();
        // ID31 is in another fragment; immutable bounds rule out fragment0
        // even after its prepared graph has been reclaimed.
        let result = resolve_row_ids(&dataset, &[31, 34, 31], true)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            result,
            [7, 8, 7].map(|offset| Some(RowAddress::new_from_parts(
                dataset.manifest.fragments[3].id as u32,
                offset
            )))
        );
        assert!(dataset.index_cache.get_with_key(&key).await.is_none());
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_bytes - before.read_bytes,
            source_bytes
        );
    }

    #[tokio::test]
    async fn resident_candidates_avoid_additional_reads_under_pressure() {
        let dataset = shuffled_dataset(true, true).await;
        let fragment = &dataset.manifest.fragments[3];
        let sequence = Arc::new(read_row_id_sequence(&dataset, fragment).await.unwrap());
        let span = sequence.row_id_range();
        let source = load_source(&dataset, fragment, sequence, true)
            .await
            .unwrap();
        let index = RowIdIndex::new(&[source]).unwrap();
        let index = Arc::new(index);
        let sizing = PreparedFragment {
            index: Some(index.clone()),
            span: span.clone(),
            deletion: None,
            source_rows: 10,
            is_range_only: false,
            retained: None,
        };
        let weight = super::super::cache::entry_bytes(&sizing);
        drop(sizing);
        // QuickCache also limits entry admission to about 97% of a shard.
        // This budget admits either unit while still excluding both together.
        let capacity = weight * 3 / 2;
        let dataset = cold_dataset(dataset, capacity, 0);
        let expected_7 = scan_address(&dataset, 1).await;
        let expected_31 = scan_address(&dataset, 33).await;
        // Measure the three source misses independently of the resolver. The
        // zero-capacity metadata cache keeps this the same physical read path.
        let before = dataset.object_store.io_stats_snapshot();
        for fragment in &dataset.manifest.fragments[..3] {
            read_row_id_sequence(&dataset, fragment).await.unwrap();
        }
        let source_bytes = dataset.object_store.io_stats_snapshot().read_bytes - before.read_bytes;
        assert!(source_bytes > 0);
        let fragment = &dataset.manifest.fragments[3];
        let key = PreparedRowIdFragmentKey {
            source: sequence_identity(&dataset, fragment).unwrap(),
            deletion: deletion_identity(&dataset, fragment, true).unwrap(),
        };
        let prepared = PreparedFragment {
            index: Some(index),
            span,
            deletion: key.deletion,
            source_rows: 10,
            is_range_only: false,
            retained: None,
        };
        dataset
            .index_cache
            .insert_with_key(&key, Arc::new(prepared))
            .await;
        assert!(dataset.index_cache.get_with_key(&key).await.is_some());
        let before = dataset.object_store.io_stats_snapshot();
        assert_eq!(
            resolve_row_ids(&dataset, &[31, 7, 31], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(expected_31), Some(expected_7), Some(expected_31)]
        );
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_bytes - before.read_bytes,
            source_bytes
        );
        assert!(dataset.index_cache.size_bytes().await <= capacity);
        let occupancy = dataset
            .index_cache
            .diagnostics_by_type(lance_core::cache::CacheSnapshotMode::Approximate)
            .await
            .by_type
            .unwrap()
            .occupancy
            .unwrap();
        let graph_entries = occupancy
            .types
            .iter()
            .find(|entry| entry.type_name == PreparedRowIdFragmentKey::stable_type_id())
            .map_or(0, |entry| entry.num_entries);
        assert!(graph_entries < dataset.manifest.fragments.len() as u64);
        // A later request must rebuild only what is absent, while preserving
        // answers even if this budget cannot retain both fragment lookups.
        assert_eq!(
            resolve_row_ids(&dataset, &[7], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(expected_7)]
        );
        assert!(dataset.index_cache.size_bytes().await <= capacity);
        let occupancy = dataset
            .index_cache
            .diagnostics_by_type(lance_core::cache::CacheSnapshotMode::Approximate)
            .await
            .by_type
            .unwrap()
            .occupancy
            .unwrap();
        let graph_entries = occupancy
            .types
            .iter()
            .find(|entry| entry.type_name == PreparedRowIdFragmentKey::stable_type_id())
            .map_or(0, |entry| entry.num_entries);
        assert!(graph_entries < dataset.manifest.fragments.len() as u64);
    }

    #[rstest]
    #[case::duplicate_live(false)]
    #[case::deleted_old_copy(true)]
    #[tokio::test]
    async fn all_fragment_candidates_validate_live_copies(#[case] deleted_old: bool) {
        let mut dataset = super::super::cache::tests::dataset(false).await;
        let expected = scan_address(&dataset, 17).await;
        if deleted_old {
            dataset.delete("i = 7").await.unwrap();
        }
        let mut fragments = dataset.manifest.fragments.as_ref().clone();
        let lineage = RowLineage {
            row_ids: RowIdSequence::from(0..10),
            created_at: RowDatasetVersionSequence::from_uniform_row_count(10, 1),
            last_updated_at: RowDatasetVersionSequence::from_uniform_row_count(10, 1),
        };
        place_row_lineage(&dataset, &lineage)
            .await
            .unwrap()
            .apply(&mut fragments[1]);
        let transaction = Transaction::new_from_version(
            dataset.version().version,
            Operation::Merge {
                fragments,
                schema: dataset.schema().clone(),
                preserves_nullability: true,
            },
        );
        dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await
            .unwrap();
        dataset = cold_dataset(dataset, 1 << 20, 1 << 20);
        let result = resolve_row_ids(&dataset, &[7, 17, 7, u64::MAX], true).await;
        if deleted_old {
            assert_eq!(
                result.unwrap().unwrap(),
                vec![Some(expected), None, Some(expected), None]
            );
        } else {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::Internal { .. }));
            assert!(error.to_string().contains("live in multiple fragments"));
        }
    }

    #[tokio::test]
    async fn fragment_answers_follow_deletions_and_old_snapshots() {
        let old = cold_dataset(
            super::super::cache::tests::dataset(true).await,
            1 << 20,
            1 << 20,
        );
        let expected_7 = scan_address(&old, 7).await;
        let expected_31 = scan_address(&old, 31).await;
        assert_eq!(
            resolve_row_ids(&old, &[7], true).await.unwrap().unwrap(),
            vec![Some(expected_7)]
        );
        // Write through another session so the old reader's resident fragment
        // states survive. Rebind the new snapshot to that shared read session.
        let mut current = cold_dataset(old.clone(), 1 << 20, 1 << 20);
        current.delete("i = 7").await.unwrap();
        current.index_cache = old.index_cache.clone();
        current.metadata_cache = old.metadata_cache.clone();
        current.session = old.session();
        assert!(7 < current.manifest.next_row_id);
        assert_eq!(
            resolve_row_ids(&current, &[7, 31, 7, u64::MAX], true)
                .await
                .unwrap()
                .unwrap(),
            vec![None, Some(expected_31), None, None]
        );
        let before = current.object_store.io_stats_snapshot();
        assert_eq!(
            resolve_row_ids(&current, &[31], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(expected_31)]
        );
        let after_distinct = current.object_store.io_stats_snapshot();
        assert!(after_distinct.read_bytes >= before.read_bytes);
        assert_eq!(
            resolve_row_ids(&current, &[31], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(expected_31)]
        );
        assert_eq!(
            resolve_row_ids(&old, &[7], true).await.unwrap().unwrap(),
            vec![Some(expected_7)]
        );
        assert_eq!(
            current.object_store.io_stats_snapshot().read_bytes,
            after_distinct.read_bytes
        );
        let batch = current
            .take_rows(
                &[7, 31, 7],
                ProjectionRequest::from_columns(["i"], current.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            &[31]
        );
    }

    #[tokio::test]
    async fn concurrent_distinct_requests_share_preparation_and_release_evicted_graphs() {
        let dataset = cold_dataset(shuffled_dataset(true, true).await, 1 << 20, 1 << 20);
        let expected_7 = scan_address(&dataset, 1).await;
        let expected_31 = scan_address(&dataset, 33).await;
        let before = dataset.object_store.io_stats_snapshot();
        for fragment in dataset.manifest.fragments.iter() {
            read_row_id_sequence(&dataset, fragment).await.unwrap();
        }
        let source_bytes = dataset.object_store.io_stats_snapshot().read_bytes - before.read_bytes;
        let before = dataset.object_store.io_stats_snapshot();
        let (first, other) = tokio::join!(
            resolve_row_ids(&dataset, &[7], true),
            resolve_row_ids(&dataset, &[31], true)
        );
        assert_eq!(first.unwrap().unwrap(), vec![Some(expected_7)]);
        assert_eq!(other.unwrap().unwrap(), vec![Some(expected_31)]);
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_bytes - before.read_bytes,
            source_bytes
        );
        let fragment = dataset
            .manifest
            .fragments
            .iter()
            .find(|fragment| fragment.id == u64::from(expected_7.fragment_id()))
            .unwrap();
        let key = PreparedRowIdFragmentKey {
            source: sequence_identity(&dataset, fragment).unwrap(),
            deletion: deletion_identity(&dataset, fragment, true).unwrap(),
        };
        let cached = dataset.index_cache.get_with_key(&key).await.unwrap();
        let held = cached.clone();
        let weak = Arc::downgrade(held.index.as_ref().unwrap());
        let complete = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert_eq!(
            complete.get_many(&[7, 31]).unwrap(),
            vec![Some(expected_7), Some(expected_31)]
        );
        assert_eq!(
            held.index.as_ref().unwrap().get(7).unwrap(),
            Some(expected_7)
        );
        dataset.index_cache.clear().await;
        drop(cached);
        assert!(
            weak.upgrade().is_some(),
            "active readers survive cache eviction"
        );
        drop(held);
        assert!(
            weak.upgrade().is_none(),
            "the preparation registry must not retain an evicted graph"
        );
        assert!(dataset.index_cache.size_bytes().await <= 1 << 20);
    }

    #[tokio::test]
    async fn resident_queries_do_not_wait_for_unrelated_builds() {
        let dataset = cold_dataset(shuffled_dataset(true, false).await, 1 << 20, 1 << 20);
        let expected = RowAddress::new_from_parts(dataset.manifest.fragments[0].id as u32, 9);
        assert_eq!(
            resolve_row_ids(&dataset, &[7], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(expected)]
        );
        let builds = &dataset.session.row_id_index_builds;
        let fragment = &dataset.manifest.fragments[3];
        let key = PreparedRowIdFragmentKey {
            source: sequence_identity(&dataset, fragment).unwrap(),
            deletion: deletion_identity(&dataset, fragment, true).unwrap(),
        };
        let cold = builds.for_fragment(&dataset.uri, &key).unwrap();
        let _cold_guard = cold.lock().await;
        let pending = resolve_row_ids(&dataset, &[31], true);
        tokio::pin!(pending);
        tokio::select! {
            result = &mut pending => panic!("cold query unexpectedly completed: {result:?}"),
            _ = tokio::task::yield_now() => {}
        }
        let before = dataset.object_store.io_stats_snapshot();
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            resolve_row_ids(&dataset, &[7], true),
        )
        .await
        .expect("resident query must not wait for unrelated builds")
        .unwrap()
        .unwrap();
        assert_eq!(result, vec![Some(expected)]);
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_bytes,
            before.read_bytes
        );
    }

    #[tokio::test]
    async fn oversized_preparation_is_shared_only_by_active_callers() {
        let dataset = cold_dataset(shuffled_dataset(true, false).await, 0, 0);
        let fragment = &dataset.manifest.fragments[0];
        let key = PreparedRowIdFragmentKey {
            source: sequence_identity(&dataset, fragment).unwrap(),
            deletion: deletion_identity(&dataset, fragment, true).unwrap(),
        };
        let held = prepare_shared(&dataset, fragment, &key, &[7])
            .await
            .unwrap();
        let before = dataset.object_store.io_stats_snapshot();
        let other = prepare_shared(&dataset, fragment, &key, &[7])
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&held, &other));
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_bytes,
            before.read_bytes
        );
        let weak = Arc::downgrade(&held);
        drop(other);
        drop(held);
        assert!(weak.upgrade().is_none());
        assert_eq!(dataset.index_cache.size_bytes().await, 0);
    }

    #[tokio::test]
    async fn concurrent_deletion_modes_preserve_their_public_contracts() {
        let mut dataset = super::super::cache::tests::dataset(true).await;
        let expected = scan_address(&dataset, 7).await;
        dataset.delete("i = 7").await.unwrap();
        let dataset = cold_dataset(dataset, 1 << 20, 1 << 20);
        let (live, physical) = tokio::join!(
            resolve_row_ids(&dataset, &[7], true),
            get_row_id_index_with(&dataset, false)
        );
        assert_eq!(live.unwrap().unwrap(), vec![None]);
        assert_eq!(physical.unwrap().unwrap().get(7).unwrap(), Some(expected));
        assert_eq!(
            resolve_row_ids(&dataset, &[7], true)
                .await
                .unwrap()
                .unwrap(),
            vec![None]
        );
    }

    #[tokio::test]
    async fn allocation_bound_rejects_only_out_of_range_requests_without_io() {
        let dataset = super::super::cache::tests::dataset(true).await;
        let bound = dataset.manifest.next_row_id;
        let before = dataset.object_store.io_stats_snapshot();
        for ids in [vec![bound, bound], vec![bound, u64::MAX], vec![]] {
            assert_eq!(
                resolve_row_ids(&dataset, &ids, true)
                    .await
                    .unwrap()
                    .unwrap(),
                vec![None; ids.len()]
            );
        }
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_iops,
            before.read_iops
        );
        let expected = scan_address(&dataset, 7).await;
        assert_eq!(
            resolve_row_ids(&dataset, &[bound, 7, bound], true)
                .await
                .unwrap()
                .unwrap(),
            vec![None, Some(expected), None]
        );
    }

    #[rstest]
    #[case::segments(false)]
    #[case::single_array(true)]
    #[tokio::test]
    async fn deleted_segment_copy_is_filtered_before_duplicate_validation(
        #[case] single_array: bool,
    ) {
        let mut dataset = super::super::cache::tests::dataset(false).await;
        let expected = scan_address(&dataset, 7).await;
        dataset.delete("i < 5").await.unwrap();
        get_row_id_index(&dataset).await.unwrap().unwrap();
        // Avoid compaction: its fragment order need not match payload order.
        // Keep two explicit inline segments in the original ten-row fragment.
        let mut fragments = dataset.manifest.fragments.as_ref().clone();
        assert_eq!(fragments[0].physical_rows, Some(10));
        assert_eq!(
            expected,
            RowAddress::new_from_parts(fragments[0].id as u32, 7)
        );
        assert!(fragments[0].deletion_file.is_some());
        let mut sequence = RowIdSequence::from(0..5);
        sequence.extend(RowIdSequence::from(0..5));
        if single_array {
            sequence = RowIdSequence::from(&[0, 1, 2, 3, 4, 0, 1, 2, 3, 4][..]);
            assert!(matches!(
                sequence.segments(),
                [lance_table::rowids::segment::U64Segment::Array(_)]
            ));
        }
        let lineage = RowLineage {
            row_ids: sequence,
            created_at: RowDatasetVersionSequence::from_uniform_row_count(10, 1),
            last_updated_at: RowDatasetVersionSequence::from_uniform_row_count(10, 1),
        };
        place_row_lineage(&dataset, &lineage)
            .await
            .unwrap()
            .apply(&mut fragments[0]);
        let transaction = Transaction::new_from_version(
            dataset.version().version,
            Operation::Merge {
                fragments,
                schema: dataset.schema().clone(),
                preserves_nullability: true,
            },
        );
        dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await
            .unwrap();
        let batch = dataset
            .take_rows(
                &[2, 2],
                ProjectionRequest::from_columns(["i"], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            &[7, 7]
        );
    }

    #[rstest]
    #[case::inline(false)]
    #[case::spilled(true)]
    #[tokio::test]
    async fn shuffled_fragments_reuse_unchanged_sources_after_append_and_delete(
        #[case] spilled: bool,
    ) {
        let mut current = cold_dataset(shuffled_dataset(spilled, true).await, 1 << 20, 1 << 20);
        let old = current.clone();
        let ids = [9, 23, 9, u64::MAX];
        let expected = [
            Some(RowAddress::new_from_parts(0, 7)),
            Some(RowAddress::new_from_parts(0, 9)),
            Some(RowAddress::new_from_parts(0, 7)),
            None,
        ];
        assert_eq!(
            resolve_row_ids(&old, &ids, true).await.unwrap().unwrap(),
            expected
        );
        let batch = arrow_array::record_batch!(("i", Int32, [40, 41])).unwrap();
        current
            .append(
                arrow_array::RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                None,
            )
            .await
            .unwrap();
        let before = current.object_store.io_stats_snapshot();
        assert_eq!(
            resolve_row_ids(&current, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            expected
        );
        assert_eq!(
            current.object_store.io_stats_snapshot().read_bytes,
            before.read_bytes,
            "append must preserve prepared shuffled fragment lookups"
        );
        let batch = current
            .take_rows(
                &[9, 23, 9, 40],
                ProjectionRequest::from_columns(["i"], current.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            &[7, 9, 7, 40]
        );
        current.delete("i = 7").await.unwrap();
        let changed = current
            .manifest
            .fragments
            .iter()
            .find(|f| f.deletion_file.is_some())
            .unwrap();
        let before = current.object_store.io_stats_snapshot();
        drop(
            load_source(&current, changed, Arc::new(RowIdSequence::default()), true)
                .await
                .unwrap(),
        );
        let deletion_bytes =
            current.object_store.io_stats_snapshot().read_bytes - before.read_bytes;
        let before = current.object_store.io_stats_snapshot();
        let deleted = [None, expected[1], None, None];
        assert_eq!(
            resolve_row_ids(&current, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            deleted
        );
        assert_eq!(
            current.object_store.io_stats_snapshot().read_bytes - before.read_bytes,
            deletion_bytes,
            "a deletion must read only visibility, never the unchanged row-ID source"
        );
        let before = current.object_store.io_stats_snapshot();
        for _ in 0..3 {
            assert_eq!(
                resolve_row_ids(&old, &ids, true).await.unwrap().unwrap(),
                expected
            );
            assert_eq!(
                resolve_row_ids(&current, &ids, true)
                    .await
                    .unwrap()
                    .unwrap(),
                deleted
            );
        }
        assert_eq!(
            current.object_store.io_stats_snapshot().read_bytes,
            before.read_bytes
        );
    }
}
