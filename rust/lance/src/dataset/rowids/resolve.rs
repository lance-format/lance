// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Resolve requested IDs while preserving complete-index lookup semantics.

use std::ops::{Range, RangeInclusive};
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::spawn_cpu;
use lance_table::format::{Fragment, RowIdMeta};
use lance_table::rowids::RowIdIndex;

use crate::session::index_caches::{PreparedRowIdFragmentKey, RowIdFragmentSpanKey, RowIdIndexKey};
use crate::{Dataset, Error, Result};

use super::cache::{
    CachedFragment, RowIdIndexSize, RowIdIndexSizeKey, deletion_identity, load_source,
    sequence_identity,
};
use super::{get_row_id_index_with, overlapping_id_spans, read_row_id_sequence};

/// Small encoded inline sources need no CPU-pool dispatch to prepare a lookup.
const INLINE_PREPARE_BYTES: usize = 16 * 1024;
/// Bound decoded source rows while allowing small fragments to retain remote
/// I/O concurrency. This schedules work; it is not an RSS limit.
const PREPARE_ROW_BUDGET: usize = 192 * 1024 * 1024;
/// Larger routing directories use the CPU pool when combining prepared units.
const INLINE_COMBINE_FRAGMENTS: usize = 64;

/// A cached empty span proves the immutable source has no IDs. The metadata
/// entry holds no sequence, deletion vector or prepared lookup allocation.
#[derive(Debug)]
pub struct FragmentSpan(Option<RangeInclusive<u64>>);

impl DeepSizeOf for FragmentSpan {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

#[derive(Debug)]
pub struct PreparedFragment {
    /// Absent after its ownership moves into a complete cached view. Keeping
    /// only the span then prevents a second cache-owned copy of the graph.
    index: Option<Arc<RowIdIndex>>,
    span: Option<RangeInclusive<u64>>,
}

impl DeepSizeOf for PreparedFragment {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.index
            .as_ref()
            .map_or(0, |index| index.deep_size_of_children(context))
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
    let key = RowIdIndexKey {
        with_deletions: true,
    };
    if let Some(cached) = dataset.index_cache.get_with_key(&key).await
        && cached.matches(dataset)
    {
        return cached.index.get_many(row_ids).map(Some);
    }
    let lock = dataset
        .session()
        .row_id_index_builds
        .for_dataset(&dataset.uri)?;
    let guard = lock.lock().await;
    if let Some(cached) = dataset.index_cache.get_with_key(&key).await {
        if cached.matches(dataset) {
            return cached.index.get_many(row_ids).map(Some);
        }
        // An earlier complete view can reuse its prepared units across versions.
        drop(guard);
        return get_row_id_index_with(dataset, true)
            .await?
            .map(|index| index.get_many(row_ids))
            .transpose();
    }
    let other_key = RowIdIndexKey {
        with_deletions: false,
    };
    if dataset.index_cache.get_with_key(&other_key).await.is_some() {
        // Keep full-view and fragment ownership mutually exclusive. Ignore-
        // deletions callers retain the existing overlap/deletion semantics.
        drop(guard);
        return get_row_id_index_with(dataset, true)
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
    let mut resident_bytes = 0;
    let mut sizing = Context::default();
    // Consume every hit before admitting misses: a scan under cache pressure
    // must not evict a useful resident entry before this request has queried it.
    for fragment in dataset.manifest.fragments.iter() {
        let key = PreparedRowIdFragmentKey {
            source: sequence_identity(dataset, fragment)?,
            deletion: deletion_identity(dataset, fragment, true)?,
        };
        let mut range = 0..unique_ids.len();
        if let Some(cached) = dataset.index_cache.get_with_key(&key).await {
            range = candidate_range(&cached.span, &unique_ids);
            if range.is_empty() {
                excluded_fragments += 1;
                continue;
            }
            if let Some(index) = &cached.index {
                resident_bytes += std::mem::size_of::<PreparedFragment>()
                    + cached.as_ref().deep_size_of_children(&mut sizing);
                resident.push((index.clone(), range));
                cached_fragments += 1;
                continue;
            }
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
    // A batch that reaches every resident fragment should use the core's
    // global routing/cost model when that working set fits one cache entry.
    // Partial requests and pressured working sets keep demand resolution.
    // Cold sources are still inspected before their bounds can prune a batch.
    let complete_fits = if let Some(limit) = dataset.index_cache.max_entry_bytes() {
        resident_bytes as u64 <= limit
            && dataset
                .metadata_cache
                .get_with_key(&RowIdIndexSizeKey {
                    with_deletions: true,
                })
                .await
                .is_none_or(|size| !size.exceeds(dataset, limit))
    } else {
        false
    };
    if unique_ids.len() > 1
        && resident.len() > 1
        && misses.is_empty()
        && excluded_fragments == 0
        && complete_fits
    {
        tracing::debug!(
            event = "row_id_fragment_batch_promotion",
            candidate_fragments = resident.len(),
            resident_bytes,
            requested_ids = row_ids.len(),
            "use complete routing for a fully resident batch"
        );
        drop(resident);
        drop(guard);
        let cached = super::cache::get(dataset, true).await?;
        if let Some(limit) = dataset.index_cache.max_entry_bytes() {
            let size = RowIdIndexSize::new(&cached);
            if size.exceeds(dataset, limit) {
                dataset
                    .metadata_cache
                    .insert_with_key(
                        &RowIdIndexSizeKey {
                            with_deletions: true,
                        },
                        Arc::new(size),
                    )
                    .await;
            }
        }
        return cached.index.get_many(row_ids).map(Some);
    }
    // Consume every resident hit before admitting any source miss.
    for (index, range) in resident {
        record_matches(
            &unique_ids[range.clone()],
            &mut found[range.clone()],
            index.get_many(&unique_ids[range])?,
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
            let prepared = prepare_fragment(dataset, fragment, &key, &ids[range]).await?;
            // A newly read source can narrow the candidate slice further than
            // the metadata available when the request was first scheduled.
            let range = candidate_range(&prepared.span, ids);
            let addresses = prepared
                .index
                .as_ref()
                .map(|index| index.get_many(&ids[range.clone()]))
                .transpose()?
                .unwrap_or_default();
            Ok::<_, Error>((key, prepared, range, addresses))
        });
    }
    let mut loads = futures::stream::iter(loads).buffer_unordered(parallelism);
    let mut loaded_fragments = 0;
    let mut prepared_fragments = 0;
    while let Some((key, prepared, range, addresses)) = loads.try_next().await? {
        loaded_fragments += 1;
        if prepared.index.is_some() {
            record_matches(&unique_ids[range.clone()], &mut found[range], addresses)?;
            prepared_fragments += 1;
            dataset
                .index_cache
                .insert_with_key(&key, Arc::new(prepared))
                .await;
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
        "resolved row ids from prepared fragments"
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

async fn prepare_fragment(
    dataset: &Dataset,
    fragment: &Fragment,
    key: &PreparedRowIdFragmentKey,
    row_ids: &[u64],
) -> Result<PreparedFragment> {
    let sequence = Arc::new(read_row_id_sequence(dataset, fragment).await?);
    let span = sequence.row_id_range();
    dataset
        .metadata_cache
        .insert_with_key(
            &RowIdFragmentSpanKey { source: key.source },
            Arc::new(FragmentSpan(span.clone())),
        )
        .await;
    if candidate_range(&span, row_ids).is_empty() {
        return Ok(PreparedFragment { index: None, span });
    }
    let source = load_source(dataset, fragment, sequence, true).await?;
    let prepare = move || RowIdIndex::new(&[source]);
    let index = if matches!(&fragment.row_id_meta,
        Some(RowIdMeta::Inline(bytes)) if bytes.len() <= INLINE_PREPARE_BYTES)
    {
        prepare()?
    } else {
        spawn_cpu(prepare).await?
    };
    Ok(PreparedFragment {
        index: Some(Arc::new(index)),
        span,
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

/// Compose cached probe fragments and missing raw sources into one charged
/// view. Unsupported representations release cached units before reconstruction;
/// the complete view never retains fragment wrapper parents.
pub(super) struct FragmentPromotion {
    pub index: RowIdIndex,
    pub fragments: Vec<CachedFragment>,
    pub units: Vec<(PreparedRowIdFragmentKey, Arc<PreparedFragment>)>,
    pub loaded_fragments: usize,
}

pub(super) async fn promote_fragments(
    dataset: &Dataset,
    with_deletions: bool,
) -> Result<Option<FragmentPromotion>> {
    let mut candidates = Vec::with_capacity(dataset.manifest.fragments.len());
    let mut retire = Vec::with_capacity(dataset.manifest.fragments.len());
    let mut spans = Vec::with_capacity(dataset.manifest.fragments.len());
    let mut spans_known = true;
    for fragment in dataset.manifest.fragments.iter() {
        let source = sequence_identity(dataset, fragment)?;
        let deletion = deletion_identity(dataset, fragment, true)?;
        let live_key = PreparedRowIdFragmentKey { source, deletion };
        let live = dataset.index_cache.get_with_key(&live_key).await;
        let raw = if deletion.is_some() {
            let raw_key = PreparedRowIdFragmentKey {
                source,
                deletion: None,
            };
            let raw = dataset.index_cache.get_with_key(&raw_key).await;
            if let Some(raw) = &raw {
                retire.push((raw_key, raw.clone()));
            }
            raw
        } else {
            live.clone()
        };
        if let Some(live) = &live {
            retire.push((live_key, live.clone()));
        }
        let span = if let Some(cached) = live.as_ref().or(raw.as_ref()) {
            Some(cached.span.clone())
        } else {
            dataset
                .metadata_cache
                .get_with_key(&RowIdFragmentSpanKey { source })
                .await
                .map(|span| span.0.clone())
        };
        spans_known &= span.is_some();
        spans.push(span.flatten());
        candidates.push((source, deletion, live, raw));
    }
    // Ignore-deletions mode still filters overlapping sources. Unknown spans
    // cannot establish that policy; the normal full loader resolves them.
    let can_select = spans_known
        || with_deletions
        || dataset
            .manifest
            .fragments
            .iter()
            .all(|fragment| fragment.deletion_file.is_none());
    let promoted = if can_select {
        let needs_deletions = if with_deletions {
            vec![true; candidates.len()]
        } else {
            overlapping_id_spans(&spans)
        };
        let mut units = Vec::with_capacity(candidates.len());
        let mut retained = Vec::with_capacity(candidates.len());
        let mut missing = Vec::with_capacity(candidates.len());
        let mut fragments = Vec::with_capacity(candidates.len());
        for (i, (((source, deletion, live, raw), span), needs_deletions)) in candidates
            .into_iter()
            .zip(spans)
            .zip(needs_deletions)
            .enumerate()
        {
            let cached = if needs_deletions { live } else { raw };
            let deletion = if needs_deletions { deletion } else { None };
            if let Some(cached) = cached.filter(|cached| cached.index.is_some()) {
                units.push((PreparedRowIdFragmentKey { source, deletion }, cached));
                retained.push(dataset.manifest.fragments[i].id as u32);
            } else {
                missing.push((i, needs_deletions));
            }
            fragments.push(CachedFragment {
                identity: source,
                span,
                deletion_identity: deletion,
            });
        }
        let inputs: Vec<_> = units
            .iter()
            .filter_map(|(_, unit)| unit.index.clone())
            .collect();
        // Reject inseparable merged inputs before loading any misses. Prepare
        // new raw sources against a combined probe parent so a small native
        // append does not force rereading all previously prepared fragments.
        let combined = if inputs.is_empty() {
            None
        } else if inputs.len() > INLINE_COMBINE_FRAGMENTS {
            spawn_cpu(move || {
                let refs: Vec<_> = inputs.iter().map(Arc::as_ref).collect();
                RowIdIndex::try_combine(&refs)
            })
            .await?
        } else {
            let refs: Vec<_> = inputs.iter().map(Arc::as_ref).collect();
            RowIdIndex::try_combine(&refs)?
        };
        if let Some(combined) = combined {
            let loaded_fragments = missing.len();
            let parallelism = preparation_parallelism(
                dataset,
                missing
                    .iter()
                    .map(|(i, _)| dataset.manifest.fragments[*i].physical_rows),
            );
            let mut loads = Vec::with_capacity(missing.len());
            for (i, needs_deletions) in missing {
                loads.push(async move {
                    let fragment = &dataset.manifest.fragments[i];
                    let sequence = Arc::new(read_row_id_sequence(dataset, fragment).await?);
                    let span = sequence.row_id_range();
                    dataset
                        .metadata_cache
                        .insert_with_key(
                            &RowIdFragmentSpanKey {
                                source: sequence_identity(dataset, fragment)?,
                            },
                            Arc::new(FragmentSpan(span.clone())),
                        )
                        .await;
                    let source = load_source(dataset, fragment, sequence, needs_deletions).await?;
                    Ok::<_, Error>((i, span, source))
                });
            }
            let loaded: Vec<_> = futures::stream::iter(loads)
                .buffered(parallelism)
                .try_collect()
                .await?;
            let mut sources = Vec::with_capacity(loaded.len());
            for (i, span, source) in loaded {
                fragments[i].span = span;
                sources.push(source);
            }
            let index = if sources.is_empty() {
                Some(combined)
            } else {
                spawn_cpu(move || combined.try_reuse_fragments(&retained, &sources)).await?
            };
            if index.is_none() {
                tracing::debug!(
                    event = "row_id_fragment_promotion_fallback",
                    loaded_fragments,
                    "global cost model requires full reconstruction"
                );
            }
            index.map(|index| FragmentPromotion {
                index,
                fragments,
                units,
                loaded_fragments,
            })
        } else {
            None
        }
    } else {
        None
    };
    // LanceCache intentionally has no per-key remove. Metadata-only replacement
    // releases matching prepared values without clearing unrelated cache entries.
    for (key, cached) in retire {
        if cached.index.is_some() {
            dataset
                .index_cache
                .insert_with_key(
                    &key,
                    Arc::new(PreparedFragment {
                        index: None,
                        span: cached.span.clone(),
                    }),
                )
                .await;
        }
    }
    Ok(promoted)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, UInt64Type};
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

    #[rstest]
    #[case::inline(false)]
    #[case::spilled(true)]
    #[tokio::test]
    async fn distinct_ids_reuse_fragments_and_bulk_keeps_global_lookup(#[case] spilled: bool) {
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
                .get_with_key(&RowIdIndexKey {
                    with_deletions: true
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
                .get_with_key(&RowIdIndexKey {
                    with_deletions: true
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
        // A complete candidate working set may promote on this first repeat.
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
    async fn rejected_complete_view_keeps_repeated_batches_on_resident_fragments(
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
        get_row_id_index(&whole).await.unwrap().unwrap();
        let full_bytes = whole
            .index_cache
            .get_with_key(&RowIdIndexKey {
                with_deletions: true,
            })
            .await
            .unwrap()
            .as_ref()
            .deep_size_of();
        let fragments = cold_dataset(dataset.clone(), 1 << 20, 1 << 20);
        assert_eq!(
            resolve_row_ids(&fragments, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            expected
        );
        let mut fragment_bytes = 0;
        for fragment in fragments.manifest.fragments.iter() {
            let key = PreparedRowIdFragmentKey {
                source: sequence_identity(&fragments, fragment).unwrap(),
                deletion: None,
            };
            fragment_bytes += fragments
                .index_cache
                .get_with_key(&key)
                .await
                .unwrap()
                .as_ref()
                .deep_size_of();
        }
        // Sorted per-fragment IDs have contiguous physical offsets. Their
        // dense union needs interleaved addresses, so its representation grows.
        assert!(
            fragment_bytes < full_bytes,
            "fixture must have a larger complete view"
        );
        let budget = if admission_boundary {
            full_bytes
                + 2 * std::mem::size_of::<std::sync::atomic::AtomicUsize>()
                + std::mem::size_of::<lance_core::cache::InternalCacheKey>()
                - 1
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
    async fn repeated_overlapping_batch_promotes_a_resident_working_set(#[case] spilled: bool) {
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
                .get_with_key(&RowIdIndexKey {
                    with_deletions: true
                })
                .await
                .is_some()
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
                .get_with_key(&RowIdIndexKey {
                    with_deletions: true
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
        let weak = Arc::downgrade(cached.index.as_ref().unwrap());
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
        assert!(weak.upgrade().is_none());
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
        let prepared = PreparedFragment {
            index: Some(Arc::new(RowIdIndex::new(&[source]).unwrap())),
            span,
        };
        let weight =
            prepared.deep_size_of() + std::mem::size_of::<lance_core::cache::InternalCacheKey>();
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
        assert!(dataset.index_cache.size().await <= 1);
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
        assert!(dataset.index_cache.size().await <= 1);
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
    async fn concurrent_distinct_requests_and_complete_view_ownership() {
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
        let held = dataset
            .index_cache
            .get_with_key(&key)
            .await
            .unwrap()
            .index
            .clone()
            .unwrap();
        let weak = Arc::downgrade(&held);
        let complete = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert_eq!(
            complete.get_many(&[7, 31]).unwrap(),
            vec![Some(expected_7), Some(expected_31)]
        );
        assert_eq!(held.get(7).unwrap(), Some(expected_7));
        assert!(
            dataset
                .index_cache
                .get_with_key(&key)
                .await
                .unwrap()
                .index
                .is_none()
        );
        drop(held);
        assert!(
            weak.upgrade().is_none(),
            "the complete view must not retain fragment wrapper parents"
        );
        assert!(dataset.index_cache.size_bytes().await <= 1 << 20);
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

    #[tokio::test]
    async fn deleted_segment_copy_is_filtered_before_duplicate_validation() {
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
}
