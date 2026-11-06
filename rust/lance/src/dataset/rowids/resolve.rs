// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Resolve requested IDs while preserving complete-index lookup semantics.

use std::ops::RangeInclusive;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::spawn_cpu;
use lance_table::format::{Fragment, RowIdMeta};
use lance_table::rowids::RowIdIndex;

use crate::session::index_caches::{PreparedRowIdFragmentKey, RowIdFragmentSpanKey, RowIdIndexKey};
use crate::{Dataset, Error, Result};

use super::cache::{CachedFragment, deletion_identity, load_source, sequence_identity};
use super::{get_row_id_index_with, overlapping_id_spans, read_row_id_sequence};

/// Small encoded inline sources need no CPU-pool dispatch to prepare a lookup.
const INLINE_PREPARE_BYTES: usize = 16 * 1024;
/// Bound decoded source rows while allowing small fragments to retain remote
/// I/O concurrency. This schedules work; it is not an RSS limit.
const PREPARE_ROW_BUDGET: usize = 128 * 1024 * 1024;
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
    let Some(&row_id) = row_ids.first() else {
        return Ok(Some(Vec::new()));
    };
    if with_deletions && row_ids.iter().all(|id| *id >= dataset.manifest.next_row_id) {
        return Ok(Some(vec![None; row_ids.len()]));
    }
    // Bulk requests keep the core's global routing/cost model. When supported,
    // the complete loader composes these same prepared fragment allocations.
    if !with_deletions
        || dataset.manifest_location.e_tag.is_none()
        || row_ids.iter().any(|id| *id != row_id)
    {
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

    let mut found = None;
    let mut misses = Vec::with_capacity(dataset.manifest.fragments.len());
    let mut cached_fragments = 0;
    let mut excluded_fragments = 0;
    // Consume every hit before admitting misses: a scan under cache pressure
    // must not evict a useful resident entry before this request has queried it.
    for fragment in dataset.manifest.fragments.iter() {
        let key = PreparedRowIdFragmentKey {
            source: sequence_identity(dataset, fragment)?,
            deletion: deletion_identity(dataset, fragment, true)?,
        };
        if let Some(cached) = dataset.index_cache.get_with_key(&key).await {
            if cached
                .span
                .as_ref()
                .is_none_or(|span| !span.contains(&row_id))
            {
                excluded_fragments += 1;
                continue;
            }
            if let Some(index) = &cached.index {
                record_match(row_id, &mut found, index.get(row_id)?)?;
                cached_fragments += 1;
                continue;
            }
        }
        let span_key = RowIdFragmentSpanKey { source: key.source };
        if let Some(span) = dataset.metadata_cache.get_with_key(&span_key).await
            && span.0.as_ref().is_none_or(|span| !span.contains(&row_id))
        {
            excluded_fragments += 1;
            continue;
        }
        misses.push((fragment, key));
    }
    let parallelism = preparation_parallelism(
        dataset,
        misses.iter().map(|(fragment, _)| fragment.physical_rows),
    );
    let mut loads = Vec::with_capacity(misses.len());
    for (fragment, key) in misses {
        loads.push(async move {
            let prepared = prepare_fragment(dataset, fragment, &key, row_id).await?;
            let address = prepared
                .index
                .as_ref()
                .map(|index| index.get(row_id))
                .transpose()?
                .flatten();
            Ok::<_, Error>((key, prepared, address))
        });
    }
    let mut loads = futures::stream::iter(loads).buffer_unordered(parallelism);
    let mut loaded_fragments = 0;
    let mut prepared_fragments = 0;
    while let Some((key, prepared, address)) = loads.try_next().await? {
        loaded_fragments += 1;
        record_match(row_id, &mut found, address)?;
        if prepared.index.is_some() {
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
        requested_ids = row_ids.len(),
        "resolved row ids from prepared fragments"
    );
    Ok(Some(vec![found; row_ids.len()]))
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
    row_id: u64,
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
    if span.as_ref().is_none_or(|span| !span.contains(&row_id)) {
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

fn record_match(
    row_id: u64,
    found: &mut Option<RowAddress>,
    address: Option<RowAddress>,
) -> Result<()> {
    if let Some(address) = address
        && found.replace(address).is_some()
    {
        return Err(Error::internal(format!(
            "row id index corrupt: stable row id {row_id} is live in multiple fragments"
        )));
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
    use crate::dataset::ProjectionRequest;
    use crate::dataset::rowids::{RowLineage, get_row_id_index, place_row_lineage};
    use crate::dataset::write::CommitBuilder;
    use crate::session::Session;

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
        let result = resolve_row_ids(&dataset, &[31], true)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            result,
            vec![Some(RowAddress::new_from_parts(
                dataset.manifest.fragments[3].id as u32,
                7
            ))]
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
            resolve_row_ids(&dataset, &[31], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(expected_31)]
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
        let result = resolve_row_ids(&dataset, &[7, 7], true).await;
        if deleted_old {
            assert_eq!(result.unwrap().unwrap(), vec![Some(expected); 2]);
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
            resolve_row_ids(&current, &[7, 7], true)
                .await
                .unwrap()
                .unwrap(),
            vec![None, None]
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
