// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Resolve the first point request without constructing every inverse mapping.

use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::DeletionVector;
use lance_core::utils::tokio::spawn_cpu;
use lance_table::format::RowIdMeta;
use lance_table::rowids::FragmentRowIdIndex;

use crate::session::index_caches::{PointRowIdResolutionKey, RowIdIndexKey, RowIdPointHistoryKey};
use crate::{Dataset, Error, Result};

use super::{cache::load_source, get_row_id_index_with, read_row_id_sequence};

/// Bound inline search work; larger encoded sources use the CPU pool.
const INLINE_POINT_LOOKUP_BYTES: usize = 16 * 1024;

#[derive(Debug)]
pub struct CachedPointResolution {
    address: Option<RowAddress>,
}

impl DeepSizeOf for CachedPointResolution {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

#[derive(Debug, DeepSizeOf)]
pub struct CachedPointHistory {
    version: u64,
    e_tag: String,
}

/// Preserve the complete-index contract for bulk and ignore-deletions callers.
/// A first live point can scan encoded sources directly; exact repeats cache
/// their answer. While the generation's history remains cached, another distinct
/// point promotes to the normal full map. Uncacheable sources stay uncached.
pub async fn resolve_row_ids(
    dataset: &Dataset,
    row_ids: &[u64],
    with_deletions: bool,
) -> Result<Option<Vec<Option<RowAddress>>>> {
    if !dataset.manifest.uses_stable_row_ids() {
        return Ok(None);
    }
    let Some(&row_id) = row_ids.first() else {
        return Ok(Some(Vec::new()));
    };
    if !with_deletions || row_ids.iter().any(|id| *id != row_id) {
        return get_row_id_index_with(dataset, with_deletions)
            .await?
            .map(|index| index.get_many(row_ids))
            .transpose();
    }
    // Stable-ID writers allocate below this exclusive high-water mark, which
    // validation preserves through clone/restore. It cannot route lower IDs.
    if row_id >= dataset.manifest.next_row_id {
        return Ok(Some(vec![None; row_ids.len()]));
    }
    let index_key = RowIdIndexKey {
        with_deletions: true,
    };
    let point_key =
        dataset
            .manifest_location
            .e_tag
            .as_deref()
            .map(|e_tag| PointRowIdResolutionKey {
                version: dataset.manifest.version,
                e_tag,
                row_id,
            });
    if let Some(key) = &point_key
        && let Some(cached) = dataset.index_cache.get_with_key(key).await
    {
        return Ok(Some(vec![cached.address; row_ids.len()]));
    }
    if let Some(cached) = dataset.index_cache.get_with_key(&index_key).await
        && cached.matches(dataset)
    {
        return cached.index.get_many(row_ids).map(Some);
    }

    let lock = dataset
        .session()
        .row_id_index_builds
        .for_dataset(&dataset.uri, true)?;
    let guard = lock.lock().await;
    if let Some(key) = &point_key
        && let Some(cached) = dataset.index_cache.get_with_key(key).await
    {
        return Ok(Some(vec![cached.address; row_ids.len()]));
    }
    let previous = dataset.index_cache.get_with_key(&index_key).await;
    let history = dataset
        .index_cache
        .get_with_key(&RowIdPointHistoryKey)
        .await;
    let has_prior_point = history.is_some_and(|history| {
        history.version == dataset.manifest.version
            && Some(history.e_tag.as_str()) == dataset.manifest_location.e_tag.as_deref()
    });
    if previous.is_some() || has_prior_point {
        // A previous whole map enables cross-version reuse; a second distinct
        // point amortizes construction. Its loader acquires this same lock.
        drop(guard);
        return get_row_id_index_with(dataset, true)
            .await?
            .map(|index| index.get_many(row_ids))
            .transpose();
    }

    let address = resolve_point(dataset, row_id).await?;
    if let Some(key) = point_key {
        dataset
            .index_cache
            .insert_with_key(&key, Arc::new(CachedPointResolution { address }))
            .await;
        dataset
            .index_cache
            .insert_with_key(
                &RowIdPointHistoryKey,
                Arc::new(CachedPointHistory {
                    version: key.version,
                    e_tag: key.e_tag.to_owned(),
                }),
            )
            .await;
    }
    Ok(Some(vec![address; row_ids.len()]))
}

async fn resolve_point(dataset: &Dataset, row_id: u64) -> Result<Option<RowAddress>> {
    let mut loads = Vec::with_capacity(dataset.manifest.fragments.len());
    for fragment in dataset.manifest.fragments.iter() {
        loads.push(async move {
            // Inline native segments resolve without object-store reads.
            // Spilled sources have no manifest ID bounds and remain candidates.
            let is_small_inline = matches!(&fragment.row_id_meta,
                Some(RowIdMeta::Inline(bytes)) if bytes.len() <= INLINE_POINT_LOOKUP_BYTES);
            let row_id_sequence = Arc::new(read_row_id_sequence(dataset, fragment).await?);
            let source = FragmentRowIdIndex {
                fragment_id: fragment.id as u32,
                row_id_sequence: row_id_sequence.clone(),
                deletion_vector: Arc::new(DeletionVector::default()),
            };
            let lookup = move || {
                source
                    .get_many(&[row_id])
                    .map(|mut addresses| addresses.pop().flatten())
            };
            let address = if is_small_inline {
                lookup()
            } else {
                spawn_cpu(lookup).await
            };
            if fragment.deletion_file.is_none() || matches!(address, Ok(None)) {
                return address;
            }
            // A raw miss needs no deletion file. Multiple raw segment matches
            // must retry with deletions: all but one copy may be deleted.
            let source = load_source(dataset, fragment, row_id_sequence, true).await?;
            match address {
                Ok(address) => Ok(address
                    .filter(|address| !source.deletion_vector.contains(address.row_offset()))),
                Err(_) => {
                    let lookup = move || {
                        source
                            .get_many(&[row_id])
                            .map(|mut addresses| addresses.pop().flatten())
                    };
                    if is_small_inline {
                        lookup()
                    } else {
                        spawn_cpu(lookup).await
                    }
                }
            }
        });
    }
    // A sparse request must not keep every large original sequence resident.
    // Four sources overlap remote reads while bounding live decoding buffers.
    let mut addresses =
        futures::stream::iter(loads).buffer_unordered(dataset.object_store.io_parallelism().min(4));
    let mut found = None;
    let mut loaded_fragments = 0;
    while let Some(address) = addresses.try_next().await? {
        loaded_fragments += 1;
        if let Some(address) = address
            && found.replace(address).is_some()
        {
            return Err(Error::internal(format!(
                "row id index corrupt: stable row id {row_id} is live in multiple fragments"
            )));
        }
    }
    tracing::debug!(
        event = "row_id_point_scan",
        row_id,
        candidate_fragments = dataset.manifest.fragments.len(),
        loaded_fragments,
        spilled_fragments = dataset
            .manifest
            .fragments
            .iter()
            .filter(|fragment| matches!(fragment.row_id_meta, Some(RowIdMeta::Column)))
            .count(),
        prepared_fragments = 0,
        "resolved point from encoded row ids"
    );
    Ok(found)
}

#[cfg(test)]
mod tests {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, UInt64Type};
    use lance_table::format::{
        ROW_CREATED_AT_VERSION_FIELD_ID, ROW_ID_FIELD_ID, ROW_LAST_UPDATED_AT_VERSION_FIELD_ID,
    };
    use lance_table::rowids::RowIdSequence;
    use lance_table::rowids::version::RowDatasetVersionSequence;
    use lance_table::transaction::{Operation, Transaction};
    use rstest::rstest;

    use super::*;
    use crate::dataset::ProjectionRequest;
    use crate::dataset::rowids::{
        RowLineage, SPILL_ROW_LINEAGE_CONFIG_KEY, get_row_id_index, place_row_lineage,
    };
    use crate::dataset::write::CommitBuilder;
    use crate::session::Session;

    fn cold_dataset(mut dataset: Dataset) -> Dataset {
        let session = Arc::new(Session::new(1 << 20, 0, dataset.session().store_registry()));
        dataset.index_cache = Arc::new(session.index_cache.for_dataset(&dataset.uri));
        dataset.metadata_cache = Arc::new(session.metadata_cache.for_dataset(&dataset.uri));
        dataset.session = session;
        dataset
    }

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

    #[tokio::test]
    async fn first_point_answer_is_not_a_complete_index() {
        let dataset = super::super::cache::tests::dataset(true).await;
        let expected_23 = scan_address(&dataset, 23).await;
        let expected_31 = scan_address(&dataset, 31).await;
        let key = RowIdIndexKey {
            with_deletions: true,
        };
        assert!(dataset.index_cache.get_with_key(&key).await.is_none());
        let before = dataset.object_store.io_stats_snapshot();
        let result = resolve_row_ids(&dataset, &[23, 23], true)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result, vec![Some(expected_23); 2]);
        assert!(dataset.object_store.io_stats_snapshot().read_iops > before.read_iops);
        assert!(dataset.index_cache.get_with_key(&key).await.is_none());

        let before = dataset.object_store.io_stats_snapshot();
        assert_eq!(
            resolve_row_ids(&dataset, &[23], true)
                .await
                .unwrap()
                .unwrap(),
            result[..1]
        );
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_iops,
            before.read_iops
        );
        let batch = dataset
            .take_rows(
                &[23],
                ProjectionRequest::from_columns(["i"], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            &[23]
        );

        // A new key promotes to the ordinary complete map; a point's negative
        // or positive answer must never be returned as the full index.
        assert_eq!(
            resolve_row_ids(&dataset, &[31], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(expected_31)]
        );
        assert!(dataset.index_cache.get_with_key(&key).await.is_some());
        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        for id in [0, 7, 19, 23, 31, 39] {
            assert!(index.get(id).unwrap().is_some());
        }
        assert_eq!(index.get(999).unwrap(), None);
    }

    #[tokio::test]
    async fn allocation_bound_rejects_only_out_of_range_points_without_io() {
        let dataset = super::super::cache::tests::dataset(true).await;
        let before = dataset.object_store.io_stats_snapshot();
        for id in [dataset.manifest.next_row_id, u64::MAX] {
            assert_eq!(
                resolve_row_ids(&dataset, &[id, id], true)
                    .await
                    .unwrap()
                    .unwrap(),
                vec![None; 2]
            );
        }
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_iops,
            before.read_iops
        );
        assert!(
            dataset
                .index_cache
                .get_with_key(&RowIdPointHistoryKey)
                .await
                .is_none()
        );
        assert!(
            resolve_row_ids(&dataset, &[7], true)
                .await
                .unwrap()
                .unwrap()[0]
                .is_some()
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
    }

    #[rstest]
    #[case::inline(false)]
    #[case::spilled(true)]
    #[tokio::test]
    async fn point_answers_follow_snapshot_deletions(#[case] spilled: bool) {
        let mut current = super::super::cache::tests::dataset(spilled).await;
        let old = current.clone();
        let (first, repeated) = tokio::join!(
            resolve_row_ids(&old, &[7, 7], true),
            resolve_row_ids(&old, &[7, 7], true),
        );
        assert_eq!(first.unwrap(), repeated.unwrap());
        current.delete("i = 7").await.unwrap();
        assert_eq!(
            resolve_row_ids(&current, &[7, 7], true)
                .await
                .unwrap()
                .unwrap(),
            vec![None; 2]
        );
        let before = current.object_store.io_stats_snapshot();
        assert_eq!(
            resolve_row_ids(&current, &[7], true)
                .await
                .unwrap()
                .unwrap(),
            vec![None]
        );
        assert_eq!(
            current.object_store.io_stats_snapshot().read_iops,
            before.read_iops
        );
        assert!(resolve_row_ids(&old, &[7], true).await.unwrap().unwrap()[0].is_some());
        assert_eq!(
            resolve_row_ids(&current, &[999], true)
                .await
                .unwrap()
                .unwrap(),
            vec![None]
        );
        assert!(
            resolve_row_ids(&current, &[], true)
                .await
                .unwrap()
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn deleted_segment_copy_is_filtered_before_duplicate_validation() {
        let mut dataset = super::super::cache::tests::dataset(true).await;
        dataset.delete("i < 10").await.unwrap();
        // Inline encoding preserves the two explicit segments. A spilled
        // sequence is re-encoded and can combine them into an unsorted array.
        dataset
            .update_config([(SPILL_ROW_LINEAGE_CONFIG_KEY, "false")])
            .await
            .unwrap();
        let mut fragments = dataset.manifest.fragments.as_ref().clone();
        // Keep user columns in their existing file, but replace every old
        // lineage reference before installing the new inline lineage.
        for file in &mut fragments[0].files {
            let (fields, columns): (Vec<_>, Vec<_>) = file
                .fields
                .iter()
                .zip(file.column_indices.iter())
                .filter(|(field, _)| {
                    !matches!(
                        **field,
                        ROW_ID_FIELD_ID
                            | ROW_CREATED_AT_VERSION_FIELD_ID
                            | ROW_LAST_UPDATED_AT_VERSION_FIELD_ID
                    )
                })
                .map(|(field, column)| (*field, *column))
                .unzip();
            file.fields = fields.into();
            file.column_indices = columns.into();
        }
        fragments[0].files.retain(|file| !file.fields.is_empty());
        let mut sequence = RowIdSequence::from(0..10);
        sequence.extend(RowIdSequence::from(0..10));
        let fragment_id = fragments[0].id as u32;
        let lineage = RowLineage {
            row_ids: sequence,
            created_at: RowDatasetVersionSequence::from_uniform_row_count(20, 1),
            last_updated_at: RowDatasetVersionSequence::from_uniform_row_count(20, 1),
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
        dataset = cold_dataset(dataset);
        assert!(
            dataset
                .index_cache
                .get_with_key(&RowIdIndexKey {
                    with_deletions: true
                })
                .await
                .is_none()
        );
        assert_eq!(
            resolve_row_ids(&dataset, &[7, 7], true)
                .await
                .unwrap()
                .unwrap(),
            vec![Some(RowAddress::new_from_parts(fragment_id, 17)); 2]
        );
        let batch = dataset
            .take_rows(
                &[7],
                ProjectionRequest::from_columns(["i"], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            &[17]
        );
    }

    #[rstest]
    #[case::duplicate_live(false)]
    #[case::deleted_old_copy(true)]
    #[tokio::test]
    async fn every_candidate_is_checked_for_live_copies(#[case] deleted_old: bool) {
        let mut dataset = super::super::cache::tests::dataset(false).await;
        if deleted_old {
            dataset.delete("i = 7").await.unwrap();
        }
        let mut fragments = dataset.manifest.fragments.as_ref().clone();
        let fragment_id = fragments[1].id as u32;
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
        dataset = cold_dataset(dataset);
        assert!(
            dataset
                .index_cache
                .get_with_key(&RowIdIndexKey {
                    with_deletions: true
                })
                .await
                .is_none()
        );
        let result = resolve_row_ids(&dataset, &[7, 7], true).await;
        if deleted_old {
            assert_eq!(
                result.unwrap().unwrap(),
                vec![Some(RowAddress::new_from_parts(fragment_id, 7)); 2]
            );
        } else {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::Internal { .. }));
            assert!(error.to_string().contains("live in multiple fragments"));
        }
    }
}
