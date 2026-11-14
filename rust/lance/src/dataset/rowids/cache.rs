// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Source identities and shared preparation for cache-owned fragment lookups.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use lance_core::cache::{CacheKey, CacheNamespace, InternalCacheKey, KeyBuilder};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::deletion::DeletionVector;
use lance_table::format::{Fragment, ROW_ID_FIELD_ID, RowIdMeta};
use lance_table::io::deletion::read_deletion_file;
use lance_table::rowids::{FragmentRowIdIndex, RowIdSequence};

use crate::session::caches::{DeletionFileKey, RowIdSequenceKey};
use crate::session::index_caches::RowIdFragmentDeletionKey;
use crate::{Dataset, Error, Result};

/// Share source preparation and active lookups, including lookups too large for
/// cache admission. Unlike a final cache entry, a source can yield just a span or
/// a lookup pruned for a particular snapshot. Only callers and caches own those
/// allocations; this registry holds weak references and prunes expired entries.
#[derive(Default)]
pub struct RowIdFragmentRegistry {
    fragments: Mutex<FragmentBuilds>,
}

type FragmentBuildKey = (String, InternalCacheKey, Option<InternalCacheKey>);
type FragmentBuild = tokio::sync::Mutex<Option<Arc<super::resolve::PreparedFragment>>>;
#[derive(Default)]
struct FragmentBuilds {
    entries: HashMap<FragmentBuildKey, FragmentBuildEntry>,
    requests_until_prune: usize,
}

#[derive(Default)]
struct FragmentBuildEntry {
    build: Weak<FragmentBuild>,
    active: Weak<super::resolve::PreparedFragment>,
}

impl RowIdFragmentRegistry {
    pub(super) fn for_fragment(
        &self,
        uri: &str,
        key: &crate::session::index_caches::PreparedRowIdFragmentKey,
    ) -> Result<Arc<FragmentBuild>> {
        let mut fragments = self.fragments.lock().map_err(|error| {
            Error::internal(format!("row id fragment build registry poisoned: {error}"))
        })?;
        if fragments.requests_until_prune == 0 {
            fragments.entries.retain(|_, entry| {
                entry.build.strong_count() > 0 || entry.active.strong_count() > 0
            });
            // Amortize cleanup across requests instead of scanning all resident
            // fragments for every miss. Retain scans capacity, so also release
            // a sparse high-water allocation after readers or datasets expire.
            let interval = fragments.entries.len().max(64);
            if fragments.entries.capacity() > interval.saturating_mul(4) {
                fragments.entries.shrink_to(interval);
            }
            fragments.requests_until_prune = interval;
        }
        fragments.requests_until_prune -= 1;
        let entry = fragments
            .entries
            .entry((uri.to_owned(), key.source, key.deletion))
            .or_default();
        if let Some(build) = entry.build.upgrade() {
            return Ok(build);
        }
        let build = Arc::new(tokio::sync::Mutex::new(entry.active.upgrade()));
        entry.build = Arc::downgrade(&build);
        Ok(build)
    }

    pub(super) fn prepared(
        &self,
        uri: &str,
        key: &crate::session::index_caches::PreparedRowIdFragmentKey,
        active: &Arc<super::resolve::PreparedFragment>,
    ) -> Result<()> {
        let mut fragments = self.fragments.lock().map_err(|error| {
            Error::internal(format!("row id fragment build registry poisoned: {error}"))
        })?;
        if let Some(entry) = fragments
            .entries
            .get_mut(&(uri.to_owned(), key.source, key.deletion))
        {
            entry.active = Arc::downgrade(active);
        }
        Ok(())
    }
}

impl DeepSizeOf for RowIdFragmentRegistry {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.fragments
            .lock()
            .map(|fragments| {
                fragments.entries.capacity()
                    * std::mem::size_of::<(FragmentBuildKey, FragmentBuildEntry)>()
                    + fragments
                        .entries
                        .keys()
                        .map(|(uri, _, _)| uri.capacity())
                        .sum::<usize>()
            })
            .unwrap_or(0)
    }
}

pub(super) fn deletion_identity(
    dataset: &Dataset,
    fragment: &Fragment,
    needs_deletions: bool,
) -> Result<Option<InternalCacheKey>> {
    fragment
        .deletion_file
        .as_ref()
        .filter(|_| needs_deletions)
        .map(|file| {
            let mut builder = KeyBuilder::new(
                CacheNamespace::root(),
                DeletionFileKey::stable_type_id(),
                DeletionFileKey::schema(),
            );
            DeletionFileKey {
                fragment_id: fragment.id,
                deletion_file: file,
            }
            .write_key(&mut builder);
            if let Some(inline) = &file.inline {
                builder.write_some();
                builder.write_bytes(inline);
            } else {
                builder.write_none();
                write_base_identity(dataset, file.base_id, &mut builder)?;
            }
            Ok(builder.finish())
        })
        .transpose()
}

pub(super) async fn load_source(
    dataset: &Dataset,
    fragment: &Fragment,
    sequence: Arc<RowIdSequence>,
    needs_deletions: bool,
) -> Result<FragmentRowIdIndex> {
    // The index owns and charges these vectors, rather than retaining them in
    // the metadata cache as well.
    let deletion_vector = match &fragment.deletion_file {
        Some(file) if needs_deletions => Arc::new(
            read_deletion_file(
                fragment.id,
                file,
                &dataset.dataset_dir_for_deletion(file)?,
                dataset.object_store_for_deletion(file).await?.as_ref(),
            )
            .await?,
        ),
        _ => Arc::new(DeletionVector::default()),
    };
    Ok(FragmentRowIdIndex {
        fragment_id: fragment.id as u32,
        row_id_sequence: sequence,
        deletion_vector,
    })
}

pub(super) async fn load_deletions(
    dataset: &Dataset,
    fragment: &Fragment,
    deletion: InternalCacheKey,
) -> Result<Arc<DeletionVector>> {
    let file = fragment
        .deletion_file
        .as_ref()
        .ok_or_else(|| Error::internal("row id deletion identity has no deletion file"))?;
    dataset
        .index_cache
        .get_or_insert_with_key(RowIdFragmentDeletionKey { deletion }, || async {
            read_deletion_file(
                fragment.id,
                file,
                &dataset.dataset_dir_for_deletion(file)?,
                dataset.object_store_for_deletion(file).await?.as_ref(),
            )
            .await
        })
        .await
}

fn write_base_identity(
    dataset: &Dataset,
    base_id: Option<u32>,
    builder: &mut KeyBuilder,
) -> Result<()> {
    if let Some(base_id) = base_id {
        let base = dataset.manifest.base_paths.get(&base_id).ok_or_else(|| {
            Error::invalid_input(format!("row id index references unknown base_id {base_id}"))
        })?;
        builder.write_str(&base.path);
        builder.write_u64(base.is_dataset_root as u64);
    } else {
        builder.write_str(&dataset.uri);
        builder.write_u64(1);
    }
    Ok(())
}

pub(super) fn sequence_identity(
    dataset: &Dataset,
    fragment: &Fragment,
) -> Result<InternalCacheKey> {
    let row_id_meta = fragment.row_id_meta.as_ref().ok_or_else(|| {
        Error::internal(format!("fragment {} has no row id metadata", fragment.id))
    })?;
    let file = fragment.row_lineage_file(ROW_ID_FIELD_ID)?;
    let key = RowIdSequenceKey {
        fragment_id: fragment.id,
        row_id_meta,
        lineage_file: file,
    };
    let mut builder = KeyBuilder::new(
        CacheNamespace::root(),
        "lance.dataset.row-id-fragment",
        RowIdSequenceKey::schema(),
    );
    key.write_key(&mut builder);
    builder.write_u64(fragment.physical_rows.map_or(u64::MAX, |rows| rows as u64));
    if let RowIdMeta::Column = row_id_meta {
        let file = file.ok_or_else(|| {
            Error::corrupt_file_named(
                &dataset.uri,
                format!("fragment {} has no spilled row id file", fragment.id),
            )
        })?;
        write_base_identity(dataset, file.base_id, &mut builder)?;
        // The same file may be projected through a different physical column.
        for (field, column) in file.fields.iter().zip(file.column_indices.iter()) {
            if *field == ROW_ID_FIELD_ID {
                builder.write_u32(*column as u32);
            }
        }
    }
    Ok(builder.finish())
}

#[cfg(test)]
pub(super) mod tests {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, UInt64Type};
    use arrow_array::{RecordBatchIterator, record_batch};
    use lance_core::utils::address::RowAddress;
    use lance_datagen::array;
    use lance_table::format::{BasePath, DeletionFile, DeletionFileType};
    use lance_table::rowids::version::RowDatasetVersionSequence;
    use lance_table::transaction::{Operation, Transaction};
    use roaring::RoaringBitmap;
    use rstest::rstest;

    use super::*;
    use crate::dataset::rowids::{
        INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, RowLineage, SPILL_ROW_LINEAGE_CONFIG_KEY,
        get_row_id_index_with, place_row_lineage, resolve_row_ids,
    };
    use crate::dataset::write::CommitBuilder;
    use crate::dataset::{ProjectionRequest, WriteParams};
    use crate::session::Session;
    use crate::session::index_caches::PreparedRowIdFragmentKey;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

    #[test]
    fn registry_reclaims_expired_metadata_without_losing_live_builds() {
        let registry = RowIdFragmentRegistry::default();
        let key = |id: u128| PreparedRowIdFragmentKey {
            source: InternalCacheKey::from_bytes(id.to_le_bytes()),
            deletion: None,
        };
        let live = registry.for_fragment("memory://", &key(0)).unwrap();
        let expired: Vec<_> = (1..=256)
            .map(|id| registry.for_fragment("memory://", &key(id)).unwrap())
            .collect();
        let weak = Arc::downgrade(&expired[0]);
        drop(expired);
        assert!(weak.upgrade().is_none());

        // Churn through unrelated sources after a large set of readers expires.
        // Metadata must stay bounded while the live source keeps its same lock.
        for id in 257..=2304 {
            drop(registry.for_fragment("memory://", &key(id)).unwrap());
            assert!(Arc::ptr_eq(
                &live,
                &registry.for_fragment("memory://", &key(0)).unwrap()
            ));
        }
        let fragments = registry.fragments.lock().unwrap();
        assert!(fragments.entries.len() < 128);
        assert!(fragments.entries.capacity() < 512);
    }

    pub async fn dataset(spilled: bool) -> Dataset {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", array::step::<Int32Type>())
            .into_ram_dataset_with_params(
                FragmentCount::from(4),
                FragmentRowCount::from(10),
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: 10,
                    session: Some(Arc::new(Session::new(1 << 20, 0, Default::default()))),
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        if spilled {
            dataset
                .update_config([
                    (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                    (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
                ])
                .await
                .unwrap();
            super::super::test::compact(&mut dataset, 20).await;
            assert!(
                dataset
                    .manifest
                    .fragments
                    .iter()
                    .all(|fragment| { matches!(fragment.row_id_meta, Some(RowIdMeta::Column)) })
            );
        }
        dataset
    }

    #[tokio::test]
    async fn fragment_identity_resolves_base_and_column() {
        let mut dataset = dataset(true).await;
        let mut fragment = dataset.manifest.fragments[0].clone();
        let file = fragment.row_lineage_file(ROW_ID_FIELD_ID).unwrap().unwrap();
        let path = file.path.clone();
        let initial = sequence_identity(&dataset, &fragment).unwrap();
        let file = fragment
            .files
            .iter_mut()
            .find(|file| file.path == path)
            .unwrap();
        file.base_id = Some(9);
        Arc::make_mut(&mut dataset.manifest).base_paths.insert(
            9,
            BasePath::new(9, "s3://first/root".to_owned(), None, true),
        );
        let first = sequence_identity(&dataset, &fragment).unwrap();
        assert_ne!(initial, first);
        Arc::make_mut(&mut dataset.manifest).base_paths.insert(
            9,
            BasePath::new(9, "s3://second/root".to_owned(), None, true),
        );
        assert_ne!(first, sequence_identity(&dataset, &fragment).unwrap());
        let second = sequence_identity(&dataset, &fragment).unwrap();
        let file = fragment
            .files
            .iter_mut()
            .find(|file| file.path == path)
            .unwrap();
        let position = file
            .fields
            .iter()
            .position(|id| *id == ROW_ID_FIELD_ID)
            .unwrap();
        Arc::make_mut(&mut file.column_indices)[position] += 1;
        assert_ne!(second, sequence_identity(&dataset, &fragment).unwrap());
    }

    #[tokio::test]
    async fn inline_deletion_contents_are_part_of_reuse_identity() {
        let mut dataset = dataset(false).await;
        let mut snapshots = Vec::new();
        for deleted_offset in [0u32, 1] {
            let mut encoded = Vec::new();
            RoaringBitmap::from_iter([deleted_offset])
                .serialize_into(&mut encoded)
                .unwrap();
            let mut fragments = dataset.manifest.fragments.as_ref().clone();
            // Inline contents determine visibility even if the file-like
            // identity fields are unchanged in a manifest supplied by a caller.
            fragments[0].deletion_file = Some(DeletionFile {
                read_version: 1,
                id: 42,
                file_type: DeletionFileType::Bitmap,
                num_deleted_rows: Some(1),
                base_id: None,
                inline: Some(encoded.into()),
            });
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
            let addresses = resolve_row_ids(&dataset, &[0, 1, 2], true)
                .await
                .unwrap()
                .unwrap();
            for (row_id, address) in addresses.into_iter().enumerate() {
                assert_eq!(
                    address,
                    if row_id == deleted_offset as usize {
                        None
                    } else {
                        Some(RowAddress::new_from_parts(0, row_id as u32))
                    }
                );
            }
            snapshots.push(dataset.clone());
        }
        assert_eq!(
            resolve_row_ids(&snapshots[0], &[0], true)
                .await
                .unwrap()
                .unwrap(),
            [None]
        );
        assert_eq!(
            resolve_row_ids(&snapshots[1], &[0], true)
                .await
                .unwrap()
                .unwrap(),
            [Some(RowAddress::new_from_parts(0, 0))]
        );
    }

    #[rstest]
    #[case::inline(false)]
    #[case::spilled(true)]
    #[tokio::test]
    async fn new_overlap_changes_effective_deletions(#[case] spilled: bool) {
        let mut current = dataset(spilled).await;
        current.delete("i = 7").await.unwrap();
        let old = current.clone();
        let old_index = get_row_id_index_with(&old, false).await.unwrap().unwrap();
        // Without overlap this mode deliberately maps even deleted IDs.
        assert!(old_index.get(7).unwrap().is_some());

        let batch = record_batch!(("i", Int32, [400])).unwrap();
        current
            .append(
                RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                None,
            )
            .await
            .unwrap();
        let mut fragments = current.manifest.fragments.as_ref().clone();
        let fragment = fragments.last_mut().unwrap();
        let new_fragment_id = fragment.id as u32;
        let lineage = RowLineage {
            row_ids: RowIdSequence::try_from_iter([7]).unwrap(),
            created_at: RowDatasetVersionSequence::from_uniform_row_count(1, 1),
            last_updated_at: RowDatasetVersionSequence::from_uniform_row_count(1, 1),
        };
        place_row_lineage(&current, &lineage)
            .await
            .unwrap()
            .apply(fragment);
        let transaction = Transaction::new_from_version(
            current.version().version,
            Operation::Merge {
                fragments,
                schema: current.schema().clone(),
                preserves_nullability: true,
            },
        );
        current = CommitBuilder::new(Arc::new(current))
            .execute(transaction)
            .await
            .unwrap();
        let index = get_row_id_index_with(&current, false)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            index.get(7).unwrap(),
            Some(RowAddress::new_from_parts(new_fragment_id, 0))
        );
        let batch = current
            .take_rows(
                &[7, 7],
                ProjectionRequest::from_columns(["i"], current.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            &[400, 400]
        );
        let revisited = get_row_id_index_with(&old, false).await.unwrap().unwrap();
        assert_eq!(revisited.get(7).unwrap(), old_index.get(7).unwrap());
        assert_ne!(revisited.get(7).unwrap(), index.get(7).unwrap());
    }

    async fn physical_oracle(dataset: &Dataset, ids: &[u64]) -> Vec<Option<RowAddress>> {
        let mut scan = dataset.scan();
        scan.with_row_id().with_row_address();
        let batch = scan.try_into_batch().await.unwrap();
        let row_ids = batch[lance_core::ROW_ID].as_primitive::<UInt64Type>();
        let addresses = batch[lance_core::ROW_ADDR].as_primitive::<UInt64Type>();
        let expected: HashMap<_, _> = row_ids
            .values()
            .iter()
            .copied()
            .zip(addresses.values().iter().copied().map(RowAddress::from))
            .collect();
        ids.iter().map(|id| expected.get(id).copied()).collect()
    }

    #[rstest]
    #[case::inline(false)]
    #[case::spilled(true)]
    #[tokio::test]
    async fn fragment_cache_reuses_seven_snapshots_and_survives_compaction_restore(
        #[case] spilled: bool,
    ) {
        let mut current = dataset(spilled).await;
        let original_version = current.version().version;
        let mut snapshots = vec![current.clone()];
        for deleted_id in 0..6 {
            let batch = record_batch!(("i", Int32, [40 + deleted_id])).unwrap();
            current
                .append(
                    RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                    None,
                )
                .await
                .unwrap();
            current.delete(&format!("i = {deleted_id}")).await.unwrap();
            snapshots.push(current.clone());
        }
        let ids = [0, 5, 7, 23, 39, 40, 45, 999, 7];
        let mut oracles = Vec::new();
        for snapshot in &snapshots {
            oracles.push(physical_oracle(snapshot, &ids).await);
        }
        current.index_cache.clear().await;
        for (snapshot, expected) in snapshots.iter().zip(&oracles) {
            assert_eq!(
                resolve_row_ids(snapshot, &ids, true)
                    .await
                    .unwrap()
                    .unwrap(),
                *expected
            );
        }
        let before = current.object_store.io_stats_snapshot();
        for _ in 0..3 {
            for (snapshot, expected) in snapshots.iter().zip(&oracles) {
                assert_eq!(
                    resolve_row_ids(snapshot, &ids, true)
                        .await
                        .unwrap()
                        .unwrap(),
                    *expected
                );
            }
        }
        assert_eq!(
            current.object_store.io_stats_snapshot().read_bytes,
            before.read_bytes,
            "resident snapshots must share unchanged fragment lookups without rereading sources"
        );
        for snapshot in &snapshots {
            assert!(
                snapshot
                    .index_cache
                    .get_with_key(&crate::session::index_caches::RowIdIndexKey {
                        version: snapshot.manifest.version,
                        with_deletions: true,
                        e_tag: snapshot.manifest_location.e_tag.as_deref(),
                    })
                    .await
                    .is_none(),
                "demand lookups must not build per-snapshot complete indices"
            );
        }
        super::super::test::compact(&mut current, 100).await;
        assert_eq!(
            resolve_row_ids(&current, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            physical_oracle(&current, &ids).await
        );
        let batch = current
            .take_rows(
                &[7, 23, 7],
                ProjectionRequest::from_columns(["i"], current.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            &[7, 23, 7]
        );
        let mut restored = current.checkout_version(original_version).await.unwrap();
        restored.restore().await.unwrap();
        assert_eq!(
            resolve_row_ids(&restored, &ids, true)
                .await
                .unwrap()
                .unwrap(),
            physical_oracle(&restored, &ids).await
        );
        assert_eq!(
            resolve_row_ids(&restored, &[0, 40], true)
                .await
                .unwrap()
                .unwrap(),
            [oracles[0][0], None]
        );
    }
}
