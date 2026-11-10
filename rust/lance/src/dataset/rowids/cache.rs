// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One charged current view, with immutable fragment state reused on replacement.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex, Weak};

use futures::{StreamExt, TryStreamExt};
use lance_core::cache::{CacheKey, CacheKeySchema, CacheNamespace, InternalCacheKey, KeyBuilder};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::deletion::DeletionVector;
use lance_table::format::{Fragment, ROW_ID_FIELD_ID, RowIdMeta};
use lance_table::io::deletion::read_deletion_file;
use lance_table::rowids::{FragmentRowIdIndex, RowIdIndex, RowIdSequence};

use crate::session::caches::{DeletionFileKey, RowIdSequenceKey};
use crate::session::index_caches::RowIdIndexKey;
use crate::{Dataset, Error, Result};

use super::{overlapping_id_spans, read_row_id_sequence};

/// Locks live only while a caller builds or waits. The registry holds no indices
/// and prunes expired entries, including datasets no longer used by the session.
#[derive(Default)]
pub struct RowIdIndexBuildLocks {
    locks: Mutex<DatasetBuildLocks>,
}

type DatasetBuildLocks = HashMap<String, Weak<tokio::sync::Mutex<()>>>;

impl RowIdIndexBuildLocks {
    pub(super) fn for_dataset(&self, uri: &str) -> Result<Arc<tokio::sync::Mutex<()>>> {
        let mut locks = self.locks.lock().map_err(|error| {
            Error::internal(format!("row id index build lock poisoned: {error}"))
        })?;
        locks.retain(|_, lock| lock.strong_count() > 0);
        let slot = locks.entry(uri.to_owned()).or_default();
        if let Some(lock) = slot.upgrade() {
            return Ok(lock);
        }
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        *slot = Arc::downgrade(&lock);
        Ok(lock)
    }
}

impl DeepSizeOf for RowIdIndexBuildLocks {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.locks
            .lock()
            .map(|locks| {
                locks.capacity() * std::mem::size_of::<(String, Weak<tokio::sync::Mutex<()>>)>()
                    + locks.keys().map(|uri| uri.capacity()).sum::<usize>()
            })
            .unwrap_or(0)
    }
}

#[derive(Debug)]
pub(super) struct CachedFragment {
    pub(super) identity: InternalCacheKey,
    pub(super) span: Option<RangeInclusive<u64>>,
    pub(super) deletion_identity: Option<InternalCacheKey>,
}

/// Replacing this entry drops the cache's ownership of the previous view. Active
/// readers may retain it, just as they may retain any evicted cache value.
#[derive(Debug)]
pub struct CachedRowIdIndex {
    version: u64,
    e_tag: String,
    pub(crate) index: Arc<RowIdIndex>,
    fragments: Vec<CachedFragment>,
}

impl DeepSizeOf for CachedRowIdIndex {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.e_tag.capacity()
            + self.index.deep_size_of_children(context)
            + self.fragments.capacity() * std::mem::size_of::<CachedFragment>()
    }
}

impl CachedRowIdIndex {
    pub(super) fn matches(&self, dataset: &Dataset) -> bool {
        self.version == dataset.manifest.version
            && Some(self.e_tag.as_str()) == dataset.manifest_location.e_tag.as_deref()
    }
}

/// One measured complete-view size per deletion mode, replaced on version
/// changes. It owns no lookup graph or request history. Keeping this in the
/// metadata cache prevents an oversized view from triggering repeated rebuilds
/// while its smaller fragment working set remains resident.
#[derive(Debug, DeepSizeOf)]
pub(super) struct RowIdIndexSize {
    version: u64,
    e_tag: String,
    bytes: usize,
}

impl RowIdIndexSize {
    pub(super) fn new(cached: &CachedRowIdIndex) -> Self {
        Self {
            version: cached.version,
            e_tag: cached.e_tag.clone(),
            // Capacity includes the canonical key as well as the cached Arc.
            bytes: cached.deep_size_of()
                + 2 * std::mem::size_of::<std::sync::atomic::AtomicUsize>()
                + std::mem::size_of::<InternalCacheKey>(),
        }
    }

    pub(super) fn exceeds(&self, dataset: &Dataset, limit: u64) -> bool {
        self.version == dataset.manifest.version
            && Some(self.e_tag.as_str()) == dataset.manifest_location.e_tag.as_deref()
            && self.bytes as u64 > limit
    }
}

pub(super) struct RowIdIndexSizeKey {
    pub with_deletions: bool,
}

impl CacheKey for RowIdIndexSizeKey {
    type ValueType = RowIdIndexSize;

    fn key(&self) -> Cow<'_, str> {
        Cow::Borrowed("row_id_index_size")
    }

    fn type_name() -> &'static str {
        "RowIdIndexSize"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.row-id-index-size", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.with_deletions as u64);
    }
}

pub(super) async fn get(dataset: &Dataset, with_deletions: bool) -> Result<Arc<CachedRowIdIndex>> {
    let key = RowIdIndexKey { with_deletions };
    if let Some(cached) = dataset.index_cache.get_with_key(&key).await
        && cached.matches(dataset)
    {
        return Ok(cached);
    }

    let lock = dataset
        .session()
        .row_id_index_builds
        .for_dataset(&dataset.uri)?;
    let _guard = lock.lock().await;
    // Recheck after waiting: another dataset handle may have built this version.
    let mut previous = dataset.index_cache.get_with_key(&key).await;
    if let Some(cached) = &previous
        && cached.matches(dataset)
    {
        return Ok(cached.clone());
    }
    if previous.is_none() {
        // Effective deletion fingerprints decide which prepared units are valid
        // across modes; a mode change need not duplicate every unchanged unit.
        previous = dataset
            .index_cache
            .get_with_key(&RowIdIndexKey {
                with_deletions: !with_deletions,
            })
            .await;
    }
    let (cached, restore_units) = if let Some(promotion) =
        super::resolve::promote_fragments(dataset, with_deletions).await?
    {
        let super::resolve::FragmentPromotion {
            index,
            fragments,
            units,
            loaded_fragments,
        } = promotion;
        tracing::debug!(
            event = "row_id_index_build",
            total_fragments = fragments.len(),
            retained_fragments = fragments.len() - loaded_fragments,
            loaded_fragments,
            prepared_fragments = loaded_fragments,
            "combined prepared row id fragments"
        );
        (
            CachedRowIdIndex {
                version: dataset.manifest.version,
                e_tag: dataset.manifest_location.e_tag.clone().ok_or_else(|| {
                    Error::internal("cached row id index requires a manifest generation token")
                })?,
                index: Arc::new(index),
                fragments,
            },
            Some(units),
        )
    } else {
        (
            build(dataset, with_deletions, previous.as_deref()).await?,
            None,
        )
    };
    let cached = Arc::new(cached);
    dataset
        .index_cache
        .insert_with_key(&key, cached.clone())
        .await;
    if let Some(units) = restore_units {
        let admitted = dataset
            .index_cache
            .get_with_key(&key)
            .await
            .is_some_and(|value| Arc::ptr_eq(&value, &cached));
        if !admitted {
            // A complete view may exceed admission even when individual units
            // fit. Keep the bounded fragment working set; the returned complete
            // index is owned only by its active caller and shares those units.
            for (key, unit) in units {
                dataset.index_cache.insert_with_key(&key, unit).await;
            }
        }
    }
    Ok(cached)
}

pub(super) async fn build(
    dataset: &Dataset,
    with_deletions: bool,
    previous: Option<&CachedRowIdIndex>,
) -> Result<CachedRowIdIndex> {
    let previous_fragments: HashMap<_, _> = previous
        .into_iter()
        .flat_map(|cached| &cached.fragments)
        .map(|fragment| (fragment.identity, fragment))
        .collect();
    let mut loads = Vec::with_capacity(dataset.manifest.fragments.len());
    for fragment in dataset.manifest.fragments.iter() {
        let identity = sequence_identity(dataset, fragment)?;
        let prior = previous_fragments.get(&identity).copied();
        loads.push(async move {
            // Prepared state owns everything needed for lookup. Unchanged
            // fragments need neither their original IDs nor another span scan.
            let sequence = if prior.is_some() {
                None
            } else {
                Some(Arc::new(read_row_id_sequence(dataset, fragment).await?))
            };
            let span = match prior {
                Some(prior) => prior.span.clone(),
                None => sequence
                    .as_ref()
                    .and_then(|sequence| sequence.row_id_range()),
            };
            Ok::<_, Error>((fragment, identity, sequence, span, prior))
        });
    }
    let loaded: Vec<_> = futures::stream::iter(loads)
        .buffered(dataset.object_store.io_parallelism())
        .try_collect()
        .await?;
    let needs_deletions = if with_deletions {
        vec![true; loaded.len()]
    } else {
        overlapping_id_spans(
            &loaded
                .iter()
                .map(|(_, _, _, span, _)| span.clone())
                .collect::<Vec<_>>(),
        )
    };
    let mut loads = Vec::with_capacity(loaded.len());
    for ((fragment, identity, sequence, span, prior), needs_deletions) in
        loaded.into_iter().zip(&needs_deletions)
    {
        loads.push(async move {
            let deletion_identity = deletion_identity(dataset, fragment, *needs_deletions)?;
            let source = if prior.is_some_and(|prior| prior.deletion_identity == deletion_identity)
            {
                None
            } else {
                let sequence = match sequence {
                    Some(sequence) => sequence,
                    None => Arc::new(read_row_id_sequence(dataset, fragment).await?),
                };
                Some(load_source(dataset, fragment, sequence, *needs_deletions).await?)
            };
            let cached = CachedFragment {
                identity,
                span,
                deletion_identity,
            };
            Ok::<_, Error>((cached, source))
        });
    }
    let loaded: Vec<_> = futures::stream::iter(loads)
        .buffered(dataset.object_store.io_parallelism())
        .try_collect()
        .await?;
    let (fragments, mut sources): (Vec<CachedFragment>, Vec<Option<FragmentRowIdIndex>>) =
        loaded.into_iter().unzip();
    let retained: Vec<_> = dataset
        .manifest
        .fragments
        .iter()
        .zip(&sources)
        .filter_map(|(fragment, source)| source.is_none().then_some(fragment.id as u32))
        .collect();
    let changed: Vec<_> = sources.iter_mut().filter_map(Option::take).collect();
    let reused = previous
        .map(|previous| previous.index.try_reuse_fragments(&retained, &changed))
        .transpose()?
        .flatten();
    let has_reused_index = reused.is_some();
    let changed_fragments = changed.len();
    let index = if let Some(index) = reused {
        index
    } else {
        // Partial changes to an inseparable merged group require reconstruction.
        // Reload only then; retain already loaded changed sources for the build.
        let mut changed: HashMap<_, _> = changed
            .into_iter()
            .map(|source| (source.fragment_id, source))
            .collect();
        let mut loads = Vec::with_capacity(fragments.len());
        for (fragment, needs_deletions) in dataset.manifest.fragments.iter().zip(needs_deletions) {
            let source = changed.remove(&(fragment.id as u32));
            loads.push(async move {
                match source {
                    Some(source) => Ok(source),
                    None => {
                        let sequence = Arc::new(read_row_id_sequence(dataset, fragment).await?);
                        load_source(dataset, fragment, sequence, needs_deletions).await
                    }
                }
            });
        }
        let sources: Vec<_> = futures::stream::iter(loads)
            .buffered(dataset.object_store.io_parallelism())
            .try_collect()
            .await?;
        RowIdIndex::new(&sources)?
    };
    tracing::debug!(
        event = "row_id_index_build",
        total_fragments = fragments.len(),
        retained_fragments = if has_reused_index { retained.len() } else { 0 },
        loaded_fragments = if has_reused_index {
            changed_fragments
        } else {
            fragments.len()
        },
        prepared_fragments = if has_reused_index {
            changed_fragments
        } else {
            fragments.len()
        },
        "prepared row id index"
    );
    Ok(CachedRowIdIndex {
        version: dataset.manifest.version,
        e_tag: dataset.manifest_location.e_tag.clone().ok_or_else(|| {
            Error::internal("cached row id index requires a manifest generation token")
        })?,
        index: Arc::new(index),
        fragments,
    })
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
        get_row_id_index, get_row_id_index_with, place_row_lineage,
    };
    use crate::dataset::write::CommitBuilder;
    use crate::dataset::{ProjectionRequest, WriteParams};
    use crate::session::Session;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

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

    async fn assert_payload(dataset: &Dataset, ids: &[u64]) {
        let batch = dataset
            .take_rows(
                ids,
                ProjectionRequest::from_columns(["i"], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            ids.iter().map(|id| *id as i32).collect::<Vec<_>>()
        );
    }

    async fn row_id_occupancy(dataset: &Dataset) -> lance_core::cache::CacheTypeOccupancy {
        dataset
            .index_cache
            .diagnostics_by_type(lance_core::cache::CacheSnapshotMode::Approximate)
            .await
            .by_type
            .unwrap()
            .occupancy
            .unwrap()
            .types
            .into_iter()
            .find(|entry| entry.type_name == RowIdIndexKey::stable_type_id())
            .unwrap()
    }

    #[rstest]
    #[case::inline(false)]
    #[case::spilled(true)]
    #[tokio::test]
    async fn reuse_append_snapshots_and_lifecycle(#[case] spilled: bool) {
        let mut current = dataset(spilled).await;
        let original = current.clone();
        let original_index = get_row_id_index(&original).await.unwrap().unwrap();
        let key = RowIdIndexKey {
            with_deletions: true,
        };
        let before = original.index_cache.get_with_key(&key).await.unwrap();
        let charge_before = row_id_occupancy(&original).await.size_bytes;
        let batch = record_batch!(("i", Int32, [40, 41, 42])).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        current.append(reader, None).await.unwrap();

        let reads_before = current.object_store.io_stats_snapshot();
        let index = get_row_id_index(&current).await.unwrap().unwrap();
        assert_eq!(
            current.object_store.io_stats_snapshot().read_iops,
            reads_before.read_iops
        );
        let after = current.index_cache.get_with_key(&key).await.unwrap();
        assert_eq!(after.fragments.len(), before.fragments.len() + 1);
        for (old, new) in before.fragments.iter().zip(&after.fragments) {
            assert_eq!(old.identity, new.identity);
            assert_eq!(old.span, new.span);
            assert_eq!(old.deletion_identity, new.deletion_identity);
        }
        assert!(index.get(40).unwrap().is_some());
        assert!(original_index.get(40).unwrap().is_none());
        assert_eq!(
            index.get_many(&[7, 999, 7]).unwrap(),
            vec![
                original_index.get(7).unwrap(),
                None,
                original_index.get(7).unwrap()
            ]
        );
        let occupancy = row_id_occupancy(&current).await;
        let charge_after = occupancy.size_bytes;
        assert!(charge_after < charge_before * 2);
        // A single replacement entry owns the current graph; old readers do
        // not leave another full charged version in the cache.
        assert_eq!(occupancy.num_entries, 1);
        assert_payload(&current, &[23, 0, 23, 40]).await;
        assert_payload(&original, &[31, 7, 31]).await;

        let previous_index = Arc::downgrade(&original_index);
        drop(before);
        drop(original_index);
        assert!(previous_index.upgrade().is_none());
        drop(after);
        drop(index);

        current.delete("i = 7").await.unwrap();
        let deleted = get_row_id_index(&current).await.unwrap().unwrap();
        assert!(deleted.get(7).unwrap().is_none());
        assert_payload(&current, &[23, 40, 0]).await;
        let old = get_row_id_index(&original).await.unwrap().unwrap();
        assert!(old.get(7).unwrap().is_some());

        super::super::test::compact(&mut current, 100).await;
        assert_payload(&current, &[23, 40, 0]).await;
        assert!(
            get_row_id_index(&current)
                .await
                .unwrap()
                .unwrap()
                .get(7)
                .unwrap()
                .is_none()
        );
        let mut restored = current
            .checkout_version(original.version().version)
            .await
            .unwrap();
        restored.restore().await.unwrap();
        assert_payload(&restored, &[31, 7, 0]).await;
        assert!(
            get_row_id_index(&restored)
                .await
                .unwrap()
                .unwrap()
                .get(40)
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn concurrent_loads_share_one_current_view() {
        let dataset = dataset(true).await;
        let (first, second, third) = tokio::join!(
            get_row_id_index(&dataset),
            get_row_id_index(&dataset),
            get_row_id_index(&dataset)
        );
        let first = first.unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &second.unwrap().unwrap()));
        assert!(Arc::ptr_eq(&first, &third.unwrap().unwrap()));
        let mut scan = dataset.scan();
        scan.with_row_address().filter("i = 23").unwrap();
        let oracle = scan.try_into_batch().await.unwrap();
        assert_eq!(
            first.get(23).unwrap(),
            Some(RowAddress::from(
                oracle[lance_core::ROW_ADDR]
                    .as_primitive::<UInt64Type>()
                    .value(0)
            ))
        );
        assert_eq!(row_id_occupancy(&dataset).await.num_entries, 1);
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

    #[rstest]
    #[case::inline(false)]
    #[case::spilled(true)]
    #[tokio::test]
    async fn shuffled_overlapping_ids_reuse_and_take(#[case] spilled: bool) {
        let mut dataset = dataset(false).await;
        if spilled {
            dataset
                .update_config([
                    (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                    (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
                ])
                .await
                .unwrap();
        }
        let mut fragments = dataset.manifest.fragments.as_ref().clone();
        for (fragment_number, fragment) in fragments.iter_mut().enumerate() {
            // Multiplication by 7 permutes 0..40. Every fragment's row-id span
            // overlaps, while the physical payload remains its original offset.
            let ids: Vec<_> = (0..10)
                .map(|offset| ((fragment_number * 10 + offset) as u64 * 7) % 40)
                .collect();
            let lineage = RowLineage {
                row_ids: RowIdSequence::try_from_iter(ids).unwrap(),
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
        let mut dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await
            .unwrap();
        let old = dataset.clone();
        let old_index = get_row_id_index(&old).await.unwrap().unwrap();
        let key = RowIdIndexKey {
            with_deletions: true,
        };
        let before = old.index_cache.get_with_key(&key).await.unwrap();
        let batch = record_batch!(("i", Int32, [40, 41])).unwrap();
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                None,
            )
            .await
            .unwrap();
        let reads_before = dataset.object_store.io_stats_snapshot();
        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_iops,
            reads_before.read_iops
        );
        let after = dataset.index_cache.get_with_key(&key).await.unwrap();
        for (old, new) in before.fragments.iter().zip(&after.fragments) {
            assert_eq!(old.identity, new.identity);
            assert_eq!(old.span, new.span);
        }
        for row_id in 0..40 {
            // 23 is the multiplicative inverse of 7 modulo 40.
            let position = (row_id * 23) % 40;
            let expected = Some(RowAddress::new_from_parts(
                (position / 10) as u32,
                (position % 10) as u32,
            ));
            assert_eq!(index.get(row_id).unwrap(), expected);
            assert_eq!(old_index.get(row_id).unwrap(), expected);
        }
        let batch = dataset
            .take_rows(
                &[9, 23, 9, 40],
                ProjectionRequest::from_columns(["i"], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            &[7, 9, 7, 40]
        );
        assert_eq!(index.get(999).unwrap(), None);
        dataset.delete("i = 7").await.unwrap();
        assert!(
            get_row_id_index(&dataset)
                .await
                .unwrap()
                .unwrap()
                .get(9)
                .unwrap()
                .is_none()
        );
        assert!(old_index.get(9).unwrap().is_some());
    }

    #[tokio::test]
    async fn inline_deletion_contents_are_part_of_reuse_identity() {
        let mut dataset = dataset(false).await;
        let mut indices = Vec::new();
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
            let index = get_row_id_index(&dataset).await.unwrap().unwrap();
            for row_id in 0..3 {
                assert_eq!(
                    index.get(row_id).unwrap(),
                    if row_id == u64::from(deleted_offset) {
                        None
                    } else {
                        Some(RowAddress::new_from_parts(0, row_id as u32))
                    }
                );
            }
            indices.push(index);
        }
        assert_eq!(indices[0].get(0).unwrap(), None);
        assert_eq!(
            indices[1].get(0).unwrap(),
            Some(RowAddress::new_from_parts(0, 0))
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

    #[tokio::test]
    async fn append_with_old_index_and_without_original_sequences() {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", array::step::<Int32Type>())
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
                (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
            ])
            .await
            .unwrap();
        let mut fragments = dataset.manifest.fragments.as_ref().clone();
        let lineage = RowLineage {
            row_ids: RowIdSequence::try_from_iter((0..4096).map(|position| (position * 7) % 4096))
                .unwrap(),
            created_at: RowDatasetVersionSequence::from_uniform_row_count(4096, 1),
            last_updated_at: RowDatasetVersionSequence::from_uniform_row_count(4096, 1),
        };
        place_row_lineage(&dataset, &lineage)
            .await
            .unwrap()
            .apply(&mut fragments[0]);
        assert!(matches!(fragments[0].row_id_meta, Some(RowIdMeta::Column)));
        let transaction = Transaction::new_from_version(
            dataset.version().version,
            Operation::Merge {
                fragments,
                schema: dataset.schema().clone(),
                preserves_nullability: true,
            },
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await
            .unwrap();
        let compact = build(&dataset, true, None).await.unwrap();
        // Admit one prepared graph and the small append directory, but not an
        // additional copy of the original shuffled IDs. Hold the old view below.
        let capacity = compact.deep_size_of() + 8192;
        drop(compact);
        let session = Arc::new(Session::new(
            capacity,
            0,
            dataset.session().store_registry(),
        ));
        // A plain memory:// reopen creates another store. Keep this fixture's
        // existing store binding while replacing both session cache scopes.
        let mut dataset = dataset;
        dataset.index_cache = Arc::new(session.index_cache.for_dataset(&dataset.uri));
        dataset.metadata_cache = Arc::new(session.metadata_cache.for_dataset(&dataset.uri));
        dataset.session = session.clone();
        let first = get_row_id_index(&dataset).await.unwrap().unwrap();
        let before = dataset.object_store.io_stats_snapshot();
        let second = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert!(session.index_cache_stats().await.size_bytes <= capacity);
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_iops,
            before.read_iops
        );
        assert_eq!(
            second.get_many(&[0, 7, 14, 4095, 9999, 7]).unwrap(),
            vec![
                Some(RowAddress::new_from_parts(0, 0)),
                Some(RowAddress::new_from_parts(0, 1)),
                Some(RowAddress::new_from_parts(0, 2)),
                Some(RowAddress::new_from_parts(0, 585)),
                None,
                Some(RowAddress::new_from_parts(0, 1)),
            ]
        );
        let batch = dataset
            .take_rows(
                &[7, 14, 0],
                ProjectionRequest::from_columns(["i"], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            &[1, 2, 0]
        );
        let old_snapshot = dataset.clone();
        let batch = record_batch!(("i", Int32, [4096])).unwrap();
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                None,
            )
            .await
            .unwrap();
        let before = dataset.object_store.io_stats_snapshot();
        let appended = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_iops,
            before.read_iops
        );
        assert_eq!(appended.get(7).unwrap(), first.get(7).unwrap());
        assert!(appended.get(4096).unwrap().is_some());
        assert_eq!(first.get(4096).unwrap(), None);
        assert!(session.index_cache_stats().await.size_bytes <= capacity);
        let revisited = get_row_id_index(&old_snapshot).await.unwrap().unwrap();
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_iops,
            before.read_iops
        );
        assert_eq!(revisited.get(4096).unwrap(), None);
        assert_eq!(revisited.get(7).unwrap(), first.get(7).unwrap());
    }
}
