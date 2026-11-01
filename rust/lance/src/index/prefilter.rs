// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Secondary Index pre-filter
//!
//! Based on the query, we might have information about which fragment ids and
//! row ids can be excluded from the search.

use std::cell::OnceCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::stream;
use lance_core::utils::tokio::spawn_cpu;
use lance_select::{RowAddrMask, RowAddrTreeMap, RowSetOps};
use lance_table::format::{Fragment, IndexMetadata, ROW_ID_FIELD_ID};
use roaring::RoaringBitmap;
use tokio::join;
use tracing::Instrument;
use tracing::instrument;

use crate::Dataset;
use crate::Result;
use crate::dataset::fragment::FileFragment;
use crate::dataset::rowids::load_row_id_sequence;
use crate::utils::future::SharedPrerequisite;

pub use lance_index::prefilter::{FilterLoader, PreFilter};

/// Filter out row ids that we know are not relevant to the query.
///
/// This could be both rows that are deleted or a prefilter
/// that should be applied to the search
///
/// This struct is for internal use only and has no stability guarantees.
pub struct DatasetPreFilter {
    // Expressing these as tasks allows us to start calculating the block list
    // and allow list at the same time we start searching the query.  We will await
    // these tasks only when we've done as much work as we can without them.
    pub(super) deleted_ids: Option<Arc<SharedPrerequisite<Arc<RowAddrMask>>>>,
    pub(super) filtered_ids: Option<Arc<SharedPrerequisite<Arc<RowAddrMask>>>>,
    // Fragment IDs whose data is still in the index but has been removed from the dataset.
    // Used by FTS merge-on-read to prune stale fragments at search time.
    pub(super) deleted_fragments: Option<RoaringBitmap>,
    // Row addresses whose index entries are stale due to a newer data overlay committed after
    // the index was built. Computed synchronously at plan time and ANDead into the final mask
    // so the index never returns those rows.
    pub(super) overlay_block: Option<RowAddrMask>,
    // When the tasks are finished this is the combined filter
    pub(super) final_mask: Mutex<OnceCell<Arc<RowAddrMask>>>,
}

impl DatasetPreFilter {
    pub fn new(
        dataset: Arc<Dataset>,
        indices: &[IndexMetadata],
        filter: Option<Box<dyn FilterLoader>>,
    ) -> Self {
        let filter = filter.map(|filter| {
            async move { filter.load().await.map(Arc::new) }
                .in_current_span()
                .boxed()
        });
        Self::new_with_filter_future(dataset, indices, filter)
    }

    pub(crate) fn new_with_filter_future(
        dataset: Arc<Dataset>,
        indices: &[IndexMetadata],
        filter: Option<BoxFuture<'static, Result<Arc<RowAddrMask>>>>,
    ) -> Self {
        let mut fragments = RoaringBitmap::new();
        let all_have_bitmaps = indices.iter().all(|idx| idx.fragment_bitmap.is_some());
        if !all_have_bitmaps {
            if let Some(max_fragment_id) = dataset.manifest.max_fragment_id() {
                fragments.insert_range(0..=max_fragment_id as u32);
            }
        } else {
            indices.iter().for_each(|idx| {
                fragments |= idx.fragment_bitmap.as_ref().unwrap();
            });
        }
        let deleted_ids = if all_have_bitmaps {
            Self::create_restricted_deletion_mask(dataset, fragments)
        } else {
            Self::create_deletion_mask(dataset, fragments)
        }
        .map(SharedPrerequisite::spawn);
        let filtered_ids = filter.map(SharedPrerequisite::spawn);
        Self {
            deleted_ids,
            filtered_ids,
            deleted_fragments: None,
            overlay_block: None,
            final_mask: Mutex::new(OnceCell::new()),
        }
    }

    /// Like [`Self::new_with_filter_future`], but restricted to an explicit
    /// fragment set instead of the union of the indices' fragment bitmaps.
    ///
    /// `combined_fields` needs the intersection of its target columns' coverage: a
    /// cross-field BM25F score is only complete when every column's index holds the
    /// row, so the union [`Self::new`] derives would admit rows that only some
    /// columns cover. Routing through [`Self::create_restricted_deletion_mask`]
    /// keeps the restriction in whichever id space the indices actually store,
    /// stable row ids or row addresses, which a fragment-address block list cannot
    /// do.
    pub(crate) fn new_restricted_to_fragments(
        dataset: Arc<Dataset>,
        fragments: RoaringBitmap,
        filter: Option<BoxFuture<'static, Result<Arc<RowAddrMask>>>>,
    ) -> Self {
        let deleted_ids = Self::create_restricted_deletion_mask(dataset, fragments)
            .map(SharedPrerequisite::spawn);
        let filtered_ids = filter.map(SharedPrerequisite::spawn);
        Self {
            deleted_ids,
            filtered_ids,
            deleted_fragments: None,
            overlay_block: None,
            final_mask: Mutex::new(OnceCell::new()),
        }
    }

    #[instrument(level = "debug", skip_all)]
    async fn do_create_deletion_mask(
        dataset: Arc<Dataset>,
        missing_frags: Vec<u32>,
        frags_with_deletion_files: Vec<u32>,
    ) -> Result<Arc<RowAddrMask>> {
        let fragments = dataset.get_fragments();
        let frag_map: Arc<HashMap<u32, &FileFragment>> = Arc::new(HashMap::from_iter(
            fragments.iter().map(|frag| (frag.id() as u32, frag)),
        ));
        let frag_id_deletion_vectors = stream::iter(
            frags_with_deletion_files
                .iter()
                .map(|frag_id| (frag_id, frag_map.clone())),
        )
        .map(|(frag_id, frag_map)| async move {
            let frag = frag_map.get(frag_id).unwrap();
            frag.get_deletion_vector()
                .await
                .transpose()
                .unwrap()
                .map(|deletion_vector| (*frag_id, RoaringBitmap::from(deletion_vector.as_ref())))
        })
        .collect::<Vec<_>>()
        .await;
        let mut frag_id_deletion_vectors = stream::iter(frag_id_deletion_vectors)
            .buffer_unordered(dataset.object_store.io_parallelism());

        let mut deleted_ids = RowAddrTreeMap::new();
        while let Some((id, deletion_vector)) = frag_id_deletion_vectors.try_next().await? {
            deleted_ids.insert_bitmap(id, deletion_vector);
        }

        for frag_id in missing_frags.into_iter() {
            deleted_ids.insert_fragment(frag_id);
        }
        Ok(Arc::new(RowAddrMask::from_block(deleted_ids)))
    }

    /// Keep the live address domain explicit: a staged index can physically
    /// retain rows whose fragments were pruned from its committed coverage.
    async fn do_create_restricted_deletion_mask(
        dataset: Arc<Dataset>,
        fragments: RoaringBitmap,
    ) -> Result<Arc<RowAddrMask>> {
        let mut fragment_rows = stream::iter(
            dataset
                .get_fragments()
                .into_iter()
                .filter(|fragment| fragments.contains(fragment.id() as u32)),
        )
        .map(|fragment| async move {
            let id = fragment.id() as u32;
            // Bound every fragment by its physical row count. Even undeleted
            // fragments can later intersect with a block filter or overlay;
            // Full - Partial would expand the entire u32 offset domain.
            let num_rows = fragment.physical_rows().await?;
            let mut live_rows = RoaringBitmap::new();
            live_rows.insert_range(0..num_rows as u32);
            if let Some(deletions) = fragment.get_deletion_vector().await? {
                live_rows -= &RoaringBitmap::from(deletions.as_ref());
            }
            Ok::<_, crate::Error>((id, live_rows))
        })
        .buffer_unordered(dataset.object_store.io_parallelism());
        let mut allowed = RowAddrTreeMap::new();
        while let Some((id, rows)) = fragment_rows.try_next().await? {
            allowed.insert_bitmap(id, rows);
        }
        Ok(Arc::new(RowAddrMask::from_allowed(allowed)))
    }

    #[instrument(level = "debug", skip_all)]
    pub(super) async fn do_create_deletion_mask_row_id(
        dataset: Arc<Dataset>,
        restrict_to: Option<RoaringBitmap>,
    ) -> Result<Arc<RowAddrMask>> {
        // The mask is an allow-list of stable row ids. When `restrict_to` is
        // set the iteration is limited to the listed fragments, so the
        // resulting list excludes stable row ids whose *current* physical home
        // is outside the restriction. This is the missing piece for the
        // stable-row-id branch of #6563: without it, the merge_insert UNION
        // (indexed scan ∪ unindexed-fragments scan) sees the same logical row
        // twice — once via the BTREE (which holds the row's stable_row_id) and
        // once via the unindexed scan (which holds the fragment the row now
        // lives in). See issue #6877.
        //
        // The allow list is assembled from per-fragment pieces, each cached
        // under its content identity (row id generation + deletion file), so
        // repeated builds with overlapping `restrict_to` sets reload only the
        // fragments they have not seen yet, and commits that leave a fragment
        // untouched keep its piece warm.
        async fn build_allow_list_piece(
            dataset: Arc<Dataset>,
            frag: FileFragment,
        ) -> Result<RowAddrTreeMap> {
            // Load the row id sequence and deletion vector asynchronously,
            // then run the CPU-bound clone, deletion masking and tree
            // construction for each piece on the CPU pool, keeping the async
            // runtime free (the whole-mask fold previously ran this work in
            // one spawn_cpu block after all loads).
            let row_ids = load_row_id_sequence(&dataset, frag.metadata());
            let deletion_vector = frag.get_deletion_vector();
            let (row_ids, deletion_vector) = join!(row_ids, deletion_vector);
            let row_ids = row_ids?;
            let deletion_vector = deletion_vector?;
            spawn_cpu(move || match deletion_vector {
                Some(deletion_vector) => {
                    let mut row_ids = row_ids.as_ref().clone();
                    row_ids.mask(deletion_vector.to_sorted_iter())?;
                    Ok(RowAddrTreeMap::from(&row_ids))
                }
                None => Ok(RowAddrTreeMap::from(row_ids.as_ref())),
            })
            .await
        }

        let restrict_hash = restrict_to.as_ref().map(|b| {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            // RoaringBitmap doesn't implement Hash; serialize the sorted u32s.
            for v in b.iter() {
                v.hash(&mut h);
            }
            h.finish()
        });

        let dataset_clone = dataset.clone();
        let restrict_for_load = restrict_to.clone();
        let load_mask = move || async move {
            let fragments: Vec<FileFragment> = dataset_clone
                .get_fragments()
                .into_iter()
                .filter(|frag| {
                    restrict_for_load
                        .as_ref()
                        .map(|allow| allow.contains(frag.id() as u32))
                        .unwrap_or(true)
                })
                .collect();

            let pieces = stream::iter(fragments)
                .map(|frag| {
                    let dataset = dataset_clone.clone();
                    async move {
                        let meta = frag.metadata();
                        let row_id_meta = meta.row_id_meta.clone().ok_or_else(|| {
                            crate::Error::internal(format!(
                                "fragment {} is missing row id meta",
                                meta.id
                            ))
                        })?;
                        let lineage_file = meta.row_lineage_file(ROW_ID_FIELD_ID)?.cloned();
                        let piece_key = crate::session::caches::RowIdAllowListPieceKey {
                            fragment_id: meta.id,
                            row_id_meta,
                            lineage_file,
                            deletion_file: meta.deletion_file.clone(),
                        };
                        dataset
                            .metadata_cache
                            .as_ref()
                            .get_or_insert_with_key(piece_key, || {
                                build_allow_list_piece(dataset.clone(), frag)
                            })
                            .await
                    }
                })
                .buffer_unordered(dataset_clone.object_store.as_ref().io_parallelism())
                .try_collect::<Vec<_>>()
                .await?;

            // Merging the pieces is CPU-bound, so we spawn it on a blocking
            // thread.
            let allow_list = spawn_cpu(move || {
                let mut allow_list = RowAddrTreeMap::new();
                for piece in &pieces {
                    allow_list |= piece.as_ref();
                }
                Result::Ok(allow_list)
            })
            .await?;

            Ok(RowAddrMask::from_allowed(allow_list))
        };
        let Some(e_tag) = dataset.manifest_location().e_tag.as_deref() else {
            return load_mask().await.map(Arc::new);
        };
        let key = crate::session::caches::RowAddrMaskKey {
            version: dataset.manifest().version,
            e_tag: Some(e_tag),
            restrict_hash,
        };
        dataset
            .metadata_cache
            .as_ref()
            .get_or_insert_with_key(key, load_mask)
            .await
    }

    /// Sets the deleted fragment IDs to block during search.
    ///
    /// Used by FTS indices which track fragments that have been removed from the
    /// dataset but whose data is still present in the index (merge-on-read).
    pub fn set_deleted_fragments(&mut self, fragments: RoaringBitmap) {
        self.deleted_fragments = Some(fragments);
    }

    /// Block specific row addresses from index results because their index entries are stale
    /// due to a data overlay committed after the index was built.
    pub fn with_overlay_block(mut self, block: RowAddrMask) -> Self {
        self.overlay_block = Some(block);
        self
    }

    /// Creates a task to load a mask that filters out deleted rows and,
    /// when `restrict_to_fragments` is true, also restricts results to only
    /// the given `fragments`.
    ///
    /// The fragment restriction blocks stale index entries from fragments
    /// whose data changed but whose index was not rewritten. It should be
    /// enabled when `fragments` represents a real index fragment bitmap. It
    /// should be disabled when `fragments` is a conservative fallback (e.g.
    /// when the index has no fragment bitmap).
    ///
    /// The deletion mask is built as a block list (from deletion files) or
    /// an allow list (when stable row ids are in use and fragments have
    /// been removed).
    ///
    /// Returns `None` if it can be synchronously determined that no
    /// filtering is needed.
    pub fn create_deletion_mask(
        dataset: Arc<Dataset>,
        fragments: RoaringBitmap,
    ) -> Option<BoxFuture<'static, Result<Arc<RowAddrMask>>>> {
        Self::create_deletion_mask_impl(dataset, fragments, false)
    }

    /// Like [`Self::create_deletion_mask`] but also restricts results to the given
    /// `fragments`, blocking any row from a fragment not in the set.
    pub fn create_restricted_deletion_mask(
        dataset: Arc<Dataset>,
        fragments: RoaringBitmap,
    ) -> Option<BoxFuture<'static, Result<Arc<RowAddrMask>>>> {
        Self::create_deletion_mask_impl(dataset, fragments, true)
    }

    fn create_deletion_mask_impl(
        dataset: Arc<Dataset>,
        fragments: RoaringBitmap,
        restrict_to_fragments: bool,
    ) -> Option<BoxFuture<'static, Result<Arc<RowAddrMask>>>> {
        let mut missing_frags = Vec::new();
        let mut frags_with_deletion_files = Vec::new();
        let frag_map: HashMap<u32, &Fragment> = HashMap::from_iter(
            dataset
                .manifest
                .fragments
                .iter()
                .map(|frag| (frag.id as u32, frag)),
        );
        // A retired fragment can be absent from both the current manifest and
        // the index bitmap after coverage pruning. The no-mask fast path is
        // safe only if every allocated fragment is still present and covered.
        // Compare cardinalities instead of enumerating historical/reserved IDs.
        // Without a stored high water mark, retired trailing IDs are unknown.
        let needs_restriction = restrict_to_fragments
            && (dataset
                .manifest
                .max_fragment_id
                .is_none_or(|max_id| u64::from(max_id) + 1 != frag_map.len() as u64)
                || frag_map.keys().any(|id| !fragments.contains(*id)));
        for frag_id in fragments.iter() {
            let frag = frag_map.get(&frag_id);
            if let Some(frag) = frag {
                if frag.deletion_file.is_some() {
                    frags_with_deletion_files.push(frag_id);
                }
            } else {
                missing_frags.push(frag_id);
            }
        }
        if missing_frags.is_empty() && frags_with_deletion_files.is_empty() && !needs_restriction {
            None
        } else if dataset.manifest.uses_stable_row_ids() {
            let restrict_to = if restrict_to_fragments {
                Some(fragments)
            } else {
                None
            };
            Some(Self::do_create_deletion_mask_row_id(dataset.clone(), restrict_to).boxed())
        } else if restrict_to_fragments {
            Some(Self::do_create_restricted_deletion_mask(dataset, fragments).boxed())
        } else {
            Some(
                Self::do_create_deletion_mask(dataset, missing_frags, frags_with_deletion_files)
                    .boxed(),
            )
        }
    }
}

/// Whether `mask` is known to pass no row: an empty allow list. Block lists
/// never prove this, whatever they hold.
pub(crate) fn mask_selects_nothing(mask: &RowAddrMask) -> bool {
    matches!(mask, RowAddrMask::AllowList(rows) if rows.is_empty())
}

/// Whether a prefilter `mask` can select any row of one index segment.
///
/// `segment_rows`, when it is an allow list of the rows the segment may
/// answer for (the live rows of its covered fragments, in the mask's own
/// domain), decides alone. Otherwise, for address-domain masks, the
/// segment's fragment `coverage` decides by fragment key. With neither (a
/// row-id mask of a segment whose rows are unknown, or unknown coverage)
/// only a globally empty allow list proves emptiness. Emptiness is never
/// claimed where it cannot be proven, so a segment that is kept may still
/// turn out to hold no selectable row; one that is dropped never does.
pub fn prefilter_may_match_segment(
    mask: &RowAddrMask,
    segment_rows: Option<&RowAddrMask>,
    coverage: Option<&RoaringBitmap>,
    address_domain: bool,
) -> bool {
    if let Some(RowAddrMask::AllowList(rows)) = segment_rows {
        return mask.may_select_any(rows);
    }
    if address_domain && let Some(coverage) = coverage {
        return mask.may_select_fragments(coverage);
    }
    !matches!(mask, RowAddrMask::AllowList(rows) if rows.is_empty())
}

#[async_trait]
impl PreFilter for DatasetPreFilter {
    /// Waits for the prefilter to be fully loaded
    ///
    /// The prefilter loads in the background while the rest of the index
    /// search is running.  When you are ready to use the prefilter you
    /// must first call this method to ensure it is fully loaded.  This
    /// allows `filter_row_ids` to be a synchronous method.
    #[instrument(level = "debug", skip(self))]
    async fn wait_for_ready(&self) -> Result<()> {
        if let Some(filtered_ids) = &self.filtered_ids {
            filtered_ids.wait_ready().await?;
        }
        if let Some(deleted_ids) = &self.deleted_ids {
            deleted_ids.wait_ready().await?;
        }
        let final_mask = self.final_mask.lock().unwrap();
        final_mask.get_or_init(|| {
            let mut combined = RowAddrMask::default();
            if let Some(filtered_ids) = &self.filtered_ids {
                combined = combined & filtered_ids.get_ready().as_ref().clone();
            }
            if let Some(deleted_ids) = &self.deleted_ids {
                combined = combined & (*deleted_ids.get_ready()).clone();
            }
            if let Some(deleted) = &self.deleted_fragments {
                let mut block_list = RowAddrTreeMap::new();
                for frag_id in deleted.iter() {
                    block_list.insert_fragment(frag_id);
                }
                combined = combined & RowAddrMask::from_block(block_list);
            }
            if let Some(overlay_block) = &self.overlay_block {
                combined = combined & overlay_block.clone();
            }
            Arc::new(combined)
        });

        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.deleted_ids.is_none()
            && self.filtered_ids.is_none()
            && self.deleted_fragments.is_none()
            && self.overlay_block.is_none()
    }

    /// Get the row id mask for this prefilter
    fn mask(&self) -> Arc<RowAddrMask> {
        self.final_mask
            .lock()
            .unwrap()
            .get()
            .expect("mask called without call to wait_for_ready")
            .clone()
    }

    fn selects_nothing(&self) -> bool {
        self.final_mask
            .lock()
            .unwrap()
            .get()
            .is_some_and(|mask| mask_selects_nothing(mask))
    }

    /// Check whether a slice of row ids should be included in a query.
    ///
    /// Returns a vector of indices into the input slice that should be included,
    /// also known as a selection vector.
    ///
    /// This method must be called after `wait_for_ready`
    #[instrument(level = "debug", skip_all)]
    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        self.mask().selected_indices(row_ids)
    }
}

#[cfg(test)]
mod test {
    use lance_select::RowSetOps;

    /// Only an empty allow list selects nothing; a block list, even of
    /// every row the caller knows about, never proves it.
    #[test]
    fn selects_nothing_needs_an_empty_allow_list() {
        use super::mask_selects_nothing;
        use lance_select::{RowAddrMask, RowAddrTreeMap};
        let mut rows = RowAddrTreeMap::new();
        rows.insert_fragment(0);
        assert!(mask_selects_nothing(&RowAddrMask::allow_nothing()));
        assert!(!mask_selects_nothing(&RowAddrMask::from_allowed(
            rows.clone()
        )));
        assert!(!mask_selects_nothing(&RowAddrMask::from_block(rows)));
        assert!(!mask_selects_nothing(&RowAddrMask::all_rows()));
    }

    /// Emptiness is proven only from the mask and a known row set or, in
    /// the address domain, known coverage; unknown coverage, a row-id mask
    /// without a row set, or a block list keep the segment.
    #[test]
    fn segment_match_proof_is_conservative() {
        use super::prefilter_may_match_segment;
        use lance_select::{RowAddrMask, RowAddrTreeMap};
        use roaring::RoaringBitmap;
        let mut allow = RowAddrTreeMap::new();
        allow.insert((3u64 << 32) | 4);
        let allow = RowAddrMask::from_allowed(allow);
        let others = RoaringBitmap::from_iter([0u32, 1]);
        let here = RoaringBitmap::from_iter([1u32, 3]);
        assert!(!prefilter_may_match_segment(
            &allow,
            None,
            Some(&others),
            true
        ));
        assert!(prefilter_may_match_segment(&allow, None, Some(&here), true));
        assert!(prefilter_may_match_segment(&allow, None, None, true));
        assert!(prefilter_may_match_segment(
            &allow,
            None,
            Some(&others),
            false
        ));
        let mut rows = RowAddrTreeMap::new();
        rows.insert((3u64 << 32) | 9);
        let rows = RowAddrMask::from_allowed(rows);
        assert!(!prefilter_may_match_segment(
            &allow,
            Some(&rows),
            Some(&here),
            false
        ));
        let mut rows = RowAddrTreeMap::new();
        rows.insert((3u64 << 32) | 4);
        let rows = RowAddrMask::from_allowed(rows);
        assert!(prefilter_may_match_segment(
            &allow,
            Some(&rows),
            Some(&others),
            false
        ));
        let block = RowAddrMask::all_rows();
        assert!(prefilter_may_match_segment(
            &block,
            None,
            Some(&others),
            true
        ));
        assert!(!prefilter_may_match_segment(
            &RowAddrMask::allow_nothing(),
            None,
            None,
            false
        ));
    }
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32};
    use rstest::rstest;

    use lance_core::utils::tempfile::TempStrDir;
    use lance_io::object_store::WrappingObjectStore;
    use lance_io::utils::tracking_store::IOTracker;
    use lance_io::{assert_io_eq, assert_io_gt};
    use lance_table::format::RowIdMeta;

    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::deletions::INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY;
    use crate::dataset::rowids::{
        INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, SPILL_ROW_LINEAGE_CONFIG_KEY,
    };
    use crate::dataset::{WriteMode, WriteParams};
    use crate::session::Session;
    use crate::session::caches::{DeletionFileKey, RowIdAllowListPieceKey, RowIdSequenceKey};

    use super::*;

    struct TestDatasets {
        no_deletions: Arc<Dataset>,
        deletions_no_missing_frags: Arc<Dataset>,
        deletions_missing_frags: Arc<Dataset>,
        only_missing_frags: Arc<Dataset>,
    }

    async fn test_datasets(use_stable_row_id: bool) -> TestDatasets {
        let test_data = BatchGenerator::new()
            .col(Box::new(IncrementingInt32::new().named("x")))
            .batch(9);
        let mut dataset = Dataset::write(
            test_data,
            "memory://test",
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids: use_stable_row_id,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let no_deletions = Arc::new(dataset.clone());

        // This will add a deletion file.
        dataset.delete("x = 8").await.unwrap();
        let deletions_no_missing_frags = Arc::new(dataset.clone());

        dataset.delete("x >= 3 and x <= 5").await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        let deletions_missing_frags = Arc::new(dataset.clone());

        dataset.delete("x >= 3").await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 1);
        assert!(
            dataset.get_fragments()[0]
                .metadata()
                .deletion_file
                .is_none()
        );
        let only_missing_frags = Arc::new(dataset.clone());

        TestDatasets {
            no_deletions,
            deletions_no_missing_frags,
            deletions_missing_frags,
            only_missing_frags,
        }
    }

    #[tokio::test]
    async fn test_deletion_mask() {
        let datasets = test_datasets(false).await;

        // If there are no deletions, we should get None
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.no_deletions.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_none());

        // If there are deletions, we should get a mask
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.deletions_no_missing_frags.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        assert_eq!(mask.block_list().and_then(|x| x.len()), Some(1)); // There was just one row deleted.

        // If there are deletions and missing fragments, we should get a mask
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.deletions_missing_frags.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        let mut expected = RowAddrTreeMap::from_iter(vec![(2 << 32) + 2]);
        expected.insert_fragment(1);
        assert_eq!(mask.block_list(), Some(&expected));

        // If we don't pass the missing fragment id, we should get a smaller mask.
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.deletions_missing_frags.clone(),
            RoaringBitmap::from_iter(2..3),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        assert_eq!(mask.block_list().and_then(|x| x.len()), Some(1));

        // If there are only missing fragments, we should still get a mask
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.only_missing_frags.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        let mut expected = RowAddrTreeMap::new();
        expected.insert_fragment(1);
        expected.insert_fragment(2);
        assert_eq!(mask.block_list(), Some(&expected));
    }

    #[rstest]
    #[case::stored_high_water_mark(false)]
    #[case::computed_high_water_mark(true)]
    #[tokio::test]
    async fn test_legacy_index_mask_includes_max_fragment(#[case] unset_max_fragment_id: bool) {
        let datasets = test_datasets(false).await;
        let mut dataset = (*datasets.deletions_missing_frags).clone();
        if unset_max_fragment_id {
            Arc::make_mut(&mut dataset.manifest).max_fragment_id = None;
        }
        let index = IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            fields: Vec::new(),
            covering_fields: vec![],
            name: "legacy".to_string(),
            dataset_version: dataset.manifest.version,
            fragment_bitmap: None,
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        };
        let prefilter = DatasetPreFilter::new(Arc::new(dataset), &[index], None);

        prefilter.wait_for_ready().await.unwrap();

        let mut expected = RowAddrTreeMap::from_iter(vec![(2 << 32) + 2]);
        expected.insert_fragment(1);
        assert_eq!(prefilter.mask().block_list(), Some(&expected));
    }

    #[rstest]
    #[tokio::test]
    async fn test_restricted_deletion_mask_pruned_fragments(
        #[values(false, true)] stable_row_ids: bool,
        #[values(false, true)] partial_deletion: bool,
        #[values(false, true)] unset_high_water_mark: bool,
    ) {
        let datasets = test_datasets(stable_row_ids).await;
        assert!(
            DatasetPreFilter::create_restricted_deletion_mask(
                datasets.no_deletions.clone(),
                RoaringBitmap::from_iter(0..3),
            )
            .is_none()
        );
        let (dataset, coverage, expected) = if partial_deletion {
            (
                datasets.deletions_missing_frags,
                vec![0, 2],
                vec![0, 1, 2, 6, 7],
            )
        } else {
            (datasets.only_missing_frags, vec![0], vec![0, 1, 2])
        };
        let mut dataset = (*dataset).clone();
        if unset_high_water_mark {
            Arc::make_mut(&mut dataset.manifest).max_fragment_id = None;
        }
        // A staged segment can still contain entries from retired fragments
        // even after publication has pruned those fragments from its bitmap.
        let mask = DatasetPreFilter::create_restricted_deletion_mask(
            Arc::new(dataset),
            RoaringBitmap::from_iter(coverage),
        )
        .expect("retired fragments still need filtering")
        .await
        .unwrap();
        let selected: Vec<_> = (0..9)
            .filter(|id| {
                let row_id = if stable_row_ids {
                    *id as u64
                } else {
                    u64::from(lance_core::utils::address::RowAddress::new_from_parts(
                        id / 3,
                        id % 3,
                    ))
                };
                mask.selected(row_id)
            })
            .collect();
        assert_eq!(selected, expected);
    }

    #[rstest]
    #[case::filter(false)]
    #[case::overlay(true)]
    #[tokio::test]
    async fn test_restricted_mask_composes_with_block_filter(#[case] overlay: bool) {
        let datasets = test_datasets(false).await;
        // Only fragment 2 has a deletion; exclusions in fragments 0 and 1
        // must not expand their allow entries to the full u32 offset domain.
        let blocked = RowAddrTreeMap::from_iter([0_u64, 1_u64 << 32]);
        let block_mask = RowAddrMask::from_block(blocked);
        let filter = if overlay {
            None
        } else {
            let block_mask = block_mask.clone();
            Some(async move { Ok(Arc::new(block_mask)) }.boxed())
        };
        let mut prefilter = DatasetPreFilter::new_restricted_to_fragments(
            datasets.deletions_no_missing_frags,
            RoaringBitmap::from_iter(0..3),
            filter,
        );
        if overlay {
            prefilter = prefilter.with_overlay_block(block_mask);
        }
        prefilter.wait_for_ready().await.unwrap();
        let mask = prefilter.mask();
        let allowed = mask.allow_list().unwrap();
        let bytes = allowed.serialized_size();
        assert!(bytes < 1024, "nine-row prefilter expanded to {bytes} bytes");
        let expected = RowAddrTreeMap::from_iter([
            1,
            2,
            (1_u64 << 32) + 1,
            (1_u64 << 32) + 2,
            2_u64 << 32,
            (2_u64 << 32) + 1,
        ]);
        assert_eq!(allowed, &expected);
    }

    #[tokio::test]
    async fn test_deletion_mask_stable_row_id() {
        // Here, behavior is different.
        let datasets = test_datasets(true).await;

        // If there are no deletions, we should get None
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.no_deletions.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_none());

        // If there are deletions but no missing files, we should get a block list
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.deletions_no_missing_frags.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        let expected = RowAddrTreeMap::from_iter(0..8);
        assert_eq!(mask.allow_list(), Some(&expected)); // There was just one row deleted.

        // If there are deletions and missing fragments, we should get an allow list
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.deletions_missing_frags.clone(),
            RoaringBitmap::from_iter(0..2),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        assert_eq!(mask.allow_list().and_then(|x| x.len()), Some(5)); // There were five rows left over;

        // If there are only missing fragments, we should get an allow list
        let mask = DatasetPreFilter::create_deletion_mask(
            datasets.only_missing_frags.clone(),
            RoaringBitmap::from_iter(0..3),
        );
        assert!(mask.is_some());
        let mask = mask.unwrap().await.unwrap();
        assert_eq!(mask.allow_list().and_then(|x| x.len()), Some(3)); // There were three rows left over;
    }

    #[tokio::test]
    async fn test_deletion_mask_stable_row_ids_recreated_at_same_uri() {
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let tmp_path = &temp_dir;
        let session = Arc::new(Session::default());
        let write = |num_rows| {
            let test_data = BatchGenerator::new()
                .col(Box::new(IncrementingInt32::new().named("x")))
                .batch(num_rows);
            let params = WriteParams {
                enable_stable_row_ids: true,
                session: Some(session.clone()),
                ..Default::default()
            };
            async move {
                Dataset::write(test_data, tmp_path, Some(params))
                    .await
                    .unwrap()
            }
        };

        let mut dataset = write(10).await;
        dataset.delete("x = 9").await.unwrap();
        assert_eq!(dataset.manifest().version, 2);
        let mask = DatasetPreFilter::create_deletion_mask(
            Arc::new(dataset.clone()),
            RoaringBitmap::from_iter(0..1),
        )
        .expect("stable-row-id deletion mask")
        .await
        .unwrap();
        assert_eq!(mask.allow_list(), Some(&RowAddrTreeMap::from_iter(0..9)));

        drop(dataset);
        std::fs::remove_dir_all(tmp_path.as_str()).unwrap();

        let mut dataset = write(20).await;
        dataset.delete("x = 19").await.unwrap();
        assert_eq!(dataset.manifest().version, 2);
        let mask = DatasetPreFilter::create_deletion_mask(
            Arc::new(dataset),
            RoaringBitmap::from_iter(0..1),
        )
        .expect("stable-row-id deletion mask")
        .await
        .unwrap();
        assert_eq!(mask.allow_list(), Some(&RowAddrTreeMap::from_iter(0..19)));
    }

    #[tokio::test]
    async fn test_deletion_mask_without_generation_token_is_not_cached() {
        let datasets = test_datasets(true).await;
        let mut dataset = (*datasets.deletions_no_missing_frags).clone();
        dataset.manifest_location.e_tag = None;
        let dataset = Arc::new(dataset);

        let load_mask = || {
            let dataset = dataset.clone();
            async move {
                DatasetPreFilter::create_deletion_mask(dataset, RoaringBitmap::from_iter(0..3))
                    .expect("stable-row-id deletion mask")
                    .await
                    .unwrap()
            }
        };
        let first = load_mask().await;
        let second = load_mask().await;

        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(second.allow_list(), Some(&RowAddrTreeMap::from_iter(0..8)));
    }

    // Regression test for issue #6877.
    //
    // `create_restricted_deletion_mask` on a stable-row-id dataset must honor
    // the bitmap restriction by excluding stable row ids whose *current*
    // physical home is outside the bitmap. Without this, the merge_insert
    // UNION (indexed-scan ∪ unindexed-fragments scan) emits the same logical
    // row twice — once via the BTREE (which holds the row's stable_row_id)
    // and once via the unindexed scan.
    #[tokio::test]
    async fn test_restricted_deletion_mask_stable_row_id_honors_bitmap() {
        // Dataset with three fragments, 3 rows each, stable_row_ids = {0..9}.
        // Row x=8 is deleted, so the live stable_row_ids are {0..7}.
        let datasets = test_datasets(true).await;
        let ds = datasets.deletions_no_missing_frags.clone();

        // Full bitmap: allow-list covers all currently-live stable row ids.
        let mask = DatasetPreFilter::create_restricted_deletion_mask(
            ds.clone(),
            RoaringBitmap::from_iter(0..3),
        )
        .expect("full-bitmap mask present on stable-row-id dataset with deletions")
        .await
        .unwrap();
        let expected_all = RowAddrTreeMap::from_iter(0..8);
        assert_eq!(mask.allow_list(), Some(&expected_all));

        // Restricted to fragments {0, 1}: allow-list must exclude stable row
        // ids whose current home is in fragment 2 (rows 6, 7 — 8 was deleted).
        let mask = DatasetPreFilter::create_restricted_deletion_mask(
            ds.clone(),
            RoaringBitmap::from_iter(0..2),
        )
        .expect("restricted mask present")
        .await
        .unwrap();
        let expected_restricted = RowAddrTreeMap::from_iter(0..6);
        assert_eq!(mask.allow_list(), Some(&expected_restricted));

        // Restricted to empty bitmap: every BTREE-returned address is filtered
        // out. Empty allow-list is the correct semantic ("no row's current home
        // is in the restriction").
        let mask =
            DatasetPreFilter::create_restricted_deletion_mask(ds.clone(), RoaringBitmap::new())
                .expect("empty-restriction mask present")
                .await
                .unwrap();
        assert_eq!(mask.allow_list().and_then(|x| x.len()), Some(0));
    }

    // The allow list is assembled from per-fragment pieces cached by content
    // identity (row id generation + deletion file). A commit that deletes rows
    // must invalidate exactly the touched fragments' pieces: a mask rebuilt in
    // the same session must reflect the new deletions while untouched
    // fragments are served from cache.
    #[tokio::test]
    async fn test_row_id_allow_list_pieces_invalidate_on_new_deletions() {
        let test_data = BatchGenerator::new()
            .col(Box::new(IncrementingInt32::new().named("x")))
            .batch(9);
        let mut dataset = Dataset::write(
            test_data,
            "memory://test_pieces_invalidate",
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Three fragments of three rows each; stable row ids are 0..8, and
        // fragment k holds ids 3k..3k+2. Delete x=8 (fragment 2) and build
        // the allow list cold.
        dataset.delete("x = 8").await.unwrap();
        let ds = Arc::new(dataset.clone());
        let mask =
            DatasetPreFilter::create_deletion_mask(ds.clone(), RoaringBitmap::from_iter(0..3))
                .expect("mask present")
                .await
                .unwrap();
        assert_eq!(mask.allow_list().and_then(|x| x.len()), Some(8));

        // Delete x=2 (fragment 0) and rebuild with the same session. The
        // rebuilt mask must exclude the newly deleted row and keep everything
        // else, including the row ids whose home fragment was untouched by
        // this commit.
        dataset.delete("x = 2").await.unwrap();
        let ds = Arc::new(dataset);
        let mask =
            DatasetPreFilter::create_deletion_mask(ds.clone(), RoaringBitmap::from_iter(0..3))
                .expect("mask present")
                .await
                .unwrap();
        assert_eq!(mask.allow_list().and_then(|x| x.len()), Some(7));
        assert!(!mask.selected(2));
        assert!(mask.selected(0));
        assert!(mask.selected(3));
        assert!(mask.selected(7));

        // The restricted variant must honor its bitmap restriction against
        // the rebuilt pieces: fragments {1, 2} hold ids 3..8 minus the
        // deleted 8.
        let mask = DatasetPreFilter::create_restricted_deletion_mask(
            ds.clone(),
            RoaringBitmap::from_iter(1..3),
        )
        .expect("restricted mask present")
        .await
        .unwrap();
        let expected = RowAddrTreeMap::from_iter(3..8);
        assert_eq!(mask.allow_list(), Some(&expected));
    }

    #[tokio::test]
    async fn test_row_id_allow_list_pieces_reuse_column_lineage_and_inline_deletions() {
        let temp_dir = TempStrDir::default();
        let uri = temp_dir.as_str();
        let write_session = Arc::new(Session::default());
        let initial = BatchGenerator::new()
            .col(Box::new(IncrementingInt32::new().named("x")))
            .batch(6);
        let mut dataset = Dataset::write(
            initial,
            uri,
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids: true,
                session: Some(write_session.clone()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset
            .update_config([
                (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
                (INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY, "4096"),
            ])
            .await
            .unwrap();

        let appended = BatchGenerator::new()
            .col(Box::new(IncrementingInt32::new().start(6).named("x")))
            .batch(6);
        dataset = Dataset::write(
            appended,
            uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: 3,
                enable_stable_row_ids: true,
                session: Some(write_session),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.delete("x = 11").await.unwrap();
        dataset.delete("x = 2").await.unwrap();

        let fragments = dataset.get_fragments();
        assert!(matches!(
            &fragments[0].metadata().row_id_meta,
            Some(RowIdMeta::Inline(_))
        ));
        assert!(matches!(
            &fragments[2].metadata().row_id_meta,
            Some(RowIdMeta::Column)
        ));
        for fragment_id in [0_u64, 3] {
            let fragment = fragments
                .iter()
                .find(|fragment| fragment.id() == fragment_id)
                .unwrap();
            assert!(
                fragment
                    .metadata()
                    .deletion_file
                    .as_ref()
                    .is_some_and(|file| file.is_inline()),
                "fragment {fragment_id} should exercise inline deletion identity"
            );
        }

        let read_session = Arc::new(Session::default());
        let mut dataset = DatasetBuilder::from_uri(uri)
            .with_session(read_session.clone())
            .load()
            .await
            .unwrap();
        dataset.manifest_location.e_tag = Some("piece-cache-proof-v1".to_string());
        let tracker = Arc::new(IOTracker::default());
        let dataset = Arc::new(
            dataset.with_object_store_wrappers([tracker.clone() as Arc<dyn WrappingObjectStore>]),
        );

        let mask =
            DatasetPreFilter::create_deletion_mask(dataset.clone(), RoaringBitmap::from_iter(0..4))
                .expect("stable-row-id deletion mask")
                .await
                .unwrap();
        assert_eq!(mask.allow_list().and_then(|rows| rows.len()), Some(10));
        let cold_io = tracker.incremental_stats();
        assert_io_gt!(
            cold_io,
            read_iops,
            0,
            "spilled row ids must produce a real cold piece-cache miss"
        );

        let fragment = dataset
            .get_fragments()
            .into_iter()
            .find(|fragment| fragment.id() == 3)
            .unwrap();
        let metadata = fragment.metadata();
        let row_id_meta = metadata.row_id_meta.as_ref().unwrap();
        let lineage_file = metadata.row_lineage_file(ROW_ID_FIELD_ID).unwrap().unwrap();
        let deletion_file = metadata.deletion_file.as_ref().unwrap();
        assert!(
            dataset
                .metadata_cache
                .as_ref()
                .peek_resident_with_key(&RowIdSequenceKey {
                    fragment_id: 3,
                    row_id_meta,
                    lineage_file: Some(lineage_file),
                })
                .await
        );
        assert!(
            dataset
                .metadata_cache
                .as_ref()
                .peek_resident_with_key(&DeletionFileKey {
                    fragment_id: 3,
                    deletion_file,
                })
                .await
        );
        assert!(
            dataset
                .metadata_cache
                .as_ref()
                .peek_resident_with_key(&RowIdAllowListPieceKey {
                    fragment_id: 3,
                    row_id_meta: row_id_meta.clone(),
                    lineage_file: Some(lineage_file.clone()),
                    deletion_file: Some(deletion_file.clone()),
                })
                .await
        );

        let cache_before = read_session.metadata_cache_stats().await;
        let mask = DatasetPreFilter::create_restricted_deletion_mask(
            dataset.clone(),
            RoaringBitmap::from_iter(3..4),
        )
        .expect("restricted stable-row-id deletion mask")
        .await
        .unwrap();
        assert_eq!(mask.allow_list(), Some(&RowAddrTreeMap::from_iter(6..11)));
        let warm_io = tracker.incremental_stats();
        assert_io_eq!(
            warm_io,
            read_iops,
            0,
            "a warm piece should avoid row-id and deletion reads"
        );
        let cache_after = read_session.metadata_cache_stats().await;
        assert_eq!(
            cache_after.hits - cache_before.hits,
            1,
            "the restricted build should hit only the warm piece"
        );
        assert_eq!(
            cache_after.misses - cache_before.misses,
            1,
            "the new restriction should miss only its composed mask"
        );

        let mut updated = (*dataset).clone();
        updated.delete("x = 4").await.unwrap();
        updated.manifest_location.e_tag = Some("piece-cache-proof-v2".to_string());
        let updated = Arc::new(updated);
        tracker.incremental_stats();
        let cache_before = read_session.metadata_cache_stats().await;
        let mask = DatasetPreFilter::create_restricted_deletion_mask(
            updated.clone(),
            RoaringBitmap::from_iter(3..4),
        )
        .expect("restricted mask after a new commit")
        .await
        .unwrap();
        assert_eq!(mask.allow_list(), Some(&RowAddrTreeMap::from_iter(6..11)));
        let committed_io = tracker.incremental_stats();
        assert_io_eq!(
            committed_io,
            read_iops,
            0,
            "a new commit should leave the untouched piece warm"
        );
        let cache_after = read_session.metadata_cache_stats().await;
        assert_eq!(cache_after.hits - cache_before.hits, 1);
        assert_eq!(cache_after.misses - cache_before.misses, 1);

        let mask = DatasetPreFilter::create_deletion_mask(updated, RoaringBitmap::from_iter(0..4))
            .expect("full mask after a new commit")
            .await
            .unwrap();
        assert_eq!(mask.allow_list().and_then(|rows| rows.len()), Some(9));
        assert!(!mask.selected(4));
    }
}
