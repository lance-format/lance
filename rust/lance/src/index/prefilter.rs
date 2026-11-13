// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Secondary Index pre-filter
//!
//! Based on the query, we might have information about which fragment ids and
//! row ids can be excluded from the search.

use std::borrow::Cow;
use std::cell::OnceCell;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use async_trait::async_trait;
use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::{BoxFuture, Shared};
use futures::stream;
use lance_core::utils::deletion::DeletionVector;
use lance_core::utils::tokio::spawn_cpu;
use lance_select::{PreparedRowAddrMask, RowAddrMask, RowAddrTreeMap, RowSetOps};
use lance_table::format::Fragment;
use lance_table::format::IndexMetadata;
use lance_table::rowids::RowIdSequence;
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

/// Query-owned completion and a lock-free result for synchronous scoring.
/// Neither the task nor its result retains the DatasetPreFilter owner.
pub(super) struct MembershipPreparation {
    result: Arc<OnceLock<Option<Arc<PreparedRowAddrMask>>>>,
    completion: Option<Shared<BoxFuture<'static, ()>>>,
}

impl MembershipPreparation {
    fn start(proof: Option<impl FnOnce() -> Option<PreparedRowAddrMask> + Send + 'static>) -> Self {
        let result = Arc::new(OnceLock::new());
        let Some(proof) = proof else {
            let _ = result.set(None);
            return Self {
                result,
                completion: None,
            };
        };
        let destination = result.clone();
        let task = tokio::spawn(async move {
            let prepared = spawn_cpu(move || Ok::<_, crate::Error>(proof())).await?;
            let _ = destination.set(prepared.map(Arc::new));
            Ok::<_, crate::Error>(())
        });
        let destination = result.clone();
        let completion = async move {
            match task.await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => tracing::warn!(%error, "optional membership preparation failed"),
                Err(error) => tracing::warn!(%error, "optional membership preparation task failed"),
            }
            // Publish fallback before notifying any waiter, including on panic.
            if destination.get().is_none() {
                let _ = destination.set(None);
            }
        }
        .boxed()
        .shared();
        Self {
            result,
            completion: Some(completion),
        }
    }

    async fn wait_ready(&self) {
        if let Some(completion) = &self.completion {
            completion.clone().await;
        }
    }
}

/// Prepares the exact interval view of `inner`'s final mask once per query.
///
/// When `inner` is the query's dataset filter itself, the preparation lives in
/// that filter's query-level cell and is shared by every index segment that
/// searches with it. A mask-equivalent wrapper with a final mask of its own,
/// such as a segment ownership filter, keeps a cell of its own; the parallel
/// probes of that segment share the wrapper and so share its preparation.
struct QueryMembershipPreFilter {
    inner: Arc<dyn PreFilter>,
    dataset: Option<Arc<DatasetPreFilter>>,
    own: tokio::sync::OnceCell<MembershipPreparation>,
    partition_row_counts: Box<dyn Fn() -> Vec<usize> + Send + Sync>,
}

impl QueryMembershipPreFilter {
    fn preparation(&self) -> &tokio::sync::OnceCell<MembershipPreparation> {
        self.dataset
            .as_ref()
            .map_or(&self.own, |dataset| &dataset.query_membership)
    }
}

#[async_trait]
impl PreFilter for QueryMembershipPreFilter {
    async fn wait_for_ready(&self) -> Result<()> {
        self.inner.wait_for_ready().await?;
        let preparation = self
            .preparation()
            .get_or_init(|| async {
                // No await between starting and publishing the shared task: a
                // cancelled first waiter must not start another CPU proof.
                MembershipPreparation::start(PreparedRowAddrMask::prepare(
                    self.inner.mask(),
                    std::iter::once_with(|| (self.partition_row_counts)()).flatten(),
                ))
            })
            .await;
        preparation.wait_ready().await;
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn selects_nothing(&self) -> bool {
        self.inner.selects_nothing()
    }

    fn needs_partition_row_ids(&self) -> bool {
        self.inner.needs_partition_row_ids()
    }

    fn is_empty_for(&self, rows: &RowAddrTreeMap) -> bool {
        self.inner.is_empty_for(rows)
    }

    fn mask(&self) -> Arc<RowAddrMask> {
        self.inner.mask()
    }

    fn prepared_mask(&self) -> Option<Arc<PreparedRowAddrMask>> {
        self.preparation()
            .get()
            .and_then(|preparation| preparation.result.get())
            .and_then(Clone::clone)
    }

    fn is_mask_equivalent(&self) -> bool {
        self.inner.is_mask_equivalent()
    }

    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        self.inner.filter_row_ids(row_ids)
    }
}

/// Hash of a fragment set for mask cache keys, so masks restricted to different
/// fragment sets do not share an entry. `RoaringBitmap` doesn't implement `Hash`,
/// so the sorted ids are hashed.
fn fragment_set_hash(fragments: &RoaringBitmap) -> u64 {
    let mut hasher = DefaultHasher::new();
    for fragment_id in fragments.iter() {
        fragment_id.hash(&mut hasher);
    }
    hasher.finish()
}

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
    // The first guaranteed-work budget, including a rejected None, is query-local.
    // A later delta retains fallback rather than repeating the proof.
    pub(super) query_membership: tokio::sync::OnceCell<MembershipPreparation>,
}

impl DatasetPreFilter {
    /// Attach lazy, query-local preparation only to this exact final-mask owner.
    /// Partition loading still precedes the existing async ready boundary.
    pub(crate) fn with_query_membership(
        self: &Arc<Self>,
        candidate: Arc<dyn PreFilter>,
        partition_row_counts: impl Fn() -> Vec<usize> + Send + Sync + 'static,
    ) -> Arc<dyn PreFilter> {
        let base: Arc<dyn PreFilter> = self.clone();
        if candidate.is_empty()
            || !candidate.is_mask_equivalent()
            || !Arc::ptr_eq(&base, &candidate)
            || self
                .query_membership
                .get()
                .is_some_and(|state| state.result.get().is_some())
        {
            return candidate;
        }
        Arc::new(QueryMembershipPreFilter {
            inner: candidate,
            dataset: Some(self.clone()),
            own: tokio::sync::OnceCell::new(),
            partition_row_counts: Box::new(partition_row_counts),
        })
    }

    /// Attach the same lazy preparation to a mask-equivalent restriction of
    /// this query's filter whose final mask differs from it, such as a segment
    /// ownership filter. The caller must have derived `restricted` from this
    /// query's filter. The wrapper prepares its own final mask once; the
    /// parallel probes of that segment share it.
    pub(crate) fn with_restricted_membership(
        restricted: Arc<dyn PreFilter>,
        partition_row_counts: impl Fn() -> Vec<usize> + Send + Sync + 'static,
    ) -> Arc<dyn PreFilter> {
        if restricted.is_empty() || !restricted.is_mask_equivalent() {
            return restricted;
        }
        Arc::new(QueryMembershipPreFilter {
            inner: restricted,
            dataset: None,
            own: tokio::sync::OnceCell::new(),
            partition_row_counts: Box::new(partition_row_counts),
        })
    }

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
            query_membership: tokio::sync::OnceCell::new(),
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
            query_membership: tokio::sync::OnceCell::new(),
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
    pub(crate) async fn do_create_deletion_mask_row_id(
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
        async fn load_row_ids_and_deletions(
            dataset: &Dataset,
            restrict_to: Option<&RoaringBitmap>,
        ) -> Result<Vec<(Arc<RowIdSequence>, Option<Arc<DeletionVector>>)>> {
            let frags: Vec<_> = dataset
                .get_fragments()
                .into_iter()
                .filter(|f| {
                    restrict_to
                        .map(|allow| allow.contains(f.id() as u32))
                        .unwrap_or(true)
                })
                .collect();
            stream::iter(frags)
                .map(|frag| async move {
                    let row_ids = load_row_id_sequence(dataset, frag.metadata());
                    let deletion_vector = frag.get_deletion_vector();
                    let (row_ids, deletion_vector) = join!(row_ids, deletion_vector);
                    Ok::<_, crate::Error>((row_ids?, deletion_vector?))
                })
                .buffer_unordered(dataset.object_store.as_ref().io_parallelism())
                .try_collect::<Vec<_>>()
                .await
        }

        let restrict_hash = restrict_to.as_ref().map(fragment_set_hash);

        let dataset_clone = dataset.clone();
        let restrict_for_load = restrict_to.clone();
        let load_mask = move || async move {
            let row_ids_and_deletions =
                load_row_ids_and_deletions(&dataset_clone, restrict_for_load.as_ref()).await?;

            // The process of computing the final mask is CPU-bound, so we spawn it
            // on a blocking thread.
            let allow_list = spawn_cpu(move || {
                Result::Ok(row_ids_and_deletions.into_iter().fold(
                    RowAddrTreeMap::new(),
                    |mut allow_list, (row_ids, deletion_vector)| {
                        let seq = if let Some(deletion_vector) = deletion_vector {
                            let mut row_ids = row_ids.as_ref().clone();
                            row_ids.mask(deletion_vector.to_sorted_iter()).unwrap();
                            Cow::<RowIdSequence>::Owned(row_ids)
                        } else {
                            Cow::<RowIdSequence>::Borrowed(row_ids.as_ref())
                        };
                        let treemap = RowAddrTreeMap::from(seq.as_ref());
                        allow_list |= treemap;
                        allow_list
                    },
                ))
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

    /// True when the only rows this prefilter removes are deleted rows and rows
    /// outside the indexed fragments: no query filter, no FTS fragment pruning and
    /// no overlay block. `self` is destructured so that a new filter field fails to
    /// compile here until it is accounted for.
    pub(crate) fn only_deletions(&self) -> bool {
        let Self {
            deleted_ids: _,
            filtered_ids,
            deleted_fragments,
            overlay_block,
            final_mask: _,
            query_membership: _,
        } = self;
        filtered_ids.is_none() && deleted_fragments.is_none() && overlay_block.is_none()
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
        } else {
            // Deletion files of a version are immutable, so the mask is built once per
            // (version, fragment set) and shared by every later query of that version,
            // as the row-id path above does. Without an e-tag a recreated dataset could
            // reuse another generation's entry, so such masks are not cached.
            let Some(e_tag) = dataset.manifest_location().e_tag.clone() else {
                return Some(Self::load_address_mask(
                    dataset,
                    fragments,
                    restrict_to_fragments,
                    missing_frags,
                    frags_with_deletion_files,
                ));
            };
            Some(
                async move {
                    let key = crate::session::caches::RowAddrMaskKey {
                        version: dataset.manifest().version,
                        e_tag: Some(e_tag.as_str()),
                        // Bit 0 separates the restricted allow list from the unrestricted
                        // block list over the same fragments. A version either uses stable
                        // row ids or not, so these never meet the row-id path's entries.
                        restrict_hash: Some(
                            (fragment_set_hash(&fragments) << 1) | restrict_to_fragments as u64,
                        ),
                    };
                    let loader_dataset = dataset.clone();
                    dataset
                        .metadata_cache
                        .as_ref()
                        .get_or_insert_with_key(key, move || async move {
                            let mask = Self::load_address_mask(
                                loader_dataset,
                                fragments,
                                restrict_to_fragments,
                                missing_frags,
                                frags_with_deletion_files,
                            )
                            .await?;
                            Ok(Arc::try_unwrap(mask).unwrap_or_else(|mask| (*mask).clone()))
                        })
                        .await
                }
                .boxed(),
            )
        }
    }

    fn load_address_mask(
        dataset: Arc<Dataset>,
        fragments: RoaringBitmap,
        restrict_to_fragments: bool,
        missing_frags: Vec<u32>,
        frags_with_deletion_files: Vec<u32>,
    ) -> BoxFuture<'static, Result<Arc<RowAddrMask>>> {
        if restrict_to_fragments {
            Self::do_create_restricted_deletion_mask(dataset, fragments).boxed()
        } else {
            Self::do_create_deletion_mask(dataset, missing_frags, frags_with_deletion_files).boxed()
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
            // Deletions only: `RowAddrMask::default()` selects every row, so the
            // combined mask is the deletion mask itself. Share it instead of copying.
            if self.only_deletions()
                && let Some(deleted_ids) = &self.deleted_ids
            {
                return deleted_ids.get_ready();
            }
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

    fn is_mask_equivalent(&self) -> bool {
        true
    }

    fn prepared_mask(&self) -> Option<Arc<PreparedRowAddrMask>> {
        self.query_membership
            .get()
            .and_then(|preparation| preparation.result.get())
            .and_then(Clone::clone)
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
    use lance_core::utils::address::RowAddress;
    use lance_select::RowSetOps;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Only an empty allow list selects nothing; a block list, even of
    /// every row the caller knows about, never proves it.
    fn query_mask() -> RowAddrMask {
        RowAddrMask::from_allowed(
            (0..32)
                .flat_map(|key| {
                    (0..32).map(move |offset| u64::from(RowAddress::new_from_parts(key, offset)))
                })
                .collect(),
        )
    }

    fn query_filter(mask: Option<RowAddrMask>) -> Arc<DatasetPreFilter> {
        Arc::new(DatasetPreFilter {
            deleted_ids: None,
            filtered_ids: mask
                .map(|mask| SharedPrerequisite::spawn(async move { Ok(Arc::new(mask)) })),
            deleted_fragments: None,
            overlay_block: None,
            final_mask: Mutex::new(OnceCell::new()),
            query_membership: tokio::sync::OnceCell::new(),
        })
    }

    #[tokio::test]
    async fn query_membership_concurrent_waiters_share_one_proof() {
        let base = query_filter(Some(query_mask()));
        let calls = Arc::new(AtomicUsize::new(0));
        let count_calls = calls.clone();
        let filter = base.with_query_membership(base.clone(), move || {
            count_calls.fetch_add(1, Ordering::SeqCst);
            vec![4096]
        });
        futures::future::join_all((0..32).map(|_| filter.wait_for_ready()))
            .await
            .into_iter()
            .for_each(|result| result.unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let prepared = filter.prepared_mask().unwrap();
        assert!(base.query_membership.get().unwrap().completion.is_some());
        for _ in 0..32 {
            assert!(Arc::ptr_eq(&prepared, &filter.prepared_mask().unwrap()));
        }
        let rows = [0, 1, 31, 32, u64::from(RowAddress::new_from_parts(31, 31))];
        assert_eq!(
            prepared.selected_indices(rows.iter()),
            filter.filter_row_ids(Box::new(rows.iter()))
        );
        let owner = Arc::downgrade(&base);
        drop(filter);
        drop(base);
        assert!(
            owner.upgrade().is_none(),
            "the query task must not retain its Dataset owner"
        );
    }

    #[tokio::test]
    async fn query_membership_cancelled_first_waiter_keeps_shared_task() {
        let base = query_filter(Some(query_mask()));
        base.wait_for_ready().await.unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let count_calls = calls.clone();
        let filter = base.with_query_membership(base.clone(), move || {
            count_calls.fetch_add(1, Ordering::SeqCst);
            vec![4096]
        });
        {
            let wait = filter.wait_for_ready();
            futures::pin_mut!(wait);
            assert!(futures::poll!(wait.as_mut()).is_pending());
        }
        tokio::time::timeout(Duration::from_secs(2), filter.wait_for_ready())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(filter.prepared_mask().is_some());
    }

    #[tokio::test]
    async fn query_membership_failed_task_publishes_fallback() {
        let preparation = MembershipPreparation::start(Some(|| -> Option<PreparedRowAddrMask> {
            panic!("injected optional proof failure")
        }));
        tokio::time::timeout(
            Duration::from_secs(2),
            futures::future::join_all((0..16).map(|_| preparation.wait_ready())),
        )
        .await
        .unwrap();
        assert!(preparation.result.get().unwrap().is_none());
    }

    #[tokio::test]
    async fn query_membership_low_budget_is_sticky_without_dispatch() {
        let base = query_filter(Some(query_mask()));
        let filter = base.with_query_membership(base.clone(), || vec![1]);
        filter.wait_for_ready().await.unwrap();
        assert!(filter.prepared_mask().is_none());
        let state = base.query_membership.get().unwrap();
        assert!(state.completion.is_none());
        let result = state.result.clone();
        let later = base.with_query_membership(base.clone(), || {
            panic!("a later budget must not be consumed")
        });
        later.wait_for_ready().await.unwrap();
        assert!(Arc::ptr_eq(
            &result,
            &base.query_membership.get().unwrap().result
        ));
        assert!(later.prepared_mask().is_none());
    }

    #[rstest]
    #[case::all(RowAddrMask::all_rows())]
    #[case::empty(RowAddrMask::allow_nothing())]
    #[case::block(RowAddrMask::from_block([0].into_iter().collect()))]
    #[tokio::test]
    async fn query_membership_unsupported_shape_does_not_consume_counts(#[case] mask: RowAddrMask) {
        let base = query_filter(Some(mask));
        let filter = base.with_query_membership(base.clone(), || {
            panic!("unsupported shapes must reject before metadata")
        });
        filter.wait_for_ready().await.unwrap();
        assert!(filter.prepared_mask().is_none());
        assert!(base.query_membership.get().unwrap().completion.is_none());
        assert_eq!(filter.selects_nothing(), base.selects_nothing());
    }

    #[tokio::test]
    async fn query_membership_holes_keep_exact_fallback() {
        let excluded = u64::from(RowAddress::new_from_parts(3, 7));
        let mask = query_mask().also_block([excluded].into_iter().collect());
        let base = query_filter(Some(mask));
        let filter = base.with_query_membership(base.clone(), || vec![4096]);
        filter.wait_for_ready().await.unwrap();
        assert!(filter.prepared_mask().is_none());
        assert!(base.query_membership.get().unwrap().completion.is_some());
        let rows = [excluded - 1, excluded, excluded + 1];
        assert_eq!(filter.filter_row_ids(Box::new(rows.iter())), [0, 2]);
    }

    #[tokio::test]
    async fn query_membership_no_filter_or_different_owner_retains_arc() {
        let no_filter = query_filter(None);
        let original: Arc<dyn PreFilter> = no_filter.clone();
        let returned =
            no_filter.with_query_membership(original.clone(), || panic!("no-filter metadata"));
        assert!(Arc::ptr_eq(&original, &returned));
        assert!(no_filter.query_membership.get().is_none());
        let base = query_filter(Some(query_mask()));
        let other: Arc<dyn PreFilter> = query_filter(Some(query_mask()));
        let returned =
            base.with_query_membership(other.clone(), || panic!("different-owner metadata"));
        assert!(Arc::ptr_eq(&other, &returned));
        assert!(base.query_membership.get().is_none());
    }

    struct AdditionalHook;

    #[async_trait]
    impl PreFilter for AdditionalHook {
        async fn wait_for_ready(&self) -> Result<()> {
            Ok(())
        }
        fn is_empty(&self) -> bool {
            false
        }
        fn mask(&self) -> Arc<RowAddrMask> {
            Arc::new(RowAddrMask::all_rows())
        }
        fn filter_row_ids<'a>(&self, rows: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
            rows.enumerate()
                .filter_map(|(position, row)| (row % 2 == 0).then_some(position as u64))
                .collect()
        }
    }

    #[tokio::test]
    async fn query_membership_custom_hook_retains_original_arc() {
        let base = query_filter(Some(query_mask()));
        let custom: Arc<dyn PreFilter> = Arc::new(AdditionalHook);
        let returned = base.with_query_membership(custom.clone(), || panic!("custom metadata"));
        assert!(Arc::ptr_eq(&custom, &returned));
        returned.wait_for_ready().await.unwrap();
        let rows = [2, 3, 4];
        assert_eq!(returned.filter_row_ids(Box::new(rows.iter())), [0, 2]);
        assert!(base.query_membership.get().is_none());
    }

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

    use crate::dataset::WriteParams;
    use crate::session::Session;

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

    #[rstest]
    #[tokio::test]
    async fn test_deletion_mask_recreated_at_same_uri(
        #[values(false, true)] use_stable_row_id: bool,
    ) {
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let tmp_path = &temp_dir;
        let session = Arc::new(Session::default());
        let write = |num_rows| {
            let test_data = BatchGenerator::new()
                .col(Box::new(IncrementingInt32::new().named("x")))
                .batch(num_rows);
            let params = WriteParams {
                enable_stable_row_ids: use_stable_row_id,
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
        .expect("deletion mask")
        .await
        .unwrap();
        if use_stable_row_id {
            assert_eq!(mask.allow_list(), Some(&RowAddrTreeMap::from_iter(0..9)));
        } else {
            assert_eq!(mask.block_list(), Some(&RowAddrTreeMap::from_iter([9])));
        }

        drop(dataset);
        std::fs::remove_dir_all(tmp_path.as_str()).unwrap();

        let mut dataset = write(20).await;
        dataset.delete("x = 19").await.unwrap();
        assert_eq!(dataset.manifest().version, 2);
        let mask = DatasetPreFilter::create_deletion_mask(
            Arc::new(dataset),
            RoaringBitmap::from_iter(0..1),
        )
        .expect("deletion mask")
        .await
        .unwrap();
        if use_stable_row_id {
            assert_eq!(mask.allow_list(), Some(&RowAddrTreeMap::from_iter(0..19)));
        } else {
            assert_eq!(mask.block_list(), Some(&RowAddrTreeMap::from_iter([19])));
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_deletion_mask_without_generation_token_is_not_cached(
        #[values(false, true)] use_stable_row_id: bool,
    ) {
        let datasets = test_datasets(use_stable_row_id).await;
        let mut dataset = (*datasets.deletions_no_missing_frags).clone();
        dataset.manifest_location.e_tag = None;
        let dataset = Arc::new(dataset);

        let load_mask = || {
            let dataset = dataset.clone();
            async move {
                DatasetPreFilter::create_deletion_mask(dataset, RoaringBitmap::from_iter(0..3))
                    .expect("deletion mask")
                    .await
                    .unwrap()
            }
        };
        let first = load_mask().await;
        let second = load_mask().await;

        assert!(!Arc::ptr_eq(&first, &second));
        if use_stable_row_id {
            assert_eq!(second.allow_list(), Some(&RowAddrTreeMap::from_iter(0..8)));
        } else {
            assert_eq!(
                second.block_list(),
                Some(&RowAddrTreeMap::from_iter([(2 << 32) + 2]))
            );
        }
    }

    #[tokio::test]
    async fn test_address_deletion_mask_cached_per_version() {
        let datasets = test_datasets(false).await;
        let dataset = datasets.deletions_no_missing_frags.clone();
        assert!(dataset.manifest_location().e_tag.is_some());

        // Fragment 0 is outside this set. An unrestricted mask may select its
        // rows; a restricted mask must not.
        let covered = RoaringBitmap::from_iter(1..3);
        let mut cached = Vec::new();
        for restrict_to_fragments in [false, true] {
            let load_mask = || {
                DatasetPreFilter::create_deletion_mask_impl(
                    dataset.clone(),
                    covered.clone(),
                    restrict_to_fragments,
                )
                .expect("deletion mask")
            };
            let first = load_mask().await.unwrap();
            let second = load_mask().await.unwrap();
            assert!(Arc::ptr_eq(&first, &second));
            assert_eq!(
                first.selected(0),
                !restrict_to_fragments,
                "restricted={restrict_to_fragments} shared the other mode's mask"
            );
            assert!(!first.selected((2 << 32) + 2));
            cached.push(first);
        }
        assert!(!Arc::ptr_eq(&cached[0], &cached[1]));

        // The same restriction over every fragment is another entry, and both
        // modes still block the one deleted row.
        let all_fragments = RoaringBitmap::from_iter(0..3);
        let full_unrestricted =
            DatasetPreFilter::create_deletion_mask(dataset.clone(), all_fragments.clone())
                .expect("deletion mask")
                .await
                .unwrap();
        let full_restricted =
            DatasetPreFilter::create_restricted_deletion_mask(dataset.clone(), all_fragments)
                .expect("deletion mask")
                .await
                .unwrap();
        assert!(!Arc::ptr_eq(&full_unrestricted, &cached[0]));
        assert!(!Arc::ptr_eq(&full_restricted, &cached[1]));
        assert!(full_unrestricted.selected(0));
        assert!(full_restricted.selected(0));
        assert!(full_unrestricted.selected((2 << 32) + 1));
        assert!(full_restricted.selected((2 << 32) + 1));
        assert!(!full_unrestricted.selected((2 << 32) + 2));
        assert!(!full_restricted.selected((2 << 32) + 2));

        // A recreated dataset restarts its version history, so an entry from
        // the same version under a different e-tag must not be reused.
        let mut other_generation = (*dataset).clone();
        other_generation.manifest_location.e_tag = Some("second-generation".to_string());
        let other_generation_mask =
            DatasetPreFilter::create_deletion_mask_impl(Arc::new(other_generation), covered, false)
                .expect("deletion mask")
                .await
                .unwrap();
        assert!(!Arc::ptr_eq(&other_generation_mask, &cached[0]));
        assert!(other_generation_mask.selected(0));

        // A newer version, which removed fragment 1, does not reuse the old entry.
        let newer = DatasetPreFilter::create_deletion_mask(
            datasets.deletions_missing_frags.clone(),
            RoaringBitmap::from_iter(0..3),
        )
        .expect("deletion mask")
        .await
        .unwrap();
        assert!(!newer.selected(1 << 32));
    }

    #[tokio::test]
    async fn test_deletions_only_prefilter_shares_mask() {
        let datasets = test_datasets(false).await;
        let prefilter = DatasetPreFilter::new_restricted_to_fragments(
            datasets.deletions_no_missing_frags.clone(),
            RoaringBitmap::from_iter(0..3),
            None,
        );
        assert!(prefilter.only_deletions());
        prefilter.wait_for_ready().await.unwrap();
        let deleted = prefilter.deleted_ids.as_ref().unwrap().get_ready();
        assert!(Arc::ptr_eq(&prefilter.mask(), &deleted));
        assert!(!prefilter.mask().selected((2 << 32) + 2));
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
}
