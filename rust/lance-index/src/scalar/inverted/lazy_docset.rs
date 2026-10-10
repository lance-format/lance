// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Deferred-load wrapper around [`DocSet`].
//!
//! The inverted-index `DocSet` holds the per-doc `row_id` and `num_tokens`
//! arrays for a partition. Eager loading on partition open pulls roughly
//! 12 bytes × num_docs per partition; across thousands of partitions on
//! cold object storage that's tens of GiB of IO before a query has even
//! checked whether a partition contains the term it's looking for.
//!
//! [`LazyDocSet`] defers the load. Cheap sync getters (`len`,
//! `total_tokens_cached`) work without IO; async getters fetch on
//! demand and cache. Wand scoring still needs per-doc num_tokens, but
//! only partitions that actually contribute hits pay
//! `ensure_num_tokens_loaded`/`ensure_loaded`.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{UInt32Type, UInt64Type};
use lance_core::ROW_ID;
use lance_core::Result;
use lance_core::cache::{CacheKey, LanceCache, WeakLanceCache};
use tokio::sync::OnceCell;

use crate::scalar::RowIdRemapper;
use crate::scalar::inverted::index::{DocSet, NUM_TOKEN_COL};
use crate::scalar::{IndexReader, IndexStore};
use lance_select::mask::RowAddrMask;

/// Lazy view over an inverted-index partition's `DocSet`.
///
/// Two variants:
/// - `Loaded`: a pre-materialized DocSet (legacy paths, tests).
///   Sync accessors return cached values; async accessors return
///   the same DocSet.
/// - `Deferred`: backed by an [`IndexReader`]; columns are read and
///   cached on first request.
pub enum LazyDocSet {
    Loaded(LoadedDocSet),
    Deferred(Box<DeferredDocSet>),
}

/// Pre-materialized DocSet view -- no reader, no IO.
pub struct LoadedDocSet {
    docs: Arc<DocSet>,
    num_rows: usize,
    total_tokens: u64,
}

/// Store-backed DocSet view that loads on demand and caches.
///
/// Holds the [`IndexStore`] and docs-file path rather than an open
/// [`IndexReader`], so a cached partition does not pin a docs-file
/// handle for its whole lifetime. The reader is re-opened on demand
/// inside each column accessor and dropped when that read completes;
/// because the resulting buffers use the session's bounded index cache,
/// a contributing partition re-opens only on a cache miss, and a
/// partition that never scores never opens the docs file at all after
/// construction.
pub struct DeferredDocSet {
    store: Arc<dyn IndexStore>,
    docs_path: String,
    is_legacy: bool,
    frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    /// Doc count cached at construction so `len()` stays sync + IO-free.
    num_rows: usize,
    /// `sum(num_tokens)` cached on first compute.
    total_tokens: OnceCell<u64>,
    // Do not retain loaded buffers here: the partition's cache weight is fixed
    // before these lazy reads, so retained buffers would escape its budget.
    cache: WeakLanceCache,
}

struct DocSetKey {
    with_row_ids: bool,
}

struct DocRowIdsKey;

// Keep cold candidate resolution bounded while retaining useful adjacent IDs
// for later queries. Each page costs 8 KiB before cache-entry overhead.
const ROW_IDS_PER_PAGE: usize = 1024;

struct DocRowIdsPageKey(usize);

impl CacheKey for DocRowIdsPageKey {
    type ValueType = Vec<u64>;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("docs-row-ids-page-{}", self.0).into()
    }

    fn type_name() -> &'static str {
        "FtsDocRowIdsPage"
    }
}

impl CacheKey for DocRowIdsKey {
    type ValueType = Vec<u64>;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        "docs-row-ids".into()
    }

    fn type_name() -> &'static str {
        "FtsDocRowIds"
    }
}

impl CacheKey for DocSetKey {
    type ValueType = DocSet;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        if self.with_row_ids {
            "docs-full".into()
        } else {
            "docs-tokens".into()
        }
    }

    fn type_name() -> &'static str {
        "DocSet"
    }
}

impl std::fmt::Debug for LazyDocSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loaded(l) => f
                .debug_struct("LazyDocSet::Loaded")
                .field("num_rows", &l.num_rows)
                .field("total_tokens", &l.total_tokens)
                .finish(),
            Self::Deferred(d) => f
                .debug_struct("LazyDocSet::Deferred")
                .field("num_rows", &d.num_rows)
                .field("total_tokens_loaded", &d.total_tokens.initialized())
                .finish(),
        }
    }
}

impl lance_core::deepsize::DeepSizeOf for LazyDocSet {
    fn deep_size_of_children(&self, ctx: &mut lance_core::deepsize::Context) -> usize {
        match self {
            Self::Loaded(l) => l.docs.deep_size_of_children(ctx),
            Self::Deferred(d) => std::mem::size_of::<DeferredDocSet>() + d.docs_path.capacity(),
        }
    }
}

impl LazyDocSet {
    pub fn new(
        store: Arc<dyn IndexStore>,
        docs_path: String,
        num_rows: usize,
        is_legacy: bool,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Self {
        Self::Deferred(Box::new(DeferredDocSet {
            store,
            docs_path,
            is_legacy,
            frag_reuse_index,
            num_rows,
            total_tokens: OnceCell::new(),
            cache: WeakLanceCache::from(cache),
        }))
    }

    /// Wrap an already-materialized [`DocSet`]. Used by legacy paths
    /// and tests that need to seed a partition without a reader.
    pub fn from_loaded(docs: DocSet) -> Self {
        let num_rows = docs.len();
        let total_tokens = docs.total_tokens_num();
        Self::Loaded(LoadedDocSet {
            docs: Arc::new(docs),
            num_rows,
            total_tokens,
        })
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Loaded(l) => l.num_rows,
            Self::Deferred(d) => d.num_rows,
        }
    }

    /// Sync read of cached `total_tokens`. Returns `None` for a
    /// `Deferred` LazyDocSet that hasn't yet had any of
    /// `total_tokens_num` / `ensure_num_tokens_loaded` / `ensure_loaded`
    /// run. Used by sync scoring code that has already paid for one
    /// of those async calls.
    pub fn total_tokens_cached(&self) -> Option<u64> {
        match self {
            Self::Loaded(l) => Some(l.total_tokens),
            Self::Deferred(d) => d.total_tokens.get().copied(),
        }
    }

    /// True if this DocSet carries a FragReuseIndex. Callers MUST
    /// avoid the deferred-row_id path when this is set: targeted
    /// row_id reads return raw stored ids, bypassing the per-id
    /// `remap_row_id` filter that `DocSet::from_columns` applies.
    pub fn has_frag_reuse_remap(&self) -> bool {
        match self {
            Self::Loaded(_) => false,
            Self::Deferred(d) => d.frag_reuse_index.is_some(),
        }
    }

    /// Sum of `num_tokens` across all docs.
    pub async fn total_tokens_num(&self) -> Result<u64> {
        match self {
            Self::Loaded(l) => Ok(l.total_tokens),
            Self::Deferred(d) => d.total_tokens_num().await,
        }
    }

    /// Materialize the full DocSet, including row_ids.
    pub async fn ensure_loaded(&self) -> Result<Arc<DocSet>> {
        match self {
            Self::Loaded(l) => Ok(l.docs.clone()),
            Self::Deferred(d) => d.ensure_loaded().await,
        }
    }

    /// Prewarm forward scoring lookups without the inverse row-to-doc mapping.
    /// Filtered searches can still obtain a complete DocSet on demand.
    pub async fn prewarm_scoring(&self) -> Result<()> {
        match self {
            Self::Loaded(_) => Ok(()),
            Self::Deferred(d) => d.prewarm_scoring().await,
        }
    }

    /// Materialize a DocSet that carries num_tokens but no row_ids.
    /// Used by the deferred-row_id scoring path; the per-partition
    /// caller resolves surviving doc_ids -> row_ids post-wand via
    /// [`Self::resolve_row_ids`]. This has a separate cache key from the full
    /// DocSet, so a later `ensure_loaded` still obtains the row ids.
    pub async fn ensure_num_tokens_loaded(&self) -> Result<Arc<DocSet>> {
        match self {
            Self::Loaded(l) => Ok(l.docs.clone()),
            Self::Deferred(d) => d.ensure_num_tokens_loaded().await,
        }
    }

    /// Pick the right DocSet shape for a wand walk under `mask`:
    /// the num_tokens-only deferred form when the mask is trivial
    /// AND no FragReuseIndex needs to filter row_ids; otherwise the
    /// full DocSet. Encapsulates the policy so callers don't have to
    /// rederive the conditions for the targeted-read fast path.
    pub async fn docs_for_wand(&self, mask: &RowAddrMask) -> Result<Arc<DocSet>> {
        if mask.is_select_all() && !self.has_frag_reuse_remap() {
            self.ensure_num_tokens_loaded().await
        } else {
            self.ensure_loaded().await
        }
    }

    /// Resolve a batch of `doc_id`s to their `row_id`s. Used by the
    /// deferred-row_id scoring path to map post-wand top-K candidates
    /// without going through a full DocSet build.
    ///
    /// Not safe with a FragReuseIndex (see
    /// [`Self::has_frag_reuse_remap`]): the targeted reads return
    /// raw stored ids without applying the remap/skip.
    pub async fn resolve_row_ids(&self, doc_ids: &[u32]) -> Result<Vec<u64>> {
        match self {
            Self::Loaded(l) => Ok(doc_ids.iter().map(|&d| l.docs.row_id(d)).collect()),
            Self::Deferred(d) => d.resolve_row_ids(doc_ids).await,
        }
    }
}

impl DeferredDocSet {
    /// Open a fresh docs-file reader. Dropped by the caller once its read
    /// completes, so no handle is pinned across the partition's lifetime.
    async fn reader(&self) -> Result<Arc<dyn IndexReader>> {
        self.store.open_index_file(&self.docs_path).await
    }

    async fn total_tokens_num(&self) -> Result<u64> {
        if let Some(v) = self.total_tokens.get() {
            return Ok(*v);
        }
        Ok(self.ensure_num_tokens_loaded().await?.total_tokens_num())
    }

    async fn ensure_loaded(&self) -> Result<Arc<DocSet>> {
        let docs = self
            .cache
            .get_or_insert_with_key(DocSetKey { with_row_ids: true }, || async {
                DocSet::load(
                    self.reader().await?,
                    self.is_legacy,
                    self.frag_reuse_index.clone(),
                )
                .await
            })
            .await?;
        let _ = self.total_tokens.set(docs.total_tokens_num());
        Ok(docs)
    }

    async fn prewarm_scoring(&self) -> Result<()> {
        if self.is_legacy || self.frag_reuse_index.is_some() {
            self.ensure_loaded().await?;
            return Ok(());
        }
        if self.ensure_num_tokens_loaded().await?.has_row_ids() {
            return Ok(());
        }
        self.cache
            .get_or_insert_with_key(DocRowIdsKey, || async {
                let batch = self
                    .reader()
                    .await?
                    .read_range(0..self.num_rows, Some(&[ROW_ID]))
                    .await?;
                Ok(batch[ROW_ID].as_primitive::<UInt64Type>().values().to_vec())
            })
            .await?;
        Ok(())
    }

    async fn ensure_num_tokens_loaded(&self) -> Result<Arc<DocSet>> {
        if let Some(full) = self
            .cache
            .get_with_key(&DocSetKey { with_row_ids: true })
            .await
        {
            let _ = self.total_tokens.set(full.total_tokens_num());
            return Ok(full);
        }
        let docs = self
            .cache
            .get_or_insert_with_key(
                DocSetKey {
                    with_row_ids: false,
                },
                || async {
                    let batch = self
                        .reader()
                        .await?
                        .read_range(0..self.num_rows, Some(&[NUM_TOKEN_COL]))
                        .await?;
                    Ok(DocSet::from_num_tokens_only(
                        batch[NUM_TOKEN_COL].as_primitive::<UInt32Type>(),
                    ))
                },
            )
            .await?;
        let _ = self.total_tokens.set(docs.total_tokens_num());
        Ok(docs)
    }

    async fn resolve_row_ids(&self, doc_ids: &[u32]) -> Result<Vec<u64>> {
        if let Some(full) = self
            .cache
            .get_with_key(&DocSetKey { with_row_ids: true })
            .await
            && full.has_row_ids()
        {
            return Ok(doc_ids.iter().map(|&d| full.row_id(d)).collect());
        }
        if let Some(row_ids) = self.cache.get_with_key(&DocRowIdsKey).await {
            return Ok(doc_ids.iter().map(|&d| row_ids[d as usize]).collect());
        }
        let mut pages = BTreeMap::<usize, Arc<Vec<u64>>>::new();
        let mut page_ids = doc_ids
            .iter()
            .map(|&doc| doc as usize / ROW_IDS_PER_PAGE)
            .collect::<Vec<_>>();
        page_ids.sort_unstable();
        page_ids.dedup();
        let mut missing = Vec::new();
        for page in page_ids {
            if let Some(row_ids) = self.cache.get_with_key(&DocRowIdsPageKey(page)).await {
                pages.insert(page, row_ids);
            } else {
                missing.push(page);
            }
        }
        if !missing.is_empty() {
            let ranges = missing
                .iter()
                .map(|&page| {
                    let start = page * ROW_IDS_PER_PAGE;
                    start..(start + ROW_IDS_PER_PAGE).min(self.num_rows)
                })
                .collect::<Vec<_>>();
            let batch = self
                .reader()
                .await?
                .read_ranges(&ranges, Some(&[ROW_ID]))
                .await?;
            let values = batch[ROW_ID].as_primitive::<UInt64Type>().values();
            let mut offset = 0;
            for (page, range) in missing.into_iter().zip(ranges) {
                let end = offset + range.len();
                let row_ids = Arc::new(values[offset..end].to_vec());
                self.cache
                    .insert_with_key(&DocRowIdsPageKey(page), row_ids.clone())
                    .await;
                pages.insert(page, row_ids);
                offset = end;
            }
        }
        Ok(doc_ids
            .iter()
            .map(|&doc| pages[&(doc as usize / ROW_IDS_PER_PAGE)][doc as usize % ROW_IDS_PER_PAGE])
            .collect())
    }
}
