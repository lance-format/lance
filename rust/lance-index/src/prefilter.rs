// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use async_trait::async_trait;
use lance_core::Result;
use lance_select::{PreparedRowAddrMask, RowAddrMask, RowAddrTreeMap};

/// A trait to be implemented by anything supplying a prefilter row addr mask
///
/// This trait is for internal use only and has no stability guarantees.
#[async_trait]
pub trait FilterLoader: Send + 'static {
    async fn load(self: Box<Self>) -> Result<RowAddrMask>;
}

/// Filter out row ids that we know are not relevant to the query.
///
/// This could be both rows that are deleted or a prefilter
/// that should be applied to the search
///
/// <section class="warning">
/// Internal use only. No API stability guarantees.
/// </section>
#[async_trait]
pub trait PreFilter: Send + Sync {
    /// Waits for the prefilter to be fully loaded
    ///
    /// The prefilter loads in the background while the rest of the index
    /// search is running.  When you are ready to use the prefilter you
    /// must first call this method to ensure it is fully loaded.  This
    /// allows `filter_row_ids` to be a synchronous method.
    async fn wait_for_ready(&self) -> Result<()>;

    /// If the filter is empty.
    fn is_empty(&self) -> bool;

    /// Whether partition-local row coverage is needed to determine if this filter
    /// can be replaced by [`NoFilter`].
    ///
    /// Most filters cannot make this proof and must not make an IVF search
    /// enumerate a partition's row addresses just to call [`Self::is_empty_for`].
    fn needs_partition_row_ids(&self) -> bool {
        false
    }

    /// Whether this filter selects every row in a known partition.
    ///
    /// Callers must first call [`Self::wait_for_ready`]. Implementations that
    /// cannot prove partition-local emptiness fall back to the global answer.
    fn is_empty_for(&self, _rows: &RowAddrTreeMap) -> bool {
        self.is_empty()
    }

    /// Get the row addr mask for this prefilter
    ///
    /// This method must be called after `wait_for_ready`
    fn mask(&self) -> Arc<RowAddrMask>;

    /// Whether the loaded filter proves that no row at all can pass it.
    ///
    /// This is the opposite end of the scale from [`Self::is_empty`], which
    /// says the filter restricts nothing. It is true only once the mask is
    /// known to be an empty allow list, so it must be read after
    /// `wait_for_ready`; before that, and for a block list of any size, it
    /// is false. Index searches use it to skip candidate scoring for a
    /// segment whose own mask selects nothing, so implementations answer
    /// from their stored mask without scanning it.
    fn selects_nothing(&self) -> bool {
        false
    }

    /// Optional exact membership view of this query's immutable final mask.
    /// Call after [`Self::wait_for_ready`]. Custom filters retain the original path.
    ///
    /// ```
    /// # use lance_index::prefilter::PreFilter;
    /// # async fn example(filter: &dyn PreFilter) -> lance_core::Result<()> {
    /// filter.wait_for_ready().await?;
    /// if let Some(view) = filter.prepared_mask() { let _ = view.selected(1); }
    /// # Ok(())
    /// # }
    /// ```
    fn prepared_mask(&self) -> Option<Arc<PreparedRowAddrMask>> {
        None
    }

    /// Whether position filtering is exactly the membership of [`Self::mask`].
    ///
    /// Call after [`Self::wait_for_ready`]. This also promises the same input
    /// ordering and duplicate-row behavior, with no additional filtering rules.
    /// The default retains [`Self::filter_row_ids`] for custom filters and
    /// ownership wrappers that have not established this equivalence.
    ///
    /// ```
    /// # use lance_index::prefilter::PreFilter;
    /// # async fn example(filter: &dyn PreFilter) -> lance_core::Result<()> {
    /// filter.wait_for_ready().await?;
    /// if filter.is_mask_equivalent() {
    ///     let rows = [1_u64, 1, 5];
    ///     assert_eq!(filter.mask().selected_indices(rows.iter()),
    ///         filter.filter_row_ids(Box::new(rows.iter())));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    fn is_mask_equivalent(&self) -> bool {
        false
    }

    /// Check whether a slice of row ids should be included in a query.
    ///
    /// Returns a vector of indices into the input slice that should be included,
    /// also known as a selection vector.
    ///
    /// This method must be called after `wait_for_ready`
    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64>;
}

/// A prefilter that does nothing
pub struct NoFilter;

#[async_trait]
impl PreFilter for NoFilter {
    async fn wait_for_ready(&self) -> Result<()> {
        Ok(())
    }

    fn is_empty(&self) -> bool {
        true
    }

    fn mask(&self) -> Arc<RowAddrMask> {
        Arc::new(RowAddrMask::all_rows())
    }

    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        row_ids.enumerate().map(|(i, _)| i as u64).collect()
    }
}
