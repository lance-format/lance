// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Snapshot-local reconciliation of index entries with current values.
//!
//! Segment metadata returned by the query listing already carries the coverage
//! resolved by fragment reuse. Keep that ownership decision there, and use each
//! selected segment's own fields and freshness version to identify stale entries.
//! Matching, scoring, and the decision to replay under `fast_search` belong to
//! the consumer. Deletions remain the responsibility of the read and prefilter.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use lance_core::Result;
use lance_select::{RowAddrMask, RowAddrTreeMap};
use lance_table::format::overlay::staleness::{
    collect_overlay_stale_rows_for_segment, overlaid_fragments,
};
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;
use tokio::sync::OnceCell;
use uuid::Uuid;

use crate::Dataset;
use crate::dataset::rowids::translate_addr_treemap_to_row_ids;
use crate::index::scalar_logical::load_named_scalar_segments;

/// The physical rows that selected index entries cannot answer, with a lazy,
/// snapshot-pinned selection in the result domain. Share that conversion between
/// exclusion and replay; a stable ID is not a physical address.
pub struct IndexVisibility {
    dataset: Arc<Dataset>,
    physical_rows: HashMap<u32, RoaringBitmap>,
    result_rows: OnceCell<RowAddrTreeMap>,
}

impl IndexVisibility {
    pub(crate) fn from_segments(
        dataset: Arc<Dataset>,
        segments: &[IndexMetadata],
        fragments: &[Fragment],
    ) -> Result<Self> {
        let overlaid = overlaid_fragments(fragments);
        let mut physical_rows = HashMap::new();
        if !overlaid.is_empty() {
            for segment in segments {
                collect_overlay_stale_rows_for_segment(
                    segment,
                    &overlaid,
                    &mut physical_rows,
                    dataset.schema(),
                )?;
            }
        }
        Ok(Self::from_physical_rows(dataset, physical_rows))
    }

    pub(crate) fn from_physical_rows(
        dataset: Arc<Dataset>,
        physical_rows: HashMap<u32, RoaringBitmap>,
    ) -> Self {
        Self {
            dataset,
            physical_rows,
            result_rows: OnceCell::new(),
        }
    }

    /// Load only the chosen scalar segments, and avoid metadata loading entirely
    /// when none of the requested fragments carries overlays.
    pub(crate) async fn for_scalar_indices(
        dataset: Arc<Dataset>,
        indices: &[(&str, &str)],
        fragments: &[Fragment],
        segment_uuids: Option<&HashSet<Uuid>>,
    ) -> Result<Self> {
        if fragments
            .iter()
            .all(|fragment| fragment.overlays.is_empty())
        {
            return Ok(Self::from_physical_rows(dataset, HashMap::new()));
        }
        let mut selected = Vec::new();
        for &(column, name) in indices {
            let segments = load_named_scalar_segments(&dataset, column, name).await?;
            selected.extend(segments.into_iter().filter(|segment| {
                segment_uuids.is_none_or(|selected| selected.contains(&segment.uuid))
            }));
        }
        Self::from_segments(dataset, &selected, fragments)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.physical_rows.is_empty()
    }

    /// Physical addresses for scans and coverage accounting, never stable IDs.
    pub(crate) fn physical_selection(&self) -> RowAddrTreeMap {
        let mut rows = RowAddrTreeMap::new();
        for (&fragment, offsets) in &self.physical_rows {
            rows.insert_bitmap(fragment, offsets.clone());
        }
        rows
    }

    /// IDs in the domain emitted by index evaluation. Translation retains the
    /// physical offset positions of deleted slots in stable-ID sequences.
    pub(crate) fn result_rows(&self) -> BoxFuture<'_, Result<&RowAddrTreeMap>> {
        async move {
            let load = async {
                let physical = self.physical_selection();
                if self.is_empty() || !self.dataset.manifest().uses_stable_row_ids() {
                    Ok(physical)
                } else {
                    translate_addr_treemap_to_row_ids(&self.dataset, &physical).await
                }
            }
            .boxed();
            self.result_rows.get_or_try_init(|| load).await
        }
        .boxed()
    }

    pub(crate) async fn block_mask(&self) -> Result<Option<RowAddrMask>> {
        if self.is_empty() {
            return Ok(None);
        }
        Ok(Some(RowAddrMask::from_block(
            self.result_rows().await?.clone(),
        )))
    }
}
