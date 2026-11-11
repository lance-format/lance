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
use std::sync::{Arc, OnceLock};

use futures::FutureExt;
use futures::future::{BoxFuture, ready};
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
    result_rows: ResultRows,
    has_stale_unknown_coverage: bool,
}

enum ResultRows {
    // Address selections need no I/O and must not wait for async task budget.
    Addresses(OnceLock<RowAddrTreeMap>),
    StableIds(OnceCell<RowAddrTreeMap>),
}

impl IndexVisibility {
    pub(crate) fn from_segments(
        dataset: Arc<Dataset>,
        segments: &[IndexMetadata],
        fragments: &[Fragment],
    ) -> Result<Self> {
        let overlaid = overlaid_fragments(fragments);
        let mut plan = Self::from_physical_rows(dataset, HashMap::new());
        plan.collect_segments(segments, &overlaid)?;
        Ok(plan)
    }

    // Construction only: the result-domain selection has not been initialized.
    fn collect_segments(
        &mut self,
        segments: &[IndexMetadata],
        overlaid: &HashMap<u32, &Fragment>,
    ) -> Result<()> {
        if overlaid.is_empty() {
            return Ok(());
        }
        for segment in segments {
            if segment.fragment_bitmap.is_none() {
                let mut stale = HashMap::new();
                collect_overlay_stale_rows_for_segment(
                    segment,
                    overlaid,
                    &mut stale,
                    self.dataset.schema(),
                )?;
                self.has_stale_unknown_coverage |= !stale.is_empty();
                for (fragment, offsets) in stale {
                    *self.physical_rows.entry(fragment).or_default() |= offsets;
                }
            } else {
                collect_overlay_stale_rows_for_segment(
                    segment,
                    overlaid,
                    &mut self.physical_rows,
                    self.dataset.schema(),
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn from_physical_rows(
        dataset: Arc<Dataset>,
        physical_rows: HashMap<u32, RoaringBitmap>,
    ) -> Self {
        let result_rows = if dataset.manifest().uses_stable_row_ids() {
            ResultRows::StableIds(OnceCell::new())
        } else {
            ResultRows::Addresses(OnceLock::new())
        };
        Self {
            dataset,
            physical_rows,
            result_rows,
            has_stale_unknown_coverage: false,
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
        let overlaid = overlaid_fragments(fragments);
        let mut plan = Self::from_physical_rows(dataset, HashMap::new());
        if overlaid.is_empty() {
            return Ok(plan);
        }
        // Scalar expressions and composite key probes require every selected
        // leaf to cover a fragment. Anything else is already read in full by
        // their fallback, and must not also receive stale-row replay.
        let mut covered: RoaringBitmap = fragments
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect();
        for &(column, name) in indices {
            let mut segments = load_named_scalar_segments(&plan.dataset, column, name).await?;
            if let Some(segment_uuids) = segment_uuids {
                segments.retain(|segment| segment_uuids.contains(&segment.uuid));
            }
            // Unknown historical coverage remains conservative, rather than
            // being interpreted as an empty fragment set.
            if segments
                .iter()
                .all(|segment| segment.fragment_bitmap.is_some())
            {
                let mut leaf_coverage = RoaringBitmap::new();
                for segment in &segments {
                    if let Some(coverage) = &segment.fragment_bitmap {
                        leaf_coverage |= coverage;
                    }
                }
                covered &= leaf_coverage;
            }
            plan.collect_segments(&segments, &overlaid)?;
        }
        plan.physical_rows
            .retain(|fragment, _| covered.contains(*fragment));
        Ok(plan)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.physical_rows.is_empty()
    }

    /// FTS preserves its historical whole-scan fallback when a stale segment
    /// cannot identify its indexed fragment set. Unknown is not empty coverage.
    pub(crate) fn has_stale_unknown_coverage(&self) -> bool {
        self.has_stale_unknown_coverage
    }

    pub(crate) fn physical_rows(&self) -> &HashMap<u32, RoaringBitmap> {
        &self.physical_rows
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
        match &self.result_rows {
            ResultRows::Addresses(cache) => {
                ready(Ok(cache.get_or_init(|| self.physical_selection()))).boxed()
            }
            ResultRows::StableIds(cache) => {
                if let Some(rows) = cache.get() {
                    return ready(Ok(rows)).boxed();
                }
                async move {
                    let load = async {
                        let physical = self.physical_selection();
                        if self.is_empty() {
                            Ok(physical)
                        } else {
                            translate_addr_treemap_to_row_ids(&self.dataset, &physical).await
                        }
                    }
                    .boxed();
                    cache.get_or_try_init(|| load).await
                }
                .boxed()
            }
        }
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

#[cfg(test)]
mod tests {
    use std::task::Poll;

    use arrow_array::types::UInt32Type;
    use futures::future::poll_fn;
    use lance_datagen::{BatchCount, RowCount, array, gen_batch};
    use tokio::task::coop::poll_proceed;

    use super::*;

    #[tokio::test]
    async fn address_selection_does_not_wait_for_async_budget() {
        let reader = gen_batch()
            .col("value", array::step::<UInt32Type>())
            .into_reader_rows(RowCount::from(4), BatchCount::from(1));
        let dataset = Dataset::write(reader, "memory://", None).await.unwrap();
        let plan = IndexVisibility::from_physical_rows(
            Arc::new(dataset),
            HashMap::from([(0, [1].into_iter().collect())]),
        );
        poll_fn(|context| {
            while let Poll::Ready(guard) = poll_proceed(context) {
                guard.made_progress();
            }
            // Address-domain selection only copies in-memory metadata. It must
            // not add a scheduler round trip before an indexed read can start.
            let rows = plan
                .result_rows()
                .now_or_never()
                .expect("address-domain visibility must be ready even without async task budget")
                .unwrap();
            assert_eq!(rows, &RowAddrTreeMap::from_iter([1]));
            Poll::Ready(())
        })
        .await;
    }
    #[rstest::rstest]
    #[case::inline(4)]
    #[case::dense(4096)]
    #[tokio::test]
    async fn stable_selection_is_shared_and_excludes_deleted_slots(#[case] fragment_rows: u32) {
        let reader = gen_batch()
            .col("value", array::step::<UInt32Type>())
            .into_reader_rows(
                RowCount::from(u64::from(fragment_rows) * 2),
                BatchCount::from(1),
            );
        let mut dataset = Dataset::write(
            reader,
            "memory://",
            Some(crate::dataset::WriteParams {
                max_rows_per_file: fragment_rows as usize,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .delete(&format!(
                "value = {fragment_rows} OR value = {}",
                fragment_rows + 2
            ))
            .await
            .unwrap();
        let offsets: RoaringBitmap = (0..fragment_rows).collect();
        let plan = IndexVisibility::from_physical_rows(
            Arc::new(dataset),
            HashMap::from([(1, offsets.clone())]),
        );
        let mut physical = RowAddrTreeMap::new();
        physical.insert_bitmap(1, offsets);
        assert_eq!(plan.physical_selection(), physical);
        poll_fn(|context| {
            while let Poll::Ready(guard) = poll_proceed(context) {
                guard.made_progress();
            }
            assert!(plan.result_rows().now_or_never().is_none());
            Poll::Ready(())
        })
        .await;
        let first = plan.result_rows().await.unwrap();
        let expected: RowAddrTreeMap = (fragment_rows..fragment_rows * 2)
            .filter(|id| *id != fragment_rows && *id != fragment_rows + 2)
            .map(u64::from)
            .collect();
        assert_eq!(first, &expected);
        let block = plan.block_mask().await.unwrap().unwrap();
        assert!(!block.selected(u64::from(fragment_rows + 1)));
        assert!(block.selected(u64::from(fragment_rows + 2)));
        assert!(std::ptr::eq(first, plan.result_rows().await.unwrap()));
        if fragment_rows >= 4096 {
            let invalid = IndexVisibility::from_physical_rows(
                plan.dataset.clone(),
                HashMap::from([(1, (0..=fragment_rows).collect())]),
            );
            let error = invalid.result_rows().await.unwrap_err();
            assert!(matches!(error, lance_core::Error::Internal { .. }));
            assert!(error.to_string().contains("physical_offset=4096"));
        }
    }

    #[rstest::rstest]
    #[case::unknown(None, true)]
    #[case::empty(Some(RoaringBitmap::new()), false)]
    #[case::covered(Some([0].into_iter().collect()), false)]
    #[tokio::test]
    async fn unknown_coverage_is_distinct_from_empty(
        #[case] coverage: Option<RoaringBitmap>,
        #[case] is_unknown: bool,
    ) {
        let reader = gen_batch()
            .col("value", array::step::<UInt32Type>())
            .into_reader_rows(RowCount::from(4), BatchCount::from(1));
        let dataset = Arc::new(Dataset::write(reader, "memory://", None).await.unwrap());
        let mut fragments = dataset.fragments().as_ref().clone();
        fragments[0]
            .overlays
            .push(lance_table::format::overlay::DataOverlayFile {
                data_file: lance_table::format::DataFile::new_legacy_from_fields(
                    "unused.lance",
                    vec![0],
                    None,
                ),
                coverage: lance_table::format::overlay::OverlayCoverage::dense(
                    [1].into_iter().collect(),
                ),
                committed_version: 2,
            });
        let segment = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "value_idx".into(),
            fields: vec![0],
            covering_fields: vec![],
            dataset_version: 1,
            fragment_bitmap: coverage.clone(),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        };
        let plan = IndexVisibility::from_segments(dataset, &[segment], &fragments).unwrap();
        assert_eq!(plan.has_stale_unknown_coverage(), is_unknown);
        assert_eq!(plan.is_empty(), coverage.is_some_and(|c| c.is_empty()));
    }
}
