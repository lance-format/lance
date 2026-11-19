// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Replacement data shared by scalar segment optimization and physical merges.

use std::borrow::Cow;
use std::cell::LazyCell;
use std::collections::HashMap;

use lance_index::scalar::OldIndexDataFilter;
use lance_select::{RowAddrTreeMap, RowSetOps};
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::TrainingRows;
use crate::dataset::overlay::{collect_overlay_stale_rows_for_segment, overlaid_fragments};
use crate::dataset::rowids::translate_addr_treemap_to_row_ids;
use crate::index::append::filters_for_segment_coverage;
use crate::{Dataset, Error, Result};

/// What a scalar merge must read from the current snapshot.
pub(in crate::index) struct MergeRefresh {
    /// Overlay replacements in physical row coordinates.
    rows: RowAddrTreeMap,
    /// Whole destinations the source filters exclude because their rows have multiple owners.
    rescan: RoaringBitmap,
}

impl MergeRefresh {
    pub(in crate::index) fn collect(
        dataset: &Dataset,
        selected_old_indices: &[&IndexMetadata],
        tagged_coverage: Option<&HashMap<Uuid, RoaringBitmap>>,
        rescan: RoaringBitmap,
    ) -> Result<Self> {
        // The same per-row, field-aware, version-gated check the scanner masks by.
        let overlaid = LazyCell::new(|| overlaid_fragments(&dataset.manifest.fragments));
        let mut stale_offsets: HashMap<u32, RoaringBitmap> = HashMap::new();
        for &segment in selected_old_indices
            .iter()
            .filter(|s| s.dataset_version < dataset.manifest.version)
        {
            if overlaid.is_empty() {
                break;
            }
            let mut segment = Cow::Borrowed(segment);
            if let Some(coverage) = tagged_coverage {
                // Tagged bitmaps name retired sources. Match overlays against each
                // segment's live coverage, retaining its own fields and version gate.
                let bitmap = coverage.get(&segment.uuid).ok_or_else(|| {
                    Error::internal(format!(
                        "tagged coverage missing for segment {}",
                        segment.uuid
                    ))
                })?;
                segment.to_mut().fragment_bitmap = Some(bitmap.clone());
            }
            // A source without known live coverage contributes no retained entries.
            if segment.fragment_bitmap.is_none() {
                continue;
            }
            collect_overlay_stale_rows_for_segment(
                segment.as_ref(),
                &overlaid,
                &mut stale_offsets,
                dataset.schema(),
            )?;
        }
        if !(stale_offsets.is_empty() && rescan.is_empty())
            && let Some(source) = selected_old_indices.first()
            && !source.covering_fields.is_empty()
        {
            return Err(Error::not_supported(format!(
                "cannot refresh index '{}' with covering fields {:?} from a single-column training stream",
                source.name, source.covering_fields
            )));
        }
        let mut stale_rows = RowAddrTreeMap::new();
        for (fragment_id, offsets) in stale_offsets {
            stale_rows.insert_bitmap(fragment_id, offsets);
        }
        Ok(Self {
            rows: stale_rows,
            rescan,
        })
    }

    pub(in crate::index) fn is_empty(&self) -> bool {
        self.rows.is_empty() && self.rescan.is_empty()
    }

    pub(in crate::index) fn fragments(&self) -> RoaringBitmap {
        let mut fragments = self.rescan.clone();
        fragments.extend(self.rows.iter().map(|(id, _)| *id));
        fragments
    }

    /// Old entries use the index's identifier domain. Training reads physical rows of this snapshot.
    pub(in crate::index) async fn training_rows(
        &self,
        dataset: &Dataset,
        unindexed: &[Fragment],
        address_domain: bool,
    ) -> Result<(RowAddrTreeMap, TrainingRows)> {
        let fragments = unindexed.iter().chain(
            self.rescan
                .iter()
                .filter_map(|id| dataset.find_fragment(u64::from(id))),
        );
        if self.rows.is_empty() {
            return Ok((
                RowAddrTreeMap::new(),
                TrainingRows::Fragments(fragments.cloned().collect()),
            ));
        }
        let mut selection = self.rows.clone();
        for fragment in fragments {
            selection.insert_fragment(fragment.id as u32);
        }
        let stale_rows = if !address_domain && dataset.manifest.uses_stable_row_ids() {
            translate_addr_treemap_to_row_ids(dataset, &self.rows).await?
        } else {
            self.rows.clone()
        };
        Ok((stale_rows, TrainingRows::RowAddrs(selection)))
    }

    pub(in crate::index) async fn row_merge_inputs(
        &self,
        dataset: &Dataset,
        segments: &[&IndexMetadata],
        tagged_coverage: Option<&HashMap<Uuid, RoaringBitmap>>,
        unindexed: &[Fragment],
        address_domain: bool,
    ) -> Result<(RoaringBitmap, Vec<Option<OldIndexDataFilter>>, TrainingRows)> {
        let (mut coverage, filters) =
            filters_for_segment_coverage(dataset, segments, tagged_coverage, address_domain)
                .await?;
        let (stale, selection) = self
            .training_rows(dataset, unindexed, address_domain)
            .await?;
        let filters = filters
            .into_iter()
            .map(|filter| block_stale_rows(filter, &stale))
            .collect();
        coverage |= &self.rescan;
        Ok((coverage, filters, selection))
    }
}

/// Remove `stale_rows` from a segment's old-data filter ([`OldIndexDataFilter`]): the rows
/// an overlay replaced after the segment was built, keyed like the filter (see
/// [`MergeRefresh::training_rows`]). The merge re-reads them.
pub(in crate::index) fn block_stale_rows(
    filter: Option<OldIndexDataFilter>,
    stale_rows: &RowAddrTreeMap,
) -> Option<OldIndexDataFilter> {
    if stale_rows.is_empty() {
        return filter;
    }
    let (to_keep, to_remove, mut deleted_rows) = match filter? {
        OldIndexDataFilter::RowIds(valid) => {
            return Some(OldIndexDataFilter::RowIds(valid - stale_rows));
        }
        OldIndexDataFilter::Fragments { to_keep, to_remove } => {
            (to_keep, to_remove, RowAddrTreeMap::new())
        }
        OldIndexDataFilter::RowAddresses {
            to_keep,
            to_remove,
            deleted_rows,
        } => (to_keep, to_remove, deleted_rows),
    };
    // Each source owns only its fragments. Copying the entire overlay mask per
    // source would multiply memory by the number of segments in the merge.
    for id in to_keep.iter() {
        if let Some(stale) = stale_rows.get_fragment_bitmap(id) {
            let mut positions = stale.clone();
            if let Some(deleted) = deleted_rows.get_fragment_bitmap(id) {
                positions |= deleted;
            }
            deleted_rows.insert_bitmap(id, positions);
        }
    }
    if deleted_rows.is_empty() {
        Some(OldIndexDataFilter::Fragments { to_keep, to_remove })
    } else {
        Some(OldIndexDataFilter::RowAddresses {
            to_keep,
            to_remove,
            deleted_rows,
        })
    }
}
