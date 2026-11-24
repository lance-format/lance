// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use datafusion::physical_plan::SendableRecordBatchStream;
use lance_geo::rtree::RTreeIndex;
use lance_index::progress::noop_progress;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, OldIndexDataFilter};
use lance_table::format::IndexMetadata;

use super::segment_merge::{SegmentMerge, open_sources};
use crate::{Dataset, Result};

/// Merge one caller-defined group of source RTree segments into a single segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<IndexMetadata> {
    let merge = SegmentMerge::new(dataset, "RTree", &segments)?;
    // Without stable row ids the two domains coincide, so once a row-id-domain
    // segment is refused on a stable-row-id dataset the filters can always be
    // address-domain. `RTreeIndex::merge_segments` keeps the sources' own domain.
    super::shared_segment_domain(dataset, "RTree", merge.segments())?;
    let (fragment_bitmap, old_data_filters) =
        merge.per_segment_filters(dataset, staged, true).await?;

    let (new_uuid, new_store) = merge.new_store(dataset)?;
    let created_index = open_and_merge_segments(
        dataset,
        merge.field_path(),
        merge.segments(),
        None,
        &new_store,
        &old_data_filters,
        staged,
    )
    .await?;

    Ok(merge.build_metadata(
        new_uuid,
        merge.dataset_version(),
        fragment_bitmap,
        created_index,
    ))
}

/// Reuse stored bounding boxes for both maintenance tasks and physical merges.
pub(in crate::index) async fn open_and_merge_segments(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    new_data: Option<SendableRecordBatchStream>,
    new_store: &LanceIndexStore,
    old_data_filters: &[Option<OldIndexDataFilter>],
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<CreatedIndex> {
    let source_indices =
        open_sources::<RTreeIndex>(dataset, field_path, segments, staged, "RTree").await?;

    RTreeIndex::merge_segments(
        &source_indices,
        new_data,
        new_store,
        old_data_filters,
        noop_progress(),
    )
    .await
}
