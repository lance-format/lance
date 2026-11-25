// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use datafusion::execution::SendableRecordBatchStream;
use lance_index::scalar::bitmap::BitmapIndex;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, OldIndexDataFilter};
use lance_index::scalar::{TrainingCriteria, TrainingOrdering};
use lance_table::format::IndexMetadata;

use super::segment_merge::{SegmentMerge, empty_value_row_id_stream, open_sources};
use crate::{Dataset, Result};

/// Open the given bitmap `segments` and k-way merge their postings, together
/// with `new_data`, into a single bitmap index written to `new_store`.
pub(in crate::index) async fn open_and_merge_segments(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    new_data: SendableRecordBatchStream,
    new_store: &LanceIndexStore,
    old_data_filters: &[Option<OldIndexDataFilter>],
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<CreatedIndex> {
    let source_indices =
        open_sources::<BitmapIndex>(dataset, field_path, segments, staged, "Bitmap").await?;
    BitmapIndex::merge_segments(&source_indices, new_data, new_store, old_data_filters).await
}

/// Merge one caller-defined group of source bitmap segments into a single segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
    refresh: &super::merge::MergeRefresh,
) -> Result<IndexMetadata> {
    let merge = SegmentMerge::new(dataset, "Bitmap", &segments)?;
    merge.reject_legacy_segments_on_stable(dataset, "Bitmap")?;
    let (fragment_bitmap, old_data_filters, new_data) = merge
        .staged_row_inputs(
            dataset,
            refresh,
            TrainingCriteria::new(TrainingOrdering::Values).with_row_addr(),
        )
        .await?;

    let (new_uuid, new_store) = merge.new_store(dataset)?;
    let new_data = match new_data {
        Some(data) => data,
        None => empty_value_row_id_stream(dataset, merge.field_id())?,
    };
    let created_index = open_and_merge_segments(
        dataset,
        merge.field_path(),
        merge.segments(),
        new_data,
        &new_store,
        &old_data_filters,
        staged,
    )
    .await?;

    Ok(merge.build_metadata(new_uuid, fragment_bitmap, created_index))
}
