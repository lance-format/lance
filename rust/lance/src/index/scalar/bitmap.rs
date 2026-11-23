// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use datafusion::execution::SendableRecordBatchStream;
use lance_index::scalar::bitmap::BitmapIndex;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, OldIndexDataFilter};
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
) -> Result<IndexMetadata> {
    let merge = SegmentMerge::new(dataset, "Bitmap", &segments)?;
    merge.ensure_details("Bitmap", "BitmapIndexDetails")?;
    merge.require_shared_domain(dataset, "Bitmap")?;
    let (fragment_bitmap, old_data_filters) =
        merge.per_segment_filters(dataset, staged, true).await?;

    let (new_uuid, new_store) = merge.new_store(dataset)?;
    // Pure segment consolidation: no dataset scan, so `new_data` is an empty stream.
    let new_data = empty_value_row_id_stream(dataset, merge.field_id())?;
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

    Ok(merge.build_metadata(
        new_uuid,
        dataset.manifest.version,
        fragment_bitmap,
        created_index,
    ))
}
