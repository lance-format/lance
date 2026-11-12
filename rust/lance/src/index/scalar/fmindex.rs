// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use datafusion::execution::SendableRecordBatchStream;
use lance_index::scalar::fmindex::FMIndexScalarIndex;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, OldIndexDataFilter};
use lance_table::format::IndexMetadata;

use super::segment_merge::{SegmentMerge, downcast_sources, open_dyn_sources};
use crate::index::frag_reuse::StagedRemappingPlans;
use crate::{Dataset, Result};

/// Merge existing partitions and optional newly appended text without rebuilding BWTs.
pub(in crate::index) async fn open_and_merge_segments(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    new_data: Option<SendableRecordBatchStream>,
    new_store: &LanceIndexStore,
    old_data_filters: &[Option<OldIndexDataFilter>],
    staged: Option<&StagedRemappingPlans>,
) -> Result<CreatedIndex> {
    let sources = open_dyn_sources(dataset, field_path, segments, staged).await?;
    let indices = downcast_sources::<FMIndexScalarIndex>(&sources, segments, "FMIndex")?;
    FMIndexScalarIndex::merge_segments(&indices, new_data, new_store, old_data_filters).await
}

/// Merge one caller-defined group of source FM-Index segments into a single segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&StagedRemappingPlans>,
) -> Result<IndexMetadata> {
    let merge = SegmentMerge::new(dataset, "FMIndex", &segments)?;
    let (fragment_bitmap, mut old_data_filters) =
        merge.per_segment_filters(dataset, staged, true).await?;
    // A pure merge keeps the source version. Preserve deleted rows in live
    // fragments: restoring that version can make them visible again. Query-time
    // deletion masks hide them until then; retired fragments can still be dropped.
    for filter in &mut old_data_filters {
        *filter = match filter.take() {
            Some(OldIndexDataFilter::RowAddresses {
                to_keep, to_remove, ..
            }) => Some(OldIndexDataFilter::Fragments { to_keep, to_remove }),
            other => other,
        };
    }
    let (new_uuid, store) = merge.new_store(dataset)?;
    let created_index = open_and_merge_segments(
        dataset,
        merge.field_path(),
        merge.segments(),
        None,
        &store,
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
