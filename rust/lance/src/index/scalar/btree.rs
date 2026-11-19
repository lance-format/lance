// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::redundant_pub_crate)]

//! BTree-specific helpers for the segmented index workflow.
use datafusion::execution::SendableRecordBatchStream;
use lance_index::pbold::BTreeIndexDetails;
use lance_index::scalar::btree::BTreeIndex;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, OldIndexDataFilter};
use lance_index::scalar::{TrainingCriteria, TrainingOrdering};
use lance_table::format::IndexMetadata;

use super::segment_merge::{SegmentMerge, empty_value_row_id_stream, open_sources};
use crate::{Dataset, Error, Result};

/// Open the given BTree `segments` and k-way merge their already-sorted page
/// data, together with `new_data`, into a single canonical BTree written to
/// `new_store`.
pub(crate) async fn open_and_merge_segments(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    new_data: SendableRecordBatchStream,
    new_store: &LanceIndexStore,
    old_data_filters: &[Option<OldIndexDataFilter>],
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<CreatedIndex> {
    let source_indices =
        open_sources::<BTreeIndex>(dataset, field_path, segments, staged, "BTree").await?;
    BTreeIndex::merge_segments(&source_indices, new_data, new_store, old_data_filters).await
}

/// Merge one caller-defined group of source BTree segments into a single
/// physical segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
    refresh: &super::merge::MergeRefresh,
) -> Result<IndexMetadata> {
    let merge = SegmentMerge::new(dataset, "BTree", &segments)?;
    merge.reject_legacy_segments_on_stable(dataset, "BTree")?;
    let (fragment_bitmap, old_data_filters, new_data) = merge
        .staged_row_inputs(
            dataset,
            refresh,
            TrainingCriteria::new(TrainingOrdering::Values).with_row_addr(),
        )
        .await?;

    let (output_uuid, new_store) = merge.new_store(dataset)?;
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

    if !created_index
        .index_details
        .type_url
        .ends_with("BTreeIndexDetails")
    {
        return Err(Error::internal(format!(
            "merge_existing_index_segments: BTree merge produced unexpected details type_url '{}'",
            created_index.index_details.type_url
        )));
    }
    debug_assert_eq!(
        created_index.index_details,
        prost_types::Any::from_msg(&BTreeIndexDetails::default()).unwrap(),
    );

    Ok(merge.build_metadata(output_uuid, fragment_bitmap, created_index))
}
