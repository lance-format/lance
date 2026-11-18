// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use datafusion::execution::SendableRecordBatchStream;
use lance_index::progress::NoopIndexBuildProgress;
use lance_index::scalar::label_list::{
    BITMAP_LOOKUP_NAME, LABEL_LIST_NULLS_METADATA_KEY, LABEL_LIST_NULLS_MIN_VERSION, LabelListIndex,
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, IndexStore, OldIndexDataFilter};
use lance_table::format::IndexMetadata;

use super::segment_merge::{SegmentMerge, open_source, open_sources};
use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

async fn validate_nullable_segment_for_merge(
    dataset: &Dataset,
    field_id: i32,
    segment: &IndexMetadata,
) -> Result<()> {
    let field = dataset.schema().field_by_id(field_id).ok_or_else(|| {
        Error::invalid_input(format!(
            "merge_existing_index_segments: field id {} does not exist",
            field_id
        ))
    })?;

    if !field.nullable {
        return Ok(());
    }

    if segment.index_version < LABEL_LIST_NULLS_MIN_VERSION {
        return Err(Error::invalid_input(format!(
            "Cannot merge nullable LabelList segment {} because it was created before list-null metadata was required. Rebuild the segment instead.",
            segment.uuid
        )));
    }

    let index_store = LanceIndexStore::from_dataset_for_existing(dataset, segment).await?;
    let reader = index_store.open_index_file(BITMAP_LOOKUP_NAME).await?;
    if !reader
        .schema()
        .metadata
        .contains_key(LABEL_LIST_NULLS_METADATA_KEY)
    {
        return Err(Error::invalid_input(format!(
            "Cannot merge nullable LabelList segment {} because it is missing required metadata key {}. Rebuild the segment instead.",
            segment.uuid, LABEL_LIST_NULLS_METADATA_KEY
        )));
    }

    Ok(())
}

/// Check whether append can reuse the stored postings and list-null rows.
pub(in crate::index) fn segments_support_merge(
    dataset: &Dataset,
    segments: &[&IndexMetadata],
) -> Result<bool> {
    let Some(first) = segments.first() else {
        return Ok(false);
    };
    let field = first
        .keyed_field()
        .and_then(|id| dataset.schema().field_by_id(id))
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "LabelList segment {} is missing its indexed field",
                first.uuid
            ))
        })?;
    let row_addresses = first.results_are_row_addrs();
    Ok(segments.iter().all(|segment| {
        segment.results_are_row_addrs() == row_addresses
            && (!field.nullable || segment.index_version >= LABEL_LIST_NULLS_MIN_VERSION)
    }))
}

pub(in crate::index) async fn open_and_merge_segments(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    new_data: SendableRecordBatchStream,
    new_store: &LanceIndexStore,
    old_data_filters: &[Option<OldIndexDataFilter>],
) -> Result<CreatedIndex> {
    let source_indices =
        open_sources::<LabelListIndex>(dataset, field_path, segments, None, "LabelList").await?;
    let segments = source_indices.iter().map(Arc::as_ref).collect::<Vec<_>>();
    LabelListIndex::merge_segments(
        &segments,
        Some(new_data),
        new_store,
        old_data_filters,
        &NoopIndexBuildProgress,
    )
    .await
}

/// Merge one caller-defined group of source LabelList segments into a single segment.
///
/// A LabelList index is a bitmap over the unnested list values plus a `list_nulls`
/// row set, so segments are recombined by unioning the underlying bitmap states and
/// null sets (see [`LabelListIndex::merge_segments`]) rather than rebuilding
/// from source text.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<IndexMetadata> {
    let merge = SegmentMerge::new(dataset, "LabelList", &segments)?;
    merge.reject_legacy_segments_on_stable(dataset, "LabelList")?;
    let field_id = merge.field_id();
    let (fragment_bitmap, old_data_filters) =
        merge.per_segment_filters(dataset, staged, true).await?;

    let mut source_indices = Vec::with_capacity(segments.len());
    for segment in merge.segments() {
        validate_nullable_segment_for_merge(dataset, field_id, segment).await?;
        source_indices.push(
            open_source::<LabelListIndex>(
                dataset,
                merge.field_path(),
                segment,
                staged.and_then(|plans| plans.get(&segment.uuid)),
                "LabelList",
            )
            .await?,
        );
    }

    let (new_uuid, new_store) = merge.new_store(dataset)?;
    let sources = source_indices.iter().map(Arc::as_ref).collect::<Vec<_>>();
    let created_index = LabelListIndex::merge_segments(
        &sources,
        None,
        &new_store,
        &old_data_filters,
        &NoopIndexBuildProgress,
    )
    .await?;

    Ok(merge.build_metadata(
        new_uuid,
        dataset.manifest.version,
        fragment_bitmap,
        created_index,
    ))
}
