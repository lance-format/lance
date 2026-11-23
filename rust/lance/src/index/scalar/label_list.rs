// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_index::scalar::IndexStore;
use lance_index::scalar::label_list::{
    BITMAP_LOOKUP_NAME, LABEL_LIST_NULLS_METADATA_KEY, LABEL_LIST_NULLS_MIN_VERSION, LabelListIndex,
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_table::format::IndexMetadata;

use super::segment_merge::{SegmentMerge, open_source};
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

/// Merge one caller-defined group of source LabelList segments into a single segment.
///
/// A LabelList index is a bitmap over the unnested list values plus a `list_nulls`
/// row set, so segments are recombined by unioning the underlying bitmap states and
/// null sets (see [`lance_index::scalar::label_list::merge_label_list_indices`])
/// rather than rebuilding from source text.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<IndexMetadata> {
    let merge = SegmentMerge::new(dataset, "LabelList", &segments)?;
    merge.ensure_details("LabelList", "LabelListIndexDetails")?;
    merge.require_shared_domain(dataset, "LabelList")?;
    let field_id = merge.field_id();

    let (effective_old_frags, deleted_old_frags) = merge.coverage_union(dataset)?;
    let fragment_bitmap = effective_old_frags.clone();
    // A live fragment can have deleted rows even when no whole fragment was
    // retired. Clean both value postings and list-null rows in that case too.
    let has_partial_deletions = effective_old_frags.iter().any(|id| {
        dataset
            .find_fragment(u64::from(id))
            .is_some_and(|fragment| fragment.deletion_file.is_some())
    });
    let old_data_filter = if deleted_old_frags.is_empty() && !has_partial_deletions {
        None
    } else {
        crate::index::append::build_old_data_filter(
            dataset,
            &effective_old_frags,
            &deleted_old_frags,
            true,
        )
        .await?
    };

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
    let created_index = lance_index::scalar::label_list::merge_label_list_indices(
        &source_indices,
        &new_store,
        old_data_filter,
        lance_index::progress::noop_progress(),
    )
    .await?;

    Ok(merge.build_metadata(
        new_uuid,
        dataset.manifest.version,
        fragment_bitmap,
        created_index,
    ))
}
