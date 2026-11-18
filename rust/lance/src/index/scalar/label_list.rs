// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use datafusion::physical_plan::SendableRecordBatchStream;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::progress::NoopIndexBuildProgress;
use lance_index::scalar::label_list::{
    BITMAP_LOOKUP_NAME, LABEL_LIST_NULLS_METADATA_KEY, LABEL_LIST_NULLS_MIN_VERSION, LabelListIndex,
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, IndexStore, OldIndexDataFilter, index_files_to_table};
use lance_table::format::IndexMetadata;
use std::sync::Arc;
use uuid::Uuid;

use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

fn ensure_label_list_details(segment: &IndexMetadata) -> Result<()> {
    if let Some(details) = segment.index_details.as_ref()
        && !details.type_url.ends_with("LabelListIndexDetails")
    {
        return Err(Error::invalid_input(format!(
            "Segment '{}' is not a LabelList segment (details type_url = '{}')",
            segment.uuid, details.type_url
        )));
    }
    Ok(())
}

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

async fn open_merge_source(
    dataset: &Dataset,
    field_path: &str,
    segment: &IndexMetadata,
    plan: Option<&crate::index::frag_reuse::SegmentRemappingPlan>,
) -> Result<Arc<LabelListIndex>> {
    let scalar_index = super::open_scalar_index_with_plan(
        dataset,
        field_path,
        segment,
        plan,
        crate::index::frag_reuse::OpenPurpose::Maintenance,
        &NoOpMetricsCollector,
    )
    .await?;
    let label_list_index = scalar_index
        .as_any()
        .downcast_ref::<LabelListIndex>()
        .ok_or_else(|| {
            Error::index(format!(
                "merge_existing_index_segments: expected label list segment {}, got {:?}",
                segment.uuid,
                scalar_index.index_type()
            ))
        })?;
    Ok(Arc::new(label_list_index.clone()))
}

pub(in crate::index) async fn open_and_merge_segments(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    new_data: SendableRecordBatchStream,
    new_store: &LanceIndexStore,
    old_data_filters: &[Option<OldIndexDataFilter>],
) -> Result<CreatedIndex> {
    let mut sources = Vec::with_capacity(segments.len());
    for segment in segments {
        sources.push(open_merge_source(dataset, field_path, segment, None).await?);
    }
    let segments = sources.iter().map(Arc::as_ref).collect::<Vec<_>>();
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
    if segments.is_empty() {
        return Err(Error::index("No segment metadata was provided".to_string()));
    }

    for segment in &segments {
        ensure_label_list_details(segment)?;
    }

    // A pre-migration segment stores row ids directly, which on a
    // stable-row-id dataset are a different domain than the row addresses a
    // modern segment stores; merging the two into one segment would silently
    // combine incompatible values in the postings column. This path never
    // rescans the dataset to repair that, so refuse rather than corrupt --
    // the caller should fully rebuild the legacy segment first (e.g. via
    // `create_index(..., replace: true)`).
    if dataset.manifest.uses_stable_row_ids() {
        for segment in &segments {
            if !segment.results_are_row_addrs() {
                return Err(Error::invalid_input(format!(
                    "LabelList merge_segments: segment {} predates row-address-domain support \
                     and cannot be merged on a dataset with stable row IDs; rebuild it first \
                     (e.g. with create_index(..., replace: true))",
                    segment.uuid,
                )));
            }
        }
    }

    let field_id = *segments[0].fields.first().ok_or_else(|| {
        Error::invalid_input(format!(
            "CreateIndex: segment {} is missing field ids",
            segments[0].uuid
        ))
    })?;
    let field_path = dataset.schema().field_path(field_id)?;

    let segment_refs = segments.iter().collect::<Vec<_>>();
    let (fragment_bitmap, old_data_filters) =
        crate::index::append::build_per_segment_filters(dataset, &segment_refs, staged, true)
            .await?;

    let mut source_indices = Vec::with_capacity(segments.len());
    for segment in &segments {
        validate_nullable_segment_for_merge(dataset, field_id, segment).await?;
        source_indices.push(
            open_merge_source(
                dataset,
                &field_path,
                segment,
                staged.and_then(|plans| plans.get(&segment.uuid)),
            )
            .await?,
        );
    }

    let new_uuid = Uuid::new_v4();
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid)?;
    let sources = source_indices.iter().map(Arc::as_ref).collect::<Vec<_>>();
    let created_index = LabelListIndex::merge_segments(
        &sources,
        None,
        &new_store,
        &old_data_filters,
        &NoopIndexBuildProgress,
    )
    .await?;

    Ok(IndexMetadata {
        uuid: new_uuid,
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(fragment_bitmap),
        index_details: Some(Arc::new(created_index.index_details)),
        index_version: created_index.index_version as i32,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: Some(index_files_to_table(created_index.files)),
        clustering_id: None,
        metadata: Default::default(),
        ..segments[0].clone()
    })
}
