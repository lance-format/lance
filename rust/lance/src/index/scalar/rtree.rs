// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use datafusion::physical_plan::SendableRecordBatchStream;
use lance_geo::rtree::RTreeIndex;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::progress::noop_progress;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, OldIndexDataFilter, index_files_to_table};
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

/// Merge one caller-defined group of source RTree segments into a single segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<IndexMetadata> {
    if segments.is_empty() {
        return Err(Error::index("No segment metadata was provided".to_string()));
    }

    let field_id = *segments[0].fields.first().ok_or_else(|| {
        Error::invalid_input(format!(
            "CreateIndex: segment {} is missing field ids",
            segments[0].uuid
        ))
    })?;
    let field_path = dataset.schema().field_path(field_id)?;
    let dataset_version = segments
        .iter()
        .map(|segment| segment.dataset_version)
        .min()
        .unwrap_or(dataset.manifest.version);
    let segment_refs = segments.iter().collect::<Vec<_>>();
    // Without stable row ids the two domains coincide, so once a row-id-domain
    // segment is refused on a stable-row-id dataset the filters can always be
    // address-domain. `RTreeIndex::merge_segments` keeps the sources' own domain.
    super::shared_segment_domain(dataset, "RTree", &segment_refs)?;
    let (fragment_bitmap, old_data_filters) =
        crate::index::append::build_per_segment_filters(dataset, &segment_refs, staged, true)
            .await?;

    let new_uuid = Uuid::new_v4();
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid)?;
    let created_index = open_and_merge_segments(
        dataset,
        &field_path,
        &segment_refs,
        None,
        &new_store,
        &old_data_filters,
        staged,
    )
    .await?;

    Ok(IndexMetadata {
        uuid: new_uuid,
        dataset_version,
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
    let mut source_indices = Vec::with_capacity(segments.len());
    for segment in segments {
        let scalar_index = super::open_scalar_index_with_plan(
            dataset,
            field_path,
            segment,
            staged.and_then(|plans| plans.get(&segment.uuid)),
            crate::index::frag_reuse::OpenPurpose::Maintenance,
            &NoOpMetricsCollector,
        )
        .await?;
        let rtree_index = scalar_index
            .as_any()
            .downcast_ref::<RTreeIndex>()
            .ok_or_else(|| {
                Error::index(format!(
                    "merge_existing_index_segments: expected RTree segment {}, got {:?}",
                    segment.uuid,
                    scalar_index.index_type()
                ))
            })?;
        source_indices.push(Arc::new(rtree_index.clone()));
    }

    RTreeIndex::merge_segments(
        &source_indices,
        new_data,
        new_store,
        old_data_filters,
        noop_progress(),
    )
    .await
}
