// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use datafusion::execution::SendableRecordBatchStream;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::fmindex::FMIndexScalarIndex;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, OldIndexDataFilter, index_files_to_table};
use lance_table::format::IndexMetadata;
use std::sync::Arc;
use uuid::Uuid;

use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

/// Merge existing partitions and optional newly appended text without rebuilding BWTs.
pub(in crate::index) async fn open_and_merge_segments(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    new_data: Option<SendableRecordBatchStream>,
    new_store: &LanceIndexStore,
    old_data_filters: &[Option<OldIndexDataFilter>],
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<CreatedIndex> {
    let mut indices = Vec::with_capacity(segments.len());
    for segment in segments {
        indices.push(
            super::open_scalar_index_with_plan(
                dataset,
                field_path,
                segment,
                staged.and_then(|plans| plans.get(&segment.uuid)),
                crate::index::frag_reuse::OpenPurpose::Maintenance,
                &NoOpMetricsCollector,
            )
            .await?,
        );
    }
    let indices = indices
        .iter()
        .map(|index| {
            index
                .as_any()
                .downcast_ref::<FMIndexScalarIndex>()
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "FMIndex merge expected Fm, got {:?}",
                        index.index_type()
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    FMIndexScalarIndex::merge_segments(&indices, new_data, new_store, old_data_filters).await
}

/// Merge one caller-defined group of source FM-Index segments into a single segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<IndexMetadata> {
    let reference = segments
        .first()
        .ok_or_else(|| Error::invalid_input("No FMIndex segments were provided"))?;
    let field_id = *reference.fields.first().ok_or_else(|| {
        Error::invalid_input(format!(
            "FMIndex segment {} is missing field ids",
            reference.uuid
        ))
    })?;
    let field_path = dataset.schema().field_path(field_id)?;
    let segment_refs = segments.iter().collect::<Vec<_>>();
    let (fragment_bitmap, mut old_data_filters) =
        crate::index::append::build_per_segment_filters(dataset, &segment_refs, staged, true)
            .await?;
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
    let new_uuid = Uuid::new_v4();
    let store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid)?;
    let created_index = open_and_merge_segments(
        dataset,
        &field_path,
        &segment_refs,
        None,
        &store,
        &old_data_filters,
        staged,
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
        ..reference.clone()
    })
}
