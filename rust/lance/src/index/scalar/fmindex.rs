// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use datafusion::execution::SendableRecordBatchStream;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::progress::noop_progress;
use lance_index::scalar::fmindex::FMIndexScalarIndex;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{
    BuiltinIndexType, CreatedIndex, OldIndexDataFilter, ScalarIndexParams, index_files_to_table,
};
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use std::sync::Arc;
use uuid::Uuid;

use crate::index::frag_reuse::{OpenPurpose, SegmentRemappingPlan, StagedRemappingPlans};
use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

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
    let mut indices = Vec::with_capacity(segments.len());
    for segment in segments {
        indices.push(
            super::open_scalar_index_with_plan(
                dataset,
                field_path,
                segment,
                staged.and_then(|plans| plans.get(&segment.uuid)),
                OpenPurpose::Maintenance,
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

/// Merge one caller-defined group after the lifecycle has reconciled its coverage.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    dataset_version: u64,
    has_remapped_coverage: bool,
    staged: Option<&StagedRemappingPlans>,
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
    let mut fragment_bitmap = RoaringBitmap::new();
    let mut sources = Vec::with_capacity(segments.len());
    let mut filters = Vec::with_capacity(segments.len());
    let mut needs_tagged_translation = false;
    for segment in &segments {
        let effective = segment
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "FMIndex segment {} is missing fragment coverage",
                    segment.uuid
                ))
            })?;
        if effective.is_empty() {
            continue;
        }
        fragment_bitmap |= &effective;
        needs_tagged_translation |= matches!(
            staged.and_then(|plans| plans.get(&segment.uuid)),
            Some(SegmentRemappingPlan::Translate { .. })
        );
        sources.push(segment);
        // Pure merges preserve values from the source snapshot. A deletion
        // in a still-live fragment can be undone before committing this segment.
        filters.push(Some(OldIndexDataFilter::Fragments {
            to_keep: effective,
            to_remove: segment
                .deleted_fragment_bitmap(&dataset.fragment_bitmap)
                .unwrap_or_default(),
        }));
    }

    // Returning the staged UUID is safe only if the lifecycle never touched
    // its provenance and every covered fragment still exists.
    if segments.len() == 1
        && staged.is_none()
        && !has_remapped_coverage
        && !fragment_bitmap.is_empty()
        && reference.fragment_bitmap.as_ref() == Some(&fragment_bitmap)
    {
        return Ok(segments.into_iter().next().expect("one source"));
    }

    let uuid = Uuid::new_v4();
    let (created, dataset_version) = if needs_tagged_translation {
        // The FMIndex reader cannot translate tagged addresses yet. Preserve
        // the staged build API's ability to merge this derived coverage by
        // training it at the snapshot whose addresses it describes.
        let created = super::build_scalar_index(
            dataset,
            &field_path,
            uuid,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::Fm),
            true,
            Some(fragment_bitmap.iter().collect()),
            None,
            noop_progress(),
        )
        .await?;
        (created, dataset.manifest.version)
    } else {
        let store = LanceIndexStore::from_dataset_for_new(dataset, &uuid)?;
        let created = open_and_merge_segments(
            dataset,
            &field_path,
            &sources,
            None,
            &store,
            &filters,
            staged,
        )
        .await?;
        (created, dataset_version)
    };
    Ok(IndexMetadata {
        uuid,
        dataset_version,
        fragment_bitmap: Some(fragment_bitmap),
        index_details: Some(Arc::new(created.index_details)),
        index_version: created.index_version as i32,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: Some(index_files_to_table(created.files)),
        clustering_id: None,
        metadata: Default::default(),
        ..reference.clone()
    })
}
