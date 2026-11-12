// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use datafusion::execution::SendableRecordBatchStream;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::bloomfilter::{
    BloomFilterIndex, merge_bloomfilter_indices, merge_bloomfilter_indices_with_new_data,
    read_bloomfilter_file_info,
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, index_files_to_table};
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::index::frag_reuse::{OpenPurpose, StagedRemappingPlans};
use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

/// Merge one caller-defined group of source BloomFilter segments into a single segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&StagedRemappingPlans>,
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
    let fragment_filters = effective_coverage(dataset, &segment_refs)?;
    let fragment_bitmap = fragment_filters
        .iter()
        .fold(RoaringBitmap::new(), |acc, effective| acc | effective);

    let new_uuid = Uuid::new_v4();
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid)?;
    let created_index = open_and_merge_segments(
        dataset,
        &field_path,
        &segment_refs,
        None,
        &new_store,
        &fragment_filters,
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

/// The live fragments of each segment's coverage, the zones a merge keeps.
pub(in crate::index) fn effective_coverage(
    dataset: &Dataset,
    segments: &[&IndexMetadata],
) -> Result<Vec<RoaringBitmap>> {
    let dataset_fragments = dataset.fragment_bitmap.as_ref();
    segments
        .iter()
        .map(|segment| {
            segment
                .effective_fragment_bitmap(dataset_fragments)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "CreateIndex: segment {} is missing fragment coverage",
                        segment.uuid
                    ))
                })
        })
        .collect()
}

/// Merge the zones `segments` hold over `fragment_filters`, and the zones
/// trained from `new_data` when there is any, into `new_store`.
pub(in crate::index) async fn open_and_merge_segments(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    new_data: Option<SendableRecordBatchStream>,
    new_store: &LanceIndexStore,
    fragment_filters: &[RoaringBitmap],
    staged: Option<&StagedRemappingPlans>,
) -> Result<CreatedIndex> {
    let covers_nothing = fragment_filters.iter().all(RoaringBitmap::is_empty);
    let mut scalar_indices = Vec::with_capacity(segments.len());
    for (position, (segment, effective)) in segments.iter().zip(fragment_filters).enumerate() {
        if effective.is_empty() && !(covers_nothing && position == 0) {
            continue;
        }
        let scalar_index = super::open_scalar_index_with_plan(
            dataset,
            field_path,
            segment,
            staged.and_then(|plans| plans.get(&segment.uuid)),
            OpenPurpose::Maintenance,
            &NoOpMetricsCollector,
        )
        .await?;
        scalar_indices.push((segment.uuid, scalar_index, effective));
    }

    let mut source_indices = Vec::with_capacity(scalar_indices.len());
    for (segment_uuid, scalar_index, fragment_filter) in &scalar_indices {
        let bloomfilter_index = scalar_index
            .as_any()
            .downcast_ref::<BloomFilterIndex>()
            .ok_or_else(|| {
                Error::index(format!(
                    "expected bloom filter segment {}, got {:?}",
                    segment_uuid,
                    scalar_index.index_type()
                ))
            })?;
        source_indices.push((bloomfilter_index, *fragment_filter));
    }

    match new_data {
        Some(new_data) => {
            merge_bloomfilter_indices_with_new_data(&source_indices, new_data, new_store).await
        }
        None => merge_bloomfilter_indices(&source_indices, new_store).await,
    }
}

/// Whether the zones of `segments` merge with zones built with the parameters
/// of `reference`: each segment records its null rows, which a merge would
/// otherwise drop for every row it holds, and was built with those parameters.
pub(in crate::index) async fn segments_merge_by_zones(
    dataset: &Dataset,
    reference: &IndexMetadata,
    segments: &[&IndexMetadata],
) -> Result<bool> {
    let file_info = async |segment: &IndexMetadata| {
        let store = LanceIndexStore::from_dataset_for_existing(dataset, segment).await?;
        read_bloomfilter_file_info(&store).await
    };
    let params = file_info(reference).await?.params;
    for segment in segments {
        let info = file_info(segment).await?;
        if !info.has_null_rows || info.params != params {
            return Ok(false);
        }
    }
    Ok(true)
}
