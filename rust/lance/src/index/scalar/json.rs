// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! JSON path extraction around the existing scalar segment mergers.

use std::sync::Arc;

use datafusion::execution::SendableRecordBatchStream;
use lance_index::IndexType;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::pb::JsonIndexDetails;
use lance_index::scalar::json::JsonIndex;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, OldIndexDataFilter, index_files_to_table};
use lance_table::format::IndexMetadata;
use prost::Message;
use uuid::Uuid;

use crate::dataset::index::LanceIndexStoreExt;
use crate::index::frag_reuse::{OpenPurpose, StagedRemappingPlans};
use crate::{Dataset, Error, Result};

/// Only targets with an incremental N:1 merge participate in the split plan.
pub(in crate::index) fn merge_target_type(details: &prost_types::Any) -> Option<IndexType> {
    if !details.type_url.ends_with("JsonIndexDetails") {
        return None;
    }
    let json = JsonIndexDetails::decode(details.value.as_slice()).ok()?;
    let target = json.target_details?;
    if target.type_url.ends_with("BTreeIndexDetails") {
        Some(IndexType::BTree)
    } else if target.type_url.ends_with("BitmapIndexDetails") {
        Some(IndexType::Bitmap)
    } else if target.type_url.ends_with("NGramIndexDetails") {
        Some(IndexType::NGram)
    } else {
        None
    }
}

pub(in crate::index) async fn open_and_merge_segments(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    new_data: Option<SendableRecordBatchStream>,
    new_store: &LanceIndexStore,
    old_data_filters: &[Option<OldIndexDataFilter>],
    staged: Option<&StagedRemappingPlans>,
) -> Result<CreatedIndex> {
    let reference = segments
        .first()
        .ok_or_else(|| Error::invalid_input("JSON merge requires at least one segment"))?;
    let details = segments
        .iter()
        .map(|segment| {
            segment.index_details.as_deref().ok_or_else(|| {
                Error::invalid_input(format!(
                    "JSON segment {} has no index details",
                    segment.uuid
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    super::SCALAR_INDEX_PLUGIN_REGISTRY
        .get_plugin_by_name("json")?
        .validate_new_segments_against_existing(&[], &details)?;
    let index = super::open_scalar_index_with_plan(
        dataset,
        field_path,
        reference,
        staged.and_then(|plans| plans.get(&reference.uuid)),
        OpenPurpose::Maintenance,
        &NoOpMetricsCollector,
    )
    .await?;
    let json = index.as_any().downcast_ref::<JsonIndex>().ok_or_else(|| {
        Error::invalid_input(format!(
            "JSON merge expected a JSON segment, got {}",
            reference.uuid
        ))
    })?;
    let target_type = reference
        .index_details
        .as_deref()
        .and_then(merge_target_type);
    if target_type == Some(IndexType::NGram) {
        // Preserve NGram's absent-data input: pure consolidation must not
        // start the tokenizer pipeline or require a value conversion.
        let new_data = match new_data {
            Some(data) => Some(json.prepare_update_data(data).await?),
            None => None,
        };
        let created = super::ngram::open_and_merge_segments(
            dataset,
            segments,
            new_data,
            new_store,
            old_data_filters,
            staged,
        )
        .await?;
        return json.wrap_target(created);
    }
    let new_data = match new_data {
        Some(data) => data,
        None => super::btree::empty_value_row_id_stream(dataset, reference.fields[0])?,
    };
    let new_data = json.prepare_update_data(new_data).await?;
    let created = match target_type {
        Some(IndexType::BTree) => {
            super::btree::open_and_merge_segments(
                dataset,
                field_path,
                segments,
                new_data,
                new_store,
                old_data_filters,
                staged,
            )
            .await?
        }
        Some(IndexType::Bitmap) => {
            super::bitmap::open_and_merge_segments(
                dataset,
                field_path,
                segments,
                new_data,
                new_store,
                old_data_filters,
                staged,
            )
            .await?
        }
        target => {
            return Err(Error::not_supported(format!(
                "JSON segment merge does not support target {target:?}"
            )));
        }
    };
    json.wrap_target(created)
}

pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&StagedRemappingPlans>,
) -> Result<IndexMetadata> {
    // Delegate NGram's remap handling and source-version accounting to its lifecycle.
    if segments[0]
        .index_details
        .as_deref()
        .and_then(merge_target_type)
        == Some(IndexType::NGram)
    {
        return super::ngram::merge_segments(dataset, segments, staged).await;
    }
    // Like the native BTree/Bitmap mergers, this path writes row addresses
    // without rescanning old data, so row-ID segments must be rebuilt first.
    if dataset.manifest.uses_stable_row_ids()
        && let Some(segment) = segments
            .iter()
            .find(|segment| !segment.results_are_row_addrs())
    {
        return Err(Error::invalid_input(format!(
            "JSON merge_segments: segment {} predates row-address-domain support; rebuild it before merging on a dataset with stable row IDs",
            segment.uuid,
        )));
    }
    // The lifecycle entry point validates non-empty inputs and keyed fields.
    let reference = &segments[0];
    let field_id = reference.fields[0];
    let field_path = dataset.schema().field_path(field_id)?;
    let segment_refs = segments.iter().collect::<Vec<_>>();
    let (fragment_bitmap, old_data_filters) =
        crate::index::append::build_per_segment_filters(dataset, &segment_refs, staged, true)
            .await?;
    let uuid = Uuid::new_v4();
    let store = LanceIndexStore::from_dataset_for_new(dataset, &uuid)?;
    let created = open_and_merge_segments(
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
        uuid,
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
