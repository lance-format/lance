// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! JSON path extraction around the existing scalar segment mergers.

use datafusion::execution::SendableRecordBatchStream;
use lance_core::error::LanceOptionExt;
use lance_index::IndexType;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::pb::JsonIndexDetails;
use lance_index::scalar::json::JsonIndex;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, OldIndexDataFilter};
use lance_table::format::IndexMetadata;
use prost::Message;

use super::segment_merge::{SegmentMerge, empty_value_row_id_stream};
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

/// Check that target payloads can share one physical segment.
fn validate_merge_segments(segments: &[&prost_types::Any]) -> Result<()> {
    let details = segments
        .iter()
        .map(|details| JsonIndexDetails::decode(details.value.as_slice()).map_err(Error::from))
        .collect::<Result<Vec<_>>>()?;
    let Some(first) = details.first() else {
        return Ok(());
    };
    let target = first.target_details.as_ref().expect_ok()?;
    let mut targets = Vec::with_capacity(details.len());
    for details in &details {
        let other = details.target_details.as_ref().expect_ok()?;
        if details.path != first.path || other.type_url != target.type_url {
            return Err(Error::invalid_input(format!(
                "JSON segments must share a path and target index type: expected path '{}' and {}, got '{}' and {}",
                first.path, target.type_url, details.path, other.type_url,
            )));
        }
        if details.target_data_type != first.target_data_type {
            return Err(Error::invalid_input(format!(
                "JSON segments for path '{}' have different target data types: {} and {}",
                first.path, first.target_data_type, details.target_data_type,
            )));
        }
        targets.push(other);
    }
    super::SCALAR_INDEX_PLUGIN_REGISTRY
        .get_plugin_by_details(target)?
        .validate_new_segments_against_existing(&[], &targets)
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
    validate_merge_segments(&details)?;
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
        None => empty_value_row_id_stream(dataset, reference.fields[0])?,
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
    let merge = SegmentMerge::new(dataset, "JSON", &segments)?;
    let details = segments
        .iter()
        .filter_map(|segment| segment.index_details.as_deref())
        .collect::<Vec<_>>();
    validate_merge_segments(&details)?;

    // Delegate NGram's remap handling and source-version accounting to its lifecycle.
    if segments[0]
        .index_details
        .as_deref()
        .and_then(merge_target_type)
        == Some(IndexType::NGram)
    {
        return super::ngram::merge_segments(dataset, segments, staged).await;
    }
    merge.reject_legacy_segments_on_stable(dataset, "JSON")?;
    let (fragment_bitmap, old_data_filters) =
        merge.per_segment_filters(dataset, staged, true).await?;
    let (uuid, store) = merge.new_store(dataset)?;
    let created = open_and_merge_segments(
        dataset,
        merge.field_path(),
        merge.segments(),
        None,
        &store,
        &old_data_filters,
        staged,
    )
    .await?;
    Ok(merge.build_metadata(uuid, merge.dataset_version(), fragment_bitmap, created))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::path(
        "other",
        lance_index::pb::JsonTargetDataType::Int64,
        false,
        "path and target index type"
    )]
    #[case::value_type(
        "x",
        lance_index::pb::JsonTargetDataType::Utf8,
        false,
        "different target data types"
    )]
    #[case::target_type(
        "x",
        lance_index::pb::JsonTargetDataType::Int64,
        true,
        "path and target index type"
    )]
    fn test_json_rejects_incompatible_segments(
        #[case] path: &str,
        #[case] data_type: lance_index::pb::JsonTargetDataType,
        #[case] bitmap: bool,
        #[case] message: &str,
    ) {
        let first = lance_index::pb::JsonIndexDetails {
            path: "x".into(),
            target_data_type: lance_index::pb::JsonTargetDataType::Int64 as i32,
            target_details: Some(
                prost_types::Any::from_msg(&lance_index::pbold::BTreeIndexDetails::default())
                    .unwrap(),
            ),
            target_index_version: Some(0),
        };
        let mut other = first.clone();
        other.path = path.into();
        other.target_data_type = data_type as i32;
        if bitmap {
            other.target_details = Some(
                prost_types::Any::from_msg(&lance_index::pbold::BitmapIndexDetails::default())
                    .unwrap(),
            );
        }
        let first = prost_types::Any::from_msg(&first).unwrap();
        let other = prost_types::Any::from_msg(&other).unwrap();
        let error = validate_merge_segments(&[&first, &other]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message), "{error}");
    }
}
