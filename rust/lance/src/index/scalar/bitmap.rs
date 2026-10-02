// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use datafusion::execution::SendableRecordBatchStream;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::bitmap::BitmapIndex;
use lance_index::scalar::index_files_to_table;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, OldIndexDataFilter};
use lance_table::format::IndexMetadata;
use std::sync::Arc;
use uuid::Uuid;

use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

/// Open the given bitmap `segments` and k-way merge their postings, together
/// with `new_data`, into a single bitmap index written to `new_store`.
pub(in crate::index) async fn open_and_merge_segments(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    new_data: SendableRecordBatchStream,
    new_store: &LanceIndexStore,
    old_data_filters: &[Option<OldIndexDataFilter>],
) -> Result<CreatedIndex> {
    let mut source_indices = Vec::with_capacity(segments.len());
    for &segment in segments {
        let scalar_index =
            super::open_scalar_index(dataset, field_path, segment, &NoOpMetricsCollector).await?;
        let bitmap_index = scalar_index
            .as_any()
            .downcast_ref::<BitmapIndex>()
            .ok_or_else(|| {
                Error::index(format!(
                    "Bitmap merge: expected bitmap segment {}, got {:?}",
                    segment.uuid,
                    scalar_index.index_type()
                ))
            })?;
        source_indices.push(Arc::new(bitmap_index.clone()));
    }
    BitmapIndex::merge_segments(&source_indices, new_data, new_store, old_data_filters).await
}

/// Merge one caller-defined group of source bitmap segments into a single segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
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

    let segment_refs: Vec<&IndexMetadata> = segments.iter().collect();
    let (fragment_bitmap, old_data_filters) =
        crate::index::append::build_per_segment_filters(dataset, &segment_refs).await?;

    let new_uuid = Uuid::new_v4();
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid)?;
    // Pure segment consolidation: no dataset scan, so `new_data` is an empty stream.
    let empty_new_data = super::btree::empty_value_row_id_stream(dataset, field_id)?;
    let created_index = open_and_merge_segments(
        dataset,
        &field_path,
        &segment_refs,
        empty_new_data,
        &new_store,
        &old_data_filters,
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
        ..segments[0].clone()
    })
}
