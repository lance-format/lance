// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Scaffolding shared by scalar index segment merges.
//!
//! A merge reads existing segments, drops rows its caller no longer keeps, and
//! writes one new segment with the same declaration. Those steps are the same
//! for every scalar index type; only the algorithm that combines the source
//! payloads differs. [`SegmentMerge`] owns the shared inputs and output shape so
//! each index module only describes its own merge.

use std::sync::Arc;

use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use lance_core::ROW_ADDR;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use lance_index::scalar::{
    CreatedIndex, OldIndexDataFilter, ScalarIndex, TrainingCriteria, index_files_to_table,
};
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::{TrainingScanOptions, scan_training_data};
use crate::dataset::index::LanceIndexStoreExt;
use crate::index::frag_reuse::{OpenPurpose, SegmentRemappingPlan, StagedRemappingPlans};
use crate::{Dataset, Error, Result};

/// One caller-defined group of source segments being merged into a single
/// segment, with the inputs every scalar merge shares.
pub(in crate::index) struct SegmentMerge<'a> {
    segments: Vec<&'a IndexMetadata>,
    field_id: i32,
    field_path: String,
    dataset_version: u64,
}

impl<'a> SegmentMerge<'a> {
    /// Collect the shared inputs of a merge, rejecting a group that cannot
    /// produce one coherent segment.
    pub(in crate::index) fn new(
        dataset: &Dataset,
        index_type: &str,
        segments: &'a [IndexMetadata],
    ) -> Result<Self> {
        let Some(first) = segments.first() else {
            return Err(Error::index("No segment metadata was provided".to_string()));
        };
        let field_id = *first.fields.first().ok_or_else(|| {
            Error::invalid_input(format!(
                "CreateIndex: segment {} is missing field ids",
                first.uuid
            ))
        })?;
        let field_path = dataset.schema().field_path(field_id)?;
        let segment_refs = segments.iter().collect::<Vec<_>>();
        for segment in segment_refs.iter().skip(1) {
            if !segment.declares_same_fields(first) {
                return Err(Error::invalid_input(format!(
                    "{index_type} merge_segments: segment {} has fields {:?}, expected {:?}",
                    segment.uuid, segment.fields, first.fields,
                )));
            }
        }
        Ok(Self {
            segments: segment_refs,
            field_id,
            field_path,
            dataset_version: dataset.manifest.version,
        })
    }

    pub(in crate::index) fn segments(&self) -> &[&'a IndexMetadata] {
        &self.segments
    }

    pub(in crate::index) fn field_id(&self) -> i32 {
        self.field_id
    }

    pub(in crate::index) fn field_path(&self) -> &str {
        &self.field_path
    }

    /// Refuse a pre-migration segment on a stable-row-id table. Without stable
    /// row ids the two identifiers are the same numbering, so such a segment
    /// remains valid input there.
    pub(in crate::index) fn reject_legacy_segments_on_stable(
        &self,
        dataset: &Dataset,
        index_type: &str,
    ) -> Result<()> {
        if !dataset.manifest.uses_stable_row_ids() {
            return Ok(());
        }
        if let Some(segment) = self
            .segments
            .iter()
            .find(|segment| !segment.results_are_row_addrs())
        {
            return Err(Error::invalid_input(format!(
                "{index_type} merge_segments: segment {} predates row-address-domain support \
                 and cannot be merged on a dataset with stable row IDs; rebuild it first (e.g. \
                 with create_index(..., replace: true))",
                segment.uuid,
            )));
        }
        Ok(())
    }

    /// Staged coverage has already been projected and checked by the merge entry point.
    pub(in crate::index) async fn staged_row_inputs(
        &self,
        dataset: &Dataset,
        refresh: &super::merge::MergeRefresh,
        criteria: TrainingCriteria,
    ) -> Result<(
        RoaringBitmap,
        Vec<Option<OldIndexDataFilter>>,
        Option<SendableRecordBatchStream>,
    )> {
        let (coverage, filters, selection) = refresh
            .row_merge_inputs(
                dataset,
                self.segments(),
                None,
                &[],
                criteria.needs_row_addrs,
            )
            .await?;
        let stream = if refresh.is_empty() {
            None
        } else {
            Some(
                scan_training_data(
                    dataset,
                    self.field_path(),
                    &criteria,
                    Some(selection),
                    TrainingScanOptions::default(),
                )
                .await?,
            )
        };
        Ok((coverage, filters, stream))
    }

    /// Union of the sources' stored coverage, requiring each source to record it.
    pub(in crate::index) fn stored_coverage_union(&self) -> Result<RoaringBitmap> {
        let mut coverage = RoaringBitmap::new();
        for segment in &self.segments {
            coverage |= segment.fragment_bitmap.as_ref().cloned().ok_or_else(|| {
                Error::invalid_input(format!(
                    "CreateIndex: segment {} is missing fragment coverage",
                    segment.uuid
                ))
            })?;
        }
        Ok(coverage)
    }

    pub(in crate::index) fn new_store(&self, dataset: &Dataset) -> Result<(Uuid, LanceIndexStore)> {
        let uuid = Uuid::new_v4();
        let store = LanceIndexStore::from_dataset_for_new(dataset, &uuid)?;
        Ok((uuid, store))
    }

    /// The committed metadata of the new segment, named and declared like its
    /// sources.
    pub(in crate::index) fn build_metadata(
        &self,
        uuid: Uuid,
        fragment_bitmap: RoaringBitmap,
        created_index: CreatedIndex,
    ) -> IndexMetadata {
        let template = self.segments[0];
        IndexMetadata {
            uuid,
            name: template.name.clone(),
            fields: template.fields.clone(),
            covering_fields: template.covering_fields.clone(),
            dataset_version: self.dataset_version,
            fragment_bitmap: Some(fragment_bitmap),
            index_details: Some(Arc::new(created_index.index_details)),
            index_version: created_index.index_version as i32,
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            files: Some(index_files_to_table(created_index.files)),
            clustering_id: None,
            metadata: Default::default(),
        }
    }
}

/// Open one source segment of a merge, without assuming a concrete type.
pub(in crate::index) async fn open_dyn_source(
    dataset: &Dataset,
    field_path: &str,
    segment: &IndexMetadata,
    plan: Option<&SegmentRemappingPlan>,
) -> Result<Arc<dyn ScalarIndex>> {
    super::open_scalar_index_with_plan(
        dataset,
        field_path,
        segment,
        plan,
        OpenPurpose::Maintenance,
        &NoOpMetricsCollector,
    )
    .await
}

/// Open every source segment of a merge, without assuming a concrete type.
pub(in crate::index) async fn open_dyn_sources(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    staged: Option<&StagedRemappingPlans>,
) -> Result<Vec<Arc<dyn ScalarIndex>>> {
    let mut sources = Vec::with_capacity(segments.len());
    for &segment in segments {
        sources.push(
            open_dyn_source(
                dataset,
                field_path,
                segment,
                staged.and_then(|plans| plans.get(&segment.uuid)),
            )
            .await?,
        );
    }
    Ok(sources)
}

/// The live fragments of each segment's coverage, the zones a merge keeps.
pub(in crate::index) fn effective_coverage(
    dataset: &Dataset,
    segments: &[&IndexMetadata],
) -> Result<Vec<RoaringBitmap>> {
    let dataset_fragments = dataset.table_state.fragment_bitmap.as_ref();
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

/// One opened source as its concrete type.
pub(in crate::index) fn downcast_source<'a, T>(
    source: &'a Arc<dyn ScalarIndex>,
    segment: &IndexMetadata,
    index_type: &str,
) -> Result<&'a T>
where
    T: ScalarIndex + 'static,
{
    source.as_any().downcast_ref::<T>().ok_or_else(|| {
        Error::index(format!(
            "merge_existing_index_segments: expected {} segment {}, got {:?}",
            index_type,
            segment.uuid,
            source.index_type()
        ))
    })
}

/// The opened sources as their concrete type.
pub(in crate::index) fn downcast_sources<'a, T>(
    sources: &'a [Arc<dyn ScalarIndex>],
    segments: &[&IndexMetadata],
    index_type: &str,
) -> Result<Vec<&'a T>>
where
    T: ScalarIndex + 'static,
{
    sources
        .iter()
        .zip(segments)
        .map(|(source, segment)| downcast_source::<T>(source, segment, index_type))
        .collect()
}

/// Open every source segment of a merge, cloning the concrete index.
pub(in crate::index) async fn open_sources<T>(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    staged: Option<&StagedRemappingPlans>,
    index_type: &str,
) -> Result<Vec<Arc<T>>>
where
    T: ScalarIndex + Clone + 'static,
{
    let mut sources = Vec::with_capacity(segments.len());
    for &segment in segments {
        sources.push(
            open_source(
                dataset,
                field_path,
                segment,
                staged.and_then(|plans| plans.get(&segment.uuid)),
                index_type,
            )
            .await?,
        );
    }
    Ok(sources)
}

/// Open one source segment of a merge, cloning the concrete index.
pub(in crate::index) async fn open_source<T>(
    dataset: &Dataset,
    field_path: &str,
    segment: &IndexMetadata,
    plan: Option<&SegmentRemappingPlan>,
    index_type: &str,
) -> Result<Arc<T>>
where
    T: ScalarIndex + Clone + 'static,
{
    let opened = open_dyn_source(dataset, field_path, segment, plan).await?;
    let source = downcast_source::<T>(&opened, segment, index_type)?;
    Ok(Arc::new(source.clone()))
}

/// A row-empty value-and-address stream for merges that have no new data.
pub(in crate::index) fn empty_value_row_id_stream(
    dataset: &Dataset,
    field_id: i32,
) -> Result<SendableRecordBatchStream> {
    let field = dataset.schema().field_by_id(field_id).ok_or_else(|| {
        Error::invalid_input(format!(
            "merge_existing_index_segments: field id {} does not exist",
            field_id
        ))
    })?;
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(VALUE_COLUMN_NAME, field.data_type(), true),
        ArrowField::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::empty(),
    )))
}
