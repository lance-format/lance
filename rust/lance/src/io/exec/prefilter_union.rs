// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bound an exact scalar selection to its fragments before adding uncovered rows.

use std::sync::Arc;

use datafusion::{
    error::{DataFusionError, Result},
    execution::TaskContext,
    physical_plan::{
        DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
        coalesce_partitions::CoalescePartitionsExec,
        execution_plan::{Boundedness, EmissionType},
        metrics::{ExecutionPlanMetricsSet, MetricsSet},
    },
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::{StreamExt, TryStreamExt};
use lance_core::{Error, utils::address::RowAddress};
use lance_index::prefilter::FilterLoader;
use lance_select::{
    IndexExprResult, RowAddrMask, RowAddrTreeMap, result::IndexExprResultWireFormat,
};
use roaring::RoaringBitmap;

use super::utils::{FilteredRowIdsToPrefilter, InstrumentedRecordBatchStreamAdapter};
use crate::{Dataset, index::prefilter::DatasetPreFilter};

/// Bound an exact scalar mask to its current fragment membership and union it
/// with matching row IDs from fragments missing scalar-index coverage.
///
/// External physical-plan codecs can serialize the dataset snapshot, fragment
/// scopes and wire format through [`Self::dataset`], [`Self::fragment_scope`],
/// [`Self::fragments_covered`] and [`Self::result_format`]. Encode the execution
/// children recursively, then reconstruct the node with [`Self::new`].
#[derive(Debug)]
pub struct PrefilterUnionExec {
    dataset: Arc<Dataset>,
    scalar: Arc<dyn ExecutionPlan>,
    missing: Option<Arc<dyn ExecutionPlan>>,
    scope: RoaringBitmap,
    coverage: RoaringBitmap,
    format: IndexExprResultWireFormat,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl PrefilterUnionExec {
    /// Construct a scoped mask source, coalescing each input to one partition.
    ///
    /// `scalar` must emit one exact [`IndexExprResult`] batch in `format`.
    /// `missing`, when present, emits `_rowid` values from the remaining evaluated
    /// fragments. `scope` bounds the scalar mask; `coverage` records every
    /// evaluated fragment, including the missing branch. Both children must use
    /// the dataset's `_rowid` domain and the same immutable dataset snapshot.
    /// Child order for codecs is scalar first, followed by the optional row-ID input.
    /// An empty `scope` removes all scalar matches: without `missing` the result
    /// is an exact empty mask; with `missing` its row-ID matches are retained.
    /// The scalar child is still executed and validated in either case.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use datafusion::physical_plan::ExecutionPlan;
    /// # use lance::{Dataset, io::exec::PrefilterUnionExec};
    /// # use lance_select::result::IndexExprResultWireFormat;
    /// # fn scoped(dataset: Arc<Dataset>, scalar: Arc<dyn ExecutionPlan>) {
    /// let source = PrefilterUnionExec::new(
    ///     dataset, scalar, None,
    ///     [1].into_iter().collect(), [1].into_iter().collect(),
    ///     IndexExprResultWireFormat::TwoMask,
    /// );
    /// assert!(source.fragment_scope().contains(1));
    /// # }
    /// ```
    pub fn new(
        dataset: Arc<Dataset>,
        scalar: Arc<dyn ExecutionPlan>,
        missing: Option<Arc<dyn ExecutionPlan>>,
        scope: RoaringBitmap,
        coverage: RoaringBitmap,
        format: IndexExprResultWireFormat,
    ) -> Self {
        let coalesce = |child: Arc<dyn ExecutionPlan>| {
            if child.properties().partitioning.partition_count() == 1 {
                child
            } else {
                Arc::new(CoalescePartitionsExec::new(child)) as Arc<dyn ExecutionPlan>
            }
        };
        let scalar = coalesce(scalar);
        let missing = missing.map(coalesce);
        Self {
            dataset,
            scalar,
            missing,
            scope,
            coverage,
            format,
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(format.schema().clone()),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// The immutable dataset snapshot whose row IDs the children produce.
    ///
    /// A codec must preserve its URI, version and storage configuration, or resolve
    /// the same snapshot from an external dataset registry before decoding.
    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    /// Current fragment IDs to which the scalar result must be restricted.
    pub fn fragment_scope(&self) -> &RoaringBitmap {
        &self.scope
    }

    /// Union of fragment IDs evaluated by the scalar and missing branches,
    /// including fragments with no matches.
    pub fn fragments_covered(&self) -> &RoaringBitmap {
        &self.coverage
    }

    /// Existing mask wire format used by the scalar input and combined output.
    pub fn result_format(&self) -> IndexExprResultWireFormat {
        self.format
    }

    async fn selection(
        dataset: Arc<Dataset>,
        scope: RoaringBitmap,
    ) -> lance_core::Result<Arc<RowAddrMask>> {
        if scope.is_empty() {
            return Ok(Arc::new(RowAddrMask::allow_nothing()));
        }
        let missing = &scope - dataset.fragment_bitmap.as_ref();
        if !missing.is_empty() {
            return Err(Error::internal(format!(
                "prefilter scope contains missing fragment_ids={missing:?}, dataset_version={}",
                dataset.version_id()
            )));
        }
        if dataset.manifest().uses_stable_row_ids() {
            // Stable IDs must be selected by their current fragment membership.
            return DatasetPreFilter::do_create_deletion_mask_row_id(dataset, Some(scope)).await;
        }
        let mut addresses = RowAddrTreeMap::new();
        for fragment in dataset
            .manifest()
            .fragments
            .iter()
            .filter(|fragment| scope.contains(fragment.id as u32))
        {
            let fragment_id = fragment.id as u32;
            let rows = fragment.physical_rows.ok_or_else(|| {
                Error::internal(format!(
                    "prefilter scope fragment_id={fragment_id} has no physical row count"
                ))
            })?;
            let start = u64::from(RowAddress::new_from_parts(fragment_id, 0));
            let rows = u32::try_from(rows).map_err(|_| Error::internal(format!(
                "prefilter scope fragment_id={fragment_id} physical_rows={rows} exceeds row-address capacity"
            )))?;
            addresses.insert_range(start..start + u64::from(rows));
        }
        Ok(Arc::new(RowAddrMask::from_allowed(addresses)))
    }
}

impl DisplayAs for PrefilterUnionExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "PrefilterUnion: scalar_fragments={}, evaluated_fragments={}",
            self.scope.len(),
            self.coverage.len()
        )
    }
}

impl ExecutionPlan for PrefilterUnionExec {
    fn name(&self) -> &str {
        "PrefilterUnionExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        let mut children = vec![&self.scalar];
        if let Some(missing) = &self.missing {
            children.push(missing);
        }
        children
    }
    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition; self.children().len()]
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != self.children().len() {
            return Err(DataFusionError::Plan(format!(
                "PrefilterUnionExec expected {} children, got {}",
                self.children().len(),
                children.len()
            )));
        }
        Ok(Arc::new(Self::new(
            self.dataset.clone(),
            children[0].clone(),
            children.get(1).cloned(),
            self.scope.clone(),
            self.coverage.clone(),
            self.format,
        )))
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Execution(format!(
                "PrefilterUnionExec requires partition=0, got {partition}"
            )));
        }
        let mut scalar = self.scalar.execute(0, context.clone())?;
        let missing = self
            .missing
            .as_ref()
            .map(|child| child.execute(0, context))
            .transpose()?;
        let selection = Self::selection(self.dataset.clone(), self.scope.clone());
        let coverage = self.coverage.clone();
        let format = self.format;
        let metrics = self.metrics.clone();
        let future = async move {
            let scalar_future = async {
                let batch = scalar.try_next().await?.ok_or_else(|| {
                    Error::internal("PrefilterUnionExec scalar child emitted no mask batch")
                })?;
                if batch.schema() != format.schema().clone() {
                    return Err(Error::internal(format!(
                        "PrefilterUnionExec scalar child schema {:?}, expected {:?}",
                        batch.schema(),
                        format.schema(),
                    )));
                }
                let (result, _) = IndexExprResult::deserialize(&batch)?;
                if !result.is_exact() {
                    return Err(Error::internal(
                        "PrefilterUnionExec scalar child returned an inexact mask",
                    ));
                }
                if scalar.try_next().await?.is_some() {
                    return Err(Error::internal(
                        "PrefilterUnionExec scalar child emitted more than one mask batch",
                    ));
                }
                Ok(result.upper)
            };
            let missing_future = async {
                match missing {
                    Some(stream) => {
                        Box::new(FilteredRowIdsToPrefilter::new(stream).with_metrics(&metrics, 0))
                            .load()
                            .await
                    }
                    None => Ok(RowAddrMask::allow_nothing()),
                }
            };
            // Start column and row-ID metadata I/O before cached scalar work,
            // which can complete its first poll without yielding.
            let (missing, selection, scalar) =
                futures::try_join!(missing_future, selection, scalar_future)?;
            let scalar = match scalar {
                RowAddrMask::AllowList(mut rows) => {
                    rows.mask(&selection);
                    RowAddrMask::from_allowed(rows)
                }
                // A complement needs the scope's allow-list to become a finite selection.
                RowAddrMask::BlockList(_) => scalar & selection.as_ref().clone(),
            };
            IndexExprResult::exact(scalar | missing)
                .serialize(&coverage, format)
                .map_err(DataFusionError::from)
        };
        let stream = futures::stream::once(future).boxed();
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            self.format.schema().clone(),
            stream,
            partition,
            &self.metrics,
        )))
    }
    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::record_batch;
    use datafusion::datasource::memory::MemorySourceConfig;
    use lance_core::ROW_ID;
    use lance_datagen::{array, gen_batch};
    use rstest::rstest;

    #[rstest]
    #[case::two_mask(IndexExprResultWireFormat::TwoMask)]
    #[case::three_variant(IndexExprResultWireFormat::ThreeVariant)]
    #[tokio::test]
    async fn scope_union_and_replacement(#[case] format: IndexExprResultWireFormat) {
        let dataset = Arc::new(
            gen_batch()
                .col("i", array::step::<arrow_array::types::Int32Type>())
                .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(3))
                .await
                .unwrap(),
        );
        let error = PrefilterUnionExec::selection(dataset.clone(), [9].into_iter().collect())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(error.to_string().contains("missing fragment_ids"));
        assert!(error.to_string().contains('9'));
        assert!(
            error
                .to_string()
                .contains(&format!("dataset_version={}", dataset.version_id()))
        );
        // A complement must be bounded to physical rows before union, including when
        // the segment also contains excluded fragments.
        let blocked = RowAddrTreeMap::from_iter([0]);
        let batch = IndexExprResult::exact(RowAddrMask::from_block(blocked))
            .serialize(&[0, 1].into_iter().collect(), format)
            .unwrap();
        let scalar = MemorySourceConfig::try_new_exec(
            &[vec![batch.clone()], vec![batch.clone()]],
            format.schema().clone(),
            None,
        )
        .unwrap();
        let missing_batch = record_batch!((
            ROW_ID,
            UInt64,
            [u64::from(RowAddress::new_from_parts(1, 2))]
        ))
        .unwrap();
        let missing = MemorySourceConfig::try_new_exec(
            &[vec![], vec![missing_batch.clone()]],
            missing_batch.schema(),
            None,
        )
        .unwrap();
        // Multiple scalar batches violate the mask contract even after coalescing.
        let invalid = PrefilterUnionExec::new(
            dataset.clone(),
            scalar,
            Some(missing.clone()),
            [0].into_iter().collect(),
            [0, 1].into_iter().collect(),
            format,
        );
        let error = invalid
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(matches!(error, DataFusionError::External(_)));
        assert!(error.to_string().contains("more than one mask batch"));
        let inexact =
            IndexExprResult::at_most(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([1])))
                .serialize(&[0].into_iter().collect(), format)
                .unwrap();
        let wrong_schema = record_batch!(("wrong", UInt64, [1_u64])).unwrap();
        for (batches, schema, message) in [
            (vec![], format.schema().clone(), "no mask batch"),
            (vec![inexact], format.schema().clone(), "inexact mask"),
            (
                vec![wrong_schema.clone()],
                wrong_schema.schema(),
                "scalar child schema",
            ),
        ] {
            let scalar = MemorySourceConfig::try_new_exec(&[batches], schema, None).unwrap();
            let invalid = PrefilterUnionExec::new(
                dataset.clone(),
                scalar,
                None,
                [0].into_iter().collect(),
                [0].into_iter().collect(),
                format,
            );
            let error = invalid
                .execute(0, Arc::new(TaskContext::default()))
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap_err();
            assert!(matches!(error, DataFusionError::External(_)));
            assert!(error.to_string().contains(message), "{error}");
        }
        let empty_mask = IndexExprResult::exact(RowAddrMask::allow_nothing())
            .serialize(&[0].into_iter().collect(), format)
            .unwrap();
        let scalar =
            MemorySourceConfig::try_new_exec(&[vec![empty_mask]], format.schema().clone(), None)
                .unwrap();
        let invalid_missing = MemorySourceConfig::try_new_exec(
            &[vec![wrong_schema.clone()]],
            wrong_schema.schema(),
            None,
        )
        .unwrap();
        let invalid = PrefilterUnionExec::new(
            dataset.clone(),
            scalar,
            Some(invalid_missing),
            [0].into_iter().collect(),
            [0, 1].into_iter().collect(),
            format,
        );
        let error = invalid
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(matches!(error, DataFusionError::External(_)));
        assert!(
            error.to_string().contains("missing row id column"),
            "{error}"
        );
        let scalar =
            MemorySourceConfig::try_new_exec(&[vec![], vec![batch]], format.schema().clone(), None)
                .unwrap();
        let plan = Arc::new(PrefilterUnionExec::new(
            dataset,
            scalar,
            Some(missing),
            [0].into_iter().collect(),
            [0, 1].into_iter().collect(),
            format,
        ));
        let rewritten = plan
            .clone()
            .with_new_children(plan.children().into_iter().cloned().collect())
            .unwrap();
        let error = plan.clone().with_new_children(vec![]).unwrap_err();
        assert!(matches!(error, DataFusionError::Plan(_)));
        assert!(error.to_string().contains("expected 2 children"));
        for plan in [plan as Arc<dyn ExecutionPlan>, rewritten] {
            for _ in 0..2 {
                let batches = plan
                    .execute(0, Arc::new(TaskContext::default()))
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
                assert_eq!(batches.len(), 1);
                assert_eq!(batches[0].schema(), format.schema().clone());
                let (result, coverage) = IndexExprResult::deserialize(&batches[0]).unwrap();
                assert!(result.is_exact());
                assert_eq!(coverage, [0, 1].into_iter().collect());
                assert_eq!(
                    result.upper,
                    RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([
                        1,
                        2,
                        RowAddress::new_from_parts(1, 2).into(),
                    ]))
                );
            }
        }
    }
}
