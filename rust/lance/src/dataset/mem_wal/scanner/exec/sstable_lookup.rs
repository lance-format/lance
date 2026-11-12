// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Resolve a flushed primary key before reading its projected payload.

use std::fmt;
use std::sync::Arc;

use arrow_array::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use lance_core::datatypes::Schema;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_datafusion::projection::ProjectionPlan;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::{SargableQuery, SearchResult};

use crate::Dataset;
use crate::dataset::mem_wal::scanner::block_list::{load_pk_index, on_disk_pk_key};
use crate::dataset::take::TakeBuilder;

/// A lazy PK-sidecar probe followed by a take of the first live matching row.
/// Kept below the bloom guard and newest-first coalesce so skipped generations
/// perform neither index probes nor payload reads.
#[derive(Debug)]
pub(in crate::dataset::mem_wal::scanner) struct SsTablePointLookupExec {
    dataset: Arc<Dataset>,
    key: ScalarValue,
    projection: Arc<ProjectionPlan>,
    properties: Arc<PlanProperties>,
}

impl SsTablePointLookupExec {
    pub fn try_new(
        dataset: Arc<Dataset>,
        values: &[ScalarValue],
        projection: Schema,
    ) -> Result<Self> {
        // SQL equality never matches a NULL, including a component of a tuple.
        let key = if values.iter().any(ScalarValue::is_null) {
            ScalarValue::Null
        } else {
            on_disk_pk_key(values)?
        };
        let projection = Arc::new(ProjectionPlan::from_schema(dataset.clone(), &projection)?);
        let schema = Arc::new(projection.output_schema()?);
        Ok(Self {
            dataset,
            key,
            projection,
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(schema),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Incremental,
                Boundedness::Bounded,
            )),
        })
    }
}

impl DisplayAs for SsTablePointLookupExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SsTablePointLookupExec: key={}", self.key)
    }
}

impl ExecutionPlan for SsTablePointLookupExec {
    fn name(&self) -> &str {
        "SsTablePointLookupExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DFResult<TreeNodeRecursion>,
    ) -> DFResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "SsTablePointLookupExec requires no children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "SsTablePointLookupExec expected partition 0, got {partition}"
            )));
        }
        let dataset = self.dataset.clone();
        let key = self.key.clone();
        let projection = self.projection.clone();
        let schema = self.schema();
        // Stop stream adapters' Send proofs at a trait object rather than
        // repeatedly traversing the dataset take's nested future types.
        let lookup: futures::future::BoxFuture<'static, DFResult<RecordBatch>> =
            Box::pin(async move {
                take_matching_row(dataset, key, projection)
                    .await
                    .map_err(Into::into)
            });
        let stream = futures::stream::once(lookup);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

async fn take_matching_row(
    dataset: Arc<Dataset>,
    key: ScalarValue,
    projection: Arc<ProjectionPlan>,
) -> Result<RecordBatch> {
    let empty = || {
        projection
            .output_schema()
            .map(|s| RecordBatch::new_empty(Arc::new(s)))
    };
    if key.is_null() {
        return empty();
    }
    let index = load_pk_index(&dataset).await?;
    let SearchResult::Exact(matches) = index
        .search(&SargableQuery::Equals(key), &NoOpMetricsCollector)
        .await?
    else {
        return Err(Error::internal(
            "SSTable PK BTree returned an inexact result",
        ));
    };
    let matches = matches.true_rows();
    let mut positions = matches
        .row_addrs()
        .ok_or_else(|| Error::internal("SSTable PK BTree returned unbounded row positions"))?
        .map(u64::from)
        .peekable();

    // The sidecar holds global physical positions, not logical offsets (which
    // exclude superseded versions) or dataset row IDs. Translate using physical
    // fragment lengths, and skip deleted versions before reading any payload.
    let mut fragment_start = 0u64;
    for fragment in dataset.get_fragments() {
        if positions.peek().is_none() {
            return empty();
        }
        let fragment_end = fragment_start
            .checked_add(fragment.physical_rows().await? as u64)
            .ok_or_else(|| Error::internal("SSTable physical row count overflow"))?;
        if positions.peek().is_some_and(|p| *p < fragment_end) {
            let deleted = fragment.get_deletion_vector().await?;
            while let Some(position) = positions.next_if(|p| *p < fragment_end) {
                let offset = u32::try_from(position - fragment_start)
                    .map_err(|e| Error::internal(format!("SSTable row offset: {e}")))?;
                if deleted.as_ref().is_some_and(|d| d.contains(offset)) {
                    continue;
                }
                let fragment_id = u32::try_from(fragment.id())
                    .map_err(|e| Error::internal(format!("SSTable fragment id: {e}")))?;
                let address = RowAddress::new_from_parts(fragment_id, offset);
                return TakeBuilder::try_new_from_addresses(
                    dataset.clone(),
                    vec![u64::from(address)],
                    projection,
                )?
                .execute()
                .await;
            }
        }
        fragment_start = fragment_end;
    }
    if let Some(position) = positions.next() {
        return Err(Error::corrupt_file(
            dataset.base.clone(),
            format!("SSTable PK row position {position} exceeds {fragment_start} physical rows"),
        ));
    }
    empty()
}
