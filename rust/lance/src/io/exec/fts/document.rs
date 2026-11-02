// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Execution node that normalizes schema-derived FTS documents.

use std::sync::Arc;

use crate::index::scalar::inverted::{
    ResolvedFtsField, fts_document_schema, transform_fts_document_stream,
};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::ExecutionPlanProperties;

/// Expands a schema-derived nested FTS source into one canonical row per
/// logical document before flat search or index building consumes it.
#[derive(Debug)]
pub struct FtsDocumentExec {
    input: Arc<dyn ExecutionPlan>,
    resolved: ResolvedFtsField,
    properties: Arc<PlanProperties>,
}

impl FtsDocumentExec {
    pub(crate) fn new(input: Arc<dyn ExecutionPlan>, resolved: ResolvedFtsField) -> Self {
        let schema = fts_document_schema(resolved.coordinate_rank());
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            input.output_partitioning().clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            input,
            resolved,
            properties,
        }
    }
}

impl DisplayAs for FtsDocumentExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "FtsDocument: column={}, granularity={:?}",
            self.resolved.canonical_path, self.resolved.document_granularity
        )
    }
}

impl ExecutionPlan for FtsDocumentExec {
    fn name(&self) -> &str {
        "FtsDocumentExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "FtsDocumentExec expects one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children.pop().unwrap(),
            self.resolved.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        transform_fts_document_stream(
            self.input.execute(partition, context)?,
            self.resolved.clone(),
        )
        .map_err(DataFusionError::from)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
}
