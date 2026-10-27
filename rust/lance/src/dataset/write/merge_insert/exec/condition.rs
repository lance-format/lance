// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{
    Array, RecordBatch, RecordBatchOptions, UInt8Array, UInt32Array,
    cast::AsArray,
    types::{UInt8Type, UInt64Type},
};
use arrow_schema::{DataType, SchemaRef};
use arrow_select::take::take;
use datafusion::common::{
    Column, DFSchema, DataFusionError, Result as DFResult, TableReference,
    tree_node::{Transformed, TransformedResult, TreeNode},
};
use datafusion::execution::{SendableRecordBatchStream, SessionState, TaskContext};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PhysicalExpr, PlanProperties,
    metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet},
    stream::RecordBatchStreamAdapter,
};
use datafusion_expr::Expr;
use futures::StreamExt;
use lance_core::ROW_ADDR;
use lance_datafusion::projection::ProjectionPlan;

use crate::Dataset;
use crate::dataset::TakeBuilder;
use crate::dataset::write::merge_insert::assign_action::Action;
use crate::dataset::write::merge_insert::{
    MERGE_ACTION_COLUMN, MERGE_SOURCE_SENTINEL, target_blob_handling,
};

/// Evaluates merge conditions over target columns on the rows they gate.
///
/// The join marks every matched row `Action::UpdateAll` and every target-only
/// row `Action::Delete`; this node takes the conditions' target columns by row
/// address for those rows alone and demotes the rejected ones to
/// `Action::Nothing`, so the target scan never reads them.
#[derive(Debug)]
pub struct MergeConditionExec {
    input: Arc<dyn ExecutionPlan>,
    evaluator: Arc<Evaluator>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

#[derive(Debug)]
struct Evaluator {
    dataset: Arc<Dataset>,
    /// Gates `UpdateAll` on the matched rows.
    update_if: Option<Arc<dyn PhysicalExpr>>,
    /// Gates `Delete` on the target-only rows.
    delete_if: Option<Arc<dyn PhysicalExpr>>,
    /// The input columns the conditions read, followed by the taken ones.
    eval_schema: SchemaRef,
    eval_inputs: Vec<usize>,
    take: Option<Arc<ProjectionPlan>>,
    action_idx: usize,
    rowaddr_idx: usize,
    sentinel_idx: usize,
}

impl MergeConditionExec {
    /// `input_schema` is the logical schema of `input`, which carries the
    /// `source` / `target` qualifiers the conditions are resolved against.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        input_schema: &DFSchema,
        dataset: Arc<Dataset>,
        update_if: Option<Expr>,
        delete_if: Option<Expr>,
        session: &SessionState,
    ) -> DFResult<Self> {
        let physical_schema = input.schema();
        let physical_names = physical_schema.fields().iter().map(|f| f.name());
        if !physical_names.eq(input_schema.fields().iter().map(|f| f.name())) {
            return Err(DataFusionError::Internal(format!(
                "merge insert input {:?} does not match its logical schema {:?}",
                physical_schema.fields(),
                input_schema.fields()
            )));
        }
        let index_of = |name: &str| {
            physical_schema
                .column_with_name(name)
                .map(|(i, _)| i)
                .ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "Expected {name} column in merge insert input"
                    ))
                })
        };
        let action_idx = index_of(MERGE_ACTION_COLUMN)?;
        let rowaddr_idx = index_of(ROW_ADDR)?;
        let sentinel_idx = index_of(MERGE_SOURCE_SENTINEL)?;

        let mut to_take = HashSet::new();
        let update_if = update_if
            .map(|condition| {
                resolve_target_columns(condition, input_schema, &dataset, &mut to_take)
            })
            .transpose()?;
        let delete_if = delete_if
            .map(|condition| {
                resolve_target_columns(condition, input_schema, &dataset, &mut to_take)
            })
            .transpose()?;
        let columns = dataset
            .schema()
            .fields
            .iter()
            .map(|field| field.name.as_str())
            .filter(|name| to_take.contains(*name))
            .collect::<Vec<_>>();
        // Every column may already be in the input, in which case the node
        // only gates; see `resolve_target_columns`.
        let take = match columns.is_empty() {
            true => None,
            false => {
                let mut take = ProjectionPlan::from_schema(
                    dataset.clone(),
                    &dataset.schema().project(&columns)?,
                )?;
                take.physical_projection = take
                    .physical_projection
                    .clone()
                    .with_blob_handling(target_blob_handling(&dataset));
                Some(Arc::new(take))
            }
        };
        // Only the input columns the conditions read are copied for evaluation.
        let mut eval_inputs = [&update_if, &delete_if]
            .into_iter()
            .flatten()
            .flat_map(|condition| condition.column_refs())
            .filter_map(|column| input_schema.index_of_column(column).ok())
            .collect::<Vec<_>>();
        eval_inputs.sort_unstable();
        eval_inputs.dedup();
        let read = DFSchema::new_with_metadata(
            eval_inputs
                .iter()
                .map(|&i| {
                    let (qualifier, field) = input_schema.qualified_field(i);
                    (qualifier.cloned(), field.clone())
                })
                .collect(),
            Default::default(),
        )?;
        let eval_df_schema = match &take {
            Some(take) => read.join(&DFSchema::try_from_qualified_schema(
                "target",
                &take.output_schema()?,
            )?)?,
            None => read,
        };
        let eval_schema: SchemaRef = Arc::new(eval_df_schema.as_arrow().clone());
        let physical = |condition: Expr, clause: &str| -> DFResult<Arc<dyn PhysicalExpr>> {
            let condition = session.create_physical_expr(condition, &eval_df_schema)?;
            let data_type = condition.data_type(eval_schema.as_ref())?;
            if data_type != DataType::Boolean {
                return Err(DataFusionError::Plan(format!(
                    "Merge insert conditions must be expressions that return a boolean value, \
                     received a '{clause}' expression ({condition}) which has data type {data_type}"
                )));
            }
            Ok(condition)
        };
        let update_if = update_if
            .map(|condition| physical(condition, "when matched update if"))
            .transpose()?;
        let delete_if = delete_if
            .map(|condition| physical(condition, "when not matched by source delete if"))
            .transpose()?;

        Ok(Self {
            properties: input.properties().clone(),
            evaluator: Arc::new(Evaluator {
                dataset,
                update_if,
                delete_if,
                eval_schema,
                eval_inputs,
                take,
                action_idx,
                rowaddr_idx,
                sentinel_idx,
            }),
            input,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

/// Binds each target column the condition reads to the input when the input
/// already carries it, and otherwise adds it to `to_take`.
///
/// A whole-row rewrite fills the columns the source omits from the target
/// side of the join as unqualified columns; a `target.`-qualified reference
/// to one is rewritten to that column, since a taken `target.x` next to a
/// filled `x` would be an ambiguous schema.
fn resolve_target_columns(
    condition: Expr,
    input_schema: &DFSchema,
    dataset: &Dataset,
    to_take: &mut HashSet<String>,
) -> DFResult<Expr> {
    let target = TableReference::bare("target");
    let dataset_schema = dataset.schema();
    condition
        .transform(|expr| {
            let Expr::Column(column) = &expr else {
                return Ok(Transformed::no(expr));
            };
            if dataset_schema.field(&column.name).is_none()
                || column
                    .relation
                    .as_ref()
                    .is_some_and(|relation| relation != &target)
            {
                return Ok(Transformed::no(expr));
            }
            let input_has = |qualifier: Option<&TableReference>| {
                input_schema
                    .iter()
                    .any(|(q, field)| q == qualifier && field.name() == &column.name)
            };
            if input_has(Some(&target)) {
                return Ok(Transformed::no(expr));
            }
            if input_has(None) {
                return Ok(Transformed::yes(Expr::Column(Column::new_unqualified(
                    &column.name,
                ))));
            }
            to_take.insert(column.name.clone());
            Ok(Transformed::no(expr))
        })
        .data()
}

impl Evaluator {
    fn taken_columns(&self) -> impl Iterator<Item = &str> {
        self.eval_schema.fields()[self.eval_inputs.len()..]
            .iter()
            .map(|field| field.name().as_str())
    }

    async fn apply(&self, batch: RecordBatch) -> DFResult<RecordBatch> {
        let actions = batch
            .column(self.action_idx)
            .as_primitive_opt::<UInt8Type>()
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "Expected UInt8Array for {MERGE_ACTION_COLUMN} column"
                ))
            })?;
        let sentinel = batch.column(self.sentinel_idx);
        let rows_with = |action: Action, target_only: bool| -> Vec<u32> {
            (0..batch.num_rows())
                .filter(|&row| {
                    actions.value(row) == action as u8 && (!target_only || sentinel.is_null(row))
                })
                .map(|row| row as u32)
                .collect()
        };
        // `WhenMatched::Delete` marks matched rows `Delete` unconditionally;
        // `DeleteIf` gates only the target-only rows, which the join left
        // without a source sentinel.
        let update_rows = match self.update_if {
            Some(_) => rows_with(Action::UpdateAll, false),
            None => Vec::new(),
        };
        let delete_rows = match self.delete_if {
            Some(_) => rows_with(Action::Delete, true),
            None => Vec::new(),
        };
        let candidates = [update_rows.as_slice(), delete_rows.as_slice()].concat();
        if candidates.is_empty() {
            return Ok(batch);
        }

        let row_addrs = batch
            .column(self.rowaddr_idx)
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(|| {
                DataFusionError::Internal(format!("Expected UInt64Array for {ROW_ADDR} column"))
            })?;
        let addrs = candidates
            .iter()
            .map(|&row| {
                if row_addrs.is_null(row as usize) {
                    Err(DataFusionError::Internal(
                        "Merge insert matched a row without a row address".to_string(),
                    ))
                } else {
                    Ok(row_addrs.value(row as usize))
                }
            })
            .collect::<DFResult<Vec<_>>>()?;
        let indices = UInt32Array::from(candidates.clone());
        let mut columns = self
            .eval_inputs
            .iter()
            .map(|&i| take(batch.column(i), &indices, None))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(take) = &self.take {
            let taken =
                TakeBuilder::try_new_from_addresses(self.dataset.clone(), addrs, take.clone())?
                    .execute()
                    .await?;
            if taken.num_rows() != candidates.len() {
                return Err(DataFusionError::Internal(format!(
                    "Took {} of {} rows for the merge insert conditions",
                    taken.num_rows(),
                    candidates.len()
                )));
            }
            for name in self.taken_columns() {
                let column = taken.column_by_name(name).ok_or_else(|| {
                    DataFusionError::Internal(format!("Take did not return column {name}"))
                })?;
                columns.push(column.clone());
            }
        }
        let eval_batch = RecordBatch::try_new_with_options(
            self.eval_schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(candidates.len())),
        )?;

        let mut new_actions = actions.values().to_vec();
        let mut reject_failing =
            |condition: &Arc<dyn PhysicalExpr>, offset: usize, rows: &[u32]| -> DFResult<()> {
                let eval_batch = eval_batch.slice(offset, rows.len());
                let verdict = condition
                    .evaluate(&eval_batch)?
                    .into_array(eval_batch.num_rows())?;
                let verdict = verdict.as_boolean();
                for (verdict_row, &row) in rows.iter().enumerate() {
                    // A null verdict rejects the row, as a WHERE clause would.
                    if !verdict.is_valid(verdict_row) || !verdict.value(verdict_row) {
                        new_actions[row as usize] = Action::Nothing as u8;
                    }
                }
                Ok(())
            };
        if let Some(condition) = &self.update_if {
            reject_failing(condition, 0, &update_rows)?;
        }
        if let Some(condition) = &self.delete_if {
            reject_failing(condition, update_rows.len(), &delete_rows)?;
        }
        let mut columns = batch.columns().to_vec();
        columns[self.action_idx] = Arc::new(UInt8Array::from(new_actions));
        Ok(RecordBatch::try_new(batch.schema(), columns)?)
    }
}

impl DisplayAs for MergeConditionExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "MergeCondition:")?;
        if let Some(condition) = &self.evaluator.update_if {
            write!(f, " update_if={condition},")?;
        }
        if let Some(condition) = &self.evaluator.delete_if {
            write!(f, " delete_if={condition},")?;
        }
        write!(
            f,
            " take=[{}]",
            self.evaluator
                .taken_columns()
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl ExecutionPlan for MergeConditionExec {
    fn name(&self) -> &str {
        "MergeConditionExec"
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "MergeConditionExec requires exactly one child".to_string(),
            ));
        }
        let input = children[0].clone();
        if input.schema() != self.input.schema() {
            return Err(DataFusionError::Internal(
                "MergeConditionExec cannot change its input schema".to_string(),
            ));
        }
        Ok(Arc::new(Self {
            properties: input.properties().clone(),
            input,
            evaluator: self.evaluator.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let input = self.input.execute(partition, context)?;
        let evaluator = self.evaluator.clone();
        let stream = input.then(move |batch| {
            let evaluator = evaluator.clone();
            let baseline = baseline.clone();
            async move {
                let batch = evaluator.apply(batch?).await?;
                baseline.record_output(batch.num_rows());
                Ok(batch)
            }
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, RecordBatchIterator, record_batch, types::Int32Type};
    use datafusion::physical_plan::collect;
    use datafusion::prelude::SessionContext;
    use datafusion_expr::col;
    use datafusion_physical_plan::test::TestMemoryExec;
    use lance_datagen::{ArrayGeneratorExt, ByteCount, RowCount, Seed, array, gen_batch};

    use std::collections::HashMap;

    use datafusion::catalog::TableProvider;
    use lance_file::version::LanceFileVersion;

    use crate::datafusion::dataframe::LanceTableProvider;
    use crate::dataset::WriteParams;
    use crate::{BlobArrayBuilder, blob_field};

    /// The taken columns must carry the types the target scan presents, or a
    /// condition that resolves against the scan fails against the take.
    #[rstest::rstest]
    #[case::legacy_blob(LanceFileVersion::V2_1, legacy_blob_batch())]
    #[case::blob_v2(LanceFileVersion::V2_2, blob_v2_batch())]
    #[case::json(LanceFileVersion::V2_2, json_batch())]
    #[case::plain_binary(LanceFileVersion::V2_2, plain_binary_batch())]
    #[tokio::test]
    async fn taken_types_match_the_target_scan(
        #[case] version: LanceFileVersion,
        #[case] batch: RecordBatch,
    ) {
        let ds = Arc::new(
            Dataset::write(
                Box::new(RecordBatchIterator::new(
                    [Ok(batch.clone())],
                    batch.schema(),
                )),
                "memory://",
                Some(WriteParams {
                    data_storage_version: Some(version),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        let scan = LanceTableProvider::new_with_ordering(ds.clone(), true, true, false)
            .with_blob_handling(target_blob_handling(&ds));
        let scanned = scan.schema().field_with_name("payload").unwrap().clone();

        let input = record_batch!(
            ("_rowid", UInt64, [0u64]),
            ("_rowaddr", UInt64, [0u64]),
            ("key", Int32, [1]),
            ("__merge_source_sentinel", Boolean, [Some(true)]),
            ("__action", UInt8, [1u8])
        )
        .unwrap();
        let input_schema = DFSchema::from_field_specific_qualified_schema(
            ["target", "target", "source", "source"]
                .map(|relation| Some(TableReference::bare(relation)))
                .into_iter()
                .chain([None])
                .collect(),
            &input.schema(),
        )
        .unwrap();
        let session = SessionContext::new();
        let exec = MergeConditionExec::try_new(
            TestMemoryExec::try_new_exec(&[vec![input.clone()]], input.schema(), None).unwrap(),
            &input_schema,
            ds,
            Some(col("target.payload").is_not_null()),
            None,
            &session.state(),
        )
        .unwrap();
        let taken = exec
            .evaluator
            .eval_schema
            .field_with_name("payload")
            .unwrap()
            .clone();
        assert_eq!(taken.data_type(), scanned.data_type());
        assert_eq!(taken.is_nullable(), scanned.is_nullable());
    }

    fn json_batch() -> RecordBatch {
        let payload = lance_arrow::json::json_field("payload", true);
        let values: ArrayRef = match payload.data_type() {
            DataType::Utf8 => Arc::new(arrow_array::StringArray::from(vec![r#"{"a":1}"#])),
            DataType::LargeUtf8 => {
                Arc::new(arrow_array::LargeStringArray::from(vec![r#"{"a":1}"#]))
            }
            _ => Arc::new(arrow_array::LargeBinaryArray::from(vec![
                lance_arrow::json::encode_json(r#"{"a":1}"#)
                    .unwrap()
                    .as_slice(),
            ])),
        };
        RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("key", DataType::Int32, false),
                payload,
            ])),
            vec![Arc::new(arrow_array::Int32Array::from(vec![1])), values],
        )
        .unwrap()
    }

    fn plain_binary_batch() -> RecordBatch {
        record_batch!(
            ("key", Int32, [1]),
            ("payload", LargeBinary, [b"old".as_slice()])
        )
        .unwrap()
    }

    fn legacy_blob_batch() -> RecordBatch {
        let payload =
            arrow_schema::Field::new("payload", DataType::LargeBinary, true).with_metadata(
                HashMap::from([(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string())]),
            );
        RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("key", DataType::Int32, false),
                payload,
            ])),
            vec![
                Arc::new(arrow_array::Int32Array::from(vec![1])),
                Arc::new(arrow_array::LargeBinaryArray::from(vec![Some(
                    b"old".as_slice(),
                )])),
            ],
        )
        .unwrap()
    }

    fn blob_v2_batch() -> RecordBatch {
        let mut blobs = BlobArrayBuilder::new(1);
        blobs.push_bytes(b"old").unwrap();
        RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("key", DataType::Int32, false),
                blob_field("payload", true),
            ])),
            vec![
                Arc::new(arrow_array::Int32Array::from(vec![1])),
                blobs.finish().unwrap(),
            ],
        )
        .unwrap()
    }

    /// `WhenMatched::Delete` marks matched rows `Delete` unconditionally; the
    /// `DeleteIf` gate applies to the target-only rows alone, which the join
    /// left without a source sentinel.
    #[tokio::test]
    async fn delete_if_gates_target_only_rows_only() {
        // Two 50-row fragments; `payload` is null at local row 25.
        let mut nulls = vec![false; 50];
        nulls[25] = true;
        let batch = gen_batch()
            .with_seed(Seed::from(7))
            .col("key", array::step::<Int32Type>())
            .col(
                "payload",
                array::rand_fixedbin(ByteCount::from(8), true).with_nulls(&nulls),
            )
            .into_batch_rows(RowCount::from(100))
            .unwrap();
        let ds = Dataset::write(
            Box::new(RecordBatchIterator::new(
                [Ok(batch.clone())],
                batch.schema(),
            )),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 50,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Row 0 is matched; rows 1 and 2 are target-only and only row 1's
        // payload is null.
        let input = record_batch!(
            ("_rowid", UInt64, [25u64, (1 << 32) | 25, (1 << 32) | 30]),
            ("_rowaddr", UInt64, [25u64, (1 << 32) | 25, (1 << 32) | 30]),
            ("key", Int32, [25, 75, 80]),
            ("__merge_source_sentinel", Boolean, [Some(true), None, None]),
            ("__action", UInt8, [3u8, 3, 3])
        )
        .unwrap();
        let input_schema = DFSchema::from_field_specific_qualified_schema(
            ["target", "target", "source", "source"]
                .map(|relation| Some(TableReference::bare(relation)))
                .into_iter()
                .chain([None])
                .collect(),
            &input.schema(),
        )
        .unwrap();
        let session = SessionContext::new();
        let exec = MergeConditionExec::try_new(
            TestMemoryExec::try_new_exec(&[vec![input.clone()]], input.schema(), None).unwrap(),
            &input_schema,
            Arc::new(ds),
            None,
            Some(col("target.payload").is_null()),
            &session.state(),
        )
        .unwrap();
        let output = collect(Arc::new(exec), session.task_ctx()).await.unwrap();
        let actions = output[0][MERGE_ACTION_COLUMN].as_primitive::<UInt8Type>();
        assert_eq!(
            actions.values(),
            &[
                Action::Delete as u8,
                Action::Delete as u8,
                Action::Nothing as u8
            ]
        );
    }
}
