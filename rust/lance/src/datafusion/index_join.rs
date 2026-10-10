// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Replaces a hash join against a Lance table with index lookups when the join
//! reads nothing from the table but its keys and row locators.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::tree_node::Transformed;
use datafusion::common::{DFSchemaRef, JoinType, NullEquality};
use datafusion::error::Result as DFResult;
use datafusion::execution::SessionState;
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use datafusion_expr::{
    Expr, Extension, Join, LogicalPlan, UserDefinedLogicalNode, UserDefinedLogicalNodeCore,
};
use lance_core::{ROW_ADDR, ROW_ID};
use roaring::RoaringBitmap;

use super::planning_context::{PlanningContext, lance_dataset, lance_table_scan};
use crate::io::exec::index_join::{IndexJoinExec, IndexJoinSpec, TargetColumn};

/// Above this many bytes of keys in rows the index does not cover, an index
/// join is not planned: holding them in memory costs more than the full scan
/// it saves, and `optimize_indices` would bring them under the index.
const MAX_UNCOVERED_KEY_BYTES: u64 = 50 * 1024 * 1024;

/// Bytes per value assumed for a variable-width key, whose size the manifest
/// does not record.
const VARIABLE_WIDTH_KEY_BYTES: u64 = 32;

/// Rewrites `Join(Lance table, input)` into an [`IndexJoinNode`].
///
/// The join must be an inner or right equi-join on one plain column with
/// [`NullEquality::NullEqualsNothing`] and no extra filter, the table's key
/// must have a lookup index in the [`PlanningContext`], and the table scan must
/// read only the key, `_rowid` and `_rowaddr`. Run it after projection pushdown
/// has pruned the scan.
#[derive(Debug)]
pub struct IndexJoinRule {
    context: Arc<PlanningContext>,
    max_uncovered_key_bytes: u64,
}

impl IndexJoinRule {
    pub fn new(context: Arc<PlanningContext>) -> Self {
        Self {
            context,
            max_uncovered_key_bytes: MAX_UNCOVERED_KEY_BYTES,
        }
    }

    #[cfg(test)]
    fn with_max_uncovered_key_bytes(mut self, bytes: u64) -> Self {
        self.max_uncovered_key_bytes = bytes;
        self
    }

    fn plan(&self, join: &Join) -> Option<IndexJoinNode> {
        if !matches!(join.join_type, JoinType::Inner | JoinType::Right)
            || join.filter.is_some()
            || join.null_equality != NullEquality::NullEqualsNothing
            || join.null_aware
        {
            return None;
        }
        let scan = lance_table_scan(&join.left)?;
        if !scan.filters.is_empty() || scan.fetch.is_some() {
            return None;
        }
        let dataset = lance_dataset(scan)?;
        let table = self.context.table(&dataset)?;

        // A composite key would need every column looked up and the matches
        // intersected. A low-cardinality column then matches most of the
        // table per key, so composite keys keep the hash join.
        let [(Expr::Column(target_key), Expr::Column(input_key))] = join.on.as_slice() else {
            return None;
        };
        let (_, target_field) = join
            .left
            .schema()
            .qualified_field_from_column(target_key)
            .ok()?;
        let (_, input_field) = join
            .right
            .schema()
            .qualified_field_from_column(input_key)
            .ok()?;
        if target_field.data_type() != input_field.data_type() {
            return None;
        }
        let key = table.lookup_index(&target_key.name)?.clone();

        let output_columns = join
            .left
            .schema()
            .fields()
            .iter()
            .map(|field| match field.name().as_str() {
                ROW_ID => Some(TargetColumn::RowId),
                ROW_ADDR => Some(TargetColumn::RowAddr),
                name if name == target_key.name => Some(TargetColumn::Key),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;

        let coverage = key.coverage.clone();
        let uncovered_fragments = dataset
            .fragments()
            .iter()
            .filter(|fragment| !coverage.contains(fragment.id as u32))
            .cloned()
            .collect::<Vec<_>>();
        let stale_rows = key.stale_rows.clone();

        let uncovered_rows = uncovered_fragments
            .iter()
            .map(|fragment| {
                fragment
                    .num_rows()
                    .or(fragment.physical_rows)
                    .map(|rows| rows as u64)
                    // A fragment without a row count is too old to estimate.
                    .unwrap_or(u64::MAX)
            })
            .chain(stale_rows.values().map(RoaringBitmap::len))
            .fold(0u64, u64::saturating_add);
        let key_bytes = target_field
            .data_type()
            .primitive_width()
            .map_or(VARIABLE_WIDTH_KEY_BYTES, |width| width as u64);
        // Plus the row id and address kept per row.
        let uncovered_bytes = uncovered_rows.saturating_mul(key_bytes + 16);
        if uncovered_bytes > self.max_uncovered_key_bytes {
            log::warn!(
                "Not joining on the scalar index of '{}': about {uncovered_rows} rows are \
                 not covered by it (~{uncovered_bytes} bytes of keys, above the \
                 {}-byte limit), so the join scans the table instead. \
                 Run optimize_indices to bring them under the index.",
                target_key.name,
                self.max_uncovered_key_bytes,
            );
            return None;
        }

        let spec = IndexJoinSpec {
            dataset,
            join_type: join.join_type,
            key,
            output_columns,
            coverage,
            uncovered_fragments,
            stale_rows,
        };
        Some(IndexJoinNode {
            input: join.right.as_ref().clone(),
            spec: Arc::new(spec),
            input_key: input_key.clone(),
            schema: join.schema.clone(),
        })
    }
}

impl OptimizerRule for IndexJoinRule {
    fn name(&self) -> &str {
        "index_join"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> DFResult<Transformed<LogicalPlan>> {
        let LogicalPlan::Join(join) = &plan else {
            return Ok(Transformed::no(plan));
        };
        Ok(match self.plan(join) {
            Some(node) => Transformed::yes(LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            })),
            None => Transformed::no(plan),
        })
    }
}

/// A join planned by [`IndexJoinRule`]: its only input is the non-table side
/// of the replaced join, and its schema is the replaced join's.
#[derive(Debug)]
pub struct IndexJoinNode {
    input: LogicalPlan,
    spec: Arc<IndexJoinSpec>,
    input_key: datafusion::common::Column,
    schema: DFSchemaRef,
}

impl PartialEq for IndexJoinNode {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.spec, &other.spec)
            && self.input == other.input
            && self.input_key == other.input_key
            && self.schema == other.schema
    }
}

impl Eq for IndexJoinNode {}

impl Hash for IndexJoinNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.spec).hash(state);
        self.input.hash(state);
        self.input_key.hash(state);
    }
}

impl PartialOrd for IndexJoinNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if Arc::ptr_eq(&self.spec, &other.spec) {
            self.input.partial_cmp(&other.input)
        } else {
            None
        }
    }
}

impl UserDefinedLogicalNodeCore for IndexJoinNode {
    fn name(&self) -> &str {
        "IndexJoin"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "IndexJoin: type={}, on=[{} = {}]",
            self.spec.join_type, self.spec.key.column, self.input_key
        )
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> DFResult<Self> {
        if !exprs.is_empty() || inputs.len() != 1 {
            return Err(datafusion::error::DataFusionError::Internal(
                "IndexJoinNode takes one input and no expressions".to_string(),
            ));
        }
        Ok(Self {
            input: inputs.remove(0),
            spec: self.spec.clone(),
            input_key: self.input_key.clone(),
            schema: self.schema.clone(),
        })
    }

    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        // Every input column passes through to the output, whose schema is
        // fixed, so none can be pruned.
        Some(vec![(0..self.input.schema().fields().len()).collect()])
    }
}

/// Plans [`IndexJoinNode`] as an [`IndexJoinExec`].
pub struct IndexJoinPlanner;

#[async_trait]
impl ExtensionPlanner for IndexJoinPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<IndexJoinNode>() else {
            return Ok(None);
        };
        let input_key = node.input.schema().index_of_column(&node.input_key)?;
        Ok(Some(Arc::new(IndexJoinExec::try_new(
            physical_inputs[0].clone(),
            node.spec.clone(),
            input_key,
            Arc::new(node.schema.as_arrow().clone()),
        )?)))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use lance_datagen::array;
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;

    use super::*;
    use crate::datafusion::LanceTableProvider;
    use crate::dataset::WriteParams;
    use crate::index::DatasetIndexExt;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

    /// The rule joins through the index only while the keys of the rows the
    /// index does not cover fit under the limit.
    #[rstest::rstest]
    #[case::at_limit(2400, true)]
    #[case::over_limit(2399, false)]
    #[tokio::test]
    async fn test_uncovered_rows_limit(#[case] limit: u64, #[case] index_join: bool) {
        let mut dataset = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int64Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(100))
            .await
            .unwrap();
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
        let appended = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(200..300))],
        )
        .unwrap();
        dataset
            .append(
                arrow_array::RecordBatchIterator::new([Ok(appended)], schema.clone()),
                Some(WriteParams::default()),
            )
            .await
            .unwrap();

        // 100 unindexed rows of an 8-byte key plus 16 bytes of row locators.
        // The join reads only the key, so projection pushdown prunes the scan
        // to it and the row locators.
        let ctx = SessionContext::new();
        let target = ctx
            .read_table(Arc::new(LanceTableProvider::new(
                Arc::new(dataset),
                true,
                true,
            )))
            .unwrap()
            .alias("target")
            .unwrap();
        let source_batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(vec![1]))])
                .unwrap();
        let source = ctx
            .read_table(Arc::new(
                MemTable::try_new(schema, vec![vec![source_batch]]).unwrap(),
            ))
            .unwrap()
            .alias("source")
            .unwrap();
        let (state, plan) = target
            .join(source, JoinType::Right, &["id"], &["id"], None)
            .unwrap()
            .into_parts();

        let context = Arc::new(PlanningContext::collect(&plan).await.unwrap());
        let rule = IndexJoinRule::new(context).with_max_uncovered_key_bytes(limit);
        let state = SessionStateBuilder::new_from_existing(state)
            .with_optimizer_rule(Arc::new(rule))
            .build();
        let optimized = state.optimize(&plan).unwrap();
        let display = optimized.display_indent().to_string();
        assert_eq!(display.contains("IndexJoin"), index_join, "{display}");
    }
}
