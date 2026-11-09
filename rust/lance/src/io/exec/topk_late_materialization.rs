// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Physical optimizer rule that late-materializes the non-sort columns of a
//! top-k (`ORDER BY ... LIMIT k`) over a single [`FilteredReadExec`].
//!
//! ```text
//!   [SortPreservingMergeExec]             ProjectionExec (original column order)
//!     SortExec(TopK)                        TakeExec (every other column)
//!       [pass-through]*          ==>          [SortPreservingMergeExec]
//!         FilteredReadExec                      SortExec(TopK)
//!                                                 [pass-through]*
//!                                                   FilteredReadExec (sort keys + _rowid)
//! ```
//!
//! A `TableProvider::scan` carries no ordering, so a DataFusion-planned top-k
//! otherwise reads every projected column for every matching row and then
//! discards all but `k` of them. [`Scanner::order_by`](crate::dataset::scanner::Scanner::order_by)
//! makes the same split when it plans the sort itself; this rule does it for
//! plans built elsewhere, such as SQL over [`LanceTableProvider`](crate::datafusion::LanceTableProvider).
//!
//! A pass-through node is one the rule can rebuild over the narrowed read
//! without changing any value the take later fetches:
//!
//! - a [`ProjectionExec`] that only selects columns, narrowed to the columns
//!   the read still has;
//! - a [`RepartitionExec`] that does not hash (hash partitioning holds column
//!   indices into the wide schema);
//! - a [`CoalescePartitionsExec`] or [`CooperativeExec`];
//! - a custom node the caller certifies with
//!   [`TopKLateMaterialization::with_pass_through`], such as one a
//!   `TableProvider` wraps around the scan. A certified node that rewrites
//!   schema metadata is applied again above the take, since [`TakeExec`] emits
//!   the dataset's metadata. The rule declines if that node may reorder rows.
//!
//! A [`TakeExec`] or row-stream [`FilteredReadExec`] in the chain (the
//! scanner's own late materialization behind a filter) is dropped: the take
//! above the sort fetches its columns instead.
//! Row-stream reads with a physical row selection prevent the rewrite, since
//! dropping them would remove a restriction that must apply before sorting.
//!
//! Rebuilding the read and fetching deferred columns can replace the planner's
//! field metadata with the dataset's storage metadata. Projections under the new
//! sort and above the take restore the original field metadata, which sort
//! expressions can inspect.

use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow_schema::{Fields, Schema as ArrowSchema};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::config::ConfigOptions;
use datafusion::error::Result as DFResult;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::coop::CooperativeExec;
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::replace_children_if_necessary;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::utils::{collect_columns, reassign_expr_columns};
use datafusion_physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use lance_core::ROW_ID;
use lance_core::datatypes::OnMissing;
use lance_select::RowSetOps;

use super::TakeExec;
use super::filtered_read::FilteredReadExec;

/// Decides whether a read may be narrowed. Returning `false` leaves the top-k
/// over that read unchanged.
pub type ReadEligibility = Arc<dyn Fn(&FilteredReadExec, &ConfigOptions) -> bool + Send + Sync>;

/// Decides whether a custom node may be walked through. See
/// [`TopKLateMaterialization::with_pass_through`].
pub type PassThrough = Arc<dyn Fn(&dyn ExecutionPlan) -> bool + Send + Sync>;

/// Defer every non-sort column of a top-k over a Lance read to a take of the
/// surviving rows. See the module docs.
///
/// ```
/// # use std::sync::Arc;
/// # use datafusion::execution::SessionStateBuilder;
/// # use lance::io::exec::topk_late_materialization::TopKLateMaterialization;
/// let state = SessionStateBuilder::new()
///     .with_default_features()
///     .with_physical_optimizer_rule(Arc::new(TopKLateMaterialization::new()))
///     .build();
/// ```
#[derive(Clone, Default)]
pub struct TopKLateMaterialization {
    /// `None`: every read qualifies.
    read_eligibility: Option<ReadEligibility>,
    /// `None`: only the built-in pass-through nodes.
    pass_through: Option<PassThrough>,
}

impl Debug for TopKLateMaterialization {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TopKLateMaterialization")
            .field("read_eligibility", &self.read_eligibility.is_some())
            .field("pass_through", &self.pass_through.is_some())
            .finish()
    }
}

impl TopKLateMaterialization {
    pub fn new() -> Self {
        Self::default()
    }

    /// Only narrow reads for which `eligibility` returns `true`, for callers
    /// that know a read cannot be served by a take (for example, one unioned
    /// with rows that have no row id).
    pub fn with_read_eligibility(
        mut self,
        eligibility: impl Fn(&FilteredReadExec, &ConfigOptions) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.read_eligibility = Some(Arc::new(eligibility));
        self
    }

    /// Also walk through custom nodes for which `pass_through` returns `true`.
    /// Such a node must keep every row, column, and value of its single input
    /// (it may rewrite schema metadata), and must hold no column indices, as
    /// it is rebuilt over the narrowed read with `with_new_children`. It must
    /// also report [`CardinalityEffect::Equal`] and no fetch, or the rule
    /// declines. A node that rewrites schema metadata is applied again above
    /// the take, so it must also keep its input's row order and report that in
    /// [`ExecutionPlan::maintains_input_order`], or the rule declines.
    pub fn with_pass_through(
        mut self,
        pass_through: impl Fn(&dyn ExecutionPlan) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.pass_through = Some(Arc::new(pass_through));
        self
    }

    /// `Ok(None)`: not a top-k. `Err`: a top-k this rule declines, and why.
    fn rewrite(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>, String> {
        let (merge, sort) = if let Some(merge) = node.downcast_ref::<SortPreservingMergeExec>() {
            match merge.input().downcast_ref::<SortExec>() {
                Some(sort) if merge.fetch().is_some() && sort.preserve_partitioning() => {
                    (Some(merge), sort)
                }
                _ => return Ok(None),
            }
        } else if let Some(sort) = node.downcast_ref::<SortExec>() {
            if sort.preserve_partitioning() {
                // The merge above it is the top-k root.
                return Ok(None);
            }
            (None, sort)
        } else {
            return Ok(None);
        };
        let Some(fetch) = sort.fetch() else {
            return Ok(None);
        };

        let mut chain = Vec::new();
        let mut dropped_takes = Vec::new();
        let mut current = sort.input().clone();
        loop {
            if let Some(read) = current.downcast_ref::<FilteredReadExec>() {
                if read.row_stream_input().is_none() {
                    break;
                }
                if read.options().physical_row_addr_prefilter.is_some() {
                    return Err("row-stream read has a physical row selection".into());
                }
                // Row-stream reads take every input row: their constructor
                // rejects filters, scan ranges, and deleted rows, and the
                // guard above excludes physical row selections.
                dropped_takes.push(read.dataset().clone());
            } else if let Some(take) = current.downcast_ref::<TakeExec>() {
                dropped_takes.push(take.dataset().clone());
            } else {
                self.check_pass_through(&current)?;
                chain.push(current.clone());
            }
            let [child] = &current.children()[..] else {
                return Err(format!(
                    "{} does not have exactly one child",
                    current.name()
                ));
            };
            current = (*child).clone();
        }
        let read = current
            .downcast_ref::<FilteredReadExec>()
            .ok_or("the loop exits on a FilteredReadExec")?;
        let dataset = read.dataset().clone();
        if dropped_takes.iter().any(|take| {
            take.uri() != dataset.uri() || take.manifest.version != dataset.manifest.version
        }) {
            return Err("a take in the chain reads a different dataset".into());
        }
        if read.options().with_deleted_rows {
            // Deleted rows carry a null row id, which a take cannot follow.
            return Err("read includes deleted rows".into());
        }
        if let Some(eligibility) = &self.read_eligibility
            && !eligibility(read, config)
        {
            return Err("read is not eligible".into());
        }

        if max_rows(read).is_some_and(|rows| fetch >= rows) {
            // The sort drops nothing, so the take would re-read every row.
            return Err("the limit keeps every row".into());
        }

        let original = node.schema();
        let full = &read.options().projection;
        let wanted = dataset
            .empty_projection()
            .union_arrow_schema(&original, OnMissing::Ignore)
            .map_err(|e| e.to_string())?
            .with_blob_handling(full.blob_handling.clone());
        let mut narrow = dataset
            .empty_projection()
            .union_columns(sort_column_names(sort.expr()), OnMissing::Ignore)
            .map_err(|e| e.to_string())?
            .with_row_id();
        narrow.with_row_addr = full.with_row_addr;
        narrow.with_row_last_updated_at_version = full.with_row_last_updated_at_version;
        narrow.with_row_created_at_version = full.with_row_created_at_version;
        narrow.blob_handling = full.blob_handling.clone();
        let narrow_read = read.with_projection(narrow).map_err(|e| e.to_string())?;
        let mut input: Arc<dyn ExecutionPlan> = Arc::new(narrow_read);
        for node in chain.iter().rev() {
            input = rebuild(node, input)?;
        }
        let input = with_field_metadata(input, &sort.input().schema())?;

        let schema = input.schema();
        let mut top: Arc<dyn ExecutionPlan> = Arc::new(
            SortExec::new(rebind(sort.expr(), &schema)?, input)
                .with_preserve_partitioning(sort.preserve_partitioning())
                .with_fetch(sort.fetch()),
        );
        if let Some(merge) = merge {
            top = Arc::new(
                SortPreservingMergeExec::new(rebind(merge.expr(), &schema)?, top)
                    .with_fetch(merge.fetch()),
            );
        }

        let take = TakeExec::try_new(dataset, top, wanted)
            .map_err(|e| e.to_string())?
            .ok_or("the sort reads every output column")?;
        // Reapplied wrappers restore the schema's own metadata.
        let mut output = select_columns(Arc::new(take), original.fields())?;
        for node in chain.iter().rev().filter(|node| rewrites_metadata(node)) {
            output = replace_children_if_necessary(node.clone(), vec![output])
                .map_err(|e| e.to_string())?;
        }
        if output.schema() != original {
            return Err(format!(
                "rewritten plan does not reproduce the original schema: expected {original:?}, got {:?}",
                output.schema()
            ));
        }
        Ok(Some(output))
    }

    /// `Ok` if [`rebuild`] can re-create `node` over the narrowed read.
    fn check_pass_through(&self, node: &Arc<dyn ExecutionPlan>) -> Result<(), String> {
        let decline = |why: &str| {
            Err(format!(
                "{} between the sort and the read {why}",
                node.name()
            ))
        };
        if node.fetch().is_some() {
            // `CoalescePartitionsExec` reports `Equal` even when a fetch caps its rows.
            return decline("has a fetch");
        }
        if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
            let selects_columns = projection.expr().iter().all(|projected| {
                projected
                    .expr
                    .downcast_ref::<Column>()
                    .is_some_and(|column| column.name() == projected.alias)
            });
            return if selects_columns {
                Ok(())
            } else {
                decline("computes or renames columns")
            };
        }
        if let Some(repartition) = node.downcast_ref::<RepartitionExec>() {
            return if matches!(repartition.partitioning(), Partitioning::Hash(..)) {
                decline("hash-partitions")
            } else {
                Ok(())
            };
        }
        if node.is::<CoalescePartitionsExec>() || node.is::<CooperativeExec>() {
            return Ok(());
        }
        // `Equal` cardinality and unchanged columns do not mean unchanged values.
        if !self
            .pass_through
            .as_ref()
            .is_some_and(|pass_through| pass_through(node.as_ref()))
        {
            return decline("is not a known pass-through node");
        }
        if !matches!(node.cardinality_effect(), CardinalityEffect::Equal) {
            return decline("may change the row count");
        }
        let [input] = &node.children()[..] else {
            return decline("does not have exactly one child");
        };
        if node.schema().fields() != input.schema().fields() {
            return decline("changes the columns");
        }
        // It is applied again above the take, where reordering would undo the sort.
        if rewrites_metadata(node) && node.maintains_input_order() != [true] {
            return decline("rewrites schema metadata and may reorder rows");
        }
        Ok(())
    }
}

impl PhysicalOptimizerRule for TopKLateMaterialization {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(plan
            .transform_down(|node| match self.rewrite(&node, config) {
                Ok(Some(rewritten)) => {
                    Ok(Transformed::new(rewritten, true, TreeNodeRecursion::Jump))
                }
                Ok(None) => Ok(Transformed::no(node)),
                Err(reason) => {
                    log::debug!("Skipping topk_late_materialization: {reason}");
                    Ok(Transformed::no(node))
                }
            })?
            .data)
    }

    fn name(&self) -> &str {
        "topk_late_materialization"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn rewrites_metadata(node: &Arc<dyn ExecutionPlan>) -> bool {
    !node.is::<ProjectionExec>()
        && !node.is::<RepartitionExec>()
        && !node.is::<CoalescePartitionsExec>()
        && !node.is::<CooperativeExec>()
        && node
            .children()
            .first()
            .is_some_and(|input| input.schema().metadata() != node.schema().metadata())
}

/// `node` over the narrowed `input`.
fn rebuild(
    node: &Arc<dyn ExecutionPlan>,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
        // Keep the selected columns the narrowed read still has, plus the row id.
        let schema = input.schema();
        let kept = projection
            .expr()
            .iter()
            .map(|projected| projected.alias.as_str())
            .filter(|name| *name != ROW_ID && schema.index_of(name).is_ok())
            .chain([ROW_ID])
            .collect::<Vec<_>>();
        return project(input, kept);
    }
    replace_children_if_necessary(node.clone(), vec![input]).map_err(|e| e.to_string())
}

/// Select `names` from `input`, in that order.
fn project<'a>(
    input: Arc<dyn ExecutionPlan>,
    names: impl IntoIterator<Item = &'a str>,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    let schema = input.schema();
    let columns = names
        .into_iter()
        .map(|name| {
            let index = schema.index_of(name).map_err(|e| e.to_string())?;
            Ok((
                Arc::new(Column::new(name, index)) as Arc<dyn PhysicalExpr>,
                name.to_string(),
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Arc::new(
        ProjectionExec::try_new(columns, input).map_err(|e| e.to_string())?,
    ))
}

/// Copy field metadata by name from `original`, leaving unmatched input fields
/// unchanged.
fn with_field_metadata(
    input: Arc<dyn ExecutionPlan>,
    original: &ArrowSchema,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    let schema = input.schema();
    let fields = schema
        .fields()
        .iter()
        .map(|field| match original.field_with_name(field.name()) {
            Ok(original_field) => Arc::new(
                field
                    .as_ref()
                    .clone()
                    .with_metadata(original_field.metadata().clone()),
            ),
            Err(_) => field.clone(),
        })
        .collect::<Fields>();
    if &fields == schema.fields() {
        return Ok(input);
    }
    select_columns(input, &fields)
}

/// Select `fields` from `input` by name. Each column keeps its type from
/// `input` and takes its field metadata from `fields`.
fn select_columns(
    input: Arc<dyn ExecutionPlan>,
    fields: &Fields,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    let schema = input.schema();
    let columns = fields
        .iter()
        .map(|field| {
            let index = schema.index_of(field.name()).map_err(|e| e.to_string())?;
            Ok((
                Arc::new(Column::new(field.name(), index)) as Arc<dyn PhysicalExpr>,
                field.name().clone(),
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let projected_schema =
        ArrowSchema::new_with_metadata(fields.clone(), schema.metadata().clone());
    Ok(Arc::new(
        ProjectionExec::try_new_with_schema_metadata(columns, input, &projected_schema)
            .map_err(|e| e.to_string())?,
    ))
}

fn sort_column_names(ordering: &LexOrdering) -> Vec<String> {
    ordering
        .iter()
        .flat_map(|sort| collect_columns(&sort.expr))
        .map(|column| column.name().to_string())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

/// `ordering` with each column re-resolved by name against `schema`.
fn rebind(ordering: &LexOrdering, schema: &ArrowSchema) -> Result<LexOrdering, String> {
    let exprs = ordering
        .iter()
        .map(|sort| {
            let expr =
                reassign_expr_columns(sort.expr.clone(), schema).map_err(|e| e.to_string())?;
            Ok(PhysicalSortExpr::new(expr, sort.options))
        })
        .collect::<Result<Vec<_>, String>>()?;
    LexOrdering::new(exprs).ok_or_else(|| "empty ordering".into())
}

/// An upper bound on the rows `read` can produce: the rows its precomputed plan
/// selects, else the live rows of its fragments.
fn max_rows(read: &FilteredReadExec) -> Option<usize> {
    if let Some(rows) = read.plan().and_then(|plan| plan.rows.len()) {
        return Some(rows as usize);
    }
    read.options()
        .fragments
        .as_deref()
        .map(Vec::as_slice)
        .unwrap_or_else(|| read.dataset().fragments().as_slice())
        .iter()
        .map(|fragment| fragment.num_rows().or(fragment.physical_rows))
        .sum()
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, StringArray, StructArray,
    };
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType, Field, SortOptions};
    use datafusion::common::DFSchema;
    use datafusion::execution::TaskContext;
    use datafusion::logical_expr::when;
    use datafusion::physical_plan::collect;
    use datafusion::prelude::{SessionContext, col as logical_col, lit};
    use datafusion_functions::core::expr_fn::arrow_metadata;
    use datafusion_physical_expr::expressions::{col, is_null};
    use rstest::rstest;

    use super::*;
    use crate::Dataset;
    use crate::dataset::WriteParams;
    use crate::io::exec::filtered_read::FilteredReadOptions;

    #[cfg(feature = "substrait")]
    use crate::io::exec::filtered_read_proto::{
        filtered_read_exec_from_proto, filtered_read_exec_to_proto,
    };
    #[cfg(feature = "substrait")]
    use lance_select::RowAddrTreeMap;

    /// `wide` and `other` are functions of `key`, so a take that fetched the
    /// wrong rows shows up as a mismatch against the unrewritten plan. With
    /// `has_deletions`, the top row (`key = 9`) and one mid row are deleted, so row
    /// ids have gaps and the answer changes.
    async fn dataset(has_stable_row_ids: bool, has_deletions: bool) -> Arc<Dataset> {
        let batch = arrow_array::record_batch!(
            ("key", Int32, [5, 3, 8, 1, 9, 2, 7, 4, 6, 0]),
            (
                "wide",
                Utf8,
                ["f", "d", "i", "b", "j", "c", "h", "e", "g", "a"]
            ),
            ("other", Int32, [50, 30, 80, 10, 90, 20, 70, 40, 60, 0])
        )
        .unwrap();
        let params = WriteParams {
            max_rows_per_file: 3,
            enable_stable_row_ids: has_stable_row_ids,
            ..Default::default()
        };
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            "memory://",
            Some(params),
        )
        .await
        .unwrap();
        if has_deletions {
            dataset.delete("key = 9 OR key = 7").await.unwrap();
        }
        Arc::new(dataset)
    }

    fn read(dataset: &Arc<Dataset>, is_filtered: bool, columns: &[&str]) -> Arc<dyn ExecutionPlan> {
        let mut options = FilteredReadOptions::basic_full_read(dataset).with_projection(
            dataset
                .empty_projection()
                .union_columns(columns, OnMissing::Error)
                .unwrap(),
        );
        if is_filtered {
            options = options
                .with_filter(None, Some(logical_col("other").gt(lit(20))))
                .unwrap();
        }
        Arc::new(FilteredReadExec::try_new(dataset.clone(), options, None).unwrap())
    }

    /// `ORDER BY key DESC LIMIT 3`, planned the way DataFusion plans it: a
    /// per-partition top-k under a merge, or one top-k over coalesced input.
    fn top_k(input: Arc<dyn ExecutionPlan>, is_partitioned: bool) -> Arc<dyn ExecutionPlan> {
        let ordering = LexOrdering::new([PhysicalSortExpr::new(
            col("key", &input.schema()).unwrap(),
            SortOptions {
                descending: true,
                nulls_first: false,
            },
        )])
        .unwrap();
        top_k_by(input, ordering, is_partitioned)
    }

    /// `ORDER BY ordering LIMIT 3`, planned as [`top_k`] plans it.
    fn top_k_by(
        input: Arc<dyn ExecutionPlan>,
        ordering: LexOrdering,
        is_partitioned: bool,
    ) -> Arc<dyn ExecutionPlan> {
        if is_partitioned {
            let input = Arc::new(
                RepartitionExec::try_new(input, Partitioning::RoundRobinBatch(4)).unwrap(),
            );
            let sort = SortExec::new(ordering.clone(), input)
                .with_preserve_partitioning(true)
                .with_fetch(Some(3));
            Arc::new(SortPreservingMergeExec::new(ordering, Arc::new(sort)).with_fetch(Some(3)))
        } else {
            let input = Arc::new(CoalescePartitionsExec::new(input));
            Arc::new(SortExec::new(ordering, input).with_fetch(Some(3)))
        }
    }

    /// `SELECT names FROM input`.
    fn select(input: Arc<dyn ExecutionPlan>, names: &[&str]) -> Arc<dyn ExecutionPlan> {
        let schema = input.schema();
        let indices = names
            .iter()
            .map(|name| schema.index_of(name).unwrap())
            .collect::<Vec<_>>();
        select_columns(input, schema.project(&indices).unwrap().fields()).unwrap()
    }

    fn optimize(
        rule: &TopKLateMaterialization,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        rule.optimize(plan.clone(), &ConfigOptions::default())
            .unwrap()
    }

    async fn run(plan: Arc<dyn ExecutionPlan>) -> RecordBatch {
        let schema = plan.schema();
        let batches = collect(plan, Arc::new(TaskContext::default()))
            .await
            .unwrap();
        arrow_select::concat::concat_batches(&schema, &batches).unwrap()
    }

    fn reads(plan: &Arc<dyn ExecutionPlan>) -> Vec<Vec<String>> {
        let mut reads = Vec::new();
        plan.apply(|node| {
            if node.is::<FilteredReadExec>() {
                reads.push(
                    node.schema()
                        .fields()
                        .iter()
                        .map(|f| f.name().clone())
                        .collect(),
                );
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        reads
    }

    #[rstest]
    #[tokio::test]
    async fn reads_sort_keys_and_takes_the_rest(
        #[values(true, false)] is_partitioned: bool,
        #[values(true, false)] is_filtered: bool,
        #[values(true, false)] is_projected: bool,
        #[values(true, false)] has_stable_row_ids: bool,
        #[values(true, false)] has_deletions: bool,
    ) {
        let dataset = dataset(has_stable_row_ids, has_deletions).await;
        let mut scan = read(&dataset, is_filtered, &["key", "wide", "other"]);
        if is_projected {
            // `SELECT wide, key`: reordered, and `other` dropped.
            scan = select(scan, &["wide", "key"]);
        }
        let plan = top_k(scan, is_partitioned);

        let rewritten = optimize(&TopKLateMaterialization::new(), &plan);

        assert_eq!(reads(&rewritten), [["key", "_rowid"]]);
        assert_eq!(rewritten.schema(), plan.schema());
        let expected = run(plan).await;
        assert_eq!(expected.num_rows(), 3);
        assert_eq!(run(rewritten).await, expected);
    }

    /// A take in the chain is the scanner's own late materialization of the
    /// filter; the rule drops it and takes those columns after the sort.
    #[tokio::test]
    async fn drops_a_take_under_the_sort() {
        let dataset = dataset(false, false).await;
        let options = FilteredReadOptions::basic_full_read(&dataset)
            .with_projection(
                dataset
                    .empty_projection()
                    .union_columns(["key", "other"], OnMissing::Error)
                    .unwrap()
                    .with_row_id(),
            )
            .with_filter(None, Some(logical_col("other").gt(lit(20))))
            .unwrap();
        let narrow: Arc<dyn ExecutionPlan> =
            Arc::new(FilteredReadExec::try_new(dataset.clone(), options, None).unwrap());
        let wide = dataset
            .empty_projection()
            .union_columns(["key", "wide", "other"], OnMissing::Error)
            .unwrap();
        let take = TakeExec::try_new(dataset.clone(), narrow, wide)
            .unwrap()
            .unwrap();
        let scan = select(Arc::new(take), &["key", "wide", "other"]);
        let plan = top_k(scan, true);

        let rewritten = optimize(&TopKLateMaterialization::new(), &plan);

        assert_eq!(reads(&rewritten), [["key", "_rowid"]]);
        let mut takes = 0;
        rewritten
            .apply(|node| {
                takes += usize::from(node.is::<TakeExec>());
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
        assert_eq!(takes, 1);
        assert_eq!(run(rewritten).await, run(plan).await);
    }

    #[cfg(feature = "substrait")]
    #[rstest]
    #[tokio::test]
    async fn preserves_physical_row_stream_selection(
        #[values(false, true)] has_stable_row_ids: bool,
        #[values(false, true)] is_partitioned: bool,
    ) {
        let dataset = dataset(has_stable_row_ids, false).await;
        let source_options = FilteredReadOptions::basic_full_read(&dataset).with_projection(
            dataset
                .empty_projection()
                .union_columns(["key"], OnMissing::Error)
                .unwrap()
                .with_row_id(),
        );
        let source: Arc<dyn ExecutionPlan> =
            Arc::new(FilteredReadExec::try_new(dataset.clone(), source_options, None).unwrap());
        let options = FilteredReadOptions::basic_full_read(&dataset).with_projection(
            dataset
                .empty_projection()
                .union_columns(["key", "wide", "other"], OnMissing::Error)
                .unwrap(),
        );
        let state = SessionContext::new().state();
        let exec = FilteredReadExec::try_new(dataset.clone(), options, None).unwrap();
        let mut proto = filtered_read_exec_to_proto(&exec, &state).await.unwrap();
        // Only the intermediate read restricts the selection; its source reads
        // every fragment, as custom decoded plans are allowed to do.
        let mut rows = RowAddrTreeMap::new();
        rows.insert_fragment(0);
        let mut encoded = Vec::new();
        rows.serialize_into(&mut encoded).unwrap();
        proto.options.as_mut().unwrap().physical_row_addr_allowlist = Some(encoded);
        let selected = filtered_read_exec_from_proto(proto, Some(dataset), Some(source), &state)
            .await
            .unwrap();
        let plan = top_k(Arc::new(selected), is_partitioned);
        let rewritten = optimize(&TopKLateMaterialization::new(), &plan);
        assert!(Arc::ptr_eq(&rewritten, &plan));
        let expected = arrow_array::record_batch!(
            ("key", Int32, [8, 5, 3]),
            ("wide", Utf8, ["i", "f", "d"]),
            ("other", Int32, [80, 50, 30])
        )
        .unwrap();
        assert_eq!(run(rewritten).await, expected);
    }

    #[rstest]
    #[case::no_fetch(None, Partitioning::RoundRobinBatch(4), &["key", "wide"], true, false)]
    #[case::limit_keeps_every_row(Some(10), Partitioning::RoundRobinBatch(4), &["key", "wide"], true, false)]
    // 10 physical rows, 8 live.
    #[case::limit_keeps_every_live_row(Some(8), Partitioning::RoundRobinBatch(4), &["key", "wide"], true, true)]
    #[case::hash_repartition(Some(3), Partitioning::Hash(vec![], 4), &["key", "wide"], true, false)]
    #[case::sort_reads_everything(Some(3), Partitioning::RoundRobinBatch(4), &["key"], true, false)]
    #[case::read_not_eligible(Some(3), Partitioning::RoundRobinBatch(4), &["key", "wide"], false, false)]
    #[tokio::test]
    async fn declines(
        #[case] fetch: Option<usize>,
        #[case] partitioning: Partitioning,
        #[case] columns: &[&str],
        #[case] is_eligible: bool,
        #[case] has_deletions: bool,
    ) {
        let dataset = dataset(false, has_deletions).await;
        let scan = read(&dataset, false, columns);
        let partitioning = match partitioning {
            Partitioning::Hash(_, n) => {
                Partitioning::Hash(vec![col("key", &scan.schema()).unwrap()], n)
            }
            other => other,
        };
        let input = Arc::new(RepartitionExec::try_new(scan, partitioning).unwrap());
        let ordering = LexOrdering::new([PhysicalSortExpr::new_default(
            col("key", &input.schema()).unwrap(),
        )])
        .unwrap();
        let sort = SortExec::new(ordering.clone(), input)
            .with_preserve_partitioning(true)
            .with_fetch(fetch);
        let plan: Arc<dyn ExecutionPlan> =
            Arc::new(SortPreservingMergeExec::new(ordering, Arc::new(sort)).with_fetch(fetch));
        let rule = TopKLateMaterialization::new().with_read_eligibility(move |_, _| is_eligible);

        assert!(Arc::ptr_eq(&optimize(&rule, &plan), &plan));
    }

    /// A sort key can read the row id's field metadata, which a projection may
    /// set.
    #[rstest]
    #[tokio::test]
    async fn sorting_on_row_id_metadata_keeps_its_order(
        #[values(false, true)] is_partitioned: bool,
    ) {
        let dataset = dataset(false, false).await;
        let options = FilteredReadOptions::basic_full_read(&dataset).with_projection(
            dataset
                .empty_projection()
                .union_columns(["key", "wide", "other"], OnMissing::Error)
                .unwrap()
                .with_row_id(),
        );
        let scan: Arc<dyn ExecutionPlan> =
            Arc::new(FilteredReadExec::try_new(dataset.clone(), options, None).unwrap());
        let schema = scan.schema();
        let fields = ["key", "wide", ROW_ID].map(|name| {
            let field = schema.field_with_name(name).unwrap().clone();
            if name == ROW_ID {
                field.with_metadata([("k".into(), "v".into())].into())
            } else {
                field
            }
        });
        let scan = select_columns(scan, &Fields::from(fields.to_vec())).unwrap();
        // ORDER BY CASE WHEN arrow_metadata(_rowid, 'k') = 'v' THEN -key ELSE key END
        let sort_key = when(
            arrow_metadata(vec![logical_col(ROW_ID), lit("k")]).eq(lit("v")),
            -logical_col("key"),
        )
        .otherwise(logical_col("key"))
        .unwrap();
        let sort_key = SessionContext::new()
            .create_physical_expr(
                sort_key,
                &DFSchema::try_from(scan.schema().as_ref().clone()).unwrap(),
            )
            .unwrap();
        let ordering = LexOrdering::new([PhysicalSortExpr::new_default(sort_key)]).unwrap();
        let plan = top_k_by(scan, ordering, is_partitioned);

        let rewritten = optimize(&TopKLateMaterialization::new(), &plan);

        assert_eq!(reads(&rewritten), [["key", "_rowid"]]);
        let expected = run(plan).await;
        let keys: ArrayRef = Arc::new(Int32Array::from(vec![9, 8, 7]));
        assert_eq!(&expected["key"], &keys);
        assert_eq!(run(rewritten).await, expected);
    }

    /// `s` is `{x, y}`, with `x` falling and `y` rising in row order, and null
    /// on even rows.
    async fn struct_dataset() -> Arc<Dataset> {
        let fields = Fields::from(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Int32, true),
        ]);
        let s = StructArray::try_new(
            fields,
            vec![
                Arc::new(Int32Array::from_iter_values((0..12).rev())),
                Arc::new(Int32Array::from_iter_values(0..12)),
            ],
            Some(NullBuffer::from(
                (0..12).map(|i| i % 2 == 1).collect::<Vec<_>>(),
            )),
        )
        .unwrap();
        let batch = RecordBatch::try_from_iter([
            ("s", Arc::new(s) as ArrayRef),
            ("other", Arc::new(Int32Array::from_iter_values(0..12))),
            (
                "t",
                Arc::new(StringArray::from_iter_values(
                    (0..12).map(|i| format!("t{i}")),
                )),
            ),
        ])
        .unwrap();
        let params = WriteParams {
            max_rows_per_file: 4,
            ..Default::default()
        };
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            "memory://",
            Some(params),
        )
        .await
        .unwrap();
        Arc::new(dataset)
    }

    /// The read holds only `s.y`, while a narrowed read holds whole columns.
    #[rstest]
    #[tokio::test]
    async fn declines_a_sort_on_a_partial_struct(
        #[values(false, true)] is_null_key: bool,
        #[values(false, true)] is_projected: bool,
    ) {
        let dataset = struct_dataset().await;
        let mut scan = read(&dataset, false, &["s.y", "other", "t"]);
        if is_projected {
            scan = select(scan, &["t", "s", "other"]);
        }
        let schema = scan.schema();
        let s = col("s", &schema).unwrap();
        let ordering = if is_null_key {
            // ORDER BY s IS NULL DESC, other
            vec![
                PhysicalSortExpr::new(
                    is_null(s).unwrap(),
                    SortOptions {
                        descending: true,
                        nulls_first: false,
                    },
                ),
                PhysicalSortExpr::new_default(col("other", &schema).unwrap()),
            ]
        } else {
            // ORDER BY s NULLS LAST
            vec![PhysicalSortExpr::new(
                s,
                SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            )]
        };
        let input = Arc::new(CoalescePartitionsExec::new(scan));
        let sort = SortExec::new(LexOrdering::new(ordering).unwrap(), input);
        let plan: Arc<dyn ExecutionPlan> = Arc::new(sort.with_fetch(Some(2)));

        assert!(Arc::ptr_eq(
            &optimize(&TopKLateMaterialization::new(), &plan),
            &plan
        ));
    }
}
