// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end tests for [`TopKLateMaterialization`] on SQL top-k plans built
//! through [`LanceTableProvider`], including one wrapped in a custom node, and
//! on reads the scanner or a distributed planner has already configured.

use std::sync::Arc;

use arrow::datatypes::{Int32Type, UInt64Type};
use arrow_array::cast::AsArray;
use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_schema::{DataType, Schema, SchemaRef, SortOptions};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::ScalarValue;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::config::ConfigOptions;
use datafusion::datasource::TableType;
use datafusion::execution::{SendableRecordBatchStream, SessionStateBuilder, TaskContext};
use datafusion::logical_expr::var_provider::{VarProvider, VarType};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_expr::expressions::col;
use datafusion::physical_expr::{EquivalenceProperties, LexOrdering, PhysicalSortExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::sanity_checker::SanityCheckPlan;
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, collect, displayable,
};
use datafusion::prelude::SessionContext;
use futures::StreamExt;
use lance::Dataset;
use lance::datafusion::LanceTableProvider;
use lance::dataset::WriteParams;
use lance::io::exec::TakeExec;
use lance::io::exec::filtered_read::{FilteredReadExec, FilteredReadOptions, FilteredReadPlan};
use lance::io::exec::topk_late_materialization::TopKLateMaterialization;
use lance_datafusion::utils::{BYTES_READ_METRIC, MetricsExt};
use lance_datagen::{BatchCount, ByteCount, RowCount, array, gen_batch};
use lance_select::RowAddrTreeMap;
use rstest::rstest;

/// 4 fragments of 250 rows: `id` is the row number, `cat` cycles 0..7, and
/// `wide` is 1 KB of text per row.
async fn make_dataset() -> Arc<Dataset> {
    let reader = gen_batch()
        .col("id", array::step::<UInt64Type>())
        .col("cat", array::cycle::<Int32Type>((0..7).collect()))
        .col("wide", array::rand_utf8(ByteCount::from(1000), false))
        .into_reader_rows(RowCount::from(250), BatchCount::from(4));
    let mut dataset = Dataset::write(
        reader,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 250,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    // `TakeExec` emits the dataset's schema metadata, which a wrapping
    // provider may have erased.
    dataset
        .update_schema_metadata([("owner", "test")])
        .await
        .unwrap();
    Arc::new(dataset)
}

fn rule() -> TopKLateMaterialization {
    TopKLateMaterialization::new().with_pass_through(|node| node.is::<MetadataEraserExec>())
}

fn context(provider: Arc<dyn TableProvider>, has_rule: bool) -> SessionContext {
    let mut state = SessionStateBuilder::new().with_default_features();
    if has_rule {
        state = state.with_physical_optimizer_rule(Arc::new(rule()));
    }
    let ctx = SessionContext::new_with_state(state.build());
    ctx.register_table("t", provider).unwrap();
    ctx
}

/// Plan and run `sql`, returning the executed plan (with its metrics) and
/// the result.
async fn run(ctx: &SessionContext, sql: &str) -> (Arc<dyn ExecutionPlan>, RecordBatch) {
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let result = execute(plan.clone()).await;
    (plan, result)
}

async fn execute(plan: Arc<dyn ExecutionPlan>) -> RecordBatch {
    let batches = collect(plan.clone(), Arc::new(TaskContext::default()))
        .await
        .unwrap();
    arrow_select::concat::concat_batches(&plan.schema(), &batches).unwrap()
}

/// `ORDER BY id DESC LIMIT 5` over `input`.
fn top_5_by_id(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let ordering = LexOrdering::new([PhysicalSortExpr::new(
        col("id", &input.schema()).unwrap(),
        SortOptions {
            descending: true,
            nulls_first: false,
        },
    )])
    .unwrap();
    Arc::new(SortExec::new(ordering, input).with_fetch(Some(5)))
}

/// The column scans in `plan`: `FilteredReadExec`s that are not fed by a row
/// stream.
fn scans(plan: &Arc<dyn ExecutionPlan>) -> Vec<Vec<String>> {
    let mut scans = Vec::new();
    plan.apply(|node| {
        if let Some(read) = node.downcast_ref::<FilteredReadExec>()
            && read.row_stream_input().is_none()
        {
            scans.push(
                read.schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect(),
            );
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    scans
}

/// Rows out of each `TakeExec` in an executed `plan`.
fn take_output_rows(plan: &Arc<dyn ExecutionPlan>) -> Vec<usize> {
    let mut rows = Vec::new();
    plan.apply(|node| {
        if node.is::<TakeExec>() {
            rows.push(node.metrics().unwrap().output_rows().unwrap());
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    rows
}

/// Bytes read from storage by every node of an executed `plan`.
fn bytes_read(plan: &Arc<dyn ExecutionPlan>) -> usize {
    let mut total = 0;
    plan.apply(|node| {
        if let Some(metrics) = node.metrics() {
            total += metrics
                .iter_gauges()
                .filter(|(name, _)| name.as_ref() == BYTES_READ_METRIC)
                .map(|(_, gauge)| gauge.value())
                .sum::<usize>();
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    total
}

/// Without a filter the scan reads `wide` directly; with one, the scanner
/// reads it behind the filter through a row-stream read. Either way the
/// rewritten plan scans only `id` and takes `wide` for the 5 surviving rows.
#[rstest]
#[case::unfiltered("SELECT id, wide FROM t ORDER BY id DESC LIMIT 5")]
#[case::filtered("SELECT id, wide FROM t WHERE cat = 3 ORDER BY id DESC LIMIT 5")]
#[case::star("SELECT * FROM t WHERE cat <> 3 ORDER BY id LIMIT 5")]
#[tokio::test]
async fn sql_topk_takes_non_sort_columns_after_the_sort(#[case] sql: &str) {
    let dataset = make_dataset().await;
    let provider = Arc::new(LanceTableProvider::new(dataset, false, false));
    let (plain, expected) = run(&context(provider.clone(), false), sql).await;
    let (rewritten, actual) = run(&context(provider, true), sql).await;
    let shown = displayable(rewritten.as_ref()).indent(true).to_string();

    assert_eq!(actual, expected, "{shown}");
    assert_eq!(actual.num_rows(), 5);
    assert_eq!(scans(&rewritten), [["id", "_rowid"]], "{shown}");
    assert_eq!(take_output_rows(&rewritten), [5], "{shown}");
    // `wide` is ~1 KB per row; the unrewritten plan reads it for every row
    // that passes the filter (at least 143 of them).
    assert!(
        bytes_read(&rewritten) * 10 < bytes_read(&plain),
        "rewritten read {} bytes, unrewritten read {}:\n{shown}",
        bytes_read(&rewritten),
        bytes_read(&plain)
    );
}

/// The window relies on the top-k's ordering instead of sorting again, so the
/// take above the sort must keep reporting it.
#[tokio::test]
async fn sql_topk_keeps_its_ordering_for_the_parent() {
    let dataset = make_dataset().await;
    let provider = Arc::new(LanceTableProvider::new(dataset, false, false));
    let sql = "SELECT id, cat, wide, row_number() OVER (PARTITION BY cat ORDER BY id) AS rn \
               FROM (SELECT id, cat, wide FROM t ORDER BY cat, id LIMIT 20)";
    let (_, expected) = run(&context(provider.clone(), false), sql).await;
    let (rewritten, actual) = run(&context(provider, true), sql).await;
    let shown = displayable(rewritten.as_ref()).indent(true).to_string();

    assert_eq!(take_output_rows(&rewritten), [20], "{shown}");
    SanityCheckPlan::new()
        .optimize(rewritten, &ConfigOptions::default())
        .unwrap_or_else(|e| panic!("{e}\n{shown}"));
    assert_eq!(actual, expected, "{shown}");
}

#[tokio::test]
async fn sql_limit_covering_the_table_is_left_alone() {
    let dataset = make_dataset().await;
    let provider = Arc::new(LanceTableProvider::new(dataset, false, false));
    let (rewritten, result) = run(
        &context(provider, true),
        "SELECT id, wide FROM t ORDER BY id LIMIT 1000",
    )
    .await;

    assert_eq!(result.num_rows(), 1000);
    assert_eq!(scans(&rewritten), [["id", "wide"]]);
    assert!(take_output_rows(&rewritten).is_empty());
}

fn erase_metadata(schema: &Schema) -> Schema {
    schema.clone().with_metadata(Default::default())
}

/// A node a `TableProvider` wraps around the Lance scan, which drops the
/// schema metadata, as lancedb's `MetadataEraserExec` does.
#[derive(Debug)]
struct MetadataEraserExec {
    input: Arc<dyn ExecutionPlan>,
    is_row_preserving: bool,
    properties: Arc<PlanProperties>,
}

impl MetadataEraserExec {
    fn new(input: Arc<dyn ExecutionPlan>, is_row_preserving: bool) -> Self {
        let schema = Arc::new(erase_metadata(&input.schema()));
        let properties = Arc::new(
            input
                .properties()
                .as_ref()
                .clone()
                .with_eq_properties(EquivalenceProperties::new(schema)),
        );
        Self {
            input,
            is_row_preserving,
            properties,
        }
    }
}

impl DisplayAs for MetadataEraserExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "MetadataEraserExec")
    }
}

impl ExecutionPlan for MetadataEraserExec {
    fn name(&self) -> &str {
        "MetadataEraserExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            children.remove(0),
            self.is_row_preserving,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let schema = self.schema();
        let stream = self.input.execute(partition, context)?.map({
            let schema = schema.clone();
            move |batch| {
                let batch = batch?;
                Ok(RecordBatch::try_new_with_options(
                    schema.clone(),
                    batch.columns().to_vec(),
                    &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
                )?)
            }
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        if self.is_row_preserving {
            CardinalityEffect::Equal
        } else {
            CardinalityEffect::Unknown
        }
    }
}

/// [`LanceTableProvider`] with its scan wrapped in a [`MetadataEraserExec`].
#[derive(Debug)]
struct ErasingTableProvider {
    inner: LanceTableProvider,
    is_row_preserving: bool,
}

#[async_trait]
impl TableProvider for ErasingTableProvider {
    fn schema(&self) -> SchemaRef {
        Arc::new(erase_metadata(&self.inner.schema()))
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(MetadataEraserExec::new(
            self.inner.scan(state, projection, filters, limit).await?,
            self.is_row_preserving,
        )))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        self.inner.supports_filters_pushdown(filters)
    }
}

/// The rule looks through a certified custom node only when it reports
/// `CardinalityEffect::Equal`, and re-applies a metadata-erasing one above the
/// take so the plan keeps its schema.
#[rstest]
#[case::keeps_every_row(true)]
#[case::unknown_cardinality(false)]
#[tokio::test]
async fn sql_topk_through_a_wrapping_provider(#[case] is_row_preserving: bool) {
    let sql = "SELECT id, wide FROM t WHERE cat = 3 ORDER BY id DESC LIMIT 5";
    let dataset = make_dataset().await;
    let provider = Arc::new(ErasingTableProvider {
        inner: LanceTableProvider::new(dataset, false, false),
        is_row_preserving,
    });
    let (plain, expected) = run(&context(provider.clone(), false), sql).await;
    let (rewritten, actual) = run(&context(provider, true), sql).await;
    let shown = displayable(rewritten.as_ref()).indent(true).to_string();

    assert!(rewritten.schema().metadata().is_empty(), "{shown}");
    assert_eq!(rewritten.schema(), plain.schema());
    assert_eq!(actual, expected, "{shown}");
    let take_rows = take_output_rows(&rewritten);
    if is_row_preserving {
        assert_eq!(take_rows, [5], "{shown}");
        assert_eq!(scans(&rewritten), [["id", "_rowid"]], "{shown}");
    } else {
        assert!(take_rows.is_empty(), "{shown}");
    }
}

/// A precomputed read selection (a distributed worker's share of the rows)
/// survives the narrowing.
#[tokio::test]
async fn topk_keeps_a_precomputed_read_plan() {
    let dataset = make_dataset().await;
    let mut rows = RowAddrTreeMap::new();
    rows.insert_bitmap(0, (0u32..20).collect());
    let read = FilteredReadExec::try_new(
        dataset.clone(),
        FilteredReadOptions::basic_full_read(&dataset),
        None,
    )
    .unwrap()
    .with_plan(FilteredReadPlan {
        rows,
        filters: Default::default(),
        scan_range_after_filter: None,
    })
    .await
    .unwrap();
    let plan = top_5_by_id(Arc::new(read));
    let rewritten = rule().optimize(plan.clone(), &Default::default()).unwrap();
    let shown = displayable(rewritten.as_ref()).indent(true).to_string();

    assert_eq!(scans(&rewritten), [["id", "_rowid"]], "{shown}");
    let ids = execute(rewritten).await;
    assert_eq!(ids, execute(plan).await, "{shown}");
    assert_eq!(
        ids.column_by_name("id")
            .unwrap()
            .as_primitive::<UInt64Type>()
            .values(),
        &[19, 18, 17, 16, 15]
    );
}

/// A precomputed selection no larger than the limit keeps every row, so the
/// read is left alone.
#[tokio::test]
async fn topk_within_a_precomputed_selection_is_left_alone() {
    let dataset = make_dataset().await;
    let mut rows = RowAddrTreeMap::new();
    rows.insert_bitmap(0, (0u32..5).collect());
    let read = FilteredReadExec::try_new(
        dataset.clone(),
        FilteredReadOptions::basic_full_read(&dataset),
        None,
    )
    .unwrap()
    .with_plan(FilteredReadPlan {
        rows,
        filters: Default::default(),
        scan_range_after_filter: None,
    })
    .await
    .unwrap();
    let plan = top_5_by_id(Arc::new(read));
    let rewritten = rule().optimize(plan.clone(), &Default::default()).unwrap();

    assert!(Arc::ptr_eq(&rewritten, &plan));
}

#[derive(Debug)]
struct Target;

impl VarProvider for Target {
    fn get_value(&self, _: Vec<String>) -> datafusion::error::Result<ScalarValue> {
        Ok(ScalarValue::Int32(Some(3)))
    }

    fn get_type(&self, _: &[String]) -> Option<DataType> {
        Some(DataType::Int32)
    }
}

/// A filter planned with session state (here a variable) keeps it.
#[tokio::test]
async fn sql_topk_keeps_session_planned_filters() {
    let sql = "SELECT id, wide FROM t WHERE cat = @target ORDER BY id DESC LIMIT 5";
    let dataset = make_dataset().await;
    let provider = Arc::new(LanceTableProvider::new(dataset, false, false));
    let plain = context(provider.clone(), false);
    let optimized = context(provider, true);
    plain.register_variable(VarType::UserDefined, Arc::new(Target));
    optimized.register_variable(VarType::UserDefined, Arc::new(Target));
    let (_, expected) = run(&plain, sql).await;
    let (rewritten, actual) = run(&optimized, sql).await;
    let shown = displayable(rewritten.as_ref()).indent(true).to_string();

    assert_eq!(actual, expected, "{shown}");
    assert_eq!(scans(&rewritten), [["id", "_rowid"]], "{shown}");
}

/// Keeps every row and column but changes `cat`'s values.
#[derive(Debug)]
struct AdjustCatExec {
    input: Arc<dyn ExecutionPlan>,
}

impl DisplayAs for AdjustCatExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "AdjustCatExec")
    }
}

impl ExecutionPlan for AdjustCatExec {
    fn name(&self) -> &str {
        "AdjustCatExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            input: children.remove(0),
        }))
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let stream = self.input.execute(partition, context)?.map(|batch| {
            let batch = batch?;
            let mut columns = batch.columns().to_vec();
            if let Ok(index) = batch.schema().index_of("cat") {
                let cat = batch.column(index).as_primitive::<Int32Type>();
                columns[index] = Arc::new(cat.unary::<_, Int32Type>(|v| v + 1000)) as _;
            }
            Ok(RecordBatch::try_new(batch.schema(), columns)?)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}

/// An uncertified node is not walked through, however row-preserving it
/// claims to be: the take above the sort would bypass it.
#[tokio::test]
async fn topk_over_an_uncertified_node_is_left_alone() {
    let dataset = make_dataset().await;
    let read = FilteredReadExec::try_new(
        dataset.clone(),
        FilteredReadOptions::basic_full_read(&dataset),
        None,
    )
    .unwrap();
    let plan = top_5_by_id(Arc::new(AdjustCatExec {
        input: Arc::new(read),
    }));
    let rewritten = rule().optimize(plan.clone(), &Default::default()).unwrap();

    assert!(Arc::ptr_eq(&rewritten, &plan));
    let cat = execute(rewritten).await;
    let cat = cat
        .column_by_name("cat")
        .unwrap()
        .as_primitive::<Int32Type>();
    assert!(cat.values().iter().all(|v| *v >= 1000));
}
