// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end tests for [`TopKLateMaterialization`] on SQL top-k plans built
//! through [`LanceTableProvider`], including one wrapped in a custom node, and
//! on reads the scanner or a distributed planner has already configured.

use std::sync::Arc;

use arrow::datatypes::{Int32Type, UInt64Type};
use arrow_array::cast::AsArray;
use arrow_array::{
    Int32Array, RecordBatch, RecordBatchIterator, RecordBatchOptions, StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
use arrow_select::take::take_record_batch;
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
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, PhysicalExpr, PhysicalSortExpr,
};
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
use lance::datafusion::LanceTableProvider;
use lance::dataset::WriteParams;
use lance::io::exec::TakeExec;
use lance::io::exec::filtered_read::{FilteredReadExec, FilteredReadOptions, FilteredReadPlan};
use lance::io::exec::topk_late_materialization::TopKLateMaterialization;
use lance::{BlobArrayBuilder, Dataset, blob_field};
use lance_core::datatypes::BlobHandling;
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

/// 3 fragments of 4 rows: `id` is the row number, `text` is derived from it,
/// and `blob` is a blob v2 column holding `id` bytes, null every third row.
async fn make_blob_dataset() -> Arc<Dataset> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
        blob_field("blob", true),
    ]));
    let ids = (0..12).collect::<Vec<i32>>();
    let texts = StringArray::from_iter_values(ids.iter().map(|id| format!("text {id}")));
    let mut blobs = BlobArrayBuilder::new(ids.len());
    for id in &ids {
        if id % 3 == 0 {
            blobs.push_null().unwrap();
        } else {
            // Distinct non-null blob sizes expose incorrect row fetches in
            // either the descriptor or binary representation.
            blobs.push_bytes(vec![*id as u8; *id as usize]).unwrap();
        }
    }
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(texts),
            blobs.finish().unwrap(),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    // A wrapping provider erases this, so the eraser must apply above the take.
    dataset
        .update_schema_metadata([("owner", "test")])
        .await
        .unwrap();
    Arc::new(dataset)
}

/// In descriptor mode, the planner stamps the provider's logical blob metadata
/// onto its projection, while the take fetches descriptors with storage metadata.
/// Binary mode also checks the representation of deferred payloads, and a
/// wrapping provider checks that schema metadata stays erased after the take.
#[rstest]
#[case::expression_sort_key("SELECT id, blob FROM t ORDER BY md5(text), id LIMIT 2", 1)]
#[case::filtered_subquery(
    "WITH s AS (SELECT id, blob, text FROM t WHERE blob IS NOT NULL \
     ORDER BY md5(concat(text, ':42')), id LIMIT 2) \
     SELECT id, blob FROM s ORDER BY id",
    0
)]
#[tokio::test]
async fn sql_topk_takes_a_blob_column(
    #[case] sql: &str,
    #[case] expected_nulls: usize,
    #[values(None, Some(BlobHandling::AllBinary))] blob_handling: Option<BlobHandling>,
    #[values(false, true)] is_wrapped: bool,
) {
    let mut provider = LanceTableProvider::new(make_blob_dataset().await, false, false);
    if let Some(blob_handling) = blob_handling {
        provider = provider.with_blob_handling(blob_handling);
    }
    let provider: Arc<dyn TableProvider> = if is_wrapped {
        Arc::new(ErasingTableProvider {
            inner: provider,
            is_row_preserving: true,
            is_reversing: false,
        })
    } else {
        Arc::new(provider)
    };
    let (_, expected) = run(&context(provider.clone(), false), sql).await;
    let (rewritten, actual) = run(&context(provider, true), sql).await;
    let shown = displayable(rewritten.as_ref()).indent(true).to_string();

    assert_eq!(take_output_rows(&rewritten), [2], "{shown}");
    assert_eq!(scans(&rewritten), [["id", "text", "_rowid"]], "{shown}");
    assert_eq!(actual["blob"].null_count(), expected_nulls, "{shown}");
    assert_eq!(actual, expected, "{shown}");
}

/// A sort key can read a field's metadata, so the sort over the narrowed read
/// must see the metadata the planner gave `blob`.
#[tokio::test]
async fn sql_topk_sorting_on_field_metadata_keeps_its_order() {
    let provider = Arc::new(LanceTableProvider::new(
        make_blob_dataset().await,
        false,
        false,
    ));
    let sql = "SELECT blob, text FROM t \
               ORDER BY CASE WHEN arrow_metadata(blob, 'ARROW:extension:name') = 'lance.blob.v2' \
               THEN -id ELSE id END LIMIT 2";
    let (_, expected) = run(&context(provider.clone(), false), sql).await;
    let (rewritten, actual) = run(&context(provider, true), sql).await;
    let shown = displayable(rewritten.as_ref()).indent(true).to_string();

    assert_eq!(
        actual["text"].as_string::<i32>(),
        &StringArray::from(vec!["text 11", "text 10"]),
        "{shown}"
    );
    assert_eq!(take_output_rows(&rewritten), [2], "{shown}");
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
    /// Reverses each batch, which a node that does not report keeping its
    /// input's order may do.
    is_reversing: bool,
    properties: Arc<PlanProperties>,
}

impl MetadataEraserExec {
    fn new(input: Arc<dyn ExecutionPlan>, is_row_preserving: bool, is_reversing: bool) -> Self {
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
            is_reversing,
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

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> datafusion::error::Result<TreeNodeRecursion>,
    ) -> datafusion::error::Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            children.remove(0),
            self.is_row_preserving,
            self.is_reversing,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let schema = self.schema();
        let is_reversing = self.is_reversing;
        let stream = self.input.execute(partition, context)?.map({
            let schema = schema.clone();
            move |batch| {
                let mut batch = batch?;
                if is_reversing {
                    let indices = UInt32Array::from_iter_values((0..batch.num_rows() as u32).rev());
                    batch = take_record_batch(&batch, &indices)?;
                }
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

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![!self.is_reversing]
    }
}

/// [`LanceTableProvider`] with its scan wrapped in a [`MetadataEraserExec`].
#[derive(Debug)]
struct ErasingTableProvider {
    inner: LanceTableProvider,
    is_row_preserving: bool,
    is_reversing: bool,
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
            self.is_reversing,
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
/// take so the plan keeps its schema. It declines when that node may reorder
/// rows, which would undo the sort.
#[rstest]
#[case::keeps_every_row(true, false)]
#[case::unknown_cardinality(false, false)]
#[case::reorders_rows(true, true)]
#[tokio::test]
async fn sql_topk_through_a_wrapping_provider(
    #[case] is_row_preserving: bool,
    #[case] is_reversing: bool,
) {
    let sql = "SELECT id, wide FROM t WHERE cat = 3 ORDER BY id DESC LIMIT 5";
    let dataset = make_dataset().await;
    let provider = Arc::new(ErasingTableProvider {
        inner: LanceTableProvider::new(dataset, false, false),
        is_row_preserving,
        is_reversing,
    });
    let (plain, expected) = run(&context(provider.clone(), false), sql).await;
    let (rewritten, actual) = run(&context(provider, true), sql).await;
    let shown = displayable(rewritten.as_ref()).indent(true).to_string();

    assert!(rewritten.schema().metadata().is_empty(), "{shown}");
    assert_eq!(rewritten.schema(), plain.schema());
    assert_eq!(actual, expected, "{shown}");
    let take_rows = take_output_rows(&rewritten);
    if is_row_preserving && !is_reversing {
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

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> datafusion::error::Result<TreeNodeRecursion>,
    ) -> datafusion::error::Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
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

/// A take that adds `s.x` to an input `s` changes that column, so a sort on
/// `s` (or a field of it) must stay above the take: below it, the sort would
/// see only `s.y`.
#[rstest]
#[case::existing_field("y")]
#[case::new_field("x")]
#[case::whole_struct("all")]
#[tokio::test]
async fn sort_stays_above_a_take_that_extends_a_struct(#[case] field_name: &str) {
    use arrow_array::{Array, ArrayRef, Int32Array, RecordBatchIterator, StructArray};
    use arrow_schema::Field;
    use datafusion::common::DFSchema;
    use datafusion::functions::core::expr_fn::get_field;
    use datafusion::physical_optimizer::ensure_requirements::EnsureRequirements;
    use datafusion::prelude::col as logical_col;
    use lance_core::datatypes::OnMissing;

    let nested = StructArray::from(vec![
        (
            Arc::new(Field::new("x", DataType::Int32, false)),
            Arc::new(Int32Array::from(vec![3, 1, 2])) as ArrayRef,
        ),
        (
            Arc::new(Field::new("y", DataType::Int32, false)),
            Arc::new(Int32Array::from(vec![2, 3, 1])) as ArrayRef,
        ),
    ]);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "s",
        nested.data_type().clone(),
        false,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(nested)]).unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://",
            None,
        )
        .await
        .unwrap(),
    );
    // Built fresh for each run: executing a read consumes it.
    let make_plan = || {
        let projection = dataset
            .empty_projection()
            .union_column("s.y", OnMissing::Error)
            .unwrap()
            .with_row_id();
        let input = Arc::new(
            FilteredReadExec::try_new(dataset.clone(), FilteredReadOptions::new(projection), None)
                .unwrap(),
        );
        let extra = dataset
            .empty_projection()
            .union_column("s.x", OnMissing::Error)
            .unwrap();
        let take: Arc<dyn ExecutionPlan> = Arc::new(
            TakeExec::try_new(dataset.clone(), input, extra)
                .unwrap()
                .unwrap(),
        );
        let logical_expr = if field_name == "all" {
            logical_col("s")
        } else {
            get_field(logical_col("s"), field_name)
        };
        let expr = SessionContext::new()
            .create_physical_expr(
                logical_expr,
                &DFSchema::try_from(take.schema().as_ref().clone()).unwrap(),
            )
            .unwrap();
        let ordering =
            LexOrdering::new([PhysicalSortExpr::new(expr, SortOptions::default())]).unwrap();
        Arc::new(SortExec::new(ordering, take)) as Arc<dyn ExecutionPlan>
    };
    let expected = execute(make_plan()).await;
    let optimized = EnsureRequirements::new()
        .optimize(make_plan(), &ConfigOptions::default())
        .unwrap();
    let shown = displayable(optimized.as_ref()).indent(true).to_string();
    SanityCheckPlan::new()
        .optimize(optimized.clone(), &ConfigOptions::default())
        .unwrap_or_else(|e| panic!("{e}\n{shown}"));
    assert_eq!(execute(optimized).await, expected, "{shown}");
}
