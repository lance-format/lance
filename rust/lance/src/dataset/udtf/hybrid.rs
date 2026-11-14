// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float16Array, Float32Array, Float64Array, UInt8Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableFunctionArgs, TableFunctionImpl, TableProvider};
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{DFSchema, DataFusionError, Result, ScalarValue};
use datafusion::datasource::provider_as_source;
use datafusion::execution::context::SessionState;
use datafusion::functions::core::expr_fn::coalesce;
use datafusion::functions_aggregate::expr_fn::sum;
use datafusion::functions_window::expr_fn::row_number;
use datafusion_expr::{
    Expr, ExprFunctionExt, ExprSchemable, LogicalPlan, LogicalPlanBuilder, TableType, Volatility,
    cast, col, lit,
};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::projection::ProjectionExec;
use lance_core::datatypes::{OnMissing, Projection, format_field_path_minimal};
use lance_core::{ROW_ID, ROW_ID_FIELD};
use lance_datafusion::expr::safe_coerce_scalar;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::parser::from_json;
use serde::Deserialize;
use serde_json::Number;

use crate::Dataset;
use crate::dataset::blob::public_blob_v2_binary_output_schema;
use crate::io::exec::take::TakeExec;

const RELEVANCE: &str = "_relevance_score";
const RANK: &str = "__hybrid_rank";
const CONTRIBUTION: &str = "__hybrid_contribution";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VectorQuery {
    column: String,
    q: Vec<Number>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    vector_candidates: usize,
    fts_candidates: usize,
    limit: usize,
    rrf_k: Option<f64>,
    filter: Option<String>,
    #[serde(default)]
    with_row_id: bool,
}

impl Options {
    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("vector_candidates", self.vector_candidates),
            ("fts_candidates", self.fts_candidates),
            ("limit", self.limit),
        ] {
            if value == 0 || i64::try_from(value).is_err() {
                return Err(DataFusionError::Plan(format!(
                    "hybrid {name} must be a positive i64 count, got {value}"
                )));
            }
        }
        if let Some(value) = self.rrf_k
            && (!value.is_finite() || value <= 0.0)
        {
            return Err(DataFusionError::Plan(format!(
                "hybrid rrf_k must be finite and positive, got {value}"
            )));
        }
        Ok(())
    }
}

impl VectorQuery {
    fn array(&self, schema: &lance_core::datatypes::Schema) -> Result<ArrayRef> {
        let field = schema.field(&self.column).ok_or_else(|| {
            DataFusionError::Plan(format!(
                "hybrid vector column {} does not exist",
                self.column
            ))
        })?;
        let dtype = field.data_type();
        let DataType::FixedSizeList(element, dimension) = &dtype else {
            return Err(DataFusionError::Plan(format!(
                "hybrid vector column {} must be a single FixedSizeList vector, got {}",
                self.column, dtype
            )));
        };
        if *dimension <= 0 || self.q.len() != *dimension as usize {
            return Err(DataFusionError::Plan(format!(
                "hybrid vector {} has {} values, expected {dimension}",
                self.column,
                self.q.len()
            )));
        }
        if element.data_type() == &DataType::UInt8 {
            let values = self.q.iter().map(|value| {
                value.as_u64().and_then(|v| u8::try_from(v).ok()).ok_or_else(|| {
                    DataFusionError::Plan(format!("hybrid UInt8 vector value must be an integer in 0..=255, got {value}"))
                })
            }).collect::<Result<Vec<_>>>()?;
            return Ok(Arc::new(UInt8Array::from(values)));
        }
        let values = self
            .q
            .iter()
            .map(|value| {
                value.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
                    DataFusionError::Plan(format!(
                        "hybrid vector value must be finite, got {value}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        match element.data_type() {
            DataType::Float64 => Ok(Arc::new(Float64Array::from(values))),
            DataType::Float32 if values.iter().all(|v| (*v as f32).is_finite()) => Ok(Arc::new(
                Float32Array::from(values.into_iter().map(|v| v as f32).collect::<Vec<_>>()),
            )),
            DataType::Float16 => {
                let values = values
                    .into_iter()
                    .map(|value| {
                        match safe_coerce_scalar(
                            &ScalarValue::Float64(Some(value)),
                            &DataType::Float16,
                        ) {
                            Some(ScalarValue::Float16(Some(value))) => Ok(value),
                            _ => Err(DataFusionError::Plan(format!(
                                "hybrid vector value {value} is not representable as Float16"
                            ))),
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(Float16Array::from(values)))
            }
            dtype => Err(DataFusionError::Plan(format!(
                "hybrid vector values are not supported or representable as {dtype}"
            ))),
        }
    }
}

/// Independent vector and full-text retrieval, fused with reciprocal rank fusion.
///
/// SQL arguments are `(table_name, vector_json, fts_json, options_json)`.
/// The vector object contains `column` and a single numeric array `q`.
/// Values must be finite and representable in the vector column's element type.
/// The FTS object uses the existing full-text query syntax. Options require positive
/// `vector_candidates`, `fts_candidates` and `limit`; `rrf_k` defaults to 60.
/// Each row's Float64 `_relevance_score` is the sum of `1 / (rrf_k + rank)`
/// over its appearances, with one-based ranks in each independent candidate set.
/// Ties within the returned candidates are resolved by logical row ID.
/// Vector search inherits Scanner's index metric, or its column-type default
/// (Hamming for UInt8 and L2 otherwise) when no index is used.
///
/// `filter` is an optional deterministic SQL predicate on the dataset, applied
/// before both searches. Parsing this SQL option requires a DataFusion
/// [`SessionState`]; searches without it support any planning [`Session`].
/// Vector column paths use Scanner's field-path syntax, including quoted and
/// nested fields. Outer SQL WHERE and LIMIT apply after the intrinsic
/// top-k, and may return fewer rows. Use ORDER BY for presentation order.
/// `with_row_id` defaults to false. Branch distance and FTS scores are hidden.
/// List-element full-text queries and multivectors are unsupported. Top-level
/// `_relevance_score`, `_score` and `_distance` business columns are unsupported
/// (the last two names are checked without regard to ASCII case).
///
/// Both searches and payload reads use the registered dataset snapshot. Fusion
/// uses DataFusion operators and the caller's runtime; only final winners need
/// output payload reads. Candidate budgets bound fusion state, not search-index
/// memory or the amount of data examined by a flat search.
///
/// ```
/// use std::sync::Arc;
/// use datafusion::prelude::SessionContext;
/// use lance::{Dataset, dataset::udtf::HybridQueryUDTFBuilder};
/// # async fn example(dataset: Arc<Dataset>) -> datafusion::error::Result<()> {
/// let ctx = SessionContext::new();
/// ctx.register_udtf("hybrid", Arc::new(HybridQueryUDTFBuilder::builder()
///     .register_table("documents", dataset).build()));
/// let result = ctx.sql(r#"SELECT text, _relevance_score FROM hybrid(
///     'documents', '{"column":"embedding","q":[0.1,0.2]}',
///     '{"match":{"column":"text","terms":"lakehouse"}}',
///     '{"vector_candidates":100,"fts_candidates":100,"limit":20}')
///     ORDER BY _relevance_score DESC"#).await?;
/// let batches = result.collect().await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct HybridQueryUDTF {
    datasets: HashMap<String, Arc<Dataset>>,
}

/// Registers resolved dataset snapshots for a [`HybridQueryUDTF`].
#[derive(Debug, Default)]
pub struct HybridQueryUDTFBuilder {
    datasets: HashMap<String, Arc<Dataset>>,
}

impl HybridQueryUDTFBuilder {
    /// Start an empty registry. See [`HybridQueryUDTF`] for a SQL example.
    pub fn builder() -> Self {
        Self::default()
    }

    /// Register a snapshot by name, replacing any earlier registration of it.
    pub fn register_table(mut self, table_name: impl Into<String>, dataset: Arc<Dataset>) -> Self {
        self.datasets.insert(table_name.into(), dataset);
        self
    }

    /// Build the function for registration with a DataFusion session.
    pub fn build(self) -> HybridQueryUDTF {
        HybridQueryUDTF {
            datasets: self.datasets,
        }
    }
}

impl TableFunctionImpl for HybridQueryUDTF {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let exprs = args.exprs();
        if exprs.len() != 4 {
            return Err(DataFusionError::Plan(format!(
                "hybrid requires table_name, vector_json, fts_json, options_json; got {} arguments",
                exprs.len()
            )));
        }
        let strings = exprs
            .iter()
            .enumerate()
            .map(|(i, expr)| match expr {
                Expr::Literal(
                    ScalarValue::Utf8(Some(value))
                    | ScalarValue::LargeUtf8(Some(value))
                    | ScalarValue::Utf8View(Some(value)),
                    _,
                ) => Ok(value.as_str()),
                _ => Err(DataFusionError::Plan(format!(
                    "hybrid argument {} must be a non-null string literal",
                    i + 1
                ))),
            })
            .collect::<Result<Vec<_>>>()?;
        let dataset = self
            .datasets
            .get(strings[0])
            .ok_or_else(|| {
                DataFusionError::Plan(format!("hybrid table {} is not registered", strings[0]))
            })?
            .clone();
        let vector: VectorQuery = serde_json::from_str(strings[1])
            .map_err(|e| DataFusionError::Plan(format!("invalid hybrid vector JSON: {e}")))?;
        let options: Options = serde_json::from_str(strings[3])
            .map_err(|e| DataFusionError::Plan(format!("invalid hybrid options JSON: {e}")))?;
        options.validate()?;
        let mut schema = Schema::from(&public_blob_v2_binary_output_schema(dataset.schema()));
        for field in schema.fields() {
            let name = field.name();
            if name == RELEVANCE
                || name.eq_ignore_ascii_case("_score")
                || name.eq_ignore_ascii_case("_distance")
            {
                return Err(DataFusionError::Plan(format!(
                    "hybrid conflicts with top-level dataset field {name}"
                )));
            }
        }
        let query = vector.array(dataset.schema())?;
        let filter = options
            .filter
            .as_deref()
            .map(|sql| {
                let state = args
                    .session()
                    .as_any()
                    .downcast_ref::<SessionState>()
                    .ok_or_else(|| {
                        DataFusionError::NotImplemented(
                            "hybrid SQL filters require a DataFusion SessionState".to_string(),
                        )
                    })?;
                let df_schema = DFSchema::try_from(schema.clone())?;
                let expr = state.create_logical_expr(sql, &df_schema)?;
                validate_filter(&expr)?;
                if expr.get_type(&df_schema)? != DataType::Boolean {
                    return Err(DataFusionError::Plan(
                        "hybrid filter must be Boolean".to_string(),
                    ));
                }
                Ok(expr)
            })
            .transpose()?;
        let mut fields = schema.fields().to_vec();
        fields.push(Arc::new(Field::new(RELEVANCE, DataType::Float64, false)));
        if options.with_row_id {
            fields.push(Arc::new(ROW_ID_FIELD.clone()));
        }
        schema = Schema::new_with_metadata(fields, schema.metadata().clone());
        Ok(Arc::new(HybridTableProvider {
            dataset,
            vector_column: vector.column,
            query,
            fts_query: FullTextSearchQuery::new_query(from_json(strings[2])?),
            options,
            filter,
            schema: Arc::new(schema),
        }))
    }
}

fn validate_filter(expr: &Expr) -> Result<()> {
    expr.apply(|expr| {
        let allowed = match expr {
            Expr::Column(_)
            | Expr::Literal(_, _)
            | Expr::BinaryExpr(_)
            | Expr::Not(_)
            | Expr::IsNull(_)
            | Expr::IsNotNull(_)
            | Expr::IsTrue(_)
            | Expr::IsFalse(_)
            | Expr::IsNotTrue(_)
            | Expr::IsNotFalse(_)
            | Expr::IsUnknown(_)
            | Expr::IsNotUnknown(_)
            | Expr::Negative(_)
            | Expr::Between(_)
            | Expr::Case(_)
            | Expr::Cast(_)
            | Expr::TryCast(_)
            | Expr::InList(_)
            | Expr::Like(_)
            | Expr::SimilarTo(_) => true,
            Expr::ScalarFunction(function) => {
                function.func.signature().volatility == Volatility::Immutable
            }
            _ => false,
        };
        if !allowed {
            return Err(DataFusionError::Plan(format!(
                "hybrid filter requires deterministic scalar expressions, got {expr}"
            )));
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(())
}

#[derive(Debug)]
struct HybridTableProvider {
    dataset: Arc<Dataset>,
    vector_column: String,
    query: ArrayRef,
    fts_query: FullTextSearchQuery,
    options: Options,
    filter: Option<Expr>,
    schema: SchemaRef,
}

#[async_trait]
impl TableProvider for HybridTableProvider {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut vector = self.dataset.scan();
        vector.nearest(
            &self.vector_column,
            self.query.as_ref(),
            self.options.vector_candidates,
        )?;
        vector.disable_scoring_autoprojection().prefilter(true);
        vector.project(&[ROW_ID, "_distance"])?;
        let mut fts = self.dataset.scan();
        let query = fts.resolve_full_text_search_query(&self.fts_query).await?;
        if fts
            .fts_document_granularity(&query.query)?
            .is_list_element()
        {
            return Err(DataFusionError::Plan(
                "hybrid requires row-level FTS; ListElement queries are unsupported".to_string(),
            ));
        }
        fts.full_text_search(query)?;
        fts.disable_scoring_autoprojection().prefilter(true);
        fts.project(&[ROW_ID, "_score"])?;
        fts.limit(Some(self.options.fts_candidates as i64), None)?;
        if let Some(filter) = &self.filter {
            vector.filter_expr(filter.clone());
            fts.filter_expr(filter.clone());
        }
        let vector = vector.create_plan_with_session(state).await?;
        let fts = fts.create_plan_with_session(state).await?;
        let rrf_k = self.options.rrf_k.unwrap_or(60.0);
        let logical = fuse_candidates(vector, fts, rrf_k, self.options.limit)?;
        let mut plan = state.create_physical_plan(&logical).await?;
        let indices = projection
            .cloned()
            .unwrap_or_else(|| (0..self.schema.fields().len()).collect());
        let projected_schema = self.schema.project(&indices)?;
        let mut payload = Projection::empty(self.dataset.clone());
        for &index in &indices {
            let name = self.schema.field(index).name();
            if name != RELEVANCE && name != ROW_ID {
                payload =
                    payload.union_column(format_field_path_minimal(&[name]), OnMissing::Error)?;
            }
        }
        if let Some(take) = TakeExec::try_new(self.dataset.clone(), plan.clone(), payload)? {
            plan = Arc::new(take);
        }
        let expressions = indices
            .iter()
            .map(|&index| {
                let name = self.schema.field(index).name();
                Ok((
                    Arc::new(Column::new_with_schema(name, plan.schema().as_ref())?) as _,
                    name.clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(ProjectionExec::try_new_with_schema_metadata(
            expressions,
            plan,
            &projected_schema,
        )?))
    }
}

fn fuse_candidates(
    vector: Arc<dyn ExecutionPlan>,
    fts: Arc<dyn ExecutionPlan>,
    rrf_k: f64,
    limit: usize,
) -> Result<LogicalPlan> {
    contributions(vector, "_distance", true, rrf_k)?
        .union(contributions(fts, "_score", false, rrf_k)?.build()?)?
        .aggregate(
            vec![col(ROW_ID)],
            vec![sum(col(CONTRIBUTION)).alias(CONTRIBUTION)],
        )?
        .project(vec![
            col(ROW_ID),
            coalesce(vec![col(CONTRIBUTION), lit(0.0_f64)]).alias(RELEVANCE),
        ])?
        .sort(vec![
            col(RELEVANCE).sort(false, false),
            col(ROW_ID).sort(true, false),
        ])?
        .limit(0, Some(limit))?
        .build()
}

// A logical leaf lets the caller's planner enforce global window/aggregate
// distribution while preserving Scanner's snapshot and search coverage plan.
#[derive(Debug)]
struct CandidatePlan(Arc<dyn ExecutionPlan>);

#[async_trait]
impl TableProvider for CandidatePlan {
    fn schema(&self) -> SchemaRef {
        self.0.schema()
    }
    fn table_type(&self) -> TableType {
        TableType::Temporary
    }
    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match projection {
            None => Ok(self.0.clone()),
            Some(indices) => {
                let expressions = indices
                    .iter()
                    .map(|&index| {
                        let name = self.0.schema().field(index).name().clone();
                        (Arc::new(Column::new(&name, index)) as _, name)
                    })
                    .collect::<Vec<_>>();
                Ok(Arc::new(ProjectionExec::try_new(
                    expressions,
                    self.0.clone(),
                )?))
            }
        }
    }
}

fn contributions(
    plan: Arc<dyn ExecutionPlan>,
    metric: &str,
    ascending: bool,
    rrf_k: f64,
) -> Result<LogicalPlanBuilder> {
    let rank = row_number()
        .order_by(vec![
            col(metric).sort(ascending, false),
            col(ROW_ID).sort(true, false),
        ])
        .build()?
        .alias(RANK);
    LogicalPlanBuilder::scan(
        "candidates",
        provider_as_source(Arc::new(CandidatePlan(plan))),
        None,
    )?
    .window(vec![rank])?
    .project(vec![
        col(ROW_ID),
        (lit(1.0_f64) / (lit(rrf_k) + cast(col(RANK), DataType::Float64))).alias(CONTRIBUTION),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float16Type, Float64Type, UInt8Type};
    use arrow_array::{RecordBatch, UInt64Array};
    use datafusion::execution::TaskContext;
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion_physical_plan::streaming::{PartitionStream, StreamingTableExec};
    use datafusion_physical_plan::{SendableRecordBatchStream, collect};
    use futures::stream;
    use rstest::rstest;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::Notify;

    struct LiveStream(Arc<AtomicUsize>);

    impl Drop for LiveStream {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[derive(Debug)]
    struct CandidateSource {
        schema: SchemaRef,
        batch: RecordBatch,
        live: Arc<AtomicUsize>,
        pending: bool,
        failure: Option<Arc<Notify>>,
    }

    impl PartitionStream for CandidateSource {
        fn schema(&self) -> &SchemaRef {
            &self.schema
        }

        fn execute(&self, _: Arc<TaskContext>) -> SendableRecordBatchStream {
            self.live.fetch_add(1, Ordering::SeqCst);
            let state = (
                Some(self.batch.clone()),
                LiveStream(self.live.clone()),
                self.pending,
                self.failure.clone(),
            );
            Box::pin(RecordBatchStreamAdapter::new(
                self.schema.clone(),
                stream::unfold(state, |(batch, guard, pending, failure)| async move {
                    if let Some(batch) = batch {
                        return Some((Ok(batch), (None, guard, pending, failure)));
                    }
                    if let Some(failure) = failure {
                        failure.notified().await;
                        return Some((
                            Err(DataFusionError::Execution(
                                "candidate branch failed".to_string(),
                            )),
                            (None, guard, true, None),
                        ));
                    }
                    if pending {
                        std::future::pending::<()>().await;
                    }
                    None
                }),
            ))
        }
    }

    fn candidate_source(
        metric: &str,
        live: Arc<AtomicUsize>,
        pending: bool,
        failure: Option<Arc<Notify>>,
    ) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            ROW_ID_FIELD.clone(),
            Field::new(metric, DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from((0..32).rev().collect::<Vec<_>>())),
                Arc::new(Float32Array::from(
                    (0..32).map(|i| i as f32).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        let partition = Arc::new(CandidateSource {
            schema: schema.clone(),
            batch,
            live,
            pending,
            failure,
        });
        Arc::new(
            StreamingTableExec::try_new(schema, vec![partition], None, [], false, None).unwrap(),
        )
    }

    #[tokio::test]
    async fn fusion_charges_caller_memory_pool() {
        let pool = Arc::new(GreedyMemoryPool::new(1));
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool.clone())
            .build()
            .unwrap();
        let ctx = SessionContext::new_with_config_rt(SessionConfig::new(), Arc::new(runtime));
        let live = Arc::new(AtomicUsize::new(0));
        let logical = fuse_candidates(
            candidate_source("_distance", live.clone(), false, None),
            candidate_source("_score", live.clone(), false, None),
            60.0,
            10,
        )
        .unwrap();
        let plan = ctx.state().create_physical_plan(&logical).await.unwrap();
        // These leaves do not reserve memory: the failure must come from fusion.
        let error = collect(plan, ctx.task_ctx()).await.unwrap_err();
        assert!(
            matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{error}"
        );
        tokio::time::timeout(Duration::from_secs(1), async {
            while live.load(Ordering::SeqCst) != 0 || pool.reserved() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[rstest]
    #[case::cancellation(false)]
    #[case::branch_error(true)]
    #[tokio::test]
    async fn fusion_releases_both_branches(#[case] fail: bool) {
        let pool = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool.clone())
            .build()
            .unwrap();
        let ctx = SessionContext::new_with_config_rt(
            SessionConfig::new().with_target_partitions(2),
            Arc::new(runtime),
        );
        let live = Arc::new(AtomicUsize::new(0));
        let failure = Arc::new(Notify::new());
        let logical = fuse_candidates(
            candidate_source("_distance", live.clone(), true, Some(failure.clone())),
            candidate_source("_score", live.clone(), true, None),
            60.0,
            10,
        )
        .unwrap();
        let plan = ctx.state().create_physical_plan(&logical).await.unwrap();
        let mut result = Box::pin(collect(plan, ctx.task_ctx()));
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                tokio::select! {
                    result = &mut result => panic!("branches finished early: {result:?}"),
                    _ = tokio::task::yield_now() => {
                        if live.load(Ordering::SeqCst) == 2 && pool.reserved() > 0 {
                            break;
                        }
                    }
                }
            }
        })
        .await
        .unwrap();
        if fail {
            failure.notify_one();
            let error = tokio::time::timeout(Duration::from_secs(1), &mut result)
                .await
                .unwrap()
                .unwrap_err();
            assert!(
                error.to_string().contains("candidate branch failed"),
                "{error}"
            );
        }
        drop(result);
        tokio::time::timeout(Duration::from_secs(1), async {
            while live.load(Ordering::SeqCst) != 0 || pool.reserved() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    fn vector_schema(dtype: DataType) -> lance_core::datatypes::Schema {
        lance_core::datatypes::Schema::try_from(&Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", dtype, true)), 2),
            false,
        )]))
        .unwrap()
    }

    #[rstest]
    #[case::f16(DataType::Float16)]
    #[case::f32(DataType::Float32)]
    #[case::f64(DataType::Float64)]
    #[case::u8(DataType::UInt8)]
    fn preserves_vector_representation(#[case] dtype: DataType) {
        let query: VectorQuery = serde_json::from_str(r#"{"column":"v","q":[1,2]}"#).unwrap();
        let values = query.array(&vector_schema(dtype.clone())).unwrap();
        assert_eq!(values.data_type(), &dtype);
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn preserves_double_precision_and_unsigned_bytes() {
        let query: VectorQuery =
            serde_json::from_str(r#"{"column":"v","q":[1.0000000000001,2]}"#).unwrap();
        assert_eq!(
            query
                .array(&vector_schema(DataType::Float64))
                .unwrap()
                .as_primitive::<Float64Type>()
                .value(0),
            1.0000000000001
        );
        let query: VectorQuery = serde_json::from_str(r#"{"column":"v","q":[0,255]}"#).unwrap();
        assert_eq!(
            query
                .array(&vector_schema(DataType::UInt8))
                .unwrap()
                .as_primitive::<UInt8Type>()
                .values(),
            &[0, 255]
        );
    }

    #[test]
    fn rounds_half_precision_without_double_rounding() {
        let query: VectorQuery =
            serde_json::from_str(r#"{"column":"v","q":[1.0004882812500002,1.00048828125]}"#)
                .unwrap();
        let values = query.array(&vector_schema(DataType::Float16)).unwrap();
        let values = values.as_primitive::<Float16Type>();
        assert_eq!(values.value(0).to_bits(), 0x3c01);
        assert_eq!(values.value(1).to_bits(), 0x3c00);
        let query: VectorQuery = serde_json::from_str(r#"{"column":"v","q":[1e-100,0]}"#).unwrap();
        let error = query.array(&vector_schema(DataType::Float16)).unwrap_err();
        assert!(matches!(error, DataFusionError::Plan(_)));
        assert!(error.to_string().contains("not representable as Float16"));
    }

    #[rstest]
    #[case::fraction("1.5")]
    #[case::negative("-1")]
    #[case::overflow("256")]
    fn rejects_lossy_byte_conversion(#[case] value: &str) {
        let query: VectorQuery =
            serde_json::from_str(&format!(r#"{{"column":"v","q":[{value},0]}}"#)).unwrap();
        let error = query.array(&vector_schema(DataType::UInt8)).unwrap_err();
        assert!(matches!(error, DataFusionError::Plan(_)));
        assert!(error.to_string().contains("integer in 0..=255"));
    }
}
