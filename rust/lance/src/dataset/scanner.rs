// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;

use datafusion::config::ConfigOptions;
use lance_select::result::IndexExprResultWireFormat;
use std::ops::Range;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use std::task::{Context, Poll};

use crate::index::DatasetIndexExt;
use arrow::array::AsArray;
use arrow_array::{Array, Float32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef, SortOptions};
use arrow_select::concat::concat_batches;
use async_recursion::async_recursion;
use chrono::Utc;
use datafusion::common::{DFSchema, JoinType, NullEquality, SchemaExt, exec_datafusion_err};
use datafusion::functions_aggregate;
use datafusion::logical_expr::{Expr, ScalarUDF, col, lit};
use datafusion::physical_expr::PhysicalSortExpr;
#[allow(deprecated)]
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::expressions;
use datafusion::physical_plan::projection::ProjectionExec as DFProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{
    ExecutionPlan, SendableRecordBatchStream,
    aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
    display::DisplayableExecutionPlan,
    limit::GlobalLimitExec,
    repartition::RepartitionExec,
    union::UnionExec,
};
use datafusion::scalar::ScalarValue;
use datafusion_expr::ExprSchemable;
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_physical_expr::expressions::{Column, Literal};
use datafusion_physical_expr::{LexOrdering, Partitioning, PhysicalExpr, create_physical_expr};
use datafusion_physical_plan::joins::PartitionMode;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{empty::EmptyExec, joins::HashJoinExec};
use futures::future::BoxFuture;
use futures::stream::{Stream, StreamExt};
use futures::{FutureExt, TryStreamExt};
use lance_arrow::floats::{FloatType, coerce_float_vector};
use lance_arrow::{DataTypeExt, SchemaExt as ArrowSchemaExt};
use lance_core::datatypes::{
    BlobHandling, Field, OnMissing, Projection, escape_field_path_for_project, format_field_path,
};
use lance_core::error::LanceOptionExt;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{ROW_ADDR, ROW_ID, ROW_OFFSET};
use lance_datafusion::aggregate::Aggregate;
use lance_datafusion::exec::{
    LanceExecutionOptions, OneShotExec, StrictBatchSizeExec, analyze_plan, execute_plan,
};
use lance_datafusion::expr::safe_coerce_scalar;
use lance_datafusion::projection::ProjectionPlan;
use lance_file::reader::FileReaderOptions;
use lance_index::IndexCriteria;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::expression::PlannerIndexExt;
use lance_index::scalar::expression::ScalarIndexExpr;
use lance_index::scalar::inverted::query::{
    FtsQuery, FtsQueryNode, FtsSearchParams, MatchQuery, PhraseQuery, fill_fts_query_column,
};
use lance_index::scalar::inverted::{SCORE_COL, SCORE_FIELD};
use lance_index::vector::{ApproxMode, DEFAULT_QUERY_PARALLELISM, DIST_COL, Query};
use lance_index::{metrics::NoOpMetricsCollector, scalar::inverted::FTS_SCHEMA};
use lance_io::stream::RecordBatchStream;
use lance_linalg::distance::MetricType;
use lance_select::{IndexExprResult, RowAddrMask, RowAddrTreeMap};
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;
use tracing::{Span, info_span, instrument};
use uuid::Uuid;

use super::Dataset;
use crate::dataset::row_offsets_to_row_addresses;
use crate::dataset::utils::SchemaAdapter;
use crate::index::DatasetIndexInternalExt;
use crate::index::scalar::inverted::{load_segment_details, load_segments};
use crate::index::scalar_logical::scalar_index_fragment_bitmap;
use crate::index::vector::utils::{
    default_distance_type_for, get_vector_dim, get_vector_type, validate_distance_type_for,
};
use crate::io::exec::filtered_read::{FilteredReadExec, FilteredReadOptions};
use crate::io::exec::fts::{
    BoostQueryExec, FlatMatchFilterExec, FlatMatchQueryExec, MatchQueryExec, PhraseQueryExec,
};
use crate::io::exec::knn::MultivectorScoringExec;
use crate::io::exec::scalar_index::{MaterializeIndexExec, ScalarIndexExec};
use crate::io::exec::{
    AddRowAddrExec, FilterPlan as ExprFilterPlan, KNNVectorDistanceExec, LancePushdownScanExec,
    LanceScanExec, Planner, PreFilterSource, ScanConfig, TakeExec,
    knn::{
        KnnBatchParams, QUERY_INDEX_COL, knn_empty_result_schema, new_knn_exec, query_index_field,
    },
    project,
};
use crate::io::exec::{
    AddRowOffsetExec, LANCE_RELATIONAL_ALGEBRA_VERSION, LanceFilterExec, LanceScanConfig,
    get_physical_optimizer,
};
use crate::{Error, Result};
use crate::{
    datatypes::Schema,
    io::exec::fts::{BoolSlot, BooleanQueryExec, build_boolean_query_children},
};

pub use lance_datafusion::exec::{ExecutionStatsCallback, ExecutionSummaryCounts};
#[cfg(feature = "substrait")]
use lance_datafusion::substrait::parse_substrait;

pub(crate) const BATCH_SIZE_FALLBACK: usize = 8192;

/// Parse an environment variable as a specific type, logging a warning on parse failure.
fn parse_env_var<T: std::str::FromStr>(env_var_name: &str, default_val: &str) -> Option<T>
where
    T::Err: std::fmt::Display,
{
    std::env::var(env_var_name)
        .ok()
        .and_then(|val| match val.parse() {
            Ok(value) => Some(value),
            Err(e) => {
                log::warn!(
                    "Failed to parse the environment variable {}='{}': {}, the default value is: {}.",
                    env_var_name,
                    val,
                    e,
                    default_val
                );
                None
            }
        })
}

// For backwards compatibility / historical reasons we re-calculate the default batch size
// on each call
pub fn get_default_batch_size() -> Option<usize> {
    parse_env_var("LANCE_DEFAULT_BATCH_SIZE", &BATCH_SIZE_FALLBACK.to_string())
}

pub const LEGACY_DEFAULT_FRAGMENT_READAHEAD: usize = 4;

pub static DEFAULT_FRAGMENT_READAHEAD: LazyLock<Option<usize>> = LazyLock::new(|| {
    parse_env_var(
        "LANCE_DEFAULT_FRAGMENT_READAHEAD",
        &LEGACY_DEFAULT_FRAGMENT_READAHEAD.to_string(),
    )
});

const DEFAULT_XTR_OVERFETCH_VALUE: u32 = 10;

pub static DEFAULT_XTR_OVERFETCH: LazyLock<u32> = LazyLock::new(|| {
    parse_env_var(
        "LANCE_XTR_OVERFETCH",
        &DEFAULT_XTR_OVERFETCH_VALUE.to_string(),
    )
    .unwrap_or(DEFAULT_XTR_OVERFETCH_VALUE)
});

// We want to support ~256 concurrent reads to maximize throughput on cloud storage systems
// Our typical page size is 8MiB (though not all reads are this large yet due to offset buffers, validity buffers, etc.)
// So we want to support 256 * 8MiB ~= 2GiB of queued reads
const DEFAULT_IO_BUFFER_SIZE_VALUE: u64 = 2 * 1024 * 1024 * 1024;

pub static DEFAULT_IO_BUFFER_SIZE: LazyLock<u64> = LazyLock::new(|| {
    parse_env_var(
        "LANCE_DEFAULT_IO_BUFFER_SIZE",
        &DEFAULT_IO_BUFFER_SIZE_VALUE.to_string(),
    )
    .unwrap_or(DEFAULT_IO_BUFFER_SIZE_VALUE)
});

/// The user-set value of `LANCE_DEFAULT_IO_BUFFER_SIZE`, or `None` if the env var
/// is unset or unparsable. Consult this from paths that have a sensible non-fixed
/// default (e.g. `SchedulerConfig::max_bandwidth`) so the env var still takes
/// precedence over that default. Re-reads the env var on each call so tests can
/// mutate it.
pub fn get_default_io_buffer_size_override() -> Option<u64> {
    parse_env_var(
        "LANCE_DEFAULT_IO_BUFFER_SIZE",
        &DEFAULT_IO_BUFFER_SIZE_VALUE.to_string(),
    )
}

/// Defines an ordering for a single column
///
/// Floats are sorted using the IEEE 754 total ordering
/// Strings are sorted using UTF-8 lexicographic order (i.e. we sort the binary)
#[derive(Debug, Clone)]
pub struct ColumnOrdering {
    pub ascending: bool,
    pub nulls_first: bool,
    pub column_name: String,
}

impl ColumnOrdering {
    pub fn asc_nulls_first(column_name: String) -> Self {
        Self {
            ascending: true,
            nulls_first: true,
            column_name,
        }
    }

    pub fn asc_nulls_last(column_name: String) -> Self {
        Self {
            ascending: true,
            nulls_first: false,
            column_name,
        }
    }

    pub fn desc_nulls_first(column_name: String) -> Self {
        Self {
            ascending: false,
            nulls_first: true,
            column_name,
        }
    }

    pub fn desc_nulls_last(column_name: String) -> Self {
        Self {
            ascending: false,
            nulls_first: false,
            column_name,
        }
    }
}

/// Materialization style for the scanner
///
/// This only affects columns that are not used in a filter
///
/// Early materialization will fetch the entire column and throw
/// away the rows that are not needed.  This fetches more data but
/// uses fewer I/O requests.
///
/// Late materialization will only fetch the rows that are needed.
/// This fetches less data but uses more I/O requests.
///
/// This parameter only affects scans.  Vector search and full text search
/// always use late materialization.
#[derive(Clone)]
pub enum MaterializationStyle {
    /// Heuristic-based materialization style
    ///
    /// The default approach depends on the type of object storage.  For
    /// cloud storage (e.g. S3, GCS, etc.) we only use late materialization
    /// for columns that are more than 1000 bytes in size.
    ///
    /// For local storage we use late materialization for columns that are
    /// more than 10 bytes in size.
    ///
    /// These values are based on experimentation and the assumption that a
    /// filter will be selecting ~0.1% of the rows in a column.
    Heuristic,
    /// All columns will be fetched with late materialization where possible
    AllLate,
    /// All columns will be fetched with early materialization where possible
    AllEarly,
    /// All columns will be fetched with late materialization except for the specified columns
    AllEarlyExcept(Vec<u32>),
}

impl MaterializationStyle {
    pub fn all_early_except(columns: &[impl AsRef<str>], schema: &Schema) -> Result<Self> {
        let field_ids = schema
            .project(columns)?
            .field_ids()
            .into_iter()
            .map(|id| id as u32)
            .collect();
        Ok(Self::AllEarlyExcept(field_ids))
    }
}

#[derive(Debug)]
struct PlannedFilteredScan {
    plan: Arc<dyn ExecutionPlan>,
    limit_pushed_down: bool,
    filter_pushed_down: bool,
}

pub struct FilterPlan {
    // Query filter plan
    query_filter: Option<QueryFilter>,
    refine_query_filter: bool,
    // Expr filter plan
    expr_filter_plan: ExprFilterPlan,
}

impl FilterPlan {
    pub fn new(query_filter: Option<QueryFilter>, expr_filter_plan: ExprFilterPlan) -> Self {
        Self {
            query_filter,
            refine_query_filter: false,
            expr_filter_plan,
        }
    }

    pub fn disable_refine(&mut self) {
        self.expr_filter_plan = ExprFilterPlan::default();
        self.refine_query_filter = false;
    }

    pub fn make_refine_only(&mut self) {
        self.expr_filter_plan.make_refine_only();
        self.refine_query_filter = true;
    }

    pub fn fts_filter(&self) -> Option<FullTextSearchQuery> {
        match &self.query_filter {
            Some(QueryFilter::Fts(query)) => Some(query.clone()),
            _ => None,
        }
    }

    pub fn vector_filter(&self) -> Option<Query> {
        match &self.query_filter {
            Some(QueryFilter::Vector(query)) => Some(query.clone()),
            _ => None,
        }
    }

    pub fn has_refine(&self) -> bool {
        self.expr_filter_plan.has_refine() || self.refine_query_filter
    }

    pub async fn refine_columns(&self, dataset: &Arc<Dataset>) -> Result<Vec<String>> {
        let mut columns = vec![];

        if self.expr_filter_plan.has_refine() {
            columns.extend(self.expr_filter_plan.refine_columns());
        }

        if self.refine_query_filter {
            match &self.query_filter {
                Some(QueryFilter::Fts(fts_query)) => {
                    let cols = if fts_query.columns().is_empty() {
                        let indexed_columns = fts_indexed_columns(dataset.clone()).await?;
                        let q = fill_fts_query_column(&fts_query.query, &indexed_columns, false)?;
                        q.columns()
                    } else {
                        fts_query.columns()
                    };

                    // Add refine column for match query since it supports `FlatMatchQueryExec`.
                    // Other fts query use join so we don't need to add refine column.
                    if let FtsQuery::Match(_) = &fts_query.query {
                        columns.extend(cols.iter().cloned().collect::<Vec<_>>());
                    }
                }
                Some(QueryFilter::Vector(vector_query)) => {
                    columns.push(vector_query.column.clone());
                }
                None => {}
            }
        }

        Ok(columns)
    }

    pub async fn refine_filter(
        &self,
        input: Arc<dyn ExecutionPlan>,
        scanner: &Scanner,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut plan = input;

        if self.refine_query_filter {
            match &self.query_filter {
                Some(QueryFilter::Fts(fts_query)) => {
                    plan = scanner.flat_fts_filter(plan, fts_query).await?;
                }
                Some(QueryFilter::Vector(vector_query)) => {
                    plan = scanner.flat_knn(plan, vector_query)?;
                }
                None => {}
            }
        }

        if let Some(refine_expr) = &self.expr_filter_plan.refine_expr {
            // We create a new planner specific to the node's schema, since
            // physical expressions reference column by index rather than by name.
            plan = Arc::new(LanceFilterExec::try_new(refine_expr.clone(), plan)?);
        }

        Ok(plan)
    }
}

#[derive(Debug, Clone, Default)]
pub struct LanceFilter {
    query_filter: Option<QueryFilter>,
    expr_filter: Option<ExprFilter>,
}

impl LanceFilter {
    pub fn is_none(&self) -> bool {
        self.query_filter.is_none() && self.expr_filter.is_none()
    }
}

/// Query filter for filtering rows
#[derive(Debug, Clone)]
pub enum QueryFilter {
    Fts(FullTextSearchQuery),
    Vector(Query),
}

/// Expr filter for filtering rows
#[derive(Debug, Clone)]
pub enum ExprFilter {
    /// The filter is an SQL string
    Sql(String),
    /// The filter is a Substrait expression
    Substrait(Vec<u8>),
    /// The filter is a Datafusion expression
    Datafusion(Expr),
}

impl ExprFilter {
    /// Converts the filter to a Datafusion expression
    ///
    /// The schema for this conversion should be the full schema available to
    /// the filter (`full_schema`).  However, due to a limitation in the way
    /// we do Substrait conversion today we can only do Substrait conversion with
    /// the dataset schema (`dataset_schema`).  This means that Substrait will
    /// not be able to access columns that are not in the dataset schema (e.g.
    /// _rowid, _rowaddr, etc.)
    #[allow(unused)]
    #[instrument(level = "trace", name = "filter_to_df", skip_all)]
    pub fn to_datafusion(&self, dataset_schema: &Schema, full_schema: &Schema) -> Result<Expr> {
        match self {
            Self::Sql(sql) => {
                let schema = Arc::new(ArrowSchema::from(full_schema));
                let planner = Planner::new(schema.clone());
                let filter = planner.parse_filter(sql)?;

                let df_schema = DFSchema::try_from(schema)?;
                let ret_field = filter.to_field(&df_schema)?.1;
                let ret_type = ret_field.data_type();
                if ret_type != &DataType::Boolean {
                    return Err(Error::invalid_input_source(
                        format!("The filter {} does not return a boolean", filter).into(),
                    ));
                }

                let optimized = planner.optimize_expr(filter).map_err(|e| {
                    Error::invalid_input(format!("Error optimizing sql filter: {sql} ({e})"))
                })?;
                Ok(optimized)
            }
            #[cfg(feature = "substrait")]
            Self::Substrait(expr) => {
                use lance_datafusion::exec::{LanceExecutionOptions, get_session_context};

                let ctx = get_session_context(&LanceExecutionOptions::default());
                let state = ctx.state();
                let schema = Arc::new(ArrowSchema::from(dataset_schema));
                let expr = parse_substrait(expr, schema.clone(), &ctx.state())
                    .now_or_never()
                    .expect("could not parse the Substrait filter in a synchronous fashion")?;
                let planner = Planner::new(schema);
                planner.optimize_expr(expr.clone()).map_err(|e| {
                    Error::invalid_input(format!(
                        "Error optimizing substrait filter: {expr:?} ({e})"
                    ))
                })
            }
            #[cfg(not(feature = "substrait"))]
            Self::Substrait(_) => Err(Error::not_supported_source(
                "Substrait filter is not supported in this build".into(),
            )),
            Self::Datafusion(expr) => Ok(expr.clone()),
        }
    }
}

/// Aggregate expression from Substrait or DataFusion.
#[derive(Debug, Clone)]
pub enum AggregateExpr {
    #[cfg(feature = "substrait")]
    Substrait(Vec<u8>),
    Datafusion {
        group_by: Vec<Expr>,
        aggregates: Vec<Expr>,
    },
}

impl AggregateExpr {
    /// Create a new builder for aggregate expressions.
    ///
    /// # Example
    /// ```ignore
    /// let agg = AggregateExpr::builder()
    ///     .group_by("category")
    ///     .count_star().alias("total_count")
    ///     .sum("amount").alias("total_amount")
    ///     .avg("price")
    ///     .build();
    /// scanner.aggregate(agg);
    /// ```
    pub fn builder() -> AggregateExprBuilder<false> {
        AggregateExprBuilder::new()
    }

    /// Create from Substrait Plan bytes.
    #[cfg(feature = "substrait")]
    pub fn substrait(bytes: impl Into<Vec<u8>>) -> Self {
        Self::Substrait(bytes.into())
    }

    /// Create from DataFusion expressions.
    /// Use `.alias()` on expressions to set output column names.
    pub fn datafusion(group_by: Vec<Expr>, aggregates: Vec<Expr>) -> Self {
        Self::Datafusion {
            group_by,
            aggregates,
        }
    }

    /// Parse into a unified Aggregate structure.
    ///
    /// For Substrait, this parses the bytes into DataFusion expressions.
    /// For DataFusion, this just wraps the expressions.
    ///
    /// The schema is used to resolve field references in Substrait expressions.
    fn parse(self, #[allow(unused_variables)] schema: Arc<ArrowSchema>) -> Result<Aggregate> {
        match self {
            #[cfg(feature = "substrait")]
            Self::Substrait(bytes) => {
                use lance_datafusion::exec::{LanceExecutionOptions, get_session_context};
                use lance_datafusion::substrait::parse_substrait_aggregate;

                let ctx = get_session_context(&LanceExecutionOptions::default());
                parse_substrait_aggregate(&bytes, schema, &ctx.state())
                    .now_or_never()
                    .expect("could not parse the Substrait aggregate in a synchronous fashion")
            }
            Self::Datafusion {
                group_by,
                aggregates,
            } => Ok(Aggregate::new(group_by, aggregates)),
        }
    }
}

/// Builder for creating aggregate expressions without using DataFusion or Substrait directly.
///
/// The const generic `HAS_PENDING` tracks whether there's a pending aggregate that can be aliased.
/// When `HAS_PENDING` is `true`, the last item in `aggregates` is the pending aggregate.
#[derive(Debug, Clone)]
pub struct AggregateExprBuilder<const HAS_PENDING: bool> {
    group_by: Vec<Expr>,
    aggregates: Vec<Expr>,
}

impl Default for AggregateExprBuilder<false> {
    fn default() -> Self {
        Self {
            group_by: Vec::new(),
            aggregates: Vec::new(),
        }
    }
}

impl AggregateExprBuilder<false> {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the aggregate expression.
    pub fn build(self) -> AggregateExpr {
        AggregateExpr::Datafusion {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }
}

impl<const HAS_PENDING: bool> AggregateExprBuilder<HAS_PENDING> {
    /// Add a column to group by.
    ///
    /// Multiple invocations will add to the list (not replace it).
    /// E.g. `.group_by("x").group_by("y")` will group by both `x` and `y`.
    pub fn group_by(mut self, column: impl Into<String>) -> AggregateExprBuilder<false> {
        self.group_by.push(col(column.into()));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add multiple columns to group by.
    ///
    /// Multiple invocations will add to the list (not replace it).
    /// E.g. `.group_by("x").group_by_columns(["y", "z"])` will group by `x`, `y`, and `z`.
    pub fn group_by_columns(
        mut self,
        columns: impl IntoIterator<Item = impl Into<String>>,
    ) -> AggregateExprBuilder<false> {
        for column in columns {
            self.group_by.push(col(column.into()));
        }
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add COUNT(*) aggregate that counts all rows.
    pub fn count_star(mut self) -> AggregateExprBuilder<true> {
        self.aggregates
            .push(functions_aggregate::count::count(lit(1)));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add COUNT(column) aggregate.
    ///
    /// Unlike `count_star`, this will only count the number of rows where `column`
    /// is not NULL.
    pub fn count(mut self, column: impl Into<String>) -> AggregateExprBuilder<true> {
        self.aggregates
            .push(functions_aggregate::count::count(col(column.into())));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add SUM(column) aggregate.
    pub fn sum(mut self, column: impl Into<String>) -> AggregateExprBuilder<true> {
        self.aggregates
            .push(functions_aggregate::sum::sum(col(column.into())));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add AVG(column) aggregate.
    pub fn avg(mut self, column: impl Into<String>) -> AggregateExprBuilder<true> {
        self.aggregates
            .push(functions_aggregate::average::avg(col(column.into())));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add MIN(column) aggregate.
    pub fn min(mut self, column: impl Into<String>) -> AggregateExprBuilder<true> {
        self.aggregates
            .push(functions_aggregate::min_max::min(col(column.into())));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add MAX(column) aggregate.
    pub fn max(mut self, column: impl Into<String>) -> AggregateExprBuilder<true> {
        self.aggregates
            .push(functions_aggregate::min_max::max(col(column.into())));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }
}

impl AggregateExprBuilder<true> {
    /// Set an alias for the pending aggregate (the last added aggregate).
    pub fn alias(mut self, name: impl Into<String>) -> AggregateExprBuilder<false> {
        let pending = self.aggregates.pop().expect("pending aggregate must exist");
        self.aggregates.push(pending.alias(name.into()));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Build the aggregate expression.
    pub fn build(self) -> AggregateExpr {
        AggregateExpr::Datafusion {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }
}

/// Dataset Scanner
///
/// ```rust,ignore
/// let dataset = Dataset::open(uri).await.unwrap();
/// let stream = dataset.scan()
///     .project(&["col", "col2.subfield"]).unwrap()
///     .limit(10)
///     .into_stream();
/// stream
///   .map(|batch| batch.num_rows())
///   .buffered(16)
///   .sum()
/// ```
#[derive(Clone)]
pub struct Scanner {
    dataset: Arc<Dataset>,

    /// The projection plan for the scanner
    ///
    /// This includes
    /// - The physical projection that must be read from the dataset
    /// - Dynamic expressions that are evaluated after the physical projection
    /// - The names of the output columns
    projection_plan: ProjectionPlan,
    blob_handling: BlobHandling,

    /// If true then the filter will be applied before an index scan
    prefilter: bool,

    /// Materialization style controls when columns are fetched
    materialization_style: MaterializationStyle,

    /// Filter.
    filter: LanceFilter,

    /// Optional full text search query
    full_text_query: Option<FullTextSearchQuery>,

    /// The batch size controls the maximum size of rows to return for each read.
    batch_size: Option<usize>,

    /// If set, the scanner will produce batches whose total size in bytes
    /// is approximately this value, overriding the row-based `batch_size`.
    batch_size_bytes: Option<u64>,

    /// Number of batches to prefetch
    batch_readahead: usize,

    /// Number of fragments to read concurrently
    fragment_readahead: Option<usize>,

    /// Number of bytes to allow to queue up in the I/O buffer
    io_buffer_size: Option<u64>,

    limit: Option<i64>,
    offset: Option<i64>,

    /// If Some then results will be ordered by the provided ordering
    ///
    /// If there are multiple columns the results will first be ordered
    /// by the first column.  Then, any values whose first column is equal
    /// will be sorted by the next column, and so on.
    ///
    /// If this is Some then the value of `ordered` is ignored.  The scan
    /// will always be unordered since we are just going to reorder it anyways.
    ordering: Option<Vec<ColumnOrdering>>,

    nearest: Option<Query>,
    nearest_query_count: usize,
    /// True when the query shape represents a batch of single-vector queries
    /// (list-like query on a fixed-size vector column, or multiple concatenated vectors).
    is_batch_nearest: bool,

    /// If false, do not use any scalar indices for the scan
    ///
    /// This can be used to pick a more efficient plan for certain queries where
    /// scalar indices do not work well (though we should also improve our planning
    /// to handle this better in the future as well)
    use_scalar_index: bool,

    /// Whether to use statistics to optimize the scan (default: true)
    ///
    /// This is used for debugging or benchmarking purposes.
    use_stats: bool,

    /// Whether to scan in deterministic order (default: true)
    ///
    /// This field is ignored if `ordering` is defined
    ordered: bool,

    /// If set, this scanner serves only these fragments.
    fragments: Option<Vec<Fragment>>,

    /// If set, this scanner will only search the specified vector index segments.
    index_segments: Option<Vec<Uuid>>,

    /// Only search the data being indexed (weak consistency search).
    ///
    /// Default value is false.
    ///
    /// This is essentially a weak consistency search. Users can run index or optimize index
    /// to make the index catch up with the latest data.
    fast_search: bool,

    /// If true, the scanner will emit deleted rows
    include_deleted_rows: bool,

    /// If set, this callback will be called after the scan with summary statistics
    scan_stats_callback: Option<ExecutionStatsCallback>,

    /// Whether the result returned by the scanner must be of the size of the batch_size.
    /// By default, it is false.
    /// Mainly, if the result is returned strictly according to the batch_size,
    /// batching and waiting are required, and the performance will decrease.
    strict_batch_size: bool,

    /// File reader options to use when reading data files.
    file_reader_options: Option<FileReaderOptions>,

    aggregate: Option<Aggregate>,

    /// Which version of the relational algebra to use when generating the physical plan
    relational_algebra_version: u32,

    /// Target degree of parallelism for the physical optimizer.
    ///
    /// This is passed as `ConfigOptions::execution::target_partitions` to the
    /// physical optimizer (e.g. `EnforceDistribution`), which uses it to decide
    /// how many parallel partitions to target when inserting exchange nodes.
    ///
    /// Defaults to `get_num_compute_intensive_cpus()`.
    target_parallelism: Option<usize>,

    // Legacy fields to help migrate some old projection behavior to new behavior
    //
    // There are two behaviors we are moving away from:
    //
    // First, the old behavior used methods like with_row_id and with_row_addr to add
    // "system" columns.  The new behavior is to specify them in the projection like any
    // other column.  The only difference between a system column and a regular column is
    // that system columns are not returned in the schema and are not returned by default
    // (i.e. "SELECT *")
    //
    // Second, the old behavior would _always_ add the _score or _distance columns to the
    // output and there was no way for the user to opt out.  The new behavior treats the
    // _score and _distance as regular output columns of the "search table function".  If
    // the user does not specify a projection (i.e. "SELECT *") then we will add the _score
    // and _distance columns to the end.  If the user does specify a projection then they
    // must request those columns for them to show up.
    //
    // --------------------------------------------------------------------------
    /// Whether the user wants the row id on top of the projection, will always come last
    /// except possibly before _rowaddr
    legacy_with_row_id: bool,
    /// Whether the user wants the row address on top of the projection, will always come last
    legacy_with_row_addr: bool,
    /// Whether the user explicitly requested a projection.  When true, scoring columns
    /// (_distance, _score) are only included if explicitly listed in the projection.
    /// When false (SELECT *), they are automatically appended.
    explicit_projection: bool,
}

/// Represents a user-requested take operation
#[derive(Debug, Clone)]
pub enum TakeOperation {
    /// Take rows by row id
    RowIds(Vec<u64>),
    /// Take rows by row address
    RowAddrs(Vec<u64>),
    /// Take rows by row offset
    ///
    /// The row offset is the offset of the row in the dataset.  This can
    /// be converted to row addresses using the fragment sizes.
    RowOffsets(Vec<u64>),
}

impl TakeOperation {
    fn extract_u64_list(list: &[Expr]) -> Option<Vec<u64>> {
        let mut u64s = Vec::with_capacity(list.len());
        for expr in list {
            if let Expr::Literal(lit, _) = expr {
                if let Some(ScalarValue::UInt64(Some(val))) =
                    safe_coerce_scalar(lit, &DataType::UInt64)
                {
                    u64s.push(val);
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
        Some(u64s)
    }

    fn merge(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::RowIds(mut left), Self::RowIds(right)) => {
                left.extend(right);
                Some(Self::RowIds(left))
            }
            (Self::RowAddrs(mut left), Self::RowAddrs(right)) => {
                left.extend(right);
                Some(Self::RowAddrs(left))
            }
            (Self::RowOffsets(mut left), Self::RowOffsets(right)) => {
                left.extend(right);
                Some(Self::RowOffsets(left))
            }
            _ => None,
        }
    }

    /// Attempts to create a take operation from an expression.  This will succeed if the expression
    /// has one of the following forms:
    ///  - `_rowid = 10`
    ///  - `_rowid = 10 OR _rowid = 20 OR _rowid = 30`
    ///  - `_rowid IN (10, 20, 30)`
    ///  - `_rowaddr = 10`
    ///  - `_rowaddr = 10 OR _rowaddr = 20 OR _rowaddr = 30`
    ///  - `_rowaddr IN (10, 20, 30)`
    ///  - `_rowoffset = 10`
    ///  - `_rowoffset = 10 OR _rowoffset = 20 OR _rowoffset = 30`
    ///  - `_rowoffset IN (10, 20, 30)`
    ///
    /// The _rowid / _rowaddr / _rowoffset determine if we are taking by row id, address, or offset.
    ///
    /// If a take expression is combined with some other filter via an AND then the remainder will be
    /// returned as well.  For example, `_rowid = 10` will return (take_op, None) and
    /// `_rowid = 10 AND x > 70` will return (take_op, Some(x > 70)).
    fn try_from_expr(expr: &Expr) -> Option<(Self, Option<Expr>)> {
        if let Expr::BinaryExpr(binary) = expr {
            match binary.op {
                datafusion_expr::Operator::And => {
                    let left_take = Self::try_from_expr(&binary.left);
                    let right_take = Self::try_from_expr(&binary.right);
                    match (left_take, right_take) {
                        (Some(_), Some(_)) => {
                            // This is something like...
                            //
                            // _rowid = 10 AND _rowid = 20
                            //
                            // ...which is kind of nonsensical.  Better to just return None.
                            return None;
                        }
                        (Some((left_op, left_rem)), None) => {
                            let remainder = match left_rem {
                                // If there is a remainder on the left side we combine it.  This _should_
                                // be something like converting (_rowid = 10 AND x > 70) AND y > 80
                                // to (_rowid = 10) AND (x > 70 AND y > 80) which should be valid
                                Some(expr) => Expr::and(expr, binary.right.as_ref().clone()),
                                None => binary.right.as_ref().clone(),
                            };
                            return Some((left_op, Some(remainder)));
                        }
                        (None, Some((right_op, right_rem))) => {
                            let remainder = match right_rem {
                                Some(expr) => Expr::and(expr, binary.left.as_ref().clone()),
                                None => binary.left.as_ref().clone(),
                            };
                            return Some((right_op, Some(remainder)));
                        }
                        (None, None) => {
                            return None;
                        }
                    }
                }
                datafusion_expr::Operator::Eq => {
                    // Check for _rowid = literal
                    if let (Expr::Column(col), Expr::Literal(lit, _)) =
                        (binary.left.as_ref(), binary.right.as_ref())
                        && let Some(ScalarValue::UInt64(Some(val))) =
                            safe_coerce_scalar(lit, &DataType::UInt64)
                    {
                        if col.name == ROW_ID {
                            return Some((Self::RowIds(vec![val]), None));
                        } else if col.name == ROW_ADDR {
                            return Some((Self::RowAddrs(vec![val]), None));
                        } else if col.name == ROW_OFFSET {
                            return Some((Self::RowOffsets(vec![val]), None));
                        }
                    }
                }
                datafusion_expr::Operator::Or => {
                    let left_take = Self::try_from_expr(&binary.left);
                    let right_take = Self::try_from_expr(&binary.right);
                    if let (Some(left), Some(right)) = (left_take, right_take) {
                        if left.1.is_some() || right.1.is_some() {
                            // This would be something like...
                            //
                            // (_rowid = 10 AND x > 70) OR _rowid = 20
                            //
                            // I don't think it's correct to convert this into a take operation
                            // which would give us (_rowid = 10 OR _rowid = 20) AND x > 70
                            return None;
                        }
                        return left.0.merge(right.0).map(|op| (op, None));
                    }
                }
                _ => {}
            }
        } else if let Expr::InList(in_expr) = expr
            && let Expr::Column(col) = in_expr.expr.as_ref()
            && let Some(u64s) = Self::extract_u64_list(&in_expr.list)
        {
            if col.name == ROW_ID {
                return Some((Self::RowIds(u64s), None));
            } else if col.name == ROW_ADDR {
                return Some((Self::RowAddrs(u64s), None));
            } else if col.name == ROW_OFFSET {
                return Some((Self::RowOffsets(u64s), None));
            }
        }
        None
    }
}

impl Scanner {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        let projection_plan = ProjectionPlan::full(dataset.clone()).unwrap();
        let file_reader_options = dataset.file_reader_options.clone();
        let mut scanner = Self {
            dataset,
            projection_plan,
            blob_handling: BlobHandling::default(),
            prefilter: false,
            materialization_style: MaterializationStyle::Heuristic,
            filter: LanceFilter::default(),
            full_text_query: None,
            batch_size: None,
            batch_size_bytes: None,
            batch_readahead: get_num_compute_intensive_cpus(),
            fragment_readahead: None,
            io_buffer_size: None,
            limit: None,
            offset: None,
            ordering: None,
            nearest: None,
            nearest_query_count: 1,
            is_batch_nearest: false,
            use_stats: true,
            ordered: true,
            fragments: None,
            index_segments: None,
            fast_search: false,
            use_scalar_index: true,
            include_deleted_rows: false,
            scan_stats_callback: None,
            strict_batch_size: false,
            file_reader_options,
            aggregate: None,
            legacy_with_row_addr: false,
            legacy_with_row_id: false,
            explicit_projection: false,
            relational_algebra_version: LANCE_RELATIONAL_ALGEBRA_VERSION,
            target_parallelism: None,
        };
        scanner.apply_blob_handling();
        scanner
    }

    fn apply_blob_handling(&mut self) {
        let projection = self
            .projection_plan
            .physical_projection
            .clone()
            .with_blob_handling(self.blob_handling.clone());
        self.projection_plan.physical_projection = projection;
    }

    pub fn blob_handling(&mut self, blob_handling: BlobHandling) -> &mut Self {
        self.blob_handling = blob_handling;
        self.apply_blob_handling();
        self
    }

    pub fn from_fragment(dataset: Arc<Dataset>, fragment: Fragment) -> Self {
        Self {
            fragments: Some(vec![fragment]),
            ..Self::new(dataset)
        }
    }

    /// Set which fragments should be scanned.
    ///
    /// If scan_in_order is set to true, the fragments will be scanned in the order of the vector.
    pub fn with_fragments(&mut self, fragments: Vec<Fragment>) -> &mut Self {
        self.fragments = Some(fragments);
        self
    }

    /// Restrict vector index search to the specified index segments.
    ///
    /// This setting is only supported for vector search.
    ///
    /// If [`Self::with_fragments`] is also set then rows from those fragments that are not covered
    /// by the selected index segments will still be searched with flat KNN. Otherwise, unindexed
    /// fragments outside the selected index segments are not searched.
    pub fn with_index_segments(&mut self, segments: Vec<Uuid>) -> Result<&mut Self> {
        if segments.is_empty() {
            return Err(Error::invalid_input(
                "with_index_segments does not accept an empty segment list".to_string(),
            ));
        }
        self.index_segments = Some(segments);
        Ok(self)
    }

    fn get_batch_size(&self) -> usize {
        // Default batch size to be large enough so that a i32 column can be
        // read in a single range request. For the object store default of
        // 64KB, this is 16K rows. For local file systems, the default block size
        // is just 4K, which would mean only 1K rows, which might be a little small.
        // So we use a default minimum of 8K rows.
        get_default_batch_size().unwrap_or_else(|| {
            self.batch_size.unwrap_or_else(|| {
                std::cmp::max(
                    self.dataset.object_store.as_ref().block_size() / 4,
                    BATCH_SIZE_FALLBACK,
                )
            })
        })
    }

    fn ensure_not_fragment_scan(&self) -> Result<()> {
        if self.is_fragment_scan() {
            Err(Error::not_supported(
                "This operation is not supported for fragment scan".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    fn is_fragment_scan(&self) -> bool {
        self.fragments.is_some()
    }

    /// Empty Projection (useful for count queries)
    ///
    /// The row_address will be scanned (no I/O required) but not included in the output
    pub fn empty_project(&mut self) -> Result<&mut Self> {
        self.project(&[] as &[&str])
    }

    /// Projection.
    ///
    /// Only select the specified columns. If not specified, all columns will be scanned.
    pub fn project<T: AsRef<str>>(&mut self, columns: &[T]) -> Result<&mut Self> {
        let transformed_columns: Vec<(&str, String)> = columns
            .iter()
            .map(|c| (c.as_ref(), escape_field_path_for_project(c.as_ref())))
            .collect();

        self.project_with_transform(&transformed_columns)
    }

    /// Projection with transform
    ///
    /// Only select the specified columns with the given transform.
    pub fn project_with_transform(
        &mut self,
        columns: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Result<&mut Self> {
        self.explicit_projection = true;
        self.projection_plan = ProjectionPlan::from_expressions(self.dataset.clone(), columns)?;
        if self.legacy_with_row_id {
            self.projection_plan.include_row_id();
        }
        if self.legacy_with_row_addr {
            self.projection_plan.include_row_addr();
        }
        self.apply_blob_handling();
        Ok(self)
    }

    /// Should the filter run before the vector index is applied
    ///
    /// If true then the filter will be applied before the vector index.  This
    /// means the results will be accurate but the overall query may be more expensive.
    ///
    /// If false then the filter will be applied to the nearest results.  This means
    /// you may get back fewer results than you ask for (or none at all) if the closest
    /// results do not match the filter.
    pub fn prefilter(&mut self, should_prefilter: bool) -> &mut Self {
        self.prefilter = should_prefilter;
        self
    }

    /// Set the callback to be called after the scan with summary statistics
    pub fn scan_stats_callback(&mut self, callback: ExecutionStatsCallback) -> &mut Self {
        self.scan_stats_callback = Some(callback);
        self
    }

    /// Set the materialization style for the scan
    ///
    /// This controls when columns are fetched from storage.  The default should work
    /// well for most cases.
    ///
    /// If you know (in advance) a query will return relatively few results (less than
    /// 0.1% of the rows) then you may want to experiment with applying late materialization
    /// to more (or all) columns.
    ///
    /// If you know a query is going to return many rows then you may want to experiment
    /// with applying early materialization to more (or all) columns.
    pub fn materialization_style(&mut self, style: MaterializationStyle) -> &mut Self {
        self.materialization_style = style;
        self
    }

    /// Apply filters
    ///
    /// The filters can be presented as the string, as in WHERE clause in SQL.
    ///
    /// ```rust,ignore
    /// let dataset = Dataset::open(uri).await.unwrap();
    /// let stream = dataset.scan()
    ///     .project(&["col", "col2.subfield"]).unwrap()
    ///     .filter("a > 10 AND b < 200").unwrap()
    ///     .limit(10)
    ///     .into_stream();
    /// ```
    ///
    /// Once the filter is applied, Lance will create an optimized I/O plan for filtering.
    ///
    pub fn filter(&mut self, filter: &str) -> Result<&mut Self> {
        self.filter.expr_filter = Some(ExprFilter::Sql(filter.to_string()));
        Ok(self)
    }

    /// Apply fts/vector query as filter.
    ///
    /// * Vector query filter can only be applied to full text search.
    /// * Fts query filter can only be applied to vector search.
    /// * Query filter couldn't be applied to normal query.
    ///
    /// ```rust,ignore
    /// let dataset = Dataset::open(uri).await.unwrap();
    /// let query_vector = Float32Array::from(vec![300f32, 300f32, 300f32, 300f32]);
    /// let stream = dataset.scan()
    ///     .nearest("vector", &query_vector, 5)
    ///     .project(&["col", "col2.subfield"]).unwrap()
    ///     .query_filter(QueryFilter::Fts(FullTextSearchQuery::new(
    ///       "hello".to_string(),
    ///     ))).unwrap()
    ///     .limit(10)
    ///     .into_stream();
    /// ```
    pub fn filter_query(&mut self, filter: QueryFilter) -> Result<&mut Self> {
        self.filter.query_filter = Some(filter);
        Ok(self)
    }

    /// Filter by full text search
    /// The column must be a string column.
    /// The query is a string to search for.
    /// The search is case-insensitive, BM25 scoring is used.
    ///
    /// ```rust,ignore
    /// let dataset = Dataset::open(uri).await.unwrap();
    /// let stream = dataset.scan()
    ///    .project(&["col", "col2.subfield"]).unwrap()
    ///    .full_text_search("col", "query").unwrap()
    ///    .limit(10)
    ///    .into_stream();
    /// ```
    pub fn full_text_search(&mut self, query: FullTextSearchQuery) -> Result<&mut Self> {
        let fields = query.columns();
        if !fields.is_empty() {
            for field in fields.iter() {
                if self.dataset.schema().field(field).is_none() {
                    return Err(Error::invalid_input(format!("Column {} not found", field)));
                }
            }
        }

        self.full_text_query = Some(query);
        Ok(self)
    }

    /// Set a filter using a Substrait ExtendedExpression message
    ///
    /// The message must contain exactly one expression and that expression
    /// must be a scalar expression whose return type is boolean.
    pub fn filter_substrait(&mut self, filter: &[u8]) -> Result<&mut Self> {
        self.filter.expr_filter = Some(ExprFilter::Substrait(filter.to_vec()));
        Ok(self)
    }

    pub fn filter_expr(&mut self, filter: Expr) -> &mut Self {
        self.filter.expr_filter = Some(ExprFilter::Datafusion(filter));
        self
    }

    /// Set aggregation.
    ///
    /// The aggregate expression is parsed immediately using the dataset schema.
    /// For Substrait aggregates, this converts them to DataFusion expressions.
    pub fn aggregate(&mut self, aggregate: AggregateExpr) -> Result<&mut Self> {
        let schema: Arc<ArrowSchema> = Arc::new(self.dataset.schema().into());
        let parsed = aggregate.parse(schema)?;
        self.aggregate = Some(parsed);
        Ok(self)
    }

    /// Set the maximum number of rows per batch.
    ///
    /// Note: this can be overridden by [`Self::batch_size_bytes`] or by a dataset-level
    /// `batch_size_bytes` set via [`ReadParams::file_reader_options`](crate::dataset::ReadParams::file_reader_options).  When a byte-based
    /// batch size is active, the row-based batch size is used only as an initial estimate.
    pub fn batch_size(&mut self, batch_size: usize) -> &mut Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set the target batch size in bytes.
    ///
    /// When set, the scanner will produce batches whose total size in bytes
    /// is approximately this value, overriding the row-based `batch_size`.
    ///
    /// This can also be configured at the dataset level via
    /// [`ReadParams::file_reader_options`](crate::dataset::ReadParams::file_reader_options).  A scanner-level setting takes
    /// precedence over the dataset-level default.
    pub fn batch_size_bytes(&mut self, batch_size_bytes: u64) -> &mut Self {
        self.batch_size_bytes = Some(batch_size_bytes);
        self
    }

    /// Include deleted rows
    ///
    /// These are rows that have been deleted from the dataset but are still present in the
    /// underlying storage.  These rows will have the `_rowid` column set to NULL.  The other columns
    /// (include _rowaddr) will be set to their deleted values.
    ///
    /// This can be useful for generating aligned fragments or debugging
    ///
    /// Note: when entire fragments are deleted, the scanner will not emit any rows for that fragment
    /// since the fragment is no longer present in the dataset.
    pub fn include_deleted_rows(&mut self) -> &mut Self {
        self.include_deleted_rows = true;
        self
    }

    /// Set the I/O buffer size
    ///
    /// This is the amount of RAM that will be reserved for holding I/O received from
    /// storage before it is processed.  This is used to control the amount of memory
    /// used by the scanner.  If the buffer is full then the scanner will block until
    /// the buffer is processed.
    ///
    /// Generally this should scale with the number of concurrent I/O threads.  The
    /// default is 2GiB which comfortably provides enough space for somewhere between
    /// 32 and 256 concurrent I/O threads.
    ///
    /// This value is not a hard cap on the amount of RAM the scanner will use.  Some
    /// space is used for the compute (which can be controlled by the batch size) and
    /// Lance does not keep track of memory after it is returned to the user.
    ///
    /// Currently, if there is a single batch of data which is larger than the io buffer
    /// size then the scanner will deadlock.  This is a known issue and will be fixed in
    /// a future release.
    pub fn io_buffer_size(&mut self, size: u64) -> &mut Self {
        self.io_buffer_size = Some(size);
        self
    }

    /// Set the prefetch size.
    /// Ignored in v2 and newer format
    pub fn batch_readahead(&mut self, nbatches: usize) -> &mut Self {
        self.batch_readahead = nbatches;
        self
    }

    /// Set the fragment readahead.
    ///
    /// This is only used if ``scan_in_order`` is set to false.
    pub fn fragment_readahead(&mut self, nfragments: usize) -> &mut Self {
        self.fragment_readahead = Some(nfragments);
        self
    }

    /// Set the target number of partitions for the physical optimizer.
    ///
    /// Overrides the default (`get_num_compute_intensive_cpus()`). Used by
    /// `EnforceDistribution` and similar rules to decide how many parallel
    /// partitions to use. Set to 1 in tests that assert specific plan shapes.
    pub fn target_parallelism(&mut self, n: usize) -> &mut Self {
        self.target_parallelism = Some(n);
        self
    }

    /// Set whether to read data in order (default: true)
    ///
    /// A scan will always read from the disk concurrently.  If this property
    /// is true then a ready batch (a batch that has been read from disk) will
    /// only be returned if it is the next batch in the sequence.  Otherwise,
    /// the batch will be held until the stream catches up.  This means the
    /// sequence is returned in order but there may be slightly less parallelism.
    ///
    /// If this is false, then batches will be returned as soon as they are
    /// available, potentially increasing throughput slightly
    ///
    /// If an ordering is defined (using [Self::order_by]) then the scan will
    /// always scan in parallel and any value set here will be ignored.
    pub fn scan_in_order(&mut self, ordered: bool) -> &mut Self {
        self.ordered = ordered;
        self
    }

    /// Set whether to use scalar index.
    ///
    /// By default, scalar indices will be used to optimize a query if available.
    /// However, in some corner cases, scalar indices may not be the best choice.
    /// This option allows users to disable scalar indices for a query.
    pub fn use_scalar_index(&mut self, use_scalar_index: bool) -> &mut Self {
        self.use_scalar_index = use_scalar_index;
        self
    }

    /// Set whether to use strict batch size.
    ///
    /// If this is true then output batches (except the last batch) will have exactly `batch_size` rows.
    /// By default, this is False and output batches are allowed to have fewer than `batch_size` rows
    /// Setting this to True will require us to merge batches, incurring a data copy, for a minor performance
    /// penalty.
    pub fn strict_batch_size(&mut self, strict_batch_size: bool) -> &mut Self {
        self.strict_batch_size = strict_batch_size;
        self
    }

    /// Set limit and offset.
    ///
    /// If offset is set, the first offset rows will be skipped. If limit is set,
    /// only the provided number of rows will be returned. These can be set
    /// independently. For example, setting offset to 10 and limit to None will
    /// skip the first 10 rows and return the rest of the rows in the dataset.
    pub fn limit(&mut self, limit: Option<i64>, offset: Option<i64>) -> Result<&mut Self> {
        if limit.unwrap_or_default() < 0 {
            return Err(Error::invalid_input(
                "Limit must be non-negative".to_string(),
            ));
        }
        if let Some(off) = offset
            && off < 0
        {
            return Err(Error::invalid_input(
                "Offset must be non-negative".to_string(),
            ));
        }
        self.limit = limit;
        self.offset = offset;
        Ok(self)
    }

    /// Returns true when `q` is a batch of single-vector queries.
    ///
    /// List-like queries against a [`DataType::List`] vector column are treated as one
    /// multivector query. The same list-like query against a fixed-size vector column is
    /// treated as a batch of single-vector queries.
    fn is_batch_nearest_query(vector_type: &DataType, query_type: &DataType) -> bool {
        matches!(vector_type, DataType::FixedSizeList(_, _))
            && matches!(
                query_type,
                DataType::List(_) | DataType::FixedSizeList(_, _)
            )
    }

    /// Find k-nearest neighbor within the vector column.
    /// the query can be a Float16Array, Float32Array, Float64Array, UInt8Array,
    /// or a ListArray/FixedSizeListArray of the above types.
    pub fn nearest(&mut self, column: &str, q: &dyn Array, k: usize) -> Result<&mut Self> {
        if !self.prefilter {
            // We can allow fragment scan if the input to nearest is a prefilter.
            // The fragment scan will be performed by the prefilter.
            self.ensure_not_fragment_scan()?;
        }

        if k == 0 {
            return Err(Error::invalid_input("k must be positive".to_string()));
        }
        if q.is_empty() {
            return Err(Error::invalid_input(
                "Query vector must have non-zero length".to_string(),
            ));
        }
        // make sure the field exists
        let (vector_type, element_type) = get_vector_type(self.dataset.schema(), column)?;
        let dim = get_vector_dim(self.dataset.schema(), column)?;
        let query_type = q.data_type().clone();

        let (q, query_count) = match &query_type {
            DataType::List(_) | DataType::FixedSizeList(_, _) => {
                if let Some(list_array) = q.as_list_opt::<i32>() {
                    for i in 0..list_array.len() {
                        let vec = list_array.value(i);
                        if vec.len() != dim {
                            return Err(Error::invalid_input(format!(
                                "query dim({}) doesn't match the column {} vector dim({})",
                                vec.len(),
                                column,
                                dim,
                            )));
                        }
                    }
                    // A list-like query against a multivector column is one multivector query.
                    // The same list-like query against a fixed-size vector column is a batch
                    // of single-vector queries.
                    let query_count = if matches!(vector_type, DataType::List(_)) {
                        1
                    } else {
                        list_array.len()
                    };
                    (list_array.values().clone(), query_count)
                } else {
                    let fsl = q.as_fixed_size_list();
                    if fsl.value_length() as usize != dim {
                        return Err(Error::invalid_input(format!(
                            "query dim({}) doesn't match the column {} vector dim({})",
                            fsl.value_length(),
                            column,
                            dim,
                        )));
                    }
                    // A list-like query against a multivector column is one multivector query.
                    // The same list-like query against a fixed-size vector column is a batch
                    // of single-vector queries.
                    let query_count = if matches!(vector_type, DataType::List(_)) {
                        1
                    } else {
                        fsl.len()
                    };
                    (fsl.values().clone(), query_count)
                }
            }
            _ => {
                if q.len() != dim {
                    return Err(Error::invalid_input(format!(
                        "query dim({}) doesn't match the column {} vector dim({})",
                        q.len(),
                        column,
                        dim,
                    )));
                }
                (q.slice(0, q.len()), 1)
            }
        };

        let is_batch_nearest = Self::is_batch_nearest_query(&vector_type, &query_type);
        if is_batch_nearest && self.dataset.schema().field(QUERY_INDEX_COL).is_some() {
            return Err(Error::invalid_input(format!(
                "batch nearest neighbor search cannot be used on datasets with column '{QUERY_INDEX_COL}'"
            )));
        }

        let key = match &element_type {
            dt if dt == q.data_type() => q,
            dt if dt.is_floating() => coerce_float_vector(
                q.as_any().downcast_ref::<Float32Array>().unwrap(),
                FloatType::try_from(dt)?,
            )?,
            _ => {
                return Err(Error::invalid_input(format!(
                    "Column {} has element type {} and the query vector is {}",
                    column,
                    element_type,
                    q.data_type(),
                )));
            }
        };

        self.nearest = Some(Query {
            column: column.to_string(),
            key,
            k,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: None,
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
        });
        self.nearest_query_count = query_count;
        self.is_batch_nearest = is_batch_nearest;
        Ok(self)
    }

    #[cfg(test)]
    fn nearest_mut(&mut self) -> Option<&mut Query> {
        self.nearest.as_mut()
    }

    /// Set the distance thresholds for the nearest neighbor search.
    pub fn distance_range(
        &mut self,
        lower_bound: Option<f32>,
        upper_bound: Option<f32>,
    ) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.lower_bound = lower_bound;
            q.upper_bound = upper_bound;
        }
        self
    }

    /// Configures how many partititions will be searched in the vector index.
    ///
    /// This method is a convenience method that sets both [Self::minimum_nprobes] and
    /// [Self::maximum_nprobes] to the same value.
    pub fn nprobes(&mut self, n: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.minimum_nprobes = n;
            q.maximum_nprobes = Some(n);
        } else {
            log::warn!("nprobes is not set because nearest has not been called yet");
        }
        self
    }

    /// Configures how many partititions will be searched in the vector index.
    ///
    /// This method is a convenience method that sets both [Self::minimum_nprobes] and
    /// [Self::maximum_nprobes] to the same value.
    #[deprecated(note = "Use nprobes instead")]
    pub fn nprobs(&mut self, n: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.minimum_nprobes = n;
            q.maximum_nprobes = Some(n);
        } else {
            log::warn!("nprobes is not set because nearest has not been called yet");
        }
        self
    }

    /// Configures the minimum number of partitions to search in the vector index.
    ///
    /// If we have found k matching results after searching this many partitions then
    /// the search will stop.  Increasing this number can increase recall but will increase
    /// latency on all queries.
    ///
    /// The default value is 1.
    pub fn minimum_nprobes(&mut self, n: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.minimum_nprobes = n;
        } else {
            log::warn!("minimum_nprobes is not set because nearest has not been called yet");
        }
        self
    }

    /// Configures the maximum number of partitions to search in the vector index.
    ///
    /// These partitions will only be searched if we have not found `k` results after
    /// searching the minimum number of partitions.  Setting this to None (the default)
    /// will search all partitions if needed.
    ///
    /// This setting only takes effect when a prefilter is in place.  In that case we
    /// can spend more effort to try and find results when the filter is highly selective.
    ///
    /// If there is no prefilter, or the results are not highly selective, this value will
    /// have no effect.
    pub fn maximum_nprobes(&mut self, n: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.maximum_nprobes = Some(n);
        } else {
            log::warn!("maximum_nprobes is not set because nearest has not been called yet");
        }
        self
    }

    pub fn ef(&mut self, ef: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.ef = Some(ef);
        }
        self
    }

    /// Only search the data being indexed.
    ///
    /// Default value is false.
    ///
    /// This is essentially a weak consistency search, only on the indexed data.
    pub fn fast_search(&mut self) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.use_index = true;
        }
        self.fast_search = true;
        self.projection_plan.include_row_id(); // fast search requires _rowid
        self
    }

    /// Apply a refine step to the vector search.
    ///
    /// A refine improves query accuracy but also makes search slower, by reading extra elements
    /// and using the original vector values to re-rank the distances.
    ///
    /// * `factor` - the factor of extra elements to read.  For example, if factor is 2, then
    ///   the search will read 2x more elements than the requested k before performing
    ///   the re-ranking. Note: even if the factor is 1, the  results will still be
    ///   re-ranked without fetching additional elements.
    pub fn refine(&mut self, factor: u32) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.refine_factor = Some(factor)
        };
        self
    }

    /// Change the distance [MetricType], i.e, L2 or Cosine distance.
    pub fn distance_metric(&mut self, metric_type: MetricType) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.metric_type = Some(metric_type)
        }
        self
    }

    /// Sort the results of the scan by one or more columns
    ///
    /// If Some, then the resulting stream will be sorted according to the given ordering.
    /// This may increase the latency of the first result since all data must be read before
    /// the first batch can be returned.
    pub fn order_by(&mut self, ordering: Option<Vec<ColumnOrdering>>) -> Result<&mut Self> {
        if let Some(ordering) = &ordering {
            if ordering.is_empty() {
                self.ordering = None;
                return Ok(self);
            }
            // Verify early that the fields exist
            for column in ordering {
                self.dataset
                    .schema()
                    .field(&column.column_name)
                    .ok_or(Error::invalid_input(format!(
                        "Column {} not found",
                        &column.column_name
                    )))?;
            }
        }
        self.ordering = ordering;
        Ok(self)
    }

    /// Set whether to use the index if available
    pub fn use_index(&mut self, use_index: bool) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.use_index = use_index
        }
        self
    }

    /// Configure the speed / accuracy tradeoff for approximate vector search.
    ///
    /// This setting is currently only used by RQ-quantized indexes, such as
    /// IVF_RQ. Other index types ignore this setting.
    pub fn approx_mode(&mut self, approx_mode: ApproxMode) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.approx_mode = approx_mode;
        } else {
            log::warn!("approx_mode is not set because nearest has not been called yet");
        }
        self
    }

    /// Configure partition-search concurrency for each vector query.
    ///
    /// The default is 0.
    /// Value 0 selects the automatic policy; today this resolves to 1 for the
    /// sequential fast path unless an index implementation overrides it.
    /// Value -1 uses the CPU pool size.
    /// Value 1 uses the single-worker sequential partition search path.
    /// Values >= 2 use the partition-parallel path and are clamped to the CPU
    /// pool size by the execution layer.
    pub fn query_parallelism(&mut self, query_parallelism: i32) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.query_parallelism = query_parallelism;
        } else {
            log::warn!("query_parallelism is not set because nearest has not been called yet");
        }
        self
    }

    /// Instruct the scanner to return the `_rowid` meta column from the dataset.
    pub fn with_row_id(&mut self) -> &mut Self {
        self.legacy_with_row_id = true;
        self.projection_plan.include_row_id();
        self
    }

    /// Instruct the scanner to return the `_rowaddr` meta column from the dataset.
    pub fn with_row_address(&mut self) -> &mut Self {
        self.legacy_with_row_addr = true;
        self.projection_plan.include_row_addr();
        self
    }

    /// No-op kept for backwards compatibility. Scoring columns (`_distance`,
    /// `_score`) are no longer auto-projected; callers should include them
    /// explicitly in their projection instead.
    pub fn disable_scoring_autoprojection(&mut self) -> &mut Self {
        self
    }

    /// Set the file reader options to use when reading data files.
    pub fn with_file_reader_options(&mut self, options: FileReaderOptions) -> &mut Self {
        self.file_reader_options = Some(options);
        self
    }

    /// Compute the resolved file reader options, merging the scanner's explicit
    /// `file_reader_options`, the dataset-level defaults, and the `batch_size_bytes`
    /// setting.
    fn resolved_file_reader_options(&self) -> Option<FileReaderOptions> {
        let base = self
            .file_reader_options
            .clone()
            .or_else(|| self.dataset.file_reader_options.clone());
        match (base, self.batch_size_bytes) {
            (Some(mut opts), Some(bsb)) => {
                if opts.batch_size_bytes.is_none() {
                    opts.batch_size_bytes = Some(bsb);
                }
                Some(opts)
            }
            (Some(opts), None) => Some(opts),
            (None, Some(bsb)) => Some(FileReaderOptions {
                batch_size_bytes: Some(bsb),
                ..Default::default()
            }),
            (None, None) => None,
        }
    }

    /// Create a physical expression for a column that may be nested
    fn create_column_expr(
        column_name: &str,
        dataset: &Dataset,
        arrow_schema: &ArrowSchema,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let lance_schema = dataset.schema();
        let field_path = lance_schema
            .resolve_case_insensitive(column_name)
            .ok_or_else(|| {
                Error::invalid_input(format!("Field '{}' not found in schema", column_name))
            })?;

        if field_path.len() == 1 {
            // Simple top-level column
            expressions::col(&field_path[0].name, arrow_schema).map_err(|e| {
                Error::internal(format!(
                    "Failed to create column expression for '{}': {}",
                    column_name, e
                ))
            })
        } else {
            // Nested field - build a chain of GetFieldFunc calls
            let get_field_func = ScalarUDF::from(GetFieldFunc::default());

            // Use Expr::Column with Column::new_unqualified to preserve exact case
            // (col() normalizes identifiers to lowercase)
            let mut expr = Expr::Column(datafusion::common::Column::new_unqualified(
                &field_path[0].name,
            ));
            for nested_field in &field_path[1..] {
                expr = get_field_func.call(vec![expr, lit(&nested_field.name)]);
            }

            // Convert logical to physical expression
            let df_schema = Arc::new(DFSchema::try_from(arrow_schema.clone())?);
            let execution_props = ExecutionProps::new().with_query_execution_start_time(Utc::now());
            create_physical_expr(&expr, &df_schema, &execution_props).map_err(|e| {
                Error::internal(format!(
                    "Failed to create physical expression for nested field '{}': {}",
                    column_name, e
                ))
            })
        }
    }

    /// Set whether to use statistics to optimize the scan (default: true)
    ///
    /// This is used for debugging or benchmarking purposes.
    pub fn use_stats(&mut self, use_stats: bool) -> &mut Self {
        self.use_stats = use_stats;
        self
    }

    /// The Arrow schema of the output, including projections and vector / _distance
    pub async fn schema(&self) -> Result<SchemaRef> {
        let plan = self.create_plan().await?;
        Ok(plan.schema())
    }

    /// Fetches the currently set expr filter
    ///
    /// Note that this forces the filter to be evaluated and the result will depend on
    /// the current state of the scanner (e.g. if with_row_id has been called then _rowid
    /// will be available for filtering but not otherwise) and so you may want to call this
    /// after setting all other options.
    pub fn get_expr_filter(&self) -> Result<Option<Expr>> {
        if let Some(filter) = &self.filter.expr_filter {
            let filter_schema = self.filterable_schema()?;
            Ok(Some(filter.to_datafusion(
                self.dataset.schema(),
                filter_schema.as_ref(),
            )?))
        } else {
            Ok(None)
        }
    }

    fn add_extra_columns(&self, schema: Schema) -> Result<Schema> {
        let mut extra_columns = vec![ArrowField::new(ROW_OFFSET, DataType::UInt64, true)];

        if self.nearest.as_ref().is_some() {
            extra_columns.push(ArrowField::new(DIST_COL, DataType::Float32, true));
            if self.is_batch_nearest {
                extra_columns.push(query_index_field());
            }
        };

        if self.full_text_query.is_some() {
            extra_columns.push(ArrowField::new(SCORE_COL, DataType::Float32, true));
        }

        schema.merge(&ArrowSchema::new(extra_columns))
    }

    /// The full schema available to filters
    ///
    /// This is the schema of the dataset, any metadata columns like _rowid or _rowaddr
    /// and any extra columns like _distance or _score
    fn filterable_schema(&self) -> Result<Arc<Schema>> {
        let base_schema = Projection::full(self.dataset.clone())
            .with_row_id()
            .with_row_addr()
            .with_row_last_updated_at_version()
            .with_row_created_at_version()
            .to_schema();

        Ok(Arc::new(self.add_extra_columns(base_schema)?))
    }

    /// This takes the current output, and the user's requested projection, and calculates the
    /// final projection expression.
    ///
    /// This final expression may reorder columns, drop columns, or calculate new columns
    pub(crate) fn calculate_final_projection(
        &self,
        current_schema: &ArrowSchema,
    ) -> Result<Vec<(Arc<dyn PhysicalExpr>, String)>> {
        // Select the columns from the output schema based on the user's projection (or the list
        // of all available columns if the user did not specify a projection)
        let mut output_expr = self.projection_plan.to_physical_exprs(current_schema)?;

        // When no explicit projection is set (SELECT *), automatically append _distance/_score
        // so callers always receive scoring columns from a search.  When the user has specified
        // a projection they must explicitly list the columns they want.
        if !self.explicit_projection {
            if self.nearest.is_some() && output_expr.iter().all(|(_, name)| name != DIST_COL) {
                let vector_expr = expressions::col(DIST_COL, current_schema)?;
                output_expr.push((vector_expr, DIST_COL.to_string()));
            }
            if self.full_text_query.is_some()
                && output_expr.iter().all(|(_, name)| name != SCORE_COL)
            {
                let score_expr = expressions::col(SCORE_COL, current_schema)?;
                output_expr.push((score_expr, SCORE_COL.to_string()));
            }
        }

        // Batch nearest queries expose the synthetic `query_index` discriminator as
        // the first output column for compatibility with LanceDB batch vector search.
        if self.is_batch_nearest {
            let query_index_expr = if let Some(pos) = output_expr
                .iter()
                .position(|(_, name)| name == QUERY_INDEX_COL)
            {
                output_expr.remove(pos)
            } else {
                (
                    expressions::col(QUERY_INDEX_COL, current_schema)?,
                    QUERY_INDEX_COL.to_string(),
                )
            };
            output_expr.insert(0, query_index_expr);
        }

        if self.legacy_with_row_id {
            let row_id_pos = output_expr
                .iter()
                .position(|(_, name)| name == ROW_ID)
                .ok_or_else(|| {
                    Error::internal(
                        "user specified with_row_id but the _rowid column was not in the output"
                            .to_string(),
                    )
                })?;
            if row_id_pos != output_expr.len() - 1 {
                // Row id is not last column.  Need to rotate it to the last spot.
                let row_id_expr = output_expr.remove(row_id_pos);
                output_expr.push(row_id_expr);
            }
        }

        if self.legacy_with_row_addr {
            let row_addr_pos = output_expr.iter().position(|(_, name)| name == ROW_ADDR).ok_or_else(|| {
                Error::internal("user specified with_row_address but the _rowaddr column was not in the output".to_string())
            })?;
            if row_addr_pos != output_expr.len() - 1 {
                // Row addr is not last column.  Need to rotate it to the last spot.
                let row_addr_expr = output_expr.remove(row_addr_pos);
                output_expr.push(row_addr_expr);
            }
        }

        Ok(output_expr)
    }

    /// Create a stream from the Scanner.
    #[instrument(skip_all)]
    pub fn try_into_stream(&self) -> BoxFuture<'_, Result<DatasetRecordBatchStream>> {
        // Future intentionally boxed here to avoid large futures on the stack
        async move {
            let plan = self.create_plan().await?;

            Ok(DatasetRecordBatchStream::new(execute_plan(
                plan,
                LanceExecutionOptions {
                    batch_size: self.batch_size,
                    execution_stats_callback: self.scan_stats_callback.clone(),
                    ..Default::default()
                },
            )?))
        }
        .boxed()
    }

    pub(crate) async fn try_into_dfstream(
        &self,
        mut options: LanceExecutionOptions,
    ) -> Result<SendableRecordBatchStream> {
        let plan = self.create_plan().await?;

        // Use the scan stats callback if the user didn't set an execution stats callback
        if options.execution_stats_callback.is_none() {
            options.execution_stats_callback = self.scan_stats_callback.clone();
        }

        execute_plan(plan, options)
    }

    pub(crate) fn execution_options(&self) -> LanceExecutionOptions {
        LanceExecutionOptions {
            batch_size: self.batch_size,
            execution_stats_callback: self.scan_stats_callback.clone(),
            ..Default::default()
        }
    }

    pub async fn try_into_batch(&self) -> Result<RecordBatch> {
        let stream = self.try_into_stream().await?;
        let schema = stream.schema();
        let batches = stream.try_collect::<Vec<_>>().await?;
        Ok(concat_batches(&schema, &batches)?)
    }

    /// Scan and return the number of matching rows
    ///
    /// Note: calling [`Dataset::count_rows`] can be more efficient than calling this method
    /// especially if there is no filter.
    #[instrument(skip_all)]
    pub fn count_rows(&self) -> BoxFuture<'_, Result<u64>> {
        // Future intentionally boxed here to avoid large futures on the stack
        async move {
            let mut scanner = self.clone();
            scanner.aggregate(AggregateExpr::builder().count_star().build())?;

            let plan = scanner.create_plan().await?;
            let mut stream = execute_plan(plan, LanceExecutionOptions::default())?;

            // A count plan will always return a single batch with a single row.
            if let Some(first_batch) = stream.next().await {
                let batch = first_batch?;
                let array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or(Error::invalid_input(
                        "Count plan did not return an Int64Array".to_string(),
                    ))?;
                Ok(array.value(0) as u64)
            } else {
                Ok(0)
            }
        }
        .boxed()
    }

    /// Create an execution plan with aggregation.
    ///
    /// Requires `aggregate()` to be called first.
    #[deprecated(note = "Use create_plan() instead, which now applies aggregate automatically")]
    pub fn create_aggregate_plan(&self) -> BoxFuture<'_, Result<Arc<dyn ExecutionPlan>>> {
        async move {
            if self.aggregate.is_none() {
                return Err(Error::invalid_input(
                    "create_aggregate_plan called but no aggregate was set",
                ));
            }
            // create_plan() now applies aggregate automatically when set
            self.create_plan().await
        }
        .boxed()
    }

    async fn apply_aggregate(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        agg: &Aggregate,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion_physical_expr::aggregate::AggregateFunctionExpr;

        let schema = plan.schema();
        let df_schema = DFSchema::try_from(schema.as_ref().clone())?;

        let group_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = agg
            .group_by
            .iter()
            .map(|expr| {
                let name = expr.schema_name().to_string();
                let physical_expr =
                    create_physical_expr(expr, &df_schema, &ExecutionProps::default())?;
                Ok((physical_expr, name))
            })
            .collect::<Result<_>>()?;

        #[allow(clippy::type_complexity)]
        let aggr_results: Vec<(Arc<AggregateFunctionExpr>, Option<Arc<dyn PhysicalExpr>>)> = agg
            .aggregates
            .iter()
            .map(|expr| self.build_physical_aggregate_expr(expr, &df_schema, &schema))
            .collect::<Result<_>>()?;

        let (aggr_exprs, filters): (Vec<_>, Vec<_>) = aggr_results.into_iter().unzip();

        Ok(Arc::new(AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(group_exprs),
            aggr_exprs,
            filters,
            plan,
            schema,
        )?) as Arc<dyn ExecutionPlan>)
    }

    #[allow(clippy::type_complexity)]
    fn build_physical_aggregate_expr(
        &self,
        expr: &Expr,
        df_schema: &DFSchema,
        input_schema: &SchemaRef,
    ) -> Result<(
        Arc<datafusion_physical_expr::aggregate::AggregateFunctionExpr>,
        Option<Arc<dyn PhysicalExpr>>,
    )> {
        use datafusion::physical_planner::create_aggregate_expr_and_maybe_filter;

        let coerced_expr = self.coerce_aggregate_expr(expr, df_schema)?;

        // Note: order_by is already embedded in the AggregateFunctionExpr for ordered aggregates
        let (agg_expr, filter, _order_by) = create_aggregate_expr_and_maybe_filter(
            &coerced_expr,
            df_schema,
            input_schema.as_ref(),
            &ExecutionProps::default(),
        )?;

        Ok((agg_expr, filter))
    }

    /// Apply type coercion to aggregate arguments for UserDefined signature functions.
    ///
    /// Most aggregate functions (SUM, COUNT, MIN, MAX) have explicit type signatures that
    /// DataFusion handles automatically. However, some functions like AVG use UserDefined
    /// type signatures in the Substrait consumer, which means DataFusion doesn't know the
    /// expected input types and won't perform automatic coercion. We must explicitly coerce
    /// arguments to the types returned by `func.coerce_types()`.
    fn coerce_aggregate_expr(&self, expr: &Expr, schema: &DFSchema) -> Result<Expr> {
        Self::coerce_aggregate_expr_impl(expr, schema)
    }

    fn coerce_aggregate_expr_impl(expr: &Expr, schema: &DFSchema) -> Result<Expr> {
        use datafusion::logical_expr::Expr;
        use datafusion::logical_expr::expr::AggregateFunction;
        use datafusion::logical_expr::type_coercion::functions::fields_with_udf;

        match expr {
            Expr::AggregateFunction(agg_func) => {
                let func = &agg_func.func;
                let args = &agg_func.params.args;

                if args.is_empty() {
                    return Ok(expr.clone());
                }

                let current_fields: Vec<arrow_schema::FieldRef> = args
                    .iter()
                    .enumerate()
                    .map(|(i, e)| {
                        let dt = e.get_type(schema)?;
                        Ok(Arc::new(arrow_schema::Field::new(
                            format!("arg_{i}"),
                            dt,
                            true,
                        )))
                    })
                    .collect::<std::result::Result<_, datafusion::common::DataFusionError>>()?;

                let coerced_fields = fields_with_udf(&current_fields, func.as_ref())?;
                let coerced_args: Vec<Expr> = args
                    .iter()
                    .zip(coerced_fields.iter())
                    .map(|(arg, target_field)| {
                        let arg_type = arg.get_type(schema)?;
                        let target_type = target_field.data_type();
                        if arg_type == *target_type {
                            Ok(arg.clone())
                        } else {
                            arg.clone().cast_to(target_type, schema)
                        }
                    })
                    .collect::<std::result::Result<_, _>>()?;

                Ok(Expr::AggregateFunction(AggregateFunction::new_udf(
                    func.clone(),
                    coerced_args,
                    agg_func.params.distinct,
                    agg_func.params.filter.clone(),
                    agg_func.params.order_by.clone(),
                    agg_func.params.null_treatment,
                )))
            }
            Expr::Alias(alias) => {
                // Recursively coerce the inner expression and preserve the alias
                let coerced_inner = Self::coerce_aggregate_expr_impl(&alias.expr, schema)?;
                Ok(coerced_inner.alias(&alias.name))
            }
            other => Err(Error::invalid_input(format!(
                "Expected aggregate function expression, got {:?}",
                other.variant_name()
            ))),
        }
    }

    // A "narrow" field is a field that is so small that we are better off reading the
    // entire column and filtering in memory rather than "take"ing the column.
    //
    // The exact threshold depends on a two factors:
    // 1. The number of rows returned by the filter
    // 2. The number of rows in the dataset
    // 3. The IOPS/bandwidth ratio of the storage system
    // 4. The size of each value in the column
    //
    // We don't (today) have a good way of knowing #1 or #4.  #2 is easy to know.  We can
    // combine 1 & 2 into "percentage of rows returned" but since we don't know #1 it
    // doesn't really help.  #3 is complex but as a rule of thumb we can use:
    //
    //   Local storage: 1 IOP for ever ten thousand bytes
    //   Cloud storage: 1 IOP for every million bytes
    //
    // Our current heuristic today is to assume a filter will return 0.1% of the rows in the dataset.
    //
    // This means, for cloud storage, a field is "narrow" if there are 1KB of data per row and
    // for local disk a field is "narrow" if there are 10 bytes of data per row.
    fn is_early_field(&self, field: &Field) -> bool {
        match self.materialization_style {
            MaterializationStyle::AllEarly => true,
            MaterializationStyle::AllLate => false,
            MaterializationStyle::AllEarlyExcept(ref cols) => !cols.contains(&(field.id as u32)),
            MaterializationStyle::Heuristic => {
                if field.is_blob() {
                    // By default, blobs are loaded as descriptions, and so should be early
                    //
                    // TODO: Once we make blob handling configurable, we should use the blob
                    // handling setting here.
                    return true;
                }

                let byte_width = field.data_type().byte_width_opt();
                let is_cloud = self.dataset.object_store.as_ref().is_cloud();
                if is_cloud {
                    byte_width.is_some_and(|bw| bw < 1000)
                } else {
                    byte_width.is_some_and(|bw| bw < 10)
                }
            }
        }
    }

    // If we are going to filter on `filter_plan`, then which columns are so small it is
    // cheaper to read the entire column and filter in memory.
    //
    // Note: only add columns that we actually need to read
    fn calc_eager_projection(
        &self,
        filter_plan: &ExprFilterPlan,
        desired_projection: &Projection,
    ) -> Result<Projection> {
        // Note: We use all_columns and not refine_columns here.  If a column is covered by an index but
        // the user has requested it, then we do not use it for late materialization.
        //
        // Either that column is covered by an exact filter (e.g. string with bitmap/btree) and there is no
        // need for late materialization or that column is covered by an inexact filter (e.g. ngram) in which
        // case we are going to load the column anyways for the recheck.
        let filter_columns = filter_plan.all_columns();

        let filter_schema = self
            .dataset
            .empty_projection()
            .union_columns(filter_columns, OnMissing::Error)?
            .into_schema();

        // Start with the desired fields
        Ok(desired_projection
            .clone()
            // Subtract columns that are expensive
            .subtract_predicate(|f| !self.is_early_field(f))
            // Add back columns that we need for filtering
            .union_schema(&filter_schema))
    }

    fn validate_options(&self) -> Result<()> {
        if self.include_deleted_rows && !self.projection_plan.physical_projection.with_row_id {
            return Err(Error::invalid_input_source(
                "include_deleted_rows is set but with_row_id is false".into(),
            ));
        }

        if self.aggregate.is_some() {
            if self.limit.is_some() || self.offset.is_some() {
                return Err(Error::invalid_input_source(
                    "Cannot use limit/offset with aggregate. Apply limit to the result instead."
                        .into(),
                ));
            }
            if self.ordering.is_some() {
                return Err(Error::invalid_input_source(
                    "Cannot use order_by with aggregate. Apply ordering to the result instead."
                        .into(),
                ));
            }
        }

        if self.index_segments.is_some() && self.nearest.is_none() {
            return Err(Error::not_supported(
                "with_index_segments is only supported for vector search".to_string(),
            ));
        }

        Ok(())
    }

    async fn create_filter_plan(&self, use_scalar_index: bool) -> Result<FilterPlan> {
        let filter_schema = self.filterable_schema()?;
        let planner = Planner::new(Arc::new(filter_schema.as_ref().into()));

        // Check expr filter
        let filter_plan = if let Some(filter) = self.filter.expr_filter.as_ref() {
            let expr = filter.to_datafusion(self.dataset.schema(), filter_schema.as_ref())?;
            let index_info = self.dataset.scalar_index_info().await?;
            let filter_plan =
                planner.create_filter_plan(expr.clone(), &index_info, use_scalar_index)?;

            // This tests if any of the fragments are missing the physical_rows property (old style)
            // If they are then we cannot use scalar indices
            if filter_plan.index_query.is_some() {
                let fragments = if let Some(fragments) = self.fragments.as_ref() {
                    fragments
                } else {
                    self.dataset.fragments()
                };
                let mut has_missing_row_count = false;
                for frag in fragments {
                    if frag.physical_rows.is_none() {
                        has_missing_row_count = true;
                        break;
                    }
                }
                if has_missing_row_count {
                    // We need row counts to use scalar indices.  If we don't have them then
                    // fallback to a non-indexed filter
                    let filter_plan =
                        planner.create_filter_plan(expr.clone(), &index_info, false)?;
                    FilterPlan::new(self.filter.query_filter.clone(), filter_plan)
                } else {
                    FilterPlan::new(self.filter.query_filter.clone(), filter_plan)
                }
            } else {
                FilterPlan::new(self.filter.query_filter.clone(), filter_plan)
            }
        } else {
            FilterPlan::new(self.filter.query_filter.clone(), ExprFilterPlan::default())
        };

        // Check query filter
        if filter_plan.query_filter.is_some()
            && self.nearest.is_none()
            && self.full_text_query.is_none()
        {
            return Err(Error::invalid_input_source(
                "Query filter can only be used with full text search or vector search".into(),
            ));
        }
        if self.nearest.is_some() && filter_plan.vector_filter().is_some() {
            return Err(Error::invalid_input_source(
                "Query filter can't be used with vector search".into(),
            ));
        }
        if self.full_text_query.is_some() && filter_plan.fts_filter().is_some() {
            return Err(Error::invalid_input_source(
                "Fts filter can't be used with fts search".into(),
            ));
        }

        Ok(filter_plan)
    }

    async fn get_scan_range(&self, filter_plan: &ExprFilterPlan) -> Result<Option<Range<u64>>> {
        if filter_plan.has_any_filter() {
            // If there is a filter we can't pushdown limit / offset
            Ok(None)
        } else if self.ordering.is_some() {
            // If there is ordering, we can't pushdown limit / offset
            // because we need to sort all data first before applying the limit
            Ok(None)
        } else if self.dataset.manifest.uses_stable_row_ids() {
            // Stable-row-id datasets can contain deleted / rewritten rows that still occupy
            // physical positions in older fragments while the live replacement rows are appended
            // to new fragments. `scan_range_before_filter` is a logical offset over visible rows,
            // but filtered-read planning trims fragments before the stable-row-id/deletion-aware
            // remapping is finished. Pushing limit / offset down here can spend the range on
            // tombstoned positions and skip still-live rows in later fragments.
            Ok(None)
        } else {
            match (self.limit, self.offset) {
                (None, None) => Ok(None),
                (Some(limit), None) => {
                    let num_rows = self.dataset.count_all_rows().await? as i64;
                    Ok(Some(0..limit.min(num_rows) as u64))
                }
                (None, Some(offset)) => {
                    let num_rows = self.dataset.count_all_rows().await? as i64;
                    Ok(Some(offset.min(num_rows) as u64..num_rows as u64))
                }
                (Some(limit), Some(offset)) => {
                    let num_rows = self.dataset.count_all_rows().await? as i64;
                    Ok(Some(
                        offset.min(num_rows) as u64..(offset + limit).min(num_rows) as u64,
                    ))
                }
            }
        }
    }

    /// Create [`ExecutionPlan`] for Scan.
    ///
    /// An ExecutionPlan is a graph of operators that can be executed.
    ///
    /// The following plans are supported:
    ///
    ///  - **Plain scan without filter or limits.**
    ///
    ///  ```ignore
    ///  Scan(projections)
    ///  ```
    ///
    ///  - **Scan with filter and/or limits.**
    ///
    ///  ```ignore
    ///  Scan(filtered_cols) -> Filter(expr)
    ///     -> (*LimitExec(limit, offset))
    ///     -> Take(remaining_cols) -> Projection()
    ///  ```
    ///
    ///  - **Use KNN Index (with filter and/or limits)**
    ///
    /// ```ignore
    /// KNNIndex() -> Take(vector) -> FlatRefine()
    ///     -> Take(filtered_cols) -> Filter(expr)
    ///     -> (*LimitExec(limit, offset))
    ///     -> Take(remaining_cols) -> Projection()
    /// ```
    ///
    /// - **Use KNN flat (brute force) with filter and/or limits**
    ///
    /// ```ignore
    /// Scan(vector) -> FlatKNN()
    ///     -> Take(filtered_cols) -> Filter(expr)
    ///     -> (*LimitExec(limit, offset))
    ///     -> Take(remaining_cols) -> Projection()
    /// ```
    ///
    /// In general, a plan has 5 stages:
    ///
    /// 1. Source (from dataset Scan or from index, may include prefilter)
    /// 2. Filter
    /// 3. Sort
    /// 4. Limit / Offset
    /// 5. Take remaining columns / Projection
    #[instrument(level = "debug", skip_all)]
    pub async fn create_plan(&self) -> Result<Arc<dyn ExecutionPlan>> {
        log::trace!("creating scanner plan");
        self.validate_options()?;

        // Scalar indices are only used when prefiltering
        let use_scalar_index = self.use_scalar_index && (self.prefilter || self.nearest.is_none());
        let mut filter_plan = self.create_filter_plan(use_scalar_index).await?;

        let mut use_limit_node = true;
        // Source: either a (K|A)NN search, full text search, or a (full|indexed) scan
        let mut plan: Arc<dyn ExecutionPlan> = match (&self.nearest, &self.full_text_query) {
            (Some(_), None) => self.vector_search_source(&mut filter_plan).await?,
            (None, Some(query)) => self.fts_search_source(&mut filter_plan, query).await?,
            (None, None) => {
                if self.projection_plan.has_output_cols()
                    && self.projection_plan.physical_projection.is_empty()
                {
                    // This means the user is doing something like `SELECT 1 AS foo`.  We don't support this and
                    // I'm not sure we should.  Users should use a full SQL API to do something like this.
                    //
                    // It's also possible we get here from `SELECT does_not_exist`

                    // Note: even though we are just going to return an error we still want to calculate the
                    // final projection here.  This lets us distinguish between a user doing something like:
                    //
                    // SELECT 1 FROM t (not supported error)
                    // SELECT non_existent_column FROM t (column not found error)
                    let output_expr = self.calculate_final_projection(&ArrowSchema::empty())?;
                    return Err(Error::not_supported_source(format!("Scans must request at least one column.  Received only dynamic expressions: {:?}", output_expr).into()));
                }

                let take_op = filter_plan
                    .expr_filter_plan
                    .full_expr
                    .as_ref()
                    .and_then(TakeOperation::try_from_expr);
                if let Some((take_op, remainder)) = take_op {
                    // If there is any remainder use it as the filter (we don't even try and combine an indexed
                    // search on the filter with a take as that seems excessive)
                    filter_plan.expr_filter_plan = remainder
                        .map(ExprFilterPlan::new_refine_only)
                        .unwrap_or(ExprFilterPlan::default());
                    self.take_source(take_op).await?
                } else {
                    let planned_read = self
                        .filtered_read_source(&mut filter_plan.expr_filter_plan)
                        .await?;
                    if planned_read.limit_pushed_down {
                        use_limit_node = false;
                    }
                    if planned_read.filter_pushed_down {
                        filter_plan.disable_refine();
                    }
                    planned_read.plan
                }
            }
            _ => {
                return Err(Error::invalid_input_source(
                    "Cannot have both nearest and full text search".into(),
                ));
            }
        };

        // Load columns needed for filter and ordering
        let mut pre_filter_projection = self.dataset.empty_projection();

        // We may need to take filter columns if we are going to refine
        // an indexed scan.
        if filter_plan.has_refine() {
            // It's ok for some filter columns to be missing (e.g. _rowid)
            pre_filter_projection = pre_filter_projection.union_columns(
                filter_plan.refine_columns(&self.dataset).await?,
                OnMissing::Ignore,
            )?;
        }

        // TODO: Does it always make sense to take the ordering columns here?  If there is a filter then
        // maybe we wait until after the filter to take the ordering columns?  Maybe it would be better to
        // grab the ordering column in the initial scan (if it is eager) and if it isn't then we should
        // take it after the filtering phase, if any (we already have a take there).
        if let Some(ordering) = &self.ordering {
            pre_filter_projection = pre_filter_projection.union_columns(
                ordering.iter().map(|col| &col.column_name),
                OnMissing::Error,
            )?;
        }

        plan = self.take(plan, pre_filter_projection)?;

        // Filter
        plan = filter_plan.refine_filter(plan, self).await?;

        // Aggregate (if set, applies aggregate and returns early)
        if let Some(agg) = &self.aggregate {
            // Take only columns needed by the aggregate, not the full projection.
            // For COUNT(*), this is empty. For SUM(x), this is just [x].
            let required_columns = agg.required_columns();
            let agg_projection = if required_columns.is_empty() {
                self.dataset.empty_projection()
            } else {
                self.dataset
                    .empty_projection()
                    .union_columns(&required_columns, OnMissing::Error)?
            };
            plan = self.take(plan, agg_projection)?;
            plan = self.apply_aggregate(plan, agg).await?;

            let optimizer = get_physical_optimizer();
            let mut options = ConfigOptions::default();
            options.execution.target_partitions = self
                .target_parallelism
                .unwrap_or_else(get_num_compute_intensive_cpus);
            for rule in optimizer.rules {
                plan = rule.optimize(plan, &options)?;
            }

            return Ok(plan);
        }

        // Sort
        if let Some(ordering) = &self.ordering {
            let ordering_columns = ordering.iter().map(|col| &col.column_name);
            let projection_with_ordering = self
                .dataset
                .empty_projection()
                .union_columns(ordering_columns, OnMissing::Error)?;
            // We haven't loaded the sort column yet so take it now
            plan = self.take(plan, projection_with_ordering)?;
            let col_exprs = ordering
                .iter()
                .map(|col| {
                    Ok(PhysicalSortExpr {
                        expr: Self::create_column_expr(
                            &col.column_name,
                            &self.dataset,
                            plan.schema().as_ref(),
                        )?,
                        options: SortOptions {
                            descending: !col.ascending,
                            nulls_first: col.nulls_first,
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            plan = Arc::new(SortExec::new(
                LexOrdering::new(col_exprs)
                    .ok_or(exec_datafusion_err!("Unexpected empty sort expressions"))?,
                plan,
            ));
        }

        // Limit / offset
        if use_limit_node && (self.limit.unwrap_or(0) > 0 || self.offset.is_some()) {
            plan = self.limit_node(plan);
        }

        // Take remaining columns required for projection
        plan = self.take(plan, self.projection_plan.physical_projection.clone())?;

        // Add system columns, if requested
        if self.projection_plan.must_add_row_offset {
            plan = Arc::new(AddRowOffsetExec::try_new(plan, self.dataset.clone()).await?);
        }

        // Final projection
        let final_projection = self.calculate_final_projection(plan.schema().as_ref())?;

        plan = Arc::new(DFProjectionExec::try_new(final_projection, plan)?);

        // If requested, apply a strict batch size to the final output
        if self.strict_batch_size {
            plan = Arc::new(StrictBatchSizeExec::new(plan, self.get_batch_size()));
        }

        let optimizer = get_physical_optimizer();
        let mut options = ConfigOptions::default();
        options.execution.target_partitions = self
            .target_parallelism
            .unwrap_or_else(get_num_compute_intensive_cpus);
        for rule in optimizer.rules {
            plan = rule.optimize(plan, &options)?;
        }

        Ok(plan)
    }

    // Check if a filter plan references version columns
    fn filter_references_version_columns(&self, filter_plan: &ExprFilterPlan) -> bool {
        use lance_core::{ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION};

        if let Some(refine_expr) = &filter_plan.refine_expr {
            let column_names = Planner::column_names_in_expr(refine_expr);
            for col_name in column_names {
                if col_name == ROW_CREATED_AT_VERSION || col_name == ROW_LAST_UPDATED_AT_VERSION {
                    return true;
                }
            }
        }
        false
    }

    // Helper function for filtered_read
    //
    // Do not call this directly, use filtered_read instead
    //
    // First return value is the plan, second is whether the limit was pushed down
    async fn legacy_filtered_read(
        &self,
        filter_plan: &ExprFilterPlan,
        projection: Projection,
        make_deletions_null: bool,
        fragments: Option<Arc<Vec<Fragment>>>,
        scan_range: Option<Range<u64>>,
        is_prefilter: bool,
    ) -> Result<PlannedFilteredScan> {
        let fragments = fragments.unwrap_or(self.dataset.fragments().clone());
        let mut filter_pushed_down = false;

        let plan: Arc<dyn ExecutionPlan> = if filter_plan.has_index_query() {
            if self.include_deleted_rows {
                return Err(Error::invalid_input_source(
                    "Cannot include deleted rows in a scalar indexed scan".into(),
                ));
            }
            self.scalar_indexed_scan(projection, filter_plan, fragments)
                .await
        } else if !is_prefilter
            && filter_plan.has_refine()
            && self.batch_size.is_none()
            && self.use_stats
            && !self.filter_references_version_columns(filter_plan)
        {
            filter_pushed_down = true;
            self.pushdown_scan(false, filter_plan)
        } else {
            let ordered = if self.ordering.is_some() || self.nearest.is_some() {
                // If we are sorting the results there is no need to scan in order
                false
            } else if projection.with_row_last_updated_at_version
                || projection.with_row_created_at_version
            {
                // Version columns require ordered scanning because version metadata
                // is indexed by position within each fragment
                true
            } else {
                self.ordered
            };

            let projection = if let Some(refine_expr) = filter_plan.refine_expr.as_ref() {
                if is_prefilter {
                    let refine_cols = Planner::column_names_in_expr(refine_expr);
                    projection.union_columns(refine_cols, OnMissing::Error)?
                } else {
                    projection
                }
            } else {
                projection
            };

            // Can't push down limit for legacy scan if there is a refine step
            let scan_range = if filter_plan.has_refine() {
                None
            } else {
                scan_range
            };

            let scan = self.scan_fragments(
                projection.with_row_id,
                self.projection_plan.physical_projection.with_row_addr,
                self.projection_plan
                    .physical_projection
                    .with_row_last_updated_at_version,
                self.projection_plan
                    .physical_projection
                    .with_row_created_at_version,
                make_deletions_null,
                Arc::new(projection.to_bare_schema()),
                fragments,
                scan_range,
                ordered,
            );

            if filter_plan.has_refine() && is_prefilter {
                Ok(Arc::new(LanceFilterExec::try_new(
                    filter_plan.refine_expr.clone().unwrap(),
                    scan,
                )?) as Arc<dyn ExecutionPlan>)
            } else {
                Ok(scan)
            }
        }?;
        Ok(PlannedFilteredScan {
            plan,
            limit_pushed_down: false,
            filter_pushed_down,
        })
    }

    fn index_expr_result_format(&self) -> IndexExprResultWireFormat {
        if self.relational_algebra_version > 1 {
            IndexExprResultWireFormat::TwoMask
        } else {
            // In version 1 we used the legacy three-variant format for index expr results
            IndexExprResultWireFormat::ThreeVariant
        }
    }

    // Helper function for filtered_read
    //
    // Do not call this directly, use filtered_read instead
    async fn new_filtered_read(
        &self,
        filter_plan: &ExprFilterPlan,
        projection: Projection,
        make_deletions_null: bool,
        fragments: Option<Arc<Vec<Fragment>>>,
        scan_range: Option<Range<u64>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut read_options = FilteredReadOptions::basic_full_read(&self.dataset)
            .with_filter_plan(filter_plan.clone())
            .with_projection(projection);

        if let Some(fragments) = fragments {
            read_options = read_options.with_fragments(fragments);
        }

        if let Some(scan_range) = scan_range {
            read_options = read_options.with_scan_range_before_filter(scan_range)?;
        }

        if let Some(batch_size) = self.batch_size {
            read_options = read_options.with_batch_size(batch_size as u32);
        }

        if let Some(file_reader_options) = self.resolved_file_reader_options() {
            read_options = read_options.with_file_reader_options(file_reader_options);
        }

        if let Some(fragment_readahead) = self.fragment_readahead {
            read_options = read_options.with_fragment_readahead(fragment_readahead);
        }

        if make_deletions_null {
            read_options = read_options.with_deleted_rows()?;
        }

        if let Some(io_buffer_size_bytes) = self.io_buffer_size {
            read_options = read_options.with_io_buffer_size(io_buffer_size_bytes);
        }

        if self.fast_search && filter_plan.has_index_query() {
            read_options = read_options.with_only_indexed_fragments();
        }

        let result_format = self.index_expr_result_format();
        let index_input = filter_plan.index_query.clone().map(|index_query| {
            Arc::new(ScalarIndexExec::new(
                self.dataset.clone(),
                index_query,
                result_format,
            )) as Arc<dyn ExecutionPlan>
        });

        Ok(Arc::new(FilteredReadExec::try_new(
            self.dataset.clone(),
            read_options,
            index_input,
        )?))
    }

    // Helper function for filtered read
    //
    // Delegates to legacy or new filtered read based on dataset storage version
    async fn filtered_read(
        &self,
        filter_plan: &ExprFilterPlan,
        projection: Projection,
        make_deletions_null: bool,
        fragments: Option<Arc<Vec<Fragment>>>,
        scan_range: Option<Range<u64>>,
        is_prefilter: bool,
    ) -> Result<PlannedFilteredScan> {
        // Use legacy path if dataset uses legacy storage format
        if self.dataset.is_legacy_storage() {
            self.legacy_filtered_read(
                filter_plan,
                projection,
                make_deletions_null,
                fragments,
                scan_range,
                is_prefilter,
            )
            .await
        } else {
            let limit_pushed_down = scan_range.is_some();
            let plan = self
                .new_filtered_read(
                    filter_plan,
                    projection,
                    make_deletions_null,
                    fragments,
                    scan_range,
                )
                .await?;
            Ok(PlannedFilteredScan {
                filter_pushed_down: true,
                limit_pushed_down,
                plan,
            })
        }
    }

    fn u64s_as_take_input(&self, u64s: Vec<u64>) -> Result<Arc<dyn ExecutionPlan>> {
        let row_addrs = RowAddrTreeMap::from_iter(u64s);
        let row_addr_mask = RowAddrMask::from_allowed(row_addrs);
        let index_result = IndexExprResult::exact(row_addr_mask);
        let fragments_covered = self.dataset.fragment_bitmap.as_ref().clone();
        let format = self.index_expr_result_format();
        let batch = index_result.serialize(&fragments_covered, format)?;
        let schema = batch.schema();
        let stream = futures::stream::once(async move { Ok(batch) });
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));
        Ok(Arc::new(OneShotExec::new(stream)))
    }

    async fn take_source(&self, take_op: TakeOperation) -> Result<Arc<dyn ExecutionPlan>> {
        // We generally assume that late materialization does not make sense for take operations
        // so we can just use the physical projection
        let projection = self.projection_plan.physical_projection.clone();

        let input = match take_op {
            TakeOperation::RowIds(ids) => self.u64s_as_take_input(ids),
            TakeOperation::RowAddrs(addrs) => self.u64s_as_take_input(addrs),
            TakeOperation::RowOffsets(offsets) => {
                let mut addrs =
                    row_offsets_to_row_addresses(&self.dataset.get_fragments(), &offsets).await?;
                addrs.retain(|addr| *addr != RowAddress::TOMBSTONE_ROW);
                self.u64s_as_take_input(addrs)
            }
        }?;

        let mut filtered_read_options = FilteredReadOptions::new(projection);
        if let Some(fragment) = self.fragments.as_ref() {
            filtered_read_options =
                filtered_read_options.with_fragments(Arc::new(fragment.clone()));
        }

        Ok(Arc::new(FilteredReadExec::try_new(
            self.dataset.clone(),
            filtered_read_options,
            Some(input),
        )?))
    }

    async fn filtered_read_source(
        &self,
        filter_plan: &mut ExprFilterPlan,
    ) -> Result<PlannedFilteredScan> {
        log::trace!("source is a filtered read");

        // Compute the effective projection based on what's actually needed.
        // If we have an aggregate, we only need the columns referenced by the aggregate,
        // not all the columns from the projection plan.
        let effective_projection = if let Some(agg) = &self.aggregate {
            let required_columns = agg.required_columns();
            if required_columns.is_empty() {
                // COUNT(*) or similar - no columns needed
                self.dataset.empty_projection()
            } else {
                // Aggregate needs specific columns
                self.dataset
                    .empty_projection()
                    .union_columns(&required_columns, OnMissing::Error)?
            }
        } else {
            self.projection_plan.physical_projection.clone()
        };

        let mut projection = if filter_plan.has_refine() {
            // If the filter plan has two steps (a scalar indexed portion and a refine portion) then
            // it makes sense to grab cheap columns during the first step to avoid taking them for
            // the second step.
            self.calc_eager_projection(filter_plan, &effective_projection)?
                .with_row_id()
        } else {
            // If the filter plan only has one step then we just do a filtered read of all the
            // columns that the user asked for.
            effective_projection
        };

        if projection.is_empty() {
            // If the user is not requesting any columns then we will scan the row address which
            // is cheap
            projection.with_row_addr = true;
        }

        let scan_range = if filter_plan.is_empty() {
            log::trace!("pushing scan_range into filtered_read");
            self.get_scan_range(filter_plan).await?
        } else {
            None
        };

        self.filtered_read(
            filter_plan,
            projection,
            self.include_deleted_rows,
            self.fragments.clone().map(Arc::new),
            scan_range,
            /*is_prefilter= */ false,
        )
        .await
    }

    async fn fts_search_source(
        &self,
        filter_plan: &mut FilterPlan,
        query: &FullTextSearchQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        log::trace!("source is an fts search");
        if self.include_deleted_rows {
            return Err(Error::invalid_input_source(
                "Cannot include deleted rows in an FTS search".into(),
            ));
        }

        // The source is an FTS search
        if self.prefilter {
            let source: Arc<dyn ExecutionPlan> = match &filter_plan.vector_filter() {
                Some(vector_query) => {
                    // Perform vector search first then rerank according to BM25 scores
                    let vector_plan = self
                        .vector_search(&filter_plan.expr_filter_plan, vector_query)
                        .await?;
                    self.fts_rerank(vector_plan, query).await?
                }
                None => self.fts(&filter_plan.expr_filter_plan, query).await?,
            };
            // If we are prefiltering then the fts node will take care of the filter
            filter_plan.disable_refine();
            Ok(source)
        } else {
            // If we are postfiltering then we can't use scalar indices for the filter
            // and will need to run the postfilter in memory
            filter_plan.make_refine_only();
            self.fts(&ExprFilterPlan::default(), query).await
        }
    }

    async fn vector_search_source(
        &self,
        filter_plan: &mut FilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if self.include_deleted_rows {
            return Err(Error::invalid_input_source(
                "Cannot include deleted rows in a nearest neighbor search".into(),
            ));
        }
        let Some(query) = self.nearest.as_ref() else {
            return Err(Error::invalid_input("No nearest query".to_string()));
        };

        if self.prefilter {
            log::trace!("source is a vector search (prefilter)");
            // If we are prefiltering then the ann / knn node will take care of the filter
            let source: Arc<dyn ExecutionPlan> = match &filter_plan.fts_filter() {
                Some(fts_query) => {
                    let fts_plan = self.fts(&filter_plan.expr_filter_plan, fts_query).await?;
                    let projection = self
                        .dataset
                        .empty_projection()
                        .union_column(&query.column, OnMissing::Error)?;
                    let plan = self.take(fts_plan, projection)?;

                    self.flat_knn(plan, query)?
                }
                None => {
                    self.vector_search(&filter_plan.expr_filter_plan, query)
                        .await?
                }
            };

            filter_plan.disable_refine();
            Ok(source)
        } else {
            log::trace!("source is a vector search (postfilter)");
            // If we are postfiltering then we can't use scalar indices for the filter
            // and will need to run the postfilter in memory
            filter_plan.make_refine_only();
            self.vector_search(&ExprFilterPlan::default(), query).await
        }
    }

    async fn fragments_covered_by_fts_leaf(
        &self,
        column: &str,
        accum: &mut RoaringBitmap,
    ) -> Result<bool> {
        let index = self
            .dataset
            .load_scalar_index(IndexCriteria::default().for_column(column).supports_fts())
            .await?;
        match index {
            Some(index) => match &index.fragment_bitmap {
                Some(fragmap) => {
                    *accum |= fragmap;
                    Ok(true)
                }
                None => Ok(false),
            },
            None => Ok(false),
        }
    }

    #[async_recursion]
    async fn fragments_covered_by_fts_query_helper(
        &self,
        query: &FtsQuery,
        accum: &mut RoaringBitmap,
    ) -> Result<bool> {
        match query {
            FtsQuery::Match(match_query) => {
                self.fragments_covered_by_fts_leaf(
                    match_query.column.as_ref().ok_or(Error::invalid_input(
                        "the column must be specified in the query".to_string(),
                    ))?,
                    accum,
                )
                .await
            }
            FtsQuery::Boost(boost) => Ok(self
                .fragments_covered_by_fts_query_helper(&boost.negative, accum)
                .await?
                & self
                    .fragments_covered_by_fts_query_helper(&boost.positive, accum)
                    .await?),
            FtsQuery::MultiMatch(multi_match) => {
                for mq in &multi_match.match_queries {
                    if !self
                        .fragments_covered_by_fts_leaf(
                            mq.column.as_ref().ok_or(Error::invalid_input(
                                "the column must be specified in the query".to_string(),
                            ))?,
                            accum,
                        )
                        .await?
                    {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            FtsQuery::Phrase(phrase_query) => {
                self.fragments_covered_by_fts_leaf(
                    phrase_query.column.as_ref().ok_or(Error::invalid_input(
                        "the column must be specified in the query".to_string(),
                    ))?,
                    accum,
                )
                .await
            }
            FtsQuery::Boolean(bool_query) => {
                for query in bool_query.must.iter() {
                    if !self
                        .fragments_covered_by_fts_query_helper(query, accum)
                        .await?
                    {
                        return Ok(false);
                    }
                }
                for query in &bool_query.should {
                    if !self
                        .fragments_covered_by_fts_query_helper(query, accum)
                        .await?
                    {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    async fn fragments_covered_by_fts_query(&self, query: &FtsQuery) -> Result<RoaringBitmap> {
        let all_fragments = self.get_fragments_as_bitmap();

        let mut referenced_fragments = RoaringBitmap::new();
        if !self
            .fragments_covered_by_fts_query_helper(query, &mut referenced_fragments)
            .await?
        {
            // One or more indices is missing the fragment bitmap, require all fragments in prefilter
            Ok(all_fragments)
        } else {
            // Fragments required for prefilter is intersection of index fragments and query fragments
            Ok(all_fragments & referenced_fragments)
        }
    }

    // Create an execution plan to do full text search
    async fn fts(
        &self,
        filter_plan: &ExprFilterPlan,
        query: &FullTextSearchQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let columns = query.columns();
        let mut params = query.params();
        if params.limit.is_none() {
            let search_limit = match (self.limit, self.offset) {
                (Some(limit), Some(offset)) => Some((limit + offset) as usize),
                (Some(limit), None) => Some(limit as usize),
                (None, Some(_)) => None, // No limit but has offset - fetch all and let limit_node handle
                (None, None) => None,
            };
            params = params.with_limit(search_limit);
        }
        let query = if columns.is_empty() {
            // the field is not specified,
            // try to search over all indexed fields including nested ones
            let indexed_columns = fts_indexed_columns(self.dataset.clone()).await?;
            fill_fts_query_column(&query.query, &indexed_columns, false)?
        } else {
            query.query.clone()
        };

        // TODO: Could maybe walk the query here to find all the indices that will be
        // involved in the query to calculate a more accuarate required_fragments than
        // get_fragments_as_bitmap but this is safe for now.
        let prefilter_source = self
            .prefilter_source(
                filter_plan,
                self.fragments_covered_by_fts_query(&query).await?,
            )
            .await?;
        let fts_exec = self
            .plan_fts(&query, &params, filter_plan, &prefilter_source)
            .await?;
        Ok(fts_exec)
    }

    async fn plan_fts(
        &self,
        query: &FtsQuery,
        params: &FtsSearchParams,
        filter_plan: &ExprFilterPlan,
        prefilter_source: &PreFilterSource,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan: Arc<dyn ExecutionPlan> = match query {
            FtsQuery::Match(query) => {
                self.plan_match_query(query, params, filter_plan, prefilter_source)
                    .await?
            }
            FtsQuery::Phrase(query) => {
                self.plan_phrase_query(query, params, prefilter_source)
                    .await?
            }

            FtsQuery::Boost(query) => {
                // for boost query, we need to erase the limit so that we can find
                // the documents that are not in the top-k results of the positive query,
                // but in the final top-k results.
                let unlimited_params = params.clone().with_limit(None);
                let positive_exec = Box::pin(self.plan_fts(
                    &query.positive,
                    &unlimited_params,
                    filter_plan,
                    prefilter_source,
                ));
                let negative_exec = Box::pin(self.plan_fts(
                    &query.negative,
                    &unlimited_params,
                    filter_plan,
                    prefilter_source,
                ));
                let (positive_exec, negative_exec) =
                    futures::future::try_join(positive_exec, negative_exec).await?;
                Arc::new(BoostQueryExec::new(
                    query.clone(),
                    params.clone(),
                    positive_exec,
                    negative_exec,
                ))
            }

            FtsQuery::MultiMatch(query) => {
                let mut children = Vec::with_capacity(query.match_queries.len());
                for match_query in &query.match_queries {
                    let child =
                        self.plan_match_query(match_query, params, filter_plan, prefilter_source);
                    children.push(child);
                }
                let children = futures::future::try_join_all(children).await?;

                let schema = children[0].schema();
                let group_expr = vec![(
                    expressions::col(ROW_ID, schema.as_ref())?,
                    ROW_ID.to_string(),
                )];

                let fts_node = UnionExec::try_new(children)?;
                let fts_node = Arc::new(RepartitionExec::try_new(
                    fts_node,
                    Partitioning::RoundRobinBatch(1),
                )?);
                // dedup by row_id and return the max score as final score
                let fts_node = Arc::new(AggregateExec::try_new(
                    AggregateMode::Single,
                    PhysicalGroupBy::new_single(group_expr),
                    vec![Arc::new(
                        datafusion_physical_expr::aggregate::AggregateExprBuilder::new(
                            functions_aggregate::min_max::max_udaf(),
                            vec![expressions::col(SCORE_COL, &schema)?],
                        )
                        .schema(schema.clone())
                        .alias(SCORE_COL)
                        .build()?,
                    )],
                    vec![None],
                    fts_node,
                    schema,
                )?);
                let sort_expr = PhysicalSortExpr {
                    expr: expressions::col(SCORE_COL, fts_node.schema().as_ref())?,
                    options: SortOptions {
                        descending: true,
                        nulls_first: false,
                    },
                };

                Arc::new(
                    SortExec::new([sort_expr].into(), fts_node)
                        .with_fetch(self.limit.map(|l| l as usize)),
                )
            }
            FtsQuery::Boolean(query) => {
                // TODO: rewrite the query for better performance

                // we need to remove the limit from the params,
                // so that we won't miss possible matches
                let unlimited_params = params.clone().with_limit(None);

                let mut should = Vec::with_capacity(query.should.len());
                for subquery in &query.should {
                    should.push(
                        Box::pin(self.plan_fts(
                            subquery,
                            &unlimited_params,
                            filter_plan,
                            prefilter_source,
                        ))
                        .await?,
                    );
                }
                let mut must = Vec::with_capacity(query.must.len());
                for subquery in &query.must {
                    must.push(
                        Box::pin(self.plan_fts(
                            subquery,
                            &unlimited_params,
                            filter_plan,
                            prefilter_source,
                        ))
                        .await?,
                    );
                }
                let mut must_not = Vec::with_capacity(query.must_not.len());
                for subquery in &query.must_not {
                    must_not.push(
                        Box::pin(self.plan_fts(
                            subquery,
                            &unlimited_params,
                            filter_plan,
                            prefilter_source,
                        ))
                        .await?,
                    );
                }

                let should = build_boolean_query_children(BoolSlot::Should, should)?
                    .expect("Should slot always returns Some");
                let must = build_boolean_query_children(BoolSlot::Must, must)?;
                let must_not = build_boolean_query_children(BoolSlot::MustNot, must_not)?
                    .expect("MustNot slot always returns Some");

                if query.should.is_empty() && must.is_none() {
                    return Err(Error::invalid_input(
                        "boolean query must have at least one should/must query".to_string(),
                    ));
                }

                Arc::new(BooleanQueryExec::new(
                    query.clone(),
                    params.clone(),
                    should,
                    must,
                    must_not,
                ))
            }
        };

        Ok(plan)
    }

    async fn plan_phrase_query(
        &self,
        query: &PhraseQuery,
        params: &FtsSearchParams,
        prefilter_source: &PreFilterSource,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let column = query.column.clone().ok_or(Error::invalid_input(
            "the column must be specified in the query".to_string(),
        ))?;

        let segments = load_segments(&self.dataset, &column)
            .await?
            .ok_or(Error::invalid_input(format!(
                "No Inverted index found for column {}",
                column
            )))?;
        let details = load_segment_details(&self.dataset, &column, &segments).await?;

        if !details.with_position {
            return Err(Error::invalid_input("position is not found but required for phrase queries, try recreating the index with position"
                .to_string()));
        }

        Ok(Arc::new(PhraseQueryExec::new(
            self.dataset.clone(),
            query.clone(),
            params.clone(),
            prefilter_source.clone(),
        )))
    }

    async fn plan_match_query(
        &self,
        query: &MatchQuery,
        params: &FtsSearchParams,
        filter_plan: &ExprFilterPlan,
        prefilter_source: &PreFilterSource,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let column = query
            .column
            .as_ref()
            .ok_or(Error::invalid_input(
                "the column must be specified in the query".to_string(),
            ))?
            .clone();

        let index = self
            .dataset
            .load_scalar_index(IndexCriteria::default().for_column(&column).supports_fts())
            .await?;

        // Get target fragments
        let target_fragments = self
            .fragments
            .clone()
            .unwrap_or_else(|| self.dataset.fragments().to_vec());

        let (match_plan, flat_match_plan) = match &index {
            Some(index) => {
                // Get unindexed fragments and filter to target fragments
                let unindexed_fragments = self
                    .retain_target_fragments(self.dataset.unindexed_fragments(&index.name).await?);

                // If all target fragments are unindexed, skip index entirely
                if unindexed_fragments.len() == target_fragments.len() {
                    if self.fast_search {
                        return Ok(Arc::new(EmptyExec::new(FTS_SCHEMA.clone())));
                    }
                    let flat_match_plan = self
                        .plan_flat_match_query(unindexed_fragments, query, params, filter_plan)
                        .await?;
                    return Ok(flat_match_plan);
                }

                // Mixed case: use index + flat search for unindexed
                let match_plan: Arc<dyn ExecutionPlan> = Arc::new(MatchQueryExec::new(
                    self.dataset.clone(),
                    query.clone(),
                    params.clone(),
                    prefilter_source.clone(),
                ));

                if self.fast_search || unindexed_fragments.is_empty() {
                    (Some(match_plan), None)
                } else {
                    let flat_match_plan = self
                        .plan_flat_match_query(unindexed_fragments, query, params, filter_plan)
                        .await?;
                    (Some(match_plan), Some(flat_match_plan))
                }
            }
            None => {
                if self.fast_search {
                    return Ok(Arc::new(EmptyExec::new(FTS_SCHEMA.clone())));
                }
                // No index: flat search all target fragments
                let flat_match_plan = self
                    .plan_flat_match_query(target_fragments.clone(), query, params, filter_plan)
                    .await?;
                (None, Some(flat_match_plan))
            }
        };

        // Combine plans
        let plan = match (match_plan, flat_match_plan) {
            (Some(match_plan), Some(flat_match_plan)) => {
                let match_plan = UnionExec::try_new(vec![match_plan, flat_match_plan])?;
                let match_plan = Arc::new(RepartitionExec::try_new(
                    match_plan,
                    Partitioning::RoundRobinBatch(1),
                )?);
                let sort_expr = PhysicalSortExpr {
                    expr: expressions::col(SCORE_COL, match_plan.schema().as_ref())?,
                    options: SortOptions {
                        descending: true,
                        nulls_first: false,
                    },
                };
                Arc::new(SortExec::new([sort_expr].into(), match_plan).with_fetch(params.limit))
            }
            (Some(match_plan), None) => match_plan,
            (None, Some(flat_match_plan)) => flat_match_plan,
            (None, None) => unreachable!(),
        };

        Ok(plan)
    }

    /// Plan match query on unindexed fragments
    async fn plan_flat_match_query(
        &self,
        fragments: Vec<Fragment>,
        query: &MatchQuery,
        params: &FtsSearchParams,
        filter_plan: &ExprFilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let column = query
            .column
            .as_ref()
            .ok_or(Error::invalid_input(
                "the column must be specified in the query".to_string(),
            ))?
            .clone();

        let mut columns = vec![column];
        if let Some(expr) = filter_plan.full_expr.as_ref() {
            let filter_columns = Planner::column_names_in_expr(expr);
            columns.extend(filter_columns);
        }
        let flat_fts_scan_schema = Arc::new(self.dataset.schema().project(&columns).unwrap());
        let mut scan_node = self.scan_fragments(
            true,
            false,
            false,
            false,
            false,
            flat_fts_scan_schema,
            Arc::new(fragments),
            None,
            false,
        );

        if let Some(expr) = filter_plan.full_expr.as_ref() {
            // If there is a prefilter we need to manually apply it to the new data
            scan_node = Arc::new(LanceFilterExec::try_new(expr.clone(), scan_node)?);
        }

        let flat_match_plan = Arc::new(FlatMatchQueryExec::new(
            self.dataset.clone(),
            query.clone(),
            params.clone(),
            scan_node,
        ));
        Ok(flat_match_plan)
    }

    // ANN/KNN search execution node with optional prefilter
    #[async_recursion]
    async fn vector_search(
        &self,
        filter_plan: &ExprFilterPlan,
        q: &Query,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut q = q.clone();

        // Sanity check
        let (vector_type, element_type) = get_vector_type(self.dataset.schema(), &q.column)?;

        let column_id = self.dataset.schema().field_id(q.column.as_str())?;
        let use_index = q.use_index;
        let indices = if use_index {
            self.dataset.load_indices().await?
        } else {
            Arc::new(vec![])
        };
        let index_and_segments = if use_index {
            if let Some(requested_segments) = self.index_segments.as_ref() {
                let requested_segment_set =
                    requested_segments.iter().copied().collect::<HashSet<_>>();
                let requested_index_segments = indices
                    .iter()
                    .filter(|idx| requested_segment_set.contains(&idx.uuid))
                    .cloned()
                    .collect::<Vec<_>>();

                if requested_index_segments.len() != requested_segment_set.len() {
                    let found_segment_set = requested_index_segments
                        .iter()
                        .map(|idx| idx.uuid)
                        .collect::<HashSet<_>>();
                    let missing_segments = requested_segment_set
                        .difference(&found_segment_set)
                        .map(ToString::to_string)
                        .collect::<Vec<_>>();
                    return Err(Error::invalid_input(format!(
                        "with_index_segments referenced unknown index segments: {missing_segments:?}",
                    )));
                }

                if requested_index_segments
                    .iter()
                    .any(|idx| !idx.fields.contains(&column_id))
                {
                    return Err(Error::invalid_input(format!(
                        "with_index_segments contained a segment that does not belong to vector column '{}'",
                        q.column
                    )));
                }

                let index_name = requested_index_segments[0].name.clone();
                if requested_index_segments
                    .iter()
                    .any(|idx| idx.name != index_name)
                {
                    return Err(Error::invalid_input(
                        "with_index_segments must reference segments from a single logical index"
                            .to_string(),
                    ));
                }

                let selected_index_segments =
                    self.retain_relevant_index_segments(requested_index_segments);
                if selected_index_segments.is_empty() {
                    None
                } else {
                    let idx = self
                        .dataset
                        .open_vector_index(
                            q.column.as_str(),
                            &selected_index_segments[0].uuid,
                            &NoOpMetricsCollector,
                        )
                        .await?;
                    let index_metric = idx.metric_type();
                    let use_this_index = match q.metric_type {
                        Some(user_metric) => {
                            if user_metric == index_metric {
                                true
                            } else {
                                return Err(Error::invalid_input(format!(
                                    "with_index_segments requested metric {:?} but the selected index segments use {:?}",
                                    user_metric, index_metric
                                )));
                            }
                        }
                        None => true,
                    };
                    if use_this_index {
                        Some((index_name, selected_index_segments, index_metric))
                    } else {
                        None
                    }
                }
            } else if let Some(index) = indices.iter().find(|i| i.fields.contains(&column_id)) {
                // Try to get metric type from index metadata first (fast path for newer indices)
                let index_metric = if let Some(metric) =
                    crate::index::vector::details::metric_type_from_index_metadata(index)
                {
                    metric
                } else {
                    // Fall back to opening the index for legacy indices without details
                    let idx = self
                        .dataset
                        .open_vector_index(q.column.as_str(), &index.uuid, &NoOpMetricsCollector)
                        .await?;
                    idx.metric_type()
                };

                let use_this_index = match q.metric_type {
                    Some(user_metric) => {
                        if user_metric == index_metric {
                            true
                        } else {
                            log::warn!(
                                "Requested metric {:?} is incompatible with index metric {:?}, falling back to brute-force search",
                                user_metric,
                                index_metric
                            );
                            false
                        }
                    }
                    None => true,
                };

                if use_this_index {
                    let index_segments = self.retain_relevant_index_segments(
                        self.dataset.load_indices_by_name(&index.name).await?,
                    );
                    let index_frags = self.get_indexed_frags(&index_segments);
                    if !index_segments.is_empty() && !index_frags.is_empty() {
                        Some((index.name.clone(), index_segments, index_metric))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some((index_name, index_segments, index_metric)) = index_and_segments {
            if self.is_batch_nearest {
                return self.batch_indexed_vector_search(filter_plan, &q).await;
            }

            log::trace!("index found for vector search");
            // Use the index's metric type
            q.metric_type = Some(index_metric);
            validate_distance_type_for(index_metric, &element_type)?;

            if matches!(q.refine_factor, Some(0)) {
                return Err(Error::invalid_input(
                    "Refine factor cannot be zero".to_string(),
                ));
            }
            let ann_node = match vector_type {
                DataType::FixedSizeList(_, _) => self.ann(&q, &index_segments, filter_plan).await?,
                DataType::List(_) => self.multivec_ann(&q, &index_segments, filter_plan).await?,
                _ => unreachable!(),
            };

            let mut knn_node = if q.refine_factor.is_some() {
                let vector_projection = self
                    .dataset
                    .empty_projection()
                    .union_column(&q.column, OnMissing::Error)
                    .unwrap();
                let knn_node_with_vector = self.take(ann_node, vector_projection)?;
                self.flat_knn(knn_node_with_vector, &q)?
            } else {
                ann_node
            }; // vector, _distance, _rowid

            if !self.fast_search {
                knn_node = self
                    .knn_combined(&q, &index_name, &index_segments, knn_node, filter_plan)
                    .await?;
            }

            Ok(knn_node)
        } else {
            if self.fast_search {
                return Ok(Arc::new(EmptyExec::new(knn_empty_result_schema(
                    self.is_batch_nearest,
                ))));
            }
            // Resolve metric type for flat search (use default if not specified)
            let metric = q
                .metric_type
                .unwrap_or_else(|| default_distance_type_for(&element_type));
            q.metric_type = Some(metric);
            validate_distance_type_for(metric, &element_type)?;
            // No index found. use flat search.
            let mut columns = vec![q.column.clone()];
            if let Some(refine_expr) = filter_plan.refine_expr.as_ref() {
                columns.extend(Planner::column_names_in_expr(refine_expr));
            }
            let mut vector_scan_projection = self
                .dataset
                .empty_projection()
                .with_row_id()
                .union_columns(&columns, OnMissing::Error)?;

            vector_scan_projection.with_row_addr =
                self.projection_plan.physical_projection.with_row_addr;

            let PlannedFilteredScan { mut plan, .. } = self
                .filtered_read(
                    filter_plan,
                    vector_scan_projection,
                    /*include_deleted_rows=*/ true,
                    self.fragments.clone().map(Arc::new),
                    None,
                    /*is_prefilter= */ true,
                )
                .await?;

            if let Some(refine_expr) = &filter_plan.refine_expr {
                plan = Arc::new(LanceFilterExec::try_new(refine_expr.clone(), plan)?);
            }
            Ok(self.flat_knn(plan, &q)?)
        }
    }

    async fn batch_indexed_vector_search(
        &self,
        filter_plan: &ExprFilterPlan,
        q: &Query,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let query_dim = q.key.len() / self.nearest_query_count;
        let mut query_plans = Vec::with_capacity(self.nearest_query_count);

        for query_index in 0..self.nearest_query_count {
            let mut single_query = q.clone();
            single_query.key = q.key.slice(query_index * query_dim, query_dim);

            let mut single_scanner = self.clone();
            single_scanner.nearest_query_count = 1;
            single_scanner.is_batch_nearest = false;
            single_scanner.nearest = Some(single_query.clone());

            let single_plan = single_scanner
                .vector_search(filter_plan, &single_query)
                .await?;
            query_plans.push(Self::add_query_index_column(
                single_plan,
                query_index as i32,
            )?);
        }

        let unioned = UnionExec::try_new(query_plans)?;
        let unioned = Arc::new(RepartitionExec::try_new(
            unioned,
            Partitioning::RoundRobinBatch(1),
        )?) as Arc<dyn ExecutionPlan>;

        let query_index_sort = PhysicalSortExpr {
            expr: expressions::col(QUERY_INDEX_COL, unioned.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        let distance_sort = PhysicalSortExpr {
            expr: expressions::col(DIST_COL, unioned.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        let row_id_sort = PhysicalSortExpr {
            expr: expressions::col(ROW_ID, unioned.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };

        Ok(Arc::new(SortExec::new(
            [query_index_sort, distance_sort, row_id_sort].into(),
            unioned,
        )))
    }

    fn add_query_index_column(
        plan: Arc<dyn ExecutionPlan>,
        query_index: i32,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = plan.schema();
        let mut projection_exprs = Vec::with_capacity(schema.fields().len() + 1);
        projection_exprs.push((
            Arc::new(Literal::new(ScalarValue::Int32(Some(query_index)))) as Arc<dyn PhysicalExpr>,
            QUERY_INDEX_COL.to_string(),
        ));
        for field in schema.fields() {
            projection_exprs.push((
                Arc::new(Column::new_with_schema(field.name(), schema.as_ref())?)
                    as Arc<dyn PhysicalExpr>,
                field.name().clone(),
            ));
        }
        Ok(Arc::new(ProjectionExec::try_new(projection_exprs, plan)?))
    }

    /// Combine ANN results with KNN results for data appended after index creation
    async fn knn_combined(
        &self,
        q: &Query,
        index_name: &str,
        indexed_segments: &[IndexMetadata],
        mut knn_node: Arc<dyn ExecutionPlan>,
        filter_plan: &ExprFilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let fallback_fragments = if let Some(target_fragments) = &self.fragments {
            let indexed_fragments = self.get_indexed_frags(indexed_segments);
            target_fragments
                .iter()
                .filter(|fragment| !indexed_fragments.contains(fragment.id as u32))
                .cloned()
                .collect::<Vec<_>>()
        } else if self.index_segments.is_some() {
            Vec::new()
        } else {
            self.dataset.unindexed_fragments(index_name).await?
        };

        if !fallback_fragments.is_empty() {
            let q = q.clone();
            debug_assert!(q.metric_type.is_some());

            // If the vector column is not present, we need to take the vector column, so
            // that the distance value is comparable with the flat search ones.
            if knn_node.schema().column_with_name(&q.column).is_none() {
                let vector_projection = self
                    .dataset
                    .empty_projection()
                    .union_column(&q.column, OnMissing::Error)
                    .unwrap();
                knn_node = self.take(knn_node, vector_projection)?;
            }

            let mut columns = vec![q.column.clone()];
            if let Some(expr) = filter_plan.full_expr.as_ref() {
                let filter_columns = Planner::column_names_in_expr(expr);
                columns.extend(filter_columns);
            }
            let vector_scan_projection = Arc::new(self.dataset.schema().project(&columns).unwrap());
            // Note: we could try and use the scalar indices here to reduce the scope of this scan but the
            // most common case is that fragments that are newer than the vector index are going to be newer
            // than the scalar indices anyways
            let mut scan_node = self.scan_fragments(
                true,
                false,
                false,
                false,
                false,
                vector_scan_projection,
                Arc::new(fallback_fragments),
                // Can't pushdown limit/offset in an ANN search
                None,
                // We are re-ordering anyways, so no need to get data in data
                // in a deterministic order.
                false,
            );

            if let Some(expr) = filter_plan.full_expr.as_ref() {
                // If there is a prefilter we need to manually apply it to the new data
                scan_node = Arc::new(LanceFilterExec::try_new(expr.clone(), scan_node)?);
            }
            // first we do flat search on just the new data
            let topk_appended = self.flat_knn(scan_node, &q)?;

            // To do a union, we need to make the schemas match. Right now
            // knn_node: _distance, _rowid, vector
            // topk_appended: vector, <filter columns?>, _rowid, _distance
            let topk_appended = project(topk_appended, knn_node.schema().as_ref())?;
            assert!(
                topk_appended
                    .schema()
                    .equivalent_names_and_types(&knn_node.schema())
            );
            // union
            let unioned = UnionExec::try_new(vec![Arc::new(topk_appended), knn_node])?;
            // Enforce only 1 partition.
            let unioned = RepartitionExec::try_new(
                unioned,
                datafusion::physical_plan::Partitioning::RoundRobinBatch(1),
            )?;
            // then we do a flat search on KNN(new data) + ANN(indexed data)
            return self.flat_knn(Arc::new(unioned), &q);
        }

        Ok(knn_node)
    }

    #[async_recursion]
    async fn fragments_covered_by_index_query(
        &self,
        index_expr: &ScalarIndexExpr,
    ) -> Result<RoaringBitmap> {
        match index_expr {
            ScalarIndexExpr::And(lhs, rhs) => {
                Ok(self.fragments_covered_by_index_query(lhs).await?
                    & self.fragments_covered_by_index_query(rhs).await?)
            }
            ScalarIndexExpr::Or(lhs, rhs) => Ok(self.fragments_covered_by_index_query(lhs).await?
                & self.fragments_covered_by_index_query(rhs).await?),
            ScalarIndexExpr::Not(expr) => self.fragments_covered_by_index_query(expr).await,
            ScalarIndexExpr::Query(search) => scalar_index_fragment_bitmap(
                self.dataset.as_ref(),
                &search.column,
                &search.index_name,
            )
            .await?
            .ok_or_else(|| {
                crate::Error::internal(format!(
                    "Index not found even though it must have been found earlier: {}",
                    search.index_name
                ))
            }),
        }
    }

    /// Given an index query, split the fragments into two sets
    ///
    /// The first set is the relevant fragments, which are covered by ALL indices in the query
    /// The second set is the missing fragments, which are missed by at least one index
    ///
    /// There is no point in handling the case where a fragment is covered by some (but not all)
    /// of the indices.  If we have to do a full scan of the fragment then we do it
    async fn partition_frags_by_coverage(
        &self,
        index_expr: &ScalarIndexExpr,
        fragments: Arc<Vec<Fragment>>,
    ) -> Result<(Vec<Fragment>, Vec<Fragment>)> {
        let covered_frags = self.fragments_covered_by_index_query(index_expr).await?;
        let mut relevant_frags = Vec::with_capacity(fragments.len());
        let mut missing_frags = Vec::with_capacity(fragments.len());
        for fragment in fragments.iter() {
            if covered_frags.contains(fragment.id as u32) {
                relevant_frags.push(fragment.clone());
            } else {
                missing_frags.push(fragment.clone());
            }
        }
        Ok((relevant_frags, missing_frags))
    }

    // First perform a lookup in a scalar index for ids and then perform a take on the
    // target fragments with those ids
    async fn scalar_indexed_scan(
        &self,
        projection: Projection,
        filter_plan: &ExprFilterPlan,
        fragments: Arc<Vec<Fragment>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        log::trace!("scalar indexed scan");
        // One or more scalar indices cover this data and there is a filter which is
        // compatible with the indices.  Use that filter to perform a take instead of
        // a full scan.

        // If this unwrap fails we have a bug because we shouldn't be using this function unless we've already
        // checked that there is an index query
        let index_expr = filter_plan.index_query.as_ref().unwrap();

        let needs_recheck = index_expr.needs_recheck();

        // Figure out which fragments are covered by ALL indices
        let (relevant_frags, missing_frags) = self
            .partition_frags_by_coverage(index_expr, fragments)
            .await?;

        let mut plan: Arc<dyn ExecutionPlan> = Arc::new(MaterializeIndexExec::new(
            self.dataset.clone(),
            index_expr.clone(),
            Arc::new(relevant_frags),
        ));

        let refine_expr = filter_plan.refine_expr.as_ref();

        // If all we want is the row ids then we can skip the take.  However, if there is a refine
        // or a recheck then we still need to do a take because we need filter columns.
        let needs_take =
            needs_recheck || projection.has_data_fields() || filter_plan.refine_expr.is_some();
        if needs_take {
            let mut take_projection = projection.clone();
            if needs_recheck {
                // If we need to recheck then we need to also take the columns used for the filter
                let filter_expr = index_expr.to_expr();
                let filter_cols = Planner::column_names_in_expr(&filter_expr);
                take_projection = take_projection.union_columns(filter_cols, OnMissing::Error)?;
            }
            if let Some(refine_expr) = refine_expr {
                let refine_cols = Planner::column_names_in_expr(refine_expr);
                take_projection = take_projection.union_columns(refine_cols, OnMissing::Error)?;
            }
            log::trace!("need to take additional columns for scalar_indexed_scan");
            plan = self.take(plan, take_projection)?;
        }

        let post_take_filter = match (needs_recheck, refine_expr) {
            (false, None) => None,
            (true, None) => {
                // If we need to recheck then we need to apply the filter to the results
                Some(index_expr.to_expr())
            }
            (true, Some(_)) => Some(filter_plan.full_expr.as_ref().unwrap().clone()),
            (false, Some(refine_expr)) => Some(refine_expr.clone()),
        };

        if let Some(post_take_filter) = post_take_filter {
            let planner = Planner::new(plan.schema());
            let optimized_filter = planner.optimize_expr(post_take_filter)?;

            log::trace!("applying post-take filter to indexed scan");
            plan = Arc::new(LanceFilterExec::try_new(optimized_filter, plan)?);
        }

        if self.projection_plan.physical_projection.with_row_addr {
            plan = Arc::new(AddRowAddrExec::try_new(plan, self.dataset.clone(), 0)?);
        }

        let new_data_path: Option<Arc<dyn ExecutionPlan>> = if !missing_frags.is_empty() {
            log::trace!(
                "scalar_indexed_scan will need full scan of {} missing fragments",
                missing_frags.len()
            );

            // If there is new data then we need this:
            //
            // MaterializeIndexExec(old_frags) -> Take -> Union
            // Scan(new_frags) -> Filter -> Project    -|
            //
            // The project is to drop any columns we had to include
            // in the full scan merely for the sake of fulfilling the
            // filter.
            //
            // If there were no extra columns then we still need the project
            // because Materialize -> Take puts the row id at the left and
            // Scan puts the row id at the right
            let filter = filter_plan.full_expr.as_ref().unwrap();
            let filter_cols = Planner::column_names_in_expr(filter);
            let scan_projection = projection.union_columns(filter_cols, OnMissing::Error)?;

            let scan_schema = Arc::new(scan_projection.to_bare_schema());
            let scan_arrow_schema = Arc::new(scan_schema.as_ref().into());
            let planner = Planner::new(scan_arrow_schema);
            let optimized_filter = planner.optimize_expr(filter.clone())?;

            let new_data_scan = self.scan_fragments(
                true,
                self.projection_plan.physical_projection.with_row_addr,
                self.projection_plan
                    .physical_projection
                    .with_row_last_updated_at_version,
                self.projection_plan
                    .physical_projection
                    .with_row_created_at_version,
                false,
                scan_schema,
                missing_frags.into(),
                // No pushdown of limit/offset when doing scalar indexed scan
                None,
                false,
            );
            let filtered = Arc::new(LanceFilterExec::try_new(optimized_filter, new_data_scan)?);
            Some(Arc::new(project(filtered, plan.schema().as_ref())?))
        } else {
            log::trace!("scalar_indexed_scan will not need full scan of any missing fragments");
            None
        };

        if let Some(new_data_path) = new_data_path {
            let unioned = UnionExec::try_new(vec![plan, new_data_path])?;
            // Enforce only 1 partition.
            let unioned = Arc::new(RepartitionExec::try_new(
                unioned,
                datafusion::physical_plan::Partitioning::RoundRobinBatch(1),
            )?);
            Ok(unioned)
        } else {
            Ok(plan)
        }
    }

    fn get_io_buffer_size(&self) -> u64 {
        self.io_buffer_size.unwrap_or(*DEFAULT_IO_BUFFER_SIZE)
    }

    /// Create an Execution plan with a scan node
    ///
    /// Setting `with_make_deletions_null` will use the validity of the _rowid
    /// column as a selection vector. Read more in [crate::io::FileReader].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scan(
        &self,
        with_row_id: bool,
        with_row_address: bool,
        with_row_last_updated_at_version: bool,
        with_row_created_at_version: bool,
        with_make_deletions_null: bool,
        range: Option<Range<u64>>,
        projection: Arc<Schema>,
    ) -> Arc<dyn ExecutionPlan> {
        let fragments = if let Some(fragment) = self.fragments.as_ref() {
            Arc::new(fragment.clone())
        } else {
            self.dataset.fragments().clone()
        };
        let ordered = if self.ordering.is_some() || self.nearest.is_some() {
            // If we are sorting the results there is no need to scan in order
            false
        } else {
            self.ordered
        };
        self.scan_fragments(
            with_row_id,
            with_row_address,
            with_row_last_updated_at_version,
            with_row_created_at_version,
            with_make_deletions_null,
            projection,
            fragments,
            range,
            ordered,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_fragments(
        &self,
        with_row_id: bool,
        with_row_address: bool,
        with_row_last_updated_at_version: bool,
        with_row_created_at_version: bool,
        with_make_deletions_null: bool,
        projection: Arc<Schema>,
        fragments: Arc<Vec<Fragment>>,
        range: Option<Range<u64>>,
        ordered: bool,
    ) -> Arc<dyn ExecutionPlan> {
        log::trace!("scan_fragments covered {} fragments", fragments.len());
        let config = LanceScanConfig {
            batch_size: self.get_batch_size(),
            batch_readahead: self.batch_readahead,
            fragment_readahead: self.fragment_readahead,
            io_buffer_size: self.get_io_buffer_size(),
            with_row_id,
            with_row_address,
            with_row_last_updated_at_version,
            with_row_created_at_version,
            with_make_deletions_null,
            ordered_output: ordered,
            file_reader_options: self.resolved_file_reader_options(),
            parallelism_cap: None,
        };
        Arc::new(LanceScanExec::new(
            self.dataset.clone(),
            fragments,
            range,
            projection,
            config,
        ))
    }

    fn pushdown_scan(
        &self,
        make_deletions_null: bool,
        filter_plan: &ExprFilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        log::trace!("pushdown_scan");

        let config = ScanConfig {
            batch_readahead: self.batch_readahead,
            fragment_readahead: self
                .fragment_readahead
                .unwrap_or(LEGACY_DEFAULT_FRAGMENT_READAHEAD),
            with_row_id: self.projection_plan.physical_projection.with_row_id,
            with_row_address: self.projection_plan.physical_projection.with_row_addr,
            make_deletions_null,
            ordered_output: self.ordered,
            file_reader_options: self.resolved_file_reader_options(),
        };

        let fragments = if let Some(fragment) = self.fragments.as_ref() {
            Arc::new(fragment.clone())
        } else {
            self.dataset.fragments().clone()
        };

        Ok(Arc::new(LancePushdownScanExec::try_new(
            self.dataset.clone(),
            fragments,
            Arc::new(self.projection_plan.physical_projection.to_bare_schema()),
            filter_plan.refine_expr.clone().unwrap(),
            config,
        )?))
    }

    /// Here we use a full text search as a post-filter.  Any rows that
    /// do not contain at least one query token are removed.
    ///
    /// Only valid (currently) for match queries.
    async fn flat_fts_filter(
        &self,
        input: Arc<dyn ExecutionPlan>,
        q: &FullTextSearchQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let fts_query = if q.columns().is_empty() {
            let indexed_columns = fts_indexed_columns(self.dataset.clone()).await?;
            fill_fts_query_column(&q.query, &indexed_columns, false)?
        } else {
            q.query.clone()
        };

        match &fts_query {
            FtsQuery::Match(match_query) => {
                let schema = Arc::new((input.schema()).try_with_column(SCORE_FIELD.clone())?);

                let column = match_query
                    .column
                    .as_ref()
                    .ok_or(Error::invalid_input(
                        "the column must be specified in the query".to_string(),
                    ))?
                    .clone();
                let input = if schema.column_with_name(&column).is_none() {
                    let projection = self
                        .dataset
                        .empty_projection()
                        .union_column(&column, OnMissing::Error)?;
                    self.take(input, projection)?
                } else {
                    input
                };

                Ok(Arc::new(FlatMatchFilterExec::new(
                    input,
                    self.dataset.clone(),
                    match_query.clone(),
                    q.params(),
                )))
            }
            _ => Err(Error::not_supported(
                "Only Match queries are supported currently when using FTS as a post-filter",
            )),
        }
    }

    /// Here we consume all input (as unindexed) and rerank according to BM25 scores
    ///
    /// If there is an index on the column then we still use the index to determine the
    /// tokenizer and inform the BM25 scoring (e.g. avg doc length, token frequency, etc.)
    async fn fts_rerank(
        &self,
        input: Arc<dyn ExecutionPlan>,
        q: &FullTextSearchQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let fts_query = if q.columns().is_empty() {
            let indexed_columns = fts_indexed_columns(self.dataset.clone()).await?;
            fill_fts_query_column(&q.query, &indexed_columns, false)?
        } else {
            q.query.clone()
        };

        match &fts_query {
            FtsQuery::Match(match_query) => {
                let schema = Arc::new((input.schema()).try_with_column(SCORE_FIELD.clone())?);

                let column = match_query
                    .column
                    .as_ref()
                    .ok_or(Error::invalid_input(
                        "the column must be specified in the query".to_string(),
                    ))?
                    .clone();
                let input = if schema.column_with_name(&column).is_none() {
                    let projection = self
                        .dataset
                        .empty_projection()
                        .union_column(&column, OnMissing::Error)?;
                    self.take(input, projection)?
                } else {
                    input
                };

                Ok(Arc::new(FlatMatchQueryExec::new(
                    self.dataset.clone(),
                    match_query.clone(),
                    q.params(),
                    input,
                )))
            }
            _ => {
                let default_filter = ExprFilterPlan::default();
                let fts_plan = self.fts(&default_filter, q).await?;

                let vector_row_id = Column::new_with_schema(ROW_ID, input.schema().as_ref())?;
                let fts_row_id = Column::new_with_schema(ROW_ID, fts_plan.schema().as_ref())?;
                let join = HashJoinExec::try_new(
                    input,
                    fts_plan,
                    vec![(Arc::new(vector_row_id), Arc::new(fts_row_id))],
                    None,
                    &JoinType::Inner,
                    None,
                    PartitionMode::CollectLeft,
                    NullEquality::NullEqualsNull,
                    false,
                )?;

                let schema = join.schema();
                let mut projection_exprs = Vec::new();
                let mut contain_rowid = false;
                for field in schema.fields() {
                    if field.name() == ROW_ID {
                        if contain_rowid {
                            continue;
                        }
                        contain_rowid = true;
                    }
                    projection_exprs.push((
                        Arc::new(Column::new_with_schema(field.name(), schema.as_ref())?)
                            as Arc<dyn PhysicalExpr>,
                        field.name().clone(),
                    ));
                }

                let projection_exec = ProjectionExec::try_new(projection_exprs, Arc::new(join))?;
                Ok(Arc::new(projection_exec))
            }
        }
    }

    /// Add a knn search node to the input plan
    fn flat_knn(&self, input: Arc<dyn ExecutionPlan>, q: &Query) -> Result<Arc<dyn ExecutionPlan>> {
        // Resolve metric_type if not set (use default for the column's element type)
        let metric_type = match q.metric_type {
            Some(m) => m,
            None => {
                let (_, element_type) = get_vector_type(self.dataset.schema(), &q.column)?;
                default_distance_type_for(&element_type)
            }
        };
        let input = if self.is_batch_nearest {
            Arc::new(CoalescePartitionsExec::new(input)) as Arc<dyn ExecutionPlan>
        } else {
            input
        };
        let flat_dist = Arc::new(KNNVectorDistanceExec::try_new_batch(
            input,
            &q.column,
            q.key.clone(),
            KnnBatchParams {
                is_batch: self.is_batch_nearest,
                query_count: self.nearest_query_count,
                k: q.k,
                lower_bound: q.lower_bound,
                upper_bound: q.upper_bound,
                distance_type: metric_type,
            },
        )?);

        if self.is_batch_nearest {
            return Ok(flat_dist);
        }

        let lower: Option<(Expr, Arc<dyn PhysicalExpr>)> = q
            .lower_bound
            .map(|v| -> Result<(Expr, Arc<dyn PhysicalExpr>)> {
                let logical = col(DIST_COL).gt_eq(lit(v));
                let schema = flat_dist.schema();
                let df_schema = DFSchema::try_from(schema)?;
                let physical = create_physical_expr(&logical, &df_schema, &ExecutionProps::new())?;
                Ok::<(Expr, Arc<dyn PhysicalExpr>), _>((logical, physical))
            })
            .transpose()?;

        let upper = q
            .upper_bound
            .map(|v| -> Result<(Expr, Arc<dyn PhysicalExpr>)> {
                let logical = col(DIST_COL).lt(lit(v));
                let schema = flat_dist.schema();
                let df_schema = DFSchema::try_from(schema)?;
                let physical = create_physical_expr(&logical, &df_schema, &ExecutionProps::new())?;
                Ok::<(Expr, Arc<dyn PhysicalExpr>), _>((logical, physical))
            })
            .transpose()?;

        let filter_expr = match (lower, upper) {
            (Some((llog, _)), Some((ulog, _))) => {
                let logical = llog.and(ulog);
                let schema = flat_dist.schema();
                let df_schema = DFSchema::try_from(schema)?;
                let physical = create_physical_expr(&logical, &df_schema, &ExecutionProps::new())?;
                Some((logical, physical))
            }
            (Some((llog, lphys)), None) => Some((llog, lphys)),
            (None, Some((ulog, uphys))) => Some((ulog, uphys)),
            (None, None) => None,
        };

        let knn_plan: Arc<dyn ExecutionPlan> = if let Some(filter_expr) = filter_expr {
            Arc::new(LanceFilterExec::try_new(filter_expr.0, flat_dist)?)
        } else {
            flat_dist
        };

        // Use DataFusion's [SortExec] for Top-K search
        let sort = SortExec::new(
            [
                PhysicalSortExpr {
                    expr: expressions::col(DIST_COL, knn_plan.schema().as_ref())?,
                    options: SortOptions {
                        descending: false,
                        nulls_first: false,
                    },
                },
                PhysicalSortExpr {
                    expr: expressions::col(ROW_ID, knn_plan.schema().as_ref())?,
                    options: SortOptions {
                        descending: false,
                        nulls_first: false,
                    },
                },
            ]
            .into(),
            knn_plan,
        )
        .with_fetch(Some(q.k));

        Self::flat_knn_not_null_filter(Arc::new(sort))
    }

    fn flat_knn_not_null_filter(
        knn_plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let logical_not_null = col(DIST_COL).is_not_null();
        Ok(Arc::new(LanceFilterExec::try_new(
            logical_not_null,
            knn_plan,
        )?))
    }

    fn get_fragments_as_bitmap(&self) -> RoaringBitmap {
        if let Some(fragments) = &self.fragments {
            RoaringBitmap::from_iter(fragments.iter().map(|f| f.id as u32))
        } else {
            self.dataset.fragment_bitmap.as_ref().clone()
        }
    }

    fn retain_relevant_index_segments(
        &self,
        index_segments: Vec<IndexMetadata>,
    ) -> Vec<IndexMetadata> {
        if let Some(fragments) = &self.fragments {
            let target_fragments = RoaringBitmap::from_iter(fragments.iter().map(|f| f.id as u32));
            index_segments
                .into_iter()
                .filter(|idx| {
                    idx.fragment_bitmap
                        .as_ref()
                        .is_some_and(|fragmap| !(fragmap & &target_fragments).is_empty())
                })
                .collect()
        } else {
            index_segments
        }
    }

    /// Retain only fragments that are in the user-specified fragment list.
    /// If no fragment list is specified, returns the fragments unchanged.
    fn retain_target_fragments(&self, mut fragments: Vec<Fragment>) -> Vec<Fragment> {
        if let Some(target) = &self.fragments {
            let bitmap = RoaringBitmap::from_iter(target.iter().map(|f| f.id as u32));
            fragments.retain(|f| bitmap.contains(f.id as u32));
        }
        fragments
    }

    fn get_indexed_frags(&self, index: &[IndexMetadata]) -> RoaringBitmap {
        let all_fragments = self.get_fragments_as_bitmap();

        let mut all_indexed_frags = RoaringBitmap::new();
        for idx in index {
            if let Some(fragmap) = idx.fragment_bitmap.as_ref() {
                all_indexed_frags |= fragmap;
            } else {
                // If any index is missing the fragment bitmap it is safest to just assume we
                // need all fragments
                return all_fragments;
            }
        }

        all_indexed_frags & all_fragments
    }

    /// Create an Execution plan to do indexed ANN search
    async fn ann(
        &self,
        q: &Query,
        index: &[IndexMetadata],
        filter_plan: &ExprFilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let prefilter_source = self
            .prefilter_source(filter_plan, self.get_indexed_frags(index))
            .await?;
        let inner_fanout_search = new_knn_exec(self.dataset.clone(), index, q, prefilter_source)?;
        let sort_expr = PhysicalSortExpr {
            expr: expressions::col(DIST_COL, inner_fanout_search.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        let sort_expr_row_id = PhysicalSortExpr {
            expr: expressions::col(ROW_ID, inner_fanout_search.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        Ok(Arc::new(
            SortExec::new([sort_expr, sort_expr_row_id].into(), inner_fanout_search)
                .with_fetch(Some(q.k * q.refine_factor.unwrap_or(1) as usize)),
        ))
    }

    // Create an Execution plan to do ANN over multivectors
    async fn multivec_ann(
        &self,
        q: &Query,
        index: &[IndexMetadata],
        filter_plan: &ExprFilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // we split the query procedure into two steps:
        // 1. collect the candidates by vector searching on each query vector
        // 2. scoring the candidates

        let over_fetch_factor = *DEFAULT_XTR_OVERFETCH;

        let prefilter_source = self
            .prefilter_source(filter_plan, self.get_indexed_frags(index))
            .await?;
        let dim = get_vector_dim(self.dataset.schema(), &q.column)?;

        let num_queries = q.key.len() / dim;
        let new_queries = (0..num_queries)
            .map(|i| q.key.slice(i * dim, dim))
            .map(|query_vec| {
                let mut new_query = q.clone();
                new_query.key = query_vec;
                // with XTR, we don't need to refine the result with original vectors,
                // but here we really need to over-fetch the candidates to reach good enough recall.
                // TODO: improve the recall with WARP, expose this parameter to the users.
                new_query.refine_factor = Some(over_fetch_factor);
                new_query
            });
        let mut ann_nodes = Vec::with_capacity(new_queries.len());
        for query in new_queries {
            // this produces `nprobes * k * over_fetch_factor * num_indices` candidates
            let ann_node = new_knn_exec(
                self.dataset.clone(),
                index,
                &query,
                prefilter_source.clone(),
            )?;
            let sort_expr = PhysicalSortExpr {
                expr: expressions::col(DIST_COL, ann_node.schema().as_ref())?,
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            };
            let sort_expr_row_id = PhysicalSortExpr {
                expr: expressions::col(ROW_ID, ann_node.schema().as_ref())?,
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            };
            let ann_node = Arc::new(
                SortExec::new([sort_expr, sort_expr_row_id].into(), ann_node)
                    .with_fetch(Some(q.k * over_fetch_factor as usize)),
            );
            ann_nodes.push(ann_node as Arc<dyn ExecutionPlan>);
        }

        let ann_node = Arc::new(MultivectorScoringExec::try_new(ann_nodes, q.clone())?);

        let sort_expr = PhysicalSortExpr {
            expr: expressions::col(DIST_COL, ann_node.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        let sort_expr_row_id = PhysicalSortExpr {
            expr: expressions::col(ROW_ID, ann_node.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        let ann_node = Arc::new(
            SortExec::new([sort_expr, sort_expr_row_id].into(), ann_node)
                .with_fetch(Some(q.k * q.refine_factor.unwrap_or(1) as usize)),
        );

        Ok(ann_node)
    }

    /// Create prefilter source from filter plan
    ///
    /// A prefilter is an input to a vector or fts search.  It tells us which rows are eligible
    /// for the search.  A prefilter is calculated by doing a filtered read of the row id column.
    async fn prefilter_source(
        &self,
        filter_plan: &ExprFilterPlan,
        required_frags: RoaringBitmap,
    ) -> Result<PreFilterSource> {
        if filter_plan.is_empty() && self.fragments.is_none() {
            log::trace!("no filter plan, no prefilter");
            return Ok(PreFilterSource::None);
        }

        // get fragments covered by index
        let fragments: Vec<Fragment> = self
            .dataset
            .manifest
            .fragments
            .iter()
            .filter(|f| required_frags.contains(f.id as u32))
            .cloned()
            .collect();

        // If explicitly specified fragments with .with_fragments(), intersect with those
        let fragments = Arc::new(self.retain_target_fragments(fragments));

        // Can only use ScalarIndexExec when the scalar index is exact and we are not scanning
        // a subset of the fragments.
        //
        // TODO: We could enhance ScalarIndexExec with a fragment bitmap to filter out rows that
        // are not in the fragments we are scanning.
        if filter_plan.is_exact_index_search() && self.fragments.is_none() {
            let index_query = filter_plan.index_query.as_ref().expect_ok()?;
            let (_, missing_frags) = self
                .partition_frags_by_coverage(index_query, fragments.clone())
                .await?;

            if missing_frags.is_empty() || self.fast_search {
                log::trace!("prefilter entirely satisfied by exact index search");
                let result_format = self.index_expr_result_format();
                // We can only avoid materializing the index for a prefilter if:
                // 1. The search is indexed
                // 2. The index search is an exact search with no recheck or refine
                // 3. The indices cover at least the same fragments as the vector index,
                //    unless fast_search allows skipping uncovered fragments.
                return Ok(PreFilterSource::ScalarIndexQuery(Arc::new(
                    ScalarIndexExec::new(self.dataset.clone(), index_query.clone(), result_format),
                )));
            } else {
                log::trace!("exact index search did not cover all fragments");
            }
        }

        // If one of our criteria is not met, we need to do a filtered read of just the row id column
        log::trace!(
            "prefilter is a filtered read of {} fragments",
            fragments.len()
        );
        let PlannedFilteredScan { plan, .. } = self
            .filtered_read(
                filter_plan,
                self.dataset.empty_projection().with_row_id(),
                false,
                Some(fragments),
                None,
                /*is_prefilter= */ true,
            )
            .await?;
        Ok(PreFilterSource::FilteredRowIds(plan))
    }

    /// Take row indices produced by input plan from the dataset (with projection)
    #[allow(deprecated)]
    fn take(
        &self,
        input: Arc<dyn ExecutionPlan>,
        output_projection: Projection,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let coalesced = Arc::new(CoalesceBatchesExec::new(
            input.clone(),
            self.get_batch_size(),
        ));
        if let Some(take_plan) =
            TakeExec::try_new(self.dataset.clone(), coalesced, output_projection)?
        {
            Ok(Arc::new(take_plan))
        } else {
            // No new columns needed
            Ok(input)
        }
    }

    /// Global offset-limit of the result of the input plan
    fn limit_node(&self, plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        Arc::new(GlobalLimitExec::new(
            plan,
            *self.offset.as_ref().unwrap_or(&0) as usize,
            self.limit.map(|l| l as usize),
        ))
    }

    #[instrument(level = "info", skip(self))]
    pub async fn analyze_plan(&self) -> Result<String> {
        let plan = self.create_plan().await?;
        analyze_plan(
            plan,
            LanceExecutionOptions {
                batch_size: self.batch_size,
                ..Default::default()
            },
        )
        .await
    }

    #[instrument(level = "info", skip(self))]
    pub async fn explain_plan(&self, verbose: bool) -> Result<String> {
        let plan = self.create_plan().await?;
        let display = DisplayableExecutionPlan::new(plan.as_ref());

        Ok(format!("{}", display.indent(verbose)))
    }

    /// Run [`Self::count_rows`]'s underlying plan and return it formatted with
    /// runtime metrics. Equivalent to [`Self::analyze_plan`] but with a
    /// `COUNT(*)` aggregate auto-applied first — the only way for callers
    /// without a hand-built `AggregateExpr` (e.g. the Python bindings) to
    /// inspect the plan that `count_rows` actually executed.
    #[instrument(level = "info", skip(self))]
    pub async fn analyze_count_plan(&self) -> Result<String> {
        let mut scanner = self.clone();
        scanner.aggregate(AggregateExpr::builder().count_star().build())?;
        let plan = scanner.create_plan().await?;
        analyze_plan(
            plan,
            LanceExecutionOptions {
                batch_size: self.batch_size,
                ..Default::default()
            },
        )
        .await
    }
}

// Search over all indexed fields including nested ones, collecting columns that have an
// inverted index
async fn fts_indexed_columns(dataset: Arc<Dataset>) -> Result<Vec<String>> {
    let mut indexed_columns = Vec::new();
    for field in dataset.schema().fields_pre_order() {
        // Check if this field is a string type that could have an inverted index
        let is_string_field = match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 => true,
            DataType::List(inner_field) | DataType::LargeList(inner_field) => {
                matches!(
                    inner_field.data_type(),
                    DataType::Utf8 | DataType::LargeUtf8
                )
            }
            _ => false,
        };

        if is_string_field {
            // Build the full field path for nested fields
            let column_path =
                if let Some(ancestors) = dataset.schema().field_ancestry_by_id(field.id) {
                    let field_refs: Vec<&str> = ancestors.iter().map(|f| f.name.as_str()).collect();
                    format_field_path(&field_refs)
                } else {
                    continue; // Skip if we can't find the field ancestry
                };

            // Check if this field has an inverted index
            let has_fts_index = dataset
                .load_scalar_index(
                    IndexCriteria::default()
                        .for_column(&column_path)
                        .supports_fts(),
                )
                .await?
                .is_some();

            if has_fts_index {
                indexed_columns.push(column_path);
            }
        }
    }
    Ok(indexed_columns)
}

/// [`DatasetRecordBatchStream`] wraps the dataset into a [`RecordBatchStream`] for
/// consumption by the user.
///
#[pin_project::pin_project]
pub struct DatasetRecordBatchStream {
    #[pin]
    exec_node: SendableRecordBatchStream,
    span: Span,
}

impl DatasetRecordBatchStream {
    pub fn new(exec_node: SendableRecordBatchStream) -> Self {
        let schema = exec_node.schema();
        let adapter = SchemaAdapter::new(schema.clone());
        let exec_node = if SchemaAdapter::requires_logical_conversion(&schema) {
            adapter.to_logical_stream(exec_node)
        } else {
            exec_node
        };

        let span = info_span!("DatasetRecordBatchStream");
        Self { exec_node, span }
    }
}

impl RecordBatchStream for DatasetRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        self.exec_node.schema()
    }
}

impl Stream for DatasetRecordBatchStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        let _guard = this.span.enter();
        match this.exec_node.poll_next_unpin(cx) {
            Poll::Ready(result) => Poll::Ready(result.map(|r| Ok(r?))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl From<DatasetRecordBatchStream> for SendableRecordBatchStream {
    fn from(stream: DatasetRecordBatchStream) -> Self {
        stream.exec_node
    }
}

#[cfg(test)]
pub mod test_dataset {

    use super::*;

    use std::{collections::HashMap, vec};

    use arrow_array::{
        ArrayRef, FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
        types::Float32Type,
    };
    use arrow_schema::{ArrowError, DataType};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;
    use lance_index::{
        IndexType,
        scalar::{ScalarIndexParams, inverted::tokenizer::InvertedIndexParams},
        vector::{
            ivf::IvfBuildParams,
            kmeans::{KMeansParams, train_kmeans},
        },
    };
    use lance_linalg::distance::DistanceType;
    use uuid::Uuid;

    use crate::dataset::WriteParams;
    use crate::index::vector::VectorIndexParams;

    // Creates a dataset with 5 batches where each batch has 80 rows
    //
    // The dataset has the following columns:
    //
    //  i   - i32      : [0, 1, ..., 399]
    //  s   - &str     : ["s-0", "s-1", ..., "s-399"]
    //  vec - [f32; 32]: [[0, 1, ... 31], [32, ..., 63], ... [..., (80 * 5 * 32) - 1]]
    //
    // An IVF-PQ index with 2 partitions is trained on this data
    pub struct TestVectorDataset {
        pub tmp_dir: TempStrDir,
        pub schema: Arc<ArrowSchema>,
        pub dataset: Dataset,
        dimension: u32,
    }

    impl TestVectorDataset {
        pub async fn new(
            data_storage_version: LanceFileVersion,
            stable_row_ids: bool,
        ) -> Result<Self> {
            Self::new_with_dimension(data_storage_version, stable_row_ids, 32).await
        }

        pub async fn new_with_dimension(
            data_storage_version: LanceFileVersion,
            stable_row_ids: bool,
            dimension: u32,
        ) -> Result<Self> {
            let path = TempStrDir::default();

            // Make sure the schema has metadata so it tests all paths that re-construct the schema along the way
            let metadata: HashMap<String, String> =
                vec![("dataset".to_string(), "vector".to_string())]
                    .into_iter()
                    .collect();

            let schema = Arc::new(ArrowSchema::new_with_metadata(
                vec![
                    ArrowField::new("i", DataType::Int32, true),
                    ArrowField::new("s", DataType::Utf8, true),
                    ArrowField::new(
                        "vec",
                        DataType::FixedSizeList(
                            Arc::new(ArrowField::new("item", DataType::Float32, true)),
                            dimension as i32,
                        ),
                        true,
                    ),
                ],
                metadata,
            ));

            let batches: Vec<RecordBatch> = (0..5)
                .map(|i| {
                    let vector_values: Float32Array =
                        (0..dimension * 80).map(|v| v as f32).collect();
                    let vectors =
                        FixedSizeListArray::try_new_from_values(vector_values, dimension as i32)
                            .unwrap();
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(Int32Array::from_iter_values(i * 80..(i + 1) * 80)),
                            Arc::new(StringArray::from_iter_values(
                                (i * 80..(i + 1) * 80).map(|v| format!("s-{}", v)),
                            )),
                            Arc::new(vectors),
                        ],
                    )
                })
                .collect::<std::result::Result<Vec<_>, ArrowError>>()?;

            let params = WriteParams {
                max_rows_per_group: 10,
                max_rows_per_file: 200,
                data_storage_version: Some(data_storage_version),
                enable_stable_row_ids: stable_row_ids,
                ..Default::default()
            };
            let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

            let dataset = Dataset::write(reader, &path, Some(params)).await?;

            Ok(Self {
                tmp_dir: path,
                schema,
                dataset,
                dimension,
            })
        }

        pub async fn make_vector_index(&mut self) -> Result<()> {
            let params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 2);
            self.dataset
                .create_index(
                    &["vec"],
                    IndexType::Vector,
                    Some("idx".to_string()),
                    &params,
                    true,
                )
                .await?;
            Ok(())
        }

        pub async fn make_segmented_vector_index(&mut self) -> Result<Vec<Uuid>> {
            let batch = self
                .dataset
                .scan()
                .project(&["vec"])
                .unwrap()
                .try_into_batch()
                .await?;
            let vectors = batch
                .column_by_name("vec")
                .expect("vector column should exist")
                .as_fixed_size_list();
            let values = vectors.values().as_primitive::<Float32Type>();
            let centroids = train_kmeans::<Float32Type>(
                values,
                KMeansParams::new(None, 10, 1, DistanceType::L2),
                self.dimension as usize,
                2,
                2,
            )
            .unwrap()
            .centroids
            .as_primitive::<Float32Type>()
            .clone();
            let centroids = Arc::new(
                FixedSizeListArray::try_new_from_values(centroids, self.dimension as i32).unwrap(),
            );
            let params = VectorIndexParams::with_ivf_flat_params(
                DistanceType::L2,
                IvfBuildParams::try_with_centroids(2, centroids).unwrap(),
            );
            let fragment_ids = self
                .dataset
                .get_fragments()
                .iter()
                .map(|fragment| fragment.id() as u32)
                .collect::<Vec<_>>();

            let mut segments = Vec::with_capacity(fragment_ids.len());
            for fragment_id in fragment_ids {
                let mut builder =
                    self.dataset
                        .create_index_builder(&["vec"], IndexType::Vector, &params);
                builder = builder.name("idx".to_string()).fragments(vec![fragment_id]);
                segments.push(builder.execute_uncommitted().await?);
            }

            let segment_ids = segments
                .iter()
                .map(|segment| segment.uuid)
                .collect::<Vec<_>>();
            self.dataset
                .commit_existing_index_segments("idx", "vec", segments)
                .await?;
            Ok(segment_ids)
        }

        pub async fn make_scalar_index(&mut self) -> Result<()> {
            self.dataset
                .create_index(
                    &["i"],
                    IndexType::Scalar,
                    None,
                    &ScalarIndexParams::default(),
                    true,
                )
                .await?;
            Ok(())
        }

        pub async fn make_fts_index(&mut self) -> Result<()> {
            let params = InvertedIndexParams::default().with_position(true);
            self.dataset
                .create_index(&["s"], IndexType::Inverted, None, &params, true)
                .await?;
            Ok(())
        }

        pub async fn append_new_data(&mut self) -> Result<()> {
            self.append_data_with_range(400, 410).await
        }

        pub async fn append_data_with_range(&mut self, start: i32, end: i32) -> Result<()> {
            let count = (end - start) as usize;
            let vector_values: Float32Array = (0..count)
                .flat_map(|i| vec![i as f32; self.dimension as usize].into_iter())
                .collect();
            let new_vectors =
                FixedSizeListArray::try_new_from_values(vector_values, self.dimension as i32)
                    .unwrap();
            let new_data: Vec<ArrayRef> = vec![
                Arc::new(Int32Array::from_iter_values(start..end)),
                Arc::new(StringArray::from_iter_values(
                    (start..end).map(|v| format!("s-{}", v)),
                )),
                Arc::new(new_vectors),
            ];
            let reader = RecordBatchIterator::new(
                vec![RecordBatch::try_new(self.schema.clone(), new_data).unwrap()]
                    .into_iter()
                    .map(Ok),
                self.schema.clone(),
            );
            self.dataset.append(reader, None).await?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod test {

    use std::collections::BTreeSet;
    use std::time::{Duration, Instant};
    use std::vec;

    use arrow::array::as_primitive_array;
    use arrow::datatypes::{Float64Type, Int32Type, Int64Type};
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, UInt32Type, UInt64Type};
    use arrow_array::{
        ArrayRef, BooleanArray, FixedSizeListArray, Float16Array, Int32Array, LargeStringArray,
        PrimitiveArray, RecordBatchIterator, StringArray, StructArray, UInt8Array, UInt32Array,
    };

    use arrow_ord::sort::sort_to_indices;
    use arrow_schema::Fields;
    use arrow_select::take;
    use datafusion::logical_expr::{col, lit};
    use half::f16;
    use lance_arrow::{FixedSizeListArrayExt, SchemaExt};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION};
    use lance_datagen::{
        ArrayGeneratorExt, BatchCount, ByteCount, Dimension, RowCount, array, gen_batch,
    };
    use lance_file::version::LanceFileVersion;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::scalar::inverted::query::{MatchQuery, PhraseQuery};
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::PQBuildParams;
    use lance_index::vector::sq::builder::SQBuildParams;
    use lance_index::{IndexType, scalar::ScalarIndexParams};
    use lance_io::assert_io_gt;
    use lance_io::object_store::ObjectStoreParams;

    use lance_linalg::distance::DistanceType;
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32, RandomVector};
    use object_store::throttle::ThrottleConfig;
    use rstest::rstest;

    use super::*;
    use crate::dataset::WriteMode;
    use crate::dataset::WriteParams;
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::scanner::test_dataset::TestVectorDataset;
    use crate::index::vector::{StageParams, VectorIndexParams};
    use crate::utils::test::{
        DatagenExt, FragmentCount, FragmentRowCount, ThrottledStoreWrapper, assert_plan_node_equals,
    };

    #[test]
    fn test_env_var_parsing() {
        // Test that invalid environment variable values don't panic

        // Test invalid LANCE_DEFAULT_BATCH_SIZE
        unsafe {
            std::env::set_var("LANCE_DEFAULT_BATCH_SIZE", "not_a_number");
        }
        let result = get_default_batch_size();
        assert_eq!(result, None, "Should return None for invalid batch size");

        // Test valid LANCE_DEFAULT_BATCH_SIZE
        unsafe {
            std::env::set_var("LANCE_DEFAULT_BATCH_SIZE", "2048");
        }
        let result = get_default_batch_size();
        assert_eq!(result, Some(2048), "Should parse valid batch size");

        // Test unset LANCE_DEFAULT_BATCH_SIZE
        unsafe {
            std::env::remove_var("LANCE_DEFAULT_BATCH_SIZE");
        }
        let result = get_default_batch_size();
        assert_eq!(result, None, "Should return None when env var is not set");
    }

    #[test]
    fn test_parse_env_var() {
        // Test parse_env_var with different types to ensure full coverage

        // Test with a unique env var name to avoid conflicts
        let test_var = "LANCE_TEST_PARSE_ENV_VAR_USIZE";

        // Test valid usize parsing
        unsafe {
            std::env::set_var(test_var, "12345");
        }
        let result: Option<usize> = parse_env_var(test_var, "Using default.");
        assert_eq!(result, Some(12345));

        // Test invalid usize parsing (triggers warning log)
        unsafe {
            std::env::set_var(test_var, "not_a_number");
        }
        let result: Option<usize> = parse_env_var(test_var, "Using default.");
        assert_eq!(result, None);

        // Test unset env var
        unsafe {
            std::env::remove_var(test_var);
        }
        let result: Option<usize> = parse_env_var(test_var, "Using default.");
        assert_eq!(result, None);

        // Test with u32 type
        let test_var_u32 = "LANCE_TEST_PARSE_ENV_VAR_U32";
        unsafe {
            std::env::set_var(test_var_u32, "42");
        }
        let result: Option<u32> = parse_env_var(test_var_u32, "Using default value.");
        assert_eq!(result, Some(42));

        unsafe {
            std::env::set_var(test_var_u32, "invalid");
        }
        let result: Option<u32> = parse_env_var(test_var_u32, "Using default value.");
        assert_eq!(result, None);

        unsafe {
            std::env::remove_var(test_var_u32);
        }

        // Test with u64 type
        let test_var_u64 = "LANCE_TEST_PARSE_ENV_VAR_U64";
        unsafe {
            std::env::set_var(test_var_u64, "9999999999");
        }
        let result: Option<u64> = parse_env_var(test_var_u64, "Using default value.");
        assert_eq!(result, Some(9999999999));

        unsafe {
            std::env::set_var(test_var_u64, "-1");
        }
        let result: Option<u64> = parse_env_var(test_var_u64, "Using default value.");
        assert_eq!(result, None);

        unsafe {
            std::env::remove_var(test_var_u64);
        }
    }

    async fn make_binary_vector_dataset() -> Result<(TempStrDir, Dataset)> {
        let tmp_dir = TempStrDir::default();
        let dim = 4;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new(
                "bin",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::UInt8, true)),
                    dim,
                ),
                false,
            ),
        ]));

        let vectors = FixedSizeListArray::try_new_from_values(
            UInt8Array::from(vec![
                0b0000_1111u8,
                0,
                0,
                0, //
                0b0000_0011u8,
                0,
                0,
                0, //
                0u8,
                0,
                0,
                0,
            ]),
            dim,
        )?;
        let ids = Int32Array::from(vec![0, 1, 2]);

        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vectors)])?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        Dataset::write(reader, &tmp_dir, None).await?;
        let dataset = Dataset::open(&tmp_dir).await?;
        Ok((tmp_dir, dataset))
    }

    #[tokio::test]
    async fn test_batch_size() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let batches: Vec<RecordBatch> = (0..5)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                        Arc::new(StringArray::from_iter_values(
                            (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();

        for use_filter in [false, true] {
            let test_dir = TempStrDir::default();
            let test_uri = &test_dir;
            let write_params = WriteParams {
                max_rows_per_file: 40,
                max_rows_per_group: 10,
                ..Default::default()
            };
            let batches =
                RecordBatchIterator::new(batches.clone().into_iter().map(Ok), schema.clone());
            Dataset::write(batches, test_uri, Some(write_params))
                .await
                .unwrap();

            let dataset = Dataset::open(test_uri).await.unwrap();
            let mut builder = dataset.scan();
            builder.batch_size(8);
            if use_filter {
                builder.filter("i IS NOT NULL").unwrap();
            }
            let mut stream = builder.try_into_stream().await.unwrap();
            let mut rows_read = 0;
            while let Some(next) = stream.next().await {
                let next = next.unwrap();
                let expected = 8.min(100 - rows_read);
                assert_eq!(next.num_rows(), expected);
                rows_read += next.num_rows();
            }
        }
    }

    #[tokio::test]
    async fn test_strict_batch_size() {
        let dataset = lance_datagen::gen_batch()
            .col("x", array::step::<Int32Type>())
            .anon_col(array::step::<Int64Type>())
            .into_ram_dataset(FragmentCount::from(7), FragmentRowCount::from(6))
            .await
            .unwrap();

        let mut scan = dataset.scan();
        scan.batch_size(10)
            .strict_batch_size(true)
            .filter("x % 2 == 0")
            .unwrap();

        let batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let batch_sizes = batches.iter().map(|b| b.num_rows()).collect::<Vec<_>>();
        assert_eq!(batch_sizes, vec![10, 10, 1]);
    }

    #[tokio::test]
    async fn test_column_not_exist() {
        let dataset = lance_datagen::gen_batch()
            .col("x", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(7), FragmentRowCount::from(6))
            .await
            .unwrap();

        let check_err_msg = |r: Result<DatasetRecordBatchStream>| {
            let Err(err) = r else {
                panic!(
                    "Expected an error to be raised saying column y is not found but got no error"
                )
            };

            assert!(
                err.to_string().contains("No field named y"),
                "Expected error to contain 'No field named y' but got {}",
                err
            );
        };

        let mut scan = dataset.scan();
        scan.project(&["x", "y"]).unwrap();
        check_err_msg(scan.try_into_stream().await);

        let mut scan = dataset.scan();
        scan.project(&["y"]).unwrap();
        check_err_msg(scan.try_into_stream().await);

        // This represents a query like `SELECT 1 AS foo` which we could _technically_ satisfy
        // but it is not supported today
        let mut scan = dataset.scan();
        scan.project_with_transform(&[("foo", "1")]).unwrap();
        match scan.try_into_stream().await {
            Ok(_) => panic!("Expected an error to be raised saying not supported"),
            Err(e) => {
                assert!(
                    e.to_string().contains("Received only dynamic expressions"),
                    "Expected error to contain 'Received only dynamic expressions' but got {}",
                    e
                );
            }
        }
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn test_local_object_store() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let batches: Vec<RecordBatch> = (0..5)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                        Arc::new(StringArray::from_iter_values(
                            (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.clone().into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(&format!("file-object-store://{}", test_uri))
            .await
            .unwrap();
        let mut builder = dataset.scan();
        builder.batch_size(8);
        let mut stream = builder.try_into_stream().await.unwrap();
        let mut rows_read = 0;
        while let Some(next) = stream.next().await {
            let next = next.unwrap();
            let expected = 8.min(100 - rows_read);
            assert_eq!(next.num_rows(), expected);
            rows_read += next.num_rows();
        }
    }

    #[tokio::test]
    async fn test_filter_parsing() -> Result<()> {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false).await?;
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        assert!(scan.filter.is_none());

        scan.filter("i > 50")?;
        assert_eq!(scan.get_expr_filter().unwrap(), Some(col("i").gt(lit(50))));

        for use_stats in [false, true] {
            let batches = scan
                .project(&["s"])?
                .use_stats(use_stats)
                .try_into_stream()
                .await?
                .try_collect::<Vec<_>>()
                .await?;
            let batch = concat_batches(&batches[0].schema(), &batches)?;

            let expected_batch = RecordBatch::try_new(
                // Projected just "s"
                Arc::new(test_ds.schema.project(&[1])?),
                vec![Arc::new(StringArray::from_iter_values(
                    (51..400).map(|v| format!("s-{}", v)),
                ))],
            )?;
            assert_eq!(batch, expected_batch);
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_scan_regexp_match_and_non_empty_captions() {
        // Build a small dataset with three Utf8 columns and verify the full
        // scan().filter(...) path handles regexp_match combined with non-null/non-empty checks.
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("keywords", DataType::Utf8, true),
            ArrowField::new("natural_caption", DataType::Utf8, true),
            ArrowField::new("poetic_caption", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![
                    Some("Liberty for all"),
                    Some("peace"),
                    Some("revolution now"),
                    Some("Liberty"),
                    Some("revolutionary"),
                    Some("none"),
                ])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("b"),
                    None,
                    Some(""),
                    Some("c"),
                    Some("d"),
                ])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    Some("x"),
                    Some(""),
                    Some("y"),
                    Some("z"),
                    None,
                    Some("w"),
                ])) as ArrayRef,
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
        let dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        let mut scan = dataset.scan();
        scan.filter(
            "regexp_match(keywords, 'Liberty|revolution') AND \
             (natural_caption IS NOT NULL AND natural_caption <> '' AND \
              poetic_caption IS NOT NULL AND poetic_caption <> '')",
        )
        .unwrap();

        let out = scan.try_into_batch().await.unwrap();
        assert_eq!(out.num_rows(), 1);

        let out_keywords = out
            .column_by_name("keywords")
            .unwrap()
            .as_string::<i32>()
            .value(0);
        let out_nat = out
            .column_by_name("natural_caption")
            .unwrap()
            .as_string::<i32>()
            .value(0);
        let out_poetic = out
            .column_by_name("poetic_caption")
            .unwrap()
            .as_string::<i32>()
            .value(0);

        assert_eq!(out_keywords, "Liberty for all");
        assert_eq!(out_nat, "a");
        assert_eq!(out_poetic, "x");
    }

    #[tokio::test]
    async fn test_nested_projection() {
        let point_fields: Fields = vec![
            ArrowField::new("x", DataType::Float32, true),
            ArrowField::new("y", DataType::Float32, true),
        ]
        .into();
        let metadata_fields: Fields = vec![
            ArrowField::new("location", DataType::Struct(point_fields), true),
            ArrowField::new("age", DataType::Int32, true),
        ]
        .into();
        let metadata_field = ArrowField::new("metadata", DataType::Struct(metadata_fields), true);
        let schema = Arc::new(ArrowSchema::new(vec![
            metadata_field,
            ArrowField::new("idx", DataType::Int32, true),
        ]));
        let data = lance_datagen::rand(&schema)
            .into_ram_dataset(FragmentCount::from(7), FragmentRowCount::from(6))
            .await
            .unwrap();

        let mut scan = data.scan();
        scan.project(&["metadata.location.x", "metadata.age"])
            .unwrap();
        let batch = scan.try_into_batch().await.unwrap();

        assert_eq!(
            batch.schema().as_ref(),
            &ArrowSchema::new(vec![
                ArrowField::new("metadata.location.x", DataType::Float32, true),
                ArrowField::new("metadata.age", DataType::Int32, true),
            ])
        );

        // 0 - metadata
        // 2 - x
        // 4 - age
        let take_schema = data.schema().project_by_ids(&[0, 2, 4], false);

        let taken = data.take_rows(&[0, 5], take_schema).await.unwrap();

        // The expected schema drops y from the location field
        let part_point_fields = Fields::from(vec![ArrowField::new("x", DataType::Float32, true)]);
        let part_metadata_fields = Fields::from(vec![
            ArrowField::new("location", DataType::Struct(part_point_fields), true),
            ArrowField::new("age", DataType::Int32, true),
        ]);
        let part_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "metadata",
            DataType::Struct(part_metadata_fields),
            true,
        )]));

        assert_eq!(taken.schema(), part_schema);
    }

    #[rstest]
    #[tokio::test]
    async fn test_limit(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) -> Result<()> {
        let test_ds = TestVectorDataset::new(data_storage_version, false).await?;
        let dataset = &test_ds.dataset;

        let full_data = dataset.scan().try_into_batch().await?.slice(19, 2);

        let actual = dataset
            .scan()
            .limit(Some(2), Some(19))?
            .try_into_batch()
            .await?;

        assert_eq!(actual.num_rows(), 2);
        assert_eq!(actual, full_data);
        Ok(())
    }

    #[tokio::test]
    async fn test_limit_with_scalar_index_and_refine_filter() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("topic", DataType::Int32, false),
            ArrowField::new("is_night", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..20)),
                Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(1, 20))),
                Arc::new(Int32Array::from_iter_values(
                    (0..20).map(|i| if i < 10 { 0 } else { 1 }),
                )),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
        dataset
            .create_index(
                &["topic"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        let actual = dataset
            .scan()
            .filter("topic = 1 AND is_night = 1")
            .unwrap()
            .limit(Some(10), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(actual.num_rows(), 10);
        let ids = actual
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>()
            .values();
        assert_eq!(ids, &(10..20).collect::<Vec<_>>());
    }

    #[test_log::test(tokio::test)]
    async fn test_limit_cancel() {
        // If there is a filter and a limit and we can't use the index to satisfy
        // the filter, then we have to read until we have enough matching rows and
        // then cancel the scan.
        //
        // This test regresses the case where we fail to cancel the scan for whatever
        // reason.

        // Make the store slow so that if we don't cancel the scan, it will take a loooong time.
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_get_per_call: Duration::from_secs(1),
                ..Default::default()
            },
        });
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled.clone()),
                ..Default::default()
            }),
            max_rows_per_file: 1,
            ..Default::default()
        };

        // Make a dataset with lots of tiny fragments, that will make it more obvious if we fail to cancel the scan.
        let dataset = gen_batch()
            .col("i", array::step::<Int32Type>().with_random_nulls(0.1))
            .into_ram_dataset_with_params(
                FragmentCount::from(2000),
                FragmentRowCount::from(1),
                Some(write_params),
            )
            .await
            .unwrap();

        let mut scan = dataset.scan();
        scan.filter("i IS NOT NULL").unwrap();
        scan.limit(Some(10), None).unwrap();

        let start = Instant::now();
        scan.try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let duration = start.elapsed();

        // This test is a timing test, which is unfortunate, as it may be flaky.  I'm hoping
        // we have enough wiggle room here.  The failure case is 30s on my machine and the pass
        // case is 2-3s.
        assert!(duration < Duration::from_secs(10));
    }

    #[rstest]
    #[tokio::test]
    async fn test_knn_nodes(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
        #[values(false, true)] build_index: bool,
    ) {
        let mut test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids)
            .await
            .unwrap();
        if build_index {
            test_ds.make_vector_index().await.unwrap();
        }
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.nearest("vec", &key, 5).unwrap();
        scan.refine(5);

        let batch = scan.try_into_batch().await.unwrap();

        assert_eq!(batch.num_rows(), 5);
        assert_eq!(
            batch.schema().as_ref(),
            &ArrowSchema::new(vec![
                ArrowField::new("i", DataType::Int32, true),
                ArrowField::new("s", DataType::Utf8, true),
                ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        32,
                    ),
                    true,
                ),
                ArrowField::new(DIST_COL, DataType::Float32, true),
            ])
            .with_metadata([("dataset".into(), "vector".into())].into())
        );

        let expected_i = BTreeSet::from_iter(vec![1, 81, 161, 241, 321]);
        let column_i = batch.column_by_name("i").unwrap();
        let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
            .values()
            .iter()
            .copied()
            .collect();
        assert_eq!(expected_i, actual_i);
    }

    fn batch_knn_two_queries() -> (FixedSizeListArray, Vec<f32>) {
        let query_values = (32..96).map(|v| v as f32).collect::<Vec<_>>();
        let queries =
            FixedSizeListArray::try_new_from_values(Float32Array::from(query_values.clone()), 32)
                .unwrap();
        (queries, query_values)
    }

    fn assert_query_index_field(batch: &RecordBatch) {
        let schema = batch.schema();
        let field = schema.field(0);
        assert_eq!(field.name(), QUERY_INDEX_COL);
        assert_eq!(field.data_type(), &DataType::Int32);
        assert!(!field.is_nullable());
    }

    async fn assert_batch_matches_single_queries(
        dataset: &Dataset,
        batch: &RecordBatch,
        query_values: &[f32],
        k: usize,
        use_index: bool,
        distance_range: Option<(Option<f32>, Option<f32>)>,
    ) {
        let query_count = query_values.len() / 32;
        assert_eq!(batch.num_rows(), query_count * k);

        for query_index in 0..query_count {
            let query =
                Float32Array::from(query_values[query_index * 32..(query_index + 1) * 32].to_vec());
            let mut scan = dataset.scan();
            scan.nearest("vec", &query, k).unwrap();
            scan.use_index(use_index);
            if let Some((lower, upper)) = distance_range {
                scan.distance_range(lower, upper);
            }
            scan.project(&["i", "_distance"]).unwrap();
            let single = scan.try_into_batch().await.unwrap();

            let query_indices = batch[QUERY_INDEX_COL].as_primitive::<Int32Type>();
            let mask = BooleanArray::from_iter(
                query_indices
                    .iter()
                    .map(|value| value.map(|value| value == query_index as i32)),
            );
            let batch_slice = arrow::compute::filter_record_batch(batch, &mask).unwrap();
            assert_eq!(
                batch_slice["i"].as_primitive::<Int32Type>().values(),
                single["i"].as_primitive::<Int32Type>().values()
            );
            assert_eq!(
                batch_slice[DIST_COL].as_primitive::<Float32Type>().values(),
                single[DIST_COL].as_primitive::<Float32Type>().values()
            );
        }
    }

    #[tokio::test]
    async fn test_batch_knn_flat() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        let dataset = &test_ds.dataset;
        let k = 2;

        let (queries, query_values) = batch_knn_two_queries();
        let mut scan = dataset.scan();
        scan.nearest("vec", &queries, k).unwrap();
        scan.use_index(false);
        scan.project(&["i", "_distance"]).unwrap();

        let plan = scan.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("KNNVectorDistance: queries=2"),
            "expected flat batch KNN plan, got:\n{}",
            plan
        );
        assert!(
            !plan.contains("ANNSubIndex"),
            "flat batch KNN should not use ANN index, got:\n{}",
            plan
        );
        assert!(
            !plan.contains("SortExec: TopK(fetch="),
            "batch flat KNN must not truncate to k rows globally, got:\n{}",
            plan
        );

        let batch = scan.try_into_batch().await.unwrap();
        assert_query_index_field(&batch);
        assert_eq!(
            batch.num_rows(),
            2 * k,
            "batch flat KNN must return k rows per query vector"
        );
        assert_eq!(
            batch[QUERY_INDEX_COL].as_primitive::<Int32Type>().values(),
            &[0, 0, 1, 1]
        );
        let query_indices = batch[QUERY_INDEX_COL].as_primitive::<Int32Type>();
        for query_index in 0..2 {
            let rows_for_query = query_indices
                .iter()
                .filter(|value| *value == Some(query_index))
                .count();
            assert_eq!(
                rows_for_query, k,
                "query_index {query_index} should have exactly {k} rows"
            );
        }
        assert_batch_matches_single_queries(dataset, &batch, &query_values, k, false, None).await;

        let query_values_one = (32..64).map(|v| v as f32).collect::<Vec<_>>();
        let queries_one = FixedSizeListArray::try_new_from_values(
            Float32Array::from(query_values_one.clone()),
            32,
        )
        .unwrap();
        let mut scan = dataset.scan();
        scan.nearest("vec", &queries_one, k).unwrap();
        scan.use_index(false);
        scan.project(&["i", "_distance"]).unwrap();

        let plan = scan.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("KNNVectorDistance: queries=1"),
            "single-vector batch query should use batch KNN path, got:\n{}",
            plan
        );
        assert!(
            !plan.contains("SortExec: TopK(fetch="),
            "batch KNN must not apply per-query SortExec top-k, got:\n{}",
            plan
        );

        let batch = scan.try_into_batch().await.unwrap();
        assert_query_index_field(&batch);
        assert_eq!(
            batch[QUERY_INDEX_COL].as_primitive::<Int32Type>().values(),
            &[0, 0]
        );
    }

    #[tokio::test]
    async fn test_primitive_query_length_multiple_of_dim_is_rejected() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        let dataset = &test_ds.dataset;
        let q: Float32Array = (32..96).map(|v| v as f32).collect();

        let err = match dataset.scan().nearest("vec", &q, 2) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("expected primitive query length mismatch error"),
        };
        assert!(
            err.contains("query dim(64) doesn't match the column vec vector dim(32)"),
            "unexpected error: {err}"
        );
    }

    async fn dataset_with_query_index_column() -> (TempStrDir, Dataset) {
        let path = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    32,
                ),
                true,
            ),
            ArrowField::new(QUERY_INDEX_COL, DataType::UInt32, true),
        ]));
        let vector_values: Float32Array = (0..32 * 80).map(|v| v as f32).collect();
        let vectors = FixedSizeListArray::try_new_from_values(vector_values, 32).unwrap();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..80)),
                Arc::new(vectors),
                Arc::new(UInt32Array::from_iter((0..80).map(|v| v as u32))),
            ],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(std::iter::once(Ok(batch)), schema.clone()),
            &path,
            None,
        )
        .await
        .unwrap();
        (path, dataset)
    }

    #[tokio::test]
    async fn test_batch_knn_rejects_dataset_query_index_column() {
        let (_tmp, dataset) = dataset_with_query_index_column().await;
        let (queries, _) = batch_knn_two_queries();
        let err = match dataset.scan().nearest("vec", &queries, 2) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("expected reserved query_index column error"),
        };
        assert!(err.contains(QUERY_INDEX_COL), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn test_single_knn_projects_dataset_query_index_column() {
        let (_tmp, dataset) = dataset_with_query_index_column().await;
        let q: Float32Array = (32..64).map(|v| v as f32).collect();

        let mut scan = dataset.scan();
        scan.nearest("vec", &q, 2).unwrap();
        scan.use_index(false);
        scan.project(&["i"]).unwrap();
        let without_query_index = scan.try_into_batch().await.unwrap();

        let mut scan = dataset.scan();
        scan.nearest("vec", &q, 2).unwrap();
        scan.use_index(false);
        scan.project(&["i", QUERY_INDEX_COL]).unwrap();
        let with_query_index = scan.try_into_batch().await.unwrap();

        assert_eq!(without_query_index.num_rows(), 2);
        assert_eq!(
            without_query_index["i"]
                .as_primitive::<Int32Type>()
                .values(),
            with_query_index["i"].as_primitive::<Int32Type>().values()
        );
        assert_eq!(
            with_query_index[QUERY_INDEX_COL]
                .as_primitive::<UInt32Type>()
                .null_count(),
            0
        );
    }

    #[tokio::test]
    async fn test_batch_knn_flat_respects_distance_range() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        let dataset = &test_ds.dataset;
        let (queries, query_values) = batch_knn_two_queries();

        let batch = dataset
            .scan()
            .nearest("vec", &queries, 2)
            .unwrap()
            .use_index(false)
            .distance_range(Some(1.0), None)
            .project(&["i", "_distance"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(
            batch[QUERY_INDEX_COL].as_primitive::<Int32Type>().values(),
            &[0, 0, 1, 1]
        );
        assert_batch_matches_single_queries(
            dataset,
            &batch,
            &query_values,
            2,
            false,
            Some((Some(1.0), None)),
        )
        .await;
    }

    #[tokio::test]
    async fn test_batch_knn_indexed() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        test_ds.make_vector_index().await.unwrap();
        let dataset = &test_ds.dataset;
        let (queries, query_values) = batch_knn_two_queries();

        let mut scan = dataset.scan();
        scan.nearest("vec", &queries, 2).unwrap();
        scan.project(&["i", "_distance"]).unwrap();

        let plan = scan.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ANNSubIndex"),
            "batch KNN should use the vector index when available, got:\n{}",
            plan
        );
        assert!(
            !plan.contains("KNNVectorDistance: queries=2"),
            "indexed batch KNN should not force the flat batch path, got:\n{}",
            plan
        );

        let batch = scan.try_into_batch().await.unwrap();
        assert_query_index_field(&batch);
        assert_eq!(
            batch[QUERY_INDEX_COL].as_primitive::<Int32Type>().values(),
            &[0, 0, 1, 1]
        );

        let batch = dataset
            .scan()
            .nearest("vec", &queries, 2)
            .unwrap()
            .distance_range(Some(1.0), None)
            .project(&["i", "_distance"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_batch_matches_single_queries(
            dataset,
            &batch,
            &query_values,
            2,
            true,
            Some((Some(1.0), None)),
        )
        .await;
    }

    #[tokio::test]
    async fn test_can_project_distance() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.nearest("vec", &key, 5).unwrap();
        scan.refine(5);
        scan.project(&["_distance"]).unwrap();

        let batch = scan.try_into_batch().await.unwrap();

        assert_eq!(batch.num_rows(), 5);
        assert_eq!(batch.num_columns(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_knn_with_new_data(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let mut test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids)
            .await
            .unwrap();
        test_ds.make_vector_index().await.unwrap();
        test_ds.append_new_data().await.unwrap();
        let dataset = &test_ds.dataset;

        // Create a bunch of queries
        let key: Float32Array = [0f32; 32].into_iter().collect();
        // Set as larger than the number of new rows that aren't in the index to
        // force result sets to be combined between index and flat scan.
        let k = 20;

        #[derive(Debug)]
        struct TestCase {
            filter: Option<&'static str>,
            limit: Option<i64>,
            use_index: bool,
        }

        let mut cases = vec![];
        for filter in [Some("i > 100"), None] {
            for limit in [None, Some(10)] {
                for use_index in [true, false] {
                    cases.push(TestCase {
                        filter,
                        limit,
                        use_index,
                    });
                }
            }
        }

        // Validate them all.
        for case in cases {
            let mut scanner = dataset.scan();
            scanner
                .nearest("vec", &key, k)
                .unwrap()
                .limit(case.limit, None)
                .unwrap()
                .refine(3)
                .use_index(case.use_index);
            if let Some(filter) = case.filter {
                scanner.filter(filter).unwrap();
            }

            let result = scanner
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            assert!(!result.is_empty());
            let result = concat_batches(&result[0].schema(), result.iter()).unwrap();

            if case.filter.is_some() {
                let result_rows = result.num_rows();
                let expected_rows = case.limit.unwrap_or(k as i64) as usize;
                assert!(
                    result_rows <= expected_rows,
                    "Expected less than {} rows, got {}",
                    expected_rows,
                    result_rows
                );
            } else {
                // Exactly equal count
                assert_eq!(result.num_rows(), case.limit.unwrap_or(k as i64) as usize);
            }

            // Top one should be the first value of new data
            assert_eq!(
                as_primitive_array::<Int32Type>(result.column(0).as_ref()).value(0),
                400
            );
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_knn_with_prefilter(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let mut test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids)
            .await
            .unwrap();
        test_ds.make_vector_index().await.unwrap();
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.filter("i > 100").unwrap();
        scan.prefilter(true);
        scan.project(&["i", "vec", "_distance"]).unwrap();
        scan.nearest("vec", &key, 5).unwrap();
        scan.use_index(false);

        let results = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        let batch = &results[0];

        assert_eq!(batch.num_rows(), 5);
        assert_eq!(
            batch.schema().as_ref(),
            &ArrowSchema::new(vec![
                ArrowField::new("i", DataType::Int32, true),
                ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        32,
                    ),
                    true,
                ),
                ArrowField::new(DIST_COL, DataType::Float32, true),
            ])
            .with_metadata([("dataset".into(), "vector".into())].into())
        );

        // These match the query exactly.  The 5 results must include these 3.
        let exact_i = BTreeSet::from_iter(vec![161, 241, 321]);
        // These also include those 1 off from the query.  The remaining 2 results must be in this set.
        let close_i = BTreeSet::from_iter(vec![161, 241, 321, 160, 162, 240, 242, 320, 322]);
        let column_i = batch.column_by_name("i").unwrap();
        let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
            .values()
            .iter()
            .copied()
            .collect();
        assert!(exact_i.is_subset(&actual_i));
        assert!(actual_i.is_subset(&close_i));
    }

    #[rstest]
    #[tokio::test]
    async fn test_knn_filter_new_data(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        // This test verifies that a filter (prefilter or postfilter) gets applied to the flat KNN results
        // in a combined KNN scan (a scan that combines results from an indexed ANN with an unindexed flat
        // search of new data)
        let mut test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids)
            .await
            .unwrap();
        test_ds.make_vector_index().await.unwrap();
        test_ds.append_new_data().await.unwrap();
        let dataset = &test_ds.dataset;

        // This query will match exactly the new row with i = 400 which should be excluded by the prefilter
        let key: Float32Array = [0f32; 32].into_iter().collect();

        let mut query = dataset.scan();
        query.nearest("vec", &key, 20).unwrap();

        // Sanity check that 400 is in our results
        let results = query
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let results_i = results[0]["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();

        assert!(results_i.contains(&400));

        // Both prefilter and postfilter should remove 400 from our results
        for prefilter in [false, true] {
            let mut query = dataset.scan();
            query
                .filter("i != 400")
                .unwrap()
                .prefilter(prefilter)
                .nearest("vec", &key, 20)
                .unwrap();

            let results = query
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            let results_i = results[0]["i"]
                .as_primitive::<Int32Type>()
                .values()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();

            assert!(!results_i.contains(&400));
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_knn_with_filter(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids)
            .await
            .unwrap();
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.nearest("vec", &key, 5).unwrap();
        scan.filter("i > 100").unwrap();
        scan.project(&["i", "vec", "_distance"]).unwrap();
        scan.refine(5);

        let results = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        let batch = &results[0];

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(
            batch.schema().as_ref(),
            &ArrowSchema::new(vec![
                ArrowField::new("i", DataType::Int32, true),
                ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        32,
                    ),
                    true,
                ),
                ArrowField::new(DIST_COL, DataType::Float32, true),
            ])
            .with_metadata([("dataset".into(), "vector".into())].into())
        );

        let expected_i = BTreeSet::from_iter(vec![161, 241, 321]);
        let column_i = batch.column_by_name("i").unwrap();
        let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
            .values()
            .iter()
            .copied()
            .collect();
        assert_eq!(expected_i, actual_i);
    }

    #[rstest]
    #[tokio::test]
    async fn test_refine_factor(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids)
            .await
            .unwrap();
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.nearest("vec", &key, 5).unwrap();
        scan.refine(5);

        let results = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        let batch = &results[0];

        assert_eq!(batch.num_rows(), 5);
        assert_eq!(
            batch.schema().as_ref(),
            &ArrowSchema::new(vec![
                ArrowField::new("i", DataType::Int32, true),
                ArrowField::new("s", DataType::Utf8, true),
                ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        32,
                    ),
                    true,
                ),
                ArrowField::new(DIST_COL, DataType::Float32, true),
            ])
            .with_metadata([("dataset".into(), "vector".into())].into())
        );

        let expected_i = BTreeSet::from_iter(vec![1, 81, 161, 241, 321]);
        let column_i = batch.column_by_name("i").unwrap();
        let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
            .values()
            .iter()
            .copied()
            .collect();
        assert_eq!(expected_i, actual_i);
    }

    #[tokio::test]
    async fn test_binary_vectors_default_to_hamming() {
        let (_tmp_dir, dataset) = make_binary_vector_dataset().await.unwrap();
        let query = UInt8Array::from(vec![0b0000_1111u8, 0, 0, 0]);

        let mut scan = dataset.scan();
        scan.nearest("bin", &query, 3).unwrap();

        // metric_type is None initially; it will be resolved to Hamming during search
        assert_eq!(scan.nearest.as_ref().unwrap().metric_type, None);

        let batch = scan.try_into_batch().await.unwrap();
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>()
            .values();
        assert_eq!(ids, &[0, 1, 2]);
        let distances = batch
            .column_by_name(DIST_COL)
            .unwrap()
            .as_primitive::<Float32Type>()
            .values();
        assert_eq!(distances, &[0.0, 2.0, 4.0]);
    }

    #[tokio::test]
    async fn test_binary_vectors_invalid_distance_error() {
        let (_tmp_dir, dataset) = make_binary_vector_dataset().await.unwrap();
        let query = UInt8Array::from(vec![0b0000_1111u8, 0, 0, 0]);

        let mut scan = dataset.scan();
        scan.nearest("bin", &query, 1).unwrap();
        scan.distance_metric(DistanceType::L2);

        let err = scan.try_into_batch().await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        let message = err.to_string();
        assert!(
            message.contains("l2") && message.contains("UInt8"),
            "unexpected message: {message}"
        );
    }

    /// Test that when query specifies a metric different from the index,
    /// we fall back to flat search and return correct distances.
    /// Regression test for https://github.com/lance-format/lance/issues/5608
    #[tokio::test]
    async fn test_knn_metric_mismatch_falls_back_to_flat_search() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        // Create IVF_PQ index with L2 metric
        test_ds.make_vector_index().await.unwrap();

        let dataset = &test_ds.dataset;
        let key: Float32Array = (32..64).map(|v| v as f32).collect();

        // Query with Dot metric (different from the L2 index)
        let mut scan = dataset.scan();
        scan.nearest("vec", &key, 5).unwrap();
        scan.distance_metric(DistanceType::Dot);

        // Verify the explain plan does NOT show ANNSubIndex (should use flat search)
        let plan = scan.explain_plan(false).await.unwrap();
        assert!(
            !plan.contains("ANNSubIndex"),
            "Expected flat search, but got ANN index in plan:\n{}",
            plan
        );
        // Should show flat KNN with Dot metric (metric is displayed lowercase)
        assert!(
            plan.contains("KNNVectorDistance") && plan.to_lowercase().contains("dot"),
            "Expected flat KNN with Dot metric in plan:\n{}",
            plan
        );

        // Also verify the distances are different from L2 results
        let dot_batch = dataset
            .scan()
            .nearest("vec", &key, 5)
            .unwrap()
            .distance_metric(DistanceType::Dot)
            .try_into_batch()
            .await
            .unwrap();

        let l2_batch = dataset
            .scan()
            .nearest("vec", &key, 5)
            .unwrap()
            .distance_metric(DistanceType::L2)
            .try_into_batch()
            .await
            .unwrap();

        let dot_distances: Vec<f32> = dot_batch
            .column_by_name(DIST_COL)
            .unwrap()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let l2_distances: Vec<f32> = l2_batch
            .column_by_name(DIST_COL)
            .unwrap()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();

        // Dot and L2 distances should be different (this verifies we're using the correct metric)
        assert_ne!(dot_distances, l2_distances);
    }

    /// Test that when query does not specify a metric, we use the index's metric.
    /// Regression test for https://github.com/lance-format/lance/issues/5608
    #[tokio::test]
    async fn test_knn_no_metric_uses_index_metric() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        // Create IVF_PQ index with L2 metric
        test_ds.make_vector_index().await.unwrap();

        let dataset = &test_ds.dataset;
        let key: Float32Array = (32..64).map(|v| v as f32).collect();

        // Query without specifying metric
        let mut scan = dataset.scan();
        scan.nearest("vec", &key, 5).unwrap();
        // Don't call distance_metric() - should use index's L2

        // Verify the explain plan shows ANNSubIndex with L2 metric
        let plan = scan.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ANNSubIndex") && plan.to_lowercase().contains("l2"),
            "Expected ANN index with L2 metric in plan:\n{}",
            plan
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_only_row_id(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_ds = TestVectorDataset::new(data_storage_version, false)
            .await
            .unwrap();
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        scan.project::<&str>(&[]).unwrap().with_row_id();

        let batch = scan.try_into_batch().await.unwrap();

        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.num_rows(), 400);
        let expected_schema =
            ArrowSchema::new(vec![ArrowField::new(ROW_ID, DataType::UInt64, true)])
                .with_metadata(dataset.schema().metadata.clone());
        assert_eq!(batch.schema().as_ref(), &expected_schema,);

        let expected_row_ids: Vec<u64> = (0..200_u64).chain((1 << 32)..((1 << 32) + 200)).collect();
        let actual_row_ids: Vec<u64> = as_primitive_array::<UInt64Type>(batch.column(0).as_ref())
            .values()
            .iter()
            .copied()
            .collect();
        assert_eq!(expected_row_ids, actual_row_ids);
    }

    #[tokio::test]
    async fn test_scan_unordered_with_row_id() {
        // This test doesn't make sense for v2 files, there is no way to get an out-of-order scan
        let test_ds = TestVectorDataset::new(LanceFileVersion::Legacy, false)
            .await
            .unwrap();
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        scan.with_row_id();

        let ordered_batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();
        assert!(ordered_batches.len() > 2);
        let ordered_batch =
            concat_batches(&ordered_batches[0].schema(), ordered_batches.iter()).unwrap();

        // Attempt to get out-of-order scan, but that might take multiple attempts.
        scan.scan_in_order(false);
        for _ in 0..10 {
            let unordered_batches = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<RecordBatch>>()
                .await
                .unwrap();
            let unordered_batch =
                concat_batches(&unordered_batches[0].schema(), unordered_batches.iter()).unwrap();

            assert_eq!(ordered_batch.num_rows(), unordered_batch.num_rows());

            // If they aren't equal, they should be equal if we sort by row id
            if ordered_batch != unordered_batch {
                let sort_indices = sort_to_indices(&unordered_batch[ROW_ID], None, None).unwrap();

                let ordered_i = ordered_batch["i"].clone();
                let sorted_i = take::take(&unordered_batch["i"], &sort_indices, None).unwrap();

                assert_eq!(&ordered_i, &sorted_i);

                break;
            }
        }
    }

    #[tokio::test]
    async fn test_scan_with_wildcard() {
        let data = gen_batch()
            .col("x", array::step::<Float64Type>())
            .col("y", array::step::<Float64Type>())
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(100))
            .await
            .unwrap();

        let check_cols = async |projection: &[&str], expected_cols: &[&str]| {
            let mut scan = data.scan();
            scan.project(projection).unwrap();
            let stream = scan.try_into_stream().await.unwrap();
            let schema = stream.schema();
            let field_names = schema.field_names();
            assert_eq!(field_names, expected_cols);
        };

        check_cols(&["*"], &["x", "y"]).await;
        check_cols(&["x", "y"], &["x", "y"]).await;
        check_cols(&["x"], &["x"]).await;
        check_cols(&["_rowid", "*"], &["_rowid", "x", "y"]).await;
        check_cols(&["*", "_rowid"], &["x", "y", "_rowid"]).await;
        check_cols(
            &["_rowid", "*", "_rowoffset"],
            &["_rowid", "x", "y", "_rowoffset"],
        )
        .await;

        let check_exprs = async |exprs: &[&str], expected_cols: &[&str]| {
            let mut scan = data.scan();
            let projection = exprs
                .iter()
                .map(|e| (e.to_string(), e.to_string()))
                .collect::<Vec<_>>();
            scan.project_with_transform(&projection).unwrap();
            let stream = scan.try_into_stream().await.unwrap();
            let schema = stream.schema();
            let field_names = schema.field_names();
            assert_eq!(field_names, expected_cols);
        };

        // Make sure we can reference * fields in exprs and add new columns
        check_exprs(&["_rowid", "*", "x * 2"], &["_rowid", "x", "y", "x * 2"]).await;

        let check_fails = |projection: &[&str]| {
            let mut scan = data.scan();
            assert!(scan.project(projection).is_err());
        };

        // Would duplicate x
        check_fails(&["x", "*"]);
        check_fails(&["_rowid", "_rowid"]);
    }

    #[rstest]
    #[tokio::test]
    async fn test_scan_order(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            true,
        )]));

        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]))],
        )
        .unwrap();

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![6, 7, 8]))],
        )
        .unwrap();

        let params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };

        let write_batch = |batch: RecordBatch| async {
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            Dataset::write(reader, test_uri, Some(params)).await
        };

        write_batch.clone()(batch1.clone()).await.unwrap();
        write_batch(batch2.clone()).await.unwrap();

        let dataset = Arc::new(Dataset::open(test_uri).await.unwrap());
        let fragment1 = dataset.get_fragment(0).unwrap().metadata().clone();
        let fragment2 = dataset.get_fragment(1).unwrap().metadata().clone();

        // 1 then 2
        let mut scanner = dataset.scan();
        scanner.with_fragments(vec![fragment1.clone(), fragment2.clone()]);
        let output = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], batch1);
        assert_eq!(output[1], batch2);

        // 2 then 1
        let mut scanner = dataset.scan();
        scanner.with_fragments(vec![fragment2, fragment1]);
        let output = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], batch2);
        assert_eq!(output[1], batch1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_scan_sort(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = gen_batch()
            .col("int", array::cycle::<Int32Type>(vec![5, 4, 1, 2, 3]))
            .col(
                "str",
                array::cycle_utf8_literals(&["a", "b", "c", "e", "d"]),
            );

        let sorted_by_int = gen_batch()
            .col("int", array::cycle::<Int32Type>(vec![1, 2, 3, 4, 5]))
            .col(
                "str",
                array::cycle_utf8_literals(&["c", "e", "d", "b", "a"]),
            )
            .into_batch_rows(RowCount::from(5))
            .unwrap();

        let sorted_by_str = gen_batch()
            .col("int", array::cycle::<Int32Type>(vec![5, 4, 1, 3, 2]))
            .col(
                "str",
                array::cycle_utf8_literals(&["a", "b", "c", "d", "e"]),
            )
            .into_batch_rows(RowCount::from(5))
            .unwrap();

        Dataset::write(
            data.into_reader_rows(RowCount::from(5), BatchCount::from(1)),
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let dataset = Arc::new(Dataset::open(test_uri).await.unwrap());

        let batches_by_int = dataset
            .scan()
            .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
                "int".to_string(),
            )]))
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(batches_by_int[0], sorted_by_int);

        let batches_by_str = dataset
            .scan()
            .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
                "str".to_string(),
            )]))
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(batches_by_str[0], sorted_by_str);

        // Ensure an empty sort vec does not break anything (sorting is disabled)
        dataset
            .scan()
            .order_by(Some(vec![]))
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn test_sort_multi_columns(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = gen_batch()
            .col("int", array::cycle::<Int32Type>(vec![5, 5, 1, 1, 3]))
            .col(
                "float",
                array::cycle::<Float32Type>(vec![7.3, -f32::NAN, f32::NAN, 4.3, f32::INFINITY]),
            );

        let sorted_by_int_then_float = gen_batch()
            .col("int", array::cycle::<Int32Type>(vec![1, 1, 3, 5, 5]))
            .col(
                "float",
                // floats should be sorted using total order so -NAN is before all and NAN is after all
                array::cycle::<Float32Type>(vec![4.3, f32::NAN, f32::INFINITY, -f32::NAN, 7.3]),
            )
            .into_batch_rows(RowCount::from(5))
            .unwrap();

        Dataset::write(
            data.into_reader_rows(RowCount::from(5), BatchCount::from(1)),
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let dataset = Arc::new(Dataset::open(test_uri).await.unwrap());

        let batches_by_int_then_float = dataset
            .scan()
            .order_by(Some(vec![
                ColumnOrdering::asc_nulls_first("int".to_string()),
                ColumnOrdering::asc_nulls_first("float".to_string()),
            ]))
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(batches_by_int_then_float[0], sorted_by_int_then_float);
    }

    #[rstest]
    #[tokio::test]
    async fn test_ann_prefilter(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
        #[values(
            VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 2),
            VectorIndexParams::with_ivf_hnsw_sq_params(
                MetricType::L2,
                IvfBuildParams::new(2),
                HnswBuildParams::default(),
                SQBuildParams::default()
            )
        )]
        index_params: VectorIndexParams,
    ) {
        use lance_arrow::{FixedSizeListArrayExt, fixed_size_list_type};

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("filterable", DataType::Int32, true),
            ArrowField::new("vector", fixed_size_list_type(2, DataType::Float32), true),
        ]));

        let vector_values = Float32Array::from_iter_values((0..600).map(|x| x as f32));

        let batches = vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..300)),
                    Arc::new(FixedSizeListArray::try_new_from_values(vector_values, 2).unwrap()),
                ],
            )
            .unwrap(),
        ];

        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 300, // At least two files to make sure stable row ids make a difference
            enable_stable_row_ids: stable_row_ids,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &index_params, false)
            .await
            .unwrap();

        let query_key = Arc::new(Float32Array::from_iter_values((0..2).map(|x| x as f32)));
        let mut scan = dataset.scan();
        scan.filter("filterable > 5").unwrap();
        scan.nearest("vector", query_key.as_ref(), 1).unwrap();
        scan.minimum_nprobes(100);
        scan.ef(100);
        scan.with_row_id();

        let batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(batches.len(), 0);

        scan.prefilter(true);

        let batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);

        let first_match = batches[0][ROW_ID].as_primitive::<UInt64Type>().values()[0];

        // HNSW+SQ is an approximate index; this test validates *prefiltering*, so
        // every row failing `filterable > 5` (row ids 0..=5) must be excluded.
        // HNSW recall is covered by dedicated vector-index tests elsewhere.
        assert!(
            first_match > 5,
            "prefilter not honored: returned row id {first_match} should satisfy `filterable > 5`"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_filter_on_large_utf8(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "ls",
            DataType::LargeUtf8,
            true,
        )]));

        let batches = vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(LargeStringArray::from_iter_values(
                    (0..10).map(|v| format!("s-{}", v)),
                ))],
            )
            .unwrap(),
        ];

        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let mut scan = dataset.scan();
        scan.filter("ls = 's-8'").unwrap();

        let batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let batch = &batches[0];

        let expected = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(LargeStringArray::from_iter_values(
                (8..9).map(|v| format!("s-{}", v)),
            ))],
        )
        .unwrap();

        assert_eq!(batch, &expected);
    }

    #[rstest]
    #[tokio::test]
    async fn test_filter_with_regex(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "ls",
            DataType::Utf8,
            true,
        )]));

        let batches = vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(StringArray::from_iter_values(
                    (0..20).map(|v| format!("s-{}", v)),
                ))],
            )
            .unwrap(),
        ];

        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let mut scan = dataset.scan();
        scan.filter("regexp_match(ls, 's-1.')").unwrap();

        let stream = scan.try_into_stream().await.unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let batch = &batches[0];

        let expected = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from_iter_values(
                (10..=19).map(|v| format!("s-{}", v)),
            ))],
        )
        .unwrap();

        assert_eq!(batch, &expected);
    }

    #[tokio::test]
    async fn test_filter_proj_bug() {
        let struct_i_field = ArrowField::new("i", DataType::Int32, true);
        let struct_o_field = ArrowField::new("o", DataType::Utf8, true);
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(
                "struct",
                DataType::Struct(vec![struct_i_field.clone(), struct_o_field.clone()].into()),
                true,
            ),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let input_batches: Vec<RecordBatch> = (0..5)
            .map(|i| {
                let struct_i_arr: Arc<Int32Array> =
                    Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20));
                let struct_o_arr: Arc<StringArray> = Arc::new(StringArray::from_iter_values(
                    (i * 20..(i + 1) * 20).map(|v| format!("o-{:02}", v)),
                ));
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(StructArray::from(vec![
                            (Arc::new(struct_i_field.clone()), struct_i_arr as ArrayRef),
                            (Arc::new(struct_o_field.clone()), struct_o_arr as ArrayRef),
                        ])),
                        Arc::new(StringArray::from_iter_values(
                            (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();
        let batches =
            RecordBatchIterator::new(input_batches.clone().into_iter().map(Ok), schema.clone());
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(LanceFileVersion::Legacy),
            ..Default::default()
        };
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let batches = dataset
            .scan()
            .filter("struct.i >= 20")
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let batch = concat_batches(&batches[0].schema(), &batches).unwrap();

        let expected_batch = concat_batches(&schema, &input_batches.as_slice()[1..]).unwrap();
        assert_eq!(batch, expected_batch);

        // different order
        let batches = dataset
            .scan()
            .filter("struct.o >= 'o-20'")
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let batch = concat_batches(&batches[0].schema(), &batches).unwrap();
        assert_eq!(batch, expected_batch);

        // other reported bug with nested top level column access
        let batches = dataset
            .scan()
            .project(vec!["struct"].as_slice())
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        concat_batches(&batches[0].schema(), &batches).unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn test_ann_with_deletion(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let vec_params = vec![
            // TODO: re-enable diskann test when we can tune to get reproducible results.
            // VectorIndexParams::with_diskann_params(MetricType::L2, DiskANNParams::new(10, 1.5, 10)),
            VectorIndexParams::ivf_pq(4, 8, 2, MetricType::L2, 2),
        ];
        for params in vec_params {
            use lance_arrow::FixedSizeListArrayExt;

            let test_dir = TempStrDir::default();
            let test_uri = &test_dir;

            // make dataset
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("i", DataType::Int32, true),
                ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        32,
                    ),
                    true,
                ),
            ]));

            // vectors are [1, 1, 1, ...] [2, 2, 2, ...]
            let vector_values: Float32Array =
                (0..32 * 512).map(|v| (v / 32) as f32 + 1.0).collect();
            let vectors = FixedSizeListArray::try_new_from_values(vector_values, 32).unwrap();

            let batches = vec![
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(0..512)),
                        Arc::new(vectors.clone()),
                    ],
                )
                .unwrap(),
            ];

            let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
            let mut dataset = Dataset::write(
                reader,
                test_uri,
                Some(WriteParams {
                    data_storage_version: Some(data_storage_version),
                    enable_stable_row_ids: stable_row_ids,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

            assert_eq!(dataset.index_cache_entry_count().await, 0);
            dataset
                .create_index(
                    &["vec"],
                    IndexType::Vector,
                    Some("idx".to_string()),
                    &params,
                    true,
                )
                .await
                .unwrap();

            let mut scan = dataset.scan();
            // closest be i = 0..5
            let key: Float32Array = (0..32).map(|_v| 1.0_f32).collect();
            scan.nearest("vec", &key, 5).unwrap();
            scan.refine(100);
            scan.minimum_nprobes(100);

            assert_eq!(
                dataset.index_cache_entry_count().await,
                2, // 2 for index metadata at version 1 and 2.
            );
            let results = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            assert_eq!(
                dataset.index_cache_entry_count().await,
                5 + dataset.versions().await.unwrap().len()
            );
            assert_eq!(results.len(), 1);
            let batch = &results[0];

            let expected_i = BTreeSet::from_iter(vec![0, 1, 2, 3, 4]);
            let column_i = batch.column_by_name("i").unwrap();
            let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
                .values()
                .iter()
                .copied()
                .collect();
            assert_eq!(expected_i, actual_i);

            // DELETE top result and search again

            dataset.delete("i = 1").await.unwrap();
            let mut scan = dataset.scan();
            scan.nearest("vec", &key, 5).unwrap();
            scan.refine(100);
            scan.minimum_nprobes(100);

            let results = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            assert_eq!(results.len(), 1);
            let batch = &results[0];

            // i=1 was deleted, and 5 is the next best, the reset shouldn't change
            let expected_i = BTreeSet::from_iter(vec![0, 2, 3, 4, 5]);
            let column_i = batch.column_by_name("i").unwrap();
            let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
                .values()
                .iter()
                .copied()
                .collect();
            assert_eq!(expected_i, actual_i);

            // Add a second fragment and test the case where there are no deletion
            // files but there are missing fragments.
            let batches = vec![
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(512..1024)),
                        Arc::new(vectors),
                    ],
                )
                .unwrap(),
            ];

            let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
            let mut dataset = Dataset::write(
                reader,
                test_uri,
                Some(WriteParams {
                    mode: WriteMode::Append,
                    data_storage_version: Some(data_storage_version),
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
            dataset
                .create_index(
                    &["vec"],
                    IndexType::Vector,
                    Some("idx".to_string()),
                    &params,
                    true,
                )
                .await
                .unwrap();

            dataset.delete("i < 512").await.unwrap();

            let mut scan = dataset.scan();
            scan.nearest("vec", &key, 5).unwrap();
            scan.refine(100);
            scan.minimum_nprobes(100);

            let results = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            assert_eq!(results.len(), 1);
            let batch = &results[0];

            // It should not pick up any results from the first fragment
            let expected_i = BTreeSet::from_iter(vec![512, 513, 514, 515, 516]);
            let column_i = batch.column_by_name("i").unwrap();
            let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
                .values()
                .iter()
                .copied()
                .collect();
            assert_eq!(expected_i, actual_i);
        }
    }

    #[tokio::test]
    async fn test_projection_order() {
        let vec_params = VectorIndexParams::ivf_pq(4, 8, 2, MetricType::L2, 2);
        let mut data = gen_batch()
            .col("vec", array::rand_vec::<Float32Type>(Dimension::from(4)))
            .col("text", array::rand_utf8(ByteCount::from(10), false))
            .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(100))
            .await
            .unwrap();
        data.create_index(&["vec"], IndexType::Vector, None, &vec_params, true)
            .await
            .unwrap();

        let mut scan = data.scan();
        scan.nearest("vec", &Float32Array::from(vec![1.0, 1.0, 1.0, 1.0]), 5)
            .unwrap();
        // Explicitly request _distance to verify column ordering
        scan.with_row_id().project(&["text", "_distance"]).unwrap();

        let results = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(
            results[0].schema().field_names(),
            vec!["text", "_distance", "_rowid"]
        );

        // Without explicit _distance, it should not be included
        let mut scan = data.scan();
        scan.nearest("vec", &Float32Array::from(vec![1.0, 1.0, 1.0, 1.0]), 5)
            .unwrap();
        scan.with_row_id().project(&["text"]).unwrap();

        let results = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(results[0].schema().field_names(), vec!["text", "_rowid"]);
    }

    #[rstest]
    #[tokio::test]
    async fn test_count_rows_with_filter(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let mut data_gen = BatchGenerator::new().col(Box::new(
            IncrementingInt32::new().named("Filter_me".to_owned()),
        ));
        Dataset::write(
            data_gen.batch(32),
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(32, dataset.count_rows(None).await.unwrap());
        assert_eq!(
            16,
            dataset
                .count_rows(Some("`Filter_me` > 15".to_string()))
                .await
                .unwrap()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_dynamic_projection(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("i".to_owned())));
        Dataset::write(
            data_gen.batch(32),
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 32);

        let mut scanner = dataset.scan();

        let scan_res = scanner
            .project_with_transform(&[("bool", "i > 15")])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(1, scan_res.num_columns());

        let bool_col = scan_res
            .column_by_name("bool")
            .expect("bool column should exist");
        let bool_arr = bool_col.as_boolean();
        for i in 0..32 {
            if i > 15 {
                assert!(bool_arr.value(i));
            } else {
                assert!(!bool_arr.value(i));
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_column_casting_function(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let mut data_gen =
            BatchGenerator::new().col(Box::new(RandomVector::new().named("vec".to_owned())));
        Dataset::write(
            data_gen.batch(32),
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 32);

        let mut scanner = dataset.scan();

        let scan_res = scanner
            .project_with_transform(&[("f16", "_cast_list_f16(vec)")])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(1, scan_res.num_columns());
        assert_eq!(32, scan_res.num_rows());
        assert_eq!("f16", scan_res.schema().field(0).name());

        let mut scanner = dataset.scan();
        let scan_res_original = scanner
            .project(&["vec"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let f32_col: &Float32Array = scan_res_original
            .column_by_name("vec")
            .unwrap()
            .as_fixed_size_list()
            .values()
            .as_primitive();
        let f16_col: &Float16Array = scan_res
            .column_by_name("f16")
            .unwrap()
            .as_fixed_size_list()
            .values()
            .as_primitive();

        for (f32_val, f16_val) in f32_col.iter().zip(f16_col.iter()) {
            let f32_val = f32_val.unwrap();
            let f16_val = f16_val.unwrap();
            assert_eq!(f16::from_f32(f32_val), f16_val);
        }
    }

    struct ScalarIndexTestFixture {
        _test_dir: TempStrDir,
        dataset: Dataset,
        sample_query: Arc<dyn Array>,
        delete_query: Arc<dyn Array>,
        // The original version of the data, two fragments, rows 0-1000
        original_version: u64,
        // The original version of the data, 1 row deleted, compacted to a single fragment
        compact_version: u64,
        // The original version of the data + an extra 1000 unindexed
        append_version: u64,
        // The original version of the data + an extra 1000 rows, with indices updated so all rows indexed
        updated_version: u64,
        // The original version of the data with 1 deleted row
        delete_version: u64,
        // The original version of the data + an extra 1000 uindexed + 1 deleted row
        append_then_delete_version: u64,
    }

    #[derive(Debug, PartialEq)]
    struct ScalarTestParams {
        use_index: bool,
        use_projection: bool,
        use_deleted_data: bool,
        use_new_data: bool,
        with_row_id: bool,
        use_compaction: bool,
        use_updated: bool,
    }

    impl ScalarIndexTestFixture {
        async fn new(data_storage_version: LanceFileVersion, use_stable_row_ids: bool) -> Self {
            let test_dir = TempStrDir::default();
            let test_uri = &test_dir;

            // Write 1000 rows.  Train indices.  Then write 1000 new rows with the same vector data.
            // Then delete a row from the trained data.
            //
            // The first row where indexed == 50 is our sample query.
            // The first row where indexed == 75 is our deleted row (and delete query)
            let data = gen_batch()
                .col(
                    "vector",
                    array::rand_vec::<Float32Type>(Dimension::from(32)),
                )
                .col("indexed", array::step::<Int32Type>())
                .col("not_indexed", array::step::<Int32Type>())
                .into_batch_rows(RowCount::from(1000))
                .unwrap();

            // Write as two batches so we can later compact
            let mut dataset = Dataset::write(
                RecordBatchIterator::new(vec![Ok(data.clone())], data.schema().clone()),
                test_uri,
                Some(WriteParams {
                    max_rows_per_file: 500,
                    data_storage_version: Some(data_storage_version),
                    enable_stable_row_ids: use_stable_row_ids,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

            dataset
                .create_index(
                    &["vector"],
                    IndexType::Vector,
                    None,
                    &VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 2),
                    false,
                )
                .await
                .unwrap();

            dataset
                .create_index(
                    &["indexed"],
                    IndexType::Scalar,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();

            let original_version = dataset.version().version;
            let sample_query = data["vector"].as_fixed_size_list().value(50);
            let delete_query = data["vector"].as_fixed_size_list().value(75);

            // APPEND DATA

            // Re-use the vector column in the new batch but add 1000 to the indexed/not_indexed columns so
            // they are distinct.  This makes our checks easier.
            let new_indexed =
                arrow_arith::numeric::add(&data["indexed"], &Int32Array::new_scalar(1000)).unwrap();
            let new_not_indexed =
                arrow_arith::numeric::add(&data["indexed"], &Int32Array::new_scalar(1000)).unwrap();
            let append_data = RecordBatch::try_new(
                data.schema(),
                vec![data["vector"].clone(), new_indexed, new_not_indexed],
            )
            .unwrap();

            dataset
                .append(
                    RecordBatchIterator::new(vec![Ok(append_data)], data.schema()),
                    Some(WriteParams {
                        data_storage_version: Some(data_storage_version),
                        ..Default::default()
                    }),
                )
                .await
                .unwrap();

            let append_version = dataset.version().version;

            // UPDATE

            dataset
                .optimize_indices(&OptimizeOptions::merge(1))
                .await
                .unwrap();
            let updated_version = dataset.version().version;

            // APPEND -> DELETE

            dataset.checkout_version(append_version).await.unwrap();
            dataset.restore().await.unwrap();

            dataset.delete("not_indexed = 75").await.unwrap();

            let append_then_delete_version = dataset.version().version;

            // DELETE

            let mut dataset = dataset.checkout_version(original_version).await.unwrap();
            dataset.restore().await.unwrap();

            dataset.delete("not_indexed = 75").await.unwrap();

            let delete_version = dataset.version().version;

            // COMPACT (this should materialize the deletion)
            compact_files(&mut dataset, CompactionOptions::default(), None)
                .await
                .unwrap();
            let compact_version = dataset.version().version;
            dataset.checkout_version(original_version).await.unwrap();
            dataset.restore().await.unwrap();

            Self {
                _test_dir: test_dir,
                dataset,
                sample_query,
                delete_query,
                original_version,
                compact_version,
                append_version,
                updated_version,
                delete_version,
                append_then_delete_version,
            }
        }

        fn sample_query(&self) -> &PrimitiveArray<Float32Type> {
            self.sample_query.as_primitive::<Float32Type>()
        }

        fn delete_query(&self) -> &PrimitiveArray<Float32Type> {
            self.delete_query.as_primitive::<Float32Type>()
        }

        async fn get_dataset(&self, params: &ScalarTestParams) -> Dataset {
            let version = if params.use_compaction {
                // These combinations should not be possible
                if params.use_deleted_data || params.use_new_data || params.use_updated {
                    panic!(
                        "There is no test data combining new/deleted/updated data with compaction"
                    );
                } else {
                    self.compact_version
                }
            } else if params.use_updated {
                // These combinations should not be possible
                if params.use_deleted_data || params.use_new_data || params.use_compaction {
                    panic!(
                        "There is no test data combining updated data with new/deleted/compaction"
                    );
                } else {
                    self.updated_version
                }
            } else {
                match (params.use_new_data, params.use_deleted_data) {
                    (false, false) => self.original_version,
                    (false, true) => self.delete_version,
                    (true, false) => self.append_version,
                    (true, true) => self.append_then_delete_version,
                }
            };
            self.dataset.checkout_version(version).await.unwrap()
        }

        async fn run_query(
            &self,
            query: &str,
            vector: Option<&PrimitiveArray<Float32Type>>,
            params: &ScalarTestParams,
        ) -> (String, RecordBatch) {
            let dataset = self.get_dataset(params).await;
            let mut scan = dataset.scan();
            if let Some(vector) = vector {
                scan.nearest("vector", vector, 10).unwrap();
            }
            if params.use_projection {
                scan.project(&["indexed"]).unwrap();
            }
            if params.with_row_id {
                scan.with_row_id();
            }
            scan.scan_in_order(true);
            scan.use_index(params.use_index);
            scan.filter(query).unwrap();
            scan.prefilter(true);

            let plan = scan.explain_plan(true).await.unwrap();
            let batch = scan.try_into_batch().await.unwrap();

            if params.use_projection {
                // 1 projected column ("indexed"); _distance is not auto-added
                // for explicit projections.
                let mut expected_columns = 1;
                if params.with_row_id {
                    expected_columns += 1;
                }
                assert_eq!(batch.num_columns(), expected_columns);
            } else {
                let mut expected_columns = 3;
                if vector.is_some() {
                    // distance column
                    expected_columns += 1;
                }
                if params.with_row_id {
                    expected_columns += 1;
                }
                // vector, indexed, not_indexed, _distance
                assert_eq!(batch.num_columns(), expected_columns);
            }

            (plan, batch)
        }

        fn assert_none<F: Fn(i32) -> bool>(
            &self,
            batch: &RecordBatch,
            predicate: F,
            message: &str,
        ) {
            let indexed = batch["indexed"].as_primitive::<Int32Type>();
            if indexed.iter().map(|val| val.unwrap()).any(predicate) {
                panic!("{}", message);
            }
        }

        fn assert_one<F: Fn(i32) -> bool>(&self, batch: &RecordBatch, predicate: F, message: &str) {
            let indexed = batch["indexed"].as_primitive::<Int32Type>();
            if !indexed.iter().map(|val| val.unwrap()).any(predicate) {
                panic!("{}", message);
            }
        }

        async fn check_vector_scalar_indexed_and_refine(&self, params: &ScalarTestParams) {
            let (query_plan, batch) = self
                .run_query(
                    "indexed != 50 AND ((not_indexed < 100) OR (not_indexed >= 1000 AND not_indexed < 1100))",
                    Some(self.sample_query()),
                    params,
                )
                .await;
            // Materialization is always required if there is a refine
            if self.dataset.is_legacy_storage() {
                assert!(query_plan.contains("MaterializeIndex"));
            }
            // The result should not include the sample query
            self.assert_none(
                &batch,
                |val| val == 50,
                "The query contained 50 even though it was filtered",
            );
            if !params.use_new_data {
                // Refine should have been applied
                self.assert_none(
                    &batch,
                    |val| (100..1000).contains(&val) || (val >= 1100),
                    "The non-indexed refine filter was not applied",
                );
            }

            // If there is new data then the dupe of row 50 should be in the results
            if params.use_new_data || params.use_updated {
                self.assert_one(
                    &batch,
                    |val| val == 1050,
                    "The query did not contain 1050 from the new data",
                );
            }
        }

        async fn check_vector_scalar_indexed_only(&self, params: &ScalarTestParams) {
            let (query_plan, batch) = self
                .run_query("indexed != 50", Some(self.sample_query()), params)
                .await;
            if self.dataset.is_legacy_storage() {
                if params.use_index {
                    // An ANN search whose prefilter is fully satisfied by the index should be
                    // able to use a ScalarIndexQuery
                    assert!(query_plan.contains("ScalarIndexQuery"));
                } else {
                    // A KNN search requires materialization of the index
                    assert!(query_plan.contains("MaterializeIndex"));
                }
            }
            // The result should not include the sample query
            self.assert_none(
                &batch,
                |val| val == 50,
                "The query contained 50 even though it was filtered",
            );
            // If there is new data then the dupe of row 50 should be in the results
            if params.use_new_data {
                self.assert_one(
                    &batch,
                    |val| val == 1050,
                    "The query did not contain 1050 from the new data",
                );
                if !params.use_new_data {
                    // Let's also make sure our filter can target something in the new data only
                    let (_, batch) = self
                        .run_query("indexed == 1050", Some(self.sample_query()), params)
                        .await;
                    assert_eq!(batch.num_rows(), 1);
                }
            }
            if params.use_deleted_data {
                let (_, batch) = self
                    .run_query("indexed == 75", Some(self.delete_query()), params)
                    .await;
                if !params.use_new_data {
                    assert_eq!(batch.num_rows(), 0);
                }
            }
        }

        async fn check_vector_queries(&self, params: &ScalarTestParams) {
            self.check_vector_scalar_indexed_only(params).await;
            self.check_vector_scalar_indexed_and_refine(params).await;
        }

        async fn check_simple_indexed_only(&self, params: &ScalarTestParams) {
            let (query_plan, batch) = self.run_query("indexed != 50", None, params).await;
            // Materialization is always required for non-vector search
            if self.dataset.is_legacy_storage() {
                assert!(query_plan.contains("MaterializeIndex"));
            } else {
                assert!(query_plan.contains("LanceRead"));
            }
            // The result should not include the targeted row
            self.assert_none(
                &batch,
                |val| val == 50,
                "The query contained 50 even though it was filtered",
            );
            let mut expected_num_rows = if params.use_new_data || params.use_updated {
                1999
            } else {
                999
            };
            if params.use_deleted_data || params.use_compaction {
                expected_num_rows -= 1;
            }
            assert_eq!(batch.num_rows(), expected_num_rows);

            // Let's also make sure our filter can target something in the new data only
            if params.use_new_data || params.use_updated {
                let (_, batch) = self.run_query("indexed == 1050", None, params).await;
                assert_eq!(batch.num_rows(), 1);
            }

            // Also make sure we don't return deleted data
            if params.use_deleted_data || params.use_compaction {
                let (_, batch) = self.run_query("indexed == 75", None, params).await;
                assert_eq!(batch.num_rows(), 0);
            }
        }

        async fn check_simple_indexed_and_refine(&self, params: &ScalarTestParams) {
            let (query_plan, batch) = self.run_query(
                "indexed != 50 AND ((not_indexed < 100) OR (not_indexed >= 1000 AND not_indexed < 1100))",
                None,
                params
            ).await;
            // Materialization is always required for non-vector search
            if self.dataset.is_legacy_storage() {
                assert!(query_plan.contains("MaterializeIndex"));
            } else {
                assert!(query_plan.contains("LanceRead"));
            }
            // The result should not include the targeted row
            self.assert_none(
                &batch,
                |val| val == 50,
                "The query contained 50 even though it was filtered",
            );
            // The refine should be applied
            self.assert_none(
                &batch,
                |val| (100..1000).contains(&val) || (val >= 1100),
                "The non-indexed refine filter was not applied",
            );

            let mut expected_num_rows = if params.use_new_data || params.use_updated {
                199
            } else {
                99
            };
            if params.use_deleted_data || params.use_compaction {
                expected_num_rows -= 1;
            }
            assert_eq!(batch.num_rows(), expected_num_rows);
        }

        async fn check_simple_queries(&self, params: &ScalarTestParams) {
            self.check_simple_indexed_only(params).await;
            self.check_simple_indexed_and_refine(params).await;
        }
    }

    // There are many different ways that a query can be run and they all have slightly different
    // effects on the plan that gets built.  This test attempts to run the same queries in various
    // different configurations to ensure that we get consistent results
    #[rstest]
    #[tokio::test]
    async fn test_secondary_index_scans(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] use_stable_row_ids: bool,
    ) {
        let fixture = Box::pin(ScalarIndexTestFixture::new(
            data_storage_version,
            use_stable_row_ids,
        ))
        .await;

        for use_index in [false, true] {
            for use_projection in [false, true] {
                for use_deleted_data in [false, true] {
                    for use_new_data in [false, true] {
                        // Don't test compaction in conjunction with deletion and new data, it's too
                        // many combinations with no clear benefit.  Feel free to update if there is
                        // a need
                        // TODO: enable compaction for stable row id once supported.
                        let compaction_choices =
                            if use_deleted_data || use_new_data || use_stable_row_ids {
                                vec![false]
                            } else {
                                vec![false, true]
                            };
                        for use_compaction in compaction_choices {
                            let updated_choices =
                                if use_deleted_data || use_new_data || use_compaction {
                                    vec![false]
                                } else {
                                    vec![false, true]
                                };
                            for use_updated in updated_choices {
                                for with_row_id in [false, true] {
                                    let params = ScalarTestParams {
                                        use_index,
                                        use_projection,
                                        use_deleted_data,
                                        use_new_data,
                                        with_row_id,
                                        use_compaction,
                                        use_updated,
                                    };
                                    fixture.check_vector_queries(&params).await;
                                    fixture.check_simple_queries(&params).await;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn can_filter_row_id() {
        let dataset = lance_datagen::gen_batch()
            .col("x", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(1000))
            .await
            .unwrap();

        let mut scan = dataset.scan();
        scan.with_row_id();
        scan.project::<&str>(&[]).unwrap();
        scan.filter("_rowid == 50").unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.column(0).as_primitive::<UInt64Type>().values()[0], 50);
    }

    #[rstest]
    #[tokio::test]
    async fn test_index_take_batch_size() {
        let fixture = Box::pin(ScalarIndexTestFixture::new(LanceFileVersion::Stable, false)).await;
        let stream = fixture
            .dataset
            .scan()
            .filter("indexed > 0")
            .unwrap()
            .batch_size(16)
            .try_into_stream()
            .await
            .unwrap();
        let batches = stream.collect::<Vec<_>>().await;
        assert_eq!(batches.len(), 1000_usize.div_ceil(16));
    }

    /// Assert that the plan when formatted matches the expected string.
    ///
    /// Within expected, you can use `...` to match any number of characters.
    async fn assert_plan_equals(
        dataset: &Dataset,
        plan: impl Fn(&mut Scanner) -> Result<&mut Scanner>,
        expected: &str,
    ) -> Result<()> {
        let mut scan = dataset.scan();
        // Pin target_parallelism=1 so EnforceDistribution produces deterministic plans
        // regardless of the machine's CPU count.
        scan.target_parallelism(1);
        plan(&mut scan)?;
        let exec_plan = scan.create_plan().await?;
        assert_plan_node_equals(exec_plan, expected).await
    }

    #[tokio::test]
    async fn test_inexact_scalar_index_plans() {
        let data = gen_batch()
            .col("ngram", array::rand_utf8(ByteCount::from(5), false))
            .col("exact", array::rand_type(&DataType::UInt32))
            .col("no_index", array::rand_type(&DataType::UInt32))
            .into_reader_rows(RowCount::from(1000), BatchCount::from(5));

        let mut dataset = Dataset::write(data, "memory://test", None).await.unwrap();
        dataset
            .create_index(
                &["ngram"],
                IndexType::NGram,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["exact"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Simple in-exact filter
        assert_plan_equals(
            &dataset,
            |scanner| scanner.filter("contains(ngram, 'test string')"),
            "LanceRead: uri=..., projection=[ngram, exact, no_index], num_fragments=1, \
             range_before=None, range_after=None, row_id=false, row_addr=false, \
             full_filter=contains(ngram, Utf8(\"test string\")), refine_filter=--
               ScalarIndexQuery: query=[contains(ngram, Utf8(\"test string\"))]@ngram_idx(NGram)",
        )
        .await
        .unwrap();

        // Combined with exact filter
        assert_plan_equals(
            &dataset,
            |scanner| scanner.filter("contains(ngram, 'test string') and exact < 50"),
            "LanceRead: uri=..., projection=[ngram, exact, no_index], num_fragments=1, \
            range_before=None, range_after=None, row_id=false, row_addr=false, \
            full_filter=contains(ngram, Utf8(\"test string\")) AND exact < UInt32(50), \
            refine_filter=--
              ScalarIndexQuery: query=AND([contains(ngram, Utf8(\"test string\"))]@ngram_idx(NGram),[exact < 50]@exact_idx(BTree))",
        )
        .await
        .unwrap();

        // All three filters
        assert_plan_equals(
            &dataset,
            |scanner| {
                scanner.filter("contains(ngram, 'test string') and exact < 50 AND no_index > 100")
            },
            "ProjectionExec: expr=[ngram@0 as ngram, exact@1 as exact, no_index@2 as no_index]
  LanceRead: uri=..., projection=[ngram, exact, no_index], num_fragments=1, range_before=None, \
  range_after=None, row_id=true, row_addr=false, full_filter=contains(ngram, Utf8(\"test string\")) AND exact < UInt32(50) AND no_index > UInt32(100), \
  refine_filter=no_index > UInt32(100)
    ScalarIndexQuery: query=AND([contains(ngram, Utf8(\"test string\"))]@ngram_idx(NGram),[exact < 50]@exact_idx(BTree))",
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_ngram_regex_index_scan() {
        use arrow::array::AsArray;

        // A small, fixed corpus written across multiple fragments so the ngram
        // index spans fragment boundaries.
        let values = [
            "rhino",       // 0
            "rhinos nose", // 1
            "cat",         // 2
            "dog",         // 3
            "cat dog",     // 4
            "elephant",    // 5
            "catalog",     // 6
            "scatter",     // 7
            "rhino horn",  // 8
            "mouse",       // 9
            "category",    // 10
            "dogma",       // 11
        ];
        let array = StringArray::from_iter_values(values);
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let write_params = WriteParams {
            max_rows_per_file: 4, // 12 rows -> 3 fragments
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, "memory://test_ngram_regex", Some(write_params))
            .await
            .unwrap();
        dataset
            .create_index(
                &["s"],
                IndexType::NGram,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        assert!(
            dataset.get_fragments().len() > 1,
            "expected a multi-fragment dataset"
        );

        // Scan with `filter` and return the matched `s` values, sorted.
        async fn matched(dataset: &Dataset, filter: &str) -> Vec<String> {
            let mut scan = dataset.scan();
            scan.filter(filter).unwrap();
            let batches = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            let mut out = Vec::new();
            for batch in batches {
                let col = batch.column_by_name("s").unwrap().as_string::<i32>();
                out.extend(col.iter().flatten().map(|s| s.to_string()));
            }
            out.sort();
            out
        }

        // `regexp_like`: a plain literal substring.
        assert_eq!(
            matched(&dataset, "regexp_like(s, 'rhino')").await,
            ["rhino", "rhino horn", "rhinos nose"]
        );
        // `regexp_match` (coerced to `IsNotNull(regexp_match(...))`) accelerates too.
        assert_eq!(
            matched(&dataset, "regexp_match(s, 'rhino')").await,
            ["rhino", "rhino horn", "rhinos nose"]
        );
        // Anchored: recheck must drop trigram false positives -- the `cat`
        // trigram also occurs in cat dog / catalog / scatter / category.
        assert_eq!(matched(&dataset, "regexp_like(s, 'cat$')").await, ["cat"]);
        // AND across `.*`: row 8 ("rhino horn") shares the rhino trigrams but
        // lacks the nose trigrams, so only "rhinos nose" survives.
        assert_eq!(
            matched(&dataset, "regexp_like(s, 'rhino.*nose')").await,
            ["rhinos nose"]
        );
        // Alternation.
        assert_eq!(
            matched(&dataset, "regexp_like(s, '(catalog|elephant)')").await,
            ["catalog", "elephant"]
        );
        // A non-accelerable pattern (no trigram derivable) still returns correct
        // results via a full recheck.
        assert_eq!(matched(&dataset, "regexp_like(s, 'o.m')").await, ["dogma"]);
        // A case-insensitive flag is not accelerated (the index normalization
        // disagrees with Unicode case folding) but must still return correct
        // results via a full recheck -- here matching despite the upper-case
        // pattern. This exercises the three-argument `regexp_like` flags path.
        assert_eq!(
            matched(&dataset, "regexp_like(s, 'RHINO', 'i')").await,
            ["rhino", "rhino horn", "rhinos nose"]
        );

        // Infix LIKE is accelerated through the same machinery (a plain-literal
        // `regexp_like` is rewritten to LIKE before it reaches the index).
        assert_eq!(
            matched(&dataset, "s LIKE '%rhino%'").await,
            ["rhino", "rhino horn", "rhinos nose"]
        );
        // Prefix LIKE: recheck drops "scatter", which contains the `cat` trigram
        // but does not start with "cat".
        assert_eq!(
            matched(&dataset, "s LIKE 'cat%'").await,
            ["cat", "cat dog", "catalog", "category"]
        );

        // The ngram index is actually engaged for every accelerated form.
        for filter in [
            "regexp_like(s, 'rhino')",
            "regexp_match(s, 'rhino')",
            "s LIKE '%rhino%'",
        ] {
            let mut scan = dataset.scan();
            scan.filter(filter).unwrap();
            let plan = scan.create_plan().await.unwrap();
            let plan_str = format!(
                "{}",
                datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
            );
            assert!(
                plan_str.contains("ScalarIndexQuery") && plan_str.contains("NGram"),
                "expected ngram index usage for `{filter}`, got plan:\n{plan_str}"
            );
        }
    }

    #[tokio::test]
    async fn test_ngram_regex_non_accelerable_recheck() {
        // `a.b` yields no trigram, so the index returns "recheck everything".
        // This must still produce ALL correct matches across fragments, not an
        // empty set (a regression test for the AtLeast recheck path, which a
        // single-match case would not catch).
        let unit = ["acb", "dog", "axb", "cat", "qqq", "rhino"];
        let values: Vec<&str> = unit.iter().copied().cycle().take(60).collect();
        let array = StringArray::from_iter_values(values);
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "text",
            DataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let write_params = WriteParams {
            max_rows_per_file: 20, // 60 rows -> 3 fragments
            ..Default::default()
        };
        let mut dataset =
            Dataset::write(reader, "memory://test_ngram_regex_ne", Some(write_params))
                .await
                .unwrap();
        dataset
            .create_index(
                &["text"],
                IndexType::NGram,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        async fn count(dataset: &Dataset, filter: &str) -> usize {
            let mut scan = dataset.scan();
            scan.filter(filter).unwrap();
            let batches = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            batches.iter().map(|b| b.num_rows()).sum()
        }

        // "acb" and "axb" each appear 10 times in the 60 rows -> 20 matches.
        assert_eq!(count(&dataset, "regexp_match(text, 'a.b')").await, 20);
        assert_eq!(count(&dataset, "regexp_like(text, 'a.b')").await, 20);
    }

    #[tokio::test]
    async fn test_like_prefix_with_btree_index() {
        // Create dataset with string data that has various prefixes
        // Avoid LIKE special characters (%, _) in data to keep tests simple
        let data = gen_batch()
            .col(
                "name",
                array::cycle_utf8_literals(&[
                    "apple",
                    "application",
                    "app",
                    "banana",
                    "band",
                    "testns1",
                    "testns2",
                    "test",
                    "testing",
                    "zoo",
                ]),
            )
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));

        let mut dataset = Dataset::write(data, "memory://test_like", None)
            .await
            .unwrap();

        // Create BTree index on string column
        dataset
            .create_index(
                &["name"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Test 1: Verify LIKE 'app%' uses scalar index and returns correct results
        assert_plan_equals(
            &dataset,
            |scanner| scanner.filter("name LIKE 'app%'"),
            "LanceRead: uri=..., projection=[name, id], num_fragments=1, \
             range_before=None, range_after=None, row_id=false, row_addr=false, \
             full_filter=name LIKE Utf8(\"app%\"), refine_filter=--
               ScalarIndexQuery: query=[name LIKE 'app%']@name_idx(BTree)",
        )
        .await
        .unwrap();

        // Verify correct results for LIKE 'app%'
        let results = dataset
            .scan()
            .filter("name LIKE 'app%'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let names: Vec<&str> = results
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect();
        // Should match: apple, application, app (repeated in cycle)
        assert!(names.iter().all(|n| n.starts_with("app")));
        assert!(!names.is_empty());

        // Test 2: Verify starts_with() uses scalar index (simple prefix without special chars)
        // Note: DataFusion optimizes starts_with() to LIKE before our index planning
        assert_plan_equals(
            &dataset,
            |scanner| scanner.filter("starts_with(name, 'ban')"),
            "LanceRead: uri=..., projection=[name, id], num_fragments=1, \
             range_before=None, range_after=None, row_id=false, row_addr=false, \
             full_filter=name LIKE Utf8(\"ban%\"), refine_filter=--
               ScalarIndexQuery: query=[name LIKE 'ban%']@name_idx(BTree)",
        )
        .await
        .unwrap();

        // Verify correct results for starts_with
        let results = dataset
            .scan()
            .filter("starts_with(name, 'ban')")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let names: Vec<&str> = results
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect();
        // Should match: banana, band
        assert!(names.iter().all(|n| n.starts_with("ban")));
        assert!(!names.is_empty());

        // Test 3: LIKE with pattern requiring refine (e.g., 'test%2')
        assert_plan_equals(
            &dataset,
            |scanner| scanner.filter("name LIKE 'test%2'"),
            "ProjectionExec: expr=[name@0 as name, id@1 as id]
  LanceRead: uri=..., projection=[name, id], num_fragments=1, \
range_before=None, range_after=None, row_id=true, row_addr=false, \
full_filter=name LIKE Utf8(\"test%2\"), refine_filter=name LIKE Utf8(\"test%2\")
    ScalarIndexQuery: query=[name LIKE 'test%']@name_idx(BTree)",
        )
        .await
        .unwrap();

        // Verify correct results for LIKE 'test%2' (needs refine)
        let results = dataset
            .scan()
            .filter("name LIKE 'test%2'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let names: Vec<&str> = results
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect();
        // Should match: testns2 (ends with '2')
        assert!(
            names
                .iter()
                .all(|n| n.starts_with("test") && n.ends_with("2"))
        );

        // Test 4: LIKE starting with wildcard should NOT use scalar index for pruning
        // Verify by checking the plan does NOT have ScalarIndexQuery
        let mut scanner = dataset.scan();
        scanner.filter("name LIKE '%app%'").unwrap();
        let plan = scanner.create_plan().await.unwrap();
        let plan_str = format!("{:?}", plan);
        assert!(
            !plan_str.contains("ScalarIndexQuery"),
            "LIKE '%app%' should not use scalar index, but got: {}",
            plan_str
        );

        // Verify correct results for LIKE '%app%'
        let results = dataset
            .scan()
            .filter("name LIKE '%app%'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let names: Vec<&str> = results
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect();
        // Should match: apple, application, app (contain 'app')
        assert!(names.iter().all(|n| n.contains("app")));

        // Test 5: NOT LIKE should NOT use scalar index
        let mut scanner = dataset.scan();
        scanner.filter("name NOT LIKE 'app%'").unwrap();
        let plan = scanner.create_plan().await.unwrap();
        let plan_str = format!("{:?}", plan);
        assert!(
            !plan_str.contains("ScalarIndexQuery"),
            "NOT LIKE should not use scalar index, but got: {}",
            plan_str
        );
    }

    #[tokio::test]
    async fn test_like_prefix_correctness_with_btree_index() {
        // Create dataset with deterministic string data for exact result verification
        let names: Vec<&str> = vec![
            "alpha", "alphabet", "beta", "gamma", "delta", "epsilon", "eta", "theta", "iota",
            "kappa",
        ];
        let data = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("name", DataType::Utf8, false),
                ArrowField::new("id", DataType::Int32, false),
            ])),
            vec![
                Arc::new(StringArray::from(names.clone())),
                Arc::new(Int32Array::from_iter_values(0..10)),
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(
            vec![Ok(data)],
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("name", DataType::Utf8, false),
                ArrowField::new("id", DataType::Int32, false),
            ])),
        );

        let mut dataset = Dataset::write(reader, "memory://test_like_correctness", None)
            .await
            .unwrap();

        // Create BTree index
        dataset
            .create_index(
                &["name"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Test with index
        let with_index = dataset
            .scan()
            .filter("name LIKE 'alpha%'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        // Test without index (for comparison)
        let without_index = dataset
            .scan()
            .use_scalar_index(false)
            .filter("name LIKE 'alpha%'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        // Both should return same results: alpha, alphabet
        assert_eq!(with_index.num_rows(), without_index.num_rows());
        assert_eq!(with_index.num_rows(), 2);

        let with_index_names: BTreeSet<String> = with_index
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap().to_string())
            .collect();

        let without_index_names: BTreeSet<String> = without_index
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap().to_string())
            .collect();

        assert_eq!(with_index_names, without_index_names);
        assert_eq!(
            with_index_names,
            BTreeSet::from(["alpha".to_string(), "alphabet".to_string()])
        );

        // Test starts_with correctness
        let starts_with_result = dataset
            .scan()
            .filter("starts_with(name, 'e')")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let starts_with_names: BTreeSet<String> = starts_with_result
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap().to_string())
            .collect();

        // Should match: epsilon, eta
        assert_eq!(
            starts_with_names,
            BTreeSet::from(["epsilon".to_string(), "eta".to_string()])
        );
    }

    #[tokio::test]
    async fn test_like_prefix_with_zone_map() {
        use lance_index::scalar::BuiltinIndexType;

        // Create dataset with string data that has various prefixes
        let data = gen_batch()
            .col(
                "name",
                array::cycle_utf8_literals(&[
                    "apple",
                    "application",
                    "app",
                    "banana",
                    "band",
                    "testns1",
                    "testns2",
                    "test",
                    "testing",
                    "zoo",
                ]),
            )
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));

        let mut dataset = Dataset::write(data, "memory://test_like_zonemap", None)
            .await
            .unwrap();

        // Create ZoneMap index on string column
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
        dataset
            .create_index(
                &["name"],
                IndexType::Scalar,
                Some("name_zonemap".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Test 1: Verify LIKE 'app%' uses zone map index
        let mut scanner = dataset.scan();
        scanner.filter("name LIKE 'app%'").unwrap();
        let plan = scanner.create_plan().await.unwrap();
        let plan_str = format!("{:?}", plan);
        // Zone map uses ScalarIndexExec with LikePrefix query
        assert!(
            plan_str.contains("ScalarIndexExec") && plan_str.contains("LikePrefix"),
            "LIKE 'app%' should use zone map index with LikePrefix, but got: {}",
            plan_str
        );

        // Verify correct results for LIKE 'app%'
        let results = dataset
            .scan()
            .filter("name LIKE 'app%'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let names: Vec<&str> = results
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect();
        assert!(names.iter().all(|n| n.starts_with("app")));
        assert!(!names.is_empty());

        // Test 2: Verify starts_with() uses zone map index
        let mut scanner = dataset.scan();
        scanner.filter("starts_with(name, 'ban')").unwrap();
        let plan = scanner.create_plan().await.unwrap();
        let plan_str = format!("{:?}", plan);
        assert!(
            plan_str.contains("ScalarIndexExec") && plan_str.contains("LikePrefix"),
            "starts_with should use zone map index with LikePrefix, but got: {}",
            plan_str
        );

        // Verify correct results
        let results = dataset
            .scan()
            .filter("starts_with(name, 'ban')")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let names: Vec<&str> = results
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap())
            .collect();
        assert!(names.iter().all(|n| n.starts_with("ban")));

        // Test 3: LIKE with refine pattern still uses zone map for prefix pruning
        let mut scanner = dataset.scan();
        scanner.filter("name LIKE 'test%2'").unwrap();
        let plan = scanner.create_plan().await.unwrap();
        let plan_str = format!("{:?}", plan);
        assert!(
            plan_str.contains("ScalarIndexExec") && plan_str.contains("LikePrefix"),
            "LIKE 'test%2' should use zone map index for prefix, but got: {}",
            plan_str
        );

        // Test 4: LIKE starting with wildcard should NOT use zone map
        let mut scanner = dataset.scan();
        scanner.filter("name LIKE '%app%'").unwrap();
        let plan = scanner.create_plan().await.unwrap();
        let plan_str = format!("{:?}", plan);
        assert!(
            !plan_str.contains("LikePrefix"),
            "LIKE '%app%' should not use LikePrefix index, but got: {}",
            plan_str
        );
    }

    /// Build an in-memory dataset with a single `Dictionary(Int16, Utf8)` column.
    /// The dictionary cycles through "a", "b", "c" so each value appears in a
    /// predictable, repeated pattern.
    async fn dictionary_string_dataset() -> Dataset {
        use arrow_array::{Int16Array, Int16DictionaryArray};

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "etld",
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            false,
        )]));

        let dictionary = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let indices = Int16Array::from((0..30).map(|i| i % 3).collect::<Vec<_>>());
        let dict_array = Int16DictionaryArray::try_new(indices, dictionary).unwrap();

        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(dict_array)]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        Dataset::write(reader, "memory://test_dict_filter", None)
            .await
            .unwrap()
    }

    /// Regression test for filtering a dictionary-encoded string column via the
    /// SQL string path (`Scanner::filter`). This used to fail to plan with
    /// "could not convert to literal of type 'Dictionary(Int16, Utf8)'".
    #[tokio::test]
    async fn test_filter_on_dictionary_string_column() {
        let dataset = dictionary_string_dataset().await;

        // Equality predicate.
        let count = dataset
            .scan()
            .filter("etld = 'a'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows();
        assert_eq!(count, 10);

        // IN-list predicate.
        let count = dataset
            .scan()
            .filter("etld IN ('a', 'b')")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows();
        assert_eq!(count, 20);
    }

    /// An `IN`/`=` predicate on a dictionary column with a scalar index should be
    /// pushed down to the index rather than falling back to a full scan.
    #[tokio::test]
    async fn test_dictionary_string_column_uses_scalar_index() {
        use lance_index::scalar::BuiltinIndexType;

        let mut dataset = dictionary_string_dataset().await;
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
        dataset
            .create_index(&["etld"], IndexType::Scalar, None, &params, true)
            .await
            .unwrap();

        let mut scanner = dataset.scan();
        scanner.filter("etld IN ('a', 'b')").unwrap();
        let plan = scanner.create_plan().await.unwrap();
        let plan_str = format!("{:?}", plan);
        assert!(
            plan_str.contains("ScalarIndexExec") || plan_str.contains("MaterializeIndex"),
            "IN on a dictionary column should use the scalar index, but got: {}",
            plan_str
        );

        let count = dataset
            .scan()
            .filter("etld IN ('a', 'b')")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows();
        assert_eq!(count, 20);
    }

    #[tokio::test]
    async fn test_like_prefix_with_segmented_zone_map() {
        use lance_index::scalar::BuiltinIndexType;

        let data = gen_batch()
            .col(
                "name",
                array::cycle_utf8_literals(&[
                    "apple",
                    "application",
                    "app",
                    "banana",
                    "band",
                    "testns1",
                    "testns2",
                    "test",
                    "testing",
                    "zoo",
                ]),
            )
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(150), BatchCount::from(6));

        let write_params = WriteParams {
            max_rows_per_file: 25,
            max_rows_per_group: 10,
            ..Default::default()
        };

        let mut dataset = Dataset::write(
            data,
            "memory://test_like_segmented_zonemap",
            Some(write_params),
        )
        .await
        .unwrap();

        let fragments = dataset.get_fragments();
        assert!(fragments.len() > 1, "expected multiple fragments");

        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
        let mut segments = Vec::with_capacity(fragments.len());
        for fragment in &fragments {
            let mut builder = dataset.create_index_builder(&["name"], IndexType::Scalar, &params);
            builder = builder
                .name("name_zonemap".to_string())
                .fragments(vec![fragment.id() as u32]);
            segments.push(builder.execute_uncommitted().await.unwrap());
        }

        dataset
            .commit_existing_index_segments("name_zonemap", "name", segments)
            .await
            .unwrap();

        let committed = dataset.load_indices_by_name("name_zonemap").await.unwrap();
        assert_eq!(committed.len(), fragments.len());

        let mut scanner = dataset.scan();
        scanner.filter("name LIKE 'app%'").unwrap();
        let plan = scanner.create_plan().await.unwrap();
        let plan_str = format!("{:?}", plan);
        assert!(
            plan_str.contains("ScalarIndexExec") && plan_str.contains("LikePrefix"),
            "segmented zonemap should use LikePrefix pruning, but got: {}",
            plan_str
        );

        let with_index = dataset
            .scan()
            .filter("name LIKE 'app%'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let without_index = dataset
            .scan()
            .use_scalar_index(false)
            .filter("name LIKE 'app%'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let with_index_ids = with_index
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let without_index_ids = without_index
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        assert_eq!(with_index_ids, without_index_ids);
        assert!(!with_index_ids.is_empty());

        let names = with_index
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|value| value.unwrap())
            .collect::<Vec<_>>();
        assert!(names.iter().all(|name| name.starts_with("app")));
    }

    #[tokio::test]
    async fn test_like_prefix_with_segmented_btree() {
        let data = gen_batch()
            .col(
                "name",
                array::cycle_utf8_literals(&[
                    "apple",
                    "application",
                    "app",
                    "banana",
                    "band",
                    "testns1",
                    "testns2",
                    "test",
                    "testing",
                    "zoo",
                ]),
            )
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(150), BatchCount::from(6));

        let write_params = WriteParams {
            max_rows_per_file: 25,
            max_rows_per_group: 10,
            ..Default::default()
        };

        let mut dataset = Dataset::write(
            data,
            "memory://test_like_segmented_btree",
            Some(write_params),
        )
        .await
        .unwrap();

        let fragments = dataset.get_fragments();
        assert!(fragments.len() > 1, "expected multiple fragments");

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        let mut segments = Vec::with_capacity(fragments.len());
        for fragment in &fragments {
            let mut builder = dataset.create_index_builder(&["name"], IndexType::BTree, &params);
            builder = builder
                .name("name_btree".to_string())
                .fragments(vec![fragment.id() as u32]);
            segments.push(builder.execute_uncommitted().await.unwrap());
        }

        dataset
            .commit_existing_index_segments("name_btree", "name", segments)
            .await
            .unwrap();

        let committed = dataset.load_indices_by_name("name_btree").await.unwrap();
        assert_eq!(committed.len(), fragments.len());

        let mut scanner = dataset.scan();
        scanner.filter("name LIKE 'app%'").unwrap();
        let plan = scanner.create_plan().await.unwrap();
        let plan_str = format!("{:?}", plan);
        assert!(
            plan_str.contains("ScalarIndexExec") && plan_str.contains("LikePrefix(Utf8(\"app\"))"),
            "segmented btree should use scalar index pruning, but got: {}",
            plan_str
        );

        let with_index = dataset
            .scan()
            .filter("name LIKE 'app%'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let without_index = dataset
            .scan()
            .use_scalar_index(false)
            .filter("name LIKE 'app%'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let with_index_ids = with_index
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let without_index_ids = without_index
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        assert_eq!(with_index_ids, without_index_ids);
        assert!(!with_index_ids.is_empty());

        let names = with_index
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|value| value.unwrap())
            .collect::<Vec<_>>();
        assert!(names.iter().all(|name| name.starts_with("app")));
    }

    #[tokio::test]
    async fn test_like_prefix_correctness_with_zone_map() {
        use lance_index::scalar::BuiltinIndexType;

        // Create dataset with deterministic string data for exact result verification
        let names: Vec<&str> = vec![
            "alpha", "alphabet", "beta", "gamma", "delta", "epsilon", "eta", "theta", "iota",
            "kappa",
        ];
        let data = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("name", DataType::Utf8, false),
                ArrowField::new("id", DataType::Int32, false),
            ])),
            vec![
                Arc::new(StringArray::from(names.clone())),
                Arc::new(Int32Array::from_iter_values(0..10)),
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(
            vec![Ok(data)],
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("name", DataType::Utf8, false),
                ArrowField::new("id", DataType::Int32, false),
            ])),
        );

        let mut dataset = Dataset::write(reader, "memory://test_like_correctness_zonemap", None)
            .await
            .unwrap();

        // Create ZoneMap index
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
        dataset
            .create_index(
                &["name"],
                IndexType::Scalar,
                Some("name_zonemap".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Test with zone map index
        let with_index = dataset
            .scan()
            .filter("name LIKE 'alpha%'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        // Test without index (for comparison)
        let without_index = dataset
            .scan()
            .use_scalar_index(false)
            .filter("name LIKE 'alpha%'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        // Both should return same results: alpha, alphabet
        assert_eq!(with_index.num_rows(), without_index.num_rows());
        assert_eq!(with_index.num_rows(), 2);

        let with_index_names: BTreeSet<String> = with_index
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap().to_string())
            .collect();

        let without_index_names: BTreeSet<String> = without_index
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap().to_string())
            .collect();

        assert_eq!(with_index_names, without_index_names);
        assert_eq!(
            with_index_names,
            BTreeSet::from(["alpha".to_string(), "alphabet".to_string()])
        );

        // Test starts_with correctness with zone map
        let starts_with_result = dataset
            .scan()
            .filter("starts_with(name, 'e')")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let starts_with_names: BTreeSet<String> = starts_with_result
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap().to_string())
            .collect();

        // Should match: epsilon, eta
        assert_eq!(
            starts_with_names,
            BTreeSet::from(["epsilon".to_string(), "eta".to_string()])
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_late_materialization(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        use lance_io::assert_io_lt;
        // Create a large dataset with a scalar indexed column and a sorted but not scalar
        // indexed column
        use lance_table::io::commit::RenameCommitHandler;
        let data = gen_batch()
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(32)),
            )
            .col("indexed", array::step::<Int32Type>())
            .col("not_indexed", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(1000), BatchCount::from(20));

        let mut dataset = Dataset::write(
            data,
            "memory://test",
            Some(WriteParams {
                commit_handler: Some(Arc::new(RenameCommitHandler)),
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["indexed"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // First run a full scan to get a baseline
        let _ = dataset.object_store.as_ref().io_stats_incremental(); // reset
        dataset.scan().try_into_batch().await.unwrap();
        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        let full_scan_bytes = io_stats.read_bytes;

        // Next do a scan without pushdown, we should still see a benefit from late materialization
        dataset
            .scan()
            .use_stats(false)
            .filter("not_indexed = 50")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_lt!(io_stats, read_bytes, full_scan_bytes);
        let filtered_scan_bytes = io_stats.read_bytes;

        // Now do a scan with pushdown, the benefit should be even greater
        // Pushdown only works with the legacy format for now.
        if data_storage_version == LanceFileVersion::Legacy {
            dataset
                .scan()
                .filter("not_indexed = 50")
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            let io_stats = dataset.object_store.as_ref().io_stats_incremental();
            assert_io_lt!(io_stats, read_bytes, filtered_scan_bytes);
        }

        // Now do a scalar index scan, this should be better than a
        // full scan but since we have to load the index might be more
        // expensive than late / pushdown scan
        dataset
            .scan()
            .filter("indexed = 50")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_lt!(io_stats, read_bytes, full_scan_bytes);
        let index_scan_bytes = io_stats.read_bytes;

        // A second scalar index scan should be cheaper than the first
        // since we should have the index in cache
        dataset
            .scan()
            .filter("indexed = 50")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_lt!(io_stats, read_bytes, index_scan_bytes);
    }

    #[rstest]
    #[tokio::test]
    async fn test_project_nested(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) -> Result<()> {
        let struct_i_field = ArrowField::new("i", DataType::Int32, true);
        let struct_o_field = ArrowField::new("o", DataType::Utf8, true);
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(
                "struct",
                DataType::Struct(vec![struct_i_field.clone(), struct_o_field.clone()].into()),
                true,
            ),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let input_batches: Vec<RecordBatch> = (0..5)
            .map(|i| {
                let struct_i_arr: Arc<Int32Array> =
                    Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20));
                let struct_o_arr: Arc<StringArray> = Arc::new(StringArray::from_iter_values(
                    (i * 20..(i + 1) * 20).map(|v| format!("o-{:02}", v)),
                ));
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(StructArray::from(vec![
                            (Arc::new(struct_i_field.clone()), struct_i_arr as ArrayRef),
                            (Arc::new(struct_o_field.clone()), struct_o_arr as ArrayRef),
                        ])),
                        Arc::new(StringArray::from_iter_values(
                            (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();
        let batches =
            RecordBatchIterator::new(input_batches.clone().into_iter().map(Ok), schema.clone());
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();

        let batches = dataset
            .scan()
            .project(&["struct.i"])
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let batch = concat_batches(&batches[0].schema(), &batches).unwrap();
        assert!(batch.column_by_name("struct.i").is_some());
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_plans(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_id: bool,
    ) -> Result<()> {
        // Create a vector dataset

        use lance_index::scalar::inverted::query::BoostQuery;
        let dim = 256;
        let mut dataset =
            TestVectorDataset::new_with_dimension(data_storage_version, stable_row_id, dim).await?;
        let lance_schema = dataset.dataset.schema();

        // Scans
        // ---------------------------------------------------------------------
        // V2 writer does not use LancePushdownScan
        if data_storage_version == LanceFileVersion::Legacy {
            log::info!("Test case: Pushdown scan");
            assert_plan_equals(
                &dataset.dataset,
                |scan| scan.project(&["s"])?.filter("i > 10 and i < 20"),
                "LancePushdownScan: uri=..., projection=[s], predicate=i > Int32(10) AND i < Int32(20), row_id=false, row_addr=false, ordered=true"
            ).await?;
        }

        log::info!("Test case: Project and filter");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[s@2 as s]
  Take: columns=\"i, _rowid, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: i@0 > 10 AND i@0 < 20
        LanceScan: uri..., projection=[i], row_id=true, row_addr=false, ordered=true, range=None"
        } else {
            "ProjectionExec: expr=[s@2 as s]
  Take: columns=\"i, _rowid, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      LanceRead: ..., projection=[i], num_fragments=2, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10) AND i < Int32(20), refine_filter=i > Int32(10) AND i < Int32(20)"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.use_stats(false)
                    .project(&["s"])?
                    .filter("i > 10 and i < 20")
            },
            expected,
        )
        .await?;

        // Integer fields will be eagerly materialized while string/vec fields
        // are not.
        log::info!("Test case: Late materialization");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@0 as i, s@1 as s, vec@3 as vec]
            Take: columns=\"i, s, _rowid, (vec)\"
              CoalesceBatchesExec: target_batch_size=8192
                FilterExec: s@1 IS NOT NULL
                  LanceScan: uri..., projection=[i, s], row_id=true, row_addr=false, ordered=true, range=None"
        } else {
            "ProjectionExec: expr=[i@0 as i, s@1 as s, vec@3 as vec]
  Take: columns=\"i, s, _rowid, (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      LanceRead: uri=..., projection=[i, s], num_fragments=2, range_before=None, range_after=None, \
      row_id=true, row_addr=false, full_filter=s IS NOT NULL, refine_filter=s IS NOT NULL"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.use_stats(false).filter("s IS NOT NULL"),
            expected,
        )
        .await?;

        // Custom materialization
        log::info!("Test case: Custom materialization (all early)");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@0 as i, s@1 as s, vec@2 as vec]
  FilterExec: s@1 IS NOT NULL
    LanceScan: uri..., projection=[i, s, vec], row_id=true, row_addr=false, ordered=true, range=None"
        } else {
            "ProjectionExec: expr=[i@0 as i, s@1 as s, vec@2 as vec]
  LanceRead: uri=..., projection=[i, s, vec], num_fragments=2, range_before=None, \
  range_after=None, row_id=true, row_addr=false, full_filter=s IS NOT NULL, refine_filter=s IS NOT NULL"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.use_stats(false)
                    .materialization_style(MaterializationStyle::AllEarly)
                    .filter("s IS NOT NULL")
            },
            expected,
        )
        .await?;

        log::info!("Test case: Custom materialization 2 (all late)");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@2 as i, s@0 as s, vec@3 as vec]
  Take: columns=\"s, _rowid, (i), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: s@0 IS NOT NULL
        LanceScan: uri..., projection=[s], row_id=true, row_addr=false, ordered=true, range=None"
        } else {
            "ProjectionExec: expr=[i@2 as i, s@0 as s, vec@3 as vec]
  Take: columns=\"s, _rowid, (i), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      LanceRead: uri=..., projection=[s], num_fragments=2, range_before=None, \
      range_after=None, row_id=true, row_addr=false, full_filter=s IS NOT NULL, refine_filter=s IS NOT NULL"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.use_stats(false)
                    .materialization_style(MaterializationStyle::AllLate)
                    .filter("s IS NOT NULL")
            },
            expected,
        )
        .await?;

        log::info!("Test case: Custom materialization 3 (mixed)");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@3 as i, s@0 as s, vec@1 as vec]
  Take: columns=\"s, vec, _rowid, (i)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: s@0 IS NOT NULL
        LanceScan: uri..., projection=[s, vec], row_id=true, row_addr=false, ordered=true, range=None"
        } else {
            "ProjectionExec: expr=[i@3 as i, s@0 as s, vec@1 as vec]
  Take: columns=\"s, vec, _rowid, (i)\"
    CoalesceBatchesExec: target_batch_size=8192
      LanceRead: uri=..., projection=[s, vec], num_fragments=2, range_before=None, range_after=None, \
      row_id=true, row_addr=false, full_filter=s IS NOT NULL, refine_filter=s IS NOT NULL"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.use_stats(false)
                    .materialization_style(
                        MaterializationStyle::all_early_except(&["i"], lance_schema).unwrap(),
                    )
                    .filter("s IS NOT NULL")
            },
            expected,
        )
        .await?;

        log::info!("Test case: Scan out of order");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "LanceScan: uri=..., projection=[s], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "LanceRead: uri=..., projection=[s], num_fragments=2, range_before=None, range_after=None, row_id=true, \
            row_addr=false, full_filter=--, refine_filter=--"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| Ok(scan.project(&["s"])?.with_row_id().scan_in_order(false)),
            expected,
        )
        .await?;

        // KNN
        // ---------------------------------------------------------------------
        let q: Float32Array = (32..32 + dim).map(|v| v as f32).collect();
        log::info!("Test case: Basic KNN");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@2 IS NOT NULL
        SortExec: TopK(fetch=5), expr=...
          KNNVectorDistance: metric=l2
            LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@2 IS NOT NULL
        SortExec: TopK(fetch=5), expr=...
          KNNVectorDistance: metric=l2
            LanceRead: uri=..., projection=[vec], num_fragments=2, range_before=None, range_after=None, \
            row_id=true, row_addr=false, full_filter=--, refine_filter=--"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 5),
            expected,
        )
        .await?;

        // KNN + Limit (arguably the user, or us, should fold the limit into the KNN but we don't today)
        // ---------------------------------------------------------------------
        let q: Float32Array = (32..32 + dim).map(|v| v as f32).collect();
        log::info!("Test case: KNN with extraneous limit");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      GlobalLimitExec: skip=0, fetch=1
        FilterExec: _distance@2 IS NOT NULL
          SortExec: TopK(fetch=5), expr=...
            KNNVectorDistance: metric=l2
              LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      GlobalLimitExec: skip=0, fetch=1
        FilterExec: _distance@2 IS NOT NULL
          SortExec: TopK(fetch=5), expr=...
            KNNVectorDistance: metric=l2
              LanceRead: uri=..., projection=[vec], num_fragments=2, range_before=None, range_after=None, \
              row_id=true, row_addr=false, full_filter=--, refine_filter=--"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 5)?.limit(Some(1), None),
            expected,
        )
        .await?;

        // ANN
        // ---------------------------------------------------------------------
        dataset.make_vector_index().await?;
        log::info!("Test case: Basic ANN");
        let expected =
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=42), expr=...
        ANNSubIndex: name=..., k=42, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1";
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 42),
            expected,
        )
        .await?;

        log::info!("Test case: ANN with refine");
        let expected =
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=10), expr=...
          KNNVectorDistance: metric=l2
            Take: columns=\"_distance, _rowid, (vec)\"
              CoalesceBatchesExec: target_batch_size=8192
                SortExec: TopK(fetch=40), expr=...
                  ANNSubIndex: name=..., k=40, deltas=1, metric=L2
                    ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1";
        assert_plan_equals(
            &dataset.dataset,
            |scan| Ok(scan.nearest("vec", &q, 10)?.refine(4)),
            expected,
        )
        .await?;

        // use_index = False -> same plan as KNN
        log::info!("Test case: ANN with index disabled");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=13), expr=...
          KNNVectorDistance: metric=l2
            LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=13), expr=...
          KNNVectorDistance: metric=l2
            LanceRead: uri=..., projection=[vec], num_fragments=2, range_before=None, range_after=None, \
            row_id=true, row_addr=false, full_filter=--, refine_filter=--"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| Ok(scan.nearest("vec", &q, 13)?.use_index(false)),
            expected,
        )
        .await?;

        log::info!("Test case: ANN with postfilter");
        // _distance is not in the final projection because the user specified an explicit
        // projection (["s", "vec"]) that does not include _distance.
        let expected = "ProjectionExec: expr=[s@3 as s, vec@4 as vec, _rowid@1 as _rowid]
  Take: columns=\"_distance, _rowid, i, (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: i@2 > 10
        Take: columns=\"_distance, _rowid, (i)\"
          CoalesceBatchesExec: target_batch_size=8192
            SortExec: TopK(fetch=17), expr=...
              ANNSubIndex: name=..., k=17, deltas=1, metric=L2
                ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1";
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 17)?
                    .filter("i > 10")?
                    .project(&["s", "vec"])?
                    .with_row_id())
            },
            expected,
        )
        .await?;

        log::info!("Test case: ANN with prefilter");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=17), expr=...
        ANNSubIndex: name=..., k=17, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
          FilterExec: i@0 > 10
            LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=17), expr=...
        ANNSubIndex: name=..., k=17, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
          LanceRead: uri=..., projection=[], num_fragments=2, range_before=None, range_after=None, \
          row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)
"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 17)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            expected,
        )
        .await?;

        dataset.append_new_data().await?;
        log::info!("Test case: Combined KNN/ANN");
        let expected = "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=6), expr=...
          KNNVectorDistance: metric=l2
            CoalescePartitionsExec
              UnionExec
                ProjectionExec: expr=[_distance@2 as _distance, _rowid@1 as _rowid, vec@0 as vec]
                  FilterExec: _distance@... IS NOT NULL
                    SortExec: TopK(fetch=6), expr=...
                      KNNVectorDistance: metric=l2
                        LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=6), expr=...
                      ANNSubIndex: name=..., k=6, deltas=1, metric=L2
                        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1";
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 6),
            // TODO: we could write an optimizer rule to eliminate the last Projection
            // by doing it as part of the last Take. This would likely have minimal impact though.
            expected,
        )
        .await?;

        // new data and with filter
        log::info!("Test case: Combined KNN/ANN with postfilter");
        let expected = "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, i, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: i@3 > 10
        Take: columns=\"_rowid, vec, _distance, (i)\"
          CoalesceBatchesExec: target_batch_size=8192
            FilterExec: _distance@... IS NOT NULL
              SortExec: TopK(fetch=15), expr=...
                KNNVectorDistance: metric=l2
                  CoalescePartitionsExec
                    UnionExec
                      ProjectionExec: expr=[_distance@2 as _distance, _rowid@1 as _rowid, vec@0 as vec]
                        FilterExec: _distance@... IS NOT NULL
                          SortExec: TopK(fetch=15), expr=...
                            KNNVectorDistance: metric=l2
                              LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None
                      Take: columns=\"_distance, _rowid, (vec)\"
                        CoalesceBatchesExec: target_batch_size=8192
                          SortExec: TopK(fetch=15), expr=...
                            ANNSubIndex: name=..., k=15, deltas=1, metric=L2
                              ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1";
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 15)?.filter("i > 10"),
            expected,
        )
        .await?;

        // new data and with prefilter
        log::info!("Test case: Combined KNN/ANN with prefilter");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=5), expr=...
          KNNVectorDistance: metric=l2
            CoalescePartitionsExec
              UnionExec
                ProjectionExec: expr=[_distance@3 as _distance, _rowid@2 as _rowid, vec@0 as vec]
                  FilterExec: _distance@... IS NOT NULL
                    SortExec: TopK(fetch=5), expr=...
                      KNNVectorDistance: metric=l2
                        FilterExec: i@1 > 10
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=5), expr=...
                      ANNSubIndex: name=..., k=5, deltas=1, metric=L2
                        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
                        FilterExec: i@0 > 10
                          LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=5), expr=...
          KNNVectorDistance: metric=l2
            CoalescePartitionsExec
              UnionExec
                ProjectionExec: expr=[_distance@3 as _distance, _rowid@2 as _rowid, vec@0 as vec]
                  FilterExec: _distance@... IS NOT NULL
                    SortExec: TopK(fetch=5), expr=...
                      KNNVectorDistance: metric=l2
                        FilterExec: i@1 > 10
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=5), expr=...
                      ANNSubIndex: name=..., k=5, deltas=1, metric=L2
                        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
                        LanceRead: uri=..., projection=[], num_fragments=2, range_before=None, range_after=None, \
                          row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 5)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            // TODO: i is scanned on both sides but is projected away mid-plan
            // only to be taken again later. We should fix this.
            expected,
        )
        .await?;

        // ANN with scalar index
        // ---------------------------------------------------------------------
        // Make sure both indices are up-to-date to start
        dataset.make_vector_index().await?;
        dataset.make_scalar_index().await?;

        log::info!("Test case: ANN with scalar index");
        let expected =
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=5), expr=...
        ANNSubIndex: name=..., k=5, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
          ScalarIndexQuery: query=[i > 10]@i_idx(BTree)";
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 5)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            expected,
        )
        .await?;

        log::info!("Test case: ANN with scalar index disabled");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=5), expr=...
        ANNSubIndex: name=..., k=5, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
          FilterExec: i@0 > 10
            LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=5), expr=...
        ANNSubIndex: name=..., k=5, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
          LanceRead: uri=..., projection=[], num_fragments=3, range_before=None, \
          range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 5)?
                    .use_scalar_index(false)
                    .filter("i > 10")?
                    .prefilter(true))
            },
            expected,
        )
        .await?;

        dataset.append_new_data().await?;

        log::info!("Test case: Combined KNN/ANN with scalar index");
        let expected = "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=8), expr=...
          KNNVectorDistance: metric=l2
            CoalescePartitionsExec
              UnionExec
                ProjectionExec: expr=[_distance@3 as _distance, _rowid@2 as _rowid, vec@0 as vec]
                  FilterExec: _distance@... IS NOT NULL
                    SortExec: TopK(fetch=8), expr=...
                      KNNVectorDistance: metric=l2
                        FilterExec: i@1 > 10
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=8), expr=...
                      ANNSubIndex: name=..., k=8, deltas=1, metric=L2
                        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
                        ScalarIndexQuery: query=[i > 10]@i_idx(BTree)";
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 8)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            expected,
        )
        .await?;

        // Update scalar index but not vector index
        log::info!(
            "Test case: Combined KNN/ANN with updated scalar index and outdated vector index"
        );
        let expected = "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=11), expr=...
          KNNVectorDistance: metric=l2
            CoalescePartitionsExec
              UnionExec
                ProjectionExec: expr=[_distance@3 as _distance, _rowid@2 as _rowid, vec@0 as vec]
                  FilterExec: _distance@... IS NOT NULL
                    SortExec: TopK(fetch=11), expr=...
                      KNNVectorDistance: metric=l2
                        FilterExec: i@1 > 10
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=11), expr=...
                      ANNSubIndex: name=..., k=11, deltas=1, metric=L2
                        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
                        ScalarIndexQuery: query=[i > 10]@i_idx(BTree)";
        dataset.make_scalar_index().await?;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 11)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            expected,
        )
        .await?;

        // Scans with scalar index
        // ---------------------------------------------------------------------
        log::info!("Test case: Filtered read with scalar index");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[s@1 as s]
  Take: columns=\"_rowid, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      MaterializeIndex: query=[i > 10]@i_idx(BTree)"
        } else {
            "LanceRead: uri=..., projection=[s], num_fragments=4, range_before=None, \
            range_after=None, row_id=false, row_addr=false, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.project(&["s"])?.filter("i > 10"),
            expected,
        )
        .await?;

        if data_storage_version != LanceFileVersion::Legacy {
            log::info!(
                "Test case: Filtered read with scalar index disabled (late materialization)"
            );
            assert_plan_equals(
                &dataset.dataset,
                |scan| {
                    scan.project(&["s"])?
                        .use_scalar_index(false)
                        .filter("i > 10")
                },
                "ProjectionExec: expr=[s@2 as s]
  Take: columns=\"i, _rowid, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      LanceRead: uri=..., projection=[i], num_fragments=4, range_before=None, \
      range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)",
            )
            .await?;
        }

        log::info!("Test case: Empty projection");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[_rowaddr@0 as _rowaddr]
  AddRowAddrExec
    MaterializeIndex: query=[i > 10]@i_idx(BTree)"
        } else {
            "LanceRead: uri=..., projection=[], num_fragments=4, range_before=None, \
            range_after=None, row_id=false, row_addr=true, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.filter("i > 10")
                    .unwrap()
                    .with_row_address()
                    .project::<&str>(&[])
            },
            expected,
        )
        .await?;

        dataset.append_new_data().await?;
        log::info!("Test case: Combined Scalar/non-scalar filtered read");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[s@1 as s]
  UnionExec
    Take: columns=\"_rowid, (s)\"
      CoalesceBatchesExec: target_batch_size=8192
        MaterializeIndex: query=[i > 10]@i_idx(BTree)
    ProjectionExec: expr=[_rowid@2 as _rowid, s@1 as s]
      FilterExec: i@0 > 10
        LanceScan: uri=..., projection=[i, s], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "LanceRead: uri=..., projection=[s], num_fragments=5, range_before=None, \
            range_after=None, row_id=false, row_addr=false, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.project(&["s"])?.filter("i > 10"),
            expected,
        )
        .await?;

        log::info!("Test case: Combined Scalar/non-scalar filtered read with empty projection");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[_rowaddr@0 as _rowaddr]
  UnionExec
    AddRowAddrExec
      MaterializeIndex: query=[i > 10]@i_idx(BTree)
    ProjectionExec: expr=[_rowaddr@2 as _rowaddr, _rowid@1 as _rowid]
      FilterExec: i@0 > 10
        LanceScan: uri=..., projection=[i], row_id=true, row_addr=true, ordered=false, range=None"
        } else {
            "LanceRead: uri=..., projection=[], num_fragments=5, range_before=None, \
            range_after=None, row_id=false, row_addr=true, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.filter("i > 10")
                    .unwrap()
                    .with_row_address()
                    .project::<&str>(&[])
            },
            expected,
        )
        .await?;

        // Scans with dynamic projection
        // When an expression is specified in the projection, the plan should include a ProjectionExec
        log::info!("Test case: Dynamic projection");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[regexp_match(s@1, .*) as matches]
  UnionExec
    Take: columns=\"_rowid, (s)\"
      CoalesceBatchesExec: target_batch_size=8192
        MaterializeIndex: query=[i > 10]@i_idx(BTree)
    ProjectionExec: expr=[_rowid@2 as _rowid, s@1 as s]
      FilterExec: i@0 > 10
        LanceScan: uri=..., row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "ProjectionExec: expr=[regexp_match(s@0, .*) as matches]
  LanceRead: uri=..., projection=[s], num_fragments=5, range_before=None, \
  range_after=None, row_id=false, row_addr=false, full_filter=i > Int32(10), refine_filter=--
    ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project_with_transform(&[("matches", "regexp_match(s, \".*\")")])?
                    .filter("i > 10")
            },
            expected,
        )
        .await?;

        // FTS
        // ---------------------------------------------------------------------
        // All rows are indexed
        dataset.make_fts_index().await?;
        log::info!("Test case: Full text search (match query)");
        let expected = r#"ProjectionExec: expr=[s@2 as s, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      MatchQuery: column=s, query=hello"#;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project(&["s"])?
                    .with_row_id()
                    .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            },
            expected,
        )
        .await?;

        log::info!("Test case: Full text search (phrase query)");
        let expected = r#"ProjectionExec: expr=[s@2 as s, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      PhraseQuery: column=s, query=hello world"#;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                let query = PhraseQuery::new("hello world".to_owned());
                scan.project(&["s"])?
                    .with_row_id()
                    .full_text_search(FullTextSearchQuery::new_query(query.into()))
            },
            expected,
        )
        .await?;

        log::info!("Test case: Full text search (boost query)");
        let expected = r#"ProjectionExec: expr=[s@2 as s, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      BoostQuery: negative_boost=1
        MatchQuery: column=s, query=hello
        MatchQuery: column=s, query=world"#;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                let positive =
                    MatchQuery::new("hello".to_owned()).with_column(Some("s".to_owned()));
                let negative =
                    MatchQuery::new("world".to_owned()).with_column(Some("s".to_owned()));
                let query = BoostQuery::new(positive.into(), negative.into(), Some(1.0));
                scan.project(&["s"])?
                    .with_row_id()
                    .full_text_search(FullTextSearchQuery::new_query(query.into()))
            },
            expected,
        )
        .await?;

        log::info!("Test case: Full text search with prefilter");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            r#"ProjectionExec: expr=[s@2 as s, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      MatchQuery: column=s, query=hello
        CoalescePartitionsExec
          UnionExec
            MaterializeIndex: query=[i > 10]@i_idx(BTree)
            ProjectionExec: expr=[_rowid@1 as _rowid]
              FilterExec: i@0 > 10
                LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None"#
        } else {
            r#"ProjectionExec: expr=[s@2 as s, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      MatchQuery: column=s, query=hello
        LanceRead: uri=..., projection=[], num_fragments=5, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=--
          ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"#
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project(&["s"])?
                    .with_row_id()
                    .filter("i > 10")?
                    .prefilter(true)
                    .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            },
            expected,
        )
        .await?;

        log::info!("Test case: Full text search with unindexed rows");
        let expected = r#"ProjectionExec: expr=[s@2 as s, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
        CoalescePartitionsExec
          UnionExec
            MatchQuery: column=s, query=hello
            FlatMatchQuery: column=s, query=hello
              LanceScan: uri=..., projection=[s], row_id=true, row_addr=false, ordered=false, range=None"#;
        dataset.append_new_data().await?;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project(&["s"])?
                    .with_row_id()
                    .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            },
            expected,
        )
        .await?;

        log::info!("Test case: Full text search with unindexed rows and fast_search");
        let expected = r#"ProjectionExec: expr=[s@2 as s, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      MatchQuery: column=s, query=hello"#;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                let scan = scan
                    .project(&["s"])?
                    .with_row_id()
                    .full_text_search(FullTextSearchQuery::new("hello".to_owned()))?;
                scan.fast_search();
                Ok(scan)
            },
            expected,
        )
        .await?;

        log::info!("Test case: Full text search with unindexed rows and prefilter");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            r#"ProjectionExec: expr=[s@2 as s, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
        CoalescePartitionsExec
          UnionExec
            MatchQuery: column=s, query=hello
              CoalescePartitionsExec
                UnionExec
                  MaterializeIndex: query=[i > 10]@i_idx(BTree)
                  ProjectionExec: expr=[_rowid@1 as _rowid]
                    FilterExec: i@0 > 10
                      LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None
            FlatMatchQuery: column=s, query=hello
              FilterExec: i@1 > 10
                LanceScan: uri=..., projection=[s, i], row_id=true, row_addr=false, ordered=false, range=None"#
        } else {
            r#"ProjectionExec: expr=[s@2 as s, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
        CoalescePartitionsExec
          UnionExec
            MatchQuery: column=s, query=hello
              LanceRead: uri=..., projection=[], num_fragments=5, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=--
                ScalarIndexQuery: query=[i > 10]@i_idx(BTree)
            FlatMatchQuery: column=s, query=hello
              FilterExec: i@1 > 10
                LanceScan: uri=..., projection=[s, i], row_id=true, row_addr=false, ordered=false, range=None"#
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project(&["s"])?
                    .with_row_id()
                    .filter("i > 10")?
                    .prefilter(true)
                    .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            },
            expected,
        )
        .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_fast_search_plan() {
        // Create a vector dataset
        let mut dataset = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        dataset.make_vector_index().await.unwrap();
        dataset.append_new_data().await.unwrap();

        let q: Float32Array = (32..64).map(|v| v as f32).collect();

        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.nearest("vec", &q, 32)?
                    .fast_search()
                    .project(&["_distance", "_rowid"])
            },
            "SortExec: TopK(fetch=32), expr=[_distance@0 ASC NULLS LAST, _rowid@1 ASC NULLS LAST]...
    ANNSubIndex: name=idx, k=32, deltas=1, metric=L2
      ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1",
        )
        .await
        .unwrap();

        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.nearest("vec", &q, 33)?
                    .fast_search()
                    .with_row_id()
                    .project(&["_distance", "_rowid"])
            },
            "SortExec: TopK(fetch=33), expr=[_distance@0 ASC NULLS LAST, _rowid@1 ASC NULLS LAST]...
    ANNSubIndex: name=idx, k=33, deltas=1, metric=L2
      ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1",
        )
        .await
        .unwrap();

        // Not `fast_scan` case
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.nearest("vec", &q, 34)?
                    .with_row_id()
                    .project(&["_distance", "_rowid"])
            },
            "ProjectionExec: expr=[_distance@2 as _distance, _rowid@0 as _rowid]
  FilterExec: _distance@2 IS NOT NULL
    SortExec: TopK(fetch=34), expr=[_distance@2 ASC NULLS LAST, _rowid@0 ASC NULLS LAST]...
      KNNVectorDistance: metric=l2
        CoalescePartitionsExec
          UnionExec
            ProjectionExec: expr=[_distance@2 as _distance, _rowid@1 as _rowid, vec@0 as vec]
              FilterExec: _distance@2 IS NOT NULL
                SortExec: TopK(fetch=34), expr=[_distance@2 ASC NULLS LAST, _rowid@1 ASC NULLS LAST]...
                  KNNVectorDistance: metric=l2
                    LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None
            Take: columns=\"_distance, _rowid, (vec)\"
              CoalesceBatchesExec: target_batch_size=8192
                SortExec: TopK(fetch=34), expr=[_distance@0 ASC NULLS LAST, _rowid@1 ASC NULLS LAST]...
                  ANNSubIndex: name=idx, k=34, deltas=1, metric=L2
                    ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1",
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_fast_search_without_vector_index_returns_empty() {
        let dataset = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        let q: Float32Array = (32..64).map(|v| v as f32).collect();

        let mut scanner = dataset.dataset.scan();
        scanner.nearest("vec", &q, 10).unwrap();
        let normal_rows = scanner.try_into_batch().await.unwrap().num_rows();

        let mut scanner = dataset.dataset.scan();
        scanner.nearest("vec", &q, 10).unwrap().fast_search();
        let fast_rows = scanner.try_into_batch().await.unwrap().num_rows();

        assert_eq!(normal_rows, 10);
        assert_eq!(fast_rows, 0);
    }

    #[tokio::test]
    async fn test_batch_fast_search_without_index_returns_empty_with_query_index() {
        let dataset = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        let query_values = (32..96).map(|v| v as f32).collect::<Vec<_>>();
        let queries =
            FixedSizeListArray::try_new_from_values(Float32Array::from(query_values), 32).unwrap();

        let mut scanner = dataset.dataset.scan();
        scanner.nearest("vec", &queries, 2).unwrap().fast_search();
        let batch = scanner.try_into_batch().await.unwrap();

        assert_eq!(batch.num_rows(), 0);
        assert_query_index_field(&batch);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_fts_multiple_unindexed_appends() {
        // An FTS query over an indexed dataset plus more than one unindexed append
        // must return matches from every unindexed fragment, not just the first. The
        // flat search over unindexed data reads only a single input partition, so when
        // the default parallelism splits the scan across partitions it used to drop
        // every append but the first.
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        test_ds.make_fts_index().await.unwrap();
        // Two separate appends -> two unindexed fragments, each large enough that the
        // physical optimizer parallelizes the flat scan across partitions.
        test_ds.append_data_with_range(400, 5400).await.unwrap();
        test_ds.append_data_with_range(5400, 10400).await.unwrap();

        // Every row's `s` value contains the token "s", so FTS("s") matches all rows.
        let total = test_ds.dataset.count_rows(None).await.unwrap();
        let returned = test_ds
            .dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("s".to_owned()))
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_fold(
                0usize,
                |acc, batch| async move { Ok(acc + batch.num_rows()) },
            )
            .await
            .unwrap();
        assert_eq!(returned, total);
    }

    #[rstest]
    #[tokio::test]
    async fn test_fast_search_scalar_index_skips_unindexed_fragments(
        #[values(LanceFileVersion::Stable)] data_storage_version: LanceFileVersion,
    ) {
        let mut dataset = TestVectorDataset::new(data_storage_version, false)
            .await
            .unwrap();
        dataset.make_scalar_index().await.unwrap();
        dataset.append_new_data().await.unwrap();

        let mut scanner = dataset.dataset.scan();
        scanner.filter("i >= 395").unwrap().project(&["i"]).unwrap();
        let normal_batch = scanner.try_into_batch().await.unwrap();

        let mut scanner = dataset.dataset.scan();
        scanner
            .filter("i >= 395")
            .unwrap()
            .fast_search()
            .project(&["i"])
            .unwrap();
        let fast_batch = scanner.try_into_batch().await.unwrap();

        assert_eq!(normal_batch.num_rows(), 15);
        assert_eq!(fast_batch.num_rows(), 5);
    }

    fn make_scalar_filter_test_batch(schema: SchemaRef, start: i32, end: i32) -> RecordBatch {
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(start..end)),
                Arc::new(Int32Array::from_iter_values(start..end)),
            ],
        )
        .unwrap()
    }

    async fn make_scalar_filter_test_dataset(
        data_storage_version: LanceFileVersion,
    ) -> (TempStrDir, SchemaRef, Dataset) {
        let tmp_dir = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("b", DataType::Int32, false),
        ]));
        let batch = make_scalar_filter_test_batch(schema.clone(), 0, 100);
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let dataset = Dataset::write(
            reader,
            &tmp_dir,
            Some(WriteParams {
                max_rows_per_file: 100,
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        (tmp_dir, schema, dataset)
    }

    async fn append_scalar_filter_test_data(
        dataset: &mut Dataset,
        schema: SchemaRef,
        start: i32,
        end: i32,
    ) {
        let batch = make_scalar_filter_test_batch(schema.clone(), start, end);
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        dataset.append(reader, None).await.unwrap();
    }

    async fn create_scalar_index(dataset: &mut Dataset, column: &str) {
        dataset
            .create_index(
                &[column],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
    }

    async fn scan_count(dataset: &Dataset, filter: &str, fast_search: bool) -> usize {
        let mut scanner = dataset.scan();
        scanner
            .filter(filter)
            .unwrap()
            .project(&["a", "b"])
            .unwrap();
        if fast_search {
            scanner.fast_search();
        }
        scanner.try_into_batch().await.unwrap().num_rows()
    }

    #[rstest]
    #[tokio::test]
    async fn test_fast_search_scalar_index_filter_coverage_cases(
        #[values(LanceFileVersion::Stable)] data_storage_version: LanceFileVersion,
    ) {
        let (_tmp_dir, schema, mut dataset) =
            make_scalar_filter_test_dataset(data_storage_version).await;
        create_scalar_index(&mut dataset, "a").await;
        append_scalar_filter_test_data(&mut dataset, schema, 100, 110).await;

        // a is indexed and b is not. The indexed side finds candidates in covered
        // fragments and b is applied as a refine filter.
        assert_eq!(scan_count(&dataset, "a >= 95 AND b >= 95", false).await, 15);
        assert_eq!(scan_count(&dataset, "a >= 95 AND b >= 95", true).await, 5);

        // OR cannot safely skip unindexed fragments: a row in an unindexed fragment
        // may satisfy `b >= 105` even if `a` is not indexed there. Skipping it would
        // silently drop valid results, so fast_search has no effect on OR queries where
        // any branch lacks a scalar index.
        assert_eq!(scan_count(&dataset, "a >= 105 OR b >= 105", false).await, 5);
        assert_eq!(scan_count(&dataset, "a >= 105 OR b >= 105", true).await, 5);

        // A single-column indexed filter skips the appended fragment in fast mode.
        assert_eq!(scan_count(&dataset, "a >= 95", false).await, 15);
        assert_eq!(scan_count(&dataset, "a >= 95", true).await, 5);

        let (_tmp_dir, schema, mut dataset) =
            make_scalar_filter_test_dataset(data_storage_version).await;
        create_scalar_index(&mut dataset, "a").await;
        append_scalar_filter_test_data(&mut dataset, schema, 100, 110).await;
        create_scalar_index(&mut dataset, "b").await;

        // a and b are both indexed, but a only covers the original fragment while b
        // covers both fragments. Fast search only reads the shared indexed coverage.
        assert_eq!(scan_count(&dataset, "a >= 95 AND b >= 95", false).await, 15);
        assert_eq!(scan_count(&dataset, "a >= 95 AND b >= 95", true).await, 5);

        let (_tmp_dir, schema, mut dataset) =
            make_scalar_filter_test_dataset(data_storage_version).await;
        append_scalar_filter_test_data(&mut dataset, schema, 100, 110).await;

        // With no scalar index query, fast_search must not enable indexed-fragment
        // pruning. This guards against treating any ordinary filter as indexed.
        assert_eq!(scan_count(&dataset, "a >= 95", false).await, 15);
        assert_eq!(scan_count(&dataset, "a >= 95", true).await, 15);
    }

    #[rstest]
    #[tokio::test]
    pub async fn test_scan_planning_io(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // Create a large dataset with a scalar indexed column and a sorted but not scalar
        // indexed column

        use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
        use lance_io::assert_io_eq;
        let data = gen_batch()
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(32)),
            )
            .col("text", array::rand_utf8(ByteCount::from(4), false))
            .col("indexed", array::step::<Int32Type>())
            .col("not_indexed", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(5));

        let mut dataset = Dataset::write(
            data,
            "memory://test",
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["indexed"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None,
                &VectorIndexParams {
                    metric_type: DistanceType::L2,
                    stages: vec![
                        StageParams::Ivf(IvfBuildParams {
                            max_iters: 2,
                            num_partitions: Some(2),
                            sample_rate: 2,
                            ..Default::default()
                        }),
                        StageParams::PQ(PQBuildParams {
                            max_iters: 2,
                            num_sub_vectors: 2,
                            ..Default::default()
                        }),
                    ],
                    version: crate::index::vector::IndexFileVersion::Legacy,
                    skip_transpose: false,
                    runtime_hints: Default::default(),
                },
                false,
            )
            .await
            .unwrap();

        // First planning cycle needs to do some I/O to determine what scalar indices are available
        dataset
            .scan()
            .prefilter(true)
            .filter("indexed > 10")
            .unwrap()
            .explain_plan(true)
            .await
            .unwrap();

        // First pass will need to perform some IOPs to determine what scalar indices are available
        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_gt!(io_stats, read_iops, 0);

        // Second planning cycle should not perform any I/O
        dataset
            .scan()
            .prefilter(true)
            .filter("indexed > 10")
            .unwrap()
            .explain_plan(true)
            .await
            .unwrap();

        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_eq!(io_stats, read_iops, 0);

        dataset
            .scan()
            .prefilter(true)
            .filter("true")
            .unwrap()
            .explain_plan(true)
            .await
            .unwrap();

        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_eq!(io_stats, read_iops, 0);

        dataset
            .scan()
            .prefilter(true)
            .materialization_style(MaterializationStyle::AllEarly)
            .filter("true")
            .unwrap()
            .explain_plan(true)
            .await
            .unwrap();

        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_eq!(io_stats, read_iops, 0);

        dataset
            .scan()
            .prefilter(true)
            .materialization_style(MaterializationStyle::AllLate)
            .filter("true")
            .unwrap()
            .explain_plan(true)
            .await
            .unwrap();

        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_eq!(io_stats, read_iops, 0);
    }

    #[rstest]
    #[tokio::test]
    pub async fn test_row_meta_columns(
        #[values(
            (true, false),  // Test row_id only
            (false, true),  // Test row_address only
            (true, true)    // Test both
        )]
        columns: (bool, bool),
    ) {
        let (with_row_id, with_row_address) = columns;
        let test_dir = TempStrDir::default();
        let uri = &test_dir;

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("data_item_id", arrow_schema::DataType::Int32, false),
            arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false),
        ]));

        let data = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1001, 1002, 1003])),
                Arc::new(Int32Array::from(vec![1, 2, 3])),
            ],
        )
        .unwrap();

        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data)], schema.clone()),
            uri,
            None,
        )
        .await
        .unwrap();

        // Test explicit projection
        let mut scanner = dataset.scan();

        let mut projection = vec!["data_item_id".to_string()];
        if with_row_id {
            scanner.with_row_id();
            projection.push(ROW_ID.to_string());
        }
        if with_row_address {
            scanner.with_row_address();
            projection.push(ROW_ADDR.to_string());
        }

        scanner.project(&projection).unwrap();
        let stream = scanner.try_into_stream().await.unwrap();
        let batch = stream.try_collect::<Vec<_>>().await.unwrap().pop().unwrap();

        // Verify column existence and data type
        if with_row_id {
            let column = batch.column_by_name(ROW_ID).unwrap();
            assert_eq!(column.data_type(), &DataType::UInt64);
        }
        if with_row_address {
            let column = batch.column_by_name(ROW_ADDR).unwrap();
            assert_eq!(column.data_type(), &DataType::UInt64);
        }

        // Test implicit inclusion
        let mut scanner = dataset.scan();
        if with_row_id {
            scanner.with_row_id();
        }
        if with_row_address {
            scanner.with_row_address();
        }
        scanner.project(&["data_item_id"]).unwrap();
        let stream = scanner.try_into_stream().await.unwrap();
        let batch = stream.try_collect::<Vec<_>>().await.unwrap().pop().unwrap();
        let meta_column = batch.column_by_name(if with_row_id { ROW_ID } else { ROW_ADDR });
        assert!(meta_column.is_some());

        // Test error case
        let mut scanner = dataset.scan();
        if with_row_id {
            scanner.project(&[ROW_ID]).unwrap();
        } else {
            scanner.project(&[ROW_ADDR]).unwrap();
        };
        let stream = scanner.try_into_stream().await.unwrap();
        assert_eq!(stream.schema().fields().len(), 1);
        if with_row_id {
            assert!(stream.schema().field_with_name(ROW_ID).is_ok());
        } else {
            assert!(stream.schema().field_with_name(ROW_ADDR).is_ok());
        }
    }

    async fn limit_offset_equivalency_test(scanner: &Scanner) {
        async fn test_one(
            scanner: &Scanner,
            full_result: &RecordBatch,
            limit: Option<i64>,
            offset: Option<i64>,
        ) {
            let mut new_scanner = scanner.clone();
            new_scanner.limit(limit, offset).unwrap();
            if let Some(nearest) = new_scanner.nearest_mut() {
                nearest.k = offset.unwrap_or(0).saturating_add(limit.unwrap_or(10_000)) as usize;
            }
            let result = new_scanner.try_into_batch().await.unwrap();

            let resolved_offset = offset.unwrap_or(0).min(full_result.num_rows() as i64);
            let resolved_length = limit
                .unwrap_or(i64::MAX)
                .min(full_result.num_rows() as i64 - resolved_offset);

            let expected = full_result.slice(resolved_offset as usize, resolved_length as usize);

            if expected != result {
                let plan = new_scanner.analyze_plan().await.unwrap();
                assert_eq!(
                    &expected, &result,
                    "Limit: {:?}, Offset: {:?}, Plan: \n{}",
                    limit, offset, plan
                );
            }
        }

        let mut scanner_full = scanner.clone();
        if let Some(nearest) = scanner_full.nearest_mut() {
            nearest.k = 500;
        }
        let full_results = scanner_full.try_into_batch().await.unwrap();

        test_one(scanner, &full_results, Some(1), None).await;
        test_one(scanner, &full_results, Some(1), Some(1)).await;
        test_one(scanner, &full_results, Some(1), Some(2)).await;
        test_one(scanner, &full_results, Some(1), Some(10)).await;

        test_one(scanner, &full_results, Some(3), None).await;
        test_one(scanner, &full_results, Some(3), Some(2)).await;
        test_one(scanner, &full_results, Some(3), Some(4)).await;

        test_one(scanner, &full_results, None, Some(3)).await;
        test_one(scanner, &full_results, None, Some(10)).await;
    }

    #[tokio::test]
    async fn test_scan_limit_offset() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let scanner = test_ds.dataset.scan();
        limit_offset_equivalency_test(&scanner).await;
    }

    #[tokio::test]
    async fn test_knn_limit_offset() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let query_vector = Float32Array::from(vec![0.0; 32]);
        let mut scanner = test_ds.dataset.scan();
        scanner
            .nearest("vec", &query_vector, 5)
            .unwrap()
            .project(&["i"])
            .unwrap();
        limit_offset_equivalency_test(&scanner).await;
    }

    #[tokio::test]
    async fn test_knn_query_parallelism_defaults_and_setter() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let query_vector = Float32Array::from(vec![0.0; 32]);
        let mut scanner = test_ds.dataset.scan();
        scanner.nearest("vec", &query_vector, 5).unwrap();
        assert_eq!(
            scanner.nearest_mut().unwrap().query_parallelism,
            DEFAULT_QUERY_PARALLELISM
        );

        scanner.query_parallelism(4);
        assert_eq!(scanner.nearest_mut().unwrap().query_parallelism, 4);

        scanner.query_parallelism(-1);
        assert_eq!(scanner.nearest_mut().unwrap().query_parallelism, -1);
    }

    #[tokio::test]
    async fn test_knn_approx_mode_defaults_and_setter() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let query_vector = Float32Array::from(vec![0.0; 32]);
        let mut scanner = test_ds.dataset.scan();
        scanner.nearest("vec", &query_vector, 5).unwrap();
        assert_eq!(
            scanner.nearest_mut().unwrap().approx_mode,
            ApproxMode::Normal
        );

        scanner.approx_mode(ApproxMode::Accurate);
        assert_eq!(
            scanner.nearest_mut().unwrap().approx_mode,
            ApproxMode::Accurate
        );
    }

    #[tokio::test]
    async fn test_ivf_pq_query_parallelism_returns_same_results() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        test_ds.make_vector_index().await.unwrap();

        let query_vector = Float32Array::from(vec![0.0; 32]);

        let mut sequential = test_ds.dataset.scan();
        sequential.nearest("vec", &query_vector, 50).unwrap();
        let sequential_results = sequential.try_into_batch().await.unwrap();

        let mut parallel = test_ds.dataset.scan();
        parallel
            .nearest("vec", &query_vector, 50)
            .unwrap()
            .query_parallelism(4);
        let parallel_results = parallel.try_into_batch().await.unwrap();

        assert_eq!(sequential_results, parallel_results);
    }

    #[tokio::test]
    async fn test_ivf_pq_limit_offset() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        test_ds.make_vector_index().await.unwrap();
        test_ds.append_new_data().await.unwrap();
        let query_vector = Float32Array::from(vec![0.0; 32]);
        let mut scanner = test_ds.dataset.scan();
        scanner.nearest("vec", &query_vector, 500).unwrap();
        limit_offset_equivalency_test(&scanner).await;
    }

    #[tokio::test]
    async fn test_fts_limit_offset() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        test_ds.make_fts_index().await.unwrap();
        test_ds.append_new_data().await.unwrap();
        let mut scanner = test_ds.dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new("4".into()))
            .unwrap();
        limit_offset_equivalency_test(&scanner).await;
    }

    #[tokio::test]
    async fn test_fts_fast_search_excludes_unindexed_rows() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        test_ds.make_fts_index().await.unwrap();
        // Append rows after index build so they stay unindexed.
        test_ds.append_data_with_range(10, 20).await.unwrap();

        let mut scanner = test_ds.dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new_query(
                MatchQuery::new("15".to_owned())
                    .with_column(Some("s".to_owned()))
                    .into(),
            ))
            .unwrap();
        let normal_rows = scanner.try_into_batch().await.unwrap().num_rows();

        let mut scanner = test_ds.dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new_query(
                MatchQuery::new("15".to_owned())
                    .with_column(Some("s".to_owned()))
                    .into(),
            ))
            .unwrap()
            .fast_search();
        let fast_rows = scanner.try_into_batch().await.unwrap().num_rows();

        assert_eq!(normal_rows, 2);
        assert_eq!(fast_rows, 1);
    }

    async fn test_row_offset_read_helper(
        ds: &Dataset,
        scan_builder: impl FnOnce(&mut Scanner) -> &mut Scanner,
        expected_cols: &[&str],
        expected_row_offsets: &[u64],
    ) {
        let mut scanner = ds.scan();
        let scanner = scan_builder(&mut scanner);
        let stream = scanner.try_into_stream().await.unwrap();

        let schema = stream.schema();
        let actual_cols = schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>();
        assert_eq!(&actual_cols, expected_cols);

        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let batch = arrow_select::concat::concat_batches(&schema, &batches).unwrap();

        let row_offsets = batch
            .column_by_name(ROW_OFFSET)
            .unwrap()
            .as_primitive::<UInt64Type>()
            .values();
        assert_eq!(row_offsets.as_ref(), expected_row_offsets);
    }

    #[tokio::test]
    async fn test_row_offset_read() {
        let mut ds = lance_datagen::gen_batch()
            .col("idx", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(3))
            .await
            .unwrap();
        // [0, 1, 2], [3, 4, 5], [6, 7, 8]

        // Delete [2, 3, 4, 5, 6]
        ds.delete("idx >= 2 AND idx <= 6").await.unwrap();

        // Normal read, all columns plus row offset
        test_row_offset_read_helper(
            &ds,
            |scanner| scanner.project(&["idx", ROW_OFFSET]).unwrap(),
            &["idx", ROW_OFFSET],
            &[0, 1, 2, 3],
        )
        .await;

        // Read with row offset only
        test_row_offset_read_helper(
            &ds,
            |scanner| scanner.project(&[ROW_OFFSET]).unwrap(),
            &[ROW_OFFSET],
            &[0, 1, 2, 3],
        )
        .await;

        // Filtered read of row offset
        test_row_offset_read_helper(
            &ds,
            |scanner| {
                scanner
                    .filter("idx > 1")
                    .unwrap()
                    .project(&[ROW_OFFSET])
                    .unwrap()
            },
            &[ROW_OFFSET],
            &[2, 3],
        )
        .await;
    }

    #[tokio::test]
    async fn test_filter_to_take() {
        let mut ds = lance_datagen::gen_batch()
            .col("idx", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(100))
            .await
            .unwrap();

        let row_ids = ds
            .scan()
            .project(&Vec::<&str>::default())
            .unwrap()
            .with_row_id()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let schema = row_ids[0].schema();
        let row_ids = concat_batches(&schema, row_ids.iter()).unwrap();
        let row_ids = row_ids.column(0).as_primitive::<UInt64Type>().clone();

        let row_addrs = ds
            .scan()
            .project(&Vec::<&str>::default())
            .unwrap()
            .with_row_address()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let schema = row_addrs[0].schema();
        let row_addrs = concat_batches(&schema, row_addrs.iter()).unwrap();
        let row_addrs = row_addrs.column(0).as_primitive::<UInt64Type>().clone();

        ds.delete("idx >= 190 AND idx < 210").await.unwrap();

        let ds_copy = ds.clone();
        let do_check = async move |filt: &str, expected_idx: &[i32], applies_optimization: bool| {
            let mut scanner = ds_copy.scan();
            scanner.filter(filt).unwrap();
            // Verify the optimization is applied
            let plan = scanner.explain_plan(true).await.unwrap();
            if applies_optimization {
                assert!(
                    plan.contains("OneShotStream"),
                    "expected take optimization to be applied. Filter: '{}'.  Plan:\n{}",
                    filt,
                    plan
                );
            } else {
                assert!(
                    !plan.contains("OneShotStream"),
                    "expected take optimization to not be applied. Filter: '{}'.  Plan:\n{}",
                    filt,
                    plan
                );
            }

            // Verify the results
            let stream = scanner.try_into_stream().await.unwrap();
            let batches = stream.try_collect::<Vec<_>>().await.unwrap();
            let idx = batches
                .iter()
                .map(|b| b.column_by_name("idx").unwrap().as_ref())
                .collect::<Vec<_>>();

            if idx.is_empty() {
                assert!(expected_idx.is_empty());
                return;
            }

            let idx = arrow::compute::concat(&idx).unwrap();
            assert_eq!(idx.as_primitive::<Int32Type>().values(), expected_idx);
        };
        let check =
            async |filt: &str, expected_idx: &[i32]| do_check(filt, expected_idx, true).await;
        let check_no_opt = async |filt: &str, expected_idx: &[i32]| {
            do_check(filt, expected_idx, false).await;
        };

        // Simple case, no deletions yet
        check("_rowid = 50", &[50]).await;
        check("_rowaddr = 50", &[50]).await;
        check("_rowoffset = 50", &[50]).await;

        check(
            "_rowid = 50 OR _rowid = 51 OR _rowid = 52 OR _rowid = 49",
            &[49, 50, 51, 52],
        )
        .await;
        check(
            "_rowaddr = 50 OR _rowaddr = 51 OR _rowaddr = 52 OR _rowaddr = 49",
            &[49, 50, 51, 52],
        )
        .await;
        check(
            "_rowoffset = 50 OR _rowoffset = 51 OR _rowoffset = 52 OR _rowoffset = 49",
            &[49, 50, 51, 52],
        )
        .await;

        check("_rowid IN (52, 51, 50, 17)", &[17, 50, 51, 52]).await;
        check("_rowaddr IN (52, 51, 50, 17)", &[17, 50, 51, 52]).await;
        check("_rowoffset IN (52, 51, 50, 17)", &[17, 50, 51, 52]).await;

        // Taking _rowid / _rowaddr of deleted row

        // When using rowid / rowaddr we get an empty
        check(&format!("_rowid = {}", row_ids.value(190)), &[]).await;
        check(&format!("_rowaddr = {}", row_addrs.value(190)), &[]).await;
        // When using rowoffset it just skips the deleted rows (impossible to create an offset
        // into a deleted row)
        check("_rowoffset = 190", &[210]).await;

        // Grabbing after the deleted rows
        check(&format!("_rowid = {}", row_ids.value(250)), &[250]).await;
        check(&format!("_rowaddr = {}", row_addrs.value(250)), &[250]).await;
        check("_rowoffset = 250", &[270]).await;

        // Grabbing past the end
        check("_rowoffset = 1000", &[]).await;

        // Combine take and filter
        check("_rowid IN (5, 10, 15) AND idx > 10", &[15]).await;
        check("_rowaddr IN (5, 10, 15) AND idx > 10", &[15]).await;
        check("_rowoffset IN (5, 10, 15) AND idx > 10", &[15]).await;
        check("idx > 10 AND _rowid IN (5, 10, 15)", &[15]).await;
        check("idx > 10 AND _rowaddr IN (5, 10, 15)", &[15]).await;
        check("idx > 10 AND _rowoffset IN (5, 10, 15)", &[15]).await;
        // Get's simplified into _rowid = 50 and so we catch it
        check("_rowid = 50 AND _rowid = 50", &[50]).await;

        // Filters that cannot be converted into a take
        check_no_opt("_rowid = 50 AND _rowid = 51", &[]).await;
        check_no_opt("(_rowid = 50 AND idx < 100) OR _rowid = 51", &[50, 51]).await;

        // Dynamic projection
        let mut scanner = ds.scan();
        scanner.filter("_rowoffset = 77").unwrap();
        scanner
            .project_with_transform(&[("foo", "idx * 2")])
            .unwrap();
        let stream = scanner.try_into_stream().await.unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(batches[0].schema().field(0).name(), "foo");
        let val = batches[0].column(0).as_primitive::<Int32Type>().values()[0];
        assert_eq!(val, 154);
    }

    #[tokio::test]
    async fn test_nested_field_ordering() {
        use arrow_array::StructArray;

        // Create test data with nested structs
        let id_array = Int32Array::from(vec![3, 1, 2]);
        let nested_values = Int32Array::from(vec![30, 10, 20]);
        let nested_struct = StructArray::from(vec![(
            Arc::new(ArrowField::new("value", DataType::Int32, false)),
            Arc::new(nested_values) as ArrayRef,
        )]);

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new(
                "nested",
                DataType::Struct(vec![ArrowField::new("value", DataType::Int32, false)].into()),
                false,
            ),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(id_array), Arc::new(nested_struct)],
        )
        .unwrap();

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone());

        let dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Test ordering by nested field
        let mut scanner = dataset.scan();
        scanner
            .order_by(Some(vec![ColumnOrdering {
                column_name: "nested.value".to_string(),
                ascending: true,
                nulls_first: true,
            }]))
            .unwrap(); // ascending order

        let stream = scanner.try_into_stream().await.unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();

        // Check that results are sorted by nested.value
        let sorted_ids = batches[0].column(0).as_primitive::<Int32Type>().values();
        assert_eq!(sorted_ids[0], 1); // id=1 has nested.value=10
        assert_eq!(sorted_ids[1], 2); // id=2 has nested.value=20
        assert_eq!(sorted_ids[2], 3); // id=3 has nested.value=30
    }

    #[tokio::test]
    async fn test_limit_with_ordering_not_pushed_down() {
        // This test verifies the fix for a bug where limit/offset could be pushed down
        // even when ordering was specified. When ordering is present, we need to load
        // all data first to sort it before applying limits.

        // Create test data with specific ordering
        let id_array = Int32Array::from(vec![5, 2, 8, 1, 3, 7, 4, 6]);
        let value_array = Int32Array::from(vec![50, 20, 80, 10, 30, 70, 40, 60]);

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(id_array), Arc::new(value_array)],
        )
        .unwrap();

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone());

        let dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Test 1: limit with ordering should return top N after sorting
        let mut scanner = dataset.scan();
        scanner
            .order_by(Some(vec![ColumnOrdering {
                column_name: "value".to_string(),
                ascending: true,
                nulls_first: true,
            }]))
            .unwrap();
        scanner.limit(Some(3), None).unwrap();

        let stream = scanner.try_into_stream().await.unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();

        // Results should be sorted by value and limited to 3
        let sorted_ids = batches[0].column(0).as_primitive::<Int32Type>().values();
        let sorted_values = batches[0].column(1).as_primitive::<Int32Type>().values();
        assert_eq!(batches[0].num_rows(), 3);
        assert_eq!(sorted_ids[0], 1); // value=10
        assert_eq!(sorted_ids[1], 2); // value=20
        assert_eq!(sorted_ids[2], 3); // value=30
        assert_eq!(sorted_values[0], 10);
        assert_eq!(sorted_values[1], 20);
        assert_eq!(sorted_values[2], 30);

        // Test 2: offset with ordering should skip first N after sorting
        let mut scanner = dataset.scan();
        scanner
            .order_by(Some(vec![ColumnOrdering {
                column_name: "value".to_string(),
                ascending: true,
                nulls_first: true,
            }]))
            .unwrap();
        scanner.limit(Some(3), Some(2)).unwrap(); // Skip first 2, take next 3

        let stream = scanner.try_into_stream().await.unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();

        let sorted_ids = batches[0].column(0).as_primitive::<Int32Type>().values();
        let sorted_values = batches[0].column(1).as_primitive::<Int32Type>().values();
        assert_eq!(batches[0].num_rows(), 3);
        assert_eq!(sorted_ids[0], 3); // value=30 (skipped 10, 20)
        assert_eq!(sorted_ids[1], 4); // value=40
        assert_eq!(sorted_ids[2], 5); // value=50
        assert_eq!(sorted_values[0], 30);
        assert_eq!(sorted_values[1], 40);
        assert_eq!(sorted_values[2], 50);

        // Test 3: without ordering, limit can be pushed down (different behavior)
        let mut scanner = dataset.scan();
        scanner.limit(Some(3), None).unwrap();

        let stream = scanner.try_into_stream().await.unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();

        // Should get first 3 rows in storage order (not sorted)
        assert_eq!(batches[0].num_rows(), 3);
        let unsorted_values = batches[0].column(1).as_primitive::<Int32Type>().values();
        // These will be in original insertion order, not sorted
        assert_eq!(unsorted_values[0], 50);
        assert_eq!(unsorted_values[1], 20);
        assert_eq!(unsorted_values[2], 80);
    }

    #[tokio::test]
    async fn test_scan_with_version_columns() {
        use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};

        // Create a simple dataset
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        let test_dir = lance_core::utils::tempfile::TempStrDir::default();
        let test_uri = test_dir.as_str();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let mut scanner = dataset.scan();

        scanner
            .project(&[ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION])
            .unwrap();

        // Check that the schema includes version columns
        let output_schema = scanner.schema().await.unwrap();
        assert!(
            output_schema
                .column_with_name("_row_last_updated_at_version")
                .is_some(),
            "Schema should include _row_last_updated_at_version"
        );
        assert!(
            output_schema
                .column_with_name("_row_created_at_version")
                .is_some(),
            "Schema should include _row_created_at_version"
        );

        // Actually read the data to ensure version columns are materialized
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(batches.len(), 1);
        let batch = &batches[0];

        // Verify version columns exist in the output
        let last_updated = batch
            .column_by_name("_row_last_updated_at_version")
            .expect("Should have _row_last_updated_at_version column");
        let created_at = batch
            .column_by_name("_row_created_at_version")
            .expect("Should have _row_created_at_version column");

        // Verify they have the correct values (all rows created in version 1)
        let last_updated_array = last_updated
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap();
        let created_at_array = created_at
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap();

        for i in 0..batch.num_rows() {
            assert_eq!(
                last_updated_array.value(i),
                1,
                "All rows last updated at version 1"
            );
            assert_eq!(
                created_at_array.value(i),
                1,
                "All rows created at version 1"
            );
        }
    }

    #[test_log::test(test)]
    fn test_scan_finishes_all_tasks() {
        // Need to use multi-threaded runtime otherwise tasks don't run unless someone is polling somewhere
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_time()
            .build()
            .unwrap();

        runtime.block_on(async move {
            let ds = lance_datagen::gen_batch()
                .col("id", lance_datagen::array::step::<Int32Type>())
                .into_ram_dataset(FragmentCount::from(1000), FragmentRowCount::from(10))
                .await
                .unwrap();

            // This scan with has a small I/O buffer size and batch size to mimic a real-world situation
            // that required a lot of data.  Many fragments will be scheduled at low priority and the data
            // buffer will fill up with data reads.  When the scan is abandoned, the tasks to read the fragment
            // metadata were left behind and would never finish because the data was never decoded to drain the
            // backpressure queue.
            //
            // The fix (that this test verifies) is to ensure we close the I/O scheduler when the scan is abandoned.
            let mut stream = ds
                .scan()
                .fragment_readahead(1000)
                .batch_size(1)
                .io_buffer_size(1)
                .batch_readahead(1)
                .try_into_stream()
                .await
                .unwrap();
            stream.next().await.unwrap().unwrap();
        });

        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(10) {
            if runtime.handle().metrics().num_alive_tasks() == 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        assert!(
            runtime.handle().metrics().num_alive_tasks() == 0,
            "Tasks should have finished within 10 seconds but there are still {} tasks running",
            runtime.handle().metrics().num_alive_tasks()
        );
    }

    fn find_filtered_read(plan: &dyn ExecutionPlan) -> Option<&FilteredReadExec> {
        if let Some(f) = plan.as_any().downcast_ref::<FilteredReadExec>() {
            return Some(f);
        }
        for child in plan.children() {
            if let Some(f) = find_filtered_read(child.as_ref()) {
                return Some(f);
            }
        }
        None
    }

    #[tokio::test]
    async fn test_io_buffer_size_explicit_propagated() {
        // Sanity check: an explicit .io_buffer_size(N) call must reach the
        // FilteredReadExec options unchanged, and the absence of one must leave
        // io_buffer_size_bytes as None so FilteredReadExec can pick its own
        // fallback (env var or max_bandwidth).
        let data = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(8), BatchCount::from(1));
        let dataset = Dataset::write(data, "memory://test_io_buffer_explicit", None)
            .await
            .unwrap();

        let plan = dataset.scan().create_plan().await.unwrap();
        let filtered = find_filtered_read(plan.as_ref())
            .expect("expected a FilteredReadExec in the scan plan");
        assert_eq!(filtered.options().io_buffer_size_bytes, None);

        let mut scanner = dataset.scan();
        scanner.io_buffer_size(7777);
        let plan = scanner.create_plan().await.unwrap();
        let filtered = find_filtered_read(plan.as_ref())
            .expect("expected a FilteredReadExec in the scan plan");
        assert_eq!(filtered.options().io_buffer_size_bytes, Some(7777));
    }

    // The env var key scopes serial_test's lock so this test only blocks others
    // that touch LANCE_DEFAULT_IO_BUFFER_SIZE — unrelated tests still run in
    // parallel.
    #[test]
    #[serial_test::serial(LANCE_DEFAULT_IO_BUFFER_SIZE)]
    fn test_default_io_buffer_size_override_env_var() {
        // Force the sibling LazyLock to evaluate before we mutate the env var.
        // It caches forever on first read, so another test concurrently reading
        // *DEFAULT_IO_BUFFER_SIZE during our mutation window would otherwise
        // cache one of our test values and poison the rest of the suite.
        let _ = *DEFAULT_IO_BUFFER_SIZE;

        // FilteredReadExec consults this when no explicit io_buffer_size was set
        // on the scanner, so the LANCE_DEFAULT_IO_BUFFER_SIZE env var takes
        // precedence over the max_bandwidth fallback.
        unsafe {
            std::env::set_var("LANCE_DEFAULT_IO_BUFFER_SIZE", "4096");
        }
        assert_eq!(get_default_io_buffer_size_override(), Some(4096));

        unsafe {
            std::env::set_var("LANCE_DEFAULT_IO_BUFFER_SIZE", "not_a_number");
        }
        assert_eq!(get_default_io_buffer_size_override(), None);

        unsafe {
            std::env::remove_var("LANCE_DEFAULT_IO_BUFFER_SIZE");
        }
        assert_eq!(get_default_io_buffer_size_override(), None);
    }

    fn assert_values_in_range(array: &Int32Array, range: std::ops::Range<i32>, msg: &str) {
        assert!(!array.is_empty(), "Expected some results but got none");
        assert!(
            array
                .iter()
                .all(|v| v.is_some_and(|val| range.contains(&val))),
            "{msg} (expected range {range:?})"
        );
    }

    // Helper to assert that results exist from all fragment ranges
    fn assert_has_all_fragments(array: &Int32Array) {
        assert!(
            array
                .iter()
                .any(|v| v.is_some_and(|val| (0..200).contains(&val)))
                && array
                    .iter()
                    .any(|v| v.is_some_and(|val| (200..400).contains(&val)))
                && array
                    .iter()
                    .any(|v| v.is_some_and(|val| (400..410).contains(&val)))
                && array
                    .iter()
                    .any(|v| v.is_some_and(|val| (410..420).contains(&val))),
            "Expected results from all fragments"
        );
    }

    // Common test function for fragment list filtering (unindexed + indexed fragments)
    async fn test_fragment_list_filtering(
        test_ds: &TestVectorDataset,
        fragments: &[Fragment],
        mut build_scanner: impl FnMut(&Dataset) -> Scanner,
    ) {
        // Test 1: Query without fragment filter - should get results from all fragments
        let batch = build_scanner(&test_ds.dataset)
            .try_into_batch()
            .await
            .unwrap();
        let i_array = batch
            .column_by_name("i")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_has_all_fragments(i_array);

        // Test 2: Query only one unindexed fragment (fragment 2), excluding fragment 3
        let mut scanner = build_scanner(&test_ds.dataset);
        scanner.with_fragments(vec![fragments[2].clone()]);
        let batch = scanner.try_into_batch().await.unwrap();
        let i_array = batch
            .column_by_name("i")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_values_in_range(i_array, 400..410, "Should only get results from fragment 2");

        // Test 3: Query a single indexed fragment (fragment 0 only)
        let mut scanner = build_scanner(&test_ds.dataset);
        scanner.with_fragments(vec![fragments[0].clone()]);
        let batch = scanner.try_into_batch().await.unwrap();
        let i_array = batch
            .column_by_name("i")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_values_in_range(i_array, 0..200, "Should only get results from fragment 0");

        // Test 4: Query all indexed fragments (0, 1) plus one unindexed fragment (2), excluding fragment 3
        let mut scanner = build_scanner(&test_ds.dataset);
        scanner.with_fragments(vec![
            fragments[0].clone(),
            fragments[1].clone(),
            fragments[2].clone(),
        ]);
        let batch = scanner.try_into_batch().await.unwrap();
        let i_array = batch
            .column_by_name("i")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_values_in_range(
            i_array,
            0..410,
            "Should get results from fragments 0, 1, and 2, excluding fragment 3",
        );

        // Test 5: One indexed fragment (0) + one unindexed fragment (2), skipping indexed fragment 1 and unindexed fragment 3
        let mut scanner = build_scanner(&test_ds.dataset);
        scanner.with_fragments(vec![fragments[0].clone(), fragments[2].clone()]);
        let batch = scanner.try_into_batch().await.unwrap();
        let i_array = batch
            .column_by_name("i")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(
            i_array
                .iter()
                .all(|v| v.is_some_and(|val| (0..200).contains(&val) || (400..410).contains(&val)))
                && i_array
                    .iter()
                    .any(|v| v.is_some_and(|val| (0..200).contains(&val)))
                && i_array
                    .iter()
                    .any(|v| v.is_some_and(|val| (400..410).contains(&val))),
            "Should only get results from fragment 0 (indexed) and fragment 2 (unindexed)"
        );
    }

    #[tokio::test]
    async fn test_vector_search_respects_fragment_list() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();

        // Create one segment per indexed fragment so fragment filtering must prune ANN fan-out.
        test_ds.make_segmented_vector_index().await.unwrap();

        let query: Float32Array = (0..32).map(|v| v as f32).collect();

        // Append two more unindexed fragments
        test_ds.append_data_with_range(400, 410).await.unwrap();
        test_ds.append_data_with_range(410, 420).await.unwrap();

        // Fragment 0: i=0..200 (indexed), Fragment 1: i=200..400 (indexed)
        // Fragment 2: i=400..410 (unindexed), Fragment 3: i=410..420 (unindexed)
        let fragments = test_ds.dataset.fragments();
        assert_eq!(fragments.len(), 4);

        test_fragment_list_filtering(&test_ds, fragments, |dataset| {
            let mut scanner = dataset.scan();
            scanner.nearest("vec", &query, 420).unwrap();
            scanner
        })
        .await;
    }

    #[tokio::test]
    async fn test_vector_search_fragment_filter_prunes_segment_fanout() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        test_ds.make_segmented_vector_index().await.unwrap();

        let query: Float32Array = (0..32).map(|v| v as f32).collect();
        test_ds.append_data_with_range(400, 410).await.unwrap();
        test_ds.append_data_with_range(410, 420).await.unwrap();
        let fragments = test_ds.dataset.fragments();

        let mut scanner = test_ds.dataset.scan();
        scanner.nearest("vec", &query, 420).unwrap();
        let full_plan = scanner.explain_plan(true).await.unwrap();
        assert!(
            full_plan.contains("ANNSubIndex: name=idx, k=420, deltas=2, metric=L2"),
            "expected two ANN deltas without fragment filter, plan was:\n{full_plan}"
        );

        let mut scanner = test_ds.dataset.scan();
        scanner
            .nearest("vec", &query, 420)
            .unwrap()
            .with_fragments(vec![fragments[0].clone()]);
        let filtered_plan = scanner.explain_plan(true).await.unwrap();
        assert!(
            filtered_plan.contains("ANNSubIndex: name=idx, k=420, deltas=1, metric=L2"),
            "expected one ANN delta with fragment filter, plan was:\n{filtered_plan}"
        );
    }

    #[tokio::test]
    async fn test_vector_search_respects_index_segments() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let segment_ids = test_ds.make_segmented_vector_index().await.unwrap();

        let query: Float32Array = (0..32).map(|v| v as f32).collect();
        test_ds.append_data_with_range(400, 410).await.unwrap();
        test_ds.append_data_with_range(410, 420).await.unwrap();

        let mut scanner = test_ds.dataset.scan();
        scanner
            .nearest("vec", &query, 420)
            .unwrap()
            .with_index_segments(vec![segment_ids[0]])
            .unwrap();
        let batch = scanner.try_into_batch().await.unwrap();
        let i_array = batch
            .column_by_name("i")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(batch.num_rows(), 200);
        assert_values_in_range(
            i_array,
            0..200,
            "Should only get results from the selected index segment",
        );
    }

    #[tokio::test]
    async fn test_vector_search_intersects_fragments_and_index_segments() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let segment_ids = test_ds.make_segmented_vector_index().await.unwrap();

        let query: Float32Array = (0..32).map(|v| v as f32).collect();
        test_ds.append_data_with_range(400, 410).await.unwrap();
        test_ds.append_data_with_range(410, 420).await.unwrap();
        let fragments = test_ds.dataset.fragments();

        let mut scanner = test_ds.dataset.scan();
        scanner
            .nearest("vec", &query, 420)
            .unwrap()
            .with_fragments(vec![fragments[0].clone(), fragments[2].clone()])
            .with_index_segments(vec![segment_ids[0]])
            .unwrap();
        let batch = scanner.try_into_batch().await.unwrap();
        let i_array = batch
            .column_by_name("i")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(
            i_array
                .iter()
                .all(|v| v.is_some_and(|val| (0..200).contains(&val) || (400..410).contains(&val)))
                && i_array
                    .iter()
                    .any(|v| v.is_some_and(|val| (0..200).contains(&val)))
                && i_array
                    .iter()
                    .any(|v| v.is_some_and(|val| (400..410).contains(&val))),
            "Should get selected segment rows plus flat fallback for target fragments outside the selected segments"
        );
    }

    #[tokio::test]
    async fn test_vector_search_rejects_unknown_index_segment() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        test_ds.make_segmented_vector_index().await.unwrap();

        let query: Float32Array = (0..32).map(|v| v as f32).collect();
        let err = test_ds
            .dataset
            .scan()
            .nearest("vec", &query, 10)
            .unwrap()
            .with_index_segments(vec![Uuid::new_v4()])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("unknown index segments"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_vector_search_rejects_metric_mismatch_for_index_segments() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let segment_ids = test_ds.make_segmented_vector_index().await.unwrap();

        let query: Float32Array = (0..32).map(|v| v as f32).collect();
        let err = test_ds
            .dataset
            .scan()
            .nearest("vec", &query, 10)
            .unwrap()
            .distance_metric(DistanceType::Dot)
            .with_index_segments(vec![segment_ids[0]])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("with_index_segments requested metric"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_with_index_segments_rejects_empty_list() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let query: Float32Array = (0..32).map(|v| v as f32).collect();

        let Err(err) = test_ds
            .dataset
            .scan()
            .nearest("vec", &query, 10)
            .unwrap()
            .with_index_segments(vec![])
        else {
            panic!("expected empty index segments to be rejected");
        };
        assert!(
            err.to_string()
                .contains("with_index_segments does not accept an empty segment list"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_with_index_segments_rejected_for_non_vector_query() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let segment_ids = test_ds.make_segmented_vector_index().await.unwrap();

        let err = test_ds
            .dataset
            .scan()
            .project(&["i"])
            .unwrap()
            .with_index_segments(vec![segment_ids[0]])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("with_index_segments is only supported for vector search"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_fts_respects_fragment_list() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();

        // Create FTS index on first 2 fragments
        test_ds.make_fts_index().await.unwrap();

        // Append two more unindexed fragments
        test_ds.append_data_with_range(400, 410).await.unwrap();
        test_ds.append_data_with_range(410, 420).await.unwrap();

        // Fragment 0: i=0..200 (indexed), Fragment 1: i=200..400 (indexed)
        // Fragment 2: i=400..410 (unindexed), Fragment 3: i=410..420 (unindexed)
        let fragments = test_ds.dataset.fragments();
        assert_eq!(fragments.len(), 4);

        // "s-5" matches: s-5, s-50..s-59, s-150..s-159 (frag 0), s-250..s-259, s-350..s-359 (frag 1), s-405 (frag 2), s-415 (frag 3)
        test_fragment_list_filtering(&test_ds, fragments, |dataset| {
            let mut scanner = dataset.scan();
            scanner
                .full_text_search(FullTextSearchQuery::new("s-5".into()))
                .unwrap();
            scanner
        })
        .await;
    }
}
