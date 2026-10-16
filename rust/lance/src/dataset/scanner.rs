// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod aggregate;
mod builder;
mod fts;
mod minhash;
mod plan;
mod source;
#[cfg(test)]
pub mod test_dataset;
mod vector;

pub use aggregate::{AggregateExpr, AggregateExprBuilder};
pub(crate) use fts::apply_dataset_planner_auto_fuzziness_compatibility_gate;
pub use source::TakeOperation;

use std::collections::HashSet;
use std::ops::Range;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use std::task::{Context, Poll};

use crate::index::DatasetIndexExt;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema as ArrowSchema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::common::DFSchema;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream};
use datafusion_expr::ExprSchemable;
use futures::stream::{Stream, StreamExt};
use lance_core::datatypes::BlobHandling;
use lance_datafusion::aggregate::Aggregate;
use lance_datafusion::projection::ProjectionPlan;
use lance_file::reader::FileReaderOptions;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::DocumentGranularity;
use lance_index::scalar::inverted::query::{FtsQuery, FtsQueryNode, fill_fts_query_column};
use lance_index::scalar::minhash_lsh::MinHashQuery;
use lance_index::vector::{ProbeParams, Query};
use lance_io::stream::RecordBatchStream;
// Re-exported so callers of `Scanner::with_row_id_prefilter` can name the mask
// type without depending on `lance-select` directly.
pub use super::fragment_slice::FragmentSlice;
pub use lance_select::{RowAddrMask, RowAddrTreeMap};
use lance_table::format::{Fragment, IndexMetadata};
use prost::Message;
use roaring::RoaringBitmap;
use tracing::{Span, info_span, instrument};
use uuid::Uuid;

use super::Dataset;
use crate::dataset::utils::SchemaAdapter;
use crate::datatypes::Schema;
use crate::index::scalar::{IndexDetails, fetch_index_details};
use crate::io::exec::LanceFilterExec;
use crate::io::exec::{FilterPlan as ExprFilterPlan, Planner};
use crate::{Error, Result};

#[cfg(feature = "substrait")]
use futures::FutureExt;
pub use lance_datafusion::exec::{ExecutionStatsCallback, ExecutionSummaryCounts};
#[cfg(feature = "substrait")]
use lance_datafusion::substrait::parse_substrait;

/// Rows per output batch when neither the scan options nor
/// `LANCE_DEFAULT_BATCH_SIZE` specify one.
pub const BATCH_SIZE_FALLBACK: usize = 8192;

pub(crate) fn validate_batch_size(batch_size: usize) -> Result<u32> {
    let validated = u32::try_from(batch_size).map_err(|_| {
        Error::invalid_input(format!(
            "batch_size must be between 1 and {}, got {batch_size}",
            u32::MAX
        ))
    })?;
    if validated == 0 {
        return Err(Error::invalid_input(format!(
            "batch_size must be between 1 and {}, got {batch_size}",
            u32::MAX
        )));
    }
    Ok(validated)
}

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
pub(super) struct PlannedFilteredScan {
    pub(super) plan: Arc<dyn ExecutionPlan>,
    pub(super) limit_pushed_down: bool,
    pub(super) filter_pushed_down: bool,
}

#[derive(Debug)]
struct ScanRangePlan {
    fragments: Option<Arc<Vec<Fragment>>>,
    range: Range<u64>,
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
                    let cols = if fts_query.query.is_missing_column() {
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
        session: Option<&dyn Session>,
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
            plan = Arc::new(match session {
                Some(session) => {
                    LanceFilterExec::try_new_with_session(refine_expr.clone(), plan, session)?
                }
                None => LanceFilterExec::try_new(refine_expr.clone(), plan)?,
            });
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
struct ScalarIndexSelection {
    index_name: String,
    segments: Arc<Vec<IndexMetadata>>,
    segment_uuids: Option<Arc<HashSet<Uuid>>>,
}

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

    /// Optional external allow/block mask keyed in `_rowid` space. On a vector
    /// search it is combined with the index-side prefilter and applied to the
    /// flat branch for fragments not covered by the index; on a plain scan it is
    /// the row source (see `use_external_mask`). Held behind an Arc so cloning it
    /// into the ANN sub-plans and the flat-branch filter is cheap regardless of
    /// mask size.
    external_row_mask: Option<Arc<RowAddrMask>>,

    /// Optional allow-list keyed by physical `(fragment_id, row_offset)` addresses.
    /// Unlike `external_row_mask`, this remains in the physical domain until
    /// filtered-read planning intersects it with fragment scope and deletions.
    physical_row_addr_prefilter: Option<Arc<RowAddrTreeMap>>,

    /// Materialization style controls when columns are fetched
    materialization_style: MaterializationStyle,

    /// Filter.
    filter: LanceFilter,

    /// Optional full text search query
    full_text_query: Option<FullTextSearchQuery>,

    /// MinHash similarity search. The scan source is the index lookup and the
    /// output gains a `_distance` column.
    minhash_query: Option<MinHashQuery>,

    /// The batch size controls the maximum size of rows to return for each read.
    batch_size: Option<usize>,

    /// If set, the scanner will produce batches whose total size in bytes
    /// is approximately this value. When both limits are set, the scanner uses
    /// the smaller row count selected by either limit.
    batch_size_bytes: Option<u64>,

    /// Number of batches to decode concurrently
    batch_readahead: usize,

    /// Number of fragments to read concurrently
    fragment_readahead: Option<usize>,

    /// Number of bytes to allow to queue up in the I/O buffer
    io_buffer_size: Option<u64>,

    /// Total bytes reserved by asynchronously materialized blob v2 batches
    materialization_readahead_bytes: Option<u64>,

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
    probe_params: ProbeParams,
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

    /// If set, scalar-index planning is restricted to this logical index and optional segment set.
    scalar_index_selection: Option<ScalarIndexSelection>,

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
    /// Whether the user explicitly requested a projection.  If they did then we will warn them
    /// if they do not specify _score / _distance unless legacy_projection_behavior is set to false
    explicit_projection: bool,
    /// Whether the user wants to use the legacy projection behavior.
    autoproject_scoring_columns: bool,
}

impl Dataset {
    /// Count rows matching `filter` while pinning scalar-index planning to `index_name`.
    ///
    /// When `segment_uuids` is present, only those physical index segments are opened and their
    /// current fragment coverage defines the count scope. That coverage is accepted only when the
    /// selection includes every segment that contributes to it. After a fragment-reuse rewrite, one
    /// source segment can advertise every destination fragment while still depending on sibling
    /// segments; an incomplete selection is rejected. A supplied `fragment_ids` list must exactly
    /// equal the selected segments' current coverage.
    #[instrument(skip_all)]
    pub async fn count_indexed_rows(
        &self,
        index_name: &str,
        filter: &str,
        segment_uuids: Option<&[Uuid]>,
        fragment_ids: Option<&[u32]>,
    ) -> Result<u64> {
        if index_name.is_empty() {
            return Err(Error::invalid_input("index_name must not be empty"));
        }
        if filter.is_empty() {
            return Err(Error::invalid_input("filter must not be empty"));
        }

        let requested_fragments = fragment_ids
            .map(|fragment_ids| {
                let fragment_set = fragment_ids.iter().copied().collect::<RoaringBitmap>();
                if fragment_set.len() != fragment_ids.len() as u64 {
                    return Err(Error::invalid_input(format!(
                        "fragment_ids contains duplicate fragment IDs: {fragment_ids:?}"
                    )));
                }
                let unknown = &fragment_set - self.fragment_bitmap.as_ref();
                if !unknown.is_empty() {
                    return Err(Error::invalid_input(format!(
                        "fragment_ids contains unknown fragment IDs: {:?}",
                        unknown.iter().collect::<Vec<_>>()
                    )));
                }
                Ok(fragment_set)
            })
            .transpose()?;

        let all_indices = self.load_indices().await?;
        let (selected_segments, selected_segment_uuids, fragment_scope) = if let Some(
            segment_uuids,
        ) = segment_uuids
        {
            if segment_uuids.is_empty() {
                return Err(Error::invalid_input(
                    "segment_uuids must contain at least one UUID",
                ));
            }
            let selected_uuid_set = segment_uuids.iter().copied().collect::<HashSet<_>>();
            if selected_uuid_set.len() != segment_uuids.len() {
                return Err(Error::invalid_input(format!(
                    "segment_uuids contains duplicate UUIDs: {segment_uuids:?}"
                )));
            }

            let mut selected_segments = Vec::with_capacity(segment_uuids.len());
            let mut selected_coverage = RoaringBitmap::new();
            for segment_uuid in segment_uuids {
                let segment = all_indices
                    .iter()
                    .find(|segment| segment.uuid == *segment_uuid)
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "Scalar index segment {segment_uuid} does not exist"
                        ))
                    })?;
                if segment.name != index_name {
                    return Err(Error::invalid_input(format!(
                        "Scalar index segment {segment_uuid} belongs to index '{}', not requested index '{index_name}'",
                        segment.name
                    )));
                }
                let Some(field_id) = segment.keyed_field() else {
                    return Err(Error::invalid_input(format!(
                        "Scalar index '{index_name}' segment {segment_uuid} has no keyed field"
                    )));
                };
                let field = self.schema().field_by_id(field_id).ok_or_else(|| {
                        Error::internal(format!(
                            "Scalar index '{index_name}' segment {segment_uuid} references missing field ID {field_id}"
                        ))
                    })?;
                let field_path =
                    if let Some(ancestors) = self.schema().field_ancestry_by_id(field.id) {
                        let field_refs = ancestors
                            .iter()
                            .map(|field| field.name.as_str())
                            .collect::<Vec<_>>();
                        lance_core::datatypes::format_field_path(&field_refs)
                    } else {
                        field.name.clone()
                    };
                let details = IndexDetails(fetch_index_details(self, &field_path, segment).await?);
                if details.is_vector() || details.get_plugin().is_err() {
                    return Err(Error::invalid_input(format!(
                        "Index '{index_name}' segment {segment_uuid} is not a usable scalar index segment"
                    )));
                }
                let coverage = segment.fragment_bitmap.as_ref().ok_or_else(|| {
                        Error::invalid_input(format!(
                            "Scalar index '{index_name}' segment {segment_uuid} is missing fragment coverage"
                        ))
                    })?;
                let current_coverage = coverage & self.fragment_bitmap.as_ref();
                if current_coverage.is_empty() {
                    return Err(Error::invalid_input(format!(
                        "Scalar index '{index_name}' segment {segment_uuid} does not cover any current dataset fragments"
                    )));
                }
                selected_coverage |= current_coverage;
                selected_segments.push(segment.clone());
            }
            // Rewritten coverage is the contributor closure: a sibling whose bitmap
            // overlaps this scope still owns rows in those fragments. Opening only
            // the requested UUIDs would publish that scope and then under-count.
            reject_incomplete_segment_contributors(
                index_name,
                all_indices.as_ref(),
                &selected_uuid_set,
                &selected_coverage,
            )?;

            if let Some(requested_fragments) = requested_fragments.as_ref()
                && requested_fragments != &selected_coverage
            {
                return Err(Error::invalid_input(format!(
                    "fragment_ids {:?} do not match selected segment coverage {:?} for scalar index '{index_name}'",
                    requested_fragments.iter().collect::<Vec<_>>(),
                    selected_coverage.iter().collect::<Vec<_>>()
                )));
            }

            (
                selected_segments,
                Some(Arc::new(selected_uuid_set)),
                Some(selected_coverage),
            )
        } else {
            let selected_segments = all_indices
                .iter()
                .filter(|segment| segment.name == index_name)
                .cloned()
                .collect::<Vec<_>>();
            if selected_segments.is_empty() {
                return Err(Error::invalid_input(format!(
                    "Scalar index '{index_name}' does not exist or has no usable segments"
                )));
            }
            (selected_segments, None, requested_fragments)
        };

        let mut scanner = self.scan();
        scanner.filter(filter)?;
        scanner.project::<String>(&[])?;
        scanner.with_row_id();
        scanner.with_scalar_index_selection(
            index_name.to_string(),
            selected_segments,
            selected_segment_uuids,
        );
        if let Some(fragment_scope) = fragment_scope {
            let fragments = self
                .fragments()
                .iter()
                .filter(|fragment| fragment_scope.contains(fragment.id as u32))
                .cloned()
                .collect();
            scanner.with_fragments(fragments);
        }
        scanner.count_rows().await
    }
}

/// Reject a UUID selection whose advertised fragments still depend on an unselected segment.
fn reject_incomplete_segment_contributors(
    index_name: &str,
    indices: &[IndexMetadata],
    selected_uuids: &HashSet<Uuid>,
    selected_coverage: &RoaringBitmap,
) -> Result<()> {
    let mut missing = Vec::new();
    let mut shared_fragments = RoaringBitmap::new();
    for segment in indices {
        if segment.name != index_name || selected_uuids.contains(&segment.uuid) {
            continue;
        }
        let Some(coverage) = segment.fragment_bitmap.as_ref() else {
            continue;
        };
        let overlap = coverage & selected_coverage;
        if overlap.is_empty() {
            continue;
        }
        shared_fragments |= overlap;
        missing.push(segment.uuid);
    }
    if missing.is_empty() {
        return Ok(());
    }
    missing.sort_unstable();
    Err(Error::invalid_input(format!(
        "Scalar index '{index_name}' selection omits contributing segments {missing:?} for fragments {:?}. Select every segment that covers those fragments",
        shared_fragments.iter().collect::<Vec<_>>()
    )))
}

// Search over all indexed fields including nested ones, collecting columns that have an
// inverted index. Automatic discovery is intentionally restricted to Row
// granularity because ListElement queries require an explicit field path.
async fn fts_indexed_columns(dataset: Arc<Dataset>) -> Result<Vec<String>> {
    let mut indexed_columns = Vec::new();
    for index in dataset.load_indices().await?.iter() {
        let Some(field_id) = index.fields.first().copied() else {
            continue;
        };
        let Ok(preliminary_path) = dataset.schema().field_path(field_id) else {
            continue;
        };
        let details =
            crate::index::scalar::fetch_index_details(dataset.as_ref(), &preliminary_path, index)
                .await?;
        if !details.type_url.ends_with("InvertedIndexDetails") {
            continue;
        }
        let details = lance_index::pbold::InvertedIndexDetails::decode(details.value.as_slice())
            .map_err(|error| {
                Error::io(format!(
                    "failed to decode InvertedIndexDetails payload: {error}"
                ))
            })?;
        if DocumentGranularity::try_from(details.document_granularity)? != DocumentGranularity::Row
        {
            continue;
        }
        let resolved = crate::index::scalar::inverted::resolve_fts_field_by_id(
            dataset.schema(),
            field_id,
            DocumentGranularity::Row,
        )?;
        indexed_columns.push(resolved.canonical_path);
    }
    indexed_columns.sort();
    indexed_columns.dedup();
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
mod tests {
    use super::*;

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
}
