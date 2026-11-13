// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Builder methods and option validation for [`Scanner`].
//!
//! Setters only update scanner state. They must preserve the independence of
//! batching, projection, progress, and query options; cross-option validation
//! belongs with plan construction. Unit tests follow the related setters in
//! this module.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::{Array, types::Float32Type};
use arrow_schema::{DataType, Schema as ArrowSchema};
use arrow_select::concat::concat;
use chrono::Utc;
use datafusion::common::DFSchema;
use datafusion::logical_expr::{Expr, ScalarUDF, lit};
use datafusion::physical_plan::{ExecutionPlan, expressions};
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{PhysicalExpr, create_physical_expr};
use datafusion_physical_plan::projection::ProjectionExec;
use lance_arrow::floats::{FloatType, coerce_float_vector};
use lance_core::datatypes::{BlobHandling, escape_field_path_for_project};
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_datafusion::exec::ExecutionStatsCallback;
use lance_datafusion::projection::ProjectionPlan;
use lance_file::reader::FileReaderOptions;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::minhash_lsh::MinHashQuery;
use lance_index::vector::{ApproxMode, DEFAULT_QUERY_PARALLELISM, ProbeParams, Query};
use lance_linalg::distance::MetricType;
use lance_select::{RowAddrMask, RowAddrTreeMap};
use lance_table::format::{Fragment, IndexMetadata};
use uuid::Uuid;

use crate::dataset::Dataset;
use crate::dataset::fragment_slice::FragmentSlice;
use crate::index::vector::utils::{get_vector_dim, get_vector_type, validate_distance_type_for};
use crate::io::exec::LANCE_RELATIONAL_ALGEBRA_VERSION;
use crate::io::exec::knn::QUERY_INDEX_COL;
use crate::{Error, Result};

use super::aggregate::AggregateExpr;
use super::vector::VectorRefinement;
use super::{
    BATCH_SIZE_FALLBACK, ColumnOrdering, ExprFilter, LanceFilter, MaterializationStyle,
    QueryFilter, ScalarIndexSelection, Scanner, get_default_batch_size,
};

impl Scanner {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        let projection_plan = ProjectionPlan::full(dataset.clone()).unwrap();
        let file_reader_options = dataset.file_reader_options.as_deref().cloned();
        let mut scanner = Self {
            dataset,
            projection_plan,
            blob_handling: BlobHandling::default(),
            prefilter: false,
            external_row_mask: None,
            physical_row_addr_prefilter: None,
            materialization_style: MaterializationStyle::Heuristic,
            filter: LanceFilter::default(),
            full_text_query: None,
            minhash_query: None,
            batch_size: None,
            batch_size_bytes: None,
            batch_readahead: get_num_compute_intensive_cpus(),
            fragment_readahead: None,
            io_buffer_size: None,
            materialization_readahead_bytes: None,
            limit: None,
            offset: None,
            ordering: None,
            nearest: None,
            nearest_refine: None,
            probe_params: ProbeParams::default(),
            quantized_refine_factor: None,
            nearest_query_count: 1,
            is_batch_nearest: false,
            use_stats: true,
            ordered: true,
            fragments: None,
            scalar_index_selection: None,
            index_segments: None,
            fast_search: false,
            use_scalar_index: true,
            ignored_scalar_index_names: HashSet::new(),
            include_deleted_rows: false,
            scan_stats_callback: None,
            strict_batch_size: false,
            file_reader_options,
            aggregate: None,
            legacy_with_row_addr: false,
            legacy_with_row_id: false,
            explicit_projection: false,
            autoproject_scoring_columns: true,
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

    /// Set the fragments that define the scan's search scope.
    ///
    /// An empty list produces no rows. For vector search, this scope is enforced before
    /// candidate selection even when [`Self::prefilter`] is false. An index segment
    /// covering other fragments cannot return rows from those fragments.
    ///
    /// Fragments not covered by the selected vector index segments are searched with
    /// flat KNN unless [`Self::fast_search`] is enabled. This includes fragments covered
    /// only by an unselected segment. Without a usable index, the scope is searched
    /// entirely with flat KNN (or returns no rows with fast search).
    ///
    /// This can be called before or after [`Self::nearest`] and
    /// [`Self::with_index_segments`]. For ordinary scans, [`Self::scan_in_order`] preserves
    /// the fragment list's order; vector search ranks results by distance instead.
    pub fn with_fragments(&mut self, fragments: Vec<Fragment>) -> &mut Self {
        self.fragments = Some(fragments);
        self
    }

    pub(super) fn with_scalar_index_selection(
        &mut self,
        index_name: String,
        segments: Vec<IndexMetadata>,
        segment_uuids: Option<Arc<HashSet<Uuid>>>,
    ) -> &mut Self {
        self.scalar_index_selection = Some(ScalarIndexSelection {
            index_name,
            segments: Arc::new(segments),
            segment_uuids,
        });
        self
    }

    /// Restrict vector index search to the specified index segments.
    ///
    /// This setting is only supported for vector search. When unset, Lance selects
    /// a suitable logical index and its relevant segments automatically.
    ///
    /// With [`Self::with_fragments`], ANN searches the intersection of the fragment scope
    /// and selected segment coverage. The remaining fragments are searched with flat
    /// KNN unless [`Self::fast_search`] is enabled, including fragments covered only by
    /// an unselected segment. With no fragment scope, indexed search searches only the
    /// selected segments; appended unindexed fragments are not added.
    ///
    /// Segment selection controls index use, not a flat scan when [`Self::use_index`]
    /// is false. It is independent of SQL [`Self::prefilter`] and can be set before or
    /// after [`Self::nearest`] and [`Self::with_fragments`].
    ///
    /// # Errors
    ///
    /// An empty list is rejected immediately. Segment selection without a vector query
    /// is rejected during planning. When index use is enabled, planning also rejects
    /// unknown UUIDs and segments from different logical indices or the wrong vector
    /// column. Query metric compatibility is checked when selected segments overlap the
    /// fragment scope. With no overlap, ordinary search uses flat KNN with the query
    /// metric, or the selected index's metric when unset, while fast search returns
    /// no rows. See [`Self::distance_metric`] for metric defaults.
    ///
    /// # Example
    ///
    /// Suppose the selected segment covers the first two fragments and a third fragment
    /// was appended later. Searching all but the first fragment combines ANN on the
    /// second fragment with flat KNN on the third. `query` must match the `vector` column.
    ///
    /// ```
    /// # use arrow_array::Array;
    /// # use lance::{Dataset, Result};
    /// # use uuid::Uuid;
    /// # async fn example(dataset: &Dataset, segment_uuid: Uuid, query: &dyn Array) -> Result<()> {
    /// let mut scanner = dataset.scan();
    /// scanner
    ///     .with_fragments(dataset.fragments()[1..].to_vec())
    ///     .with_index_segments(vec![segment_uuid])?
    ///     .nearest("vector", query, 6)?
    ///     .prefilter(false);
    /// let batch = scanner.try_into_batch().await?;
    /// # let _ = batch;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_index_segments(&mut self, segments: Vec<Uuid>) -> Result<&mut Self> {
        if segments.is_empty() {
            return Err(Error::invalid_input(
                "with_index_segments does not accept an empty segment list".to_string(),
            ));
        }
        self.index_segments = Some(segments);
        Ok(self)
    }

    pub(super) fn get_batch_size(&self) -> usize {
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
    ///
    /// When a list is only used by `array_length` (`list_length`), nonempty
    /// comparisons, or NULL checks, structural-format readers omit element
    /// payloads. This applies to nested element types and struct field paths.
    /// Selecting the list itself or otherwise inspecting its elements keeps the
    /// ordinary value read. Legacy array-format files keep their normal reader.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let counts = dataset.scan()
    ///     .project_with_transform(&[("image_count", "array_length(images)")])?
    ///     .try_into_batch().await?;
    /// # Ok(())
    /// # }
    /// ```
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

    /// Projection by a (possibly partial) schema.
    ///
    /// Unlike [`Self::project`], which routes through expressions, this can
    /// select *part* of a nested field — `meta` narrowed to just its `a` child.
    /// Expressions cannot express that, so a nested projection must come
    /// through here.
    ///
    /// `projection` must be a subset of the dataset schema. Fields are matched
    /// by name, nested children included, and their types must agree; anything
    /// the dataset does not have is an error rather than a silently narrower
    /// projection. Field ids are resolved against the dataset, so a schema
    /// numbered independently of it is accepted as long as the names and types
    /// line up.
    ///
    /// The resulting plan contains no complex expressions. See
    /// [`ProjectionPlan::from_schema`].
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance_core::datatypes::Schema;
    /// # fn example(dataset: &Dataset, projection: &Schema) -> Result<()> {
    /// let mut scan = dataset.scan();
    /// scan.project_with_schema(projection)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn project_with_schema(
        &mut self,
        projection: &lance_core::datatypes::Schema,
    ) -> Result<&mut Self> {
        self.explicit_projection = true;
        self.projection_plan = ProjectionPlan::from_schema(self.dataset.clone(), projection)?;
        if self.legacy_with_row_id {
            self.projection_plan.include_row_id();
        }
        if self.legacy_with_row_addr {
            self.projection_plan.include_row_addr();
        }
        self.apply_blob_handling();
        Ok(self)
    }

    /// Set whether the SQL filter runs before vector top-k selection (default: false).
    ///
    /// If true, top-k is selected from rows matching the filter. This may be more
    /// expensive and does not guarantee exact results from an approximate index.
    ///
    /// If false, the filter is applied after the combined ANN and flat top-k result.
    /// Rows rejected by the filter are not replaced, so fewer than k rows (or none)
    /// may be returned. This also applies to searches without a vector index.
    ///
    /// The fragment scope, deletions, and [`Self::with_row_id_prefilter`] restrict
    /// candidates regardless of this setting. See [`Self::with_index_segments`] for
    /// a mixed-search example.
    pub fn prefilter(&mut self, should_prefilter: bool) -> &mut Self {
        self.prefilter = should_prefilter;
        self
    }

    /// Set an external [`RowAddrMask`] allow/block prefilter.
    ///
    /// Build the mask with [`RowAddrMask::from_allowed`] to keep only the listed
    /// rows or [`RowAddrMask::from_block`] to drop them. On a vector
    /// ([`nearest`](Self::nearest)) search the mask is combined with any
    /// filter-derived prefilter on the index branch and applied to the flat
    /// branch for fragments not covered by the vector index. On a
    /// [`full_text_search`](Self::full_text_search) (match or phrase query) the
    /// mask is combined into the FTS prefilter so BM25 top-k is computed over
    /// masked rows, and the flat branch that scores unindexed fragments
    /// (plan_flat_match_query) is masked with RowAddrMaskFilterExec. On a plain
    /// scan the mask is used directly as the row source, with any
    /// [`filter`](Self::filter) applied as a refine on top.
    ///
    /// The mask is keyed in the dataset's `_rowid` space, so build it from the
    /// same dataset you query. That space is the row address when stable row ids
    /// are disabled and the stable row id when they are enabled; both are handled
    /// (index prefilter and filtered read branch on `uses_stable_row_ids`), so no
    /// caller-side translation is needed either way.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lance::dataset::Dataset;
    /// # async fn example(dataset: &Dataset) -> lance::Result<()> {
    /// use lance::dataset::scanner::{RowAddrMask, RowAddrTreeMap};
    ///
    /// // Restrict the scan to rows whose _rowid is 0, 2, or 4.
    /// let mask = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([0u64, 2, 4]));
    /// let mut scanner = dataset.scan();
    /// scanner.with_row_id_prefilter(mask);
    /// let batch = scanner.try_into_batch().await?;
    /// # let _ = batch;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_row_id_prefilter(&mut self, mask: RowAddrMask) -> &mut Self {
        self.external_row_mask = Some(Arc::new(mask));
        self
    }

    /// Deprecated name for [`Self::with_row_id_prefilter`].
    ///
    /// The mask remains in the dataset's `_rowid` domain, including stable logical IDs.
    #[deprecated(
        note = "Use with_row_id_prefilter for _rowid masks, or with_physical_row_addr_prefilter for _rowaddr selections"
    )]
    pub fn with_row_addr_prefilter(&mut self, mask: RowAddrMask) -> &mut Self {
        self.with_row_id_prefilter(mask)
    }

    /// Restrict the scan to physical slices in this dataset snapshot.
    ///
    /// Overlapping slices have set semantics. Deleted rows do not compact offsets.
    /// Empty slices are allowed, but their offsets must still be within the fragment.
    /// Any explicit fragment scope is intersected with these slices, regardless of
    /// setter order. Output follows scan order, not slice order. Replaces an existing
    /// physical selection. V1 storage, ANN and FTS are rejected when planning the scan.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::scanner::FragmentSlice;
    /// # async fn example(dataset: &Dataset, fragment_id: u32) -> Result<()> {
    /// let mut scan = dataset.scan();
    /// scan.with_fragment_slices(&[FragmentSlice {
    ///     fragment_id, row_offset: 0, row_count: 10,
    /// }]).await?;
    /// let batch = scan.try_into_batch().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn with_fragment_slices(&mut self, slices: &[FragmentSlice]) -> Result<&mut Self> {
        let rows = crate::dataset::fragment_slice::physical_rows_from_slices(&self.dataset, slices)
            .await?;
        Ok(self.with_physical_row_addr_prefilter(rows))
    }

    /// Restrict the scan to physical row addresses.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::scanner::RowAddrTreeMap;
    /// # async fn example(dataset: &Dataset, rows: RowAddrTreeMap) -> Result<()> {
    /// let mut scan = dataset.scan();
    /// scan.with_physical_row_addr_prefilter(rows);
    /// let batch = scan.try_into_batch().await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// This selection is always keyed by physical `(fragment_id, row_offset)`
    /// addresses, including on datasets that use stable row ids. It is kept in
    /// that domain until scan planning, where it is intersected with fragment
    /// scope, deletion vectors, and any scalar-index result.
    ///
    /// This is independent from [`with_row_id_prefilter`](Self::with_row_id_prefilter),
    /// whose mask is keyed in the dataset's `_rowid` domain. If both are set,
    /// both restrictions apply.
    pub fn with_physical_row_addr_prefilter(&mut self, rows: RowAddrTreeMap) -> &mut Self {
        self.physical_row_addr_prefilter = Some(Arc::new(rows));
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
        self.full_text_query = Some(query);
        Ok(self)
    }

    /// Return the rows of `query.column` most similar to `query.text` under
    /// the column's MinHash LSH index.
    ///
    /// The number of rows is the scan [`limit`](Self::limit), which is
    /// required. Results are ordered by ascending `_distance`
    /// (`1 - estimated Jaccard similarity`), which is added to the output.
    pub fn minhash_search(&mut self, query: MinHashQuery) -> Result<&mut Self> {
        if query.column.is_empty() {
            return Err(Error::invalid_input(
                "MinHash search requires the column to search".to_string(),
            ));
        }
        self.minhash_query = Some(query);
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
    /// The batch size must be between 1 and [`u32::MAX`], inclusive.
    ///
    /// When a byte limit is also configured through [`Self::batch_size_bytes`] or
    /// [`ReadParams::file_reader_options`](crate::dataset::ReadParams::file_reader_options),
    /// both limits apply and the one reached first determines the batch size.
    pub fn batch_size(&mut self, batch_size: usize) -> &mut Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set the target batch size in bytes.
    ///
    /// When set, the scanner will produce batches whose total size in bytes
    /// is approximately this value. When a row-based `batch_size` is also set,
    /// both limits apply and the one reached first determines the batch size.
    /// This cannot be combined with [`Self::strict_batch_size`] because strict
    /// row batching can merge batches beyond the byte limit.
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
    /// Generally this should scale with the number of concurrent I/O threads.  If
    /// unset, v2 scans choose a default based on the object store and
    /// `LANCE_DEFAULT_IO_BUFFER_SIZE` can override that default.
    ///
    /// This value is not a hard cap on the amount of RAM the scanner will use.  Some
    /// space is used for the compute (which can be controlled by the batch size) and
    /// Lance does not keep track of memory after it is returned to the user.
    ///
    pub fn io_buffer_size(&mut self, size: u64) -> &mut Self {
        self.io_buffer_size = Some(size);
        self
    }

    /// Set the memory budget for asynchronous blob v2 materialization.
    ///
    /// Blob descriptors are decoded before their payloads are fetched. When this
    /// budget is set, payload materialization may run ahead while the aggregate
    /// descriptor arrays, output offsets, and payload bytes awaiting ordered
    /// emission stay within `size`. Admission follows output order, and each
    /// reservation is retained until its batch is emitted. A single oversized
    /// batch is admitted when no other materialization is reserved, which
    /// guarantees forward progress. External descriptors without a stored size
    /// resolve the complete object length before admission.
    ///
    /// This budget is separate from [`Self::io_buffer_size`], which controls the
    /// storage I/O scheduler, and [`Self::batch_size_bytes`], which targets the
    /// size of individual decoded batches. If this setting is not provided,
    /// Blob v2 materialization has no independent memory bound. A size of zero
    /// is rejected when the scan plan is built.
    pub fn materialization_readahead_bytes(&mut self, size: u64) -> &mut Self {
        self.materialization_readahead_bytes = Some(size);
        self
    }

    /// Set the number of batches to decode concurrently.
    ///
    /// This bounds the decode fan-out of the scan: at most this many batch-decode
    /// tasks run in flight at once. Defaults to `get_num_compute_intensive_cpus()`.
    ///
    /// `nbatches` must be greater than zero.
    pub fn batch_readahead(&mut self, nbatches: usize) -> &mut Self {
        self.batch_readahead = nbatches;
        self
    }

    /// Set the number of fragments whose reads may be scheduled concurrently.
    ///
    /// This applies to both ordered and unordered scans. [`Self::scan_in_order`]
    /// controls result ordering, not whether fragment I/O overlaps. Set this to
    /// `1` to read one fragment at a time.
    pub fn fragment_readahead(&mut self, nfragments: usize) -> &mut Self {
        self.fragment_readahead = Some(nfragments);
        self
    }

    /// Set the target number of partitions for the physical optimizer.
    ///
    /// Overrides the default (`get_num_compute_intensive_cpus()`). Used by
    /// `EnforceDistribution` and similar rules to decide how many parallel
    /// partitions to use. Set to 1 in tests that assert specific plan shapes.
    ///
    /// A scan that reads in order (see [`Self::scan_in_order`]) and has no sort
    /// does not split its filters and projections across partitions, because
    /// merging the partitions would lose the order.
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

    /// Ignore named scalar indices from expression-filter planning.
    ///
    /// Other scalar indices remain eligible, and the original filter is still
    /// evaluated for rows selected by the remaining scan plan. Names that do
    /// not identify an available scalar index have no effect. This option is
    /// ignored when scalar indices are disabled with [`Self::use_scalar_index`].
    ///
    /// Inverted index names are also honored for expression filters such as
    /// `contains_tokens`, which fall back to evaluating the function without that
    /// index. This does not control vector index segments or the index selected
    /// by an explicit full-text search query.
    ///
    /// With [`Self::fast_search`], only a filter plan that still uses an index
    /// skips unindexed fragments. If ignoring indices removes every index query,
    /// the scan falls back to evaluating the filter over all selected fragments,
    /// potentially returning more matching rows. Explicit fragment and row
    /// selections still apply; use them to preserve a caller-defined scan scope.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn scan_without_one_index(dataset: &Dataset) -> Result<()> {
    /// let mut scanner = dataset.scan();
    /// scanner
    ///     .filter("id = 42")?
    ///     .with_ignored_scalar_indices(["id_zonemap"]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_ignored_scalar_indices<I, S>(&mut self, index_names: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.ignored_scalar_index_names = index_names.into_iter().map(Into::into).collect();
        self
    }

    /// Set whether to use strict batch size.
    ///
    /// If this is true then output batches (except the last batch) will have exactly `batch_size` rows.
    /// By default, this is False and output batches are allowed to have fewer than `batch_size` rows
    /// Setting this to True will require us to merge batches, incurring a data copy, for a minor performance
    /// penalty.
    ///
    /// This cannot be enabled when a byte limit is configured through
    /// [`Self::batch_size_bytes`] or
    /// [`ReadParams::file_reader_options`](crate::dataset::ReadParams::file_reader_options).
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

    /// Find up to k nearest neighbors within the vector column.
    ///
    /// The query can be a Float16Array, Float32Array, Float64Array, UInt8Array,
    /// or a ListArray/FixedSizeListArray of the above types. A suitable vector index
    /// is used when available; otherwise the search uses flat KNN.
    ///
    /// [`Self::with_fragments`] defines the search scope with either SQL filtering
    /// strategy. It and [`Self::with_index_segments`] can be configured before or after
    /// this method. See [`Self::with_index_segments`] for a mixed ANN and flat example,
    /// [`Self::prefilter`] for SQL filter placement, and [`Self::fast_search`] for
    /// searches limited to indexed data.
    pub fn nearest(&mut self, column: &str, q: &dyn Array, k: usize) -> Result<&mut Self> {
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
                    let offsets = list_array.value_offsets();
                    let start = offsets[0] as usize;
                    let end = offsets[list_array.len()] as usize;
                    (list_array.values().slice(start, end - start), query_count)
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
            dt if dt.is_floating() && q.data_type() == &DataType::Float32 => coerce_float_vector(
                q.as_primitive::<arrow_array::types::Float32Type>(),
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

        self.probe_params = ProbeParams::default();
        self.nearest = Some(Query {
            column: column.to_string(),
            key,
            k,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            search_effort: 0.5,
            ef: None,
            refine_factor: None,
            metric_type: None,
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
        });
        self.quantized_refine_factor = None;
        self.nearest_refine = None;
        self.nearest_query_count = query_count;
        self.is_batch_nearest = is_batch_nearest;
        Ok(self)
    }

    #[cfg(test)]
    pub(super) fn nearest_mut(&mut self) -> Option<&mut Query> {
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

    /// Configures how many partitions are searched in the vector index.
    ///
    /// Used only when [`Self::search_effort`] has not been set. Explicit minimum
    /// and maximum probe bounds override the corresponding fixed-count bounds.
    /// With none of these options set, adaptive defaults apply.
    pub fn nprobes(&mut self, n: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            self.probe_params.nprobes = Some(n);
            (q.minimum_nprobes, q.maximum_nprobes, q.search_effort) = self.probe_params.resolve();
        } else {
            log::warn!("nprobes is not set because nearest has not been called yet");
        }
        self
    }

    /// Set the initial IVF search effort to a finite value in [0, 1].
    ///
    /// Unset effort resolves to 0.5 and preserves Auto probing. Smaller values reduce the
    /// initial partition budget toward [`Self::minimum_nprobes`]; larger values
    /// increase it toward all available partitions, subject to
    /// [`Self::maximum_nprobes`]. At 0, at least one available partition is
    /// searched. Later expansion to find enough candidates remains unchanged.
    /// For L2, cosine and dot with k <= 100000, effort 0.1, 0.25 and 0.5 target
    /// 80%, 90% and 95% mean partition-routing recall, respectively. These are
    /// empirical calibration objectives, not guaranteed end-to-end Recall@k.
    /// This is independent of refinement, HNSW ef and approximation mode, and
    /// is not a recall guarantee. Setting any effort, including 0.5, ignores
    /// [`Self::nprobes`] regardless of setter order. Explicit minimum and maximum
    /// bounds remain effective. Call [`Self::nearest`] first.
    ///
    /// ```
    /// # use arrow_array::Float32Array;
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset, vector: &Float32Array) -> Result<()> {
    /// let mut scanner = dataset.scan();
    /// scanner.nearest("vector", vector, 10)?.search_effort(0.25)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn search_effort(&mut self, effort: f64) -> Result<&mut Self> {
        let query = self.nearest.as_mut().ok_or_else(|| {
            Error::invalid_input("search_effort requires nearest to be called first")
        })?;
        let previous = self.probe_params;
        self.probe_params.search_effort = Some(effort);
        (
            query.minimum_nprobes,
            query.maximum_nprobes,
            query.search_effort,
        ) = self.probe_params.resolve();
        if let Err(error) = query.validate_search_effort() {
            self.probe_params = previous;
            (
                query.minimum_nprobes,
                query.maximum_nprobes,
                query.search_effort,
            ) = previous.resolve();
            return Err(error);
        }
        Ok(self)
    }

    /// Configures how many partitions are searched in the vector index.
    #[deprecated(note = "Use nprobes instead")]
    pub fn nprobs(&mut self, n: usize) -> &mut Self {
        self.nprobes(n)
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
            self.probe_params.minimum_nprobes = Some(n);
            (q.minimum_nprobes, q.maximum_nprobes, q.search_effort) = self.probe_params.resolve();
        } else {
            log::warn!("minimum_nprobes is not set because nearest has not been called yet");
        }
        self
    }

    /// Configures the maximum number of partitions to search in the vector index.
    ///
    /// This bounds both the initial budget and later probing per index segment,
    /// without disabling Auto probing. An unset maximum (the default) allows all
    /// available partitions.
    /// With [`Self::search_effort`] set to 1, all partitions up to this limit are
    /// included in the initial search. At lower efforts, additional partitions
    /// may be searched if the initial budget does not find enough candidates.
    pub fn maximum_nprobes(&mut self, n: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            self.probe_params.maximum_nprobes = Some(n);
            (q.minimum_nprobes, q.maximum_nprobes, q.search_effort) = self.probe_params.resolve();
        } else {
            log::warn!("maximum_nprobes is not set because nearest has not been called yet");
        }
        self
    }

    /// Set the beam width used when searching an HNSW graph.
    ///
    /// Defaults to `1.5 * k * refine_factor`, floored at
    /// [`lance_index::vector::hnsw::builder::DEFAULT_MIN_EF`].
    pub fn ef(&mut self, ef: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.ef = Some(ef);
        } else {
            log::warn!("ef is not set because nearest has not been called yet");
        }
        self
    }

    /// Only search indexed data, skipping flat fallback (default: false).
    ///
    /// For vector search, only the selected index segments' coverage within the
    /// [`Self::with_fragments`] scope is searched. Fragments outside that coverage and
    /// stale vector entries requiring overlay replay are skipped. With no usable
    /// vector index coverage, the result is empty.
    ///
    /// This is a weak consistency search: newly appended or updated rows may be absent.
    /// SQL [`Self::prefilter`] placement is unchanged. See [`Self::with_index_segments`]
    /// for the default mixed-search behavior.
    ///
    /// This is essentially a weak consistency search, only on the indexed data.
    /// For scalar expression filters, this applies only when the filter plan
    /// uses an index. If no index query remains, including after
    /// [`Self::with_ignored_scalar_indices`], the scan falls back to all selected
    /// fragments while retaining the original filter and explicit row selections.
    ///
    /// Row IDs are used internally but are only returned when explicitly requested.
    pub fn fast_search(&mut self) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.use_index = true;
        }
        self.fast_search = true;
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

    /// Refine candidates using a separate Float32 fixed-size vector column.
    ///
    /// Call this after [`Self::nearest`] and set a positive [`Self::refine`]
    /// factor before execution. The coarse search returns at most `k * factor`
    /// candidates; their final `_distance` and [`Self::distance_range`] use
    /// `metric` and `query`. This replaces same-column exact refinement,
    /// including when a quantized index supplies the coarse candidates.
    /// Without an index, coarse exact TopM still precedes cross-column refinement.
    ///
    /// Both columns must contain single Float32 vectors. Query dimensions may
    /// differ, but batch queries must have matching counts and both queries must
    /// be batches. Scalar prefilters apply before candidate selection, and
    /// postfilters apply after the final TopK. Vector and full-text query filters
    /// are not supported. Null or invalid refinement vectors can result in fewer
    /// than `k` rows. Calling [`Self::nearest`] again clears this setting.
    ///
    /// ```
    /// # use arrow_array::Float32Array;
    /// # use lance::{Dataset, Result};
    /// # use lance_linalg::distance::MetricType;
    /// # async fn example(
    /// #     dataset: &Dataset,
    /// #     coarse: &Float32Array,
    /// #     full: &Float32Array,
    /// # ) -> Result<()> {
    /// let mut scan = dataset.scan();
    /// scan.nearest("short_vector", coarse, 10)?.refine(2);
    /// scan.with_refine_column("full_vector", full, MetricType::Cosine)?;
    /// let results = scan.try_into_batch().await?;
    /// # let _ = results;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_refine_column(
        &mut self,
        column: impl AsRef<str>,
        query: &dyn Array,
        metric: MetricType,
    ) -> Result<&mut Self> {
        let column = column.as_ref();
        let nearest = self.nearest.as_ref().ok_or_else(|| {
            Error::invalid_input("nearest must be set before refine_column".to_string())
        })?;
        for name in [nearest.column.as_str(), column] {
            let (vector_type, element_type) = get_vector_type(self.dataset.schema(), name)?;
            if !matches!(vector_type, DataType::FixedSizeList(_, _))
                || element_type != DataType::Float32
            {
                return Err(Error::not_supported(format!(
                    "cross-column refinement requires Float32 fixed-size vectors; column {name} has type {vector_type}"
                )));
            }
        }
        validate_distance_type_for(metric, &DataType::Float32)?;
        let query_element_type = match query.data_type() {
            DataType::List(field) | DataType::FixedSizeList(field, _) => field.data_type(),
            data_type => data_type,
        };
        if query_element_type != &DataType::Float32 {
            return Err(Error::invalid_input(format!(
                "refine_q must contain Float32 values, got {}",
                query.data_type()
            )));
        }
        if query.null_count() > 0 {
            return Err(Error::invalid_input(
                "refine_q must contain only finite, non-null values".to_string(),
            ));
        }

        let mut validation = Self::new(self.dataset.clone());
        validation.nearest(column, query, nearest.k)?;
        if validation.nearest_query_count != self.nearest_query_count
            || validation.is_batch_nearest != self.is_batch_nearest
        {
            return Err(Error::invalid_input(format!(
                "refine_q batch shape must match q: expected {} queries (batch={}), got {} (batch={})",
                self.nearest_query_count,
                self.is_batch_nearest,
                validation.nearest_query_count,
                validation.is_batch_nearest
            )));
        }

        // Respect offsets when a caller passes a slice of a batch query array.
        let vectors = if let Some(list) = query.as_list_opt::<i32>() {
            (0..list.len()).map(|i| list.value(i)).collect::<Vec<_>>()
        } else if let Some(list) = query.as_fixed_size_list_opt() {
            (0..list.len()).map(|i| list.value(i)).collect::<Vec<_>>()
        } else {
            vec![query.slice(0, query.len())]
        };
        let key = concat(
            &vectors
                .iter()
                .map(|value| value.as_ref())
                .collect::<Vec<_>>(),
        )?;
        let values = key.as_primitive::<Float32Type>();
        if values.null_count() > 0 || values.values().iter().any(|value| !value.is_finite()) {
            return Err(Error::invalid_input(
                "refine_q must contain only finite, non-null values".to_string(),
            ));
        }
        let dim = get_vector_dim(self.dataset.schema(), column)?;
        if metric == MetricType::Cosine
            && values
                .values()
                .chunks_exact(dim)
                .any(|vector| vector.iter().all(|value| *value == 0.0))
        {
            return Err(Error::invalid_input(
                "refine_q must have nonzero norm for cosine distance".to_string(),
            ));
        }

        self.nearest_refine = Some(VectorRefinement {
            column: column.to_string(),
            key,
            metric,
        });
        Ok(self)
    }

    /// Re-rank an over-fetched IVF_RQ candidate set with its multi-bit index data.
    ///
    /// Unlike [`Self::refine`], indexed candidates are re-ranked without reading
    /// their original vectors. It first discovers `factor * k` candidates with
    /// one-bit RQ scores and then re-ranks them with the index's most accurate
    /// stored RQ representation. Unindexed fragments and stale overlay rows still
    /// use flat fallback and may read original vectors. If exact refinement is also
    /// configured, the quantized stage emits enough candidates for that exact stage.
    /// When a distance range is set, the selected partitions use Accurate RQ search
    /// directly so the range applies before top-k truncation.
    ///
    /// ```
    /// # use arrow_array::Float32Array;
    /// # use lance::{Dataset, Result};
    /// # async fn search(dataset: &Dataset) -> Result<()> {
    /// let query = Float32Array::from(vec![0.0_f32; 128]);
    /// let mut scanner = dataset.scan();
    /// scanner.nearest("vector", &query, 10)?;
    /// scanner.quantized_refine(4);
    /// let results = scanner.try_into_batch().await?;
    /// # let _ = results;
    /// # Ok(())
    /// # }
    /// ```
    pub fn quantized_refine(&mut self, factor: u32) -> &mut Self {
        self.quantized_refine_factor = Some(factor);
        self
    }

    /// Change the distance [MetricType], i.e, L2 or Cosine distance.
    ///
    /// When unset, the search ranks by the metric of the vector index it chooses for the
    /// column, including when it falls back to flat KNN for filtered rows or fragments
    /// that index does not cover. A search with [`Self::use_index`] false uses the column
    /// type's default: Hamming for uint8 vectors and L2 otherwise.
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
                        column.column_name
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
    /// This setting currently affects RQ-quantized vector indexes (such as
    /// IVF_RQ) and HNSW sub-index searches that exclude rows (for example
    /// through a prefilter or deletions), where `Fast` enables the ACORN
    /// traversal. Other index types ignore this setting.
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

    /// Instruct the scanner to disable automatic projection of scoring columns
    ///
    /// In the future, this will be the default behavior.  This method is useful for
    /// opting in to the new behavior early to avoid breaking changes (and a warning
    /// message)
    ///
    /// Once the default switches, the old autoprojection behavior will be removed.
    ///
    /// The autoprojection behavior (current default) includes the _score or _distance
    /// column even if a projection is manually specified with `[project]` or
    /// `[project_with_transform]`.
    ///
    /// The new behavior will only include the _score or _distance column if no projection
    /// is specified or if the user explicitly includes the _score or _distance column
    /// in the projection.
    pub fn disable_scoring_autoprojection(&mut self) -> &mut Self {
        self.autoproject_scoring_columns = false;
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
    pub(super) fn resolved_file_reader_options(&self) -> Option<FileReaderOptions> {
        let base = self
            .file_reader_options
            .clone()
            .or_else(|| self.dataset.file_reader_options.as_deref().cloned());
        match (base, self.batch_size_bytes) {
            (Some(mut opts), Some(bsb)) => {
                opts.batch_size_bytes = Some(bsb);
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
    pub(super) fn create_column_expr(
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
            create_physical_expr(
                &expr,
                &df_schema,
                &execution_props,
                &PhysicalPlanningContext::default(),
            )
            .map_err(|e| {
                Error::internal(format!(
                    "Failed to create physical expression for nested field '{}': {}",
                    column_name, e
                ))
            })
        }
    }

    /// Ensure `input` exposes `column_name` as a top-level column.
    ///
    /// Nested FTS flat-search paths read the projected struct column from storage
    /// but the FTS executor consumes a single document column by name.
    pub(super) fn ensure_column_alias(
        &self,
        input: Arc<dyn ExecutionPlan>,
        column_name: &str,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let input_schema = input.schema();
        if input_schema.column_with_name(column_name).is_some() {
            return Ok(input);
        }

        let mut projection_exprs = Vec::with_capacity(input_schema.fields().len() + 1);
        for field in input_schema.fields() {
            projection_exprs.push((
                Arc::new(Column::new_with_schema(
                    field.name(),
                    input_schema.as_ref(),
                )?) as Arc<dyn PhysicalExpr>,
                field.name().clone(),
            ));
        }
        projection_exprs.push((
            Self::create_column_expr(column_name, self.dataset.as_ref(), input_schema.as_ref())?,
            column_name.to_string(),
        ));

        Ok(Arc::new(ProjectionExec::try_new(projection_exprs, input)?))
    }

    /// Set whether to use statistics to optimize the scan (default: true)
    ///
    /// This is used for debugging or benchmarking purposes.
    pub fn use_stats(&mut self, use_stats: bool) -> &mut Self {
        self.use_stats = use_stats;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::datatypes::Int32Type;
    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, ListBuilder};
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, UInt64Type};
    use arrow_array::{
        ArrayRef, FixedSizeListArray, Float16Array, Float32Array, Float64Array, Int32Array,
        ListArray, RecordBatch, RecordBatchIterator, StringArray, UInt8Array,
    };
    use arrow_schema::{Field as ArrowField, Fields};
    use arrow_select::concat::concat_batches;
    use datafusion::logical_expr::{col, lit};
    use futures::TryStreamExt;
    use half::f16;
    use lance_arrow::{FixedSizeListArrayExt, SchemaExt};
    use lance_core::{ROW_ADDR, ROW_ID};
    use lance_file::version::LanceFileVersion;
    use lance_index::IndexType;
    use lance_index::vector::DIST_COL;
    use lance_linalg::distance::DistanceType;
    use rstest::rstest;

    use crate::dataset::scanner::test_dataset::{
        TestVectorDataset, batch_knn_two_queries, limit_offset_equivalency_test,
        make_binary_vector_dataset,
    };
    use crate::index::DatasetIndexExt;
    use crate::index::vector::VectorIndexParams;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

    #[rstest]
    #[case::missing_fragment(42, 0, 1, "not present")]
    #[case::out_of_bounds(0, 199, 2, "outside fragment bounds")]
    #[case::empty_out_of_bounds(0, 201, 0, "outside fragment bounds")]
    #[case::overflow(0, u64::MAX, 1, "end overflow")]
    #[tokio::test]
    async fn fragment_slices_invalid(
        #[case] fragment_id: u32,
        #[case] row_offset: u64,
        #[case] row_count: u64,
        #[case] message: &str,
    ) {
        let fixture = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let mut scan = fixture.dataset.scan();
        let error = scan
            .with_fragment_slices(&[FragmentSlice {
                fragment_id,
                row_offset,
                row_count,
            }])
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message), "{error}");
        // Failed validation must not leave a partially applied selection.
        assert!(scan.physical_row_addr_prefilter.is_none());
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
    async fn test_project_with_schema() {
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
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("metadata", DataType::Struct(metadata_fields), true),
            ArrowField::new("idx", DataType::Int32, true),
        ]));
        let data = lance_datagen::rand(&schema)
            .into_ram_dataset(FragmentCount::from(7), FragmentRowCount::from(6))
            .await
            .unwrap();

        // 0 - metadata
        // 2 - x
        // 4 - age
        // A partial struct that keeps `location` but drops `y` from it.
        let projection = data.schema().project_by_ids(&[0, 2, 4], false);

        let mut scan = data.scan();
        scan.with_row_id()
            .with_row_address()
            .blob_handling(BlobHandling::AllBinary)
            .project_with_schema(&projection)
            .unwrap();
        // The blob handling configured before the projection must survive it,
        // as it does for `project_with_transform`.
        assert_eq!(
            scan.projection_plan.physical_projection.blob_handling,
            BlobHandling::AllBinary
        );
        let batch = scan.try_into_batch().await.unwrap();

        // Unlike the expression form, the output keeps the nested shape of the schema.
        let part_point_fields = Fields::from(vec![ArrowField::new("x", DataType::Float32, true)]);
        let part_metadata_fields = Fields::from(vec![
            ArrowField::new("location", DataType::Struct(part_point_fields), true),
            ArrowField::new("age", DataType::Int32, true),
        ]);
        assert_eq!(
            batch.schema().field_names(),
            vec!["metadata", ROW_ID, ROW_ADDR]
        );
        assert_eq!(
            batch["metadata"].data_type(),
            &DataType::Struct(part_metadata_fields)
        );

        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values().to_vec();
        let taken = data.take_rows(&row_ids, projection).await.unwrap();
        assert_eq!(&batch["metadata"], &taken["metadata"]);
    }

    #[tokio::test]
    async fn test_sliced_list_batch_nearest() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        test_ds.make_vector_index().await.unwrap();
        let dataset = &test_ds.dataset;
        let query_values = (0..4)
            .map(|query| {
                (query * 32..(query + 1) * 32)
                    .map(|value| Some(value as f32))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let queries = ListArray::from_iter_primitive::<Float32Type, _, _>(
            query_values.iter().cloned().map(Some),
        );

        for (start, length) in [(0, 4), (0, 2), (1, 2), (2, 2)] {
            let sliced = queries.slice(start, length);
            let equivalent = ListArray::from_iter_primitive::<Float32Type, _, _>(
                query_values[start..start + length]
                    .iter()
                    .cloned()
                    .map(Some),
            );
            let mut scanner = dataset.scan();
            scanner.nearest("vec", &sliced, 5).unwrap();
            assert_eq!(scanner.nearest_query_count, length);
            assert_eq!(scanner.nearest.as_ref().unwrap().key.len(), length * 32);
            assert_eq!(
                scanner.nearest.as_ref().unwrap().key.as_ref(),
                equivalent.values().as_ref()
            );

            for use_index in [false, true] {
                let search = |query: &ListArray| {
                    let mut scanner = dataset.scan();
                    scanner.nearest("vec", query, 5).unwrap();
                    scanner.use_index(use_index).nprobes(2);
                    scanner.project(&["i"]).unwrap();
                    scanner
                };
                let actual = search(&sliced).try_into_batch().await.unwrap();
                let expected = search(&equivalent).try_into_batch().await.unwrap();
                assert_eq!(
                    actual[QUERY_INDEX_COL].as_ref(),
                    expected[QUERY_INDEX_COL].as_ref()
                );
                assert_eq!(actual["i"].as_ref(), expected["i"].as_ref());
                assert_eq!(actual[DIST_COL].as_ref(), expected[DIST_COL].as_ref());
                assert_eq!(actual.num_rows(), length * 5);
                let row_numbers = actual["i"].as_primitive::<Int32Type>();
                assert!(row_numbers.iter().flatten().any(|value| value < 200));
                assert!(row_numbers.iter().flatten().any(|value| value >= 200));
            }
        }

        let fixed_size_queries = FixedSizeListArray::try_new_from_values(
            Float32Array::from((0..128).map(|value| value as f32).collect::<Vec<_>>()),
            32,
        )
        .unwrap();
        let mut scanner = dataset.scan();
        scanner
            .nearest("vec", &fixed_size_queries.slice(1, 2), 5)
            .unwrap();
        assert_eq!(scanner.nearest_query_count, 2);
        assert_eq!(scanner.nearest.as_ref().unwrap().key.len(), 64);
    }

    #[tokio::test]
    async fn test_sliced_list_multivector_nearest_key() {
        let mut vectors = ListBuilder::new(FixedSizeListBuilder::new(Float32Builder::new(), 2));
        for value in [0.0, 1.0] {
            vectors.values().values().append_value(value);
            vectors.values().values().append_value(value);
            vectors.values().append(true);
            vectors.append(true);
        }
        let vectors = vectors.finish();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "vec",
            vectors.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        let queries = ListArray::from_iter_primitive::<Float32Type, _, _>(vec![
            Some(vec![Some(0.0), Some(0.0)]),
            Some(vec![Some(1.0), Some(1.0)]),
            Some(vec![Some(2.0), Some(2.0)]),
            Some(vec![Some(3.0), Some(3.0)]),
        ]);
        let sliced = queries.slice(1, 2);
        let mut scanner = dataset.scan();
        scanner.nearest("vec", &sliced, 1).unwrap();
        assert!(!scanner.is_batch_nearest);
        assert_eq!(scanner.nearest_query_count, 1);
        assert_eq!(
            scanner.nearest.as_ref().unwrap().key.as_ref(),
            &Float32Array::from(vec![1.0, 1.0, 2.0, 2.0])
        );
    }

    #[tokio::test]
    async fn test_nearest_query_element_type_conversion() {
        let float16 = FixedSizeListArray::try_new_from_values(
            Float16Array::from(vec![f16::from_f32(1.0), f16::from_f32(2.0)]),
            2,
        )
        .unwrap();
        let float32 =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![1.0, 2.0]), 2).unwrap();
        let float64 =
            FixedSizeListArray::try_new_from_values(Float64Array::from(vec![1.0, 2.0]), 2).unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("float16", float16.data_type().clone(), false),
            ArrowField::new("float32", float32.data_type().clone(), false),
            ArrowField::new("float64", float64.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(float16), Arc::new(float32), Arc::new(float64)],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        let float16_query: ArrayRef = Arc::new(Float16Array::from(vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
        ]));
        let float32_query: ArrayRef = Arc::new(Float32Array::from(vec![1.0, 2.0]));
        let float64_query: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        let integer_query: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let list_query: ArrayRef =
            Arc::new(ListArray::from_iter_primitive::<Float32Type, _, _>(vec![
                Some(vec![Some(1.0), Some(2.0)]),
            ]));

        for (column, query) in [
            ("float32", &float64_query),
            ("float32", &float16_query),
            ("float32", &integer_query),
        ] {
            let Err(error) = dataset.scan().nearest(column, query.as_ref(), 1) else {
                panic!("expected {column} to reject {}", query.data_type());
            };
            assert!(matches!(error, Error::InvalidInput { .. }));
            let message = error.to_string();
            assert!(
                message.contains("Float32") && message.contains(&query.data_type().to_string()),
                "unexpected error: {message}"
            );
        }

        for (column, query, expected_type) in [
            ("float16", &float32_query, DataType::Float16),
            ("float64", &float32_query, DataType::Float64),
            ("float32", &float32_query, DataType::Float32),
            ("float64", &float64_query, DataType::Float64),
            ("float16", &float16_query, DataType::Float16),
            ("float64", &list_query, DataType::Float64),
        ] {
            let mut scan = dataset.scan();
            scan.nearest(column, query.as_ref(), 1).unwrap();
            assert_eq!(
                scan.nearest.as_ref().unwrap().key.data_type(),
                &expected_type
            );
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_search_effort_nprobes_precedence(
        #[values(0.0, 0.25, 0.5, 0.75, 1.0)] effort: f64,
        #[values(false, true)] effort_first: bool,
    ) {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        let (queries, _) = batch_knn_two_queries();
        let mut scan = test_ds.dataset.scan();
        scan.nearest("vec", &queries, 2).unwrap();
        assert_eq!(scan.probe_params.search_effort, None);
        assert_eq!(scan.nearest.as_ref().unwrap().search_effort, 0.5);
        if effort_first {
            scan.search_effort(effort).unwrap().nprobes(2);
        } else {
            scan.nprobes(2);
            assert_eq!(scan.nearest.as_ref().unwrap().maximum_nprobes, Some(2));
            scan.search_effort(effort).unwrap();
        }
        let mut reference = test_ds.dataset.scan();
        reference
            .nearest("vec", &queries, 2)
            .unwrap()
            .search_effort(effort)
            .unwrap();
        assert_eq!(
            scan.try_into_batch().await.unwrap(),
            reference.try_into_batch().await.unwrap()
        );
        assert_eq!(scan.nearest.as_ref().unwrap().minimum_nprobes, 1);
        assert_eq!(scan.nearest.as_ref().unwrap().maximum_nprobes, None);
        // Bounds supplied by the caller remain effective, even when equal.
        scan.minimum_nprobes(2).maximum_nprobes(2).nprobes(9);
        let query = scan.nearest.as_ref().unwrap();
        assert_eq!(query.minimum_nprobes, 2);
        assert_eq!(query.maximum_nprobes, Some(2));
        assert_eq!(query.search_effort, effort);
    }

    #[rstest]
    #[case::no_index(false)]
    #[case::untrained_l2_index(true)]
    #[tokio::test]
    async fn test_binary_vectors_default_to_hamming(#[case] has_untrained_index: bool) {
        let (_tmp_dir, mut dataset) = make_binary_vector_dataset().await.unwrap();
        if has_untrained_index {
            // An untrained index records its metric unchecked, and L2 cannot rank uint8.
            dataset
                .create_index_builder(
                    &["bin"],
                    IndexType::Vector,
                    &VectorIndexParams::ivf_flat(1, DistanceType::L2),
                )
                .train(false)
                .await
                .unwrap();
            let indices = dataset.load_indices().await.unwrap();
            assert_eq!(
                crate::index::vector::details::metric_type_from_index_metadata(&indices[0]),
                Some(DistanceType::L2)
            );
        }
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

        assert_eq!(scanner.quantized_refine_factor, None);
        scanner.quantized_refine(4);
        assert_eq!(scanner.quantized_refine_factor, Some(4));

        scanner.nearest("vec", &query_vector, 5).unwrap();
        assert_eq!(scanner.quantized_refine_factor, None);

        scanner.quantized_refine(0);
        let error = scanner.explain_plan(false).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("cannot be zero"));
    }

    #[tokio::test]
    async fn test_knn_probe_setters_preserve_independent_fields() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let query_vector = Float32Array::from(vec![0.0; 32]);
        let mut scanner = test_ds.dataset.scan();
        scanner.nearest("vec", &query_vector, 5).unwrap();

        scanner.nprobes(20);
        let query = scanner.nearest_mut().unwrap();
        assert_eq!(query.minimum_nprobes, 20);
        assert_eq!(query.maximum_nprobes, Some(20));

        scanner.minimum_nprobes(5);
        let query = scanner.nearest_mut().unwrap();
        assert_eq!(query.minimum_nprobes, 5);
        assert_eq!(query.maximum_nprobes, Some(20));

        scanner.maximum_nprobes(10);
        let query = scanner.nearest_mut().unwrap();
        assert_eq!(query.minimum_nprobes, 5);
        assert_eq!(query.maximum_nprobes, Some(10));
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
}
