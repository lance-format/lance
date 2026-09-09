// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use crate::datafusion::LanceTableProvider;
use crate::dataset::scanner::validate_batch_size;
use crate::dataset::utils::SchemaAdapter;
use arrow_array::RecordBatch;
use datafusion::common::DataFusionError;
use datafusion::dataframe::DataFrame;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::logical_expr::{Expr as LogicalExpr, LogicalPlan};
use datafusion::physical_plan::execute_stream;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion::sql::{
    parser::Statement as DFStatement,
    sqlparser::ast::{Expr, Ident, SelectItem, SetExpr, Statement},
};
use futures::TryStreamExt;
use lance_core::{ROW_ADDR, ROW_ID, datatypes::BlobHandling};
use lance_datafusion::udf::register_functions;
use std::sync::Arc;

/// A SQL builder to prepare options for running SQL queries against a Lance dataset.
#[derive(Clone, Debug)]
pub struct SqlQueryBuilder {
    /// The dataset to run the SQL query
    pub(crate) dataset: Arc<Dataset>,

    /// The SQL query to run
    pub(crate) sql: String,

    /// the name of the table to register in the datafusion context
    pub(crate) table_name: String,

    /// If true, the query result will include the internal row id
    pub(crate) with_row_id: bool,

    /// If true, the query result will include the internal row address
    pub(crate) with_row_addr: bool,

    /// Override how blob columns are materialized for this query.
    pub(crate) blob_handling: Option<BlobHandling>,

    /// Override the maximum number of rows in each scan batch.
    pub(crate) batch_size: Option<usize>,

    /// Override the approximate maximum bytes in each scan batch.
    pub(crate) batch_size_bytes: Option<u64>,
}

impl SqlQueryBuilder {
    pub fn new(dataset: Dataset, sql: &str) -> Self {
        Self {
            dataset: Arc::new(dataset),
            sql: sql.to_string(),
            table_name: "dataset".to_string(),
            with_row_id: false,
            with_row_addr: false,
            blob_handling: None,
            batch_size: None,
            batch_size_bytes: None,
        }
    }

    /// The table name to register in the datafusion context.
    /// This is used to specify a "table name" for the dataset.
    /// So that you can run SQL queries against it.
    /// If not set, the default table name is "dataset".
    pub fn table_name(mut self, table_name: &str) -> Self {
        self.table_name = table_name.to_string();
        self
    }

    /// Specify if the query result should include the internal row id.
    /// If true, the query result will include an additional column named "_rowid".
    ///
    /// The column is appended only when output rows map one-to-one to dataset
    /// rows. For other queries (DISTINCT, GROUP BY, aggregates, ...) it is not
    /// appended, but can still be referenced explicitly in the SQL text.
    pub fn with_row_id(mut self, row_id: bool) -> Self {
        self.with_row_id = row_id;
        self
    }

    /// Specify if the query result should include the internal row address.
    /// If true, the query result will include an additional column named "_rowaddr".
    ///
    /// The column is appended only when output rows map one-to-one to dataset
    /// rows. For other queries (DISTINCT, GROUP BY, aggregates, ...) it is not
    /// appended, but can still be referenced explicitly in the SQL text.
    pub fn with_row_addr(mut self, row_addr: bool) -> Self {
        self.with_row_addr = row_addr;
        self
    }

    /// Override how blob columns are materialized for this query.
    ///
    /// When unset, the underlying dataset scan uses its default
    /// [`BlobHandling::BlobsDescriptions`] policy.
    pub fn blob_handling(mut self, blob_handling: BlobHandling) -> Self {
        self.blob_handling = Some(blob_handling);
        self
    }

    /// Set the maximum number of rows produced by each query batch.
    ///
    /// The batch size must be between 1 and [`u32::MAX`], inclusive.
    ///
    /// When [`Self::batch_size_bytes`] is also set, both limits apply and the
    /// one reached first determines the scan batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set the approximate maximum number of bytes produced by each scan batch.
    ///
    /// When [`Self::batch_size`] is also set, both limits apply and the one
    /// reached first determines the scan batch size.
    pub fn batch_size_bytes(mut self, batch_size_bytes: u64) -> Self {
        self.batch_size_bytes = Some(batch_size_bytes);
        self
    }

    pub async fn build(self) -> lance_core::Result<SqlQuery> {
        if let Some(batch_size) = self.batch_size {
            validate_batch_size(batch_size)?;
        }

        let ctx = if let Some(batch_size) = self.batch_size {
            SessionContext::new_with_config(SessionConfig::new().with_batch_size(batch_size))
        } else {
            SessionContext::new()
        };
        let row_id = self.with_row_id;
        let row_addr = self.with_row_addr;
        let mut provider = LanceTableProvider::new(self.dataset.clone(), row_id, row_addr);
        if let Some(blob_handling) = self.blob_handling {
            provider = provider.with_blob_handling(blob_handling);
        }
        if let Some(batch_size) = self.batch_size {
            provider = provider.with_batch_size(batch_size);
        }
        if let Some(batch_size_bytes) = self.batch_size_bytes {
            provider = provider.with_batch_size_bytes(batch_size_bytes);
        }
        ctx.register_table(self.table_name, Arc::new(provider))?;
        register_functions(&ctx);
        let state = ctx.state();
        let dialect = state.config_options().sql_parser.dialect;
        let statement = state
            .sql_to_statement(&self.sql, &dialect)
            .map_err(planning_error)?;
        let mut projected = statement.clone();
        let columns = [(self.with_row_id, ROW_ID), (self.with_row_addr, ROW_ADDR)];
        let plan = state
            .statement_to_plan(statement)
            .await
            .map_err(planning_error)?;
        let plan = if safe_to_inject_system_columns(&plan, &columns)
            && project_system_columns(&mut projected, &columns)
        {
            // Fall back to the original plan when the rewritten statement
            // fails to plan (e.g. another expression aliased to a system
            // column name), so the query still runs without the extra columns.
            state.statement_to_plan(projected).await.unwrap_or(plan)
        } else {
            plan
        };
        let df = ctx
            .execute_logical_plan(plan)
            .await
            .map_err(planning_error)?;
        Ok(SqlQuery::new(df))
    }
}

/// Returns true when appending the enabled system columns to the query's
/// top-level SELECT list is provably safe:
///
/// 1. Row identity: every output row maps to exactly one scanned source row
///    (whitelist of row-preserving operators; aggregates, DISTINCT, joins,
///    unions, ... collapse, duplicate, or synthesize rows), so the injection
///    cannot change the other columns' values or cardinality.
/// 2. Name lineage: no intermediate projection redefines an enabled system
///    column name (e.g. `SELECT (_rowid + 1) AS _rowid` in a subquery), so
///    the injected identifiers can only bind to the real scan columns.
fn safe_to_inject_system_columns(plan: &LogicalPlan, columns: &[(bool, &str)]) -> bool {
    match plan {
        LogicalPlan::TableScan(_) => true,
        LogicalPlan::Projection(projection) => {
            let shadows_system_column = projection
                .schema
                .fields()
                .iter()
                .zip(&projection.expr)
                .filter(|(field, _)| {
                    columns
                        .iter()
                        .any(|&(enabled, name)| enabled && field.name().as_str() == name)
                })
                .any(|(field, expr)| {
                    let mut expr = expr;
                    while let LogicalExpr::Alias(alias) = expr {
                        expr = &alias.expr;
                    }
                    !matches!(expr, LogicalExpr::Column(column) if &column.name == field.name())
                });
            !shadows_system_column && safe_to_inject_system_columns(&projection.input, columns)
        }
        LogicalPlan::Filter(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::SubqueryAlias(_) => plan
            .inputs()
            .iter()
            .all(|input| safe_to_inject_system_columns(input, columns)),
        _ => false,
    }
}

/// Appends each enabled system column in `columns` to the statement's SELECT
/// list unless the query already projects it (directly or via a wildcard).
/// Returns true if the statement was modified.
///
/// Only rewrites top-level `SELECT` statements; the caller must separately
/// verify that the injection is safe (see [`safe_to_inject_system_columns`])
/// before planning the rewritten statement.
fn project_system_columns(statement: &mut DFStatement, columns: &[(bool, &str)]) -> bool {
    let DFStatement::Statement(statement) = statement else {
        return false;
    };
    let Statement::Query(query) = statement.as_mut() else {
        return false;
    };
    let SetExpr::Select(select) = query.body.as_mut() else {
        return false;
    };

    let mut changed = false;
    for &(enabled, name) in columns {
        if !enabled {
            continue;
        }
        let already_projected = select
            .projection
            .iter()
            .any(|item| projects_column(item, name));
        if already_projected {
            continue;
        }
        select
            .projection
            .push(SelectItem::UnnamedExpr(Expr::Identifier(Ident::new(name))));
        changed = true;
    }
    changed
}

/// Returns true if the SELECT item already yields the column `name`, either
/// as a bare/qualified identifier (e.g. `_rowid`, `t._rowid`) or through a
/// wildcard (`*`, `t.*`), so injecting it again would duplicate the column.
///
/// Expressions that merely reference the column (e.g. `_rowid + 1`, aliases)
/// intentionally don't count: they produce a different output column.
fn projects_column(item: &SelectItem, name: &str) -> bool {
    match item {
        SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(_, _) => true,
        SelectItem::UnnamedExpr(Expr::Identifier(ident)) => ident_matches(ident, name),
        SelectItem::UnnamedExpr(Expr::CompoundIdentifier(idents)) => idents
            .last()
            .is_some_and(|ident| ident_matches(ident, name)),
        _ => false,
    }
}

fn ident_matches(ident: &Ident, name: &str) -> bool {
    if ident.quote_style.is_some() {
        ident.value == name
    } else {
        ident.value.eq_ignore_ascii_case(name)
    }
}

pub struct SqlQuery {
    dataframe: DataFrame,
}

/// Classify a failure raised while DataFusion was building a plan.
///
/// A query that fails to plan is malformed input, but DataFusion does not
/// report these under a consistent error variant: `get_field` rejects an
/// unsupported base type with `exec_err!`, and the analyzer wraps that in
/// `Context`, so it would otherwise reach callers as an internal failure.
///
/// Only the execution category is re-classified. Every other category is either
/// already accurate or describes a failure the caller did not cause: an
/// unreachable object store is not a malformed query, and any other SQL would
/// have hit it too.
///
/// Errors Lance itself raised reach DataFusion through `External` and keep the
/// category they came with. Lance uses the execution category for its own
/// internal failures as well — a spill file that cannot be created, a poisoned
/// lock — and those are not bad input either.
fn planning_error(error: DataFusionError) -> lance_core::Error {
    let raised_by_lance = matches!(error.find_root(), DataFusionError::External(_));
    let error = lance_core::Error::from(error);
    if raised_by_lance {
        return error;
    }
    match error {
        error @ lance_core::Error::Execution { .. } => {
            lance_core::Error::invalid_input_source(Box::new(error))
        }
        error => error,
    }
}

impl SqlQuery {
    pub fn new(dataframe: DataFrame) -> Self {
        Self { dataframe }
    }

    pub async fn into_stream(self) -> lance_core::Result<SendableRecordBatchStream> {
        // Physical planning runs the analyzer, so a malformed query can still
        // fail here rather than in `SqlQueryBuilder::build`. Keep it separate
        // from execution so the two phases can be classified differently.
        let task_ctx = Arc::new(self.dataframe.task_ctx());
        let plan = self
            .dataframe
            .create_physical_plan()
            .await
            .map_err(planning_error)?;
        let exec_node = execute_stream(plan, task_ctx).map_err(lance_core::Error::from)?;
        let schema = exec_node.schema();
        if SchemaAdapter::requires_logical_conversion(&schema) {
            let adapter = SchemaAdapter::new(schema);
            Ok(adapter.to_logical_stream(exec_node))
        } else {
            Ok(exec_node)
        }
    }

    pub async fn into_batch_records(self) -> lance_core::Result<Vec<RecordBatch>> {
        self.into_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| e.into())
    }

    pub fn into_dataframe(self) -> DataFrame {
        self.dataframe
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount, assert_string_matches};
    use crate::{BlobArrayBuilder, blob_field};
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::planning_error;
    use crate::dataset::ReadParams;
    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::write::WriteParams;
    use crate::{Dataset, Error};
    use all_asserts::assert_true;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, Int64Type, UInt64Type};
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::Schema as ArrowSchema;
    use arrow_schema::{DataType, Field};
    use datafusion::common::DataFusionError;
    use lance_arrow::json::ARROW_JSON_EXT_NAME;
    use lance_arrow::{ARROW_EXT_NAME_KEY, SchemaExt};
    use lance_core::datatypes::BlobHandling;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{array, gen_batch};
    use lance_file::reader::FileReaderOptions;
    use lance_file::version::LanceFileVersion;
    use rstest::rstest;

    #[tokio::test]
    async fn test_sql_execute() {
        let ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .col("y", array::step_custom::<Int32Type>(0, 2))
            .into_dataset(
                "memory://test_sql_dataset",
                FragmentCount::from(10),
                FragmentRowCount::from(10),
            )
            .await
            .unwrap();

        let results = ds
            .sql("SELECT SUM(x) FROM foo WHERE y > 100")
            .table_name("foo")
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        pretty_assertions::assert_eq!(results.len(), 1);
        let results = results.into_iter().next().unwrap();
        pretty_assertions::assert_eq!(results.num_columns(), 1);
        pretty_assertions::assert_eq!(results.num_rows(), 1);
        // SUM(0..100) - SUM(0..50) = 3675
        pretty_assertions::assert_eq!(results.column(0).as_primitive::<Int64Type>().value(0), 3675);

        let results = ds
            .sql("SELECT x, y, _rowid, _rowaddr FROM foo where y > 100")
            .table_name("foo")
            .with_row_id(true)
            .with_row_addr(true)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let total_rows: usize = results.iter().map(|batch| batch.num_rows()).sum();
        let expect_rows = ds.count_rows(Some("y > 100".to_string())).await.unwrap();
        pretty_assertions::assert_eq!(total_rows, expect_rows);
        let results = results.into_iter().next().unwrap();
        pretty_assertions::assert_eq!(results.num_columns(), 4);
        assert_true!(results.column(2).as_primitive::<UInt64Type>().value(0) > 100);
        assert_true!(results.column(3).as_primitive::<UInt64Type>().value(0) > 100);
    }

    /// Requested system columns are appended after the user's columns when
    /// injection is safe, are not duplicated when already projected under any
    /// accepted spelling, and are skipped when a subquery alias shadows them
    /// (the injected identifiers would bind to the derived expressions and
    /// return arbitrary values as row metadata).
    #[rstest]
    #[case::plain("SELECT x FROM dataset", vec!["x", "_rowid", "_rowaddr"], vec![0, 1])]
    #[case::filter_sort_limit(
        "SELECT x FROM dataset WHERE x >= 0 ORDER BY x DESC LIMIT 2",
        vec!["x", "_rowid", "_rowaddr"],
        vec![1, 0]
    )]
    #[case::wildcard("SELECT * FROM dataset", vec!["x", "_rowid", "_rowaddr"], vec![0, 1])]
    #[case::already_projected(
        "SELECT x, _rowid, _rowaddr FROM dataset",
        vec!["x", "_rowid", "_rowaddr"],
        vec![0, 1]
    )]
    #[case::unquoted_uppercase(
        "SELECT x, _ROWID, _ROWADDR FROM dataset",
        vec!["x", "_rowid", "_rowaddr"],
        vec![0, 1]
    )]
    #[case::quoted(
        r#"SELECT x, "_rowid", "_rowaddr" FROM dataset"#,
        vec!["x", "_rowid", "_rowaddr"],
        vec![0, 1]
    )]
    #[case::table_qualified(
        "SELECT x, dataset._rowid, dataset._rowaddr FROM dataset",
        vec!["x", "_rowid", "_rowaddr"],
        vec![0, 1]
    )]
    #[case::expression_reference(
        "SELECT _rowid + 1 AS y FROM dataset",
        vec!["y", "_rowid", "_rowaddr"],
        vec![0, 1]
    )]
    #[case::system_columns_only("SELECT _rowid FROM dataset", vec!["_rowid", "_rowaddr"], vec![0, 1])]
    #[case::passthrough_subquery(
        "SELECT x FROM (SELECT x, _rowid, _rowaddr FROM dataset) s",
        vec!["x", "_rowid", "_rowaddr"],
        vec![0, 1]
    )]
    #[case::shadowed_subquery(
        "SELECT x FROM (SELECT x, (_rowid + 1) AS _rowid, (_rowaddr + 1) AS _rowaddr FROM dataset) s",
        vec!["x"],
        vec![]
    )]
    #[tokio::test]
    async fn test_sql_system_column_injection(
        #[case] sql: &str,
        #[case] expected_columns: Vec<&str>,
        #[case] expected_row_ids: Vec<u64>,
    ) {
        let ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .into_dataset(
                "memory://test_sql_system_column_injection",
                FragmentCount::from(1),
                FragmentRowCount::from(2),
            )
            .await
            .unwrap();

        let batches = ds
            .sql(sql)
            .with_row_id(true)
            .with_row_addr(true)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();

        let batch = &batches[0];
        assert_eq!(batch.schema().field_names(), expected_columns);
        for name in ["_rowid", "_rowaddr"] {
            if expected_columns.contains(&name) {
                assert_eq!(
                    batch[name].as_primitive::<UInt64Type>().values().as_ref(),
                    expected_row_ids.as_slice(),
                    "unexpected values for column {name}",
                );
            }
        }
    }

    /// System columns must never be injected into queries whose output rows
    /// are not one-to-one with dataset rows: under GROUP BY ALL or DISTINCT
    /// the injected columns would become extra grouping/dedup keys and change
    /// the relational results.
    #[rstest]
    #[case::group_by_all("SELECT x % 1 AS k, COUNT(*) AS n FROM dataset GROUP BY ALL ORDER BY k")]
    #[case::group_by_expr("SELECT x % 1 AS k, COUNT(*) AS n FROM dataset GROUP BY k ORDER BY k")]
    #[case::distinct("SELECT DISTINCT x % 1 AS k FROM dataset ORDER BY k")]
    #[case::distinct_in_subquery(
        "SELECT k FROM (SELECT DISTINCT x % 1 AS k FROM dataset) ORDER BY k"
    )]
    #[case::bare_aggregate("SELECT COUNT(*) AS n FROM dataset")]
    #[tokio::test]
    async fn test_sql_system_columns_skip_cardinality_changing_queries(#[case] sql: &str) {
        let ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .into_dataset(
                "memory://test_sql_system_columns_cardinality",
                FragmentCount::from(1),
                FragmentRowCount::from(2),
            )
            .await
            .unwrap();

        let baseline = ds
            .sql(sql)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();

        let with_system_columns = ds
            .sql(sql)
            .with_row_id(true)
            .with_row_addr(true)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();

        pretty_assertions::assert_eq!(with_system_columns, baseline);
    }

    #[tokio::test]
    async fn test_sql_batch_size() {
        let ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .into_dataset(
                "memory://test_sql_batch_size",
                FragmentCount::from(2),
                FragmentRowCount::from(25),
            )
            .await
            .unwrap();

        let batches = ds
            .sql("SELECT x FROM dataset")
            .batch_size(7)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();

        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 50);
        assert!(batches.iter().all(|batch| batch.num_rows() <= 7));
    }

    #[tokio::test]
    async fn test_sql_rejects_invalid_batch_size() {
        let ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .into_dataset(
                "memory://test_sql_rejects_invalid_batch_size",
                FragmentCount::from(1),
                FragmentRowCount::from(3),
            )
            .await
            .unwrap();

        for batch_size in [0, u32::MAX as usize + 1] {
            let error = ds
                .sql("SELECT x FROM dataset")
                .batch_size(batch_size)
                .build()
                .await
                .err()
                .expect("invalid batch size should be rejected");
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(
                error
                    .to_string()
                    .contains(&format!("batch_size must be between 1 and {}", u32::MAX))
            );
        }
    }

    #[tokio::test]
    async fn test_sql_batch_size_bytes_overrides_dataset_default() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "x",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..1000))],
        )
        .unwrap();
        let test_dir = TempStrDir::default();
        Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let dataset = DatasetBuilder::from_uri(&test_dir)
            .with_read_params(ReadParams {
                file_reader_options: Some(FileReaderOptions {
                    batch_size_bytes: Some(8_000),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();

        let batches = dataset
            .sql("SELECT x FROM dataset")
            .batch_size_bytes(64)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();

        assert_eq!(
            batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
            1000
        );
        assert!(batches.iter().all(|batch| batch.num_rows() <= 16));
    }

    #[tokio::test]
    async fn test_sql_blob_all_binary() {
        let schema = Arc::new(ArrowSchema::new(vec![blob_field("blob", true)]));
        let mut blobs = BlobArrayBuilder::new(2);
        blobs.push_bytes(b"foo").unwrap();
        blobs.push_bytes(b"bar").unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://test_sql_blob_all_binary",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let batches = dataset
            .sql("SELECT blob FROM dataset")
            .blob_handling(BlobHandling::AllBinary)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let blobs = batches[0].column(0).as_binary::<i64>();
        assert_eq!(blobs.value(0), b"foo");
        assert_eq!(blobs.value(1), b"bar");

        // Expressions over the blob column require the planner to see the
        // materialized LargeBinary type instead of the blob descriptor struct.
        let batches = dataset
            .sql("SELECT blob = X'666f6f' FROM dataset")
            .blob_handling(BlobHandling::AllBinary)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let is_foo = batches[0].column(0).as_boolean();
        assert!(is_foo.value(0));
        assert!(!is_foo.value(1));

        let batches = dataset
            .sql("SELECT blob, _rowid, _rowaddr FROM dataset")
            .with_row_id(true)
            .with_row_addr(true)
            .blob_handling(BlobHandling::AllBinary)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let batch = &batches[0];
        let blobs = batch.column(0).as_binary::<i64>();
        assert_eq!(blobs.value(0), b"foo");
        assert_eq!(blobs.value(1), b"bar");
        let row_ids = batch.column(1).as_primitive::<UInt64Type>();
        assert_eq!(row_ids.value(0), 0);
        assert_eq!(row_ids.value(1), 1);
        let row_addrs = batch.column(2).as_primitive::<UInt64Type>();
        assert_eq!(row_addrs.value(0), 0);
        assert_eq!(row_addrs.value(1), 1);
    }

    #[tokio::test]
    async fn test_sql_count() {
        let ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .col("y", array::step_custom::<Int32Type>(0, 2))
            .into_dataset(
                "memory://test_sql_dataset",
                FragmentCount::from(10),
                FragmentRowCount::from(10),
            )
            .await
            .unwrap();

        let results = ds
            .sql("SELECT COUNT(*) FROM foo")
            .table_name("foo")
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        pretty_assertions::assert_eq!(results.len(), 1);
        let results = results.into_iter().next().unwrap();
        pretty_assertions::assert_eq!(results.num_columns(), 1);
        pretty_assertions::assert_eq!(results.num_rows(), 1);
        pretty_assertions::assert_eq!(results.column(0).as_primitive::<Int64Type>().value(0), 100);

        let results = ds
            .sql("SELECT COUNT(*) FROM foo where y >= 100")
            .table_name("foo")
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        pretty_assertions::assert_eq!(results.len(), 1);
        let results = results.into_iter().next().unwrap();
        pretty_assertions::assert_eq!(results.num_columns(), 1);
        pretty_assertions::assert_eq!(results.num_rows(), 1);
        pretty_assertions::assert_eq!(results.column(0).as_primitive::<Int64Type>().value(0), 50);
    }

    #[tokio::test]
    async fn test_explain() {
        let ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .col("y", array::step_custom::<Int32Type>(0, 2))
            .into_dataset(
                "memory://test_sql_dataset",
                FragmentCount::from(10),
                FragmentRowCount::from(10),
            )
            .await
            .unwrap();

        let results = ds
            .sql("EXPLAIN SELECT * FROM foo where y >= 100")
            .table_name("foo")
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let results = results.into_iter().next().unwrap();

        let plan = format!("{:?}", results);
        let expected_pattern = r#"...columns: [StringArray
[
  "logical_plan",
  "physical_plan",
], StringArray
[
  "TableScan: foo projection=[x, y], full_filters=[foo.y >= Int32(100)]",
  "ProjectionExec: expr=[x@0 as x, y@1 as y]\n  CooperativeExec\n    LanceRead: uri=test_sql_dataset/data, projection=[x, y], num_fragments=10, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=y >= Int32(100), refine_filter=y >= Int32(100)\n",
]], row_count: 2 }"#;
        assert_string_matches(&plan, expected_pattern).unwrap();
    }

    #[tokio::test]
    async fn test_analyze() {
        let ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .col("y", array::step_custom::<Int32Type>(0, 2))
            .into_dataset(
                "memory://test_sql_dataset",
                FragmentCount::from(10),
                FragmentRowCount::from(10),
            )
            .await
            .unwrap();

        let results = ds
            .sql("EXPLAIN ANALYZE SELECT * FROM foo where y >= 100")
            .table_name("foo")
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let results = results.into_iter().next().unwrap();

        let plan = format!("{:?}", results);
        let expected_pattern = r#"...columns: [StringArray
[
  "Plan with Metrics",
], StringArray
[
  "ProjectionExec: expr=[x@0 as x, y@1 as y], metrics=[output_rows=50, elapsed_compute=...]\n  CooperativeExec, metrics=[]\n    LanceRead: uri=test_sql_dataset/data, projection=[x, y], num_fragments=..., range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=y >= Int32(100), refine_filter=y >= Int32(100), metrics=[output_rows=..., elapsed_compute=..., fragments_scanned=..., ranges_scanned=..., rows_scanned=..., bytes_read=..., iops=..., requests=..., task_wait_time=...]\n",
]], row_count: 1 }"#;
        assert_string_matches(&plan, expected_pattern).unwrap();
    }

    #[tokio::test]
    async fn test_nested_json_access() {
        let json_rows = vec![
            Some(r#"{"user": {"profile": {"name": "Alice", "settings": {"theme": "dark"}}}}"#),
            Some(r#"{"user": {"profile": {"name": "Bob", "settings": {"theme": "light"}}}}"#),
        ];
        let json_array = StringArray::from(json_rows);
        let id_array = Int32Array::from(vec![1, 2]);

        let mut metadata = HashMap::new();
        metadata.insert(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        );

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::Utf8, true).with_metadata(metadata),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(id_array), Arc::new(json_array)],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
        let ds = Dataset::write(reader, "memory://test_nested_json_access", None)
            .await
            .unwrap();

        let results = ds
            .sql(
                "SELECT id FROM dataset WHERE \
                 json_get_string(json_get(json_get(data, 'user'), 'profile'), 'name') = 'Alice'",
            )
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let batch = results.into_iter().next().unwrap();
        pretty_assertions::assert_eq!(batch.num_rows(), 1);
        pretty_assertions::assert_eq!(batch.num_columns(), 1);
        pretty_assertions::assert_eq!(batch.column(0).as_primitive::<Int32Type>().value(0), 1);

        let results = ds
            .sql(
                "SELECT id FROM dataset WHERE \
                 json_extract(data, '$.user.profile.settings.theme') = '\"dark\"'",
            )
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let batch = results.into_iter().next().unwrap();
        pretty_assertions::assert_eq!(batch.num_rows(), 1);
        pretty_assertions::assert_eq!(batch.num_columns(), 1);
        pretty_assertions::assert_eq!(batch.column(0).as_primitive::<Int32Type>().value(0), 1);
    }

    /// A malformed query is user error and must surface as [`Error::InvalidInput`]
    /// so that bindings and servers can map it to a client error rather than a
    /// generic failure.
    ///
    /// The cases differ in how DataFusion reports them: a syntax error is a
    /// `SQL` error, an unknown function is a `Plan` error wrapped in
    /// `Diagnostic`, and a subscript on a non-struct column is `exec_err!`
    /// wrapped in `Context`. Only the first is classified as user input on its
    /// own; the other two would otherwise read as internal failures.
    #[rstest]
    #[case::syntax_error("SELEC id FROM dataset", "found: SELEC at")]
    #[case::unknown_function(
        "SELECT id FROM dataset WHERE no_such_function(data) = 'Alice'",
        "no_such_function"
    )]
    #[case::subscript_on_json_column(
        "SELECT id FROM dataset WHERE data['user'] = 'Alice'",
        "Cannot access field"
    )]
    #[tokio::test]
    async fn test_sql_malformed_query_is_invalid_input(
        #[case] sql: &str,
        #[case] expected_message: &str,
    ) {
        let mut metadata = HashMap::new();
        metadata.insert(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        );
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::Utf8, true).with_metadata(metadata),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(StringArray::from(vec![Some(r#"{"user": "Alice"}"#)])),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let ds = Dataset::write(reader, "memory://test_sql_malformed_query", None)
            .await
            .unwrap();

        let result = async { ds.sql(sql).build().await?.into_batch_records().await }.await;
        let Err(error) = result else {
            panic!("expected `{sql}` to fail");
        };
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected InvalidInput, got {error:?}"
        );
        assert!(
            error.to_string().contains(expected_message),
            "expected the message to name {expected_message:?}, got: {error}"
        );
    }

    /// `LanceTableProvider::scan` runs during physical planning and reports
    /// every Lance failure as `DataFusionError::External`. Lance raises
    /// [`Error::Execution`] for its own internal failures, which is the one
    /// category `planning_error` re-classifies, so these must be recognized as
    /// Lance's own and left alone rather than blamed on the query.
    #[test]
    fn test_planning_error_keeps_lance_error_category() {
        let df_error = DataFusionError::Context(
            "while scanning".to_string(),
            Box::new(DataFusionError::from(Error::execution(
                "failed to create spill file",
            ))),
        );

        match planning_error(df_error) {
            Error::Execution { message, .. } => assert!(
                message.contains("failed to create spill file"),
                "expected the original message, got: {message}"
            ),
            other => panic!("expected the original execution error, got {other:?}"),
        }
    }

    /// A failure that is neither the caller's fault nor Lance's own — DataFusion
    /// hitting a resource limit while planning — must not be reported as a
    /// malformed query just because it surfaced during planning.
    #[test]
    fn test_planning_error_keeps_internal_datafusion_error() {
        let df_error = DataFusionError::Context(
            "while planning".to_string(),
            Box::new(DataFusionError::ResourcesExhausted(
                "failed to allocate memory".to_string(),
            )),
        );

        let error = planning_error(df_error);
        assert!(
            !matches!(error, Error::InvalidInput { .. }),
            "expected the error not to be blamed on the query, got {error:?}"
        );
    }
}
