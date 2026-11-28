// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableFunctionImpl, TableProvider};
use datafusion::common::{DataFusionError, ScalarValue};
use datafusion_expr::{Expr, TableType};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::projection::ProjectionExec;
use lance_arrow::SchemaExt;
use lance_core::{Error, ROW_ADDR_FIELD, ROW_ID_FIELD};
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::parser::from_json;
use lance_index::scalar::inverted::{SCORE_COL, SCORE_FIELD};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

mod hybrid;
pub use hybrid::{HybridQueryUDTF, HybridQueryUDTFBuilder};

/// Provide a table based on full text search query.
///
/// The table has the dataset's columns, then `_score`, then `_rowid` and
/// `_rowaddr` when requested.
#[derive(Debug)]
struct FtsTableProvider {
    dataset: Arc<Dataset>,
    fts_query: FullTextSearchQuery,
    full_schema: Arc<Schema>,
    ordered: bool,
}

impl FtsTableProvider {
    pub fn new(
        dataset: Arc<Dataset>,
        fts_query: FullTextSearchQuery,
        with_row_id: bool,
        with_row_addr: bool,
        ordered: bool,
    ) -> datafusion::common::Result<Self> {
        let dataset_schema = Schema::from(dataset.schema());
        if dataset_schema.column_with_name(SCORE_COL).is_some() {
            return Err(DataFusionError::Plan(format!(
                "fts() returns a {SCORE_COL} column, which conflicts with the dataset column \
                 of the same name"
            )));
        }
        let mut full_schema = dataset_schema.try_with_column(SCORE_FIELD.clone())?;
        if with_row_id {
            full_schema = full_schema.try_with_column(ROW_ID_FIELD.clone())?;
        }
        if with_row_addr {
            full_schema = full_schema.try_with_column(ROW_ADDR_FIELD.clone())?;
        }
        Ok(Self {
            dataset,
            fts_query,
            full_schema: Arc::new(full_schema),
            ordered,
        })
    }
}

#[async_trait]
impl TableProvider for FtsTableProvider {
    fn schema(&self) -> SchemaRef {
        self.full_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let mut scan = self.dataset.scan();
        scan.full_text_search(self.fts_query.clone())?;
        let columns: Vec<&str> = match projection {
            Some(projection) => projection
                .iter()
                .map(|idx| self.full_schema.field(*idx).name().as_str())
                .collect(),
            None => self
                .full_schema
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect(),
        };
        scan.disable_scoring_autoprojection();
        scan.project(&columns)?;

        let combined_filter = match filters.len() {
            0 => None,
            1 => Some(filters[0].clone()),
            _ => {
                let mut expr = filters[0].clone();
                for filter in &filters[1..] {
                    expr = Expr::and(expr, filter.clone());
                }
                Some(expr)
            }
        };
        if let Some(combined_filter) = combined_filter {
            scan.filter_expr(combined_filter);
        }
        scan.limit(limit.map(|l| l as i64), None)?;
        scan.scan_in_order(self.ordered);

        // DataFusion reads the scan's output by position, so it must have
        // exactly the requested columns. A list-element query also outputs
        // `_doc_index`, which this table does not expose.
        let plan = scan.create_plan().await.map_err(DataFusionError::from)?;
        let plan_schema = plan.schema();
        if plan_schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .eq(columns.iter().copied())
        {
            return Ok(plan);
        }
        let exprs = columns
            .iter()
            .map(|name| {
                let column = Column::new_with_schema(name, plan_schema.as_ref())?;
                Ok((Arc::new(column) as Arc<dyn PhysicalExpr>, name.to_string()))
            })
            .collect::<datafusion::common::Result<Vec<_>>>()?;
        Ok(Arc::new(ProjectionExec::try_new(exprs, plan)?))
    }
}

/// This function creates a virtual table from the input fts query.
///
/// It takes 3 parameters:
/// 1. table_name: the name of the table to be queried.
/// 2. fts_query: the fts query in json format.
/// 3. options: the query options in json format.
///
/// ```ignore
/// use crate::datafusion::LanceTableProvider;
/// use crate::dataset::udtf::FtsQueryUDTF;
/// use crate::dataset::udtf::FtsQueryUDTFBuilder;
/// use datafusion::prelude::SessionContext;
///
/// let ctx = SessionContext::new();
/// let fts_query_udtf = FtsQueryUDTFBuilder::builder()
///   .register_table("HarryPotter_Chapter1", data1.clone())
///   .register_table("HarryPotter_Chapter2", data2.clone())
///   .build();
/// ctx.register_udtf("fts", Arc::new(fts_query_udtf));
///
/// let fts_query = r#"
///             {
///                 "match": {
///                     "column": "text",
///                     "terms": "catch fish",
///                     "operator": "And"
///                 }
///             }
///      "#;
/// let options = r#"
///             {
///                 "with_row_id": true
///             }
///      "#;
/// let df = ctx
///  .sql(&format!("SELECT * FROM fts('HarryPotter_Chapter1', '{}', '{}') WHERE number > 1",
///        fts_query, options))
///  .await
///  .unwrap();
/// ```
#[derive(Debug)]
pub struct FtsQueryUDTF {
    datasets: HashMap<String, Arc<Dataset>>,
}

impl TableFunctionImpl for FtsQueryUDTF {
    fn call(&self, expr: &[Expr]) -> datafusion::common::Result<Arc<dyn TableProvider>> {
        // Parse params: table_name, fts_query, options
        if expr.len() < 2 || expr.len() > 3 {
            return Err(DataFusionError::Execution(
                "FtsQueryUDTF function takes table_name, fts_query and optional options as parameters".to_string(),
            ));
        }

        let Some(Expr::Literal(ScalarValue::Utf8(Some(table_name)), _)) = expr.first() else {
            return Err(DataFusionError::Execution(
                "FtsQueryUDTF first argument should be table name in string".to_string(),
            ));
        };

        let Some(Expr::Literal(ScalarValue::Utf8(Some(fts_query)), _)) = expr.get(1) else {
            return Err(DataFusionError::Execution(
                "FtsQueryUDTF second argument should be fts query in json format".to_string(),
            ));
        };

        let (with_row_id, with_row_addr, ordered) =
            if let Some(Expr::Literal(ScalarValue::Utf8(Some(options)), _)) = expr.get(2) {
                parse_query_options(options)?
            } else {
                (false, false, false)
            };

        // Fts query
        let dataset = self
            .datasets
            .get(table_name)
            .ok_or_else(|| DataFusionError::Execution(format!("Table {} not found", table_name)))?;

        let provider = FtsTableProvider::new(
            dataset.clone(),
            FullTextSearchQuery::new_query(from_json(fts_query)?),
            with_row_id,
            with_row_addr,
            ordered,
        )?;
        Ok(Arc::new(provider))
    }
}

fn parse_query_options(options: &str) -> datafusion::common::Result<(bool, bool, bool)> {
    let value: Value = serde_json::from_str(options)
        .map_err(|e| Error::invalid_input(format!("invalid json options: {}", e)))?;
    Ok((
        parse_bool_option(&value, "with_row_id")?,
        parse_bool_option(&value, "with_row_addr")?,
        parse_bool_option(&value, "ordered")?,
    ))
}

/// Reads a boolean option. An absent or null value is false.
fn parse_bool_option(options: &Value, key: &str) -> datafusion::common::Result<bool> {
    match options.get(key) {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        Some(value) => Err(Error::invalid_input(format!(
            "invalid fts option '{key}': expected a boolean, got {value}"
        ))
        .into()),
    }
}

/// Builder of `FtsQueryUDTF`
pub struct FtsQueryUDTFBuilder {
    datasets: HashMap<String, Arc<Dataset>>,
}

impl FtsQueryUDTFBuilder {
    pub fn builder() -> Self {
        Self {
            datasets: HashMap::new(),
        }
    }

    pub fn register_table(mut self, table_name: &str, dataset: Arc<Dataset>) -> Self {
        self.datasets.insert(table_name.to_string(), dataset);
        self
    }

    pub fn build(self) -> FtsQueryUDTF {
        FtsQueryUDTF {
            datasets: self.datasets,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Dataset;
    use crate::dataset::udtf::{FtsQueryUDTFBuilder, parse_query_options};
    use crate::index::DatasetIndexExt;
    use arrow::compute::concat_batches;
    use arrow_array::builder::{ListBuilder, StringBuilder};
    use arrow_array::cast::AsArray;
    use arrow_array::record_batch;
    use arrow_array::types::{Int32Type, Int64Type, UInt64Type};
    use arrow_array::{
        Array, ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, StringArray, UInt64Array,
    };
    use arrow_schema::{DataType, Field};
    use datafusion::common::DataFusionError;
    use datafusion::prelude::SessionContext;
    use lance_index::IndexType;
    use lance_index::scalar::InvertedIndexParams;
    use lance_index::scalar::inverted::DocumentGranularity;
    use rstest::rstest;
    use std::sync::Arc;

    #[tokio::test]
    pub async fn test_fts_query_udtf() {
        let text_col = Arc::new(StringArray::from(vec![
            "a cat catch a fish",
            "a fish catch a cat",
            "a white cat catch a big fish",
            "cat catchup fish",
            "cat fish catch",
        ]));
        let number_col = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));

        // Prepare dataset
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                Field::new("text", DataType::Utf8, false),
                Field::new("number", DataType::Int32, false),
            ])
            .into(),
            vec![text_col.clone(), number_col.clone()],
        )
        .unwrap();
        let schema = batch.schema();
        let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let mut data = Dataset::write(stream, "memory://test/table", None)
            .await
            .unwrap();
        data.create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default(),
            true,
        )
        .await
        .unwrap();
        let data = Arc::new(data);

        // Prepare datafusion
        let ctx = Arc::new(SessionContext::new());
        let fts_query_udtf = FtsQueryUDTFBuilder::builder()
            .register_table("foo", data.clone())
            .build();
        ctx.register_udtf("fts", Arc::new(fts_query_udtf));

        // Full text search
        let fts_query = r#"
            {
                "match": {
                    "column": "text",
                    "terms": "catch fish",
                    "operator": "And"
                }
            }
            "#;
        let options = r#"
            {
                "with_row_id": true
            }
            "#;

        let df = ctx
            .sql(&format!(
                "SELECT * FROM fts('foo', '{}', '{}') WHERE number > 1",
                fts_query, options
            ))
            .await
            .unwrap();

        let results = df.collect().await.unwrap();
        assert_eq!(results.len(), 1);
        let results = results.into_iter().next().unwrap();
        assert_eq!(results.num_columns(), 4); // text, number, _score, _rowid
        assert_eq!(results.num_rows(), 3);
        let row_id_col = results.column_by_name("_rowid").unwrap();
        let row_id_col = row_id_col.as_any().downcast_ref::<UInt64Array>().unwrap();
        row_id_col
            .iter()
            .for_each(|v| assert!([1u64, 2u64, 4u64].contains(&v.unwrap())));

        // The scan returns only the projected columns, so DataFusion reads
        // each column it planned on at the index it expects.
        let fts = format!("fts('foo', '{}', '{}')", fts_query, options);
        let sql = |query: String| {
            let ctx = ctx.clone();
            async move { ctx.sql(&query).await.unwrap().collect().await.unwrap() }
        };
        let results = sql(format!("SELECT count(*) AS n FROM {fts}")).await;
        assert_eq!(results[0]["n"].as_primitive::<Int64Type>().value(0), 4);

        let results = sql(format!("SELECT _rowid FROM {fts} ORDER BY _rowid DESC")).await;
        let results = concat_batches(&results[0].schema(), &results).unwrap();
        assert_eq!(results.num_columns(), 1);
        assert_eq!(
            results["_rowid"].as_primitive::<UInt64Type>().values(),
            &[4, 2, 1, 0]
        );

        let results = sql(format!(
            "SELECT number, _rowid FROM {fts} ORDER BY _rowid DESC"
        ))
        .await;
        let results = concat_batches(&results[0].schema(), &results).unwrap();
        assert_eq!(results.num_columns(), 2);
        assert_eq!(
            results["number"].as_primitive::<Int32Type>().values(),
            &[5, 3, 2, 1]
        );

        let results = sql(format!("SELECT _score FROM {fts}")).await;
        assert_eq!(results[0].num_columns(), 1);
        assert_eq!(results[0]["_score"].data_type(), &DataType::Float32);
    }

    #[tokio::test]
    async fn test_fts_query_udtf_list_element() {
        // A list-element query also outputs `_doc_index`, which the table
        // does not expose.
        let mut tags = ListBuilder::new(StringBuilder::new());
        for row in [&["cat fish", "dog"][..], &["bird"], &["fish"]] {
            for value in row {
                tags.values().append_value(value);
            }
            tags.append(true);
        }
        let batch = RecordBatch::try_from_iter([
            ("id", Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef),
            ("tags", Arc::new(tags.finish()) as ArrayRef),
        ])
        .unwrap();
        let schema = batch.schema();
        let mut data = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://",
            None,
        )
        .await
        .unwrap();
        data.create_index(
            &["tags"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().document_granularity(DocumentGranularity::ListElement),
            true,
        )
        .await
        .unwrap();
        let ctx = SessionContext::new();
        let udtf = FtsQueryUDTFBuilder::builder()
            .register_table("t", Arc::new(data))
            .build();
        ctx.register_udtf("fts", Arc::new(udtf));
        let fts = r#"fts('t', '{"match": {"column": "tags", "terms": "fish", "document_granularity": "list_element"}}')"#;

        let results = ctx
            .sql(&format!("SELECT count(*) AS n FROM {fts}"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(results[0]["n"].as_primitive::<Int64Type>().value(0), 2);

        let results = ctx
            .sql(&format!("SELECT * FROM {fts} ORDER BY id"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let results = concat_batches(&results[0].schema(), &results).unwrap();
        let schema = results.schema();
        let names: Vec<&str> = schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect();
        assert_eq!(names, ["id", "tags", "_score"]);
        assert_eq!(results["id"].as_primitive::<Int32Type>().values(), &[1, 3]);
    }

    #[tokio::test]
    async fn test_fts_query_udtf_rejects_score_column() {
        let batch = record_batch!(("text", Utf8, ["a cat"]), ("_score", Int32, [1])).unwrap();
        let schema = batch.schema();
        let data = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://",
            None,
        )
        .await
        .unwrap();
        let ctx = SessionContext::new();
        let udtf = FtsQueryUDTFBuilder::builder()
            .register_table("t", Arc::new(data))
            .build();
        ctx.register_udtf("fts", Arc::new(udtf));
        let err = ctx
            .sql(r#"SELECT * FROM fts('t', '{"match": {"column": "text", "terms": "cat"}}')"#)
            .await
            .unwrap_err();
        assert!(matches!(err, DataFusionError::Plan(_)), "{err}");
        assert!(
            err.to_string()
                .contains("conflicts with the dataset column"),
            "{err}"
        );
    }

    #[rstest]
    #[case::absent("{}", Some((false, false, false)))]
    #[case::null(r#"{"with_row_id": null}"#, Some((false, false, false)))]
    #[case::booleans(
        r#"{"with_row_id": true, "with_row_addr": false, "ordered": true}"#,
        Some((true, false, true))
    )]
    #[case::number(r#"{"with_row_id": 1}"#, None)]
    #[case::string(r#"{"ordered": "true"}"#, None)]
    fn test_parse_query_options(
        #[case] options: &str,
        #[case] expected: Option<(bool, bool, bool)>,
    ) {
        let parsed = parse_query_options(options);
        match expected {
            Some(expected) => assert_eq!(parsed.unwrap(), expected),
            None => {
                let err = parsed.unwrap_err().to_string();
                assert!(err.contains("expected a boolean"), "{err}");
            }
        }
    }
}
