// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::builder::{ListBuilder, StringBuilder, StructBuilder};
use arrow_array::types::Int32Type;
use arrow_array::{
    ArrayRef, Int32Array, LargeStringArray, RecordBatch, RecordBatchIterator, StringArray,
    record_batch,
};
use arrow_schema::{Field, Fields, Schema};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;
use lance_datafusion::exec::execute_plan;
use rstest::rstest;

fn input(batch: &RecordBatch) -> Arc<dyn ExecutionPlan> {
    // Exercise more than one input partition; filtering must preserve all columns
    // and partition properties without adding a row-id requirement.
    let mid = batch.num_rows() / 2;
    MemorySourceConfig::try_new_exec(
        &[
            vec![batch.slice(0, mid)],
            vec![batch.slice(mid, batch.num_rows() - mid)],
        ],
        batch.schema(),
        None,
    )
    .unwrap()
}

async fn rows(exec: Arc<dyn ExecutionPlan>) -> DataFusionResult<RecordBatch> {
    let schema = exec.schema();
    let batches: Vec<RecordBatch> = execute_plan(exec, Default::default())?
        .try_collect()
        .await?;
    let batch = arrow::compute::concat_batches(&schema, &batches)?;
    let order = arrow::compute::sort_to_indices(&batch["id"], None, None)?;
    Ok(arrow::compute::take_record_batch(&batch, &order)?)
}

async fn dataset(batch: RecordBatch) -> Arc<Dataset> {
    Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            "memory://",
            None,
        )
        .await
        .unwrap(),
    )
}

fn query(column: &str, granularity: Option<DocumentGranularity>) -> MatchQuery {
    let mut query = MatchQuery::new("alpha beta".into())
        .with_column(Some(column.into()))
        .with_fuzziness(Some(0))
        .with_operator(Operator::And);
    query.document_granularity = granularity;
    query
}

fn restore(exec: &FlatMatchFilterExec, decoded: Arc<dyn ExecutionPlan>) -> FlatMatchFilterExec {
    FlatMatchFilterExec::try_new(
        decoded,
        exec.dataset().clone(),
        exec.query().clone(),
        exec.params().clone(),
        exec.document_binding(),
        exec.preset_segments().map(|s| s.to_vec()),
    )
    .unwrap()
}

#[rstest]
#[case::utf8(false)]
#[case::large_utf8(true)]
#[tokio::test]
async fn flat_filter_binding_preserves_alias(#[case] is_large: bool) {
    let dataset = dataset(record_batch!(("text", Utf8, ["alpha beta"])).unwrap()).await;
    let values = vec![
        Some("alpha"),
        Some("beta"),
        Some("alpha beta"),
        None,
        Some(""),
    ];
    let text: ArrayRef = if is_large {
        Arc::new(LargeStringArray::from(values))
    } else {
        Arc::new(StringArray::from(values))
    };
    let batch = RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(Int32Array::from_iter_values(0..5)) as ArrayRef,
        ),
        ("input.text", text),
        (
            "_distance",
            Arc::new(arrow_array::Float32Array::from(vec![
                0.1, 0.2, 0.3, 0.4, 0.5,
            ])) as ArrayRef,
        ),
    ])
    .unwrap();
    let native = FlatMatchFilterExec::new_with_document_column(
        input(&batch),
        dataset,
        query("text", Some(DocumentGranularity::Row)),
        FtsSearchParams::default(),
        "input.text".into(),
    );
    assert_eq!(
        native.document_binding(),
        FtsFilterDocumentBinding::Column("input.text".into())
    );
    let rebuilt = restore(&native, input(&batch));
    assert_eq!(rebuilt.schema(), batch.schema());
    assert_eq!(
        rebuilt.properties().output_partitioning().partition_count(),
        2
    );
    let actual = rows(Arc::new(rebuilt)).await.unwrap();
    assert_eq!(actual, rows(Arc::new(native)).await.unwrap());
    assert_eq!(actual["id"].as_primitive::<Int32Type>().values(), &[2]);
    assert_eq!(actual.schema(), batch.schema());
    assert!(actual.column_by_name("_score").is_none());
}

fn nested_batch() -> RecordBatch {
    let fields = Fields::from(vec![
        Field::new("ignored", DataType::Utf8, true),
        Field::new("text", DataType::Utf8, true),
    ]);
    let mut builder = ListBuilder::new(StructBuilder::from_fields(fields, 8));
    let rows = [
        Some(vec![Some(Some("alpha")), Some(Some("beta"))]),
        Some(vec![None, Some(Some("alpha beta"))]),
        Some(vec![]),
        None,
        Some(vec![Some(None)]),
        Some(vec![Some(Some("alpha gamma beta"))]),
    ];
    for row in rows {
        if let Some(items) = row {
            for item in items {
                builder
                    .values()
                    .field_builder::<StringBuilder>(0)
                    .unwrap()
                    .append_value("alpha beta");
                builder
                    .values()
                    .field_builder::<StringBuilder>(1)
                    .unwrap()
                    .append_option(item.unwrap_or(Some("hidden alpha beta")));
                builder.values().append(item.is_some());
            }
            builder.append(true);
        } else {
            builder.append(false);
        }
    }
    RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(Int32Array::from_iter_values(0..6)) as ArrayRef,
        ),
        ("docs`raw", Arc::new(builder.finish()) as ArrayRef),
    ])
    .unwrap()
}

#[rstest]
#[case::row(DocumentGranularity::Row, vec![0, 1, 5])]
#[case::element(DocumentGranularity::ListElement, vec![1, 5])]
#[tokio::test]
async fn flat_filter_binding_preserves_resolved_documents(
    #[case] granularity: DocumentGranularity,
    #[case] expected_ids: Vec<i32>,
) {
    let batch = nested_batch();
    let dataset = dataset(batch.clone()).await;
    let path = "`docs``raw`.text";
    let resolved = resolve_fts_field(dataset.schema(), path, granularity).unwrap();
    let native = FlatMatchFilterExec::new_with_resolved_field(
        input(&batch),
        dataset,
        query(path, None),
        FtsSearchParams::default(),
        resolved,
    );
    assert_eq!(
        native.document_binding(),
        FtsFilterDocumentBinding::ResolvedField {
            path: path.into(),
            document_granularity: granularity,
        }
    );
    let rebuilt = restore(&native, input(&batch));
    assert_eq!(rebuilt.query().document_granularity, None);
    assert_eq!(rebuilt.schema(), batch.schema());
    let actual = rows(Arc::new(rebuilt)).await.unwrap();
    assert_eq!(actual, rows(Arc::new(native)).await.unwrap());
    assert_eq!(
        actual["id"].as_primitive::<Int32Type>().values().as_ref(),
        expected_ids.as_slice()
    );
    assert_eq!(actual.schema(), batch.schema());
}

#[rstest]
#[case::sensitive(false, vec![0, 2])]
#[case::folded(true, vec![0, 1, 2])]
#[tokio::test]
async fn flat_filter_binding_preserves_first_tokenizer(
    #[case] folded_first: bool,
    #[case] expected_ids: Vec<i32>,
) {
    let stored = record_batch!(("text", Utf8, ["hello", "HELLO"])).unwrap();
    let mut dataset = dataset(stored).await.as_ref().clone();
    for (name, lower_case) in [("sensitive", false), ("folded", true)] {
        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                Some(name.to_owned()),
                &InvertedIndexParams::default()
                    .lower_case(lower_case)
                    .stem(false)
                    .remove_stop_words(false),
                false,
            )
            .await
            .unwrap();
    }
    let mut segments = dataset.load_indices_by_name("sensitive").await.unwrap();
    segments.extend(dataset.load_indices_by_name("folded").await.unwrap());
    assert_eq!(segments.len(), 2);
    if folded_first {
        segments.reverse();
    }
    let expected_segments = segments.iter().map(|s| s.uuid).collect::<Vec<_>>();
    let batch = record_batch!(
        ("id", Int32, [0, 1, 2, 3]),
        ("text", Utf8, ["hello", "HELLO", "hello world", "other"])
    )
    .unwrap();
    let native = FlatMatchFilterExec::new_with_segments(
        input(&batch),
        Arc::new(dataset),
        MatchQuery::new("hello".into())
            .with_column(Some("text".into()))
            .with_fuzziness(Some(0))
            .with_document_granularity(DocumentGranularity::Row),
        FtsSearchParams::default(),
        segments,
    );
    let rebuilt = restore(&native, input(&batch));
    assert_eq!(
        rebuilt
            .preset_segments()
            .unwrap()
            .iter()
            .map(|s| s.uuid)
            .collect::<Vec<_>>(),
        expected_segments
    );
    let actual = rows(Arc::new(rebuilt)).await.unwrap();
    assert_eq!(actual, rows(Arc::new(native)).await.unwrap());
    assert_eq!(
        actual["id"].as_primitive::<Int32Type>().values().as_ref(),
        expected_ids.as_slice()
    );
}

#[rstest]
#[case::automatic(None)]
#[case::explicit(Some(1))]
#[tokio::test]
async fn flat_filter_binding_keeps_fuzzy_rejection(#[case] fuzziness: Option<u32>) {
    let batch = record_batch!(("id", Int32, [0]), ("text", Utf8, ["alpha beta"])).unwrap();
    let dataset = dataset(batch.clone()).await;
    let native = FlatMatchFilterExec::new(
        input(&batch),
        dataset,
        query("text", Some(DocumentGranularity::Row)).with_fuzziness(fuzziness),
        FtsSearchParams::default(),
    );
    let error = rows(Arc::new(restore(&native, input(&batch))))
        .await
        .unwrap_err();
    assert!(
        matches!(error, DataFusionError::NotImplemented(_)),
        "{error}"
    );
    assert!(
        error
            .to_string()
            .contains("Fuzzy MatchQuery is not supported"),
        "{error}"
    );
    assert_eq!(
        error.to_string(),
        rows(Arc::new(native)).await.unwrap_err().to_string()
    );
}

#[tokio::test]
async fn flat_filter_binding_keeps_empty_explicit_segments() {
    let batch = record_batch!(("id", Int32, [0]), ("text", Utf8, ["alpha beta"])).unwrap();
    let native = FlatMatchFilterExec::new_with_segments(
        input(&batch),
        dataset(batch.clone()).await,
        query("text", Some(DocumentGranularity::Row)),
        FtsSearchParams::default(),
        vec![],
    );
    let rebuilt = restore(&native, input(&batch));
    assert!(rebuilt.preset_segments().is_some_and(<[_]>::is_empty));
    let error = rows(Arc::new(rebuilt)).await.unwrap_err();
    assert!(matches!(error, DataFusionError::Execution(_)), "{error}");
    assert!(error.to_string().contains("has no segments"), "{error}");
    assert_eq!(
        error.to_string(),
        rows(Arc::new(native)).await.unwrap_err().to_string()
    );
}

#[rstest]
#[case::missing_alias(false, false)]
#[case::wrong_alias_type(false, true)]
#[case::missing_root(true, false)]
#[case::wrong_root_type(true, true)]
#[tokio::test]
async fn flat_filter_binding_rejects_incompatible_input(
    #[case] is_resolved: bool,
    #[case] has_wrong_type: bool,
) {
    let dataset = dataset(nested_batch()).await;
    let column = if is_resolved { "docs`raw" } else { "alias" };
    let fields = if has_wrong_type {
        vec![Field::new(column, DataType::Int32, true)]
    } else {
        vec![]
    };
    let binding = if is_resolved {
        FtsFilterDocumentBinding::ResolvedField {
            path: "`docs``raw`.text".into(),
            document_granularity: DocumentGranularity::Row,
        }
    } else {
        FtsFilterDocumentBinding::Column("alias".into())
    };
    let error = FlatMatchFilterExec::try_new(
        Arc::new(EmptyExec::new(Arc::new(Schema::new(fields)))),
        dataset,
        query("`docs``raw`.text", Some(DocumentGranularity::Row)),
        FtsSearchParams::default(),
        binding,
        None,
    )
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains(if has_wrong_type { "must" } else { "missing" }),
        "{error}"
    );
}
