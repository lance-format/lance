// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;
use std::vec;

use crate::index::vector::VectorIndexParams;
use crate::io::exec::fts::FlatMatchFilterExec;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::physical_plan::ExecutionPlan;
use lance_arrow::json::{ARROW_JSON_EXT_NAME, JsonArray, is_arrow_json_field, json_field};
use lance_arrow::{ARROW_EXT_NAME_KEY, FixedSizeListArrayExt, RecordBatchExt};

use crate::index::DatasetIndexExt;
use arrow::compute::concat_batches;
use arrow::datatypes::Int32Type;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, LargeListArray, ListArray, StructArray};
use arrow_array::{Decimal128Array, Float32Array, Int32Array, RecordBatch, RecordBatchIterator};
use arrow_array::{Int64Array, StringArray, UInt32Array, UInt64Array, record_batch};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema, SchemaRef};
use arrow_select::take::take_record_batch;
use futures::TryStreamExt;
use lance_arrow::SchemaExt;
use lance_core::cache::LanceCache;
use lance_datafusion::exec::execute_plan;
use lance_encoding::decoder::DecoderPlugins;
use lance_file::reader::{FileReader, FileReaderOptions, describe_encoding};
use lance_file::version::LanceFileVersion;
use lance_index::scalar::inverted::{
    SCORE_FIELD,
    query::{FtsQuery, MatchQuery, Operator, PhraseQuery},
    tokenizer::InvertedIndexParams,
};
use lance_index::scalar::{FullTextSearchQuery, ScalarIndexParams};
use lance_index::{IndexType, vector::DIST_COL};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_linalg::distance::MetricType;
use uuid::Uuid;

use crate::dataset::NewColumnTransform;
use crate::dataset::scanner::{DatasetRecordBatchStream, QueryFilter};
use crate::dataset::write::WriteParams;
use crate::{Dataset, Error};
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::{DEFAULT_QUERY_PARALLELISM, Query};
use pretty_assertions::assert_eq;
use rstest::rstest;

#[rstest]
#[case::struct_null_check(false, "c IS NOT NULL", vec![0, 1, 2])]
#[case::struct_value(false, "c.t > 0", vec![1, 2])]
#[case::struct_null(false, "c IS NULL", vec![3])]
#[case::list_null_check(true, "c IS NOT NULL", vec![0, 1, 2])]
#[case::list_null(true, "c IS NULL", vec![3])]
#[tokio::test]
async fn test_partial_projection_with_filter_on_same_column(
    #[case] is_list: bool,
    #[case] filter: &str,
    #[case] expected_indices: Vec<u32>,
) {
    let children = record_batch!(
        ("t", Int32, [-1, 2, 3, 4]),
        ("b", Utf8, [Some("x"), Some("y"), None, Some("z")])
    )
    .unwrap();
    let item = StructArray::new(
        children.schema().fields().clone(),
        children.columns().to_vec(),
        Some(NullBuffer::from(vec![true, true, true, false])),
    );
    let c: ArrayRef = if is_list {
        Arc::new(ListArray::new(
            Arc::new(ArrowField::new("item", item.data_type().clone(), true)),
            OffsetBuffer::from_lengths([1, 0, 2, 1]),
            Arc::new(item),
            Some(NullBuffer::from(vec![true, true, true, false])),
        ))
    } else {
        Arc::new(item)
    };
    let batch = record_batch!(("k", Int32, [0, 1, 2, 3]))
        .unwrap()
        .try_with_column(ArrowField::new("c", c.data_type().clone(), true), c)
        .unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);

    let leaf = if is_list { "c.item.b" } else { "c.b" };
    let projection = dataset.schema().project(&["k", leaf]).unwrap();
    let expected_schema = ArrowSchema::from(&projection);
    let expected = batch.project_by_schema(&expected_schema).unwrap();
    let expected = take_record_batch(&expected, &UInt32Array::from(expected_indices)).unwrap();

    let mut scan = dataset.scan();
    scan.project_with_schema(&projection).unwrap();
    scan.filter(filter).unwrap();
    let result = scan.try_into_batch().await.unwrap();
    assert_eq!(result.schema().as_ref(), &expected_schema, "{filter}");
    assert_eq!(result, expected, "{filter}");
}

/// A null struct must not read back as a valid struct with null children.
///
/// A scan merges the per-column batches with `lance_arrow::merge`, which used to read an all-null
/// validity buffer as "this side carries no validity" and drop it. A filter that selects only null
/// rows leaves exactly that shape, so the scan reported those rows as valid while `IS NULL`
/// counted them as null. The dataset is created empty and then appended to because that is the
/// path this was found on, and the version is pinned because 2.0 does not encode struct validity.
#[tokio::test]
async fn test_filtered_scan_preserves_nullable_struct_validity() {
    let struct_fields = Fields::from(vec![
        ArrowField::new("a", DataType::Int64, true),
        ArrowField::new("b", DataType::Utf8, true),
    ]);
    let item_field = Arc::new(ArrowField::new(
        "item",
        DataType::Struct(struct_fields.clone()),
        true,
    ));
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::UInt64, false),
        ArrowField::new("s", DataType::Struct(struct_fields.clone()), true),
        ArrowField::new("l", DataType::List(item_field.clone()), true),
    ]));

    let empty = RecordBatch::new_empty(schema.clone());
    let reader = RecordBatchIterator::new([Ok(empty)], schema.clone());
    let mut dataset = Dataset::write(
        reader,
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..WriteParams::default()
        }),
    )
    .await
    .unwrap();

    // Rows 100 and 177 are null in both nested columns while their children still carry values,
    // so losing the top-level validity turns them into valid values instead of null ones.
    let validity = NullBuffer::from(vec![false, true, false]);
    let structs = StructArray::new(
        struct_fields.clone(),
        vec![
            Arc::new(Int64Array::from(vec![Some(10), Some(11), Some(12)])),
            Arc::new(StringArray::from(vec![Some("x"), Some("y"), Some("z")])),
        ],
        Some(validity.clone()),
    );
    let items = StructArray::new(
        struct_fields.clone(),
        vec![
            Arc::new(Int64Array::from(vec![Some(20), Some(21), Some(22)])),
            Arc::new(StringArray::from(vec![Some("p"), Some("q"), Some("r")])),
        ],
        None,
    );
    let lists = ListArray::new(
        item_field,
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 2, 3])),
        Arc::new(items),
        Some(validity),
    );
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![100, 116, 177])),
            Arc::new(structs),
            Arc::new(lists),
        ],
    )
    .unwrap();
    dataset
        .append(
            Box::new(RecordBatchIterator::new([Ok(batch)], schema)),
            None,
        )
        .await
        .unwrap();

    for (id, expected_is_null) in [(100, true), (116, false), (177, true)] {
        let mut scan = dataset.scan();
        scan.filter(&format!("id = {id}")).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        let structs = batch["s"].as_struct();
        assert_eq!(structs.is_null(0), expected_is_null, "s, row id {id}");
        // A null struct masks its children, so both levels have to agree.
        assert_eq!(
            structs.column(0).is_null(0),
            expected_is_null,
            "s.a, row id {id}"
        );
        assert_eq!(
            batch["l"].as_list::<i32>().is_null(0),
            expected_is_null,
            "l, row id {id}"
        );
    }

    assert_eq!(
        dataset
            .count_rows(Some("s IS NULL".to_owned()))
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        dataset
            .count_rows(Some("l IS NULL".to_owned()))
            .await
            .unwrap(),
        2
    );

    // A struct column added as all-nulls reaches the same merge through schema evolution, where
    // every row is null and there is no other side to recover the validity from.
    dataset
        .add_columns(
            NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "t",
                DataType::Struct(struct_fields),
                true,
            )]))),
            None,
            None,
        )
        .await
        .unwrap();
    let mut scan = dataset.scan();
    scan.filter("id = 116").unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    assert!(batch["t"].as_struct().is_null(0));
    assert_eq!(
        dataset
            .count_rows(Some("t IS NULL".to_owned()))
            .await
            .unwrap(),
        3
    );
}

#[tokio::test]
async fn test_scan_wide_fixed_size_list_at_batch_boundary() {
    const DIM_A: usize = 140_000;
    const DIM_B: usize = 4_096;
    const SHORT_ROWS: usize = 68;
    const LONG_ROWS: usize = 128;

    fn make_batch(schema: SchemaRef, rows: usize, base: usize) -> RecordBatch {
        let values_a = Float32Array::from_iter_values(
            (0..rows * DIM_A).map(|idx| ((idx + base) % 1009) as f32 / 1009.0),
        );
        let values_b = Float32Array::from_iter_values(
            (0..rows * DIM_B).map(|idx| ((idx + base) % 251) as f32 / 251.0),
        );
        let arr_a = FixedSizeListArray::try_new_from_values(values_a, DIM_A as i32).unwrap();
        let arr_b = FixedSizeListArray::try_new_from_values(values_b, DIM_B as i32).unwrap();
        RecordBatch::try_new(schema, vec![Arc::new(arr_a), Arc::new(arr_b)]).unwrap()
    }

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(
            "a",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                DIM_A as i32,
            ),
            true,
        ),
        ArrowField::new(
            "b",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                DIM_B as i32,
            ),
            true,
        ),
    ]));

    let batches = vec![
        make_batch(schema.clone(), SHORT_ROWS, 0),
        make_batch(schema.clone(), LONG_ROWS, 17),
    ];
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let write_params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_1),
        ..WriteParams::default()
    };
    let dir = tempfile::tempdir().unwrap();
    let dataset = Dataset::write(reader, dir.path().to_str().unwrap(), Some(write_params))
        .await
        .unwrap();

    // The first column splits into 9 read chunks. The second column is a
    // higher-priority request that can reserve the remaining buffer while the
    // first column is still awaited.
    let mut scanner = dataset.scan();
    scanner.io_buffer_size(70 * 1024 * 1024);
    scanner
        .limit(Some(LONG_ROWS as i64), Some(SHORT_ROWS as i64))
        .unwrap();
    let mut stream = tokio::time::timeout(
        std::time::Duration::from_secs(20),
        scanner.try_into_stream(),
    )
    .await
    .expect("stream creation timed out")
    .unwrap();
    let batch = tokio::time::timeout(std::time::Duration::from_secs(20), stream.try_next())
        .await
        .expect("first batch timed out")
        .unwrap()
        .unwrap();

    assert_eq!(batch.num_rows(), LONG_ROWS);
}

#[tokio::test]
async fn test_vector_filter_fts_search() {
    let dataset = prepare_query_filter_dataset().await;
    let schema: ArrowSchema = dataset.schema().into();

    let query_vector = Arc::new(Float32Array::from(vec![300f32, 300f32, 300f32, 300f32]));
    let vector_query = Query {
        column: "vector".to_string(),
        key: query_vector,
        k: 5,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: 20,
        maximum_nprobes: None,
        search_effort: 0.5,
        ef: None,
        refine_factor: None,
        metric_type: Some(MetricType::L2),
        use_index: true,
        query_parallelism: DEFAULT_QUERY_PARALLELISM,
        dist_q_c: 0.0,
        approx_mode: Default::default(),
    };

    // Refine the approximate IVF-PQ distances so adjacent vectors have a
    // deterministic distance order for the strict assertion below.
    let mut case1_vector_query = vector_query.clone();
    case1_vector_query.refine_factor = Some(1);

    // Case 1: search with prefilter=true, query_filter=vector([300,300,300,300])
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new("text".to_string()))
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(case1_vector_query))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300, 299],
    )
    .await;

    // Case 2: search with prefilter=true, query_filter=vector([300,300,300,300]), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new("text".to_string()))
        .unwrap()
        .prefilter(true)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300],
    )
    .await;

    // Case 3: search with prefilter=true, phrase query, query_filter=vector([300,300,300,300])
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[299, 300],
    )
    .await;

    // Case 4: search with prefilter=true, phrase query, query_filter=vector([300,300,300,300]), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .filter("category='geography'")
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300],
    )
    .await;

    // Case 5: search with prefilter=false, phrase query, query_filter=vector([300,300,300,300])
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(false)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300, 299, 255, 254, 253],
    )
    .await;

    // Case 6: search with prefilter=false, phrase query, query_filter=vector([300,300,300,300]), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(false)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300, 255],
    )
    .await;

    // The scan's limit pages the results above. It must not cut the FTS
    // matches down before the vector query bounds them.
    let phrase_query = FullTextSearchQuery::new_query(FtsQuery::Phrase(
        PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
    ));
    // Case 7: Case 3 with limit 2
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(phrase_query.clone())
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .limit(Some(2), None)
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[299, 300],
    )
    .await;

    // Case 8: Case 5 with limit 3
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(phrase_query.clone())
        .unwrap()
        .prefilter(false)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .limit(Some(3), None)
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300, 299, 255],
    )
    .await;

    // Case 9: Case 3 with limit 0
    let batch = dataset
        .scan()
        .full_text_search(phrase_query)
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(vector_query))
        .unwrap()
        .limit(Some(0), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch.num_rows(), 0);
}

#[tokio::test]
async fn test_fts_filter_vector_search() {
    let dataset = prepare_query_filter_dataset().await;
    let schema: ArrowSchema = dataset.schema().into();

    // Assert filtering and exact distance order without depending on PQ's
    // approximate distances, which can tie for adjacent vectors.
    // Case 1: search with prefilter=true, query_filter=match("text")
    let query_vector = Float32Array::from(vec![300f32, 300f32, 300f32, 300f32]);
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .refine(1)
        .prefilter(true)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300, 299, 255, 254, 253],
    )
    .await;

    // Case 2: search with prefilter=true, query_filter=match("text"), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .refine(1)
        .prefilter(true)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300, 255, 252, 249, 246],
    )
    .await;

    // Case 3: search with prefilter=false, query_filter=match("text")
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .refine(1)
        .prefilter(false)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300, 299],
    )
    .await;

    // Case 4: search with prefilter=false, query_filter=match("text"), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .refine(1)
        .prefilter(false)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300],
    )
    .await;

    // Case 5: search with prefilter=false, query_filter=phrase("text")
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new_query(
            FtsQuery::Phrase(
                PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
            ),
        )))
        .unwrap()
        .try_into_stream()
        .await;
    assert!(stream.is_err());

    // Case 6: search with prefilter=false, query_filter=phrase("text")
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new_query(
            FtsQuery::Phrase(
                PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
            ),
        )))
        .unwrap()
        .try_into_stream()
        .await;
    assert!(stream.is_err());

    // Case 7: postfilter FTS filter with .limit(): the limit applies to the
    // rows the FTS filter keeps. The five nearest rows to [298; 4] are ids
    // 296..=300; 296..=298 hold "noop" text and are dropped by the filter. A
    // limit pushed below the filter would only let 298 and one of the tied
    // 297/299 reach it.
    let postfilter_query_vector = Float32Array::from(vec![298f32, 298f32, 298f32, 298f32]);
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &postfilter_query_vector, 5)
        .unwrap()
        .refine(1)
        .prefilter(false)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .limit(Some(2), None)
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[299, 300],
    )
    .await;

    // The scan's limit pages the results above. It must not cut the FTS
    // matches down before the vector search picks its neighbours from them.
    // Case 8: Case 1 with limit 3
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .refine(1)
        .prefilter(true)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .limit(Some(3), None)
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300, 299, 255],
    )
    .await;

    // Case 9: Case 1 with limit 0
    let batch = dataset
        .scan()
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .refine(1)
        .prefilter(true)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .limit(Some(0), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch.num_rows(), 0);
}

#[rstest]
#[case::list(false)]
#[case::large_list(true)]
#[tokio::test]
async fn test_fts_list_postfilter_vector_search(#[case] is_large_list: bool) {
    async fn indexed_ids(dataset: &Dataset, query: FullTextSearchQuery) -> Vec<i32> {
        let result = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let mut ids = result["id"]
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        ids.sort_unstable();
        ids
    }

    async fn postfilter_ids(dataset: &Dataset, query: FullTextSearchQuery) -> Vec<i32> {
        let query_vector = Float32Array::from(vec![0.0, 0.0]);
        let mut scanner = dataset.scan();
        scanner
            .nearest("vector", &query_vector, 5)
            .unwrap()
            .prefilter(false)
            .filter_query(QueryFilter::Fts(query))
            .unwrap();
        let plan = scanner.explain_plan(false).await.unwrap();
        let post_filter_position = plan
            .find("FlatMatchFilter: column=docs")
            .expect("expected FTS to run as a flat match filter");
        let vector_search_position = plan
            .find("ANNSubIndex")
            .expect("expected the query to use the vector index");
        assert!(
            post_filter_position < vector_search_position,
            "expected FTS to wrap the vector search as a post-filter, got:\n{plan}"
        );
        let original = scanner.create_plan().await.unwrap();
        let original_schema = original.schema();
        let mut rebuilt_filters = 0;
        let rebuilt = original
            .transform_up(|plan| {
                if let Some(filter) = plan.downcast_ref::<FlatMatchFilterExec>() {
                    rebuilt_filters += 1;
                    let filter = FlatMatchFilterExec::try_new(
                        filter.children()[0].clone(),
                        filter.dataset().clone(),
                        filter.query().clone(),
                        filter.params().clone(),
                        filter.document_binding(),
                        filter.preset_segments().map(|segments| segments.to_vec()),
                    )?;
                    return Ok(Transformed::yes(Arc::new(filter) as Arc<dyn ExecutionPlan>));
                }
                Ok(Transformed::no(plan))
            })
            .unwrap()
            .data;
        assert_eq!(rebuilt_filters, 1);
        assert_eq!(rebuilt.schema(), original_schema);
        let batches: Vec<RecordBatch> = execute_plan(rebuilt, Default::default())
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let result = concat_batches(&original_schema, &batches).unwrap();
        let mut ids = result["id"]
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        ids.sort_unstable();
        ids
    }

    fn match_query(terms: &str, operator: Operator) -> FullTextSearchQuery {
        FullTextSearchQuery::new_query(FtsQuery::Match(
            MatchQuery::new(terms.to_owned())
                .with_column(Some("docs".to_owned()))
                .with_operator(operator),
        ))
    }

    let item_field = Arc::new(ArrowField::new("item", DataType::Utf8, true));
    let values = Arc::new(StringArray::from(vec![
        Some("target"),
        Some("alpha"),
        Some("beta"),
        Some(""),
        None,
        Some("target"),
    ])) as ArrayRef;
    let validity = Some(NullBuffer::from(vec![true, true, true, true, false]));
    let docs: ArrayRef = if is_large_list {
        Arc::new(LargeListArray::new(
            item_field.clone(),
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i64, 2, 3, 3, 6, 6])),
            values,
            validity,
        ))
    } else {
        Arc::new(ListArray::new(
            item_field,
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 3, 3, 6, 6])),
            values,
            validity,
        ))
    };
    let vectors = FixedSizeListArray::try_new_from_values(
        Float32Array::from(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]),
        2,
    )
    .unwrap();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("vector", vectors.data_type().clone(), false),
        ArrowField::new("docs", docs.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..5)),
            Arc::new(vectors),
            docs,
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 3);

    dataset
        .create_index(
            &["docs"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            None,
            &VectorIndexParams::ivf_flat(1, MetricType::L2),
            true,
        )
        .await
        .unwrap();

    let query = match_query("target", Operator::Or);
    assert_eq!(indexed_ids(&dataset, query.clone()).await, [0, 3]);
    assert_eq!(postfilter_ids(&dataset, query).await, [0, 3]);

    let query = match_query("target alpha", Operator::And);
    assert_eq!(indexed_ids(&dataset, query.clone()).await, [0]);
    assert_eq!(postfilter_ids(&dataset, query).await, [0]);

    let query = match_query("target missing", Operator::And);
    assert!(indexed_ids(&dataset, query.clone()).await.is_empty());
    assert!(postfilter_ids(&dataset, query).await.is_empty());

    dataset
        .create_index(
            &["docs"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().base_tokenizer("raw".to_owned()),
            true,
        )
        .await
        .unwrap();
    let query = match_query("target", Operator::Or);
    assert_eq!(indexed_ids(&dataset, query.clone()).await, [3]);
    assert_eq!(postfilter_ids(&dataset, query).await, [3]);

    dataset
        .create_index(
            &["docs"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::code()
                .split_identifiers(true)
                .preserve_original(true),
            true,
        )
        .await
        .unwrap();
    let query = match_query("targetAlpha", Operator::And);
    assert_eq!(indexed_ids(&dataset, query.clone()).await, [0]);
    assert_eq!(postfilter_ids(&dataset, query).await, [0]);

    let fuzzy_query = FullTextSearchQuery::new_query(FtsQuery::Match(
        MatchQuery::new("targets".to_owned())
            .with_column(Some("docs".to_owned()))
            .with_fuzziness(Some(1)),
    ));
    let query_vector = Float32Array::from(vec![0.0, 0.0]);
    let mut scanner = dataset.scan();
    scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter_query(QueryFilter::Fts(fuzzy_query))
        .unwrap();
    let error = scanner.try_into_batch().await.unwrap_err();
    assert!(matches!(&error, Error::NotSupported { .. }));
    assert!(
        error
            .to_string()
            .contains("Fuzzy MatchQuery is not supported when FTS is used as a post-filter"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn test_scan_limit_offset_preserves_json_extension_metadata() {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        json_field("meta", true),
    ]));

    let json_array = JsonArray::try_from_iter((0..50).map(|i| Some(format!(r#"{{"i":{i}}}"#))))
        .unwrap()
        .into_inner();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..50)),
            Arc::new(json_array),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    let mut scanner = dataset.scan();
    scanner.limit(Some(10), None).unwrap();
    let batch_no_offset = scanner.try_into_batch().await.unwrap();
    assert!(is_arrow_json_field(
        batch_no_offset.schema().field_with_name("meta").unwrap()
    ));

    let mut scanner = dataset.scan();
    scanner.limit(Some(10), Some(10)).unwrap();
    let batch_with_offset = scanner.try_into_batch().await.unwrap();
    assert!(is_arrow_json_field(
        batch_with_offset.schema().field_with_name("meta").unwrap()
    ));
    assert_eq!(batch_no_offset.schema(), batch_with_offset.schema());
}

#[tokio::test]
async fn test_scan_nested_arrow_json_extension_v2() {
    let mut json_metadata = HashMap::new();
    json_metadata.insert(
        ARROW_EXT_NAME_KEY.to_string(),
        ARROW_JSON_EXT_NAME.to_string(),
    );
    let item_fields = Fields::from(vec![
        Arc::new(ArrowField::new("uri", DataType::Utf8, false)),
        Arc::new(ArrowField::new("extra", DataType::Utf8, true).with_metadata(json_metadata)),
    ]);
    let item = Arc::new(ArrowField::new(
        "item",
        DataType::Struct(item_fields.clone()),
        true,
    ));
    let media_field = ArrowField::new("media", DataType::List(item.clone()), true);
    let schema = Arc::new(ArrowSchema::new(vec![media_field]));

    for version in [
        LanceFileVersion::V2_1,
        LanceFileVersion::V2_2,
        LanceFileVersion::V2_3,
    ] {
        let values = StructArray::new(
            item_fields.clone(),
            vec![
                Arc::new(StringArray::from(vec![Some("a.jpg"), Some("b.jpg")])) as Arc<dyn Array>,
                Arc::new(StringArray::from(vec![
                    Some(r#"{"codec":"h264"}"#),
                    None::<&str>,
                ])) as Arc<dyn Array>,
            ],
            None,
        );
        let media = ListArray::new(
            item.clone(),
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 2])),
            Arc::new(values),
            None,
        );
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(media)]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone());
        let write_params = WriteParams {
            data_storage_version: Some(version),
            ..WriteParams::default()
        };
        let uri = format!("memory://{}", Uuid::new_v4());
        let dataset = Dataset::write(reader, &uri, Some(write_params))
            .await
            .unwrap();

        let batch = dataset.scan().try_into_batch().await.unwrap();
        let batch_schema = batch.schema();
        let DataType::List(item) = batch_schema.field(0).data_type() else {
            panic!("expected media list field");
        };
        let DataType::Struct(fields) = item.data_type() else {
            panic!("expected media item struct");
        };
        assert!(is_arrow_json_field(&fields[1]));

        let media: &ListArray = batch.column(0).as_list();
        let items = media.values().as_struct();
        let extra = items
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(extra.value(0).contains("h264"));
        assert!(extra.is_null(1));
    }
}

#[tokio::test]
async fn test_scan_miniblock_dictionary_out_of_line_bitpacking_does_not_panic() {
    let rows: usize = 10_000;
    let unique_values: usize = 2_000;
    let batch_size: usize = 8_192;

    let mut field_meta = HashMap::new();
    field_meta.insert(
        "lance-encoding:structural-encoding".to_string(),
        "miniblock".to_string(),
    );
    field_meta.insert(
        "lance-encoding:dict-size-ratio".to_string(),
        "0.99".to_string(),
    );

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("d", DataType::UInt64, false).with_metadata(field_meta),
    ]));

    let values = (0..rows)
        .map(|i| (i % unique_values) as u64)
        .collect::<Vec<_>>();
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(UInt64Array::from(values))]).unwrap();

    let uri = format!("memory://{}", Uuid::new_v4());
    let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone());

    let write_params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_2),
        ..WriteParams::default()
    };
    let dataset = Dataset::write(reader, &uri, Some(write_params))
        .await
        .unwrap();

    let field_id = dataset.schema().field("d").unwrap().id as u32;
    let fragment = dataset.get_fragment(0).unwrap();
    let data_file = fragment.data_file_for_field(field_id).unwrap();
    let field_pos = data_file
        .fields
        .iter()
        .position(|id| *id == field_id as i32)
        .unwrap();
    let column_idx = data_file.column_indices[field_pos] as usize;

    let file_path = dataset.data_dir().join(data_file.path.as_str());
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );
    let file_scheduler = scheduler
        .open_file(&file_path, &CachedFileSize::unknown())
        .await
        .unwrap();

    let cache = LanceCache::with_capacity(8 * 1024 * 1024);
    let file_reader = FileReader::try_open(
        file_scheduler,
        None,
        Arc::<DecoderPlugins>::default(),
        &cache,
        FileReaderOptions::default(),
    )
    .await
    .unwrap();

    let col_meta = &file_reader.metadata().column_metadatas[column_idx];
    let encoding = describe_encoding(col_meta.pages.first().unwrap());
    assert!(
        encoding.contains("OutOfLineBitpacking") && encoding.contains("dictionary"),
        "Expected a mini-block dictionary page with out-of-line bitpacking, got: {encoding}"
    );

    let mut scanner = dataset.scan();
    scanner.batch_size(batch_size);
    scanner.project(&["d"]).unwrap();

    let mut stream = scanner.try_into_stream().await.unwrap();
    let batch = stream.try_next().await.unwrap().unwrap();
    assert_eq!(batch.num_columns(), 1);
}

async fn prepare_query_filter_dataset() -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                4,
            ),
            true,
        ),
        ArrowField::new("text", DataType::Utf8, false),
        ArrowField::new("category", DataType::Utf8, false),
    ]));

    // Prepare dataset
    let mut vectors = vec![];
    for i in 1..=300 {
        vectors.extend(vec![i as f32; 4]);
    }

    // id 256..298 has noop, others has text
    let mut text = vec![];
    for i in 1..=255 {
        text.push(format!("text {}", i));
    }
    for i in 256..=298 {
        text.push(format!("noop {}", i));
    }
    text.extend(vec!["text 299".to_string(), "text 300".to_string()]);

    let mut category = vec![];
    for i in 1..=300 {
        if i % 3 == 1 {
            category.push("literature".to_string());
        } else if i % 3 == 2 {
            category.push("science".to_string());
        } else {
            category.push("geography".to_string());
        }
    }

    let vectors = Float32Array::from(vectors);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(1..=300)),
            Arc::new(FixedSizeListArray::try_new_from_values(vectors, 4).unwrap()),
            Arc::new(StringArray::from(text)),
            Arc::new(StringArray::from(category)),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    // Create index
    let params = VectorIndexParams::with_ivf_pq_params(
        MetricType::L2,
        IvfBuildParams::new(2),
        PQBuildParams::new(4, 8),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().with_position(true),
            true,
        )
        .await
        .unwrap();

    dataset
}

async fn check_results(
    stream: DatasetRecordBatchStream,
    expected_schema: SchemaRef,
    expected_ids: &[i32],
) {
    let results = stream.try_collect::<Vec<_>>().await.unwrap();
    let batch = concat_batches(&results[0].schema(), &results).unwrap();
    assert_eq!(batch.schema(), expected_schema);

    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.values(), expected_ids);
}

/// Filter strings compare a decimal column with fractional literals exactly, as
/// TPC-H Q6 does, and a scalar index on the column answers them. Rows appended
/// after the index was built are answered by the scan.
#[tokio::test]
async fn test_decimal_column_filters_with_fractional_literals() {
    // Hundredths, NULL on every 7th row
    let discount = |id: i32| (id % 7 != 0).then_some(i128::from(id % 10));
    let batch = |ids: std::ops::Range<i32>| {
        let discounts = Decimal128Array::from_iter(ids.clone().map(discount))
            .with_precision_and_scale(15, 2)
            .unwrap();
        RecordBatch::try_from_iter(vec![
            (
                "id",
                Arc::new(Int32Array::from_iter_values(ids.clone())) as ArrayRef,
            ),
            ("discount", Arc::new(discounts) as ArrayRef),
            (
                "quantity",
                Arc::new(Int32Array::from_iter_values(ids)) as ArrayRef,
            ),
        ])
        .unwrap()
    };
    let reader = |ids| {
        let batch = batch(ids);
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema())
    };
    let params = WriteParams {
        max_rows_per_file: 20,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader(0..40), "memory://", Some(params))
        .await
        .unwrap();
    dataset
        .create_index(
            &["discount"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    dataset.append(reader(40..60), None).await.unwrap();

    // Whether a row with this discount (in hundredths) and quantity matches
    type Matches = fn(Option<i128>, i32) -> bool;
    let cases: [(&str, Matches); 6] = [
        (
            "discount BETWEEN 0.05 AND 0.07 AND quantity < 24",
            |d, q| d.is_some_and(|d| (5..=7).contains(&d)) && q < 24,
        ),
        ("discount < 0.055", |d, _| d.is_some_and(|d| d <= 5)),
        ("discount > 0.055", |d, _| d.is_some_and(|d| d >= 6)),
        ("discount = 0.055", |_, _| false),
        ("discount != 0.055", |d, _| d.is_some()),
        ("discount IN (0.05, 0.070)", |d, _| {
            d == Some(5) || d == Some(7)
        }),
    ];
    for (filter, predicate) in cases {
        let expected = (0..60)
            .filter(|&id| predicate(discount(id), id))
            .collect::<Vec<_>>();
        let plan = dataset
            .scan()
            .filter(filter)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(
            plan.contains("ScalarIndexQuery"),
            "`{filter}` should use the scalar index, got plan:\n{plan}"
        );
        for use_scalar_index in [true, false] {
            let mut scanner = dataset.scan();
            scanner
                .project(&["id"])
                .unwrap()
                .filter(filter)
                .unwrap()
                .use_scalar_index(use_scalar_index);
            let batch = scanner.try_into_batch().await.unwrap();
            let mut ids = batch["id"].as_primitive::<Int32Type>().values().to_vec();
            ids.sort_unstable();
            assert_eq!(ids, expected, "`{filter}`, index: {use_scalar_index}");
        }
    }

    // Write predicates take the same literals
    let deleted = (0..60)
        .filter(|&id| discount(id).is_some_and(|d| d <= 5))
        .count();
    dataset.delete("discount < 0.055").await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 60 - deleted);
}

/// label, confidence, area
type Detection = (&'static str, f32, f32);
type Detections = Option<Vec<Option<Detection>>>;
type DetectionCheck = fn(i32, &Detections) -> bool;

const DETECTION_ROWS: i32 = 2000;

/// Row `id` of the `any_element` fixture: (width, detections). Rows 0 to 5 are
/// the edge cases; the rest are generated, with null lists and null detections.
fn detection_row(id: i32) -> (i32, Detections) {
    let det = |label: &'static str, confidence: f32, area: f32| Some((label, confidence, area));
    match id {
        // A low-confidence car next to a high-confidence person.
        0 => (
            3840,
            Some(vec![det("car", 0.30, 20000.0), det("person", 0.95, 5000.0)]),
        ),
        // Confidence exactly at the boundary.
        1 => (3840, Some(vec![det("car", 0.90, 20000.0)])),
        // The area condition holds on another detection.
        2 => (
            3840,
            Some(vec![det("car", 0.95, 9999.0), det("person", 0.1, 20000.0)]),
        ),
        // `carrot` is not `car`.
        3 => (3840, Some(vec![det("carrot", 0.99, 20000.0)])),
        4 => (3840, Some(vec![])),
        5 => (3840, None),
        _ => {
            let i = id as usize;
            let width = [1280, 1920, 2560, 3840][i % 4];
            if i % 97 == 13 {
                return (width, None);
            }
            let detections = (0..(i * 7) % 6)
                .map(|j| {
                    if (i + 3 * j).is_multiple_of(23) {
                        return None;
                    }
                    let label = ["car", "person", "truck", "carrot"][(i * 5 + j * 3) % 4];
                    let confidence = ((i * 37 + j * 11) % 101) as f32 / 100.0;
                    let area = ((i * 13 + j * 29) % 41) as f32 * 500.0;
                    det(label, confidence, area)
                })
                .collect();
            (width, Some(detections))
        }
    }
}

/// SQL three-valued `any_element` over a fixture row: a null detection makes
/// every comparison NULL.
fn any_detection(detections: &Detections, predicate: impl Fn(&Detection) -> bool) -> Option<bool> {
    let detections = detections.as_ref()?;
    let mut is_unknown = false;
    for detection in detections {
        match detection {
            Some(detection) if predicate(detection) => return Some(true),
            Some(_) => {}
            None => is_unknown = true,
        }
    }
    if is_unknown { None } else { Some(false) }
}

fn detection_batch(ids: std::ops::Range<i32>) -> RecordBatch {
    let fields = Fields::from(vec![
        ArrowField::new("label", DataType::Utf8, true),
        ArrowField::new("confidence", DataType::Float32, true),
        ArrowField::new("area", DataType::Float32, true),
    ]);
    let item = Arc::new(ArrowField::new(
        "item",
        DataType::Struct(fields.clone()),
        true,
    ));
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("width", DataType::Int32, false),
        ArrowField::new("detections", DataType::List(item.clone()), true),
    ]));
    let (mut labels, mut confidences, mut areas) = (vec![], vec![], vec![]);
    let (mut is_valid, mut lengths, mut is_list_valid, mut widths) =
        (vec![], vec![], vec![], vec![]);
    for id in ids.clone() {
        let (width, detections) = detection_row(id);
        widths.push(width);
        is_list_valid.push(detections.is_some());
        let detections = detections.unwrap_or_default();
        lengths.push(detections.len());
        for detection in detections {
            is_valid.push(detection.is_some());
            // A null detection hides values that would match every filter.
            let (label, confidence, area) = detection.unwrap_or(("car", 0.99, 99999.0));
            labels.push(label);
            confidences.push(confidence);
            areas.push(area);
        }
    }
    let elements = StructArray::try_new(
        fields,
        vec![
            Arc::new(StringArray::from(labels)) as ArrayRef,
            Arc::new(Float32Array::from(confidences)),
            Arc::new(Float32Array::from(areas)),
        ],
        Some(NullBuffer::from(is_valid)),
    )
    .unwrap();
    let detections = ListArray::try_new(
        item,
        OffsetBuffer::from_lengths(lengths),
        Arc::new(elements),
        Some(NullBuffer::from(is_list_valid)),
    )
    .unwrap();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from_iter_values(ids)),
            Arc::new(Int32Array::from(widths)),
            Arc::new(detections),
        ],
    )
    .unwrap()
}

/// Three fragments with a BTree on `width`, then a fourth the index does not
/// cover.
async fn detection_dataset() -> Dataset {
    let params = WriteParams {
        max_rows_per_file: 500,
        ..Default::default()
    };
    let indexed = detection_batch(0..1500);
    let schema = indexed.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(indexed)], schema.clone()),
        "memory://",
        Some(params.clone()),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["width"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(detection_batch(1500..DETECTION_ROWS))], schema),
            Some(params),
        )
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 4);
    dataset
}

async fn any_element_ids(dataset: &Dataset, filter: &str, use_scalar_index: bool) -> Vec<i32> {
    let mut scanner = dataset.scan();
    scanner
        .project(&["id"])
        .unwrap()
        .filter(filter)
        .unwrap()
        .use_scalar_index(use_scalar_index);
    let batch = scanner.try_into_batch().await.unwrap();
    let mut ids = batch["id"].as_primitive::<Int32Type>().values().to_vec();
    ids.sort_unstable();
    ids
}

#[rstest]
#[case::same_element(
    "any_element(detections, 'label = ''car'' AND confidence > 0.9')",
    |_, d: &Detections| any_detection(d, |d| d.0 == "car" && d.1 > 0.9) == Some(true)
)]
#[case::inclusive_range(
    "any_element(detections, 'label = ''car'' AND confidence >= 0.9')",
    |_, d: &Detections| any_detection(d, |d| d.0 == "car" && d.1 >= 0.9) == Some(true)
)]
#[case::in_list(
    "any_element(detections, 'label IN (''truck'', ''carrot'') AND confidence > 0.95')",
    |_, d: &Detections| any_detection(d, |d| (d.0 == "truck" || d.0 == "carrot") && d.1 > 0.95) == Some(true)
)]
#[case::three_conditions(
    "any_element(detections, 'label = ''car'' AND confidence > 0.9 AND area > 10000')",
    |_, d: &Detections| any_detection(d, |d| d.0 == "car" && d.1 > 0.9 && d.2 > 10000.0) == Some(true)
)]
#[case::negated(
    "NOT any_element(detections, 'label = ''car'' AND confidence > 0.9')",
    |_, d: &Detections| any_detection(d, |d| d.0 == "car" && d.1 > 0.9) == Some(false)
)]
#[case::and_indexed_column(
    "any_element(detections, 'label = ''car'' AND confidence > 0.9') AND width > 1920",
    |w, d: &Detections| any_detection(d, |d| d.0 == "car" && d.1 > 0.9) == Some(true) && w > 1920
)]
#[case::or_indexed_column(
    "any_element(detections, 'label = ''car'' AND confidence > 0.9') OR width = 1280",
    |w, d: &Detections| any_detection(d, |d| d.0 == "car" && d.1 > 0.9) == Some(true) || w == 1280
)]
#[case::lambda(
    "array_any_match(detections, d -> d.label = 'car' AND d.confidence > 0.9)",
    |_, d: &Detections| any_detection(d, |d| d.0 == "car" && d.1 > 0.9) == Some(true)
)]
#[case::lambda_alias_and_subscript(
    "any_match(detections, (x) -> x['label'] IN ('truck', 'carrot') AND x.confidence > 0.95)",
    |_, d: &Detections| any_detection(d, |d| (d.0 == "truck" || d.0 == "carrot") && d.1 > 0.95) == Some(true)
)]
#[case::lambda_negated_and_indexed_column(
    "NOT array_any_match(detections, d -> d.label = 'car' AND d.confidence > 0.9) AND width > 1920",
    |w, d: &Detections| any_detection(d, |d| d.0 == "car" && d.1 > 0.9) == Some(false) && w > 1920
)]
#[tokio::test]
async fn test_any_element_filter(#[case] filter: &str, #[case] keep: DetectionCheck) {
    let dataset = detection_dataset().await;
    let expected: Vec<i32> = (0..DETECTION_ROWS)
        .filter(|id| {
            let (width, detections) = detection_row(*id);
            keep(width, &detections)
        })
        .collect();
    assert!(!expected.is_empty(), "`{filter}` should match some rows");
    for use_scalar_index in [true, false] {
        assert_eq!(
            any_element_ids(&dataset, filter, use_scalar_index).await,
            expected,
            "`{filter}`, index: {use_scalar_index}"
        );
    }
    assert_eq!(
        dataset.count_rows(Some(filter.to_string())).await.unwrap(),
        expected.len()
    );
}

#[tokio::test]
async fn test_any_element_edge_rows() {
    let dataset = detection_dataset().await;
    let filter = "any_element(detections, 'label = ''car'' AND confidence > 0.9')";
    let matched = any_element_ids(&dataset, filter, true).await;
    // Only row 2 has a car above 0.9 among the edge cases.
    for id in [0, 1, 3, 4, 5] {
        assert!(!matched.contains(&id), "row {id} must not match `{filter}`");
    }
    assert!(matched.contains(&2));
    // The empty list is FALSE, so NOT selects it; the null list is NULL, so
    // neither does.
    let negated = any_element_ids(&dataset, &format!("NOT {filter}"), true).await;
    assert!(negated.contains(&4) && !negated.contains(&5));
}

/// A predicate that names a column outside the element, or calls a function
/// that is not immutable (compiled predicates are reused across queries, so
/// `now()` would freeze), is rejected before any data is read.
#[rstest]
#[case::outside_column("any_element(detections, 'width > 1')", "width")]
#[case::stable_function("any_element(detections, 'now() IS NOT NULL')", "calls now()")]
#[tokio::test]
async fn test_any_element_rejects_invalid_predicates(#[case] filter: &str, #[case] reason: &str) {
    let dataset = detection_dataset().await;
    let mut scanner = dataset.scan();
    scanner.filter(filter).unwrap();
    let err = scanner.try_into_batch().await.unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
    let message = err.to_string();
    assert!(
        message.contains("any_element") && message.contains(reason),
        "{message}"
    );
    let err = dataset
        .count_rows(Some(filter.to_string()))
        .await
        .unwrap_err();
    assert!(err.to_string().contains(reason), "{err}");
}

/// A lambda filter that `any_element` cannot express is rejected when the
/// filter is parsed, with a message that says why.
#[rstest]
#[case::outside_column(
    "array_any_match(detections, d -> d.label = 'car' AND width > 1)",
    "`width` is outside the element"
)]
#[case::other_lambda_function(
    "cardinality(array_filter(detections, d -> d.confidence > 0.5)) > 0",
    "only accept lambdas as array_any_match"
)]
#[tokio::test]
async fn test_lambda_filter_rejects_unsupported_forms(#[case] filter: &str, #[case] reason: &str) {
    let dataset = detection_dataset().await;
    let mut scanner = dataset.scan();
    scanner.filter(filter).unwrap();
    let err = scanner.try_into_batch().await.unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
    assert!(err.to_string().contains(reason), "{err}");
    let err = dataset
        .count_rows(Some(filter.to_string()))
        .await
        .unwrap_err();
    assert!(err.to_string().contains(reason), "{err}");
}
