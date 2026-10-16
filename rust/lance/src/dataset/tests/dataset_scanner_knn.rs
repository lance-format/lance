// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::array::as_primitive_array;
use arrow::datatypes::Int32Type;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Int32Array, RecordBatch,
    RecordBatchIterator, StructArray, UInt8Array, UInt32Array,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::concat::concat_batches;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use futures::TryStreamExt;
use lance_arrow::{FixedSizeListArrayExt, SchemaExt};
use lance_core::utils::tempfile::TempStrDir;
use lance_core::{ROW_ADDR, ROW_ID};
use lance_datagen::{ArrayGeneratorExt, Dimension, array, gen_batch};
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::vector::DIST_COL;
use lance_linalg::distance::{DistanceType, MetricType};
use rstest::rstest;

use crate::Error;
use crate::dataset::scanner::test_dataset::{
    TestVectorDataset, assert_query_index_field, batch_knn_two_queries, batch_row_ids,
    make_binary_vector_dataset,
};
use crate::dataset::scanner::{RowAddrMask, RowAddrTreeMap};
use crate::dataset::{Dataset, WriteParams};
use crate::index::DatasetIndexExt;
use crate::index::vector::VectorIndexParams;
use crate::io::exec::PreFilterSource;
// Imported through the public `io::exec` re-export rather than the crate-private
// `knn` module, so the tests below cover that surface too.
use crate::io::exec::{ANNIvfBatchExec, QUERY_INDEX_COL};
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

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

async fn nested_vector_test_dataset(dim: u32) -> (TempStrDir, Dataset) {
    let path = TempStrDir::default();
    let vec_field = ArrowField::new(
        "vec",
        DataType::FixedSizeList(
            Arc::new(ArrowField::new("item", DataType::Float32, true)),
            dim as i32,
        ),
        true,
    );
    let payload_field = ArrowField::new(
        "payload",
        DataType::Struct(vec![vec_field.clone()].into()),
        true,
    );
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("i", DataType::Int32, true),
        payload_field.clone(),
    ]));

    let batches: Vec<RecordBatch> = (0..5)
        .map(|batch_idx| {
            let vector_values: Float32Array = (0..dim * 80).map(|v| v as f32).collect();
            let vectors =
                FixedSizeListArray::try_new_from_values(vector_values, dim as i32).unwrap();
            let payload = StructArray::from(vec![(
                Arc::new(vec_field.clone()),
                Arc::new(vectors) as ArrayRef,
            )]);
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(
                        batch_idx * 80..(batch_idx + 1) * 80,
                    )),
                    Arc::new(payload),
                ],
            )
            .unwrap()
        })
        .collect();

    let params = WriteParams {
        max_rows_per_group: 10,
        max_rows_per_file: 200,
        data_storage_version: Some(LanceFileVersion::Stable),
        enable_stable_row_ids: true,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    let dataset = Dataset::write(reader, &path, Some(params)).await.unwrap();
    (path, dataset)
}

async fn escaped_nested_vector_test_dataset(dim: u32) -> (TempStrDir, Dataset) {
    let path = TempStrDir::default();
    let vec_field = ArrowField::new(
        "vec.with.dot",
        DataType::FixedSizeList(
            Arc::new(ArrowField::new("item", DataType::Float32, true)),
            dim as i32,
        ),
        true,
    );
    let payload_field = ArrowField::new(
        "payload",
        DataType::Struct(vec![vec_field.clone()].into()),
        true,
    );
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("i", DataType::Int32, true),
        payload_field.clone(),
    ]));

    let batches: Vec<RecordBatch> = (0..5)
        .map(|batch_idx| {
            let vector_values: Float32Array = (0..dim * 80).map(|v| v as f32).collect();
            let vectors =
                FixedSizeListArray::try_new_from_values(vector_values, dim as i32).unwrap();
            let payload = StructArray::from(vec![(
                Arc::new(vec_field.clone()),
                Arc::new(vectors) as ArrayRef,
            )]);
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(
                        batch_idx * 80..(batch_idx + 1) * 80,
                    )),
                    Arc::new(payload),
                ],
            )
            .unwrap()
        })
        .collect();

    let params = WriteParams {
        max_rows_per_group: 10,
        max_rows_per_file: 200,
        data_storage_version: Some(LanceFileVersion::Stable),
        enable_stable_row_ids: true,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    let dataset = Dataset::write(reader, &path, Some(params)).await.unwrap();
    (path, dataset)
}

fn assert_batch_knn_output_has_no_vector(batch: &RecordBatch, vector_column: &str) {
    assert!(
        batch.schema().column_with_name(vector_column).is_none(),
        "batch flat KNN output must not include vector column '{vector_column}' when it is not projected; columns: {:?}",
        batch.schema().field_names()
    );
}

async fn assert_batch_matches_single_queries(
    dataset: &Dataset,
    batch: &RecordBatch,
    query_values: &[f32],
    k: usize,
    use_index: bool,
    distance_range: Option<(Option<f32>, Option<f32>)>,
    fixed_nprobes: Option<usize>,
) {
    let query_count = query_values.len() / 32;
    assert_eq!(batch.num_rows(), query_count * k);

    for query_index in 0..query_count {
        let query =
            Float32Array::from(query_values[query_index * 32..(query_index + 1) * 32].to_vec());
        let mut scan = dataset.scan();
        scan.nearest("vec", &query, k).unwrap();
        scan.use_index(use_index);
        // Pin nprobes to match the batch query: the single-query indexed path
        // otherwise adaptively expands nprobes, which would make equivalence
        // depend on data distribution rather than be guaranteed.
        if let Some(nprobes) = fixed_nprobes {
            scan.minimum_nprobes(nprobes).maximum_nprobes(nprobes);
        }
        if let Some((lower, upper)) = distance_range {
            scan.distance_range(lower, upper);
        }
        scan.project(&["i"]).unwrap();
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
    scan.project(&["i"]).unwrap();

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
    assert_batch_knn_output_has_no_vector(&batch, "vec");
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
    assert_batch_matches_single_queries(dataset, &batch, &query_values, k, false, None, None).await;

    let mut scan_with_vec = dataset.scan();
    scan_with_vec.nearest("vec", &queries, k).unwrap();
    scan_with_vec.use_index(false);
    scan_with_vec.project(&["i", "vec"]).unwrap();
    let batch_with_vec = scan_with_vec.try_into_batch().await.unwrap();
    assert!(
        batch_with_vec.schema().column_with_name("vec").is_some(),
        "batch flat KNN should return vector column when projected"
    );
    assert_batch_matches_single_queries(
        dataset,
        &batch_with_vec,
        &query_values,
        k,
        false,
        None,
        None,
    )
    .await;

    let query_values_one = (32..64).map(|v| v as f32).collect::<Vec<_>>();
    let queries_one =
        FixedSizeListArray::try_new_from_values(Float32Array::from(query_values_one.clone()), 32)
            .unwrap();
    let mut scan = dataset.scan();
    scan.nearest("vec", &queries_one, k).unwrap();
    scan.use_index(false);
    scan.project(&["i"]).unwrap();

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
    assert_batch_knn_output_has_no_vector(&batch, "vec");
    assert_eq!(
        batch[QUERY_INDEX_COL].as_primitive::<Int32Type>().values(),
        &[0, 0]
    );
}

#[tokio::test]
async fn test_batch_knn_flat_omits_vector_without_projection() {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    let dataset = &test_ds.dataset;
    let k = 2;
    let (queries, query_values) = batch_knn_two_queries();

    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, k).unwrap();
    scan.use_index(false);
    scan.project(&["i"]).unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    assert_batch_knn_output_has_no_vector(&batch, "vec");
    assert_query_index_field(&batch);
    assert!(batch.schema().column_with_name("i").is_some());
    assert!(batch.schema().column_with_name(DIST_COL).is_some());
    assert_batch_matches_single_queries(dataset, &batch, &query_values, k, false, None, None).await;

    let mut scan_rowid_only = dataset.scan();
    scan_rowid_only.nearest("vec", &queries, k).unwrap();
    scan_rowid_only.use_index(false);
    scan_rowid_only.project(&[ROW_ID]).unwrap();
    let batch_rowid_only = scan_rowid_only.try_into_batch().await.unwrap();
    assert_batch_knn_output_has_no_vector(&batch_rowid_only, "vec");
    assert!(batch_rowid_only.schema().column_with_name(ROW_ID).is_some());
    assert!(batch_rowid_only.schema().column_with_name("i").is_none());

    let mut scan_with_vec = dataset.scan();
    scan_with_vec.nearest("vec", &queries, k).unwrap();
    scan_with_vec.use_index(false);
    scan_with_vec.project(&["vec"]).unwrap();
    let batch_with_vec = scan_with_vec.try_into_batch().await.unwrap();
    assert!(
        batch_with_vec.schema().column_with_name("vec").is_some(),
        "batch flat KNN must include vector column when vec is projected"
    );
}

#[tokio::test]
async fn test_batch_knn_flat_filter_keeps_non_vector_columns() {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    let dataset = &test_ds.dataset;
    let k = 2;
    let (queries, query_values) = batch_knn_two_queries();

    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, k).unwrap();
    scan.use_index(false);
    scan.filter("i >= 0").unwrap();
    scan.project(&["i"]).unwrap();
    let batch = scan.try_into_batch().await.unwrap();

    assert_query_index_field(&batch);
    assert_batch_knn_output_has_no_vector(&batch, "vec");
    assert!(batch.schema().column_with_name("i").is_some());

    let query_indices = batch[QUERY_INDEX_COL].as_primitive::<Int32Type>();
    for query_index in 0..2 {
        let query =
            Float32Array::from(query_values[query_index * 32..(query_index + 1) * 32].to_vec());
        let mut single = dataset.scan();
        single.nearest("vec", &query, k).unwrap();
        single.use_index(false);
        single.filter("i >= 0").unwrap();
        single.project(&["i"]).unwrap();
        let single_batch = single.try_into_batch().await.unwrap();

        let mask = BooleanArray::from_iter(
            query_indices
                .iter()
                .map(|value| value.map(|value| value == query_index as i32)),
        );
        let batch_slice = arrow::compute::filter_record_batch(&batch, &mask).unwrap();
        assert_eq!(
            batch_slice["i"].as_primitive::<Int32Type>().values(),
            single_batch["i"].as_primitive::<Int32Type>().values()
        );
    }
}

#[tokio::test]
async fn test_batch_knn_flat_nested_vector_projection() {
    const VECTOR_COLUMN: &str = "payload.vec";
    let (_tmp, dataset) = nested_vector_test_dataset(32).await;
    let k = 2;
    let (queries, _query_values) = batch_knn_two_queries();

    let mut scan = dataset.scan();
    scan.nearest(VECTOR_COLUMN, &queries, k).unwrap();
    scan.use_index(false);
    scan.project(&["i"]).unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    assert_batch_knn_output_has_no_vector(&batch, VECTOR_COLUMN);
    assert_eq!(batch.num_rows(), 2 * k);
    assert!(batch.schema().column_with_name("i").is_some());

    let mut scan_with_vec = dataset.scan();
    scan_with_vec.nearest(VECTOR_COLUMN, &queries, k).unwrap();
    scan_with_vec.use_index(false);
    scan_with_vec.project(&[VECTOR_COLUMN]).unwrap();
    let batch_with_vec = scan_with_vec.try_into_batch().await.unwrap();
    assert!(
        batch_with_vec
            .schema()
            .column_with_name(VECTOR_COLUMN)
            .is_some(),
        "batch flat KNN must include nested vector column when projected; columns: {:?}",
        batch_with_vec.schema().field_names()
    );
}

#[tokio::test]
async fn test_batch_knn_flat_escaped_nested_vector_projection() {
    const VECTOR_COLUMN: &str = "payload.`vec.with.dot`";
    let (_tmp, dataset) = escaped_nested_vector_test_dataset(32).await;
    let k = 2;
    let (queries, _) = batch_knn_two_queries();

    let mut scan = dataset.scan();
    scan.nearest(VECTOR_COLUMN, &queries, k).unwrap();
    scan.use_index(false);
    scan.project(&["i"]).unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    assert_batch_knn_output_has_no_vector(&batch, VECTOR_COLUMN);
    assert_eq!(batch.num_rows(), 2 * k);
    assert!(batch.schema().column_with_name("i").is_some());

    let mut scan_with_vec = dataset.scan();
    scan_with_vec.nearest(VECTOR_COLUMN, &queries, k).unwrap();
    scan_with_vec.use_index(false);
    scan_with_vec.project(&[VECTOR_COLUMN]).unwrap();
    let batch_with_vec = scan_with_vec.try_into_batch().await.unwrap();
    assert!(
        batch_with_vec
            .schema()
            .column_with_name(VECTOR_COLUMN)
            .is_some(),
        "batch flat KNN must include escaped nested vector column when projected; columns: {:?}",
        batch_with_vec.schema().field_names()
    );
}

#[tokio::test]
async fn test_batch_knn_flat_projects_row_id_and_row_addr_without_vector() {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    let dataset = &test_ds.dataset;
    let k = 2;
    let (queries, _) = batch_knn_two_queries();

    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, k).unwrap();
    scan.use_index(false);
    scan.project(&[ROW_ID]).unwrap();
    scan.with_row_address();

    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    assert_batch_knn_output_has_no_vector(&batch, "vec");
    assert_eq!(batch.num_rows(), 2 * k);
    assert!(batch.schema().column_with_name(ROW_ID).is_some());
    assert!(batch.schema().column_with_name(ROW_ADDR).is_some());
    assert!(batch.schema().column_with_name(DIST_COL).is_some());
    assert_eq!(
        batch[ROW_ADDR].as_primitive::<UInt64Type>().null_count(),
        0,
        "row addresses should be materialized for all top-k rows"
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
        .project(&["i"])
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
        None,
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
    // Probe both partitions (minimum == maximum) so the per-query top-k is
    // merged across multiple partitions and the batch result is
    // deterministically equivalent to repeated single-query search (which
    // would otherwise adaptively expand nprobes).
    scan.minimum_nprobes(2).maximum_nprobes(2);
    scan.project(&["i"]).unwrap();

    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ANNIvfBatch"),
        "IVF batch KNN should use the shared-scan batch node, got:\n{}",
        plan
    );
    assert!(
        !plan.contains("ANNSubIndex"),
        "IVF batch KNN should not fall back to per-query ANN search, got:\n{}",
        plan
    );
    assert!(
        !plan.contains("KNNVectorDistance: queries=2"),
        "indexed batch KNN should not force the flat batch path, got:\n{}",
        plan
    );

    // The batch node loads each probed partition once and scores every query
    // that probes it, so it must report the *distinct* partitions read: with
    // 2 partitions and fixed bounds of 2, both queries probe both partitions, so the
    // union is 2 -- not the per-query sum (2 queries x 2 = 4), and never 0
    // (which is what a dropped metric would show). This guards the observed
    // `partitions_searched` against silently regressing to either.
    let analyzed = scan.analyze_plan().await.unwrap();
    let batch_line = analyzed
        .lines()
        .find(|line| line.contains("ANNIvfBatch"))
        .expect("analyzed plan should contain the ANNIvfBatch node");
    assert!(
        batch_line.contains("partitions_searched=2"),
        "batch node must report the distinct partitions searched, got:\n{}",
        batch_line
    );

    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    assert_eq!(
        batch[QUERY_INDEX_COL].as_primitive::<Int32Type>().values(),
        &[0, 0, 1, 1]
    );
    // Shared-scan batch search must return the same rows/distances as
    // issuing the queries one at a time against the index.
    assert_batch_matches_single_queries(dataset, &batch, &query_values, 2, true, None, Some(2))
        .await;

    let batch = dataset
        .scan()
        .nearest("vec", &queries, 2)
        .unwrap()
        .minimum_nprobes(2)
        .maximum_nprobes(2)
        .distance_range(Some(1.0), None)
        .project(&["i"])
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
        Some(2),
    )
    .await;
}

/// End-to-end contract: equal-distance neighbors come back in a canonical,
/// deterministic order — ascending row id within a distance tie — and the
/// shared-scan batch path returns exactly what repeated single-query search
/// does. A single-partition exact (IVF_FLAT) index queried with a vector that
/// matches a row duplicated once per fragment yields five neighbors tied at
/// distance 0; `k = 5` returns all of them, so their order is fixed solely by
/// the tie-break, which orders ties by ascending row id (here ascending `i`,
/// since the data is written in `i` order).
///
/// This pins the user-visible ordering guarantee; it does not isolate a
/// single internal sort. The final `(distance, row_id)` order is enforced by
/// the downstream consumer, and the stable partition scan order in
/// `search_partitions_batch` only changes *which* tied row survives when a tie
/// is truncated across partitions — which this single-partition,
/// all-ties-fit-within-`k` case deliberately does not exercise.
#[rstest]
#[tokio::test]
async fn test_batch_knn_indexed_orders_ties_by_row_id(#[values(false, true)] stable_row_ids: bool) {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, stable_row_ids)
        .await
        .unwrap();
    // Single partition + exact (flat) storage: distances are exact, so the
    // vectors duplicated across fragments tie at distance 0, and both the
    // batch and single-query paths scan the one partition. That isolates the
    // tie-break as the only thing determining the emitted order.
    let params = VectorIndexParams::ivf_flat(1, MetricType::L2);
    test_ds
        .dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("idx".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();
    let dataset = &test_ds.dataset;

    let (queries, query_values) = batch_knn_two_queries();
    // Each query exactly matches a vector that appears once per 80-row
    // fragment (5 copies), all at distance 0. `k = 5` returns every tied
    // copy, so no truncation can hide the ordering.
    let k = 5;
    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, k).unwrap();
    scan.minimum_nprobes(1).maximum_nprobes(1);
    scan.project(&["i"]).unwrap();

    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ANNIvfBatch"),
        "single-partition IVF batch KNN should use the shared-scan batch node, got:\n{}",
        plan
    );

    let batch = scan.try_into_batch().await.unwrap();
    // The batch node must return the same rows/distances as issuing each
    // query on its own against the index.
    assert_batch_matches_single_queries(dataset, &batch, &query_values, k, true, None, Some(1))
        .await;

    // Query 0 matches vector index 1, stored at i = 1, 81, 161, 241, 321
    // (once per fragment). All tie at distance 0, so the canonical
    // (distance, row_id) order surfaces them by ascending row id, which here
    // is ascending `i`.
    let query_indices = batch[QUERY_INDEX_COL].as_primitive::<Int32Type>();
    let q0 = arrow::compute::filter_record_batch(
        &batch,
        &BooleanArray::from_iter(query_indices.iter().map(|value| Some(value == Some(0)))),
    )
    .unwrap();
    assert_eq!(
        q0["i"].as_primitive::<Int32Type>().values(),
        &[1, 81, 161, 241, 321]
    );
    let q0_dists = q0[DIST_COL].as_primitive::<Float32Type>();
    assert!(
        q0_dists
            .values()
            .iter()
            .all(|dist| *dist == q0_dists.value(0)),
        "the five duplicated neighbors must be genuine ties, got distances {:?}",
        q0_dists.values()
    );
}

/// Any `refine_factor` sends the query onto a reranking path the shared-scan
/// batch node does not implement, so the scanner must fall back to the
/// per-query indexed loop and still produce correctly grouped results.
///
/// All of `refine(0)`, `refine(1)`, and `refine(2)` must fall back:
/// `refine(1)` still reranks on the single-query path (a factor of 1 is not
/// a no-op), and `refine(0)` is rejected there with `Refine factor cannot be
/// zero` — the batch path would instead return empty results. Covering the
/// boundary factors guards the `refine_factor.is_some()` gate against
/// regressing back to a `> 1` check.
#[tokio::test]
async fn test_batch_knn_indexed_refine_falls_back() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    test_ds.make_vector_index().await.unwrap();
    let dataset = &test_ds.dataset;
    let (queries, _query_values) = batch_knn_two_queries();

    for refine_factor in [1u32, 2] {
        let mut scan = dataset.scan();
        scan.nearest("vec", &queries, 2).unwrap();
        scan.refine(refine_factor);
        scan.project(&["i"]).unwrap();

        let plan = scan.explain_plan(false).await.unwrap();
        assert!(
            !plan.contains("ANNIvfBatch"),
            "refine({refine_factor}) must not use the shared-scan batch node, got:\n{plan}"
        );
        assert!(
            plan.contains("ANNSubIndex"),
            "refine({refine_factor}) batch search should fall back to the per-query \
                 indexed loop, got:\n{plan}"
        );

        let batch = scan.try_into_batch().await.unwrap();
        assert_query_index_field(&batch);
        assert_eq!(
            batch[QUERY_INDEX_COL].as_primitive::<Int32Type>().values(),
            &[0, 0, 1, 1],
            "refine({refine_factor}) should still group results per query"
        );
    }

    // refine(0) is rejected on the fallback (per-query) path; the batch path
    // must not silently accept it and return empty results instead.
    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, 2).unwrap();
    scan.refine(0);
    scan.project(&["i"]).unwrap();
    let result = scan.try_into_batch().await;
    assert!(
        result.is_err(),
        "refine(0) must error rather than fall through to an empty batch result"
    );
}

/// Without pinned nprobes the shared-scan fast path is not equivalent to
/// single-query search (the single-query path applies an adaptive
/// `early_pruning` floor and late-search expansion that the batch path does
/// not), so the scanner must fall back to the per-query loop, which reuses
/// the real adaptive search and stays exact.
#[tokio::test]
async fn test_batch_knn_indexed_adaptive_nprobes_falls_back() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    test_ds.make_vector_index().await.unwrap();
    let dataset = &test_ds.dataset;
    let (queries, query_values) = batch_knn_two_queries();
    let k = 2;

    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, k).unwrap();
    // No nprobes() call: adaptive (minimum_nprobes=1, maximum_nprobes=None).
    scan.project(&["i"]).unwrap();

    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        !plan.contains("ANNIvfBatch"),
        "adaptive nprobes must not use the shared-scan batch node, got:\n{}",
        plan
    );
    assert!(
        plan.contains("ANNSubIndex"),
        "adaptive nprobes batch search should fall back to the per-query loop, got:\n{}",
        plan
    );

    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    // The fallback runs real single-query searches, so it stays exact even
    // with adaptive nprobes.
    assert_batch_matches_single_queries(dataset, &batch, &query_values, k, true, None, None).await;
}

/// `nprobes(0)` is not rejected by the query builder, so both probe bounds
/// become zero. The single-query
/// path then probes nothing and returns an empty result, whereas the batch
/// node would clamp `nprobes` up to one partition — a silent divergence. The
/// scanner must fall back so the per-query loop defines the semantics of
/// `nprobes(0)`, and the grouped batch result must equal repeated single-query
/// search (both empty here).
#[tokio::test]
async fn test_batch_knn_indexed_zero_nprobes_falls_back() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    test_ds.make_vector_index().await.unwrap();
    let dataset = &test_ds.dataset;
    let (queries, query_values) = batch_knn_two_queries();
    let k = 2;

    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, k).unwrap();
    scan.nprobes(0);
    scan.project(&["i"]).unwrap();

    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        !plan.contains("ANNIvfBatch"),
        "nprobes(0) must not use the shared-scan batch node (which would clamp \
             to one partition), got:\n{plan}"
    );
    assert!(
        plan.contains("ANNSubIndex"),
        "nprobes(0) batch search should fall back to the per-query indexed loop, got:\n{plan}"
    );

    // The fallback runs the real single-query path per query, so the grouped
    // batch result must match issuing each query on its own with nprobes(0).
    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    let query_count = query_values.len() / 32;
    for query_index in 0..query_count {
        let query =
            Float32Array::from(query_values[query_index * 32..(query_index + 1) * 32].to_vec());
        let mut single_scan = dataset.scan();
        single_scan.nearest("vec", &query, k).unwrap();
        single_scan.nprobes(0);
        single_scan.project(&["i"]).unwrap();
        let single = single_scan.try_into_batch().await.unwrap();

        let query_indices = batch[QUERY_INDEX_COL].as_primitive::<Int32Type>();
        let mask = BooleanArray::from_iter(
            query_indices
                .iter()
                .map(|value| value.map(|value| value == query_index as i32)),
        );
        let batch_slice = arrow::compute::filter_record_batch(&batch, &mask).unwrap();
        assert_eq!(
            batch_slice["i"].as_primitive::<Int32Type>().values(),
            single["i"].as_primitive::<Int32Type>().values(),
            "nprobes(0) query {query_index}: batch rows must match single-query rows"
        );
    }
}

/// A caller-supplied external row-address mask (`with_row_id_prefilter`) is
/// applied per query on the single-query prefilter path (`with_external_mask`)
/// but is not carried by the shared batch scan. An otherwise batch-eligible
/// query must therefore fall back to the per-query loop when a mask is present,
/// and every returned row must honor the mask — otherwise the batch path would
/// silently return masked-out rows.
#[tokio::test]
async fn test_batch_knn_indexed_external_mask_falls_back() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    test_ds.make_vector_index().await.unwrap();
    let dataset = &test_ds.dataset;
    let (queries, _query_values) = batch_knn_two_queries();
    let k = 15;

    // Same query shape as `test_batch_knn_indexed`: without a mask it is
    // batch-eligible, so the mask is the only thing that forces the fallback.
    let mut unmasked = dataset.scan();
    unmasked.nearest("vec", &queries, k).unwrap();
    unmasked.minimum_nprobes(2).maximum_nprobes(2);
    unmasked.project(&["i"]).unwrap();
    let unmasked_plan = unmasked.explain_plan(false).await.unwrap();
    assert!(
        unmasked_plan.contains("ANNIvfBatch"),
        "without a mask this query should use the shared-scan batch node, got:\n{unmasked_plan}"
    );

    // Build an allowlist from the dataset's row addresses (freshly created
    // single fragment, so _rowid == row address).
    let mut scan = dataset.scan();
    scan.with_row_id();
    let all_ids = batch_row_ids(&scan.try_into_batch().await.unwrap());
    let allow: Vec<u64> = all_ids.iter().copied().step_by(2).collect();
    let allow_set: BTreeSet<u64> = allow.iter().copied().collect();

    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, k).unwrap();
    scan.minimum_nprobes(2).maximum_nprobes(2);
    scan.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(
        allow.iter().copied(),
    )));
    scan.with_row_id();

    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        !plan.contains("ANNIvfBatch"),
        "an external row mask must not use the shared-scan batch node, which \
             does not carry the mask, got:\n{plan}"
    );
    assert!(
        plan.contains("ANNSubIndex"),
        "a masked batch query should fall back to the per-query indexed loop, got:\n{plan}"
    );

    // The fallback must honor the mask: every returned row is in the allowlist.
    let got = batch_row_ids(&scan.try_into_batch().await.unwrap());
    assert!(
        !got.is_empty(),
        "masked batch KNN should still return allowed rows"
    );
    for id in got {
        assert!(
            allow_set.contains(&id),
            "returned _rowid {id} not in allowlist"
        );
    }
}

/// The shared-scan fast path is only equivalent to repeated single-query
/// search when the selected `index_segments` cover every requested fragment.
/// With one segment per fragment, requesting both fragments but selecting
/// only the first segment leaves fragment 1 covered solely by the
/// *unselected* segment: the batch node would search just the selected
/// segment and silently drop it. Eligibility must be computed from the
/// selected segments' coverage (as `knn_combined` does), not the whole
/// logical index, so the scanner falls back to the per-query loop, which
/// re-scores the uncovered fragment on the flat path and returns every row.
#[tokio::test]
async fn test_batch_knn_indexed_partial_segment_selection_falls_back() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    // One segment per fragment: segment_ids[0] covers fragment 0 (i=0..200),
    // segment_ids[1] covers fragment 1 (i=200..400).
    let segment_ids = test_ds.make_segmented_vector_index().await.unwrap();
    let dataset = &test_ds.dataset;
    let fragments = dataset.fragments();
    assert_eq!(fragments.len(), 2, "base dataset should have two fragments");

    let (queries, _query_values) = batch_knn_two_queries();
    // k covers every row in both requested fragments (200 each), so a
    // complete search returns 400 rows per query.
    let k = 400;

    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, k).unwrap();
    scan.minimum_nprobes(2).maximum_nprobes(2);
    // Request both indexed fragments but select only the segment covering
    // fragment 0; fragment 1 is covered only by the unselected segment.
    scan.with_fragments(vec![fragments[0].clone(), fragments[1].clone()]);
    scan.with_index_segments(vec![segment_ids[0]]).unwrap();
    scan.project(&["i"]).unwrap();

    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        !plan.contains("ANNIvfBatch"),
        "a requested fragment outside the selected segments must force a fallback, \
             not a shared scan that drops it, got:\n{plan}"
    );

    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    assert_eq!(
        batch.num_rows(),
        2 * k,
        "each query must return all 400 rows across both requested fragments"
    );
    let query_indices = batch[QUERY_INDEX_COL].as_primitive::<Int32Type>();
    for query_index in 0..2 {
        let rows_for_query = query_indices
            .iter()
            .filter(|value| *value == Some(query_index))
            .count();
        assert_eq!(
            rows_for_query, k,
            "query_index {query_index} must cover both fragments (got {rows_for_query})"
        );
    }
    // Fragment 0 (i in 0..200) comes from the selected segment; fragment 1
    // (i in 200..400) must appear via the flat fallback.
    let i_array = batch["i"].as_primitive::<Int32Type>();
    assert!(
        i_array
            .iter()
            .any(|v| v.is_some_and(|val| (0..200).contains(&val)))
            && i_array
                .iter()
                .any(|v| v.is_some_and(|val| (200..400).contains(&val))),
        "results must include rows from both the selected segment and the flat-fallback fragment"
    );
}

/// A wide batch probes more distinct partitions than one streaming chunk holds
/// (`STREAMING_SEARCH_BATCH_SIZE` = 16), so `search_partitions_batch` scores
/// them in several `spawn_cpu` dispatches, threading the per-query top-k heaps
/// across chunk boundaries. The small indexes in the other tests fit in a
/// single chunk and never exercise that seam; here an exact (flat) index with
/// more partitions than the chunk size, probed in full, pins the multi-chunk
/// path to repeated single-query search.
#[tokio::test]
async fn test_batch_knn_indexed_streams_multiple_chunks() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    // More partitions than one streaming chunk so scoring spans multiple
    // chunks; exact (flat) storage with every partition probed keeps the batch
    // result an exact match for single-query search.
    let num_partitions = 20;
    let params = VectorIndexParams::ivf_flat(num_partitions, MetricType::L2);
    test_ds
        .dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("idx".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();
    let dataset = &test_ds.dataset;

    // Guard the premise of this test: `nprobes(num_partitions)` probes every
    // partition, so the batch spans multiple streaming chunks only if the
    // partition count exceeds the chunk size. If the default chunk size is
    // ever raised past `num_partitions`, fail loudly here rather than let the
    // test silently collapse to a single chunk and stop covering the seam.
    let chunk_size = *crate::index::vector::ivf::v2::STREAMING_SEARCH_BATCH_SIZE;
    assert!(
        num_partitions > chunk_size,
        "test needs more partitions ({num_partitions}) than the streaming chunk size \
             ({chunk_size}) to span multiple chunks",
    );

    let (queries, query_values) = batch_knn_two_queries();
    let k = 2;
    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, k).unwrap();
    // Probe every partition so both paths are exact regardless of centroid
    // proximity, and so the batch spans multiple streaming chunks.
    scan.minimum_nprobes(num_partitions)
        .maximum_nprobes(num_partitions);
    scan.project(&["i"]).unwrap();

    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ANNIvfBatch"),
        "wide IVF batch KNN should use the shared-scan batch node, got:\n{plan}"
    );

    let batch = scan.try_into_batch().await.unwrap();
    assert_batch_matches_single_queries(
        dataset,
        &batch,
        &query_values,
        k,
        true,
        None,
        Some(num_partitions),
    )
    .await;
}

/// IVF_HNSW is an unsupported index type for the shared-scan batch path (its
/// graph sub-index has no global top-k heap), so batch search must fall back
/// to the per-query indexed loop and still produce correct grouped results.
#[tokio::test]
async fn test_batch_knn_indexed_hnsw_falls_back() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    test_ds.make_ivf_hnsw_index().await.unwrap();
    let dataset = &test_ds.dataset;
    let (queries, query_values) = batch_knn_two_queries();
    let k = 2;

    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, k).unwrap();
    scan.minimum_nprobes(2).maximum_nprobes(2);
    scan.project(&["i"]).unwrap();

    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        !plan.contains("ANNIvfBatch"),
        "HNSW batch search must not use the shared-scan batch node, got:\n{}",
        plan
    );
    assert!(
        plan.contains("ANNSubIndex"),
        "HNSW batch search should fall back to the per-query indexed loop, got:\n{}",
        plan
    );

    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    assert_batch_matches_single_queries(dataset, &batch, &query_values, k, true, None, Some(2))
        .await;
}

/// Regression test for cosine batch search: each query vector must be
/// normalized independently. The two queries below have very different
/// magnitudes, so normalizing the concatenated batch key by a single global
/// norm (the bug) would scale them unequally and diverge from per-query
/// single search.
#[tokio::test]
async fn test_batch_knn_indexed_cosine_normalizes_per_query() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    test_ds
        .make_vector_index_with_metric(MetricType::Cosine)
        .await
        .unwrap();
    let dataset = &test_ds.dataset;

    // q0: small-magnitude constant direction; q1: large-magnitude ramp.
    let mut query_values = vec![0.05f32; 32];
    query_values.extend((1..=32).map(|v| v as f32 * 3.0));
    let queries =
        FixedSizeListArray::try_new_from_values(Float32Array::from(query_values.clone()), 32)
            .unwrap();

    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, 2).unwrap();
    scan.minimum_nprobes(2).maximum_nprobes(2);
    scan.project(&["i"]).unwrap();

    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ANNIvfBatch"),
        "cosine IVF batch KNN should use the shared-scan batch node, got:\n{}",
        plan
    );

    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    assert_batch_matches_single_queries(dataset, &batch, &query_values, 2, true, None, Some(2))
        .await;
}

/// Batch indexed search builds a single shared prefilter for all queries;
/// results must match per-query single search with the same prefilter.
#[tokio::test]
async fn test_batch_knn_indexed_with_prefilter() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    test_ds.make_vector_index().await.unwrap();
    let dataset = &test_ds.dataset;
    let (queries, query_values) = batch_knn_two_queries();
    let k = 2;

    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, k).unwrap();
    scan.minimum_nprobes(2).maximum_nprobes(2);
    scan.filter("i > 100").unwrap();
    scan.prefilter(true);
    scan.project(&["i"]).unwrap();

    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ANNIvfBatch"),
        "prefiltered IVF batch KNN should use the shared-scan batch node, got:\n{}",
        plan
    );

    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    // The shared prefilter must exclude i <= 100 for every query.
    assert!(
        batch["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .all(|i| *i > 100),
        "shared prefilter should remove rows with i <= 100"
    );

    let query_indices = batch[QUERY_INDEX_COL].as_primitive::<Int32Type>();
    for query_index in 0..2 {
        let query =
            Float32Array::from(query_values[query_index * 32..(query_index + 1) * 32].to_vec());
        let single = dataset
            .scan()
            .nearest("vec", &query, k)
            .unwrap()
            .minimum_nprobes(2)
            .maximum_nprobes(2)
            .filter("i > 100")
            .unwrap()
            .prefilter(true)
            .project(&["i"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let mask = BooleanArray::from_iter(
            query_indices
                .iter()
                .map(|v| v.map(|v| v == query_index as i32)),
        );
        let slice = arrow::compute::filter_record_batch(&batch, &mask).unwrap();
        assert_eq!(
            slice["i"].as_primitive::<Int32Type>().values(),
            single["i"].as_primitive::<Int32Type>().values(),
            "prefiltered batch query {query_index} should match single-query search"
        );
    }
}

/// Finds the batch vector-search node in a physical plan.
fn find_ann_ivf_batch_exec(plan: &dyn ExecutionPlan) -> Option<&ANNIvfBatchExec> {
    if let Some(batch_exec) = plan.downcast_ref::<ANNIvfBatchExec>() {
        return Some(batch_exec);
    }
    plan.children()
        .into_iter()
        .find_map(|child| find_ann_ivf_batch_exec(child.as_ref()))
}

/// A caller that matches the batch node in a plan reads the search back out
/// of it through the public `io::exec` surface, so the accessors must return
/// what the scanner fed the constructor.
#[rstest]
#[case::no_prefilter(None)]
#[case::prefilter(Some("i > 100"))]
#[tokio::test]
async fn test_batch_knn_indexed_exposes_plan_inputs(#[case] filter: Option<&str>) {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    test_ds.make_vector_index().await.unwrap();
    let dataset = &test_ds.dataset;
    let (queries, query_values) = batch_knn_two_queries();

    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, 2).unwrap();
    scan.nprobes(2);
    if let Some(filter) = filter {
        scan.filter(filter).unwrap();
        scan.prefilter(true);
    }
    scan.project(&["i"]).unwrap();

    let plan = scan.create_plan().await.unwrap();
    let batch_exec = find_ann_ivf_batch_exec(plan.as_ref())
        .expect("indexed batch KNN should plan an ANNIvfBatchExec");

    let query = batch_exec.query();
    assert_eq!(query.column, "vec");
    assert_eq!(query.k, 2);
    assert_eq!(query.minimum_nprobes, 2);
    assert_eq!(query.maximum_nprobes, Some(2));
    assert_eq!(query.metric_type, Some(DistanceType::L2));
    assert_eq!(
        query.key.as_primitive::<Float32Type>().values(),
        query_values.as_slice(),
        "query key must hold both query vectors concatenated"
    );

    assert_eq!(batch_exec.query_count(), 2);
    assert_eq!(
        query.key.len() / batch_exec.query_count(),
        32,
        "query count must divide the key into the column's vectors"
    );

    assert_eq!(batch_exec.dataset().uri(), dataset.uri());
    assert_eq!(
        batch_exec.dataset().version().version,
        dataset.version().version
    );

    let expected_indices = dataset.load_indices_by_name("idx").await.unwrap();
    assert!(!expected_indices.is_empty());
    assert_eq!(
        batch_exec
            .indices()
            .iter()
            .map(|index| index.uuid)
            .collect::<Vec<_>>(),
        expected_indices
            .iter()
            .map(|index| index.uuid)
            .collect::<Vec<_>>()
    );

    match (filter, batch_exec.prefilter_source()) {
        (None, PreFilterSource::None) => {}
        (Some(_), PreFilterSource::FilteredRowIds(_)) => {}
        (_, source) => panic!("unexpected prefilter source {source:?} for filter {filter:?}"),
    }

    assert_eq!(batch_exec.schema().field(0).name(), QUERY_INDEX_COL);
}

/// Batch indexed search must merge each query's top-k across multiple delta
/// indices, not just within a single delta.
#[tokio::test]
async fn test_batch_knn_indexed_multiple_deltas() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    test_ds.make_vector_index().await.unwrap();
    // Append new data and optimize with `append` to add a second delta
    // index (rather than merging into the existing one).
    test_ds.append_data_with_range(400, 480).await.unwrap();
    test_ds
        .dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let dataset = &test_ds.dataset;
    let segments = dataset.load_indices_by_name("idx").await.unwrap();
    assert!(
        segments.len() >= 2,
        "expected multiple delta index segments to exercise cross-delta merge, got {}",
        segments.len()
    );

    let (queries, query_values) = batch_knn_two_queries();
    let k = 3;
    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, k).unwrap();
    scan.minimum_nprobes(2).maximum_nprobes(2);
    scan.project(&["i"]).unwrap();

    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ANNIvfBatch"),
        "multi-delta IVF batch KNN should use the shared-scan batch node, got:\n{}",
        plan
    );

    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    assert_batch_matches_single_queries(dataset, &batch, &query_values, k, true, None, Some(2))
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
    scan.project(&["i", "vec"]).unwrap();
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
    scan.project(&["i", "vec"]).unwrap();
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
#[case::nearest_only(false, None, false)]
#[case::late_materialization(true, None, false)]
#[case::late_materialization_limit(true, Some(0), false)]
#[case::late_materialization_offset(true, Some(37), false)]
#[case::nulls_and_ties(true, Some(37), true)]
#[tokio::test(flavor = "multi_thread")]
async fn test_flat_knn_large_limit_preserves_global_order(
    #[case] materialize_id: bool,
    #[case] offset: Option<usize>,
    #[case] nulls_and_ties: bool,
) {
    // Cover both the root merge (#7865) and a merge below GlobalLimitExec
    // followed by late materialization (lancedb/lancedb#4214).
    let rows = 12_288usize;
    let k = 9_000usize; // Span multiple default-sized output batches.
    let vector_value = |id: usize| {
        if nulls_and_ties {
            (rows - id).div_ceil(2)
        } else {
            rows - id
        }
    };
    let mut vectors = array::cycle_vec(
        array::cycle::<Float32Type>((0..rows).map(|id| vector_value(id) as f32).collect()),
        Dimension::from(1),
    );
    if nulls_and_ties {
        vectors = vectors.with_nulls(&[false, false, false, true]);
    }
    let dataset = gen_batch()
        .col("id", array::step::<UInt32Type>())
        .col("vec", vectors)
        .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(4_096))
        .await
        .unwrap();

    let query = Float32Array::from(vec![0.0]);
    let mut scan = dataset.scan();
    scan.nearest("vec", &query, k + offset.unwrap_or(0))
        .unwrap();
    scan.target_parallelism(8);
    if materialize_id {
        scan.project(&["id"]).unwrap();
    } else {
        scan.project(&["vec"]).unwrap();
    }
    if let Some(offset) = offset {
        scan.limit(Some(k as i64), Some(offset as i64)).unwrap();
    }

    let plan = scan.create_plan().await.unwrap();
    let displayed = DisplayableExecutionPlan::new(plan.as_ref())
        .indent(true)
        .to_string();
    assert_eq!(displayed.matches("SortExec:").count(), 1, "{displayed}");
    assert_eq!(
        displayed.contains("source=stream(_rowid)"),
        materialize_id,
        "{displayed}"
    );
    let batch = scan.try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), k);
    let distances = batch[DIST_COL].as_primitive::<Float32Type>();
    assert_eq!(distances.null_count(), 0);
    let mut expected_ids = (0..rows)
        .filter(|id| !nulls_and_ties || id % 4 != 3)
        .collect::<Vec<_>>();
    expected_ids.sort_by_key(|id| (vector_value(*id), *id));
    for (rank, id) in expected_ids
        .into_iter()
        .skip(offset.unwrap_or(0))
        .take(k)
        .enumerate()
    {
        let value = vector_value(id) as f32;
        assert_eq!(distances.value(rank), value * value);
        if materialize_id {
            // Distances decrease with row id; ties use ascending row id.
            assert_eq!(
                batch["id"].as_primitive::<UInt32Type>().value(rank),
                id as u32
            );
        }
    }
    // Catch the ordering contract deterministically even if this execution
    // happened to receive the parallel batches in distance order.
    assert!(
        plan.output_ordering().is_some(),
        "flat KNN must retain its output ordering:\n{displayed}"
    );
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
