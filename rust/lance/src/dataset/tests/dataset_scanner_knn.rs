// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use arrow::array::as_primitive_array;
use arrow::datatypes::Int32Type;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Int32Array, ListArray,
    RecordBatch, RecordBatchIterator, StructArray, UInt8Array, UInt32Array,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::concat::concat_batches;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use futures::TryStreamExt;
use lance_arrow::{FixedSizeListArrayExt, SchemaExt};
use lance_core::utils::address::RowAddress;
use lance_core::utils::tempfile::TempStrDir;
use lance_core::{ROW_ADDR, ROW_ID};
use lance_datagen::{ArrayGeneratorExt, BatchCount, Dimension, RowCount, array, gen_batch};
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::{RQBuildParams, RQRotationType};
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::vector::{ApproxMode, DIST_COL};
use lance_linalg::distance::{DistanceType, MetricType};
use roaring::RoaringBitmap;
use rstest::rstest;

use crate::Error;
use crate::dataset::scanner::test_dataset::{
    TestVectorDataset, assert_query_index_field, batch_knn_two_queries, batch_row_ids,
    make_binary_vector_dataset,
};
use crate::dataset::scanner::{RowAddrMask, RowAddrTreeMap, Scanner};
use crate::dataset::tests::dataset_scanner_segments::FragmentVectorFixture;
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

#[rstest]
#[case::flat(None)]
#[case::ivf_flat(Some(VectorIndexParams::with_ivf_flat_params(
    MetricType::L2,
    IvfBuildParams::new(2)
)))]
#[case::ivf_pq(Some(VectorIndexParams::ivf_pq(2, 4, 2, MetricType::L2, 2)))]
#[tokio::test]
async fn test_cross_column_refinement_candidates(
    #[case] index_params: Option<VectorIndexParams>,
    #[values(MetricType::L2, MetricType::Cosine, MetricType::Dot)] metric: MetricType,
) {
    let mut dataset = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("coarse", array::rand_vec::<Float32Type>(Dimension::from(4)))
        .col("full", array::rand_vec::<Float32Type>(Dimension::from(8)))
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(32))
        .await
        .unwrap();
    if let Some(params) = &index_params {
        dataset
            .create_index(&["coarse"], IndexType::Vector, None, params, false)
            .await
            .unwrap();
    }
    let coarse_query = Float32Array::from(vec![0.1, 0.2, 0.3, 0.4]);
    let full_query = Float32Array::from(vec![0.4, 0.3, 0.2, 0.1, 0.5, 0.6, 0.7, 0.8]);
    let candidates = dataset
        .scan()
        .nearest("coarse", &coarse_query, 12)
        .unwrap()
        .distance_metric(MetricType::L2)
        .nprobes(2)
        .project(&["id", "full"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(candidates.num_rows(), 12);

    let vectors = candidates["full"].as_fixed_size_list();
    let ids = candidates["id"].as_primitive::<Int32Type>();
    let mut expected = (0..candidates.num_rows())
        .map(|i| {
            let vector = vectors.value(i);
            let vector = vector.as_primitive::<Float32Type>();
            let pairs = vector.values().iter().zip(full_query.values().iter());
            let dot = pairs
                .clone()
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>();
            let distance = match metric {
                MetricType::L2 => pairs
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                    .sum(),
                MetricType::Cosine => {
                    let norm = vector
                        .values()
                        .iter()
                        .map(|&value| f64::from(value).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let query_norm = full_query
                        .values()
                        .iter()
                        .map(|&value| f64::from(value).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    1.0 - dot / (norm * query_norm)
                }
                MetricType::Dot => 1.0 - dot,
                _ => unreachable!(),
            };
            (ids.value(i), distance as f32)
        })
        .collect::<Vec<_>>();
    expected.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    expected.truncate(6);

    let mut scan = dataset.scan();
    scan.nearest("coarse", &coarse_query, 6)
        .unwrap()
        .distance_metric(MetricType::L2)
        .nprobes(2)
        .refine(2)
        .with_refine_column("full", &full_query, metric)
        .unwrap()
        .project(&["id", DIST_COL])
        .unwrap();
    let plan = scan.explain_plan(false).await.unwrap();
    if index_params.is_some() {
        assert!(plan.contains("ANNSubIndex"), "{plan}");
        // Quantized candidates go directly to final-space scoring.
        assert_eq!(plan.matches("KNNVectorDistance").count(), 1, "{plan}");
    }

    let result = scan.try_into_batch().await.unwrap();
    let actual_ids = result["id"].as_primitive::<Int32Type>();
    let actual_distances = result[DIST_COL].as_primitive::<Float32Type>();
    assert_eq!(result.num_rows(), expected.len());
    for (i, (id, distance)) in expected.iter().enumerate() {
        assert_eq!(actual_ids.value(i), *id);
        assert!((actual_distances.value(i) - distance).abs() < 1e-5);
    }
    assert!(result.schema().column_with_name("full").is_none());
    assert!(result.schema().column_with_name("coarse").is_none());
}

#[tokio::test]
async fn test_cross_column_refinement_validation_and_reset() {
    let dataset = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("coarse", array::rand_vec::<Float32Type>(Dimension::from(4)))
        .col("full", array::rand_vec::<Float32Type>(Dimension::from(8)))
        .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(8))
        .await
        .unwrap();
    let coarse = Float32Array::from(vec![0.1; 4]);
    let full = Float32Array::from(vec![0.1; 8]);

    let mut scan = dataset.scan();
    let error = scan
        .with_refine_column("full", &full, MetricType::Cosine)
        .err()
        .unwrap();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("nearest must be set"));

    scan.nearest("coarse", &coarse, 2).unwrap();
    let error = scan
        .with_refine_column("id", &full, MetricType::Cosine)
        .err()
        .unwrap();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("vector"));

    let wrong_type = arrow_array::Float64Array::from(vec![0.1; 8]);
    let error = scan
        .with_refine_column("full", &wrong_type, MetricType::Cosine)
        .err()
        .unwrap();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("Float32"));

    scan.with_refine_column("full", &full, MetricType::Cosine)
        .unwrap();
    let error = scan.explain_plan(false).await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("positive refine_factor"));

    scan.refine(0);
    let error = scan.explain_plan(false).await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("positive refine_factor"));

    let mut overflow = dataset.scan();
    overflow
        .nearest("coarse", &coarse, usize::MAX)
        .unwrap()
        .refine(2)
        .with_refine_column("full", &full, MetricType::Cosine)
        .unwrap();
    let error = overflow.explain_plan(false).await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("overflows"));

    scan.nearest("coarse", &coarse, 2).unwrap();
    scan.explain_plan(false).await.unwrap();
}

#[rstest]
#[tokio::test]
async fn test_cross_column_refinement_sliced_batch_query(#[values(false, true)] build_index: bool) {
    let mut dataset = gen_batch()
        .col("coarse", array::rand_vec::<Float32Type>(Dimension::from(4)))
        .col("full", array::rand_vec::<Float32Type>(Dimension::from(8)))
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(8))
        .await
        .unwrap();
    if build_index {
        dataset
            .create_index(
                &["coarse"],
                IndexType::Vector,
                None,
                &VectorIndexParams::ivf_flat(2, MetricType::L2),
                false,
            )
            .await
            .unwrap();
    }

    let coarse_values =
        Float32Array::from((1..9).map(|value| value as f32 / 10.0).collect::<Vec<_>>());
    let coarse = FixedSizeListArray::try_new_from_values(coarse_values, 4).unwrap();
    let full_values = Float32Array::from((0..32).map(|value| value as f32).collect::<Vec<_>>());
    let full = FixedSizeListArray::try_new_from_values(full_values, 8)
        .unwrap()
        .slice(1, 2);

    let mut scan = dataset.scan();
    scan.nearest("coarse", &coarse, 2)
        .unwrap()
        .nprobes(2)
        .refine(2)
        .with_refine_column("full", &full, MetricType::L2)
        .unwrap()
        .project(&[DIST_COL])
        .unwrap()
        .with_row_id();
    let batch = scan.try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(
        batch[QUERY_INDEX_COL].as_primitive::<Int32Type>().values(),
        &[0, 0, 1, 1]
    );

    for query_index in 0..coarse.len() {
        let mut single = dataset.scan();
        single
            .nearest("coarse", coarse.value(query_index).as_ref(), 2)
            .unwrap()
            .nprobes(2)
            .refine(2)
            .with_refine_column("full", full.value(query_index).as_ref(), MetricType::L2)
            .unwrap()
            .project(&[DIST_COL])
            .unwrap()
            .with_row_id();
        let single = single.try_into_batch().await.unwrap();
        let query_batch = batch.slice(query_index * 2, 2);
        assert_eq!(
            query_batch[ROW_ID].as_primitive::<UInt64Type>().values(),
            single[ROW_ID].as_primitive::<UInt64Type>().values()
        );
        assert_eq!(
            query_batch[DIST_COL].as_primitive::<Float32Type>().values(),
            single[DIST_COL].as_primitive::<Float32Type>().values()
        );
    }
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

#[rstest]
#[case::whole_dataset(false)]
#[case::fragments_first(true)]
#[tokio::test]
async fn test_primitive_query_length_multiple_of_dim_is_rejected(#[case] has_fragments: bool) {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    let dataset = &test_ds.dataset;
    let q: Float32Array = (32..96).map(|v| v as f32).collect();

    let mut scanner = dataset.scan();
    if has_fragments {
        scanner.with_fragments(vec![dataset.fragments()[0].clone()]);
    }
    let Err(err) = scanner.nearest("vec", &q, 2) else {
        panic!("expected primitive query length mismatch error");
    };
    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(
        err.to_string()
            .contains("query dim(64) doesn't match the column vec vector dim(32)"),
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

/// The batch ANN stage must translate the exact cosine range to the SQ
/// index's L2 scale before collecting refinement candidates.
#[tokio::test]
async fn test_batch_refined_knn_scales_cosine_range_for_ann_stage() {
    use lance_linalg::distance::cosine_distance;

    const DIM: usize = 16;
    const ROWS: usize = 512;
    let mut seed: u64 = 42;
    let values: Vec<f32> = (0..ROWS * DIM)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect();
    let query_values = values[..2 * DIM].to_vec();
    let queries = FixedSizeListArray::try_new_from_values(
        Float32Array::from(query_values.clone()),
        DIM as i32,
    )
    .unwrap();
    let rows = values.as_chunks::<DIM>().0;
    let exact = query_values
        .as_chunks::<DIM>()
        .0
        .iter()
        .map(|query| {
            let mut distances = rows
                .iter()
                .map(|row| cosine_distance(query, row))
                .collect::<Vec<_>>();
            distances.sort_by(f32::total_cmp);
            distances
        })
        .collect::<Vec<_>>();
    // Make the unscaled ANN range exclude every exact in-range candidate.
    let lower = exact
        .iter()
        .map(|distances| distances[20])
        .fold(f32::NEG_INFINITY, f32::max);
    let upper = exact
        .iter()
        .map(|distances| distances[60])
        .fold(f32::INFINITY, f32::min)
        .min(2.0 * lower);
    assert!(upper > lower);

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new(
            "vec",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..ROWS as i32)),
            Arc::new(
                FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();
    let tmp = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &tmp,
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            None,
            &VectorIndexParams::with_ivf_sq_params(
                DistanceType::Cosine,
                IvfBuildParams::new(1),
                SQBuildParams::default(),
            ),
            true,
        )
        .await
        .unwrap();

    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, 5)
        .unwrap()
        .minimum_nprobes(1)
        .maximum_nprobes(1)
        .refine(10)
        .distance_range(Some(lower), Some(upper))
        .project(&["id"])
        .unwrap();
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("ANNIvfBatch"), "{plan}");
    assert!(plan.contains("BatchRefine"), "{plan}");
    let batch = scan.try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), 10);
    for query_index in 0..2 {
        let query =
            Float32Array::from(query_values[query_index * DIM..(query_index + 1) * DIM].to_vec());
        let single = dataset
            .scan()
            .nearest("vec", &query, 5)
            .unwrap()
            .minimum_nprobes(1)
            .maximum_nprobes(1)
            .refine(10)
            .distance_range(Some(lower), Some(upper))
            .project(&["id"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(single.num_rows(), 5);
        let mask = BooleanArray::from_iter(
            batch[QUERY_INDEX_COL]
                .as_primitive::<Int32Type>()
                .iter()
                .map(|value| value.map(|value| value == query_index as i32)),
        );
        let actual = arrow::compute::filter_record_batch(&batch, &mask).unwrap();
        assert_eq!(actual["id"].as_ref(), single["id"].as_ref());
        assert_eq!(actual[DIST_COL].as_ref(), single[DIST_COL].as_ref());
    }
}

#[rstest]
#[case::factor_one(1, true, MetricType::L2, "none")]
#[case::shared(2, true, MetricType::L2, "none")]
#[case::cosine(2, true, MetricType::Cosine, "none")]
#[case::dot(2, true, MetricType::Dot, "none")]
#[case::adaptive(2, false, MetricType::L2, "none")]
#[case::prefilter(2, true, MetricType::L2, "prefilter")]
#[case::postfilter(2, true, MetricType::L2, "postfilter")]
#[case::empty(2, true, MetricType::L2, "empty")]
#[case::deleted(2, true, MetricType::L2, "deleted")]
#[case::bounded(2, true, MetricType::L2, "bounded")]
#[tokio::test]
async fn test_batch_knn_indexed_refine(
    #[case] factor: u32,
    #[case] fixed_probes: bool,
    #[case] metric: MetricType,
    #[case] scenario: &str,
) {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    test_ds.make_vector_index_with_metric(metric).await.unwrap();
    if scenario == "deleted" {
        test_ds.dataset.delete("i % 2 = 0").await.unwrap();
    }
    let dataset = &test_ds.dataset;
    let (queries, query_values) = batch_knn_two_queries();
    let mut scan = dataset.scan();
    scan.nearest("vec", &queries, 10).unwrap();
    scan.refine(factor).project(&["i", "vec"]).unwrap();
    if fixed_probes {
        scan.nprobes(2);
    }
    match scenario {
        "prefilter" | "postfilter" => {
            scan.filter("i > 100")
                .unwrap()
                .prefilter(scenario == "prefilter");
        }
        "empty" => {
            scan.filter("i < 0").unwrap().prefilter(true);
        }
        "bounded" => {
            scan.distance_range(Some(1.0), Some(1_000_000.0));
        }
        _ => {}
    }
    let plan = scan.explain_plan(false).await.unwrap();
    assert_eq!(plan.contains("BatchRefine"), fixed_probes, "{plan}");
    assert_eq!(plan.contains("ANNIvfBatch"), fixed_probes, "{plan}");
    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    for query_index in 0..2 {
        let key =
            Float32Array::from(query_values[query_index * 32..(query_index + 1) * 32].to_vec());
        let mut single = dataset.scan();
        single.nearest("vec", &key, 10).unwrap();
        single.refine(factor).project(&["i", "vec"]).unwrap();
        if fixed_probes {
            single.nprobes(2);
        }
        match scenario {
            "prefilter" | "postfilter" => {
                single
                    .filter("i > 100")
                    .unwrap()
                    .prefilter(scenario == "prefilter");
            }
            "empty" => {
                single.filter("i < 0").unwrap().prefilter(true);
            }
            "bounded" => {
                single.distance_range(Some(1.0), Some(1_000_000.0));
            }
            _ => {}
        }
        let single = single.try_into_batch().await.unwrap();
        let mask = BooleanArray::from_iter(
            batch[QUERY_INDEX_COL]
                .as_primitive::<Int32Type>()
                .iter()
                .map(|value| value.map(|value| value == query_index as i32)),
        );
        let actual = arrow::compute::filter_record_batch(&batch, &mask).unwrap();
        assert_eq!(actual["i"].as_ref(), single["i"].as_ref());
        assert_eq!(actual["vec"].as_ref(), single["vec"].as_ref());
        assert_eq!(actual[DIST_COL].as_ref(), single[DIST_COL].as_ref());
        if scenario != "none" {
            continue;
        }
        let exact = dataset
            .scan()
            .nearest("vec", &key, 10)
            .unwrap()
            .use_index(false)
            .distance_metric(metric)
            .project(&["i"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let expected_ids = exact["i"].as_primitive::<Int32Type>();
        let hits = actual["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .filter(|id| expected_ids.values().contains(id))
            .count();
        assert!(
            hits as f32 / 10.0 >= 0.5,
            "recall={hits}/10, metric={metric:?}"
        );
    }
}

#[rstest]
#[case::fixed(true)]
#[case::adaptive(false)]
#[tokio::test]
async fn test_batch_knn_indexed_refine_zero(#[case] fixed_probes: bool) {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    test_ds.make_vector_index().await.unwrap();
    let (queries, _) = batch_knn_two_queries();
    let mut scan = test_ds.dataset.scan();
    scan.nearest("vec", &queries, 2).unwrap().refine(0);
    if fixed_probes {
        scan.nprobes(2);
    }
    let error = scan.try_into_batch().await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("Refine factor cannot be zero"));
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
#[rstest]
#[tokio::test]
async fn test_batch_knn_indexed_external_mask_falls_back(#[values(false, true)] refine: bool) {
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
    if refine {
        scan.refine(2);
    }
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
#[rstest]
#[tokio::test]
async fn test_batch_knn_indexed_partial_segment_selection_falls_back(
    #[values(false, true)] refine: bool,
) {
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
    if refine {
        scan.refine(2);
    }
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

/// A cosine SQ index ranks as twice `1 - cos`, so a refined search that also
/// applied the range inside the index would drop rows whose exact distance
/// is in range.
#[tokio::test]
async fn test_refined_knn_applies_distance_range_to_exact_distances() {
    use lance_linalg::distance::cosine_distance;

    const DIM: usize = 16;
    const ROWS: usize = 2000;
    let mut seed: u64 = 42;
    let values: Vec<f32> = (0..ROWS * DIM)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "vec",
        DataType::FixedSizeList(
            Arc::new(ArrowField::new("item", DataType::Float32, true)),
            DIM as i32,
        ),
        true,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values.clone()), DIM as i32)
                .unwrap(),
        )],
    )
    .unwrap();
    let tmp = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &tmp,
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            None,
            &VectorIndexParams::with_ivf_hnsw_sq_params(
                DistanceType::Cosine,
                IvfBuildParams::new(1),
                HnswBuildParams::default(),
                SQBuildParams::default(),
            ),
            true,
        )
        .await
        .unwrap();

    let query = values[..DIM].to_vec();
    let mut exact: Vec<f32> = values
        .as_chunks::<DIM>()
        .0
        .iter()
        .map(|row| cosine_distance(&query, row))
        .collect();
    exact.sort_by(f32::total_cmp);
    // Rows from the 20th nearest out, capped so that `upper <= 2 * lower`:
    // no row is in range on both the exact and the doubled scale.
    let lower = exact[20];
    let upper = exact[60].min(2.0 * lower);
    assert!(upper > lower);

    let batch = dataset
        .scan()
        .nearest("vec", &Float32Array::from(query), 5)
        .unwrap()
        .distance_metric(DistanceType::Cosine)
        .refine(1)
        .distance_range(Some(lower), Some(upper))
        .try_into_batch()
        .await
        .unwrap();
    let distances = batch[DIST_COL].as_primitive::<Float32Type>().values();
    assert!(!distances.is_empty(), "no rows within the exact range");
    for distance in distances {
        assert!(
            (lower..upper).contains(distance),
            "{distance} is outside [{lower}, {upper})"
        );
    }
}

#[tokio::test]
async fn test_batch_knn_quantized_refinement_uses_rerank_path() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    test_ds.make_rq_vector_index(4).await.unwrap();
    let (queries, _) = batch_knn_two_queries();

    let mut scan = test_ds.dataset.scan();
    scan.nearest("vec", &queries, 10).unwrap();
    scan.nprobes(2);
    scan.quantized_refine(40);

    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        !plan.contains("ANNIvfBatch"),
        "quantized refinement must not use the shared-scan batch node, got:\n{plan}"
    );
    assert!(plan.contains("quantized_refine_factor=40"), "{plan}");
}

/// An in-place vector update (`update_columns` + `Operation::Update`) prunes the
/// fragment from the old segment's bitmap while that segment still physically
/// holds the pre-update vector, and `optimize_indices(append)` indexes the fresh
/// vector in a new delta. Like the single-query path, the batch node must apply
/// each segment's ownership so the stale copy is neither returned nor counted.
#[rstest]
#[case::fixed_nprobes(Some(2))]
#[tokio::test]
async fn test_batch_knn_skips_rows_owned_by_newer_delta(
    #[case] nprobes: Option<usize>,
    #[values(false, true)] enable_stable_row_ids: bool,
) {
    use crate::dataset::transaction::{Operation, UpdateMode, UpdatedFragmentOffsets};
    use arrow_array::{RecordBatchReader, UInt64Array};

    const DIM: usize = 8;
    const BULK_ROWS: usize = 1000;
    const UPDATED_ID: u32 = 10_000;
    const STALE_VALUE: f32 = 10.0;
    const FRESH_VALUE: f32 = -10.0;
    let k = 5;

    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let vector_type = DataType::FixedSizeList(
        Arc::new(ArrowField::new("item", DataType::Float32, true)),
        DIM as i32,
    );
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::UInt32, false),
        ArrowField::new("vec", vector_type.clone(), true),
    ]));
    let constant_vector = |value: f32| {
        FixedSizeListArray::try_new_from_values(
            Float32Array::from_iter_values(std::iter::repeat_n(value, DIM)),
            DIM as i32,
        )
        .unwrap()
    };

    let bulk = lance_testing::datagen::generate_random_array_with_seed::<Float32Type>(
        BULK_ROWS * DIM,
        [7; 32],
    );
    let bulk_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..BULK_ROWS as u32)),
            Arc::new(FixedSizeListArray::try_new_from_values(bulk, DIM as i32).unwrap()),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(bulk_batch)], schema.clone()),
        test_uri,
        Some(WriteParams {
            enable_stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let stale_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values([UPDATED_ID])),
            Arc::new(constant_vector(STALE_VALUE)),
        ],
    )
    .unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(stale_batch)], schema.clone()),
            None,
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("idx".into()),
            &VectorIndexParams::ivf_flat(4, MetricType::L2),
            true,
        )
        .await
        .unwrap();

    let mut fragment = dataset.get_fragment(1).unwrap();
    let mut fragment_scan = fragment.scan();
    fragment_scan.with_row_id();
    let updated_row_id = fragment_scan.try_into_batch().await.unwrap()[ROW_ID]
        .as_primitive::<UInt64Type>()
        .value(0);
    let update_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(ROW_ID, DataType::UInt64, false),
        ArrowField::new("vec", vector_type, true),
    ]));
    let update_batch = RecordBatch::try_new(
        update_schema.clone(),
        vec![
            Arc::new(UInt64Array::from_iter_values([updated_row_id])),
            Arc::new(constant_vector(FRESH_VALUE)),
        ],
    )
    .unwrap();
    let right_stream: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        vec![Ok(update_batch)],
        update_schema,
    ));
    let updated = fragment
        .update_columns_with_offsets(right_stream, ROW_ID, ROW_ID)
        .await
        .unwrap();
    let updated_fragment_id = updated.fragment.id;
    let mut dataset = Dataset::commit(
        test_uri,
        Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![updated.fragment],
            new_fragments: vec![],
            fields_modified: updated.fields_modified,
            mem_wal_dir_id: None,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: Some(UpdatedFragmentOffsets(HashMap::from([(
                updated_fragment_id,
                updated.matched_offsets,
            )]))),
        },
        Some(dataset.version().version),
        None,
        None,
        Default::default(),
        true,
    )
    .await
    .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    assert_eq!(dataset.load_indices_by_name("idx").await.unwrap().len(), 2);

    // Query 0 sits on the stale value, query 1 on the fresh one.
    let query_values: Vec<f32> = [STALE_VALUE, FRESH_VALUE]
        .iter()
        .flat_map(|&value| std::iter::repeat_n(value, DIM))
        .collect();
    let queries = FixedSizeListArray::try_new_from_values(
        Float32Array::from(query_values.clone()),
        DIM as i32,
    )
    .unwrap();
    let configure = |scan: &mut Scanner, query: &dyn Array| {
        scan.nearest("vec", query, k).unwrap();
        match nprobes {
            Some(nprobes) => {
                scan.nprobes(nprobes);
            }
            None => {
                scan.minimum_nprobes(1);
            }
        }
        scan.project(&["id"]).unwrap();
    };

    let mut scan = dataset.scan();
    configure(&mut scan, &queries);
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("ANNIvfBatch"), "got:\n{plan}");
    let batch = scan.try_into_batch().await.unwrap();
    for query_index in 0..2 {
        let query =
            Float32Array::from(query_values[query_index * DIM..(query_index + 1) * DIM].to_vec());
        let mut single_scan = dataset.scan();
        configure(&mut single_scan, &query);
        let single = single_scan.try_into_batch().await.unwrap();
        let mask = BooleanArray::from_iter(
            batch[QUERY_INDEX_COL]
                .as_primitive::<Int32Type>()
                .iter()
                .map(|value| value.map(|value| value == query_index as i32)),
        );
        let batch_rows = arrow::compute::filter_record_batch(&batch, &mask).unwrap();
        let ids = batch_rows["id"].as_primitive::<UInt32Type>().values();
        assert_eq!(
            ids.iter().filter(|&&id| id == UPDATED_ID).count(),
            usize::from(query_index == 1),
            "query {query_index}: the updated row must appear once, with its fresh vector only; ids {ids:?}"
        );
        assert_eq!(batch_rows.num_rows(), k);
        assert_eq!(
            batch_rows[DIST_COL].as_primitive::<Float32Type>().values(),
            single[DIST_COL].as_primitive::<Float32Type>().values(),
            "query {query_index}: batch distances must match single-query"
        );
    }
}

/// A column can carry more than one vector index. The query's metric has to
/// take part in selecting which index answers, not merely veto the first one
/// keyed on the column: a cosine query on a column holding both an L2 and a
/// cosine index must reach the cosine index instead of brute-forcing.
/// Regression test for https://github.com/lance-format/lance/issues/9873
#[tokio::test]
async fn test_knn_selects_index_matching_requested_metric() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    // Two indices on the same `vec` column: L2 created first, cosine second.
    test_ds
        .make_named_vector_index_with_metric("idx_l2", MetricType::L2)
        .await
        .unwrap();
    test_ds
        .make_named_vector_index_with_metric("idx_cos", MetricType::Cosine)
        .await
        .unwrap();

    let dataset = &test_ds.dataset;
    let key: Float32Array = (32..64).map(|v| v as f32).collect();

    // The cosine query must use the cosine index. Before the fix the first
    // keyed index (idx_l2) was selected, vetoed on the metric mismatch, and
    // the whole column brute-forced even though idx_cos matched.
    let mut scan = dataset.scan();
    scan.nearest("vec", &key, 5).unwrap();
    scan.distance_metric(MetricType::Cosine);
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ANNSubIndex") && plan.to_lowercase().contains("cosine"),
        "Expected the cosine index to answer a cosine query, got plan:\n{}",
        plan
    );

    // An L2 query still selects the L2 index.
    let mut scan = dataset.scan();
    scan.nearest("vec", &key, 5).unwrap();
    scan.distance_metric(MetricType::L2);
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ANNSubIndex") && plan.to_lowercase().contains("l2"),
        "Expected the L2 index to answer an L2 query, got plan:\n{}",
        plan
    );
}

/// A query that names no metric ranks by the chosen index's metric even when its
/// fragment scope misses every segment of that index and the search runs flat KNN, so
/// `_distance` keeps one scale whichever fragments the scope selects.
#[rstest]
#[case::automatic_selection(false)]
#[case::explicit_segments(true)]
#[tokio::test]
async fn test_knn_no_metric_uses_index_metric_outside_index_coverage(
    #[case] has_explicit_segments: bool,
) {
    // F0 (ids 0..3) is indexed and F1 (ids 3..6) is appended afterwards. On F1 cosine
    // and L2 rank in reverse order: id 3 is nearest by angle and farthest by L2.
    let vectors: [[f32; 4]; 6] = [
        [3.0, 1.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
        [4.0, 3.0, 0.0, 0.0],
        [10.0, 1.0, 0.0, 0.0],
        [0.6, 0.6, 0.0, 0.0],
        [2.0, 1.0, 0.0, 0.0],
    ];
    let rows_for = |ids: std::ops::Range<usize>| {
        gen_batch()
            .col("id", array::step_custom::<Int32Type>(ids.start as i32, 1))
            .col(
                "vec",
                array::cycle_vec(
                    array::cycle::<Float32Type>(vectors[ids].concat()),
                    Dimension::from(4),
                ),
            )
    };
    let mut dataset = rows_for(0..3)
        .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(3))
        .await
        .unwrap();
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            None,
            &VectorIndexParams::ivf_flat(1, DistanceType::Cosine),
            true,
        )
        .await
        .unwrap();
    dataset
        .append(
            rows_for(3..6).into_reader_rows(RowCount::from(3), BatchCount::from(1)),
            None,
        )
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(
        indices[0].fragment_bitmap.as_ref(),
        Some(&RoaringBitmap::from_iter([0]))
    );
    assert_eq!(dataset.fragments().len(), 2);

    let query = [1.0_f32, 0.0, 0.0, 0.0];
    let norm = |vector: &[f32]| {
        vector
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let cosine_distance = |vector: &[f32; 4]| {
        let dot: f64 = vector
            .iter()
            .zip(query)
            .map(|(value, query_value)| f64::from(*value) * f64::from(query_value))
            .sum();
        1.0 - dot / (norm(vector) * norm(&query))
    };
    let mut scan = dataset.scan();
    scan.nearest("vec", &Float32Array::from(query.to_vec()), 3)
        .unwrap();
    scan.with_fragments(vec![dataset.fragments()[1].clone()]);
    if has_explicit_segments {
        scan.with_index_segments(vec![indices[0].uuid]).unwrap();
    }
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        !plan.contains("ANNSubIndex") && plan.contains("KNNVectorDistance: metric=cosine"),
        "{plan}"
    );
    let batch = scan.try_into_batch().await.unwrap();
    let ids = batch["id"].as_primitive::<Int32Type>().values();
    assert_eq!(ids, &[3, 5, 4], "{plan}");
    let distances = batch[DIST_COL].as_primitive::<Float32Type>().values();
    for (id, distance) in ids.iter().zip(distances.iter()) {
        let expected = cosine_distance(&vectors[*id as usize]);
        assert!(
            (f64::from(*distance) - expected).abs() <= 1e-6,
            "id={id}: distance={distance}, expected={expected}"
        );
    }
}

#[rstest]
#[case::zero_k(Arc::new(Float32Array::from(vec![0.0; 4])), 0, "k must be positive")]
#[case::empty_query(
    Arc::new(Float32Array::from(Vec::<f32>::new())),
    1,
    "Query vector must have non-zero length"
)]
#[case::primitive_dimension(
    Arc::new(Float32Array::from(vec![0.0; 3])),
    1,
    "query dim(3) doesn't match the column vec vector dim(4)"
)]
#[case::list_dimension(
    Arc::new(ListArray::from_iter_primitive::<Float32Type, _, _>([Some(vec![Some(0.0); 3])])),
    1,
    "query dim(3) doesn't match the column vec vector dim(4)"
)]
#[case::fixed_size_list_dimension(
    Arc::new(FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 3]), 3).unwrap()),
    1,
    "query dim(3) doesn't match the column vec vector dim(4)"
)]
#[tokio::test]
async fn test_nearest_rejects_invalid_query(
    #[case] query: ArrayRef,
    #[case] k: usize,
    #[case] message: &str,
    #[values(false, true)] has_fragments: bool,
) {
    let fixture = FragmentVectorFixture::new(false, false).await;
    let mut scanner = fixture.dataset.scan();
    if has_fragments {
        scanner.with_fragments(fixture.dataset.fragments()[1..].to_vec());
    }
    let Err(err) = scanner.nearest("vec", query.as_ref(), k) else {
        panic!("expected invalid query to be rejected");
    };
    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(err.to_string().contains(message), "unexpected error: {err}");
}

#[rstest]
#[case::whole_dataset(false)]
#[case::fragments_first(true)]
#[tokio::test]
async fn test_nearest_rejects_query_type_mismatch(#[case] has_fragments: bool) {
    let (_tmp_dir, dataset) = make_binary_vector_dataset().await.unwrap();
    let mut scanner = dataset.scan();
    if has_fragments {
        scanner.with_fragments(dataset.fragments().to_vec());
    }
    let query = Float32Array::from(vec![0.0; 4]);
    let Err(err) = scanner.nearest("bin", &query, 1) else {
        panic!("expected incompatible query type to be rejected");
    };
    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(
        err.to_string()
            .contains("Column bin has element type UInt8 and the query vector is Float32"),
        "unexpected error: {err}"
    );
}

#[rstest]
#[case::rq4(4)]
#[case::rq8(8)]
#[tokio::test]
async fn test_ivf_rq_quantized_refinement(#[case] num_bits: u8) {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    test_ds.make_rq_vector_index(num_bits).await.unwrap();
    let query_vector = Float32Array::from_iter_values((0..32).map(|value| value as f32));

    let mut accurate = test_ds.dataset.scan();
    accurate
        .nearest("vec", &query_vector, 10)
        .unwrap()
        .nprobes(2)
        .approx_mode(ApproxMode::Accurate)
        .with_row_id()
        .project(&["i"])
        .unwrap();
    let accurate_results = accurate.try_into_batch().await.unwrap();

    // The fixture has 400 rows across multiple fragments. This factor makes
    // the Fast candidate pass exhaustive, so selected-candidate scoring
    // must produce the same top-k set as native Accurate RQ search. Ties in
    // approximate scores may appear in a different order.
    let mut quantized = test_ds.dataset.scan();
    quantized
        .nearest("vec", &query_vector, 10)
        .unwrap()
        .nprobes(2)
        .quantized_refine(40)
        .with_row_id()
        .project(&["i"])
        .unwrap();
    let plan = quantized.explain_plan(false).await.unwrap();
    assert!(plan.contains("quantized_refine_factor=40"), "{plan}");
    assert!(!plan.contains("KNNVectorDistance"), "{plan}");
    let quantized_results = quantized.try_into_batch().await.unwrap();
    let quantized_ids = BTreeSet::from_iter(batch_row_ids(&quantized_results));
    let accurate_ids = BTreeSet::from_iter(batch_row_ids(&accurate_results));
    assert_eq!(quantized_ids, accurate_ids);
    assert!(
        quantized_results[DIST_COL]
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .all(|distance| distance.is_finite())
    );

    // Exact refinement remains a separate final stage and still reads the
    // original vectors only when the caller explicitly asks for it.
    let mut exact = test_ds.dataset.scan();
    exact
        .nearest("vec", &query_vector, 10)
        .unwrap()
        .use_index(false)
        .with_row_id()
        .project(&["i"])
        .unwrap();
    let exact_results = exact.try_into_batch().await.unwrap();

    let mut quantized_then_exact = test_ds.dataset.scan();
    quantized_then_exact
        .nearest("vec", &query_vector, 10)
        .unwrap()
        .nprobes(2)
        .quantized_refine(40)
        .refine(2)
        .with_row_id()
        .project(&["i"])
        .unwrap();
    let plan = quantized_then_exact.explain_plan(false).await.unwrap();
    assert!(plan.contains("quantized_refine_factor=40"), "{plan}");
    assert!(plan.contains("KNNVectorDistance"), "{plan}");
    let quantized_then_exact_results = quantized_then_exact.try_into_batch().await.unwrap();
    assert_eq!(quantized_then_exact_results, exact_results);
}

#[rstest]
#[case::rq4(4)]
#[case::rq8(8)]
#[tokio::test]
async fn test_quantized_refinement_merges_unindexed_distance_scales(
    #[case] num_bits: u8,
    #[values(DistanceType::Cosine, DistanceType::L2, DistanceType::Dot)] metric: DistanceType,
    #[values(false, true)] exact_refine: bool,
    #[values(None, Some((0.15, 0.25)), Some((0.28, 0.32)))] range: Option<(f32, f32)>,
) {
    const DIM: usize = 32;
    let make_batch = |rows: usize, first: f32, nearest_boost: f32| {
        let mut values = Vec::with_capacity(rows * DIM);
        for row in 0..rows {
            // Separate the nearest ten so recall measures the merge rather
            // than approximation noise among nearly tied RQ4 scores.
            let boost = if row < 10 { nearest_boost } else { 0.0 };
            let x = first + boost + row as f32 * 0.0001;
            let tail: Vec<f32> = (1..DIM)
                .map(|dim| ((row * 37 + dim * 19) as f32).sin())
                .collect();
            let norm = tail.iter().map(|value| value * value).sum::<f32>().sqrt();
            values.push(x);
            values.extend(tail.iter().map(|value| value / norm * (1.0 - x * x).sqrt()));
        }
        RecordBatch::try_from_iter([(
            "vec",
            Arc::new(
                FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                    .unwrap(),
            ) as ArrayRef,
        )])
        .unwrap()
    };
    let indexed = make_batch(128, 0.8, 0.04);
    let centroids = indexed["vec"].as_fixed_size_list().slice(0, 4);
    let tmp = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(indexed.clone())], indexed.schema()),
        &tmp,
        Some(WriteParams {
            max_rows_per_file: 32,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let mut rq_params = RQBuildParams::with_rotation_type(num_bits, RQRotationType::Fast);
    let mut rotation = RabitQuantizer::new::<Float32Type>(num_bits, DIM as i32)
        .metadata_ref()
        .clone();
    // Reuse a fixed rotation so recall assertions are deterministic.
    for (index, signs) in rotation
        .fast_rotation_signs
        .as_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *signs = (index * 37 + 19) as u8;
    }
    rq_params.rotation = Some(rotation);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            None,
            &VectorIndexParams::with_ivf_rq_params(
                metric,
                IvfBuildParams::try_with_centroids(4, Arc::new(centroids)).unwrap(),
                rq_params,
            ),
            true,
        )
        .await
        .unwrap();
    let appended = make_batch(16, 0.7, 0.0);
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema()),
            None,
        )
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 5);

    let mut query = vec![0.0; DIM];
    query[0] = 1.0;
    let query = Float32Array::from(query);
    let scale = if metric == DistanceType::Cosine && !exact_refine {
        2.0
    } else {
        1.0
    };
    // Fixture ranges are expressed as 1 - cos. L2 on unit vectors is twice
    // that, as are native RQ cosine scores without exact refinement.
    let metric_scale = if metric == DistanceType::L2 { 2.0 } else { 1.0 };
    let mut exact = dataset.scan();
    exact
        .nearest("vec", &query, 10)
        .unwrap()
        .distance_metric(metric)
        .use_index(false)
        .with_row_id()
        .project::<&str>(&[])
        .unwrap();
    if let Some((lower, upper)) = range {
        exact.distance_range(Some(lower * metric_scale), Some(upper * metric_scale));
    }
    let expected = exact.try_into_batch().await.unwrap();
    let expected_ids = BTreeSet::from_iter(batch_row_ids(&expected));

    let mut scanner = dataset.scan();
    scanner
        .nearest("vec", &query, 10)
        .unwrap()
        .distance_metric(metric)
        .nprobes(4)
        .quantized_refine(64)
        .with_row_id()
        .project::<&str>(&[])
        .unwrap();
    if exact_refine {
        scanner.refine(2);
    }
    if let Some((lower, upper)) = range {
        scanner.distance_range(
            Some(lower * metric_scale * scale),
            Some(upper * metric_scale * scale),
        );
    }
    let plan = scanner.explain_plan(false).await.unwrap();
    if !exact_refine {
        // Only the unindexed scan computes vector distances. Indexed rows
        // retain their RQ scores without taking their original vectors.
        assert_eq!(plan.matches("KNNVectorDistance").count(), 1, "{plan}");
        assert!(!plan.contains("Take:"), "{plan}");
    }
    let results = scanner.try_into_batch().await.unwrap();
    assert_eq!(results.num_rows(), 10);
    let ids = batch_row_ids(&results);
    let recall = ids.iter().filter(|id| expected_ids.contains(id)).count();
    assert!(
        recall >= 5,
        "recall={recall}/10, metric={metric:?}, ids={ids:?}"
    );
    for (row, distance) in results[DIST_COL]
        .as_primitive::<Float32Type>()
        .values()
        .iter()
        .enumerate()
    {
        if let Some((lower, upper)) = range {
            assert!(
                (lower * metric_scale * scale..upper * metric_scale * scale).contains(distance)
            );
        }
        if RowAddress::from(ids[row]).fragment_id() == 4 {
            let offset = RowAddress::from(ids[row]).row_offset();
            let expected_distance = (1.0 - (0.7 + offset as f32 * 0.0001)) * metric_scale * scale;
            assert!((distance - expected_distance).abs() < 1e-5);
        } else {
            assert!(
                range != Some((0.28, 0.32)),
                "indexed row passed appended-only bounds"
            );
        }
    }
}

#[tokio::test]
async fn test_quantized_refinement_scores_selective_prefilter_shortcut_rows() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    test_ds.make_rq_vector_index(4).await.unwrap();
    let query_vector = Float32Array::from_iter_values((0..32).map(|value| value as f32));

    let mut scanner = test_ds.dataset.scan();
    scanner
        .nearest("vec", &query_vector, 10)
        .unwrap()
        .minimum_nprobes(1)
        .maximum_nprobes(2)
        .prefilter(true)
        .filter("i = 0 OR i = 399")
        .unwrap()
        .quantized_refine(2)
        .distance_range(None, Some(f32::MAX));
    let results = scanner.try_into_batch().await.unwrap();

    assert_eq!(results.num_rows(), 2);
    assert!(
        results[DIST_COL]
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .all(|distance| distance.is_finite())
    );
}

#[tokio::test]
async fn test_quantized_refinement_rejects_non_rq_index() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    test_ds.make_vector_index().await.unwrap();
    let query_vector = Float32Array::from(vec![0.0; 32]);
    let mut scanner = test_ds.dataset.scan();
    scanner
        .nearest("vec", &query_vector, 5)
        .unwrap()
        .quantized_refine(2);
    let error = scanner.try_into_batch().await.unwrap_err();
    assert!(error.to_string().contains("requires an IVF_RQ flat index"));
}

#[tokio::test]
async fn test_quantized_refinement_applies_range_before_candidate_truncation() {
    const DIM: usize = 32;
    const ROWS: usize = 64;
    let mut state = 0x1234_5678_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f32 / (u32::MAX >> 8) as f32) * 2.0 - 1.0
    };
    let mut values = Vec::with_capacity(ROWS * DIM);
    for _ in 0..ROWS {
        let start = values.len();
        values.extend((0..DIM).map(|_| next()));
        let norm = values[start..]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        values[start..].iter_mut().for_each(|value| *value /= norm);
    }
    let vectors =
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32).unwrap();
    let batch = RecordBatch::try_from_iter([("vec", Arc::new(vectors) as ArrayRef)]).unwrap();
    let tmp = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        &tmp,
        None,
    )
    .await
    .unwrap();
    let mut rq_params = RQBuildParams::with_rotation_type(4, RQRotationType::Fast);
    let mut rotation = RabitQuantizer::new::<Float32Type>(4, DIM as i32)
        .metadata_ref()
        .clone();
    for (index, signs) in rotation
        .fast_rotation_signs
        .as_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *signs = (index * 37 + 19) as u8;
    }
    rq_params.rotation = Some(rotation);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            None,
            &VectorIndexParams::with_ivf_rq_params(
                DistanceType::L2,
                IvfBuildParams::new(1),
                rq_params,
            ),
            true,
        )
        .await
        .unwrap();

    let mut query_state = 0x9e37_79b9_u32;
    let mut query = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        query_state = query_state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        query.push(((query_state >> 8) as f32 / (u32::MAX >> 8) as f32) * 2.0 - 1.0);
    }
    let norm = query.iter().map(|value| value * value).sum::<f32>().sqrt();
    query.iter_mut().for_each(|value| *value /= norm);
    let query = Float32Array::from(query);

    let mut fast = dataset.scan();
    fast.nearest("vec", &query, 2)
        .unwrap()
        .nprobes(1)
        .approx_mode(ApproxMode::Fast)
        .fast_search()
        .with_row_id()
        .project::<&str>(&[])
        .unwrap();
    let fast = fast.try_into_batch().await.unwrap();
    let fast_ids = batch_row_ids(&fast);

    let mut accurate = dataset.scan();
    accurate
        .nearest("vec", &query, 2)
        .unwrap()
        .nprobes(1)
        .approx_mode(ApproxMode::Accurate)
        .fast_search()
        .with_row_id()
        .project::<&str>(&[])
        .unwrap();
    let accurate = accurate.try_into_batch().await.unwrap();
    let accurate_ids = batch_row_ids(&accurate);
    let accurate_distances = accurate[DIST_COL]
        .as_primitive::<Float32Type>()
        .values()
        .iter()
        .copied()
        .collect::<Vec<_>>();

    assert_eq!(fast_ids, vec![accurate_ids[1], accurate_ids[0]]);
    assert!(accurate_distances[0] < accurate_distances[1]);

    // The Fast winner is the farther Accurate candidate. With factor=1, a
    // post-truncation upper bound would discard it without ever scoring the
    // in-range candidate that the Fast pass ranked second.
    let upper_bound = (accurate_distances[0] + accurate_distances[1]) / 2.0;
    let mut accurate_range = dataset.scan();
    accurate_range
        .nearest("vec", &query, 1)
        .unwrap()
        .nprobes(1)
        .approx_mode(ApproxMode::Accurate)
        .distance_range(None, Some(upper_bound))
        .fast_search()
        .with_row_id()
        .project::<&str>(&[])
        .unwrap();
    let accurate_range = accurate_range.try_into_batch().await.unwrap();

    let mut quantized = dataset.scan();
    quantized
        .nearest("vec", &query, 1)
        .unwrap()
        .nprobes(1)
        .quantized_refine(1)
        .distance_range(None, Some(upper_bound))
        .fast_search()
        .with_row_id()
        .project::<&str>(&[])
        .unwrap();
    let quantized = quantized.try_into_batch().await.unwrap();

    assert_eq!(batch_row_ids(&quantized), batch_row_ids(&accurate_range));
    assert_eq!(quantized.num_rows(), 1);
    assert!(quantized[DIST_COL].as_primitive::<Float32Type>().value(0) < upper_bound);
}

/// A multivector query must score each row the same whatever the execution
/// batch size. Each query vector's top-k sort emits its output in batches of
/// that size, and one row id can appear in several of them because the index
/// stores one entry per sub-vector.
#[tokio::test]
async fn test_multivector_scores_do_not_depend_on_batch_size() {
    const DIM: i32 = 8;
    const SUB_VECTORS: usize = 8;
    const ROWS: usize = 200;
    let values = (0..ROWS * SUB_VECTORS * DIM as usize)
        .map(|i| ((i * 7919) % 1009) as f32 / 1009.0)
        .collect::<Vec<_>>();
    let sub_vectors =
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM).unwrap();
    let item = ArrowField::new("item", sub_vectors.data_type().clone(), true);
    let offsets = OffsetBuffer::from_lengths(vec![SUB_VECTORS; ROWS]);
    let vectors = ListArray::new(Arc::new(item), offsets, Arc::new(sub_vectors), None);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "vec",
        vectors.data_type().clone(),
        false,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors.clone())]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            None,
            &VectorIndexParams::ivf_flat(2, MetricType::Cosine),
            true,
        )
        .await
        .unwrap();

    let query = vectors.value(0);
    let search = |batch_size: Option<usize>| {
        let dataset = &dataset;
        let query = &query;
        async move {
            let mut scan = dataset.scan();
            scan.nearest("vec", query.as_ref(), 10)
                .unwrap()
                .minimum_nprobes(2)
                .with_row_id()
                .project::<&str>(&[])
                .unwrap();
            if let Some(batch_size) = batch_size {
                scan.batch_size(batch_size);
            }
            let batch = scan.try_into_batch().await.unwrap();
            batch_row_ids(&batch)
                .into_iter()
                .zip(
                    batch[DIST_COL]
                        .as_primitive::<Float32Type>()
                        .values()
                        .to_vec(),
                )
                .collect::<Vec<_>>()
        }
    };

    let expected = search(None).await;
    assert_eq!(expected.len(), 10);
    assert_eq!(search(Some(8)).await, expected);
}
