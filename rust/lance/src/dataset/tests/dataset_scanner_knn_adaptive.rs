// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Batch KNN with adaptive nprobes: the shared-scan path must match single-query search.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use arrow::datatypes::Int32Type;
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, BooleanArray, FixedSizeListArray, Float32Array, RecordBatch};
use lance_arrow::FixedSizeListArrayExt;
use lance_datafusion::exec::ExecutionSummaryCounts;
use lance_datafusion::utils::INDEX_SEGMENTS_SKIPPED_METRIC;
use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::ScalarIndexParams;
use lance_index::vector::DIST_COL;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_linalg::distance::MetricType;
use rstest::rstest;

use crate::dataset::Dataset;
use crate::dataset::scanner::Scanner;
use crate::dataset::scanner::test_dataset::assert_query_index_field;
use crate::index::DatasetIndexExt;
use crate::index::vector::VectorIndexParams;
use crate::io::exec::QUERY_INDEX_COL;
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

const ADAPTIVE_DIM: usize = 8;

const ADAPTIVE_QUERIES: usize = 6;

/// 2,000 random vectors over 4 fragments with a 16-partition index, enough
/// partitions for the adaptive budget and late search to diverge from a
/// fixed probe count. With `is_multi_delta`, 500 more rows are appended and
/// indexed as a second delta.
async fn adaptive_batch_dataset(
    index_type: &str,
    is_multi_delta: bool,
    metric: MetricType,
) -> Dataset {
    let mut dataset = gen_batch()
        .col("i", array::step::<Int32Type>())
        .col(
            "vec",
            array::rand_vec::<Float32Type>(Dimension::from(ADAPTIVE_DIM as u32)),
        )
        .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(500))
        .await
        .unwrap();
    let params = match index_type {
        "IVF_FLAT" => VectorIndexParams::ivf_flat(16, metric),
        "IVF_PQ" => VectorIndexParams::ivf_pq(16, 8, 4, metric, 10),
        "IVF_SQ" => VectorIndexParams::with_ivf_sq_params(
            metric,
            IvfBuildParams::new(16),
            SQBuildParams::default(),
        ),
        "IVF_RQ" => VectorIndexParams::ivf_rq(16, 8, metric),
        other => panic!("unsupported index type {other}"),
    };
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("idx".into()),
            &params,
            true,
        )
        .await
        .unwrap();
    if is_multi_delta {
        let extra = gen_batch()
            .col("i", array::step_custom::<Int32Type>(2000, 1))
            .col(
                "vec",
                array::rand_vec::<Float32Type>(Dimension::from(ADAPTIVE_DIM as u32)),
            )
            .into_reader_rows(RowCount::from(500), BatchCount::from(1));
        dataset.append(extra, None).await.unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        assert!(dataset.load_indices_by_name("idx").await.unwrap().len() >= 2);
    }
    dataset
}

/// The first `ADAPTIVE_QUERIES` stored vectors, shifted slightly so each
/// query is near (but not exactly on) an indexed row.
async fn adaptive_batch_queries(dataset: &Dataset) -> Vec<f32> {
    let batch = dataset
        .scan()
        .project(&["vec"])
        .unwrap()
        .limit(Some(ADAPTIVE_QUERIES as i64), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    batch["vec"]
        .as_fixed_size_list()
        .values()
        .as_primitive::<Float32Type>()
        .values()
        .iter()
        .map(|v| v + 0.01)
        .collect()
}

fn configure_adaptive_scan(
    scan: &mut Scanner,
    query: &dyn Array,
    k: usize,
    filter: Option<&str>,
    maximum_nprobes: Option<usize>,
    use_index: bool,
) {
    scan.nearest("vec", query, k).unwrap();
    scan.use_index(use_index);
    scan.minimum_nprobes(1);
    if let Some(maximum) = maximum_nprobes {
        scan.maximum_nprobes(maximum);
    }
    if let Some(filter) = filter {
        scan.filter(filter).unwrap();
        scan.prefilter(true);
    }
    scan.project(&["i"]).unwrap();
}

fn rows_for_query(batch: &RecordBatch, query_index: usize) -> RecordBatch {
    let mask = BooleanArray::from_iter(
        batch[QUERY_INDEX_COL]
            .as_primitive::<Int32Type>()
            .iter()
            .map(|value| value.map(|value| value == query_index as i32)),
    );
    arrow::compute::filter_record_batch(batch, &mask).unwrap()
}

/// Batch search with adaptive nprobes must return exactly what each query
/// returns on its own: same initial budget, same late-search stop point, and
/// the same "fewer than k rows match" shortcut. RQ preserves this through
/// the per-query fallback until its batch scorer has matching semantics.
#[rstest]
#[case::no_filter(None, None)]
#[case::late_search(Some("i % 97 = 0"), None)]
#[case::late_search_capped(Some("i % 97 = 0"), Some(3))]
#[case::maximum_exceeds_partitions(Some("i % 97 = 0"), Some(99))]
#[case::fewer_than_k_match(Some("i < 5"), None)]
#[case::no_matches(Some("i < 0"), None)]
#[tokio::test]
async fn test_batch_knn_adaptive_nprobes_matches_single(
    #[values("IVF_FLAT", "IVF_PQ", "IVF_SQ", "IVF_RQ")] index_type: &str,
    #[case] filter: Option<&str>,
    #[case] maximum_nprobes: Option<usize>,
) {
    let dataset = adaptive_batch_dataset(index_type, false, MetricType::L2).await;
    let query_values = adaptive_batch_queries(&dataset).await;
    let queries = FixedSizeListArray::try_new_from_values(
        Float32Array::from(query_values.clone()),
        ADAPTIVE_DIM as i32,
    )
    .unwrap();
    let k = 10;

    let mut scan = dataset.scan();
    configure_adaptive_scan(&mut scan, &queries, k, filter, maximum_nprobes, true);
    let plan = scan.explain_plan(false).await.unwrap();
    assert_eq!(
        plan.contains("ANNIvfBatch"),
        index_type != "IVF_RQ",
        "adaptive RQ must preserve the per-query path; other types share scans:\n{plan}"
    );
    let batch = scan.try_into_batch().await.unwrap();

    let mut hits = 0;
    let mut expected_total = 0;
    for query_index in 0..ADAPTIVE_QUERIES {
        let query = Float32Array::from(
            query_values[query_index * ADAPTIVE_DIM..(query_index + 1) * ADAPTIVE_DIM].to_vec(),
        );
        let mut single_scan = dataset.scan();
        configure_adaptive_scan(&mut single_scan, &query, k, filter, maximum_nprobes, true);
        let single = single_scan.try_into_batch().await.unwrap();
        let batch_rows = rows_for_query(&batch, query_index);
        assert_eq!(
            batch_rows["i"].as_primitive::<Int32Type>().values(),
            single["i"].as_primitive::<Int32Type>().values(),
            "{index_type} query {query_index}: batch rows must match single-query rows"
        );
        assert_eq!(
            batch_rows[DIST_COL].as_primitive::<Float32Type>().values(),
            single[DIST_COL].as_primitive::<Float32Type>().values(),
            "{index_type} query {query_index}: batch distances must match single-query"
        );

        let mut flat_scan = dataset.scan();
        configure_adaptive_scan(&mut flat_scan, &query, k, filter, None, false);
        let flat = flat_scan.try_into_batch().await.unwrap();
        let expected: HashSet<i32> = flat["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect();
        expected_total += expected.len();
        hits += batch_rows["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .filter(|i| expected.contains(i))
            .count();
    }
    // Capping maximum_nprobes limits recall by design; the equality above
    // is what matters there.
    if maximum_nprobes.is_none() && expected_total > 0 {
        let recall = hits as f32 / expected_total as f32;
        assert!(recall >= 0.5, "{index_type} recall {recall} below 0.5");
    }
    if filter == Some("i < 0") {
        assert_eq!(batch.num_rows(), 0);
    } else if filter == Some("i < 5") {
        // Only 5 rows can match, and the shortcut returns all of them.
        assert_eq!(batch.num_rows(), ADAPTIVE_QUERIES * 5);
    } else if maximum_nprobes.is_none() {
        // Late search keeps probing until every query has k rows.
        assert_eq!(batch.num_rows(), ADAPTIVE_QUERIES * k);
    }
    if index_type == "IVF_RQ" {
        let mut fixed_scan = dataset.scan();
        configure_adaptive_scan(&mut fixed_scan, &queries, k, filter, None, true);
        // The explicit minimum above takes precedence over `nprobes`, so pin
        // equal bounds to select fixed probing.
        fixed_scan.minimum_nprobes(16).maximum_nprobes(16);
        assert!(
            fixed_scan
                .explain_plan(false)
                .await
                .unwrap()
                .contains("ANNIvfBatch"),
            "fixed-probe RQ must retain its existing shared-scan path"
        );
        let fixed = fixed_scan.try_into_batch().await.unwrap();
        let rows_per_query = match filter {
            Some("i < 0") => 0,
            Some("i < 5") => 5,
            _ => k,
        };
        assert_eq!(fixed.num_rows(), ADAPTIVE_QUERIES * rows_per_query);
    }
}

/// Distance ranges are applied while scoring each partition, so with adaptive
/// probing a narrow range leaves queries short of `k` rows and sends them into
/// late search. The shared node must still return the single-query rows.
#[rstest]
#[tokio::test]
async fn test_batch_knn_adaptive_distance_range_matches_single(
    #[values("IVF_FLAT", "IVF_PQ")] index_type: &str,
    #[values((Some(0.1), Some(0.5)), (Some(0.3), Some(0.4)), (None, Some(0.2)))] range: (
        Option<f32>,
        Option<f32>,
    ),
    #[values(None, Some("i % 7 = 0"))] filter: Option<&str>,
) {
    let dataset = adaptive_batch_dataset(index_type, false, MetricType::L2).await;
    let query_values = adaptive_batch_queries(&dataset).await;
    let queries = FixedSizeListArray::try_new_from_values(
        Float32Array::from(query_values.clone()),
        ADAPTIVE_DIM as i32,
    )
    .unwrap();
    let k = 10;
    let (lower, upper) = range;

    let mut scan = dataset.scan();
    configure_adaptive_scan(&mut scan, &queries, k, filter, None, true);
    scan.distance_range(lower, upper);
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("ANNIvfBatch"), "got:\n{plan}");
    let batch = scan.try_into_batch().await.unwrap();
    // A range that excluded every row would make the comparison vacuous.
    assert!(batch.num_rows() > 0);

    for query_index in 0..ADAPTIVE_QUERIES {
        let query = Float32Array::from(
            query_values[query_index * ADAPTIVE_DIM..(query_index + 1) * ADAPTIVE_DIM].to_vec(),
        );
        let mut single_scan = dataset.scan();
        configure_adaptive_scan(&mut single_scan, &query, k, filter, None, true);
        single_scan.distance_range(lower, upper);
        let single = single_scan.try_into_batch().await.unwrap();
        let batch_rows = rows_for_query(&batch, query_index);
        assert_eq!(
            batch_rows["i"].as_primitive::<Int32Type>().values(),
            single["i"].as_primitive::<Int32Type>().values(),
            "{index_type} {range:?} {filter:?} query {query_index}: rows must match single-query"
        );
        assert_eq!(
            batch_rows[DIST_COL].as_primitive::<Float32Type>().values(),
            single[DIST_COL].as_primitive::<Float32Type>().values(),
            "{index_type} {range:?} {filter:?} query {query_index}: distances must match"
        );
    }
}

/// `search_effort` scales each query's initial budget inside
/// `AutoProbePolicy`, so the shared-scan batch node must return what
/// single-query search returns at every effort level.
#[rstest]
#[tokio::test]
async fn test_batch_knn_search_effort_matches_single(
    #[values("IVF_FLAT", "IVF_PQ", "IVF_SQ")] index_type: &str,
    #[values(0.0, 0.25, 0.75, 1.0)] effort: f64,
    #[values(None, Some("i % 97 = 0"))] filter: Option<&str>,
) {
    let dataset = adaptive_batch_dataset(index_type, false, MetricType::L2).await;
    let query_values = adaptive_batch_queries(&dataset).await;
    let queries = FixedSizeListArray::try_new_from_values(
        Float32Array::from(query_values.clone()),
        ADAPTIVE_DIM as i32,
    )
    .unwrap();
    let k = 10;

    let mut scan = dataset.scan();
    configure_adaptive_scan(&mut scan, &queries, k, filter, None, true);
    scan.search_effort(effort).unwrap();
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("ANNIvfBatch"), "got:\n{plan}");
    let batch = scan.try_into_batch().await.unwrap();

    for query_index in 0..ADAPTIVE_QUERIES {
        let query = Float32Array::from(
            query_values[query_index * ADAPTIVE_DIM..(query_index + 1) * ADAPTIVE_DIM].to_vec(),
        );
        let mut single_scan = dataset.scan();
        configure_adaptive_scan(&mut single_scan, &query, k, filter, None, true);
        single_scan.search_effort(effort).unwrap();
        let single = single_scan.try_into_batch().await.unwrap();
        let batch_rows = rows_for_query(&batch, query_index);
        assert_eq!(
            batch_rows["i"].as_primitive::<Int32Type>().values(),
            single["i"].as_primitive::<Int32Type>().values(),
            "{index_type} effort {effort} query {query_index}: rows must match single-query"
        );
        assert_eq!(
            batch_rows[DIST_COL].as_primitive::<Float32Type>().values(),
            single[DIST_COL].as_primitive::<Float32Type>().values(),
            "{index_type} effort {effort} query {query_index}: distances must match single-query"
        );
    }
}

#[rstest]
#[case::auto(0, true)]
#[case::sequential(1, true)]
#[case::parallel(4, false)]
#[case::all_cpus(-1, false)]
#[tokio::test]
async fn test_batch_knn_adaptive_query_parallelism(
    #[case] parallelism: i32,
    #[case] uses_batch: bool,
) {
    let dataset = adaptive_batch_dataset("IVF_FLAT", false, MetricType::L2).await;
    let query_values = adaptive_batch_queries(&dataset).await;
    let queries = FixedSizeListArray::try_new_from_values(
        Float32Array::from(query_values.clone()),
        ADAPTIVE_DIM as i32,
    )
    .unwrap();
    let filter = Some("i % 97 = 0");
    let mut scan = dataset.scan();
    configure_adaptive_scan(&mut scan, &queries, 10, filter, Some(3), true);
    scan.query_parallelism(parallelism);
    let plan = scan.explain_plan(false).await.unwrap();
    assert_eq!(plan.contains("ANNIvfBatch"), uses_batch, "got:\n{plan}");
    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    assert!(batch.num_rows() <= ADAPTIVE_QUERIES * 10);
    // Parallel late search may admit partitions past the sequential cut, so
    // only the shared sequential path is held to single-query equality.
    if !uses_batch {
        return;
    }
    for query_index in 0..ADAPTIVE_QUERIES {
        let query = Float32Array::from(
            query_values[query_index * ADAPTIVE_DIM..(query_index + 1) * ADAPTIVE_DIM].to_vec(),
        );
        let mut single_scan = dataset.scan();
        configure_adaptive_scan(&mut single_scan, &query, 10, filter, Some(3), true);
        single_scan.query_parallelism(parallelism);
        let single = single_scan.try_into_batch().await.unwrap();
        let batch_rows = rows_for_query(&batch, query_index);
        assert_eq!(
            batch_rows["i"].as_primitive::<Int32Type>().values(),
            single["i"].as_primitive::<Int32Type>().values(),
            "parallelism {parallelism} query {query_index}: rows must match single-query"
        );
        assert_eq!(
            batch_rows[DIST_COL].as_primitive::<Float32Type>().values(),
            single[DIST_COL].as_primitive::<Float32Type>().values(),
            "parallelism {parallelism} query {query_index}: distances must match single-query"
        );
    }
}

/// Across delta segments the late search shares each query's found count,
/// so every query still reaches k rows without duplicates.
#[rstest]
#[case::no_filter(None)]
#[case::late_search(Some("i % 97 = 0"))]
// Matches live only in the appended delta, so the late search must advance it
// alongside the first delta instead of exhausting the first delta beforehand.
#[case::late_search_newer_delta(Some("i >= 2000 AND i % 7 = 0"))]
#[tokio::test]
async fn test_batch_knn_adaptive_nprobes_multiple_deltas(#[case] filter: Option<&str>) {
    let dataset = adaptive_batch_dataset("IVF_FLAT", true, MetricType::L2).await;
    let query_values = adaptive_batch_queries(&dataset).await;
    let queries = FixedSizeListArray::try_new_from_values(
        Float32Array::from(query_values.clone()),
        ADAPTIVE_DIM as i32,
    )
    .unwrap();
    let k = 10;

    let mut scan = dataset.scan();
    configure_adaptive_scan(&mut scan, &queries, k, filter, None, true);
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("ANNIvfBatch"), "got:\n{plan}");
    let batch = scan.try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), ADAPTIVE_QUERIES * k);

    let mut hits = 0;
    for query_index in 0..ADAPTIVE_QUERIES {
        let batch_rows = rows_for_query(&batch, query_index);
        let ids: HashSet<i32> = batch_rows["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect();
        assert_eq!(ids.len(), k, "query {query_index} returned duplicate rows");

        let query = Float32Array::from(
            query_values[query_index * ADAPTIVE_DIM..(query_index + 1) * ADAPTIVE_DIM].to_vec(),
        );
        if filter == Some("i >= 2000 AND i % 7 = 0") {
            // Only one delta can contribute rows, so the result is deterministic.
            let mut single_scan = dataset.scan();
            configure_adaptive_scan(&mut single_scan, &query, k, filter, None, true);
            let single = single_scan.try_into_batch().await.unwrap();
            assert_eq!(
                batch_rows["i"].as_primitive::<Int32Type>().values(),
                single["i"].as_primitive::<Int32Type>().values(),
                "query {query_index}: batch rows must match single-query rows"
            );
        }

        let mut flat_scan = dataset.scan();
        configure_adaptive_scan(&mut flat_scan, &query, k, filter, None, false);
        let flat = flat_scan.try_into_batch().await.unwrap();
        hits += flat["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .filter(|i| ids.contains(i))
            .count();
    }
    let recall = hits as f32 / (ADAPTIVE_QUERIES * k) as f32;
    assert!(recall >= 0.5, "recall {recall} below 0.5");
}

#[rstest]
#[case::unfiltered(None)]
#[case::prefiltered(Some("i % 97 = 0"))]
#[tokio::test]
async fn test_batch_knn_adaptive_metrics(
    #[values(MetricType::Cosine, MetricType::Dot)] metric: MetricType,
    #[case] filter: Option<&str>,
) {
    let dataset = adaptive_batch_dataset("IVF_FLAT", false, metric).await;
    let query_values = adaptive_batch_queries(&dataset).await;
    let queries = FixedSizeListArray::try_new_from_values(
        Float32Array::from(query_values.clone()),
        ADAPTIVE_DIM as i32,
    )
    .unwrap();
    let mut scan = dataset.scan();
    configure_adaptive_scan(&mut scan, &queries, 10, filter, None, true);
    scan.distance_metric(metric);
    assert!(
        scan.explain_plan(false)
            .await
            .unwrap()
            .contains("ANNIvfBatch")
    );
    let batch = scan.try_into_batch().await.unwrap();
    let mut hits = 0;
    for query_index in 0..ADAPTIVE_QUERIES {
        let query = Float32Array::from(
            query_values[query_index * ADAPTIVE_DIM..(query_index + 1) * ADAPTIVE_DIM].to_vec(),
        );
        let mut single_scan = dataset.scan();
        configure_adaptive_scan(&mut single_scan, &query, 10, filter, None, true);
        single_scan.distance_metric(metric);
        let single = single_scan.try_into_batch().await.unwrap();
        let rows = rows_for_query(&batch, query_index);
        assert_eq!(
            rows["i"].as_primitive::<Int32Type>().values(),
            single["i"].as_primitive::<Int32Type>().values()
        );
        assert_eq!(
            rows[DIST_COL].as_primitive::<Float32Type>().values(),
            single[DIST_COL].as_primitive::<Float32Type>().values()
        );

        let mut flat_scan = dataset.scan();
        configure_adaptive_scan(&mut flat_scan, &query, 10, filter, None, false);
        flat_scan.distance_metric(metric);
        let flat = flat_scan.try_into_batch().await.unwrap();
        let expected: HashSet<_> = flat["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect();
        hits += rows["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .filter(|id| expected.contains(id))
            .count();
    }
    assert_eq!(batch.num_rows(), ADAPTIVE_QUERIES * 10);
    let recall = hits as f32 / (ADAPTIVE_QUERIES * 10) as f32;
    assert!(recall >= 0.5, "{metric:?} recall {recall} below 0.5");
}

/// The filter selects only rows of the appended delta, so the original delta's
/// own prefilter selects nothing. Like single-query search, the batch must skip
/// that delta's late search and count it in `segments_skipped`.
#[tokio::test]
async fn test_batch_knn_adaptive_skips_delta_whose_prefilter_selects_nothing() {
    let dataset = adaptive_batch_dataset("IVF_FLAT", true, MetricType::L2).await;
    let query_values = adaptive_batch_queries(&dataset).await;
    let queries = FixedSizeListArray::try_new_from_values(
        Float32Array::from(query_values.clone()),
        ADAPTIVE_DIM as i32,
    )
    .unwrap();
    let filter = Some("i >= 2000 AND i % 7 = 0");
    let segments_skipped = |summary: &Option<ExecutionSummaryCounts>| {
        summary
            .as_ref()
            .and_then(|summary| {
                summary
                    .all_counts
                    .get(INDEX_SEGMENTS_SKIPPED_METRIC)
                    .copied()
            })
            .unwrap_or_default()
    };

    let batch_stats = Arc::new(Mutex::new(None));
    let collected = batch_stats.clone();
    let mut scan = dataset.scan();
    configure_adaptive_scan(&mut scan, &queries, 10, filter, None, true);
    scan.scan_stats_callback(Arc::new(move |summary| {
        *collected.lock().unwrap() = Some(summary.clone());
    }));
    assert!(
        scan.explain_plan(false)
            .await
            .unwrap()
            .contains("ANNIvfBatch")
    );
    let batch = scan.try_into_batch().await.unwrap();

    let single_stats = Arc::new(Mutex::new(None));
    let collected = single_stats.clone();
    let query = Float32Array::from(query_values[..ADAPTIVE_DIM].to_vec());
    let mut single_scan = dataset.scan();
    configure_adaptive_scan(&mut single_scan, &query, 10, filter, None, true);
    single_scan.scan_stats_callback(Arc::new(move |summary| {
        *collected.lock().unwrap() = Some(summary.clone());
    }));
    let single = single_scan.try_into_batch().await.unwrap();

    let batch_skipped = segments_skipped(&batch_stats.lock().unwrap());
    let single_skipped = segments_skipped(&single_stats.lock().unwrap());
    assert!(
        single_skipped > 0,
        "fixture must leave a delta the filter cannot reach"
    );
    assert_eq!(batch_skipped, single_skipped);
    let rows = rows_for_query(&batch, 0);
    assert_eq!(
        rows["i"].as_primitive::<Int32Type>().values(),
        single["i"].as_primitive::<Int32Type>().values()
    );
}

/// A scoped exact scalar predicate reaches the ANN scan as a hybrid prefilter
/// producer (`PrefilterUnionExec`): the scalar index masks the fragments it
/// covers and a row-ID read supplies the rest. The shared adaptive batch node
/// must consume that producer instead of falling back to per-query search, and
/// the mask must stay exact through the shared scan.
#[rstest]
#[case::explicit_fragments(true)]
#[case::partial_scalar_coverage(false)]
#[tokio::test]
async fn test_batch_knn_adaptive_hybrid_scalar_prefilter(#[case] explicit_fragments: bool) {
    // The predicate spans both ends of the middle fragments so the scan scope
    // and the predicate each exclude rows the other would admit.
    const FILTER: &str = "i >= 250 AND i < 1750";
    const SCOPE: std::ops::Range<i32> = 500..1500;
    let mut dataset = adaptive_batch_dataset("IVF_FLAT", false, MetricType::L2).await;
    let mut scalar_fragments = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect::<Vec<_>>();
    if !explicit_fragments {
        // The exact predicate then has no index coverage for the last two
        // fragments, so the union must read their row IDs through its second
        // child instead of dropping those fragments from the scan.
        scalar_fragments.truncate(2);
    }
    let segment = dataset
        .create_index_builder(&["i"], IndexType::BTree, &ScalarIndexParams::default())
        .name("i_idx".to_owned())
        .fragments(scalar_fragments)
        .execute_uncommitted()
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments("i_idx", "i", vec![segment])
        .await
        .unwrap();

    let query_values = adaptive_batch_queries(&dataset).await;
    let queries = FixedSizeListArray::try_new_from_values(
        Float32Array::from(query_values.clone()),
        ADAPTIVE_DIM as i32,
    )
    .unwrap();
    let k = 10;
    // Only the explicit case scopes the scan; a partial scalar index already
    // produces the union through its missing-fragment child.
    let selected = explicit_fragments.then(|| dataset.fragments()[1..3].to_vec());

    let mut scan = dataset.scan();
    configure_adaptive_scan(&mut scan, &queries, k, Some(FILTER), None, true);
    if let Some(fragments) = &selected {
        scan.with_fragments(fragments.clone());
    }
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("ANNIvfBatch:"), "got:\n{plan}");
    assert!(
        !plan.contains("ANNSubIndex"),
        "the adaptive batch must not fall back to per-query search:\n{plan}"
    );
    // The union shape is what distinguishes the two producer paths: the
    // explicit case masks only the two selected fragments, while partial
    // coverage also evaluates the two fragments the scalar index omits.
    let expected_union = if explicit_fragments {
        "PrefilterUnion: scalar_fragments=2, evaluated_fragments=2"
    } else {
        "PrefilterUnion: scalar_fragments=2, evaluated_fragments=4"
    };
    assert!(
        plan.contains(expected_union),
        "expected `{expected_union}` in:\n{plan}"
    );

    let batch = scan.try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), ADAPTIVE_QUERIES * k);

    let mut hits = 0;
    let mut expected_total = 0;
    for query_index in 0..ADAPTIVE_QUERIES {
        let query = Float32Array::from(
            query_values[query_index * ADAPTIVE_DIM..(query_index + 1) * ADAPTIVE_DIM].to_vec(),
        );
        let mut single_scan = dataset.scan();
        configure_adaptive_scan(&mut single_scan, &query, k, Some(FILTER), None, true);
        if let Some(fragments) = &selected {
            single_scan.with_fragments(fragments.clone());
        }
        let single = single_scan.try_into_batch().await.unwrap();
        let rows = rows_for_query(&batch, query_index);
        assert_eq!(
            rows["i"].as_primitive::<Int32Type>().values(),
            single["i"].as_primitive::<Int32Type>().values(),
            "query {query_index}: batch rows must match single-query rows"
        );
        assert_eq!(
            rows[DIST_COL].as_primitive::<Float32Type>().values(),
            single[DIST_COL].as_primitive::<Float32Type>().values(),
            "query {query_index}: batch distances must match single-query"
        );
        // Eligibility comes from the mask the producer built, so validate it on
        // the rows the shared scan returned rather than trusting the plan.
        for id in rows["i"].as_primitive::<Int32Type>().values() {
            assert!(
                (250..1750).contains(id),
                "query {query_index}: row {id} violates `{FILTER}`"
            );
            if explicit_fragments {
                assert!(
                    SCOPE.contains(id),
                    "query {query_index}: row {id} is outside the selected fragments"
                );
            }
        }

        let mut flat_scan = dataset.scan();
        configure_adaptive_scan(&mut flat_scan, &query, k, Some(FILTER), None, false);
        if let Some(fragments) = &selected {
            flat_scan.with_fragments(fragments.clone());
        }
        let flat = flat_scan.try_into_batch().await.unwrap();
        let expected: HashSet<i32> = flat["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect();
        expected_total += expected.len();
        hits += rows["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .filter(|id| expected.contains(id))
            .count();
    }
    let recall = hits as f32 / expected_total as f32;
    assert!(recall >= 0.5, "recall {recall} below 0.5");
}
