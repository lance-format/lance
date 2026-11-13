// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use arrow::datatypes::Int32Type;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{Array, BooleanArray, FixedSizeListArray, Float32Array, Int32Array, RecordBatch};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{ROW_ADDR, ROW_ID};
use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::DocumentGranularity;
use lance_index::scalar::inverted::query::PhraseQuery;
use lance_index::vector::DIST_COL;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_io::object_store::ObjectStoreParams;
use lance_io::utils::tracking_store::IOTracker;
use lance_linalg::distance::{DistanceType, MetricType};
use lance_table::format::Fragment;
use roaring::RoaringBitmap;
use rstest::rstest;
use uuid::Uuid;

use crate::Error;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::scanner::test_dataset::{
    TestVectorDataset, assert_query_index_field, batch_knn_two_queries,
};
use crate::dataset::scanner::{RowAddrMask, RowAddrTreeMap, Scanner};
use crate::dataset::{Dataset, WriteParams};
use crate::index::DatasetIndexExt;
use crate::index::scalar::inverted::fts_index_fragment_bitmap;
use crate::index::vector::VectorIndexParams;
use crate::io::exec::QUERY_INDEX_COL;
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

#[rstest]
#[tokio::test]
async fn test_segment_prefilter_execution(
    #[values(false, true)] segmented: bool,
    #[values(false, true)] stable_row_ids: bool,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] has_unindexed: bool,
) {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, stable_row_ids)
        .await
        .unwrap();
    let segments = if segmented {
        test_ds.make_segmented_vector_index().await.unwrap()
    } else {
        test_ds.make_vector_index().await.unwrap();
        vec![test_ds.dataset.load_indices().await.unwrap()[0].uuid]
    };
    let indices = test_ds.dataset.load_indices().await.unwrap();
    let selected_segment = indices
        .iter()
        .find(|index| index.uuid == segments[0])
        .unwrap();
    // The unsegmented index spans both fragments; selecting only fragment 0 must
    // retain the prefilter even without a predicate or unindexed fallback.
    let expected_coverage: RoaringBitmap = if segmented { vec![0] } else { vec![0, 1] }
        .into_iter()
        .collect();
    assert_eq!(
        selected_segment.fragment_bitmap.as_ref(),
        Some(&expected_coverage)
    );
    test_ds.dataset.delete("i = 7").await.unwrap();
    if has_unindexed {
        test_ds.append_new_data().await.unwrap();
    }
    let fragments = test_ds.dataset.manifest.fragments.clone();
    let mut scope = vec![fragments[0].clone()];
    if has_unindexed {
        scope.push(fragments[2].clone());
    }
    let stats = Arc::new(Mutex::new(None));
    let collected = stats.clone();
    let mut scanner = test_ds.dataset.scan();
    scanner
        .prefilter(true)
        .with_fragments(scope)
        .with_index_segments(vec![segments[0]])
        .unwrap()
        .scan_stats_callback(Arc::new(move |summary| {
            *collected.lock().unwrap() = Some(summary.clone());
        }));
    scanner
        .nearest("vec", &Float32Array::from(vec![0.0; 32]), 500)
        .unwrap();
    scanner.nprobes(2);
    if filtered {
        scanner.filter("i >= 100").unwrap();
    }
    let batch = scanner.try_into_batch().await.unwrap();
    let actual: BTreeSet<i32> = batch["i"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect();
    let expected: BTreeSet<i32> = (0..200)
        .chain((400..410).filter(|_| has_unindexed))
        .filter(|i| *i != 7 && (!filtered || *i >= 100))
        .collect();
    assert_eq!(actual, expected);
    assert_eq!(batch.num_rows(), expected.len());
    let summary = stats.lock().unwrap().take().unwrap();
    let loads = summary
        .all_counts
        .get("prefilter_loads")
        .copied()
        .unwrap_or(0);
    assert_eq!(loads, if filtered || !segmented { 1 } else { 0 });
    for name in [
        "ANNSubIndexExec_elapsed_compute",
        "index_open_time",
        "index_partition_load_time",
        "index_partition_prepare_time",
        "index_cpu_queue_wait_time",
        "index_search_time",
        "index_query_prepare_time",
        "index_distance_topk_time",
        "index_result_materialize_time",
    ] {
        assert!(
            summary.all_times.get(name).is_some_and(|time| *time > 0),
            "missing ANN stage timing: {name}: {:?}",
            summary.all_times
        );
    }
}

#[rstest]
#[tokio::test]
async fn test_full_snapshot_prefilter_preserves_deleted_and_unindexed_rows(
    #[values(false, true)] stable_row_ids: bool,
    #[values(false, true)] pin_segment: bool,
    #[values(false, true)] prefilter: bool,
) {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, stable_row_ids)
        .await
        .unwrap();
    test_ds.make_vector_index().await.unwrap();
    let indexed_snapshot = test_ds.dataset.clone();
    test_ds.dataset.delete("i = 7").await.unwrap();
    test_ds.append_new_data().await.unwrap();
    for dataset in [&indexed_snapshot, &test_ds.dataset] {
        let mut scanner = dataset.scan();
        scanner.prefilter(prefilter);
        scanner.with_fragments(dataset.manifest.fragments.as_ref().clone());
        scanner
            .nearest("vec", &Float32Array::from(vec![0.0; 32]), 500)
            .unwrap();
        scanner.nprobes(2);
        if pin_segment {
            let indices = dataset.load_indices().await.unwrap();
            scanner.with_index_segments(vec![indices[0].uuid]).unwrap();
        }
        let batch = scanner.try_into_batch().await.unwrap();
        let actual: BTreeSet<i32> = batch["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect();
        let expected: BTreeSet<i32> =
            if dataset.version().version == indexed_snapshot.version().version {
                (0..400).collect()
            } else {
                (0..410).filter(|i| *i != 7).collect()
            };
        assert_eq!(actual, expected);
        assert_eq!(batch.num_rows(), expected.len());
    }
}

#[rstest]
#[tokio::test]
async fn test_full_snapshot_prefilter_4bit_pq_recall(
    #[values(MetricType::L2, MetricType::Dot)] metric: MetricType,
    #[values(false, true)] postfilter: bool,
) {
    const DIM: usize = 8;
    const K: usize = 10;
    let mut dataset = gen_batch()
        .with_seed(lance_datagen::Seed(42))
        .col("id", array::step::<UInt64Type>())
        .col("vec", array::rand_vec::<Float32Type>((DIM as u32).into()))
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(256))
        .await
        .unwrap();
    let centroids = Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; DIM]), DIM as i32)
            .unwrap(),
    );
    let codebook = Arc::new(Float32Array::from_iter_values(
        (0..DIM).flat_map(|_| (0..16).map(|i| i as f32 / 15.0)),
    ));
    let params = VectorIndexParams::with_ivf_pq_params(
        metric,
        IvfBuildParams::try_with_centroids(1, centroids).unwrap(),
        PQBuildParams::with_codebook(DIM, 4, codebook),
    );
    dataset
        .create_index(&["vec"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    let data = dataset.scan().try_into_batch().await.unwrap();
    let vectors = data["vec"].as_fixed_size_list();

    // One 512-row partition and k=10 leave most rows to the bulk scan's
    // quantized screening; a large refinement budget would hide rows it
    // wrongly drops.
    for query_row in [240, 360, 480] {
        let query = vectors.value(query_row);
        let make_scanner = |use_index| {
            let mut scanner = dataset.scan();
            scanner.project(&["id"]).unwrap();
            scanner
                .nearest("vec", query.as_ref(), K)
                .unwrap()
                .distance_metric(metric)
                .nprobes(1)
                .use_index(use_index);
            if postfilter {
                // Postfilter predicates are not passed to the ANN prefilter.
                scanner.filter("id >= 16").unwrap();
            } else {
                scanner.prefilter(true);
            }
            scanner
        };
        let expected = make_scanner(false).try_into_batch().await.unwrap();
        let implicit = make_scanner(true).try_into_batch().await.unwrap();
        let mut scoped = make_scanner(true);
        scoped.with_fragments(dataset.manifest.fragments.as_ref().clone());
        let plan = scoped.explain_plan(false).await.unwrap();
        assert!(plan.contains("ANNSubIndex"), "{plan}");
        let actual = scoped.try_into_batch().await.unwrap();
        assert_eq!(actual, implicit);
        let ids = |batch: &RecordBatch| {
            batch["id"]
                .as_primitive::<UInt64Type>()
                .values()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
        };
        let expected = ids(&expected);
        assert!(!expected.is_empty());
        let actual = ids(&actual);
        let recall = actual.intersection(&expected).count() as f64 / expected.len() as f64;
        assert!(
            recall >= 0.5,
            "{metric:?}, postfilter={postfilter}, query={query_row}: {recall}"
        );
    }
}

#[rstest]
#[case::single("single")]
#[case::batch("batch")]
#[case::fts("fts")]
#[tokio::test]
async fn test_full_snapshot_prefilter_execution_metrics(
    #[values(false, true)] filtered: bool,
    #[case] search_type: &str,
) {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    if search_type == "fts" {
        test_ds.make_fts_index().await.unwrap();
    } else {
        test_ds.make_vector_index().await.unwrap();
    }
    let stats = Arc::new(Mutex::new(None));
    let collected = stats.clone();
    let mut scanner = test_ds.dataset.scan();
    scanner
        .prefilter(true)
        .with_fragments(test_ds.dataset.manifest.fragments.as_ref().clone())
        .scan_stats_callback(Arc::new(move |summary| {
            *collected.lock().unwrap() = Some(summary.clone());
        }));
    if search_type == "batch" {
        let queries =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 64]), 32).unwrap();
        scanner.nearest("vec", &queries, 100).unwrap();
    } else if search_type == "fts" {
        scanner
            .full_text_search(FullTextSearchQuery::new("s".to_owned()))
            .unwrap()
            .limit(Some(100), None)
            .unwrap();
    } else {
        scanner
            .nearest("vec", &Float32Array::from(vec![0.0; 32]), 100)
            .unwrap();
    }
    scanner.nprobes(2);
    if filtered {
        scanner.filter("i >= 200").unwrap();
    }
    if search_type == "batch" {
        assert!(
            scanner
                .explain_plan(false)
                .await
                .unwrap()
                .contains("ANNIvfBatch")
        );
    }
    let batch = scanner.try_into_batch().await.unwrap();
    assert_eq!(
        batch.num_rows(),
        if search_type == "batch" { 200 } else { 100 }
    );
    let summary = stats.lock().unwrap().take().unwrap();
    // The batch node must materialize the shared filter once, not once per query.
    for (name, expected) in [
        ("prefilter_loads", 1),
        ("prefilter_input_rows", 200),
        ("prefilter_row_ids", 200),
    ] {
        let value = summary.all_counts.get(name).copied().unwrap_or_default();
        assert_eq!(value, if filtered { expected } else { 0 }, "{name}");
    }
    let batches = summary
        .all_counts
        .get("prefilter_input_batches")
        .copied()
        .unwrap_or_default();
    assert_eq!(batches > 0, filtered);
    for name in [
        "prefilter_load_time",
        "prefilter_input_time",
        "prefilter_build_time",
    ] {
        let value = summary.all_times.get(name).copied().unwrap_or_default();
        assert_eq!(value > 0, filtered, "{name}: {value}");
    }
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

    // A complete explicit scope must preserve indexed hits and unindexed fallback.
    let mut scanner = build_scanner(&test_ds.dataset);
    scanner.with_fragments(fragments.to_vec());
    let scoped = scanner.try_into_batch().await.unwrap();
    let mut expected = i_array.values().to_vec();
    let mut actual = scoped["i"].as_primitive::<Int32Type>().values().to_vec();
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(actual, expected);

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

#[rstest]
#[tokio::test]
async fn test_hybrid_prefilter_consumers(
    #[values(false, true)] stable_row_ids: bool,
    #[values(false, true)] fast_search: bool,
    #[values("ann", "batch", "fts")] consumer: &str,
) {
    let mut fixture = TestVectorDataset::new(LanceFileVersion::Stable, stable_row_ids)
        .await
        .unwrap();
    fixture.make_scalar_index().await.unwrap();
    fixture.append_data_with_range(400, 410).await.unwrap();
    fixture
        .dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            None,
            &VectorIndexParams::ivf_flat(1, MetricType::L2),
            true,
        )
        .await
        .unwrap();
    fixture.make_fts_index().await.unwrap();
    let mut before_delete = fixture.dataset.scan();
    before_delete.use_scalar_index(false).with_row_id();
    before_delete.filter("i >= 250 AND i % 2 = 0").unwrap();
    before_delete.with_fragments(fixture.dataset.fragments()[1..].to_vec());
    let before_delete = before_delete.try_into_batch().await.unwrap();
    let external = lance_select::RowAddrMask::from_allowed(
        before_delete[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect(),
    );
    fixture.dataset.delete("i = 252").await.unwrap();
    let dataset = &fixture.dataset;
    let mut oracle = dataset.scan();
    oracle.use_scalar_index(false).with_row_id();
    oracle
        .filter(if consumer == "batch" {
            "i >= 250"
        } else {
            "i >= 250 AND i % 2 = 0"
        })
        .unwrap();
    oracle.with_fragments(dataset.fragments()[1..].to_vec());
    let expected = oracle.try_into_batch().await.unwrap();
    let mut scanner = dataset.scan();
    scanner.with_fragments(dataset.fragments()[1..].to_vec());
    scanner.filter("i >= 250").unwrap().prefilter(true);
    if consumer != "batch" {
        scanner.with_row_id_prefilter(external);
    }
    scanner.project(&["i"]).unwrap();
    if fast_search {
        scanner.fast_search();
    }
    if consumer == "fts" {
        scanner
            .full_text_search(FullTextSearchQuery::new("s".to_owned()))
            .unwrap();
    } else if consumer == "batch" {
        scanner
            .nearest("vec", &batch_knn_two_queries().0, 200)
            .unwrap()
            .nprobes(1);
    } else {
        scanner
            .nearest("vec", &Float32Array::from(vec![0.0; 32]), 200)
            .unwrap()
            .nprobes(1);
    }
    let plan = scanner.explain_plan(true).await.unwrap();
    assert!(plan.contains("PrefilterUnion:"), "{plan}");
    if consumer == "batch" {
        assert!(plan.contains("ANNIvfBatch:"), "{plan}");
    }
    assert_eq!(
        plan.contains("projection=[], num_fragments=1"),
        !fast_search,
        "{plan}"
    );
    let actual = scanner.try_into_batch().await.unwrap();
    let ids = |batch: &RecordBatch| {
        batch["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
    };
    let expected: BTreeSet<_> = ids(&expected)
        .into_iter()
        .filter(|id| !fast_search || *id < 400)
        .collect();
    let actual_ids = ids(&actual);
    assert_eq!(actual_ids, expected);
    assert_eq!(
        actual.num_rows(),
        expected.len() * if consumer == "batch" { 2 } else { 1 }
    );
    assert_eq!(
        actual_ids.intersection(&expected).count() as f64 / expected.len() as f64,
        1.0
    );
}

#[rstest]
#[tokio::test]
async fn test_vector_prefilter_prunes_scalar_segments(
    #[values(false, true)] stable_row_ids: bool,
    #[values(false, true)] explicit_fragments: bool,
) {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, stable_row_ids)
        .await
        .unwrap();
    let vector_segments = test_ds.make_segmented_vector_index().await.unwrap();
    test_ds.make_segmented_scalar_index().await.unwrap();
    let excluded_segment = test_ds
        .dataset
        .load_indices_by_name("i_idx")
        .await
        .unwrap()
        .into_iter()
        .find(|segment| segment.fragment_bitmap.as_ref().unwrap().contains(0))
        .unwrap();
    let excluded_path = test_ds
        .dataset
        .indices_dir()
        .join(excluded_segment.uuid.to_string());
    let tracker = Arc::new(IOTracker::default());
    test_ds.dataset = crate::dataset::builder::DatasetBuilder::from_uri(test_ds.tmp_dir.as_str())
        .with_store_params(ObjectStoreParams {
            object_store_wrapper: Some(tracker.clone()),
            ..Default::default()
        })
        .load()
        .await
        .unwrap();
    tracker.incremental_stats();
    let fragments = test_ds.dataset.fragments();
    let query: Float32Array = (0..32).map(|value| value as f32).collect();

    let mut scanner = test_ds.dataset.scan();
    scanner.project(&["i"]).unwrap();
    scanner
        .nearest("vec", &query, 200)
        .unwrap()
        .nprobes(2)
        .prefilter(true)
        .with_index_segments(vec![vector_segments[1]])
        .unwrap();
    if explicit_fragments {
        scanner.with_fragments(vec![fragments[1].clone()]);
    }
    scanner.filter("NOT (i < 250)").unwrap();

    let analyzed = scanner.analyze_plan().await.unwrap();
    let stats = tracker.incremental_stats();
    assert!(
        !stats
            .requests
            .iter()
            .any(|request| request.path.as_ref().starts_with(excluded_path.as_ref())),
        "excluded scalar segment was opened: {:?}",
        stats.requests
    );
    let scalar_index_node = analyzed
        .lines()
        .find(|line| line.trim_start().starts_with("ScalarIndexQuery:"))
        .unwrap_or_else(|| panic!("expected a scalar index query, plan was:\n{analyzed}"));
    assert!(
        scalar_index_node.contains("indices_loaded=1,"),
        "expected one scalar segment to be loaded, plan was:\n{analyzed}"
    );
    assert_eq!(
        analyzed.contains("PrefilterUnion:"),
        explicit_fragments,
        "{analyzed}"
    );
    assert!(
        !analyzed.contains("projection=[], num_fragments=1"),
        "{analyzed}"
    );

    let actual = scanner.try_into_batch().await.unwrap();
    let actual = actual["i"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();

    let mut oracle = test_ds.dataset.scan();
    oracle.project(&["i"]).unwrap();
    oracle
        .nearest("vec", &query, 200)
        .unwrap()
        .nprobes(2)
        .prefilter(true)
        .use_index(false)
        .use_scalar_index(false)
        .with_fragments(vec![fragments[1].clone()]);
    oracle.filter("NOT (i < 250)").unwrap();
    let expected = oracle.try_into_batch().await.unwrap();
    let expected = expected["i"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();

    assert_eq!(actual, expected);
    assert!(!expected.is_empty());
    let recall = actual.intersection(&expected).count() as f64 / expected.len() as f64;
    assert_eq!(recall, 1.0);
}

#[rstest]
#[case::plain_all(false, false)]
#[case::plain_fragment(false, true)]
#[case::fts_fragment(true, true)]
#[tokio::test]
async fn test_filtered_read_prunes_scalar_segments(
    #[values(false, true)] stable_row_ids: bool,
    #[case] full_text_search: bool,
    #[case] explicit_fragments: bool,
) {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, stable_row_ids)
        .await
        .unwrap();
    test_ds.make_segmented_scalar_index().await.unwrap();
    if full_text_search {
        test_ds.make_fts_index().await.unwrap();
    }
    let fragments = test_ds.dataset.fragments();
    let mut scanner = test_ds.dataset.scan();
    scanner.project(&["i"]).unwrap();
    scanner.filter("NOT (i < 150)").unwrap();
    if explicit_fragments {
        scanner.with_fragments(vec![fragments[1].clone()]);
    }
    if full_text_search {
        // Every fixture row contains the token "s", so the scalar-only scan is an oracle.
        scanner
            .full_text_search(FullTextSearchQuery::new("s".to_owned()))
            .unwrap()
            .prefilter(true);
    }
    let analyzed = scanner.analyze_plan().await.unwrap();
    let scalar_index_node = analyzed
        .lines()
        .find(|line| line.trim_start().starts_with("ScalarIndexQuery:"))
        .unwrap_or_else(|| panic!("expected a scalar index query, plan was:\n{analyzed}"));
    let expected_segments = if explicit_fragments {
        1
    } else {
        fragments.len()
    };
    assert!(
        scalar_index_node.contains(&format!("indices_loaded={expected_segments},")),
        "unexpected scalar segment count, plan was:\n{analyzed}"
    );
    let actual = scanner.try_into_batch().await.unwrap();
    let actual = actual["i"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();

    let mut oracle = test_ds.dataset.scan();
    oracle.project(&["i"]).unwrap();
    oracle.filter("NOT (i < 150)").unwrap();
    oracle.use_scalar_index(false);
    if explicit_fragments {
        oracle.with_fragments(vec![fragments[1].clone()]);
    }
    let expected = oracle.try_into_batch().await.unwrap();
    let expected = expected["i"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    assert!(!expected.is_empty());
    assert_eq!(actual, expected);
}

#[rstest]
#[tokio::test]
async fn test_vector_search_respects_index_segments(
    #[values(false, true)] is_fast: bool,
    #[values(5, 80)] k: usize,
) {
    let fixture = FragmentVectorFixture::new(true, false).await;
    let mut scanner = fixture.dataset.scan();
    if is_fast {
        scanner.fast_search();
    }
    scanner
        .with_index_segments(vec![fixture.segments[0]])
        .unwrap()
        .nearest("vec", &fixture.query, k)
        .unwrap()
        .nprobes(1);
    fixture
        .assert_result(
            &mut scanner,
            &[0, 1],
            k,
            FragmentFilterPhase::None,
            FragmentSearchPath::Ann(1),
        )
        .await;
}

#[rstest]
#[tokio::test]
async fn test_vector_search_intersects_fragments_and_index_segments(
    #[values(false, true)] is_fast: bool,
    #[values(5, 80)] k: usize,
) {
    let fixture = FragmentVectorFixture::new(true, false).await;
    let mut scanner = fixture.dataset.scan();
    if is_fast {
        scanner.fast_search();
    }
    scanner
        .with_fragments(fixture.dataset.fragments()[1..].to_vec())
        .with_index_segments(vec![fixture.segments[0]])
        .unwrap()
        .nearest("vec", &fixture.query, k)
        .unwrap()
        .nprobes(1);
    let (expected_fragments, path) = if is_fast {
        (vec![1], FragmentSearchPath::Ann(1))
    } else {
        (vec![1, 2, 3], FragmentSearchPath::Mixed(1))
    };
    let ids = fixture
        .assert_result(
            &mut scanner,
            &expected_fragments,
            k,
            FragmentFilterPhase::None,
            path,
        )
        .await;
    if !is_fast && k == 5 {
        assert_eq!(ids, [16, 32, 48, 17, 33]);
    }
}

#[rstest]
#[case::whole_dataset(false)]
#[case::fragments_first(true)]
#[tokio::test]
async fn test_vector_search_rejects_unknown_index_segment(#[case] has_fragments: bool) {
    let fixture = FragmentVectorFixture::new(true, false).await;
    let mut scanner = fixture.dataset.scan();
    if has_fragments {
        scanner.with_fragments(fixture.dataset.fragments()[1..].to_vec());
    }
    let unknown_segment = Uuid::nil();
    let err = scanner
        .nearest("vec", &fixture.query, 10)
        .unwrap()
        .with_index_segments(vec![unknown_segment])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(
        err.to_string().contains("unknown index segments")
            && err.to_string().contains(&unknown_segment.to_string()),
        "unexpected error: {err}"
    );
}

#[rstest]
#[case::whole_dataset(false)]
#[case::fragments_first(true)]
#[tokio::test]
async fn test_vector_search_rejects_metric_mismatch_for_index_segments(
    #[case] has_fragments: bool,
) {
    let fixture = FragmentVectorFixture::new(true, false).await;
    let mut scanner = fixture.dataset.scan();
    if has_fragments {
        scanner.with_fragments(fixture.dataset.fragments()[1..].to_vec());
    }
    let err = scanner
        .nearest("vec", &fixture.query, 10)
        .unwrap()
        .distance_metric(DistanceType::Dot)
        .with_index_segments(vec![fixture.segments[0]])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(
        err.to_string().contains(
            "with_index_segments requested metric Dot but the selected index segments use L2"
        ),
        "unexpected error: {err}"
    );
}

#[rstest]
#[case::whole_dataset(false)]
#[case::fragments_first(true)]
#[tokio::test]
async fn test_with_index_segments_rejects_empty_list(#[case] has_fragments: bool) {
    let fixture = FragmentVectorFixture::new(false, false).await;
    let mut scanner = fixture.dataset.scan();
    if has_fragments {
        scanner.with_fragments(fixture.dataset.fragments()[1..].to_vec());
    }
    let Err(err) = scanner
        .nearest("vec", &fixture.query, 10)
        .unwrap()
        .with_index_segments(vec![])
    else {
        panic!("expected empty index segments to be rejected");
    };
    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(
        err.to_string()
            .contains("with_index_segments does not accept an empty segment list"),
        "unexpected error: {err}"
    );
}

#[rstest]
#[case::whole_dataset(false)]
#[case::fragments_first(true)]
#[tokio::test]
async fn test_with_index_segments_rejected_for_non_vector_query(#[case] has_fragments: bool) {
    let fixture = FragmentVectorFixture::new(true, false).await;
    let mut scanner = fixture.dataset.scan();
    if has_fragments {
        scanner.with_fragments(fixture.dataset.fragments()[1..].to_vec());
    }
    let err = scanner
        .project(&["id"])
        .unwrap()
        .with_index_segments(vec![fixture.segments[0]])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotSupported { .. }));
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

    // Create one FTS physical segment per indexed fragment.
    test_ds.make_segmented_fts_index().await.unwrap();
    let expected_index_coverage = test_ds
        .dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<RoaringBitmap>();
    let fragment_bitmap =
        fts_index_fragment_bitmap(&test_ds.dataset, "s", DocumentGranularity::Row)
            .await
            .unwrap()
            .expect("segmented FTS index");
    assert_eq!(fragment_bitmap, expected_index_coverage);

    // Append two more unindexed fragments
    test_ds.append_data_with_range(400, 410).await.unwrap();
    test_ds.append_data_with_range(410, 420).await.unwrap();

    // Fragment 0: i=0..200 (indexed), Fragment 1: i=200..400 (indexed)
    // Fragment 2: i=400..410 (unindexed), Fragment 3: i=410..420 (unindexed)
    let fragments = test_ds.dataset.fragments();
    assert_eq!(fragments.len(), 4);

    // "s-5" matches: s-5, s-50..s-59, s-150..s-159 (frag 0), s-250..s-259, s-350..s-359 (frag 1), s-405 (frag 2), s-415 (frag 3)
    async fn fts_ids(dataset: &Dataset, fragments: Option<Vec<Fragment>>) -> Vec<i32> {
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new("s-5".into()))
            .unwrap();
        if let Some(fragments) = fragments {
            scanner.with_fragments(fragments);
        }
        let batch = scanner.try_into_batch().await.unwrap();
        let mut ids = batch
            .column_by_name("i")
            .unwrap()
            .as_primitive::<Int32Type>()
            .values()
            .to_vec();
        ids.sort_unstable();
        ids
    }

    let global_ids = fts_ids(&test_ds.dataset, None).await;
    let mut fragmented_ids = Vec::with_capacity(global_ids.len());
    for (fragment, expected_range) in fragments.iter().zip([0..200, 200..400, 400..410, 410..420]) {
        let ids = fts_ids(&test_ds.dataset, Some(vec![fragment.clone()])).await;
        assert!(
            !ids.is_empty() && ids.iter().all(|id| expected_range.contains(id)),
            "fragment {} should return only matching rows in {expected_range:?}",
            fragment.id,
        );
        fragmented_ids.extend(ids);
    }
    fragmented_ids.sort_unstable();
    assert_eq!(fragmented_ids, global_ids);

    test_fragment_list_filtering(&test_ds, fragments, |dataset| {
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new("s-5".into()))
            .unwrap();
        scanner
    })
    .await;
    for (fragment, (phrase, expected_i)) in fragments[..2].iter().zip([("s 5", 5), ("s 205", 205)])
    {
        let mut scanner = test_ds.dataset.scan();
        scanner.with_fragments(vec![fragment.clone()]);
        scanner
            .full_text_search(FullTextSearchQuery::new_query(
                PhraseQuery::new(phrase.to_string())
                    .with_column(Some("s".to_string()))
                    .into(),
            ))
            .unwrap();
        let batch = scanner.try_into_batch().await.unwrap();
        let i_array = batch
            .column_by_name("i")
            .unwrap()
            .as_primitive::<Int32Type>();
        assert_eq!(i_array.values(), &[expected_i]);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FragmentFilterPhase {
    None,
    Pre,
    Post,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FragmentIndexMode {
    Indexed,
    Unindexed,
    Flat,
}

#[derive(Clone, Copy, Debug)]
enum FragmentSearchPath {
    Ann(usize),
    Mixed(usize),
    Flat,
    Empty,
}

pub(super) struct FragmentVectorFixture {
    pub(super) dataset: Dataset,
    segments: Vec<Uuid>,
    query: Float32Array,
}

#[rstest]
#[case::mixed_postfilter(false, false, 5)]
#[case::mixed_complete(false, false, 80)]
#[case::fast_postfilter(true, false, 5)]
#[case::removed_index_fragment(false, true, 80)]
#[tokio::test]
async fn test_vector_search_fragment_deletions(
    #[case] is_fast: bool,
    #[case] remove_fragment: bool,
    #[case] k: usize,
    #[values(false, true)] stable_row_ids: bool,
) {
    let mut fixture = FragmentVectorFixture::new(true, stable_row_ids).await;
    fixture.dataset.delete("id IN (16, 32, 48)").await.unwrap();
    if remove_fragment {
        fixture
            .dataset
            .delete("id >= 16 AND id < 32")
            .await
            .unwrap();
    }
    let fragments = fixture
        .dataset
        .fragments()
        .iter()
        .filter(|fragment| fragment.id > 0)
        .cloned()
        .collect();
    let mut scan = fixture.dataset.scan();
    scan.with_fragments(fragments);
    scan.with_index_segments(vec![fixture.segments[0]]).unwrap();
    scan.nearest("vec", &fixture.query, k).unwrap().nprobes(1);
    scan.filter("id % 2 = 1").unwrap().prefilter(false);
    if is_fast {
        scan.fast_search();
    }
    let expected = if is_fast { &[1][..] } else { &[1, 2, 3][..] };
    let path = if remove_fragment {
        FragmentSearchPath::Flat
    } else if is_fast {
        FragmentSearchPath::Ann(1)
    } else {
        FragmentSearchPath::Mixed(1)
    };
    fixture
        .assert_result(&mut scan, expected, k, FragmentFilterPhase::Post, path)
        .await;
    if remove_fragment {
        // Removing the final covered fragment retires the segment itself.
        fixture.dataset.delete("id < 16").await.unwrap();
        let mut retired = fixture.dataset.scan();
        retired.with_fragments(fixture.dataset.fragments().to_vec());
        retired
            .with_index_segments(vec![fixture.segments[0]])
            .unwrap();
        retired.nearest("vec", &fixture.query, k).unwrap();
        let error = retired.explain_plan(false).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("unknown index segments"),
            "{error}"
        );
        assert!(
            error.to_string().contains(&fixture.segments[0].to_string()),
            "{error}"
        );
    }
}

#[rstest]
#[case::allow("allow")]
#[case::block("block")]
#[case::intersection("intersection")]
#[case::empty("empty")]
#[tokio::test]
async fn test_vector_search_fragment_row_mask(
    #[case] mode: &str,
    #[values(false, true)] stable_row_ids: bool,
) {
    let fixture = FragmentVectorFixture::new(true, stable_row_ids).await;
    let source = fixture
        .dataset
        .scan()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let ids = source["id"].as_primitive::<Int32Type>();
    let row_ids = source[ROW_ID].as_primitive::<UInt64Type>();
    let allowed: RowAddrTreeMap = (0..source.num_rows())
        .filter(|row| ids.value(*row) % 3 != 1)
        .map(|row| row_ids.value(row))
        .collect();
    let blocked: RowAddrTreeMap = (0..source.num_rows())
        .filter(|row| ids.value(*row) % 4 == 0)
        .map(|row| row_ids.value(row))
        .collect();
    let mask = match mode {
        "allow" => RowAddrMask::from_allowed(allowed),
        "block" => RowAddrMask::from_block(blocked),
        "intersection" => {
            RowAddrMask::from_allowed(allowed).intersect(RowAddrMask::from_block(blocked))
        }
        "empty" => RowAddrMask::allow_nothing(),
        _ => unreachable!(),
    };
    // Define the oracle in business-ID space, independently of RowAddrMask.
    let eligible: RoaringBitmap = (0..64u32)
        .filter(|id| match mode {
            "allow" => id % 3 != 1,
            "block" => id % 4 != 0,
            "intersection" => id % 3 != 1 && id % 4 != 0,
            "empty" => false,
            _ => unreachable!(),
        })
        .collect();
    let mut scan = fixture.dataset.scan();
    scan.with_fragments(fixture.dataset.fragments()[1..].to_vec());
    scan.with_index_segments(vec![fixture.segments[0]]).unwrap();
    scan.with_row_id_prefilter(mask);
    scan.nearest("vec", &fixture.query, 5).unwrap().nprobes(1);
    scan.filter("id % 2 = 1").unwrap().prefilter(false);
    scan.project(&["id"]).unwrap();
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ANNSubIndex") && plan.contains("UnionExec"),
        "{plan}"
    );
    let expected = fixture
        .expected_result(&[1, 2, 3], 5, FragmentFilterPhase::Post, Some(&eligible))
        .await;
    let actual = scan.try_into_batch().await.unwrap();
    FragmentVectorFixture::assert_rows(&actual, &expected, &plan);
}

#[rstest]
#[case::postfilter(false, false, false, false, 5)]
#[case::prefilter(true, false, false, false, 5)]
#[case::complete(false, false, false, false, 80)]
#[case::fast(false, true, false, false, 5)]
#[case::fully_covered(false, false, true, false, 5)]
#[case::external_mask(false, false, false, true, 5)]
#[tokio::test]
async fn test_vector_search_fragment_batch(
    #[case] prefilter: bool,
    #[case] is_fast: bool,
    #[case] fully_covered: bool,
    #[case] has_mask: bool,
    #[case] k: usize,
) {
    let mut fixture = FragmentVectorFixture::new(true, false).await;
    let queries = FixedSizeListArray::try_new_from_values(
        Float32Array::from(vec![0.0, 0.0, 0.0, 0.0, 139.25, 0.0, 0.0, 0.0]),
        4,
    )
    .unwrap();
    let mut scan = fixture.dataset.scan();
    let fragments = if fully_covered {
        vec![fixture.dataset.fragments()[1].clone()]
    } else {
        fixture.dataset.fragments()[1..].to_vec()
    };
    scan.with_fragments(fragments);
    scan.with_index_segments(vec![fixture.segments[0]]).unwrap();
    scan.nearest("vec", &queries, k).unwrap().nprobes(1);
    scan.filter("id % 2 = 1").unwrap().prefilter(prefilter);
    scan.project(&["id"]).unwrap();
    if is_fast {
        scan.fast_search();
    }
    let eligible: RoaringBitmap = (0..64).filter(|id| id % 3 != 1).collect();
    if has_mask {
        let source = fixture
            .dataset
            .scan()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let allowed = (0..source.num_rows())
            .filter(|row| {
                eligible.contains(source["id"].as_primitive::<Int32Type>().value(*row) as u32)
            })
            .map(|row| source[ROW_ID].as_primitive::<UInt64Type>().value(row))
            .collect();
        scan.with_row_id_prefilter(RowAddrMask::from_allowed(allowed));
    }
    let plan = scan.explain_plan(false).await.unwrap();
    assert_eq!(
        plan.contains("ANNIvfBatch"),
        is_fast || fully_covered,
        "{plan}"
    );
    let batch = scan.try_into_batch().await.unwrap();
    assert_query_index_field(&batch);
    let query_indices = batch[QUERY_INDEX_COL].as_primitive::<Int32Type>();
    assert!(
        query_indices
            .values()
            .iter()
            .all(|index| (0..2).contains(index))
    );
    assert!(
        query_indices
            .values()
            .windows(2)
            .all(|pair| pair[0] <= pair[1])
    );
    for query_index in 0..2 {
        fixture.query = queries
            .value(query_index)
            .as_primitive::<Float32Type>()
            .clone();
        let mut single = scan.clone();
        single.nearest("vec", &fixture.query, k).unwrap().nprobes(1);
        let single = single.try_into_batch().await.unwrap();
        let selection = BooleanArray::from_iter(
            query_indices
                .values()
                .iter()
                .map(|index| Some(*index == query_index as i32)),
        );
        let actual = arrow::compute::filter_record_batch(&batch, &selection).unwrap();
        let phase = if prefilter {
            FragmentFilterPhase::Pre
        } else {
            FragmentFilterPhase::Post
        };
        let expected_fragments = if is_fast || fully_covered {
            &[1][..]
        } else {
            &[1, 2, 3][..]
        };
        let expected = fixture
            .expected_result(expected_fragments, k, phase, has_mask.then_some(&eligible))
            .await;
        FragmentVectorFixture::assert_rows(&actual, &expected, &plan);
        FragmentVectorFixture::assert_rows(&single, &expected, &plan);
        if !prefilter && k == 5 {
            assert!(actual.num_rows() < k);
        }
    }
}

#[rstest]
#[case::eager(false, false)]
#[case::eager_stable(true, false)]
#[case::deferred_stable_rejected(true, true)]
#[tokio::test]
async fn test_vector_search_fragment_compaction(
    #[case] stable_row_ids: bool,
    #[case] defer_index_remap: bool,
) {
    let mut fixture = FragmentVectorFixture::new(true, stable_row_ids).await;
    fixture.dataset.delete("id IN (16, 32, 48)").await.unwrap();
    let before = fixture
        .dataset
        .scan()
        .with_row_id()
        .with_row_address()
        .try_into_batch()
        .await
        .unwrap();
    let expected = fixture
        .expected_result(&[1, 2, 3], 5, FragmentFilterPhase::Post, None)
        .await;
    let version = fixture.dataset.version().version;
    let result = compact_files(
        &mut fixture.dataset,
        CompactionOptions {
            target_rows_per_fragment: 32,
            excluded_fragment_ids: vec![0],
            // Different index coverage prevents merging F1/F2; force rewrites after
            // the excluded prefix so the compaction cannot become a no-op.
            materialize_deletions_threshold: 0.0,
            defer_index_remap,
            ..Default::default()
        },
        None,
    )
    .await;
    if defer_index_remap {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("defer_index_remap"), "{error}");
        assert!(error.to_string().contains("idx"), "{error}");
        assert_eq!(fixture.dataset.version().version, version);
        return;
    }
    assert!(result.unwrap().fragments_removed > 0);
    let after = fixture
        .dataset
        .scan()
        .with_row_id()
        .with_row_address()
        .try_into_batch()
        .await
        .unwrap();
    for row in 0..after.num_rows() {
        let id = after["id"].as_primitive::<Int32Type>().value(row);
        let old_row = before["id"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .position(|v| *v == id)
            .unwrap();
        if stable_row_ids {
            assert_eq!(
                after[ROW_ID].as_primitive::<UInt64Type>().value(row),
                before[ROW_ID].as_primitive::<UInt64Type>().value(old_row)
            );
        }
        if (16..48).contains(&id) {
            assert_ne!(
                after[ROW_ADDR].as_primitive::<UInt64Type>().value(row),
                before[ROW_ADDR].as_primitive::<UInt64Type>().value(old_row)
            );
        }
    }
    let mut scan = fixture.dataset.scan();
    scan.with_fragments(
        fixture
            .dataset
            .fragments()
            .iter()
            .filter(|f| f.id != 0)
            .cloned()
            .collect(),
    );
    scan.nearest("vec", &fixture.query, 5).unwrap().nprobes(1);
    scan.filter("id % 2 = 1")
        .unwrap()
        .prefilter(false)
        .project(&["id"])
        .unwrap();
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("ANNSubIndex"), "{plan}");
    let actual = scan.try_into_batch().await.unwrap();
    FragmentVectorFixture::assert_rows(&actual, &expected, &plan);
}

#[rstest]
#[case::postfilter(FragmentFilterPhase::Post)]
#[case::prefilter(FragmentFilterPhase::Pre)]
#[tokio::test]
async fn test_vector_search_fragment_filter_phase(
    #[case] phase: FragmentFilterPhase,
    #[values(false, true)] is_fast: bool,
    #[values(5, 80)] k: usize,
) {
    let fixture = FragmentVectorFixture::new(true, false).await;
    let mut scanner = fixture.dataset.scan();
    scanner.filter("id % 2 = 1").unwrap();
    scanner.prefilter(phase == FragmentFilterPhase::Pre);
    if is_fast {
        scanner.fast_search();
    }
    scanner
        .with_fragments(fixture.dataset.fragments()[1..].to_vec())
        .with_index_segments(vec![fixture.segments[0]])
        .unwrap()
        .nearest("vec", &fixture.query, k)
        .unwrap()
        .nprobes(1);
    let (expected_fragments, path) = if is_fast {
        (vec![1], FragmentSearchPath::Ann(1))
    } else {
        (vec![1, 2, 3], FragmentSearchPath::Mixed(1))
    };
    let ids = fixture
        .assert_result(&mut scanner, &expected_fragments, k, phase, path)
        .await;
    if k == 5 {
        let expected = match (phase, is_fast) {
            (FragmentFilterPhase::Post, false) => vec![17, 33],
            (FragmentFilterPhase::Pre, false) => vec![17, 33, 49, 19, 35],
            (FragmentFilterPhase::Post, true) => vec![17, 19],
            (FragmentFilterPhase::Pre, true) => vec![17, 19, 21, 23, 25],
            (FragmentFilterPhase::None, _) => unreachable!("a SQL filter is always set"),
        };
        assert_eq!(ids, expected);
    }
}

#[rstest]
#[case::implicit_segments(FragmentIndexMode::Indexed, &[1, 2, 3], false, false, (&[1, 2, 3][..], FragmentSearchPath::Mixed(2)))]
#[case::implicit_segments_fast(FragmentIndexMode::Indexed, &[1, 2, 3], false, true, (&[1, 2][..], FragmentSearchPath::Ann(2)))]
#[case::no_index(FragmentIndexMode::Unindexed, &[1, 2, 3], false, false, (&[1, 2, 3][..], FragmentSearchPath::Flat))]
#[case::no_index_fast(FragmentIndexMode::Unindexed, &[1, 2, 3], false, true, (&[][..], FragmentSearchPath::Empty))]
#[case::index_disabled(FragmentIndexMode::Flat, &[1, 2, 3], true, false, (&[1, 2, 3][..], FragmentSearchPath::Flat))]
#[case::disjoint(FragmentIndexMode::Indexed, &[2, 3], true, false, (&[2, 3][..], FragmentSearchPath::Flat))]
#[case::disjoint_fast(FragmentIndexMode::Indexed, &[2, 3], true, true, (&[][..], FragmentSearchPath::Empty))]
#[case::within_segment(FragmentIndexMode::Indexed, &[1], true, false, (&[1][..], FragmentSearchPath::Ann(1)))]
#[case::empty(FragmentIndexMode::Indexed, &[], true, false, (&[][..], FragmentSearchPath::Empty))]
#[case::empty_fast(FragmentIndexMode::Indexed, &[], true, true, (&[][..], FragmentSearchPath::Empty))]
#[tokio::test]
async fn test_vector_search_fragment_scope_edges(
    #[case] mode: FragmentIndexMode,
    #[case] fragment_ids: &[usize],
    #[case] has_explicit_segments: bool,
    #[case] is_fast: bool,
    #[case] expected: (&[usize], FragmentSearchPath),
    #[values(5, 80)] k: usize,
) {
    let (expected_fragments, path) = expected;
    let fixture = FragmentVectorFixture::new(mode != FragmentIndexMode::Unindexed, false).await;
    let mut scanner = fixture.dataset.scan();
    scanner.with_fragments(
        fragment_ids
            .iter()
            .map(|id| fixture.dataset.fragments()[*id].clone())
            .collect(),
    );
    if has_explicit_segments {
        scanner
            .with_index_segments(vec![fixture.segments[0]])
            .unwrap();
    }
    scanner
        .nearest("vec", &fixture.query, k)
        .unwrap()
        .nprobes(1);
    if mode == FragmentIndexMode::Flat {
        scanner.use_index(false);
    }
    if is_fast {
        scanner.fast_search();
    }
    fixture
        .assert_result(
            &mut scanner,
            expected_fragments,
            k,
            FragmentFilterPhase::None,
            path,
        )
        .await;
}

#[rstest]
#[case::disjoint(false, false)]
#[case::disjoint_fast(false, true)]
#[case::empty(true, false)]
#[case::empty_fast(true, true)]
#[tokio::test]
async fn test_vector_search_disjoint_index_segments_metric(
    #[case] is_empty: bool,
    #[case] is_fast: bool,
) {
    let fixture = FragmentVectorFixture::new(true, false).await;
    let fragments = if is_empty {
        Vec::new()
    } else {
        fixture.dataset.fragments()[2..].to_vec()
    };
    let mut scanner = fixture.dataset.scan();
    scanner
        .with_fragments(fragments)
        .with_index_segments(vec![fixture.segments[0]])
        .unwrap()
        .nearest("vec", &fixture.query, 5)
        .unwrap()
        .distance_metric(DistanceType::Dot)
        .project(&["id"])
        .unwrap();
    if is_fast {
        scanner.fast_search();
    }
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(!plan.contains("ANNSubIndex"), "{plan}");
    let actual = scanner.try_into_batch().await.unwrap();
    let expected = if is_empty || is_fast {
        Vec::new()
    } else {
        assert!(plan.contains("KNNVectorDistance"), "{plan}");
        // The zero query ties all dot distances; row IDs break ties.
        (32..37).map(|id| (id, 1.0)).collect()
    };
    FragmentVectorFixture::assert_rows(&actual, &expected, &plan);
}

impl FragmentVectorFixture {
    #[cfg_attr(coverage, coverage(off))]
    pub(super) async fn new(is_indexed: bool, enable_stable_row_ids: bool) -> Self {
        // F0 is closer than every allowed row. F1/F2/F3 interleave by distance,
        // so even a small top-k must include both ANN and flat candidates.
        let mut values = Vec::with_capacity(64 * 4);
        for fragment in 0..4 {
            for row in 0..16 {
                let x = if fragment == 0 {
                    1 + row
                } else {
                    100 + 3 * row + fragment - 1
                };
                values.extend_from_slice(&[x as f32, 0.0, 0.0, 0.0]);
            }
        }
        let mut dataset = gen_batch()
            .col("id", array::step::<Int32Type>())
            .col(
                "vec",
                array::cycle_vec(
                    array::cycle::<Float32Type>(values[..48 * 4].to_vec()),
                    Dimension::from(4),
                ),
            )
            .into_ram_dataset_with_params(
                FragmentCount::from(3),
                FragmentRowCount::from(16),
                Some(WriteParams {
                    max_rows_per_file: 16,
                    data_storage_version: Some(LanceFileVersion::Stable),
                    enable_stable_row_ids,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        let mut segments = Vec::with_capacity(2);
        if is_indexed {
            // Supplied centroids avoid training randomness and minimum sample sizes.
            let centroids = Arc::new(
                FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 4]), 4)
                    .unwrap(),
            );
            let params = VectorIndexParams::with_ivf_flat_params(
                DistanceType::L2,
                IvfBuildParams::try_with_centroids(1, centroids).unwrap(),
            );
            let mut built_segments = Vec::with_capacity(2);
            for fragment_ids in [vec![0, 1], vec![2]] {
                let segment = dataset
                    .create_index_builder(&["vec"], IndexType::Vector, &params)
                    .name("idx".to_owned())
                    .fragments(fragment_ids)
                    .execute_uncommitted()
                    .await
                    .unwrap();
                segments.push(segment.uuid);
                built_segments.push(segment);
            }
            dataset
                .commit_existing_index_segments("idx", "vec", built_segments)
                .await
                .unwrap();
        }

        let appended = gen_batch()
            .col("id", array::cycle::<Int32Type>((48..64).collect()))
            .col(
                "vec",
                array::cycle_vec(
                    array::cycle::<Float32Type>(values[48 * 4..].to_vec()),
                    Dimension::from(4),
                ),
            )
            .into_reader_rows(RowCount::from(16), BatchCount::from(1));
        dataset.append(appended, None).await.unwrap();
        assert_eq!(dataset.fragments().len(), 4);
        for (id, fragment) in dataset.fragments().iter().enumerate() {
            assert_eq!(fragment.id, id as u64);
            assert_eq!(fragment.physical_rows, Some(16));
        }
        let metadata = dataset.load_indices().await.unwrap();
        assert_eq!(metadata.len(), segments.len());
        for (uuid, coverage) in segments.iter().zip([vec![0, 1], vec![2]]) {
            let segment = metadata.iter().find(|index| index.uuid == *uuid).unwrap();
            assert_eq!(
                segment.fragment_bitmap.as_ref(),
                Some(&coverage.into_iter().collect::<RoaringBitmap>())
            );
        }
        Self {
            dataset,
            segments,
            query: Float32Array::from(vec![0.0; 4]),
        }
    }

    #[cfg_attr(coverage, coverage(off))]
    async fn expected_result(
        &self,
        expected_fragments: &[usize],
        k: usize,
        phase: FragmentFilterPhase,
        allowed_ids: Option<&RoaringBitmap>,
    ) -> Vec<(i32, f64)> {
        // The oracle only scans stored vectors; it shares neither KNN distance
        // kernels nor index-coverage selection with the query under test.
        let mut oracle = self.dataset.scan();
        oracle.with_fragments(
            self.dataset
                .fragments()
                .iter()
                .filter(|fragment| expected_fragments.contains(&(fragment.id as usize)))
                .cloned()
                .collect(),
        );
        oracle.project(&["id", "vec"]).unwrap();
        let source = oracle.try_into_batch().await.unwrap();
        let ids = source["id"].as_primitive::<Int32Type>();
        let vectors = source["vec"].as_fixed_size_list();
        let mut expected = Vec::with_capacity(source.num_rows());
        for row in 0..source.num_rows() {
            if allowed_ids.is_some_and(|allowed| !allowed.contains(ids.value(row) as u32)) {
                continue;
            }
            let vector = vectors.value(row);
            let distance = vector
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .zip(self.query.values())
                .map(|(value, query)| (f64::from(*value) - f64::from(*query)).powi(2))
                .sum::<f64>();
            expected.push((ids.value(row), distance));
        }
        expected.sort_unstable_by(|left, right| left.1.total_cmp(&right.1));
        assert!(expected.windows(2).all(|pair| pair[0].1 < pair[1].1));
        if phase == FragmentFilterPhase::Pre {
            expected.retain(|(id, _)| id % 2 == 1);
        }
        expected.truncate(k);
        if phase == FragmentFilterPhase::Post {
            expected.retain(|(id, _)| id % 2 == 1);
        }
        expected
    }

    #[cfg_attr(coverage, coverage(off))]
    fn assert_rows(actual: &RecordBatch, expected: &[(i32, f64)], plan: &str) {
        let actual_ids = actual["id"].as_primitive::<Int32Type>();
        let distances = actual[DIST_COL].as_primitive::<Float32Type>();
        assert_eq!(actual_ids.null_count(), 0);
        assert_eq!(distances.null_count(), 0);
        let unique_ids: RoaringBitmap = actual_ids.values().iter().map(|id| *id as u32).collect();
        assert_eq!(unique_ids.len() as usize, actual.num_rows(), "{plan}");
        let expected_ids = expected.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        if !expected.is_empty() {
            let recalled = expected_ids
                .iter()
                .filter(|id| unique_ids.contains(**id as u32))
                .count();
            assert_eq!(recalled as f64 / expected.len() as f64, 1.0, "{plan}");
        }
        assert_eq!(actual.num_rows(), expected.len(), "{plan}");
        assert_eq!(
            actual_ids.values().as_ref(),
            expected_ids.as_slice(),
            "{plan}"
        );
        for (row, ((id, expected_distance), actual_distance)) in
            expected.iter().zip(distances.values()).enumerate()
        {
            let actual_distance = f64::from(*actual_distance);
            let tolerance = 1e-5 + 1e-6 * expected_distance.abs();
            assert!(
                actual_distance.is_finite()
                    && (actual_distance - expected_distance).abs() <= tolerance,
                "row={row}, id={id}: distance={actual_distance}, expected={expected_distance}, tolerance={tolerance}\n{plan}"
            );
        }
    }

    #[cfg_attr(coverage, coverage(off))]
    async fn assert_result(
        &self,
        scanner: &mut Scanner,
        expected_fragments: &[usize],
        k: usize,
        phase: FragmentFilterPhase,
        path: FragmentSearchPath,
    ) -> Vec<i32> {
        let expected = self
            .expected_result(expected_fragments, k, phase, None)
            .await;
        scanner.project(&["id"]).unwrap();
        let plan = scanner.explain_plan(false).await.unwrap();
        let actual = scanner.try_into_batch().await.unwrap();
        Self::assert_rows(&actual, &expected, &plan);
        match path {
            FragmentSearchPath::Ann(segments) | FragmentSearchPath::Mixed(segments) => {
                assert!(
                    plan.contains(&format!(
                        "ANNSubIndex: name=idx, k={k}, deltas={segments}, metric=L2"
                    )),
                    "{plan}"
                );
                let is_mixed = matches!(path, FragmentSearchPath::Mixed(_));
                assert_eq!(plan.contains("UnionExec"), is_mixed, "{plan}");
                assert_eq!(plan.contains("KNNVectorDistance"), is_mixed, "{plan}");
            }
            FragmentSearchPath::Flat => {
                assert!(!plan.contains("ANNSubIndex"), "{plan}");
                assert!(plan.contains("KNNVectorDistance"), "{plan}");
            }
            FragmentSearchPath::Empty => {
                assert_eq!(actual.num_rows(), 0);
                assert!(!plan.contains("ANNSubIndex"), "{plan}");
            }
        }
        actual["id"].as_primitive::<Int32Type>().values().to_vec()
    }
}

/// One supplied centroid: no IVF training, so no k-means randomness or minimum
/// sample size.
fn one_centroid() -> IvfBuildParams {
    let centroids = Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 4]), 4).unwrap(),
    );
    IvfBuildParams::try_with_centroids(1, centroids).unwrap()
}

/// Segments of index `idx` on `vec` over fragments 0 and 1, the first an L2
/// IVF_FLAT and the second built with `second`, published without the commit
/// check, as an earlier release could have left them.
async fn mixed_vector_segments(second: VectorIndexParams) -> (Dataset, Uuid, Uuid) {
    let mut dataset = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(4)))
        .into_ram_dataset_with_params(
            FragmentCount::from(2),
            FragmentRowCount::from(16),
            Some(WriteParams {
                max_rows_per_file: 16,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let first = VectorIndexParams::with_ivf_flat_params(DistanceType::L2, one_centroid());
    let mut segments = Vec::with_capacity(2);
    for (params, fragment) in [(first, 0), (second, 1)] {
        let segment = dataset
            .create_index_builder(&["vec"], IndexType::Vector, &params)
            .name("idx".to_owned())
            .fragments(vec![fragment])
            .execute_uncommitted()
            .await
            .unwrap();
        segments.push(segment);
    }
    let uuids = (segments[0].uuid, segments[1].uuid);
    crate::index::frag_reuse_reader::tests::persist_fixture(&mut dataset, segments).await;
    (dataset, uuids.0, uuids.1)
}

/// Segments of one index that rank by different metrics or index types cannot
/// answer one search: merging their distances would rank part of the data by
/// the wrong one.
#[rstest]
#[case::metric(
    VectorIndexParams::with_ivf_flat_params(DistanceType::Dot, one_centroid()),
    "Dot"
)]
#[case::index_type(
    VectorIndexParams::with_ivf_sq_params(
        DistanceType::L2,
        one_centroid(),
        SQBuildParams::default()
    ),
    "IVF_SQ"
)]
#[tokio::test]
async fn test_vector_search_refuses_segments_with_different_contracts(
    #[case] second_params: VectorIndexParams,
    #[case] mismatch: &str,
    #[values(false, true)] batch: bool,
) {
    let (dataset, first, second) = mixed_vector_segments(second_params).await;
    let query = Float32Array::from(vec![0.5; 4]);
    let queries =
        FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.5; 8]), 4).unwrap();

    let mut scanner = dataset.scan();
    if batch {
        scanner
            .nearest("vec", &queries, 5)
            .unwrap()
            .minimum_nprobes(1)
            .maximum_nprobes(1);
    } else {
        scanner.nearest("vec", &query, 5).unwrap();
    }
    let plan = scanner.explain_plan(false).await.unwrap();
    let node = if batch {
        "ANNIvfBatch"
    } else {
        "ANNIvfPartition"
    };
    assert!(plan.contains(node), "{plan}");
    let err = scanner.try_into_batch().await.unwrap_err();
    assert!(
        matches!(err, Error::Index { .. }),
        "expected an index error, got {err:?}"
    );
    let message = err.to_string();
    for expected in [first.to_string(), second.to_string(), mismatch.to_string()] {
        assert!(
            message.contains(&expected),
            "the error should name {expected}: {message}"
        );
    }

    // One segment alone ranks under one contract.
    let mut scanner = dataset.scan();
    scanner
        .nearest("vec", &query, 5)
        .unwrap()
        .with_index_segments(vec![first])
        .unwrap();
    assert_eq!(scanner.try_into_batch().await.unwrap().num_rows(), 5);
}
