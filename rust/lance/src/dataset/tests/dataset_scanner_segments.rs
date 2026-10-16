// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use arrow::datatypes::Int32Type;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch};
use lance_arrow::FixedSizeListArrayExt;
use lance_datagen::{array, gen_batch};
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::DocumentGranularity;
use lance_index::scalar::inverted::query::PhraseQuery;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_linalg::distance::{DistanceType, MetricType};
use lance_table::format::Fragment;
use roaring::RoaringBitmap;
use rstest::rstest;
use uuid::Uuid;

use crate::dataset::Dataset;
use crate::dataset::scanner::Scanner;
use crate::dataset::scanner::test_dataset::TestVectorDataset;
use crate::index::DatasetIndexExt;
use crate::index::scalar::inverted::fts_index_fragment_bitmap;
use crate::index::vector::VectorIndexParams;
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
        scanner.prefilter(true);
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
async fn test_vector_prefilter_prunes_scalar_segments(
    #[values(false, true)] stable_row_ids: bool,
    #[values(false, true)] explicit_fragments: bool,
) {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, stable_row_ids)
        .await
        .unwrap();
    let vector_segments = test_ds.make_segmented_vector_index().await.unwrap();
    test_ds.make_segmented_scalar_index().await.unwrap();
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
    let scalar_index_node = analyzed
        .lines()
        .find(|line| line.trim_start().starts_with("ScalarIndexQuery:"))
        .unwrap_or_else(|| panic!("expected a scalar index query, plan was:\n{analyzed}"));
    assert!(
        scalar_index_node.contains("indices_loaded=1,"),
        "expected one scalar segment to be loaded, plan was:\n{analyzed}"
    );
    if explicit_fragments {
        assert!(
            analyzed.contains("projection=[], num_fragments=1"),
            "segment pruning must preserve the FilteredRead row-ID path, plan was:\n{analyzed}"
        );
    }

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

#[tokio::test]
async fn test_vector_search_respects_index_segments() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    let segment_ids = test_ds.make_segmented_vector_index().await.unwrap();

    let query: Float32Array = (0..32).map(|v| v as f32).collect();
    test_ds.append_data_with_range(400, 410).await.unwrap();
    test_ds.append_data_with_range(410, 420).await.unwrap();

    let mut scanner = test_ds.dataset.scan();
    scanner
        .nearest("vec", &query, 420)
        .unwrap()
        .with_index_segments(vec![segment_ids[0]])
        .unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    let i_array = batch
        .column_by_name("i")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(batch.num_rows(), 200);
    assert_values_in_range(
        i_array,
        0..200,
        "Should only get results from the selected index segment",
    );
}

#[tokio::test]
async fn test_vector_search_intersects_fragments_and_index_segments() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    let segment_ids = test_ds.make_segmented_vector_index().await.unwrap();

    let query: Float32Array = (0..32).map(|v| v as f32).collect();
    test_ds.append_data_with_range(400, 410).await.unwrap();
    test_ds.append_data_with_range(410, 420).await.unwrap();
    let fragments = test_ds.dataset.fragments();

    let mut scanner = test_ds.dataset.scan();
    scanner
        .nearest("vec", &query, 420)
        .unwrap()
        .with_fragments(vec![fragments[0].clone(), fragments[2].clone()])
        .with_index_segments(vec![segment_ids[0]])
        .unwrap();
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
        "Should get selected segment rows plus flat fallback for target fragments outside the selected segments"
    );
}

#[tokio::test]
async fn test_vector_search_rejects_unknown_index_segment() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    test_ds.make_segmented_vector_index().await.unwrap();

    let query: Float32Array = (0..32).map(|v| v as f32).collect();
    let err = test_ds
        .dataset
        .scan()
        .nearest("vec", &query, 10)
        .unwrap()
        .with_index_segments(vec![Uuid::new_v4()])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("unknown index segments"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn test_vector_search_rejects_metric_mismatch_for_index_segments() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    let segment_ids = test_ds.make_segmented_vector_index().await.unwrap();

    let query: Float32Array = (0..32).map(|v| v as f32).collect();
    let err = test_ds
        .dataset
        .scan()
        .nearest("vec", &query, 10)
        .unwrap()
        .distance_metric(DistanceType::Dot)
        .with_index_segments(vec![segment_ids[0]])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("with_index_segments requested metric"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn test_with_index_segments_rejects_empty_list() {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    let query: Float32Array = (0..32).map(|v| v as f32).collect();

    let Err(err) = test_ds
        .dataset
        .scan()
        .nearest("vec", &query, 10)
        .unwrap()
        .with_index_segments(vec![])
    else {
        panic!("expected empty index segments to be rejected");
    };
    assert!(
        err.to_string()
            .contains("with_index_segments does not accept an empty segment list"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn test_with_index_segments_rejected_for_non_vector_query() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    let segment_ids = test_ds.make_segmented_vector_index().await.unwrap();

    let err = test_ds
        .dataset
        .scan()
        .project(&["i"])
        .unwrap()
        .with_index_segments(vec![segment_ids[0]])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap_err();
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
