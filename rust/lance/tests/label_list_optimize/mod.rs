// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exercise bounded LabelList maintenance through the public distributed API.

use std::sync::Arc;

use arrow_array::{
    Int32Array, RecordBatch, RecordBatchIterator,
    cast::AsArray,
    types::{Int32Type, Int64Type},
};
use datafusion::common::ScalarValue;
use lance::{
    Dataset, Error,
    dataset::{WriteParams, transaction::Operation},
    index::{
        DatasetIndexExt, DatasetIndexInternalExt,
        optimize::{
            IndexSegmentResult, OptimizeIndicesPlan, commit_optimize_indices, plan_optimize_indices,
        },
        scalar,
    },
};
use lance_core::utils::{address::RowAddress, tempfile::TempStrDir};
use lance_index::{
    IndexType,
    metrics::NoOpMetricsCollector,
    optimize::OptimizeOptions,
    scalar::{
        BuiltinIndexType, LabelListQuery, ScalarIndexParams, SearchResult,
        label_list::{LABEL_LIST_ROW_ADDR_DOMAIN_VERSION, LABEL_LIST_ROW_ID_DOMAIN_VERSION},
    },
};
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;

fn params(rows_per_fragment: usize) -> WriteParams {
    WriteParams {
        max_rows_per_file: rows_per_fragment,
        ..Default::default()
    }
}

async fn execute_plan(dataset: &Dataset, plan: &OptimizeIndicesPlan) -> Vec<IndexSegmentResult> {
    let mut built = Vec::new();
    for task in &plan.tasks {
        built.extend(task.execute(dataset).await.unwrap());
    }
    let mut merged = Vec::new();
    for merge in &plan.merges {
        merged.extend(merge.execute(dataset, &built).await.unwrap());
    }
    merged
}

async fn segments(dataset: &Dataset, name: &str) -> Vec<IndexMetadata> {
    dataset.load_indices_by_name(name).await.unwrap()
}

fn coverage(segments: &[IndexMetadata]) -> RoaringBitmap {
    segments
        .iter()
        .filter_map(|segment| segment.fragment_bitmap.clone())
        .fold(RoaringBitmap::new(), |acc, bitmap| acc | bitmap)
}

fn task_groups(plan: &OptimizeIndicesPlan) -> Vec<(u32, Vec<u32>)> {
    plan.tasks
        .iter()
        .map(|task| (task.group, task.fragments.clone()))
        .collect()
}

#[tokio::test]
async fn test_label_list_bounded_optimize_splits_tasks() {
    let labels = arrow_array::ListArray::from_iter_primitive::<arrow_array::types::Int64Type, _, _>(
        (0..8).map(|i| Some(vec![Some(i % 2)])),
    );
    let batch = RecordBatch::try_from_iter([("labels", Arc::new(labels) as arrow_array::ArrayRef)])
        .unwrap();
    let test_dir = TempStrDir::default();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(reader, &*test_dir, Some(params(8)))
        .await
        .unwrap();
    dataset
        .create_index(
            &["labels"],
            IndexType::LabelList,
            Some("labels_idx".to_string()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::LabelList),
            true,
        )
        .await
        .unwrap();
    for _ in 0..3 {
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        dataset.append(reader, Some(params(8))).await.unwrap();
    }

    let options = OptimizeOptions::append().max_rows_per_task(Some(8));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!(
        task_groups(&plan),
        vec![(0, vec![1]), (0, vec![2]), (0, vec![3])]
    );
}

fn label_list_optimize_batch(first_id: i32, rows: usize) -> RecordBatch {
    let labels = arrow_array::ListArray::from_iter_primitive::<Int64Type, _, _>(
        (first_id..first_id + rows as i32).map(|id| match id % 8 {
            0 => Some(vec![Some(1), Some(2)]),
            1 => None,
            2 => Some(vec![]),
            3 => Some(vec![Some(1), None]),
            4 => Some(vec![None]),
            5 => Some(vec![Some(2)]),
            6 => Some(vec![Some(1), Some(1)]),
            _ => Some(vec![Some(3)]),
        }),
    );
    RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(Int32Array::from_iter_values(
                first_id..first_id + rows as i32,
            )) as arrow_array::ArrayRef,
        ),
        ("labels", Arc::new(labels) as arrow_array::ArrayRef),
    ])
    .unwrap()
}

async fn label_list_optimize_ids(dataset: &Dataset, predicate: &str, use_index: bool) -> Vec<i32> {
    let mut scan = dataset.scan();
    scan.project(&["id"]).unwrap();
    scan.filter(predicate).unwrap();
    scan.use_scalar_index(use_index);
    let batch = scan.try_into_batch().await.unwrap();
    let mut ids = batch["id"].as_primitive::<Int32Type>().values().to_vec();
    ids.sort_unstable();
    ids
}

#[rstest::rstest]
#[tokio::test]
async fn test_label_list_split_round_trip_and_queries(#[values(false, true)] stable_row_ids: bool) {
    let test_dir = TempStrDir::default();
    let mut dataset = write_label_list_dataset(&test_dir, 8, stable_row_ids).await;
    let batch = label_list_optimize_batch(8, 24);
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    dataset.append(reader, Some(params(8))).await.unwrap();

    let options = OptimizeOptions::merge(1).max_rows_per_task(Some(8));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    let plan: OptimizeIndicesPlan =
        serde_json::from_str(&serde_json::to_string(&plan).unwrap()).unwrap();
    assert_eq!(
        task_groups(&plan),
        vec![(0, vec![1]), (0, vec![2]), (0, vec![3])],
    );
    assert_eq!(plan.merges.len(), 1);
    assert_eq!(plan.merges[0].old_segments.len(), 1);
    assert!(plan.tasks.iter().all(|task| task.old_segments.is_empty()));

    let mut built = Vec::new();
    for task in &plan.tasks {
        let result = task.execute(&dataset).await.unwrap().unwrap();
        let result: IndexSegmentResult =
            serde_json::from_str(&serde_json::to_string(&result).unwrap()).unwrap();
        built.push(result);
    }
    let duplicate = vec![built[0].clone(), built[0].clone()];
    let error = plan.merges[0]
        .execute(&dataset, &duplicate)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("duplicate segment uuid"));

    let retried = plan.tasks[0].execute(&dataset).await.unwrap().unwrap();
    let error = plan.merges[0]
        .execute(&dataset, &[built[0].clone(), retried])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("overlapping fragment coverage"));

    built.reverse();
    let merged = plan.merges[0]
        .execute(&dataset, &built)
        .await
        .unwrap()
        .unwrap();
    let merged: IndexSegmentResult =
        serde_json::from_str(&serde_json::to_string(&merged).unwrap()).unwrap();
    commit_optimize_indices(&mut dataset, vec![merged], &options)
        .await
        .unwrap();
    let actual = segments(&dataset, "labels_idx").await;
    assert_eq!(actual.len(), 1);
    assert_eq!(coverage(&actual), (0u32..4).collect::<RoaringBitmap>());
    for predicate in [
        "array_has_any(labels, [1])",
        "array_has_any(labels, [2, 3])",
        "array_has_all(labels, [1, 2])",
        "array_contains(labels, 1)",
        "array_contains(labels, 3)",
        "NOT array_has_any(labels, [1])",
        "array_has_any(labels, [])",
        "array_has_all(labels, [])",
        "array_contains(labels, NULL)",
    ] {
        assert_eq!(
            label_list_optimize_ids(&dataset, predicate, true).await,
            label_list_optimize_ids(&dataset, predicate, false).await,
            "{predicate}; stable_row_ids={stable_row_ids}",
        );
    }
    let mut scan = dataset.scan();
    scan.filter("array_has_any(labels, [1])").unwrap();
    let explain = scan.explain_plan(false).await.unwrap();
    assert!(explain.contains("ScalarIndexQuery"), "{explain}");
}

#[tokio::test]
async fn test_label_list_single_task_keeps_old_segments_in_merge() {
    let test_dir = TempStrDir::default();
    let mut dataset = write_label_list_dataset(&test_dir, 16, true).await;
    let batch = label_list_optimize_batch(16, 8);
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    dataset.append(reader, Some(params(8))).await.unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    assert_eq!(segments(&dataset, "labels_idx").await.len(), 2);
    dataset
        .delete("id < 8 OR id = 17 OR id = 19")
        .await
        .unwrap();
    let batch = label_list_optimize_batch(24, 8);
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    dataset.append(reader, Some(params(8))).await.unwrap();

    let options = OptimizeOptions::merge(2).max_rows_per_task(Some(8));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!(task_groups(&plan), vec![(0, vec![3])]);
    assert!(plan.tasks[0].old_segments.is_empty());
    assert_eq!(plan.tasks[0].num_indices_to_merge, Some(0));
    assert_eq!(plan.merges.len(), 1);
    assert_eq!(plan.merges[0].old_segments.len(), 2);
    let built = plan.tasks[0].execute(&dataset).await.unwrap().unwrap();
    assert_eq!(
        built.segment.fragment_bitmap,
        Some([3u32].into_iter().collect())
    );
    assert!(built.replaces.is_empty());
    let merged = plan.merges[0]
        .execute(&dataset, &[built])
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        merged.segment.fragment_bitmap,
        Some([1u32, 2, 3].into_iter().collect()),
    );

    let index =
        scalar::open_scalar_index(&dataset, "labels", &merged.segment, &NoOpMetricsCollector)
            .await
            .unwrap();
    let query = LabelListQuery::HasAnyLabel(vec![ScalarValue::Int64(Some(1))]);
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    let SearchResult::Exact(matches) = result else {
        panic!("expected exact LabelList result");
    };
    for row_set in [matches.true_rows(), matches.null_rows().clone()] {
        assert!(
            row_set
                .row_addrs()
                .unwrap()
                .all(|addr| addr.fragment_id() != 0),
            "retired fragment must disappear from value and list-null postings",
        );
    }
    drop(index);
    commit_optimize_indices(&mut dataset, vec![merged], &options)
        .await
        .unwrap();
    let actual = segments(&dataset, "labels_idx").await;
    assert_eq!(actual.len(), 1);
    assert_eq!(coverage(&actual), [1u32, 2, 3].into_iter().collect());
    for predicate in [
        "array_has_any(labels, [1])",
        "array_contains(labels, 1)",
        "NOT array_has_any(labels, [1])",
        "array_has_all(labels, [])",
        "labels IS NULL",
    ] {
        assert_eq!(
            label_list_optimize_ids(&dataset, predicate, true).await,
            label_list_optimize_ids(&dataset, predicate, false).await,
            "{predicate}",
        );
    }
    let ids = label_list_optimize_ids(&dataset, "array_contains(labels, 1)", true).await;
    assert!(!ids.contains(&19));
    assert!(ids.contains(&24));
}

#[tokio::test]
async fn test_label_list_split_merge_cleans_partial_deletions() {
    let test_dir = TempStrDir::default();
    let mut dataset = write_label_list_dataset(&test_dir, 8, true).await;
    dataset.delete("id = 1 OR id = 3").await.unwrap();
    let batch = label_list_optimize_batch(8, 8);
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    dataset.append(reader, Some(params(8))).await.unwrap();
    assert_eq!(**dataset.fragment_bitmap(), [0u32, 1].into_iter().collect());

    let options = OptimizeOptions::merge(1).max_rows_per_task(Some(8));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!(task_groups(&plan), vec![(0, vec![1])]);
    assert!(plan.tasks[0].old_segments.is_empty());
    assert_eq!(plan.merges.len(), 1);
    assert_eq!(plan.merges[0].old_segments.len(), 1);
    let built = plan.tasks[0].execute(&dataset).await.unwrap().unwrap();
    let merged = plan.merges[0]
        .execute(&dataset, &[built])
        .await
        .unwrap()
        .unwrap();
    let index =
        scalar::open_scalar_index(&dataset, "labels", &merged.segment, &NoOpMetricsCollector)
            .await
            .unwrap();
    let query = LabelListQuery::HasAnyLabel(vec![ScalarValue::Int64(Some(1))]);
    let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
    let SearchResult::Exact(matches) = result else {
        panic!("expected exact LabelList result");
    };
    let true_rows: Vec<_> = matches.true_rows().row_addrs().unwrap().collect();
    assert!(!true_rows.contains(&RowAddress::new_from_parts(0, 3)));
    assert!(true_rows.contains(&RowAddress::new_from_parts(0, 0)));
    assert!(true_rows.contains(&RowAddress::new_from_parts(1, 3)));
    let null_rows: Vec<_> = matches.null_rows().row_addrs().unwrap().collect();
    assert!(!null_rows.contains(&RowAddress::new_from_parts(0, 1)));
    assert!(null_rows.contains(&RowAddress::new_from_parts(1, 1)));
    drop(index);
    commit_optimize_indices(&mut dataset, vec![merged], &options)
        .await
        .unwrap();
    for predicate in [
        "array_has_any(labels, [1])",
        "NOT array_contains(labels, 1)",
    ] {
        assert_eq!(
            label_list_optimize_ids(&dataset, predicate, true).await,
            label_list_optimize_ids(&dataset, predicate, false).await,
            "{predicate}",
        );
    }
}

async fn write_label_list_dataset(uri: &str, rows: usize, stable_row_ids: bool) -> Dataset {
    let batch = label_list_optimize_batch(0, rows);
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            enable_stable_row_ids: stable_row_ids,
            ..params(8)
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["labels"],
            IndexType::LabelList,
            Some("labels_idx".into()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::LabelList),
            true,
        )
        .await
        .unwrap();
    dataset
}

#[rstest::rstest]
#[tokio::test]
async fn test_label_list_segment_bounds(#[values(None, Some(8))] task_bound: Option<u64>) {
    let test_dir = TempStrDir::default();
    let mut dataset = write_label_list_dataset(&test_dir, 8, false).await;
    let batch = label_list_optimize_batch(8, 32);
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    dataset.append(reader, Some(params(8))).await.unwrap();
    let options = OptimizeOptions::merge(1)
        .max_rows_per_task(task_bound)
        .max_rows_per_segment(Some(16));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert!(plan.tasks.len() > 1);
    let task_limit = task_bound.unwrap_or(16);
    assert!(
        plan.tasks
            .iter()
            .all(|task| task.fragments.len() as u64 * 8 <= task_limit)
    );
    let results = execute_plan(&dataset, &plan).await;
    commit_optimize_indices(&mut dataset, results, &options)
        .await
        .unwrap();
    let actual = segments(&dataset, "labels_idx").await;
    assert_eq!(coverage(&actual), (0u32..5).collect());
    assert!(
        actual
            .iter()
            .all(|segment| segment.fragment_bitmap.as_ref().unwrap().len() * 8 <= 16)
    );
    assert_eq!(
        label_list_optimize_ids(&dataset, "array_contains(labels, 1)", true).await,
        label_list_optimize_ids(&dataset, "array_contains(labels, 1)", false).await,
    );
}

#[tokio::test]
async fn test_label_list_merge_only_and_covered_noop() {
    let test_dir = TempStrDir::default();
    let mut dataset = write_label_list_dataset(&test_dir, 8, false).await;
    let batch = label_list_optimize_batch(8, 8);
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    dataset.append(reader, Some(params(8))).await.unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    assert_eq!(segments(&dataset, "labels_idx").await.len(), 2);
    let options = OptimizeOptions::merge(2).max_rows_per_task(Some(8));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert!(plan.tasks.is_empty());
    assert_eq!(plan.merges.len(), 1);
    assert_eq!(plan.merges[0].old_segments.len(), 2);
    let results = execute_plan(&dataset, &plan).await;
    commit_optimize_indices(&mut dataset, results, &options)
        .await
        .unwrap();
    assert_eq!(segments(&dataset, "labels_idx").await.len(), 1);
    for options in [OptimizeOptions::append(), OptimizeOptions::merge(1)] {
        let options = options.max_rows_per_task(Some(8));
        let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
        assert!(plan.tasks.is_empty());
        assert!(plan.merges.is_empty());
    }
    assert_eq!(
        label_list_optimize_ids(&dataset, "array_contains(labels, 1)", true).await,
        label_list_optimize_ids(&dataset, "array_contains(labels, 1)", false).await,
    );
}

#[rstest::rstest]
#[tokio::test]
async fn test_label_list_incompatible_domain_stays_unsplit(
    #[values(false, true)] stable_row_ids: bool,
) {
    let test_dir = TempStrDir::default();
    let mut dataset = write_label_list_dataset(&test_dir, 8, stable_row_ids).await;
    let old_segments = segments(&dataset, "labels_idx").await;
    let mut opposite_domain = old_segments[0].clone();
    // Both identifier domains coincide within the initial fragment, so the
    // same data is a valid fixture for the opposite domain declaration.
    opposite_domain.index_version = if stable_row_ids {
        LABEL_LIST_ROW_ID_DOMAIN_VERSION as i32
    } else {
        LABEL_LIST_ROW_ADDR_DOMAIN_VERSION as i32
    };
    let uri: &str = &test_dir;
    Dataset::commit(
        uri,
        Operation::CreateIndex {
            new_indices: vec![opposite_domain],
            removed_indices: old_segments,
        },
        Some(dataset.manifest.version),
        None,
        None,
        dataset.session(),
        false,
    )
    .await
    .unwrap();
    dataset = Dataset::open(&test_dir).await.unwrap();
    let batch = label_list_optimize_batch(8, 16);
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    dataset.append(reader, Some(params(8))).await.unwrap();
    let options = OptimizeOptions::merge(1).max_rows_per_task(Some(8));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!(task_groups(&plan), vec![(0, vec![1, 2])]);
    let results = execute_plan(&dataset, &plan).await;
    commit_optimize_indices(&mut dataset, results, &options)
        .await
        .unwrap();
    assert_eq!(
        label_list_optimize_ids(&dataset, "array_contains(labels, 1)", true).await,
        label_list_optimize_ids(&dataset, "array_contains(labels, 1)", false).await,
    );
}

#[tokio::test]
async fn test_label_list_legacy_sibling_keeps_current_reference_unsplit() {
    let test_dir = TempStrDir::default();
    let mut dataset = write_label_list_dataset(&test_dir, 8, false).await;
    let batch = label_list_optimize_batch(8, 8);
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    dataset.append(reader, Some(params(8))).await.unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let old_segments = segments(&dataset, "labels_idx").await;
    assert_eq!(old_segments.len(), 2);
    let mut legacy_siblings = old_segments.clone();
    // A synthetic v0 declaration shares the current non-stable-ID domain.
    // The latest sibling remains current, so checking only it is insufficient.
    legacy_siblings[0].index_version = 0;
    let uri: &str = &test_dir;
    Dataset::commit(
        uri,
        Operation::CreateIndex {
            new_indices: legacy_siblings,
            removed_indices: old_segments,
        },
        Some(dataset.manifest.version),
        None,
        None,
        dataset.session(),
        false,
    )
    .await
    .unwrap();
    dataset = Dataset::open(uri).await.unwrap();
    let batch = label_list_optimize_batch(16, 16);
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    dataset.append(reader, Some(params(8))).await.unwrap();
    let options = OptimizeOptions::merge(2).max_rows_per_task(Some(8));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!(
        plan.tasks[0].reference.index_version,
        LABEL_LIST_ROW_ID_DOMAIN_VERSION as i32,
    );
    assert_eq!(task_groups(&plan), vec![(0, vec![2, 3])]);
    let results = execute_plan(&dataset, &plan).await;
    commit_optimize_indices(&mut dataset, results, &options)
        .await
        .unwrap();
    assert_eq!(
        coverage(&segments(&dataset, "labels_idx").await),
        (0u32..4).collect()
    );
    assert_eq!(
        label_list_optimize_ids(&dataset, "array_has_all(labels, [1, 2])", true).await,
        label_list_optimize_ids(&dataset, "array_has_all(labels, [1, 2])", false).await,
    );
}

#[tokio::test]
async fn test_label_list_partial_group_can_be_completed_later() {
    let test_dir = TempStrDir::default();
    let mut dataset = write_label_list_dataset(&test_dir, 8, true).await;
    let batch = label_list_optimize_batch(8, 24);
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    dataset.append(reader, Some(params(8))).await.unwrap();
    let options = OptimizeOptions::merge(1).max_rows_per_task(Some(8));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks.len(), 3);
    let mut built = Vec::new();
    for task in &plan.tasks[..2] {
        built.push(task.execute(&dataset).await.unwrap().unwrap());
    }
    let merged = plan.merges[0]
        .execute(&dataset, &built)
        .await
        .unwrap()
        .unwrap();
    commit_optimize_indices(&mut dataset, vec![merged], &options)
        .await
        .unwrap();
    assert_eq!(
        coverage(&segments(&dataset, "labels_idx").await),
        (0u32..3).collect(),
    );
    assert_eq!(
        dataset
            .unindexed_fragments("labels_idx")
            .await
            .unwrap()
            .into_iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>(),
        vec![3],
    );

    let remaining = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!(task_groups(&remaining), vec![(0, vec![3])]);
    let results = execute_plan(&dataset, &remaining).await;
    commit_optimize_indices(&mut dataset, results, &options)
        .await
        .unwrap();
    assert_eq!(
        coverage(&segments(&dataset, "labels_idx").await),
        (0u32..4).collect(),
    );
    assert_eq!(
        label_list_optimize_ids(&dataset, "array_has_all(labels, [1, 2])", true).await,
        label_list_optimize_ids(&dataset, "array_has_all(labels, [1, 2])", false).await,
    );
}
