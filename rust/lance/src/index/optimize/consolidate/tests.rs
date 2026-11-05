// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::WriteParams;
use crate::index::{CreateIndexBuilder, vector::VectorIndexParams};
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator,
};
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::vector::pq::{PQBuildParams, ProductQuantizer};
use lance_index::{IndexType, vector::ivf::IvfBuildParams};
use lance_linalg::distance::MetricType;
use lance_testing::datagen::generate_random_array;
use rstest::rstest;

const DIM: usize = 16;
const BASE: usize = 1024;
const SMALL: usize = 256;
const NAME: &str = "vector_idx";

fn batch(start: usize, rows: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from_iter_values(
                (start..start + rows).map(|id| id as i32),
            )),
            Arc::new(
                FixedSizeListArray::try_new_from_values(
                    generate_random_array(rows * DIM),
                    DIM as i32,
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap()
}

async fn append(dataset: &mut Dataset, start: usize, rows: usize) {
    let batch = batch(start, rows);
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            None,
        )
        .await
        .unwrap();
}

async fn fixture(uri: &str, stable: bool) -> Dataset {
    let first = batch(0, BASE);
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(first.clone())], first.schema()),
        uri,
        Some(WriteParams {
            enable_stable_row_ids: stable,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    append(&mut dataset, BASE, SMALL).await;
    append(&mut dataset, BASE + SMALL, SMALL).await;
    let params = VectorIndexParams::with_ivf_pq_params(
        MetricType::Cosine,
        IvfBuildParams::new(2),
        PQBuildParams {
            num_bits: 4,
            num_sub_vectors: 2,
            ..Default::default()
        },
    );
    let mut segments = Vec::new();
    for fragment in 0..3 {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
                .name(NAME.to_string())
                .fragments(vec![fragment])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    dataset
        .commit_existing_index_segments(NAME, "vector", segments)
        .await
        .unwrap();
    dataset = Dataset::open(uri).await.unwrap();
    let logical = dataset
        .open_logical_vector_index("vector", NAME)
        .await
        .unwrap();
    assert_eq!(
        vector_segment_compatibility(&logical.as_ivf().unwrap(), "fixture").unwrap(),
        VectorSegmentCompatibility::QueryCompatibleModelsDiffer
    );
    dataset
}

async fn segments(dataset: &Dataset) -> Vec<IndexMetadata> {
    dataset.load_indices_by_name(NAME).await.unwrap()
}

async fn model(
    dataset: &Dataset,
    segment: &IndexMetadata,
) -> (arrow::array::ArrayData, arrow::array::ArrayData) {
    let index = dataset
        .open_vector_index("vector", &segment.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let pq: ProductQuantizer = index.quantizer().try_into().unwrap();
    (
        index.ivf_model().centroids_array().unwrap().to_data(),
        pq.codebook.to_data(),
    )
}

fn files(uri: &str, uuid: Uuid) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(format!("{uri}/_indices/{uuid}"))
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().to_string_lossy().into_owned(),
                std::fs::read(entry.path()).unwrap(),
            )
        })
        .collect()
}

/// Fast search excludes unindexed fallback. Full probing/refinement must return
/// every live indexed ID once, and the top ten must match brute force.
async fn assert_query(dataset: &Dataset, expected: usize) {
    let query = Float32Array::from(vec![0.2; DIM]);
    let mut scan = dataset.scan();
    scan.nearest("vector", &query, expected + 10)
        .unwrap()
        .distance_metric(MetricType::Cosine)
        .nprobes(2)
        .refine(1)
        .fast_search();
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("ANN"), "{plan}");
    let result = scan.try_into_batch().await.unwrap();
    assert_eq!(result.num_rows(), expected);
    let ids = result
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .values()
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    assert_eq!(ids.len(), expected);
    let mut exact = dataset.scan();
    exact
        .nearest("vector", &query, 10)
        .unwrap()
        .distance_metric(MetricType::Cosine)
        .use_index(false);
    let exact = exact.try_into_batch().await.unwrap();
    assert_eq!(
        result.column_by_name("id").unwrap().slice(0, 10).to_data(),
        exact.column_by_name("id").unwrap().to_data()
    );
}

async fn execute(dataset: &Dataset, plan: &OptimizeIndicesPlan) -> Vec<IndexSegmentResult> {
    // Drivers can send both stages across a serialization boundary.
    let plan: OptimizeIndicesPlan =
        serde_json::from_str(&serde_json::to_string(plan).unwrap()).unwrap();
    let mut built = Vec::new();
    for task in &plan.tasks {
        built.extend(task.execute(dataset).await.unwrap());
    }
    let mut results = Vec::new();
    for merge in &plan.merges {
        results.extend(merge.execute(dataset, &built).await.unwrap());
    }
    results
}

#[rstest]
#[case(false)]
#[case(true)]
#[tokio::test]
async fn bounded_consolidation_preserves_base_and_model(#[case] stable: bool) {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, stable).await;
    let old = segments(&dataset).await;
    let base_files = files(&dir, old[0].uuid);
    let target = model(&dataset, &old[2]).await;
    assert_ne!(model(&dataset, &old[1]).await, target);
    let options =
        OptimizeOptions::consolidate(2 * SMALL as u64).max_rows_per_task(Some(SMALL as u64));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks.len(), 2);
    assert_eq!(
        plan.tasks
            .iter()
            .flat_map(|task| &task.fragments)
            .copied()
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert!(
        plan.tasks
            .iter()
            .all(|task| !task.retrain && task.reference.uuid == old[2].uuid)
    );
    dataset.optimize_indices(&options).await.unwrap();
    let new = segments(&dataset).await;
    assert_eq!(new.len(), 2);
    assert_eq!(new[0], old[0]);
    assert_eq!(files(&dir, old[0].uuid), base_files);
    assert_eq!(model(&dataset, &new[1]).await, target);
    assert_query(&dataset, BASE + 2 * SMALL).await;
    append(&mut dataset, BASE + 2 * SMALL, 64).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    let merged = segments(&dataset).await;
    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0], old[0]);
    assert_eq!(model(&dataset, &merged[1]).await, target);
    assert_eq!(files(&dir, old[0].uuid), base_files);
    assert_query(&dataset, BASE + 2 * SMALL + 64).await;
}

#[tokio::test]
async fn budget_and_backlog_are_respected() {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, false).await;
    let old = segments(&dataset).await;
    let version = dataset.manifest.version;
    dataset
        .optimize_indices(&OptimizeOptions::consolidate(2 * SMALL as u64 - 1))
        .await
        .unwrap();
    assert_eq!(dataset.manifest.version, version);
    append(&mut dataset, BASE + 2 * SMALL, 4096).await;
    let options = OptimizeOptions::consolidate(2 * SMALL as u64);
    dataset.optimize_indices(&options).await.unwrap();
    assert_eq!(segments(&dataset).await[0], old[0]);
    assert_eq!(
        dataset
            .unindexed_fragments(NAME)
            .await
            .unwrap()
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>(),
        vec![3]
    );
    let bounded_dir = TempStrDir::default();
    let error = plan_optimize_indices(
        &fixture(&bounded_dir, false).await,
        &options.clone().max_rows_per_task(Some(1)),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("exceeds max_rows_per_task"));
    for invalid in [
        OptimizeOptions::consolidate(0),
        options.clone().num_indices_to_merge(Some(2)),
        options.clone().max_rows_per_segment(Some(512)),
        options.with_partition_rebalancing(),
    ] {
        assert!(plan_optimize_indices(&dataset, &invalid).await.is_err());
    }
}

#[tokio::test]
async fn shared_model_consolidation_needs_no_build_tasks() {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, false).await;
    dataset
        .optimize_indices(&OptimizeOptions::consolidate(512))
        .await
        .unwrap();
    append(&mut dataset, BASE + 2 * SMALL, 64).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let options = OptimizeOptions::consolidate(576);
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert!(plan.tasks.is_empty());
    assert_eq!(plan.merges.len(), 1);
    assert_eq!(plan.merges[0].old_segments.len(), 2);
    dataset.optimize_indices(&options).await.unwrap();
    assert_eq!(segments(&dataset).await.len(), 2);
    assert_query(&dataset, BASE + 2 * SMALL + 64).await;
}

#[tokio::test]
async fn incomplete_or_duplicate_results_cannot_replace_sources() {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, false).await;
    let options = OptimizeOptions::consolidate(512).max_rows_per_task(Some(256));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    let first = plan.tasks[0].execute(&dataset).await.unwrap().unwrap();
    let partial = plan.merges[0]
        .execute(&dataset, &[first])
        .await
        .unwrap()
        .unwrap();
    let before = segments(&dataset).await;
    let error = commit_optimize_indices(&mut dataset, vec![partial], &options)
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("every surviving source fragment")
    );
    assert_eq!(segments(&dataset).await, before);
    let results = execute(&dataset, &plan).await;
    let error = commit_optimize_indices(
        &mut dataset,
        vec![results[0].clone(), results[0].clone()],
        &options,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("replace a source twice"));
    assert_eq!(segments(&dataset).await, before);
    commit_optimize_indices(&mut dataset, results, &options)
        .await
        .unwrap();
    assert_query(&dataset, BASE + 2 * SMALL).await;
}

#[rstest]
#[case(false)]
#[case(true)]
#[tokio::test]
async fn concurrent_delete_and_append_reuses_staged_output(#[case] stable: bool) {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, stable).await;
    let old = segments(&dataset).await;
    let base_files = files(&dir, old[0].uuid);
    let options = OptimizeOptions::consolidate(512).max_rows_per_task(Some(256));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    let results = execute(&dataset, &plan).await;
    let uuid = results[0].segment.uuid;
    let staged_files = files(&dir, uuid);
    let mut concurrent = dataset.clone();
    concurrent
        .delete(&format!("id >= {}", BASE + SMALL))
        .await
        .unwrap();
    concurrent
        .delete(&format!("id >= {BASE} AND id < {}", BASE + 16))
        .await
        .unwrap();
    append(&mut concurrent, BASE + 2 * SMALL, 64).await;
    concurrent
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    // Compare persisted metadata: manifest serialization rounds creation times
    // to milliseconds, whereas the writer retains the original timestamp.
    concurrent = Dataset::open(dir.as_str()).await.unwrap();
    let appended = segments(&concurrent).await.last().unwrap().clone();
    let append_files = files(&dir, appended.uuid);
    assert!(
        commit_optimize_indices(&mut dataset, results.clone(), &options)
            .await
            .is_err()
    );
    dataset = Dataset::open(dir.as_str()).await.unwrap();
    commit_optimize_indices(&mut dataset, results, &options)
        .await
        .unwrap();
    let committed = segments(&dataset).await;
    assert!(committed.iter().any(|segment| segment.uuid == uuid
        && segment.fragment_bitmap == Some(RoaringBitmap::from_iter([1]))));
    assert!(committed.contains(&appended));
    assert!(committed.contains(&old[0]));
    assert_eq!(files(&dir, uuid), staged_files);
    assert_eq!(files(&dir, old[0].uuid), base_files);
    assert_eq!(files(&dir, appended.uuid), append_files);
    assert_query(&dataset, BASE + SMALL - 16 + 64).await;
    append(&mut dataset, BASE + 2 * SMALL + 64, 32).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    assert_query(&dataset, BASE + SMALL - 16 + 96).await;
}

#[tokio::test]
async fn concurrent_source_replacement_requires_replanning() {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, false).await;
    let options = OptimizeOptions::consolidate(512);
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    let results = execute(&dataset, &plan).await;
    dataset.optimize_indices(&options).await.unwrap();
    let committed = segments(&dataset).await;
    let error = commit_optimize_indices(&mut dataset, results, &options)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("was replaced; replan"));
    assert_eq!(segments(&dataset).await, committed);
    assert_query(&dataset, BASE + 2 * SMALL).await;
}

#[tokio::test]
async fn budget_counts_retired_index_entries() {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, false).await;
    append(&mut dataset, BASE + 2 * SMALL, 64).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    // The tail still physically contains 320 entries, although deleting its
    // first fragment leaves only 64 live rows. The preceding segment has 256.
    dataset
        .delete(&format!(
            "id >= {} AND id < {}",
            BASE + SMALL,
            BASE + 2 * SMALL
        ))
        .await
        .unwrap();
    let before = segments(&dataset).await;
    let version = dataset.manifest.version;
    dataset
        .optimize_indices(&OptimizeOptions::consolidate(320))
        .await
        .unwrap();
    assert_eq!(dataset.manifest.version, version);
    assert_eq!(segments(&dataset).await, before);
    dataset
        .optimize_indices(&OptimizeOptions::consolidate(576))
        .await
        .unwrap();
    assert_eq!(segments(&dataset).await.len(), 2);
    assert_query(&dataset, BASE + SMALL + 64).await;
}

#[tokio::test]
async fn fully_deleted_output_needs_no_commit() {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, false).await;
    let options = OptimizeOptions::consolidate(512);
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    let results = execute(&dataset, &plan).await;
    dataset.delete(&format!("id >= {BASE}")).await.unwrap();
    append(&mut dataset, BASE + 2 * SMALL, 64).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let before = segments(&dataset).await;
    let version = dataset.manifest.version;
    commit_optimize_indices(&mut dataset, results, &options)
        .await
        .unwrap();
    assert_eq!(dataset.manifest.version, version);
    assert_eq!(segments(&dataset).await, before);
    assert_query(&dataset, BASE + 64).await;
}

#[rstest]
#[case(false, false)]
#[case(true, false)]
#[case(false, true)]
#[case(true, true)]
#[tokio::test]
async fn restored_source_rows_require_replanning(
    #[case] stable: bool,
    #[case] delete_different_rows: bool,
) {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, stable).await;
    let before_delete = dataset.manifest.version;
    dataset
        .delete(&format!("id >= {BASE} AND id < {}", BASE + SMALL / 2))
        .await
        .unwrap();
    let options = OptimizeOptions::consolidate(512);
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert!(!plan.tasks.is_empty());
    let results = execute(&dataset, &plan).await;
    let mut restored = dataset.checkout_version(before_delete).await.unwrap();
    restored.restore().await.unwrap();
    dataset = Dataset::open(dir.as_str()).await.unwrap();
    let expected = if delete_different_rows {
        // The deletion count is unchanged, but the omitted rows are live again.
        dataset
            .delete(&format!(
                "id >= {} AND id < {}",
                BASE + SMALL / 2,
                BASE + SMALL
            ))
            .await
            .unwrap();
        BASE + 2 * SMALL - SMALL / 2
    } else {
        BASE + 2 * SMALL
    };
    assert_query(&dataset, expected).await;
    let version = dataset.manifest.version;
    let before = segments(&dataset).await;
    let error = commit_optimize_indices(&mut dataset, results, &options)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("has restored rows; replan"));
    dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(dataset.manifest.version, version);
    assert_eq!(segments(&dataset).await, before);
    assert_eq!(dataset.count_rows(None).await.unwrap(), expected);
    assert_query(&dataset, expected).await;
    dataset.optimize_indices(&options).await.unwrap();
    assert_eq!(segments(&dataset).await.len(), 2);
    assert_query(&dataset, expected).await;
}

#[rstest]
#[case(false)]
#[case(true)]
#[tokio::test]
async fn additional_deletions_reuse_staged_output(#[case] stable: bool) {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, stable).await;
    dataset.delete(&format!("id = {BASE}")).await.unwrap();
    let options = OptimizeOptions::consolidate(512);
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    let results = execute(&dataset, &plan).await;
    let uuid = results[0].segment.uuid;
    let staged_files = files(&dir, uuid);
    dataset.delete(&format!("id = {}", BASE + 1)).await.unwrap();
    commit_optimize_indices(&mut dataset, results, &options)
        .await
        .unwrap();
    assert!(
        segments(&dataset)
            .await
            .iter()
            .any(|segment| segment.uuid == uuid)
    );
    assert_eq!(files(&dir, uuid), staged_files);
    assert_query(&dataset, BASE + 2 * SMALL - 2).await;
}
