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
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::{RQBuildParams, RQRotationType};
use lance_index::vector::pq::{PQBuildParams, ProductQuantizer};
use lance_index::vector::quantizer::Quantization;
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
    let params = VectorIndexParams::with_ivf_pq_params(
        MetricType::Cosine,
        IvfBuildParams::new(2),
        PQBuildParams {
            num_bits: 4,
            num_sub_vectors: 2,
            ..Default::default()
        },
    );
    fixture_with(uri, stable, &params).await
}

/// Three segments, one per fragment, each trained on its own fragment.
async fn fixture_with(uri: &str, stable: bool, params: &VectorIndexParams) -> Dataset {
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
    let mut segments = Vec::new();
    for fragment in 0..3 {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, params)
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

async fn indexed_rows(dataset: &Dataset, segment: &IndexMetadata) -> u64 {
    dataset
        .open_vector_index("vector", &segment.uuid, &NoOpMetricsCollector)
        .await
        .unwrap()
        .num_rows()
}

fn select(segments: &[&IndexMetadata]) -> OptimizeOptions {
    OptimizeOptions::consolidate_selected(
        NAME,
        segments.iter().map(|segment| segment.uuid).collect(),
    )
}

#[rstest]
#[case(false)]
#[case(true)]
#[tokio::test]
async fn selected_segments_with_different_models_are_retrained(#[case] stable: bool) {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, stable).await;
    let old = segments(&dataset).await;
    let newest_files = files(&dir, old[2].uuid);
    dataset
        .delete(&format!("id < {} OR id = {BASE}", BASE / 2))
        .await
        .unwrap();
    let live = (BASE / 2 + SMALL - 1) as u64;
    // The old segments still hold entries for the deleted rows.
    assert_eq!(
        indexed_rows(&dataset, &old[0]).await + indexed_rows(&dataset, &old[1]).await,
        (BASE + SMALL) as u64
    );

    let mut models = Vec::new();
    for segment in &old {
        models.push(model(&dataset, segment).await);
    }
    let options = select(&[&old[0], &old[1]]);
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!(plan.merges.len(), 1);
    assert_eq!(plan.merges[0].replaces, vec![old[0].uuid, old[1].uuid]);
    // One model trained for the selection, outside the manifest.
    let trained = plan.tasks[0].reference.uuid;
    assert!(plan.tasks.iter().all(|task| task.reference.uuid == trained));
    assert!(old.iter().all(|segment| segment.uuid != trained));
    dataset.optimize_indices(&options).await.unwrap();

    let new = segments(&dataset).await;
    assert_eq!(new.len(), 2);
    assert_eq!(new[0], old[2]);
    assert_eq!(files(&dir, old[2].uuid), newest_files);
    assert_eq!(indexed_rows(&dataset, &new[1]).await, live);
    let retrained = model(&dataset, &new[1]).await;
    assert!(models.iter().all(|source| *source != retrained));
    // Sized from the selected rows, not copied from a source's two partitions.
    let index = dataset
        .open_vector_index("vector", &new[1].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(index.ivf_model().num_partitions(), 1);
    assert_query(&dataset, live as usize + SMALL).await;
}

#[tokio::test]
async fn a_retrained_rq_selection_keeps_the_shared_rotation() {
    let dir = TempStrDir::default();
    let shared = RabitQuantizer::new_with_rotation::<arrow_array::types::Float32Type>(
        RQBuildParams::default().num_bits,
        DIM as i32,
        RQRotationType::default(),
    )
    .metadata(None);
    let params = VectorIndexParams::with_ivf_rq_params(
        MetricType::Cosine,
        IvfBuildParams::new(2),
        RQBuildParams {
            rotation: Some(shared.clone()),
            ..Default::default()
        },
    );
    let mut dataset = fixture_with(&dir, false, &params).await;
    let rotation = async |dataset: &Dataset, segment: &IndexMetadata| {
        let index = dataset
            .open_vector_index("vector", &segment.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let quantizer: RabitQuantizer = index.quantizer().try_into().unwrap();
        let metadata = quantizer.metadata_ref();
        (
            metadata.fast_rotation_signs.clone(),
            metadata.rotate_mat.as_ref().map(|matrix| matrix.to_data()),
            index.ivf_model().centroids_array().unwrap().to_data(),
        )
    };
    let expected = (
        shared.fast_rotation_signs.clone(),
        shared.rotate_mat.as_ref().map(|matrix| matrix.to_data()),
    );
    let old = segments(&dataset).await;
    let source = rotation(&dataset, &old[1]).await;
    assert_eq!((source.0.clone(), source.1.clone()), expected);

    dataset
        .optimize_indices(&select(&[&old[1], &old[2]]))
        .await
        .unwrap();
    let new = segments(&dataset).await;
    assert_eq!(new.len(), 2);
    let output = rotation(&dataset, &new[1]).await;
    // Retrained centroids, under the rotation every segment shares.
    assert_ne!(output.2, source.2);
    assert_eq!((output.0, output.1), expected);
    assert_query(&dataset, BASE + 2 * SMALL).await;
}

#[tokio::test]
async fn a_shared_model_selection_keeps_its_model() {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, false).await;
    let old = segments(&dataset).await;
    // An append is encoded under the newest model, so it shares old[2]'s.
    append(&mut dataset, BASE + 2 * SMALL, SMALL).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let appended = segments(&dataset).await[3].clone();
    let shared = model(&dataset, &old[2]).await;
    assert_eq!(model(&dataset, &appended).await, shared);

    let options = select(&[&old[2], &appended]);
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert!(
        plan.tasks
            .iter()
            .all(|task| task.reference.uuid == appended.uuid)
    );
    dataset.optimize_indices(&options).await.unwrap();
    let new = segments(&dataset).await;
    assert_eq!(new.len(), 3);
    assert_eq!(model(&dataset, &new[2]).await, shared);
    assert_query(&dataset, BASE + 3 * SMALL).await;
}

#[tokio::test]
async fn a_selection_packs_its_output_and_splits_a_segment() {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, false).await;
    let old = segments(&dataset).await;
    dataset
        .optimize_indices(&select(&[&old[1], &old[2]]))
        .await
        .unwrap();
    let merged = segments(&dataset).await[1].clone();
    assert_eq!(indexed_rows(&dataset, &merged).await, 2 * SMALL as u64);
    let merged_model = model(&dataset, &merged).await;

    // One segment shares its own model, so splitting it keeps that model.
    let options = select(&[&merged]).max_rows_per_segment(Some(SMALL as u64));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!(plan.merges.len(), 2);
    dataset.optimize_indices(&options).await.unwrap();
    let new = segments(&dataset).await;
    assert_eq!(new.len(), 3);
    assert_eq!(new[0], old[0]);
    for output in &new[1..] {
        assert_eq!(indexed_rows(&dataset, output).await, SMALL as u64);
        assert_eq!(model(&dataset, output).await, merged_model);
    }
    assert_query(&dataset, BASE + 2 * SMALL).await;
}

#[tokio::test]
async fn deleted_rows_do_not_count_toward_the_segment_limit() {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, false).await;
    let old = segments(&dataset).await;
    let half = SMALL / 2;
    dataset
        .delete(&format!(
            "(id >= {BASE} AND id < {}) OR (id >= {} AND id < {})",
            BASE + half,
            BASE + SMALL,
            BASE + SMALL + half
        ))
        .await
        .unwrap();
    // Two fragments of SMALL physical rows hold SMALL live rows: one output.
    let options = select(&[&old[1], &old[2]]).max_rows_per_segment(Some(SMALL as u64));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!(plan.merges.len(), 1);
    dataset.optimize_indices(&options).await.unwrap();
    let new = segments(&dataset).await;
    assert_eq!(new.len(), 2);
    assert_eq!(indexed_rows(&dataset, &new[1]).await, SMALL as u64);
    assert_query(&dataset, BASE + SMALL).await;
}

#[tokio::test]
async fn a_fragment_larger_than_the_limits_is_an_output_of_its_own() {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, false).await;
    let old = segments(&dataset).await;
    let limit = Some(SMALL as u64 / 2);
    let options = select(&[&old[1], &old[2]])
        .max_rows_per_task(limit)
        .max_rows_per_segment(limit);
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!(plan.merges.len(), 2);
    dataset.optimize_indices(&options).await.unwrap();
    let new = segments(&dataset).await;
    assert_eq!(new.len(), 3);
    for output in &new[1..] {
        assert_eq!(indexed_rows(&dataset, output).await, SMALL as u64);
    }
    assert_query(&dataset, BASE + 2 * SMALL).await;
}

#[tokio::test]
async fn a_selected_segment_already_removed_is_skipped() {
    let dir = TempStrDir::default();
    let mut dataset = fixture(&dir, false).await;
    let old = segments(&dataset).await;
    // Deleting all of a segment's rows removes the segment.
    dataset
        .delete(&format!("id >= {BASE} AND id < {}", BASE + SMALL))
        .await
        .unwrap();
    let plan = plan_optimize_indices(&dataset, &select(&[&old[1], &old[2]]))
        .await
        .unwrap();
    assert_eq!(plan.merges.len(), 1);
    assert_eq!(plan.merges[0].replaces, vec![old[2].uuid]);

    let plan = plan_optimize_indices(&dataset, &select(&[&old[1]]))
        .await
        .unwrap();
    assert!(plan.tasks.is_empty() && plan.merges.is_empty());
}

#[tokio::test]
async fn a_selection_commits_over_concurrent_deletes_and_appends() {
    let dir = TempStrDir::default();
    let dataset = fixture(&dir, false).await;
    let old = segments(&dataset).await;
    let options = select(&[&old[0], &old[1]]);
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    let results = execute(&dataset, &plan).await;
    let output = results[0].segment.uuid;

    let mut concurrent = dataset.clone();
    concurrent.delete("id < 16").await.unwrap();
    append(&mut concurrent, BASE + 2 * SMALL, 64).await;
    concurrent
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    let appended = segments(&dataset).await.last().unwrap().clone();
    commit_optimize_indices(&mut dataset, results, &options)
        .await
        .unwrap();

    let committed = segments(&dataset).await;
    assert_eq!(committed.len(), 3);
    assert!(committed.contains(&old[2]) && committed.contains(&appended));
    assert!(committed.iter().any(|segment| segment.uuid == output
        && segment.fragment_bitmap == Some(RoaringBitmap::from_iter([0, 1]))));
    assert_query(&dataset, BASE + 2 * SMALL - 16 + 64).await;
}

#[tokio::test]
async fn invalid_selections_are_rejected() {
    let dir = TempStrDir::default();
    let dataset = fixture(&dir, false).await;
    let old = segments(&dataset).await;
    let mut unnamed = select(&[&old[0]]);
    unnamed.index_names = None;
    let mut budgeted = select(&[&old[0]]);
    budgeted.max_rows_to_consolidate = Some(4 * BASE as u64);
    for invalid in [
        unnamed,
        select(&[&old[0], &old[0]]),
        select(&[]),
        select(&[&old[0]]).index_names(vec![NAME.to_string(), "other".to_string()]),
        budgeted,
        select(&[&old[0]]).num_indices_to_merge(Some(1)),
        select(&[&old[0]]).max_rows_per_segment(Some(0)),
        OptimizeOptions::consolidate(BASE as u64).max_rows_per_segment(Some(SMALL as u64)),
    ] {
        let error = plan_optimize_indices(&dataset, &invalid).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }
}
