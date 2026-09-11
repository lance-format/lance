// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::vec;

use crate::dataset::InsertBuilder;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::index::DatasetIndexExt;
use crate::utils::test::copy_test_data_to_tmp;
use crate::{Dataset, Result};
use lance_index::{IndexCriteria, IndexType, scalar::ScalarIndexParams};
use lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
use lance_table::format::{Fragment, IndexMetadata, RowIdMeta};
use lance_table::rowids::read_row_ids;

use crate::dataset::write::{WriteMode, WriteParams};
use arrow::compute::concat_batches;
use arrow_array::RecordBatch;
use arrow_array::{Array, Float32Array, Int64Array, ListArray, RecordBatchIterator, UInt32Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_file::version::LanceFileVersion;

use futures::{StreamExt, TryStreamExt};
use rstest::rstest;

pub(super) async fn scan_dataset(uri: &str) -> Result<Vec<RecordBatch>> {
    let results = Dataset::open(uri)
        .await?
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    Ok(results)
}

#[rstest]
#[tokio::test]
async fn test_v0_7_5_migration() {
    // We migrate to add Fragment.physical_rows and DeletionFile.num_deletions
    // after this version.

    // Copy over table
    let test_dir = copy_test_data_to_tmp("v0.7.5/with_deletions").unwrap();
    let test_uri = test_dir.path_str();

    // Assert num rows, deletions, and physical rows are all correct.
    let dataset = Dataset::open(&test_uri).await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 90);
    assert_eq!(dataset.count_deleted_rows().await.unwrap(), 10);
    let total_physical_rows = futures::stream::iter(dataset.get_fragments())
        .then(|f| async move { f.physical_rows().await })
        .try_fold(0, |acc, x| async move { Ok(acc + x) })
        .await
        .unwrap();
    assert_eq!(total_physical_rows, 100);

    // Append 5 rows
    let schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(100..105))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };
    let dataset = Dataset::write(batches, &test_uri, Some(write_params))
        .await
        .unwrap();

    // Assert num rows, deletions, and physical rows are all correct.
    assert_eq!(dataset.count_rows(None).await.unwrap(), 95);
    assert_eq!(dataset.count_deleted_rows().await.unwrap(), 10);
    let total_physical_rows = futures::stream::iter(dataset.get_fragments())
        .then(|f| async move { f.physical_rows().await })
        .try_fold(0, |acc, x| async move { Ok(acc + x) })
        .await
        .unwrap();
    assert_eq!(total_physical_rows, 105);

    dataset.validate().await.unwrap();

    // Scan data and assert it is as expected.
    let expected = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(
            (0..10).chain(20..105),
        ))],
    )
    .unwrap();
    let actual_batches = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let actual = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
#[tokio::test]
async fn test_fix_v0_8_0_broken_migration() {
    // The migration from v0.7.5 was broken in 0.8.0. This validates we can
    // automatically fix tables that have this problem.

    // Copy over table
    let test_dir = copy_test_data_to_tmp("v0.8.0/migrated_from_v0.7.5").unwrap();
    let test_uri = test_dir.path_str();
    let test_uri = &test_uri;

    // Assert num rows, deletions, and physical rows are all correct, even
    // though stats are bad.
    let dataset = Dataset::open(test_uri).await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 92);
    assert_eq!(dataset.count_deleted_rows().await.unwrap(), 10);
    let total_physical_rows = futures::stream::iter(dataset.get_fragments())
        .then(|f| async move { f.physical_rows().await })
        .try_fold(0, |acc, x| async move { Ok(acc + x) })
        .await
        .unwrap();
    assert_eq!(total_physical_rows, 102);

    // Append 5 rows to table.
    let schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(100..105))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Append,
        data_storage_version: Some(LanceFileVersion::Legacy),
        ..Default::default()
    };
    let dataset = Dataset::write(batches, test_uri, Some(write_params))
        .await
        .unwrap();

    // Assert statistics are all now correct.
    let physical_rows: Vec<_> = dataset
        .get_fragments()
        .iter()
        .map(|f| f.metadata.physical_rows)
        .collect();
    assert_eq!(physical_rows, vec![Some(100), Some(2), Some(5)]);
    let num_deletions: Vec<_> = dataset
        .get_fragments()
        .iter()
        .map(|f| {
            f.metadata
                .deletion_file
                .as_ref()
                .and_then(|df| df.num_deleted_rows)
        })
        .collect();
    assert_eq!(num_deletions, vec![Some(10), None, None]);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 97);

    // Scan data and assert it is as expected.
    let expected = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(
            (0..10).chain(20..100).chain(0..2).chain(100..105),
        ))],
    )
    .unwrap();
    let actual_batches = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let actual = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::v1_v2_mixed_rejected(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_v0_8_14_invalid_index_fragment_bitmap(
    #[case] data_storage_version: LanceFileVersion,
) {
    // Old versions of lance could create an index whose fragment bitmap was
    // invalid because it did not include fragments that were part of the index
    //
    // We need to make sure we do not rely on the fragment bitmap in these older
    // versions and instead fall back to a slower legacy behavior
    let test_dir = copy_test_data_to_tmp("v0.8.14/corrupt_index").unwrap();
    let test_uri = test_dir.path_str();
    let test_uri = &test_uri;

    let mut dataset = Dataset::open(test_uri).await.unwrap();

    // Uncomment to reproduce the issue.  The below query will panic
    // let mut scan = dataset.scan();
    // let query_vec = Float32Array::from(vec![0_f32; 128]);
    // let scan_fut = scan
    //     .nearest("vector", &query_vec, 2000)
    //     .unwrap()
    //     .nprobes(4)
    //     .prefilter(true)
    //     .try_into_stream()
    //     .await
    //     .unwrap()
    //     .try_collect::<Vec<_>>()
    //     .await
    //     .unwrap();

    // Add some data and recalculate the index, forcing a migration
    let mut scan = dataset.scan();
    let data = scan
        .limit(Some(10), None)
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let schema = data[0].schema();
    let data = RecordBatchIterator::new(data.into_iter().map(arrow::error::Result::Ok), schema);

    let broken_version = dataset.version().version;

    // Any transaction, no matter how simple, should trigger the fragment bitmap to be recalculated
    let append_result = dataset
        .append(
            data,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await;

    if matches!(data_storage_version, LanceFileVersion::Stable) {
        let error = append_result.unwrap_err();
        assert!(
            error.to_string().contains("do not have a single version"),
            "{error}"
        );
        return;
    }
    append_result.unwrap();

    for idx in dataset.load_indices().await.unwrap().iter() {
        // The corrupt fragment_bitmap does not contain 0 but the
        // restored one should
        assert!(idx.fragment_bitmap.as_ref().unwrap().contains(0));
    }

    let mut dataset = dataset.checkout_version(broken_version).await.unwrap();
    dataset.restore().await.unwrap();

    // Running compaction right away should work (this is verifying compaction
    // is not broken by the potentially malformed fragment bitmaps)
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();

    for idx in dataset.load_indices().await.unwrap().iter() {
        assert!(idx.fragment_bitmap.as_ref().unwrap().contains(0));
    }

    let mut scan = dataset.scan();
    let query_vec = Float32Array::from(vec![0_f32; 128]);
    let batches = scan
        .nearest("vector", &query_vec, 2000)
        .unwrap()
        .nprobes(4)
        .prefilter(true)
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let row_count = batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
    assert_eq!(row_count, 1900);
}

/// The repair above is triggered by the writer version of the manifest being
/// committed *from*, and a successful commit stamps the current one. So a
/// commit that cannot open the index has exactly one chance at the corrupt
/// bitmap, and carrying it through unverified would hand a later build a
/// bitmap that looks migrated.
#[tokio::test]
async fn test_v0_8_14_invalid_index_fragment_bitmap_repair_is_not_lost() {
    let test_dir = copy_test_data_to_tmp("v0.8.14/corrupt_index").unwrap();
    let test_uri = test_dir.path_str();

    let indices_dir = test_dir.std_path().join("_indices");
    let stashed_dir = test_dir.std_path().join("_indices_stashed");
    std::fs::rename(&indices_dir, &stashed_dir).unwrap();

    let mut dataset = Dataset::open(&test_uri).await.unwrap();
    dataset.delete("false").await.unwrap();

    for idx in dataset.load_indices().await.unwrap().iter() {
        assert_eq!(
            idx.fragment_bitmap, None,
            "a bitmap the migration could not verify must be recorded as unknown"
        );
    }

    std::fs::rename(&stashed_dir, &indices_dir).unwrap();

    let mut dataset = Dataset::open(&test_uri).await.unwrap();
    dataset.delete("false").await.unwrap();

    for idx in dataset.load_indices().await.unwrap().iter() {
        assert!(
            idx.fragment_bitmap.as_ref().unwrap().contains(0),
            "the first build that can open the index must repair the coverage"
        );
    }
}

#[tokio::test]
async fn test_fix_v0_10_5_corrupt_schema() {
    // Schemas could be corrupted by successive calls to `add_columns` and
    // `drop_columns`. We should be able to detect this by checking for
    // duplicate field ids. We should be able to fix this in new commits
    // by dropping unused data files and re-writing the schema.

    // Copy over table
    let test_dir = copy_test_data_to_tmp("v0.10.5/corrupt_schema").unwrap();
    let test_uri = test_dir.path_str();
    let test_uri = &test_uri;

    let mut dataset = Dataset::open(test_uri).await.unwrap();

    let validate_res = dataset.validate().await;
    assert!(validate_res.is_err());

    // Force a migration.
    dataset.delete("false").await.unwrap();
    dataset.validate().await.unwrap();

    let data = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(
        data["b"]
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[0, 4, 8, 12]
    );
    assert_eq!(
        data["c"]
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[0, 5, 10, 15]
    );
}

#[tokio::test]
async fn test_fix_v0_21_0_corrupt_fragment_bitmap() {
    // In v0.21.0 and earlier, delta indices had a bug where the fragment bitmap
    // could contain fragments that are part of other index deltas.

    // Copy over table
    let test_dir = copy_test_data_to_tmp("v0.21.0/bad_index_fragment_bitmap").unwrap();
    let test_uri = test_dir.path_str();
    let test_uri = &test_uri;

    let mut dataset = Dataset::open(test_uri).await.unwrap();

    let validate_res = dataset.validate().await;
    assert!(validate_res.is_err());
    assert_eq!(dataset.load_indices().await.unwrap()[0].name, "vector_idx");

    // Calling index statistics will force a migration
    let stats = dataset.index_statistics("vector_idx").await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(stats["num_indexed_fragments"], 2);

    dataset.checkout_latest().await.unwrap();
    dataset.validate().await.unwrap();

    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 2);
    fn get_bitmap(meta: &IndexMetadata) -> Vec<u32> {
        meta.fragment_bitmap.as_ref().unwrap().iter().collect()
    }
    assert_eq!(get_bitmap(&indices[0]), vec![0]);
    assert_eq!(get_bitmap(&indices[1]), vec![1]);
}

/// Unlike the pre-0.8.15 trigger, an overlap is re-derived from the index
/// metadata on every commit, so it asks to be recalculated again on its own. A
/// commit that cannot open the index has nothing to preserve and must leave the
/// coverage alone: `None` is a state modern indices are not built to recover
/// from, since `calculate_included_frags` exists only for old manifests.
#[tokio::test]
async fn test_v0_21_0_corrupt_fragment_bitmap_kept_when_index_cannot_be_opened() {
    let test_dir = copy_test_data_to_tmp("v0.21.0/bad_index_fragment_bitmap").unwrap();
    let test_uri = test_dir.path_str();

    std::fs::rename(
        test_dir.std_path().join("_indices"),
        test_dir.std_path().join("_indices_stashed"),
    )
    .unwrap();

    fn coverage(indices: &[IndexMetadata]) -> Vec<(String, Option<Vec<u32>>)> {
        let mut coverage = indices
            .iter()
            .map(|idx| {
                (
                    idx.uuid.to_string(),
                    idx.fragment_bitmap
                        .as_ref()
                        .map(|bitmap| bitmap.iter().collect()),
                )
            })
            .collect::<Vec<_>>();
        coverage.sort();
        coverage
    }

    let mut dataset = Dataset::open(&test_uri).await.unwrap();
    let before = coverage(&dataset.load_indices().await.unwrap());

    dataset.delete("false").await.unwrap();

    assert_eq!(
        coverage(&dataset.load_indices().await.unwrap()),
        before,
        "coverage the overlap check will ask about again must be left as it stands"
    );
}

#[tokio::test]
async fn test_v8_decimal_zonemap_missing_extrema() {
    async fn query_ids(
        dataset: &Dataset,
        predicate: &str,
        use_scalar_index: bool,
    ) -> (String, Vec<i64>) {
        let mut scan = dataset.scan();
        scan.project(&["id"])
            .unwrap()
            .use_scalar_index(use_scalar_index)
            .filter(predicate)
            .unwrap();
        let plan = scan.explain_plan(false).await.unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        let ids = batch["id"]
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        (plan, ids)
    }

    let test_dir = copy_test_data_to_tmp("v8.0.0/decimal_zonemap").unwrap();
    let dataset = Dataset::open(&test_dir.path_str()).await.unwrap();

    for predicate in [
        "value = arrow_cast(2.00, 'Decimal128(10, 2)')",
        "value >= arrow_cast(2.00, 'Decimal128(10, 2)') AND \
         value < arrow_cast(3.00, 'Decimal128(10, 2)')",
        "value IN (arrow_cast(2.00, 'Decimal128(10, 2)'), \
         arrow_cast(4.00, 'Decimal128(10, 2)'))",
    ] {
        let (indexed_plan, indexed_ids) = query_ids(&dataset, predicate, true).await;
        let (flat_plan, flat_ids) = query_ids(&dataset, predicate, false).await;

        assert!(indexed_plan.contains("ScalarIndexQuery"), "{indexed_plan}");
        assert!(!flat_plan.contains("ScalarIndexQuery"), "{flat_plan}");
        assert_eq!(
            indexed_ids, flat_ids,
            "indexed query diverged for {predicate}"
        );
        assert_eq!(flat_ids, vec![2]);
    }
}

#[tokio::test]
async fn test_max_fragment_id_migration() {
    // v0.5.9 and earlier did not store the max fragment id in the manifest.
    // This test ensures that we can read such datasets and migrate them to
    // the latest version, which requires the max fragment id to be present.
    {
        let test_dir = copy_test_data_to_tmp("v0.5.9/no_fragments").unwrap();
        let test_uri = test_dir.path_str();
        let test_uri = &test_uri;
        let dataset = Dataset::open(test_uri).await.unwrap();

        assert_eq!(dataset.manifest.max_fragment_id, None);
        assert_eq!(dataset.manifest.max_fragment_id(), None);
    }

    {
        let test_dir = copy_test_data_to_tmp("v0.5.9/dataset_with_fragments").unwrap();
        let test_uri = test_dir.path_str();
        let test_uri = &test_uri;
        let dataset = Dataset::open(test_uri).await.unwrap();

        assert_eq!(dataset.manifest.max_fragment_id, None);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(2));
    }
}

#[tokio::test]
async fn test_index_without_file_sizes() {
    // Test that we can open indices created before the `files` field was added
    // to IndexMetadata. The index should still work correctly, falling back to
    // HEAD calls for file sizes.

    let test_dir = copy_test_data_to_tmp("pre_file_sizes/index_without_file_sizes").unwrap();
    let test_uri = test_dir.path_str();

    // Open the dataset
    let dataset = Dataset::open(&test_uri).await.unwrap();

    // Verify the index exists and has no file size info
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    let index = &indices[0];
    assert_eq!(index.name, "values_idx");
    assert!(
        index.files.is_none() || index.files.as_ref().unwrap().is_empty(),
        "Index should not have file size info (created with old version)"
    );
    // A manifest predating `covering_fields` decodes it as empty, so the keyed
    // prefix is the whole of `fields` -- exactly what selection assumed before
    // covering existed.
    assert!(
        index.covering_fields.is_empty(),
        "Index from old version should declare no covered columns"
    );

    // Selection derives the keyed count as `fields.len() - covering_fields.len()`.
    // On this old metadata it must still resolve to the one indexed column;
    // otherwise the filter below would quietly fall back to a full scan and
    // still return the right row.
    let selected = dataset
        .load_scalar_index(IndexCriteria::default().for_column("values"))
        .await
        .unwrap();
    assert_eq!(
        selected.map(|idx| idx.name),
        Some("values_idx".to_string()),
        "an old single-field index must still be selected for its column"
    );

    // Verify the index still works - scan with a filter that uses the index
    let batch = dataset
        .scan()
        .filter("values = 'value_42'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch.num_rows(), 1);

    // Verify describe_indices returns None for total_size_bytes for old indices
    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert_eq!(descriptions.len(), 1);
    assert!(
        descriptions[0].total_size_bytes().is_none(),
        "Old index without file sizes should return None for total_size_bytes"
    );
}

#[tokio::test]
async fn test_index_file_size_migration() {
    // Test that file sizes are migrated when a write operation is performed
    // on a dataset with an index missing file sizes.

    let test_dir = copy_test_data_to_tmp("pre_file_sizes/index_without_file_sizes").unwrap();
    let test_uri = test_dir.path_str();

    // Open the dataset and verify the index has no file sizes
    let dataset = Dataset::open(&test_uri).await.unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert!(
        indices[0].files.is_none() || indices[0].files.as_ref().unwrap().is_empty(),
        "Index should not have file size info before migration"
    );

    // Perform a write operation (append) to trigger migration
    let batch = arrow_array::record_batch!(
        ("id", Int64, [100, 101]),
        ("values", Utf8, ["value_100", "value_101"])
    )
    .unwrap();
    let dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap();

    // Verify the index now has file sizes after migration
    let indices = dataset.load_indices().await.unwrap();
    let index = &indices[0];
    assert!(
        index.files.is_some() && !index.files.as_ref().unwrap().is_empty(),
        "Index should have file size info after migration"
    );

    // Verify each file has a positive size
    for file in index.files.as_ref().unwrap() {
        assert!(
            file.size_bytes > 0,
            "File {} should have positive size after migration",
            file.path
        );
    }

    // Verify describe_indices now returns total_size_bytes
    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert!(
        descriptions[0].total_size_bytes().is_some(),
        "Index should have total_size_bytes after migration"
    );
    assert!(
        descriptions[0].total_size_bytes().unwrap() > 0,
        "Total size should be positive after migration"
    );
}

/// Regression test for issue #5702: project_by_schema should reorder fields inside List<Struct>.
///
/// This test reads a dataset with:
/// - Fragment 0: List<Struct<a, b, c>> with all fields + "extra" column
/// - Fragment 1: List<Struct<c, b>> with reordered/missing inner struct fields
///
/// Before the fix, reading would fail with:
/// "Incorrect datatype for StructArray field expected List(Struct(...)) got List(Struct(...))"
#[tokio::test]
async fn test_list_struct_field_reorder_issue_5702() {
    let test_dir = copy_test_data_to_tmp("v1.0.1/list_struct_reorder.lance")
        .expect("Failed to copy test data");
    let test_uri = test_dir.path_str();

    let dataset = Dataset::open(&test_uri)
        .await
        .expect("Failed to open dataset");

    // Verify we have 2 fragments
    assert_eq!(dataset.get_fragments().len(), 2);

    // This read would fail before the fix for #5702
    let batches = scan_dataset(&test_uri)
        .await
        .expect("Failed to scan dataset");
    let batch = concat_batches(&batches[0].schema(), batches.iter()).expect("Failed to concat");

    // Verify we got all 4 rows
    assert_eq!(batch.num_rows(), 4);

    // Verify schema has expected columns
    assert_eq!(batch.schema().fields().len(), 3); // id, data, extra
}

/// Regression test for issue #6936: v6.0.1 truncated a miniblock's structural
/// level count to u16 while retaining the complete RLE payload.
#[tokio::test]
async fn test_v6_0_1_miniblock_level_count_overflow() {
    let test_dir = copy_test_data_to_tmp("v6.0.1/miniblock_level_count_overflow.lance").unwrap();
    let test_uri = test_dir.path_str();
    let dataset = Dataset::open(&test_uri).await.unwrap();

    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), 66_049);

    let captions = batch["captions"]
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let values = captions
        .values()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    assert_eq!(values.len(), 16_416);
    assert_eq!(values.values().as_ref(), (0..16_416).collect::<Vec<_>>());

    let offsets = captions.value_offsets();
    assert_eq!(offsets[513], 16_416);
    assert!(offsets[513..].iter().all(|offset| *offset == 16_416));
    assert_eq!(captions.null_count(), 32_768);
    assert!(captions.is_valid(513));
    assert_eq!(captions.value(513).len(), 0);
    assert!(captions.is_null(514));
    assert!(captions.is_valid(66_047));
    assert_eq!(captions.value(66_047).len(), 0);
    assert!(captions.is_null(66_048));
}

// Helper: create a simple dataset with one fragment of `n` rows at the given URI.
async fn make_simple_dataset(uri: &str, n: i64) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..n))],
    )
    .unwrap();
    Dataset::write(RecordBatchIterator::new(vec![Ok(batch)], schema), uri, None)
        .await
        .unwrap()
}

#[tokio::test]
async fn test_migrate_to_stable_row_ids_basic() {
    // Create a dataset without stable row IDs (the default).
    let mut dataset = make_simple_dataset("memory://migrate_basic", 10).await;
    assert!(
        !dataset.manifest.uses_stable_row_ids(),
        "should not have stable row IDs yet"
    );

    // Append a second batch using InsertBuilder so we share the same object store.
    let schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(10..20))],
    )
    .unwrap();
    dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![batch2])
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);

    // Run the migration.
    dataset.migrate_to_stable_row_ids().await.unwrap();

    // FLAG_STABLE_ROW_IDS must be set in both reader and writer flags.
    assert_ne!(
        dataset.manifest.reader_feature_flags & FLAG_STABLE_ROW_IDS,
        0,
        "reader_feature_flags should have FLAG_STABLE_ROW_IDS"
    );
    assert_ne!(
        dataset.manifest.writer_feature_flags & FLAG_STABLE_ROW_IDS,
        0,
        "writer_feature_flags should have FLAG_STABLE_ROW_IDS"
    );
    assert!(dataset.manifest.uses_stable_row_ids());

    // All fragments must have row_id_meta set.
    for frag in dataset.manifest.fragments.iter() {
        assert!(
            frag.row_id_meta.is_some(),
            "fragment {} should have row_id_meta after migration",
            frag.id
        );
    }

    // next_row_id should equal the total number of physical rows (10 + 10 = 20).
    assert_eq!(dataset.manifest.next_row_id, 20);

    // Appending after migration should correctly assign row IDs from next_row_id.
    let batch3 = RecordBatch::try_new(
        Arc::new(ArrowSchema::from(dataset.schema())),
        vec![Arc::new(Int64Array::from_iter_values(20..25))],
    )
    .unwrap();
    let dataset_after_append = InsertBuilder::new(Arc::new(dataset.clone()))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![batch3])
        .await
        .unwrap();

    // The new fragment should also have row_id_meta.
    let new_frag = dataset_after_append.manifest.fragments.last().unwrap();
    assert!(
        new_frag.row_id_meta.is_some(),
        "new fragment after migration should have row_id_meta"
    );
    // next_row_id should have advanced by the 5 newly appended rows.
    assert_eq!(dataset_after_append.manifest.next_row_id, 25);

    dataset.validate().await.unwrap();
}

#[tokio::test]
async fn test_migrate_to_stable_row_ids_already_migrated() {
    // Create a dataset that already uses stable row IDs.
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..5))],
    )
    .unwrap();
    let write_params = WriteParams {
        enable_stable_row_ids: true,
        ..Default::default()
    };
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        "memory://already_migrated",
        Some(write_params),
    )
    .await
    .unwrap();

    assert!(dataset.manifest.uses_stable_row_ids());
    let version_before = dataset.manifest.version;

    // Calling migrate on an already-migrated dataset should be a no-op.
    dataset.migrate_to_stable_row_ids().await.unwrap();

    // Version must not have changed.
    assert_eq!(
        dataset.manifest.version, version_before,
        "migrate should be a no-op when already migrated"
    );
    assert!(dataset.manifest.uses_stable_row_ids());
}

#[tokio::test]
async fn test_migrate_to_stable_row_ids_empty() {
    // Create an empty dataset (schema-only, no fragments).
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let empty_reader = RecordBatchIterator::new(
        std::iter::empty::<std::result::Result<RecordBatch, arrow_schema::ArrowError>>(),
        schema.clone(),
    );
    let mut dataset = Dataset::write(empty_reader, "memory://migrate_empty", None)
        .await
        .unwrap();

    assert!(!dataset.manifest.uses_stable_row_ids());
    assert_eq!(dataset.get_fragments().len(), 0);

    // Migration on an empty dataset should succeed without error.
    dataset.migrate_to_stable_row_ids().await.unwrap();

    assert!(dataset.manifest.uses_stable_row_ids());
    assert_eq!(dataset.manifest.next_row_id, 0);
    dataset.validate().await.unwrap();
}

#[tokio::test]
async fn test_migrate_to_stable_row_ids_with_deletions() {
    // Create a single-fragment dataset of 10 rows then soft-delete 3 of them.
    let mut dataset = make_simple_dataset("memory://migrate_deletions", 10).await;
    dataset.delete("id < 3").await.unwrap();

    assert_eq!(dataset.count_rows(None).await.unwrap(), 7);
    assert_eq!(dataset.count_deleted_rows().await.unwrap(), 3);

    // physical_rows counts the pre-deletion slots; row IDs must cover all of
    // them so that the deleted rows' IDs are never reused.
    let physical_rows = dataset.get_fragments()[0].metadata.physical_rows.unwrap();
    assert_eq!(physical_rows, 10);

    dataset.migrate_to_stable_row_ids().await.unwrap();

    assert!(dataset.manifest.uses_stable_row_ids());
    assert!(dataset.manifest.fragments[0].row_id_meta.is_some());

    // next_row_id must equal physical_rows (10), not logical rows (7).
    assert_eq!(dataset.manifest.next_row_id, 10);

    dataset.validate().await.unwrap();
}

#[tokio::test]
async fn test_migrate_to_stable_row_ids_blocked_by_index() {
    // Create a 2-fragment dataset and build a BTree index on it.
    let mut dataset = make_simple_dataset("memory://btree_blocked", 10).await;
    let schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(10..20))],
    )
    .unwrap();
    dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![batch2])
        .await
        .unwrap();

    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("my_btree".to_string()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    // Migration must be rejected because the BTree index exists.
    let err = dataset
        .migrate_to_stable_row_ids()
        .await
        .expect_err("migration should fail when indexes exist");

    assert!(
        err.to_string().contains("my_btree"),
        "error should name the blocking index, got: {err}"
    );

    // After dropping the index the migration succeeds.
    dataset.drop_index("my_btree").await.unwrap();
    dataset.migrate_to_stable_row_ids().await.unwrap();
    assert!(dataset.manifest.uses_stable_row_ids());

    // Re-create the index and verify it works correctly.
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    let results = dataset
        .scan()
        .filter("id = 15")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(results.num_rows(), 1);
    let id_col = results["id"].as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(id_col.value(0), 15);
}

/// The migration numbers from the mark it is handed, so any manifest carrying a
/// non-zero one cannot reissue ids the earlier versions still hold.
#[rstest]
#[case::fresh(0, vec![0..4, 4..10])]
#[case::carried_mark(30, vec![30..34, 34..40])]
fn test_migration_allocates_from_the_given_mark(
    #[case] start: u64,
    #[case] expected: Vec<std::ops::Range<u64>>,
) {
    let mut fragments: Vec<Fragment> = [4usize, 6]
        .iter()
        .enumerate()
        .map(|(i, rows)| {
            let mut f = Fragment::new(i as u64);
            f.physical_rows = Some(*rows);
            f
        })
        .collect();

    let next = Dataset::assign_stable_row_ids_for_migration(&mut fragments, start).unwrap();
    assert_eq!(next, expected.last().unwrap().end);

    let sequences: Vec<Vec<u64>> = fragments
        .iter()
        .map(|f| {
            let RowIdMeta::Inline(data) = f.row_id_meta.as_ref().unwrap() else {
                panic!("migration writes inline row id meta");
            };
            read_row_ids(data).unwrap().iter().collect()
        })
        .collect();
    let expected: Vec<Vec<u64>> = expected.into_iter().map(|r| r.collect()).collect();
    assert_eq!(sequences, expected);
}
