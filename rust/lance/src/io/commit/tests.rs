// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
    time::Duration,
};

use arrow_array::types::Int32Type;
use arrow_array::{Int32Array, Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::future::join_all;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::Error;
use lance_core::datatypes::{Field, LANCE_UNENFORCED_PRIMARY_KEY, Schema};
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::{BatchCount, RowCount, array, gen_batch};
use lance_index::{IndexType, metrics::NoOpMetricsCollector};
use lance_linalg::distance::MetricType;
use lance_table::format::{IndexMetadata, pb};
use lance_table::io::commit::{
    CommitConfig, CommitError, CommitLease, CommitLock, ManifestWriter, RenameCommitHandler,
    UnsafeCommitHandler, commit_handler_from_url,
};
use lance_table::transaction::{Operation, resolve_arrow_field_ids};
use lance_table_runtime::commit::retry::DEFAULT_COMMIT_RETRY_TIMEOUT;
use lance_testing::datagen::generate_random_array;
use prost::Message;
use roaring::RoaringBitmap;

use super::*;

use crate::Dataset;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::cleanup::{CleanupPolicyBuilder, auto_cleanup_hook};
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::{
    CommitBuilder, DeleteBuilder, InsertBuilder, MergeInsertBuilder, UpdateBuilder, WhenMatched,
    WhenNotMatched, WriteMode, WriteParams,
};
use crate::index::vector::VectorIndexParams;
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

#[tokio::test]
async fn test_auto_cleanup_failure_preserves_successful_commit() {
    let mut dataset = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(2))
        .await
        .unwrap();
    dataset
        .update_config([("lance.auto_cleanup.interval", "invalid")])
        .await
        .unwrap();

    dataset.checkout_latest().await.unwrap();
    assert_eq!(dataset.version().version, 2);
    assert_eq!(
        dataset.manifest.config["lance.auto_cleanup.interval"],
        "invalid"
    );
    assert_eq!(dataset.count_rows(None).await.unwrap(), 4);
    let error = auto_cleanup_hook(&dataset, &dataset.manifest)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Cleanup { .. }));
    assert!(error.to_string().contains("lance.auto_cleanup.interval"));
}

#[rstest::rstest]
#[case::delete("delete")]
#[case::update("update")]
#[case::merge_insert("merge_insert")]
#[case::compact("compact")]
#[tokio::test]
async fn test_stale_mutation_preserves_deletes_after_cleanup(#[case] operation: &str) {
    let stale = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(4))
        .await
        .unwrap();
    let mut latest = stale.clone();
    latest.delete("id = 0").await.unwrap();
    let latest = CommitBuilder::new(Arc::new(latest))
        .execute(Transaction::new(
            2,
            Operation::ReserveFragments { num_fragments: 1 },
            None,
        ))
        .await
        .unwrap();
    let stats = latest
        .cleanup(
            CleanupPolicyBuilder::default()
                .versions(vec![1, 2])
                .unwrap()
                .build(),
        )
        .execute()
        .await
        .unwrap();
    assert_eq!(stats.old_versions, 2);
    assert_eq!(stale.version().version, 1);

    let result = match operation {
        "update" => {
            UpdateBuilder::new(Arc::new(stale))
                .update_where("id = 3")
                .unwrap()
                .set("id", "30")
                .unwrap()
                .build()
                .unwrap()
                .execute()
                .await
                .unwrap()
                .new_dataset
        }
        "delete" => {
            DeleteBuilder::new(Arc::new(stale), "id = 3")
                .execute()
                .await
                .unwrap()
                .new_dataset
        }
        "merge_insert" => {
            let batch = arrow_array::record_batch!(("id", Int32, [3])).unwrap();
            MergeInsertBuilder::try_new(Arc::new(stale), vec!["id".into()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .try_build()
                .unwrap()
                .execute_reader(RecordBatchIterator::new(
                    vec![Ok(batch.clone())],
                    batch.schema(),
                ))
                .await
                .unwrap()
                .0
        }
        "compact" => {
            let mut stale = stale;
            let error = compact_files(&mut stale, CompactionOptions::default(), None)
                .await
                .unwrap_err();
            assert!(
                matches!(error, Error::RetryableCommitConflict { .. }),
                "{error}"
            );
            assert!(error.to_string().contains("versions [2]"), "{error}");
            Arc::new(latest)
        }
        _ => unreachable!(),
    };
    assert_eq!(result.count_rows(Some("id = 0".into())).await.unwrap(), 0);
    assert_eq!(
        result.count_rows(Some("id = 3".into())).await.unwrap(),
        usize::from(matches!(operation, "merge_insert" | "compact"))
    );
    assert_eq!(
        result.count_rows(None).await.unwrap(),
        if operation == "delete" { 6 } else { 7 }
    );
    if operation == "update" {
        assert_eq!(result.count_rows(Some("id = 30".into())).await.unwrap(), 1);
    }
}

#[rstest::rstest]
#[case::partial("id = 3")]
#[case::whole_fragment("id < 4")]
#[tokio::test]
async fn test_staged_delete_rejects_cleaned_history(
    #[case] predicate: &str,
    #[values(false, true)] is_compaction: bool,
    #[values(false, true)] keep_read_version: bool,
) {
    let stale = Arc::new(
        gen_batch()
            .col("id", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(4))
            .await
            .unwrap(),
    );
    // Stage before cleanup, so a missing data file cannot mask the unsafe commit.
    let staged = DeleteBuilder::new(stale.clone(), predicate)
        .execute_uncommitted()
        .await
        .unwrap();
    let mut latest = stale.as_ref().clone();
    if is_compaction {
        compact_files(&mut latest, CompactionOptions::default(), None)
            .await
            .unwrap();
    } else {
        latest.delete("id = 0").await.unwrap();
    }
    let concurrent_version = latest.version().version;
    let latest = CommitBuilder::new(Arc::new(latest))
        .execute(Transaction::new(
            concurrent_version,
            Operation::ReserveFragments { num_fragments: 1 },
            None,
        ))
        .await
        .unwrap();
    let missing_versions = (2..=concurrent_version).collect::<Vec<_>>();
    let versions = (if keep_read_version { 2 } else { 1 }..=concurrent_version).collect::<Vec<_>>();
    let stats = latest
        .cleanup(
            CleanupPolicyBuilder::default()
                .versions(versions.clone())
                .unwrap()
                .build(),
        )
        .execute()
        .await
        .unwrap();
    assert_eq!(stats.old_versions, versions.len() as u64);

    let mut builder = CommitBuilder::new(stale);
    if let Some(affected_rows) = staged.affected_rows {
        builder = builder.with_affected_rows(affected_rows);
    }
    let error = builder.execute(staged.transaction).await.unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
    assert!(
        error
            .to_string()
            .contains(&format!("versions {missing_versions:?}")),
        "{error}"
    );
    assert!(error.to_string().contains("existing fragments"), "{error}");
    assert_eq!(
        latest.latest_version_id().await.unwrap(),
        concurrent_version + 1
    );
    assert_eq!(
        latest.count_rows(None).await.unwrap(),
        if is_compaction { 8 } else { 7 }
    );
}

#[rstest::rstest]
#[case::intact_history(&[])]
#[case::gap_read_retained(&[2])]
#[case::gap_read_removed(&[1, 2])]
#[tokio::test]
async fn test_insert_only_merge_insert_after_cleanup(
    #[case] cleaned_versions: &[u64],
    #[values(false, true)] is_staged: bool,
    #[values("id", "value")] on: &str,
) {
    let initial =
        arrow_array::record_batch!(("id", Int32, [0, 1, 2, 3]), ("value", Int32, [0, 1, 2, 3]))
            .unwrap();
    let mut fields = initial.schema().fields().to_vec();
    fields[0] = Arc::new(
        fields[0]
            .as_ref()
            .clone()
            .with_nullable(false)
            .with_metadata(HashMap::from([(
                LANCE_UNENFORCED_PRIMARY_KEY.to_string(),
                "true".to_string(),
            )])),
    );
    let initial = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(fields)),
        initial.columns().to_vec(),
    )
    .unwrap();
    let stale = Arc::new(
        crate::dataset::InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            })
            .execute(vec![initial])
            .await
            .unwrap(),
    );
    assert_eq!(stale.get_fragments().len(), 2);
    let upsert = |dataset| {
        MergeInsertBuilder::try_new(dataset, vec![on.into()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
    };
    let batch = if on == "id" {
        arrow_array::record_batch!(("id", Int32, [100]), ("value", Int32, [101])).unwrap()
    } else {
        arrow_array::record_batch!(("id", Int32, [101]), ("value", Int32, [100])).unwrap()
    };
    // Stage before the concurrent insert and cleanup, as a separate writer would.
    let staged = if is_staged {
        let staged = upsert(stale.clone())
            .execute_uncommitted_batches(vec![batch.clone()])
            .await
            .unwrap();
        let Operation::Update {
            updated_fragments,
            removed_fragment_ids,
            inserted_rows_filter,
            ..
        } = &staged.transaction.operation
        else {
            panic!("expected an Update transaction");
        };
        assert!(updated_fragments.is_empty());
        assert!(removed_fragment_ids.is_empty());
        assert_eq!(inserted_rows_filter.is_some(), on == "id");
        Some(staged)
    } else {
        None
    };
    let concurrent =
        arrow_array::record_batch!(("id", Int32, [100]), ("value", Int32, [100])).unwrap();
    let (latest, _) = upsert(stale.clone())
        .execute_batches(vec![concurrent])
        .await
        .unwrap();
    let latest = CommitBuilder::new(latest)
        .execute(Transaction::new(
            2,
            Operation::ReserveFragments { num_fragments: 1 },
            None,
        ))
        .await
        .unwrap();
    if !cleaned_versions.is_empty() {
        let stats = latest
            .cleanup(
                CleanupPolicyBuilder::default()
                    .versions(cleaned_versions.to_vec())
                    .unwrap()
                    .build(),
            )
            .execute()
            .await
            .unwrap();
        assert_eq!(stats.old_versions, cleaned_versions.len() as u64);
    }

    let result = if let Some(staged) = staged {
        let mut builder = CommitBuilder::new(stale);
        if let Some(affected_rows) = staged.affected_rows {
            builder = builder.with_affected_rows(affected_rows);
        }
        let result = builder.execute(staged.transaction).await;
        if on == "id" {
            let error = result.unwrap_err();
            assert!(
                matches!(error, Error::RetryableCommitConflict { .. }),
                "{error}"
            );
            let message = if cleaned_versions.is_empty() {
                "preempted by concurrent transaction"
            } else {
                "inserted keys"
            };
            assert!(error.to_string().contains(message), "{error}");
            if !cleaned_versions.is_empty() {
                assert!(error.to_string().contains("versions [2]"), "{error}");
            }
            assert_eq!(latest.latest_version_id().await.unwrap(), 3);
            assert_eq!(latest.count_rows(None).await.unwrap(), 5);
            assert_eq!(latest.count_rows(Some("id = 100".into())).await.unwrap(), 1);
            return;
        }
        result.unwrap()
    } else {
        let (result, stats) = upsert(stale).execute_batches(vec![batch]).await.unwrap();
        assert_eq!(stats.num_attempts, if on == "id" { 2 } else { 1 });
        result.as_ref().clone()
    };
    assert_eq!(result.version().version, 4);
    assert_eq!(
        result
            .count_rows(Some(format!("{on} = 100")))
            .await
            .unwrap(),
        if on == "id" { 1 } else { 2 }
    );
    assert_eq!(
        result
            .count_rows(Some(if on == "id" {
                "id = 100 AND value = 101".into()
            } else {
                "id = 101 AND value = 100".into()
            }))
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        result.count_rows(None).await.unwrap(),
        if on == "id" { 5 } else { 6 }
    );
}

#[rstest::rstest]
#[case::append(false, false)]
#[case::index_after_delete(true, false)]
#[case::index_after_add_columns(true, true)]
#[tokio::test]
async fn test_safe_commit_survives_cleaned_history(
    #[case] is_index: bool,
    #[case] is_column_add: bool,
) {
    let stale = Arc::new(
        gen_batch()
            .col("id", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(4))
            .await
            .unwrap(),
    );
    let transaction = if is_index {
        let params = lance_index::scalar::ScalarIndexParams::for_builtin(
            lance_index::scalar::BuiltinIndexType::BTree,
        );
        let mut index_dataset = stale.as_ref().clone();
        let index = crate::index::CreateIndexBuilder::new(
            &mut index_dataset,
            &["id"],
            IndexType::BTree,
            &params,
        )
        .execute_uncommitted()
        .await
        .unwrap();
        Transaction::new(
            1,
            Operation::CreateIndex {
                new_indices: vec![index],
                removed_indices: vec![],
            },
            None,
        )
    } else {
        let batch = arrow_array::record_batch!(("id", Int32, [100])).unwrap();
        crate::dataset::InsertBuilder::new(stale.clone())
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute_uncommitted(vec![batch])
            .await
            .unwrap()
    };
    let mut latest = stale.as_ref().clone();
    if is_column_add {
        latest
            .add_columns(
                crate::dataset::NewColumnTransform::SqlExpressions(vec![(
                    "extra".into(),
                    "id * 2".into(),
                )]),
                None,
                None,
            )
            .await
            .unwrap();
    } else {
        latest.delete("id = 0").await.unwrap();
    }
    let latest = CommitBuilder::new(Arc::new(latest))
        .execute(Transaction::new(
            2,
            Operation::ReserveFragments { num_fragments: 1 },
            None,
        ))
        .await
        .unwrap();
    latest
        .cleanup(
            CleanupPolicyBuilder::default()
                .versions(vec![1, 2])
                .unwrap()
                .build(),
        )
        .execute()
        .await
        .unwrap();

    let result = CommitBuilder::new(stale)
        .execute(transaction)
        .await
        .unwrap();
    assert_eq!(result.version().version, 4);
    assert_eq!(
        result.count_rows(Some("id = 0".into())).await.unwrap(),
        usize::from(is_column_add)
    );
    assert_eq!(
        result.count_rows(None).await.unwrap(),
        if is_index {
            7 + usize::from(is_column_add)
        } else {
            8
        }
    );
    if is_index {
        let indices = result.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(
            indices[0].fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([0u32, 1])
        );
        assert_eq!(result.count_rows(Some("id = 3".into())).await.unwrap(), 1);
    }
}

#[rstest::rstest]
#[case::drop_columns_after_add_columns("add_columns")]
#[case::append_after_cast("cast")]
#[case::append_null_after_set_not_null("set_not_null")]
#[case::append_after_overwrite_reusing_field_ids("overwrite")]
#[case::append_after_overwrite_swapping_same_type_columns("overwrite_swap")]
#[case::append_after_overwrite_swapping_same_type_structs("overwrite_nested_swap")]
#[case::append_after_overwrite_struct_to_list("overwrite_struct_to_list")]
#[case::append_after_overwrite_list_to_struct("overwrite_list_to_struct")]
#[tokio::test]
async fn test_schema_dependent_commit_rejects_cleaned_history(#[case] concurrent: &str) {
    let (initial, staged) = match concurrent {
        "overwrite_nested_swap" => (
            struct_pair_batch(["a", "b"], [0, 1]),
            struct_pair_batch(["a", "b"], [7, 70]),
        ),
        "overwrite_struct_to_list" => (container_batch(false, 1), container_batch(false, 7)),
        "overwrite_list_to_struct" => (container_batch(true, 1), container_batch(true, 7)),
        _ => (
            arrow_array::record_batch!(
                ("id", Int32, [0, 1, 2, 3]),
                ("x", Int32, [Some(0), Some(1), Some(2), Some(3)])
            )
            .unwrap(),
            arrow_array::record_batch!(("id", Int32, [4]), ("x", Int32, [None])).unwrap(),
        ),
    };
    let stale = Arc::new(
        crate::dataset::InsertBuilder::new("memory://")
            .execute(vec![initial])
            .await
            .unwrap(),
    );
    let staged_append = crate::dataset::InsertBuilder::new(stale.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted(vec![staged])
        .await
        .unwrap();
    let mut latest = stale.as_ref().clone();
    match concurrent {
        "add_columns" => latest
            .add_columns(
                crate::dataset::NewColumnTransform::SqlExpressions(vec![(
                    "extra".into(),
                    "id * 2".into(),
                )]),
                None,
                None,
            )
            .await
            .unwrap(),
        "cast" => latest
            .alter_columns(&[
                crate::dataset::ColumnAlteration::new("x".into()).cast_to(DataType::Int64)
            ])
            .await
            .unwrap(),
        "set_not_null" => latest
            .alter_columns(&[crate::dataset::ColumnAlteration::new("x".into()).set_nullable(false)])
            .await
            .unwrap(),
        "overwrite" => {
            latest = crate::dataset::InsertBuilder::new(Arc::new(latest))
                .with_params(&WriteParams {
                    mode: WriteMode::Overwrite,
                    ..Default::default()
                })
                .execute(vec![
                    arrow_array::record_batch!(("id", Utf8, ["a"]), ("x", Utf8, ["b"])).unwrap(),
                ])
                .await
                .unwrap();
        }
        // The reused field ids keep their types but now name the other column.
        "overwrite_swap" => {
            latest = crate::dataset::InsertBuilder::new(Arc::new(latest))
                .with_params(&WriteParams {
                    mode: WriteMode::Overwrite,
                    ..Default::default()
                })
                .execute(vec![
                    arrow_array::record_batch!(("x", Int32, [Some(5)]), ("id", Int32, [6]))
                        .unwrap(),
                ])
                .await
                .unwrap();
        }
        // Only the parents swap: both written leaves are still `x: int32`.
        "overwrite_nested_swap" => {
            latest = crate::dataset::InsertBuilder::new(Arc::new(latest))
                .with_params(&WriteParams {
                    mode: WriteMode::Overwrite,
                    ..Default::default()
                })
                .execute(vec![struct_pair_batch(["b", "a"], [5, 6])])
                .await
                .unwrap();
        }
        // The leaf keeps its path `a.item` and type under another container kind.
        "overwrite_struct_to_list" | "overwrite_list_to_struct" => {
            latest = crate::dataset::InsertBuilder::new(Arc::new(latest))
                .with_params(&WriteParams {
                    mode: WriteMode::Overwrite,
                    ..Default::default()
                })
                .execute(vec![container_batch(
                    concurrent == "overwrite_struct_to_list",
                    20,
                )])
                .await
                .unwrap();
        }
        _ => unreachable!(),
    }
    let latest = CommitBuilder::new(Arc::new(latest))
        .execute(Transaction::new(
            2,
            Operation::ReserveFragments { num_fragments: 1 },
            None,
        ))
        .await
        .unwrap();
    latest
        .cleanup(
            CleanupPolicyBuilder::default()
                .versions(vec![1, 2])
                .unwrap()
                .build(),
        )
        .execute()
        .await
        .unwrap();

    let result = if concurrent == "add_columns" {
        let mut stale = stale.as_ref().clone();
        stale.drop_columns(&["x"]).await
    } else {
        CommitBuilder::new(stale)
            .execute(staged_append)
            .await
            .map(|_| ())
    };
    let error = result.unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("versions [2]"), "{error}");
    assert_eq!(latest.latest_version_id().await.unwrap(), 3);
}

/// One row with two `struct<x: int32>` columns.
fn struct_pair_batch(names: [&str; 2], values: [i32; 2]) -> RecordBatch {
    let column = |value: i32| -> arrow_array::ArrayRef {
        Arc::new(arrow_array::StructArray::from(vec![(
            Arc::new(ArrowField::new("x", DataType::Int32, true)),
            Arc::new(Int32Array::from(vec![value])) as arrow_array::ArrayRef,
        )]))
    };
    RecordBatch::try_from_iter([(names[0], column(values[0])), (names[1], column(values[1]))])
        .unwrap()
}

/// One row with an `a` column that is either `list<item: int32>` or
/// `struct<item: int32>`.
fn container_batch(is_list: bool, value: i32) -> RecordBatch {
    let column: arrow_array::ArrayRef = if is_list {
        Arc::new(
            arrow_array::ListArray::from_iter_primitive::<Int32Type, _, _>([Some(vec![
                Some(value),
                Some(value + 1),
            ])]),
        )
    } else {
        Arc::new(arrow_array::StructArray::from(vec![(
            Arc::new(ArrowField::new("item", DataType::Int32, true)),
            Arc::new(Int32Array::from(vec![value])) as arrow_array::ArrayRef,
        )]))
    };
    RecordBatch::try_from_iter([("a", column)]).unwrap()
}

async fn test_commit_handler(handler: Arc<dyn CommitHandler>, should_succeed: bool) {
    // Create a dataset, passing handler as commit handler
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "x",
        DataType::Int64,
        false,
    )]));
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(data)], schema);

    let options = WriteParams {
        commit_handler: Some(handler),
        ..Default::default()
    };
    let dataset = Dataset::write(reader, "memory://test", Some(options))
        .await
        .unwrap();

    // Create 10 concurrent tasks to write into the table
    // Record how many succeed and how many fail
    let tasks = (0..10).map(|_| {
        let mut dataset = dataset.clone();
        tokio::task::spawn(async move {
            dataset
                .delete("x = 2")
                .await
                .map(|_| dataset.manifest.version)
        })
    });

    let task_results: Vec<Option<u64>> = join_all(tasks)
        .await
        .iter()
        .map(|res| match res {
            Ok(Ok(version)) => Some(*version),
            _ => None,
        })
        .collect();

    let num_successes = task_results.iter().filter(|x| x.is_some()).count();
    let distinct_results: HashSet<_> = task_results.iter().filter_map(|x| x.as_ref()).collect();

    if should_succeed {
        assert_eq!(
            num_successes,
            distinct_results.len(),
            "Expected no two tasks to succeed for the same version. Got {:?}",
            task_results
        );
    } else {
        // All we can promise here is at least one tasks succeeds, but multiple
        // could in theory.
        assert!(num_successes >= distinct_results.len(),);
    }
}

#[tokio::test]
async fn test_rename_commit_handler() {
    // Rename is default for memory
    let handler = Arc::new(RenameCommitHandler);
    test_commit_handler(handler, true).await;
}

/// The rename handler's staging object is not the object readers resolve:
/// a store without atomic rename publishes by copy, so the staging ETag
/// need not describe the final path. The location returned by a successful
/// commit must describe the published object, or writer and reader disagree
/// on the manifest generation token.
#[tokio::test]
async fn test_rename_commit_handler_reports_published_e_tag() {
    let temp_dir = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "x",
        DataType::Int64,
        false,
    )]));
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(data)], schema),
        temp_dir.as_str(),
        Some(WriteParams {
            commit_handler: Some(Arc::new(RenameCommitHandler)),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let published = dataset.manifest_location();
    let resolved = dataset
        .table_state
        .commit_handler
        .resolve_version_location(
            &dataset.table_state.base,
            dataset.manifest.version,
            &dataset.table_state.object_store.inner,
        )
        .await
        .unwrap();

    assert!(
        published.e_tag.is_some(),
        "the local store reports a generation token for the published manifest"
    );
    assert_eq!(
        published.e_tag, resolved.e_tag,
        "the commit must report the published object's generation token"
    );
}

#[tokio::test]
async fn test_custom_commit() {
    #[derive(Debug)]
    struct CustomCommitHandler {
        locked_version: Arc<Mutex<Option<u64>>>,
    }

    struct CustomCommitLease {
        version: u64,
        locked_version: Arc<Mutex<Option<u64>>>,
    }

    #[async_trait::async_trait]
    impl CommitLock for CustomCommitHandler {
        type Lease = CustomCommitLease;

        async fn lock(&self, version: u64) -> std::result::Result<Self::Lease, CommitError> {
            let mut locked_version = self.locked_version.lock().unwrap();
            if locked_version.is_some() {
                // Already locked
                return Err(CommitError::CommitConflict);
            }

            // Lock the version
            *locked_version = Some(version);

            Ok(CustomCommitLease {
                version,
                locked_version: self.locked_version.clone(),
            })
        }
    }

    #[async_trait::async_trait]
    impl CommitLease for CustomCommitLease {
        async fn release(&self, _success: bool) -> std::result::Result<(), CommitError> {
            let mut locked_version = self.locked_version.lock().unwrap();
            if *locked_version != Some(self.version) {
                // Already released
                return Err(CommitError::CommitConflict);
            }

            // Release the version
            *locked_version = None;

            Ok(())
        }
    }

    let locked_version = Arc::new(Mutex::new(None));
    let handler = Arc::new(CustomCommitHandler { locked_version });
    test_commit_handler(handler, true).await;
}

#[derive(Debug)]
struct InjectForeignCommitHandler {
    foreign: Mutex<Option<(Manifest, lance_table::format::Transaction)>>,
}

#[async_trait::async_trait]
impl CommitHandler for InjectForeignCommitHandler {
    fn is_version_not_found_definitive(&self) -> bool {
        true
    }

    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<lance_table::format::Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        let foreign = self.foreign.lock().unwrap().take();
        if let Some((mut foreign_manifest, foreign_transaction)) = foreign {
            foreign_manifest.version = manifest.version;
            RenameCommitHandler
                .commit(
                    &mut foreign_manifest,
                    None,
                    base_path,
                    object_store,
                    manifest_writer,
                    naming_scheme,
                    Some(foreign_transaction),
                )
                .await?;
            return Err(CommitError::CommitConflict);
        }

        RenameCommitHandler
            .commit(
                manifest,
                indices,
                base_path,
                object_store,
                manifest_writer,
                naming_scheme,
                transaction,
            )
            .await
    }
}

fn inject_foreign_commit_handler(
    manifest: Manifest,
    transaction: &Transaction,
) -> Arc<dyn CommitHandler> {
    let transaction = lance_table::format::Transaction::try_from(transaction).unwrap();
    Arc::new(InjectForeignCommitHandler {
        foreign: Mutex::new(Some((manifest, transaction))),
    })
}

#[tokio::test]
async fn raw_arrow_project_retry_preserves_identity_and_metadata() {
    let tmp = TempStrDir::default();
    let uri = tmp.as_str();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(
            vec![Ok(simple_batch(&simple_schema(), vec![1, 2, 3]))],
            simple_schema(),
        ),
        uri,
        None,
    )
    .await
    .unwrap();
    dataset.migrate_to_non_reusable_field_ids().await.unwrap();

    let mut foreign_manifest = dataset.manifest.as_ref().clone();
    foreign_manifest.max_fragment_id = Some(foreign_manifest.max_fragment_id.unwrap_or(0) + 1);
    let foreign_transaction = Transaction::new(
        dataset.version().version,
        Operation::ReserveFragments { num_fragments: 1 },
        None,
    );
    let handler = inject_foreign_commit_handler(foreign_manifest, &foreign_transaction);

    let raw_schema = Schema {
        fields: vec![Field::new_arrow("x", DataType::Int32, false).unwrap()],
        metadata: HashMap::from([("input-note".to_string(), "project".to_string())]),
    };
    let mut expected_schema = raw_schema.clone();
    expected_schema.fields[0].id = 0;
    let mut operation = Operation::Project {
        schema: raw_schema,
        preserves_nullability: true,
    };
    resolve_arrow_field_ids(Some(&dataset.manifest), &mut operation).unwrap();

    let committed = CommitBuilder::new(Arc::new(dataset.clone()))
        .with_commit_handler(handler)
        .execute(Transaction::new(dataset.version().version, operation, None))
        .await
        .unwrap();

    assert_eq!(committed.schema(), &expected_schema);
}

#[rstest::rstest]
#[tokio::test]
async fn raw_arrow_retry_rebinds_after_allocator_advance(#[values(false, true)] overwrite: bool) {
    let tmp = TempStrDir::default();
    let uri = tmp.as_str();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(
            vec![Ok(simple_batch(&simple_schema(), vec![1, 2, 3]))],
            simple_schema(),
        ),
        uri,
        None,
    )
    .await
    .unwrap();
    dataset.migrate_to_non_reusable_field_ids().await.unwrap();

    let mut foreign_manifest = dataset.manifest.as_ref().clone();
    foreign_manifest.max_fragment_id = Some(foreign_manifest.max_fragment_id.unwrap_or(0) + 1);
    foreign_manifest.max_allocated_field_id = Some(foreign_manifest.max_field_id() + 1);
    let foreign_transaction = Transaction::new(
        dataset.version().version,
        Operation::ReserveFragments { num_fragments: 1 },
        None,
    );
    let handler = inject_foreign_commit_handler(foreign_manifest, &foreign_transaction);

    let mut merged_fragment = dataset.manifest.fragments[0].clone();
    let mut new_file = merged_fragment.files[0].clone();
    new_file.path = "retry-new-column.lance".to_string();
    new_file.fields = Arc::from([1]);
    merged_fragment.files.push(new_file);
    let raw_schema = Schema {
        fields: vec![
            Field::new_arrow("x", DataType::Int32, false).unwrap(),
            Field::new_arrow("new_column", DataType::Int32, true).unwrap(),
        ],
        metadata: HashMap::new(),
    };
    let mut expected_schema = raw_schema.clone();
    expected_schema.fields[0].id = if overwrite { 2 } else { 0 };
    expected_schema.fields[1].id = if overwrite { 3 } else { 2 };
    let mut operation = if overwrite {
        Operation::Overwrite {
            fragments: vec![merged_fragment],
            schema: raw_schema,
            config_upsert_values: None,
            initial_bases: None,
        }
    } else {
        Operation::Merge {
            fragments: vec![merged_fragment],
            schema: raw_schema,
            preserves_nullability: true,
        }
    };
    resolve_arrow_field_ids(Some(&dataset.manifest), &mut operation).unwrap();

    let committed = CommitBuilder::new(Arc::new(dataset.clone()))
        .with_commit_handler(handler)
        .execute(Transaction::new(dataset.version().version, operation, None))
        .await
        .unwrap();

    assert_eq!(committed.schema(), &expected_schema);
    assert_eq!(
        committed.manifest.max_allocated_field_id,
        Some(if overwrite { 3 } else { 2 })
    );
    assert_eq!(
        committed.manifest.fragments[0].files[0].fields.as_ref(),
        &[expected_schema.fields[0].id]
    );
    assert_eq!(
        committed.manifest.fragments[0].files[1].fields.as_ref(),
        &[expected_schema.fields[1].id]
    );
}

#[tokio::test]
async fn test_unsafe_commit_handler() {
    let handler = Arc::new(UnsafeCommitHandler);
    test_commit_handler(handler, false).await;
}

#[tokio::test]
async fn test_roundtrip_transaction_file() {
    let object_store = ObjectStore::memory();
    let base_path = Path::from("test");
    let transaction = Transaction::new(
        42,
        Operation::Append { fragments: vec![] },
        Some("hello world".to_string()),
    );

    let file_name = write_transaction_file(
        &object_store,
        &base_path,
        &pb::Transaction::try_from(&transaction).unwrap(),
        None,
    )
    .await
    .unwrap();
    let read_transaction = read_transaction_file(&object_store, &base_path, &file_name)
        .await
        .unwrap();

    assert_eq!(transaction.read_version, read_transaction.read_version);
    assert_eq!(transaction.uuid, read_transaction.uuid);
    assert!(matches!(
        read_transaction.operation,
        Operation::Append { .. }
    ));
    assert_eq!(transaction.tag, read_transaction.tag);
}

#[tokio::test]
async fn test_concurrent_create_index() {
    // Create a table with two vector columns
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    let dimension = 16;
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(
            "vector1",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                dimension,
            ),
            false,
        ),
        ArrowField::new(
            "vector2",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                dimension,
            ),
            false,
        ),
    ]));
    let float_arr = generate_random_array(512 * dimension as usize);
    let vectors = Arc::new(
        <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
            float_arr, dimension,
        )
        .unwrap(),
    );
    let batches =
        vec![RecordBatch::try_new(schema.clone(), vec![vectors.clone(), vectors.clone()]).unwrap()];

    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write(reader, test_uri, None).await.unwrap();
    dataset.validate().await.unwrap();

    // From initial version, concurrently call create index 3 times,
    // two of which will be for the same column.
    let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 50);
    let futures: Vec<_> = ["vector1", "vector1", "vector2"]
        .iter()
        .map(|col_name| {
            let mut dataset = dataset.clone();
            let params = params.clone();
            tokio::spawn(async move {
                dataset
                    .create_index(&[col_name], IndexType::Vector, None, &params, true)
                    .await
            })
        })
        .collect();

    let results = join_all(futures).await;
    let success_count = results
        .iter()
        .filter(|result| matches!(result, Ok(Ok(_))))
        .count();
    let retryable_count = results
        .iter()
        .filter(|result| matches!(result, Ok(Err(Error::RetryableCommitConflict { .. }))))
        .count();
    assert_eq!(success_count, 2, "{results:?}");
    assert_eq!(retryable_count, 1, "{results:?}");

    // Validate that each version has the anticipated number of indexes
    let dataset = dataset.checkout_version(1).await.unwrap();
    assert!(dataset.load_indices().await.unwrap().is_empty());

    let dataset = dataset.checkout_version(2).await.unwrap();
    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);

    let dataset = dataset.checkout_version(3).await.unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert!(!indices.is_empty() && indices.len() <= 2);

    // At this point, we have created two indices. If they are both for the same column,
    // it must be vector1 and not vector2.
    if indices.len() == 2 {
        let mut fields: Vec<i32> = indices.iter().flat_map(|i| i.fields.clone()).collect();
        fields.sort();
        assert_eq!(fields, vec![0, 1]);
    } else {
        assert_eq!(indices[0].fields, vec![0]);
    }

    assert!(dataset.checkout_version(4).await.is_err());
}

/// Every commit runs `migrate_indices`, and recalculating a missing
/// `fragment_bitmap` there means opening the index. An index this build
/// cannot open must not take the write path down with it: the dataset would
/// be unwritable, not merely unreadable, and every later commit would fail
/// the same way.
#[tokio::test]
async fn test_commit_survives_an_index_it_cannot_open() {
    use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
    use lance_table::io::manifest::read_manifest_indexes;

    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    let reader = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("payload", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    // The readable companion is what makes the difference visible: with a
    // single index, "carried through the one it cannot open" and "stopped
    // recalculating altogether" answer every assertion below the same way.
    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    for column in ["id", "payload"] {
        dataset
            .create_index_builder(&[column], IndexType::BTree, &btree_params)
            .name(format!("{column}_idx"))
            .await
            .unwrap();
    }

    let broken = dataset.load_index_by_name("id_idx").await.unwrap().unwrap();
    dataset
        .table_state
        .object_store
        .remove_dir_all(dataset.indices_dir().join(broken.uuid.to_string()))
        .await
        .unwrap();

    // Reopened so the fixture is judged on what is on disk rather than on
    // what this process still holds from building the index.
    let mut dataset = Dataset::open(test_uri).await.unwrap();
    assert!(
        dataset
            .open_generic_index("id", &broken.uuid, &NoOpMetricsCollector)
            .await
            .is_err(),
        "the fixture is supposed to leave an index this build cannot open"
    );

    // Migration recalculates a bitmap that is missing, and no current writer
    // emits one - untrained indices get an empty bitmap, not none at all - so
    // the state an old manifest arrives in is set here by hand.
    let indices = read_manifest_indexes(
        &dataset.table_state.object_store,
        &dataset.table_state.manifest_location,
        &dataset.manifest,
    )
    .await
    .unwrap();
    let without_bitmaps = indices
        .iter()
        .map(|index| IndexMetadata {
            fragment_bitmap: None,
            ..index.clone()
        })
        .collect::<Vec<_>>();
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: without_bitmaps,
            removed_indices: indices,
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();

    // And an unrelated commit after it, since the missing bitmap is now what
    // the manifest holds and migration retries on every commit.
    dataset.delete("false").await.unwrap();

    let migrated = read_manifest_indexes(
        &dataset.table_state.object_store,
        &dataset.table_state.manifest_location,
        &dataset.manifest,
    )
    .await
    .unwrap();
    let coverage = |name: &str| {
        migrated
            .iter()
            .find(|index| index.name == name)
            .unwrap_or_else(|| panic!("no index named {name} in the manifest"))
            .fragment_bitmap
            .as_ref()
            .map(|bitmap| bitmap.iter().collect::<Vec<_>>())
    };
    assert_eq!(
        coverage("id_idx"),
        None,
        "an index that cannot be opened must report unknown coverage"
    );
    assert_eq!(
        coverage("payload_idx"),
        Some(vec![0]),
        "an index that opens must still have its coverage recalculated"
    );
}

#[tokio::test]
async fn test_load_and_sort_new_transactions() {
    // Create a dataset
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(10))
        .await
        .unwrap();

    // Create 100 small UpdateConfig transactions
    for i in 0..100 {
        dataset
            .update_config(vec![(format!("key_{}", i), format!("value_{}", i))])
            .await
            .unwrap();
    }

    // Now load the dataset at version 1 and check that load_and_sort_new_transactions
    // returns transactions in order
    let dataset_v1 = dataset.checkout_version(1).await.unwrap();
    let (_, transactions) =
        load_and_sort_new_transactions(&dataset_v1, TransactionReadSource::SessionCache)
            .await
            .unwrap();

    // Verify transactions are sorted by version
    let versions: Vec<u64> = transactions.iter().map(|(v, _)| *v).collect();
    for i in 1..versions.len() {
        assert!(
            versions[i] > versions[i - 1],
            "Transactions not in order: version {} came after version {}",
            versions[i],
            versions[i - 1]
        );
    }

    // Also verify we have exactly 100 transactions (versions 2-101)
    assert_eq!(transactions.len(), 100);
    assert_eq!(versions.first(), Some(&2));
    assert_eq!(versions.last(), Some(&101));
}

#[tokio::test]
async fn test_concurrent_writes() {
    // Test concurrent appends - all should succeed
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));

    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone()),
        test_uri,
        None,
    )
    .await
    .unwrap();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();

    let futures: Vec<_> = (0..5)
        .map(|_| {
            let batch = batch.clone();
            let schema = schema.clone();
            let uri = test_uri.to_string();
            tokio::spawn(async move {
                let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
                Dataset::write(
                    reader,
                    &uri,
                    Some(WriteParams {
                        mode: WriteMode::Append,
                        ..Default::default()
                    }),
                )
                .await
            })
        })
        .collect();
    let results = join_all(futures).await;

    for result in results {
        assert!(matches!(result, Ok(Ok(_))), "{:?}", result);
    }

    let dataset = dataset.checkout_version(6).await.unwrap();
    assert_eq!(dataset.get_fragments().len(), 5);
    dataset.validate().await.unwrap()
}

#[tokio::test]
async fn test_restore_does_not_decrease_max_fragment_id() {
    let reader = gen_batch()
        .col("i", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(3), BatchCount::from(1));
    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    // Append a few times to advance max_fragment_id and create newer versions.
    for _ in 0..2 {
        let reader = gen_batch()
            .col("i", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(3), BatchCount::from(1));
        dataset.append(reader, None).await.unwrap();
    }

    let latest_max = dataset.manifest.max_fragment_id().unwrap_or(0);

    // Restore an earlier version (version 1) as the latest.
    let mut dataset_v1 = dataset.checkout_version(1).await.unwrap();
    dataset_v1.restore().await.unwrap();

    // After restore, max_fragment_id should not decrease compared to the latest value before restore.
    let restored_max = dataset_v1.manifest.max_fragment_id().unwrap_or(0);
    assert!(
        restored_max >= latest_max,
        "max_fragment_id should not decrease on restore: before={}, after={}",
        latest_max,
        restored_max
    );
}

async fn get_empty_dataset() -> (TempStrDir, Dataset) {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));

    let ds = Dataset::write(
        RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone()),
        test_uri,
        None,
    )
    .await
    .unwrap();
    (test_dir, ds)
}

#[tokio::test]
async fn every_commit_path_checks_required_writer_flags() {
    let (_test_dir, mut dataset) = get_empty_dataset().await;
    Arc::make_mut(&mut dataset.manifest).writer_feature_flags |=
        lance_table::feature_flags::FLAG_UNKNOWN;

    let err = dataset
        .update_config(HashMap::from([(
            "test.required-writer-gate".to_string(),
            Some("value".to_string()),
        )]))
        .await
        .unwrap_err();

    assert!(matches!(err, Error::NotSupported { .. }), "{err}");
}

#[tokio::test]
async fn test_good_concurrent_config_writes() {
    let (_tmpdir, dataset) = get_empty_dataset().await;
    let original_num_config_keys = dataset.manifest.config.len();

    // Test successful concurrent insert config operations
    let futures: Vec<_> = ["key1", "key2", "key3", "key4", "key5"]
        .iter()
        .map(|key| {
            let mut dataset = dataset.clone();
            tokio::spawn(async move {
                dataset
                    .update_config(HashMap::from([(
                        key.to_string(),
                        Some("value".to_string()),
                    )]))
                    .await
            })
        })
        .collect();
    let results = join_all(futures).await;

    // Assert all succeeded
    for result in results {
        assert!(matches!(result, Ok(Ok(_))), "{:?}", result);
    }

    let dataset = dataset.checkout_version(6).await.unwrap();
    assert_eq!(dataset.manifest.config.len(), 5 + original_num_config_keys);

    dataset.validate().await.unwrap();

    // Test successful concurrent delete operations. If multiple delete
    // operations attempt to delete the same key, they are all successful.
    let futures: Vec<_> = ["key1", "key1", "key1", "key2", "key2"]
        .iter()
        .map(|key| {
            let mut dataset = dataset.clone();
            tokio::spawn(async move {
                dataset
                    .update_config(HashMap::from([(key.to_string(), None)]))
                    .await
            })
        })
        .collect();
    let results = join_all(futures).await;

    // Assert all succeeded
    for result in results {
        assert!(matches!(result, Ok(Ok(_))), "{:?}", result);
    }

    let dataset = dataset.checkout_version(11).await.unwrap();

    // There are now two fewer keys
    assert_eq!(dataset.manifest.config.len(), 3 + original_num_config_keys);

    dataset.validate().await.unwrap()
}

#[tokio::test]
async fn test_bad_concurrent_config_writes() {
    // If two concurrent insert config operations occur for the same key, a
    // `CommitConflict` should be returned
    let (_tmpdir, dataset) = get_empty_dataset().await;

    let futures: Vec<_> = ["key1", "key1", "key2", "key3", "key4"]
        .iter()
        .map(|key| {
            let mut dataset = dataset.clone();
            tokio::spawn(async move {
                dataset
                    .update_config(HashMap::from([(
                        key.to_string(),
                        Some("value".to_string()),
                    )]))
                    .await
            })
        })
        .collect();

    let results = join_all(futures).await;

    // Assert that either the first or the second operation fails
    let mut first_operation_failed = false;
    for (i, result) in results.into_iter().enumerate() {
        let result = result.unwrap();
        match i {
            0 => {
                if result.is_err() {
                    first_operation_failed = true;
                    assert!(
                        matches!(&result, &Err(Error::IncompatibleTransaction { .. })),
                        "{:?}",
                        result,
                    );
                }
            }
            1 => match first_operation_failed {
                true => assert!(result.is_ok(), "{:?}", result),
                false => {
                    assert!(
                        matches!(&result, &Err(Error::IncompatibleTransaction { .. })),
                        "{:?}",
                        result,
                    );
                }
            },
            _ => assert!(result.is_ok(), "{:?}", result),
        }
    }
}

/// A CommitHandler that always fails with OtherError, used to simulate
/// a manifest write failure so we can verify orphaned transaction files
/// are cleaned up.
#[derive(Debug)]
struct FailingCommitHandler;

#[async_trait::async_trait]
impl CommitHandler for FailingCommitHandler {
    fn is_version_not_found_definitive(&self) -> bool {
        true
    }

    async fn commit(
        &self,
        _manifest: &mut Manifest,
        _indices: Option<Vec<IndexMetadata>>,
        _base_path: &Path,
        _object_store: &ObjectStore,
        _manifest_writer: ManifestWriter,
        _naming_scheme: ManifestNamingScheme,
        _transaction: Option<lance_table::format::Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        Err(CommitError::OtherError(lance_core::Error::io(
            "simulated commit failure",
        )))
    }
}

fn count_txn_files(uri: &str) -> usize {
    let tx_dir = std::path::Path::new(uri).join("_transactions");
    std::fs::read_dir(&tx_dir)
        .map(|rd| rd.filter_map(|e| e.ok()).count())
        .unwrap_or(0)
}

#[tokio::test]
async fn test_transaction_file_cleanup_on_commit_failure() {
    let tmp = TempStrDir::default();
    let uri = tmp.as_str();

    // Create initial dataset with a normal commit handler.
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "x",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
    Dataset::write(reader, uri, None).await.unwrap();

    let txn_files_before = count_txn_files(uri);

    // Attempt to append with a commit handler that always fails.
    let params = WriteParams {
        mode: WriteMode::Append,
        commit_handler: Some(Arc::new(FailingCommitHandler)),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let result = Dataset::write(reader, uri, Some(params)).await;
    assert!(result.is_err(), "expected commit to fail");

    // The failed commit must not leave any extra transaction files behind.
    let txn_files_after = count_txn_files(uri);
    assert_eq!(
        txn_files_after,
        txn_files_before,
        "failed commit left {extra} orphaned transaction file(s)",
        extra = txn_files_after.saturating_sub(txn_files_before),
    );
}

#[tokio::test]
async fn test_cos_commit_failure_preserves_error_and_cleans_up_transaction() {
    let tmp = TempStrDir::default();
    let uri = tmp.as_str();
    let schema = simple_schema();
    let batch = simple_batch(&schema, vec![1, 2, 3]);
    let commit_handler = commit_handler_from_url("cos://bucket/dataset", &None)
        .await
        .unwrap();
    let params = WriteParams {
        commit_handler: Some(commit_handler),
        ..Default::default()
    };

    let error = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(params),
    )
    .await
    .expect_err("the default Tencent COS handler should reject writes");

    assert!(
        matches!(&error, Error::NotSupported { .. }),
        "expected NotSupported, got: {error:?}"
    );
    assert!(
        error.to_string().contains("distributed commit_lock"),
        "unexpected error: {error}"
    );
    assert_eq!(
        count_txn_files(uri),
        0,
        "a definitively failed COS commit must clean up its transaction file"
    );
}

fn simple_batch(schema: &Arc<ArrowSchema>, values: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(values))]).unwrap()
}

fn simple_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "x",
        DataType::Int32,
        false,
    )]))
}

const APPEND_KEY_PROPERTY: &str = "job_id";

async fn create_append_test_dataset(uri: &str) -> Arc<Dataset> {
    Arc::new(
        InsertBuilder::new(uri)
            .execute(vec![simple_batch(&simple_schema(), vec![0])])
            .await
            .unwrap(),
    )
}

async fn stage_append(dataset: Arc<Dataset>, value: i32, key: &str) -> Transaction {
    InsertBuilder::new(dataset)
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            transaction_properties: Some(Arc::new(HashMap::from([(
                APPEND_KEY_PROPERTY.to_string(),
                key.to_string(),
            )]))),
            ..Default::default()
        })
        .execute_uncommitted(vec![simple_batch(&simple_schema(), vec![value])])
        .await
        .unwrap()
}

#[derive(Debug)]
struct ConcurrentAppendCommitHandler(tokio::sync::Barrier);

#[async_trait::async_trait]
impl CommitHandler for ConcurrentAppendCommitHandler {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<lance_table::format::Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        // Both writers must reach publication for version 2 before either
        // publishes. A distinct-key writer may then retry at version 3.
        if manifest.version == 2 {
            self.0.wait().await;
        }
        RenameCommitHandler
            .commit(
                manifest,
                indices,
                base_path,
                object_store,
                manifest_writer,
                naming_scheme,
                transaction,
            )
            .await
    }
}

#[rstest::rstest]
#[case::same_key("job-a", false)]
#[case::same_transaction("job-a", true)]
#[case::distinct_keys("job-b", false)]
#[tokio::test]
async fn test_append_idempotency_concurrent(
    #[case] second_key: &str,
    #[case] same_transaction: bool,
) {
    let tmp = TempStrDir::default();
    let dataset = create_append_test_dataset(tmp.as_str()).await;
    let first_key = "job-a";
    let mut first = stage_append(dataset.clone(), 1, first_key).await;
    let second = if same_transaction {
        // Force both submissions to use an external transaction file, so
        // reopening the winner detects a loser unlinking the shared file.
        Arc::make_mut(first.transaction_properties.as_mut().unwrap())
            .insert("padding".into(), "x".repeat(MAX_INLINE_TRANSACTION_BYTES));
        first.clone()
    } else {
        stage_append(dataset.clone(), 2, second_key).await
    };
    let handler = Arc::new(ConcurrentAppendCommitHandler(tokio::sync::Barrier::new(2)));
    let same_key = first_key == second_key;
    let retries = if same_key { 0 } else { 4 };
    let commit = |transaction| {
        CommitBuilder::new(dataset.clone())
            .with_append_idempotency_property(APPEND_KEY_PROPERTY)
            .with_commit_handler(handler.clone())
            .with_max_retries(retries)
            .with_timeout(Some(Duration::from_secs(5)))
            .execute(transaction)
    };
    let results = tokio::join!(commit(first), commit(second));
    let expected_version = if same_key {
        let (winner, duplicate) = match results {
            (Ok(winner), Err(duplicate)) | (Err(duplicate), Ok(winner)) => (winner, duplicate),
            results => panic!("expected one append and one duplicate: {results:?}"),
        };
        assert_eq!(winner.version().version, 2);
        assert_eq!(duplicate.append_already_committed_version(), Some(2));
        assert!(
            duplicate
                .to_string()
                .contains("Append already committed at version 2")
        );
        2
    } else {
        let mut versions = [
            results.0.unwrap().version().version,
            results.1.unwrap().version().version,
        ];
        versions.sort_unstable();
        assert_eq!(versions, [2, 3]);
        3
    };

    let latest = Dataset::open(tmp.as_str()).await.unwrap();
    assert_eq!(latest.version().version, expected_version);
    assert_eq!(
        latest.count_rows(None).await.unwrap(),
        expected_version as usize
    );
    assert_eq!(count_txn_files(tmp.as_str()), expected_version as usize);
    let committed = latest.read_transaction().await.unwrap().unwrap();
    let properties = committed.transaction_properties.unwrap();
    assert!(properties.contains_key(APPEND_KEY_PROPERTY));
}

#[rstest::rstest]
#[case::verified(crate::utils::test::AmbiguousFailure::LandAndConflict, false)]
#[case::unverifiable(
    crate::utils::test::AmbiguousFailure::LandAndErrorWithoutVerification,
    true
)]
#[tokio::test]
async fn test_append_idempotency_replay_after_lost_acknowledgment(
    #[case] failure: crate::utils::test::AmbiguousFailure,
    #[case] outcome_unknown: bool,
) {
    let tmp = TempStrDir::default();
    let dataset = create_append_test_dataset(tmp.as_str()).await;
    let transaction = stage_append(dataset.clone(), 1, "job-a").await;
    let handler = Arc::new(crate::utils::test::AmbiguousCommitHandler::default());
    handler.fail_next(failure);
    let result = CommitBuilder::new(dataset.clone())
        .with_append_idempotency_property(APPEND_KEY_PROPERTY)
        .with_commit_handler(handler)
        .execute(transaction)
        .await;
    if outcome_unknown {
        assert!(result.unwrap_err().is_commit_status_unknown());
    } else {
        assert_eq!(result.unwrap().version().version, 2);
    }

    // An ordinary append can reuse the same metadata, advancing the branch
    // while the logical append is waiting to be retried.
    let ordinary = stage_append(dataset.clone(), 2, "job-a").await;
    let advanced = CommitBuilder::new(dataset).execute(ordinary).await.unwrap();
    assert_eq!(advanced.version().version, 3);

    // Use a fresh handle at the original base to stage different files.
    let latest = Arc::new(Dataset::open(tmp.as_str()).await.unwrap());
    let base = Arc::new(latest.checkout_version(1).await.unwrap());
    let replay = stage_append(base, 3, "job-a").await;
    let duplicate = CommitBuilder::new(latest)
        .with_append_idempotency_property(APPEND_KEY_PROPERTY)
        .execute(replay)
        .await
        .unwrap_err();
    assert_eq!(duplicate.append_already_committed_version(), Some(2));
    let latest = Dataset::open(tmp.as_str()).await.unwrap();
    assert_eq!(latest.version().version, 3);
    assert_eq!(
        latest.scan().try_into_batch().await.unwrap(),
        simple_batch(&simple_schema(), vec![0, 1, 2]),
    );
}

#[rstest::rstest]
#[case::old_key("old-job", false)]
#[case::new_key("new-job", true)]
#[tokio::test]
async fn test_append_idempotency_after_recreate_at_same_uri(
    #[case] key: &str,
    #[case] is_duplicate: bool,
) {
    let tmp = TempStrDir::default();
    let uri = tmp.as_str();
    let original = create_append_test_dataset(uri).await;
    let old_append = stage_append(original.clone(), -1, "old-job").await;
    let old_uuid = old_append.uuid.clone();
    let original = CommitBuilder::new(original)
        .with_append_idempotency_property(APPEND_KEY_PROPERTY)
        .execute(old_append)
        .await
        .unwrap();
    let session = original.session.clone();
    std::fs::remove_dir_all(uri).unwrap();

    let recreated = create_append_test_dataset(uri).await;
    let mut new_append = stage_append(recreated.clone(), 1, "new-job").await;
    // A spilled transaction is not prefetched when version 2 is loaded.
    // This leaves the original table's cached version 2 in the session.
    Arc::make_mut(new_append.transaction_properties.as_mut().unwrap())
        .insert("padding".into(), "x".repeat(MAX_INLINE_TRANSACTION_BYTES));
    let recreated = CommitBuilder::new(recreated)
        .with_append_idempotency_property(APPEND_KEY_PROPERTY)
        .execute(new_append)
        .await
        .unwrap();
    assert_eq!(recreated.version().version, 2);
    assert!(recreated.manifest.transaction_section.is_none());

    let base = Arc::new(
        DatasetBuilder::from_uri(uri)
            .with_session(session)
            .with_version(1)
            .load()
            .await
            .unwrap(),
    );
    let (_, cached) = load_and_sort_new_transactions(&base, TransactionReadSource::SessionCache)
        .await
        .unwrap();
    assert_eq!(cached.len(), 1);
    assert_eq!(cached[0].0, 2);
    assert_eq!(cached[0].1.uuid, old_uuid);

    let append = stage_append(base.clone(), 2, key).await;
    let result = CommitBuilder::new(base)
        .with_append_idempotency_property(APPEND_KEY_PROPERTY)
        .execute(append)
        .await;
    let latest = Dataset::open(uri).await.unwrap();
    let (expected_version, expected_values) = if is_duplicate {
        let error = result.unwrap_err();
        assert_eq!(error.append_already_committed_version(), Some(2));
        (2, vec![0, 1])
    } else {
        assert_eq!(result.unwrap().version().version, 3);
        (3, vec![0, 1, 2])
    };
    assert_eq!(latest.version().version, expected_version);
    assert_eq!(
        latest.scan().try_into_batch().await.unwrap(),
        simple_batch(&simple_schema(), expected_values),
    );
}

#[rstest::rstest]
#[case::missing(None)]
#[case::empty(Some(""))]
#[tokio::test]
async fn test_append_idempotency_requires_a_key(#[case] key: Option<&str>) {
    let mut transaction = Transaction::new(1, Operation::Append { fragments: vec![] }, None);
    transaction.transaction_properties = key.map(|key| {
        Arc::new(HashMap::from([(
            APPEND_KEY_PROPERTY.to_string(),
            key.to_string(),
        )]))
    });
    let error = CommitBuilder::new("memory://")
        .with_append_idempotency_property(APPEND_KEY_PROPERTY)
        .execute(transaction)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("requires a nonempty transaction property")
    );
}

#[tokio::test]
async fn test_append_idempotency_refuses_missing_history() {
    let tmp = TempStrDir::default();
    let dataset = create_append_test_dataset(tmp.as_str()).await;
    let transaction = stage_append(dataset.clone(), 1, "job-a").await;
    let builder =
        CommitBuilder::new(dataset.clone()).with_append_idempotency_property(APPEND_KEY_PROPERTY);
    let winner = builder.clone().execute(transaction.clone()).await.unwrap();
    let other = stage_append(dataset.clone(), 2, "job-b").await;
    assert_eq!(builder.execute(other).await.unwrap().version().version, 3);
    dataset
        .table_state
        .object_store
        .delete(&winner.table_state.manifest_location.path)
        .await
        .unwrap();

    let latest = Arc::new(Dataset::open(tmp.as_str()).await.unwrap());
    let error = CommitBuilder::new(latest)
        .with_append_idempotency_property(APPEND_KEY_PROPERTY)
        .execute(transaction)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::CommitConflict { version: 3, .. }));
    assert!(error.to_string().contains("missing at versions [2]"));
    let latest = Dataset::open(tmp.as_str()).await.unwrap();
    assert_eq!(latest.version().version, 3);
    assert_eq!(latest.count_rows(None).await.unwrap(), 3);
}

/// A commit whose manifest lands but is reported as a conflict (the
/// incident shape: successful conditional PUT, response lost, internal
/// retry sees "already exists") must be recognized as our own commit and
/// returned as success — with the rows appearing exactly once and the
/// transaction file left in place.
#[tokio::test]
async fn test_commit_succeeds_when_conflict_is_own_commit() {
    use crate::utils::test::{AmbiguousCommitHandler, AmbiguousFailure};

    let tmp = TempStrDir::default();
    let uri = tmp.as_str();
    let schema = simple_schema();
    let handler = Arc::new(AmbiguousCommitHandler::default());

    let params = WriteParams {
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(
        vec![Ok(simple_batch(&schema, vec![1, 2, 3]))],
        schema.clone(),
    );
    Dataset::write(reader, uri, Some(params)).await.unwrap();
    assert_eq!(
        handler.resolve_calls(),
        0,
        "an uncontended commit must not perform verification reads"
    );
    let txn_files_before = count_txn_files(uri);

    handler.fail_next(AmbiguousFailure::LandAndConflict);
    let params = WriteParams {
        mode: WriteMode::Append,
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(
        vec![Ok(simple_batch(&schema, vec![4, 5, 6]))],
        schema.clone(),
    );
    let ds = Dataset::write(reader, uri, Some(params))
        .await
        .expect("a conflict with our own landed commit must be reported as success");

    assert_eq!(ds.version().version, 2);
    assert_eq!(
        ds.count_rows(None).await.unwrap(),
        6,
        "rows must appear exactly once (no duplicate re-commit)"
    );
    assert_eq!(
        count_txn_files(uri),
        txn_files_before + 1,
        "the landed commit's transaction file is referenced by the manifest and must survive"
    );

    // A fresh reader sees the committed version.
    let ds2 = Dataset::open(uri).await.unwrap();
    assert_eq!(ds2.version().version, 2);
    assert_eq!(ds2.count_rows(None).await.unwrap(), 6);
}

/// Same as above, but the landed commit is reported as a plain I/O error
/// (e.g. the store's retries all returned 5xx while the first attempt had
/// landed). Verification must still recognize the commit as ours.
#[tokio::test]
async fn test_commit_succeeds_when_landed_with_other_error() {
    use crate::utils::test::{AmbiguousCommitHandler, AmbiguousFailure};

    let tmp = TempStrDir::default();
    let uri = tmp.as_str();
    let schema = simple_schema();
    let handler = Arc::new(AmbiguousCommitHandler::default());

    let params = WriteParams {
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(
        vec![Ok(simple_batch(&schema, vec![1, 2, 3]))],
        schema.clone(),
    );
    Dataset::write(reader, uri, Some(params)).await.unwrap();
    let txn_files_before = count_txn_files(uri);

    handler.fail_next(AmbiguousFailure::LandAndError);
    let params = WriteParams {
        mode: WriteMode::Append,
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(
        vec![Ok(simple_batch(&schema, vec![4, 5, 6]))],
        schema.clone(),
    );
    let ds = Dataset::write(reader, uri, Some(params))
        .await
        .expect("an errored commit that actually landed must be reported as success");

    assert_eq!(ds.version().version, 2);
    assert_eq!(ds.count_rows(None).await.unwrap(), 6);
    assert_eq!(count_txn_files(uri), txn_files_before + 1);
}

#[tokio::test]
async fn test_commit_retries_temporarily_invisible_manifest() {
    use crate::utils::test::{AmbiguousCommitHandler, AmbiguousFailure};

    let tmp = TempStrDir::default();
    let uri = tmp.as_str();
    let schema = simple_schema();
    let handler = Arc::new(AmbiguousCommitHandler::default());
    let params = WriteParams {
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(
        vec![Ok(simple_batch(&schema, vec![1, 2, 3]))],
        schema.clone(),
    );
    Dataset::write(reader, uri, Some(params)).await.unwrap();

    handler.fail_next(AmbiguousFailure::LandAndError);
    handler.fail_next_resolves_with_not_found(2);
    let calls_before = handler.resolve_calls();
    let params = WriteParams {
        mode: WriteMode::Append,
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(simple_batch(&schema, vec![4, 5, 6]))], schema);
    let dataset = Dataset::write(reader, uri, Some(params))
        .await
        .expect("verification must retry a non-definitive NotFound result");

    assert_eq!(handler.resolve_calls() - calls_before, 3);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 6);
}

#[tokio::test]
async fn test_commit_verifies_inline_transaction_without_transaction_file() {
    use crate::utils::test::{AmbiguousCommitHandler, AmbiguousFailure};

    let tmp = TempStrDir::default();
    let uri = tmp.as_str();
    let schema = simple_schema();
    let handler = Arc::new(AmbiguousCommitHandler::default());
    let params = WriteParams {
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(simple_batch(&schema, vec![1, 2, 3]))], schema);
    let dataset = Dataset::write(reader, uri, Some(params)).await.unwrap();
    let transaction = Transaction::new(
        dataset.version().version,
        Operation::Append { fragments: vec![] },
        None,
    );
    let build_config = ManifestBuildConfig::default().with_disable_transaction_file(true);
    let commit_config = CommitConfig::default();

    handler.fail_next(AmbiguousFailure::LandAndConflict);
    let (manifest, _) = commit_transaction(
        &dataset,
        dataset.table_state.object_store.as_ref(),
        handler.as_ref(),
        &transaction,
        CommitTransactionOptions::new(
            &build_config,
            &commit_config,
            dataset.table_state.manifest_location.naming_scheme,
        )
        .with_retry_timeout(DEFAULT_COMMIT_RETRY_TIMEOUT),
    )
    .await
    .expect("the inline transaction must identify the landed commit");

    assert_eq!(manifest.version, 2);
    assert!(manifest.transaction_file.is_none());
    assert!(manifest.transaction_section.is_some());
}

/// A commit that errors without landing keeps today's behavior:
/// verification finds no manifest at the target version, the original
/// error propagates (not status-unknown), and the orphaned transaction
/// file is cleaned up.
#[tokio::test]
async fn test_commit_definite_failure_cleans_up_and_keeps_error() {
    use crate::utils::test::{AmbiguousCommitHandler, AmbiguousFailure};

    let tmp = TempStrDir::default();
    let uri = tmp.as_str();
    let schema = simple_schema();
    let handler = Arc::new(AmbiguousCommitHandler::default());

    let params = WriteParams {
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(
        vec![Ok(simple_batch(&schema, vec![1, 2, 3]))],
        schema.clone(),
    );
    Dataset::write(reader, uri, Some(params)).await.unwrap();
    let txn_files_before = count_txn_files(uri);

    handler.fail_next(AmbiguousFailure::FailOutright);
    let params = WriteParams {
        mode: WriteMode::Append,
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(
        vec![Ok(simple_batch(&schema, vec![4, 5, 6]))],
        schema.clone(),
    );
    let result = Dataset::write(reader, uri, Some(params)).await;
    let err = result.expect_err("commit that did not land must fail");
    assert!(
        !err.is_commit_status_unknown(),
        "a verified-absent commit is a definite failure, got: {:?}",
        err
    );
    assert_eq!(
        count_txn_files(uri),
        txn_files_before,
        "orphaned transaction file of a definitely-failed commit must be cleaned up"
    );
}

/// When the commit errors AND verification itself is unavailable, the
/// commit status is unknown: surface `CommitStatusUnknown` and delete
/// nothing.
#[tokio::test]
async fn test_commit_status_unknown_when_verification_unavailable() {
    use crate::utils::test::{AmbiguousCommitHandler, AmbiguousFailure};

    let tmp = TempStrDir::default();
    let uri = tmp.as_str();
    let schema = simple_schema();
    let handler = Arc::new(AmbiguousCommitHandler::default());

    let params = WriteParams {
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(
        vec![Ok(simple_batch(&schema, vec![1, 2, 3]))],
        schema.clone(),
    );
    Dataset::write(reader, uri, Some(params)).await.unwrap();
    let txn_files_before = count_txn_files(uri);

    // The commit lands but errors, and verification reads fail too.
    handler.fail_next(AmbiguousFailure::LandAndError);
    handler
        .fail_resolve
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let params = WriteParams {
        mode: WriteMode::Append,
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(
        vec![Ok(simple_batch(&schema, vec![4, 5, 6]))],
        schema.clone(),
    );
    let result = Dataset::write(reader, uri, Some(params)).await;
    let err = result.expect_err("unknown status must not be reported as success");
    assert!(
        err.is_commit_status_unknown(),
        "expected CommitStatusUnknown, got: {:?}",
        err
    );
    assert_eq!(
        count_txn_files(uri),
        txn_files_before + 1,
        "nothing may be deleted while the commit status is unknown"
    );

    // The commit did land: a fresh reader must see a consistent v2.
    handler
        .fail_resolve
        .store(false, std::sync::atomic::Ordering::SeqCst);
    let ds = Dataset::open(uri).await.unwrap();
    assert_eq!(ds.version().version, 2);
    assert_eq!(ds.count_rows(None).await.unwrap(), 6);
}

/// Dataset creation whose manifest lands but is reported as a conflict
/// must succeed instead of returning "dataset already exists".
#[tokio::test]
async fn test_create_dataset_succeeds_when_conflict_is_own_commit() {
    use crate::utils::test::{AmbiguousCommitHandler, AmbiguousFailure};

    let tmp = TempStrDir::default();
    let uri = tmp.as_str();
    let schema = simple_schema();
    let handler = Arc::new(AmbiguousCommitHandler::default());
    handler.fail_next(AmbiguousFailure::LandAndConflict);

    let params = WriteParams {
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(
        vec![Ok(simple_batch(&schema, vec![1, 2, 3]))],
        schema.clone(),
    );
    let ds = Dataset::write(reader, uri, Some(params))
        .await
        .expect("creation whose commit landed must succeed");
    assert_eq!(ds.version().version, 1);
    assert_eq!(ds.count_rows(None).await.unwrap(), 3);
}

/// Reproduces the debug-only panic `migrate_indices`'s fragment-bitmap
/// recalculation guard used to contain: a legal covered index
/// (`fields=[a,b]`, `covering_fields=[b]`) has `fields.len() == 2`, which
/// the old `debug_assert_eq!(index.fields.len(), 1)` rejected outright even
/// though the following line only ever reads `fields[0]`.
/// `must_recalculate_fragment_bitmap` takes this branch whenever
/// `fragment_bitmap` is `None`, so committing with it unset drives the
/// assert during the index's own commit.
#[tokio::test]
async fn test_covered_index_commit_recalculates_fragment_bitmap_without_panicking() {
    use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};

    let data = gen_batch()
        .col("a", array::step::<Int32Type>())
        .col("b", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(20), BatchCount::from(1));
    let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();
    dataset
        .create_index(
            &["a"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            false,
        )
        .await
        .unwrap();

    let b_id = dataset.schema().field("b").unwrap().id;
    let current = dataset.load_indices().await.unwrap();
    let mut covered = current[0].clone();
    covered.fields.push(b_id);
    covered.covering_fields = vec![b_id];
    // Force the fragment-bitmap recalculation branch this guard sits in.
    covered.fragment_bitmap = None;

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![covered],
            removed_indices: current.to_vec(),
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();

    let recomputed = dataset.load_indices().await.unwrap();
    assert_eq!(recomputed.len(), 1);
    assert!(
        recomputed[0].fragment_bitmap.is_some(),
        "migrate_indices should have recalculated the fragment bitmap for the covered index"
    );
}

/// A list-element index persists `fields=[input, dependency]` with nothing
/// carried, so it has no single keyed field. Recovering its coverage must
/// still open it through the one query input.
#[tokio::test]
async fn test_list_element_index_commit_recalculates_fragment_bitmap() {
    use arrow_array::{Array, ArrayRef, ListArray, StructArray};
    use arrow_buffer::OffsetBuffer;
    use lance_index::scalar::ScalarIndexParams;

    let reader = |ids: std::ops::Range<i32>| {
        let elements = StructArray::from(vec![
            (
                Arc::new(ArrowField::new("key", DataType::Int32, true)),
                Arc::new(Int32Array::from_iter_values(ids.clone())) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("label", DataType::Int32, true)),
                Arc::new(Int32Array::from_iter_values(ids.clone().map(|id| id * 10))) as ArrayRef,
            ),
        ]);
        let objects = ListArray::new(
            Arc::new(ArrowField::new("item", elements.data_type().clone(), true)),
            OffsetBuffer::from_lengths(vec![1; ids.len()]),
            Arc::new(elements),
            None,
        );
        let batch = RecordBatch::try_from_iter(vec![
            (
                "id",
                Arc::new(Int32Array::from_iter_values(ids)) as ArrayRef,
            ),
            ("objects", Arc::new(objects) as ArrayRef),
        ])
        .unwrap();
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema())
    };
    let mut dataset = Dataset::write(
        reader(0..4),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let trained_fragments = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<RoaringBitmap>();
    assert_eq!(trained_fragments.len(), 2);
    dataset
        .create_index(
            &["objects"],
            IndexType::Scalar,
            Some("objects_idx".to_string()),
            &ScalarIndexParams::new("listelement".to_string())
                .with_params(&serde_json::json!({"keys": ["key"]})),
            false,
        )
        .await
        .unwrap();
    // An unindexed fragment distinguishes the trained coverage from the table.
    dataset.append(reader(4..6), None).await.unwrap();
    assert_eq!(dataset.get_fragments().len(), 3);

    let root = dataset.schema().field("objects").unwrap();
    let dependency_fields = vec![root.id, root.children[0].children[0].id];
    let current = dataset.load_indices().await.unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].fields, dependency_fields);
    assert_eq!(current[0].keyed_field(), None);
    let mut missing_coverage = current[0].clone();
    missing_coverage.fragment_bitmap = None;

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![missing_coverage],
            removed_indices: current.to_vec(),
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();

    let recomputed = dataset.load_indices().await.unwrap();
    assert_eq!(recomputed.len(), 1);
    assert_eq!(recomputed[0].fields, dependency_fields);
    assert_eq!(recomputed[0].fragment_bitmap, Some(trained_fragments));
}

fn index_segment(name: &str, fragment_bitmap: Option<RoaringBitmap>) -> IndexMetadata {
    IndexMetadata {
        uuid: uuid::Uuid::new_v4(),
        name: name.to_string(),
        fields: vec![0],
        covering_fields: vec![],
        dataset_version: 1,
        fragment_bitmap,
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    }
}

#[test]
fn test_detect_overlapping_fragments() {
    let indices = vec![
        index_segment("idx_a", Some(RoaringBitmap::from_iter(0..5))),
        index_segment("idx_a", Some(RoaringBitmap::from_iter([3, 4, 10]))),
        index_segment("idx_a", None),
        index_segment("idx_b", Some(RoaringBitmap::from_iter(0..5))),
    ];
    let err = detect_overlapping_fragments(&indices).unwrap_err();
    assert_eq!(err.bad_indices.len(), 1);
    let (name, overlapping) = &err.bad_indices[0];
    assert_eq!(name, "idx_a");
    assert_eq!(overlapping, &vec![3, 4]);

    let disjoint = vec![
        index_segment("idx_a", Some(RoaringBitmap::from_iter(0..5))),
        index_segment("idx_a", Some(RoaringBitmap::from_iter(5..10))),
    ];
    assert!(detect_overlapping_fragments(&disjoint).is_ok());
}

/// Commit-outcome verification must compare durable (protobuf) forms.
///
/// A rewrite's frag reuse payload is intentionally dropped by
/// serialization and restored as `None` on read, so the in-memory
/// transaction never equals its own read-back form. Comparing durable
/// forms classifies the landed commit as ours anyway.
#[test]
fn test_frag_reuse_rewrite_own_commit_comparison_uses_durable_form() {
    use lance_table::transaction::Operation;

    let entry = IndexMetadata {
        uuid: uuid::Uuid::new_v4(),
        name: lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME.to_string(),
        fields: vec![],
        covering_fields: vec![],
        dataset_version: 42,
        fragment_bitmap: None,
        index_details: None,
        index_version: 1,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let transaction = Transaction::new(
        42,
        Operation::Rewrite {
            groups: vec![],
            rewritten_indices: vec![],
            frag_reuse_index: Some(entry),
        },
        None,
    );
    // What the commit wrote, and what verification reads back.
    let durable = pb::Transaction::try_from(&transaction).unwrap();
    let read_back = pb::Transaction::decode(durable.encode_to_vec().as_slice()).unwrap();
    // The old comparison (read-back deserialized into memory, compared
    // with `Transaction::eq`) misclassifies our own landed commit: the
    // round trip loses the payload.
    assert_ne!(
        Transaction::try_from(read_back.clone()).unwrap(),
        transaction,
        "the round-tripped transaction must differ in memory (frag_reuse_index is not serialized); \
         if this starts holding, the durable-form comparison is merely redundant"
    );
    // The comparison `verify_commit_outcome` performs: read-back durable
    // form against the regenerated durable form of this attempt.
    assert_eq!(read_back, pb::Transaction::try_from(&transaction).unwrap());
}

/// The manifest message of the file `dataset` was committed to, without
/// its length prefix.
async fn committed_manifest_message(dataset: &Dataset) -> Vec<u8> {
    let location = dataset.manifest_location();
    let file = dataset
        .table_state
        .object_store
        .read_one_all(&location.path)
        .await
        .unwrap();
    let footer = file.len() - 16;
    let pos = i64::from_le_bytes(file[footer..footer + 8].try_into().unwrap()) as usize;
    let len = u32::from_le_bytes(file[pos..pos + 4].try_into().unwrap()) as usize;
    assert_eq!(pos + 4 + len, footer);
    file[pos + 4..footer].to_vec()
}

/// The committed message is byte-identical to encoding the committed
/// manifest from scratch.
async fn assert_committed_manifest_matches(dataset: &Dataset) {
    use prost::Message;

    assert_eq!(
        committed_manifest_message(dataset).await,
        lance_table::format::pb::Manifest::from(dataset.manifest.as_ref()).encode_to_vec()
    );
}

#[tokio::test]
async fn test_commit_manifests_match_fresh_encoding() {
    let test_dir = TempStrDir::default();
    let rows = |start: i32| {
        gen_batch()
            .col("i", array::step_custom::<Int32Type>(start, 1))
            .into_reader_rows(RowCount::from(10), BatchCount::from(2))
    };
    let params = WriteParams {
        max_rows_per_file: 10,
        ..Default::default()
    };

    let mut dataset = Dataset::write(rows(0), &test_dir, Some(params.clone()))
        .await
        .unwrap();
    assert_committed_manifest_matches(&dataset).await;

    // Append.
    dataset
        .append(rows(20), Some(params.clone()))
        .await
        .unwrap();
    assert_eq!(dataset.manifest.fragments.len(), 4);
    assert_committed_manifest_matches(&dataset).await;

    // Update: fragment 0 is rewritten entirely (removed), fragment 1 in
    // part (updated with a deletion file), and the new rows are added.
    let dataset = crate::dataset::UpdateBuilder::new(Arc::new(dataset))
        .update_where("i < 15")
        .unwrap()
        .set("i", "i + 1000")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap()
        .new_dataset;
    let ids: Vec<u64> = dataset.manifest.fragments.iter().map(|f| f.id).collect();
    assert_eq!(ids, vec![1, 2, 3, 4]);
    assert!(dataset.manifest.fragments[0].deletion_file.is_some());
    assert_committed_manifest_matches(&dataset).await;

    // Delete: fragment 2 entirely, fragment 3 in part.
    let mut dataset = dataset.as_ref().clone();
    dataset.delete("i >= 20 AND i < 35").await.unwrap();
    let ids: Vec<u64> = dataset.manifest.fragments.iter().map(|f| f.id).collect();
    assert_eq!(ids, vec![1, 3, 4]);
    assert_committed_manifest_matches(&dataset).await;

    // Rewrite: compaction replaces the fragments with deletions.
    crate::dataset::optimize::compact_files(
        &mut dataset,
        crate::dataset::optimize::CompactionOptions::default(),
        None,
    )
    .await
    .unwrap();
    assert!(
        dataset
            .manifest
            .fragments
            .iter()
            .all(|fragment| fragment.deletion_file.is_none())
    );
    assert_committed_manifest_matches(&dataset).await;

    // Cold start: commit after reopening the dataset.
    let mut dataset = Dataset::open(&test_dir).await.unwrap();
    dataset.delete("i = 1001").await.unwrap();
    assert_committed_manifest_matches(&dataset).await;

    assert_eq!(dataset.count_rows(None).await.unwrap(), 24);
}
