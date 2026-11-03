// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

/// The first-activation fence judges each read field by its own identity:
/// a child added under a struct the writer read is fine, while a child of
/// that struct renamed, retyped or moved is not.
#[test]
fn additive_merge_keeps_every_read_field_but_a_changed_child_does_not() {
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    let schema = |children: Vec<Field>| {
        lance_core::datatypes::Schema::try_from(&ArrowSchema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("info", DataType::Struct(children.into()), true),
        ]))
        .unwrap()
    };
    let zip = Field::new("zip", DataType::Int32, true);
    let read = schema(vec![zip.clone()]);
    // Added child under the read struct: every read field keeps its identity.
    let added = schema(vec![zip, Field::new("note", DataType::Utf8, true)]);
    assert!(keeps_every_field_identity(&read, &added));
    // Renamed, retyped or missing child: not the same field any more.
    assert!(!keeps_every_field_identity(
        &read,
        &schema(vec![Field::new("postcode", DataType::Int32, true)])
    ));
    assert!(!keeps_every_field_identity(
        &read,
        &schema(vec![Field::new("zip", DataType::Int64, true)])
    ));
    assert!(!keeps_every_field_identity(&read, &schema(vec![])));
    // Moved: the same name and type under a different parent.
    let moved = lance_core::datatypes::Schema::try_from(&ArrowSchema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("info", DataType::Struct(Vec::<Field>::new().into()), true),
        Field::new("zip", DataType::Int32, true),
    ]))
    .unwrap();
    let mut moved = moved;
    // Give the moved column the read child's id so only its parent differs.
    let read_zip = read.field("info.zip").unwrap().id;
    for field in moved.fields.iter_mut() {
        if field.name == "zip" {
            field.id = read_zip;
        }
    }
    assert!(!keeps_every_field_identity(&read, &moved));
}

#[tokio::test]
async fn tagged_fri_rebase_returns_error_instead_of_panicking() {
    let dataset = test_dataset(4, 2).await;
    let tagged = IndexMetadata {
        uuid: Uuid::new_v4(),
        name: FRAG_REUSE_INDEX_NAME.into(),
        fields: vec![],
        covering_fields: vec![],
        dataset_version: 1,
        fragment_bitmap: None,
        index_details: None,
        index_version: 1,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let operation = Operation::Rewrite {
        groups: vec![],
        rewritten_indices: vec![],
        frag_reuse_index: Some(tagged),
    };
    let transaction = Transaction::new_from_version(dataset.manifest.version, operation);
    // A rewrite's tagged entry is read for what it adds relative to the
    // read version before anything is rebased: an entry without readable
    // details is an error at that point, never a panic.
    let Err(error) = TransactionRebase::try_new(&dataset, transaction, None).await else {
        panic!("a tagged entry without details is refused");
    };
    assert!(matches!(error, Error::Index { .. }), "{error}");
    assert!(error.to_string().contains("details"), "{error}");
}

#[rstest::rstest]
#[case::config(false)]
#[case::table_metadata(true)]
#[tokio::test]
async fn test_schema_change_preserves_unrelated_update_config_compatibility(
    #[case] update_table_metadata: bool,
    #[values("merge", "drop", "rename")] schema_change: &str,
    #[values(true, false)] schema_change_commits_first: bool,
) {
    let dataset = Arc::new(test_dataset(4, 2).await);
    let read_version = dataset.manifest.version;

    let mut schema = dataset.schema().clone();
    schema
        .metadata
        .insert("schema.key".to_string(), "preserved".to_string());
    let field_id = schema.fields[0].id;
    schema.fields[0]
        .metadata
        .insert("field.key".to_string(), "preserved".to_string());
    let operation = match schema_change {
        "merge" => Operation::Merge {
            fragments: dataset.manifest.fragments.as_ref().clone(),
            schema,
            preserves_nullability: true,
        },
        "drop" | "rename" => {
            if schema_change == "drop" {
                schema.fields.truncate(1);
            } else {
                schema.fields[1].name = "renamed_b".to_string();
            }
            Operation::Project {
                schema,
                preserves_nullability: true,
            }
        }
        _ => unreachable!(),
    };
    let schema_transaction = Transaction::new_from_version(read_version, operation);

    let replacement = UpdateMap {
        update_entries: vec![("key", Some("value")).into()],
        replace: true,
    };
    let update_config = Transaction::new_from_version(
        read_version,
        Operation::UpdateConfig {
            config_updates: (!update_table_metadata).then_some(replacement.clone()),
            table_metadata_updates: update_table_metadata.then_some(replacement),
            schema_metadata_updates: None,
            field_metadata_updates: HashMap::new(),
        },
    );

    let (first, stale) = if schema_change_commits_first {
        (schema_transaction, update_config)
    } else {
        (update_config, schema_transaction)
    };
    CommitBuilder::new(dataset.clone())
        .execute(first)
        .await
        .unwrap();
    let latest_dataset = CommitBuilder::new(dataset).execute(stale).await.unwrap();

    assert_eq!(
        latest_dataset
            .schema()
            .metadata
            .get("schema.key")
            .map(String::as_str),
        Some("preserved")
    );
    assert_eq!(
        latest_dataset
            .schema()
            .field_by_id(field_id)
            .unwrap()
            .metadata
            .get("field.key")
            .map(String::as_str),
        Some("preserved")
    );
    let updated_map = if update_table_metadata {
        &latest_dataset.manifest.table_metadata
    } else {
        &latest_dataset.manifest.config
    };
    assert_eq!(updated_map.get("key").map(String::as_str), Some("value"));
}

#[rstest::rstest]
#[case::schema_metadata(true)]
#[case::field_metadata(false)]
#[tokio::test]
async fn test_concurrent_schema_change_and_metadata_update_conflict(
    #[case] replace_schema_metadata: bool,
    #[values("merge", "drop", "rename")] schema_change: &str,
    #[values(true, false)] replace: bool,
    #[values(true, false)] schema_change_commits_first: bool,
) {
    let dataset = Arc::new(test_dataset(4, 2).await);
    let read_version = dataset.manifest.version;

    let mut schema = dataset.schema().clone();
    schema
        .metadata
        .insert("schema.key".to_string(), "coordinated".to_string());
    let field_id = schema.fields[0].id;
    schema.fields[0]
        .metadata
        .insert("field.key".to_string(), "coordinated".to_string());
    let operation = match schema_change {
        "merge" => Operation::Merge {
            fragments: dataset.manifest.fragments.as_ref().clone(),
            schema,
            preserves_nullability: true,
        },
        "drop" | "rename" => {
            if schema_change == "drop" {
                schema.fields.truncate(1);
            } else {
                schema.fields[1].name = "renamed_b".to_string();
            }
            Operation::Project {
                schema,
                preserves_nullability: true,
            }
        }
        _ => unreachable!(),
    };
    let schema_transaction = Transaction::new_from_version(read_version, operation);

    let metadata_update = UpdateMap {
        update_entries: vec![("replacement", Some("coordinated")).into()],
        replace,
    };
    let update_config = Transaction::new_from_version(
        read_version,
        Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: None,
            schema_metadata_updates: replace_schema_metadata.then_some(metadata_update.clone()),
            field_metadata_updates: if replace_schema_metadata {
                HashMap::new()
            } else {
                HashMap::from_iter([(field_id, metadata_update)])
            },
        },
    );

    let (first, stale) = if schema_change_commits_first {
        (schema_transaction, update_config)
    } else {
        (update_config, schema_transaction)
    };
    CommitBuilder::new(dataset.clone())
        .execute(first)
        .await
        .unwrap();
    let error = CommitBuilder::new(dataset)
        .execute(stale)
        .await
        .unwrap_err();

    assert!(matches!(error, Error::RetryableCommitConflict { .. }));
    assert!(error.to_string().contains("Please retry"), "{error}");
}

#[tokio::test]
async fn test_non_overlapping_rebase_delete_update() {
    let dataset = test_dataset(5, 5).await;
    let operation = Operation::Update {
        updated_fragments: vec![Fragment::new(0)],
        removed_fragment_ids: vec![],
        new_fragments: vec![],
        fields_modified: vec![],
        mem_wal_dir_id: None,
        compacted_sstables: Vec::new(),
        fields_for_preserving_frag_bitmap: vec![],
        update_mode: None,
        inserted_rows_filter: None,
        updated_fragment_offsets: None,
    };
    let transaction = Transaction::new_from_version(1, operation);
    let other_operations = [
        Operation::Update {
            updated_fragments: vec![Fragment::new(1)],
            removed_fragment_ids: vec![2],
            new_fragments: vec![],
            fields_modified: vec![],
            mem_wal_dir_id: None,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: None,
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        },
        Operation::Delete {
            deleted_fragment_ids: vec![3],
            updated_fragments: vec![],
            predicate: "a > 0".to_string(),
        },
        Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![Fragment::new(4)],
            new_fragments: vec![],
            fields_modified: vec![],
            mem_wal_dir_id: None,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: None,
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        },
    ];
    let other_transactions = other_operations.map(|op| Transaction::new_from_version(2, op));
    let mut rebase = TransactionRebase::try_new(&dataset, transaction.clone(), None)
        .await
        .unwrap();

    dataset.object_store.as_ref().io_stats_incremental(); // reset
    for (other_version, other_transaction) in other_transactions.iter().enumerate() {
        rebase
            .check_txn(other_transaction, other_version as u64)
            .unwrap();
        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_eq!(io_stats, read_iops, 0);
        assert_io_eq!(io_stats, write_iops, 0);
    }

    let expected_transaction = Transaction {
        // This doesn't really exercise it, since the other transactions
        // haven't been applied yet, but just doing this for completeness.
        read_version: dataset.manifest.version,
        ..transaction
    };
    let rebased_transaction = rebase.finish(&dataset).await.unwrap();
    assert_eq!(rebased_transaction, expected_transaction);
    // We didn't need to do any IO, so the stats should be 0.
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 0);
    assert_io_eq!(io_stats, write_iops, 0);
}

#[tokio::test]
async fn test_rebase_errors_when_read_version_was_cleaned_up() {
    // Regression test: `initial_fragments_for_rebase` used to `unwrap()` the
    // result of `checkout_version(read_version)`. If a concurrent
    // `cleanup_old_versions` removed that version between the conflicting
    // commit and the rebase, this panicked (aborting the whole process when
    // built with `panic = "abort"`). The rebase should fail with an error
    // instead so the commit can be retried.
    let tmp_dir = lance_core::utils::tempfile::TempStrDir::default();
    let uri = tmp_dir.as_str().to_string();
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..5)),
            Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(0, 5))),
        ],
    )
    .unwrap();

    // Write version 1, then append version 2.
    let write_params = WriteParams {
        max_rows_per_file: 1,
        ..Default::default()
    };
    InsertBuilder::new(&uri)
        .with_params(&write_params)
        .execute(vec![batch.clone()])
        .await
        .unwrap();
    let append_params = WriteParams {
        mode: WriteMode::Append,
        max_rows_per_file: 1,
        ..Default::default()
    };
    let dataset = InsertBuilder::new(&uri)
        .with_params(&append_params)
        .execute(vec![batch])
        .await
        .unwrap();
    assert_eq!(dataset.manifest.version, 2);

    // A transaction that read version 1 and modified fragment 0.
    let operation = Operation::Update {
        updated_fragments: vec![Fragment::new(0)],
        removed_fragment_ids: vec![],
        new_fragments: vec![],
        fields_modified: vec![],
        mem_wal_dir_id: None,
        compacted_sstables: Vec::new(),
        fields_for_preserving_frag_bitmap: vec![],
        update_mode: None,
        inserted_rows_filter: None,
        updated_fragment_offsets: None,
    };
    let transaction = Transaction::new_from_version(1, operation);

    // Simulate a concurrent `cleanup_old_versions` removing version 1.
    let naming_scheme = dataset.manifest_location().naming_scheme;
    let v1_manifest = naming_scheme.manifest_path(&dataset.base, 1);
    dataset.object_store.delete(&v1_manifest).await.unwrap();

    // Rebasing now needs to check out version 1, which no longer exists.
    // This used to panic; it should return `DatasetNotFound` instead.
    let err = TransactionRebase::try_new(&dataset, transaction, None)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::DatasetNotFound { .. }));
}

#[tokio::test]
#[rstest::rstest]
async fn test_non_conflicting_rebase_delete_update() {
    // 5 rows, all in one fragment. Each transaction modifies a different row.
    let mut dataset = test_dataset(5, 1).await;
    let mut fragment = dataset.fragments().as_slice()[0].clone();

    // Other operations modify the 1st, 2nd, and 3rd rows sequentially.
    let sample_file = Fragment::new(0)
        .with_file(
            "path1",
            vec![0],
            vec![0],
            LanceFileVersion::Stable.resolve(),
            NonZero::new(10),
        )
        .with_physical_rows(3);
    let operations = [
        Operation::Update {
            updated_fragments: vec![apply_deletion(&[0], &mut fragment, &dataset).await],
            removed_fragment_ids: vec![],
            new_fragments: vec![sample_file.clone()],
            fields_modified: vec![],
            mem_wal_dir_id: None,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: None,
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        },
        Operation::Delete {
            updated_fragments: vec![apply_deletion(&[1], &mut fragment, &dataset).await],
            deleted_fragment_ids: vec![],
            predicate: "a > 0".to_string(),
        },
        Operation::Update {
            updated_fragments: vec![apply_deletion(&[2], &mut fragment, &dataset).await],
            removed_fragment_ids: vec![],
            new_fragments: vec![sample_file],
            fields_modified: vec![],
            mem_wal_dir_id: None,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: None,
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        },
    ];
    let transactions =
        operations.map(|op| Transaction::new_from_version(dataset.manifest.version, op));

    for (i, transaction) in transactions.iter().enumerate() {
        let previous_transactions = transactions.iter().take(i).cloned().collect::<Vec<_>>();

        let affected_rows = RowAddrTreeMap::from_iter([i as u64]);
        let mut rebase =
            TransactionRebase::try_new(&dataset, transaction.clone(), Some(&affected_rows))
                .await
                .unwrap();

        dataset.object_store.as_ref().io_stats_incremental(); // reset
        for (other_version, other_transaction) in previous_transactions.iter().enumerate() {
            rebase
                .check_txn(other_transaction, other_version as u64)
                .unwrap();
            let io_stats = dataset.object_store.as_ref().io_stats_incremental();
            assert_io_eq!(io_stats, read_iops, 0);
            assert_io_eq!(io_stats, write_iops, 0);
        }

        // First iteration, we don't need to rewrite the deletion file.
        let expected_rewrite = i > 0;

        let rebased_transaction = rebase.finish(&dataset).await.unwrap();
        assert_eq!(rebased_transaction.read_version, dataset.manifest.version);

        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        if expected_rewrite {
            // Read the current deletion file, and write the new one.
            assert_io_eq!(io_stats, read_iops, 0, "deletion file should be cached");
            assert_io_eq!(io_stats, write_iops, 1, "write one deletion file");

            // TODO: The old deletion file should be gone.
            // This can be done later, as it will be cleaned up by the
            // background cleanup process for now.
            // let original_fragment = match &original_transaction.operation {
            //     Operation::Update {
            //         updated_fragments, ..
            //     }
            //     | Operation::Delete {
            //         updated_fragments, ..
            //     } => updated_fragments[0].clone(),
            //     _ => {
            //         panic!("Expected an update or delete operation");
            //     }
            // };
            // let old_path = deletion_file_path(
            //     &dataset.base,
            //     original_fragment.id,
            //     original_fragment.deletion_file.as_ref().unwrap(),
            // );
            // assert!(!dataset.object_store.as_ref().exists(&old_path).await.unwrap());
            // The new deletion file should exist.
            let final_fragment = match &rebased_transaction.operation {
                Operation::Update {
                    updated_fragments, ..
                }
                | Operation::Delete {
                    updated_fragments, ..
                } => updated_fragments[0].clone(),
                _ => {
                    panic!("Expected an update or delete operation");
                }
            };
            let new_path = deletion_file_path(
                &dataset.base,
                final_fragment.id,
                final_fragment.deletion_file.as_ref().unwrap(),
            );
            assert!(
                dataset
                    .object_store
                    .as_ref()
                    .exists(&new_path)
                    .await
                    .unwrap()
            );

            assert_io_eq!(io_stats, num_stages, 1);
        } else {
            // No IO should have happened.
            assert_io_eq!(io_stats, read_iops, 0);
            assert_io_eq!(io_stats, write_iops, 0);
        }

        dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(rebased_transaction)
            .await
            .unwrap();
    }
}

/// Validate we get a conflict error when rebasing `operation` on top of `other`.
#[tokio::test]
#[rstest::rstest]
async fn test_conflicting_rebase(
    #[values("update_full", "update_partial", "delete_full", "delete_partial")] ours: &str,
    #[values("update_full", "update_partial", "delete_full", "delete_partial")] other: &str,
) {
    // 5 rows, all in one fragment. Each transaction modifies the same row.
    let dataset = test_dataset(5, 1).await;
    let mut fragment = dataset.fragments().as_slice()[0].clone();

    let sample_file = Fragment::new(0)
        .with_file(
            "path1",
            vec![0],
            vec![0],
            LanceFileVersion::Stable.resolve(),
            NonZero::new(10),
        )
        .with_physical_rows(3);

    let operations = [
        (
            "update_full",
            Operation::Update {
                updated_fragments: vec![],
                removed_fragment_ids: vec![0],
                new_fragments: vec![sample_file.clone()],
                fields_modified: vec![],
                mem_wal_dir_id: None,
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
        ),
        (
            "update_partial",
            Operation::Update {
                updated_fragments: vec![apply_deletion(&[0], &mut fragment, &dataset).await],
                removed_fragment_ids: vec![],
                new_fragments: vec![sample_file.clone()],
                fields_modified: vec![],
                mem_wal_dir_id: None,
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
        ),
        (
            "delete_full",
            Operation::Delete {
                updated_fragments: vec![],
                deleted_fragment_ids: vec![0],
                predicate: "a > 0".to_string(),
            },
        ),
        (
            "delete_partial",
            Operation::Delete {
                updated_fragments: vec![apply_deletion(&[0], &mut fragment, &dataset).await],
                deleted_fragment_ids: vec![],
                predicate: "a > 0".to_string(),
            },
        ),
    ];

    let operation = operations
        .iter()
        .find(|(name, _)| *name == ours)
        .unwrap()
        .1
        .clone();
    let other_op = operations
        .iter()
        .find(|(name, _)| *name == other)
        .unwrap()
        .1
        .clone();

    let other_txn = Transaction::new_from_version(dataset.manifest.version, other_op);
    let txn = Transaction::new_from_version(dataset.manifest.version, operation);

    // Can apply first transaction to create the conflict
    let latest_dataset = CommitBuilder::new(Arc::new(dataset.clone()))
        .execute(other_txn.clone())
        .await
        .unwrap();

    let affected_rows = RowAddrTreeMap::from_iter([0]);

    dataset.object_store.as_ref().io_stats_incremental(); // reset
    let mut rebase = TransactionRebase::try_new(&dataset, txn.clone(), Some(&affected_rows))
        .await
        .unwrap();

    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 0);
    assert_io_eq!(io_stats, write_iops, 0);

    let res = rebase.check_txn(&other_txn, 1);
    if other.ends_with("full") || ours.ends_with("full") {
        // If the other transaction fully deleted a fragment, we can error early.
        assert!(matches!(
            res,
            Err(crate::Error::RetryableCommitConflict { .. })
        ));
        return;
    } else {
        assert!(res.is_ok());
    }

    assert_eq!(
        rebase
            .initial_fragments
            .iter()
            .map(|(id, (_, needs_rewrite))| (*id, *needs_rewrite))
            .collect::<Vec<_>>(),
        vec![(0, true)],
    );

    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 0);
    assert_io_eq!(io_stats, write_iops, 0);

    let res = rebase.finish(&latest_dataset).await;
    assert!(matches!(
        res,
        Err(crate::Error::RetryableCommitConflict { .. })
    ));

    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 0, "deletion file should be cached");
    assert_io_eq!(io_stats, write_iops, 0, "failed before writing");
}
