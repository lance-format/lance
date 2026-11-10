// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[test]
fn test_mem_wal_install_conflicts_with_merge() {
    let mem_wal_index = IndexMetadata {
        uuid: uuid::Uuid::new_v4(),
        name: MEM_WAL_INDEX_NAME.to_string(),
        fields: vec![],
        covering_fields: vec![],
        dataset_version: 1,
        fragment_bitmap: None,
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let column_index = IndexMetadata {
        uuid: uuid::Uuid::new_v4(),
        name: "btree".to_string(),
        ..mem_wal_index.clone()
    };
    let merge = Transaction::new(
        0,
        Operation::Merge {
            fragments: vec![],
            schema: lance_core::datatypes::Schema::default(),
            preserves_nullability: true,
        },
        None,
    );

    // Install rebasing over a committed Merge conflicts; a column index
    // stays compatible.
    for (index, conflicts) in [(mem_wal_index.clone(), true), (column_index, false)] {
        let txn = Transaction::new(
            0,
            Operation::CreateIndex {
                new_indices: vec![index],
                removed_indices: vec![],
            },
            None,
        );
        let mut rebase = TransactionRebase {
            transaction: txn,
            initial_fragments: HashMap::new(),
            modified_fragment_ids: HashSet::new(),
            affected_rows: None,
            frag_reuse_base: None,
            conflicting_mem_wal_compacted_sstables: Vec::new(),
            current_lineage: None,
            untagged_reuse: None,
            current_live: None,
            current_schema: None,
            read_fragments: None,
            read_schema: None,
            reuse: Default::default(),
            staged_replay: false,
        };
        let result = rebase.check_txn(&merge, 1);
        assert_eq!(result.is_err(), conflicts, "{result:?}");
    }

    // And the reverse: a Merge rebasing over a committed install conflicts.
    let install = Transaction::new(
        0,
        Operation::CreateIndex {
            new_indices: vec![mem_wal_index],
            removed_indices: vec![],
        },
        None,
    );
    let mut rebase = TransactionRebase {
        transaction: merge,
        initial_fragments: HashMap::new(),
        modified_fragment_ids: HashSet::new(),
        affected_rows: None,
        frag_reuse_base: None,
        conflicting_mem_wal_compacted_sstables: Vec::new(),
        current_lineage: None,
        untagged_reuse: None,
        current_live: None,
        current_schema: None,
        read_fragments: None,
        read_schema: None,
        reuse: Default::default(),
        staged_replay: false,
    };
    let result = rebase.check_txn(&install, 1);
    assert!(
        matches!(result, Err(Error::RetryableCommitConflict { .. })),
        "{result:?}"
    );
}

#[test]
fn test_create_index_conflicts_only_on_same_name() {
    let index0 = IndexMetadata {
        uuid: uuid::Uuid::new_v4(),
        name: "test".to_string(),
        fields: vec![0],
        covering_fields: vec![],
        dataset_version: 1,
        fragment_bitmap: None,
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let index1 = IndexMetadata {
        uuid: uuid::Uuid::new_v4(),
        name: "other".to_string(),
        ..index0.clone()
    };

    let txn = Transaction::new(
        0,
        Operation::CreateIndex {
            new_indices: vec![index0.clone()],
            removed_indices: vec![],
        },
        None,
    );
    let mut rebase = TransactionRebase {
        transaction: txn,
        initial_fragments: HashMap::new(),
        modified_fragment_ids: HashSet::new(),
        affected_rows: None,
        frag_reuse_base: None,
        conflicting_mem_wal_compacted_sstables: Vec::new(),
        current_lineage: None,
        untagged_reuse: None,
        current_live: None,
        current_schema: None,
        read_fragments: None,
        read_schema: None,
        reuse: Default::default(),
        staged_replay: false,
    };

    let same_name = Transaction::new(
        0,
        Operation::CreateIndex {
            new_indices: vec![IndexMetadata {
                uuid: uuid::Uuid::new_v4(),
                ..index0
            }],
            removed_indices: vec![],
        },
        None,
    );
    let different_name = Transaction::new(
        0,
        Operation::CreateIndex {
            new_indices: vec![index1],
            removed_indices: vec![],
        },
        None,
    );

    let same_name_result = rebase.check_txn(&same_name, 1);
    assert!(
        matches!(same_name_result, Err(Error::RetryableCommitConflict { .. })),
        "Expected retryable conflict for same-name CreateIndex, got {:?}",
        same_name_result
    );

    let mut rebase = TransactionRebase {
        transaction: Transaction::new(
            0,
            Operation::CreateIndex {
                new_indices: vec![IndexMetadata {
                    uuid: uuid::Uuid::new_v4(),
                    name: "test".to_string(),
                    fields: vec![0],
                    covering_fields: vec![],
                    dataset_version: 1,
                    fragment_bitmap: None,
                    index_details: None,
                    index_version: 0,
                    created_at: None,
                    base_id: None,
                    files: None,
                    clustering_id: None,
                    metadata: Default::default(),
                }],
                removed_indices: vec![],
            },
            None,
        ),
        initial_fragments: HashMap::new(),
        modified_fragment_ids: HashSet::new(),
        affected_rows: None,
        frag_reuse_base: None,
        conflicting_mem_wal_compacted_sstables: Vec::new(),
        current_lineage: None,
        untagged_reuse: None,
        current_live: None,
        current_schema: None,
        read_fragments: None,
        read_schema: None,
        reuse: Default::default(),
        staged_replay: false,
    };
    let different_name_result = rebase.check_txn(&different_name, 1);
    assert!(
        different_name_result.is_ok(),
        "Expected compatibility for different-name CreateIndex, got {:?}",
        different_name_result
    );
}

#[rstest::rstest]
#[case::optimize_after_drop(false, true)]
#[case::drop_after_optimize(true, true)]
#[case::append_segment_after_drop(false, false)]
#[case::drop_after_append_segment(true, false)]
fn test_create_index_conflicts_with_concurrent_drop(
    #[case] drop_is_current: bool,
    #[case] maintenance_replaces_existing: bool,
) {
    let existing = IndexMetadata {
        uuid: Uuid::new_v4(),
        name: "test".to_string(),
        fields: vec![0],
        covering_fields: vec![],
        dataset_version: 1,
        fragment_bitmap: None,
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let drop_operation = Operation::CreateIndex {
        new_indices: vec![],
        removed_indices: vec![existing.clone()],
    };
    let maintenance_operation = Operation::CreateIndex {
        new_indices: vec![IndexMetadata {
            uuid: Uuid::new_v4(),
            ..existing.clone()
        }],
        removed_indices: maintenance_replaces_existing
            .then(|| existing.clone())
            .into_iter()
            .collect(),
    };
    let (current_operation, committed_operation) = if drop_is_current {
        (drop_operation, maintenance_operation)
    } else {
        (maintenance_operation, drop_operation)
    };

    let mut rebase = TransactionRebase {
        transaction: Transaction::new(0, current_operation, None),
        initial_fragments: HashMap::new(),
        modified_fragment_ids: HashSet::new(),
        affected_rows: None,
        frag_reuse_base: None,
        conflicting_mem_wal_compacted_sstables: Vec::new(),
        current_lineage: None,
        untagged_reuse: None,
        current_live: None,
        current_schema: None,
        read_fragments: None,
        read_schema: None,
        reuse: Default::default(),
        staged_replay: false,
    };
    let result = rebase.check_txn(&Transaction::new(0, committed_operation, None), 1);

    assert!(
        matches!(result, Err(Error::RetryableCommitConflict { .. })),
        "Expected a retryable conflict between index maintenance and drop, got {result:?}"
    );
    let message = result.unwrap_err().to_string();
    assert!(
        message.contains("preempted by concurrent transaction") && message.contains("version 1"),
        "Unexpected conflict message: {message}"
    );
}

#[test]
fn test_concurrent_index_drops_are_compatible() {
    let existing = IndexMetadata {
        uuid: Uuid::new_v4(),
        name: "test".to_string(),
        fields: vec![0],
        covering_fields: vec![],
        dataset_version: 1,
        fragment_bitmap: None,
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let drop_operation = Operation::CreateIndex {
        new_indices: vec![],
        removed_indices: vec![existing],
    };
    let mut rebase = TransactionRebase {
        transaction: Transaction::new(0, drop_operation.clone(), None),
        initial_fragments: HashMap::new(),
        modified_fragment_ids: HashSet::new(),
        affected_rows: None,
        frag_reuse_base: None,
        conflicting_mem_wal_compacted_sstables: Vec::new(),
        current_lineage: None,
        untagged_reuse: None,
        current_live: None,
        current_schema: None,
        read_fragments: None,
        read_schema: None,
        reuse: Default::default(),
        staged_replay: false,
    };

    let result = rebase.check_txn(&Transaction::new(0, drop_operation, None), 1);

    assert!(
        result.is_ok(),
        "Concurrent drops of the same index should be idempotent, got {result:?}"
    );
}

#[test]
fn test_index_replacement_compatible_with_removing_disjoint_segment() {
    let index = |uuid| IndexMetadata {
        uuid,
        name: "test".to_string(),
        fields: vec![0],
        covering_fields: vec![],
        dataset_version: 1,
        fragment_bitmap: None,
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let replaced = index(Uuid::new_v4());
    let concurrently_removed = index(Uuid::new_v4());
    let replacement_operation = Operation::CreateIndex {
        new_indices: vec![index(Uuid::new_v4())],
        removed_indices: vec![replaced],
    };
    let removal_operation = Operation::CreateIndex {
        new_indices: vec![],
        removed_indices: vec![concurrently_removed],
    };
    let mut rebase = TransactionRebase {
        transaction: Transaction::new(0, replacement_operation, None),
        initial_fragments: HashMap::new(),
        modified_fragment_ids: HashSet::new(),
        affected_rows: None,
        frag_reuse_base: None,
        conflicting_mem_wal_compacted_sstables: Vec::new(),
        current_lineage: None,
        untagged_reuse: None,
        current_live: None,
        current_schema: None,
        read_fragments: None,
        read_schema: None,
        reuse: Default::default(),
        staged_replay: false,
    };

    let result = rebase.check_txn(&Transaction::new(0, removal_operation, None), 1);

    assert!(
        result.is_ok(),
        "Replacing one segment should be compatible with removing another, got {result:?}"
    );
}

#[test]
fn test_create_ngram_index_conflicts_with_overlapping_deferred_rewrite() {
    let ngram_index = |fragment_id| IndexMetadata {
        uuid: Uuid::new_v4(),
        name: "text_ngram".to_string(),
        fields: vec![0],
        covering_fields: vec![],
        dataset_version: 1,
        fragment_bitmap: Some(RoaringBitmap::from_iter([fragment_id])),
        index_details: Some(Arc::new(prost_types::Any {
            type_url: "lance.index.NGramIndexDetails".to_string(),
            value: Vec::new(),
        })),
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let frag_reuse_index = IndexMetadata {
        uuid: Uuid::new_v4(),
        name: FRAG_REUSE_INDEX_NAME.to_string(),
        fields: vec![],
        covering_fields: vec![],
        dataset_version: 2,
        fragment_bitmap: Some(RoaringBitmap::from_iter([2u32])),
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let rewrite = Transaction::new(
        1,
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: vec![Fragment::new(1)],
                new_fragments: vec![Fragment::new(2)],
            }],
            rewritten_indices: vec![],
            frag_reuse_index: Some(frag_reuse_index),
        },
        None,
    );

    for (covered_fragment, expect_conflict) in [(1u32, true), (3u32, false)] {
        let mut rebase = TransactionRebase {
            transaction: Transaction::new(
                1,
                Operation::CreateIndex {
                    new_indices: vec![ngram_index(covered_fragment)],
                    removed_indices: vec![],
                },
                None,
            ),
            initial_fragments: HashMap::new(),
            modified_fragment_ids: HashSet::new(),
            affected_rows: None,
            frag_reuse_base: None,
            conflicting_mem_wal_compacted_sstables: Vec::new(),
            current_lineage: None,
            untagged_reuse: None,
            current_live: None,
            current_schema: None,
            read_fragments: None,
            read_schema: None,
            reuse: Default::default(),
            staged_replay: false,
        };
        let result = rebase.check_txn(&rewrite, 2);
        if expect_conflict {
            assert!(
                matches!(result, Err(Error::RetryableCommitConflict { .. })),
                "overlapping staged NGram index should conflict, got {result:?}"
            );
        } else {
            assert!(
                result.is_ok(),
                "disjoint staged NGram index should remain compatible, got {result:?}"
            );
        }
    }
}

#[test]
fn test_create_index_conflicts_with_deferred_rewrite_under_stable_row_ids() {
    let row_id_domain_index = |fragment_id| IndexMetadata {
        uuid: Uuid::new_v4(),
        name: "vector_ivf".to_string(),
        fields: vec![0],
        covering_fields: vec![],
        dataset_version: 1,
        fragment_bitmap: Some(RoaringBitmap::from_iter([fragment_id])),
        index_details: Some(Arc::new(prost_types::Any {
            type_url: "lance.index.IvfPqIndexDetails".to_string(),
            value: Vec::new(),
        })),
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let frag_reuse_index = IndexMetadata {
        uuid: Uuid::new_v4(),
        name: FRAG_REUSE_INDEX_NAME.to_string(),
        fields: vec![],
        covering_fields: vec![],
        dataset_version: 2,
        fragment_bitmap: Some(RoaringBitmap::from_iter([2u32])),
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };

    // The rewrite's own `old_fragments` are what the SRID inference reads:
    // `row_id_meta` set means stable row ids were on at the read version.
    for (row_id_meta, expect_conflict) in [
        (Some(RowIdMeta::Inline(InlineRowIds::from(vec![7]))), true),
        (None, false),
    ] {
        let mut old_fragment = Fragment::new(1);
        old_fragment.row_id_meta = row_id_meta;

        let mut rebase = TransactionRebase {
            transaction: Transaction::new(
                1,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: vec![old_fragment],
                        new_fragments: vec![Fragment::new(2)],
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: Some(frag_reuse_index.clone()),
                },
                None,
            ),
            initial_fragments: HashMap::new(),
            modified_fragment_ids: HashSet::new(),
            affected_rows: None,
            frag_reuse_base: None,
            conflicting_mem_wal_compacted_sstables: Vec::new(),
            current_lineage: None,
            untagged_reuse: None,
            current_live: None,
            current_schema: None,
            read_fragments: None,
            read_schema: None,
            reuse: Default::default(),

            staged_replay: false,
        };
        // Disjoint from the rewritten fragment, so this only exercises the
        // stable-row-ids check and not the group-straddling check below it.
        let create_index = Transaction::new(
            1,
            Operation::CreateIndex {
                new_indices: vec![row_id_domain_index(5u32)],
                removed_indices: vec![],
            },
            None,
        );
        let result = rebase.check_txn(&create_index, 2);
        if expect_conflict {
            assert!(
                matches!(result, Err(Error::RetryableCommitConflict { .. })),
                "stable row ids should conflict with a row-id-domain index built \
                     during a deferred rewrite, got {result:?}"
            );
        } else {
            assert!(
                result.is_ok(),
                "fragments without row ids should not trip the stable-row-ids \
                     inference, got {result:?}"
            );
        }
    }
}

#[tokio::test]
async fn test_add_bases_non_conflicting() {
    let dataset = test_dataset(10, 2).await;

    // Create two transactions adding different bases
    let txn1 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 1,
                path: "s3://bucket1/path1".to_string(),
                name: Some("base1".to_string()),
                is_dataset_root: false,
            }],
        },
    );

    let txn2 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 2,
                path: "s3://bucket2/path2".to_string(),
                name: Some("base2".to_string()),
                is_dataset_root: false,
            }],
        },
    );

    // txn1 should not conflict with txn2
    let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
        .await
        .unwrap();
    assert!(rebase.check_txn(&txn2, 2).is_ok());
}

#[tokio::test]
async fn test_add_bases_name_conflict() {
    let dataset = test_dataset(10, 2).await;

    // Create two transactions adding bases with the same name
    let txn1 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 1,
                path: "s3://bucket1/path1".to_string(),
                name: Some("duplicate_name".to_string()),
                is_dataset_root: false,
            }],
        },
    );

    let txn2 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 2,
                path: "s3://bucket2/path2".to_string(),
                name: Some("duplicate_name".to_string()),
                is_dataset_root: false,
            }],
        },
    );

    // txn1 should conflict with txn2 due to duplicate name
    let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
        .await
        .unwrap();
    let result = rebase.check_txn(&txn2, 2);
    assert!(
        matches!(result, Err(Error::IncompatibleTransaction { .. })),
        "Expected IncompatibleTransaction error for duplicate name, got {:?}",
        result
    );
}

#[tokio::test]
async fn test_add_bases_path_conflict() {
    let dataset = test_dataset(10, 2).await;

    // Create two transactions adding bases with the same path
    let txn1 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 1,
                path: "s3://bucket/duplicate_path".to_string(),
                name: Some("base1".to_string()),
                is_dataset_root: false,
            }],
        },
    );

    let txn2 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 2,
                path: "s3://bucket/duplicate_path".to_string(),
                name: Some("base2".to_string()),
                is_dataset_root: false,
            }],
        },
    );

    // txn1 should conflict with txn2 due to duplicate path
    let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
        .await
        .unwrap();
    let result = rebase.check_txn(&txn2, 2);
    assert!(
        matches!(result, Err(Error::IncompatibleTransaction { .. })),
        "Expected IncompatibleTransaction error for duplicate path, got {:?}",
        result
    );
}

#[tokio::test]
async fn test_add_bases_id_conflict() {
    let dataset = test_dataset(10, 2).await;

    // Create two transactions adding bases with the same non-zero ID
    let txn1 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 42,
                path: "s3://bucket1/path1".to_string(),
                name: Some("base1".to_string()),
                is_dataset_root: false,
            }],
        },
    );

    let txn2 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 42,
                path: "s3://bucket2/path2".to_string(),
                name: Some("base2".to_string()),
                is_dataset_root: false,
            }],
        },
    );

    // txn1 should conflict with txn2 due to duplicate non-zero ID
    let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
        .await
        .unwrap();
    let result = rebase.check_txn(&txn2, 2);
    assert!(
        matches!(result, Err(Error::IncompatibleTransaction { .. })),
        "Expected IncompatibleTransaction error for duplicate ID, got {:?}",
        result
    );
}

#[tokio::test]
async fn test_add_bases_no_conflict_with_data_operations() {
    let dataset = test_dataset(10, 2).await;

    let add_bases_txn = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 1,
                path: "s3://bucket/path".to_string(),
                name: Some("base1".to_string()),
                is_dataset_root: false,
            }],
        },
    );

    // Test against various data operations
    let data_operations = vec![
        Operation::Append { fragments: vec![] },
        Operation::Delete {
            deleted_fragment_ids: vec![0],
            updated_fragments: vec![],
            predicate: "a > 5".to_string(),
        },
        Operation::Update {
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
        },
    ];

    for operation in data_operations {
        let data_txn = Transaction::new_from_version(1, operation.clone());
        let mut rebase = TransactionRebase::try_new(&dataset, add_bases_txn.clone(), None)
            .await
            .unwrap();
        assert!(
            rebase.check_txn(&data_txn, 2).is_ok(),
            "UpdateBases should not conflict with {:?}",
            operation
        );
    }
}

#[tokio::test]
async fn test_add_bases_multiple_bases() {
    let dataset = test_dataset(10, 2).await;

    // txn1 adds two bases
    let txn1 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![
                lance_table::format::BasePath {
                    id: 1,
                    path: "s3://bucket1/path1".to_string(),
                    name: Some("base1".to_string()),
                    is_dataset_root: false,
                },
                lance_table::format::BasePath {
                    id: 2,
                    path: "s3://bucket2/path2".to_string(),
                    name: Some("base2".to_string()),
                    is_dataset_root: false,
                },
            ],
        },
    );

    // txn2 adds a base that conflicts with one of txn1's bases
    let txn2 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 3,
                path: "s3://bucket1/path1".to_string(), // Same path as txn1's first base
                name: Some("base3".to_string()),
                is_dataset_root: false,
            }],
        },
    );

    // Should conflict due to path conflict
    let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
        .await
        .unwrap();
    let result = rebase.check_txn(&txn2, 2);
    assert!(
        matches!(result, Err(Error::IncompatibleTransaction { .. })),
        "Expected IncompatibleTransaction error, got {:?}",
        result
    );
}

#[tokio::test]
async fn test_add_bases_with_none_name() {
    let dataset = test_dataset(10, 2).await;

    // Bases with None names should not conflict on name
    let txn1 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 1,
                path: "s3://bucket1/path1".to_string(),
                name: None,
                is_dataset_root: false,
            }],
        },
    );

    let txn2 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 2,
                path: "s3://bucket2/path2".to_string(),
                name: None,
                is_dataset_root: false,
            }],
        },
    );

    // Should not conflict despite both having None names
    let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
        .await
        .unwrap();
    assert!(rebase.check_txn(&txn2, 2).is_ok());
}

#[tokio::test]
async fn test_add_bases_with_zero_id() {
    let dataset = test_dataset(10, 2).await;

    // Bases with zero IDs should not conflict on ID
    let txn1 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 0,
                path: "s3://bucket1/path1".to_string(),
                name: Some("base1".to_string()),
                is_dataset_root: false,
            }],
        },
    );

    let txn2 = Transaction::new_from_version(
        1,
        Operation::UpdateBases {
            new_bases: vec![lance_table::format::BasePath {
                id: 0,
                path: "s3://bucket2/path2".to_string(),
                name: Some("base2".to_string()),
                is_dataset_root: false,
            }],
        },
    );

    // Should not conflict despite both having zero IDs
    let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
        .await
        .unwrap();
    assert!(rebase.check_txn(&txn2, 2).is_ok());
}

#[test]
fn test_compacted_sstables_conflict_lower_generation_fails() {
    // Test: committed generation >= to_commit generation should be incompatible (no retry)
    let shard = Uuid::new_v4();

    // Committed has generation 10, we're trying to commit generation 5
    let committed_txn = Transaction::new(
        0,
        Operation::UpdateMemWalState {
            dir_id: String::new(),
            compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
        },
        None,
    );

    let to_commit_txn = Transaction::new(
        0,
        Operation::UpdateMemWalState {
            dir_id: String::new(),
            compacted_sstables: vec![CompactedSsTable::new(shard, 5)],
        },
        None,
    );

    let mut rebase = TransactionRebase {
        transaction: to_commit_txn,
        initial_fragments: HashMap::new(),
        modified_fragment_ids: HashSet::new(),
        affected_rows: None,
        frag_reuse_base: None,
        conflicting_mem_wal_compacted_sstables: Vec::new(),
        current_lineage: None,
        untagged_reuse: None,
        current_live: None,
        current_schema: None,
        read_fragments: None,
        read_schema: None,
        reuse: Default::default(),
        staged_replay: false,
    };

    let result = rebase.check_txn(&committed_txn, 1);
    assert!(
        matches!(result, Err(Error::IncompatibleTransaction { .. })),
        "Expected non-retryable IncompatibleTransaction for lower generation, got {:?}",
        result
    );
}

#[test]
fn test_compacted_sstables_conflict_equal_generation_fails() {
    // Test: committed generation == to_commit generation should be incompatible (no retry)
    let shard = Uuid::new_v4();

    let committed_txn = Transaction::new(
        0,
        Operation::UpdateMemWalState {
            dir_id: String::new(),
            compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
        },
        None,
    );

    let to_commit_txn = Transaction::new(
        0,
        Operation::UpdateMemWalState {
            dir_id: String::new(),
            compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
        },
        None,
    );

    let mut rebase = TransactionRebase {
        transaction: to_commit_txn,
        initial_fragments: HashMap::new(),
        modified_fragment_ids: HashSet::new(),
        affected_rows: None,
        frag_reuse_base: None,
        conflicting_mem_wal_compacted_sstables: Vec::new(),
        current_lineage: None,
        untagged_reuse: None,
        current_live: None,
        current_schema: None,
        read_fragments: None,
        read_schema: None,
        reuse: Default::default(),
        staged_replay: false,
    };

    let result = rebase.check_txn(&committed_txn, 1);
    assert!(
        matches!(result, Err(Error::IncompatibleTransaction { .. })),
        "Expected non-retryable IncompatibleTransaction for equal generation, got {:?}",
        result
    );
}

#[test]
fn test_compacted_sstables_conflict_higher_generation_retryable() {
    // Test: committed generation < to_commit generation should be retryable
    let shard = Uuid::new_v4();

    // Committed has generation 5, we're trying to commit generation 10
    let committed_txn = Transaction::new(
        0,
        Operation::UpdateMemWalState {
            dir_id: String::new(),
            compacted_sstables: vec![CompactedSsTable::new(shard, 5)],
        },
        None,
    );

    let to_commit_txn = Transaction::new(
        0,
        Operation::UpdateMemWalState {
            dir_id: String::new(),
            compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
        },
        None,
    );

    let mut rebase = TransactionRebase {
        transaction: to_commit_txn,
        initial_fragments: HashMap::new(),
        modified_fragment_ids: HashSet::new(),
        affected_rows: None,
        frag_reuse_base: None,
        conflicting_mem_wal_compacted_sstables: Vec::new(),
        current_lineage: None,
        untagged_reuse: None,
        current_live: None,
        current_schema: None,
        read_fragments: None,
        read_schema: None,
        reuse: Default::default(),
        staged_replay: false,
    };

    let result = rebase.check_txn(&committed_txn, 1);
    assert!(
        matches!(result, Err(Error::RetryableCommitConflict { .. })),
        "Expected retryable conflict for higher generation, got {:?}",
        result
    );
}

#[test]
fn test_compacted_sstables_different_shards_ok() {
    // Test: different shards should not conflict
    let shard1 = Uuid::new_v4();
    let shard2 = Uuid::new_v4();

    let committed_txn = Transaction::new(
        0,
        Operation::UpdateMemWalState {
            dir_id: String::new(),
            compacted_sstables: vec![CompactedSsTable::new(shard1, 10)],
        },
        None,
    );

    let to_commit_txn = Transaction::new(
        0,
        Operation::UpdateMemWalState {
            dir_id: String::new(),
            compacted_sstables: vec![CompactedSsTable::new(shard2, 5)],
        },
        None,
    );

    let mut rebase = TransactionRebase {
        transaction: to_commit_txn,
        initial_fragments: HashMap::new(),
        modified_fragment_ids: HashSet::new(),
        affected_rows: None,
        frag_reuse_base: None,
        conflicting_mem_wal_compacted_sstables: Vec::new(),
        current_lineage: None,
        untagged_reuse: None,
        current_live: None,
        current_schema: None,
        read_fragments: None,
        read_schema: None,
        reuse: Default::default(),
        staged_replay: false,
    };

    let result = rebase.check_txn(&committed_txn, 1);
    assert!(
        result.is_ok(),
        "Expected OK for different shards, got {:?}",
        result
    );
}

#[test]
fn test_update_mem_wal_state_vs_create_index_with_compacted_sstables() {
    use crate::index::mem_wal::new_mem_wal_index_meta;
    use lance_index::mem_wal::MemWalIndexDetails;

    let shard = Uuid::new_v4();

    // Create a MemWalIndex with compacted_sstables
    let details = MemWalIndexDetails {
        compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
        ..Default::default()
    };
    let mem_wal_index = new_mem_wal_index_meta(1, details).unwrap();

    // CreateIndex with MemWalIndex that has generation 10
    let committed_txn = Transaction::new(
        0,
        Operation::CreateIndex {
            new_indices: vec![mem_wal_index],
            removed_indices: vec![],
        },
        None,
    );

    // UpdateMemWalState trying to set generation 5 (lower than committed)
    let to_commit_txn = Transaction::new(
        0,
        Operation::UpdateMemWalState {
            dir_id: String::new(),
            compacted_sstables: vec![CompactedSsTable::new(shard, 5)],
        },
        None,
    );

    let mut rebase = TransactionRebase {
        transaction: to_commit_txn,
        initial_fragments: HashMap::new(),
        modified_fragment_ids: HashSet::new(),
        affected_rows: None,
        frag_reuse_base: None,
        conflicting_mem_wal_compacted_sstables: Vec::new(),
        current_lineage: None,
        untagged_reuse: None,
        current_live: None,
        current_schema: None,
        read_fragments: None,
        read_schema: None,
        reuse: Default::default(),
        staged_replay: false,
    };

    let result = rebase.check_txn(&committed_txn, 1);
    assert!(
        matches!(result, Err(Error::IncompatibleTransaction { .. })),
        "Expected non-retryable IncompatibleTransaction when UpdateMemWalState generation is lower than CreateIndex, got {:?}",
        result
    );

    // Now test with higher generation (should be retryable)
    let to_commit_txn_higher = Transaction::new(
        0,
        Operation::UpdateMemWalState {
            dir_id: String::new(),
            compacted_sstables: vec![CompactedSsTable::new(shard, 15)],
        },
        None,
    );

    let mut rebase_higher = TransactionRebase {
        transaction: to_commit_txn_higher,
        initial_fragments: HashMap::new(),
        modified_fragment_ids: HashSet::new(),
        affected_rows: None,
        frag_reuse_base: None,
        conflicting_mem_wal_compacted_sstables: Vec::new(),
        current_lineage: None,
        untagged_reuse: None,
        current_live: None,
        current_schema: None,
        read_fragments: None,
        read_schema: None,
        reuse: Default::default(),
        staged_replay: false,
    };

    let result_higher = rebase_higher.check_txn(&committed_txn, 1);
    assert!(
        matches!(result_higher, Err(Error::RetryableCommitConflict { .. })),
        "Expected retryable conflict when UpdateMemWalState generation is higher than CreateIndex, got {:?}",
        result_higher
    );
}

#[test]
fn test_create_index_vs_update_mem_wal_state_rebase() {
    use crate::index::mem_wal::new_mem_wal_index_meta;
    use lance_index::mem_wal::MemWalIndexDetails;

    let shard = Uuid::new_v4();

    // CreateIndex with MemWalIndex (no compacted_sstables initially)
    let details = MemWalIndexDetails::default();
    let mem_wal_index = new_mem_wal_index_meta(1, details).unwrap();

    let to_commit_txn = Transaction::new(
        0,
        Operation::CreateIndex {
            new_indices: vec![mem_wal_index],
            removed_indices: vec![],
        },
        None,
    );

    // UpdateMemWalState with generation 10
    let committed_txn = Transaction::new(
        0,
        Operation::UpdateMemWalState {
            dir_id: String::new(),
            compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
        },
        None,
    );

    let mut rebase = TransactionRebase {
        transaction: to_commit_txn,
        initial_fragments: HashMap::new(),
        modified_fragment_ids: HashSet::new(),
        affected_rows: None,
        frag_reuse_base: None,
        conflicting_mem_wal_compacted_sstables: Vec::new(),
        current_lineage: None,
        untagged_reuse: None,
        current_live: None,
        current_schema: None,
        read_fragments: None,
        read_schema: None,
        reuse: Default::default(),
        staged_replay: false,
    };

    // CreateIndex of MemWalIndex should be compatible with UpdateMemWalState
    // and should collect the compacted_sstables for rebasing
    let result = rebase.check_txn(&committed_txn, 1);
    assert!(
        result.is_ok(),
        "Expected OK for CreateIndex vs UpdateMemWalState, got {:?}",
        result
    );

    // Verify that compacted_sstables were collected
    assert_eq!(rebase.conflicting_mem_wal_compacted_sstables.len(), 1);
    assert_eq!(
        rebase.conflicting_mem_wal_compacted_sstables[0].shard_id,
        shard
    );
    assert_eq!(
        rebase.conflicting_mem_wal_compacted_sstables[0].generation,
        10
    );
}

#[tokio::test]
async fn test_concurrent_overwrites_retryable() {
    let dataset = test_dataset(5, 1).await;
    let dataset_v1_reader1 = Arc::new(dataset.checkout_version(1).await.unwrap());
    let dataset_v1_reader2 = Arc::new(dataset.checkout_version(1).await.unwrap());

    let data = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(10..15)),
            Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(1, 5))),
        ],
    )
    .unwrap();

    // First overwrite succeeds
    let txn1 = InsertBuilder::new(dataset_v1_reader1.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        })
        .execute_uncommitted(vec![data.clone()])
        .await
        .unwrap();
    let dataset_v2 = CommitBuilder::new(dataset_v1_reader1)
        .execute(txn1)
        .await
        .unwrap();
    assert_eq!(dataset_v2.manifest.version, 2);

    // Second overwrite should fail with retryable conflict
    let txn2 = InsertBuilder::new(dataset_v1_reader2.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        })
        .execute_uncommitted(vec![data])
        .await
        .unwrap();
    let result = CommitBuilder::new(dataset_v1_reader2).execute(txn2).await;
    assert!(
        matches!(result, Err(Error::RetryableCommitConflict { .. })),
        "Expected RetryableCommitConflict but got: {:?}",
        result
    );

    assert_eq!(dataset_v2.count_rows(None).await.unwrap(), 5);
}
