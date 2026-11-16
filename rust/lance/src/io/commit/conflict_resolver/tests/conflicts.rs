// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[test]
fn test_conflicts() {
    use io::commit::conflict_resolver::tests::{ConflictResult::*, modified_fragment_ids};

    let index0 = IndexMetadata {
        uuid: uuid::Uuid::new_v4(),
        name: "test".to_string(),
        fields: vec![0],
        covering_fields: vec![],
        dataset_version: 1,
        fragment_bitmap: None,
        index_details: None,
        index_version: 0,
        created_at: None, // Test index, not setting timestamp
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let fragment0 = Fragment::new(0);
    let fragment1 = Fragment::new(1);
    let fragment2 = Fragment::new(2);
    // The transactions that will be checked against
    let other_operations = [
        Operation::Append {
            fragments: vec![fragment0.clone()],
        },
        Operation::CreateIndex {
            new_indices: vec![index0.clone()],
            removed_indices: vec![index0.clone()],
        },
        Operation::Delete {
            updated_fragments: vec![fragment0.clone()],
            deleted_fragment_ids: vec![2],
            predicate: "x > 2".to_string(),
        },
        Operation::Merge {
            fragments: vec![fragment0.clone(), fragment2.clone()],
            schema: lance_core::datatypes::Schema::default(),
            preserves_nullability: true,
        },
        Operation::Overwrite {
            fragments: vec![fragment0.clone(), fragment2.clone()],
            schema: lance_core::datatypes::Schema::default(),
            config_upsert_values: Some(HashMap::from_iter(vec![(
                "overwrite-key".to_string(),
                "value".to_string(),
            )])),
            initial_bases: None,
        },
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: vec![fragment0.clone()],
                new_fragments: vec![fragment1.clone()],
            }],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        },
        Operation::ReserveFragments { num_fragments: 3 },
        Operation::Update {
            removed_fragment_ids: vec![1],
            updated_fragments: vec![fragment0.clone()],
            new_fragments: vec![fragment2.clone()],
            fields_modified: vec![0],
            mem_wal_dir_id: None,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: None,
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        },
        create_update_config_for_test(
            Some(HashMap::from_iter(vec![(
                "lance.test".to_string(),
                "value".to_string(),
            )])),
            Some(vec!["remove-key".to_string()]),
            Some(HashMap::from_iter(vec![(
                "schema-key".to_string(),
                "schema-value".to_string(),
            )])),
            Some(HashMap::from_iter(vec![(
                0,
                HashMap::from_iter(vec![("field-key".to_string(), "field-value".to_string())]),
            )])),
        ),
    ];
    let other_transactions = other_operations
        .iter()
        .map(|op| Transaction::new(0, op.clone(), None))
        .collect::<Vec<_>>();

    // Transactions and whether they are expected to conflict with each
    // of other_transactions
    let cases = [
        (
            Operation::Append {
                fragments: vec![fragment0.clone()],
            },
            [
                Compatible,    // append
                Compatible,    // create index
                Compatible,    // delete
                Compatible,    // merge
                NotCompatible, // overwrite
                Compatible,    // rewrite
                Compatible,    // reserve
                Compatible,    // update
                Compatible,    // update config
            ],
        ),
        (
            Operation::Delete {
                // Delete that affects fragments different from other transactions
                updated_fragments: vec![fragment1.clone()],
                deleted_fragment_ids: vec![],
                predicate: "x > 2".to_string(),
            },
            [
                Compatible,    // append
                Compatible,    // create index
                Compatible,    // delete
                Retryable,     // merge
                NotCompatible, // overwrite
                Compatible,    // rewrite
                Compatible,    // reserve
                Retryable,     // update
                Compatible,    // update config
            ],
        ),
        (
            Operation::Delete {
                // Delete that affects same fragments as other transactions
                updated_fragments: vec![fragment0.clone(), fragment2.clone()],
                deleted_fragment_ids: vec![],
                predicate: "x > 2".to_string(),
            },
            [
                Compatible,    // append
                Compatible,    // create index
                Retryable,     // delete
                Retryable,     // merge
                NotCompatible, // overwrite
                Retryable,     // rewrite
                Compatible,    // reserve
                Retryable,     // update
                Compatible,    // update config
            ],
        ),
        (
            Operation::Overwrite {
                fragments: vec![fragment0.clone(), fragment2.clone()],
                schema: lance_core::datatypes::Schema::default(),
                config_upsert_values: None,
                initial_bases: None,
            },
            // Concurrent overwrites are retryable so user can decide
            // if their overwrite should still proceed.
            [
                Compatible, // append
                Compatible, // create index
                Compatible, // delete
                Compatible, // merge
                Retryable,  // overwrite
                Compatible, // rewrite
                Compatible, // reserve
                Compatible, // update
                Compatible, // update config
            ],
        ),
        (
            Operation::CreateIndex {
                new_indices: vec![index0.clone()],
                removed_indices: vec![index0],
            },
            // Conflicts with row-id-changing operations and same-name CreateIndex.
            [
                Compatible,    // append
                Retryable,     // create index
                Compatible,    // delete
                Compatible,    // merge
                NotCompatible, // overwrite
                Retryable,     // rewrite
                Compatible,    // reserve
                Compatible,    // update
                Compatible,    // update config
            ],
        ),
        (
            // Rewrite that affects different fragments
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: vec![fragment1],
                    new_fragments: vec![fragment0.clone()],
                }],
                rewritten_indices: Vec::new(),
                frag_reuse_index: None,
            },
            [
                Compatible,    // append
                Retryable,     // create index
                Compatible,    // delete
                Retryable,     // merge
                NotCompatible, // overwrite
                Compatible,    // rewrite
                Compatible,    // reserve
                Retryable,     // update
                Compatible,    // update config
            ],
        ),
        (
            // Rewrite that affects the same fragments
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: vec![fragment0.clone(), fragment2.clone()],
                    new_fragments: vec![fragment0.clone()],
                }],
                rewritten_indices: Vec::new(),
                frag_reuse_index: None,
            },
            [
                Compatible,    // append
                Retryable,     // create index
                Retryable,     // delete
                Retryable,     // merge
                NotCompatible, // overwrite
                Retryable,     // rewrite
                Compatible,    // reserve
                Retryable,     // update
                Compatible,    // update config
            ],
        ),
        (
            Operation::Merge {
                fragments: vec![fragment0.clone(), fragment2.clone()],
                schema: lance_core::datatypes::Schema::default(),
                preserves_nullability: true,
            },
            // Merge also conflicts with schema and field metadata updates.
            [
                Retryable,     // append
                Compatible,    // create index
                Retryable,     // delete
                Retryable,     // merge
                NotCompatible, // overwrite
                Retryable,     // rewrite
                Compatible,    // reserve
                Retryable,     // update
                Retryable,     // update config
            ],
        ),
        (
            Operation::ReserveFragments { num_fragments: 2 },
            // ReserveFragments only conflicts with Overwrite and Restore.
            [
                Compatible,    // append
                Compatible,    // create index
                Compatible,    // delete
                Compatible,    // merge
                NotCompatible, // overwrite
                Compatible,    // rewrite
                Compatible,    // reserve
                Compatible,    // update
                Compatible,    // update config
            ],
        ),
        (
            Operation::Update {
                // Update that affects same fragments as other transactions
                updated_fragments: vec![fragment0],
                removed_fragment_ids: vec![],
                new_fragments: vec![fragment2],
                fields_modified: vec![0],
                mem_wal_dir_id: None,
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
            [
                Compatible,    // append
                Compatible,    // create index
                Retryable,     // delete
                Retryable,     // merge
                NotCompatible, // overwrite
                Retryable,     // rewrite
                Compatible,    // reserve
                Retryable,     // update
                Compatible,    // update config
            ],
        ),
        (
            // Update config that should not conflict with anything
            create_update_config_for_test(
                Some(HashMap::from_iter(vec![(
                    "other-key".to_string(),
                    "new-value".to_string(),
                )])),
                None,
                None,
                None,
            ),
            [Compatible; 9],
        ),
        (
            // Update config that conflicts with key being upserted by other UpdateConfig operation
            create_update_config_for_test(
                Some(HashMap::from_iter(vec![(
                    "lance.test".to_string(),
                    "new-value".to_string(),
                )])),
                None,
                None,
                None,
            ),
            [
                Compatible,    // append
                Compatible,    // create index
                Compatible,    // delete
                Compatible,    // merge
                Compatible,    // overwrite
                Compatible,    // rewrite
                Compatible,    // reserve
                Compatible,    // update
                NotCompatible, // update config
            ],
        ),
        (
            // Update config that conflicts with key being deleted by other UpdateConfig operation
            create_update_config_for_test(
                Some(HashMap::from_iter(vec![(
                    "remove-key".to_string(),
                    "new-value".to_string(),
                )])),
                None,
                None,
                None,
            ),
            [
                Compatible,    // append
                Compatible,    // create index
                Compatible,    // delete
                Compatible,    // merge
                Compatible,    // overwrite
                Compatible,    // rewrite
                Compatible,    // reserve
                Compatible,    // update
                NotCompatible, // update config
            ],
        ),
        (
            // Delete config keys currently being deleted by other UpdateConfig operation
            create_update_config_for_test(None, Some(vec!["remove-key".to_string()]), None, None),
            [Compatible; 9],
        ),
        (
            // Delete config keys currently being upserted by other UpdateConfig operation
            create_update_config_for_test(None, Some(vec!["lance.test".to_string()]), None, None),
            [
                Compatible,    // append
                Compatible,    // create index
                Compatible,    // delete
                Compatible,    // merge
                Compatible,    // overwrite
                Compatible,    // rewrite
                Compatible,    // reserve
                Compatible,    // update
                NotCompatible, // update config
            ],
        ),
        (
            // Changing schema metadata conflicts with another update changing schema
            // metadata or with an overwrite
            create_update_config_for_test(
                None,
                None,
                Some(HashMap::from_iter(vec![(
                    "schema-key".to_string(),
                    "new-value".to_string(),
                )])),
                None,
            ),
            [
                Compatible,    // append
                Compatible,    // create index
                Compatible,    // delete
                Retryable,     // merge
                NotCompatible, // overwrite
                Compatible,    // rewrite
                Compatible,    // reserve
                Compatible,    // update
                NotCompatible, // update config
            ],
        ),
        (
            // Changing field metadata conflicts with another update changing same field
            // metadata or overwrite
            create_update_config_for_test(
                None,
                None,
                None,
                Some(HashMap::from_iter(vec![(
                    0,
                    HashMap::from_iter(vec![("field_key".to_string(), "field_value".to_string())]),
                )])),
            ),
            [
                Compatible,    // append
                Compatible,    // create index
                Compatible,    // delete
                Retryable,     // merge
                NotCompatible, // overwrite
                Compatible,    // rewrite
                Compatible,    // reserve
                Compatible,    // update
                NotCompatible, // update config
            ],
        ),
        (
            // Updates to different field metadata are allowed
            create_update_config_for_test(
                None,
                None,
                None,
                Some(HashMap::from_iter(vec![(
                    1,
                    HashMap::from_iter(vec![("field_key".to_string(), "field_value".to_string())]),
                )])),
            ),
            [
                Compatible,    // append
                Compatible,    // create index
                Compatible,    // delete
                Retryable,     // merge
                NotCompatible, // overwrite
                Compatible,    // rewrite
                Compatible,    // reserve
                Compatible,    // update
                Compatible,    // update config
            ],
        ),
    ];

    // An operation written by a newer Lance may touch anything, so every
    // known operation must treat it as a conflict.
    let unknown = Transaction::try_from(lance_table::format::pb::Transaction {
        uuid: "unknown".to_string(),
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(unknown.operation, Operation::Unknown { .. }));

    for (operation, expected_conflicts) in &cases {
        let transaction = Transaction::new(0, operation.clone(), None);
        let mut rebase = TransactionRebase {
            transaction,
            initial_fragments: HashMap::new(),
            modified_fragment_ids: modified_fragment_ids(operation).collect::<HashSet<_>>(),
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

        let result = rebase.check_txn(&unknown, 1);
        assert!(
            matches!(result, Err(Error::IncompatibleTransaction { .. })),
            "Transaction {:?} should be incompatible with an unknown operation, but was {:?}",
            operation,
            result
        );

        for (other, expected_conflict) in other_transactions.iter().zip(expected_conflicts) {
            match expected_conflict {
                Compatible => {
                    let result = rebase.check_txn(other, 1);
                    assert!(
                        result.is_ok(),
                        "Transaction {:?} should {:?} with {:?}, but was {:?}",
                        operation,
                        expected_conflict,
                        other,
                        result
                    )
                }
                NotCompatible => {
                    let result = rebase.check_txn(other, 1);
                    assert!(
                        matches!(result, Err(Error::IncompatibleTransaction { .. })),
                        "Transaction {:?} should be {:?} with {:?}, but was: {:?}",
                        operation,
                        expected_conflict,
                        other,
                        result
                    )
                }
                Retryable => {
                    let result = rebase.check_txn(other, 1);
                    assert!(
                        matches!(result, Err(Error::RetryableCommitConflict { .. })),
                        "Transaction {:?} should be {:?} with {:?}, but was {:?}",
                        operation,
                        expected_conflict,
                        other,
                        result
                    )
                }
            }
        }
    }
}

#[test]
fn test_data_overlay_conflicts() {
    use crate::dataset::transaction::{DataOverlayGroup, UpdateMode};
    use ConflictResult::*;
    use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
    use roaring::RoaringBitmap;

    // Our transaction overlays fragment 1.
    let overlay_op = |fragment_id: u64| Operation::DataOverlay {
        groups: vec![DataOverlayGroup {
            fragment_id,
            overlays: vec![DataOverlayFile {
                data_file: DataFile::new_legacy_from_fields("overlay.lance", vec![0], None),
                coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
                committed_version: 0,
            }],
        }],
    };
    let update_removing = |removed_fragment_ids: Vec<u64>| Operation::Update {
        removed_fragment_ids,
        updated_fragments: vec![],
        new_fragments: vec![],
        fields_modified: vec![],
        mem_wal_dir_id: None,
        compacted_sstables: Vec::new(),
        fields_for_preserving_frag_bitmap: vec![],
        update_mode: None,
        inserted_rows_filter: None,
        updated_fragment_offsets: None,
    };
    let delete = |updated: Vec<Fragment>, deleted: Vec<u64>| Operation::Delete {
        updated_fragments: updated,
        deleted_fragment_ids: deleted,
        predicate: "x > 2".to_string(),
    };
    // A row-moving update (RewriteRows) relocates the updated rows into
    // new_fragments; an in-place column rewrite (RewriteColumns) leaves rows
    // where they are.
    let update_moving = |updated: Vec<Fragment>, new: Vec<Fragment>| Operation::Update {
        removed_fragment_ids: vec![],
        updated_fragments: updated,
        new_fragments: new,
        fields_modified: vec![],
        mem_wal_dir_id: None,
        compacted_sstables: Vec::new(),
        fields_for_preserving_frag_bitmap: vec![],
        update_mode: Some(UpdateMode::RewriteRows),
        inserted_rows_filter: None,
        updated_fragment_offsets: None,
    };
    let update_rewrite_columns = |updated: Vec<Fragment>| Operation::Update {
        removed_fragment_ids: vec![],
        updated_fragments: updated,
        new_fragments: vec![],
        fields_modified: vec![0],
        mem_wal_dir_id: None,
        compacted_sstables: Vec::new(),
        fields_for_preserving_frag_bitmap: vec![],
        update_mode: Some(UpdateMode::RewriteColumns),
        inserted_rows_filter: None,
        updated_fragment_offsets: None,
    };
    let rewrite_of = |old: &Fragment| Operation::Rewrite {
        groups: vec![RewriteGroup {
            old_fragments: vec![old.clone()],
            new_fragments: vec![],
        }],
        rewritten_indices: vec![],
        frag_reuse_index: None,
    };

    let fragment0 = Fragment::new(0);
    let fragment1 = Fragment::new(1);
    // Fields 0 and 1; our overlay covers field 0.
    let two_field_schema =
        lance_core::datatypes::Schema::try_from(&arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("a", arrow_schema::DataType::Int32, true),
            arrow_schema::Field::new("b", arrow_schema::DataType::Int32, true),
        ]))
        .unwrap();

    // Each case is checked against our overlay on fragment 1.
    let cases: Vec<(Operation, ConflictResult)> = vec![
        // Permissive: preserves physical offsets / leaves fragment 1 in place.
        (
            Operation::Append {
                fragments: vec![fragment0.clone()],
            },
            Compatible,
        ),
        (
            Operation::CreateIndex {
                new_indices: vec![],
                removed_indices: vec![],
            },
            Compatible,
        ),
        (
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(
                    1,
                    DataFile::new_legacy_from_fields("r.lance", vec![0], None),
                )],
            },
            Compatible,
        ),
        // Another overlay on the same fragment stacks rather than conflicts.
        (overlay_op(1), Compatible),
        // A Delete only tombstones rows (deletion vector) on fragment 1, and
        // an in-place column rewrite preserves offsets, so both are compatible.
        (delete(vec![fragment1.clone()], vec![]), Compatible),
        (update_rewrite_columns(vec![fragment1.clone()]), Compatible),
        (update_removing(vec![2]), Compatible),
        // ...but removing our overlaid fragment 1 orphans the overlay -> conflict.
        (delete(vec![], vec![1]), Retryable),
        (update_removing(vec![1]), Retryable),
        // A row-moving update re-creates the rows it touches from the
        // pre-overlay base. Whether that actually drops any overlaid cell is
        // a per-row question answered in `finish_data_overlay` (see
        // test_data_overlay_finish_conflicts_with_row_moving_update), so the
        // check itself defers rather than conflicting; a moving update on any
        // fragment is compatible at this stage.
        (
            update_moving(vec![fragment1.clone()], vec![fragment0.clone()]),
            Compatible,
        ),
        (
            update_moving(vec![fragment0.clone()], vec![fragment0.clone()]),
            Compatible,
        ),
        // Rewriting fragment 1 invalidates its physical offsets -> conflict;
        // a rewrite of a different fragment does not.
        (rewrite_of(&fragment1), Retryable),
        (rewrite_of(&fragment0), Compatible),
        // Merge rewrites the whole fragment list; Restore replaces the dataset.
        (
            Operation::Merge {
                fragments: vec![fragment1.clone()],
                schema: lance_core::datatypes::Schema::default(),
                preserves_nullability: true,
            },
            Retryable,
        ),
        (Operation::Restore { version: 1 }, NotCompatible),
        // A projection that keeps our overlaid field 0 is compatible; one that
        // drops it would leave our overlay covering a missing field.
        (
            Operation::Project {
                schema: two_field_schema.clone(),
                preserves_nullability: true,
            },
            Compatible,
        ),
        (
            Operation::Project {
                schema: two_field_schema.project_by_ids(&[1], true),
                preserves_nullability: true,
            },
            Retryable,
        ),
        // Overwrite/Restore replace the dataset, and UpdateMemWalState does
        // not rebase against data operations — all hard conflicts.
        (
            Operation::Overwrite {
                fragments: vec![fragment0.clone()],
                schema: lance_core::datatypes::Schema::default(),
                config_upsert_values: None,
                initial_bases: None,
            },
            NotCompatible,
        ),
        (
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![],
            },
            NotCompatible,
        ),
    ];

    for (other, expected) in cases {
        let mut rebase = TransactionRebase {
            transaction: Transaction::new(0, overlay_op(1), None),
            initial_fragments: HashMap::new(),
            modified_fragment_ids: modified_fragment_ids(&overlay_op(1)).collect::<HashSet<_>>(),
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
        let other_txn = Transaction::new(0, other.clone(), None);
        let result = rebase.check_txn(&other_txn, 1);
        match expected {
            Compatible => assert!(
                result.is_ok(),
                "overlay should be compatible with {other:?}, got {result:?}"
            ),
            Retryable => assert!(
                matches!(result, Err(Error::RetryableCommitConflict { .. })),
                "overlay should retryably conflict with {other:?}, got {result:?}"
            ),
            NotCompatible => assert!(
                matches!(result, Err(Error::IncompatibleTransaction { .. })),
                "overlay should be incompatible with {other:?}, got {result:?}"
            ),
        }
    }
}

#[test]
fn test_rewrite_conflicts_with_data_overlay() {
    // Reverse direction of test_data_overlay_conflicts: our transaction is a
    // Rewrite and a concurrent DataOverlay has already committed. A rewrite
    // changes the physical row addresses of the fragments it touches, so an
    // overlay on one of those fragments is invalidated (retryable); an
    // overlay on any other fragment is unaffected.
    use crate::dataset::transaction::DataOverlayGroup;
    use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
    use roaring::RoaringBitmap;

    let overlay_on = |fragment_id: u64| Operation::DataOverlay {
        groups: vec![DataOverlayGroup {
            fragment_id,
            overlays: vec![DataOverlayFile {
                data_file: DataFile::new_legacy_from_fields("overlay.lance", vec![0], None),
                coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
                committed_version: 0,
            }],
        }],
    };
    // Our transaction rewrites fragment 1.
    let rewrite_op = Operation::Rewrite {
        groups: vec![RewriteGroup {
            old_fragments: vec![Fragment::new(1)],
            new_fragments: vec![],
        }],
        rewritten_indices: vec![],
        frag_reuse_index: None,
    };

    for (other, expect_conflict) in [(overlay_on(1), true), (overlay_on(0), false)] {
        let mut rebase = TransactionRebase {
            transaction: Transaction::new(0, rewrite_op.clone(), None),
            initial_fragments: HashMap::new(),
            modified_fragment_ids: modified_fragment_ids(&rewrite_op).collect::<HashSet<_>>(),
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
        let other_txn = Transaction::new(0, other.clone(), None);
        let result = rebase.check_txn(&other_txn, 1);
        if expect_conflict {
            assert!(
                matches!(result, Err(Error::RetryableCommitConflict { .. })),
                "rewrite of fragment 1 should retryably conflict with {other:?}, got {result:?}"
            );
        } else {
            assert!(
                result.is_ok(),
                "rewrite of fragment 1 should not conflict with {other:?}, got {result:?}"
            );
        }
    }
}

#[test]
fn test_update_conflicts_with_data_overlay() {
    // Reverse direction of test_data_overlay_conflicts: our transaction is an
    // Update and a concurrent DataOverlay has already committed. A row-moving
    // update relocates the rows it touches, so an overlay on one of those
    // fragments can no longer be applied (retryable); an overlay on any other
    // fragment is compatible. An in-place column rewrite preserves rows but
    // replaces the whole column from its own snapshot, so it conflicts
    // whenever the overlay covers a fragment and field it rewrote.
    use crate::dataset::transaction::{DataOverlayGroup, UpdateMode};
    use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
    use roaring::RoaringBitmap;

    let overlay_on_field = |fragment_id: u64, field: i32| Operation::DataOverlay {
        groups: vec![DataOverlayGroup {
            fragment_id,
            overlays: vec![DataOverlayFile {
                data_file: DataFile::new_legacy_from_fields("overlay.lance", vec![field], None),
                coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
                committed_version: 0,
            }],
        }],
    };
    let overlay_on = |fragment_id: u64| overlay_on_field(fragment_id, 0);
    // Our update always touches fragment 1.
    let update =
        |update_mode: Option<UpdateMode>, new_fragments: Vec<Fragment>| Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![Fragment::new(1)],
            new_fragments,
            fields_modified: vec![0],
            mem_wal_dir_id: None,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode,
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        };

    // The overlay covers physical offset 0 of its fragment. Row addresses
    // pack the fragment id in the high 32 bits and the offset in the low 32.
    let rows_on = |fragment_id: u64, offsets: &[u32]| {
        let mut map = RowAddrTreeMap::new();
        map.insert_bitmap(
            fragment_id as u32,
            RoaringBitmap::from_iter(offsets.iter().copied()),
        );
        map
    };

    // (update, committed overlay, moved rows the update carries, expect conflict)
    let cases = [
        // Row-moving update whose moved rows include the overlaid cell -> the
        // update would undo the overlay, so conflict.
        (
            update(Some(UpdateMode::RewriteRows), vec![Fragment::new(2)]),
            overlay_on(1),
            Some(rows_on(1, &[0])),
            true,
        ),
        // ...but if the moved rows miss the overlaid cell, the overlay survives.
        (
            update(Some(UpdateMode::RewriteRows), vec![Fragment::new(2)]),
            overlay_on(1),
            Some(rows_on(1, &[5])),
            false,
        ),
        // An overlay on a fragment the update did not touch is fine.
        (
            update(Some(UpdateMode::RewriteRows), vec![Fragment::new(2)]),
            overlay_on(0),
            Some(rows_on(1, &[0])),
            false,
        ),
        // An in-place column rewrite replaces field 0 across all of fragment
        // 1 from its own snapshot, and `build_manifest` tombstones the
        // overlay for that field, so the overlay's value would be lost even
        // though it sits on a row the update never matched -> conflict.
        (
            update(Some(UpdateMode::RewriteColumns), vec![]),
            overlay_on(1),
            Some(rows_on(1, &[0])),
            true,
        ),
        // ...and the coverage is irrelevant: an overlay on a row the update
        // did not match is exactly the case that gets silently dropped.
        (
            update(Some(UpdateMode::RewriteColumns), vec![]),
            overlay_on(1),
            Some(rows_on(1, &[5])),
            true,
        ),
        // An overlay on a field the rewrite did not touch survives the
        // tombstoning, so it stays compatible.
        (
            update(Some(UpdateMode::RewriteColumns), vec![]),
            overlay_on_field(1, 7),
            Some(rows_on(1, &[0])),
            false,
        ),
        // So does an overlay on a fragment the rewrite did not touch.
        (
            update(Some(UpdateMode::RewriteColumns), vec![]),
            overlay_on(0),
            Some(rows_on(1, &[0])),
            false,
        ),
        // Without affected rows we cannot be precise, so a row-moving update
        // on the overlaid fragment falls back to a conservative conflict.
        (
            update(Some(UpdateMode::RewriteRows), vec![Fragment::new(2)]),
            overlay_on(1),
            None,
            true,
        ),
    ];

    for (update_op, other, affected_rows, expect_conflict) in cases {
        let mut rebase = TransactionRebase {
            transaction: Transaction::new(0, update_op.clone(), None),
            initial_fragments: HashMap::new(),
            modified_fragment_ids: modified_fragment_ids(&update_op).collect::<HashSet<_>>(),
            affected_rows: affected_rows.as_ref(),
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
        let other_txn = Transaction::new(0, other.clone(), None);
        let result = rebase.check_txn(&other_txn, 1);
        if expect_conflict {
            assert!(
                matches!(result, Err(Error::RetryableCommitConflict { .. })),
                "update should retryably conflict with {other:?}, got {result:?}"
            );
        } else {
            assert!(
                result.is_ok(),
                "update should be compatible with {other:?}, got {result:?}"
            );
        }
    }
}

/// Appends can cross merges only if their written fields survive and no
/// nullability claim is invalidated. A projection can drop written fields,
/// since it writes no replacement data. A merge in the reverse order always
/// conflicts because it replaces the fragment list.
#[rstest::rstest]
#[case::surviving_fields(vec![0, 1, 2], false)]
#[case::removed_field(vec![0, 3], true)]
#[case::removed_nested_field(vec![0, 1, 3], true)]
#[case::unwritten_field(vec![0], false)]
#[case::reserved_ids(vec![0, -2, -3, -4, -5], false)]
fn test_schema_change_blocks_stale_append(
    #[case] written_fields: Vec<i32>,
    #[case] field_removed: bool,
    #[values(false, true)] project: bool,
) {
    for claims in [true, false] {
        let schema = (&Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new_struct("s", vec![Field::new("x", DataType::Int64, true)], true),
        ]))
            .try_into()
            .unwrap();
        let schema_change = if project {
            Operation::Project {
                schema,
                preserves_nullability: !claims,
            }
        } else {
            Operation::Merge {
                fragments: vec![Fragment::new(0)],
                schema,
                preserves_nullability: !claims,
            }
        };
        let append = Operation::Append {
            fragments: vec![Fragment {
                files: vec![DataFile::new_legacy_from_fields(
                    "append.lance",
                    written_fields.clone(),
                    None,
                )],
                ..Fragment::new(1)
            }],
        };

        let mut append_rebase = TransactionRebase {
            transaction: Transaction::new(0, append.clone(), None),
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
        let result = append_rebase.check_txn(&Transaction::new(0, schema_change.clone(), None), 1);
        assert_eq!(
            matches!(result, Err(Error::RetryableCommitConflict { .. })),
            claims || (!project && field_removed),
            "append rebasing over {schema_change}/claims={claims}: got {result:?}"
        );

        let mut schema_rebase = TransactionRebase {
            transaction: Transaction::new(0, schema_change.clone(), None),
            initial_fragments: HashMap::new(),
            modified_fragment_ids: HashSet::from_iter([0]),
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
        let result = schema_rebase.check_txn(&Transaction::new(0, append, None), 1);
        assert_eq!(
            matches!(result, Err(Error::RetryableCommitConflict { .. })),
            claims || !project,
            "{schema_change} rebasing over append/claims={claims}: got {result:?}"
        );
    }
}

/// A claim conflicts with any write that can supply values, either order.
#[test]
fn test_non_null_claim_barrier() {
    use crate::dataset::transaction::{DataOverlayGroup, UpdateMode};
    use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
    use roaring::RoaringBitmap;

    let file = || DataFile::new_legacy_from_fields("w.lance", vec![0], None);
    let writers = [
        (
            "append",
            Operation::Append {
                fragments: vec![Fragment::new(1)],
            },
        ),
        (
            "update",
            Operation::Update {
                removed_fragment_ids: vec![],
                updated_fragments: vec![Fragment::new(0)],
                new_fragments: vec![],
                fields_modified: vec![0],
                mem_wal_dir_id: None,
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(UpdateMode::RewriteColumns),
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
        ),
        (
            "replacement",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, file())],
            },
        ),
        (
            "overlay",
            Operation::DataOverlay {
                groups: vec![DataOverlayGroup {
                    fragment_id: 0,
                    overlays: vec![DataOverlayFile {
                        data_file: file(),
                        coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
                        committed_version: 0,
                    }],
                }],
            },
        ),
    ];

    for (writer_name, writer) in &writers {
        for claims in [true, false] {
            // Keeps field 0, the one every writer here names: this case
            // is about the nullability claim, not about dropped fields.
            let project = Operation::Project {
                schema: (&Schema::new(vec![Field::new("a", DataType::Int32, true)]))
                    .try_into()
                    .unwrap(),
                preserves_nullability: !claims,
            };
            for (order, ours, theirs) in [
                ("project-rebasing", project.clone(), writer.clone()),
                ("writer-rebasing", writer.clone(), project.clone()),
            ] {
                let mut rebase = TransactionRebase {
                    transaction: Transaction::new(0, ours.clone(), None),
                    initial_fragments: HashMap::new(),
                    modified_fragment_ids: modified_fragment_ids(&ours).collect::<HashSet<_>>(),
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
                let result = rebase.check_txn(&Transaction::new(0, theirs, None), 1);
                assert_eq!(
                    matches!(result, Err(Error::RetryableCommitConflict { .. })),
                    claims,
                    "{writer_name}/claims={claims}/{order}: got {result:?}"
                );
            }
        }
    }
}

/// An in-place column rewrite cannot land on a projection that dropped the
/// field it rewrote: the fragment would keep a replacement data file for a
/// field the schema no longer has. `check_data_replacement_txn` has always
/// refused this for the other operation that rewrites a field in place.
#[rstest::rstest]
#[case::field_survives_the_projection(0, false)]
#[case::field_dropped_by_the_projection(1, true)]
fn test_column_rewrite_conflicts_with_a_projection_that_dropped_its_field(
    #[case] field_modified: u32,
    #[case] expect_conflict: bool,
) {
    use crate::dataset::transaction::UpdateMode;

    let rewrite = Operation::Update {
        removed_fragment_ids: vec![],
        updated_fragments: vec![Fragment::new(0)],
        new_fragments: vec![],
        fields_modified: vec![field_modified],
        mem_wal_dir_id: None,
        compacted_sstables: Vec::new(),
        fields_for_preserving_frag_bitmap: vec![],
        update_mode: Some(UpdateMode::RewriteColumns),
        inserted_rows_filter: None,
        updated_fragment_offsets: None,
    };
    // The projection keeps field 0 and nothing else.
    let project = Operation::Project {
        schema: (&Schema::new(vec![Field::new("a", DataType::Int32, true)]))
            .try_into()
            .unwrap(),
        preserves_nullability: true,
    };
    let mut rebase = TransactionRebase {
        transaction: Transaction::new(0, rewrite.clone(), None),
        initial_fragments: HashMap::new(),
        modified_fragment_ids: modified_fragment_ids(&rewrite).collect::<HashSet<_>>(),
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
    let result = rebase.check_txn(&Transaction::new(0, project, None), 1);
    assert_eq!(
        matches!(result, Err(Error::RetryableCommitConflict { .. })),
        expect_conflict,
        "got {result:?}"
    );
}

#[tokio::test]
#[rstest::rstest]
#[case::coverage_overlaps_moved_row(vec![0u32], true)]
#[case::coverage_disjoint_from_moved_row(vec![3u32], false)]
async fn test_data_overlay_finish_conflicts_with_row_moving_update(
    #[case] coverage_offsets: Vec<u32>,
    #[case] expect_conflict: bool,
) {
    // 5 rows in one fragment. A concurrent RewriteRows update moves row 0 out
    // to a new fragment (deleting it from fragment 0). Our overlay on fragment
    // 0 conflicts only when its coverage includes the moved row; the decision
    // is made in finish, which reads the deletion vectors.
    use crate::dataset::transaction::{DataOverlayGroup, UpdateMode};
    use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
    use roaring::RoaringBitmap;

    let dataset = test_dataset(5, 1).await;
    let mut fragment = dataset.fragments().as_slice()[0].clone();

    let moved_fragment = Fragment::new(0)
        .with_file(
            "moved.lance",
            vec![0],
            vec![0],
            LanceFileVersion::Stable.resolve(),
            NonZero::new(10),
        )
        .with_physical_rows(1);
    let update_op = Operation::Update {
        updated_fragments: vec![apply_deletion(&[0], &mut fragment, &dataset).await],
        removed_fragment_ids: vec![],
        new_fragments: vec![moved_fragment],
        fields_modified: vec![],
        mem_wal_dir_id: None,
        compacted_sstables: Vec::new(),
        fields_for_preserving_frag_bitmap: vec![],
        update_mode: Some(UpdateMode::RewriteRows),
        inserted_rows_filter: None,
        updated_fragment_offsets: None,
    };
    let update_txn = Transaction::new_from_version(dataset.manifest.version, update_op);

    let overlay_op = Operation::DataOverlay {
        groups: vec![DataOverlayGroup {
            fragment_id: 0,
            overlays: vec![DataOverlayFile {
                data_file: DataFile::new_legacy_from_fields("overlay.lance", vec![0], None),
                coverage: OverlayCoverage::dense(RoaringBitmap::from_iter(coverage_offsets)),
                committed_version: 0,
            }],
        }],
    };
    let overlay_txn = Transaction::new_from_version(dataset.manifest.version, overlay_op);

    // Commit the update so the latest dataset reflects the moved (deleted) row.
    let latest_dataset = CommitBuilder::new(Arc::new(dataset.clone()))
        .execute(update_txn.clone())
        .await
        .unwrap();

    let mut rebase = TransactionRebase::try_new(&dataset, overlay_txn.clone(), None)
        .await
        .unwrap();
    // The check defers the row-level decision to finish, flagging fragment 0.
    rebase.check_txn(&update_txn, 1).unwrap();
    assert_eq!(
        rebase
            .initial_fragments
            .iter()
            .map(|(id, (_, needs_check))| (*id, *needs_check))
            .collect::<Vec<_>>(),
        vec![(0, true)],
    );

    let res = rebase.finish(&latest_dataset).await;
    if expect_conflict {
        assert!(
            matches!(res, Err(crate::Error::RetryableCommitConflict { .. })),
            "overlay covering the moved row should conflict, got {res:?}"
        );
    } else {
        assert!(
            res.is_ok(),
            "overlay disjoint from the moved row should succeed, got {res:?}"
        );
    }
}

#[rstest::rstest]
#[test]
#[case::indexed_field_updated(0, vec![0])]
#[case::other_field_updated(1, vec![0, 1])]
fn test_create_index_rebase_prunes_updated_field_coverage(
    #[case] field_modified: u32,
    #[case] expected_fragment_ids: Vec<u32>,
) {
    let index = IndexMetadata {
        uuid: Uuid::new_v4(),
        name: "test".to_string(),
        fields: vec![0],
        covering_fields: vec![],
        dataset_version: 1,
        fragment_bitmap: Some(RoaringBitmap::from_iter([0, 1])),
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let mut rebase = TransactionRebase {
        transaction: Transaction::new(
            1,
            Operation::CreateIndex {
                new_indices: vec![index],
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
    let update = Transaction::new(
        1,
        Operation::Update {
            updated_fragments: vec![Fragment::new(1)],
            removed_fragment_ids: vec![],
            new_fragments: vec![],
            fields_modified: vec![field_modified],
            mem_wal_dir_id: None,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        },
        None,
    );

    rebase.check_txn(&update, 2).unwrap();

    let Operation::CreateIndex { new_indices, .. } = &rebase.transaction.operation else {
        panic!("expected CreateIndex operation");
    };
    assert_eq!(
        new_indices[0].fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter(expected_fragment_ids)
    );
}

const KEYED: i32 = 0;
const CARRIED: i32 = 1;
const UNRELATED: i32 = 2;

/// Released writers repeat each carried column at the end of `fields`;
/// the keys-only shape lists it in `covering_fields` alone.
#[derive(Debug, Clone, Copy)]
enum CoveringShape {
    Suffix,
    KeysOnly,
}

impl CoveringShape {
    fn index(self) -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "covered".to_string(),
            fields: match self {
                Self::Suffix => vec![KEYED, CARRIED],
                Self::KeysOnly => vec![KEYED],
            },
            covering_fields: vec![CARRIED],
            dataset_version: 1,
            fragment_bitmap: Some(RoaringBitmap::from_iter([0])),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum CoveringConflictSite {
    /// The index, on a tagged table, over an `alter_columns` cast.
    IndexOverCast,
    IndexOverUntaggedCast,
    IndexOverProjection,
    IndexOverDataReplacement,
    DataReplacementOverIndex,
    IndexOverFoldedOverlay,
}

impl CoveringConflictSite {
    /// Whether the rebase retries, or withdraws the index's coverage of
    /// fragment 0, when the concurrent change reaches field `changed`.
    fn invalidates(self, index: IndexMetadata, changed: i32) -> bool {
        let schema = lance_core::datatypes::Schema::try_from(&Schema::new(
            ["keyed", "carried", "unrelated"]
                .map(|name| Field::new(name, DataType::Int32, false))
                .to_vec(),
        ))
        .unwrap();
        let rebase_of = |ours: Operation| TransactionRebase {
            transaction: Transaction::new(1, ours, None),
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
        let create_index = Operation::CreateIndex {
            new_indices: vec![index],
            removed_indices: vec![],
        };
        let replacement = Operation::DataReplacement {
            replacements: vec![DataReplacementGroup(
                0,
                DataFile::new_legacy_from_fields("replaced.lance", vec![changed], None),
            )],
        };
        let (mut rebase, theirs) = match self {
            Self::IndexOverCast | Self::IndexOverUntaggedCast | Self::IndexOverProjection => {
                let mut rebase = rebase_of(create_index);
                let history = IndexMetadata {
                    uuid: Uuid::new_v4(),
                    name: FRAG_REUSE_INDEX_NAME.to_string(),
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
                if matches!(self, Self::IndexOverCast) {
                    rebase.current_lineage = Some(TaggedLineage::new(&history, None));
                }
                rebase.read_schema = Some(schema.clone());
                let kept: Vec<i32> = [KEYED, CARRIED, UNRELATED]
                    .into_iter()
                    .filter(|field| *field != changed)
                    .collect();
                // A cast moves the column to a new field id; a projection drops it.
                let schema = schema.project_by_ids(&kept, true);
                let change = if matches!(self, Self::IndexOverProjection) {
                    Operation::Project {
                        schema,
                        preserves_nullability: true,
                    }
                } else {
                    Operation::Merge {
                        fragments: vec![],
                        schema,
                        preserves_nullability: true,
                    }
                };
                (rebase, change)
            }
            Self::IndexOverDataReplacement => (rebase_of(create_index), replacement),
            Self::DataReplacementOverIndex => (rebase_of(replacement), create_index),
            Self::IndexOverFoldedOverlay => {
                let mut rebase = rebase_of(create_index);
                rebase.untagged_reuse = Some(Box::new(UntaggedReuse {
                    groups: HashSet::from([Arc::from([0u64])]),
                    sources: HashMap::from([(7, Arc::from([0u64]))]),
                    manifest: Arc::new(lance_table::format::Manifest::new(
                        schema,
                        Arc::new(vec![]),
                        Default::default(),
                        HashMap::new(),
                    )),
                }));
                let mut overlaid = Fragment::new(0);
                overlaid.overlays = vec![lance_table::format::overlay::DataOverlayFile {
                    data_file: DataFile::new_legacy_from_fields(
                        "overlay.lance",
                        vec![changed],
                        None,
                    ),
                    coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
                    committed_version: 2,
                }];
                let compaction = Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: vec![overlaid],
                        new_fragments: vec![Fragment::new(7)],
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                };
                (rebase, compaction)
            }
        };
        match rebase.check_txn(&Transaction::new(1, theirs, None), 2) {
            Err(Error::RetryableCommitConflict { .. }) => true,
            Err(error) => panic!("{self:?}: {error}"),
            Ok(()) => match &rebase.transaction.operation {
                Operation::CreateIndex { new_indices, .. } => {
                    !new_indices[0].fragment_bitmap.as_ref().unwrap().contains(0)
                }
                _ => false,
            },
        }
    }
}

/// A concurrent change to a new index's keyed or carried column makes
/// the rebase retry or withdraw coverage, whether or not `fields`
/// repeats the carried column; a change to another column does not.
#[rstest::rstest]
#[case::index_over_cast(CoveringConflictSite::IndexOverCast)]
#[case::index_over_untagged_cast(CoveringConflictSite::IndexOverUntaggedCast)]
#[case::index_over_projection(CoveringConflictSite::IndexOverProjection)]
#[case::index_over_data_replacement(CoveringConflictSite::IndexOverDataReplacement)]
#[case::data_replacement_over_index(CoveringConflictSite::DataReplacementOverIndex)]
#[case::index_over_folded_overlay(CoveringConflictSite::IndexOverFoldedOverlay)]
fn test_covering_dependency(#[case] site: CoveringConflictSite) {
    for shape in [CoveringShape::Suffix, CoveringShape::KeysOnly] {
        for (column, changed, invalidates) in [
            ("keyed", KEYED, true),
            ("carried", CARRIED, true),
            ("unrelated", UNRELATED, false),
        ] {
            assert_eq!(
                site.invalidates(shape.index(), changed),
                invalidates,
                "{site:?}, {shape:?}: a change to the {column} column"
            );
        }
    }
}

/// A restore rewinds table, schema and field metadata with the rest of the
/// manifest, so a metadata update must not rebase over it; a plain config
/// upsert carries no metadata and still rebases.
#[rstest::rstest]
#[case::field_metadata(Some(HashMap::from([(0, HashMap::from([("fresh".to_string(), "{}".to_string())]))])), None, None, true)]
#[case::schema_metadata(None, Some(HashMap::from([("owner".to_string(), "refresh".to_string())])), None, true)]
#[case::table_upsert(None, None, Some(UpdateMap { update_entries: vec![("provider.active", Some("2")).into()], replace: false }), true)]
#[case::table_delete(None, None, Some(UpdateMap { update_entries: vec![("provider.active", None::<&str>).into()], replace: false }), true)]
#[case::table_replace(None, None, Some(UpdateMap { update_entries: vec![("provider.active", Some("2")).into()], replace: true }), true)]
#[case::table_clear(None, None, Some(UpdateMap { update_entries: vec![], replace: true }), true)]
#[case::empty_table_update(None, None, Some(UpdateMap { update_entries: vec![], replace: false }), false)]
#[case::config_only(None, None, None, false)]
fn test_update_config_conflicts_with_restore(
    #[case] field_metadata: Option<HashMap<u32, HashMap<String, String>>>,
    #[case] schema_metadata: Option<HashMap<String, String>>,
    #[case] table_metadata: Option<UpdateMap>,
    #[case] conflicts: bool,
) {
    let restore = Transaction::new(0, Operation::Restore { version: 1 }, None);
    let mut update = create_update_config_for_test(
        Some(HashMap::from([("key".to_string(), "value".to_string())])),
        None,
        schema_metadata,
        field_metadata,
    );
    let Operation::UpdateConfig {
        table_metadata_updates,
        ..
    } = &mut update
    else {
        unreachable!()
    };
    *table_metadata_updates = table_metadata;
    // Either commit order: the metadata the restore discarded must not
    // come back, whichever transaction rebases.
    for (ours, theirs) in [
        (update.clone(), restore.operation.clone()),
        (restore.operation, update),
    ] {
        let mut rebase = TransactionRebase {
            transaction: Transaction::new(0, ours, None),
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
        let result = rebase.check_txn(&Transaction::new(0, theirs, None), 1);
        if conflicts {
            let error = result.unwrap_err();
            assert!(
                matches!(error, Error::IncompatibleTransaction { .. }),
                "{error}"
            );
            assert!(error.to_string().contains("Restore"), "{error}");
        } else {
            result.unwrap();
        }
    }
}

#[tokio::test]
async fn test_conflicts_data_replacement() {
    use io::commit::conflict_resolver::tests::{ConflictResult::*, modified_fragment_ids};

    let fragment0 = Fragment::new(0);
    let fragment1 = Fragment::new(1);

    let data_file_frag0_fields01 = DataFile::new_legacy_from_fields("path0_01", vec![0, 1], None);
    let data_file_frag0_fields23 = DataFile::new_legacy_from_fields("path0_23", vec![2, 3], None);
    let data_file_frag1_fields01 = DataFile::new_legacy_from_fields("path1_01", vec![0, 1], None);

    let cases = vec![
        (
            "Different fragments",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(1, data_file_frag1_fields01)],
            },
            Compatible,
        ),
        (
            "Same fragment, different fields",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields23)],
            },
            Compatible,
        ),
        (
            "Same fragment, same fields",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Retryable,
        ),
        (
            "Same fragment, overlapping fields",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(
                    0,
                    DataFile::new_legacy_from_fields("path0_12", vec![1, 2], None),
                )],
            },
            Retryable,
        ),
        (
            "DataReplacement vs Rewrite on same fragment",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: vec![fragment0.clone()],
                    new_fragments: vec![fragment1.clone()],
                }],
                rewritten_indices: vec![],
                frag_reuse_index: None,
            },
            Retryable,
        ),
        (
            "DataReplacement vs Rewrite on different fragment",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: vec![fragment1],
                    new_fragments: vec![fragment0],
                }],
                rewritten_indices: vec![],
                frag_reuse_index: None,
            },
            Compatible,
        ),
        // A concurrent Update/Delete only invalidates our positional file when it
        // removes our target fragment outright, or (a horizontal update) rewrites
        // one of our fields. A deletion-vector-only change stays aligned.
        (
            "DataReplacement vs Update (RewriteColumns) on a different field",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::Update {
                updated_fragments: vec![Fragment::new(0)],
                removed_fragment_ids: vec![],
                new_fragments: vec![],
                fields_modified: vec![2],
                mem_wal_dir_id: None,
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(RewriteColumns),
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
            Compatible,
        ),
        (
            // RewriteColumns new_fragments are unrelated inserts, not moved rows.
            "DataReplacement vs Update (RewriteColumns) with inserts on a different field",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::Update {
                updated_fragments: vec![Fragment::new(0)],
                removed_fragment_ids: vec![],
                new_fragments: vec![Fragment::new(5)],
                fields_modified: vec![2],
                mem_wal_dir_id: None,
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(RewriteColumns),
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
            Compatible,
        ),
        (
            "DataReplacement vs Update (RewriteColumns) that rewrote one of our fields",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::Update {
                updated_fragments: vec![Fragment::new(0)],
                removed_fragment_ids: vec![],
                new_fragments: vec![],
                fields_modified: vec![1],
                mem_wal_dir_id: None,
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(RewriteColumns),
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
            Retryable,
        ),
        (
            "DataReplacement vs Update (RewriteRows) that moved our rows",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::Update {
                updated_fragments: vec![Fragment::new(0)],
                removed_fragment_ids: vec![],
                new_fragments: vec![Fragment::new(5)],
                fields_modified: vec![],
                mem_wal_dir_id: None,
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(RewriteRows),
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
            Retryable,
        ),
        (
            "DataReplacement vs Update that removed our fragment",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::Update {
                updated_fragments: vec![],
                removed_fragment_ids: vec![0],
                new_fragments: vec![],
                fields_modified: vec![],
                mem_wal_dir_id: None,
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
            NotCompatible,
        ),
        (
            "DataReplacement vs Update (RewriteRows) that moved a different fragment's rows",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::Update {
                updated_fragments: vec![Fragment::new(1)],
                removed_fragment_ids: vec![],
                new_fragments: vec![Fragment::new(5)],
                fields_modified: vec![],
                mem_wal_dir_id: None,
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(RewriteRows),
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
            Compatible,
        ),
        (
            "DataReplacement vs Delete (deletion-vector only) on same fragment",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::Delete {
                deleted_fragment_ids: vec![],
                updated_fragments: vec![Fragment::new(0)],
                predicate: "a > 0".to_string(),
            },
            Compatible,
        ),
        (
            "DataReplacement vs Delete that removes the fragment",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01.clone())],
            },
            Operation::Delete {
                deleted_fragment_ids: vec![0],
                updated_fragments: vec![],
                predicate: "a > 0".to_string(),
            },
            NotCompatible,
        ),
        // Merge rewrites the whole fragment list -> always conflicts.
        (
            "DataReplacement vs Merge",
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file_frag0_fields01)],
            },
            Operation::Merge {
                fragments: vec![Fragment::new(0)],
                schema: lance_core::datatypes::Schema::default(),
                preserves_nullability: true,
            },
            Retryable,
        ),
        (
            // Unlike every other case here, op1 is CreateIndex and op2 is
            // DataReplacement. This is deliberate, not an inconsistency:
            // `check_txn` dispatches on op1's operation type, so only
            // op1 == CreateIndex reaches `check_create_index_txn`'s
            // `Operation::DataReplacement` arm, which is the arm this case
            // targets. Swapping the order to match the other cases would
            // instead exercise the mirrored `check_data_replacement_txn`'s
            // `Operation::CreateIndex` arm, leaving the intended arm with
            // zero coverage.
            "CreateIndex covering a field vs DataReplacement of that field",
            Operation::CreateIndex {
                new_indices: vec![IndexMetadata {
                    uuid: Uuid::new_v4(),
                    name: "covering_idx".to_string(),
                    fields: vec![0, 3],
                    covering_fields: vec![3],
                    dataset_version: 1,
                    fragment_bitmap: Some(RoaringBitmap::from_iter([0u32])),
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
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(
                    0,
                    DataFile::new_legacy_from_fields("path0_3", vec![3], None),
                )],
            },
            Retryable,
        ),
    ];

    for (description, op1, op2, expected) in cases {
        let txn1 = Transaction::new(0, op1.clone(), None);
        let txn2 = Transaction::new(0, op2.clone(), None);

        let mut rebase = TransactionRebase {
            transaction: txn1,
            initial_fragments: HashMap::new(),
            modified_fragment_ids: modified_fragment_ids(&op1).collect::<HashSet<_>>(),
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

        let result = rebase.check_txn(&txn2, 1);
        match expected {
            Compatible => {
                assert!(
                    result.is_ok(),
                    "{}: expected Compatible but got {:?}",
                    description,
                    result
                );
            }
            NotCompatible => {
                // Removal returns a non-retryable IncompatibleTransaction so the
                // caller can drop the fragment instead of retrying.
                assert!(
                    matches!(result, Err(Error::IncompatibleTransaction { .. })),
                    "{}: expected NotCompatible but got {:?}",
                    description,
                    result
                )
            }
            Retryable => {
                assert!(
                    matches!(result, Err(Error::RetryableCommitConflict { .. })),
                    "{}: expected Retryable but got {:?}",
                    description,
                    result
                );
            }
        }
    }
}
