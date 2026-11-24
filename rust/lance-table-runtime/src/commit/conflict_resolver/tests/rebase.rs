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

    dataset.state.object_store.as_ref().io_stats_incremental(); // reset
    let mut rebase = TransactionRebase::try_new(&dataset, txn.clone(), Some(&affected_rows))
        .await
        .unwrap();

    let io_stats = dataset.state.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 0);
    assert_io_eq!(io_stats, write_iops, 0);

    let res = rebase.check_txn(&other_txn, 1);
    if other.ends_with("full") || ours.ends_with("full") {
        // If the other transaction fully deleted a fragment, we can error early.
        assert!(matches!(res, Err(Error::RetryableCommitConflict { .. })));
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

    let io_stats = dataset.state.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 0);
    assert_io_eq!(io_stats, write_iops, 0);

    let res = rebase.finish(&latest_dataset).await;
    assert!(matches!(res, Err(Error::RetryableCommitConflict { .. })));

    let io_stats = dataset.state.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 0, "deletion file should be cached");
    assert_io_eq!(io_stats, write_iops, 0, "failed before writing");
}
