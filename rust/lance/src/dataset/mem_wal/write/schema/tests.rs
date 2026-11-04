// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Writer schema derivation and batch shaping internals.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Field as ArrowField};
use lance_core::datatypes::LANCE_FIELD_ID_KEY;
use uuid::Uuid;

use super::*;
use crate::dataset::mem_wal::index::MemIndexSpec;
use crate::dataset::mem_wal::write::test_support::*;
use crate::dataset::mem_wal::write::{ShardWriter, ShardWriterConfig};
use crate::dataset::mem_wal::{TOMBSTONE, schema_with_tombstone};

#[test]
fn test_conform_injects_tombstone_false() {
    let base = create_test_schema();
    let storage = schema_with_tombstone(&base);
    let pk = ["id".to_string()];
    let out = conform_to_storage_schema(create_test_batch(&base, 0, 3), &storage, &pk).unwrap();
    assert_eq!(out.schema(), storage);
    let ts = out
        .column_by_name(TOMBSTONE)
        .unwrap()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert!(
        (0..3).all(|i| !ts.value(i)),
        "put injects _tombstone = false"
    );
    // Idempotent: a batch already carrying the column passes through.
    let again = conform_to_storage_schema(out.clone(), &storage, &pk).unwrap();
    assert_eq!(again.schema(), out.schema());
}

/// A WAL entry written before a column was added still replays: the column
/// it never held becomes null rather than a width mismatch.
#[test]
fn test_conform_fills_a_column_added_since_the_entry() {
    let pk = ["id".to_string()];
    let entry = create_test_batch(&create_test_schema(), 0, 2);

    let widened = ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("added_later", DataType::Int64, true),
    ]);
    let storage = schema_with_tombstone(&widened);

    let out = conform_to_storage_schema(entry, &storage, &pk).unwrap();
    assert_eq!(out.schema(), storage);
    assert_eq!(out.num_rows(), 2);
    let added = out.column_by_name("added_later").unwrap();
    assert_eq!(added.null_count(), 2, "the new column replays as all-null");
    let ids = out
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.values(), &[0, 1], "the entry's own columns survive");
}

/// A WAL entry written before a column was dropped still replays: the
/// column the schema no longer declares is left behind, and the columns
/// that remain keep their own values rather than their neighbour's.
#[test]
fn test_conform_drops_a_column_removed_since_the_entry() {
    let pk = ["id".to_string()];
    let wide = ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("dropped_later", DataType::Utf8, true),
        Field::new("name", DataType::Utf8, true),
    ]);
    let entry = RecordBatch::try_new(
        Arc::new(wide),
        vec![
            Arc::new(Int32Array::from(vec![7, 8])),
            Arc::new(StringArray::from(vec!["gone", "gone"])),
            Arc::new(StringArray::from(vec!["kept-7", "kept-8"])),
        ],
    )
    .unwrap();

    let storage = schema_with_tombstone(&create_test_schema());
    let out = conform_to_storage_schema(entry, &storage, &pk).unwrap();
    assert_eq!(out.schema(), storage);
    let names = out
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        (names.value(0), names.value(1)),
        ("kept-7", "kept-8"),
        "positional re-labelling would have stored `dropped_later` as `name`"
    );
}

/// A primary key is the one column a null cannot stand in for.
#[test]
fn test_conform_refuses_an_entry_missing_a_primary_key() {
    let storage = schema_with_tombstone(&create_test_schema());
    let keyless = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![Field::new(
            "name",
            DataType::Utf8,
            true,
        )])),
        vec![Arc::new(StringArray::from(vec!["a"]))],
    )
    .unwrap();

    let error = conform_to_storage_schema(keyless, &storage, &["id".to_string()]).unwrap_err();
    assert!(
        matches!(error, Error::InvalidInput { .. }),
        "expected InvalidInput, got {error:?}"
    );
    assert!(
        error.to_string().contains("id"),
        "the error should name the missing key: {error}"
    );
}

/// A column whose type has moved is refused rather than cast: a table with a
/// MemWAL does not accept a retype, so this is a disagreement to surface.
#[test]
fn test_conform_refuses_a_column_whose_type_moved() {
    let widened = ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("count", DataType::Int64, true),
    ]);
    let storage = schema_with_tombstone(&widened);

    let narrow = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("count", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(Int32Array::from(vec![7])),
        ],
    )
    .unwrap();

    let err = conform_to_storage_schema(narrow, &storage, &["id".to_string()])
        .expect_err("a column's type cannot change");
    assert!(
        matches!(err, Error::InvalidInput { .. }),
        "expected InvalidInput, got {err:?}"
    );
    assert!(
        err.to_string().contains("count"),
        "the refusal should name the column, got: {err}"
    );
}

/// A struct column renamed at the top level is matched by the id on the
/// column itself, and taken whole.
///
/// Only the parent moves here. Reconciliation is recursive — a child
/// carries its own id and is resolved on its own — which the nested cases
/// in `reconcile` cover.
#[test]
fn test_conform_matches_a_struct_column_by_its_own_id() {
    fn with_id(field: ArrowField, id: i32) -> ArrowField {
        let mut metadata = field.metadata().clone();
        metadata.insert(LANCE_FIELD_ID_KEY.to_string(), id.to_string());
        field.with_metadata(metadata)
    }

    let child = Arc::new(ArrowField::new("inner", DataType::Int32, true));
    let struct_type = DataType::Struct(vec![child.clone()].into());
    let values: ArrayRef = Arc::new(arrow_array::StructArray::from(vec![(
        child,
        Arc::new(Int32Array::from(vec![42])) as ArrayRef,
    )]));

    let entry = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            with_id(Field::new("id", DataType::Int32, false), 0),
            with_id(Field::new("before", struct_type.clone(), true), 1),
        ])),
        vec![Arc::new(Int32Array::from(vec![1])), values],
    )
    .unwrap();

    // The column was renamed; its type, and so its children, are unchanged.
    let storage = schema_with_tombstone(&ArrowSchema::new(vec![
        with_id(Field::new("id", DataType::Int32, false), 0),
        with_id(Field::new("after", struct_type, true), 1),
    ]));

    let out = conform_to_storage_schema(entry, &storage, &["id".to_string()]).unwrap();
    let after = out.column_by_name("after").expect("renamed struct column");
    assert_eq!(
        after.null_count(),
        0,
        "the struct column should carry values"
    );
    assert_eq!(out.schema(), storage);
}

/// A field id outlives a rename, so matching on it keeps the column's rows
/// where matching on the name would null them.
#[test]
fn test_conform_follows_a_field_id_through_a_rename() {
    fn with_id(field: ArrowField, id: i32) -> ArrowField {
        let mut metadata = field.metadata().clone();
        metadata.insert(LANCE_FIELD_ID_KEY.to_string(), id.to_string());
        field.with_metadata(metadata)
    }

    // The entry was written while field 1 was called `before`.
    let entry = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            with_id(Field::new("id", DataType::Int32, false), 0),
            with_id(Field::new("before", DataType::Utf8, true), 1),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["kept"])),
        ],
    )
    .unwrap();

    // The schema now calls field 1 `after`.
    let storage = schema_with_tombstone(&ArrowSchema::new(vec![
        with_id(Field::new("id", DataType::Int32, false), 0),
        with_id(Field::new("after", DataType::Utf8, true), 1),
    ]));

    let out = conform_to_storage_schema(entry, &storage, &["id".to_string()]).unwrap();
    let after = out
        .column_by_name("after")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        after.value(0),
        "kept",
        "the id should carry the value to the new name; a name match would null it"
    );
}

#[test]
fn test_build_tombstone_batch_shape() {
    let storage = schema_with_tombstone(&create_test_schema());
    let tomb =
        build_tombstone_batch(&id_only_keys(&[5, 7]), &storage, &["id".to_string()]).unwrap();
    assert_eq!(tomb.schema(), storage);
    assert_eq!(tomb.num_rows(), 2);
    let ids = tomb
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.values(), &[5, 7]);
    let name = tomb.column_by_name("name").unwrap();
    assert!(
        name.is_null(0) && name.is_null(1),
        "non-PK columns are null in a tombstone"
    );
    let ts = tomb
        .column_by_name(TOMBSTONE)
        .unwrap()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert!(ts.value(0) && ts.value(1), "_tombstone = true");
}

#[test]
fn test_build_tombstone_batch_missing_pk_errors() {
    let storage = schema_with_tombstone(&create_test_schema());
    let keys = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![Field::new(
            "other",
            DataType::Int32,
            false,
        )])),
        vec![Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();
    assert!(build_tombstone_batch(&keys, &storage, &["id".to_string()]).is_err());
}

#[test]
fn test_build_tombstone_batch_nulls_non_nullable_base_column() {
    // The point of the storage schema: a tombstone nulls `v` even though the
    // base table declares it non-nullable.
    let pk = ["id".to_string()];
    let base = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]));
    let storage = relax_non_pk_nullability(&schema_with_tombstone(&base), &pk);

    let batch = build_tombstone_batch(&id_only_keys(&[1]), &storage, &pk).unwrap();

    assert!(batch["v"].is_null(0), "the tombstone must null `v`");
    assert!(!batch["id"].is_null(0), "the primary key survives");
    assert!(
        batch[TOMBSTONE]
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .value(0)
    );
}

#[test]
fn test_build_tombstone_batch_rejects_null_primary_key() {
    // Primary keys are never relaxed, so the storage schema still rejects a
    // null key — the delete path needs no separate check for it.
    let pk = ["id".to_string()];
    let base = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]));
    let storage = relax_non_pk_nullability(&schema_with_tombstone(&base), &pk);
    let keys = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            true,
        )])),
        vec![Arc::new(Int32Array::from(vec![None::<i32>]))],
    )
    .unwrap();

    let error = build_tombstone_batch(&keys, &storage, &pk).unwrap_err();
    assert!(
        matches!(error, Error::InvalidInput { .. }),
        "expected InvalidInput, got {error:?}"
    );
    assert!(
        error.to_string().contains("non-nullable"),
        "error should name the nullability violation: {error}"
    );
}

#[tokio::test]
async fn test_shard_writer_delete_round_trip() {
    use crate::dataset::mem_wal::scanner::LsmScanner;
    use futures::TryStreamExt;

    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let schema = create_pk_test_schema();
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        durable_write: true,
        ..Default::default()
    };
    let shard_id = config.shard_id;
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri.clone(),
        config,
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    // ids 0..5, then delete id=2.
    writer
        .put(vec![create_test_batch(&schema, 0, 5)])
        .await
        .unwrap();
    writer.delete(vec![id_only_keys(&[2])]).await.unwrap();

    let refs = writer.in_memory_memtable_refs().await.unwrap();
    let scanner =
        LsmScanner::without_base_table(schema.clone(), base_uri, vec![], vec!["id".to_string()])
            .with_in_memory_memtables(shard_id, refs);
    let batches: Vec<RecordBatch> = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let mut ids: Vec<i32> = Vec::new();
    for b in &batches {
        let arr = b
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        ids.extend((0..arr.len()).map(|i| arr.value(i)));
    }
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![0, 1, 3, 4],
        "id=2 deleted; tombstone not surfaced"
    );

    writer.close().await.unwrap();
}

/// A rebuild that changes only the distance metric is a different index.
///
/// Name, kind, column and field id all survive it, so comparing those alone
/// would leave the writer on the old metric while queries use the new one.
#[test]
fn test_a_changed_metric_is_a_changed_index_set() {
    use lance_linalg::distance::DistanceType;
    let old = MemIndexSpec::hnsw("vector_idx", 1, "vector", DistanceType::L2);
    let new = MemIndexSpec::hnsw("vector_idx", 1, "vector", DistanceType::Cosine);
    assert!(!same_index_set(&[old], &[new]));
}
