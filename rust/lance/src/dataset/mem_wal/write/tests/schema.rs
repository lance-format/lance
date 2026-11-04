// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Public put and delete contract tests that cross the writer boundary.

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance_core::Error;
use uuid::Uuid;

use super::super::test_support::*;
use super::super::{ShardWriter, ShardWriterConfig};
use crate::dataset::mem_wal::index::MemIndexSpec;

/// Two same-typed columns handed over swapped pass every positional check
/// downstream, so the logical-schema gate has to catch them by name.
#[tokio::test]
async fn test_put_rejects_swapped_same_typed_columns() {
    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("name", DataType::Utf8, true),
        Field::new("email", DataType::Utf8, true),
    ]));
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            ..Default::default()
        },
        schema,
        vec![],
    )
    .await
    .unwrap();

    let swapped = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new("email", DataType::Utf8, true),
            Field::new("name", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["a@example.com"])),
            Arc::new(StringArray::from(vec!["a"])),
        ],
    )
    .unwrap();

    let error = writer.put(vec![swapped]).await.unwrap_err();
    assert!(
        matches!(error, Error::InvalidInput { .. }),
        "expected InvalidInput, got {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains("column 0") && message.contains("email") && message.contains("name"),
        "error should name the position and both columns: {message}"
    );

    writer.close().await.unwrap();
}

/// Delete works against a base table with non-nullable non-PK columns, and
/// survivors come back through the narrowing egress relabel intact.
#[tokio::test]
async fn test_delete_against_non_nullable_base_column_round_trip() {
    use crate::dataset::mem_wal::scanner::LsmScanner;
    use futures::TryStreamExt;

    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let schema = create_strict_pk_test_schema();
    assert!(
        !schema.field_with_name("name").unwrap().is_nullable(),
        "the point of this test is a non-nullable non-PK column"
    );
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

    let mut rows: Vec<(i32, String)> = Vec::new();
    for b in &batches {
        assert!(
            !b.schema().field_with_name("name").unwrap().is_nullable(),
            "egress must narrow back to the logical schema"
        );
        let ids = b["id"].as_any().downcast_ref::<Int32Array>().unwrap();
        let names = b["name"].as_any().downcast_ref::<StringArray>().unwrap();
        rows.extend((0..ids.len()).map(|i| (ids.value(i), names.value(i).to_string())));
    }
    rows.sort_unstable();

    assert_eq!(
        rows,
        vec![
            (0, "name_0".to_string()),
            (1, "name_1".to_string()),
            (3, "name_3".to_string()),
            (4, "name_4".to_string()),
        ],
        "id=2 deleted; every survivor keeps its non-nullable value"
    );

    writer.close().await.unwrap();
}

/// The storage schema no longer rejects a caller's null, so `put` is the
/// only thing standing between a null and a non-nullable base column.
#[tokio::test]
async fn test_put_rejects_null_in_non_nullable_base_column() {
    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let schema = create_strict_pk_test_schema();
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            ..Default::default()
        },
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    let error = writer.put(vec![null_name_batch()]).await.unwrap_err();
    assert!(
        matches!(error, Error::InvalidInput { .. }),
        "expected InvalidInput, got {error:?}"
    );
    assert!(
        error.to_string().contains("base table schema"),
        "error should point at the schema contract: {error}"
    );

    writer.close().await.unwrap();
}

/// WAL-only mode validates too — it has no memtable, so before this gate
/// nothing checked its input at all.
#[tokio::test]
async fn test_wal_only_put_rejects_null_in_non_nullable_base_column() {
    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let schema = create_strict_pk_test_schema();
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        wal_only_config(Uuid::new_v4()),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    let error = writer.put(vec![null_name_batch()]).await.unwrap_err();
    assert!(
        matches!(error, Error::InvalidInput { .. }),
        "expected InvalidInput, got {error:?}"
    );

    writer.close().await.unwrap();
}

/// `delete_no_wait` lands the tombstone in the in-memory tier (visible at
/// the batch-store level the instant it returns) and hands back the
/// durability watcher *without* awaiting it. Index-driven LSM read
/// visibility — folding the tombstone out — follows once the watcher
/// resolves the flush that advances the visibility watermark and updates
/// the PK index. The delete analog of
/// `test_put_no_wait_durable_visible_then_durable`.
#[tokio::test]
async fn test_shard_writer_delete_no_wait_durable_visible_after_watcher() {
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

    writer
        .put(vec![create_test_batch(&schema, 0, 5)])
        .await
        .unwrap();

    // Tombstone id=2 without blocking on durability.
    let (_result, watcher) = writer
        .delete_no_wait(vec![id_only_keys(&[2])])
        .await
        .unwrap();

    // The tombstone is in the in-memory tier immediately (5 rows + 1
    // tombstone), even though the index-driven read can't fold it until the
    // flush behind the watcher lands. `durable_write` is on, so a watcher is
    // returned to await.
    assert_eq!(writer.memtable_stats().await.unwrap().row_count, 6);
    let mut watcher = watcher.expect("durable_write returns a watcher");

    // Awaiting the watcher waits for the flush, which advances the
    // visibility watermark and updates the PK index — only then does the LSM
    // read fold the delete.
    watcher.wait().await.unwrap();

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
        "delete folded by the LSM read once the watcher resolves"
    );

    writer.close().await.unwrap();
}

/// With `durable_write` off, `delete_no_wait` returns no watcher (nothing to
/// await), but the tombstone still lands in the in-memory tier. The delete
/// analog of `test_put_no_wait_non_durable_returns_no_watcher`.
#[tokio::test]
async fn test_non_durable_delete_is_read_your_writes() {
    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let schema = create_pk_test_schema();
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        durable_write: false,
        ..Default::default()
    };
    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    writer
        .put(vec![create_test_batch(&schema, 0, 5)])
        .await
        .unwrap();

    let (_result, watcher) = writer
        .delete_no_wait(vec![id_only_keys(&[2])])
        .await
        .unwrap();
    // As with a put: a non-durable delete awaits its index apply, not an S3
    // round-trip. It is read-your-writes, just not durable.
    let mut watcher = watcher.expect("a non-durable delete awaits its index apply");
    watcher.wait().await.unwrap();

    // Tombstone landed in the in-memory tier (5 rows + 1 tombstone).
    assert_eq!(writer.memtable_stats().await.unwrap().row_count, 6);

    writer.close().await.unwrap();
}

/// Delete a key, then flush: the tombstone and the live row land in the
/// *same* SSTable, so flush-time dedup must keep the tombstone
/// (newest) and the read must fold it away. Regression for the wallop
/// phantom (deleted row resurfacing in a filtered read after flush).
#[tokio::test]
async fn test_shard_writer_delete_then_flush_round_trip() {
    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let schema = create_pk_test_schema();
    let shard_id = Uuid::new_v4();
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri.clone(),
        flush_test_config(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    // ids 0..5, delete id=2, THEN flush (insert + tombstone in one gen).
    writer
        .put(vec![create_test_batch(&schema, 0, 5)])
        .await
        .unwrap();
    writer.delete(vec![id_only_keys(&[2])]).await.unwrap();
    writer.force_freeze_active().await.unwrap();
    writer.wait_for_flush_drain().await.unwrap();

    assert_eq!(
        read_sstable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, None).await,
        vec![0, 1, 3, 4],
        "id=2 deleted before flush; tombstone must not surface in an SSTable scan"
    );
    // The filtered read path (folds NOT _tombstone into the predicate) must
    // also drop it — this is the exact wallop failure shape (`id < 3`).
    assert_eq!(
        read_sstable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, Some("id < 3"))
            .await,
        vec![0, 1],
        "filtered read after flush must not resurface deleted id=2"
    );

    writer.close().await.unwrap();
}

/// A user index may carry any name, including one shaped like an internal
/// key index. It keeps its own column through writes, overwrites, deletes,
/// a flush and a reopen, and the key index keeps deduplicating beside it.
#[tokio::test]
async fn test_user_index_named_like_a_key_index_keeps_its_column() {
    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let schema = create_pk_test_schema();
    let shard_id = Uuid::new_v4();
    let index_specs = vec![MemIndexSpec::btree("__pk__id", 1, "name")];
    let open = || {
        ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            flush_test_config(shard_id),
            schema.clone(),
            index_specs.clone(),
        )
    };
    let renamed = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["renamed"])),
        ],
    )
    .unwrap();
    let cases = [
        ("name = 'renamed'", vec![1]),
        ("name = 'name_1'", vec![]),
        ("name = 'name_2'", vec![]),
        ("name = 'name_3'", vec![3]),
        ("id = 1", vec![1]),
        ("id = 2", vec![]),
    ];

    let writer = open().await.unwrap();
    writer
        .put(vec![create_test_batch(&schema, 0, 5)])
        .await
        .unwrap();
    writer.put(vec![renamed]).await.unwrap();
    writer.delete(vec![id_only_keys(&[2])]).await.unwrap();
    for (filter, expected) in &cases {
        let ids =
            read_memtable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, Some(filter))
                .await;
        assert_eq!(&ids, expected, "fresh read of {filter}");
    }

    writer.force_freeze_active().await.unwrap();
    writer.wait_for_flush_drain().await.unwrap();
    writer.close().await.unwrap();
    let writer = open().await.unwrap();
    for (filter, expected) in &cases {
        let ids =
            read_sstable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, Some(filter))
                .await;
        assert_eq!(&ids, expected, "flushed read of {filter}");
    }
    writer.close().await.unwrap();
}

/// Insert+flush, then delete+flush: the tombstone lands in a *newer*
/// generation than the live row, so the cross-generation block-list must
/// mask the older row by PK. This is the wallop scenario (seed flushed,
/// then delete, then flush).
#[tokio::test]
async fn test_shard_writer_delete_across_sstables() {
    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let schema = create_pk_test_schema();
    let shard_id = Uuid::new_v4();
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri.clone(),
        flush_test_config(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    // gen 1: ids 0..5.
    writer
        .put(vec![create_test_batch(&schema, 0, 5)])
        .await
        .unwrap();
    writer.force_freeze_active().await.unwrap();
    writer.wait_for_flush_drain().await.unwrap();

    // gen 2: tombstone for id=0 (a later generation than the live row).
    writer.delete(vec![id_only_keys(&[0])]).await.unwrap();
    writer.force_freeze_active().await.unwrap();
    writer.wait_for_flush_drain().await.unwrap();

    assert_eq!(
        read_sstable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, None).await,
        vec![1, 2, 3, 4],
        "id=0 tombstoned in a newer gen must mask the older gen's live row"
    );
    assert_eq!(
        read_sstable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, Some("id < 1"))
            .await,
        Vec::<i32>::new(),
        "filtered read 'id < 1' must not resurface cross-gen deleted id=0"
    );

    writer.close().await.unwrap();
}

/// Same as the cross-generation case, but the SSTables carry a
/// BTree index on `id` (as every wallop table does). A filtered read
/// `id < 1` resolves through the scalar index; the `NOT _tombstone` residual
/// must still be applied or the deleted row leaks. This is the exact wallop
/// failure (BTree id + `FilteredRead 'id < 1'` resurfacing deleted id=0).
#[tokio::test]
async fn test_shard_writer_delete_across_sstables_indexed() {
    let (store, base_path, base_uri, _temp) = create_local_store().await;
    let schema = create_pk_test_schema();
    let shard_id = Uuid::new_v4();
    let index_specs = vec![MemIndexSpec::btree("id_idx", 0, "id")];
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri.clone(),
        flush_test_config(shard_id),
        schema.clone(),
        index_specs,
    )
    .await
    .unwrap();

    // gen 1: ids 0..5 (indexed). gen 2: tombstone for id=0.
    writer
        .put(vec![create_test_batch(&schema, 0, 5)])
        .await
        .unwrap();
    writer.force_freeze_active().await.unwrap();
    writer.wait_for_flush_drain().await.unwrap();
    writer.delete(vec![id_only_keys(&[0])]).await.unwrap();
    writer.force_freeze_active().await.unwrap();
    writer.wait_for_flush_drain().await.unwrap();

    assert_eq!(
        read_sstable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, None).await,
        vec![1, 2, 3, 4],
        "indexed cross-gen: full scan must mask deleted id=0"
    );
    assert_eq!(
        read_sstable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, Some("id < 1"))
            .await,
        Vec::<i32>::new(),
        "indexed filtered read 'id < 1' must not resurface deleted id=0 (wallop repro)"
    );

    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_delete_validation_errors() {
    let (store, base_path, base_uri, _temp) = create_local_store().await;

    // No primary key on the schema → delete is rejected.
    let no_pk = create_test_schema();
    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            durable_write: false,
            ..Default::default()
        },
        no_pk,
        vec![],
    )
    .await
    .unwrap();
    assert!(
        writer.delete(vec![id_only_keys(&[1])]).await.is_err(),
        "delete without a primary key must error"
    );
    // Empty key list is rejected.
    assert!(writer.delete(vec![]).await.is_err());
    writer.close().await.unwrap();
}

/// The compaction Phase-1 hard-delete primitive: a key-only `merge_insert`
/// with `WhenMatched::Delete` + `use_index` over a COMPOSITE join key
/// `(id, shard_key)` where only `id` carries a scalar index. The partial
/// index makes `indexed_join_keys` non-empty, so the scalar-index merge path
/// engages — and it must still delete exactly the `(0, 0)` row. If it
/// no-ops (or mismatches on the partial key), the compactor trims the WAL
/// tombstone while the base row survives → the wallop resurface phantom.
#[tokio::test]
async fn test_indexed_composite_key_delete_removes_row() {
    use crate::Dataset;
    use crate::dataset::{WhenMatched, WhenNotMatched, WhenNotMatchedBySource};
    use crate::index::DatasetIndexExt;
    use arrow_array::{Int64Array, RecordBatchIterator};
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("shard_key", DataType::Int64, false),
        Field::new("value", DataType::Int64, true),
    ]));
    // ids 0..5, shard_key = id % 8 (the wallop composite PK), value = id*100.
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(0..5)),
            Arc::new(Int64Array::from_iter_values((0..5).map(|i| i % 8))),
            Arc::new(Int64Array::from_iter_values((0..5).map(|i| i * 100))),
        ],
    )
    .unwrap();

    let uri = format!("shared-memory://phase1-delete-{}/", Uuid::new_v4().simple());
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema.clone()),
        &uri,
        None,
    )
    .await
    .unwrap();
    // Scalar index on `id` only — `shard_key` is unindexed, mirroring the
    // WAL base table (BTree on `id`, composite PK `(id, shard_key)`).
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_btree".to_string()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // Phase-1 call: key-only source `(id=0, shard_key=0)`, delete-when-matched.
    let keys = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("shard_key", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![0_i64])),
            Arc::new(Int64Array::from(vec![0_i64])),
        ],
    )
    .unwrap();
    let mut builder = crate::dataset::MergeInsertBuilder::try_new(
        Arc::new(dataset),
        vec!["id".to_string(), "shard_key".to_string()],
    )
    .unwrap();
    builder
        .when_matched(WhenMatched::Delete)
        .when_not_matched(WhenNotMatched::DoNothing)
        .when_not_matched_by_source(WhenNotMatchedBySource::Keep)
        .use_index(true);
    let job = builder.try_build().unwrap();
    let keys_schema = keys.schema();
    let (dataset, _stats) = job
        .execute_reader(RecordBatchIterator::new([Ok(keys)], keys_schema))
        .await
        .unwrap();

    // id=0 must be gone — both from a full scan and from an indexed filter.
    let all = dataset.scan().try_into_batch().await.unwrap();
    let mut ids: Vec<i64> = all
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![1, 2, 3, 4],
        "Phase-1 must hard-delete (0, 0) from base"
    );

    let filtered = dataset
        .scan()
        .filter("id < 1")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(
        filtered.num_rows(),
        0,
        "indexed filter must not resurface the deleted composite key"
    );
}
