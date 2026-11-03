// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Blob v2 payload preparation and replay through writer state.

use super::super::*;
use crate::Dataset;
use crate::blob::blob_v2_layout;
use crate::dataset::mem_wal::test_util::failing_memory_store;
use crate::dataset::mem_wal::wal::WalTailer;
use crate::dataset::mem_wal::write::test_support::*;
use arrow_array::{Array, Float32Array, StructArray, UInt8Array, UInt32Array};
use lance_core::datatypes::{BlobKind, BlobV2Layout};
use object_store::ObjectStoreExt;
use rstest::rstest;

#[tokio::test]
async fn test_durable_blob_puts_share_pack_until_wal_flush() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let shard_id = Uuid::new_v4();
    let first = create_blob_v2_batch(0, &[BlobTestValue::Bytes(b"first".to_vec())]);
    let second = create_blob_v2_batch(1, &[BlobTestValue::Bytes(b"later".to_vec())]);
    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri,
        ShardWriterConfig {
            shard_id,
            durable_write: true,
            max_wal_buffer_size: usize::MAX,
            max_wal_flush_interval: Some(Duration::from_secs(60)),
            ..Default::default()
        },
        first.schema(),
        vec![],
    )
    .await
    .unwrap();

    let (_, first_watcher) = writer.put_no_wait(vec![first]).await.unwrap();
    let (_, second_watcher) = writer.put_no_wait(vec![second]).await.unwrap();
    let fence = writer.force_freeze_active().await.unwrap();

    let mut first_watcher = first_watcher.unwrap();
    let mut second_watcher = second_watcher.unwrap();
    first_watcher.wait().await.unwrap();
    second_watcher.wait().await.unwrap();
    fence.wait().await.unwrap();

    let tailer = WalTailer::new(store.clone(), base_path.clone(), "", shard_id);
    let entry = tailer.read_entry(1).await.unwrap().unwrap();
    assert_eq!(entry.batches.len(), 2);
    assert!(tailer.read_entry(2).await.unwrap().is_none());

    let target = entry.target.unwrap();
    let data_dir = target
        .generation_path(&base_path, "", &shard_id)
        .join(DATA_DIR);
    let sidecars = blob_sidecars(store.as_ref(), &data_dir).await;
    assert_eq!(sidecars.len(), 1, "both puts must share one packed object");
    let payload = store
        .inner
        .get(&sidecars[0])
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(payload.as_ref(), b"firstlater");

    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_failed_shared_blob_pack_cannot_enter_wal() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let shard_id = Uuid::new_v4();
    let batch = create_blob_v2_batch(0, &[BlobTestValue::Bytes(b"first".to_vec())]);
    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri,
        ShardWriterConfig {
            shard_id,
            durable_write: true,
            max_wal_buffer_size: usize::MAX,
            max_wal_flush_interval: Some(Duration::from_secs(60)),
            ..Default::default()
        },
        batch.schema(),
        vec![],
    )
    .await
    .unwrap();

    let (_, watcher) = writer.put_no_wait(vec![batch]).await.unwrap();
    let batch_store = match &writer.mode {
        WriterMode::MemTable { state, .. } => state.read().await.memtable.batch_store(),
        WriterMode::WalOnly { .. } => unreachable!(),
    };
    batch_store
        .fail_blob_pack(&Error::io("injected shared-pack failure"))
        .await;

    let error = writer
        .wal_flusher
        .flush(&WalFlushSource::BatchStore { batch_store }, 1)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("terminally failed Blob v2 pack"));

    let mut watcher = watcher.unwrap();
    assert!(watcher.wait().await.is_err());
    let tailer = WalTailer::new(store, base_path, "", shard_id);
    assert!(tailer.read_entry(1).await.unwrap().is_none());
}

#[tokio::test]
async fn test_failed_shared_blob_pack_retry_cannot_enter_wal() {
    let (store, base_path, controls) = failing_memory_store().await;
    let shard_id = Uuid::new_v4();
    let batch = create_blob_v2_batch(0, &[BlobTestValue::Bytes(b"first".to_vec())]);
    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        "memory://".to_string(),
        ShardWriterConfig {
            shard_id,
            durable_write: true,
            max_wal_buffer_size: usize::MAX,
            max_wal_flush_interval: Some(Duration::from_secs(60)),
            ..Default::default()
        },
        batch.schema(),
        vec![],
    )
    .await
    .unwrap();

    let (_, watcher) = writer.put_no_wait(vec![batch]).await.unwrap();
    let batch_store = match &writer.mode {
        WriterMode::MemTable { state, .. } => state.read().await.memtable.batch_store(),
        WriterMode::WalOnly { .. } => unreachable!(),
    };
    let source = WalFlushSource::BatchStore {
        batch_store: batch_store.clone(),
    };
    controls.fail_blob_puts(1);
    let first_error = writer.wal_flusher.flush(&source, 1).await.unwrap_err();
    assert!(
        first_error
            .to_string()
            .contains("failed to finalize Blob v2 pack")
    );
    assert!(writer.wal_flusher.check_poisoned().is_err());
    assert!(watcher.unwrap().wait().await.is_err());

    assert!(writer.wal_flusher.flush(&source, 1).await.is_err());
    assert!(
        WalTailer::new(store.clone(), base_path.clone(), "", shard_id)
            .read_entry(1)
            .await
            .unwrap()
            .is_none()
    );
    let data_dir = batch_store
        .target()
        .unwrap()
        .generation_path(&base_path, "", &shard_id)
        .join(DATA_DIR);
    assert!(blob_sidecars(store.as_ref(), &data_dir).await.is_empty());
}

#[tokio::test]
async fn test_cancelled_shared_blob_pack_rollover_cannot_ack_missing_payload() {
    let (store, base_path, controls) = failing_memory_store().await;
    let shard_id = Uuid::new_v4();
    let with_small_pack = |batch: RecordBatch| {
        let schema = batch.schema();
        let fields = schema
            .fields()
            .iter()
            .map(|field| {
                let mut field = field.as_ref().clone();
                if field.name() == "blob" {
                    let mut metadata = field.metadata().clone();
                    metadata.insert(
                        lance_arrow::BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY.to_string(),
                        "8".to_string(),
                    );
                    field = field.with_metadata(metadata);
                }
                field
            })
            .collect::<Vec<_>>();
        RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), batch.columns().to_vec()).unwrap()
    };
    let first = with_small_pack(create_blob_v2_batch(
        0,
        &[BlobTestValue::Bytes(b"first".to_vec())],
    ));
    let second = with_small_pack(create_blob_v2_batch(
        1,
        &[BlobTestValue::Bytes(b"later".to_vec())],
    ));
    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        "memory://".to_string(),
        ShardWriterConfig {
            shard_id,
            durable_write: true,
            max_wal_buffer_size: usize::MAX,
            max_wal_flush_interval: Some(Duration::from_secs(60)),
            ..Default::default()
        },
        first.schema(),
        vec![],
    )
    .await
    .unwrap();

    let (_, first_watcher) = writer.put_no_wait(vec![first]).await.unwrap();
    controls.set_block_blob_puts(true);
    let cancelled =
        tokio::time::timeout(Duration::from_millis(50), writer.put_no_wait(vec![second])).await;
    assert!(cancelled.is_err());
    assert_eq!(controls.blob_attempts(), 1);
    controls.set_block_blob_puts(false);

    let batch_store = match &writer.mode {
        WriterMode::MemTable { state, .. } => state.read().await.memtable.batch_store(),
        WriterMode::WalOnly { .. } => unreachable!(),
    };
    let error = writer
        .wal_flusher
        .flush(
            &WalFlushSource::BatchStore {
                batch_store: batch_store.clone(),
            },
            1,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("terminally failed Blob v2 pack"));
    assert!(first_watcher.unwrap().wait().await.is_err());
    assert!(
        WalTailer::new(store.clone(), base_path.clone(), "", shard_id)
            .read_entry(1)
            .await
            .unwrap()
            .is_none()
    );
    let data_dir = batch_store
        .target()
        .unwrap()
        .generation_path(&base_path, "", &shard_id)
        .join(DATA_DIR);
    assert!(blob_sidecars(store.as_ref(), &data_dir).await.is_empty());
}

/// A writer schema whose fields carry their ids, a blob's children
/// included: a blob write is accepted, and the flushed generation keeps the
/// ids the schema gives the blob's children, so it reads back as the same
/// columns. `replayed` drops the writer unclosed, as a crash would, so the
/// blob reaches the generation through WAL replay.
#[rstest]
#[case::non_durable(false, false)]
#[case::durable(true, false)]
#[case::replayed(true, true)]
#[tokio::test]
async fn test_blob_v2_keeps_field_ids_through_put_and_flush(
    #[case] durable_write: bool,
    #[case] replayed: bool,
) {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let batch = create_blob_v2_batch(
        0,
        &[
            BlobTestValue::Bytes(vec![7; 1024 * 1024]),
            BlobTestValue::Bytes(b"abc".to_vec()),
        ],
    );
    let with_ids = Schema::try_from(batch.schema().as_ref()).unwrap();
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        durable_write,
        ..Default::default()
    };
    let schema = Arc::new(crate::dataset::mem_wal::arrow_schema_with_field_ids(
        &with_ids,
    ));
    let open = || {
        ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            config.clone(),
            schema.clone(),
            vec![],
        )
    };
    let mut writer = open().await.unwrap();
    writer.put(vec![batch]).await.unwrap();
    if replayed {
        drop(writer);
        writer = open().await.unwrap();
        assert_eq!(writer.memtable_stats().await.unwrap().row_count, 2);
    }
    writer.close().await.unwrap();

    let manifest = ShardManifestStore::new(store.clone(), &base_path, "", config.shard_id, 2)
        .latest()
        .await
        .unwrap()
        .unwrap();
    let generation = Dataset::open(&format!(
        "{}/_mem_wal/{}/{}",
        base_uri, config.shard_id, manifest.sstables[0].path
    ))
    .await
    .unwrap();
    let blob_children = |schema: &Schema| -> Vec<(String, i32)> {
        schema
            .field("blob")
            .unwrap()
            .children
            .iter()
            .map(|c| (c.name.clone(), c.id))
            .collect()
    };
    assert_eq!(blob_children(generation.schema()), blob_children(&with_ids));
}

#[tokio::test]
async fn test_blob_v2_target_is_reused_across_replay_and_flush() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let external_dir = tempfile::tempdir().unwrap();
    let external_path = external_dir.path().join("external.bin");
    std::fs::write(&external_path, b"external").unwrap();
    let external_uri = format!("file://{}", external_path.display());
    let shard_id = Uuid::new_v4();
    let first = create_blob_v2_batch(
        0,
        &[
            BlobTestValue::Bytes(b"abc".to_vec()),
            BlobTestValue::Bytes(b"de".to_vec()),
            BlobTestValue::Bytes(b"12345".to_vec()),
            BlobTestValue::Bytes(b"123456".to_vec()),
            BlobTestValue::Bytes(vec![7; 1024 * 1024]),
            BlobTestValue::Empty,
            BlobTestValue::Null,
            BlobTestValue::Uri(external_uri),
        ],
    );
    let schema = first.schema();
    let second = create_blob_v2_batch(8, &[BlobTestValue::Bytes(b"later".to_vec())]);
    let index_specs = vec![MemIndexSpec::hnsw(
        "vector_idx",
        4,
        "vector",
        lance_linalg::distance::DistanceType::L2,
    )];
    let config = ShardWriterConfig {
        shard_id,
        durable_write: true,
        max_wal_buffer_size: 1,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        ..Default::default()
    };

    let (target, sidecars_before) = {
        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            config.clone(),
            schema.clone(),
            index_specs.clone(),
        )
        .await
        .unwrap();
        writer.put(vec![first]).await.unwrap();
        writer.put(vec![second]).await.unwrap();

        let (target, row_bytes) = match &writer.mode {
            WriterMode::MemTable { state, .. } => {
                let state = state.read().await;
                (
                    state.memtable.target().unwrap().clone(),
                    state.memtable.batch_store().row_bytes(),
                )
            }
            WriterMode::WalOnly { .. } => unreachable!(),
        };
        assert!(
            row_bytes < 128 * 1024,
            "the 1 MiB payload must not remain resident: {row_bytes} bytes"
        );

        let tailer = WalTailer::new(store.clone(), base_path.clone(), "", config.shard_id);
        for position in [1, 2] {
            let entry = tailer.read_entry(position).await.unwrap().unwrap();
            assert_eq!(entry.target.as_ref(), Some(&target));
            assert_eq!(
                blob_v2_layout(entry.batches[0].schema().field(1)),
                Some(BlobV2Layout::Prepared)
            );
        }

        let first_wal = store
            .inner
            .head(&writer.wal_flusher.wal_appender().wal_entry_path(1))
            .await
            .unwrap();
        assert!(first_wal.size < 128 * 1024);

        let scanned = writer.scan().await.unwrap().try_into_batch().await.unwrap();
        let descriptions = scanned
            .column_by_name("blob")
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let kinds = descriptions
            .column_by_name("kind")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();
        let blob_ids = descriptions
            .column_by_name("blob_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let sizes = descriptions
            .column_by_name("size")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap();
        assert_eq!(kinds.value(0), BlobKind::Inline as u8);
        assert_eq!(sizes.value(0), 3);
        assert_eq!(kinds.value(1), BlobKind::Inline as u8);
        assert_eq!(sizes.value(1), 2);
        assert_eq!(kinds.value(2), BlobKind::Packed as u8);
        assert_eq!(kinds.value(3), BlobKind::Packed as u8);
        assert_eq!(blob_ids.value(2), blob_ids.value(3));
        assert_eq!(kinds.value(4), BlobKind::Dedicated as u8);
        assert_eq!(kinds.value(5), BlobKind::Inline as u8);
        assert!(descriptions.is_null(6));
        assert_eq!(kinds.value(7), BlobKind::External as u8);
        assert_eq!(kinds.value(8), BlobKind::Packed as u8);
        assert_ne!(blob_ids.value(2), blob_ids.value(8));

        let mut point_lookup = writer.scan().await.unwrap();
        point_lookup.filter("id = 0").unwrap();
        let point_lookup = point_lookup.try_into_batch().await.unwrap();
        assert_eq!(point_lookup.num_rows(), 1);
        assert_eq!(
            blob_v2_layout(point_lookup.schema().field(1)),
            Some(BlobV2Layout::Descriptor)
        );

        let data_dir = target
            .generation_path(&base_path, "", &shard_id)
            .join(DATA_DIR);
        let sidecars = blob_sidecars(store.as_ref(), &data_dir).await;
        assert_eq!(sidecars.len(), 3, "two packed puts plus one dedicated blob");
        (target, sidecars)
    };

    // Reopening claims a new epoch. Replay reserves the predecessor's Blob
    // IDs so this writer can safely continue the same target without an
    // eager flush. The lower batch limit also verifies that both the batch
    // store and HNSW storage retain the target's persisted capacity.
    let replay_config = ShardWriterConfig {
        max_memtable_batches: 1,
        ..config
    };
    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        replay_config,
        schema,
        index_specs,
    )
    .await
    .unwrap();
    assert_eq!(writer.memtable_stats().await.unwrap().row_count, 9);
    let replayed_target = match &writer.mode {
        WriterMode::MemTable { state, .. } => state.read().await.memtable.target().unwrap().clone(),
        WriterMode::WalOnly { .. } => unreachable!(),
    };
    assert_eq!(replayed_target, target);
    let manifest = writer.manifest().await.unwrap().unwrap();
    assert!(manifest.sstables.is_empty());

    writer
        .put(vec![create_blob_v2_batch(
            9,
            &[BlobTestValue::Bytes(b"successor".to_vec())],
        )])
        .await
        .unwrap();
    assert_eq!(writer.memtable_stats().await.unwrap().row_count, 10);
    let data_dir = target
        .generation_path(&base_path, "", &shard_id)
        .join(DATA_DIR);
    let sidecars_after_successor = blob_sidecars(store.as_ref(), &data_dir).await;
    assert_eq!(sidecars_after_successor.len(), sidecars_before.len() + 1);
    assert!(
        sidecars_before
            .iter()
            .all(|path| sidecars_after_successor.contains(path)),
        "successor must preserve every predecessor sidecar"
    );

    writer.close().await.unwrap();
    let manifest = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 10)
        .latest()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(manifest.sstables.len(), 1);
    assert_eq!(manifest.sstables[0].path, target.generation_dir);

    let generation_uri = format!(
        "{}/_mem_wal/{}/{}",
        base_uri.trim_end_matches('/'),
        shard_id,
        target.generation_dir
    );
    let dataset = Dataset::open(&generation_uri).await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 10);
    let fragment = dataset.get_fragments().pop().unwrap();
    assert_eq!(fragment.metadata().files.len(), 1);
    assert_eq!(fragment.metadata().files[0].path, target.data_file_name);

    assert_eq!(
        blob_sidecars(store.as_ref(), &data_dir).await,
        sidecars_after_successor,
        "flush must reuse every sidecar written before and after replay"
    );
}

#[tokio::test]
async fn test_mem_wal_keeps_small_blobs_inline_without_sidecars() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let shard_id = Uuid::new_v4();
    let first = create_blob_v2_batch(0, &[BlobTestValue::Bytes(vec![1])]);
    let schema = first.schema();
    let second = create_blob_v2_batch(1, &[BlobTestValue::Bytes(vec![2])]);
    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri,
        ShardWriterConfig {
            shard_id,
            durable_write: false,
            ..Default::default()
        },
        schema,
        vec![],
    )
    .await
    .unwrap();
    writer.put(vec![first]).await.unwrap();
    writer.put(vec![second]).await.unwrap();

    let target = match &writer.mode {
        WriterMode::MemTable { state, .. } => state.read().await.memtable.target().unwrap().clone(),
        WriterMode::WalOnly { .. } => unreachable!(),
    };
    let data_dir = target
        .generation_path(&base_path, "", &shard_id)
        .join(DATA_DIR);
    assert!(blob_sidecars(store.as_ref(), &data_dir).await.is_empty());

    let scanned = writer.scan().await.unwrap().try_into_batch().await.unwrap();
    let descriptions = scanned["blob"]
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let kinds = descriptions
        .column_by_name("kind")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt8Array>()
        .unwrap();
    let sizes = descriptions
        .column_by_name("size")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::UInt64Array>()
        .unwrap();
    assert_eq!(kinds.value(0), BlobKind::Inline as u8);
    assert_eq!(kinds.value(1), BlobKind::Inline as u8);
    assert_eq!(sizes.value(0), 1);
    assert_eq!(sizes.value(1), 1);
    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_rejected_blob_put_does_not_leave_sidecars() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let shard_id = Uuid::new_v4();
    let rejected = create_blob_v2_batch(
        0,
        &[
            BlobTestValue::Bytes(vec![7; 9]),
            BlobTestValue::Uri("relative-uri".to_string()),
        ],
    );
    let schema = rejected.schema();
    let writer = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri,
        ShardWriterConfig {
            shard_id,
            durable_write: false,
            ..Default::default()
        },
        schema,
        vec![],
    )
    .await
    .unwrap();
    assert!(writer.put(vec![rejected]).await.is_err());

    let target = match &writer.mode {
        WriterMode::MemTable { state, .. } => state.read().await.memtable.target().unwrap().clone(),
        WriterMode::WalOnly { .. } => unreachable!(),
    };
    let data_dir = target
        .generation_path(&base_path, "", &shard_id)
        .join(DATA_DIR);
    assert!(blob_sidecars(store.as_ref(), &data_dir).await.is_empty());

    writer
        .put(vec![create_blob_v2_batch(
            2,
            &[BlobTestValue::Bytes(vec![8])],
        )])
        .await
        .unwrap();
    writer.close().await.unwrap();
    assert!(blob_sidecars(store.as_ref(), &data_dir).await.is_empty());
}

#[tokio::test]
async fn test_blob_descriptor_filter_on_fresh_vector_scan() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let batch = create_blob_v2_batch(0, &[BlobTestValue::Bytes(vec![7; 9]), BlobTestValue::Empty]);
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        ShardWriterConfig {
            durable_write: false,
            ..Default::default()
        },
        batch.schema(),
        vec![],
    )
    .await
    .unwrap();
    writer.put(vec![batch]).await.unwrap();

    let mut full = writer.scan().await.unwrap();
    full.filter("blob.size > 0").unwrap();
    assert_eq!(full.try_into_batch().await.unwrap().num_rows(), 1);

    let mut vector = writer.scan().await.unwrap();
    vector.filter("blob.size > 0").unwrap();
    vector
        .nearest("vector", &Float32Array::from(vec![0.0, 1.0]), 1)
        .unwrap();
    assert_eq!(vector.try_into_batch().await.unwrap().num_rows(), 1);
    writer.close().await.unwrap();
}
