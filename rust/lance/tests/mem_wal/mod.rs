// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end MemWAL tests through the public dataset and [`ShardWriter`] APIs, as
//! opposed to the lib-level unit tests that can reach internals directly.

use std::{cell::RefCell, time::Duration};

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{RecordBatchIterator, record_batch};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::dataset::mem_wal::{DatasetMemWalExt, ShardWriter, ShardWriterConfig};
use lance::index::DatasetIndexExt;
use lance_core::FenceReason;
use lance_index::IndexType;
use lance_index::mem_wal::MEM_WAL_INDEX_NAME;
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_io::object_store::ObjectStore;
use uuid::Uuid;

mod future_depth;

#[tokio::test(flavor = "current_thread")]
async fn full_text_search_after_initialize_mem_wal_does_not_log_errors() {
    // Keep capture on this test's runtime thread so parallel tests cannot contribute logs.
    thread_local! {
        static INDEX_ERRORS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    }
    struct IndexErrorLogger;
    impl log::Log for IndexErrorLogger {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.target() == "lance::index" && metadata.level() == log::Level::Error
        }

        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                INDEX_ERRORS.with_borrow_mut(|errors| errors.push(record.args().to_string()));
            }
        }

        fn flush(&self) {}
    }
    log::set_logger(&IndexErrorLogger).unwrap();
    log::set_max_level(log::LevelFilter::Error);

    let batch = record_batch!(
        ("id", Int32, [0, 1]),
        (
            "code",
            Utf8,
            ["def getUserName(): pass", "def getOrder(): pass"]
        )
    )
    .unwrap();
    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().to_str().unwrap();
    let mut dataset = Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    dataset
        .create_index(
            &["code"],
            IndexType::Inverted,
            Some("code_idx".to_string()),
            &InvertedIndexParams::code().split_identifiers(true),
            true,
        )
        .await
        .unwrap();
    dataset
        .initialize_mem_wal()
        .unsharded()
        .maintained_indexes(["code_idx"])
        .execute()
        .await
        .unwrap();

    let dataset = Dataset::open(uri).await.unwrap();
    let indices = dataset.load_indices().await.unwrap();
    let mem_wal = indices
        .iter()
        .find(|index| index.name == MEM_WAL_INDEX_NAME)
        .unwrap();
    assert!(mem_wal.fields.is_empty());
    INDEX_ERRORS.with_borrow_mut(Vec::clear);
    for _ in 0..2 {
        let result = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("user".to_string()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result["id"].as_primitive::<Int32Type>().values(), &[0]);
        INDEX_ERRORS.with_borrow(|errors| assert!(errors.is_empty(), "{errors:?}"));
    }
}

fn durable_writer_config(shard_id: Uuid) -> ShardWriterConfig {
    ShardWriterConfig {
        shard_id,
        durable_write: true,
        max_wal_buffer_size: 64 * 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    }
}

#[tokio::test]
async fn durable_put_is_readable_through_public_scan() {
    let base_uri = "memory://".to_string();
    let (store, base_path) = ObjectStore::from_uri(&base_uri).await.unwrap();
    let batch = record_batch!(("id", Int32, [1, 2, 3])).unwrap();
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        durable_writer_config(Uuid::new_v4()),
        batch.schema(),
        vec![],
    )
    .await
    .unwrap();

    writer.put(vec![batch]).await.unwrap();

    let scanned = writer.scan().await.unwrap().try_into_batch().await.unwrap();
    let scanned_ids = scanned["id"].as_primitive::<Int32Type>();
    assert_eq!(
        scanned_ids.values(),
        &[1, 2, 3],
        "a durable put must be readable through the public scan API"
    );

    writer.close().await.unwrap();
}

/// A durable put into a post-rotation MemTable must be acknowledged only by
/// its own WAL flush, never by a flush from an older generation.
///
/// Batch positions restart at 0 in every MemTable generation. A durability
/// watch keyed on that generation-local position is satisfied by the previous
/// generation's flush at the same position — so the first durable put after a
/// rotation used to return success before its own WAL append completed. Here
/// a peer writer has claimed a higher epoch in between, so the false success
/// is observable: the put's own flush is fenced, and the put must surface
/// that fence instead of reporting durability.
#[tokio::test]
async fn durable_put_does_not_alias_across_memtable_generations() {
    let base_uri = "memory://".to_string();
    let (store, base_path) = ObjectStore::from_uri(&base_uri).await.unwrap();
    let shard_id = Uuid::new_v4();
    let first_batch = record_batch!(("id", Int32, [1])).unwrap();
    let schema = first_batch.schema();
    let config = durable_writer_config(shard_id);

    let writer_a = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        config.clone(),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    // Write and fully flush generation N, then rotate to generation N+1.
    writer_a.put(vec![first_batch]).await.unwrap();
    let first_generation = writer_a.memtable_stats().await.unwrap().generation;
    writer_a.force_seal_active().await.unwrap();
    writer_a.wait_for_flush_drain().await.unwrap();
    let current_generation = writer_a.memtable_stats().await.unwrap().generation;
    assert_eq!(
        current_generation,
        first_generation + 1,
        "expected rotation from generation {first_generation}, got {current_generation}"
    );

    // A peer claims a higher epoch, fencing writer A's next WAL append.
    let writer_b = ShardWriter::open(store, base_path, base_uri, config, schema, vec![])
        .await
        .unwrap();
    assert!(
        writer_b.epoch() > writer_a.epoch(),
        "expected peer epoch to increase: writer_a={}, writer_b={}",
        writer_a.epoch(),
        writer_b.epoch()
    );

    // Generation N+1's first durable put lands at the same local batch
    // position (0..1) that generation N already flushed. It must wait for its
    // own flush and therefore surface the fence, not ack from the stale
    // generation.
    let second_batch = record_batch!(("id", Int32, [2])).unwrap();
    let error = writer_a
        .put(vec![second_batch])
        .await
        .expect_err("generation-2 durable put must wait for its own WAL flush");
    assert_eq!(
        error.fence_reason(),
        Some(FenceReason::PeerClaimedEpoch),
        "expected a peer-claimed-epoch fence, got: {error}"
    );
    assert!(
        error.to_string().contains("Writer fenced"),
        "expected the writer-fenced error prefix, got: {error}"
    );

    writer_b.close().await.unwrap();
}
