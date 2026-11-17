// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Core put, visibility, and index configuration behavior.

use std::sync::Arc;
use std::time::Duration;

use uuid::Uuid;

use super::super::test_support::*;
use super::super::{ShardWriter, ShardWriterConfig};
use crate::dataset::mem_wal::index::MemIndexSpec;

#[tokio::test]
async fn test_shard_writer_basic_write() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        config.clone(),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    // Write a batch
    let batch = create_test_batch(&schema, 0, 10);
    let result = writer.put(vec![batch]).await.unwrap();

    assert_eq!(result.batch_positions, 0..1);

    // Check stats
    let stats = writer.memtable_stats().await.unwrap();
    assert_eq!(stats.row_count, 10);
    assert_eq!(stats.batch_count, 1);

    // Close writer
    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_put_no_wait_durable_visible_then_durable() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: true,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    let batch = create_test_batch(&schema, 0, 10);
    let (result, watcher) = writer.put_no_wait(vec![batch]).await.unwrap();
    assert_eq!(result.batch_positions, 0..1);

    // Row is visible in memory before durability is awaited.
    let stats = writer.memtable_stats().await.unwrap();
    assert_eq!(stats.row_count, 10);

    // durable_write is on, so a watcher is returned and resolves once the
    // triggered flush lands.
    let mut watcher = watcher.expect("durable_write returns a watcher");
    watcher.wait().await.unwrap();

    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_non_durable_put_is_read_your_writes() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    let batch = create_test_batch(&schema, 0, 10);
    let (result, watcher) = writer.put_no_wait(vec![batch]).await.unwrap();
    assert_eq!(result.batch_positions, 0..1);

    // A non-durable put still has something to await: its *index apply*.
    // Skipping durability now costs the caller durability only — not
    // visibility. Before the index apply was split off the WAL flush, a
    // non-durable write was not read-your-writes at all: the row stayed
    // invisible until some later flush happened to index it.
    let mut watcher = watcher.expect("a non-durable put awaits its index apply");
    watcher.wait().await.unwrap();

    let scanned = writer.scan().await.unwrap().try_into_batch().await.unwrap();
    assert_eq!(
        scanned.num_rows(),
        10,
        "a non-durable put must be readable as soon as it returns"
    );

    let stats = writer.memtable_stats().await.unwrap();
    assert_eq!(stats.row_count, 10);

    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_shard_writer_multiple_writes() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    // Write multiple batches in a single put call
    let batches: Vec<_> = (0..5)
        .map(|i| create_test_batch(&schema, i * 10, 10))
        .collect();
    let result = writer.put(batches).await.unwrap();
    assert_eq!(result.batch_positions, 0..5);

    let stats = writer.memtable_stats().await.unwrap();
    assert_eq!(stats.row_count, 50);
    assert_eq!(stats.batch_count, 5);

    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_shard_writer_with_indexes() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let index_specs = vec![MemIndexSpec::btree("id_idx", 0, "id")];

    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        config,
        schema.clone(),
        index_specs,
    )
    .await
    .unwrap();

    // Write a batch
    let batch = create_test_batch(&schema, 0, 10);
    writer.put(vec![batch]).await.unwrap();

    let stats = writer.memtable_stats().await.unwrap();
    assert_eq!(stats.row_count, 10);

    writer.close().await.unwrap();
}

/// Swapping the set under a live write load neither loses a write nor
/// wedges one.
///
/// The swap takes the same lock a write takes, so a writer that is mid-put
/// waits for it and every writer after it sees one complete set. This pins
/// that the wait is all that happens: no write fails, no swap fails, and
/// the writer is still usable afterwards with the set last installed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_replace_index_configs_under_concurrent_writes() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 256 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };
    let id_idx = MemIndexSpec::btree("id_idx", 0, "id");
    let writer = Arc::new(
        ShardWriter::open(
            store,
            base_path,
            base_uri,
            config,
            schema.clone(),
            vec![id_idx.clone()],
        )
        .await
        .unwrap(),
    );

    const WRITERS: i32 = 4;
    const BATCHES: i32 = 25;
    let writes = (0..WRITERS).map(|w| {
        let writer = Arc::clone(&writer);
        let schema = schema.clone();
        tokio::spawn(async move {
            for b in 0..BATCHES {
                let offset = (w * BATCHES + b) * 10;
                writer
                    .put(vec![create_test_batch(&schema, offset, 10)])
                    .await
                    .expect("a write must not fail while the set is swapped");
            }
        })
    });

    let swaps = {
        let writer = Arc::clone(&writer);
        let id_idx = id_idx.clone();
        tokio::spawn(async move {
            for round in 0..10 {
                let configs = if round % 2 == 0 {
                    Vec::new()
                } else {
                    vec![id_idx.clone()]
                };
                writer
                    .replace_index_configs(configs)
                    .await
                    .expect("a swap must not fail while writes are in flight");
                tokio::task::yield_now().await;
            }
        })
    };

    for write in writes {
        write.await.unwrap();
    }
    swaps.await.unwrap();

    // Ten rounds starting at 0 end on an odd round, which installs the index.
    assert_eq!(
        writer.maintained_index_names().await,
        vec!["id_idx".to_string()]
    );

    // Still serving: the swaps left no lock held and no state half-applied.
    writer
        .put(vec![create_test_batch(&schema, 100_000, 10)])
        .await
        .expect("the writer is usable after the swaps");
    let refs = writer.in_memory_memtable_refs().await.unwrap();
    assert!(
        refs.active.index_store.get_btree("id_idx").is_some(),
        "the memtable in force carries the set last installed"
    );

    Arc::into_inner(writer)
        .expect("every task has finished, so this is the last handle")
        .close()
        .await
        .unwrap();
}

/// Replacing the set reaches the next memtable.
///
/// The memtable holding rows is frozen so the one that follows carries the
/// new index, and the writer keeps serving throughout.
#[tokio::test]
async fn test_replace_index_configs_reaches_the_next_memtable() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };
    let id_idx = MemIndexSpec::btree("id_idx", 0, "id");
    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri,
        config,
        schema.clone(),
        vec![id_idx.clone()],
    )
    .await
    .unwrap();

    // Empty memtable: replaced in place, no new generation.
    let generation = writer.active_memtable_ref().await.unwrap().generation;
    for set in [Vec::new(), vec![id_idx.clone()]] {
        let fence = writer
            .replace_index_configs(set)
            .await
            .unwrap()
            .expect("the set changed, so a fence is returned");
        assert!(
            fence.frozen_generation().is_none(),
            "an empty memtable is not frozen"
        );
        assert_eq!(
            writer.active_memtable_ref().await.unwrap().generation,
            generation,
        );
    }

    writer
        .put(vec![create_test_batch(&schema, 0, 10)])
        .await
        .unwrap();

    // Narrowing to nothing freezes the memtable that holds the rows.
    let fence = writer
        .replace_index_configs(Vec::new())
        .await
        .unwrap()
        .expect("the set changed, so a fence is returned");
    assert!(
        fence.frozen_generation().is_some(),
        "a memtable holding rows is frozen so the next one carries the new set"
    );

    // The memtable that follows is built from the replacement.
    writer
        .put(vec![create_test_batch(&schema, 10, 10)])
        .await
        .unwrap();
    let refs = writer.in_memory_memtable_refs().await.unwrap();
    assert!(
        refs.active.index_store.get_btree("id_idx").is_none(),
        "the memtable opened after the replacement must not carry the dropped index"
    );
    assert!(
        refs.frozen
            .iter()
            .any(|f| f.index_store.get_btree("id_idx").is_some()),
        "the frozen memtable keeps the index it was written with"
    );

    // Re-applying the same set is a no-op: nothing is frozen.
    assert!(
        writer
            .replace_index_configs(Vec::new())
            .await
            .unwrap()
            .is_none(),
        "an unchanged set freezes nothing"
    );

    writer.close().await.unwrap();
}

/// End-to-end check that the background flush handler rebuilds secondary
/// indexes on every SSTable. Before this, the handler flushed
/// via plain `flush`, leaving SSTables unindexed — point
/// lookups had to full-scan and vector search's index-only `fast_search`
/// couldn't see the data at all.
#[tokio::test]
async fn test_sstable_is_indexed() {
    use crate::index::DatasetIndexExt;

    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();
    let shard_id = Uuid::new_v4();

    let config = ShardWriterConfig {
        shard_id,
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let index_specs = vec![MemIndexSpec::btree("id_idx", 0, "id")];

    let writer = ShardWriter::open(
        store,
        base_path,
        base_uri.clone(),
        config,
        schema.clone(),
        index_specs,
    )
    .await
    .unwrap();

    writer
        .put(vec![create_test_batch(&schema, 0, 10)])
        .await
        .unwrap();

    // Freeze the active memtable and wait until it lands on disk.
    writer.force_freeze_active().await.unwrap();
    writer.wait_for_flush_drain().await.unwrap();

    // Resolve the SSTable recorded in the manifest.
    let manifest = writer.manifest().await.unwrap().unwrap();
    assert_eq!(manifest.sstables.len(), 1, "expected exactly one SSTable");
    let gen_uri = format!(
        "{}/_mem_wal/{}/{}",
        base_uri, shard_id, manifest.sstables[0].path
    );

    // The SSTable must carry the BTree index built during flush.
    let dataset = crate::Dataset::open(&gen_uri).await.unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1, "SSTable should have one index");
    assert_eq!(indices[0].name, "id_idx");

    // A PK filter over it must resolve through the index, not a full scan.
    let mut scan = dataset.scan();
    scan.filter("id = 5").unwrap();
    scan.prefilter(true);
    let plan = scan.create_plan().await.unwrap();
    crate::utils::test::assert_plan_node_equals(
        plan,
        "LanceRead: ...full_filter=id = Int32(5)...
ScalarIndexQuery: query=[id = 5]@id_idx(BTree)",
    )
    .await
    .unwrap();

    // And the index returns the correct row.
    let batch = dataset
        .scan()
        .filter("id = 5")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch.num_rows(), 1);

    writer.close().await.unwrap();
}

/// Test memtable auto-flush triggered by size threshold.
#[tokio::test]
async fn test_shard_writer_auto_flush_by_size() {
    let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
    let schema = create_test_schema();

    // Use a small memtable size to trigger auto-flush
    let config = ShardWriterConfig {
        shard_id: Uuid::new_v4(),
        shard_spec_id: 0,
        durable_write: false,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 1024, // Very small - will trigger flush quickly
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
        .await
        .unwrap();

    let initial_gen = writer.memtable_stats().await.unwrap().generation;

    // Write batches until auto-flush triggers
    for i in 0..20 {
        let batch = create_test_batch(&schema, i * 10, 10);
        writer.put(vec![batch]).await.unwrap();
    }

    // Give time for background flush to process
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Check that generation increased (indicating flush happened)
    let stats = writer.memtable_stats().await.unwrap();
    assert!(
        stats.generation > initial_gen,
        "Generation should increment after auto-flush"
    );

    writer.close().await.unwrap();
}
