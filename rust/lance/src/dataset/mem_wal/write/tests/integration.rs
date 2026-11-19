// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset-level shard writer initialization, indexing, and recovery.

use std::sync::Arc;

use arrow_array::{
    FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::IndexType;
use lance_index::scalar::inverted::{InvertedIndexParams, InvertedListFormatVersion};
use lance_index::scalar::{FullTextSearchQuery, ScalarIndexParams};
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::builder::PQBuildParams;
use lance_linalg::distance::MetricType;
use uuid::Uuid;

use super::super::ShardWriterConfig;
use crate::dataset::mem_wal::DatasetMemWalExt;
use crate::dataset::{Dataset, WriteParams};
use crate::index::DatasetIndexExt;
use crate::index::vector::VectorIndexParams;

pub(super) fn create_test_schema(vector_dim: i32) -> Arc<ArrowSchema> {
    use std::collections::HashMap;

    let mut id_metadata = HashMap::new();
    id_metadata.insert(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    );
    let id_field = Field::new("id", DataType::Int64, false).with_metadata(id_metadata);

    Arc::new(ArrowSchema::new(vec![
        id_field,
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                vector_dim,
            ),
            true,
        ),
        Field::new("text", DataType::Utf8, true),
    ]))
}

fn create_append_only_schema(vector_dim: i32) -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                vector_dim,
            ),
            true,
        ),
        Field::new("text", DataType::Utf8, true),
    ]))
}

pub(super) fn create_test_batch(
    schema: &ArrowSchema,
    start_id: i64,
    num_rows: usize,
    vector_dim: i32,
) -> RecordBatch {
    let vectors: Vec<f32> = (0..num_rows)
        .flat_map(|i| {
            let seed = (start_id as usize + i) as f32;
            (0..vector_dim as usize).map(move |d| (seed * 0.1 + d as f32 * 0.01).sin())
        })
        .collect();

    let vector_array =
        FixedSizeListArray::try_new_from_values(Float32Array::from(vectors), vector_dim).unwrap();

    let texts: Vec<String> = (0..num_rows)
        .map(|i| format!("Sample text for row {}", start_id as usize + i))
        .collect();

    RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from_iter_values(
                start_id..start_id + num_rows as i64,
            )),
            Arc::new(vector_array),
            Arc::new(StringArray::from_iter_values(texts)),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn test_initialize_mem_wal_records_writer_config_defaults() {
    let vector_dim = 128;
    let schema = create_test_schema(vector_dim);
    let uri = format!("memory://test_writer_config_defaults_{}", Uuid::new_v4());

    let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");

    let writer_config = ShardWriterConfig::default()
        .with_durable_write(false)
        .with_max_memtable_size(8 * 1024 * 1024);

    dataset
        .initialize_mem_wal()
        .writer_config_defaults(writer_config)
        .add_writer_config_default("custom_knob", "custom_value")
        .execute()
        .await
        .expect("Failed to initialize MemWAL");

    // Defaults must survive the manifest round-trip so all writers share them.
    let details = dataset
        .mem_wal_index_details()
        .await
        .expect("Failed to read MemWAL index details")
        .expect("MemWAL index details should exist");

    let defaults = &details.writer_config_defaults;
    // ShardWriterConfig tunables are recorded under their field names.
    assert_eq!(
        defaults.get("durable_write").map(String::as_str),
        Some("false")
    );
    assert_eq!(
        defaults.get("max_memtable_size").map(String::as_str),
        Some("8388608")
    );
    // Duration knobs are recorded in milliseconds with a `_ms` suffix.
    assert_eq!(
        defaults
            .get("max_wal_flush_interval_ms")
            .map(String::as_str),
        Some("100")
    );
    // Every tunable field is present.
    assert!(defaults.contains_key("enable_memtable"));
    // add_writer_config_default records arbitrary keys.
    assert_eq!(
        defaults.get("custom_knob").map(String::as_str),
        Some("custom_value")
    );
    // Shard identity is not a configuration default.
    assert!(!defaults.contains_key("shard_id"));
    assert!(!defaults.contains_key("shard_spec_id"));
}

/// A maintained index can be split across multiple physical segments once a
/// delta is appended over previously uncovered fragments (the distributed
/// indexer / `optimize_indices(append)` flow). `mem_wal_writer` must resolve
/// such an index by name without tripping the singular loader's "multiple
/// indices of the same name" error — it only reads the shared type/params,
/// which every segment carries identically.
#[tokio::test]
async fn test_mem_wal_writer_with_multi_segment_index() {
    use lance_index::optimize::OptimizeOptions;

    let vector_dim = 32;
    let schema = create_test_schema(vector_dim);
    // The generation flusher reopens by URI, so this independent open must
    // resolve to the same in-memory backend. The unique authority isolates the test.
    let uri = format!(
        "shared-memory://multi-segment-index-{}/",
        Uuid::new_v4().simple()
    );

    // Initial fragment + an IVF vector index covering it.
    let initial = create_test_batch(&schema, 0, 256, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");
    let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".to_string()),
            &vector_params,
            true,
        )
        .await
        .expect("Failed to create vector index");

    // Append a second fragment and index it as a *delta* (no merge), so the
    // index ends up with two physical segments sharing the name "vector_idx".
    let appended = create_test_batch(&schema, 256, 256, vector_dim);
    let append_batches = RecordBatchIterator::new([Ok(appended)], schema.clone());
    dataset
        .append(append_batches, None)
        .await
        .expect("Failed to append fragment");
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .expect("Failed to append index delta");

    // Precondition: the index is genuinely multi-segment, so the singular
    // `load_index_by_name` would error here.
    assert_eq!(
        dataset
            .load_indices_by_name("vector_idx")
            .await
            .unwrap()
            .len(),
        2,
        "expected two physical segments for the maintained index"
    );

    dataset
        .initialize_mem_wal()
        .maintained_indexes(["vector_idx"])
        .execute()
        .await
        .expect("Failed to initialize MemWAL");

    // The regression: loading the multi-segment maintained index must succeed.
    let shard_id = Uuid::new_v4();
    let writer = dataset
        .mem_wal_writer(
            shard_id,
            ShardWriterConfig::new(shard_id).with_durable_write(false),
        )
        .await
        .expect("mem_wal_writer must accept a multi-segment maintained index");

    // And the resulting writer is functional.
    writer
        .put(vec![create_test_batch(&schema, 200, 10, vector_dim)])
        .await
        .unwrap();
    assert_eq!(writer.memtable_stats().await.unwrap().row_count, 10);
    writer.close().await.unwrap();
}

/// A generation of only tombstones (delete nulls the vector, and the HNSW
/// skips nulls) has an empty vector index. Flushing it must succeed —
/// writing the data with no HNSW index for that generation — not fail the
/// drain with "HnswMemIndex is empty". Regression for the wallop fuzz
/// `flush drain failed: HnswMemIndex is empty` on a delete-only memtable.
#[tokio::test]
async fn test_flush_all_tombstone_generation_skips_empty_hnsw() {
    let vector_dim = 32;
    let schema = create_test_schema(vector_dim);
    // `shared-memory` (not `memory`): the flush writes the generation dataset
    // and the drain reopens it by URI, which a plain in-memory store wouldn't
    // share across opens. A unique authority isolates this test.
    let uri = format!("shared-memory://tombstone-{}/", Uuid::new_v4().simple());

    // Base dataset + maintained vector index, so the writer carries an HNSW
    // mem index.
    let initial = create_test_batch(&schema, 0, 256, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .unwrap();
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".to_string()),
            &VectorIndexParams::ivf_flat(1, MetricType::L2),
            true,
        )
        .await
        .unwrap();
    dataset
        .initialize_mem_wal()
        .maintained_indexes(["vector_idx"])
        .execute()
        .await
        .unwrap();

    let shard_id = Uuid::new_v4();
    let writer = dataset
        .mem_wal_writer(
            shard_id,
            ShardWriterConfig::new(shard_id).with_durable_write(false),
        )
        .await
        .unwrap();

    // Delete only — the memtable holds a single tombstone with a null vector,
    // so its HNSW index is empty.
    let keys = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )])),
        vec![Arc::new(Int64Array::from(vec![0_i64]))],
    )
    .unwrap();
    writer.delete(vec![keys]).await.unwrap();

    // The drain must not error on the empty HNSW.
    writer.force_freeze_active().await.unwrap();
    writer.wait_for_flush_drain().await.unwrap();

    // The tombstone-only generation still flushed (data without an HNSW index).
    let manifest = writer.manifest().await.unwrap().expect("manifest exists");
    assert_eq!(
        manifest.sstables.len(),
        1,
        "the all-tombstone generation must still flush"
    );
    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_mem_wal_maintained_fts_v1_flush_preserves_format() {
    use tempfile::TempDir;

    let vector_dim = 32;
    let schema = create_test_schema(vector_dim);
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let uri = format!("file://{}", temp_dir.path().display());

    let initial = create_test_batch(&schema, 0, 16, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");

    let fts_params = InvertedIndexParams::default().format_version(InvertedListFormatVersion::V1);
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            Some("text_fts".to_string()),
            &fts_params,
            false,
        )
        .await
        .expect("Failed to create v1 FTS index");
    let base_indices = dataset.load_indices().await.unwrap();
    assert_eq!(base_indices.len(), 1);
    assert_eq!(base_indices[0].index_version, 1);

    dataset
        .initialize_mem_wal()
        .maintained_indexes(["text_fts"])
        .execute()
        .await
        .expect("Failed to initialize MemWAL");

    let shard_id = Uuid::new_v4();
    let config = ShardWriterConfig::new(shard_id).with_durable_write(true);
    let writer = dataset
        .mem_wal_writer(shard_id, config)
        .await
        .expect("Failed to create MemWAL writer");
    writer
        .put(vec![create_test_batch(&schema, 1_000, 3, vector_dim)])
        .await
        .expect("Failed to write MemWAL batch");
    writer.close().await.expect("Failed to close writer");

    let (store, base_path) = lance_io::object_store::ObjectStore::from_uri(&uri)
        .await
        .expect("Failed to open store");
    let dir_id = dataset
        .mem_wal_index_details()
        .await
        .unwrap()
        .unwrap()
        .dir_id;
    let manifest_store = crate::dataset::mem_wal::manifest::ShardManifestStore::new(
        store, &base_path, &dir_id, shard_id, 2,
    );
    let manifest = manifest_store
        .latest()
        .await
        .expect("Failed to read manifest")
        .expect("Manifest should exist");
    assert_eq!(manifest.sstables.len(), 1);

    let sstable = &manifest.sstables[0];
    let gen_uri =
        crate::dataset::mem_wal::util::sstable_uri(&uri, &dir_id, &shard_id, &sstable.path);
    let sstable = Dataset::open(&gen_uri)
        .await
        .expect("Failed to open SSTable");
    let sstable_indices = sstable.load_indices().await.unwrap();
    assert_eq!(sstable_indices.len(), 1);
    assert_eq!(sstable_indices[0].name, "text_fts");
    assert_eq!(
        sstable_indices[0].index_version, 1,
        "maintained v1 FTS index must flush as v1"
    );

    let results = sstable
        .scan()
        .full_text_search(FullTextSearchQuery::new("Sample".to_owned()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 3);
}

#[tokio::test]
async fn test_writer_hnsw_params_override() {
    use lance_index::vector::hnsw::builder::HnswBuildParams;

    let vector_dim = 32;
    let schema = create_test_schema(vector_dim);
    // The generation flusher reopens by URI, so this independent open must
    // resolve to the same in-memory backend. The unique authority isolates the test.
    let uri = format!(
        "shared-memory://writer-hnsw-params-{}/",
        Uuid::new_v4().simple()
    );

    let initial = create_test_batch(&schema, 0, 256, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");
    let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".to_string()),
            &vector_params,
            true,
        )
        .await
        .expect("Failed to create vector index");

    // Persisting a ShardWriterConfig that carries HNSW params records them
    // in the writer-config defaults map under `hnsw.<index>.<field>` keys.
    let configured = ShardWriterConfig::new(Uuid::new_v4()).with_hnsw_params(
        "vector_idx",
        HnswBuildParams::default().num_edges(7).ef_construction(48),
    );
    dataset
        .initialize_mem_wal()
        .maintained_indexes(["vector_idx"])
        .writer_config_defaults(configured)
        .execute()
        .await
        .expect("Failed to initialize MemWAL");
    let defaults = dataset
        .mem_wal_index_details()
        .await
        .unwrap()
        .expect("MemWAL details should exist")
        .writer_config_defaults;
    assert_eq!(
        defaults
            .get("hnsw.vector_idx.num_edges")
            .map(String::as_str),
        Some("7")
    );
    assert_eq!(
        defaults
            .get("hnsw.vector_idx.ef_construction")
            .map(String::as_str),
        Some("48")
    );

    // The writer's per-index HNSW override flows into the in-memory index.
    let shard_id = Uuid::new_v4();
    let config = ShardWriterConfig::new(shard_id)
        .with_durable_write(false)
        .with_hnsw_params(
            "vector_idx",
            HnswBuildParams::default().num_edges(7).ef_construction(48),
        );
    let writer = dataset
        .mem_wal_writer(shard_id, config)
        .await
        .expect("mem_wal_writer must accept the HNSW-param override");
    writer
        .put(vec![create_test_batch(&schema, 300, 10, vector_dim)])
        .await
        .unwrap();

    let active = writer.active_memtable_ref().await.unwrap();
    let hnsw = active
        .index_store
        .get_hnsw("vector_idx")
        .expect("maintained HNSW index should exist in the memtable");
    assert_eq!(hnsw.build_params().m, 7);
    assert_eq!(hnsw.build_params().ef_construction, 48);
    drop(active);

    writer.close().await.unwrap();
}

#[tokio::test]
async fn test_initialize_mem_wal_bucket_sharding() {
    let vector_dim = 128;
    let schema = create_test_schema(vector_dim);
    let uri = format!("memory://test_bucket_sharding_{}", Uuid::new_v4());

    let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");

    // num_buckets out of range is rejected.
    let result = dataset
        .initialize_mem_wal()
        .bucket_sharding("id", 0)
        .execute()
        .await;
    assert!(result.is_err(), "num_buckets = 0 should be rejected");

    dataset
        .initialize_mem_wal()
        .bucket_sharding("text", 8)
        .execute()
        .await
        .expect("Failed to initialize MemWAL");

    let details = dataset
        .mem_wal_index_details()
        .await
        .expect("Failed to read MemWAL index details")
        .expect("MemWAL index details should exist");

    assert_eq!(details.num_shards, 8);
    assert_eq!(details.sharding_specs.len(), 1);
    let field = &details.sharding_specs[0].fields[0];
    assert_eq!(field.transform.as_deref(), Some("bucket"));
    assert_eq!(
        field.parameters.get("num_buckets").map(String::as_str),
        Some("8")
    );
    assert_eq!(field.source_ids.len(), 1);
    let source_id = field.source_ids[0];
    let source_field = dataset.schema().field("text").expect("text field exists");
    assert_eq!(source_id, source_field.id);
}

#[tokio::test]
async fn test_initialize_mem_wal_bucket_sharding_without_primary_key() {
    let vector_dim = 128;
    let schema = create_append_only_schema(vector_dim);
    let uri = format!(
        "memory://test_bucket_sharding_no_primary_key_{}",
        Uuid::new_v4()
    );

    let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");

    dataset
        .initialize_mem_wal()
        .bucket_sharding("id", 8)
        .execute()
        .await
        .expect("Failed to initialize append-only MemWAL");

    let details = dataset
        .mem_wal_index_details()
        .await
        .expect("Failed to read MemWAL index details")
        .expect("MemWAL index details should exist");

    assert_eq!(details.num_shards, 8);
    assert_eq!(details.sharding_specs.len(), 1);
    let field = &details.sharding_specs[0].fields[0];
    assert_eq!(field.transform.as_deref(), Some("bucket"));
    assert_eq!(
        field.parameters.get("num_buckets").map(String::as_str),
        Some("8")
    );
}

#[tokio::test]
async fn test_initialize_mem_wal_unsharded() {
    let vector_dim = 128;
    let schema = create_test_schema(vector_dim);
    let uri = format!("memory://test_unsharded_{}", Uuid::new_v4());

    let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");

    dataset
        .initialize_mem_wal()
        .unsharded()
        .execute()
        .await
        .expect("Failed to initialize MemWAL");

    let details = dataset
        .mem_wal_index_details()
        .await
        .expect("Failed to read MemWAL index details")
        .expect("MemWAL index details should exist");

    assert_eq!(details.num_shards, 1);
    assert_eq!(details.sharding_specs.len(), 1);
    assert_eq!(
        details.sharding_specs[0].fields[0].transform.as_deref(),
        Some("unsharded")
    );
}

#[tokio::test]
async fn test_initialize_mem_wal_identity_sharding() {
    let vector_dim = 128;
    let schema = create_test_schema(vector_dim);
    let uri = format!("memory://test_identity_sharding_{}", Uuid::new_v4());

    let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");

    // A column that does not exist is rejected.
    let result = dataset
        .initialize_mem_wal()
        .identity_sharding("nonexistent")
        .execute()
        .await;
    assert!(
        result.is_err(),
        "an unknown identity column should be rejected"
    );

    // A non-scalar column cannot be a shard key.
    let result = dataset
        .initialize_mem_wal()
        .identity_sharding("vector")
        .execute()
        .await;
    assert!(
        result.is_err(),
        "a non-scalar identity column should be rejected"
    );

    dataset
        .initialize_mem_wal()
        .identity_sharding("text")
        .execute()
        .await
        .expect("Failed to initialize MemWAL");

    let details = dataset
        .mem_wal_index_details()
        .await
        .expect("Failed to read MemWAL index details")
        .expect("MemWAL index details should exist");

    // Identity sharding has an open-ended shard count.
    assert_eq!(details.num_shards, 0);
    assert_eq!(details.sharding_specs.len(), 1);
    let field = &details.sharding_specs[0].fields[0];
    assert_eq!(field.transform.as_deref(), Some("identity"));
    assert_eq!(field.result_type.as_str(), "utf8");
    assert_eq!(field.source_ids.len(), 1);
}

/// Quick smoke test for shard writer - runs against memory://
/// Run with: cargo test -p lance tests::integration::test_shard_writer_smoke -- --nocapture
#[tokio::test]
async fn test_shard_writer_smoke() {
    let vector_dim = 128;
    let batch_size = 20;
    let num_batches = 100;

    let schema = create_test_schema(vector_dim);
    let uri = format!("memory://test_shard_writer_{}", Uuid::new_v4());

    // Create initial dataset
    let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");

    // Initialize MemWAL (no indexes for smoke test)
    dataset
        .initialize_mem_wal()
        .execute()
        .await
        .expect("Failed to initialize MemWAL");

    // Create shard writer
    let shard_id = Uuid::new_v4();
    let config = ShardWriterConfig::new(shard_id).with_durable_write(false);

    let writer = dataset
        .mem_wal_writer(shard_id, config)
        .await
        .expect("Failed to create writer");

    // Pre-generate batches
    let batches: Vec<RecordBatch> = (0..num_batches)
        .map(|i| create_test_batch(&schema, (i * batch_size) as i64, batch_size, vector_dim))
        .collect();

    // Write all batches in a single put call for efficiency
    writer.put(batches).await.expect("Failed to write");

    writer.close().await.expect("Failed to close");
}

#[tokio::test]
async fn test_shard_writer_with_vector_index_searches_active_memtable() {
    let vector_dim = 32;
    let batch_size = 20;
    let target_id = 1_000i64 + 37;

    let schema = create_test_schema(vector_dim);
    // The generation flusher reopens by URI, so this independent open must
    // resolve to the same in-memory backend. The unique authority isolates the test.
    let uri = format!(
        "shared-memory://shard-writer-hnsw-{}/",
        Uuid::new_v4().simple()
    );

    let initial_batch = create_test_batch(&schema, 0, 256, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");

    let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".to_string()),
            &vector_params,
            true,
        )
        .await
        .expect("Failed to create base vector index");

    dataset
        .initialize_mem_wal()
        .maintained_indexes(["vector_idx"])
        .execute()
        .await
        .expect("Failed to initialize MemWAL");

    let shard_id = Uuid::new_v4();
    let config = ShardWriterConfig::new(shard_id).with_durable_write(true);

    let writer = dataset
        .mem_wal_writer(shard_id, config)
        .await
        .expect("Failed to create writer");

    let batches: Vec<RecordBatch> = (0..4)
        .map(|i| {
            create_test_batch(
                &schema,
                1_000 + (i * batch_size) as i64,
                batch_size,
                vector_dim,
            )
        })
        .collect();
    writer.put(batches).await.expect("Failed to write");

    let query = Float32Array::from_iter_values(
        (0..vector_dim as usize).map(|d| (target_id as f32 * 0.1 + d as f32 * 0.01).sin()),
    );
    let mut scanner = writer.scan().await.unwrap();
    scanner.nearest("vector", &query, 80).unwrap();
    let result = scanner.try_into_batch().await.expect("Failed to scan");

    assert!(result.num_rows() > 0, "vector query returned no rows");
    let id_col = result
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let dist_col = result
        .column_by_name("_distance")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    let target_idx = (0..result.num_rows())
        .find(|&idx| id_col.value(idx) == target_id)
        .expect("target vector was not returned by active MemTable HNSW search");
    assert!(
        dist_col.value(target_idx) < 1e-6,
        "expected self-match distance near zero, got {}",
        dist_col.value(target_idx)
    );

    writer.close().await.expect("Failed to close");
}

/// Test shard writer against S3 with IVF-PQ, BTree, and FTS indexes (requires DATASET_PREFIX env var)
/// Run with: DATASET_PREFIX=s3://bucket/path cargo test -p lance --release tests::integration::test_shard_writer_s3_ivfpq -- --nocapture --ignored
#[tokio::test]
#[ignore]
async fn test_shard_writer_s3_ivfpq() {
    let prefix = std::env::var("DATASET_PREFIX").expect("DATASET_PREFIX not set");

    let vector_dim = 512;
    let batch_size = 20;
    let num_batches = 10000;
    let num_partitions = 16;
    let num_sub_vectors = 64; // 512 / 8 = 64 subvectors

    let schema = create_test_schema(vector_dim);
    let uri = format!(
        "{}/test_s3_{}",
        prefix.trim_end_matches('/'),
        Uuid::new_v4()
    );

    // Create initial dataset with enough data for IVF-PQ training
    let initial_batch = create_test_batch(&schema, 0, 1000, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");

    // Create BTree index on id column
    let scalar_params = ScalarIndexParams::default();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_btree".to_string()),
            &scalar_params,
            false,
        )
        .await
        .expect("Failed to create BTree index");

    // Create FTS index on text column
    let fts_params = InvertedIndexParams::default();
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            Some("text_fts".to_string()),
            &fts_params,
            false,
        )
        .await
        .expect("Failed to create FTS index");

    // Create IVF-PQ index on dataset

    let ivf_params = IvfBuildParams {
        num_partitions: Some(num_partitions),
        ..Default::default()
    };
    let pq_params = PQBuildParams {
        num_sub_vectors,
        num_bits: 8,
        ..Default::default()
    };
    let vector_params =
        VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);

    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".to_string()),
            &vector_params,
            true,
        )
        .await
        .expect("Failed to create IVF-PQ index");

    // Initialize MemWAL with all three indexes
    dataset
        .initialize_mem_wal()
        .maintained_indexes(["id_btree", "text_fts", "vector_idx"])
        .execute()
        .await
        .expect("Failed to initialize MemWAL");

    // Create shard writer with default config
    let shard_id = Uuid::new_v4();
    let config = ShardWriterConfig::new(shard_id).with_durable_write(false);

    let writer = dataset
        .mem_wal_writer(shard_id, config)
        .await
        .expect("Failed to create writer");

    // Pre-generate batches
    let batches: Vec<RecordBatch> = (0..num_batches)
        .map(|i| create_test_batch(&schema, (i * batch_size) as i64, batch_size, vector_dim))
        .collect();

    // Write all batches in a single put call for efficiency
    writer.put(batches).await.expect("Failed to write");

    writer.close().await.expect("Failed to close");
}

/// End-to-end correctness test for ShardWriter with multiple memtable flushes.
///
/// This test verifies:
/// 1. Multiple memtable flushes are triggered via small memtable size
/// 2. File system layout is correct (WAL files, manifest, generation directories)
/// 3. WAL entries contain expected data
/// 4. Data can be read after each flush cycle
/// 5. Manifest tracks SSTables correctly
///
/// Run with: cargo test -p lance tests::integration::test_shard_writer_e2e_correctness -- --nocapture
#[tokio::test]
async fn test_shard_writer_e2e_correctness() {
    use std::time::Duration;
    use tempfile::TempDir;

    let vector_dim = 32;
    let rows_per_batch = 50;
    // Write enough to trigger ~3 memtable flushes with 50KB memtable size
    // Each batch is ~6KB (50 rows * 32 dims * 4 bytes/float + overhead)
    let num_write_rounds = 3;
    let batches_per_round = 3;

    // Create temp directory for the test
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let uri = format!("file://{}", temp_dir.path().display());

    let schema = create_test_schema(vector_dim);

    // Create initial dataset with enough rows for IVF-PQ training
    let initial_batch = create_test_batch(&schema, 0, 500, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");

    // Create BTree index
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_btree".to_string()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .expect("Failed to create BTree index");

    // Initialize MemWAL with BTree index only (simpler for this test)
    dataset
        .initialize_mem_wal()
        .maintained_indexes(["id_btree"])
        .execute()
        .await
        .expect("Failed to initialize MemWAL");

    // Create shard writer with small memtable size to trigger flushes
    let shard_id = Uuid::new_v4();
    let config = ShardWriterConfig::new(shard_id)
        .with_durable_write(true) // Ensure WAL files are written
        .with_max_memtable_size(50 * 1024) // 50KB - triggers flush after ~8 batches
        .with_max_wal_buffer_size(10 * 1024) // 10KB WAL buffer
        .with_max_wal_flush_interval(Duration::from_millis(50)); // Fast flush

    let writer = dataset
        .mem_wal_writer(shard_id, config)
        .await
        .expect("Failed to create writer");

    let mut total_rows_written = 0i64;

    // Write data in rounds
    for _round in 0..num_write_rounds {
        let start_id = 500 + total_rows_written;
        let batches_to_write: Vec<RecordBatch> = (0..batches_per_round)
            .map(|i| {
                create_test_batch(
                    &schema,
                    start_id + (i * rows_per_batch) as i64,
                    rows_per_batch,
                    vector_dim,
                )
            })
            .collect();

        writer.put(batches_to_write).await.expect("Failed to write");

        total_rows_written += (batches_per_round * rows_per_batch) as i64;

        // Give time for WAL flush and potential memtable flush
        tokio::time::sleep(Duration::from_millis(150)).await;
    }

    // Close writer to ensure final flush
    let dir_id = writer.config().dir_id.clone();
    writer.close().await.expect("Failed to close");

    // === VERIFY FILE SYSTEM LAYOUT ===
    let mem_wal_dir = temp_dir
        .path()
        .join(crate::dataset::mem_wal::util::mem_wal_dir_name(&dir_id))
        .join(shard_id.to_string());
    assert!(mem_wal_dir.exists(), "MemWAL directory should exist");

    // Check WAL directory
    let wal_dir = mem_wal_dir.join("wal");
    assert!(wal_dir.exists(), "WAL directory should exist");
    let wal_files: Vec<_> = std::fs::read_dir(&wal_dir)
        .expect("Failed to read WAL dir")
        .filter_map(|e| e.ok())
        .collect();
    assert!(
        !wal_files.is_empty(),
        "WAL directory should contain at least one file"
    );

    // Check manifest directory
    let manifest_dir = mem_wal_dir.join("manifest");
    assert!(manifest_dir.exists(), "Manifest directory should exist");
    let manifest_files: Vec<_> = std::fs::read_dir(&manifest_dir)
        .expect("Failed to read manifest dir")
        .filter_map(|e| e.ok())
        .collect();
    assert!(
        !manifest_files.is_empty(),
        "Manifest directory should contain at least one file"
    );

    // Read and verify manifest
    let (store, base_path) = lance_io::object_store::ObjectStore::from_uri(&uri)
        .await
        .expect("Failed to open store");
    let dir_id = dataset
        .mem_wal_index_details()
        .await
        .unwrap()
        .unwrap()
        .dir_id;
    let manifest_store = crate::dataset::mem_wal::manifest::ShardManifestStore::new(
        store, &base_path, &dir_id, shard_id, 2,
    );
    let manifest = manifest_store
        .latest()
        .await
        .expect("Failed to read manifest")
        .expect("Manifest should exist");

    // Verify SSTables exist on disk
    assert!(
        !manifest.sstables.is_empty(),
        "Should have at least one SSTable"
    );
    for sstable in &manifest.sstables {
        // The path stored in manifest is relative to the shard directory
        // Construct full path: temp_dir/{mem_wal_dir}/shard_id/generation_folder
        let gen_path = temp_dir
            .path()
            .join(crate::dataset::mem_wal::util::mem_wal_dir_name(&dir_id))
            .join(shard_id.to_string())
            .join(&sstable.path);

        // The generation directory should exist
        assert!(
            gen_path.exists(),
            "SSTable directory should exist at {:?}",
            gen_path
        );

        // Verify generation directory has files
        let gen_contents_count = std::fs::read_dir(&gen_path)
            .expect("Failed to read gen dir")
            .filter_map(|e| e.ok())
            .count();
        assert!(
            gen_contents_count > 0,
            "Generation directory should have files"
        );
    }

    // === VERIFY WAL ENTRIES ===
    // Verify WAL files have correct extension
    for wal_file in wal_files.iter().take(1) {
        let wal_path = wal_file.path();
        let file_name = wal_path.file_name().unwrap().to_string_lossy();
        assert!(
            file_name.ends_with(".arrow"),
            "WAL file should have .arrow extension"
        );
    }

    // === VERIFY DATA CAN BE READ FROM NEW WRITER ===
    // Re-open dataset and create new writer to verify recovery
    let dataset = Dataset::open(&uri).await.expect("Failed to reopen dataset");
    let new_shard_id = Uuid::new_v4();
    // `durable_write(true)` so the put waits for its flush, which is what
    // currently publishes the rows. A non-durable put is *not* yet
    // read-your-writes: the index apply is welded to the WAL flush, so the
    // rows stay invisible until the next flush. This test used to pass with
    // `durable_write(false)` only because an un-advanced cursor of 0 was
    // misread as "batch 0 is visible" — it was asserting the dirty read.
    let new_config = ShardWriterConfig::new(new_shard_id).with_durable_write(true);

    let new_writer = dataset
        .mem_wal_writer(new_shard_id, new_config)
        .await
        .expect("Failed to create new writer");

    // Write a test batch to verify the new shard works
    let verify_batch = create_test_batch(&schema, 10000, 10, vector_dim);
    new_writer
        .put(vec![verify_batch])
        .await
        .expect("Failed to write to new shard");

    let scanner = new_writer.scan().await.unwrap();
    let result = scanner.try_into_batch().await.expect("Failed to scan");
    assert_eq!(result.num_rows(), 10, "New shard should have 10 rows");

    new_writer
        .close()
        .await
        .expect("Failed to close new writer");
}

/// Regression: a base opened with a *path-bound* store binding (the
/// deprecated `ObjectStoreParams::object_store`) must still flush and read
/// generations at their own paths.
///
/// The binding pins a store to one location, and both
/// `ObjectStore::from_uri_and_params` and `DatasetBuilder::build_object_store`
/// take the path from it while ignoring the URI they were handed. Reusing the
/// base's params verbatim therefore aimed every generation write and open at
/// the base table itself: the flush failed ("dataset already exists") and any
/// derived open returned base rows as generation rows.
#[tokio::test]
async fn test_flush_and_read_with_path_bound_object_store() {
    use crate::dataset::mem_wal::scanner::{LsmScanner, ShardSnapshot};
    use futures::TryStreamExt;
    use lance_io::object_store::ObjectStoreParams;
    use tempfile::TempDir;

    let vector_dim = 8;
    let schema = create_test_schema(vector_dim);
    let temp_dir = TempDir::new().unwrap();
    let uri = format!("file://{}", temp_dir.path().display());

    let initial = create_test_batch(&schema, 0, 16, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");
    dataset
        .initialize_mem_wal()
        .execute()
        .await
        .expect("Failed to initialize MemWAL");

    // Re-bind the base to a store pinned at the base's own path — what
    // `DatasetBuilder::with_object_store` leaves on an opened dataset.
    #[allow(deprecated)]
    let store_params = ObjectStoreParams {
        object_store: Some((
            Arc::new(object_store::local::LocalFileSystem::new()),
            url::Url::parse(&uri).unwrap(),
        )),
        ..Default::default()
    };
    let dataset = dataset.with_object_store(dataset.object_store.clone(), Some(store_params));

    let shard_id = Uuid::new_v4();
    let writer = dataset
        .mem_wal_writer(shard_id, ShardWriterConfig::new(shard_id))
        .await
        .expect("Failed to create writer");
    writer
        .put(vec![create_test_batch(&schema, 1_000, 8, vector_dim)])
        .await
        .expect("Failed to write");
    writer.force_freeze_active().await.unwrap();
    writer
        .wait_for_flush_drain()
        .await
        .expect("flush must not be redirected at the base table");

    let manifest = writer.manifest().await.unwrap().expect("manifest exists");
    assert_eq!(manifest.sstables.len(), 1);
    let sstable = manifest.sstables[0].clone();

    // The generation landed under the MemWAL directory, and the base table is untouched.
    let gen_uri = crate::dataset::mem_wal::util::sstable_uri(
        &uri,
        &writer.config().dir_id,
        &shard_id,
        &sstable.path,
    );
    let generation = Dataset::open(&gen_uri)
        .await
        .expect("generation must exist at its own path");
    assert_eq!(generation.count_rows(None).await.unwrap(), 8);
    let base = Dataset::open(&uri).await.unwrap();
    assert_eq!(
        base.count_rows(None).await.unwrap(),
        16,
        "the generation write must not land in the base table"
    );

    // The read path resolves the generation, not the base: 16 base + 8 flushed.
    // Opening the base instead would dedup back down to 16 rows.
    let snapshot = ShardSnapshot::new(shard_id)
        .with_current_generation(manifest.current_generation)
        .with_sstable(sstable.generation, sstable.path.clone());
    let scanner = LsmScanner::new(Arc::new(dataset), vec![snapshot], vec!["id".to_string()])
        .with_mem_wal_dir_id(writer.config().dir_id.clone());
    let rows: usize = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .expect("scan must open the generation, not the base")
        .iter()
        .map(|batch| batch.num_rows())
        .sum();
    assert_eq!(rows, 24);

    writer.close().await.unwrap();
}

/// The store params a base was opened with must reach every *derived* open:
/// the flush that writes a generation and the scan that reads it back.
///
/// This is the point of threading them at all. A namespace-vended store
/// exists only on the params (credentials, endpoint, wrapper), so a
/// generation resolved by URI alone would silently sign with the ambient
/// identity instead — succeeding against a local store and failing against
/// the vended one. Asserting on the *generation folder* rather than
/// `_mem_wal/` is what makes this bite: WAL entries are written through the
/// base dataset's own store, so they would show up here either way.
#[tokio::test]
async fn test_store_params_reach_generation_write_and_read() {
    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::mem_wal::scanner::{LsmScanner, ShardSnapshot};
    use crate::dataset::mem_wal::test_util::observable_store_params;
    use futures::TryStreamExt;
    use tempfile::TempDir;

    let vector_dim = 8;
    let schema = create_test_schema(vector_dim);
    let temp_dir = TempDir::new().unwrap();
    let uri = format!("file://{}", temp_dir.path().display());

    let initial = create_test_batch(&schema, 0, 16, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
    Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");

    // Open the base through an observable store, exactly as a namespace
    // client would hand in a vended-credential store.
    let (store_params, controls) = observable_store_params();
    let mut dataset = DatasetBuilder::from_uri(&uri)
        .with_store_params(store_params)
        .load()
        .await
        .expect("Failed to open dataset");
    dataset
        .initialize_mem_wal()
        .execute()
        .await
        .expect("Failed to initialize MemWAL");

    let shard_id = Uuid::new_v4();
    let writer = dataset
        .mem_wal_writer(shard_id, ShardWriterConfig::new(shard_id))
        .await
        .expect("Failed to create writer");
    writer
        .put(vec![create_test_batch(&schema, 1_000, 8, vector_dim)])
        .await
        .expect("Failed to write");
    writer.force_freeze_active().await.unwrap();
    writer.wait_for_flush_drain().await.expect("flush failed");

    let manifest = writer.manifest().await.unwrap().expect("manifest exists");
    assert_eq!(manifest.sstables.len(), 1);
    let sstable = manifest.sstables[0].clone();

    // The generation's own Lance manifest is the signal to key on. Keying on
    // the generation folder alone would pass vacuously: sidecars like
    // `{gen}/bloom_filter.bin` are written through the *base* dataset's
    // store, which is observable no matter what the params do. And the
    // fragments can't be used either — `ObjectStore::create` writes local
    // files through `tokio::fs`, bypassing the object store entirely, so
    // `{gen}/data/` never reaches a wrapper under `file://`. The manifest
    // goes through `put_opts`, and only the flusher's `Dataset::write` /
    // `open_generation` writes it — both of which must carry the params.
    let gen_manifest = format!("{}/_versions", sstable.path);

    assert!(
        controls.wrote_under(&gen_manifest),
        "the flush must write the generation through the base's store params, \
         not a store resolved from the generation URI alone"
    );

    // And the read path must resolve the generation through them too.
    let snapshot = ShardSnapshot::new(shard_id)
        .with_current_generation(manifest.current_generation)
        .with_sstable(sstable.generation, sstable.path.clone());
    let scanner = LsmScanner::new(Arc::new(dataset), vec![snapshot], vec!["id".to_string()])
        .with_mem_wal_dir_id(writer.config().dir_id.clone());
    let rows: usize = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .expect("scan failed")
        .iter()
        .map(|batch| batch.num_rows())
        .sum();
    assert_eq!(rows, 24);

    // Reads key on the data files, not the manifest: the flusher already
    // pulled the generation's manifest into the shared session cache, so the
    // scan's open serves it from memory and never touches the store. The
    // fragments are read through it (reads have no local bypass), as is the
    // generation's standalone PK index.
    assert!(
        controls.read_under(&format!("{}/data/", sstable.path)),
        "the scan must read the generation through the base's store params"
    );

    writer.close().await.unwrap();
}

/// A fresh-tier-only scanner reaches its store params through
/// `with_store_params`, not `new()`, so the setter must strip the path-bound
/// store binding too. Left raw, it redirects the generation open at the base
/// table and the scan silently returns base rows as WAL rows.
#[tokio::test]
async fn test_fresh_tier_scan_with_path_bound_object_store() {
    use crate::dataset::mem_wal::scanner::{LsmScanner, ShardSnapshot};
    use futures::TryStreamExt;
    use lance_io::object_store::ObjectStoreParams;
    use tempfile::TempDir;

    let vector_dim = 8;
    let schema = create_test_schema(vector_dim);
    let temp_dir = TempDir::new().unwrap();
    let uri = format!("file://{}", temp_dir.path().display());

    // 16 base rows with ids 0..16; the WAL gets 8 rows with ids 1000..1008,
    // so a redirected generation open is unambiguous in the output.
    let initial = create_test_batch(&schema, 0, 16, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");
    dataset
        .initialize_mem_wal()
        .execute()
        .await
        .expect("Failed to initialize MemWAL");

    let shard_id = Uuid::new_v4();
    let writer = dataset
        .mem_wal_writer(shard_id, ShardWriterConfig::new(shard_id))
        .await
        .expect("Failed to create writer");
    writer
        .put(vec![create_test_batch(&schema, 1_000, 8, vector_dim)])
        .await
        .expect("Failed to write");
    writer.force_freeze_active().await.unwrap();
    writer.wait_for_flush_drain().await.expect("flush failed");

    let manifest = writer.manifest().await.unwrap().expect("manifest exists");
    let sstable = manifest.sstables[0].clone();
    let snapshot = ShardSnapshot::new(shard_id)
        .with_current_generation(manifest.current_generation)
        .with_sstable(sstable.generation, sstable.path.clone());

    // What `DatasetBuilder::with_object_store` leaves on an opened dataset:
    // a store pinned at the base's own path.
    #[allow(deprecated)]
    let store_params = ObjectStoreParams {
        object_store: Some((
            Arc::new(object_store::local::LocalFileSystem::new()),
            url::Url::parse(&uri).unwrap(),
        )),
        ..Default::default()
    };

    let arrow_schema: Arc<ArrowSchema> = schema.clone();
    let batches = LsmScanner::without_base_table(
        arrow_schema,
        uri.clone(),
        vec![snapshot],
        vec!["id".to_string()],
    )
    .with_mem_wal_dir_id(writer.config().dir_id.clone())
    .with_session(dataset.session())
    .with_store_params(store_params)
    .try_into_stream()
    .await
    .unwrap()
    .try_collect::<Vec<_>>()
    .await
    .expect("scan must open the generation, not the base");

    let rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(
        rows, 8,
        "fresh tier holds only the 8 WAL rows; 16 means the generation open \
         was redirected at the base table"
    );
    let ids: Vec<i64> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert!(
        ids.iter().all(|id| (1_000..1_008).contains(id)),
        "expected the WAL's own rows, got {ids:?}"
    );

    writer.close().await.unwrap();
}

/// The other paths now prevent a nullable key, so this forges one to stand
/// in for a table written before they were closed.
#[tokio::test]
async fn test_initialize_mem_wal_rejects_a_nullable_primary_key() {
    let vector_dim = 128;
    let schema = create_append_only_schema(vector_dim);
    let uri = format!("memory://test_mem_wal_nullable_pk_{}", Uuid::new_v4());

    let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
    let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
    let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
        .await
        .expect("Failed to create dataset");

    {
        let manifest = Arc::make_mut(&mut dataset.manifest);
        let id_field = manifest
            .schema
            .fields
            .iter_mut()
            .find(|field| field.name == "id")
            .expect("schema has an id column");
        id_field.unenforced_primary_key_position = Some(1);
        id_field.nullable = true;
    }

    let err = dataset
        .initialize_mem_wal()
        .unsharded()
        .execute()
        .await
        .expect_err("MemWAL must not enable on a nullable primary key");
    assert!(
        err.to_string().contains("must not be nullable"),
        "unexpected error: {err}"
    );
}
