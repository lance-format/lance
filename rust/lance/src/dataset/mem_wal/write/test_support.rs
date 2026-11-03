// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared fixtures for MemWAL write-path tests.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Result, datatypes::Schema};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use tempfile::TempDir;
use uuid::Uuid;

use super::{BackpressureController, ShardMemory, ShardWriter, ShardWriterConfig};
use crate::blob::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options};
use crate::dataset::mem_wal::index::{IndexStore, MemIndexSpec};
use crate::dataset::mem_wal::manifest::ShardManifestStore;

pub(super) async fn create_local_store() -> (Arc<ObjectStore>, Path, String, TempDir) {
    let temp_dir = tempfile::tempdir().unwrap();
    let uri = format!("file://{}", temp_dir.path().display());
    let (store, path) = ObjectStore::from_uri(&uri).await.unwrap();
    (store, path, uri, temp_dir)
}

pub(super) fn create_blob_v2_batch(start_id: i32, values: &[BlobTestValue]) -> RecordBatch {
    let field = blob_field_with_options(
        "blob",
        true,
        BlobFieldOptions {
            inline_size_threshold: Some(4),
            dedicated_size_threshold: NonZeroUsize::new(8),
        },
    );
    let id = Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    )]));
    let vector = Field::new(
        "vector",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
        false,
    );
    let schema = Arc::new(ArrowSchema::new(vec![id, field, vector]));
    let mut blobs = BlobArrayBuilder::new(values.len());
    for value in values {
        match value {
            BlobTestValue::Bytes(bytes) => blobs.push_bytes(bytes).unwrap(),
            BlobTestValue::Uri(uri) => blobs.push_uri(uri.clone()).unwrap(),
            BlobTestValue::Empty => blobs.push_empty().unwrap(),
            BlobTestValue::Null => blobs.push_null().unwrap(),
        }
    }
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from_iter_values(
                start_id..start_id + values.len() as i32,
            )),
            blobs.finish().unwrap(),
            Arc::new(
                FixedSizeListArray::try_new_from_values(
                    Float32Array::from_iter_values((0..values.len() * 2).map(|value| value as f32)),
                    2,
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap()
}

pub(super) enum BlobTestValue {
    Bytes(Vec<u8>),
    Uri(String),
    Empty,
    Null,
}

pub(super) async fn blob_sidecars(store: &ObjectStore, data_dir: &Path) -> Vec<Path> {
    use futures::TryStreamExt;

    let mut paths = store
        .inner
        .list(Some(data_dir))
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .into_iter()
        .map(|metadata| metadata.location)
        .filter(|path| path.as_ref().ends_with(".blob"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

/// Base schema with `id` marked as the unenforced primary key (delete needs
/// a PK). `name` is nullable so a tombstone can null it.
pub(super) fn create_pk_test_schema() -> Arc<ArrowSchema> {
    let mut id_metadata = std::collections::HashMap::new();
    id_metadata.insert(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    );
    let id = Field::new("id", DataType::Int32, false).with_metadata(id_metadata);
    Arc::new(ArrowSchema::new(vec![
        id,
        Field::new("name", DataType::Utf8, true),
    ]))
}

/// [`create_pk_test_schema`] with a non-nullable `name` — the shape that
/// used to make `delete` fail.
pub(super) fn create_strict_pk_test_schema() -> Arc<ArrowSchema> {
    let fields: Vec<Field> = create_pk_test_schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone().with_nullable(false))
        .collect();
    Arc::new(ArrowSchema::new(fields))
}

pub(super) fn id_only_keys(ids: &[i32]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )])),
        vec![Arc::new(Int32Array::from(ids.to_vec()))],
    )
    .unwrap()
}

/// A caller-shaped batch that declares `name` nullable and carries a null —
/// legal Arrow, illegal against a base table that declares it non-nullable.
pub(super) fn null_name_batch() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(StringArray::from(vec![Some("a"), None])),
        ],
    )
    .unwrap()
}

/// Read every surviving `id` through the full LSM read path (which folds
/// `NOT _tombstone` per source) over the writer's *flushed* generations,
/// with an optional filter. Mirrors how a query reads a WAL table after a
/// flush — the path the wallop fuzz exercised when it caught a deleted row
/// resurfacing.
pub(super) async fn read_sstable_ids_via_lsm(
    writer: &ShardWriter,
    schema: Arc<ArrowSchema>,
    base_uri: &str,
    shard_id: Uuid,
    filter: Option<&str>,
) -> Vec<i32> {
    use crate::dataset::mem_wal::scanner::{LsmScanner, ShardSnapshot};

    let manifest = writer.manifest().await.unwrap().unwrap();
    let mut snapshot =
        ShardSnapshot::new(shard_id).with_current_generation(manifest.current_generation);
    for sstable in &manifest.sstables {
        snapshot = snapshot.with_sstable(sstable.generation, sstable.path.clone());
    }
    let scanner = LsmScanner::without_base_table(
        schema,
        base_uri.to_string(),
        vec![snapshot],
        vec!["id".to_string()],
    );
    scan_ids(scanner, filter).await
}

/// The ids a read of the writer's in-memory memtables returns.
pub(super) async fn read_memtable_ids_via_lsm(
    writer: &ShardWriter,
    schema: Arc<ArrowSchema>,
    base_uri: &str,
    shard_id: Uuid,
    filter: Option<&str>,
) -> Vec<i32> {
    use crate::dataset::mem_wal::scanner::LsmScanner;

    let refs = writer.in_memory_memtable_refs().await.unwrap();
    let scanner = LsmScanner::without_base_table(
        schema,
        base_uri.to_string(),
        vec![],
        vec!["id".to_string()],
    )
    .with_in_memory_memtables(shard_id, refs);
    scan_ids(scanner, filter).await
}

pub(super) async fn scan_ids(
    mut scanner: crate::dataset::mem_wal::scanner::LsmScanner,
    filter: Option<&str>,
) -> Vec<i32> {
    use futures::TryStreamExt;

    if let Some(predicate) = filter {
        scanner = scanner.filter(predicate).unwrap();
    }
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
    ids
}

pub(super) fn flush_test_config(shard_id: Uuid) -> ShardWriterConfig {
    ShardWriterConfig {
        shard_id,
        durable_write: false,
        manifest_scan_batch_size: 2,
        ..Default::default()
    }
}

pub(super) fn create_test_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

pub(super) fn create_test_batch(
    schema: &ArrowSchema,
    start_id: i32,
    num_rows: usize,
) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from_iter_values(
                start_id..start_id + num_rows as i32,
            )),
            Arc::new(StringArray::from_iter_values(
                (0..num_rows).map(|i| format!("name_{}", start_id as usize + i)),
            )),
        ],
    )
    .unwrap()
}

pub(super) fn wal_only_config(shard_id: Uuid) -> ShardWriterConfig {
    ShardWriterConfig {
        shard_id,
        shard_spec_id: 0,
        durable_write: true,
        enable_memtable: false,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        manifest_scan_batch_size: 2,
        ..Default::default()
    }
}

pub(super) fn memtable_config_with_pk(shard_id: Uuid) -> ShardWriterConfig {
    ShardWriterConfig {
        shard_id,
        shard_spec_id: 0,
        durable_write: true,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    }
}

pub(super) fn schema_with_pk() -> Arc<ArrowSchema> {
    use arrow_schema::Field;
    // Mark `id` as the unenforced primary key.
    let pk_meta: std::collections::HashMap<String, String> = [(
        "lance-schema:unenforced-primary-key".to_string(),
        "1".to_string(),
    )]
    .into_iter()
    .collect();
    let id_field = Field::new("id", DataType::Int32, false).with_metadata(pk_meta);
    Arc::new(ArrowSchema::new(vec![
        id_field,
        Field::new("name", DataType::Utf8, true),
    ]))
}

/// Rows in SSTables after a fresh writer replays the WAL and closes.
pub(super) async fn rows_after_reopen(
    store: &Arc<ObjectStore>,
    base_path: &Path,
    shard_id: Uuid,
    schema: &Arc<ArrowSchema>,
) -> usize {
    ShardWriter::open(
        store.clone(),
        base_path.clone(),
        "memory:///",
        memtable_config_with_pk(shard_id),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap()
    .close()
    .await
    .unwrap();
    ShardManifestStore::new(store.clone(), base_path, "", shard_id, 2)
        .latest()
        .await
        .unwrap()
        .unwrap()
        .sstables
        .iter()
        .map(|sstable| sstable.physical_rows.unwrap() as usize)
        .sum()
}

pub(super) const COVERED_TEST_ROW_CAP: usize = 16;

/// Durable writes so `put` returns only once the row is indexed and
/// WAL-durable. Both fence tests tear the background tasks down before
/// freezing, and a freeze still owing an index apply or a WAL append
/// would dispatch onto a closed channel and poison the writer.
pub(super) fn freeze_fence_test_config(shard_id: Uuid) -> ShardWriterConfig {
    ShardWriterConfig {
        shard_id,
        shard_spec_id: 0,
        durable_write: true,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    }
}

/// Schema for the reservation tests: `vector` is field 1, the id 0 is `id`.
pub(super) fn hnsw_schema(dim: i32) -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            true,
        ),
    ]))
}

/// The shard schema the reservation tests build memtables against.
pub(super) fn test_lance_schema() -> Schema {
    Schema::try_from(hnsw_schema(8).as_ref()).unwrap()
}

pub(super) fn hnsw_specs() -> Vec<MemIndexSpec> {
    vec![MemIndexSpec::hnsw(
        "vec_idx",
        1,
        "vector",
        lance_linalg::distance::DistanceType::L2,
    )]
}

/// What the configured indexes owe before a single row arrives — the figure
/// `open` validates against, computed the way the memtable will build them.
pub(super) fn reserved_index_bytes(config: &ShardWriterConfig, specs: &[MemIndexSpec]) -> usize {
    IndexStore::from_specs(
        specs,
        &test_lance_schema(),
        config.max_memtable_rows,
        config.max_memtable_batches,
    )
    .unwrap()
    .resident_bytes()
        + super::super::memtable::pk_bloom_filter_bytes()
}

/// A controller that admits every write but refuses every freeze — the shape
/// a pod-wide controller takes once the tier its flushes land in is full.
#[derive(Debug)]
pub(super) struct FreezeBlocker {
    pub(super) blocked: std::sync::atomic::AtomicBool,
}

#[async_trait::async_trait]
impl BackpressureController for FreezeBlocker {
    async fn maybe_apply_backpressure(&self, _shard: ShardMemory) -> Result<()> {
        Ok(())
    }

    fn may_freeze_memtable(&self) -> bool {
        !self.blocked.load(Ordering::Relaxed)
    }
}
