// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::num::NonZeroUsize;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use arrow_array::types::{Int32Type, UInt64Type};
use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, cast::AsArray, record_batch};
use arrow_schema::Schema as ArrowSchema;
use bytes::Bytes;
use futures::stream;
use lance_core::{
    ROW_ADDR, ROW_ID, Result, datatypes::BlobKind, datatypes::Schema, utils::address::RowAddress,
};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use lance_io::utils::CachedFileSize;
use lance_table::feature_flags::FLAG_ZONE_WRITE_SEEDS;
use lance_table::format::BasePath;
use lance_table::format::index_seed::IndexSeedWriter;
use lance_table::transaction::{Operation, Transaction};
use lance_table_runtime::blob::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options};
use lance_table_runtime::io::deletion::INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY;
use lance_table_runtime::{
    CommitBuilder, InsertBuilder, Table, WriteMode, WriteParams, WriteSeedHook,
};
use rstest::rstest;

fn values(batch: &RecordBatch) -> &[i32] {
    batch["value"].as_primitive::<Int32Type>().values()
}

fn batch_stream(batch: RecordBatch) -> SendableRecordBatchStream {
    Box::pin(RecordBatchStreamAdapter::new(
        batch.schema(),
        stream::iter([Ok(batch)]),
    ))
}

#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::current(LanceFileVersion::V2_3)]
#[tokio::test]
async fn create_append_overwrite_and_reopen(#[case] version: LanceFileVersion) {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let params = WriteParams {
        data_storage_version: Some(version),
        max_rows_per_file: 2,
        max_rows_per_group: 1,
        enable_stable_row_ids: true,
        ..Default::default()
    };

    let initial = record_batch!(("value", Int32, [0, 1, 2, 3, 4])).unwrap();
    let table = InsertBuilder::new(uri)
        .with_params(&params)
        .execute_stream(batch_stream(initial))
        .await
        .unwrap();
    assert_eq!(table.version_id(), 1);
    assert_eq!(table.count_fragments(), 3);
    assert_eq!(table.count_rows().await.unwrap(), 5);

    let last = table.get_fragments().pop().unwrap();
    let projection = table
        .schema()
        .project_preserve_system_columns(&["value", ROW_ID])
        .unwrap();
    let batch = last.take(&[0], &projection).await.unwrap();
    assert_eq!(values(&batch), &[4]);
    assert_eq!(batch[ROW_ID].as_primitive::<UInt64Type>().value(0), 4);

    let append_params = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };
    let append = record_batch!(("value", Int32, [5, 6])).unwrap();
    let reader = RecordBatchIterator::new([Ok(append.clone())], append.schema());
    let table = InsertBuilder::new(uri)
        .with_params(&append_params)
        .execute_stream(reader)
        .await
        .unwrap();
    assert_eq!(table.version_id(), 2);
    assert!(table.manifest().uses_stable_row_ids());
    let projection = table.schema().project(&["value"]).unwrap();
    let taken = table.take_rows(&[6, 0, 4], &projection).await.unwrap();
    assert_eq!(values(&taken), &[6, 0, 4]);

    let overwrite_params = WriteParams {
        mode: WriteMode::Overwrite,
        ..Default::default()
    };
    let replacement = record_batch!(("value", Int32, [99])).unwrap();
    InsertBuilder::new(&table)
        .with_params(&overwrite_params)
        .execute_stream(batch_stream(replacement))
        .await
        .unwrap();

    let reopened = Table::open(uri).await.unwrap();
    assert_eq!(reopened.version_id(), 3);
    assert_eq!(reopened.count_fragments(), 1);
    assert_eq!(reopened.count_rows().await.unwrap(), 1);
    let taken = reopened.take(&[0], reopened.schema()).await.unwrap();
    assert_eq!(values(&taken), &[99]);
    let initial = reopened.checkout_version(1).await.unwrap();
    assert_eq!(initial.count_rows().await.unwrap(), 5);
}

#[tokio::test]
async fn write_and_read_blob_storage_kinds() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("table");
    let external = dir.path().join("external");
    std::fs::create_dir(&external).unwrap();
    let payload_file = external.join("payload.bin");
    std::fs::write(&payload_file, b"external").unwrap();
    let external_uri = url::Url::from_file_path(payload_file).unwrap();
    let mut blobs = BlobArrayBuilder::new(5);
    blobs.push_bytes(b"tiny").unwrap();
    blobs.push_bytes(b"packed").unwrap();
    blobs.push_bytes(b"dedicated payload").unwrap();
    blobs.push_uri(external_uri.to_string()).unwrap();
    blobs.push_null().unwrap();
    let field = blob_field_with_options(
        "blob",
        true,
        BlobFieldOptions::default()
            .with_inline_size_threshold(4)
            .with_dedicated_size_threshold(NonZeroUsize::new(9).unwrap()),
    );
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![field])),
        vec![blobs.finish().unwrap()],
    )
    .unwrap();
    let base_uri = url::Url::from_directory_path(&external)
        .unwrap()
        .to_string();
    let params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_2),
        max_rows_per_file: 3,
        initial_bases: Some(vec![BasePath::new(1, base_uri, None, false)]),
        ..Default::default()
    };
    let uri = output.to_str().unwrap();
    InsertBuilder::new(uri)
        .with_params(&params)
        .execute_stream(batch_stream(batch))
        .await
        .unwrap();
    let table = Table::open(uri).await.unwrap();
    assert_eq!(table.count_fragments(), 2);
    let handles = table
        .take_blobs_by_indices(&[0, 1, 2, 3, 4], "blob")
        .await
        .unwrap();
    let expected: [(BlobKind, &[u8]); 4] = [
        (BlobKind::Inline, b"tiny"),
        (BlobKind::Packed, b"packed"),
        (BlobKind::Dedicated, b"dedicated payload"),
        (BlobKind::External, b"external"),
    ];
    for (handle, (kind, payload)) in handles.iter().zip(expected) {
        let handle = handle.as_ref().unwrap();
        assert_eq!(handle.kind(), kind);
        assert_eq!(handle.read().await.unwrap().as_ref(), payload);
    }
    assert_eq!(handles.len(), 5);
    assert!(handles[4].is_none());
}

#[rstest]
#[case::external(0)]
#[case::inline(1024)]
#[tokio::test]
async fn publish_overlay_and_extend_deletions(#[case] max_inline_bytes: usize) {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let params = WriteParams {
        max_rows_per_file: 4,
        data_storage_version: Some(LanceFileVersion::V2_3),
        ..Default::default()
    };
    let batch = record_batch!(("value", Int32, [0, 1, 2, 3, 4, 5])).unwrap();
    let table = InsertBuilder::new(uri)
        .with_params(&params)
        .execute_stream(batch_stream(batch))
        .await
        .unwrap();
    let table = CommitBuilder::new(&table)
        .execute(Transaction::new(
            table.version_id(),
            Operation::UpdateConfig {
                config_updates: Some(lance_table::transaction::UpdateMap {
                    update_entries: vec![
                        (
                            INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY.to_string(),
                            max_inline_bytes.to_string(),
                        )
                            .into(),
                    ],
                    replace: false,
                }),
                table_metadata_updates: None,
                schema_metadata_updates: None,
                field_metadata_updates: std::collections::HashMap::new(),
            },
            None,
        ))
        .await
        .unwrap();

    let fragment = &table.get_fragments()[0];
    let addresses =
        [0, 1, 2].map(|offset| u64::from(RowAddress::new_from_parts(fragment.id() as u32, offset)));
    let batch = record_batch!(
        (ROW_ADDR, UInt64, addresses.to_vec()),
        ("value", Int32, [10, 11, 12])
    )
    .unwrap();
    let mut overlay = fragment.write_overlay(table.schema()).await.unwrap();
    overlay.write_batch(&batch).await.unwrap();
    let group = overlay.finish().await.unwrap().unwrap();
    let table = CommitBuilder::new(&table)
        .execute(Transaction::new(
            table.version_id(),
            Operation::DataOverlay {
                groups: vec![group],
            },
            None,
        ))
        .await
        .unwrap();
    let taken = table.take(&[1, 2, 4], table.schema()).await.unwrap();
    assert_eq!(values(&taken), &[11, 12, 4]);

    let mut fragments = table.get_fragments().into_iter();
    let first = fragments.next().unwrap();
    let second = fragments.next().unwrap();
    let deleted_fragment_id = second.id() as u64;
    let staged = first.extend_deletions([1]).await.unwrap().unwrap();
    let staged = staged.extend_deletions([3]).await.unwrap().unwrap();
    let deletion = staged.metadata().deletion_file.as_ref().unwrap();
    assert_eq!(deletion.num_deleted_rows, Some(2));
    assert_eq!(deletion.is_inline(), max_inline_bytes > 0);
    assert!(second.extend_deletions([0, 1]).await.unwrap().is_none());
    let table = CommitBuilder::new(&table)
        .execute(Transaction::new(
            table.version_id(),
            Operation::Delete {
                updated_fragments: vec![staged.into()],
                deleted_fragment_ids: vec![deleted_fragment_id],
                predicate: "physical offsets".to_string(),
            },
            None,
        ))
        .await
        .unwrap();

    let latest = Table::open(uri).await.unwrap();
    assert_eq!(latest.version_id(), table.version_id());
    assert_eq!(latest.count_fragments(), 1);
    assert_eq!(latest.count_rows().await.unwrap(), 2);
    let taken = latest.take(&[1, 0], latest.schema()).await.unwrap();
    assert_eq!(values(&taken), &[12, 10]);
}

#[derive(Debug)]
struct TestSeed {
    key: &'static str,
    sum: i32,
    bias: i32,
    flags: u64,
}

impl IndexSeedWriter for TestSeed {
    fn column_name(&self) -> &str {
        "value"
    }

    fn observe_batch(&mut self, values: &ArrayRef) -> Result<()> {
        self.sum += values
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .sum::<i32>();
        Ok(())
    }

    fn finish(&mut self) -> Result<Option<Bytes>> {
        let sum = std::mem::take(&mut self.sum) + self.bias;
        Ok(Some(Bytes::copy_from_slice(&sum.to_le_bytes())))
    }

    fn schema_metadata_key(&self) -> String {
        self.key.to_string()
    }

    fn schema_metadata_value(&self, buf_index: u32) -> String {
        buf_index.to_string()
    }

    fn required_feature_flags(&self) -> u64 {
        self.flags
    }
}

#[derive(Default)]
struct TestSeedHook(AtomicUsize);

#[async_trait::async_trait]
impl WriteSeedHook for TestSeedHook {
    async fn seed_writers(
        &self,
        table: &Table,
        version: ConcreteFileVersion,
        schema: &Schema,
    ) -> Result<Vec<Box<dyn IndexSeedWriter>>> {
        assert_eq!(table.version_id(), 1);
        assert_eq!(version, ConcreteFileVersion::V2_3);
        assert_eq!(schema.field_id("value")?, table.schema().field_id("value")?);
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok([
            ("test.seed", 0, 0),
            ("test.seed.biased", 100, 0),
            ("test.seed.flagged", 0, FLAG_ZONE_WRITE_SEEDS),
        ]
        .map(|(key, bias, flags)| {
            Box::new(TestSeed {
                key,
                sum: 0,
                bias,
                flags,
            }) as Box<dyn IndexSeedWriter>
        })
        .into_iter()
        .collect())
    }
}

#[tokio::test]
async fn append_seeds_record_actual_buffer_ids_and_reset_between_fragments() {
    let hook = Arc::new(TestSeedHook::default());
    let params = WriteParams::with_storage_version(LanceFileVersion::V2_3);
    let batch = record_batch!(("value", Int32, [0])).unwrap();
    let table = InsertBuilder::new("memory://")
        .with_params(&params)
        .with_seed_hook(hook.clone())
        .execute_stream(batch_stream(batch))
        .await
        .unwrap();
    assert_eq!(hook.0.load(Ordering::Relaxed), 0);

    let params = WriteParams {
        mode: WriteMode::Append,
        max_rows_per_file: 2,
        ..Default::default()
    };
    let batch = record_batch!(("value", Int32, [1, 2, 4])).unwrap();
    let table = InsertBuilder::new(table)
        .with_params(&params)
        .with_seed_hook(hook.clone())
        .execute_stream(batch_stream(batch))
        .await
        .unwrap();
    assert_eq!(hook.0.load(Ordering::Relaxed), 1);
    let store = table.object_store(None).await.unwrap();
    let scheduler = ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store));
    assert_eq!(table.count_fragments(), 3);
    for (fragment, expected) in table.fragments()[1..].iter().zip([3i32, 4]) {
        let path = table.data_dir().join(fragment.files[0].path.as_str());
        let file = scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = FileReader::try_open(
            file,
            None,
            Default::default(),
            &lance_core::cache::LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();
        let metadata = &reader.metadata().file_schema.metadata;
        let plain = metadata["test.seed"].parse().unwrap();
        let biased = metadata["test.seed.biased"].parse().unwrap();
        assert_ne!(plain, biased);
        assert!(!metadata.contains_key("test.seed.flagged"));
        assert_eq!(
            reader.read_global_buffer(plain).await.unwrap().as_ref(),
            &expected.to_le_bytes(),
        );
        assert_eq!(
            reader.read_global_buffer(biased).await.unwrap().as_ref(),
            &(expected + 100).to_le_bytes(),
        );
    }

    let params = WriteParams {
        mode: WriteMode::Overwrite,
        ..Default::default()
    };
    let batch = record_batch!(("value", Int32, [9])).unwrap();
    InsertBuilder::new(table)
        .with_params(&params)
        .with_seed_hook(hook.clone())
        .execute_stream(batch_stream(batch))
        .await
        .unwrap();
    assert_eq!(hook.0.load(Ordering::Relaxed), 1);
}
