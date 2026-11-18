// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{
    Dataset,
    blob::{BlobArrayBuilder, blob_field},
    dataset::WriteParams,
    utils::test::TestDatasetGenerator,
};
use arrow::datatypes::UInt64Type;
use arrow_array::RecordBatch;
use arrow_array::{
    Array, ArrayRef, LargeBinaryArray, RecordBatchIterator, StringArray, StructArray, UInt32Array,
    UInt64Array,
};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use bytes::Bytes;
use chrono::Utc;
use futures::StreamExt;
use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, BLOB_V2_EXT_NAME};
use lance_core::datatypes::BLOB_V2_LOGICAL_FIELDS;
use lance_core::{
    Result,
    utils::tempfile::{TempDir, TempStrDir},
};
use lance_datagen::{BatchCount, RowCount, array};
use lance_file::version::LanceFileVersion;
use lance_io::storage::Storage;
use lance_io::storage::{
    Attributes, CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, ObjectMeta,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, path::Path,
};
use lance_table::feature_flags::MANAGED_BLOBS_META_KEY;
use lance_table::format::BasePath;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::{Notify, Semaphore};
use url::Url;

pub fn managed_blob_schema(fields: Vec<Field>) -> Schema {
    Schema::new_with_metadata(
        fields,
        HashMap::from([(MANAGED_BLOBS_META_KEY.to_string(), "true".to_string())]),
    )
}

pub struct BlobTestFixture {
    pub _test_dir: TempStrDir,
    pub dataset: Arc<Dataset>,
    pub data: Vec<RecordBatch>,
}

impl BlobTestFixture {
    pub async fn new() -> Self {
        let test_dir = TempStrDir::default();

        let data = lance_datagen::gen_batch()
            .col("filterme", array::step::<UInt64Type>())
            .col("blobs", array::blob())
            .into_reader_rows(RowCount::from(10), BatchCount::from(4))
            .map(|batch| Ok(batch?))
            .collect::<Result<Vec<_>>>()
            .unwrap();

        let dataset = Arc::new(
            TestDatasetGenerator::new(data.clone(), LanceFileVersion::V2_1)
                .make_hostile(&test_dir)
                .await,
        );

        Self {
            _test_dir: test_dir,
            dataset,
            data,
        }
    }
}

pub struct MultiBaseBlobFixture {
    pub _test_dir: TempDir,
    pub dataset: Arc<Dataset>,
    pub expected: Vec<u8>,
}

pub fn complete_blob_v2_field(name: &str, nullable: bool) -> Field {
    Field::new(
        name,
        DataType::Struct(BLOB_V2_LOGICAL_FIELDS.clone()),
        nullable,
    )
    .with_metadata(HashMap::from([(
        ARROW_EXT_NAME_KEY.to_string(),
        BLOB_V2_EXT_NAME.to_string(),
    )]))
}

pub fn complete_blob_v2_array(
    data: Vec<Option<Vec<u8>>>,
    uris: Vec<Option<String>>,
    positions: Vec<Option<u64>>,
    sizes: Vec<Option<u64>>,
    validity: Option<NullBuffer>,
) -> ArrayRef {
    Arc::new(
        StructArray::try_new(
            BLOB_V2_LOGICAL_FIELDS.clone(),
            vec![
                Arc::new(LargeBinaryArray::from_iter(
                    data.iter().map(|value| value.as_deref()),
                )) as ArrayRef,
                Arc::new(StringArray::from_iter(
                    uris.iter().map(|value| value.as_deref()),
                )) as ArrayRef,
                Arc::new(UInt64Array::from(positions)) as ArrayRef,
                Arc::new(UInt64Array::from(sizes)) as ArrayRef,
            ],
            validity,
        )
        .unwrap(),
    )
}

pub fn nested_blob_v2_batch_with_field(
    blob_field: Field,
    blob_array: ArrayRef,
) -> (Arc<Schema>, RecordBatch) {
    let info_fields = vec![Field::new("name", DataType::Utf8, false), blob_field];
    let info_array: ArrayRef = Arc::new(
        StructArray::try_new(
            info_fields.clone().into(),
            vec![
                Arc::new(StringArray::from_iter_values(
                    (0..blob_array.len()).map(|idx| format!("name-{idx}")),
                )) as ArrayRef,
                blob_array,
            ],
            None,
        )
        .unwrap(),
    );

    let schema = Arc::new(Schema::new(vec![Field::new(
        "info",
        DataType::Struct(info_fields.into()),
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![info_array]).unwrap();
    (schema, batch)
}

pub fn nested_blob_v2_batch(blob_array: ArrayRef) -> (Arc<Schema>, RecordBatch) {
    nested_blob_v2_batch_with_field(blob_field("blob", true), blob_array)
}

#[derive(Debug)]
pub struct RejectEmptyRangeObjectStore;

impl std::fmt::Display for RejectEmptyRangeObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RejectEmptyRangeObjectStore")
    }
}

#[async_trait]
impl lance_io::storage::StorageBackend for RejectEmptyRangeObjectStore {
    async fn list_paginated(
        &self,
        prefix: Option<&str>,
        options: lance_io::storage::PaginatedListOptions,
    ) -> lance_io::storage::Result<lance_io::storage::PaginatedListResult> {
        let _ = (prefix, options);
        Err(lance_io::storage::Error::NotSupported {
            source: "native pagination is unavailable".into(),
        })
    }

    async fn put_opts(
        &self,
        _location: &Path,
        _bytes: PutPayload,
        _opts: PutOptions,
    ) -> lance_io::storage::Result<PutResult> {
        unimplemented!("put_opts is not used by these tests")
    }

    async fn put_multipart_opts(
        &self,
        _location: &Path,
        _opts: PutMultipartOptions,
    ) -> lance_io::storage::Result<Box<dyn lance_io::storage::backend::MultipartUpload>> {
        unimplemented!("put_multipart_opts is not used by these tests")
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> lance_io::storage::Result<GetResult> {
        let Some(range) = options.range else {
            unreachable!("blob reads should always request a bounded range")
        };
        if range.start == range.end {
            return Err(lance_io::storage::Error::Generic {
                store: "RejectEmptyRangeObjectStore",
                source: format!(
                    "Range started at {} and ended at {}",
                    range.start, range.end
                )
                .into(),
            });
        }
        Err(lance_io::storage::Error::NotSupported {
            source: format!("unexpected non-empty range {range:?} for {location}").into(),
        })
    }

    fn delete_stream(
        &self,
        _locations: futures::stream::BoxStream<'static, lance_io::storage::Result<Path>>,
    ) -> futures::stream::BoxStream<'static, lance_io::storage::Result<Path>> {
        unimplemented!("delete_stream is not used by these tests")
    }

    fn list(
        &self,
        _prefix: Option<&Path>,
    ) -> futures::stream::BoxStream<'static, lance_io::storage::Result<ObjectMeta>> {
        unimplemented!("list is not used by these tests")
    }

    async fn list_with_delimiter(
        &self,
        _prefix: Option<&Path>,
    ) -> lance_io::storage::Result<ListResult> {
        unimplemented!("list_with_delimiter is not used by these tests")
    }

    async fn copy_opts(
        &self,
        _from: &Path,
        _to: &Path,
        _options: CopyOptions,
    ) -> lance_io::storage::Result<()> {
        unimplemented!("copy_opts is not used by these tests")
    }
    async fn get_ranges(
        &self,
        location: &lance_io::storage::Path,
        ranges: &[std::ops::Range<u64>],
    ) -> lance_io::storage::Result<Vec<bytes::Bytes>> {
        futures::future::try_join_all(ranges.iter().map(|range| {
            (self as &dyn lance_io::storage::StorageBackend).get_range(location, range.clone())
        }))
        .await
    }
    fn list_with_offset(
        &self,
        prefix: Option<&lance_io::storage::Path>,
        offset: &lance_io::storage::Path,
    ) -> futures::stream::BoxStream<'static, lance_io::storage::Result<lance_io::storage::ObjectMeta>>
    {
        let offset = offset.clone();
        Box::pin(futures::TryStreamExt::try_filter(
            self.list(prefix),
            move |meta| futures::future::ready(meta.location > offset),
        ))
    }
    async fn rename_opts(
        &self,
        from: &lance_io::storage::Path,
        to: &lance_io::storage::Path,
        options: lance_io::storage::RenameOptions,
    ) -> lance_io::storage::Result<()> {
        let mode = match options.target_mode {
            lance_io::storage::RenameTargetMode::Create => lance_io::storage::CopyMode::Create,
            lance_io::storage::RenameTargetMode::Overwrite => {
                lance_io::storage::CopyMode::Overwrite
            }
        };
        self.copy_opts(
            from,
            to,
            lance_io::storage::CopyOptions {
                mode,
                extensions: options.extensions,
            },
        )
        .await?;
        (self as &dyn lance_io::storage::StorageBackend)
            .delete(from)
            .await
    }
}

pub fn reject_empty_range_store() -> Arc<Storage> {
    Arc::new({
        let url = Url::parse("mock:///blob-tests").unwrap();
        let mut config = lance_io::storage::BackendConfig::new(
            Arc::new(RejectEmptyRangeObjectStore),
            url.scheme(),
        );
        config.io_parallelism = lance_io::storage::DEFAULT_LOCAL_IO_PARALLELISM;
        config.list_is_lexically_ordered = true;
        lance_io::utils::testing::TestStorageProvider(config).build(url, &Default::default())
    })
}

#[derive(Debug)]
pub struct RecordingRangeObjectStore {
    pub data: Bytes,
    pub gate: Option<Arc<Semaphore>>,
    pub head_requests: AtomicUsize,
    pub requested_ranges: std::sync::Mutex<Vec<Range<u64>>>,
    pub started_blob_requests: AtomicUsize,
    pub active_blob_requests: AtomicUsize,
    pub peak_active_blob_requests: AtomicUsize,
    pub request_started: Notify,
}

impl RecordingRangeObjectStore {
    pub fn new(data: Bytes) -> Self {
        Self {
            data,
            gate: None,
            head_requests: AtomicUsize::new(0),
            requested_ranges: std::sync::Mutex::new(Vec::new()),
            started_blob_requests: AtomicUsize::new(0),
            active_blob_requests: AtomicUsize::new(0),
            peak_active_blob_requests: AtomicUsize::new(0),
            request_started: Notify::new(),
        }
    }

    pub fn with_gate(data: Bytes, gate: Arc<Semaphore>) -> Self {
        Self {
            data,
            gate: Some(gate),
            head_requests: AtomicUsize::new(0),
            requested_ranges: std::sync::Mutex::new(Vec::new()),
            started_blob_requests: AtomicUsize::new(0),
            active_blob_requests: AtomicUsize::new(0),
            peak_active_blob_requests: AtomicUsize::new(0),
            request_started: Notify::new(),
        }
    }

    pub async fn wait_for_blob_requests(&self, expected: usize) {
        loop {
            let request_started = self.request_started.notified();
            if self.started_blob_requests.load(Ordering::Acquire) >= expected {
                return;
            }
            request_started.await;
        }
    }

    pub fn peak_active_blob_requests(&self) -> usize {
        self.peak_active_blob_requests.load(Ordering::Acquire)
    }

    pub fn requested_ranges(&self) -> Vec<Range<u64>> {
        self.requested_ranges.lock().unwrap().clone()
    }

    pub fn requested_blob_ranges(&self) -> Vec<Range<u64>> {
        let full_object = 0..self.data.len() as u64;
        self.requested_ranges()
            .into_iter()
            .filter(|range| range != &full_object)
            .collect()
    }

    pub fn object_meta(&self, location: &Path) -> ObjectMeta {
        ObjectMeta {
            location: location.clone(),
            last_modified: Utc::now(),
            size: self.data.len() as u64,
            e_tag: None,
            version: None,
        }
    }
}

impl std::fmt::Display for RecordingRangeObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RecordingRangeObjectStore")
    }
}

#[async_trait]
impl lance_io::storage::StorageBackend for RecordingRangeObjectStore {
    async fn list_paginated(
        &self,
        prefix: Option<&str>,
        options: lance_io::storage::PaginatedListOptions,
    ) -> lance_io::storage::Result<lance_io::storage::PaginatedListResult> {
        let _ = (prefix, options);
        Err(lance_io::storage::Error::NotSupported {
            source: "native pagination is unavailable".into(),
        })
    }

    async fn put_opts(
        &self,
        _location: &Path,
        _bytes: PutPayload,
        _opts: PutOptions,
    ) -> lance_io::storage::Result<PutResult> {
        unimplemented!("put_opts is not used by these tests")
    }

    async fn put_multipart_opts(
        &self,
        _location: &Path,
        _opts: PutMultipartOptions,
    ) -> lance_io::storage::Result<Box<dyn lance_io::storage::backend::MultipartUpload>> {
        unimplemented!("put_multipart_opts is not used by these tests")
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> lance_io::storage::Result<GetResult> {
        if options.head {
            self.head_requests.fetch_add(1, Ordering::Relaxed);
        }
        let range = options.range.unwrap_or(0..self.data.len() as u64);
        let is_full_object_probe = range.start == 0 && range.end == self.data.len() as u64;
        self.requested_ranges.lock().unwrap().push(range.clone());
        if !is_full_object_probe {
            let active = self.active_blob_requests.fetch_add(1, Ordering::AcqRel) + 1;
            self.peak_active_blob_requests
                .fetch_max(active, Ordering::AcqRel);
            self.started_blob_requests.fetch_add(1, Ordering::AcqRel);
            self.request_started.notify_waiters();
            if let Some(gate) = &self.gate {
                gate.acquire()
                    .await
                    .expect("test gate should remain open")
                    .forget();
            }
            self.active_blob_requests.fetch_sub(1, Ordering::AcqRel);
        }
        let bytes = self.data.slice(range.start as usize..range.end as usize);
        Ok(GetResult {
            payload: GetResultPayload::Stream(
                futures::stream::once(async move { Ok(bytes) }).boxed(),
            ),
            meta: self.object_meta(location),
            range,
            attributes: Attributes::default(),
            extensions: Default::default(),
        })
    }

    fn delete_stream(
        &self,
        _locations: futures::stream::BoxStream<'static, lance_io::storage::Result<Path>>,
    ) -> futures::stream::BoxStream<'static, lance_io::storage::Result<Path>> {
        unimplemented!("delete_stream is not used by these tests")
    }

    fn list(
        &self,
        _prefix: Option<&Path>,
    ) -> futures::stream::BoxStream<'static, lance_io::storage::Result<ObjectMeta>> {
        unimplemented!("list is not used by these tests")
    }

    async fn list_with_delimiter(
        &self,
        _prefix: Option<&Path>,
    ) -> lance_io::storage::Result<ListResult> {
        unimplemented!("list_with_delimiter is not used by these tests")
    }

    async fn copy_opts(
        &self,
        _from: &Path,
        _to: &Path,
        _options: CopyOptions,
    ) -> lance_io::storage::Result<()> {
        unimplemented!("copy_opts is not used by these tests")
    }
    async fn get_ranges(
        &self,
        location: &lance_io::storage::Path,
        ranges: &[std::ops::Range<u64>],
    ) -> lance_io::storage::Result<Vec<bytes::Bytes>> {
        futures::future::try_join_all(ranges.iter().map(|range| {
            (self as &dyn lance_io::storage::StorageBackend).get_range(location, range.clone())
        }))
        .await
    }
    fn list_with_offset(
        &self,
        prefix: Option<&lance_io::storage::Path>,
        offset: &lance_io::storage::Path,
    ) -> futures::stream::BoxStream<'static, lance_io::storage::Result<lance_io::storage::ObjectMeta>>
    {
        let offset = offset.clone();
        Box::pin(futures::TryStreamExt::try_filter(
            self.list(prefix),
            move |meta| futures::future::ready(meta.location > offset),
        ))
    }
    async fn rename_opts(
        &self,
        from: &lance_io::storage::Path,
        to: &lance_io::storage::Path,
        options: lance_io::storage::RenameOptions,
    ) -> lance_io::storage::Result<()> {
        let mode = match options.target_mode {
            lance_io::storage::RenameTargetMode::Create => lance_io::storage::CopyMode::Create,
            lance_io::storage::RenameTargetMode::Overwrite => {
                lance_io::storage::CopyMode::Overwrite
            }
        };
        self.copy_opts(
            from,
            to,
            lance_io::storage::CopyOptions {
                mode,
                extensions: options.extensions,
            },
        )
        .await?;
        (self as &dyn lance_io::storage::StorageBackend)
            .delete(from)
            .await
    }
}

pub fn recording_range_store_with_url_and_block_size(
    data: Bytes,
    url: &str,
    block_size: Option<usize>,
) -> (Arc<Storage>, Arc<RecordingRangeObjectStore>) {
    const TEST_RANGE_STORE_SIZE: usize = 128 * 1024;
    let mut padded = vec![0; TEST_RANGE_STORE_SIZE.max(data.len())];
    padded[..data.len()].copy_from_slice(data.as_ref());
    let inner = Arc::new(RecordingRangeObjectStore::new(Bytes::from(padded)));
    let store = Arc::new({
        let url = Url::parse(url).unwrap();
        let mut config = lance_io::storage::BackendConfig::new(inner.clone(), url.scheme());
        config.io_parallelism = lance_io::storage::DEFAULT_LOCAL_IO_PARALLELISM;
        config.list_is_lexically_ordered = true;
        lance_io::utils::testing::TestStorageProvider(config).build(
            url,
            &lance_io::storage::StorageParams {
                block_size,
                ..Default::default()
            },
        )
    });
    (store, inner)
}

pub fn recording_range_store_with_url(
    data: Bytes,
    url: &str,
) -> (Arc<Storage>, Arc<RecordingRangeObjectStore>) {
    recording_range_store_with_url_and_block_size(data, url, None)
}

pub fn recording_range_store(data: Bytes) -> (Arc<Storage>, Arc<RecordingRangeObjectStore>) {
    recording_range_store_with_url(data, "mock://recording/blob-range-tests")
}

pub fn gated_range_store(
    data: Bytes,
    url: &str,
) -> (Arc<Storage>, Arc<RecordingRangeObjectStore>, Arc<Semaphore>) {
    const TEST_RANGE_STORE_SIZE: usize = 128 * 1024;
    let mut padded = vec![0; TEST_RANGE_STORE_SIZE.max(data.len())];
    padded[..data.len()].copy_from_slice(data.as_ref());
    let gate = Arc::new(Semaphore::new(0));
    let inner = Arc::new(RecordingRangeObjectStore::with_gate(
        Bytes::from(padded),
        gate.clone(),
    ));
    let store = Arc::new({
        let url = Url::parse(url).unwrap();
        let mut config = lance_io::storage::BackendConfig::new(inner.clone(), url.scheme());
        config.io_parallelism = lance_io::storage::DEFAULT_LOCAL_IO_PARALLELISM;
        config.list_is_lexically_ordered = true;
        lance_io::utils::testing::TestStorageProvider(config).build(url, &Default::default())
    });
    (store, inner, gate)
}

pub async fn create_multi_base_blob_v2_fixture(
    payload: Vec<u8>,
    dedicated_threshold: Option<usize>,
    is_dataset_root: bool,
) -> MultiBaseBlobFixture {
    let test_dir = TempDir::default();
    let primary_uri = test_dir.path_str();
    let base_dir = test_dir.std_path().join("blob_base");
    std::fs::create_dir_all(&base_dir).unwrap();
    let base_uri = format!("file://{}", base_dir.display());

    let mut blob_builder = BlobArrayBuilder::new(2);
    blob_builder.push_bytes(payload.clone()).unwrap();
    blob_builder.push_bytes(payload.clone()).unwrap();
    let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

    let mut blob_column = blob_field("blob", true);
    if let Some(threshold) = dedicated_threshold {
        let mut metadata = blob_column.metadata().clone();
        metadata.insert(
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
            threshold.to_string(),
        );
        blob_column = blob_column.with_metadata(metadata);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        blob_column,
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from(vec![0, 1])), blob_array],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

    let dataset = Arc::new(
        Dataset::write(
            reader,
            &primary_uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                initial_bases: Some(vec![BasePath {
                    id: 1,
                    name: Some("blob_base".to_string()),
                    path: base_uri,
                    is_dataset_root,
                }]),
                target_bases: Some(vec![1]),
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    assert!(
        dataset
            .fragments()
            .iter()
            .all(|frag| frag.files.iter().all(|file| file.base_id == Some(1)))
    );
    assert_eq!(dataset.fragments().len(), 2);

    MultiBaseBlobFixture {
        _test_dir: test_dir,
        dataset,
        expected: payload,
    }
}
