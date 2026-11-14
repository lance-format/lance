// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::super::*;
use crate::blob::{BlobArrayBuilder, blob_field};
use crate::dataset::blob::external::{ExternalBaseCandidate, ExternalBaseResolver};
use crate::dataset::blob::read::{
    BlobEntry, ReadBlobsExecution, execute_blob_entries_with_execution,
};
use crate::dataset::blob::test_util::create_multi_base_blob_v2_fixture;
use crate::dataset::{DatasetBuilder, ReadParams, WriteParams};
use crate::session::Session;
use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::Schema;
use async_trait::async_trait;
use futures::stream::BoxStream;
use lance_core::utils::tempfile::TempStdDir;
use lance_file::version::LanceFileVersion;
use lance_io::object_store::{
    ChainedWrappingObjectStore, ObjectStoreParams, ObjectStoreProvider, ObjectStoreRegistry,
    WrappingObjectStore,
};
use lance_io::utils::tracking_store::IOTracker;
use object_store::list::PaginatedListStore;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStoreExt as _, PutMultipartOptions, PutOptions, PutPayload, PutResult, memory::InMemory,
};
use rstest::rstest;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use url::Url;

#[derive(Debug, Default)]
struct BlobProvider {
    origin: Arc<InMemory>,
    tracker: IOTracker,
    calls: AtomicUsize,
    params: Mutex<Vec<ObjectStoreParams>>,
}

#[async_trait]
impl ObjectStoreProvider for BlobProvider {
    async fn new_store(&self, _uri: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.params.lock().unwrap().push(params.clone());
        Ok(ObjectStore::new(
            self.tracker.wrap("", self.origin.clone()),
            Url::parse("memory://").unwrap(),
            params.block_size,
            None,
            false,
            true,
            4,
            0,
            params.storage_options(),
        ))
    }

    fn extract_path(&self, uri: &Url) -> Result<Path> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Path::from_url_path(uri.path())
            .map_err(|error| Error::invalid_input(format!("Invalid source path: {error}")))
    }

    fn calculate_object_store_prefix(
        &self,
        _uri: &Url,
        _options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok("probe".to_string())
    }
}

#[derive(Debug)]
struct ReadThroughCache {
    cache: Arc<InMemory>,
    original: Arc<dyn object_store::ObjectStore>,
}

impl std::fmt::Display for ReadThroughCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ReadThroughCache")
    }
}

#[async_trait]
impl object_store::ObjectStore for ReadThroughCache {
    async fn get_opts(&self, path: &Path, options: GetOptions) -> object_store::Result<GetResult> {
        if self.cache.head(path).await.is_err() {
            let bytes = self.original.get(path).await?.bytes().await?;
            self.cache.put(path, bytes.into()).await?;
        }
        self.cache.get_opts(path, options).await
    }

    async fn put_opts(
        &self,
        path: &Path,
        bytes: PutPayload,
        options: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.original.put_opts(path, bytes, options).await
    }

    async fn put_multipart_opts(
        &self,
        path: &Path,
        options: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.original.put_multipart_opts(path, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.original.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.original.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.original.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.original.copy_opts(from, to, options).await
    }
}

/// A read-through cache with the server page cache's immutable-object assumption.
/// Its state survives opening another wrapped store.
#[derive(Debug, Default)]
struct ImmutableCache {
    cache: Arc<InMemory>,
    external_wraps: AtomicUsize,
}

impl WrappingObjectStore for ImmutableCache {
    fn wrap(
        &self,
        _prefix: &str,
        original: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(ReadThroughCache {
            cache: self.cache.clone(),
            original,
        })
    }

    fn wrap_external_blob(
        &self,
        _prefix: &str,
        original: Arc<dyn object_store::ObjectStore>,
        _wrapped: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        self.external_wraps.fetch_add(1, Ordering::Relaxed);
        original
    }

    fn wrap_paginated(
        &self,
        _prefix: &str,
        _original: Arc<dyn PaginatedListStore>,
    ) -> Option<Arc<dyn PaginatedListStore>> {
        None
    }
}

#[rstest]
#[case::local_reject(false, false)]
#[case::local_opt_in(false, true)]
#[case::controlled_reject(true, false)]
#[case::controlled_overrides_local_opt_in(true, true)]
#[tokio::test]
async fn test_external_write_policy_and_error(
    #[case] is_controlled: bool,
    #[case] allow_outside: bool,
    #[values(false, true)] is_append: bool,
) {
    let dir = TempStdDir::default();
    let provider = Arc::new(BlobProvider::default());
    let registry = Arc::new(ObjectStoreRegistry::default());
    registry.insert("probe", provider.clone());
    let policy = if is_controlled {
        UnregisteredExternalBlobPolicy::Deny
    } else {
        UnregisteredExternalBlobPolicy::Allow
    };
    let session =
        Arc::new(Session::new(0, 0, registry).with_unregistered_external_blob_policy(policy));
    let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
    let mut blobs = BlobArrayBuilder::new(2);
    blobs.push_uri("probe://bucket/unregistered").unwrap();
    blobs.push_uri("probe://bucket/unregistered").unwrap();
    let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut params = WriteParams {
        session: Some(session.clone()),
        allow_external_blob_outside_bases: allow_outside,
        data_storage_version: Some(LanceFileVersion::V2_2),
        max_rows_per_file: 1,
        ..Default::default()
    };
    let result = if is_append {
        let mut initial = BlobArrayBuilder::new(2);
        initial.push_bytes(b"a").unwrap();
        initial.push_bytes(b"b").unwrap();
        let initial =
            RecordBatch::try_new(schema.clone(), vec![initial.finish().unwrap()]).unwrap();
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(initial)], schema),
            dir.to_str().unwrap(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let mut dataset = DatasetBuilder::from_uri(dir.to_str().unwrap())
            .with_session(session)
            .load()
            .await
            .unwrap();
        // Appends inherit the dataset's session policy even without a write override.
        params.session = None;
        dataset.append(reader, Some(params)).await.map(|_| dataset)
    } else {
        Dataset::write(reader, dir.to_str().unwrap(), Some(params)).await
    };
    if !is_controlled && allow_outside {
        let dataset = result.unwrap();
        assert_eq!(dataset.fragments().len(), if is_append { 4 } else { 2 });
        let session = dataset
            .session()
            .as_ref()
            .clone()
            .with_unregistered_external_blob_policy(UnregisteredExternalBlobPolicy::Deny);
        let guarded = Arc::new(
            DatasetBuilder::from_uri(dataset.uri())
                .with_session(Arc::new(session))
                .load()
                .await
                .unwrap(),
        );
        provider.calls.store(0, Ordering::Relaxed);
        let indices = if is_append { vec![2, 3] } else { vec![0, 1] };
        let error = guarded
            .take_blobs_by_indices(&indices, "blob")
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("base_id=0"), "{error}");
        let error = guarded
            .read_blobs("blob")
            .unwrap()
            .with_row_indices(indices)
            .execute()
            .await
            .unwrap_err();
        assert!(error.to_string().contains("base_id=0"), "{error}");
        assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    } else {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        if is_controlled {
            assert!(
                error.to_string().contains(
                    "Remote tables only support external blobs under registered storage bases"
                ),
                "{error}"
            );
            assert!(
                !error
                    .to_string()
                    .contains("allow_external_blob_outside_bases"),
                "{error}"
            );
        } else {
            assert!(
                error
                    .to_string()
                    .contains("Set allow_external_blob_outside_bases=true"),
                "{error}"
            );
        }
    }
    assert_eq!(provider.tracker.stats().read_iops, 0);
    assert!(provider.params.lock().unwrap().is_empty());
}

#[rstest]
#[case::file("file:///etc/passwd")]
#[case::object_store("file-object-store:///etc/passwd")]
#[case::uring("file+uring:///etc/passwd")]
#[case::bare("/etc/passwd")]
#[case::relative("relative/payload")]
#[tokio::test]
async fn test_external_write_denies_filtered_providers(
    #[case] uri: &str,
    #[values(false, true)] allow_outside: bool,
) {
    let registry = ObjectStoreRegistry::default();
    let source = Arc::new(BlobProvider::default());
    for scheme in ["file", "file-object-store", "file+uring"] {
        registry.insert(scheme, source.clone());
    }
    let registry = Arc::new(
        registry
            .without_providers(["file", "file-object-store", "file+uring"])
            .unwrap(),
    );
    let session = Arc::new(
        Session::new(0, 0, registry)
            .with_unregistered_external_blob_policy(UnregisteredExternalBlobPolicy::Deny),
    );
    let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
    let mut blobs = BlobArrayBuilder::new(1);
    blobs.push_uri(uri).unwrap();
    let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
    let error = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        "memory://table",
        Some(WriteParams {
            session: Some(session),
            allow_external_blob_outside_bases: allow_outside,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("Remote tables only support external blobs under registered storage bases"),
        "{error}"
    );
    assert!(
        !error
            .to_string()
            .contains("allow_external_blob_outside_bases"),
        "{error}"
    );
    assert_eq!(source.calls.load(Ordering::Relaxed), 0);
    assert!(source.params.lock().unwrap().is_empty());
    assert_eq!(source.tracker.stats().read_iops, 0);
}

#[tokio::test]
async fn test_explicit_ingest_keeps_its_source_access_under_deny() {
    let dir = TempStdDir::default();
    let source_path = dir.join("payload");
    std::fs::write(&source_path, b"ingested").unwrap();
    let source_uri = Url::from_file_path(&source_path).unwrap();
    let session = Arc::new(
        Session::default()
            .with_unregistered_external_blob_policy(UnregisteredExternalBlobPolicy::Deny),
    );
    let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
    let mut blobs = BlobArrayBuilder::new(1);
    blobs.push_uri(source_uri.as_str()).unwrap();
    let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://table",
            Some(WriteParams {
                session: Some(session),
                external_blob_mode: crate::dataset::write::ExternalBlobMode::Ingest,
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    assert_eq!(
        blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"ingested"
    );
}

#[rstest]
#[case::absolute("/tmp/external-base/payload")]
#[case::relative("external-base/payload")]
#[tokio::test]
async fn test_external_reference_matches_local_paths_when_provider_is_enabled(#[case] uri: &str) {
    let registry = Arc::new(ObjectStoreRegistry::default());
    let path = ObjectStore::extract_path_from_uri(registry.clone(), uri).unwrap();
    let base = path.parent().unwrap();
    let mut resolver = ExternalBaseResolver::new(
        vec![ExternalBaseCandidate {
            base_id: 2,
            store_prefix: "file".to_string(),
            base_path: base,
            store_params: ObjectStoreParams::default(),
        }],
        registry,
    );
    resolver.unregistered_external_blob_policy = UnregisteredExternalBlobPolicy::Deny;
    let matched = resolver.resolve_external_uri(uri).await.unwrap().unwrap();
    assert_eq!(matched.base_id, 2);
    assert_eq!(matched.relative_path, "payload");
}

#[tokio::test]
async fn test_external_reference_preserves_registered_matches_and_input_errors() {
    let registry = Arc::new(
        ObjectStoreRegistry::default()
            .without_providers(["file", "file-object-store", "file+uring"])
            .unwrap(),
    );
    let provider = Arc::new(BlobProvider::default());
    registry.insert("probe", provider.clone());
    let mut resolver = ExternalBaseResolver::new(
        vec![ExternalBaseCandidate {
            base_id: 2,
            store_prefix: "probe".to_string(),
            base_path: Path::from("base"),
            store_params: ObjectStoreParams::default(),
        }],
        registry,
    );
    resolver.unregistered_external_blob_policy = UnregisteredExternalBlobPolicy::Deny;
    let matched = resolver
        .resolve_external_uri("probe://bucket/base/payload")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(matched.base_id, 2);
    assert_eq!(matched.relative_path, "payload");
    let error = resolver
        .resolve_external_uri("probe://bucket/base/%FF")
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("Invalid source path"), "{error}");
    assert!(!error.to_string().contains("Remote tables"), "{error}");
    resolver.unregistered_external_blob_policy = UnregisteredExternalBlobPolicy::Allow;
    let error = resolver
        .resolve_external_uri("file:///etc/passwd")
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error.to_string().contains("Unknown scheme: file"),
        "{error}"
    );
    assert!(provider.params.lock().unwrap().is_empty());
    assert_eq!(provider.tracker.stats().read_iops, 0);
}

#[rstest]
#[case::unknown_size(0)]
#[case::known_size(3)]
#[tokio::test]
async fn test_unregistered_external_blob_policy(#[case] size: u64) {
    let fixture = create_multi_base_blob_v2_fixture(vec![1], None, false).await;
    let provider = Arc::new(BlobProvider::default());
    let registry = Arc::new(ObjectStoreRegistry::default());
    registry.insert("probe", provider.clone());
    let session = Session::new(0, 0, registry)
        .with_unregistered_external_blob_policy(UnregisteredExternalBlobPolicy::Deny);
    let dataset = DatasetBuilder::from_uri(fixture.dataset.uri())
        .with_read_params(ReadParams {
            session: Some(Arc::new(session)),
            ..Default::default()
        })
        .load()
        .await
        .unwrap();
    let field_id = dataset.schema().field("blob").unwrap().id as u32;
    let mut locator = BlobLocator::new(&dataset, field_id);
    for uri in [
        "probe://bucket/secret",
        "file:///etc/passwd",
        "s3://unregistered-bucket/secret",
        "invalid://[bad",
    ] {
        let err = locator
            .locate(
                &StoredBlob::External {
                    base_id: 0,
                    uri,
                    position: 0,
                    size,
                },
                0,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(err.to_string().contains("base_id=0"), "{err}");
        assert!(err.to_string().contains("registered storage base"), "{err}");
    }
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    assert_eq!(provider.tracker.stats().read_iops, 0);
    assert!(provider.params.lock().unwrap().is_empty());
}

#[rstest]
#[case::plain(false)]
#[case::wrapped(true)]
#[tokio::test]
async fn test_unregistered_external_blob_allowed_by_default(#[case] is_wrapped: bool) {
    let fixture = create_multi_base_blob_v2_fixture(vec![1], None, false).await;
    let provider = Arc::new(BlobProvider::default());
    provider
        .origin
        .put(&Path::from("payload"), b"old".as_slice().into())
        .await
        .unwrap();
    let registry = Arc::new(ObjectStoreRegistry::default());
    registry.insert("probe", provider.clone());
    let session = Arc::new(Session::new(0, 0, registry));
    assert_eq!(
        session.unregistered_external_blob_policy(),
        UnregisteredExternalBlobPolicy::Allow
    );
    let cache = Arc::new(ImmutableCache::default());
    let params = ObjectStoreParams {
        object_store_wrapper: is_wrapped.then(|| cache.clone() as Arc<dyn WrappingObjectStore>),
        ..Default::default()
    };
    let dataset = DatasetBuilder::from_uri(fixture.dataset.uri())
        .with_session(session)
        .with_store_params(params)
        .load()
        .await
        .unwrap();
    let field_id = dataset.schema().field("blob").unwrap().id as u32;
    let file = BlobLocator::new(&dataset, field_id)
        .locate(
            &StoredBlob::External {
                base_id: 0,
                uri: "probe://bucket/payload",
                position: 0,
                size: 0,
            },
            0,
        )
        .await
        .unwrap();
    assert_eq!(file.read().await.unwrap().as_ref(), b"old");
    assert!(provider.calls.load(Ordering::Relaxed) > 0);
    assert!(provider.tracker.stats().read_iops > 0);
    assert_eq!(cache.external_wraps.load(Ordering::Relaxed) > 0, is_wrapped);
}

#[rstest]
#[tokio::test]
async fn test_external_preserves_runtime_base_store(#[values(false, true)] is_wrapped: bool) {
    let fixture = create_multi_base_blob_v2_fixture(vec![1], None, false).await;
    fixture
        .dataset
        .add_bases(
            vec![lance_table::format::BasePath::new(
                2,
                "memory://".to_string(),
                None,
                false,
            )],
            None,
        )
        .await
        .unwrap();
    let dataset = DatasetBuilder::from_uri(fixture.dataset.uri())
        .with_store_params(ObjectStoreParams {
            object_store_wrapper: is_wrapped
                .then(|| Arc::new(IOTracker::default()) as Arc<dyn WrappingObjectStore>),
            ..Default::default()
        })
        .load()
        .await
        .unwrap();
    let store = dataset.object_store(Some(2)).await.unwrap();
    let root = dataset.blob_base_path(Some(2)).unwrap();
    store
        .put(&root.clone().join("payload"), b"old")
        .await
        .unwrap();
    let field_id = dataset.schema().field("blob").unwrap().id as u32;
    let file = BlobLocator::new(&dataset, field_id)
        .locate(
            &StoredBlob::External {
                base_id: 2,
                uri: "payload",
                position: 0,
                size: 3,
            },
            0,
        )
        .await
        .unwrap();
    if !is_wrapped {
        assert!(Arc::ptr_eq(&file.source.object_store, &store));
    }
    assert_eq!(file.read().await.unwrap().as_ref(), b"old");
    store.put(&root.join("payload"), b"new").await.unwrap();
    assert_eq!(file.read_range(0..3).await.unwrap().as_ref(), b"new");
}

#[rstest]
#[case::exact_binding(true, true)]
#[case::scoped_options(true, false)]
#[case::absolute_uri(false, false)]
#[tokio::test]
async fn test_external_bypasses_cache_and_managed_keeps_cache(
    #[case] is_registered: bool,
    #[case] exact_binding: bool,
) {
    let fixture = create_multi_base_blob_v2_fixture(vec![1], None, false).await;
    let base =
        lance_table::format::BasePath::new(2, "probe://bucket/base".to_string(), None, false);
    fixture
        .dataset
        .add_bases(vec![base.clone()], None)
        .await
        .unwrap();
    let provider = Arc::new(BlobProvider::default());
    let payload_path = Path::from("base/payload");
    provider
        .origin
        .put(&payload_path, b"old".as_slice().into())
        .await
        .unwrap();
    let registry = Arc::new(ObjectStoreRegistry::default());
    registry.insert("probe", provider.clone());
    let session = Arc::new(
        Session::new(0, 0, registry).with_unregistered_external_blob_policy(if is_registered {
            UnregisteredExternalBlobPolicy::Deny
        } else {
            UnregisteredExternalBlobPolicy::Allow
        }),
    );
    let cache = Arc::new(ImmutableCache::default());
    let observer = Arc::new(IOTracker::default());
    let wrapper = Arc::new(ChainedWrappingObjectStore::new(vec![
        cache.clone(),
        observer.clone(),
    ]));
    let options = HashMap::from([
        ("credential".to_string(), "base-secret".to_string()),
        ("endpoint".to_string(), "base-endpoint".to_string()),
    ]);
    let params = ObjectStoreParams {
        object_store_wrapper: Some(wrapper.clone()),
        block_size: Some(4096),
        ..Default::default()
    };
    let builder = DatasetBuilder::from_uri(fixture.dataset.uri()).with_session(session);
    let dataset = if !is_registered {
        builder
            .with_store_params(params)
            .with_storage_options(options.clone())
            .load()
            .await
            .unwrap()
    } else if exact_binding {
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(
                lance_io::object_store::StorageOptionsAccessor::with_static_options(
                    options.clone(),
                ),
            )),
            ..params
        };
        builder
            .with_base_store_params(&base.path, params)
            .load()
            .await
            .unwrap()
    } else {
        builder
            .with_store_params(params)
            .with_storage_options(
                options
                    .iter()
                    .map(|(k, v)| (format!("base_2.{k}"), v.clone()))
                    .collect(),
            )
            .load()
            .await
            .unwrap()
    };
    let field_id = dataset.schema().field("blob").unwrap().id as u32;
    let external_uri = if is_registered {
        "payload"
    } else {
        "probe://bucket/base/payload"
    };
    let external = StoredBlob::External {
        base_id: if is_registered { 2 } else { 0 },
        uri: external_uri,
        position: 0,
        size: 3,
    };
    let managed = StoredBlob::Managed {
        base_id: Some(2),
        uri: "payload",
        position: 0,
        size: 3,
    };
    let mut locator = BlobLocator::new(&dataset, field_id);
    let managed_file = locator.locate(&managed, 0).await.unwrap();
    assert_eq!(managed_file.read().await.unwrap().as_ref(), b"old");
    assert!(provider.tracker.incremental_stats().read_iops > 0);
    assert_eq!(
        managed_file.read_range(0..3).await.unwrap().as_ref(),
        b"old"
    );
    assert_eq!(provider.tracker.incremental_stats().read_iops, 0);

    let external_file = locator.locate(&external, 0).await.unwrap();
    assert!(!Arc::ptr_eq(&managed_file.source, &external_file.source));
    observer.incremental_stats();
    assert_eq!(external_file.read().await.unwrap().as_ref(), b"old");
    assert!(observer.incremental_stats().read_iops > 0);
    assert!(provider.tracker.incremental_stats().read_iops > 0);
    provider
        .origin
        .put(&payload_path, b"new".as_slice().into())
        .await
        .unwrap();
    let mut next_locator = BlobLocator::new(&dataset, field_id);
    let external_file = next_locator.locate(&external, 0).await.unwrap();
    assert_eq!(external_file.read().await.unwrap().as_ref(), b"new");
    assert!(provider.tracker.incremental_stats().read_iops > 0);
    assert_eq!(
        next_locator
            .locate(&managed, 0)
            .await
            .unwrap()
            .read()
            .await
            .unwrap()
            .as_ref(),
        b"old"
    );
    assert_eq!(provider.tracker.incremental_stats().read_iops, 0);

    // Wrapped registered bases intentionally have a dataset-scoped binding;
    // absolute URIs use a separate registry binding. Neither may reopen per batch.
    let expected_bindings = if is_registered { 1 } else { 2 };
    assert_eq!(provider.params.lock().unwrap().len(), expected_bindings);
    // A scan shares this execution across batches, but creates a new locator per batch.
    let execution = Arc::new(ReadBlobsExecution::new(None));
    for batch in 0..4 {
        let payload: &[u8] = if batch % 2 == 0 { b"new" } else { b"now" };
        provider
            .origin
            .put(&payload_path, payload.into())
            .await
            .unwrap();
        let mut locator = BlobLocator::new(&dataset, field_id);
        let mut entries: Vec<BlobEntry> = Vec::new();
        for (selection_index, (position, size)) in [(0, 2), (1, 2), (0, 2)].into_iter().enumerate()
        {
            let file = locator
                .locate(
                    &StoredBlob::External {
                        base_id: if is_registered { 2 } else { 0 },
                        uri: external_uri,
                        position,
                        size,
                    },
                    0,
                )
                .await
                .unwrap();
            if let Some(first) = entries.first() {
                assert!(Arc::ptr_eq(&first.file.source, &file.source));
            }
            entries.push(BlobEntry {
                selection_index,
                file,
                requested_range: None,
            });
        }
        provider.tracker.incremental_stats();
        let mut blobs = execute_blob_entries_with_execution(entries, 2, execution.clone())
            .await
            .unwrap();
        blobs.sort_by_key(|blob| blob.selection_index);
        assert_eq!(blobs[0].data.as_ref(), &payload[..2]);
        assert_eq!(blobs[1].data.as_ref(), &payload[1..]);
        assert_eq!(blobs[2].data.as_ref(), &payload[..2]);
        let stats = provider.tracker.incremental_stats();
        assert_eq!(stats.read_iops, 1, "{stats:?}");
        assert_eq!(stats.requests.len(), 1);
        assert_eq!(stats.requests[0].range, Some(0..3));

        let managed_file = locator.locate(&managed, 0).await.unwrap();
        let blobs = execute_blob_entries_with_execution(
            vec![BlobEntry {
                selection_index: 0,
                file: managed_file,
                requested_range: None,
            }],
            2,
            execution.clone(),
        )
        .await
        .unwrap();
        assert_eq!(blobs[0].data.as_ref(), b"old");
        assert_eq!(provider.tracker.incremental_stats().read_iops, 0);
        assert_eq!(execution.num_schedulers(), 2);
        assert_eq!(provider.params.lock().unwrap().len(), expected_bindings);
    }
    assert!(observer.stats().read_iops > 0);
    assert!(cache.external_wraps.load(Ordering::Relaxed) > 0);
    assert_eq!(provider.params.lock().unwrap().len(), expected_bindings);
    for params in provider.params.lock().unwrap().iter() {
        assert_eq!(params.storage_options().unwrap(), &options);
        assert_eq!(params.block_size, Some(4096));
    }
}
