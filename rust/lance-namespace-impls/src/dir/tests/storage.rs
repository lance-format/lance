// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

/// The early-stop path (ordered stores) and the collect-then-sort path
/// must return the same results for every descending/limit combination.
#[tokio::test]
async fn test_list_versions_under_ordering_and_limit() {
    use lance_table::io::commit::ManifestNamingScheme;

    async fn seed_and_check(ns: &DirectoryNamespace) {
        let table_path = ns.base_path.clone().join("lv_test.lance");
        for v in 1..=7u64 {
            let p = ManifestNamingScheme::V2.manifest_path(&table_path, v);
            ns.object_store.put(&p, b"m".as_slice()).await.unwrap();
        }
        // A retained staging blob (sorts ahead of every committed
        // manifest) and a detached manifest (sorts after) must be excluded
        // without breaking naming-scheme detection.
        let staging = Path::parse(format!(
            "{}-cee4fbbb-eb19-4ea3-8ca7-54f5ec33dedc",
            ManifestNamingScheme::V2.manifest_path(&table_path, 8)
        ))
        .unwrap();
        ns.object_store
            .put(&staging, b"s".as_slice())
            .await
            .unwrap();
        let detached = table_path.clone().join(VERSIONS_DIR).join("d123.manifest");
        ns.object_store
            .put(&detached, b"d".as_slice())
            .await
            .unwrap();
        fn versions(r: &[TableVersion]) -> Vec<i64> {
            r.iter().map(|t| t.version).collect()
        }

        let got = ns
            .list_versions_under(&table_path, true, Some(1))
            .await
            .unwrap();
        assert_eq!(versions(&got), vec![7]);
        let got = ns
            .list_versions_under(&table_path, true, Some(3))
            .await
            .unwrap();
        assert_eq!(versions(&got), vec![7, 6, 5]);

        let got = ns
            .list_versions_under(&table_path, false, Some(2))
            .await
            .unwrap();
        assert_eq!(versions(&got), vec![1, 2]);

        let got = ns
            .list_versions_under(&table_path, true, None)
            .await
            .unwrap();
        assert_eq!(versions(&got), vec![7, 6, 5, 4, 3, 2, 1]);
        let got = ns
            .list_versions_under(&table_path, false, None)
            .await
            .unwrap();
        assert_eq!(versions(&got), vec![1, 2, 3, 4, 5, 6, 7]);

        let got = ns
            .list_versions_under(&table_path, true, Some(0))
            .await
            .unwrap();
        assert!(got.is_empty());
        let got = ns
            .list_versions_under(&table_path, true, Some(100))
            .await
            .unwrap();
        assert_eq!(versions(&got), vec![7, 6, 5, 4, 3, 2, 1]);

        // Negative limits are ignored, matching `apply_pagination`.
        let got = ns
            .list_versions_under(&table_path, true, Some(-1))
            .await
            .unwrap();
        assert_eq!(versions(&got), vec![7, 6, 5, 4, 3, 2, 1]);
    }

    let ns_mem = DirectoryNamespaceBuilder::new("memory://lv-test")
        .build()
        .await
        .unwrap();
    assert!(ns_mem.object_store.list_is_lexically_ordered);
    seed_and_check(&ns_mem).await;

    let (ns_fs, _tmp) = create_test_namespace().await;
    assert!(!ns_fs.object_store.list_is_lexically_ordered);
    seed_and_check(&ns_fs).await;
}

/// A retained staging blob sorts ahead of the newest committed manifest;
/// if scheme detection reads it, the `descending, limit=1` hot path falls
/// back to consuming the whole directory. Asserts the consumption bound.
#[tokio::test]
async fn test_list_versions_under_early_stop_bounded_consumption() {
    use lance_io::object_store::providers::memory::MemoryStoreProvider;
    use lance_table::io::commit::ManifestNamingScheme;

    #[derive(Debug)]
    struct EntryCountingStore {
        target: Arc<dyn OSObjectStore>,
        entries_listed: Arc<AtomicUsize>,
    }

    impl std::fmt::Display for EntryCountingStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "EntryCountingStore({})", self.target)
        }
    }

    #[async_trait]
    impl OSObjectStore for EntryCountingStore {
        async fn put_opts(
            &self,
            location: &Path,
            bytes: PutPayload,
            opts: PutOptions,
        ) -> OSResult<PutResult> {
            self.target.put_opts(location, bytes, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            self.target.put_multipart_opts(location, opts).await
        }

        async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
            self.target.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            self.target.delete_stream(locations)
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            let entries_listed = self.entries_listed.clone();
            self.target
                .list(prefix)
                .inspect(move |_| {
                    entries_listed.fetch_add(1, Ordering::SeqCst);
                })
                .boxed()
        }

        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
            self.target.list_with_delimiter(prefix).await
        }

        async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
            self.target.copy_opts(from, to, opts).await
        }
    }

    #[derive(Debug)]
    struct EntryCountingMemoryProvider {
        entries_listed: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl lance_io::object_store::ObjectStoreProvider for EntryCountingMemoryProvider {
        async fn new_store(
            &self,
            base_path: Url,
            params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            let mut store = MemoryStoreProvider.new_store(base_path, params).await?;
            store.inner = Arc::new(EntryCountingStore {
                target: store.inner.clone(),
                entries_listed: self.entries_listed.clone(),
            });
            Ok(store)
        }

        fn extract_path(&self, url: &Url) -> Result<Path> {
            MemoryStoreProvider.extract_path(url)
        }

        fn calculate_object_store_prefix(
            &self,
            url: &Url,
            storage_options: Option<&HashMap<String, String>>,
        ) -> Result<String> {
            MemoryStoreProvider.calculate_object_store_prefix(url, storage_options)
        }
    }

    let entries_listed = Arc::new(AtomicUsize::new(0));
    let registry = Arc::new(ObjectStoreRegistry::default());
    registry.insert(
        "memory-object-store",
        Arc::new(EntryCountingMemoryProvider {
            entries_listed: entries_listed.clone(),
        }),
    );
    let session = Arc::new(Session::new(0, 0, registry));
    let ns = DirectoryNamespaceBuilder::new("memory-object-store://lv-count")
        .session(session)
        .build()
        .await
        .unwrap();
    assert!(ns.object_store.list_is_lexically_ordered);

    let table_path = ns.base_path.clone().join("lv_count.lance");
    for v in 1..=100u64 {
        let p = ManifestNamingScheme::V2.manifest_path(&table_path, v);
        ns.object_store.put(&p, b"m".as_slice()).await.unwrap();
    }
    // Sorts ahead of every committed manifest: the first raw entry.
    let staging = Path::parse(format!(
        "{}-cee4fbbb-eb19-4ea3-8ca7-54f5ec33dedc",
        ManifestNamingScheme::V2.manifest_path(&table_path, 101)
    ))
    .unwrap();
    ns.object_store
        .put(&staging, b"s".as_slice())
        .await
        .unwrap();

    let consumed_before = entries_listed.load(Ordering::SeqCst);
    let got = ns
        .list_versions_under(&table_path, true, Some(1))
        .await
        .unwrap();
    let consumed = entries_listed.load(Ordering::SeqCst) - consumed_before;

    assert_eq!(got.len(), 1);
    assert_eq!(got[0].version, 100);
    assert_eq!(
        consumed, 2,
        "latest-version query must consume only the staging entry plus the \
             first committed manifest, not the whole directory (consumed {} of \
             101 entries)",
        consumed
    );
}

#[derive(Debug)]
#[allow(dead_code)]
struct CountingFileStoreProvider {
    listing_count: Arc<AtomicUsize>,
}

#[async_trait]
impl lance_io::object_store::ObjectStoreProvider for CountingFileStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let provider = FileStoreProvider;
        let mut store = provider.new_store(base_path, params).await?;
        store.inner = Arc::new(CountingObjectStore::new(
            store.inner.clone(),
            self.listing_count.clone(),
        ));
        Ok(store)
    }

    fn extract_path(&self, url: &Url) -> Result<Path> {
        let provider = FileStoreProvider;
        provider.extract_path(url)
    }

    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        let provider = FileStoreProvider;
        provider.calculate_object_store_prefix(url, storage_options)
    }
}

#[allow(dead_code)]
fn file_object_store_uri(path: &str) -> String {
    let file_url = uri_to_url(path).unwrap();
    let mut url = Url::parse("file-object-store:///").unwrap();
    url.set_path(file_url.path());
    url.to_string()
}

#[allow(dead_code)]
fn build_listing_counting_session(listing_count: Arc<AtomicUsize>) -> Arc<Session> {
    let registry = Arc::new(ObjectStoreRegistry::default());
    registry.insert(
        "file-object-store",
        Arc::new(CountingFileStoreProvider { listing_count }),
    );
    Arc::new(Session::new(0, 0, registry))
}

// Fault-injection store: returns a runtime-toggleable result from
// `list_with_delimiter` (the call `check_table_status` makes) and delegates
// everything else, so a table can be created before failures are injected.
use futures::stream::BoxStream;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    PutMultipartOptions, PutPayload, PutResult, Result as OSResult,
};
use std::ops::Range;

#[derive(Debug, Clone, Copy)]
enum ListBehavior {
    Throttle,
    ServiceUnavailable,
    Internal,
    NotFound,
    EmptyListing,
}

#[derive(Debug)]
struct FailingListStore {
    target: Arc<dyn OSObjectStore>,
    behavior: Arc<Mutex<Option<ListBehavior>>>,
}

impl std::fmt::Display for FailingListStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FailingListStore({})", self.target)
    }
}

#[async_trait]
impl OSObjectStore for FailingListStore {
    async fn put_opts(
        &self,
        location: &Path,
        bytes: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.target.put_opts(location, bytes, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        self.target.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.target.get_opts(location, options).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        self.target.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OSResult<Path>>,
    ) -> BoxStream<'static, OSResult<Path>> {
        self.target.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.target.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        let behavior = *self.behavior.lock().unwrap();
        match behavior {
            None => self.target.list_with_delimiter(prefix).await,
            Some(ListBehavior::EmptyListing) => Ok(ListResult {
                common_prefixes: Vec::new(),
                objects: Vec::new(),
                extensions: Default::default(),
            }),
            // Mirrors the object_store retry-exhaustion message shape for an
            // Azure ServerBusy response, which is what the incident produced.
            Some(ListBehavior::Throttle) => Err(ObjectStoreError::Generic {
                store: "test",
                source: "Error performing list request: response error, after 3 retries, \
                             max_retries: 3, retry_timeout: 180s - HTTP status server error \
                             (503 Service Unavailable): ServerBusy: The server is busy"
                    .into(),
            }),
            Some(ListBehavior::ServiceUnavailable) => Err(ObjectStoreError::Generic {
                store: "test",
                source: "Error performing list request: 503 Service Unavailable".into(),
            }),
            Some(ListBehavior::Internal) => Err(ObjectStoreError::Generic {
                store: "test",
                source: "Error performing list request: catastrophic unclassified failure".into(),
            }),
            Some(ListBehavior::NotFound) => Err(ObjectStoreError::NotFound {
                path: "test_table.lance".to_string(),
                source: "entity not found".into(),
            }),
        }
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
        self.target.copy_opts(from, to, opts).await
    }
}

#[derive(Debug)]
struct FailingListStoreProvider {
    behavior: Arc<Mutex<Option<ListBehavior>>>,
}

#[async_trait]
impl lance_io::object_store::ObjectStoreProvider for FailingListStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let mut store = FileStoreProvider.new_store(base_path, params).await?;
        store.inner = Arc::new(FailingListStore {
            target: store.inner.clone(),
            behavior: self.behavior.clone(),
        });
        Ok(store)
    }

    fn extract_path(&self, url: &Url) -> Result<Path> {
        FileStoreProvider.extract_path(url)
    }

    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        FileStoreProvider.calculate_object_store_prefix(url, storage_options)
    }
}

fn build_failing_list_session(behavior: Arc<Mutex<Option<ListBehavior>>>) -> Arc<Session> {
    let registry = Arc::new(ObjectStoreRegistry::default());
    registry.insert(
        "file-object-store",
        Arc::new(FailingListStoreProvider { behavior }),
    );
    Arc::new(Session::new(0, 0, registry))
}

/// Build a dir-listing namespace whose object store's listing calls follow a
/// shared, runtime-toggleable behavior. Returns the namespace, the temp dir
/// (kept alive for the store), and the behavior toggle.
async fn failing_list_namespace() -> (
    DirectoryNamespace,
    TempStdDir,
    Arc<Mutex<Option<ListBehavior>>>,
) {
    let temp_dir = TempStdDir::default();
    let root_uri = file_object_store_uri(temp_dir.to_str().unwrap());
    let behavior = Arc::new(Mutex::new(None));
    let session = build_failing_list_session(behavior.clone());
    let namespace = DirectoryNamespaceBuilder::new(root_uri)
        .session(session)
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();
    (namespace, temp_dir, behavior)
}

async fn create_named_dir_table(namespace: &DirectoryNamespace, name: &str) {
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec![name.to_string()]);
    namespace
        .create_table(create_req, Bytes::from(ipc_data))
        .await
        .unwrap();
}

/// Regression test for the throttling-induced TableNotFound bug: a storage
/// error while resolving a table must surface as a typed storage error
/// (Throttling / ServiceUnavailable / Internal) carrying the underlying
/// evidence in its message — never as TableNotFound.
#[tokio::test]
async fn test_table_resolution_propagates_storage_errors_not_table_not_found() {
    for (behavior, expected_code, evidence) in [
        (ListBehavior::Throttle, ErrorCode::Throttling, "serverbusy"),
        (
            ListBehavior::ServiceUnavailable,
            ErrorCode::ServiceUnavailable,
            "503 service unavailable",
        ),
        (ListBehavior::Internal, ErrorCode::Internal, "catastrophic"),
    ] {
        let (namespace, _temp_dir, toggle) = failing_list_namespace().await;
        create_named_dir_table(&namespace, "checkpoint").await;
        *toggle.lock().unwrap() = Some(behavior);

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["checkpoint".to_string()]);
        let err = namespace.describe_table(describe_req).await.unwrap_err();
        let msg = err.to_string();
        assert_eq!(
            mutation_error_code(err),
            expected_code,
            "describe_table under {behavior:?}; msg: {msg}"
        );
        assert!(
            msg.to_ascii_lowercase().contains(evidence),
            "describe_table message must carry storage evidence '{evidence}', got: {msg}"
        );

        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["checkpoint".to_string()]);
        let err = namespace.table_exists(exists_req).await.unwrap_err();
        let msg = err.to_string();
        assert_eq!(
            mutation_error_code(err),
            expected_code,
            "table_exists under {behavior:?}; msg: {msg}"
        );
        assert!(
            msg.to_ascii_lowercase().contains(evidence),
            "table_exists message must carry storage evidence '{evidence}', got: {msg}"
        );
    }
}

/// A genuine not-found error and an empty listing must both still resolve to
/// TableNotFound (the local-FS and object-store representations of "missing").
#[tokio::test]
async fn test_table_resolution_missing_table_yields_table_not_found() {
    for behavior in [ListBehavior::NotFound, ListBehavior::EmptyListing] {
        let (namespace, _temp_dir, toggle) = failing_list_namespace().await;
        *toggle.lock().unwrap() = Some(behavior);

        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["missing".to_string()]);
        let err = namespace.describe_table(describe_req).await.unwrap_err();
        assert_eq!(
            mutation_error_code(err),
            ErrorCode::TableNotFound,
            "describe_table under {behavior:?} should be TableNotFound"
        );

        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["missing".to_string()]);
        let err = namespace.table_exists(exists_req).await.unwrap_err();
        assert_eq!(
            mutation_error_code(err),
            ErrorCode::TableNotFound,
            "table_exists under {behavior:?} should be TableNotFound"
        );
    }
}

/// Hybrid (manifest + directory) resolution must exercise the
/// manifest→directory fall-through for a table that exists on disk but is not
/// registered in the manifest: the fall-through must succeed normally, and
/// must not degrade a storage error into TableNotFound.
///
/// A `__manifest` table must actually exist for the manifest branch to run;
/// otherwise `manifest_ns_for_read()` is None and the manifest branch (and its
/// fall-through arm) is skipped entirely. We therefore create a *separate*
/// table through a manifest-enabled namespace first so `__manifest` exists.
#[tokio::test]
async fn test_hybrid_resolution_falls_through_and_does_not_mask_throttle() {
    let temp_dir = TempStdDir::default();
    let root_uri = file_object_store_uri(temp_dir.to_str().unwrap());
    let behavior = Arc::new(Mutex::new(None));
    let session = build_failing_list_session(behavior.clone());

    // Seed table via a manifest-enabled namespace so `__manifest` exists.
    let manifest_ns = DirectoryNamespaceBuilder::new(root_uri.clone())
        .session(session.clone())
        .manifest_enabled(true)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();
    create_named_dir_table(&manifest_ns, "seed").await;

    // The table under test: on disk but never registered in the manifest.
    let dir_ns = DirectoryNamespaceBuilder::new(root_uri.clone())
        .session(session.clone())
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();
    create_named_dir_table(&dir_ns, "checkpoint").await;

    // Migration enabled so root-level reads consult the manifest and fall
    // through to the directory check on a manifest miss.
    let hybrid_ns = DirectoryNamespaceBuilder::new(root_uri)
        .session(session)
        .manifest_enabled(true)
        .dir_listing_enabled(true)
        .dir_listing_to_manifest_migration_enabled(true)
        .build()
        .await
        .unwrap();

    // (a) Healthy fall-through: the manifest reports "checkpoint" absent and it
    // resolves via the directory listing (guards the migration lookup).
    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["checkpoint".to_string()]);
    hybrid_ns
        .describe_table(describe_req)
        .await
        .expect("unregistered on-disk table should resolve via the manifest fall-through");

    // (b) The throttle here surfaces from the directory check after the manifest
    // reports absent; the fall-through arm's own storage-error guard is covered
    // by the classify_storage_error / is_manifest_table_absent_error unit tests.
    *behavior.lock().unwrap() = Some(ListBehavior::Throttle);
    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["checkpoint".to_string()]);
    let err = hybrid_ns.describe_table(describe_req).await.unwrap_err();
    let code = mutation_error_code(err);
    assert_ne!(
        code,
        ErrorCode::TableNotFound,
        "hybrid resolution masked a throttle as TableNotFound"
    );
    assert!(
        matches!(
            code,
            ErrorCode::Throttling | ErrorCode::ServiceUnavailable | ErrorCode::Internal
        ),
        "hybrid resolution should surface a storage error, got {code:?}"
    );
}

#[tokio::test]
async fn test_dir_listing_no_extra_calls_without_migration() {
    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();
    let root_uri = file_object_store_uri(temp_path);
    let listing_count = Arc::new(AtomicUsize::new(0));
    let session = build_listing_counting_session(listing_count.clone());

    // Create a table using dir-listing-only namespace
    let dir_only_ns = DirectoryNamespaceBuilder::new(root_uri.clone())
        .session(session.clone())
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    dir_only_ns
        .create_table(create_req, Bytes::from(ipc_data))
        .await
        .unwrap();

    // Build a namespace with both enabled but migration disabled (default)
    let hybrid_ns = DirectoryNamespaceBuilder::new(root_uri)
        .session(session)
        .manifest_enabled(true)
        .dir_listing_enabled(true)
        .dir_listing_to_manifest_migration_enabled(false)
        .build()
        .await
        .unwrap();

    // Reset counter before the operation we want to measure
    listing_count.store(0, Ordering::SeqCst);

    // table_exists should use dir listing directly, making only 1 listing call
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["test_table".to_string()]);
    hybrid_ns.table_exists(exists_req).await.unwrap();

    let count = listing_count.load(Ordering::SeqCst);
    assert_eq!(
        count, 1,
        "Expected exactly 1 listing call for table_exists \
             without migration mode, but got {}",
        count
    );

    // Reset and test describe_table
    listing_count.store(0, Ordering::SeqCst);

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    hybrid_ns.describe_table(describe_req).await.unwrap();

    let count = listing_count.load(Ordering::SeqCst);
    assert_eq!(
        count, 1,
        "Expected exactly 1 listing call for describe_table \
             without migration mode, but got {}",
        count
    );
}

#[tokio::test]
async fn test_build_and_root_reads_do_not_create_manifest() {
    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();
    let manifest_path = std::path::Path::new(temp_path).join("__manifest");

    let dir_only_ns = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();
    create_scalar_table(&dir_only_ns, "catalog").await;
    assert!(!manifest_path.exists());

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();
    assert!(!manifest_path.exists());

    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["catalog".to_string()]);
    namespace.table_exists(exists_req).await.unwrap();
    assert!(!manifest_path.exists());

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["catalog".to_string()]);
    namespace.describe_table(describe_req).await.unwrap();
    assert!(!manifest_path.exists());

    let list_response = namespace
        .list_tables(ListTablesRequest {
            id: Some(vec![]),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(list_response.tables, vec!["catalog".to_string()]);
    assert!(!manifest_path.exists());

    let mut list_namespaces_req = ListNamespacesRequest::new();
    list_namespaces_req.id = Some(vec!["workspace".to_string()]);
    let err = namespace
        .list_namespaces(list_namespaces_req)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("__manifest"));
    assert!(!manifest_path.exists());

    let err = namespace
        .list_tables(ListTablesRequest {
            id: Some(vec!["workspace".to_string()]),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(err.to_string().contains("__manifest"));
    assert!(!manifest_path.exists());

    let mut child_describe_req = DescribeTableRequest::new();
    child_describe_req.id = Some(vec!["workspace".to_string(), "catalog".to_string()]);
    let err = namespace
        .describe_table(child_describe_req)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("__manifest"));
    assert!(!manifest_path.exists());

    let mut child_exists_req = TableExistsRequest::new();
    child_exists_req.id = Some(vec!["workspace".to_string(), "catalog".to_string()]);
    let err = namespace.table_exists(child_exists_req).await.unwrap_err();
    assert!(err.to_string().contains("__manifest"));
    assert!(!manifest_path.exists());

    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["workspace".to_string()]);
    namespace.create_namespace(create_ns_req).await.unwrap();
    assert!(manifest_path.exists());
}

#[tokio::test]
async fn test_migrate_updates_read_opened_legacy_manifest() {
    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();
    create_legacy_manifest_without_primary_key_metadata(temp_path).await;
    assert!(!manifest_has_primary_key_metadata(temp_path).await);

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();
    assert!(!manifest_has_primary_key_metadata(temp_path).await);

    let migrated = namespace.migrate().await.unwrap();
    assert_eq!(migrated, 0);
    assert!(manifest_has_primary_key_metadata(temp_path).await);
}

#[tokio::test]
async fn test_describe_declared_table_checks_versions_only_when_requested() {
    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();
    let root_uri = file_object_store_uri(temp_path);
    let listing_count = Arc::new(AtomicUsize::new(0));
    let session = build_listing_counting_session(listing_count.clone());

    let namespace = DirectoryNamespaceBuilder::new(root_uri)
        .session(session)
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    namespace.declare_table(declare_req).await.unwrap();

    listing_count.store(0, Ordering::SeqCst);

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();

    assert_eq!(describe_response.is_only_declared, None);
    assert_eq!(
        listing_count.load(Ordering::SeqCst),
        1,
        "Default describe_table should only list the table directory"
    );

    listing_count.store(0, Ordering::SeqCst);

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    describe_req.check_declared = Some(true);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();

    assert_eq!(describe_response.is_only_declared, Some(true));
    assert_eq!(
        listing_count.load(Ordering::SeqCst),
        2,
        "check_declared describe_table should list the table directory and _versions"
    );
}

#[tokio::test]
async fn test_dir_listing_extra_calls_with_migration() {
    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();
    let root_uri = file_object_store_uri(temp_path);
    let listing_count = Arc::new(AtomicUsize::new(0));
    let session = build_listing_counting_session(listing_count.clone());

    // Create a table using dir-listing-only namespace so it exists physically but is absent from __manifest.
    let dir_only_ns = DirectoryNamespaceBuilder::new(root_uri.clone())
        .session(session.clone())
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    dir_only_ns
        .create_table(create_req, Bytes::from(ipc_data))
        .await
        .unwrap();

    let hybrid_ns = DirectoryNamespaceBuilder::new(root_uri)
        .session(session)
        .manifest_enabled(true)
        .dir_listing_enabled(true)
        .dir_listing_to_manifest_migration_enabled(true)
        .build()
        .await
        .unwrap();

    // In migration mode the manifest is authoritative, so table_exists first
    // probes __manifest via ensure_read_manifest() to self-heal any
    // manifest-registered aliases (why the gate now admits this mode). Here
    // the table is dir-only and __manifest does not exist yet, so that probe
    // costs one list to confirm absence; the table-directory fallback is the
    // second. (When __manifest exists the probe uses the version hint and
    // adds no list, so a real self-heal is free.)
    listing_count.store(0, Ordering::SeqCst);

    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["test_table".to_string()]);
    hybrid_ns.table_exists(exists_req).await.unwrap();

    let count = listing_count.load(Ordering::SeqCst);
    assert_eq!(
        count, 2,
        "Expected 2 listing calls for table_exists with migration mode \
             (absent-__manifest probe + table directory fallback), but got {}",
        count
    );

    // describe_table follows the same path: an ensure_read_manifest() probe
    // of the (absent) __manifest, then the table-directory fallback.
    listing_count.store(0, Ordering::SeqCst);

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    hybrid_ns.describe_table(describe_req).await.unwrap();

    let count = listing_count.load(Ordering::SeqCst);
    assert_eq!(
        count, 2,
        "Expected 2 listing calls for describe_table with migration mode \
             (absent-__manifest probe + table directory fallback), but got {}",
        count
    );
}

#[tokio::test]
async fn test_manifest_reload_observes_new_version_from_other_namespace() {
    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace_a = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(false)
        .build()
        .await
        .unwrap();
    create_scalar_table(&namespace_a, "alpha").await;

    let namespace_b = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(false)
        .build()
        .await
        .unwrap();
    create_scalar_table(&namespace_b, "beta").await;

    let response = namespace_a
        .list_tables(ListTablesRequest {
            id: Some(vec![]),
            ..Default::default()
        })
        .await
        .unwrap();

    let mut tables = response.tables;
    tables.sort();
    assert_eq!(tables, vec!["alpha", "beta"]);
}

#[tokio::test]
async fn test_migration_not_found_errors_include_table_id() {
    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(true)
        .dir_listing_to_manifest_migration_enabled(true)
        .build()
        .await
        .unwrap();

    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["missing_table".to_string()]);
    let err = namespace.table_exists(exists_req).await.unwrap_err();
    assert!(matches!(err, Error::Namespace { .. }));
    let err_msg = err.to_string();
    assert!(err_msg.contains("Table not found"));
    assert!(err_msg.contains("table id 'missing_table'"));

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["missing_table".to_string()]);
    let err = namespace.describe_table(describe_req).await.unwrap_err();
    assert!(matches!(err, Error::Namespace { .. }));
    let err_msg = err.to_string();
    assert!(err_msg.contains("Table not found"));
    assert!(err_msg.contains("table id 'missing_table'"));
}

#[tokio::test]
async fn test_manifest_not_found_errors_include_full_table_id() {
    use lance_namespace::models::CreateNamespaceRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["workspace".to_string()]);
    namespace.create_namespace(create_ns_req).await.unwrap();

    let missing_table_id = vec!["workspace".to_string(), "missing_table".to_string()];

    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(missing_table_id.clone());
    let err = namespace.table_exists(exists_req).await.unwrap_err();
    assert!(matches!(err, Error::Namespace { .. }));
    let err_msg = err.to_string();
    assert!(err_msg.contains("Table not found"));
    assert!(err_msg.contains("table id 'workspace$missing_table'"));

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(missing_table_id);
    let err = namespace.describe_table(describe_req).await.unwrap_err();
    assert!(matches!(err, Error::Namespace { .. }));
    let err_msg = err.to_string();
    assert!(err_msg.contains("Table not found"));
    assert!(err_msg.contains("table id 'workspace$missing_table'"));
}
