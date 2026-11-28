// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use lance_table::io::commit::ManifestNamingScheme::{V1, V2};

/// The early-stop path (ordered stores) and the collect-then-sort path
/// must return the same results for every descending/limit combination.
#[tokio::test]
async fn test_list_versions_under_ordering_and_limit() {
    use lance_table::io::commit::ManifestNamingScheme;

    fn versions(r: &[TableVersion]) -> Vec<i64> {
        r.iter().map(|t| t.version).collect()
    }

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

        // A descending page bound keeps the early stop and skips the
        // versions at or above it.
        let got = ns
            .list_versions_below(&table_path, true, Some(2), Some(5))
            .await
            .unwrap();
        assert_eq!(versions(&got), vec![4, 3]);
        let got = ns
            .list_versions_below(&table_path, true, Some(3), Some(100))
            .await
            .unwrap();
        assert_eq!(versions(&got), vec![7, 6, 5]);
        let got = ns
            .list_versions_below(&table_path, true, None, Some(1))
            .await
            .unwrap();
        assert!(got.is_empty());
    }

    /// V1 manifest names are unpadded (`10.manifest` sorts lexically before
    /// `2.manifest`), so even a lexically-ordered listing does not arrive in
    /// version order: V1 listings must be collected and sorted numerically,
    /// never early-stopped as if the stream matched the request.
    async fn seed_v1_and_check(ns: &DirectoryNamespace) {
        let table_path = ns.base_path.clone().join("lv_test_v1.lance");
        for v in 1..=12u64 {
            let p = ManifestNamingScheme::V1.manifest_path(&table_path, v);
            ns.object_store.put(&p, b"m".as_slice()).await.unwrap();
        }

        let got = ns
            .list_versions_under(&table_path, false, Some(5))
            .await
            .unwrap();
        assert_eq!(versions(&got), vec![1, 2, 3, 4, 5]);

        let got = ns
            .list_versions_under(&table_path, false, None)
            .await
            .unwrap();
        assert_eq!(versions(&got), (1..=12i64).collect::<Vec<_>>());

        let got = ns
            .list_versions_under(&table_path, true, Some(3))
            .await
            .unwrap();
        assert_eq!(versions(&got), vec![12, 11, 10]);
    }

    let ns_mem = DirectoryNamespaceBuilder::new("memory://lv-test")
        .build()
        .await
        .unwrap();
    assert!(ns_mem.object_store.list_is_lexically_ordered);
    seed_and_check(&ns_mem).await;
    seed_v1_and_check(&ns_mem).await;

    let (ns_fs, _tmp) = create_test_namespace().await;
    assert!(!ns_fs.object_store.list_is_lexically_ordered);
    seed_and_check(&ns_fs).await;
}

/// A partly migrated table mixes V1 and V2 manifest names. On an ordered
/// store, a V1 name can sort before the V2 names (`10.manifest`), between
/// them (a 19-digit version), or after them (`2.manifest`). Whenever the
/// early stop can meet that V1 name, every bounded descending walk must
/// match the unbounded listing.
#[rstest::rstest]
#[case::v2_prefix_v1_tail(&[2, 3], &[4, 5])]
#[case::v1_sorts_before_v2(&[3, 10], &[11, 12])]
// `1844674407370955161.manifest` sorts between the V2 names of 6 and 1.
#[case::v1_between_v2_names(&[1_844_674_407_370_955_161], &[1, 6])]
#[case::v2_only(&[], &[1, 2, 3, 4, 5])]
#[tokio::test]
async fn test_descending_pages_across_manifest_layouts(
    #[case] v1_versions: &[u64],
    #[case] v2_versions: &[u64],
) {
    let ns = DirectoryNamespaceBuilder::new("memory://mixed-layout-pages")
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();
    assert!(ns.object_store.list_is_lexically_ordered);
    let table = ns.table_path("events");
    let manifests = v1_versions
        .iter()
        .map(|v| (V1, *v))
        .chain(v2_versions.iter().map(|v| (V2, *v)));
    for (scheme, version) in manifests {
        ns.object_store
            .put(&scheme.manifest_path(&table, version), b"m")
            .await
            .unwrap();
    }
    let request = ListTableVersionsRequest {
        id: Some(vec!["events".into()]),
        descending: Some(true),
        ..Default::default()
    };
    let numbers = |rows: &[TableVersion]| rows.iter().map(|row| row.version).collect::<Vec<_>>();
    let unbounded = ns.list_table_versions(request.clone()).await.unwrap();
    let expected = numbers(&unbounded.versions);
    let total = v1_versions.len() + v2_versions.len();
    assert_eq!(expected.len(), total);
    assert!(expected.windows(2).all(|pair| pair[0] > pair[1]));

    for limit in 1..=total as i32 {
        let mut page_request = ListTableVersionsRequest {
            limit: Some(limit),
            ..request.clone()
        };
        let mut walked = Vec::new();
        loop {
            let page = ns.list_table_versions(page_request.clone()).await.unwrap();
            walked.extend(numbers(&page.versions));
            match page.page_token {
                Some(token) => page_request.page_token = Some(token),
                None => break,
            }
            assert!(walked.len() <= total, "walk did not terminate");
        }
        assert_eq!(walked, expected, "limit = {limit}");
    }
}

#[tokio::test]
async fn test_list_table_versions_pagination_mixed_manifests_and_cleanup() {
    use lance_namespace::models::VersionRange;
    use lance_table::io::commit::{
        ManifestNamingScheme::{V1, V2},
        VERSIONS_DIR,
    };

    let ns = DirectoryNamespaceBuilder::new("memory://version-pages")
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();
    let table = ns.table_path("events");
    for (scheme, version) in [
        (V1, 2),
        (V2, 5),
        (V1, 7),
        (V1, 10),
        (V2, 11),
        (V1, 12),
        (V2, 20),
    ] {
        ns.object_store
            .put(&scheme.manifest_path(&table, version), b"m")
            .await
            .unwrap();
    }
    let dir = table.clone().join(VERSIONS_DIR);
    for filename in [
        "02.manifest",
        "1.manifest-staging",
        "1.backup.manifest",
        "d9223372036854775809.manifest",
    ] {
        ns.object_store
            .put(&dir.clone().join(filename), b"other")
            .await
            .unwrap();
    }
    let nested = dir.clone().join("staging").join("1.manifest");
    ns.object_store.put(&nested, b"other").await.unwrap();

    let mut request = ListTableVersionsRequest {
        id: Some(vec!["events".into()]),
        limit: Some(2),
        ..Default::default()
    };
    let numbers = |rows: &[TableVersion]| rows.iter().map(|row| row.version).collect::<Vec<_>>();
    let descending = |limit| ListTableVersionsRequest {
        descending: Some(true),
        limit,
        ..request.clone()
    };
    let history = ns.list_table_versions(descending(None)).await.unwrap();
    assert_eq!(numbers(&history.versions), [20, 12, 11, 10, 7, 5, 2]);
    let latest = ns.list_table_versions(descending(Some(1))).await.unwrap();
    assert_eq!(numbers(&latest.versions), [20]);
    assert_eq!(latest.page_token.as_deref(), Some("20"));
    // Descending pages continue below the last version returned.
    let mut pages = Vec::new();
    let mut page_request = descending(Some(3));
    loop {
        let page = ns.list_table_versions(page_request.clone()).await.unwrap();
        pages.push(numbers(&page.versions));
        match page.page_token {
            Some(token) => page_request.page_token = Some(token),
            None => break,
        }
    }
    assert_eq!(pages, [vec![20, 12, 11], vec![10, 7, 5], vec![2]]);
    let invalid = ListTableVersionsRequest {
        page_token: Some("01".into()),
        ..descending(Some(3))
    };
    ns.list_table_versions(invalid).await.unwrap_err();
    let first = ns.list_table_versions(request.clone()).await.unwrap();
    assert_eq!(numbers(&first.versions), [2, 5]);
    let deleted = ns
        .batch_delete_table_versions(BatchDeleteTableVersionsRequest {
            id: request.id.clone(),
            ranges: vec![VersionRange::new(1, 2)],
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(deleted.deleted_count, Some(0));
    ns.object_store.stat(&nested).await.unwrap();
    ns.object_store
        .stat(&dir.join("1.backup.manifest"))
        .await
        .unwrap();
    request.page_token = Some(first.page_token.expect("expected a continuation"));
    for (scheme, version) in [(V2, 5), (V1, 7)] {
        ns.object_store
            .inner
            .delete(&scheme.manifest_path(&table, version))
            .await
            .unwrap();
    }
    request.limit = Some(-1);
    let rest = ns.list_table_versions(request.clone()).await.unwrap();
    assert_eq!(numbers(&rest.versions), [10, 11, 12, 20]);
    assert!(rest.page_token.is_none());
    request.limit = Some(0);
    let empty = ns.list_table_versions(request.clone()).await.unwrap();
    assert!(empty.versions.is_empty());
    assert!(empty.page_token.is_none());

    // A logical cursor ending at 10 cannot distinguish two manifests for version 10.
    ns.object_store
        .put(&V2.manifest_path(&table, 10), b"conflict")
        .await
        .unwrap();
    request.limit = Some(1);
    let err = ns.list_table_versions(request).await.unwrap_err();
    assert!(
        err.to_string()
            .contains("Multiple manifests for version 10")
    );
    assert_eq!(mutation_error_code(err), ErrorCode::Internal);
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

        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> BoxStream<'static, OSResult<ObjectMeta>> {
            let entries_listed = self.entries_listed.clone();
            self.target
                .list_with_offset(prefix, offset)
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
            let store = MemoryStoreProvider
                .new_store(base_path.clone(), params)
                .await?;
            let inner = Arc::new(EntryCountingStore {
                target: store.inner,
                entries_listed: self.entries_listed.clone(),
            });
            Ok(ObjectStore::new(
                inner, base_path, None, None, false, true, 1, 0, None,
            ))
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
        "s3",
        Arc::new(EntryCountingMemoryProvider {
            entries_listed: entries_listed.clone(),
        }),
    );
    let session = Arc::new(Session::new(0, 0, registry));
    let ns = DirectoryNamespaceBuilder::new("s3://lv-count")
        .manifest_enabled(false)
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

    // A large V1 version can sort inside the physical V2 run.
    let interleaved = 1_844_674_407_370_955_161;
    ns.object_store
        .put(
            &ManifestNamingScheme::V1.manifest_path(&table_path, interleaved),
            b"m",
        )
        .await
        .unwrap();
    let request = ListTableVersionsRequest {
        id: Some(vec!["lv_count".into()]),
        limit: Some(3),
        page_token: Some("35".into()),
        ..Default::default()
    };
    let before = entries_listed.load(Ordering::SeqCst);
    let page = ns.list_table_versions(request.clone()).await.unwrap();
    assert_eq!(
        page.versions.iter().map(|v| v.version).collect::<Vec<_>>(),
        [36, 37, 38]
    );
    assert_eq!(page.page_token.as_deref(), Some("38"));
    let consumed = entries_listed.load(Ordering::SeqCst) - before;
    assert!(
        consumed <= 5,
        "dense continuation consumed {consumed} entries"
    );

    let history = ns
        .list_table_versions(ListTableVersionsRequest {
            id: request.id.clone(),
            ..Default::default()
        })
        .await
        .unwrap()
        .versions;
    assert_eq!(history.last().unwrap().version, interleaved as i64);
    let mut token = None;
    for expected in history.chunks(3) {
        let page = ns
            .list_table_versions(ListTableVersionsRequest {
                page_token: token,
                ..request.clone()
            })
            .await
            .unwrap();
        assert_eq!(page.versions, expected);
        token = page.page_token;
    }
    assert!(token.is_none());

    // An apparently complete V2 window still has to account for V1 aliases.
    let alias = ManifestNamingScheme::V1.manifest_path(&table_path, 37);
    ns.object_store.put(&alias, b"m").await.unwrap();
    let err = ns.list_table_versions(request.clone()).await.unwrap_err();
    assert!(
        err.to_string()
            .contains("Multiple manifests for version 37")
    );
    assert_eq!(mutation_error_code(err), ErrorCode::Internal);

    ns.object_store.inner.delete(&alias).await.unwrap();
    for v in [35, 36] {
        let key = ManifestNamingScheme::V2.manifest_path(&table_path, v);
        ns.object_store.inner.delete(&key).await.unwrap();
    }
    let page = ns.list_table_versions(request).await.unwrap();
    assert_eq!(
        page.versions.iter().map(|v| v.version).collect::<Vec<_>>(),
        [37, 38, 39]
    );
    assert_eq!(page.page_token.as_deref(), Some("39"));
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
use futures::{TryStreamExt, stream::BoxStream};
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
    WithoutEtags,
    RetargetTagOnBranchRead,
    DeleteTagOnRead,
    DeleteTagOnManifestRead,
    DeleteBranchTagAfterLookup,
    DeleteBranchTagOnManifestRead,
    ReadUntaggedManifestOnce,
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
        let behavior = *self.behavior.lock().unwrap();
        if matches!(behavior, Some(ListBehavior::ReadUntaggedManifestOnce))
            && let Some((root, version)) = location.as_ref().rsplit_once("/users.lance/_versions/")
        {
            // Manifest reads use the length from HEAD to request the data range.
            let other = Path::from(format!("{root}/untagged.lance/_versions/{version}"));
            if !options.head {
                self.behavior.lock().unwrap().take();
            }
            return self.target.get_opts(&other, options).await;
        }
        if !options.head {
            match behavior {
                Some(ListBehavior::DeleteBranchTagAfterLookup)
                    if location
                        .as_ref()
                        .ends_with("/users.lance/_refs/tags/release.json") =>
                {
                    *self.behavior.lock().unwrap() =
                        Some(ListBehavior::DeleteBranchTagOnManifestRead);
                }
                Some(ListBehavior::DeleteBranchTagOnManifestRead)
                    if let Some((root, _)) =
                        location.as_ref().rsplit_once("/tree/dev.lance/_versions/") =>
                {
                    self.target
                        .delete(&lance::dataset::refs::tag_path(
                            &Path::from(root),
                            "release",
                        ))
                        .await?;
                    self.behavior.lock().unwrap().take();
                }
                Some(ListBehavior::DeleteTagOnRead)
                    if location.as_ref().ends_with("/_refs/tags/release.json") =>
                {
                    self.target.delete(location).await?;
                }
                Some(ListBehavior::RetargetTagOnBranchRead)
                    if let Some(root) =
                        location.as_ref().strip_suffix("/_refs/branches/dev.json") =>
                {
                    let root = Path::from(root);
                    self.target
                        .copy(
                            &lance::dataset::refs::tag_path(&root, "replacement"),
                            &lance::dataset::refs::tag_path(&root, "release"),
                        )
                        .await?;
                }
                Some(ListBehavior::DeleteTagOnManifestRead)
                    if let Some((root, _)) = location.as_ref().rsplit_once("/_versions/") =>
                {
                    self.behavior.lock().unwrap().take();
                    self.target
                        .delete(&lance::dataset::refs::tag_path(
                            &Path::from(root),
                            "release",
                        ))
                        .await?;
                }
                _ => {}
            }
        }
        let mut result = self.target.get_opts(location, options).await?;
        if matches!(behavior, Some(ListBehavior::WithoutEtags)) {
            result.meta.e_tag = None;
        }
        Ok(result)
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
        let entries = self.target.list(prefix);
        if matches!(
            *self.behavior.lock().unwrap(),
            Some(ListBehavior::WithoutEtags)
        ) {
            entries
                .map_ok(|mut meta| {
                    meta.e_tag = None;
                    meta
                })
                .boxed()
        } else {
            entries
        }
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        let behavior = *self.behavior.lock().unwrap();
        match behavior {
            None
            | Some(
                ListBehavior::WithoutEtags
                | ListBehavior::RetargetTagOnBranchRead
                | ListBehavior::DeleteTagOnRead
                | ListBehavior::DeleteTagOnManifestRead
                | ListBehavior::DeleteBranchTagAfterLookup
                | ListBehavior::DeleteBranchTagOnManifestRead
                | ListBehavior::ReadUntaggedManifestOnce,
            ) => self.target.list_with_delimiter(prefix).await,
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

#[tokio::test]
async fn test_describe_table_snapshot_selectors_recreated_branch() {
    use lance::dataset::ColumnAlteration;

    let (writer, temp) = create_test_namespace().await;
    create_scalar_table(&writer, "users").await;
    let mut main = open_dataset(&writer, "users").await;
    let registry =
        build_failing_list_session(Arc::new(Mutex::new(Some(ListBehavior::WithoutEtags))))
            .store_registry();
    let session = Arc::new(Session::new(0, 1024 * 1024, registry));
    let reader = DirectoryNamespaceBuilder::new(file_object_store_uri(temp.to_str().unwrap()))
        .session(session.clone())
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();
    let request = DescribeTableRequest {
        id: Some(vec!["users".into()]),
        branch: Some("dev.lance".into()),
        version: Some(2),
        load_detailed_metadata: Some(true),
        ..Default::default()
    };
    for field in ["old_id", "new_id"] {
        let mut dev = main.create_branch("dev.lance", 1, None).await.unwrap();
        dev.alter_columns(&[ColumnAlteration::new("id".into()).rename(field.into())])
            .await
            .unwrap();
        dev.migrate_manifest_paths_v2().await.unwrap();
        dev.tags().create("release", 2).await.unwrap();
        let meta = reader
            .object_store
            .inner
            .head(&dev.manifest_location().path)
            .await
            .unwrap();
        assert!(meta.e_tag.is_none());
        let described = reader.describe_table(request.clone()).await.unwrap();
        assert_eq!(
            described.schema.unwrap().fields[0].name,
            field,
            "DescribeTable must resolve the selected snapshot after a same-name branch recreation"
        );
        let listing = DirectoryNamespaceBuilder::new(file_object_store_uri(
            temp.as_ref().join("users.lance/tree").to_str().unwrap(),
        ))
        .session(session.clone())
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();
        for (namespace, table) in [(&reader, "users"), (&listing, "dev")] {
            let tagged = namespace
                .describe_table(DescribeTableRequest {
                    id: Some(vec![table.into()]),
                    tag: Some("release".into()),
                    branch: None,
                    version: None,
                    ..request.clone()
                })
                .await
                .unwrap();
            assert_eq!(tagged.schema.unwrap().fields[0].name, field);
        }
        for branch in [None, request.branch.clone()] {
            let alias = listing
                .describe_table(DescribeTableRequest {
                    id: Some(vec!["dev".into()]),
                    branch: branch.clone(),
                    ..request.clone()
                })
                .await
                .unwrap();
            assert_eq!(alias.schema.unwrap().fields[0].name, field, "{branch:?}");
        }
        dev.tags().delete("release").await.unwrap();
        main.delete_branch("dev.lance").await.unwrap();
    }
}

#[rstest]
#[case::with_etags(None)]
#[case::without_etags(Some(ListBehavior::WithoutEtags))]
#[tokio::test]
async fn test_describe_republished_branch_version(#[case] behavior: Option<ListBehavior>) {
    use lance::dataset::ColumnAlteration;
    use lance_namespace::models::{BatchDeleteTableVersionsRequest, VersionRange};

    let (writer, temp) = create_test_namespace().await;
    create_scalar_table(&writer, "users").await;
    let mut main = open_dataset(&writer, "users").await;
    let mut dev = main.create_branch("dev", 1, None).await.unwrap();
    dev.migrate_manifest_paths_v2().await.unwrap();
    let registry = build_failing_list_session(Arc::new(Mutex::new(behavior))).store_registry();
    let reader = DirectoryNamespaceBuilder::new(file_object_store_uri(temp.to_str().unwrap()))
        .session(Arc::new(Session::new(0, 1024 * 1024, registry)))
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();
    let request = DescribeTableRequest {
        id: Some(vec!["users".into()]),
        branch: Some("dev".into()),
        version: Some(2),
        load_detailed_metadata: Some(true),
        ..Default::default()
    };
    let publish_and_describe = async |dev: &mut Dataset, name: &str| {
        dev.alter_columns(&[ColumnAlteration::new("id".into()).rename(name.into())])
            .await
            .unwrap();
        assert_eq!(dev.version().version, 2);
        let result = reader.describe_table(request.clone()).await.unwrap();
        assert_eq!(result.schema.unwrap().fields[0].name, name);
    };
    publish_and_describe(&mut dev, "old_id").await;
    let deleted = writer
        .batch_delete_table_versions(BatchDeleteTableVersionsRequest {
            id: request.id.clone(),
            branch: request.branch.clone(),
            ranges: vec![VersionRange::new(2, 3)],
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(deleted.deleted_count, Some(1));
    let mut reopened = Dataset::open(dev.uri()).await.unwrap();
    assert_eq!(reopened.version().version, 1);
    publish_and_describe(&mut reopened, "new_id").await;
}

#[rstest]
#[case::retarget_branch(
    ListBehavior::RetargetTagOnBranchRead,
    ErrorCode::ConcurrentModification
)]
#[case::delete_before_tag_read(ListBehavior::DeleteTagOnRead, ErrorCode::TableTagNotFound)]
#[case::delete_before_main_snapshot(
    ListBehavior::DeleteTagOnManifestRead,
    ErrorCode::TableTagNotFound
)]
#[tokio::test]
async fn test_describe_table_snapshot_selectors_racing_tag(
    #[case] event: ListBehavior,
    #[case] expected: ErrorCode,
) {
    let (reader, temp, toggle) = failing_list_namespace().await;
    let writer = DirectoryNamespaceBuilder::new(temp.to_str().unwrap())
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();
    create_scalar_table(&writer, "users").await;
    let mut main = open_dataset(&writer, "users").await;
    if matches!(event, ListBehavior::RetargetTagOnBranchRead) {
        let dev = main.create_branch("dev", 1, None).await.unwrap();
        dev.tags().create("release", 1).await.unwrap();
        main.tags().create("replacement", 1).await.unwrap();
    } else {
        main.tags().create("release", 1).await.unwrap();
    }
    let request = DescribeTableRequest {
        id: Some(vec!["users".into()]),
        tag: Some("release".into()),
        load_detailed_metadata: Some(true),
        ..Default::default()
    };
    *toggle.lock().unwrap() = Some(event);
    let error = reader.describe_table(request).await.unwrap_err();
    assert_eq!(mutation_error_code(error), expected, "{event:?}");
}

#[rstest]
#[case::table_root(false)]
#[case::direct_branch(true)]
#[tokio::test]
async fn test_branch_tag_deleted_during_snapshot_open(#[case] direct: bool) {
    let (reader, temp, toggle) = failing_list_namespace().await;
    let writer = DirectoryNamespaceBuilder::new(temp.to_str().unwrap())
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();
    create_scalar_table(&writer, "users").await;
    let mut main = open_dataset(&writer, "users").await;
    let dev = main.create_branch("dev.lance", 1, None).await.unwrap();
    dev.tags().create("release", 1).await.unwrap();
    let (namespace, table) = if direct {
        let tree = temp.as_ref().join("users.lance/tree");
        let listed = DirectoryNamespaceBuilder::new(file_object_store_uri(tree.to_str().unwrap()))
            .session(reader.session.clone().unwrap())
            .manifest_enabled(false)
            .build()
            .await
            .unwrap();
        (listed, "dev")
    } else {
        (reader, "users")
    };
    *toggle.lock().unwrap() = Some(ListBehavior::DeleteBranchTagAfterLookup);
    let error = namespace
        .describe_table(DescribeTableRequest {
            id: Some(vec![table.into()]),
            tag: Some("release".into()),
            load_detailed_metadata: Some(true),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(toggle.lock().unwrap().is_none());
    assert_eq!(mutation_error_code(error), ErrorCode::TableTagNotFound);
}

#[rstest]
#[case::modern(false)]
#[case::legacy(true)]
#[tokio::test]
async fn test_tag_rejects_a_manifest_from_an_untagged_table(#[case] legacy: bool) {
    use lance::dataset::ColumnAlteration;

    let (reader, temp, toggle) = failing_list_namespace().await;
    let writer = DirectoryNamespaceBuilder::new(temp.to_str().unwrap())
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();
    for (name, field) in [("users", "old_id"), ("untagged", "untagged_table_field")] {
        create_scalar_table(&writer, name).await;
        let mut dataset = open_dataset(&writer, name).await;
        dataset
            .alter_columns(&[ColumnAlteration::new("id".into()).rename(field.into())])
            .await
            .unwrap();
    }
    let main = open_dataset(&writer, "users").await;
    main.tags().create("release", 2).await.unwrap();
    if legacy {
        let mut tag = main.tags().get("release").await.unwrap();
        tag.created_at = None;
        tag.updated_at = None;
        std::fs::write(
            temp.as_ref().join("users.lance/_refs/tags/release.json"),
            serde_json::to_vec(&tag).unwrap(),
        )
        .unwrap();
    }
    let request = DescribeTableRequest {
        id: Some(vec!["users".into()]),
        tag: Some("release".into()),
        load_detailed_metadata: Some(true),
        ..Default::default()
    };
    // The tag reads agree, but the intervening manifest GET sees a different table.
    *toggle.lock().unwrap() = Some(ListBehavior::ReadUntaggedManifestOnce);
    let err = reader.describe_table(request.clone()).await.unwrap_err();
    assert_eq!(mutation_error_code(err), ErrorCode::ConcurrentModification);
    let stable = reader.describe_table(request).await.unwrap();
    assert_eq!(stable.version, Some(2));
    assert_eq!(stable.schema.unwrap().fields[0].name, "old_id");
}

#[tokio::test]
async fn test_describe_table_snapshot_selectors_recreated_main_tag() {
    use lance::dataset::ColumnAlteration;
    use lance_namespace::models::DropTableRequest;

    let (writer, temp) = create_test_namespace().await;
    let registry =
        build_failing_list_session(Arc::new(Mutex::new(Some(ListBehavior::WithoutEtags))))
            .store_registry();
    let reader = DirectoryNamespaceBuilder::new(file_object_store_uri(temp.to_str().unwrap()))
        .session(Arc::new(Session::new(0, 1024 * 1024, registry)))
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();
    let request = DescribeTableRequest {
        id: Some(vec!["users".into()]),
        tag: Some("release".into()),
        load_detailed_metadata: Some(true),
        ..Default::default()
    };
    for field in ["old_id", "new_id"] {
        create_scalar_table(&writer, "users").await;
        let mut main = open_dataset(&writer, "users").await;
        main.alter_columns(&[ColumnAlteration::new("id".into()).rename(field.into())])
            .await
            .unwrap();
        main.migrate_manifest_paths_v2().await.unwrap();
        main.tags().create("release", 2).await.unwrap();
        let tagged = reader.describe_table(request.clone()).await.unwrap();
        assert_eq!(tagged.version, Some(2));
        assert_eq!(tagged.schema.unwrap().fields[0].name, field);
        writer
            .drop_table(DropTableRequest {
                id: request.id.clone(),
                ..Default::default()
            })
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn test_describe_directory_branch_alias_replacing_an_ordinary_table() {
    use lance::dataset::ColumnAlteration;
    use lance_namespace::models::DropTableRequest;

    let (writer, temp) = create_test_namespace().await;
    create_scalar_table(&writer, "users").await;
    let mut main = open_dataset(&writer, "users").await;
    let root = temp.as_ref().join("users.lance/tree/feature");
    std::fs::create_dir_all(&root).unwrap();
    let root = root.to_str().unwrap();
    let alias_writer = DirectoryNamespaceBuilder::new(root)
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();
    let registry =
        build_failing_list_session(Arc::new(Mutex::new(Some(ListBehavior::WithoutEtags))))
            .store_registry();
    let reader = DirectoryNamespaceBuilder::new(file_object_store_uri(root))
        .session(Arc::new(Session::new(0, 1024 * 1024, registry)))
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();
    let request = DescribeTableRequest {
        id: Some(vec!["dev".into()]),
        version: Some(2),
        load_detailed_metadata: Some(true),
        ..Default::default()
    };
    for (branch, field) in [
        (false, "original"),
        (true, "branched"),
        (false, "replacement"),
    ] {
        let mut dataset = if branch {
            main.create_branch("feature/dev.lance", 1, None)
                .await
                .unwrap()
        } else {
            create_scalar_table(&alias_writer, "dev").await;
            open_dataset(&alias_writer, "dev").await
        };
        dataset
            .alter_columns(&[ColumnAlteration::new("id".into()).rename(field.into())])
            .await
            .unwrap();
        dataset.migrate_manifest_paths_v2().await.unwrap();
        let described = reader.describe_table(request.clone()).await.unwrap();
        assert_eq!(described.schema.unwrap().fields[0].name, field);
        if branch {
            main.delete_branch("feature/dev.lance").await.unwrap();
        } else {
            alias_writer
                .drop_table(DropTableRequest {
                    id: request.id.clone(),
                    ..Default::default()
                })
                .await
                .unwrap();
        }
    }
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
