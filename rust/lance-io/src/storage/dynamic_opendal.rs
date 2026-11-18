// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::storage::backend::MultipartUpload;
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use crate::storage::path::Path;
use crate::storage::{
    CopyOptions, GetOptions, GetResult, ListResult, ObjectMeta, PutMultipartOptions, PutOptions,
    PutPayload, PutResult, RenameOptions, StorageBackend,
};
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, stream, stream::BoxStream};
use tokio::sync::RwLock;

use crate::storage::StorageOptionsAccessor;
use crate::storage::opendal_store::OpendalStore;
use lance_core::Result;

type NormalizeConfigFn = fn(&HashMap<String, String>) -> Result<HashMap<String, String>>;
type BuildStoreFn = fn(HashMap<String, String>) -> Result<OpendalStore>;

#[derive(Debug, Clone)]
struct CachedOpenDalStore {
    config: HashMap<String, String>,
    store: Arc<OpendalStore>,
}

#[derive(Clone)]
pub(in crate::storage) struct DynamicOpenDalStore {
    name: Arc<str>,
    base_options: Arc<HashMap<String, String>>,
    accessor: Arc<StorageOptionsAccessor>,
    normalize_config: NormalizeConfigFn,
    build_store: BuildStoreFn,
    protected_keys: Vec<&'static str>,
    cache: Arc<RwLock<Option<CachedOpenDalStore>>>,
}

impl fmt::Debug for DynamicOpenDalStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicOpenDalStore")
            .field("name", &self.name)
            .field("accessor", &self.accessor)
            .finish()
    }
}

impl DynamicOpenDalStore {
    pub(in crate::storage) fn new(
        name: impl Into<Arc<str>>,
        base_options: HashMap<String, String>,
        accessor: Arc<StorageOptionsAccessor>,
        normalize_config: NormalizeConfigFn,
        build_store: BuildStoreFn,
    ) -> Self {
        Self {
            name: name.into(),
            base_options: Arc::new(base_options),
            accessor,
            normalize_config,
            build_store,
            protected_keys: Vec::new(),
            cache: Arc::new(RwLock::new(None)),
        }
    }

    pub(in crate::storage) fn with_protected_keys(
        mut self,
        keys: impl IntoIterator<Item = &'static str>,
    ) -> Self {
        self.protected_keys = keys.into_iter().collect();
        self
    }

    fn merge_options(
        &self,
        mut dynamic_options: HashMap<String, String>,
    ) -> HashMap<String, String> {
        for key in &self.protected_keys {
            dynamic_options.remove(*key);
        }
        let mut merged = self.base_options.as_ref().clone();
        merged.extend(dynamic_options);
        merged
    }

    pub(in crate::storage) async fn current_store(&self) -> Result<Arc<OpendalStore>> {
        let merged_options = self.merge_options(self.accessor.get_storage_options().await?.0);
        let config = (self.normalize_config)(&merged_options)?;

        // Cache reuse depends on exact normalized config equality. Providers
        // should return stable, canonicalized values for semantically identical
        // configurations to avoid unnecessary store rebuilds.
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.as_ref()
                && cached.config == config
            {
                return Ok(cached.store.clone());
            }
        }

        let store = Arc::new((self.build_store)(config.clone())?);
        let mut cache = self.cache.write().await;
        if let Some(cached) = cache.as_ref()
            && cached.config == config
        {
            return Ok(cached.store.clone());
        }

        *cache = Some(CachedOpenDalStore {
            config,
            store: store.clone(),
        });
        Ok(store)
    }
}

impl fmt::Display for DynamicOpenDalStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DynamicOpenDalStore({})", self.name)
    }
}

#[async_trait::async_trait]
impl StorageBackend for DynamicOpenDalStore {
    async fn list_paginated(
        &self,
        _prefix: Option<&str>,
        _options: crate::storage::PaginatedListOptions,
    ) -> crate::storage::Result<crate::storage::PaginatedListResult> {
        Err(crate::storage::Error::NotSupported {
            source: "OpenDAL pagination is unavailable".into(),
        })
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, crate::storage::Result<ObjectMeta>> {
        let prefix = prefix.cloned();
        let offset = offset.clone();
        let this = self.clone();
        stream::once(async move {
            this.current_store()
                .await
                .map_err(crate::storage::Error::from)
        })
        .map_ok(move |store| store.list_with_offset(prefix.as_ref(), &offset))
        .try_flatten()
        .boxed()
    }

    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> crate::storage::Result<PutResult> {
        self.current_store()
            .await?
            .put_opts(location, payload, opts)
            .await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> crate::storage::Result<Box<dyn MultipartUpload>> {
        self.current_store()
            .await?
            .put_multipart_opts(location, opts)
            .await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> crate::storage::Result<GetResult> {
        self.current_store()
            .await?
            .get_opts(location, options)
            .await
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> crate::storage::Result<Vec<Bytes>> {
        self.current_store()
            .await?
            .get_ranges(location, ranges)
            .await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, crate::storage::Result<Path>>,
    ) -> BoxStream<'static, crate::storage::Result<Path>> {
        let this = self.clone();
        stream::once(async move {
            let store = this
                .current_store()
                .await
                .map_err(crate::storage::Error::from)?;
            Ok::<_, crate::storage::Error>((store, locations))
        })
        .map_ok(|(store, locations)| store.delete_stream(locations))
        .try_flatten()
        .boxed()
    }

    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> BoxStream<'static, crate::storage::Result<ObjectMeta>> {
        let prefix = prefix.cloned();
        let this = self.clone();
        stream::once(async move {
            this.current_store()
                .await
                .map_err(crate::storage::Error::from)
        })
        .map_ok(move |store| store.list(prefix.as_ref()))
        .try_flatten()
        .boxed()
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&Path>,
    ) -> crate::storage::Result<ListResult> {
        self.current_store()
            .await?
            .list_with_delimiter(prefix)
            .await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        opts: CopyOptions,
    ) -> crate::storage::Result<()> {
        self.current_store().await?.copy_opts(from, to, opts).await
    }

    async fn rename_opts(
        &self,
        from: &Path,
        to: &Path,
        opts: RenameOptions,
    ) -> crate::storage::Result<()> {
        self.current_store()
            .await?
            .rename_opts(from, to, opts)
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use async_trait::async_trait;

    use opendal::{Operator, services::Memory};

    use super::*;
    use crate::storage::Storage;
    use crate::storage::providers::opendal::finish_opendal_operator;
    use crate::storage::test_utils::StaticMockStorageOptionsProvider;
    use crate::storage::{StorageOptions, StorageOptionsProvider};

    fn build_memory_store(config: HashMap<String, String>) -> Result<OpendalStore> {
        let storage_options = StorageOptions(config);
        let operator = Operator::new(Memory::default()).map_err(|e| {
            lance_core::Error::invalid_input(format!("Failed to create memory operator: {e:?}"))
        })?;
        Ok(OpendalStore::new(finish_opendal_operator(
            operator,
            storage_options.client_max_retries(),
        )))
    }

    #[tokio::test]
    async fn invalid_dynamic_configuration_keeps_core_io_boundary() {
        let store = DynamicOpenDalStore::new(
            "memory",
            HashMap::new(),
            Arc::new(StorageOptionsAccessor::with_static_options(HashMap::new())),
            |_| Err(lance_core::Error::invalid_input("invalid endpoint")),
            |_| panic!("invalid configuration must not build a backend"),
        );
        let error = (&store as &dyn StorageBackend)
            .head(&Path::from("data"))
            .await
            .unwrap_err();
        assert!(matches!(error, crate::storage::Error::Generic { .. }));
        let error = lance_core::Error::from(error);
        assert!(matches!(error, lance_core::Error::IO { .. }));
        assert!(error.to_string().contains("invalid endpoint"));
    }

    #[tokio::test]
    async fn test_dynamic_store_caches_by_normalized_config() {
        let accessor = Arc::new(StorageOptionsAccessor::with_provider(Arc::new(
            StaticMockStorageOptionsProvider {
                options: HashMap::from([("token".to_string(), "value".to_string())]),
            },
        )));

        let store = DynamicOpenDalStore::new(
            "memory",
            HashMap::new(),
            accessor,
            |options| Ok(options.clone()),
            build_memory_store,
        );

        let first = store
            .current_store()
            .await
            .expect("first store should build");
        let second = store
            .current_store()
            .await
            .expect("second store should reuse cache");

        assert!(Arc::ptr_eq(&first, &second));
        let dynamic = Arc::new(store);
        let (mut storage, _) = Storage::from_uri("memory://").await.unwrap();
        let storage_mut = Arc::get_mut(&mut storage).unwrap();
        storage_mut.apply_wrapper(&crate::utils::testing::TestBackendWrapper::new(
            |_backend| dynamic.clone(),
        ));
        (first.as_ref() as &dyn StorageBackend)
            .put(
                &Path::from("dataset/file"),
                Bytes::from_static(b"test").into(),
            )
            .await
            .unwrap();
        storage.remove_dir_all("dataset").await.unwrap();
        assert!(matches!(
            (first.as_ref() as &dyn StorageBackend)
                .head(&Path::from("dataset/file"))
                .await,
            Err(crate::storage::Error::NotFound { .. })
        ));
    }

    #[derive(Debug)]
    struct ChangingRetryConfigProvider {
        fetch_count: AtomicUsize,
    }

    #[async_trait]
    impl StorageOptionsProvider for ChangingRetryConfigProvider {
        async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
            let fetch_count = self.fetch_count.fetch_add(1, Ordering::SeqCst);
            Ok(Some(HashMap::from([
                (
                    "client_max_retries".to_string(),
                    (fetch_count + 1).to_string(),
                ),
                ("expires_at_millis".to_string(), "0".to_string()),
            ])))
        }

        fn provider_id(&self) -> String {
            "ChangingRetryConfigProvider".to_string()
        }
    }

    #[tokio::test]
    async fn test_dynamic_store_rebuilds_when_retry_config_changes() {
        let accessor = Arc::new(StorageOptionsAccessor::with_provider(Arc::new(
            ChangingRetryConfigProvider {
                fetch_count: AtomicUsize::new(0),
            },
        )));
        let store = DynamicOpenDalStore::new(
            "memory",
            HashMap::new(),
            accessor,
            |options| Ok(options.clone()),
            build_memory_store,
        );

        let first = store
            .current_store()
            .await
            .expect("first store should build");
        let second = store
            .current_store()
            .await
            .expect("changed retry config should rebuild store");

        assert!(!Arc::ptr_eq(&first, &second));
    }
    #[test]
    fn test_merge_options_preserves_protected_base_keys() {
        let accessor = Arc::new(StorageOptionsAccessor::with_provider(Arc::new(
            StaticMockStorageOptionsProvider {
                options: HashMap::new(),
            },
        )));
        let store = DynamicOpenDalStore::new(
            "memory",
            HashMap::from([
                ("bucket".to_string(), "url-bucket".to_string()),
                ("root".to_string(), "/".to_string()),
                ("token".to_string(), "base-token".to_string()),
            ]),
            accessor,
            |options| Ok(options.clone()),
            build_memory_store,
        )
        .with_protected_keys(["bucket", "root"]);

        let merged = store.merge_options(HashMap::from([
            ("bucket".to_string(), "provider-bucket".to_string()),
            ("root".to_string(), "/provider-root".to_string()),
            ("token".to_string(), "provider-token".to_string()),
        ]));

        assert_eq!(merged.get("bucket").unwrap(), "url-bucket");
        assert_eq!(merged.get("root").unwrap(), "/");
        assert_eq!(merged.get("token").unwrap(), "provider-token");
    }
}
