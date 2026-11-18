// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::storage::BackendConfig;
use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};

use crate::storage::path::Path;
use crate::storage::{StorageParams, StorageProvider, providers::memory::MemoryStoreProvider};
use ::object_store::memory::InMemory;
use lance_core::error::Result;
use url::Url;

/// Process-global pool of in-memory backends keyed by URL authority.
///
/// Different authorities map to different backends (act as "buckets"); same
/// authority across any caller in the process resolves to the same `Arc<InMemory>`.
/// The pool grows for the lifetime of the process — entries are never evicted.
static SHARED_BACKENDS: LazyLock<Mutex<HashMap<String, Arc<InMemory>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn shared_backend_for(url: &Url) -> Arc<InMemory> {
    SHARED_BACKENDS
        .lock()
        .expect("SHARED_BACKENDS mutex poisoned")
        .entry(url.authority().to_string())
        .or_insert_with(|| Arc::new(InMemory::new()))
        .clone()
}

/// Like [`MemoryStoreProvider`], but every caller resolving a `shared-memory://<authority>/...`
/// URL with the same `<authority>` sees the same backing bytes — across `StorageRegistry`
/// instances, threads, and unrelated components in the same process.
///
/// Intended for tests and harnesses that need multiple actors to coordinate through a
/// common in-memory object store (e.g. a writer and an independent reader, multi-pod
/// fence simulations). Choose distinct authorities for isolation
/// (`shared-memory://test-a` vs `shared-memory://test-b`).
///
/// Unlike `memory://` — which mints a fresh `InMemory` per `new_backend` call — this
/// provider is opt-in precisely so existing tests relying on per-call isolation are
/// unaffected.
#[derive(Default, Debug)]
pub struct SharedMemoryStoreProvider {
    inner: MemoryStoreProvider,
}

#[async_trait::async_trait]
impl StorageProvider for SharedMemoryStoreProvider {
    async fn new_backend(&self, base_path: Url, params: &StorageParams) -> Result<BackendConfig> {
        let mut store = self.inner.new_backend(base_path.clone(), params).await?;
        store.backend = crate::storage::backend::object_store::from_object_store(
            shared_backend_for(&base_path),
        );
        store.scheme = String::from("shared-memory");

        Ok(store)
    }

    fn extract_path(&self, url: &Url) -> Result<Path> {
        // The authority is the bucket; the URL path is the object path within it.
        Ok(Path::from(url.path().trim_start_matches('/')))
    }

    fn calculate_storage_prefix(
        &self,
        url: &Url,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        Ok(format!("shared-memory${}", url.authority()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::PutPayload;
    use crate::storage::Storage;
    use crate::storage::StorageRegistry;
    use bytes::Bytes;

    async fn store_for(uri: &str) -> (Arc<Storage>, Path) {
        let registry = Arc::new(StorageRegistry::default());
        let (store, path) = Storage::from_uri_and_params(registry, uri, &Default::default())
            .await
            .unwrap();
        (store, path)
    }

    #[tokio::test]
    async fn same_authority_shares_bytes_across_registries() {
        let (writer, _) = store_for("shared-memory://bucket-a/").await;
        writer
            .inner
            .put(&Path::from("file"), PutPayload::from_static(b"hello"))
            .await
            .unwrap();

        // Build a *separate* registry — no shared state at the registry layer.
        let (reader, _) = store_for("shared-memory://bucket-a/").await;
        let bytes = reader.inner.get(&Path::from("file")).await.unwrap();
        assert_eq!(bytes.bytes().await.unwrap(), Bytes::from_static(b"hello"));
    }

    #[tokio::test]
    async fn different_authorities_are_isolated() {
        let (a, _) = store_for("shared-memory://iso-a/").await;
        let (b, _) = store_for("shared-memory://iso-b/").await;
        a.inner
            .put(&Path::from("k"), PutPayload::from_static(b"in-a"))
            .await
            .unwrap();
        assert!(b.inner.get(&Path::from("k")).await.is_err());
    }

    #[tokio::test]
    async fn extract_path_strips_authority() {
        let provider = SharedMemoryStoreProvider::default();
        let url = Url::parse("shared-memory://bucket/foo/bar").unwrap();
        assert_eq!(provider.extract_path(&url).unwrap(), Path::from("foo/bar"));
    }

    #[tokio::test]
    async fn from_uri_and_params_resolves_path_correctly() {
        let (store, path) = store_for("shared-memory://path-test/sub/dir/obj").await;
        assert_eq!(path, Path::from("sub/dir/obj"));
        store
            .inner
            .put(&path, PutPayload::from_static(b"payload"))
            .await
            .unwrap();

        let (peer, peer_path) = store_for("shared-memory://path-test/sub/dir/obj").await;
        let bytes = peer.inner.get(&peer_path).await.unwrap();
        assert_eq!(bytes.bytes().await.unwrap(), Bytes::from_static(b"payload"));
    }

    #[test]
    fn calculate_prefix_is_per_authority() {
        let provider = SharedMemoryStoreProvider::default();
        let a = provider
            .calculate_storage_prefix(&Url::parse("shared-memory://x/p").unwrap(), None)
            .unwrap();
        let b = provider
            .calculate_storage_prefix(&Url::parse("shared-memory://y/p").unwrap(), None)
            .unwrap();
        assert_ne!(a, b);
        assert_eq!(a, "shared-memory$x");
    }
}
