// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Testing utilities

use crate::object_store::backend::{MultipartUpload, StorageBackend};
use crate::object_store::{
    CopyOptions, Error as OSError, GetOptions, GetResult, ListResult, ObjectMeta,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, Result as OSResult,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use lance_core::Result;
use lance_core::storage::path::Path;
use std::collections::HashMap;
use std::fmt::Debug;
use std::future;
use std::ops::Range;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

// A policy function takes in the name of the operation (e.g. "put") and the location
// that is being accessed / modified and returns an optional error.
pub trait PolicyFnT: Fn(&str, &Path) -> Result<()> + Send + Sync {}
impl<F> PolicyFnT for F where F: Fn(&str, &Path) -> Result<()> + Send + Sync {}
impl Debug for dyn PolicyFnT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PolicyFn")
    }
}
type PolicyFn = Arc<dyn PolicyFnT>;

// These policy functions receive (and optionally transform) an ObjectMeta
// They apply to functions that list file info
pub trait ObjectMetaPolicyFnT: Fn(&str, ObjectMeta) -> Result<ObjectMeta> + Send + Sync {}
impl<F> ObjectMetaPolicyFnT for F where F: Fn(&str, ObjectMeta) -> Result<ObjectMeta> + Send + Sync {}
impl Debug for dyn ObjectMetaPolicyFnT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PolicyFn")
    }
}
type ObjectMetaPolicyFn = Arc<dyn ObjectMetaPolicyFnT>;

/// A policy container, meant to be shared between test code and the proxy object store.
///
/// This container allows you to configure policies that should apply to the proxied calls.
///
/// Typically, you would use this to simulate I/O errors or mock out data.
///
/// Currently, for simplicity, we only proxy calls that involve some kind of path.  Calls
/// to copy functions, which have a src and dst, will provide the source to the policy
#[derive(Debug, Default)]
pub struct ProxyBackendPolicy {
    /// Policies which run before a method is invoked.  If the policy returns
    /// an error then the target method will not be invoked and the error will
    /// be returned instead.
    before_policies: HashMap<String, PolicyFn>,
    /// Policies which run after calls that return ObjectMeta.  The policy can
    /// transform the returned ObjectMeta to mock out file listing results.
    object_meta_policies: HashMap<String, ObjectMetaPolicyFn>,
}

impl ProxyBackendPolicy {
    pub fn new() -> Self {
        Default::default()
    }

    /// Set a new policy with the given name
    ///
    /// The name can be used to later remove this policy
    pub fn set_before_policy(&mut self, name: &str, policy: PolicyFn) {
        self.before_policies.insert(name.to_string(), policy);
    }

    pub fn clear_before_policy(&mut self, name: &str) {
        self.before_policies.remove(name);
    }

    pub fn set_obj_meta_policy(&mut self, name: &str, policy: ObjectMetaPolicyFn) {
        self.object_meta_policies.insert(name.to_string(), policy);
    }
}

/// A proxy object store
///
/// This store wraps another object store and applies the given policy to all calls
/// made to the underlying store.  This can be used to simulate failures or, perhaps
/// in the future, to mock out results or provide other fine-grained control.
#[derive(Debug)]
pub struct ProxyBackend {
    target: Arc<dyn StorageBackend>,
    policy: Arc<Mutex<ProxyBackendPolicy>>,
}

impl ProxyBackend {
    pub fn new(target: Arc<dyn StorageBackend>, policy: Arc<Mutex<ProxyBackendPolicy>>) -> Self {
        Self { target, policy }
    }

    fn before_method(&self, method: &str, location: &Path) -> OSResult<()> {
        let policy = self.policy.lock().unwrap();
        for policy in policy.before_policies.values() {
            policy(method, location).map_err(OSError::from)?;
        }
        Ok(())
    }

    fn list_with_policy(
        &self,
        prefix: Option<&Path>,
        offset: Option<&Path>,
    ) -> BoxStream<'static, OSResult<ObjectMeta>> {
        if let Err(error) = self.before_method("list", prefix.unwrap_or(&Path::default())) {
            return futures::stream::once(future::ready(Err(error))).boxed();
        }
        let stream = match offset {
            Some(offset) => self.target.list_with_offset(prefix, offset),
            None => self.target.list(prefix),
        };
        let policy = Arc::clone(&self.policy);
        stream
            .and_then(move |meta| {
                let policy = policy.lock().unwrap();
                let transformed = policy
                    .object_meta_policies
                    .values()
                    .try_fold(meta, |meta, policy| {
                        policy("list", meta).map_err(OSError::from)
                    });
                future::ready(transformed)
            })
            .boxed()
    }

    fn transform_meta(&self, method: &str, meta: ObjectMeta) -> OSResult<ObjectMeta> {
        let policy = self.policy.lock().unwrap();
        let mut meta = meta;
        for policy in policy.object_meta_policies.values() {
            meta = policy(method, meta).map_err(OSError::from)?;
        }
        Ok(meta)
    }
}

impl std::fmt::Display for ProxyBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ProxyBackend({})", self.target)
    }
}

/// An object store wrapper that counts listing operations.
///
/// This increments the shared counter for `list`, `list_with_offset`, and `list_with_delimiter`
/// so tests can observe all listing-based directory and version discovery calls.
#[derive(Debug)]
pub struct CountingBackend {
    target: Arc<dyn StorageBackend>,
    listing_count: Arc<AtomicUsize>,
}

impl CountingBackend {
    pub fn new(target: Arc<dyn StorageBackend>, listing_count: Arc<AtomicUsize>) -> Self {
        Self {
            target,
            listing_count,
        }
    }

    fn record_listing(&self) {
        self.listing_count.fetch_add(1, Ordering::SeqCst);
    }
}

impl std::fmt::Display for CountingBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CountingBackend({})", self.target)
    }
}

#[async_trait]
impl StorageBackend for ProxyBackend {
    async fn list_paginated(
        &self,
        prefix: Option<&str>,
        options: crate::object_store::PaginatedListOptions,
    ) -> crate::object_store::Result<crate::object_store::PaginatedListResult> {
        let _ = (prefix, options);
        Err(crate::object_store::Error::NotSupported {
            source: "native pagination is unavailable".into(),
        })
    }

    async fn put_opts(
        &self,
        location: &Path,
        bytes: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.before_method("put", location)?;
        self.target.put_opts(location, bytes, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        self.before_method("put_multipart", location)?;
        self.target.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.before_method("get_opts", location)?;
        let is_head = options.head;
        let mut result = self.target.get_opts(location, options).await?;
        if is_head {
            result.meta = self.transform_meta("head", result.meta)?;
        }
        Ok(result)
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        self.before_method("get_ranges", location)?;
        self.target.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OSResult<Path>>,
    ) -> BoxStream<'static, OSResult<Path>> {
        let policy = Arc::clone(&self.policy);
        let checked = locations
            .and_then(move |location| {
                let result = {
                    let policy = policy.lock().unwrap();
                    policy
                        .before_policies
                        .values()
                        .try_for_each(|policy| policy("delete", &location).map_err(OSError::from))
                };
                future::ready(result.map(|_| location))
            })
            .boxed();
        self.target.delete_stream(checked)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.list_with_policy(prefix, None)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.list_with_policy(prefix, Some(offset))
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.target.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
        self.before_method("copy", from)?;
        self.target.copy_opts(from, to, opts).await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> OSResult<()> {
        self.before_method("rename", from)?;
        self.target.rename_opts(from, to, opts).await
    }
}

#[async_trait]
impl StorageBackend for CountingBackend {
    async fn list_paginated(
        &self,
        prefix: Option<&str>,
        options: crate::object_store::PaginatedListOptions,
    ) -> crate::object_store::Result<crate::object_store::PaginatedListResult> {
        self.listing_count.fetch_add(1, Ordering::SeqCst);
        self.target.list_paginated(prefix, options).await
    }

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
        self.record_listing();
        self.target.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.record_listing();
        self.target.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.record_listing();
        self.target.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
        self.target.copy_opts(from, to, opts).await
    }
    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> OSResult<()> {
        self.target.rename_opts(from, to, opts).await
    }
}

mod delay;
pub use delay::{DelayConfig, DelayedBackend};

/// Install a test backend transformation through Storage's wrapper policy.
///
pub struct TestWrappingObjectStore<F> {
    transform: F,
}

impl<F> TestWrappingObjectStore<F>
where
    F: Fn(
            Arc<dyn crate::object_store::backend::StorageBackend>,
        ) -> Arc<dyn crate::object_store::backend::StorageBackend>
        + Send
        + Sync,
{
    pub fn new(transform: F) -> Self {
        Self { transform }
    }
}

impl<F> Debug for TestWrappingObjectStore<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TestWrappingObjectStore")
    }
}

impl<F> crate::object_store::WrappingObjectStore for TestWrappingObjectStore<F>
where
    F: Fn(
            Arc<dyn crate::object_store::backend::StorageBackend>,
        ) -> Arc<dyn crate::object_store::backend::StorageBackend>
        + Send
        + Sync,
{
    fn wrap(
        &self,
        _store_prefix: &str,
        backend: Arc<dyn crate::object_store::backend::StorageBackend>,
    ) -> Arc<dyn crate::object_store::backend::StorageBackend> {
        (self.transform)(backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn offset_precedes_metadata_policy_and_policy_errors_propagate() {
        let backend = crate::object_store::backend::native::from_object_store(Arc::new(
            object_store::memory::InMemory::new(),
        ));
        for name in ["a", "b", "c"] {
            backend
                .put(&name.into(), Bytes::from_static(b"value").into())
                .await
                .unwrap();
        }
        let policy = Arc::new(Mutex::new(ProxyBackendPolicy::new()));
        policy.lock().unwrap().set_obj_meta_policy(
            "rewrite",
            Arc::new(|_, mut meta| {
                meta.location = Path::from(format!("0/{}", meta.location));
                Ok(meta)
            }),
        );
        let proxy = ProxyBackend::new(backend, policy.clone());
        let objects = proxy
            .list_with_offset(None, &"b".into())
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].location, Path::from("0/c"));

        policy.lock().unwrap().set_obj_meta_policy(
            "rewrite",
            Arc::new(|_, _| Err(lance_core::Error::invalid_input("rejected metadata"))),
        );
        for stream in [proxy.list(None), proxy.list_with_offset(None, &"b".into())] {
            let error = stream.try_collect::<Vec<_>>().await.unwrap_err();
            assert!(error.to_string().contains("rejected metadata"));
        }
    }
}

/// Supplies an already-created test backend through the production registry path.
#[derive(Debug, Clone)]
pub struct TestObjectStoreProvider(pub crate::object_store::BackendConfig);

#[async_trait]
impl crate::object_store::ObjectStoreProvider for TestObjectStoreProvider {
    async fn new_backend(
        &self,
        _url: url::Url,
        _params: &crate::object_store::ObjectStoreParams,
    ) -> Result<crate::object_store::BackendConfig> {
        Ok(self.0.clone())
    }

    fn calculate_storage_prefix(
        &self,
        url: &url::Url,
        options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        match crate::object_store::ObjectStoreRegistry::default().get_provider(url.scheme()) {
            Some(provider) => provider.calculate_storage_prefix(url, options),
            None => Ok(format!("{}${}", url.scheme(), url.authority())),
        }
    }
}

impl TestObjectStoreProvider {
    /// Register the fixture under its scheme in an isolated registry.
    pub fn registry(&self) -> Arc<crate::object_store::ObjectStoreRegistry> {
        let registry = Arc::new(crate::object_store::ObjectStoreRegistry::default());
        registry.insert(&self.0.scheme, Arc::new(self.clone()));
        registry
    }

    /// Build a fresh fixture with production parameter validation and wrappers.
    pub fn build(
        &self,
        url: url::Url,
        params: &crate::object_store::ObjectStoreParams,
    ) -> crate::object_store::ObjectStore {
        use futures::FutureExt;
        self.registry()
            .new_store(url, params)
            .now_or_never()
            .expect("static test provider is synchronous")
            .expect("valid test storage configuration")
            .as_ref()
            .clone()
    }
}
