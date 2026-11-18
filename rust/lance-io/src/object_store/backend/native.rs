// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One-way adapters for native object_store providers, pagination, and uploads.
//! This is the only conversion boundary; Lance's runtime accepts StorageBackend.
use super::*;
use crate::object_store::Error;
mod conversions;
use ::object_store::ObjectStore as OSObjectStore;
pub use conversions::from_error;
use conversions::to_error;
pub(crate) use conversions::to_path;
use std::sync::Arc;

struct NativeObjectStore {
    store: Arc<dyn OSObjectStore>,
    pagination: Option<Arc<dyn ::object_store::list::PaginatedListStore>>,
}
impl Debug for NativeObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("NativeObjectStore")
            .field(&self.store)
            .finish()
    }
}
impl Display for NativeObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.store, f)
    }
}

#[async_trait]
impl StorageBackend for NativeObjectStore {
    async fn list_paginated(
        &self,
        prefix: Option<&str>,
        options: crate::object_store::PaginatedListOptions,
    ) -> Result<crate::object_store::PaginatedListResult> {
        let lister = self
            .pagination
            .as_ref()
            .ok_or_else(|| Error::NotSupported {
                source: "native pagination is unavailable".into(),
            })?;
        let page = lister
            .list_paginated(prefix, options.into())
            .await
            .map_err(from_error)?;
        Ok(crate::object_store::PaginatedListResult {
            result: page.result.try_into()?,
            page_token: page.page_token,
        })
    }

    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        OSObjectStore::put_opts(
            self.store.as_ref(),
            &to_path(location)?,
            payload.into(),
            opts.into(),
        )
        .await
        .map(Into::into)
        .map_err(from_error)
    }
    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        OSObjectStore::put_multipart_opts(self.store.as_ref(), &to_path(location)?, opts.into())
            .await
            .map(from_multipart)
            .map_err(from_error)
    }
    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        OSObjectStore::get_opts(self.store.as_ref(), &to_path(location)?, options.into())
            .await
            .map_err(from_error)?
            .try_into()
    }
    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        OSObjectStore::get_ranges(self.store.as_ref(), &to_path(location)?, ranges)
            .await
            .map_err(from_error)
    }
    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        OSObjectStore::delete_stream(
            self.store.as_ref(),
            locations
                .map(|item| item.and_then(|path| to_path(&path)).map_err(to_error))
                .boxed(),
        )
        .map(|item| {
            item.map_err(from_error)
                .and_then(|path| Path::parse(path.as_ref()).map_err(Error::from))
        })
        .boxed()
    }
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        let prefix = match prefix.map(to_path).transpose() {
            Ok(prefix) => prefix,
            Err(error) => return futures::stream::once(async move { Err(error) }).boxed(),
        };
        OSObjectStore::list(self.store.as_ref(), prefix.as_ref())
            .map(|item| item.map_err(from_error).and_then(TryInto::try_into))
            .boxed()
    }
    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        let paths = prefix
            .map(to_path)
            .transpose()
            .and_then(|prefix| to_path(offset).map(|offset| (prefix, offset)));
        let (prefix, offset) = match paths {
            Ok(paths) => paths,
            Err(error) => return futures::stream::once(async move { Err(error) }).boxed(),
        };
        OSObjectStore::list_with_offset(self.store.as_ref(), prefix.as_ref(), &offset)
            .map(|item| item.map_err(from_error).and_then(TryInto::try_into))
            .boxed()
    }
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        OSObjectStore::list_with_delimiter(
            self.store.as_ref(),
            prefix.map(to_path).transpose()?.as_ref(),
        )
        .await
        .map_err(from_error)?
        .try_into()
    }
    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> Result<()> {
        OSObjectStore::copy_opts(
            self.store.as_ref(),
            &to_path(from)?,
            &to_path(to)?,
            opts.into(),
        )
        .await
        .map_err(from_error)
    }
    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> Result<()> {
        OSObjectStore::rename_opts(
            self.store.as_ref(),
            &to_path(from)?,
            &to_path(to)?,
            opts.into(),
        )
        .await
        .map_err(from_error)
    }
}

/// Adapt a type-erased native provider without rebuilding its credentials.
///
/// ```
/// # use std::sync::Arc;
/// # use lance_io::object_store::backend::native::from_object_store;
/// # async fn example() -> lance_io::object_store::Result<()> {
/// let backend = from_object_store(Arc::new(::object_store::memory::InMemory::new()));
/// backend.put(&"data".into(), bytes::Bytes::from_static(b"payload").into()).await?;
/// assert_eq!(backend.head(&"data".into()).await?.size, 7);
/// # Ok(())
/// # }
/// ```
pub fn from_object_store(store: Arc<dyn OSObjectStore>) -> Arc<dyn StorageBackend> {
    Arc::new(NativeObjectStore {
        store,
        pagination: None,
    })
}

/// Adapt a native provider and its paginated interface as one Lance backend.
/// Both handles must refer to the same physical store and authorization scope.
pub fn from_object_store_with_pagination(
    store: Arc<dyn OSObjectStore>,
    pagination: Arc<dyn ::object_store::list::PaginatedListStore>,
) -> Arc<dyn StorageBackend> {
    Arc::new(NativeObjectStore {
        store,
        pagination: Some(pagination),
    })
}

#[derive(Debug)]
struct NativeUpload(Box<dyn ::object_store::MultipartUpload>);
#[async_trait]
impl MultipartUpload for NativeUpload {
    fn put_part(&mut self, data: PutPayload) -> crate::object_store::UploadPart {
        {
            let future = self.0.put_part(data.into());
            Box::pin(async move { future.await.map_err(from_error) })
        }
    }
    async fn complete(&mut self) -> Result<PutResult> {
        self.0.complete().await.map(Into::into).map_err(from_error)
    }
    async fn abort(&mut self) -> Result<()> {
        self.0.abort().await.map_err(from_error)
    }
}

/// Adapt an upload opened by a provider-specific native client.
pub(crate) fn from_multipart(
    upload: Box<dyn ::object_store::MultipartUpload>,
) -> Box<dyn MultipartUpload> {
    Box::new(NativeUpload(upload))
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_errors_keep_storage_categories_and_core_io_boundary() {
        assert!(matches!(
            super::from_error(::object_store::Error::NotImplemented {
                operation: "list_paginated".into(),
                implementer: "test".into()
            }),
            crate::object_store::Error::NotSupported { .. }
        ));
        let error = super::from_error(::object_store::Error::UnknownConfigurationKey {
            store: "test",
            key: "unknown".into(),
        });
        assert!(matches!(
            error,
            crate::object_store::Error::InvalidInput { .. }
        ));
        assert!(matches!(
            lance_core::Error::from(error),
            lance_core::Error::IO { .. }
        ));
    }

    use super::*;
    use crate::object_store::{Attribute, Attributes, Error};
    use std::error::Error as _;

    #[tokio::test]
    async fn adapter_preserves_encoded_keys_conditions_and_stream_errors() {
        let backend = from_object_store(Arc::new(::object_store::memory::InMemory::new()));
        let path = Path::parse("data/literal%2Fkey").unwrap();
        let attributes = Attributes::from_iter([(Attribute::ContentType, "text/plain".into())]);
        let result = backend
            .put_opts(
                &path,
                [Bytes::from_static(b"ab"), Bytes::from_static(b"cd")]
                    .into_iter()
                    .collect(),
                PutOptions {
                    mode: crate::object_store::PutMode::Create,
                    attributes,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let response = backend
            .get_opts(
                &path,
                GetOptions {
                    if_match: result.e_tag.clone(),
                    range: Some(1..3),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(response.meta.location, path);
        assert_eq!(response.range, 1..3);
        assert_eq!(
            response
                .attributes
                .get(&Attribute::ContentType)
                .unwrap()
                .as_ref(),
            "text/plain"
        );
        assert_eq!(response.bytes().await.unwrap().as_ref(), b"bc");
        assert!(matches!(
            backend.get(&Path::from("data/literal/key")).await,
            Err(Error::NotFound { .. })
        ));
        assert!(matches!(
            backend
                .put_opts(
                    &path,
                    Bytes::new().into(),
                    crate::object_store::PutMode::Create.into()
                )
                .await,
            Err(Error::AlreadyExists { .. })
        ));
        assert!(matches!(
            backend
                .get_opts(
                    &path,
                    GetOptions {
                        if_match: Some("wrong".into()),
                        ..Default::default()
                    }
                )
                .await,
            Err(Error::Precondition { .. })
        ));
        assert!(matches!(
            backend
                .get_opts(
                    &path,
                    GetOptions {
                        if_none_match: result.e_tag,
                        ..Default::default()
                    }
                )
                .await,
            Err(Error::NotModified { .. })
        ));

        assert!(matches!(
            backend
                .put_opts(
                    &path,
                    Bytes::from_static(b"new").into(),
                    crate::object_store::PutMode::Update(crate::object_store::UpdateVersion {
                        e_tag: Some("wrong".into()),
                        version: None
                    })
                    .into()
                )
                .await,
            Err(Error::Precondition { .. })
        ));
        let current = backend.head(&path).await.unwrap();
        backend
            .put_opts(
                &path,
                Bytes::from_static(b"new").into(),
                crate::object_store::PutMode::Update(crate::object_store::UpdateVersion {
                    e_tag: current.e_tag,
                    version: current.version,
                })
                .into(),
            )
            .await
            .unwrap();
        assert_eq!(
            backend
                .get(&path)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap()
                .as_ref(),
            b"new"
        );

        let input = futures::stream::once(async {
            Err(Error::Unauthenticated {
                path: "forbidden".into(),
                source: "request credentials expired".into(),
            })
        })
        .boxed();
        let error = backend.delete_stream(input).try_next().await.unwrap_err();
        assert!(matches!(error, Error::Unauthenticated { .. }));
        assert_eq!(
            error.source().unwrap().to_string(),
            "request credentials expired"
        );

        let native = ::object_store::GetResult {
            payload: ::object_store::GetResultPayload::Stream(
                futures::stream::once(async {
                    Err(::object_store::Error::PermissionDenied {
                        path: "data/literal%2Fkey".into(),
                        source: "denied while streaming".into(),
                    })
                })
                .boxed(),
            ),
            meta: ::object_store::ObjectMeta {
                location: ::object_store::path::Path::parse(path.as_ref()).unwrap(),
                last_modified: chrono::Utc::now(),
                size: 4,
                e_tag: Some("opaque-etag".into()),
                version: Some("opaque-generation".into()),
            },
            range: 0..4,
            attributes: Default::default(),
            extensions: Default::default(),
        };
        let response = GetResult::try_from(native).unwrap();
        assert_eq!(response.meta.e_tag.as_deref(), Some("opaque-etag"));
        assert_eq!(response.meta.version.as_deref(), Some("opaque-generation"));
        let error = response.bytes().await.unwrap_err();
        assert!(matches!(error, Error::PermissionDenied { .. }));
        assert_eq!(
            error.source().unwrap().to_string(),
            "denied while streaming"
        );
    }

    #[tokio::test]
    async fn pagination_preserves_opaque_tokens_paths_and_extensions() {
        #[derive(Clone, Debug, PartialEq)]
        struct Context(u64);
        struct Page;
        #[async_trait]
        impl ::object_store::list::PaginatedListStore for Page {
            async fn list_paginated(
                &self,
                prefix: Option<&str>,
                options: ::object_store::list::PaginatedListOptions,
            ) -> ::object_store::Result<::object_store::list::PaginatedListResult> {
                assert_eq!(prefix, Some("keys/%2F"));
                assert_eq!(options.page_token.as_deref(), Some("opaque+/=%2F"));
                assert_eq!(options.offset.as_deref(), Some("keys/%2Fa"));
                assert_eq!(options.delimiter.as_deref(), Some("/"));
                assert_eq!(options.max_keys, Some(7));
                assert_eq!(options.extensions.get::<Context>(), Some(&Context(42)));
                Ok(::object_store::list::PaginatedListResult {
                    page_token: Some("next+/=%2F".into()),
                    result: ::object_store::ListResult {
                        common_prefixes: vec![
                            ::object_store::path::Path::parse("keys/%2Fb").unwrap(),
                        ],
                        objects: vec![],
                        extensions: options.extensions,
                    },
                })
            }
        }
        let mut extensions = http::Extensions::new();
        extensions.insert(Context(42));
        let result = from_object_store_with_pagination(
            Arc::new(::object_store::memory::InMemory::new()),
            Arc::new(Page),
        )
        .list_paginated(
            Some("keys/%2F"),
            crate::object_store::PaginatedListOptions {
                page_token: Some("opaque+/=%2F".into()),
                offset: Some("keys/%2Fa".into()),
                delimiter: Some("/".into()),
                max_keys: Some(7),
                extensions,
            },
        )
        .await
        .unwrap();
        assert_eq!(result.page_token.as_deref(), Some("next+/=%2F"));
        assert_eq!(
            result.result.common_prefixes,
            [Path::parse("keys/%2Fb").unwrap()]
        );
        assert_eq!(
            result.result.extensions.get::<Context>(),
            Some(&Context(42))
        );
    }

    #[rstest::rstest]
    #[case::bounded(2, 7, Some(b"23456".as_slice()))]
    #[case::short_read(8, 20, Some(b"89".as_slice()))]
    #[case::empty(2, 2, None)]
    #[case::reversed(7, 2, None)]
    #[case::past_eof(10, 12, None)]
    #[tokio::test]
    async fn adapter_keeps_local_file_payload_and_requested_range(
        #[case] start: u64,
        #[case] end: u64,
        #[case] expected: Option<&[u8]>,
    ) {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("file"), b"0123456789").unwrap();
        let backend = from_object_store(Arc::new(
            ::object_store::local::LocalFileSystem::new_with_prefix(directory.path()).unwrap(),
        ));
        let response = backend
            .get_opts(
                &"file".into(),
                GetOptions {
                    range: Some(start..end),
                    ..Default::default()
                },
            )
            .await;
        let Some(expected) = expected else {
            assert!(response.is_err());
            return;
        };
        let response = response.unwrap();
        assert!(matches!(
            response.payload,
            crate::object_store::GetResultPayload::File(..)
        ));
        assert_eq!(response.bytes().await.unwrap().as_ref(), expected);
    }
}
