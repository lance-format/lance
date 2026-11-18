// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::object_store::backend::MultipartUpload;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use crate::object_store::path::Path;
use crate::object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, ObjectMeta, PutMultipartOptions, PutOptions,
    PutPayload, PutResult, RenameOptions, StorageBackend,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, future, stream::BoxStream};
use object_store_opendal::OpendalStore as InnerOpendalStore;
use opendal::Operator;

/// Adapts OpenDAL listing paths to the spelling used by the request.
///
/// The upstream bridge builds listed locations with [`Path::from`], which
/// percent-encodes reserved characters. Lance builds dataset base paths with
/// [`Path::from_url_path`], so mismatched listed locations must be decoded.
/// Locations that already match the requested prefix retain their spelling to
/// preserve paths containing literal percent escapes.
#[derive(Debug, Clone)]
pub(super) struct OpendalStore {
    inner: Arc<dyn StorageBackend>,
    list_with_start_after: bool,
}

impl OpendalStore {
    pub(super) fn new(operator: Operator) -> Self {
        Self {
            list_with_start_after: operator.info().capability().list_with_start_after,
            inner: crate::object_store::backend::native::from_object_store(Arc::new(
                InnerOpendalStore::new(operator),
            )),
        }
    }
}

impl fmt::Display for OpendalStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(formatter)
    }
}

fn normalize_location(location: &Path, prefix: Option<&Path>) -> crate::object_store::Result<Path> {
    if prefix.is_none_or(|prefix| location.prefix_matches(prefix)) {
        return Ok(location.clone());
    }

    Path::from_url_path(location.as_ref()).map_err(|source| crate::object_store::Error::Generic {
        store: "OpendalStore",
        source: Box::new(source),
    })
}

fn normalize_object_meta(
    mut meta: ObjectMeta,
    prefix: Option<&Path>,
) -> crate::object_store::Result<ObjectMeta> {
    meta.location = normalize_location(&meta.location, prefix)?;
    Ok(meta)
}

#[async_trait]
impl StorageBackend for OpendalStore {
    async fn list_paginated(
        &self,
        _prefix: Option<&str>,
        _options: crate::object_store::PaginatedListOptions,
    ) -> crate::object_store::Result<crate::object_store::PaginatedListResult> {
        Err(crate::object_store::Error::NotSupported {
            source: "OpenDAL pagination is unavailable".into(),
        })
    }

    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> crate::object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, options).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        options: PutMultipartOptions,
    ) -> crate::object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, options).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> crate::object_store::Result<GetResult> {
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> crate::object_store::Result<Vec<Bytes>> {
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, crate::object_store::Result<Path>>,
    ) -> BoxStream<'static, crate::object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> BoxStream<'static, crate::object_store::Result<ObjectMeta>> {
        let listed = self.inner.list(prefix);
        let prefix = prefix.cloned();
        listed
            .map(move |result| result.and_then(|meta| normalize_object_meta(meta, prefix.as_ref())))
            .boxed()
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, crate::object_store::Result<ObjectMeta>> {
        if self.list_with_start_after {
            let listed = self.inner.list_with_offset(prefix, offset);
            let prefix = prefix.cloned();
            listed
                .map(move |result| {
                    result.and_then(|meta| normalize_object_meta(meta, prefix.as_ref()))
                })
                .boxed()
        } else {
            // The bridge's fallback compares its encoded output with the raw
            // offset. Filter normalized locations so both sides use one form.
            let offset = offset.clone();
            self.list(prefix)
                .try_filter(move |meta| future::ready(meta.location > offset))
                .boxed()
        }
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&Path>,
    ) -> crate::object_store::Result<ListResult> {
        let mut result = self.inner.list_with_delimiter(prefix).await?;
        for object in &mut result.objects {
            object.location = normalize_location(&object.location, prefix)?;
        }
        for common_prefix in &mut result.common_prefixes {
            *common_prefix = normalize_location(common_prefix, prefix)?;
        }
        Ok(result)
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> crate::object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }

    async fn rename_opts(
        &self,
        from: &Path,
        to: &Path,
        options: RenameOptions,
    ) -> crate::object_store::Result<()> {
        self.inner.rename_opts(from, to, options).await
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use futures::TryStreamExt;

    use opendal::services::Memory;
    use rstest::rstest;

    use super::*;

    #[cfg(feature = "tencent")]
    #[rstest]
    #[case::put("put")]
    #[case::copy("copy")]
    #[case::rename("rename")]
    #[tokio::test]
    async fn cos_forwards_conditional_operations(#[case] operation: &str) {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        opendal::install_default();
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/target"))
            .and(header("x-cos-forbid-overwrite", "true"))
            .respond_with(ResponseTemplate::new(200).insert_header("etag", "test")
                .set_body_string("<CopyObjectResult><ETag>test</ETag><LastModified>2026-01-01T00:00:00Z</LastModified></CopyObjectResult>"))
            .expect(1)
            .mount(&server).await;
        Mock::given(method("HEAD"))
            .and(path("/source"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-length", "4")
                    .insert_header("etag", "source")
                    .insert_header("last-modified", "Wed, 01 Jan 2025 00:00:00 GMT"),
            )
            .expect(if operation == "put" { 0 } else { 1 })
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path("/source"))
            .respond_with(ResponseTemplate::new(204))
            .expect(if operation == "rename" { 1 } else { 0 })
            .mount(&server)
            .await;
        let operator = Operator::from_iter::<opendal::services::Cos>([
            // COS prefixes the endpoint with the bucket; retain the loopback host.
            ("bucket".to_owned(), "127".to_owned()),
            ("endpoint".to_owned(), server.uri()),
            ("secret_id".to_owned(), "test".to_owned()),
            ("secret_key".to_owned(), "test".to_owned()),
        ])
        .unwrap();
        let store = OpendalStore::new(operator);
        let source = Path::from("source");
        let target = Path::from("target");
        match operation {
            "put" => {
                store
                    .put_opts(
                        &target,
                        Bytes::from_static(b"data").into(),
                        crate::object_store::PutMode::Create.into(),
                    )
                    .await
                    .unwrap();
            }
            "copy" => store
                .copy_opts(
                    &source,
                    &target,
                    CopyOptions {
                        mode: crate::object_store::CopyMode::Create,
                        ..Default::default()
                    },
                )
                .await
                .unwrap(),
            "rename" => store
                .rename_opts(
                    &source,
                    &target,
                    RenameOptions {
                        target_mode: crate::object_store::RenameTargetMode::Create,
                        ..Default::default()
                    },
                )
                .await
                .unwrap(),
            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case::raw_reserved_character("tables/run~1/t.lance")]
    #[case::literal_percent_escape("tables/run%25231/t.lance")]
    #[tokio::test]
    async fn test_list_preserves_request_path_spelling(#[case] base_url: &str) {
        let operator = Operator::new(Memory::default()).unwrap();
        let store = OpendalStore::new(operator);
        let store = &store as &dyn StorageBackend;
        let base = Path::from_url_path(base_url).unwrap();
        let direct_location = base.clone().join("manifest.lance");
        let nested_location = Path::from_url_path(format!("{base_url}/data/part.lance")).unwrap();
        for location in [&direct_location, &nested_location] {
            store
                .put(location, Bytes::from_static(b"data").into())
                .await
                .unwrap();
        }

        let listed = store
            .list(Some(&base))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut listed_locations = listed
            .into_iter()
            .map(|meta| meta.location)
            .collect::<Vec<_>>();
        listed_locations.sort();
        let mut expected_locations = vec![direct_location.clone(), nested_location.clone()];
        expected_locations.sort();
        assert_eq!(listed_locations, expected_locations);
        assert!(
            listed_locations
                .iter()
                .all(|location| location.prefix_matches(&base))
        );

        let listed_after_nested = store
            .list_with_offset(Some(&base), &nested_location)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(listed_after_nested.len(), 1);
        assert_eq!(listed_after_nested[0].location, direct_location);

        let delimited = store.list_with_delimiter(Some(&base)).await.unwrap();
        assert_eq!(delimited.objects.len(), 1);
        assert_eq!(delimited.objects[0].location, direct_location);
        assert_eq!(delimited.common_prefixes, vec![base.clone().join("data")]);
    }
}
