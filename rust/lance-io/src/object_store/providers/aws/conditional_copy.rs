// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Conditional S3 copies using only the published object_store API.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use http::{HeaderValue, header::IF_NONE_MATCH};
use object_store::ClientOptions;
use object_store::aws::{AmazonS3, AmazonS3Builder};
use object_store::multipart::MultipartStore;
use object_store::path::Path;
use object_store::{
    CopyMode, CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, ObjectStoreExt, PutMode, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    RenameOptions, Result,
};

/// S3's whole-object copy and UploadPart size limit.
const MAX_COPY_SIZE: u64 = 5 * 1024 * 1024 * 1024;
/// Bound client memory while leaving room for S3's 10,000-part limit.
const MIN_PART_SIZE: u64 = 8 * 1024 * 1024;

#[derive(Debug)]
pub(super) struct ConditionalS3Store {
    inner: Arc<AmazonS3>,
    /// Used ONLY for completion: its default If-None-Match header makes the
    /// published MultipartStore::complete_multipart operation conditional.
    completion: AmazonS3,
}

impl fmt::Display for ConditionalS3Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl ConditionalS3Store {
    pub(super) fn new(
        inner: Arc<AmazonS3>,
        builder: AmazonS3Builder,
        client_options: ClientOptions,
    ) -> Result<Self> {
        // Only completion calls use this client. In particular, an ordinary
        // overwrite must never inherit the conditional completion header.
        let mut headers = client_options
            .get_default_headers()
            .cloned()
            .unwrap_or_default();
        headers.insert(IF_NONE_MATCH, HeaderValue::from_static("*"));
        let completion = builder
            .with_client_options(client_options.with_default_headers(headers))
            .build()?;
        Ok(Self { inner, completion })
    }

    async fn copy_large(&self, from: &Path, to: &Path, size: u64, part_size: u64) -> Result<()> {
        let upload_id = self.inner.create_multipart(to).await?;
        let result = async {
            let mut parts = Vec::with_capacity(size.div_ceil(part_size) as usize);
            for (index, start) in (0..size).step_by(part_size as usize).enumerate() {
                let end = start.saturating_add(part_size).min(size);
                let bytes = self.inner.get_range(from, start..end).await?;
                if bytes.len() as u64 != end - start {
                    return Err(object_store::Error::Generic {
                        store: "S3 conditional copy",
                        source: format!(
                            "source {from} range {start}..{end} returned {} bytes",
                            bytes.len()
                        )
                        .into(),
                    });
                }
                parts.push(
                    self.inner
                        .put_part(to, &upload_id, index, bytes.into())
                        .await?,
                );
            }
            self.completion
                .complete_multipart(to, &upload_id, parts)
                .await?;
            Ok(())
        }
        .await;
        if result.is_err()
            && let Err(error) = self.inner.abort_multipart(to, &upload_id).await
        {
            log::warn!("Failed to abort conditional copy to {to}: {error}");
        }
        result
    }
}

#[async_trait]
impl ObjectStore for ConditionalS3Store {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> Result<PutResult> {
        self.inner.put_opts(path, payload, options).await
    }

    async fn put_multipart_opts(
        &self,
        path: &Path,
        options: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(path, options).await
    }

    async fn get_opts(&self, path: &Path, options: GetOptions) -> Result<GetResult> {
        self.inner.get_opts(path, options).await
    }

    fn delete_stream(
        &self,
        paths: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.inner.delete_stream(paths)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        if options.mode == CopyMode::Create {
            let size = self.inner.head(from).await?.size;
            if size == 0 {
                self.inner
                    .put_opts(
                        to,
                        PutPayload::default(),
                        PutOptions {
                            mode: PutMode::Create,
                            ..Default::default()
                        },
                    )
                    .await?;
                return Ok(());
            }
            if size > MAX_COPY_SIZE {
                let part_size = MIN_PART_SIZE.max(size.div_ceil(10_000));
                return self.copy_large(from, to, size, part_size).await;
            }
        }
        self.inner.copy_opts(from, to, options).await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, options: RenameOptions) -> Result<()> {
        self.inner.rename_opts(from, to, options).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use object_store::{RetryConfig, aws::S3CopyIfNotExists};
    use rstest::rstest;
    use wiremock::matchers::{header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::object_store::providers::aws::AwsStoreProvider;
    use crate::object_store::{
        ObjectStore as LanceObjectStore, ObjectStoreParams, ObjectStoreProvider,
        StorageOptionsAccessor,
    };

    async fn provider_store(server: &MockServer, mode: Option<&str>) -> LanceObjectStore {
        let mut options = HashMap::from([
            ("aws_endpoint".into(), server.uri()),
            ("aws_region".into(), "us-east-1".into()),
            ("aws_access_key_id".into(), "test".into()),
            ("aws_secret_access_key".into(), "test".into()),
            ("aws_allow_http".into(), "true".into()),
            ("client_max_retries".into(), "0".into()),
            ("lance_aimd_max_retries".into(), "0".into()),
        ]);
        if let Some(mode) = mode {
            options.insert("aws_copy_if_not_exists".into(), mode.into());
        }
        AwsStoreProvider
            .new_store(
                "s3://bucket".parse().unwrap(),
                &ObjectStoreParams {
                    storage_options_accessor: Some(Arc::new(
                        StorageOptionsAccessor::with_static_options(options),
                    )),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
    }

    async fn source_metadata(server: &MockServer, size: u64) {
        Mock::given(method("HEAD"))
            .and(path("/bucket/source"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-length", size.to_string())
                    .insert_header("last-modified", "Tue, 27 Oct 2026 00:00:00 GMT")
                    .insert_header("etag", "\"source\""),
            )
            .mount(server)
            .await;
    }

    async fn initiate(server: &MockServer) {
        Mock::given(method("POST"))
            .and(path("/bucket/destination"))
            .and(query_param("uploads", ""))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "<InitiateMultipartUploadResult><UploadId>upload-1</UploadId></InitiateMultipartUploadResult>"))
            .expect(1)
            .mount(server).await;
    }

    fn completed() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_string(
            "<CompleteMultipartUploadResult><ETag>\"destination\"</ETag></CompleteMultipartUploadResult>")
    }

    #[rstest]
    #[case::default(None)]
    #[case::multipart(Some("multipart"))]
    #[tokio::test]
    async fn default_s3_conditional_promotion(#[case] mode: Option<&str>) {
        let server = MockServer::start().await;
        source_metadata(&server, 3).await;
        initiate(&server).await;
        Mock::given(method("PUT"))
            .and(query_param("partNumber", "1"))
            .and(header("x-amz-copy-source", "bucket/source"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("<CopyPartResult><ETag>\"part\"</ETag></CopyPartResult>"),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(query_param("uploadId", "upload-1"))
            .and(header("if-none-match", "*"))
            .respond_with(completed())
            .expect(1)
            .mount(&server)
            .await;
        let store = provider_store(&server, mode).await;
        let result = store
            .copy_if_not_exists(&Path::from("source"), &Path::from("destination"))
            .await
            .unwrap();
        assert_eq!(result.size, 3);
    }

    #[tokio::test]
    async fn explicit_conditional_copy_header_is_preserved() {
        let server = MockServer::start().await;
        source_metadata(&server, 3).await;
        Mock::given(method("PUT"))
            .and(path("/bucket/destination"))
            .and(header("cf-copy-destination-if-none-match", "*"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "<CopyObjectResult><ETag>\"destination\"</ETag></CopyObjectResult>",
            ))
            .expect(1)
            .mount(&server)
            .await;
        provider_store(&server, Some("header:cf-copy-destination-if-none-match:*"))
            .await
            .copy_if_not_exists(&Path::from("source"), &Path::from("destination"))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn empty_s3_conditional_promotion() {
        let server = MockServer::start().await;
        source_metadata(&server, 0).await;
        Mock::given(method("PUT"))
            .and(path("/bucket/destination"))
            .and(header("if-none-match", "*"))
            .respond_with(ResponseTemplate::new(200).insert_header("etag", "\"empty\""))
            .expect(1)
            .mount(&server)
            .await;
        let result = provider_store(&server, None)
            .await
            .copy_if_not_exists(&Path::from("source"), &Path::from("destination"))
            .await
            .unwrap();
        assert_eq!(result.size, 0);
    }

    #[rstest]
    #[case::default(None)]
    #[case::multipart(Some("multipart"))]
    #[tokio::test]
    async fn large_s3_conditional_promotion_uses_bounded_reads(#[case] mode: Option<&str>) {
        let server = MockServer::start().await;
        source_metadata(&server, 6 * 1024 * 1024 * 1024).await;
        initiate(&server).await;
        // Stop after the first bounded GET: no multi-gigabyte fixture is needed
        // to verify the public helper routes large copies through this path.
        Mock::given(method("GET"))
            .and(path("/bucket/source"))
            .and(header("range", "bytes=0-8388607"))
            .respond_with(ResponseTemplate::new(403))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(query_param("uploadId", "upload-1"))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount(&server)
            .await;
        let error = provider_store(&server, mode)
            .await
            .copy_if_not_exists(&Path::from("source"), &Path::from("destination"))
            .await
            .unwrap_err();
        assert!(matches!(error, lance_core::Error::IO { .. }));
        assert!(error.to_string().contains("403"), "{error}");
        assert!(
            server
                .received_requests()
                .await
                .unwrap()
                .iter()
                .all(|request| request.method.as_str() != "PUT")
        );
    }

    #[tokio::test]
    async fn ordinary_s3_writes_remain_unconditional() {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/bucket/destination"))
            .respond_with(ResponseTemplate::new(200).insert_header("etag", "\"overwrite\""))
            .expect(1)
            .mount(&server)
            .await;
        provider_store(&server, None)
            .await
            .put(&Path::from("destination"), b"overwrite")
            .await
            .unwrap();
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests[0].body, b"overwrite");
        assert!(!requests[0].headers.contains_key("if-none-match"));
    }

    #[rstest]
    #[case::success(200, false)]
    #[case::existing_destination(412, false)]
    #[case::part_failure(403, false)]
    #[case::completion_failure(409, false)]
    #[case::embedded_completion_error(200, true)]
    #[tokio::test]
    async fn multipart_copy_preserves_conditional_completion(
        #[case] outcome: u16,
        #[case] is_embedded_error: bool,
    ) {
        let server = MockServer::start().await;
        initiate(&server).await;
        // Scale part sizes down while exercising the same upload loop and real
        // S3 client. The provider-level test above checks the 5 GiB dispatch.
        for (range, content_range, body) in [
            ("bytes=0-4", "bytes 0-4/8", b"first".as_slice()),
            ("bytes=5-7", "bytes 5-7/8", b"end".as_slice()),
        ] {
            Mock::given(method("GET"))
                .and(path("/bucket/source"))
                .and(header("range", range))
                .respond_with(
                    ResponseTemplate::new(206)
                        .insert_header("content-range", content_range)
                        .insert_header("last-modified", "Tue, 27 Oct 2026 00:00:00 GMT")
                        .insert_header("etag", "\"source\"")
                        .set_body_bytes(body),
                )
                .mount(&server)
                .await;
        }
        Mock::given(method("PUT"))
            .and(query_param("uploadId", "upload-1"))
            .respond_with(if outcome == 403 {
                ResponseTemplate::new(403)
            } else {
                ResponseTemplate::new(200).insert_header("etag", "\"part\"")
            })
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(query_param("uploadId", "upload-1"))
            .and(header("if-none-match", "*"))
            .respond_with(match (outcome, is_embedded_error) {
                (200, false) => completed(),
                (_, true) => ResponseTemplate::new(200).set_body_string(
                    "<Error><Code>AccessDenied</Code><Message>denied</Message></Error>",
                ),
                (code, false) => ResponseTemplate::new(code),
            })
            .expect(if outcome == 403 { 0 } else { 1 })
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(query_param("uploadId", "upload-1"))
            .respond_with(ResponseTemplate::new(204))
            .expect(if outcome == 200 && !is_embedded_error {
                0
            } else {
                1
            })
            .mount(&server)
            .await;
        let options = ClientOptions::new().with_allow_http(true);
        let builder = AmazonS3Builder::new()
            .with_bucket_name("bucket")
            .with_region("us-east-1")
            .with_access_key_id("test")
            .with_secret_access_key("test")
            .with_endpoint(server.uri())
            .with_client_options(options.clone())
            .with_copy_if_not_exists(S3CopyIfNotExists::Multipart)
            .with_retry(RetryConfig {
                max_retries: 0,
                ..Default::default()
            });
        let store =
            ConditionalS3Store::new(Arc::new(builder.clone().build().unwrap()), builder, options)
                .unwrap();
        let result = store
            .copy_large(&Path::from("source"), &Path::from("destination"), 8, 5)
            .await;
        if outcome == 200 && !is_embedded_error {
            result.unwrap();
        } else {
            let error = result.unwrap_err();
            if outcome == 412 {
                assert!(
                    matches!(error, object_store::Error::Precondition { .. }),
                    "{error}"
                );
            }
        }
        let requests = server.received_requests().await.unwrap();
        for request in &requests {
            if request.method.as_str() == "PUT"
                || request.method.as_str() == "GET"
                || request.url.query() == Some("uploads=")
            {
                assert!(
                    !request.headers.contains_key("if-none-match"),
                    "{request:?}"
                );
            }
        }
        let parts: Vec<_> = requests
            .iter()
            .filter(|request| request.method.as_str() == "PUT")
            .collect();
        assert_eq!(parts[0].body, b"first");
        if outcome != 403 {
            assert_eq!(parts[1].body, b"end");
            let completion = requests
                .iter()
                .find(|request| {
                    request.method.as_str() == "POST"
                        && request.headers.contains_key("if-none-match")
                })
                .unwrap();
            let body = String::from_utf8_lossy(&completion.body);
            assert!(body.contains("<PartNumber>1</PartNumber>"), "{body}");
            assert!(body.contains("<PartNumber>2</PartNumber>"), "{body}");
        }
    }
}
