// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Conditional S3 multipart completion using the published object_store API.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{
    StreamExt, TryStreamExt,
    stream::{self, BoxStream},
};
use http::{
    HeaderName, HeaderValue,
    header::{IF_MATCH, IF_NONE_MATCH},
};
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

pub(super) struct ConditionalS3Store {
    inner: Arc<AmazonS3>,
    builder: AmazonS3Builder,
    client_options: ClientOptions,
    copy_completion: Option<AmazonS3>,
    single_update: tokio::sync::OnceCell<AmazonS3>,
}

impl fmt::Debug for ConditionalS3Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The builder can contain static credentials.
        f.debug_struct("ConditionalS3Store")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
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
        multipart_copy: bool,
    ) -> Result<Self> {
        let mut store = Self {
            inner,
            builder,
            client_options,
            copy_completion: None,
            single_update: Default::default(),
        };
        if multipart_copy {
            store.copy_completion = Some(
                store
                    .completion_builder(IF_NONE_MATCH, HeaderValue::from_static("*"))
                    .build()?,
            );
        }
        Ok(store)
    }

    // Only completion uses this client. Upload creation, individual parts and
    // ordinary overwrites must never inherit the destination precondition.
    fn completion_builder(&self, name: HeaderName, value: HeaderValue) -> AmazonS3Builder {
        let mut headers = self
            .client_options
            .get_default_headers()
            .cloned()
            .unwrap_or_default();
        headers.insert(name, value);
        self.builder
            .clone()
            .with_client_options(self.client_options.clone().with_default_headers(headers))
    }

    async fn copy_large(
        &self,
        completion: &AmazonS3,
        from: &Path,
        to: &Path,
        size: u64,
        part_size: u64,
    ) -> Result<()> {
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
            completion.complete_multipart(to, &upload_id, parts).await?;
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

    async fn update_large(
        &self,
        path: &Path,
        payload: PutPayload,
        options: &PutOptions,
        update: &super::MultipartUpdate,
    ) -> Result<PutResult> {
        let header =
            HeaderValue::from_str(&update.etag).map_err(|source| object_store::Error::Generic {
                store: "S3 conditional update",
                source: Box::new(source),
            })?;
        // A lost success response makes the outcome unknown. Replaying completion
        // may return NoSuchUpload or a mismatch against our own published ETag.
        // The SDK accepts the signed header on a separate client. Large tag
        // updates pay for another pool and, on S3 Express, a session.
        let completion = self
            .completion_builder(IF_MATCH, header)
            .with_retry(object_store::RetryConfig {
                max_retries: 0,
                ..Default::default()
            })
            .build()?;
        let upload_id = self
            .inner
            .create_multipart_opts(
                path,
                PutMultipartOptions {
                    tags: options.tags.clone(),
                    attributes: options.attributes.clone(),
                    extensions: options.extensions.clone(),
                },
            )
            .await?;
        let pending = super::multipart_update::PendingUpload::new(
            (self.inner.clone(), path.clone(), upload_id.clone()),
            |(store, path, id)| {
                Box::pin(async move {
                    if let Err(error) = store.abort_multipart(&path, &id).await {
                        log::warn!("Failed to abort conditional update to {path}: {error}");
                    }
                })
            },
        );
        let chunks =
            opendal::Buffer::chunks(&opendal::Buffer::from_iter(payload), update.part_size);
        let result = async {
            let parts = stream::iter(chunks.enumerate())
                .map(|(index, chunk)| {
                    self.inner
                        .put_part(path, &upload_id, index, PutPayload::from_iter(chunk))
                })
                .buffered(update.concurrency)
                .try_collect()
                .await?;
            completion.complete_multipart(path, &upload_id, parts).await
        }
        .await;
        pending.finish(result).await
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
        use super::ConditionalUpdate;
        if let Some(update) = options.extensions.get::<ConditionalUpdate>() {
            let ConditionalUpdate::Multipart(update) = update else {
                // Share one client across small conditional updates. Generic puts
                // keep the ordinary client's configured retry and encryption policy.
                let client = self
                    .single_update
                    .get_or_try_init(|| async {
                        self.builder
                            .clone()
                            .with_retry(object_store::RetryConfig {
                                max_retries: 0,
                                ..Default::default()
                            })
                            .build()
                    })
                    .await?;
                return client.put_opts(path, payload, options).await;
            };
            return self
                .update_large(path, payload, &options, update)
                .await
                .map_err(|error| match error {
                    // The missing resource could be an upload ID, not the tag.
                    object_store::Error::NotFound { ref source, .. } => {
                        // The SDK's typed service response error is private.
                        let response = source.source().map(ToString::to_string);
                        let code = response
                            .as_deref()
                            .and_then(|s| {
                                s.strip_prefix(
                                    "Server returned non-2xx status code: 404 Not Found: ",
                                )
                            })
                            .and_then(crate::object_store::throttle::error_document_code);
                        if code.as_deref() == Some("NoSuchKey") {
                            object_store::Error::Precondition {
                                path: path.to_string(),
                                source: Box::new(error),
                            }
                        } else {
                            object_store::Error::Generic {
                                store: "S3 multipart update",
                                source: Box::new(error),
                            }
                        }
                    }
                    error => error,
                });
        }
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
        if options.mode == CopyMode::Create
            && let Some(completion) = &self.copy_completion
        {
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
                return self.copy_large(completion, from, to, size, part_size).await;
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

    async fn provider_store(
        server: &MockServer,
        mode: Option<&str>,
        opendal: bool,
    ) -> LanceObjectStore {
        let mut options = HashMap::from([
            ("endpoint".into(), server.uri()),
            ("region".into(), "us-east-1".into()),
            ("access_key_id".into(), "test".into()),
            ("secret_access_key".into(), "test".into()),
            ("allow_http".into(), "true".into()),
            ("use_opendal".into(), opendal.to_string()),
            ("client_max_retries".into(), "2".into()),
            ("lance_aimd_max_retries".into(), "2".into()),
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

    fn multipart_fixture_options() -> PutOptions {
        let etag = "\"original\"";
        let mut options: PutOptions = PutMode::Update(object_store::UpdateVersion {
            e_tag: Some(etag.into()),
            version: None,
        })
        .into();
        // Keep wire fixtures small regardless of the process upload threshold.
        options
            .extensions
            .insert(super::super::ConditionalUpdate::Multipart(
                super::super::MultipartUpdate {
                    etag: etag.into(),
                    part_size: 5 * 1024 * 1024,
                    concurrency: 1,
                },
            ));
        options
    }

    #[rstest]
    #[case::applied(200, "", false)]
    #[case::stale(412, "PreconditionFailed", true)]
    #[case::concurrent(409, "ConditionalRequestConflict", true)]
    #[case::locked(409, "BucketAlreadyOwnedByYou", false)]
    #[case::denied(403, "AccessDenied", false)]
    #[case::expired_upload(404, "NoSuchUpload", false)]
    #[case::deleted_destination(404, "NoSuchKey", true)]
    #[case::ambiguous_completion(503, "SlowDown", false)]
    #[tokio::test]
    async fn multipart_update_checks_etag_at_completion(
        #[case] status: u16,
        #[case] code: &str,
        #[case] conflict: bool,
        #[values(false, true)] opendal: bool,
    ) {
        let server = MockServer::start().await;
        initiate(&server).await;
        let retry_first_part = std::sync::atomic::AtomicBool::new(status == 200);
        Mock::given(method("PUT"))
            .and(query_param("uploadId", "upload-1"))
            .respond_with(move |request: &wiremock::Request| {
                if request
                    .url
                    .query_pairs()
                    .any(|(k, v)| k == "partNumber" && v == "1")
                    && retry_first_part.swap(false, std::sync::atomic::Ordering::Relaxed)
                {
                    ResponseTemplate::new(503)
                        .set_body_string("<Error><Code>SlowDown</Code></Error>")
                } else {
                    ResponseTemplate::new(200).insert_header("etag", "\"part\"")
                }
            })
            .expect(if status == 200 { 3 } else { 2 })
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(query_param("uploadId", "upload-1"))
            .and(header("if-match", "\"original\""))
            .respond_with(if status == 200 {
                completed()
            } else {
                ResponseTemplate::new(status).set_body_string(format!(
                    "<Error><Code>{code}</Code><Message>write rejected</Message></Error>"
                ))
            })
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(query_param("uploadId", "upload-1"))
            .respond_with(ResponseTemplate::new(204))
            .expect(u64::from(status != 200))
            .mount(&server)
            .await;
        // A custom copy policy must not disable conditional update completion.
        let store = provider_store(
            &server,
            Some("header:cf-copy-destination-if-none-match:*"),
            opendal,
        )
        .await;
        let first = bytes::Bytes::from(vec![b'a'; 5 * 1024 * 1024]);
        let payload = PutPayload::from_iter([first.clone(), bytes::Bytes::from_static(b"b")]);
        let result = store
            .put_payload_opts(
                &Path::from("destination"),
                payload,
                multipart_fixture_options(),
            )
            .await;
        if status == 200 {
            assert_eq!(result.unwrap().size, first.len() + 1);
        } else {
            let error = result.unwrap_err();
            assert_eq!(
                matches!(error, object_store::Error::Precondition { .. })
                    || super::super::is_conditional_put_conflict(&error),
                conflict,
                "{error}"
            );
            assert!(
                !matches!(error, object_store::Error::NotFound { .. }),
                "{error}"
            );
        }
        let requests = server.received_requests().await.unwrap();
        for request in requests {
            let part = request
                .url
                .query_pairs()
                .find(|(name, _)| name == "partNumber");
            if let Some((_, number)) = part {
                assert_eq!(
                    request.body.as_slice(),
                    if number == "1" { first.as_ref() } else { b"b" }
                );
            }
            let completion = request.method == http::Method::POST
                && request.url.query_pairs().any(|(key, _)| key == "uploadId");
            assert_eq!(request.headers.contains_key(IF_MATCH), completion);
            if completion && !opendal {
                let signature = request.headers[http::header::AUTHORIZATION]
                    .to_str()
                    .unwrap();
                assert!(signature.contains("if-match"), "{signature}");
            }
        }
    }

    #[rstest]
    #[case::timeout(400, "RequestTimeout", true)]
    #[case::in_progress(409, "OperationAborted", true)]
    #[case::invalid(400, "InvalidRequest", false)]
    #[case::locked(409, "ObjectLockedByBucketPolicy", false)]
    #[tokio::test]
    async fn opendal_retries_only_temporary_multipart_update_parts(
        #[case] status: u16,
        #[case] code: &'static str,
        #[case] retry: bool,
    ) {
        let server = MockServer::start().await;
        initiate(&server).await;
        let attempts = std::sync::atomic::AtomicUsize::new(0);
        Mock::given(method("PUT"))
            .and(query_param("partNumber", "1"))
            .respond_with(move |_: &wiremock::Request| {
                if attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
                    ResponseTemplate::new(status)
                        .set_body_string(format!("<Error><Code>{code}</Code></Error>"))
                } else {
                    ResponseTemplate::new(200).insert_header("etag", "\"part\"")
                }
            })
            .expect(if retry { 2 } else { 1 })
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .and(query_param("partNumber", "2"))
            .respond_with(ResponseTemplate::new(200).insert_header("etag", "\"tail\""))
            .expect(u64::from(retry))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(query_param("uploadId", "upload-1"))
            .and(header("if-match", "\"original\""))
            .respond_with(completed())
            .expect(u64::from(retry))
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(query_param("uploadId", "upload-1"))
            .respond_with(ResponseTemplate::new(204))
            .expect(u64::from(!retry))
            .mount(&server)
            .await;
        let store = provider_store(&server, None, true).await;
        let result = store
            .put_payload_opts(
                &Path::from("destination"),
                vec![0; 5 * 1024 * 1024 + 1].into(),
                multipart_fixture_options(),
            )
            .await;
        if retry {
            result.unwrap();
        } else {
            let error = result.unwrap_err();
            assert!(error.to_string().contains(code), "{error}");
            assert!(!matches!(error, object_store::Error::Precondition { .. }));
        }
    }

    #[rstest]
    #[tokio::test]
    async fn cancelled_multipart_update_aborts(#[values(false, true)] opendal: bool) {
        use std::time::Duration;
        // An already dispatched part may arrive after cancellation. Give it an
        // exclusive server so it cannot reach another test's pooled endpoint.
        let server = MockServer::builder().start().await;
        initiate(&server).await;
        let part = Mock::given(method("PUT"))
            .and(query_param("uploadId", "upload-1"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_secs(30))
                    .insert_header("etag", "\"part\""),
            )
            .expect(1..=2)
            .mount_as_scoped(&server)
            .await;
        let abort = Mock::given(method("DELETE"))
            .and(query_param("uploadId", "upload-1"))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount_as_scoped(&server)
            .await;
        let store = provider_store(&server, None, opendal).await;
        let write = tokio::spawn(async move {
            store
                .put_payload_opts(
                    &Path::from("destination"),
                    vec![0; 5 * 1024 * 1024 + 1].into(),
                    multipart_fixture_options(),
                )
                .await
        });
        tokio::time::timeout(Duration::from_secs(4), part.wait_until_satisfied())
            .await
            .unwrap();
        write.abort();
        assert!(write.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(4), abort.wait_until_satisfied())
            .await
            .unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn ordinary_large_conditional_put_remains_single_request(
        #[values(false, true)] opendal: bool,
    ) {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/bucket/destination"))
            .and(header("if-match", "\"original\""))
            .respond_with(ResponseTemplate::new(200).insert_header("etag", "\"next\""))
            .expect(2)
            .mount(&server)
            .await;
        let store = provider_store(&server, None, opendal).await;
        let path = Path::from("destination");
        let mode = PutMode::Update(object_store::UpdateVersion {
            e_tag: Some("\"original\"".into()),
            version: None,
        });
        let payload = PutPayload::from(vec![0; 5 * 1024 * 1024 + 1]);
        store
            .put_payload_opts(&path, payload.clone(), mode.clone().into())
            .await
            .unwrap();
        store
            .inner
            .put_opts(&path, payload, mode.into())
            .await
            .unwrap();
        assert!(
            server
                .received_requests()
                .await
                .unwrap()
                .iter()
                .all(|r| r.url.query().is_none())
        );
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
        let store = provider_store(&server, mode, false).await;
        let result = store
            .copy_if_not_exists_or_same_size(&Path::from("source"), &Path::from("destination"))
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
        provider_store(
            &server,
            Some("header:cf-copy-destination-if-none-match:*"),
            false,
        )
        .await
        .copy_if_not_exists_or_same_size(&Path::from("source"), &Path::from("destination"))
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
        let result = provider_store(&server, None, false)
            .await
            .copy_if_not_exists_or_same_size(&Path::from("source"), &Path::from("destination"))
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
        let error = provider_store(&server, mode, false)
            .await
            .copy_if_not_exists_or_same_size(&Path::from("source"), &Path::from("destination"))
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
        provider_store(&server, None, false)
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
        let store = ConditionalS3Store::new(
            Arc::new(builder.clone().build().unwrap()),
            builder,
            options,
            true,
        )
        .unwrap();
        let result = store
            .copy_large(
                store.copy_completion.as_ref().unwrap(),
                &Path::from("source"),
                &Path::from("destination"),
                8,
                5,
            )
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
