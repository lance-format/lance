// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::time::Duration;

use http::header::{ACCEPT_ENCODING, HeaderValue, RETRY_AFTER};
use http::{Request, Response, StatusCode};
use opendal::layers::RetryLayer;
use opendal::raw::{SystemTime, Timestamp};
use opendal::{Buffer, Error, ErrorKind, HttpBody, HttpTransport, HttpTransporter, Operator};

/// Largest server-provided `Retry-After` delay this transport waits for.
///
/// A larger delay is surfaced as a permanent error instead of retrying before
/// the server allowed it.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(300);

/// Parse the delay-seconds or HTTP-date form of `Retry-After`.
fn parse_retry_after(headers: &http::HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    let deadline = Timestamp::parse_rfc2822(value).ok()?;
    let deadline = SystemTime::from(deadline);
    Some(
        deadline
            .duration_since(SystemTime::now())
            .unwrap_or_default(),
    )
}

fn retry_after_exceeded(uri: &str, delay: Duration) -> Error {
    Error::new(
        ErrorKind::RateLimited,
        format!(
            "Hugging Face asked to retry after {}s, which exceeds the {}s retry bound",
            delay.as_secs(),
            MAX_RETRY_AFTER.as_secs()
        ),
    )
    .with_context("uri", uri)
}

/// Sends `Accept-Encoding: identity` unless the request already sets an encoding.
///
/// OpenDAL takes an object's size from the GET response. reqwest asks for gzip as
/// soon as any crate in the build enables its `gzip` feature, and some stores then
/// compress the body and drop `Content-Length`: Alibaba Cloud OSS does this for
/// `text/plain` objects on a full GET, and the read fails with "HTTP response does
/// not contain file content length".
///
/// When `honor_retry_after` is set, a `429` carrying `Retry-After` is held until
/// the server-directed delay has elapsed before the response is returned to the
/// retry layer. This is the only boundary where the header is available as a
/// typed response header; OpenDAL's retry layer only sees the error. The retry
/// layer keeps ownership of every attempt, so mixed responses share its
/// `client_max_retries` budget.
#[derive(Debug)]
struct IdentityEncodingTransport {
    inner: HttpTransporter,
    honor_retry_after: bool,
}

impl HttpTransport for IdentityEncodingTransport {
    async fn fetch(&self, mut req: Request<Buffer>) -> opendal::Result<Response<HttpBody>> {
        req.headers_mut()
            .entry(ACCEPT_ENCODING)
            .or_insert(HeaderValue::from_static("identity"));

        let uri = req.uri().to_string();
        let response = self.inner.fetch(req).await?;
        if !self.honor_retry_after || response.status() != StatusCode::TOO_MANY_REQUESTS {
            return Ok(response);
        }
        let Some(delay) = parse_retry_after(response.headers()) else {
            return Ok(response);
        };
        if delay > MAX_RETRY_AFTER {
            return Err(retry_after_exceeded(&uri, delay));
        }
        if delay.is_zero() {
            return Ok(response);
        }

        // Release the response body and its connection before waiting. The HF
        // service only reads error headers, so an empty body preserves the
        // error classification while avoiding a connection held across the delay.
        let (parts, _) = response.into_parts();
        tokio::time::sleep(delay).await;
        Ok(Response::from_parts(
            parts,
            HttpBody::new(futures::stream::empty::<opendal::Result<Buffer>>(), Some(0)),
        ))
    }
}

pub(in crate::object_store) fn finish_opendal_operator(
    operator: Operator,
    max_retries: usize,
) -> Operator {
    finish_opendal_operator_inner(operator, max_retries, false)
}

/// Finish a Hugging Face operator.
///
/// The transport waits out `Retry-After` on a `429` before the response reaches
/// OpenDAL's retry layer, so the first retry already honors the server-directed
/// delay and every attempt shares one `client_max_retries` budget.
pub(in crate::object_store) fn finish_huggingface_operator(
    operator: Operator,
    max_retries: usize,
) -> Operator {
    finish_opendal_operator_inner(operator, max_retries, true)
}

fn finish_opendal_operator_inner(
    operator: Operator,
    max_retries: usize,
    honor_retry_after: bool,
) -> Operator {
    // OpenDAL's auto-registration process constructor can emit an ARM64 branch
    // outside the +/-128 MiB range in large dynamic libraries. Initialize the
    // registry and default transport explicitly at the operator boundary.
    opendal::install_default();

    let base_context = operator.base_context();
    let context =
        base_context.with_http_transport(HttpTransporter::new(IdentityEncodingTransport {
            inner: base_context.http_transport().clone(),
            honor_retry_after: honor_retry_after && max_retries > 0,
        }));
    let operator = operator.with_context(context);

    if max_retries == 0 {
        return operator;
    }

    let retry_layer = RetryLayer::new()
        .with_max_times(max_retries)
        .with_min_delay(Duration::from_millis(100))
        .with_max_delay(Duration::from_secs(15))
        .with_factor(2.0)
        .with_jitter();

    operator.layer(retry_layer)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use bytes::Bytes;
    use object_store::ObjectStoreExt as _;
    use opendal::raw::oio;
    use opendal::raw::{
        OpCopy, OpCreateDir, OpList, OpPresign, OpRead, OpRename, OpStat, OpWrite, RpCreateDir,
        RpPresign, RpRename, RpStat, Service, ServiceInfo,
    };
    #[cfg(feature = "aws")]
    use opendal::services::S3;
    use opendal::{
        Buffer, Capability, Error, ErrorKind, Metadata, MetadataBuilder, OperationContext, Result,
    };
    use rstest::rstest;
    #[cfg(feature = "aws")]
    use wiremock::matchers::{header, method, path};
    #[cfg(feature = "aws")]
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    #[test]
    fn test_finish_opendal_operator_installs_defaults() {
        let operator = Operator::new(opendal::services::Memory::default())
            .expect("memory operator should be created");
        let _ = finish_opendal_operator(operator, 0);

        let operator =
            Operator::via_iter("memory", []).expect("memory service should be registered");
        assert_eq!(operator.info().scheme(), "memory");
    }

    #[rstest]
    #[case::seconds("3", Some(Duration::from_secs(3)))]
    #[case::zero("0", Some(Duration::ZERO))]
    #[case::invalid("soon", None)]
    fn test_parse_retry_after_delay_seconds(
        #[case] value: &str,
        #[case] expected: Option<Duration>,
    ) {
        let mut headers = http::HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_str(value).unwrap());
        assert_eq!(parse_retry_after(&headers), expected);
    }

    #[test]
    fn test_parse_retry_after_http_date() {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            RETRY_AFTER,
            HeaderValue::from_static("Wed, 21 Oct 2015 07:28:00 GMT"),
        );
        assert_eq!(parse_retry_after(&headers), Some(Duration::ZERO));

        let deadline = Timestamp::now() + Duration::from_secs(10);
        headers.insert(
            RETRY_AFTER,
            HeaderValue::from_str(&deadline.format_http_date()).unwrap(),
        );
        let delay = parse_retry_after(&headers).expect("future HTTP date must parse");
        assert!(
            delay > Duration::from_secs(5),
            "unexpected delay: {delay:?}"
        );
    }

    #[derive(Debug, Clone)]
    struct RetryTestService {
        stat_attempts: Arc<AtomicUsize>,
        stat_failures: usize,
        is_stat_error_temporary: bool,
        write_attempts: Arc<AtomicUsize>,
        write_failures: usize,
        write_payloads: Arc<Mutex<Vec<Bytes>>>,
        close_attempts: Arc<AtomicUsize>,
        close_failures: usize,
    }

    impl RetryTestService {
        fn new(stat_failures: usize, is_stat_error_temporary: bool) -> Self {
            Self {
                stat_attempts: Arc::new(AtomicUsize::new(0)),
                stat_failures,
                is_stat_error_temporary,
                write_attempts: Arc::new(AtomicUsize::new(0)),
                write_failures: 0,
                write_payloads: Arc::new(Mutex::new(Vec::new())),
                close_attempts: Arc::new(AtomicUsize::new(0)),
                close_failures: 0,
            }
        }

        fn with_write_failures(mut self, write_failures: usize) -> Self {
            self.write_failures = write_failures;
            self
        }

        fn with_close_failures(mut self, close_failures: usize) -> Self {
            self.close_failures = close_failures;
            self
        }

        fn stat_attempts(&self) -> usize {
            self.stat_attempts.load(Ordering::Relaxed)
        }

        fn write_attempts(&self) -> usize {
            self.write_attempts.load(Ordering::Relaxed)
        }

        fn write_payloads(&self) -> Vec<Bytes> {
            self.write_payloads.lock().unwrap().clone()
        }

        fn close_attempts(&self) -> usize {
            self.close_attempts.load(Ordering::Relaxed)
        }
    }

    impl Service for RetryTestService {
        type Reader = ();
        type Writer = RetryTestWriter;
        type Lister = ();
        type Deleter = ();
        type Copier = ();
        type Composer = ();

        fn info(&self) -> ServiceInfo {
            ServiceInfo::with_scheme("retry_test")
        }

        fn capability(&self) -> Capability {
            Capability {
                stat: true,
                write: true,
                write_can_multi: true,
                ..Default::default()
            }
        }

        async fn create_dir(
            &self,
            _: &OperationContext,
            _: &str,
            _: OpCreateDir,
        ) -> Result<RpCreateDir> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }

        async fn stat(&self, _: &OperationContext, _: &str, _: OpStat) -> Result<RpStat> {
            let attempt = self.stat_attempts.fetch_add(1, Ordering::Relaxed) + 1;
            if attempt <= self.stat_failures {
                let error = Error::new(ErrorKind::Unexpected, "injected stat failure");
                return if self.is_stat_error_temporary {
                    Err(error.set_temporary())
                } else {
                    Err(error)
                };
            }

            Ok(RpStat::new(MetadataBuilder::file(0).build()))
        }

        fn read(&self, _: &OperationContext, _: &str, _: OpRead) -> Result<Self::Reader> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }

        fn write(&self, _: &OperationContext, _: &str, _: OpWrite) -> Result<Self::Writer> {
            Ok(RetryTestWriter {
                write_attempts: self.write_attempts.clone(),
                write_failures: self.write_failures,
                write_payloads: self.write_payloads.clone(),
                close_attempts: self.close_attempts.clone(),
                close_failures: self.close_failures,
                content_length: 0,
            })
        }

        fn delete(&self, _: &OperationContext) -> Result<Self::Deleter> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }

        fn list(&self, _: &OperationContext, _: &str, _: OpList) -> Result<Self::Lister> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }

        fn copy(&self, _: &OperationContext, _: &str, _: &str, _: OpCopy) -> Result<Self::Copier> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }

        async fn rename(
            &self,
            _: &OperationContext,
            _: &str,
            _: &str,
            _: OpRename,
        ) -> Result<RpRename> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }

        async fn presign(&self, _: &OperationContext, _: &str, _: OpPresign) -> Result<RpPresign> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "operation is not supported",
            ))
        }
    }

    fn build_test_operator(service: RetryTestService, max_retries: usize) -> Operator {
        let operator = Operator::from_parts(OperationContext::default(), Arc::new(service));
        finish_opendal_operator(operator, max_retries)
    }

    #[tokio::test(start_paused = true)]
    async fn test_opendal_store_retries_temporary_stat() {
        let service = RetryTestService::new(2, true);
        let store = crate::object_store::opendal_store::OpendalStore::new(build_test_operator(
            service.clone(),
            3,
        ));

        store
            .head(&object_store::path::Path::from("file"))
            .await
            .expect("head should succeed after retries");
        assert_eq!(service.stat_attempts(), 3);
    }

    #[derive(Debug)]
    struct RetryTestWriter {
        write_attempts: Arc<AtomicUsize>,
        write_failures: usize,
        write_payloads: Arc<Mutex<Vec<Bytes>>>,
        close_attempts: Arc<AtomicUsize>,
        close_failures: usize,
        content_length: usize,
    }

    impl oio::Write for RetryTestWriter {
        async fn write(&mut self, buffer: Buffer) -> Result<()> {
            let attempt = self.write_attempts.fetch_add(1, Ordering::Relaxed) + 1;
            self.write_payloads.lock().unwrap().push(buffer.to_bytes());
            if attempt <= self.write_failures {
                return Err(
                    Error::new(ErrorKind::Unexpected, "injected write failure").set_temporary()
                );
            }

            self.content_length += buffer.len();
            Ok(())
        }

        async fn close(&mut self) -> Result<Metadata> {
            let attempt = self.close_attempts.fetch_add(1, Ordering::Relaxed) + 1;
            if attempt <= self.close_failures {
                return Err(
                    Error::new(ErrorKind::Unexpected, "injected close failure").set_temporary()
                );
            }

            Ok(MetadataBuilder::file(self.content_length as u64).build())
        }

        async fn abort(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[rstest]
    #[case::recovers(2, true, 3, 3, None)]
    #[case::persistent(1, false, 3, 1, Some(false))]
    #[case::disabled(1, true, 0, 1, Some(true))]
    #[case::exhausted(usize::MAX, true, 3, 4, Some(false))]
    #[tokio::test(start_paused = true)]
    async fn test_stat_retry(
        #[case] failures: usize,
        #[case] is_temporary: bool,
        #[case] max_retries: usize,
        #[case] expected_attempts: usize,
        #[case] expected_temporary: Option<bool>,
    ) {
        let service = RetryTestService::new(failures, is_temporary);
        let operator = build_test_operator(service.clone(), max_retries);

        let result = operator.stat("file").await;
        assert_eq!(service.stat_attempts(), expected_attempts);
        match expected_temporary {
            None => {
                result.expect("stat should succeed");
            }
            Some(expected) => {
                let error = result.expect_err("stat should fail");
                assert_eq!(error.is_temporary(), expected);
                assert_eq!(error.is_persistent(), !expected);
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn test_write_retry() {
        let service = RetryTestService::new(0, true).with_write_failures(1);
        let operator = build_test_operator(service.clone(), 3);

        operator
            .write("file", "data")
            .await
            .expect("write should succeed");
        assert_eq!(service.write_attempts(), 2);
        assert_eq!(
            service.write_payloads(),
            vec![Bytes::from_static(b"data"), Bytes::from_static(b"data")]
        );
        assert_eq!(service.close_attempts(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn test_close_retry() {
        let service = RetryTestService::new(0, true).with_close_failures(1);
        let operator = build_test_operator(service.clone(), 3);

        operator
            .write("file", "data")
            .await
            .expect("write should succeed");
        assert_eq!(service.close_attempts(), 2);
    }

    #[cfg(feature = "aws")]
    #[tokio::test]
    async fn test_reads_request_identity_encoding() {
        let server = MockServer::start().await;
        // A store that compresses the body would leave out Content-Length, so the
        // GET is only answered when it asks for the body unencoded.
        Mock::given(method("GET"))
            .and(path("/bucket/data.manifest"))
            .and(header("accept-encoding", "identity"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"manifest".to_vec()))
            .expect(1)
            .mount(&server)
            .await;

        let builder = S3::default()
            .endpoint(&server.uri())
            .bucket("bucket")
            .region("us-east-1")
            .access_key_id("test")
            .secret_access_key("test")
            .disable_config_load()
            .disable_ec2_metadata();
        let operator = finish_opendal_operator(Operator::new(builder).unwrap(), 0);

        let data = operator
            .read("data.manifest")
            .await
            .expect("read should succeed");
        assert_eq!(data.to_bytes(), Bytes::from_static(b"manifest"));
    }
}
