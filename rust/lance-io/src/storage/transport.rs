// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! HTTP customization below storage retries, signing, AIMD and metrics.
//!
//! [`HttpClientWrapper`] receives a client with the provider's resolved transport
//! settings. It may decorate that client or replace it. Requests contain a
//! cheaply cloneable [`PutPayload`]; response bodies remain streaming.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use http_body_util::{BodyExt, combinators::BoxBody};
use object_store::client as native;

use super::PutPayload;

/// A signed request for one HTTP attempt. Multipart parts are separate requests.
pub type HttpRequest = http::Request<PutPayload>;
/// A streaming response body, including trailers and errors after the headers.
pub type HttpBody = BoxBody<Bytes, HttpError>;
/// HTTP status failures are responses, not transport errors.
pub type HttpResponse = http::Response<HttpBody>;

/// Transport failure categories used by the provider's retry policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpErrorKind {
    /// Failed to connect; eligible for retry.
    Connect,
    /// Failed while sending a request; eligible for retry.
    Request,
    /// Timed out; retry requires an idempotent request.
    Timeout,
    /// Connection interrupted; retry requires an idempotent request.
    Interrupted,
    /// Invalid response data; not retried.
    Decode,
    /// Unclassified failure; not retried.
    Unknown,
}

/// A transport failure with its retry category and original cause.
#[derive(Debug, thiserror::Error)]
#[error("HTTP error: {source}")]
pub struct HttpError {
    kind: HttpErrorKind,
    #[source]
    source: Box<dyn std::error::Error + Send + Sync>,
}

impl HttpError {
    /// Classify a failure without discarding its original cause.
    pub fn new(
        kind: HttpErrorKind,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self {
            kind,
            source: source.into(),
        }
    }

    /// The failure category used to decide whether a request can be retried.
    pub fn kind(&self) -> HttpErrorKind {
        self.kind
    }
}

/// Executes a single HTTP attempt. The provider owns signing and retries.
#[async_trait]
pub trait HttpClient: std::fmt::Debug + Send + Sync + 'static {
    /// Preserve status failures as responses and transport failures as errors.
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, HttpError>;
}

/// Decorates or replaces each configured HTTP client opened by a cloud provider.
///
/// Reuse one `Arc` in [`StorageParams`](super::StorageParams) to share cached stores.
/// A replacement client owns its transport settings; a decorator retains those
/// of the supplied client. Neither should add retries already owned by the provider.
///
/// ```
/// use std::sync::Arc;
/// use lance_io::storage::transport::{HttpClient, HttpClientWrapper};
/// #[derive(Debug)]
/// struct UseConfiguredClient;
/// impl HttpClientWrapper for UseConfiguredClient {
///     fn wrap(&self, client: Arc<dyn HttpClient>) -> lance_core::Result<Arc<dyn HttpClient>> {
///         Ok(client)
///     }
/// }
/// ```
pub trait HttpClientWrapper: std::fmt::Debug + Send + Sync + 'static {
    /// Called once per client, with resolved timeout, proxy and TLS configuration.
    fn wrap(&self, client: Arc<dyn HttpClient>) -> lance_core::Result<Arc<dyn HttpClient>>;
}

/// Preserve the native transport and its options when no customization is requested.
#[derive(Debug)]
pub(crate) struct Transport(Option<Arc<dyn HttpClientWrapper>>);

impl Transport {
    pub(crate) fn new(wrapper: Option<&Arc<dyn HttpClientWrapper>>) -> Self {
        Self(wrapper.cloned())
    }
}

impl native::HttpConnector for Transport {
    fn connect(
        &self,
        options: &object_store::ClientOptions,
    ) -> object_store::Result<native::HttpClient> {
        let client = native::ReqwestConnector::default().connect(options)?;
        match &self.0 {
            None => Ok(client),
            Some(wrapper) => {
                let client =
                    wrapper
                        .wrap(Arc::new(ConfiguredClient(client)))
                        .map_err(|source| object_store::Error::Generic {
                            store: "Lance HTTP",
                            source: Box::new(source),
                        })?;
                Ok(native::HttpClient::new(CustomizedClient(client)))
            }
        }
    }
}

#[derive(Debug)]
struct ConfiguredClient(native::HttpClient);

#[async_trait]
impl HttpClient for ConfiguredClient {
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        let (parts, payload) = request.into_parts();
        let body = match payload.iter().as_slice() {
            [] => native::HttpRequestBody::empty(),
            [bytes] => bytes.clone().into(),
            _ => native::HttpRequestBody::from(object_store::PutPayload::from(payload)),
        };
        let response = self
            .0
            .execute(http::Request::from_parts(parts, body))
            .await
            .map_err(from_error)?;
        Ok(response.map(|body| body.map_err(from_error).boxed()))
    }
}

#[derive(Debug)]
struct CustomizedClient(Arc<dyn HttpClient>);

#[async_trait]
impl native::HttpService for CustomizedClient {
    async fn call(
        &self,
        request: native::HttpRequest,
    ) -> Result<native::HttpResponse, native::HttpError> {
        let (parts, mut body) = request.into_parts();
        // Native request bodies are already materialized. Keep the shared chunks
        // instead of concatenating multipart payloads into another allocation.
        let payload = if let Some(bytes) = body.as_bytes() {
            PutPayload::from(bytes.clone())
        } else {
            let mut chunks = Vec::new();
            while let Some(frame) = body.frame().await {
                if let Ok(bytes) = frame?.into_data() {
                    chunks.push(bytes);
                }
            }
            chunks.into_iter().collect()
        };
        let response = self
            .0
            .execute(http::Request::from_parts(parts, payload))
            .await
            .map_err(to_error)?;
        Ok(response.map(|body| native::HttpResponseBody::new(body.map_err(to_error))))
    }
}

fn from_error(error: native::HttpError) -> HttpError {
    let kind = match error.kind() {
        native::HttpErrorKind::Connect => HttpErrorKind::Connect,
        native::HttpErrorKind::Request => HttpErrorKind::Request,
        native::HttpErrorKind::Timeout => HttpErrorKind::Timeout,
        native::HttpErrorKind::Interrupted => HttpErrorKind::Interrupted,
        native::HttpErrorKind::Decode => HttpErrorKind::Decode,
        _ => HttpErrorKind::Unknown,
    };
    HttpError::new(kind, error)
}

fn to_error(error: HttpError) -> native::HttpError {
    let kind = match error.kind {
        HttpErrorKind::Connect => native::HttpErrorKind::Connect,
        HttpErrorKind::Request => native::HttpErrorKind::Request,
        HttpErrorKind::Timeout => native::HttpErrorKind::Timeout,
        HttpErrorKind::Interrupted => native::HttpErrorKind::Interrupted,
        HttpErrorKind::Decode => native::HttpErrorKind::Decode,
        HttpErrorKind::Unknown => native::HttpErrorKind::Unknown,
    };
    match error.source.downcast::<native::HttpError>() {
        Ok(source) if source.kind() == kind => *source,
        Ok(source) => native::HttpError::new_boxed(kind, source),
        Err(source) => native::HttpError::new_boxed(kind, source),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use native::{HttpConnector, HttpService};
    #[cfg(feature = "aws")]
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

    #[derive(Debug)]
    struct InspectClient {
        inner: Arc<dyn HttpClient>,
        payload: PutPayload,
    }

    #[async_trait]
    impl HttpClient for InspectClient {
        async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            assert_eq!(request.extensions().get::<usize>(), Some(&42));
            assert_eq!(request.body().iter().len(), self.payload.iter().len());
            for (actual, expected) in request.body().iter().zip(self.payload.iter()) {
                assert_eq!(actual, expected);
                assert_eq!(actual.as_ptr(), expected.as_ptr(), "payload was copied");
            }
            self.inner.execute(request).await
        }
    }

    #[derive(Debug)]
    struct InspectWrapper(PutPayload);

    impl HttpClientWrapper for InspectWrapper {
        fn wrap(&self, client: Arc<dyn HttpClient>) -> lance_core::Result<Arc<dyn HttpClient>> {
            Ok(Arc::new(InspectClient {
                inner: client,
                payload: self.0.clone(),
            }))
        }
    }

    #[rstest::rstest]
    #[case::response(false)]
    #[case::timeout(true)]
    #[tokio::test]
    async fn configured_transport_preserves_options_and_shared_payload(#[case] timeout: bool) {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200).set_body_bytes("response");
        Mock::given(matchers::method("PUT"))
            .and(matchers::body_bytes("firstsecond"))
            .respond_with(if timeout {
                response.set_delay(Duration::from_secs(1))
            } else {
                response
            })
            .mount(&server)
            .await;
        let payload: PutPayload = [Bytes::from_static(b"first"), Bytes::from_static(b"second")]
            .into_iter()
            .collect();
        let wrapper: Arc<dyn HttpClientWrapper> = Arc::new(InspectWrapper(payload.clone()));
        let client = Transport::new(Some(&wrapper))
            .connect(
                &object_store::ClientOptions::new()
                    .with_allow_http(true)
                    .with_timeout(if timeout {
                        Duration::from_millis(50)
                    } else {
                        Duration::from_secs(5)
                    }),
            )
            .unwrap();
        let mut request = http::Request::builder()
            .method("PUT")
            .uri(server.uri())
            .body(native::HttpRequestBody::from(
                object_store::PutPayload::from(payload),
            ))
            .unwrap();
        request.extensions_mut().insert(42_usize);
        let result = client.execute(request).await;
        if timeout {
            assert_eq!(result.unwrap_err().kind(), native::HttpErrorKind::Timeout);
        } else {
            let response = result.unwrap();
            assert_eq!(response.status(), http::StatusCode::OK);
            assert_eq!(response.into_body().bytes().await.unwrap(), "response");
        }
    }

    #[cfg(feature = "aws")]
    #[derive(Debug)]
    struct FailOnce {
        kind: HttpErrorKind,
        calls: AtomicUsize,
    }

    #[cfg(feature = "aws")]
    #[async_trait]
    impl HttpClient for FailOnce {
        async fn execute(&self, _request: HttpRequest) -> Result<HttpResponse, HttpError> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(HttpError::new(
                    self.kind,
                    std::io::Error::other("injected transport failure"),
                ));
            }
            Ok(http::Response::builder()
                .status(404)
                .body(
                    http_body_util::Full::new(Bytes::new())
                        .map_err(|never| match never {})
                        .boxed(),
                )
                .unwrap())
        }
    }

    #[cfg(feature = "aws")]
    #[derive(Debug)]
    struct ReplaceClient(Arc<FailOnce>);

    #[cfg(feature = "aws")]
    impl HttpClientWrapper for ReplaceClient {
        fn wrap(&self, _client: Arc<dyn HttpClient>) -> lance_core::Result<Arc<dyn HttpClient>> {
            Ok(self.0.clone())
        }
    }

    #[cfg(feature = "aws")]
    #[rstest::rstest]
    #[case(HttpErrorKind::Connect, 2)]
    #[case(HttpErrorKind::Request, 2)]
    #[case(HttpErrorKind::Timeout, 2)]
    #[case(HttpErrorKind::Interrupted, 2)]
    #[case(HttpErrorKind::Decode, 1)]
    #[case(HttpErrorKind::Unknown, 1)]
    #[tokio::test(start_paused = true)]
    async fn provider_retries_each_attempt_through_wrapper(
        #[case] kind: HttpErrorKind,
        #[case] attempts: usize,
    ) {
        use object_store::ObjectStoreExt;
        let failure = Arc::new(FailOnce {
            kind,
            calls: AtomicUsize::new(0),
        });
        let wrapper: Arc<dyn HttpClientWrapper> = Arc::new(ReplaceClient(failure.clone()));
        let store = object_store::aws::AmazonS3Builder::new()
            .with_bucket_name("bucket")
            .with_region("us-west-2")
            .with_skip_signature(true)
            .with_retry(object_store::RetryConfig {
                max_retries: 1,
                ..Default::default()
            })
            .with_http_connector(Transport::new(Some(&wrapper)))
            .build()
            .unwrap();
        let error = store
            .head(&object_store::path::Path::from("file"))
            .await
            .unwrap_err();
        assert_eq!(failure.calls.load(Ordering::SeqCst), attempts);
        if attempts == 2 {
            assert!(
                matches!(error, object_store::Error::NotFound { .. }),
                "{error:?}"
            );
        } else {
            assert!(
                error.to_string().contains("injected transport failure"),
                "{error:?}"
            );
        }
    }

    #[derive(Debug)]
    struct BrokenResponse;

    #[async_trait]
    impl HttpClient for BrokenResponse {
        async fn execute(&self, _request: HttpRequest) -> Result<HttpResponse, HttpError> {
            let body = http_body_util::Limited::new(
                http_body_util::Full::new(Bytes::from_static(b"data")),
                0,
            )
            .map_err(|source| HttpError::new(HttpErrorKind::Decode, source))
            .boxed();
            Ok(http::Response::new(body))
        }
    }

    #[tokio::test]
    async fn response_body_errors_are_not_consumed_with_headers() {
        let client = CustomizedClient(Arc::new(BrokenResponse));
        let response = client
            .call(http::Request::new(native::HttpRequestBody::empty()))
            .await
            .unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);
        let error = response.into_body().bytes().await.unwrap_err();
        assert_eq!(error.kind(), native::HttpErrorKind::Decode);
    }
}
