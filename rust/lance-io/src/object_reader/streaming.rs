// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use futures::{StreamExt, stream};
use lance_core::Error;
use object_store::{GetOptions, GetResult, ObjectMeta, ObjectStore, Result, path::Path};

use super::{ConditionalGetSupport, GetRequest};
use crate::object_store::hf_throttle::is_huggingface_rate_limited;
use crate::traits::ByteStream;

// One budget bounds the entire download, including requests to reopen a failed body.
async fn request_with_retry(request: &GetRequest, retries: &mut usize) -> Result<GetResult> {
    loop {
        match request
            .object_store
            .get_opts(&request.path, request.options.clone())
            .await
        {
            Ok(result) => return Ok(result),
            Err(error) => {
                // OpenDAL already owns retries for handled Hugging Face rate limits.
                if *retries == 0
                    || !matches!(error, object_store::Error::Generic { .. })
                    || is_huggingface_rate_limited(&error)
                {
                    return Err(error);
                }
                *retries -= 1;
            }
        }
    }
}

pub(super) async fn get_stream(request: GetRequest, retries: usize) -> Result<ByteStream> {
    Ok(open_download(request, retries).await?.into_stream())
}

pub async fn read_with_meta(
    object_store: Arc<dyn ObjectStore>,
    path: &Path,
    options: GetOptions,
    retries: usize,
) -> Result<(Bytes, ObjectMeta)> {
    let request = GetRequest {
        object_store,
        path: path.clone(),
        options,
    };
    let download = open_download(request, retries).await?;
    let meta = download.meta.clone();
    let bytes = object_store::collect_bytes(download.into_stream(), Some(meta.size))
        .await
        .map_err(|error| match error {
            object_store::Error::NotFound { path, source } => {
                object_store::Error::Precondition { path, source }
            }
            error => error,
        })?;
    Ok((bytes, meta))
}

async fn open_download(request: GetRequest, mut retries: usize) -> Result<StreamingDownload> {
    let result = request_with_retry(&request, &mut retries).await?;
    let expected_range = match &request.options.range {
        Some(range) => range.as_range(result.meta.size).map_err(|error| {
            Error::invalid_input(format!(
                "invalid range {range:?} for '{}' with {} bytes: {error}",
                request.path, result.meta.size
            ))
        })?,
        None => 0..result.meta.size,
    };
    if result.range != expected_range {
        return Err(Error::io(format!(
            "object at '{}' returned range {:?}, expected {:?}",
            request.path, result.range, expected_range
        ))
        .into());
    }
    Ok(StreamingDownload {
        request,
        remaining: expected_range,
        meta: result.meta.clone(),
        conditions: result.extensions.get::<ConditionalGetSupport>().copied(),
        body: result.into_stream(),
        retries,
    })
}

struct StreamingDownload {
    request: GetRequest,
    remaining: Range<u64>,
    meta: ObjectMeta,
    conditions: Option<ConditionalGetSupport>,
    body: ByteStream,
    retries: usize,
}

impl StreamingDownload {
    fn into_stream(self) -> ByteStream {
        stream::try_unfold(self, |download| download.next()).boxed()
    }

    async fn next(mut self) -> Result<Option<(Bytes, Self)>> {
        loop {
            let error = match self.body.next().await {
                Some(Ok(bytes)) => {
                    if bytes.len() as u64 > self.remaining.end - self.remaining.start {
                        let error = Error::io(format!(
                            "object body at '{}' exceeds remaining range {:?}: chunk length {}",
                            self.request.path,
                            self.remaining,
                            bytes.len()
                        ));
                        return Err(error.into());
                    }
                    // The length check above bounds this addition by the range end.
                    self.remaining.start += bytes.len() as u64;
                    return Ok(Some((bytes, self)));
                }
                None if self.remaining.is_empty() => return Ok(None),
                None => Error::io(format!(
                    "object body at '{}' ended before remaining range {:?}",
                    self.request.path, self.remaining
                ))
                .into(),
                Some(Err(error)) => error,
            };
            if self.remaining.is_empty()
                || self.retries == 0
                || !matches!(error, object_store::Error::Generic { .. })
            {
                return Err(error);
            }
            // Prefer one supported guard: some adapters return versions in metadata
            // but cannot use them as request conditions (e.g. OpenDAL GCS).
            self.request.options.if_match = self
                .meta
                .e_tag
                .clone()
                .filter(|_| self.conditions.is_none_or(|support| support.has_if_match));
            self.request.options.version = self.meta.version.clone().filter(|_| {
                self.request.options.if_match.is_none()
                    && self.conditions.is_none_or(|support| support.has_version)
            });
            if self.request.options.if_match.is_none() && self.request.options.version.is_none() {
                // Preserve the body failure when no condition can prevent mixing objects.
                return Err(error);
            }
            self.retries -= 1;
            self.request.options.range = Some(self.remaining.clone().into());
            // Release the failed response before opening another connection.
            self.body = stream::empty().boxed();
            let result = match request_with_retry(&self.request, &mut self.retries).await {
                // Custom stores may reject conditions without advertising capabilities.
                Err(object_store::Error::NotSupported { .. }) => return Err(error),
                result => result?,
            };
            // Also verify metadata: custom stores may ignore conditional GET options.
            if result.meta.e_tag != self.meta.e_tag
                || result.meta.version != self.meta.version
                || result.meta.size != self.meta.size
            {
                return Err(object_store::Error::Precondition {
                    path: self.request.path.to_string(),
                    source: format!(
                        "object changed while resuming a streaming download: \
                         ETag {:?} -> {:?}, version {:?} -> {:?}, size {} -> {}",
                        self.meta.e_tag,
                        result.meta.e_tag,
                        self.meta.version,
                        result.meta.version,
                        self.meta.size,
                        result.meta.size
                    )
                    .into(),
                });
            }
            if result.range != self.remaining {
                return Err(Error::io(format!(
                    "object at '{}' returned range {:?}, expected remaining range {:?}",
                    self.request.path, result.range, self.remaining
                ))
                .into());
            }
            self.body = result.into_stream();
        }
    }
}

#[cfg(all(test, feature = "gcp"))]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use object_store::path::Path;
    use rstest::rstest;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::Notify;
    use tokio::task::JoinHandle;
    use url::Url;

    use super::*;
    use crate::object_store::providers::gcp::GcsStoreProvider;
    #[cfg(feature = "huggingface")]
    use crate::object_store::providers::huggingface::HuggingfaceStoreProvider;
    use crate::object_store::{ObjectStoreParams, ObjectStoreProvider, StorageOptionsAccessor};

    const DATA: &str = "abcdefghijklmnop";
    const UNAVAILABLE: &str =
        "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

    /// A real HTTP body whose connection closes only after the test consumes its
    /// first chunk. This ensures an actual mid-body failure, without a sleep.
    struct PartialBodyServer {
        endpoint: String,
        requests: Arc<Mutex<Vec<String>>>,
        close_body: Arc<Notify>,
        task: JoinHandle<()>,
    }

    impl PartialBodyServer {
        async fn start(responses: Vec<String>, partial_response: usize) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let requests = Arc::new(Mutex::new(Vec::new()));
            let close_body = Arc::new(Notify::new());
            let recorded = requests.clone();
            let closed = close_body.clone();
            let task = tokio::spawn(async move {
                let mut responses = responses.into_iter();
                let mut index = 0;
                loop {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let mut request = Vec::new();
                    let mut buffer = [0; 1024];
                    while !request.ends_with(b"\r\n\r\n") {
                        let count = socket.read(&mut buffer).await.unwrap();
                        assert_ne!(count, 0);
                        request.extend_from_slice(&buffer[..count]);
                    }
                    recorded
                        .lock()
                        .unwrap()
                        .push(String::from_utf8(request).unwrap().to_ascii_lowercase());
                    let response = responses.next().unwrap_or_else(|| UNAVAILABLE.into());
                    socket.write_all(response.as_bytes()).await.unwrap();
                    if index == partial_response {
                        closed.notified().await;
                    }
                    socket.shutdown().await.unwrap();
                    index += 1;
                }
            });
            Self {
                endpoint,
                requests,
                close_body,
                task,
            }
        }
    }

    impl Drop for PartialBodyServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    fn response(range: Range<usize>, is_partial: bool, etag: &str) -> String {
        let status = if range == (0..16) {
            "200 OK"
        } else {
            "206 Partial Content"
        };
        let body_end = if is_partial {
            range.start + 4
        } else {
            range.end
        };
        format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/16\r\n\
             ETag: \"{etag}\"\r\nx-goog-generation: 123\r\nConnection: close\r\n\r\n{}",
            range.len(),
            range.start,
            range.end - 1,
            &DATA[range.start..body_end]
        )
    }

    #[rstest]
    #[case::resume(1, false, false)]
    #[case::changed_object(1, true, false)]
    #[case::shared_budget_exhausted(2, false, true)]
    #[case::shared_budget_recovered(3, false, true)]
    #[tokio::test]
    async fn streaming_opendal_gcs_resume(
        #[case] retry_count: usize,
        #[case] is_replaced: bool,
        #[case] has_request_errors: bool,
        #[values(false, true)] is_range: bool,
    ) {
        let range = if is_range { 3..13 } else { 0..16 };
        let mut responses = Vec::new();
        if has_request_errors {
            responses.push(UNAVAILABLE.into());
        }
        let partial_response = responses.len();
        responses.push(response(range.clone(), true, "original"));
        if has_request_errors {
            responses.push(UNAVAILABLE.into());
        }
        responses.push(response(
            range.start + 4..range.end,
            false,
            if is_replaced {
                "replacement"
            } else {
                "original"
            },
        ));
        let server = PartialBodyServer::start(responses, partial_response).await;
        let options = HashMap::from([
            ("use_opendal".into(), "true".into()),
            ("endpoint".into(), server.endpoint.clone()),
            ("skip_signature".into(), "true".into()),
            ("disable_config_load".into(), "true".into()),
            ("disable_vm_metadata".into(), "true".into()),
            ("client_max_retries".into(), "0".into()),
            ("download_retry_count".into(), retry_count.to_string()),
        ]);
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                options,
            ))),
            ..Default::default()
        };
        let store = GcsStoreProvider
            .new_store(Url::parse("gs://test-bucket").unwrap(), &params)
            .await
            .unwrap();
        let reader = store.open(&Path::from("data.lance")).await.unwrap();
        let mut stream = if is_range {
            reader.get_range_stream(range.clone()).await
        } else {
            reader.get_stream().await
        }
        .unwrap();
        assert_eq!(
            stream.next().await.unwrap().unwrap(),
            &DATA[range.start..range.start + 4]
        );
        server.close_body.notify_one();
        let next = stream.next().await.unwrap();
        if is_replaced {
            let error = next.unwrap_err();
            assert!(matches!(error, object_store::Error::Precondition { .. }));
            assert!(error.to_string().contains("object changed"), "{error}");
        } else if retry_count == 2 {
            let error = next.unwrap_err();
            assert!(matches!(error, object_store::Error::Generic { .. }));
            assert!(error.to_string().contains("503"), "{error}");
        } else {
            assert_eq!(next.unwrap(), &DATA[range.start + 4..range.end]);
        }
        assert!(stream.next().await.is_none());
        let requests = server.requests.lock().unwrap();
        assert_eq!(requests.len(), retry_count + 1);
        for request in &requests[partial_response + 1..] {
            assert!(
                request.starts_with("get /storage/v1/b/test-bucket/o/data.lance?"),
                "{request}"
            );
            assert!(request.contains("if-match: \"original\"\r\n"), "{request}");
            assert!(
                request.contains(&format!(
                    "range: bytes={}-{}\r\n",
                    range.start + 4,
                    range.end - 1
                )),
                "{request}"
            );
        }
    }

    #[cfg(feature = "huggingface")]
    #[rstest]
    #[tokio::test]
    async fn streaming_opendal_huggingface_preserves_body_error(
        #[values(false, true)] is_range: bool,
    ) {
        let range = if is_range { 3..13 } else { 0..16 };
        let server =
            PartialBodyServer::start(vec![response(range.clone(), true, "original")], 0).await;
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("client_max_retries".into(), "0".into()),
                    ("download_retry_count".into(), "3".into()),
                    ("download_mode".into(), "http".into()),
                ]),
            ))),
            ..Default::default()
        };
        let store = HuggingfaceStoreProvider
            .new_store_with_test_endpoint(
                Url::parse("hf://datasets/acme/repo").unwrap(),
                &params,
                &server.endpoint,
            )
            .await
            .unwrap();
        let reader = store.open(&Path::from("data.lance")).await.unwrap();
        let mut stream = if is_range {
            reader.get_range_stream(range.clone()).await
        } else {
            reader.get_stream().await
        }
        .unwrap();
        assert_eq!(
            stream.next().await.unwrap().unwrap(),
            &DATA[range.start..range.start + 4]
        );
        server.close_body.notify_one();
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(
            error,
            object_store::Error::Generic {
                store: "IoError",
                ..
            }
        ));
        assert!(error.to_string().contains("body"), "{error}");
        assert!(stream.next().await.is_none());
        assert_eq!(server.requests.lock().unwrap().len(), 1);
    }
}
