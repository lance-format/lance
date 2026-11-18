// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use async_trait::async_trait;

use super::StorageOptionsProvider;
use lance_core::Result;

#[derive(Debug)]
pub struct StaticMockStorageOptionsProvider {
    pub options: HashMap<String, String>,
}

#[async_trait]
impl StorageOptionsProvider for StaticMockStorageOptionsProvider {
    async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
        Ok(Some(self.options.clone()))
    }

    fn provider_id(&self) -> String {
        "StaticMockStorageOptionsProvider".to_string()
    }
}

/// Answers every request with 404 and records what reached it.
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
#[derive(Debug, Default)]
pub struct RecordingHttpClientWrapper {
    pub clients: std::sync::atomic::AtomicUsize,
    pub requests: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    pub headers: std::sync::Arc<std::sync::Mutex<Vec<http::HeaderMap>>>,
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
#[derive(Debug)]
struct NotFound {
    requests: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    headers: std::sync::Arc<std::sync::Mutex<Vec<http::HeaderMap>>>,
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
#[async_trait]
impl super::transport::HttpClient for NotFound {
    async fn execute(
        &self,
        request: super::transport::HttpRequest,
    ) -> std::result::Result<super::transport::HttpResponse, super::transport::HttpError> {
        use http_body_util::BodyExt;
        self.requests
            .lock()
            .unwrap()
            .push(request.uri().to_string());
        self.headers.lock().unwrap().push(request.headers().clone());
        Ok(http::Response::builder()
            .status(http::StatusCode::NOT_FOUND)
            .body(
                http_body_util::Full::new(bytes::Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap())
    }
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
impl super::transport::HttpClientWrapper for RecordingHttpClientWrapper {
    fn wrap(
        &self,
        _client: std::sync::Arc<dyn super::transport::HttpClient>,
    ) -> Result<std::sync::Arc<dyn super::transport::HttpClient>> {
        self.clients
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(std::sync::Arc::new(NotFound {
            requests: self.requests.clone(),
            headers: self.headers.clone(),
        }))
    }
}
