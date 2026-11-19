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
pub struct RecordingConnector {
    pub connects: std::sync::atomic::AtomicUsize,
    pub requests: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
#[derive(Debug)]
struct NotFound(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
#[async_trait::async_trait]
impl object_store::client::HttpService for NotFound {
    async fn call(
        &self,
        req: object_store::client::HttpRequest,
    ) -> std::result::Result<object_store::client::HttpResponse, object_store::client::HttpError>
    {
        self.0.lock().unwrap().push(req.uri().to_string());
        let mut response = object_store::client::HttpResponse::new(
            object_store::client::HttpResponseBody::from(String::new()),
        );
        *response.status_mut() = http::StatusCode::NOT_FOUND;
        Ok(response)
    }
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
impl object_store::client::HttpConnector for RecordingConnector {
    fn connect(
        &self,
        _options: &object_store::ClientOptions,
    ) -> object_store::Result<object_store::client::HttpClient> {
        self.connects
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(object_store::client::HttpClient::new(NotFound(
            self.requests.clone(),
        )))
    }
}
