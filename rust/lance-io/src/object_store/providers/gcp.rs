// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, str::FromStr, sync::Arc, time::Duration};

use object_store::list::PaginatedListStore;
use object_store::{
    ClientOptions, CredentialProvider, ObjectStore as OSObjectStore, Result as ObjectStoreResult,
    client::{HttpClient, HttpConnector, HttpRequestBody, ReqwestConnector},
};
use opendal::{Operator, services::Gcs};
use reqsign_core::{Context as ReqsignContext, HttpSend, OsEnv, ProvideCredential};
use reqsign_file_read_tokio::TokioFileRead;
use reqsign_google::{Credential as ReqsignCredential, FileCredentialProvider};
use tokio::sync::RwLock;

use object_store::{
    RetryConfig, StaticCredentialProvider,
    gcp::{GcpCredential, GoogleCloudStorage, GoogleCloudStorageBuilder, GoogleConfigKey},
};
use url::Url;

use crate::object_store::opendal_store::OpendalStore;
use crate::object_store::{
    DEFAULT_CLOUD_BLOCK_SIZE, DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE, ObjectStore,
    ObjectStoreParams, ObjectStoreProvider, StorageOptions, StorageOptionsAccessor,
    dynamic_credentials::build_dynamic_credential_provider,
    throttle::{AimdThrottleConfig, AimdThrottleState, cloud_http_connector, with_throttling},
};
use lance_core::error::{Error, Result};
use lance_core::utils::parse::str_is_truthy;
#[derive(Default, Debug)]
pub struct GcsStoreProvider;

#[derive(Debug)]
struct ObjectStoreHttpSend {
    client: HttpClient,
}

impl HttpSend for ObjectStoreHttpSend {
    async fn http_send(
        &self,
        request: http::Request<bytes::Bytes>,
    ) -> reqsign_core::Result<http::Response<bytes::Bytes>> {
        let (parts, body) = request.into_parts();
        let request = http::Request::from_parts(parts, HttpRequestBody::from(body));
        let response = self.client.execute(request).await.map_err(|source| {
            reqsign_core::Error::unexpected("failed to send Google workload identity HTTP request")
                .with_source(source)
        })?;
        let (parts, body) = response.into_parts();
        let body = body.bytes().await.map_err(|source| {
            reqsign_core::Error::unexpected("failed to read Google workload identity HTTP response")
                .with_source(source)
        })?;
        Ok(http::Response::from_parts(parts, body))
    }
}

#[derive(Debug)]
struct WorkloadIdentityCredentialProvider {
    provider: FileCredentialProvider,
    context: ReqsignContext,
    cached_credential: RwLock<Option<ReqsignCredential>>,
}

impl WorkloadIdentityCredentialProvider {
    fn new(application_credentials_path: String, http_client: HttpClient) -> Self {
        let context = ReqsignContext::new()
            .with_file_read(TokioFileRead)
            .with_http_send(ObjectStoreHttpSend {
                client: http_client,
            })
            .with_env(OsEnv);
        Self {
            provider: FileCredentialProvider::new(application_credentials_path),
            context,
            cached_credential: RwLock::new(None),
        }
    }
}

fn usable_gcp_credential(credential: &ReqsignCredential) -> Option<GcpCredential> {
    credential
        .token
        .as_ref()
        .filter(|_| credential.has_valid_token())
        .map(|token| GcpCredential {
            bearer: token.access_token.clone(),
        })
}

fn workload_identity_error(
    source: impl std::error::Error + Send + Sync + 'static,
) -> object_store::Error {
    object_store::Error::Generic {
        store: "GCS workload identity credentials",
        source: Box::new(source),
    }
}

#[async_trait::async_trait]
impl CredentialProvider for WorkloadIdentityCredentialProvider {
    type Credential = GcpCredential;

    async fn get_credential(&self) -> ObjectStoreResult<Arc<Self::Credential>> {
        if let Some(credential) = self
            .cached_credential
            .read()
            .await
            .as_ref()
            .and_then(usable_gcp_credential)
        {
            return Ok(Arc::new(credential));
        }

        let mut cached_credential = self.cached_credential.write().await;
        if let Some(credential) = cached_credential.as_ref().and_then(usable_gcp_credential) {
            return Ok(Arc::new(credential));
        }

        let credential = self
            .provider
            .provide_credential(&self.context)
            .await
            .map_err(workload_identity_error)?
            .ok_or_else(|| {
                workload_identity_error(std::io::Error::other(
                    "application credentials did not provide a Google access token",
                ))
            })?;
        let gcp_credential = usable_gcp_credential(&credential).ok_or_else(|| {
            workload_identity_error(std::io::Error::other(
                "application credentials provided an expired or unusable Google access token",
            ))
        })?;
        *cached_credential = Some(credential);
        Ok(Arc::new(gcp_credential))
    }
}

#[derive(serde::Deserialize)]
struct ApplicationCredentialKind {
    #[serde(rename = "type")]
    credential_type: String,
}

struct GcsClientOptions {
    object_requests: ClientOptions,
    credential_requests: ClientOptions,
}

fn gcs_client_options(storage_options: &StorageOptions) -> Result<GcsClientOptions> {
    let mut object_requests = storage_options.client_options()?;
    // headers.* options are scoped to object requests and may contain secrets. Credential
    // exchanges can target unrelated identity endpoints, so only share typed client settings.
    let mut credential_requests = object_requests
        .clone()
        .with_default_headers(Default::default());
    for (key, value) in storage_options.as_gcs_options() {
        if let GoogleConfigKey::Client(key) = key {
            object_requests = object_requests.with_config(key, value.clone());
            credential_requests = credential_requests.with_config(key, value);
        }
    }
    Ok(GcsClientOptions {
        object_requests,
        credential_requests,
    })
}

fn workload_identity_credential_provider(
    storage_options: &StorageOptions,
    client_options: &ClientOptions,
) -> Result<Option<Arc<dyn CredentialProvider<Credential = GcpCredential>>>> {
    let gcs_options = storage_options.as_gcs_options();
    if gcs_options.contains_key(&GoogleConfigKey::ServiceAccount)
        || gcs_options.contains_key(&GoogleConfigKey::ServiceAccountKey)
    {
        return Ok(None);
    }

    let Some(application_credentials_path) =
        gcs_options.get(&GoogleConfigKey::ApplicationCredentials)
    else {
        return Ok(None);
    };
    let Ok(contents) = std::fs::read(application_credentials_path) else {
        return Ok(None);
    };
    let Ok(credential_kind) = serde_json::from_slice::<ApplicationCredentialKind>(&contents) else {
        return Ok(None);
    };
    if credential_kind.credential_type != "external_account" {
        return Ok(None);
    }

    let http_client = ReqwestConnector::default().connect(client_options)?;
    Ok(Some(Arc::new(WorkloadIdentityCredentialProvider::new(
        application_credentials_path.clone(),
        http_client,
    ))))
}

impl GcsStoreProvider {
    async fn build_opendal_gcs_store(
        &self,
        base_path: &Url,
        storage_options: &StorageOptions,
    ) -> Result<Arc<dyn OSObjectStore>> {
        let bucket = base_path
            .host_str()
            .ok_or_else(|| Error::invalid_input("GCS URL must contain bucket name"))?
            .to_string();

        let prefix = base_path.path().trim_start_matches('/').to_string();

        // Start with all storage options as the config map
        // OpenDAL will handle environment variables through its default credentials chain
        let mut config_map: HashMap<String, String> = storage_options.0.clone();

        // Set required OpenDAL configuration
        config_map.insert("bucket".to_string(), bucket);

        if !prefix.is_empty() {
            config_map.insert("root".to_string(), format!("/{}", prefix));
        }

        let operator = Operator::from_iter::<Gcs>(config_map)
            .map_err(|e| Error::invalid_input(format!("Failed to create GCS operator: {:?}", e)))?;

        Ok(Arc::new(OpendalStore::new(operator)) as Arc<dyn OSObjectStore>)
    }

    async fn build_google_cloud_store(
        &self,
        base_path: &Url,
        storage_options: &StorageOptions,
        accessor: Option<Arc<StorageOptionsAccessor>>,
        throttle_state: Option<&AimdThrottleState>,
        // Concrete rather than `dyn`, so the caller keeps the handle a paginated listing
        // needs: `PaginatedListStore` is a separate trait from `ObjectStore`.
    ) -> Result<Arc<GoogleCloudStorage>> {
        // Use a low retry count since the AIMD throttle layer handles
        // throttle recovery with its own retry loop.
        let retry_config = RetryConfig {
            backoff: Default::default(),
            max_retries: storage_options.client_max_retries(),
            retry_timeout: Duration::from_secs(storage_options.client_retry_timeout()),
        };

        let client_options = gcs_client_options(storage_options)?;

        let mut builder = GoogleCloudStorageBuilder::new()
            .with_url(base_path.as_ref())
            .with_retry(retry_config)
            .with_client_options(client_options.object_requests.clone());
        for (key, value) in storage_options.as_gcs_options() {
            builder = builder.with_config(key, value);
        }

        if let Some(credentials) =
            build_dynamic_credential_provider::<GcpCredential>(accessor).await?
        {
            builder = builder.with_credentials(credentials);
        } else if let Some(storage_token) = storage_options.get("google_storage_token") {
            let credential = GcpCredential {
                bearer: storage_token.clone(),
            };
            let credential_provider = Arc::new(StaticCredentialProvider::new(credential)) as _;
            builder = builder.with_credentials(credential_provider);
        } else if let Some(credential_provider) = workload_identity_credential_provider(
            storage_options,
            &client_options.credential_requests,
        )? {
            // object_store cannot exchange external-account ADC files, while reqsign supports
            // the workload identity format emitted by google-github-actions/auth.
            builder = builder.with_credentials(credential_provider);
        }

        let store_prefix =
            self.calculate_object_store_prefix(base_path, Some(&storage_options.0))?;
        builder = builder.with_http_connector(cloud_http_connector(throttle_state, store_prefix));

        Ok(Arc::new(builder.build()?))
    }
}

#[async_trait::async_trait]
impl ObjectStoreProvider for GcsStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_CLOUD_BLOCK_SIZE);
        let mut storage_options =
            StorageOptions::new(params.storage_options().cloned().unwrap_or_default());
        storage_options.with_env_gcs();
        let download_retry_count = storage_options.download_retry_count();

        let use_opendal = storage_options
            .0
            .get("use_opendal")
            .map(|v| str_is_truthy(v.as_str()))
            .unwrap_or(false);

        let accessor = params.get_accessor();

        let throttle_config = AimdThrottleConfig::from_storage_options(params.storage_options())?;
        let throttle_state = if throttle_config.is_disabled() {
            None
        } else {
            Some(AimdThrottleState::new(throttle_config)?)
        };

        let (inner, paginated_lister) = if use_opendal {
            // OpenDAL GCS intentionally uses static/environment-backed configuration only.
            // Namespace-vended dynamic credentials are supported on the native object_store path.
            // Listed in full: no paginated lister covers OpenDAL yet.
            (
                self.build_opendal_gcs_store(&base_path, &storage_options)
                    .await?,
                None,
            )
        } else {
            let store = self
                .build_google_cloud_store(
                    &base_path,
                    &storage_options,
                    accessor,
                    throttle_state.as_ref(),
                )
                .await?;
            (
                store.clone() as Arc<dyn OSObjectStore>,
                Some(store as Arc<dyn PaginatedListStore>),
            )
        };
        let (inner, paginated_lister) =
            with_throttling(throttle_state, !use_opendal, inner, paginated_lister);

        Ok(ObjectStore {
            inner,
            local_dir_operations: None,
            scheme: String::from("gs"),
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: true,
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count,
            io_tracker: Default::default(),
            store_prefix: self
                .calculate_object_store_prefix(&base_path, params.storage_options())?,
            paginated_lister,
        })
    }
}

impl StorageOptions {
    /// Add values from the environment to storage options
    pub fn with_env_gcs(&mut self) {
        for (os_key, os_value) in std::env::vars_os() {
            if let (Some(key), Some(value)) = (os_key.to_str(), os_value.to_str()) {
                let lowercase_key = key.to_ascii_lowercase();
                let token_key = "google_storage_token";

                if let Ok(config_key) = GoogleConfigKey::from_str(&lowercase_key) {
                    if !self.0.contains_key(config_key.as_ref()) {
                        self.0
                            .insert(config_key.as_ref().to_string(), value.to_string());
                    }
                }
                // Check for GOOGLE_STORAGE_TOKEN until GoogleConfigKey supports storage token
                else if lowercase_key == token_key && !self.0.contains_key(token_key) {
                    self.0.insert(token_key.to_string(), value.to_string());
                }
            }
        }
    }

    /// Subset of options relevant for gcs storage
    pub fn as_gcs_options(&self) -> HashMap<GoogleConfigKey, String> {
        self.0
            .iter()
            .filter_map(|(key, value)| {
                let gcs_key = GoogleConfigKey::from_str(&key.to_ascii_lowercase()).ok()?;
                Some((gcs_key, value.clone()))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashMap, fs, sync::Arc};

    use crate::object_store::test_utils::StaticMockStorageOptionsProvider;
    use crate::object_store::{ObjectStoreParams, StorageOptionsAccessor};
    use tempfile::TempDir;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    fn external_account_storage_options(
        temp_dir: &TempDir,
        token_url: String,
    ) -> HashMap<String, String> {
        let subject_token_path = temp_dir.path().join("oidc-token");
        fs::write(&subject_token_path, "github-oidc-token").unwrap();
        let application_credentials_path = temp_dir.path().join("credentials.json");
        fs::write(
            &application_credentials_path,
            serde_json::to_vec(&serde_json::json!({
                "type": "external_account",
                "audience": "test-audience",
                "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                "token_url": token_url,
                "credential_source": {
                    "file": subject_token_path.to_string_lossy(),
                    "format": { "type": "text" }
                }
            }))
            .unwrap(),
        )
        .unwrap();

        HashMap::from([
            (
                "google_application_credentials".to_string(),
                application_credentials_path.to_string_lossy().into_owned(),
            ),
            ("allow_http".to_string(), "true".to_string()),
        ])
    }

    #[test]
    fn test_gcs_store_path() {
        let provider = GcsStoreProvider;

        let url = Url::parse("gs://bucket/path/to/file").unwrap();
        let path = provider.extract_path(&url).unwrap();
        let expected_path = object_store::path::Path::from("path/to/file");
        assert_eq!(path, expected_path);
    }

    #[tokio::test]
    async fn test_use_opendal_flag() {
        let provider = GcsStoreProvider;
        let url = Url::parse("gs://test-bucket/path").unwrap();
        let params_with_flag = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("use_opendal".to_string(), "true".to_string()),
                    (
                        "service_account".to_string(),
                        "test@example.iam.gserviceaccount.com".to_string(),
                    ),
                ]),
            ))),
            ..Default::default()
        };

        let store = provider
            .new_store(url.clone(), &params_with_flag)
            .await
            .unwrap();
        assert_eq!(store.scheme, "gs");
    }

    #[tokio::test]
    async fn test_dynamic_gcp_credentials_provider() {
        let accessor = Arc::new(StorageOptionsAccessor::with_provider(Arc::new(
            StaticMockStorageOptionsProvider {
                options: HashMap::from([(
                    "google_storage_token".to_string(),
                    "gcp-token".to_string(),
                )]),
            },
        )));

        let credentials = build_dynamic_credential_provider::<GcpCredential>(Some(accessor))
            .await
            .expect("dynamic gcp credentials should build")
            .expect("expected credential provider")
            .get_credential()
            .await
            .expect("expected gcp credential");

        assert_eq!(credentials.bearer, "gcp-token");
    }

    #[tokio::test]
    async fn test_external_account_application_credentials() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "federated-token",
                "expires_in": 3600
            })))
            .expect(1)
            .mount(&mock_server)
            .await;

        let temp_dir = tempfile::tempdir().unwrap();
        let mut storage_options =
            external_account_storage_options(&temp_dir, format!("{}/token", mock_server.uri()));
        storage_options.insert(
            "headers.Authorization".to_string(),
            "Bearer storage-secret".to_string(),
        );
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                storage_options.clone(),
            ))),
            ..Default::default()
        };

        let store = GcsStoreProvider
            .new_store(Url::parse("gs://test-bucket/path").unwrap(), &params)
            .await
            .expect("external account credentials should build a GCS store");
        assert_eq!(store.scheme, "gs");

        let storage_options = StorageOptions::new(storage_options);
        let client_options = gcs_client_options(&storage_options).unwrap();
        let credential_provider = workload_identity_credential_provider(
            &storage_options,
            &client_options.credential_requests,
        )
        .expect("external account credential provider should build")
        .expect("external account credentials should select the reqsign provider");
        for _ in 0..2 {
            let credential = credential_provider
                .get_credential()
                .await
                .expect("workload identity token exchange should succeed");
            assert_eq!(credential.bearer, "federated-token");
        }
        mock_server.verify().await;
        let requests = mock_server.received_requests().await.unwrap();
        assert!(!requests[0].headers.contains_key("authorization"));
    }

    #[tokio::test]
    async fn test_external_account_respects_client_timeout() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/token"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_secs(1))
                    .set_body_json(serde_json::json!({
                        "access_token": "federated-token",
                        "expires_in": 3600
                    })),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let temp_dir = tempfile::tempdir().unwrap();
        let mut storage_options =
            external_account_storage_options(&temp_dir, format!("{}/token", mock_server.uri()));
        storage_options.insert("timeout".to_string(), "50ms".to_string());
        let storage_options = StorageOptions::new(storage_options);
        let client_options = gcs_client_options(&storage_options).unwrap();
        let credential_provider = workload_identity_credential_provider(
            &storage_options,
            &client_options.credential_requests,
        )
        .expect("external account credential provider should build")
        .expect("external account credentials should select the reqsign provider");

        let credential_result = tokio::time::timeout(
            Duration::from_millis(200),
            credential_provider.get_credential(),
        )
        .await
        .expect("configured client timeout should bound the credential exchange");
        let error = credential_result.expect_err("the delayed token exchange should time out");
        assert!(matches!(&error, object_store::Error::Generic { .. }));
        assert!(
            error
                .to_string()
                .contains("failed to send Google workload identity HTTP request"),
            "unexpected error: {error}"
        );
        mock_server.verify().await;
    }
}
