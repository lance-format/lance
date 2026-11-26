// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::object_store::copy_diagnostics::{
    CopyOperationEvent, credential_lock_acquired, record_current,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::{collections::HashMap, str::FromStr, sync::Arc, time::Duration};

#[cfg(test)]
use mock_instant::thread_local::{SystemTime, UNIX_EPOCH};

#[cfg(not(test))]
use std::time::{SystemTime, UNIX_EPOCH};

use object_store::ObjectStore as OSObjectStore;
use object_store::list::PaginatedListStore;
use opendal::{Operator, services::S3};

use aws_config::default_provider::credentials::DefaultCredentialsChain;
use aws_config::ecs::EcsCredentialsProvider;
use aws_config::provider_config::ProviderConfig;
use aws_config::web_identity_token::WebIdentityTokenCredentialsProvider;
use aws_config::{BehaviorVersion, Region, SdkConfig};
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use object_store::{
    ClientOptions, CredentialProvider, Result as ObjectStoreResult, RetryConfig,
    StaticCredentialProvider,
    aws::{
        AmazonS3, AmazonS3Builder, AmazonS3ConfigKey, AwsCredential as ObjectStoreAwsCredential,
        AwsCredentialProvider, S3CopyIfNotExists,
    },
};
use tokio::sync::{Mutex, RwLock};
use url::Url;

use crate::object_store::opendal_store::OpendalStore;
use crate::object_store::{
    DEFAULT_CLOUD_BLOCK_SIZE, DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE, MutationBackend,
    ObjectStore, ObjectStoreParams, ObjectStoreProvider, StorageOptions, StorageOptionsAccessor,
    dynamic_credentials::{NamespaceCredentialsProvider, build_dynamic_credential_provider},
    throttle::{
        AimdThrottleConfig, AimdThrottleState, Transport, cloud_http_connector,
        reject_http_connector_for_opendal, shared_throttle_state, with_throttling,
    },
};
use lance_core::error::{Error, Result};
use lance_core::utils::{aimd::AimdConfig, parse::str_is_truthy};

mod conditional_copy;
mod multipart_update;

use conditional_copy::ConditionalS3Store;
pub(in crate::object_store) use multipart_update::{
    ConditionalUpdate, MultipartUpdate, conditional_operator,
};

use super::opendal::finish_opendal_operator;

#[derive(Default, Debug)]
pub struct AwsStoreProvider;

/// object_store converts every HTTP 409 into AlreadyExists, including S3
/// bucket-lock policy errors. Only a conditional-write conflict is retryable
/// by a tag writer which rereads the current tag.
pub(in crate::object_store) fn is_conditional_put_conflict(error: &object_store::Error) -> bool {
    match error {
        object_store::Error::AlreadyExists { source, .. } => {
            // The native client's typed HTTP response error is private.
            let Some(response) = source.source() else {
                return false;
            };
            let message = response.to_string();
            let Some(body) =
                message.strip_prefix("Server returned non-2xx status code: 409 Conflict: ")
            else {
                return false;
            };
            matches!(
                crate::object_store::throttle::error_document_code(body).as_deref(),
                Some("ConditionalRequestConflict" | "OperationAborted")
            )
        }
        object_store::Error::Generic { source, .. } => source
            .downcast_ref::<opendal::Error>()
            .is_some_and(|e| e.kind() == opendal::ErrorKind::Conflict),
        _ => false,
    }
}

struct ResolvedS3StorageOptions {
    options: HashMap<AmazonS3ConfigKey, String>,
    profile_region: Option<String>,
    profile_credentials: Option<SharedCredentialsProvider>,
}

impl ResolvedS3StorageOptions {
    fn new(
        mut options: HashMap<AmazonS3ConfigKey, String>,
        profile_config: Option<&SdkConfig>,
        profile_credentials: Option<SharedCredentialsProvider>,
    ) -> Self {
        if effective_s3_endpoint(&options).is_none()
            && let Some(endpoint) = profile_config.and_then(SdkConfig::endpoint_url)
        {
            options.insert(AmazonS3ConfigKey::Endpoint, endpoint.to_string());
        }
        let profile_region = profile_config
            .and_then(SdkConfig::region)
            .map(|region| region.as_ref().to_string());
        Self {
            options,
            profile_region,
            profile_credentials,
        }
    }

    fn effective_endpoint(&self) -> Option<&str> {
        effective_s3_endpoint(&self.options)
    }

    fn supports_conditional_update(&self) -> bool {
        // Match the native S3 client's default for both S3 implementations.
        self.options
            .get(&AmazonS3ConfigKey::ConditionalPut)
            .is_none_or(|mode| mode.trim() == "etag")
    }

    fn throttle_defaults(&self, is_s3_express: bool, use_opendal: bool) -> AimdThrottleConfig {
        let defaults = AimdThrottleConfig::default();
        if is_s3_express || use_opendal || self.effective_endpoint().is_some() {
            return defaults;
        }
        // AWS general-purpose S3 sustains at least 11,000 GET/HEAD requests/s
        // per partitioned prefix. Preserve the existing startup rate for
        // independent processes that may be reading the same prefix.
        defaults.with_read_aimd(AimdConfig::default().with_max_rate(11_000.0))
    }

    fn requires_constant_size_upload_parts(&self) -> bool {
        self.effective_endpoint()
            .is_some_and(|endpoint| endpoint.contains("r2.cloudflarestorage.com"))
    }
}

impl AwsStoreProvider {
    async fn build_amazon_s3_store(
        &self,
        base_path: &mut Url,
        params: &ObjectStoreParams,
        storage_options: &StorageOptions,
        mut resolved_s3_options: ResolvedS3StorageOptions,
        is_s3_express: bool,
        throttle_state: Option<&AimdThrottleState>,
        // Keep the concrete client for paginated listing alongside the copy adapter.
    ) -> Result<(Arc<dyn OSObjectStore>, Arc<AmazonS3>)> {
        // Use a low retry count since the AIMD throttle layer handles
        // throttle recovery with its own retry loop.
        let retry_config = RetryConfig {
            backoff: Default::default(),
            max_retries: storage_options.client_max_retries(),
            retry_timeout: Duration::from_secs(storage_options.client_retry_timeout()),
        };

        let region = resolve_s3_region(base_path, &resolved_s3_options).await?;

        // Get accessor from params
        let accessor = params.get_accessor();

        let provider_scheme = storage_options.aws_provider_scheme()?;

        let (aws_creds, region) = build_aws_credential_with_profile(
            params.s3_credentials_refresh_offset,
            params.aws_credentials.clone(),
            Some(&resolved_s3_options.options),
            region,
            accessor,
            provider_scheme,
            resolved_s3_options.profile_credentials.clone(),
        )
        .await?;

        // Set S3Express flag if detected
        if is_s3_express {
            resolved_s3_options
                .options
                .insert(AmazonS3ConfigKey::S3Express, true.to_string());
        }

        // Compute the metrics label before rewriting the URL below so it
        // matches the prefix the registry uses to key this store.
        let store_prefix =
            self.calculate_object_store_prefix(base_path, Some(&storage_options.0))?;
        let metrics_base = crate::object_store::metrics_base(&store_prefix, base_path);

        // before creating the OSObjectStore we need to rewrite the url to drop ddb related parts
        base_path.set_scheme("s3").unwrap();
        base_path.set_query(None);

        // we can't use parse_url_opts here because we need to manually set the credentials provider
        // Reuse the same transport settings for the completion client, including
        // settings that with_config would otherwise apply only to this builder.
        let mut client_options = storage_options.client_options()?;
        for (key, value) in &resolved_s3_options.options {
            if let AmazonS3ConfigKey::Client(key) = key {
                client_options = client_options.with_config(*key, value);
            }
        }
        let conditional_updates = resolved_s3_options.supports_conditional_update();
        let mut builder = AmazonS3Builder::new()
            .with_client_options(client_options.clone())
            .with_copy_if_not_exists(S3CopyIfNotExists::Multipart);
        for (key, value) in resolved_s3_options.options {
            builder = builder.with_config(key, value);
        }
        builder = builder
            .with_url(base_path.as_ref())
            .with_credentials(aws_creds)
            .with_retry(retry_config)
            .with_region(region);

        builder = builder.with_http_connector(cloud_http_connector(
            throttle_state,
            metrics_base,
            Transport::new(params.http_connector.as_ref()),
        ));

        let use_multipart_copy = builder
            .get_config_value(&AmazonS3ConfigKey::CopyIfNotExists)
            .is_some_and(|mode| mode.trim() == "multipart");
        let store = Arc::new(builder.clone().build()?);
        let inner: Arc<dyn OSObjectStore> = if use_multipart_copy || conditional_updates {
            Arc::new(ConditionalS3Store::new(
                store.clone(),
                builder,
                client_options,
                use_multipart_copy,
            )?)
        } else {
            store.clone()
        };
        Ok((inner, store))
    }

    async fn build_opendal_s3_store(
        &self,
        base_path: &Url,
        storage_options: &StorageOptions,
        aws_credentials: Option<&AwsCredentialProvider>,
    ) -> Result<Arc<dyn OSObjectStore>> {
        let bucket = base_path
            .host_str()
            .ok_or_else(|| Error::invalid_input("S3 URL must contain bucket name"))?
            .to_string();

        let prefix = base_path.path().trim_start_matches('/').to_string();

        // Start with all storage options as the config map
        // OpenDAL will handle environment variables through its default credentials chain
        let mut config_map: HashMap<String, String> = storage_options.0.clone();

        if let Some(provider_scheme) = storage_options.aws_provider_scheme()? {
            return Result::Err(Error::not_supported(format!(
                "OpendalStore does not currently support an explicit provider_scheme (currently set to {:?})",
                provider_scheme
            )));
        }

        // The provider reaches the native store through the params, which this
        // arm does not pass on, so a request here would be signed with whatever
        // credentials OpenDAL resolves for itself instead.
        if aws_credentials.is_some() {
            return Result::Err(Error::not_supported(
                "OpendalStore does not currently support an explicit aws_credentials provider"
                    .to_string(),
            ));
        }

        // Set required OpenDAL configuration
        config_map.insert("bucket".to_string(), bucket);

        if !prefix.is_empty() {
            config_map.insert("root".to_string(), "/".to_string());
        }

        let operator = Operator::from_iter::<S3>(config_map)
            .map_err(|e| Error::invalid_input(format!("Failed to create S3 operator: {:?}", e)))?;
        let retries = storage_options.client_max_retries();
        // OpenDAL's RetryLayer takes ownership of the writer during write/close.
        // Cancellation would leave no writer to abort. Both handles share the
        // same service and HTTP client; staged parts retry at the transport.
        let conditional =
            conditional_operator(&finish_opendal_operator(operator.clone(), 0), retries);
        let operator = finish_opendal_operator(operator, retries);
        Ok(Arc::new(OpendalStore::new_s3(operator, conditional)) as Arc<dyn OSObjectStore>)
    }
}

#[async_trait::async_trait]
impl ObjectStoreProvider for AwsStoreProvider {
    async fn new_store(
        &self,
        mut base_path: Url,
        params: &ObjectStoreParams,
    ) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_CLOUD_BLOCK_SIZE);
        let mut storage_options =
            StorageOptions::new(params.storage_options().cloned().unwrap_or_default());
        storage_options.with_env_s3();
        let download_retry_count = storage_options.download_retry_count();

        let use_opendal = storage_options
            .0
            .get("use_opendal")
            .map(|v| str_is_truthy(v))
            .unwrap_or(false);

        let explicit_profile = explicit_s3_profile(&storage_options);
        let SelectedS3Profile {
            config,
            credentials,
        } = select_s3_profile(explicit_profile).await;
        let resolved_s3_options = ResolvedS3StorageOptions::new(
            storage_options.as_s3_options(),
            config.as_ref(),
            credentials,
        );

        // Determine S3 Express and constant size upload parts before building the store
        let is_s3_express = check_s3_express(&base_path, &storage_options);

        let use_constant_size_upload_parts =
            resolved_s3_options.requires_constant_size_upload_parts();

        // Keyed like the registry cache so per-dataset stores share the bucket's budget.
        let store_prefix =
            self.calculate_object_store_prefix(&base_path, params.storage_options())?;
        let throttle_defaults = resolved_s3_options.throttle_defaults(is_s3_express, use_opendal);
        let throttle_state = shared_throttle_state(&store_prefix, params, throttle_defaults)?;

        let update_conditions = resolved_s3_options.supports_conditional_update();
        let (inner, paginated_lister) = if use_opendal {
            reject_http_connector_for_opendal(params)?;
            // Use OpenDAL implementation
            // Listed in full: no paginated lister covers OpenDAL yet.
            (
                self.build_opendal_s3_store(
                    &base_path,
                    &storage_options,
                    params.aws_credentials.as_ref(),
                )
                .await?,
                None,
            )
        } else {
            // Use default Amazon S3 implementation
            let (inner, store) = self
                .build_amazon_s3_store(
                    &mut base_path,
                    params,
                    &storage_options,
                    resolved_s3_options,
                    is_s3_express,
                    throttle_state.as_ref(),
                )
                .await?;
            (inner, Some(store as Arc<dyn PaginatedListStore>))
        };
        let (inner, paginated_lister) =
            with_throttling(throttle_state, !use_opendal, inner, paginated_lister);

        Ok(ObjectStore {
            mutation_backend: update_conditions.then(|| {
                let kind = if use_opendal {
                    MutationBackend::OpenDalS3
                } else {
                    MutationBackend::ETag
                };
                (kind, Arc::downgrade(&inner))
            }),
            inner,
            external_blob_inner: None,
            _external_blob_origin: None,
            local_dir_operations: None,
            local_fsync: false,
            scheme: String::from(base_path.scheme()),
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts,
            list_is_lexically_ordered: !is_s3_express,
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count,
            scheduler_error_mode: Default::default(),
            io_tracker: Default::default(),
            store_prefix: self
                .calculate_object_store_prefix(&base_path, params.storage_options())?,
            paginated_lister,
        })
    }
}

/// Storage option naming an AWS shared profile for one store.
const AWS_PROFILE_STORAGE_OPTION: &str = "aws_profile";

fn explicit_s3_profile(storage_options: &StorageOptions) -> Option<&str> {
    storage_options
        .0
        .get(AWS_PROFILE_STORAGE_OPTION)
        .map(String::as_str)
        .filter(|profile| !profile.is_empty())
}

/// Load the AWS SDK config for the store's selected profile.
///
/// An explicit `aws_profile` storage option overrides the process-wide
/// `AWS_PROFILE`. When neither is set, the store keeps using the ambient
/// default credential chain without loading a profile config here.
async fn load_s3_profile_config(profile_name: Option<&str>) -> Option<SdkConfig> {
    if profile_name.is_none() && std::env::var_os("AWS_PROFILE").is_none() {
        return None;
    }

    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    if let Some(profile_name) = profile_name {
        loader = loader.profile_name(profile_name);
    }
    Some(loader.load().await)
}

#[derive(Default)]
struct SelectedS3Profile {
    config: Option<SdkConfig>,
    credentials: Option<SharedCredentialsProvider>,
}

impl SelectedS3Profile {
    fn region(&self) -> Option<String> {
        self.config
            .as_ref()
            .and_then(SdkConfig::region)
            .map(|region| region.as_ref().to_string())
    }
}

async fn select_s3_profile(explicit_profile: Option<&str>) -> SelectedS3Profile {
    let config = load_s3_profile_config(explicit_profile).await;
    let credentials = if explicit_profile.is_some() {
        config.as_ref().and_then(SdkConfig::credentials_provider)
    } else {
        None
    };
    SelectedS3Profile {
        config,
        credentials,
    }
}

/// Check if the storage is S3 Express
fn check_s3_express(url: &Url, storage_options: &StorageOptions) -> bool {
    storage_options
        .0
        .get("s3_express")
        .map(|v| str_is_truthy(v))
        .unwrap_or(false)
        || url.authority().ends_with("--x-s3")
}

fn effective_s3_endpoint(storage_options: &HashMap<AmazonS3ConfigKey, String>) -> Option<&str> {
    storage_options
        .get(&AmazonS3ConfigKey::S3Endpoint)
        .or_else(|| storage_options.get(&AmazonS3ConfigKey::Endpoint))
        .map(String::as_str)
}

/// Figure out the S3 region of the bucket.
///
/// This resolves in order of precedence:
/// 1. The region provided in the storage options
/// 2. The selected AWS profile's region when a custom endpoint is configured
/// 3. (If endpoint is not set), the region returned by the S3 API for the bucket
///
/// It can return None if no region is provided and the endpoint is set.
async fn resolve_s3_region(
    url: &Url,
    resolved_s3_options: &ResolvedS3StorageOptions,
) -> Result<Option<String>> {
    let storage_options = &resolved_s3_options.options;
    if let Some(region) = storage_options.get(&AmazonS3ConfigKey::Region) {
        Ok(Some(region.clone()))
    } else if resolved_s3_options.effective_endpoint().is_none() {
        // If no endpoint is set, we can assume this is AWS S3 and the region
        // can be resolved from the bucket.
        let bucket = url.host_str().ok_or_else(|| {
            Error::invalid_input(format!("Could not parse bucket from url: {}", url))
        })?;

        let mut client_options = ClientOptions::default();
        for (key, value) in storage_options {
            if let AmazonS3ConfigKey::Client(client_key) = key {
                client_options = client_options.with_config(*client_key, value.clone());
            }
        }

        let bucket_region =
            object_store::aws::resolve_bucket_region(bucket, &client_options).await?;
        Ok(Some(bucket_region))
    } else {
        Ok(resolved_s3_options.profile_region.clone())
    }
}

/// Selects which AWS credential provider to use for a dataset.
///
/// When set, overrides automatic credential resolution for everything except an
/// explicitly-supplied `credentials` provider or `storage_options_accessor`.
#[derive(Debug, Clone, PartialEq)]
pub enum AwsProviderScheme {
    /// Require static access-key credentials (`aws_access_key_id` +
    /// `aws_secret_access_key`). Returns an error if they are absent.
    Token,
    /// Use the ECS/Pod Identity container credential endpoint.
    /// The endpoint URI is read from the `AWS_CONTAINER_CREDENTIALS_FULL_URI`
    /// or `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` environment variables.
    Ecs,
    /// Use IRSA (IAM Roles for Service Accounts) web identity token credentials.
    /// The token file and role ARN are read from the `AWS_WEB_IDENTITY_TOKEN_FILE`
    /// and `AWS_ROLE_ARN` environment variables.
    Irsa,
}

/// Build AWS credentials with the store's explicit `aws_profile` option.
///
/// This keeps [`build_aws_credential`]'s precedence. An explicit profile also
/// supplies the signing region when `region` is absent. Ambient `AWS_PROFILE`
/// keeps the existing default-chain behavior.
pub async fn build_aws_credential_with_storage_options(
    credentials_refresh_offset: Duration,
    credentials: Option<AwsCredentialProvider>,
    storage_options: Option<&HashMap<AmazonS3ConfigKey, String>>,
    region: Option<String>,
    storage_options_accessor: Option<Arc<StorageOptionsAccessor>>,
    provider_scheme: Option<AwsProviderScheme>,
    storage_options_raw: &StorageOptions,
) -> Result<(AwsCredentialProvider, String)> {
    let explicit_profile = explicit_s3_profile(storage_options_raw);
    let profile = if explicit_profile.is_some() {
        select_s3_profile(explicit_profile).await
    } else {
        SelectedS3Profile::default()
    };
    let region = if explicit_profile.is_some() {
        region.or_else(|| profile.region())
    } else {
        region
    };
    build_aws_credential_with_profile(
        credentials_refresh_offset,
        credentials,
        storage_options,
        region,
        storage_options_accessor,
        provider_scheme,
        profile.credentials,
    )
    .await
}

/// Build AWS credentials
///
/// This resolves credentials from the following sources in order:
/// 1. An explicit `credentials` provider
/// 2. An explicit `storage_options_accessor` with a provider
/// 3. If `provider_scheme` is set:
///    - [`AwsProviderScheme::Token`]: static access-key credentials (error if absent)
///    - [`AwsProviderScheme::Ecs`]: ECS container credential provider
///    - [`AwsProviderScheme::Irsa`]: web identity token (IRSA) provider
/// 4. Static access-key credentials from `storage_options`, if present
/// 5. The default AWS credential provider chain
///
/// # Storage Options Accessor
///
/// When `storage_options_accessor` is provided and has a dynamic provider,
/// credentials are fetched and cached by the accessor with automatic refresh
/// before expiration.
///
/// `credentials_refresh_offset` is the amount of time before expiry to refresh credentials.
pub async fn build_aws_credential(
    credentials_refresh_offset: Duration,
    credentials: Option<AwsCredentialProvider>,
    storage_options: Option<&HashMap<AmazonS3ConfigKey, String>>,
    region: Option<String>,
    storage_options_accessor: Option<Arc<StorageOptionsAccessor>>,
    provider_scheme: Option<AwsProviderScheme>,
) -> Result<(AwsCredentialProvider, String)> {
    build_aws_credential_with_profile(
        credentials_refresh_offset,
        credentials,
        storage_options,
        region,
        storage_options_accessor,
        provider_scheme,
        None,
    )
    .await
}

/// Like [`build_aws_credential`], but uses `profile_credentials` from the
/// store's selected profile before the ambient default chain.
async fn build_aws_credential_with_profile(
    credentials_refresh_offset: Duration,
    credentials: Option<AwsCredentialProvider>,
    storage_options: Option<&HashMap<AmazonS3ConfigKey, String>>,
    region: Option<String>,
    storage_options_accessor: Option<Arc<StorageOptionsAccessor>>,
    provider_scheme: Option<AwsProviderScheme>,
    profile_credentials: Option<SharedCredentialsProvider>,
) -> Result<(AwsCredentialProvider, String)> {
    use aws_config::meta::region::RegionProviderChain;
    const DEFAULT_REGION: &str = "us-west-2";

    let region = if let Some(region) = region {
        region
    } else {
        RegionProviderChain::default_provider()
            .or_else(DEFAULT_REGION)
            .region()
            .await
            .map(|r| r.as_ref().to_string())
            .unwrap_or(DEFAULT_REGION.to_string())
    };

    // If the user supplied their own credential provider that takes top priority
    if let Some(creds) = credentials {
        return Ok((creds, region));
    }

    // Otherwise, if the user provided a storage_options_accessor, try and use that
    if let Some(dynamic_creds) = build_dynamic_credential_provider::<ObjectStoreAwsCredential>(
        storage_options_accessor.clone(),
    )
    .await?
    {
        return Ok((dynamic_creds, region));
    }

    // If the user provided a storage_options_accessor, then it must not have matched AWS.
    // Log a message and ignore it.
    if storage_options_accessor
        .as_ref()
        .is_some_and(|a| a.has_provider())
    {
        log::debug!(
            "Storage options from provider do not contain explicit AWS credentials, \
             falling back to default AWS credentials chain."
        );
    }

    // If the caller specified an explicit provider scheme, use only that provider.
    if let Some(scheme) = provider_scheme {
        return match scheme {
            AwsProviderScheme::Token => {
                let creds = storage_options
                    .and_then(extract_static_s3_credentials)
                    .ok_or_else(|| {
                        Error::invalid_input(
                            "aws_provider_scheme=token requires aws_access_key_id \
                             and aws_secret_access_key to be set",
                        )
                    })?;
                Ok((Arc::new(creds), region))
            }
            AwsProviderScheme::Ecs => {
                let provider = EcsCredentialsProvider::builder().build();
                Ok((
                    Arc::new(AwsCredentialAdapter::new(
                        Arc::new(provider),
                        credentials_refresh_offset,
                    )),
                    region,
                ))
            }
            AwsProviderScheme::Irsa => {
                let conf = ProviderConfig::default().with_region(Some(Region::new(region.clone())));
                let provider = WebIdentityTokenCredentialsProvider::builder()
                    .configure(&conf)
                    .build();
                Ok((
                    Arc::new(AwsCredentialAdapter::new(
                        Arc::new(provider),
                        credentials_refresh_offset,
                    )),
                    region,
                ))
            }
        };
    }

    if let Some(opts) = storage_options {
        // Check for static credentials (access key & secret)
        if let Some(creds) = extract_static_s3_credentials(opts) {
            return Ok((Arc::new(creds), region));
        }
    }

    if let Some(profile_credentials) = profile_credentials {
        return Ok((
            Arc::new(AwsCredentialAdapter::new(
                Arc::new(profile_credentials),
                credentials_refresh_offset,
            )),
            region,
        ));
    }

    let credentials_provider = DefaultCredentialsChain::builder().build().await;
    Ok((
        Arc::new(AwsCredentialAdapter::new(
            Arc::new(credentials_provider),
            credentials_refresh_offset,
        )),
        region,
    ))
}

fn extract_static_s3_credentials(
    options: &HashMap<AmazonS3ConfigKey, String>,
) -> Option<StaticCredentialProvider<ObjectStoreAwsCredential>> {
    let key_id = options.get(&AmazonS3ConfigKey::AccessKeyId).cloned();
    let secret_key = options.get(&AmazonS3ConfigKey::SecretAccessKey).cloned();
    let token = options.get(&AmazonS3ConfigKey::Token).cloned();
    match (key_id, secret_key, token) {
        (Some(key_id), Some(secret_key), token) => {
            Some(StaticCredentialProvider::new(ObjectStoreAwsCredential {
                key_id,
                secret_key,
                token,
            }))
        }
        _ => None,
    }
}

/// Adapt an AWS SDK cred into object_store credentials
#[derive(Debug)]
pub struct AwsCredentialAdapter {
    pub inner: Arc<dyn ProvideCredentials>,

    // The cached credential, if any.
    credentials: RwLock<Option<Arc<aws_credential_types::Credentials>>>,

    // Process-local identity shared by all operations using this cache.
    cache_id: u64,

    // Serialize refreshes without holding the cache lock during network requests.
    refresh_lock: Mutex<()>,

    // The amount of time before expiry to refresh credentials
    credentials_refresh_offset: Duration,
}

impl AwsCredentialAdapter {
    pub fn new(
        provider: Arc<dyn ProvideCredentials>,
        credentials_refresh_offset: Duration,
    ) -> Self {
        Self {
            inner: provider,
            cache_id: {
                static NEXT_CACHE: AtomicU64 = AtomicU64::new(1);
                NEXT_CACHE.fetch_add(1, Ordering::Relaxed)
            },
            credentials: RwLock::new(None),
            refresh_lock: Mutex::new(()),
            credentials_refresh_offset,
        }
    }
}

/// Convert std::time::SystemTime from AWS SDK to our mockable SystemTime
fn to_system_time(time: std::time::SystemTime) -> SystemTime {
    let duration_since_epoch = time
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time should be after UNIX_EPOCH");
    UNIX_EPOCH + duration_since_epoch
}

#[async_trait::async_trait]
impl CredentialProvider for AwsCredentialAdapter {
    type Credential = ObjectStoreAwsCredential;

    async fn get_credential(&self) -> ObjectStoreResult<Arc<Self::Credential>> {
        let cached = self.credentials.read().await.clone();
        let credentials = match cached {
            None => self.must_refresh().await?,
            Some(credentials) => {
                let (is_expired, is_stale) = self.check_staleness(&credentials);
                if is_expired {
                    self.must_refresh().await?
                } else if is_stale {
                    self.maybe_refresh(credentials).await?
                } else {
                    credentials
                }
            }
        };
        Ok(Arc::new(Self::Credential {
            key_id: credentials.access_key_id().to_string(),
            secret_key: credentials.secret_access_key().to_string(),
            token: credentials.session_token().map(|s| s.to_string()),
        }))
    }
}

impl AwsCredentialAdapter {
    // Return a tuple of (credential expired, credential stale).
    fn check_staleness(&self, credentials: &aws_credential_types::Credentials) -> (bool, bool) {
        match credentials.expiry() {
            None => (false, false),
            Some(expiry) => {
                let remaining = to_system_time(expiry).duration_since(SystemTime::now());
                match remaining {
                    Ok(remaining) => (
                        remaining.is_zero(),
                        remaining <= self.credentials_refresh_offset,
                    ),
                    Err(_) => (true, true),
                }
            }
        }
    }

    /// Wait for an in-flight refresh, or fetch credentials ourselves.
    async fn must_refresh(&self) -> ObjectStoreResult<Arc<aws_credential_types::Credentials>> {
        record_current(|| CopyOperationEvent::CredentialLockWaiting {
            cache: self.cache_id,
        });
        let _lock = self.refresh_lock.lock().await;
        let _owner = credential_lock_acquired(self.cache_id);
        if let Some(credentials) = self.credentials.read().await.clone()
            && !self.check_staleness(&credentials).0
        {
            return Ok(credentials);
        }
        self.do_refresh().await
    }

    /// Let concurrent requests use valid cached credentials during a refresh.
    async fn maybe_refresh(
        &self,
        old_credentials: Arc<aws_credential_types::Credentials>,
    ) -> ObjectStoreResult<Arc<aws_credential_types::Credentials>> {
        let Ok(_lock) = self.refresh_lock.try_lock() else {
            return Ok(old_credentials);
        };
        let _owner = credential_lock_acquired(self.cache_id);
        if let Some(credentials) = self.credentials.read().await.clone()
            && !Arc::ptr_eq(&credentials, &old_credentials)
            && !self.check_staleness(&credentials).0
        {
            return Ok(credentials);
        }
        match self.do_refresh().await {
            Ok(credentials) => Ok(credentials),
            // The cached credentials may have expired while the request was in flight.
            Err(_) if !self.check_staleness(&old_credentials).0 => Ok(old_credentials),
            Err(error) => Err(error),
        }
    }

    async fn do_refresh(&self) -> ObjectStoreResult<Arc<aws_credential_types::Credentials>> {
        record_current(|| CopyOperationEvent::CredentialRefreshStarted {
            cache: self.cache_id,
        });
        let result = self.inner.provide_credentials().await;
        record_current(|| CopyOperationEvent::CredentialRefreshReturned {
            cache: self.cache_id,
            success: result.is_ok(),
        });
        let credentials = Arc::new(
            result.map_err(|e| Error::io(format!("Failed to get AWS credentials: {:?}", e)))?,
        );
        *self.credentials.write().await = Some(credentials.clone());
        Ok(credentials)
    }
}

impl StorageOptions {
    /// Add values from the environment to storage options.
    ///
    /// Explicitly-set options (including aliases, case variants, and empty-string
    /// sentinels) always take precedence over env vars.
    pub fn with_env_s3(&mut self) {
        self.merge_env_options(std::env::vars_os(), |key| {
            AmazonS3ConfigKey::from_str(key)
                .ok()
                .map(|key| key.as_ref().to_string())
        });
    }

    /// Return options recognized by [`AmazonS3ConfigKey`], ignoring unknown keys.
    ///
    /// When aliases conflict, the exact canonical name (such as `aws_region`)
    /// wins, followed by case variants of that name. Remaining ties use the
    /// lexicographically smallest original key. Empty values are preserved.
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use lance_io::object_store::StorageOptions;
    /// use object_store::aws::AmazonS3ConfigKey;
    ///
    /// let options = StorageOptions(HashMap::from([
    ///     ("region".into(), "us-east-1".into()),
    ///     ("aws_region".into(), "eu-west-1".into()),
    /// ]));
    /// assert_eq!(options.as_s3_options()[&AmazonS3ConfigKey::Region], "eu-west-1");
    /// ```
    pub fn as_s3_options(&self) -> HashMap<AmazonS3ConfigKey, String> {
        self.as_cloud_options()
    }

    /// Parse the `aws_provider_scheme` storage option, if set.
    pub fn aws_provider_scheme(&self) -> Result<Option<AwsProviderScheme>> {
        match self.0.get("aws_provider_scheme").map(|s| s.as_str()) {
            None | Some("") => Ok(None),
            Some("token") => Ok(Some(AwsProviderScheme::Token)),
            Some("ecs") => Ok(Some(AwsProviderScheme::Ecs)),
            Some("irsa") => Ok(Some(AwsProviderScheme::Irsa)),
            Some(other) => Err(Error::invalid_input(format!(
                "Invalid aws_provider_scheme '{}'. Valid values are: token, ecs, irsa",
                other
            ))),
        }
    }
}

impl ObjectStoreParams {
    /// Create a new instance of [`ObjectStoreParams`] based on the AWS credentials.
    pub fn with_aws_credentials(
        aws_credentials: Option<AwsCredentialProvider>,
        region: Option<String>,
    ) -> Self {
        let storage_options_accessor = region.map(|region| {
            let opts: HashMap<String, String> =
                [("region".into(), region)].iter().cloned().collect();
            Arc::new(StorageOptionsAccessor::with_static_options(opts))
        });
        Self {
            aws_credentials,
            storage_options_accessor,
            ..Default::default()
        }
    }
}

pub type DynamicStorageOptionsCredentialProvider =
    NamespaceCredentialsProvider<ObjectStoreAwsCredential>;

#[cfg(test)]
mod tests {
    use crate::object_store::ObjectStoreRegistry;
    use crate::object_store::StorageOptionsProvider;
    #[allow(deprecated)]
    use aws_config::profile::profile_file::{ProfileFileKind, ProfileFiles};
    use aws_credential_types::provider::error::CredentialsError;
    use mock_instant::thread_local::MockClock;
    use object_store::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use wiremock::matchers::{any, header, method};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::object_store::test_utils::RecordingConnector;

    #[rstest::rstest]
    #[case::stale(412, "PreconditionFailed", true)]
    #[case::concurrent(409, "ConditionalRequestConflict", true)]
    #[case::locked(409, "ObjectLockedByBucketPolicy", false)]
    #[case::denied(403, "AccessDenied", false)]
    #[case::uncertain_publication(503, "SlowDown", false)]
    #[tokio::test]
    async fn test_read_for_update_on_s3(
        #[values(false, true)] use_opendal: bool,
        #[case] status: u16,
        #[case] code: &str,
        #[case] conflict: bool,
    ) {
        let server = MockServer::start().await;
        let old_etag = "\"old\"";
        let object = ResponseTemplate::new(200)
            .insert_header("etag", old_etag)
            .insert_header("last-modified", "Tue, 15 Nov 1994 12:45:26 GMT")
            .set_body_string("old");
        let mut get = Mock::given(method("GET"));
        if use_opendal {
            Mock::given(method("HEAD"))
                .respond_with(object.clone())
                .expect(1)
                .mount(&server)
                .await;
            get = get.and(header("if-match", old_etag));
        }
        get.respond_with(object).expect(1).mount(&server).await;
        Mock::given(method("PUT"))
            .and(header("if-match", old_etag))
            .respond_with(ResponseTemplate::new(status).set_body_string(format!(
                "<Error><Code>{code}</Code><Message>Write rejected</Message></Error>",
            )))
            .expect(1)
            .mount(&server)
            .await;

        let options = HashMap::from([
            ("endpoint".into(), server.uri()),
            ("region".into(), "us-east-1".into()),
            ("access_key_id".into(), "key".into()),
            ("secret_access_key".into(), "secret".into()),
            ("allow_http".into(), "true".into()),
            ("use_opendal".into(), use_opendal.to_string()),
            ("client_max_retries".into(), "2".into()),
            ("lance_aimd_max_retries".into(), "2".into()),
        ]);
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                options,
            ))),
            ..Default::default()
        };
        let (store, path) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "s3://bucket/tag",
            &params,
        )
        .await
        .unwrap();

        let (bytes, version) = store.read_for_update(&path).await.unwrap().unwrap();
        assert_eq!(bytes.as_ref(), b"old");
        assert_eq!(version.e_tag.as_deref(), Some(old_etag));
        let error = store
            .replace_if_current(&path, "new".into(), version)
            .await
            .unwrap_err();
        assert_eq!(
            matches!(error, object_store::Error::Precondition { .. }),
            conflict,
            "{error}"
        );
        assert!(error.to_string().contains(code), "{error}");
    }

    #[rstest::rstest]
    #[case::default(None, true)]
    #[case::enabled(Some("etag"), true)]
    #[case::disabled(Some("disabled"), false)]
    #[case::unknown(Some("unknown"), false)]
    fn test_conditional_update_configuration(#[case] mode: Option<&str>, #[case] expected: bool) {
        let mut options = HashMap::new();
        if let Some(mode) = mode {
            options.insert(AmazonS3ConfigKey::ConditionalPut, mode.into());
        }
        assert_eq!(
            ResolvedS3StorageOptions::new(options, None, None).supports_conditional_update(),
            expected
        );
    }

    #[rstest::rstest]
    #[case::canonical("aws_region", "region")]
    #[case::canonical_over_case_variant("aws_region", "AWS_REGION")]
    #[case::canonical_case_tie("AWS_REGION", "Aws_Region")]
    #[case::case_variant_over_alias("AWS_REGION", "region")]
    #[case::lexical_alias("REGION", "region")]
    fn test_storage_options_alias_precedence(
        #[case] preferred_key: &str,
        #[case] conflicting_key: &str,
        #[values("preferred", "")] preferred_value: &str,
        #[values(false, true)] is_reversed: bool,
    ) {
        // Exercise fresh hash seeds as well as both insertion orders.
        for _ in 0..16 {
            let mut entries = [
                (preferred_key.to_string(), preferred_value.to_string()),
                (conflicting_key.to_string(), "conflicting".to_string()),
                ("lance_custom_option".to_string(), "ignored".to_string()),
            ];
            if is_reversed {
                entries.reverse();
            }
            let options = StorageOptions(HashMap::from(entries));
            assert_eq!(
                options.as_s3_options(),
                HashMap::from([(AmazonS3ConfigKey::Region, preferred_value.to_string())])
            );
        }
    }

    #[derive(Debug, Default)]
    struct MockAwsCredentialsProvider {
        called: AtomicBool,
    }

    #[async_trait::async_trait]
    impl CredentialProvider for MockAwsCredentialsProvider {
        type Credential = ObjectStoreAwsCredential;

        async fn get_credential(&self) -> ObjectStoreResult<Arc<Self::Credential>> {
            self.called.store(true, Ordering::Relaxed);
            Ok(Arc::new(Self::Credential {
                key_id: "".to_string(),
                secret_key: "".to_string(),
                token: None,
            }))
        }
    }

    #[allow(deprecated)]
    async fn load_test_profile(config: &str) -> SdkConfig {
        let profile_files = ProfileFiles::builder()
            .with_contents(ProfileFileKind::Config, config)
            .build();
        aws_config::defaults(BehaviorVersion::latest())
            .profile_name("selected")
            .profile_files(profile_files)
            .load()
            .await
    }

    #[derive(Debug)]
    struct FailingAwsCredentialsProvider;

    impl ProvideCredentials for FailingAwsCredentialsProvider {
        fn provide_credentials<'a>(
            &'a self,
        ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            aws_credential_types::provider::future::ProvideCredentials::new(async {
                Err(CredentialsError::provider_error(Box::new(
                    std::io::Error::other("Glue credential endpoint unavailable"),
                )))
            })
        }
    }

    #[tokio::test]
    async fn test_aws_credential_failure_is_io_error() {
        let provider = AwsCredentialAdapter::new(
            Arc::new(FailingAwsCredentialsProvider),
            Duration::from_secs(60),
        );

        let error = provider.get_credential().await.unwrap_err();
        let object_store::Error::Generic { source, .. } = &error else {
            panic!("expected a generic object store error, got {error}");
        };
        assert!(matches!(
            source.downcast_ref::<Error>(),
            Some(Error::IO { .. })
        ));

        let message = error.to_string();
        assert!(message.contains("Failed to get AWS credentials"));
        assert!(message.contains("Glue credential endpoint unavailable"));
        assert!(!message.contains("Encountered internal error"));
    }

    #[derive(Debug, Default)]
    struct RefreshingAwsCredentialsProvider {
        calls: AtomicUsize,
        fail: AtomicBool,
        advance_seconds: u64,
    }

    impl ProvideCredentials for RefreshingAwsCredentialsProvider {
        fn provide_credentials<'a>(
            &'a self,
        ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            aws_credential_types::provider::future::ProvideCredentials::new(async {
                let call = self.calls.fetch_add(1, Ordering::Relaxed);
                tokio::task::yield_now().await;
                MockClock::advance_system_time(Duration::from_secs(self.advance_seconds));
                if self.fail.load(Ordering::Relaxed) {
                    return Err(CredentialsError::provider_error(std::io::Error::other(
                        "STS unavailable",
                    )));
                }
                Ok(aws_credential_types::Credentials::new(
                    format!("key-{call}"),
                    "secret",
                    Some("token".to_string()),
                    Some(
                        std::time::UNIX_EPOCH + MockClock::system_time() + Duration::from_secs(120),
                    ),
                    "test",
                ))
            })
        }
    }

    #[rstest::rstest]
    #[case::fresh(Some(120), 0, true, 0)]
    #[case::no_expiry(None, 0, true, 0)]
    #[case::refresh_boundary(Some(60), 0, true, 1)]
    #[case::stale(Some(30), 0, true, 1)]
    #[case::expires_during_refresh(Some(30), 30, false, 1)]
    #[case::expiry_boundary(Some(0), 0, false, 1)]
    #[case::expired(Some(-1), 0, false, 1)]
    #[tokio::test]
    async fn test_aws_credential_refresh_failure(
        #[case] expires_in: Option<i64>,
        #[case] advance_seconds: u64,
        #[case] succeeds: bool,
        #[case] expected_calls: usize,
    ) {
        MockClock::set_system_time(Duration::from_secs(100_000));
        let inner = Arc::new(RefreshingAwsCredentialsProvider {
            fail: AtomicBool::new(true),
            advance_seconds,
            ..Default::default()
        });
        let provider = AwsCredentialAdapter::new(inner.clone(), Duration::from_secs(60));
        *provider.credentials.write().await =
            Some(Arc::new(aws_credential_types::Credentials::new(
                "cached-key",
                "cached-secret",
                Some("cached-token".to_string()),
                expires_in.map(|seconds| {
                    std::time::UNIX_EPOCH + Duration::from_secs((100_000 + seconds) as u64)
                }),
                "test",
            )));

        let result = provider.get_credential().await;
        if succeeds {
            let credentials = result.unwrap();
            assert_eq!(credentials.key_id, "cached-key");
            assert_eq!(credentials.secret_key, "cached-secret");
            assert_eq!(credentials.token.as_deref(), Some("cached-token"));
        } else {
            let error = result.unwrap_err();
            assert!(matches!(error, object_store::Error::Generic { .. }));
            assert!(error.to_string().contains("STS unavailable"));
        }
        assert_eq!(inner.calls.load(Ordering::Relaxed), expected_calls);
    }

    #[tokio::test]
    async fn test_aws_credential_concurrent_refresh() {
        MockClock::set_system_time(Duration::from_secs(100_000));
        let inner = Arc::new(RefreshingAwsCredentialsProvider::default());
        let provider = AwsCredentialAdapter::new(inner.clone(), Duration::from_secs(60));

        use crate::object_store::copy_diagnostics::{CopyDiagnostics, CopyOperation};
        let (owner, owner_events) = CopyDiagnostics::recording();
        let (waiter, waiter_events) = CopyDiagnostics::recording();
        let first = owner.run(CopyOperation::SourceMetadata, provider.get_credential());
        let second = waiter.run(
            CopyOperation::DestinationMetadata,
            provider.get_credential(),
        );
        tokio::pin!(first, second);
        assert!(futures::poll!(&mut first).is_pending());
        assert!(futures::poll!(&mut second).is_pending());
        {
            let owner = owner_events.lock().unwrap();
            let waiter = waiter_events.lock().unwrap();
            let cache = provider.cache_id;
            assert_ne!(owner[0].id, waiter[0].id);
            assert!(
                owner
                    .iter()
                    .any(|e| e.event == (CopyOperationEvent::CredentialRefreshStarted { cache }))
            );
            assert_eq!(
                waiter.last().unwrap().event,
                CopyOperationEvent::CredentialLockWaiting { cache }
            );
        }
        let (first, second) = tokio::join!(first, second);
        assert!(owner_events.lock().unwrap().iter().any(|e| e.event
            == (CopyOperationEvent::CredentialRefreshReturned {
                cache: provider.cache_id,
                success: true
            })));
        assert!(waiter_events.lock().unwrap().iter().any(|e| e.event
            == (CopyOperationEvent::CredentialLockAcquired {
                cache: provider.cache_id
            })));
        assert_eq!(first.unwrap().key_id, "key-0");
        assert_eq!(second.unwrap().key_id, "key-0");
        assert_eq!(inner.calls.load(Ordering::Relaxed), 1);

        MockClock::advance_system_time(Duration::from_secs(60));
        let (refreshed, cached) =
            tokio::join!(provider.get_credential(), provider.get_credential());
        assert_eq!(refreshed.unwrap().key_id, "key-1");
        assert_eq!(cached.unwrap().key_id, "key-0");
        assert_eq!(provider.get_credential().await.unwrap().key_id, "key-1");
        assert_eq!(inner.calls.load(Ordering::Relaxed), 2);

        MockClock::advance_system_time(Duration::from_secs(120));
        let (first, second) = tokio::join!(provider.get_credential(), provider.get_credential());
        assert_eq!(first.unwrap().key_id, "key-2");
        assert_eq!(second.unwrap().key_id, "key-2");
        assert_eq!(inner.calls.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn test_injected_aws_creds_option_is_used() {
        let mock_provider = Arc::new(MockAwsCredentialsProvider::default());
        let registry = Arc::new(ObjectStoreRegistry::default());
        let connector = Arc::new(RecordingConnector::default());

        let params = ObjectStoreParams {
            aws_credentials: Some(mock_provider.clone() as AwsCredentialProvider),
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([("region".to_string(), "us-east-1".to_string())]),
            ))),
            http_connector: Some(connector.clone()),
            ..ObjectStoreParams::default()
        };

        let (store, _) = ObjectStore::from_uri_and_params(registry, "s3://not-a-bucket", &params)
            .await
            .unwrap();

        assert!(!mock_provider.called.load(Ordering::Relaxed));
        assert!(connector.requests.lock().unwrap().is_empty());

        let result =
            object_store::ObjectStoreExt::get(store.inner.as_ref(), &Path::from("file")).await;

        assert!(mock_provider.called.load(Ordering::Relaxed));
        let error = result.unwrap_err();
        assert!(matches!(error, object_store::Error::NotFound { .. }));
        assert!(error.to_string().contains("file"), "{error}");
        assert!(connector.connects.load(Ordering::SeqCst) >= 1);
        let requests = connector.requests.lock().unwrap();
        assert_eq!(requests.len(), 1, "{requests:?}");
        assert!(requests[0].ends_with("/file"), "{requests:?}");
    }

    #[tokio::test]
    async fn test_resolve_s3_region_from_aws_profile() {
        let profile_config = load_test_profile(
            "[profile selected]\n\
             region = us-west-004\n\
             endpoint_url = https://s3.us-west-004.backblazeb2.com\n\
             aws_access_key_id = test-key\n\
             aws_secret_access_key = test-secret",
        )
        .await;
        let url = Url::parse("s3://test-bucket/path").unwrap();

        let resolved_s3_options =
            ResolvedS3StorageOptions::new(HashMap::new(), Some(&profile_config), None);
        let region = resolve_s3_region(&url, &resolved_s3_options).await.unwrap();

        assert_eq!(region.as_deref(), Some("us-west-004"));
        assert_eq!(
            resolved_s3_options
                .options
                .get(&AmazonS3ConfigKey::Endpoint),
            Some(&"https://s3.us-west-004.backblazeb2.com".to_string())
        );

        let explicit_options = HashMap::from([
            (AmazonS3ConfigKey::Region, "explicit-region".to_string()),
            (
                AmazonS3ConfigKey::Endpoint,
                "https://explicit.example.com".to_string(),
            ),
        ]);
        let resolved_s3_options =
            ResolvedS3StorageOptions::new(explicit_options, Some(&profile_config), None);
        let region = resolve_s3_region(&url, &resolved_s3_options).await.unwrap();

        assert_eq!(region.as_deref(), Some("explicit-region"));
        assert_eq!(
            resolved_s3_options
                .options
                .get(&AmazonS3ConfigKey::Endpoint),
            Some(&"https://explicit.example.com".to_string())
        );
    }

    #[tokio::test]
    async fn test_region_only_aws_profile_preserves_bucket_discovery() {
        let profile_config = load_test_profile("[profile selected]\nregion = us-east-1").await;
        let resolved_s3_options =
            ResolvedS3StorageOptions::new(HashMap::new(), Some(&profile_config), None);

        let url = Url::parse("s3:///path").unwrap();
        let error = resolve_s3_region(&url, &resolved_s3_options)
            .await
            .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("Could not parse bucket"));
    }

    #[tokio::test]
    async fn test_r2_aws_profile_requires_constant_size_upload_parts() {
        let profile_config = load_test_profile(
            "[profile selected]\n\
             region = auto\n\
             endpoint_url = https://account.r2.cloudflarestorage.com",
        )
        .await;
        let resolved_s3_options =
            ResolvedS3StorageOptions::new(HashMap::new(), Some(&profile_config), None);

        assert!(resolved_s3_options.requires_constant_size_upload_parts());
    }

    const S3_PROFILE_CHILD_ENV: &str = "LANCE_TEST_S3_PROFILE_CHILD";
    const S3_PROFILE_ENDPOINT_ENV: &str = "LANCE_TEST_S3_PROFILE_ENDPOINT";
    const S3_PROFILE_SCRUBBED_ENV: &[&str] = &[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "AWS_ENDPOINT_URL_DYNAMODB",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_EC2_METADATA_DISABLED",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ];

    #[tokio::test]
    async fn test_s3_profile_selection_through_public_store() {
        if std::env::var_os(S3_PROFILE_CHILD_ENV).is_none() {
            run_s3_profile_parent().await;
        } else {
            run_s3_profile_child().await;
        }
    }

    async fn run_s3_profile_parent() {
        let server = MockServer::start().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config");
        let credentials_path = dir.path().join("credentials");
        std::fs::write(
            &config_path,
            "[profile secondary]\n\
             region = ap-southeast-2\n\
             \n\
             [profile primary]\n\
             region = eu-west-1\n",
        )
        .unwrap();
        std::fs::write(
            &credentials_path,
            "[secondary]\n\
             aws_access_key_id = TESTSECONDARY\n\
             aws_secret_access_key = secondary-secret\n\
             \n\
             [primary]\n\
             aws_access_key_id = TESTPRIMARY\n\
             aws_secret_access_key = primary-secret\n",
        )
        .unwrap();

        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args([
            "--exact",
            std::thread::current().name().unwrap(),
            "--nocapture",
        ]);
        for &key in S3_PROFILE_SCRUBBED_ENV {
            command.env_remove(key);
        }
        let output = command
            .env(S3_PROFILE_CHILD_ENV, "1")
            .env(S3_PROFILE_ENDPOINT_ENV, server.uri())
            .env("AWS_CONFIG_FILE", &config_path)
            .env("AWS_SHARED_CREDENTIALS_FILE", &credentials_path)
            .env("AWS_PROFILE", "secondary")
            .env("AWS_EC2_METADATA_DISABLED", "true")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "child test failed: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "child did not run exactly one test:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );

        let requests = server
            .received_requests()
            .await
            .expect("wiremock request recording is enabled");
        let probes = requests
            .iter()
            .filter(|request| request.url.path().ends_with("/probe"))
            .collect::<Vec<_>>();
        assert_eq!(probes.len(), 5, "{probes:?}");
        let authorizations = probes
            .iter()
            .map(|request| {
                request
                    .headers
                    .get("authorization")
                    .and_then(|value| value.to_str().ok())
            })
            .collect::<Vec<_>>();

        assert_signed_as(authorizations[0], "TESTSECONDARY", "ap-southeast-2");
        assert_signed_as(authorizations[1], "TESTPRIMARY", "eu-west-1");
        assert_signed_as(authorizations[2], "TESTSECONDARY", "ap-southeast-2");
        assert_signed_as(authorizations[3], "TESTSECONDARY", "ap-southeast-2");
        assert_signed_as(authorizations[4], "TESTPRIMARY", "ca-central-1");
    }

    async fn run_s3_profile_child() {
        let endpoint = std::env::var(S3_PROFILE_ENDPOINT_ENV).unwrap();
        let registry = Arc::new(ObjectStoreRegistry::default());
        let uri = "s3://bucket/prefix";

        let secondary =
            open_profile_store(registry.clone(), uri, &endpoint, Some("secondary"), None).await;
        let primary =
            open_profile_store(registry.clone(), uri, &endpoint, Some("primary"), None).await;

        // Both stores exist before the first request, so opening the second one
        // cannot change how the first store signs.
        probe_store(&secondary).await;
        probe_store(&primary).await;
        probe_store(&secondary).await;

        let ambient = open_profile_store(registry.clone(), uri, &endpoint, None, None).await;
        probe_store(&ambient).await;

        let explicit_region = open_profile_store(
            registry,
            uri,
            &endpoint,
            Some("primary"),
            Some("ca-central-1"),
        )
        .await;
        probe_store(&explicit_region).await;
    }

    async fn open_profile_store(
        registry: Arc<ObjectStoreRegistry>,
        uri: &str,
        endpoint: &str,
        profile: Option<&str>,
        region: Option<&str>,
    ) -> Arc<ObjectStore> {
        let mut options = HashMap::from([
            ("aws_endpoint".to_string(), endpoint.to_string()),
            ("aws_allow_http".to_string(), "true".to_string()),
            ("client_max_retries".to_string(), "0".to_string()),
            ("lance_aimd_max_retries".to_string(), "0".to_string()),
        ]);
        if let Some(profile) = profile {
            options.insert("aws_profile".to_string(), profile.to_string());
        }
        if let Some(region) = region {
            options.insert("region".to_string(), region.to_string());
        }
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                options,
            ))),
            ..Default::default()
        };
        ObjectStore::from_uri_and_params(registry, uri, &params)
            .await
            .unwrap()
            .0
    }

    async fn probe_store(store: &ObjectStore) {
        let result =
            object_store::ObjectStoreExt::get(store.inner.as_ref(), &Path::from("probe")).await;
        assert!(
            matches!(result, Err(object_store::Error::NotFound { .. })),
            "{result:?}"
        );
    }

    fn assert_signed_as(authorization: Option<&str>, access_key: &str, region: &str) {
        let authorization = authorization.expect("request should be signed");
        assert!(
            authorization.contains(&format!("Credential={access_key}/")),
            "expected access key {access_key} in {authorization}"
        );
        assert!(
            authorization.contains(&format!("/{region}/s3/aws4_request")),
            "expected region {region} in {authorization}"
        );
    }

    #[test]
    fn test_s3_read_rate_defaults() {
        let native = ResolvedS3StorageOptions::new(HashMap::new(), None, None);
        let defaults = native.throttle_defaults(false, false);
        assert_eq!(defaults.read.max_rate, 11_000.0);
        for category in [defaults.write, defaults.delete, defaults.list] {
            assert_eq!(category.max_rate, AimdConfig::default().max_rate);
        }
        let custom = ResolvedS3StorageOptions::new(
            HashMap::from([(
                AmazonS3ConfigKey::Endpoint,
                "https://s3.example.com".to_string(),
            )]),
            None,
            None,
        );
        for config in [
            native.throttle_defaults(true, false),
            native.throttle_defaults(false, true),
            custom.throttle_defaults(false, false),
        ] {
            assert_eq!(config.read.max_rate, AimdConfig::default().max_rate);
        }
    }

    #[test]
    fn test_s3_path_parsing() {
        let provider = AwsStoreProvider;

        let cases = [
            ("s3://bucket/path/to/file", "path/to/file"),
            // for non ASCII string tests: the URL encodes them, extract_path must decode back
            ("s3://bucket/测试path/to/file", "测试path/to/file"),
            ("s3://bucket/path/&to/file", "path/&to/file"),
            ("s3://bucket/path/=to/file", "path/=to/file"),
            (
                "s3+ddb://bucket/path/to/file?ddbTableName=test",
                "path/to/file",
            ),
        ];

        for (uri, expected_path) in cases {
            let url = Url::parse(uri).unwrap();
            let path = provider.extract_path(&url).unwrap();
            // extract_path decodes url.path(), so the Path stores the raw (decoded)
            // string. Path::parse keeps its input verbatim, matching that, whereas
            // Path::from would percent-encode non-ASCII bytes and not match.
            let expected_path = Path::parse(expected_path).unwrap();
            assert_eq!(path, expected_path)
        }
    }

    #[test]
    fn test_is_s3_express() {
        let cases = [
            (
                "s3://bucket/path/to/file",
                HashMap::from([("s3_express".to_string(), "true".to_string())]),
                true,
            ),
            (
                "s3://bucket/path/to/file",
                HashMap::from([("s3_express".to_string(), "false".to_string())]),
                false,
            ),
            ("s3://bucket/path/to/file", HashMap::from([]), false),
            (
                "s3://bucket--x-s3/path/to/file",
                HashMap::from([("s3_express".to_string(), "true".to_string())]),
                true,
            ),
            (
                "s3://bucket--x-s3/path/to/file",
                HashMap::from([("s3_express".to_string(), "false".to_string())]),
                true, // URL takes precedence
            ),
            ("s3://bucket--x-s3/path/to/file", HashMap::from([]), true),
        ];

        for (uri, storage_map, expected) in cases {
            let url = Url::parse(uri).unwrap();
            let storage_options = StorageOptions(storage_map);
            let is_s3_express = check_s3_express(&url, &storage_options);
            assert_eq!(is_s3_express, expected);
        }
    }

    #[tokio::test]
    async fn test_use_opendal_flag() {
        use crate::object_store::StorageOptionsAccessor;
        let provider = AwsStoreProvider;
        let url = Url::parse("s3://test-bucket/path").unwrap();
        let params_with_flag = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("use_opendal".to_string(), "true".to_string()),
                    ("region".to_string(), "us-west-2".to_string()),
                ]),
            ))),
            ..Default::default()
        };

        let store = provider
            .new_store(url.clone(), &params_with_flag)
            .await
            .unwrap();
        assert_eq!(store.scheme, "s3");
    }

    /// A credential provider handed to the OpenDAL arm would be dropped, and
    /// the requests signed from the environment instead.
    #[tokio::test]
    async fn test_use_opendal_rejects_an_explicit_credential_provider() {
        use crate::object_store::StorageOptionsAccessor;
        let provider = AwsStoreProvider;
        let url = Url::parse("s3://test-bucket/path").unwrap();
        let params = ObjectStoreParams {
            aws_credentials: Some(
                Arc::new(MockAwsCredentialsProvider::default()) as AwsCredentialProvider
            ),
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("use_opendal".to_string(), "true".to_string()),
                    ("region".to_string(), "us-west-2".to_string()),
                ]),
            ))),
            ..Default::default()
        };

        let err = provider.new_store(url, &params).await.unwrap_err();
        assert!(
            matches!(err, Error::NotSupported { .. }),
            "got: {err:?}, expected NotSupported"
        );
        assert!(err.to_string().contains("aws_credentials"), "got: {err}");
    }

    /// OpenDAL opens its own clients, so a connector handed to it would be dropped.
    #[tokio::test]
    async fn test_use_opendal_rejects_an_explicit_http_connector() {
        let params = ObjectStoreParams {
            http_connector: Some(Arc::new(RecordingConnector::default())),
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("use_opendal".to_string(), "true".to_string()),
                    ("region".to_string(), "us-west-2".to_string()),
                ]),
            ))),
            ..Default::default()
        };

        let url = Url::parse("s3://test-bucket/path").unwrap();
        let err = AwsStoreProvider.new_store(url, &params).await.unwrap_err();
        assert!(
            matches!(err, Error::NotSupported { .. }),
            "got: {err:?}, expected NotSupported"
        );
        assert!(err.to_string().contains("http_connector"), "got: {err}");
    }

    /// S3 Express ignores `start-after` and does not list in key order, but it does hand back
    /// continuation tokens, which is all the native store resumes from — so an Express bucket
    /// pages like any other. The OpenDAL arm has no lister to page with at all.
    #[rstest::rstest]
    #[case::native("false", true)]
    #[case::opendal("true", false)]
    #[tokio::test]
    async fn test_s3_express_is_paged_by_continuation_token(
        #[case] use_opendal: &str,
        #[case] paginated: bool,
    ) {
        let provider = AwsStoreProvider;
        // Express bucket names carry their availability zone, which the S3 client validates.
        let url = Url::parse("s3://test-bucket--use1-az4--x-s3/path").unwrap();
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("use_opendal".to_string(), use_opendal.to_string()),
                    ("region".to_string(), "us-east-1".to_string()),
                ]),
            ))),
            ..Default::default()
        };

        let store = provider.new_store(url, &params).await.unwrap();

        assert!(!store.list_is_lexically_ordered);
        assert_eq!(store.paginated_lister.is_some(), paginated);
    }

    #[tokio::test]
    async fn test_s3_store_opens_its_clients_with_the_given_http_connector() {
        let connector = Arc::new(RecordingConnector::default());
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("region".to_string(), "us-east-1".to_string()),
                    ("aws_access_key_id".to_string(), "key".to_string()),
                    ("aws_secret_access_key".to_string(), "secret".to_string()),
                    ("aws_endpoint".to_string(), "http://127.0.0.1:1".to_string()),
                    ("allow_http".to_string(), "true".to_string()),
                ]),
            ))),
            http_connector: Some(connector.clone()),
            ..Default::default()
        };
        assert_ne!(params, ObjectStoreParams::default());

        let url = Url::parse("s3://bucket/path").unwrap();
        let store = AwsStoreProvider.new_store(url, &params).await.unwrap();
        let found =
            object_store::ObjectStoreExt::get(store.inner.as_ref(), &Path::from("path/file")).await;

        assert!(
            matches!(found, Err(object_store::Error::NotFound { .. })),
            "{found:?}"
        );
        assert!(connector.connects.load(Ordering::SeqCst) >= 1);
        let requests = connector.requests.lock().unwrap();
        assert_eq!(requests.len(), 1, "{requests:?}");
        assert!(requests[0].ends_with("/bucket/path/file"), "{requests:?}");
    }

    #[derive(Debug)]
    struct MockStorageOptionsProvider {
        call_count: Arc<RwLock<usize>>,
        expires_in_millis: Option<u64>,
    }

    impl MockStorageOptionsProvider {
        fn new(expires_in_millis: Option<u64>) -> Self {
            Self {
                call_count: Arc::new(RwLock::new(0)),
                expires_in_millis,
            }
        }

        async fn get_call_count(&self) -> usize {
            *self.call_count.read().await
        }
    }

    #[async_trait::async_trait]
    impl StorageOptionsProvider for MockStorageOptionsProvider {
        async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
            let count = {
                let mut c = self.call_count.write().await;
                *c += 1;
                *c
            };

            let mut options = HashMap::from([
                ("aws_access_key_id".to_string(), format!("AKID_{}", count)),
                (
                    "aws_secret_access_key".to_string(),
                    format!("SECRET_{}", count),
                ),
                ("aws_session_token".to_string(), format!("TOKEN_{}", count)),
            ]);

            if let Some(expires_in) = self.expires_in_millis {
                let now_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;
                let expires_at = now_ms + expires_in;
                options.insert("expires_at_millis".to_string(), expires_at.to_string());
            }

            Ok(Some(options))
        }

        fn provider_id(&self) -> String {
            let ptr = Arc::as_ptr(&self.call_count) as usize;
            format!("MockStorageOptionsProvider {{ id: {} }}", ptr)
        }
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_with_initial_cache() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let now_ms = MockClock::system_time().as_millis() as u64;

        // Create a mock provider that returns credentials expiring in 10 minutes
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            600_000, // Expires in 10 minutes
        )));

        // Create initial options with cached credentials that expire in 10 minutes
        let expires_at = now_ms + 600_000; // 10 minutes from now
        let initial_options = HashMap::from([
            ("aws_access_key_id".to_string(), "AKID_CACHED".to_string()),
            (
                "aws_secret_access_key".to_string(),
                "SECRET_CACHED".to_string(),
            ),
            ("aws_session_token".to_string(), "TOKEN_CACHED".to_string()),
            ("expires_at_millis".to_string(), expires_at.to_string()),
            ("refresh_offset_millis".to_string(), "300000".to_string()), // 5 minute refresh offset
        ]);

        let provider = DynamicStorageOptionsCredentialProvider::from_provider_with_initial(
            mock.clone(),
            initial_options,
        );

        // First call should use cached credentials (not expired yet)
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_CACHED");
        assert_eq!(cred.secret_key, "SECRET_CACHED");
        assert_eq!(cred.token, Some("TOKEN_CACHED".to_string()));

        // Should not have called the provider yet
        assert_eq!(mock.get_call_count().await, 0);
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_with_expired_cache() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let now_ms = MockClock::system_time().as_millis() as u64;

        // Create a mock provider that returns credentials expiring in 10 minutes
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            600_000, // Expires in 10 minutes
        )));

        // Create initial options with credentials that expired 1 second ago
        let expired_time = now_ms - 1_000; // 1 second ago
        let initial_options = HashMap::from([
            ("aws_access_key_id".to_string(), "AKID_EXPIRED".to_string()),
            (
                "aws_secret_access_key".to_string(),
                "SECRET_EXPIRED".to_string(),
            ),
            ("expires_at_millis".to_string(), expired_time.to_string()),
            ("refresh_offset_millis".to_string(), "300000".to_string()), // 5 minute refresh offset
        ]);

        let provider = DynamicStorageOptionsCredentialProvider::from_provider_with_initial(
            mock.clone(),
            initial_options,
        );

        // First call should fetch new credentials because cached ones are expired
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(cred.secret_key, "SECRET_1");
        assert_eq!(cred.token, Some("TOKEN_1".to_string()));

        // Should have called the provider once
        assert_eq!(mock.get_call_count().await, 1);
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_refresh_lead_time() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        // Create a mock provider that returns credentials expiring in 30 seconds
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            30_000, // Expires in 30 seconds
        )));

        // Create credential provider with default 60 second refresh offset
        // This means credentials should be refreshed when they have less than 60 seconds left
        let provider = DynamicStorageOptionsCredentialProvider::from_provider(mock.clone());

        // First call should fetch credentials from provider (no initial cache)
        // Credentials expire in 30 seconds, which is less than our 60 second refresh offset,
        // so they should be considered "needs refresh" immediately
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(mock.get_call_count().await, 1);

        // Second call should trigger refresh because credentials expire in 30 seconds
        // but our refresh lead time is 60 seconds (now + 60sec > expires_at)
        // The mock will return new credentials (AKID_2) with the same expiration
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_2");
        assert_eq!(mock.get_call_count().await, 2);
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_no_initial_cache() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        // Create a mock provider that returns credentials expiring in 2 minutes
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            120_000, // Expires in 2 minutes
        )));

        // Create credential provider without initial cache, using default 60 second refresh offset
        let provider = DynamicStorageOptionsCredentialProvider::from_provider(mock.clone());

        // First call should fetch from provider (call count = 1)
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(cred.secret_key, "SECRET_1");
        assert_eq!(cred.token, Some("TOKEN_1".to_string()));
        assert_eq!(mock.get_call_count().await, 1);

        // Second call should use cached credentials (not expired yet, still > 60 seconds remaining)
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(mock.get_call_count().await, 1); // Still 1, didn't fetch again

        // Advance time to 90 seconds - should trigger refresh (within 60 sec refresh offset)
        // At this point, credentials expire in 30 seconds (< 60 sec offset)
        MockClock::set_system_time(Duration::from_secs(100_000 + 90));
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_2");
        assert_eq!(cred.secret_key, "SECRET_2");
        assert_eq!(cred.token, Some("TOKEN_2".to_string()));
        assert_eq!(mock.get_call_count().await, 2);

        // Advance time to 210 seconds total (90 + 120) - should trigger another refresh
        MockClock::set_system_time(Duration::from_secs(100_000 + 210));
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_3");
        assert_eq!(cred.secret_key, "SECRET_3");
        assert_eq!(mock.get_call_count().await, 3);
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_with_initial_options() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let now_ms = MockClock::system_time().as_millis() as u64;

        // Create a mock provider that returns credentials expiring in 10 minutes
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            600_000, // Expires in 10 minutes
        )));

        // Create initial options with expiration in 10 minutes
        let expires_at = now_ms + 600_000; // 10 minutes from now
        let initial_options = HashMap::from([
            ("aws_access_key_id".to_string(), "AKID_INITIAL".to_string()),
            (
                "aws_secret_access_key".to_string(),
                "SECRET_INITIAL".to_string(),
            ),
            ("aws_session_token".to_string(), "TOKEN_INITIAL".to_string()),
            ("expires_at_millis".to_string(), expires_at.to_string()),
            ("refresh_offset_millis".to_string(), "300000".to_string()), // 5 minute refresh offset
        ]);

        // Create credential provider with initial options
        let provider = DynamicStorageOptionsCredentialProvider::from_provider_with_initial(
            mock.clone(),
            initial_options,
        );

        // First call should use the initial credential (not expired yet)
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_INITIAL");
        assert_eq!(cred.secret_key, "SECRET_INITIAL");
        assert_eq!(cred.token, Some("TOKEN_INITIAL".to_string()));

        // Should not have called the provider yet
        assert_eq!(mock.get_call_count().await, 0);

        // Advance time to 6 minutes - this should trigger a refresh
        // (5 minute refresh offset means we refresh 5 minutes before expiration)
        MockClock::set_system_time(Duration::from_secs(100_000 + 360));
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(cred.secret_key, "SECRET_1");
        assert_eq!(cred.token, Some("TOKEN_1".to_string()));

        // Should have called the provider once
        assert_eq!(mock.get_call_count().await, 1);

        // Advance time to 11 minutes total - this should trigger another refresh
        MockClock::set_system_time(Duration::from_secs(100_000 + 660));
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_2");
        assert_eq!(cred.secret_key, "SECRET_2");
        assert_eq!(cred.token, Some("TOKEN_2".to_string()));

        // Should have called the provider twice
        assert_eq!(mock.get_call_count().await, 2);

        // Advance time to 16 minutes total - this should trigger yet another refresh
        MockClock::set_system_time(Duration::from_secs(100_000 + 960));
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_3");
        assert_eq!(cred.secret_key, "SECRET_3");
        assert_eq!(cred.token, Some("TOKEN_3".to_string()));

        // Should have called the provider three times
        assert_eq!(mock.get_call_count().await, 3);
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_concurrent_access() {
        // Create a mock provider with far future expiration
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(9999999999999)));

        let provider = Arc::new(DynamicStorageOptionsCredentialProvider::from_provider(
            mock.clone(),
        ));

        // Spawn 10 concurrent tasks that all try to get credentials at the same time
        let mut handles = vec![];
        for i in 0..10 {
            let provider = provider.clone();
            let handle = tokio::spawn(async move {
                let cred = provider.get_credential().await.unwrap();
                // Verify we got the correct credentials (should all be AKID_1 from first fetch)
                assert_eq!(cred.key_id, "AKID_1");
                assert_eq!(cred.secret_key, "SECRET_1");
                assert_eq!(cred.token, Some("TOKEN_1".to_string()));
                i // Return task number for verification
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Verify all 10 tasks completed successfully
        assert_eq!(results.len(), 10);
        for i in 0..10 {
            assert!(results.contains(&i));
        }

        // The provider should have been called exactly once (first request triggers fetch,
        // subsequent requests use cache)
        let call_count = mock.get_call_count().await;
        assert_eq!(
            call_count, 1,
            "Provider should be called exactly once despite concurrent access"
        );
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_concurrent_refresh() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let now_ms = MockClock::system_time().as_millis() as u64;

        // Create initial options with credentials that expired in the past (1000 seconds ago)
        let expires_at = now_ms - 1_000_000;
        let initial_options = HashMap::from([
            ("aws_access_key_id".to_string(), "AKID_OLD".to_string()),
            (
                "aws_secret_access_key".to_string(),
                "SECRET_OLD".to_string(),
            ),
            ("aws_session_token".to_string(), "TOKEN_OLD".to_string()),
            ("expires_at_millis".to_string(), expires_at.to_string()),
            ("refresh_offset_millis".to_string(), "300000".to_string()), // 5 minute refresh offset
        ]);

        // Mock will return credentials expiring in 1 hour
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            3_600_000, // Expires in 1 hour
        )));

        let provider = Arc::new(
            DynamicStorageOptionsCredentialProvider::from_provider_with_initial(
                mock.clone(),
                initial_options,
            ),
        );

        // Spawn 20 concurrent tasks that all try to get credentials at the same time
        // Since the initial credential is expired, they'll all try to refresh
        let mut handles = vec![];
        for i in 0..20 {
            let provider = provider.clone();
            let handle = tokio::spawn(async move {
                let cred = provider.get_credential().await.unwrap();
                // All should get the new credentials (AKID_1 from first fetch)
                assert_eq!(cred.key_id, "AKID_1");
                assert_eq!(cred.secret_key, "SECRET_1");
                assert_eq!(cred.token, Some("TOKEN_1".to_string()));
                i
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Verify all 20 tasks completed successfully
        assert_eq!(results.len(), 20);

        // The provider should have been called at least once, but possibly more times
        // due to the try_write mechanism and race conditions
        let call_count = mock.get_call_count().await;
        assert!(
            call_count >= 1,
            "Provider should be called at least once, was called {} times",
            call_count
        );

        // It shouldn't be called 20 times though - the lock should prevent most concurrent fetches
        assert!(
            call_count < 10,
            "Provider should not be called too many times due to lock contention, was called {} times",
            call_count
        );
    }

    #[tokio::test]
    async fn test_explicit_aws_credentials_takes_precedence_over_accessor() {
        // Create a mock storage options provider that should NOT be called
        let mock_storage_provider = Arc::new(MockStorageOptionsProvider::new(Some(600_000)));

        // Create an accessor with the mock provider
        let accessor = Arc::new(StorageOptionsAccessor::with_provider(
            mock_storage_provider.clone(),
        ));

        // Create an explicit AWS credentials provider
        let explicit_cred_provider = Arc::new(MockAwsCredentialsProvider::default());

        // Build credentials with both aws_credentials AND accessor
        // The explicit aws_credentials should take precedence
        let (result, _region) = build_aws_credential(
            Duration::from_secs(300),
            Some(explicit_cred_provider.clone() as AwsCredentialProvider),
            None, // no storage_options
            Some("us-west-2".to_string()),
            Some(accessor),
            None,
        )
        .await
        .unwrap();

        // Get credential from the result
        let cred = result.get_credential().await.unwrap();

        // The explicit provider should have been called (it returns empty strings)
        assert!(explicit_cred_provider.called.load(Ordering::Relaxed));

        // The storage options provider should NOT have been called
        assert_eq!(
            mock_storage_provider.get_call_count().await,
            0,
            "Storage options provider should not be called when explicit aws_credentials is provided"
        );

        // Verify we got credentials from the explicit provider (empty strings)
        assert_eq!(cred.key_id, "");
        assert_eq!(cred.secret_key, "");
    }

    #[tokio::test]
    async fn test_accessor_used_when_no_explicit_aws_credentials() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let now_ms = MockClock::system_time().as_millis() as u64;

        // Create a mock storage options provider
        let mock_storage_provider = Arc::new(MockStorageOptionsProvider::new(Some(600_000)));

        // Create initial options
        let expires_at = now_ms + 600_000; // 10 minutes from now
        let initial_options = HashMap::from([
            (
                "aws_access_key_id".to_string(),
                "AKID_FROM_ACCESSOR".to_string(),
            ),
            (
                "aws_secret_access_key".to_string(),
                "SECRET_FROM_ACCESSOR".to_string(),
            ),
            (
                "aws_session_token".to_string(),
                "TOKEN_FROM_ACCESSOR".to_string(),
            ),
            ("expires_at_millis".to_string(), expires_at.to_string()),
            ("refresh_offset_millis".to_string(), "300000".to_string()), // 5 minute refresh offset
        ]);

        // Create an accessor with initial options and provider
        let accessor = Arc::new(StorageOptionsAccessor::with_initial_and_provider(
            initial_options,
            mock_storage_provider.clone(),
        ));

        // Build credentials with accessor but NO explicit aws_credentials
        let (result, _region) = build_aws_credential(
            Duration::from_secs(300),
            None, // no explicit aws_credentials
            None, // no storage_options
            Some("us-west-2".to_string()),
            Some(accessor),
            None,
        )
        .await
        .unwrap();

        // Get credential - should use the initial accessor credentials
        let cred = result.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_FROM_ACCESSOR");
        assert_eq!(cred.secret_key, "SECRET_FROM_ACCESSOR");

        // Storage options provider should NOT have been called yet (using cached initial creds)
        assert_eq!(mock_storage_provider.get_call_count().await, 0);

        // Advance time to trigger refresh (past the 5 minute refresh offset)
        MockClock::set_system_time(Duration::from_secs(100_000 + 360));

        // Get credential again - should now fetch from provider
        let cred = result.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(cred.secret_key, "SECRET_1");

        // Storage options provider should have been called once
        assert_eq!(mock_storage_provider.get_call_count().await, 1);
    }

    // Test that aws_provider_scheme=token selects static credentials.
    #[tokio::test]
    async fn test_provider_scheme_token() {
        let opts = HashMap::from([
            (AmazonS3ConfigKey::AccessKeyId, "AKID".to_string()),
            (AmazonS3ConfigKey::SecretAccessKey, "SECRET".to_string()),
        ]);

        let (provider, _) = build_aws_credential(
            Duration::from_secs(300),
            None,
            Some(&opts),
            Some("us-east-1".to_string()),
            None,
            Some(AwsProviderScheme::Token),
        )
        .await
        .unwrap();

        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID");
        assert_eq!(cred.secret_key, "SECRET");
    }

    // Test that aws_provider_scheme=token errors when no static credentials are present.
    #[tokio::test]
    async fn test_provider_scheme_token_errors_without_credentials() {
        let opts: HashMap<AmazonS3ConfigKey, String> = HashMap::new();

        let result = build_aws_credential(
            Duration::from_secs(300),
            None,
            Some(&opts),
            Some("us-east-1".to_string()),
            None,
            Some(AwsProviderScheme::Token),
        )
        .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("aws_provider_scheme=token"),
            "error should mention aws_provider_scheme=token"
        );
    }

    // Test that aws_provider_scheme=ecs builds a provider without error.
    // The ECS provider itself reads from env vars lazily; construction always succeeds.
    #[tokio::test]
    async fn test_provider_scheme_ecs() {
        let opts: HashMap<AmazonS3ConfigKey, String> = HashMap::new();

        let result = build_aws_credential(
            Duration::from_secs(300),
            None,
            Some(&opts),
            Some("us-east-1".to_string()),
            None,
            Some(AwsProviderScheme::Ecs),
        )
        .await;
        assert!(result.is_ok(), "ECS provider should build without error");
    }

    // Test that aws_provider_scheme=irsa builds a provider and attempts credential
    // retrieval (which fails with a provider error, not a config error like
    // "Missing Region" — confirming the region is wired through to the STS client).
    #[tokio::test]
    async fn test_provider_scheme_irsa() {
        let opts: HashMap<AmazonS3ConfigKey, String> = HashMap::new();

        let (provider, _) = build_aws_credential(
            Duration::from_secs(300),
            None,
            Some(&opts),
            Some("us-east-1".to_string()),
            None,
            Some(AwsProviderScheme::Irsa),
        )
        .await
        .unwrap();

        // Credential retrieval must fail with a provider error (missing env vars or
        // network), NOT a configuration error like "Invalid Configuration: Missing Region".
        let err = provider.get_credential().await.unwrap_err();
        assert!(
            !err.to_string().contains("Missing Region"),
            "should not fail with Missing Region; region was provided. got: {err}"
        );
    }

    // Test that an invalid aws_provider_scheme value produces a clear error.
    #[test]
    fn test_provider_scheme_invalid_value() {
        let opts = StorageOptions::new(HashMap::from([(
            "aws_provider_scheme".to_string(),
            "magic".to_string(),
        )]));
        let result = opts.aws_provider_scheme();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("magic"));
    }

    // Test that no aws_provider_scheme falls through to DefaultCredentialsChain without error.
    #[tokio::test]
    async fn test_no_provider_scheme_uses_default_chain() {
        let opts: HashMap<AmazonS3ConfigKey, String> = HashMap::new();

        let result = build_aws_credential(
            Duration::from_secs(300),
            None,
            Some(&opts),
            Some("us-east-1".to_string()),
            None,
            None,
        )
        .await;
        assert!(result.is_ok());
    }
}
