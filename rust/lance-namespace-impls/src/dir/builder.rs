// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use lance::session::Session;
use lance_core::Result;
use lance_core::utils::parse::str_to_bool;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_namespace::error::NamespaceError;
use object_store::path::Path;
use tokio::sync::OnceCell;

use super::{DirectoryNamespace, OpsMetrics, manifest, manifest_feature_flags};
use crate::context::DynamicContextProvider;
use crate::credentials::{create_credential_vendor_for_location, has_credential_vendor_config};

/// Builder for creating a DirectoryNamespace.
///
/// This builder provides a fluent API for configuring and establishing
/// connections to directory-based Lance namespaces.
///
/// # Examples
///
/// ```no_run
/// # use lance_namespace_impls::DirectoryNamespaceBuilder;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a local directory namespace
/// let namespace = DirectoryNamespaceBuilder::new("/path/to/data")
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # use lance_namespace_impls::DirectoryNamespaceBuilder;
/// # use lance::session::Session;
/// # use std::sync::Arc;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create with custom storage options and session
/// let session = Arc::new(Session::default());
/// let namespace = DirectoryNamespaceBuilder::new("s3://bucket/path")
///     .storage_option("region", "us-west-2")
///     .storage_option("access_key_id", "key")
///     .session(session)
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct DirectoryNamespaceBuilder {
    root: String,
    pub(super) storage_options: Option<HashMap<String, String>>,
    session: Option<Arc<Session>>,
    pub(super) manifest_enabled: bool,
    pub(super) dir_listing_enabled: bool,
    pub(super) inline_optimization_enabled: bool,
    table_version_tracking_enabled: bool,
    /// When true, enables migration mode where the namespace checks the manifest first
    /// before falling back to directory listing for root-level tables. When false (default),
    /// root-level tables use directory listing directly without checking the manifest,
    /// avoiding extra object store calls.
    dir_listing_to_manifest_migration_enabled: bool,
    credential_vendor_properties: HashMap<String, String>,
    context_provider: Option<Arc<dyn DynamicContextProvider>>,
    commit_retries: Option<u32>,
    /// When true, returns input storage options in describe_table/declare_table responses
    /// when no credential vendor is configured. Useful for testing. Default: false.
    vend_input_storage_options: bool,
    /// When set, adds expires_at_millis to vended storage options. The value is calculated
    /// as current_time_millis + this interval. This allows clients to know when to refresh
    /// credentials by calling describe_table again. Only effective when vend_input_storage_options
    /// is true.
    vend_input_storage_options_refresh_interval_millis: Option<u64>,
    /// When true, tracks operation metrics. Default: false.
    ops_metrics_enabled: bool,
}

impl std::fmt::Debug for DirectoryNamespaceBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectoryNamespaceBuilder")
            .field("root", &self.root)
            .field("storage_options", &self.storage_options)
            .field("manifest_enabled", &self.manifest_enabled)
            .field("dir_listing_enabled", &self.dir_listing_enabled)
            .field(
                "inline_optimization_enabled",
                &self.inline_optimization_enabled,
            )
            .field(
                "table_version_tracking_enabled",
                &self.table_version_tracking_enabled,
            )
            .field(
                "dir_listing_to_manifest_migration_enabled",
                &self.dir_listing_to_manifest_migration_enabled,
            )
            .field(
                "context_provider",
                &self.context_provider.as_ref().map(|_| "Some(...)"),
            )
            .field(
                "vend_input_storage_options",
                &self.vend_input_storage_options,
            )
            .field(
                "vend_input_storage_options_refresh_interval_millis",
                &self.vend_input_storage_options_refresh_interval_millis,
            )
            .field("ops_metrics_enabled", &self.ops_metrics_enabled)
            .finish()
    }
}

impl DirectoryNamespaceBuilder {
    /// Create a new DirectoryNamespaceBuilder with the specified root path.
    ///
    /// # Arguments
    ///
    /// * `root` - Root directory path (local path or cloud URI like s3://bucket/path)
    pub fn new(root: impl Into<String>) -> Self {
        Self {
            root: root.into().trim_end_matches('/').to_string(),
            storage_options: None,
            session: None,
            manifest_enabled: true,
            dir_listing_enabled: true, // Default to enabled for backwards compatibility
            inline_optimization_enabled: false,
            table_version_tracking_enabled: false, // Default to disabled
            dir_listing_to_manifest_migration_enabled: false, // Default to disabled
            credential_vendor_properties: HashMap::new(),
            context_provider: None,
            commit_retries: None,
            vend_input_storage_options: false,
            vend_input_storage_options_refresh_interval_millis: None,
            ops_metrics_enabled: false,
        }
    }

    /// Enable or disable manifest-based listing.
    ///
    /// When enabled (default), the namespace uses a `__manifest` table to track tables.
    /// When disabled, relies solely on directory scanning.
    pub fn manifest_enabled(mut self, enabled: bool) -> Self {
        self.manifest_enabled = enabled;
        self
    }

    /// Enable or disable directory-based listing fallback.
    ///
    /// When enabled (default), falls back to directory scanning for tables not in the manifest.
    /// When disabled, only consults the manifest table.
    pub fn dir_listing_enabled(mut self, enabled: bool) -> Self {
        self.dir_listing_enabled = enabled;
        self
    }

    /// Enable or disable migration mode from directory listing to manifest.
    ///
    /// When enabled, root-level table operations check the manifest first before
    /// falling back to directory listing. When disabled (default), root-level tables
    /// use directory listing directly, avoiding extra object store calls.
    /// Only relevant when both `manifest_enabled` and `dir_listing_enabled` are true.
    pub fn dir_listing_to_manifest_migration_enabled(mut self, enabled: bool) -> Self {
        self.dir_listing_to_manifest_migration_enabled = enabled;
        self
    }

    /// Enable or disable replacement index maintenance for the __manifest table.
    ///
    /// When enabled, copy-on-write manifest rewrites build replacement indices for fast
    /// reads. This is disabled by default so rewrites only replace data files.
    pub fn inline_optimization_enabled(mut self, enabled: bool) -> Self {
        self.inline_optimization_enabled = enabled;
        self
    }

    /// Enable or disable table version tracking through the namespace.
    ///
    /// When enabled, `describe_table` returns `managed_versioning: true` to indicate
    /// that commits should go through the namespace's table version APIs rather than
    /// direct object store operations.
    ///
    /// When disabled (default), `managed_versioning` is not set.
    pub fn table_version_tracking_enabled(mut self, enabled: bool) -> Self {
        self.table_version_tracking_enabled = enabled;
        self
    }

    /// Create a DirectoryNamespaceBuilder from properties HashMap.
    ///
    /// This method parses a properties map into builder configuration.
    /// It expects:
    /// - `root`: The root directory path (required)
    /// - `manifest_enabled`: Enable manifest-based table tracking (optional, default: true)
    /// - `dir_listing_enabled`: Enable directory listing for table discovery (optional, default: true)
    /// - `inline_optimization_enabled`: Enable replacement indices on __manifest rewrites (optional, default: false)
    /// - `storage.*`: Storage options (optional, prefix will be stripped)
    ///
    /// Credential vendor properties (prefixed with `credential_vendor.`, prefix is stripped):
    /// - `credential_vendor.enabled`: Set to "true" to enable credential vending (required)
    /// - `credential_vendor.permission`: Permission level: read, write, or admin (default: read)
    ///
    /// AWS-specific properties (for s3:// locations):
    /// - `credential_vendor.aws_role_arn`: AWS IAM role ARN (required for AWS)
    /// - `credential_vendor.aws_external_id`: AWS external ID (optional)
    /// - `credential_vendor.aws_region`: AWS region (optional)
    /// - `credential_vendor.aws_role_session_name`: AWS role session name (optional)
    /// - `credential_vendor.aws_duration_millis`: Credential duration in ms (default: 3600000, range: 15min-12hrs)
    ///
    /// GCP-specific properties (for gs:// locations):
    /// - `credential_vendor.gcp_service_account`: Service account to impersonate (optional)
    /// - `credential_vendor.gcp_workload_identity_provider`: Workload Identity Provider for OIDC token exchange (optional)
    /// - `credential_vendor.gcp_impersonation_service_account`: Service account to impersonate after workload identity exchange (optional)
    ///
    /// Note: GCP uses Application Default Credentials (ADC). To use a service account key file,
    /// set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable before starting.
    /// GCP token duration cannot be configured; it's determined by the STS endpoint (typically 1 hour).
    ///
    /// Azure-specific properties (for az:// locations):
    /// - `credential_vendor.azure_account_name`: Azure storage account name (required for Azure)
    /// - `credential_vendor.azure_tenant_id`: Azure tenant ID (optional)
    /// - `credential_vendor.azure_federated_client_id`: Client ID used for workload identity federation (optional)
    /// - `credential_vendor.azure_duration_millis`: Credential duration in ms (default: 3600000, up to 7 days)
    ///
    /// # Arguments
    ///
    /// * `properties` - Configuration properties
    /// * `session` - Optional Lance session to reuse object store registry
    ///
    /// # Returns
    ///
    /// Returns a `DirectoryNamespaceBuilder` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the `root` property is missing.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use lance_namespace_impls::DirectoryNamespaceBuilder;
    /// # use std::collections::HashMap;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut properties = HashMap::new();
    /// properties.insert("root".to_string(), "/path/to/data".to_string());
    /// properties.insert("manifest_enabled".to_string(), "true".to_string());
    /// properties.insert("dir_listing_enabled".to_string(), "false".to_string());
    /// properties.insert("storage.region".to_string(), "us-west-2".to_string());
    ///
    /// let namespace = DirectoryNamespaceBuilder::from_properties(properties, None)?
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_properties(
        properties: HashMap<String, String>,
        session: Option<Arc<Session>>,
    ) -> Result<Self> {
        // Extract root from properties (required)
        let root = properties.get("root").cloned().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Missing required property 'root' for directory namespace".to_string(),
            })
        })?;

        // Extract storage options (properties prefixed with "storage.")
        let storage_options: HashMap<String, String> = properties
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix("storage.")
                    .map(|key| (key.to_string(), v.clone()))
            })
            .collect();

        let storage_options = if storage_options.is_empty() {
            None
        } else {
            Some(storage_options)
        };

        // Extract manifest_enabled (default: true)
        let manifest_enabled = properties
            .get("manifest_enabled")
            .and_then(|v| str_to_bool(v))
            .unwrap_or(true);

        // Extract dir_listing_enabled (default: true)
        let dir_listing_enabled = properties
            .get("dir_listing_enabled")
            .and_then(|v| str_to_bool(v))
            .unwrap_or(true);

        // Extract inline_optimization_enabled (default: false)
        let inline_optimization_enabled = properties
            .get("inline_optimization_enabled")
            .and_then(|v| str_to_bool(v))
            .unwrap_or(false);

        // Extract table_version_tracking_enabled (default: false)
        let table_version_tracking_enabled = properties
            .get("table_version_tracking_enabled")
            .and_then(|v| str_to_bool(v))
            .unwrap_or(false);

        // Extract dir_listing_to_manifest_migration_enabled (default: false)
        let dir_listing_to_manifest_migration_enabled = properties
            .get("dir_listing_to_manifest_migration_enabled")
            .and_then(|v| str_to_bool(v))
            .unwrap_or(false);

        // Extract credential vendor properties (properties prefixed with "credential_vendor.")
        // The prefix is stripped to get short property names
        // The build() method will check if enabled=true before creating the vendor
        let credential_vendor_properties: HashMap<String, String> = properties
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix("credential_vendor.")
                    .map(|key| (key.to_string(), v.clone()))
            })
            .collect();

        let commit_retries = properties
            .get("commit_retries")
            .and_then(|v| v.parse::<u32>().ok());

        // Extract vend_input_storage_options (default: false)
        let vend_input_storage_options = properties
            .get("vend_input_storage_options")
            .and_then(|v| str_to_bool(v))
            .unwrap_or(false);

        // Extract vend_input_storage_options_refresh_interval_millis (optional)
        let vend_input_storage_options_refresh_interval_millis = properties
            .get("vend_input_storage_options_refresh_interval_millis")
            .and_then(|v| v.parse::<u64>().ok());

        // Extract ops_metrics_enabled (default: false)
        let ops_metrics_enabled = properties
            .get("ops_metrics_enabled")
            .and_then(|v| str_to_bool(v))
            .unwrap_or(false);

        Ok(Self {
            root: root.trim_end_matches('/').to_string(),
            storage_options,
            session,
            manifest_enabled,
            dir_listing_enabled,
            inline_optimization_enabled,
            table_version_tracking_enabled,
            dir_listing_to_manifest_migration_enabled,
            credential_vendor_properties,
            context_provider: None,
            commit_retries,
            vend_input_storage_options,
            vend_input_storage_options_refresh_interval_millis,
            ops_metrics_enabled,
        })
    }

    /// Add a storage option.
    ///
    /// # Arguments
    ///
    /// * `key` - Storage option key (e.g., "region", "access_key_id")
    /// * `value` - Storage option value
    pub fn storage_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.storage_options
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value.into());
        self
    }

    /// Add multiple storage options.
    ///
    /// # Arguments
    ///
    /// * `options` - HashMap of storage options to add
    pub fn storage_options(mut self, options: HashMap<String, String>) -> Self {
        self.storage_options
            .get_or_insert_with(HashMap::new)
            .extend(options);
        self
    }

    /// Set the Lance session to use for this namespace.
    ///
    /// When a session is provided, the namespace will reuse the session's
    /// object store registry, allowing multiple namespaces and datasets
    /// to share the same underlying storage connections.
    ///
    /// # Arguments
    ///
    /// * `session` - Arc-wrapped Lance session
    pub fn session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Set the number of retries for commit operations on the manifest table.
    /// If not set, defaults to [`lance_table::io::commit::CommitConfig`] default (20).
    pub fn commit_retries(mut self, retries: u32) -> Self {
        self.commit_retries = Some(retries);
        self
    }

    /// Add a credential vendor property.
    ///
    /// Use short property names without the `credential_vendor.` prefix.
    /// Common properties: `enabled`, `permission`.
    /// AWS properties: `aws_role_arn`, `aws_external_id`, `aws_region`, `aws_role_session_name`, `aws_duration_millis`.
    /// GCP properties: `gcp_service_account`, `gcp_workload_identity_provider`, `gcp_impersonation_service_account`.
    /// Azure properties: `azure_account_name`, `azure_tenant_id`, `azure_federated_client_id`, `azure_duration_millis`.
    ///
    /// # Arguments
    ///
    /// * `key` - Property key (e.g., "enabled", "aws_role_arn")
    /// * `value` - Property value
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lance_namespace_impls::DirectoryNamespaceBuilder;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let namespace = DirectoryNamespaceBuilder::new("s3://my-bucket/data")
    ///     .credential_vendor_property("enabled", "true")
    ///     .credential_vendor_property("aws_role_arn", "arn:aws:iam::123456789012:role/MyRole")
    ///     .credential_vendor_property("permission", "read")
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn credential_vendor_property(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.credential_vendor_properties
            .insert(key.into(), value.into());
        self
    }

    /// Add multiple credential vendor properties.
    ///
    /// Use short property names without the `credential_vendor.` prefix.
    ///
    /// # Arguments
    ///
    /// * `properties` - HashMap of credential vendor properties to add
    pub fn credential_vendor_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.credential_vendor_properties.extend(properties);
        self
    }

    /// Set a dynamic context provider for per-request context.
    ///
    /// The provider can be used to generate additional context for operations.
    /// For DirectoryNamespace, the context is stored but not directly used
    /// in operations (unlike RestNamespace where it's converted to HTTP headers).
    ///
    /// # Arguments
    ///
    /// * `provider` - The context provider implementation
    pub fn context_provider(mut self, provider: Arc<dyn DynamicContextProvider>) -> Self {
        self.context_provider = Some(provider);
        self
    }

    /// Enable or disable returning input storage options in responses.
    ///
    /// When enabled, `describe_table` and `declare_table` will return the storage
    /// options passed to the builder when no credential vendor is configured.
    /// This is useful for testing scenarios where you want to pass storage options
    /// through to clients.
    ///
    /// Default is false (storage options are not returned unless credential vending is configured).
    pub fn vend_input_storage_options(mut self, enabled: bool) -> Self {
        self.vend_input_storage_options = enabled;
        self
    }

    /// Set the refresh interval for vended input storage options.
    ///
    /// When set, vended storage options will include an `expires_at_millis` field
    /// calculated as `current_time_millis + interval_millis`. This allows clients
    /// to know when to refresh credentials by calling `describe_table` again.
    ///
    /// This only has effect when `vend_input_storage_options` is enabled.
    ///
    /// # Arguments
    ///
    /// * `interval_millis` - The refresh interval in milliseconds
    pub fn vend_input_storage_options_refresh_interval_millis(
        mut self,
        interval_millis: u64,
    ) -> Self {
        self.vend_input_storage_options_refresh_interval_millis = Some(interval_millis);
        self
    }

    /// Enable or disable operation metrics tracking.
    ///
    /// When enabled, the namespace will track how many times each API operation
    /// is called. Use `retrieve_ops_metrics()` on the built namespace to get
    /// the current counts.
    ///
    /// Default is false.
    pub fn ops_metrics_enabled(mut self, enabled: bool) -> Self {
        self.ops_metrics_enabled = enabled;
        self
    }

    /// Build the DirectoryNamespace.
    ///
    /// # Returns
    ///
    /// Returns a `DirectoryNamespace` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The root path is invalid
    /// - Connection to the storage backend fails
    /// - Storage options are invalid
    pub async fn build(self) -> Result<DirectoryNamespace> {
        let (object_store, base_path) =
            Self::initialize_object_store(&self.root, &self.storage_options, &self.session).await?;

        let manifest_ns = if self.manifest_enabled {
            match manifest::ManifestNamespace::open_from_directory(
                self.root.clone(),
                self.storage_options.clone(),
                self.session.clone(),
                object_store.clone(),
                base_path.clone(),
                self.dir_listing_enabled,
                self.inline_optimization_enabled,
                self.commit_retries,
            )
            .await
            {
                Ok(ns) => Some(Arc::new(ns)),
                Err(e) if manifest_feature_flags::is_incompatible_manifest_error(&e) => {
                    // The manifest exists but was written with a feature flag this
                    // build does not understand. Refuse rather than silently
                    // degrading to a directory-listing view that ignores it.
                    return Err(e);
                }
                Err(e) if manifest::ManifestNamespace::is_not_found_load_error(&e) => {
                    log::debug!("Manifest namespace does not exist yet: {}", e);
                    None
                }
                Err(e) => return Err(e),
            }
        } else {
            None
        };
        let manifest_cell = OnceCell::new();
        if let Some(manifest_ns) = manifest_ns {
            let _ = manifest_cell.set(manifest_ns);
        }

        // Create credential vendor once during initialization if enabled
        let credential_vendor = if has_credential_vendor_config(&self.credential_vendor_properties)
        {
            create_credential_vendor_for_location(&self.root, &self.credential_vendor_properties)
                .await?
                .map(Arc::from)
        } else {
            None
        };

        let ops_metrics = if self.ops_metrics_enabled {
            Some(Arc::new(OpsMetrics::default()))
        } else {
            None
        };

        Ok(DirectoryNamespace {
            root: self.root,
            storage_options: self.storage_options,
            session: self.session,
            object_store,
            base_path,
            manifest_ns: manifest_cell,
            write_manifest_ns: OnceCell::new(),
            manifest_enabled: self.manifest_enabled,
            dir_listing_enabled: self.dir_listing_enabled,
            inline_optimization_enabled: self.inline_optimization_enabled,
            commit_retries: self.commit_retries,
            dir_listing_to_manifest_migration_enabled: self
                .dir_listing_to_manifest_migration_enabled,
            table_version_tracking_enabled: self.table_version_tracking_enabled,
            credential_vendor,
            context_provider: self.context_provider,
            vend_input_storage_options: self.vend_input_storage_options,
            vend_input_storage_options_refresh_interval_millis: self
                .vend_input_storage_options_refresh_interval_millis,
            ops_metrics,
        })
    }

    /// Initialize the Lance ObjectStore based on the configuration
    async fn initialize_object_store(
        root: &str,
        storage_options: &Option<HashMap<String, String>>,
        session: &Option<Arc<Session>>,
    ) -> Result<(Arc<ObjectStore>, Path)> {
        // Build ObjectStoreParams from storage options
        let accessor = storage_options.clone().map(|opts| {
            Arc::new(lance_io::object_store::StorageOptionsAccessor::with_static_options(opts))
        });
        let params = ObjectStoreParams {
            storage_options_accessor: accessor,
            ..Default::default()
        };

        // Use object store registry from session if provided, otherwise create a new one
        let registry = if let Some(session) = session {
            session.store_registry()
        } else {
            Arc::new(ObjectStoreRegistry::default())
        };

        // Use Lance's object store factory to create from URI
        let (object_store, base_path) = ObjectStore::from_uri_and_params(registry, root, &params)
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to create object store: {:?}", e),
                })
            })?;

        Ok((object_store, base_path))
    }
}
